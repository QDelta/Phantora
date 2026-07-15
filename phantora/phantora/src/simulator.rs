use crate::args;
use crate::cuda_estimate::CudaEstimator;
use crate::event_queue::{Action, EventId, EventQueue, QueueStep};
use crate::nccl_ops::{NcclOps, SimpleRing, Trace};
use crate::perf_db::PerfDb;
use crate::torch_call::TorchCall;
use crate::torch_estimate::TorchEstimator;
use cuda_call::{
    capi, CudaCall, CudaCallMsg, CudaEvent, CudaMemcpyKind, CudaStream, HostId, NcclComm,
    NcclDatatype, ResponseId, SplitResponse, SyncResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::mem;
use std::os::unix::net::UnixDatagram;
use std::time::Duration;

fn mask_process(pid: u32, cores: &[usize]) {
    unsafe {
        let mut cpu_set = mem::zeroed();
        for &core in cores {
            libc::CPU_SET(core, &mut cpu_set);
        }
        let ret = libc::sched_setaffinity(pid as _, mem::size_of::<libc::cpu_set_t>(), &cpu_set);
        if ret != 0 {
            log::error!("mask_process({}, {:?}) = {}", pid, cores, ret);
        }
    }
}

fn mask_new_host(host: &HostId) {
    if let Some(ref cores) = args::get_args().available_cores {
        mask_process(host.pid, cores);
    }
}

fn send_response_to(host: &ResponseId, resp: Vec<u8>) {
    let send_socket = UnixDatagram::unbound().unwrap();
    let node_socket_path = capi::node_socket_path(host.host.pid, host.tid);
    match send_socket.connect(&node_socket_path) {
        Ok(_) => {
            send_socket.send(&resp).unwrap();
        }
        Err(e) => {
            log::warn!(
                "Failed to connect to {:?} because {:?}",
                node_socket_path,
                e
            );
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncPoint {
    Static(i64),
    EventEnd(EventId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostTime {
    pub sync: SyncPoint,
    pub curr: i64,
}

fn update_host_sync(
    host_times: &mut HashMap<HostId, HostTime>,
    host: HostId,
    curr: i64,
    event: Option<EventId>,
) {
    let sync = match event {
        None => SyncPoint::Static(curr),
        Some(event) => SyncPoint::EventEnd(event),
    };
    host_times
        .entry(host)
        .and_modify(|t| {
            t.sync = sync.clone();
            t.curr = t.curr.max(curr)
        })
        .or_insert(HostTime { sync, curr });
}

fn update_host_curr(host_times: &mut HashMap<HostId, HostTime>, host: HostId, curr: i64) {
    host_times
        .entry(host)
        .and_modify(|t| t.curr = t.curr.max(curr))
        .or_insert(HostTime {
            sync: SyncPoint::Static(curr),
            curr,
        });
}

fn send_sync_response_to(
    host_times: &mut HashMap<HostId, HostTime>,
    host: ResponseId,
    end_time: i64,
    event: Option<EventId>,
) {
    let resp = bincode::serialize(&SyncResponse { end_time }).unwrap();
    send_response_to(&host, resp);
    update_host_sync(host_times, host.host, end_time, event);
}

/// Reply to a non-blocking poll. The wire payload is `Option<i64>`:
/// `Some(completion_time)` when the event has completed (recorded as a sync
/// point), or `None` when not ready (host time left untouched -- it charges its
/// own poll cost -- and we just record its current position). `curr_time` is
/// only used for that not-ready bookkeeping.
fn send_query_response_to(
    host_times: &mut HashMap<HostId, HostTime>,
    host: ResponseId,
    ready: Option<(i64, Option<EventId>)>,
    curr_time: i64,
) {
    let resp = bincode::serialize(&ready.map(|(end_time, _)| end_time)).unwrap();
    send_response_to(&host, resp);
    match ready {
        Some((end_time, event)) => update_host_sync(host_times, host.host, end_time, event),
        None => update_host_curr(host_times, host.host, curr_time),
    }
}

#[derive(strum::Display, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeMeta {
    Torch(TorchCall),
    Cuda(CudaCall),
}

pub type CommMeta = CudaCall;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct NodeId(pub HostId);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NcclId(pub [u8; 128]);

#[derive(Default)]
struct StreamInfo {
    events: Vec<EventId>,
    buffered: Vec<(TorchCall, Duration)>,
}

struct InitWaiting {
    joined_ranks: HashMap<i32, (ResponseId, i32)>,
    waiting_for: HashSet<i32>,
}

struct SplitInfo {
    host: ResponseId,
    device: i32,
    color: i32,
    key: i32,
}

struct SplitWaiting {
    joined_ranks: HashMap<i32, SplitInfo>,
    waiting_for: HashSet<i32>,
}

type P2PKey = (NcclId, i32, i32);

#[derive(Clone, Copy)]
enum P2PCallKind {
    Send,
    Recv,
}

struct P2PCallInfo {
    kind: P2PCallKind,
    key: P2PKey,
    stream: CudaStream,
    count: usize,
    dtype: NcclDatatype,
}

struct P2PFlowEndpoint {
    host: HostId,
    stream: CudaStream,
    curr_time: i64,
    count: usize,
    dtype: NcclDatatype,
    // Nodes that gate this one matched network flow.
    //
    // Ungrouped P2P: these are real stream-ordered start/end points created with
    // `add_event`, so the CUDA stream is occupied until the matched flow ends.
    //
    // Grouped P2P: these are raw off-stream flow_start/flow_end points created
    // with `add_action_or_point`.  The stream-visible group_start/group_end
    // shell is separate, which lets multiple internal flows run concurrently
    // instead of being serialized by CUDA stream order.
    flow_start: EventId,
    flow_end: EventId,
    // Keeps this local flow endpoint from starting before it has found a peer.
    // Both grouped and ungrouped P2P create one unmatched barrier per endpoint;
    // matching releases the two endpoint barriers and then the scheduled flow is
    // gated by both sides' flow_start points.
    unmatched_barrier: EventId,
}

struct CCWaiting {
    trace: Trace,
    comm_start_meta: EventId,
    comm_end_meta: EventId,
    comm_barrier: EventId,
    joined_ranks: HashSet<i32>,
    waiting_for: HashSet<i32>,
}

pub struct Simulator {
    cuda_estimator: CudaEstimator,
    torch_estimator: TorchEstimator,

    queue: EventQueue,
    stream_info: HashMap<(HostId, CudaStream), StreamInfo>,
    id_of_cuda: HashMap<(HostId, CudaEvent), EventId>,
    syncing: HashMap<EventId, ResponseId>,
    host_times: HashMap<HostId, HostTime>,
    exited_hosts: HashMap<HostId, i64>,

    nccl_unique_id: [u8; 128],
    // TODO(cjr): update to Box<dyn NcclOps> when adding more algorithms
    nccl_ops: SimpleRing,
    comm_groups: HashMap<NcclId, Vec<(HostId, i32)>>,
    init_waiting: HashMap<NcclId, InitWaiting>,
    split_waiting: HashMap<NcclId, SplitWaiting>,
    bcast_waiting: HashMap<NcclId, VecDeque<CCWaiting>>,
    allreduce_waiting: HashMap<NcclId, VecDeque<CCWaiting>>,
    allgather_waiting: HashMap<NcclId, VecDeque<CCWaiting>>,
    reduce_scatter_waiting: HashMap<NcclId, VecDeque<CCWaiting>>,
    // P2P key = (comm_id, sender_rank, receiver_rank). FIFO queues preserve
    // NCCL's peer ordering for repeated sends/receives.
    send_waiting: HashMap<P2PKey, VecDeque<P2PFlowEndpoint>>,
    recv_waiting: HashMap<P2PKey, VecDeque<P2PFlowEndpoint>>,
}

fn incr_nccl_id(id: &mut [u8; 128]) -> [u8; 128] {
    let prev_id = id.clone();
    let mut i = 0;
    loop {
        if id[i] != 255 {
            id[i] += 1;
            break;
        } else if i != 127 {
            id[i] = 0;
            i += 1;
        } else {
            id.fill(0);
            break;
        }
    }
    prev_id
}

enum TorchCallSeq {
    Single(TorchCall, Duration),
    Seq(Vec<TorchCall>),
}

impl TorchCallSeq {
    fn new(mut calls: Vec<(TorchCall, Duration)>) -> Self {
        if calls.len() == 1 {
            let (call, dur) = calls.pop().unwrap();
            TorchCallSeq::Single(call, dur)
        } else {
            TorchCallSeq::Seq(calls.into_iter().map(|(c, _)| c).collect())
        }
    }
}

impl Simulator {
    /// Build the timing estimators according to the perf-db mode:
    /// - `--perf-db <dir>`: replay — load the DB, no GPU init, miss = hard error.
    /// - `--record-perf-db <dir>`: GPU estimators, preloaded from an existing DB
    ///   (if any) so the run merges into it; dumped on exit (see `handle_exit`).
    /// - neither: GPU estimators, profile-on-miss (original behavior).
    fn build_estimators() -> (CudaEstimator, TorchEstimator) {
        let args = args::get_args();
        if let Some(dir) = &args.perf_db {
            let db = PerfDb::load(dir)
                .unwrap_or_else(|e| panic!("perf-db: failed to load {}: {e}", dir.display()));
            log::info!(
                "perf-db replay: loaded {} compute, {} sequence, {} memcpy, {} flash_attn entries \
                 (recorded on '{}') -- GPU not required",
                db.compute.len(),
                db.sequence.len(),
                db.memcpy.len(),
                db.flash_attn.len(),
                db.gpu_name,
            );
            return (
                CudaEstimator::new_replay(db.memcpy, db.flash_attn),
                TorchEstimator::new_replay(db.compute, db.sequence),
            );
        }

        let mut cuda = CudaEstimator::new();
        let mut torch = TorchEstimator::new();
        if let Some(dir) = &args.record_perf_db {
            if dir.is_dir() {
                match PerfDb::load(dir) {
                    Ok(db) => {
                        // Preloading merges this run's new shapes into the existing
                        // DB -- but only the timings of *this* GPU may be added to
                        // it. Recording into a DB captured on another GPU would
                        // leave the old entries untouched (they hit the preloaded
                        // cache and are never re-profiled) while save() restamps
                        // the manifest with this GPU's name, silently producing a
                        // DB that blends two GPUs under one label. Refuse instead.
                        let local = cuda.gpu_name();
                        if !db.gpu_name.is_empty() && !local.is_empty() && db.gpu_name != local {
                            panic!(
                                "perf-db record: {} was recorded on '{}' but this machine is \
                                 '{}'. Recording would mix timings from two GPUs under one \
                                 name. Record into a different directory, or delete {} to \
                                 re-record it from scratch on this GPU.",
                                dir.display(),
                                db.gpu_name,
                                local,
                                dir.display(),
                            );
                        }
                        log::info!(
                            "perf-db record: merging into existing DB at {} ({} compute entries, \
                             recorded on '{}'). Entries already present are NOT re-profiled; \
                             delete the directory to re-record them.",
                            dir.display(),
                            db.compute.len(),
                            db.gpu_name,
                        );
                        cuda.preload(db.memcpy, db.flash_attn);
                        torch.preload(db.compute, db.sequence);
                    }
                    Err(e) => log::warn!(
                        "perf-db record: could not load existing DB at {} ({e}); starting fresh",
                        dir.display(),
                    ),
                }
            }
        }
        (cuda, torch)
    }

    pub fn new(netsim: netsim::simulator::Simulator) -> Self {
        let (cuda_estimator, torch_estimator) = Self::build_estimators();
        Simulator {
            cuda_estimator,
            torch_estimator,

            queue: EventQueue::new(netsim),
            stream_info: HashMap::new(),
            id_of_cuda: HashMap::new(),
            syncing: HashMap::new(),
            host_times: HashMap::new(),
            exited_hosts: HashMap::new(),

            nccl_unique_id: [0u8; 128],
            nccl_ops: SimpleRing::default(),
            comm_groups: HashMap::new(),
            init_waiting: HashMap::new(),
            split_waiting: HashMap::new(),
            bcast_waiting: HashMap::new(),
            allreduce_waiting: HashMap::new(),
            allgather_waiting: HashMap::new(),
            reduce_scatter_waiting: HashMap::new(),
            send_waiting: HashMap::new(),
            recv_waiting: HashMap::new(),
        }
    }

    fn split_sequences(calls: Vec<(TorchCall, Duration)>) -> Vec<TorchCallSeq> {
        if calls.is_empty() {
            vec![]
        } else {
            let args = args::get_args();
            // perf-db record/replay forces single-op timing. Sequence grouping
            // below is timing-dependent (it splits on the simulated clock), so the
            // groups -- and thus their hashes -- are not reproducible between a
            // record run and a replay run, causing spurious misses. compute_cache
            // is keyed purely by op+shape and is deterministic, so the DB is
            // complete and replay never misses.
            if args.disable_sequence_call || args.perf_db.is_some() || args.record_perf_db.is_some() {
                calls
                    .into_iter()
                    .map(|(call, dur)| TorchCallSeq::Single(call, dur))
                    .collect()
            } else {
                let mut seqs = vec![];
                let mut last_seq = vec![];

                let mut curr_end_time = calls[0].0.time;
                for (call, dur) in calls {
                    if call.time <= curr_end_time {
                        last_seq.push((call, dur));
                        curr_end_time += dur.as_micros() as i64;
                    } else {
                        curr_end_time = call.time + dur.as_micros() as i64;
                        seqs.push(TorchCallSeq::new(last_seq));
                        last_seq = vec![(call, dur)];
                    }
                }

                seqs.push(TorchCallSeq::new(last_seq));
                seqs
            }
        }
    }

    /// Stream add event
    fn add_event(
        torch_estimator: &mut TorchEstimator,
        stream_info: &mut HashMap<(HostId, CudaStream), StreamInfo>,
        queue: &mut EventQueue,
        host_stream: (HostId, CudaStream),
        mut depends_on: Vec<Option<EventId>>,
        action: Option<Action>,
        current_time: i64,
    ) -> EventId {
        // log::debug!("Simulator::add_event: {:?}", host_stream);

        let sinfo = stream_info
            .entry(host_stream)
            .or_insert_with(Default::default);

        Self::clear_compute_buffer_on(torch_estimator, sinfo, queue);
        if let Some(last_event) = sinfo.events.last() {
            depends_on.push(Some(*last_event));
        };

        let this_event = queue.add_action_or_point(depends_on, action, current_time);
        sinfo.events.push(this_event);
        this_event
    }

    fn clear_compute_buffer_on(
        torch_estimator: &mut TorchEstimator,
        sinfo: &mut StreamInfo,
        queue: &mut EventQueue,
    ) {
        let calls = mem::replace(&mut sinfo.buffered, vec![]);
        // log::debug!("Split: {:?}", calls);
        let call_seqs = Self::split_sequences(calls);
        // log::debug!(
        //     "Split results: {:?}",
        //     call_seqs
        //         .iter()
        //         .map(|x| match x {
        //             TorchCallSeq::Single(..) => 1,
        //             TorchCallSeq::Seq(seq) => seq.len(),
        //         })
        //         .collect::<Vec<_>>()
        // );
        for call_seq in call_seqs {
            let (call_seq, call_durs) = match call_seq {
                TorchCallSeq::Single(call, dur) => (vec![call], vec![dur]),
                TorchCallSeq::Seq(call_seq) => {
                    let call_durs = torch_estimator.estimate_sequence(&call_seq);
                    (call_seq, call_durs)
                }
            };
            for (i, call) in call_seq.into_iter().enumerate() {
                let estimate_dur = call_durs[i].as_micros() as i64;
                log::debug!("GPU estimate: {} {:?}", estimate_dur, call);
                let call_for_meta = call.clone();
                let depends_on = match sinfo.events.last() {
                    None => vec![],
                    Some(last_event) => vec![Some(*last_event)],
                };
                let this_event = queue.add_action_or_point(
                    depends_on,
                    Some(Action::Computation(
                        NodeId(call.id.host),
                        estimate_dur,
                        ComputeMeta::Torch(call_for_meta),
                    )),
                    call.time,
                );
                sinfo.events.push(this_event);
            }
        }
    }

    fn execute_to_event(&mut self, until: EventId) -> bool {
        loop {
            match self.queue.execute() {
                QueueStep::EmptyQueue => return false,
                QueueStep::EventStarted(..) => (),
                QueueStep::EventEnded(id, time) | QueueStep::ReachedPoint(id, time) => {
                    if let Some(host) = self.syncing.remove(&id) {
                        send_sync_response_to(&mut self.host_times, host, time, Some(id));
                    }
                    if id == until {
                        return true;
                    }
                }
            }
        }
    }

    /// Run the event queue to exhaustion and deliver any parked sync replies
    /// whose events have completed. No-op when no reply is parked.
    pub fn pump_parked_syncs(&mut self) {
        if self.syncing.is_empty() {
            return;
        }
        loop {
            match self.queue.execute() {
                QueueStep::EmptyQueue => return,
                QueueStep::EventStarted(..) => (),
                QueueStep::EventEnded(id, time) | QueueStep::ReachedPoint(id, time) => {
                    if let Some(host) = self.syncing.remove(&id) {
                        send_sync_response_to(&mut self.host_times, host, time, Some(id));
                    }
                }
            }
        }
    }

    fn execute_to_time(
        queue: &mut EventQueue,
        syncing: &mut HashMap<EventId, ResponseId>,
        host_times: &mut HashMap<HostId, HostTime>,
        until: i64,
    ) -> bool {
        loop {
            match queue.peek_next_time() {
                None => return false,
                Some(next_time) => {
                    if next_time >= until {
                        return true;
                    }
                    match queue.execute() {
                        QueueStep::EmptyQueue => return false,
                        QueueStep::EventStarted(..) => (),
                        QueueStep::EventEnded(id, time) | QueueStep::ReachedPoint(id, time) => {
                            if let Some(host) = syncing.remove(&id) {
                                send_sync_response_to(host_times, host, time, Some(id))
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn handle_cuda_call(&mut self, msg: CudaCallMsg) {
        let host = msg.id;
        let curr_time = msg.curr_time;

        mask_new_host(&host.host);
        update_host_curr(&mut self.host_times, host.host.clone(), curr_time);

        let call = msg.call.clone();

        macro_rules! handle_nccl_op {
            ($op:ident, $waiting:ident, $count: expr, $dtype:expr, $comm:expr, $stream:expr $(,)?) => {{
                let ranks = &self.comm_groups[&NcclId($comm.id)];
                let trace = self.nccl_ops.$op(ranks, $count, $dtype);
                Self::nccl_call(
                    ranks,
                    &mut self.torch_estimator,
                    &mut self.stream_info,
                    &mut self.queue,
                    &mut self.$waiting,
                    host.host,
                    curr_time,
                    trace,
                    $comm,
                    $stream,
                    call,
                )
            }};
        }

        match msg.call {
            CudaCall::CudaMemcpyAsync { size, kind, stream } => {
                self.cuda_memcpy_async(host, curr_time, size, kind, stream, call)
            }
            CudaCall::CudaDeviceSynchronize(device) => {
                self.cuda_device_synchronize(host, curr_time, device);
            }
            CudaCall::CudaStreamSynchronize(stream) => {
                self.cuda_stream_synchronize(host, curr_time, stream);
            }
            CudaCall::CudaStreamWaitEvent { stream, event } => {
                self.cuda_stream_wait_event(host, curr_time, stream, event)
            }
            CudaCall::CudaStreamQuery(stream) => self.cuda_stream_query(host, curr_time, stream),
            CudaCall::CudaEventRecord(event) => self.cuda_event_record(host, curr_time, event),
            CudaCall::CudaEventSynchronize(event) => {
                self.cuda_event_synchronize(host, curr_time, event);
            }
            CudaCall::CudaEventQuery(event) => {
                self.cuda_event_query(host, curr_time, event);
            }
            CudaCall::CudaAddLatency(stream, latency) => {
                self.cuda_add_latency(host, curr_time, stream, latency, call);
            }

            CudaCall::FlashAttnCall {
                stream,
                is_fwd,
                is_bf16,
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                window_size_left,
                window_size_right,
                is_causal,
            } => self.flash_attn_call(
                host,
                curr_time,
                call,
                stream,
                is_fwd,
                is_bf16,
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                window_size_left,
                window_size_right,
                is_causal,
            ),

            CudaCall::NcclGetUniqueId => {
                self.nccl_get_unique_id(&host);
            }
            CudaCall::NcclCommInitRank {
                device,
                rank,
                nranks,
                id,
            } => self.nccl_comm_init_rank(host, curr_time, device, rank, nranks, id),
            CudaCall::NcclCommSplit {
                comm, color, key, ..
            } => self.nccl_comm_split(host, curr_time, comm.rank, comm.id, color, key),
            CudaCall::NcclBcast {
                count,
                dtype,
                root,
                comm,
                stream,
            } => self.nccl_bcast(host, curr_time, count, dtype, root, comm, stream, call),
            CudaCall::NcclAllReduce {
                count,
                dtype,
                op: _,
                comm,
                stream,
            } => handle_nccl_op!(allreduce, allreduce_waiting, count, dtype, comm, stream,),
            CudaCall::NcclAllGather {
                count,
                dtype,
                comm,
                stream,
            } => handle_nccl_op!(allgather, allgather_waiting, count, dtype, comm, stream,),
            CudaCall::NcclReduceScatter {
                count,
                dtype,
                op: _,
                comm,
                stream,
            } => handle_nccl_op!(
                reduce_scatter,
                reduce_scatter_waiting,
                count,
                dtype,
                comm,
                stream,
            ),
            CudaCall::NcclSend {
                count,
                dtype,
                peer,
                comm,
                stream,
            } => self.nccl_ungrouped_send(host.host, curr_time, count, dtype, peer, comm, stream),
            CudaCall::NcclRecv {
                count,
                dtype,
                peer,
                comm,
                stream,
            } => self.nccl_ungrouped_recv(host.host, curr_time, count, dtype, peer, comm, stream),
            CudaCall::NcclP2pGroup { calls } => self.nccl_grouped_p2p(host.host, curr_time, calls),

            CudaCall::ReadTimer(stream) => {
                self.read_timer(host, curr_time, stream);
            }
        }
    }

    fn cuda_memcpy_async(
        &mut self,
        host: ResponseId,
        curr_time: i64,
        size: usize,
        kind: CudaMemcpyKind,
        stream: CudaStream,
        call: CudaCall,
    ) {
        let comp_time = self.cuda_estimator.memcpy(kind, size);

        Self::add_event(
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            (host.host.clone(), stream),
            vec![],
            Some(Action::Computation(
                NodeId(host.host),
                comp_time.as_micros() as i64,
                ComputeMeta::Cuda(call),
            )),
            curr_time,
        );
    }

    fn cuda_device_synchronize(&mut self, host: ResponseId, curr_time: i64, device: i32) {
        let mut depends_on = vec![];
        for ((other_host, other_stream), sinfo) in self.stream_info.iter_mut() {
            if *other_host == host.host && other_stream.device == device {
                Self::clear_compute_buffer_on(&mut self.torch_estimator, sinfo, &mut self.queue);
                if let Some(id) = sinfo.events.last() {
                    depends_on.push(Some(*id));
                }
            }
        }
        let event = self.queue.add_action_or_point(depends_on, None, curr_time);
        self.syncing.insert(event, host);
        self.execute_to_event(event);
    }

    fn cuda_stream_synchronize(&mut self, host: ResponseId, curr_time: i64, stream: CudaStream) {
        let event = Self::add_event(
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            (host.host.clone(), stream),
            vec![],
            None,
            curr_time,
        );
        self.syncing.insert(event, host);
        self.execute_to_event(event);
    }

    fn cuda_stream_wait_event(
        &mut self,
        host: ResponseId,
        curr_time: i64,
        stream: CudaStream,
        event: CudaEvent,
    ) {
        let event = self.id_of_cuda[&(host.host.clone(), event)];
        Self::add_event(
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            (host.host, stream),
            vec![Some(event)],
            None,
            curr_time,
        );
    }

    fn cuda_stream_query(&mut self, host: ResponseId, curr_time: i64, stream: CudaStream) {
        // Non-blocking poll on a stream: report whether its last event has
        // completed at curr_time. See cuda_event_query.
        let key = (host.host.clone(), stream);
        let last_event = match self.stream_info.get_mut(&key) {
            Some(sinfo) => {
                Self::clear_compute_buffer_on(&mut self.torch_estimator, sinfo, &mut self.queue);
                sinfo.events.last().copied()
            }
            None => None,
        };
        Self::execute_to_time(
            &mut self.queue,
            &mut self.syncing,
            &mut self.host_times,
            curr_time,
        );
        match last_event {
            Some(event) => match self.queue.query(&event) {
                Some(time) => send_query_response_to(
                    &mut self.host_times,
                    host,
                    Some((time, Some(event))),
                    curr_time,
                ),
                None => send_query_response_to(&mut self.host_times, host, None, curr_time),
            },
            None => send_query_response_to(
                &mut self.host_times,
                host,
                Some((curr_time, None)),
                curr_time,
            ),
        }
    }

    fn cuda_event_record(&mut self, host: ResponseId, curr_time: i64, event: CudaEvent) {
        let id = Self::add_event(
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            (
                host.host.clone(),
                CudaStream {
                    device: event.device,
                    id: event.stream,
                },
            ),
            vec![],
            None,
            curr_time,
        );
        self.id_of_cuda.insert((host.host, event), id);
    }

    fn cuda_event_synchronize(&mut self, host: ResponseId, curr_time: i64, event: CudaEvent) {
        Self::execute_to_time(
            &mut self.queue,
            &mut self.syncing,
            &mut self.host_times,
            curr_time,
        );

        let event = self.id_of_cuda[&(host.host.clone(), event)];
        log::trace!(
            "cuda_event_synchronize: {}, curr_time: {}",
            event,
            curr_time
        );
        if let Some(time) = self.queue.query(&event) {
            log::trace!("time: {}", time);
            send_sync_response_to(&mut self.host_times, host, time, Some(event));
        } else {
            log::trace!("executing_to_event: {}", event);
            self.syncing.insert(event, host);
            self.execute_to_event(event);
        }
    }

    fn cuda_event_query(&mut self, host: ResponseId, curr_time: i64, event: CudaEvent) {
        // Non-blocking event poll: report whether the event has completed at
        // curr_time, so application control flow that branches on a not-ready
        // event (e.g. "if not ready, do other work") is preserved. Under
        // ignore-cpu-time mode the host charges a poll cost per not-ready reply,
        // so a pure poll loop still advances virtual time and eventually
        // observes completion instead of spinning forever on a clock that
        // cannot move during a poll.
        Self::execute_to_time(
            &mut self.queue,
            &mut self.syncing,
            &mut self.host_times,
            curr_time,
        );
        let event = self.id_of_cuda[&(host.host.clone(), event)];
        match self.queue.query(&event) {
            Some(time) => send_query_response_to(
                &mut self.host_times,
                host,
                Some((time, Some(event))),
                curr_time,
            ),
            None => send_query_response_to(&mut self.host_times, host, None, curr_time),
        }
    }

    fn cuda_add_latency(
        &mut self,
        host: ResponseId,
        curr_time: i64,
        stream: CudaStream,
        latency: i64,
        call: CudaCall,
    ) {
        Self::add_event(
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            (host.host.clone(), stream),
            vec![],
            Some(Action::Computation(
                NodeId(host.host),
                latency,
                ComputeMeta::Cuda(call),
            )),
            curr_time,
        );
    }

    fn flash_attn_call(
        &mut self,
        host: ResponseId,
        curr_time: i64,
        call: CudaCall,
        stream: CudaStream,
        is_fwd: bool,
        is_bf16: bool,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_size: i32,
        window_size_left: i32,
        window_size_right: i32,
        is_causal: bool,
    ) {
        let comp_time = self.cuda_estimator.flash_attn(
            is_fwd,
            is_bf16,
            batch_size,
            seqlen_q,
            seqlen_k,
            num_heads,
            num_heads_k,
            head_size,
            window_size_left,
            window_size_right,
            is_causal,
        );

        Self::add_event(
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            (host.host.clone(), stream),
            vec![],
            Some(Action::Computation(
                NodeId(host.host),
                comp_time.as_micros() as i64,
                ComputeMeta::Cuda(call),
            )),
            curr_time,
        );
    }

    fn nccl_get_unique_id(&mut self, host: &ResponseId) {
        let nccl_id = incr_nccl_id(&mut self.nccl_unique_id);

        send_response_to(&host, nccl_id.to_vec());
    }

    fn nccl_comm_init_rank(
        &mut self,
        host: ResponseId,
        curr_time: i64,
        device: i32,
        rank: i32,
        nranks: i32,
        id: [u8; 128],
    ) {
        Self::execute_to_time(
            &mut self.queue,
            &mut self.syncing,
            &mut self.host_times,
            curr_time,
        );

        let nccl_id = NcclId(id);

        match self.init_waiting.get_mut(&nccl_id) {
            None => {
                let mut joined_ranks = HashMap::new();
                joined_ranks.insert(rank, (host, device));
                let mut waiting_for = HashSet::new();
                for i in 0..nranks {
                    if i != rank {
                        waiting_for.insert(i);
                    }
                }
                self.init_waiting.insert(
                    nccl_id,
                    InitWaiting {
                        joined_ranks,
                        waiting_for,
                    },
                );
            }
            Some(grp) => {
                grp.waiting_for.remove(&rank);
                grp.joined_ranks.insert(rank, (host, device));
                if grp.waiting_for.is_empty() {
                    if let Some(grp) = self.init_waiting.remove(&nccl_id) {
                        let mut ranks = Vec::new();
                        for i in 0..nranks {
                            ranks.push(grp.joined_ranks[&i].clone());
                        }

                        for (h, _) in ranks.iter() {
                            send_sync_response_to(&mut self.host_times, h.clone(), curr_time, None)
                        }

                        self.comm_groups.insert(
                            nccl_id,
                            ranks
                                .into_iter()
                                .map(|(host, device)| (host.host, device))
                                .collect(),
                        );
                    }
                }
            }
        };
    }

    fn nccl_comm_split(
        &mut self,
        host: ResponseId,
        curr_time: i64,
        rank: i32,
        id: [u8; 128],
        color: i32,
        key: i32,
    ) {
        Self::execute_to_time(
            &mut self.queue,
            &mut self.syncing,
            &mut self.host_times,
            curr_time,
        );

        let nccl_id = NcclId(id);

        if let Some(hosts) = self.comm_groups.get(&nccl_id) {
            let nranks = hosts.len() as i32;
            let device = hosts.iter().find_map(|(hostid, dev)| {
                if host.host == *hostid {
                    Some(dev)
                } else {
                    None
                }
            });
            let device = match device {
                Some(device) => *device,
                None => {
                    log::error!("NCCL group {:?} does not contain host {:?}.", id, host);
                    0
                }
            };
            match self.split_waiting.get_mut(&nccl_id) {
                None => {
                    let mut joined_ranks = HashMap::new();
                    joined_ranks.insert(
                        rank,
                        SplitInfo {
                            host,
                            device,
                            color,
                            key,
                        },
                    );
                    let mut waiting_for = HashSet::new();
                    for i in 0..nranks {
                        if i != rank {
                            waiting_for.insert(i);
                        }
                    }
                    self.split_waiting.insert(
                        nccl_id,
                        SplitWaiting {
                            joined_ranks,
                            waiting_for,
                        },
                    );
                }
                Some(waiting) => {
                    waiting.waiting_for.remove(&rank);
                    waiting.joined_ranks.insert(
                        rank,
                        SplitInfo {
                            host,
                            device,
                            color,
                            key,
                        },
                    );
                    if waiting.waiting_for.is_empty() {
                        let mut new_groups = HashMap::new();
                        for info in waiting.joined_ranks.values() {
                            if info.color == capi::NCCL_SPLIT_NOCOLOR {
                                send_response_to(&info.host, vec![0]);
                            } else {
                                new_groups.entry(info.color).or_insert(vec![]).push(info);
                            }
                        }
                        for new_grp in new_groups.values_mut() {
                            new_grp.sort_by_key(|info| info.key);
                            let new_id = incr_nccl_id(&mut self.nccl_unique_id);
                            for (i, info) in new_grp.iter().enumerate() {
                                log::debug!(
                                    "NcclCommSplit color={}, host={:?}, rank={}/{}, id={:?}",
                                    info.color,
                                    info.host,
                                    i,
                                    new_grp.len(),
                                    new_id
                                );
                                send_response_to(
                                    &info.host,
                                    bincode::serialize(&SplitResponse {
                                        rank: i as i32,
                                        nranks: new_grp.len() as i32,
                                        id: new_id.clone(),
                                        sync: SyncResponse {
                                            end_time: curr_time,
                                        },
                                    })
                                    .unwrap(),
                                );
                                update_host_sync(
                                    &mut self.host_times,
                                    info.host.host.clone(),
                                    curr_time,
                                    None,
                                );
                            }
                            self.comm_groups.insert(
                                NcclId(new_id),
                                new_grp
                                    .iter()
                                    .map(|info| (info.host.host.clone(), info.device))
                                    .collect(),
                            );
                        }
                        self.split_waiting.remove(&nccl_id);
                    }
                }
            }
        } else {
            // error
            log::error!("NCCL id {:?} from {:?} does not exist.", id, host);

            send_response_to(
                &host,
                bincode::serialize(&SplitResponse {
                    rank: 0,
                    nranks: 0,
                    id: [128u8; 128],
                    sync: SyncResponse {
                        end_time: curr_time,
                    },
                })
                .unwrap(),
            );
        }
    }

    fn nccl_ungrouped_send(
        &mut self,
        host: HostId,
        curr_time: i64,
        count: usize,
        dtype: NcclDatatype,
        peer: i32,
        comm: NcclComm,
        stream: CudaStream,
    ) {
        self.nccl_ungrouped_p2p(
            P2PCallKind::Send,
            host,
            curr_time,
            count,
            dtype,
            peer,
            comm,
            stream,
        );
    }

    fn nccl_ungrouped_recv(
        &mut self,
        host: HostId,
        curr_time: i64,
        count: usize,
        dtype: NcclDatatype,
        peer: i32,
        comm: NcclComm,
        stream: CudaStream,
    ) {
        self.nccl_ungrouped_p2p(
            P2PCallKind::Recv,
            host,
            curr_time,
            count,
            dtype,
            peer,
            comm,
            stream,
        );
    }

    fn nccl_ungrouped_p2p(
        &mut self,
        kind: P2PCallKind,
        host: HostId,
        curr_time: i64,
        count: usize,
        dtype: NcclDatatype,
        peer: i32,
        comm: NcclComm,
        stream: CudaStream,
    ) {
        // Ungrouped NCCL P2P is stream ordered: one send/recv call occupies its
        // CUDA stream until the matched network flow completes.  Each endpoint
        // owns a private zero-time barrier, so the same match/enqueue helper can
        // either store this endpoint or match it and release both barriers.
        let key = Self::p2p_key(kind, comm, peer);
        let unmatched_barrier = self.queue.add_action_or_point(vec![None], None, curr_time);
        let endpoint = self.create_ungrouped_p2p_endpoint(
            host,
            stream,
            curr_time,
            count,
            dtype,
            unmatched_barrier,
        );
        self.match_or_enqueue_p2p_endpoint(kind, key, endpoint);
    }

    fn nccl_grouped_p2p(&mut self, host: HostId, curr_time: i64, calls: Vec<CudaCall>) {
        let mut calls_by_stream: HashMap<CudaStream, Vec<P2PCallInfo>> = HashMap::new();
        for call in calls {
            let call_info = match call {
                CudaCall::NcclSend {
                    count,
                    dtype,
                    peer,
                    comm,
                    stream,
                } => Some(P2PCallInfo {
                    kind: P2PCallKind::Send,
                    key: Self::p2p_key(P2PCallKind::Send, comm, peer),
                    stream,
                    count,
                    dtype,
                }),
                CudaCall::NcclRecv {
                    count,
                    dtype,
                    peer,
                    comm,
                    stream,
                } => Some(P2PCallInfo {
                    kind: P2PCallKind::Recv,
                    key: Self::p2p_key(P2PCallKind::Recv, comm, peer),
                    stream,
                    count,
                    dtype,
                }),
                other => {
                    log::warn!("Ignoring non-P2P call in NcclP2pGroup: {:?}", other);
                    None
                }
            };
            if let Some(call_info) = call_info {
                calls_by_stream
                    .entry(call_info.stream)
                    .or_default()
                    .push(call_info);
            }
        }
        if calls_by_stream.is_empty() {
            return;
        }

        // Grouped NCCL P2P is modeled as a single stream-visible grouped op per
        // stream, plus independent raw DAG communication flows.
        //
        // The group start/end points preserve CUDA stream semantics:
        //   previous stream work -> group_start -> group_end -> following work
        //
        // Individual P2P flows are *not* stream events.  A matched pair creates a
        // raw Action::Communication node between off-stream flow_start/flow_end
        // points, so flow A may run while flow B is still unmatched.
        //
        // Each stream group has:
        //   group_barrier -> prevents internal flow points from firing while the
        //                    group's DAG is still being constructed.
        //   group_start   -> stream-ordered start of the fused grouped NCCL op.
        //   group_end     -> stream-ordered end; following stream work waits here.
        //
        // Each P2P call in the group has:
        //   unmatched_barrier -> held until this call matches its peer.
        //   flow_start/end    -> raw DAG points, not stream events.
        //
        // Once the setup is complete, group_barrier is released.  Matching a pair
        // releases only the two per-call unmatched barriers and attaches the real
        // flow events to the corresponding flow_end points.  Since group_end
        // depends on every flow_end, the stream advances only after all internal
        // transfers finish.
        for (stream, grouped_calls) in calls_by_stream {
            let group_barrier = self.queue.add_action_or_point(vec![None], None, curr_time);
            let group_start = Self::add_event(
                &mut self.torch_estimator,
                &mut self.stream_info,
                &mut self.queue,
                (host.clone(), stream),
                vec![Some(group_barrier)],
                None,
                curr_time,
            );

            let mut flow_setups = Vec::new();
            for call in grouped_calls {
                let unmatched_barrier = self.queue.add_action_or_point(vec![None], None, curr_time);
                // Internal grouped flows are raw DAG points, not stream events:
                // the per-flow barrier controls matching, while group_end below
                // is the only stream-visible join for following work.
                let flow_start = self.queue.add_action_or_point(
                    vec![Some(group_start), Some(unmatched_barrier)],
                    None,
                    curr_time,
                );
                let flow_end =
                    self.queue
                        .add_action_or_point(vec![Some(flow_start)], None, curr_time);
                flow_setups.push((call, unmatched_barrier, flow_start, flow_end));
            }

            let mut group_end_dependencies = vec![Some(group_start)];
            group_end_dependencies.extend(
                flow_setups
                    .iter()
                    .map(|(_, _, _, flow_end)| Some(*flow_end)),
            );
            Self::add_event(
                &mut self.torch_estimator,
                &mut self.stream_info,
                &mut self.queue,
                (host.clone(), stream),
                group_end_dependencies,
                None,
                curr_time,
            );

            for (call, unmatched_barrier, flow_start, flow_end) in flow_setups {
                let endpoint = P2PFlowEndpoint {
                    host: host.clone(),
                    stream: call.stream,
                    curr_time,
                    count: call.count,
                    dtype: call.dtype,
                    flow_start,
                    flow_end,
                    unmatched_barrier,
                };
                self.match_or_enqueue_p2p_endpoint(call.kind, call.key, endpoint);
            }

            self.queue.remove_none_dependency(group_barrier);
        }
    }

    fn p2p_key(kind: P2PCallKind, comm: NcclComm, peer: i32) -> P2PKey {
        match kind {
            P2PCallKind::Send => (NcclId(comm.id), comm.rank, peer),
            P2PCallKind::Recv => (NcclId(comm.id), peer, comm.rank),
        }
    }

    fn create_ungrouped_p2p_endpoint(
        &mut self,
        host: HostId,
        stream: CudaStream,
        curr_time: i64,
        count: usize,
        dtype: NcclDatatype,
        unmatched_barrier: EventId,
    ) -> P2PFlowEndpoint {
        let flow_start = Self::add_event(
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            (host.clone(), stream),
            vec![Some(unmatched_barrier)],
            None,
            curr_time,
        );
        let flow_end = Self::add_event(
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            (host.clone(), stream),
            vec![Some(flow_start)],
            None,
            curr_time,
        );

        P2PFlowEndpoint {
            host,
            stream,
            curr_time,
            count,
            dtype,
            flow_start,
            flow_end,
            unmatched_barrier,
        }
    }

    fn match_or_enqueue_p2p_endpoint(
        &mut self,
        kind: P2PCallKind,
        key: P2PKey,
        endpoint: P2PFlowEndpoint,
    ) {
        if let Some(peer_endpoint) = self.pop_matching_peer_endpoint(kind, &key) {
            self.schedule_matched_p2p_flow(kind, endpoint, peer_endpoint, &key);
        } else {
            self.enqueue_unmatched_p2p_endpoint(kind, key, endpoint);
        }
    }

    fn pop_matching_peer_endpoint(
        &mut self,
        kind: P2PCallKind,
        key: &P2PKey,
    ) -> Option<P2PFlowEndpoint> {
        match kind {
            P2PCallKind::Send => Self::pop_p2p_endpoint(&mut self.recv_waiting, key),
            P2PCallKind::Recv => Self::pop_p2p_endpoint(&mut self.send_waiting, key),
        }
    }

    fn pop_p2p_endpoint(
        waiting_map: &mut HashMap<P2PKey, VecDeque<P2PFlowEndpoint>>,
        key: &P2PKey,
    ) -> Option<P2PFlowEndpoint> {
        let endpoint = waiting_map.get_mut(key).and_then(|queue| queue.pop_front());
        if waiting_map.get(key).map_or(false, |queue| queue.is_empty()) {
            waiting_map.remove(key);
        }
        endpoint
    }

    fn enqueue_unmatched_p2p_endpoint(
        &mut self,
        kind: P2PCallKind,
        key: P2PKey,
        endpoint: P2PFlowEndpoint,
    ) {
        let waiting_map = match kind {
            P2PCallKind::Send => &mut self.send_waiting,
            P2PCallKind::Recv => &mut self.recv_waiting,
        };
        waiting_map.entry(key).or_default().push_back(endpoint);
    }

    fn schedule_matched_p2p_flow(
        &mut self,
        kind: P2PCallKind,
        endpoint: P2PFlowEndpoint,
        peer_endpoint: P2PFlowEndpoint,
        key: &P2PKey,
    ) {
        let (send_endpoint, recv_endpoint) = match kind {
            P2PCallKind::Send => (endpoint, peer_endpoint),
            P2PCallKind::Recv => (peer_endpoint, endpoint),
        };
        Self::schedule_p2p_flow(&mut self.queue, send_endpoint, recv_endpoint, key);
    }

    fn release_unmatched_p2p_barrier(queue: &mut EventQueue, endpoint: &P2PFlowEndpoint) {
        queue.remove_none_dependency(endpoint.unmatched_barrier);
    }

    fn schedule_p2p_flow(
        queue: &mut EventQueue,
        send: P2PFlowEndpoint,
        recv: P2PFlowEndpoint,
        key: &P2PKey,
    ) {
        let send_bytes = send.count * send.dtype.size();
        let recv_bytes = recv.count * recv.dtype.size();
        if send_bytes != recv_bytes {
            log::warn!(
                "Mismatched NCCL P2P sizes for {}->{}: send={} recv={}",
                key.1,
                key.2,
                send_bytes,
                recv_bytes
            );
        }
        let flow = netsim::Flow::new(send_bytes, &send.host.hostname, &recv.host.hostname, None);

        let curr_time = send.curr_time.max(recv.curr_time);
        let flow_event = queue.add_action_or_point(
            vec![Some(send.flow_start), Some(recv.flow_start)],
            Some(Action::Communication(
                flow,
                CudaCall::NcclSend {
                    count: send_bytes,
                    dtype: NcclDatatype::U8,
                    peer: key.2,
                    comm: NcclComm {
                        rank: key.1,
                        id: key.0 .0,
                    },
                    stream: send.stream.clone(),
                },
            )),
            curr_time,
        );
        queue.add_dependency(send.flow_end, Some(flow_event));
        queue.add_dependency(recv.flow_end, Some(flow_event));
        Self::release_unmatched_p2p_barrier(queue, &send);
        Self::release_unmatched_p2p_barrier(queue, &recv);
    }

    fn nccl_call(
        ranks: &[(HostId, i32)],
        torch_estimator: &mut TorchEstimator,
        stream_info: &mut HashMap<(HostId, CudaStream), StreamInfo>,
        queue: &mut EventQueue,
        waiting_map: &mut HashMap<NcclId, VecDeque<CCWaiting>>,
        host: HostId,
        curr_time: i64,
        trace: Trace,
        comm: NcclComm,
        stream: CudaStream,
        call: CudaCall,
    ) {
        let nccl_id = NcclId(comm.id);
        let new_cc_waiting = || {
            let mut joined_ranks = HashSet::new();
            joined_ranks.insert(comm.rank);
            let mut waiting_for = HashSet::new();
            for i in 0..(ranks.len() as i32) {
                if i != comm.rank {
                    waiting_for.insert(i);
                }
            }

            let comm_barrier = queue.add_action_or_point(vec![None], None, curr_time);

            let stream_start = Self::add_event(
                torch_estimator,
                stream_info,
                queue,
                (host.clone(), stream.clone()),
                vec![Some(comm_barrier)],
                None,
                curr_time,
            );

            let comm_start_meta =
                queue.add_action_or_point(vec![Some(stream_start)], None, curr_time);
            let comm_end_meta =
                queue.add_action_or_point(vec![Some(comm_start_meta)], None, curr_time);

            let _stream_end = Self::add_event(
                torch_estimator,
                stream_info,
                queue,
                (host.clone(), stream.clone()),
                vec![Some(comm_end_meta)],
                None,
                curr_time,
            );

            CCWaiting {
                trace,
                comm_start_meta,
                comm_end_meta,
                comm_barrier,
                joined_ranks,
                waiting_for,
            }
        };

        match waiting_map.get_mut(&nccl_id) {
            None => {
                let mut waitings = VecDeque::new();
                waitings.push_back(new_cc_waiting());
                waiting_map.insert(nccl_id, waitings);
            }
            Some(waitings) => {
                match waitings.iter_mut().enumerate().find_map(|(i, waiting)| {
                    if waiting.joined_ranks.contains(&comm.rank) {
                        None
                    } else {
                        Some((i, waiting))
                    }
                }) {
                    None => waitings.push_back(new_cc_waiting()),
                    Some((idx, waiting)) => {
                        waiting.waiting_for.remove(&comm.rank);
                        waiting.joined_ranks.insert(comm.rank);

                        let stream_start = Self::add_event(
                            torch_estimator,
                            stream_info,
                            queue,
                            (host.clone(), stream),
                            vec![Some(waiting.comm_barrier)],
                            None,
                            curr_time,
                        );
                        queue.add_dependency(waiting.comm_start_meta, Some(stream_start));
                        let _stream_end = Self::add_event(
                            torch_estimator,
                            stream_info,
                            queue,
                            (host, stream),
                            vec![Some(waiting.comm_end_meta)],
                            None,
                            curr_time,
                        );

                        if waiting.waiting_for.is_empty() {
                            if let Some(waiting) = waitings.remove(idx) {
                                let comm_events: Vec<_> = waiting
                                    .trace
                                    .into_iter()
                                    .map(|flow| {
                                        queue.add_action_or_point(
                                            vec![Some(waiting.comm_start_meta)],
                                            Some(Action::Communication(flow, call.clone())),
                                            curr_time,
                                        )
                                    })
                                    .collect();
                                for comm_event in &comm_events {
                                    queue.add_dependency(waiting.comm_end_meta, Some(*comm_event));
                                }
                                queue.remove_none_dependency(waiting.comm_barrier);
                            }
                        }
                    }
                }
            }
        }
    }

    fn nccl_bcast(
        &mut self,
        host: ResponseId,
        curr_time: i64,
        count: usize,
        dtype: NcclDatatype,
        root: i32,
        comm: NcclComm,
        stream: CudaStream,
        call: CudaCall,
    ) {
        let ranks = &self.comm_groups[&NcclId(comm.id)];
        let trace = self.nccl_ops.bcast(root, ranks, count, dtype);
        Self::nccl_call(
            ranks,
            &mut self.torch_estimator,
            &mut self.stream_info,
            &mut self.queue,
            &mut self.bcast_waiting,
            host.host,
            curr_time,
            trace,
            comm,
            stream,
            call,
        )
    }

    fn read_timer(&mut self, host: ResponseId, curr_time: i64, _stream: CudaStream) {
        update_host_curr(&mut self.host_times, host.host, curr_time);
    }

    pub fn handle_torch_call(&mut self, call: TorchCall) {
        mask_new_host(&call.id.host);
        update_host_curr(&mut self.host_times, call.id.host.clone(), call.time);

        let comp_dur = self.torch_estimator.estimate(&call.info);
        self.stream_info
            .entry((call.id.host.clone(), call.stream.clone()))
            .or_insert_with(Default::default)
            .buffered
            .push((call, comp_dur));
    }

    pub fn handle_exit(&mut self, host: ResponseId, curr_time: i64) {
        log::debug!("{:?} exited at {}", host, curr_time);
        self.exited_hosts.insert(host.host, curr_time);

        // In record mode, dump the profiled timing caches to the DB. Idempotent
        // overwrite on each rank's exit (runs a handful of times); the last write
        // holds the full set of shapes seen this run, merged with any preexisting
        // DB that was preloaded at startup.
        if let Some(dir) = &args::get_args().record_perf_db {
            let db = PerfDb {
                gpu_name: self.cuda_estimator.gpu_name(),
                compute: self.torch_estimator.compute_cache().clone(),
                sequence: self.torch_estimator.sequence_cache().clone(),
                memcpy: self.cuda_estimator.memcpy_cache().clone(),
                flash_attn: self.cuda_estimator.flash_attn_cache().clone(),
            };
            match db.save(dir) {
                Ok(()) => log::info!(
                    "perf-db record: wrote {} ({} compute, {} sequence, {} memcpy, {} flash_attn)",
                    dir.display(),
                    db.compute.len(),
                    db.sequence.len(),
                    db.memcpy.len(),
                    db.flash_attn.len(),
                ),
                Err(e) => log::error!("perf-db record: failed to write {}: {e}", dir.display()),
            }
        }

        // In replay mode, report any shapes that weren't in the DB (discovery).
        // Write them to `<db>.missing/` so they can be profiled on a GPU with
        // bench.py and merged/contributed back. A run with no misses clears any
        // stale manifest. The reported numbers are INVALID when misses occurred
        // (unrecorded ops were charged zero time).
        if let Some(dir) = &args::get_args().perf_db {
            let missing_path =
                std::path::PathBuf::from(format!("{}.missing", dir.display()));
            let compute: HashMap<_, _> = self
                .torch_estimator
                .missing()
                .iter()
                .map(|k| (k.clone(), Duration::ZERO))
                .collect();
            let memcpy: HashMap<_, _> = self
                .cuda_estimator
                .missing_memcpy()
                .iter()
                .map(|&k| (k, Duration::ZERO))
                .collect();
            let flash_attn: HashMap<_, _> = self
                .cuda_estimator
                .missing_flash()
                .iter()
                .map(|k| (k.clone(), Duration::ZERO))
                .collect();
            if compute.is_empty() && memcpy.is_empty() && flash_attn.is_empty() {
                let _ = std::fs::remove_dir_all(&missing_path);
            } else {
                let db = PerfDb {
                    gpu_name: String::new(),
                    compute,
                    sequence: Default::default(),
                    memcpy,
                    flash_attn,
                };
                match db.save(&missing_path) {
                    Ok(()) => {
                        // Go to stderr as well as the log: this invalidates every
                        // number the run just printed, and a log::warn! alone is
                        // invisible when PHANTORA_LOG is set below `warn`.
                        let msg = format!(
                            "perf-db replay: {} compute / {} memcpy / {} flash_attn shape(s) were \
                             NOT in the database -- the simulated numbers from this run are \
                             INVALID. Wrote the missing shapes to {m}. To complete the DB, profile \
                             them on a GPU with `python3 tests/perfdb/bench.py --ref {m} --out {d} \
                             --merge` and re-run (and consider contributing the result back).",
                            db.compute.len(),
                            db.memcpy.len(),
                            db.flash_attn.len(),
                            m = missing_path.display(),
                            d = dir.display(),
                        );
                        log::warn!("{msg}");
                        eprintln!("WARNING: {msg}");
                    }
                    Err(e) => log::error!(
                        "perf-db replay: failed to write missing manifest {}: {e}",
                        missing_path.display()
                    ),
                }
            }
        }
    }
}
