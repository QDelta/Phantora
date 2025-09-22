use crate::args;
use crate::cuda_estimate::CudaEstimator;
use crate::event_queue::{Action, EventId, EventQueue, QueueStep};
use crate::nccl_ops::{NcclOps, SimpleRing, Trace};
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

#[derive(strum::Display, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeMeta {
    Torch(TorchCall),
    Cuda(CudaCall),
}

pub type CommMeta = CudaCall;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct NodeId(pub HostId);

#[derive(PartialEq, Eq, Hash)]
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
    pub fn new(netsim: netsim::simulator::Simulator) -> Self {
        Simulator {
            cuda_estimator: CudaEstimator::new(),
            torch_estimator: TorchEstimator::new(),

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
        }
    }

    fn split_sequences(calls: Vec<(TorchCall, Duration)>) -> Vec<TorchCallSeq> {
        if calls.is_empty() {
            vec![]
        } else {
            if args::get_args().disable_sequence_call {
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
        let key = (host.host.clone(), stream);
        let stream_completed = match self.stream_info.get_mut(&key) {
            Some(sinfo) => {
                Self::clear_compute_buffer_on(&mut self.torch_estimator, sinfo, &mut self.queue);
                Self::execute_to_time(
                    &mut self.queue,
                    &mut self.syncing,
                    &mut self.host_times,
                    curr_time,
                );
                if let Some(event) = sinfo.events.last() {
                    self.queue.query(event).is_some()
                } else {
                    true
                }
            }
            None => true,
        };
        send_response_to(&host, bincode::serialize(&stream_completed).unwrap());
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
        Self::execute_to_time(
            &mut self.queue,
            &mut self.syncing,
            &mut self.host_times,
            curr_time,
        );
        let event = self.id_of_cuda[&(host.host.clone(), event)];
        let resp = self.queue.query(&event);
        send_response_to(&host, bincode::serialize(&resp).unwrap());
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
    }
}
