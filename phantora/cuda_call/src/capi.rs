use crate::*;
use ctor::ctor;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::env;
use std::ffi;
use std::fs;
use std::mem;
use std::os::unix::net::UnixDatagram;
use std::process;
use std::ptr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{LazyLock, Mutex, RwLock};

static TIME_OFFSET: AtomicI64 = AtomicI64::new(0); // microseconds
static INITIAL_CPU_TIME: AtomicI64 = AtomicI64::new(0);
static INITIAL_SYSTEM_TIME: AtomicI64 = AtomicI64::new(0);

#[ctor]
fn initialize_times() {
    INITIAL_CPU_TIME.store(current_cpu_time_us(), Ordering::SeqCst);
    INITIAL_SYSTEM_TIME.store(current_sys_time_us(), Ordering::SeqCst);
}

fn current_cpu_time_us() -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts);
    }
    ts.tv_sec * 1_000_000 + ts.tv_nsec / 1_000
}

fn current_sys_time_us() -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts);
    }
    ts.tv_sec * 1_000_000 + ts.tv_nsec / 1_000
}

fn enum_to_nccl_datatype(dtype: i32) -> NcclDatatype {
    match dtype {
        0 => NcclDatatype::I8,
        1 => NcclDatatype::U8,
        2 => NcclDatatype::I32,
        3 => NcclDatatype::U32,
        4 => NcclDatatype::I64,
        5 => NcclDatatype::U64,
        6 => NcclDatatype::F16,
        7 => NcclDatatype::F32,
        8 => NcclDatatype::F64,
        9 => NcclDatatype::Bf16,
        _ => unreachable!("Invalid ncclDataType_t"),
    }
}

fn enum_to_nccl_reduce_op(op: i32) -> NcclReduceOp {
    match op {
        0 => NcclReduceOp::Sum,
        1 => NcclReduceOp::Prod,
        2 => NcclReduceOp::Max,
        3 => NcclReduceOp::Min,
        4 => NcclReduceOp::Avg,
        _ => unreachable!("Invalid ncclRedOp_t"),
    }
}

fn ignore_cpu_time() -> bool {
    env::var_os("PHANTORA_IGNORE_CPU_TIME").is_some()
}

fn get_current_sim_time() -> i64 {
    let initial_system = INITIAL_SYSTEM_TIME.load(Ordering::SeqCst);
    let delta = TIME_OFFSET.load(Ordering::SeqCst);

    if ignore_cpu_time() {
        initial_system + delta
    } else {
        let current_cpu = current_cpu_time_us();
        let initial_cpu = INITIAL_CPU_TIME.load(Ordering::SeqCst);
        initial_system + current_cpu - initial_cpu + delta
    }
}

#[no_mangle]
pub extern "C" fn get_time_long() -> ffi::c_long {
    get_current_sim_time() as _
}

#[no_mangle]
pub extern "C" fn get_time_double() -> ffi::c_double {
    get_current_sim_time() as ffi::c_double / 1e6
}

#[no_mangle]
pub extern "C" fn subtract_cpu_time(us: ffi::c_long) {
    if !ignore_cpu_time() {
        TIME_OFFSET.fetch_sub(us, Ordering::SeqCst);
    }
}

extern "C" {
    pub fn _get_current_device() -> i32;
}

#[no_mangle]
pub unsafe extern "C" fn read_timer() {
    send_cuda_call(CudaCall::ReadTimer(CudaStream {
        device: _get_current_device(),
        id: 0,
    }))
}

const SOCKET_ENV_NAME: &str = "PHANTORA_SOCKET_PREFIX";

pub fn simulator_socket_path() -> ffi::OsString {
    let mut path = env::var_os(SOCKET_ENV_NAME).unwrap();
    path.push(".simulator.sock");
    path
}

pub fn node_socket_path(pid: u32, tid: i32) -> ffi::OsString {
    let mut path = env::var_os(SOCKET_ENV_NAME).unwrap();
    path.push(format!(".node{}_t{}.sock", pid, tid));
    path
}

static SIM_SOCKET: LazyLock<UnixDatagram> = LazyLock::new(|| {
    let sim_socket_path = simulator_socket_path();
    let sim_socket = UnixDatagram::unbound().unwrap();
    sim_socket.connect(sim_socket_path).unwrap();
    sim_socket
});

thread_local! {
    static THIS_SOCKET: UnixDatagram = {
        let pid = process::id();
        let tid = unsafe { libc::gettid() };
        let this_socket_path = node_socket_path(pid, tid);
        let _ = fs::remove_file(&this_socket_path);
        UnixDatagram::bind(this_socket_path).unwrap()
    }
}

fn send_to_simulator(curr_sim_time: i64, call: CudaCall) {
    THIS_SOCKET.with(|_| {}); // create recv socket before sending

    let tid = unsafe { libc::gettid() };
    let msg = CudaCallMsg {
        id: ResponseId {
            host: HostId {
                hostname: hostname::get().map_or(String::from("UNKNOWN_HOST"), |s| {
                    s.into_string().unwrap_or(String::from("UNKNOWN_HOST"))
                }),
                pid: process::id(),
            },
            tid,
        },
        call,
        curr_time: curr_sim_time,
    };
    let mut buf = Vec::new();
    bincode::serialize_into(&mut buf, &msg).unwrap();
    buf.push(1);

    SIM_SOCKET.send(&buf).unwrap();
}

fn recv_from_simulator() -> Vec<u8> {
    const RECV_BUF_SIZE: usize = 256;

    THIS_SOCKET.with(|this_socket| {
        let mut recv_buf = vec![0u8; RECV_BUF_SIZE];
        let sz = this_socket.recv(&mut recv_buf).unwrap();
        assert!(sz < RECV_BUF_SIZE);
        recv_buf.truncate(sz);
        recv_buf
    })
}

fn send_cuda_call(call: CudaCall) {
    let start_time = current_cpu_time_us();
    send_to_simulator(get_current_sim_time(), call);

    let end_time = current_cpu_time_us();
    subtract_cpu_time(end_time - start_time);
}

fn send_cuda_call_get_response(call: CudaCall) -> Vec<u8> {
    let start_time = current_cpu_time_us();
    send_to_simulator(get_current_sim_time(), call);

    let recv_buf = recv_from_simulator();

    let end_time = current_cpu_time_us();
    subtract_cpu_time(end_time - start_time);

    recv_buf
}

fn handle_sync_response(resp: Vec<u8>) -> i64 {
    let msg = bincode::deserialize::<SyncResponse>(&resp).unwrap();
    let curr_sim_time = get_current_sim_time();
    TIME_OFFSET.fetch_add(msg.end_time - curr_sim_time, Ordering::SeqCst);
    msg.end_time
}

struct GPUMemory {
    size: usize,
    blocks: BTreeMap<usize, usize>,
}

impl GPUMemory {
    fn new() -> Self {
        GPUMemory {
            size: 0,
            blocks: BTreeMap::new(),
        }
    }
}

static GPU_MEM: Mutex<BTreeMap<i32, GPUMemory>> = Mutex::new(BTreeMap::new());

#[no_mangle]
pub extern "C" fn cuda_register_malloc(device: i32, ptr: usize, size: usize, total: usize) -> bool {
    let mut gpu_mem = GPU_MEM.lock().unwrap();
    let mem = gpu_mem
        .borrow_mut()
        .entry(device)
        .or_insert_with(GPUMemory::new);
    if mem.size + size > total {
        false
    } else {
        mem.size += size;
        mem.blocks.insert(ptr, size);
        true
    }
}

#[no_mangle]
pub extern "C" fn cuda_register_free(device: i32, ptr: usize) {
    let mut gpu_mem = GPU_MEM.lock().unwrap();
    let mem = gpu_mem.borrow_mut().get_mut(&device).unwrap();
    let size = mem.blocks.remove(&ptr).unwrap();
    assert!(mem.size >= size);
    mem.size -= size;
}

#[no_mangle]
pub extern "C" fn cuda_mem_get_sizeinfo(device: i32) -> usize {
    let mut gpu_mem = GPU_MEM.lock().unwrap();
    let mem = gpu_mem
        .borrow_mut()
        .entry(device)
        .or_insert_with(GPUMemory::new);
    mem.size
}

static PINNED_MEM: RwLock<BTreeMap<usize, usize>> = RwLock::new(BTreeMap::new());

#[no_mangle]
pub extern "C" fn cuda_host_register(ptr: usize, size: usize) {
    PINNED_MEM.write().unwrap().insert(ptr, size);
}

#[no_mangle]
pub extern "C" fn cuda_host_unregister(ptr: usize) {
    PINNED_MEM.write().unwrap().remove(&ptr);
}

#[no_mangle]
pub extern "C" fn cuda_memcpy_async(
    src: usize,
    dst: usize,
    size: usize,
    kind: i32,
    device: i32,
    stream: i32,
) {
    let is_pinned = |loc: usize| {
        let pinned_mem = PINNED_MEM.read().unwrap();
        let mut pinned_before = pinned_mem.range(..loc);
        match pinned_before.next_back() {
            None => false,
            Some((pinned_start, pinned_size)) => pinned_start + pinned_size >= loc + size,
        }
    };

    let kind = match kind {
        0 => CudaMemcpyKind::HostToHost,
        1 => {
            if is_pinned(src) {
                CudaMemcpyKind::PinnedHostToDevice
            } else {
                CudaMemcpyKind::HostToDevice
            }
        }
        2 => {
            if is_pinned(dst) {
                CudaMemcpyKind::DeviceToPinnedHost
            } else {
                CudaMemcpyKind::DeviceToHost
            }
        }
        3 => CudaMemcpyKind::DeviceToDevice,
        4 => CudaMemcpyKind::HostToDevice, // treat Default as HostToDevice
        _ => unreachable!("Invalid cudaMemcpyKind"),
    };

    send_cuda_call(CudaCall::CudaMemcpyAsync {
        size,
        kind,
        stream: CudaStream { device, id: stream },
    })
}

/// Destroy all allocations and reset all state on the current device in the current process.
///
/// # Returns
/// cudaSuccess
#[no_mangle]
pub extern "C" fn cuda_device_reset() {
    GPU_MEM.lock().unwrap().clear();
}

#[no_mangle]
pub extern "C" fn cuda_device_synchronize(device: i32) {
    let resp = send_cuda_call_get_response(CudaCall::CudaDeviceSynchronize(device));
    handle_sync_response(resp);
}

#[no_mangle]
pub extern "C" fn cuda_stream_synchronize(device: i32, id: i32) {
    let resp =
        send_cuda_call_get_response(CudaCall::CudaStreamSynchronize(CudaStream { device, id }));
    handle_sync_response(resp);
}

#[no_mangle]
pub extern "C" fn cuda_stream_wait_event(
    stream_device: i32,
    stream_id: i32,
    event_device: i32,
    event_stream: i32,
    event_id: i32,
) {
    send_cuda_call(CudaCall::CudaStreamWaitEvent {
        stream: CudaStream {
            device: stream_device,
            id: stream_id,
        },
        event: CudaEvent {
            device: event_device,
            stream: event_stream,
            id: event_id,
        },
    });
}

#[no_mangle]
pub extern "C" fn cuda_event_record(device: i32, stream: i32, id: i32) {
    send_cuda_call(CudaCall::CudaEventRecord(CudaEvent { device, stream, id }))
}

#[no_mangle]
pub extern "C" fn cuda_event_synchronize(device: i32, stream: i32, id: i32) -> i64 {
    let resp = send_cuda_call_get_response(CudaCall::CudaEventSynchronize(CudaEvent {
        device,
        stream,
        id,
    }));
    handle_sync_response(resp)
}

#[no_mangle]
pub extern "C" fn cuda_event_query(
    device: i32,
    stream: i32,
    id: i32,
    time_ref: *mut ffi::c_long,
) -> i32 {
    let resp =
        send_cuda_call_get_response(CudaCall::CudaEventQuery(CudaEvent { device, stream, id }));
    match bincode::deserialize::<Option<i64>>(&resp).unwrap() {
        None => 0,
        Some(time) => {
            unsafe { *time_ref = time };
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn cuda_add_latency(device: i32, stream: i32, latency: i64) {
    send_cuda_call(CudaCall::CudaAddLatency(
        CudaStream { device, id: stream },
        latency,
    ));
}

#[no_mangle]
pub extern "C" fn cuda_stream_query(device: i32, stream: i32) -> i32 {
    let resp =
        send_cuda_call_get_response(CudaCall::CudaStreamQuery(CudaStream { device, id: stream }));
    bincode::deserialize::<bool>(&resp).unwrap() as i32
}

// #[repr(C)]
// #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
// pub struct Dim3 {
//     pub x: ffi::c_uint,
//     pub y: ffi::c_uint,
//     pub z: ffi::c_uint,
// }

#[no_mangle]
pub extern "C" fn cuda_launch_kernel(
    func: *const ffi::c_void,
    // grid_dim: &Dim3,
    // block_dim: &Dim3,
    args: *mut *mut ffi::c_void,
    // shared_mem: usize,
    device: i32,
    stream: i32,
) {
    // dladdr and demangle (especially dladdr) can be slow
    let start_time = current_cpu_time_us();

    let raw_kernel_name;
    unsafe {
        let mut dlinfo = mem::zeroed::<libc::Dl_info>();
        libc::dladdr(func, &mut dlinfo);
        if dlinfo.dli_sname == ptr::null() {
            return;
        }
        raw_kernel_name = ffi::CStr::from_ptr(dlinfo.dli_sname).to_str().unwrap();
    }
    let kernel_name = match cpp_demangle::Symbol::new(raw_kernel_name.as_bytes()) {
        Ok(symbol) => symbol.to_string(),
        Err(_) => raw_kernel_name.to_owned(),
    };

    if kernel_name.starts_with("void flash_fwd_kernel") {
        let stream = CudaStream { device, id: stream };
        let params = unsafe { &*(*args as *const FlashFwdParams) };
        let call = params.extract_call(
            stream,
            // raw_kernel_name.to_string(),
            // grid_dim.clone(),
            // block_dim.clone(),
            // shared_mem,
            true,
        );
        send_cuda_call(call);
    } else if kernel_name.starts_with("void flash_bwd_dot_do_o_kernel") {
        // 3 kernels for one backward pass:
        // flash_bwd_dot_do_o_kernel
        // -> flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel
        // -> flash_bwd_convert_dq_kernel
        let stream = CudaStream { device, id: stream };
        let params = unsafe { &(*(*args as *const FlashBwdParams)).fwd };
        let call = params.extract_call(
            stream,
            // raw_kernel_name.to_string(),
            // grid_dim.clone(),
            // block_dim.clone(),
            // shared_mem,
            false,
        );
        send_cuda_call(call);
    }

    let end_time = current_cpu_time_us();
    subtract_cpu_time(end_time - start_time);
}

/// https://github.com/Dao-AILab/flash-attention/blob/v2.7.3/csrc/flash_attn/src/flash.h
type Index = i64;

#[repr(C)]
pub struct QkvParams {
    pub q_ptr: *mut ffi::c_void,
    pub k_ptr: *mut ffi::c_void,
    pub v_ptr: *mut ffi::c_void,

    pub q_batch_stride: Index,
    pub k_batch_stride: Index,
    pub v_batch_stride: Index,
    pub q_row_stride: Index,
    pub k_row_stride: Index,
    pub v_row_stride: Index,
    pub q_head_stride: Index,
    pub k_head_stride: Index,
    pub v_head_stride: Index,

    pub h: ffi::c_int,
    pub h_k: ffi::c_int,
    pub h_h_k_ratio: ffi::c_int,
}

#[repr(C)]
pub struct FlashFwdParams {
    pub qkv: QkvParams, // C++ inheritance

    pub o_ptr: *mut ffi::c_void,
    pub oaccum_ptr: *mut ffi::c_void,

    pub o_batch_stride: Index,
    pub o_row_stride: Index,
    pub o_head_stride: Index,

    pub p_ptr: *mut ffi::c_void,

    pub softmax_lse_ptr: *mut ffi::c_void,
    pub softmax_lseaccum_ptr: *mut ffi::c_void,

    pub b: ffi::c_int,
    pub seqlen_q: ffi::c_int,
    pub seqlen_k: ffi::c_int,
    pub seqlen_knew: ffi::c_int,
    pub d: ffi::c_int,
    pub seqlen_q_rounded: ffi::c_int,
    pub seqlen_k_rounded: ffi::c_int,
    pub d_rounded: ffi::c_int,
    pub rotary_dim: ffi::c_int,
    pub total_q: ffi::c_int,

    pub scale_softmax: ffi::c_float,
    pub scale_softmax_log2: ffi::c_float,

    pub cu_seqlens_q: *mut ffi::c_int,
    pub cu_seqlens_k: *mut ffi::c_int,
    pub leftpad_k: *mut ffi::c_int,

    pub seqused_k: *mut ffi::c_int,

    pub blockmask: *mut ffi::c_int,

    pub knew_ptr: *mut ffi::c_void,
    pub vnew_ptr: *mut ffi::c_void,

    pub knew_batch_stride: Index,
    pub vnew_batch_stride: Index,
    pub knew_row_stride: Index,
    pub vnew_row_stride: Index,
    pub knew_head_stride: Index,
    pub vnew_head_stride: Index,

    pub rotary_cos_ptr: *mut ffi::c_void,
    pub rotary_sin_ptr: *mut ffi::c_void,

    pub cache_batch_idx: *mut ffi::c_int,

    pub block_table: *mut ffi::c_int,
    pub block_table_batch_stride: Index,
    pub page_block_size: ffi::c_int,

    pub p_dropout: ffi::c_float,
    pub p_dropout_in_uint8_t: u8,

    pub rp_dropout: ffi::c_float,
    pub scale_softmax_rp_dropout: ffi::c_float,

    pub window_size_left: ffi::c_int,
    pub window_size_right: ffi::c_int,
    pub softcap: ffi::c_float,

    // Placeholder for at::PhiloxCudaState
    // sizeof == 24
    // struct PhiloxCudaState {
    //   union Payload {
    //     uint64_t val;
    //     int64_t* ptr;
    //   };
    //   Payload seed_{};
    //   Payload offset_{};
    //   uint32_t offset_intragraph_ = 0;
    //   bool captured_ = false;
    // };
    pub philox_args: [u8; 24],

    pub rng_state: *mut u64,

    pub is_bf16: bool,
    pub is_causal: bool,

    pub is_seqlens_k_cumulative: bool,

    pub is_rotary_interleaved: bool,

    pub num_splits: ffi::c_int,

    pub alibi_slopes_ptr: *mut ffi::c_void,
    pub alibi_slopes_batch_stride: Index,

    pub unpadded_lse: bool,
    pub seqlenq_ngroups_swapped: bool,
}

#[repr(C)]
pub struct FlashBwdParams {
    pub fwd: FlashFwdParams, // C++ inheritance

    pub do_ptr: *mut ffi::c_void,
    pub dq_ptr: *mut ffi::c_void,
    pub dk_ptr: *mut ffi::c_void,
    pub dv_ptr: *mut ffi::c_void,

    pub dq_accum_ptr: *mut ffi::c_void,
    pub dk_accum_ptr: *mut ffi::c_void,
    pub dv_accum_ptr: *mut ffi::c_void,

    pub do_batch_stride: Index,
    pub do_row_stride: Index,
    pub do_head_stride: Index,
    pub dq_batch_stride: Index,
    pub dk_batch_stride: Index,
    pub dv_batch_stride: Index,
    pub dq_row_stride: Index,
    pub dk_row_stride: Index,
    pub dv_row_stride: Index,
    pub dq_head_stride: Index,
    pub dk_head_stride: Index,
    pub dv_head_stride: Index,

    pub dsoftmax_sum: *mut ffi::c_void,

    pub deterministic: bool,
    pub dq_accum_split_stride: Index,
}

impl FlashFwdParams {
    fn extract_call(
        &self,
        stream: CudaStream,
        // kernel_name: String,
        // grid_dim: Dim3,
        // block_dim: Dim3,
        // shared_mem: usize,
        is_fwd: bool,
    ) -> CudaCall {
        CudaCall::FlashAttnCall {
            stream,
            // kernel_name,
            // grid_dim,
            // block_dim,
            // shared_mem,
            is_fwd,
            is_bf16: self.is_bf16,
            batch_size: self.b,
            seqlen_q: self.seqlen_q,
            seqlen_k: self.seqlen_k,
            num_heads: self.qkv.h,
            num_heads_k: self.qkv.h_k,
            head_size: self.d,
            window_size_left: self.window_size_left,
            window_size_right: self.window_size_right,
            is_causal: self.is_causal,
        }
    }
}

pub const NCCL_SPLIT_NOCOLOR: i32 = -1;

#[no_mangle]
pub extern "C" fn nccl_get_unique_id(id: *mut ffi::c_char) {
    let resp = send_cuda_call_get_response(CudaCall::NcclGetUniqueId);
    assert!(resp.len() == 128);
    unsafe {
        ptr::copy_nonoverlapping(resp.as_ptr(), id as _, 128);
    }
}

fn get_comm_id(comm_id: *const ffi::c_char) -> [u8; 128] {
    let mut id = [0u8; 128];
    unsafe {
        ptr::copy_nonoverlapping(comm_id as _, id.as_mut_ptr(), 128);
    }
    id
}

struct NcclCaller {
    group_counter: i32,
    grouped_calls: Vec<CudaCall>,
}

impl NcclCaller {
    fn group_start(&mut self) {
        self.group_counter += 1;
    }

    fn group_end(&mut self) {
        if self.group_counter > 0 {
            self.group_counter -= 1;
        }
        if self.group_counter == 0 {
            let mut responsed_calls = Vec::new();
            for call in self.grouped_calls.drain(..) {
                if let CudaCall::NcclCommInitRank { nranks, .. } = &call {
                    if *nranks > 1 {
                        responsed_calls.push(call.clone());
                        send_cuda_call(call);
                    }
                } else if let CudaCall::NcclCommSplit { .. } = &call {
                    responsed_calls.push(call.clone());
                    send_cuda_call(call);
                } else {
                    send_cuda_call(call);
                }
            }
            for call in responsed_calls {
                match call {
                    CudaCall::NcclCommInitRank { .. } => {
                        let resp = recv_from_simulator();
                        handle_sync_response(resp);
                    }
                    CudaCall::NcclCommSplit {
                        color,
                        rank_out,
                        nrank_out,
                        id_out,
                        ..
                    } => {
                        let resp = recv_from_simulator();
                        if color != NCCL_SPLIT_NOCOLOR {
                            let resp = bincode::deserialize::<SplitResponse>(&resp).unwrap();
                            unsafe {
                                rank_out.inner.write(resp.rank);
                                nrank_out.inner.write(resp.nranks);
                                ptr::copy_nonoverlapping(resp.id.as_ptr(), id_out.inner, 128);
                            }
                        }
                    }
                    _ => (),
                }
            }
        }
    }

    fn call(&mut self, call: CudaCall) {
        if self.group_counter > 0 {
            self.grouped_calls.push(call);
        } else if let CudaCall::NcclCommInitRank { nranks, .. } = &call {
            if *nranks > 1 {
                let resp = send_cuda_call_get_response(call);
                handle_sync_response(resp);
            }
        } else if let CudaCall::NcclCommSplit {
            color,
            rank_out,
            nrank_out,
            id_out,
            ..
        } = &call
        {
            let rank_out = rank_out.clone();
            let nrank_out = nrank_out.clone();
            let id_out = id_out.clone();
            let color = *color;
            let resp = send_cuda_call_get_response(call);
            if color != NCCL_SPLIT_NOCOLOR {
                let resp = bincode::deserialize::<SplitResponse>(&resp).unwrap();
                unsafe {
                    rank_out.inner.write(resp.rank);
                    nrank_out.inner.write(resp.nranks);
                    ptr::copy_nonoverlapping(resp.id.as_ptr(), id_out.inner, 128);
                }
            }
        } else {
            send_cuda_call(call);
        }
    }
}

thread_local! {
    static NCCL_CALLER: RefCell<NcclCaller> = const {
        RefCell::new(NcclCaller {
            group_counter: 0,
            grouped_calls: Vec::new(),
        })
    };
}

#[no_mangle]
pub extern "C" fn nccl_group_start() {
    NCCL_CALLER.with_borrow_mut(|caller| caller.group_start());
}

#[no_mangle]
pub extern "C" fn nccl_group_end() {
    NCCL_CALLER.with_borrow_mut(|caller| caller.group_end());
}

#[no_mangle]
pub extern "C" fn nccl_comm_init_rank(
    nranks: i32,
    comm_id: *const ffi::c_char,
    rank: i32,
    device: i32,
) {
    let id = get_comm_id(comm_id);
    NCCL_CALLER.with_borrow_mut(|caller| {
        caller.call(CudaCall::NcclCommInitRank {
            device,
            rank,
            nranks,
            id,
        })
    });
}

#[no_mangle]
pub extern "C" fn nccl_comm_split(
    rank: i32,
    comm_id: *const ffi::c_char,
    color: i32,
    key: i32,
    rank_out: *mut i32,
    nrank_out: *mut i32,
    id_out: *mut u8,
) {
    let id = get_comm_id(comm_id);
    NCCL_CALLER.with_borrow_mut(|caller| {
        caller.call(CudaCall::NcclCommSplit {
            comm: NcclComm { rank, id },
            color,
            key,
            rank_out: LocalPtr { inner: rank_out },
            nrank_out: LocalPtr { inner: nrank_out },
            id_out: LocalPtr { inner: id_out },
        })
    });
}

#[no_mangle]
pub extern "C" fn nccl_bcast(
    count: usize,
    dtype: i32,
    root: i32,
    comm_id: *const ffi::c_char,
    rank: i32,
    device: i32,
    stream: i32,
) {
    let id = get_comm_id(comm_id);
    NCCL_CALLER.with_borrow_mut(|caller| {
        caller.call(CudaCall::NcclBcast {
            count,
            dtype: enum_to_nccl_datatype(dtype),
            root,
            comm: NcclComm { rank, id },
            stream: CudaStream { device, id: stream },
        })
    });
}

#[no_mangle]
pub extern "C" fn nccl_all_reduce(
    count: usize,
    dtype: i32,
    op: i32,
    comm_id: *const ffi::c_char,
    rank: i32,
    device: i32,
    stream: i32,
) {
    let id = get_comm_id(comm_id);
    NCCL_CALLER.with_borrow_mut(|caller| {
        caller.call(CudaCall::NcclAllReduce {
            count,
            dtype: enum_to_nccl_datatype(dtype),
            op: enum_to_nccl_reduce_op(op),
            comm: NcclComm { rank, id },
            stream: CudaStream { device, id: stream },
        })
    });
}

#[no_mangle]
pub extern "C" fn nccl_all_gather(
    count: usize,
    dtype: i32,
    comm_id: *const ffi::c_char,
    rank: i32,
    device: i32,
    stream: i32,
) {
    let id = get_comm_id(comm_id);
    NCCL_CALLER.with_borrow_mut(|caller| {
        caller.call(CudaCall::NcclAllGather {
            count,
            dtype: enum_to_nccl_datatype(dtype),
            comm: NcclComm { rank, id },
            stream: CudaStream { device, id: stream },
        })
    });
}

#[no_mangle]
pub extern "C" fn nccl_reduce_scatter(
    count: usize,
    dtype: i32,
    op: i32,
    comm_id: *const ffi::c_char,
    rank: i32,
    device: i32,
    stream: i32,
) {
    let id = get_comm_id(comm_id);
    NCCL_CALLER.with_borrow_mut(|caller| {
        caller.call(CudaCall::NcclReduceScatter {
            count,
            dtype: enum_to_nccl_datatype(dtype),
            op: enum_to_nccl_reduce_op(op),
            comm: NcclComm { rank, id },
            stream: CudaStream { device, id: stream },
        })
    });
}
