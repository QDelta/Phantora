// use capi::Dim3;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaStream {
    pub device: i32,
    pub id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaEvent {
    pub device: i32,
    pub stream: i32,
    pub id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CudaMemcpyKind {
    HostToHost,
    HostToDevice,
    PinnedHostToDevice,
    DeviceToHost,
    DeviceToPinnedHost,
    DeviceToDevice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NcclDatatype {
    I8,
    U8,
    I32,
    U32,
    I64,
    U64,
    F16,
    F32,
    F64,
    Bf16,
}

impl NcclDatatype {
    pub fn size(&self) -> usize {
        match self {
            NcclDatatype::I8 => 1,
            NcclDatatype::U8 => 1,
            NcclDatatype::I32 => 4,
            NcclDatatype::U32 => 4,
            NcclDatatype::I64 => 8,
            NcclDatatype::U64 => 8,
            NcclDatatype::F16 => 2,
            NcclDatatype::F32 => 4,
            NcclDatatype::F64 => 8,
            NcclDatatype::Bf16 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NcclReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Avg,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NcclComm {
    pub rank: i32,
    #[serde(with = "BigArray")]
    pub id: [u8; 128],
}

// Host use only, will be ignored in (de)serialization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalPtr<T> {
    pub inner: *mut T,
}

impl<T> Serialize for LocalPtr<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(0)
    }
}

impl<'de, T> Deserialize<'de> for LocalPtr<T> {
    fn deserialize<D>(_: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(LocalPtr {
            inner: std::ptr::null_mut(),
        })
    }
}

#[derive(strum::Display, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CudaCall {
    // CUDA runtime calls
    CudaMemcpyAsync {
        size: usize,
        kind: CudaMemcpyKind,
        stream: CudaStream,
    },
    CudaDeviceSynchronize(i32),
    CudaStreamSynchronize(CudaStream),
    CudaStreamWaitEvent {
        stream: CudaStream,
        event: CudaEvent,
    },
    CudaStreamQuery(CudaStream),
    CudaEventRecord(CudaEvent),
    CudaEventSynchronize(CudaEvent),
    CudaEventQuery(CudaEvent),
    CudaAddLatency(CudaStream, i64),

    // we just add flash attention calls in an ad-hoc way
    FlashAttnCall {
        // cudaLaunchKernel params
        stream: CudaStream,
        // kernel_name: String,
        // grid_dim: Dim3,
        // block_dim: Dim3,
        // shared_mem: usize,
        // flash attention params
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
    },

    // NCCL calls
    NcclGetUniqueId,
    NcclCommInitRank {
        device: i32,
        rank: i32,
        nranks: i32,
        #[serde(with = "BigArray")]
        id: [u8; 128],
    },
    NcclCommSplit {
        comm: NcclComm,
        color: i32,
        key: i32,
        // FOLLOWING FIELDS ARE HOST USE ONLY
        // IGNORE IN SIMULATOR
        rank_out: LocalPtr<i32>,
        nrank_out: LocalPtr<i32>,
        id_out: LocalPtr<u8>,
    },
    NcclBcast {
        count: usize,
        dtype: NcclDatatype,
        root: i32,
        comm: NcclComm,
        stream: CudaStream,
    },
    NcclAllReduce {
        count: usize,
        dtype: NcclDatatype,
        op: NcclReduceOp,
        comm: NcclComm,
        stream: CudaStream,
    },
    NcclAllGather {
        count: usize,
        dtype: NcclDatatype,
        comm: NcclComm,
        stream: CudaStream,
    },
    NcclReduceScatter {
        count: usize,
        dtype: NcclDatatype,
        op: NcclReduceOp,
        comm: NcclComm,
        stream: CudaStream,
    },

    ReadTimer(CudaStream),
}

impl CudaCall {
    pub fn get_nccl_comm(&self) -> Option<NcclComm> {
        use CudaCall::*;
        match self {
            NcclBcast { comm, .. }
            | NcclAllReduce { comm, .. }
            | NcclAllGather { comm, .. }
            | NcclReduceScatter { comm, .. } => Some(comm.clone()),
            _ => None,
        }
    }

    pub fn get_cuda_stream(&self) -> Option<CudaStream> {
        use CudaCall::*;
        match self {
            CudaMemcpyAsync { stream, .. } => Some(*stream),
            CudaStreamSynchronize(stream) => Some(*stream),
            CudaStreamWaitEvent { stream, .. } => Some(*stream),
            CudaEventRecord(event) | CudaEventSynchronize(event) | CudaEventQuery(event) => {
                Some(CudaStream {
                    device: event.device,
                    id: event.stream,
                })
            }
            NcclBcast { stream, .. }
            | NcclAllReduce { stream, .. }
            | NcclAllGather { stream, .. }
            | NcclReduceScatter { stream, .. } => Some(*stream),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HostId {
    pub hostname: String,
    pub pid: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResponseId {
    pub host: HostId,
    pub tid: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaCallMsg {
    pub id: ResponseId,
    pub curr_time: i64,
    pub call: CudaCall,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyncResponse {
    pub end_time: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SplitResponse {
    pub rank: i32,
    pub nranks: i32,
    #[serde(with = "BigArray")]
    pub id: [u8; 128],
    pub sync: SyncResponse,
}

pub mod capi;
