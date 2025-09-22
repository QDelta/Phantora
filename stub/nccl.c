#include "nccl.h"
#include "common.h"
#include "phantora.h"
#include <string.h>

ncclResult_t
ncclGetUniqueId(ncclUniqueId* uniqueId)
{
    nccl_get_unique_id(uniqueId->internal);
    return ncclSuccess;
}

ncclResult_t
ncclGroupStart()
{
    nccl_group_start();
    return ncclSuccess;
}

ncclResult_t
ncclGroupEnd()
{
    nccl_group_end();
    return ncclSuccess;
}

struct ncclComm
{
    int rank;
    int nranks;
    char id[NCCL_UNIQUE_ID_BYTES];
};

ncclResult_t
ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
{
    ncclComm_t c = (ncclComm_t)malloc(sizeof(struct ncclComm));
    c->rank = rank;
    c->nranks = nranks;
    memcpy(c->id, commId.internal, NCCL_UNIQUE_ID_BYTES);
    *comm = c;
    nccl_comm_init_rank(nranks, commId.internal, rank, _get_current_device());
    return ncclSuccess;
}

ncclResult_t
ncclBcast(void* buff,
          size_t count,
          ncclDataType_t datatype,
          int root,
          ncclComm_t comm,
          cudaStream_t stream)
{
    if (comm->nranks <= 1)
        return ncclSuccess;
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    nccl_bcast(
      count, datatype, root, comm->id, comm->rank, stream_.device, stream_.id);
    return ncclSuccess;
}

ncclResult_t
ncclAllReduce(const void* sendbuff,
              void* recvbuff,
              size_t count,
              ncclDataType_t datatype,
              ncclRedOp_t op,
              ncclComm_t comm,
              cudaStream_t stream)
{
    if (comm->nranks <= 1)
        return ncclSuccess;
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    nccl_all_reduce(
      count, datatype, op, comm->id, comm->rank, stream_.device, stream_.id);
    return ncclSuccess;
}

ncclResult_t
ncclAllGather(const void* sendbuff,
              void* recvbuff,
              size_t sendcount,
              ncclDataType_t datatype,
              ncclComm_t comm,
              cudaStream_t stream)
{
    if (comm->nranks <= 1)
        return ncclSuccess;
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    nccl_all_gather(
      sendcount, datatype, comm->id, comm->rank, stream_.device, stream_.id);
    return ncclSuccess;
}

ncclResult_t
ncclReduceScatter(const void* sendbuff,
                  void* recvbuff,
                  size_t recvcount,
                  ncclDataType_t datatype,
                  ncclRedOp_t op,
                  ncclComm_t comm,
                  cudaStream_t stream)
{
    if (comm->nranks <= 1)
        return ncclSuccess;
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    nccl_reduce_scatter(recvcount,
                        datatype,
                        op,
                        comm->id,
                        comm->rank,
                        stream_.device,
                        stream_.id);
    return ncclSuccess;
}

ncclResult_t
ncclCommAbort(ncclComm_t comm)
{
    free(comm);
    return ncclSuccess;
}

ncclResult_t
ncclCommDestroy(ncclComm_t comm)
{
    free(comm);
    return ncclSuccess;
}

ncclResult_t
ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError)
{
    *asyncError = ncclSuccess;
    return ncclSuccess;
}

ncclResult_t
ncclGetVersion(int* version)
{
    if (version == NULL)
        return ncclInvalidArgument;
    *version = NCCL_VERSION_CODE;
    return ncclSuccess;
}

ncclResult_t
ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist)
{
    ncclUniqueId uniqueId;
    nccl_group_start();
    nccl_get_unique_id(uniqueId.internal);
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err) {
        return ncclSystemError;
    }
    if (ndev > count) {
        return ncclInvalidArgument;
    }
    for (int i = 0; i < ndev; i++) {
        if (devlist[i] >= count || devlist[i] < 0) {
            return ncclInvalidArgument;
        }
    }
    int nranks = ndev;
    for (int rank = 0; rank < nranks; rank++) {
        ncclComm_t c = (ncclComm_t)malloc(sizeof(struct ncclComm));
        comm[rank] = c;
        c->rank = rank;
        c->nranks = ndev;
        memcpy(c->id, uniqueId.internal, NCCL_UNIQUE_ID_BYTES);
        if (devlist) {
            nccl_comm_init_rank(nranks, uniqueId.internal, rank, devlist[rank]);
        } else {
            nccl_comm_init_rank(nranks, uniqueId.internal, rank, rank);
        }
    }
    nccl_group_end();
    return ncclSuccess;
}

const char*
ncclGetLastError(ncclComm_t comm)
{
    return "no error";
}

const char*
ncclGetErrorString(ncclResult_t result)
{
    switch (result) {
        case ncclSuccess:
            return "no error";
        case ncclUnhandledCudaError:
            return "unhandled cuda error (run with NCCL_DEBUG=INFO for "
                   "details)";
        case ncclSystemError:
            return "unhandled system error (run with NCCL_DEBUG=INFO for "
                   "details)";
        case ncclInternalError:
            return "internal error - please report this issue to the NCCL "
                   "developers";
        case ncclInvalidArgument:
            return "invalid argument (run with NCCL_DEBUG=WARN for details)";
        case ncclInvalidUsage:
            return "invalid usage (run with NCCL_DEBUG=WARN for details)";
        case ncclRemoteError:
            return "remote process exited or there was a network error";
        case ncclInProgress:
            return "NCCL operation in progress";
        default:
            return "unknown result code";
    }
}

ncclResult_t
ncclCommInitRankConfig(ncclComm_t* comm,
                       int nranks,
                       ncclUniqueId commId,
                       int rank,
                       ncclConfig_t* config)
{
    return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t
ncclCommInitRankScalable(ncclComm_t* comm,
                         int nranks,
                         int rank,
                         int nId,
                         ncclUniqueId* commIds,
                         ncclConfig_t* config)
{
    return ncclCommInitRank(comm, nranks, commIds[0], rank);
}

ncclResult_t
ncclCommFinalize(ncclComm_t comm)
{
    return ncclSuccess;
}

ncclResult_t
ncclCommSplit(ncclComm_t comm,
              int color,
              int key,
              ncclComm_t* newcomm,
              ncclConfig_t* config)
{
    ncclComm_t c = (ncclComm_t)malloc(sizeof(struct ncclComm));
    nccl_comm_split(comm->rank,
                    comm->id,
                    color,
                    key,
                    &(c->rank),
                    &(c->nranks),
                    (unsigned char*)(&(c->id)));
    *newcomm = c;
    return ncclSuccess;
}

ncclResult_t
ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle)
{
    return ncclSuccess;
}

ncclResult_t
ncclCommDeregister(const ncclComm_t comm, void* handle)
{
    return ncclSuccess;
}

ncclResult_t
ncclCommCount(const ncclComm_t comm, int* count)
{
    NOT_IMPLEMENTED;
}

ncclResult_t
ncclCommCuDevice(const ncclComm_t comm, int* device)
{
    NOT_IMPLEMENTED;
}

ncclResult_t
ncclCommUserRank(const ncclComm_t comm, int* rank)
{
    NOT_IMPLEMENTED;
}

ncclResult_t
ncclRedOpCreatePreMulSum(ncclRedOp_t* op,
                         void* scalar,
                         ncclDataType_t datatype,
                         ncclScalarResidence_t residence,
                         ncclComm_t comm)
{
    NOT_IMPLEMENTED;
}

ncclResult_t
ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm)
{
    NOT_IMPLEMENTED;
}

ncclResult_t
ncclReduce(const void* sendbuff,
           void* recvbuff,
           size_t count,
           ncclDataType_t datatype,
           ncclRedOp_t op,
           int root,
           ncclComm_t comm,
           cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

ncclResult_t
ncclBroadcast(const void* sendbuff,
              void* recvbuff,
              size_t count,
              ncclDataType_t datatype,
              int root,
              ncclComm_t comm,
              cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

ncclResult_t
ncclSend(const void* sendbuff,
         size_t count,
         ncclDataType_t datatype,
         int peer,
         ncclComm_t comm,
         cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

ncclResult_t
ncclRecv(void* recvbuff,
         size_t count,
         ncclDataType_t datatype,
         int peer,
         ncclComm_t comm,
         cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}
