#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>

__attribute__((unused)) static void*
load_original_func(const char* name)
{
    const char* LIBCUDA_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
    static void* lib = NULL;
    if (!lib) {
        lib = dlopen(LIBCUDA_PATH, RTLD_LAZY);
        if (!lib) {
            fprintf(stderr, "DLOPEN: can not load \"%s\"", LIBCUDA_PATH);
            exit(1);
        }
    }
    void* f = dlsym(lib, name);
    if (!f) {
        fprintf(stderr,
                "DLSYM: can not load \"%s\" from \"%s\"\n",
                name,
                LIBCUDA_PATH);
        exit(1);
    }
    return f;
}

CUresult
cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
{
    *flags = CU_CTX_SCHED_AUTO;
    *active = 1;
    return CUDA_SUCCESS;
}

CUresult
cuCtxGetCurrent(CUcontext* pctx)
{
    static unsigned char zeros[256] = { 0 };
    *pctx = (CUcontext)zeros;
    return CUDA_SUCCESS;
}

CUresult
cuMemAddressReserve(CUdeviceptr* ptr,
                    size_t size,
                    size_t alignment,
                    CUdeviceptr addr,
                    unsigned long long flags)
{
    // TODO
    *ptr = addr;
    return CUDA_SUCCESS;
}

CUresult
cuMemCreate(CUmemGenericAllocationHandle* handle,
            size_t size,
            const CUmemAllocationProp* prop,
            unsigned long long flags)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuMemMap(CUdeviceptr ptr,
         size_t size,
         size_t offset,
         CUmemGenericAllocationHandle handle,
         unsigned long long flags)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuMemSetAccess(CUdeviceptr ptr,
               size_t size,
               const CUmemAccessDesc* desc,
               size_t count)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuMemRelease(CUmemGenericAllocationHandle handle)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuModuleLoadData(CUmodule* module, const void* image)
{
    // TODO
    static unsigned char zeros[256] = { 0 };
    *module = (CUmodule)zeros;
    return CUDA_SUCCESS;
}

CUresult
cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name)
{
    // TODO
    static unsigned char zeros[256] = { 0 };
    *hfunc = (CUfunction)zeros;
    return CUDA_SUCCESS;
}

CUresult
cuLaunchKernel(CUfunction f,
               unsigned int gridDimX,
               unsigned int gridDimY,
               unsigned int gridDimZ,
               unsigned int blockDimX,
               unsigned int blockDimY,
               unsigned int blockDimZ,
               unsigned int sharedMemBytes,
               CUstream hStream,
               void** kernelParams,
               void** extra)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
    // TODO
    *pi = 0;
    return CUDA_SUCCESS;
}

CUresult
cuMemGetAllocationGranularity(size_t* granularity,
                              const CUmemAllocationProp* prop,
                              CUmemAllocationGranularity_flags option)
{
    // TODO
    *granularity = 0x100;
    return CUDA_SUCCESS;
}

CUresult
cuMemExportToShareableHandle(void* shareableHandle,
                             CUmemGenericAllocationHandle handle,
                             CUmemAllocationHandleType handleType,
                             unsigned long long flags)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
                               void* osHandle,
                               CUmemAllocationHandleType shHandleType)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuMemsetD32Async(CUdeviceptr dstDevice,
                 unsigned int ui,
                 size_t N,
                 CUstream hStream)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuStreamWriteValue32(CUstream stream,
                     CUdeviceptr addr,
                     cuuint32_t value,
                     unsigned int flags)
{
    // TODO
    return CUDA_SUCCESS;
}

CUresult
cuGetErrorString(CUresult error, const char** pStr)
{
    switch (error) {
        case CUDA_SUCCESS:
            *pStr = "CUDA_SUCCESS";
            break;
        case CUDA_ERROR_INVALID_VALUE:
            *pStr = "CUDA_ERROR_INVALID_VALUE";
            break;
        case CUDA_ERROR_OUT_OF_MEMORY:
            *pStr = "CUDA_ERROR_OUT_OF_MEMORY";
            break;
        case CUDA_ERROR_NOT_INITIALIZED:
            *pStr = "CUDA_ERROR_NOT_INITIALIZED";
            break;
        case CUDA_ERROR_DEINITIALIZED:
            *pStr = "CUDA_ERROR_DEINITIALIZED";
            break;
        case CUDA_ERROR_PROFILER_DISABLED:
            *pStr = "CUDA_ERROR_PROFILER_DISABLED";
            break;
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            *pStr = "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
            break;
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            *pStr = "CUDA_ERROR_PROFILER_ALREADY_STARTED";
            break;
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            *pStr = "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
            break;
        case CUDA_ERROR_STUB_LIBRARY:
            *pStr = "CUDA_ERROR_STUB_LIBRARY";
            break;
        case CUDA_ERROR_DEVICE_UNAVAILABLE:
            *pStr = "CUDA_ERROR_DEVICE_UNAVAILABLE";
            break;
        case CUDA_ERROR_NO_DEVICE:
            *pStr = "CUDA_ERROR_NO_DEVICE";
            break;
        case CUDA_ERROR_INVALID_DEVICE:
            *pStr = "CUDA_ERROR_INVALID_DEVICE";
            break;
        case CUDA_ERROR_DEVICE_NOT_LICENSED:
            *pStr = "CUDA_ERROR_DEVICE_NOT_LICENSED";
            break;
        case CUDA_ERROR_INVALID_IMAGE:
            *pStr = "CUDA_ERROR_INVALID_IMAGE";
            break;
        case CUDA_ERROR_INVALID_CONTEXT:
            *pStr = "CUDA_ERROR_INVALID_CONTEXT";
            break;
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            *pStr = "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
            break;
        case CUDA_ERROR_MAP_FAILED:
            *pStr = "CUDA_ERROR_MAP_FAILED";
            break;
        case CUDA_ERROR_UNMAP_FAILED:
            *pStr = "CUDA_ERROR_UNMAP_FAILED";
            break;
        case CUDA_ERROR_ARRAY_IS_MAPPED:
            *pStr = "CUDA_ERROR_ARRAY_IS_MAPPED";
            break;
        case CUDA_ERROR_ALREADY_MAPPED:
            *pStr = "CUDA_ERROR_ALREADY_MAPPED";
            break;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            *pStr = "CUDA_ERROR_NO_BINARY_FOR_GPU";
            break;
        case CUDA_ERROR_ALREADY_ACQUIRED:
            *pStr = "CUDA_ERROR_ALREADY_ACQUIRED";
            break;
        case CUDA_ERROR_NOT_MAPPED:
            *pStr = "CUDA_ERROR_NOT_MAPPED";
            break;
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            *pStr = "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
            break;
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            *pStr = "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
            break;
        case CUDA_ERROR_ECC_UNCORRECTABLE:
            *pStr = "CUDA_ERROR_ECC_UNCORRECTABLE";
            break;
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            *pStr = "CUDA_ERROR_UNSUPPORTED_LIMIT";
            break;
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            *pStr = "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
            break;
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            *pStr = "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
            break;
        case CUDA_ERROR_INVALID_PTX:
            *pStr = "CUDA_ERROR_INVALID_PTX";
            break;
        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            *pStr = "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
            break;
        case CUDA_ERROR_NVLINK_UNCORRECTABLE:
            *pStr = "CUDA_ERROR_NVLINK_UNCORRECTABLE";
            break;
        case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
            *pStr = "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
            break;
        case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
            *pStr = "CUDA_ERROR_UNSUPPORTED_PTX_VERSION";
            break;
        case CUDA_ERROR_JIT_COMPILATION_DISABLED:
            *pStr = "CUDA_ERROR_JIT_COMPILATION_DISABLED";
            break;
        case CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY:
            *pStr = "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY";
            break;
        case CUDA_ERROR_INVALID_SOURCE:
            *pStr = "CUDA_ERROR_INVALID_SOURCE";
            break;
        case CUDA_ERROR_FILE_NOT_FOUND:
            *pStr = "CUDA_ERROR_FILE_NOT_FOUND";
            break;
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            *pStr = "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
            break;
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            *pStr = "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
            break;
        case CUDA_ERROR_OPERATING_SYSTEM:
            *pStr = "CUDA_ERROR_OPERATING_SYSTEM";
            break;
        case CUDA_ERROR_INVALID_HANDLE:
            *pStr = "CUDA_ERROR_INVALID_HANDLE";
            break;
        case CUDA_ERROR_ILLEGAL_STATE:
            *pStr = "CUDA_ERROR_ILLEGAL_STATE";
            break;
        case CUDA_ERROR_NOT_FOUND:
            *pStr = "CUDA_ERROR_NOT_FOUND";
            break;
        case CUDA_ERROR_NOT_READY:
            *pStr = "CUDA_ERROR_NOT_READY";
            break;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            *pStr = "CUDA_ERROR_ILLEGAL_ADDRESS";
            break;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            *pStr = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
            break;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            *pStr = "CUDA_ERROR_LAUNCH_TIMEOUT";
            break;
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            *pStr = "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
            break;
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            *pStr = "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
            break;
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            *pStr = "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
            break;
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            *pStr = "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
            break;
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            *pStr = "CUDA_ERROR_CONTEXT_IS_DESTROYED";
            break;
        case CUDA_ERROR_ASSERT:
            *pStr = "CUDA_ERROR_ASSERT";
            break;
        case CUDA_ERROR_TOO_MANY_PEERS:
            *pStr = "CUDA_ERROR_TOO_MANY_PEERS";
            break;
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            *pStr = "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
            break;
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            *pStr = "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
            break;
        case CUDA_ERROR_HARDWARE_STACK_ERROR:
            *pStr = "CUDA_ERROR_HARDWARE_STACK_ERROR";
            break;
        case CUDA_ERROR_ILLEGAL_INSTRUCTION:
            *pStr = "CUDA_ERROR_ILLEGAL_INSTRUCTION";
            break;
        case CUDA_ERROR_MISALIGNED_ADDRESS:
            *pStr = "CUDA_ERROR_MISALIGNED_ADDRESS";
            break;
        case CUDA_ERROR_INVALID_ADDRESS_SPACE:
            *pStr = "CUDA_ERROR_INVALID_ADDRESS_SPACE";
            break;
        case CUDA_ERROR_INVALID_PC:
            *pStr = "CUDA_ERROR_INVALID_PC";
            break;
        case CUDA_ERROR_LAUNCH_FAILED:
            *pStr = "CUDA_ERROR_LAUNCH_FAILED";
            break;
        case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
            *pStr = "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
            break;
        case CUDA_ERROR_NOT_PERMITTED:
            *pStr = "CUDA_ERROR_NOT_PERMITTED";
            break;
        case CUDA_ERROR_NOT_SUPPORTED:
            *pStr = "CUDA_ERROR_NOT_SUPPORTED";
            break;
        case CUDA_ERROR_SYSTEM_NOT_READY:
            *pStr = "CUDA_ERROR_SYSTEM_NOT_READY";
            break;
        case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
            *pStr = "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH";
            break;
        case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
            *pStr = "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE";
            break;
        case CUDA_ERROR_MPS_CONNECTION_FAILED:
            *pStr = "CUDA_ERROR_MPS_CONNECTION_FAILED";
            break;
        case CUDA_ERROR_MPS_RPC_FAILURE:
            *pStr = "CUDA_ERROR_MPS_RPC_FAILURE";
            break;
        case CUDA_ERROR_MPS_SERVER_NOT_READY:
            *pStr = "CUDA_ERROR_MPS_SERVER_NOT_READY";
            break;
        case CUDA_ERROR_MPS_MAX_CLIENTS_REACHED:
            *pStr = "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED";
            break;
        case CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED:
            *pStr = "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED";
            break;
        case CUDA_ERROR_MPS_CLIENT_TERMINATED:
            *pStr = "CUDA_ERROR_MPS_CLIENT_TERMINATED";
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
            *pStr = "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";
            break;
        case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
            *pStr = "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";
            break;
        case CUDA_ERROR_STREAM_CAPTURE_MERGE:
            *pStr = "CUDA_ERROR_STREAM_CAPTURE_MERGE";
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
            *pStr = "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED";
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
            *pStr = "CUDA_ERROR_STREAM_CAPTURE_UNJOINED";
            break;
        case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
            *pStr = "CUDA_ERROR_STREAM_CAPTURE_ISOLATION";
            break;
        case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
            *pStr = "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT";
            break;
        case CUDA_ERROR_CAPTURED_EVENT:
            *pStr = "CUDA_ERROR_CAPTURED_EVENT";
            break;
        case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
            *pStr = "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD";
            break;
        case CUDA_ERROR_TIMEOUT:
            *pStr = "CUDA_ERROR_TIMEOUT";
            break;
        case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
            *pStr = "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE";
            break;
        case CUDA_ERROR_EXTERNAL_DEVICE:
            *pStr = "CUDA_ERROR_EXTERNAL_DEVICE";
            break;
        case CUDA_ERROR_INVALID_CLUSTER_SIZE:
            *pStr = "CUDA_ERROR_INVALID_CLUSTER_SIZE";
            break;
        case CUDA_ERROR_UNKNOWN:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}
