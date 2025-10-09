#include "common.h"
#include "phantora.h"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

// __attribute__((unused)) static void*
// load_original_func(const char* name)
// {
//     void* f = dlsym(RTLD_NEXT, name);
//     if (!f) {
//         fprintf(stderr, "DLSYM: can not load \"%s\"\n", name);
//         exit(1);
//     }
//     return f;
// }

static _Thread_local int CURRENT_DEVICE = 0;

static atomic_int STREAM_COUNTER = ATOMIC_VAR_INIT(1);
static atomic_int EVENT_COUNTER = ATOMIC_VAR_INIT(0);

int
_get_device_count()
{
    static int DEVICE_COUNT = 0;
    if (DEVICE_COUNT == 0) {
        const char* device_count_str = getenv("PHANTORA_NGPU");
        if (device_count_str) {
            DEVICE_COUNT = atoi(device_count_str);
        } else {
            DEVICE_COUNT = 1;
        }
    }
    return DEVICE_COUNT;
}

int
_get_current_device()
{
    return CURRENT_DEVICE;
}

size_t
_get_device_memory()
{
    static size_t DEVICE_MEM = 0;
    if (DEVICE_MEM == 0) {
        const char* device_mem_str = getenv("PHANTORA_VRAM_MIB");
        if (device_mem_str) {
            DEVICE_MEM = atoll(device_mem_str) << 20;
        } else {
            DEVICE_MEM = 24576ull << 20; // 24 GiB
        }
    }
    return DEVICE_MEM;
}

cudaError_t
cudaGetDeviceCount(int* count)
{
    *count = _get_device_count();
    return cudaSuccess;
}

cudaError_t
cudaGetDevice(int* device)
{
    *device = CURRENT_DEVICE;
    return cudaSuccess;
}

cudaError_t
cudaStreamIsCapturing(cudaStream_t stream,
                      enum cudaStreamCaptureStatus* pCaptureStatus)
{
    *pCaptureStatus = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

// TODO: make these more configurable
// or get these from simulator
static struct cudaDeviceProp DEVICE_PROP = {
    .name = { 0 },
    .uuid = { .bytes = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    .luid = { 0, 0, 0, 0, 0, 0, 0, 0 },
    .luidDeviceNodeMask = 0,
    .totalGlobalMem = 0, // 25438126080,
    .sharedMemPerBlock = 49152,
    .regsPerBlock = 65536,
    .warpSize = 32,
    .memPitch = 2147483647,
    .maxThreadsPerBlock = 1024,
    .maxThreadsDim = { 1024, 1024, 64 },
    .maxGridSize = { 2147483647, 65535, 65535 },
    .clockRate = 1695000,
    .totalConstMem = 65536,
    .major = 8,
    .minor = 6,
    .textureAlignment = 512,
    .texturePitchAlignment = 32,
    .deviceOverlap = 1,
    .multiProcessorCount = 82,
    .kernelExecTimeoutEnabled = 0,
    .integrated = 0,
    .canMapHostMemory = 1,
    .computeMode = 0,
    .maxTexture1D = 131072,
    .maxTexture1DMipmap = 32768,
    .maxTexture1DLinear = 268435456,
    .maxTexture2D = { 131072, 65536 },
    .maxTexture2DMipmap = { 32768, 32768 },
    .maxTexture2DLinear = { 131072, 65000, 2097120 },
    .maxTexture2DGather = { 32768, 32768 },
    .maxTexture3D = { 16384, 16384, 16384 },
    .maxTexture3DAlt = { 8192, 8192, 32768 },
    .maxTextureCubemap = 32768,
    .maxTexture1DLayered = { 32768, 2048 },
    .maxTexture2DLayered = { 32768, 32768, 2048 },
    .maxTextureCubemapLayered = { 32768, 2046 },
    .maxSurface1D = 32768,
    .maxSurface2D = { 131072, 65536 },
    .maxSurface3D = { 16384, 16384, 16384 },
    .maxSurface1DLayered = { 32768, 2048 },
    .maxSurface2DLayered = { 32768, 32768, 2048 },
    .maxSurfaceCubemap = 32768,
    .maxSurfaceCubemapLayered = { 32768, 2046 },
    .surfaceAlignment = 512,
    .concurrentKernels = 1,
    .ECCEnabled = 0,
    .pciBusID = 94,
    .pciDeviceID = 0,
    .pciDomainID = 0,
    .tccDriver = 0,
    .asyncEngineCount = 2,
    .unifiedAddressing = 1,
    .memoryClockRate = 9751000,
    .memoryBusWidth = 384,
    .l2CacheSize = 6291456,
    .persistingL2CacheMaxSize = 4718592,
    .maxThreadsPerMultiProcessor = 1536,
    .streamPrioritiesSupported = 1,
    .globalL1CacheSupported = 1,
    .sharedMemPerMultiprocessor = 102400,
    .regsPerMultiprocessor = 65536,
    .managedMemory = 1,
    .isMultiGpuBoard = 0,
    .hostNativeAtomicSupported = 0,
    .singleToDoublePrecisionPerfRatio = 32,
    .pageableMemoryAccess = 0,
    .concurrentManagedAccess = 1,
    .computePreemptionSupported = 1,
    .canUseHostPointerForRegisteredMem = 1,
    .cooperativeLaunch = 1,
    .cooperativeMultiDeviceLaunch = 1,
    .sharedMemPerBlockOptin = 101376,
    .pageableMemoryAccessUsesHostPageTables = 0,
    .directManagedMemAccessFromHost = 0,
    .accessPolicyMaxWindowSize = 134213632,
    .reservedSharedMemPerBlock = 1024,
};

void
_init_device_props(void)
{
    if (DEVICE_PROP.totalGlobalMem == 0) {
        DEVICE_PROP.totalGlobalMem = _get_device_memory();
        const char* device_name_str = getenv("PHANTORA_GPU_NAME");
        if (device_name_str) {
            strncpy(DEVICE_PROP.name, device_name_str, 255);
        } else {
            strncpy(DEVICE_PROP.name, "Phantora GPU", 255);
        }
    }
}

cudaError_t
cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device)
{
    _init_device_props();
    *prop = DEVICE_PROP;
    return cudaSuccess;
}

cudaError_t
cudaGetLastError(void)
{
    return cudaSuccess;
}

cudaError_t
cudaPeekAtLastError(void)
{
    return cudaSuccess;
}

static const size_t MEM_ALIGNMENT_MASK = 0xFF;
static atomic_uintptr_t MEM_LIMIT = MEM_ALIGNMENT_MASK + 1;

cudaError_t
cudaMalloc(void** devPtr, size_t size)
{
    if (size == 0) {
        *devPtr = NULL;
        return cudaSuccess;
    }

    size_t aligned_size = (size + MEM_ALIGNMENT_MASK) & ~MEM_ALIGNMENT_MASK;
    size_t total_mem = _get_device_memory();
    if (cuda_register_malloc(
          _get_current_device(), MEM_LIMIT, aligned_size, total_mem)) {
        *devPtr = (void*)MEM_LIMIT;
        MEM_LIMIT += aligned_size;
        return cudaSuccess;
    } else {
        return cudaErrorMemoryAllocation;
    }
}

cudaError_t
cudaLaunchKernel(const void* func,
                 dim3 gridDim,
                 dim3 blockDim,
                 void** args,
                 size_t sharedMem,
                 cudaStream_t stream)
{
    // Dl_info info;
    // dladdr(func, &info);
    // const char* func_name = info.dli_sname;

    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    cuda_launch_kernel(func,
                       // (const Dim3*)&gridDim,
                       // (const Dim3*)&blockDim,
                       args,
                       // sharedMem,
                       stream_.device,
                       stream_.id);

    return cudaSuccess;
}

cudaError_t
cudaMemcpyAsync(void* dst,
                const void* src,
                size_t count,
                enum cudaMemcpyKind kind,
                cudaStream_t stream)
{
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    cuda_memcpy_async(
      (uintptr_t)src, (uintptr_t)dst, count, kind, stream_.device, stream_.id);
    // may crash things
    // if (kind == cudaMemcpyDeviceToHost) {
    //     memset(dst, 0, count);
    // }
    return cudaSuccess;
}

cudaError_t
cudaStreamSynchronize(cudaStream_t stream)
{
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    cuda_stream_synchronize(stream_.device, stream_.id);
    return cudaSuccess;
}

cudaError_t
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                       const void* func,
                                                       int blockSize,
                                                       size_t dynamicSMemSize,
                                                       unsigned int flags)
{
    *numBlocks = 1;
    return cudaSuccess;
}

cudaError_t
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                              const void* func,
                                              int blockSize,
                                              size_t dynamicSMemSize)
{
    *numBlocks = 1;
    return cudaSuccess;
}

cudaError_t
cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream)
{
    return cudaSuccess;
}

const struct cudaFuncAttributes FUNC_ATTRS = {
    .sharedSizeBytes = 41760,
    .constSizeBytes = 0,
    .localSizeBytes = 960,
    .maxThreadsPerBlock = 640,
    .numRegs = 96,
    .ptxVersion = 86,
    .binaryVersion = 86,
    .cacheModeCA = 0,
    .maxDynamicSharedSizeBytes = 7392,
    .preferredShmemCarveout = -1,
};

cudaError_t
cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func)
{
    *attr = FUNC_ATTRS;
    return cudaSuccess;
}

cudaError_t
cudaFree(void* devPtr)
{
    if (devPtr) {
        cuda_register_free(_get_current_device(), (uintptr_t)devPtr);
    }
    return cudaSuccess;
}

cudaError_t
cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode)
{
    return cudaSuccess;
}

cudaError_t
cudaStreamCreateWithPriority(cudaStream_t* pStream,
                             unsigned int flags,
                             int priority)
{
    struct phantora_cudaStream* stream_ =
      malloc(sizeof(struct phantora_cudaStream));
    stream_->device = CURRENT_DEVICE;
    stream_->id = STREAM_COUNTER++;
    *pStream = (cudaStream_t)stream_;
    return cudaSuccess;
}

cudaError_t
cudaSetDevice(int device)
{
    if (device < 0 || device >= _get_device_count()) {
        return cudaErrorInvalidDevice;
    }
    CURRENT_DEVICE = device;
    return cudaSuccess;
}

cudaError_t
cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
{
    struct phantora_cudaEvent* event_ =
      malloc(sizeof(struct phantora_cudaEvent));
    event_->device = CURRENT_DEVICE;
    event_->stream = 0;
    event_->id = EVENT_COUNTER++;
    event_->finished_time = 0.0;
    *event = (cudaEvent_t)event_;
    return cudaSuccess;
}

cudaError_t
cudaDeviceGetPCIBusId(char* pciBusId, int len, int device)
{
    // const char *PCI_ID = "0000:01:00.0";
    int dev = device & 0xFF;
    int bus = (device >> 8) & 0xFF;
    int domain = ((device >> 16) & 0xFFF) | 0x1000;
    snprintf(pciBusId, len, "%04X:%02X:%02X.0", domain, bus, dev);
    return cudaSuccess;
}

cudaError_t
cudaDriverGetVersion(int* driverVersion)
{
    *driverVersion = 11080;
    return cudaSuccess;
}

const int DEVICE_ATTRS[] = {
    1024,   1024,  1024,       64,      2147483647, 65535,   65535, 49152,
    65536,  32,    2147483647, 65536,   1695000,    512,     1,     82,
    0,      0,     1,          0,       131072,     131072,  65536, 16384,
    16384,  16384, 32768,      32768,   2048,       512,     1,     0,
    94,     0,     0,          9751000, 384,        6291456, 1536,  2,
    1,      32768, 2048,       1,       32768,      32768,   8192,  8192,
    32768,  0,     32,         32768,   32768,      2046,    32768, 131072,
    65536,  16384, 16384,      16384,   32768,      2048,    32768, 32768,
    2048,   32768, 32768,      2046,    268435456,  131072,  65000, 2097120,
    32768,  32768, 8,          6,       32768,      1,       1,     1,
    102400, 65536, 1,          0,       0,          0,       32,    0,
    1,      1,     1,          0,       0,          0,       1,     1,
    101376, 0,     1,          0,       0,          1,       1,     0,
    0,      16,    1,          4718592, 134213632,  0,       1024,  1,
    1,      1,     1,          0,       1,          0,       1,     0,
    1
};

cudaError_t
cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device)
{
    *value = DEVICE_ATTRS[attr - 1];
    return cudaSuccess;
}

cudaError_t
cudaDeviceGetByPCIBusId(int* device, const char* pciBusId)
{
    int domain, bus, dev;
    sscanf(pciBusId, "%X:%X:%X.0", &domain, &bus, &dev);
    *device = ((domain & 0x7FFF) << 16) + (bus << 8) + dev;
    return cudaSuccess;
}

cudaError_t
cudaStreamDestroy(cudaStream_t stream)
{
    if (stream)
        free(stream);
    return cudaSuccess;
}

cudaError_t
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    struct phantora_cudaEvent* event_ = (struct phantora_cudaEvent*)event;
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    event_->stream = stream_.id;
    cuda_event_record(event_->device, event_->stream, event_->id);
    return cudaSuccess;
}

cudaError_t
cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    struct phantora_cudaEvent* event_ = (struct phantora_cudaEvent*)event;
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    cuda_stream_wait_event(
      stream_.device, stream_.id, event_->device, event_->stream, event_->id);
    return cudaSuccess;
}

cudaError_t
cudaEventDestroy(cudaEvent_t event)
{
    free(event);
    return cudaSuccess;
}

cudaError_t
cudaEventSynchronize(cudaEvent_t event)
{
    struct phantora_cudaEvent* event_ = (struct phantora_cudaEvent*)event;
    long end_time_ms =
      cuda_event_synchronize(event_->device, event_->stream, event_->id);
    event_->finished_time = end_time_ms;
    return cudaSuccess;
}

cudaError_t
cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
    struct phantora_cudaEvent* start_ = (struct phantora_cudaEvent*)start;
    struct phantora_cudaEvent* end_ = (struct phantora_cudaEvent*)end;
    *ms = (float)(end_->finished_time - start_->finished_time);
    return cudaSuccess;
}

cudaError_t
cudaEventQuery(cudaEvent_t event)
{
    struct phantora_cudaEvent* event_ = (struct phantora_cudaEvent*)event;
    // cuda_add_latency(event_->device, event_->stream, _query_gpu_latency());
    int finished = cuda_event_query(
      event_->device, event_->stream, event_->id, &(event_->finished_time));
    return finished ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t
cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)
{
    *canAccessPeer = 1;
    return cudaSuccess;
}

cudaError_t
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    return cudaSuccess;
}

cudaError_t
cudaDeviceSynchronize(void)
{
    cuda_device_synchronize(CURRENT_DEVICE);
    return cudaSuccess;
}

cudaError_t
cudaPointerGetAttributes(struct cudaPointerAttributes* attributes,
                         const void* ptr)
{
    // TODO, maybe
    attributes->type = cudaMemoryTypeUnregistered;
    attributes->device = CURRENT_DEVICE;
    attributes->devicePointer = NULL;
    attributes->hostPointer = NULL;
    return cudaSuccess;
}

cudaError_t
cudaHostAlloc(void** pHost, size_t size, unsigned int flags)
{
    int ret = posix_memalign(pHost, MEM_ALIGNMENT_MASK + 1, size);
    if (ret != 0) {
        return cudaErrorMemoryAllocation;
    } else {
        cuda_host_register((uintptr_t)(*pHost), size);
        return cudaSuccess;
    }
}

cudaError_t
cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
{
    return cudaStreamCreate(pStream);
}

cudaError_t
cudaFreeHost(void* ptr)
{
    cuda_host_unregister((uintptr_t)ptr);
    return cudaSuccess;
}

cudaError_t
cudaDeviceReset(void)
{
    cuda_device_reset();
    return cudaSuccess;
}

int
_dummy() // accept any number of arguments
{
    return 0;
}

cudaError_t
cudaGetDriverEntryPoint(const char* symbol,
                        void** funcPtr,
                        unsigned long long flags)
{
    // TODO
    *funcPtr = _dummy;
    return cudaSuccess;
}

cudaError_t
cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
    if (leastPriority)
        *leastPriority = 0;
    if (greatestPriority)
        *greatestPriority = 0;
    return cudaSuccess;
}

cudaError_t
cudaStreamCreate(cudaStream_t* pStream)
{
    return cudaStreamCreateWithPriority(pStream, cudaStreamDefault, 0);
}

cudaError_t
cudaStreamQuery(cudaStream_t stream)
{
    struct phantora_cudaStream stream_ = phantora_cudaStream(stream);
    int finished = cuda_stream_query(stream_.device, stream_.id);
    return finished ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t
cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value)
{
    return cudaSuccess;
}

cudaError_t
cudaMemGetInfo(size_t* free, size_t* total)
{
    size_t total_mem = _get_device_memory();
    size_t alloced_mem = cuda_mem_get_sizeinfo(_get_current_device());
    *free = total_mem - alloced_mem;
    *total = total_mem;
    return cudaSuccess;
}

cudaError_t
cudaMemset(void* devPtr, int value, size_t count)
{
    return cudaSuccess;
}
