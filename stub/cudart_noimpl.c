#include "common.h"
#include <cuda_runtime_api.h>

cudaError_t
cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceGetTexture1DLinearMaxWidth(
  size_t* maxWidthInElements,
  const struct cudaChannelFormatDesc* fmtDesc,
  int device)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceGetCacheConfig(enum cudaFuncCache* pCacheConfig)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig* pConfig)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaIpcOpenMemHandle(void** devPtr,
                     cudaIpcMemHandle_t handle,
                     unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaIpcCloseMemHandle(void* devPtr)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceFlushGPUDirectRDMAWrites(
  enum cudaFlushGPUDirectRDMAWritesTarget target,
  enum cudaFlushGPUDirectRDMAWritesScope scope)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaThreadSynchronize(void)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaThreadGetCacheConfig(enum cudaFuncCache* pCacheConfig)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    NOT_IMPLEMENTED;
}

const char*
cudaGetErrorName(cudaError_t error)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceSetMemPool(int device, cudaMemPool_t memPool)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceGetP2PAttribute(int* value,
                          enum cudaDeviceP2PAttr attr,
                          int srcDevice,
                          int dstDevice)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaChooseDevice(int* device, const struct cudaDeviceProp* prop)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaSetDeviceFlags(unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetDeviceFlags(unsigned int* flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamGetPriority(cudaStream_t hStream, int* priority)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaCtxResetPersistingL2Cache(void)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamGetAttribute(cudaStream_t hStream,
                       enum cudaStreamAttrID attr,
                       union cudaStreamAttrValue* value_out)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamSetAttribute(cudaStream_t hStream,
                       enum cudaStreamAttrID attr,
                       const union cudaStreamAttrValue* value)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamAddCallback(cudaStream_t stream,
                      cudaStreamCallback_t callback,
                      void* userData,
                      unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamAttachMemAsync(cudaStream_t stream,
                         void* devPtr,
                         size_t length,
                         unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamGetCaptureInfo(cudaStream_t stream,
                         enum cudaStreamCaptureStatus* pCaptureStatus,
                         unsigned long long* pId)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamGetCaptureInfo_v2(cudaStream_t stream,
                            enum cudaStreamCaptureStatus* captureStatus_out,
                            unsigned long long* id_out /* = 0 */,
                            cudaGraph_t* graph_out /* = 0 */,
                            const cudaGraphNode_t** dependencies_out /* = 0 */,
                            size_t* numDependencies_out /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                    cudaGraphNode_t* dependencies,
                                    size_t numDependencies,
                                    unsigned int flags /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaEventCreate(cudaEvent_t* event)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaEventRecordWithFlags(cudaEvent_t event,
                         cudaStream_t stream,
                         unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaImportExternalMemory(
  cudaExternalMemory_t* extMem_out,
  const struct cudaExternalMemoryHandleDesc* memHandleDesc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaExternalMemoryGetMappedBuffer(
  void** devPtr,
  cudaExternalMemory_t extMem,
  const struct cudaExternalMemoryBufferDesc* bufferDesc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaExternalMemoryGetMappedMipmappedArray(
  cudaMipmappedArray_t* mipmap,
  cudaExternalMemory_t extMem,
  const struct cudaExternalMemoryMipmappedArrayDesc* mipmapDesc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDestroyExternalMemory(cudaExternalMemory_t extMem)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaImportExternalSemaphore(
  cudaExternalSemaphore_t* extSem_out,
  const struct cudaExternalSemaphoreHandleDesc* semHandleDesc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaSignalExternalSemaphoresAsync(
  const cudaExternalSemaphore_t* extSemArray,
  const struct cudaExternalSemaphoreSignalParams* paramsArray,
  unsigned int numExtSems,
  cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaWaitExternalSemaphoresAsync(
  const cudaExternalSemaphore_t* extSemArray,
  const struct cudaExternalSemaphoreWaitParams* paramsArray,
  unsigned int numExtSems,
  cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaLaunchCooperativeKernel(const void* func,
                            dim3 gridDim,
                            dim3 blockDim,
                            void** args,
                            size_t sharedMem,
                            cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaLaunchCooperativeKernelMultiDevice(
  struct cudaLaunchParams* launchParamsList,
  unsigned int numDevices,
  unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaFuncSetSharedMemConfig(const void* func, enum cudaSharedMemConfig config)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaSetDoubleForHost(double* d)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize,
                                          const void* func,
                                          int numBlocks,
                                          int blockSize)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMallocHost(void** ptr, size_t size)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMallocArray(cudaArray_t* array,
                const struct cudaChannelFormatDesc* desc,
                size_t width,
                size_t height,
                unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaFreeArray(cudaArray_t array)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaHostRegister(void* ptr, size_t size, unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaHostUnregister(void* ptr)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaHostGetFlags(unsigned int* pFlags, void* pHost)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMalloc3DArray(cudaArray_t* array,
                  const struct cudaChannelFormatDesc* desc,
                  struct cudaExtent extent,
                  unsigned int flags /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray,
                         const struct cudaChannelFormatDesc* desc,
                         struct cudaExtent extent,
                         unsigned int numLevels,
                         unsigned int flags /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetMipmappedArrayLevel(cudaArray_t* levelArray,
                           cudaMipmappedArray_const_t mipmappedArray,
                           unsigned int level)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy3D(const struct cudaMemcpy3DParms* p)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms* p)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p,
                  cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms* p,
                      cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaArrayGetInfo(struct cudaChannelFormatDesc* desc,
                 struct cudaExtent* extent,
                 unsigned int* flags,
                 cudaArray_t array)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaArrayGetPlane(cudaArray_t* pPlaneArray,
                  cudaArray_t hArray,
                  unsigned int planeIdx)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaArrayGetMemoryRequirements(
  struct cudaArrayMemoryRequirements* memoryRequirements,
  cudaArray_t array,
  int device)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMipmappedArrayGetMemoryRequirements(
  struct cudaArrayMemoryRequirements* memoryRequirements,
  cudaMipmappedArray_t mipmap,
  int device)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties,
                             cudaArray_t array)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMipmappedArrayGetSparseProperties(
  struct cudaArraySparseProperties* sparseProperties,
  cudaMipmappedArray_t mipmap)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyPeer(void* dst,
               int dstDevice,
               const void* src,
               int srcDevice,
               size_t count)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy2D(void* dst,
             size_t dpitch,
             const void* src,
             size_t spitch,
             size_t width,
             size_t height,
             enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy2DToArray(cudaArray_t dst,
                    size_t wOffset,
                    size_t hOffset,
                    const void* src,
                    size_t spitch,
                    size_t width,
                    size_t height,
                    enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy2DFromArray(void* dst,
                      size_t dpitch,
                      cudaArray_const_t src,
                      size_t wOffset,
                      size_t hOffset,
                      size_t width,
                      size_t height,
                      enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy2DArrayToArray(
  cudaArray_t dst,
  size_t wOffsetDst,
  size_t hOffsetDst,
  cudaArray_const_t src,
  size_t wOffsetSrc,
  size_t hOffsetSrc,
  size_t width,
  size_t height,
  enum cudaMemcpyKind kind /* = cudaMemcpyDeviceToDevice */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyToSymbol(const void* symbol,
                   const void* src,
                   size_t count,
                   size_t offset /* = 0 */,
                   enum cudaMemcpyKind kind /* = cudaMemcpyHostToDevice */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyFromSymbol(void* dst,
                     const void* symbol,
                     size_t count,
                     size_t offset /* = 0 */,
                     enum cudaMemcpyKind kind /* = cudaMemcpyDeviceToHost */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyPeerAsync(void* dst,
                    int dstDevice,
                    const void* src,
                    int srcDevice,
                    size_t count,
                    cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy2DAsync(void* dst,
                  size_t dpitch,
                  const void* src,
                  size_t spitch,
                  size_t width,
                  size_t height,
                  enum cudaMemcpyKind kind,
                  cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy2DToArrayAsync(cudaArray_t dst,
                         size_t wOffset,
                         size_t hOffset,
                         const void* src,
                         size_t spitch,
                         size_t width,
                         size_t height,
                         enum cudaMemcpyKind kind,
                         cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpy2DFromArrayAsync(void* dst,
                           size_t dpitch,
                           cudaArray_const_t src,
                           size_t wOffset,
                           size_t hOffset,
                           size_t width,
                           size_t height,
                           enum cudaMemcpyKind kind,
                           cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyToSymbolAsync(const void* symbol,
                        const void* src,
                        size_t count,
                        size_t offset,
                        enum cudaMemcpyKind kind,
                        cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyFromSymbolAsync(void* dst,
                          const void* symbol,
                          size_t count,
                          size_t offset,
                          enum cudaMemcpyKind kind,
                          cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr,
             int value,
             struct cudaExtent extent)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemset2DAsync(void* devPtr,
                  size_t pitch,
                  int value,
                  size_t width,
                  size_t height,
                  cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr,
                  int value,
                  struct cudaExtent extent,
                  cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetSymbolAddress(void** devPtr, const void* symbol)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetSymbolSize(size_t* size, const void* symbol)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPrefetchAsync(const void* devPtr,
                     size_t count,
                     int dstDevice,
                     cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemAdvise(const void* devPtr,
              size_t count,
              enum cudaMemoryAdvise advice,
              int device)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemRangeGetAttribute(void* data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void* devPtr,
                         size_t count)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemRangeGetAttributes(void** data,
                          size_t* dataSizes,
                          enum cudaMemRangeAttribute* attributes,
                          size_t numAttributes,
                          const void* devPtr,
                          size_t count)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyToArray(cudaArray_t dst,
                  size_t wOffset,
                  size_t hOffset,
                  const void* src,
                  size_t count,
                  enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyFromArray(void* dst,
                    cudaArray_const_t src,
                    size_t wOffset,
                    size_t hOffset,
                    size_t count,
                    enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyArrayToArray(cudaArray_t dst,
                       size_t wOffsetDst,
                       size_t hOffsetDst,
                       cudaArray_const_t src,
                       size_t wOffsetSrc,
                       size_t hOffsetSrc,
                       size_t count,
                       enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyToArrayAsync(cudaArray_t dst,
                       size_t wOffset,
                       size_t hOffset,
                       const void* src,
                       size_t count,
                       enum cudaMemcpyKind kind,
                       cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemcpyFromArrayAsync(void* dst,
                         cudaArray_const_t src,
                         size_t wOffset,
                         size_t hOffset,
                         size_t count,
                         enum cudaMemcpyKind kind,
                         cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaFreeAsync(void* devPtr, cudaStream_t hStream)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolSetAttribute(cudaMemPool_t memPool,
                        enum cudaMemPoolAttr attr,
                        void* value)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolGetAttribute(cudaMemPool_t memPool,
                        enum cudaMemPoolAttr attr,
                        void* value)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolSetAccess(cudaMemPool_t memPool,
                     const struct cudaMemAccessDesc* descList,
                     size_t count)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolGetAccess(enum cudaMemAccessFlags* flags,
                     cudaMemPool_t memPool,
                     struct cudaMemLocation* location)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolCreate(cudaMemPool_t* memPool,
                  const struct cudaMemPoolProps* poolProps)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolDestroy(cudaMemPool_t memPool)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMallocFromPoolAsync(void** ptr,
                        size_t size,
                        cudaMemPool_t memPool,
                        cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolExportToShareableHandle(void* shareableHandle,
                                   cudaMemPool_t memPool,
                                   enum cudaMemAllocationHandleType handleType,
                                   unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolImportFromShareableHandle(
  cudaMemPool_t* memPool,
  void* shareableHandle,
  enum cudaMemAllocationHandleType handleType,
  unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData* exportData, void* ptr)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaMemPoolImportPointer(void** ptr,
                         cudaMemPool_t memPool,
                         struct cudaMemPoolPtrExportData* exportData)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDeviceDisablePeerAccess(int peerDevice)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource,
                                unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphicsMapResources(int count,
                         cudaGraphicsResource_t* resources,
                         cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphicsUnmapResources(int count,
                           cudaGraphicsResource_t* resources,
                           cudaStream_t stream /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphicsResourceGetMappedPointer(void** devPtr,
                                     size_t* size,
                                     cudaGraphicsResource_t resource)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array,
                                      cudaGraphicsResource_t resource,
                                      unsigned int arrayIndex,
                                      unsigned int mipLevel)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphicsResourceGetMappedMipmappedArray(
  cudaMipmappedArray_t* mipmappedArray,
  cudaGraphicsResource_t resource)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaBindTexture(size_t* offset,
                const struct textureReference* texref,
                const void* devPtr,
                const struct cudaChannelFormatDesc* desc,
                size_t size /* = UINT_MAX */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaBindTexture2D(size_t* offset,
                  const struct textureReference* texref,
                  const void* devPtr,
                  const struct cudaChannelFormatDesc* desc,
                  size_t width,
                  size_t height,
                  size_t pitch)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaBindTextureToArray(const struct textureReference* texref,
                       cudaArray_const_t array,
                       const struct cudaChannelFormatDesc* desc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaBindTextureToMipmappedArray(const struct textureReference* texref,
                                cudaMipmappedArray_const_t mipmappedArray,
                                const struct cudaChannelFormatDesc* desc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaUnbindTexture(const struct textureReference* texref)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetTextureAlignmentOffset(size_t* offset,
                              const struct textureReference* texref)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetTextureReference(const struct textureReference** texref,
                        const void* symbol)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaBindSurfaceToArray(const struct surfaceReference* surfref,
                       cudaArray_const_t array,
                       const struct cudaChannelFormatDesc* desc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetSurfaceReference(const struct surfaceReference** surfref,
                        const void* symbol)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetChannelDesc(struct cudaChannelFormatDesc* desc, cudaArray_const_t array)
{
    NOT_IMPLEMENTED;
}

struct cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaCreateTextureObject(cudaTextureObject_t* pTexObject,
                        const struct cudaResourceDesc* pResDesc,
                        const struct cudaTextureDesc* pTexDesc,
                        const struct cudaResourceViewDesc* pResViewDesc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDestroyTextureObject(cudaTextureObject_t texObject)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetTextureObjectResourceDesc(struct cudaResourceDesc* pResDesc,
                                 cudaTextureObject_t texObject)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetTextureObjectTextureDesc(struct cudaTextureDesc* pTexDesc,
                                cudaTextureObject_t texObject)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc* pResViewDesc,
                                     cudaTextureObject_t texObject)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject,
                        const struct cudaResourceDesc* pResDesc)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc* pResDesc,
                                 cudaSurfaceObject_t surfObject)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaRuntimeGetVersion(int* runtimeVersion)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode,
                       cudaGraph_t graph,
                       const cudaGraphNode_t* pDependencies,
                       size_t numDependencies,
                       const struct cudaKernelNodeParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphKernelNodeGetParams(cudaGraphNode_t node,
                             struct cudaKernelNodeParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphKernelNodeSetParams(cudaGraphNode_t node,
                             const struct cudaKernelNodeParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode,
                                enum cudaKernelNodeAttrID attr,
                                union cudaKernelNodeAttrValue* value_out)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode,
                                enum cudaKernelNodeAttrID attr,
                                const union cudaKernelNodeAttrValue* value)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode,
                       cudaGraph_t graph,
                       const cudaGraphNode_t* pDependencies,
                       size_t numDependencies,
                       const struct cudaMemcpy3DParms* pCopyParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t* pGraphNode,
                               cudaGraph_t graph,
                               const cudaGraphNode_t* pDependencies,
                               size_t numDependencies,
                               const void* symbol,
                               const void* src,
                               size_t count,
                               size_t offset,
                               enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t* pGraphNode,
                                 cudaGraph_t graph,
                                 const cudaGraphNode_t* pDependencies,
                                 size_t numDependencies,
                                 void* dst,
                                 const void* symbol,
                                 size_t count,
                                 size_t offset,
                                 enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode,
                         cudaGraph_t graph,
                         const cudaGraphNode_t* pDependencies,
                         size_t numDependencies,
                         void* dst,
                         const void* src,
                         size_t count,
                         enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,
                             struct cudaMemcpy3DParms* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node,
                             const struct cudaMemcpy3DParms* pNodeParams)
{
    NOT_IMPLEMENTED;
}

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node,
                                     const void* symbol,
                                     const void* src,
                                     size_t count,
                                     size_t offset,
                                     enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node,
                                       void* dst,
                                       const void* symbol,
                                       size_t count,
                                       size_t offset,
                                       enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node,
                               void* dst,
                               const void* src,
                               size_t count,
                               enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}
#endif

cudaError_t
cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode,
                       cudaGraph_t graph,
                       const cudaGraphNode_t* pDependencies,
                       size_t numDependencies,
                       const struct cudaMemsetParams* pMemsetParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphMemsetNodeGetParams(cudaGraphNode_t node,
                             struct cudaMemsetParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphMemsetNodeSetParams(cudaGraphNode_t node,
                             const struct cudaMemsetParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode,
                     cudaGraph_t graph,
                     const cudaGraphNode_t* pDependencies,
                     size_t numDependencies,
                     const struct cudaHostNodeParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphHostNodeGetParams(cudaGraphNode_t node,
                           struct cudaHostNodeParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphHostNodeSetParams(cudaGraphNode_t node,
                           const struct cudaHostNodeParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode,
                           cudaGraph_t graph,
                           const cudaGraphNode_t* pDependencies,
                           size_t numDependencies,
                           cudaGraph_t childGraph)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode,
                      cudaGraph_t graph,
                      const cudaGraphNode_t* pDependencies,
                      size_t numDependencies)
{
    NOT_IMPLEMENTED;
}

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode,
                            cudaGraph_t graph,
                            const cudaGraphNode_t* pDependencies,
                            size_t numDependencies,
                            cudaEvent_t event)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode,
                          cudaGraph_t graph,
                          const cudaGraphNode_t* pDependencies,
                          size_t numDependencies,
                          cudaEvent_t event)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11020
cudaError_t
cudaGraphAddExternalSemaphoresSignalNode(
  cudaGraphNode_t* pGraphNode,
  cudaGraph_t graph,
  const cudaGraphNode_t* pDependencies,
  size_t numDependencies,
  const struct cudaExternalSemaphoreSignalNodeParams* nodeParams)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11020
cudaError_t
cudaGraphExternalSemaphoresSignalNodeGetParams(
  cudaGraphNode_t hNode,
  struct cudaExternalSemaphoreSignalNodeParams* params_out)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11020
cudaError_t
cudaGraphExternalSemaphoresSignalNodeSetParams(
  cudaGraphNode_t hNode,
  const struct cudaExternalSemaphoreSignalNodeParams* nodeParams)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11020
cudaError_t
cudaGraphAddExternalSemaphoresWaitNode(
  cudaGraphNode_t* pGraphNode,
  cudaGraph_t graph,
  const cudaGraphNode_t* pDependencies,
  size_t numDependencies,
  const struct cudaExternalSemaphoreWaitNodeParams* nodeParams)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11020
cudaError_t
cudaGraphExternalSemaphoresWaitNodeGetParams(
  cudaGraphNode_t hNode,
  struct cudaExternalSemaphoreWaitNodeParams* params_out)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11020
cudaError_t
cudaGraphExternalSemaphoresWaitNodeSetParams(
  cudaGraphNode_t hNode,
  const struct cudaExternalSemaphoreWaitNodeParams* nodeParams)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11040
cudaError_t
cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode,
                         cudaGraph_t graph,
                         const cudaGraphNode_t* pDependencies,
                         size_t numDependencies,
                         struct cudaMemAllocNodeParams* nodeParams)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11040
cudaError_t
cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node,
                               struct cudaMemAllocNodeParams* params_out)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11040
cudaError_t
cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode,
                        cudaGraph_t graph,
                        const cudaGraphNode_t* pDependencies,
                        size_t numDependencies,
                        void* dptr)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11040
cudaError_t
cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11040
cudaError_t
cudaDeviceGraphMemTrim(int device)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11040
cudaError_t
cudaDeviceGetGraphMemAttribute(int device,
                               enum cudaGraphMemAttributeType attr,
                               void* value)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11040
cudaError_t
cudaDeviceSetGraphMemAttribute(int device,
                               enum cudaGraphMemAttributeType attr,
                               void* value)
{
    NOT_IMPLEMENTED;
}
#endif

cudaError_t
cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphNodeFindInClone(cudaGraphNode_t* pNode,
                         cudaGraphNode_t originalNode,
                         cudaGraph_t clonedGraph)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType* pType)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphGetRootNodes(cudaGraph_t graph,
                      cudaGraphNode_t* pRootNodes,
                      size_t* pNumRootNodes)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphGetEdges(cudaGraph_t graph,
                  cudaGraphNode_t* from,
                  cudaGraphNode_t* to,
                  size_t* numEdges)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphNodeGetDependencies(cudaGraphNode_t node,
                             cudaGraphNode_t* pDependencies,
                             size_t* pNumDependencies)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
                               cudaGraphNode_t* pDependentNodes,
                               size_t* pNumDependentNodes)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphAddDependencies(cudaGraph_t graph,
                         const cudaGraphNode_t* from,
                         const cudaGraphNode_t* to,
                         size_t numDependencies)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphRemoveDependencies(cudaGraph_t graph,
                            const cudaGraphNode_t* from,
                            const cudaGraphNode_t* to,
                            size_t numDependencies)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphDestroyNode(cudaGraphNode_t node)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                     cudaGraph_t graph,
                     cudaGraphNode_t* pErrorNode,
                     char* pLogBuffer,
                     size_t bufferSize)
{
    NOT_IMPLEMENTED;
}

#if __CUDART_API_VERSION >= 11040
cudaError_t
cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec,
                              cudaGraph_t graph,
                              unsigned long long flags)
{
    NOT_IMPLEMENTED;
}
#endif

cudaError_t
cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec,
                                 cudaGraphNode_t node,
                                 const struct cudaKernelNodeParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec,
                                 cudaGraphNode_t node,
                                 const struct cudaMemcpy3DParms* pNodeParams)
{
    NOT_IMPLEMENTED;
}

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec,
                                         cudaGraphNode_t node,
                                         const void* symbol,
                                         const void* src,
                                         size_t count,
                                         size_t offset,
                                         enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec,
                                           cudaGraphNode_t node,
                                           void* dst,
                                           const void* symbol,
                                           size_t count,
                                           size_t offset,
                                           enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec,
                                   cudaGraphNode_t node,
                                   void* dst,
                                   const void* src,
                                   size_t count,
                                   enum cudaMemcpyKind kind)
{
    NOT_IMPLEMENTED;
}
#endif

cudaError_t
cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec,
                                 cudaGraphNode_t node,
                                 const struct cudaMemsetParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec,
                               cudaGraphNode_t node,
                               const struct cudaHostNodeParams* pNodeParams)
{
    NOT_IMPLEMENTED;
}

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec,
                                     cudaGraphNode_t node,
                                     cudaGraph_t childGraph)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec,
                                     cudaGraphNode_t hNode,
                                     cudaEvent_t event)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec,
                                   cudaGraphNode_t hNode,
                                   cudaEvent_t event)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11020
cudaError_t
cudaGraphExecExternalSemaphoresSignalNodeSetParams(
  cudaGraphExec_t hGraphExec,
  cudaGraphNode_t hNode,
  const struct cudaExternalSemaphoreSignalNodeParams* nodeParams)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11020
cudaError_t
cudaGraphExecExternalSemaphoresWaitNodeSetParams(
  cudaGraphExec_t hGraphExec,
  cudaGraphNode_t hNode,
  const struct cudaExternalSemaphoreWaitNodeParams* nodeParams)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11060
cudaError_t
cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec,
                        cudaGraphNode_t hNode,
                        unsigned int isEnabled)
{
    NOT_IMPLEMENTED;
}
#endif

#if __CUDART_API_VERSION >= 11060
cudaError_t
cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec,
                        cudaGraphNode_t hNode,
                        unsigned int* isEnabled)
{
    NOT_IMPLEMENTED;
}
#endif

cudaError_t
cudaGraphExecUpdate(cudaGraphExec_t hGraphExec,
                    cudaGraph_t hGraph,
                    cudaGraphNode_t* hErrorNode_out,
                    enum cudaGraphExecUpdateResult* updateResult_out)
{
    NOT_IMPLEMENTED;
}

#if __CUDART_API_VERSION >= 11010
cudaError_t
cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}
#endif

cudaError_t
cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphDestroy(cudaGraph_t graph)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaUserObjectCreate(cudaUserObject_t* object_out,
                     void* ptr,
                     cudaHostFn_t destroy,
                     unsigned int initialRefcount,
                     unsigned int flags)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaUserObjectRetain(cudaUserObject_t object, unsigned int count /* = 1 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaUserObjectRelease(cudaUserObject_t object, unsigned int count /* = 1 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphRetainUserObject(cudaGraph_t graph,
                          cudaUserObject_t object,
                          unsigned int count /* = 1 */,
                          unsigned int flags /* = 0 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGraphReleaseUserObject(cudaGraph_t graph,
                           cudaUserObject_t object,
                           unsigned int count /* = 1 */)
{
    NOT_IMPLEMENTED;
}

cudaError_t
cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId)
{
    NOT_IMPLEMENTED;
}

cudaError_t CUDARTAPI_CDECL
cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbolPtr)
{
    NOT_IMPLEMENTED;
}
