#define CUBLASAPI

#include "common.h"
#include <cublasLt.h>
#include <cublas_api.h>

cublasStatus_t
cublasCreate_v2(cublasHandle_t* handle)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasSetWorkspace_v2(cublasHandle_t handle,
                      void* workspace,
                      size_t workspaceSizeInBytes)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode)
{
    *mode = CUBLAS_DEFAULT_MATH;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
    return CUBLAS_STATUS_SUCCESS;
}

/* ---------------- CUBLAS BLAS3 functions ---------------- */

/* GEMM */
cublasStatus_t
cublasSgemm_v2(cublasHandle_t handle,
               cublasOperation_t transa,
               cublasOperation_t transb,
               int m,
               int n,
               int k,
               const float* alpha,
               const float* A,
               int lda,
               const float* B,
               int ldb,
               const float* beta,
               float* C,
               int ldc)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasGemmEx(cublasHandle_t handle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const void* alpha,
             const void* A,
             cudaDataType Atype,
             int lda,
             const void* B,
             cudaDataType Btype,
             int ldb,
             const void* beta,
             void* C,
             cudaDataType Ctype,
             int ldc,
             cublasComputeType_t computeType,
             cublasGemmAlgo_t algo)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasGemmStridedBatchedEx(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const void* alpha,
                           const void* A,
                           cudaDataType Atype,
                           int lda,
                           long long int strideA, /* purposely signed */
                           const void* B,
                           cudaDataType Btype,
                           int ldb,
                           long long int strideB,
                           const void* beta,
                           void* C,
                           cudaDataType Ctype,
                           int ldc,
                           long long int strideC,
                           int batchCount,
                           cublasComputeType_t computeType,
                           cublasGemmAlgo_t algo)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasSgemmStridedBatched(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const float* alpha,
                          const float* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const float* B,
                          int ldb,
                          long long int strideB,
                          const float* beta,
                          float* C,
                          int ldc,
                          long long int strideC,
                          int batchCount)
{
    return CUBLAS_STATUS_SUCCESS;
}

/** Execute matrix multiplication (D = alpha * op(A) * op(B) + beta * C).
 *
 * \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been
 * initialized \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in
 * conflict or in an impossible configuration; e.g. when workspaceSizeInBytes is
 * less than workspace required by configured algo \retval
 * CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device
 * doesn't support configured operation \retval     CUBLAS_STATUS_ARCH_MISMATCH
 * if configured operation cannot be run using selected device \retval
 * CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the
 * device \retval     CUBLAS_STATUS_SUCCESS           if the operation completed
 * successfully
 */
cublasStatus_t
cublasLtMatmul(cublasLtHandle_t lightHandle,
               cublasLtMatmulDesc_t computeDesc,
               const void* alpha, /* host or device pointer */
               const void* A,
               cublasLtMatrixLayout_t Adesc,
               const void* B,
               cublasLtMatrixLayout_t Bdesc,
               const void* beta, /* host or device pointer */
               const void* C,
               cublasLtMatrixLayout_t Cdesc,
               void* D,
               cublasLtMatrixLayout_t Ddesc,
               const cublasLtMatmulAlgo_t* algo,
               void* workspace,
               size_t workspaceSizeInBytes,
               cudaStream_t stream)
{
    return CUBLAS_STATUS_SUCCESS;
}

/** Create new matrix layout descriptor.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
cublasStatus_t
cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout,
                           cudaDataType type,
                           uint64_t rows,
                           uint64_t cols,
                           int64_t ld)
{
    return CUBLAS_STATUS_SUCCESS;
}

/** Create new matmul operation descriptor.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
cublasStatus_t
cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                         cublasComputeType_t computeType,
                         cudaDataType_t scaleType)
{
    return CUBLAS_STATUS_SUCCESS;
}

/** Set matmul operation descriptor attribute.
 *
 * \param[in]  matmulDesc   The descriptor
 * \param[in]  attr         The attribute
 * \param[in]  buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t
cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                               cublasLtMatmulDescAttributes_t attr,
                               const void* buf,
                               size_t sizeInBytes)
{
    return CUBLAS_STATUS_SUCCESS;
}

/** Create new matmul heuristic search preference descriptor.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
cublasStatus_t
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref)
{
    return CUBLAS_STATUS_SUCCESS;
}

/** Set matmul heuristic search preference descriptor attribute.
 *
 * \param[in]  pref         The descriptor
 * \param[in]  attr         The attribute
 * \param[in]  buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t
cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                     cublasLtMatmulPreferenceAttributes_t attr,
                                     const void* buf,
                                     size_t sizeInBytes)
{
    return CUBLAS_STATUS_SUCCESS;
}

/** Query cublasLt heuristic for algorithm appropriate for given use case.
 *
 * \param[in]      lightHandle            Pointer to the allocated cuBLASLt
 * handle for the cuBLASLt context. See cublasLtHandle_t. \param[in]
 * operationDesc          Handle to the matrix multiplication descriptor.
 * \param[in]      Adesc                  Handle to the layout descriptors for
 * matrix A. \param[in]      Bdesc                  Handle to the layout
 * descriptors for matrix B. \param[in]      Cdesc                  Handle to
 * the layout descriptors for matrix C. \param[in]      Ddesc Handle to the
 * layout descriptors for matrix D. \param[in]      preference Pointer to the
 * structure holding the heuristic search preferences descriptor. See
 * cublasLtMatrixLayout_t. \param[in]      requestedAlgoCount     Size of
 * heuristicResultsArray (in elements) and requested maximum number of
 * algorithms to return. \param[in, out] heuristicResultsArray  Output
 * algorithms and associated runtime characteristics, ordered in increasing
 * estimated compute time. \param[out]     returnAlgoCount        The number of
 * heuristicResultsArray elements written.
 *
 * \retval  CUBLAS_STATUS_INVALID_VALUE   if requestedAlgoCount is less or equal
 * to zero \retval  CUBLAS_STATUS_NOT_SUPPORTED   if no heuristic function
 * available for current configuration \retval  CUBLAS_STATUS_SUCCESS         if
 * query was successful, inspect heuristicResultsArray[0 to (returnAlgoCount -
 * 1)].state for detail status of results
 */
cublasStatus_t
cublasLtMatmulAlgoGetHeuristic(
  cublasLtHandle_t lightHandle,
  cublasLtMatmulDesc_t operationDesc,
  cublasLtMatrixLayout_t Adesc,
  cublasLtMatrixLayout_t Bdesc,
  cublasLtMatrixLayout_t Cdesc,
  cublasLtMatrixLayout_t Ddesc,
  cublasLtMatmulPreference_t preference,
  int requestedAlgoCount,
  cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
  int* returnAlgoCount)
{
    *returnAlgoCount = requestedAlgoCount;
    return CUBLAS_STATUS_SUCCESS;
}
