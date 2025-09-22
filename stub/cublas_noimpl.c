#define CUBLASAPI

#include "common.h"
#include <cublasLt.h>
#include <cublas_api.h>

cublasStatus_t
cublasDestroy_v2(cublasHandle_t handle)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetVersion_v2(cublasHandle_t handle, int* version)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetProperty(libraryPropertyType type, int* value)
{
    NOT_IMPLEMENTED;
}

size_t
cublasGetCudartVersion(void)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetSmCountTarget(cublasHandle_t handle, int* smCountTarget)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget)
{
    NOT_IMPLEMENTED;
}

const char*
cublasGetStatusName(cublasStatus_t status)
{
    NOT_IMPLEMENTED;
}

const char*
cublasGetStatusString(cublasStatus_t status)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasLoggerConfigure(int logIsOn,
                      int logToStdOut,
                      int logToStdErr,
                      const char* logFileName)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSetLoggerCallback(cublasLogCallback userCallback)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetLoggerCallback(cublasLogCallback* userCallback)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSetVector(int n,
                int elemSize,
                const void* x,
                int incx,
                void* devicePtr,
                int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSetMatrix(int rows,
                int cols,
                int elemSize,
                const void* A,
                int lda,
                void* B,
                int ldb)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetMatrix(int rows,
                int cols,
                int elemSize,
                const void* A,
                int lda,
                void* B,
                int ldb)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSetVectorAsync(int n,
                     int elemSize,
                     const void* hostPtr,
                     int incx,
                     void* devicePtr,
                     int incy,
                     cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetVectorAsync(int n,
                     int elemSize,
                     const void* devicePtr,
                     int incx,
                     void* hostPtr,
                     int incy,
                     cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSetMatrixAsync(int rows,
                     int cols,
                     int elemSize,
                     const void* A,
                     int lda,
                     void* B,
                     int ldb,
                     cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGetMatrixAsync(int rows,
                     int cols,
                     int elemSize,
                     const void* A,
                     int lda,
                     void* B,
                     int ldb,
                     cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

void
cublasXerbla(const char* srName, int info)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasNrm2Ex(cublasHandle_t handle,
             int n,
             const void* x,
             cudaDataType xType,
             int incx,
             void* result,
             cudaDataType resultType,
             cudaDataType executionType)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSnrm2_v2(cublasHandle_t handle,
               int n,
               const float* x,
               int incx,
               float* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDnrm2_v2(cublasHandle_t handle,
               int n,
               const double* x,
               int incx,
               double* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasScnrm2_v2(cublasHandle_t handle,
                int n,
                const cuComplex* x,
                int incx,
                float* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDznrm2_v2(cublasHandle_t handle,
                int n,
                const cuDoubleComplex* x,
                int incx,
                double* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDotEx(cublasHandle_t handle,
            int n,
            const void* x,
            cudaDataType xType,
            int incx,
            const void* y,
            cudaDataType yType,
            int incy,
            void* result,
            cudaDataType resultType,
            cudaDataType executionType)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDotcEx(cublasHandle_t handle,
             int n,
             const void* x,
             cudaDataType xType,
             int incx,
             const void* y,
             cudaDataType yType,
             int incy,
             void* result,
             cudaDataType resultType,
             cudaDataType executionType)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSdot_v2(cublasHandle_t handle,
              int n,
              const float* x,
              int incx,
              const float* y,
              int incy,
              float* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDdot_v2(cublasHandle_t handle,
              int n,
              const double* x,
              int incx,
              const double* y,
              int incy,
              double* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCdotu_v2(cublasHandle_t handle,
               int n,
               const cuComplex* x,
               int incx,
               const cuComplex* y,
               int incy,
               cuComplex* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCdotc_v2(cublasHandle_t handle,
               int n,
               const cuComplex* x,
               int incx,
               const cuComplex* y,
               int incy,
               cuComplex* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZdotu_v2(cublasHandle_t handle,
               int n,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* y,
               int incy,
               cuDoubleComplex* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZdotc_v2(cublasHandle_t handle,
               int n,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* y,
               int incy,
               cuDoubleComplex* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasScalEx(cublasHandle_t handle,
             int n,
             const void* alpha,
             cudaDataType alphaType,
             void* x,
             cudaDataType xType,
             int incx,
             cudaDataType executionType)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSscal_v2(cublasHandle_t handle,
               int n,
               const float* alpha,
               float* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDscal_v2(cublasHandle_t handle,
               int n,
               const double* alpha,
               double* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCscal_v2(cublasHandle_t handle,
               int n,
               const cuComplex* alpha,
               cuComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsscal_v2(cublasHandle_t handle,
                int n,
                const float* alpha,
                cuComplex* x,
                int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZscal_v2(cublasHandle_t handle,
               int n,
               const cuDoubleComplex* alpha,
               cuDoubleComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZdscal_v2(cublasHandle_t handle,
                int n,
                const double* alpha,
                cuDoubleComplex* x,
                int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasAxpyEx(cublasHandle_t handle,
             int n,
             const void* alpha,
             cudaDataType alphaType,
             const void* x,
             cudaDataType xType,
             int incx,
             void* y,
             cudaDataType yType,
             int incy,
             cudaDataType executiontype)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSaxpy_v2(cublasHandle_t handle,
               int n,
               const float* alpha,
               const float* x,
               int incx,
               float* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDaxpy_v2(cublasHandle_t handle,
               int n,
               const double* alpha,
               const double* x,
               int incx,
               double* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCaxpy_v2(cublasHandle_t handle,
               int n,
               const cuComplex* alpha,
               const cuComplex* x,
               int incx,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZaxpy_v2(cublasHandle_t handle,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* x,
               int incx,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCopyEx(cublasHandle_t handle,
             int n,
             const void* x,
             cudaDataType xType,
             int incx,
             void* y,
             cudaDataType yType,
             int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasScopy_v2(cublasHandle_t handle,
               int n,
               const float* x,
               int incx,
               float* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDcopy_v2(cublasHandle_t handle,
               int n,
               const double* x,
               int incx,
               double* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCcopy_v2(cublasHandle_t handle,
               int n,
               const cuComplex* x,
               int incx,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZcopy_v2(cublasHandle_t handle,
               int n,
               const cuDoubleComplex* x,
               int incx,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSswap_v2(cublasHandle_t handle,
               int n,
               float* x,
               int incx,
               float* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDswap_v2(cublasHandle_t handle,
               int n,
               double* x,
               int incx,
               double* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCswap_v2(cublasHandle_t handle,
               int n,
               cuComplex* x,
               int incx,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZswap_v2(cublasHandle_t handle,
               int n,
               cuDoubleComplex* x,
               int incx,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSwapEx(cublasHandle_t handle,
             int n,
             void* x,
             cudaDataType xType,
             int incx,
             void* y,
             cudaDataType yType,
             int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIsamax_v2(cublasHandle_t handle,
                int n,
                const float* x,
                int incx,
                int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIdamax_v2(cublasHandle_t handle,
                int n,
                const double* x,
                int incx,
                int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIcamax_v2(cublasHandle_t handle,
                int n,
                const cuComplex* x,
                int incx,
                int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIzamax_v2(cublasHandle_t handle,
                int n,
                const cuDoubleComplex* x,
                int incx,
                int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIamaxEx(cublasHandle_t handle,
              int n,
              const void* x,
              cudaDataType xType,
              int incx,
              int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIsamin_v2(cublasHandle_t handle,
                int n,
                const float* x,
                int incx,
                int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIdamin_v2(cublasHandle_t handle,
                int n,
                const double* x,
                int incx,
                int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIcamin_v2(cublasHandle_t handle,
                int n,
                const cuComplex* x,
                int incx,
                int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIzamin_v2(cublasHandle_t handle,
                int n,
                const cuDoubleComplex* x,
                int incx,
                int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasIaminEx(cublasHandle_t handle,
              int n,
              const void* x,
              cudaDataType xType,
              int incx,
              int* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasAsumEx(cublasHandle_t handle,
             int n,
             const void* x,
             cudaDataType xType,
             int incx,
             void* result,
             cudaDataType resultType,
             cudaDataType executiontype)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSasum_v2(cublasHandle_t handle,
               int n,
               const float* x,
               int incx,
               float* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDasum_v2(cublasHandle_t handle,
               int n,
               const double* x,
               int incx,
               double* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasScasum_v2(cublasHandle_t handle,
                int n,
                const cuComplex* x,
                int incx,
                float* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDzasum_v2(cublasHandle_t handle,
                int n,
                const cuDoubleComplex* x,
                int incx,
                double* result)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSrot_v2(cublasHandle_t handle,
              int n,
              float* x,
              int incx,
              float* y,
              int incy,
              const float* c,
              const float* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDrot_v2(cublasHandle_t handle,
              int n,
              double* x,
              int incx,
              double* y,
              int incy,
              const double* c,
              const double* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCrot_v2(cublasHandle_t handle,
              int n,
              cuComplex* x,
              int incx,
              cuComplex* y,
              int incy,
              const float* c,
              const cuComplex* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsrot_v2(cublasHandle_t handle,
               int n,
               cuComplex* x,
               int incx,
               cuComplex* y,
               int incy,
               const float* c,
               const float* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZrot_v2(cublasHandle_t handle,
              int n,
              cuDoubleComplex* x,
              int incx,
              cuDoubleComplex* y,
              int incy,
              const double* c,
              const cuDoubleComplex* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZdrot_v2(cublasHandle_t handle,
               int n,
               cuDoubleComplex* x,
               int incx,
               cuDoubleComplex* y,
               int incy,
               const double* c,
               const double* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasRotEx(cublasHandle_t handle,
            int n,
            void* x,
            cudaDataType xType,
            int incx,
            void* y,
            cudaDataType yType,
            int incy,
            const void* c,
            const void* s,
            cudaDataType csType,
            cudaDataType executiontype)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSrotg_v2(cublasHandle_t handle, float* a, float* b, float* c, float* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDrotg_v2(cublasHandle_t handle,
               double* a,
               double* b,
               double* c,
               double* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCrotg_v2(cublasHandle_t handle,
               cuComplex* a,
               cuComplex* b,
               float* c,
               cuComplex* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZrotg_v2(cublasHandle_t handle,
               cuDoubleComplex* a,
               cuDoubleComplex* b,
               double* c,
               cuDoubleComplex* s)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasRotgEx(cublasHandle_t handle,
             void* a,
             void* b,
             cudaDataType abType,
             void* c,
             void* s,
             cudaDataType csType,
             cudaDataType executiontype)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSrotm_v2(cublasHandle_t handle,
               int n,
               float* x,
               int incx,
               float* y,
               int incy,
               const float* param)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDrotm_v2(cublasHandle_t handle,
               int n,
               double* x,
               int incx,
               double* y,
               int incy,
               const double* param)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasRotmEx(cublasHandle_t handle,
             int n,
             void* x,
             cudaDataType xType,
             int incx,
             void* y,
             cudaDataType yType,
             int incy,
             const void* param,
             cudaDataType paramType,
             cudaDataType executiontype)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSrotmg_v2(cublasHandle_t handle,
                float* d1,
                float* d2,
                float* x1,
                const float* y1,
                float* param)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDrotmg_v2(cublasHandle_t handle,
                double* d1,
                double* d2,
                double* x1,
                const double* y1,
                double* param)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasRotmgEx(cublasHandle_t handle,
              void* d1,
              cudaDataType d1Type,
              void* d2,
              cudaDataType d2Type,
              void* x1,
              cudaDataType x1Type,
              const void* y1,
              cudaDataType y1Type,
              void* param,
              cudaDataType paramType,
              cudaDataType executiontype)
{
    NOT_IMPLEMENTED;
}
/* --------------- CUBLAS BLAS2 functions  ---------------- */

/* GEMV */
cublasStatus_t
cublasSgemv_v2(cublasHandle_t handle,
               cublasOperation_t trans,
               int m,
               int n,
               const float* alpha,
               const float* A,
               int lda,
               const float* x,
               int incx,
               const float* beta,
               float* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgemv_v2(cublasHandle_t handle,
               cublasOperation_t trans,
               int m,
               int n,
               const double* alpha,
               const double* A,
               int lda,
               const double* x,
               int incx,
               const double* beta,
               double* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemv_v2(cublasHandle_t handle,
               cublasOperation_t trans,
               int m,
               int n,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* x,
               int incx,
               const cuComplex* beta,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgemv_v2(cublasHandle_t handle,
               cublasOperation_t trans,
               int m,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* beta,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}
/* GBMV */
cublasStatus_t
cublasSgbmv_v2(cublasHandle_t handle,
               cublasOperation_t trans,
               int m,
               int n,
               int kl,
               int ku,
               const float* alpha,
               const float* A,
               int lda,
               const float* x,
               int incx,
               const float* beta,
               float* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgbmv_v2(cublasHandle_t handle,
               cublasOperation_t trans,
               int m,
               int n,
               int kl,
               int ku,
               const double* alpha,
               const double* A,
               int lda,
               const double* x,
               int incx,
               const double* beta,
               double* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgbmv_v2(cublasHandle_t handle,
               cublasOperation_t trans,
               int m,
               int n,
               int kl,
               int ku,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* x,
               int incx,
               const cuComplex* beta,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgbmv_v2(cublasHandle_t handle,
               cublasOperation_t trans,
               int m,
               int n,
               int kl,
               int ku,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* beta,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

/* TRMV */
cublasStatus_t
cublasStrmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const float* A,
               int lda,
               float* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtrmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const double* A,
               int lda,
               double* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtrmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const cuComplex* A,
               int lda,
               cuComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtrmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const cuDoubleComplex* A,
               int lda,
               cuDoubleComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

/* TBMV */
cublasStatus_t
cublasStbmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               int k,
               const float* A,
               int lda,
               float* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtbmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               int k,
               const double* A,
               int lda,
               double* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtbmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               int k,
               const cuComplex* A,
               int lda,
               cuComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtbmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               int k,
               const cuDoubleComplex* A,
               int lda,
               cuDoubleComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

/* TPMV */
cublasStatus_t
cublasStpmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const float* AP,
               float* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtpmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const double* AP,
               double* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtpmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const cuComplex* AP,
               cuComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtpmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const cuDoubleComplex* AP,
               cuDoubleComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

/* TRSV */
cublasStatus_t
cublasStrsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const float* A,
               int lda,
               float* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtrsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const double* A,
               int lda,
               double* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtrsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const cuComplex* A,
               int lda,
               cuComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtrsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const cuDoubleComplex* A,
               int lda,
               cuDoubleComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

/* TPSV */
cublasStatus_t
cublasStpsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const float* AP,
               float* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtpsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const double* AP,
               double* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtpsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const cuComplex* AP,
               cuComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtpsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               const cuDoubleComplex* AP,
               cuDoubleComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}
/* TBSV */
cublasStatus_t
cublasStbsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               int k,
               const float* A,
               int lda,
               float* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtbsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               int k,
               const double* A,
               int lda,
               double* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtbsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               int k,
               const cuComplex* A,
               int lda,
               cuComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtbsv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int n,
               int k,
               const cuDoubleComplex* A,
               int lda,
               cuDoubleComplex* x,
               int incx)
{
    NOT_IMPLEMENTED;
}

/* SYMV/HEMV */
cublasStatus_t
cublasSsymv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const float* alpha,
               const float* A,
               int lda,
               const float* x,
               int incx,
               const float* beta,
               float* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDsymv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const double* alpha,
               const double* A,
               int lda,
               const double* x,
               int incx,
               const double* beta,
               double* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsymv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* x,
               int incx,
               const cuComplex* beta,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZsymv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* beta,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasChemv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* x,
               int incx,
               const cuComplex* beta,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZhemv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* beta,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

/* SBMV/HBMV */
cublasStatus_t
cublasSsbmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               int k,
               const float* alpha,
               const float* A,
               int lda,
               const float* x,
               int incx,
               const float* beta,
               float* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDsbmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               int k,
               const double* alpha,
               const double* A,
               int lda,
               const double* x,
               int incx,
               const double* beta,
               double* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasChbmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               int k,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* x,
               int incx,
               const cuComplex* beta,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZhbmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               int k,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* beta,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

/* SPMV/HPMV */
cublasStatus_t
cublasSspmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const float* alpha,
               const float* AP,
               const float* x,
               int incx,
               const float* beta,
               float* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDspmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const double* alpha,
               const double* AP,
               const double* x,
               int incx,
               const double* beta,
               double* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasChpmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuComplex* alpha,
               const cuComplex* AP,
               const cuComplex* x,
               int incx,
               const cuComplex* beta,
               cuComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZhpmv_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* AP,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* beta,
               cuDoubleComplex* y,
               int incy)
{
    NOT_IMPLEMENTED;
}

/* GER */
cublasStatus_t
cublasSger_v2(cublasHandle_t handle,
              int m,
              int n,
              const float* alpha,
              const float* x,
              int incx,
              const float* y,
              int incy,
              float* A,
              int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDger_v2(cublasHandle_t handle,
              int m,
              int n,
              const double* alpha,
              const double* x,
              int incx,
              const double* y,
              int incy,
              double* A,
              int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgeru_v2(cublasHandle_t handle,
               int m,
               int n,
               const cuComplex* alpha,
               const cuComplex* x,
               int incx,
               const cuComplex* y,
               int incy,
               cuComplex* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgerc_v2(cublasHandle_t handle,
               int m,
               int n,
               const cuComplex* alpha,
               const cuComplex* x,
               int incx,
               const cuComplex* y,
               int incy,
               cuComplex* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgeru_v2(cublasHandle_t handle,
               int m,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* y,
               int incy,
               cuDoubleComplex* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgerc_v2(cublasHandle_t handle,
               int m,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* y,
               int incy,
               cuDoubleComplex* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

/* SYR/HER */
cublasStatus_t
cublasSsyr_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const float* alpha,
              const float* x,
              int incx,
              float* A,
              int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDsyr_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const double* alpha,
              const double* x,
              int incx,
              double* A,
              int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsyr_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const cuComplex* alpha,
              const cuComplex* x,
              int incx,
              cuComplex* A,
              int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZsyr_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const cuDoubleComplex* alpha,
              const cuDoubleComplex* x,
              int incx,
              cuDoubleComplex* A,
              int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCher_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const float* alpha,
              const cuComplex* x,
              int incx,
              cuComplex* A,
              int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZher_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const double* alpha,
              const cuDoubleComplex* x,
              int incx,
              cuDoubleComplex* A,
              int lda)
{
    NOT_IMPLEMENTED;
}

/* SPR/HPR */
cublasStatus_t
cublasSspr_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const float* alpha,
              const float* x,
              int incx,
              float* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDspr_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const double* alpha,
              const double* x,
              int incx,
              double* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasChpr_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const float* alpha,
              const cuComplex* x,
              int incx,
              cuComplex* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZhpr_v2(cublasHandle_t handle,
              cublasFillMode_t uplo,
              int n,
              const double* alpha,
              const cuDoubleComplex* x,
              int incx,
              cuDoubleComplex* AP)
{
    NOT_IMPLEMENTED;
}

/* SYR2/HER2 */
cublasStatus_t
cublasSsyr2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const float* alpha,
               const float* x,
               int incx,
               const float* y,
               int incy,
               float* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDsyr2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const double* alpha,
               const double* x,
               int incx,
               const double* y,
               int incy,
               double* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsyr2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuComplex* alpha,
               const cuComplex* x,
               int incx,
               const cuComplex* y,
               int incy,
               cuComplex* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZsyr2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* y,
               int incy,
               cuDoubleComplex* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCher2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuComplex* alpha,
               const cuComplex* x,
               int incx,
               const cuComplex* y,
               int incy,
               cuComplex* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZher2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* y,
               int incy,
               cuDoubleComplex* A,
               int lda)
{
    NOT_IMPLEMENTED;
}

/* SPR2/HPR2 */
cublasStatus_t
cublasSspr2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const float* alpha,
               const float* x,
               int incx,
               const float* y,
               int incy,
               float* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDspr2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const double* alpha,
               const double* x,
               int incx,
               const double* y,
               int incy,
               double* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasChpr2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuComplex* alpha,
               const cuComplex* x,
               int incx,
               const cuComplex* y,
               int incy,
               cuComplex* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZhpr2_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* x,
               int incx,
               const cuDoubleComplex* y,
               int incy,
               cuDoubleComplex* AP)
{
    NOT_IMPLEMENTED;
}
/* BATCH GEMV */
cublasStatus_t
cublasSgemvBatched(cublasHandle_t handle,
                   cublasOperation_t trans,
                   int m,
                   int n,
                   const float* alpha,
                   const float* const Aarray[],
                   int lda,
                   const float* const xarray[],
                   int incx,
                   const float* beta,
                   float* const yarray[],
                   int incy,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgemvBatched(cublasHandle_t handle,
                   cublasOperation_t trans,
                   int m,
                   int n,
                   const double* alpha,
                   const double* const Aarray[],
                   int lda,
                   const double* const xarray[],
                   int incx,
                   const double* beta,
                   double* const yarray[],
                   int incy,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemvBatched(cublasHandle_t handle,
                   cublasOperation_t trans,
                   int m,
                   int n,
                   const cuComplex* alpha,
                   const cuComplex* const Aarray[],
                   int lda,
                   const cuComplex* const xarray[],
                   int incx,
                   const cuComplex* beta,
                   cuComplex* const yarray[],
                   int incy,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgemvBatched(cublasHandle_t handle,
                   cublasOperation_t trans,
                   int m,
                   int n,
                   const cuDoubleComplex* alpha,
                   const cuDoubleComplex* const Aarray[],
                   int lda,
                   const cuDoubleComplex* const xarray[],
                   int incx,
                   const cuDoubleComplex* beta,
                   cuDoubleComplex* const yarray[],
                   int incy,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSgemvStridedBatched(cublasHandle_t handle,
                          cublasOperation_t trans,
                          int m,
                          int n,
                          const float* alpha,
                          const float* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const float* x,
                          int incx,
                          long long int stridex,
                          const float* beta,
                          float* y,
                          int incy,
                          long long int stridey,
                          int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgemvStridedBatched(cublasHandle_t handle,
                          cublasOperation_t trans,
                          int m,
                          int n,
                          const double* alpha,
                          const double* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const double* x,
                          int incx,
                          long long int stridex,
                          const double* beta,
                          double* y,
                          int incy,
                          long long int stridey,
                          int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemvStridedBatched(cublasHandle_t handle,
                          cublasOperation_t trans,
                          int m,
                          int n,
                          const cuComplex* alpha,
                          const cuComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuComplex* x,
                          int incx,
                          long long int stridex,
                          const cuComplex* beta,
                          cuComplex* y,
                          int incy,
                          long long int stridey,
                          int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgemvStridedBatched(cublasHandle_t handle,
                          cublasOperation_t trans,
                          int m,
                          int n,
                          const cuDoubleComplex* alpha,
                          const cuDoubleComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuDoubleComplex* x,
                          int incx,
                          long long int stridex,
                          const cuDoubleComplex* beta,
                          cuDoubleComplex* y,
                          int incy,
                          long long int stridey,
                          int batchCount)
{
    NOT_IMPLEMENTED;
}

/* ---------------- CUBLAS BLAS3 functions ---------------- */

cublasStatus_t
cublasDgemm_v2(cublasHandle_t handle,
               cublasOperation_t transa,
               cublasOperation_t transb,
               int m,
               int n,
               int k,
               const double* alpha,
               const double* A,
               int lda,
               const double* B,
               int ldb,
               const double* beta,
               double* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemm_v2(cublasHandle_t handle,
               cublasOperation_t transa,
               cublasOperation_t transb,
               int m,
               int n,
               int k,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* B,
               int ldb,
               const cuComplex* beta,
               cuComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemm3m(cublasHandle_t handle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int m,
              int n,
              int k,
              const cuComplex* alpha,
              const cuComplex* A,
              int lda,
              const cuComplex* B,
              int ldb,
              const cuComplex* beta,
              cuComplex* C,
              int ldc)
{
    NOT_IMPLEMENTED;
}
cublasStatus_t
cublasCgemm3mEx(cublasHandle_t handle,
                cublasOperation_t transa,
                cublasOperation_t transb,
                int m,
                int n,
                int k,
                const cuComplex* alpha,
                const void* A,
                cudaDataType Atype,
                int lda,
                const void* B,
                cudaDataType Btype,
                int ldb,
                const cuComplex* beta,
                void* C,
                cudaDataType Ctype,
                int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgemm_v2(cublasHandle_t handle,
               cublasOperation_t transa,
               cublasOperation_t transb,
               int m,
               int n,
               int k,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* B,
               int ldb,
               const cuDoubleComplex* beta,
               cuDoubleComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgemm3m(cublasHandle_t handle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int m,
              int n,
              int k,
              const cuDoubleComplex* alpha,
              const cuDoubleComplex* A,
              int lda,
              const cuDoubleComplex* B,
              int ldb,
              const cuDoubleComplex* beta,
              cuDoubleComplex* C,
              int ldc)
{
    NOT_IMPLEMENTED;
}

/* IO in FP16/FP32, computation in float */
cublasStatus_t
cublasSgemmEx(cublasHandle_t handle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int m,
              int n,
              int k,
              const float* alpha,
              const void* A,
              cudaDataType Atype,
              int lda,
              const void* B,
              cudaDataType Btype,
              int ldb,
              const float* beta,
              void* C,
              cudaDataType Ctype,
              int ldc)
{
    NOT_IMPLEMENTED;
}

/* IO in Int8 complex/cuComplex, computation in cuComplex */
cublasStatus_t
cublasCgemmEx(cublasHandle_t handle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int m,
              int n,
              int k,
              const cuComplex* alpha,
              const void* A,
              cudaDataType Atype,
              int lda,
              const void* B,
              cudaDataType Btype,
              int ldb,
              const cuComplex* beta,
              void* C,
              cudaDataType Ctype,
              int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasUint8gemmBias(cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    cublasOperation_t transc,
                    int m,
                    int n,
                    int k,
                    const unsigned char* A,
                    int A_bias,
                    int lda,
                    const unsigned char* B,
                    int B_bias,
                    int ldb,
                    unsigned char* C,
                    int C_bias,
                    int ldc,
                    int C_mult,
                    int C_shift)
{
    NOT_IMPLEMENTED;
}

/* SYRK */
cublasStatus_t
cublasSsyrk_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               int n,
               int k,
               const float* alpha,
               const float* A,
               int lda,
               const float* beta,
               float* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDsyrk_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               int n,
               int k,
               const double* alpha,
               const double* A,
               int lda,
               const double* beta,
               double* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsyrk_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               int n,
               int k,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* beta,
               cuComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZsyrk_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               int n,
               int k,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* beta,
               cuDoubleComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}
/* IO in Int8 complex/cuComplex, computation in cuComplex */
cublasStatus_t
cublasCsyrkEx(cublasHandle_t handle,
              cublasFillMode_t uplo,
              cublasOperation_t trans,
              int n,
              int k,
              const cuComplex* alpha,
              const void* A,
              cudaDataType Atype,
              int lda,
              const cuComplex* beta,
              void* C,
              cudaDataType Ctype,
              int ldc)
{
    NOT_IMPLEMENTED;
}

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
cublasStatus_t
cublasCsyrk3mEx(cublasHandle_t handle,
                cublasFillMode_t uplo,
                cublasOperation_t trans,
                int n,
                int k,
                const cuComplex* alpha,
                const void* A,
                cudaDataType Atype,
                int lda,
                const cuComplex* beta,
                void* C,
                cudaDataType Ctype,
                int ldc)
{
    NOT_IMPLEMENTED;
}

/* HERK */
cublasStatus_t
cublasCherk_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               int n,
               int k,
               const float* alpha,
               const cuComplex* A,
               int lda,
               const float* beta,
               cuComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZherk_v2(cublasHandle_t handle,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               int n,
               int k,
               const double* alpha,
               const cuDoubleComplex* A,
               int lda,
               const double* beta,
               cuDoubleComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

/* IO in Int8 complex/cuComplex, computation in cuComplex */
cublasStatus_t
cublasCherkEx(cublasHandle_t handle,
              cublasFillMode_t uplo,
              cublasOperation_t trans,
              int n,
              int k,
              const float* alpha,
              const void* A,
              cudaDataType Atype,
              int lda,
              const float* beta,
              void* C,
              cudaDataType Ctype,
              int ldc)
{
    NOT_IMPLEMENTED;
}

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
cublasStatus_t
cublasCherk3mEx(cublasHandle_t handle,
                cublasFillMode_t uplo,
                cublasOperation_t trans,
                int n,
                int k,
                const float* alpha,
                const void* A,
                cudaDataType Atype,
                int lda,
                const float* beta,
                void* C,
                cudaDataType Ctype,
                int ldc)
{
    NOT_IMPLEMENTED;
}

/* SYR2K */
cublasStatus_t
cublasSsyr2k_v2(cublasHandle_t handle,
                cublasFillMode_t uplo,
                cublasOperation_t trans,
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
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDsyr2k_v2(cublasHandle_t handle,
                cublasFillMode_t uplo,
                cublasOperation_t trans,
                int n,
                int k,
                const double* alpha,
                const double* A,
                int lda,
                const double* B,
                int ldb,
                const double* beta,
                double* C,
                int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsyr2k_v2(cublasHandle_t handle,
                cublasFillMode_t uplo,
                cublasOperation_t trans,
                int n,
                int k,
                const cuComplex* alpha,
                const cuComplex* A,
                int lda,
                const cuComplex* B,
                int ldb,
                const cuComplex* beta,
                cuComplex* C,
                int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZsyr2k_v2(cublasHandle_t handle,
                cublasFillMode_t uplo,
                cublasOperation_t trans,
                int n,
                int k,
                const cuDoubleComplex* alpha,
                const cuDoubleComplex* A,
                int lda,
                const cuDoubleComplex* B,
                int ldb,
                const cuDoubleComplex* beta,
                cuDoubleComplex* C,
                int ldc)
{
    NOT_IMPLEMENTED;
}
/* HER2K */
cublasStatus_t
cublasCher2k_v2(cublasHandle_t handle,
                cublasFillMode_t uplo,
                cublasOperation_t trans,
                int n,
                int k,
                const cuComplex* alpha,
                const cuComplex* A,
                int lda,
                const cuComplex* B,
                int ldb,
                const float* beta,
                cuComplex* C,
                int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZher2k_v2(cublasHandle_t handle,
                cublasFillMode_t uplo,
                cublasOperation_t trans,
                int n,
                int k,
                const cuDoubleComplex* alpha,
                const cuDoubleComplex* A,
                int lda,
                const cuDoubleComplex* B,
                int ldb,
                const double* beta,
                cuDoubleComplex* C,
                int ldc)
{
    NOT_IMPLEMENTED;
}
/* SYRKX : eXtended SYRK*/
cublasStatus_t
cublasSsyrkx(cublasHandle_t handle,
             cublasFillMode_t uplo,
             cublasOperation_t trans,
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
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDsyrkx(cublasHandle_t handle,
             cublasFillMode_t uplo,
             cublasOperation_t trans,
             int n,
             int k,
             const double* alpha,
             const double* A,
             int lda,
             const double* B,
             int ldb,
             const double* beta,
             double* C,
             int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsyrkx(cublasHandle_t handle,
             cublasFillMode_t uplo,
             cublasOperation_t trans,
             int n,
             int k,
             const cuComplex* alpha,
             const cuComplex* A,
             int lda,
             const cuComplex* B,
             int ldb,
             const cuComplex* beta,
             cuComplex* C,
             int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZsyrkx(cublasHandle_t handle,
             cublasFillMode_t uplo,
             cublasOperation_t trans,
             int n,
             int k,
             const cuDoubleComplex* alpha,
             const cuDoubleComplex* A,
             int lda,
             const cuDoubleComplex* B,
             int ldb,
             const cuDoubleComplex* beta,
             cuDoubleComplex* C,
             int ldc)
{
    NOT_IMPLEMENTED;
}
/* HERKX : eXtended HERK */
cublasStatus_t
cublasCherkx(cublasHandle_t handle,
             cublasFillMode_t uplo,
             cublasOperation_t trans,
             int n,
             int k,
             const cuComplex* alpha,
             const cuComplex* A,
             int lda,
             const cuComplex* B,
             int ldb,
             const float* beta,
             cuComplex* C,
             int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZherkx(cublasHandle_t handle,
             cublasFillMode_t uplo,
             cublasOperation_t trans,
             int n,
             int k,
             const cuDoubleComplex* alpha,
             const cuDoubleComplex* A,
             int lda,
             const cuDoubleComplex* B,
             int ldb,
             const double* beta,
             cuDoubleComplex* C,
             int ldc)
{
    NOT_IMPLEMENTED;
}
/* SYMM */
cublasStatus_t
cublasSsymm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               int m,
               int n,
               const float* alpha,
               const float* A,
               int lda,
               const float* B,
               int ldb,
               const float* beta,
               float* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDsymm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               int m,
               int n,
               const double* alpha,
               const double* A,
               int lda,
               const double* B,
               int ldb,
               const double* beta,
               double* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCsymm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               int m,
               int n,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* B,
               int ldb,
               const cuComplex* beta,
               cuComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZsymm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               int m,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* B,
               int ldb,
               const cuDoubleComplex* beta,
               cuDoubleComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

/* HEMM */
cublasStatus_t
cublasChemm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               int m,
               int n,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* B,
               int ldb,
               const cuComplex* beta,
               cuComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZhemm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               int m,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* B,
               int ldb,
               const cuDoubleComplex* beta,
               cuDoubleComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

/* TRSM */
cublasStatus_t
cublasStrsm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int m,
               int n,
               const float* alpha,
               const float* A,
               int lda,
               float* B,
               int ldb)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtrsm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int m,
               int n,
               const double* alpha,
               const double* A,
               int lda,
               double* B,
               int ldb)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtrsm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int m,
               int n,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               cuComplex* B,
               int ldb)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtrsm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int m,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               cuDoubleComplex* B,
               int ldb)
{
    NOT_IMPLEMENTED;
}

/* TRMM */
cublasStatus_t
cublasStrmm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int m,
               int n,
               const float* alpha,
               const float* A,
               int lda,
               const float* B,
               int ldb,
               float* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtrmm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int m,
               int n,
               const double* alpha,
               const double* A,
               int lda,
               const double* B,
               int ldb,
               double* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtrmm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int m,
               int n,
               const cuComplex* alpha,
               const cuComplex* A,
               int lda,
               const cuComplex* B,
               int ldb,
               cuComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtrmm_v2(cublasHandle_t handle,
               cublasSideMode_t side,
               cublasFillMode_t uplo,
               cublasOperation_t trans,
               cublasDiagType_t diag,
               int m,
               int n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int lda,
               const cuDoubleComplex* B,
               int ldb,
               cuDoubleComplex* C,
               int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasSgemmBatched(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* const Aarray[],
                   int lda,
                   const float* const Barray[],
                   int ldb,
                   const float* beta,
                   float* const Carray[],
                   int ldc,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgemmBatched(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const double* alpha,
                   const double* const Aarray[],
                   int lda,
                   const double* const Barray[],
                   int ldb,
                   const double* beta,
                   double* const Carray[],
                   int ldc,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemmBatched(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const cuComplex* alpha,
                   const cuComplex* const Aarray[],
                   int lda,
                   const cuComplex* const Barray[],
                   int ldb,
                   const cuComplex* beta,
                   cuComplex* const Carray[],
                   int ldc,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemm3mBatched(cublasHandle_t handle,
                     cublasOperation_t transa,
                     cublasOperation_t transb,
                     int m,
                     int n,
                     int k,
                     const cuComplex* alpha,
                     const cuComplex* const Aarray[],
                     int lda,
                     const cuComplex* const Barray[],
                     int ldb,
                     const cuComplex* beta,
                     cuComplex* const Carray[],
                     int ldc,
                     int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgemmBatched(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const cuDoubleComplex* alpha,
                   const cuDoubleComplex* const Aarray[],
                   int lda,
                   const cuDoubleComplex* const Barray[],
                   int ldb,
                   const cuDoubleComplex* beta,
                   cuDoubleComplex* const Carray[],
                   int ldc,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasGemmBatchedEx(cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const void* alpha,
                    const void* const Aarray[],
                    cudaDataType Atype,
                    int lda,
                    const void* const Barray[],
                    cudaDataType Btype,
                    int ldb,
                    const void* beta,
                    void* const Carray[],
                    cudaDataType Ctype,
                    int ldc,
                    int batchCount,
                    cublasComputeType_t computeType,
                    cublasGemmAlgo_t algo)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgemmStridedBatched(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const double* alpha,
                          const double* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const double* B,
                          int ldb,
                          long long int strideB,
                          const double* beta,
                          double* C,
                          int ldc,
                          long long int strideC,
                          int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemmStridedBatched(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const cuComplex* alpha,
                          const cuComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuComplex* B,
                          int ldb,
                          long long int strideB,
                          const cuComplex* beta,
                          cuComplex* C,
                          int ldc,
                          long long int strideC,
                          int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgemm3mStridedBatched(cublasHandle_t handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m,
                            int n,
                            int k,
                            const cuComplex* alpha,
                            const cuComplex* A,
                            int lda,
                            long long int strideA, /* purposely signed */
                            const cuComplex* B,
                            int ldb,
                            long long int strideB,
                            const cuComplex* beta,
                            cuComplex* C,
                            int ldc,
                            long long int strideC,
                            int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgemmStridedBatched(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const cuDoubleComplex* alpha,
                          const cuDoubleComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuDoubleComplex* B,
                          int ldb,
                          long long int strideB,
                          const cuDoubleComplex* beta, /* host or device poi */
                          cuDoubleComplex* C,
                          int ldc,
                          long long int strideC,
                          int batchCount)
{
    NOT_IMPLEMENTED;
}

/* ---------------- CUBLAS BLAS-like extension ---------------- */
/* GEAM */
cublasStatus_t
cublasSgeam(cublasHandle_t handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            const float* alpha,
            const float* A,
            int lda,
            const float* beta,
            const float* B,
            int ldb,
            float* C,
            int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgeam(cublasHandle_t handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            const double* alpha,
            const double* A,
            int lda,
            const double* beta,
            const double* B,
            int ldb,
            double* C,
            int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgeam(cublasHandle_t handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            const cuComplex* alpha,
            const cuComplex* A,
            int lda,
            const cuComplex* beta,
            const cuComplex* B,
            int ldb,
            cuComplex* C,
            int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgeam(cublasHandle_t handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* A,
            int lda,
            const cuDoubleComplex* beta,
            const cuDoubleComplex* B,
            int ldb,
            cuDoubleComplex* C,
            int ldc)
{
    NOT_IMPLEMENTED;
}

/* Batched LU - GETRF*/
cublasStatus_t
cublasSgetrfBatched(cublasHandle_t handle,
                    int n,
                    float* const A[], /*Device pointer*/
                    int lda,
                    int* P,    /*Device Pointer*/
                    int* info, /*Device Pointer*/
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgetrfBatched(cublasHandle_t handle,
                    int n,
                    double* const A[], /*Device pointer*/
                    int lda,
                    int* P,    /*Device Pointer*/
                    int* info, /*Device Pointer*/
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgetrfBatched(cublasHandle_t handle,
                    int n,
                    cuComplex* const A[], /*Device pointer*/
                    int lda,
                    int* P,    /*Device Pointer*/
                    int* info, /*Device Pointer*/
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgetrfBatched(cublasHandle_t handle,
                    int n,
                    cuDoubleComplex* const A[], /*Device pointer*/
                    int lda,
                    int* P,    /*Device Pointer*/
                    int* info, /*Device Pointer*/
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

/* Batched inversion based on LU factorization from getrf */
cublasStatus_t
cublasSgetriBatched(cublasHandle_t handle,
                    int n,
                    const float* const A[], /*Device pointer*/
                    int lda,
                    const int* P,     /*Device pointer*/
                    float* const C[], /*Device pointer*/
                    int ldc,
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgetriBatched(cublasHandle_t handle,
                    int n,
                    const double* const A[], /*Device pointer*/
                    int lda,
                    const int* P,      /*Device pointer*/
                    double* const C[], /*Device pointer*/
                    int ldc,
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgetriBatched(cublasHandle_t handle,
                    int n,
                    const cuComplex* const A[], /*Device pointer*/
                    int lda,
                    const int* P,         /*Device pointer*/
                    cuComplex* const C[], /*Device pointer*/
                    int ldc,
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgetriBatched(cublasHandle_t handle,
                    int n,
                    const cuDoubleComplex* const A[], /*Device pointer*/
                    int lda,
                    const int* P,               /*Device pointer*/
                    cuDoubleComplex* const C[], /*Device pointer*/
                    int ldc,
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

/* Batched solver based on LU factorization from getrf */

cublasStatus_t
cublasSgetrsBatched(cublasHandle_t handle,
                    cublasOperation_t trans,
                    int n,
                    int nrhs,
                    const float* const Aarray[],
                    int lda,
                    const int* devIpiv,
                    float* const Barray[],
                    int ldb,
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgetrsBatched(cublasHandle_t handle,
                    cublasOperation_t trans,
                    int n,
                    int nrhs,
                    const double* const Aarray[],
                    int lda,
                    const int* devIpiv,
                    double* const Barray[],
                    int ldb,
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgetrsBatched(cublasHandle_t handle,
                    cublasOperation_t trans,
                    int n,
                    int nrhs,
                    const cuComplex* const Aarray[],
                    int lda,
                    const int* devIpiv,
                    cuComplex* const Barray[],
                    int ldb,
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgetrsBatched(cublasHandle_t handle,
                    cublasOperation_t trans,
                    int n,
                    int nrhs,
                    const cuDoubleComplex* const Aarray[],
                    int lda,
                    const int* devIpiv,
                    cuDoubleComplex* const Barray[],
                    int ldb,
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

/* TRSM - Batched Triangular Solver */
cublasStatus_t
cublasStrsmBatched(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const float* alpha, /*Host or Device Pointer*/
                   const float* const A[],
                   int lda,
                   float* const B[],
                   int ldb,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtrsmBatched(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const double* alpha, /*Host or Device Pointer*/
                   const double* const A[],
                   int lda,
                   double* const B[],
                   int ldb,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtrsmBatched(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const cuComplex* alpha, /*Host or Device Pointer*/
                   const cuComplex* const A[],
                   int lda,
                   cuComplex* const B[],
                   int ldb,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtrsmBatched(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const cuDoubleComplex* alpha, /*Host or Device Pointer*/
                   const cuDoubleComplex* const A[],
                   int lda,
                   cuDoubleComplex* const B[],
                   int ldb,
                   int batchCount)
{
    NOT_IMPLEMENTED;
}

/* Batched - MATINV*/
cublasStatus_t
cublasSmatinvBatched(cublasHandle_t handle,
                     int n,
                     const float* const A[], /*Device pointer*/
                     int lda,
                     float* const Ainv[], /*Device pointer*/
                     int lda_inv,
                     int* info, /*Device Pointer*/
                     int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDmatinvBatched(cublasHandle_t handle,
                     int n,
                     const double* const A[], /*Device pointer*/
                     int lda,
                     double* const Ainv[], /*Device pointer*/
                     int lda_inv,
                     int* info, /*Device Pointer*/
                     int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCmatinvBatched(cublasHandle_t handle,
                     int n,
                     const cuComplex* const A[], /*Device pointer*/
                     int lda,
                     cuComplex* const Ainv[], /*Device pointer*/
                     int lda_inv,
                     int* info, /*Device Pointer*/
                     int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZmatinvBatched(cublasHandle_t handle,
                     int n,
                     const cuDoubleComplex* const A[], /*Device pointer*/
                     int lda,
                     cuDoubleComplex* const Ainv[], /*Device pointer*/
                     int lda_inv,
                     int* info, /*Device Pointer*/
                     int batchSize)
{
    NOT_IMPLEMENTED;
}

/* Batch QR Factorization */
cublasStatus_t
cublasSgeqrfBatched(cublasHandle_t handle,
                    int m,
                    int n,
                    float* const Aarray[], /*Device pointer*/
                    int lda,
                    float* const TauArray[], /*Device pointer*/
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgeqrfBatched(cublasHandle_t handle,
                    int m,
                    int n,
                    double* const Aarray[], /*Device pointer*/
                    int lda,
                    double* const TauArray[], /*Device pointer*/
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgeqrfBatched(cublasHandle_t handle,
                    int m,
                    int n,
                    cuComplex* const Aarray[], /*Device pointer*/
                    int lda,
                    cuComplex* const TauArray[], /*Device pointer*/
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgeqrfBatched(cublasHandle_t handle,
                    int m,
                    int n,
                    cuDoubleComplex* const Aarray[], /*Device pointer*/
                    int lda,
                    cuDoubleComplex* const TauArray[], /*Device pointer*/
                    int* info,
                    int batchSize)
{
    NOT_IMPLEMENTED;
}
/* Least Square Min only m >= n and Non-transpose supported */
cublasStatus_t
cublasSgelsBatched(cublasHandle_t handle,
                   cublasOperation_t trans,
                   int m,
                   int n,
                   int nrhs,
                   float* const Aarray[], /*Device pointer*/
                   int lda,
                   float* const Carray[], /*Device pointer*/
                   int ldc,
                   int* info,
                   int* devInfoArray, /*Device pointer*/
                   int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDgelsBatched(cublasHandle_t handle,
                   cublasOperation_t trans,
                   int m,
                   int n,
                   int nrhs,
                   double* const Aarray[], /*Device pointer*/
                   int lda,
                   double* const Carray[], /*Device pointer*/
                   int ldc,
                   int* info,
                   int* devInfoArray, /*Device pointer*/
                   int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCgelsBatched(cublasHandle_t handle,
                   cublasOperation_t trans,
                   int m,
                   int n,
                   int nrhs,
                   cuComplex* const Aarray[], /*Device pointer*/
                   int lda,
                   cuComplex* const Carray[], /*Device pointer*/
                   int ldc,
                   int* info,
                   int* devInfoArray,
                   int batchSize)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZgelsBatched(cublasHandle_t handle,
                   cublasOperation_t trans,
                   int m,
                   int n,
                   int nrhs,
                   cuDoubleComplex* const Aarray[], /*Device pointer*/
                   int lda,
                   cuDoubleComplex* const Carray[], /*Device pointer*/
                   int ldc,
                   int* info,
                   int* devInfoArray,
                   int batchSize)
{
    NOT_IMPLEMENTED;
}
/* DGMM */
cublasStatus_t
cublasSdgmm(cublasHandle_t handle,
            cublasSideMode_t mode,
            int m,
            int n,
            const float* A,
            int lda,
            const float* x,
            int incx,
            float* C,
            int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDdgmm(cublasHandle_t handle,
            cublasSideMode_t mode,
            int m,
            int n,
            const double* A,
            int lda,
            const double* x,
            int incx,
            double* C,
            int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCdgmm(cublasHandle_t handle,
            cublasSideMode_t mode,
            int m,
            int n,
            const cuComplex* A,
            int lda,
            const cuComplex* x,
            int incx,
            cuComplex* C,
            int ldc)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZdgmm(cublasHandle_t handle,
            cublasSideMode_t mode,
            int m,
            int n,
            const cuDoubleComplex* A,
            int lda,
            const cuDoubleComplex* x,
            int incx,
            cuDoubleComplex* C,
            int ldc)
{
    NOT_IMPLEMENTED;
}

/* TPTTR : Triangular Pack format to Triangular format */
cublasStatus_t
cublasStpttr(cublasHandle_t handle,
             cublasFillMode_t uplo,
             int n,
             const float* AP,
             float* A,
             int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtpttr(cublasHandle_t handle,
             cublasFillMode_t uplo,
             int n,
             const double* AP,
             double* A,
             int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtpttr(cublasHandle_t handle,
             cublasFillMode_t uplo,
             int n,
             const cuComplex* AP,
             cuComplex* A,
             int lda)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtpttr(cublasHandle_t handle,
             cublasFillMode_t uplo,
             int n,
             const cuDoubleComplex* AP,
             cuDoubleComplex* A,
             int lda)
{
    NOT_IMPLEMENTED;
}
/* TRTTP : Triangular format to Triangular Pack format */
cublasStatus_t
cublasStrttp(cublasHandle_t handle,
             cublasFillMode_t uplo,
             int n,
             const float* A,
             int lda,
             float* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasDtrttp(cublasHandle_t handle,
             cublasFillMode_t uplo,
             int n,
             const double* A,
             int lda,
             double* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasCtrttp(cublasHandle_t handle,
             cublasFillMode_t uplo,
             int n,
             const cuComplex* A,
             int lda,
             cuComplex* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasZtrttp(cublasHandle_t handle,
             cublasFillMode_t uplo,
             int n,
             const cuDoubleComplex* A,
             int lda,
             cuDoubleComplex* AP)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasLtCreate(cublasLtHandle_t* lightHandle)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasLtDestroy(cublasLtHandle_t lightHandle)
{
    NOT_IMPLEMENTED;
}

const char*
cublasLtGetStatusName(cublasStatus_t status)
{
    NOT_IMPLEMENTED;
}

const char*
cublasLtGetStatusString(cublasStatus_t status)
{
    NOT_IMPLEMENTED;
}

size_t
cublasLtGetVersion(void)
{
    NOT_IMPLEMENTED;
}

size_t
cublasLtGetCudartVersion(void)
{
    NOT_IMPLEMENTED;
}

cublasStatus_t
cublasLtGetProperty(libraryPropertyType type, int* value)
{
    NOT_IMPLEMENTED;
}

/** Matrix layout conversion helper (C = alpha * op(A) + beta * op(B))
 *
 * Can be used to change memory order of data or to scale and shift the values.
 *
 * \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been
 * initialized \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in
 * conflict or in an impossible configuration; e.g. when A is not NULL, but
 * Adesc is NULL \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current
 * implementation on selected device doesn't support configured operation
 * \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be
 * run using selected device \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda
 * reported execution error from the device \retval     CUBLAS_STATUS_SUCCESS if
 * the operation completed successfully
 */
cublasStatus_t
cublasLtMatrixTransform(cublasLtHandle_t lightHandle,
                        cublasLtMatrixTransformDesc_t transformDesc,
                        const void* alpha, /* host or device pointer */
                        const void* A,
                        cublasLtMatrixLayout_t Adesc,
                        const void* beta, /* host or device pointer */
                        const void* B,
                        cublasLtMatrixLayout_t Bdesc,
                        void* C,
                        cublasLtMatrixLayout_t Cdesc,
                        cudaStream_t stream)
{
    NOT_IMPLEMENTED;
}

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatrixLayout_t */
/* ---------------------------------------------------------------------------------------*/

/** Internal. Do not use directly.
 */
cublasStatus_t
cublasLtMatrixLayoutInit_internal( //
  cublasLtMatrixLayout_t matLayout,
  size_t size,
  cudaDataType type,
  uint64_t rows,
  uint64_t cols,
  int64_t ld)
{
    NOT_IMPLEMENTED;
}

/** Destroy matrix layout descriptor.
 *
 * \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
 */
cublasStatus_t
cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
{
    NOT_IMPLEMENTED;
}

/** Set matrix layout descriptor attribute.
 *
 * \param[in]  matLayout    The descriptor
 * \param[in]  attr         The attribute
 * \param[in]  buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t
cublasLtMatrixLayoutSetAttribute( //
  cublasLtMatrixLayout_t matLayout,
  cublasLtMatrixLayoutAttribute_t attr,
  const void* buf,
  size_t sizeInBytes)
{
    NOT_IMPLEMENTED;
}

/** Get matrix layout descriptor attribute.
 *
 * \param[in]  matLayout    The descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is
 * CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of bytes actually
 * written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten
 * is NULL, or if  sizeInBytes is non-zero and buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to
 * user memory
 */
cublasStatus_t
cublasLtMatrixLayoutGetAttribute( //
  cublasLtMatrixLayout_t matLayout,
  cublasLtMatrixLayoutAttribute_t attr,
  void* buf,
  size_t sizeInBytes,
  size_t* sizeWritten)
{
    NOT_IMPLEMENTED;
}

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatmulDesc_t */
/* ---------------------------------------------------------------------------------------*/

/** Matmul descriptor attributes to define details of the operation. */

/** Internal. Do not use directly.
 */
cublasStatus_t
cublasLtMatmulDescInit_internal( //
  cublasLtMatmulDesc_t matmulDesc,
  size_t size,
  cublasComputeType_t computeType,
  cudaDataType_t scaleType)
{
    NOT_IMPLEMENTED;
}

/** Destroy matmul operation descriptor.
 *
 * \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
 */
cublasStatus_t
cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
{
    NOT_IMPLEMENTED;
}

/** Get matmul operation descriptor attribute.
 *
 * \param[in]  matmulDesc   The descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is
 * CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of bytes actually
 * written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten
 * is NULL, or if  sizeInBytes is non-zero and buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to
 * user memory
 */
cublasStatus_t
cublasLtMatmulDescGetAttribute( //
  cublasLtMatmulDesc_t matmulDesc,
  cublasLtMatmulDescAttributes_t attr,
  void* buf,
  size_t sizeInBytes,
  size_t* sizeWritten)
{
    NOT_IMPLEMENTED;
}

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatrixTransformDesc_t */
/* ---------------------------------------------------------------------------------------*/

/** Internal. Do not use directly.
 */
cublasStatus_t
cublasLtMatrixTransformDescInit_internal(
  cublasLtMatrixTransformDesc_t transformDesc,
  size_t size,
  cudaDataType scaleType)
{
    NOT_IMPLEMENTED;
}

/** Create new matrix transform operation descriptor.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
cublasStatus_t
cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t* transformDesc,
                                  cudaDataType scaleType)
{
    NOT_IMPLEMENTED;
}

/** Destroy matrix transform operation descriptor.
 *
 * \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
 */
cublasStatus_t
cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc)
{
    NOT_IMPLEMENTED;
}

/** Set matrix transform operation descriptor attribute.
 *
 * \param[in]  transformDesc  The descriptor
 * \param[in]  attr           The attribute
 * \param[in]  buf            memory address containing the new value
 * \param[in]  sizeInBytes    size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t
cublasLtMatrixTransformDescSetAttribute( //
  cublasLtMatrixTransformDesc_t transformDesc,
  cublasLtMatrixTransformDescAttributes_t attr,
  const void* buf,
  size_t sizeInBytes)
{
    NOT_IMPLEMENTED;
}

/** Get matrix transform operation descriptor attribute.
 *
 * \param[in]  transformDesc  The descriptor
 * \param[in]  attr           The attribute
 * \param[out] buf            memory address containing the new value
 * \param[in]  sizeInBytes    size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten    only valid when return value is
 * CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of bytes actually
 * written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten
 * is NULL, or if  sizeInBytes is non-zero and buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to
 * user memory
 */
cublasStatus_t
cublasLtMatrixTransformDescGetAttribute( //
  cublasLtMatrixTransformDesc_t transformDesc,
  cublasLtMatrixTransformDescAttributes_t attr,
  void* buf,
  size_t sizeInBytes,
  size_t* sizeWritten)
{
    NOT_IMPLEMENTED;
}

/** Internal. Do not use directly.
 */
cublasStatus_t
cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t pref,
                                      size_t size)
{
    NOT_IMPLEMENTED;
}

/** Destroy matmul heuristic search preference descriptor.
 *
 * \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
 */
cublasStatus_t
cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
{
    NOT_IMPLEMENTED;
}

/** Get matmul heuristic search preference descriptor attribute.
 *
 * \param[in]  pref         The descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is
 * CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of bytes actually
 * written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten
 * is NULL, or if  sizeInBytes is non-zero and buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to
 * user memory
 */
cublasStatus_t
cublasLtMatmulPreferenceGetAttribute( //
  cublasLtMatmulPreference_t pref,
  cublasLtMatmulPreferenceAttributes_t attr,
  void* buf,
  size_t sizeInBytes,
  size_t* sizeWritten)
{
    NOT_IMPLEMENTED;
}

/** Initialize algo structure
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if algo is NULL or algoId is outside
 * of recognized range \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algoId is not
 * supported for given combination of data types \retval CUBLAS_STATUS_SUCCESS
 * if the structure was successfully initialized
 */
cublasStatus_t
cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle,
                       cublasComputeType_t computeType,
                       cudaDataType_t scaleType,
                       cudaDataType_t Atype,
                       cudaDataType_t Btype,
                       cudaDataType_t Ctype,
                       cudaDataType_t Dtype,
                       int algoId,
                       cublasLtMatmulAlgo_t* algo)
{
    NOT_IMPLEMENTED;
}

/** Check configured algo descriptor for correctness and support on current
 * device.
 *
 * Result includes required workspace size and calculated wave count.
 *
 * CUBLAS_STATUS_SUCCESS doesn't fully guarantee algo will run (will fail if
 * e.g. buffers are not correctly aligned){
    NOT_IMPLEMENTED;
} but if cublasLtMatmulAlgoCheck
 * fails, the algo will not run.
 *
 * \param[in]  algo    algo configuration to check
 * \param[out] result  result structure to report algo runtime characteristics;
 * algo field is never updated
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if matrix layout descriptors or
 * operation descriptor don't match algo descriptor \retval
 * CUBLAS_STATUS_NOT_SUPPORTED  if algo configuration or data type combination
 * is not currently supported on given device \retval
 * CUBLAS_STATUS_ARCH_MISMATCH  if algo configuration cannot be run using the
 * selected device \retval     CUBLAS_STATUS_SUCCESS        if check was
 * successful
 */
cublasStatus_t
cublasLtMatmulAlgoCheck( //
  cublasLtHandle_t lightHandle,
  cublasLtMatmulDesc_t operationDesc,
  cublasLtMatrixLayout_t Adesc,
  cublasLtMatrixLayout_t Bdesc,
  cublasLtMatrixLayout_t Cdesc,
  cublasLtMatrixLayout_t Ddesc,
  const cublasLtMatmulAlgo_t* algo, ///< may point to result->algo
  cublasLtMatmulHeuristicResult_t* result)
{
    NOT_IMPLEMENTED;
}

/** Get algo capability attribute.
 *
 * E.g. to get list of supported Tile IDs:
 *      cublasLtMatmulTile_t tiles[CUBLASLT_MATMUL_TILE_END];
 *      size_t num_tiles, size_written;
 *      if (cublasLtMatmulAlgoCapGetAttribute(algo, CUBLASLT_ALGO_CAP_TILE_IDS,
 * tiles, sizeof(tiles), size_written) == CUBLAS_STATUS_SUCCESS) { num_tiles =
 * size_written / sizeof(tiles[0]){
    NOT_IMPLEMENTED;
}
 *      }
 *
 * \param[in]  algo         The algo descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is
 * CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of bytes actually
 * written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten
 * is NULL, or if  sizeInBytes is non-zero and buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to
 * user memory
 */
cublasStatus_t
cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t* algo,
                                  cublasLtMatmulAlgoCapAttributes_t attr,
                                  void* buf,
                                  size_t sizeInBytes,
                                  size_t* sizeWritten)
{
    NOT_IMPLEMENTED;
}

/** Set algo configuration attribute.
 *
 * \param[in]  algo         The algo descriptor
 * \param[in]  attr         The attribute
 * \param[in]  buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t
cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t* algo,
                                     cublasLtMatmulAlgoConfigAttributes_t attr,
                                     const void* buf,
                                     size_t sizeInBytes)
{
    NOT_IMPLEMENTED;
}

/** Get algo configuration attribute.
 *
 * \param[in]  algo         The algo descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is
 * CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of bytes actually
 * written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten
 * is NULL, or if  sizeInBytes is non-zero and buf is NULL or sizeInBytes
 * doesn't match size of internal storage for selected attribute \retval
 * CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to
 * user memory
 */
cublasStatus_t
cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo,
                                     cublasLtMatmulAlgoConfigAttributes_t attr,
                                     void* buf,
                                     size_t sizeInBytes,
                                     size_t* sizeWritten)
{
    NOT_IMPLEMENTED;
}

/** Experimental: Logger callback setter.
 *
 * \param[in]  callback                     a user defined callback function to
 * be called by the logger
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if callback was set successfully
 */
cublasStatus_t
cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback)
{
    NOT_IMPLEMENTED;
}

/** Experimental: Log file setter.
 *
 * \param[in]  file                         an open file with write permissions
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if log file was set successfully
 */
cublasStatus_t
cublasLtLoggerSetFile(FILE* file)
{
    NOT_IMPLEMENTED;
}

/** Experimental: Open log file.
 *
 * \param[in]  logFile                      log file path. if the log file does
 * not exist, it will be created
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if log file was created successfully
 */
cublasStatus_t
cublasLtLoggerOpenFile(const char* logFile)
{
    NOT_IMPLEMENTED;
}

/** Experimental: Log level setter.
 *
 * \param[in]  level                        log level, should be one of the
 * following: 0. Off
 *                                          1. Errors
 *                                          2. Performance Trace
 *                                          3. Performance Hints
 *                                          4. Heuristics Trace
 *                                          5. API Trace
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if log level is not one of the above
 * levels
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if log level was set successfully
 */
cublasStatus_t
cublasLtLoggerSetLevel(int level)
{
    NOT_IMPLEMENTED;
}

/** Experimental: Log mask setter.
 *
 * \param[in]  mask                         log mask, should be a combination of
 * the following masks: 0.  Off
 *                                          1.  Errors
 *                                          2.  Performance Trace
 *                                          4.  Performance Hints
 *                                          8.  Heuristics Trace
 *                                          16. API Trace
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if log mask was set successfully
 */
cublasStatus_t
cublasLtLoggerSetMask(int mask)
{
    NOT_IMPLEMENTED;
}

/** Experimental: Disable logging for the entire session.
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if disabled logging
 */
cublasStatus_t
cublasLtLoggerForceDisable()
{
    NOT_IMPLEMENTED;
}
