#include "common.h"
#include <nvml.h>

nvmlReturn_t
nvmlInit_v2(void)
{
    return NVML_SUCCESS;
}

// PyTorch 2.9.1's c10::cuda::DriverAPI (driver_api.cpp) looks up this symbol
// during BF16 cast and aborts with INTERNAL ASSERT FAILED if it's missing.
// Match the version reported by cudaDriverGetVersion / cudaRuntimeGetVersion.
nvmlReturn_t
nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion)
{
    if (cudaDriverVersion == NULL)
        return NVML_ERROR_INVALID_ARGUMENT;
    *cudaDriverVersion = 12080;
    return NVML_SUCCESS;
}

nvmlReturn_t
nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device)
{
    return NVML_SUCCESS;
}

nvmlReturn_t
nvmlDeviceGetNvLinkRemoteDeviceType(
  nvmlDevice_t device,
  unsigned int link,
  nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType)
{
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t
nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device,
                                    unsigned int link,
                                    nvmlPciInfo_t* pci)
{
    return NVML_ERROR_NOT_SUPPORTED;
}

#undef nvmlDeviceGetComputeRunningProcesses
nvmlReturn_t
nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device,
                                     unsigned int* infoCount,
                                     nvmlProcessInfo_t* infos)
{
    if (*infoCount == 0) {
        *infoCount = 1;
        return NVML_ERROR_INSUFFICIENT_SIZE;
    } else {
        *infoCount = 1;
        return NVML_SUCCESS;
    }
}
