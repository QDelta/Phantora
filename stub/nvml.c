#include "common.h"
#include <nvml.h>

nvmlReturn_t
nvmlInit_v2(void)
{
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
