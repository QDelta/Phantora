use std::ffi;

pub const CUDA_MEMCPY_HOST_HOST: ffi::c_int = 0;
pub const CUDA_MEMCPY_HOST_DEVICE: ffi::c_int = 1;
pub const CUDA_MEMCPY_DEVICE_HOST: ffi::c_int = 2;
pub const CUDA_MEMCPY_DEVICE_DEVICE: ffi::c_int = 3;

#[link(name = "cudart", kind = "dylib")]
extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut ffi::c_void, size: usize) -> ffi::c_int;

    pub fn cudaFree(devPtr: *mut ffi::c_void) -> ffi::c_int;

    pub fn cudaEventCreate(event: *mut *mut ffi::c_void) -> ffi::c_int;

    pub fn cudaEventRecord(event: *mut ffi::c_void, stream: *mut ffi::c_void) -> ffi::c_int;

    pub fn cudaEventSynchronize(event: *mut ffi::c_void) -> ffi::c_int;

    pub fn cudaEventElapsedTime(
        ms: *mut ffi::c_float,
        start: *mut ffi::c_void,
        end: *mut ffi::c_void,
    ) -> ffi::c_int;

    pub fn cudaEventDestroy(event: *mut ffi::c_void) -> ffi::c_int;

    pub fn cudaMallocHost(ptr: *mut *mut ffi::c_void, size: usize) -> ffi::c_int;

    pub fn cudaFreeHost(ptr: *mut ffi::c_void) -> ffi::c_int;

    pub fn cudaMemcpyAsync(
        dst: *mut ffi::c_void,
        src: *const ffi::c_void,
        count: usize,
        kind: ffi::c_int,
        stream: *mut ffi::c_void,
    ) -> ffi::c_int;

    pub fn cudaStreamSynchronize(stream: *mut ffi::c_void) -> ffi::c_int;
}
