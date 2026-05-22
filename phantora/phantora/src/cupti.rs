//! CUPTI-based per-op GPU activity timing for the `estimate!` macro.
//!
//! Default is CUPTI GPU activity timing (kernel, memcpy, and memset activity records).
//! Set `PHANTORA_USE_CUPTI=0` (or `false`) to fall back to the older
//! `cudaEventElapsedTime` path.
//!
//! CUPTI activity records contain GPU-clock timestamps captured by the driver
//! when kernels, memcpy, and memset operations actually started and ended on the device.
//! Host stalls between the two `cudaEventRecord` calls are invisible to
//! CUPTI, giving device activity time.

use std::sync::OnceLock;

#[link(name = "cupti", kind = "dylib")]
extern "C" {
    fn phantora_cupti_init() -> i32;
    fn phantora_cupti_clear();
    fn phantora_cupti_flush();
    fn phantora_cupti_sum_activity_ns() -> u64;
}

/// Read the env var once and cache. Default is **on** — CUPTI gives GPU
/// activity time. Set `PHANTORA_USE_CUPTI=0` (or `false`) to
/// fall back to `cudaEventElapsedTime`.
pub fn enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("PHANTORA_USE_CUPTI") {
        Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        Err(_) => true,
    })
}

/// Initialise CUPTI activity tracking. No-ops when CUPTI is disabled.
/// Call once at simulator startup after the first CUDA call.
pub fn init() {
    if !enabled() {
        return;
    }
    let r = unsafe { phantora_cupti_init() };
    if r != 0 {
        panic!(
            "CUPTI init failed (code {}) — cannot continue with PHANTORA_USE_CUPTI enabled",
            r
        );
    }
}

/// Reset the activity-ns accumulator before a measurement window opens.
#[inline]
pub fn clear() {
    if enabled() {
        unsafe { phantora_cupti_clear() }
    }
}

/// Force CUPTI to drain its activity buffers, then read the sum of
/// GPU activity time observed since the last `clear()`. Returns ns.
/// Returns 0 when CUPTI is disabled.
#[inline]
pub fn flush_and_sum_ns() -> u64 {
    if enabled() {
        unsafe {
            phantora_cupti_flush();
            phantora_cupti_sum_activity_ns()
        }
    } else {
        0
    }
}
