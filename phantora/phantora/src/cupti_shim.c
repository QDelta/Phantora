// Small C wrapper around CUPTI's Activity API.
//
// Why this isn't pure-Rust FFI: the CUpti_ActivityKernel record layout
// has gone through 9+ versions (CUpti_ActivityKernel ... Kernel9 in the
// CUDA 12.x headers, more in newer CUDA), each adding fields. Writing
// version-correct #[repr(C)] structs in Rust would couple us to a
// specific CUDA version. Casting to the right struct in C and exposing
// only the fields we need (start, end timestamps in ns) keeps the
// version dependency in one place.
//
// Threading model: clear() / sum() run on the simulator thread that
// drives estimate!; buf_completed runs on CUPTI's *internal* worker
// thread (which CUPTI spawns on its own to drain activity buffers
// without blocking the producer). The accumulator therefore has to be
// shared across threads — using __thread silently zeros the result
// because the simulator thread reads its own TLS slot while CUPTI
// updates a different one. _Atomic gives cross-thread visibility
// without a mutex; clear/flush bracket each estimate! window so the
// counter is single-writer for that window once the flush returns.

#include <cupti.h>
#include <cupti_version.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Atomic uint64_t g_activity_ns = 0;

#if CUPTI_API_VERSION >= 130000
#define PHANTORA_CUPTI_MEMCPY_RECORD CUpti_ActivityMemcpy6
#else
#define PHANTORA_CUPTI_MEMCPY_RECORD CUpti_ActivityMemcpy5
#endif

static void add_activity_duration(uint64_t start, uint64_t end)
{
    if (start == 0 && end == 0) return;
    if (end < start) return;
    atomic_fetch_add_explicit(&g_activity_ns, end - start, memory_order_relaxed);
}

static void CUPTIAPI buf_requested(uint8_t **buffer, size_t *size,
                                   size_t *max_records)
{
    *size = 32 * 1024;
    *max_records = 0;
    if (posix_memalign((void **)buffer, 8, *size) != 0) *buffer = NULL;
}

static void CUPTIAPI buf_completed(CUcontext ctx, uint32_t stream_id,
                                   uint8_t *buffer, size_t size,
                                   size_t valid_size)
{
    (void)ctx;
    (void)stream_id;
    (void)size;
    CUpti_Activity *r = NULL;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &r) == CUPTI_SUCCESS) {
        if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
            r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            // CUDA 12.x ships CUpti_ActivityKernel9 as the latest. The
            // start/end fields are stable across versions, but the
            // surrounding fields differ — cast to the latest available
            // type so we read the right offsets. If the image is later
            // upgraded to a CUDA toolkit with newer kernel records,
            // bump this cast.
            CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)r;
            add_activity_duration(k->start, k->end);
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            PHANTORA_CUPTI_MEMCPY_RECORD *m = (PHANTORA_CUPTI_MEMCPY_RECORD *)r;
            add_activity_duration(m->start, m->end);
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMCPY2) {
            CUpti_ActivityMemcpyPtoP4 *m = (CUpti_ActivityMemcpyPtoP4 *)r;
            add_activity_duration(m->start, m->end);
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            CUpti_ActivityMemset4 *m = (CUpti_ActivityMemset4 *)r;
            add_activity_duration(m->start, m->end);
        }
    }
    free(buffer);
}

int phantora_cupti_init(void)
{
    static int initialised = 0;
    if (initialised) return 0;
    CUptiResult r1 = cuptiActivityRegisterCallbacks(buf_requested, buf_completed);
    if (r1 != CUPTI_SUCCESS) return (int)r1;
    CUptiResult r2 = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (r2 != CUPTI_SUCCESS) return (int)r2;
    CUptiResult r3 = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    if (r3 != CUPTI_SUCCESS) return (int)r3;
    CUptiResult r4 = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2);
    if (r4 != CUPTI_SUCCESS) return (int)r4;
    CUptiResult r5 = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    if (r5 != CUPTI_SUCCESS) return (int)r5;
    initialised = 1;
    return 0;
}

void phantora_cupti_clear(void)
{
    atomic_store_explicit(&g_activity_ns, 0, memory_order_release);
}

void phantora_cupti_flush(void)
{
    cuptiActivityFlushAll(0);
}

uint64_t phantora_cupti_sum_activity_ns(void)
{
    return atomic_load_explicit(&g_activity_ns, memory_order_acquire);
}
