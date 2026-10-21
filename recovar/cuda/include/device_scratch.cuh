/*
 * Stream-ordered scratch for FFI launcher temporaries.
 *
 * Every launcher in this directory allocated its temporaries with cudaMalloc and
 * released them with cudaFree on the way out. An nsys capture of K=1 100k/256
 * (em_work/codex/vdam_nsys_20260920, job 14175588) measured 108 081 cudaMalloc and
 * 108 084 cudaFree calls in a 200 s window -- about 25 700 pairs per EM iteration --
 * costing 7.6 s of host CUDA API time. cudaFree is worse than its 55.9 us average
 * suggests: it synchronizes the whole device, so each one also drains the pipeline of
 * a run that is already launch-bound, issuing about 870 000 kernel launches per
 * iteration against 9.7 s of actual kernel time.
 *
 * These temporaries are scratch: written and consumed inside one launcher call and
 * never read afterwards, with sizes that repeat from call to call because they come
 * from bucket shapes that repeat. CUDA's stream-ordered allocator is built for
 * exactly that. It pools freed blocks and hands them back to a later allocation on
 * the same stream without a driver round trip.
 *
 * Why not a hand-rolled free list: `cudaFree` synchronizes the device, and code here
 * legitimately relies on that -- the VDAM M-step runs worker lanes on several
 * streams. A plain free list would hand a buffer to a second stream while the first
 * still had pending work on it. `cudaFreeAsync` orders the release on the stream that
 * used the buffer, so the driver, not this file, is responsible for that hazard.
 *
 * The device pool's default release threshold is zero, which returns memory to the
 * driver at every synchronization and would give back most of the benefit, so the
 * threshold is raised once per device on first use.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace recovar {

// Retain up to this much pooled scratch per device before releasing to the driver.
// Scratch peaks in the low hundreds of MB at 100k/256; this leaves headroom without
// competing with the XLA allocator for a meaningful share of an 80 GB device.
constexpr size_t kScratchPoolRetainBytes = 2ull << 30;

namespace detail {

inline cudaError_t configure_scratch_pool()
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    cudaMemPool_t pool = nullptr;
    err = cudaDeviceGetDefaultMemPool(&pool, device);
    if (err != cudaSuccess) return err;
    uint64_t threshold = kScratchPoolRetainBytes;
    return cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

inline cudaError_t scratch_pool_ready()
{
    // Per device, and only once: a device is configured the first time a launcher on
    // it asks for scratch. A failure here is not fatal -- the allocation still works,
    // it just returns memory to the driver more eagerly -- so it is not propagated.
    static thread_local int configured_device = -1;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    if (device != configured_device) {
        configure_scratch_pool();
        configured_device = device;
    }
    return cudaSuccess;
}

}  // namespace detail

// Drop-in for cudaMalloc, plus the stream the buffer will be used on.
inline cudaError_t scratch_alloc(void** out, size_t bytes, cudaStream_t stream)
{
    if (out == nullptr) return cudaErrorInvalidValue;
    *out = nullptr;
    if (bytes == 0) return cudaSuccess;
    cudaError_t err = detail::scratch_pool_ready();
    if (err != cudaSuccess) return err;
    return cudaMallocAsync(out, bytes, stream);
}

// Drop-in for cudaFree. Must name the stream the buffer was used on.
inline cudaError_t scratch_free(void* ptr, cudaStream_t stream)
{
    if (ptr == nullptr) return cudaSuccess;
    return cudaFreeAsync(ptr, stream);
}

}  // namespace recovar
