// Cuda/JuicerCudaResourcesInternal.h
//
// Internal seam declarations for JuicerCudaResources single-TU split sections.
#pragma once

#include "Cuda/JuicerCudaResources.h"

#include <cstddef>
#include <string>

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#endif

namespace JuicerCuda {

static inline bool validate_resource_owner_locked(Resources& resources, std::string& outError, bool bindIfUnset = true) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)resources;
    (void)bindIfUnset;
    outError = "CUDA is not enabled";
    return false;
#else
    int cur = -1;
    const cudaError_t devErr = cudaGetDevice(&cur);
    if (devErr != cudaSuccess || cur < 0) {
        outError = std::string("cudaGetDevice failed: ")
            + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
        return false;
    }
    if (bindIfUnset && resources.deviceId < 0) {
        resources.deviceId = cur;
    }
    if (resources.deviceId != cur) {
        outError = "CUDA device mismatch for cached resources";
        return false;
    }
    return true;
#endif
}

static bool enqueue_host_to_device_copy(
    const char* stage,
    const char* label,
    void* dst,
    const void* src,
    std::size_t bytes,
    void* cudaStreamOpaque,
    std::string& outError);

static void free_stbn(Resources& resources) noexcept;
static void free_wang(Resources& resources) noexcept;
static void free_scan_error_flag(Resources& resources) noexcept;
static void free_tables(Resources& resources) noexcept;
static bool retire_tables_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError);
static void free_scan_medium(Resources::DeviceScanMedium& m) noexcept;
static bool retire_scan_medium_locked(Resources& resources, Resources::DeviceScanMedium& m, void* cudaStreamOpaque, const char* label, std::string& outError);
static void free_scan_lut(Resources::DeviceSpectralLut& lut) noexcept;
static void free_gaussian_kernel(Resources::DeviceGaussianKernel& k) noexcept;
static bool is_async_device_ptr_tracked_locked(const Resources& resources, const void* ptr) noexcept;
static void untrack_async_device_ptr_locked(Resources& resources, void* ptr) noexcept;
static void free_auto_exposure(Resources& resources) noexcept;
static void free_optics_scratch(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque) noexcept;
static void free_spatial_dir_scratch(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque) noexcept;
static void free_shared_tmp_plane(Resources& resources) noexcept;

} // namespace JuicerCuda
