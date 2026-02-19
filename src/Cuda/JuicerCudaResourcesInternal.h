// Cuda/JuicerCudaResourcesInternal.h
//
// Internal seam declarations for JuicerCudaResources single-TU split sections.
#pragma once

#include "Cuda/JuicerCudaResources.h"

#include <cstddef>
#include <string>

namespace JuicerCuda {

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
