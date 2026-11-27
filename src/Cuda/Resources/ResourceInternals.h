#pragma once

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaResourceOperations.h"

namespace JuicerCuda {

    struct PinnedUploadContextKey {
        int deviceId = -1;
        void* contextOpaque = nullptr;

        std::uint64_t epoch = 0;

        bool operator==(const PinnedUploadContextKey&) const noexcept = default;
    };


    struct PinnedUploadReservation {
        bool staged = false;
        PinnedUploadContextKey key{};
        std::uint64_t blockId = 0;
        void* stagingPtr = nullptr;
        void* doneEventOpaque = nullptr;
    };

    struct StagedHostUpload {
        PinnedUploadReservation reservation;
        bool capExceeded = false;

        StagedHostUpload() = default;
        StagedHostUpload(const StagedHostUpload&) = delete;
        StagedHostUpload& operator=(const StagedHostUpload&) = delete;
        ~StagedHostUpload() noexcept;
    };

    struct CurveUploadView {
        std::span<const float> axis;
        std::span<const float> samples;
    };

    bool validate_resource_owner_locked(Resources& resources, Failure& outError);

    bool allocate_tracked_device_buffer(
        Resources& resources,
        std::size_t bytes,
        const char* label,
        TrackedDeviceAllocations::node_type& node,
        Failure& outError);

    bool allocate_owned_device(
        Resources& resources,
        void** outPtr,
        std::size_t bytes,
        const char* label,
        Failure& outError);
    void free_owned_device_noexcept(Resources& resources, void* ptr) noexcept;
    void reap_retire_queue_locked(Resources& resources) noexcept;
    bool wait_for_frame_use_events_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError);
    bool retire_ptr_locked(Resources& resources, void* ptr, std::size_t bytes, Resources::RetireKind kind, void* cudaStreamOpaque, const char* label, Failure& outError, bool scratchTier = false, TrackedDeviceBuffer* externalAllocation = nullptr);
    bool stage_host_to_device_copy(
        const PinnedUploadContextKey& key,
        const HostToDeviceCopyRequest& request,
        const void* src,
        std::size_t bytes,
        StagedHostUpload& staged,
        Failure& outError) noexcept;
    bool enqueue_staged_host_to_device_copy(
        Resources& resources,
        const HostToDeviceCopyRequest& request,
        void* dst,
        std::size_t bytes,
        void* cudaStreamOpaque,
        StagedHostUpload& staged,
        bool overwrite,
        Failure& outError) noexcept;
    bool alloc_and_upload_bytes(
        Resources& resources,
        void*& dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError);
    bool alloc_and_upload_bytes_locked(
        Resources& resources,
        void*& dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        Failure& outError);
    bool upload_array_locked(
        Resources& resources,
        float*& dst,
        int currentN,
        const float* src,
        int n,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        const char* label,
        Failure& outError);
    bool upload_curve_locked(
        Resources& resources,
        DeviceCurve& dst,
        const CurveUploadView& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        const char* label,
        Failure& outError);
    bool upload_spectral_samples_locked(
        Resources& resources,
        DeviceCurve& dst,
        const std::vector<float>& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        const char* label,
        Failure& outError);
    void free_curve(Resources& resources, DeviceCurve& c) noexcept;
    bool retire_curve_locked(Resources& resources, DeviceCurve& c, void* cudaStreamOpaque, const char* label, Failure& outError);
    void free_density_layers(Resources& resources) noexcept;
    void free_film_tc_lut(Resources& resources) noexcept;
    void free_output_gamut_table(Resources& resources) noexcept;
    void free_mallett_basis(Resources& resources) noexcept;
    void free_print_payloads(Resources& resources) noexcept;
    void free_stbn(Resources& resources) noexcept;
    void free_wang(Resources& resources) noexcept;
    void free_tables(Resources& resources) noexcept;
    void free_optics_scratch(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque = nullptr) noexcept;
    void free_spatial_dir_scratch(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque = nullptr) noexcept;
    void free_shared_tmp_plane(Resources& resources) noexcept;
    void clear_gaussian_kernel_views(Resources& resources) noexcept;
    void free_gaussian_cache(Resources& resources) noexcept;

} // namespace JuicerCuda
