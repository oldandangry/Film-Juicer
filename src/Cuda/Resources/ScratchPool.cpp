#include "Cuda/Resources/ResourceInternals.h"

#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "SpectralData.h"
#include "GamutCompression.h"
#include "Scanner.h"
#include "JuicerDirBoundary.h"
#include "juicer_host_preparation.h"

#include "Logging.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <numeric>
#include <atomic>
#include <span>
#include <sstream>
#include <exception>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace JuicerCuda {

    static void free_tracked_device_ptr_locked(
        Resources& resources,
        void*& ptr,
        void* cudaStreamOpaque) noexcept {
        (void)cudaStreamOpaque;
        if (!ptr) {
            return;
        }
        free_owned_device_noexcept(resources, ptr);
        ptr = nullptr;
    }

    template <typename T>
    static void free_tracked_device_ptr_locked(
        Resources& resources,
        T*& ptr,
        void* cudaStreamOpaque) noexcept {
        void* raw = reinterpret_cast<void*>(ptr);
        free_tracked_device_ptr_locked(resources, raw, cudaStreamOpaque);
        ptr = reinterpret_cast<T*>(raw);
    }

    static bool allocate_scratch_device_ptr_locked(
        Resources& resources,
        void*& outPtr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        (void)cudaStreamOpaque;
        outPtr = nullptr;
        if (bytes == 0) {
            outError.diagnostic = std::string(label ? label : "scratch") + " bytes invalid";
            return false;
        }

        return allocate_owned_device(
            resources,
            &outPtr,
            bytes,
            label ? label : "scratch",
            outError);
    }

    template <typename T>
    static bool allocate_scratch_device_ptr_locked(
        Resources& resources,
        T*& outPtr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        void* raw = nullptr;
        const bool ok = allocate_scratch_device_ptr_locked(
            resources,
            raw,
            bytes,
            cudaStreamOpaque,
            label,
            outError);
        outPtr = reinterpret_cast<T*>(raw);
        return ok;
    }

    void free_optics_scratch(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque) noexcept {
        free_tracked_device_ptr_locked(resources, s.rgbR, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rgbG, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rgbB, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.blurred, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.aux, cudaStreamOpaque);
        free_tracked_device_ptr_locked(
            resources,
            s.grainFrameUniforms,
            cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmp, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmpShared, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.gateTransmittance, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filmDustTransmittance, cudaStreamOpaque);
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateTransmittanceCapacityElements = 0;
        s.filmDustCapacityElements = 0;
    }

    static bool retire_optics_scratch_locked(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque, const char* label, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (s.filmDustTransmittance || s.gateTransmittance) {
            outError.diagnostic = "whole-optics retirement requires staged defect attachment retirement";
            return false;
        }
        const size_t planeBytes = s.capacityElements * sizeof(float);
        if (s.rgbR) {
            if (!retire_ptr_locked(resources, s.rgbR, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rgbR = nullptr;
        }
        if (s.rgbG) {
            if (!retire_ptr_locked(resources, s.rgbG, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rgbG = nullptr;
        }
        if (s.rgbB) {
            if (!retire_ptr_locked(resources, s.rgbB, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rgbB = nullptr;
        }
        if (s.blurred) {
            if (!retire_ptr_locked(resources, s.blurred, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.blurred = nullptr;
        }
        if (s.aux) {
            if (!retire_ptr_locked(resources, s.aux, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.aux = nullptr;
        }
        if (s.grainFrameUniforms) {
            if (!retire_ptr_locked(
                    resources,
                    s.grainFrameUniforms,
                    sizeof(GrainFrameUniforms),
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    label,
                    outError,
                    true)) {
                return false;
            }
            s.grainFrameUniforms = nullptr;
        }
        if (s.grainTmp) {
            if (!retire_ptr_locked(resources, s.grainTmp, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.grainTmp = nullptr;
        }
        if (s.grainTmpShared) {
            if (!retire_ptr_locked(resources, s.grainTmpShared, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.grainTmpShared = nullptr;
        }

        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateTransmittanceCapacityElements = 0;
        s.filmDustCapacityElements = 0;
        return true;
    }

    void free_spatial_dir_scratch(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque) noexcept {
        free_tracked_device_ptr_locked(resources, s.rawCorrectionY, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rawCorrectionM, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rawCorrectionC, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filteredCorrectionY, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filteredCorrectionM, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filteredCorrectionC, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filterTempM, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filterTempC, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.logRawB, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.logRawG, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.logRawR, cudaStreamOpaque);
        s.filterTemp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
    }

    static bool retire_spatial_dir_scratch_locked(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque, const char* label, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const size_t bytes = s.capacityElements * sizeof(float);
        if (s.rawCorrectionY) {
            if (!retire_ptr_locked(resources, s.rawCorrectionY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rawCorrectionY = nullptr;
        }
        if (s.rawCorrectionM) {
            if (!retire_ptr_locked(resources, s.rawCorrectionM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rawCorrectionM = nullptr;
        }
        if (s.rawCorrectionC) {
            if (!retire_ptr_locked(resources, s.rawCorrectionC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rawCorrectionC = nullptr;
        }
        if (s.filteredCorrectionY) {
            if (!retire_ptr_locked(resources, s.filteredCorrectionY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filteredCorrectionY = nullptr;
        }
        if (s.filteredCorrectionM) {
            if (!retire_ptr_locked(resources, s.filteredCorrectionM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filteredCorrectionM = nullptr;
        }
        if (s.filteredCorrectionC) {
            if (!retire_ptr_locked(resources, s.filteredCorrectionC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filteredCorrectionC = nullptr;
        }
        if (s.filterTempM) {
            if (!retire_ptr_locked(resources, s.filterTempM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filterTempM = nullptr;
        }
        if (s.filterTempC) {
            if (!retire_ptr_locked(resources, s.filterTempC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filterTempC = nullptr;
        }
        if (s.logRawB) {
            if (!retire_ptr_locked(resources, s.logRawB, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.logRawB = nullptr;
        }
        if (s.logRawG) {
            if (!retire_ptr_locked(resources, s.logRawG, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.logRawG = nullptr;
        }
        if (s.logRawR) {
            if (!retire_ptr_locked(resources, s.logRawR, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.logRawR = nullptr;
        }
        s.filterTemp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        return true;
    }

    void free_shared_tmp_plane(Resources& resources) noexcept {
        free_tracked_device_ptr_locked(
            resources,
            resources.sharedTmpPlane,
            nullptr);
        resources.sharedTmpPlane = nullptr;
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;
    }

    static bool optics_base_live_locked(const Resources& resources) noexcept {
        const auto& scratch = resources.scannerScratch;
        return scratch.rgbR || scratch.rgbG || scratch.rgbB ||
               scratch.grainFrameUniforms;
    }

    static bool optics_any_live_locked(const Resources::DeviceOpticsScratch& scratch) noexcept {
        return scratch.rgbR || scratch.rgbG || scratch.rgbB ||
               scratch.blurred || scratch.aux ||
               scratch.grainTmp || scratch.grainTmpShared ||
               scratch.grainFrameUniforms ||
               scratch.gateTransmittance || scratch.filmDustTransmittance;
    }

    static std::size_t optics_stage_live_bytes_locked(
        const Resources::DeviceOpticsScratch& scratch) noexcept {
        const std::size_t planeBytes = scratch.capacityElements * sizeof(float);
        std::size_t bytes = 0;
        auto add_plane = [&](const float* ptr) noexcept {
            if (ptr) {
                bytes += planeBytes;
            }
        };
        add_plane(scratch.rgbR);
        add_plane(scratch.rgbG);
        add_plane(scratch.rgbB);
        add_plane(scratch.blurred);
        add_plane(scratch.aux);
        if (scratch.grainFrameUniforms) {
            bytes += sizeof(GrainFrameUniforms);
        }
        add_plane(scratch.grainTmp);
        add_plane(scratch.grainTmpShared);
        if (scratch.filmDustTransmittance) {
            bytes += scratch.filmDustCapacityElements * sizeof(float);
        }
        if (scratch.gateTransmittance) {
            bytes += scratch.gateTransmittanceCapacityElements * sizeof(float);
        }
        return bytes;
    }

    static bool spatial_dir_base_live_locked(const Resources& resources) noexcept {
        const auto& scratch = resources.spatialDirScratch;
        return scratch.rawCorrectionY || scratch.rawCorrectionM || scratch.rawCorrectionC ||
               scratch.filteredCorrectionY || scratch.filteredCorrectionM ||
               scratch.filteredCorrectionC || scratch.filterTempM || scratch.filterTempC ||
               scratch.logRawB || scratch.logRawG || scratch.logRawR;
    }

    static std::size_t spatial_dir_stage_live_bytes_locked(
        const Resources::DeviceSpatialDirScratch& scratch) noexcept {
        const std::size_t planeBytes = scratch.capacityElements * sizeof(float);
        std::size_t bytes = 0;
        auto add_plane = [&](const float* ptr) noexcept {
            if (ptr) {
                bytes += planeBytes;
            }
        };
        add_plane(scratch.rawCorrectionY);
        add_plane(scratch.rawCorrectionM);
        add_plane(scratch.rawCorrectionC);
        add_plane(scratch.filteredCorrectionY);
        add_plane(scratch.filteredCorrectionM);
        add_plane(scratch.filteredCorrectionC);
        add_plane(scratch.filterTempM);
        add_plane(scratch.filterTempC);
        add_plane(scratch.logRawB);
        add_plane(scratch.logRawG);
        add_plane(scratch.logRawR);
        return bytes;
    }

    static bool spatial_dir_scratch_has_required_roles_locked(
        const Resources::DeviceSpatialDirScratch& scratch,
        const Spektrafilm::DirScratchPlaneRoles& planeRoles) noexcept {
        const bool hasThreeRaw =
            scratch.rawCorrectionY && scratch.rawCorrectionM && scratch.rawCorrectionC;
        const bool rawCorrectionMatch =
            (planeRoles.rawCorrectionPlanes == 3 && hasThreeRaw) ||
            (planeRoles.rawCorrectionPlanes == 1 && scratch.rawCorrectionY);
        const bool tier1Base =
            rawCorrectionMatch &&
            scratch.filteredCorrectionY && scratch.filteredCorrectionM && scratch.filteredCorrectionC;
        if (!tier1Base) {
            return false;
        }
        const bool hasAliasedForwardTemps =
            scratch.filterTempM && scratch.filterTempC;
        const bool channelTempsMatch =
            planeRoles.filterTempPlanes == 1 ||
            (planeRoles.filterTempPlanes == 2 && scratch.filterTempM) ||
            (planeRoles.filterTempPlanes == 3 && hasAliasedForwardTemps);
        const bool cachedLogRawMatch =
            planeRoles.cachedLogRawPlanes == 0 ||
            (planeRoles.cachedLogRawPlanes == 2 &&
             scratch.logRawB && scratch.logRawG) ||
            (planeRoles.cachedLogRawPlanes == 3 &&
             scratch.logRawB && scratch.logRawG && scratch.logRawR);
        return channelTempsMatch && cachedLogRawMatch;
    }

    static bool spatial_dir_scratch_has_retained_final_roles_locked(
        const Resources::DeviceSpatialDirScratch& scratch,
        const Spektrafilm::DirScratchPlaneRoles& planeRoles) noexcept {
        if (!scratch.filteredCorrectionY || !scratch.filteredCorrectionM ||
            !scratch.filteredCorrectionC) {
            return false;
        }
        const bool cachedLogRawCompatible =
            (planeRoles.cachedLogRawPlanes == 0 &&
             scratch.logRawB == nullptr && scratch.logRawG == nullptr && scratch.logRawR == nullptr) ||
            (planeRoles.cachedLogRawPlanes == 2 && scratch.logRawR == nullptr) ||
            (planeRoles.cachedLogRawPlanes == 3);
        if (!cachedLogRawCompatible) {
            return false;
        }
        const bool filterTempsCompatible =
            planeRoles.filterTempPlanes == 3 ||
            (planeRoles.filterTempPlanes == 2 && !scratch.filterTempC) ||
            (!scratch.filterTempM && !scratch.filterTempC);
        return filterTempsCompatible;
    }

    static bool spatial_dir_request_needs_shared_tmp_plane(
        const Spektrafilm::DirScratchPlaneRoles& planeRoles) noexcept {
        return planeRoles.filterTempPlanes > 0;
    }

    static bool retire_shared_tmp_plane_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!resources.sharedTmpPlane) {
            return true;
        }

        const std::size_t bytes = resources.sharedTmpCapacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                resources.sharedTmpPlane,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                label ? label : "shared tmp plane",
                outError,
                true)) {
            return false;
        }
        resources.sharedTmpPlane = nullptr;
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;
        resources.scannerScratch.tmp = nullptr;
        resources.spatialDirScratch.filterTemp = nullptr;
        return true;
    }

    static bool retire_orphaned_shared_tmp_plane_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!resources.sharedTmpPlane) {
            return true;
        }
        if (optics_base_live_locked(resources) || spatial_dir_base_live_locked(resources)) {
            return true;
        }
        return retire_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            label ? label : "orphaned shared tmp plane",
            outError);
    }


    bool acquire_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        void* cudaStreamOpaque,
        Failure& outError) {
        outError = {};
        if (leaseGeneration == 0) {
            outError.diagnostic = "retained frame scratch lease generation is invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        reap_retire_queue_locked(resources);

        if (resources.retainedScratchLeaseGeneration == leaseGeneration) {
            return true;
        }
        if (resources.retainedScratchLeaseGeneration != 0) {
            outError.diagnostic =
                "retained frame scratch lease is already active under the serialized render contract";
            return false;
        }
        if (!wait_for_frame_use_events_locked(
                resources,
                cudaStreamOpaque,
                "retained frame scratch lease",
                outError)) {
            return false;
        }
        resources.retainedScratchLeaseGeneration = leaseGeneration;
        return true;
    }

    bool release_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        Failure& outError) {
        outError = {};
        if (leaseGeneration == 0) {
            outError.diagnostic = "retained frame scratch lease generation is invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (resources.retainedScratchLeaseGeneration == 0) {
            return true;
        }
        if (resources.retainedScratchLeaseGeneration != leaseGeneration) {
            outError.diagnostic = "retained frame scratch lease generation mismatch";
            return false;
        }
        resources.retainedScratchLeaseGeneration = 0;
        return true;
    }

    bool ensure_retained_spatial_dir_cached_log_raw_stage(
        Resources& resources,
        std::uint64_t leaseGeneration,
        const ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        SpatialDirCachedLogRawStageStats& outStats,
        Failure& outError) {
        outStats = SpatialDirCachedLogRawStageStats{};
        outError = {};
        if (leaseGeneration == 0) {
            outError.diagnostic = "retained spatial DIR cached log raw stage lease generation is invalid";
            return false;
        }
        if (!ResourceManager::scratch_request_descriptor_is_valid(scratchRequest)) {
            outError.diagnostic = "retained spatial DIR cached log raw stage request invalid";
            return false;
        }
        if (scratchRequest.requestedWidth <= 0 || scratchRequest.requestedHeight <= 0) {
            outError.diagnostic = "retained spatial DIR cached log raw stage dimensions invalid";
            return false;
        }

        const std::size_t widthElements = static_cast<std::size_t>(scratchRequest.requestedWidth);
        const std::size_t heightElements = static_cast<std::size_t>(scratchRequest.requestedHeight);
        if (widthElements > (std::numeric_limits<std::size_t>::max() / heightElements)) {
            outError.diagnostic = "retained spatial DIR cached log raw stage element count overflow";
            return false;
        }
        const std::size_t requiredElements = widthElements * heightElements;
        if (requiredElements == 0 ||
            requiredElements > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            outError.diagnostic = "retained spatial DIR cached log raw stage byte count overflow";
            return false;
        }
        const std::size_t bytes = requiredElements * sizeof(float);

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        reap_retire_queue_locked(resources);
        if (resources.retainedScratchLeaseGeneration != leaseGeneration) {
            outError.diagnostic = "retained spatial DIR cached log raw stage lease generation mismatch";
            return false;
        }

        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
        if (scratch.width != scratchRequest.requestedWidth ||
            scratch.height != scratchRequest.requestedHeight ||
            scratch.capacityElements < requiredElements ||
            !scratch.filteredCorrectionY || !scratch.filteredCorrectionM ||
            !scratch.filteredCorrectionC) {
            outError.diagnostic = "retained spatial DIR cached log raw stage missing filtered correction residency";
            return false;
        }
        if (scratch.logRawB && scratch.logRawG && scratch.logRawR) {
            return true;
        }

        auto ensure_plane = [&](float*& ptr, const char* label) {
            if (ptr) {
                return true;
            }
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    ptr,
                    bytes,
                    cudaStreamOpaque,
                    label,
                    outError)) {
                return false;
            }
            outStats.cachedLogRawAllocatedBytes += bytes;
            return true;
        };

        if (!ensure_plane(scratch.logRawB, "retained spatial DIR final logRawB") ||
            !ensure_plane(scratch.logRawG, "retained spatial DIR final logRawG") ||
            !ensure_plane(scratch.logRawR, "retained spatial DIR final logRawR")) {
            free_tracked_device_ptr_locked(resources, scratch.logRawB, cudaStreamOpaque);
            free_tracked_device_ptr_locked(resources, scratch.logRawG, cudaStreamOpaque);
            free_tracked_device_ptr_locked(resources, scratch.logRawR, cudaStreamOpaque);
            return false;
        }
        return true;
    }

    static bool prepare_defect_attachments(Resources& resources,
                                           const ResourceManager::ScratchRequestDescriptor& request,
                                           std::uint64_t expectedLease,
                                           void* streamOpaque,
                                           Failure& outError);

    bool reclaim_large_scratch_transition(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        LargeScratchTransitionReclaimStats& outStats,
        Failure& outError) {
        outStats = LargeScratchTransitionReclaimStats{};
        outError = {};
        if (!ResourceManager::scratch_request_descriptor_is_valid(scratchRequest)) {
            outError.diagnostic = "large scratch transition request descriptor is invalid";
            return false;
        }
        const std::size_t requestedWidth = static_cast<std::size_t>(scratchRequest.requestedWidth);
        const std::size_t requestedHeight = static_cast<std::size_t>(scratchRequest.requestedHeight);
        if (requestedHeight != 0 &&
            requestedWidth > (std::numeric_limits<std::size_t>::max() / requestedHeight)) {
            outError.diagnostic = "large scratch transition element count overflow";
            return false;
        }
        const std::size_t requiredElements = requestedWidth * requestedHeight;

        bool retireAttachments = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (resources.retainedScratchLeaseGeneration != 0) {
                return true;
            }
            const auto& optics = resources.scannerScratch;
            const bool baseLive = optics.rgbR && optics.rgbG && optics.rgbB;
            retireAttachments = !scratchRequest.needOptics || optics.capacityElements < requiredElements ||
                                (scratchRequest.aliasScannerRgbFromSpatialDirFiltered ? baseLive : !baseLive);
            if (retireAttachments) {
                outStats.opticsRetiredBytes =
                    (optics.filmDustTransmittance ? optics.filmDustCapacityElements * sizeof(float) : 0) +
                    (optics.gateTransmittance ? optics.gateTransmittanceCapacityElements * sizeof(float) : 0);
            }
        }
        if (retireAttachments && !prepare_defect_attachments(
                                     resources, ResourceManager::ScratchRequestDescriptor{}, 0, cudaStreamOpaque, outError)) {
            return false;
        }

        bool syncBeforeReap = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (!validate_resource_owner_locked(resources, outError)) {
                return false;
            }
            reap_retire_queue_locked(resources);
            outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
            if (resources.retainedScratchLeaseGeneration != 0) {
                return true;
            }

            Resources::DeviceOpticsScratch& optics = resources.scannerScratch;
            if (optics_any_live_locked(optics)) {
                const bool baseLive = optics.rgbR && optics.rgbG && optics.rgbB;
                const bool capacityMatch = optics.capacityElements >= requiredElements;
                const bool retireOptics =
                    !scratchRequest.needOptics ||
                    !capacityMatch ||
                    (scratchRequest.aliasScannerRgbFromSpatialDirFiltered ? baseLive : !baseLive);
                if (retireOptics) {
                    outStats.opticsRetiredBytes += optics_stage_live_bytes_locked(optics);
                    if (!retire_optics_scratch_locked(
                            resources,
                            optics,
                            cudaStreamOpaque,
                            "large scratch transition optics scratch",
                            outError)) {
                        return false;
                    }
                }
            }

            Resources::DeviceSpatialDirScratch& spatialDir = resources.spatialDirScratch;
            const std::size_t spatialDirBytesBefore =
                spatial_dir_stage_live_bytes_locked(spatialDir);
            if (spatialDirBytesBefore > 0) {
                const bool capacityMatch = spatialDir.capacityElements >= requiredElements;
                const bool rolesMatch =
                    scratchRequest.needSpatialDir &&
                    spatial_dir_scratch_has_required_roles_locked(
                        spatialDir,
                        scratchRequest.spatialDirPlaneRoles);
                const bool retireSpatialDir =
                    !scratchRequest.needSpatialDir || !capacityMatch || !rolesMatch;
                if (retireSpatialDir) {
                    outStats.spatialDirRetiredBytes = spatialDirBytesBefore;
                    if (!retire_spatial_dir_scratch_locked(
                            resources,
                            spatialDir,
                            cudaStreamOpaque,
                            "large scratch transition spatial DIR scratch",
                            outError)) {
                        return false;
                    }
                }
            }

            const std::size_t sharedTmpBytesBefore =
                resources.sharedTmpPlane ? resources.sharedTmpCapacityElements * sizeof(float) : 0;
            if (!retire_orphaned_shared_tmp_plane_locked(
                    resources,
                    cudaStreamOpaque,
                    "large scratch transition shared tmp plane",
                    outError)) {
                return false;
            }
            if (sharedTmpBytesBefore > 0 && !resources.sharedTmpPlane) {
                outStats.sharedTmpRetiredBytes = sharedTmpBytesBefore;
            }

            syncBeforeReap =
                outStats.pendingScratchBytesBefore > 0 ||
                outStats.opticsRetiredBytes > 0 ||
                outStats.spatialDirRetiredBytes > 0 ||
                outStats.sharedTmpRetiredBytes > 0;
        }

        if (!syncBeforeReap) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError.status = runtime_failure_status(syncErr);
            outError.diagnostic = std::string("cudaStreamSynchronize(large scratch transition) failed: ") +
                                  (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return reap_retired_allocations(resources, outStats.reclaimedBytes, outError);
    }


    bool shed_retained_scratch_after_frame(
        Resources& resources,
        void* cudaStreamOpaque,
        PostFrameScratchShedStats& outStats,
        Failure& outError) {
        outStats = PostFrameScratchShedStats{};
        outError = {};

        bool retireAttachments = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (resources.retainedScratchLeaseGeneration != 0) {
                return true;
            }
            const auto& optics = resources.scannerScratch;
            retireAttachments = true;
            if (retireAttachments) {
                outStats.opticsRetiredBytes =
                    (optics.filmDustTransmittance ? optics.filmDustCapacityElements * sizeof(float) : 0) +
                    (optics.gateTransmittance ? optics.gateTransmittanceCapacityElements * sizeof(float) : 0);
            }
        }
        if (retireAttachments && !prepare_defect_attachments(
                                     resources, ResourceManager::ScratchRequestDescriptor{}, 0, cudaStreamOpaque, outError)) {
            return false;
        }

        bool syncBeforeReap = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (!validate_resource_owner_locked(resources, outError)) {
                return false;
            }
            reap_retire_queue_locked(resources);
            outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
            if (resources.retainedScratchLeaseGeneration != 0) {
                return true;
            }

            Resources::DeviceOpticsScratch& optics = resources.scannerScratch;
            if (optics_any_live_locked(optics)) {
                outStats.opticsRetiredBytes += optics_stage_live_bytes_locked(optics);
                if (!retire_optics_scratch_locked(
                        resources,
                        optics,
                        cudaStreamOpaque,
                        "post-frame optics scratch",
                        outError)) {
                    return false;
                }
            }

            Resources::DeviceSpatialDirScratch& spatialDir = resources.spatialDirScratch;
            outStats.spatialDirRetiredBytes = spatial_dir_stage_live_bytes_locked(spatialDir);
            if (outStats.spatialDirRetiredBytes > 0 &&
                !retire_spatial_dir_scratch_locked(
                    resources,
                    spatialDir,
                    cudaStreamOpaque,
                    "post-frame spatial DIR scratch",
                    outError)) {
                return false;
            }

            const std::size_t sharedTmpBytesBefore =
                resources.sharedTmpPlane ? resources.sharedTmpCapacityElements * sizeof(float) : 0;
            if (!retire_orphaned_shared_tmp_plane_locked(
                    resources,
                    cudaStreamOpaque,
                    "post-frame shared tmp plane",
                    outError)) {
                return false;
            }
            if (sharedTmpBytesBefore > 0 && !resources.sharedTmpPlane) {
                outStats.sharedTmpRetiredBytes = sharedTmpBytesBefore;
            }

            syncBeforeReap =
                outStats.pendingScratchBytesBefore > 0 ||
                outStats.opticsRetiredBytes > 0 ||
                outStats.spatialDirRetiredBytes > 0 ||
                outStats.sharedTmpRetiredBytes > 0;
        }

        if (!syncBeforeReap) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError.status = runtime_failure_status(syncErr);
            outError.diagnostic = std::string("cudaStreamSynchronize(post-frame scratch shed) failed: ") +
                                  (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return reap_retired_allocations(resources, outStats.reclaimedBytes, outError);
    }

    static bool ensure_shared_tmp_plane_locked(Resources& resources, int width, int height, void* cudaStreamOpaque, const char* label, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (width <= 0 || height <= 0) {
            outError.diagnostic = std::string(label ? label : "shared tmp") + " dimensions invalid";
            return false;
        }

        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (resources.sharedTmpPlane &&
            resources.sharedTmpCapacityElements >= requiredElements) {
            resources.sharedTmpWidth = width;
            resources.sharedTmpHeight = height;
            return true;
        }

        if (resources.sharedTmpPlane) {
            const size_t oldBytes = resources.sharedTmpCapacityElements * sizeof(float);
            if (!retire_ptr_locked(resources, resources.sharedTmpPlane, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label ? label : "shared tmp", outError, true)) {
                return false;
            }
            resources.sharedTmpPlane = nullptr;
        }
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;

        const size_t bytes = requiredElements * sizeof(float);
        if (!allocate_scratch_device_ptr_locked(
                resources,
                resources.sharedTmpPlane,
                bytes,
                cudaStreamOpaque,
                label ? label : "shared tmp",
                outError)) {
            resources.sharedTmpPlane = nullptr;
            return false;
        }

        resources.sharedTmpWidth = width;
        resources.sharedTmpHeight = height;
        resources.sharedTmpCapacityElements = requiredElements;
        return true;
    }

    static bool prepare_defect_attachments(Resources& resources,
                                           const ResourceManager::ScratchRequestDescriptor& request,
                                           std::uint64_t expectedLease,
                                           void* streamOpaque,
                                           Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        ResourceManager::DeviceContextKey context{};
        std::uint64_t epoch = 0;
        float* previous[2]{};
        std::size_t capacity[2]{};
        const std::size_t desired[2] = {
            request.needFilmDustTransmittance ? static_cast<std::size_t>(request.requestedWidth) * request.requestedHeight : 0,
            request.needGateTransmittance ? static_cast<std::size_t>(request.gateWidth) * request.gateHeight : 0};
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (resources.retainedScratchLeaseGeneration != expectedLease ||
                (expectedLease == 0 && (desired[0] != 0 || desired[1] != 0))) {
                outError.diagnostic = "defect attachment lease mismatch";
                return false;
            }
            context = resources.ownerContextKey;
            epoch = resources.contextEpoch;
            previous[0] = resources.scannerScratch.filmDustTransmittance;
            previous[1] = resources.scannerScratch.gateTransmittance;
            capacity[0] = resources.scannerScratch.filmDustCapacityElements;
            capacity[1] = resources.scannerScratch.gateTransmittanceCapacityElements;
        }
        auto& events = resources.defectRetirementEvents;
        for (auto& event : events) {
            if (const auto status = event.destroy(); status != cudaSuccess) {
                set_failure(outError, runtime_failure_status(status), "defect attachment retirement event destruction failed");
                return false;
            }
        }
        struct PendingAttachments {
            Resources& owner;
            Failure& failure;
            void* pointers[2]{};
            ~PendingAttachments() {
                for (int i = 0; i < 2; ++i) {
                    if (pointers[i]) {
                        free_owned_device_noexcept(owner, pointers[i]);
                    }
                    const auto status = owner.defectRetirementEvents[i].destroy();
                    record_cleanup_context_loss(failure, runtime_failure_status(status));
                }
            }
        } pending{resources, outError};
        bool replace[2]{};
        for (int i = 0; i < 2; ++i) {
            if (desired[i] > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
                outError.diagnostic = "defect attachment byte overflow";
                return false;
            }
            replace[i] = desired[i] == 0 ? previous[i] != nullptr : !previous[i] || capacity[i] < desired[i];
            if (!replace[i]) {
                continue;
            }
            if (desired[i] && !allocate_owned_device(resources, &pending.pointers[i], desired[i] * sizeof(float), i == 0 ? "film dust transmittance" : "gate transmittance", outError)) {
                return false;
            }
            if (previous[i]) {
                cudaEvent_t event = nullptr;
                auto status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                events[i] = RetiredEvent(event);
                if (status == cudaSuccess) {
                    status = ResourceOperations::record_event(event, reinterpret_cast<cudaStream_t>(streamOpaque));
                }
                if (status != cudaSuccess) {
                    set_failure(outError, runtime_failure_status(status), "defect attachment retirement fence failed");
                    return false;
                }
            }
        }
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (resources.retainedScratchLeaseGeneration != expectedLease || resources.contextEpoch != epoch ||
                !(resources.ownerContextKey == context) || resources.scannerScratch.filmDustTransmittance != previous[0] ||
                resources.scannerScratch.gateTransmittance != previous[1]) {
                outError.diagnostic = "defect attachment publication identity mismatch";
                return false;
            }
            // State precedes allocation bookkeeping; no CUDA call occurs under either lock.
            std::lock_guard<std::mutex> recordsLock(resources.deviceAllocationRecordsMutex);
            std::size_t retiringBytes = 0;
            for (int i = 0; i < 2; ++i) {
                if (replace[i] && previous[i]) {
                    const auto record = resources.deviceAllocationRecords.find(previous[i]);
                    if (record == resources.deviceAllocationRecords.end() || record->second.bytes() != capacity[i] * sizeof(float)) {
                        outError.diagnostic = "defect attachment retirement inventory mismatch";
                        return false;
                    }
                    retiringBytes += capacity[i] * sizeof(float);
                }
            }
            if (retiringBytes > std::numeric_limits<std::size_t>::max() - resources.retireBytes ||
                retiringBytes > std::numeric_limits<std::size_t>::max() - resources.retireScratchBytes) {
                outError.diagnostic = "defect attachment retirement byte overflow";
                return false;
            }
            try {
                resources.retireQueue.reserve(resources.retireQueue.size() + 2);
            } catch (...) {
                JuicerLogging::discard_current_exception();
                outError.diagnostic = "defect attachment retirement reservation failed";
                return false;
            }
            for (int i = 0; i < 2; ++i) {
                if (replace[i] && previous[i]) {
                    auto& reservation = resources.deviceAllocationRecords.find(previous[i])->second;
                    if (!reservation.mark_retiring(outError)) {
                        return false;
                    }
                }
            }
            for (int i = 0; i < 2; ++i) {
                if (!replace[i]) {
                    continue;
                }
                if (previous[i]) {
                    Resources::RetireEntry entry{};
                    entry.bytes = capacity[i] * sizeof(float);
                    entry.kind = Resources::RetireKind::DeviceFree;
                    entry.scratchTier = true;
                    entry.completion = std::move(events[i]);
                    entry.completionRecorded = true;
                    auto record = resources.deviceAllocationRecords.find(previous[i]);
                    entry.deviceBuffer = std::move(record->second);
                    resources.deviceAllocationRecords.erase(record);
                    resources.retireQueue.push_back(std::move(entry));
                }
                if (i == 0) {
                    resources.scannerScratch.filmDustTransmittance = static_cast<float*>(pending.pointers[i]);
                    resources.scannerScratch.filmDustCapacityElements = desired[i];
                } else {
                    resources.scannerScratch.gateTransmittance = static_cast<float*>(pending.pointers[i]);
                    resources.scannerScratch.gateTransmittanceCapacityElements = desired[i];
                }
                pending.pointers[i] = nullptr;
            }
            resources.retireBytes += retiringBytes;
            resources.retireScratchBytes += retiringBytes;
            resources.scannerScratch.gateWidth = request.gateWidth;
            resources.scannerScratch.gateHeight = request.gateHeight;
        }
        return true;
    }

    bool ensure_optics_scratch(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& request,
        std::uint64_t expectedLease,
        void* cudaStreamOpaque,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const int width = request.requestedWidth;
        const int height = request.requestedHeight;
        const bool needBlurredScratch = request.needBlurred;
        const bool aliasScannerRgbFromSpatialDirFiltered =
            request.aliasScannerRgbFromSpatialDirFiltered;
        const bool needAuxScratch = request.needAux;
        const bool needSharedTmpScratch = request.needSharedTmp;
        const bool needGrainFrameUniforms = request.needGrainFrameUniforms;
        const bool needGrainLayerWorkScratch = request.needGrainLayerWork;
        const bool needGrainSharedScratch = request.needGrainShared;

        if (width <= 0 || height <= 0) {
            outError.diagnostic = "optics scratch dimensions invalid";
            return false;
        }

        bool replaceBase = false;
        {
            std::lock_guard<std::mutex> snapshotLock(resources.m);
            const auto& scratch = resources.scannerScratch;
            replaceBase = scratch.capacityElements < static_cast<std::size_t>(width) * height ||
                          (!aliasScannerRgbFromSpatialDirFiltered && (!scratch.rgbR || !scratch.rgbG || !scratch.rgbB));
        }
        if (replaceBase) {
            auto cleared = request;
            cleared.needFilmDustTransmittance = false;
            cleared.needGateTransmittance = false;
            cleared.gateWidth = cleared.gateHeight = 0;
            if (!prepare_defect_attachments(resources, cleared, expectedLease, cudaStreamOpaque, outError)) {
                return false;
            }
        }

        std::unique_lock<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }

        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        const bool capacityMatch = resources.scannerScratch.capacityElements >= requiredElements;
        const bool needScannerRgbScratch = !aliasScannerRgbFromSpatialDirFiltered;
        const bool haveBase =
            !needScannerRgbScratch ||
            (resources.scannerScratch.rgbR && resources.scannerScratch.rgbG &&
             resources.scannerScratch.rgbB);

        if (!capacityMatch || !haveBase) {
            if (resources.scannerScratch.rgbR || resources.scannerScratch.rgbG || resources.scannerScratch.rgbB ||
                resources.scannerScratch.blurred || resources.scannerScratch.aux || resources.scannerScratch.grainTmp ||
                resources.scannerScratch.grainTmpShared ||
                resources.scannerScratch.grainFrameUniforms ||
                resources.scannerScratch.gateTransmittance || resources.scannerScratch.filmDustTransmittance) {
                if (!retire_optics_scratch_locked(resources, resources.scannerScratch, cudaStreamOpaque, "optics scratch", outError)) {
                    return false;
                }
            } else {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
            }

            const size_t bytes = requiredElements * sizeof(float);
            if (needScannerRgbScratch) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.rgbR,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.rgbR",
                        outError)) {
                    free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                    return false;
                }
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.rgbG,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.rgbG",
                        outError)) {
                    free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                    return false;
                }
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.rgbB,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.rgbB",
                        outError)) {
                    free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                    return false;
                }
            }

            resources.scannerScratch.capacityElements = requiredElements;
        }
        resources.scannerScratch.width = width;
        resources.scannerScratch.height = height;

        if (needSharedTmpScratch) {
            if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                return false;
            }
        }
        resources.scannerScratch.tmp = needSharedTmpScratch ? resources.sharedTmpPlane : nullptr;
        const size_t planeBytes = resources.scannerScratch.capacityElements * sizeof(float);

        if (!needScannerRgbScratch) {
            auto retire_rgb_plane = [&](float*& ptr, const char* label) {
                if (!ptr) {
                    return true;
                }
                if (!retire_ptr_locked(
                        resources,
                        ptr,
                        planeBytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        label,
                        outError,
                        true)) {
                    return false;
                }
                ptr = nullptr;
                return true;
            };
            if (!retire_rgb_plane(resources.scannerScratch.rgbR, "scannerScratch.rgbR alias release") ||
                !retire_rgb_plane(resources.scannerScratch.rgbG, "scannerScratch.rgbG alias release") ||
                !retire_rgb_plane(resources.scannerScratch.rgbB, "scannerScratch.rgbB alias release")) {
                return false;
            }
        }

        if (needBlurredScratch) {
            if (!resources.scannerScratch.blurred) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.blurred,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.blurred",
                        outError)) {
                    return false;
                }
            }
        } else {
            if (resources.scannerScratch.blurred) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.blurred, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unsharp scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.blurred = nullptr;
            }
        }

        if (needAuxScratch) {
            if (!resources.scannerScratch.aux) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.aux,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.aux",
                        outError)) {
                    return false;
                }
            }
        } else {
            if (resources.scannerScratch.aux) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.aux, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.aux = nullptr;
            }
        }

        if (needGrainFrameUniforms) {
            if (!resources.scannerScratch.grainFrameUniforms) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainFrameUniforms,
                        sizeof(GrainFrameUniforms),
                        cudaStreamOpaque,
                        "scannerScratch.grainFrameUniforms",
                        outError)) {
                    return false;
                }
            }
        } else if (resources.scannerScratch.grainFrameUniforms) {
            if (!retire_ptr_locked(
                    resources,
                    resources.scannerScratch.grainFrameUniforms,
                    sizeof(GrainFrameUniforms),
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "grain frame uniforms",
                    outError,
                    true)) {
                return false;
            }
            resources.scannerScratch.grainFrameUniforms = nullptr;
        }

        if (needGrainLayerWorkScratch) {
            if (!resources.scannerScratch.grainTmp) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmp,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmp",
                        outError)) {
                    return false;
                }
            }
        } else {
            if (resources.scannerScratch.grainTmp) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmp, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain mix scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.grainTmp = nullptr;
            }
        }

        if (needGrainSharedScratch) {
            if (!resources.scannerScratch.grainTmpShared) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmpShared,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmpShared",
                        outError)) {
                    return false;
                }
            }
        } else {
            if (resources.scannerScratch.grainTmpShared) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmpShared, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain shared scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.grainTmpShared = nullptr;
            }
        }

        lock.unlock();
        return prepare_defect_attachments(resources, request, expectedLease, cudaStreamOpaque, outError);
    }

    bool ensure_spatial_dir_scratch(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& request,
        void* cudaStreamOpaque,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const int width = request.requestedWidth;
        const int height = request.requestedHeight;
        const Spektrafilm::DirScratchTier scratchTier = request.spatialDirScratchTier;
        const Spektrafilm::DirScratchPlaneRoles& planeRoles = request.spatialDirPlaneRoles;
        if (width <= 0 || height <= 0) {
            outError.diagnostic = "spatial DIR scratch dimensions invalid";
            return false;
        }
        if (!Spektrafilm::spatial_dir_roles_match_tier(scratchTier, planeRoles)) {
            outError.diagnostic = "spatial DIR scratch request tier/roles invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }


        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        const bool dimsMatch = (scratch.width == width && scratch.height == height);
        const bool capacityMatch = scratch.capacityElements >= requiredElements;
        const bool needSharedTmp = spatial_dir_request_needs_shared_tmp_plane(planeRoles);
        const bool haveRequiredRoles =
            spatial_dir_scratch_has_required_roles_locked(scratch, planeRoles);
        if (dimsMatch && capacityMatch && haveRequiredRoles) {
            if (needSharedTmp) {
                if (!ensure_shared_tmp_plane_locked(
                        resources,
                        width,
                        height,
                        cudaStreamOpaque,
                        "shared tmp plane",
                        outError)) {
                    return false;
                }
                scratch.filterTemp = resources.sharedTmpPlane;
            } else {
                scratch.filterTemp = nullptr;
            }
            return true;
        }

        const bool haveRetainedFinalRoles =
            spatial_dir_scratch_has_retained_final_roles_locked(scratch, planeRoles);
        if (dimsMatch && capacityMatch && haveRetainedFinalRoles) {
            const size_t bytes = requiredElements * sizeof(float);
            auto ensure_plane = [&](float*& ptr, const char* label) {
                if (ptr) {
                    return true;
                }
                return allocate_scratch_device_ptr_locked(
                    resources,
                    ptr,
                    bytes,
                    cudaStreamOpaque,
                    label,
                    outError);
            };
            if (!ensure_plane(scratch.rawCorrectionY, "spatial DIR rawCorrectionY") ||
                (planeRoles.rawCorrectionPlanes == 3 &&
                 (!ensure_plane(scratch.rawCorrectionM, "spatial DIR rawCorrectionM") ||
                  !ensure_plane(scratch.rawCorrectionC, "spatial DIR rawCorrectionC"))) ||
                (planeRoles.filterTempPlanes == 2 &&
                 !ensure_plane(scratch.filterTempM, "spatial DIR filterTempM")) ||
                (planeRoles.filterTempPlanes == 3 &&
                 (!ensure_plane(scratch.filterTempM, "spatial DIR filterTempM") ||
                  !ensure_plane(scratch.filterTempC, "spatial DIR filterTempC"))) ||
                (planeRoles.cachedLogRawPlanes >= 1 &&
                 !ensure_plane(scratch.logRawB, "spatial DIR logRawB")) ||
                (planeRoles.cachedLogRawPlanes >= 2 &&
                 !ensure_plane(scratch.logRawG, "spatial DIR logRawG")) ||
                (planeRoles.cachedLogRawPlanes >= 3 &&
                 !ensure_plane(scratch.logRawR, "spatial DIR logRawR"))) {
                free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
                return false;
            }
            if (needSharedTmp) {
                if (!ensure_shared_tmp_plane_locked(
                        resources,
                        width,
                        height,
                        cudaStreamOpaque,
                        "shared tmp plane",
                        outError)) {
                    free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
                    return false;
                }
                scratch.filterTemp = resources.sharedTmpPlane;
            } else {
                scratch.filterTemp = nullptr;
            }
            scratch.width = width;
            scratch.height = height;
            scratch.capacityElements = requiredElements;
            return true;
        }

        if (scratch.rawCorrectionY || scratch.rawCorrectionM || scratch.rawCorrectionC ||
            scratch.filteredCorrectionY || scratch.filteredCorrectionM ||
            scratch.filteredCorrectionC || scratch.filterTempM || scratch.filterTempC ||
            scratch.logRawB || scratch.logRawG || scratch.logRawR) {
            if (!capacityMatch || !haveRequiredRoles) {
                if (!retire_spatial_dir_scratch_locked(resources, scratch, cudaStreamOpaque, "spatial DIR scratch", outError)) {
                    return false;
                }
            }
        }

        if (capacityMatch && haveRequiredRoles) {
            scratch.width = width;
            scratch.height = height;
            if (needSharedTmp) {
                if (!ensure_shared_tmp_plane_locked(
                        resources,
                        width,
                        height,
                        cudaStreamOpaque,
                        "shared tmp plane",
                        outError)) {
                    return false;
                }
                scratch.filterTemp = resources.sharedTmpPlane;
            } else {
                scratch.filterTemp = nullptr;
            }
            return true;
        }

        const size_t bytes = requiredElements * sizeof(float);
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.rawCorrectionY,
                bytes,
                cudaStreamOpaque,
                "spatial DIR rawCorrectionY",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.filteredCorrectionY,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filteredCorrectionY",
                outError) ||
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.filteredCorrectionM,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filteredCorrectionM",
                outError) ||
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.filteredCorrectionC,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filteredCorrectionC",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.rawCorrectionPlanes == 3) {
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    scratch.rawCorrectionM,
                    bytes,
                    cudaStreamOpaque,
                    "spatial DIR rawCorrectionM",
                    outError) ||
                !allocate_scratch_device_ptr_locked(
                    resources,
                    scratch.rawCorrectionC,
                    bytes,
                    cudaStreamOpaque,
                    "spatial DIR rawCorrectionC",
                    outError)) {
                free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
                return false;
            }
        }
        if (planeRoles.filterTempPlanes >= 2 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.filterTempM,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filterTempM",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.filterTempPlanes == 3 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.filterTempC,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filterTempC",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.cachedLogRawPlanes >= 1 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.logRawB,
                bytes,
                cudaStreamOpaque,
                "spatial DIR logRawB",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.cachedLogRawPlanes >= 2 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.logRawG,
                bytes,
                cudaStreamOpaque,
                "spatial DIR logRawG",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.cachedLogRawPlanes >= 3 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.logRawR,
                bytes,
                cudaStreamOpaque,
                "spatial DIR logRawR",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (needSharedTmp) {
            if (!ensure_shared_tmp_plane_locked(
                    resources,
                    width,
                    height,
                    cudaStreamOpaque,
                    "shared tmp plane",
                    outError)) {
                free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
                return false;
            }
            scratch.filterTemp = resources.sharedTmpPlane;
        } else {
            scratch.filterTemp = nullptr;
        }

        scratch.width = width;
        scratch.height = height;
        scratch.capacityElements = requiredElements;
        return true;
    }


} // namespace JuicerCuda
