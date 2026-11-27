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


    ThreeChannelSamplesView::ThreeChannelSamplesView(std::span<const std::array<float, 3>> rows)
        : _samples(rows) {
    }

    ThreeChannelSamplesView::ThreeChannelSamplesView(std::span<const float> scalars)
        : _samples(std::as_bytes(scalars)) {
    }

    ThreeChannelSamplesView::ThreeChannelSamplesView(std::span<const std::byte> scalarBytes)
        : _samples(scalarBytes) {
    }

    std::span<const std::byte> ThreeChannelSamplesView::object_bytes() const noexcept {
        static_assert(sizeof(std::array<float, 3>) == 3u * sizeof(float));
        static_assert(std::is_trivially_copyable_v<std::array<float, 3>>);
        if (const auto* rows = std::get_if<std::span<const std::array<float, 3>>>(&_samples)) {
            return std::as_bytes(*rows);
        }
        return *std::get_if<std::span<const std::byte>>(&_samples);
    }

    std::size_t ThreeChannelSamplesView::sample_count() const noexcept {
        if (const auto* rows = std::get_if<std::span<const std::array<float, 3>>>(&_samples)) {
            return rows->size();
        }
        return std::get_if<std::span<const std::byte>>(&_samples)->size() / sizeof(std::array<float, 3>);
    }

    std::size_t ThreeChannelSamplesView::scalar_count() const noexcept {
        if (const auto* rows = std::get_if<std::span<const std::array<float, 3>>>(&_samples)) {
            return rows->size() * 3u;
        }
        return std::get_if<std::span<const std::byte>>(&_samples)->size() / sizeof(float);
    }

    std::array<float, 3> ThreeChannelSamplesView::sample(std::size_t index) const {
        if (const auto* rows = std::get_if<std::span<const std::array<float, 3>>>(&_samples)) {
            return (*rows)[index];
        }
        const auto bytes = *std::get_if<std::span<const std::byte>>(&_samples);
        std::array<float, 3> result{};
        std::memcpy(result.data(), bytes.data() + index * sizeof(result), sizeof(result));
        return result;
    }

    static bool overwrite_array_locked(
        Resources& resources,
        const HostToDeviceCopyRequest& request,
        float*& dst,
        const float* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        Failure& outError);

    static bool alloc_and_upload_array(Resources& resources, float*& dst, const float* src, int n, void* cudaStreamOpaque, const char* label, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!src || n <= 0) {
            outError.diagnostic = std::string(label) + " array is empty";
            return false;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        if (!allocate_owned_device(
                resources,
                reinterpret_cast<void**>(&dst),
                bytes,
                label,
                outError)) {
            return false;
        }
        if (!enqueue_host_to_device_copy(
                resources,
                HostToDeviceCopyRequest{"alloc_and_upload_array", label},
                dst,
                src,
                bytes,
                cudaStreamOpaque,
                outError)) {
            free_owned_device_noexcept(resources, dst);
            dst = nullptr;
            return false;
        }
        return true;
    }

    static bool relock_resources_after_offlock_upload(
        Resources& resources,
        std::unique_lock<std::mutex>* resourcesLock,
        Failure& outError) {
        if (!resourcesLock) {
            return true;
        }
        resourcesLock->lock();
        Failure ownerError;
        if (!validate_resource_owner_locked(resources, ownerError)) {
            outError = std::move(ownerError);
            return false;
        }
        reap_retire_queue_locked(resources);
        return true;
    }

    bool alloc_and_upload_bytes(
        Resources& resources,
        void*& dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!src || bytes == 0) {
            outError.diagnostic = std::string(label ? label : "buffer") + " buffer is empty";
            return false;
        }

        if (!allocate_owned_device(
                resources,
                &dst,
                bytes,
                label ? label : "buffer",
                outError)) {
            return false;
        }

        if (!enqueue_host_to_device_copy(
                resources,
                HostToDeviceCopyRequest{"alloc_and_upload_bytes", label},
                dst,
                src,
                bytes,
                cudaStreamOpaque,
                outError)) {
            free_owned_device_noexcept(resources, dst);
            dst = nullptr;
            return false;
        }

        return true;
    }

    bool alloc_and_upload_bytes_locked(
        Resources& resources,
        void*& dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_bytes(resources, dst, src, bytes, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            if (dst) {
                free_owned_device_noexcept(resources, dst);
                dst = nullptr;
            }
            return false;
        }
        return allocOk;
    }

    bool upload_array_locked(
        Resources& resources,
        float*& dst,
        int currentN,
        const float* src,
        int n,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!src || n <= 0) {
            outError.diagnostic = std::string(label ? label : "array") + " array is empty";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        if (dst && currentN == n) {
            return overwrite_array_locked(resources, {"upload_array_locked", label}, dst, src, bytes, cudaStreamOpaque, resourcesLock, outError);
        }

        float* tmp = nullptr;
        resourcesLock.unlock();
        const bool allocOk =
            alloc_and_upload_array(resources, tmp, src, n, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, &resourcesLock, outError)) {
            if (tmp) {
                free_owned_device_noexcept(resources, tmp);
            }
            return false;
        }
        if (!allocOk) {
            return false;
        }

        const size_t oldBytes = static_cast<size_t>(std::max(0, currentN)) * sizeof(float);
        if (dst) {
            if (!retire_ptr_locked(resources, dst, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                free_owned_device_noexcept(resources, tmp);
                return false;
            }
        }

        dst = tmp;
        return true;
    }


    static bool overwrite_curve_locked(
        Resources& resources,
        DeviceCurve& dst,
        const CurveUploadView& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        const char* label,
        Failure& outError);

    static bool alloc_and_upload_curve(
        Resources& resources,
        DeviceCurve& dst,
        const CurveUploadView& src,
        void* cudaStreamOpaque,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const int n = static_cast<int>(src.axis.size());
        const size_t bytes = src.axis.size_bytes();
        dst.domainBegin = 0;
        dst.domainEnd = n - 1;
        if (!allocate_owned_device(
                resources,
                reinterpret_cast<void**>(&dst.x),
                bytes,
                "curve.x",
                outError)) {
            free_curve(resources, dst);
            return false;
        }
        if (!allocate_owned_device(
                resources,
                reinterpret_cast<void**>(&dst.y),
                bytes,
                "curve.y",
                outError)) {
            free_curve(resources, dst);
            return false;
        }

        if (!enqueue_host_to_device_copy(
                resources,
                HostToDeviceCopyRequest{"alloc_and_upload_curve", "curve.x"},
                dst.x,
                src.axis.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            free_curve(resources, dst);
            outError.diagnostic = std::string("curve.x upload failed: ") + outError.diagnostic;
            return false;
        }
        if (!enqueue_host_to_device_copy(
                resources,
                HostToDeviceCopyRequest{"alloc_and_upload_curve", "curve.y"},
                dst.y,
                src.samples.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            free_curve(resources, dst);
            outError.diagnostic = std::string("curve.y upload failed: ") + outError.diagnostic;
            return false;
        }

        dst.n = n;
        return true;
    }

    bool upload_curve_locked(
        Resources& resources,
        DeviceCurve& dst,
        const CurveUploadView& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const size_t sampleCount = src.axis.size();
        if (sampleCount == 0 || src.samples.size() != sampleCount) {
            outError.diagnostic = std::string(label) + ": curve has no samples or mismatched arrays";
            return false;
        }
        if (sampleCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            sampleCount > std::numeric_limits<size_t>::max() / sizeof(float)) {
            outError.diagnostic = std::string(label) + ": curve sample count invalid";
            return false;
        }
        const int n = static_cast<int>(sampleCount);

        // If the allocation matches, update in place to avoid alloc/free churn (common during slider scrubs).
        if (dst.x && dst.y && dst.n == n) {
            if (!overwrite_curve_locked(resources, dst, src, cudaStreamOpaque, resourcesLock, label, outError)) {
                return false;
            }

            dst.domainBegin = 0;
            dst.domainEnd = n - 1;
            dst.n = n;
            return true;
        }

        DeviceCurve tmp{};
        resourcesLock.unlock();
        const bool allocOk =
            alloc_and_upload_curve(
                resources,
                tmp,
                src,
                cudaStreamOpaque,
                outError);
        if (!relock_resources_after_offlock_upload(resources, &resourcesLock, outError)) {
            free_curve(resources, tmp);
            return false;
        }
        if (!allocOk) {
            outError.diagnostic = std::string(label) + ": " + outError.diagnostic;
            return false;
        }

        const size_t oldBytes = static_cast<size_t>(std::max(0, dst.n)) * sizeof(float);
        if (dst.x) {
            if (!retire_ptr_locked(resources, dst.x, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                free_curve(resources, tmp);
                return false;
            }
            dst.x = nullptr;
        }
        if (dst.y) {
            if (!retire_ptr_locked(resources, dst.y, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                free_curve(resources, tmp);
                return false;
            }
            dst.y = nullptr;
        }

        dst = tmp;
        return true;
    }

    static bool alloc_and_upload_spectral_samples(Resources& resources, DeviceCurve& dst, const std::vector<float>& src, void* cudaStreamOpaque, const char* label, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (src.empty()) {
            outError.diagnostic = std::string(label) + " array is empty";
            return false;
        }
        const int n = static_cast<int>(src.size());
        if (n <= 0) {
            outError.diagnostic = std::string(label) + " sample count invalid";
            return false;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        if (!allocate_owned_device(
                resources,
                reinterpret_cast<void**>(&dst.y),
                bytes,
                label,
                outError)) {
            free_curve(resources, dst);
            return false;
        }
        if (!enqueue_host_to_device_copy(
                resources,
                HostToDeviceCopyRequest{"alloc_and_upload_spectral_samples", label},
                dst.y,
                src.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            free_curve(resources, dst);
            return false;
        }
        dst.n = n;
        dst.domainBegin = 0;
        dst.domainEnd = n - 1;
        return true;
    }

    bool upload_spectral_samples_locked(
        Resources& resources,
        DeviceCurve& dst,
        const std::vector<float>& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (src.empty()) {
            outError.diagnostic = std::string(label ? label : "spectral samples") + " array is empty";
            return false;
        }

        const int n = static_cast<int>(src.size());
        if (n <= 0) {
            outError.diagnostic = std::string(label ? label : "spectral samples") + " sample count invalid";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        if (dst.y && dst.n == n) {
            if (!overwrite_array_locked(resources, {"upload_spectral_samples_locked", label}, dst.y, src.data(), bytes, cudaStreamOpaque, resourcesLock, outError)) {
                return false;
            }
            dst.n = n;
            dst.domainBegin = 0;
            dst.domainEnd = n - 1;
            return true;
        }

        DeviceCurve tmp{};
        resourcesLock.unlock();
        const bool allocOk =
            alloc_and_upload_spectral_samples(resources, tmp, src, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, &resourcesLock, outError)) {
            free_curve(resources, tmp);
            return false;
        }
        if (!allocOk) {
            return false;
        }

        if (!retire_curve_locked(resources, dst, cudaStreamOpaque, label, outError)) {
            free_curve(resources, tmp);
            return false;
        }

        dst = tmp;
        return true;
    }


    static bool validate_overwrite_owner_locked(
        Resources& resources,
        const ResourceManager::DeviceContextKey& key,
        std::uint64_t epoch,
        Failure& outError) {
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        if (resources.contextInvalidatedByProvenLoss || resources.contextEpoch != epoch ||
            !(resources.ownerContextKey == key)) {
            outError.diagnostic = "CUDA resource owner changed during upload staging";
            return false;
        }
        return true;
    }

    static bool overwrite_allocations_match_locked(
        Resources& resources,
        std::initializer_list<float*> pointers,
        std::size_t bytes) {
        std::lock_guard<std::mutex> lock(resources.deviceAllocationRecordsMutex);
        return std::all_of(pointers.begin(), pointers.end(), [&](float* pointer) {
            const auto record = resources.deviceAllocationRecords.find(pointer);
            return record != resources.deviceAllocationRecords.end() && record->second.bytes() == bytes &&
                   record->second.state() == DeviceReservationState::Committed;
        });
    }

    static bool overwrite_array_locked(
        Resources& resources,
        const HostToDeviceCopyRequest& request,
        float*& dst,
        const float* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        Failure& outError) {
        const auto key = resources.ownerContextKey;
        const auto epoch = resources.contextEpoch;
        float* const destination = dst;
        const PinnedUploadContextKey stagingKey{key.deviceId, key.contextOpaque, epoch};
        StagedHostUpload staged;
        Failure stagingError;

        resourcesLock.unlock();
        const bool stagedOk = stage_host_to_device_copy(stagingKey, request, src, bytes, staged, stagingError);
        resourcesLock.lock();
        if (!validate_overwrite_owner_locked(resources, key, epoch, outError)) {
            return false;
        }
        if (dst != destination || !overwrite_allocations_match_locked(resources, {dst}, bytes)) {
            outError.diagnostic = std::string(request.label) + ": CUDA upload destination changed during staging";
            return false;
        }
        if (!wait_for_frame_use_events_locked(resources, cudaStreamOpaque, request.label, outError)) {
            return false;
        }
        if (!stagedOk) {
            outError = std::move(stagingError);
            return false;
        }
        return enqueue_staged_host_to_device_copy(resources, request, dst, bytes, cudaStreamOpaque, staged, true, outError);
    }

    static bool overwrite_curve_locked(
        Resources& resources,
        DeviceCurve& dst,
        const CurveUploadView& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        const char* label,
        Failure& outError) {
        const auto key = resources.ownerContextKey;
        const auto epoch = resources.contextEpoch;
        const DeviceCurve destination = dst;
        const std::size_t bytes = src.axis.size_bytes();
        const PinnedUploadContextKey stagingKey{key.deviceId, key.contextOpaque, epoch};
        const char* baseLabel = label ? label : "curve";
        const std::string labelX = std::string(baseLabel) + ".x";
        const std::string labelY = std::string(baseLabel) + ".y";
        const HostToDeviceCopyRequest axisRequest{"upload_curve_locked", labelX.c_str()};
        const HostToDeviceCopyRequest sampleRequest{"upload_curve_locked", labelY.c_str()};
        StagedHostUpload axis;
        StagedHostUpload samples;
        Failure axisError;
        Failure sampleError;

        resourcesLock.unlock();
        const bool axisStaged = stage_host_to_device_copy(stagingKey, axisRequest, src.axis.data(), bytes, axis, axisError);
        bool samplesStaged = axisStaged && stage_host_to_device_copy(stagingKey, sampleRequest, src.samples.data(), bytes, samples, sampleError);
        resourcesLock.lock();
        if (!validate_overwrite_owner_locked(resources, key, epoch, outError)) {
            return false;
        }
        if (dst.x != destination.x || dst.y != destination.y || dst.n != destination.n ||
            !overwrite_allocations_match_locked(resources, {dst.x, dst.y}, bytes)) {
            outError.diagnostic = std::string(baseLabel) + ": CUDA upload destination changed during staging";
            return false;
        }

        // Registration also takes resources.m. Fence the current list and keep
        // it locked through both overwrites, including exceptional completion.
        if (!wait_for_frame_use_events_locked(resources, cudaStreamOpaque, baseLabel, outError)) {
            outError.diagnostic = labelX + " upload failed: " + outError.diagnostic;
            return false;
        }
        if (!axisStaged) {
            outError = std::move(axisError);
        }
        if (!axisStaged ||
            !enqueue_staged_host_to_device_copy(resources, axisRequest, dst.x, bytes, cudaStreamOpaque, axis, true, outError)) {
            outError.diagnostic = labelX + " upload failed: " + outError.diagnostic;
            return false;
        }

        // A pair can exhaust the staging cap while X is Reserved. In that
        // case retain the original post-X reservation opportunity for Y.
        if (samples.capExceeded) {
            samplesStaged = stage_host_to_device_copy(stagingKey, sampleRequest, src.samples.data(), bytes, samples, sampleError);
        }
        if (!samplesStaged ||
            !enqueue_staged_host_to_device_copy(resources, sampleRequest, dst.y, bytes, cudaStreamOpaque, samples, true, outError)) {
            if (!samplesStaged) {
                outError = std::move(sampleError);
            }
            outError.diagnostic = labelY + " upload failed: " + outError.diagnostic;
            return false;
        }
        return true;
    }


} // namespace JuicerCuda
