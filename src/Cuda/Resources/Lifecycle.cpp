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
#include <cassert>
#include <cstdio>
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

    static bool drain_invalidated_host_allocations(Resources& resources, Failure& outError) noexcept;

    static bool free_scan_error_buffers(Resources& resources, Failure& outError) noexcept;

    static bool validate_resource_context_locked(
        Resources& resources,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (resources.contextInvalidatedByProvenLoss) {
            set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "CUDA resource owner epoch was invalidated");
            return false;
        }
        int cur = -1;
        const cudaError_t devErr = cudaGetDevice(&cur);
        if (devErr != cudaSuccess || cur < 0) {
            set_failure(outError, runtime_failure_status(devErr), "cudaGetDevice failed for cached resources");
            return false;
        }
        if (resources.ownerContextKey.deviceId != cur ||
            resources.deviceId != cur || resources.contextEpoch == 0 ||
            !resources.deviceLedger) {
            outError.diagnostic = "CUDA device mismatch for cached resources";
            return false;
        }
        void* currentContextOpaque = nullptr;
        std::string contextError;
        int contextCode = 0;
        bool queried = false;
        try {
            queried = query_current_cuda_context(currentContextOpaque, contextError, &contextCode);
        } catch (...) {
            if (contextCode == 0) {
                throw;
            }
        }
        if (!queried || currentContextOpaque != resources.ownerContextKey.contextOpaque) {
            set_failure(outError, driver_failure_status(contextCode), contextError.empty() ? "CUDA context query or identity mismatch for cached resources" : std::string_view(contextError));
            return false;
        }
        return true;
    }

    bool validate_resource_owner_locked(Resources& resources, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!validate_resource_context_locked(resources, outError)) {
            return false;
        }
        if (resources.frameUseFenceQuarantined) {
            outError.diagnostic = "CUDA resources quarantined after unfenced frame use";
            return false;
        }
        return true;
    }

    struct DeferredDestroyQueue {
        Resources* head = nullptr;

        void push(Resources* resources) noexcept {
            resources->deferredDestroyNext = head;
            head = resources;
        }

        Resources* pop() noexcept {
            auto* resources = head;
            head = resources->deferredDestroyNext;
            resources->deferredDestroyNext = nullptr;
            return resources;
        }

        std::size_t size() const noexcept {
            std::size_t count = 0;
            for (auto* entry = head; entry; entry = entry->deferredDestroyNext) {
                ++count;
            }
            return count;
        }

        void extract(const ResourceManager::DeviceContextKey& key, std::uint64_t epoch, DeferredDestroyQueue& ready) noexcept {
            auto** link = &head;
            while (*link) {
                auto* entry = *link;
                if (entry->ownerContextKey == key && entry->contextEpoch == epoch) {
                    *link = entry->deferredDestroyNext;
                    ready.push(entry);
                } else {
                    link = &entry->deferredDestroyNext;
                }
            }
        }

        void extract_invalidated(DeferredDestroyQueue& ready) noexcept {
            auto** link = &head;
            while (*link) {
                auto* entry = *link;
                if (entry->contextInvalidatedByProvenLoss) {
                    *link = entry->deferredDestroyNext;
                    ready.push(entry);
                } else {
                    link = &entry->deferredDestroyNext;
                }
            }
        }
    };

    static DeferredDestroyQueue& deferred_destroy_queue() noexcept {
        static DeferredDestroyQueue queue;
        return queue;
    }

    // Only queue links are protected. Nonthrowing acquisition allows an extracted
    // batch to restore ownership even during allocation or mutex-exception unwind.
    class DeferredDestroyLock final {
    public:
        DeferredDestroyLock() noexcept {
            while (_held.test_and_set(std::memory_order_acquire)) {
                _held.wait(true, std::memory_order_relaxed);
            }
        }
        ~DeferredDestroyLock() {
            _held.clear(std::memory_order_release);
            _held.notify_one();
        }
        DeferredDestroyLock(const DeferredDestroyLock&) = delete;
        DeferredDestroyLock& operator=(const DeferredDestroyLock&) = delete;

    private:
        static inline std::atomic_flag _held = ATOMIC_FLAG_INIT;
    };

    struct DeferredDestroyBatch {
        DeferredDestroyBatch() = default;
        DeferredDestroyBatch(const DeferredDestroyBatch&) = delete;
        DeferredDestroyBatch& operator=(const DeferredDestroyBatch&) = delete;

        DeferredDestroyQueue entries;

        ~DeferredDestroyBatch() {
            const DeferredDestroyLock lock;
            while (entries.head) {
                deferred_destroy_queue().push(entries.pop());
            }
        }
    };

    static void defer_destroy(Resources* resources) noexcept {
        const DeferredDestroyLock lock;
        deferred_destroy_queue().push(resources);
    }

    static bool query_current_cuda_device(int& outDeviceId, std::string& outError) {
        outDeviceId = -1;
        outError.clear();
        const cudaError_t err = cudaGetDevice(&outDeviceId);
        if (err != cudaSuccess || outDeviceId < 0) {
            outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            outDeviceId = -1;
            return false;
        }
        return true;
    }

    static bool owner_matches_current(
        int ownerDeviceId,
        void* ownerContextOpaque,
        int currentDeviceId,
        bool currentDeviceValid,
        void* currentContextOpaque,
        bool currentContextValid,
        bool& outDeviceMatch,
        bool& outContextMatch) {
        outDeviceMatch = (ownerDeviceId < 0) ||
                         (currentDeviceValid && currentDeviceId == ownerDeviceId);
        outContextMatch = (ownerContextOpaque == nullptr) ||
                          (currentContextValid && currentContextOpaque == ownerContextOpaque);
        return outDeviceMatch && outContextMatch;
    }

    static void trace_teardown_event(
        const char* stage,
        const char* outcome,
        int ownerDeviceId,
        void* ownerContextOpaque,
        int currentDeviceId,
        void* currentContextOpaque,
        bool deviceMatch,
        bool contextMatch,
        bool switchAttempted,
        bool switchSucceeded,
        bool managerRetireAttempted,
        bool managerRetireAccepted,
        std::size_t deferredQueueDepth,
        const std::string& detail) noexcept {
        try {
            if (!JTRACE_ENABLED(2)) {
                return;
            }
            const std::uintptr_t ownerContextBits =
                reinterpret_cast<std::uintptr_t>(ownerContextOpaque);
            const std::uintptr_t currentContextBits =
                reinterpret_cast<std::uintptr_t>(currentContextOpaque);
            std::ostringstream oss;
            oss << "stage=" << (stage ? stage : "unknown")
                << " outcome=" << (outcome ? outcome : "unknown")
                << " owner_device=" << ownerDeviceId
                << " current_device=" << currentDeviceId
                << " owner_context=" << ownerContextBits
                << " current_context=" << currentContextBits
                << " device_match=" << (deviceMatch ? 1 : 0)
                << " context_match=" << (contextMatch ? 1 : 0)
                << " switch_attempted=" << (switchAttempted ? 1 : 0)
                << " switch_succeeded=" << (switchSucceeded ? 1 : 0)
                << " retire_attempted=" << (managerRetireAttempted ? 1 : 0)
                << " retire_accepted=" << (managerRetireAccepted ? 1 : 0)
                << " deferred_queue_depth=" << static_cast<unsigned long long>(deferredQueueDepth);
            if (!detail.empty()) {
                oss << " detail=" << detail;
            }
            JTRACE("MSTDN", oss.str());
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    static bool reap_deferred_destroy_queue(
        const char* stage,
        const ResourceManager::DeviceContextKey& requiredContext,
        std::uint64_t epoch,
        Failure* outError = nullptr) {
        DeferredDestroyBatch ready;
        std::size_t remainingDepth = 0;
        int currentDeviceId = -1;
        const cudaError_t deviceCode = cudaGetDevice(&currentDeviceId);
        const bool currentDeviceValid = deviceCode == cudaSuccess && currentDeviceId >= 0;
        if (!currentDeviceValid) {
            if (outError) {
                set_failure(*outError, runtime_failure_status(deviceCode), "deferred destruction could not query the current CUDA device");
            }
            return false;
        }
        void* currentContextOpaque = nullptr;
        std::string contextError;
        int contextCode = 0;
        bool currentContextValid = false;
        try {
            currentContextValid = query_current_cuda_context(currentContextOpaque, contextError, &contextCode);
        } catch (...) {
            if (contextCode == 0) {
                throw;
            }
        }
        if (!currentContextValid ||
            (requiredContext.deviceId != currentDeviceId || requiredContext.contextOpaque != currentContextOpaque)) {
            if (outError) {
                set_failure(*outError, driver_failure_status(contextCode), contextError.empty() ? "deferred destruction requires the exact current owner context" : std::string_view(contextError));
            }
            return false;
        }

        {
            const DeferredDestroyLock lock;
            deferred_destroy_queue().extract({currentDeviceId, currentContextOpaque}, epoch, ready.entries);
            remainingDepth = deferred_destroy_queue().size();
        }

        while (auto* resources = ready.entries.head) {
            trace_teardown_event(stage, "deferred_reap", resources->ownerContextKey.deviceId, resources->ownerContextKey.contextOpaque, currentDeviceId, currentContextOpaque, true, true, false, false, false, false, remainingDepth, contextError);
            Failure drainError;
            if (!drain_for_context_retire(*resources, drainError)) {
                if (outError) {
                    *outError = std::move(drainError);
                }
                // Restore the failed and all unprocessed entries exactly once.
                return false;
            }
            delete ready.entries.pop();
        }
        return true;
    }

    static void retry_invalidated_host_allocations() noexcept {
        DeferredDestroyQueue ready;
        {
            const DeferredDestroyLock lock;
            deferred_destroy_queue().extract_invalidated(ready);
        }
        while (ready.head) {
            auto* retained = ready.pop();
            Failure ignored;
            if (drain_invalidated_host_allocations(*retained, ignored)) {
                delete retained;
            } else {
                defer_destroy(retained);
            }
        }
    }

    void invalidate_deferred_resources_after_proven_context_loss(const ResourceManager::DeviceContextKey& key, std::uint64_t epoch) {
        DeferredDestroyQueue ready;
        {
            const DeferredDestroyLock lock;
            deferred_destroy_queue().extract(key, epoch, ready);
        }
        while (ready.head) {
            auto* retained = ready.pop();
            invalidate_resources_after_proven_context_loss(*retained);
            Failure ignored;
            if (drain_invalidated_host_allocations(*retained, ignored)) {
                delete retained;
            } else {
                defer_destroy(retained);
            }
        }
    }

    bool drain_deferred_resources(const ResourceManager::DeviceContextKey& key, std::uint64_t epoch, Failure& outError) {
        return reap_deferred_destroy_queue("context_retire", key, epoch, &outError);
    }

    TrackedDeviceBuffer::TrackedDeviceBuffer(void* pointer, DeviceByteReservation reservation) noexcept
        : _pointer(pointer), _reservation(std::move(reservation)) {
    }

    TrackedDeviceBuffer::TrackedDeviceBuffer(TrackedDeviceBuffer&& other) noexcept
        : _pointer(std::exchange(other._pointer, nullptr)), _reservation(std::move(other._reservation)) {
    }

    TrackedDeviceBuffer& TrackedDeviceBuffer::operator=(TrackedDeviceBuffer&& other) noexcept {
        if (this != &other) {
            assert(!_pointer && !_reservation.active());
            _pointer = std::exchange(other._pointer, nullptr);
            _reservation = std::move(other._reservation);
        }
        return *this;
    }

    void TrackedDeviceBuffer::forget_after_context_loss() noexcept {
        _pointer = nullptr;
        _reservation = {};
    }

    bool TrackedDeviceBuffer::mark_retiring(Failure& failure) noexcept try {
        if (_reservation.state() == DeviceReservationState::Retiring) {
            return true;
        }
        return _reservation.mark_retiring(failure.diagnostic);
    } catch (...) {
        set_failure(failure, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "device retirement bookkeeping failed");
        return false;
    }

    bool TrackedDeviceBuffer::free_after_completion(Failure& failure) noexcept {
        return free_after_completion(failure, ResourceOperations::free_device);
    }

    bool TrackedDeviceBuffer::free_after_completion(Failure& failure, cudaError_t (*freeDevice)(void*)) noexcept try {
        if (!_reservation.active()) {
            return !_pointer;
        }
        if (!mark_retiring(failure)) {
            return false;
        }
        if (_pointer) {
            const auto status = freeDevice(_pointer);
            if (status != cudaSuccess) {
                set_failure(failure, runtime_failure_status(status), "CUDA device buffer free failed");
                return false;
            }
            _pointer = nullptr;
        }
        return _reservation.release_after_physical_free(true, failure.diagnostic);
    } catch (...) {
        set_failure(failure, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA device buffer release bookkeeping failed");
        return false;
    }

    RetiredEvent& RetiredEvent::operator=(RetiredEvent&& other) noexcept {
        if (this != &other) {
            assert(!_event);
            _event = other.release();
        }
        return *this;
    }

    cudaError_t RetiredEvent::destroy() noexcept {
        if (!_event) {
            return cudaSuccess;
        }
        const auto result = ResourceOperations::destroy_event(_event);
        if (result == cudaSuccess) {
            _event = nullptr;
        }
        return result;
    }

    PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            assert(!_pointer);
            _pointer = std::exchange(other._pointer, nullptr);
        }
        return *this;
    }

    cudaError_t PinnedBuffer::free_after_completion() noexcept {
        if (!_pointer) {
            return cudaSuccess;
        }
        const auto result = ResourceOperations::free_host(_pointer);
        if (result == cudaSuccess) {
            _pointer = nullptr;
        }
        return result;
    }

    bool allocate_tracked_device_buffer(
        Resources& resources,
        std::size_t bytes,
        const char* label,
        TrackedDeviceAllocations::node_type& node,
        Failure& outError) {
        TrackedDeviceAllocations staged;
        try {
            staged.try_emplace(nullptr);
            node = staged.extract(staged.begin());
        } catch (...) {
            set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "failed to stage CUDA allocation record");
            return false;
        }

        DeviceByteReservation reservation;
        if (!resources.deviceLedger->reserve(
                {resources.ownerContextKey, resources.contextEpoch, static_cast<std::uint64_t>(bytes)},
                reservation,
                outError)) {
            return false;
        }

        void* allocated = nullptr;
        const auto status = cudaMalloc(&allocated, bytes);
        if (status != cudaSuccess || !allocated) {
            std::array<char, 320> message{};
            std::snprintf(message.data(), message.size(), "cudaMalloc(%s) failed: %s", label, cudaGetErrorString(status));
            set_failure(outError, runtime_failure_status(status), message.data());
            return false;
        }
        if (!reservation.commit(static_cast<std::uint64_t>(bytes), outError.diagnostic)) {
            (void)cudaFree(allocated);
            return false;
        }
        node.key() = allocated;
        node.mapped() = TrackedDeviceBuffer(allocated, std::move(reservation));
        return true;
    }

    bool allocate_owned_device(
        Resources& resources,
        void** outPtr,
        std::size_t bytes,
        const char* label,
        Failure& outError) {
        outError = {};
        if (!outPtr || *outPtr || bytes == 0 || !label || !label[0] || !resources.deviceLedger) {
            outError.diagnostic = "invalid owned CUDA allocation request";
            return false;
        }
        TrackedDeviceAllocations::node_type node;
        if (!allocate_tracked_device_buffer(resources, bytes, label, node, outError)) {
            return false;
        }
        void* const allocated = node.key();
        {
            std::lock_guard<std::mutex> lock(resources.deviceAllocationRecordsMutex);
            auto insertion = resources.deviceAllocationRecords.insert(std::move(node));
            node = std::move(insertion.node);
        }
        if (!node.empty()) {
            Failure ignored;
            (void)node.mapped().free_after_completion(ignored);
            set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "owned CUDA allocation pointer collision");
            return false;
        }
        *outPtr = allocated;
        return true;
    }

    static bool free_owned_device(Resources& resources, void* ptr, Failure& outError) noexcept {
        outError = {};
        if (!ptr) {
            return true;
        }
        try {
            TrackedDeviceAllocations::node_type record;
            {
                std::lock_guard<std::mutex> lock(resources.deviceAllocationRecordsMutex);
                record = resources.deviceAllocationRecords.extract(ptr);
            }
            if (record.empty()) {
                outError.diagnostic = "unregistered owned CUDA free";
                return false;
            }
            if (!record.mapped().free_after_completion(outError, ResourceOperations::free_device)) {
                std::lock_guard<std::mutex> lock(resources.deviceAllocationRecordsMutex);
                // The key remains the lookup identity even if only the ledger
                // record is left after a successful physical free.
                resources.deviceAllocationRecords.insert(std::move(record));
                return false;
            }
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            return false;
        }
    }

    void free_owned_device_noexcept(Resources& resources, void* ptr) noexcept {
        Failure ignored;
        (void)free_owned_device(resources, ptr, ignored);
    }

    static bool recycle_completion_event_locked(Resources& resources, RetiredEvent& event) noexcept {
        if (!event) {
            return true;
        }
        try {
            resources.completionEventPool.push_back(std::move(event));
            return true;
        } catch (...) {
            return event.destroy() == cudaSuccess;
        }
    }

    static bool release_retired_resource(Resources::RetireEntry& entry, Failure& failure) noexcept {
        if (entry.kind == Resources::RetireKind::DeviceFree) {
            return entry.deviceBuffer.free_after_completion(failure);
        }
        if (const auto result = entry.pinnedBuffer.free_after_completion(); result != cudaSuccess) {
            set_failure(failure, runtime_failure_status(result), "retired pinned allocation free failed");
            return false;
        }
        return true;
    }

    static void erase_retire_entry(Resources& resources, std::size_t index) noexcept {
        auto& entry = resources.retireQueue[index];
        resources.retireBytes -= entry.bytes;
        if (entry.scratchTier) {
            resources.retireScratchBytes -= entry.bytes;
        }
        entry = std::move(resources.retireQueue.back());
        resources.retireQueue.pop_back();
    }

    void reap_retire_queue_locked(Resources& resources) noexcept {
        for (std::size_t index = 0; index < resources.retireQueue.size();) {
            auto& entry = resources.retireQueue[index];
            if (entry.stagingActive || entry.requiresContextDrain ||
                (entry.completionRecorded && cudaEventQuery(entry.completion.get()) != cudaSuccess)) {
                ++index;
                continue;
            }
            Failure ignored;
            if (!release_retired_resource(entry, ignored)) {
                ++index;
                continue;
            }
            const bool eventReleased = entry.completionRecorded
                                           ? recycle_completion_event_locked(resources, entry.completion)
                                           : entry.completion.destroy() == cudaSuccess;
            if (!eventReleased) {
                ++index;
                continue;
            }
            erase_retire_entry(resources, index);
        }
    }

    static bool drain_retire_queue_blocking(Resources& resources, Failure& outError) noexcept {
        outError = {{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
        for (std::size_t index = 0; index < resources.retireQueue.size();) {
            auto& entry = resources.retireQueue[index];
            Failure failure;
            if (entry.stagingActive) {
                set_failure(failure, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "frame status staging still leased");
                retain_first_failure(outError, std::move(failure));
                ++index;
                continue;
            }
            if (entry.completionRecorded || entry.requiresContextDrain) {
                const auto result = entry.requiresContextDrain ? cudaDeviceSynchronize() : cudaEventSynchronize(entry.completion.get());
                if (result != cudaSuccess) {
                    set_failure(failure, runtime_failure_status(result), "retired allocation completion wait failed");
                    retain_first_failure(outError, std::move(failure));
                    ++index;
                    continue;
                }
                entry.completionRecorded = false;
                entry.requiresContextDrain = false;
            }
            if (!release_retired_resource(entry, failure)) {
                retain_first_failure(outError, std::move(failure));
                ++index;
                continue;
            }
            if (const auto status = entry.completion.destroy(); status != cudaSuccess) {
                set_failure(failure, runtime_failure_status(status), "retired completion event destruction failed");
                retain_first_failure(outError, std::move(failure));
                ++index;
                continue;
            }
            erase_retire_entry(resources, index);
        }
        for (auto& event : resources.completionEventPool) {
            if (const auto status = event.destroy(); status != cudaSuccess) {
                Failure failure;
                set_failure(failure, runtime_failure_status(status), "pooled completion event destruction failed");
                retain_first_failure(outError, std::move(failure));
            }
        }
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            return false;
        }
        resources.completionEventPool.clear();
        return true;
    }

    bool allocate_frame_status_staging(Resources& resources, std::size_t bytes, void*& host, cudaEvent_t& event, Failure& outError) {
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        if (bytes > std::numeric_limits<std::size_t>::max() - resources.retireBytes) {
            set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "frame status staging byte accounting overflow");
            return false;
        }
        try {
            resources.retireQueue.emplace_back();
        } catch (const std::bad_alloc&) {
            set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "frame status staging retention failed");
            return false;
        }
        auto& entry = resources.retireQueue.back();
        entry.kind = Resources::RetireKind::HostPinnedFree;
        cudaEvent_t completion = nullptr;
        const auto createStatus = cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
        entry.completion = RetiredEvent(completion);
        if (createStatus != cudaSuccess || !completion) {
            set_failure(outError, runtime_failure_status(createStatus), "highlight boost status event allocation failed");
            return false;
        }
        void* pinned = nullptr;
        const auto allocationStatus = cudaMallocHost(&pinned, bytes);
        entry.pinnedBuffer = PinnedBuffer(pinned);
        if (pinned) {
            entry.bytes = bytes;
            resources.retireBytes += bytes;
        }
        if (allocationStatus != cudaSuccess || !pinned) {
            set_failure(outError, runtime_failure_status(allocationStatus), "highlight boost status staging allocation failed");
            return false;
        }
        entry.stagingActive = true;
        host = pinned;
        event = completion;
        return true;
    }

    bool retire_frame_status_staging(Resources& resources, void* host, bool completionRecorded, bool requiresContextDrain, Failure& outError) {
        std::lock_guard<std::mutex> lock(resources.m);
        const auto entry = std::find_if(resources.retireQueue.begin(), resources.retireQueue.end(), [host](const auto& e) {
            return e.stagingActive && e.pinnedBuffer.get() == host;
        });
        if (entry == resources.retireQueue.end()) {
            set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "unregistered frame status staging");
            return false;
        }
        entry->stagingActive = false;
        entry->completionRecorded = completionRecorded;
        entry->requiresContextDrain = requiresContextDrain;
        if (completionRecorded || requiresContextDrain) {
            return true;
        }
        if (!release_retired_resource(*entry, outError)) {
            return false;
        }
        if (const auto status = entry->completion.destroy(); status != cudaSuccess) {
            set_failure(outError, runtime_failure_status(status), "highlight boost event destruction failed");
            return false;
        }
        erase_retire_entry(resources, static_cast<std::size_t>(entry - resources.retireQueue.begin()));
        return true;
    }

    static bool acquire_retire_event_locked(Resources& resources, RetiredEvent& outEvent, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        while (!resources.completionEventPool.empty()) {
            outEvent = std::move(resources.completionEventPool.back());
            resources.completionEventPool.pop_back();
            if (outEvent) {
                return true;
            }
        }
        cudaEvent_t ev = nullptr;
        const auto status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        outEvent = RetiredEvent(ev);
        if (status != cudaSuccess || !ev) {
            set_failure(outError, runtime_failure_status(status), "cudaEventCreateWithFlags failed");
            return false;
        }
        return true;
    }

    static void reap_frame_use_events_locked(Resources& resources) noexcept {
        std::vector<Resources::PendingFrameUseEvent>& pending = resources.pendingFrameUseEvents;
        for (std::size_t i = 0; i < pending.size();) {
            Resources::PendingFrameUseEvent& entry = pending[i];
            cudaEvent_t ev = entry.event.get();
            const bool released = entry.recorded
                                      ? cudaEventQuery(ev) == cudaSuccess &&
                                            recycle_completion_event_locked(resources, entry.event)
                                      : entry.event.destroy() == cudaSuccess;
            if (released) {
                pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
                continue;
            }
            ++i;
        }
    }

    bool wait_for_frame_use_events_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        reap_frame_use_events_locked(resources);
        if (resources.pendingFrameUseEvents.empty()) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        for (const Resources::PendingFrameUseEvent& entry : resources.pendingFrameUseEvents) {
            if (!entry.recorded) {
                continue;
            }
            const cudaEvent_t ev = entry.event.get();
            const cudaError_t waitErr = cudaStreamWaitEvent(stream, ev, 0);
            if (waitErr != cudaSuccess) {
                std::array<char, 320> diagnostic{};
                std::snprintf(diagnostic.data(), diagnostic.size(), "cudaStreamWaitEvent before %s update failed", label ? label : "resource");
                set_failure(outError, runtime_failure_status(waitErr), diagnostic.data());
                return false;
            }
        }
        return true;
    }

    struct RetireFenceRecord {
        cudaEvent_t event = nullptr;
        void* cudaStreamOpaque = nullptr;
        const char* label = nullptr;
    };

    static bool record_retire_fence_locked(
        Resources& resources,
        const RetireFenceRecord& record,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        cudaEvent_t retireEv = record.event;
        if (!retireEv) {
            outError.diagnostic = "retire fence event missing";
            return false;
        }
        const cudaStream_t stream = record.cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(record.cudaStreamOpaque)
                                        : nullptr;
        if (!wait_for_frame_use_events_locked(resources, record.cudaStreamOpaque, record.label, outError)) {
            return false;
        }
        const cudaError_t recErr = cudaEventRecord(retireEv, stream);
        if (recErr != cudaSuccess) {
            std::array<char, 320> diagnostic{};
            std::snprintf(diagnostic.data(), diagnostic.size(), "cudaEventRecord for %s retire failed", record.label ? record.label : "resource");
            set_failure(outError, runtime_failure_status(recErr), diagnostic.data());
            return false;
        }
        return true;
    }

    bool retire_ptr_locked(Resources& resources, void* ptr, std::size_t bytes, Resources::RetireKind kind, void* cudaStreamOpaque, const char* label, Failure& outError, bool scratchTier, TrackedDeviceBuffer* externalAllocation) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!ptr) {
            return true;
        }
        if (bytes > std::numeric_limits<std::size_t>::max() -
                        resources.retireBytes ||
            (scratchTier &&
             bytes > std::numeric_limits<std::size_t>::max() -
                         resources.retireScratchBytes)) {
            outError.diagnostic = "CUDA retirement byte accounting overflow";
            return false;
        }

        reap_retire_queue_locked(resources);

        try {
            resources.retireQueue.emplace_back();
        } catch (...) {
            JuicerLogging::discard_current_exception();
            set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "failed to reserve CUDA retirement record");
            return false;
        }
        auto& entry = resources.retireQueue.back();
        entry.kind = Resources::RetireKind::EventOnly;
        const auto discardFence = [&] {
            const auto result = entry.completion.destroy();
            if (result == cudaSuccess) {
                resources.retireQueue.pop_back();
            } else {
                record_cleanup_context_loss(outError, runtime_failure_status(result));
            }
            return false;
        };
        if (!acquire_retire_event_locked(resources, entry.completion, outError)) {
            return discardFence();
        }
        const RetireFenceRecord fenceRecord{entry.completion.get(), cudaStreamOpaque, label};
        if (!record_retire_fence_locked(resources, fenceRecord, outError)) {
            return discardFence();
        }

        if (kind == Resources::RetireKind::DeviceFree) {
            if (externalAllocation) {
                if (externalAllocation->get() != ptr || externalAllocation->bytes() != bytes ||
                    !externalAllocation->mark_retiring(outError)) {
                    if (outError.diagnostic.empty()) {
                        set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "invalid external CUDA retirement allocation");
                    }
                    return discardFence();
                }
                entry.deviceBuffer = std::move(*externalAllocation);
            } else {
                std::lock_guard<std::mutex> lock(resources.deviceAllocationRecordsMutex);
                const auto allocation = resources.deviceAllocationRecords.find(ptr);
                if (allocation == resources.deviceAllocationRecords.end() ||
                    allocation->second.bytes() != bytes) {
                    set_failure(outError, outError.status, "unregistered or size-mismatched CUDA retirement");
                    return discardFence();
                }
                if (!allocation->second.mark_retiring(outError)) {
                    return discardFence();
                }
                entry.deviceBuffer = std::move(allocation->second);
                resources.deviceAllocationRecords.erase(allocation);
            }
        }
        entry.bytes = bytes;
        entry.kind = kind;
        entry.scratchTier = scratchTier;
        entry.completionRecorded = true;
        resources.retireBytes += bytes;
        if (scratchTier) {
            resources.retireScratchBytes += bytes;
        }
        return true;
    }


    static bool drain_all_resource_allocations(Resources& resources, Failure& outError) noexcept {
        outError = {{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
        for (Resources::PendingFrameUseEvent& entry : resources.pendingFrameUseEvents) {
            if (entry.recorded) {
                const auto result = cudaEventSynchronize(entry.event.get());
                if (result != cudaSuccess) {
                    Failure failure;
                    set_failure(failure, runtime_failure_status(result), "frame-use completion wait failed");
                    retain_first_failure(outError, std::move(failure));
                } else {
                    entry.recorded = false;
                }
            }
        }
        for (auto& event : resources.defectRetirementEvents) {
            if (const auto status = event.destroy(); status != cudaSuccess) {
                Failure failure;
                set_failure(failure, runtime_failure_status(status), "defect attachment retirement event destruction failed");
                retain_first_failure(outError, std::move(failure));
            }
        }
        Failure retireError;
        if (!drain_retire_queue_blocking(resources, retireError)) {
            retain_first_failure(outError, std::move(retireError));
        }
        Failure diffusionError;
        if (!Diffusion::drain_diffusion_resources(resources.diffusion, diffusionError)) {
            retain_first_failure(outError, std::move(diffusionError));
        }
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            return false;
        }

        free_curve(resources, resources.densB);
        free_curve(resources, resources.densG);
        free_curve(resources, resources.densR);
        free_density_layers(resources);
        free_curve(resources, resources.dirDensB);
        free_curve(resources, resources.dirDensG);
        free_curve(resources, resources.dirDensR);
        free_curve(resources, resources.sensB);
        free_curve(resources, resources.sensG);
        free_curve(resources, resources.sensR);
        free_tables(resources);
        clear_gaussian_kernel_views(resources);
        free_gaussian_cache(resources);
        for (auto& boundary : resources.spatialDirBoundaries) {
            free_owned_device_noexcept(resources, boundary.weights);
            boundary = Resources::DeviceSpatialDirBoundary{};
        }
        free_optics_scratch(resources, resources.scannerScratch, nullptr);
        free_spatial_dir_scratch(
            resources,
            resources.spatialDirScratch,
            nullptr);
        free_shared_tmp_plane(resources);
        free_stbn(resources);
        free_wang(resources);
        free_print_payloads(resources);
        free_film_tc_lut(resources);
        free_mallett_basis(resources);
        free_output_gamut_table(resources);
        Failure scanError;
        if (!free_scan_error_buffers(resources, scanError)) {
            outError = std::move(scanError);
            return false;
        }
        for (auto& [key, slot] : resources.grainV2Banks) {
            (void)key;
            Failure bankError;
            if (slot->bank && !free_owned_device(resources, slot->bank, bankError)) {
                outError = std::move(bankError);
                return false;
            }
            slot->bank = nullptr;
        }
        resources.grainV2Banks.clear();
        for (Resources::PendingFrameUseEvent& entry :
             resources.pendingFrameUseEvents) {
            if (const auto status = entry.event.destroy(); status != cudaSuccess) {
                Failure failure;
                set_failure(failure, runtime_failure_status(status), "frame use event destruction failed");
                retain_first_failure(outError, std::move(failure));
                continue;
            }
        }
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            return false;
        }
        resources.pendingFrameUseEvents.clear();

        for (;;) {
            void* orphanedPtr = nullptr;
            {
                std::lock_guard<std::mutex> allocationLock(
                    resources.deviceAllocationRecordsMutex);
                if (resources.deviceAllocationRecords.empty()) {
                    break;
                }
                orphanedPtr = resources.deviceAllocationRecords.begin()->first;
            }
            Failure freeError;
            if (!free_owned_device(resources, orphanedPtr, freeError)) {
                outError = std::move(freeError);
                break;
            }
        }

        std::lock_guard<std::mutex> allocationLock(
            resources.deviceAllocationRecordsMutex);
        if (resources.deviceAllocationRecords.empty() &&
            resources.contextLossOnlyDeviceAllocationRecords.empty() &&
            resources.retireQueue.empty()) {
            return true;
        }
        if (outError.status.category == FJ_STATUS_SUCCESS) {
            set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "CUDA context retains live allocations");
        }
        return false;
    }

    Resources::~Resources() noexcept {
        try {
            if (contextInvalidatedByProvenLoss) {
                return;
            }
            Failure ignored;
            (void)drain_all_resource_allocations(*this, ignored);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    bool drain_for_context_retire(Resources& resources, Failure& outError) noexcept try {
        outError = {};
        std::lock_guard<std::mutex> servingLock(resources.servingUpdateMutex);
        if (resources.contextInvalidatedByProvenLoss) {
            return drain_invalidated_host_allocations(resources, outError);
        }
        // The explicit lifecycle owner has excluded new use. Destruction never
        // reenters Root policy; uncertainty stays in native retirement ownership.
        if (!validate_resource_context_locked(resources, outError)) {
            return false;
        }
        bool frameUseFenceQuarantined = false;
        {
            std::lock_guard<std::mutex> resourceLock(resources.m);
            frameUseFenceQuarantined = resources.frameUseFenceQuarantined;
        }
        if (frameUseFenceQuarantined) {
            const auto syncError = ResourceOperations::synchronize_context();
            if (syncError != cudaSuccess) {
                set_failure(outError, runtime_failure_status(syncError), "quarantined frame-use context synchronization failed");
                return false;
            }
            std::lock_guard<std::mutex> resourceLock(resources.m);
            resources.frameUseFenceQuarantined = false;
        }
        if (resources.scannerScratch.filmDustTransmittance || resources.scannerScratch.gateTransmittance) {
            for (const auto& entry : resources.pendingFrameUseEvents) {
                if (entry.recorded) {
                    const auto result = cudaEventSynchronize(entry.event.get());
                    if (result != cudaSuccess) {
                        set_failure(outError, runtime_failure_status(result), "context defect attachment completion wait failed");
                        return false;
                    }
                }
            }
            for (float** plane : {&resources.scannerScratch.filmDustTransmittance, &resources.scannerScratch.gateTransmittance}) {
                if (!free_owned_device(resources, *plane, outError)) {
                    return false;
                }
                std::lock_guard<std::mutex> lock(resources.m);
                *plane = nullptr;
            }
        }
        std::lock_guard<std::mutex> resourceLock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        if (!drain_all_resource_allocations(resources, outError)) {
            if (outError.diagnostic.empty()) {
                set_failure(outError, outError.status, "CUDA context resource drain retained failed physical frees");
            }
            return false;
        }
        return true;

    } catch (const std::bad_alloc&) {
        set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "CUDA resource drain allocation failed");
        return false;
    } catch (const std::exception& detail) {
        set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, detail.what());
        return false;
    } catch (...) {
        set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA resource drain failed with unknown exception");
        return false;
    }

    void invalidate_resources_after_proven_context_loss(
        Resources& resources) noexcept {
        try {
            std::lock_guard<std::mutex> servingLock(resources.servingUpdateMutex);
            std::lock_guard<std::mutex> resourceLock(resources.m);
            std::lock_guard<std::mutex> allocationLock(
                resources.deviceAllocationRecordsMutex);
            resources.contextInvalidatedByProvenLoss = true;
            Diffusion::invalidate_diffusion_resources_after_proven_context_loss(
                resources.diffusion);
            clear_gaussian_kernel_views(resources);
            for (auto& boundary : resources.spatialDirBoundaries) {
                boundary = Resources::DeviceSpatialDirBoundary{};
            }
            resources.gaussianKernelCache.clear();
            resources.grainV2Banks.clear();
            resources.deviceAllocationRecords.clear();
            resources.contextLossOnlyDeviceAllocationRecords.clear();
            for (std::size_t index = 0; index < resources.retireQueue.size();) {
                auto& entry = resources.retireQueue[index];
                entry.deviceBuffer.forget_after_context_loss();
                (void)entry.completion.release();
                if (entry.kind != Resources::RetireKind::HostPinnedFree) {
                    erase_retire_entry(resources, index);
                    continue;
                }
                entry.completionRecorded = false;
                entry.requiresContextDrain = false;
                entry.stagingActive = false;
                ++index;
            }
            resources.completionEventPool.clear();
            resources.pendingFrameUseEvents.clear();
            for (auto& event : resources.defectRetirementEvents) {
                (void)event.release();
            }
            resources.pendingScanErrorReadbacks.clear();
            for (auto& slot : resources.scanErrorBuffers) {
                (void)slot.event.release();
                slot.device = nullptr;
                slot.leased = false;
            }
            resources.frameUseFenceQuarantined = false;
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    static bool drain_invalidated_host_allocations(Resources& resources, Failure& outError) noexcept try {
        std::lock_guard<std::mutex> lock(resources.m);
        outError = {{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
        for (std::size_t index = 0; index < resources.retireQueue.size();) {
            auto& entry = resources.retireQueue[index];
            assert(entry.kind == Resources::RetireKind::HostPinnedFree);
            if (const auto status = entry.pinnedBuffer.free_after_completion(); status != cudaSuccess) {
                Failure failure;
                set_failure(failure, runtime_failure_status(status), "invalidated frame status staging free failed");
                retain_first_failure(outError, std::move(failure));
                ++index;
                continue;
            }
            erase_retire_entry(resources, index);
        }
        for (auto& slot : resources.scanErrorBuffers) {
            if (const auto status = slot.host.free_after_completion(); status != cudaSuccess) {
                Failure failure;
                set_failure(failure, runtime_failure_status(status), "invalidated scan error pinned buffer free failed");
                retain_first_failure(outError, std::move(failure));
            }
        }
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            return false;
        }
        resources.scanErrorBuffers.clear();
        return true;
    } catch (...) {
        set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "invalidated pinned host disposal failed");
        return false;
    }

    Resources::Resources(
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t epoch,
        std::shared_ptr<DeviceAllocationLedger> ledger)
        : ownerContextKey(contextKey),
          contextEpoch(epoch),
          deviceLedger(std::move(ledger)),
          deviceId(contextKey.deviceId) {
    }

    Resources* create(
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        std::shared_ptr<DeviceAllocationLedger> deviceLedger,
        Failure& outError) noexcept {
        try {
            outError = {};
            if (contextKey.deviceId < 0 || contextKey.contextOpaque == nullptr ||
                contextEpoch == 0 || deviceLedger == nullptr) {
                outError.diagnostic =
                    "ResourceDescriptorMismatch component=cuda_resources field=allocation_identity";
                return nullptr;
            }
            retry_invalidated_host_allocations();
            reap_deferred_destroy_queue("create", contextKey, contextEpoch);
            return new Resources(
                contextKey,
                contextEpoch,
                std::move(deviceLedger));
        } catch (...) {
            try {
                outError.diagnostic = "failed to allocate exact-context CUDA resources";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return nullptr;
        }
    }

    void destroy(Resources* resources) noexcept {
        std::unique_ptr<Resources, decltype(&defer_destroy)> retained(resources, &defer_destroy);
        try {
            if (!resources) {
                return;
            }

            if (resources->contextInvalidatedByProvenLoss) {
                Failure ignored;
                if (drain_invalidated_host_allocations(*resources, ignored)) {
                    delete retained.release();
                }
                return;
            }

            const auto ownerKey = resources->ownerContextKey;
            const auto ownerEpoch = resources->contextEpoch;
            reap_deferred_destroy_queue("destroy_pre", ownerKey, ownerEpoch);

            const int ownerDeviceId = resources->deviceId;
            void* ownerContextOpaque = resources->ownerContextKey.contextOpaque;
            int currentDeviceId = -1;
            std::string currentDeviceError;
            const bool currentDeviceValid = query_current_cuda_device(currentDeviceId, currentDeviceError);
            void* currentContextOpaque = nullptr;
            std::string currentContextError;
            const bool currentContextValid = query_current_cuda_context(currentContextOpaque, currentContextError);

            bool deviceMatch = false;
            bool contextMatch = false;
            bool ownerMatch = owner_matches_current(
                ownerDeviceId,
                ownerContextOpaque,
                currentDeviceId,
                currentDeviceValid,
                currentContextOpaque,
                currentContextValid,
                deviceMatch,
                contextMatch);

            bool switchAttempted = false;
            bool switchSucceeded = false;
            int restoreDeviceId = currentDeviceId;
            bool restoreDevice = false;

            if (!ownerMatch && ownerDeviceId >= 0 &&
                (!currentDeviceValid || currentDeviceId != ownerDeviceId)) {
                switchAttempted = true;
                const cudaError_t setErr = cudaSetDevice(ownerDeviceId);
                if (setErr == cudaSuccess) {
                    switchSucceeded = true;
                    restoreDevice = currentDeviceValid && currentDeviceId != ownerDeviceId;

                    int switchedDeviceId = -1;
                    std::string switchedDeviceError;
                    const bool switchedDeviceValid = query_current_cuda_device(switchedDeviceId, switchedDeviceError);

                    void* switchedContextOpaque = nullptr;
                    std::string switchedContextError;
                    const bool switchedContextValid =
                        query_current_cuda_context(switchedContextOpaque, switchedContextError);

                    ownerMatch = owner_matches_current(
                        ownerDeviceId,
                        ownerContextOpaque,
                        switchedDeviceId,
                        switchedDeviceValid,
                        switchedContextOpaque,
                        switchedContextValid,
                        deviceMatch,
                        contextMatch);

                    if (switchedDeviceValid) {
                        currentDeviceId = switchedDeviceId;
                    }
                    if (switchedContextValid) {
                        currentContextOpaque = switchedContextOpaque;
                    }
                    if (!switchedDeviceError.empty()) {
                        currentDeviceError = switchedDeviceError;
                    }
                    if (!switchedContextError.empty()) {
                        currentContextError = switchedContextError;
                    }
                } else {
                    currentContextError = std::string("cudaSetDevice failed: ") + (cudaGetErrorString(setErr) ? cudaGetErrorString(setErr) : "(unknown)");
                }
            }

            if (ownerMatch) {
                std::size_t deferredQueueDepth = 0;
                {
                    const DeferredDestroyLock lock;
                    deferredQueueDepth = deferred_destroy_queue().size();
                }
                std::string detail;
                if (!currentDeviceError.empty()) {
                    detail = currentDeviceError;
                }
                if (!currentContextError.empty()) {
                    if (!detail.empty()) {
                        detail += "; ";
                    }
                    detail += currentContextError;
                }
                trace_teardown_event(
                    "destroy",
                    "direct_free",
                    ownerDeviceId,
                    ownerContextOpaque,
                    currentDeviceId,
                    currentContextOpaque,
                    deviceMatch,
                    contextMatch,
                    switchAttempted,
                    switchSucceeded,
                    false,
                    false,
                    deferredQueueDepth,
                    detail);
                Failure drainError;
                if (!drain_for_context_retire(*resources, drainError)) {
                    if (restoreDevice && restoreDeviceId >= 0 &&
                        restoreDeviceId != ownerDeviceId) {
                        (void)cudaSetDevice(restoreDeviceId);
                    }
                    return;
                }
                delete retained.release();

                if (restoreDevice && restoreDeviceId >= 0 && restoreDeviceId != ownerDeviceId) {
                    (void)cudaSetDevice(restoreDeviceId);
                }
                reap_deferred_destroy_queue("destroy_post", ownerKey, ownerEpoch);
                return;
            }

            if (restoreDevice && restoreDeviceId >= 0 && restoreDeviceId != ownerDeviceId) {
                (void)cudaSetDevice(restoreDeviceId);
            }

            std::size_t deferredQueueDepth = 0;
            {
                const DeferredDestroyLock lock;
                deferred_destroy_queue().push(retained.release());
                deferredQueueDepth = deferred_destroy_queue().size();
            }

            std::string detail;
            if (!currentDeviceError.empty()) {
                detail = currentDeviceError;
            }
            if (!currentContextError.empty()) {
                if (!detail.empty()) {
                    detail += "; ";
                }
                detail += currentContextError;
            }
            trace_teardown_event(
                "destroy",
                "deferred_enqueue",
                ownerDeviceId,
                ownerContextOpaque,
                currentDeviceId,
                currentContextOpaque,
                deviceMatch,
                contextMatch,
                switchAttempted,
                switchSucceeded,
                false,
                false,
                deferredQueueDepth,
                detail);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    bool reap_retired_allocations(Resources& resources, std::size_t& reclaimedBytes, Failure& outError) {
        reclaimedBytes = 0;
        outError = {};
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        const std::size_t beforeBytes = resources.retireBytes;
        reap_retire_queue_locked(resources);
        const std::size_t afterBytes = resources.retireBytes;
        reclaimedBytes = (beforeBytes >= afterBytes) ? (beforeBytes - afterBytes) : 0;
        return true;
    }

    bool record_frame_use_event(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError) noexcept try {
        outError = {};
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        reap_retire_queue_locked(resources);
        reap_frame_use_events_locked(resources);
        // Reserve the owner before acquiring an event. An unrecorded event
        // stays here for destruction only if cleanup fails or unwinds.
        auto& entry = resources.pendingFrameUseEvents.emplace_back();
        if (!acquire_retire_event_locked(resources, entry.event, outError)) {
            const auto destroyResult = entry.event.destroy();
            record_cleanup_context_loss(outError, runtime_failure_status(destroyResult));
            if (destroyResult == cudaSuccess) {
                resources.pendingFrameUseEvents.pop_back();
            }
            return false;
        }
        auto* const stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        const auto recordError = ResourceOperations::record_event(entry.event.get(), stream);
        if (recordError == cudaSuccess) {
            entry.recorded = true;
            return true;
        }
        const auto destroyResult = entry.event.destroy();
        if (destroyResult == cudaSuccess) {
            resources.pendingFrameUseEvents.pop_back();
        }
        std::array<char, 320> diagnostic{};
        std::snprintf(diagnostic.data(), diagnostic.size(), "cudaEventRecord(%s) failed", label ? label : "frame use");
        set_failure(outError, runtime_failure_status(recordError), diagnostic.data());
        record_cleanup_context_loss(outError, runtime_failure_status(destroyResult));
        return false;
    } catch (const std::bad_alloc&) {
        set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "frame use event retention failed");
        return false;
    } catch (...) {
        set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "frame use event submission failed");
        return false;
    }

    void quarantine_unfenced_frame_use(Resources& resources) noexcept {
        try {
            std::lock_guard<std::mutex> lock(resources.m);
            resources.frameUseFenceQuarantined = true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    static bool free_scan_error_buffers(Resources& resources, Failure& outError) noexcept {
        for (const auto& entry : resources.pendingScanErrorReadbacks) {
            if (const auto status = cudaEventSynchronize(entry.event); status != cudaSuccess) {
                set_failure(outError, runtime_failure_status(status), "scan error buffer completion wait failed");
                return false;
            }
        }
        resources.pendingScanErrorReadbacks.clear();
        for (auto& slot : resources.scanErrorBuffers) {
            if (const auto status = slot.event.destroy(); status != cudaSuccess) {
                set_failure(outError, runtime_failure_status(status), "scan error completion event destruction failed");
                return false;
            }
            if (const auto status = slot.host.free_after_completion(); status != cudaSuccess) {
                set_failure(outError, runtime_failure_status(status), "scan error pinned buffer free failed");
                return false;
            }
            if (!free_owned_device(resources, slot.device, outError)) {
                return false;
            }
            slot.device = nullptr;
        }
        resources.scanErrorBuffers.clear();
        return true;
    }

    bool acquire_scan_error_buffer(Resources& resources, ScanErrorBuffer& outBuffer, Failure& outError) {
        outError = {};
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }

        auto& buffers = resources.scanErrorBuffers;
        // Prepare two independent leases at first use, before either can be pending.
        auto available = std::ranges::find(buffers, false, &Resources::ScanErrorBufferSlot::leased);
        if (buffers.size() < 2 || available == buffers.end()) {
            const auto capacity = std::max(std::size_t{2}, buffers.size() + 1);
            try {
                buffers.reserve(capacity);
                resources.pendingScanErrorReadbacks.reserve(capacity);
            } catch (...) {
                set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "reserve scan error buffers failed");
                return false;
            }
            while (buffers.size() < capacity) {
                void* pointer = nullptr;
                if (!allocate_owned_device(resources, &pointer, sizeof(int), "frame scan error flag", outError)) {
                    return false;
                }
                int* pinned = nullptr;
                const auto status = cudaMallocHost(reinterpret_cast<void**>(&pinned), sizeof(int));
                if (status != cudaSuccess) {
                    free_owned_device_noexcept(resources, pointer);
                    set_failure(outError, runtime_failure_status(status), "allocate pinned scan error buffer failed");
                    return false;
                }
                buffers.push_back({PinnedBuffer(pinned), {}, static_cast<int*>(pointer), false});
            }
            available = std::ranges::find(buffers, false, &Resources::ScanErrorBufferSlot::leased);
        }
        if (const auto status = available->event.destroy(); status != cudaSuccess) {
            set_failure(outError, runtime_failure_status(status), "retry scan error completion event destruction failed");
            return false;
        }
        cudaEvent_t event = nullptr;
        const auto status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        available->event = RetiredEvent(event);
        if (status != cudaSuccess || !event) {
            set_failure(outError, runtime_failure_status(status), "allocate scan error event failed");
            return false;
        }
        available->leased = true;
        outBuffer = {available->device, static_cast<int*>(available->host.get()), event};
        return true;
    }

    static bool release_scan_error_buffer_locked(Resources& resources, int* host, bool pendingReadback, Failure& outError) {
        auto& buffers = resources.scanErrorBuffers;
        const auto buffer = std::ranges::find(buffers, host, [](const auto& slot) {
            return static_cast<int*>(slot.host.get());
        });
        if (!host || buffer == buffers.end() || !buffer->leased) {
            set_failure(outError, outError.status, "scan error buffer lease missing");
            return false;
        }
        const auto status = buffer->event.destroy();
        buffer->leased = pendingReadback && status != cudaSuccess;
        if (status != cudaSuccess) {
            set_failure(outError, runtime_failure_status(status), "scan error completion event destruction failed");
            return false;
        }
        return true;
    }

    bool release_scan_error_buffer(Resources& resources, int* host, Failure& outError) {
        outError = {};
        std::lock_guard<std::mutex> lock(resources.m);
        return release_scan_error_buffer_locked(resources, host, false, outError);
    }


    bool retain_scan_error_readback(
        Resources& resources,
        const ScanErrorBuffer& buffer,
        ScanErrorReadbackIdentity& identity,
        bool hasReadback,
        Failure& outError) {
        outError = {};
        if (!buffer.host || !buffer.event) {
            set_failure(outError, outError.status, "scan error readback staging missing");
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }

        try {
            static_assert(std::is_nothrow_move_assignable_v<ScanErrorReadbackIdentity>);
            auto& entry = resources.pendingScanErrorReadbacks.emplace_back();
            entry.host = buffer.host;
            entry.event = buffer.event;
            entry.identity = std::move(identity);
            entry.hasReadback = hasReadback;
        } catch (...) {
            set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "scan error readback retention failed");
            return false;
        }

        return true;
    }

    bool poll_scan_error_readbacks(
        Resources& resources,
        ScanErrorReadbackResult& outResult,
        Failure& outError) {
        outResult = ScanErrorReadbackResult{};
        outError = {};

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }

        std::vector<Resources::PendingScanErrorReadback>& pending =
            resources.pendingScanErrorReadbacks;
        for (std::size_t i = 0; i < pending.size();) {
            Resources::PendingScanErrorReadback& entry = pending[i];
            cudaEvent_t ev = entry.event;
            if (!entry.host || !ev) {
                outError.diagnostic = "scan error buffer completion is missing its lease or event";
                return false;
            }

            const cudaError_t pollErr = cudaEventQuery(ev);
            if (pollErr == cudaSuccess) {
                if (!release_scan_error_buffer_locked(resources, entry.host, true, outError)) {
                    return false;
                }
                if (entry.hasReadback && *entry.host != 0) {
                    outResult.status = *entry.host;
                    outResult.identity = std::move(entry.identity);
                }
                pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
                if (outResult.status != 0) {
                    return true;
                }
                continue;
            }
            if (pollErr == cudaErrorNotReady) {
                ++i;
                continue;
            }

            outError.status = runtime_failure_status(pollErr);
            outError.diagnostic = std::string("CUDA scan error event query failed: ") +
                                  (cudaGetErrorString(pollErr) ? cudaGetErrorString(pollErr) : "(unknown)");
            return false;
        }
        return true;
    }


    bool retire_frame_scratch_allocation(
        Resources& resources,
        void* ptr,
        std::size_t bytes,
        TrackedDeviceBuffer& allocation,
        void* cudaStreamOpaque,
        const char* label,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!ptr) {
            return true;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        return retire_ptr_locked(
            resources,
            ptr,
            bytes,
            Resources::RetireKind::DeviceFree,
            cudaStreamOpaque,
            label ? label : "frame scratch",
            outError,
            true,
            &allocation);
    }

    bool adopt_failed_frame_allocation_record(
        Resources& resources,
        TrackedDeviceAllocations::node_type& allocationRecord,
        bool completionCertain,
        Failure& outError) noexcept {
        try {
            outError = {};
            if (allocationRecord.empty() || !allocationRecord.key() ||
                !allocationRecord.mapped().active()) {
                outError.diagnostic = "invalid failed frame allocation record";
                return false;
            }
            std::lock_guard<std::mutex> lock(
                resources.deviceAllocationRecordsMutex);
            auto& destination = completionCertain
                                    ? resources.deviceAllocationRecords
                                    : resources.contextLossOnlyDeviceAllocationRecords;
            auto insertion = destination.insert(std::move(allocationRecord));
            if (!insertion.inserted) {
                allocationRecord = std::move(insertion.node);
                outError.diagnostic = "failed frame allocation pointer collision";
                return false;
            }
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError.diagnostic = "failed frame allocation adoption failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }


} // namespace JuicerCuda
