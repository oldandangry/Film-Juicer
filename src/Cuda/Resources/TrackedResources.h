#pragma once

#include <map>
#include <utility>

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaDeviceLedger.h"

namespace JuicerCuda {

    // These handles move within one context epoch. The owning frame or pool
    // must prove completion before release, or retain them for context
    // retirement. Destruction alone cannot establish either fact.
    class TrackedDeviceBuffer final {
    public:
        TrackedDeviceBuffer() noexcept = default;
        TrackedDeviceBuffer(void* pointer, DeviceByteReservation reservation) noexcept;
        TrackedDeviceBuffer(const TrackedDeviceBuffer&) = delete;
        TrackedDeviceBuffer& operator=(const TrackedDeviceBuffer&) = delete;
        TrackedDeviceBuffer(TrackedDeviceBuffer&& other) noexcept;
        TrackedDeviceBuffer& operator=(TrackedDeviceBuffer&& other) noexcept;

        [[nodiscard]] void* get() const noexcept {
            return _pointer;
        }
        [[nodiscard]] std::uint64_t bytes() const noexcept {
            return _reservation.bytes();
        }
        [[nodiscard]] DeviceReservationState state() const noexcept {
            return _reservation.state();
        }
        [[nodiscard]] bool active() const noexcept {
            return _reservation.active();
        }
        void forget_after_context_loss() noexcept;
        bool mark_retiring(Failure& failure) noexcept;
        bool free_after_completion(Failure& failure) noexcept;
        bool free_after_completion(Failure& failure, cudaError_t (*freeDevice)(void*)) noexcept;

    private:
        void* _pointer = nullptr;
        DeviceByteReservation _reservation;
    };

    using TrackedDeviceAllocations = std::map<void*, TrackedDeviceBuffer>;

    class RetiredEvent final {
    public:
        RetiredEvent() noexcept = default;
        explicit RetiredEvent(cudaEvent_t event) noexcept : _event(event) {}
        RetiredEvent(const RetiredEvent&) = delete;
        RetiredEvent& operator=(const RetiredEvent&) = delete;
        RetiredEvent(RetiredEvent&& other) noexcept : _event(other.release()) {}
        RetiredEvent& operator=(RetiredEvent&& other) noexcept;

        [[nodiscard]] cudaEvent_t get() const noexcept {
            return _event;
        }
        explicit operator bool() const noexcept {
            return _event != nullptr;
        }
        cudaEvent_t release() noexcept {
            return std::exchange(_event, nullptr);
        }
        cudaError_t destroy() noexcept;

    private:
        cudaEvent_t _event = nullptr;
    };

    class PinnedBuffer final {
    public:
        PinnedBuffer() noexcept = default;
        explicit PinnedBuffer(void* pointer) noexcept : _pointer(pointer) {}
        PinnedBuffer(const PinnedBuffer&) = delete;
        PinnedBuffer& operator=(const PinnedBuffer&) = delete;
        PinnedBuffer(PinnedBuffer&& other) noexcept : _pointer(std::exchange(other._pointer, nullptr)) {}
        PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

        [[nodiscard]] void* get() const noexcept {
            return _pointer;
        }
        explicit operator bool() const noexcept {
            return _pointer != nullptr;
        }
        cudaError_t free_after_completion() noexcept;

    private:
        void* _pointer = nullptr;
    };

} // namespace JuicerCuda
