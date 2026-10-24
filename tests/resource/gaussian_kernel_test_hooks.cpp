#include "gaussian_kernel_test_hooks.h"

#include <cuda_runtime.h>

#include "../ffi/spectral_allocation_probe.h"

namespace JuicerCuda::GaussianKernelTest {
    namespace {
        thread_local Controls* controls = nullptr;
        thread_local unsigned participant = 0;
        Participant* match(Resources& resources, std::uint64_t key) noexcept {
            if (!controls || &resources != &controls->resources || key != controls->key ||
                resources.ownerContextKey != controls->context || resources.contextEpoch != controls->epoch)
                return nullptr;
            return &controls->participants[participant];
        }
        constexpr auto kWait = std::chrono::seconds(10);
    } // namespace
    Controls::Controls(Resources& owner, std::uint64_t cacheKey)
        : resources(owner), context(owner.ownerContextKey), epoch(owner.contextEpoch), key(cacheKey) {
    }
    bool Controls::await_arrivals(unsigned count) {
        std::unique_lock lock(mutex);
        if (!changed.wait_for(lock, kWait, [&] {
                unsigned arrived = 0;
                for (const auto& p : participants)
                    arrived += p.arrived ? 1u : 0u;
                return canceled || arrived >= count;
            })) {
            timedOut = canceled = true;
            changed.notify_all();
            return false;
        }
        return !canceled;
    }
    bool Controls::await_done(unsigned index) {
        std::unique_lock lock(mutex);
        if (!changed.wait_for(lock, kWait, [&] {
                return canceled || participants[index].done;
            })) {
            timedOut = canceled = true;
            changed.notify_all();
            return false;
        }
        return !canceled && participants[index].done;
    }
    void Controls::release(unsigned index) {
        std::lock_guard lock(mutex);
        participants[index].released = true;
        changed.notify_all();
    }
    void Controls::cancel() noexcept {
        std::lock_guard lock(mutex);
        canceled = true;
        changed.notify_all();
    }
    Binding::Binding(Controls& value, unsigned index) : _controls(value), _participant(index) {
        if (controls || index >= 2)
            std::terminate();
        controls = &value;
        participant = index;
        SpectralAllocationProbe::clear();
    }
    Binding::~Binding() {
        _controls.participants[_participant].rejectedBytes = SpectralAllocationProbe::bytes();
        SpectralAllocationProbe::clear();
        controls = nullptr;
    }
    std::int32_t copy(Resources& resources, std::uint64_t key, void* destination, const void* source, std::size_t bytes) noexcept {
        auto* p = match(resources, key);
        if (p) {
            ++p->copies;
            p->candidate = static_cast<float*>(destination);
            if (p->failCopy) {
                p->failCopy = false;
                p->copyConsumed = true;
                return static_cast<std::int32_t>(cudaErrorInvalidValue);
            }
        }
        return static_cast<std::int32_t>(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
    }
    void before_insert(Resources& resources, std::uint64_t key) noexcept {
        if (auto* p = match(resources, key); p && p->failInsertion) {
            p->failInsertion = false;
            p->insertionConsumed = true;
            SpectralAllocationProbe::arm();
        }
    }
    bool after_upload(Resources& resources, std::uint64_t key, float* candidate) noexcept {
        auto* p = match(resources, key);
        if (!p || !controls->rendezvous)
            return true;
        std::unique_lock lock(controls->mutex);
        p->candidate = candidate;
        p->arrived = true;
        controls->changed.notify_all();
        if (!controls->changed.wait_for(lock, kWait, [&] {
                return p->released || controls->canceled;
            })) {
            controls->timedOut = controls->canceled = true;
            controls->changed.notify_all();
        }
        return !controls->canceled;
    }
    void observed_free(Resources& resources, std::uint64_t key, void* pointer, bool success) noexcept {
        if (auto* p = match(resources, key)) {
            ++p->frees;
            p->freed = pointer;
            p->freeSucceeded = success;
        }
    }
} // namespace JuicerCuda::GaussianKernelTest
