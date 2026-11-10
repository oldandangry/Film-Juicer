#include "boundary_test_hooks.h"

#include <bit>
#include <chrono>
#include <exception>
#include <string_view>

#include <cuda_runtime.h>

#include "../ffi/juicer_test_api.h"

namespace JuicerCuda::DirBoundaryTest {
    namespace {
        thread_local Watch* watch = nullptr;
        thread_local Controls* controls = nullptr;
        thread_local unsigned participant = 0;
        thread_local std::uint64_t requestKey = 0;
        constexpr auto kWait = std::chrono::seconds(10);
        Participant* match(Resources& resources, std::uint64_t key) noexcept {
            if (!controls || &resources != &controls->resources || key != controls->key ||
                resources.ownerContextKey != controls->context || resources.contextEpoch != controls->epoch)
                return nullptr;
            return &controls->participants[participant];
        }
        using Json = nlohmann::json;
        double scalar(const Json& value) {
            return std::bit_cast<double>(value.get<std::uint64_t>());
        }
    } // namespace

    Watch::Watch(ResourceManager::DeviceContextKey expectedContext, std::uint64_t expectedKey)
        : context(expectedContext), key(expectedKey) {
        if (watch)
            std::terminate();
        watch = this;
    }
    Watch::~Watch() {
        watch = nullptr;
    }
    Controls::Controls(Resources& owner, std::uint64_t cacheKey)
        : resources(owner), context(owner.ownerContextKey), epoch(owner.contextEpoch), key(cacheKey) {}
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
    Binding::Binding(Controls& value, unsigned index) {
        if (controls || index >= 2)
            std::terminate();
        controls = &value;
        participant = index;
    }
    Binding::~Binding() {
        controls = nullptr;
        requestKey = 0;
    }
    void request(Resources& resources, std::uint64_t key) noexcept {
        requestKey = key;
        if (watch && resources.ownerContextKey == watch->context && key == watch->key) {
            watch->resources = &resources;
            watch->epoch = resources.contextEpoch;
            ++watch->requests;
        }
    }
    std::int32_t staged(Resources& resources, const char* label, void* candidate, const void* source, const void* staging, std::size_t bytes) noexcept {
        auto* p = match(resources, requestKey);
        if (!p || !label || std::string_view(label) != "spatial DIR reflected boundary")
            return static_cast<std::int32_t>(cudaSuccess);
        ++p->copies;
        p->candidate = candidate;
        p->staging = static_cast<const double*>(staging);
        p->sourceAddress = reinterpret_cast<std::uintptr_t>(source);
        p->hostOwners = fj_test_dir_boundary_count(3);
        if (candidate == source || candidate == staging || source == staging) {
            p->observationFailed = true;
            return static_cast<std::int32_t>(cudaErrorInvalidValue);
        }
        try {
            p->stagedWeights.assign(p->staging, p->staging + bytes / sizeof(double));
        } catch (...) {
            p->observationFailed = true;
            return static_cast<std::int32_t>(cudaErrorMemoryAllocation);
        }
        if (p->failUpload) {
            p->failUpload = false;
            p->uploadRejected = true;
            return static_cast<std::int32_t>(cudaErrorInvalidValue);
        }
        return static_cast<std::int32_t>(cudaSuccess);
    }
    bool after_upload(Resources& resources, std::uint64_t key, void* candidate) noexcept {
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
    void observed_free(Resources& resources, void* pointer, bool success) noexcept {
        if (auto* p = match(resources, requestKey); p && p->candidate == pointer) {
            ++p->frees;
            p->freed = pointer;
            p->freeSucceeded = success;
        }
    }

    Spektrafilm::DirGaussianComponentPlan component(const Json& input) {
        Spektrafilm::DirGaussianComponentPlan result{};
        result.sigmaPixels = std::bit_cast<float>(input.at("sigma").get<std::uint32_t>());
        result.weight = std::bit_cast<float>(input.at("weight").get<std::uint32_t>());
        result.radius = input.at("radius");
        result.referenceOperator = static_cast<Spektrafilm::DirReferenceOperator>(input.at("operator").get<unsigned>());
        result.targetScratchTier = static_cast<Spektrafilm::DirScratchTier>(input.at("target").get<unsigned>());
        result.iir.feedforward = scalar(input.at("feedforward"));
        result.iir.normalizationDenominator = scalar(input.at("denominator"));
        result.iir.feedforwardNumerator = scalar(input.at("numerator"));
        for (std::size_t index = 0; index < 3; ++index) {
            result.iir.feedback[index] = scalar(input.at("feedback").at(index));
            result.iir.feedbackNumerators[index] = scalar(input.at("numerators").at(index));
        }
        result.boundaryTruncationAccuracy = scalar(input.at("accuracy"));
        result.boundaryCertificationTolerance = scalar(input.at("tolerance"));
        result.boundaryDerivationVersion = input.at("version");
        return result;
    }

} // namespace JuicerCuda::DirBoundaryTest
