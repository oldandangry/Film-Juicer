#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

#include <nlohmann/json.hpp>

#include "RenderRecipe.h"
#include "Cuda/JuicerCudaResources.h"

namespace JuicerCuda::DirBoundaryTest {
    class Watch final {
    public:
        Watch(ResourceManager::DeviceContextKey, std::uint64_t key);
        ~Watch();
        Watch(const Watch&) = delete;
        Watch& operator=(const Watch&) = delete;
        const ResourceManager::DeviceContextKey context;
        const std::uint64_t key;
        Resources* resources = nullptr;
        std::uint64_t epoch = 0;
        unsigned requests = 0;
    };
    struct Participant {
        bool failUpload = false;
        bool uploadRejected = false;
        bool arrived = false;
        bool released = false;
        bool done = false;
        bool success = false;
        bool observationFailed = false;
        unsigned copies = 0;
        unsigned frees = 0;
        void* candidate = nullptr;
        void* freed = nullptr;
        bool freeSucceeded = false;
        const double* staging = nullptr;
        std::uintptr_t sourceAddress = 0;
        std::vector<double> stagedWeights;
        std::uint64_t hostOwners = 0;
    };
    struct Controls {
        Resources& resources;
        const ResourceManager::DeviceContextKey context;
        const std::uint64_t epoch;
        const std::uint64_t key;
        bool rendezvous = false;
        bool canceled = false;
        bool timedOut = false;
        std::array<Participant, 2> participants{};
        std::mutex mutex;
        std::condition_variable changed;
        Controls(Resources&, std::uint64_t key);
        bool await_arrivals(unsigned count);
        bool await_done(unsigned index);
        void release(unsigned index);
        void cancel() noexcept;
    };
    class Binding final {
    public:
        Binding(Controls&, unsigned index);
        ~Binding();
        Binding(const Binding&) = delete;
        Binding& operator=(const Binding&) = delete;
    };
    void request(Resources&, std::uint64_t key) noexcept;
    std::int32_t staged(Resources&, const char* label, void* candidate, const void* source, const void* staging, std::size_t bytes) noexcept;
    bool after_upload(Resources&, std::uint64_t key, void* candidate) noexcept;
    void observed_free(Resources&, void* pointer, bool success) noexcept;

    Spektrafilm::DirGaussianComponentPlan component(const nlohmann::json&);
} // namespace JuicerCuda::DirBoundaryTest
