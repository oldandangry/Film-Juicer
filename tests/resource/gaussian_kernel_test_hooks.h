#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "Cuda/JuicerCudaResources.h"

namespace JuicerCuda::GaussianKernelTest {
    struct Participant {
        bool failCopy = false;
        bool failInsertion = false;
        bool copyConsumed = false;
        bool insertionConsumed = false;
        bool arrived = false;
        bool released = false;
        bool done = false;
        bool success = false;
        unsigned copies = 0;
        unsigned frees = 0;
        std::size_t rejectedBytes = 0;
        float* candidate = nullptr;
        void* freed = nullptr;
        bool freeSucceeded = false;
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

        Controls(Resources&, std::uint64_t);
        bool await_arrivals(unsigned);
        bool await_done(unsigned);
        void release(unsigned);
        void cancel() noexcept;
    };
    class Binding final {
    public:
        Binding(Controls&, unsigned);
        ~Binding();
        Binding(const Binding&) = delete;
        Binding& operator=(const Binding&) = delete;

    private:
        Controls& _controls;
        unsigned _participant;
    };
} // namespace JuicerCuda::GaussianKernelTest
