// Cuda/ResourceManager/JuicerCudaResourceTypes.h
//
// Phase-0 ResourceManager scaffolding types.
// These are intentionally lightweight and non-intrusive: no serving behavior changes.
#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>

namespace JuicerCuda {
namespace ResourceManager {

struct InstanceToken {
    std::uint64_t value = 0;
};

struct FrameToken {
    std::uint64_t value = 0;
};

struct DeviceContextKey {
    int deviceId = -1;
    void* contextOpaque = nullptr;

    bool operator==(const DeviceContextKey& other) const noexcept {
        return deviceId == other.deviceId && contextOpaque == other.contextOpaque;
    }
};

struct DeviceContextKeyHash {
    std::size_t operator()(const DeviceContextKey& key) const noexcept {
        const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
        const std::size_t hDevice = std::hash<int>{}(key.deviceId);
        const std::size_t hContext = std::hash<std::uintptr_t>{}(contextBits);
        return hDevice ^ (hContext + 0x9e3779b9u + (hDevice << 6u) + (hDevice >> 2u));
    }
};

struct KeyDigests {
    std::uint64_t uploadCoreHash = 0;
    std::uint64_t dirHash = 0;
    std::uint64_t scannerHash = 0;
    std::uint64_t autoExposureHash = 0;
};

struct SubmissionSnapshot {
    InstanceToken instanceToken{};
    FrameToken frameToken{};
    std::uint64_t snapshotId = 0;
    DeviceContextKey deviceContextKey{};
    std::uint64_t registryGeneration = 1;
    std::uint64_t contextEpoch = 1;
    KeyDigests keyDigests{};
    std::uint32_t keySchemaVersion = 1;
    std::uint32_t traceSchemaVersion = 1;
};

struct SubmissionTransaction {
    std::uint64_t transactionId = 0;
    std::uint64_t leaseGeneration = 0;
    SubmissionSnapshot snapshot{};
    bool active = false;
    bool committed = false;
};

} // namespace ResourceManager
} // namespace JuicerCuda
