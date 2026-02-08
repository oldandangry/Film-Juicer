// Cuda/ResourceManager/JuicerCudaResourceTypes.h
//
// Phase-0 ResourceManager scaffolding types.
// These are intentionally lightweight and non-intrusive: no serving behavior changes.
#pragma once

#include <cstdint>

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
};

struct KeyDigests {
    std::uint64_t uploadCoreHash = 0;
    std::uint64_t dirHash = 0;
    std::uint64_t scannerHash = 0;
};

struct SubmissionSnapshot {
    InstanceToken instanceToken{};
    FrameToken frameToken{};
    std::uint64_t snapshotId = 0;
    DeviceContextKey deviceContextKey{};
    KeyDigests keyDigests{};
    std::uint32_t keySchemaVersion = 1;
    std::uint32_t traceSchemaVersion = 1;
};

struct SubmissionTransaction {
    std::uint64_t transactionId = 0;
    SubmissionSnapshot snapshot{};
    bool active = false;
    bool committed = false;
};

} // namespace ResourceManager
} // namespace JuicerCuda

