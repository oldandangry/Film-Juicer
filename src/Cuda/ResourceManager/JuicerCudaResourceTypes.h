// Cuda/ResourceManager/JuicerCudaResourceTypes.h
//
// Phase-0 ResourceManager scaffolding types.
// These are intentionally lightweight and non-intrusive: no serving behavior changes.
#pragma once

#include <array>
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

enum class ResourceKind : std::uint8_t {
    UploadCore = 0,
    Dir = 1,
    Scanner = 2,
    AutoExposure = 3,
    Count = 4
};

enum class ResourceTier : std::uint8_t {
    Immutable = 0,
    Lut = 1,
    Scratch = 2,
    Graph = 3
};

struct ResourceKindContractEntry {
    ResourceKind kind = ResourceKind::UploadCore;
    ResourceTier tier = ResourceTier::Immutable;
    const char* keyField = nullptr;
    const char* invalidationLane = nullptr;
    const char* resourceNode = nullptr;
    const char* acquireStatusField = nullptr;
    const char* telemetryTag = nullptr;
};

constexpr std::size_t kResourceKindCount = static_cast<std::size_t>(ResourceKind::Count);

constexpr bool resource_kind_is_valid(ResourceKind kind) noexcept {
    return static_cast<std::size_t>(kind) < kResourceKindCount;
}

constexpr std::size_t resource_kind_index(ResourceKind kind) noexcept {
    const std::size_t idx = static_cast<std::size_t>(kind);
    return (idx < kResourceKindCount) ? idx : kResourceKindCount;
}

constexpr std::array<ResourceKind, 4> kResourceKindOrder = {
    ResourceKind::UploadCore,
    ResourceKind::Dir,
    ResourceKind::Scanner,
    ResourceKind::AutoExposure
};

constexpr std::array<ResourceKindContractEntry, 4> kResourceKindContract = {
    ResourceKindContractEntry{
        ResourceKind::UploadCore,
        ResourceTier::Immutable,
        "uploadCoreHash",
        "UploadCoreKey",
        "UploadCoreResources",
        "upload_status",
        "MSUPL"
    },
    ResourceKindContractEntry{
        ResourceKind::Dir,
        ResourceTier::Immutable,
        "dirHash",
        "DirKey",
        "DirResources",
        "dir_status",
        "MSDIR"
    },
    ResourceKindContractEntry{
        ResourceKind::Scanner,
        ResourceTier::Lut,
        "scannerHash",
        "ScannerColorKey",
        "ScannerColorResources",
        "scanner_status",
        "MSSCN"
    },
    ResourceKindContractEntry{
        ResourceKind::AutoExposure,
        ResourceTier::Scratch,
        "autoExposureHash",
        "AutoExposureKey",
        "AutoExposureResources",
        "auto_exposure_status",
        "MSAEX"
    }
};

constexpr bool resource_kind_contract_is_valid() noexcept {
    if (kResourceKindOrder.size() != kResourceKindCount || kResourceKindContract.size() != kResourceKindCount) {
        return false;
    }
    std::array<bool, kResourceKindCount> seen{};
    for (std::size_t i = 0; i < kResourceKindCount; ++i) {
        const ResourceKind kind = kResourceKindOrder[i];
        const std::size_t idx = resource_kind_index(kind);
        if (idx >= kResourceKindCount || seen[idx]) {
            return false;
        }
        seen[idx] = true;
        const ResourceKindContractEntry& entry = kResourceKindContract[i];
        if (entry.kind != kind ||
            entry.keyField == nullptr ||
            entry.invalidationLane == nullptr ||
            entry.resourceNode == nullptr ||
            entry.acquireStatusField == nullptr ||
            entry.telemetryTag == nullptr) {
            return false;
        }
    }
    for (bool ok : seen) {
        if (!ok) {
            return false;
        }
    }
    return true;
}

static_assert(resource_kind_contract_is_valid(),
    "ResourceKind onboarding contract is incomplete: update Types/Keys/Policy/State/Telemetry for every ResourceKind.");

inline const ResourceKindContractEntry& resource_kind_contract_entry(ResourceKind kind) noexcept {
    for (const auto& entry : kResourceKindContract) {
        if (entry.kind == kind) {
            return entry;
        }
    }
    return kResourceKindContract[0];
}

inline const char* to_cstr(ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::UploadCore:
        return "UploadCore";
    case ResourceKind::Dir:
        return "Dir";
    case ResourceKind::Scanner:
        return "Scanner";
    case ResourceKind::AutoExposure:
        return "AutoExposure";
    default:
        return "Unknown";
    }
}

// Trace schema contract: single source of truth for submission traces.
// Bump only when required trace fields/tags or their required semantics change.
constexpr std::uint32_t kTraceSchemaVersion = 1u;

constexpr std::uint32_t sanitize_trace_schema_version(std::uint32_t value) noexcept {
    return (value == 0u) ? kTraceSchemaVersion : value;
}

constexpr bool trace_schema_matches_contract(std::uint32_t value) noexcept {
    return sanitize_trace_schema_version(value) == kTraceSchemaVersion;
}

static_assert(kTraceSchemaVersion >= 1u,
    "trace schema version must be a positive non-zero value");

struct SubmissionSnapshot {
    InstanceToken instanceToken{};
    FrameToken frameToken{};
    std::uint64_t snapshotId = 0;
    DeviceContextKey deviceContextKey{};
    std::uint64_t registryGeneration = 1;
    std::uint64_t contextEpoch = 1;
    KeyDigests keyDigests{};
    std::uint32_t keySchemaVersion = 1;
    std::uint32_t traceSchemaVersion = kTraceSchemaVersion;
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
