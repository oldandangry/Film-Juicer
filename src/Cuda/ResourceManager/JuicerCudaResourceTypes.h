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

enum class AllocatorBackendPreference : std::uint8_t {
    Legacy = 0,
    AsyncPool = 1,
    Slab = 2,
    Auto = 3
};

enum class AllocatorBackendMode : std::uint8_t {
    Legacy = 0,
    AsyncPool = 1,
    Slab = 2
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

enum class ScratchPolicyCandidate : std::uint8_t {
    OpticsBase = 0,
    OpticsBlurred = 1,
    OpticsAux = 2,
    OpticsGrainTriplet = 3,
    OpticsGrainShared = 4,
    OpticsGateMask = 5,
    SpatialDirBase = 6,
    Count = 7
};

struct ScratchPolicyCandidateContractEntry {
    ScratchPolicyCandidate candidate = ScratchPolicyCandidate::OpticsBase;
    const char* label = nullptr;
    const char* telemetryField = nullptr;
};

constexpr std::size_t kScratchPolicyCandidateCount =
    static_cast<std::size_t>(ScratchPolicyCandidate::Count);

constexpr std::size_t scratch_policy_candidate_index(ScratchPolicyCandidate candidate) noexcept {
    const std::size_t idx = static_cast<std::size_t>(candidate);
    return (idx < kScratchPolicyCandidateCount) ? idx : kScratchPolicyCandidateCount;
}

constexpr std::array<ScratchPolicyCandidate, 7> kScratchPolicyCandidateOrder = {
    ScratchPolicyCandidate::OpticsBase,
    ScratchPolicyCandidate::OpticsBlurred,
    ScratchPolicyCandidate::OpticsAux,
    ScratchPolicyCandidate::OpticsGrainTriplet,
    ScratchPolicyCandidate::OpticsGrainShared,
    ScratchPolicyCandidate::OpticsGateMask,
    ScratchPolicyCandidate::SpatialDirBase
};

constexpr std::array<ScratchPolicyCandidateContractEntry, 7> kScratchPolicyCandidateContract = {
    ScratchPolicyCandidateContractEntry{
        ScratchPolicyCandidate::OpticsBase,
        "OpticsBase",
        "optics_base_bytes"
    },
    ScratchPolicyCandidateContractEntry{
        ScratchPolicyCandidate::OpticsBlurred,
        "OpticsBlurred",
        "optics_blurred_bytes"
    },
    ScratchPolicyCandidateContractEntry{
        ScratchPolicyCandidate::OpticsAux,
        "OpticsAux",
        "optics_aux_bytes"
    },
    ScratchPolicyCandidateContractEntry{
        ScratchPolicyCandidate::OpticsGrainTriplet,
        "OpticsGrainTriplet",
        "optics_grain_triplet_bytes"
    },
    ScratchPolicyCandidateContractEntry{
        ScratchPolicyCandidate::OpticsGrainShared,
        "OpticsGrainShared",
        "optics_grain_shared_bytes"
    },
    ScratchPolicyCandidateContractEntry{
        ScratchPolicyCandidate::OpticsGateMask,
        "OpticsGateMask",
        "optics_gate_mask_bytes"
    },
    ScratchPolicyCandidateContractEntry{
        ScratchPolicyCandidate::SpatialDirBase,
        "SpatialDirBase",
        "spatial_dir_base_bytes"
    }
};

constexpr std::array<ScratchPolicyCandidate, 7> kScratchPolicySheddingOrder = {
    ScratchPolicyCandidate::OpticsGateMask,
    ScratchPolicyCandidate::OpticsGrainShared,
    ScratchPolicyCandidate::OpticsGrainTriplet,
    ScratchPolicyCandidate::OpticsAux,
    ScratchPolicyCandidate::OpticsBlurred,
    ScratchPolicyCandidate::SpatialDirBase,
    ScratchPolicyCandidate::OpticsBase
};

constexpr bool scratch_policy_candidate_contract_is_valid() noexcept {
    if (kScratchPolicyCandidateOrder.size() != kScratchPolicyCandidateCount ||
        kScratchPolicyCandidateContract.size() != kScratchPolicyCandidateCount ||
        kScratchPolicySheddingOrder.size() != kScratchPolicyCandidateCount) {
        return false;
    }

    std::array<bool, kScratchPolicyCandidateCount> seenOrder{};
    std::array<bool, kScratchPolicyCandidateCount> seenShedding{};
    for (std::size_t i = 0; i < kScratchPolicyCandidateCount; ++i) {
        const ScratchPolicyCandidate ordered = kScratchPolicyCandidateOrder[i];
        const std::size_t orderedIndex = scratch_policy_candidate_index(ordered);
        if (orderedIndex >= kScratchPolicyCandidateCount || seenOrder[orderedIndex]) {
            return false;
        }
        seenOrder[orderedIndex] = true;

        const ScratchPolicyCandidateContractEntry& entry = kScratchPolicyCandidateContract[i];
        if (entry.candidate != ordered || entry.label == nullptr || entry.telemetryField == nullptr) {
            return false;
        }

        const ScratchPolicyCandidate shedding = kScratchPolicySheddingOrder[i];
        const std::size_t sheddingIndex = scratch_policy_candidate_index(shedding);
        if (sheddingIndex >= kScratchPolicyCandidateCount || seenShedding[sheddingIndex]) {
            return false;
        }
        seenShedding[sheddingIndex] = true;
    }

    for (bool ok : seenOrder) {
        if (!ok) {
            return false;
        }
    }
    for (bool ok : seenShedding) {
        if (!ok) {
            return false;
        }
    }
    return true;
}

static_assert(scratch_policy_candidate_contract_is_valid(),
    "Scratch candidate onboarding contract is incomplete or inconsistent.");

inline const ScratchPolicyCandidateContractEntry& scratch_policy_candidate_contract_entry(
    ScratchPolicyCandidate candidate) noexcept {
    for (const auto& entry : kScratchPolicyCandidateContract) {
        if (entry.candidate == candidate) {
            return entry;
        }
    }
    return kScratchPolicyCandidateContract[0];
}

inline const char* to_cstr(ScratchPolicyCandidate candidate) noexcept {
    return scratch_policy_candidate_contract_entry(candidate).label;
}

enum class ScratchHelperNonPolicyAllocation : std::uint8_t {
    ScanErrorFlag = 0,
    ScanErrorHost = 1,
    AutoExposureExposureScale = 2,
    AutoExposureAutoEV = 3,
    AutoExposureValid = 4,
    AutoExposureMaxYBits = 5,
    AutoExposureHistogram = 6,
    AutoExposureWeightsX = 7,
    AutoExposureWeightsY = 8,
    AutoExposurePartialsA = 9,
    AutoExposurePartialsB = 10,
    Count = 11
};

struct ScratchHelperNonPolicyContractEntry {
    ScratchHelperNonPolicyAllocation allocation = ScratchHelperNonPolicyAllocation::ScanErrorFlag;
    const char* label = nullptr;
    const char* telemetryField = nullptr;
};

constexpr std::size_t kScratchHelperNonPolicyAllocationCount =
    static_cast<std::size_t>(ScratchHelperNonPolicyAllocation::Count);

constexpr std::size_t scratch_helper_non_policy_index(
    ScratchHelperNonPolicyAllocation allocation) noexcept {
    const std::size_t idx = static_cast<std::size_t>(allocation);
    return (idx < kScratchHelperNonPolicyAllocationCount)
        ? idx
        : kScratchHelperNonPolicyAllocationCount;
}

constexpr std::array<ScratchHelperNonPolicyAllocation, 11> kScratchHelperNonPolicyOrder = {
    ScratchHelperNonPolicyAllocation::ScanErrorFlag,
    ScratchHelperNonPolicyAllocation::ScanErrorHost,
    ScratchHelperNonPolicyAllocation::AutoExposureExposureScale,
    ScratchHelperNonPolicyAllocation::AutoExposureAutoEV,
    ScratchHelperNonPolicyAllocation::AutoExposureValid,
    ScratchHelperNonPolicyAllocation::AutoExposureMaxYBits,
    ScratchHelperNonPolicyAllocation::AutoExposureHistogram,
    ScratchHelperNonPolicyAllocation::AutoExposureWeightsX,
    ScratchHelperNonPolicyAllocation::AutoExposureWeightsY,
    ScratchHelperNonPolicyAllocation::AutoExposurePartialsA,
    ScratchHelperNonPolicyAllocation::AutoExposurePartialsB
};

constexpr std::array<ScratchHelperNonPolicyContractEntry, 11> kScratchHelperNonPolicyContract = {
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::ScanErrorFlag,
        "ScanErrorFlag",
        "scan_error_flag_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::ScanErrorHost,
        "ScanErrorHost",
        "scan_error_host_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposureExposureScale,
        "AutoExposureExposureScale",
        "auto_exposure_scale_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposureAutoEV,
        "AutoExposureAutoEV",
        "auto_exposure_ev_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposureValid,
        "AutoExposureValid",
        "auto_exposure_valid_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposureMaxYBits,
        "AutoExposureMaxYBits",
        "auto_exposure_max_y_bits_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposureHistogram,
        "AutoExposureHistogram",
        "auto_exposure_histogram_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposureWeightsX,
        "AutoExposureWeightsX",
        "auto_exposure_weights_x_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposureWeightsY,
        "AutoExposureWeightsY",
        "auto_exposure_weights_y_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposurePartialsA,
        "AutoExposurePartialsA",
        "auto_exposure_partials_a_bytes"
    },
    ScratchHelperNonPolicyContractEntry{
        ScratchHelperNonPolicyAllocation::AutoExposurePartialsB,
        "AutoExposurePartialsB",
        "auto_exposure_partials_b_bytes"
    }
};

constexpr bool scratch_helper_non_policy_contract_is_valid() noexcept {
    if (kScratchHelperNonPolicyOrder.size() != kScratchHelperNonPolicyAllocationCount ||
        kScratchHelperNonPolicyContract.size() != kScratchHelperNonPolicyAllocationCount) {
        return false;
    }

    std::array<bool, kScratchHelperNonPolicyAllocationCount> seen{};
    for (std::size_t i = 0; i < kScratchHelperNonPolicyAllocationCount; ++i) {
        const ScratchHelperNonPolicyAllocation allocation = kScratchHelperNonPolicyOrder[i];
        const std::size_t idx = scratch_helper_non_policy_index(allocation);
        if (idx >= kScratchHelperNonPolicyAllocationCount || seen[idx]) {
            return false;
        }
        seen[idx] = true;
        const ScratchHelperNonPolicyContractEntry& entry = kScratchHelperNonPolicyContract[i];
        if (entry.allocation != allocation || entry.label == nullptr || entry.telemetryField == nullptr) {
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

static_assert(scratch_helper_non_policy_contract_is_valid(),
    "Scratch helper-owned non-policy onboarding contract is incomplete or inconsistent.");

inline const ScratchHelperNonPolicyContractEntry& scratch_helper_non_policy_contract_entry(
    ScratchHelperNonPolicyAllocation allocation) noexcept {
    for (const auto& entry : kScratchHelperNonPolicyContract) {
        if (entry.allocation == allocation) {
            return entry;
        }
    }
    return kScratchHelperNonPolicyContract[0];
}

inline const char* to_cstr(ScratchHelperNonPolicyAllocation allocation) noexcept {
    return scratch_helper_non_policy_contract_entry(allocation).label;
}

constexpr std::uint64_t kScratchNormalizationHysteresisBytes =
    128ull * 1024ull * 1024ull;

struct ScratchRequestDescriptor {
    bool needOptics = false;
    bool needSpatialDir = false;
    int requestedWidth = 0;
    int requestedHeight = 0;
    bool needBlurred = false;
    bool needAux = false;
    bool needGrainTriplet = false;
    bool needGrainShared = false;
    bool needGateMask = false;
    std::uint64_t generation = 0;

    bool has_any_family() const noexcept {
        return needOptics || needSpatialDir;
    }
};

inline bool scratch_request_descriptor_is_valid(const ScratchRequestDescriptor& descriptor) noexcept {
    if (!descriptor.has_any_family()) {
        return false;
    }
    if (descriptor.requestedWidth <= 0 || descriptor.requestedHeight <= 0) {
        return false;
    }
    if (!descriptor.needOptics &&
        (descriptor.needBlurred ||
         descriptor.needAux ||
         descriptor.needGrainTriplet ||
         descriptor.needGrainShared ||
         descriptor.needGateMask)) {
        return false;
    }
    return true;
}

inline std::uint64_t hash_scratch_request_descriptor(const ScratchRequestDescriptor& descriptor) noexcept {
    constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    std::uint64_t hash = kFnvOffset;
    auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= kFnvPrime;
    };
    mix(descriptor.needOptics ? 1ull : 0ull);
    mix(descriptor.needSpatialDir ? 1ull : 0ull);
    mix(static_cast<std::uint64_t>(descriptor.requestedWidth));
    mix(static_cast<std::uint64_t>(descriptor.requestedHeight));
    mix(descriptor.needBlurred ? 1ull : 0ull);
    mix(descriptor.needAux ? 1ull : 0ull);
    mix(descriptor.needGrainTriplet ? 1ull : 0ull);
    mix(descriptor.needGrainShared ? 1ull : 0ull);
    mix(descriptor.needGateMask ? 1ull : 0ull);
    return hash;
}

inline ScratchRequestDescriptor make_scratch_request_descriptor(
    bool needOptics,
    bool needSpatialDir,
    int requestedWidth,
    int requestedHeight,
    bool needBlurred,
    bool needAux,
    bool needGrainTriplet,
    bool needGrainShared,
    bool needGateMask) noexcept {
    ScratchRequestDescriptor descriptor{};
    descriptor.needOptics = needOptics;
    descriptor.needSpatialDir = needSpatialDir;
    descriptor.requestedWidth = requestedWidth;
    descriptor.requestedHeight = requestedHeight;
    descriptor.needBlurred = needOptics && needBlurred;
    descriptor.needAux = needOptics && needAux;
    descriptor.needGrainTriplet = needOptics && needGrainTriplet;
    descriptor.needGrainShared = needOptics && needGrainShared;
    descriptor.needGateMask = needOptics && needGateMask;
    descriptor.generation = hash_scratch_request_descriptor(descriptor);
    return descriptor;
}

struct ScratchStage1DecisionState {
    std::uint64_t policyLiveRetainedBytes = 0;
    std::uint64_t retainedGeneration = 1;
};

struct ScratchResidencyCandidateRow {
    ScratchPolicyCandidate candidate = ScratchPolicyCandidate::OpticsBase;
    std::uint64_t liveRetainedBytes = 0;
};

struct ScratchResidencyView {
    std::array<ScratchResidencyCandidateRow, kScratchPolicyCandidateCount> candidates{};
    std::array<std::uint64_t, kScratchHelperNonPolicyAllocationCount> helperNonPolicyBytes{};
    std::uint64_t helperSharedBytes = 0;
    std::uint64_t helperNonPolicyTotalBytes = 0;
    std::uint64_t policyLiveRetainedBytes = 0;
    std::uint64_t totalLiveRetainedBytes = 0;
    std::uint64_t retirePendingScratchBytes = 0;
    std::uint64_t retainedGeneration = 1;
    bool overflow = false;
};

// Trace schema contract: single source of truth for submission traces.
// Bump only when required trace fields/tags or their required semantics change.
constexpr std::uint32_t kTraceSchemaVersion = 4u;

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
