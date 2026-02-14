// Cuda/ResourceManager/JuicerCudaResourceManager.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaManagerRegistry.h"
#include "Cuda/ResourceManager/JuicerCudaResourceConfig.h"
#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"
#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"
#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"
#include "Print.h"
#include "WorkingState.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
extern "C" cudaError_t juicer_cuda_negative_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_pipeline(
    const JuicerCuda::PipelineRunParams* hParams,
    void* cudaStreamOpaque);
#endif

namespace JuicerCuda {
namespace ResourceManager {

namespace {

struct ShadowHistoryKey {
    std::uint64_t instanceToken = 0;
    DeviceContextKey deviceContextKey{};

    bool operator==(const ShadowHistoryKey& other) const noexcept {
        return instanceToken == other.instanceToken &&
            deviceContextKey == other.deviceContextKey;
    }
};

struct ShadowHistoryKeyHasher {
    std::size_t operator()(const ShadowHistoryKey& key) const noexcept {
        const std::size_t hInstance = std::hash<std::uint64_t>{}(key.instanceToken);
        const std::size_t hContext = DeviceContextKeyHash{}(key.deviceContextKey);
        return hInstance ^ (hContext + 0x9e3779b9u + (hInstance << 6u) + (hInstance >> 2u));
    }
};

struct ShadowHistoryEntry {
    bool valid = false;
    KeyDigests digests{};
    std::uint32_t keySchemaVersion = 1;
    std::uint64_t snapshotId = 0;
};

struct ShadowHistoryState {
    std::mutex mutex;
    std::unordered_map<ShadowHistoryKey, ShadowHistoryEntry, ShadowHistoryKeyHasher> bySubmissionKey;
};

ShadowHistoryState& shadow_history_state() noexcept {
    static ShadowHistoryState state{};
    return state;
}

struct AutoExposureOwnershipEntry {
    bool valid = false;
    std::uint64_t keyHash = 0;
    int meterWidth = 0;
    int meterHeight = 0;
    std::uint32_t keySchemaVersion = 1;
};

struct AutoExposureOwnershipState {
    std::mutex mutex;
    std::unordered_map<ShadowHistoryKey, AutoExposureOwnershipEntry, ShadowHistoryKeyHasher> bySubmissionKey;
};

AutoExposureOwnershipState& auto_exposure_ownership_state() noexcept {
    static AutoExposureOwnershipState state{};
    return state;
}

struct FrameSnapshotKey {
    std::uint64_t instanceToken = 0;
    DeviceContextKey deviceContextKey{};

    bool operator==(const FrameSnapshotKey& other) const noexcept {
        return instanceToken == other.instanceToken &&
            deviceContextKey == other.deviceContextKey;
    }
};

struct FrameSnapshotKeyHasher {
    std::size_t operator()(const FrameSnapshotKey& key) const noexcept {
        const std::size_t hInstance = std::hash<std::uint64_t>{}(key.instanceToken);
        const std::size_t hContext = DeviceContextKeyHash{}(key.deviceContextKey);
        return hInstance ^ (hContext + 0x9e3779b9u + (hInstance << 6u) + (hInstance >> 2u));
    }
};

struct FrameSnapshotEntry {
    bool valid = false;
    std::uint64_t frameToken = 0;
    KeyDigests digests{};
    std::uint32_t keySchemaVersion = 1;
    std::uint64_t snapshotId = 0;
};

struct FrameSnapshotState {
    std::mutex mutex;
    std::unordered_map<FrameSnapshotKey, FrameSnapshotEntry, FrameSnapshotKeyHasher> bySubmissionKey;
};

FrameSnapshotState& frame_snapshot_state() noexcept {
    static FrameSnapshotState state{};
    return state;
}

enum class ScratchWorkClass : std::uint8_t {
    Optics = 0,
    SpatialDir = 1
};

enum class PressureLane : std::uint8_t {
    Builder = 0,
    Upload = 1
};

enum class BuilderReservationTier : std::uint8_t {
    Scratch = 0,
    Lut = 1,
    Graph = 2
};

const char* to_cstr(ScratchWorkClass workClass) noexcept {
    switch (workClass) {
    case ScratchWorkClass::Optics:
        return "optics";
    case ScratchWorkClass::SpatialDir:
        return "spatial_dir";
    default:
        return "unknown";
    }
}

const char* to_cstr(PressureLane lane) noexcept {
    switch (lane) {
    case PressureLane::Builder:
        return "builder";
    case PressureLane::Upload:
        return "upload";
    default:
        return "unknown";
    }
}

const char* to_cstr(BuilderReservationTier tier) noexcept {
    switch (tier) {
    case BuilderReservationTier::Scratch:
        return "scratch";
    case BuilderReservationTier::Lut:
        return "lut";
    case BuilderReservationTier::Graph:
        return "graph";
    default:
        return "unknown";
    }
}

struct ScratchBucketKey {
    std::uint32_t widthBucket = 0;
    std::uint32_t heightBucket = 0;
    std::uint16_t bucketStepPx = 0;
    ScratchWorkClass workClass = ScratchWorkClass::Optics;
    bool largeFrame = false;

    bool operator==(const ScratchBucketKey& other) const noexcept {
        return widthBucket == other.widthBucket &&
            heightBucket == other.heightBucket &&
            bucketStepPx == other.bucketStepPx &&
            workClass == other.workClass &&
            largeFrame == other.largeFrame;
    }
};

struct ScratchBucketKeyHash {
    std::size_t operator()(const ScratchBucketKey& key) const noexcept {
        const std::size_t hW = std::hash<std::uint32_t>{}(key.widthBucket);
        const std::size_t hH = std::hash<std::uint32_t>{}(key.heightBucket);
        const std::size_t hStep = std::hash<std::uint16_t>{}(key.bucketStepPx);
        const std::size_t hWork = std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(key.workClass));
        const std::size_t hLarge = std::hash<bool>{}(key.largeFrame);
        return (((hW ^ (hH + 0x9e3779b9u + (hW << 6u) + (hW >> 2u)))
            ^ (hStep + 0x9e3779b9u + (hW << 6u) + (hW >> 2u)))
            ^ (hWork + 0x9e3779b9u + (hW << 6u) + (hW >> 2u)))
            ^ (hLarge + 0x9e3779b9u + (hW << 6u) + (hW >> 2u));
    }
};

struct ScratchBucketEntry {
    std::size_t inFlightBytes = 0;
    std::uint32_t inFlightSets = 0;
    std::uint64_t attemptCount = 0;
    std::uint64_t exhaustedCount = 0;
    std::uint64_t allocGrowthEvents = 0;
    std::uint64_t reuseEvents = 0;
    bool starvationLatched = false;
};

struct ScratchQuarantineEntry {
    ScratchBucketKey key{};
    std::size_t bytes = 0;
    std::uint64_t touchedMs = 0;
    std::uint64_t sequence = 0;
};

struct ScratchContextState {
    std::unordered_map<ScratchBucketKey, ScratchBucketEntry, ScratchBucketKeyHash> buckets;
    std::vector<ScratchQuarantineEntry> largeFrameQuarantine;
    std::size_t largeFrameQuarantineBytes = 0;
    std::size_t totalInFlightBytes = 0;
    std::uint64_t nextQuarantineSequence = 1;
};

struct ScratchPolicyState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, ScratchContextState, DeviceContextKeyHash> byContext;
    std::uint64_t totalInFlightBytes = 0;
};

ScratchPolicyState& scratch_policy_state() noexcept {
    static ScratchPolicyState state{};
    return state;
}

struct UploadReservationContextState {
    std::uint64_t inFlightBytes = 0;
    struct FairnessEntry {
        std::uint32_t sharedTokens = 0;
        std::uint32_t criticalTokens = 0;
        std::uint64_t lastRefillMs = 0;
    };
    std::unordered_map<std::uint64_t, FairnessEntry> fairnessByInstance;
};

struct UploadReservationState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, UploadReservationContextState, DeviceContextKeyHash> byContext;
    std::uint64_t totalInFlightBytes = 0;
};

UploadReservationState& upload_reservation_state() noexcept {
    static UploadReservationState state{};
    return state;
}

struct BuilderReservationContextState {
    std::uint64_t inFlightScratchBytes = 0;
    std::uint64_t inFlightLutBytes = 0;
    std::uint64_t inFlightGraphBytes = 0;
    struct FairnessEntry {
        std::uint32_t sharedTokens = 0;
        std::uint32_t criticalTokens = 0;
        std::uint64_t lastRefillMs = 0;
    };
    std::unordered_map<std::uint64_t, FairnessEntry> fairnessByInstance;
};

struct BuilderReservationState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, BuilderReservationContextState, DeviceContextKeyHash> byContext;
    std::uint64_t totalScratchInFlightBytes = 0;
    std::uint64_t totalLutInFlightBytes = 0;
    std::uint64_t totalGraphInFlightBytes = 0;
};

BuilderReservationState& builder_reservation_state() noexcept {
    static BuilderReservationState state{};
    return state;
}

struct PressureContextState {
    bool valid = false;
    std::uint64_t lastSampleMs = 0;
    PressureDecision lastDecision{};
    PressureState lastState = PressureState::Normal;
    std::uint64_t lastStateChangeMs = 0;
    std::uint64_t transitionWindowStartMs = 0;
    std::uint32_t transitionsInWindow = 0;
    bool reserveCrossed = false;
    bool headroomSourceValid = false;
    HeadroomSource lastHeadroomSource = HeadroomSource::FreeVramOnly;
};

struct PressurePolicyState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, PressureContextState, DeviceContextKeyHash> byContext;
};

PressurePolicyState& pressure_policy_state() noexcept {
    static PressurePolicyState state{};
    return state;
}

struct ManagerMemorySnapshot {
    std::uint64_t activeBytes = 0;
    std::uint64_t reclaimableBytes = 0;
    std::uint64_t retirePendingBytes = 0;
    std::uint64_t transientNonManagerBytes = 0;
    bool overflow = false;
};

struct HeadroomTelemetry {
    std::uint64_t effectiveHeadroomBytes = 0;
    std::uint64_t driverFreeBytes = 0;
    std::uint64_t allocatorPoolReservedBytes = 0;
    std::uint64_t allocatorPoolUsedBytes = 0;
    HeadroomSource source = HeadroomSource::FreeVramOnly;
    bool poolTelemetryAvailable = false;
};

struct PressureCheckpoint {
    PressureInput input{};
    PressureDecision decision{};
    PressureState previousState = PressureState::Normal;
    PressureState desiredState = PressureState::Normal;
    bool sampled = false;
    bool transition = false;
    bool transitionDeferredByDwell = false;
    bool transitionDeferredByRate = false;
    bool reserveCrossing = false;
    bool reserveCrossedNow = false;
    std::uint32_t pollIntervalMs = 0;
    ManagerMemorySnapshot memory{};
};

struct ScratchPolicyClaim {
    DeviceContextKey contextKey{};
    ScratchBucketKey bucketKey{};
    std::size_t bytes = 0;
    bool acquired = false;
};

struct ScratchPolicySnapshot {
    std::size_t inFlightBytes = 0;
    std::uint32_t inFlightSets = 0;
    std::size_t quarantineBytes = 0;
    std::uint32_t quarantineEntries = 0;
    std::uint64_t bucketAttempts = 0;
    std::uint64_t bucketExhausted = 0;
    std::uint64_t bucketAllocGrowth = 0;
    std::uint64_t bucketReuse = 0;
    bool starvationLatched = false;
    ScratchBucketKey bucketKey{};
};

struct ReservationAttemptInfo {
    ReservationDecision decision{};
    std::uint64_t bytesInFlight = 0;
    std::uint64_t capBytes = 0;
    std::uint64_t thresholdBytes = 0;
    std::uint64_t instanceToken = 0;
    std::uint32_t sharedTokens = 0;
    std::uint32_t criticalTokens = 0;
    bool considered = false;
};

struct UploadReservationClaim {
    DeviceContextKey contextKey{};
    std::uint64_t bytes = 0;
    bool acquired = false;
};

struct BuilderReservationClaim {
    DeviceContextKey contextKey{};
    BuilderReservationTier tier = BuilderReservationTier::Scratch;
    std::uint64_t bytes = 0;
    bool acquired = false;
};

constexpr std::uint32_t kMaxTempScratchSets = 1;
constexpr std::size_t kMaxTempScratchBytes = static_cast<std::size_t>(1024ull * 1024ull * 1024ull);
constexpr int kScratchWaitStepMs = 1;
constexpr int kScratchWaitMaxMs = 4;
constexpr int kScratchWaitMaxMediumMs = 16;
constexpr int kScratchWaitMaxLargeMs = 64;
constexpr std::size_t kScratchWaitMediumRequestBytes = static_cast<std::size_t>(256ull * 1024ull * 1024ull);
constexpr std::size_t kScratchWaitLargeRequestBytes = static_cast<std::size_t>(768ull * 1024ull * 1024ull);
constexpr int kScratchBucketStepBasePx = 64;
constexpr int kScratchBucketStepLargePx = 128;
constexpr int kScratchBucketStepXLargePx = 256;
constexpr int kScratchBucketMinPx = 128;
constexpr std::uint64_t kLargeFrameThresholdPixels = static_cast<std::uint64_t>(7680ull * 4320ull);
constexpr std::size_t kLargeFrameQuarantineMaxBytes = static_cast<std::size_t>(1024ull * 1024ull * 1024ull);
constexpr std::size_t kLargeFrameQuarantineMaxEntries = 2;
constexpr std::uint64_t kLargeFrameQuarantineDecayMs = 2000;
constexpr const char* kScratchExhaustedPrefix = "scratch_exhausted:";
constexpr const char* kReservationDeferredPrefix = "reservation_deferred:";
constexpr std::uint64_t kTransientReservationCapDefaultBytes = 512ull * 1024ull * 1024ull;
constexpr std::uint64_t kTransientReservationThresholdDefaultBytes = 64ull * 1024ull * 1024ull;
constexpr std::uint64_t kScratchBuilderReservationCapDefaultBytes = 256ull * 1024ull * 1024ull;
constexpr std::uint64_t kLutBuilderReservationCapDefaultBytes = 128ull * 1024ull * 1024ull;
constexpr std::uint64_t kGraphBuilderReservationCapDefaultBytes = 128ull * 1024ull * 1024ull;
constexpr std::uint64_t kBuilderReservationThresholdDefaultBytes = 16ull * 1024ull * 1024ull;
constexpr std::uint64_t kUploadReservationCapDefaultBytes = 256ull * 1024ull * 1024ull;
constexpr std::uint64_t kUploadReservationThresholdDefaultBytes = 16ull * 1024ull * 1024ull;
constexpr std::uint64_t kStbnUploadDefaultBytes = 512ull * 512ull * 256ull;
constexpr std::uint64_t kWangTilesUploadDefaultBytes = 256ull * 256ull * 16ull;
constexpr std::uint64_t kWangLutUploadDefaultBytes = 8ull * 8ull * 8ull * 8ull;
constexpr std::uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr int kBuilderReservationWaitStepMs = 1;
constexpr int kBuilderReservationWaitMaxMs = 8;
constexpr std::uint64_t kBuilderFairnessTickMs = 4;
constexpr int kUploadReservationWaitStepMs = 1;
constexpr int kUploadReservationWaitMaxMs = 8;
constexpr std::uint64_t kUploadFairnessTickMs = 4;

const ResourceManagerConfigEffective& manager_effective_config() noexcept {
    static const ResourceManagerConfigEffective cfg = sanitize_config(ResourceManagerConfigRaw{});
    return cfg;
}

bool contains_ascii_case_insensitive(const std::string& haystack, const char* needle) noexcept {
    if (!needle || !*needle) {
        return true;
    }
    if (haystack.empty()) {
        return false;
    }
    const std::size_t needleLen = std::char_traits<char>::length(needle);
    if (needleLen == 0 || needleLen > haystack.size()) {
        return false;
    }
    for (std::size_t i = 0; i + needleLen <= haystack.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needleLen; ++j) {
            const unsigned char a = static_cast<unsigned char>(haystack[i + j]);
            const unsigned char b = static_cast<unsigned char>(needle[j]);
            if (std::tolower(a) != std::tolower(b)) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

bool is_allocator_oom_error(const std::string& error) noexcept {
    if (error.empty()) {
        return false;
    }
    return contains_ascii_case_insensitive(error, "out of memory") ||
        contains_ascii_case_insensitive(error, "memory allocation");
}

inline std::uint64_t monotonic_time_ms() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline std::uint64_t area_pixels_for_extent(int width, int height) noexcept {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const std::uint64_t w = static_cast<std::uint64_t>(width);
    const std::uint64_t h = static_cast<std::uint64_t>(height);
    if (w != 0 && h > (std::numeric_limits<std::uint64_t>::max() / w)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return w * h;
}

inline bool is_large_frame_extent(int width, int height) noexcept {
    const std::uint64_t area = area_pixels_for_extent(width, height);
    const int maxDim = std::max(width, height);
    return maxDim >= 7680 || area >= kLargeFrameThresholdPixels;
}

inline int adaptive_bucket_step_px(int width, int height) noexcept {
    const int maxDim = std::max(width, height);
    const std::uint64_t area = area_pixels_for_extent(width, height);
    if (maxDim >= 8192 || area >= static_cast<std::uint64_t>(8192ull * 4320ull)) {
        return kScratchBucketStepXLargePx;
    }
    if (maxDim >= 4096 || area >= static_cast<std::uint64_t>(4096ull * 2160ull)) {
        return kScratchBucketStepLargePx;
    }
    return kScratchBucketStepBasePx;
}

inline std::uint32_t round_up_bucket_dim(int value, int step) noexcept {
    const int safeStep = std::max(1, step);
    const int clampedValue = std::max(kScratchBucketMinPx, value);
    const int rounded = ((clampedValue + safeStep - 1) / safeStep) * safeStep;
    return static_cast<std::uint32_t>(std::max(kScratchBucketMinPx, rounded));
}

inline ScratchBucketKey make_scratch_bucket_key(
    int width,
    int height,
    ScratchWorkClass workClass) noexcept {
    const int step = adaptive_bucket_step_px(width, height);
    ScratchBucketKey key{};
    key.widthBucket = round_up_bucket_dim(width, step);
    key.heightBucket = round_up_bucket_dim(height, step);
    key.bucketStepPx = static_cast<std::uint16_t>(std::max(1, step));
    key.workClass = workClass;
    key.largeFrame = is_large_frame_extent(width, height);
    return key;
}

inline std::size_t effective_temp_scratch_bytes_cap(std::size_t requestBytes) noexcept {
    return std::max<std::size_t>(kMaxTempScratchBytes, requestBytes);
}

inline int scratch_wait_budget_ms(const ScratchBucketKey& key, std::size_t requestBytes) noexcept {
    if (requestBytes == 0) {
        return 0;
    }
    if (key.largeFrame || requestBytes >= kScratchWaitLargeRequestBytes) {
        return kScratchWaitMaxLargeMs;
    }
    if (requestBytes >= kScratchWaitMediumRequestBytes) {
        return kScratchWaitMaxMediumMs;
    }
    return kScratchWaitMaxMs;
}

inline std::size_t plane_bytes_for_extent(int width, int height) noexcept {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
    if (h > (std::numeric_limits<std::size_t>::max() / w)) {
        return std::numeric_limits<std::size_t>::max();
    }
    const std::size_t n = w * h;
    if (n > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
        return std::numeric_limits<std::size_t>::max();
    }
    return n * sizeof(float);
}

inline bool add_bytes_checked(std::size_t value, std::size_t add, std::size_t& out) noexcept {
    if (add == 0) {
        out = value;
        return true;
    }
    if (value > (std::numeric_limits<std::size_t>::max() - add)) {
        return false;
    }
    out = value + add;
    return true;
}

inline bool add_u64_checked(std::uint64_t value, std::uint64_t add, std::uint64_t& out) noexcept {
    if (add == 0) {
        out = value;
        return true;
    }
    if (value > (std::numeric_limits<std::uint64_t>::max() - add)) {
        return false;
    }
    out = value + add;
    return true;
}

inline bool mul_u64_checked(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > (std::numeric_limits<std::uint64_t>::max() / b)) {
        return false;
    }
    out = a * b;
    return true;
}

inline std::size_t saturating_u64_to_size_t(std::uint64_t value) noexcept {
    const std::uint64_t maxSizeT = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (value >= maxSizeT) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(value);
}

inline std::uint64_t non_negative_u64(int value) noexcept {
    return value > 0 ? static_cast<std::uint64_t>(value) : 0ull;
}

void add_snapshot_bytes(ManagerMemorySnapshot& snapshot, std::uint64_t bytes) noexcept {
    std::uint64_t next = 0;
    if (!add_u64_checked(snapshot.activeBytes, bytes, next)) {
        snapshot.activeBytes = std::numeric_limits<std::uint64_t>::max();
        snapshot.overflow = true;
        return;
    }
    snapshot.activeBytes = next;
}

std::uint64_t bytes_for_count_u64(std::uint64_t count, std::size_t elementBytes, bool& overflow) noexcept {
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(count, static_cast<std::uint64_t>(elementBytes), bytes)) {
        overflow = true;
        return std::numeric_limits<std::uint64_t>::max();
    }
    return bytes;
}

std::uint64_t bytes_for_plane_extent_u64(int width, int height, bool& overflow) noexcept {
    const std::uint64_t w = non_negative_u64(width);
    const std::uint64_t h = non_negative_u64(height);
    std::uint64_t count = 0;
    if (!mul_u64_checked(w, h, count)) {
        overflow = true;
        return std::numeric_limits<std::uint64_t>::max();
    }
    return bytes_for_count_u64(count, sizeof(float), overflow);
}

void add_curve_bytes(const JuicerCuda::DeviceCurve& curve, ManagerMemorySnapshot& snapshot) noexcept {
    const std::uint64_t n = non_negative_u64(curve.n);
    if (n == 0) {
        return;
    }
    bool overflow = false;
    const std::uint64_t bytes = bytes_for_count_u64(n, sizeof(float), overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (curve.x) {
        add_snapshot_bytes(snapshot, bytes);
    }
    if (curve.y) {
        add_snapshot_bytes(snapshot, bytes);
    }
}

void add_spectral_tables_bytes(
    const JuicerCuda::Resources::DeviceSpectralTables& tables,
    ManagerMemorySnapshot& snapshot) noexcept {
    const std::uint64_t k = non_negative_u64(tables.K);
    if (k == 0) {
        return;
    }
    bool overflow = false;
    const std::uint64_t bytes = bytes_for_count_u64(k, sizeof(float), overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (tables.epsC) add_snapshot_bytes(snapshot, bytes);
    if (tables.epsM) add_snapshot_bytes(snapshot, bytes);
    if (tables.epsY) add_snapshot_bytes(snapshot, bytes);
    if (tables.Ax) add_snapshot_bytes(snapshot, bytes);
    if (tables.Ay) add_snapshot_bytes(snapshot, bytes);
    if (tables.Az) add_snapshot_bytes(snapshot, bytes);
    if (tables.baseMin) add_snapshot_bytes(snapshot, bytes);
}

void add_scan_medium_bytes(
    const JuicerCuda::Resources::DeviceScanMedium& medium,
    ManagerMemorySnapshot& snapshot) noexcept {
    add_spectral_tables_bytes(medium.tables, snapshot);
}

void add_scan_lut_bytes(
    const JuicerCuda::Resources::DeviceSpectralLut& lut,
    ManagerMemorySnapshot& snapshot) noexcept {
    if (!lut.log2XYZ || lut.res == 0u) {
        return;
    }
    const std::uint64_t res = static_cast<std::uint64_t>(lut.res);
    std::uint64_t count = 0;
    if (!mul_u64_checked(res, res, count) || !mul_u64_checked(count, res, count) || !mul_u64_checked(count, 3ull, count)) {
        snapshot.overflow = true;
        add_snapshot_bytes(snapshot, std::numeric_limits<std::uint64_t>::max());
        return;
    }
    bool overflow = false;
    const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(double), overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    add_snapshot_bytes(snapshot, bytes);
}

void add_kernel_bytes(
    const JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    ManagerMemorySnapshot& snapshot) noexcept {
    if (!kernel.weights || kernel.capacity <= 0) {
        return;
    }
    const std::uint64_t count = non_negative_u64(kernel.capacity);
    bool overflow = false;
    const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(float), overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    add_snapshot_bytes(snapshot, bytes);
}

void add_optics_scratch_bytes(
    const JuicerCuda::Resources::DeviceOpticsScratch& scratch,
    ManagerMemorySnapshot& snapshot) noexcept {
    bool overflow = false;
    const std::uint64_t planeBytes = bytes_for_plane_extent_u64(scratch.width, scratch.height, overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (scratch.rgbR) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.rgbG) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.rgbB) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.blurred) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.aux) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.grainTmp) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.grainTmpShared) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.grainTmpMid) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.grainTmpCoarse) add_snapshot_bytes(snapshot, planeBytes);

    overflow = false;
    const std::uint64_t gateBytes = bytes_for_plane_extent_u64(scratch.gateWidth, scratch.gateHeight, overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (scratch.gateMask) {
        add_snapshot_bytes(snapshot, gateBytes);
    }
}

void add_spatial_dir_scratch_bytes(
    const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch,
    ManagerMemorySnapshot& snapshot) noexcept {
    bool overflow = false;
    const std::uint64_t planeBytes = bytes_for_plane_extent_u64(scratch.width, scratch.height, overflow);
    if (overflow) {
        snapshot.overflow = true;
    }
    if (scratch.corrY) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.corrM) add_snapshot_bytes(snapshot, planeBytes);
    if (scratch.corrC) add_snapshot_bytes(snapshot, planeBytes);
}

void add_resources_active_bytes_locked(
    const JuicerCuda::Resources& resources,
    ManagerMemorySnapshot& snapshot) noexcept {
    add_curve_bytes(resources.densB, snapshot);
    add_curve_bytes(resources.densG, snapshot);
    add_curve_bytes(resources.densR, snapshot);
    add_curve_bytes(resources.dirDensB, snapshot);
    add_curve_bytes(resources.dirDensG, snapshot);
    add_curve_bytes(resources.dirDensR, snapshot);
    add_curve_bytes(resources.sensB, snapshot);
    add_curve_bytes(resources.sensG, snapshot);
    add_curve_bytes(resources.sensR, snapshot);

    const std::uint64_t layerN = non_negative_u64(resources.densityCurvesLayersN);
    if (layerN > 0) {
        bool overflow = false;
        const std::uint64_t layerBytes = bytes_for_count_u64(layerN, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    add_snapshot_bytes(snapshot, layerBytes);
                }
            }
        }
    }

    {
        const std::uint64_t k = non_negative_u64(resources.tablesK);
        if (k > 0) {
            bool overflow = false;
            const std::uint64_t bytes = bytes_for_count_u64(k, sizeof(float), overflow);
            if (overflow) {
                snapshot.overflow = true;
            }
            if (resources.tablesAx) add_snapshot_bytes(snapshot, bytes);
            if (resources.tablesAy) add_snapshot_bytes(snapshot, bytes);
            if (resources.tablesAz) add_snapshot_bytes(snapshot, bytes);
            if (resources.tablesIllum) add_snapshot_bytes(snapshot, bytes);
        }
    }

    if (resources.mallettBasis && resources.mallettBasisK > 0) {
        std::uint64_t count = non_negative_u64(resources.mallettBasisK);
        if (!mul_u64_checked(count, 3ull, count)) {
            snapshot.overflow = true;
            count = std::numeric_limits<std::uint64_t>::max();
        }
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }

    add_scan_medium_bytes(resources.scanNegative, snapshot);
    add_scan_medium_bytes(resources.scanPrint, snapshot);
    add_scan_lut_bytes(resources.scanNegativeLut, snapshot);
    add_scan_lut_bytes(resources.scanPrintLut, snapshot);

    {
        bool overflow = false;
        const std::uint64_t sharedTmpBytes = bytes_for_plane_extent_u64(resources.sharedTmpWidth, resources.sharedTmpHeight, overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        if (resources.sharedTmpPlane) {
            add_snapshot_bytes(snapshot, sharedTmpBytes);
        }
    }

    add_optics_scratch_bytes(resources.scannerScratch, snapshot);
    add_spatial_dir_scratch_bytes(resources.spatialDirScratch, snapshot);

    add_kernel_bytes(resources.scannerLensBlurKernel, snapshot);
    add_kernel_bytes(resources.scannerUnsharpKernel, snapshot);
    add_kernel_bytes(resources.scannerGlareKernel, snapshot);
    add_kernel_bytes(resources.grainBlurKernel, snapshot);
    add_kernel_bytes(resources.grainBlurKernelMid, snapshot);
    add_kernel_bytes(resources.grainBlurKernelCoarse, snapshot);
    add_kernel_bytes(resources.spatialDirKernel, snapshot);
    for (int layer = 0; layer < 3; ++layer) {
        for (int ch = 0; ch < 3; ++ch) {
            add_kernel_bytes(resources.grainDyeKernel[layer][ch], snapshot);
        }
    }
    for (int ch = 0; ch < 3; ++ch) {
        add_kernel_bytes(resources.halationKernel[ch], snapshot);
        add_kernel_bytes(resources.halationScatterKernel[ch], snapshot);
    }

    if (resources.stbnData) {
        std::uint64_t count = non_negative_u64(resources.stbnWidth);
        std::uint64_t tmp = 0;
        if (!mul_u64_checked(count, non_negative_u64(resources.stbnHeight), tmp) ||
            !mul_u64_checked(tmp, non_negative_u64(resources.stbnFrames), tmp)) {
            snapshot.overflow = true;
            tmp = std::numeric_limits<std::uint64_t>::max();
        }
        add_snapshot_bytes(snapshot, tmp);
    }
    if (resources.wangTilesData) {
        std::uint64_t count = non_negative_u64(resources.wangWidth);
        std::uint64_t tmp = 0;
        if (!mul_u64_checked(count, non_negative_u64(resources.wangHeight), tmp) ||
            !mul_u64_checked(tmp, non_negative_u64(resources.wangCount), tmp)) {
            snapshot.overflow = true;
            tmp = std::numeric_limits<std::uint64_t>::max();
        }
        add_snapshot_bytes(snapshot, tmp);
    }
    if (resources.wangLutData) {
        std::uint64_t count = non_negative_u64(resources.wangColors);
        std::uint64_t tmp = count;
        if (!mul_u64_checked(tmp, count, tmp) ||
            !mul_u64_checked(tmp, count, tmp) ||
            !mul_u64_checked(tmp, count, tmp)) {
            snapshot.overflow = true;
            tmp = std::numeric_limits<std::uint64_t>::max();
        }
        add_snapshot_bytes(snapshot, tmp);
    }

    add_curve_bytes(resources.printDcC, snapshot);
    add_curve_bytes(resources.printDcM, snapshot);
    add_curve_bytes(resources.printDcY, snapshot);
    add_curve_bytes(resources.printSensC, snapshot);
    add_curve_bytes(resources.printSensM, snapshot);
    add_curve_bytes(resources.printSensY, snapshot);

    if (resources.printIllumFiltered && resources.printIllumK > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(non_negative_u64(resources.printIllumK), sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }

    if (resources.hanatosLut && resources.hanatosN > 0) {
        std::uint64_t n = non_negative_u64(resources.hanatosN);
        std::uint64_t count = 0;
        if (!mul_u64_checked(n, n, count) ||
            !mul_u64_checked(count, static_cast<std::uint64_t>(81u), count)) {
            snapshot.overflow = true;
            count = std::numeric_limits<std::uint64_t>::max();
        }
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
    if (resources.hanatosLutIntegrated && resources.hanatosNIntegrated > 0) {
        std::uint64_t n = non_negative_u64(resources.hanatosNIntegrated);
        std::uint64_t count = 0;
        if (!mul_u64_checked(n, n, count) || !mul_u64_checked(count, 4ull, count)) {
            snapshot.overflow = true;
            count = std::numeric_limits<std::uint64_t>::max();
        }
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(count, sizeof(float), overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }

    if (resources.scanErrorFlag) {
        add_snapshot_bytes(snapshot, sizeof(int));
    }
    if (resources.scanErrorHost) {
        add_snapshot_bytes(snapshot, sizeof(int));
    }

    if (resources.autoExposureExposureScale) add_snapshot_bytes(snapshot, sizeof(float));
    if (resources.autoExposureAutoEV) add_snapshot_bytes(snapshot, sizeof(double));
    if (resources.autoExposureValid) add_snapshot_bytes(snapshot, sizeof(int));

    if (resources.autoExposureScratch.maxYBits) {
        add_snapshot_bytes(snapshot, sizeof(unsigned int));
    }
    if (resources.autoExposureScratch.histogram) {
        add_snapshot_bytes(snapshot, static_cast<std::uint64_t>(2048u) * sizeof(unsigned int));
    }
    if (resources.autoExposureScratch.weightsX && resources.autoExposureScratch.weightsXCapacity > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.weightsXCapacity),
            sizeof(float),
            overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
    if (resources.autoExposureScratch.weightsY && resources.autoExposureScratch.weightsYCapacity > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.weightsYCapacity),
            sizeof(float),
            overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
    if (resources.autoExposureScratch.partialsA && resources.autoExposureScratch.partialCapacity > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.partialCapacity),
            sizeof(JuicerCudaAutoExposurePartial),
            overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
    if (resources.autoExposureScratch.partialsB && resources.autoExposureScratch.partialCapacity > 0) {
        bool overflow = false;
        const std::uint64_t bytes = bytes_for_count_u64(
            non_negative_u64(resources.autoExposureScratch.partialCapacity),
            sizeof(JuicerCudaAutoExposurePartial),
            overflow);
        if (overflow) {
            snapshot.overflow = true;
        }
        add_snapshot_bytes(snapshot, bytes);
    }
}

ManagerMemorySnapshot snapshot_manager_memory(JuicerCuda::Resources& resources) noexcept {
    ManagerMemorySnapshot snapshot{};
    {
        std::lock_guard<std::mutex> lock(resources.m);
        add_resources_active_bytes_locked(resources, snapshot);
        snapshot.retirePendingBytes = static_cast<std::uint64_t>(
            std::min<std::size_t>(
                resources.retireBytes,
                static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
    }
    snapshot.reclaimableBytes = snapshot.retirePendingBytes;
    snapshot.transientNonManagerBytes =
        global_state().transientNonManagerBytes.load(std::memory_order_relaxed);
    return snapshot;
}

void publish_manager_memory_snapshot(const ManagerMemorySnapshot& snapshot) noexcept {
    ResourceManagerState& state = global_state();
    state.managerActiveBytes.store(snapshot.activeBytes, std::memory_order_relaxed);
    state.managerReclaimableBytes.store(snapshot.reclaimableBytes, std::memory_order_relaxed);
    state.managerRetirePendingBytes.store(snapshot.retirePendingBytes, std::memory_order_relaxed);
    state.transientNonManagerBytes.store(snapshot.transientNonManagerBytes, std::memory_order_relaxed);
}

inline bool pressure_policy_enabled(const ResourceManagerConfigEffective& cfg) noexcept {
    return cfg.managerSoftTargetBytes > 0;
}

inline bool should_collect_manager_memory_snapshots(const ResourceManagerConfigEffective& cfg) noexcept {
    return pressure_policy_enabled(cfg) || JTRACE_ENABLED(3);
}

std::uint64_t transient_reservation_cap_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    if (cfg.managerSoftTargetBytes == 0) {
        return kTransientReservationCapDefaultBytes;
    }
    const std::uint64_t quarterTarget = cfg.managerSoftTargetBytes / 4ull;
    return std::max<std::uint64_t>(
        kTransientReservationThresholdDefaultBytes,
        std::min<std::uint64_t>(kTransientReservationCapDefaultBytes, quarterTarget));
}

std::uint64_t transient_reservation_threshold_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    const std::uint64_t capBytes = transient_reservation_cap_bytes(cfg);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kTransientReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t builder_reservation_cap_bytes(
    const ResourceManagerConfigEffective& cfg,
    BuilderReservationTier tier) noexcept {
    std::uint32_t configCapMb = 0;
    std::uint64_t defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
    switch (tier) {
    case BuilderReservationTier::Scratch:
        configCapMb = cfg.scratchBuilderBytesInFlightLimitMB;
        defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
        break;
    case BuilderReservationTier::Lut:
        configCapMb = cfg.lutBuilderBytesInFlightLimitMB;
        defaultCapBytes = kLutBuilderReservationCapDefaultBytes;
        break;
    case BuilderReservationTier::Graph:
        configCapMb = cfg.graphBuilderBytesInFlightLimitMB;
        defaultCapBytes = kGraphBuilderReservationCapDefaultBytes;
        break;
    default:
        configCapMb = cfg.scratchBuilderBytesInFlightLimitMB;
        defaultCapBytes = kScratchBuilderReservationCapDefaultBytes;
        break;
    }

    std::uint64_t capBytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(configCapMb), kBytesPerMiB, capBytes)) {
        capBytes = std::numeric_limits<std::uint64_t>::max();
    }
    if (capBytes == 0) {
        capBytes = defaultCapBytes;
    }
    return std::max<std::uint64_t>(kBuilderReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t builder_reservation_threshold_bytes(
    const ResourceManagerConfigEffective& cfg,
    BuilderReservationTier tier) noexcept {
    const std::uint64_t capBytes = builder_reservation_cap_bytes(cfg, tier);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kBuilderReservationThresholdDefaultBytes, capBytes);
}

std::uint64_t& builder_context_bytes_for_tier(
    BuilderReservationContextState& contextState,
    BuilderReservationTier tier) noexcept {
    switch (tier) {
    case BuilderReservationTier::Scratch:
        return contextState.inFlightScratchBytes;
    case BuilderReservationTier::Lut:
        return contextState.inFlightLutBytes;
    case BuilderReservationTier::Graph:
        return contextState.inFlightGraphBytes;
    default:
        return contextState.inFlightScratchBytes;
    }
}

std::uint64_t& builder_total_bytes_for_tier(
    BuilderReservationState& state,
    BuilderReservationTier tier) noexcept {
    switch (tier) {
    case BuilderReservationTier::Scratch:
        return state.totalScratchInFlightBytes;
    case BuilderReservationTier::Lut:
        return state.totalLutInFlightBytes;
    case BuilderReservationTier::Graph:
        return state.totalGraphInFlightBytes;
    default:
        return state.totalScratchInFlightBytes;
    }
}

std::atomic<std::uint64_t>& builder_global_gauge_for_tier(
    ResourceManagerState& state,
    BuilderReservationTier tier) noexcept {
    switch (tier) {
    case BuilderReservationTier::Scratch:
        return state.scratchBuilderBytesInFlight;
    case BuilderReservationTier::Lut:
        return state.lutBuilderBytesInFlight;
    case BuilderReservationTier::Graph:
        return state.graphBuilderBytesInFlight;
    default:
        return state.scratchBuilderBytesInFlight;
    }
}

bool builder_context_has_inflight(const BuilderReservationContextState& contextState) noexcept {
    return contextState.inFlightScratchBytes > 0 ||
        contextState.inFlightLutBytes > 0 ||
        contextState.inFlightGraphBytes > 0;
}

std::uint64_t upload_reservation_cap_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    std::uint64_t configCapBytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(cfg.uploadBytesInFlightLimitMB),
            kBytesPerMiB,
            configCapBytes)) {
        configCapBytes = std::numeric_limits<std::uint64_t>::max();
    }
    if (configCapBytes == 0) {
        configCapBytes = kUploadReservationCapDefaultBytes;
    }
    if (cfg.managerSoftTargetBytes == 0) {
        return std::max<std::uint64_t>(kUploadReservationThresholdDefaultBytes, configCapBytes);
    }
    const std::uint64_t quarterTarget = cfg.managerSoftTargetBytes / 4ull;
    const std::uint64_t boundedCap = std::min<std::uint64_t>(configCapBytes, quarterTarget);
    return std::max<std::uint64_t>(kUploadReservationThresholdDefaultBytes, boundedCap);
}

std::uint64_t upload_reservation_threshold_bytes(const ResourceManagerConfigEffective& cfg) noexcept {
    const std::uint64_t capBytes = upload_reservation_cap_bytes(cfg);
    if (capBytes == 0) {
        return 0;
    }
    return std::min<std::uint64_t>(kUploadReservationThresholdDefaultBytes, capBytes);
}

void add_estimate_bytes_u64(std::uint64_t bytes, std::uint64_t& total, bool& overflow) noexcept {
    if (overflow) {
        return;
    }
    std::uint64_t next = 0;
    if (!add_u64_checked(total, bytes, next)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    total = next;
}

template <typename T>
void add_vector_upload_estimate_bytes(
    const std::vector<T>& values,
    std::uint64_t& total,
    bool& overflow) noexcept {
    if (values.empty() || overflow) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(values.size()),
            static_cast<std::uint64_t>(sizeof(T)),
            bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(bytes, total, overflow);
}

void add_curve_upload_estimate_bytes(
    const Spectral::Curve& curve,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const std::size_t n = std::min(curve.lambda_nm.size(), curve.linear.size());
    if (n == 0 || overflow) {
        return;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(2u * sizeof(float)), bytes)) {
        total = std::numeric_limits<std::uint64_t>::max();
        overflow = true;
        return;
    }
    add_estimate_bytes_u64(bytes, total, overflow);
}

void add_scan_medium_upload_estimate_bytes(
    const Scanner::ScannerMediumRuntime& medium,
    std::uint64_t& total,
    bool& overflow) noexcept {
    const Spectral::SpectralTables* t = medium.tables;
    if (!t) {
        return;
    }
    add_vector_upload_estimate_bytes(t->epsC, total, overflow);
    add_vector_upload_estimate_bytes(t->epsM, total, overflow);
    add_vector_upload_estimate_bytes(t->epsY, total, overflow);
    add_vector_upload_estimate_bytes(t->Ax, total, overflow);
    add_vector_upload_estimate_bytes(t->Ay, total, overflow);
    add_vector_upload_estimate_bytes(t->Az, total, overflow);
    if (t->hasBaseline) {
        add_vector_upload_estimate_bytes(t->baseMin, total, overflow);
    }
}

std::uint64_t estimate_upload_core_request_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws) noexcept {
    const std::uint64_t wsCoreHash =
        (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
    const std::uint64_t wsDirHash = ws.dirHash;
    if (wsCoreHash == 0 || wsDirHash == 0) {
        return kUploadReservationThresholdDefaultBytes;
    }

    bool coreUpToDate = false;
    bool dirUpToDate = false;
    bool needStbnUpload = false;
    bool needWangUpload = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        coreUpToDate = (resources.uploadedCoreHash != 0) && (resources.uploadedCoreHash == wsCoreHash);
        dirUpToDate = (resources.uploadedDirHash != 0) && (resources.uploadedDirHash == wsDirHash);
        needStbnUpload = (resources.stbnData == nullptr);
        needWangUpload = (resources.wangTilesData == nullptr) || (resources.wangLutData == nullptr);
    }

    std::uint64_t estimateBytes = 0;
    bool overflow = false;
    if (needStbnUpload) {
        add_estimate_bytes_u64(kStbnUploadDefaultBytes, estimateBytes, overflow);
    }
    if (needWangUpload) {
        add_estimate_bytes_u64(kWangTilesUploadDefaultBytes, estimateBytes, overflow);
        add_estimate_bytes_u64(kWangLutUploadDefaultBytes, estimateBytes, overflow);
    }

    if (coreUpToDate && dirUpToDate) {
        return estimateBytes;
    }

    if (coreUpToDate && !dirUpToDate) {
        add_curve_upload_estimate_bytes(ws.dirDensB, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensG, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(ws.dirDensR, estimateBytes, overflow);
        return estimateBytes;
    }

    add_curve_upload_estimate_bytes(ws.densB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.densG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.densR, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.dirDensR, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensB, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensG, estimateBytes, overflow);
    add_curve_upload_estimate_bytes(ws.sensR, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Ax, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Ay, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.Az, estimateBytes, overflow);
    add_vector_upload_estimate_bytes(ws.tablesRef.illum, estimateBytes, overflow);

    for (int layer = 0; layer < 3; ++layer) {
        for (int ch = 0; ch < 3; ++ch) {
            add_vector_upload_estimate_bytes(ws.densityCurvesLayers[layer][ch], estimateBytes, overflow);
        }
    }

    add_scan_medium_upload_estimate_bytes(ws.negativeMediumRuntime, estimateBytes, overflow);
    add_scan_medium_upload_estimate_bytes(ws.printMediumRuntime, estimateBytes, overflow);

    if (ws.printRT && Print::profile_is_valid(ws.printRT->profile)) {
        const Print::Profile& p = ws.printRT->profile;
        add_curve_upload_estimate_bytes(p.dcC, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(p.dcM, estimateBytes, overflow);
        add_curve_upload_estimate_bytes(p.dcY, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensC_log.linear, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensM_log.linear, estimateBytes, overflow);
        add_vector_upload_estimate_bytes(p.sensY_log.linear, estimateBytes, overflow);
    }

    // Keep a deterministic floor so large rebuilds always enter upload reservation admission.
    if (estimateBytes < kUploadReservationThresholdDefaultBytes) {
        estimateBytes = kUploadReservationThresholdDefaultBytes;
    }
    return estimateBytes;
}

void refill_upload_fairness_tokens(
    UploadReservationContextState::FairnessEntry& entry,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t nowMs) noexcept {
    const std::uint32_t sharedPerTick = std::max<std::uint32_t>(1u, cfg.uploadFairnessTokensPerTick);
    const std::uint32_t criticalPerTick = std::min<std::uint32_t>(
        std::max<std::uint32_t>(1u, cfg.criticalUploadReservedTokens),
        sharedPerTick);

    if (entry.lastRefillMs == 0) {
        entry.sharedTokens = sharedPerTick;
        entry.criticalTokens = criticalPerTick;
        entry.lastRefillMs = nowMs;
        return;
    }

    const std::uint64_t elapsedMs = (nowMs > entry.lastRefillMs) ? (nowMs - entry.lastRefillMs) : 0;
    if (elapsedMs < kUploadFairnessTickMs) {
        return;
    }

    entry.sharedTokens = sharedPerTick;
    entry.criticalTokens = criticalPerTick;
    entry.lastRefillMs = nowMs;
}

void refill_builder_fairness_tokens(
    BuilderReservationContextState::FairnessEntry& entry,
    const ResourceManagerConfigEffective& cfg,
    std::uint64_t nowMs) noexcept {
    const std::uint32_t sharedPerTick = std::max<std::uint32_t>(1u, cfg.builderFairnessTokensPerTick);
    const std::uint32_t criticalPerTick = std::min<std::uint32_t>(
        std::max<std::uint32_t>(1u, cfg.criticalBuilderReservedTokens),
        sharedPerTick);

    if (entry.lastRefillMs == 0) {
        entry.sharedTokens = sharedPerTick;
        entry.criticalTokens = criticalPerTick;
        entry.lastRefillMs = nowMs;
        return;
    }

    const std::uint64_t elapsedMs = (nowMs > entry.lastRefillMs) ? (nowMs - entry.lastRefillMs) : 0;
    if (elapsedMs < kBuilderFairnessTickMs) {
        return;
    }

    entry.sharedTokens = sharedPerTick;
    entry.criticalTokens = criticalPerTick;
    entry.lastRefillMs = nowMs;
}

std::uint64_t estimate_scan_lut_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium) noexcept {
    const Scanner::ScannerStaticKey& staticKey = negativeMedium ? ws.negativeStaticKey : ws.printStaticKey;
    const std::uint32_t res =
        ResourceManager::normalize_scan_lut_resolution(staticKey.lutResolution);
    std::uint64_t voxelCount = 0;
    if (!mul_u64_checked(static_cast<std::uint64_t>(res), static_cast<std::uint64_t>(res), voxelCount)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    if (!mul_u64_checked(voxelCount, static_cast<std::uint64_t>(res), voxelCount)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t values = 0;
    if (!mul_u64_checked(voxelCount, 3ull, values)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(values, static_cast<std::uint64_t>(sizeof(double)), bytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    const Scanner::ScannerMediumRuntime& medium = negativeMedium ? ws.negativeMediumRuntime : ws.printMediumRuntime;
    if (!medium.tables || medium.tables->tablesHash == 0 || medium.range.digest == 0) {
        return 0;
    }
    const std::uint64_t expectedHash = ResourceManager::make_scan_lut_key_digest(
        static_cast<std::uint32_t>(medium.medium),
        medium.tables->tablesHash,
        medium.range.digest,
        res);
    if (expectedHash == 0) {
        return 0;
    }

    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        const JuicerCuda::Resources::DeviceSpectralLut& dst =
            negativeMedium ? resources.scanNegativeLut : resources.scanPrintLut;
        cached = dst.log2XYZ && dst.res == res && dst.hash == expectedHash;
    }
    return cached ? 0 : bytes;
}

std::uint64_t estimate_print_illuminant_upload_bytes(
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params) noexcept {
    const int k = Spectral::gShape.K;
    if (k <= 0) {
        return 0;
    }
    const std::uint64_t wsCoreHash =
        (ws.uploadCoreHash != 0) ? ws.uploadCoreHash : ws.coreHash;
    if (wsCoreHash == 0) {
        return 0;
    }
    const float yKey = std::isfinite(params.yFilter) ? params.yFilter : 0.0f;
    const float mKey = std::isfinite(params.mFilter) ? params.mFilter : 0.0f;
    const float cKey = std::isfinite(params.cFilter) ? params.cFilter : 0.0f;
    const std::uint64_t neutralFilterHash =
        (prt.neutralFilterHash != 0) ? prt.neutralFilterHash : Print::kDefaultNeutralFilterHash;

    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(resources.m);
        cached =
            resources.printIllumFiltered &&
            resources.printIllumK == k &&
            resources.printIllumShapeK == k &&
            resources.printIllumCoreHash == wsCoreHash &&
            resources.printIllumYShiftSteps == yKey &&
            resources.printIllumMShiftSteps == mKey &&
            resources.printIllumCShiftSteps == cKey &&
            resources.printIllumNeutralFilterHash == neutralFilterHash;
    }
    if (cached) {
        return 0;
    }
    std::uint64_t bytes = 0;
    if (!mul_u64_checked(
            static_cast<std::uint64_t>(k),
            static_cast<std::uint64_t>(sizeof(float)),
            bytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return bytes;
}

void maybe_publish_manager_memory_snapshot(
    JuicerCuda::Resources& resources,
    bool enabled) noexcept {
    if (!enabled) {
        return;
    }
    publish_manager_memory_snapshot(snapshot_manager_memory(resources));
}

HeadroomTelemetry sample_headroom_telemetry() noexcept {
    HeadroomTelemetry out{};
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
        (void)totalBytes;
        out.driverFreeBytes = static_cast<std::uint64_t>(
            std::min<std::size_t>(freeBytes, static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
        out.effectiveHeadroomBytes = out.driverFreeBytes;
    }
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    int currentDevice = 0;
    if (cudaGetDevice(&currentDevice) == cudaSuccess) {
        cudaMemPool_t defaultPool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&defaultPool, currentDevice) == cudaSuccess && defaultPool != nullptr) {
            std::size_t poolReservedBytes = 0;
            std::size_t poolUsedBytes = 0;
            const cudaError_t reservedErr = cudaMemPoolGetAttribute(
                defaultPool,
                cudaMemPoolAttrReservedMemCurrent,
                &poolReservedBytes);
            const cudaError_t usedErr = cudaMemPoolGetAttribute(
                defaultPool,
                cudaMemPoolAttrUsedMemCurrent,
                &poolUsedBytes);
            if (reservedErr == cudaSuccess && usedErr == cudaSuccess) {
                out.poolTelemetryAvailable = true;
                out.source = HeadroomSource::AllocatorPool;
                out.allocatorPoolReservedBytes = static_cast<std::uint64_t>(
                    std::min<std::size_t>(
                        poolReservedBytes,
                        static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
                out.allocatorPoolUsedBytes = static_cast<std::uint64_t>(
                    std::min<std::size_t>(
                        poolUsedBytes,
                        static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
                if (out.allocatorPoolUsedBytes > out.allocatorPoolReservedBytes) {
                    out.allocatorPoolUsedBytes = out.allocatorPoolReservedBytes;
                }

                const std::uint64_t poolFreeBytes =
                    out.allocatorPoolReservedBytes - out.allocatorPoolUsedBytes;
                std::uint64_t combinedHeadroom = out.driverFreeBytes;
                if (!add_u64_checked(combinedHeadroom, poolFreeBytes, combinedHeadroom)) {
                    combinedHeadroom = std::numeric_limits<std::uint64_t>::max();
                }
                out.effectiveHeadroomBytes = combinedHeadroom;
            }
        }
    }
#endif
#endif
    return out;
}

std::uint64_t pressure_total_bytes(const PressureInput& input) noexcept {
    std::uint64_t total = input.managerResidentBytes;
    std::uint64_t next = 0;
    if (!add_u64_checked(total, input.retirePendingBytes, next)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    total = next;
    if (!add_u64_checked(total, input.transientNonManagerBytes, next)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return next;
}

bool reserve_crossed(const PressureInput& input) noexcept {
    if (input.softTargetBytes == 0) {
        return false;
    }
    return pressure_total_bytes(input) > input.softTargetBytes;
}

int pressure_state_rank(PressureState state) noexcept {
    switch (state) {
    case PressureState::Normal:
        return 0;
    case PressureState::Constrained:
        return 1;
    case PressureState::Critical:
        return 2;
    case PressureState::Emergency:
        return 3;
    default:
        return 0;
    }
}

std::uint32_t pressure_poll_interval_ms_for_state(
    const ResourceManagerConfigEffective& cfg,
    PressureState state) noexcept {
    const std::uint32_t fallback = std::max<std::uint32_t>(1u, cfg.pressurePollIntervalMs);
    const std::uint32_t normal = std::max<std::uint32_t>(1u, cfg.pressurePollIntervalMsNormal);
    const std::uint32_t critical = std::max<std::uint32_t>(
        1u,
        std::min<std::uint32_t>(cfg.pressurePollIntervalMsCritical, normal));
    switch (state) {
    case PressureState::Normal:
        return normal;
    case PressureState::Constrained:
        return std::max<std::uint32_t>(critical, (normal + critical) / 2u);
    case PressureState::Critical:
    case PressureState::Emergency:
        return critical;
    default:
        return fallback;
    }
}

std::size_t estimate_optics_growth_bytes(
    JuicerCuda::Resources& resources,
    int width,
    int height,
    bool needBlurredScratch,
    bool needAuxScratch,
    bool needGrainScratch,
    bool needGrainSharedScratch,
    bool needGateMask) noexcept {
    const std::size_t planeBytes = plane_bytes_for_extent(width, height);
    if (planeBytes == 0 || planeBytes == std::numeric_limits<std::size_t>::max()) {
        return planeBytes;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    const auto& scratch = resources.scannerScratch;
    const bool dimsMatch = (scratch.width == width && scratch.height == height);
    const bool haveBase = scratch.rgbR && scratch.rgbG && scratch.rgbB;
    const bool fullRebuild = !dimsMatch || !haveBase;

    std::size_t estimate = 0;
    auto addPlane = [&](std::size_t multiplier = 1) {
        for (std::size_t i = 0; i < multiplier; ++i) {
            std::size_t next = 0;
            if (!add_bytes_checked(estimate, planeBytes, next)) {
                estimate = std::numeric_limits<std::size_t>::max();
                return;
            }
            estimate = next;
        }
    };

    if (fullRebuild) {
        addPlane(3); // rgbR/rgbG/rgbB
    }

    const bool sharedTmpMatch = resources.sharedTmpPlane &&
        resources.sharedTmpWidth == width &&
        resources.sharedTmpHeight == height;
    if (!sharedTmpMatch) {
        addPlane(); // shared tmp plane
    }

    if (needBlurredScratch && (fullRebuild || !scratch.blurred)) {
        addPlane();
    }
    if (needAuxScratch && (fullRebuild || !scratch.aux)) {
        addPlane();
    }
    if (needGrainScratch) {
        if (fullRebuild || !scratch.grainTmp) {
            addPlane();
        }
        if (fullRebuild || !scratch.grainTmpMid) {
            addPlane();
        }
        if (fullRebuild || !scratch.grainTmpCoarse) {
            addPlane();
        }
    }
    if (needGrainSharedScratch && (fullRebuild || !scratch.grainTmpShared)) {
        addPlane();
    }
    if (needGateMask) {
        const bool gateDimsMatch = (scratch.gateWidth == width && scratch.gateHeight == height);
        if (fullRebuild || !scratch.gateMask || !gateDimsMatch) {
            addPlane();
        }
    }

    return estimate;
}

std::size_t estimate_spatial_dir_growth_bytes(
    JuicerCuda::Resources& resources,
    int width,
    int height) noexcept {
    const std::size_t planeBytes = plane_bytes_for_extent(width, height);
    if (planeBytes == 0 || planeBytes == std::numeric_limits<std::size_t>::max()) {
        return planeBytes;
    }

    std::lock_guard<std::mutex> lock(resources.m);
    const auto& scratch = resources.spatialDirScratch;
    const bool haveBase = (scratch.width == width && scratch.height == height && scratch.corrY && scratch.corrM && scratch.corrC);
    const bool sharedTmpMatch = resources.sharedTmpPlane &&
        resources.sharedTmpWidth == width &&
        resources.sharedTmpHeight == height;

    std::size_t estimate = 0;
    if (!haveBase) {
        std::size_t next = 0;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
    }
    if (!sharedTmpMatch) {
        std::size_t next = 0;
        if (!add_bytes_checked(estimate, planeBytes, next)) {
            return std::numeric_limits<std::size_t>::max();
        }
        estimate = next;
    }
    return estimate;
}

void trim_large_frame_quarantine_decay_locked(
    ScratchContextState& contextState,
    std::uint64_t nowMs) noexcept {
    std::size_t index = 0;
    std::uint64_t removedCount = 0;
    while (index < contextState.largeFrameQuarantine.size()) {
        const ScratchQuarantineEntry& entry = contextState.largeFrameQuarantine[index];
        const std::uint64_t ageMs = (nowMs > entry.touchedMs) ? (nowMs - entry.touchedMs) : 0;
        if (ageMs < kLargeFrameQuarantineDecayMs) {
            ++index;
            continue;
        }
        if (contextState.largeFrameQuarantineBytes >= entry.bytes) {
            contextState.largeFrameQuarantineBytes -= entry.bytes;
        }
        else {
            contextState.largeFrameQuarantineBytes = 0;
        }
        contextState.largeFrameQuarantine.erase(
            contextState.largeFrameQuarantine.begin() + static_cast<std::ptrdiff_t>(index));
        ++removedCount;
    }
    if (removedCount > 0) {
        global_state().scratchLargeQuarantineDecayEvents.fetch_add(removedCount, std::memory_order_relaxed);
    }
}

void trim_large_frame_quarantine_caps_locked(
    ScratchContextState& contextState) noexcept {
    std::uint64_t trimmedCount = 0;
    while ((contextState.largeFrameQuarantineBytes > kLargeFrameQuarantineMaxBytes) ||
           (contextState.largeFrameQuarantine.size() > kLargeFrameQuarantineMaxEntries)) {
        if (contextState.largeFrameQuarantine.empty()) {
            contextState.largeFrameQuarantineBytes = 0;
            break;
        }

        std::size_t victimIndex = 0;
        ScratchQuarantineEntry victim = contextState.largeFrameQuarantine[0];
        for (std::size_t i = 1; i < contextState.largeFrameQuarantine.size(); ++i) {
            const ScratchQuarantineEntry& candidate = contextState.largeFrameQuarantine[i];
            const bool older = (candidate.touchedMs < victim.touchedMs) ||
                ((candidate.touchedMs == victim.touchedMs) && (candidate.sequence < victim.sequence));
            if (older) {
                victim = candidate;
                victimIndex = i;
            }
        }

        if (contextState.largeFrameQuarantineBytes >= victim.bytes) {
            contextState.largeFrameQuarantineBytes -= victim.bytes;
        }
        else {
            contextState.largeFrameQuarantineBytes = 0;
        }
        contextState.largeFrameQuarantine.erase(
            contextState.largeFrameQuarantine.begin() + static_cast<std::ptrdiff_t>(victimIndex));
        ++trimmedCount;
    }

    if (trimmedCount > 0) {
        global_state().scratchLargeQuarantineTrimEvents.fetch_add(trimmedCount, std::memory_order_relaxed);
    }
}

std::uint64_t trim_large_frame_quarantine_for_context(
    const DeviceContextKey& contextKey) noexcept {
    ScratchPolicyState& state = scratch_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(contextKey);
    if (contextIt == state.byContext.end()) {
        return 0;
    }
    ScratchContextState& contextState = contextIt->second;
    const std::uint64_t removedEntries = static_cast<std::uint64_t>(contextState.largeFrameQuarantine.size());
    if (removedEntries == 0) {
        return 0;
    }
    contextState.largeFrameQuarantine.clear();
    contextState.largeFrameQuarantineBytes = 0;
    return removedEntries;
}

void snapshot_bucket_state_locked(
    const ScratchContextState& contextState,
    const ScratchBucketEntry& bucketEntry,
    const ScratchBucketKey& bucketKey,
    ScratchPolicySnapshot& outSnapshot) noexcept {
    outSnapshot.inFlightBytes = bucketEntry.inFlightBytes;
    outSnapshot.inFlightSets = bucketEntry.inFlightSets;
    outSnapshot.quarantineBytes = contextState.largeFrameQuarantineBytes;
    outSnapshot.quarantineEntries = static_cast<std::uint32_t>(contextState.largeFrameQuarantine.size());
    outSnapshot.bucketAttempts = bucketEntry.attemptCount;
    outSnapshot.bucketExhausted = bucketEntry.exhaustedCount;
    outSnapshot.bucketAllocGrowth = bucketEntry.allocGrowthEvents;
    outSnapshot.bucketReuse = bucketEntry.reuseEvents;
    outSnapshot.starvationLatched = bucketEntry.starvationLatched;
    outSnapshot.bucketKey = bucketKey;
}

void release_scratch_policy_claim(ScratchPolicyClaim& claim) noexcept {
    if (!claim.acquired) {
        return;
    }

    ScratchPolicyState& state = scratch_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(claim.contextKey);
    if (contextIt == state.byContext.end()) {
        claim = ScratchPolicyClaim{};
        return;
    }

    ScratchContextState& contextState = contextIt->second;
    auto bucketIt = contextState.buckets.find(claim.bucketKey);
    std::size_t releasedBytes = 0;
    if (bucketIt != contextState.buckets.end()) {
        ScratchBucketEntry& bucketEntry = bucketIt->second;
        if (bucketEntry.inFlightSets > 0) {
            --bucketEntry.inFlightSets;
        }
        if (bucketEntry.inFlightBytes >= claim.bytes) {
            bucketEntry.inFlightBytes -= claim.bytes;
            releasedBytes = claim.bytes;
        }
        else {
            releasedBytes = bucketEntry.inFlightBytes;
            bucketEntry.inFlightBytes = 0;
        }
    }
    if (contextState.totalInFlightBytes >= releasedBytes) {
        contextState.totalInFlightBytes -= releasedBytes;
    }
    else {
        contextState.totalInFlightBytes = 0;
    }
    if (state.totalInFlightBytes >= static_cast<std::uint64_t>(releasedBytes)) {
        state.totalInFlightBytes -= static_cast<std::uint64_t>(releasedBytes);
    }
    else {
        state.totalInFlightBytes = 0;
    }
    global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);

    const std::uint64_t nowMs = monotonic_time_ms();
    trim_large_frame_quarantine_decay_locked(contextState, nowMs);

    if (claim.bucketKey.largeFrame && claim.bytes > 0) {
        ScratchQuarantineEntry entry{};
        entry.key = claim.bucketKey;
        entry.bytes = claim.bytes;
        entry.touchedMs = nowMs;
        entry.sequence = contextState.nextQuarantineSequence++;
        contextState.largeFrameQuarantine.push_back(entry);
        contextState.largeFrameQuarantineBytes += claim.bytes;
        trim_large_frame_quarantine_caps_locked(contextState);
    }

    claim = ScratchPolicyClaim{};
}

class ScratchPolicyGuard {
public:
    explicit ScratchPolicyGuard(ScratchPolicyClaim&& claim) noexcept
        : _claim(std::move(claim)) {
    }

    ~ScratchPolicyGuard() noexcept {
        release_scratch_policy_claim(_claim);
    }

    ScratchPolicyGuard(const ScratchPolicyGuard&) = delete;
    ScratchPolicyGuard& operator=(const ScratchPolicyGuard&) = delete;

private:
    ScratchPolicyClaim _claim{};
};

void release_builder_reservation_claim(BuilderReservationClaim& claim) noexcept {
    if (!claim.acquired) {
        return;
    }

    BuilderReservationState& state = builder_reservation_state();
    ResourceManagerState& managerState = global_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    std::uint64_t& tierTotalBytes = builder_total_bytes_for_tier(state, claim.tier);
    auto contextIt = state.byContext.find(claim.contextKey);
    if (contextIt == state.byContext.end()) {
        if (tierTotalBytes >= claim.bytes) {
            tierTotalBytes -= claim.bytes;
        }
        else {
            tierTotalBytes = 0;
        }
        builder_global_gauge_for_tier(managerState, claim.tier).store(tierTotalBytes, std::memory_order_relaxed);
        claim = BuilderReservationClaim{};
        return;
    }

    BuilderReservationContextState& contextState = contextIt->second;
    std::uint64_t& contextTierBytes = builder_context_bytes_for_tier(contextState, claim.tier);
    if (contextTierBytes >= claim.bytes) {
        contextTierBytes -= claim.bytes;
    }
    else {
        contextTierBytes = 0;
    }
    if (tierTotalBytes >= claim.bytes) {
        tierTotalBytes -= claim.bytes;
    }
    else {
        tierTotalBytes = 0;
    }

    if (!builder_context_has_inflight(contextState)) {
        state.byContext.erase(contextIt);
    }
    builder_global_gauge_for_tier(managerState, claim.tier).store(tierTotalBytes, std::memory_order_relaxed);
    claim = BuilderReservationClaim{};
}

class BuilderReservationGuard {
public:
    explicit BuilderReservationGuard(BuilderReservationClaim&& claim) noexcept
        : _claim(std::move(claim)) {
    }

    ~BuilderReservationGuard() noexcept {
        release_builder_reservation_claim(_claim);
    }

    BuilderReservationGuard(const BuilderReservationGuard&) = delete;
    BuilderReservationGuard& operator=(const BuilderReservationGuard&) = delete;

private:
    BuilderReservationClaim _claim{};
};

void release_upload_reservation_claim(UploadReservationClaim& claim) noexcept {
    if (!claim.acquired) {
        return;
    }

    UploadReservationState& state = upload_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto contextIt = state.byContext.find(claim.contextKey);
    if (contextIt == state.byContext.end()) {
        if (state.totalInFlightBytes >= claim.bytes) {
            state.totalInFlightBytes -= claim.bytes;
        }
        else {
            state.totalInFlightBytes = 0;
        }
        claim = UploadReservationClaim{};
        global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
        return;
    }

    UploadReservationContextState& contextState = contextIt->second;
    if (contextState.inFlightBytes >= claim.bytes) {
        contextState.inFlightBytes -= claim.bytes;
    }
    else {
        contextState.inFlightBytes = 0;
    }
    if (state.totalInFlightBytes >= claim.bytes) {
        state.totalInFlightBytes -= claim.bytes;
    }
    else {
        state.totalInFlightBytes = 0;
    }
    if (contextState.inFlightBytes == 0) {
        state.byContext.erase(contextIt);
    }
    global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
    claim = UploadReservationClaim{};
}

class UploadReservationGuard {
public:
    explicit UploadReservationGuard(UploadReservationClaim&& claim) noexcept
        : _claim(std::move(claim)) {
    }

    ~UploadReservationGuard() noexcept {
        release_upload_reservation_claim(_claim);
    }

    UploadReservationGuard(const UploadReservationGuard&) = delete;
    UploadReservationGuard& operator=(const UploadReservationGuard&) = delete;

private:
    UploadReservationClaim _claim{};
};

void trace_scratch_policy_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const char* result,
    std::size_t requestBytes,
    const ScratchPolicySnapshot& snapshot,
    int waitMs) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::uint64_t churnDenom = snapshot.bucketAllocGrowth + snapshot.bucketReuse;
    const std::uint64_t churnRatioMilli = (churnDenom == 0)
        ? 0
        : (snapshot.bucketAllocGrowth * 1000ull) / churnDenom;
    const std::size_t effectiveTempCap = effective_temp_scratch_bytes_cap(requestBytes);

    const std::string msg = std::string("event=scratch_policy")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " result=" + (result ? result : "unknown")
        + " work_class=" + to_cstr(snapshot.bucketKey.workClass)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.inFlightBytes))
        + " in_flight_sets=" + std::to_string(snapshot.inFlightSets)
        + " bucket_w=" + std::to_string(snapshot.bucketKey.widthBucket)
        + " bucket_h=" + std::to_string(snapshot.bucketKey.heightBucket)
        + " bucket_step_px=" + std::to_string(snapshot.bucketKey.bucketStepPx)
        + " large_frame=" + std::to_string(snapshot.bucketKey.largeFrame ? 1 : 0)
        + " bucket_attempts=" + std::to_string(snapshot.bucketAttempts)
        + " bucket_exhausted=" + std::to_string(snapshot.bucketExhausted)
        + " bucket_alloc_growth=" + std::to_string(snapshot.bucketAllocGrowth)
        + " bucket_reuse=" + std::to_string(snapshot.bucketReuse)
        + " bucket_starvation_latched=" + std::to_string(snapshot.starvationLatched ? 1 : 0)
        + " scratch_churn_ratio_milli=" + std::to_string(churnRatioMilli)
        + " quarantine_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.quarantineBytes))
        + " quarantine_entries=" + std::to_string(snapshot.quarantineEntries)
        + " wait_ms=" + std::to_string(waitMs)
        + " max_temp_sets=" + std::to_string(kMaxTempScratchSets)
        + " max_temp_bytes=" + std::to_string(static_cast<unsigned long long>(kMaxTempScratchBytes))
        + " effective_max_temp_bytes=" + std::to_string(static_cast<unsigned long long>(effectiveTempCap))
        + " quarantine_max_bytes=" + std::to_string(static_cast<unsigned long long>(kLargeFrameQuarantineMaxBytes))
        + " quarantine_max_entries=" + std::to_string(static_cast<unsigned long long>(kLargeFrameQuarantineMaxEntries));
    JTRACE("MSACQ", msg);
}

void trace_transient_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=transient_reservation")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " kind=" + to_cstr(ReservationKind::TransientNonManager)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + (decision.reason ? decision.reason : "unspecified")
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSTRS", msg);
}

void trace_upload_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t instanceToken,
    std::uint32_t sharedTokens,
    std::uint32_t criticalTokens,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=upload_reservation")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " kind=" + to_cstr(ReservationKind::UploadCopy)
        + " instance_token=" + std::to_string(static_cast<unsigned long long>(instanceToken))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " shared_tokens=" + std::to_string(static_cast<unsigned long long>(sharedTokens))
        + " critical_tokens=" + std::to_string(static_cast<unsigned long long>(criticalTokens))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + (decision.reason ? decision.reason : "unspecified")
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSUPL", msg);
}

void trace_builder_reservation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    std::uint64_t bytesInFlight,
    std::uint64_t capBytes,
    std::uint64_t thresholdBytes,
    const ReservationDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t instanceToken,
    std::uint32_t sharedTokens,
    std::uint32_t criticalTokens,
    int waitMs,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=builder_reservation")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " kind=" + to_cstr(ReservationKind::BuilderWork)
        + " tier=" + to_cstr(tier)
        + " instance_token=" + std::to_string(static_cast<unsigned long long>(instanceToken))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(bytesInFlight))
        + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(capBytes))
        + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(thresholdBytes))
        + " shared_tokens=" + std::to_string(static_cast<unsigned long long>(sharedTokens))
        + " critical_tokens=" + std::to_string(static_cast<unsigned long long>(criticalTokens))
        + " granted=" + std::to_string(decision.granted ? 1 : 0)
        + " should_wait=" + std::to_string(decision.shouldWait ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " decision_reason=" + (decision.reason ? decision.reason : "unspecified")
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSBPR", msg);
}

void trace_budget_reclaim_retry(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint32_t attempt,
    std::size_t reclaimedBytes,
    bool success,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=reclaim_retry")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " attempt=" + std::to_string(static_cast<unsigned long long>(attempt))
        + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(reclaimedBytes))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " success=" + std::to_string(success ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSEVICT", msg);
}

void trace_pressure_checkpoint(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t pendingGrowthBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=pressure_checkpoint")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " state=" + to_cstr(checkpoint.decision.state)
        + " prev_state=" + to_cstr(checkpoint.previousState)
        + " desired_state=" + to_cstr(checkpoint.desiredState)
        + " transition=" + std::to_string(checkpoint.transition ? 1 : 0)
        + " transition_deferred_dwell=" + std::to_string(checkpoint.transitionDeferredByDwell ? 1 : 0)
        + " transition_deferred_rate=" + std::to_string(checkpoint.transitionDeferredByRate ? 1 : 0)
        + " reserve_crossing=" + std::to_string(checkpoint.reserveCrossing ? 1 : 0)
        + " reserve_crossed=" + std::to_string(checkpoint.reserveCrossedNow ? 1 : 0)
        + " sampled=" + std::to_string(checkpoint.sampled ? 1 : 0)
        + " poll_interval_ms=" + std::to_string(checkpoint.pollIntervalMs)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " soft_target_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.softTargetBytes))
        + " reserve_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.reserveBytes))
        + " manager_resident_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.managerResidentBytes))
        + " retire_pending_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.retirePendingBytes))
        + " transient_non_manager_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.transientNonManagerBytes))
        + " pressure_total_bytes=" + std::to_string(static_cast<unsigned long long>(pressure_total_bytes(checkpoint.input)))
        + " active_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.memory.activeBytes))
        + " reclaimable_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.memory.reclaimableBytes))
        + " pending_growth_bytes=" + std::to_string(static_cast<unsigned long long>(pendingGrowthBytes))
        + " allow_opportunistic=" + std::to_string(checkpoint.decision.allowOpportunistic ? 1 : 0)
        + " request_reclaim_pass=" + std::to_string(checkpoint.decision.requestReclaimPass ? 1 : 0)
        + " should_shed_non_critical=" + std::to_string(checkpoint.decision.shouldShedNonCritical ? 1 : 0)
        + " effective_reserve_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.decision.effectiveReserveBytes))
        + " effective_headroom_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.decision.effectiveHeadroomBytes))
        + " headroom_source=" + to_cstr(checkpoint.decision.headroomSource)
        + " driver_free_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.driverFreeBytes))
        + " allocator_pool_reserved_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolReservedBytes))
        + " allocator_pool_used_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolUsedBytes))
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSPRS", msg);
}

void trace_headroom_sample(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t requestBytes,
    bool sourceSwitch,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=headroom")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " effective_headroom_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.effectiveHeadroomBytes))
        + " headroom_source=" + to_cstr(checkpoint.input.headroomSource)
        + " driver_free_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.driverFreeBytes))
        + " allocator_pool_reserved_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolReservedBytes))
        + " allocator_pool_used_bytes=" + std::to_string(
            static_cast<unsigned long long>(checkpoint.input.allocatorPoolUsedBytes))
        + " source_switch=" + std::to_string(sourceSwitch ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSHDR", msg);
}

void trace_transient_non_manager_sample(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const PressureCheckpoint& checkpoint,
    std::size_t pendingGrowthBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=transient_non_manager_sample")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " transient_non_manager_bytes=" + std::to_string(static_cast<unsigned long long>(checkpoint.input.transientNonManagerBytes))
        + " pending_growth_bytes=" + std::to_string(static_cast<unsigned long long>(pendingGrowthBytes))
        + " pressure_total_bytes=" + std::to_string(static_cast<unsigned long long>(pressure_total_bytes(checkpoint.input)))
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSTRN", msg);
}

void trace_emergency_shed_action(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    PressureState state,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    bool allowed,
    std::uint64_t uploadBytesInFlight,
    std::uint64_t uploadCapBytes,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=emergency_shed")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " lane=" + to_cstr(lane)
        + " state=" + to_cstr(state)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " upload_bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(uploadBytesInFlight))
        + " upload_cap_bytes=" + std::to_string(static_cast<unsigned long long>(uploadCapBytes))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSEMS", msg);
}

void trace_lane_wait_event(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureLane lane,
    bool criticalCurrentFrame,
    int waitMs,
    const char* outcome,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=lane_wait")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " lane=" + to_cstr(lane)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " wait_ms=" + std::to_string(waitMs)
        + " outcome=" + (outcome ? outcome : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSFAIR", msg);
}

void trace_copy_compute_guard(
    const SubmissionTransaction& transaction,
    const char* commandName,
    PressureState state,
    std::uint64_t uploadBytesInFlight,
    std::uint64_t uploadCapBytes,
    bool criticalCurrentFrame,
    bool allowed,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=copy_compute_guard")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " state=" + to_cstr(state)
        + " upload_bytes_in_flight=" + std::to_string(static_cast<unsigned long long>(uploadBytesInFlight))
        + " upload_cap_bytes=" + std::to_string(static_cast<unsigned long long>(uploadCapBytes))
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " allowed=" + std::to_string(allowed ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSCOPY", msg);
}

void trace_cache_admission_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const CacheAdmissionDecision& decision,
    bool criticalCurrentFrame,
    std::uint64_t requestBytes,
    std::uint32_t observedProbationHits,
    std::uint64_t entryDigest,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=cache_admission")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " class=" + to_cstr(decision.admissionClass)
        + " allow_durable=" + std::to_string(decision.allowDurableAdmission ? 1 : 0)
        + " probation_applied=" + std::to_string(decision.probationApplied ? 1 : 0)
        + " probation_hits_required=" + std::to_string(decision.probationHitsRequired)
        + " observed_probation_hits=" + std::to_string(observedProbationHits)
        + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " max_durable_bytes=" + std::to_string(static_cast<unsigned long long>(decision.maxDurableBytes))
        + " entry_digest=" + std::to_string(static_cast<unsigned long long>(entryDigest))
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " decision_reason=" + (decision.reason ? decision.reason : "unspecified")
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSADM", msg);
}

void trace_probation_decision(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t entryDigest,
    std::uint32_t observedProbationHits,
    std::uint32_t requiredProbationHits,
    bool admitted,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=probation")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " entry_digest=" + std::to_string(static_cast<unsigned long long>(entryDigest))
        + " observed_probation_hits=" + std::to_string(observedProbationHits)
        + " required_probation_hits=" + std::to_string(requiredProbationHits)
        + " admitted=" + std::to_string(admitted ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSPRB", msg);
}

void trace_reap_pass(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t reclaimedBytes,
    bool success,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=reap_pass")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " reclaimed_bytes=" + std::to_string(static_cast<unsigned long long>(reclaimedBytes))
        + " success=" + std::to_string(success ? 1 : 0)
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSREAP", msg);
}

void trace_fragmentation_recovery(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint32_t attempt,
    std::size_t requestBytes,
    std::size_t reapedBytes,
    std::uint64_t quarantineTrimmedEntries,
    std::uint64_t graphEvictedEntries,
    bool success,
    const char* stage,
    const char* reason) {
    if (!JTRACE_ENABLED(2)) {
        return;
    }

    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("event=fragmentation_recovery")
        + " transaction_id=" + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " command=" + (commandName ? commandName : "unknown")
        + " attempt=" + std::to_string(static_cast<unsigned long long>(attempt))
        + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
        + " reaped_bytes=" + std::to_string(static_cast<unsigned long long>(reapedBytes))
        + " quarantine_trimmed_entries=" + std::to_string(
            static_cast<unsigned long long>(quarantineTrimmedEntries))
        + " graph_evicted_entries=" + std::to_string(static_cast<unsigned long long>(graphEvictedEntries))
        + " success=" + std::to_string(success ? 1 : 0)
        + " stage=" + (stage ? stage : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSFRAG", msg);
}

PressureCheckpoint evaluate_pressure_checkpoint(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    std::size_t pendingGrowthBytes,
    bool forceSample,
    const char* commandName) {
    PressureCheckpoint checkpoint{};
    checkpoint.memory = snapshot_manager_memory(resources);
    publish_manager_memory_snapshot(checkpoint.memory);

    const HeadroomTelemetry headroom = sample_headroom_telemetry();
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    ResourceManagerState& managerState = global_state();
    checkpoint.input.softTargetBytes = cfg.managerSoftTargetBytes;
    checkpoint.input.reserveBytes = cfg.managerReserveBytes;
    checkpoint.input.retirePendingBytes = checkpoint.memory.retirePendingBytes;
    checkpoint.input.transientNonManagerBytes = checkpoint.memory.transientNonManagerBytes;
    checkpoint.input.effectiveHeadroomBytes = headroom.effectiveHeadroomBytes;
    checkpoint.input.driverFreeBytes = headroom.driverFreeBytes;
    checkpoint.input.allocatorPoolReservedBytes = headroom.allocatorPoolReservedBytes;
    checkpoint.input.allocatorPoolUsedBytes = headroom.allocatorPoolUsedBytes;
    checkpoint.input.headroomSource = headroom.source;

    managerState.allocatorEffectiveHeadroomBytes.store(
        checkpoint.input.effectiveHeadroomBytes,
        std::memory_order_relaxed);
    managerState.allocatorPoolReservedBytes.store(
        checkpoint.input.allocatorPoolReservedBytes,
        std::memory_order_relaxed);
    managerState.allocatorPoolUsedBytes.store(
        checkpoint.input.allocatorPoolUsedBytes,
        std::memory_order_relaxed);

    std::uint64_t predictedResident = checkpoint.memory.activeBytes;
    std::uint64_t nextResident = 0;
    if (add_u64_checked(predictedResident, static_cast<std::uint64_t>(pendingGrowthBytes), nextResident)) {
        predictedResident = nextResident;
    }
    else {
        predictedResident = std::numeric_limits<std::uint64_t>::max();
        checkpoint.memory.overflow = true;
    }
    checkpoint.input.managerResidentBytes = predictedResident;

    const PressureDecision computedDecision = classify_pressure(checkpoint.input);
    checkpoint.decision = computedDecision;
    checkpoint.desiredState = computedDecision.state;
    checkpoint.reserveCrossedNow = reserve_crossed(checkpoint.input);
    bool headroomSourceSwitch = false;

    const std::uint64_t nowMs = monotonic_time_ms();
    PressurePolicyState& policyState = pressure_policy_state();
    {
        std::lock_guard<std::mutex> lock(policyState.mutex);
        PressureContextState& contextState = policyState.byContext[transaction.snapshot.deviceContextKey];
        const PressureState cadenceState = contextState.valid
            ? contextState.lastState
            : computedDecision.state;
        checkpoint.pollIntervalMs = pressure_poll_interval_ms_for_state(cfg, cadenceState);

        bool sampleDue = forceSample || !contextState.valid;
        if (!sampleDue && contextState.headroomSourceValid &&
            contextState.lastHeadroomSource != checkpoint.input.headroomSource) {
            sampleDue = true;
            headroomSourceSwitch = true;
        }
        if (!sampleDue) {
            const std::uint64_t elapsedMs =
                (nowMs > contextState.lastSampleMs) ? (nowMs - contextState.lastSampleMs) : 0;
            sampleDue = elapsedMs >= static_cast<std::uint64_t>(checkpoint.pollIntervalMs);
        }

        if (sampleDue) {
            checkpoint.sampled = true;
            checkpoint.previousState = contextState.valid ? contextState.lastState : computedDecision.state;
            checkpoint.desiredState = computedDecision.state;

            if (contextState.transitionWindowStartMs == 0 ||
                nowMs < contextState.transitionWindowStartMs ||
                (nowMs - contextState.transitionWindowStartMs) >= 60000ull) {
                contextState.transitionWindowStartMs = nowMs;
                contextState.transitionsInWindow = 0;
            }

            PressureDecision effectiveDecision = computedDecision;
            const bool transitionRequested =
                contextState.valid && (contextState.lastState != computedDecision.state);
            bool transitionAllowed = true;
            if (transitionRequested) {
                const bool escalation =
                    pressure_state_rank(computedDecision.state) >
                    pressure_state_rank(contextState.lastState);
                if (!escalation) {
                    const std::uint64_t stateAgeMs = (contextState.lastStateChangeMs == 0 || nowMs < contextState.lastStateChangeMs)
                        ? std::numeric_limits<std::uint64_t>::max()
                        : (nowMs - contextState.lastStateChangeMs);
                    const bool dwellOk =
                        stateAgeMs >= static_cast<std::uint64_t>(cfg.pressureStateMinDwellMs);
                    const bool rateOk =
                        contextState.transitionsInWindow < cfg.pressureStateMaxTransitionsPerMin;
                    checkpoint.transitionDeferredByDwell = !dwellOk;
                    checkpoint.transitionDeferredByRate = !rateOk;
                    transitionAllowed = dwellOk && rateOk;
                }
            }

            checkpoint.transition = transitionRequested && transitionAllowed;
            if (transitionRequested && !transitionAllowed) {
                effectiveDecision = contextState.lastDecision;
                checkpoint.decision = effectiveDecision;
                checkpoint.reserveCrossedNow = contextState.reserveCrossed;
                if (checkpoint.transitionDeferredByDwell) {
                    managerState.pressureTransitionDwellDefers.fetch_add(1, std::memory_order_relaxed);
                }
                if (checkpoint.transitionDeferredByRate) {
                    managerState.pressureTransitionRateDefers.fetch_add(1, std::memory_order_relaxed);
                }
            }
            else {
                checkpoint.decision = effectiveDecision;
            }

            checkpoint.reserveCrossing = contextState.valid &&
                (contextState.reserveCrossed != checkpoint.reserveCrossedNow);
            if (contextState.headroomSourceValid &&
                contextState.lastHeadroomSource != checkpoint.input.headroomSource) {
                headroomSourceSwitch = true;
            }
            contextState.valid = true;
            contextState.lastSampleMs = nowMs;
            contextState.lastDecision = checkpoint.decision;
            contextState.lastState = checkpoint.decision.state;
            if (!contextState.lastStateChangeMs) {
                contextState.lastStateChangeMs = nowMs;
            }
            if (checkpoint.transition) {
                contextState.lastStateChangeMs = nowMs;
                if (contextState.transitionsInWindow < std::numeric_limits<std::uint32_t>::max()) {
                    ++contextState.transitionsInWindow;
                }
            }
            contextState.reserveCrossed = checkpoint.reserveCrossedNow;
            contextState.headroomSourceValid = true;
            contextState.lastHeadroomSource = checkpoint.input.headroomSource;

            if (checkpoint.transition) {
                managerState.pressureStateTransitions.fetch_add(1, std::memory_order_relaxed);
            }
            if (checkpoint.reserveCrossing) {
                managerState.reserveCrossingEvents.fetch_add(1, std::memory_order_relaxed);
            }
            if (headroomSourceSwitch) {
                managerState.headroomSourceSwitches.fetch_add(1, std::memory_order_relaxed);
            }
        }
        else if (contextState.valid) {
            checkpoint.decision = contextState.lastDecision;
            checkpoint.previousState = contextState.lastState;
            checkpoint.desiredState = computedDecision.state;
            checkpoint.reserveCrossedNow = contextState.reserveCrossed;
        }
    }

    if (checkpoint.sampled || checkpoint.transition || checkpoint.reserveCrossing || headroomSourceSwitch) {
        const char* pressureReason = "sample";
        if (checkpoint.transitionDeferredByRate) {
            pressureReason = "state_transition_rate_deferred";
        }
        else if (checkpoint.transitionDeferredByDwell) {
            pressureReason = "state_transition_dwell_deferred";
        }
        else if (checkpoint.transition) {
            pressureReason = "state_transition";
        }
        else if (checkpoint.reserveCrossing) {
            pressureReason = "reserve_crossing";
        }
        trace_pressure_checkpoint(
            transaction,
            commandName,
            checkpoint,
            pendingGrowthBytes,
            pressureReason);
        trace_transient_non_manager_sample(
            transaction,
            commandName,
            checkpoint,
            pendingGrowthBytes,
            checkpoint.sampled ? "sampled" : "cached");
        trace_headroom_sample(
            transaction,
            commandName,
            checkpoint,
            pendingGrowthBytes,
            headroomSourceSwitch,
            headroomSourceSwitch ? "source_switch" : (checkpoint.sampled ? "sampled" : "cached"));
    }

    return checkpoint;
}

void record_allocator_oom_headroom_observation(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::size_t requestBytes) {
    const HeadroomTelemetry headroom = sample_headroom_telemetry();
    ResourceManagerState& managerState = global_state();
    managerState.allocatorEffectiveHeadroomBytes.store(headroom.effectiveHeadroomBytes, std::memory_order_relaxed);
    managerState.allocatorPoolReservedBytes.store(headroom.allocatorPoolReservedBytes, std::memory_order_relaxed);
    managerState.allocatorPoolUsedBytes.store(headroom.allocatorPoolUsedBytes, std::memory_order_relaxed);

    const bool aboveHeadroom =
        (requestBytes > 0) && (headroom.effectiveHeadroomBytes >= static_cast<std::uint64_t>(requestBytes));
    if (aboveHeadroom) {
        managerState.allocFailAboveHeadroomEvents.fetch_add(1, std::memory_order_relaxed);
    }

    PressureCheckpoint checkpoint{};
    checkpoint.input.effectiveHeadroomBytes = headroom.effectiveHeadroomBytes;
    checkpoint.input.driverFreeBytes = headroom.driverFreeBytes;
    checkpoint.input.allocatorPoolReservedBytes = headroom.allocatorPoolReservedBytes;
    checkpoint.input.allocatorPoolUsedBytes = headroom.allocatorPoolUsedBytes;
    checkpoint.input.headroomSource = headroom.source;
    trace_headroom_sample(
        transaction,
        commandName,
        checkpoint,
        requestBytes,
        false,
        aboveHeadroom ? "alloc_fail_above_headroom" : "alloc_fail_below_headroom");
}

bool run_reap_pass_for_pressure(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    const char* reason,
    std::string& outError) {
    std::size_t reclaimedBytes = 0;
    std::string reclaimError;
    if (!JuicerCuda::reap_retired_allocations(resources, reclaimedBytes, reclaimError)) {
        outError = reclaimError.empty() ? "reap pass failed" : reclaimError;
        trace_reap_pass(
            transaction,
            commandName,
            reclaimedBytes,
            false,
            outError.c_str());
        return false;
    }

    if (reclaimedBytes > 0) {
        global_state().retireReapPasses.fetch_add(1, std::memory_order_relaxed);
        global_state().retireReapBytes.fetch_add(reclaimedBytes, std::memory_order_relaxed);
    }
    trace_reap_pass(
        transaction,
        commandName,
        reclaimedBytes,
        true,
        reason);
    publish_manager_memory_snapshot(snapshot_manager_memory(resources));
    return true;
}

bool enforce_pressure_gate(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    PressureLane lane,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    bool& outRequestReclaimPass,
    std::string& outError) {
    outRequestReclaimPass = false;
    outError.clear();
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const std::uint64_t uploadCapBytes = upload_reservation_cap_bytes(cfg);
    const std::uint64_t uploadBytesInFlight =
        global_state().uploadBytesInFlight.load(std::memory_order_relaxed);
    const bool uploadCapEnabled = (uploadCapBytes > 0);
    const bool uploadCapSaturated = uploadCapEnabled && (uploadBytesInFlight >= uploadCapBytes);
    const bool nonCritical = !criticalCurrentFrame;
    PressureState pressureState = PressureState::Normal;
    const bool pressureEnabled = pressure_policy_enabled(cfg);
    if (pressureEnabled) {
        PressureCheckpoint checkpoint = evaluate_pressure_checkpoint(
            transaction,
            resources,
            requestBytes,
            requestBytes > 0,
            commandName);
        outRequestReclaimPass = checkpoint.decision.requestReclaimPass && (requestBytes > 0);
        pressureState = checkpoint.decision.state;
    }

    if (nonCritical &&
        (pressureState == PressureState::Critical ||
         pressureState == PressureState::Emergency)) {
        const char* denyReason = "deny_non_critical_growth";
        if (pressureState == PressureState::Emergency) {
            if (lane == PressureLane::Upload) {
                global_state().uploadEmergencyShedDenials.fetch_add(1, std::memory_order_relaxed);
                denyReason = "emergency_shed_upload_precedence";
            }
            else {
                global_state().builderEmergencyShedDenials.fetch_add(1, std::memory_order_relaxed);
                denyReason = "emergency_shed_builder_precedence";
            }
        }
        trace_emergency_shed_action(
            transaction,
            commandName,
            lane,
            pressureState,
            requestBytes,
            criticalCurrentFrame,
            false,
            uploadBytesInFlight,
            uploadCapBytes,
            denyReason);
        outError = std::string("pressure_shed_noncritical: lane=")
            + to_cstr(lane)
            + " state="
            + to_cstr(pressureState);
        return false;
    }

    if (lane == PressureLane::Upload && nonCritical && uploadCapSaturated) {
        global_state().copyComputeGuardShedEvents.fetch_add(1, std::memory_order_relaxed);
        trace_copy_compute_guard(
            transaction,
            commandName,
            pressureState,
            uploadBytesInFlight,
            uploadCapBytes,
            criticalCurrentFrame,
            false,
            "upload_cap_saturated_noncritical");
        outError = "pressure_copy_compute_guard: upload_cap_saturated_noncritical";
        return false;
    }

    if (pressureState == PressureState::Emergency) {
        trace_emergency_shed_action(
            transaction,
            commandName,
            lane,
            pressureState,
            requestBytes,
            criticalCurrentFrame,
            true,
            uploadBytesInFlight,
            uploadCapBytes,
            criticalCurrentFrame ? "allow_critical_progress" : "allow_noncritical_progress");
    }

    if (lane == PressureLane::Upload && uploadCapSaturated) {
        trace_copy_compute_guard(
            transaction,
            commandName,
            pressureState,
            uploadBytesInFlight,
            uploadCapBytes,
            criticalCurrentFrame,
            true,
            criticalCurrentFrame
                ? "critical_upload_allowed_despite_saturation"
                : "upload_allowed_below_shed_threshold");
    }
    return true;
}

bool try_acquire_scratch_policy_claim(
    const SubmissionTransaction& transaction,
    const char* commandName,
    const DeviceContextKey& contextKey,
    const ScratchBucketKey& bucketKey,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ScratchPolicyClaim& outClaim,
    ScratchPolicySnapshot& outSnapshot,
    ReservationAttemptInfo& outReservation) noexcept {
    outReservation = ReservationAttemptInfo{};
    ScratchPolicyState& state = scratch_policy_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    ScratchContextState& contextState = state.byContext[contextKey];
    const std::uint64_t nowMs = monotonic_time_ms();
    trim_large_frame_quarantine_decay_locked(contextState, nowMs);

    ScratchBucketEntry& bucketEntry = contextState.buckets[bucketKey];
    bucketEntry.attemptCount += 1;
    global_state().scratchBucketAcquireAttempts.fetch_add(1, std::memory_order_relaxed);

    if (requestBytes == 0) {
        bucketEntry.reuseEvents += 1;
        global_state().scratchReuseEvents.fetch_add(1, std::memory_order_relaxed);
        global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);
        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    outReservation.considered = true;
    outReservation.bytesInFlight = state.totalInFlightBytes;
    outReservation.capBytes = transient_reservation_cap_bytes(cfg);
    outReservation.thresholdBytes = transient_reservation_threshold_bytes(cfg);

    ReservationDecision reservationDecision{};
    if (static_cast<std::uint64_t>(requestBytes) < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::TransientNonManager;
        reservationInput.requestBytes = static_cast<std::uint64_t>(requestBytes);
        reservationInput.bytesInFlight = state.totalInFlightBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;
    global_state().transientReservationRequests.fetch_add(1, std::memory_order_relaxed);
    if (!reservationDecision.granted) {
        if (reservationDecision.shouldWait) {
            global_state().transientReservationDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        else {
            global_state().transientReservationDenied.fetch_add(1, std::memory_order_relaxed);
        }
        trace_transient_reservation_decision(
            transaction,
            commandName,
            requestBytes,
            outReservation.bytesInFlight,
            outReservation.capBytes,
            outReservation.thresholdBytes,
            reservationDecision,
            criticalCurrentFrame,
            static_cast<int>(reservationDecision.waitMs),
            reservationDecision.shouldWait ? "deferred" : "denied");
        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return false;
    }

    const std::size_t effectiveCapBytes = effective_temp_scratch_bytes_cap(requestBytes);
    const bool setsOk = bucketEntry.inFlightSets < kMaxTempScratchSets;
    const bool bytesOk = requestBytes <=
        (effectiveCapBytes - std::min(bucketEntry.inFlightBytes, effectiveCapBytes));
    if (!setsOk || !bytesOk) {
        bucketEntry.exhaustedCount += 1;
        global_state().scratchBucketExhaustedEvents.fetch_add(1, std::memory_order_relaxed);

        const bool starvationNow =
            (bucketEntry.attemptCount >= 8) &&
            (bucketEntry.exhaustedCount * 4 >= bucketEntry.attemptCount);
        if (starvationNow && !bucketEntry.starvationLatched) {
            bucketEntry.starvationLatched = true;
            global_state().scratchBucketStarvationEvents.fetch_add(1, std::memory_order_relaxed);
        }
        else if (!starvationNow) {
            bucketEntry.starvationLatched = false;
        }

        snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
        return false;
    }

    bucketEntry.inFlightSets += 1;
    bucketEntry.inFlightBytes += requestBytes;
    bucketEntry.allocGrowthEvents += 1;
    {
        std::size_t nextContextBytes = 0;
        if (add_bytes_checked(contextState.totalInFlightBytes, requestBytes, nextContextBytes)) {
            contextState.totalInFlightBytes = nextContextBytes;
        }
        else {
            contextState.totalInFlightBytes = std::numeric_limits<std::size_t>::max();
        }
    }
    {
        std::uint64_t nextGlobalBytes = 0;
        if (add_u64_checked(state.totalInFlightBytes, static_cast<std::uint64_t>(requestBytes), nextGlobalBytes)) {
            state.totalInFlightBytes = nextGlobalBytes;
        }
        else {
            state.totalInFlightBytes = std::numeric_limits<std::uint64_t>::max();
        }
    }
    global_state().scratchAllocGrowthEvents.fetch_add(1, std::memory_order_relaxed);
    bucketEntry.starvationLatched = false;
    global_state().transientNonManagerBytes.store(state.totalInFlightBytes, std::memory_order_relaxed);
    global_state().transientReservationGranted.fetch_add(1, std::memory_order_relaxed);
    if (reservationDecision.reason && std::string_view(reservationDecision.reason) == "critical_last_resort") {
        trace_transient_reservation_decision(
            transaction,
            commandName,
            requestBytes,
            outReservation.bytesInFlight,
            outReservation.capBytes,
            outReservation.thresholdBytes,
            reservationDecision,
            criticalCurrentFrame,
            0,
            "critical_last_resort");
    }

    outClaim.contextKey = contextKey;
    outClaim.bucketKey = bucketKey;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;

    snapshot_bucket_state_locked(contextState, bucketEntry, bucketKey, outSnapshot);
    return true;
}

bool acquire_scratch_policy_claim_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    ScratchWorkClass workClass,
    int width,
    int height,
    std::size_t requestBytes,
    bool criticalCurrentFrame,
    ScratchPolicyClaim& outClaim,
    std::string& outError) {
    outClaim = ScratchPolicyClaim{};
    outError.clear();
    ResourceManagerState& state = global_state();

    const ScratchBucketKey bucketKey = make_scratch_bucket_key(width, height, workClass);
    const int waitBudgetMs = scratch_wait_budget_ms(bucketKey, requestBytes);
    ScratchPolicySnapshot snapshot{};
    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_scratch_policy_claim(
                transaction,
                commandName,
                transaction.snapshot.deviceContextKey,
                bucketKey,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                snapshot,
                reservation)) {
            if (waitedMs > 0) {
                state.scratchPolicyWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Builder,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        "scratch_policy_wait");
                }
                trace_scratch_policy_decision(
                    transaction,
                    commandName,
                    "admit_after_wait",
                    requestBytes,
                    snapshot,
                    waitedMs);
            }
            else if (requestBytes == 0) {
                trace_scratch_policy_decision(
                    transaction,
                    commandName,
                    "reuse_hit",
                    requestBytes,
                    snapshot,
                    waitedMs);
            }
            return true;
        }

        if (waitedMs >= waitBudgetMs) {
            state.scratchPolicyExhaustedEvents.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    "starved",
                    (reservation.considered && reservation.decision.reason)
                        ? reservation.decision.reason
                        : "scratch_exhausted");
            }
            trace_scratch_policy_decision(
                transaction,
                commandName,
                "exhausted",
                requestBytes,
                snapshot,
                waitedMs);
            if (reservation.considered && !reservation.decision.granted) {
                outError = std::string(kReservationDeferredPrefix)
                    + " command=" + (commandName ? commandName : "unknown")
                    + " work_class=" + to_cstr(workClass)
                    + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                    + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                    + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                    + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                    + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                    + " wait_ms=" + std::to_string(waitedMs)
                    + " wait_budget_ms=" + std::to_string(waitBudgetMs)
                    + " reason=" + (reservation.decision.reason ? reservation.decision.reason : "unspecified");
            }
            else {
                outError = std::string(kScratchExhaustedPrefix)
                    + " command=" + (commandName ? commandName : "unknown")
                    + " work_class=" + to_cstr(workClass)
                    + " bucket_w=" + std::to_string(bucketKey.widthBucket)
                    + " bucket_h=" + std::to_string(bucketKey.heightBucket)
                    + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                    + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(snapshot.inFlightBytes))
                    + " in_flight_sets=" + std::to_string(snapshot.inFlightSets)
                    + " bucket_exhausted=" + std::to_string(snapshot.bucketExhausted)
                    + " bucket_attempts=" + std::to_string(snapshot.bucketAttempts)
                    + " wait_ms=" + std::to_string(waitedMs)
                    + " wait_budget_ms=" + std::to_string(waitBudgetMs);
            }
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kScratchWaitStepMs));
        waitedMs += kScratchWaitStepMs;
    }
}

bool try_acquire_builder_reservation_claim(
    const SubmissionTransaction& transaction,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) noexcept {
    outClaim = BuilderReservationClaim{};
    outReservation = ReservationAttemptInfo{};

    if (requestBytes == 0) {
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    BuilderReservationState& state = builder_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.instanceToken = transaction.snapshot.instanceToken.value;

    outReservation.considered = true;
    std::uint64_t& tierTotalBytes = builder_total_bytes_for_tier(state, tier);
    outReservation.bytesInFlight = tierTotalBytes;
    outReservation.capBytes = builder_reservation_cap_bytes(cfg, tier);
    outReservation.thresholdBytes = builder_reservation_threshold_bytes(cfg, tier);

    ReservationDecision reservationDecision{};
    if (requestBytes < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::BuilderWork;
        reservationInput.requestBytes = requestBytes;
        reservationInput.bytesInFlight = tierTotalBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;

    if (!reservationDecision.granted) {
        return false;
    }

    BuilderReservationContextState& contextState =
        state.byContext[transaction.snapshot.deviceContextKey];
    BuilderReservationContextState::FairnessEntry& fairness =
        contextState.fairnessByInstance[outReservation.instanceToken];
    refill_builder_fairness_tokens(fairness, cfg, monotonic_time_ms());

    bool fairnessGranted = false;
    if (criticalCurrentFrame && fairness.criticalTokens > 0) {
        --fairness.criticalTokens;
        fairnessGranted = true;
    }
    else if (fairness.sharedTokens > 0) {
        --fairness.sharedTokens;
        fairnessGranted = true;
    }
    outReservation.sharedTokens = fairness.sharedTokens;
    outReservation.criticalTokens = fairness.criticalTokens;
    if (!fairnessGranted) {
        ReservationDecision fairnessDecision{};
        fairnessDecision.granted = false;
        fairnessDecision.shouldWait = true;
        fairnessDecision.waitMs = static_cast<std::uint32_t>(kBuilderReservationWaitStepMs);
        fairnessDecision.reason = "fairness_tokens_exhausted";
        outReservation.decision = fairnessDecision;
        return false;
    }

    std::uint64_t& contextTierBytes = builder_context_bytes_for_tier(contextState, tier);
    std::uint64_t nextContextBytes = 0;
    if (add_u64_checked(contextTierBytes, requestBytes, nextContextBytes)) {
        contextTierBytes = nextContextBytes;
    }
    else {
        contextTierBytes = std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextTotalBytes = 0;
    if (add_u64_checked(tierTotalBytes, requestBytes, nextTotalBytes)) {
        tierTotalBytes = nextTotalBytes;
    }
    else {
        tierTotalBytes = std::numeric_limits<std::uint64_t>::max();
    }
    builder_global_gauge_for_tier(global_state(), tier).store(tierTotalBytes, std::memory_order_relaxed);
    outClaim.contextKey = transaction.snapshot.deviceContextKey;
    outClaim.tier = tier;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;
    return true;
}

bool acquire_builder_reservation_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    BuilderReservationTier tier,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    BuilderReservationClaim& outClaim,
    std::string& outError) {
    outClaim = BuilderReservationClaim{};
    outError.clear();

    if (requestBytes == 0) {
        return true;
    }

    ResourceManagerState& state = global_state();
    state.builderReservationRequests.fetch_add(1, std::memory_order_relaxed);

    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_builder_reservation_claim(
                transaction,
                tier,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                reservation)) {
            state.builderReservationGranted.fetch_add(1, std::memory_order_relaxed);
            if (waitedMs > 0) {
                state.builderFairnessWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Builder,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        reservation.decision.reason ? reservation.decision.reason : "waited");
                }
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admit_after_wait");
            }
            else if (reservation.decision.reason &&
                     std::string_view(reservation.decision.reason) == "critical_last_resort") {
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "critical_last_resort");
            }
            else if (JTRACE_ENABLED(3)) {
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admitted");
            }
            return true;
        }

        const ReservationDecision& decision = reservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (fairnessDeferred) {
            state.builderFairnessTokenDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        if (!decision.shouldWait) {
            state.builderReservationDenied.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    "denied",
                    decision.reason ? decision.reason : "reservation_denied");
            }
            trace_builder_reservation_decision(
                transaction,
                commandName,
                tier,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "denied");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::BuilderWork)
                + " tier=" + to_cstr(tier)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kBuilderReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "unspecified");
            return false;
        }

        if (waitedMs >= kBuilderReservationWaitMaxMs) {
            state.builderReservationDeferred.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                state.builderReservationBypass.fetch_add(1, std::memory_order_relaxed);
                if (fairnessDeferred) {
                    state.builderFairnessTokenBypass.fetch_add(1, std::memory_order_relaxed);
                }
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalBuilderWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalBuilderWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Builder,
                    criticalCurrentFrame,
                    waitedMs,
                    fairnessDeferred ? "bypass_after_starvation" : "deferred_bypass",
                    decision.reason ? decision.reason : "wait_budget_reached");
                trace_builder_reservation_decision(
                    transaction,
                    commandName,
                    tier,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "deferred_bypass_wait_budget");
                return true;
            }
            trace_builder_reservation_decision(
                transaction,
                commandName,
                tier,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "deferred_wait_budget");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::BuilderWork)
                + " tier=" + to_cstr(tier)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kBuilderReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "wait_budget_reached");
            return false;
        }

        const int sleepMs = std::max<int>(
            1,
            std::min<int>(
                static_cast<int>(decision.waitMs > 0 ? decision.waitMs : 1),
                kBuilderReservationWaitStepMs));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        waitedMs += sleepMs;
    }
}

bool try_acquire_upload_reservation_claim(
    const SubmissionTransaction& transaction,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    UploadReservationClaim& outClaim,
    ReservationAttemptInfo& outReservation) noexcept {
    outClaim = UploadReservationClaim{};
    outReservation = ReservationAttemptInfo{};

    if (requestBytes == 0) {
        return true;
    }

    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    UploadReservationState& state = upload_reservation_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    outReservation.instanceToken = transaction.snapshot.instanceToken.value;

    outReservation.considered = true;
    outReservation.bytesInFlight = state.totalInFlightBytes;
    outReservation.capBytes = upload_reservation_cap_bytes(cfg);
    outReservation.thresholdBytes = upload_reservation_threshold_bytes(cfg);

    ReservationDecision reservationDecision{};
    if (requestBytes < outReservation.thresholdBytes) {
        reservationDecision.granted = true;
        reservationDecision.reason = "below_threshold";
    }
    else {
        ReservationInput reservationInput{};
        reservationInput.kind = ReservationKind::UploadCopy;
        reservationInput.requestBytes = requestBytes;
        reservationInput.bytesInFlight = state.totalInFlightBytes;
        reservationInput.capBytes = outReservation.capBytes;
        reservationInput.criticalCurrentFrame = criticalCurrentFrame;
        reservationDecision = classify_reservation(reservationInput);
    }
    outReservation.decision = reservationDecision;

    if (!reservationDecision.granted) {
        return false;
    }

    UploadReservationContextState& contextState =
        state.byContext[transaction.snapshot.deviceContextKey];
    UploadReservationContextState::FairnessEntry& fairness =
        contextState.fairnessByInstance[outReservation.instanceToken];
    refill_upload_fairness_tokens(fairness, cfg, monotonic_time_ms());

    bool fairnessGranted = false;
    if (criticalCurrentFrame && fairness.criticalTokens > 0) {
        --fairness.criticalTokens;
        fairnessGranted = true;
    }
    else if (fairness.sharedTokens > 0) {
        --fairness.sharedTokens;
        fairnessGranted = true;
    }
    outReservation.sharedTokens = fairness.sharedTokens;
    outReservation.criticalTokens = fairness.criticalTokens;
    if (!fairnessGranted) {
        ReservationDecision fairnessDecision{};
        fairnessDecision.granted = false;
        fairnessDecision.shouldWait = true;
        fairnessDecision.waitMs = static_cast<std::uint32_t>(kUploadReservationWaitStepMs);
        fairnessDecision.reason = "fairness_tokens_exhausted";
        outReservation.decision = fairnessDecision;
        return false;
    }

    std::uint64_t nextContextBytes = 0;
    if (add_u64_checked(contextState.inFlightBytes, requestBytes, nextContextBytes)) {
        contextState.inFlightBytes = nextContextBytes;
    }
    else {
        contextState.inFlightBytes = std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextTotalBytes = 0;
    if (add_u64_checked(state.totalInFlightBytes, requestBytes, nextTotalBytes)) {
        state.totalInFlightBytes = nextTotalBytes;
    }
    else {
        state.totalInFlightBytes = std::numeric_limits<std::uint64_t>::max();
    }
    global_state().uploadBytesInFlight.store(state.totalInFlightBytes, std::memory_order_relaxed);
    outClaim.contextKey = transaction.snapshot.deviceContextKey;
    outClaim.bytes = requestBytes;
    outClaim.acquired = true;
    return true;
}

bool acquire_upload_reservation_with_wait(
    const SubmissionTransaction& transaction,
    const char* commandName,
    std::uint64_t requestBytes,
    bool criticalCurrentFrame,
    UploadReservationClaim& outClaim,
    std::string& outError) {
    outClaim = UploadReservationClaim{};
    outError.clear();

    if (requestBytes == 0) {
        return true;
    }

    ResourceManagerState& state = global_state();
    state.uploadReservationRequests.fetch_add(1, std::memory_order_relaxed);

    ReservationAttemptInfo reservation{};
    int waitedMs = 0;
    while (true) {
        if (try_acquire_upload_reservation_claim(
                transaction,
                requestBytes,
                criticalCurrentFrame,
                outClaim,
                reservation)) {
            state.uploadReservationGranted.fetch_add(1, std::memory_order_relaxed);
            if (waitedMs > 0) {
                state.uploadFairnessWaitEvents.fetch_add(1, std::memory_order_relaxed);
                if (criticalCurrentFrame) {
                    const std::uint64_t waitedMsU64 =
                        static_cast<std::uint64_t>(std::max(waitedMs, 0));
                    state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                    state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                    trace_lane_wait_event(
                        transaction,
                        commandName,
                        PressureLane::Upload,
                        criticalCurrentFrame,
                        waitedMs,
                        "admit_after_wait",
                        reservation.decision.reason ? reservation.decision.reason : "waited");
                }
                trace_upload_reservation_decision(
                    transaction,
                    commandName,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "admit_after_wait");
            }
            else if (reservation.decision.reason &&
                     std::string_view(reservation.decision.reason) == "critical_last_resort") {
                trace_upload_reservation_decision(
                    transaction,
                    commandName,
                    requestBytes,
                    reservation.bytesInFlight,
                    reservation.capBytes,
                    reservation.thresholdBytes,
                    reservation.decision,
                    criticalCurrentFrame,
                    reservation.instanceToken,
                    reservation.sharedTokens,
                    reservation.criticalTokens,
                    waitedMs,
                    "critical_last_resort");
            }
            return true;
        }

        const ReservationDecision& decision = reservation.decision;
        const bool fairnessDeferred =
            (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
        if (fairnessDeferred) {
            state.uploadFairnessTokenDeferred.fetch_add(1, std::memory_order_relaxed);
        }
        if (!decision.shouldWait) {
            state.uploadReservationDenied.fetch_add(1, std::memory_order_relaxed);
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Upload,
                    criticalCurrentFrame,
                    waitedMs,
                    "denied",
                    decision.reason ? decision.reason : "reservation_denied");
            }
            trace_upload_reservation_decision(
                transaction,
                commandName,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "denied");
            outError = std::string(kReservationDeferredPrefix)
                + " command=" + (commandName ? commandName : "unknown")
                + " kind=" + to_cstr(ReservationKind::UploadCopy)
                + " request_bytes=" + std::to_string(static_cast<unsigned long long>(requestBytes))
                + " in_flight_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.bytesInFlight))
                + " cap_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.capBytes))
                + " threshold_bytes=" + std::to_string(static_cast<unsigned long long>(reservation.thresholdBytes))
                + " critical_current_frame=" + std::to_string(criticalCurrentFrame ? 1 : 0)
                + " wait_ms=" + std::to_string(waitedMs)
                + " wait_budget_ms=" + std::to_string(kUploadReservationWaitMaxMs)
                + " reason=" + (decision.reason ? decision.reason : "unspecified");
            return false;
        }

        if (waitedMs >= kUploadReservationWaitMaxMs) {
            state.uploadReservationDeferred.fetch_add(1, std::memory_order_relaxed);
            state.uploadReservationBypass.fetch_add(1, std::memory_order_relaxed);
            if (fairnessDeferred) {
                state.uploadFairnessTokenBypass.fetch_add(1, std::memory_order_relaxed);
            }
            if (criticalCurrentFrame) {
                const std::uint64_t waitedMsU64 =
                    static_cast<std::uint64_t>(std::max(waitedMs, 0));
                state.criticalUploadWaitEvents.fetch_add(1, std::memory_order_relaxed);
                state.criticalUploadWaitTotalMs.fetch_add(waitedMsU64, std::memory_order_relaxed);
                if (fairnessDeferred) {
                    state.criticalLaneStarvationEvents.fetch_add(1, std::memory_order_relaxed);
                }
                trace_lane_wait_event(
                    transaction,
                    commandName,
                    PressureLane::Upload,
                    criticalCurrentFrame,
                    waitedMs,
                    fairnessDeferred ? "bypass_after_starvation" : "deferred_bypass",
                    decision.reason ? decision.reason : "wait_budget_reached");
            }
            trace_upload_reservation_decision(
                transaction,
                commandName,
                requestBytes,
                reservation.bytesInFlight,
                reservation.capBytes,
                reservation.thresholdBytes,
                decision,
                criticalCurrentFrame,
                reservation.instanceToken,
                reservation.sharedTokens,
                reservation.criticalTokens,
                waitedMs,
                "deferred_bypass_wait_budget");
            return true;
        }

        const int sleepMs = std::max<int>(
            1,
            std::min<int>(
                static_cast<int>(decision.waitMs > 0 ? decision.waitMs : 1),
                kUploadReservationWaitStepMs));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        waitedMs += sleepMs;
    }
}

bool key_digests_equal(const KeyDigests& lhs, const KeyDigests& rhs) noexcept {
    return lhs.uploadCoreHash == rhs.uploadCoreHash &&
        lhs.dirHash == rhs.dirHash &&
        lhs.scannerHash == rhs.scannerHash &&
        lhs.autoExposureHash == rhs.autoExposureHash;
}

ResourcePlan make_uniform_resource_plan(AcquireStatus status) noexcept {
    ResourcePlan plan{};
    for (ResourceKind kind : kResourceKindOrder) {
        ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        entry.acquire.status = status;
        entry.acquire.shouldBuild = (status == AcquireStatus::Miss);
        entry.invalidated = false;
    }
    return plan;
}

AcquireStatus combine_status(const ResourcePlan& plan) noexcept {
    bool sawMiss = false;
    bool sawBusy = false;
    bool sawExhausted = false;
    for (ResourceKind kind : kResourceKindOrder) {
        const AcquireStatus status = resource_plan_entry(plan, kind).acquire.status;
        if (status == AcquireStatus::Error) {
            return AcquireStatus::Error;
        }
        if (status == AcquireStatus::Exhausted) {
            sawExhausted = true;
        }
        else if (status == AcquireStatus::Busy) {
            sawBusy = true;
        }
        else if (status == AcquireStatus::Miss) {
            sawMiss = true;
        }
    }
    if (sawExhausted) {
        return AcquireStatus::Exhausted;
    }
    if (sawBusy) {
        return AcquireStatus::Busy;
    }
    if (sawMiss) {
        return AcquireStatus::Miss;
    }
    return AcquireStatus::Hit;
}

bool validate_resource_kind_onboarding_contract(std::string& outError) noexcept {
    if (!resource_kind_contract_is_valid()) {
        outError = "resource kind onboarding contract invalid";
        return false;
    }
    for (ResourceKind kind : kResourceKindOrder) {
        const ResourceKindContractEntry& entry = resource_kind_contract_entry(kind);
        if (entry.kind != kind ||
            entry.keyField == nullptr ||
            entry.invalidationLane == nullptr ||
            entry.resourceNode == nullptr ||
            entry.acquireStatusField == nullptr ||
            entry.telemetryTag == nullptr) {
            outError = "resource kind onboarding entry missing required fields";
            return false;
        }
    }
    return true;
}

void trace_lifecycle_stage_decision(
    const SubmissionTransaction& transaction,
    ContextLifecycleState observedState,
    const char* stage,
    bool accepted,
    const char* reason) {
    const std::uintptr_t contextBits =
        reinterpret_cast<std::uintptr_t>(transaction.snapshot.deviceContextKey.contextOpaque);
    const std::string msg = std::string("transaction_id=") + std::to_string(transaction.transactionId)
        + " snapshot_id=" + std::to_string(transaction.snapshot.snapshotId)
        + " trace_schema=" + std::to_string(transaction.snapshot.traceSchemaVersion)
        + " stage=" + (stage ? stage : "unknown")
        + " device_id=" + std::to_string(transaction.snapshot.deviceContextKey.deviceId)
        + " context=" + std::to_string(contextBits)
        + " observed_state=" + to_cstr(observedState)
        + " accepted=" + std::to_string(accepted ? 1 : 0)
        + " reason=" + (reason ? reason : "unspecified");
    JTRACE("MSLCY", msg);
}

bool lifecycle_state_allowed_for_stage(ContextLifecycleState state, bool allowNonActiveRelease) {
    if (allowNonActiveRelease) {
        return state != ContextLifecycleState::Unbound;
    }
    return state == ContextLifecycleState::Active;
}

bool validate_lifecycle_for_stage(const SubmissionTransaction& transaction,
                                  const char* stage,
                                  bool allowNonActiveRelease,
                                  std::string* outError) {
    ContextLifecycleState lifecycleState = ContextLifecycleState::Unbound;
    if (!registry_get_lifecycle_state(transaction.snapshot.deviceContextKey, lifecycleState)) {
        global_state().lifecycleStageRejects.fetch_add(1, std::memory_order_relaxed);
        trace_lifecycle_stage_decision(transaction, lifecycleState, stage, false, "missing_registry_entry");
        if (outError) {
            *outError = "missing registry entry for lifecycle validation";
        }
        return false;
    }
    if (!lifecycle_state_allowed_for_stage(lifecycleState, allowNonActiveRelease)) {
        global_state().lifecycleStageRejects.fetch_add(1, std::memory_order_relaxed);
        trace_lifecycle_stage_decision(transaction, lifecycleState, stage, false, "lifecycle_state_not_allowed");
        if (outError) {
            *outError = std::string("lifecycle state not allowed for stage (state=")
                + to_cstr(lifecycleState) + ")";
        }
        return false;
    }
    if (JTRACE_ENABLED(3)) {
        trace_lifecycle_stage_decision(transaction, lifecycleState, stage, true, "stage_allowed");
    }
    return true;
}

bool ensure_active_for_command(
    const SubmissionTransaction& transaction,
    std::string& outError,
    const char* commandName) {
    if (!validate_lifecycle_for_stage(transaction, commandName ? commandName : "command", false, &outError)) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            commandName ? commandName : "command_requires_active_submission");
        return false;
    }

    ResourceManagerState& state = global_state();
    StaleInput staleInput{};
    staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
    staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
    staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
    staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
    staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
    staleInput.observedLeaseGeneration = transaction.active ? transaction.leaseGeneration : 0;
    staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);

    const StaleDecision staleDecision = classify_stale_path(staleInput);
    telemetry_trace_stale_decision(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        commandName ? commandName : "command",
        staleInput,
        staleDecision);
    if (!staleDecision.hardStale && !staleDecision.hardMiss) {
        return true;
    }
    telemetry_record_stale_tuple_hard_reject();
    outError = std::string("stale transaction in command path (reason=") +
        to_cstr(staleDecision.reason) + ")";
    telemetry_record_module_boundary_violation();
    telemetry_trace_module_boundary_violation(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        commandName ? commandName : "command_requires_active_submission");
    return false;
}

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
struct BaseGraphKey {
    int width = 0;
    int height = 0;
    int nComponents = 0;
    int renderMode = 0;
};

struct BaseGraphEntry {
    BaseGraphKey key{};
    void* graphOpaque = nullptr;
    void* execOpaque = nullptr;
    void* kernelNodeOpaque = nullptr;
    void* kernelFuncOpaque = nullptr;
    unsigned int gridX = 0;
    unsigned int gridY = 0;
    unsigned int gridZ = 0;
    unsigned int blockX = 0;
    unsigned int blockY = 0;
    unsigned int blockZ = 0;
    unsigned int sharedMemBytes = 0;
    std::uint64_t lastUseTick = 0;
};

struct BaseGraphBucketState {
    std::mutex mutex;
    std::uint64_t contextEpoch = 0;
    std::uint64_t useTick = 0;
    std::vector<BaseGraphEntry> entries;
    std::unordered_map<std::uint64_t, std::uint32_t> probationHitsByDigest;
};

struct BaseGraphCacheState {
    std::mutex mutex;
    std::unordered_map<DeviceContextKey, std::shared_ptr<BaseGraphBucketState>, DeviceContextKeyHash> byContext;
};

BaseGraphCacheState& base_graph_cache_state() noexcept {
    static BaseGraphCacheState state{};
    return state;
}

using BasePipelineLaunchFn = cudaError_t(*)(const JuicerCuda::PipelineRunParams*, void*);

BasePipelineLaunchFn base_pipeline_launch_fn_for_mode(int renderModeKey) noexcept {
    switch (renderModeKey) {
    case 0:
        return juicer_cuda_negative_pipeline;
    case 1:
        return juicer_cuda_print_pipeline;
    default:
        return nullptr;
    }
}

bool base_graph_key_equal(const BaseGraphKey& a, const BaseGraphKey& b) noexcept {
    return a.width == b.width &&
        a.height == b.height &&
        a.nComponents == b.nComponents &&
        a.renderMode == b.renderMode;
}

std::uint64_t base_graph_key_digest(const BaseGraphKey& key) noexcept {
    std::uint64_t digest = 1469598103934665603ull;
    auto fold = [&digest](std::uint64_t value) noexcept {
        digest ^= value + 0x9e3779b97f4a7c15ull + (digest << 6u) + (digest >> 2u);
    };
    fold(static_cast<std::uint64_t>(std::max(0, key.width)));
    fold(static_cast<std::uint64_t>(std::max(0, key.height)));
    fold(static_cast<std::uint64_t>(std::max(0, key.nComponents)));
    fold(static_cast<std::uint64_t>(std::max(0, key.renderMode)));
    return digest;
}

std::uint64_t estimate_base_graph_request_bytes(const BaseGraphKey& key) noexcept {
    const std::uint64_t w = static_cast<std::uint64_t>(std::max(0, key.width));
    const std::uint64_t h = static_cast<std::uint64_t>(std::max(0, key.height));
    const std::uint64_t n = static_cast<std::uint64_t>(std::max(0, key.nComponents));
    std::uint64_t pixels = 0;
    if (!mul_u64_checked(w, h, pixels)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    constexpr std::uint64_t kBaseGraphMetadataBytes = 2ull * 1024ull * 1024ull;
    constexpr std::uint64_t kPerMegapixelMetadataBytes = 64ull * 1024ull;
    constexpr std::uint64_t kPerComponentMetadataBytes = 128ull * 1024ull;
    constexpr std::uint64_t kMegapixelDivisor = 1024ull * 1024ull;

    std::uint64_t bytes = kBaseGraphMetadataBytes;
    std::uint64_t mpBytes = 0;
    if (!mul_u64_checked(pixels / kMegapixelDivisor, kPerMegapixelMetadataBytes, mpBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    std::uint64_t nextBytes = 0;
    if (!add_u64_checked(bytes, mpBytes, nextBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    bytes = nextBytes;

    std::uint64_t componentBytes = 0;
    if (!mul_u64_checked(n, kPerComponentMetadataBytes, componentBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    if (!add_u64_checked(bytes, componentBytes, nextBytes)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return nextBytes;
}

void destroy_base_graph_entry(BaseGraphEntry& entry) noexcept {
    if (entry.execOpaque) {
        cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(entry.execOpaque));
        entry.execOpaque = nullptr;
    }
    if (entry.graphOpaque) {
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(entry.graphOpaque));
        entry.graphOpaque = nullptr;
    }
    entry.kernelNodeOpaque = nullptr;
    entry.kernelFuncOpaque = nullptr;
    entry.gridX = 0;
    entry.gridY = 0;
    entry.gridZ = 0;
    entry.blockX = 0;
    entry.blockY = 0;
    entry.blockZ = 0;
    entry.sharedMemBytes = 0;
    entry.lastUseTick = 0;
}

std::shared_ptr<BaseGraphBucketState> get_or_create_base_graph_bucket(
    const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.byContext.find(key);
    if (it != state.byContext.end() && it->second) {
        return it->second;
    }
    auto bucket = std::make_shared<BaseGraphBucketState>();
    state.byContext[key] = bucket;
    return bucket;
}

void clear_base_graph_bucket(BaseGraphBucketState& bucket) noexcept {
    for (auto& entry : bucket.entries) {
        destroy_base_graph_entry(entry);
    }
    bucket.entries.clear();
    bucket.probationHitsByDigest.clear();
    bucket.useTick = 0;
}

std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return 0;
        }
        bucket = it->second;
    }
    if (!bucket) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(bucket->mutex);
    const std::uint64_t evictedEntries = static_cast<std::uint64_t>(bucket->entries.size());
    if (evictedEntries == 0) {
        return 0;
    }
    clear_base_graph_bucket(*bucket);
    return evictedEntries;
}

void retire_base_graph_cache_for_context(const DeviceContextKey& key) noexcept {
    BaseGraphCacheState& state = base_graph_cache_state();
    std::shared_ptr<BaseGraphBucketState> bucket;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.byContext.find(key);
        if (it == state.byContext.end()) {
            return;
        }
        bucket = it->second;
        state.byContext.erase(it);
    }
    if (bucket) {
        std::lock_guard<std::mutex> lock(bucket->mutex);
        clear_base_graph_bucket(*bucket);
    }
}

BaseGraphEntry* find_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key) noexcept {
    for (auto& entry : bucket.entries) {
        if (base_graph_key_equal(entry.key, key) &&
            entry.execOpaque &&
            entry.graphOpaque &&
            entry.kernelNodeOpaque) {
            return &entry;
        }
    }
    return nullptr;
}

BaseGraphEntry* build_base_graph_entry(
    BaseGraphBucketState& bucket,
    const BaseGraphKey& key,
    BasePipelineLaunchFn launchFn,
    JuicerCuda::PipelineRunParams& run,
    cudaStream_t stream) noexcept {
    if (!launchFn) {
        return nullptr;
    }

    constexpr std::size_t kBaseGraphCap = 4;
    if (bucket.entries.size() >= kBaseGraphCap && !bucket.entries.empty()) {
        std::size_t victim = 0;
        std::uint64_t bestTick = bucket.entries[0].lastUseTick;
        for (std::size_t i = 1; i < bucket.entries.size(); ++i) {
            if (bucket.entries[i].lastUseTick < bestTick) {
                bestTick = bucket.entries[i].lastUseTick;
                victim = i;
            }
        }
        destroy_base_graph_entry(bucket.entries[victim]);
        bucket.entries.erase(bucket.entries.begin() + static_cast<std::ptrdiff_t>(victim));
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t kernelNode = nullptr;

    cudaError_t capErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (capErr != cudaSuccess) {
        return nullptr;
    }

    cudaError_t launchErr = launchFn(&run, reinterpret_cast<void*>(stream));
    if (launchErr != cudaSuccess) {
        cudaGraph_t abortGraph = nullptr;
        cudaStreamEndCapture(stream, &abortGraph);
        if (abortGraph) {
            cudaGraphDestroy(abortGraph);
        }
        return nullptr;
    }

    capErr = cudaStreamEndCapture(stream, &graph);
    if (capErr != cudaSuccess || !graph) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
        return nullptr;
    }

    cudaError_t instErr = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (instErr != cudaSuccess || !exec) {
        cudaGraphDestroy(graph);
        return nullptr;
    }

    std::size_t nodeCount = 0;
    cudaError_t nodeErr = cudaGraphGetNodes(graph, nullptr, &nodeCount);
    if (nodeErr == cudaSuccess && nodeCount > 0) {
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        nodeErr = cudaGraphGetNodes(graph, nodes.data(), &nodeCount);
        if (nodeErr == cudaSuccess) {
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType nodeType = cudaGraphNodeTypeEmpty;
                if (cudaGraphNodeGetType(node, &nodeType) == cudaSuccess && nodeType == cudaGraphNodeTypeKernel) {
                    kernelNode = node;
                    break;
                }
            }
        }
    }

    if (!kernelNode) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return nullptr;
    }

    cudaKernelNodeParams baseParams{};
    if (cudaGraphKernelNodeGetParams(kernelNode, &baseParams) != cudaSuccess || !baseParams.func) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return nullptr;
    }

    BaseGraphEntry entry{};
    entry.key = key;
    entry.graphOpaque = reinterpret_cast<void*>(graph);
    entry.execOpaque = reinterpret_cast<void*>(exec);
    entry.kernelNodeOpaque = reinterpret_cast<void*>(kernelNode);
    entry.kernelFuncOpaque = baseParams.func;
    entry.gridX = baseParams.gridDim.x;
    entry.gridY = baseParams.gridDim.y;
    entry.gridZ = baseParams.gridDim.z;
    entry.blockX = baseParams.blockDim.x;
    entry.blockY = baseParams.blockDim.y;
    entry.blockZ = baseParams.blockDim.z;
    entry.sharedMemBytes = baseParams.sharedMemBytes;
    entry.lastUseTick = bucket.useTick;
    bucket.entries.push_back(entry);
    return &bucket.entries.back();
}
#else
std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey&) noexcept {
    return 0;
}
#endif

bool run_fragmentation_recovery_once(
    const SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const char* commandName,
    std::size_t requestBytes,
    std::uint32_t attempt,
    bool captureMemorySnapshots,
    std::string& outError) {
    outError.clear();
    ResourceManagerState& managerState = global_state();
    managerState.fragmentationRecoveryAttempts.fetch_add(1, std::memory_order_relaxed);

    std::size_t reapedBytes = 0;
    std::string reapError;
    if (!JuicerCuda::reap_retired_allocations(resources, reapedBytes, reapError)) {
        trace_reap_pass(
            transaction,
            commandName,
            reapedBytes,
            false,
            reapError.empty() ? "fragmentation_recovery_reap_failed" : reapError.c_str());
        trace_fragmentation_recovery(
            transaction,
            commandName,
            attempt,
            requestBytes,
            reapedBytes,
            0,
            0,
            false,
            "attempt",
            reapError.empty() ? "reap_failed" : reapError.c_str());
        managerState.fragmentationRecoveryFailures.fetch_add(1, std::memory_order_relaxed);
        outError = reapError.empty() ? "fragmentation recovery reap failed" : reapError;
        return false;
    }

    if (reapedBytes > 0) {
        managerState.retireReapPasses.fetch_add(1, std::memory_order_relaxed);
        managerState.retireReapBytes.fetch_add(reapedBytes, std::memory_order_relaxed);
    }
    trace_reap_pass(
        transaction,
        commandName,
        reapedBytes,
        true,
        (reapedBytes > 0) ? "fragmentation_recovery_reap" : "fragmentation_recovery_reap_no_progress");

    const std::uint64_t quarantineTrimmedEntries =
        trim_large_frame_quarantine_for_context(transaction.snapshot.deviceContextKey);
    const std::uint64_t graphEvictedEntries =
        evict_noncritical_graph_entries_for_context(transaction.snapshot.deviceContextKey);

    if (quarantineTrimmedEntries > 0) {
        managerState.fragmentationRecoveryQuarantineTrimmedEntries.fetch_add(
            quarantineTrimmedEntries,
            std::memory_order_relaxed);
    }
    if (graphEvictedEntries > 0) {
        managerState.fragmentationRecoveryGraphEvictedEntries.fetch_add(
            graphEvictedEntries,
            std::memory_order_relaxed);
    }

    if (captureMemorySnapshots) {
        maybe_publish_manager_memory_snapshot(resources, true);
    }

    trace_fragmentation_recovery(
        transaction,
        commandName,
        attempt,
        requestBytes,
        reapedBytes,
        quarantineTrimmedEntries,
        graphEvictedEntries,
        true,
        "attempt",
        "retry_once");
    return true;
}

} // namespace

bool query_submission_active(const SubmissionTransaction& transaction) noexcept {
    return transaction.active;
}

bool begin_submission(
    SubmissionTransaction& outTransaction,
    const SubmissionSnapshot& snapshot,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("begin_submission");
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected begin_submission";
        return false;
    }

    if (outTransaction.active) {
        outError = "submission transaction already active";
        return false;
    }

    ResourceManagerState& state = global_state();
    outTransaction.transactionId = state.nextTransactionId.fetch_add(1, std::memory_order_relaxed);
    if (outTransaction.transactionId == 0) {
        outTransaction.transactionId = state.nextTransactionId.fetch_add(1, std::memory_order_relaxed);
    }

    outTransaction.snapshot = snapshot;
    const std::uint32_t requestedTraceSchemaVersion = outTransaction.snapshot.traceSchemaVersion;
    if (!trace_schema_matches_contract(requestedTraceSchemaVersion)) {
        outError = "trace schema mismatch";
        telemetry_record_trace_schema_mismatch();
        telemetry_trace_schema_mismatch(
            outTransaction.transactionId,
            outTransaction.snapshot.snapshotId,
            requestedTraceSchemaVersion);
        return false;
    }

    std::uint64_t registryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
    std::uint64_t contextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
    if (registryGeneration == 0) {
        registryGeneration = 1;
    }
    if (contextEpoch == 0) {
        contextEpoch = 1;
    }
    outTransaction.snapshot.registryGeneration = registryGeneration;
    outTransaction.snapshot.contextEpoch = contextEpoch;
    outTransaction.snapshot.keySchemaVersion = std::max<std::uint32_t>(1u, outTransaction.snapshot.keySchemaVersion);
    outTransaction.snapshot.traceSchemaVersion = sanitize_trace_schema_version(outTransaction.snapshot.traceSchemaVersion);
    outTransaction.snapshot.keyDigests = normalize_key_digests(outTransaction.snapshot.keyDigests);

    std::uint64_t leaseGeneration = state.nextLeaseGeneration.fetch_add(1, std::memory_order_relaxed);
    if (leaseGeneration == 0) {
        leaseGeneration = state.nextLeaseGeneration.fetch_add(1, std::memory_order_relaxed);
    }
    outTransaction.leaseGeneration = leaseGeneration;
    outTransaction.active = true;
    outTransaction.committed = false;

    (void)registry_get_or_create(outTransaction.snapshot.deviceContextKey);
    if (!validate_lifecycle_for_stage(outTransaction, "begin", false, &outError)) {
        outTransaction.active = false;
        outTransaction.committed = false;
        return false;
    }
    if (!registry_note_submission_begin(outTransaction.snapshot.deviceContextKey)) {
        outError = "registry submission-begin tracking rejected";
        outTransaction.active = false;
        outTransaction.committed = false;
        return false;
    }
    telemetry_trace_schema_announcement(
        outTransaction.transactionId,
        outTransaction.snapshot.snapshotId,
        outTransaction.snapshot.traceSchemaVersion);
    telemetry_record_begin_submission();
    return true;
}

bool acquire_plan(
    SubmissionTransaction& transaction,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("acquire_plan");
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected acquire_plan";
        return false;
    }
    const std::uint64_t acquireId = telemetry_next_acquire_attempt_id();

    if (!validate_lifecycle_for_stage(transaction, "acquire", false, &outError)) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            false);
        return false;
    }

    {
        ResourceManagerState& state = global_state();
        StaleInput staleInput{};
        staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
        staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
        staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
        staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
        staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
        staleInput.observedLeaseGeneration = transaction.active ? transaction.leaseGeneration : 0;
        staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);
        const StaleDecision staleDecision = classify_stale_path(staleInput);
        telemetry_trace_stale_decision(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            "acquire",
            staleInput,
            staleDecision);
        if (staleDecision.hardStale || staleDecision.hardMiss) {
            const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
            telemetry_record_stale_tuple_hard_reject();
            outError = std::string("stale transaction in acquire path (reason=") +
                to_cstr(staleDecision.reason) + ")";
            telemetry_record_acquire_status(AcquireStatus::Error);
            for (ResourceKind kind : kResourceKindOrder) {
                telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
            }
            telemetry_trace_acquire(
                acquireId,
                transaction.transactionId,
                transaction.snapshot.snapshotId,
                transaction.snapshot.traceSchemaVersion,
                AcquireStatus::Error,
                errorPlan,
                false);
            return false;
        }
    }

    if (!transaction.active) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        outError = "submission transaction is not active";
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            false);
        return false;
    }

    SubmissionSnapshot& snapshot = transaction.snapshot;
    if (!trace_schema_matches_contract(snapshot.traceSchemaVersion)) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        outError = "trace schema mismatch";
        telemetry_record_trace_schema_mismatch();
        telemetry_trace_schema_mismatch(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion);
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            false);
        return false;
    }

    if (snapshot.keySchemaVersion == 0) {
        snapshot.keySchemaVersion = 1;
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "key_schema_version_zero_sanitized");
    }

    const KeyDigests rawDigests = snapshot.keyDigests;
    snapshot.keyDigests = normalize_key_digests(snapshot.keyDigests);
    if (rawDigests.uploadCoreHash != snapshot.keyDigests.uploadCoreHash ||
        rawDigests.dirHash != snapshot.keyDigests.dirHash ||
        rawDigests.scannerHash != snapshot.keyDigests.scannerHash ||
        rawDigests.autoExposureHash != snapshot.keyDigests.autoExposureHash) {
        telemetry_trace_key_normalization(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            rawDigests,
            snapshot.keyDigests);
    }

    std::uint64_t expectedSnapshotIdForFrame = 0;
    bool frameSnapshotMismatch = false;
    {
        const FrameSnapshotKey frameKey{ snapshot.instanceToken.value, snapshot.deviceContextKey };
        FrameSnapshotState& state = frame_snapshot_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        FrameSnapshotEntry& entry = state.bySubmissionKey[frameKey];
        const bool sameFrameToken = entry.valid && entry.frameToken == snapshot.frameToken.value;
        const bool sameSchema = entry.valid && entry.keySchemaVersion == snapshot.keySchemaVersion;
        const bool sameDigests = entry.valid && key_digests_equal(entry.digests, snapshot.keyDigests);
        if (sameFrameToken && sameSchema && sameDigests) {
            if (entry.snapshotId == 0) {
                entry.snapshotId = snapshot.snapshotId;
            }
            else if (entry.snapshotId != snapshot.snapshotId) {
                frameSnapshotMismatch = true;
                expectedSnapshotIdForFrame = entry.snapshotId;
            }
        }
        else {
            entry.valid = true;
            entry.frameToken = snapshot.frameToken.value;
            entry.digests = snapshot.keyDigests;
            entry.keySchemaVersion = snapshot.keySchemaVersion;
            entry.snapshotId = snapshot.snapshotId;
        }
    }
    if (frameSnapshotMismatch) {
        telemetry_record_frame_snapshot_mismatch();
        telemetry_trace_frame_snapshot_mismatch(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            snapshot.frameToken.value,
            expectedSnapshotIdForFrame,
            snapshot.snapshotId);
    }

    const ShadowHistoryKey key{ snapshot.instanceToken.value, snapshot.deviceContextKey };
    ShadowHistoryEntry previous{};
    bool hasPrevious = false;
    {
        ShadowHistoryState& state = shadow_history_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.bySubmissionKey.find(key);
        if (it != state.bySubmissionKey.end() && it->second.valid) {
            previous = it->second;
            hasPrevious = true;
        }
    }

    ShadowKeyDelta delta{};
    delta.hasPrevious = hasPrevious;
    delta.keySchemaChanged = !hasPrevious || (previous.keySchemaVersion != snapshot.keySchemaVersion);
    delta.uploadCoreChanged = !hasPrevious || (previous.digests.uploadCoreHash != snapshot.keyDigests.uploadCoreHash);
    delta.dirChanged = !hasPrevious || (previous.digests.dirHash != snapshot.keyDigests.dirHash);
    delta.scannerChanged = !hasPrevious || (previous.digests.scannerHash != snapshot.keyDigests.scannerHash);
    delta.autoExposureChanged = !hasPrevious || (previous.digests.autoExposureHash != snapshot.keyDigests.autoExposureHash);

    const ResourcePlan plan = build_shadow_resource_plan(delta);

    if (!validate_resource_kind_onboarding_contract(outError)) {
        const ResourcePlan errorPlan = make_uniform_resource_plan(AcquireStatus::Error);
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "resource_kind_onboarding_contract_invalid");
        telemetry_record_acquire_status(AcquireStatus::Error);
        for (ResourceKind kind : kResourceKindOrder) {
            telemetry_record_acquire_status_for_kind(kind, AcquireStatus::Error);
        }
        telemetry_trace_acquire(
            acquireId,
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            AcquireStatus::Error,
            errorPlan,
            hasPrevious);
        return false;
    }

    for (ResourceKind kind : kResourceKindOrder) {
        const ResourcePlanEntry& entry = resource_plan_entry(plan, kind);
        const ResourceKindContractEntry& contract = resource_kind_contract_entry(kind);
        const bool laneChanged = shadow_key_changed_for_kind(delta, kind);
        const std::uint64_t previousHash = hasPrevious ? key_digest_for_kind(previous.digests, kind) : 0;
        const std::uint64_t currentHash = key_digest_for_kind(snapshot.keyDigests, kind);

        if (entry.invalidated) {
            telemetry_trace_invalidation(
                transaction.transactionId,
                snapshot.snapshotId,
                snapshot.traceSchemaVersion,
                contract.invalidationLane,
                delta.keySchemaChanged ? "key_schema_changed" : (laneChanged ? "lane_hash_changed" : "policy_invalidated"),
                previousHash,
                currentHash);
        }

        if (delta.keySchemaChanged || laneChanged) {
            telemetry_trace_dag_edge(
                transaction.transactionId,
                snapshot.snapshotId,
                snapshot.traceSchemaVersion,
                contract.invalidationLane,
                contract.resourceNode,
                true,
                "allowed_lane_invalidation");
        }

        telemetry_record_acquire_status_for_kind(kind, entry.acquire.status);
    }

    bool forbiddenEdgeDetected = false;
    const ResourceKindContractEntry& scannerContract = resource_kind_contract_entry(ResourceKind::Scanner);
    const ResourceKindContractEntry& dirContract = resource_kind_contract_entry(ResourceKind::Dir);
    const ResourceKindContractEntry& uploadContract = resource_kind_contract_entry(ResourceKind::UploadCore);
    if (delta.scannerChanged &&
        !delta.uploadCoreChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::UploadCore).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            scannerContract.invalidationLane,
            uploadContract.resourceNode,
            false,
            "forbidden_edge_scanner_to_upload");
    }
    if (delta.scannerChanged &&
        !delta.dirChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::Dir).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            scannerContract.invalidationLane,
            dirContract.resourceNode,
            false,
            "forbidden_edge_scanner_to_dir");
    }
    if (delta.dirChanged &&
        !delta.uploadCoreChanged &&
        !delta.keySchemaChanged &&
        resource_plan_entry(plan, ResourceKind::UploadCore).invalidated) {
        forbiddenEdgeDetected = true;
        telemetry_record_forbidden_invalidation_edge();
        telemetry_trace_dag_edge(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            dirContract.invalidationLane,
            uploadContract.resourceNode,
            false,
            "forbidden_edge_dir_to_upload");
    }

    if (forbiddenEdgeDetected) {
        telemetry_record_module_boundary_violation();
        telemetry_trace_module_boundary_violation(
            transaction.transactionId,
            snapshot.snapshotId,
            snapshot.traceSchemaVersion,
            "forbidden_invalidation_edge_detected");
    }

    {
        ShadowHistoryState& state = shadow_history_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        ShadowHistoryEntry& entry = state.bySubmissionKey[key];
        entry.valid = true;
        entry.digests = snapshot.keyDigests;
        entry.keySchemaVersion = snapshot.keySchemaVersion;
        entry.snapshotId = snapshot.snapshotId;
    }

    const AcquireStatus finalStatus = combine_status(plan);

    telemetry_record_acquire_status(finalStatus);
    telemetry_trace_acquire(
        acquireId,
        transaction.transactionId,
        snapshot.snapshotId,
        snapshot.traceSchemaVersion,
        finalStatus,
        plan,
        hasPrevious);
    telemetry_record_acquire_plan();
    return true;
}

bool commit_submission(
    SubmissionTransaction& transaction,
    void* cudaStreamOpaque,
    std::string& outError) {
    (void)cudaStreamOpaque;
    outError.clear();
    MetadataMutationGuard mutationGuard("commit_submission");
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected commit_submission";
        return false;
    }
    if (!validate_lifecycle_for_stage(transaction, "commit", false, &outError)) {
        return false;
    }
    {
        ResourceManagerState& state = global_state();
        StaleInput staleInput{};
        staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
        staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
        staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
        staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
        staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
        staleInput.observedLeaseGeneration = transaction.active ? transaction.leaseGeneration : 0;
        staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);
        const StaleDecision staleDecision = classify_stale_path(staleInput);
        telemetry_trace_stale_decision(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            "commit",
            staleInput,
            staleDecision);
        if (staleDecision.hardStale || staleDecision.hardMiss) {
            telemetry_record_stale_tuple_hard_reject();
            outError = std::string("stale transaction in commit path (reason=") +
                to_cstr(staleDecision.reason) + ")";
            return false;
        }
    }
    if (!transaction.active) {
        outError = "submission transaction is not active";
        return false;
    }
    (void)registry_note_submission_end(transaction.snapshot.deviceContextKey);
    transaction.committed = true;
    transaction.active = false;
    telemetry_record_commit_submission();
    return true;
}

bool command_freeze_drain_bump_resume(
    const DeviceContextKey& key,
    const char* reason,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard("command_freeze_drain_bump_resume");
    if (!mutationGuard.ok()) {
        outError = "metadata mutation guard rejected command_freeze_drain_bump_resume";
        return false;
    }
    if (!registry_freeze_drain_bump_resume(key, reason)) {
        outError = "freeze-drain-bump-resume barrier rejected";
        return false;
    }
    return true;
}

namespace {
bool command_retire_context_with_reason(
    const DeviceContextKey& key,
    RegistryRetireReason reason,
    const char* commandName,
    std::string& outError) {
    outError.clear();
    MetadataMutationGuard mutationGuard(commandName ? commandName : "command_retire_context");
    if (!mutationGuard.ok()) {
        outError = std::string("metadata mutation guard rejected ")
            + (commandName ? commandName : "command_retire_context");
        return false;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    retire_base_graph_cache_for_context(key);
#endif

    RegistryHandle handle{};
    if (!registry_get(key, handle) || handle.value == 0) {
        return true;
    }

    registry_retire(handle, reason);
    return true;
}
} // namespace

bool command_retire_context_reset(
    const DeviceContextKey& key,
    std::string& outError) {
    return command_retire_context_with_reason(
        key,
        RegistryRetireReason::ContextReset,
        "command_retire_context_reset",
        outError);
}

bool command_retire_context_idle(
    const DeviceContextKey& key,
    std::string& outError) {
    return command_retire_context_with_reason(
        key,
        RegistryRetireReason::Idle,
        "command_retire_context_idle",
        outError);
}

bool command_ensure_uploaded(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_uploaded")) {
        return false;
    }
    const std::uint64_t uploadRequestBytes =
        estimate_upload_core_request_bytes(resources, ws);
    if (uploadRequestBytes == std::numeric_limits<std::uint64_t>::max()) {
        outError = "upload reservation request byte estimation overflow (core)";
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_uploaded",
            PressureLane::Upload,
            saturating_u64_to_size_t(uploadRequestBytes),
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_uploaded",
                "pressure_pre_upload",
                reclaimError)) {
            outError = reclaimError.empty() ? "pressure pre-upload reclaim failed" : reclaimError;
            return false;
        }
    }
    UploadReservationClaim uploadClaim{};
    if (!acquire_upload_reservation_with_wait(
            transaction,
            "command_ensure_uploaded",
            uploadRequestBytes,
            true,
            uploadClaim,
            outError)) {
        return false;
    }
    UploadReservationGuard uploadGuard(std::move(uploadClaim));
    const bool ok = JuicerCuda::ensure_uploaded(resources, ws, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_scan_lut(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    bool negativeMedium,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_scan_lut")) {
        return false;
    }
    const std::uint64_t uploadRequestBytes =
        estimate_scan_lut_upload_bytes(resources, ws, negativeMedium);
    if (uploadRequestBytes == std::numeric_limits<std::uint64_t>::max()) {
        outError = "upload reservation request byte estimation overflow (scan LUT)";
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_scan_lut",
            PressureLane::Upload,
            saturating_u64_to_size_t(uploadRequestBytes),
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_scan_lut",
                "pressure_pre_upload",
                reclaimError)) {
            outError = reclaimError.empty() ? "pressure pre-upload reclaim failed" : reclaimError;
            return false;
        }
    }
    UploadReservationClaim uploadClaim{};
    if (!acquire_upload_reservation_with_wait(
            transaction,
            "command_ensure_scan_lut",
            uploadRequestBytes,
            true,
            uploadClaim,
            outError)) {
        return false;
    }
    UploadReservationGuard uploadGuard(std::move(uploadClaim));
    BuilderReservationClaim builderClaim{};
    if (!acquire_builder_reservation_with_wait(
            transaction,
            "command_ensure_scan_lut",
            BuilderReservationTier::Lut,
            uploadRequestBytes,
            true,
            builderClaim,
            outError)) {
        return false;
    }
    BuilderReservationGuard builderGuard(std::move(builderClaim));
    const bool ok = JuicerCuda::ensure_scan_lut(resources, ws, negativeMedium, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_scan_error_flag(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_scan_error_flag")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = JuicerCuda::ensure_scan_error_flag(resources, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_print_illuminant_filtered(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    const WorkingState& ws,
    const Print::Runtime& prt,
    const Print::Params& params,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_print_illuminant_filtered")) {
        return false;
    }
    const std::uint64_t uploadRequestBytes =
        estimate_print_illuminant_upload_bytes(resources, ws, prt, params);
    if (uploadRequestBytes == std::numeric_limits<std::uint64_t>::max()) {
        outError = "upload reservation request byte estimation overflow (print illuminant)";
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_print_illuminant_filtered",
            PressureLane::Upload,
            saturating_u64_to_size_t(uploadRequestBytes),
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_print_illuminant_filtered",
                "pressure_pre_upload",
                reclaimError)) {
            outError = reclaimError.empty() ? "pressure pre-upload reclaim failed" : reclaimError;
            return false;
        }
    }
    UploadReservationClaim uploadClaim{};
    if (!acquire_upload_reservation_with_wait(
            transaction,
            "command_ensure_print_illuminant_filtered",
            uploadRequestBytes,
            true,
            uploadClaim,
            outError)) {
        return false;
    }
    UploadReservationGuard uploadGuard(std::move(uploadClaim));
    const bool ok =
        JuicerCuda::ensure_print_illuminant_filtered(resources, ws, prt, params, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_optics_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int width,
    int height,
    bool needBlurredScratch,
    bool needAuxScratch,
    bool needGrainScratch,
    bool needGrainSharedScratch,
    bool needGateMask,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_optics_scratch")) {
        return false;
    }
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(cfg);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::size_t growthBytes = estimate_optics_growth_bytes(
        resources,
        width,
        height,
        needBlurredScratch,
        needAuxScratch,
        needGrainScratch,
        needGrainSharedScratch,
        needGateMask);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "scratch growth byte estimation overflow";
        return false;
    }

    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_optics_scratch",
            PressureLane::Builder,
            growthBytes,
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_optics_scratch",
                "pressure_pre_growth",
                reclaimError)) {
            outError = reclaimError.empty() ? "pressure pre-growth reclaim failed" : reclaimError;
            return false;
        }
    }

    BuilderReservationClaim builderClaim{};
    if (!acquire_builder_reservation_with_wait(
            transaction,
            "command_ensure_optics_scratch",
            BuilderReservationTier::Scratch,
            static_cast<std::uint64_t>(growthBytes),
            true,
            builderClaim,
            outError)) {
        return false;
    }
    BuilderReservationGuard builderGuard(std::move(builderClaim));

    ScratchPolicyClaim scratchClaim{};
    if (!acquire_scratch_policy_claim_with_wait(
            transaction,
            "command_ensure_optics_scratch",
            ScratchWorkClass::Optics,
            width,
            height,
            growthBytes,
            true,
            scratchClaim,
            outError)) {
        return false;
    }
    ScratchPolicyGuard scratchGuard(std::move(scratchClaim));

    ResourceManagerState& managerState = global_state();
    std::uint32_t attempts = 0;
    bool fragmentationRecoveryTriggered = false;
    bool fragmentationRecoveryPendingOutcome = false;
    auto finalizeFragmentationOutcome = [&](bool success, const char* reason) {
        if (!fragmentationRecoveryPendingOutcome) {
            return;
        }
        if (success) {
            managerState.fragmentationRecoverySuccess.fetch_add(1, std::memory_order_relaxed);
        }
        else {
            managerState.fragmentationRecoveryFailures.fetch_add(1, std::memory_order_relaxed);
        }
        trace_fragmentation_recovery(
            transaction,
            "command_ensure_optics_scratch",
            attempts,
            growthBytes,
            0,
            0,
            0,
            success,
            "outcome",
            reason);
        fragmentationRecoveryPendingOutcome = false;
    };

    while (true) {
        outError.clear();
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        if (JuicerCuda::ensure_optics_scratch(
                resources,
                width,
                height,
                needBlurredScratch,
                needAuxScratch,
                needGrainScratch,
                needGrainSharedScratch,
                needGateMask,
                cudaStreamOpaque,
                outError)) {
            maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
            if (attempts > 0) {
                managerState.budgetReclaimRetrySuccess.fetch_add(1, std::memory_order_relaxed);
            }
            finalizeFragmentationOutcome(true, "allocation_retry_success");
            return true;
        }
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        const bool allocatorOom = is_allocator_oom_error(outError);
        if (!allocatorOom) {
            finalizeFragmentationOutcome(false, "non_allocator_error");
            return false;
        }
        if (attempts >= cfg.reclaimRetryMaxAttempts) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        "command_ensure_optics_scratch",
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        "command_ensure_optics_scratch",
                        growthBytes);
                    managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                "command_ensure_optics_scratch",
                growthBytes);
            managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
            finalizeFragmentationOutcome(false, "allocator_oom_final");
            return false;
        }

        ++attempts;
        managerState.budgetReclaimRetryAttempts.fetch_add(1, std::memory_order_relaxed);
        std::size_t reclaimedBytes = 0;
        std::string reclaimError;
        if (!JuicerCuda::reap_retired_allocations(resources, reclaimedBytes, reclaimError)) {
            trace_reap_pass(
                transaction,
                "command_ensure_optics_scratch",
                reclaimedBytes,
                false,
                reclaimError.empty() ? "reap_failed" : reclaimError.c_str());
            trace_budget_reclaim_retry(
                transaction,
                "command_ensure_optics_scratch",
                attempts,
                reclaimedBytes,
                false,
                reclaimError.empty() ? "reap_failed" : reclaimError.c_str());
            if (!reclaimError.empty()) {
                outError += " | reclaim_retry_failed: " + reclaimError;
            }
            managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
            finalizeFragmentationOutcome(false, "reap_retry_failed");
            return false;
        }

        if (reclaimedBytes > 0) {
            managerState.retireReapPasses.fetch_add(1, std::memory_order_relaxed);
            managerState.retireReapBytes.fetch_add(reclaimedBytes, std::memory_order_relaxed);
        }
        trace_reap_pass(
            transaction,
            "command_ensure_optics_scratch",
            reclaimedBytes,
            true,
            (reclaimedBytes > 0) ? "retry_after_reap" : "reap_no_progress");
        trace_budget_reclaim_retry(
            transaction,
            "command_ensure_optics_scratch",
            attempts,
            reclaimedBytes,
            true,
            (reclaimedBytes > 0) ? "retry_after_reap" : "reap_no_progress");
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        if (reclaimedBytes == 0) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        "command_ensure_optics_scratch",
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        "command_ensure_optics_scratch",
                        growthBytes);
                    managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                "command_ensure_optics_scratch",
                growthBytes);
            managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
            finalizeFragmentationOutcome(false, "reap_no_progress");
            return false;
        }
    }
}

bool command_ensure_spatial_dir_scratch(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int width,
    int height,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_spatial_dir_scratch")) {
        return false;
    }
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(cfg);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::size_t growthBytes = estimate_spatial_dir_growth_bytes(resources, width, height);
    if (growthBytes == std::numeric_limits<std::size_t>::max()) {
        outError = "spatial dir scratch growth byte estimation overflow";
        return false;
    }

    bool requestPreReclaim = false;
    if (!enforce_pressure_gate(
            transaction,
            resources,
            "command_ensure_spatial_dir_scratch",
            PressureLane::Builder,
            growthBytes,
            true,
            requestPreReclaim,
            outError)) {
        return false;
    }
    if (requestPreReclaim) {
        std::string reclaimError;
        if (!run_reap_pass_for_pressure(
                transaction,
                resources,
                "command_ensure_spatial_dir_scratch",
                "pressure_pre_growth",
                reclaimError)) {
            outError = reclaimError.empty() ? "pressure pre-growth reclaim failed" : reclaimError;
            return false;
        }
    }

    BuilderReservationClaim builderClaim{};
    if (!acquire_builder_reservation_with_wait(
            transaction,
            "command_ensure_spatial_dir_scratch",
            BuilderReservationTier::Scratch,
            static_cast<std::uint64_t>(growthBytes),
            true,
            builderClaim,
            outError)) {
        return false;
    }
    BuilderReservationGuard builderGuard(std::move(builderClaim));

    ScratchPolicyClaim scratchClaim{};
    if (!acquire_scratch_policy_claim_with_wait(
            transaction,
            "command_ensure_spatial_dir_scratch",
            ScratchWorkClass::SpatialDir,
            width,
            height,
            growthBytes,
            true,
            scratchClaim,
            outError)) {
        return false;
    }
    ScratchPolicyGuard scratchGuard(std::move(scratchClaim));

    ResourceManagerState& managerState = global_state();
    std::uint32_t attempts = 0;
    bool fragmentationRecoveryTriggered = false;
    bool fragmentationRecoveryPendingOutcome = false;
    auto finalizeFragmentationOutcome = [&](bool success, const char* reason) {
        if (!fragmentationRecoveryPendingOutcome) {
            return;
        }
        if (success) {
            managerState.fragmentationRecoverySuccess.fetch_add(1, std::memory_order_relaxed);
        }
        else {
            managerState.fragmentationRecoveryFailures.fetch_add(1, std::memory_order_relaxed);
        }
        trace_fragmentation_recovery(
            transaction,
            "command_ensure_spatial_dir_scratch",
            attempts,
            growthBytes,
            0,
            0,
            0,
            success,
            "outcome",
            reason);
        fragmentationRecoveryPendingOutcome = false;
    };

    while (true) {
        outError.clear();
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        if (JuicerCuda::ensure_spatial_dir_scratch(resources, width, height, cudaStreamOpaque, outError)) {
            maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
            if (attempts > 0) {
                managerState.budgetReclaimRetrySuccess.fetch_add(1, std::memory_order_relaxed);
            }
            finalizeFragmentationOutcome(true, "allocation_retry_success");
            return true;
        }
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        const bool allocatorOom = is_allocator_oom_error(outError);
        if (!allocatorOom) {
            finalizeFragmentationOutcome(false, "non_allocator_error");
            return false;
        }
        if (attempts >= cfg.reclaimRetryMaxAttempts) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        "command_ensure_spatial_dir_scratch",
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        "command_ensure_spatial_dir_scratch",
                        growthBytes);
                    managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                "command_ensure_spatial_dir_scratch",
                growthBytes);
            managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
            finalizeFragmentationOutcome(false, "allocator_oom_final");
            return false;
        }

        ++attempts;
        managerState.budgetReclaimRetryAttempts.fetch_add(1, std::memory_order_relaxed);
        std::size_t reclaimedBytes = 0;
        std::string reclaimError;
        if (!JuicerCuda::reap_retired_allocations(resources, reclaimedBytes, reclaimError)) {
            trace_reap_pass(
                transaction,
                "command_ensure_spatial_dir_scratch",
                reclaimedBytes,
                false,
                reclaimError.empty() ? "reap_failed" : reclaimError.c_str());
            trace_budget_reclaim_retry(
                transaction,
                "command_ensure_spatial_dir_scratch",
                attempts,
                reclaimedBytes,
                false,
                reclaimError.empty() ? "reap_failed" : reclaimError.c_str());
            if (!reclaimError.empty()) {
                outError += " | reclaim_retry_failed: " + reclaimError;
            }
            managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
            finalizeFragmentationOutcome(false, "reap_retry_failed");
            return false;
        }

        if (reclaimedBytes > 0) {
            managerState.retireReapPasses.fetch_add(1, std::memory_order_relaxed);
            managerState.retireReapBytes.fetch_add(reclaimedBytes, std::memory_order_relaxed);
        }
        trace_reap_pass(
            transaction,
            "command_ensure_spatial_dir_scratch",
            reclaimedBytes,
            true,
            (reclaimedBytes > 0) ? "retry_after_reap" : "reap_no_progress");
        trace_budget_reclaim_retry(
            transaction,
            "command_ensure_spatial_dir_scratch",
            attempts,
            reclaimedBytes,
            true,
            (reclaimedBytes > 0) ? "retry_after_reap" : "reap_no_progress");
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

        if (reclaimedBytes == 0) {
            if (cfg.fragmentationRecoveryEnabled && !fragmentationRecoveryTriggered) {
                std::string recoveryError;
                if (!run_fragmentation_recovery_once(
                        transaction,
                        resources,
                        "command_ensure_spatial_dir_scratch",
                        growthBytes,
                        attempts,
                        captureMemorySnapshots,
                        recoveryError)) {
                    if (!recoveryError.empty()) {
                        outError += " | fragmentation_recovery_failed: " + recoveryError;
                    }
                    record_allocator_oom_headroom_observation(
                        transaction,
                        "command_ensure_spatial_dir_scratch",
                        growthBytes);
                    managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                fragmentationRecoveryTriggered = true;
                fragmentationRecoveryPendingOutcome = true;
                continue;
            }
            record_allocator_oom_headroom_observation(
                transaction,
                "command_ensure_spatial_dir_scratch",
                growthBytes);
            managerState.budgetAllocatorOomEvents.fetch_add(1, std::memory_order_relaxed);
            finalizeFragmentationOutcome(false, "reap_no_progress");
            return false;
        }
    }
}

bool command_ensure_spatial_dir_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_spatial_dir_kernel")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = JuicerCuda::ensure_spatial_dir_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_gaussian_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_gaussian_kernel")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = JuicerCuda::ensure_gaussian_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_halation_kernel(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    JuicerCuda::Resources::DeviceGaussianKernel& kernel,
    float sigma,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_halation_kernel")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const bool ok = JuicerCuda::ensure_halation_kernel(resources, kernel, sigma, cudaStreamOpaque, outError);
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    return ok;
}

bool command_ensure_auto_exposure_buffers(
    SubmissionTransaction& transaction,
    JuicerCuda::Resources& resources,
    int meterWidth,
    int meterHeight,
    std::uint64_t autoExposureKeyHash,
    void* cudaStreamOpaque,
    std::string& outError) {
    if (!ensure_active_for_command(transaction, outError, "command_ensure_auto_exposure_buffers")) {
        return false;
    }
    const bool captureMemorySnapshots = should_collect_manager_memory_snapshots(manager_effective_config());
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
    const std::uint64_t normalizedKeyHash = normalize_key_u64(autoExposureKeyHash);

    const ShadowHistoryKey ownershipKey{
        transaction.snapshot.instanceToken.value,
        transaction.snapshot.deviceContextKey
    };

    bool hadPrevious = false;
    bool metadataHit = false;
    {
        AutoExposureOwnershipState& ownershipState = auto_exposure_ownership_state();
        std::lock_guard<std::mutex> lock(ownershipState.mutex);
        auto it = ownershipState.bySubmissionKey.find(ownershipKey);
        if (it != ownershipState.bySubmissionKey.end() && it->second.valid) {
            hadPrevious = true;
            const AutoExposureOwnershipEntry& previous = it->second;
            metadataHit =
                previous.keySchemaVersion == transaction.snapshot.keySchemaVersion &&
                previous.keyHash == normalizedKeyHash &&
                previous.meterWidth == meterWidth &&
                previous.meterHeight == meterHeight;
        }
    }

    telemetry_trace_auto_exposure_ownership(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "ManagerOnly",
        "acquire",
        metadataHit,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        hadPrevious,
        metadataHit ? "metadata_hit" : "metadata_miss");

    if (!JuicerCuda::ensure_auto_exposure_buffers(resources, meterWidth, meterHeight, cudaStreamOpaque, outError)) {
        maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);
        telemetry_trace_auto_exposure_ownership(
            transaction.transactionId,
            transaction.snapshot.snapshotId,
            transaction.snapshot.traceSchemaVersion,
            "ManagerOnly",
            "ensure_fail",
            false,
            normalizedKeyHash,
            meterWidth,
            meterHeight,
            hadPrevious,
            outError.empty() ? "ensure_failed" : outError.c_str());
        return false;
    }
    maybe_publish_manager_memory_snapshot(resources, captureMemorySnapshots);

    {
        AutoExposureOwnershipState& ownershipState = auto_exposure_ownership_state();
        std::lock_guard<std::mutex> lock(ownershipState.mutex);
        AutoExposureOwnershipEntry& entry = ownershipState.bySubmissionKey[ownershipKey];
        entry.valid = true;
        entry.keyHash = normalizedKeyHash;
        entry.meterWidth = meterWidth;
        entry.meterHeight = meterHeight;
        entry.keySchemaVersion = transaction.snapshot.keySchemaVersion;
    }

    telemetry_trace_auto_exposure_ownership(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "ManagerOnly",
        "publish",
        metadataHit,
        normalizedKeyHash,
        meterWidth,
        meterHeight,
        hadPrevious,
        metadataHit ? "reuse" : "refresh");
    return true;
}

bool command_launch_base_pipeline_graph(
    SubmissionTransaction& transaction,
    JuicerCuda::PipelineRunParams& run,
    int renderModeKey,
    void* cudaStreamOpaque,
    int& outCudaErrorCode,
    std::string& outError) {
    outError.clear();

    if (!ensure_active_for_command(transaction, outError, "command_launch_base_pipeline_graph")) {
        return false;
    }

#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
    (void)run;
    (void)renderModeKey;
    (void)cudaStreamOpaque;
    outCudaErrorCode = 0;
    outError = "CUDA is not enabled";
    return false;
#else
    outCudaErrorCode = static_cast<int>(cudaErrorUnknown);
    BasePipelineLaunchFn launchFn = base_pipeline_launch_fn_for_mode(renderModeKey);
    if (!launchFn) {
        outCudaErrorCode = static_cast<int>(cudaErrorInvalidValue);
        return true;
    }

    const cudaStream_t stream = cudaStreamOpaque
        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
        : nullptr;

    BaseGraphKey key{};
    key.width = run.width;
    key.height = run.height;
    key.nComponents = run.nComponents;
    key.renderMode = renderModeKey;
    const std::uint64_t keyDigest = base_graph_key_digest(key);
    const std::uint64_t requestBytes = estimate_base_graph_request_bytes(key);
    const ResourceManagerConfigEffective& cfg = manager_effective_config();
    constexpr bool kGraphAdmissionCriticalCurrentFrame = false;

    std::shared_ptr<BaseGraphBucketState> bucketPtr =
        get_or_create_base_graph_bucket(transaction.snapshot.deviceContextKey);
    if (!bucketPtr) {
        outCudaErrorCode = static_cast<int>(cudaErrorUnknown);
        outError = "base graph bucket unavailable";
        return false;
    }
    std::lock_guard<std::mutex> bucketLock(bucketPtr->mutex);
    BaseGraphBucketState& bucket = *bucketPtr;
    std::uint64_t activeEpoch = transaction.snapshot.contextEpoch;
    if (activeEpoch == 0) {
        activeEpoch = 1;
    }
    if (bucket.contextEpoch != activeEpoch) {
        clear_base_graph_bucket(bucket);
        bucket.contextEpoch = activeEpoch;
    }

    bucket.useTick++;
    if (bucket.useTick == 0) {
        bucket.useTick = 1;
    }
    const std::uint64_t useTick = bucket.useTick;

    BaseGraphEntry* found = find_base_graph_entry(bucket, key);
    if (!found) {
        std::uint32_t observedProbationHits = 0;
        auto probationIt = bucket.probationHitsByDigest.find(keyDigest);
        if (probationIt != bucket.probationHitsByDigest.end()) {
            observedProbationHits = probationIt->second;
        }

        CacheAdmissionInput admissionInput{};
        admissionInput.requestBytes = requestBytes;
        admissionInput.cacheTargetBytes = cfg.managerSoftTargetBytes;
        admissionInput.maxCacheableEntryBytes = cfg.maxCacheableEntryBytes;
        admissionInput.maxCacheableEntryPctOfTarget = cfg.maxCacheableEntryPctOfTarget;
        admissionInput.largeEntryProbationThresholdBytes = cfg.largeEntryProbationThresholdBytes;
        admissionInput.largeEntryProbationHitsRequired = cfg.largeEntryProbationHitsRequired;
        admissionInput.observedProbationHits = observedProbationHits;
        admissionInput.criticalCurrentFrame = kGraphAdmissionCriticalCurrentFrame;
        const CacheAdmissionDecision admissionDecision = classify_cache_admission(admissionInput);
        trace_cache_admission_decision(
            transaction,
            "command_launch_base_pipeline_graph",
            admissionDecision,
            kGraphAdmissionCriticalCurrentFrame,
            requestBytes,
            observedProbationHits,
            keyDigest,
            "graph_miss");

        ResourceManagerState& managerState = global_state();
        if (admissionDecision.admissionClass == CacheAdmissionClass::TooLargeToCache) {
            managerState.cacheAdmissionTooLargeEvents.fetch_add(1, std::memory_order_relaxed);
        }
        if (admissionDecision.probationApplied) {
            const std::uint32_t nextObservedHits =
                (observedProbationHits < std::numeric_limits<std::uint32_t>::max())
                ? (observedProbationHits + 1u)
                : std::numeric_limits<std::uint32_t>::max();
            trace_probation_decision(
                transaction,
                "command_launch_base_pipeline_graph",
                keyDigest,
                nextObservedHits,
                admissionDecision.probationHitsRequired,
                admissionDecision.allowDurableAdmission,
                admissionDecision.reason);
            if (admissionDecision.allowDurableAdmission) {
                managerState.cacheAdmissionProbationAdmitEvents.fetch_add(1, std::memory_order_relaxed);
            }
            else {
                managerState.cacheAdmissionProbationDeferredEvents.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (admissionDecision.reason &&
            std::string_view(admissionDecision.reason).find("critical_override") != std::string_view::npos) {
            managerState.cacheAdmissionCriticalOverrideEvents.fetch_add(1, std::memory_order_relaxed);
        }

        if (!admissionDecision.allowDurableAdmission) {
            if (admissionDecision.probationApplied) {
                std::uint32_t& probationHits = bucket.probationHitsByDigest[keyDigest];
                if (probationHits < std::numeric_limits<std::uint32_t>::max()) {
                    ++probationHits;
                }
            }
            else {
                bucket.probationHitsByDigest.erase(keyDigest);
            }
            managerState.graphNonResidentServeEvents.fetch_add(1, std::memory_order_relaxed);
            outCudaErrorCode = static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
            return true;
        }

        BuilderReservationClaim builderClaim{};
        ReservationAttemptInfo builderReservation{};
        managerState.builderReservationRequests.fetch_add(1, std::memory_order_relaxed);
        if (!try_acquire_builder_reservation_claim(
                transaction,
                BuilderReservationTier::Graph,
                requestBytes,
                kGraphAdmissionCriticalCurrentFrame,
                builderClaim,
                builderReservation)) {
            const ReservationDecision& decision = builderReservation.decision;
            const bool fairnessDeferred =
                (decision.reason && std::string_view(decision.reason) == "fairness_tokens_exhausted");
            if (decision.shouldWait) {
                managerState.builderReservationDeferred.fetch_add(1, std::memory_order_relaxed);
                if (fairnessDeferred) {
                    managerState.builderFairnessTokenDeferred.fetch_add(1, std::memory_order_relaxed);
                }
                trace_builder_reservation_decision(
                    transaction,
                    "command_launch_base_pipeline_graph",
                    BuilderReservationTier::Graph,
                    requestBytes,
                    builderReservation.bytesInFlight,
                    builderReservation.capBytes,
                    builderReservation.thresholdBytes,
                    decision,
                    kGraphAdmissionCriticalCurrentFrame,
                    builderReservation.instanceToken,
                    builderReservation.sharedTokens,
                    builderReservation.criticalTokens,
                    0,
                    "deferred_nonresident");
            }
            else {
                managerState.builderReservationDenied.fetch_add(1, std::memory_order_relaxed);
                trace_builder_reservation_decision(
                    transaction,
                    "command_launch_base_pipeline_graph",
                    BuilderReservationTier::Graph,
                    requestBytes,
                    builderReservation.bytesInFlight,
                    builderReservation.capBytes,
                    builderReservation.thresholdBytes,
                    decision,
                    kGraphAdmissionCriticalCurrentFrame,
                    builderReservation.instanceToken,
                    builderReservation.sharedTokens,
                    builderReservation.criticalTokens,
                    0,
                    "denied_nonresident");
            }
            managerState.graphNonResidentServeEvents.fetch_add(1, std::memory_order_relaxed);
            outCudaErrorCode = static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
            return true;
        }
        managerState.builderReservationGranted.fetch_add(1, std::memory_order_relaxed);
        if (JTRACE_ENABLED(3)) {
            trace_builder_reservation_decision(
                transaction,
                "command_launch_base_pipeline_graph",
                BuilderReservationTier::Graph,
                requestBytes,
                builderReservation.bytesInFlight,
                builderReservation.capBytes,
                builderReservation.thresholdBytes,
                builderReservation.decision,
                kGraphAdmissionCriticalCurrentFrame,
                builderReservation.instanceToken,
                builderReservation.sharedTokens,
                builderReservation.criticalTokens,
                0,
                "admitted");
        }
        BuilderReservationGuard builderGuard(std::move(builderClaim));

        bucket.probationHitsByDigest.erase(keyDigest);
        found = build_base_graph_entry(bucket, key, launchFn, run, stream);
    }
    else {
        bucket.probationHitsByDigest.erase(keyDigest);
    }

    if (!found || !found->execOpaque || !found->kernelNodeOpaque) {
        outCudaErrorCode = static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
        return true;
    }

    found->lastUseTick = useTick;

    cudaGraphExec_t exec = reinterpret_cast<cudaGraphExec_t>(found->execOpaque);
    cudaGraphNode_t node = reinterpret_cast<cudaGraphNode_t>(found->kernelNodeOpaque);

    cudaKernelNodeParams nodeParams{};
    nodeParams.func = found->kernelFuncOpaque;
    nodeParams.gridDim = dim3(found->gridX, found->gridY, found->gridZ);
    nodeParams.blockDim = dim3(found->blockX, found->blockY, found->blockZ);
    nodeParams.sharedMemBytes = found->sharedMemBytes;
    void* kernelArgs[] = { &run };
    nodeParams.kernelParams = kernelArgs;
    nodeParams.extra = nullptr;
    cudaError_t setErr = cudaGraphExecKernelNodeSetParams(exec, node, &nodeParams);
    if (setErr != cudaSuccess) {
        destroy_base_graph_entry(*found);
        outCudaErrorCode = static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
        return true;
    }

    cudaError_t runErr = cudaGraphLaunch(exec, stream);
    if (runErr == cudaSuccess) {
        runErr = cudaGetLastError();
    }
    if (runErr != cudaSuccess) {
        destroy_base_graph_entry(*found);
        outCudaErrorCode = static_cast<int>(launchFn(&run, reinterpret_cast<void*>(stream)));
        return true;
    }

    outCudaErrorCode = static_cast<int>(cudaSuccess);
    return true;
#endif
}

bool error_is_scratch_exhausted(const std::string& error) noexcept {
    return error.rfind(kScratchExhaustedPrefix, 0) == 0 ||
        error.rfind(kReservationDeferredPrefix, 0) == 0;
}

void rollback_submission(
    SubmissionTransaction& transaction,
    const char* reason) noexcept {
    (void)reason;
    MetadataMutationGuard mutationGuard("rollback_submission");
    if (!mutationGuard.ok()) {
        return;
    }
    (void)validate_lifecycle_for_stage(transaction, "release", true, nullptr);
    ResourceManagerState& state = global_state();
    StaleInput staleInput{};
    staleInput.expectedRegistryGeneration = transaction.snapshot.registryGeneration;
    staleInput.observedRegistryGeneration = state.registryGeneration.load(std::memory_order_relaxed);
    staleInput.expectedContextEpoch = transaction.snapshot.contextEpoch;
    staleInput.observedContextEpoch = state.contextEpoch.load(std::memory_order_relaxed);
    staleInput.expectedLeaseGeneration = transaction.leaseGeneration;
    staleInput.observedLeaseGeneration = transaction.leaseGeneration;
    staleInput.keySchemaMismatch = (transaction.snapshot.keySchemaVersion == 0);
    const StaleDecision staleDecision = classify_stale_path(staleInput);
    telemetry_trace_stale_decision(
        transaction.transactionId,
        transaction.snapshot.snapshotId,
        transaction.snapshot.traceSchemaVersion,
        "release",
        staleInput,
        staleDecision);
    if (staleDecision.hardStale || staleDecision.hardMiss) {
        telemetry_record_stale_tuple_hard_reject();
    }
    if (!transaction.active) {
        return;
    }
    (void)registry_note_submission_end(transaction.snapshot.deviceContextKey);
    transaction.committed = false;
    transaction.active = false;
    telemetry_record_rollback_submission();
}

} // namespace ResourceManager
} // namespace JuicerCuda
