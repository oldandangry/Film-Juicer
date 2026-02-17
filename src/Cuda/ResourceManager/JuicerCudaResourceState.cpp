// Cuda/ResourceManager/JuicerCudaResourceState.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"

#include <mutex>
#include <unordered_map>

namespace JuicerCuda {
namespace ResourceManager {

namespace {

struct LatestSnapshotKey {
    std::uint64_t instanceToken = 0;
    DeviceContextKey deviceContextKey{};

    bool operator==(const LatestSnapshotKey& other) const noexcept {
        return instanceToken == other.instanceToken &&
            deviceContextKey == other.deviceContextKey;
    }
};

struct LatestSnapshotKeyHasher {
    std::size_t operator()(const LatestSnapshotKey& key) const noexcept {
        const std::size_t hInstance = std::hash<std::uint64_t>{}(key.instanceToken);
        const std::size_t hContext = DeviceContextKeyHash{}(key.deviceContextKey);
        return hInstance ^ (hContext + 0x9e3779b9u + (hInstance << 6u) + (hInstance >> 2u));
    }
};

struct LatestSnapshotState {
    std::mutex mutex;
    std::unordered_map<LatestSnapshotKey, std::uint64_t, LatestSnapshotKeyHasher> bySubmissionKey;
};

LatestSnapshotState& latest_snapshot_state() {
    static LatestSnapshotState state{};
    return state;
}

} // namespace

ResourceManagerState& global_state() noexcept {
    static ResourceManagerState state{};
    return state;
}

void state_record_acquire_status_for_kind(ResourceKind kind, AcquireStatus status) noexcept {
    ResourceManagerState& state = global_state();
    const std::size_t kindIndex = resource_kind_index(kind);
    if (kindIndex >= kResourceKindCount) {
        telemetry_record_module_boundary_violation();
        return;
    }
    ResourceKindAcquireCounters& counters = state.acquireStatusByKind[kindIndex];
    switch (status) {
    case AcquireStatus::Hit:
        counters.hit.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Miss:
        counters.miss.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Busy:
        counters.busy.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Exhausted:
        counters.exhausted.fetch_add(1, std::memory_order_relaxed);
        break;
    case AcquireStatus::Error:
    default:
        counters.error.fetch_add(1, std::memory_order_relaxed);
        break;
    }
}

void state_note_latest_snapshot(const SubmissionSnapshot& snapshot) noexcept {
    const std::uint64_t instanceToken = snapshot.instanceToken.value;
    if (instanceToken == 0 || snapshot.snapshotId == 0) {
        return;
    }

    const LatestSnapshotKey key{ instanceToken, snapshot.deviceContextKey };
    LatestSnapshotState& state = latest_snapshot_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::uint64_t& latest = state.bySubmissionKey[key];
    if (snapshot.snapshotId > latest) {
        latest = snapshot.snapshotId;
    }
}

bool state_snapshot_is_superseded(
    const SubmissionSnapshot& snapshot,
    std::uint64_t* outLatestSnapshotId) noexcept {
    if (outLatestSnapshotId) {
        *outLatestSnapshotId = 0;
    }

    const std::uint64_t instanceToken = snapshot.instanceToken.value;
    if (instanceToken == 0 || snapshot.snapshotId == 0) {
        return false;
    }

    const LatestSnapshotKey key{ instanceToken, snapshot.deviceContextKey };
    LatestSnapshotState& state = latest_snapshot_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.bySubmissionKey.find(key);
    if (it == state.bySubmissionKey.end()) {
        return false;
    }
    if (outLatestSnapshotId) {
        *outLatestSnapshotId = it->second;
    }
    return it->second > snapshot.snapshotId;
}

void state_clear_latest_snapshot_for_context(const DeviceContextKey& key) noexcept {
    LatestSnapshotState& state = latest_snapshot_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    for (auto it = state.bySubmissionKey.begin(); it != state.bySubmissionKey.end();) {
        if (it->first.deviceContextKey == key) {
            it = state.bySubmissionKey.erase(it);
        }
        else {
            ++it;
        }
    }
}

} // namespace ResourceManager
} // namespace JuicerCuda
