// Cuda/ResourceManager/JuicerCudaResourceState.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"

#include <mutex>

namespace JuicerCuda {
namespace ResourceManager {

namespace {

std::recursive_mutex& metadata_mutation_mutex() {
    static std::recursive_mutex m;
    return m;
}

std::uint64_t& metadata_last_issued_sequence() {
    static std::uint64_t value = 0;
    return value;
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

bool metadata_mutation_begin(const char* stage, MetadataMutationScope& outScope) noexcept {
    if (outScope.active) {
        telemetry_record_metadata_mutation_reject();
        telemetry_trace_metadata_mutation(
            "begin",
            stage,
            outScope.sequence,
            false,
            outScope.sequence,
            "scope_already_active");
        return false;
    }

    std::recursive_mutex& m = metadata_mutation_mutex();
    m.lock();

    telemetry_record_metadata_mutation_begin();
    ResourceManagerState& state = global_state();
    std::uint64_t sequence = state.nextMetadataMutationSequence.fetch_add(1, std::memory_order_relaxed);
    if (sequence == 0) {
        sequence = state.nextMetadataMutationSequence.fetch_add(1, std::memory_order_relaxed);
    }
    std::uint64_t expectedSequence = metadata_last_issued_sequence() + 1;
    if (expectedSequence == 0) {
        expectedSequence = 1;
    }
    const bool orderOk = (sequence == expectedSequence);
    if (!orderOk) {
        telemetry_record_metadata_mutation_order_violation();
    }
    metadata_last_issued_sequence() = sequence;

    outScope.sequence = sequence;
    outScope.active = true;
    telemetry_trace_metadata_mutation(
        "begin",
        stage,
        sequence,
        true,
        expectedSequence,
        orderOk ? "ok" : "sequence_order_violation");
    return true;
}

void metadata_mutation_end(MetadataMutationScope& scope, const char* stage) noexcept {
    if (!scope.active) {
        telemetry_record_metadata_mutation_reject();
        telemetry_trace_metadata_mutation(
            "end",
            stage,
            scope.sequence,
            false,
            scope.sequence,
            "scope_not_active");
        return;
    }
    telemetry_record_metadata_mutation_end();
    telemetry_trace_metadata_mutation(
        "end",
        stage,
        scope.sequence,
        true,
        scope.sequence,
        "ok");
    scope.active = false;
    scope.sequence = 0;
    metadata_mutation_mutex().unlock();
}

MetadataMutationGuard::MetadataMutationGuard(const char* stage) noexcept
    : _stage(stage) {
    (void)metadata_mutation_begin(_stage, _scope);
}

MetadataMutationGuard::~MetadataMutationGuard() noexcept {
    metadata_mutation_end(_scope, _stage);
}

} // namespace ResourceManager
} // namespace JuicerCuda
