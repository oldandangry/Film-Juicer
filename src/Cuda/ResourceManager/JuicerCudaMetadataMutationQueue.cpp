// Cuda/ResourceManager/JuicerCudaMetadataMutationQueue.cpp
//
// Deterministic metadata-mutation queue semantics (ticketed, bounded, re-entrant).

#include "Cuda/ResourceManager/JuicerCudaResourceState.h"
#include "Cuda/ResourceManager/JuicerCudaResourceTelemetry.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace JuicerCuda {
namespace ResourceManager {

namespace {

constexpr std::uint64_t kMetadataMutationQueueDepthLimit = 64ull;
constexpr auto kMetadataMutationQueueWaitStep = std::chrono::milliseconds(1);
thread_local std::uint32_t gMutationThreadDepth = 0;
thread_local std::uint64_t gMutationThreadTicket = 0;
thread_local std::uint64_t gMutationThreadBeginCount = 0;

struct MetadataMutationLane {
    std::mutex mutex;
    std::condition_variable cv;
    std::uint64_t nextTicket = 1;
    std::uint64_t servingTicket = 1;
    std::uint64_t activeTicket = 0;
    std::thread::id ownerThread{};
    std::uint32_t ownerDepth = 0;
};

struct MetadataMutationLaneDirectory {
    std::mutex mutex;
    std::shared_ptr<MetadataMutationLane> fallbackLane = std::make_shared<MetadataMutationLane>();
    std::unordered_map<DeviceContextKey, std::shared_ptr<MetadataMutationLane>, DeviceContextKeyHash> byManagerKey;
};

struct LaneAcquireResult {
    std::uint64_t ticket = 0;
    std::uint64_t waitedMs = 0;
    bool observedWait = false;
    bool observedBackpressure = false;
    bool reentrant = false;
};

MetadataMutationLaneDirectory& mutation_lane_directory() {
    static MetadataMutationLaneDirectory directory{};
    return directory;
}

std::mutex& mutation_sequence_mutex() {
    static std::mutex m;
    return m;
}

std::uint64_t& mutation_last_issued_sequence() {
    static std::uint64_t sequence = 0;
    return sequence;
}

inline std::uint64_t lane_depth_nolock(const MetadataMutationLane& lane) noexcept {
    if (lane.nextTicket < lane.servingTicket) {
        return 0;
    }
    return lane.nextTicket - lane.servingTicket;
}

inline std::uint64_t elapsed_ms(const std::chrono::steady_clock::time_point& start) noexcept {
    const auto delta = std::chrono::steady_clock::now() - start;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
}

std::shared_ptr<MetadataMutationLane> resolve_mutation_lane(const DeviceContextKey* managerKey) {
    MetadataMutationLaneDirectory& directory = mutation_lane_directory();
    if (!managerKey || managerKey->deviceId < 0) {
        return directory.fallbackLane;
    }

    std::lock_guard<std::mutex> lock(directory.mutex);
    auto it = directory.byManagerKey.find(*managerKey);
    if (it != directory.byManagerKey.end()) {
        return it->second;
    }

    std::shared_ptr<MetadataMutationLane> lane = std::make_shared<MetadataMutationLane>();
    directory.byManagerKey.emplace(*managerKey, lane);
    return lane;
}

void reset_scope(MetadataMutationScope& scope) noexcept {
    scope.sequence = 0;
    scope.queueTicket = 0;
    scope.managerKey = DeviceContextKey{};
    scope.hasManagerKey = false;
    scope.active = false;
}

void reset_thread_mutation_context() noexcept {
    gMutationThreadDepth = 0;
    gMutationThreadTicket = 0;
}

void trace_mutation_reject(
    const char* action,
    const char* stage,
    std::uint64_t sequence,
    std::uint64_t expectedSequence,
    std::uint64_t queueTicket,
    std::uint64_t queueDepth,
    std::uint64_t waitedMs,
    bool accepted,
    const char* reason) noexcept {
    telemetry_record_metadata_mutation_reject();
    telemetry_record_metadata_queue_reject();
    telemetry_trace_metadata_queue(
        "reject",
        stage,
        queueTicket,
        queueDepth,
        waitedMs,
        accepted,
        reason);
    telemetry_trace_metadata_mutation(
        action,
        stage,
        sequence,
        false,
        expectedSequence,
        reason);
}

void note_queue_wait_if_needed(bool& observedWait) noexcept {
    if (!observedWait) {
        observedWait = true;
        telemetry_record_metadata_queue_wait();
    }
}

void wait_on_lane_for_step(
    std::unique_lock<std::mutex>& lock,
    MetadataMutationLane& lane,
    std::uint64_t& waitedMs) {
    const auto waitStart = std::chrono::steady_clock::now();
    lane.cv.wait_for(lock, kMetadataMutationQueueWaitStep);
    waitedMs += elapsed_ms(waitStart);
}

bool acquire_lane_ticket(
    MetadataMutationLane& lane,
    const std::thread::id& currentThread,
    const char* stage,
    LaneAcquireResult& out) {
    std::unique_lock<std::mutex> lock(lane.mutex);
    if (lane.ownerDepth > 0 && lane.ownerThread == currentThread) {
        out.reentrant = true;
        ++lane.ownerDepth;
        out.ticket = lane.activeTicket;
        if (out.ticket == 0) {
            out.ticket = lane.servingTicket;
            lane.activeTicket = out.ticket;
        }
        telemetry_trace_metadata_queue(
            "reenter",
            stage,
            out.ticket,
            lane_depth_nolock(lane),
            0,
            true,
            "owner_reentrant");
        return true;
    }

    while (lane_depth_nolock(lane) >= kMetadataMutationQueueDepthLimit) {
        if (!out.observedBackpressure) {
            out.observedBackpressure = true;
            telemetry_record_metadata_queue_backpressure();
            telemetry_trace_metadata_queue(
                "backpressure",
                stage,
                0,
                lane_depth_nolock(lane),
                out.waitedMs,
                true,
                "depth_limit");
        }
        note_queue_wait_if_needed(out.observedWait);
        wait_on_lane_for_step(lock, lane, out.waitedMs);
    }

    out.ticket = lane.nextTicket++;
    if (out.ticket == 0) {
        trace_mutation_reject(
            "begin",
            stage,
            0,
            0,
            0,
            lane_depth_nolock(lane),
            out.waitedMs,
            false,
            "ticket_overflow");
        return false;
    }

    telemetry_record_metadata_queue_enqueue();
    const std::uint64_t depthAfterEnqueue = lane_depth_nolock(lane);
    telemetry_note_metadata_queue_depth(depthAfterEnqueue);
    telemetry_trace_metadata_queue(
        "enqueue",
        stage,
        out.ticket,
        depthAfterEnqueue,
        out.waitedMs,
        true,
        out.observedBackpressure ? "accepted_after_backpressure" : "accepted");

    while (out.ticket != lane.servingTicket || lane.ownerDepth != 0) {
        note_queue_wait_if_needed(out.observedWait);
        wait_on_lane_for_step(lock, lane, out.waitedMs);
    }

    lane.ownerThread = currentThread;
    lane.ownerDepth = 1;
    lane.activeTicket = out.ticket;
    telemetry_record_metadata_queue_dequeue();
    telemetry_trace_metadata_queue(
        "dequeue",
        stage,
        out.ticket,
        lane_depth_nolock(lane),
        out.waitedMs,
        true,
        out.observedWait ? "turn_wait" : "immediate");
    return true;
}

} // namespace

bool metadata_mutation_begin(
    const char* stage,
    MetadataMutationScope& outScope,
    const DeviceContextKey* managerKey) noexcept {
    if (outScope.active) {
        trace_mutation_reject(
            "begin",
            stage,
            outScope.sequence,
            outScope.sequence,
            outScope.queueTicket,
            0,
            0,
            false,
            "scope_already_active");
        return false;
    }

    std::shared_ptr<MetadataMutationLane> lane = resolve_mutation_lane(managerKey);
    if (!lane) {
        trace_mutation_reject(
            "begin",
            stage,
            0,
            0,
            0,
            0,
            0,
            false,
            "missing_lane");
        return false;
    }

    LaneAcquireResult acquire{};
    if (!acquire_lane_ticket(*lane, std::this_thread::get_id(), stage, acquire)) {
        return false;
    }

    telemetry_record_metadata_mutation_begin();
    std::uint64_t sequence = 0;
    std::uint64_t expectedSequence = 0;
    bool orderOk = true;
    {
        std::lock_guard<std::mutex> lock(mutation_sequence_mutex());
        ResourceManagerState& state = global_state();
        sequence = state.nextMetadataMutationSequence.fetch_add(1, std::memory_order_relaxed);
        if (sequence == 0) {
            sequence = state.nextMetadataMutationSequence.fetch_add(1, std::memory_order_relaxed);
        }
        expectedSequence = mutation_last_issued_sequence() + 1;
        if (expectedSequence == 0) {
            expectedSequence = 1;
        }
        orderOk = (sequence == expectedSequence);
        if (!orderOk) {
            telemetry_record_metadata_mutation_order_violation();
        }
        mutation_last_issued_sequence() = sequence;
    }

    outScope.sequence = sequence;
    outScope.queueTicket = acquire.ticket;
    outScope.hasManagerKey = (managerKey && managerKey->deviceId >= 0);
    outScope.managerKey = outScope.hasManagerKey ? *managerKey : DeviceContextKey{};
    outScope.active = true;
    gMutationThreadDepth += 1;
    gMutationThreadTicket = acquire.ticket;
    gMutationThreadBeginCount += 1;

    telemetry_trace_metadata_mutation(
        "begin",
        stage,
        sequence,
        true,
        expectedSequence,
        orderOk ? (acquire.reentrant ? "ok_reentrant" : "ok") : "sequence_order_violation");
    return true;
}

void metadata_mutation_end(MetadataMutationScope& scope, const char* stage) noexcept {
    if (!scope.active) {
        trace_mutation_reject(
            "end",
            stage,
            scope.sequence,
            scope.sequence,
            scope.queueTicket,
            0,
            0,
            false,
            "scope_not_active");
        return;
    }

    std::shared_ptr<MetadataMutationLane> lane =
        resolve_mutation_lane(scope.hasManagerKey ? &scope.managerKey : nullptr);
    if (!lane) {
        trace_mutation_reject(
            "end",
            stage,
            scope.sequence,
            scope.sequence,
            scope.queueTicket,
            0,
            0,
            false,
            "missing_lane");
        reset_thread_mutation_context();
        reset_scope(scope);
        return;
    }

    bool releaseTopLevel = false;
    std::uint64_t depthAfterRelease = 0;
    const std::thread::id currentThread = std::this_thread::get_id();
    {
        std::lock_guard<std::mutex> lock(lane->mutex);
        if (lane->ownerDepth == 0 || lane->ownerThread != currentThread) {
            trace_mutation_reject(
                "end",
                stage,
                scope.sequence,
                scope.sequence,
                scope.queueTicket,
                lane_depth_nolock(*lane),
                0,
                false,
                "owner_mismatch");
            reset_thread_mutation_context();
            reset_scope(scope);
            return;
        }

        telemetry_record_metadata_mutation_end();
        --lane->ownerDepth;
        if (lane->ownerDepth == 0) {
            if (lane->servingTicket != scope.queueTicket) {
                telemetry_record_metadata_mutation_order_violation();
                telemetry_trace_metadata_queue(
                    "order_violation",
                    stage,
                    scope.queueTicket,
                    lane_depth_nolock(*lane),
                    0,
                    false,
                    "release_ticket_mismatch");
            }
            if (lane->servingTicket < lane->nextTicket) {
                ++lane->servingTicket;
            }
            lane->ownerThread = std::thread::id{};
            lane->activeTicket = 0;
            releaseTopLevel = true;
        }
        depthAfterRelease = lane_depth_nolock(*lane);
    }

    if (releaseTopLevel) {
        lane->cv.notify_all();
        telemetry_trace_metadata_queue(
            "release",
            stage,
            scope.queueTicket,
            depthAfterRelease,
            0,
            true,
            "ok");
    }
    else {
        telemetry_trace_metadata_queue(
            "release_nested",
            stage,
            scope.queueTicket,
            depthAfterRelease,
            0,
            true,
            "owner_reentrant");
    }

    telemetry_trace_metadata_mutation(
        "end",
        stage,
        scope.sequence,
        true,
        scope.sequence,
        "ok");
    if (gMutationThreadDepth > 0) {
        --gMutationThreadDepth;
    }
    if (gMutationThreadDepth == 0) {
        gMutationThreadTicket = 0;
    }
    reset_scope(scope);
}

bool metadata_mutation_thread_active() noexcept {
    return gMutationThreadDepth > 0;
}

std::uint32_t metadata_mutation_thread_depth() noexcept {
    return gMutationThreadDepth;
}

std::uint64_t metadata_mutation_thread_ticket() noexcept {
    return gMutationThreadTicket;
}

std::uint64_t metadata_mutation_thread_begin_count() noexcept {
    return gMutationThreadBeginCount;
}

MetadataMutationGuard::MetadataMutationGuard(
    const char* stage,
    const DeviceContextKey* managerKey) noexcept
    : _stage(stage) {
    if (managerKey) {
        _hasManagerKey = true;
        _managerKey = *managerKey;
    }
    const DeviceContextKey* key = _hasManagerKey ? &_managerKey : nullptr;
    (void)metadata_mutation_begin(_stage, _scope, key);
}

MetadataMutationGuard::~MetadataMutationGuard() noexcept {
    metadata_mutation_end(_scope, _stage);
}

} // namespace ResourceManager
} // namespace JuicerCuda
