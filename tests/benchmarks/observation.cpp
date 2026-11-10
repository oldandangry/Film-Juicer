#include "observation.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>

#include "Cuda/JuicerCudaPerformanceObservation.h"

extern "C" void fj_test_instance_prepare_hooks(
    void (*before)(void*), void (*after)(void*), void* context);

// These scalars are independent of the resource owners' locks and lifetimes.
namespace {
    std::atomic<std::uint64_t> device_values[5]{};
    std::atomic<std::uint64_t> device_peaks[5]{};
    std::atomic<std::uint64_t> upload_live{0};
    std::atomic<std::uint64_t> upload_peak{0};
    std::atomic<int> observed_device{-1};
    std::atomic<std::uintptr_t> observed_context{0};
    std::atomic<std::uint64_t> observed_epoch{0};
    void peak(std::atomic<std::uint64_t>& target, std::uint64_t value) noexcept {
        auto old = target.load(std::memory_order_relaxed);
        while (old < value && !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
        }
    }
} // namespace

namespace {
    std::mutex mutex;
    FjMeasurementRecord observed{};
    bool active = false;
    bool initialized = false;
    bool pending = false;
    std::atomic<bool> overflow{false};
    std::atomic<bool> observation_failed{false};
    std::chrono::steady_clock::time_point started;

    bool valid(const FjMeasurementRecord* record) noexcept {
        return record && record->version == FJ_MEASUREMENT_VERSION &&
               record->size == sizeof(FjMeasurementRecord);
    }
    bool add(std::uint64_t& value, std::uint64_t amount) noexcept {
        if (amount > std::numeric_limits<std::uint64_t>::max() - value) {
            overflow = true;
            return false;
        }
        value += amount;
        return true;
    }
    void before(void*) noexcept {
        try {
            std::lock_guard lock(mutex);
            if (pending) {
                add(observed.construction_incomplete, 1);
            }
            add(observed.construction_starts, 1);
            pending = true;
            started = std::chrono::steady_clock::now();
        } catch (...) {
            overflow = true;
        }
    }
    void after(void*) noexcept {
        try {
            const auto now = std::chrono::steady_clock::now();
            std::lock_guard lock(mutex);
            if (!pending) {
                add(observed.construction_incomplete, 1);
                return;
            }
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - started).count();
            if (elapsed < 0) {
                overflow = true;
            } else {
                add(observed.construction_nanoseconds, static_cast<std::uint64_t>(elapsed));
                add(observed.construction_completed, 1);
            }
            pending = false;
        } catch (...) {
            overflow = true;
        }
    }
    // Called with mutex held; pool state is copied only after releasing it.
    FjMeasurementStatus copy_locked(FjMeasurementRecord& record) noexcept {
        record = observed;
        record.device_id = observed_device.load();
        record.context_opaque = observed_context.load();
        record.context_epoch = observed_epoch.load();
        std::uint64_t* current[] = {&record.device_reserved_bytes, &record.device_committed_bytes, &record.device_retiring_bytes, &record.device_charged_bytes, &record.device_record_count};
        std::uint64_t* high[] = {&record.device_peak_reserved_bytes, &record.device_peak_committed_bytes, &record.device_peak_retiring_bytes, &record.device_peak_charged_bytes, &record.device_peak_record_count};
        for (unsigned i = 0; i < 5; ++i) {
            *current[i] = device_values[i].load();
            *high[i] = device_peaks[i].load();
        }
        record.upload_live_bytes = upload_live.load();
        record.upload_peak_bytes = upload_peak.load();
        return pending || observed.construction_incomplete ? FJ_MEASUREMENT_INCOMPLETE : FJ_MEASUREMENT_OK;
    }

    FjMeasurementStatus finish_copy(FjMeasurementRecord& record, FjMeasurementStatus status) noexcept {
        JuicerCuda::PerformanceObservation::UploadPoolSnapshot pool{};
        if (!JuicerCuda::PerformanceObservation::upload_snapshot(pool))
            observation_failed = true;
        for (unsigned i = 0; i < 4; ++i) {
            record.upload_state_bytes[i] = pool.bytes[i];
            record.upload_state_counts[i] = pool.counts[i];
        }
        if (overflow)
            return FJ_MEASUREMENT_OVERFLOW;
        if (observation_failed)
            return FJ_MEASUREMENT_OBSERVATION_FAILED;
        return status;
    }

} // namespace

extern "C" FjMeasurementStatus fj_test_ofx_measurement_begin(FjMeasurementRecord* record) {
    if (!valid(record))
        return FJ_MEASUREMENT_INVALID_ARGUMENT;
    try {
        FjMeasurementStatus status;
        {
            std::lock_guard lock(mutex);
            if (active)
                return FJ_MEASUREMENT_ALREADY_ACTIVE;
            observed = {};
            observed.version = FJ_MEASUREMENT_VERSION;
            observed.size = sizeof(observed);
            observed.device_id = -1;
            pending = false;
            if (device_values[4].load() == 0 && upload_live.load() == 0) {
                observed_device.store(-1);
                observed_context.store(0);
                observed_epoch.store(0);
            }
            for (unsigned i = 0; i < 5; ++i)
                device_peaks[i].store(device_values[i].load());
            upload_peak.store(upload_live.load());
            active = true;
            initialized = true;
            fj_test_instance_prepare_hooks(before, after, nullptr);
            status = copy_locked(*record);
        }
        return finish_copy(*record, status);
    } catch (...) {
        return FJ_MEASUREMENT_OBSERVATION_FAILED;
    }
}
extern "C" FjMeasurementStatus fj_test_ofx_measurement_read(FjMeasurementRecord* record) {
    try {
        FjMeasurementStatus status;
        {
            std::lock_guard lock(mutex);
            if (!initialized)
                return FJ_MEASUREMENT_NOT_ACTIVE;
            if (!valid(record))
                return FJ_MEASUREMENT_INVALID_ARGUMENT;
            status = copy_locked(*record);
        }
        return finish_copy(*record, status);
    } catch (...) {
        return FJ_MEASUREMENT_OBSERVATION_FAILED;
    }
}
extern "C" FjMeasurementStatus fj_test_ofx_measurement_end(FjMeasurementRecord* record) {
    if (!valid(record))
        return FJ_MEASUREMENT_INVALID_ARGUMENT;
    try {
        FjMeasurementStatus status;
        {
            std::lock_guard lock(mutex);
            if (!active)
                return FJ_MEASUREMENT_NOT_ACTIVE;
            fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
            active = false;
            if (pending) {
                add(observed.construction_incomplete, 1);
                pending = false;
            }
            status = copy_locked(*record);
        }
        return finish_copy(*record, status);
    } catch (...) {
        return FJ_MEASUREMENT_OBSERVATION_FAILED;
    }
}

namespace JuicerCuda::PerformanceObservation {
    void context(int device, const void* opaque, std::uint64_t epoch) noexcept {
        int expected = -1;
        observed_device.compare_exchange_strong(expected, device, std::memory_order_relaxed);
        auto context_value = reinterpret_cast<std::uintptr_t>(opaque);
        std::uintptr_t empty = 0;
        observed_context.compare_exchange_strong(empty, context_value, std::memory_order_relaxed);
        std::uint64_t no_epoch = 0;
        observed_epoch.compare_exchange_strong(no_epoch, epoch, std::memory_order_relaxed);
        if (observed_device.load() != device || observed_context.load() != context_value || observed_epoch.load() != epoch)
            observation_failed = true;
    }
    void ledger(int device, const LedgerSnapshot& snapshot) noexcept {
        const auto reserved = snapshot.reserved_bytes;
        const auto committed = snapshot.committed_bytes;
        const auto retiring = snapshot.retiring_bytes;
        if (observed_device.load() != device) {
            observation_failed = true;
            return;
        }
        if (reserved > UINT64_MAX - committed || reserved + committed > UINT64_MAX - retiring) {
            overflow = true;
            return;
        }
        const std::uint64_t values[] = {reserved, committed, retiring, reserved + committed + retiring, snapshot.records};
        for (unsigned i = 0; i < 5; ++i) {
            device_values[i].store(values[i], std::memory_order_relaxed);
            peak(device_peaks[i], values[i]);
        }
    }
    void upload_allocate(std::uint64_t bytes) noexcept {
        auto old = upload_live.load(std::memory_order_relaxed);
        do {
            if (bytes > UINT64_MAX - old) {
                overflow = true;
                return;
            }
        } while (!upload_live.compare_exchange_weak(old, old + bytes, std::memory_order_relaxed));
        peak(upload_peak, old + bytes);
    }
    void upload_free(std::uint64_t bytes) noexcept {
        auto old = upload_live.load(std::memory_order_relaxed);
        do {
            if (bytes > old) {
                overflow = true;
                return;
            }
        } while (!upload_live.compare_exchange_weak(old, old - bytes, std::memory_order_relaxed));
    }
} // namespace JuicerCuda::PerformanceObservation
