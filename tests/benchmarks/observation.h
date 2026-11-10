#ifndef JUICER_MEASUREMENT_OBSERVATION_H
#define JUICER_MEASUREMENT_OBSERVATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FJ_MEASUREMENT_VERSION 1U

typedef uint32_t FjMeasurementStatus;
#define FJ_MEASUREMENT_OK 0U
#define FJ_MEASUREMENT_INVALID_ARGUMENT 1U
#define FJ_MEASUREMENT_ALREADY_ACTIVE 2U
#define FJ_MEASUREMENT_NOT_ACTIVE 3U
#define FJ_MEASUREMENT_OVERFLOW 4U
#define FJ_MEASUREMENT_INCOMPLETE 5U
#define FJ_MEASUREMENT_OBSERVATION_FAILED 6U

typedef struct FjMeasurementRecord {
    uint32_t version;
    uint32_t size;
    uint64_t construction_starts;
    uint64_t construction_completed;
    uint64_t construction_incomplete;
    uint64_t construction_nanoseconds;
    int32_t device_id;
    uint32_t reserved;
    uint64_t context_opaque;
    uint64_t context_epoch;
    uint64_t device_reserved_bytes;
    uint64_t device_committed_bytes;
    uint64_t device_retiring_bytes;
    uint64_t device_charged_bytes;
    uint64_t device_record_count;
    uint64_t device_peak_reserved_bytes;
    uint64_t device_peak_committed_bytes;
    uint64_t device_peak_retiring_bytes;
    uint64_t device_peak_charged_bytes;
    uint64_t device_peak_record_count;
    uint64_t upload_live_bytes;
    uint64_t upload_peak_bytes;
    uint64_t upload_state_bytes[4];
    uint64_t upload_state_counts[4];
} FjMeasurementRecord;

#if defined(__GNUC__)
#define FJ_MEASUREMENT_EXPORT __attribute__((visibility("default")))
#else
#define FJ_MEASUREMENT_EXPORT
#endif

FJ_MEASUREMENT_EXPORT FjMeasurementStatus fj_test_ofx_measurement_begin(FjMeasurementRecord* record);
FJ_MEASUREMENT_EXPORT FjMeasurementStatus fj_test_ofx_measurement_read(FjMeasurementRecord* record);
FJ_MEASUREMENT_EXPORT FjMeasurementStatus fj_test_ofx_measurement_end(FjMeasurementRecord* record);

#undef FJ_MEASUREMENT_EXPORT

#ifdef __cplusplus
}

namespace JuicerMeasurement {
    enum class ConstructionOutcome : uint8_t {
        Constructed,
        Reused,
        Unknown,
        Invalid
    };

    struct ConstructionCheck {
        bool expected;
        bool callback_succeeded;
        bool output_valid;
        bool lifecycle_valid;
    };

    ConstructionOutcome classify_construction(
        const FjMeasurementRecord& before, FjMeasurementStatus before_status, const FjMeasurementRecord& after, FjMeasurementStatus after_status, ConstructionCheck check) noexcept;
} // namespace JuicerMeasurement
#endif
#endif
