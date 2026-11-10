#include "observation.h"

namespace JuicerMeasurement {
    ConstructionOutcome classify_construction(
        const FjMeasurementRecord& before, FjMeasurementStatus before_status, const FjMeasurementRecord& after, FjMeasurementStatus after_status, ConstructionCheck check) noexcept {
        if (!check.callback_succeeded) {
            return ConstructionOutcome::Unknown;
        }
        if (before_status != FJ_MEASUREMENT_OK || after_status != FJ_MEASUREMENT_OK ||
            before.version != FJ_MEASUREMENT_VERSION || after.version != FJ_MEASUREMENT_VERSION ||
            before.size != sizeof(FjMeasurementRecord) || after.size != sizeof(FjMeasurementRecord) ||
            after.construction_starts < before.construction_starts ||
            after.construction_completed < before.construction_completed ||
            after.construction_incomplete != before.construction_incomplete ||
            after.construction_nanoseconds < before.construction_nanoseconds) {
            return ConstructionOutcome::Invalid;
        }
        if (!check.output_valid || !check.lifecycle_valid) {
            return ConstructionOutcome::Invalid;
        }
        const auto starts = after.construction_starts - before.construction_starts;
        const auto completed = after.construction_completed - before.construction_completed;
        if (check.expected && starts == 1 && completed == 1) {
            return ConstructionOutcome::Constructed;
        }
        if (!check.expected && starts == 0 && completed == 0 &&
            after.construction_nanoseconds == before.construction_nanoseconds) {
            return ConstructionOutcome::Reused;
        }
        return ConstructionOutcome::Invalid;
    }
} // namespace JuicerMeasurement
