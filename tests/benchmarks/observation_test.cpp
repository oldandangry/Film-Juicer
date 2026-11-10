#include "observation.h"
#include "Cuda/JuicerCudaPerformanceObservation.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {
    struct PrepareHooks {
        void (*before)(void*) = nullptr;
        void (*after)(void*) = nullptr;
        void* context = nullptr;
    };
    PrepareHooks hooks;
    FjMeasurementRecord record() {
        FjMeasurementRecord result{};
        result.version = FJ_MEASUREMENT_VERSION;
        result.size = sizeof(result);
        return result;
    }

} // namespace
extern "C" int fj_measurement_c11_record_size(void);
extern "C" void fj_test_instance_prepare_hooks(void (*before)(void*), void (*after)(void*), void* context) {
    hooks = {before, after, context};
}
namespace JuicerCuda::PerformanceObservation {
    bool upload_snapshot(UploadPoolSnapshot& snapshot) noexcept {
        for (unsigned i = 0; i < 4; ++i) {
            snapshot.bytes[i] = 0;
            snapshot.counts[i] = 0;
        }
        return true;
    }
} // namespace JuicerCuda::PerformanceObservation

TEST(Measurement, RecordVersionAndSize) {
    EXPECT_EQ(fj_measurement_c11_record_size(), sizeof(FjMeasurementRecord));
    auto value = record();
    ++value.version;
    EXPECT_EQ(fj_test_ofx_measurement_begin(&value), FJ_MEASUREMENT_INVALID_ARGUMENT);
    value = record();
    --value.size;
    EXPECT_EQ(fj_test_ofx_measurement_begin(&value), FJ_MEASUREMENT_INVALID_ARGUMENT);
}
TEST(Measurement, UnpairedSpan) {
    auto value = record();
    ASSERT_EQ(fj_test_ofx_measurement_begin(&value), FJ_MEASUREMENT_OK);
    hooks.before(hooks.context);
    EXPECT_EQ(fj_test_ofx_measurement_read(&value), FJ_MEASUREMENT_INCOMPLETE);
    EXPECT_EQ(fj_test_ofx_measurement_end(&value), FJ_MEASUREMENT_INCOMPLETE);
}
TEST(Measurement, CallbackClassification) {
    auto initial = record();
    ASSERT_EQ(fj_test_ofx_measurement_begin(&initial), FJ_MEASUREMENT_OK);
    hooks.before(hooks.context);
    hooks.after(hooks.context);
    auto paired = record();
    ASSERT_EQ(fj_test_ofx_measurement_read(&paired), FJ_MEASUREMENT_OK);
    using JuicerMeasurement::classify_construction;
    using JuicerMeasurement::ConstructionOutcome;
    EXPECT_EQ(classify_construction(initial, FJ_MEASUREMENT_OK, paired, FJ_MEASUREMENT_OK, {.expected = true, .callback_succeeded = true, .output_valid = true, .lifecycle_valid = true}), ConstructionOutcome::Constructed);
    EXPECT_EQ(classify_construction(initial, FJ_MEASUREMENT_OK, paired, FJ_MEASUREMENT_OK, {.expected = true, .callback_succeeded = false, .output_valid = true, .lifecycle_valid = true}), ConstructionOutcome::Unknown);
    EXPECT_EQ(classify_construction(paired, FJ_MEASUREMENT_OK, paired, FJ_MEASUREMENT_OK, {.expected = false, .callback_succeeded = true, .output_valid = true, .lifecycle_valid = true}), ConstructionOutcome::Reused);
    EXPECT_EQ(classify_construction(paired, FJ_MEASUREMENT_OK, paired, FJ_MEASUREMENT_OK, {.expected = true, .callback_succeeded = true, .output_valid = true, .lifecycle_valid = true}), ConstructionOutcome::Invalid);
    EXPECT_EQ(classify_construction(initial, FJ_MEASUREMENT_OK, paired, FJ_MEASUREMENT_OK, {.expected = false, .callback_succeeded = true, .output_valid = true, .lifecycle_valid = true}), ConstructionOutcome::Invalid);
    EXPECT_EQ(classify_construction(initial, FJ_MEASUREMENT_OK, paired, FJ_MEASUREMENT_INCOMPLETE, {.expected = true, .callback_succeeded = true, .output_valid = true, .lifecycle_valid = true}), ConstructionOutcome::Invalid);
    EXPECT_EQ(classify_construction(initial, FJ_MEASUREMENT_OK, paired, FJ_MEASUREMENT_OK, {.expected = true, .callback_succeeded = true, .output_valid = false, .lifecycle_valid = true}), ConstructionOutcome::Invalid);
    EXPECT_EQ(fj_test_ofx_measurement_end(&paired), FJ_MEASUREMENT_OK);
}
TEST(Measurement, HighWaterSurvivesRelease) {
    auto value = record();
    ASSERT_EQ(fj_test_ofx_measurement_begin(&value), FJ_MEASUREMENT_OK);
    JuicerCuda::PerformanceObservation::context(0, reinterpret_cast<void*>(1), 1);
    JuicerCuda::PerformanceObservation::ledger(0, {20, 0, 0, 1});
    JuicerCuda::PerformanceObservation::ledger(0, {0, 7, 0, 1});
    ASSERT_EQ(fj_test_ofx_measurement_read(&value), FJ_MEASUREMENT_OK);
    EXPECT_EQ(value.device_charged_bytes, 7u);
    EXPECT_EQ(value.device_peak_charged_bytes, 20u);
    JuicerCuda::PerformanceObservation::ledger(0, {0, 0, 0, 0});
    EXPECT_EQ(fj_test_ofx_measurement_end(&value), FJ_MEASUREMENT_OK);
}
TEST(Measurement, FailedFreeRetainsUploadOwnership) {
    auto value = record();
    ASSERT_EQ(fj_test_ofx_measurement_begin(&value), FJ_MEASUREMENT_OK);
    JuicerCuda::PerformanceObservation::upload_allocate(32);
    // A failed physical free produces no free notification.
    ASSERT_EQ(fj_test_ofx_measurement_read(&value), FJ_MEASUREMENT_OK);
    EXPECT_EQ(value.upload_live_bytes, 32u);
    JuicerCuda::PerformanceObservation::upload_free(32);
    EXPECT_EQ(fj_test_ofx_measurement_end(&value), FJ_MEASUREMENT_OK);
}
TEST(Measurement, ResetPreservesLiveOwnership) {
    auto value = record();
    JuicerCuda::PerformanceObservation::upload_allocate(16);
    ASSERT_EQ(fj_test_ofx_measurement_begin(&value), FJ_MEASUREMENT_OK);
    EXPECT_EQ(value.upload_live_bytes, 16u);
    EXPECT_EQ(value.upload_peak_bytes, 16u);
    JuicerCuda::PerformanceObservation::upload_free(16);
    EXPECT_EQ(fj_test_ofx_measurement_end(&value), FJ_MEASUREMENT_OK);
}
