#pragma once

#include <cstdint>
#include <vector>

#include "juicer_test_api.h"

namespace JuicerTests {
    class Cohort;
}

namespace JuicerCudaTest {
    // The caller retains Runtime/Assets and the inputs through both synchronous passes.
    void check_production_state_contract(FjRuntimeView runtime, const FjPreparationParams& params, std::uint64_t expectedInputIdentity, std::uint32_t route, const std::vector<float>& expectedPixels, const JuicerTests::Cohort& rows);
} // namespace JuicerCudaTest
