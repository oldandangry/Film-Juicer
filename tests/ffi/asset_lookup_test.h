#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace AssetLookupTest {

    std::optional<float> interp_clamped_monotonic_for_test(
        float query,
        const std::vector<float>& axis,
        const std::vector<float>& values);
    std::optional<float> sample_density_curve_for_test(
        float query,
        const std::vector<float>& axis,
        const std::vector<std::array<float, 3>>& densityCurves,
        std::size_t channel);
    float hanatos_window_sample_for_test(float wavelength, const std::array<float, 4>& params);

} // namespace AssetLookupTest
