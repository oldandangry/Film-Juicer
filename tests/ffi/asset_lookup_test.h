#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#ifndef JUICER_ASSET_LOOKUP_TEST_HOOK
#error Asset lookup wrappers require the dedicated test objects
#endif

namespace Profiles {
    struct FilmProfileSamples;
}

namespace AssetLookupTest {

    std::optional<float> sample_synthetic_density_for_test(
        float query,
        const std::vector<float>& axis,
        const std::vector<std::array<float, 3>>& curves,
        std::size_t channel);
    std::optional<float> interp_clamped_monotonic_for_test(
        float query,
        const std::vector<float>& axis,
        const std::vector<float>& values);
    std::optional<float> sample_density_curve_for_test(
        float query,
        const Profiles::FilmProfileSamples& data,
        std::size_t channel);
    float hanatos_window_sample_for_test(float wavelength, const std::array<float, 4>& params);

} // namespace AssetLookupTest
