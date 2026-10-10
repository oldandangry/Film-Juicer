#include "SpectralProcessing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#if defined(JUICER_SPECTRAL_TEST_HOOK)
#include "RustAssetBridge.h"
#endif

namespace Spectral {
    void set_cie_1931_2deg_cmf(
        const std::vector<std::pair<float, float>>& xbar,
        const std::vector<std::pair<float, float>>& ybar,
        const std::vector<std::pair<float, float>>& zbar) {
        gXBar = {};
        gYBar = {};
        gZBar = {};
        const auto assign_cmf = [](Curve& curve, const std::vector<std::pair<float, float>>& samples) {
            if (!samples_follow_reference_axis(samples)) {
                return false;
            }
#if defined(JUICER_SPECTRAL_TEST_HOOK)
            JuicerAssets::SpectralTest::before_curve_allocation();
#endif
            curve.lambda_nm.reserve(samples.size());
            curve.linear.reserve(samples.size());
            for (const auto& sample : samples) {
                if (!std::isfinite(sample.first)) {
                    return false;
                }
                curve.lambda_nm.push_back(sample.first);
                curve.linear.push_back(sample.second);
            }
            return true;
        };
        Curve x, y, z;
        const bool xOk = assign_cmf(x, xbar);
        const bool yOk = assign_cmf(y, ybar);
        const bool zOk = assign_cmf(z, zbar);
        if (!(xOk && yOk && zOk)) {
            log_resample_failure("CIE 1931 CMF resample failed", {{"x", xOk}, {"y", yOk}, {"z", zOk}});
            return;
        }
        gXBar = std::move(x);
        gYBar = std::move(y);
        gZBar = std::move(z);
    }

} // namespace Spectral
