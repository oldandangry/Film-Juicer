#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

struct WorkingState;
namespace Couplers {
    struct Runtime;
}

namespace JuicerProc {

    struct SpatialDIRWorkspace {
        std::vector<float> filmRaw_B, filmRaw_G, filmRaw_R;
        std::vector<float> corrY, corrM, corrC;
        std::vector<float> corrYBlur, corrMBlur, corrCBlur;
        std::vector<float> tmp;
    };

} // namespace JuicerProc

namespace SpatialDIR {

    struct Callbacks {
        void* user = nullptr;
        bool (*fetchRGB)(void* user, int xx, int yy, float rgb[3]) = nullptr;
        bool (*abortCheck)(void* user) = nullptr;
    };

    // Separable Gaussian kernel builder, with radius cap for safety.
    // Kept inline so test wrappers can use it without linking a specific TU.
    inline void buildGaussianKernel(float sigma, std::vector<float>& kernel) {
        kernel.clear();
        if (!(std::isfinite(sigma)) || sigma <= 0.0f) {
            kernel.push_back(1.0f);
            return;
        }

        const int radiusRaw = std::max(1, int(std::ceil(3.0f * sigma)));
        const int radius = std::min(radiusRaw, 75); // cap at 75 taps each side

        kernel.resize(size_t(2 * radius + 1));
        const float s2 = sigma * sigma * 2.0f;
        float wsum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            const float w = std::exp(-(i * i) / s2);
            kernel[size_t(i + radius)] = w;
            wsum += w;
        }
        for (float& w : kernel) w /= wsum;
    }

    void buildSpatialDIRCorrections(
        int width,
        int height,
        const WorkingState& ws,
        const Couplers::Runtime& dirRT,
        float exposureScale,
        const Callbacks& callbacks,
        JuicerProc::SpatialDIRWorkspace& work,
        std::vector<float>& kernelCache);

} // namespace SpatialDIR

