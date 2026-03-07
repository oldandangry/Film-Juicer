// Cuda/JuicerCudaPrecompute.cpp
// CPU precompute helpers for CUDA resources.
#include "WorkingState.h"
#include "SpectralContext.h"
#include "ScanStage.h"
#include "Print.h"
#include "PrintPreflashShared.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace JuicerCuda {
namespace Precompute {

bool build_scan_lut_cpu(const Scanner::ScannerMediumRuntime& medium, std::uint32_t res, std::vector<double>& out, std::string& outError) {
    // Match the GPU hot path: store log2(XYZ) so device code can use exp2() instead of pow(10, ...).
    // Note: Pipeline::ScanStage::spectral_to_log_xyz returns log10(XYZ) (with epsilon).
    constexpr double kLog2_10 = 3.32192809488736234787;

    const size_t sRes = static_cast<size_t>(res);
    const size_t voxels = sRes * sRes * sRes;
    const size_t count = voxels * 3u;
    out.clear();
    out.resize(count);

    for (std::uint32_t z = 0; z < res; ++z) {
        const double nz = (res > 1u) ? static_cast<double>(z) / static_cast<double>(res - 1u) : 0.0;
        for (std::uint32_t y = 0; y < res; ++y) {
            const double ny = (res > 1u) ? static_cast<double>(y) / static_cast<double>(res - 1u) : 0.0;
            for (std::uint32_t x = 0; x < res; ++x) {
                const double nx = (res > 1u) ? static_cast<double>(x) / static_cast<double>(res - 1u) : 0.0;
                const size_t idx = (static_cast<size_t>(z) * sRes + static_cast<size_t>(y)) * sRes + static_cast<size_t>(x);
                const size_t base = idx * 3u;
                const double D_norm[3] = { nx, ny, nz };
                double logXYZ[3] = { 0.0, 0.0, 0.0 };
                Pipeline::ScanStage::spectral_to_log_xyz(medium, D_norm, logXYZ);
                if (!std::isfinite(logXYZ[0]) || !std::isfinite(logXYZ[1]) || !std::isfinite(logXYZ[2])) {
                    outError = "scan LUT build produced non-finite logXYZ";
                    return false;
                }
                out[base + 0] = logXYZ[0];
                out[base + 1] = logXYZ[1];
                out[base + 2] = logXYZ[2];

                out[base + 0] *= kLog2_10;
                out[base + 1] *= kLog2_10;
                out[base + 2] *= kLog2_10;
            }
        }
    }

    return true;
}

void build_hanatos_integrated_lut_cpu(const Spectral::SpectralContext& ctx, const WorkingState& ws, std::vector<float>& out) {
    const int N = ctx.hanSpectra.size;
    const int K = ctx.hanSpectra.numSamples;
    out.clear();
    out.resize(static_cast<size_t>(N) * static_cast<size_t>(N) * 4u, 0.0f);

    const float* lut = ctx.hanSpectra.data.data();
    const float* sB = ws.sensB.linear.data();
    const float* sG = ws.sensG.linear.data();
    const float* sR = ws.sensR.linear.data();
    const size_t stride = static_cast<size_t>(K);
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            double accB = 0.0;
            double accG = 0.0;
            double accR = 0.0;
            const size_t base = (static_cast<size_t>(x) * static_cast<size_t>(N) + static_cast<size_t>(y)) * stride;
            for (int k = 0; k < K; ++k) {
                const float raw = lut[base + static_cast<size_t>(k)];
                if (!std::isfinite(raw)) {
                    continue;
                }
                const float e = (raw > 0.0f) ? raw : 0.0f;
                if (!std::isfinite(e)) {
                    continue;
                }
                const double e64 = static_cast<double>(e);
                const float sb = sB[k];
                const float sg = sG[k];
                const float sr = sR[k];
                if (std::isfinite(sb)) accB += e64 * static_cast<double>(sb);
                if (std::isfinite(sg)) accG += e64 * static_cast<double>(sg);
                if (std::isfinite(sr)) accR += e64 * static_cast<double>(sr);
            }

            const size_t outBase = (static_cast<size_t>(x) * static_cast<size_t>(N) + static_cast<size_t>(y)) * 4u;
            out[outBase + 0] = static_cast<float>(accR);
            out[outBase + 1] = static_cast<float>(accG);
            out[outBase + 2] = static_cast<float>(accB);
            out[outBase + 3] = 0.0f;
        }
    }
}

bool build_print_preflash_raw(const WorkingState& ws, const Print::Runtime& prt, float outRaw[3], int& outShapeK) {
    return PrintPreflashShared::compute_raw(ws, prt, outRaw, outShapeK);
}

} // namespace Precompute
} // namespace JuicerCuda
