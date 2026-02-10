#include "SpatialDIR.h"

#include <algorithm>
#include <sstream>

#include "Couplers.h"
#include "Logging.h"
#include "PipelineRunner.h"
#include "WorkingState.h"

namespace SpatialDIR {

    namespace {

        void blurChannelSeparable(
            const std::vector<float>& src,
            std::vector<float>& tmp,
            std::vector<float>& dst,
            int width,
            int height,
            const std::vector<float>& k)
        {
            tmp.assign(size_t(width * height), 0.0f);
            const int radius = int(k.size() / 2);
            const auto reflectIndex = [](int idx, int size) -> int {
                if (size <= 1) {
                    return 0;
                }
                while (idx < 0 || idx >= size) {
                    if (idx < 0) {
                        idx = -idx;
                    }
                    else {
                        idx = 2 * size - idx - 2;
                    }
                }
                return idx;
                };
            for (int y = 0; y < height; ++y) {
                const float* srow = &src[size_t(y * width)];
                float* trow = &tmp[size_t(y * width)];
                for (int x = 0; x < width; ++x) {
                    float acc = 0.0f;
                    for (int j = -radius; j <= radius; ++j) {
                        const int xx = reflectIndex(x + j, width);
                        acc += srow[xx] * k[size_t(j + radius)];
                    }
                    trow[x] = acc;
                }
            }
            dst.assign(size_t(width * height), 0.0f);
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    float acc = 0.0f;
                    for (int j = -radius; j <= radius; ++j) {
                        const int yy = reflectIndex(y + j, height);
                        acc += tmp[size_t(yy * width + x)] * k[size_t(j + radius)];
                    }
                    dst[size_t(y * width + x)] = acc;
                }
            }
        }

    } // namespace

    void buildSpatialDIRCorrections(
        int width,
        int height,
        const WorkingState& ws,
        const Couplers::Runtime& dirRT,
        float exposureScale,
        const Callbacks& callbacks,
        JuicerProc::SpatialDIRWorkspace& work,
        std::vector<float>& kernelCache)
    {
        const size_t total = size_t(width) * size_t(height);
        work.filmRaw_B.assign(total, 0.0f);
        work.filmRaw_G.assign(total, 0.0f);
        work.filmRaw_R.assign(total, 0.0f);
        work.corrY.assign(total, 0.0f);
        work.corrM.assign(total, 0.0f);
        work.corrC.assign(total, 0.0f);
        work.corrYBlur.assign(total, 0.0f);
        work.corrMBlur.assign(total, 0.0f);
        work.corrCBlur.assign(total, 0.0f);
        work.tmp.assign(total, 0.0f);

        if (!dirRT.active) {
            return;
        }

        if (!callbacks.fetchRGB || !callbacks.abortCheck) {
            JTRACE("DIR", "FATAL: SpatialDIR callbacks not provided");
            return;
        }

        const bool verboseDiagnostics = JTRACE_ENABLED(3);
        auto should_abort = [&]() -> bool {
            return callbacks.abortCheck(callbacks.user);
            };
        auto trace_abort_fast = [&](const char* stage) {
            if (!verboseDiagnostics) {
                return;
            }
            JTRACE_VERBOSE("MSCPU", std::string("path=spatial_dir event=abort_fast stage=") + stage);
            };

        // Per agx-emulsion parity: use same sensitivities everywhere (no separate "before balance" state).
        // Profiles contain pre-balanced sensitivities; spatial DIR and mid-gray must match pixel render.
        const Spectral::Curve& sensB_forExposure = ws.sensB;
        const Spectral::Curve& sensG_forExposure = ws.sensG;
        const Spectral::Curve& sensR_forExposure = ws.sensR;

        // DEBUG: Log WorkingState sensitivity curves before use
        if (verboseDiagnostics) {
            const int idx_450 = 14, idx_520 = 28, idx_650 = 54;
            std::ostringstream oss;
            oss << "WS_SENS_PREUSE: B[450nm]=" << (sensB_forExposure.linear.size() > idx_450 ? sensB_forExposure.linear[idx_450] : -999.0f)
                << " G[450nm]=" << (sensG_forExposure.linear.size() > idx_450 ? sensG_forExposure.linear[idx_450] : -999.0f)
                << " R[450nm]=" << (sensR_forExposure.linear.size() > idx_450 ? sensR_forExposure.linear[idx_450] : -999.0f)
                << " | B[520nm]=" << (sensB_forExposure.linear.size() > idx_520 ? sensB_forExposure.linear[idx_520] : -999.0f)
                << " G[520nm]=" << (sensG_forExposure.linear.size() > idx_520 ? sensG_forExposure.linear[idx_520] : -999.0f)
                << " R[520nm]=" << (sensR_forExposure.linear.size() > idx_520 ? sensR_forExposure.linear[idx_520] : -999.0f)
                << " | B[650nm]=" << (sensB_forExposure.linear.size() > idx_650 ? sensB_forExposure.linear[idx_650] : -999.0f)
                << " G[650nm]=" << (sensG_forExposure.linear.size() > idx_650 ? sensG_forExposure.linear[idx_650] : -999.0f)
                << " R[650nm]=" << (sensR_forExposure.linear.size() > idx_650 ? sensR_forExposure.linear[idx_650] : -999.0f);
            JTRACE_VERBOSE("SPECTRAL", oss.str());
        }

        // Pass A: sample exposures, convert to logE, compute DIR corrections per pixel.
        const int center_xx = width / 2;
        const int center_yy = height / 2;

        Pipeline::PipelineRunnerConfig runnerCfg{};
        runnerCfg.enablePrint = false;
        const Pipeline::PipelineRunner runner(runnerCfg);

        bool aborted = false;
        for (int yy = 0; yy < height; ++yy) {
            if (should_abort()) {
                aborted = true;
                break;
            }
            for (int xx = 0; xx < width; ++xx) {
                if ((xx & 63) == 0 && should_abort()) {
                    aborted = true;
                    break;
                }
                const size_t idx = size_t(yy) * size_t(width) + size_t(xx);
                float rgbIn[3] = { 0.0f, 0.0f, 0.0f };
                if (!callbacks.fetchRGB(callbacks.user, xx, yy, rgbIn)) {
                    work.filmRaw_B[idx] = work.filmRaw_G[idx] = work.filmRaw_R[idx] = 0.0f;
                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
                    continue;
                }

                // SPD DEBUG: Log input RGB for center pixel
                if (verboseDiagnostics && xx == center_xx && yy == center_yy) {
                    std::ostringstream oss;
                    oss << "INPUT_RGB tile_pixel(" << xx << "," << yy << "): R=" << rgbIn[0] << " G=" << rgbIn[1] << " B=" << rgbIn[2];
                    JTRACE_VERBOSE("SPECTRAL", oss.str());
                }

                Pipeline::DensityPixelInputs pxIn{};
                pxIn.rgb.v[0] = rgbIn[0];
                pxIn.rgb.v[1] = rgbIn[1];
                pxIn.rgb.v[2] = rgbIn[2];
                pxIn.exposureScale = exposureScale;
                pxIn.dirRuntime = &dirRT;
                pxIn.applyDirRuntime = false; // Pass A wants pre-DIR densities for correction computation.

                Pipeline::DensityPixelOutputs pxOut{};
                if (!runner.run_density_pixel(ws, pxIn, pxOut)) {
                    work.filmRaw_B[idx] = work.filmRaw_G[idx] = work.filmRaw_R[idx] = 0.0f;
                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
                    continue;
                }

                work.filmRaw_B[idx] = pxOut.filmRaw.v[0];
                work.filmRaw_G[idx] = pxOut.filmRaw.v[1];
                work.filmRaw_R[idx] = pxOut.filmRaw.v[2];

                // SPD DEBUG: Log film raw exposure (pre-log) for center pixel
                if (verboseDiagnostics && xx == center_xx && yy == center_yy) {
                    std::ostringstream oss;
                    oss << "FILM_RAW tile_pixel(" << xx << "," << yy << "): B=" << pxOut.filmRaw.v[0]
                        << " G=" << pxOut.filmRaw.v[1]
                        << " R=" << pxOut.filmRaw.v[2];
                    JTRACE_VERBOSE("SPECTRAL", oss.str());

                    // Log sensitivity curve values at key wavelengths
                    const int idx_450 = 14;  // (450-380)/5 = 14
                    const int idx_520 = 28;  // (520-380)/5 = 28
                    const int idx_650 = 54;  // (650-380)/5 = 54
                    if (!sensB_forExposure.linear.empty() && !sensG_forExposure.linear.empty() && !sensR_forExposure.linear.empty()) {
                        std::ostringstream oss1, oss2, oss3;
                        oss1 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[450nm]=" << sensB_forExposure.linear[idx_450]
                            << " G[450nm]=" << sensG_forExposure.linear[idx_450] << " R[450nm]=" << sensR_forExposure.linear[idx_450];
                        oss2 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[520nm]=" << sensB_forExposure.linear[idx_520]
                            << " G[520nm]=" << sensG_forExposure.linear[idx_520] << " R[520nm]=" << sensR_forExposure.linear[idx_520];
                        oss3 << "SENS_CURVES tile_pixel(" << xx << "," << yy << "): B[650nm]=" << sensB_forExposure.linear[idx_650]
                            << " G[650nm]=" << sensG_forExposure.linear[idx_650] << " R[650nm]=" << sensR_forExposure.linear[idx_650];
                        JTRACE_VERBOSE("SPECTRAL", oss1.str());
                        JTRACE_VERBOSE("SPECTRAL", oss2.str());
                        JTRACE_VERBOSE("SPECTRAL", oss3.str());
                    }
                }

                const float leB = pxOut.filmLogRaw.v[0];
                const float leG = pxOut.filmLogRaw.v[1];
                const float leR = pxOut.filmLogRaw.v[2];

                // Convert CMY -> YMC to match Couplers::ApplyInputLogE contract.
                const float D_Y = pxOut.negativeDensity.v[2];
                const float D_M = pxOut.negativeDensity.v[1];
                const float D_C = pxOut.negativeDensity.v[0];

                float aCorr[3];
                Couplers::ApplyInputLogE io{ { leB, leG, leR }, { D_Y, D_M, D_C } };
                Couplers::compute_logE_corrections(io, dirRT, aCorr);
                for (float& v : aCorr) {
                    if (!std::isfinite(v)) v = 0.0f;
                }
                work.corrY[idx] = aCorr[0];
                work.corrM[idx] = aCorr[1];
                work.corrC[idx] = aCorr[2];
            }
            if (aborted) {
                break;
            }
        }

        if (aborted || should_abort()) {
            trace_abort_fast("pre_blur");
            return;
        }

        // Blur corrections spatially (shared between preview + render path)
        kernelCache.clear();
        buildGaussianKernel(dirRT.spatialSigmaPixels, kernelCache);
        if (should_abort()) {
            trace_abort_fast("post_kernel");
            return;
        }
        blurChannelSeparable(work.corrY, work.tmp, work.corrYBlur, width, height, kernelCache);
        if (should_abort()) {
            trace_abort_fast("post_blur_y");
            return;
        }
        blurChannelSeparable(work.corrM, work.tmp, work.corrMBlur, width, height, kernelCache);
        if (should_abort()) {
            trace_abort_fast("post_blur_m");
            return;
        }
        blurChannelSeparable(work.corrC, work.tmp, work.corrCBlur, width, height, kernelCache);
        if (should_abort()) {
            trace_abort_fast("post_blur_c");
            return;
        }
        if (should_abort()) {
            trace_abort_fast("pre_clamp");
            return;
        }

        auto scrubClamp = [](std::vector<float>& v) {
            for (float& t : v) {
                if (!std::isfinite(t)) t = 0.0f;
                if (t < -10.0f) t = -10.0f;
                if (t > 10.0f) t = 10.0f;
            }
            };
        scrubClamp(work.corrYBlur);
        scrubClamp(work.corrMBlur);
        scrubClamp(work.corrCBlur);
    }

} // namespace SpatialDIR
