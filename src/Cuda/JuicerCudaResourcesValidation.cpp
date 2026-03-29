// Cuda/JuicerCudaResourcesValidation.cpp
//
// Included by JuicerCudaResources.cpp (single-TU split).
    bool validate_density_primitives(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
        std::lock_guard<std::mutex> lock(resources.m);
        if (resources.validatedBuildCounter == ws.buildCounter && ws.buildCounter != 0) {
            return true;
        }

        // Lightweight parity probe: sample density curves at a few canonical logE values and compare
        // against the CPU implementation (FilmProcessing.h). This is a development-only check and is
        // expected to be compiled out in shipping builds.
        static constexpr float kLogE[5] = { -3.0f, -1.0f, 0.0f, 2.0f, 4.0f };
        float outB[5] = {};
        float outG[5] = {};
        float outR[5] = {};

        cudaError_t err = ::juicer_cuda_probe_density_curve(resources.densB.x, resources.densB.y, resources.densB.n,
            ws.gammaFactorB, kLogE, 5, outB, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densB failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = ::juicer_cuda_probe_density_curve(resources.densG.x, resources.densG.y, resources.densG.n,
            ws.gammaFactorG, kLogE, 5, outG, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densG failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = ::juicer_cuda_probe_density_curve(resources.densR.x, resources.densR.y, resources.densR.n,
            ws.gammaFactorR, kLogE, 5, outR, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densR failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        float maxAbs = 0.0f;
        for (int i = 0; i < 5; ++i) {
            const float cpuB = Spectral::sample_density_at_logE(ws.densB, kLogE[i], ws.gammaFactorB);
            const float cpuG = Spectral::sample_density_at_logE(ws.densG, kLogE[i], ws.gammaFactorG);
            const float cpuR = Spectral::sample_density_at_logE(ws.densR, kLogE[i], ws.gammaFactorR);

            if (!std::isfinite(cpuB) || !std::isfinite(cpuG) || !std::isfinite(cpuR) ||
                !std::isfinite(outB[i]) || !std::isfinite(outG[i]) || !std::isfinite(outR[i])) {
                outError = "density probe produced non-finite values";
                return false;
            }

            maxAbs = std::max(maxAbs, std::fabs(outB[i] - cpuB));
            maxAbs = std::max(maxAbs, std::fabs(outG[i] - cpuG));
            maxAbs = std::max(maxAbs, std::fabs(outR[i] - cpuR));
        }

        // This primitive is expected to match exactly (same math, float-only) within a tiny epsilon.
        if (!(maxAbs <= 1e-6f)) {
            outError = "density probe mismatch: maxAbs=" + std::to_string(maxAbs);
            return false;
        }

        // Validate +/-inf sanitization before sampling (DevelopFilmStage parity).
        {
            const float kLogEInf[3] = {
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                0.0f
            };
            float outBInf[3] = {};
            float outGInf[3] = {};
            float outRInf[3] = {};

            auto cpu_sanitize_inf = [](float logE, const Spectral::Curve& c) -> float {
                if (std::isfinite(logE) || std::isnan(logE)) {
                    return logE;
                }
                float xmin = 0.0f, xmax = 0.0f;
                const size_t n = c.lambda_nm.size();
                if (n == 0 || c.linear.size() != n) {
                    return logE;
                }
                size_t begin = 0;
                while (begin < n && !std::isfinite(c.lambda_nm[begin])) {
                    ++begin;
                }
                if (begin == n) {
                    return logE;
                }
                size_t end = n - 1;
                while (end > begin && !std::isfinite(c.lambda_nm[end])) {
                    --end;
                }
                xmin = c.lambda_nm[begin];
                xmax = c.lambda_nm[end];
                if (!std::isfinite(xmin) || !std::isfinite(xmax) || !(xmax >= xmin)) {
                    return logE;
                }
                return (logE > 0.0f) ? xmax : xmin;
            };

            cudaError_t err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densB.x, resources.densB.y, resources.densB.n,
                ws.gammaFactorB, kLogEInf, 3, outBInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densB sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densG.x, resources.densG.y, resources.densG.n,
                ws.gammaFactorG, kLogEInf, 3, outGInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densG sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densR.x, resources.densR.y, resources.densR.n,
                ws.gammaFactorR, kLogEInf, 3, outRInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densR sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }

            float maxDiff = 0.0f;
            for (int i = 0; i < 3; ++i) {
                const float cpuB = Spectral::sample_density_at_logE(ws.densB, cpu_sanitize_inf(kLogEInf[i], ws.densB), ws.gammaFactorB);
                const float cpuG = Spectral::sample_density_at_logE(ws.densG, cpu_sanitize_inf(kLogEInf[i], ws.densG), ws.gammaFactorG);
                const float cpuR = Spectral::sample_density_at_logE(ws.densR, cpu_sanitize_inf(kLogEInf[i], ws.densR), ws.gammaFactorR);
                maxDiff = std::max(maxDiff, std::fabs(outBInf[i] - cpuB));
                maxDiff = std::max(maxDiff, std::fabs(outGInf[i] - cpuG));
                maxDiff = std::max(maxDiff, std::fabs(outRInf[i] - cpuR));
            }
            if (!(maxDiff <= 1e-6f)) {
                outError = "sanitize_inf_logE mismatch: maxAbs=" + std::to_string(maxDiff);
                return false;
            }
        }

        // Validate film log-raw computation parity (DevelopFilmStage::compute_log_raw).
        {
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f },
                { -0.5f, 0.0f, 2.0f },
                { std::numeric_limits<float>::quiet_NaN(), 1.0f, std::numeric_limits<float>::infinity() }
            };

            for (const auto& filmRaw : samples) {
                float gpuLogRaw[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_film_log_raw(filmRaw, gpuLogRaw, cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("film log-raw probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                constexpr float kEps = 1e-10f;
                float cpuLogRaw[3] = {
                    std::log10(std::fmax(filmRaw[0], 0.0f) + kEps),
                    std::log10(std::fmax(filmRaw[1], 0.0f) + kEps),
                    std::log10(std::fmax(filmRaw[2], 0.0f) + kEps)
                };

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    if (std::isfinite(cpuLogRaw[c]) && std::isfinite(gpuLogRaw[c])) {
                        maxDiff = std::max(maxDiff, std::fabs(cpuLogRaw[c] - gpuLogRaw[c]));
                    }
                    else if (std::isnan(cpuLogRaw[c]) != std::isnan(gpuLogRaw[c])) {
                        outError = "film log-raw NaN mismatch";
                        return false;
                    }
                    else if (std::isinf(cpuLogRaw[c]) != std::isinf(gpuLogRaw[c])) {
                        outError = "film log-raw inf mismatch";
                        return false;
                    }
                }
                if (!(maxDiff <= 1e-6f)) {
                    outError = "film log-raw mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate input RGB -> DWG conversion against the CPU implementation.
        {
            const int inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(ws.filmRaw.inputColorSpace);
            const int applyDecode = ws.filmRaw.applyCctfDecoding ? 1 : 0;
            const int applyAdapt = ws.filmRaw.applyInputChromaticAdapt ? 1 : 0;

            const float samplesIn[][3] = {
                { 0.184f, 0.184f, 0.184f },
                { 1.2f, -0.1f, 0.5f },   // includes negative input channel (must match CPU sanitize-only behavior)
                { 0.0f, 0.5f, 2.0f }
            };

            for (const auto& rgbIn : samplesIn) {
                float gpuDWG[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_convert_input_to_DWG(
                    rgbIn,
                    inputColorSpaceIndex,
                    applyDecode,
                    applyAdapt,
                    ws.filmRaw.inputRGBToXYZ.m,
                    ws.filmRaw.inputXYZAdapt.m,
                    gpuDWG,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("convert_input_rgb_to_DWG probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                float cpuDWG[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::convert_input_rgb_to_DWG(ws.filmRaw, rgbIn, cpuDWG);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuDWG[c] - cpuDWG[c]));
                }
                if (!(maxDiff <= 2e-5f)) {
                    outError = "convert_input_rgb_to_DWG mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate DIR "clamp corrected logE to curve domain" behavior (Couplers::apply_runtime_logE_with_curves clamp_to).
        {
            auto cpu_clamp_to = [](float le, const Spectral::Curve& c) -> float {
                if (c.lambda_nm.empty()) return le;
                const float xmin = c.lambda_nm.front();
                const float xmax = c.lambda_nm.back();
                if (!std::isfinite(le)) return xmin;
                return std::min(std::max(le, xmin), xmax);
            };

            const float samples[] = {
                -1000.0f,
                1000.0f,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f
            };

            for (float le : samples) {
                float outB = 0.0f, outG = 0.0f, outR = 0.0f;
                cudaError_t err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densB.x, resources.densB.n, le, &outB, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densB probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }
                err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densG.x, resources.densG.n, le, &outG, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densG probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }
                err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densR.x, resources.densR.n, le, &outR, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densR probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }

                const float cpuB = cpu_clamp_to(le, ws.densB);
                const float cpuG = cpu_clamp_to(le, ws.densG);
                const float cpuR = cpu_clamp_to(le, ws.densR);

                auto eq = [](float a, float b) -> bool {
                    if (std::isnan(a) && std::isnan(b)) return true;
                    return a == b;
                };

                if (!eq(outB, cpuB) || !eq(outG, cpuG) || !eq(outR, cpuR)) {
                    outError = "DIR clamp_to mismatch";
                    return false;
                }
            }
        }

        // Validate scan-stage spectral_to_log_xyz primitive (negative medium) against CPU.
        if (resources.scanNegative.tables.K == Spectral::gShape.K &&
            resources.scanNegative.tables.epsC &&
            resources.scanNegative.tables.Ax)
        {
            const double samples[][3] = {
                { 0.0, 0.0, 0.0 },
                { 0.5, 0.5, 0.5 },
                { 1.0, 1.0, 1.0 },
                { 0.1, 0.7, 0.3 }
            };

            for (const auto& D_norm : samples) {
                double gpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                const cudaError_t err = ::juicer_cuda_probe_scan_spectral_to_log_xyz(
                    D_norm,
                    resources.scanNegative.mediumIsNegative,
                    resources.scanNegative.min_cmy,
                    resources.scanNegative.inv_max_cmy,
                    resources.scanNegative.tables.epsC,
                    resources.scanNegative.tables.epsM,
                    resources.scanNegative.tables.epsY,
                    resources.scanNegative.tables.Ax,
                    resources.scanNegative.tables.Ay,
                    resources.scanNegative.tables.Az,
                    resources.scanNegative.tables.baseMin,
                    resources.scanNegative.tables.K,
                    resources.scanNegative.tables.hasBaseline,
                    resources.scanNegative.tables.invYn,
                    gpuLogXYZ,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("scan logXYZ probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                double cpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                Scanner::spectral_to_log_xyz(ws.negativeMediumRuntime, D_norm, cpuLogXYZ);

                double maxDiff = 0.0;
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(cpuLogXYZ[c]) || !std::isfinite(gpuLogXYZ[c])) {
                        outError = "scan logXYZ produced non-finite values";
                        return false;
                    }
                    maxDiff = std::max(maxDiff, std::fabs(cpuLogXYZ[c] - gpuLogXYZ[c]));
                }
                if (!(maxDiff <= 1e-4)) {
                    outError = "scan logXYZ mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate scan-stage spectral_to_log_xyz primitive (print medium) against CPU when available.
        if (ws.printMediumRuntime.tables &&
            resources.scanPrint.tables.K == Spectral::gShape.K &&
            resources.scanPrint.tables.epsC &&
            resources.scanPrint.tables.Ax)
        {
            const double samples[][3] = {
                { 0.0, 0.0, 0.0 },
                { 0.5, 0.5, 0.5 },
                { 1.0, 1.0, 1.0 },
                { 0.1, 0.7, 0.3 }
            };

            for (const auto& D_norm : samples) {
                double gpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                const cudaError_t err = ::juicer_cuda_probe_scan_spectral_to_log_xyz(
                    D_norm,
                    resources.scanPrint.mediumIsNegative,
                    resources.scanPrint.min_cmy,
                    resources.scanPrint.inv_max_cmy,
                    resources.scanPrint.tables.epsC,
                    resources.scanPrint.tables.epsM,
                    resources.scanPrint.tables.epsY,
                    resources.scanPrint.tables.Ax,
                    resources.scanPrint.tables.Ay,
                    resources.scanPrint.tables.Az,
                    resources.scanPrint.tables.baseMin,
                    resources.scanPrint.tables.K,
                    resources.scanPrint.tables.hasBaseline,
                    resources.scanPrint.tables.invYn,
                    gpuLogXYZ,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("scan logXYZ (print) probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                double cpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                Scanner::spectral_to_log_xyz(ws.printMediumRuntime, D_norm, cpuLogXYZ);

                double maxDiff = 0.0;
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(cpuLogXYZ[c]) || !std::isfinite(gpuLogXYZ[c])) {
                        outError = "scan logXYZ (print) produced non-finite values";
                        return false;
                    }
                    maxDiff = std::max(maxDiff, std::fabs(cpuLogXYZ[c] - gpuLogXYZ[c]));
                }
                if (!(maxDiff <= 1e-4)) {
                    outError = "scan logXYZ (print) mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate table-based (S_inv) exposure primitive against CPU (forced non-Hanatos path).
        if (ws.spdReady && resources.tablesAx && resources.tablesAy && resources.tablesAz && resources.tablesK == Spectral::gShape.K) {
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f }, // mid-gray
                { 0.9f, 0.1f, 0.1f },       // red-ish
                { 0.05f, 0.2f, 0.9f }       // blue-ish
            };

            for (const auto& rgbDWG : samples) {
                float gpuE[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_tables_layer_exposures(
                    rgbDWG,
                    resources.spdSInv,
                    resources.refIllumWhiteXYZ,
                    resources.tablesAx,
                    resources.tablesAy,
                    resources.tablesAz,
                    resources.tablesK,
                    resources.sensB.y,
                    resources.sensG.y,
                    resources.sensR.y,
                    1.0f,
                    1,
                    gpuE,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("tables exposure probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                std::vector<float> Ee;
                Spectral::reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, ws.tablesRef, ws.spdSInv, Ee);
                float cpuE[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::layerExposures_from_sceneSPD_with_curves(Ee, ws.sensB, ws.sensG, ws.sensR, cpuE, 1.0f, true);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuE[c] - cpuE[c]));
                }
                if (!(maxDiff <= 2e-4f)) {
                    outError = "tables exposure mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // If Hanatos is available and uploaded, validate the SPD→exposure integration primitive on a
        // few representative DWG RGB samples.
        if (resources.hanatosLut && resources.hanatosN > 0) {
            const float refWhite[3] = {
                ws.tablesRef.refIllumWhiteXYZ[0],
                ws.tablesRef.refIllumWhiteXYZ[1],
                ws.tablesRef.refIllumWhiteXYZ[2]
            };
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f }, // mid-gray
                { 0.9f, 0.1f, 0.1f },       // red-ish
                { 0.05f, 0.2f, 0.9f }       // blue-ish
            };

            for (const auto& rgbDWG : samples) {
                float gpuE[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_hanatos_layer_exposures(
                    rgbDWG,
                    resources.hanatosLut,
                    resources.hanatosN,
                    refWhite,
                    resources.sensB.y,
                    resources.sensG.y,
                    resources.sensR.y,
                    Spectral::kNumSamples,
                    1.0f,
                    0,
                    gpuE,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("Hanatos exposure probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                // CPU reference: match Spectral::rgbDWG_to_layerExposures_from_tables_with_curves
                // semantics for Hanatos path (applyDeltaLambda = false).
                std::vector<float> Ee;
                Spectral::reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee, refWhite);
                float cpuE[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::layerExposures_from_sceneSPD_with_curves(Ee, ws.sensB, ws.sensG, ws.sensR, cpuE, 1.0f, false);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuE[c] - cpuE[c]));
                }
                if (!(maxDiff <= 2e-4f)) {
                    outError = "Hanatos exposure mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        resources.validatedBuildCounter = ws.buildCounter;
        return true;
#else
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        (void)outError;
        return true;
#endif
#endif
    }

    bool validate_print_primitives(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        float midgrayFactor,
        void* cudaStreamOpaque,
        std::string& outError)
    {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)prt;
        (void)prm;
        (void)midgrayFactor;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)
        // Cache: avoid re-running the (expensive) CPU-vs-GPU probe every frame. This is for debug-only
        // validation; correctness is still enforced when the key changes.
        std::uint64_t paramsHash = Hash::kFnvOffset;
        auto hash_u32 = [&](std::uint32_t v) {
            Hash::hash_bytes_update(paramsHash, &v, sizeof(v));
        };
        auto hash_u64 = [&](std::uint64_t v) {
            Hash::hash_bytes_update(paramsHash, &v, sizeof(v));
        };
        auto hash_f32 = [&](float v) {
            float vv = v;
            if (vv == 0.0f) {
                vv = 0.0f; // canonicalize -0.0f to +0.0f
            }
            std::uint32_t bits = 0;
            std::memcpy(&bits, &vv, sizeof(bits));
            hash_u32(bits);
        };
        auto hash_bool = [&](bool v) {
            const std::uint32_t b = v ? 1U : 0U;
            hash_u32(b);
        };

        hash_u64(ws.buildCounter);
        hash_f32(prm.exposure);
        hash_f32(prm.preflashExposure);
        hash_f32(prm.yFilter);
        hash_f32(prm.mFilter);
        hash_f32(prm.cFilter);
        hash_u64((prt.neutralFilterHash != 0) ? prt.neutralFilterHash : Print::kDefaultNeutralFilterHash);
        hash_bool(prm.exposureCompensationEnabled);
        hash_f32(prm.exposureCompensationScale);
        hash_f32(midgrayFactor);

        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (resources.validatedPrintBuildCounter == ws.buildCounter &&
                resources.validatedPrintParamsHash == paramsHash) {
                outError.clear();
                return true;
            }
        }

        // Ensure current print illuminant (filtered by print params) is available on device.
        {
            std::string illumError;
            if (!ensure_print_illuminant_filtered(resources, ws, prt, prm, cudaStreamOpaque, illumError)) {
                outError = std::string("ensure_print_illuminant_filtered failed: ") + illumError;
                return false;
            }
        }

        static constexpr float kNegCmySamples[][3] = {
            { 0.0f, 0.0f, 0.0f },
            { 0.2f, 0.3f, 0.4f },
            { 0.5f, 0.5f, 0.5f },
            { 1.0f, 0.8f, 0.6f },
            { 2.0f, 1.5f, 1.0f },
            { 3.0f, 3.0f, 3.0f }
        };
        constexpr int kCount = static_cast<int>(sizeof(kNegCmySamples) / sizeof(kNegCmySamples[0]));

        const float kMid = (std::isfinite(midgrayFactor) && midgrayFactor > 0.0f) ? midgrayFactor : 1.0f;

        float cpuOut[kCount * 3] = {};
        Pipeline::PrintPipelineScratch scratch{};
        for (int i = 0; i < kCount; ++i) {
            Pipeline::ExposePrintInputs in{};
            in.printRuntime = &prt;
            in.printParams = &prm;
            in.negativeDensity.v[0] = kNegCmySamples[i][0];
            in.negativeDensity.v[1] = kNegCmySamples[i][1];
            in.negativeDensity.v[2] = kNegCmySamples[i][2];
            in.midgrayFactor = kMid;

            Pipeline::ExposePrintOutputs ex{};
            if (!Pipeline::ExposePrintStage::run(ws, in, ex, scratch)) {
                outError = "CPU ExposePrintStage::run failed";
                return false;
            }

            Pipeline::DevelopPrintInputs din{};
            din.printRuntime = &prt;
            din.printLogRaw = ex.printLogRaw;
            Pipeline::DevelopPrintOutputs dout{};
            if (!Pipeline::DevelopPrintStage::run(din, dout)) {
                outError = "CPU DevelopPrintStage::run failed";
                return false;
            }

            cpuOut[i * 3 + 0] = dout.printDensity.v[0];
            cpuOut[i * 3 + 1] = dout.printDensity.v[1];
            cpuOut[i * 3 + 2] = dout.printDensity.v[2];
        }

        JuicerCuda::PipelineRunParams run{};
        run.printExpose.active = 1;
        run.printExpose.printExposure = prm.exposure;
        run.printExpose.printPreflashExposure = prm.preflashExposure;
        run.printExpose.printMidgrayFactor = kMid;

        float inNeg[kCount * 3] = {};
        for (int i = 0; i < kCount; ++i) {
            inNeg[i * 3 + 0] = kNegCmySamples[i][0];
            inNeg[i * 3 + 1] = kNegCmySamples[i][1];
            inNeg[i * 3 + 2] = kNegCmySamples[i][2];
        }
        float gpuOut[kCount * 3] = {};

        {
            std::lock_guard<std::mutex> lock(resources.m);
            run.printExpose.negTables.epsC = resources.scanNegative.tables.epsC;
            run.printExpose.negTables.epsM = resources.scanNegative.tables.epsM;
            run.printExpose.negTables.epsY = resources.scanNegative.tables.epsY;
            run.printExpose.negTables.Ax = resources.scanNegative.tables.Ax;
            run.printExpose.negTables.Ay = resources.scanNegative.tables.Ay;
            run.printExpose.negTables.Az = resources.scanNegative.tables.Az;
            run.printExpose.negTables.baseMin = resources.scanNegative.tables.baseMin;
            run.printExpose.negTables.K = resources.scanNegative.tables.K;
            run.printExpose.negTables.hasBaseline = resources.scanNegative.tables.hasBaseline;
            run.printExpose.negTables.invYn = resources.scanNegative.tables.invYn;
            run.printExpose.negTables.mediumIsNegative = resources.scanNegative.mediumIsNegative;
            for (int j = 0; j < 3; ++j) {
                run.printExpose.negTables.min_cmy[j] = resources.scanNegative.min_cmy[j];
                run.printExpose.negTables.inv_max_cmy[j] = resources.scanNegative.inv_max_cmy[j];
            }

            run.printExpose.printIllumFiltered = resources.printIllumFiltered;
            run.printExpose.printIllumK = resources.printIllumK;
            run.printExpose.printSensC = { resources.printSensC.x, resources.printSensC.y, resources.printSensC.n, resources.printSensC.domainBegin, resources.printSensC.domainEnd };
            run.printExpose.printSensM = { resources.printSensM.x, resources.printSensM.y, resources.printSensM.n, resources.printSensM.domainBegin, resources.printSensM.domainEnd };
            run.printExpose.printSensY = { resources.printSensY.x, resources.printSensY.y, resources.printSensY.n, resources.printSensY.domainBegin, resources.printSensY.domainEnd };
            run.printDevelop.printDcC = { resources.printDcC.x, resources.printDcC.y, resources.printDcC.n, resources.printDcC.domainBegin, resources.printDcC.domainEnd };
            run.printDevelop.printDcM = { resources.printDcM.x, resources.printDcM.y, resources.printDcM.n, resources.printDcM.domainBegin, resources.printDcM.domainEnd };
            run.printDevelop.printDcY = { resources.printDcY.x, resources.printDcY.y, resources.printDcY.n, resources.printDcY.domainBegin, resources.printDcY.domainEnd };
            run.printDevelop.printGammaC = resources.printGammaC;
            run.printDevelop.printGammaM = resources.printGammaM;
            run.printDevelop.printGammaY = resources.printGammaY;
            for (int j = 0; j < 3; ++j) {
                run.printExpose.printPreflashRaw[j] = resources.printPreflashRaw[j];
            }
        }

        const cudaError_t probeErr = ::juicer_cuda_probe_print_pipeline(inNeg, kCount, &run, gpuOut, cudaStreamOpaque);
        if (probeErr != cudaSuccess) {
            outError = std::string("GPU print pipeline probe failed: ") + (cudaGetErrorString(probeErr) ? cudaGetErrorString(probeErr) : "(unknown)");
            return false;
        }

        float maxAbs = 0.0f;
        for (int i = 0; i < kCount; ++i) {
            const float* cpu = cpuOut + i * 3;
            const float* gpu = gpuOut + i * 3;
            bool mismatch = false;
            for (int c = 0; c < 3; ++c) {
                const float a = cpu[c];
                const float b = gpu[c];
                if (std::isfinite(a) && std::isfinite(b)) {
                    maxAbs = std::max(maxAbs, std::fabs(a - b));
                } else if (!(std::isnan(a) && std::isnan(b))) {
                    mismatch = true;
                }
            }
            if (mismatch) {
                std::ostringstream oss;
                oss << "CUDA print validation non-finite mismatch"
                    << " sample=" << i
                    << " neg=[" << kNegCmySamples[i][0] << "," << kNegCmySamples[i][1] << "," << kNegCmySamples[i][2] << "]"
                    << " cpu=[" << cpu[0] << "," << cpu[1] << "," << cpu[2] << "]"
                    << " gpu=[" << gpu[0] << "," << gpu[1] << "," << gpu[2] << "]"
                    << " printExposure=" << run.printExpose.printExposure
                    << " preflash=" << run.printExpose.printPreflashExposure
                    << " midgray=" << run.printExpose.printMidgrayFactor
                    << " illumK=" << run.printExpose.printIllumK
                    << " illumPtr=" << (run.printExpose.printIllumFiltered ? 1 : 0)
                    << " sensC.n=" << run.printExpose.printSensC.n
                    << " sensM.n=" << run.printExpose.printSensM.n
                    << " sensY.n=" << run.printExpose.printSensY.n
                    << " dcC.n=" << run.printDevelop.printDcC.n
                    << " dcM.n=" << run.printDevelop.printDcM.n
                    << " dcY.n=" << run.printDevelop.printDcY.n;
                JTRACE("CUDA", oss.str());
                outError = "print pipeline probe produced non-finite mismatch";
                return false;
            }
        }

        if (!(maxAbs <= 1e-3f)) {
            outError = "print pipeline mismatch: maxAbs=" + std::to_string(maxAbs);
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(resources.m);
            resources.validatedPrintBuildCounter = ws.buildCounter;
            resources.validatedPrintParamsHash = paramsHash;
        }

        return true;
#else
        (void)resources;
        (void)ws;
        (void)prt;
        (void)prm;
        (void)midgrayFactor;
        (void)cudaStreamOpaque;
        (void)outError;
        return true;
#endif
#endif
    }
