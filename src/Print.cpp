#include "Print.h"
#include "Scanner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>
#include "GaussianSciPy.h"

namespace Print {
    namespace {

        using FloatPairs = std::vector<std::pair<float, float>>;
        using DoublePairs = std::vector<std::pair<double, double>>;

        inline bool is_finite(double value);

        template <typename PairT>
        inline bool pair_has_finite_components(const PairT& p) {
            return is_finite(p.first) && is_finite(p.second);
        }

        inline bool is_nonzero_finite(double value) {
            return is_finite(value) && value != 0.0;
        }

        inline bool is_finite(double value) {
            return std::isfinite(value);
        }

        inline void accumulate_if_finite(double value, double& sum, int& count) {
            if (!is_finite(value)) {
                return;
            }
            sum += value;
            ++count;
        }

        void reset_profile_state(Profile& out, Runtime* runtime) {
            const float nanDensity = std::numeric_limits<float>::quiet_NaN();
            out.gammaFactor = {1.0f, 1.0f, 1.0f};
            out.hasMidNeutralDensity = false;
            out.midNeutralDensity = {nanDensity, nanDensity, nanDensity};
            out.hasMidNeutralLogE = false;
            out.midNeutralLogE = {nanDensity, nanDensity, nanDensity};
            out.hasSensitivityCorrection = false;
            out.sensitivityNeutralCorr = {1.0f, 1.0f, 1.0f};
            out.glareRemoved = false;
            out.hasGlareCompensation = false;
            out.glareCompensationFactor = 0.0f;
            out.glareCompensationDensity = 1.2f;
            out.glareCompensationTransition = 0.3f;
            out.glare = Profiles::ProfileGlare{};

            if (runtime) {
                runtime->hasMidNeutralDensity = false;
                runtime->midNeutralDensity = {nanDensity, nanDensity, nanDensity};
                runtime->hasMidNeutralLogE = false;
                runtime->midNeutralLogE = {nanDensity, nanDensity, nanDensity};
                runtime->hasSensitivityCorrection = false;
                runtime->sensitivityNeutralCorr = {1.0f, 1.0f, 1.0f};
                runtime->referenceIlluminant.clear();
                runtime->viewingIlluminant.clear();
                runtime->glare = Profiles::ProfileGlare{};
                runtime->densityCurvesRaw = DensityCurves{};
            }
        }

        struct JsonProfileContext {
            bool hasProfile = false;
            Profiles::SpektrafilmProfileJson profile;
        };

        JsonProfileContext load_json_profile(const JuicerAssets::SelectedPrintProfileAsset& asset) {
            JsonProfileContext ctx;
            ctx.hasProfile = JuicerProcess::root().assets().load_spektrafilm_print_profile(asset, ctx.profile);
            return ctx;
        }

        bool channel_has_finite_samples(const FloatPairs& pairs, size_t minimumFiniteSamples = 4u) {
            size_t finiteCount = 0;
            const std::pair<float, float>* sampleData = pairs.data();
            const size_t sampleCount = pairs.size();
            for (size_t i = 0; i < sampleCount; ++i, ++sampleData) {
                if (is_finite(sampleData->second)) {
                    if (++finiteCount >= minimumFiniteSamples) {
                        return true;
                    }
                }
            }
            return false;
        }

        bool json_profile_has_complete_print_baseline_min(const JsonProfileContext& ctx) {
            return ctx.hasProfile &&
                   channel_has_finite_samples(ctx.profile.baseDensityMin);
        }

        bool json_profile_has_complete_print_baseline_mid(const JsonProfileContext& ctx) {
            return ctx.hasProfile &&
                   channel_has_finite_samples(ctx.profile.baseDensityMid);
        }

        struct JsonDyeSensitivityOverrideTargets {
            FloatPairs& c_eps;
            FloatPairs& m_eps;
            FloatPairs& y_eps;
            FloatPairs& r_sens;
            FloatPairs& g_sens;
            FloatPairs& b_sens;
            bool& usedJsonEps;
            bool& usedJsonSens;
        };

        void apply_json_dye_and_sensitivity_overrides(
            const JsonProfileContext& ctx,
            Profile& out,
            Runtime* runtime,
            const JsonDyeSensitivityOverrideTargets& targets) {
            targets.usedJsonEps = false;
            targets.usedJsonSens = false;
            if (!ctx.hasProfile) {
                return;
            }

            const auto& profileJson = ctx.profile;
            const bool jsonDyeC = channel_has_finite_samples(profileJson.dyeC);
            const bool jsonDyeM = channel_has_finite_samples(profileJson.dyeM);
            const bool jsonDyeY = channel_has_finite_samples(profileJson.dyeY);
            if (jsonDyeC && jsonDyeM && jsonDyeY) {
                targets.c_eps = profileJson.dyeC;
                targets.m_eps = profileJson.dyeM;
                targets.y_eps = profileJson.dyeY;
                targets.usedJsonEps = true;
            } else if (!profileJson.dyeC.empty() || !profileJson.dyeM.empty() || !profileJson.dyeY.empty()) {
                std::ostringstream warn;
                warn << "PROFILE_LOAD JSON dye overrides skipped (finite samples C/M/Y="
                     << (jsonDyeC ? "ok" : "missing") << "/"
                     << (jsonDyeM ? "ok" : "missing") << "/"
                     << (jsonDyeY ? "ok" : "missing") << ")";
                JTRACE("PRINT", warn.str());
            }

            const bool jsonSensR = channel_has_finite_samples(profileJson.logSensR);
            const bool jsonSensG = channel_has_finite_samples(profileJson.logSensG);
            const bool jsonSensB = channel_has_finite_samples(profileJson.logSensB);
            if (jsonSensR)
                targets.r_sens = profileJson.logSensR;
            if (jsonSensG)
                targets.g_sens = profileJson.logSensG;
            if (jsonSensB)
                targets.b_sens = profileJson.logSensB;
            targets.usedJsonSens = jsonSensR && jsonSensG && jsonSensB;
            if ((jsonSensR || jsonSensG || jsonSensB) && !(jsonSensR && jsonSensG && jsonSensB)) {
                std::ostringstream warn;
                warn << "PROFILE_LOAD partial JSON sensitivity override (finite R/G/B="
                     << (jsonSensR ? "ok" : "missing") << "/"
                     << (jsonSensG ? "ok" : "missing") << "/"
                     << (jsonSensB ? "ok" : "missing") << ")";
                JTRACE("PRINT", warn.str());
            }

            if (profileJson.hasGammaFactor) {
                out.gammaFactor = profileJson.gammaFactor;
            }

            if (profileJson.hasGlareCompensation) {
                out.glareCompensationFactor = profileJson.glareCompensationFactor;
                out.glareCompensationDensity = profileJson.glareCompensationDensity;
                out.glareCompensationTransition = profileJson.glareCompensationTransition;
            }
            out.hasGlareCompensation = profileJson.hasGlareCompensation;

            if (!profileJson.densityMidNeutral.empty()) {
                bool updated = false;
                if (profileJson.densityMidNeutral.size() == 1) {
                    const float v = profileJson.densityMidNeutral[0];
                    if (is_finite(v)) {
                        out.midNeutralDensity = {v, v, v};
                        updated = true;
                    }
                } else {
                    const size_t count = std::min<size_t>(out.midNeutralDensity.size(), profileJson.densityMidNeutral.size());
                    const float* midNeutralData = profileJson.densityMidNeutral.data();
                    float* outMidNeutralData = out.midNeutralDensity.data();
                    for (size_t i = 0; i < count; ++i, ++midNeutralData, ++outMidNeutralData) {
                        const float v = *midNeutralData;
                        if (is_finite(v)) {
                            *outMidNeutralData = v;
                            updated = true;
                        }
                    }

                    if (is_finite(out.midNeutralDensity[0])) {
                        float* outMidNeutralData = out.midNeutralDensity.data();
                        const float firstFiniteDensity = *outMidNeutralData;
                        ++outMidNeutralData;
                        for (int i = 1; i < 3; ++i, ++outMidNeutralData) {
                            if (!is_finite(*outMidNeutralData)) {
                                *outMidNeutralData = firstFiniteDensity;
                            }
                        }
                    }
                }

                const bool allFinite = std::all_of(
                    out.midNeutralDensity.begin(),
                    out.midNeutralDensity.end(),
                    [](float x) {
                        return is_finite(x);
                    });

                out.hasMidNeutralDensity = updated && allFinite;
                if (runtime) {
                    runtime->midNeutralDensity = out.midNeutralDensity;
                    runtime->hasMidNeutralDensity = out.hasMidNeutralDensity;
                }
            }

            if (!profileJson.referenceIlluminant.empty()) {
                out.referenceIlluminant = profileJson.referenceIlluminant;
                if (runtime) {
                    runtime->referenceIlluminant = profileJson.referenceIlluminant;
                }
            }
            if (!profileJson.viewingIlluminant.empty()) {
                out.viewingIlluminant = profileJson.viewingIlluminant;
                if (runtime) {
                    runtime->viewingIlluminant = profileJson.viewingIlluminant;
                }
            }
        }

        void pad_pairs_to_shape_domain(
            FloatPairs& pairs,
            std::optional<float> padMinValue = std::optional<float>(0.0f),
            std::optional<float> padMaxValue = std::optional<float>(0.0f)) {
            if (pairs.empty()) {
                return;
            }

            const float Lmin = Spectral::gShape.lambdaMin;
            const float Lmax = Spectral::gShape.lambdaMax;

            std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });

            // Trim leading/trailing samples without finite data.
            const auto firstFinite = std::find_if(pairs.begin(), pairs.end(), [](const auto& p) {
                return pair_has_finite_components(p);
            });
            if (firstFinite == pairs.end()) {
                pairs.clear();
                return;
            }
            const auto lastFinite = std::find_if(pairs.rbegin(), pairs.rend(), [](const auto& p) {
                                        return pair_has_finite_components(p);
                                    }).base();

            FloatPairs trimmed;
            trimmed.reserve(static_cast<size_t>(std::distance(firstFinite, lastFinite)));
            trimmed.insert(trimmed.end(), firstFinite, lastFinite);
            pairs = std::move(trimmed);
            if (pairs.empty()) {
                return;
            }

            // Ensure remaining wavelengths and dye samples are finite.
            pairs.erase(std::remove_if(pairs.begin(), pairs.end(), [](const auto& p) {
                            return !pair_has_finite_components(p);
                        }),
                        pairs.end());
            if (pairs.empty()) {
                return;
            }

            const float firstL = pairs.front().first;
            const float lastL = pairs.back().first;
            const float firstV = pairs.front().second;
            const float lastV = pairs.back().second;
            const float padMin = padMinValue.has_value() ? *padMinValue : firstV;
            const float padMax = padMaxValue.has_value() ? *padMaxValue : lastV;
            constexpr float kEndpointEps = 1e-3f;

            if (std::fabs(firstL - Lmin) > kEndpointEps && firstL > Lmin) {
                pairs.insert(pairs.begin(), {Lmin, padMin});
            }
            if (std::fabs(lastL - Lmax) > kEndpointEps && lastL < Lmax) {
                pairs.emplace_back(Lmax, padMax);
            }

            // Re-sort if we inserted endpoints (stable so first-occurrence duplicate semantics match agx).
            std::stable_sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });

            // Collapse duplicate wavelengths keeping the first finite sample.
            pairs.erase(std::unique(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                            constexpr float kEps = 1e-5f;
                            return std::fabs(a.first - b.first) <= kEps;
                        }),
                        pairs.end());
        }

        bool build_eps_curves(const FloatPairs& c_eps,
                              const FloatPairs& m_eps,
                              const FloatPairs& y_eps,
                              Profile& out) {
            auto build_eps_curve = [](Spectral::Curve& dst, const FloatPairs& pairs) -> bool {
                // JSON profiles are authored on the reference axis and may contain NaNs at spectral edges.
                if (Spectral::samples_follow_reference_axis(pairs)) {
                    return Spectral::build_curve_on_reference_axis_from_aligned_pairs(
                        dst, pairs, /*clampNegative*/ false);
                }

                // CSV input is expected to be finite; keep existing pad/resample behavior.
                FloatPairs sanitized = pairs;
                pad_pairs_to_shape_domain(sanitized);
                const std::vector<std::pair<float, float>> resampled =
                    Spectral::resample_pairs_linear_to_reference_axis(sanitized);
                return Spectral::build_curve_on_reference_axis_from_aligned_pairs(
                    dst, resampled, /*clampNegative*/ false);
            };

            const bool cOk = build_eps_curve(out.epsC, c_eps);
            const bool mOk = build_eps_curve(out.epsM, m_eps);
            const bool yOk = build_eps_curve(out.epsY, y_eps);
            return cOk && mOk && yOk;
        }

        template <typename OutPairs, typename InPairs>
        OutPairs cast_pairs(const InPairs& in) {
            using OutScalar = typename OutPairs::value_type::first_type;
            OutPairs out;
            const size_t count = in.size();
            out.reserve(count);
            const typename InPairs::value_type* input = in.data();
            const typename InPairs::value_type* const inputEnd = input + count;
            for (; input != inputEnd; ++input) {
                out.emplace_back(
                    static_cast<OutScalar>(input->first),
                    static_cast<OutScalar>(input->second));
            }
            return out;
        }

        DoublePairs promote_pairs(const FloatPairs& in) {
            return cast_pairs<DoublePairs>(in);
        }

        FloatPairs demote_pairs(const DoublePairs& in) {
            return cast_pairs<FloatPairs>(in);
        }

        DensityCurves load_density_curves(const JsonProfileContext& ctx, const std::string& profileKey) {
            DensityCurves curves;
            const bool traceInfo = JTRACE_ENABLED(1);
            if (ctx.hasProfile) {
                const auto& profileJson = ctx.profile;
                const auto& jsonR = profileJson.densityCurveR;
                const auto& jsonG = profileJson.densityCurveG;
                const auto& jsonB = profileJson.densityCurveB;
                if (!jsonR.empty() && !jsonG.empty() && !jsonB.empty()) {
                    curves.cyan = promote_pairs(jsonR);    // R -> C
                    curves.magenta = promote_pairs(jsonG); // G -> M
                    curves.yellow = promote_pairs(jsonB);  // B -> Y
                    curves.usedJson = true;
                    if (traceInfo) {
                        std::string msg;
                        msg.reserve(128 + profileKey.size());
                        msg = "PROFILE_LOAD density curves from JSON profile key '";
                        msg += profileKey;
                        msg += "' samples C/M/Y=";
                        msg += std::to_string(jsonR.size());
                        msg += "/";
                        msg += std::to_string(jsonG.size());
                        msg += "/";
                        msg += std::to_string(jsonB.size());
                        JTRACE("PRINT",
                               msg);
                    }
                } else if (traceInfo) {
                    JTRACE("PRINT",
                           "PROFILE_LOAD missing JSON density curves for profile key '" + profileKey + "'");
                }
            } else if (traceInfo) {
                JTRACE("PRINT",
                       "PROFILE_LOAD no JSON profile for density curves for profile key '" + profileKey + "'");
            }

            return curves;
        }

        double sample_curve_at(const DoublePairs& curve, double x) {
            const size_t n = curve.size();
            if (n == 0)
                return 0.0;
            if (n == 1 || !is_finite(x))
                return curve.front().second;
            if (x <= curve.front().first)
                return curve.front().second;
            if (x >= curve.back().first)
                return curve.back().second;
            const auto upper = std::lower_bound(
                curve.begin() + 1,
                curve.end(),
                x,
                [](const std::pair<double, double>& sample, double value) {
                    return sample.first < value;
                });
            const auto& hi = *upper;
            const auto& lo = *(upper - 1);
            const double x0 = lo.first;
            const double x1 = hi.first;
            const double y0 = lo.second;
            const double y1 = hi.second;
            const double span = x1 - x0;
            if (!(span > 0.0)) {
                return y1;
            }
            const double t = (x - x0) / span;
            return y0 + t * (y1 - y0);
        }

        bool find_logE_for_density(const DoublePairs& curve, double targetDensity, double& outLogE) {
            const size_t curveSize = curve.size();
            if (curveSize == 0) {
                outLogE = 0.0;
                return false;
            }
            const std::pair<double, double>* curveData = curve.data();
            const double firstLogE = curveData[0].first;
            auto fail_with_first_logE = [&]() -> bool {
                outLogE = firstLogE;
                return false;
            };

            if (!is_finite(targetDensity)) {
                return fail_with_first_logE();
            }

            double yMin = std::numeric_limits<double>::infinity();
            double yMax = -std::numeric_limits<double>::infinity();
            size_t minIndex = 0;
            size_t maxIndex = 0;
            bool anyFinite = false;

            const std::pair<double, double>* sampleIt = curveData;
            for (size_t i = 0; i < curveSize; ++i, ++sampleIt) {
                const double y = sampleIt->second;
                if (!is_finite(y)) {
                    continue;
                }
                if (!anyFinite) {
                    yMin = yMax = y;
                    minIndex = maxIndex = i;
                    anyFinite = true;
                    continue;
                }
                if (y < yMin) {
                    yMin = y;
                    minIndex = i;
                }
                if (y > yMax) {
                    yMax = y;
                    maxIndex = i;
                }
            }

            if (!anyFinite) {
                return fail_with_first_logE();
            }

            const double xAtMin = curveData[minIndex].first;
            const double xAtMax = curveData[maxIndex].first;

            if (targetDensity <= yMin) {
                outLogE = xAtMin;
                return true;
            }
            if (targetDensity >= yMax) {
                outLogE = curveData[curveSize - 1].first;
                return true;
            }

            const std::pair<double, double>* prev = curveData;
            const std::pair<double, double>* curr = curveData + 1;
            for (size_t i = 1; i < curveSize; ++i, ++prev, ++curr) {
                const double y0 = prev->second;
                const double y1 = curr->second;
                if (!is_finite(y0) || !is_finite(y1)) {
                    continue;
                }
                if ((targetDensity >= y0 && targetDensity <= y1) ||
                    (targetDensity >= y1 && targetDensity <= y0)) {
                    const double x0 = prev->first;
                    const double x1 = curr->first;
                    const double t = (y1 == y0) ? 0.0 : (targetDensity - y0) / (y1 - y0);
                    outLogE = x0 + t * (x1 - x0);
                    return true;
                }
            }

            outLogE = xAtMax;
            return false;
        }

        bool build_density_curves(Profile& out,
                                  const DoublePairs& c_dc,
                                  const DoublePairs& m_dc,
                                  const DoublePairs& y_dc) {
            auto build_channel = [&](Spectral::Curve& dst, const DoublePairs& src, const char* label) -> bool {
                if (src.empty()) {
                    dst.lambda_nm.clear();
                    dst.linear.clear();
                    return false;
                }

                // agx-emulsion parity: preserve NaNs in density curves (toe region). These NaNs must
                // propagate through interpolation and only later become 0 transmitted light.
                FloatPairs pairs = demote_pairs(src);
                std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                    return a.first < b.first;
                });

                constexpr float kDedupEps = 1e-6f;
                FloatPairs deduped;
                deduped.reserve(pairs.size());
                const std::pair<float, float>* pairData = pairs.data();
                const size_t pairCount = pairs.size();
                for (size_t i = 0; i < pairCount; ++i, ++pairData) {
                    const auto& p = *pairData;
                    if (!deduped.empty() && std::fabs(p.first - deduped.back().first) <= kDedupEps) {
                        deduped.back().second = p.second;
                        continue;
                    }
                    deduped.emplace_back(p);
                }

                const bool ok = Spectral::build_curve_on_log_exposure_axis(dst, deduped, /*clampNegative*/ false);
                if (!ok) {
                    dst.lambda_nm.clear();
                    dst.linear.clear();
                    std::ostringstream warn;
                    warn << "WARN: density curve build failed"
                         << " (" << label << " samples=" << deduped.size() << ")";
                    JTRACE("PRINT", warn.str());
                }
                return ok;
            };

            const bool cOk = build_channel(out.dcC, c_dc, "C");
            const bool mOk = build_channel(out.dcM, m_dc, "M");
            const bool yOk = build_channel(out.dcY, y_dc, "Y");
            if (!(cOk && mOk && yOk)) {
                std::ostringstream warn;
                warn << "WARN: density curve assembly failed"
                     << " (C=" << (cOk ? "ok" : "fail")
                     << ", M=" << (mOk ? "ok" : "fail")
                     << ", Y=" << (yOk ? "ok" : "fail")
                     << ")";
                JTRACE("PRINT", warn.str());
            }
            return cOk && mOk && yOk;
        }

        bool invert_density_curve_at_target(const Spectral::Curve& curve, float targetDensity, float& outLogE) {
            const size_t n = curve.lambda_nm.size();
            if (n == 0 || curve.linear.size() != n || !is_finite(targetDensity)) {
                return false;
            }

            size_t domainBegin = 0;
            while (domainBegin < n && (!is_finite(curve.lambda_nm[domainBegin]) || !is_finite(curve.linear[domainBegin]))) {
                ++domainBegin;
            }
            if (domainBegin >= n) {
                return false;
            }

            size_t domainEnd = n - 1;
            while (domainEnd > domainBegin && (!is_finite(curve.lambda_nm[domainEnd]) || !is_finite(curve.linear[domainEnd]))) {
                --domainEnd;
            }
            if (domainEnd < domainBegin) {
                return false;
            }

            struct Sample {
                float x;
                float y;
            };
            std::vector<Sample> samples;
            samples.reserve(domainEnd - domainBegin + 1);
            const float eps = 1e-6f;
            for (size_t i = domainBegin; i <= domainEnd; ++i) {
                const float x = curve.lambda_nm[i];
                const float y = curve.linear[i];
                if (!is_finite(x) || !is_finite(y)) {
                    continue;
                }
                if (!samples.empty() && std::fabs(samples.back().x - x) <= eps) {
                    samples.back().y = std::max(samples.back().y, y);
                    continue;
                }
                samples.emplace_back(Sample{x, y});
            }

            if (samples.empty()) {
                return false;
            }

            const float yMin = samples.front().y;
            const float yMax = samples.back().y;
            if (!is_finite(yMin) || !is_finite(yMax)) {
                return false;
            }

            if (targetDensity <= yMin) {
                outLogE = samples.front().x;
                return true;
            }
            if (targetDensity >= yMax) {
                outLogE = samples.back().x;
                return true;
            }

            const size_t sampleCount = samples.size();
            const Sample* sampleData = samples.data();
            for (size_t i = 1; i < sampleCount; ++i) {
                const Sample& prev = sampleData[i - 1];
                const Sample& cur = sampleData[i];
                if (!(cur.y >= prev.y)) {
                    continue;
                }
                if (targetDensity <= cur.y) {
                    const float dy = cur.y - prev.y;
                    if (!(dy > 0.0f) || !is_finite(dy)) {
                        outLogE = cur.x;
                        return true;
                    }
                    const float t = std::clamp((targetDensity - prev.y) / dy, 0.0f, 1.0f);
                    outLogE = prev.x + t * (cur.x - prev.x);
                    return is_finite(outLogE);
                }
            }

            outLogE = samples.back().x;
            return true;
        }

        std::vector<double> gaussian_filter_reflect(const std::vector<double>& input, double sigmaSamples) {
            if (input.empty()) {
                return input;
            }

            const int radius = JuicerGaussian::scipy_gaussian_radius(sigmaSamples, 4.0);
            if (radius == 0) {
                return input;
            }

            const int size = static_cast<int>(input.size());
            const size_t radiusSize = static_cast<size_t>(radius);
            std::vector<double> kernel(radiusSize * 2u + 1u);
            const double sigma = sigmaSamples;
            const double invTwoSigmaSq = 1.0 / (2.0 * sigma * sigma);
            double norm = 0.0;
            size_t kernelIndex = 0;
            for (int k = -radius; k <= radius; ++k, ++kernelIndex) {
                const double kDouble = static_cast<double>(k);
                const double w = std::exp(-(kDouble * kDouble) * invTwoSigmaSq);
                kernel[kernelIndex] = w;
                norm += w;
            }
            if (!(norm > 0.0)) {
                return input;
            }
            double* kernelWeights = kernel.data();
            const size_t kernelCount = kernel.size();
            for (size_t i = 0; i < kernelCount; ++i, ++kernelWeights) {
                *kernelWeights /= norm;
            }

            auto reflect_index = [size](int idx) {
                if (size <= 1) {
                    return 0;
                }
                while (idx < 0 || idx >= size) {
                    if (idx < 0) {
                        idx = -idx;
                    } else {
                        idx = 2 * size - 2 - idx;
                    }
                }
                return idx;
            };

            std::vector<double> output(input.size(), 0.0);
            const double* inputData = input.data();
            const double* kernelData = kernel.data();
            double* outputData = output.data();
            for (int i = 0; i < size; ++i) {
                double accum = 0.0;
                const double* kernelIt = kernelData;
                for (int k = -radius; k <= radius; ++k) {
                    const double weight = *kernelIt++;
                    const int candidate = i + k;
                    const int idx = (candidate >= 0 && candidate < size)
                                        ? candidate
                                        : reflect_index(candidate);
                    accum += inputData[static_cast<size_t>(idx)] * weight;
                }
                outputData[static_cast<size_t>(i)] = accum;
            }
            return output;
        }

        struct SlopeMeasureWindow {
            double leCenter = 0.0;
            double rangeEv = 1.0;
        };

        double measure_slope(const DoublePairs& curve, const SlopeMeasureWindow& window) {
            if (curve.size() < 2) {
                return 0.0;
            }
            constexpr double kHalfLog10_2 = 0.15051499783199059761; // 0.5 * log10(2)
            const double leDelta = window.rangeEv * kHalfLog10_2;
            const double le0 = window.leCenter - leDelta;
            const double le1 = window.leCenter + leDelta;
            if (!is_finite(le0) || !is_finite(le1) || !(le1 > le0)) {
                return 0.0;
            }
            const double density0 = sample_curve_at(curve, le0);
            const double density1 = sample_curve_at(curve, le1);
            const double denom = le1 - le0;
            if (!is_finite(density0) || !is_finite(density1) || std::fabs(denom) < 1e-8) {
                return 0.0;
            }
            return (density1 - density0) / denom;
        }

        bool apply_print_shadow_compensation_to_curves_internal(Profile& profile, DensityCurves& curves) {
            if (!profile.hasGlareCompensation || profile.glareCompensationFactor <= 0.0f) {
                return false;
            }

            const DoublePairs* reference = nullptr;
            if (!curves.magenta.empty()) {
                reference = &curves.magenta;
            } else if (!curves.cyan.empty()) {
                reference = &curves.cyan;
            } else if (!curves.yellow.empty()) {
                reference = &curves.yellow;
            }

            if (!reference || reference->size() < 2) {
                return false;
            }

            const size_t sampleCount = reference->size();
            std::vector<double> logExposures(sampleCount, 0.0);
            const std::pair<double, double>* referenceData = reference->data();
            double* logExposureWrite = logExposures.data();
            for (size_t i = 0; i < sampleCount; ++i, ++referenceData, ++logExposureWrite) {
                *logExposureWrite = referenceData->first;
            }
            const size_t logExposureCount = logExposures.size();
            const double* const logExposureData = logExposures.data();
            const DoublePairs* const cyanCurve = &curves.cyan;
            const DoublePairs* const magentaCurve = &curves.magenta;
            const DoublePairs* const yellowCurve = &curves.yellow;
            const size_t cyanCount = cyanCurve->size();
            const size_t magentaCount = magentaCurve->size();
            const size_t yellowCount = yellowCurve->size();
            const std::pair<double, double>* cyanData = cyanCurve->data();
            const std::pair<double, double>* magentaData = magentaCurve->data();
            const std::pair<double, double>* yellowData = yellowCurve->data();

            DoublePairs meanCurve;
            meanCurve.reserve(sampleCount);
            const double* logExposureRead = logExposureData;
            for (size_t i = 0; i < logExposureCount; ++i, ++logExposureRead) {
                const double le = *logExposureRead;
                if (!is_finite(le)) {
                    continue;
                }
                double sum = 0.0;
                int count = 0;
                if (i < cyanCount) {
                    accumulate_if_finite(cyanData[i].second, sum, count);
                }
                if (i < magentaCount) {
                    accumulate_if_finite(magentaData[i].second, sum, count);
                }
                if (i < yellowCount) {
                    accumulate_if_finite(yellowData[i].second, sum, count);
                }
                if (count > 0) {
                    const double mean = sum / static_cast<double>(count);
                    meanCurve.emplace_back(le, mean);
                }
            }

            if (meanCurve.size() < 2) {
                return false;
            }

            double leCenter = std::numeric_limits<double>::quiet_NaN();
            const double* leCenterRead = logExposureData;
            for (size_t i = 0; i < logExposureCount; ++i, ++leCenterRead) {
                const double le = *leCenterRead;
                if (is_finite(le)) {
                    leCenter = le;
                    break;
                }
            }
            if (!is_finite(leCenter)) {
                return false;
            }
            const double targetDensity = static_cast<double>(profile.glareCompensationDensity);
            if (is_finite(targetDensity)) {
                find_logE_for_density(meanCurve, targetDensity, leCenter);
            }

            const double slope = std::fabs(measure_slope(meanCurve, SlopeMeasureWindow{leCenter}));
            if (!(slope > 0.0)) {
                return false;
            }

            double leStep = 0.0;
            int stepCount = 0;
            if (logExposureCount > 1) {
                const double* prev = logExposureData;
                const double* curr = prev + 1;
                for (size_t i = 1; i < logExposureCount; ++i, ++prev, ++curr) {
                    const double a = *curr;
                    const double b = *prev;
                    if (is_finite(a) && is_finite(b)) {
                        const double diff = a - b;
                        if (is_nonzero_finite(diff)) {
                            leStep += std::fabs(diff);
                            ++stepCount;
                        }
                    }
                }
            }
            if (stepCount > 0) {
                leStep /= static_cast<double>(stepCount);
            }
            if (!(leStep > 0.0)) {
                return false;
            }

            const double factor = static_cast<double>(profile.glareCompensationFactor);
            const double transitionDensity = static_cast<double>(profile.glareCompensationTransition);
            std::vector<double> leAdjusted = logExposures;
            double* adjustedData = leAdjusted.data();
            const size_t adjustedCount = leAdjusted.size();
            for (size_t i = 0; i < adjustedCount; ++i, ++adjustedData) {
                double& le = *adjustedData;
                if (!is_finite(le)) {
                    le = leCenter;
                } else if (le > leCenter) {
                    le -= (le - leCenter) * factor;
                }
            }

            double leTransition = 0.0;
            if (is_finite(transitionDensity)) {
                leTransition = transitionDensity / slope;
            }
            if (!is_finite(leTransition)) {
                leTransition = 0.0;
            }
            const double sigmaSamples = std::fabs(leTransition) / leStep;
            std::vector<double> leFiltered = gaussian_filter_reflect(leAdjusted, sigmaSamples);
            const double* filteredData = leFiltered.data();

            const DoublePairs originalCyan = curves.cyan;
            const DoublePairs originalMagenta = curves.magenta;
            const DoublePairs originalYellow = curves.yellow;

            auto rebuild_curve = [&](const DoublePairs& source) {
                DoublePairs rebuilt;
                rebuilt.reserve(logExposureCount);
                if (source.empty()) {
                    return rebuilt;
                }
                const double* exposureRead = logExposureData;
                const double* queryRead = filteredData;
                for (size_t i = 0; i < logExposureCount; ++i, ++exposureRead, ++queryRead) {
                    const double le = *exposureRead;
                    if (!is_finite(le)) {
                        continue;
                    }
                    const double density = sample_curve_at(source, *queryRead);
                    rebuilt.emplace_back(le, density);
                }
                return rebuilt;
            };

            DoublePairs newCyan = rebuild_curve(originalCyan);
            DoublePairs newMagenta = rebuild_curve(originalMagenta);
            DoublePairs newYellow = rebuild_curve(originalYellow);

            bool applied = false;
            if (!newCyan.empty()) {
                curves.cyan = std::move(newCyan);
                applied = true;
            }
            if (!newMagenta.empty()) {
                curves.magenta = std::move(newMagenta);
                applied = true;
            }
            if (!newYellow.empty()) {
                curves.yellow = std::move(newYellow);
                applied = true;
            }

            if (!applied) {
                return false;
            }

            profile.glareRemoved = true;
            return true;
        }

        struct BaselineCurves {
            FloatPairs minPairs;
            FloatPairs midPairs;
        };

        void merge_baseline_with_json(const JsonProfileContext& ctx,
                                      BaselineCurves& curves) {
            if (!ctx.hasProfile) {
                return;
            }

            const auto& profileJson = ctx.profile;
            if (curves.minPairs.empty() &&
                json_profile_has_complete_print_baseline_min(ctx)) {
                curves.minPairs = profileJson.baseDensityMin;
            }
            if (curves.midPairs.empty() &&
                json_profile_has_complete_print_baseline_mid(ctx)) {
                curves.midPairs = profileJson.baseDensityMid;
            }
        }

        void apply_baseline_to_profile(const BaselineCurves& curves,
                                       Profile& out) {
            if (curves.minPairs.empty()) {
                out.baseDensityMin.lambda_nm.clear();
                out.baseDensityMin.linear.clear();
                out.baseDensityMid.lambda_nm.clear();
                out.baseDensityMid.linear.clear();
                out.hasBaseline = false;
                return;
            }

            auto scaledMin = curves.minPairs;
            auto scaledMid = curves.midPairs;

            auto build_baseline_curve = [](Spectral::Curve& dst, FloatPairs& pairs) -> bool {
                if (Spectral::samples_follow_reference_axis(pairs)) {
                    // Preserve NaNs (missingness) and clamp only finite negatives.
                    return Spectral::build_curve_on_reference_axis_from_aligned_pairs(
                        dst, pairs, /*clampNegative*/ true);
                }

                pad_pairs_to_shape_domain(pairs, std::nullopt, std::nullopt);
                return Spectral::build_curve_on_reference_axis_from_linear_pairs(dst, pairs);
            };

            const bool minOk = build_baseline_curve(out.baseDensityMin, scaledMin);
            bool midOk = true;
            if (!scaledMid.empty()) {
                midOk = build_baseline_curve(out.baseDensityMid, scaledMid);
            } else {
                out.baseDensityMid.lambda_nm.clear();
                out.baseDensityMid.linear.clear();
            }

            out.hasBaseline = minOk && !out.baseDensityMin.linear.empty();
            if (!out.hasBaseline) {
                std::ostringstream warn;
                warn << "WARN: Failed to resample print baseline min curve"
                     << " (min=" << (minOk ? "ok" : "empty after filter") << ")";
                JTRACE("PRINT", warn.str());
            } else if (!midOk && !scaledMid.empty()) {
                std::ostringstream warn;
                warn << "WARN: Failed to resample print baseline mid curve"
                     << " (mid=" << (midOk ? "ok" : "empty after filter") << ")";
                JTRACE("PRINT", warn.str());
            }
        }

    } // namespace

    bool apply_print_shadow_compensation_to_curves(Profile& profile, DensityCurves& curves) {
        return apply_print_shadow_compensation_to_curves_internal(profile, curves);
    }

    bool rebuild_density_curves(Profile& profile, const DensityCurves& curves) {
        const bool densityCurvesOk = build_density_curves(profile, curves.cyan, curves.magenta, curves.yellow);
        // agx-emulsion parity: do not "heal" NaNs (toe) or baseline-shift print density curves.
        // NaNs must propagate through sampling and become 0 transmitted light in density->light.
        return densityCurvesOk;
    }

    void recompute_mid_neutral(Profile& profile, Runtime* runtime) {
        const float nanLogE = std::numeric_limits<float>::quiet_NaN();
        profile.hasMidNeutralLogE = false;
        profile.midNeutralLogE = {nanLogE, nanLogE, nanLogE};
        if (profile.hasMidNeutralDensity) {
            std::array<float, 3> logE{nanLogE, nanLogE, nanLogE};
            const Spectral::Curve* curvesByChannel[3] = {&profile.dcC, &profile.dcM, &profile.dcY};
            const float* densityIt = profile.midNeutralDensity.data();
            float* logEIt = logE.data();
            bool allChannelsOk = true;
            for (const Spectral::Curve* const* curveIt = curvesByChannel;
                 curveIt < curvesByChannel + 3;
                 ++curveIt, ++densityIt, ++logEIt) {
                if (!invert_density_curve_at_target(**curveIt, *densityIt, *logEIt)) {
                    allChannelsOk = false;
                }
            }
            profile.hasMidNeutralLogE = allChannelsOk;
            if (profile.hasMidNeutralLogE) {
                profile.midNeutralLogE = logE;
            }
        }

        if (runtime) {
            runtime->hasMidNeutralDensity = profile.hasMidNeutralDensity;
            runtime->midNeutralDensity = profile.midNeutralDensity;
            runtime->hasMidNeutralLogE = profile.hasMidNeutralLogE;
            runtime->midNeutralLogE = profile.midNeutralLogE;
        }
    }

    void load_profile_from_asset(
        const JuicerAssets::SelectedPrintProfileAsset& asset,
        Profile& out,
        Runtime* runtime) {
        using Spectral::build_curve_on_reference_axis_from_log10_pairs;

        reset_profile_state(out, runtime);
        const std::string profileKey = asset.jsonKey.empty() ? std::string("<null>") : asset.jsonKey;
        const bool traceInfo = JTRACE_ENABLED(1);
        auto trace_print_json_profile = [&](const char* prefix, const char* suffix) {
            if (!traceInfo) {
                return;
            }
            std::string msg;
            msg.reserve((prefix ? std::strlen(prefix) : 0u) + profileKey.size() + (suffix ? std::strlen(suffix) : 0u) + 2);
            msg = prefix ? prefix : "";
            msg.push_back('\'');
            msg += profileKey;
            msg.push_back('\'');
            if (suffix) {
                msg += suffix;
            }
            JTRACE("PRINT", msg);
        };

        JsonProfileContext jsonCtx = load_json_profile(asset);
        if (!jsonCtx.hasProfile) {
            trace_print_json_profile("FATAL: missing JSON profile ", " for print paper (glare metadata required)");
            return;
        }

        out.glare = jsonCtx.profile.glare;
        out.hasGlareCompensation = jsonCtx.profile.hasGlareCompensation;
        out.glareCompensationFactor = jsonCtx.profile.glareCompensationFactor;
        out.glareCompensationDensity = jsonCtx.profile.glareCompensationDensity;
        out.glareCompensationTransition = jsonCtx.profile.glareCompensationTransition;
        if (runtime) {
            runtime->glare = out.glare;
        }

        FloatPairs c_eps;
        FloatPairs m_eps;
        FloatPairs y_eps;
        FloatPairs r_sens;
        FloatPairs g_sens;
        FloatPairs b_sens;

        bool usedJsonEps = false;
        bool usedJsonSens = false;
        apply_json_dye_and_sensitivity_overrides(
            jsonCtx,
            out,
            runtime,
            JsonDyeSensitivityOverrideTargets{
                c_eps,
                m_eps,
                y_eps,
                r_sens,
                g_sens,
                b_sens,
                usedJsonEps,
                usedJsonSens});

        if (jsonCtx.hasProfile && usedJsonEps) {
            trace_print_json_profile("PROFILE_LOAD dye densities from JSON ", "");
        }

        const bool epsOk = build_eps_curves(c_eps, m_eps, y_eps, out);
        if (!epsOk) {
            JTRACE("PRINT", "FATAL: Failed to assemble CMY epsilon curves; rejecting print profile");
            return;
        }

        DensityCurves densityCurves = load_density_curves(jsonCtx, profileKey);
        if (!densityCurves.usedJson) {
            densityCurves.cyan.clear();
            densityCurves.magenta.clear();
            densityCurves.yellow.clear();
        }
        if (runtime) {
            runtime->densityCurvesRaw = densityCurves;
        }

        // Per agx-emulsion parity: print paper sensitivities are used AS-IS from the profile.
        // NO balancing is applied to print sensitivities (only film sensitivities are balanced).
        // This preserves the referenceIlluminant metadata for documentation purposes only.
        std::string referenceLabel = out.referenceIlluminant;
        if (referenceLabel.empty() && runtime && !runtime->referenceIlluminant.empty()) {
            referenceLabel = runtime->referenceIlluminant;
        }
        if (referenceLabel.empty() && jsonCtx.hasProfile && !jsonCtx.profile.referenceIlluminant.empty()) {
            referenceLabel = jsonCtx.profile.referenceIlluminant;
        }
        if (referenceLabel.empty()) {
            referenceLabel = "D65";
        }

        out.referenceIlluminant = referenceLabel;
        if (runtime) {
            runtime->referenceIlluminant = referenceLabel;
        }

        // Print paper sensitivities: no correction applied (agx-emulsion parity)
        out.sensitivityNeutralCorr = {1.0f, 1.0f, 1.0f};
        out.hasSensitivityCorrection = false;
        if (runtime) {
            runtime->sensitivityNeutralCorr = {1.0f, 1.0f, 1.0f};
            runtime->hasSensitivityCorrection = false;
        }

        // Build print sensitivity curves directly from profile data (no parity alignment).
        // Per agx-emulsion: only FILM sensitivities undergo parity alignment; print paper
        // sensitivities are used exactly as provided in the JSON profile.
        const bool sensYOk = build_curve_on_reference_axis_from_log10_pairs(out.sensY_log, b_sens);
        const bool sensMOk = build_curve_on_reference_axis_from_log10_pairs(out.sensM_log, g_sens);
        const bool sensCOk = build_curve_on_reference_axis_from_log10_pairs(out.sensC_log, r_sens);

        const bool sensOk = sensYOk && sensMOk && sensCOk;
        if (!sensOk) {
            std::ostringstream warn;
            warn << "WARN: Failed to resample log sensitivities"
                 << " (Y=" << (sensYOk ? "ok" : "empty after filter")
                 << ", M=" << (sensMOk ? "ok" : "empty after filter")
                 << ", C=" << (sensCOk ? "ok" : "empty after filter")
                 << ")";
            JTRACE("PRINT", warn.str());
        }

        if (!sensOk) {
            JTRACE("PRINT", "FATAL: log sensitivity curves incomplete; rejecting print profile");
            return;
        }

        // Print paper sensitivities are not balanced, so no neutral exposure probe needed here.
        // (Neutral exposure probing applies only to film development, not print paper.)
        const bool densityCurvesOk = rebuild_density_curves(out, densityCurves);

        BaselineCurves baselineCurves;
        merge_baseline_with_json(jsonCtx, baselineCurves);
        apply_baseline_to_profile(baselineCurves, out);
        recompute_mid_neutral(out, runtime);


        if (!densityCurvesOk) {
            std::ostringstream fatal;
            fatal << "FATAL: missing spectral data (print profile)";
            fatal << " profileKey='" << profileKey << "'";
            JTRACE("PRINT", fatal.str());
            return;
        }
    }


} // namespace Print
