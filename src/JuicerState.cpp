#include "JuicerState.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#include "Illuminants.h"
#include "Logging.h"
#include "ProcessRoot.h"
#include "SpectralProcessing.h"

namespace {
    inline void copy_float3(float dst[3], const float src[3]) {
        std::memcpy(dst, src, 3u * sizeof(float));
    }

    static bool build_scanner_illuminant(
        const std::string& source,
        const char* label,
        Scanner::ScannerIlluminant& out);

    inline bool build_focused_film_payload(
        const RenderRecipe& recipe,
        FocusedRenderPayload& payload) {
        if (!recipe.profileRoute.filmProfile) {
            return false;
        }

        const Profiles::ValidatedFilmProfile& profile = *recipe.profileRoute.filmProfile;
        auto assign_channel = [&](Spectral::Curve& curve, std::size_t channel) {
            curve.lambda_nm.assign(profile.data.wavelengths.begin(), profile.data.wavelengths.end());
            curve.linear.resize(profile.data.channelDensity.size());
            for (std::size_t sample = 0; sample < profile.data.channelDensity.size(); ++sample) {
                curve.linear[sample] = profile.data.channelDensity[sample][channel];
            }
        };
        Spectral::Curve epsC;
        Spectral::Curve epsM;
        Spectral::Curve epsY;
        Spectral::Curve baseDensityMin;
        Spectral::Curve baseDensityMid;
        assign_channel(epsC, 0u);
        assign_channel(epsM, 1u);
        assign_channel(epsY, 2u);
        baseDensityMin.lambda_nm.assign(profile.data.wavelengths.begin(), profile.data.wavelengths.end());
        baseDensityMin.linear.assign(profile.data.baseDensity.begin(), profile.data.baseDensity.end());

        Scanner::ScannerIlluminant referenceIlluminant;
        if (!build_scanner_illuminant(
                profile.info.referenceIlluminant.value,
                "focused film reference",
                referenceIlluminant)) {
            return false;
        }
        Spectral::build_tables_from_curves_non_global(
            epsY,
            epsM,
            epsC,
            Spectral::gXBar,
            Spectral::gYBar,
            Spectral::gZBar,
            referenceIlluminant.curve,
            baseDensityMin,
            baseDensityMid,
            true,
            0.0f,
            payload.exposureTables,
            referenceIlluminant.hash);
        const auto valid_white = [](const float white[3]) {
            return std::isfinite(white[0]) &&
                   std::isfinite(white[1]) &&
                   std::isfinite(white[2]) &&
                   white[1] > 0.0f;
        };
        if (payload.exposureTables.K != Spectral::kNumSamples ||
            payload.exposureTables.tablesHash == 0 ||
            !valid_white(payload.exposureTables.whiteXYZ) ||
            !valid_white(payload.exposureTables.refIllumWhiteXYZ)) {
            return false;
        }
        Spectral::compute_S_inverse_from_tables(
            payload.exposureTables,
            payload.spdSInv.data());

        payload.filmRawConfig = Spectral::FilmRawConfig{};
        payload.filmRawConfig.inputColorSpace =
            Spectral::inputColorSpaceFromIndex(recipe.filmRaw.inputColorSpace);
        payload.filmRawConfig.applyCctfDecoding = recipe.filmRaw.inputCctfDecoding;
        payload.filmRawConfig.spectralUpsamplingMode =
            recipe.filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019
                ? Spectral::SpectralUpsamplingMode::ForceMallett
                : Spectral::SpectralUpsamplingMode::PreferHanatos;
        Spectral::prepare_film_raw_config(payload.filmRawConfig);
        std::copy_n(
            payload.exposureTables.refIllumWhiteXYZ,
            3,
            payload.filmRawConfig.refIllumWhiteXYZ);
        payload.filmRawConfig.hasRefIllumWhite = true;

        Spectral::Curve sensB;
        Spectral::Curve sensG;
        Spectral::Curve sensR;
        sensB.lambda_nm.assign(profile.data.wavelengths.begin(), profile.data.wavelengths.end());
        sensG.lambda_nm = sensB.lambda_nm;
        sensR.lambda_nm = sensB.lambda_nm;
        sensB.linear.resize(recipe.filmRaw.finalSensitivity.size());
        sensG.linear.resize(recipe.filmRaw.finalSensitivity.size());
        sensR.linear.resize(recipe.filmRaw.finalSensitivity.size());
        for (std::size_t sample = 0; sample < recipe.filmRaw.finalSensitivity.size(); ++sample) {
            const auto& rgb = recipe.filmRaw.finalSensitivity[sample];
            sensB.linear[sample] = rgb[2];
            sensG.linear[sample] = rgb[1];
            sensR.linear[sample] = rgb[0];
        }
        Spectral::compute_film_raw_midgray(
            payload.filmRawConfig,
            &payload.exposureTables,
            payload.spdSInv.data(),
            sensB,
            sensG,
            sensR);
        return payload.filmRawConfig.valid;
    }

    inline bool build_direct_scanner_payload(
        const RenderRecipe& recipe,
        FocusedRenderPayload& payload) {
        if (Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ||
            !recipe.directStructuralReady ||
            !recipe.profileRoute.filmProfile) {
            return false;
        }

        const Profiles::ValidatedFilmProfile& profile = *recipe.profileRoute.filmProfile;
        auto assign_channel = [&](Spectral::Curve& curve, std::size_t channel) {
            curve.lambda_nm.assign(profile.data.wavelengths.begin(), profile.data.wavelengths.end());
            curve.linear.resize(profile.data.channelDensity.size());
            for (std::size_t sample = 0; sample < profile.data.channelDensity.size(); ++sample) {
                curve.linear[sample] = profile.data.channelDensity[sample][channel];
            }
        };
        Spectral::Curve epsC;
        Spectral::Curve epsM;
        Spectral::Curve epsY;
        Spectral::Curve baseDensityMin;
        Spectral::Curve baseDensityMid;
        assign_channel(epsC, 0u);
        assign_channel(epsM, 1u);
        assign_channel(epsY, 2u);
        baseDensityMin.lambda_nm.assign(profile.data.wavelengths.begin(), profile.data.wavelengths.end());
        baseDensityMin.linear.assign(profile.data.baseDensity.begin(), profile.data.baseDensity.end());

        Scanner::ScannerIlluminant scannerIlluminant;
        if (!build_scanner_illuminant(
                recipe.scannerOutput.viewingIlluminant,
                "focused direct viewing",
                scannerIlluminant)) {
            return false;
        }
        Spectral::build_tables_from_curves_non_global(
            epsY,
            epsM,
            epsC,
            Spectral::gXBar,
            Spectral::gYBar,
            Spectral::gZBar,
            scannerIlluminant.curve,
            baseDensityMin,
            baseDensityMid,
            true,
            0.0f,
            payload.scannerTables,
            scannerIlluminant.hash);
        if (payload.scannerTables.K != Spectral::kNumSamples ||
            payload.scannerTables.tablesHash == 0) {
            return false;
        }

        OutputEncoding::Params encoding{};
        encoding.colorSpace = OutputEncoding::colorSpaceFromIndex(recipe.scannerOutput.outputColorSpace);
        encoding.applyCctfEncoding = recipe.scannerOutput.outputCctfEncoding;
        encoding.preserveLinearRange = recipe.scannerOutput.outputLinearPassThrough;
        encoding.inputIsOutputSpace = true;
        payload.scannerColor = Scanner::build_color_runtime(
            Scanner::ScannerMedium::Negative,
            scannerIlluminant,
            encoding);
        payload.uploadCoreHash = recipe.hash;
        payload.scannerHash = Hash::hash_uint64_values(
            {recipe.densityBounds.hash,
             payload.scannerTables.tablesHash,
             static_cast<std::uint64_t>(recipe.scannerOutput.lutResolution)});
        return payload.scannerColor.hash != 0 &&
               payload.uploadCoreHash != 0 &&
               payload.scannerHash != 0;
    }

    inline bool build_print_scanner_payload(
        const RenderRecipe& recipe,
        FocusedRenderPayload& payload) {
        if (!Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ||
            !recipe.printStructuralReady ||
            !recipe.profileRoute.printProfile) {
            return false;
        }

        const Profiles::ValidatedPrintProfile& profile = *recipe.profileRoute.printProfile;
        auto assign_channel = [&](Spectral::Curve& curve, std::size_t channel) {
            curve.lambda_nm.assign(profile.data.wavelengths.begin(), profile.data.wavelengths.end());
            curve.linear.resize(profile.data.channelDensity.size());
            for (std::size_t sample = 0; sample < profile.data.channelDensity.size(); ++sample) {
                curve.linear[sample] = profile.data.channelDensity[sample][channel];
            }
        };
        Spectral::Curve epsC;
        Spectral::Curve epsM;
        Spectral::Curve epsY;
        Spectral::Curve baseDensityMin;
        Spectral::Curve baseDensityMid;
        assign_channel(epsC, 0u);
        assign_channel(epsM, 1u);
        assign_channel(epsY, 2u);
        baseDensityMin.lambda_nm.assign(profile.data.wavelengths.begin(), profile.data.wavelengths.end());
        baseDensityMin.linear.assign(profile.data.baseDensity.begin(), profile.data.baseDensity.end());

        Scanner::ScannerIlluminant scannerIlluminant;
        if (!build_scanner_illuminant(
                recipe.scannerOutput.viewingIlluminant,
                "focused print viewing",
                scannerIlluminant)) {
            return false;
        }
        Spectral::build_tables_from_curves_non_global(
            epsY,
            epsM,
            epsC,
            Spectral::gXBar,
            Spectral::gYBar,
            Spectral::gZBar,
            scannerIlluminant.curve,
            baseDensityMin,
            baseDensityMid,
            true,
            0.0f,
            payload.scannerTables,
            scannerIlluminant.hash);
        if (payload.scannerTables.K != Spectral::kNumSamples ||
            payload.scannerTables.tablesHash == 0) {
            return false;
        }

        OutputEncoding::Params encoding{};
        encoding.colorSpace = OutputEncoding::colorSpaceFromIndex(recipe.scannerOutput.outputColorSpace);
        encoding.applyCctfEncoding = recipe.scannerOutput.outputCctfEncoding;
        encoding.preserveLinearRange = recipe.scannerOutput.outputLinearPassThrough;
        encoding.inputIsOutputSpace = true;
        payload.scannerColor = Scanner::build_color_runtime(
            Scanner::ScannerMedium::Print,
            scannerIlluminant,
            encoding);
        payload.uploadCoreHash = recipe.hash;
        payload.scannerHash = Hash::hash_uint64_values(
            {recipe.densityBounds.hash,
             payload.scannerTables.tablesHash,
             static_cast<std::uint64_t>(recipe.scannerOutput.lutResolution)});
        return payload.scannerColor.hash != 0 &&
               payload.uploadCoreHash != 0 &&
               payload.scannerHash != 0;
    }

    inline bool is_finite(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite(double value) {
        return std::isfinite(value);
    }

    inline bool is_positive_finite(double value) {
        return is_finite(value) && value > 0.0;
    }

    inline uint64_t hash_mix(uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }

    template <typename MixFn, typename TValue>
    inline void mix_hash_field(uint64_t& h, TValue value, const MixFn& mix) {
        h = mix(h, static_cast<uint64_t>(value));
    }

    inline std::uint32_t canonical_float_bits(float value) {
        const float canonical = value == 0.0f ? 0.0f : value;
        return std::bit_cast<std::uint32_t>(canonical);
    }

    template <typename MixFn>
    inline void mix_canonical_float_bits(uint64_t& h, float value, const MixFn& mix) {
        mix_hash_field(h, canonical_float_bits(value), mix);
    }

    template <typename MixFn>
    inline void mix_hash_string(uint64_t& h, const std::string& value, const MixFn& mix) {
        h = mix(h, static_cast<uint64_t>(value.size()));
        if (!value.empty()) {
            h = mix(h, Hash::hash_bytes(value.data(), value.size()));
        }
    }

    template <typename MixFn>
    inline void mix_hash_field_scaled(uint64_t& h, double value, double scale, const MixFn& mix) {
        h = mix(h, static_cast<uint64_t>(value * scale));
    }

    template <typename MixFn>
    inline void mix_hash_field_scaled_rounded_if_finite(
        uint64_t& h,
        double value,
        double scale,
        const MixFn& mix) {
        if (!is_finite(value)) {
            return;
        }
        const int64_t scaled = static_cast<int64_t>(std::llround(value * scale));
        h = mix(h, static_cast<uint64_t>(scaled));
    }

    template <typename MixFn>
    inline void mix_glare_print_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field_scaled(h, p.printShadowCompensationFactor, 10000.0, mix);
        mix_hash_field_scaled(h, p.printShadowCompensationDensity, 10000.0, mix);
        mix_hash_field_scaled(h, p.printShadowCompensationTransition, 10000.0, mix);
        mix_hash_field_scaled(h, p.printDminFactor, 10000.0, mix);
        mix_hash_field(h, p.glareActive ? 1 : 0, mix);
        mix_hash_field_scaled(h, p.glarePercent, 10000.0, mix);
        mix_hash_field_scaled(h, p.glareRoughness, 10000.0, mix);
        mix_hash_field_scaled(h, p.glareBlurSigmaPx, 10000.0, mix);
    }

    template <typename MixFn>
    inline void mix_camera_filter_hash(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field(h, p.cameraFilterOverride ? 1 : 0, mix);
        if (!p.cameraFilterOverride) {
            return;
        }

        auto mix_triplet = [&](const std::array<double, 3>& triplet) {
            const double* values = triplet.data();
            const double* const valuesEnd = values + triplet.size();
            for (; values < valuesEnd; ++values) {
                mix_hash_field_scaled_rounded_if_finite(h, *values, 10000.0, mix);
            }
        };
        mix_triplet(p.cameraFilterUV);
        mix_triplet(p.cameraFilterIR);
    }

    template <typename MixFn>
    inline void mix_scan_route_hash_field(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field(h, static_cast<std::uint8_t>(p.scanRoute), mix);
    }

    template <typename MixFn>
    inline void mix_profile_selection_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        // Phase 3A hash builders consume pre-resolved selected asset tokens. Selected loader/cache
        // access happens before hashing, so direct-route hashes perform no catalog/file/assets lookup.
        mix_hash_string(h, p.filmProfileKey, mix);
        mix_hash_field(h, p.filmProfileAssetVersionToken, mix);
        mix_scan_route_hash_field(h, p, mix);
        if (Spektrafilm::scan_route_is_print(p.scanRoute)) {
            mix_hash_string(h, p.printProfileKey, mix);
            mix_hash_field(h, p.printProfileAssetVersionToken, mix);
            mix_hash_field(h, p.enlIll, mix);
            mix_hash_field(h, p.enlDichroicSet, mix);
            mix_hash_field_scaled_rounded_if_finite(h, p.printExposure, 10000.0, mix);
            mix_hash_field_scaled_rounded_if_finite(h, p.printPreflashExposure, 10000.0, mix);
            mix_hash_field(h, p.normalizePrintExposure, mix);
            mix_hash_field(h, p.printExposureCompensation, mix);
            for (double value : p.printUiYmcCc) {
                mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
            }
            mix_hash_field_scaled_rounded_if_finite(h, p.preflashMFilterCc, 10000.0, mix);
            mix_hash_field_scaled_rounded_if_finite(h, p.preflashYFilterCc, 10000.0, mix);
        }
        mix_hash_field(h, p.spectralUpsamplingMode, mix);
        mix_hash_field(h, p.refIll, mix);
    }

    template <typename MixFn>
    inline void mix_output_encoding_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field(h, p.scannerLutResolution, mix);
        mix_hash_field(h, p.inputColorSpace, mix);
        mix_hash_field(h, p.inputCctfDecoding, mix);
        mix_hash_field(h, p.outputColorSpace, mix);
        mix_hash_field(h, p.outputCctfEncoding, mix);
        mix_hash_field(h, p.outputLinearPassThrough, mix);
    }

    template <typename MixFn>
    inline void mix_hanatos_adaptation_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        if (p.spectralUpsamplingMode == 1) {
            return;
        }
        mix_hash_field(h, p.hanatos2025AdaptationWindow, mix);
        mix_hash_field(h, p.hanatos2025AdaptationSurface, mix);
    }

    template <typename MixFn>
    inline void mix_direct_phase3a_recipe_hash_fields(
        uint64_t& h,
        const ParamSnapshot& p,
        const MixFn& mix) {
        mix_hash_field(h, p.cameraAutoExposureEnabled, mix);
        mix_hash_field(h, p.cameraMeteringMethod, mix);
        mix_hash_field_scaled_rounded_if_finite(
            h,
            p.cameraExposureCompensationEv,
            10000.0,
            mix);
        mix_canonical_float_bits(h, p.cameraFilmFormatLongEdgeMm, mix);
        mix_hash_field(h, p.scatterHalationControls.active ? 1 : 0, mix);
        if (p.scatterHalationControls.active) {
            mix_canonical_float_bits(h, p.scatterHalationControls.scatterAmount, mix);
            mix_canonical_float_bits(h, p.scatterHalationControls.scatterSpatialScale, mix);
            mix_canonical_float_bits(h, p.scatterHalationControls.halationAmount, mix);
            mix_canonical_float_bits(h, p.scatterHalationControls.halationSpatialScale, mix);
        }
        mix_hash_field_scaled_rounded_if_finite(h, p.scannerLensBlurSigmaPx, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, p.scannerUnsharpMask[0], 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, p.scannerUnsharpMask[1], 10000.0, mix);
        mix_hash_field(h, p.scannerBlackCorrection, mix);
        mix_hash_field(h, p.scannerWhiteCorrection, mix);
        mix_hash_field_scaled_rounded_if_finite(h, p.scannerBlackLevel, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, p.scannerWhiteLevel, 10000.0, mix);
    }

    template <typename MixFn>
    inline void mix_diffusion_authored_hash_fields(
        uint64_t& h,
        const ParamSnapshot& p,
        const MixFn& mix) {
        mix_hash_field(
            h,
            Spektrafilm::SpatialOpticsComponent::CameraDiffusion,
            mix);
        mix_hash_field(
            h,
            Spektrafilm::hash_diffusion_authored_controls(p.cameraDiffusion),
            mix);
        mix_hash_field(
            h,
            Spektrafilm::SpatialOpticsComponent::EnlargerDiffusion,
            mix);
        mix_hash_field(
            h,
            Spektrafilm::hash_diffusion_authored_controls(p.enlargerDiffusion),
            mix);
    }

    template <typename MixFn>
    inline void mix_focused_grain_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        const Spektrafilm::VisualGrainControls& grain = p.grainControls;
        if (!grain.active) {
            return;
        }
        mix_hash_field(h, 1, mix);
        mix_hash_field(h, grain.sublayersActive ? 1 : 0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.particleAreaUm2, 10000.0, mix);
        for (float value : grain.particleScaleCmy) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
        for (float value : grain.particleScaleLayers) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
        for (float value : grain.visualParticleDensityMinCmy) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
        for (float value : grain.uniformityCmy) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
        mix_hash_field_scaled_rounded_if_finite(h, grain.correlationSigmaPx, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.dyeCloudBlurUm, 10000.0, mix);
        for (float value : grain.microStructure) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
        mix_hash_field(h, grain.nSubLayers, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.amplitude, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.chromaSharedWeight, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.chromaIndependentWeight, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.coarseWeight, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.midWeight, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.sizeMixScale, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.clumpTemporalMix, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, grain.clumpMorphPeriodSec, 10000.0, mix);
        mix_hash_field(h, grain.debugView, mix);
    }

    template <typename MixFn>
    inline void mix_film_juicer_effects_hash_fields(
        uint64_t& h,
        const ParamSnapshot& p,
        const MixFn& mix) {
        const auto positive_finite = [](auto value) {
            return std::isfinite(value) && value > 0;
        };
        if (!positive_finite(p.filmDustAmount) &&
            !positive_finite(p.filmScratchAmount) &&
            !positive_finite(p.gateDustAmount) &&
            !positive_finite(p.gateScratchAmount) &&
            !positive_finite(p.gateWeaveAmount)) {
            return;
        }
        mix_hash_field(h, 1, mix);
        mix_hash_field_scaled_rounded_if_finite(
            h,
            p.filmDustAmount,
            10000.0,
            mix);
        mix_hash_field_scaled_rounded_if_finite(
            h,
            p.filmScratchAmount,
            10000.0,
            mix);
        mix_hash_field_scaled_rounded_if_finite(
            h,
            p.gateDustAmount,
            10000.0,
            mix);
        mix_hash_field_scaled_rounded_if_finite(
            h,
            p.gateScratchAmount,
            10000.0,
            mix);
        mix_hash_field_scaled_rounded_if_finite(
            h,
            p.gateWeaveAmount,
            10000.0,
            mix);
    }

    template <typename MixFn>
    inline void mix_coupler_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field(h, p.couplersActive, mix);
        mix_hash_field_scaled_rounded_if_finite(h, p.couplersAmount, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, p.couplersInhibitionSameLayer, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, p.couplersInhibitionInterlayer, 10000.0, mix);
        mix_hash_field_scaled_rounded_if_finite(h, p.couplersDiffusionSizeUm, 10000.0, mix);
        mix_hash_field(h, p.couplersGammaUseStock, mix);
        for (double value : p.couplersGammaSameLayerRgb) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
        for (double value : p.couplersGammaInterlayerRToGb) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
        for (double value : p.couplersGammaInterlayerGToRb) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
        for (double value : p.couplersGammaInterlayerBToRg) {
            mix_hash_field_scaled_rounded_if_finite(h, value, 10000.0, mix);
        }
    }

    Spektrafilm::DirCouplersControls focused_dir_couplers_controls_from_snapshot(const ParamSnapshot& p) {
        Spektrafilm::DirCouplersControls controls{};
        controls.active = p.couplersActive != 0;
        controls.amount = static_cast<float>(p.couplersAmount);
        controls.inhibitionSameLayer = static_cast<float>(p.couplersInhibitionSameLayer);
        controls.inhibitionInterlayer = static_cast<float>(p.couplersInhibitionInterlayer);
        controls.diffusionSizeUm = static_cast<float>(p.couplersDiffusionSizeUm);
        controls.gammaUseStock = p.couplersGammaUseStock != 0;
        controls.gammaSameLayerRgb = {
            static_cast<float>(p.couplersGammaSameLayerRgb[0]),
            static_cast<float>(p.couplersGammaSameLayerRgb[1]),
            static_cast<float>(p.couplersGammaSameLayerRgb[2])};
        controls.gammaInterlayerRToGb = {
            static_cast<float>(p.couplersGammaInterlayerRToGb[0]),
            static_cast<float>(p.couplersGammaInterlayerRToGb[1])};
        controls.gammaInterlayerGToRb = {
            static_cast<float>(p.couplersGammaInterlayerGToRb[0]),
            static_cast<float>(p.couplersGammaInterlayerGToRb[1])};
        controls.gammaInterlayerBToRg = {
            static_cast<float>(p.couplersGammaInterlayerBToRg[0]),
            static_cast<float>(p.couplersGammaInterlayerBToRg[1])};
        return controls;
    }

    Spektrafilm::SpatialOpticsControls focused_spatial_optics_controls_from_snapshot(
        const ParamSnapshot& params) {
        Spektrafilm::SpatialOpticsControls controls{};
        controls.cameraDiffusion = params.cameraDiffusion;
        controls.scatterHalation = params.scatterHalationControls;
        controls.enlargerDiffusion = params.enlargerDiffusion;
        return controls;
    }


    static Spectral::Curve build_blackbody_curve(float temperature) {
        Spectral::Curve curve;
        if (!(temperature > 0.0f)) {
            return curve;
        }
        const int K = Spectral::gShape.K;
        Spectral::assign_reference_axis(curve.lambda_nm);
        curve.linear.resize(static_cast<size_t>(K));
        const float* wavelengths = Spectral::gShape.wavelengths.data();
        float* outLinear = curve.linear.data();
        for (int i = 0; i < K; ++i) {
            Spectral::PlanckBlackbodySample sample{};
            sample.wavelengthNm = wavelengths[i];
            sample.temperatureKelvin = temperature;
            outLinear[i] = Spectral::planck_blackbody(sample);
        }
        Spectral::mean_power_normalize(curve.linear);
        return curve;
    }

    static Spectral::Curve build_illuminant_from_string(const std::string& source) {
        const std::string normalized = IlluminantKeys::normalize(source);
        const JuicerAssets::IlluminantFilterCurveSet& curveAssets =
            JuicerProcess::root().assets().illuminant_filter_curves();

        auto build_or_log = [&](auto builder, const char* label) -> Spectral::Curve {
            try {
                return builder();
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "failed to load illuminant '" << source << "' (" << label
                    << "): " << e.what();
                JTRACE("ILLUM", oss.str());
            } catch (...) {
                std::ostringstream oss;
                oss << "failed to load illuminant '" << source << "' (" << label
                    << "): unknown error";
                JTRACE("ILLUM", oss.str());
            }
            return Spectral::Curve{};
        };

        if (IlluminantKeys::matches_any(normalized, {"D65"})) {
            return build_or_log([&]() {
                return curveAssets.d65;
            },
                                "D65");
        }
        if (IlluminantKeys::matches_any(normalized, {"D55"})) {
            return build_or_log([&]() {
                return curveAssets.d55;
            },
                                "D55");
        }
        if (IlluminantKeys::matches_any(normalized, {"D50"})) {
            return build_or_log([&]() {
                return curveAssets.d50;
            },
                                "D50");
        }
        if (IlluminantKeys::matches_any(normalized, {"TH-KG3-L", "THKG3L", "TH-KG3L"})) {
            return build_or_log([&]() {
                return curveAssets.tungstenKg3Lens;
            },
                                "TH-KG3-L");
        }
        if (IlluminantKeys::matches_any(normalized, {"T", "INCANDESCENT"})) {
            return build_or_log([&]() {
                return curveAssets.tungsten;
            },
                                "T");
        }
        if (IlluminantKeys::matches_any(normalized, {"K75P", "KINOTON75P"})) {
            return build_or_log([&]() {
                return curveAssets.kinoton75P;
            },
                                "K75P");
        }
        if (IlluminantKeys::matches_any(normalized, {"EQUAL", "EQUALENERGY", "EQUAL-ENERGY"})) {
            return Spectral::build_curve_equal_energy_pinned();
        }

        if (normalized.size() > 2 && normalized[0] == 'B' && normalized[1] == 'B') {
            const char* start = normalized.c_str() + 2;
            while (*start != '\0' && std::isspace(static_cast<unsigned char>(*start))) {
                ++start;
            }
            char* endPtr = nullptr;
            const double temperature = std::strtod(start, &endPtr);
            while (endPtr && *endPtr != '\0' && std::isspace(static_cast<unsigned char>(*endPtr))) {
                ++endPtr;
            }
            if (start != endPtr && endPtr && *endPtr == '\0' && temperature > 0.0) {
                return build_blackbody_curve(static_cast<float>(temperature));
            }
        }

        if (!normalized.empty()) {
            std::ostringstream oss;
            oss << "unknown illuminant string '" << source << "'; returning empty curve";
            JTRACE("ILLUM", oss.str());
        }
        return Spectral::Curve{};
    }

    static bool curve_matches_reference_axis(const Spectral::Curve& curve) {
        const size_t expected = static_cast<size_t>(Spectral::gShape.K);
        if (curve.linear.size() != expected || curve.lambda_nm.size() != expected) {
            return false;
        }
        const float* lambdaData = curve.lambda_nm.data();
        const float* axisData = Spectral::gShape.wavelengths.data();
        constexpr float kAxisMatchTolerance = 1e-3f;
        const float* const lambdaEnd = lambdaData + expected;
        for (; lambdaData < lambdaEnd; ++lambdaData, ++axisData) {
            const float lambda = *lambdaData;
            if (!is_finite(lambda) ||
                std::abs(lambda - *axisData) > kAxisMatchTolerance) {
                return false;
            }
        }
        return true;
    }

    static bool build_scanner_illuminant(
        const std::string& source,
        const char* label,
        Scanner::ScannerIlluminant& out) {
        out = Scanner::ScannerIlluminant{};
        if (source.empty()) {
            std::ostringstream oss;
            oss << "FATAL: missing viewing illuminant for " << label;
            JTRACE("ILLUM", oss.str());
            return false;
        }

        Spectral::Curve curve = build_illuminant_from_string(source);
        if (!curve_matches_reference_axis(curve)) {
            std::ostringstream oss;
            oss << "FATAL: viewing illuminant '" << source
                << "' for " << label << " not pinned to agx axis";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        const int K = Spectral::gShape.K;
        const auto& xBar = Spectral::gXBar.linear;
        const auto& yBar = Spectral::gYBar.linear;
        const auto& zBar = Spectral::gZBar.linear;
        if (xBar.size() != static_cast<size_t>(K) ||
            yBar.size() != static_cast<size_t>(K) ||
            zBar.size() != static_cast<size_t>(K)) {
            std::ostringstream oss;
            oss << "FATAL: CMFs unavailable for " << label << " (K mismatch)";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        double sumX = 0.0;
        double sumY = 0.0;
        double sumZ = 0.0;
        const float* spdData = curve.linear.data();
        const float* xData = xBar.data();
        const float* yData = yBar.data();
        const float* zData = zBar.data();
        for (int i = 0; i < K; ++i) {
            const float spd = spdData[i];
            const float xb = xData[i];
            const float yb = yData[i];
            const float zb = zData[i];
            if (!(is_finite(spd) && is_finite(xb) && is_finite(yb) && is_finite(zb))) {
                std::ostringstream oss;
                oss << "FATAL: non-finite CMF/SPD sample in " << label << " illuminant";
                JTRACE("ILLUM", oss.str());
                return false;
            }
            sumX += static_cast<double>(spd) * static_cast<double>(xb);
            sumY += static_cast<double>(spd) * static_cast<double>(yb);
            sumZ += static_cast<double>(spd) * static_cast<double>(zb);
        }

        if (!is_positive_finite(sumY)) {
            std::ostringstream oss;
            oss << "FATAL: invalid luminance sum for " << label << " (Yn=" << sumY << ")";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        out.curve = std::move(curve);
        out.normalization = static_cast<float>(sumY);
        const double invYn = 1.0 / sumY;
        const float whiteXYZ[3] = {
            static_cast<float>(sumX * invYn),
            1.0f,
            static_cast<float>(sumZ * invYn)};
        copy_float3(out.whiteXYZ, whiteXYZ);

        const double whiteSum = sumX + sumY + sumZ;
        if (!is_positive_finite(whiteSum)) {
            JTRACE("ILLUM", "FATAL: invalid white sum while building scanner illuminant");
            return false;
        }
        out.whiteXY[0] = static_cast<float>(sumX / whiteSum);
        out.whiteXY[1] = static_cast<float>(sumY / whiteSum);

        constexpr int kReferenceAxisSamples = 81;
        constexpr size_t kReferenceAxisSampleCount = 81u;
        const size_t sampleCount = out.curve.linear.size();
        if (K == kReferenceAxisSamples && sampleCount == kReferenceAxisSampleCount) {
            float hashSamples[kReferenceAxisSampleCount + 1u];
            std::memcpy(
                hashSamples,
                out.curve.linear.data(),
                kReferenceAxisSampleCount * sizeof(float));
            hashSamples[kReferenceAxisSampleCount] = out.normalization;
            out.hash = Hash::hash_float_span(hashSamples, kReferenceAxisSampleCount + 1u);
        } else {
            std::vector<float> hashSamples(sampleCount + 1);
            if (sampleCount > 0) {
                std::memcpy(
                    hashSamples.data(),
                    out.curve.linear.data(),
                    sampleCount * sizeof(float));
            }
            hashSamples[sampleCount] = out.normalization;
            out.hash = Hash::hash_float_span(hashSamples.data(), hashSamples.size());
        }
        if (out.hash == 0) {
            std::ostringstream oss;
            oss << "FATAL: failed to hash viewing illuminant for " << label;
            JTRACE("ILLUM", oss.str());
            return false;
        }
        return true;
    }

    Spektrafilm::DichroicFilterSet dichroic_filter_set_from_choice(int choice) {
        switch (choice) {
            case 1:
                return Spektrafilm::DichroicFilterSet::DurstDigitalLight;
            case 2:
                return Spektrafilm::DichroicFilterSet::Thorlabs;
            case 3:
                return Spektrafilm::DichroicFilterSet::EdmundOptics;
            default:
                return Spektrafilm::DichroicFilterSet::Custom;
        }
    }

    std::string dichroic_filter_set_key(Spektrafilm::DichroicFilterSet set) {
        switch (set) {
            case Spektrafilm::DichroicFilterSet::DurstDigitalLight:
                return "durst_digital_light";
            case Spektrafilm::DichroicFilterSet::Thorlabs:
                return "thorlabs";
            case Spektrafilm::DichroicFilterSet::EdmundOptics:
                return "edmund_optics";
            case Spektrafilm::DichroicFilterSet::Custom:
            default:
                return "custom";
        }
    }

    std::string print_illuminant_key_from_choice(int choice) {
        switch (choice) {
            case 0:
                return "D65";
            case 1:
                return "D55";
            case 2:
                return "D50";
            case 4:
                return "TH-KG3-L";
            case 5:
                return "T";
            case 6:
                return "K75P";
            case 7:
                return "EQUAL";
            case 3:
            default:
                return "TH-KG3";
        }
    }

    Spektrafilm::FilmFoundationBuildInput film_foundation_input_from_snapshot(
        const ParamSnapshot& params,
        const std::shared_ptr<const Profiles::ValidatedFilmProfile>& filmProfile,
        Spektrafilm::ScanRoute route,
        const JuicerAssets::IlluminantFilterCurveSet& illuminants) {
        Spektrafilm::FilmFoundationBuildInput input{};
        input.filmProfileKey = params.filmProfileKey;
        input.scanRoute = route;
        input.filmProfile = filmProfile;
        input.visualGrain = params.grainControls;
        input.filmDustAmount = params.filmDustAmount;
        input.filmScratchAmount = params.filmScratchAmount;
        input.gateDustAmount = params.gateDustAmount;
        input.gateScratchAmount = params.gateScratchAmount;
        input.gateWeaveAmount = params.gateWeaveAmount;
        input.dirCouplers =
            focused_dir_couplers_controls_from_snapshot(params);
        input.spatialOptics =
            focused_spatial_optics_controls_from_snapshot(params);
        input.spectralUpsamplingMode = params.spectralUpsamplingMode;
        input.inputColorSpace = params.inputColorSpace;
        input.inputCctfDecoding = params.inputCctfDecoding != 0;
        input.applyHanatos2025AdaptationWindow =
            params.hanatos2025AdaptationWindow != 0;
        input.applyHanatos2025AdaptationSurface =
            params.hanatos2025AdaptationSurface != 0;
        input.cameraAutoExposureEnabled =
            params.cameraAutoExposureEnabled != 0;
        input.cameraMeteringMethod = params.cameraMeteringMethod;
        input.manualExposureCompensationEv =
            static_cast<float>(params.cameraExposureCompensationEv);
        input.filmFormatLongEdgeMm =
            static_cast<float>(params.cameraFilmFormatLongEdgeMm);
        input.cameraFilterOverride = params.cameraFilterOverride;
        input.cameraFilterUV = params.cameraFilterUV;
        input.cameraFilterIR = params.cameraFilterIR;

        if (!filmProfile) {
            return input;
        }
        const std::string illuminantKey =
            IlluminantKeys::normalize(
                filmProfile->info.referenceIlluminant.value);
        const Spectral::Curve* referenceIlluminant = nullptr;
        if (IlluminantKeys::matches_any(illuminantKey, {"D65"})) {
            referenceIlluminant = &illuminants.d65;
        } else if (IlluminantKeys::matches_any(illuminantKey, {"D55"})) {
            referenceIlluminant = &illuminants.d55;
        } else if (IlluminantKeys::matches_any(illuminantKey, {"D50"})) {
            referenceIlluminant = &illuminants.d50;
        } else if (IlluminantKeys::matches_any(
                       illuminantKey,
                       {"T", "TUNGSTEN"})) {
            referenceIlluminant = &illuminants.tungsten;
        } else if (IlluminantKeys::matches_any(
                       illuminantKey,
                       {"TH-KG3", "TUNGSTEN-KG3"})) {
            referenceIlluminant = &illuminants.tungstenKg3Lens;
        }
        if (referenceIlluminant &&
            referenceIlluminant->linear.size() ==
                input.referenceIlluminant.size()) {
            std::copy(
                referenceIlluminant->linear.begin(),
                referenceIlluminant->linear.end(),
                input.referenceIlluminant.begin());
            input.referenceIlluminantValid = true;
        }
        return input;
    }

    bool build_direct_render_state_product_impl(
        const ParamSnapshot& params,
        FocusedRenderStateBuildProduct& out,
        std::string& outError) {
        out = FocusedRenderStateBuildProduct{};
        outError.clear();
        if (Spektrafilm::scan_route_is_print(params.scanRoute)) {
            outError = "ResourceDescriptorMismatch route=direct field=scan_route";
            return false;
        }

        JuicerAssets::Library& assets = JuicerProcess::root().assets();
        const JuicerAssets::SelectedProfileResult selected =
            assets.selected_profiles_for_route(
                JuicerAssets::SelectedProfileRequest{
                    params.filmProfileKey,
                    params.printProfileKey,
                    params.scanRoute});
        if (!selected.valid || !selected.filmProfile) {
            outError = selected.diagnostic.empty()
                           ? "MissingRequiredResource route=direct field=selected_profile"
                           : selected.diagnostic;
            return false;
        }
        Spektrafilm::DirectRecipeBuildInput input{};
        input.film = film_foundation_input_from_snapshot(
            params,
            selected.filmProfile,
            params.scanRoute,
            assets.illuminant_filter_curves());
        input.scannerLutResolution =
            static_cast<std::uint32_t>(std::clamp(params.scannerLutResolution, 17, 128));
        input.outputColorSpace = params.outputColorSpace;
        input.outputCctfEncoding = params.outputCctfEncoding != 0;
        input.outputLinearPassThrough = params.outputLinearPassThrough != 0;
        input.scannerBlackCorrection = params.scannerBlackCorrection != 0;
        input.scannerWhiteCorrection = params.scannerWhiteCorrection != 0;
        input.scannerBlackLevel = static_cast<float>(params.scannerBlackLevel);
        input.scannerWhiteLevel = static_cast<float>(params.scannerWhiteLevel);
        input.scannerLensBlurSigmaPx = static_cast<float>(params.scannerLensBlurSigmaPx);
        input.scannerUnsharpSigmaPx = static_cast<float>(params.scannerUnsharpMask[0]);
        input.scannerUnsharpAmount = static_cast<float>(params.scannerUnsharpMask[1]);

        Spektrafilm::DirectRecipeBuildResult built =
            Spektrafilm::build_direct_render_recipe(input);
        if (!built.valid) {
            outError = built.diagnostic.empty()
                           ? "ResourceDescriptorMismatch phase=3A direct recipe build failed"
                           : built.diagnostic;
            return false;
        }

        Scanner::ScannerSpectralLutDescriptor descriptor{};
        std::string descriptorDiagnostic;
        if (!Scanner::build_direct_scanner_spectral_lut_descriptor(
                Scanner::DirectScannerSpectralLutDescriptorInput{
                    &built.recipe.profileRoute,
                    &built.recipe.densityBounds,
                    &built.recipe.scannerOutput},
                descriptor,
                descriptorDiagnostic)) {
            outError = descriptorDiagnostic.empty()
                           ? "ResourceDescriptorMismatch phase=3A scanner descriptor build failed"
                           : descriptorDiagnostic;
            return false;
        }

        out.recipe = std::move(built.recipe);
        if (!build_focused_film_payload(out.recipe, out.payload) ||
            !build_direct_scanner_payload(out.recipe, out.payload)) {
            out = FocusedRenderStateBuildProduct{};
            outError =
                "ResourceDescriptorMismatch focused direct publication payload build failed";
            return false;
        }
        return true;
    }

    bool build_print_render_state_product_impl(
        const ParamSnapshot& params,
        FocusedRenderStateBuildProduct& out,
        std::string& outError) {
        out = FocusedRenderStateBuildProduct{};
        outError.clear();
        if (!Spektrafilm::scan_route_is_print(params.scanRoute)) {
            outError = "ResourceDescriptorMismatch route=print field=scan_route";
            return false;
        }

        JuicerAssets::Library& assets = JuicerProcess::root().assets();
        const JuicerAssets::SelectedProfileResult selected =
            assets.selected_profiles_for_route(
                JuicerAssets::SelectedProfileRequest{
                    params.filmProfileKey,
                    params.printProfileKey,
                    params.scanRoute});
        if (!selected.valid || !selected.filmProfile || !selected.printProfile) {
            outError = selected.diagnostic.empty()
                           ? "MissingRequiredResource phase=4A field=selected_profile"
                           : selected.diagnostic;
            return false;
        }

        Spektrafilm::PrintRecipeBuildInput input{};
        input.film = film_foundation_input_from_snapshot(
            params,
            selected.filmProfile,
            params.scanRoute,
            assets.illuminant_filter_curves());
        input.printProfileKey = params.printProfileKey;
        input.printProfile = selected.printProfile;
        input.dichroic.set = dichroic_filter_set_from_choice(params.enlDichroicSet);
        input.dichroic.setKey = dichroic_filter_set_key(input.dichroic.set);
        if (input.dichroic.set != Spektrafilm::DichroicFilterSet::Custom) {
            input.dichroic.kind = Spektrafilm::DichroicResourceKind::MeasuredCsv;
            input.dichroic.percentTransmittanceDividedBy100 = true;
            input.dichroic.duplicateWavelengthsKeepFirst = true;
            input.dichroic.akimaResampledToReferenceAxis = true;
            const JuicerAssets::MeasuredDichroicResourceIdentity measured =
                assets.measured_dichroic_resource_identity(input.dichroic.setKey);
            if (!measured.valid) {
                outError = measured.diagnostic.empty()
                               ? "MalformedSelectedDichroicResource phase=4A"
                               : measured.diagnostic;
                return false;
            }
            input.dichroic.resourcePathsCmy = measured.resourcePathsCmy;
            input.dichroic.resourceHashesCmy = measured.resourceHashesCmy;
            input.dichroic.hash = measured.hash;
        }

        input.printIlluminantKey = print_illuminant_key_from_choice(params.enlIll);
        const JuicerAssets::NeutralPrintCalibrationResult neutral =
            assets.neutral_print_calibration(
                selected.printProfile->info.stock,
                input.printIlluminantKey,
                selected.filmProfile->info.stock);
        if (neutral.status == JuicerAssets::NeutralPrintCalibrationStatus::Malformed) {
            outError = neutral.diagnostic.empty()
                           ? "MalformedNeutralPrintCalibration phase=4A"
                           : neutral.diagnostic;
            return false;
        }
        switch (neutral.status) {
            case JuicerAssets::NeutralPrintCalibrationStatus::MissingFile:
                input.neutralCalibrationStatus =
                    Spektrafilm::NeutralCalibrationStatus::MissingFile;
                break;
            case JuicerAssets::NeutralPrintCalibrationStatus::Found:
                input.neutralCalibrationStatus =
                    Spektrafilm::NeutralCalibrationStatus::Calibrated;
                input.calibratedNeutralCmyCc =
                    CmyCcTriplet{neutral.cmyCc[0], neutral.cmyCc[1], neutral.cmyCc[2]};
                break;
            case JuicerAssets::NeutralPrintCalibrationStatus::MissingEntry:
            default:
                input.neutralCalibrationStatus =
                    Spektrafilm::NeutralCalibrationStatus::MissingEntry;
                break;
        }
        input.uiYmcCc = {
            static_cast<float>(params.printUiYmcCc[0]),
            static_cast<float>(params.printUiYmcCc[1]),
            static_cast<float>(params.printUiYmcCc[2])};
        input.preflashMFilterCc = static_cast<float>(params.preflashMFilterCc);
        input.preflashYFilterCc = static_cast<float>(params.preflashYFilterCc);
        input.printExposure = static_cast<float>(params.printExposure);
        input.preflashExposure = static_cast<float>(params.printPreflashExposure);
        input.normalizePrintExposure = params.normalizePrintExposure != 0;
        input.printExposureCompensation = params.printExposureCompensation != 0;
        input.scannerLutResolution =
            static_cast<std::uint32_t>(std::clamp(params.scannerLutResolution, 17, 128));
        input.outputColorSpace = params.outputColorSpace;
        input.outputCctfEncoding = params.outputCctfEncoding != 0;
        input.outputLinearPassThrough = params.outputLinearPassThrough != 0;
        input.scannerBlackCorrection = params.scannerBlackCorrection != 0;
        input.scannerWhiteCorrection = params.scannerWhiteCorrection != 0;
        input.scannerBlackLevel = static_cast<float>(params.scannerBlackLevel);
        input.scannerWhiteLevel = static_cast<float>(params.scannerWhiteLevel);
        input.glareActive = params.glareActive;
        input.glarePercent = static_cast<float>(params.glarePercent);
        input.glareRoughness = static_cast<float>(params.glareRoughness);
        input.glareBlurSigmaPx = static_cast<float>(params.glareBlurSigmaPx);
        input.scannerLensBlurSigmaPx = static_cast<float>(params.scannerLensBlurSigmaPx);
        input.scannerUnsharpSigmaPx = static_cast<float>(params.scannerUnsharpMask[0]);
        input.scannerUnsharpAmount = static_cast<float>(params.scannerUnsharpMask[1]);

        Spektrafilm::PrintRecipeBuildResult built =
            Spektrafilm::build_print_render_recipe(input);
        if (!built.valid) {
            outError = built.diagnostic.empty()
                           ? "ResourceDescriptorMismatch phase=4A print recipe build failed"
                           : built.diagnostic;
            return false;
        }
        out.recipe = std::move(built.recipe);
        if (!build_focused_film_payload(out.recipe, out.payload) ||
            !build_print_scanner_payload(out.recipe, out.payload)) {
            out = FocusedRenderStateBuildProduct{};
            outError =
                "ResourceDescriptorMismatch phase=4C focused print publication payload build failed";
            return false;
        }
        return true;
    }
} // namespace

bool build_direct_render_state_product(
    const ParamSnapshot& snapshot,
    FocusedRenderStateBuildProduct& out,
    std::string& outError) {
    return build_direct_render_state_product_impl(snapshot, out, outError);
}

bool build_print_render_state_product(
    const ParamSnapshot& snapshot,
    FocusedRenderStateBuildProduct& out,
    std::string& outError) {
    return build_print_render_state_product_impl(snapshot, out, outError);
}

namespace {
    bool publish_direct_recipe_if_selected(
        const ParamSnapshot& params,
        FocusedRenderStateBuildProduct& outProduct) {
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        if (!build_direct_render_state_product(params, product, diagnostic)) {
            JTRACE("SPEKTRAFILM", diagnostic);
            return false;
        }
        outProduct = std::move(product);
        return true;
    }

    bool publish_print_recipe_if_selected(
        const ParamSnapshot& params,
        FocusedRenderStateBuildProduct& outProduct) {
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        if (!build_print_render_state_product(params, product, diagnostic)) {
            JTRACE("SPEKTRAFILM", diagnostic);
            return false;
        }
        outProduct = std::move(product);
        return true;
    }
} // namespace

uint64_t hash_params(const ParamSnapshot& p) {
    uint64_t h = 0;
    mix_profile_selection_hash_fields(h, p, hash_mix);
    mix_glare_print_hash_fields(h, p, hash_mix);
    mix_coupler_hash_fields(h, p, hash_mix);
    mix_output_encoding_hash_fields(h, p, hash_mix);
    mix_hanatos_adaptation_hash_fields(h, p, hash_mix);
    mix_camera_filter_hash(h, p, hash_mix);
    mix_direct_phase3a_recipe_hash_fields(h, p, hash_mix);
    mix_diffusion_authored_hash_fields(h, p, hash_mix);
    mix_focused_grain_hash_fields(h, p, hash_mix);
    mix_film_juicer_effects_hash_fields(h, p, hash_mix);
    return h;
}

namespace {
    const Spektrafilm::ProfileCatalog& current_profile_catalog() {
        return JuicerProcess::root().assets().spektrafilm_profile_catalog();
    }

    const Spektrafilm::ProfileCatalogEntry* film_profile_entry_for_index(int index) {
        const Spektrafilm::ProfileCatalog& catalog = current_profile_catalog();
        if (index < 0 || index >= static_cast<int>(catalog.filmProfiles.size())) {
            return nullptr;
        }
        return &catalog.filmProfiles[static_cast<std::size_t>(index)];
    }

    const Spektrafilm::ProfileCatalogEntry* print_profile_entry_for_index(int index) {
        const Spektrafilm::ProfileCatalog& catalog = current_profile_catalog();
        if (index < 0 || index >= static_cast<int>(catalog.printProfiles.size())) {
            return nullptr;
        }
        return &catalog.printProfiles[static_cast<std::size_t>(index)];
    }
} // namespace

bool spektrafilm_profile_catalog_ready() {
    return current_profile_catalog().valid;
}

const char* spektrafilm_profile_catalog_failure() {
    const Spektrafilm::ProfileCatalog& catalog = current_profile_catalog();
    return catalog.failure.empty() ? "" : catalog.failure.c_str();
}

int film_profile_option_count() {
    return static_cast<int>(current_profile_catalog().filmProfiles.size());
}

const char* film_profile_option_key(int index) {
    const Spektrafilm::ProfileCatalogEntry* entry = film_profile_entry_for_index(index);
    return entry ? entry->key.c_str() : "";
}

const char* film_profile_option_label(int index) {
    const Spektrafilm::ProfileCatalogEntry* entry = film_profile_entry_for_index(index);
    return entry ? entry->label.c_str() : "";
}

int print_profile_option_count() {
    return static_cast<int>(current_profile_catalog().printProfiles.size());
}

const char* print_profile_option_key(int index) {
    const Spektrafilm::ProfileCatalogEntry* entry = print_profile_entry_for_index(index);
    return entry ? entry->key.c_str() : "";
}

const char* print_profile_option_label(int index) {
    const Spektrafilm::ProfileCatalogEntry* entry = print_profile_entry_for_index(index);
    return entry ? entry->label.c_str() : "";
}

namespace {
    bool rebuild_direct_render_state_for_hash(
        InstanceState& S,
        const ParamSnapshot& P,
        std::uint64_t fullHash) {
        if (Spektrafilm::scan_route_is_print(P.scanRoute)) {
            return false;
        }

        JTRACE_SCOPE("BUILD", "rebuild_direct_render_state");
        std::unique_lock<std::mutex> rebuildLock(S.rebuildMutex);
        if (S.lastHash.load(std::memory_order_acquire) == fullHash) {
            const std::shared_ptr<const DirectRenderState> active =
                JuicerAtomic::load_shared_ptr(&S.activeDirectState);
            if (active && active->buildCounter != 0 && active->recipe.directStructuralReady) {
                return true;
            }
        }
        FocusedRenderStateBuildProduct product;
        if (!publish_direct_recipe_if_selected(P, product)) {
            std::lock_guard<std::mutex> stateLock(S.m);
            JuicerAtomic::store_shared_ptr(
                &S.activeDirectState,
                std::shared_ptr<const DirectRenderState>{});
            return false;
        }

        auto next = std::make_shared<DirectRenderState>();
        next->recipe = std::move(product.recipe);
        next->payload = std::move(product.payload);

        next->buildCounter = S.buildCounterNext.fetch_add(1, std::memory_order_relaxed) + 1;
        {
            std::lock_guard<std::mutex> stateLock(S.m);
            JuicerAtomic::store_shared_ptr(
                &S.activeDirectState,
                std::shared_ptr<const DirectRenderState>(next));
            JuicerAtomic::store_shared_ptr(
                &S.activePrintState,
                std::shared_ptr<const PrintRenderState>{});
            S.lastHash.store(fullHash, std::memory_order_release);
        }
        return true;
    }

    bool rebuild_print_render_state_for_hash(
        InstanceState& S,
        const ParamSnapshot& P,
        std::uint64_t fullHash) {
        if (!Spektrafilm::scan_route_is_print(P.scanRoute)) {
            return false;
        }

        JTRACE_SCOPE("BUILD", "rebuild_print_render_state");
        std::unique_lock<std::mutex> rebuildLock(S.rebuildMutex);
        if (S.lastHash.load(std::memory_order_acquire) == fullHash) {
            const std::shared_ptr<const PrintRenderState> active =
                JuicerAtomic::load_shared_ptr(&S.activePrintState);
            if (active && active->buildCounter != 0 && active->recipe.printStructuralReady) {
                return true;
            }
        }
        FocusedRenderStateBuildProduct product;
        if (!publish_print_recipe_if_selected(P, product)) {
            std::lock_guard<std::mutex> stateLock(S.m);
            JuicerAtomic::store_shared_ptr(
                &S.activePrintState,
                std::shared_ptr<const PrintRenderState>{});
            return false;
        }

        auto next = std::make_shared<PrintRenderState>();
        next->recipe = std::move(product.recipe);
        next->payload = std::move(product.payload);

        next->buildCounter = S.buildCounterNext.fetch_add(1, std::memory_order_relaxed) + 1;
        {
            std::lock_guard<std::mutex> stateLock(S.m);
            JuicerAtomic::store_shared_ptr(
                &S.activeDirectState,
                std::shared_ptr<const DirectRenderState>{});
            JuicerAtomic::store_shared_ptr(
                &S.activePrintState,
                std::shared_ptr<const PrintRenderState>(next));
            S.lastHash.store(fullHash, std::memory_order_release);
        }

        JTRACE_VERBOSE(
            "SPEKTRAFILM",
            "phase=4C focused print state published from validated profiles and RenderRecipe");
        return true;
    }
} // namespace

bool rebuild_direct_render_state(InstanceState& S, const ParamSnapshot& P) {
    return rebuild_direct_render_state_for_hash(S, P, hash_params(P));
}

bool rebuild_print_render_state(InstanceState& S, const ParamSnapshot& P) {
    return rebuild_print_render_state_for_hash(S, P, hash_params(P));
}

PendingRenderAdmissionResult admit_pending_render_state(InstanceState& state) {
    for (;;) {
        ParamSnapshot snapshot;
        std::uint64_t fullHash = 0;
        {
            std::lock_guard<std::mutex> pendingLock(state.pending.m);
            if (std::holds_alternative<PendingParamsState::Uninitialized>(
                    state.pending.value)) {
                return PendingRenderAdmissionResult{};
            }
            if (const auto* invalid =
                    std::get_if<PendingParamsState::InvalidSnapshotControls>(
                        &state.pending.value)) {
                PendingRenderAdmissionResult result;
                result.status = PendingRenderAdmissionStatus::InvalidSnapshotControls;
                result.diagnostic = invalid->diagnostic;
                return result;
            }
            const auto& valid = std::get<PendingParamsState::Valid>(state.pending.value);
            snapshot = valid.params;
            fullHash = valid.fullHash;
        }

        const bool rebuilt = Spektrafilm::scan_route_is_print(snapshot.scanRoute)
                                 ? rebuild_print_render_state_for_hash(
                                       state,
                                       snapshot,
                                       fullHash)
                                 : rebuild_direct_render_state_for_hash(
                                       state,
                                       snapshot,
                                       fullHash);

        std::lock_guard<std::mutex> pendingLock(state.pending.m);
        const auto* current = std::get_if<PendingParamsState::Valid>(&state.pending.value);
        if (!current || current->fullHash != fullHash) {
            continue;
        }

        PendingRenderAdmissionResult result;
        result.snapshot = snapshot;
        if (!rebuilt || state.lastHash.load(std::memory_order_acquire) != fullHash) {
            result.status = PendingRenderAdmissionStatus::RebuildFailed;
            result.diagnostic =
                "ResourceDescriptorMismatch focused render state rebuild failed";
            return result;
        }

        if (Spektrafilm::scan_route_is_print(snapshot.scanRoute)) {
            result.printState = JuicerAtomic::load_shared_ptr(&state.activePrintState);
            if (!result.printState || result.printState->buildCounter == 0 ||
                !result.printState->recipe.printStructuralReady) {
                result.status = PendingRenderAdmissionStatus::RebuildFailed;
                result.diagnostic =
                    "ResourceDescriptorMismatch focused print render state not ready";
                result.printState.reset();
                return result;
            }
            result.status = PendingRenderAdmissionStatus::AdmittedPrint;
            return result;
        }

        result.directState = JuicerAtomic::load_shared_ptr(&state.activeDirectState);
        if (!result.directState || result.directState->buildCounter == 0 ||
            !result.directState->recipe.directStructuralReady) {
            result.status = PendingRenderAdmissionStatus::RebuildFailed;
            result.diagnostic =
                "ResourceDescriptorMismatch focused direct render state not ready";
            result.directState.reset();
            return result;
        }
        result.status = PendingRenderAdmissionStatus::AdmittedDirect;
        return result;
    }
}
