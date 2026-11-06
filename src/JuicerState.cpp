#include "JuicerState.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "Hash.h"
#include "ProcessRoot.h"

namespace {

    inline bool is_finite(float value) {
        return std::isfinite(value);
    }

    inline bool is_finite(double value) {
        return std::isfinite(value);
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

    inline std::uint64_t exact_double_bits(double value) {
        return std::bit_cast<std::uint64_t>(value);
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

    template <typename MixFn, typename TValue>
    inline void mix_retained_float_bits_if_finite(
        uint64_t& h,
        TValue value,
        const MixFn& mix) {
        if (!is_finite(value)) {
            return;
        }
        mix_hash_field(h, std::bit_cast<std::uint32_t>(static_cast<float>(value)), mix);
    }

    template <typename MixFn, typename TValue>
    inline void mix_canonical_retained_float_bits_if_finite(
        uint64_t& h,
        TValue value,
        const MixFn& mix) {
        if (!is_finite(value)) {
            return;
        }
        mix_canonical_float_bits(h, static_cast<float>(value), mix);
    }

    template <typename MixFn>
    inline void mix_retained_double_bits_if_finite(
        uint64_t& h,
        double value,
        const MixFn& mix) {
        if (!is_finite(value)) {
            return;
        }
        mix_hash_field(h, exact_double_bits(value), mix);
    }

    template <typename MixFn>
    inline void mix_glare_print_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        mix_hash_field_scaled(h, p.printShadowCompensationFactor, 10000.0, mix);
        mix_hash_field_scaled(h, p.printShadowCompensationDensity, 10000.0, mix);
        mix_hash_field_scaled(h, p.printShadowCompensationTransition, 10000.0, mix);
        mix_hash_field(h, p.glareActive ? 1 : 0, mix);
        mix_retained_float_bits_if_finite(h, p.glarePercent, mix);
        mix_retained_float_bits_if_finite(h, p.glareRoughness, mix);
        mix_retained_float_bits_if_finite(h, p.glareBlurSigmaPx, mix);
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
                mix_retained_float_bits_if_finite(h, *values, mix);
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
        mix_hash_string(h, p.filmProfileKey, mix);
        mix_scan_route_hash_field(h, p, mix);
        if (Spektrafilm::scan_route_is_print(p.scanRoute)) {
            mix_hash_string(h, p.printProfileKey, mix);
            mix_hash_field(h, p.enlIll, mix);
            mix_retained_float_bits_if_finite(h, p.printExposure, mix);
            mix_retained_float_bits_if_finite(h, p.printPreflashExposure, mix);
            mix_hash_field(h, exact_double_bits(p.printGammaFactor), mix);
            mix_hash_field(h, p.normalizePrintExposure, mix);
            mix_hash_field(h, p.printExposureCompensation, mix);
            for (double value : p.printUiYmcCc) {
                mix_canonical_retained_float_bits_if_finite(h, value, mix);
            }
            mix_canonical_retained_float_bits_if_finite(h, p.preflashMFilterCc, mix);
            mix_canonical_retained_float_bits_if_finite(h, p.preflashYFilterCc, mix);
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
        mix_hash_field(h, p.outputGamutCompressionEnabled, mix);
    }

    template <typename MixFn>
    inline void mix_hanatos_adaptation_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        if (p.spectralUpsamplingMode != 0) {
            return;
        }
        mix_hash_field(h, p.hanatos2025AdaptationWindow, mix);
        mix_hash_field(h, p.hanatos2025AdaptationSurface, mix);
    }

    template <typename MixFn>
    inline void mix_input_compression_hash_field(
        uint64_t& h,
        const ParamSnapshot& p,
        const MixFn& mix) {
        if (p.spectralUpsamplingMode == 0 || p.spectralUpsamplingMode == 2) {
            mix_hash_field(h, p.inputCompressionEnabled, mix);
        }
    }

    template <typename MixFn>
    inline void mix_direct_phase3a_recipe_hash_fields(
        uint64_t& h,
        const ParamSnapshot& p,
        const MixFn& mix) {
        mix_hash_field(h, p.cameraAutoExposureEnabled, mix);
        mix_hash_field(h, p.cameraMeteringMethod, mix);
        mix_retained_float_bits_if_finite(h, p.cameraExposureCompensationEv, mix);
        mix_canonical_float_bits(h, p.cameraFilmFormatLongEdgeMm, mix);
        mix_canonical_float_bits(h, p.filmGammaFactor, mix);
        mix_hash_field(h, p.scatterHalationControls.active ? 1 : 0, mix);
        if (p.scatterHalationControls.active) {
            mix_canonical_float_bits(h, p.scatterHalationControls.scatterAmount, mix);
            mix_canonical_float_bits(h, p.scatterHalationControls.scatterSpatialScale, mix);
            mix_canonical_float_bits(h, p.scatterHalationControls.halationAmount, mix);
            mix_canonical_float_bits(h, p.scatterHalationControls.halationSpatialScale, mix);
        }
        mix_retained_float_bits_if_finite(h, p.scannerLensBlurSigmaPx, mix);
        mix_retained_float_bits_if_finite(h, p.scannerUnsharpMask[0], mix);
        mix_retained_float_bits_if_finite(h, p.scannerUnsharpMask[1], mix);
        mix_hash_field(h, p.scannerBlackCorrection, mix);
        mix_hash_field(h, p.scannerWhiteCorrection, mix);
        mix_retained_float_bits_if_finite(h, p.scannerBlackLevel, mix);
        mix_retained_float_bits_if_finite(h, p.scannerWhiteLevel, mix);
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
        mix_retained_float_bits_if_finite(h, grain.particleAreaUm2, mix);
        for (float value : grain.particleScaleCmy) {
            mix_retained_float_bits_if_finite(h, value, mix);
        }
        for (float value : grain.particleScaleLayers) {
            mix_retained_float_bits_if_finite(h, value, mix);
        }
        for (float value : grain.visualParticleDensityMinCmy) {
            mix_retained_float_bits_if_finite(h, value, mix);
        }
        for (float value : grain.uniformityCmy) {
            mix_retained_float_bits_if_finite(h, value, mix);
        }
        mix_retained_float_bits_if_finite(h, grain.correlationSigmaPx, mix);
        mix_retained_float_bits_if_finite(h, grain.dyeCloudBlurUm, mix);
        for (float value : grain.microStructure) {
            mix_retained_float_bits_if_finite(h, value, mix);
        }
        mix_hash_field(h, grain.nSubLayers, mix);
        mix_retained_float_bits_if_finite(h, grain.amplitude, mix);
        mix_retained_float_bits_if_finite(h, grain.chromaSharedWeight, mix);
        mix_retained_float_bits_if_finite(h, grain.chromaIndependentWeight, mix);
        mix_retained_float_bits_if_finite(h, grain.coarseWeight, mix);
        mix_retained_float_bits_if_finite(h, grain.midWeight, mix);
        mix_retained_float_bits_if_finite(h, grain.sizeMixScale, mix);
        mix_retained_float_bits_if_finite(h, grain.clumpTemporalMix, mix);
        mix_retained_float_bits_if_finite(h, grain.clumpMorphPeriodSec, mix);
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
        mix_canonical_retained_float_bits_if_finite(h, p.filmDustAmount, mix);
        mix_canonical_retained_float_bits_if_finite(h, p.filmScratchAmount, mix);
        mix_canonical_retained_float_bits_if_finite(h, p.gateDustAmount, mix);
        mix_canonical_retained_float_bits_if_finite(h, p.gateScratchAmount, mix);
        mix_retained_double_bits_if_finite(h, p.gateWeaveAmount, mix);
    }

    template <typename MixFn>
    inline void mix_coupler_hash_fields(uint64_t& h, const ParamSnapshot& p, const MixFn& mix) {
        const Spektrafilm::DirCouplersControls& controls = p.dirCouplers;
        mix_hash_field(h, controls.active, mix);
        mix_canonical_float_bits(h, controls.amount, mix);
        mix_canonical_float_bits(h, controls.inhibitionSameLayer, mix);
        mix_canonical_float_bits(h, controls.inhibitionInterlayer, mix);
        mix_canonical_float_bits(h, controls.diffusionSizeUm, mix);
        mix_canonical_float_bits(h, controls.diffusionTailUm, mix);
        mix_canonical_float_bits(h, controls.diffusionTailWeight, mix);
        mix_hash_field(h, controls.gammaUseStock, mix);
        for (float value : controls.gammaSameLayerRgb) {
            mix_canonical_float_bits(h, value, mix);
        }
        for (float value : controls.gammaInterlayerRToGb) {
            mix_canonical_float_bits(h, value, mix);
        }
        for (float value : controls.gammaInterlayerGToRb) {
            mix_canonical_float_bits(h, value, mix);
        }
        for (float value : controls.gammaInterlayerBToRg) {
            mix_canonical_float_bits(h, value, mix);
        }
        for (float value : controls.langmuirDonorKRgb) {
            mix_canonical_float_bits(h, value, mix);
        }
        for (float value : controls.langmuirReceiverKRgb) {
            mix_canonical_float_bits(h, value, mix);
        }
    }

} // namespace

bool set_gamma_snapshot_values(
    double authoredFilmGammaFactor,
    double authoredPrintGammaFactor,
    ParamSnapshot& snapshot,
    std::string& outDiagnostic) {
    outDiagnostic.clear();
    if (!std::isfinite(authoredFilmGammaFactor) ||
        authoredFilmGammaFactor < Spektrafilm::kFilmGammaFactorMinimum ||
        authoredFilmGammaFactor > Spektrafilm::kFilmGammaFactorMaximum) {
        outDiagnostic =
            "InvalidAuthoredControl component=film_development field=film_gamma_factor";
        return false;
    }

    const float retainedFilmGammaFactor =
        static_cast<float>(authoredFilmGammaFactor);
    if (Spektrafilm::scan_route_is_print(snapshot.scanRoute) &&
        (!std::isfinite(authoredPrintGammaFactor) ||
         authoredPrintGammaFactor < Spektrafilm::kPrintGammaFactorMinimum ||
         authoredPrintGammaFactor > Spektrafilm::kPrintGammaFactorMaximum)) {
        outDiagnostic =
            "InvalidAuthoredControl component=print_development field=print_gamma_factor";
        return false;
    }

    snapshot.filmGammaFactor = retainedFilmGammaFactor;
    snapshot.printGammaFactor = authoredPrintGammaFactor;
    return true;
}

uint64_t hash_params(const ParamSnapshot& p) {
    uint64_t h = 0;
    mix_profile_selection_hash_fields(h, p, hash_mix);
    mix_glare_print_hash_fields(h, p, hash_mix);
    mix_coupler_hash_fields(h, p, hash_mix);
    mix_output_encoding_hash_fields(h, p, hash_mix);
    mix_hanatos_adaptation_hash_fields(h, p, hash_mix);
    mix_input_compression_hash_field(h, p, hash_mix);
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
