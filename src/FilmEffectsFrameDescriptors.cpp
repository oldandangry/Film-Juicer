#include "FilmEffectsFrameDescriptors.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include "Hash.h"
#include "juicer_cuda_call_scope.h"


namespace {

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    Spektrafilm::VisualGrainScratchShape select_scratch_shape(
        bool sublayers,
        bool sharedChroma) {
        if (sublayers) {
            return sharedChroma
                       ? Spektrafilm::VisualGrainScratchShape::StreamedLayersShared
                       : Spektrafilm::VisualGrainScratchShape::StreamedLayers;
        }
        return sharedChroma
                   ? Spektrafilm::VisualGrainScratchShape::StreamedShared
                   : Spektrafilm::VisualGrainScratchShape::Streamed;
    }

    template <typename Extent>
    void hash_extent(std::uint64_t& hash, const Extent& extent) {
        hash_value(hash, extent.x);
        hash_value(hash, extent.y);
        hash_value(hash, extent.width);
        hash_value(hash, extent.height);
    }

    std::uint64_t hash_descriptor(
        const Spektrafilm::VisualGrainFrameDescriptor& descriptor) {
        if (!descriptor.active) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, descriptor.active);
        hash_value(hash, descriptor.capturePolarity);
        hash_extent(hash, descriptor.renderExtent);
        hash_extent(hash, descriptor.fullFrameExtent);
        hash_value(hash, descriptor.pixelSizeUm);
        hash_value(hash, descriptor.frame0);
        hash_value(hash, descriptor.frameAlpha);
        hash_value(hash, descriptor.seedBase);
        hash_value(hash, descriptor.seedBaseNext);
        hash_value(hash, descriptor.sessionSeed);
        hash_value(hash, descriptor.clipToken);
        hash_value(hash, descriptor.pitchPx);
        hash_value(hash, descriptor.breathingPeriodFrames);
        hash_value(hash, descriptor.clumpMorphPeriodFrames);
        hash_value(hash, descriptor.wangCellMm);
        hash_value(hash, descriptor.breathingAmplitude);
        hash_value(hash, descriptor.breathingCellUmSmall);
        hash_value(hash, descriptor.breathingCellUmLarge);
        hash_value(hash, descriptor.breathingMix);
        hash_value(hash, descriptor.breathingDriftUmPerFrame);
        hash_value(hash, descriptor.debugScale);
        Hash::hash_bytes_update(
            hash,
            descriptor.densityMaxCmy.data(),
            sizeof(descriptor.densityMaxCmy));
        Hash::hash_bytes_update(
            hash,
            descriptor.nParticlesCmy.data(),
            sizeof(descriptor.nParticlesCmy));
        Hash::hash_bytes_update(
            hash,
            descriptor.odParticleCmy.data(),
            sizeof(descriptor.odParticleCmy));
        Hash::hash_bytes_update(
            hash,
            descriptor.densityMinLayers.data(),
            sizeof(descriptor.densityMinLayers));
        Hash::hash_bytes_update(
            hash,
            descriptor.densityMaxLayers.data(),
            sizeof(descriptor.densityMaxLayers));
        Hash::hash_bytes_update(
            hash,
            descriptor.nParticlesLayers.data(),
            sizeof(descriptor.nParticlesLayers));
        Hash::hash_bytes_update(
            hash,
            descriptor.odParticleLayers.data(),
            sizeof(descriptor.odParticleLayers));
        for (const auto& gaussian : descriptor.correlation) {
            hash_value(hash, gaussian.hash);
        }
        for (const auto& layer : descriptor.dyeCloud) {
            for (const auto& gaussian : layer) {
                hash_value(hash, gaussian.hash);
            }
        }
        hash_value(hash, descriptor.effectiveFineWeight);
        hash_value(hash, descriptor.effectiveMidWeight);
        hash_value(hash, descriptor.effectiveCoarseWeight);
        hash_value(hash, descriptor.sizeMixGain);
        hash_value(hash, descriptor.scratchShape);
        hash_value(hash, descriptor.requiresFullFrame);
        hash_value(hash, descriptor.recipeHash);
        hash_value(hash, descriptor.densityCurvesLayersHash);
        return hash;
    }

    bool zero_float(float value) {
        return std::bit_cast<std::uint32_t>(value) == 0;
    }

    bool empty_grain_math(const FjGrainFrameMath& value) {
        if (value.present != 0 || value.capture_polarity != 0 || value.render_extent.x != 0 ||
            value.render_extent.y != 0 || value.render_extent.width != 0 || value.render_extent.height != 0 ||
            value.full_frame_extent.x != 0 || value.full_frame_extent.y != 0 ||
            value.full_frame_extent.width != 0 || value.full_frame_extent.height != 0 ||
            !zero_float(value.pixel_size_um) || value.frame0 != 0 || !zero_float(value.frame_alpha) ||
            value.seed_base != 0 || value.seed_base_next != 0 || value.session_seed != 0 ||
            value.clip_token != 0 || value.pitch_px != 0 || value.breathing_period_frames != 0 ||
            value.clump_morph_period_frames != 0 || !zero_float(value.wang_cell_mm) ||
            !zero_float(value.breathing_amplitude) || !zero_float(value.breathing_cell_um_small) ||
            !zero_float(value.breathing_cell_um_large) || !zero_float(value.breathing_mix) ||
            !zero_float(value.breathing_drift_um_per_frame) || !zero_float(value.debug_scale) ||
            value.requires_full_frame != 0 || value.recipe_hash != 0 || value.density_layers_hash != 0) {
            return false;
        }
        for (const float item : value.density_max_cmy)
            if (!zero_float(item))
                return false;
        for (const float item : value.n_particles_cmy)
            if (!zero_float(item))
                return false;
        for (const float item : value.od_particle_cmy)
            if (!zero_float(item))
                return false;
        for (const auto& row : value.density_min_layers)
            for (const float item : row)
                if (!zero_float(item))
                    return false;
        for (const auto& row : value.density_max_layers)
            for (const float item : row)
                if (!zero_float(item))
                    return false;
        for (const auto& row : value.n_particles_layers)
            for (const float item : row)
                if (!zero_float(item))
                    return false;
        for (const auto& row : value.od_particle_layers)
            for (const float item : row)
                if (!zero_float(item))
                    return false;
        for (const FjGrainGaussian& item : value.correlation) {
            if (!zero_float(item.sigma_px) || item.radius != 0 || item.hash != 0)
                return false;
        }
        for (const auto& row : value.dye_cloud) {
            for (const FjGrainGaussian& item : row) {
                if (!zero_float(item.sigma_px) || item.radius != 0 || item.hash != 0)
                    return false;
            }
        }
        for (const float item : value.size_mix_weights)
            if (!zero_float(item))
                return false;
        return zero_float(value.size_mix_gain);
    }

    bool bind_visual_grain_math(
        const FjGrainFrameMath& math,
        Spektrafilm::ProfilePolarity capturePolarity,
        Spektrafilm::VisualGrainScratchShape scratchShape,
        Spektrafilm::VisualGrainFrameDescriptor& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (math.present != 1 || math.capture_polarity > 1 ||
            math.capture_polarity != static_cast<std::uint32_t>(capturePolarity) ||
            math.recipe_hash == 0 || math.requires_full_frame > 1 ||
            math.render_extent.width <= 0 || math.render_extent.height <= 0 ||
            math.full_frame_extent.width <= 0 || math.full_frame_extent.height <= 0 ||
            math.pitch_px != math.full_frame_extent.height) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result";
            return false;
        }
        const auto gaussian = [](const FjGrainGaussian& value) {
            return value.radius == 0
                       ? (std::bit_cast<std::uint32_t>(value.sigma_px) == 0 && value.hash == 0)
                       : value.radius >= 1 && value.radius <= 75 &&
                             std::isfinite(value.sigma_px) && value.sigma_px > 0.0f;
        };
        bool requiresFullFrame = false;
        for (const FjGrainGaussian& value : math.correlation) {
            if (!gaussian(value)) {
                diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=gaussian";
                return false;
            }
            requiresFullFrame = requiresFullFrame || value.radius > 0;
        }
        for (const auto& layer : math.dye_cloud) {
            for (const FjGrainGaussian& value : layer) {
                if (!gaussian(value)) {
                    diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=gaussian";
                    return false;
                }
                requiresFullFrame = requiresFullFrame || value.radius > 0;
            }
        }
        if (math.requires_full_frame != static_cast<std::uint32_t>(requiresFullFrame) ||
            (requiresFullFrame && (math.render_extent.x != math.full_frame_extent.x ||
                                   math.render_extent.y != math.full_frame_extent.y ||
                                   math.render_extent.width != math.full_frame_extent.width ||
                                   math.render_extent.height != math.full_frame_extent.height))) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=full_frame";
            return false;
        }
        Spektrafilm::VisualGrainFrameDescriptor descriptor{};
        descriptor.active = true;
        descriptor.capturePolarity = capturePolarity;
        descriptor.renderExtent = {math.render_extent.x, math.render_extent.y, math.render_extent.width, math.render_extent.height};
        descriptor.fullFrameExtent = {math.full_frame_extent.x, math.full_frame_extent.y, math.full_frame_extent.width, math.full_frame_extent.height};
        descriptor.pixelSizeUm = math.pixel_size_um;
        descriptor.frame0 = math.frame0;
        descriptor.frameAlpha = math.frame_alpha;
        descriptor.seedBase = math.seed_base;
        descriptor.seedBaseNext = math.seed_base_next;
        descriptor.sessionSeed = math.session_seed;
        descriptor.clipToken = math.clip_token;
        descriptor.pitchPx = math.pitch_px;
        descriptor.breathingPeriodFrames = math.breathing_period_frames;
        descriptor.clumpMorphPeriodFrames = math.clump_morph_period_frames;
        descriptor.wangCellMm = math.wang_cell_mm;
        descriptor.breathingAmplitude = math.breathing_amplitude;
        descriptor.breathingCellUmSmall = math.breathing_cell_um_small;
        descriptor.breathingCellUmLarge = math.breathing_cell_um_large;
        descriptor.breathingMix = math.breathing_mix;
        descriptor.breathingDriftUmPerFrame = math.breathing_drift_um_per_frame;
        descriptor.debugScale = math.debug_scale;
        std::copy_n(math.density_max_cmy, 3, descriptor.densityMaxCmy.begin());
        std::copy_n(math.n_particles_cmy, 3, descriptor.nParticlesCmy.begin());
        std::copy_n(math.od_particle_cmy, 3, descriptor.odParticleCmy.begin());
        for (std::size_t layer = 0; layer < 3; ++layer) {
            std::copy_n(math.density_min_layers[layer], 3, descriptor.densityMinLayers[layer].begin());
            std::copy_n(math.density_max_layers[layer], 3, descriptor.densityMaxLayers[layer].begin());
            std::copy_n(math.n_particles_layers[layer], 3, descriptor.nParticlesLayers[layer].begin());
            std::copy_n(math.od_particle_layers[layer], 3, descriptor.odParticleLayers[layer].begin());
        }
        const auto decodeGaussian = [](const FjGrainGaussian& value) -> Spektrafilm::VisualGrainGaussian {
            return {value.sigma_px, value.radius, value.hash};
        };
        for (std::size_t index = 0; index < 3; ++index) {
            descriptor.correlation[index] = decodeGaussian(math.correlation[index]);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                descriptor.dyeCloud[index][channel] = decodeGaussian(math.dye_cloud[index][channel]);
            }
        }
        descriptor.effectiveFineWeight = math.size_mix_weights[0];
        descriptor.effectiveMidWeight = math.size_mix_weights[1];
        descriptor.effectiveCoarseWeight = math.size_mix_weights[2];
        descriptor.sizeMixGain = math.size_mix_gain;
        descriptor.scratchShape = scratchShape;
        descriptor.requiresFullFrame = requiresFullFrame;
        descriptor.recipeHash = math.recipe_hash;
        descriptor.densityCurvesLayersHash = math.density_layers_hash;
        descriptor.hash = hash_descriptor(descriptor);
        if (descriptor.hash == 0) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=hash";
            return false;
        }
        out = descriptor;
        return true;
    }

} // namespace


namespace Spektrafilm {

    VisualGrainScratchShape visual_grain_scratch_shape(const FjGrainCoefficients& coefficients) noexcept {
        if ((coefficients.flags & FJ_GRAIN_ACTIVE) == 0) {
            return VisualGrainScratchShape::None;
        }
        const bool sharedChroma =
            (coefficients.debug_view == FJ_GRAIN_DEBUG_OFF || coefficients.debug_view == FJ_GRAIN_DEBUG_DELTA_MIX) &&
            coefficients.chroma_mix < 0.999f && coefficients.chroma_shared_weight > 0.0f;
        return select_scratch_shape((coefficients.flags & FJ_GRAIN_SUBLAYERS) != 0, sharedChroma);
    }

    bool compose_visual_grain_frame_descriptor(
        const FjGrainCoefficients& coefficients,
        ProfilePolarity capturePolarity,
        const FjGrainFrameMath& math,
        VisualGrainFrameDescriptor& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        constexpr std::uint32_t allowedFlags = FJ_GRAIN_ACTIVE | FJ_GRAIN_SUBLAYERS |
                                               FJ_GRAIN_AXIS_FINITE_C | FJ_GRAIN_AXIS_FINITE_M |
                                               FJ_GRAIN_AXIS_FINITE_Y;
        if ((coefficients.flags & ~allowedFlags) != 0 ||
            coefficients.debug_view > FJ_GRAIN_DEBUG_MEAN_DENSITY ||
            coefficients.sublayer_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=coefficients";
            return false;
        }
        if ((coefficients.flags & FJ_GRAIN_ACTIVE) == 0) {
            if (coefficients.flags != 0 || coefficients.debug_view != 0 || coefficients.sublayer_count != 0 ||
                !empty_grain_math(math)) {
                diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=inactive";
                return false;
            }
            return true;
        }
        if (capturePolarity == ProfilePolarity::Unsupported ||
            coefficients.sublayer_count == 0 ||
            (((coefficients.flags & FJ_GRAIN_SUBLAYERS) != 0) != (math.density_layers_hash != 0))) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=coefficients";
            return false;
        }
        return bind_visual_grain_math(math, capturePolarity, visual_grain_scratch_shape(coefficients), out, diagnostic);
    }

} // namespace Spektrafilm

namespace JuicerEffects {
    namespace {
        constexpr const char* kInvalidResult = "ResourceDescriptorMismatch phase=effects_descriptor field=rust_result";
        bool zero(float v) {
            return std::bit_cast<std::uint32_t>(v) == 0;
        }
        bool empty(const FjExtent& v) {
            return v.x == 0 && v.y == 0 && v.width == 0 && v.height == 0;
        }
        bool empty(const FjDefectOrigin& v) {
            return v.cell_x == 0 && v.cell_y == 0 && zero(v.local_x_mm) && zero(v.local_y_mm);
        }
        bool empty(const FjEffectsFrameMath& v) {
            return v.present == 0 && v.flags == 0 && empty(v.render_extent) && empty(v.full_frame_extent) && std::all_of(std::begin(v.origins), std::end(v.origins), [](const auto& o) {
                       return empty(o);
                   }) &&
                   zero(v.sample_step_x_mm) && zero(v.sample_step_y_mm) && v.roi_offset_x == 0 && v.roi_offset_y == 0 && v.gate_width == 0 && v.gate_height == 0 && v.session_seed == 0 && v.clip_token == 0 && zero(v.weave_dx_px) && zero(v.weave_dy_px) && zero(v.weave_cos_rot) && zero(v.weave_sin_rot) && v.recipe_hash == 0 && v.hash == 0;
        }
        template <typename Extent>
        bool same_extent(const FjExtent& a, const Extent& b) {
            return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
        }
        DefectDustRecipe native_dust(const FjDust& v) {
            DefectDustRecipe out{};
            out.cellWidthMm = v.cell_width_mm;
            out.cellHeightMm = v.cell_height_mm;
            out.slotProbability = v.slot_probability;
            out.softnessMinMm = v.softness_min_mm;
            out.softnessMaxMm = v.softness_max_mm;
            out.softnessSizeCapFraction = v.softness_size_cap_fraction;
            out.supportXMm = v.support_x_mm;
            out.supportYMm = v.support_y_mm;
            out.fiberFraction = v.fiber_fraction;
            out.fiberDriftFraction = v.fiber_drift_fraction;
            out.fiberFirstKnotMin = v.fiber_first_knot_min;
            out.fiberFirstKnotMax = v.fiber_first_knot_max;
            out.fiberSecondKnotMin = v.fiber_second_knot_min;
            out.fiberSecondKnotMax = v.fiber_second_knot_max;
            out.fiberInteriorWidthMinFraction = v.fiber_interior_width_min_fraction;
            out.fiberInteriorWidthMaxFraction = v.fiber_interior_width_max_fraction;
            out.diameterMinMm = v.diameter_min_mm;
            out.diameterBulkMaxMm = v.diameter_bulk_max_mm;
            out.diameterMaxMm = v.diameter_max_mm;
            out.diameterTailFraction = v.diameter_tail_fraction;
            out.fiberLengthMinMm = v.fiber_length_min_mm;
            out.fiberLengthMaxMm = v.fiber_length_max_mm;
            out.fiberWidthMinMm = v.fiber_width_min_mm;
            out.fiberWidthMaxMm = v.fiber_width_max_mm;
            out.opacityFaintCumulative = v.opacity_faint_cumulative;
            out.opacityIntermediateCumulative = v.opacity_intermediate_cumulative;
            out.compactOpacityMin = v.compact_opacity_min;
            out.compactOpacityFaintEnd = v.compact_opacity_faint_end;
            out.compactOpacityIntermediateEnd = v.compact_opacity_intermediate_end;
            out.compactOpacityMax = v.compact_opacity_max;
            out.fiberOpacityMin = v.fiber_opacity_min;
            out.fiberOpacityFaintEnd = v.fiber_opacity_faint_end;
            out.fiberOpacityIntermediateEnd = v.fiber_opacity_intermediate_end;
            out.fiberOpacityMax = v.fiber_opacity_max;
            out.compactDominantAspectMin = v.compact_dominant_aspect_min;
            out.compactDominantAspectMax = v.compact_dominant_aspect_max;
            out.compactSubsidiaryScaleMin = v.compact_subsidiary_scale_min;
            out.compactSubsidiaryScaleMax = v.compact_subsidiary_scale_max;
            out.compactSubsidiaryAspectMin = v.compact_subsidiary_aspect_min;
            out.compactSubsidiaryAspectMax = v.compact_subsidiary_aspect_max;
            out.compactSubsidiaryOffsetMax = v.compact_subsidiary_offset_max;
            out.compactSubsidiaryAngleMaxRadians = v.compact_subsidiary_angle_max_radians;
            return out;
        }
        DefectScratchRecipe native_scratch(const FjScratch& v) {
            DefectScratchRecipe out{};
            out.cellWidthMm = v.cell_width_mm;
            out.cellHeightMm = v.cell_height_mm;
            out.slotProbability = v.slot_probability;
            out.softnessMinMm = v.softness_min_mm;
            out.softnessMaxMm = v.softness_max_mm;
            out.softnessSizeCapFraction = v.softness_size_cap_fraction;
            out.supportXMm = v.support_x_mm;
            out.supportYMm = v.support_y_mm;
            out.lengthMinMm = v.length_min_mm;
            out.lengthBulkMaxMm = v.length_bulk_max_mm;
            out.lengthMaxMm = v.length_max_mm;
            out.lengthTailFraction = v.length_tail_fraction;
            out.widthMinMm = v.width_min_mm;
            out.widthBulkMaxMm = v.width_bulk_max_mm;
            out.widthMaxMm = v.width_max_mm;
            out.widthTailFraction = v.width_tail_fraction;
            out.driftFraction = v.drift_fraction;
            out.firstKnotMin = v.first_knot_min;
            out.firstKnotMax = v.first_knot_max;
            out.secondKnotMin = v.second_knot_min;
            out.secondKnotMax = v.second_knot_max;
            out.interiorWidthMinFraction = v.interior_width_min_fraction;
            out.interiorWidthMaxFraction = v.interior_width_max_fraction;
            out.interiorDepthMinFraction = v.interior_depth_min_fraction;
            out.interiorDepthMaxFraction = v.interior_depth_max_fraction;
            out.endpointAbruptProbability = v.endpoint_abrupt_probability;
            out.interruptionProbability = v.interruption_probability;
            out.gapCenterMin = v.gap_center_min;
            out.gapCenterMax = v.gap_center_max;
            out.gapSpanMin = v.gap_span_min;
            out.gapSpanMax = v.gap_span_max;
            out.scuffProbability = v.scuff_probability;
            out.scuffLengthMaxMm = v.scuff_length_max_mm;
            out.scuffAngleMaxRadians = v.scuff_angle_max_radians;
            out.strengthMin = v.strength_min;
            out.strengthMax = v.strength_max;
            return out;
        }

    } // namespace

    bool compose_frame_descriptor(
        const FjEffectsFrameMath& math,
        const FjDust& filmDust,
        const FjScratch& filmScratch,
        const FjDust& gateDust,
        const FjScratch& gateScratch,
        Spektrafilm::FilmJuicerEffectsFrameDescriptor& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (math.present == 0) {
            if (empty(math)) {
                return true;
            }
            diagnostic = kInvalidResult;
            return false;
        }
        if (math.present != 1 || (math.flags & ~31u) != 0 ||
            math.render_extent.width <= 0 || math.render_extent.height <= 0 ||
            math.full_frame_extent.width <= 0 || math.full_frame_extent.height <= 0) {
            diagnostic = kInvalidResult;
            return false;
        }
        const bool weave = (math.flags & FJ_EFFECTS_WEAVE) != 0;
        const bool film = (math.flags & FJ_EFFECTS_FILM) != 0;
        const bool gate = (math.flags & FJ_EFFECTS_GATE_TRANSMITTANCE) != 0;
        const bool output = (math.flags & FJ_EFFECTS_GATE_OUTPUT) != 0;
        const bool full = (math.flags & FJ_EFFECTS_FULL_FRAME) != 0;
        if (output != (weave || gate) || full != weave ||
            (weave && !same_extent(math.render_extent, math.full_frame_extent)) ||
            ((film || gate) && (!std::isfinite(math.sample_step_x_mm) || !(math.sample_step_x_mm > 0) ||
                                !std::isfinite(math.sample_step_y_mm) || !(math.sample_step_y_mm > 0))) ||
            (!(film || gate) && (!zero(math.sample_step_x_mm) || !zero(math.sample_step_y_mm))) ||
            math.roi_offset_x < 0 || math.roi_offset_y < 0 ||
            (gate ? (math.gate_width <= 0 || math.gate_height <= 0 ||
                     math.gate_width > math.full_frame_extent.width || math.gate_height > math.full_frame_extent.height)
                  : (math.gate_width != 0 || math.gate_height != 0)) ||
            !std::isfinite(math.weave_dx_px) || !std::isfinite(math.weave_dy_px) ||
            !std::isfinite(math.weave_cos_rot) || !std::isfinite(math.weave_sin_rot) ||
            (!weave && (!zero(math.weave_dx_px) || !zero(math.weave_dy_px) ||
                        math.weave_cos_rot != 1.0f || !zero(math.weave_sin_rot)))) {
            diagnostic = kInvalidResult;
            return false;
        }
        for (const FjDefectOrigin& origin : math.origins) {
            if (std::abs(static_cast<double>(origin.cell_x)) >= 4503599627370496.0 ||
                std::abs(static_cast<double>(origin.cell_y)) >= 4503599627370496.0 ||
                !std::isfinite(origin.local_x_mm) || origin.local_x_mm < 0 ||
                !std::isfinite(origin.local_y_mm) || origin.local_y_mm < 0) {
                diagnostic = kInvalidResult;
                return false;
            }
        }
        out.renderExtent = {math.render_extent.x, math.render_extent.y, math.render_extent.width, math.render_extent.height};
        out.fullFrameExtent = {math.full_frame_extent.x, math.full_frame_extent.y, math.full_frame_extent.width, math.full_frame_extent.height};
        out.filmDust = native_dust(filmDust);
        out.filmScratch = native_scratch(filmScratch);
        out.gateDust = native_dust(gateDust);
        out.gateScratch = native_scratch(gateScratch);
        for (std::size_t i = 0; i < out.origins.size(); ++i) {
            const auto& origin = math.origins[i];
            out.origins[i] = {origin.cell_x, origin.cell_y, origin.local_x_mm, origin.local_y_mm};
        }
        out.sampleStepXMm = math.sample_step_x_mm;
        out.sampleStepYMm = math.sample_step_y_mm;
        out.roiOffsetX = math.roi_offset_x;
        out.roiOffsetY = math.roi_offset_y;
        out.gateWidth = math.gate_width;
        out.gateHeight = math.gate_height;
        out.sessionSeed = math.session_seed;
        out.clipToken = math.clip_token;
        out.weaveActive = weave;
        out.filmActive = film;
        out.gateTransmittanceActive = gate;
        out.gateOutputActive = output;
        out.requiresFullFrame = full;
        out.weaveDxPx = math.weave_dx_px;
        out.weaveDyPx = math.weave_dy_px;
        out.weaveCosRot = math.weave_cos_rot;
        out.weaveSinRot = math.weave_sin_rot;
        out.recipeHash = math.recipe_hash;
        out.hash = math.hash;
        return true;
    }
} // namespace JuicerEffects
