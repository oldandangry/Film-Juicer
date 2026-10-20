#include "RustFilmRecipeBridge.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <span>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string_view>

#include "Cuda/JuicerCudaExecutor.h"
#include "Logging.h"
#include "RenderRecipe.h"
#include "juicer_legacy_api.h"

namespace JuicerAssets {
#if defined(JUICER_FILM_RECIPE_TEST_HOOK)
    namespace RecipeTest {
        void after_call(unsigned operation, FjStatus& status, FjFilmRecipeFailure& failure);
        void view(FjFilmDirView& view);
        void grain(FjVisualGrainRecipe& recipe);
        void effects(FjPhysicalEffectsRecipe& recipe);
        void optics(FjStatus& status, FjSpatialOpticsFailure& failure, FjSpatialOpticsRecipe& recipe, FjErrorBuffer& error);
    } // namespace RecipeTest
#endif
    namespace {
        enum class Operation : std::uint8_t {
            Minimum,
            Acquire,
            View,
            Bounds,
            Grain,
            Effects
        };

        [[noreturn]] void integrity(std::string_view operation) {
            JuicerCuda::Failure failure;
            const std::string message = std::string(operation) + ": invalid failure contract";
            JuicerCuda::set_failure(failure, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, message);
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        const char* operation_name(Operation operation) {
            if (operation == Operation::Grain)
                return "Visual grain preparation";
            return operation == Operation::Minimum ? "Film density minimum preparation" : operation == Operation::Bounds ? "Film density bounds preparation"
                                                                                                                         : "Film DIR preparation";
        }
        bool valid_failure(Operation operation, FjStatus status, const FjFilmRecipeFailure& failure, bool positive, std::size_t samples) {
            if (status.api != FJ_API_NONE || status.native_code != 0)
                return false;
            const bool noChannel = failure.channel == 3;
            const bool noSample = failure.sample == UINT64_MAX;
            const bool acquire = operation == Operation::Acquire;
            const bool bounds = operation == Operation::Bounds;
            switch (failure.reason) {
                case FJ_FILM_RECIPE_REASON_NONE:
                    return status.category == FJ_STATUS_SUCCESS && noChannel && noSample;
                case FJ_FILM_RECIPE_RAW:
                    return status.category == FJ_STATUS_UNSUPPORTED_INPUT && noChannel && noSample;
                case FJ_FILM_RECIPE_PANIC:
                case FJ_FILM_RECIPE_INTEGRITY:
                    return status.category == FJ_STATUS_INTERNAL_FAILURE && noChannel && noSample;
                case FJ_FILM_RECIPE_SIZE:
                case FJ_FILM_RECIPE_ALLOCATION:
                    return acquire && status.category == FJ_STATUS_ALLOCATION_FAILURE && noChannel && noSample;
                default:
                    break;
            }
            if (status.category != FJ_STATUS_PREPARATION_FAILURE)
                return false;
            switch (failure.reason) {
                case FJ_FILM_RECIPE_GRAIN_CONTROLS:
                case FJ_FILM_RECIPE_GRAIN_LAYERS:
                    return operation == Operation::Grain && noChannel && noSample;
                case FJ_FILM_RECIPE_MINIMUM:
                    return (bounds || operation == Operation::Minimum) && noChannel && noSample;
                case FJ_FILM_RECIPE_IDENTITY:
                    return (acquire || bounds || operation == Operation::Grain) && noChannel && noSample;
                case FJ_FILM_RECIPE_BOUNDS_SPAN:
                    return bounds && failure.channel < 3 && noSample;
                case FJ_FILM_RECIPE_MATRIX:
                case FJ_FILM_RECIPE_DENSITY_MAX:
                    return acquire && noChannel && noSample;
                case FJ_FILM_RECIPE_LOG_EXPOSURE:
                    return acquire && noChannel && failure.sample >= 1 && failure.sample < samples;
                case FJ_FILM_RECIPE_RECEIVER_K:
                    return acquire && positive && failure.channel < 3 && noSample;
                case FJ_FILM_RECIPE_DONOR_K:
                    return acquire && !positive && failure.channel < 3 && noSample;
                case FJ_FILM_RECIPE_DONOR_LANGMUIR:
                    return acquire && !positive && failure.channel < 3 && failure.sample < samples;
                case FJ_FILM_RECIPE_RECEIVER_LANGMUIR:
                    return acquire && positive && failure.channel < 3 && failure.sample < samples;
                case FJ_FILM_RECIPE_COMPENSATED_AXIS:
                    return acquire && failure.channel < 3 && failure.sample < samples;
                default:
                    return false;
            }
        }
        bool translate(Operation operation, FjStatus status, const FjFilmRecipeFailure& failure, const Profiles::FilmProfile* profile, Spektrafilm::ScanRoute route, std::size_t samples, std::string& diagnostic) {
            const bool positive = profile && profile->info.type == Spektrafilm::ProfilePolarity::Positive;
            if (!valid_failure(operation, status, failure, positive, samples))
                integrity(operation_name(operation));
            if (status.category == FJ_STATUS_SUCCESS)
                return true;
            if (status.category == FJ_STATUS_ALLOCATION_FAILURE)
                throw std::bad_alloc();
            if (status.category != FJ_STATUS_PREPARATION_FAILURE) {
                JuicerCuda::Failure error;
                JuicerCuda::set_failure(error, status, operation_name(operation));
                throw JuicerCuda::ExecutionFailure{std::move(error)};
            }
            diagnostic.clear();
            if (operation == Operation::Grain) {
                diagnostic = "MalformedRequiredProfileData phase=9A field=visual_grain";
                return false;
            }
            if (operation == Operation::Minimum) {
                diagnostic = "MalformedRequiredProfileData phase=9A field=grain_contract";
                return false;
            }
            if (operation == Operation::Bounds) {
                diagnostic = "MalformedRequiredProfileData phase=3A field=density_bounds";
                return false;
            }
            if (failure.reason == FJ_FILM_RECIPE_IDENTITY)
                return false;
            const char* field = nullptr;
            const char* requirement = nullptr;
            switch (failure.reason) {
                case FJ_FILM_RECIPE_MATRIX:
                    field = "matrix";
                    requirement = "finite";
                    break;
                case FJ_FILM_RECIPE_LOG_EXPOSURE:
                    field = "log_exposure";
                    requirement = "strictly_increasing";
                    break;
                case FJ_FILM_RECIPE_DENSITY_MAX:
                    field = "density_max";
                    requirement = "finite_positive";
                    break;
                case FJ_FILM_RECIPE_RECEIVER_K:
                    field = "receiver_k";
                    requirement = "finite_positive_when_coupled";
                    break;
                case FJ_FILM_RECIPE_DONOR_K:
                    field = "donor_k";
                    requirement = "finite_positive";
                    break;
                case FJ_FILM_RECIPE_DONOR_LANGMUIR:
                    field = "donor_langmuir";
                    requirement = "finite_defined";
                    break;
                case FJ_FILM_RECIPE_RECEIVER_LANGMUIR:
                    field = "receiver_langmuir";
                    requirement = "finite_defined";
                    break;
                case FJ_FILM_RECIPE_COMPENSATED_AXIS:
                    field = "compensated_axis";
                    requirement = "finite_float_representable";
                    break;
                default:
                    integrity(operation_name(operation));
            }
            std::ostringstream text;
            text << "component=dir route=" << Spektrafilm::scan_route_key(route) << " profile=" << profile->info.stock << " field=" << field;
            if (failure.channel < 3)
                text << " channel=" << failure.channel;
            if (failure.sample != UINT64_MAX)
                text << " sample=" << failure.sample;
            text << " requirement=" << requirement;
            diagnostic = text.str();
            return false;
        }
        bool admit(Operation operation, FjStatus status, const FjFilmRecipeFailure& failure, const FjErrorBuffer& error, const Profiles::FilmProfile* profile, Spektrafilm::ScanRoute route, std::size_t samples, std::string& diagnostic) {
            if (error.length >= error.capacity)
                integrity(operation_name(operation));
            return translate(operation, status, failure, profile, route, samples, diagnostic);
        }
        bool project_policy(const FjDust& source, DefectDustRecipe& out) {
            const std::array coefficients{
                source.cell_width_mm,
                source.cell_height_mm,
                source.slot_probability,
                source.softness_min_mm,
                source.softness_max_mm,
                source.softness_size_cap_fraction,
                source.support_x_mm,
                source.support_y_mm,
                source.fiber_fraction,
                source.fiber_drift_fraction,
                source.fiber_first_knot_min,
                source.fiber_first_knot_max,
                source.fiber_second_knot_min,
                source.fiber_second_knot_max,
                source.fiber_interior_width_min_fraction,
                source.fiber_interior_width_max_fraction,
                source.diameter_min_mm,
                source.diameter_bulk_max_mm,
                source.diameter_max_mm,
                source.diameter_tail_fraction,
                source.fiber_length_min_mm,
                source.fiber_length_max_mm,
                source.fiber_width_min_mm,
                source.fiber_width_max_mm,
                source.opacity_faint_cumulative,
                source.opacity_intermediate_cumulative,
                source.compact_opacity_min,
                source.compact_opacity_faint_end,
                source.compact_opacity_intermediate_end,
                source.compact_opacity_max,
                source.fiber_opacity_min,
                source.fiber_opacity_faint_end,
                source.fiber_opacity_intermediate_end,
                source.fiber_opacity_max,
                source.compact_dominant_aspect_min,
                source.compact_dominant_aspect_max,
                source.compact_subsidiary_scale_min,
                source.compact_subsidiary_scale_max,
                source.compact_subsidiary_aspect_min,
                source.compact_subsidiary_aspect_max,
                source.compact_subsidiary_offset_max,
                source.compact_subsidiary_angle_max_radians};
            if (!std::all_of(coefficients.begin(), coefficients.end(), [](float coefficient) {
                    return std::isfinite(coefficient);
                }))
                return false;
            out.cellWidthMm = source.cell_width_mm;
            out.cellHeightMm = source.cell_height_mm;
            out.slotProbability = source.slot_probability;
            out.softnessMinMm = source.softness_min_mm;
            out.softnessMaxMm = source.softness_max_mm;
            out.softnessSizeCapFraction = source.softness_size_cap_fraction;
            out.supportXMm = source.support_x_mm;
            out.supportYMm = source.support_y_mm;
            out.fiberFraction = source.fiber_fraction;
            out.fiberDriftFraction = source.fiber_drift_fraction;
            out.fiberFirstKnotMin = source.fiber_first_knot_min;
            out.fiberFirstKnotMax = source.fiber_first_knot_max;
            out.fiberSecondKnotMin = source.fiber_second_knot_min;
            out.fiberSecondKnotMax = source.fiber_second_knot_max;
            out.fiberInteriorWidthMinFraction = source.fiber_interior_width_min_fraction;
            out.fiberInteriorWidthMaxFraction = source.fiber_interior_width_max_fraction;
            out.diameterMinMm = source.diameter_min_mm;
            out.diameterBulkMaxMm = source.diameter_bulk_max_mm;
            out.diameterMaxMm = source.diameter_max_mm;
            out.diameterTailFraction = source.diameter_tail_fraction;
            out.fiberLengthMinMm = source.fiber_length_min_mm;
            out.fiberLengthMaxMm = source.fiber_length_max_mm;
            out.fiberWidthMinMm = source.fiber_width_min_mm;
            out.fiberWidthMaxMm = source.fiber_width_max_mm;
            out.opacityFaintCumulative = source.opacity_faint_cumulative;
            out.opacityIntermediateCumulative = source.opacity_intermediate_cumulative;
            out.compactOpacityMin = source.compact_opacity_min;
            out.compactOpacityFaintEnd = source.compact_opacity_faint_end;
            out.compactOpacityIntermediateEnd = source.compact_opacity_intermediate_end;
            out.compactOpacityMax = source.compact_opacity_max;
            out.fiberOpacityMin = source.fiber_opacity_min;
            out.fiberOpacityFaintEnd = source.fiber_opacity_faint_end;
            out.fiberOpacityIntermediateEnd = source.fiber_opacity_intermediate_end;
            out.fiberOpacityMax = source.fiber_opacity_max;
            out.compactDominantAspectMin = source.compact_dominant_aspect_min;
            out.compactDominantAspectMax = source.compact_dominant_aspect_max;
            out.compactSubsidiaryScaleMin = source.compact_subsidiary_scale_min;
            out.compactSubsidiaryScaleMax = source.compact_subsidiary_scale_max;
            out.compactSubsidiaryAspectMin = source.compact_subsidiary_aspect_min;
            out.compactSubsidiaryAspectMax = source.compact_subsidiary_aspect_max;
            out.compactSubsidiaryOffsetMax = source.compact_subsidiary_offset_max;
            out.compactSubsidiaryAngleMaxRadians = source.compact_subsidiary_angle_max_radians;
            return true;
        }
        bool project_policy(const FjScratch& source, DefectScratchRecipe& out) {
            const std::array coefficients{
                source.cell_width_mm,
                source.cell_height_mm,
                source.slot_probability,
                source.softness_min_mm,
                source.softness_max_mm,
                source.softness_size_cap_fraction,
                source.support_x_mm,
                source.support_y_mm,
                source.length_min_mm,
                source.length_bulk_max_mm,
                source.length_max_mm,
                source.length_tail_fraction,
                source.width_min_mm,
                source.width_bulk_max_mm,
                source.width_max_mm,
                source.width_tail_fraction,
                source.drift_fraction,
                source.first_knot_min,
                source.first_knot_max,
                source.second_knot_min,
                source.second_knot_max,
                source.interior_width_min_fraction,
                source.interior_width_max_fraction,
                source.interior_depth_min_fraction,
                source.interior_depth_max_fraction,
                source.endpoint_abrupt_probability,
                source.interruption_probability,
                source.gap_center_min,
                source.gap_center_max,
                source.gap_span_min,
                source.gap_span_max,
                source.scuff_probability,
                source.scuff_length_max_mm,
                source.scuff_angle_max_radians,
                source.strength_min,
                source.strength_max};
            if (!std::all_of(coefficients.begin(), coefficients.end(), [](float coefficient) {
                    return std::isfinite(coefficient);
                }))
                return false;
            out.cellWidthMm = source.cell_width_mm;
            out.cellHeightMm = source.cell_height_mm;
            out.slotProbability = source.slot_probability;
            out.softnessMinMm = source.softness_min_mm;
            out.softnessMaxMm = source.softness_max_mm;
            out.softnessSizeCapFraction = source.softness_size_cap_fraction;
            out.supportXMm = source.support_x_mm;
            out.supportYMm = source.support_y_mm;
            out.lengthMinMm = source.length_min_mm;
            out.lengthBulkMaxMm = source.length_bulk_max_mm;
            out.lengthMaxMm = source.length_max_mm;
            out.lengthTailFraction = source.length_tail_fraction;
            out.widthMinMm = source.width_min_mm;
            out.widthBulkMaxMm = source.width_bulk_max_mm;
            out.widthMaxMm = source.width_max_mm;
            out.widthTailFraction = source.width_tail_fraction;
            out.driftFraction = source.drift_fraction;
            out.firstKnotMin = source.first_knot_min;
            out.firstKnotMax = source.first_knot_max;
            out.secondKnotMin = source.second_knot_min;
            out.secondKnotMax = source.second_knot_max;
            out.interiorWidthMinFraction = source.interior_width_min_fraction;
            out.interiorWidthMaxFraction = source.interior_width_max_fraction;
            out.interiorDepthMinFraction = source.interior_depth_min_fraction;
            out.interiorDepthMaxFraction = source.interior_depth_max_fraction;
            out.endpointAbruptProbability = source.endpoint_abrupt_probability;
            out.interruptionProbability = source.interruption_probability;
            out.gapCenterMin = source.gap_center_min;
            out.gapCenterMax = source.gap_center_max;
            out.gapSpanMin = source.gap_span_min;
            out.gapSpanMax = source.gap_span_max;
            out.scuffProbability = source.scuff_probability;
            out.scuffLengthMaxMm = source.scuff_length_max_mm;
            out.scuffAngleMaxRadians = source.scuff_angle_max_radians;
            out.strengthMin = source.strength_min;
            out.strengthMax = source.strength_max;
            return true;
        }
        bool valid_grain_projection(const FjVisualGrainRecipe& value, const VisualGrainControls& controls) {
            if (value.active > 1 || value.sublayers_active > 1 || value.active != (controls.active ? 1u : 0u) ||
                value.sublayers_active != (controls.active && controls.sublayersActive ? 1u : 0u))
                return false;
            const auto zero = [](float v) {
                return std::bit_cast<std::uint32_t>(v) == 0;
            };
            const auto coefficients = std::array{
                std::span<const float>{&value.particle_area_um2, 1},
                std::span<const float>{value.particle_scale_cmy},
                std::span<const float>{value.particle_scale_layers},
                std::span<const float>{value.particle_density_min_cmy},
                std::span<const float>{value.uniformity_cmy},
                std::span<const float>{&value.correlation_sigma_px, 1},
                std::span<const float>{&value.dye_cloud_blur_um, 1},
                std::span<const float>{value.micro_structure},
                std::span<const float>{&value.amplitude, 1},
                std::span<const float>{&value.chroma_mix, 1},
                std::span<const float>{&value.chroma_shared_weight, 1},
                std::span<const float>{&value.chroma_independent_weight, 1},
                std::span<const float>{&value.coarse_weight, 1},
                std::span<const float>{&value.mid_weight, 1},
                std::span<const float>{&value.clump_temporal_mix, 1},
                std::span<const float>{&value.clump_morph_period_sec, 1},
                std::span<const float>{&value.fine_weight, 1}};
            // Check wire alternatives/representation; Rust owns derivation and identity.
            for (const auto lane : coefficients) {
                if (!std::all_of(lane.begin(), lane.end(), [&](float v) {
                        return value.active ? std::isfinite(v) : zero(v);
                    }))
                    return false;
            }
            for (std::size_t channel = 0; channel < 3; ++channel) {
                if (value.axis_finite_cmy[channel] > 1 || (!value.sublayers_active && value.axis_finite_cmy[channel] != 0))
                    return false;
                for (float v : value.axis_block_prefix_max_cmy[channel]) {
                    if (!std::isfinite(v) || (!value.sublayers_active && !zero(v)))
                        return false;
                }
            }
            if (!value.active)
                return value.sublayer_count == 0 && value.debug_view == 0 && value.layers_hash == 0 && value.hash == 0 && std::bit_cast<std::uint32_t>(value.size_mix_scale) == std::bit_cast<std::uint32_t>(1.0f);
            return value.hash != 0 && value.sublayer_count > 0 && value.debug_view >= 0 && value.debug_view <= 6 &&
                   value.particle_area_um2 > 0.0f && std::isfinite(value.size_mix_scale) && value.size_mix_scale >= 1.0f &&
                   (value.sublayers_active ? value.layers_hash != 0 : value.layers_hash == 0);
        }
        void dir_cleanup(FjStatus status) noexcept {
            if (status.category != FJ_STATUS_SUCCESS) {
                try {
                    JTRACE("ASSETS", "unexpected Rust asset owner release failure category=" + std::to_string(status.category));
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
            }
        }
        struct DirDeleter {
            void operator()(FjFilmDirOwner* owner) const noexcept {
                dir_cleanup(fj_legacy_film_dir_release(owner, nullptr));
            }
        };
    } // namespace

    bool prepare_visual_grain(const DevelopmentHold& development, const VisualGrainControls& controls, VisualGrainRecipe& out, std::string& diagnostic) {
        out = VisualGrainRecipe{};
        diagnostic.clear();
        const FjVisualGrainParams params{
            controls.active ? 1u : 0u,
            controls.sublayersActive ? 1u : 0u,
            controls.particleAreaUm2,
            {controls.particleScaleCmy[0], controls.particleScaleCmy[1], controls.particleScaleCmy[2]},
            {controls.particleScaleLayers[0], controls.particleScaleLayers[1], controls.particleScaleLayers[2]},
            {controls.visualParticleDensityMinCmy[0], controls.visualParticleDensityMinCmy[1], controls.visualParticleDensityMinCmy[2]},
            {controls.uniformityCmy[0], controls.uniformityCmy[1], controls.uniformityCmy[2]},
            controls.correlationSigmaPx,
            controls.dyeCloudBlurUm,
            {controls.microStructure[0], controls.microStructure[1]},
            controls.nSubLayers,
            controls.amplitude,
            controls.chromaMix,
            controls.chromaSharedWeight,
            controls.chromaIndependentWeight,
            controls.coarseWeight,
            controls.midWeight,
            controls.sizeMixScale,
            controls.clumpTemporalMix,
            controls.clumpMorphPeriodSec,
            controls.debugView};
        FjVisualGrainRecipe prepared{};
        FjFilmRecipeFailure failure{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        auto status = fj_legacy_visual_grain_prepare(development.get(), &params, &prepared, &failure, &error);
#if defined(JUICER_FILM_RECIPE_TEST_HOOK)
        RecipeTest::after_call(static_cast<unsigned>(Operation::Grain), status, failure);
        RecipeTest::grain(prepared);
#endif
        if ((!controls.active && status.category == FJ_STATUS_PREPARATION_FAILURE) ||
            (!controls.sublayersActive && failure.reason == FJ_FILM_RECIPE_GRAIN_LAYERS))
            integrity("Visual grain preparation");
        if (!admit(Operation::Grain, status, failure, error, nullptr, Spektrafilm::kDefaultScanRoute, 0, diagnostic))
            return false;
        if (!valid_grain_projection(prepared, controls))
            integrity("Visual grain preparation");
        VisualGrainRecipe result;
        result.active = prepared.active != 0;
        result.sublayersActive = prepared.sublayers_active != 0;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            result.grainLayerAxisFinite[channel] = prepared.axis_finite_cmy[channel] != 0;
            std::copy_n(prepared.axis_block_prefix_max_cmy[channel], 16, result.grainLayerAxisBlockPrefixMax[channel].begin());
        }
        result.particleAreaUm2 = prepared.particle_area_um2;
        std::copy_n(prepared.particle_scale_cmy, 3, result.particleScaleCmy.begin());
        std::copy_n(prepared.particle_scale_layers, 3, result.particleScaleLayers.begin());
        std::copy_n(prepared.particle_density_min_cmy, 3, result.visualParticleDensityMinCmy.begin());
        std::copy_n(prepared.uniformity_cmy, 3, result.uniformityCmy.begin());
        result.correlationSigmaPx = prepared.correlation_sigma_px;
        result.dyeCloudBlurUm = prepared.dye_cloud_blur_um;
        std::copy_n(prepared.micro_structure, 2, result.microStructure.begin());
        result.nSubLayers = prepared.sublayer_count;
        result.amplitude = prepared.amplitude;
        result.chromaMix = prepared.chroma_mix;
        result.chromaSharedWeight = prepared.chroma_shared_weight;
        result.chromaIndependentWeight = prepared.chroma_independent_weight;
        result.coarseWeight = prepared.coarse_weight;
        result.midWeight = prepared.mid_weight;
        result.sizeMixScale = prepared.size_mix_scale;
        result.clumpTemporalMix = prepared.clump_temporal_mix;
        result.clumpMorphPeriodSec = prepared.clump_morph_period_sec;
        result.debugView = prepared.debug_view;
        result.fineWeight = prepared.fine_weight;
        result.densityCurvesLayersHash = prepared.layers_hash;
        result.hash = prepared.hash;
        out = result;
        return true;
    }

    bool prepare_physical_effects(const PhysicalEffectsControls& controls, const VisualGrainRecipe& grain, FilmJuicerEffectsRecipe& out, std::string& diagnostic) {
        out = FilmJuicerEffectsRecipe{};
        diagnostic.clear();
        const FjPhysicalEffectsParams params{controls.filmDustAmount, controls.filmScratchAmount, controls.gateDustAmount, controls.gateScratchAmount, controls.gateWeaveAmount, grain.active ? 1u : 0u, grain.debugView};
        FjPhysicalEffectsRecipe prepared{};
        FjFilmRecipeFailure failure{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        auto status = fj_legacy_physical_effects_prepare(&params, &prepared, &failure, &error);
#if defined(JUICER_FILM_RECIPE_TEST_HOOK)
        RecipeTest::after_call(static_cast<unsigned>(Operation::Effects), status, failure);
        RecipeTest::effects(prepared);
#endif
        const bool envelope = status.api == FJ_API_NONE && status.native_code == 0 && failure.channel == 3 && failure.sample == UINT64_MAX &&
                              ((status.category == FJ_STATUS_SUCCESS && failure.reason == FJ_FILM_RECIPE_REASON_NONE) ||
                               (status.category == FJ_STATUS_UNSUPPORTED_INPUT && failure.reason == FJ_FILM_RECIPE_RAW) ||
                               (status.category == FJ_STATUS_PREPARATION_FAILURE && failure.reason == FJ_FILM_RECIPE_EFFECTS_PROBABILITY) ||
                               (status.category == FJ_STATUS_INTERNAL_FAILURE && failure.reason == FJ_FILM_RECIPE_PANIC));
        if (!envelope || error.length >= error.capacity)
            integrity("Physical effects preparation");
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            diagnostic = "ResourceDescriptorMismatch phase=effects_recipe field=physical_policy";
            return false;
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure rejected;
            JuicerCuda::set_failure(rejected, status, "Physical effects preparation");
            throw JuicerCuda::ExecutionFailure{std::move(rejected)};
        }
        if (prepared.active > 1 || !std::isfinite(prepared.gate_weave_amount))
            integrity("Physical effects preparation");
        FilmJuicerEffectsRecipe result;
        if (!project_policy(prepared.film_dust, result.filmDust) || !project_policy(prepared.film_scratch, result.filmScratch) ||
            !project_policy(prepared.gate_dust, result.gateDust) || !project_policy(prepared.gate_scratch, result.gateScratch))
            integrity("Physical effects preparation");
        result.gateWeaveAmount = prepared.gate_weave_amount;
        result.active = prepared.active != 0;
        result.hash = prepared.hash;
        out = result;
        return true;
    }

    bool prepare_film_density_minimum(const GrainContract& requested, GrainContract& out, std::string& diagnostic) {
        const FjFilmDensityMinimum input{{requested.densityMinCmy[0], requested.densityMinCmy[1], requested.densityMinCmy[2]}};
        FjFilmDensityMinimum prepared{};
        FjFilmRecipeFailure failure{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        auto status = fj_legacy_film_density_minimum_prepare(&input, &prepared, &failure, &error);
#if defined(JUICER_FILM_RECIPE_TEST_HOOK)
        RecipeTest::after_call(static_cast<unsigned>(Operation::Minimum), status, failure);
#endif
        if (!admit(Operation::Minimum, status, failure, error, nullptr, Spektrafilm::kDefaultScanRoute, 0, diagnostic))
            return false;
        std::copy_n(prepared.cmy, 3, out.densityMinCmy.begin());
        return true;
    }

    bool prepare_film_dir(const DevelopmentHold& development, const DirCouplersControls& controls, const Profiles::FilmProfile& profile, Spektrafilm::ScanRoute route, std::size_t samples, DirCouplersRecipe& out, std::string& diagnostic) {
        const FjFilmDirParams params{controls.active ? 1u : 0u, controls.amount, controls.inhibitionSameLayer, controls.inhibitionInterlayer, controls.diffusionSizeUm, controls.diffusionTailUm, controls.diffusionTailWeight, controls.gammaUseStock ? 1u : 0u, {controls.gammaSameLayerRgb[0], controls.gammaSameLayerRgb[1], controls.gammaSameLayerRgb[2]}, {controls.gammaInterlayerRToGb[0], controls.gammaInterlayerRToGb[1]}, {controls.gammaInterlayerGToRb[0], controls.gammaInterlayerGToRb[1]}, {controls.gammaInterlayerBToRg[0], controls.gammaInterlayerBToRg[1]}, {controls.langmuirDonorKRgb[0], controls.langmuirDonorKRgb[1], controls.langmuirDonorKRgb[2]}, {controls.langmuirReceiverKRgb[0], controls.langmuirReceiverKRgb[1], controls.langmuirReceiverKRgb[2]}};
        FjFilmDirOwner* acquired = nullptr;
        FjFilmRecipeFailure failure{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        auto status = fj_legacy_film_dir_acquire(development.get(), &params, &acquired, &failure, &error);
        std::unique_ptr<FjFilmDirOwner, DirDeleter> owner(acquired);
#if defined(JUICER_FILM_RECIPE_TEST_HOOK)
        RecipeTest::after_call(static_cast<unsigned>(Operation::Acquire), status, failure);
#endif
        if (!admit(Operation::Acquire, status, failure, error, &profile, route, samples, diagnostic))
            return false;
        if (!owner)
            integrity("Film DIR preparation");
        FjFilmDirView view{};
        status = fj_legacy_film_dir_view(owner.get(), &view, &failure, &error);
#if defined(JUICER_FILM_RECIPE_TEST_HOOK)
        RecipeTest::after_call(static_cast<unsigned>(Operation::View), status, failure);
        RecipeTest::view(view);
#endif
        if (!admit(Operation::View, status, failure, error, &profile, route, samples, diagnostic))
            return false;
        const bool active = view.mode != FJ_FILM_DIR_INACTIVE;
        const auto expectedPolarity = static_cast<std::uint32_t>(profile.info.type);
        if (view.mode > 2 || view.polarity > 2 || (view.polarity != expectedPolarity && (active || view.polarity != 2)) ||
            (active && (view.mode != (expectedPolarity == 0 ? 1u : 2u) || view.hash == 0 || view.compensated_axes_hash == 0)) ||
            (!active && (view.hash != 0 || view.compensated_axes_hash != 0)))
            integrity("Film DIR preparation");
        // These are wire-alternative lanes; active chemistry is already completed in Rust.
        const auto zero = [](float value) {
            return std::bit_cast<std::uint32_t>(value) == 0;
        };
        const auto zero_triplet = [&](const float (&values)[3]) {
            return std::all_of(std::begin(values), std::end(values), zero);
        };
        if ((!active && (!std::all_of(std::begin(view.matrix_rgb), std::end(view.matrix_rgb), zero) ||
                         !zero(view.diffusion_size_um) || !zero(view.diffusion_tail_um) || !zero(view.diffusion_tail_weight) ||
                         !zero_triplet(view.density_max_rgb) || !zero_triplet(view.density_ref_rgb) || !zero_triplet(view.donor_k_rgb) ||
                         !zero_triplet(view.receiver_c_ref_rgb) || !zero_triplet(view.receiver_kr_rgb))) ||
            (view.mode == FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR && (!zero_triplet(view.receiver_c_ref_rgb) || !zero_triplet(view.receiver_kr_rgb))) ||
            (view.mode == FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR && !zero_triplet(view.donor_k_rgb)))
            integrity("Film DIR preparation");
        DirCouplersRecipe result;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const auto span = view.compensated_axes_rgb[channel];
            const auto address = reinterpret_cast<std::uintptr_t>(span.data);
            if (span.count != (active ? samples : 0) || (!active && span.data) ||
                (active && (!span.data || address % alignof(float) != 0 || span.count > result.compensatedDensityCurveAxesRgb[channel].max_size() ||
                            span.count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float) || span.count * sizeof(float) > std::numeric_limits<std::uintptr_t>::max() - address)))
                integrity("Film DIR preparation");
        }
        result.polarity = static_cast<Spektrafilm::ProfilePolarity>(view.polarity);
        result.active = active;
        result.nonlinearMode = static_cast<DirNonlinearMode>(view.mode);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            std::copy_n(view.matrix_rgb + 3 * channel, 3, result.matrixRgb[channel].begin());
            if (active)
                result.compensatedDensityCurveAxesRgb[channel].assign(view.compensated_axes_rgb[channel].data, view.compensated_axes_rgb[channel].data + samples);
        }
        result.diffusionSizeUm = view.diffusion_size_um;
        result.diffusionTailUm = view.diffusion_tail_um;
        result.diffusionTailWeight = view.diffusion_tail_weight;
        std::copy_n(view.density_max_rgb, 3, result.densityMaxRgb.begin());
        std::copy_n(view.density_ref_rgb, 3, result.densityRefRgb.begin());
        std::copy_n(view.donor_k_rgb, 3, result.donorKRgb.begin());
        std::copy_n(view.receiver_c_ref_rgb, 3, result.receiverCRefRgb.begin());
        std::copy_n(view.receiver_kr_rgb, 3, result.receiverKrRgb.begin());
        result.compensatedDensityCurveAxesHash = view.compensated_axes_hash;
        result.hash = view.hash;
        // All borrowed spans have been copied; asynchronous consumers own result/staging.
        const auto released = fj_legacy_film_dir_release(owner.release(), &error);
        if (released.category != FJ_STATUS_SUCCESS || released.api != FJ_API_NONE || released.native_code != 0) {
            JuicerCuda::Failure releaseFailure;
            JuicerCuda::set_failure(releaseFailure, released, "Film DIR preparation");
            throw JuicerCuda::ExecutionFailure{std::move(releaseFailure)};
        }
        out = std::move(result);
        return true;
    }

    bool prepare_film_density_bounds(const DevelopmentHold& development, const GrainContract& requested, const Profiles::FilmProfile& profile, Spektrafilm::ScanRoute route, DensityBoundsRecipe& out, std::string& diagnostic) {
        const FjFilmDensityMinimum input{{requested.densityMinCmy[0], requested.densityMinCmy[1], requested.densityMinCmy[2]}};
        FjFilmDensityBounds bounds{};
        FjFilmRecipeFailure failure{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        auto status = fj_legacy_film_density_bounds_prepare(development.get(), &input, &bounds, &failure, &error);
#if defined(JUICER_FILM_RECIPE_TEST_HOOK)
        RecipeTest::after_call(static_cast<unsigned>(Operation::Bounds), status, failure);
#endif
        if (!admit(Operation::Bounds, status, failure, error, &profile, route, 0, diagnostic))
            return false;
        const bool print = Spektrafilm::scan_route_is_print(route);
        if (bounds.route != static_cast<std::uint32_t>(route) || bounds.medium != 0 || bounds.polarity != static_cast<std::uint32_t>(profile.info.type) || bounds.source != (print ? 1u : 0u) || bounds.hash == 0)
            integrity("Film density bounds preparation");
        DensityBoundsRecipe result;
        result.route = route;
        result.medium = Spektrafilm::DensityMedium::Film;
        result.polarity = profile.info.type;
        result.source = static_cast<Spektrafilm::DensityBoundsSource>(bounds.source);
        std::copy_n(bounds.minimum_cmy, 3, result.dataMinCmy.begin());
        std::copy_n(bounds.maximum_cmy, 3, result.dataMaxCmy.begin());
        std::copy_n(bounds.inv_span_cmy, 3, result.invSpanCmy.begin());
        std::copy_n(bounds.authored_minimum_cmy, 3, result.authoredMinCmy.begin());
        std::copy_n(bounds.authored_maximum_cmy, 3, result.authoredMaxCmy.begin());
        result.hash = bounds.hash;
        out = result;
        return true;
    }
    namespace {
        FjDiffusionParams bind_optics_diffusion(const Spektrafilm::DiffusionFilterAuthoredControls& v) {
            return {
                .active = v.active ? 1u : 0u,
                .family = static_cast<std::uint32_t>(v.family),
                .strength = v.strength,
                .spatial_scale = v.spatialScale,
                .halo_warmth = v.haloWarmth,
                .core_intensity = v.coreIntensity,
                .core_size = v.coreSize,
                .halo_intensity = v.haloIntensity,
                .halo_size = v.haloSize,
                .bloom_intensity = v.bloomIntensity,
                .bloom_size = v.bloomSize};
        }
        bool optics_failure_valid(FjStatus status, const FjSpatialOpticsFailure& failure) {
            if (status.api != FJ_API_NONE || status.native_code != 0)
                return false;
            const bool empty = failure.component == FJ_OPTICS_COMPONENT_NONE && failure.field == FJ_OPTICS_FIELD_NONE && failure.channel == 3;
            if (failure.reason == FJ_OPTICS_REASON_NONE)
                return status.category == FJ_STATUS_SUCCESS && empty;
            if (failure.reason == FJ_OPTICS_REASON_RAW)
                return status.category == FJ_STATUS_UNSUPPORTED_INPUT && empty;
            if (failure.reason == FJ_OPTICS_REASON_PANIC)
                return status.category == FJ_STATUS_INTERNAL_FAILURE && empty;
            if (status.category != FJ_STATUS_PREPARATION_FAILURE)
                return false;
            if (failure.reason == FJ_OPTICS_REASON_DIFFUSION) {
                return (failure.component == FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION || failure.component == FJ_OPTICS_COMPONENT_ENLARGER_DIFFUSION) && failure.field >= FJ_OPTICS_FIELD_FAMILY && failure.field <= FJ_OPTICS_FIELD_RESOLVED_HASH && failure.channel == 3;
            }
            if (failure.reason == FJ_OPTICS_REASON_LENS_BLUR) {
                return failure.component == FJ_OPTICS_COMPONENT_CAMERA_LENS_BLUR && failure.field == FJ_OPTICS_FIELD_SIGMA_UM && failure.channel == 3;
            }
            return failure.reason == FJ_OPTICS_REASON_HALATION && failure.component == FJ_OPTICS_COMPONENT_SCATTER_HALATION && failure.field == FJ_OPTICS_FIELD_TOTAL_STRENGTH && failure.channel < 3;
        }
        const char* optics_field_name(std::uint32_t field) {
            constexpr std::array<const char*, 17> fields{"none", "family", "strength", "spatial_scale", "halo_warmth", "core_intensity", "core_size", "halo_intensity", "halo_size", "bloom_intensity", "bloom_size", "scatter_fraction", "group_weights", "group_centers_um", "resolved_hash", "sigma_um", "total_strength"};
            return fields[field];
        }
        bool project_optics_diffusion(const FjDiffusionRecipe& wire, DiffusionFilterOpticsRecipe& out) {
            if (wire.active > 1)
                return false;
            switch (wire.domain) {
                case 0:
                    out.domain = Spektrafilm::SpatialOpticsDomain::FilmLinearExposure;
                    break;
                case 1:
                    out.domain = Spektrafilm::SpatialOpticsDomain::PrintLinearExposure;
                    break;
                default:
                    return false;
            }
            switch (wire.family) {
                case 0:
                    out.resolved.family = Spektrafilm::DiffusionFilterFamily::Glimmerglass;
                    break;
                case 1:
                    out.resolved.family = Spektrafilm::DiffusionFilterFamily::BlackProMist;
                    break;
                case 2:
                    out.resolved.family = Spektrafilm::DiffusionFilterFamily::ProMist;
                    break;
                case 3:
                    out.resolved.family = Spektrafilm::DiffusionFilterFamily::Cinebloom;
                    break;
                default:
                    return false;
            }
            out.resolved.active = wire.active != 0;
            out.resolved.scatterFraction = wire.scatter_fraction;
            std::copy_n(wire.group_weights, 3, out.resolved.groupWeightsCoreHaloBloom.begin());
            std::copy_n(wire.group_lambda_um, 3, out.resolved.groupCenterLambdaUm.begin());
            out.resolved.effectiveWarmth = wire.warmth;
            out.resolved.spatialScale = wire.spatial_scale;
            out.resolved.hash = wire.resolved_hash;
            out.hash = wire.hash;
            return true;
        }
    } // namespace

    // SF_TEMP_BRIDGE_SPATIAL_OPTICS: fixed named-field projection; remove S4.E.
    bool prepare_spatial_optics(const Spektrafilm::SpatialOpticsControls& controls,
                                const Profiles::ProfileDigest& digest,
                                bool printDomain,
                                SpatialOptics& out,
                                std::string& diagnostic) {
        out = SpatialOptics{};
        diagnostic.clear();
        FjSpatialOpticsParams params{};
        params.camera_diffusion = bind_optics_diffusion(controls.cameraDiffusion);
        params.camera_lens_blur_um = controls.cameraLensBlurUm;
        const auto& h = controls.scatterHalation;
        params.scatter_halation.active = h.active ? 1u : 0u;
        params.scatter_halation.scatter_amount = h.scatterAmount;
        params.scatter_halation.scatter_spatial_scale = h.scatterSpatialScale;
        params.scatter_halation.halation_amount = h.halationAmount;
        params.scatter_halation.halation_spatial_scale = h.halationSpatialScale;
        std::copy_n(digest.halationFirstSigmaUm.begin(), 3, params.scatter_halation.first_sigma_um);
        std::copy_n(digest.halationPrimaryAmount.begin(), 3, params.scatter_halation.primary_amount);
        params.enlarger_diffusion = bind_optics_diffusion(controls.enlargerDiffusion);
        params.print_domain = printDomain ? 1u : 0u;
        FjSpatialOpticsRecipe wire{};
        FjSpatialOpticsFailure failure{};
        std::array<char, 512> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjStatus status = fj_legacy_spatial_optics_prepare(&params, &wire, &failure, &error);
#if defined(JUICER_FILM_RECIPE_TEST_HOOK)
        RecipeTest::optics(status, failure, wire, error);
#endif
        if (error.data != text.data() || error.capacity != text.size() || error.length >= error.capacity || text[error.length] != '\0' || !optics_failure_valid(status, failure))
            integrity("Spatial optics preparation");
        if (status.category != FJ_STATUS_SUCCESS) {
            if (status.category != FJ_STATUS_PREPARATION_FAILURE) {
                JuicerCuda::Failure rejected;
                JuicerCuda::set_failure(rejected, status, "Spatial optics preparation");
                throw JuicerCuda::ExecutionFailure{std::move(rejected)};
            }
            if (failure.reason == FJ_OPTICS_REASON_HALATION) {
                diagnostic = "ResourceDescriptorMismatch component=scatter_halation field=total_strength channel=" + std::to_string(failure.channel);
            } else {
                const char* component = failure.component == FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION     ? "camera_diffusion"
                                        : failure.component == FJ_OPTICS_COMPONENT_ENLARGER_DIFFUSION ? "enlarger_diffusion"
                                                                                                      : "camera_lens_blur";
                diagnostic = std::string("ResourceDescriptorMismatch phase=6A component=") + component + " field=" + optics_field_name(failure.field);
            }
            return false;
        }
        SpatialOptics projected;
        if (!project_optics_diffusion(wire.camera_diffusion, projected.cameraDiffusion) || !project_optics_diffusion(wire.enlarger_diffusion, projected.enlargerDiffusion) || wire.scatter_halation.scatter_active > 1 || wire.scatter_halation.back_reflection_active > 1)
            integrity("Spatial optics preparation");
        projected.cameraLensBlur.sigmaUm = wire.camera_lens_blur.sigma_um;
        projected.cameraLensBlur.hash = wire.camera_lens_blur.hash;
        auto& halation = projected.scatterHalation;
        halation.scatterActive = wire.scatter_halation.scatter_active != 0;
        halation.backReflectionActive = wire.scatter_halation.back_reflection_active != 0;
        halation.scatterAmount = wire.scatter_halation.scatter_amount;
        halation.scatterSpatialScale = wire.scatter_halation.scatter_spatial_scale;
        halation.halationSpatialScale = wire.scatter_halation.halation_spatial_scale;
        std::copy_n(wire.scatter_halation.first_sigma_um, 3, halation.halationFirstSigmaUm.begin());
        std::copy_n(wire.scatter_halation.total_strength, 3, halation.totalStrength.begin());
        halation.hash = wire.scatter_halation.hash;
        projected.hash = wire.hash;
        out = projected;
        return true;
    }

} // namespace JuicerAssets
