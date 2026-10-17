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
    } // namespace RecipeTest
#endif
    namespace {
        enum class Operation : std::uint8_t {
            Minimum,
            Acquire,
            View,
            Bounds,
            Grain
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
} // namespace JuicerAssets
