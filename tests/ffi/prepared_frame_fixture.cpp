#include "prepared_frame_fixture.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

#include "FilmEffectsFrameDescriptors.h"
#include "ColorTransforms.h"
#include "GamutCompression.h"
#include "RustEffectsFrameBridge.h"
#include "Scanner.h"
#include "host_preparation_fixture.h"
#include "juicer_test_api.h"

namespace JuicerCudaTest::PreparedFrameFixture {
    namespace {

        std::span<const float> span(const FjFloatSpan& source) {
            return {source.data, source.count};
        }

        std::span<const std::uint8_t> bytes(const FjByteSpan& source) {
            return {source.data, source.count};
        }

        JuicerCuda::ThreeChannelSamplesView channel_samples(const FjFloatSpan& source) {
            return JuicerCuda::ThreeChannelSamplesView(std::span<const std::byte>{
                reinterpret_cast<const std::byte*>(source.data), source.count * sizeof(float)});
        }

        FjGrainCoefficients grain_coefficients(const FjVisualGrainRecipe& recipe) {
            FjGrainCoefficients out{};
            if (!recipe.active)
                return out;
            out.flags = FJ_GRAIN_ACTIVE | (recipe.sublayers_active ? FJ_GRAIN_SUBLAYERS : 0u);
            out.debug_view = static_cast<std::uint32_t>(recipe.debug_view);
            out.sublayer_count = static_cast<std::size_t>(recipe.sublayer_count);
            std::copy_n(recipe.micro_structure, 2, out.micro_structure);
            out.clump_temporal_mix = recipe.clump_temporal_mix;
            out.size_mix_scale = recipe.size_mix_scale;
            out.amplitude = recipe.amplitude;
            out.chroma_mix = recipe.chroma_mix;
            out.chroma_shared_weight = recipe.chroma_shared_weight;
            out.chroma_independent_weight = recipe.chroma_independent_weight;
            std::copy_n(recipe.particle_density_min_cmy, 3, out.particle_density_min_cmy);
            std::copy_n(recipe.uniformity_cmy, 3, out.uniformity_cmy);
            constexpr std::uint32_t axisFlags[] = {
                FJ_GRAIN_AXIS_FINITE_C, FJ_GRAIN_AXIS_FINITE_M, FJ_GRAIN_AXIS_FINITE_Y};
            for (std::size_t channel = 0; channel < 3; ++channel) {
                if (recipe.axis_finite_cmy[channel])
                    out.flags |= axisFlags[channel];
                std::copy_n(recipe.axis_block_prefix_max_cmy[channel], 16, out.layer_axis_block_prefix_max[channel]);
            }
            return out;
        }

        bool facade_status(FjStatus status, const std::array<char, 1024>& text, const FjErrorBuffer& error, std::string& diagnostic, const char* fallback) {
            if (status.category == FJ_STATUS_SUCCESS)
                return true;
            const auto length = std::min(error.length, text.size());
            diagnostic.assign(text.data(), length);
            if (diagnostic.empty())
                diagnostic = fallback;
            return false;
        }

        class NoiseHold final {
        public:
            NoiseHold() = default;
            NoiseHold(const NoiseHold&) = delete;
            NoiseHold& operator=(const NoiseHold&) = delete;
            ~NoiseHold() {
                if (_owner != nullptr &&
                    fj_legacy_noise_release(_owner, nullptr).category != FJ_STATUS_SUCCESS) {
                    std::abort();
                }
            }

            FjStatus acquire(const FjAssets* assets, FjErrorBuffer* error) {
                return fj_legacy_noise_acquire(assets, &_owner, error);
            }

            FjStatus view(FjStaticNoise& out, FjErrorBuffer* error) const {
                return fj_legacy_noise_view(_owner, &out, error);
            }

            FjStatus release(FjErrorBuffer* error) {
                if (_owner == nullptr)
                    return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                // Release consumes the handle even if diagnostic delivery fails.
                FjNoise* const owner = std::exchange(_owner, nullptr);
                return fj_legacy_noise_release(owner, error);
            }

        private:
            FjNoise* _owner = nullptr;
        };

        bool static_noise_input(const FjStaticNoise& source,
                                JuicerCuda::StaticNoiseInput& out,
                                std::string& diagnostic) {
            if (source.wang_tile_count >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                diagnostic = "Rust noise owner tile count exceeds native limits";
                return false;
            }
            out = {bytes(source.stbn), bytes(source.wang_tiles), bytes(source.wang_lut), source.stbn_width, source.stbn_height, source.stbn_frames, source.wang_width, source.wang_height, static_cast<int>(source.wang_tile_count), source.wang_colors};
            return true;
        }

        bool release_noise(NoiseHold& owner, std::string& diagnostic) {
            std::array<char, 1024> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            const FjStatus status = owner.release(&error);
            if (status.category == FJ_STATUS_SUCCESS)
                return true;
            const auto length = std::min(error.length, text.size());
            const std::string cleanup = length != 0
                                            ? std::string(text.data(), length)
                                            : std::string("Rust noise owner release failed");
            if (diagnostic.empty())
                diagnostic = cleanup;
            else
                std::fprintf(stderr, "noise cleanup failed category=%u api=%u native=%d diagnostic=%s\n", status.category, status.api, status.native_code, cleanup.c_str());
            return false;
        }

        const char* route_label(std::uint32_t route) {
            switch (route) {
                case FJ_ROUTE_NEGATIVE_DIRECT:
                case FJ_ROUTE_POSITIVE_DIRECT:
                    return "direct";
                case FJ_ROUTE_NEGATIVE_PRINT:
                case FJ_ROUTE_POSITIVE_PRINT:
                    return "print";
                default:
                    return nullptr;
            }
        }

        ScatterHalationGaussianDescriptor halation_gaussian(
            const FjHalationGaussian& source) {
            ScatterHalationGaussianDescriptor out{};
            out.kind = static_cast<ScatterHalationGaussianKind>(source.kind);
            out.radius = source.radius;
            std::copy_n(source.fir_weights, out.firWeights.size(), out.firWeights.begin());
            out.B = source.feedforward;
            out.B1 = source.feedback[0];
            out.B2 = source.feedback[1];
            out.B3 = source.feedback[2];
            return out;
        }

        ScatterHalationFrameDescriptor halation_descriptor(const FjScatterHalation& source) {
            ScatterHalationFrameDescriptor out{};
            out.recipeHash = source.recipe_hash;
            out.scatterAmount = source.scatter_amount;
            for (std::size_t channel = 0; channel < out.channels.size(); ++channel) {
                const auto& input = source.channels[channel];
                auto& target = out.channels[channel];
                target.totalStrength = input.total_strength;
                target.core = halation_gaussian(input.core);
                for (std::size_t index = 0; index < target.tail.size(); ++index) {
                    target.tail[index] = halation_gaussian(input.tail[index]);
                    target.bounce[index] = halation_gaussian(input.bounce[index]);
                }
            }
            return out;
        }

        Spektrafilm::DiffusionStageFrameDescriptor diffusion_stage(
            const FjDiffusion& source, Spektrafilm::DiffusionLinearStage stage) {
            Spektrafilm::DiffusionStageFrameDescriptor out{};
            out.stage = stage;
            out.scatterFraction = source.scatter_fraction;
            out.hash = source.hash;
            out.sample.family = static_cast<Spektrafilm::DiffusionFilterFamily>(source.family);
            std::copy_n(source.group_weights, 3, out.sample.groupWeightsCoreHaloBloom.begin());
            std::copy_n(source.group_lambda_um, 3, out.sample.groupCenterLambdaUm.begin());
            out.sample.effectiveWarmth = source.warmth;
            out.sample.spatialScale = source.spatial_scale;
            out.sample.pixelSizeUm = source.pixel_size_um;
            out.sample.radiusPixels = source.radius_px;
            out.sample.hash = source.sample_hash;
            return out;
        }

        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion_descriptor(
            const FjDiffusionFrameSet& source) {
            if (source.hash == 0)
                return std::nullopt;
            Spektrafilm::DiffusionFrameSetDescriptor out{};
            out.route = static_cast<Spektrafilm::ScanRoute>(source.route);
            out.fullFrame = {source.full_frame.x, source.full_frame.y, source.full_frame.width, source.full_frame.height};
            out.hash = source.hash;
            if (source.flags & 1u)
                out.camera = diffusion_stage(source.camera,
                                             Spektrafilm::DiffusionLinearStage::CameraFilmLinear);
            if (source.flags & 2u)
                out.enlarger = diffusion_stage(source.enlarger,
                                               Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear);
            return out;
        }

        void print_input(const FjPreparedStateView& view, JuicerCuda::PrintResourceInput& out) {
            const FjPrint& source = view.print;
            out.filmChannelDensityCmy = channel_samples(source.film_density_cmy);
            out.filmBaseDensity = span(source.film_base_density);
            out.printSensitivityCmy = channel_samples(source.sensitivity_cmy);
            out.printLogExposure = span(source.log_exposure);
            out.printDensityCurvesCmy = channel_samples(source.density_cmy);
            out.mainIlluminant = span(source.main_illuminant);
            out.preflashIlluminant = span(source.preflash_illuminant);
            std::copy_n(source.preflash_raw_cmy, 3, out.preflashRawCmy.begin());
            out.factorMidgray = view.print_balance.factor_midgray;
            out.factorMidgrayComp = view.print_balance.factor_midgray_comp;
            out.normalizer = source.normalizer;

            auto& descriptors = out.descriptors;
            descriptors.filmDensityTables.filmProfileAssetVersionToken =
                view.film_profile_asset_version;
            descriptors.filmDensityTables.spectralSampleCount = 81;
            descriptors.filmDensityTables.hash = source.film_density_hash;
            descriptors.profileTables.printProfileAssetVersionToken =
                view.print_profile_asset_version;
            descriptors.profileTables.densitySampleCount =
                static_cast<std::uint32_t>(source.log_exposure.count);
            descriptors.profileTables.spectralSampleCount = 81;
            descriptors.profileTables.hash = source.profile_tables_hash;
            descriptors.mainIlluminant.hash = source.main_illuminant_hash;
            descriptors.mainIlluminant.cmyCc = {
                source.main_cc_cmy[0], source.main_cc_cmy[1], source.main_cc_cmy[2]};
            descriptors.preflashIlluminant.hash = source.preflash_illuminant_hash;
            descriptors.preflashIlluminant.cmyCc = {
                source.preflash_cc_cmy[0], source.preflash_cc_cmy[1], source.preflash_cc_cmy[2]};
            descriptors.preflashRaw.hash = source.preflash_raw_hash;
            descriptors.preflashRaw.filmProfileAssetVersionToken =
                view.film_profile_asset_version;
            descriptors.preflashRaw.printProfileAssetVersionToken =
                view.print_profile_asset_version;
            descriptors.preflashRaw.filteredPreflashIlluminantHash =
                source.preflash_illuminant_hash;
            descriptors.balance.filmProfileAssetVersionToken =
                view.film_profile_asset_version;
            descriptors.balance.printProfileAssetVersionToken =
                view.print_profile_asset_version;
            descriptors.balance.filmRawRecipeHash = view.film_exposure.hash;
            descriptors.balance.filmDevelopRecipeHash = view.film_development.hash;
            descriptors.balance.filteredMainIlluminantHash = source.main_illuminant_hash;
            descriptors.balance.hash = source.balance_hash;
            descriptors.preflashActive = (source.flags & FJ_PRINT_PREFLASH) != 0;
            descriptors.hash = source.hash;
        }

        void fill_output_gamut(const FjOutputColor& source, OutputGamutRecipe& out) {
            out.enabled = (source.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0;
            out.outputColorSpace = static_cast<int>(source.color_space);
            out.hash = source.gamut_recipe_hash;
            out.lightnessKneeThreshold = source.lightness_knee[0];
            out.lightnessKneeLimit = source.lightness_knee[1];
            out.lightnessKneePower = source.lightness_knee[2];
            out.chromaKneeThreshold = source.chroma_knee[0];
            out.chromaKneeLimit = source.chroma_knee[1];
            out.chromaKneePower = source.chroma_knee[2];
            std::copy_n(source.native_rgb_to_d65_xyz, 9, out.transform.nativeRgbToD65Xyz.begin());
            std::copy_n(source.d65_xyz_to_native_rgb, 9, out.transform.d65XyzToNativeRgb.begin());
            out.transform.outputColorSpace =
                static_cast<OutputEncoding::ColorSpace>(source.color_space);
        }

        void fill_scanner_color(const FjScannerColor& source, Scanner::ColorRuntime& out) {
            std::copy_n(source.cat02, 9, out.cat02);
            std::copy_n(source.xyz_to_rgb, 9, out.xyzToRgb);
            std::copy_n(source.illuminant_xyz, 3, out.illuminantXYZ);
            out.preparedEncoding = source.encoding;
            out.outputGamutRecipeHash = (source.flags & FJ_COLOR_GAMUT_COMPRESSION)
                                            ? source.active_gamut_recipe_hash
                                            : 0;
            out.hash = source.hash;
        }

        void fill_focused(const FjPreparedStateView& view,
                          JuicerCuda::FocusedRouteResourceInput& out) {
            const auto& exposure = view.film_exposure;
            const auto& development = view.film_development;
            const auto& dir = view.dir_recipe;
            const auto& scanner = view.scanner_tables;
            const auto& bounds = view.scanner_bounds;
            out.route = static_cast<Spektrafilm::ScanRoute>(view.route);
            out.filmRaw = {static_cast<Spektrafilm::RgbToRawMethod>(exposure.method),
                           channel_samples(exposure.sensitivity_rgb),
                           exposure.sensitivity_hash,
                           exposure.tc_lut_hash};
            out.filmDevelop.logExposure = span(development.log_exposure);
            out.filmDevelop.normalizedDensityCurvesRgb = channel_samples(development.density_rgb);
            out.filmDevelop.normalizedDensityCurvesHash = development.density_curves_hash;
            out.filmDevelop.densityCurvesLayersHash = development.density_layers_hash;
            out.wantDensityLayers = (view.grain.sublayers_active != 0);
            out.filmDevelop.densityCurvesLayersRequired = out.wantDensityLayers;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                out.dirCouplers.compensatedDensityCurveAxesRgb[channel] =
                    span(dir.compensated_axes_rgb[channel]);
                for (std::size_t layer = 0; layer < 3; ++layer)
                    out.filmDevelop.densityCurvesLayers[layer][channel] =
                        span(development.density_layers[layer][channel]);
                out.densityBounds.dataMinCmy[channel] = bounds.minimum_cmy[channel];
                out.densityBounds.invSpanCmy[channel] = bounds.inv_span_cmy[channel];
            }
            out.dirCouplers.active = dir.mode != FJ_DIR_INACTIVE;
            out.dirCouplers.compensatedDensityCurveAxesHash = dir.compensated_axes_hash;
            out.dirCouplers.hash = dir.hash;
            out.densityBounds.hash = bounds.hash;
            out.scannerLutDescriptor.route = static_cast<Spektrafilm::ScanRoute>(view.route);
            out.scannerLutDescriptor.medium = Spektrafilm::scan_route_is_print(out.route)
                                                  ? Scanner::ScannedMediumKind::Print
                                                  : Scanner::ScannedMediumKind::Film;
            out.scannerLutDescriptor.polarity = static_cast<Spektrafilm::ProfilePolarity>(view.polarity);
            out.scannerLutDescriptor.densityBoundsHash = view.scanner_lut.density_bounds_hash;
            out.scannerLutDescriptor.lutResolution = view.scanner_lut.resolution;
            out.scannerLutDescriptor.hash = view.scanner_lut.hash;
            out.scannerLutDescriptor.channelDensityHash = view.scanner_lut.channel_density_hash;
            out.scannerLutDescriptor.baseDensityHash = view.scanner_lut.base_density_hash;
            out.scannerLutDescriptor.scanIlluminantHash = view.scanner_lut.viewing_illuminant_hash;
            out.scannerLutDescriptor.observerHash = view.scanner_lut.observer_hash;
            out.scannerTables = {span(scanner.dyes_cmy[2]), span(scanner.dyes_cmy[1]), span(scanner.dyes_cmy[0]), span(scanner.weighted_xyz[0]), span(scanner.weighted_xyz[1]), span(scanner.weighted_xyz[2]), span(scanner.baseline_min), 81, scanner.inv_yn, scanner.has_baseline != 0};
            out.exposureSampleCount = 81;
            out.exposureIlluminant = span(exposure.mallett_illuminant);
            out.filmTcLut = span(exposure.tc_lut_rgba);
            out.mallettBasis = span(exposure.mallett_basis);
            out.mallettAvailable = exposure.method == FJ_RAW_MALLETT_2019;
            out.mallettRows = out.mallettAvailable ? 81 : 0;
            out.mallettColumns = out.mallettAvailable ? 3 : 0;
            out.outputGamut.cmax = span(view.output_color.gamut_cmax);
            out.outputGamut.tableHash = view.output_color.gamut_table_hash;
            out.outputGamut.recipeHash = view.output_color.gamut_recipe_hash;
            out.outputGamut.enabled = (view.output_color.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0;
        }

        void fill_film_payload(const FjPreparedStateView& view,
                               JuicerCuda::FilmPayloadInput& out) {
            const auto& exposure = view.film_exposure;
            const auto& development = view.film_development;
            const auto& dir = view.dir_recipe;
            out.inputColorSpace = static_cast<int>(exposure.input_color_space);
            out.inputCctfDecoding = (exposure.flags & FJ_FILM_DECODE_CCTF) != 0;
            out.method = static_cast<Spektrafilm::RgbToRawMethod>(exposure.method);
            out.manualExposureEv = exposure.manual_exposure_ev;
            out.mallettGreenMidgrayScale = exposure.mallett_green_midgray_scale;
            out.sensitivityHash = exposure.sensitivity_hash;
            out.densityCurvesHash = development.density_curves_hash;
            out.densitySampleCount = development.log_exposure.count;
            std::copy_n(development.gamma_rgb, 3, out.gammaRgb.begin());
            out.dirMode = static_cast<DirNonlinearMode>(dir.mode);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                std::copy_n(dir.matrix_rgb + channel * 3, 3, out.dirMatrixRgb[channel].begin());
                out.densityMaxRgb[channel] = dir.density_max_rgb[channel];
                out.densityRefRgb[channel] = dir.density_ref_rgb[channel];
                out.donorKRgb[channel] = dir.donor_k_rgb[channel];
                out.receiverCRefRgb[channel] = dir.receiver_c_ref_rgb[channel];
                out.receiverKrRgb[channel] = dir.receiver_kr_rgb[channel];
            }
            out.dirAxesHash = dir.compensated_axes_hash;
            out.dirHash = dir.hash;
            out.densityBoundsHash = Spektrafilm::scan_route_is_print(
                                        static_cast<Spektrafilm::ScanRoute>(view.route))
                                        ? view.enlarger_film_bounds.hash
                                        : view.scanner_bounds.hash;
        }

    } // namespace

    bool with_prepared_frame_inputs(const PreparedState& state,
                                    const FrameFacts& facts,
                                    void* user,
                                    Consumer consume,
                                    std::string& diagnostic) {
        diagnostic.clear();
        if (!consume) {
            diagnostic = "missing prepared frame consumer";
            return false;
        }

        FjPreparedStateView view{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        if (!facade_status(state.view(view, error), text, error, diagnostic, "prepared state view failed"))
            return false;

        const auto& dir = view.dir_recipe;
        const char* const label = route_label(view.route);
        if (label == nullptr) {
            diagnostic = "prepared state has unsupported scan route";
            return false;
        }
        const FjDirFrameInput dirInput{
            dir.mode != FJ_DIR_INACTIVE ? 1u : 0u,
            dir.diffusion_size_um,
            dir.diffusion_tail_um,
            dir.diffusion_tail_weight,
            facts.pixelSizeUm,
            facts.renderExtent,
            facts.fullFrameExtent,
            dir.hash};
        FjDirFrameMath dirMath{};
        FjDirFrameFailure dirFailure{};
        error.length = 0;
        if (!facade_status(fj_test_dir_frame_prepare(&dirInput, &dirMath, &dirFailure, &error),
                           text,
                           error,
                           diagnostic,
                           "DIR frame facade failed"))
            return false;
        Spektrafilm::SpatialDirDescriptor spatialDir{};
        if (!Spektrafilm::compose_spatial_dir_descriptor(
                dirMath, facts.pixelSizeUm, label, spatialDir)) {
            diagnostic = "native spatial DIR descriptor composition failed";
            return false;
        }

        FjGrainFrameInput grainInput{};
        const auto& grain = view.grain;
        grainInput.active = grain.active;
        grainInput.sublayers_active = grain.sublayers_active;
        grainInput.layer_curves_required = grain.sublayers_active;
        grainInput.capture_polarity = view.polarity;
        grainInput.film_polarity = view.polarity;
        grainInput.sublayer_count = grain.sublayer_count;
        grainInput.pixel_size_um = facts.pixelSizeUm;
        grainInput.particle_area_um2 = grain.particle_area_um2;
        std::copy_n(grain.particle_scale_cmy, 3, grainInput.particle_scale_cmy);
        std::copy_n(grain.particle_scale_layers, 3, grainInput.particle_scale_layers);
        std::copy_n(grain.particle_density_min_cmy, 3, grainInput.particle_density_min_cmy);
        const auto& selectedBounds = (view.route & 1u) ? view.enlarger_film_bounds : view.scanner_bounds;
        std::copy_n(selectedBounds.maximum_cmy, 3, grainInput.authored_max_cmy);
        grainInput.correlation_sigma_px = grain.correlation_sigma_px;
        grainInput.dye_cloud_blur_um = grain.dye_cloud_blur_um;
        grainInput.size_mix_scale = grain.size_mix_scale;
        grainInput.fine_weight = grain.fine_weight;
        grainInput.mid_weight = grain.mid_weight;
        grainInput.coarse_weight = grain.coarse_weight;
        grainInput.clump_morph_period_sec = grain.clump_morph_period_sec;
        grainInput.render_extent = facts.renderExtent;
        grainInput.full_frame_extent = facts.fullFrameExtent;
        grainInput.frame_time = facts.frameTime;
        grainInput.frame_rate = facts.frameRate;
        grainInput.session_seed = facts.sessionSeed;
        grainInput.clip_token = facts.clipToken;
        grainInput.recipe_hash = grain.hash;
        grainInput.recipe_layers_hash = grain.layers_hash;
        grainInput.film_layers_hash = view.film_development.density_layers_hash;
        for (std::size_t layer = 0; layer < 3; ++layer)
            for (std::size_t channel = 0; channel < 3; ++channel)
                grainInput.layer_curves[layer][channel] = view.film_development.density_layers[layer][channel];
        FjGrainFrameMath grainMath{};
        FjGrainFrameFailure grainFailure{};
        error.length = 0;
        if (!facade_status(fj_test_grain_frame_prepare(&grainInput, &grainMath, &grainFailure, &error),
                           text,
                           error,
                           diagnostic,
                           "grain frame facade failed"))
            return false;
        const FjGrainCoefficients grainCoefficients = grain_coefficients(grain);
        Spektrafilm::VisualGrainFrameDescriptor grainDescriptor{};
        if (!Spektrafilm::compose_visual_grain_frame_descriptor(
                grainCoefficients, static_cast<Spektrafilm::ProfilePolarity>(view.polarity), grainMath, grainDescriptor, diagnostic)) {
            if (diagnostic.empty())
                diagnostic = "native visual grain descriptor composition failed";
            return false;
        }

        const FjEffectsFrameInput effectsInput{
            view.effects.active, facts.renderExtent, facts.pixelDefinition, facts.pixelSizeUm, facts.filmFormatLongEdgeMm, facts.canonicalX, facts.canonicalY, facts.canonicalWidth, facts.canonicalHeight, facts.scaleX, facts.scaleY, facts.pixelAspectRatio, facts.frameTime, facts.frameRate, view.effects.gate_weave_amount, facts.sessionSeed, facts.clipToken, view.effects.hash, view.effects.film_dust, view.effects.film_scratch, view.effects.gate_dust, view.effects.gate_scratch};
        FjEffectsFrameMath effectsMath{};
        FjEffectsFrameFailure effectsFailure{};
        error.length = 0;
        if (!facade_status(fj_test_effects_frame_prepare(&effectsInput, &effectsMath, &effectsFailure, &error),
                           text,
                           error,
                           diagnostic,
                           "effects frame facade failed"))
            return false;
        Spektrafilm::FilmJuicerEffectsFrameDescriptor effectsDescriptor{};
        if (!JuicerEffects::compose_frame_descriptor(
                effectsMath, view.effects.film_dust, view.effects.film_scratch, view.effects.gate_dust, view.effects.gate_scratch, effectsDescriptor, diagnostic)) {
            if (diagnostic.empty())
                diagnostic = "native effects descriptor composition failed";
            return false;
        }

        FjDiffusionFrameInput diffusionInput{};
        diffusionInput.route = view.route;
        diffusionInput.full_frame = facts.fullFrameExtent;
        diffusionInput.pixel_size_um = facts.pixelSizeUm;
        diffusionInput.camera = view.optics.camera_diffusion;
        diffusionInput.enlarger = view.optics.enlarger_diffusion;
        FjDiffusionFrameSet diffusionMath{};
        FjDiffusionFailure diffusionFailure{};
        error.length = 0;
        if (!facade_status(fj_test_diffusion_frame_prepare(&diffusionInput, &diffusionMath, &diffusionFailure, &error),
                           text,
                           error,
                           diagnostic,
                           "diffusion frame facade failed"))
            return false;
        auto diffusion = diffusion_descriptor(diffusionMath);

        FjHalationFrameInput halationInput{view.optics.scatter_halation, facts.pixelSizeUm};
        FjScatterHalation halationMath{};
        FjHalationFrameFailure halationFailure{};
        error.length = 0;
        if (!facade_status(fj_test_scatter_halation_frame_prepare(
                               &halationInput, &halationMath, &halationFailure, &error),
                           text,
                           error,
                           diagnostic,
                           "halation frame facade failed"))
            return false;
        std::optional<ScatterHalationFrameDescriptor> halation;
        if (halationMath.recipe_hash != 0)
            halation = halation_descriptor(halationMath);

        JuicerCuda::PreparedDescriptors descriptors{};
        descriptors.route = static_cast<Spektrafilm::ScanRoute>(view.route);
        descriptors.capturePolarity = static_cast<Spektrafilm::ProfilePolarity>(view.polarity);
        const bool print = Spektrafilm::scan_route_is_print(descriptors.route);
        descriptors.scanner.route = descriptors.route;
        descriptors.scanner.medium = print ? Scanner::ScannedMediumKind::Print : Scanner::ScannedMediumKind::Film;
        descriptors.scanner.polarity = descriptors.capturePolarity;
        descriptors.scanner.densityBoundsHash = view.scanner_lut.density_bounds_hash;
        descriptors.scanner.channelDensityHash = view.scanner_lut.channel_density_hash;
        descriptors.scanner.baseDensityHash = view.scanner_lut.base_density_hash;
        descriptors.scanner.scanIlluminantHash = view.scanner_lut.viewing_illuminant_hash;
        descriptors.scanner.observerHash = view.scanner_lut.observer_hash;
        descriptors.scanner.lutResolution = view.scanner_lut.resolution;
        descriptors.scanner.hash = view.scanner_lut.hash;
        const auto& correction = view.scanner_correction;
        descriptors.correction.route = descriptors.route;
        descriptors.correction.active = correction.active != 0;
        descriptors.correction.blackCorrection = correction.black_correction != 0;
        descriptors.correction.whiteCorrection = correction.white_correction != 0;
        descriptors.correction.targetBlackLinear = correction.target_black_linear;
        descriptors.correction.targetWhiteLinear = correction.target_white_linear;
        descriptors.correction.referenceBlackY = correction.reference_black_y;
        descriptors.correction.referenceWhiteY = correction.reference_white_y;
        descriptors.correction.xyzSlope = correction.xyz_slope;
        descriptors.correction.xyzOffset = correction.xyz_offset;
        descriptors.correction.exposureScale = correction.exposure_scale;
        descriptors.correction.schemaVersion = correction.schema_version;
        descriptors.correction.hash = correction.hash;
        const auto& post = view.scanner_post_effects;
        descriptors.post.route = descriptors.route;
        descriptors.post.glareActive = post.glare_active != 0;
        descriptors.post.glarePercent = post.glare_percent;
        descriptors.post.glareRoughness = post.glare_roughness;
        descriptors.post.glareBlurSigmaPx = post.glare_blur_sigma_px;
        descriptors.post.glareBlurRadius = post.glare_blur_radius;
        descriptors.post.lensBlurSigmaPx = post.lens_blur_sigma_px;
        descriptors.post.lensBlurRadius = post.lens_blur_radius;
        descriptors.post.unsharpSigmaPx = post.unsharp_sigma_px;
        descriptors.post.unsharpRadius = post.unsharp_radius;
        descriptors.post.unsharpAmount = post.unsharp_amount;
        descriptors.post.schemaVersion = post.schema;
        descriptors.post.hash = post.hash;
        fill_scanner_color(view.scanner_color, descriptors.color);
        fill_output_gamut(view.output_color, descriptors.outputGamut);
        descriptors.diffusion = diffusion;
        descriptors.halation = halation;
        descriptors.spatialDir = spatialDir;
        if (grainDescriptor.active)
            descriptors.grain = grainDescriptor;
        if (effectsDescriptor.hash != 0)
            descriptors.effects = effectsDescriptor;

        JuicerCuda::FocusedRouteResourceInput focused{};
        fill_focused(view, focused);
        Spectral::FilmRawConfig filmRawConfig{};
        std::copy_n(view.film_exposure.input_rgb_to_xyz, 9, filmRawConfig.inputRGBToXYZ.m);
        std::copy_n(view.film_exposure.input_xyz_adapt, 9, filmRawConfig.inputXYZAdapt.m);
        std::copy_n(view.film_exposure.xyz_to_linear_srgb, 9, filmRawConfig.xyzToLinearSrgb.m);
        filmRawConfig.applyInputChromaticAdapt =
            (view.film_exposure.flags & FJ_FILM_ADAPT_INPUT) != 0;
        Scanner::ColorRuntime scannerColor{};
        fill_scanner_color(view.scanner_color, scannerColor);
        OutputGamutRecipe outputGamut{};
        fill_output_gamut(view.output_color, outputGamut);
        JuicerCuda::FilmPayloadInput filmPayload{};
        fill_film_payload(view, filmPayload);
        JuicerCuda::VisualGrainPayloadInput grainPayload{
            grainCoefficients, view.grain.hash, view.grain.layers_hash};
        JuicerCuda::PrintResourceInput printInput{};
        PrintExposureRecipe printExposure{};
        std::variant<std::monostate, JuicerCuda::PrintResourcePreparation, JuicerCuda::PrintResourceInput> printResources;
        if (print) {
            print_input(view, printInput);
            printExposure.printExposure = view.print.exposure;
            printExposure.preflashExposure = view.print.preflash_exposure;
            printExposure.normalizePrintExposure = view.normalize_print_exposure != 0;
            printExposure.printExposureCompensation = view.print_exposure_compensation != 0;
            printExposure.normalizationMode =
                static_cast<Spektrafilm::PrintNormalizationMode>(view.print_normalization_mode);
            printExposure.hash = view.print_exposure_hash;
            printResources = printInput;
        }

        JuicerProcess::Root::PreparationIdentity identity{};
        identity.scanRoute = descriptors.route;
        identity.capturePolarity = descriptors.capturePolarity;
        identity.filmProfileKey = {view.film_profile_key.data, view.film_profile_key.count};
        identity.filmProfileAssetVersionToken = view.film_profile_asset_version;
        identity.scatterHalationHash = view.optics.scatter_halation.hash;
        identity.grainHash = view.grain.hash;
        identity.grainDensityLayersHash = view.film_development.density_layers_hash;
        identity.grainActive = view.grain.active != 0;
        identity.effectsHash = view.effects.hash;
        identity.effectsActive = descriptors.effects.has_value();

        auto* const assets = JuicerProcess::root().assets().borrow_rust_assets();
        if (assets == nullptr) {
            diagnostic = "prepared frame fixture has no process Rust assets owner";
            return false;
        }
        NoiseHold noiseOwner;
        FjStaticNoise noiseView{};
        if (view.grain.active != 0) {
            error.length = 0;
            if (!facade_status(noiseOwner.acquire(assets, &error), text, error, diagnostic, "Rust noise owner acquisition failed")) {
                (void)release_noise(noiseOwner, diagnostic);
                return false;
            }
            error.length = 0;
            if (!facade_status(noiseOwner.view(noiseView, &error), text, error, diagnostic, "Rust noise owner view failed")) {
                (void)release_noise(noiseOwner, diagnostic);
                return false;
            }
        }
        JuicerCuda::StaticNoiseInput staticNoise{};
        if (!static_noise_input(noiseView, staticNoise, diagnostic)) {
            (void)release_noise(noiseOwner, diagnostic);
            return false;
        }

        JuicerProcess::Root::PreparedFrameInput preparation{
            identity, focused, filmRawConfig, scannerColor, outputGamut, printResources, &descriptors.post, &descriptors.spatialDir, descriptors.diffusion ? &*descriptors.diffusion : nullptr, descriptors.halation ? &*descriptors.halation : nullptr, descriptors.grain, descriptors.effects, &staticNoise, facts.renderExtent.width, facts.renderExtent.height};
        bool consumed = false;
        std::exception_ptr primary;
        try {
            consumed = JuicerTestSupport::with_host_preparation(
                [&](const FjHostPreparation* hostPreparation) {
                    const Inputs inputs{view, descriptors, preparation, filmPayload, grainPayload, hostPreparation, view.film_exposure.route_correction_scale, printExposure};
                    return consume(user, inputs, diagnostic);
                });
        } catch (...) {
            primary = std::current_exception();
        }

        bool released = false;
        try {
            released = release_noise(noiseOwner, diagnostic);
        } catch (...) {
            if (!primary)
                throw;
        }
        if (primary)
            std::rethrow_exception(primary);
        return released && consumed;
    }

} // namespace JuicerCudaTest::PreparedFrameFixture
