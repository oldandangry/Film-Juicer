#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "JuicerState.h"
#include "PreparedState.h"
#include "ProcessRuntime.h"
#include "juicer_test_api.h"

namespace JuicerTestSupport {

    struct SubmissionObservation {
        bool valid = false;
        FjSubmission submission{};
        FjCudaContext context{};
    };

    inline SubmissionObservation submission_observation(const RenderInstance& instance) {
        SubmissionObservation result{};
        std::uint32_t valid = 0;
        if (fj_test_instance_submission(instance.borrow(), &valid, &result.submission, &result.context, nullptr).category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("cannot observe Rust submission");
        }
        result.valid = valid != 0;
        return result;
    }

    class PublicationObservation {
    public:
        PublicationObservation(const PublicationObservation&) = delete;
        PublicationObservation& operator=(const PublicationObservation&) = delete;
        PublicationObservation(PublicationObservation&& other) noexcept
            : view(other.view), owner_(std::exchange(other.owner_, nullptr)) {}
        ~PublicationObservation() {
            if (owner_) {
                (void)fj_test_instance_observation_release(owner_, nullptr);
            }
        }
        FjInstanceObservationView view{};

    private:
        explicit PublicationObservation(FjInstanceObservation* owner) : owner_(owner) {
            if (fj_test_instance_observation_view(owner_, &view, nullptr).category != FJ_STATUS_SUCCESS) {
                (void)fj_test_instance_observation_release(owner_, nullptr);
                owner_ = nullptr;
                throw std::runtime_error("cannot inspect Rust publication");
            }
        }
        FjInstanceObservation* owner_;
        friend class PublicationObserver;
    };

    struct PublicationMetrics {
        std::uint32_t pendingKind = 0;
        std::uint64_t inputIdentity = 0;
        std::uint64_t publishedIdentity = 0;
        std::uint64_t build = 0;
        std::size_t backingIdentity = 0;
        std::uint32_t route = 0;
        std::uint64_t recipeHash = 0;
        std::uint64_t issued = 0;
        std::uint64_t writeRevision = 0;
        std::uint64_t outstandingWrites = 0;
        std::string diagnostic;
    };

    class PublicationObserver {
    public:
        static std::size_t backing(const PreparedState& state) {
            return fj_test_prepared_state_identity(state.owner_);
        }
        static PublicationObservation observe(const RenderInstance& publication) {
            FjInstanceObservation* owner = nullptr;
            if (fj_test_instance_observation_acquire(publication.owner_, &owner, nullptr).category != FJ_STATUS_SUCCESS || !owner) {
                throw std::runtime_error("cannot observe Rust publication");
            }
            return PublicationObservation(owner);
        }
        static PublicationMetrics metrics(const RenderInstance& publication) {
            const auto observation = observe(publication);
            const auto& view = observation.view;
            return {view.pending_kind, view.input_identity, view.published_identity, view.build, view.backing_identity, view.published.route, view.published.recipe_hash, view.issued, view.write_revision, view.outstanding_writes, view.diagnostic.count ? std::string(view.diagnostic.data, view.diagnostic.count) : std::string{}};
        }
    };

    // Test-only copying observation for the existing captured-input assertions.
    // This never prepares or selects production state.
    inline ParamSnapshot captured_snapshot(const FjPreparationParams& raw) {
        ParamSnapshot snapshot;
        snapshot.printExposure = raw.print_exposure;
        snapshot.printPreflashExposure = raw.print_preflash_exposure;
        snapshot.printGammaFactor = raw.print_gamma_factor;
        snapshot.preflashMFilterCc = raw.preflash_m_filter_cc;
        snapshot.preflashYFilterCc = raw.preflash_y_filter_cc;
        snapshot.printShadowCompensationFactor = raw.print_shadow_compensation_factor;
        snapshot.printShadowCompensationDensity = raw.print_shadow_compensation_density;
        snapshot.printShadowCompensationTransition = raw.print_shadow_compensation_transition;
        snapshot.glarePercent = raw.glare_percent;
        snapshot.glareRoughness = raw.glare_roughness;
        snapshot.glareBlurSigmaPx = raw.glare_blur_sigma_px;
        snapshot.cameraExposureCompensationEv = raw.camera_exposure_compensation_ev;
        snapshot.cameraFilmFormatLongEdgeMm = raw.camera_film_format_long_edge_mm;
        snapshot.filmGammaFactor = raw.film_gamma_factor;
        snapshot.scannerLensBlurSigmaPx = raw.scanner_lens_blur_sigma_px;
        snapshot.scannerBlackLevel = raw.scanner_black_level;
        snapshot.scannerWhiteLevel = raw.scanner_white_level;
        snapshot.gateWeaveAmount = raw.gate_weave_amount;
        snapshot.spectralUpsamplingMode = raw.spectral_upsampling_mode;
        snapshot.refIll = raw.ref_ill;
        snapshot.enlIll = raw.enl_ill;
        snapshot.normalizePrintExposure = raw.normalize_print_exposure;
        snapshot.printExposureCompensation = raw.print_exposure_compensation;
        snapshot.inputColorSpace = raw.input_color_space;
        snapshot.inputCctfDecoding = raw.input_cctf_decoding;
        snapshot.inputCompressionEnabled = raw.input_compression_enabled;
        snapshot.hanatos2025AdaptationWindow = raw.hanatos2025_adaptation_window;
        snapshot.hanatos2025AdaptationSurface = raw.hanatos2025_adaptation_surface;
        snapshot.cameraAutoExposureEnabled = raw.camera_auto_exposure_enabled;
        snapshot.cameraMeteringMethod = raw.camera_metering_method;
        snapshot.scannerBlackCorrection = raw.scanner_black_correction;
        snapshot.scannerWhiteCorrection = raw.scanner_white_correction;
        snapshot.scannerLutResolution = raw.scanner_lut_resolution;
        snapshot.outputColorSpace = raw.output_color_space;
        snapshot.outputCctfEncoding = raw.output_cctf_encoding;
        snapshot.outputGamutCompressionEnabled = raw.output_gamut_compression_enabled;
        snapshot.filmDustAmount = raw.film_dust_amount;
        snapshot.filmScratchAmount = raw.film_scratch_amount;
        snapshot.gateDustAmount = raw.gate_dust_amount;
        snapshot.gateScratchAmount = raw.gate_scratch_amount;
        snapshot.filmProfileKey.assign(raw.film_profile_key.data, raw.film_profile_key.count);
        snapshot.printProfileKey.assign(raw.print_profile_key.data, raw.print_profile_key.count);
        snapshot.scanRoute = static_cast<Spektrafilm::ScanRoute>(raw.scan_route);
        snapshot.glareActive = raw.glare_active != 0;
        snapshot.cameraFilterOverride = raw.camera_filter_override != 0;
        std::copy(std::begin(raw.scanner_unsharp_mask), std::end(raw.scanner_unsharp_mask), snapshot.scannerUnsharpMask.begin());
        std::copy(std::begin(raw.print_ui_ymc_cc), std::end(raw.print_ui_ymc_cc), snapshot.printUiYmcCc.begin());
        std::copy(std::begin(raw.camera_filter_uv), std::end(raw.camera_filter_uv), snapshot.cameraFilterUV.begin());
        std::copy(std::begin(raw.camera_filter_ir), std::end(raw.camera_filter_ir), snapshot.cameraFilterIR.begin());
        const auto diffusion = [](const FjDiffusionParams& raw) {
            Spektrafilm::DiffusionFilterAuthoredControls out;
            out.active = raw.active != 0;
            out.family = static_cast<Spektrafilm::DiffusionFilterFamily>(raw.family);
            out.strength = raw.strength;
            out.spatialScale = raw.spatial_scale;
            out.haloWarmth = raw.halo_warmth;
            out.coreIntensity = raw.core_intensity;
            out.coreSize = raw.core_size;
            out.haloIntensity = raw.halo_intensity;
            out.haloSize = raw.halo_size;
            out.bloomIntensity = raw.bloom_intensity;
            out.bloomSize = raw.bloom_size;
            return out;
        };
        snapshot.cameraDiffusion = diffusion(raw.camera_diffusion);
        snapshot.enlargerDiffusion = diffusion(raw.enlarger_diffusion);
        const auto& dir = raw.dir_couplers;
        auto& controls = snapshot.dirCouplers;
        controls.active = dir.active != 0;
        controls.amount = dir.amount;
        controls.inhibitionSameLayer = dir.inhibition_same_layer;
        controls.inhibitionInterlayer = dir.inhibition_interlayer;
        controls.diffusionSizeUm = dir.diffusion_size_um;
        controls.diffusionTailUm = dir.diffusion_tail_um;
        controls.diffusionTailWeight = dir.diffusion_tail_weight;
        controls.gammaUseStock = dir.gamma_use_stock != 0;
        std::copy(std::begin(dir.gamma_same_layer_rgb), std::end(dir.gamma_same_layer_rgb), controls.gammaSameLayerRgb.begin());
        std::copy(std::begin(dir.gamma_interlayer_r_to_gb), std::end(dir.gamma_interlayer_r_to_gb), controls.gammaInterlayerRToGb.begin());
        std::copy(std::begin(dir.gamma_interlayer_g_to_rb), std::end(dir.gamma_interlayer_g_to_rb), controls.gammaInterlayerGToRb.begin());
        std::copy(std::begin(dir.gamma_interlayer_b_to_rg), std::end(dir.gamma_interlayer_b_to_rg), controls.gammaInterlayerBToRg.begin());
        std::copy(std::begin(dir.langmuir_donor_k_rgb), std::end(dir.langmuir_donor_k_rgb), controls.langmuirDonorKRgb.begin());
        std::copy(std::begin(dir.langmuir_receiver_k_rgb), std::end(dir.langmuir_receiver_k_rgb), controls.langmuirReceiverKRgb.begin());
        snapshot.scatterHalationControls = {raw.scatter_halation.active != 0, raw.scatter_halation.scatter_amount, raw.scatter_halation.scatter_spatial_scale, raw.scatter_halation.halation_amount, raw.scatter_halation.halation_spatial_scale};
        const auto& grain = raw.grain;
        auto& visual = snapshot.grainControls;
        visual.active = grain.active != 0;
        visual.sublayersActive = grain.sublayers_active != 0;
        visual.particleAreaUm2 = grain.particle_area_um2;
        visual.correlationSigmaPx = grain.correlation_sigma_px;
        visual.dyeCloudBlurUm = grain.dye_cloud_blur_um;
        visual.nSubLayers = grain.sublayer_count;
        visual.amplitude = grain.amplitude;
        visual.chromaMix = grain.chroma_mix;
        visual.chromaSharedWeight = grain.chroma_shared_weight;
        visual.chromaIndependentWeight = grain.chroma_independent_weight;
        visual.coarseWeight = grain.coarse_weight;
        visual.midWeight = grain.mid_weight;
        visual.sizeMixScale = grain.size_mix_scale;
        visual.clumpTemporalMix = grain.clump_temporal_mix;
        visual.clumpMorphPeriodSec = grain.clump_morph_period_sec;
        visual.debugView = grain.debug_view;
        std::copy(std::begin(grain.particle_scale_cmy), std::end(grain.particle_scale_cmy), visual.particleScaleCmy.begin());
        std::copy(std::begin(grain.particle_scale_layers), std::end(grain.particle_scale_layers), visual.particleScaleLayers.begin());
        std::copy(std::begin(grain.particle_density_min_cmy), std::end(grain.particle_density_min_cmy), visual.visualParticleDensityMinCmy.begin());
        std::copy(std::begin(grain.uniformity_cmy), std::end(grain.uniformity_cmy), visual.uniformityCmy.begin());
        std::copy(std::begin(grain.micro_structure), std::end(grain.micro_structure), visual.microStructure.begin());
        return snapshot;
    }

    inline ParamSnapshot captured_snapshot(const RenderInstance& publication) {
        const auto observation = PublicationObserver::observe(publication);
        if (observation.view.pending_kind != 4) {
            throw std::runtime_error("publication has no captured input");
        }
        return captured_snapshot(observation.view.captured);
    }

    struct PublicationAdmission {
        std::uint32_t kind = FJ_ADMISSION_NONE;
        FjStatus delivery{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        std::shared_ptr<const PreparedState> state;
        std::string diagnostic;
    };

    inline void capture(RenderInstance& publication, const ParamSnapshot& snapshot, std::optional<std::uint64_t> identity = std::nullopt) {
        FjErrorBuffer error{nullptr, 0, 0};
        FjCaptureStart capture{};
        FjSnapshotStart snapshotStart{};
        std::uint32_t update = FJ_UPDATE_NONE;
        if (publication.begin_capture(FJ_CAPTURE_CHANGE, capture, error).category != FJ_STATUS_SUCCESS ||
            capture.kind != FJ_CAPTURE_STARTED ||
            publication.begin_snapshot(capture.capture, snapshotStart, error).category != FJ_STATUS_SUCCESS ||
            snapshotStart.kind != FJ_SNAPSHOT_READY ||
            publication.install(capture.capture, snapshotStart.write_revision, snapshot, identity.value_or(hash_params(snapshot)), update, error).category != FJ_STATUS_SUCCESS ||
            update != FJ_UPDATE_APPLIED) {
            throw std::runtime_error("fixture capture did not install its complete snapshot");
        }
    }

    inline void invalidate(RenderInstance& publication, const std::string& diagnostic) {
        FjErrorBuffer error{nullptr, 0, 0};
        FjCaptureStart capture{};
        std::uint32_t update = FJ_UPDATE_NONE;
        if (publication.begin_capture(FJ_CAPTURE_CHANGE, capture, error).category != FJ_STATUS_SUCCESS ||
            publication.invalidate(capture.capture, FJ_CAPTURE_INVALID_CONTROLS, {diagnostic.data(), diagnostic.size()}, update, error).category != FJ_STATUS_SUCCESS ||
            update != FJ_UPDATE_APPLIED) {
            throw std::runtime_error("fixture invalidation did not complete capture");
        }
    }

    inline PublicationAdmission admit(const FjAssets* assets, RenderInstance& publication) {
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        PublicationAdmission result;
        std::optional<PreparedState> state;
        result.delivery = publication.admit(assets, result.kind, state, error);
        result.diagnostic.assign(text.data(), error.length);
        if (result.delivery.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        if (state) {
            result.state = std::make_shared<PreparedState>(std::move(*state));
        }
        return result;
    }

    inline FjPreparedStateView published_view(const PublicationAdmission& publication) {
        if (!publication.state) {
            throw std::runtime_error("publication has no admitted prepared state");
        }
        FjPreparedStateView view{};
        FjErrorBuffer error{nullptr, 0, 0};
        if (publication.state->view(view, error).category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("cannot view admitted Rust state");
        }
        return view;
    }

    inline PreparedState prepare_product(const FjAssets* assets, const ParamSnapshot& snapshot) {
        if (assets == nullptr)
            throw std::runtime_error("prepared product has no process Rust assets owner");

        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        std::optional<PreparedState> state;
        const FjStatus result = prepare_state(assets, snapshot, state, error);
        if (result.category != FJ_STATUS_SUCCESS || !state) {
            const auto length = std::min(error.length, diagnostic.size());
            throw std::runtime_error(std::string(diagnostic.data(), length));
        }
        return std::move(*state);
    }

    // Legacy complete-result captures record three inactive native defaults.
    // Their metadata does not describe a PrintState on a direct route.
    inline std::array<float, 3> captured_print_factors(const FjPreparedStateView& state) {
        const auto& balance = state.print_balance;
        const std::array factors{balance.factor_midgray, balance.factor_midgray_comp, balance.normalizer};
        if ((state.route & 1u) != 0)
            return factors;

        const auto zero = [](float value) {
            return std::bit_cast<std::uint32_t>(value) == 0;
        };
        if (state.print_recipe_hash != 0 || state.print.hash != 0 || state.print.flags != 0 ||
            state.print_profile_asset_version != 0 || state.print_profile_key.count != 0 ||
            balance.hash != 0 || balance.filtered_main_illuminant_hash != 0 ||
            !std::all_of(factors.begin(), factors.end(), zero) ||
            !std::all_of(std::begin(balance.baseline_raw_rgb), std::end(balance.baseline_raw_rgb), zero) ||
            !std::all_of(std::begin(balance.compensated_raw_rgb), std::end(balance.compensated_raw_rgb), zero)) {
            throw std::runtime_error("direct print projection is not inactive/zero before capture metadata mapping");
        }
        for (const auto span : {state.print.film_density_cmy, state.print.film_base_density, state.print.sensitivity_cmy, state.print.log_exposure, state.print.density_cmy, state.print.main_illuminant, state.print.preflash_illuminant}) {
            if (span.data != nullptr || span.count != 0)
                throw std::runtime_error("direct print projection has backing before capture metadata mapping");
        }
        return {1.0f, 1.0f, 1.0f};
    }

} // namespace JuicerTestSupport
