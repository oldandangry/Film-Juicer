#include "../ffi/spatial_optics_fixture.h"
#include "../ffi/print_illuminant_fixture.h"
#include "../ffi/exposure_fixture.h"
#include "../ffi/prepared_boundary.h"
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
#include "../ffi/production_state_contract.h"
#endif
#include "../ffi/prepared_product_fixture.h"
#include "../ffi/capture_identity.h"
#include "../ffi/prepared_frame_fixture.h"
#include "../ffi/case_results.h"
#include "../ffi/test_asset_workspace.h"
#include "juicer_cuda_call_scope.h"
#include "juicer_cuda_api.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "render_assertions.h"
#include "instance_host.h"
#include "module_test.h"
#include "ofxImageEffectExt.h"
#include "nlohmann/json.hpp"

#include "Hash.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "../ffi/preparation_fixture.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "ColorTransforms.h"
#include "ScanRoute.h"


#if defined(JUICER_PREPARED_BOUNDARY_TEST)
#include "Cuda/JuicerCudaExecutor.h"
#endif

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
#include <cstdarg>
#include <cstring>
#include "Cuda/JuicerCudaExecutor.h"

#include "../ffi/native_path.h"
#include "../ffi/effects_fixture.h"
#include "../ffi/juicer_test_api.h"

#include "processor_test_hooks.h"
#endif

namespace {
    struct AbortObservation {
        std::thread::id thread = std::this_thread::get_id();
        unsigned calls = 0;
        unsigned cancelAt = 0;
        unsigned throwAt = 0;
        bool sameThread = true;
    };
    thread_local AbortObservation* s_abortObservation = nullptr;

    int effect_abort(OfxImageEffectHandle) {
        if (!s_abortObservation) {
            return 0;
        }
        auto& observation = *s_abortObservation;
        ++observation.calls;
        observation.sameThread = observation.sameThread && observation.thread == std::this_thread::get_id();
        if (observation.calls == observation.throwAt) {
            throw std::runtime_error("abort suite exception");
        }
        return observation.calls == observation.cancelAt ? 1 : 0;
    }

    void require_cuda(cudaError_t status, const char* action) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
        }
    }

    constexpr int kWidth = 7;
    constexpr int kHeight = 5;
    constexpr float kCanary = -8.0f;

    struct Case {
        const char* name = nullptr;
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        int reconstruction = 0;
        int components = 0;
        int originX = 0;
        int originY = 0;
        bool combined = false;
        bool signedZeroGlare = false;
        bool negativeZero = false;
        bool uniform = false;
        int width = kWidth;
        int height = kHeight;
        double time = 37.0;
        double scaleX = 1.0;
        double scaleY = 1.0;
        const std::vector<float>* sourcePixels = nullptr;
        int fullWidth = 0;
        int fullHeight = 0;
    };

    struct DeviceFrame {
        float* source = nullptr;
        float* destination = nullptr;
        cudaStream_t stream = nullptr;
        bool destinationInSource = false;

        ~DeviceFrame() {
            if (stream) {
                (void)cudaStreamSynchronize(stream);
            }
            if (destination && !destinationInSource) {
                (void)cudaFree(destination);
            }
            if (source) {
                (void)cudaFree(source);
            }
            if (stream) {
                (void)cudaStreamDestroy(stream);
            }
        }
    };

#if defined(JUICER_PREPARED_BOUNDARY_TEST)
    std::array<std::vector<float>, 2> production_pixels() {
        std::ifstream fixture(JUICER_PROCESSOR_REFERENCE_PATH);
        if (!fixture)
            throw std::runtime_error("production state pixel fixture unavailable");
        constexpr std::array names{"negative-direct", "negative-print", "positive-direct", "positive-print", "combined-print", "glare-plus-zero", "glare-minus-zero"};
        std::array<std::vector<float>, 2> result;
        for (std::size_t index = 0; index < names.size(); ++index) {
            std::string name;
            std::size_t count = 0;
            fixture >> name >> count;
            if (!fixture || name != names[index] || count == 0)
                throw std::runtime_error("production state pixel fixture membership");
            std::vector<float> pixels(count);
            for (float& pixel : pixels)
                fixture >> pixel;
            if (!fixture)
                throw std::runtime_error("production state pixel fixture truncated");
            if (index < result.size()) {
                nlohmann::json words = nlohmann::json::array();
                for (float pixel : pixels)
                    words.push_back(std::bit_cast<std::uint32_t>(pixel));
                const auto selected = PrintIlluminantFixtures::apply(
                    std::string(JUICER_PROCESSOR_REFERENCE_PATH) + "." + name, words);
                if (selected.size() != pixels.size())
                    throw std::runtime_error("production state pixel fixture supplement shape");
                for (std::size_t sample = 0; sample < pixels.size(); ++sample)
                    pixels[sample] = std::bit_cast<float>(selected.at(sample).get<std::uint32_t>());
                result[index] = std::move(pixels);
            }
        }
        std::string extra;
        if (fixture >> extra)
            throw std::runtime_error("production state pixel fixture extra row");
        return result;
    }
#endif

    JuicerTestSupport::PreparationInput parameters_for(const Case& test) {
        JuicerTestSupport::PreparationInput parameters;
        parameters.values.scan_route = static_cast<std::uint32_t>(test.route);
        if (Spektrafilm::scan_route_metadata(test.route).capturePolarity ==
            Spektrafilm::ProfilePolarity::Positive) {
            parameters.film_key = "fujifilm_provia_100f";
        }
        parameters.values.spectral_upsampling_mode = test.reconstruction;
        parameters.values.camera_auto_exposure_enabled = 0;
        if (!test.combined) {
            parameters.values.scanner_unsharp_mask[0] = parameters.values.scanner_unsharp_mask[1] = 0.0;
        }
        parameters.values.gate_weave_amount = 0.0;
        if (test.combined) {
            parameters.values.camera_diffusion.active = true;
            parameters.values.grain.active = true;
            parameters.values.scatter_halation = {1, 1.0f, 1.0f, 1.0f, 1.0f};
        }
        if (test.signedZeroGlare) {
            parameters.values.glare_active = true;
            parameters.values.glare_percent = 0.3;
            parameters.values.camera_exposure_compensation_ev = test.negativeZero ? -0.0 : 0.0;
        }
        return parameters;
    }

    enum class ExecutionPath : std::uint8_t {
        Processor,
        Direct,
        Boundary,
        State,
        StateContract,
        StateLifetime
    };

    enum class DestinationLayout : std::uint8_t {
        Matching,
        Offset,
        Uncovered,
        DisjointRows
    };

    enum class PublishedRequestAvailability : std::uint8_t {
        Complete,
        MissingState,
        MissingInstance,
        UninitializedIdentity,
        MissingStateUninitialized
    };

    // A session borrows the Runtime held by the loaded descriptor. Its effects
    // and every callback end before Unload can consume that Runtime.
    class CallbackSession final {
    public:
        CallbackSession() = default;
        CallbackSession(const CallbackSession&) = delete;
        CallbackSession& operator=(const CallbackSession&) = delete;
        ~CallbackSession() {
            if (loaded_ && close().category != FJ_STATUS_SUCCESS)
                std::fputs("secondary failure: callback session unload failed\n", stderr);
        }

        void create(const std::filesystem::path&) {
            if (loaded_)
                throw std::runtime_error("callback session already loaded");
            plugin_ = OfxGetPlugin(0);
            if (!plugin_)
                throw std::runtime_error("standard OFX descriptor unavailable");
            host_ = std::make_unique<ofx_test::InstanceHost>();
            plugin_->setHost(host_->binding());
            if (call(kOfxActionLoad) != kOfxStatOK)
                throw std::runtime_error("callback Load failed");
            loaded_ = true;
            if (call(kOfxImageEffectActionDescribeInContext, host_->descriptor_handle(), host_->context_args()) != kOfxStatOK) {
                (void)close();
                throw std::runtime_error("callback DescribeInContext failed");
            }
            (void)borrow();
        }

        FjRuntimeView borrow() const {
            FjRuntimeView view{static_cast<const FjAssets*>(fj_test_ofx_assets()), fj_test_ofx_native()};
            if (!loaded_ || !view.assets || !view.cuda)
                throw std::runtime_error("loaded callback Runtime unavailable");
            return view;
        }

        FjStatus shutdown() noexcept {
            return fj_test_ofx_runtime_shutdown();
        }

        FjStatus close() noexcept {
            if (!loaded_)
                return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            if (!effects_.empty())
                return {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
            const auto result = call(kOfxActionUnload);
            if (result != kOfxStatOK)
                return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
            loaded_ = false;
            plugin_ = nullptr;
            fj_test_ofx_render_tokens(0, 0, 0);
            host_.reset();
            return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        }

        OfxImageEffectHandle find(const JuicerTestSupport::InstanceFixture& identity) const {
            const auto it = effects_.find(&identity);
            return it == effects_.end() ? nullptr : it->second;
        }

        JuicerTestSupport::SubmissionObservation submission(const JuicerTestSupport::InstanceFixture& identity) const {
            auto* effect = find(identity);
            if (!effect)
                return JuicerTestSupport::submission_observation(identity);
            JuicerTestSupport::SubmissionObservation result{};
            std::uint32_t valid = 0;
            if (fj_test_ofx_instance_submission(effect, &valid, &result.submission, &result.context, nullptr).category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("cannot observe callback submission");
            result.valid = valid != 0;
            return result;
        }

        OfxImageEffectHandle effect(const JuicerTestSupport::InstanceFixture& identity, std::uint64_t token) {
            if (auto* found = find(identity))
                return found;
            fj_test_ofx_render_tokens(0x20260923, token, 0x5312);
            auto* effect = host_->create_effect();
            if (call(kOfxActionCreateInstance, effect) != kOfxStatOK)
                throw std::runtime_error("callback CreateInstance failed");
            try {
                effects_.emplace(&identity, effect);
            } catch (...) {
                if (call(kOfxActionDestroyInstance, effect) != kOfxStatOK)
                    std::fputs("secondary failure: unpublished fixture effect destruction failed\n", stderr);
                throw;
            }
            return effect;
        }

        bool release(const JuicerTestSupport::InstanceFixture& identity) noexcept {
            const auto it = effects_.find(&identity);
            if (it == effects_.end())
                return true;
            const bool ok = call(kOfxActionDestroyInstance, it->second) == kOfxStatOK;
            if (ok)
                effects_.erase(it);
            return ok;
        }

        OfxStatus call(const char* action, const void* handle = nullptr, OfxPropertySetHandle input = nullptr) const noexcept {
            return plugin_->mainEntry(action, handle, input, nullptr);
        }
        ofx_test::InstanceHost& host() noexcept {
            return *host_;
        }

    private:
        std::unique_ptr<ofx_test::InstanceHost> host_;
        OfxPlugin* plugin_ = nullptr;
        bool loaded_ = false;
        std::map<const JuicerTestSupport::InstanceFixture*, OfxImageEffectHandle> effects_;
    };

    struct CallbackStatus : std::exception {
        explicit CallbackStatus(OfxStatus value) noexcept : value(value) {}
        OfxStatus value;
        OfxStatus status() const noexcept {
            return value;
        }
        const char* what() const noexcept override {
            return "OFX callback failed";
        }
    };

    struct CallbackReleases {
        ofx_test::InstanceHost& host;
        OfxImageEffectHandle effect;
        CallbackReleases* parent;
        std::size_t sourceBefore, outputBefore;
        std::size_t nestedSource = 0, nestedOutput = 0;
        static thread_local CallbackReleases* current;

        CallbackReleases(ofx_test::InstanceHost& host, OfxImageEffectHandle effect)
            : host(host), effect(effect), parent(current),
              sourceBefore(host.image_release_count(effect, "Source")),
              outputBefore(host.image_release_count(effect, "Output")) {
            current = this;
        }
        ~CallbackReleases() {
            current = parent;
            if (parent && parent->effect == effect) {
                parent->nestedSource += host.image_release_count(effect, "Source") - sourceBefore;
                parent->nestedOutput += host.image_release_count(effect, "Output") - outputBefore;
            }
        }
    };
    thread_local CallbackReleases* CallbackReleases::current = nullptr;

    void callback_controls(ofx_test::InstanceHost& host, OfxImageEffectHandle effect, const JuicerTestSupport::PreparationInput& input) {
        const auto& p = input.values;
        const auto integer = [&](const char* name, int value) {
            host.set_value(effect, name, {value});
        };
        const auto scalar = [&](const char* name, double value) {
            host.set_value(effect, name, {value});
        };
        const auto pair = [&](const char* name, const auto& value) {
            host.set_value(effect, name, {double(value[0]), double(value[1])});
        };
        const auto triplet = [&](const char* name, const auto& value) {
            host.set_value(effect, name, {double(value[0]), double(value[1]), double(value[2])});
        };
        host.set_value(effect, "FilmProfileKey", {input.film_key});
        host.set_value(effect, "PrintProfileKey", {input.print_key});
        const std::array routeKeys{"NegativeDirectScan", "NegativePrintScan", "PositiveDirectScan", "PositivePrintScan"};
        if (p.scan_route >= routeKeys.size())
            throw std::runtime_error("invalid callback route");
        host.set_value(effect, "ScanRoute", {std::string(routeKeys[p.scan_route])});
        scalar("Exposure", p.camera_exposure_compensation_ev);
        integer("CameraAutoExposure", p.camera_auto_exposure_enabled);
        scalar("CameraFilmFormatMm", p.camera_film_format_long_edge_mm);
        integer("CameraMeteringMethod", p.camera_metering_method);
        integer("SpectralUpsampling", p.spectral_upsampling_mode);
        integer("ReferenceIlluminant", p.ref_ill);
        integer("EnlargerIlluminant", p.enl_ill);
        integer("InputColorSpace", p.input_color_space);
        integer("InputCctfDecoding", p.input_cctf_decoding);
        integer("InputCompression", p.input_compression_enabled);
        integer("Hanatos2025AdaptationWindow", p.hanatos2025_adaptation_window);
        integer("Hanatos2025AdaptationSurface", p.hanatos2025_adaptation_surface);
        scalar("FilmGammaFactor", p.film_gamma_factor);
        scalar("PrintGammaFactor", p.print_gamma_factor);
        integer("OutputColorSpace", p.output_color_space);
        integer("OutputCctfEncoding", p.output_cctf_encoding);
        integer("OutputGamutCompression", p.output_gamut_compression_enabled);
        const auto& d = p.dir_couplers;
        integer("CouplersActive", int(d.active));
        scalar("CouplersAmount", d.amount);
        scalar("CouplersInhibitionSameLayer", d.inhibition_same_layer);
        scalar("CouplersInhibitionInterlayer", d.inhibition_interlayer);
        scalar("CouplersDiffusionSizeUm", d.diffusion_size_um);
        scalar("CouplersDiffusionTailUm", d.diffusion_tail_um);
        scalar("CouplersDiffusionTailWeight", d.diffusion_tail_weight);
        integer("CouplersGammaUseStock", int(d.gamma_use_stock));
        triplet("CouplersGammaSameLayerRgb", d.gamma_same_layer_rgb);
        pair("CouplersGammaInterlayerRToGb", d.gamma_interlayer_r_to_gb);
        pair("CouplersGammaInterlayerGToRb", d.gamma_interlayer_g_to_rb);
        pair("CouplersGammaInterlayerBToRg", d.gamma_interlayer_b_to_rg);
        triplet("CouplersLangmuirDonorKRgb", d.langmuir_donor_k_rgb);
        triplet("CouplersLangmuirReceiverKRgb", d.langmuir_receiver_k_rgb);
        scalar("ScannerLensBlurSigmaPx", p.scanner_lens_blur_sigma_px);
        pair("ScannerUnsharpMask", p.scanner_unsharp_mask);
        integer("ScannerBlackCorrection", p.scanner_black_correction);
        integer("ScannerWhiteCorrection", p.scanner_white_correction);
        scalar("ScannerBlackLevel", p.scanner_black_level);
        scalar("ScannerWhiteLevel", p.scanner_white_level);
        integer("ScannerLutResolution", p.scanner_lut_resolution);
        scalar("PrintExposure", p.print_exposure);
        scalar("PrintPreflash", p.print_preflash_exposure);
        integer("PrintExposureCompensation", p.print_exposure_compensation);
        scalar("EnlargerY", p.print_ui_ymc_cc[0]);
        scalar("EnlargerM", p.print_ui_ymc_cc[1]);
        scalar("EnlargerC", p.print_ui_ymc_cc[2]);
        integer("HalationActive", int(p.scatter_halation.active));
        scalar("HalationScatterAmount", p.scatter_halation.scatter_amount);
        scalar("HalationScatterSpatialScale", p.scatter_halation.scatter_spatial_scale);
        scalar("HalationAmount", p.scatter_halation.halation_amount);
        scalar("HalationSpatialScale", p.scatter_halation.halation_spatial_scale);
        const auto& g = p.grain;
        integer("GrainActive", int(g.active));
        integer("GrainSublayersActive", int(g.sublayers_active));
        scalar("GrainParticleAreaUm2", g.particle_area_um2);
        scalar("GrainAmplitude", std::log2(double(g.amplitude)));
        scalar("GrainChroma", g.chroma_mix);
        triplet("GrainParticleScale", g.particle_scale_cmy);
        triplet("GrainParticleScaleLayers", g.particle_scale_layers);
        triplet("GrainDensityMin", g.particle_density_min_cmy);
        triplet("GrainUniformity", g.uniformity_cmy);
        scalar("GrainBlur", g.correlation_sigma_px);
        scalar("GrainBlurDyeCloudsUm", g.dye_cloud_blur_um);
        scalar("GrainSizeMixWeight", g.coarse_weight);
        scalar("GrainSizeMixWeightMid", g.mid_weight);
        scalar("GrainSizeMixScale", g.size_mix_scale);
        scalar("GrainClumpTemporalMix", g.clump_temporal_mix);
        scalar("GrainClumpMorphPeriodSec", g.clump_morph_period_sec);
        integer("GrainDebugView", g.debug_view);
        pair("GrainMicroStructure", g.micro_structure);
        scalar("FilmDustAmount", p.film_dust_amount);
        scalar("FilmScratchAmount", p.film_scratch_amount);
        scalar("GateDustAmount", p.gate_dust_amount);
        scalar("GateScratchAmount", p.gate_scratch_amount);
        scalar("GateWeaveAmount", p.gate_weave_amount);
        const auto diffusion = [&](const char* prefix, const FjDiffusionParams& f) {
            const std::string key(prefix);
            integer((key + "Enabled").c_str(), int(f.active));
            integer((key + "Family").c_str(), int(f.family));
            scalar((key + "Strength").c_str(), f.strength);
            scalar((key + "SpatialScale").c_str(), f.spatial_scale);
            scalar((key + "HaloWarmth").c_str(), f.halo_warmth);
            scalar((key + "CoreIntensity").c_str(), f.core_intensity);
            scalar((key + "CoreSize").c_str(), f.core_size);
            scalar((key + "HaloIntensity").c_str(), f.halo_intensity);
            scalar((key + "HaloSize").c_str(), f.halo_size);
            scalar((key + "BloomIntensity").c_str(), f.bloom_intensity);
            scalar((key + "BloomSize").c_str(), f.bloom_size);
        };
        diffusion("CameraDiffusion", p.camera_diffusion);
        diffusion("PrintDiffusion", p.enlarger_diffusion);
        integer("GlareActive", int(p.glare_active));
        scalar("GlarePercent", p.glare_percent);
        scalar("GlareRoughness", p.glare_roughness);
        scalar("GlareBlurSigmaPx", p.glare_blur_sigma_px);
        scalar("PrintShadowCompensationFactor", p.print_shadow_compensation_factor);
        scalar("PrintShadowCompensationDensity", p.print_shadow_compensation_density);
        scalar("PrintShadowCompensationTransition", p.print_shadow_compensation_transition);
    }

    struct CallbackLayout {
        int sourcePitch;
        int destinationPitch;
        int border;
        bool uncovered;
        bool emptyWindow;
    };
    void callback_render(CallbackSession& runtime, const Case& test, const JuicerTestSupport::PreparationInput& input, JuicerTestSupport::InstanceFixture& identity, DeviceFrame& device, CallbackLayout layout) {
        const auto& [pitch, destinationPitch, border, uncovered, emptyWindow] = layout;
        const std::uint64_t token = 0x641207ull + static_cast<std::uint64_t>(test.route) + (test.combined ? 4ull : 0ull);
        auto* effect = runtime.effect(identity, token);
        auto& host = runtime.host();
        callback_controls(host, effect, input);
        auto* change = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "Exposure"}, test.time);
        if (runtime.call(kOfxActionInstanceChanged, effect, change) != kOfxStatReplyDefault)
            throw std::runtime_error("callback input capture failed");
        auto* args = host.action_args();
        const int x = test.originX, y = test.originY, width = test.width, height = test.height;
        const int fullWidth = test.fullWidth > 0 ? test.fullWidth : width;
        const int fullHeight = test.fullHeight > 0 ? test.fullHeight : height;
        const int fullX = test.fullWidth > 0 ? 0 : x;
        const int fullY = test.fullHeight > 0 ? 0 : y;
        host.set_effect_property(effect, kOfxImageEffectPropFrameRate, {24.0});
        host.set_property(args, kOfxPropTime, {test.time});
        host.set_property(args, kOfxImageEffectPropRenderScale, {test.scaleX, test.scaleY});
        host.set_property(args, kOfxImageEffectPropRenderWindow, {x, y, emptyWindow ? x : x + width, y + height});
        host.set_property(args, kOfxImageEffectPropCudaEnabled, {1});
        host.set_property(args, kOfxImageEffectPropCudaStream, {static_cast<void*>(device.stream)});
        host.set_property(args, kOfxImageEffectPropFieldToRender, {std::string(kOfxImageFieldNone)});
        for (const char* clip : {"Source", "Output"}) {
            const bool source = std::string_view(clip) == "Source";
            const int pad = source ? 0 : border;
            const int endX = x + width + pad - (!source && uncovered ? 1 : 0);
            const auto set = [&](const char* name, std::vector<ofx_test::InstanceValue> value) {
                host.set_image_property(effect, clip, name, std::move(value));
            };
            set(kOfxImagePropData, {static_cast<void*>(source ? device.source : device.destination)});
            set(kOfxImagePropBounds, {x - pad, y - pad, endX, y + height + pad});
            set(kOfxImagePropRegionOfDefinition, {fullX, fullY, fullX + fullWidth, fullY + fullHeight});
            set(kOfxImagePropRowBytes, {(source ? pitch : destinationPitch) * int(sizeof(float))});
            set(kOfxImageEffectPropComponents, {std::string(test.components == 4 ? kOfxImageComponentRGBA : kOfxImageComponentRGB)});
            set(kOfxImageEffectPropPixelDepth, {std::string(kOfxBitDepthFloat)});
            set(kOfxImageEffectPropPreMultiplication, {std::string(kOfxImageUnPreMultiplied)});
            set(kOfxImagePropField, {std::string(kOfxImageFieldNone)});
            set(kOfxImagePropUniqueIdentifier, {std::string(clip)});
            set(kOfxImageEffectPropRenderScale, {test.scaleX, test.scaleY});
            set(kOfxImagePropPixelAspectRatio, {1.0});
            host.set_clip_property(effect, clip, kOfxImageEffectPropFrameRate, {24.0});
            host.set_clip_region(effect, clip, {double(fullX), double(fullY), double(fullX + fullWidth), double(fullY + fullHeight)});
        }
        auto previousAbort = host.set_abort_callback([] {
            try {
                return effect_abort(nullptr);
            } catch (...) {
                fj_test_ofx_render_control(7);
                return 0;
            }
        });
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        auto previousMessage = host.set_message_callback([&runtime](std::string_view type, std::string_view id, std::string_view text) {
            auto* observation = JuicerCuda::ExecutorTest::observation;
            if (!observation)
                return kOfxStatOK;
            JuicerCuda::NativeCall call(runtime.borrow().cuda);
            observation->record(JuicerCuda::ExecutorTest::Event::Message);
            observation->messageMatches = type == kOfxMessageError && id == "FilmJuicerDeferredCudaFailure" && text == observation->diagnostic;
            if (observation->delivery == JuicerCuda::ExecutorTest::Delivery::Throw)
                throw std::runtime_error("injected host message exception");
            return observation->delivery == JuicerCuda::ExecutorTest::Delivery::Failure ? kOfxStatFailed : kOfxStatOK;
        });
#endif
        struct ResetCallbacks {
            ofx_test::InstanceHost& host;
            std::function<int()> abort;
            std::function<OfxStatus(std::string_view, std::string_view, std::string_view)> message;
            ~ResetCallbacks() {
                host.set_abort_callback(std::move(abort));
                host.set_message_callback(std::move(message));
            }
        } reset{host, std::move(previousAbort),
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
                std::move(previousMessage)
#else
                {}
#endif
        };
        CallbackReleases releases(host, effect);
        fj_test_ofx_render_tokens(0x20260923, token, 0x5312);
        OfxStatus status;
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        const auto* observation = JuicerCuda::ExecutorTest::observation;
        if (observation && observation->diffusionComponentsPanic) {
            struct ClearDiffusionFault {
                ~ClearDiffusionFault() {
                    (void)fj_test_diffusion_frame_fault(2, 0);
                }
            } clear;
            if (fj_test_diffusion_frame_fault(2, 1).category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("could not arm diffusion components panic");
            }
            status = runtime.call(kOfxImageEffectActionRender, effect, args);
        } else
#endif
        {
            status = runtime.call(kOfxImageEffectActionRender, effect, args);
        }
        if (status != kOfxStatOK)
            throw CallbackStatus{status};
        FjOfxRenderObservation observed{};
        if (fj_test_ofx_render_observe(effect, &observed) && observed.native_entry_count != 0 &&
            observed.input_identity != JuicerTestSupport::hash_input_identity(input))
            throw std::runtime_error("callback captured input differs from the requested fixture");
        if (!emptyWindow && (host.image_release_count(effect, "Source") != releases.sourceBefore + releases.nestedSource + 1 ||
                             host.image_release_count(effect, "Output") != releases.outputBefore + releases.nestedOutput + 1))
            throw std::runtime_error("callback image release count changed");
    }

    struct MembershipRetirement {
        CallbackSession& runtime;
        const JuicerTestSupport::InstanceFixture& instance;
        bool active = true;

        bool close() noexcept {
            if (!std::exchange(active, false))
                return true;
            if (runtime.find(instance))
                return runtime.release(instance);
            std::uint32_t initialized = 0;
            std::uint64_t seed = 0, token = 0;
            FjErrorBuffer error{nullptr, 0, 0};
            if (fj_test_instance_identity(instance.borrow(), &initialized, &seed, &token, &error).category != FJ_STATUS_SUCCESS)
                return false;
            if (!initialized)
                return true;
            try {
                return fj_cuda_retire_instance(runtime.borrow().cuda, token, &error).category == FJ_STATUS_SUCCESS;
            } catch (...) {
                return false;
            }
        }
        ~MembershipRetirement() noexcept(false) {
            if (!close()) {
                if (std::uncaught_exceptions() == 0)
                    throw std::runtime_error("fixture submission membership retirement failed");
                std::fputs("secondary failure: fixture submission membership retirement failed\n", stderr);
            }
        }
    };

    struct ScopedSubmissionOwner {
        JuicerTestSupport::InstanceFixture instance;
        FjCuda* cuda = nullptr;
        std::uint64_t token = 0;

        bool close() noexcept {
            if (!cuda)
                return true;
            FjErrorBuffer ignored{nullptr, 0, 0};
            FjCuda* retiredCuda = cuda;
            cuda = nullptr;
            return fj_cuda_retire_instance(retiredCuda, token, &ignored).category == FJ_STATUS_SUCCESS;
        }
        ~ScopedSubmissionOwner() {
            (void)close();
        }
    };

    struct FrameDescriptorObservation;
    struct PreparedRenderCompletion;
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
    FjCallCompletionResult render_unavailable_case(CallbackSession&, const JuicerTestSupport::PreparedStateHold*, const FjFrame&, PublishedRequestAvailability, JuicerTestSupport::InstanceFixture&, std::string&);
#endif
    FjCallCompletionResult render_held_prepared_state(
        CallbackSession& runtime,
        const JuicerTestSupport::PreparedStateHold&,
        const FjFrame&,
        std::uint64_t,
        FjAbortCallback,
        FrameDescriptorObservation*,
        bool*,
        std::string&,
        const FjAssets* = nullptr,
        PreparedRenderCompletion* = nullptr,
        bool = false,
        JuicerTestSupport::InstanceFixture* = nullptr);

    struct DirSubmissionWitness {
        unsigned selectedCalls = 0;
        unsigned renderCalls = 0;
        std::uint64_t completedDirHash = ~std::uint64_t{};
        FjSubmission selected{};
        FjSubmission consumed{};
        const FjInstance* instance = nullptr;
        OfxImageEffectHandle effect = nullptr;
        bool traceUnlocked = false;
        FjRenderOutcome (*forward)(void*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, const FjHostPreparation*, FjAbortCallback, FjErrorBuffer*) = nullptr;
    };
    thread_local DirSubmissionWitness* dirSubmissionWitness = nullptr;

    std::vector<float> render_case(CallbackSession& runtime, const Case& test, const JuicerTestSupport::PreparationInput& parameters, JuicerTestSupport::InstanceFixture& state, bool emptyWindow = false, ExecutionPath path = ExecutionPath::Processor, DestinationLayout destinationLayout = DestinationLayout::Matching, [[maybe_unused]] std::uint64_t snapshotId = 1, [[maybe_unused]] JuicerCudaTest::StateLifetimeCase lifetimeCase = JuicerCudaTest::StateLifetimeCase::SuppliedStream, PublishedRequestAvailability availability = PublishedRequestAvailability::Complete, [[maybe_unused]] std::string_view invocationPhase = "cold") {
        std::array<char, 512> identityText{};
        FjErrorBuffer identityError{identityText.data(), identityText.size(), 0};
        std::uint32_t initialized = 0;
        std::uint64_t seed = 0, token = 0;
        if (fj_test_instance_identity(state.borrow(), &initialized, &seed, &token, &identityError).category != FJ_STATUS_SUCCESS)
            throw std::runtime_error("reference instance identity read failed");
        const std::uint64_t expectedToken = 0x641207ull + static_cast<std::uint64_t>(test.route) + (test.combined ? 4ull : 0ull);
        const bool leaveUninitialized = availability == PublishedRequestAvailability::UninitializedIdentity ||
                                        availability == PublishedRequestAvailability::MissingStateUninitialized;
        if (!initialized && !leaveUninitialized && state.bind_identity(0x20260923, expectedToken, identityError).category != FJ_STATUS_SUCCESS)
            throw std::runtime_error("reference instance identity bind failed");
        if (initialized && (seed != 0x20260923 || token != expectedToken))
            throw std::runtime_error("reference instance identity changed");
        const int width = test.width;
        const int height = test.height;
        const int pitch = width * test.components + 5;
        const auto sourceSamples = static_cast<std::size_t>(pitch) * static_cast<std::size_t>(height);
        const std::size_t bytes = sourceSamples * sizeof(float);
        std::vector<float> input(sourceSamples, -7.0f);
        const int border = destinationLayout == DestinationLayout::Offset ? 1 : 0;
        const int destinationPitch = pitch + 2 * border * test.components;
        const int destinationHeight = height + 2 * border;
        const auto destinationSamples = static_cast<std::size_t>(destinationPitch) * static_cast<std::size_t>(destinationHeight);
        const std::size_t destinationBytes = destinationSamples * sizeof(float);
        std::vector<float> output(destinationSamples, kCanary);
        std::vector<float> initialDestination(output);
        for (int y = 0; y < height; ++y) {
            const std::size_t rowOffset = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixelOffset = rowOffset +
                                                static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    input[pixelOffset + static_cast<std::size_t>(c)] =
                        c == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f
                               : static_cast<float>(((test.uniform ? 0 : x + 3 * y) + 7 * c) % 19) / 19.0f;
                }
            }
        }
        if (test.sourcePixels) {
            const std::size_t rowSamples = static_cast<std::size_t>(width) * static_cast<std::size_t>(test.components);
            if (test.sourcePixels->size() != rowSamples * static_cast<std::size_t>(height)) {
                throw std::runtime_error("accepted capture input shape mismatch");
            }
            for (int y = 0; y < height; ++y) {
                const std::size_t sourceOffset = static_cast<std::size_t>(y) * rowSamples;
                const std::size_t destinationOffset = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch);
                std::copy_n(test.sourcePixels->data() + sourceOffset, rowSamples, input.data() + destinationOffset);
            }
        }
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create stream");
        device.destinationInSource = destinationLayout == DestinationLayout::DisjointRows;
        require_cuda(cudaMalloc(&device.source, bytes + (device.destinationInSource ? destinationBytes : 0)), "allocate source");
        if (device.destinationInSource) {
            device.destination = device.source + bytes / sizeof(float);
        } else {
            require_cuda(cudaMalloc(&device.destination, destinationBytes), "allocate destination");
        }
        require_cuda(cudaMemcpy(device.source, input.data(), bytes, cudaMemcpyHostToDevice), "copy source");
        require_cuda(cudaMemcpy(device.destination, initialDestination.data(), destinationBytes, cudaMemcpyHostToDevice), "copy destination");

        if (path == ExecutionPath::Processor && availability == PublishedRequestAvailability::Complete) {
            callback_render(runtime, test, parameters, state, device, {.sourcePitch = pitch, .destinationPitch = destinationPitch, .border = border, .uncovered = destinationLayout == DestinationLayout::Uncovered, .emptyWindow = emptyWindow});
            if (s_abortObservation && s_abortObservation->cancelAt != 0 &&
                s_abortObservation->calls >= s_abortObservation->cancelAt)
                return {};
        }
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        else if (path == ExecutionPath::State || path == ExecutionPath::StateContract || path == ExecutionPath::StateLifetime) {
            const FjRect bounds{test.originX, test.originY, test.originX + width, test.originY + height};
            FjFrame frame{};
            frame.source = {reinterpret_cast<std::uintptr_t>(device.source), bounds, pitch * static_cast<std::ptrdiff_t>(sizeof(float)), static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
            frame.destination = frame.source;
            frame.destination.address = reinterpret_cast<std::uintptr_t>(device.destination);
            frame.render_window = bounds;
            frame.full_frame_extent = bounds;
            frame.stream = reinterpret_cast<std::uintptr_t>(device.stream);
            frame.flags = FJ_FRAME_STREAM_PRESENT;
            frame.pixel_size_um = 35'000.0f / static_cast<float>(width);
            frame.time_frames = test.time;
            frame.frame_rate = 24.0;
            frame.session_seed = 0x20260923;
            frame.clip_token = 0x5312;
            std::string diagnostic;
            if (!JuicerCudaTest::execute_prepared_state(runtime.borrow(), parameters, frame, 0x641207 + static_cast<int>(test.route), path == ExecutionPath::StateContract, path == ExecutionPath::StateLifetime, diagnostic, lifetimeCase))
                throw std::runtime_error(std::string(test.name) + ": held state: " + diagnostic);
        }
#endif
        else {
            std::array<char, 2048> stateText{};
            FjErrorBuffer stateError{stateText.data(), stateText.size(), 0};
            std::optional<JuicerTestSupport::PreparedStateHold> held;
            if (JuicerTestSupport::prepare_state(runtime.borrow().assets, parameters, held, stateError).category != FJ_STATUS_SUCCESS || !held)
                throw std::runtime_error(std::string(test.name) + ": state preparation failed");
            FjPreparedStateView stateView{};
            if (held->view(stateView, stateError).category != FJ_STATUS_SUCCESS)
                throw std::runtime_error(std::string(test.name) + ": state view failed");
            const FjRect bounds{test.originX, test.originY, test.originX + width, test.originY + height};
            const FjRect destinationBounds{bounds.x1 - border, bounds.y1 - border, bounds.x2 + border - (destinationLayout == DestinationLayout::Uncovered ? 1 : 0), bounds.y2 + border};
            const FjRect full = test.fullWidth > 0 && test.fullHeight > 0
                                    ? FjRect{0, 0, test.fullWidth, test.fullHeight}
                                    : bounds;
            const FjExtent fullExtent{full.x1, full.y1, full.x2 - full.x1, full.y2 - full.y1};
            FjPreparedFramePolicy policy{};
            if (held->frame_policy(fullExtent, policy, stateError).category != FJ_STATUS_SUCCESS)
                throw std::runtime_error(std::string(test.name) + ": frame policy failed");
            FjFrame frame{};
            frame.source = {reinterpret_cast<std::uintptr_t>(device.source), bounds, pitch * static_cast<std::ptrdiff_t>(sizeof(float)), static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
            frame.destination = {reinterpret_cast<std::uintptr_t>(device.destination), destinationBounds, destinationPitch * static_cast<std::ptrdiff_t>(sizeof(float)), static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
            frame.render_window = emptyWindow ? FjRect{bounds.x1, bounds.y1, bounds.x1, bounds.y1} : bounds;
            frame.full_frame_extent = full;
            frame.stream = reinterpret_cast<std::uintptr_t>(device.stream);
            frame.flags = FJ_FRAME_STREAM_PRESENT;
            frame.pixel_size_um = policy.geometry.pixel_size_um;
            frame.time_frames = test.time;
            frame.frame_rate = 24.0;
            frame.session_seed = 0x20260923;
            frame.clip_token = 0x5312;
            frame.effects_geometry = {{full.x1, full.y1, full.x2 - full.x1, full.y2 - full.y1},
                                      double(full.x1),
                                      double(full.y1),
                                      double(full.x2 - full.x1),
                                      double(full.y2 - full.y1),
                                      test.scaleX,
                                      test.scaleY,
                                      1.0};
            const auto token = 0x641207ull + static_cast<std::uint64_t>(test.route) + (test.combined ? 4ull : 0ull);
            std::string diagnostic;
            if (path == ExecutionPath::Boundary) {
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
                snapshot.instanceToken.value = token;
                snapshot.frameToken.value = static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(test.time)));
                snapshot.snapshotId = snapshotId;
                FjCudaContext context{};
                if (fj_cuda_inspect(runtime.borrow().cuda, &frame, &context, nullptr).category != FJ_STATUS_SUCCESS)
                    throw std::runtime_error("reference frame inspection failed");
                snapshot.deviceContextKey = {context.device_id, reinterpret_cast<void*>(context.context)};
                const auto frameBounds = JuicerCuda::FrameRect{bounds.x1, bounds.y1, bounds.x2, bounds.y2};
                const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    frameBounds, frameBounds, static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                    stateView.upload_core_hash, stateView.dir_recipe.hash, stateView.scanner_hash, meter.hash);
                if (!JuicerCudaTest::execute_boundary(runtime.borrow(), *held, frame, snapshot, diagnostic, std::string(test.name) + "/" + std::string(invocationPhase)))
                    throw std::runtime_error(std::string(test.name) + ": C boundary: " + diagnostic);
#else
                throw std::runtime_error("boundary path requires the prepared fixture target");
#endif
            } else if (availability != PublishedRequestAvailability::Complete) {
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
                const auto result = render_unavailable_case(runtime, held ? &*held : nullptr, frame, availability, state, diagnostic);
                if (result.fatal)
                    throw CallbackStatus{kOfxStatErrFatal};
                if (s_abortObservation && s_abortObservation->cancelAt != 0 && s_abortObservation->calls >= s_abortObservation->cancelAt)
                    return {};
#else
                throw std::runtime_error("unavailable request cases require the executor fixture target");
#endif
            } else {
                const auto rendered = render_held_prepared_state(runtime, *held, frame, token, {}, nullptr, nullptr, diagnostic, nullptr, nullptr, false, &state);
                if (rendered.outcome.status.category != FJ_STATUS_SUCCESS)
                    throw JuicerCuda::ExecutionFailure{{rendered.outcome.status, diagnostic}};
            }
        }
        require_cuda(cudaMemcpyAsync(output.data(), device.destination, destinationBytes, cudaMemcpyDeviceToHost, device.stream),
                     "copy destination");
        require_cuda(cudaStreamSynchronize(device.stream), "synchronize");
        if (emptyWindow) {
            for (float value : output) {
                if (std::bit_cast<std::uint32_t>(value) != std::bit_cast<std::uint32_t>(kCanary)) {
                    throw std::runtime_error("empty processor window wrote output");
                }
            }
            return {};
        }
        std::vector<float> pixels;
        pixels.reserve(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                       static_cast<std::size_t>(test.components));
        for (int y = 0; y < height; ++y) {
            const std::size_t rowOffset = static_cast<std::size_t>(y + border) * static_cast<std::size_t>(destinationPitch);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixelOffset = rowOffset +
                                                static_cast<std::size_t>(x + border) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    const std::size_t index = pixelOffset + static_cast<std::size_t>(c);
                    const float value = output[index];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(std::string(test.name) + ": nonfinite output");
                    }
                    if (c == 3) {
                        RenderAssertions::require_same_bits(std::string(test.name) + ": alpha", value, input[static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components) + static_cast<std::size_t>(c)]);
                    }
                    pixels.push_back(value);
                }
            }
        }
        for (int y = 0; y < destinationHeight; ++y) {
            for (int x = 0; x < destinationPitch; ++x) {
                const bool rendered = y >= border && y < border + height &&
                                      x >= border * test.components && x < (border + width) * test.components;
                if (!rendered) {
                    RenderAssertions::require_same_bits(std::string(test.name) + ": destination outside render window", output[static_cast<std::size_t>(y) * static_cast<std::size_t>(destinationPitch) + static_cast<std::size_t>(x)], kCanary);
                }
            }
        }
        return pixels;
    }

    using RenderAssertions::compare_pixels;

} // namespace

namespace {
    [[maybe_unused]] void check_cutover_callbacks(CallbackSession& runtime, const Case& test, const std::vector<float>& expected) {
        std::vector<std::string> intended;
        intended.reserve(12);
        for (const char* id : {"disjoint-rows", "preflight-keeps-submission", "dir-disabled-submission", "initial-uniform", "changed-geometry-time-scale", "sequential-supplied-stream", "worker-thread", "second-context", "throwing-cancellation", "render-after-cancellation"})
            intended.push_back(std::string(test.name) + "/" + id);
        for (const unsigned checkpoint : {1U, 2U, 3U, 4U, 5U})
            intended.push_back(std::string(test.name) + "/cancel/" + std::to_string(checkpoint));
        JuicerTests::Cohort rows(std::string("cutover_callback/") + test.name, intended);
        JuicerTestSupport::InstanceFixture state;
        MembershipRetirement stateRetirement{runtime, state};
        auto disjoint_row = rows.case_event(std::string(test.name) + "/disjoint-rows");
        compare_pixels("same allocation disjoint rows", render_case(runtime, test, parameters_for(test), state, false, ExecutionPath::Processor, DestinationLayout::DisjointRows), expected);
        disjoint_row.passed();
        const auto sameSubmission = [](const JuicerTestSupport::SubmissionObservation& left, const JuicerTestSupport::SubmissionObservation& right) {
            const auto& a = left.submission;
            const auto& b = right.submission;
            return left.valid && right.valid && left.context.device_id == right.context.device_id && left.context.context == right.context.context &&
                   a.instance_token == b.instance_token && a.frame_token == b.frame_token && a.submission_id == b.submission_id &&
                   a.upload_core_hash == b.upload_core_hash && a.dir_hash == b.dir_hash &&
                   a.scanner_hash == b.scanner_hash && a.auto_exposure_hash == b.auto_exposure_hash;
        };
        auto preflight_row = rows.case_event(std::string(test.name) + "/preflight-keeps-submission");
        const auto beforePreflight = runtime.submission(state);
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        fj_test_prepared_noise_observation_reset();
#endif
        bool preflightFatal = false;
        try {
            (void)render_case(runtime, test, parameters_for(test), state, false, ExecutionPath::Processor, DestinationLayout::Uncovered);
        } catch (const CallbackStatus& error) {
            preflightFatal = error.status() == kOfxStatErrFatal;
        }
        if (!preflightFatal || !sameSubmission(beforePreflight, runtime.submission(state)))
            throw std::runtime_error("preflight failure changed the existing submission");
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        FjPreparedNoiseObservation preflightObservation{};
        if (fj_test_prepared_noise_observation(&preflightObservation).category != FJ_STATUS_SUCCESS || preflightObservation.native_render_calls != 0)
            throw std::runtime_error("preflight failure reached native rendering");
#endif
        preflight_row.passed();
        auto dir_row = rows.case_event(std::string(test.name) + "/dir-disabled-submission");
        auto dirDisabled = parameters_for(test);
        dirDisabled.values.dir_couplers.active = false;
        DirSubmissionWitness firstDir{};
        DirSubmissionWitness repeatDir{};
        try {
            dirSubmissionWitness = &firstDir;
            const auto firstPixels = render_case(runtime, test, dirDisabled, state, false, ExecutionPath::Direct);
            dirSubmissionWitness = &repeatDir;
            const auto repeatPixels = render_case(runtime, test, dirDisabled, state, false, ExecutionPath::Direct);
            dirSubmissionWitness = nullptr;
            compare_pixels("DIR-disabled repeat", repeatPixels, firstPixels);
        } catch (...) {
            dirSubmissionWitness = nullptr;
            throw;
        }
        const auto validDir = [](const DirSubmissionWitness& witness) {
            const auto& selected = witness.selected;
            const auto& consumed = witness.consumed;
            return witness.completedDirHash == 0 && witness.selectedCalls == 1 && witness.renderCalls == 1 && witness.traceUnlocked &&
                   selected.upload_core_hash != 0 && selected.dir_hash == 1 && selected.scanner_hash != 0 && selected.auto_exposure_hash != 0 &&
                   selected.submission_id != 0 && consumed.submission_id == selected.submission_id &&
                   consumed.instance_token == selected.instance_token && consumed.frame_token == selected.frame_token &&
                   consumed.upload_core_hash == selected.upload_core_hash && consumed.dir_hash == selected.dir_hash &&
                   consumed.scanner_hash == selected.scanner_hash && consumed.auto_exposure_hash == selected.auto_exposure_hash;
        };
        if (!validDir(firstDir) || !validDir(repeatDir) || firstDir.selected.submission_id != repeatDir.selected.submission_id)
            throw std::runtime_error("DIR-disabled completed, selected, or native submission identity changed");
        dir_row.passed();
        Case uniform = test;
        uniform.uniform = true;
        auto uniformParameters = parameters_for(uniform);
        uniformParameters.values.glare_active = false;
        auto initial_row = rows.case_event(std::string(test.name) + "/initial-uniform");
        const auto initialUniform = render_case(runtime, uniform, uniformParameters, state);
        // This is a constant-input invariance test, complementary to the
        // independent accepted pixel fixtures above. Glare would add noise.
        const auto uniform_expected = [&](const Case& frame) {
            std::vector<float> pixels(static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height) * static_cast<std::size_t>(frame.components));
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                const auto channel = i % static_cast<std::size_t>(frame.components);
                // Alpha is positional even for uniform RGB in this fixture.
                const auto pixel = i / static_cast<std::size_t>(frame.components);
                const auto x = pixel % static_cast<std::size_t>(frame.width);
                const auto y = pixel / static_cast<std::size_t>(frame.width);
                pixels[i] = channel == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f : initialUniform[channel];
            }
            return pixels;
        };
        compare_pixels("initial uniform callback", initialUniform, uniform_expected(uniform));
        initial_row.passed();
        const auto firstId = runtime.submission(state).submission.submission_id;
        uniform.width += 2;
        uniform.height -= 2;
        uniform.originX = -11;
        uniform.originY = 7;
        uniform.time = -17.25;
        uniform.scaleX = 0.5;
        uniform.scaleY = 1.5;
        const auto changedExpected = uniform_expected(uniform);
        auto changed_row = rows.case_event(std::string(test.name) + "/changed-geometry-time-scale");
        compare_pixels("changed geometry time and scale", render_case(runtime, uniform, uniformParameters, state), changedExpected);
        const auto changedId = runtime.submission(state).submission.submission_id;
        if (changedId == firstId || runtime.submission(state).submission.frame_token != static_cast<std::uint64_t>(std::int64_t{-18})) {
            throw std::runtime_error("changed callback facts reused the previous latch");
        }
        changed_row.passed();
        // Each render_case allocates a new supplied stream; the same facts reuse the latch.
        auto supplied_stream_row = rows.case_event(std::string(test.name) + "/sequential-supplied-stream");
        compare_pixels("sequential supplied stream", render_case(runtime, uniform, uniformParameters, state), changedExpected);
        if (runtime.submission(state).submission.submission_id != changedId) {
            throw std::runtime_error("unchanged callback facts failed to reuse the latch");
        }
        supplied_stream_row.passed();
        CUcontext current = nullptr;
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
            throw std::runtime_error("get callback context");
        }
        std::exception_ptr workerFailure;
        auto worker_row = rows.case_event(std::string(test.name) + "/worker-thread");
        std::thread worker([&] {
            try {
                if (cuCtxSetCurrent(current) != CUDA_SUCCESS) {
                    throw std::runtime_error("set worker callback context");
                }
                compare_pixels("callback on another host thread", render_case(runtime, uniform, uniformParameters, state), changedExpected);
            } catch (...) {
                workerFailure = std::current_exception();
            }
            (void)cuCtxSetCurrent(nullptr);
        });
        worker.join();
        if (workerFailure) {
            std::rethrow_exception(workerFailure);
        }
        worker_row.passed();
        CUcontext other = nullptr;
        if (cuCtxCreate(&other, nullptr, 0, 0) != CUDA_SUCCESS) {
            throw std::runtime_error("create fixture context");
        }
        auto context_row = rows.case_event(std::string(test.name) + "/second-context");
        try {
            compare_pixels("second context on same device", render_case(runtime, uniform, uniformParameters, state), changedExpected);
            if (reinterpret_cast<void*>(runtime.submission(state).context.context) != other) {
                throw std::runtime_error("callback retained prior context identity");
            }
            std::string diagnostic;
            JuicerCuda::NativeCall admission(runtime.borrow().cuda);
            if (!JuicerCuda::borrow_root(runtime.borrow().cuda).retire_idle_context(0, other, diagnostic)) {
                throw std::runtime_error(diagnostic);
            }
        } catch (...) {
            (void)cuCtxDestroy(other);
            (void)cuCtxSetCurrent(current);
            throw;
        }
        if (cuCtxDestroy(other) != CUDA_SUCCESS || cuCtxSetCurrent(current) != CUDA_SUCCESS) {
            throw std::runtime_error("release fixture context");
        }
        context_row.passed();
        JuicerTestSupport::SubmissionObservation selectedOnCancel{};
        for (const unsigned checkpoint : {1U, 2U, 3U, 4U, 5U}) {
            const auto beforeCancel = runtime.submission(state);
            auto cancel_row = rows.case_event(std::string(test.name) + "/cancel/" + std::to_string(checkpoint));
            AbortObservation observation;
            observation.cancelAt = checkpoint;
            s_abortObservation = &observation;
            try {
                (void)render_case(runtime, test, parameters_for(test), state);
            } catch (...) {
                s_abortObservation = nullptr;
                throw;
            }
            s_abortObservation = nullptr;
            if (observation.calls != checkpoint || !observation.sameThread) {
                throw std::runtime_error("production abort delivery or callback lifetime changed");
            }
            const auto afterCancel = runtime.submission(state);
            if (checkpoint <= 2 && !sameSubmission(beforeCancel, afterCancel))
                throw std::runtime_error("early cancellation changed the existing submission");
            if (checkpoint == 3)
                selectedOnCancel = afterCancel;
            if (checkpoint >= 3 && (!afterCancel.valid || (checkpoint > 3 && !sameSubmission(selectedOnCancel, afterCancel))))
                throw std::runtime_error("post-selection cancellation failed to commit or reuse the submission");
            cancel_row.passed();
        }
        auto throwing_row = rows.case_event(std::string(test.name) + "/throwing-cancellation");
        AbortObservation throwing;
        throwing.throwAt = 4;
        s_abortObservation = &throwing;
        bool fatal = false;
        try {
            (void)render_case(runtime, test, parameters_for(test), state);
        } catch (const CallbackStatus& error) {
            fatal = error.status() == kOfxStatErrFatal;
        } catch (...) {
            s_abortObservation = nullptr;
            throw;
        }
        s_abortObservation = nullptr;
        if (!fatal || throwing.calls != 4 || !throwing.sameThread ||
            !sameSubmission(selectedOnCancel, runtime.submission(state))) {
            throw std::runtime_error("abort suite exception escaped or became successful cancellation");
        }
        throwing_row.passed();
        auto recovery_row = rows.case_event(std::string(test.name) + "/render-after-cancellation");
        compare_pixels("render after cancellation", render_case(runtime, test, parameters_for(test), state), expected);
        if (!sameSubmission(selectedOnCancel, runtime.submission(state)))
            throw std::runtime_error("render after cancellation did not reuse the committed submission");
        // This fixture's next case reuses its deterministic instance token with
        // a fresh counter. Match real instance teardown before that reuse.
        if (!stateRetirement.close()) {
            throw std::runtime_error("retire cutover fixture instance");
        }
        recovery_row.passed();
        std::cout << test.name << ": production geometry/time/scale, streams, threads, contexts, disjoint alias and abort contracts passed\n";
    }
} // namespace

namespace {
    [[maybe_unused]] void check_physical_effects_pixels(CallbackSession& runtime, const Case& original) {
        Case test = original;
        test.components = 4;
        // A fresh fixture state restarts snapshot IDs; retire the preceding instance first.
        const std::uint64_t instanceToken = 0x641207 + static_cast<int>(test.route) + (test.combined ? 4 : 0);
        if (fj_cuda_retire_instance(runtime.borrow().cuda, instanceToken, nullptr).category != FJ_STATUS_SUCCESS)
            throw std::runtime_error("retire instance before physical fixture");
        JuicerTestSupport::InstanceFixture state;
        MembershipRetirement stateRetirement{runtime, state};
        auto off = parameters_for(test);
        off.values.grain.active = false;
        off.values.film_dust_amount = 0;
        off.values.film_scratch_amount = 0;
        off.values.gate_dust_amount = 0;
        off.values.gate_scratch_amount = 0;
        off.values.gate_weave_amount = 0;
        const auto same_bits = [](const std::vector<float>& left, const std::vector<float>& right) {
            return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(), [](float a, float b) {
                       return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
                   });
        };
        const auto baseline = render_case(runtime, test, off, state);
        if (!same_bits(render_case(runtime, test, off, state), baseline))
            throw std::runtime_error("all-off physical effects is not deterministic identity");
        const std::array<const char*, 5> familyNames{"film-dust", "film-scratch", "gate-dust", "gate-scratch", "gate-weave"};
        std::vector<std::string> intended;
        intended.reserve(5);
        for (const char* family : familyNames)
            intended.push_back(std::string(test.name) + "/" + family);
        JuicerTests::Cohort rows(std::string("physical_effect_pixels/") + test.name, intended);
        for (unsigned family = 0; family < 5; ++family) {
            auto case_event = rows.case_event(std::string(test.name) + "/" + familyNames[family]);
            auto enabled = off;
            if (family == 0)
                enabled.values.film_dust_amount = 10;
            if (family == 1)
                enabled.values.film_scratch_amount = 10;
            if (family == 2)
                enabled.values.gate_dust_amount = 10;
            if (family == 3)
                enabled.values.gate_scratch_amount = 10;
            if (family == 4)
                enabled.values.gate_weave_amount = 10;
            const auto pixels = render_case(runtime, test, enabled, state);
            if (same_bits(pixels, baseline))
                throw std::runtime_error("physical effects family did not reach CUDA output: " + std::to_string(family));
            enabled.values.grain.debug_view = 3;
            if (!same_bits(render_case(runtime, test, enabled, state), pixels))
                throw std::runtime_error("inactive grain debug incorrectly suppresses physical effects");
            case_event.passed();
        }
        auto debug = off;
        debug.values.grain.active = true;
        debug.values.grain.debug_view = 3;
        const auto debugOff = render_case(runtime, test, debug, state);
        debug.values.film_dust_amount = 10;
        debug.values.film_scratch_amount = 10;
        debug.values.gate_dust_amount = 10;
        debug.values.gate_scratch_amount = 10;
        debug.values.gate_weave_amount = 10;
        if (!same_bits(render_case(runtime, test, debug, state), debugOff))
            throw std::runtime_error("completed grain debug did not suppress physical effects");
        if (!stateRetirement.close())
            throw std::runtime_error("retire instance after physical fixture");
        std::cout << test.name << " physical families/off/debug/alpha CUDA paths passed\n";
    }

} // namespace

namespace {
    using NativeFrameDescriptorCall = FjStatus (*)(void*, const FjFrameMathInput*, FjNativeFrameDescriptors*, FjErrorBuffer*);

    struct FrameDescriptorObservation {
        NativeFrameDescriptorCall forward = nullptr;
        unsigned callbackCalls = 0;
        unsigned providerCalls = 0;
    };

    thread_local FrameDescriptorObservation* frameDescriptorObservation = nullptr;

    FjStatus observe_frame_descriptors(
        void* scopeUser,
        const FjFrameMathInput* input,
        FjNativeFrameDescriptors* out,
        FjErrorBuffer* error) noexcept {
        if (out)
            *out = {};
        auto* const override = frameDescriptorObservation;
        if (!override || !override->forward)
            return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        ++override->callbackCalls;
        ++override->providerCalls;
        return override->forward(scopeUser, input, out, error);
    }

    struct ScopedFrameDescriptorObservation {
        explicit ScopedFrameDescriptorObservation(FrameDescriptorObservation* current) noexcept
            : _previous(std::exchange(frameDescriptorObservation, current)) {}
        ~ScopedFrameDescriptorObservation() {
            frameDescriptorObservation = _previous;
        }
        ScopedFrameDescriptorObservation(const ScopedFrameDescriptorObservation&) = delete;
        ScopedFrameDescriptorObservation& operator=(const ScopedFrameDescriptorObservation&) = delete;

    private:
        FrameDescriptorObservation* _previous;
    };

    struct PreparedRenderCompletion {
        unsigned calls = 0;
        bool retirementFailed = false;
        bool gateEntered = false;
        std::optional<JuicerCuda::NativeCall>* gate = nullptr;
        FjCuda* gateCuda = nullptr;
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        bool observationRead = false;
        FjPreparedNoiseObservation noise{};
#endif
    };

    FjCallCompletionResult prepared_render_complete(void* user, const FjCallCompletionInput* input) noexcept {
        auto* const completion = static_cast<PreparedRenderCompletion*>(user);
        if (completion) {
            ++completion->calls;
            if (completion->gate) {
                try {
                    completion->gate->emplace(completion->gateCuda);
                    completion->gateEntered = true;
                } catch (...) {
                    completion->gateEntered = false;
                }
            }
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            completion->observationRead =
                fj_test_prepared_noise_observation(&completion->noise).category == FJ_STATUS_SUCCESS;
#endif
        }
        // Rust owns the fatal decision, including failures before native rendering.
        return input ? FjCallCompletionResult{input->outcome, input->force_fatal}
                     : FjCallCompletionResult{{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
    }

    struct PreparedSubmissionObservation {
        FjSubmission expected{};
        unsigned calls = 0;
        bool matched = true;
    };

    thread_local PreparedSubmissionObservation* preparedSubmissionObservation = nullptr;

    void observe_prepared_submission(void*, const FjSubmission* selected, std::uint32_t) noexcept {
        auto* observation = preparedSubmissionObservation;
        if (!observation || !selected)
            return;
        ++observation->calls;
        if (dirSubmissionWitness) {
            ++dirSubmissionWitness->selectedCalls;
            dirSubmissionWitness->selected = *selected;
            std::uint32_t valid = 0;
            FjSubmission observed{};
            FjCudaContext context{};
            const auto status = dirSubmissionWitness->effect
                                    ? fj_test_ofx_instance_submission(dirSubmissionWitness->effect, &valid, &observed, &context, nullptr)
                                    : fj_test_instance_submission(dirSubmissionWitness->instance, &valid, &observed, &context, nullptr);
            dirSubmissionWitness->traceUnlocked = status.category == FJ_STATUS_SUCCESS && valid != 0 &&
                                                  observed.submission_id == selected->submission_id && observed.instance_token == selected->instance_token &&
                                                  observed.frame_token == selected->frame_token && observed.upload_core_hash == selected->upload_core_hash &&
                                                  observed.dir_hash == selected->dir_hash && observed.scanner_hash == selected->scanner_hash &&
                                                  observed.auto_exposure_hash == selected->auto_exposure_hash;
        }
        const auto& expected = observation->expected;
        observation->matched = selected->instance_token == expected.instance_token &&
                               selected->frame_token == expected.frame_token && selected->submission_id != 0 &&
                               selected->upload_core_hash == expected.upload_core_hash && selected->dir_hash == expected.dir_hash &&
                               selected->scanner_hash == expected.scanner_hash && selected->auto_exposure_hash == expected.auto_exposure_hash;
    }

    FjRenderOutcome observe_native_submission(void* user, const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, const FjHostPreparation* host, FjAbortCallback abort, FjErrorBuffer* error) noexcept {
        auto* witness = dirSubmissionWitness;
        if (!witness || !witness->forward || !submission)
            return {{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0};
        ++witness->renderCalls;
        witness->consumed = *submission;
        return witness->forward(user, context, frame, submission, prepared, host, abort, error);
    }

    FjCallCompletionResult render_held_prepared_state(
        CallbackSession& runtime,
        const JuicerTestSupport::PreparedStateHold& state,
        const FjFrame& frame,
        std::uint64_t instanceToken,
        FjAbortCallback abort,
        FrameDescriptorObservation* descriptorObservation,
        bool* recoveryPending,
        std::string& diagnostic,
        const FjAssets* assetOwner,
        PreparedRenderCompletion* completionObservation,
        bool missingInstance,
        JuicerTestSupport::InstanceFixture* sharedSubmissionOwner) {
        diagnostic.clear();
        const FjRuntimeView runtimeView = runtime.borrow();
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjPreparedStateView view{};
        const FjStatus viewed = state.view(view, error);
        if (viewed.category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(text.data(), error.length);
            return {{viewed, 0}, 0};
        }
        if (dirSubmissionWitness)
            dirSubmissionWitness->completedDirHash = view.dir_recipe.hash;
        const FjExtent full{frame.full_frame_extent.x1,
                            frame.full_frame_extent.y1,
                            frame.full_frame_extent.x2 - frame.full_frame_extent.x1,
                            frame.full_frame_extent.y2 - frame.full_frame_extent.y1};
        FjPreparedFramePolicy policy{};
        error.length = 0;
        const FjStatus policyStatus = state.frame_policy(full, policy, error);
        if (policyStatus.category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(text.data(), error.length);
            return {{policyStatus, 0}, 0};
        }
        const JuicerCuda::FrameRect frameRect{frame.source.bounds.x1,
                                              frame.source.bounds.y1,
                                              frame.source.bounds.x2,
                                              frame.source.bounds.y2};
        const auto preview = JuicerCuda::make_auto_exposure_preview_descriptor(
            frameRect, frameRect, static_cast<Spektrafilm::AutoExposureMethod>(view.input_policy.meter));
        const auto normalized = JuicerCuda::ResourceManager::make_key_digests(
            view.upload_core_hash, view.dir_recipe.hash, view.scanner_hash, preview.hash);
        PreparedSubmissionObservation selection{};
        selection.expected = {instanceToken,
                              static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(frame.time_frames))),
                              1,
                              normalized.uploadCoreHash,
                              normalized.dirHash,
                              normalized.scannerHash,
                              normalized.autoExposureHash};
        PreparedRenderCompletion completion{};
        if (completionObservation) {
            completion.gate = completionObservation->gate;
            completion.gateCuda = completionObservation->gateCuda;
        }
        JuicerCuda::CallScope activeScope(runtimeView.cuda, view.route, &prepared_render_complete, &completion);
        FjCudaCallAccess access = activeScope.access();
        if (descriptorObservation) {
            descriptorObservation->forward = access.frame_descriptors;
            access.frame_descriptors = &observe_frame_descriptors;
        }
        ScopedSubmissionOwner localSubmissionOwner;
        JuicerTestSupport::InstanceFixture& submissionOwner = sharedSubmissionOwner ? *sharedSubmissionOwner : localSubmissionOwner.instance;
        if (!sharedSubmissionOwner && submissionOwner.bind_identity(frame.session_seed, instanceToken, error).category != FJ_STATUS_SUCCESS) {
            diagnostic = "prepared submission identity bind failed";
            return {{{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        access.trace_submission = &observe_prepared_submission;
        auto* const callbackEffect = sharedSubmissionOwner ? runtime.find(*sharedSubmissionOwner) : nullptr;
        if (dirSubmissionWitness) {
            dirSubmissionWitness->effect = callbackEffect;
            dirSubmissionWitness->instance = callbackEffect ? nullptr : submissionOwner.borrow();
            dirSubmissionWitness->forward = access.render;
            access.render = &observe_native_submission;
        }
        FjCallCompletionResult result{};
        auto admission = JuicerCuda::borrow_root(runtimeView.cuda).begin_frame_preparation();
        if (!admission.active()) {
            diagnostic = "prepared frame render admission unavailable";
            return {{{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        const FjAssets* const renderAssets = assetOwner ? assetOwner : runtimeView.assets;
        {
            ScopedFrameDescriptorObservation overrideScope(descriptorObservation);
            preparedSubmissionObservation = &selection;
            result = missingInstance
                         ? state.reject_missing_instance(renderAssets, access, frame, abort, error)
                     : callbackEffect
                         ? fj_test_ofx_prepared_state_render(callbackEffect, state.borrow(), access, &frame, &policy, selection.expected.frame_token, abort, &error)
                         : state.render(renderAssets, access, frame, policy, submissionOwner.borrow(), selection.expected.frame_token, abort, error);
            preparedSubmissionObservation = nullptr;
        }
        if (!sharedSubmissionOwner && selection.calls != 0) {
            localSubmissionOwner.cuda = runtimeView.cuda;
            localSubmissionOwner.token = instanceToken;
        }
        if (completion.calls != 1 || selection.calls > 1 || !selection.matched) {
            diagnostic = activeScope.diagnostic();
            if (diagnostic.empty() && error.length != 0)
                diagnostic.assign(text.data(), error.length);
            diagnostic = "prepared frame completion=" + std::to_string(completion.calls) +
                         " submission=" + std::to_string(selection.calls) +
                         " matched=" + std::to_string(selection.matched) +
                         " status=" + std::to_string(result.outcome.status.category) +
                         " detail=" + diagnostic;
            return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        if (recoveryPending)
            *recoveryPending = activeScope.recovery().pending;
        diagnostic = activeScope.diagnostic();
        if (diagnostic.empty() && error.length != 0)
            diagnostic.assign(text.data(), error.length);
        completion.retirementFailed = !sharedSubmissionOwner && !localSubmissionOwner.close();
        if (completionObservation)
            *completionObservation = completion;
        if (completion.retirementFailed) {
            if (result.outcome.status.category == FJ_STATUS_SUCCESS && result.fatal == 0) {
                diagnostic = "prepared submission owner retirement failed";
                return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
            }
            if (!completionObservation)
                (void)std::fputs("secondary failure: prepared submission owner retirement failed\n", stderr);
        }
        return result;
    }
} // namespace

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
namespace {
    using RawFrameMathBuilder = FjStatus (*)(void*, FjFrameMathInput&, FjErrorBuffer*) noexcept;

    struct RawFrameJson {
        const nlohmann::json* input = nullptr;
        unsigned facadeCalls = 0;
        FjDirFrameFailure dirFailure{};
    };

    FjStatus build_literal_dir_math(void* user, FjFrameMathInput& math, FjErrorBuffer* error) noexcept {
        try {
            auto& context = *static_cast<RawFrameJson*>(user);
            const auto& value = *context.input;
            const auto extent = [](const nlohmann::json& v) -> FjExtent {
                return {v.at(0), v.at(1), v.at(2), v.at(3)};
            };
            const FjDirFrameInput raw{value.at("active").get<std::uint32_t>(),
                                      std::bit_cast<float>(value.at("size").get<std::uint32_t>()),
                                      std::bit_cast<float>(value.at("tail").get<std::uint32_t>()),
                                      std::bit_cast<float>(value.at("weight").get<std::uint32_t>()),
                                      std::bit_cast<float>(value.at("pixel").get<std::uint32_t>()),
                                      extent(value.at("render")),
                                      extent(value.at("full")),
                                      value.at("hash").get<std::uint64_t>()};
            math.dir = {};
            ++context.facadeCalls;
            return fj_test_dir_frame_prepare(&raw, &math.dir, &context.dirFailure, error);
        } catch (...) {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }
    }

    FjStatus build_literal_grain_math(void* user, FjFrameMathInput& math, FjErrorBuffer* error) noexcept {
        try {
            auto& context = *static_cast<RawFrameJson*>(user);
            const auto& value = *context.input;
            const auto input = JuicerTestSupport::grain_frame_input(value);
            const FjGrainFrameInput raw = input.bind();
            FjGrainFrameFailure failure{};
            math.grain = {};
            math.grain_coefficients = input.coefficients;
            ++context.facadeCalls;
            return fj_test_grain_frame_prepare(&raw, &math.grain, &failure, error);
        } catch (...) {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }
    }

    FjStatus build_literal_effects_math(void* user, FjFrameMathInput& math, FjErrorBuffer* error) noexcept {
        try {
            auto& context = *static_cast<RawFrameJson*>(user);
            const auto& value = *context.input;
            Spektrafilm::FilmJuicerEffectsRecipe recipe{};
            const auto input = JuicerTestSupport::effects_frame_input(value, recipe);
            FjEffectsFrameInput raw{};
            std::string diagnostic;
            if (!JuicerEffects::project_input(input, raw, diagnostic))
                return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
            FjEffectsFrameFailure failure{};
            math.effects = {};
            math.film_dust = raw.film_dust;
            math.film_scratch = raw.film_scratch;
            math.gate_dust = raw.gate_dust;
            math.gate_scratch = raw.gate_scratch;
            ++context.facadeCalls;
            return fj_test_effects_frame_prepare(&raw, &math.effects, &failure, error);
        } catch (...) {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }
    }

    enum class FrameObservationKind : std::uint8_t {
        Dir,
        Grain,
        Effects
    };

    struct FrameConstructionResult {
        explicit FrameConstructionResult(FrameObservationKind observed)
            : kind(observed) {}

        FrameObservationKind kind;
        FjStatus status{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        unsigned rawFacadeCalls = 0;
        std::string diagnostic;
        std::optional<FjDirFrameFailure> dirFailure;
        std::optional<Spektrafilm::SpatialDirDescriptor> dir;
        std::optional<Spektrafilm::VisualGrainFrameDescriptor> grain;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effects;
    };

    bool capture_held_frame_construction(
        void* user,
        const JuicerCudaTest::PreparedFrameFixture::Inputs& inputs,
        std::string&) {
        auto& result = *static_cast<FrameConstructionResult*>(user);
        result.status = {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        if (result.kind == FrameObservationKind::Dir)
            result.dir = inputs.descriptors.spatialDir;
        else if (result.kind == FrameObservationKind::Grain)
            result.grain = inputs.descriptors.grain;
        else
            result.effects = inputs.descriptors.effects;
        return true;
    }

    FrameConstructionResult prepare_literal_frame_construction(
        FrameObservationKind kind,
        const nlohmann::json& value,
        Spektrafilm::ScanRoute route,
        float pixelSizeUm,
        RawFrameMathBuilder build) {
        FrameConstructionResult result{kind};
        FjFrameMathInput math{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        RawFrameJson raw{&value};
        result.status = build(&raw, math, &error);
        result.rawFacadeCalls = raw.facadeCalls;
        if (kind == FrameObservationKind::Dir)
            result.dirFailure = raw.dirFailure;
        if (result.status.category != FJ_STATUS_SUCCESS) {
            result.diagnostic.assign(text.data(), std::min(error.length, text.size()));
            return result;
        }
        if (kind == FrameObservationKind::Dir) {
            Spektrafilm::SpatialDirDescriptor descriptor{};
            if (!Spektrafilm::compose_spatial_dir_descriptor(
                    math.dir, pixelSizeUm, Spektrafilm::scan_route_is_print(route) ? "print" : "direct", descriptor)) {
                result.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
                result.diagnostic = "literal DIR native descriptor composition failed";
                return result;
            }
            result.dir = descriptor;
        } else if (kind == FrameObservationKind::Grain) {
            Spektrafilm::VisualGrainFrameDescriptor descriptor{};
            if (!Spektrafilm::compose_visual_grain_frame_descriptor(
                    math.grain_coefficients,
                    static_cast<Spektrafilm::ProfilePolarity>(math.grain.capture_polarity),
                    math.grain,
                    descriptor,
                    result.diagnostic)) {
                result.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
                return result;
            }
            result.grain = descriptor;
        } else {
            Spektrafilm::FilmJuicerEffectsFrameDescriptor descriptor{};
            if (!JuicerEffects::compose_frame_descriptor(
                    math.effects, math.film_dust, math.film_scratch, math.gate_dust, math.gate_scratch, descriptor, result.diagnostic)) {
                result.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
                return result;
            }
            result.effects = descriptor;
        }
        return result;
    }

    FrameConstructionResult capture_prepared_frame_descriptor(
        CallbackSession& runtime,
        const JuicerTestSupport::PreparedStateHold& state,
        FrameObservationKind kind,
        const JuicerCudaTest::PreparedFrameFixture::FrameFacts& facts) {
        FrameConstructionResult result{kind};
        std::string diagnostic;
        if (!JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                runtime.borrow().assets, state, facts, &result, &capture_held_frame_construction, diagnostic))
            throw std::runtime_error("same-facts prepared descriptor construction: " + diagnostic);
        return result;
    }

    float prepared_frame_pixel_size_um(const JuicerTestSupport::PreparedStateHold& state, FjExtent fullFrame) {
        FjPreparedFramePolicy policy{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        if (state.frame_policy(fullFrame, policy, error).category != FJ_STATUS_SUCCESS)
            throw std::runtime_error("state-derived frame policy: " + std::string(text.data(), error.length));
        return policy.geometry.pixel_size_um;
    }

    struct AbortAfterNativeDescriptor {
        unsigned calls = 0;
        unsigned requestOn = 3;
    };

    std::uint32_t abort_after_native_descriptor(void* user) noexcept {
        auto* const state = static_cast<AbortAfterNativeDescriptor*>(user);
        if (!state)
            return 2u; // Invalid callback state must fail abort transport.
        ++state->calls;
        return state->calls >= state->requestOn ? FJ_ABORT_REQUESTED : FJ_ABORT_CONTINUE;
    }

    FjCallCompletionResult render_missing_prepared_state(
        CallbackSession& runtime,
        const FjAssets* assets,
        const FjFrame& frame,
        FjAbortCallback abort,
        PreparedRenderCompletion& completion,
        std::string& diagnostic) {
        diagnostic.clear();
        const FjRuntimeView runtimeView = runtime.borrow();
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        JuicerCuda::CallScope activeScope(runtimeView.cuda, std::nullopt, &prepared_render_complete, &completion);
        const auto admission = JuicerCuda::borrow_root(runtimeView.cuda).begin_frame_preparation();
        if (!admission.active()) {
            diagnostic = "prepared missing-state render admission unavailable";
            return {{{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        const auto result = JuicerTestSupport::reject_missing_render_state(assets, activeScope.access(), frame, abort, error);
        diagnostic = activeScope.diagnostic();
        if (diagnostic.empty() && error.length != 0) {
            diagnostic.assign(text.data(), error.length);
        }
        return result;
    }

    FjCallCompletionResult render_unavailable_case(CallbackSession& runtime,
                                                   const JuicerTestSupport::PreparedStateHold* held,
                                                   const FjFrame& frame,
                                                   PublishedRequestAvailability availability,
                                                   JuicerTestSupport::InstanceFixture& identity,
                                                   std::string& diagnostic) {
        const FjAbortCallback abort{[](void*) noexcept -> std::uint32_t {
                                        try {
                                            return effect_abort(nullptr) ? FJ_ABORT_REQUESTED : FJ_ABORT_CONTINUE;
                                        } catch (...) {
                                            return 2;
                                        }
                                    },
                                    nullptr};
        if (availability == PublishedRequestAvailability::MissingState ||
            availability == PublishedRequestAvailability::MissingStateUninitialized) {
            PreparedRenderCompletion completion{};
            return render_missing_prepared_state(runtime, runtime.borrow().assets, frame, abort, completion, diagnostic);
        }
        if (!held)
            return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        FjPreparedStateView view{};
        FjErrorBuffer error{nullptr, 0, 0};
        if (held->view(view, error).category != FJ_STATUS_SUCCESS)
            return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        return render_held_prepared_state(runtime, *held, frame, 0x641207ull + static_cast<std::uint64_t>(view.route), abort, nullptr, nullptr, diagnostic, nullptr, nullptr, availability == PublishedRequestAvailability::MissingInstance, &identity);
    }

    FjFrame prepared_raw_frame(const JuicerCuda::ExecutionFrame& source) {
        const auto rect = [](const JuicerCuda::FrameRect& value) {
            return FjRect{value.x1, value.y1, value.x2, value.y2};
        };
        FjFrame out{};
        out.source = {reinterpret_cast<std::uintptr_t>(source.source), rect(source.sourceBounds), source.sourceRowBytes, static_cast<std::uint32_t>(source.components), FJ_DEPTH_FLOAT32};
        out.destination = {reinterpret_cast<std::uintptr_t>(source.destination), rect(source.sourceBounds), source.destinationRowBytes, static_cast<std::uint32_t>(source.components), FJ_DEPTH_FLOAT32};
        out.render_window = rect(source.renderWindow);
        out.full_frame_extent = rect(source.fullFrameExtent);
        out.stream = reinterpret_cast<std::uintptr_t>(source.stream);
        out.flags = FJ_FRAME_STREAM_PRESENT;
        out.pixel_size_um = source.pixelSizeUm;
        out.time_frames = source.timeFrames;
        out.frame_rate = source.frameRate;
        out.session_seed = source.sessionSeed;
        out.clip_token = static_cast<std::uint64_t>(source.clipToken);
        const auto& geometry = source.effectsGeometry;
        out.effects_geometry = {{geometry.pixelDefinition.x, geometry.pixelDefinition.y, geometry.pixelDefinition.width, geometry.pixelDefinition.height},
                                geometry.canonicalX,
                                geometry.canonicalY,
                                geometry.canonicalWidth,
                                geometry.canonicalHeight,
                                geometry.scaleX,
                                geometry.scaleY,
                                geometry.pixelAspectRatio};
        return out;
    }

    struct WorkspacePreparedFrameContext {
        JuicerProcess::Root* root = nullptr;
        JuicerCuda::ResourceManager::DeviceContextKey deviceContextKey{};
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{};
        void* stream = nullptr;
        JuicerProcess::Root::PreparedCudaFrame* frame = nullptr;
        JuicerCuda::Failure* failure = nullptr;
        bool invalidateGrainExtent = false;
        bool invalidateDirRoles = false;
        bool invalidateGateWidth = false;
        bool omitGrain = false;
        bool omitNoise = false;
        bool omitDir = false;
        bool dirWasActive = false;
        bool grainWasValid = false;
        bool gateWasActive = false;
        bool cameraDiffusionWasPresent = false;
    };

    bool prepare_workspace_frame_from_state(
        void* user,
        const JuicerCudaTest::PreparedFrameFixture::Inputs& inputs,
        std::string&) {
        auto& context = *static_cast<WorkspacePreparedFrameContext*>(user);
        if (!context.frame || !context.failure)
            return false;
        context.dirWasActive = inputs.descriptors.spatialDir.hash != 0;
        context.grainWasValid = inputs.preparation.visualGrainDescriptor.has_value() &&
                                inputs.preparation.visualGrainDescriptor->hash != 0;
        context.gateWasActive = inputs.preparation.effectsDescriptor.has_value() &&
                                inputs.preparation.effectsDescriptor->gateTransmittanceActive &&
                                inputs.preparation.effectsDescriptor->hash != 0;
        context.cameraDiffusionWasPresent = inputs.descriptors.diffusion.has_value() &&
                                            inputs.descriptors.diffusion->camera.has_value();
        auto request = inputs.preparation;
        std::optional<Spektrafilm::SpatialDirDescriptor> dir;
        std::optional<Spektrafilm::VisualGrainFrameDescriptor> grain;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effects;
        if (context.invalidateDirRoles && request.spatialDirDescriptor) {
            dir = *request.spatialDirDescriptor;
            dir->scratchTier = Spektrafilm::DirScratchTier::Tier1F;
            dir->planeRoles = {3, 3, 3, 0};
            request.spatialDirDescriptor = &*dir;
        }
        if (context.invalidateGrainExtent && request.visualGrainDescriptor) {
            grain = *request.visualGrainDescriptor;
            ++grain->renderExtent.width;
            request.visualGrainDescriptor = *grain;
        }
        if (context.invalidateGateWidth && request.effectsDescriptor) {
            effects = *request.effectsDescriptor;
            effects->gateWidth = 0;
            request.effectsDescriptor = *effects;
        }
        if (context.omitGrain)
            request.visualGrainDescriptor.reset();
        if (context.omitNoise)
            request.noise = nullptr;
        if (context.omitDir)
            request.spatialDirDescriptor = nullptr;
        context.snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            inputs.state.upload_core_hash,
            inputs.state.dir_recipe.hash,
            inputs.state.scanner_hash,
            0);
        *context.frame = context.root->prepare_cuda_frame(
            context.deviceContextKey,
            context.snapshot,
            request,
            {},
            context.stream,
            *context.failure,
            inputs.hostPreparation);
        return true;
    }

    std::vector<std::string> workspace_candidate_ids() {
        std::vector<std::string> ids;
        ids.reserve(50);
        constexpr std::array<const char*, 4> routes{"negative-direct", "negative-print", "positive-direct", "positive-print"};
        for (unsigned route = 0; route < routes.size(); ++route) {
            const std::string prefix = std::string(routes[route]) + "/";
            ids.push_back(prefix + "actual-schedule");
            ids.push_back(prefix + "no-work");
            ids.push_back(prefix + "dir-unsharp");
            if (route < 2) {
                for (unsigned reject = 0; reject <= 6; ++reject)
                    ids.push_back(prefix + "strict-yvv/capacity/" + std::to_string(reject));
                ids.push_back(prefix + "strict-yvv/invalid-value");
                ids.push_back(prefix + "strict-yvv/context-loss");
                for (unsigned shape = 0; shape < 4; ++shape)
                    ids.push_back(prefix + "strict-yvv/shape/" + std::to_string(shape));
                ids.push_back(prefix + "strict-yvv/debug-view");
                ids.push_back(prefix + "strict-yvv/camera-diffusion");
            } else {
                ids.push_back(prefix + "strict-yvv/positive-storage");
            }
            ids.push_back(prefix + "physical-effects");
        }
        ids.push_back("halation-carrier/dedicated");
        ids.push_back("halation-carrier/camera-diffusion");
        return ids;
    }

    void run_workspace_transitions(CallbackSession& runtime) {
        using Frame = JuicerProcess::Root::PreparedCudaFrame;
        using namespace JuicerCuda::ResourceManager;
        const auto require = [](bool ok, const std::string& message) {
            if (!ok) {
                throw std::runtime_error(message);
            }
        };
        int deviceId = 0;
        CUcontext context = nullptr;
        require_cuda(cudaGetDevice(&deviceId), "workspace device");
        require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context, "workspace context");
        const DeviceContextKey key{deviceId, context};
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "workspace stream");
        std::uint64_t next = 1;
        std::vector<std::string> intended;
        intended.reserve(10);
        for (const char* route : {"negative-direct", "negative-print"}) {
            for (const char* mode : {"competing", "malformed"})
                intended.push_back(std::string(route) + "/" + mode);
            for (const char* carrier : {"disabled", "dedicated", "camera-diffusion"})
                intended.push_back(std::string(route) + "/" + carrier);
        }
        JuicerTests::Cohort transition_rows("workspace_transition", intended);
        const auto candidate_ids = workspace_candidate_ids();
        JuicerTests::Cohort candidate_rows("workspace_candidate", candidate_ids);
        unsigned routeIndex = 0;
        for (const auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan}) {
            const std::string routeName(routeIndex++ == 0 ? "negative-direct" : "negative-print");
            const bool print = Spektrafilm::scan_route_is_print(route);
            JuicerTestSupport::PreparationInput params;
            params.values.scan_route = static_cast<std::uint32_t>(route);
            params.values.grain.active = true;
            params.values.scatter_halation = {};
            JuicerTestSupport::PreparedStateHold state = JuicerTestSupport::prepare_product(runtime.borrow().assets, params);
            std::string diagnostic;
            JuicerCudaTest::PreparedFrameFixture::FrameFacts facts;
            facts.renderExtent = {0, 0, 64, 48};
            facts.fullFrameExtent = facts.renderExtent;
            facts.pixelDefinition = facts.fullFrameExtent;
            facts.pixelSizeUm = 10.0f;
            facts.filmFormatLongEdgeMm = 35.0f;
            facts.canonicalWidth = 64.0;
            facts.canonicalHeight = 48.0;
            facts.scaleX = facts.scaleY = 1.0;
            facts.frameRate = 24.0;
            facts.sessionSeed = 17;
            facts.clipToken = 23;
            for (bool competing : {true, false}) {
                auto row = transition_rows.case_event(routeName + "/" + (competing ? "competing" : "malformed"));
                SubmissionSnapshot snapshot;
                snapshot.instanceToken.value = 0x53344438;
                snapshot.frameToken.value = next;
                snapshot.snapshotId = next++;
                snapshot.deviceContextKey = key;
                JuicerCuda::Failure error;
                Frame frame;
                WorkspacePreparedFrameContext preparation{&JuicerCuda::borrow_root(runtime.borrow().cuda), key, snapshot, device.stream, &frame, &error, competing, !competing, false};
                JuicerCuda::ExecutorTest::reset_workspace_transitions();
                require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                            runtime.borrow().assets, state, facts, &preparation, &prepare_workspace_frame_from_state, diagnostic),
                        diagnostic);
                require(preparation.dirWasActive, "workspace DIR must be active");
                require(preparation.grainWasValid, "workspace grain frame descriptor must be valid");
                // Root preparation is inside the actual prepared-state fixture callback.
                const std::string routeKey = Spektrafilm::scan_route_key(route);
                if (competing) {
                    require(!frame.active() && !frame.workspace_lease().active(), "competing failure published frame/marker");
                    require(error.diagnostic == "ResourceDescriptorMismatch phase=workspace field=grain_extent route=" + routeKey,
                            "competing failure precedence: " + error.diagnostic);
                    require(std::string(frame.failure_stage_tag()) == "derive_cuda_frame_workspace_request" &&
                                std::string(frame.failure_prefix()) == "CUDA frame workspace derivation failed",
                            "competing failure envelope");
                } else {
                    require(!frame.active() && !frame.workspace_lease().active(), "malformed roles published frame/marker");
                    require(error.diagnostic == "invalid frame scratch request route=" + routeKey, "malformed role diagnostic: " + error.diagnostic);
                    require(std::string(frame.failure_stage_tag()) == "derive_cuda_frame_workspace_request" &&
                                std::string(frame.failure_prefix()) == "CUDA frame workspace derivation failed",
                            "malformed role envelope");
                }
                require(error.status.category == FJ_STATUS_PREPARATION_FAILURE && error.status.api == FJ_API_NONE && error.status.native_code == 0,
                        "workspace failure status changed");
                const auto observed = JuicerCuda::ExecutorTest::workspace_transitions();
                require(!observed.requested && !observed.admitted && observed.resourcesAcquired == 0 && observed.candidateCount == 0 &&
                            observed.finishes == 0 && !observed.recoveryPending,
                        "malformed construction acquired or published workspace");
                const auto primary = error;
                frame.abort();
                require(error.diagnostic == primary.diagnostic && error.status.category == primary.status.category, "abort lost primary failure");
                std::cout << "replacement R2 route=" << routeKey << " competing=" << competing << " status={3,0,0} diagnostic=" << error.diagnostic
                          << " stage=derive_cuda_frame_workspace_request prefix=CUDA frame workspace derivation failed" << '\n';
                row.passed();
            }
            const std::array<const char*, 3> carrier_names{"disabled", "dedicated", "camera-diffusion"};
            for (unsigned carrier : {0u, 1u, 2u}) {
                auto row = transition_rows.case_event(routeName + "/" + carrier_names[carrier]);
                params.values.grain.active = false;
                params.values.dir_couplers.active = false;
                params.values.gate_dust_amount = 1.0;
                params.values.camera_diffusion.active = carrier == 2;
                params.values.scatter_halation = carrier != 0
                                                     ? decltype(params.values.scatter_halation){1, 1.0f, 1.0f, 1.0f, 1.0f}
                                                     : decltype(params.values.scatter_halation){};
                JuicerTestSupport::PreparedStateHold state = JuicerTestSupport::prepare_product(runtime.borrow().assets, params);
                auto gateFacts = facts;
                gateFacts.pixelSizeUm = 35'000.0f / 64;
                gateFacts.filmFormatLongEdgeMm = 35.0f;
                SubmissionSnapshot snapshot;
                snapshot.instanceToken.value = 0x53344438;
                snapshot.frameToken.value = next;
                snapshot.snapshotId = next++;
                snapshot.deviceContextKey = key;
                JuicerCuda::ExecutorTest::reset_workspace_transitions();
                JuicerCuda::Failure gateFailure;
                Frame frame;
                WorkspacePreparedFrameContext preparation{&JuicerCuda::borrow_root(runtime.borrow().cuda), key, snapshot, device.stream, &frame, &gateFailure};
                preparation.invalidateGateWidth = true;
                preparation.omitGrain = true;
                preparation.omitDir = true;
                require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                            runtime.borrow().assets, state, gateFacts, &preparation, &prepare_workspace_frame_from_state, diagnostic),
                        diagnostic);
                require(preparation.gateWasActive, "gate fixture not active");
                require(preparation.cameraDiffusionWasPresent == (carrier == 2),
                        "camera diffusion descriptor presence changed");
                const auto marker = frame.workspace_lease();
                require(frame.active() && marker.active(), "attachment failure moved ahead of checkpoint: " + gateFailure.diagnostic);
                require(!frame.scanner_workspace(marker).active && !frame.spatial_dir_scratch(marker).active, "failed construction exposed a workspace view");
                auto halationView = frame.scatter_halation_resources();
                require((halationView.descriptor != nullptr) == (carrier != 0), "attachment failure changed halation activation");
                if (carrier != 0) {
                    require(halationView.fullFrameWidth == 64 && halationView.fullFrameHeight == 48,
                            "attachment failure erased prepared halation extent");
                    require(halationView.carrierSource == (carrier == 2 ? JuicerCuda::ScatterHalationCarrierSource::CameraDiffusionStagePlanes
                                                                        : JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes),
                            "attachment failure changed halation carrier");
                    if (carrier == 2) {
                        const auto preparedDiffusion = frame.diffusion_resources();
                        require(preparedDiffusion.active && !halationView.currentCarrier.redSensitive && !halationView.currentCarrier.greenSensitive && !halationView.currentCarrier.blueSensitive,
                                "shared halation carrier was bound before diffusion");
                        const auto& planes = preparedDiffusion.execution.stagePlanes;
                        halationView.currentCarrier = {planes.redSensitive, planes.greenSensitive, planes.blueSensitive, planes.rowStrideFloats};
                        frame.mark_diffusion_work_enqueued();
                    }
                    const std::size_t planeBytes = halationView.currentCarrier.rowStrideFloats * halationView.fullFrameHeight * sizeof(float);
                    require_cuda(cudaMemsetAsync(halationView.currentCarrier.redSensitive, 0, planeBytes, device.stream), "attachment halation red carrier");
                    require_cuda(cudaMemsetAsync(halationView.currentCarrier.greenSensitive, 0, planeBytes, device.stream), "attachment halation green carrier");
                    require_cuda(cudaMemsetAsync(halationView.currentCarrier.blueSensitive, 0, planeBytes, device.stream), "attachment halation blue carrier");
                    require_cuda(JuicerCuda::launch_scatter_halation(halationView, device.stream).status, "attachment halation launch");
                }
                require(!frame.checkpoint_large_scratch_transition(marker, device.stream, print ? "print_large_scratch_transition" : "direct_large_scratch_transition", gateFailure), "malformed attachment passed checkpoint");
                require(gateFailure.status.category == FJ_STATUS_PREPARATION_FAILURE && gateFailure.status.api == FJ_API_NONE && gateFailure.status.native_code == 0 &&
                            gateFailure.diagnostic == "invalid frame scratch request",
                        "attachment checkpoint envelope changed");
                const auto observed = JuicerCuda::ExecutorTest::workspace_transitions();
                require(observed.resourcesAcquired == 1 && !observed.requested && !observed.admitted && observed.candidateCount == 0, "attachment failure fabricated a completed layout");
                frame.abort();
                require(gateFailure.diagnostic == "invalid frame scratch request" && gateFailure.status.category == FJ_STATUS_PREPARATION_FAILURE &&
                            gateFailure.status.api == FJ_API_NONE && gateFailure.status.native_code == 0,
                        "abort lost attachment checkpoint failure");
                require(!frame.scatter_halation_resources().descriptor && !frame.diffusion_resources().active, "aborted attachment failure exposed prepared optics");
                std::cout << "preserved attachment checkpoint route=" << Spektrafilm::scan_route_key(route) << " carrier=" << carrier << " status={3,0,0} diagnostic=" << gateFailure.diagnostic << '\n';
                row.passed();
            }
        }
        const std::array cases{Case{"negative-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"negative-print", Spektrafilm::ScanRoute::NegativePrintScan, 1, 4, 0, 0},
                               Case{"positive-direct", Spektrafilm::ScanRoute::PositiveDirectScan, 2, 3, 11, -3},
                               Case{"positive-print", Spektrafilm::ScanRoute::PositivePrintScan, 0, 4, 0, 0}};
        std::ifstream fixture(JUICER_PROCESSOR_REFERENCE_PATH);
        require(static_cast<bool>(fixture), "workspace processor fixture unavailable");
        unsigned pixels = 0;
        for (const auto& test : cases) {
            auto case_event = candidate_rows.case_event(std::string(test.name) + "/actual-schedule");
            std::string name;
            std::size_t count = 0;
            fixture >> name >> count;
            require(fixture && name == test.name && count > 0, "workspace fixture membership");
            std::vector<float> expected(count);
            for (float& value : expected) {
                fixture >> value;
            }
            require(static_cast<bool>(fixture), "workspace fixture truncated");
            nlohmann::json words = nlohmann::json::array();
            for (float value : expected) {
                words.push_back(std::bit_cast<std::uint32_t>(value));
            }
            words = PrintIlluminantFixtures::apply(std::string(JUICER_PROCESSOR_REFERENCE_PATH) + "." + test.name, words);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                expected[i] = std::bit_cast<float>(words.at(i).get<std::uint32_t>());
            }
            JuicerTestSupport::InstanceFixture state;
            MembershipRetirement stateRetirement{runtime, state};
            JuicerCuda::ExecutorTest::reset_workspace_transitions();
            compare_pixels(test.name, render_case(runtime, test, parameters_for(test), state, false, ExecutionPath::Direct, DestinationLayout::Matching, next++), expected);
            require(JuicerCuda::ExecutorTest::workspaceHandoff.has_value(), "actual schedule handoff not observed");
            ++pixels;
            std::cout << "workspace actual schedule route=" << Spektrafilm::scan_route_key(test.route)
                      << " handoff=" << static_cast<unsigned>(*JuicerCuda::ExecutorTest::workspaceHandoff) << '\n';
            case_event.passed();
        }
        unsigned injectedFailures = 0;
        std::vector<float> latestPixels;
        const auto run = [&](const Case& test, const JuicerTestSupport::PreparationInput& params, unsigned reject, JuicerCuda::ExecutorTest::WorkspaceRejection kind, bool failureExpected) {
            using namespace JuicerCuda::ExecutorTest;
            reset_workspace_transitions();
            require(arm_workspace_rejection(reject, kind), "workspace seam configuration rejected");
            require(!arm_workspace_rejection(7, kind) && !arm_workspace_rejection(1, static_cast<WorkspaceRejection>(255)),
                    "invalid seam configuration replaced armed input");
            JuicerTestSupport::InstanceFixture state;
            MembershipRetirement stateRetirement{runtime, state};
            bool failed = false;
            JuicerCuda::Failure failure;
            try {
                const auto actual = render_case(runtime, test, params, state, false, ExecutionPath::Direct, DestinationLayout::Matching, next++);
                require(!actual.empty() && std::all_of(actual.begin(), actual.end(), [](float value) {
                    return std::isfinite(value);
                }),
                        "workspace actual pixels unavailable");
                latestPixels = actual;
                ++pixels;
            } catch (const JuicerCuda::ExecutionFailure& error) {
                failure = error.failure;
                failed = true;
                ++injectedFailures;
            }
            require(failed == failureExpected, std::string(test.name) + ": unexpected workspace failure: " + failure.diagnostic);
            const auto observed = workspace_transitions();
            if (failed) {
                const auto status = kind == WorkspaceRejection::Capacity       ? FjStatus{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}
                                    : kind == WorkspaceRejection::InvalidValue ? JuicerCuda::runtime_failure_status(cudaErrorInvalidValue)
                                                                               : JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed);
                const std::string text = kind == WorkspaceRejection::Capacity       ? "workspace candidate capacity rejection"
                                         : kind == WorkspaceRejection::InvalidValue ? "workspace candidate invalid-value rejection"
                                                                                    : "workspace candidate context-loss rejection";
                require(failure.status.category == status.category && failure.status.api == status.api && failure.status.native_code == status.native_code && failure.diagnostic == text,
                        "workspace failure envelope or primary failure changed");
                require(observed.classified.category == status.category && observed.recoveryPending == (kind == WorkspaceRejection::ContextLoss), "workspace typed recovery classification changed");
                require(observed.candidateCount == (kind == WorkspaceRejection::Capacity ? 6u : 1u) && !observed.admitted &&
                            observed.finishes == 0 && observed.aborts == 1 && !observed.captureEnqueued && !observed.scannerEnqueued,
                        "failed workspace published or advanced candidates");
                require(observed.postFrame && *observed.postFrame == *observed.candidates[observed.candidateCount - 1].layout,
                        "failed candidate lost post-frame shedding layout");
            } else {
                require(observed.finishes == 1 && observed.aborts == 0 && observed.scannerEnqueued && workspaceHandoff.has_value(), "workspace successful schedule/finish missing");
                if (observed.admitted) {
                    require(observed.bindingsObserved && observed.scannerMapping && observed.grainMapping && observed.postFrame.has_value(), "workspace plane association or retention changed");
                    require(JuicerCuda::ResourceManager::scratch_request_descriptor_is_valid(observed.admitted->allocation()), "admitted projection invalid");
                    require(*observed.admitted == *observed.postFrame, "post-frame retention differs from admitted build candidate");
                }
                require(observed.captureEnqueued == (*workspaceHandoff == WorkspaceHandoff::MaterializedDensity), "capture result did not follow actual development");
            }
            unsigned visitedCandidates = 0;
            for (unsigned i = 0; i < observed.candidateCount; ++i) {
                const auto& candidate = observed.candidates[i];
                require(candidate.layout.has_value() && candidate.backendCalls >= 1, "candidate did not reach validated backend");
                require(candidate.succeeded == (!failureExpected && i == reject), "outer candidate decision changed across backend retries");
                if (!candidate.succeeded) {
                    require(candidate.result.category == (kind == WorkspaceRejection::Capacity       ? FJ_STATUS_ALLOCATION_FAILURE
                                                          : kind == WorkspaceRejection::InvalidValue ? FJ_STATUS_CUDA_FAILURE
                                                                                                     : FJ_STATUS_CONTEXT_LOSS),
                            "candidate typed rejection changed");
                }
                std::cout << test.name << " candidate=" << i << " backend_calls=" << candidate.backendCalls << " success=" << candidate.succeeded
                          << " generation=" << candidate.layout->allocation().generation << '\n';
                ++visitedCandidates;
            }
            require(visitedCandidates == observed.candidateCount, "workspace candidate comparison visit count mismatch");
            std::cout << "workspace cohort route=" << Spektrafilm::scan_route_key(test.route) << " case=" << test.name
                      << " reject=" << reject << " failed=" << failed << " source_cache=" << observed.sourceCacheChannels
                      << " final_cache=" << observed.finalCacheChannels << '\n';
            return observed;
        };
        for (unsigned route = 0; route < 4; ++route) {
            const Case test{"workspace-route", static_cast<Spektrafilm::ScanRoute>(route), 0, 4, 0, 0};
            const std::string routeName = std::array{"negative-direct", "negative-print", "positive-direct", "positive-print"}[route];
            auto params = parameters_for(test);
            params.values.grain.active = false;
            params.values.dir_couplers.active = false;
            params.values.glare_active = false;
            params.values.scanner_unsharp_mask[0] = params.values.scanner_unsharp_mask[1] = 0.0;
            auto no_work_row = candidate_rows.case_event(routeName + "/no-work");
            const auto noWork = run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
            require(!noWork.requested && !noWork.admitted && noWork.candidateCount == 0, "no-family schedule requested scratch");
            no_work_row.passed();
            params.values.dir_couplers.active = true;
            params.values.scanner_unsharp_mask[0] = params.values.scanner_unsharp_mask[1] = 0.5;
            auto dir_unsharp_row = candidate_rows.case_event(routeName + "/dir-unsharp");
            run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
            dir_unsharp_row.passed();
            params.values.grain.active = true;
            params.values.grain.sublayers_active = true;
            params.values.grain.chroma_shared_weight = 1.0f;
            Case strict = test;
            strict.name = "strict-yvv";
            strict.width = 512;
            strict.height = 48;
            if (route < 2) {
                std::vector<float> baseline;
                for (unsigned reject = 0; reject <= 6; ++reject) {
                    auto reject_row = candidate_rows.case_event(routeName + "/strict-yvv/capacity/" + std::to_string(reject));
                    const auto observed = run(strict, params, reject, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, reject == 6);
                    const std::array raw{3, 3, 3, 3, 3, 1};
                    const std::array temp{3, 3, 3, 2, 1, 1};
                    const std::array cache{3, 2, 0, 0, 0, 0};
                    for (unsigned i = 0; i < observed.candidateCount; ++i) {
                        const auto& allocation = observed.candidates[i].layout->allocation();
                        require(allocation.spatialDirPlaneRoles.rawCorrectionPlanes == raw[i] && allocation.spatialDirPlaneRoles.filteredCorrectionPlanes == 3 &&
                                    allocation.spatialDirPlaneRoles.filterTempPlanes == temp[i] && allocation.spatialDirPlaneRoles.cachedLogRawPlanes == cache[i] &&
                                    allocation.spatialDirTargetPlaneRoles.rawCorrectionPlanes == raw[i] && allocation.spatialDirTargetPlaneRoles.filterTempPlanes == temp[i] &&
                                    allocation.spatialDirTargetPlaneRoles.cachedLogRawPlanes == cache[i],
                                "strict-YVV candidate order/target changed");
                    }
                    if (reject < 6) {
                        if (reject == 0) {
                            baseline = latestPixels;
                        } else {
                            require(latestPixels.size() == baseline.size(), "fallback pixel count changed");
                            std::size_t visitedPixels = 0;
                            for (std::size_t i = 0; i < baseline.size(); ++i) {
                                RenderAssertions::require_same_bits("strict-YVV candidate output", latestPixels[i], baseline[i]);
                                ++visitedPixels;
                            }
                            require(visitedPixels == baseline.size(), "strict-YVV scalar comparison visit count mismatch");
                        }
                        require(observed.admitted->grain_storage() == (reject == 5 ? GrainStorage::Dedicated : GrainStorage::DirRaw), "fallback grain binding changed");
                        require(observed.sourceCacheChannels == (reject == 0 ? 3u : reject == 1 ? 2u
                                                                                                : 0u),
                                "admitted source-cache count changed");
                    }
                    reject_row.passed();
                }
                auto invalid_value_row = candidate_rows.case_event(routeName + "/strict-yvv/invalid-value");
                run(strict, params, 1, JuicerCuda::ExecutorTest::WorkspaceRejection::InvalidValue, true);
                invalid_value_row.passed();
                auto context_loss_row = candidate_rows.case_event(routeName + "/strict-yvv/context-loss");
                run(strict, params, 1, JuicerCuda::ExecutorTest::WorkspaceRejection::ContextLoss, true);
                context_loss_row.passed();
                for (unsigned shape = 0; shape < 4; ++shape) {
                    auto shape_row = candidate_rows.case_event(routeName + "/strict-yvv/shape/" + std::to_string(shape));
                    params.values.grain.sublayers_active = shape >= 2;
                    params.values.grain.chroma_shared_weight = shape % 2 ? 1.0f : 0.0f;
                    params.values.grain.chroma_independent_weight = shape % 2 ? 0.0f : 1.0f;
                    run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
                    shape_row.passed();
                }
                params.values.grain.debug_view = 1;
                auto debug_row = candidate_rows.case_event(routeName + "/strict-yvv/debug-view");
                run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
                debug_row.passed();
                params.values.grain.debug_view = 0;
                params.values.camera_diffusion.active = true;
                auto diffusion_row = candidate_rows.case_event(routeName + "/strict-yvv/camera-diffusion");
                const auto diffusion = run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
                require(diffusion.admitted->grain_storage() == GrainStorage::Dedicated, "diffusion used DIR raw grain");
                diffusion_row.passed();
            } else {
                auto positive_row = candidate_rows.case_event(routeName + "/strict-yvv/positive-storage");
                const auto positive = run(strict, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
                require(positive.admitted->scanner_storage() == ScannerStorage::Dedicated && positive.admitted->grain_storage() == GrainStorage::Dedicated,
                        "positive route aliased corrections");
                require(positive.sourceCacheChannels == 0 && positive.finalCacheChannels == 3 && positive.finalDevelopment.has_value(), "later cache staging missing");
                positive_row.passed();
            }
            params.values.grain.active = false;
            params.values.camera_diffusion.active = false;
            params.values.film_dust_amount = 1.0;
            params.values.film_scratch_amount = 1.0;
            params.values.gate_dust_amount = 1.0;
            params.values.gate_scratch_amount = 1.0;
            params.values.gate_weave_amount = 1.0;
            auto physical_row = candidate_rows.case_event(routeName + "/physical-effects");
            run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
            physical_row.passed();
        }
        for (bool camera : {false, true}) {
            auto carrier_row = candidate_rows.case_event(std::string("halation-carrier/") + (camera ? "camera-diffusion" : "dedicated"));
            Case test{"halation-carrier", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0, true};
            auto params = parameters_for(test);
            params.values.camera_diffusion.active = camera;
            params.values.enlarger_diffusion.active = true;
            run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
            carrier_row.passed();
        }
        JuicerCuda::ExecutorTest::reset_workspace_transitions();
        std::cout << "WorkspaceTransitions replacement: genuine_pixels=" << pixels << " construction_failures=4 checkpoint_failures=6 injected_failures=" << injectedFailures << '\n';
    }

    enum class EffectsCallerPath : std::uint8_t {
        Pixels,
        StopBeforePreparation,
        BeforeBuilder
    };

    void run_effects_frame_cases(CallbackSession& runtime, const std::filesystem::path& manifestPath, const std::string& preset) {
        using Json = nlohmann::json;
        const auto manifest = Json::parse(std::ifstream(manifestPath));
        const auto captured = load_shared_fixture(manifestPath.parent_path() / manifest.at("presets").at(preset).at("native").get<std::string>(), "manifest.json");
        if (captured.at("preset") != preset || captured.at("cases").size() != 347 || captured.at("products").size() != 72)
            throw std::runtime_error("effects caller independent context/membership changed");
        const std::array caseNames{
            "mixed", "active-zero-work", "active-zero-work-recipe-zero", "inactive", "inactive-bad-identity", "film_dust", "film_scratch", "gate_dust", "gate_scratch", "weave-1", "weave-nan", "geometry-roi", "geometry-negative", "weave-roi", "film-time--1.75", "film_dust-ignored-frame_rate", "gate_dust-ignored-frame_time", "gate_dust-ignored-frame_rate", "gate_dust-ignored-pixel_size_um", "weave-frame_rate-0", "film_format_long_edge_mm-zero"};
        std::vector<std::string> intended;
        intended.reserve(156);
        for (unsigned route = 0; route < 4; ++route) {
            for (const auto& row : captured.at("products"))
                if (row.at("route") == route)
                    intended.push_back("route-" + std::to_string(route) + "/product/" + row.at("spec").at("id").get<std::string>());
            for (const auto& row : captured.at("cases")) {
                const auto name = row.at("input").at("id").get<std::string>();
                if (std::find(caseNames.begin(), caseNames.end(), name) != caseNames.end())
                    intended.push_back("route-" + std::to_string(route) + "/case/" + name);
            }
        }
        JuicerTests::Cohort rows("effects_frame", intended);
        const auto scalar = [](const Json& v) {
            return std::bit_cast<float>(v.get<std::uint32_t>());
        };
        const auto wide = [](const Json& v) {
            return std::bit_cast<double>(v.get<std::uint64_t>());
        };
        const auto extent = [](const Json& v) -> Spektrafilm::FilmJuicerEffectsFrameExtent {
            return {v.at(0), v.at(1), v.at(2), v.at(3)};
        };
        const auto rect = [](const Spektrafilm::FilmJuicerEffectsFrameExtent& v) -> JuicerCuda::FrameRect {
            return {v.x, v.y, v.x + v.width, v.y + v.height};
        };
        const auto fields = [](const Spektrafilm::FilmJuicerEffectsFrameExtent& v) -> Json {
            return {v.x, v.y, v.width, v.height};
        };
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "effects caller stream");
        constexpr std::size_t kSamples = std::size_t{96} * 71 * 4;
        constexpr std::size_t kBytes = kSamples * sizeof(float);
        require_cuda(cudaMalloc(&device.source, kBytes), "effects caller input");
        require_cuda(cudaMalloc(&device.destination, kBytes), "effects caller output");
        const std::vector<float> source(kSamples, 0.18f), canary(kSamples, kCanary);
        require_cuda(cudaMemcpy(device.source, source.data(), kBytes, cudaMemcpyHostToDevice), "effects caller initialized input");
        unsigned runs = 0, pixels = 0, heldConstructions = 0, failures = 0, actualCompanions = 0;
        for (unsigned route = 0; route < 4; ++route) {
            const auto run = [&](const Json& row, EffectsCallerPath path, bool fullFallback = false) {
                const bool genuine = row.contains("spec");
                if (!genuine && path == EffectsCallerPath::Pixels)
                    throw std::runtime_error("literal effects construction cannot execute pixels");
                const auto& v = row.at("input");
                const auto render = extent(v.at("render_extent")), full = extent(v.at("pixel_definition"));
                const auto fullFrame = extent(v.at("full_frame_extent"));
                JuicerTestSupport::PreparationInput params;
                params.values.scan_route = route;
                params.film_key = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                params.values.camera_film_format_long_edge_mm = 36.0f;
                params.values.dir_couplers.active = false;
                params.values.grain.active = false;
                if (genuine) {
                    const auto& spec = row.at("spec");
                    params.values.grain.active = spec.at("grain_active").get<bool>() ? 1u : 0u;
                    params.values.grain.debug_view = spec.at("debug");
                    params.values.film_dust_amount = scalar(spec.at("amounts")[0]);
                    params.values.film_scratch_amount = scalar(spec.at("amounts")[1]);
                    params.values.gate_dust_amount = scalar(spec.at("amounts")[2]);
                    params.values.gate_scratch_amount = scalar(spec.at("amounts")[3]);
                    params.values.gate_weave_amount = wide(spec.at("weave"));
                }
                JuicerTestSupport::PreparedStateHold state = JuicerTestSupport::prepare_product(runtime.borrow().assets, params);
                std::array<char, 1024> stateErrorText{};
                FjErrorBuffer stateError{stateErrorText.data(), stateErrorText.size(), 0};
                FjPreparedStateView stateView{};
                if (state.view(stateView, stateError).category != FJ_STATUS_SUCCESS)
                    throw std::runtime_error(std::string(stateErrorText.data(), stateError.length));
                const auto bounds = rect(full);
                auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    bounds, bounds, static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
                const std::optional<ScatterHalationFrameDescriptor> halation;
                Spektrafilm::FilmJuicerEffectsRecipe ignored;
                auto geometry = JuicerTestSupport::effects_frame_input(v, ignored).geometry;
                const auto rowBytes = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(full.width) * 4 * sizeof(float));
                JuicerCuda::ExecutionFrame frame{
                    bounds, rect(render), fullFallback ? JuicerCuda::FrameRect{} : bounds, reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<unsigned char*>(device.destination), rowBytes, rowBytes, path == EffectsCallerPath::BeforeBuilder ? 2 : 4, device.stream, diffusion, halation, geometry, scalar(v.at("pixel_size_um")), wide(v.at("frame_time")), wide(v.at("frame_rate")), v.at("session_seed").get<std::uint64_t>(), v.at("clip_token").get<std::uintptr_t>(), meter, false, false};
                const bool actualHeld = genuine && path != EffectsCallerPath::BeforeBuilder && !fullFallback;
                actualCompanions += actualHeld ? 1u : 0u;
                Json actualInput = v;
                double actualFrameTime = wide(v.at("frame_time"));
                double actualFrameRate = wide(v.at("frame_rate"));
                if (!std::isfinite(actualFrameTime))
                    actualFrameTime = 37.0;
                if (!std::isfinite(actualFrameRate))
                    actualFrameRate = 24.0;
                if (actualHeld) {
                    actualInput["frame_time"] = std::bit_cast<std::uint64_t>(actualFrameTime);
                    actualInput["frame_rate"] = std::bit_cast<std::uint64_t>(actualFrameRate);
                    frame.timeFrames = actualFrameTime;
                    frame.frameRate = actualFrameRate;
                }
                auto actualFull = fullFrame;
                auto actualPixelDefinition = full;
                auto actualRender = render;
                auto actualGeometry = geometry;
                std::optional<FrameConstructionResult> actualExpected;
                if (actualHeld) {
                    const FjExtent policyExtent{fullFrame.x, fullFrame.y, fullFrame.width, fullFrame.height};
                    const float pixelSizeUm = prepared_frame_pixel_size_um(state, policyExtent);
                    actualFull = fullFrame;
                    actualPixelDefinition = fullFrame;
                    actualRender = fullFrame;
                    actualInput["render_extent"] = Json::array({actualRender.x, actualRender.y, actualRender.width, actualRender.height});
                    actualInput["full_frame_extent"] = Json::array({actualFull.x, actualFull.y, actualFull.width, actualFull.height});
                    actualInput["pixel_definition"] = Json::array({actualPixelDefinition.x, actualPixelDefinition.y, actualPixelDefinition.width, actualPixelDefinition.height});
                    actualInput["pixel_size_um"] = std::bit_cast<std::uint32_t>(pixelSizeUm);
                    actualInput["film_format_long_edge_mm"] = std::bit_cast<std::uint32_t>(stateView.input_policy.film_format_long_edge_mm);
                    actualInput["canonical_x"] = std::bit_cast<std::uint64_t>(static_cast<double>(actualFull.x));
                    actualInput["canonical_y"] = std::bit_cast<std::uint64_t>(static_cast<double>(actualFull.y));
                    actualInput["canonical_width"] = std::bit_cast<std::uint64_t>(static_cast<double>(actualFull.width));
                    actualInput["canonical_height"] = std::bit_cast<std::uint64_t>(static_cast<double>(actualFull.height));
                    actualInput["scale_x"] = std::bit_cast<std::uint64_t>(1.0);
                    actualInput["scale_y"] = std::bit_cast<std::uint64_t>(1.0);
                    actualInput["pixel_aspect_ratio"] = std::bit_cast<std::uint64_t>(1.0);
                    actualGeometry = JuicerTestSupport::effects_frame_input(actualInput, ignored).geometry;
                    geometry = actualGeometry;
                    meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                        rect(actualFull), rect(actualFull), static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                    frame.sourceBounds = rect(actualFull);
                    frame.renderWindow = rect(actualRender);
                    frame.fullFrameExtent = rect(actualFull);
                    frame.sourceRowBytes = frame.destinationRowBytes = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(actualFull.width) * 4 * sizeof(float));
                    frame.pixelSizeUm = pixelSizeUm;
                    JuicerCudaTest::PreparedFrameFixture::FrameFacts actualFacts;
                    actualFacts.renderExtent = {actualRender.x, actualRender.y, actualRender.width, actualRender.height};
                    actualFacts.fullFrameExtent = {actualFull.x, actualFull.y, actualFull.width, actualFull.height};
                    actualFacts.pixelDefinition = {actualPixelDefinition.x, actualPixelDefinition.y, actualPixelDefinition.width, actualPixelDefinition.height};
                    actualFacts.pixelSizeUm = pixelSizeUm;
                    actualFacts.filmFormatLongEdgeMm = stateView.input_policy.film_format_long_edge_mm;
                    actualFacts.canonicalX = actualGeometry.canonicalX;
                    actualFacts.canonicalY = actualGeometry.canonicalY;
                    actualFacts.canonicalWidth = actualGeometry.canonicalWidth;
                    actualFacts.canonicalHeight = actualGeometry.canonicalHeight;
                    actualFacts.scaleX = actualGeometry.scaleX;
                    actualFacts.scaleY = actualGeometry.scaleY;
                    actualFacts.pixelAspectRatio = actualGeometry.pixelAspectRatio;
                    actualFacts.frameTime = actualFrameTime;
                    actualFacts.frameRate = actualFrameRate;
                    actualFacts.sessionSeed = v.at("session_seed");
                    actualFacts.clipToken = v.at("clip_token");
                    if (v.at("active").get<bool>())
                        actualExpected.emplace(capture_prepared_frame_descriptor(runtime, state, FrameObservationKind::Effects, actualFacts));
                    // Compare the captured fixed facts with the held product and raw facade.
                    const bool fixedSuccess = row.at("success").get<bool>();
                    if (fixedSuccess && v.at("active").get<bool>()) {
                        const auto fixedRender = extent(v.at("render_extent"));
                        JuicerCudaTest::PreparedFrameFixture::FrameFacts fixedFacts;
                        fixedFacts.renderExtent = {fixedRender.x, fixedRender.y, fixedRender.width, fixedRender.height};
                        fixedFacts.fullFrameExtent = {fullFrame.x, fullFrame.y, fullFrame.width, fullFrame.height};
                        fixedFacts.pixelDefinition = {full.x, full.y, full.width, full.height};
                        fixedFacts.pixelSizeUm = scalar(v.at("pixel_size_um"));
                        fixedFacts.filmFormatLongEdgeMm = scalar(v.at("film_format_long_edge_mm"));
                        const auto fixedGeometry = JuicerTestSupport::effects_frame_input(v, ignored).geometry;
                        fixedFacts.canonicalX = fixedGeometry.canonicalX;
                        fixedFacts.canonicalY = fixedGeometry.canonicalY;
                        fixedFacts.canonicalWidth = fixedGeometry.canonicalWidth;
                        fixedFacts.canonicalHeight = fixedGeometry.canonicalHeight;
                        fixedFacts.scaleX = fixedGeometry.scaleX;
                        fixedFacts.scaleY = fixedGeometry.scaleY;
                        fixedFacts.pixelAspectRatio = fixedGeometry.pixelAspectRatio;
                        fixedFacts.frameTime = wide(v.at("frame_time"));
                        fixedFacts.frameRate = wide(v.at("frame_rate"));
                        fixedFacts.sessionSeed = v.at("session_seed");
                        fixedFacts.clipToken = v.at("clip_token");
                        const auto fixed = capture_prepared_frame_descriptor(runtime, state, FrameObservationKind::Effects, fixedFacts);
                        if (fixed.status.category != FJ_STATUS_SUCCESS || !fixed.effects ||
                            JuicerTestSupport::effects_frame_fields(*fixed.effects) != row.at("native"))
                            throw std::runtime_error(v.at("id").get<std::string>() + ": held effects descriptor differs from the captured fixed-fact row");
                    } else if (!fixedSuccess) {
                        const auto fixed = prepare_literal_frame_construction(
                            FrameObservationKind::Effects, v, static_cast<Spektrafilm::ScanRoute>(route), scalar(v.at("pixel_size_um")), &build_literal_effects_math);
                        if (fixed.rawFacadeCalls != 1 || fixed.status.category != FJ_STATUS_PREPARATION_FAILURE ||
                            fixed.diagnostic != row.at("diagnostic").get<std::string>())
                            throw std::runtime_error(v.at("id").get<std::string>() + ": raw effects failure differs from the captured fixed-fact row");
                    }
                    // The frozen raw failure is separate from its finite held-frame companion.
                    if (!fixedSuccess)
                        ++failures;
                }
                ++runs;
                unsigned abortCalls = 0;
                require_cuda(cudaMemcpy(device.destination, canary.data(), kBytes, cudaMemcpyHostToDevice), "effects caller canary");
                JuicerCuda::ExecutorTest::reset_effects_frame_observation();
                JuicerCuda::ExecutorTest::FailureObservation events;
                JuicerCuda::ExecutorTest::ResourceFailures resources;
                bool success = !v.at("active").get<bool>() || row.at("success").get<bool>();
                if (actualHeld)
                    success = !v.at("active").get<bool>() || (actualExpected && actualExpected->status.category == FJ_STATUS_SUCCESS);
                const bool early = !success;
                const bool observeNoResourceWork = early ||
                                                   (path == EffectsCallerPath::StopBeforePreparation && success);
                if (actualHeld && observeNoResourceWork) {
                    resources.lutPosition = 1;
                    JuicerCuda::ExecutorTest::observation = &events;
                    JuicerCuda::ExecutorTest::resourceFailures = &resources;
                }
                FjStatus result{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                std::string failureText;
                bool recoveryPending = false;
                FrameDescriptorObservation frameMath{};
                JuicerCuda::ExecutorTest::EffectsFrameObservation observed{};
                if (actualHeld) {
                    AbortAfterNativeDescriptor stagedAbort{};
                    const FjAbortCallback actualAbort{&abort_after_native_descriptor, &stagedAbort};
                    const FjAbortCallback selectedAbort = path == EffectsCallerPath::Pixels ||
                                                                  (!success && path != EffectsCallerPath::BeforeBuilder)
                                                              ? FjAbortCallback{}
                                                              : actualAbort;
                    const FjCallCompletionResult rendered = render_held_prepared_state(
                        runtime, state, prepared_raw_frame(frame), 0x645700 + route, selectedAbort, &frameMath, &recoveryPending, failureText);
                    result = rendered.outcome.status;
                    abortCalls = selectedAbort.query == &abort_after_native_descriptor ? stagedAbort.calls : 0;
                    observed = JuicerCuda::ExecutorTest::effects_frame_observation();
                } else {
                    const auto raw = prepare_literal_frame_construction(
                        FrameObservationKind::Effects, v, static_cast<Spektrafilm::ScanRoute>(route), scalar(v.at("pixel_size_um")), &build_literal_effects_math);
                    const bool rawSuccess = row.at("success").get<bool>();
                    if (raw.rawFacadeCalls != 1 || raw.status.category != (rawSuccess ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE) ||
                        (rawSuccess && (!raw.effects || JuicerTestSupport::effects_frame_fields(*raw.effects) != row.at("native"))) ||
                        (!rawSuccess && raw.diagnostic != row.at("diagnostic").get<std::string>()))
                        throw std::runtime_error(v.at("id").get<std::string>() + ": raw effects facade/native composer witness changed");
                    if (!rawSuccess)
                        ++failures;
                    return;
                }
                JuicerCuda::ExecutorTest::observation = nullptr;
                JuicerCuda::ExecutorTest::resourceFailures = nullptr;
                const auto name = v.at("id").get<std::string>();
                if (frameMath.callbackCalls != 1u || frameMath.providerCalls != 1u) {
                    std::string message = name;
                    message += ": actual prepared-state callback/provider visit count changed callback=";
                    message += std::to_string(frameMath.callbackCalls);
                    message += " provider=";
                    message += std::to_string(frameMath.providerCalls);
                    message += " status=";
                    message += std::to_string(result.category);
                    message += " diagnostic=";
                    message += failureText;
                    throw std::runtime_error(message);
                }
                const auto geometryFields = [&](const Spektrafilm::FilmJuicerEffectsGeometry& g) -> Json {
                    return {fields(g.pixelDefinition), std::bit_cast<std::uint64_t>(g.canonicalX), std::bit_cast<std::uint64_t>(g.canonicalY), std::bit_cast<std::uint64_t>(g.canonicalWidth), std::bit_cast<std::uint64_t>(g.canonicalHeight), std::bit_cast<std::uint64_t>(g.scaleX), std::bit_cast<std::uint64_t>(g.scaleY), std::bit_cast<std::uint64_t>(g.pixelAspectRatio)};
                };
                if (actualHeld && path == EffectsCallerPath::StopBeforePreparation && success && v.at("active").get<bool>()) {
                    JuicerCudaTest::PreparedFrameFixture::FrameFacts witnessFacts;
                    witnessFacts.renderExtent = {actualRender.x, actualRender.y, actualRender.width, actualRender.height};
                    witnessFacts.fullFrameExtent = {actualFull.x, actualFull.y, actualFull.width, actualFull.height};
                    witnessFacts.pixelDefinition = witnessFacts.fullFrameExtent;
                    witnessFacts.pixelSizeUm = frame.pixelSizeUm;
                    witnessFacts.filmFormatLongEdgeMm = stateView.input_policy.film_format_long_edge_mm;
                    witnessFacts.canonicalX = actualGeometry.canonicalX;
                    witnessFacts.canonicalY = actualGeometry.canonicalY;
                    witnessFacts.canonicalWidth = actualGeometry.canonicalWidth;
                    witnessFacts.canonicalHeight = actualGeometry.canonicalHeight;
                    witnessFacts.scaleX = actualGeometry.scaleX;
                    witnessFacts.scaleY = actualGeometry.scaleY;
                    witnessFacts.pixelAspectRatio = actualGeometry.pixelAspectRatio;
                    witnessFacts.frameTime = actualFrameTime;
                    witnessFacts.frameRate = actualFrameRate;
                    witnessFacts.sessionSeed = v.at("session_seed");
                    witnessFacts.clipToken = v.at("clip_token");
                    FrameConstructionResult witness{FrameObservationKind::Effects};
                    std::string witnessDiagnostic;
                    if (!JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                            runtime.borrow().assets, state, witnessFacts, &witness, &capture_held_frame_construction, witnessDiagnostic)) {
                        std::string message = name;
                        message += ": held-state construction witness: ";
                        message += witnessDiagnostic;
                        throw std::runtime_error(message);
                    }
                    if (witness.status.category != FJ_STATUS_SUCCESS || !witness.effects || !actualExpected->effects ||
                        JuicerTestSupport::effects_frame_fields(*witness.effects) != JuicerTestSupport::effects_frame_fields(*actualExpected->effects))
                        throw std::runtime_error(name + ": held effects construction witness fields/hash changed");
                }
                if (observed.count != 1u)
                    throw std::runtime_error(name + ": effects helper invocation count");
                const auto& facts = actualInput;
                const Json expected = actualExpected && actualExpected->effects
                                          ? JuicerTestSupport::effects_frame_fields(*actualExpected->effects)
                                          : Json{};
                const std::uint32_t heldFormatBits = std::bit_cast<std::uint32_t>(stateView.input_policy.film_format_long_edge_mm);
                if (static_cast<std::uint32_t>(observed.route) != params.values.scan_route || fields(observed.renderExtent) != facts.at("render_extent") ||
                    fields(observed.fullFrameExtent) != fields(actualFull) ||
                    geometryFields(observed.geometry) != geometryFields(actualGeometry) ||
                    observed.filmFormatLongEdgeMm != 0.0f || facts.at("film_format_long_edge_mm") != heldFormatBits ||
                    std::bit_cast<std::uint32_t>(observed.pixelSizeUm) != facts.at("pixel_size_um") ||
                    std::bit_cast<std::uint64_t>(observed.frameTime) != facts.at("frame_time") ||
                    std::bit_cast<std::uint64_t>(observed.frameRate) != facts.at("frame_rate") ||
                    observed.sessionSeed != facts.at("session_seed") || observed.clipToken != facts.at("clip_token") || observed.success != success)
                    throw std::runtime_error(name + ": current callback effects facts");
                const bool present = success && v.at("active").get<bool>() && actualExpected && actualExpected->effects.has_value();
                if (observed.descriptor.has_value() != present || (present && JuicerTestSupport::effects_frame_fields(*observed.descriptor) != expected))
                    throw std::runtime_error(name + ": complete effects policies/geometry/identity or failed/inactive omission");
                const auto expectedStatus = early ? FJ_STATUS_PREPARATION_FAILURE : path == EffectsCallerPath::Pixels ? FJ_STATUS_SUCCESS
                                                                                                                      : FJ_STATUS_CANCELLED;
                const unsigned expectedAbortCalls = !early && path == EffectsCallerPath::StopBeforePreparation ? 3u : 0u;
                if (result.category != expectedStatus || result.api != 0 || result.native_code != 0 || recoveryPending || abortCalls != expectedAbortCalls)
                    throw std::runtime_error(std::string(name).append(": effects caller stage/status: ").append(failureText));
                if (early) {
                    const auto expectedText = path == EffectsCallerPath::BeforeBuilder
                                                  ? route % 2 ? "UnsupportedPrintComponentCountForPhase4C" : "UnsupportedDirectComponentCountForPhase3C"
                                              : actualHeld ? actualExpected->diagnostic
                                                           : row.at("diagnostic").get<std::string>();
                    if (failureText != expectedText || events.count != 0 || !events.armed || resources.lutAttempts != 0 || resources.stbnAttempts != 0 || resources.wangAttempts != 0 || resources.injections != 0)
                        throw std::runtime_error(name + ": failed effects reached preparation/upload/launch");
                } else if (path == EffectsCallerPath::StopBeforePreparation &&
                           (events.count != 0 || resources.lutAttempts != 0 || resources.stbnAttempts != 0 ||
                            resources.wangAttempts != 0 || resources.injections != 0)) {
                    throw std::runtime_error(name + ": cancellation entered resource preparation");
                }
                std::vector<float> actual(kSamples);
                require_cuda(cudaMemcpyAsync(actual.data(), device.destination, kBytes, cudaMemcpyDeviceToHost, device.stream), "effects caller readback");
                require_cuda(cudaStreamSynchronize(device.stream), "effects caller completion");
                if (path == EffectsCallerPath::Pixels) {
                    if (!genuine || early || actual == canary)
                        throw std::runtime_error("genuine effects pixel execution missing");
                    const auto count = static_cast<std::size_t>(actualHeld ? actualFull.width : full.width) * static_cast<std::size_t>(actualHeld ? actualFull.height : full.height) * 4;
                    if (count > actual.size())
                        throw std::runtime_error("effects intended pixel extent exceeds captured output");
                    std::size_t visited = 0;
                    for (std::size_t i = 0; i < count; ++i) {
                        if (i % 4 == 3 ? std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(0.18f) : !std::isfinite(actual[i]) || actual[i] < 0.0f || actual[i] > 1.0f)
                            throw std::runtime_error("effects genuine pixel/alpha contract");
                        ++visited;
                    }
                    if (visited != count)
                        throw std::runtime_error("effects scalar comparison visit count mismatch");
                    ++pixels;
                } else {
                    if (actual != canary)
                        throw std::runtime_error("effects construction-only execution wrote pixels");
                    if (!early)
                        ++heldConstructions;
                }
                std::cout << "effects caller route=" << route << " name=" << name << " pixels=" << (path == EffectsCallerPath::Pixels) << " fallback=" << fullFallback << " builder_calls=" << observed.count << " success=" << success << " passed\n";
            };
            for (const auto& row : captured.at("products")) {
                if (row.at("route") != route)
                    continue;
                auto case_event = rows.case_event("route-" + std::to_string(route) + "/product/" + row.at("spec").at("id").get<std::string>());
                const auto& spec = row.at("spec");
                const bool pixel = spec.at("pixels");
                const bool witness = spec.at("id") == "off";
                if (witness && fj_test_effects_frame_fault(1).category != 0)
                    throw std::runtime_error("arm inactive effects delegation witness");
                run(row, pixel ? EffectsCallerPath::Pixels : EffectsCallerPath::StopBeforePreparation);
                if (witness) {
                    FjEffectsFrameInput active{};
                    active.active = 1;
                    FjEffectsFrameMath output{};
                    FjEffectsFrameFailure failure{};
                    if (fj_test_effects_frame_prepare(&active, &output, &failure, nullptr).category != 8 || failure.reason != 4)
                        throw std::runtime_error("inactive effects helper consumed Rust mathematics");
                }
                if (spec.at("id") == "mixed") {
                    run(row, EffectsCallerPath::BeforeBuilder);
                    run(row, EffectsCallerPath::StopBeforePreparation, true);
                }
                case_event.passed();
            }
            for (const auto& row : captured.at("cases")) {
                const auto name = row.at("input").at("id").get<std::string>();
                if (std::find(caseNames.begin(), caseNames.end(), name) != caseNames.end()) {
                    auto case_event = rows.case_event("route-" + std::to_string(route) + "/case/" + name);
                    run(row, EffectsCallerPath::StopBeforePreparation);
                    case_event.passed();
                }
            }
            if (fj_cuda_retire_instance(runtime.borrow().cuda, 0x645700 + route, nullptr).category != 0)
                throw std::runtime_error("effects caller membership retirement");
        }
        if (runs != 164 || pixels != 32 || heldConstructions != 40 || failures != 20)
            throw std::runtime_error("effects caller exact pixel/failure/stop membership");
        if (actualCompanions != 72)
            throw std::runtime_error("effects actual full-frame companion membership");
        std::cout << "effects caller runs=" << runs << " genuine pixels=" << pixels << " held constructions=" << heldConstructions << " failures=" << failures << " actual_full_frame_companions=" << actualCompanions << " passed\n";
    }

    enum class GrainCallerPath : std::uint8_t {
        Pixels,
        StopBeforePreparation,
        BeforeBuilder
    };

    std::string grain_frame_product_id(const nlohmann::json& spec) {
        return std::string("product/active-") + (spec.at("active").get<bool>() ? "1" : "0") +
               "/layers-" + (spec.at("layers").get<bool>() ? "1" : "0") +
               "/debug-" + std::to_string(spec.at("debug").get<int>()) +
               "/chroma-" + std::to_string(spec.at("chroma").get<std::uint32_t>()) +
               "/amplitude-" + std::to_string(spec.value("amplitude", std::uint32_t{0})) +
               "/area-" + std::to_string(spec.value("area", std::uint32_t{0})) +
               "/ev-" + std::to_string(spec.value("ev", 0.0));
    }

    void run_grain_frame_cases(CallbackSession& runtime, const std::filesystem::path& manifestPath, const std::string& preset) {
        using Json = nlohmann::json;
        const auto manifest = Json::parse(std::ifstream(manifestPath));
        const auto captured = Json::parse(std::ifstream(manifestPath.parent_path() / manifest.at("presets").at(preset).at("native").get<std::string>()));
        if (captured.at("preset") != preset || captured.at("cases").size() != 153 || captured.at("products").size() != 44)
            throw std::runtime_error("grain caller independent context/membership changed");
        const std::array caseNames{"time--1.75", "time-1.5", "fps-23.976", "fps-0.2", "negative-origin", "partial-unblurred", "sigma-0.4", "sigma-0.4000001", "new-session", "new-clip", "mix", "S4D6-R1", "pixel-half", "pixel-eight", "geometry-96x71", "geometry-offset-96x71", "layers-film_layers_hash", "model-0", "frame-frame_time"};
        std::vector<std::string> intended;
        intended.reserve(114);
        for (unsigned route = 0; route < 4; ++route) {
            for (const auto& row : captured.at("products"))
                if (row.at("route") == route)
                    intended.push_back("route-" + std::to_string(route) + "/" + grain_frame_product_id(row.at("spec")));
            for (const auto& row : captured.at("cases")) {
                const auto& input = row.at("input");
                const auto name = input.at("name").get<std::string>();
                const auto logical = name.starts_with("positive/") ? name.substr(9) : name;
                if (input.at("capture_polarity") == route / 2 && input.at("film_polarity") == route / 2 &&
                    std::find(caseNames.begin(), caseNames.end(), logical) != caseNames.end())
                    intended.push_back("route-" + std::to_string(route) + "/case/" + name);
            }
        }
        JuicerTests::Cohort rows("grain_frame", intended);
        const auto scalar = [](const Json& v) {
            return std::bit_cast<float>(v.get<std::uint32_t>());
        };
        const auto wide = [](const Json& v) {
            return std::bit_cast<double>(v.get<std::uint64_t>());
        };
        const auto extent = [](const Json& v) -> Spektrafilm::VisualGrainFrameExtent {
            return {v.at(0), v.at(1), v.at(2), v.at(3)};
        };
        const auto rect = [](const Spektrafilm::VisualGrainFrameExtent& v) -> JuicerCuda::FrameRect {
            return {v.x, v.y, v.x + v.width, v.y + v.height};
        };
        const auto fields = [](const Spektrafilm::VisualGrainFrameExtent& v) -> Json {
            return {v.x, v.y, v.width, v.height};
        };
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "grain caller stream");
        constexpr std::size_t kSamples = std::size_t{96} * 71 * 4;
        constexpr std::size_t kBytes = kSamples * sizeof(float);
        require_cuda(cudaMalloc(&device.source, kBytes), "grain caller input");
        require_cuda(cudaMalloc(&device.destination, kBytes), "grain caller output");
        const std::vector<float> source(kSamples, 0.18f), canary(kSamples, kCanary);
        require_cuda(cudaMemcpy(device.source, source.data(), kBytes, cudaMemcpyHostToDevice), "grain caller initialized input");
        unsigned runs = 0, pixels = 0, heldConstructions = 0, failures = 0, actualCompanions = 0;
        const auto& base = captured.at("cases")[0].at("input");
        for (unsigned route = 0; route < 4; ++route) {
            const auto run = [&](const Json& row, GrainCallerPath path, bool fullFallback = false) {
                const bool genuine = row.contains("spec");
                if (!genuine && path == GrainCallerPath::Pixels)
                    throw std::runtime_error("literal grain construction cannot execute pixels");
                const auto& v = genuine ? base : row.at("input");
                const auto render = extent(v.at("render_extent")), full = extent(v.at("full_frame_extent"));
                JuicerTestSupport::PreparationInput params;
                params.values.scan_route = route;
                params.film_key = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                if (genuine) {
                    const auto& spec = row.at("spec");
                    params.values.grain.active = spec.at("active").get<bool>() ? 1u : 0u;
                    params.values.grain.sublayers_active = spec.at("layers").get<bool>() ? 1u : 0u;
                    params.values.grain.debug_view = spec.at("debug");
                    params.values.grain.chroma_mix = scalar(spec.at("chroma"));
                    params.values.grain.amplitude = scalar(spec.value("amplitude", Json(1065353216u)));
                    params.values.grain.particle_area_um2 = scalar(spec.value("area", Json(1051428127u)));
                    params.values.camera_exposure_compensation_ev = spec.value("ev", 0.0);
                } else {
                    params.values.dir_couplers.active = false;
                }
                JuicerTestSupport::PreparedStateHold state = JuicerTestSupport::prepare_product(runtime.borrow().assets, params);
                const auto bounds = rect(full);
                std::array<char, 1024> stateErrorText{};
                FjErrorBuffer stateError{stateErrorText.data(), stateErrorText.size(), 0};
                FjPreparedStateView stateView{};
                if (state.view(stateView, stateError).category != FJ_STATUS_SUCCESS)
                    throw std::runtime_error(std::string(stateErrorText.data(), stateError.length));
                const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    bounds, bounds, static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
                const std::optional<ScatterHalationFrameDescriptor> halation;
                const Spektrafilm::FilmJuicerEffectsGeometry geometry{};
                const auto rowBytes = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(full.width) * 4 * sizeof(float));
                JuicerCuda::ExecutionFrame frame{
                    bounds, rect(render), fullFallback ? JuicerCuda::FrameRect{} : bounds, reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<unsigned char*>(device.destination), rowBytes, rowBytes, path == GrainCallerPath::BeforeBuilder ? 2 : 4, device.stream, diffusion, halation, geometry, scalar(v.at("pixel_size_um")), wide(v.at("frame_time")), wide(v.at("frame_rate")), v.at("session_seed").get<std::uint64_t>(), v.at("clip_token").get<std::uintptr_t>(), meter, false, false};
                const bool actualHeld = genuine && path != GrainCallerPath::BeforeBuilder && !fullFallback;
                actualCompanions += actualHeld ? 1u : 0u;
                Json actualInput = v;
                std::optional<FrameConstructionResult> actualExpected;
                if (actualHeld) {
                    const FjExtent policyExtent{full.x, full.y, full.width, full.height};
                    const float pixelSizeUm = prepared_frame_pixel_size_um(state, policyExtent);
                    actualInput["render_extent"] = Json::array({full.x, full.y, full.width, full.height});
                    actualInput["full_frame_extent"] = actualInput["render_extent"];
                    actualInput["pixel_size_um"] = std::bit_cast<std::uint32_t>(pixelSizeUm);
                    frame.renderWindow = bounds;
                    frame.fullFrameExtent = bounds;
                    frame.pixelSizeUm = pixelSizeUm;
                    JuicerCudaTest::PreparedFrameFixture::FrameFacts actualFacts;
                    actualFacts.renderExtent = {full.x, full.y, full.width, full.height};
                    actualFacts.fullFrameExtent = actualFacts.renderExtent;
                    actualFacts.pixelDefinition = actualFacts.fullFrameExtent;
                    actualFacts.pixelSizeUm = pixelSizeUm;
                    actualFacts.frameTime = wide(v.at("frame_time"));
                    actualFacts.frameRate = wide(v.at("frame_rate"));
                    actualFacts.sessionSeed = v.at("session_seed");
                    actualFacts.clipToken = v.at("clip_token");
                    if (params.values.grain.active)
                        actualExpected.emplace(capture_prepared_frame_descriptor(runtime, state, FrameObservationKind::Grain, actualFacts));
                    // Compare the captured fixed facts with the held product.
                    if (params.values.grain.active) {
                        JuicerCudaTest::PreparedFrameFixture::FrameFacts fixedFacts;
                        fixedFacts.renderExtent = {render.x, render.y, render.width, render.height};
                        fixedFacts.fullFrameExtent = {full.x, full.y, full.width, full.height};
                        fixedFacts.pixelDefinition = fixedFacts.fullFrameExtent;
                        fixedFacts.pixelSizeUm = scalar(v.at("pixel_size_um"));
                        fixedFacts.frameTime = wide(v.at("frame_time"));
                        fixedFacts.frameRate = wide(v.at("frame_rate"));
                        fixedFacts.sessionSeed = v.at("session_seed");
                        fixedFacts.clipToken = v.at("clip_token");
                        const auto fixed = capture_prepared_frame_descriptor(runtime, state, FrameObservationKind::Grain, fixedFacts);
                        if (fixed.status.category != FJ_STATUS_SUCCESS || !fixed.grain ||
                            JuicerTestSupport::grain_frame_fields(*fixed.grain) != row.at("native"))
                            throw std::runtime_error(v.at("name").get<std::string>() + ": held grain descriptor differs from the captured fixed-fact row");
                    }
                }
                ++runs;
                unsigned abortCalls = 0;
                require_cuda(cudaMemcpy(device.destination, canary.data(), kBytes, cudaMemcpyHostToDevice), "grain caller canary");
                JuicerCuda::ExecutorTest::reset_grain_frame_observation();
                JuicerCuda::ExecutorTest::FailureObservation events;
                JuicerCuda::ExecutorTest::ResourceFailures resources;
                bool success = row.at("success");
                if (actualHeld)
                    success = !params.values.grain.active || actualExpected->status.category == FJ_STATUS_SUCCESS;
                const bool early = !success;
                const bool observeNoResourceWork = early ||
                                                   (path == GrainCallerPath::StopBeforePreparation && success);
                if (actualHeld && observeNoResourceWork) {
                    resources.lutPosition = 1;
                    JuicerCuda::ExecutorTest::observation = &events;
                    JuicerCuda::ExecutorTest::resourceFailures = &resources;
                }
                FjStatus result{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                std::string failureText;
                bool recoveryPending = false;
                FrameDescriptorObservation frameMath{};
                JuicerCuda::ExecutorTest::GrainFrameObservation observed{};
                if (actualHeld) {
                    AbortAfterNativeDescriptor stagedAbort{};
                    const FjAbortCallback actualAbort{&abort_after_native_descriptor, &stagedAbort};
                    const FjAbortCallback selectedAbort = path == GrainCallerPath::Pixels ||
                                                                  (!success && path != GrainCallerPath::BeforeBuilder)
                                                              ? FjAbortCallback{}
                                                              : actualAbort;
                    const FjCallCompletionResult rendered = render_held_prepared_state(
                        runtime, state, prepared_raw_frame(frame), 0x645600 + route, selectedAbort, &frameMath, &recoveryPending, failureText);
                    result = rendered.outcome.status;
                    abortCalls = selectedAbort.query == &abort_after_native_descriptor ? stagedAbort.calls : 0;
                    observed = JuicerCuda::ExecutorTest::grain_frame_observation();
                } else {
                    if (genuine) {
                        // Compare the captured fixed facts with the held product.
                        JuicerCudaTest::PreparedFrameFixture::FrameFacts fixedFacts;
                        fixedFacts.renderExtent = {render.x, render.y, render.width, render.height};
                        fixedFacts.fullFrameExtent = {full.x, full.y, full.width, full.height};
                        fixedFacts.pixelDefinition = fixedFacts.fullFrameExtent;
                        fixedFacts.pixelSizeUm = scalar(v.at("pixel_size_um"));
                        fixedFacts.frameTime = wide(v.at("frame_time"));
                        fixedFacts.frameRate = wide(v.at("frame_rate"));
                        fixedFacts.sessionSeed = v.at("session_seed");
                        fixedFacts.clipToken = v.at("clip_token");
                        const auto fixed = capture_prepared_frame_descriptor(runtime, state, FrameObservationKind::Grain, fixedFacts);
                        if (fixed.status.category != FJ_STATUS_SUCCESS || !fixed.grain ||
                            JuicerTestSupport::grain_frame_fields(*fixed.grain) != row.at("native"))
                            throw std::runtime_error("held grain descriptor differs from the captured fixed-fact row");
                    } else {
                        const auto raw = prepare_literal_frame_construction(
                            FrameObservationKind::Grain, v, static_cast<Spektrafilm::ScanRoute>(route), scalar(v.at("pixel_size_um")), &build_literal_grain_math);
                        if (raw.rawFacadeCalls != 1 || raw.status.category != (success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE) ||
                            (success && (!raw.grain || JuicerTestSupport::grain_frame_fields(*raw.grain) != row.at("native"))) ||
                            (!success && raw.diagnostic != row.at("diagnostic").get<std::string>()))
                            throw std::runtime_error(v.at("name").get<std::string>() + ": raw grain facade/native composer witness changed");
                        if (!success)
                            ++failures;
                        return;
                    }
                    return;
                }
                JuicerCuda::ExecutorTest::observation = nullptr;
                JuicerCuda::ExecutorTest::resourceFailures = nullptr;
                const auto name = genuine ? row.at("spec").dump() : v.at("name").get<std::string>();
                if (frameMath.callbackCalls != 1u || frameMath.providerCalls != 1u) {
                    std::string message = name;
                    message += ": actual prepared-state callback/provider visit count changed callback=";
                    message += std::to_string(frameMath.callbackCalls);
                    message += " provider=";
                    message += std::to_string(frameMath.providerCalls);
                    message += " status=";
                    message += std::to_string(result.category);
                    message += " diagnostic=";
                    message += failureText;
                    throw std::runtime_error(message);
                }
                const auto verifyGrainWitness = [&](const JuicerCuda::ExecutorTest::GrainFrameObservation& witness) {
                    const Json expected = actualExpected && actualExpected->grain
                                              ? JuicerTestSupport::grain_frame_fields(*actualExpected->grain)
                                              : Json{};
                    if (witness.count != 1 || static_cast<std::uint32_t>(witness.route) != params.values.scan_route ||
                        fields(witness.renderExtent) != actualInput.at("render_extent") ||
                        fields(witness.fullFrameExtent) != actualInput.at("full_frame_extent") ||
                        std::bit_cast<std::uint32_t>(witness.pixelSizeUm) != actualInput.at("pixel_size_um") ||
                        std::bit_cast<std::uint64_t>(witness.frameTime) != actualInput.at("frame_time") ||
                        std::bit_cast<std::uint64_t>(witness.frameRate) != actualInput.at("frame_rate") ||
                        witness.sessionSeed != actualInput.at("session_seed") || witness.clipToken != actualInput.at("clip_token") ||
                        witness.success != success)
                        throw std::runtime_error(name + ": current callback grain facts");
                    const bool present = success && actualExpected && actualExpected->grain.has_value();
                    if (witness.descriptor.has_value() != present ||
                        (present && JuicerTestSupport::grain_frame_fields(*witness.descriptor) != expected))
                        throw std::runtime_error(name + ": complete successful grain fields/hash or absent failed/inactive descriptor");
                };
                if (path == GrainCallerPath::StopBeforePreparation && success) {
                    JuicerCuda::ExecutorTest::reset_grain_frame_observation();
                    if (genuine) {
                        JuicerCudaTest::PreparedFrameFixture::FrameFacts witnessFacts;
                        witnessFacts.renderExtent = {full.x, full.y, full.width, full.height};
                        witnessFacts.fullFrameExtent = {full.x, full.y, full.width, full.height};
                        witnessFacts.pixelDefinition = witnessFacts.fullFrameExtent;
                        witnessFacts.pixelSizeUm = frame.pixelSizeUm;
                        witnessFacts.frameTime = wide(v.at("frame_time"));
                        witnessFacts.frameRate = wide(v.at("frame_rate"));
                        witnessFacts.sessionSeed = v.at("session_seed");
                        witnessFacts.clipToken = v.at("clip_token");
                        FrameConstructionResult witness{FrameObservationKind::Grain};
                        std::string witnessDiagnostic;
                        if (!JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                                runtime.borrow().assets, state, witnessFacts, &witness, &capture_held_frame_construction, witnessDiagnostic)) {
                            std::string message = name;
                            message += ": held-state construction witness: ";
                            message += witnessDiagnostic;
                            throw std::runtime_error(message);
                        }
                        if (witness.status.category != FJ_STATUS_SUCCESS || !witness.grain ||
                            JuicerTestSupport::grain_frame_fields(*witness.grain) != JuicerTestSupport::grain_frame_fields(*actualExpected->grain))
                            throw std::runtime_error(name + ": held grain construction witness fields/hash changed");
                    }
                }
                if (observed.count != 1u)
                    throw std::runtime_error(name + ": grain helper invocation count");
                verifyGrainWitness(observed);
                const auto expectedStatus = early ? FJ_STATUS_PREPARATION_FAILURE : path == GrainCallerPath::Pixels ? FJ_STATUS_SUCCESS
                                                                                                                    : FJ_STATUS_CANCELLED;
                const unsigned expectedAbortCalls = !early && path == GrainCallerPath::StopBeforePreparation ? 3u : 0u;
                if (result.category != expectedStatus || result.api != 0 || result.native_code != 0 || recoveryPending || abortCalls != expectedAbortCalls)
                    throw std::runtime_error(std::string(name).append(": grain caller stage/status: ").append(failureText));
                if (early) {
                    const auto expectedText = actualExpected->diagnostic;
                    if (failureText != expectedText || events.count != 0 || !events.armed || resources.lutAttempts != 0 || resources.stbnAttempts != 0 || resources.wangAttempts != 0 || resources.injections != 0)
                        throw std::runtime_error(name + ": failed grain reached preparation/upload/launch");
                    ++failures;
                } else if (path == GrainCallerPath::StopBeforePreparation &&
                           (events.count != 0 || resources.lutAttempts != 0 || resources.stbnAttempts != 0 ||
                            resources.wangAttempts != 0 || resources.injections != 0)) {
                    throw std::runtime_error(name + ": cancellation entered resource preparation");
                }
                std::vector<float> actual(kSamples);
                require_cuda(cudaMemcpyAsync(actual.data(), device.destination, kBytes, cudaMemcpyDeviceToHost, device.stream), "grain caller readback");
                require_cuda(cudaStreamSynchronize(device.stream), "grain caller completion");
                if (path == GrainCallerPath::Pixels) {
                    if (!genuine || early || actual == canary)
                        throw std::runtime_error("genuine grain pixel execution missing");
                    const auto count = static_cast<std::size_t>(full.width) * static_cast<std::size_t>(full.height) * 4;
                    if (count > actual.size())
                        throw std::runtime_error("grain intended pixel extent exceeds captured output");
                    std::size_t visited = 0;
                    for (std::size_t i = 0; i < count; ++i) {
                        if (i % 4 == 3 ? std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(0.18f) : !std::isfinite(actual[i]) || actual[i] < 0.0f || actual[i] > 1.0f)
                            throw std::runtime_error("grain genuine pixel/alpha contract");
                        ++visited;
                    }
                    if (visited != count)
                        throw std::runtime_error("grain scalar comparison visit count mismatch");
                    ++pixels;
                } else {
                    if (actual != canary)
                        throw std::runtime_error("grain construction-only execution wrote pixels");
                    if (!early)
                        ++heldConstructions;
                }
                std::cout << "grain caller route=" << route << " name=" << name << " pixels=" << (path == GrainCallerPath::Pixels) << " fallback=" << fullFallback << " builder_calls=" << observed.count << " success=" << success << " passed\n";
            };
            for (const auto& row : captured.at("products")) {
                if (row.at("route") != route)
                    continue;
                auto case_event = rows.case_event("route-" + std::to_string(route) + "/" + grain_frame_product_id(row.at("spec")));
                const auto& spec = row.at("spec");
                const bool pixel = !spec.at("active").get<bool>() || (spec.at("debug") == 0 && !spec.contains("amplitude"));
                if (!spec.at("active").get<bool>() && fj_test_grain_frame_fault(1).category != 0)
                    throw std::runtime_error("arm inactive delegation witness");
                run(row, pixel ? GrainCallerPath::Pixels : GrainCallerPath::StopBeforePreparation);
                if (!spec.at("active").get<bool>()) {
                    FjGrainFrameInput active{};
                    active.active = 1;
                    FjGrainFrameMath output{};
                    FjGrainFrameFailure failure{};
                    if (fj_test_grain_frame_prepare(&active, &output, &failure, nullptr).category != 8 || failure.reason != 3)
                        throw std::runtime_error("inactive executor consumed Rust frame mathematics");
                }
                if (spec.at("active").get<bool>() && !spec.at("layers").get<bool>() && !spec.contains("amplitude")) {
                    run(row, GrainCallerPath::BeforeBuilder);
                    run(row, GrainCallerPath::StopBeforePreparation, true);
                }
                case_event.passed();
            }
            for (const auto& row : captured.at("cases")) {
                const auto& v = row.at("input");
                const auto name = v.at("name").get<std::string>();
                const auto logical = name.starts_with("positive/") ? name.substr(9) : name;
                if (v.at("capture_polarity") == route / 2 && v.at("film_polarity") == route / 2 && std::find(caseNames.begin(), caseNames.end(), logical) != caseNames.end()) {
                    auto case_event = rows.case_event("route-" + std::to_string(route) + "/case/" + name);
                    run(row, GrainCallerPath::StopBeforePreparation);
                    case_event.passed();
                }
            }
            if (fj_cuda_retire_instance(runtime.borrow().cuda, 0x645600 + route, nullptr).category != 0)
                throw std::runtime_error("grain caller membership retirement");
        }
        if (runs != 122 || pixels != 16 || heldConstructions != 28 || failures != 10)
            throw std::runtime_error("grain caller exact pixel/failure/stop membership");
        if (actualCompanions != 44)
            throw std::runtime_error("grain actual full-frame companion membership");
        std::cout << "grain caller runs=" << runs << " genuine pixels=" << pixels << " held constructions=" << heldConstructions << " failures=" << failures << " actual_full_frame_companions=" << actualCompanions << " passed\n";
    }

    void run_dir_frame_cases(CallbackSession& runtime, const std::filesystem::path& manifestPath, const std::string& preset) {
        using Json = nlohmann::json;
        const auto manifest = Json::parse(std::ifstream(manifestPath));
        const auto captured = load_shared_fixture(manifestPath.parent_path() / manifest.at("presets").at(preset).at("native").get<std::string>(), "manifest.json");
        if (captured.at("preset") != preset || captured.at("cases").size() != 74 || captured.at("products").size() != 24) {
            throw std::runtime_error("DIR caller capture membership changed");
        }
        const std::array caseNames{"inactive", "size-zero", "size-nan", "core-word-1", "core-word-1042983595", "fir", "core-word-1077936128", "tail-mixed", "tail-weight-one", "tail-zero", "pixel-change", "negative-origin", "pixel-zero", "offset", "tail-radius", "core-inf", "active-hash-zero"};
        std::vector<std::string> intended;
        intended.reserve(92);
        for (unsigned route = 0; route < 4; ++route) {
            for (const auto& row : captured.at("products"))
                if (row.at("route") == route)
                    intended.push_back("route-" + std::to_string(route) + "/product/" + row.at("input").at("name").get<std::string>());
            for (const char* name : caseNames)
                intended.push_back("route-" + std::to_string(route) + "/case/" + name);
        }
        JuicerTests::Cohort rows("dir_frame", intended);
        const auto extent = [](const Json& value) -> Spektrafilm::DirFrameExtent {
            return {value.at(0), value.at(1), value.at(2), value.at(3)};
        };
        const auto rect = [](Spektrafilm::DirFrameExtent value) -> JuicerCuda::FrameRect {
            return {value.x, value.y, value.x + value.width, value.y + value.height};
        };
        const auto fields = [](Spektrafilm::DirFrameExtent value) -> Json {
            return {value.x, value.y, value.width, value.height};
        };
        const auto scalar = [](const Json& value) {
            return std::bit_cast<float>(value.get<std::uint32_t>());
        };
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "DIR caller stream");
        constexpr std::size_t kSamples = std::size_t{96} * 71 * 4;
        constexpr std::size_t kBytes = kSamples * sizeof(float);
        require_cuda(cudaMalloc(&device.source, kBytes), "DIR caller source");
        require_cuda(cudaMalloc(&device.destination, kBytes), "DIR caller destination");
        const std::vector<float> source(kSamples, 0.18f), canary(kSamples, kCanary);
        require_cuda(cudaMemcpy(device.source, source.data(), kBytes, cudaMemcpyHostToDevice), "DIR caller input");
        unsigned runs = 0, actualCompanions = 0;
        for (unsigned route = 0; route < 4; ++route) {
            JuicerTestSupport::PreparationInput params;
            params.values.scan_route = route;
            params.film_key = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.values.grain.active = false;
            params.values.dir_couplers.diffusion_size_um = 20.0f;
            params.values.dir_couplers.diffusion_tail_um = 200.0f;
            params.values.dir_couplers.diffusion_tail_weight = 0.03f;
            JuicerTestSupport::PreparedStateHold state = JuicerTestSupport::prepare_product(runtime.borrow().assets, params);
            std::array<char, 1024> stateErrorText{};
            FjErrorBuffer stateError{stateErrorText.data(), stateErrorText.size(), 0};
            FjPreparedStateView stateView{};
            if (state.view(stateView, stateError).category != FJ_STATUS_SUCCESS)
                throw std::runtime_error(std::string(stateErrorText.data(), stateError.length));
            const auto run = [&](const Json& row, bool genuine, bool beforeBuilder, bool executePixels, bool fullFallback) {
                const auto& input = row.at("input");
                const auto name = input.at("name").get<std::string>() + "/route-" + std::to_string(route);
                if (genuine && (row.at("recipe_hash") != stateView.recipe_hash || input.at("hash") != stateView.dir_recipe.hash)) {
                    throw std::runtime_error(name + ": genuine recipe identity changed");
                }
                const auto render = extent(input.at("render")), full = extent(input.at("full"));
                auto expected = row.at("native");
                if (!expected.at("label").is_null())
                    expected["label"] = route % 2 ? "print" : "direct";
                const auto sourceDomain = full.width > 0 && full.height > 0 ? full : Spektrafilm::DirFrameExtent{0, 0, 64, 48};
                const auto sourceBounds = rect(sourceDomain);
                const auto effectiveFull = fullFallback || full.width <= 0 || full.height <= 0 ? sourceDomain : full;
                const auto fullBounds = fullFallback ? JuicerCuda::FrameRect{} : rect(full);
                auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    sourceBounds, sourceBounds, static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
                const std::optional<ScatterHalationFrameDescriptor> halation;
                const Spektrafilm::FilmJuicerEffectsGeometry geometry{};
                const auto rowBytes = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(sourceDomain.width) * 4 * sizeof(float));
                JuicerCuda::ExecutionFrame frame{sourceBounds, rect(render), fullBounds, reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<unsigned char*>(device.destination), rowBytes, rowBytes, beforeBuilder ? 2 : 4, device.stream, diffusion, halation, geometry, scalar(input.at("pixel")), 37.0, 24.0, 0x20260923, 0x5312, meter, false, false};
                const bool actualHeld = genuine && !beforeBuilder && !fullFallback;
                actualCompanions += actualHeld ? 1u : 0u;
                Json actualInput = input;
                auto actualFull = effectiveFull;
                auto actualRender = render;
                std::optional<FrameConstructionResult> actualExpected;
                if (actualHeld) {
                    const FjExtent policyExtent{effectiveFull.x, effectiveFull.y, effectiveFull.width, effectiveFull.height};
                    const float pixelSizeUm = prepared_frame_pixel_size_um(state, policyExtent);
                    actualRender = effectiveFull;
                    actualInput["render"] = Json::array({actualRender.x, actualRender.y, actualRender.width, actualRender.height});
                    actualInput["full"] = Json::array({actualFull.x, actualFull.y, actualFull.width, actualFull.height});
                    actualInput["pixel"] = std::bit_cast<std::uint32_t>(pixelSizeUm);
                    JuicerCudaTest::PreparedFrameFixture::FrameFacts actualFacts;
                    actualFacts.renderExtent = {actualRender.x, actualRender.y, actualRender.width, actualRender.height};
                    actualFacts.fullFrameExtent = actualFacts.renderExtent;
                    actualFacts.pixelDefinition = actualFacts.fullFrameExtent;
                    actualFacts.pixelSizeUm = pixelSizeUm;
                    actualFacts.frameTime = 37.0;
                    actualFacts.frameRate = 24.0;
                    actualFacts.sessionSeed = 0x20260923;
                    actualFacts.clipToken = 0x5312;
                    actualExpected.emplace(capture_prepared_frame_descriptor(runtime, state, FrameObservationKind::Dir, actualFacts));
                    frame.renderWindow = rect(actualRender);
                    frame.fullFrameExtent = rect(actualFull);
                    frame.pixelSizeUm = pixelSizeUm;
                    meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                        rect(actualFull), rect(actualFull), static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                    // Compare the captured fixed facts with the held product and raw facade.
                    if (row.at("success").get<bool>()) {
                        JuicerCudaTest::PreparedFrameFixture::FrameFacts fixedFacts;
                        fixedFacts.renderExtent = {render.x, render.y, render.width, render.height};
                        fixedFacts.fullFrameExtent = {effectiveFull.x, effectiveFull.y, effectiveFull.width, effectiveFull.height};
                        fixedFacts.pixelDefinition = fixedFacts.fullFrameExtent;
                        fixedFacts.pixelSizeUm = scalar(input.at("pixel"));
                        fixedFacts.frameTime = 37.0;
                        fixedFacts.frameRate = 24.0;
                        fixedFacts.sessionSeed = 0x20260923;
                        fixedFacts.clipToken = 0x5312;
                        const auto fixed = capture_prepared_frame_descriptor(runtime, state, FrameObservationKind::Dir, fixedFacts);
                        if (fixed.status.category != FJ_STATUS_SUCCESS || !fixed.dir ||
                            JuicerTestSupport::dir_frame_fields(*fixed.dir) != expected)
                            throw std::runtime_error(name + ": held DIR descriptor differs from the captured fixed-fact row");
                    } else {
                        const auto fixed = prepare_literal_frame_construction(
                            FrameObservationKind::Dir, input, static_cast<Spektrafilm::ScanRoute>(route), scalar(input.at("pixel")), &build_literal_dir_math);
                        if (fixed.rawFacadeCalls != 1 || fixed.status.category != FJ_STATUS_PREPARATION_FAILURE ||
                            !fixed.dirFailure || fixed.dirFailure->reason != FJ_DIR_FRAME_PREPARATION ||
                            fixed.dirFailure->component != input.at("expected_component").get<std::uint32_t>() ||
                            fixed.dirFailure->field != input.at("expected_field").get<std::uint32_t>())
                            throw std::runtime_error(name + ": raw DIR failure differs from the captured fixed-fact row");
                    }
                }
                ++runs;
                unsigned abortCalls = 0;
                require_cuda(cudaMemcpy(device.destination, canary.data(), kBytes, cudaMemcpyHostToDevice), "DIR caller canary");
                JuicerCuda::ExecutorTest::reset_dir_frame_observation();
                JuicerCuda::ExecutorTest::FailureObservation failureEvents;
                JuicerCuda::ExecutorTest::ResourceFailures resources;
                bool success = row.at("success");
                if (actualHeld)
                    success = actualExpected->status.category == FJ_STATUS_SUCCESS;
                const bool earlyFailure = beforeBuilder || !success;
                const bool observeNoResourceWork = earlyFailure || (!executePixels && !beforeBuilder && success);
                if (actualHeld && observeNoResourceWork) {
                    resources.lutPosition = 1;
                    JuicerCuda::ExecutorTest::observation = &failureEvents;
                    JuicerCuda::ExecutorTest::resourceFailures = &resources;
                }
                FjStatus status{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                std::string failureText;
                bool recoveryPending = false;
                FrameDescriptorObservation frameMath{};
                JuicerCuda::ExecutorTest::DirFrameObservation observed{};
                if (actualHeld) {
                    AbortAfterNativeDescriptor stagedAbort{};
                    const FjAbortCallback actualAbort{&abort_after_native_descriptor, &stagedAbort};
                    const FjAbortCallback selectedAbort = executePixels || (!success && !beforeBuilder)
                                                              ? FjAbortCallback{}
                                                              : actualAbort;
                    const FjCallCompletionResult rendered = render_held_prepared_state(
                        runtime, state, prepared_raw_frame(frame), 0x641300 + route, selectedAbort, &frameMath, &recoveryPending, failureText);
                    status = rendered.outcome.status;
                    abortCalls = selectedAbort.query == &abort_after_native_descriptor ? stagedAbort.calls : 0;
                    observed = JuicerCuda::ExecutorTest::dir_frame_observation();
                } else {
                    const auto raw = prepare_literal_frame_construction(
                        FrameObservationKind::Dir, input, static_cast<Spektrafilm::ScanRoute>(route), scalar(input.at("pixel")), &build_literal_dir_math);
                    if (raw.rawFacadeCalls != 1 || raw.status.category != (success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE) ||
                        (success && (!raw.dir || JuicerTestSupport::dir_frame_fields(*raw.dir) != expected)) ||
                        (!success && (!raw.dirFailure || raw.dirFailure->reason != FJ_DIR_FRAME_PREPARATION ||
                                      raw.dirFailure->component != input.at("expected_component").get<std::uint32_t>() ||
                                      raw.dirFailure->field != input.at("expected_field").get<std::uint32_t>())))
                        throw std::runtime_error(name + ": raw DIR facade/native composer witness changed");
                    return;
                }
                JuicerCuda::ExecutorTest::observation = nullptr;
                JuicerCuda::ExecutorTest::resourceFailures = nullptr;
                if (frameMath.callbackCalls != 1u || frameMath.providerCalls != 1u) {
                    std::string message = name;
                    message += ": actual prepared-state callback/provider visit count changed callback=";
                    message += std::to_string(frameMath.callbackCalls);
                    message += " provider=";
                    message += std::to_string(frameMath.providerCalls);
                    message += " status=";
                    message += std::to_string(status.category);
                    message += " diagnostic=";
                    message += failureText;
                    throw std::runtime_error(message);
                }
                const auto verifyDirWitness = [&](const JuicerCuda::ExecutorTest::DirFrameObservation& witness,
                                                  bool expectedSuccess) {
                    const Json expectedFields = actualExpected && actualExpected->dir
                                                    ? JuicerTestSupport::dir_frame_fields(*actualExpected->dir)
                                                    : expected;
                    if (witness.count != 1 || static_cast<std::uint32_t>(witness.route) != params.values.scan_route ||
                        witness.pixelSizeBits != actualInput.at("pixel") ||
                        fields(witness.renderExtent) != actualInput.at("render") ||
                        fields(witness.fullFrameExtent) != fields(actualFull) ||
                        witness.success != expectedSuccess ||
                        JuicerTestSupport::dir_frame_fields(witness.descriptor) != expectedFields)
                        throw std::runtime_error(name + ": actual builder facts, fields or final hash changed");
                };
                if (!executePixels && success) {
                    JuicerCudaTest::PreparedFrameFixture::FrameFacts witnessFacts;
                    witnessFacts.renderExtent = {actualRender.x, actualRender.y, actualRender.width, actualRender.height};
                    witnessFacts.fullFrameExtent = {actualFull.x, actualFull.y, actualFull.width, actualFull.height};
                    witnessFacts.pixelDefinition = witnessFacts.fullFrameExtent;
                    witnessFacts.pixelSizeUm = frame.pixelSizeUm;
                    witnessFacts.frameTime = 37.0;
                    witnessFacts.frameRate = 24.0;
                    witnessFacts.sessionSeed = 0x20260923;
                    witnessFacts.clipToken = 0x5312;
                    FrameConstructionResult witness{FrameObservationKind::Dir};
                    std::string witnessDiagnostic;
                    if (!JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                            runtime.borrow().assets, state, witnessFacts, &witness, &capture_held_frame_construction, witnessDiagnostic)) {
                        std::string message = name;
                        message += ": held-state construction witness: ";
                        message += witnessDiagnostic;
                        throw std::runtime_error(message);
                    }
                    if (witness.status.category != FJ_STATUS_SUCCESS || !witness.dir || !actualExpected->dir ||
                        JuicerTestSupport::dir_frame_fields(*witness.dir) != JuicerTestSupport::dir_frame_fields(*actualExpected->dir))
                        throw std::runtime_error(name + ": held DIR construction witness fields/hash changed");
                }
                if (observed.count != 1u) {
                    throw std::runtime_error(name + ": actual builder facts, fields or final hash changed");
                }
                verifyDirWitness(observed, success);
                const auto expectedStatus = earlyFailure ? FJ_STATUS_PREPARATION_FAILURE : executePixels ? FJ_STATUS_SUCCESS
                                                                                                         : FJ_STATUS_CANCELLED;
                if (status.category != expectedStatus || status.api != FJ_API_NONE || status.native_code != 0 || recoveryPending ||
                    abortCalls != (!earlyFailure && !executePixels ? 3u : 0u)) {
                    std::string message = name;
                    message += ": caller did not stop at the expected stage: ";
                    message += failureText;
                    throw std::runtime_error(message);
                }
                if (earlyFailure) {
                    const std::string expectedText = actualExpected->diagnostic;
                    if (failureText != expectedText || failureEvents.count != 0 || !failureEvents.armed ||
                        resources.lutAttempts != 0 || resources.stbnAttempts != 0 || resources.wangAttempts != 0 || resources.injections != 0) {
                        throw std::runtime_error(name + ": failed mathematics reached resource preparation or launch");
                    }
                } else if (!executePixels && !beforeBuilder &&
                           (failureEvents.count != 0 || resources.lutAttempts != 0 || resources.stbnAttempts != 0 ||
                            resources.wangAttempts != 0 || resources.injections != 0)) {
                    throw std::runtime_error(name + ": cancellation entered resource preparation");
                }
                if (!executePixels) {
                    std::vector<float> actual(kSamples);
                    require_cuda(cudaMemcpyAsync(actual.data(), device.destination, kBytes, cudaMemcpyDeviceToHost, device.stream), "DIR caller output canary");
                    require_cuda(cudaStreamSynchronize(device.stream), "DIR caller completion");
                    if (actual != canary) {
                        throw std::runtime_error(name + ": stopped caller wrote pixels");
                    }
                } else {
                    require_cuda(cudaStreamSynchronize(device.stream), "DIR caller rendered completion");
                }
                std::cout << "DIR caller " << name << " fallback=" << fullFallback << " before=" << beforeBuilder << " pixels=" << executePixels << " passed\n";
            };
            for (const auto& row : captured.at("products")) {
                if (row.at("route") == route) {
                    auto case_event = rows.case_event("route-" + std::to_string(route) + "/product/" + row.at("input").at("name").get<std::string>());
                    const bool executePixels = row.at("input").at("name") == std::to_string(route) + "/pixel-8";
                    run(row, true, false, executePixels, false);
                    if (executePixels) {
                        run(row, true, true, false, false);
                        run(row, true, false, false, true);
                    }
                    case_event.passed();
                }
            }
            for (const char* name : caseNames) {
                auto case_event = rows.case_event("route-" + std::to_string(route) + "/case/" + name);
                const auto found = std::find_if(captured.at("cases").begin(), captured.at("cases").end(), [&](const auto& row) {
                    return row.at("input").at("name") == name;
                });
                if (found == captured.at("cases").end()) {
                    throw std::runtime_error("DIR caller literal missing");
                }
                run(*found, false, false, false, false);
                case_event.passed();
            }
        }
        if (runs != 100) {
            throw std::runtime_error("DIR caller exact membership changed");
        }
        if (actualCompanions != 24)
            throw std::runtime_error("DIR actual full-frame companion membership");
        std::cout << "DIR caller fixed_membership_runs=" << runs << " actual_full_frame_companions=" << actualCompanions << " passed\n";
    }

    void check_failure_order(CallbackSession& runtime, const Case& test, JuicerCuda::ExecutorTest::Delivery delivery, bool deferredDirFailure, FjStatus failureStatus, const char* wording) {
        const bool contextLoss = failureStatus.category == FJ_STATUS_CONTEXT_LOSS;
        using JuicerCuda::ExecutorTest::Event;
        JuicerCuda::ExecutorTest::FailureObservation observation;
        observation.delivery = delivery;
        observation.status = failureStatus;
        observation.diagnostic = wording;
        observation.scanResult = deferredDirFailure
                                     ? JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::DeferredDirFailure
                                     : JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed;
        observation.stage = Spektrafilm::scan_route_is_print(test.route)
                                ? "print_scan_error_stage"
                                : "direct_scan_error_stage";
        JuicerCuda::ExecutorTest::observation = &observation;
        JuicerTestSupport::InstanceFixture state;
        MembershipRetirement stateRetirement{runtime, state};
        bool fatal = false;
        try {
            (void)render_case(runtime, test, parameters_for(test), state);
        } catch (const CallbackStatus& error) {
            fatal = error.status() == kOfxStatErrFatal;
            observation.record(Event::FatalMapped);
        } catch (...) {
            JuicerCuda::ExecutorTest::observation = nullptr;
            throw;
        }
        JuicerCuda::ExecutorTest::observation = nullptr;
        std::vector<Event> expected{Event::Injected, Event::Classified, Event::FrameAborted};
        if (deferredDirFailure) {
            expected.push_back(Event::Message);
        }
        expected.insert(expected.end(), {Event::RecoveryStarted, Event::RecoveryEnded, Event::FatalMapped});
        if (!fatal || observation.overflow || observation.count != expected.size() ||
            !std::equal(expected.begin(), expected.end(), observation.events.begin()) ||
            !observation.failureStageMatches || !observation.failureStatusMatches ||
            observation.nativeRecoveryPending != contextLoss ||
            observation.adapterRecoveryPending != contextLoss ||
            (deferredDirFailure && !observation.messageMatches) ||
            runtime.submission(state).valid == contextLoss) {
            throw std::runtime_error(std::string(test.name) + ": executor failure order or host mapping changed");
        }
        std::cout << test.name << " dir=" << deferredDirFailure << " context_loss=" << contextLoss
                  << " delivery=" << static_cast<int>(delivery)
                  << " classify/abort/message/recovery/fatal order passed\n";
    }

    void check_mismatching_recovery(CallbackSession& runtime, const Case& test) {
        using namespace JuicerCuda::ExecutorTest;
        CUcontext original = nullptr;
        CUcontext other = nullptr;
        if (cuCtxGetCurrent(&original) != CUDA_SUCCESS || !original || cuCtxCreate(&other, nullptr, 0, 0) != CUDA_SUCCESS ||
            cuCtxSetCurrent(original) != CUDA_SUCCESS)
            throw std::runtime_error("create mismatch recovery contexts failed");
        struct HookState {
            CallbackSession* runtime;
            const Case* test;
            JuicerTestSupport::InstanceFixture* instance;
            CUcontext original;
            CUcontext other;
            std::exception_ptr failure;
            JuicerTestSupport::SubmissionObservation selected;
        } hook{&runtime, &test, nullptr, original, other, {}, {}};
        JuicerTestSupport::InstanceFixture state;
        MembershipRetirement stateRetirement{runtime, state};
        hook.instance = &state;
        FailureObservation failure;
        failure.delivery = Delivery::Success;
        failure.status = JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED);
        failure.diagnostic = "mismatching context recovery";
        failure.scanResult = JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed;
        failure.stage = Spektrafilm::scan_route_is_print(test.route) ? "print_scan_error_stage" : "direct_scan_error_stage";
        observation = &failure;
        recoveryStartHookUser = &hook;
        recoveryStartHook = [](void* user) noexcept {
            auto& data = *static_cast<HookState*>(user);
            auto* saved = observation;
            observation = nullptr;
            try {
                if (cuCtxSetCurrent(data.other) != CUDA_SUCCESS)
                    throw std::runtime_error("set nested recovery context failed");
                (void)render_case(*data.runtime, *data.test, parameters_for(*data.test), *data.instance);
                data.selected = data.runtime->submission(*data.instance);
            } catch (...) {
                data.failure = std::current_exception();
            }
            if (cuCtxSetCurrent(data.original) != CUDA_SUCCESS && !data.failure)
                data.failure = std::make_exception_ptr(std::runtime_error("restore recovery context failed"));
            observation = saved;
        };
        bool fatal = false;
        try {
            (void)render_case(runtime, test, parameters_for(test), state);
        } catch (const CallbackStatus& error) {
            fatal = error.status() == kOfxStatErrFatal;
        } catch (...) {
            hook.failure = std::current_exception();
        }
        recoveryStartHook = nullptr;
        recoveryStartHookUser = nullptr;
        observation = nullptr;
        if (hook.failure)
            std::rethrow_exception(hook.failure);
        const auto after = runtime.submission(state);
        if (!fatal || !hook.selected.valid || !after.valid || after.context.context != reinterpret_cast<std::uintptr_t>(other) ||
            after.submission.submission_id != hook.selected.submission.submission_id)
            throw std::runtime_error("old-context recovery cleared the new-context submission");
        if (cuCtxSetCurrent(other) != CUDA_SUCCESS)
            throw std::runtime_error("restore nested context failed");
        (void)render_case(runtime, test, parameters_for(test), state);
        const auto reused = runtime.submission(state);
        if (!reused.valid || reused.submission.submission_id != after.submission.submission_id)
            throw std::runtime_error("new-context submission was not reused after old-context recovery");
        std::string diagnostic;
        {
            JuicerCuda::NativeCall admission(runtime.borrow().cuda);
            if (!JuicerCuda::borrow_root(runtime.borrow().cuda).retire_idle_context(0, other, diagnostic))
                throw std::runtime_error(diagnostic);
        }
        if (cuCtxDestroy(other) != CUDA_SUCCESS || cuCtxSetCurrent(original) != CUDA_SUCCESS)
            throw std::runtime_error("release mismatch recovery context failed");
        if (!stateRetirement.close())
            throw std::runtime_error("retire mismatch recovery instance failed");
    }

    void check_recovery_interference(CallbackSession& runtime, const Case& test, bool holdGate) {
        using namespace JuicerCuda::ExecutorTest;
        JuicerTestSupport::InstanceFixture state;
        MembershipRetirement stateRetirement{runtime, state};
        struct HookState {
            FjCuda* cuda = nullptr;
            CallbackSession* runtime = nullptr;
            JuicerTestSupport::InstanceFixture* instance = nullptr;
            bool holdGate = false;
            std::optional<JuicerCuda::NativeCall> gate;
            JuicerCuda::ResourceManager::DeviceContextKey key{};
            bool registryHeld = false;
            bool failed = false;
            JuicerTestSupport::SubmissionObservation selected{};
            bool release() noexcept {
                gate.reset();
                if (!std::exchange(registryHeld, false))
                    return true;
                return JuicerCuda::ResourceManager::registry_note_submission_end(key);
            }
            ~HookState() {
                (void)release();
            }
        } hook;
        hook.cuda = runtime.borrow().cuda;
        hook.runtime = &runtime;
        hook.instance = &state;
        hook.holdGate = holdGate;
        FailureObservation failure;
        failure.delivery = Delivery::Success;
        failure.status = JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED);
        failure.diagnostic = "recovery interference original failure";
        failure.scanResult = JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed;
        failure.stage = Spektrafilm::scan_route_is_print(test.route) ? "print_scan_error_stage" : "direct_scan_error_stage";
        observation = &failure;
        recoveryStartHookUser = &hook;
        recoveryStartHook = [](void* user) noexcept {
            auto& data = *static_cast<HookState*>(user);
            try {
                data.selected = data.runtime->submission(*data.instance);
                if (!data.selected.valid) {
                    data.failed = true;
                    return;
                }
                data.key = {data.selected.context.device_id, reinterpret_cast<void*>(data.selected.context.context)};
                if (data.holdGate) {
                    data.gate.emplace(data.cuda);
                } else {
                    JuicerCuda::ResourceManager::RegistryContextSnapshot snapshot;
                    data.registryHeld = JuicerCuda::ResourceManager::registry_begin_submission(data.key, snapshot);
                    data.failed = !data.registryHeld;
                }
            } catch (...) {
                data.failed = true;
            }
        };
        bool fatal = false;
        try {
            (void)render_case(runtime, test, parameters_for(test), state);
        } catch (const CallbackStatus& error) {
            fatal = error.status() == kOfxStatErrFatal;
            failure.record(Event::FatalMapped);
        } catch (...) {
            recoveryStartHook = nullptr;
            recoveryStartHookUser = nullptr;
            observation = nullptr;
            (void)hook.release();
            throw;
        }
        recoveryStartHook = nullptr;
        recoveryStartHookUser = nullptr;
        observation = nullptr;
        if (!hook.release())
            throw std::runtime_error("recovery interference registry release failed");
        const std::vector<Event> expected = holdGate
                                                ? std::vector<Event>{Event::Injected, Event::Classified, Event::FrameAborted, Event::RecoveryStarted, Event::FatalMapped}
                                                : std::vector<Event>{Event::Injected, Event::Classified, Event::FrameAborted, Event::RecoveryStarted, Event::RecoveryEnded, Event::FatalMapped};
        const auto after = runtime.submission(state);
        const auto& outcome = failure.completionOutcome;
        if (!fatal || hook.failed || !hook.selected.valid || failure.overflow || failure.count != expected.size() ||
            !std::equal(expected.begin(), expected.end(), failure.events.begin()) || !failure.failureStageMatches ||
            !failure.failureStatusMatches || !failure.failureDiagnosticMatches || !failure.nativeRecoveryPending ||
            !failure.adapterRecoveryPending || !failure.completionObserved || !failure.completionFatal ||
            !failure.completionDiagnosticMatches || outcome.status.category != failure.status.category ||
            outcome.status.api != failure.status.api || outcome.status.native_code != failure.status.native_code || outcome.flags != 0 ||
            (holdGate && (failure.retirementObserved || !after.valid || after.submission.submission_id != hook.selected.submission.submission_id)) ||
            (!holdGate && (!failure.retirementObserved || failure.retirementAccepted ||
                           failure.invalidationStatus.category != FJ_STATUS_SUCCESS || !failure.invalidationCleared || after.valid)))
            throw std::runtime_error("recovery interference changed original failure or latch behavior");
        (void)render_case(runtime, test, parameters_for(test), state);
        const auto recovered = runtime.submission(state);
        if (!recovered.valid || (holdGate ? recovered.submission.submission_id != hook.selected.submission.submission_id
                                          : recovered.submission.submission_id <= hook.selected.submission.submission_id))
            throw std::runtime_error("recovery interference subsequent submission behavior changed");
    }

    constexpr std::string_view kDiffusionComponentsPanicCase = "failure-direct/diffusion-components-panic";

    void check_diffusion_components_panic(CallbackSession& runtime, const Case& test) {
        using namespace JuicerCuda::ExecutorTest;
        JuicerTestSupport::InstanceFixture state;
        MembershipRetirement stateRetirement{runtime, state};
        auto parameters = parameters_for(test);
        parameters.values.camera_diffusion.active = true;
        parameters.values.grain.active = false;

        FailureObservation failure;
        failure.armed = false;
        failure.diffusionComponentsPanic = true;
        failure.stage = "prepare_diffusion_resources";
        failure.status = {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        failure.diagnostic = "diffusion components preparation panicked";
        reset_workspace_transitions();
        bool fatal = false;
        {
            struct ClearFailureObservation {
                ~ClearFailureObservation() {
                    observation = nullptr;
                }
            } clear;
            observation = &failure;
            try {
                (void)render_case(runtime, test, parameters, state);
            } catch (const CallbackStatus& error) {
                fatal = error.status() == kOfxStatErrFatal;
                failure.record(Event::FatalMapped);
            }
        }

        constexpr std::array expected{Event::FrameAborted, Event::Classified, Event::RecoveryStarted, Event::RecoveryEnded, Event::FatalMapped};
        const auto& completion = failure.completionOutcome;
        if (!fatal || failure.overflow || failure.count != expected.size() ||
            !std::equal(expected.begin(), expected.end(), failure.events.begin()) ||
            !failure.failureStageMatches || !failure.failureStatusMatches || !failure.failureDiagnosticMatches ||
            !failure.completionObserved || !failure.completionFatal || !failure.completionDiagnosticMatches ||
            completion.status.category != failure.status.category || completion.status.api != failure.status.api ||
            completion.status.native_code != failure.status.native_code || completion.flags != 0 ||
            failure.nativeRecoveryPending || failure.adapterRecoveryPending || failure.retirementObserved) {
            throw std::runtime_error("diffusion components panic propagation changed; category=" +
                                     std::to_string(failure.classifiedStatus.category) + " api=" +
                                     std::to_string(failure.classifiedStatus.api) + " code=" +
                                     std::to_string(failure.classifiedStatus.native_code) + " events=" +
                                     std::to_string(failure.count) + " stage_match=" +
                                     std::to_string(failure.failureStageMatches) + " diagnostic_match=" +
                                     std::to_string(failure.failureDiagnosticMatches) + " completion_diagnostic_match=" +
                                     std::to_string(failure.completionDiagnosticMatches));
        }
        const auto workspace = workspace_transitions();
        const auto submission = runtime.submission(state);
        auto* const effect = runtime.find(state);
        if (workspace.resourcesAcquired != 1 || workspace.aborts != 1 || workspace.candidateCount != 0 ||
            workspace.captureEnqueued || workspace.scannerEnqueued || workspace.finishes != 0 ||
            !submission.valid || runtime.host().image_release_count(effect, "Source") != 1 ||
            runtime.host().image_release_count(effect, "Output") != 1) {
            throw std::runtime_error("diffusion components panic did not stop preparation and release the frame");
        }
        (void)render_case(runtime, test, parameters, state);
        const auto retry = runtime.submission(state);
        if (!retry.valid || retry.context.device_id != submission.context.device_id ||
            retry.context.context != submission.context.context ||
            retry.submission.submission_id != submission.submission.submission_id) {
            throw std::runtime_error("diffusion components panic changed same-context retry");
        }
        std::cout << kDiffusionComponentsPanicCase << " stage=" << failure.stage
                  << " category=" << failure.status.category << " api=" << failure.status.api
                  << " code=" << failure.status.native_code << " abort/classify/recovery/fatal order and retry passed\n";
    }

    void run_failure_order_cases(CallbackSession& runtime) {
        const std::array<Case, 2> cases{{{"failure-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"failure-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        std::vector<std::string> intended;
        intended.reserve(43);
        intended.emplace_back(kDiffusionComponentsPanicCase);
        for (const Case& test : cases) {
            for (const char* delivery : {"success", "failure", "throw"})
                intended.push_back(std::string(test.name) + "/delivery/" + delivery);
            for (unsigned status = 0; status < 4; ++status)
                for (unsigned wording = 0; wording < 3; ++wording)
                    intended.push_back(std::string(test.name) + "/status/" + std::to_string(status) + "/wording/" + std::to_string(wording));
            for (const char* terminal : {"context-loss", "mismatching-context", "gate-reentry", "active-submission", "non-dir-failure", "long-diagnostic"})
                intended.push_back(std::string(test.name) + "/terminal/" + terminal);
        }
        JuicerTests::Cohort rows("failure_order", intended);
        {
            auto row = rows.case_event(kDiffusionComponentsPanicCase);
            check_diffusion_components_panic(runtime, cases.front());
            row.passed();
        }
        for (const Case& test : cases) {
            unsigned deliveryIndex = 0;
            for (const auto delivery : {JuicerCuda::ExecutorTest::Delivery::Success,
                                        JuicerCuda::ExecutorTest::Delivery::Failure,
                                        JuicerCuda::ExecutorTest::Delivery::Throw}) {
                const auto* const deliveryName = std::array{"success", "failure", "throw"}[deliveryIndex++];
                auto row = rows.case_event(std::string(test.name) + "/delivery/" + deliveryName);
                check_failure_order(runtime, test, delivery, true, JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed), "renamed receiver arithmetic failure 100%");
                row.passed();
            }
            const std::array statuses{JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED),
                                      JuicerCuda::runtime_failure_status(cudaErrorDeviceUninitialized),
                                      JuicerCuda::runtime_failure_status(cudaErrorInvalidValue),
                                      FjStatus{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}};
            const std::array<const char*, 3> wordings{"test component=dir 100% device lost out of memory", "renamed arithmetic failure 100%", ""};
            for (std::size_t statusIndex = 0; statusIndex < statuses.size(); ++statusIndex) {
                for (std::size_t wordingIndex = 0; wordingIndex < wordings.size(); ++wordingIndex) {
                    auto row = rows.case_event(std::string(test.name) + "/status/" + std::to_string(statusIndex) + "/wording/" + std::to_string(wordingIndex));
                    check_failure_order(runtime, test, JuicerCuda::ExecutorTest::Delivery::Success, true, statuses[statusIndex], wordings[wordingIndex]);
                    row.passed();
                }
            }
            for (const bool holdGate : {true, false}) {
                auto row = rows.case_event(std::string(test.name) + "/terminal/" + (holdGate ? "gate-reentry" : "active-submission"));
                check_recovery_interference(runtime, test, holdGate);
                row.passed();
            }
            {
                auto row = rows.case_event(std::string(test.name) + "/terminal/mismatching-context");
                check_mismatching_recovery(runtime, test);
                row.passed();
            }
            {
                auto row = rows.case_event(std::string(test.name) + "/terminal/context-loss");
                check_failure_order(runtime, test, JuicerCuda::ExecutorTest::Delivery::Success, false, JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED), "test component=scanner 100% renamed diagnostic");
                row.passed();
            }
            {
                auto row = rows.case_event(std::string(test.name) + "/terminal/non-dir-failure");
                check_failure_order(runtime, test, JuicerCuda::ExecutorTest::Delivery::Success, false, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "non-DIR failure mentioning component=dir 100%");
                row.passed();
            }
            {
                auto row = rows.case_event(std::string(test.name) + "/terminal/long-diagnostic");
                const std::string longDiagnostic = std::string(8192, 'x') + " 100% complete diagnostic";
                check_failure_order(runtime, test, JuicerCuda::ExecutorTest::Delivery::Success, true, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, longDiagnostic.c_str());
                row.passed();
            }
        }
    }

    FjPreparedNoiseObservation prepared_noise_observation();
    bool noise_ordered_through_completion(const FjPreparedNoiseObservation& observation);

    void check_resource_failure(CallbackSession& runtime, const Case& test, const JuicerTestSupport::PreparationInput& parameters, JuicerTestSupport::InstanceFixture& state, JuicerCuda::ExecutorTest::ResourceFailures& failures, const JuicerCuda::Failure& expectedFailure, const char* stage, bool preparationFailure = false) {
        using JuicerCuda::ExecutorTest::Event;
        if (parameters.values.grain.active) {
            fj_test_prepared_noise_observation_reset();
        }
        JuicerCuda::ExecutorTest::FailureObservation observation;
        observation.armed = false;
        observation.status = expectedFailure.status;
        observation.diagnostic = expectedFailure.diagnostic.c_str();
        observation.stage = stage;
        JuicerCuda::ExecutorTest::observation = &observation;
        JuicerCuda::ExecutorTest::resourceFailures = &failures;
        bool fatal = false;
        try {
            (void)render_case(runtime, test, parameters, state);
        } catch (const CallbackStatus& error) {
            fatal = error.status() == kOfxStatErrFatal;
            observation.record(Event::FatalMapped);
        } catch (...) {
            JuicerCuda::ExecutorTest::observation = nullptr;
            JuicerCuda::ExecutorTest::resourceFailures = nullptr;
            throw;
        }
        JuicerCuda::ExecutorTest::observation = nullptr;
        JuicerCuda::ExecutorTest::resourceFailures = nullptr;
        if (parameters.values.grain.active) {
            const auto noise = prepared_noise_observation();
            if (noise.acquisition_attempts != 1 || noise.successful_results != 1 ||
                noise.held_at_complete != 1 || noise.drops_after_complete != 1 ||
                noise.native_render_calls != 1 || !noise_ordered_through_completion(noise)) {
                throw std::runtime_error("actual Rust noise source did not survive native upload failure and recovery");
            }
        }
        const bool contextLoss = expectedFailure.status.category == FJ_STATUS_CONTEXT_LOSS;
        std::array expectedEvents{Event::Injected, Event::Classified, Event::FrameAborted, Event::RecoveryStarted, Event::RecoveryEnded, Event::FatalMapped};
        if (preparationFailure) {
            // Root aborts failed preparation before returning it to the executor.
            std::swap(expectedEvents[1], expectedEvents[2]);
        }
        if (!fatal || observation.overflow || observation.count != expectedEvents.size() ||
            !std::equal(expectedEvents.begin(), expectedEvents.end(), observation.events.begin()) ||
            !observation.failureStageMatches || !observation.failureStatusMatches ||
            !observation.failureDiagnosticMatches ||
            observation.nativeRecoveryPending != contextLoss ||
            observation.adapterRecoveryPending != contextLoss ||
            runtime.submission(state).valid == contextLoss) {
            throw std::runtime_error(std::string(test.name) + ": resource failure propagation changed; category=" +
                                     std::to_string(observation.classifiedStatus.category) + " api=" +
                                     std::to_string(observation.classifiedStatus.api) + " code=" +
                                     std::to_string(observation.classifiedStatus.native_code) + " events=" +
                                     std::to_string(observation.count) + " stage_match=" +
                                     std::to_string(observation.failureStageMatches) + " diagnostic_match=" +
                                     std::to_string(observation.failureDiagnosticMatches));
        }
        std::cout << test.name << " stage=" << stage << " category=" << expectedFailure.status.category
                  << " api=" << expectedFailure.status.api << " code=" << expectedFailure.status.native_code
                  << " injections=" << failures.injections
                  << (preparationFailure ? " abort/classify" : " classify/abort")
                  << "/recovery/fatal order passed\n";
    }

    void run_scanner_lut_upload_failure_cases(CallbackSession& runtime) {
        const std::array cases{Case{"lut-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"lut-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0},
                               Case{"lut-positive-direct", Spektrafilm::ScanRoute::PositiveDirectScan, 0, 3, 0, 0},
                               Case{"lut-positive-print", Spektrafilm::ScanRoute::PositivePrintScan, 0, 4, 0, 0}};
        std::vector<std::string> intended;
        intended.reserve(28);
        for (const auto& test : cases) {
            for (unsigned position = 1; position <= 6; ++position)
                intended.push_back(std::string(test.name) + "/position/" + std::to_string(position));
            intended.push_back(std::string(test.name) + "/warm-render");
        }
        JuicerTests::Cohort rows("scanner_lut_upload", intended);
        for (const auto& test : cases) {
            JuicerTestSupport::InstanceFixture state;
            MembershipRetirement stateRetirement{runtime, state};
            auto parameters = parameters_for(test);
            parameters.values.grain.active = false;
            for (unsigned position = 1; position <= 6; ++position) {
                auto row = rows.case_event(std::string(test.name) + "/position/" + std::to_string(position));
                JuicerCuda::ExecutorTest::ResourceFailures failures;
                failures.lutPosition = position;
                const std::string diagnostic = std::string("scanner LUT upload fixture failure route=") + Spektrafilm::scan_route_metadata(test.route).key;
                check_resource_failure(runtime, test, parameters, state, failures, {{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, diagnostic}, "prepare_focused_route_resources", true);
                if (failures.lutAttempts != position || failures.injections != 1 || fj_test_scanner_lut_live_owners() != 0)
                    throw std::runtime_error("scanner LUT upload short circuit or host-owner cleanup changed");
                row.passed();
            }
            auto warm = rows.case_event(std::string(test.name) + "/warm-render");
            (void)render_case(runtime, test, parameters, state);
            warm.passed();
        }
    }
    void run_defect_fence_failure_cases(CallbackSession& runtime) {
        const std::array<Case, 2> cases{{{"defect-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"defect-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        std::vector<std::string> intended;
        intended.reserve(12);
        for (const auto& test : cases)
            for (const char* timing : {"event-create", "event-record"})
                for (unsigned code = 0; code < 3; ++code)
                    intended.push_back(std::string(test.name) + "/" + timing + "/" + std::to_string(code));
        JuicerTests::Cohort rows("defect_fence", intended);
        for (const Case& test : cases) {
            JuicerTestSupport::InstanceFixture state;
            MembershipRetirement stateRetirement{runtime, state};
            for (const bool failCreate : {true, false}) {
                const std::array codes{cudaErrorContextIsDestroyed, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation};
                for (std::size_t codeIndex = 0; codeIndex < codes.size(); ++codeIndex) {
                    auto row = rows.case_event(std::string(test.name) + "/" + (failCreate ? "event-create" : "event-record") + "/" + std::to_string(codeIndex));
                    const auto code = codes[codeIndex];
                    auto parameters = parameters_for(test);
                    parameters.values.film_dust_amount = 1.0f;
                    (void)render_case(runtime, test, parameters, state);
                    parameters.values.film_dust_amount = 0.0f;
                    parameters.values.gate_dust_amount = 1.0f;
                    JuicerCuda::ExecutorTest::ResourceFailures failures;
                    (failCreate ? failures.eventCreate : failures.eventRecord) = code;
                    check_resource_failure(runtime, test, parameters, state, failures, {JuicerCuda::runtime_failure_status(code), "defect attachment retirement fence failed"}, Spektrafilm::scan_route_is_print(test.route) ? "print_focused_workspace_stage" : "direct_focused_workspace_stage");
                    if (failures.injections != 1) {
                        throw std::runtime_error("defect fence failure did not reach the resource owner");
                    }
                    row.passed();
                }
            }
        }
    }

    std::filesystem::path callback_resource_directory() {
        const std::filesystem::path emptyPath;
        const JuicerAssets::NativePathArgument empty(emptyPath);
        const auto length = fj_test_ofx_path(0, empty.view(), nullptr, 0);
        if (length == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("callback resource path unavailable");
        std::filesystem::path::string_type units(length, 0);
        if (fj_test_ofx_path(0, empty.view(), units.data(), units.size()) != length)
            throw std::runtime_error("callback resource path changed");
        return std::filesystem::path(units);
    }

    struct ScratchNoiseResources {
        std::filesystem::path root;
        std::filesystem::path stbn;
        bool verified = false;

        ScratchNoiseResources() {
            const char* const artifactDirectory = std::getenv("JUICER_TEST_ARTIFACT_DIR");
            if (!artifactDirectory || artifactDirectory[0] == '\0') {
                throw std::runtime_error("Ofx.Gpu.NoiseSourceLifetime requires JUICER_TEST_ARTIFACT_DIR");
            }
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const std::filesystem::path artifactRoot(artifactDirectory);
            std::filesystem::create_directories(artifactRoot);
            root = artifactRoot / ("scratch-noise-call-" + std::to_string(stamp));
            if (std::filesystem::exists(root)) {
                throw std::runtime_error("noise scratch directory collision under test artifacts");
            }
            const auto source = callback_resource_directory();
            JuicerTestSupport::copy_print_preparation_assets(source, root);
            // The held noise bundle consumes STBN and both Wang inputs even
            // though this case mutates only its STBN backing.
            JuicerTestSupport::copy_resource_files(source, root, std::array<std::string_view, 3>{"Noise/stbn_scalar_512x512x256_u8.bin", "Noise/Wang/tiles.json", "Noise/Wang/wang_tiles_256x256x16_u8.bin"});
            stbn = root / "Noise/stbn_scalar_512x512x256_u8.bin";
        }

        ~ScratchNoiseResources() noexcept {
            if (!verified || std::uncaught_exceptions() != 0) {
                return;
            }
            try {
                std::error_code error;
                std::filesystem::remove_all(root, error);
            } catch (...) {
                std::fputs("verified noise scratch cleanup threw\n", stderr);
            }
        }

        void mark_verified() noexcept {
            verified = true;
        }
    };

    class ScratchRustAssets final {
    public:
        explicit ScratchRustAssets(const std::filesystem::path& root) {
            const JuicerAssets::NativePathArgument path(root);
            const FjStatus status = fj_test_assets_create(path.view(), &owner_, nullptr);
            if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0 || !owner_) {
                throw std::runtime_error("scratch Rust Assets create failed");
            }
        }

        ~ScratchRustAssets() {
            if (owner_) {
                const FjStatus status = fj_test_assets_destroy(std::exchange(owner_, nullptr), nullptr);
                if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
                    std::abort();
                }
            }
        }

        FjAssets* get() const noexcept {
            return owner_;
        }

    private:
        FjAssets* owner_ = nullptr;
    };

    FjPreparedNoiseObservation prepared_noise_observation() {
        FjPreparedNoiseObservation observation{};
        const FjStatus status = fj_test_prepared_noise_observation(&observation);
        if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
            throw std::runtime_error("prepared noise observation query failed");
        }
        return observation;
    }

    bool noise_ordered_through_completion(const FjPreparedNoiseObservation& observation) {
        return observation.acquisition_attempts == 1 &&
               observation.precheck_order != 0 &&
               observation.acquisition_order > observation.precheck_order &&
               observation.admission_order > observation.acquisition_order &&
               observation.inspection_order > observation.admission_order &&
               observation.release_order > observation.inspection_order &&
               observation.complete_order > observation.release_order &&
               observation.native_render_calls <= 1 &&
               (observation.native_render_calls == 0
                    ? observation.native_render_order == 0
                    : observation.native_render_order > observation.inspection_order &&
                          observation.native_render_order < observation.release_order) &&
               observation.drops_after_complete == 1 &&
               observation.drop_order > observation.complete_order;
    }

    void run_published_noise_call_cases(CallbackSession& runtime) {
        const std::array cases{Case{"noise-published-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"noise-published-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}};
        const std::array<std::string, 4> intended{{"direct/cold", "direct/warm", "print/cold", "print/warm"}};
        JuicerTests::Cohort rows("noise_published_call", std::vector<std::string>(intended.begin(), intended.end()));
        const FjAssets* rootAssets = runtime.borrow().assets;
        for (const Case& test : cases) {
            const std::string route = Spektrafilm::scan_route_is_print(test.route) ? "print" : "direct";
            JuicerTestSupport::InstanceFixture state;
            MembershipRetirement stateRetirement{runtime, state};
            auto active = parameters_for(test);
            active.values.grain.active = true;
            for (bool cold : {true, false}) {
                auto case_event = rows.case_event(route + (cold ? "/cold" : "/warm"));
                if (cold) {
                    fj_test_assets_release_cached_payloads(rootAssets, nullptr);
                }
                fj_test_prepared_noise_observation_reset();
                const auto pixels = render_case(runtime, test, active, state);
                const auto observation = prepared_noise_observation();
                if (pixels.empty() || observation.acquisition_attempts != 1 ||
                    observation.successful_results != 1 || observation.held_at_complete != 1 ||
                    observation.complete_calls != 1 || observation.drops_after_complete != 1 ||
                    observation.native_render_calls != 1 || observation.arc_references_at_complete < 2 ||
                    !noise_ordered_through_completion(observation)) {
                    throw std::runtime_error("published direct/print Rust noise source was not held through completion");
                }
                case_event.passed();
            }
        }
        std::puts("PASS published direct/print state borrows: actual Rust noise acquisition, completion hold, post-completion drop and native render");
    }

    void run_processor_unavailable_request_cases(CallbackSession& runtime) {
        const std::array cases{Case{"noise-missing-state-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"noise-missing-state-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}};
        std::vector<std::string> intended;
        intended.reserve(8);
        for (const char* route : {"direct", "print"})
            for (const char* availability : {"missing-state", "missing-instance", "uninitialized-instance", "missing-state-uninitialized"})
                for (const char* cancellation : {"cancel", "no-cancel"})
                    intended.push_back(std::string(route) + "/" + availability + "/" + cancellation);
        JuicerTests::Cohort rows("noise_unavailable_request", intended);
        for (const Case& test : cases) {
            const std::string route = Spektrafilm::scan_route_is_print(test.route) ? "print" : "direct";
            auto active = parameters_for(test);
            active.values.grain.active = true;
            const auto run = [&](PublishedRequestAvailability availability, unsigned cancelAt, bool expectFatal) {
                JuicerTestSupport::InstanceFixture state;
                MembershipRetirement stateRetirement{runtime, state};
                const bool latchValidBefore = runtime.submission(state).valid;
                const char* const availability_name = availability == PublishedRequestAvailability::MissingState ? "missing-state" : availability == PublishedRequestAvailability::MissingInstance     ? "missing-instance"
                                                                                                                                 : availability == PublishedRequestAvailability::UninitializedIdentity ? "uninitialized-instance"
                                                                                                                                                                                                       : "missing-state-uninitialized";
                auto case_event = rows.case_event(route + "/" + availability_name + "/" + (cancelAt ? "cancel" : "no-cancel"));
                fj_test_prepared_noise_observation_reset();
                AbortObservation abort{};
                abort.cancelAt = cancelAt;
                s_abortObservation = cancelAt ? &abort : nullptr;
                bool fatal = false;
                std::vector<float> pixels;
                try {
                    pixels = render_case(runtime, test, active, state, false, ExecutionPath::Processor, DestinationLayout::Matching, 1, JuicerCudaTest::StateLifetimeCase::SuppliedStream, availability);
                } catch (const CallbackStatus& error) {
                    fatal = error.status() == kOfxStatErrFatal;
                } catch (...) {
                    s_abortObservation = nullptr;
                    throw;
                }
                s_abortObservation = nullptr;
                const auto observation = prepared_noise_observation();
                if (fatal != expectFatal || (cancelAt && (abort.calls != cancelAt || !pixels.empty())) ||
                    runtime.submission(state).valid != latchValidBefore ||
                    observation.complete_calls != 1 || observation.native_render_calls != 0) {
                    throw std::runtime_error(std::string(test.name) + ": unavailable processor request admitted a latch, rendered, or skipped completion");
                }
                if (availability == PublishedRequestAvailability::MissingState || availability == PublishedRequestAvailability::MissingStateUninitialized) {
                    if (observation.acquisition_attempts != 0 || observation.held_at_complete != 0 ||
                        observation.drops_after_complete != 0 || observation.arc_references_at_complete != 0 ||
                        observation.inspection_order <= observation.admission_order ||
                        observation.release_order <= observation.inspection_order ||
                        observation.complete_order <= observation.release_order) {
                        throw std::runtime_error(std::string(test.name) + ": missing-state processor request acquired noise or skipped inspect/release/completion");
                    }
                } else if (observation.acquisition_attempts != 1 || observation.successful_results != 1 ||
                           observation.held_at_complete != 1 || observation.drops_after_complete != 1 ||
                           observation.arc_references_at_complete < 2 ||
                           !noise_ordered_through_completion(observation)) {
                    throw std::runtime_error(std::string(test.name) + ": missing-instance processor request did not hold active noise through completion");
                }
                case_event.passed();
            };

            run(PublishedRequestAvailability::MissingState, 1, false);
            run(PublishedRequestAvailability::MissingState, 0, true);
            run(PublishedRequestAvailability::MissingInstance, 1, false);
            run(PublishedRequestAvailability::MissingInstance, 0, true);
            run(PublishedRequestAvailability::UninitializedIdentity, 1, false);
            run(PublishedRequestAvailability::UninitializedIdentity, 0, true);
            run(PublishedRequestAvailability::MissingStateUninitialized, 1, false);
            run(PublishedRequestAvailability::MissingStateUninitialized, 0, true);
        }
        std::puts("PASS raw unavailable-state and missing-instance requests: inspect/abort precedence, source hold, no latch and no render");
    }

    void run_scratch_noise_failure_recovery_cases(CallbackSession& runtime) {
        const std::array cases{Case{"noise-held-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"noise-held-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}};
        const std::array<std::string, 6> intended{{"direct", "print", "direct/retire-cancel", "direct/retire-abort-failure", "print/retire-cancel", "print/retire-abort-failure"}};
        JuicerTests::Cohort rows("noise_scratch_failure", std::vector<std::string>(intended.begin(), intended.end()));
        for (const Case& test : cases) {
            auto case_event = rows.case_event(Spektrafilm::scan_route_is_print(test.route) ? "print" : "direct");
            ScratchNoiseResources scratch;
            ScratchRustAssets assets(scratch.root);
            auto parameters = parameters_for(test);
            parameters.values.grain.active = true;
            std::optional<JuicerTestSupport::PreparedStateHold> prepared;
            std::array<char, 2048> prepareText{};
            FjErrorBuffer prepareError{prepareText.data(), prepareText.size(), 0};
            const FjStatus preparedStatus = prepare_state(assets.get(), parameters, prepared, prepareError);
            if (preparedStatus.category != FJ_STATUS_SUCCESS || !prepared) {
                throw std::runtime_error(std::string(test.name) + ": complete Rust prepared state creation failed");
            }

            DeviceFrame device;
            const std::size_t elements = static_cast<std::size_t>(test.width) * static_cast<std::size_t>(test.height) * static_cast<std::size_t>(test.components);
            std::vector<float> host(elements, 0.25f);
            require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "noise held stream");

            int deviceId = 0;
            CUcontext context = nullptr;
            require_cuda(cudaGetDevice(&deviceId), "missing-noise device");
            if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) {
                throw std::runtime_error("missing-noise current context");
            }
            using Frame = JuicerProcess::Root::PreparedCudaFrame;
            const JuicerCuda::ResourceManager::DeviceContextKey contextKey{deviceId, context};
            JuicerCuda::ResourceManager::SubmissionSnapshot missingNoiseSnapshot;
            missingNoiseSnapshot.instanceToken.value = 0x53344500 + static_cast<unsigned>(test.route);
            missingNoiseSnapshot.frameToken.value = 1;
            missingNoiseSnapshot.snapshotId = 1;
            missingNoiseSnapshot.deviceContextKey = contextKey;
            Frame missingNoiseFrame;
            JuicerCuda::Failure missingNoiseError;
            JuicerCudaTest::PreparedFrameFixture::FrameFacts missingNoiseFacts;
            missingNoiseFacts.renderExtent = {test.originX, test.originY, test.originX + test.width, test.originY + test.height};
            missingNoiseFacts.fullFrameExtent = missingNoiseFacts.renderExtent;
            missingNoiseFacts.pixelDefinition = missingNoiseFacts.renderExtent;
            missingNoiseFacts.pixelSizeUm = 35'000.0f / static_cast<float>(test.width);
            missingNoiseFacts.filmFormatLongEdgeMm = 35.0f;
            missingNoiseFacts.canonicalWidth = static_cast<double>(test.width);
            missingNoiseFacts.canonicalHeight = static_cast<double>(test.height);
            missingNoiseFacts.scaleX = test.scaleX;
            missingNoiseFacts.scaleY = test.scaleY;
            missingNoiseFacts.frameRate = 24.0;
            missingNoiseFacts.sessionSeed = 0x20260923;
            missingNoiseFacts.clipToken = 0x5312;
            WorkspacePreparedFrameContext missingNoisePreparation{
                &JuicerCuda::borrow_root(runtime.borrow().cuda), contextKey, missingNoiseSnapshot, device.stream, &missingNoiseFrame, &missingNoiseError};
            missingNoisePreparation.omitNoise = true;
            JuicerCuda::ExecutorTest::reset_workspace_transitions();
            std::string missingNoiseDiagnostic;
            if (!JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                    runtime.borrow().assets, *prepared, missingNoiseFacts, &missingNoisePreparation, &prepare_workspace_frame_from_state, missingNoiseDiagnostic)) {
                throw std::runtime_error(std::string(test.name) + ": missing-noise fixture failed: " + missingNoiseDiagnostic);
            }
            const std::string routeKey = Spektrafilm::scan_route_key(test.route);
            const auto missingNoiseObserved = JuicerCuda::ExecutorTest::workspace_transitions();
            if (!missingNoisePreparation.grainWasValid || missingNoiseFrame.active() ||
                missingNoiseFrame.workspace_lease().active() ||
                missingNoiseError.status.category != FJ_STATUS_PREPARATION_FAILURE ||
                missingNoiseError.status.api != FJ_API_NONE || missingNoiseError.status.native_code != 0 ||
                missingNoiseError.diagnostic != "MissingRequiredResource phase=grain_static field=noise route=" + routeKey ||
                missingNoiseObserved.requested || missingNoiseObserved.admitted ||
                missingNoiseObserved.resourcesAcquired != 0 || missingNoiseObserved.candidateCount != 0 ||
                missingNoiseObserved.finishes != 0 || missingNoiseObserved.recoveryPending) {
                throw std::runtime_error(std::string(test.name) + ": missing-noise preflight changed: " + missingNoiseError.diagnostic);
            }
            const auto missingNoisePrimary = missingNoiseError;
            missingNoiseFrame.abort();
            if (missingNoiseError.status.category != missingNoisePrimary.status.category ||
                missingNoiseError.status.api != missingNoisePrimary.status.api ||
                missingNoiseError.status.native_code != missingNoisePrimary.status.native_code ||
                missingNoiseError.diagnostic != missingNoisePrimary.diagnostic) {
                throw std::runtime_error(std::string(test.name) + ": frame abort lost missing-noise preflight failure");
            }
            std::cout << "preserved missing-noise preflight route=" << routeKey
                      << " status={3,0,0} diagnostic=" << missingNoiseError.diagnostic << '\n';
            fj_test_prepared_noise_observation_reset();

            require_cuda(cudaMalloc(&device.source, elements * sizeof(float)), "noise held source allocation");
            require_cuda(cudaMalloc(&device.destination, elements * sizeof(float)), "noise held destination allocation");
            require_cuda(cudaMemcpy(device.source, host.data(), elements * sizeof(float), cudaMemcpyHostToDevice), "noise held source upload");
            require_cuda(cudaMemset(device.destination, 0, elements * sizeof(float)), "noise held destination clear");
            const FjRect bounds{test.originX, test.originY, test.originX + test.width, test.originY + test.height};
            FjFrame frame{};
            frame.source = {reinterpret_cast<std::uintptr_t>(device.source), bounds, static_cast<std::ptrdiff_t>(test.width) * test.components * static_cast<std::ptrdiff_t>(sizeof(float)), static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
            frame.destination = {reinterpret_cast<std::uintptr_t>(device.destination), bounds, frame.source.row_bytes, static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
            frame.render_window = bounds;
            frame.full_frame_extent = bounds;
            frame.stream = reinterpret_cast<std::uintptr_t>(device.stream);
            frame.flags = FJ_FRAME_STREAM_PRESENT;
            frame.pixel_size_um = 35'000.0f / static_cast<float>(test.width);
            frame.time_frames = test.time;
            frame.frame_rate = 24.0;
            frame.session_seed = 0x20260923;
            frame.clip_token = 0x5312;
            frame.effects_geometry = {{bounds.x1, bounds.y1, test.width, test.height},
                                      static_cast<double>(bounds.x1),
                                      static_cast<double>(bounds.y1),
                                      static_cast<double>(test.width),
                                      static_cast<double>(test.height),
                                      test.scaleX,
                                      test.scaleY,
                                      1.0};

            // The standalone view check populated this owner; preserve cold source failure below.
            if (fj_test_assets_release_cached_payloads(assets.get(), nullptr).category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("scratch noise cache release failed before missing-source cases");
            }
            std::filesystem::remove(scratch.stbn);
            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion failedCompletion{};
            std::string diagnostic;
            const auto failed = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, assets.get(), &failedCompletion);
            const auto failedObservation = prepared_noise_observation();
            if (failed.outcome.status.category != FJ_STATUS_PREPARATION_FAILURE ||
                diagnostic.find("stbn_scalar_512x512x256_u8.bin") == std::string::npos ||
                !failedCompletion.observationRead || failedCompletion.retirementFailed || failedCompletion.noise.held_at_complete != 1 ||
                failedCompletion.noise.preparation_failures != 1 ||
                failedCompletion.noise.drops_after_complete != 0 ||
                failedCompletion.noise.arc_references_at_complete < 2 ||
                failedObservation.preparation_failures != 1 || failedObservation.native_render_calls != 0 ||
                !noise_ordered_through_completion(failedObservation)) {
                throw std::runtime_error(std::string(test.name) + ": actual missing-noise error was not deferred, held, and dropped after completion");
            }

            AbortAfterNativeDescriptor firstAbort{0, 1};
            const FjAbortCallback cancel{&abort_after_native_descriptor, &firstAbort};
            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion cancelledCompletion{};
            diagnostic.clear();
            const auto cancelled = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), cancel, nullptr, nullptr, diagnostic, assets.get(), &cancelledCompletion);
            const auto cancelledObservation = prepared_noise_observation();
            if (cancelled.outcome.status.category != FJ_STATUS_CANCELLED ||
                !cancelledCompletion.observationRead || cancelledCompletion.retirementFailed || cancelledCompletion.noise.held_at_complete != 1 ||
                cancelledCompletion.noise.preparation_failures != 1 ||
                cancelledObservation.native_render_calls != 0 ||
                !noise_ordered_through_completion(cancelledObservation)) {
                throw std::runtime_error(std::string(test.name) + ": first abort did not retain then drop the deferred source error");
            }

            FjFrame inspectionFrame = frame;
            --inspectionFrame.destination.bounds.x2;
            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion inspectionCompletion{};
            diagnostic.clear();
            const auto inspection = render_held_prepared_state(
                runtime, *prepared, inspectionFrame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, assets.get(), &inspectionCompletion);
            const auto inspectionObservation = prepared_noise_observation();
            if (inspection.outcome.status.category == FJ_STATUS_SUCCESS ||
                diagnostic.find("stbn_scalar_512x512x256_u8.bin") != std::string::npos ||
                !inspectionCompletion.observationRead || inspectionCompletion.retirementFailed || inspectionCompletion.noise.held_at_complete != 1 ||
                inspectionCompletion.noise.preparation_failures != 1 ||
                inspectionObservation.native_render_calls != 0 ||
                !noise_ordered_through_completion(inspectionObservation)) {
                throw std::runtime_error(std::string(test.name) + ": inspection did not precede deferred source delivery");
            }

            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion missingInstanceCompletion{};
            diagnostic.clear();
            const auto missingInstance = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, assets.get(), &missingInstanceCompletion, true);
            const auto missingInstanceObservation = prepared_noise_observation();
            if (missingInstance.outcome.status.category != FJ_STATUS_PREPARATION_FAILURE ||
                diagnostic.find("instance state missing") == std::string::npos ||
                !missingInstanceCompletion.observationRead || missingInstanceCompletion.retirementFailed || missingInstanceCompletion.noise.held_at_complete != 1 ||
                missingInstanceCompletion.noise.preparation_failures != 1 ||
                missingInstanceObservation.native_render_calls != 0 ||
                !noise_ordered_through_completion(missingInstanceObservation)) {
                throw std::runtime_error(std::string(test.name) + ": missing-instance rejection lost precedence over the held source error");
            }

            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion missingStateCompletion{};
            diagnostic.clear();
            const auto missingState = render_missing_prepared_state(
                runtime, assets.get(), frame, {}, missingStateCompletion, diagnostic);
            const auto missingStateObservation = prepared_noise_observation();
            if (missingState.outcome.status.category != FJ_STATUS_PREPARATION_FAILURE ||
                diagnostic.find("render state unavailable") == std::string::npos ||
                missingStateObservation.acquisition_attempts != 0 ||
                missingStateObservation.held_at_complete != 0 ||
                missingStateObservation.complete_calls != 1 ||
                missingStateObservation.native_render_calls != 0 ||
                !missingStateCompletion.observationRead) {
                throw std::runtime_error(std::string(test.name) + ": missing-state path acquired noise or skipped completion");
            }

            std::filesystem::copy_file(
                callback_resource_directory() / "Noise/stbn_scalar_512x512x256_u8.bin",
                scratch.stbn,
                std::filesystem::copy_options::overwrite_existing);
            if (fj_test_assets_release_cached_payloads(assets.get(), nullptr).category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("scratch noise cache release failed before recovery");
            }
            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion recoveredCompletion{};
            diagnostic.clear();
            const auto recovered = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, assets.get(), &recoveredCompletion);
            const auto recoveredObservation = prepared_noise_observation();
            if (recovered.outcome.status.category != FJ_STATUS_SUCCESS ||
                !recoveredCompletion.observationRead || recoveredCompletion.retirementFailed || recoveredCompletion.noise.held_at_complete != 1 ||
                recoveredCompletion.noise.successful_results != 1 || recoveredObservation.native_render_calls != 1 ||
                recoveredObservation.drops_after_complete != 1 || recoveredObservation.arc_references_at_complete < 2 ||
                !noise_ordered_through_completion(recoveredObservation)) {
                throw std::runtime_error(std::string(test.name) + ": actual noise source did not recover on the held state");
            }
            require_cuda(cudaStreamSynchronize(device.stream), "noise held recovery drain");

            // The decoder capacity fault is injected at its real reservation
            // boundary. Its failed result is retained by this call only; the
            // next request on the same Assets owner must retry without cache
            // clearing.
            const FjStatus clearedCache = fj_test_assets_release_cached_payloads(assets.get(), nullptr);
            if (clearedCache.category != FJ_STATUS_SUCCESS ||
                fj_test_noise_reservation_failure(1).category != FJ_STATUS_SUCCESS ||
                fj_test_noise_reservation_consumed() != 0) {
                throw std::runtime_error(std::string(test.name) + ": could not arm cold decoder reservation failure");
            }
            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion allocationCompletion{};
            diagnostic.clear();
            const auto allocationFailure = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, assets.get(), &allocationCompletion);
            const auto allocationObservation = prepared_noise_observation();
            if (allocationFailure.outcome.status.category != FJ_STATUS_ALLOCATION_FAILURE ||
                diagnostic.empty() || fj_test_noise_reservation_consumed() != 1 ||
                !allocationCompletion.observationRead || allocationCompletion.retirementFailed || allocationCompletion.noise.held_at_complete != 1 ||
                allocationCompletion.noise.allocation_failures != 1 ||
                allocationCompletion.noise.preparation_failures != 0 ||
                allocationCompletion.noise.internal_failures != 0 ||
                allocationCompletion.noise.drops_after_complete != 0 ||
                allocationCompletion.noise.arc_references_at_complete != 1 ||
                allocationObservation.allocation_failures != 1 || allocationObservation.native_render_calls != 0 ||
                allocationObservation.drops_after_complete != 1 || !noise_ordered_through_completion(allocationObservation)) {
                throw std::runtime_error(std::string(test.name) + ": actual decoder capacity failure was not typed, call-held, and dropped after completion");
            }

            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion allocationRecoveryCompletion{};
            diagnostic.clear();
            const auto allocationRecovery = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, assets.get(), &allocationRecoveryCompletion);
            const auto allocationRecoveryObservation = prepared_noise_observation();
            if (allocationRecovery.outcome.status.category != FJ_STATUS_SUCCESS ||
                !allocationRecoveryCompletion.observationRead || allocationRecoveryCompletion.retirementFailed || allocationRecoveryCompletion.noise.held_at_complete != 1 ||
                allocationRecoveryCompletion.noise.successful_results != 1 || allocationRecoveryObservation.native_render_calls != 1 ||
                allocationRecoveryObservation.drops_after_complete != 1 || allocationRecoveryObservation.arc_references_at_complete < 2 ||
                !noise_ordered_through_completion(allocationRecoveryObservation)) {
                throw std::runtime_error(std::string(test.name) + ": same-Assets noise request did not retry after the consumed capacity failure");
            }
            require_cuda(cudaStreamSynchronize(device.stream), "noise capacity recovery drain");

            // A separately poisoned scratch cache produces the actual sticky
            // Assets error. Recreating only Assets (not the prepared state)
            // proves the complete state has no source/cache dependency.
            ScratchRustAssets poisonedAssets(scratch.root);
            std::array<char, 256> poisonText{};
            FjErrorBuffer poisonError{poisonText.data(), poisonText.size(), 0};
            const FjStatus poisoned = fj_test_assets_poison_noise(poisonedAssets.get(), &poisonError);
            if (poisoned.category != FJ_STATUS_SUCCESS || poisoned.api != FJ_API_NONE || poisoned.native_code != 0) {
                throw std::runtime_error(std::string(test.name) + ": could not poison isolated noise cache");
            }
            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion internalCompletion{};
            diagnostic.clear();
            const auto internalFailure = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, poisonedAssets.get(), &internalCompletion);
            const auto internalObservation = prepared_noise_observation();
            if (internalFailure.outcome.status.category != FJ_STATUS_INTERNAL_FAILURE ||
                diagnostic.find("noise cache poisoned") == std::string::npos ||
                !internalCompletion.observationRead || internalCompletion.retirementFailed || internalCompletion.noise.held_at_complete != 1 ||
                internalCompletion.noise.internal_failures != 1 || internalCompletion.noise.preparation_failures != 0 ||
                internalCompletion.noise.allocation_failures != 0 || internalCompletion.noise.arc_references_at_complete != 0 ||
                internalCompletion.noise.drops_after_complete != 0 || internalObservation.internal_failures != 1 ||
                internalObservation.native_render_calls != 0 || internalObservation.drops_after_complete != 1 ||
                !noise_ordered_through_completion(internalObservation)) {
                throw std::runtime_error(std::string(test.name) + ": poisoned noise cache error was not deferred and retained through completion");
            }

            ScratchRustAssets freshAssets(scratch.root);
            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion internalRecoveryCompletion{};
            diagnostic.clear();
            const auto internalRecovery = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, freshAssets.get(), &internalRecoveryCompletion);
            const auto internalRecoveryObservation = prepared_noise_observation();
            if (internalRecovery.outcome.status.category != FJ_STATUS_SUCCESS ||
                !internalRecoveryCompletion.observationRead || internalRecoveryCompletion.retirementFailed || internalRecoveryCompletion.noise.held_at_complete != 1 ||
                internalRecoveryCompletion.noise.successful_results != 1 || internalRecoveryObservation.native_render_calls != 1 ||
                internalRecoveryObservation.drops_after_complete != 1 || internalRecoveryObservation.arc_references_at_complete < 2 ||
                !noise_ordered_through_completion(internalRecoveryObservation)) {
                throw std::runtime_error(std::string(test.name) + ": fresh Assets did not recover the same held prepared state after cache poisoning");
            }
            require_cuda(cudaStreamSynchronize(device.stream), "noise poisoned-cache recovery drain");

            // Shorten a borrowed view after the real owner acquisition. The
            // Rust owner remains held through completion; the real native
            // callback rejects this literal malformed span before CUDA prepare.
            if (fj_test_prepared_noise_short_view(1).category != FJ_STATUS_SUCCESS ||
                fj_test_prepared_noise_short_view_consumed() != 0) {
                throw std::runtime_error(std::string(test.name) + ": could not arm call-local short noise view");
            }
            fj_test_prepared_noise_observation_reset();
            PreparedRenderCompletion malformedCompletion{};
            diagnostic.clear();
            const auto malformedView = render_held_prepared_state(
                runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, diagnostic, freshAssets.get(), &malformedCompletion);
            const auto malformedObservation = prepared_noise_observation();
            if (malformedView.outcome.status.category != FJ_STATUS_UNSUPPORTED_INPUT ||
                diagnostic.find("MalformedPreparedHostData field=noise.stbn") == std::string::npos ||
                fj_test_prepared_noise_short_view_consumed() != 1 ||
                !malformedCompletion.observationRead || malformedCompletion.retirementFailed || malformedCompletion.noise.held_at_complete != 1 ||
                malformedCompletion.noise.successful_results != 1 || malformedCompletion.noise.arc_references_at_complete < 2 ||
                malformedCompletion.noise.drops_after_complete != 0 || malformedObservation.native_render_calls != 1 ||
                malformedObservation.drops_after_complete != 1 || !noise_ordered_through_completion(malformedObservation)) {
                throw std::runtime_error(std::string(test.name) + ": malformed borrowed noise view did not reach and fail the raw callback while held");
            }
            const std::string route = Spektrafilm::scan_route_is_print(test.route) ? "print" : "direct";
            for (const bool invalidAbort : {false, true}) {
                auto retirementCase = rows.case_event(route + (invalidAbort ? "/retire-abort-failure" : "/retire-cancel"));
                struct AbortAtSelection {
                    unsigned calls = 0;
                    bool invalid = false;
                } abortState{0, invalidAbort};
                const FjAbortCallback abort{[](void* user) noexcept -> std::uint32_t {
                                                auto& observed = *static_cast<AbortAtSelection*>(user);
                                                ++observed.calls;
                                                return observed.calls < 3 ? FJ_ABORT_CONTINUE : (observed.invalid ? 2u : FJ_ABORT_REQUESTED);
                                            },
                                            &abortState};
                PreparedRenderCompletion baselineCompletion{};
                std::string baselineDiagnostic;
                const auto baseline = render_held_prepared_state(runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), abort, nullptr, nullptr, baselineDiagnostic, freshAssets.get(), &baselineCompletion);
                if (abortState.calls != 3 || baselineCompletion.calls != 1 || baselineCompletion.retirementFailed ||
                    baseline.outcome.status.category != (invalidAbort ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_CANCELLED) ||
                    baseline.outcome.status.api != FJ_API_NONE || baseline.outcome.status.native_code != 0 ||
                    (invalidAbort && baselineDiagnostic != "invalid CUDA abort query result"))
                    throw std::runtime_error("retirement precedence control did not reach the expected abort outcome");
                abortState.calls = 0;
                std::optional<JuicerCuda::NativeCall> gate;
                PreparedRenderCompletion heldCompletion{};
                heldCompletion.gate = &gate;
                heldCompletion.gateCuda = runtime.borrow().cuda;
                std::string heldDiagnostic;
                const auto held = render_held_prepared_state(runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), abort, nullptr, nullptr, heldDiagnostic, freshAssets.get(), &heldCompletion);
                const bool gateHeld = gate.has_value();
                gate.reset();
                if (abortState.calls != 3 || heldCompletion.calls != 1 || !heldCompletion.gateEntered || !gateHeld || !heldCompletion.retirementFailed ||
                    held.outcome.status.category != baseline.outcome.status.category || held.outcome.status.api != baseline.outcome.status.api ||
                    held.outcome.status.native_code != baseline.outcome.status.native_code || held.outcome.flags != baseline.outcome.flags ||
                    held.fatal != baseline.fatal || heldDiagnostic != baselineDiagnostic)
                    throw std::runtime_error("retirement failure changed the original abort outcome");
                PreparedRenderCompletion recoveredCompletion{};
                std::string recoveredDiagnostic;
                const auto recovered = render_held_prepared_state(runtime, *prepared, frame, 0x641207 + static_cast<int>(test.route), {}, nullptr, nullptr, recoveredDiagnostic, freshAssets.get(), &recoveredCompletion);
                if (recovered.outcome.status.category != FJ_STATUS_SUCCESS || recovered.fatal != 0 || recoveredCompletion.calls != 1 || recoveredCompletion.retirementFailed)
                    throw std::runtime_error("render did not recover after retirement failure");
                retirementCase.passed();
            }
            scratch.mark_verified();
            case_event.passed();
        }
        std::puts("PASS complete Rust held state with typed missing/capacity/internal source failures, malformed borrowed view, recovery, and post-completion owner drops");
    }

    void run_grain_upload_failure_cases(CallbackSession& runtime) {
        const std::array<Case, 2> cases{{{"grain-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"grain-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        std::vector<std::string> intended;
        intended.reserve(32);
        for (const auto& test : cases)
            for (unsigned wording = 0; wording < 2; ++wording)
                for (unsigned upload = 0; upload < 8; ++upload)
                    intended.push_back(std::string(test.name) + "/wording/" + std::to_string(wording) + "/upload/" + std::to_string(upload));
        JuicerTests::Cohort rows("grain_upload", intended);
        for (const Case& test : cases) {
            const std::array<const char*, 2> wordings{"renamed upload component=dir diagnostic", "device lost out of memory 100%"};
            for (std::size_t wordingIndex = 0; wordingIndex < wordings.size(); ++wordingIndex) {
                const char* wording = wordings[wordingIndex];
                const JuicerCuda::Failure capacity{{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, wording};
                const JuicerCuda::Failure contextLoss{JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed), wording};
                const JuicerCuda::Failure driverLoss{JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED), wording};
                const JuicerCuda::Failure ordinary{JuicerCuda::runtime_failure_status(cudaErrorInvalidValue), wording};
                struct UploadCase {
                    std::optional<JuicerCuda::Failure> stbn;
                    std::optional<JuicerCuda::Failure> wang;
                    FjStatus expectedStatus{};
                };
                const std::array<UploadCase, 8> uploads{{{capacity, contextLoss, contextLoss.status},
                                                         {contextLoss, ordinary, contextLoss.status},
                                                         {capacity, std::nullopt, capacity.status},
                                                         {contextLoss, std::nullopt, contextLoss.status},
                                                         {std::nullopt, driverLoss, driverLoss.status},
                                                         {capacity, ordinary, capacity.status},
                                                         {ordinary, capacity, ordinary.status},
                                                         {contextLoss, driverLoss, contextLoss.status}}};
                for (std::size_t uploadIndex = 0; uploadIndex < uploads.size(); ++uploadIndex) {
                    auto row = rows.case_event(std::string(test.name) + "/wording/" + std::to_string(wordingIndex) + "/upload/" + std::to_string(uploadIndex));
                    const auto& upload = uploads[uploadIndex];
                    JuicerTestSupport::InstanceFixture state;
                    MembershipRetirement stateRetirement{runtime, state};
                    auto parameters = parameters_for(test);
                    parameters.values.grain.active = true;
                    JuicerCuda::ExecutorTest::ResourceFailures failures;
                    failures.stbn = upload.stbn;
                    failures.wang = upload.wang;
                    if (failures.stbn) {
                        failures.stbn->diagnostic = std::string("STBN: ") + wording;
                    }
                    if (failures.wang) {
                        failures.wang->diagnostic = std::string("Wang: ") + wording;
                    }
                    std::string diagnostic = std::string(upload.stbn ? "STBN: " : "Wang: ") + wording;
                    if (upload.stbn && upload.wang) {
                        diagnostic += " | Wang upload failed: Wang: ";
                        diagnostic += wording;
                    }
                    diagnostic += " route=";
                    diagnostic += Spektrafilm::scan_route_metadata(test.route).key;
                    check_resource_failure(runtime, test, parameters, state, failures, {upload.expectedStatus, diagnostic}, "ensure_grain_static_assets_uploaded", true);
                    if (failures.stbnAttempts != 1 || failures.wangAttempts != 1 ||
                        failures.injections != static_cast<int>(upload.stbn.has_value()) + static_cast<int>(upload.wang.has_value())) {
                        throw std::runtime_error("grain upload attempts or failure order changed");
                    }
                    row.passed();
                }
            }
        }
    }
} // namespace
#endif

#if defined(JUICER_ACCEPTED_CAPTURE_PATH)
namespace {
    void run_accepted_clean_captures(CallbackSession& runtime) {
        std::ifstream fixture(JUICER_ACCEPTED_CAPTURE_PATH);
        if (!fixture) {
            throw std::runtime_error("accepted CUDA capture fixture unavailable");
        }
        const auto captures = nlohmann::json::parse(fixture);
        const auto decode = [](const nlohmann::json& bits) {
            std::vector<float> pixels;
            pixels.reserve(bits.size());
            for (const auto& value : bits) {
                pixels.push_back(std::bit_cast<float>(value.get<std::uint32_t>()));
            }
            return pixels;
        };
        const auto input = decode(captures.at("input_bits"));
        std::vector<std::string> intended;
        intended.reserve(captures.at("cases").size());
        for (const auto& capture : captures.at("cases"))
            intended.push_back(capture.at("name").get<std::string>());
        JuicerTests::Cohort rows("accepted_capture", intended);
        for (const auto& capture : captures.at("cases")) {
            const auto& settings = capture.at("settings");
            const std::string name = capture.at("name").get<std::string>();
            auto row = rows.case_event(name);
            Case test{};
            test.name = name.c_str();
            test.route = static_cast<Spektrafilm::ScanRoute>(settings.at("route").get<int>());
            test.components = 3;
            test.sourcePixels = &input;
            test.fullWidth = settings.at("width").get<int>();
            test.fullHeight = settings.at("height").get<int>();
            auto parameters = parameters_for(test);
            parameters.film_key = settings.at("film").get<std::string>();
            parameters.print_key = settings.at("print").get<std::string>();
            parameters.values.print_gamma_factor = settings.at("print_gamma").get<double>();
            parameters.values.input_color_space = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::ITU_R_BT2020);
            parameters.values.input_cctf_decoding = 0;
            parameters.values.output_color_space = OutputEncoding::toIndex(OutputEncoding::ColorSpace::ITU_R_BT2020);
            parameters.values.output_cctf_encoding = 0;
            parameters.values.dir_couplers.active = false;
            parameters.values.grain.active = false;
            parameters.values.glare_active = false;
            parameters.values.glare_percent = 0.03;
            JuicerTestSupport::InstanceFixture state;
            MembershipRetirement stateRetirement{runtime, state};
            const auto pixels = render_case(runtime, test, parameters, state);
            compare_pixels(name, pixels, decode(capture.at("output_bits")));
            std::cerr << name << ": exact stored CUDA samples within retained pixel bounds\n";
            row.passed();
        }
    }
} // namespace
#endif

int main(int argc, char** argv) try {
    const bool listing = JuicerTests::list_requested(argc, argv);
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
    constexpr std::array<std::pair<const char*, JuicerCudaTest::StateLifetimeCase>, 12> lifetime_cases{{{"supplied-stream", JuicerCudaTest::StateLifetimeCase::SuppliedStream},
                                                                                                        {"present-null-stream", JuicerCudaTest::StateLifetimeCase::PresentNullStream},
                                                                                                        {"absent-stream", JuicerCudaTest::StateLifetimeCase::AbsentStream},
                                                                                                        {"cancel-after-prepare", JuicerCudaTest::StateLifetimeCase::CancelAfterPrepare},
                                                                                                        {"cancel-after-execute", JuicerCudaTest::StateLifetimeCase::CancelAfterExecute},
                                                                                                        {"pinned-allocation", JuicerCudaTest::StateLifetimeCase::PinnedAllocation},
                                                                                                        {"pinned-event-creation", JuicerCudaTest::StateLifetimeCase::PinnedEventCreation},
                                                                                                        {"pinned-copy", JuicerCudaTest::StateLifetimeCase::PinnedCopy},
                                                                                                        {"pinned-event-record", JuicerCudaTest::StateLifetimeCase::PinnedEventRecord},
                                                                                                        {"pinned-synchronization", JuicerCudaTest::StateLifetimeCase::PinnedSynchronization},
                                                                                                        {"idle-retirement", JuicerCudaTest::StateLifetimeCase::IdleRetirement},
                                                                                                        {"deferred-destruction", JuicerCudaTest::StateLifetimeCase::DeferredDestruction}}};
#endif
    if (listing) {
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        if (argc == 4 && std::string_view(argv[1]) == "--prepared-case") {
            JuicerTests::list_case("processor_operation", std::string("prepared_boundary/") + argv[2]);
            JuicerTests::list_case("prepared_boundary", argv[2]);
            for (const char* path : {"boundary-cold", "boundary-warm", "direct-executor", "processor-render"})
                JuicerTests::list_case("processor_render_path", std::string(argv[2]) + "/" + path);
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "--state-contract") {
            JuicerTests::list_case("processor_operation", std::string("state_contract/") + argv[2]);
            JuicerTests::list_case("state_contract", argv[2]);
            for (const char* path : {"boundary-cold", "boundary-warm", "direct-executor", "processor-render", "grain-boundary"})
                JuicerTests::list_case("processor_render_path", std::string(argv[2]) + "/" + path);
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "--state-lifetime") {
            JuicerTests::list_case("processor_operation", "state_lifetime");
            for (const char* route : {"negative-direct", "negative-print"})
                JuicerTests::list_case("state_lifetime_route", route);
            for (const char* route : {"negative-direct", "negative-print"})
                for (const auto& [name, lifetime_case] : lifetime_cases) {
                    (void)lifetime_case;
                    JuicerTests::list_case("state_lifetime", std::string(route) + "/" + name);
                }
            for (const char* route : {"negative-direct", "negative-print"})
                for (const char* pass : {"cold", "warm"})
                    JuicerTests::list_case("production-state-contract", std::string(route) + "/" + pass);
            return 0;
        }
#else
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        if (argc == 3 && std::string_view(argv[1]) == "--workspace-transitions") {
            JuicerTests::list_case("processor_operation", "workspace_transitions");
            for (const char* route : {"negative-direct", "negative-print"}) {
                for (const char* mode : {"competing", "malformed"})
                    JuicerTests::list_case("workspace_transition", std::string(route) + "/" + mode);
                for (const char* carrier : {"disabled", "dedicated", "camera-diffusion"})
                    JuicerTests::list_case("workspace_transition", std::string(route) + "/" + carrier);
            }
            for (const auto& id : workspace_candidate_ids())
                JuicerTests::list_case("workspace_candidate", id);
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--noise-lifetime") {
            JuicerTests::list_case("processor_operation", "noise_lifetime");
            for (const char* id : {"published_noise_call", "unavailable_request", "scratch_failure_recovery"})
                JuicerTests::list_case("noise_lifetime", id);
            for (const char* route : {"direct", "print"})
                for (const char* cache : {"cold", "warm"})
                    JuicerTests::list_case("noise_published_call", std::string(route) + "/" + cache);
            for (const char* route : {"direct", "print"})
                for (const char* availability : {"missing-state", "missing-instance", "uninitialized-instance", "missing-state-uninitialized"})
                    for (const char* cancellation : {"cancel", "no-cancel"})
                        JuicerTests::list_case("noise_unavailable_request", std::string(route) + "/" + availability + "/" + cancellation);
            for (const char* route : {"direct", "print"}) {
                JuicerTests::list_case("noise_scratch_failure", route);
                for (const char* failure : {"retire-cancel", "retire-abort-failure"})
                    JuicerTests::list_case("noise_scratch_failure", std::string(route) + "/" + failure);
            }
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--failure-order") {
            JuicerTests::list_case("processor_operation", "failure_order");
            JuicerTests::list_case("failure_order", kDiffusionComponentsPanicCase);
            for (const char* route : {"failure-direct", "failure-print"}) {
                for (const char* delivery : {"success", "failure", "throw"})
                    JuicerTests::list_case("failure_order", std::string(route) + "/delivery/" + delivery);
                for (unsigned status = 0; status < 4; ++status)
                    for (unsigned wording = 0; wording < 3; ++wording)
                        JuicerTests::list_case("failure_order", std::string(route) + "/status/" + std::to_string(status) + "/wording/" + std::to_string(wording));
                for (const char* terminal : {"context-loss", "mismatching-context", "gate-reentry", "active-submission", "non-dir-failure", "long-diagnostic"})
                    JuicerTests::list_case("failure_order", std::string(route) + "/terminal/" + terminal);
            }
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "--resource-failure") {
            JuicerTests::list_case("processor_operation", std::string("resource_failure/") + argv[2]);
            JuicerTests::list_case("resource_failure", argv[2]);
            if (std::string_view(argv[2]) == "ScannerLut") {
                for (const char* route : {"lut-direct", "lut-print", "lut-positive-direct", "lut-positive-print"}) {
                    for (unsigned position = 1; position <= 6; ++position)
                        JuicerTests::list_case("scanner_lut_upload", std::string(route) + "/position/" + std::to_string(position));
                    JuicerTests::list_case("scanner_lut_upload", std::string(route) + "/warm-render");
                }
            } else if (std::string_view(argv[2]) == "DefectFence") {
                for (const char* route : {"defect-direct", "defect-print"})
                    for (const char* timing : {"event-create", "event-record"})
                        for (unsigned code = 0; code < 3; ++code)
                            JuicerTests::list_case("defect_fence", std::string(route) + "/" + timing + "/" + std::to_string(code));
            } else if (std::string_view(argv[2]) == "GrainUpload") {
                for (const char* route : {"grain-direct", "grain-print"})
                    for (unsigned wording = 0; wording < 2; ++wording)
                        for (unsigned upload = 0; upload < 8; ++upload)
                            JuicerTests::list_case("grain_upload", std::string(route) + "/wording/" + std::to_string(wording) + "/upload/" + std::to_string(upload));
            }
            return 0;
        }
        if (argc == 5 && (std::string_view(argv[1]) == "--effects-frame" || std::string_view(argv[1]) == "--grain-frame" || std::string_view(argv[1]) == "--dir-frame")) {
            using Json = nlohmann::json;
            const std::filesystem::path fixturePath(argv[2]);
            const Json manifest = Json::parse(std::ifstream(fixturePath));
            const std::string_view mode(argv[1]);
            const auto logicalCapture = fixturePath.parent_path() / manifest.at("presets").at(argv[3]).at("native").get<std::string>();
            const Json captured = mode == "--grain-frame"
                                      ? Json::parse(std::ifstream(logicalCapture))
                                      : load_shared_fixture(logicalCapture, "manifest.json");
            const std::string prefix = mode == "--effects-frame" ? "effects_frame" : mode == "--grain-frame" ? "grain_frame"
                                                                                                             : "dir_frame";
            JuicerTests::list_case("processor_operation", prefix);
            if (mode == "--effects-frame") {
                const std::array caseNames{"mixed", "active-zero-work", "active-zero-work-recipe-zero", "inactive", "inactive-bad-identity", "film_dust", "film_scratch", "gate_dust", "gate_scratch", "weave-1", "weave-nan", "geometry-roi", "geometry-negative", "weave-roi", "film-time--1.75", "film_dust-ignored-frame_rate", "gate_dust-ignored-frame_time", "gate_dust-ignored-frame_rate", "gate_dust-ignored-pixel_size_um", "weave-frame_rate-0", "film_format_long_edge_mm-zero"};
                for (unsigned route = 0; route < 4; ++route) {
                    for (const auto& row : captured.at("products"))
                        if (row.at("route") == route)
                            JuicerTests::list_case(prefix, "route-" + std::to_string(route) + "/product/" + row.at("spec").at("id").get<std::string>());
                    for (const auto& row : captured.at("cases")) {
                        const auto name = row.at("input").at("id").get<std::string>();
                        if (std::find(caseNames.begin(), caseNames.end(), name) != caseNames.end())
                            JuicerTests::list_case(prefix, "route-" + std::to_string(route) + "/case/" + name);
                    }
                }
            } else if (mode == "--grain-frame") {
                const std::array caseNames{"time--1.75", "time-1.5", "fps-23.976", "fps-0.2", "negative-origin", "partial-unblurred", "sigma-0.4", "sigma-0.4000001", "new-session", "new-clip", "mix", "S4D6-R1", "pixel-half", "pixel-eight", "geometry-96x71", "geometry-offset-96x71", "layers-film_layers_hash", "model-0", "frame-frame_time"};
                for (unsigned route = 0; route < 4; ++route) {
                    for (const auto& row : captured.at("products"))
                        if (row.at("route") == route)
                            JuicerTests::list_case(prefix, "route-" + std::to_string(route) + "/" + grain_frame_product_id(row.at("spec")));
                    for (const auto& row : captured.at("cases")) {
                        const auto& input = row.at("input");
                        const auto name = input.at("name").get<std::string>();
                        const auto logical = name.starts_with("positive/") ? name.substr(9) : name;
                        if (input.at("capture_polarity") == route / 2 && input.at("film_polarity") == route / 2 &&
                            std::find(caseNames.begin(), caseNames.end(), logical) != caseNames.end())
                            JuicerTests::list_case(prefix, "route-" + std::to_string(route) + "/case/" + name);
                    }
                }
            } else {
                const std::array caseNames{"inactive", "size-zero", "size-nan", "core-word-1", "core-word-1042983595", "fir", "core-word-1077936128", "tail-mixed", "tail-weight-one", "tail-zero", "pixel-change", "negative-origin", "pixel-zero", "offset", "tail-radius", "core-inf", "active-hash-zero"};
                for (unsigned route = 0; route < 4; ++route) {
                    for (const auto& row : captured.at("products"))
                        if (row.at("route") == route)
                            JuicerTests::list_case(prefix, "route-" + std::to_string(route) + "/product/" + row.at("input").at("name").get<std::string>());
                    for (const char* name : caseNames)
                        JuicerTests::list_case(prefix, "route-" + std::to_string(route) + "/case/" + name);
                }
            }
            return 0;
        }
#endif
        if (argc == 3 && std::string_view(argv[1]) == "--sequential-owners") {
            JuicerTests::list_case("processor_operation", "sequential_owners");
            for (const char* name : {"negative-direct", "negative-print", "positive-direct", "positive-print", "combined-print", "glare-plus-zero", "glare-minus-zero"})
                JuicerTests::list_case("processor_reference", name);
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--cutover-contract") {
            JuicerTests::list_case("processor_operation", "cutover_contract");
            for (const char* name : {"negative-direct", "negative-print", "positive-direct", "positive-print", "combined-print", "glare-plus-zero", "glare-minus-zero"})
                JuicerTests::list_case("processor_reference", name);
            for (const char* name : {"negative-direct", "negative-print", "positive-direct", "positive-print", "combined-print"})
                for (const char* family : {"film-dust", "film-scratch", "gate-dust", "gate-scratch", "gate-weave"})
                    JuicerTests::list_case(std::string("physical_effect_pixels/") + name, std::string(name) + "/" + family);
            for (const char* name : {"negative-direct", "negative-print"}) {
                for (const char* id : {"disjoint-rows", "preflight-keeps-submission", "dir-disabled-submission", "initial-uniform", "changed-geometry-time-scale", "sequential-supplied-stream", "worker-thread", "second-context", "throwing-cancellation", "render-after-cancellation"})
                    JuicerTests::list_case(std::string("cutover_callback/") + name, std::string(name) + "/" + id);
                for (const unsigned checkpoint : {1U, 2U, 3U, 4U, 5U})
                    JuicerTests::list_case(std::string("cutover_callback/") + name, std::string(name) + "/cancel/" + std::to_string(checkpoint));
            }
            return 0;
        }
#if defined(JUICER_ACCEPTED_CAPTURE_PATH)
        if (argc == 3 && std::string_view(argv[1]) == "--accepted-clean-captures") {
            using Json = nlohmann::json;
            const auto manifest = Json::parse(std::ifstream(JUICER_ACCEPTED_CAPTURE_PATH));
            JuicerTests::list_case("processor_operation", "accepted_clean_captures");
            for (const auto& row : manifest.at("cases"))
                JuicerTests::list_case("accepted_capture", row.at("name").get_ref<const std::string&>());
            return 0;
        }
#endif
        JuicerTests::list_case("processor_operation", "processor_reference");
        for (const char* name : {"negative-direct", "negative-print", "positive-direct", "positive-print", "combined-print", "glare-plus-zero", "glare-minus-zero"})
            JuicerTests::list_case("processor_reference", name);
        return 0;
#endif
        return 2;
    }
    JuicerTests::CaseLog case_log;
    std::string operation_id = "processor_reference";
    if (argc > 1) {
        const std::string_view mode(argv[1]);
        if (mode == "--prepared-case" && argc > 2)
            operation_id = std::string("prepared_boundary/") + argv[2];
        else if (mode == "--state-contract" && argc > 2)
            operation_id = std::string("state_contract/") + argv[2];
        else if (mode == "--state-lifetime")
            operation_id = "state_lifetime";
        else if (mode == "--sequential-owners")
            operation_id = "sequential_owners";
        else if (mode == "--cutover-contract")
            operation_id = "cutover_contract";
        else if (mode == "--accepted-clean-captures")
            operation_id = "accepted_clean_captures";
        else if (mode == "--workspace-transitions")
            operation_id = "workspace_transitions";
        else if (mode == "--noise-lifetime")
            operation_id = "noise_lifetime";
        else if (mode == "--failure-order")
            operation_id = "failure_order";
        else if (mode == "--resource-failure" && argc > 2)
            operation_id = std::string("resource_failure/") + argv[2];
        else if (mode == "--effects-frame")
            operation_id = "effects_frame";
        else if (mode == "--grain-frame")
            operation_id = "grain_frame";
        else if (mode == "--dir-frame")
            operation_id = "dir_frame";
    }
    JuicerTests::CaseEvent operation_event("processor_operation", operation_id);
    CallbackSession runtime;
    try {
        runtime.create({});
        require_cuda(cudaSetDevice(0), "select device");
        require_cuda(cudaFree(nullptr), "initialize CUDA");
#if defined(JUICER_ACCEPTED_CAPTURE_PATH)
        if (argc == 2 && std::string(argv[1]) == "--accepted-clean-captures") {
            run_accepted_clean_captures(runtime);
            if (runtime.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("accepted capture owner cleanup failed");
            }
            operation_event.passed();
            case_log.finish();
            return 0;
        }
#endif
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        if (argc == 2 && std::string(argv[1]) == "--workspace-transitions") {
            run_workspace_transitions(runtime);
            if (runtime.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("workspace transitions owner cleanup failed");
            }
            operation_event.passed();
            case_log.finish();
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--effects-frame") {
            run_effects_frame_cases(runtime, argv[2], argv[3]);
            if (runtime.close().category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("effects caller owner cleanup failed");
            operation_event.passed();
            case_log.finish();
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--grain-frame") {
            run_grain_frame_cases(runtime, argv[2], argv[3]);
            if (runtime.close().category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("grain caller owner cleanup failed");
            operation_event.passed();
            case_log.finish();
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--dir-frame") {
            run_dir_frame_cases(runtime, argv[2], argv[3]);
            if (runtime.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("DIR caller owner cleanup failed");
            }
            operation_event.passed();
            case_log.finish();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--noise-lifetime") {
            const std::vector<std::string> ids{"published_noise_call", "unavailable_request", "scratch_failure_recovery"};
            JuicerTests::Cohort rows("noise_lifetime", ids);
            {
                auto row = rows.case_event("published_noise_call");
                run_published_noise_call_cases(runtime);
                row.passed();
            }
            {
                auto row = rows.case_event("unavailable_request");
                run_processor_unavailable_request_cases(runtime);
                row.passed();
            }
            {
                auto row = rows.case_event("scratch_failure_recovery");
                run_scratch_noise_failure_recovery_cases(runtime);
                row.passed();
            }
            if (runtime.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("noise lifetime owner cleanup failed");
            }
            operation_event.passed();
            case_log.finish();
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--resource-failure") {
            const std::string resourceFailure = argv[2];
            JuicerTests::CaseEvent resource_event("resource_failure", resourceFailure);
            if (resourceFailure == "DefectFence") {
                run_defect_fence_failure_cases(runtime);
            } else if (resourceFailure == "ScannerLut") {
                run_scanner_lut_upload_failure_cases(runtime);
            } else if (resourceFailure == "GrainUpload") {
                run_grain_upload_failure_cases(runtime);
            } else {
                throw std::runtime_error("unknown resource failure case: " + resourceFailure);
            }
            const auto closed = runtime.shutdown();
            if (closed.category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("terminal cleanup after native failure failed");
            }
            if (runtime.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("resource failure owner cleanup failed");
            }
            resource_event.passed();
            operation_event.passed();
            case_log.finish();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--failure-order") {
            run_failure_order_cases(runtime);
            const auto closed = runtime.shutdown();
            if (closed.category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("terminal cleanup after native failure failed");
            }
            if (runtime.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("failure order owner cleanup failed");
            }
            operation_event.passed();
            case_log.finish();
            return 0;
        }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        const bool stateContract = argc == 3 && std::string(argv[1]) == "--state-contract";
        const bool stateLifetime = argc == 3 && std::string(argv[1]) == "--state-lifetime";
        if (argc != 3 || (!stateContract && !stateLifetime && std::string(argv[1]) != "--prepared-case")) {
            throw std::runtime_error("usage: JuicerPreparedBoundaryProbe --prepared-case <name>");
        }
        const std::string preparedCase = argv[2];
        bool preparedCaseFound = false;
        constexpr bool emit = false;
#else
        const bool cutoverContract = argc == 2 && std::string(argv[1]) == "--cutover-contract";
        const bool sequentialOwners = argc == 2 && std::string(argv[1]) == "--sequential-owners";
        const bool emit = argc == 2 && std::string(argv[1]) == "--emit-reference";
        if (argc != 1 && !emit && !sequentialOwners && !cutoverContract) {
            throw std::runtime_error("usage: JuicerProcessorReferenceProbe [--emit-reference]");
        }
#endif
        const Case cases[]{{"negative-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                           {"negative-print", Spektrafilm::ScanRoute::NegativePrintScan, 1, 4, 0, 0},
                           {"positive-direct", Spektrafilm::ScanRoute::PositiveDirectScan, 2, 3, 11, -3},
                           {"positive-print", Spektrafilm::ScanRoute::PositivePrintScan, 0, 4, 0, 0},
                           {"combined-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, true},
                           {"glare-plus-zero", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, false, true, false},
                           {"glare-minus-zero", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, false, true, true}
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                           ,
                           {"mallett-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 1, 4, 0, 0},
                           {"arctic-print", Spektrafilm::ScanRoute::NegativePrintScan, 2, 4, 0, 0}
#endif
        };
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        if (stateLifetime) {
            const std::array<std::string, 2> route_ids{{"negative-direct", "negative-print"}};
            JuicerTests::Cohort route_rows("state_lifetime_route", std::vector<std::string>(route_ids.begin(), route_ids.end()));
            std::vector<std::string> intended;
            intended.reserve(2 * lifetime_cases.size());
            for (const char* route : {"negative-direct", "negative-print"})
                for (const auto& [name, lifetime_case] : lifetime_cases) {
                    (void)lifetime_case;
                    intended.push_back(std::string(route) + "/" + name);
                }
            JuicerTests::Cohort lifetime_rows("state_lifetime", intended);
            for (std::size_t index = 0; index < 2; ++index) {
                auto route_row = route_rows.case_event(cases[index].name);
                JuicerTestSupport::InstanceFixture unused;
                MembershipRetirement unusedRetirement{runtime, unused};
                for (const auto& [name, lifetimeCase] : lifetime_cases) {
                    auto row = lifetime_rows.case_event(std::string(cases[index].name) + "/" + name);
                    (void)render_case(runtime, cases[index], parameters_for(cases[index]), unused, false, ExecutionPath::StateLifetime, DestinationLayout::Matching, 1, lifetimeCase);
                    row.passed();
                }
                if (!unusedRetirement.close())
                    throw std::runtime_error("held-state lifetime instance retirement failed");
                if (runtime.close().category != FJ_STATUS_SUCCESS)
                    throw std::runtime_error("held-state lifetime owner close");
                if (index == 0) {
                    runtime.create({});
                }
                route_row.passed();
            }
            runtime.create({});
            const JuicerTests::Cohort production_rows("production-state-contract", {"negative-direct/cold", "negative-direct/warm", "negative-print/cold", "negative-print/warm"});
            const auto expected = production_pixels();
            for (std::size_t index = 0; index < expected.size(); ++index) {
                const auto parameters = parameters_for(cases[index]);
                const auto input = parameters.record();
                JuicerCudaTest::check_production_state_contract(runtime.borrow(), input, JuicerTestSupport::hash_input_identity(parameters), static_cast<std::uint32_t>(cases[index].route), expected[index], production_rows);
            }
            if (runtime.close().category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("production state contract Runtime close");
            operation_event.passed();
            case_log.finish();
            return 0;
        }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        constexpr bool numericalFixture = false;
#else
        const bool numericalFixture = !emit && !sequentialOwners && !cutoverContract;
#endif
        std::ifstream fixture;
        if (numericalFixture) {
            fixture.open(JUICER_PROCESSOR_REFERENCE_PATH);
            if (!fixture) {
                throw std::runtime_error("processor reference fixture unavailable");
            }
        }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
        void* initialContext = nullptr;
        std::uint64_t previousEpoch = 0;
        if (sequentialOwners) {
            CUcontext current = nullptr;
            if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
                throw std::runtime_error("missing sequential-owner CUDA context");
            }
            initialContext = current;
        }
#endif
        std::vector<float> freshPlus;
        std::vector<float> freshMinus;
        std::vector<std::string> intended_cases;
        intended_cases.reserve(std::size(cases));
        for (const Case& test : cases) {
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
            if (preparedCase == test.name)
                intended_cases.emplace_back(test.name);
#else
            intended_cases.emplace_back(test.name);
#endif
        }
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        const std::string_view case_caller = stateContract ? "state_contract" : "prepared_boundary";
        std::vector<std::string> render_path_ids;
        render_path_ids.reserve(stateContract ? 5 : 4);
        for (const Case& test : cases) {
            if (preparedCase != test.name)
                continue;
            for (const char* path : {"boundary-cold", "boundary-warm", "direct-executor", "processor-render"})
                render_path_ids.push_back(std::string(test.name) + "/" + path);
            if (stateContract)
                render_path_ids.push_back(std::string(test.name) + "/grain-boundary");
        }
        JuicerTests::Cohort render_path_rows("processor_render_path", render_path_ids);
#else
        const std::string_view case_caller = "processor_reference";
#endif
        JuicerTests::Cohort cases_cohort(std::string(case_caller), intended_cases);
        std::vector<std::unique_ptr<JuicerTests::CaseEvent>> case_events;
        case_events.reserve(intended_cases.size());
        for (const Case& test : cases) {
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
            if (preparedCase != test.name) {
                if (numericalFixture) {
                    fixture.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
                continue;
            }
#endif
            case_events.push_back(std::make_unique<JuicerTests::CaseEvent>(case_caller, test.name));
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
            preparedCaseFound = true;
            // Each CTest case starts a fresh process and owner. C crosses the
            // boundary first, so no other caller can hide its table uploads.
            std::cerr << "cold C case: " << test.name << '\n';
            JuicerTestSupport::InstanceFixture boundaryState;
            MembershipRetirement boundaryStateRetirement{runtime, boundaryState};
            JuicerTests::CaseEvent boundary_path("processor_render_path", std::string(test.name) + "/boundary-cold");
            const auto boundary = render_case(runtime, test, parameters_for(test), boundaryState, false, stateContract ? ExecutionPath::StateContract : ExecutionPath::Boundary);
            JuicerTests::CaseEvent warm_boundary_path("processor_render_path", std::string(test.name) + "/boundary-warm");
            const auto warmBoundary = render_case(runtime, test, parameters_for(test), boundaryState, false, stateContract ? ExecutionPath::State : ExecutionPath::Boundary, DestinationLayout::Matching, 1, JuicerCudaTest::StateLifetimeCase::SuppliedStream, PublishedRequestAvailability::Complete, "warm");
            JuicerTests::CaseEvent direct_path("processor_render_path", std::string(test.name) + "/direct-executor");
            const auto direct = [&] {
                JuicerTestSupport::InstanceFixture directState;
                MembershipRetirement directStateRetirement{runtime, directState};
                return render_case(runtime, test, parameters_for(test), directState, false, ExecutionPath::Direct);
            }();
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
            JuicerTests::CaseEvent processor_path("processor_render_path", std::string(test.name) + "/processor-render");
#endif
            std::cerr << "processor case: " << test.name << '\n';
            const std::vector<float> pixels = [&] {
                JuicerTestSupport::InstanceFixture state;
                MembershipRetirement stateRetirement{runtime, state};
                auto rendered = render_case(runtime, test, parameters_for(test), state);
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
                if (sequentialOwners) {
                    const auto observed = runtime.submission(state);
                    const JuicerCuda::ResourceManager::DeviceContextKey key{observed.context.device_id, reinterpret_cast<void*>(observed.context.context)};
                    JuicerCuda::ResourceManager::RegistryContextSnapshot admitted;
                    if (key.contextOpaque != initialContext ||
                        !JuicerCuda::ResourceManager::registry_begin_submission(key, admitted)) {
                        throw std::runtime_error("sequential owner changed or failed to admit the live context");
                    }
                    const bool ended = JuicerCuda::ResourceManager::registry_note_submission_end(key);
                    if (!ended || admitted.contextEpoch <= previousEpoch) {
                        throw std::runtime_error("owner recreation did not advance the native epoch");
                    }
                    std::cerr << "owner transition case=" << test.name << " previous_epoch=" << previousEpoch << " admitted_epoch=" << admitted.contextEpoch << '\n';
                    previousEpoch = admitted.contextEpoch;
                }
#endif
                return rendered;
            }();
            if (test.signedZeroGlare) {
                (test.negativeZero ? freshMinus : freshPlus) = pixels;
            }
            if (emit) {
                std::cout << test.name << ' ' << pixels.size();
                for (float pixel : pixels) {
                    std::cout << ' ' << std::setprecision(std::numeric_limits<float>::max_digits10) << pixel;
                }
                std::cout << '\n';
            } else {
                std::vector<float> expected = pixels;
                if (numericalFixture) {
                    std::string name;
                    std::size_t count = 0;
                    fixture >> name >> count;
                    if (!fixture || name != test.name || count != pixels.size()) {
                        throw std::runtime_error(std::string(test.name) + ": fixture row mismatch");
                    }
                    expected.resize(count);
                    for (float& pixel : expected) {
                        fixture >> pixel;
                    }
                    if (!fixture) {
                        throw std::runtime_error(std::string(test.name) + ": incomplete fixture row");
                    }
                    nlohmann::json originalWords = nlohmann::json::array();
                    for (float value : expected)
                        originalWords.push_back(std::bit_cast<std::uint32_t>(value));
                    const auto selectedWords = PrintIlluminantFixtures::apply(
                        std::string(JUICER_PROCESSOR_REFERENCE_PATH) + "." + test.name, originalWords);
                    if (selectedWords.size() != expected.size())
                        throw std::runtime_error(std::string(test.name) + ": selected fixture shape mismatch");
                    for (std::size_t i = 0; i < expected.size(); ++i)
                        expected[i] = std::bit_cast<float>(selectedWords.at(i).get<std::uint32_t>());
                    compare_pixels(test.name, pixels, expected);
                }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
                if (cutoverContract && (std::string_view(test.name) == "negative-direct" || std::string_view(test.name) == "negative-print")) {
                    check_cutover_callbacks(runtime, test, expected);
                }
                if (cutoverContract && !test.signedZeroGlare)
                    check_physical_effects_pixels(runtime, test);
                if (sequentialOwners && std::string_view(test.name) == "negative-direct") {
                    Case uniform = test;
                    uniform.uniform = true;
                    std::vector<float> uniformExpected(expected.size());
                    for (std::size_t sample = 0; sample < uniformExpected.size(); ++sample) {
                        uniformExpected[sample] = expected[sample % 3];
                    }
                    // At this fixture's physical scale the DIR FIR has no nonzero
                    // off-center float samples. Normalized diffusion preserves a
                    // constant exposure. Both controls use the immutable first pixel.
                    {
                        JuicerTestSupport::InstanceFixture control;
                        MembershipRetirement controlRetirement{runtime, control};
                        compare_pixels("uniform direct control", render_case(runtime, uniform, parameters_for(uniform), control), uniformExpected);
                    }
                    if (runtime.close().category != FJ_STATUS_SUCCESS) {
                        throw std::runtime_error("direct diffusion predecessor close failed");
                    }
                    runtime.create({});
                    {
                        auto parameters = parameters_for(uniform);
                        parameters.values.camera_diffusion.active = true;
                        JuicerTestSupport::InstanceFixture diffusionState;
                        MembershipRetirement diffusionStateRetirement{runtime, diffusionState};
                        compare_pixels("uniform direct diffusion", render_case(runtime, uniform, parameters, diffusionState), uniformExpected);
                        JuicerCuda::ResourceManager::RegistryContextSnapshot admitted;
                        const auto observed = runtime.submission(diffusionState);
                        const JuicerCuda::ResourceManager::DeviceContextKey key{observed.context.device_id, reinterpret_cast<void*>(observed.context.context)};
                        if (key.contextOpaque != initialContext || !JuicerCuda::ResourceManager::registry_begin_submission(key, admitted)) {
                            throw std::runtime_error("direct diffusion context admission failed");
                        }
                        const bool ended = JuicerCuda::ResourceManager::registry_note_submission_end(key);
                        if (!ended || admitted.contextEpoch <= previousEpoch) {
                            throw std::runtime_error("direct diffusion did not use a later admitted epoch");
                        }
                        previousEpoch = admitted.contextEpoch;
                        std::cerr << "uniform direct diffusion matches immutable constant sample at epoch=" << previousEpoch << '\n';
                    }
                }
                if (sequentialOwners && !test.combined) {
                    JuicerTestSupport::InstanceFixture originState;
                    MembershipRetirement originStateRetirement{runtime, originState};
                    compare_pixels(std::string(test.name) + ": independent destination origin", render_case(runtime, test, parameters_for(test), originState, false, ExecutionPath::Processor, DestinationLayout::Offset), expected);
                    JuicerTestSupport::InstanceFixture rejectedState;
                    MembershipRetirement rejectedStateRetirement{runtime, rejectedState};
                    bool fatal = false;
                    try {
                        (void)render_case(runtime, test, parameters_for(test), rejectedState, false, ExecutionPath::Processor, DestinationLayout::Uncovered);
                    } catch (const CallbackStatus& error) {
                        fatal = error.status() == kOfxStatErrFatal;
                    }
                    if (!fatal || runtime.submission(rejectedState).valid) {
                        throw std::runtime_error("uncovered destination was not rejected before submission latch publication");
                    }
                }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                RenderAssertions::compare_bits(std::string(test.name) + ": processor/direct", pixels, direct);
                RenderAssertions::compare_bits(std::string(test.name) + ": direct/cold C", direct, boundary);
                RenderAssertions::compare_bits(std::string(test.name) + ": cold/warm C", boundary, warmBoundary);
                std::cerr << test.name << ": processor/direct/cold C/warm C bit-exact\n";
                if (stateContract) {
                    auto grainParameters = parameters_for(test);
                    grainParameters.values.grain.active = true;
                    JuicerTestSupport::InstanceFixture grainState;
                    MembershipRetirement grainStateRetirement{runtime, grainState};
                    JuicerTests::CaseEvent grain_path("processor_render_path", std::string(test.name) + "/grain-boundary");
                    (void)render_case(runtime, test, grainParameters, grainState, false, ExecutionPath::Boundary, DestinationLayout::Matching, 2);
                    grain_path.passed();
                }
#endif
            }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
            if (sequentialOwners) {
                if (runtime.close().category != FJ_STATUS_SUCCESS) {
                    throw std::runtime_error("sequential native owner failed to close");
                }
                runtime.create({});
                CUcontext current = nullptr;
                if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != initialContext) {
                    throw std::runtime_error("owner close/recreate replaced the live CUDA context");
                }
            }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
            boundary_path.passed();
            warm_boundary_path.passed();
            direct_path.passed();
            processor_path.passed();
#endif
        }
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        if (!preparedCaseFound) {
            throw std::runtime_error("unknown prepared-boundary case: " + preparedCase);
        }
#else
        if (!emit) {
            if (std::equal(freshPlus.begin(), freshPlus.end(), freshMinus.begin())) {
                throw std::runtime_error("signed-zero fresh print renders did not differ");
            }
            JuicerTestSupport::InstanceFixture transition;
            MembershipRetirement transitionRetirement{runtime, transition};
            compare_pixels("glare plus to minus initial", render_case(runtime, cases[5], parameters_for(cases[5]), transition), freshPlus);
            compare_pixels("glare plus to minus", render_case(runtime, cases[6], parameters_for(cases[6]), transition), freshMinus);
            compare_pixels("glare minus to plus", render_case(runtime, cases[5], parameters_for(cases[5]), transition), freshPlus);
        }
        JuicerTestSupport::InstanceFixture emptyWindowState;
        MembershipRetirement emptyWindowStateRetirement{runtime, emptyWindowState};
        (void)render_case(runtime, cases[0], parameters_for(cases[0]), emptyWindowState, true);
        if (!emptyWindowStateRetirement.close())
            throw std::runtime_error("empty-window instance retirement failed");
#endif
        if (runtime.close().category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("final native owner close failed");
        }
        for (auto& event : case_events)
            event->passed();
        operation_event.passed();
        case_log.finish();
        return 0;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        const auto& failure = error.failure;
        (void)std::fprintf(stderr, "held rendering status=%u api=%u native=%d %s\n", failure.status.category, failure.status.api, failure.status.native_code, failure.diagnostic.c_str());
        try {
            runtime.shutdown();
        } catch (...) {
            (void)std::fputs("shutdown failed after processor test error\n", stderr);
        }
        return 1;
    } catch (const std::exception& error) {
        (void)std::fprintf(stderr, "%s\n", error.what());
        try {
            runtime.shutdown();
        } catch (...) {
            (void)std::fputs("shutdown failed after processor test error\n", stderr);
        }
        return 1;
    }
} catch (const std::exception& error) {
    (void)std::fprintf(stderr, "processor test setup failed: %s\n", error.what());
    return 1;
} catch (...) {
    (void)std::fputs("processor test setup failed: unknown exception\n", stderr);
    return 1;
}
