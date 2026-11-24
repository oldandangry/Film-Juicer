#include "render_gpu_fixture.h"
#include "render_assertions.h"
#include "production_images.h"
#include "Cuda/Scan/JuicerCudaScanDensityCaptureTest.h"
#include "GamutCompression.h"
#include "Hash.h"
#include "../ffi/print_illuminant_fixture.h"
#include "processor_test_hooks.h"
#include "../ffi/case_results.h"
#include "../ffi/juicer_test_api.h"
#include "../ffi/preparation_fixture.h"
#include "../ffi/owner_fixture.h"
#include "module_test.h"

#include <array>
#include <bit>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda.h>

extern "C" FjStatus fj_test_ofx_instance_observe(OfxImageEffectHandle, FjInstanceObservation**, FjErrorBuffer*);

namespace {
    constexpr std::string_view kCaller = "ofx.gpu.render_trace";
    constexpr std::array kRows{"temporal/initial", "temporal/time_a", "temporal/time_b", "temporal/backward", "temporal/repeat", "temporal/preset", "temporal/reset", "temporal/undo", "temporal/redo", "temporal/invalid", "temporal/recovery", "geometry/changed_size", "geometry/half_scale", "geometry/origin", "effects/full_frame", "identity/fixed", "failure/message_order", "failure/recovery", "failure/holds", "time/valid_after_rejection"};
    constexpr std::array kSupplemental{"fixed/negative-print", "fixed/positive-direct", "fixed/positive-print", "fixed/combined-print", "fixed/glare-plus-zero", "fixed/glare-minus-zero", "time/positive_zero", "time/negative_zero", "time/negative_half", "time/below_negative_one", "time/below_positive_one", "time/nan", "time/positive_infinity", "time/negative_infinity", "abort/terminal_recovery"};
    constexpr std::array kIdentityPairs{"direct-exposure-small", "direct-exposure-zero", "print-exposure-small", "print-exposure-zero", "print-medium-exposure-small"};
    constexpr std::array kCarrierRows{"highlight/direct-scatter", "highlight/print-reflection", "highlight/positive-direct-diffusion", "highlight/positive-print-composed"};
    constexpr std::uint64_t kSession = 0x20260923;
    constexpr std::uint64_t kInstance = 0x641207;
    constexpr std::uint64_t kClip = 0x5312;

    struct FixedScenario {
        const char* name;
        const char* route;
        const char* profile;
        int reconstruction;
        int components;
        OfxPointI origin;
        std::uint64_t instance;
        bool combined;
        bool glareZero;
        bool negativeZero;
    };
    constexpr FixedScenario kFixedNegativePrint{"negative-print", "NegativePrintScan", "kodak_portra_400", 1, 4, {0, 0}, kInstance + 1, false, false, false};

    void configure_fixed_scan(ofx_test::RenderHost& fixture, ofx_test::DeviceImages& images, const FixedScenario& scenario) {
        fixture.configure(images, scenario.origin);
        fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string(scenario.route)});
        fixture.host().set_value(fixture.effect(), "FilmProfileKey", {std::string(scenario.profile)});
        fixture.host().set_value(fixture.effect(), "SpectralUpsampling", {scenario.reconstruction});
        fixture.host().set_value(fixture.effect(), "GlareActive", {1});
    }

    struct Observation {
        FjInstanceObservation* owner = nullptr;
        FjInstanceObservationView view{};
        explicit Observation(OfxImageEffectHandle effect) {
            ofx_test::require_render(fj_test_ofx_instance_observe(effect, &owner, nullptr).category == FJ_STATUS_SUCCESS && owner,
                                     "instance observation acquisition");
            ofx_test::require_render(fj_test_instance_observation_view(owner, &view, nullptr).category == FJ_STATUS_SUCCESS,
                                     "instance observation view");
        }
        ~Observation() {
            if (owner)
                (void)fj_test_instance_observation_release(owner, nullptr);
        }
        Observation(const Observation&) = delete;
        Observation& operator=(const Observation&) = delete;
    };
    FjOfxRenderObservation render_observation(OfxImageEffectHandle effect) {
        FjOfxRenderObservation value{};
        ofx_test::require_render(fj_test_ofx_render_observe(effect, &value) == 1, "render observation");
        return value;
    }
    void tokens() {
        fj_test_ofx_render_tokens(kSession, kInstance, kClip);
    }
    void change(ofx_test::RenderHost& fixture, const char* name, const char* reason, double time) {
        auto* args = fixture.host().change_args({reason, kOfxTypeParameter, name}, time);
        ofx_test::require_render(fixture.call(kOfxActionInstanceChanged, fixture.effect(), args) == kOfxStatReplyDefault, "parameter change callback");
    }
    std::vector<float> render(ofx_test::RenderHost& fixture, ofx_test::DeviceImages& images, double time, std::uint64_t instance = kInstance) {
        fixture.host().set_property(fixture.host().action_args(), kOfxPropTime, {time});
        images.reset();
        fj_test_ofx_render_tokens(kSession, instance, kClip);
        ofx_test::require_render(fixture.render() == kOfxStatOK, "trace render");
        const auto seen = render_observation(fixture.effect());
        ofx_test::require_render(seen.native_entry_count == 1 && seen.session_seed == kSession && seen.instance_token == instance &&
                                     seen.clip_token == kClip && seen.frame_time == time && seen.frame_rate == 24.0 &&
                                     seen.frame_token == static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(time))) &&
                                     seen.submission_id != 0 && seen.backing_identity != 0 && !seen.fatal,
                                 "render frame, identity or submission observation");
        return images.pixels(images.read());
    }

    class DensityCaptureScope {
    public:
        static constexpr std::size_t kPixels = 35;
        using Sample = JuicerCuda::ScanDensityCaptureTest::ScanSample;
        using Samples = std::array<Sample, kPixels>;

        explicit DensityCaptureScope(cudaStream_t stream) : stream_(stream) {
            using namespace JuicerCuda::ScanDensityCaptureTest;
            CUcontext streamContext = nullptr;
            ofx_test::require_render(cuCtxGetCurrent(&context_) == CUDA_SUCCESS && context_ &&
                                         cuStreamGetCtx(stream_, &streamContext) == CUDA_SUCCESS && streamContext == context_,
                                     "density observation requires the active frame context");
            ofx_test::require_cuda(cudaMalloc(&device_, sizeof(Samples)), "allocate scan observation");
            const auto status = cudaMemsetAsync(device_, 255, sizeof(Samples), stream_);
            if (status != cudaSuccess) {
                (void)cudaFree(device_);
                ofx_test::require_cuda(status, "clear density observation");
            }
            previous_ = swap_density_capture({device_, kPixels, 7, 5});
        }
        ~DensityCaptureScope() {
            JuicerCuda::ScanDensityCaptureTest::swap_density_capture(previous_);
            if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
                (void)cudaStreamSynchronize(stream_);
                (void)cudaFree(device_);
                CUcontext restored = nullptr;
                (void)cuCtxPopCurrent(&restored);
            }
        }
        DensityCaptureScope(const DensityCaptureScope&) = delete;
        DensityCaptureScope& operator=(const DensityCaptureScope&) = delete;

        Samples read() const {
            CUcontext context = nullptr;
            ofx_test::require_render(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context == context_, "density observation context changed");
            Samples words{};
            ofx_test::require_cuda(cudaMemcpyAsync(words.data(), device_, sizeof(words), cudaMemcpyDeviceToHost, stream_), "read density observation");
            ofx_test::require_cuda(cudaStreamSynchronize(stream_), "complete density observation");
            return words;
        }

    private:
        cudaStream_t stream_;
        CUcontext context_ = nullptr;
        Sample* device_ = nullptr;
        JuicerCuda::ScanDensityCaptureTest::DensityCapture previous_{};
    };

    nlohmann::json float_bits(std::span<const float> values) {
        auto words = nlohmann::json::array();
        for (const float value : values) {
            words.push_back(std::bit_cast<std::uint32_t>(value));
        }
        return words;
    }

    nlohmann::json source_bits(const ofx_test::Case& test, std::span<const float> source) {
        auto pixels = nlohmann::json::array();
        const auto activeWidth = static_cast<std::size_t>(test.width) * static_cast<std::size_t>(test.components);
        const auto pitch = activeWidth + ofx_test::kProductionPitchPadding;
        for (std::size_t y = 0; y < static_cast<std::size_t>(test.height); ++y) {
            for (const float pixel : source.subspan(y * pitch, activeWidth)) {
                pixels.push_back(std::bit_cast<std::uint32_t>(pixel));
            }
        }
        return pixels;
    }

    std::array<float, DensityCaptureScope::kPixels> glare_field(const DensityCaptureScope::Samples& samples) {
        std::array<float, DensityCaptureScope::kPixels> field{};
        std::ranges::transform(samples, field.begin(), &DensityCaptureScope::Sample::glarePercent);
        return field;
    }

    std::string_view profile_key(FjStringView key) {
        return key.count ? std::string_view(key.data, key.count) : std::string_view{};
    }

    std::vector<float> natural_glare_pixels(
        const FjInstanceObservationView& held,
        const FjOfxRenderObservation& seen,
        std::span<const float> glare) {
        const auto& state = held.published;
        const auto& post = state.scanner_post_effects;
        const auto& scenario = kFixedNegativePrint;
        const ofx_test::Case image{"fixed/negative-print", scenario.name, scenario.route, scenario.reconstruction, scenario.components};
        const std::array glareControls{post.glare_percent, post.glare_roughness, post.glare_blur_sigma_px};
        std::ifstream stream(JUICER_SPECTRAL_TRACE_PATH);
        ofx_test::require_render(static_cast<bool>(stream), "open versioned natural-glare fixture");
        const auto fixture = nlohmann::json::parse(stream);
        ofx_test::require_render(fixture.at("schema") == "spectral-scanner-trace-v1" && fixture.at("preset") == JUICER_SPECTRAL_TRACE_PRESET,
                                 "natural-glare fixture schema or preset changed");
        const nlohmann::json inputs{
            {"id", image.id},
            {"shape", {image.height, image.width, image.components}},
            {"route", state.route},
            {"film", profile_key(state.film_profile_key)},
            {"paper", profile_key(state.print_profile_key)},
            {"reconstruction", held.captured.spectral_upsampling_mode},
            {"frame_token", seen.frame_token},
            {"frame_rate", seen.frame_rate},
            {"source_origin", {seen.render_bounds[0], seen.render_bounds[1]}},
            {"session_token", seen.session_seed},
            {"instance_token", seen.instance_token},
            {"clip_token", seen.clip_token},
            {"grain", state.grain.active},
            {"glare", post.glare_active},
            {"glare_controls_bits", float_bits(glareControls)},
            {"glare_blur_radius", post.glare_blur_radius},
            {"recipe_hash", state.recipe_hash},
            {"scanner_hash", state.scanner_hash},
            {"source_bits", source_bits(image, ofx_test::initial_source(image))},
            {"glare_field_bits", float_bits(glare)}};
        ofx_test::require_render(fixture.at("inputs") == inputs, "natural-glare fixture inputs changed");
        const auto& expected = fixture.at("output_bits");
        const auto samples = static_cast<std::size_t>(image.components) * DensityCaptureScope::kPixels;
        ofx_test::require_render(expected.is_array() && expected.size() == samples, "natural-glare fixture output shape");
        std::vector<float> pixels;
        pixels.reserve(samples);
        for (const auto& word : expected) {
            ofx_test::require_render(word.is_number_unsigned() && word.get<std::uint64_t>() <= std::numeric_limits<std::uint32_t>::max(),
                                     "natural-glare fixture output word");
            pixels.push_back(std::bit_cast<float>(word.get<std::uint32_t>()));
        }
        return pixels;
    }

    template <typename T>
    nlohmann::json scan_sample_bits(const DensityCaptureScope::Samples& samples, T (DensityCaptureScope::Sample::*field)[3]) {
        using Word = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
        auto words = nlohmann::json::array();
        for (const auto& sample : samples) {
            for (const T value : sample.*field) {
                ofx_test::require_render(std::isfinite(value), "scan observation is missing a finite stage value");
                words.push_back(std::bit_cast<Word>(value));
            }
        }
        return words;
    }

    struct ScannerReferenceCase {
        ofx_test::Case image;
        const char* film = nullptr;
        std::uint32_t route = 0;
        const FixedScenario* trace = nullptr;
    };

    nlohmann::json capture_reference_inputs(const ScannerReferenceCase& reference) {
        const auto& test = reference.image;
        ofx_test::DeviceImages images({test.width, test.height, test.components});
        const auto instance = reference.trace ? reference.trace->instance : kInstance;
        fj_test_ofx_render_tokens(kSession, instance, kClip);
        ofx_test::RenderHost fixture;
        if (reference.trace) {
            configure_fixed_scan(fixture, images, *reference.trace);
        } else {
            ofx_test::author_parameters(fixture.host(), fixture.effect(), test);
            fixture.host().set_value(fixture.effect(), "FilmProfileKey", {std::string(reference.film)});
            fixture.host().set_value(fixture.effect(), "GlareActive", {0});
            ofx_test::configure(fixture.host(), fixture.effect(), test, {images.stream(), images.source(), images.destination()});
            change(fixture, "GlareActive", kOfxChangeUserEdited, test.time);
        }
        const auto capture = [&] {
            using Sample = DensityCaptureScope::Sample;
            DensityCaptureScope witness(images.stream());
            const auto delivered = render(fixture, images, test.time, instance);
            const auto samples = witness.read();
            const auto glare = glare_field(samples);
            ofx_test::require_render(std::ranges::all_of(glare, [&](float value) {
                                         return std::isfinite(value) && (reference.trace ? value > 0.0F : value == 0.0F);
                                     }),
                                     "scanner reference captured the wrong glare field");
            return nlohmann::json{{"developed_cmy_bits", scan_sample_bits(samples, &Sample::density)},
                                  {"glare_field_bits", float_bits(glare)},
                                  {"device_xyz_bits", scan_sample_bits(samples, &Sample::xyz)},
                                  {"device_corrected_xyz_bits", scan_sample_bits(samples, &Sample::correctedXYZ)},
                                  {"device_adapted_xyz_bits", scan_sample_bits(samples, &Sample::adaptedXYZ)},
                                  {"device_signed_rgb_bits", scan_sample_bits(samples, &Sample::signedRGB)},
                                  {"device_output_bits", float_bits(delivered)}};
        };
        auto cold = capture();
        const auto warm = capture();
        ofx_test::require_render(cold == warm, "scanner stage capture changed on a fixed grain-off render");
        const auto source = ofx_test::initial_source(test);
        std::vector<float> returnedSource(source.size());
        ofx_test::require_cuda(cudaMemcpy(returnedSource.data(), images.source(), images.bytes(), cudaMemcpyDeviceToHost), "verify scanner reference source");
        ofx_test::require_render(float_bits(source) == float_bits(returnedSource), "scanner reference source changed");

        const Observation held(fixture.effect());
        const auto& state = held.view.published;
        const auto& post = state.scanner_post_effects;
        ofx_test::require_render(state.route == reference.route && static_cast<bool>(post.glare_active) == static_cast<bool>(reference.trace) && !state.grain.active, "scanner reference captured the wrong route or stochastic controls");
        ofx_test::require_render(held.view.captured.spectral_upsampling_mode == test.reconstruction, "scanner reference captured the wrong reconstruction");
        ofx_test::require_render(!state.effects.active && post.lens_blur_radius == 0 && post.unsharp_amount == 0.0F, "scanner reference captured a spatial output effect");
        const auto& scanner = state.scanner_tables;
        const auto& color = state.output_color;
        const auto& correction = state.scanner_correction;
        ofx_test::require_render(profile_key(state.film_profile_key) == reference.film, "scanner reference published the wrong film profile");
        const auto lane = [](const FjFloatSpan& values) {
            ofx_test::require_render(values.data && values.count == 81, "missing held scanner spectrum");
            return float_bits({values.data, values.count});
        };
        const auto& encoding = color.encoding;
        const std::array coefficients{encoding.gamma, encoding.a, encoding.b, encoding.c, encoding.d, encoding.linear_cutoff};
        constexpr auto flags = FJ_COLOR_ENCODE_CCTF | FJ_COLOR_INPUT_IS_OUTPUT_SPACE | FJ_COLOR_GAMUT_COMPRESSION;
        const auto& boundary = color.gamut_cmax;
        ofx_test::require_render(color.flags == flags && encoding.cctf_kind == FJ_CCTF_SRGB && color.color_space == FJ_OUTPUT_SRGB &&
                                     boundary.data && boundary.count == Gamut::kOutputBoundaryValueCount,
                                 "scanner reference requires the complete sRGB gamut and transfer inputs");
        const auto seen = render_observation(fixture.effect());
        const std::array glareControls{post.glare_percent, post.glare_roughness, post.glare_blur_sigma_px};
        cold.update({{"schema", "scanner-reference-input-v4"},
                     {"id", test.id},
                     {"shape", {test.height, test.width, test.components}},
                     {"route", state.route},
                     {"film", reference.film},
                     {"paper", profile_key(state.print_profile_key)},
                     {"frame", test.time},
                     {"frame_rate", 24},
                     {"frame_token", seen.frame_token},
                     {"source_origin", {seen.render_bounds[0], seen.render_bounds[1]}},
                     {"session_token", seen.session_seed},
                     {"instance_token", seen.instance_token},
                     {"clip_token", seen.clip_token},
                     {"reconstruction", held.view.captured.spectral_upsampling_mode},
                     {"grain", 0},
                     {"glare", post.glare_active},
                     {"glare_controls_bits", float_bits(glareControls)},
                     {"glare_blur_radius", post.glare_blur_radius},
                     {"source_bits", source_bits(test, source)},
                     {"dye_bits", {lane(scanner.dyes_cmy[0]), lane(scanner.dyes_cmy[1]), lane(scanner.dyes_cmy[2])}},
                     {"weight_bits", {lane(scanner.weighted_xyz[0]), lane(scanner.weighted_xyz[1]), lane(scanner.weighted_xyz[2])}},
                     {"base_bits", scanner.has_baseline ? lane(scanner.baseline_min) : nlohmann::json::array()},
                     {"inverse_y_bits", std::bit_cast<std::uint32_t>(scanner.inv_yn)},
                     {"white_bits", float_bits(color.illuminant_xyz)},
                     {"cat02_bits", float_bits(color.cat02)},
                     {"xyz_to_rgb_bits", float_bits(color.xyz_to_rgb)},
                     {"gamut_hash", color.gamut_recipe_hash},
                     {"gamut_table_hash", color.gamut_table_hash},
                     {"gamut_cmax_bits", float_bits({boundary.data, boundary.count})},
                     {"gamut_lightness_bits", float_bits(color.lightness_knee)},
                     {"gamut_chroma_bits", float_bits(color.chroma_knee)},
                     {"native_to_d65_bits", float_bits(color.native_rgb_to_d65_xyz)},
                     {"d65_to_native_bits", float_bits(color.d65_xyz_to_native_rgb)},
                     {"oklab_matrices_bits", {float_bits(Gamut::kOklabXyzToLms), float_bits(Gamut::kOklabLmsToXyz), float_bits(Gamut::kOklabLmsRootToLab), float_bits(Gamut::kOklabLabToLmsRoot)}},
                     {"output_color_space", color.color_space},
                     {"output_color_flags", color.flags},
                     {"cctf_kind", encoding.cctf_kind},
                     {"cctf_bits", float_bits(coefficients)},
                     {"correction_active", correction.active},
                     {"slope_bits", std::bit_cast<std::uint32_t>(correction.xyz_slope)},
                     {"offset_bits", std::bit_cast<std::uint32_t>(correction.xyz_offset)},
                     {"recipe_hash", state.recipe_hash},
                     {"scanner_hash", state.scanner_hash}});
        return cold;
    }

    int run_reference_inputs(int argc, char** argv) {
        constexpr std::string_view caller = "ofx.gpu.scanner_reference_inputs";
        constexpr std::array cases{
            ScannerReferenceCase{{"negative-direct", "negative-direct", "NegativeDirectScan", 0, 3}, "kodak_portra_400", FJ_ROUTE_NEGATIVE_DIRECT},
            ScannerReferenceCase{{"negative-print", "negative-print", "NegativePrintScan", 0, 4}, "kodak_portra_400", FJ_ROUTE_NEGATIVE_PRINT},
            ScannerReferenceCase{{"positive-direct", "positive-direct", "NegativeDirectScan", 0, 3}, "fujifilm_provia_100f", FJ_ROUTE_POSITIVE_DIRECT},
            ScannerReferenceCase{{"positive-print", "positive-print", "NegativePrintScan", 0, 4}, "fujifilm_provia_100f", FJ_ROUTE_POSITIVE_PRINT},
            ScannerReferenceCase{{"fixed/negative-print", kFixedNegativePrint.name, kFixedNegativePrint.route, kFixedNegativePrint.reconstruction, kFixedNegativePrint.components},
                                 kFixedNegativePrint.profile,
                                 FJ_ROUTE_NEGATIVE_PRINT,
                                 &kFixedNegativePrint}};
        if (JuicerTests::list_requested(argc, argv)) {
            for (const auto& test : cases) {
                JuicerTests::list_case(caller, test.image.id);
            }
            return 0;
        }
        JuicerTests::CaseLog log;
        std::vector<std::string> ids;
        ids.reserve(cases.size());
        for (const auto& test : cases) {
            ids.emplace_back(test.image.id);
        }
        JuicerTests::Cohort cohort(std::string(caller), ids);
        std::puts("CTEST_FULL_OUTPUT");
        for (const auto& test : cases) {
            auto row = cohort.case_event(test.image.id);
            const auto input = capture_reference_inputs(test);
            std::puts(input.dump().c_str());
            row.passed();
        }
        log.finish();
        return 0;
    }

    void check_highlight_carrier(std::size_t index) {
        using namespace JuicerCuda::ExecutorTest;
        constexpr int width = 97;
        constexpr int height = 37;
        const bool print = index == 1 || index == 3;
        const bool cameraDiffusion = index >= 2;
        const bool spatial = index != 2;
        ofx_test::DeviceImages images({width, height, 4});
        ofx_test::RenderHost fixture;
        fixture.configure(images, {-13, 7});
        const auto set = [&](const char* name, std::vector<ofx_test::InstanceValue> values) {
            fixture.host().set_value(fixture.effect(), name, std::move(values));
        };
        set("FilmProfileKey", {std::string(index >= 2 ? "fujifilm_provia_100f" : "kodak_portra_400")});
        set("ScanRoute", {std::string(print ? "NegativePrintScan" : "NegativeDirectScan")});
        set("CameraAutoExposure", {0});
        set("CameraFilmFormatMm", {8.0});
        set("CouplersActive", {1});
        set("CouplersDiffusionSizeUm", {120.0});
        set("CameraDiffusionEnabled", {cameraDiffusion ? 1 : 0});
        set("HalationActive", {spatial ? 1 : 0});
        set("HalationScatterAmount", {index == 1 ? 0.0 : 1.0});
        set("HalationAmount", {index == 0 ? 0.0 : 1.0});
        set("HalationHighlightBoostEv", {2.0});
        set("HalationHighlightProtectionEv", {0.0});
        set("HalationHighlightBoostRange", {1.0});
        const auto capture = [&](double exposure) {
            set("Exposure", {exposure});
            change(fixture, "Exposure", kOfxChangeUserEdited, 37.0);
            CarrierReadbacks readbacks;
            set_carrier_observer(&readbacks);
            try {
                (void)render(fixture, images, 37.0);
            } catch (...) {
                set_carrier_observer(nullptr);
                throw;
            }
            set_carrier_observer(nullptr);
            return readbacks;
        };
        const auto initial = capture(0.0);
        const auto increased = capture(1.0);
        const auto stage = [](const CarrierReadbacks& captured, CarrierStage which) -> const CarrierReadback& {
            return captured.at(static_cast<std::size_t>(which));
        };
        const auto& raw = stage(increased, CarrierStage::Exposure);
        const auto& boost = stage(increased, CarrierStage::Boost);
        const auto& camera = stage(increased, CarrierStage::CameraDiffusion);
        const auto& halation = stage(increased, CarrierStage::Spatial);
        const auto& log = stage(increased, CarrierStage::DirLog);
        const auto& initialRaw = stage(initial, CarrierStage::Exposure);
        ofx_test::require_render(raw.count == 1 && initialRaw.count == 1 && boost.count == 1 &&
                                     camera.count == static_cast<unsigned>(cameraDiffusion) &&
                                     halation.count == static_cast<unsigned>(spatial),
                                 "each configured carrier stage executes once");
        double peak = 0.0;
        for (const auto& plane : raw.rgb) {
            peak = std::max(peak, static_cast<double>(*std::max_element(plane.begin(), plane.end())));
        }
        constexpr double onset = 0.184;
        ofx_test::require_render(peak > onset, "boost observation requires an active global maximum");
        const auto shape = [](double t) {
            return std::expm1(t) - t;
        };
        const auto denominator = shape((peak - onset) / peak);
        const auto& final = spatial ? halation : cameraDiffusion ? camera
                                                                 : boost;
        unsigned logPlanes = 0;
        for (std::size_t channel = 0; channel < raw.rgb.size(); ++channel) {
            logPlanes += !log.rgb[channel].empty() ? 1U : 0U;
            for (std::size_t i = 0; i < raw.rgb[channel].size(); ++i) {
                const float x = raw.rgb[channel][i];
                RenderAssertions::require_same_bits("manual gain precedes boost", x, 2.0f * initialRaw.rgb[channel][i]);
                const float y = boost.rgb[channel][i];
                if (x <= onset) {
                    RenderAssertions::require_same_bits("below-onset carrier", y, x);
                } else if (x == peak) {
                    RenderAssertions::require_same_bits("RGB global peak", y, 4.0f * x);
                } else {
                    const double expected = x + 3.0 * peak * shape((x - onset) / peak) / denominator;
                    ofx_test::require_render(std::abs(y - expected) <= 2e-6 * expected, "boost uses the same image-global peak in all channels");
                }
                if (!log.rgb[channel].empty()) {
                    const float expected = std::log10(final.rgb[channel][i] * log.routeCorrection + 1e-10f);
                    ofx_test::require_render(std::abs(log.rgb[channel][i] - expected) <= 2e-6f, "DIR cached log consumes the post-optical carrier");
                }
            }
        }
        ofx_test::require_render(log.count == 1 && logPlanes >= 2, "at least two first-consumer DIR log planes observed");
    }

    void report_control_differences(const FjPreparationParams& actual, const FjPreparationParams& expected) {
        const auto field = [](const char* name, double left, double right) {
            if (std::bit_cast<std::uint64_t>(left) != std::bit_cast<std::uint64_t>(right))
                std::fprintf(stderr, "control %s actual=%.17g expected=%.17g\n", name, left, right);
        };
        field("print_exposure", actual.print_exposure, expected.print_exposure);
        field("print_preflash_exposure", actual.print_preflash_exposure, expected.print_preflash_exposure);
        field("print_gamma_factor", actual.print_gamma_factor, expected.print_gamma_factor);
        field("glare_percent", actual.glare_percent, expected.glare_percent);
        field("glare_roughness", actual.glare_roughness, expected.glare_roughness);
        field("glare_blur_sigma_px", actual.glare_blur_sigma_px, expected.glare_blur_sigma_px);
        field("camera_exposure_compensation_ev", actual.camera_exposure_compensation_ev, expected.camera_exposure_compensation_ev);
        field("camera_film_format_long_edge_mm", actual.camera_film_format_long_edge_mm, expected.camera_film_format_long_edge_mm);
        field("film_gamma_factor", actual.film_gamma_factor, expected.film_gamma_factor);
        field("scanner_lens_blur_sigma_px", actual.scanner_lens_blur_sigma_px, expected.scanner_lens_blur_sigma_px);
        field("scanner_black_level", actual.scanner_black_level, expected.scanner_black_level);
        field("scanner_white_level", actual.scanner_white_level, expected.scanner_white_level);
        field("gate_weave_amount", actual.gate_weave_amount, expected.gate_weave_amount);
        field("spectral_upsampling_mode", actual.spectral_upsampling_mode, expected.spectral_upsampling_mode);
        field("ref_ill", actual.ref_ill, expected.ref_ill);
        field("enl_ill", actual.enl_ill, expected.enl_ill);
        field("normalize_print_exposure", actual.normalize_print_exposure, expected.normalize_print_exposure);
        field("print_exposure_compensation", actual.print_exposure_compensation, expected.print_exposure_compensation);
        field("input_color_space", actual.input_color_space, expected.input_color_space);
        field("input_cctf_decoding", actual.input_cctf_decoding, expected.input_cctf_decoding);
        field("input_compression_enabled", actual.input_compression_enabled, expected.input_compression_enabled);
        field("hanatos2025_adaptation_window", actual.hanatos2025_adaptation_window, expected.hanatos2025_adaptation_window);
        field("hanatos2025_adaptation_surface", actual.hanatos2025_adaptation_surface, expected.hanatos2025_adaptation_surface);
        field("camera_auto_exposure_enabled", actual.camera_auto_exposure_enabled, expected.camera_auto_exposure_enabled);
        field("camera_metering_method", actual.camera_metering_method, expected.camera_metering_method);
        field("scanner_black_correction", actual.scanner_black_correction, expected.scanner_black_correction);
        field("scanner_white_correction", actual.scanner_white_correction, expected.scanner_white_correction);
        field("scanner_lut_resolution", actual.scanner_lut_resolution, expected.scanner_lut_resolution);
        field("output_color_space", actual.output_color_space, expected.output_color_space);
        field("output_cctf_encoding", actual.output_cctf_encoding, expected.output_cctf_encoding);
        field("output_gamut_compression_enabled", actual.output_gamut_compression_enabled, expected.output_gamut_compression_enabled);
        field("scan_route", actual.scan_route, expected.scan_route);
        field("glare_active", actual.glare_active, expected.glare_active);
        field("camera_filter_override", actual.camera_filter_override, expected.camera_filter_override);
        field("film_dust_amount", actual.film_dust_amount, expected.film_dust_amount);
        field("film_scratch_amount", actual.film_scratch_amount, expected.film_scratch_amount);
        field("gate_dust_amount", actual.gate_dust_amount, expected.gate_dust_amount);
        field("gate_scratch_amount", actual.gate_scratch_amount, expected.gate_scratch_amount);
        for (std::size_t i = 0; i < 2; ++i)
            field("scanner_unsharp_mask", actual.scanner_unsharp_mask[i], expected.scanner_unsharp_mask[i]);
        for (std::size_t i = 0; i < 3; ++i) {
            field("camera_filter_uv", actual.camera_filter_uv[i], expected.camera_filter_uv[i]);
            field("camera_filter_ir", actual.camera_filter_ir[i], expected.camera_filter_ir[i]);
            field("print_ui_ymc_cc", actual.print_ui_ymc_cc[i], expected.print_ui_ymc_cc[i]);
            field("dir_gamma_same", actual.dir_couplers.gamma_same_layer_rgb[i], expected.dir_couplers.gamma_same_layer_rgb[i]);
            field("grain_scale", actual.grain.particle_scale_cmy[i], expected.grain.particle_scale_cmy[i]);
            field("grain_layers", actual.grain.particle_scale_layers[i], expected.grain.particle_scale_layers[i]);
            field("grain_density_min", actual.grain.particle_density_min_cmy[i], expected.grain.particle_density_min_cmy[i]);
            field("grain_uniformity", actual.grain.uniformity_cmy[i], expected.grain.uniformity_cmy[i]);
        }
        field("dir_active", actual.dir_couplers.active, expected.dir_couplers.active);
        field("dir_amount", actual.dir_couplers.amount, expected.dir_couplers.amount);
        field("dir_stock", actual.dir_couplers.gamma_use_stock, expected.dir_couplers.gamma_use_stock);
        field("grain_active", actual.grain.active, expected.grain.active);
        field("grain_sublayers_active", actual.grain.sublayers_active, expected.grain.sublayers_active);
        field("grain_sublayer_count", actual.grain.sublayer_count, expected.grain.sublayer_count);
        field("grain_blur_dye_clouds", actual.grain.dye_cloud_blur_um, expected.grain.dye_cloud_blur_um);
        field("grain_coarse_weight", actual.grain.coarse_weight, expected.grain.coarse_weight);
        field("grain_mid_weight", actual.grain.mid_weight, expected.grain.mid_weight);
        field("grain_size_mix_scale", actual.grain.size_mix_scale, expected.grain.size_mix_scale);
        field("grain_clump_temporal", actual.grain.clump_temporal_mix, expected.grain.clump_temporal_mix);
        field("grain_clump_period", actual.grain.clump_morph_period_sec, expected.grain.clump_morph_period_sec);
        field("grain_micro_cell", actual.grain.micro_structure[0], expected.grain.micro_structure[0]);
        field("grain_micro_sigma", actual.grain.micro_structure[1], expected.grain.micro_structure[1]);
        field("grain_amplitude", actual.grain.amplitude, expected.grain.amplitude);
        field("grain_correlation", actual.grain.correlation_sigma_px, expected.grain.correlation_sigma_px);
        field("grain_area", actual.grain.particle_area_um2, expected.grain.particle_area_um2);
        field("grain_chroma", actual.grain.chroma_mix, expected.grain.chroma_mix);
        field("camera_diffusion_active", actual.camera_diffusion.active, expected.camera_diffusion.active);
        field("camera_diffusion_family", actual.camera_diffusion.family, expected.camera_diffusion.family);
        field("halation_active", actual.scatter_halation.active, expected.scatter_halation.active);
    }
    void frozen_scenarios(const JuicerTests::Cohort& rows) {
        constexpr std::array scenarios{
            kFixedNegativePrint,
            FixedScenario{"positive-direct", "NegativeDirectScan", "fujifilm_provia_100f", 2, 3, {11, -3}, kInstance + 2, false, false, false},
            FixedScenario{"positive-print", "NegativePrintScan", "fujifilm_provia_100f", 0, 4, {0, 0}, kInstance + 3, false, false, false},
            FixedScenario{"combined-print", "NegativePrintScan", "kodak_portra_400", 0, 3, {0, 0}, kInstance + 5, true, false, false},
            FixedScenario{"glare-plus-zero", "NegativePrintScan", "kodak_portra_400", 0, 3, {0, 0}, kInstance + 1, false, true, false},
            FixedScenario{"glare-minus-zero", "NegativePrintScan", "kodak_portra_400", 0, 3, {0, 0}, kInstance + 1, false, true, true}};
        for (const auto& scenario : scenarios) {
            const bool naturalGlare = std::string_view(scenario.name) == kFixedNegativePrint.name;
            auto event = rows.case_event(std::string("fixed/") + scenario.name);
            ofx_test::DeviceImages images({7, 5, scenario.components});
            fj_test_ofx_render_tokens(kSession, scenario.instance, kClip);
            ofx_test::RenderHost fixture(OfxGetPlugin(0));
            configure_fixed_scan(fixture, images, scenario);
            if (scenario.combined) {
                fixture.host().set_value(fixture.effect(), "ScannerUnsharpMask", {0.7, 0.7});
                fixture.host().set_value(fixture.effect(), "CameraDiffusionEnabled", {1});
                fixture.host().set_value(fixture.effect(), "GrainActive", {1});
                fixture.host().set_value(fixture.effect(), "GrainSublayersActive", {1});
                fixture.host().set_value(fixture.effect(), "GrainAmplitude", {0.0});
                fixture.host().set_value(fixture.effect(), "GrainBlur", {0.65});
                fixture.host().set_value(fixture.effect(), "GrainChroma", {0.0});
                fixture.host().set_value(fixture.effect(), "GrainParticleAreaUm2", {0.2});
                fixture.host().set_value(fixture.effect(), "GrainParticleScale", {0.8, 1.0, 2.0});
                fixture.host().set_value(fixture.effect(), "GrainParticleScaleLayers", {2.5, 1.0, 0.5});
                fixture.host().set_value(fixture.effect(), "GrainDensityMin", {0.07, 0.08, 0.12});
                fixture.host().set_value(fixture.effect(), "GrainUniformity", {0.97, 0.97, 0.99});
                fixture.host().set_value(fixture.effect(), "GrainBlurDyeCloudsUm", {1.0});
                fixture.host().set_value(fixture.effect(), "GrainMicroStructure", {0.2, 30.0});
                fixture.host().set_value(fixture.effect(), "GrainSizeMixWeight", {0.0});
                fixture.host().set_value(fixture.effect(), "GrainSizeMixWeightMid", {0.0});
                fixture.host().set_value(fixture.effect(), "GrainSizeMixScale", {1.0});
                fixture.host().set_value(fixture.effect(), "GrainClumpTemporalMix", {0.0});
                fixture.host().set_value(fixture.effect(), "GrainClumpMorphPeriodSec", {8.0});
                fixture.host().set_value(fixture.effect(), "HalationActive", {1});
            }
            if (scenario.glareZero) {
                fixture.host().set_value(fixture.effect(), "GlarePercent", {0.3});
                fixture.host().set_value(fixture.effect(), "Exposure", {scenario.negativeZero ? -0.0 : 0.0});
            }
            fj_test_ofx_render_tokens(kSession, scenario.instance, kClip);
            std::optional<DensityCaptureScope> witness;
            if (naturalGlare) {
                witness.emplace(images.stream());
            }
            ofx_test::require_render(fixture.render() == kOfxStatOK, "frozen scenario render");
            const auto seen = render_observation(fixture.effect());
            ofx_test::require_render(seen.native_entry_count == 1 && seen.instance_token == scenario.instance &&
                                         seen.session_seed == kSession && seen.clip_token == kClip && seen.frame_token == 37,
                                     "frozen scenario frame identity");
            if (naturalGlare || scenario.combined) {
                JuicerTestSupport::PreparationInput snapshot;
                snapshot.film_key = scenario.profile;
                snapshot.values.scan_route = FJ_ROUTE_NEGATIVE_PRINT;
                snapshot.values.spectral_upsampling_mode = scenario.reconstruction;
                snapshot.values.camera_auto_exposure_enabled = 0;
                if (scenario.combined) {
                    snapshot.values.camera_diffusion.active = 1;
                    snapshot.values.grain.active = 1;
                    snapshot.values.scatter_halation.active = 1;
                    snapshot.values.scatter_halation.scatter_amount = 1.0f;
                    snapshot.values.scatter_halation.scatter_spatial_scale = 1.0f;
                    snapshot.values.scatter_halation.halation_amount = 1.0f;
                    snapshot.values.scatter_halation.halation_spatial_scale = 1.0f;
                } else {
                    snapshot.values.scanner_unsharp_mask[0] = 0.0;
                    snapshot.values.scanner_unsharp_mask[1] = 0.0;
                }
                std::optional<JuicerTestSupport::PreparedStateHold> baseline;
                std::array<char, 1024> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                const auto* assets = static_cast<const FjAssets*>(fj_test_ofx_assets());
                ofx_test::require_render(JuicerTestSupport::prepare_state(assets, snapshot, baseline, error).category == FJ_STATUS_SUCCESS && baseline,
                                         "typed print baseline preparation");
                FjPreparedStateView expected{};
                ofx_test::require_render(baseline->view(expected, error).category == FJ_STATUS_SUCCESS, "typed print baseline view");
                Observation observed(fixture.effect());
                const auto& actual = observed.view.published;
                if (actual.recipe_hash != expected.recipe_hash || actual.print_recipe_hash != expected.print_recipe_hash || actual.film_exposure.hash != expected.film_exposure.hash ||
                    actual.scanner_post_effects.hash != expected.scanner_post_effects.hash) {
                    std::fprintf(stderr,
                                 "%s identities actual/typed recipe=%llu/%llu film=%llu/%llu print=%llu/%llu post=%llu/%llu glare=%g/%g rough=%g/%g blur=%g/%g\n",
                                 scenario.name,
                                 static_cast<unsigned long long>(actual.recipe_hash),
                                 static_cast<unsigned long long>(expected.recipe_hash),
                                 static_cast<unsigned long long>(actual.film_exposure.hash),
                                 static_cast<unsigned long long>(expected.film_exposure.hash),
                                 static_cast<unsigned long long>(actual.print_recipe_hash),
                                 static_cast<unsigned long long>(expected.print_recipe_hash),
                                 static_cast<unsigned long long>(actual.scanner_post_effects.hash),
                                 static_cast<unsigned long long>(expected.scanner_post_effects.hash),
                                 static_cast<double>(actual.scanner_post_effects.glare_percent),
                                 static_cast<double>(expected.scanner_post_effects.glare_percent),
                                 static_cast<double>(actual.scanner_post_effects.glare_roughness),
                                 static_cast<double>(expected.scanner_post_effects.glare_roughness),
                                 static_cast<double>(actual.scanner_post_effects.glare_blur_sigma_px),
                                 static_cast<double>(expected.scanner_post_effects.glare_blur_sigma_px));
                    const FjPreparationParams expectedControls = snapshot.record();
                    report_control_differences(observed.view.captured, expectedControls);
                    throw std::runtime_error("actual print controls differ from independent typed reference inputs");
                }
                if (witness) {
                    const auto glare = glare_field(witness->read());
                    const auto predicted = natural_glare_pixels(observed.view, seen, glare);
                    RenderAssertions::compare_pixels(scenario.name, images.pixels(images.read()), predicted);
                    witness.reset();
                }
            }
            if (!naturalGlare) {
                RenderAssertions::compare_pixels(scenario.name, images.pixels(images.read()), ofx_test::reference_pixels(JUICER_PROCESSOR_RUST_PATH, scenario.name));
            }
            ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == 1 &&
                                         fixture.host().image_release_count(fixture.effect(), "Output") == 1,
                                     "frozen scenario image releases");
            fixture.close();
            event.passed();
        }
    }
    template <class F>
    void row(const JuicerTests::Cohort& rows, const char* id, F&& body) {
        auto event = rows.case_event(id);
        body();
        event.passed();
    }
    void snapshot_defaults(ofx_test::RenderHost& fixture) {
        auto& host = fixture.host();
        auto* const effect = fixture.effect();
        host.set_value(effect, "CameraAutoExposure", {1});
        host.set_value(effect, "ScannerUnsharpMask", {0.7, 0.7});
        host.set_value(effect, "GlareActive", {1});
        host.set_value(effect, "GrainActive", {0});
        host.set_value(effect, "GrainSublayersActive", {1});
        host.set_value(effect, "GrainAmplitude", {0.0});
        host.set_value(effect, "GrainBlur", {0.65});
        host.set_value(effect, "GrainChroma", {0.0});
        host.set_value(effect, "GrainParticleAreaUm2", {0.2});
        host.set_value(effect, "GrainParticleScale", {0.8, 1.0, 2.0});
        host.set_value(effect, "GrainParticleScaleLayers", {2.5, 1.0, 0.5});
        host.set_value(effect, "GrainDensityMin", {0.07, 0.08, 0.12});
        host.set_value(effect, "GrainUniformity", {0.97, 0.97, 0.99});
        host.set_value(effect, "GrainBlurDyeCloudsUm", {1.0});
        host.set_value(effect, "GrainMicroStructure", {0.2, 30.0});
        host.set_value(effect, "GrainSizeMixWeight", {0.0});
        host.set_value(effect, "GrainSizeMixWeightMid", {0.0});
        host.set_value(effect, "GrainSizeMixScale", {1.0});
        host.set_value(effect, "GrainClumpTemporalMix", {0.0});
        host.set_value(effect, "GrainClumpMorphPeriodSec", {8.0});
    }
    void accepted_identities(const JuicerTests::Cohort& rows) {
        std::ifstream stream(JUICER_PARAMETER_IDENTITIES_PATH);
        ofx_test::require_render(static_cast<bool>(stream), "identity fixture unavailable");
        const auto expected = PrintIlluminantFixtures::apply(JUICER_PARAMETER_IDENTITIES_PATH, nlohmann::json::parse(stream));
        for (const auto* name : kIdentityPairs) {
            const std::string_view pair(name);
            const bool print = pair.starts_with("print-");
            const bool medium = pair == "print-medium-exposure-small";
            const bool signed_zero = pair.ends_with("-zero");
            ofx_test::DeviceImages images({7, 5, 3});
            tokens();
            ofx_test::RenderHost fixture(OfxGetPlugin(0));
            fixture.configure(images);
            snapshot_defaults(fixture);
            fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string(print ? "NegativePrintScan" : "NegativeDirectScan")});
            std::uint64_t previous_build = 0;
            std::uint64_t previous_recipe = 0;
            for (const char* suffix : {"A", "B"}) {
                const std::string fixture_key = std::string(name) + "/" + suffix;
                auto event = rows.case_event(std::string("identity/") + fixture_key);
                const bool second = std::string_view(suffix) == "B";
                const double value = signed_zero ? (second ? -0.0 : 0.0) : (second ? 1.00001 : 1.0);
                const char* control = medium ? "PrintExposure" : "Exposure";
                fixture.host().set_value(fixture.effect(), control, {value});
                if (second)
                    change(fixture, control, kOfxChangeTime, 37.0);
                (void)render(fixture, images, 37.0);
                Observation actual(fixture.effect());
                const auto& state = actual.view.published;
                const auto& accepted = expected.at(fixture_key);
                const std::uint64_t seed = print ? Hash::hash_uint64_values({state.print_recipe_hash, 37, 5, 9}) : 0;
                const std::array actual_identity{state.recipe_hash, state.film_exposure.hash, state.print_recipe_hash, seed};
                const std::array expected_identity{accepted.at("recipe").get<std::uint64_t>(), accepted.at("filmRaw").get<std::uint64_t>(), accepted.at("printSeedInput").get<std::uint64_t>(), accepted.at("fixedGlareSeed").get<std::uint64_t>()};
                const bool matches = RenderAssertions::identities_match(actual_identity, expected_identity) &&
                                     (!second || (actual.view.build > previous_build && state.recipe_hash != previous_recipe)) &&
                                     (!signed_zero || std::bit_cast<std::uint32_t>(state.film_exposure.manual_exposure_ev) == (second ? 0x80000000U : 0U));
                if (!matches) {
                    std::fprintf(stderr, "%s actual=(%llu,%llu,%llu,%llu) expected=(%llu,%llu,%llu,%llu) build=%llu previous=%llu\n", fixture_key.c_str(), static_cast<unsigned long long>(actual_identity[0]), static_cast<unsigned long long>(actual_identity[1]), static_cast<unsigned long long>(actual_identity[2]), static_cast<unsigned long long>(actual_identity[3]), static_cast<unsigned long long>(expected_identity[0]), static_cast<unsigned long long>(expected_identity[1]), static_cast<unsigned long long>(expected_identity[2]), static_cast<unsigned long long>(expected_identity[3]), static_cast<unsigned long long>(actual.view.build), static_cast<unsigned long long>(previous_build));
                    JuicerTestSupport::PreparationInput baseline;
                    baseline.values.scan_route = print ? FJ_ROUTE_NEGATIVE_PRINT : FJ_ROUTE_NEGATIVE_DIRECT;
                    if (medium)
                        baseline.values.print_exposure = value;
                    else
                        baseline.values.camera_exposure_compensation_ev = value;
                    const FjPreparationParams expectedControls = baseline.record();
                    report_control_differences(actual.view.captured, expectedControls);
                    throw std::runtime_error("accepted actual-callback recipe identities");
                }
                previous_build = actual.view.build;
                previous_recipe = state.recipe_hash;
                event.passed();
            }
            fixture.close();
        }
    }
    void run(const JuicerTests::Cohort& rows) {
        row(rows, "identity/fixed", [&] {
            frozen_scenarios(rows);
        });
        accepted_identities(rows);
        ofx_test::DeviceImages images({7, 5, 3});
        tokens();
        ofx_test::RenderHost fixture(OfxGetPlugin(0));
        fixture.configure(images);
        std::vector<float> original;
        std::vector<float> time_a_pixels;
        std::vector<float> time_b_pixels;
        std::uint64_t initial_build = 0;
        row(rows, "temporal/initial", [&] {
            original = render(fixture, images, 37.0);
            RenderAssertions::compare_pixels("initial frozen pixels", original, ofx_test::reference_pixels(JUICER_PROCESSOR_RUST_PATH, "negative-direct"));
            Observation state(fixture.effect());
            ofx_test::require_render(state.view.pending_kind == 4 && state.view.build > 0 && state.view.backing_identity != 0, "initial publication");
            initial_build = state.view.build;
        });
        row(rows, "temporal/time_a", [&] {
            fixture.host().set_value(fixture.effect(), "Exposure", {0.5});
            fixture.host().clear_calls();
            change(fixture, "Exposure", kOfxChangeTime, 41.0);
            const auto& calls = fixture.host().calls();
            ofx_test::require_render(std::any_of(calls.begin(), calls.end(), [](const auto& call) {
                                         return call.operation == "param_get" && call.target == "Exposure";
                                     }) &&
                                         std::none_of(calls.begin(), calls.end(), [](const auto& call) {
                                             return call.operation == "param_get_time";
                                         }),
                                     "current-value capture getters");
            time_a_pixels = render(fixture, images, 41.0);
            Observation state(fixture.effect());
            ofx_test::require_render(state.view.captured.camera_exposure_compensation_ev == 0.5 && state.view.build > initial_build,
                                     "time A captured publication");
        });
        row(rows, "temporal/time_b", [&] {
            fixture.host().set_value(fixture.effect(), "Exposure", {1.0});
            change(fixture, "Exposure", kOfxChangeTime, 49.0);
            time_b_pixels = render(fixture, images, 49.0);
            ofx_test::require_render(time_a_pixels != time_b_pixels, "different exposures rendered identical pixels");
            Observation state(fixture.effect());
            ofx_test::require_render(state.view.captured.camera_exposure_compensation_ev == 1.0, "time B capture");
        });
        row(rows, "temporal/backward", [&] {
            fixture.host().set_value(fixture.effect(), "Exposure", {0.5});
            change(fixture, "Exposure", kOfxChangeTime, 41.0);
            RenderAssertions::compare_bits("backward fixed token pixels", render(fixture, images, 41.0), time_a_pixels);
            Observation state(fixture.effect());
            ofx_test::require_render(state.view.captured.camera_exposure_compensation_ev == 0.5, "backward capture");
        });
        row(rows, "temporal/repeat", [&] {
            Observation before(fixture.effect());
            const auto pixels = render(fixture, images, 41.0);
            Observation after(fixture.effect());
            ofx_test::require_render(before.view.build == after.view.build && before.view.backing_identity == after.view.backing_identity,
                                     "repeat must reuse immutable publication");
            RenderAssertions::compare_bits("repeat pixels", pixels, render(fixture, images, 41.0));
        });
        row(rows, "temporal/preset", [&] {
            fixture.host().set_value(fixture.effect(), "GrainPreset", {1});
            change(fixture, "GrainPreset", kOfxChangeUserEdited, 41.0);
            const auto pixels = render(fixture, images, 41.0);
            RenderAssertions::compare_bits("preset fixed token pixels", pixels, render(fixture, images, 41.0));
            Observation state(fixture.effect());
            ofx_test::require_render(state.view.pending_kind == 4 &&
                                         state.view.captured.grain.amplitude == static_cast<float>(std::exp2(-1.20)) &&
                                         state.view.captured.grain.chroma_mix == 0.3f &&
                                         state.view.captured.grain.particle_area_um2 == 0.25f,
                                     "preset captured grain values");
        });
        row(rows, "temporal/reset", [&] {
            fixture.host().set_value(fixture.effect(), "GrainBlurDyeCloudsUm", {9.0});
            fixture.host().set_value(fixture.effect(), "GrainSizeMixWeight", {0.0});
            fixture.host().set_value(fixture.effect(), "GrainMicroStructure", {1.0, 2.0});
            change(fixture, "GrainResetAdvanced", kOfxChangeUserEdited, 41.0);
            const auto pixels = render(fixture, images, 41.0);
            RenderAssertions::compare_bits("reset fixed token pixels", pixels, render(fixture, images, 41.0));
            Observation state(fixture.effect());
            const double expected_weight = 0.072 + (0.38 - 0.072) * 0.55;
            const auto micro = fixture.host().value(fixture.effect(), "GrainMicroStructure");
            ofx_test::require_render(state.view.pending_kind == 4 &&
                                         std::get<double>(fixture.host().value(fixture.effect(), "GrainBlurDyeCloudsUm").at(0)) == 1.0 &&
                                         std::get<double>(fixture.host().value(fixture.effect(), "GrainSizeMixWeight").at(0)) == expected_weight &&
                                         std::get<double>(micro.at(0)) == 61.0 && std::get<double>(micro.at(1)) == 173.0 &&
                                         state.view.captured.grain.dye_cloud_blur_um == 1.0f &&
                                         state.view.captured.grain.coarse_weight == static_cast<float>(expected_weight),
                                     "reset advanced values and capture");
        });
        row(rows, "temporal/undo", [&] {
            fixture.host().set_value(fixture.effect(), "Exposure", {0.0});
            change(fixture, "Exposure", kOfxChangeUserEdited, 41.0);
            (void)render(fixture, images, 41.0);
            Observation state(fixture.effect());
            ofx_test::require_render(state.view.captured.camera_exposure_compensation_ev == 0.0, "undo capture");
        });
        row(rows, "temporal/redo", [&] {
            fixture.host().set_value(fixture.effect(), "Exposure", {0.5});
            change(fixture, "Exposure", kOfxChangeUserEdited, 41.0);
            (void)render(fixture, images, 41.0);
            Observation state(fixture.effect());
            ofx_test::require_render(state.view.captured.camera_exposure_compensation_ev == 0.5, "redo capture");
        });
        row(rows, "temporal/invalid", [&] {
            fixture.host().set_value(fixture.effect(), "CameraFilmFormatMm", {121.0});
            change(fixture, "CameraFilmFormatMm", kOfxChangeUserEdited, 41.0);
            tokens();
            ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "invalid controls admitted render");
        });
        row(rows, "temporal/recovery", [&] {
            fixture.host().set_value(fixture.effect(), "CameraFilmFormatMm", {35.0});
            change(fixture, "CameraFilmFormatMm", kOfxChangeUserEdited, 41.0);
            (void)render(fixture, images, 41.0);
            Observation state(fixture.effect());
            ofx_test::require_render(state.view.pending_kind == 4, "recovery publication");
        });
        row(rows, "geometry/changed_size", [&] {
            ofx_test::DeviceImages changed({9, 6, 3});
            fixture.configure(changed);
            (void)render(fixture, changed, 41.0);
            const auto seen = render_observation(fixture.effect());
            ofx_test::require_render(seen.full_bounds[2] == 9 && seen.full_bounds[3] == 6 &&
                                         std::abs(seen.pixel_size_um - 35000.0f / 9.0f) < 0.01f,
                                     "changed size frame facts");
        });
        row(rows, "geometry/half_scale", [&] {
            fixture.configure(images);
            const auto full_scale = render(fixture, images, 41.0);
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderScale, {0.5, 0.5});
            for (const char* clip : {"Source", "Output"})
                fixture.host().set_image_property(fixture.effect(), clip, kOfxImageEffectPropRenderScale, {0.5, 0.5});
            const auto half_scale = render(fixture, images, 41.0);
            RenderAssertions::compare_bits("scale applied only once", half_scale, full_scale);
            const auto seen = render_observation(fixture.effect());
            ofx_test::require_render(seen.full_bounds[2] == 7 && seen.full_bounds[3] == 5 && seen.pixel_size_um == 5000.0f,
                                     "half-scale frame facts");
        });
        row(rows, "geometry/origin", [&] {
            fixture.configure(images, {11, -3});
            (void)render(fixture, images, 41.0);
            const auto seen = render_observation(fixture.effect());
            ofx_test::require_render(seen.full_bounds[0] == 11 && seen.full_bounds[1] == -3 &&
                                         seen.full_bounds[2] == 18 && seen.full_bounds[3] == 2 &&
                                         std::equal(std::begin(seen.full_bounds), std::end(seen.full_bounds), std::begin(seen.render_bounds)),
                                     "origin frame facts");
        });
        row(rows, "effects/full_frame", [&] {
            fixture.configure(images);
            fixture.host().set_value(fixture.effect(), "FilmDustAmount", {0.5});
            change(fixture, "FilmDustAmount", kOfxChangeUserEdited, 41.0);
            fixture.host().set_value(fixture.effect(), "GateWeaveAmount", {0.5});
            change(fixture, "GateWeaveAmount", kOfxChangeUserEdited, 41.0);
            const auto source_before = fixture.host().image_release_count(fixture.effect(), "Source");
            const auto output_before = fixture.host().image_release_count(fixture.effect(), "Output");
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {0, 0, 6, 5});
            images.reset();
            tokens();
            ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "weave accepted partial frame");
            for (float sample : images.read())
                RenderAssertions::require_same_bits("weave partial frame destination", sample, ofx_test::DeviceImages::kCanary);
            ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == source_before + 1 &&
                                         fixture.host().image_release_count(fixture.effect(), "Output") == output_before + 1,
                                     "weave partial frame image releases");
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {0, 0, 7, 5});
            fixture.host().clear_calls();
            JuicerCuda::ExecutorTest::reset_effects_frame_observation();
            const auto pixels = render(fixture, images, 41.0);
            const auto effects = JuicerCuda::ExecutorTest::effects_frame_observation();
            const auto seen = render_observation(fixture.effect());
            const auto& calls = fixture.host().calls();
            ofx_test::require_render(std::equal(std::begin(seen.full_bounds), std::end(seen.full_bounds), std::begin(seen.render_bounds)) &&
                                         std::any_of(calls.begin(), calls.end(), [](const auto& call) {
                                             return call.operation == "clip_get_region" && call.target == "Source" &&
                                                    !call.values.empty() && std::get<double>(call.values[0]) == 41.0;
                                         }),
                                     "physical effects full frame and live canonical definition");
            ofx_test::require_render(effects.count == 1 && effects.success && effects.descriptor && effects.descriptor->filmActive &&
                                         effects.descriptor->weaveActive && effects.descriptor->requiresFullFrame &&
                                         effects.descriptor->hash != 0 && effects.descriptor->recipeHash != 0 &&
                                         effects.geometry.pixelDefinition.x == 0 && effects.geometry.pixelDefinition.y == 0 &&
                                         effects.geometry.pixelDefinition.width == 7 && effects.geometry.pixelDefinition.height == 5 &&
                                         effects.geometry.canonicalWidth == 7.0 && effects.geometry.canonicalHeight == 5.0 &&
                                         effects.geometry.scaleX == 1.0 && effects.geometry.scaleY == 1.0 && effects.geometry.pixelAspectRatio == 1.0 &&
                                         effects.sessionSeed == kSession && effects.clipToken == kClip && effects.pixelSizeUm == 5000.0f,
                                     "physical effects descriptor and geometry");
            const auto hash = effects.descriptor->hash;
            JuicerCuda::ExecutorTest::reset_effects_frame_observation();
            RenderAssertions::compare_bits("physical effects repeat", pixels, render(fixture, images, 41.0));
            const auto repeated = JuicerCuda::ExecutorTest::effects_frame_observation();
            ofx_test::require_render(repeated.descriptor && repeated.descriptor->hash == hash, "physical effects frame hash repeat");
            fixture.host().set_value(fixture.effect(), "FilmDustAmount", {0.0});
            change(fixture, "FilmDustAmount", kOfxChangeUserEdited, 41.0);
            fixture.host().set_value(fixture.effect(), "GateWeaveAmount", {0.0});
            change(fixture, "GateWeaveAmount", kOfxChangeUserEdited, 41.0);
        });
        row(rows, "failure/message_order", [&] {
            using namespace JuicerCuda::ExecutorTest;
            Observation held(fixture.effect());
            const auto held_backing = held.view.backing_identity;
            const auto held_recipe = held.view.published.recipe_hash;
            FailureObservation failure;
            failure.status = JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed);
            failure.diagnostic = "trace deferred failure 100%";
            failure.stage = "direct_scan_error_stage";
            failure.scanResult = JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::DeferredDirFailure;
            bool observed_at_message = false;
            fixture.host().clear_calls();
            fixture.host().reenter_next("message", [&] {
                observed_at_message = failure.count == 3 && failure.events[0] == Event::Injected &&
                                      failure.events[1] == Event::Classified && failure.events[2] == Event::FrameAborted;
            });
            observation = &failure;
            tokens();
            const auto status = fixture.render();
            observation = nullptr;
            ofx_test::require_render(status == kOfxStatErrFatal && observed_at_message && failure.nativeRecoveryPending &&
                                         failure.adapterRecoveryPending && failure.count == 5 &&
                                         failure.events[3] == Event::RecoveryStarted && failure.events[4] == Event::RecoveryEnded,
                                     "deferred failure message and recovery order");
            const auto& calls = fixture.host().calls();
            const auto message = std::find_if(calls.begin(), calls.end(), [](const auto& call) {
                return call.operation == "message";
            });
            const auto release = std::find_if(calls.begin(), calls.end(), [](const auto& call) {
                return call.operation == "clip_release_image";
            });
            ofx_test::require_render(message != calls.end() && release != calls.end() && message < release &&
                                         message->property == kOfxMessageError && message->values.size() == 2 &&
                                         std::get<std::string>(message->values[0]) == "FilmJuicerDeferredCudaFailure" &&
                                         std::get<std::string>(message->values[1]) == failure.diagnostic &&
                                         release->target.starts_with("Output.image.") &&
                                         std::next(release) != calls.end() && std::next(release)->operation == "clip_release_image" &&
                                         std::next(release)->target.starts_with("Source.image."),
                                     "exact deferred message and image release order");
            ofx_test::require_render(held.view.backing_identity == held_backing && held.view.published.recipe_hash == held_recipe,
                                     "held publication survives native failure");
            (void)render(fixture, images, 41.0);
        });
        row(rows, "failure/holds", [&] {
            Observation held(fixture.effect());
            const auto previous = held.view.captured.camera_exposure_compensation_ev;
            const auto backing = held.view.backing_identity;
            fixture.host().set_value(fixture.effect(), "Exposure", {0.25});
            change(fixture, "Exposure", kOfxChangeUserEdited, 41.0);
            (void)render(fixture, images, 41.0);
            Observation updated(fixture.effect());
            ofx_test::require_render(held.view.captured.camera_exposure_compensation_ev == previous && held.view.backing_identity == backing &&
                                         updated.view.captured.camera_exposure_compensation_ev == 0.25 && updated.view.backing_identity != backing,
                                     "admitted backing remains held through subsequent publication");
        });
        struct TimeCase {
            const char* id;
            std::uint64_t input, normalized, token;
        };
        constexpr std::array time_cases{
            TimeCase{"time/positive_zero", 0x0000000000000000, 0x0000000000000000, 0},
            TimeCase{"time/negative_zero", 0x8000000000000000, 0x8000000000000000, 0},
            TimeCase{"time/negative_half", 0xbfe0000000000000, 0xbfe0000000000000, UINT64_MAX},
            TimeCase{"time/below_negative_one", 0xbff0000000000001, 0xbff0000000000001, UINT64_MAX - 1},
            TimeCase{"time/below_positive_one", 0x3fefffffffffffff, 0x3fefffffffffffff, 0},
            TimeCase{"time/nan", 0x7ff8000000000000, 0, 0},
            TimeCase{"time/positive_infinity", 0x7ff0000000000000, 0, 0},
            TimeCase{"time/negative_infinity", 0xfff0000000000000, 0, 0}};
        for (const auto& time : time_cases) {
            row(rows, time.id, [&] {
                const auto source_releases = fixture.host().image_release_count(fixture.effect(), "Source");
                const auto output_releases = fixture.host().image_release_count(fixture.effect(), "Output");
                fixture.host().set_property(fixture.host().action_args(), kOfxPropTime, {std::bit_cast<double>(time.input)});
                images.reset();
                tokens();
                ofx_test::require_render(fixture.render() == kOfxStatOK, "literal time render");
                const auto seen = render_observation(fixture.effect());
                ofx_test::require_render(seen.native_entry_count == 1 && std::bit_cast<std::uint64_t>(seen.frame_time) == time.normalized &&
                                             seen.frame_token == time.token && seen.session_seed == kSession && seen.clip_token == kClip &&
                                             fixture.host().image_release_count(fixture.effect(), "Source") == source_releases + 1 &&
                                             fixture.host().image_release_count(fixture.effect(), "Output") == output_releases + 1,
                                         "literal time facts and image releases");
                (void)images.pixels(images.read());
            });
        }
        row(rows, "time/valid_after_rejection", [&] {
            (void)render(fixture, images, 41.0);
            const auto before = render_observation(fixture.effect());
            const auto source_releases = fixture.host().image_release_count(fixture.effect(), "Source");
            const auto output_releases = fixture.host().image_release_count(fixture.effect(), "Output");
            fixture.host().set_property(fixture.host().action_args(), kOfxPropTime, {std::bit_cast<double>(std::uint64_t{0x43e0000000000000})});
            fixture.host().clear_calls();
            tokens();
            ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "out-of-domain time accepted");
            const auto rejected = render_observation(fixture.effect());
            ofx_test::require_render(rejected.native_entry_count == 0 &&
                                         fixture.host().image_release_count(fixture.effect(), "Source") == source_releases + 1 &&
                                         fixture.host().image_release_count(fixture.effect(), "Output") == output_releases + 1,
                                     "time rejection entered native or retained images");
            const auto& rejected_calls = fixture.host().calls();
            const auto release = std::find_if(rejected_calls.begin(), rejected_calls.end(), [](const auto& call) {
                return call.operation == "clip_release_image";
            });
            ofx_test::require_render(release != rejected_calls.end() && release->target.starts_with("Output.image.") &&
                                         std::next(release) != rejected_calls.end() && std::next(release)->operation == "clip_release_image" &&
                                         std::next(release)->target.starts_with("Source.image."),
                                     "time rejection image release order");
            (void)render(fixture, images, 41.0);
            const auto recovered = render_observation(fixture.effect());
            ofx_test::require_render(recovered.submission_id == before.submission_id && recovered.backing_identity == before.backing_identity,
                                     "time rejection changed submission or state");
            (void)render(fixture, images, 42.0);
            const auto next = render_observation(fixture.effect());
            ofx_test::require_render(next.submission_id == before.submission_id + 1, "time rejection consumed submission counter");
        });
        row(rows, "failure/recovery", [&] {
            (void)render(fixture, images, 41.0);
            ofx_test::require_render(render_observation(fixture.effect()).submission_id != 0, "submission after context recovery");
            fixture.close();
            auto abort_case = rows.case_event("abort/terminal_recovery");
            {
                tokens();
                ofx_test::RenderHost terminal(OfxGetPlugin(0));
                terminal.configure(images);
                terminal.host().clear_calls();
                fj_test_ofx_render_control(5);
                tokens();
                ofx_test::require_render(terminal.render() == kOfxStatErrFatal, "abort panic returned success");
                ofx_test::require_render(terminal.host().image_release_count(terminal.effect(), "Source") == 1 &&
                                             terminal.host().image_release_count(terminal.effect(), "Output") == 1,
                                         "abort panic image releases");
                const auto& calls = terminal.host().calls();
                const auto release = std::find_if(calls.begin(), calls.end(), [](const auto& call) {
                    return call.operation == "clip_release_image";
                });
                ofx_test::require_render(release != calls.end() && release->target.starts_with("Output.image.") &&
                                             std::next(release) != calls.end() && std::next(release)->target.starts_with("Source.image."),
                                         "abort panic release order");
                terminal.close();
            }
            {
                tokens();
                ofx_test::RenderHost recovered(OfxGetPlugin(0));
                recovered.configure(images);
                RenderAssertions::compare_pixels("new module after abort panic", render(recovered, images, 37.0), original);
                recovered.close();
            }
            abort_case.passed();
        });
    }
} // namespace

int main(int argc, char** argv) try {
    if (argc >= 2 && std::string_view(argv[1]) == "--scanner-reference-inputs") {
        return run_reference_inputs(argc, argv);
    }
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto* id : kRows)
            JuicerTests::list_case(kCaller, id);
        for (const auto* id : kSupplemental)
            JuicerTests::list_case(kCaller, id);
        for (const auto* id : kCarrierRows)
            JuicerTests::list_case(kCaller, id);
        for (const auto* pair : kIdentityPairs)
            for (const char* suffix : {"A", "B"})
                JuicerTests::list_case(kCaller, std::string("identity/") + pair + "/" + suffix);
        return 0;
    }
    JuicerTests::CaseLog log;
    std::vector<std::string> ids(kRows.begin(), kRows.end());
    ids.insert(ids.end(), kSupplemental.begin(), kSupplemental.end());
    ids.insert(ids.end(), kCarrierRows.begin(), kCarrierRows.end());
    ids.reserve(ids.size() + kIdentityPairs.size() * 2);
    for (const auto* pair : kIdentityPairs)
        for (const char* suffix : {"A", "B"})
            ids.emplace_back(std::string("identity/") + pair + "/" + suffix);
    JuicerTests::Cohort rows(std::string(kCaller), ids);
    run(rows);
    for (std::size_t index = 0; index < kCarrierRows.size(); ++index) {
        auto row = rows.case_event(kCarrierRows[index]);
        check_highlight_carrier(index);
        row.passed();
    }
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "GPU render trace fixture: %s\n", error.what());
    return 1;
}
