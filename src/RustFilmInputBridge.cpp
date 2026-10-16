#include "RustAssetBridge.h"

#include <algorithm>
#include <array>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "Logging.h"
#include "RenderRecipe.h"

namespace JuicerAssets {
#if defined(JUICER_FILM_INPUT_TEST_HOOK)
    namespace InputTest {
        void before_prepare(const FilmSource& source, const Spektrafilm::FilmFoundationBuildInput& input);
        void after_prepare(FjStatus& status, FjFilmInputFailure& failure);
    } // namespace InputTest
#endif
    namespace {
        [[noreturn]] void throw_failure(FjStatus status, std::string_view message) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, message);
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }

        bool valid_failure(FjStatus status, const FjFilmInputFailure& failure, const FjFilmInputParams& params) {
            if (status.category > FJ_STATUS_INTERNAL_FAILURE || status.api != FJ_API_NONE || status.native_code != 0) {
                return false;
            }
            const auto& sensitivity = failure.sensitivity;
            const bool metadata = sensitivity.hash_failure != 0 || sensitivity.hash_sample_index != 0;
            if ((sensitivity.hash_failure == 0 && sensitivity.hash_sample_index != 0) ||
                (sensitivity.hash_failure == 1 && sensitivity.hash_sample_index >= 243) ||
                (sensitivity.hash_failure == 2 && sensitivity.hash_sample_index != 0) || sensitivity.hash_failure > 2) {
                return false;
            }
            const bool tc = params.method != FJ_RAW_MALLETT_2019;
            switch (failure.origin) {
                case FJ_FILM_INPUT_ORIGIN_NONE:
                    return status.category == FJ_STATUS_SUCCESS && failure.reason == FJ_FILM_INPUT_REASON_NONE && !metadata;
                case FJ_FILM_INPUT_ORIGIN_RAW:
                    return status.category == FJ_STATUS_UNSUPPORTED_INPUT && failure.reason >= FJ_FILM_INPUT_RAW_RECORDS && failure.reason <= FJ_FILM_INPUT_RAW_SPANS && !metadata;
                case FJ_FILM_INPUT_ORIGIN_RESOURCE:
                    if (status.category != FJ_STATUS_PREPARATION_FAILURE || metadata) {
                        return false;
                    }
                    switch (failure.reason) {
                        case FJ_FILM_INPUT_RESOURCE_HANATOS_WINDOW:
                        case FJ_FILM_INPUT_RESOURCE_HANATOS_SURFACE:
                        case FJ_FILM_INPUT_RESOURCE_HANATOS_BLUR:
                            return params.method == FJ_RAW_HANATOS_2025;
                        case FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE:
                        case FJ_FILM_INPUT_RESOURCE_TC_SOURCE:
                            return tc;
                        case FJ_FILM_INPUT_RESOURCE_INPUT_HULL:
                            return tc && params.input_compression != 0;
                        default:
                            return false;
                    }
                case FJ_FILM_INPUT_ORIGIN_COLOR:
                    return status.category != FJ_STATUS_SUCCESS && !metadata &&
                           (failure.reason == FJ_FILM_INPUT_COLOR_MATRICES ||
                            (failure.reason == FJ_FILM_INPUT_COLOR_CAT16 && tc) ||
                            (failure.reason == FJ_FILM_INPUT_COLOR_CAT02 && !tc));
                case FJ_FILM_INPUT_ORIGIN_SENSITIVITY:
                    return (failure.reason == FJ_FILM_INPUT_SENSITIVITY_COMPUTED && status.category == FJ_STATUS_PREPARATION_FAILURE) ||
                           (failure.reason == FJ_FILM_INPUT_SENSITIVITY_BOUNDARY && status.category != FJ_STATUS_SUCCESS);
                case FJ_FILM_INPUT_ORIGIN_IDENTITY:
                    return status.category == FJ_STATUS_PREPARATION_FAILURE && failure.reason == FJ_FILM_INPUT_IDENTITY_REQUIRED && !metadata;
                case FJ_FILM_INPUT_ORIGIN_INTERNAL:
                    if (status.category != FJ_STATUS_INTERNAL_FAILURE) {
                        return false;
                    }
                    return failure.reason == FJ_FILM_INPUT_INTERNAL_PANIC || failure.reason == FJ_FILM_INPUT_INTERNAL_INTEGRITY ||
                           failure.reason == FJ_FILM_INPUT_INTERNAL_SENSITIVITY_PANIC ||
                           (!metadata && (failure.reason == FJ_FILM_INPUT_INTERNAL_MATRICES_PANIC ||
                                          (failure.reason == FJ_FILM_INPUT_INTERNAL_CAT16_PANIC && tc) ||
                                          (failure.reason == FJ_FILM_INPUT_INTERNAL_CAT02_PANIC && !tc)));
                default:
                    return false;
            }
        }

        bool translate(FjStatus status, const FjFilmInputFailure& failure, const FjFilmInputParams& params, std::string_view foreign, std::string& diagnostic) {
            if (!valid_failure(status, failure, params)) {
                throw_failure({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "Film input preparation: invalid failure contract");
            }
            if (failure.sensitivity.hash_failure == 1) {
                std::ostringstream message;
                message << "FATAL: hash_float_span encountered non-finite sample at index " << failure.sensitivity.hash_sample_index;
                JTRACE("HASH", message.str());
            }
            if (status.category == FJ_STATUS_SUCCESS) {
                return true;
            }
            if (failure.origin == FJ_FILM_INPUT_ORIGIN_RESOURCE) {
                switch (failure.reason) {
                    case FJ_FILM_INPUT_RESOURCE_HANATOS_WINDOW:
                        diagnostic = "MalformedRequiredProfileData phase=3D-2 field=data.hanatos2025_adaptation_window_params";
                        break;
                    case FJ_FILM_INPUT_RESOURCE_HANATOS_SURFACE:
                        diagnostic = "MalformedRequiredProfileData phase=3D-2 field=data.hanatos2025_adaptation_surface_params";
                        break;
                    case FJ_FILM_INPUT_RESOURCE_HANATOS_BLUR:
                        diagnostic = "MalformedRequiredProfileData phase=3D-2 field=settings.spectral_gaussian_blur";
                        break;
                    case FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE:
                        diagnostic = "MissingRequiredResource component=film_tc_lut requirement=integrated_projection_white";
                        break;
                    case FJ_FILM_INPUT_RESOURCE_TC_SOURCE:
                        diagnostic = params.method == FJ_RAW_ARCTIC_2026_BETA04
                                         ? "MissingRequiredResource resource=arctic2026beta04_reflectance_xy_tc.npy requirement=finite_float16_c_order_192x192x81"
                                         : "MissingRequiredResource resource=hanatos_2025_spectra_lut.npy requirement=finite_192x192x81";
                        break;
                    case FJ_FILM_INPUT_RESOURCE_INPUT_HULL:
                        diagnostic = "MissingRequiredResource component=film_tc_lut requirement=input_compression_hull";
                        break;
                }
                return false;
            }
            if (failure.origin == FJ_FILM_INPUT_ORIGIN_IDENTITY) {
                diagnostic = "MalformedRequiredProfileData phase=3B field=data.log_sensitivity";
                return false;
            }
            if (failure.origin == FJ_FILM_INPUT_ORIGIN_COLOR ||
                (failure.origin == FJ_FILM_INPUT_ORIGIN_INTERNAL && failure.reason >= FJ_FILM_INPUT_INTERNAL_MATRICES_PANIC && failure.reason <= FJ_FILM_INPUT_INTERNAL_CAT02_PANIC)) {
                const auto reason = failure.origin == FJ_FILM_INPUT_ORIGIN_COLOR
                                        ? failure.reason
                                        : failure.reason - FJ_FILM_INPUT_INTERNAL_MATRICES_PANIC + FJ_FILM_INPUT_COLOR_MATRICES;
                const char* operation = reason == FJ_FILM_INPUT_COLOR_MATRICES ? "Input matrix preparation failed" : reason == FJ_FILM_INPUT_COLOR_CAT16 ? "CAT16 matrix preparation failed"
                                                                                                                                                         : "CAT02 matrix preparation failed";
                throw_failure(status, operation);
            }
            const bool sensitivity = failure.origin == FJ_FILM_INPUT_ORIGIN_SENSITIVITY ||
                                     (failure.origin == FJ_FILM_INPUT_ORIGIN_INTERNAL && failure.reason == FJ_FILM_INPUT_INTERNAL_SENSITIVITY_PANIC);
            if (sensitivity && status.category == FJ_STATUS_PREPARATION_FAILURE) {
                diagnostic = "MalformedRequiredProfileData phase=3B field=final_sensitivity";
                return false;
            }
            if (failure.origin == FJ_FILM_INPUT_ORIGIN_SENSITIVITY && status.category == FJ_STATUS_ALLOCATION_FAILURE) {
                throw std::bad_alloc();
            }
            std::string message(sensitivity ? "Film sensitivity preparation: " : "Film input preparation: ");
            message.append(foreign);
            throw_failure(status, message);
        }
    } // namespace

    static_assert(static_cast<std::uint32_t>(Spektrafilm::RgbToRawMethod::Hanatos2025) == FJ_RAW_HANATOS_2025);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::RgbToRawMethod::Mallett2019) == FJ_RAW_MALLETT_2019);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::RgbToRawMethod::Arctic2026beta04) == FJ_RAW_ARCTIC_2026_BETA04);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::AutoExposureMethod::CenterWeighted) == FJ_METER_CENTER_WEIGHTED);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::AutoExposureMethod::Average) == FJ_METER_AVERAGE);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::AutoExposureMethod::Median) == FJ_METER_MEDIAN);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::AutoExposureMethod::Partial) == FJ_METER_PARTIAL);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::AutoExposureMethod::Matrix) == FJ_METER_MATRIX);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::AutoExposureMethod::MultiZone) == FJ_METER_MULTI_ZONE);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::AutoExposureMethod::HighlightWeighted) == FJ_METER_HIGHLIGHT_WEIGHTED);

    // SF_TEMP_BRIDGE_FILM_INPUT: same-source synchronous construction/value binding; remove S4.E.
    bool FilmSource::prepare_input(const Spektrafilm::FilmFoundationBuildInput& input, FilmRawRecipe& out, std::string& diagnostic) const {
        const FjFilmInputParams params{
            static_cast<std::uint32_t>(input.inputColorSpace), input.inputCctfDecoding ? 1u : 0u, static_cast<std::uint32_t>(input.spectralUpsamplingMode), input.cameraAutoExposureEnabled ? 1u : 0u, static_cast<std::uint32_t>(input.cameraMeteringMethod), input.manualExposureCompensationEv, input.filmFormatLongEdgeMm, input.cameraFilterOverride ? 1u : 0u, {input.cameraFilterUV[0], input.cameraFilterUV[1], input.cameraFilterUV[2]}, {input.cameraFilterIR[0], input.cameraFilterIR[1], input.cameraFilterIR[2]}, input.applyHanatos2025AdaptationWindow ? 1u : 0u, input.applyHanatos2025AdaptationSurface ? 1u : 0u, input.inputCompressionRequested ? 1u : 0u};
        const bool arctic = params.method == FJ_RAW_ARCTIC_2026_BETA04;
        const FjFilmInputResources resources{
            {input.referenceIlluminant.data(), input.referenceIlluminant.size()},
            {!arctic && input.projectionWhiteValid ? input.projectionWhiteXYZ.data() : nullptr, !arctic && input.projectionWhiteValid ? 3u : 0u},
            {arctic && input.projectionWhiteValid ? input.projectionWhiteXYZ.data() : nullptr, arctic && input.projectionWhiteValid ? 3u : 0u},
            {input.reconstructedReferenceWhiteValid ? input.reconstructedReferenceWhite.data() : nullptr, input.reconstructedReferenceWhiteValid ? 81u : 0u},
            input.tcSourceAssetHash,
            input.inputCompressionHull ? input.inputCompressionHull->hash : 0};
        FjFilmInput prepared{};
        FjFilmInputFailure failure{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
#if defined(JUICER_FILM_INPUT_TEST_HOOK)
        InputTest::before_prepare(*this, input);
#endif
        auto result = fj_legacy_film_input_prepare(_owner, &params, &resources, &prepared, &failure, &error);
#if defined(JUICER_FILM_INPUT_TEST_HOOK)
        InputTest::after_prepare(result, failure);
#endif
        if (error.length >= text.size()) {
            throw_failure({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "Film input preparation: invalid failure contract");
        }
        if (!translate(result, failure, params, std::string_view(text.data(), error.length), diagnostic)) {
            return false;
        }
        out.inputColorSpace = static_cast<int>(prepared.color.input_space);
        out.inputCctfDecoding = prepared.color.decode_cctf != 0;
        out.rgbToRawMethod = static_cast<Spektrafilm::RgbToRawMethod>(prepared.method);
        out.autoExposureEnabled = prepared.auto_exposure_enabled != 0;
        out.autoExposureMethod = static_cast<Spektrafilm::AutoExposureMethod>(prepared.meter);
        out.manualExposureCompensationEv = prepared.manual_exposure_ev;
        out.filmFormatLongEdgeMm = prepared.film_format_long_edge_mm;
        out.cameraBandPass.active = prepared.band_pass_active != 0;
        std::copy_n(prepared.uv, 3, out.cameraBandPass.uv.begin());
        std::copy_n(prepared.ir, 3, out.cameraBandPass.ir.begin());
        std::copy_n(prepared.color.rgb_to_xyz, 9, out.inputRgbToXyz.begin());
        std::copy_n(prepared.color.xyz_adapt, 9, out.inputXyzAdapt.begin());
        std::copy_n(prepared.xyz_to_linear_srgb, 9, out.xyzToLinearSrgb.begin());
        std::copy_n(prepared.nominal_white_xyz, 3, out.inputNominalWhiteXYZ.begin());
        std::copy_n(prepared.projection_white_xyz, 3, out.projectionWhiteXYZ.begin());
        out.referenceIlluminant = _profile->info.referenceIlluminant.value;
        out.projectionIlluminant = prepared.projection_illuminant == FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE ? out.referenceIlluminant : "D65";
        out.hanatos.applyWindow = prepared.apply_hanatos_window != 0;
        out.hanatos.applySurface = prepared.apply_hanatos_surface != 0;
        out.hanatos.spectralGaussianBlur = prepared.hanatos_spectral_blur;
        std::copy_n(prepared.hanatos_window, 4, out.hanatos.windowParams.begin());
        for (std::size_t channel = 0; channel < 3; ++channel) {
            std::copy_n(prepared.hanatos_surface_rgb[channel], 15, out.hanatos.surfaceParams[channel].begin());
        }
        out.hanatos.referenceIlluminant = prepared.method == FJ_RAW_HANATOS_2025 ? out.referenceIlluminant : "";
        out.inputCompressionActive = prepared.input_compression_active != 0;
        out.inputCompressionHullHash = prepared.input_hull_hash;
        out.inputCompressionHull = prepared.method == FJ_RAW_MALLETT_2019 ? nullptr : input.inputCompressionHull;
        out.tcSourceAssetHash = prepared.tc_source_hash;
        for (std::size_t wavelength = 0; wavelength < 81; ++wavelength) {
            std::copy_n(prepared.sensitivity_rgb[wavelength], 3, out.finalSensitivity[wavelength].begin());
        }
        out.finalSensitivityHash = prepared.sensitivity_hash;
        out.mallettGreenMidgrayScale = prepared.recipe_mallett_green_scale;
        out.tcLutHash = prepared.tc_hash;
        out.hash = prepared.hash;
        return true;
    }
#if defined(JUICER_FILM_INPUT_TEST_HOOK)
    namespace InputTest {
        bool translate_failure(FjStatus status, const FjFilmInputFailure& failure, const FjFilmInputParams& params, std::string_view foreign, std::string& diagnostic) {
            return translate(status, failure, params, foreign, diagnostic);
        }
    } // namespace InputTest
#endif
} // namespace JuicerAssets
