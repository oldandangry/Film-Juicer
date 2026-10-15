#include "RustExposureBridge.h"

#include <algorithm>
#include <cstddef>
#include <new>
#include <sstream>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "ColorTransforms.h"
#include "Logging.h"
#include "RenderRecipe.h"
#include "SpectralData.h"
#include "juicer_legacy_api.h"

namespace JuicerExposure {
    namespace {
        struct Diagnostic {
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        };
        void require_success(FjStatus status, const Diagnostic& diagnostic, const char* operation) {
            if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
                // Preserve the original vector allocation's OFX memory terminal.
                throw std::bad_alloc();
            }
            if (status.category != FJ_STATUS_SUCCESS) {
                JuicerCuda::Failure failure;
                JuicerCuda::set_failure(failure, status, std::string(operation) + ": " + std::string(diagnostic.bytes.data(), diagnostic.error.length));
                throw JuicerCuda::ExecutionFailure{std::move(failure)};
            }
        }
        bool completed(FjStatus status, const Diagnostic& error, const char* operation, std::string& diagnostic) {
            if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
                diagnostic.assign(error.bytes.data(), error.error.length);
                return false;
            }
            require_success(status, error, operation);
            return true;
        }
    } // namespace

    static_assert(sizeof(std::array<std::array<float, 3>, 81>) == 243 * sizeof(float));
    static_assert(sizeof(FjInputColorConversion) == 84 && alignof(FjInputColorConversion) == 4);
    static_assert(offsetof(FjInputColorConversion, input_space) == 0 && offsetof(FjInputColorConversion, decode_cctf) == 4);
    static_assert(offsetof(FjInputColorConversion, adapt_xyz) == 8 && offsetof(FjInputColorConversion, rgb_to_xyz) == 12 && offsetof(FjInputColorConversion, xyz_adapt) == 48);
    static_assert(alignof(std::array<std::array<float, 3>, 81>) == alignof(float));
    static_assert(sizeof(FjReferenceWhiteInput) == 32 && alignof(FjReferenceWhiteInput) == 8);
    static_assert(sizeof(FjReferenceWhite) == 324 && alignof(FjReferenceWhite) == 4);
    static_assert(sizeof(FjSensitivityInput) == 104 && alignof(FjSensitivityInput) == 8);
    static_assert(sizeof(FjSensitivity) == 984 && alignof(FjSensitivity) == 8);
    static_assert(sizeof(FjSensitivityFailure) == 8 && alignof(FjSensitivityFailure) == 4);
    static_assert(static_cast<unsigned>(Spektrafilm::RgbToRawMethod::Hanatos2025) == 0);
    static_assert(static_cast<unsigned>(Spektrafilm::RgbToRawMethod::Mallett2019) == 1);
    static_assert(static_cast<unsigned>(Spektrafilm::RgbToRawMethod::Arctic2026beta04) == 2);

    bool reference_white(const Spectral::ReconstructionLut& spectra, float spectralBlur, const std::array<float, 3>& whiteXyz, std::array<float, 81>& out, std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        const FjReferenceWhiteInput input{{spectra.data.data(), spectra.data.size()},
                                          {whiteXyz[0], whiteXyz[1], whiteXyz[2]},
                                          spectralBlur};
        FjReferenceWhite result{};
        Diagnostic error;
        const FjStatus status = fj_legacy_reconstruction_reference_white(&input, &result, &error.error);
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            diagnostic.assign(error.bytes.data(), error.error.length);
            return false;
        }
        require_success(status, error, "Hanatos reference preparation");
        std::copy_n(result.samples, out.size(), out.begin());
        return true;
    }

    bool prepare_sensitivity(const std::array<std::array<float, 3>, 81>& linearSensitivity,
                             FilmRawRecipe& recipe,
                             const Spektrafilm::FilmFoundationBuildInput& input) {
        const bool windowActive = recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025 && recipe.hanatos.applyWindow;
        const FjSensitivityInput request{
            {linearSensitivity.front().data(), 243},
            {input.referenceIlluminant.data(), input.referenceIlluminant.size()},
            static_cast<std::uint32_t>(recipe.rgbToRawMethod),
            recipe.cameraBandPass.active ? 1u : 0u,
            recipe.hanatos.applyWindow ? 1u : 0u,
            {recipe.cameraBandPass.uv[0], recipe.cameraBandPass.uv[1], recipe.cameraBandPass.uv[2]},
            {recipe.cameraBandPass.ir[0], recipe.cameraBandPass.ir[1], recipe.cameraBandPass.ir[2]},
            {windowActive ? recipe.hanatos.windowParams.data() : nullptr, windowActive ? 4u : 0u},
            {windowActive && input.reconstructedReferenceWhiteValid ? input.reconstructedReferenceWhite.data() : nullptr,
             windowActive && input.reconstructedReferenceWhiteValid ? 81u : 0u}};
        FjSensitivity result{};
        FjSensitivityFailure failure{};
        Diagnostic error;
        const FjStatus status = fj_legacy_exposure_sensitivity(&request, &result, &failure, &error.error);
        if (failure.hash_failure == 1) {
            std::ostringstream message;
            message << "FATAL: hash_float_span encountered non-finite sample at index " << failure.hash_sample_index;
            JTRACE("HASH", message.str());
        }
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            return false;
        }
        require_success(status, error, "Film sensitivity preparation");
        for (std::size_t i = 0; i < recipe.finalSensitivity.size(); ++i) {
            std::copy_n(result.values_rgb[i], 3, recipe.finalSensitivity[i].begin());
        }
        recipe.finalSensitivityHash = result.hash;
        recipe.mallettGreenMidgrayScale = result.mallett_green_scale;
        return true;
    }

    // FJ_TEMP_BRIDGE: focused Mallett mid-gray value binding; remove S4.E.
    bool mallett_midgray(const Spectral::MallettBasis& basis,
                         std::span<const float> illuminant,
                         const std::array<std::array<float, 3>, 81>& sensitivity,
                         Spectral::FilmRawConfig& config,
                         std::string& diagnostic) {
        diagnostic.clear();
        FjMallettMidgrayInput input{};
        input.color.input_space = static_cast<std::uint32_t>(config.inputColorSpace);
        input.color.decode_cctf = config.applyCctfDecoding ? 1u : 0u;
        input.color.adapt_xyz = config.applyInputChromaticAdapt ? 1u : 0u;
        std::copy_n(config.inputRGBToXYZ.m, 9, input.color.rgb_to_xyz);
        std::copy_n(config.inputXYZAdapt.m, 9, input.color.xyz_adapt);
        std::copy_n(config.xyzToLinearSrgb.m, 9, input.xyz_to_linear_srgb);
        input.basis_rgb = {basis.data.data(), basis.data.size()};
        input.illuminant = {illuminant.data(), illuminant.size()};
        input.sensitivity_rgb = {sensitivity.front().data(), 243};
        FjMallettMidgray result{};
        Diagnostic error;
        const auto status = fj_legacy_exposure_mallett_midgray(&input, &result, &error.error);
        if (!completed(status, error, "Mallett mid-gray preparation", diagnostic)) {
            return false;
        }
        std::copy_n(result.midgray_dwg_rgb, 3, config.midgrayDWG);
        std::copy_n(result.raw_midgray_bgr, 3, config.rawMidgray);
        config.rawMidgrayGreen = result.raw_green;
        config.midgrayScale = result.scale;
        return true;
    }

    // FJ_TEMP_BRIDGE: TC normalization value binding; remove S4.E.
    bool tc_midgray(float green, Spectral::FilmRawConfig& config, std::string& diagnostic) {
        diagnostic.clear();
        FjMidgrayNormalization result{};
        Diagnostic error;
        const auto status = fj_legacy_exposure_tc_midgray(green, &result, &error.error);
        if (!completed(status, error, "TC mid-gray normalization", diagnostic)) {
            return false;
        }
        config.rawMidgrayGreen = result.raw_green;
        config.midgrayScale = result.scale;
        return true;
    }

    // FJ_TEMP_BRIDGE: shared reference source value binding; remove S4.E.
    bool reference_source(float exposureEv, float& out, std::string& diagnostic) {
        out = 0.0f;
        diagnostic.clear();
        Diagnostic error;
        return completed(fj_legacy_exposure_reference_source(exposureEv, &out, &error.error), error, "Reference source preparation", diagnostic);
    }

    // FJ_TEMP_BRIDGE: synthetic Mallett reference value binding; remove S4.E.
    bool mallett_reference_raw(const Spectral::MallettBasis& basis,
                               std::span<const float> illuminant,
                               const std::array<std::array<float, 3>, 81>& sensitivity,
                               float source,
                               float greenScale,
                               std::array<float, 3>& out,
                               std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        const FjMallettReferenceInput input{
            {basis.data.data(), basis.data.size()},
            {illuminant.data(), illuminant.size()},
            {sensitivity.front().data(), 243},
            source,
            greenScale};
        FjReferenceRaw result{};
        Diagnostic error;
        const auto status = fj_legacy_exposure_mallett_reference_raw(&input, &result, &error.error);
        if (!completed(status, error, "Mallett reference preparation", diagnostic)) {
            return false;
        }
        std::copy_n(result.rgb, 3, out.begin());
        return true;
    }
} // namespace JuicerExposure
