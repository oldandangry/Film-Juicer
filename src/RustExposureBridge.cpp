#include "RustExposureBridge.h"

#include <algorithm>
#include <cstddef>
#include <new>
#include <sstream>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
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
    } // namespace

    static_assert(sizeof(std::array<std::array<float, 3>, 81>) == 243 * sizeof(float));
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
} // namespace JuicerExposure
