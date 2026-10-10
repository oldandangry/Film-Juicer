#include "RustColorBridge.h"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "ColorTransforms.h"
#include "juicer_legacy_api.h"

namespace JuicerColor {
    std::array<float, 9> cat16_matrix(const Spectral::ChromaticAdaptationWhites& whites) {
        std::array<float, 9> result{};
        const FjStatus status = fj_legacy_cat16_matrix(whites.source, whites.destination, result.data());
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "CAT16 matrix preparation failed");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        return result;
    }

    std::array<float, 3> adapt_cat16(const std::array<float, 3>& xyz, const Spectral::ChromaticAdaptationWhites& whites) {
        std::array<float, 3> result{};
        const FjStatus status = fj_legacy_adapt_cat16(xyz.data(), whites.source, whites.destination, result.data());
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "CAT16 scalar adaptation failed");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        return result;
    }

    std::array<float, 9> cat02_matrix(const Spectral::ChromaticAdaptationWhites& whites) {
        std::array<float, 9> result{};
        const FjStatus status = fj_legacy_cat02_matrix(whites.source, whites.destination, result.data());
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "CAT02 matrix preparation failed");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        return result;
    }

    namespace {
        void require_success(FjStatus status, const char* diagnostic) {
            if (status.category != FJ_STATUS_SUCCESS) {
                JuicerCuda::Failure failure;
                JuicerCuda::set_failure(failure, status, diagnostic);
                throw JuicerCuda::ExecutionFailure{std::move(failure)};
            }
        }
    } // namespace

    static_assert(sizeof(FjInputColorMatrices) == 96 && alignof(FjInputColorMatrices) == 4);
    static_assert(offsetof(FjInputColorMatrices, rgb_to_xyz) == 0 && offsetof(FjInputColorMatrices, nominal_white_xyz) == 36);
    static_assert(offsetof(FjInputColorMatrices, xyz_to_linear_srgb) == 48 && offsetof(FjInputColorMatrices, d65_white_xyz) == 84);
    static_assert(static_cast<std::uint32_t>(Spectral::InputColorSpace::DaVinciWideGamut) == FJ_INPUT_DWG);
    static_assert(static_cast<std::uint32_t>(Spectral::InputColorSpace::ITU_R_BT2020) == FJ_INPUT_BT2020);
    static_assert(static_cast<std::uint32_t>(Spectral::InputColorSpace::ACES2065_1) == FJ_INPUT_ACES2065_1);
    static_assert(static_cast<std::uint32_t>(Spectral::InputColorSpace::SRGB_Rec709) == FJ_INPUT_SRGB_REC709);

    InputMatrices input_matrices(Spectral::InputColorSpace space) {
        FjInputColorMatrices foreign{};
        require_success(fj_legacy_input_matrices(static_cast<std::uint32_t>(space), &foreign), "Input matrix preparation failed");
        InputMatrices out{};
        std::copy_n(foreign.rgb_to_xyz, 9, out.rgbToXyz.begin());
        std::copy_n(foreign.nominal_white_xyz, 3, out.nominalWhiteXYZ.begin());
        std::copy_n(foreign.xyz_to_linear_srgb, 9, out.xyzToLinearSrgb.begin());
        std::copy_n(foreign.d65_white_xyz, 3, out.d65WhiteXYZ.begin());
        return out;
    }

    std::array<float, 3> linear_srgb_to_xyz(const std::array<float, 3>& rgb) {
        std::array<float, 3> out{};
        require_success(fj_legacy_linear_srgb_to_xyz(rgb.data(), out.data()), "Linear sRGB to XYZ conversion failed");
        return out;
    }

    std::array<float, 3> project_linear_rgb_to_xyz(const std::array<float, 3>& rgb, const Spectral::Mat3& rgbToXyz, const Spectral::Mat3& xyzAdapt) {
        std::array<float, 3> out{};
        require_success(fj_legacy_project_linear_rgb_to_xyz(rgb.data(), rgbToXyz.m, xyzAdapt.m, out.data()), "Linear RGB to XYZ projection failed");
        return out;
    }
} // namespace JuicerColor
