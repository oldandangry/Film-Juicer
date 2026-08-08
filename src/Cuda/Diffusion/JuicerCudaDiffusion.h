#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <cstdint>

#include "DiffusionExecution.h"
#include "DiffusionHostBehavior.h"

namespace JuicerCuda::Diffusion {

    enum class SemanticRgbChannel : std::uint8_t {
        Red,
        Green,
        Blue
    };

    struct SpectrumPackageView {
        cufftComplex* red = nullptr;
        cufftComplex* green = nullptr;
        cufftComplex* blue = nullptr;
    };

    struct SpectrumBuildRequest {
        Spektrafilm::PlanLayout layout;
        Spektrafilm::DiffusionPsfComponents components;
        int radiusPixels = 0;
        float* transformBuffer = nullptr;
        double* normalizationScratch = nullptr;
        std::size_t normalizationScratchBytes = 0;
        cufftHandle r2cPlan = 0;
        SpectrumPackageView destination;
        cudaStream_t stream = nullptr;
    };

    struct StagePlaneSet {
        float* redSensitive = nullptr;
        float* greenSensitive = nullptr;
        float* blueSensitive = nullptr;
        float* auxiliary = nullptr;
        std::size_t rowStrideFloats = 0;
    };

    struct ExecutionWorkspaceView {
        float* transformBuffer = nullptr;
        cufftHandle r2cPlan = 0;
        cufftHandle c2rPlan = 0;
    };

    struct StageLaunchRequest {
        Spektrafilm::PlanLayout layout;
        Spektrafilm::DiffusionStageTileGeometry geometry;
        Spektrafilm::DiffusionFrameDomain fullFrame;
        SpectrumPackageView spectra;
        ExecutionWorkspaceView execution;
        StagePlaneSet* planes = nullptr;
        cudaStream_t stream = nullptr;
    };

    enum class FailureApi : std::uint8_t {
        None,
        Validation,
        Cuda,
        Cufft
    };

    struct LaunchResult {
        FailureApi api = FailureApi::None;
        int code = 0;
        const char* stage = nullptr;

        [[nodiscard]] bool ok() const noexcept {
            return api == FailureApi::None;
        }
    };

    LaunchResult build_psf_plane(
        const SpectrumBuildRequest& request,
        SemanticRgbChannel channel) noexcept;

    LaunchResult build_spectrum_package(
        const SpectrumBuildRequest& request) noexcept;

    LaunchResult launch_stage(StageLaunchRequest& request) noexcept;

} // namespace JuicerCuda::Diffusion
