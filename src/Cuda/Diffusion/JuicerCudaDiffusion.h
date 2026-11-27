#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaFailure.h"
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

    struct LaunchResult {
        LaunchResult() noexcept = default;
        LaunchResult(const LaunchResult&) = delete;
        LaunchResult& operator=(const LaunchResult&) = delete;
        LaunchResult(LaunchResult&&) noexcept = default;
        LaunchResult& operator=(LaunchResult&&) noexcept = default;

        Failure failure{{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
        const char* stage = nullptr;

        [[nodiscard]] bool ok() const noexcept {
            return failure.status.category == FJ_STATUS_SUCCESS;
        }
    };

    LaunchResult build_psf_plane(
        const SpectrumBuildRequest& request,
        SemanticRgbChannel channel) noexcept;

    LaunchResult build_spectrum_package(
        const SpectrumBuildRequest& request) noexcept;

    LaunchResult launch_stage(StageLaunchRequest& request) noexcept;

} // namespace JuicerCuda::Diffusion
