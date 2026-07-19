#include "Cuda/Diffusion/JuicerCudaDiffusion.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace JuicerCuda::Diffusion::Detail {

    constexpr int kThreadsPerBlock = 256;
    constexpr double kTwoPi =
        2.0 * 3.141592653589793238462643383279502884;

    struct PsfKernelComponents {
        double lambdaPixels[Spektrafilm::DiffusionPsfComponents::kCount]{};
        double weights[Spektrafilm::DiffusionPsfComponents::kCount]{};
    };

    struct PsfKernelLayout {
        int width = 0;
        int height = 0;
        int rowStride = 0;
        int radius = 0;
    };

    struct StageKernelLayout {
        int transformWidth = 0;
        int transformHeight = 0;
        int transformRowStride = 0;
        int frameWidth = 0;
        int frameHeight = 0;
        std::uint64_t frameRowStride = 0;
        int tileOriginX = 0;
        int tileOriginY = 0;
        int radius = 0;
        int outputWidth = 0;
        int outputHeight = 0;
        float scatterFraction = 0.0f;
    };

    struct SpectrumMultiplyParameters {
        std::uint64_t complexCount = 0;
        float normalization = 0.0f;
    };

    JuicerCuda::Diffusion::LaunchResult validation_failure(
        const char* stage,
        int code) noexcept {
        return {JuicerCuda::Diffusion::FailureApi::Validation, code, stage};
    }

    JuicerCuda::Diffusion::LaunchResult cuda_failure(
        const char* stage,
        cudaError_t code) noexcept {
        return {
            JuicerCuda::Diffusion::FailureApi::Cuda,
            static_cast<int>(code),
            stage};
    }

    JuicerCuda::Diffusion::LaunchResult cufft_failure(
        const char* stage,
        cufftResult code) noexcept {
        return {
            JuicerCuda::Diffusion::FailureApi::Cufft,
            static_cast<int>(code),
            stage};
    }

    JuicerCuda::Diffusion::LaunchResult success() noexcept {
        return {};
    }

    bool valid_channel(JuicerCuda::Diffusion::SemanticRgbChannel channel) noexcept {
        using JuicerCuda::Diffusion::SemanticRgbChannel;
        switch (channel) {
            case SemanticRgbChannel::Red:
            case SemanticRgbChannel::Green:
            case SemanticRgbChannel::Blue:
                return true;
            default:
                return false;
        }
    }

    const std::array<double, Spektrafilm::DiffusionPsfComponents::kCount>&
    channel_weights(
        const Spektrafilm::DiffusionPsfComponents& components,
        JuicerCuda::Diffusion::SemanticRgbChannel channel) noexcept {
        using JuicerCuda::Diffusion::SemanticRgbChannel;
        if (channel == SemanticRgbChannel::Red) {
            return components.redWeights;
        }
        if (channel == SemanticRgbChannel::Green) {
            return components.greenWeights;
        }
        return components.blueWeights;
    }

    bool valid_components(
        const Spektrafilm::DiffusionPsfComponents& components) noexcept {
        double redSum = 0.0;
        double greenSum = 0.0;
        double blueSum = 0.0;
        for (std::size_t index = 0;
             index < Spektrafilm::DiffusionPsfComponents::kCount;
             ++index) {
            const double lambda = components.lambdaPixels[index];
            const double red = components.redWeights[index];
            const double green = components.greenWeights[index];
            const double blue = components.blueWeights[index];
            if (!std::isfinite(lambda) || lambda <= 0.0 ||
                !std::isfinite(red) || red < 0.0 ||
                !std::isfinite(green) || green < 0.0 ||
                !std::isfinite(blue) || blue < 0.0) {
                return false;
            }
            redSum += red;
            greenSum += green;
            blueSum += blue;
        }
        return std::isfinite(redSum) && redSum > 0.0 &&
               std::isfinite(greenSum) && greenSum > 0.0 &&
               std::isfinite(blueSum) && blueSum > 0.0;
    }

    JuicerCuda::Diffusion::LaunchResult validate_plane_request(
        const JuicerCuda::Diffusion::SpectrumBuildRequest& request,
        JuicerCuda::Diffusion::SemanticRgbChannel channel) noexcept {
        if (!valid_channel(channel)) {
            return validation_failure("validate_psf_channel", 1);
        }
        Spektrafilm::PlanLayout expected{};
        if (!Spektrafilm::make_plan_layout(
                request.layout.width,
                request.layout.height,
                expected) ||
            expected != request.layout) {
            return validation_failure("validate_psf_layout", 2);
        }
        if ((request.layout.width & 1) != 0 ||
            request.layout.physicalRealRowFloats != request.layout.width + 2) {
            return validation_failure("validate_psf_sum_storage", 3);
        }
        if (request.radiusPixels <= 0 ||
            request.radiusPixels > (request.layout.width - 1) / 2 ||
            request.radiusPixels > (request.layout.height - 1) / 2) {
            return validation_failure("validate_psf_radius", 4);
        }
        if (request.layout.transformBytes >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return validation_failure("validate_psf_transform_bytes", 5);
        }
        if (request.transformBuffer == nullptr) {
            return validation_failure("validate_psf_transform_buffer", 6);
        }
        const std::uint64_t diameter =
            2 * static_cast<std::uint64_t>(request.radiusPixels) + 1;
        const std::uint64_t sampleCount = diameter * diameter;
        const std::uint64_t partialCount =
            (sampleCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
        if (request.normalizationScratch == nullptr ||
            reinterpret_cast<std::uintptr_t>(request.normalizationScratch) %
                    alignof(double) !=
                0 ||
            partialCount >
                std::numeric_limits<std::size_t>::max() / sizeof(double) ||
            request.normalizationScratchBytes <
                static_cast<std::size_t>(partialCount) * sizeof(double)) {
            return validation_failure("validate_psf_normalization_scratch", 27);
        }
        if (request.stream == nullptr) {
            return validation_failure("validate_psf_stream", 7);
        }
        if (!valid_components(request.components)) {
            return validation_failure("validate_psf_components", 8);
        }
        return success();
    }

    JuicerCuda::Diffusion::LaunchResult validate_package_request(
        const JuicerCuda::Diffusion::SpectrumBuildRequest& request) noexcept {
        const auto planeValidation = validate_plane_request(
            request,
            JuicerCuda::Diffusion::SemanticRgbChannel::Red);
        if (!planeValidation.ok()) {
            return planeValidation;
        }
        if (request.r2cPlan == 0) {
            return validation_failure("validate_spectrum_r2c_plan", 9);
        }
        if (request.destination.red == nullptr ||
            request.destination.green == nullptr ||
            request.destination.blue == nullptr) {
            return validation_failure("validate_spectrum_destination", 10);
        }
        if (request.destination.red == request.destination.green ||
            request.destination.red == request.destination.blue ||
            request.destination.green == request.destination.blue) {
            return validation_failure("validate_spectrum_destination_alias", 11);
        }
        auto* transform = reinterpret_cast<cufftComplex*>(request.transformBuffer);
        if (request.destination.red == transform ||
            request.destination.green == transform ||
            request.destination.blue == transform) {
            return validation_failure("validate_spectrum_transform_alias", 12);
        }
        return success();
    }

    bool valid_stage(Spektrafilm::DiffusionLinearStage stage) noexcept {
        switch (stage) {
            case Spektrafilm::DiffusionLinearStage::CameraFilmLinear:
            case Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear:
                return true;
            default:
                return false;
        }
    }

    bool distinct_stage_planes(
        const JuicerCuda::Diffusion::StagePlaneSet& planes) noexcept {
        float* const values[]{
            planes.redSensitive,
            planes.greenSensitive,
            planes.blueSensitive,
            planes.auxiliary};
        for (std::size_t left = 0; left < 4; ++left) {
            if (values[left] == nullptr) {
                return false;
            }
            for (std::size_t right = left + 1; right < 4; ++right) {
                if (values[left] == values[right]) {
                    return false;
                }
            }
        }
        return true;
    }

    JuicerCuda::Diffusion::LaunchResult validate_stage_request(
        const JuicerCuda::Diffusion::StageLaunchRequest& request) noexcept {
        Spektrafilm::PlanLayout expected{};
        if (!Spektrafilm::make_plan_layout(
                request.layout.width,
                request.layout.height,
                expected) ||
            expected != request.layout) {
            return validation_failure("validate_stage_layout", 14);
        }
        if (request.layout.transformBytes >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return validation_failure("validate_stage_transform_bytes", 15);
        }
        if (!valid_stage(request.geometry.stage) ||
            request.geometry.stageDescriptorHash == 0 ||
            !std::isfinite(request.geometry.scatterFraction) ||
            request.geometry.scatterFraction <= 0.0 ||
            request.geometry.scatterFraction > 1.0) {
            return validation_failure("validate_stage_identity", 16);
        }
        if (request.fullFrame.width < 2 || request.fullFrame.height < 2) {
            return validation_failure("validate_stage_frame", 17);
        }
        const int radius = request.geometry.radiusPixels;
        if (radius <= 0 || radius > (request.layout.width - 1) / 2 ||
            radius > (request.layout.height - 1) / 2) {
            return validation_failure("validate_stage_radius", 18);
        }
        const int validWidth = request.layout.width - 2 * radius;
        const int validHeight = request.layout.height - 2 * radius;
        const int tileCountX =
            request.fullFrame.width / validWidth +
            (request.fullFrame.width % validWidth != 0 ? 1 : 0);
        const int tileCountY =
            request.fullFrame.height / validHeight +
            (request.fullFrame.height % validHeight != 0 ? 1 : 0);
        if (request.geometry.validTileWidth != validWidth ||
            request.geometry.validTileHeight != validHeight ||
            request.geometry.tileCountX != tileCountX ||
            request.geometry.tileCountY != tileCountY) {
            return validation_failure("validate_stage_tile_geometry", 19);
        }
        if (request.execution.transformBuffer == nullptr ||
            request.execution.r2cPlan == 0 || request.execution.c2rPlan == 0 ||
            request.stream == nullptr) {
            return validation_failure("validate_stage_execution", 20);
        }
        if (request.spectra.red == nullptr || request.spectra.green == nullptr ||
            request.spectra.blue == nullptr ||
            request.spectra.red == request.spectra.green ||
            request.spectra.red == request.spectra.blue ||
            request.spectra.green == request.spectra.blue) {
            return validation_failure("validate_stage_spectra", 21);
        }
        if (request.planes == nullptr || !distinct_stage_planes(*request.planes) ||
            request.planes->rowStrideFloats <
                static_cast<std::size_t>(request.fullFrame.width) ||
            request.planes->rowStrideFloats >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(request.fullFrame.height)) {
            return validation_failure("validate_stage_planes", 22);
        }
        const auto* transform = request.execution.transformBuffer;
        float* const planeValues[]{
            request.planes->redSensitive,
            request.planes->greenSensitive,
            request.planes->blueSensitive,
            request.planes->auxiliary};
        for (float* plane : planeValues) {
            if (plane == transform) {
                return validation_failure("validate_stage_transform_alias", 23);
            }
        }
        auto* transformComplex =
            reinterpret_cast<cufftComplex*>(request.execution.transformBuffer);
        if (request.spectra.red == transformComplex ||
            request.spectra.green == transformComplex ||
            request.spectra.blue == transformComplex) {
            return validation_failure("validate_stage_spectrum_alias", 24);
        }
        return success();
    }

    PsfKernelComponents kernel_components(
        const Spektrafilm::DiffusionPsfComponents& components,
        JuicerCuda::Diffusion::SemanticRgbChannel channel) noexcept {
        PsfKernelComponents result{};
        const auto& weights = channel_weights(components, channel);
        for (std::size_t index = 0;
             index < Spektrafilm::DiffusionPsfComponents::kCount;
             ++index) {
            result.lambdaPixels[index] = components.lambdaPixels[index];
            result.weights[index] = weights[index];
        }
        return result;
    }

    __global__ void build_wrapped_psf_kernel(
        float* transform,
        PsfKernelLayout layout,
        PsfKernelComponents components,
        double* partialSums) {
        __shared__ double blockValues[kThreadsPerBlock];
        const int diameter = 2 * layout.radius + 1;
        const std::uint64_t sampleCount =
            static_cast<std::uint64_t>(diameter) *
            static_cast<std::uint64_t>(diameter);
        const std::uint64_t index =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        double storedAsDouble = 0.0;
        if (index < sampleCount) {
            const int dx = static_cast<int>(index % diameter) - layout.radius;
            const int dy = static_cast<int>(index / diameter) - layout.radius;
            const double radial = sqrt(
                static_cast<double>(dx) * static_cast<double>(dx) +
                static_cast<double>(dy) * static_cast<double>(dy));
            double value = 0.0;
            for (std::size_t component = 0;
                 component < Spektrafilm::DiffusionPsfComponents::kCount;
                 ++component) {
                const double lambda = components.lambdaPixels[component];
                value += components.weights[component] * exp(-radial / lambda) /
                         (kTwoPi * lambda * lambda);
            }
            const float stored = static_cast<float>(value);
            const int x = dx >= 0 ? dx : layout.width + dx;
            const int y = dy >= 0 ? dy : layout.height + dy;
            transform[static_cast<std::uint64_t>(y) * layout.rowStride + x] =
                stored;
            storedAsDouble = static_cast<double>(stored);
        }
        blockValues[threadIdx.x] = storedAsDouble;
        __syncthreads();
        for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                blockValues[threadIdx.x] += blockValues[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partialSums[blockIdx.x] = blockValues[0];
        }
    }

    __global__ void reduce_psf_sum_kernel(
        double* partialSums,
        std::uint64_t partialCount) {
        __shared__ double blockValues[kThreadsPerBlock];
        double local = 0.0;
        for (std::uint64_t index = threadIdx.x; index < partialCount;
             index += kThreadsPerBlock) {
            local += partialSums[index];
        }
        blockValues[threadIdx.x] = local;
        __syncthreads();
        for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                blockValues[threadIdx.x] += blockValues[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partialSums[0] = blockValues[0];
        }
    }

    __global__ void normalize_wrapped_psf_kernel(
        float* transform,
        PsfKernelLayout layout,
        const double* sum) {
        const int diameter = 2 * layout.radius + 1;
        const std::uint64_t sampleCount =
            static_cast<std::uint64_t>(diameter) *
            static_cast<std::uint64_t>(diameter);
        const std::uint64_t index =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= sampleCount) {
            return;
        }

        const double total = *sum;
        const int dx = static_cast<int>(index % diameter) - layout.radius;
        const int dy = static_cast<int>(index / diameter) - layout.radius;
        const int x = dx >= 0 ? dx : layout.width + dx;
        const int y = dy >= 0 ? dy : layout.height + dy;
        const std::uint64_t offset =
            static_cast<std::uint64_t>(y) * layout.rowStride + x;
        if (isfinite(total) && total > 0.0) {
            transform[offset] = static_cast<float>(
                static_cast<double>(transform[offset]) / total);
        } else {
            transform[offset] = nanf("");
        }
    }

    __global__ void clear_psf_padding_kernel(float* transform, PsfKernelLayout layout) {
        const unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= static_cast<unsigned int>(layout.height)) {
            return;
        }
        const std::uint64_t row = static_cast<std::uint64_t>(y) * layout.rowStride;
        transform[row + layout.width] = 0.0f;
        transform[row + layout.width + 1] = 0.0f;
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    __device__ long long reflect_index(long long index, int extent) {
        const long long period = 2LL * static_cast<long long>(extent - 1);
        long long value = index % period;
        if (value < 0) {
            value += period;
        }
        return value < extent ? value : period - value;
    }

    __global__ void load_reflect_tile_kernel(
        float* transform,
        const float* input,
        StageKernelLayout layout) {
        const std::uint64_t index =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::uint64_t physicalCount =
            static_cast<std::uint64_t>(layout.transformHeight) *
            static_cast<std::uint64_t>(layout.transformRowStride);
        if (index >= physicalCount) {
            return;
        }
        const int transformX =
            static_cast<int>(index % layout.transformRowStride);
        if (transformX >= layout.transformWidth) {
            transform[index] = 0.0f;
            return;
        }
        const int transformY =
            static_cast<int>(index / layout.transformRowStride);
        const long long sourceX = reflect_index(
            static_cast<long long>(layout.tileOriginX) - layout.radius + transformX,
            layout.frameWidth);
        const long long sourceY = reflect_index(
            static_cast<long long>(layout.tileOriginY) - layout.radius + transformY,
            layout.frameHeight);
        transform[index] =
            input[static_cast<std::uint64_t>(sourceY) * layout.frameRowStride +
                  static_cast<std::uint64_t>(sourceX)];
    }

    __global__ void multiply_spectrum_kernel(
        cufftComplex* transform,
        const cufftComplex* spectrum,
        SpectrumMultiplyParameters parameters) {
        const std::uint64_t index =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= parameters.complexCount) {
            return;
        }
        const cufftComplex value = transform[index];
        const cufftComplex filter = spectrum[index];
        transform[index] = {
            (value.x * filter.x - value.y * filter.y) * parameters.normalization,
            (value.x * filter.y + value.y * filter.x) * parameters.normalization};
    }

    __global__ void crop_mix_tile_kernel(
        const float* input,
        float* output,
        const float* transform,
        StageKernelLayout layout) {
        const std::uint64_t index =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::uint64_t outputCount =
            static_cast<std::uint64_t>(layout.outputWidth) *
            static_cast<std::uint64_t>(layout.outputHeight);
        if (index >= outputCount) {
            return;
        }
        const int localX = static_cast<int>(index % layout.outputWidth);
        const int localY = static_cast<int>(index / layout.outputWidth);
        const std::uint64_t planeOffset =
            static_cast<std::uint64_t>(layout.tileOriginY + localY) *
                layout.frameRowStride +
            static_cast<std::uint64_t>(layout.tileOriginX + localX);
        const std::uint64_t transformOffset =
            static_cast<std::uint64_t>(layout.radius + localY) *
                layout.transformRowStride +
            static_cast<std::uint64_t>(layout.radius + localX);
        const float current = input[planeOffset];
        const float convolved = transform[transformOffset];
        output[planeOffset] =
            (1.0f - layout.scatterFraction) * current +
            layout.scatterFraction * convolved;
    }

    JuicerCuda::Diffusion::LaunchResult check_kernel_launch(
        const char* stage) noexcept {
        const cudaError_t result = cudaPeekAtLastError();
        return result == cudaSuccess ? success() : cuda_failure(stage, result);
    }

    bool kernel_block_count(
        std::uint64_t sampleCount,
        unsigned int& blockCount) noexcept {
        const std::uint64_t blocks =
            (sampleCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
        if (sampleCount == 0 ||
            blocks >
                static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
            blockCount = 0;
            return false;
        }
        blockCount = static_cast<unsigned int>(blocks);
        return true;
    }

    JuicerCuda::Diffusion::LaunchResult launch_all_tiles(
        const JuicerCuda::Diffusion::StageLaunchRequest& request,
        const float* input,
        float* output,
        const cufftComplex* spectrum) noexcept {
        const std::uint64_t physicalCount =
            static_cast<std::uint64_t>(request.layout.realDistance);
        const std::uint64_t complexCount =
            static_cast<std::uint64_t>(request.layout.complexDistance);
        unsigned int physicalBlocks = 0;
        unsigned int complexBlocks = 0;
        if (!kernel_block_count(physicalCount, physicalBlocks) ||
            !kernel_block_count(complexCount, complexBlocks)) {
            return validation_failure("validate_stage_launch_extent", 25);
        }
        const float normalization =
            1.0f /
            static_cast<float>(request.layout.width * request.layout.height);

        for (int tileY = 0; tileY < request.geometry.tileCountY; ++tileY) {
            const int originY = tileY * request.geometry.validTileHeight;
            const int remainingHeight = request.fullFrame.height - originY;
            const int outputHeight =
                remainingHeight < request.geometry.validTileHeight
                    ? remainingHeight
                    : request.geometry.validTileHeight;
            for (int tileX = 0; tileX < request.geometry.tileCountX; ++tileX) {
                const int originX = tileX * request.geometry.validTileWidth;
                const int remainingWidth = request.fullFrame.width - originX;
                const int outputWidth =
                    remainingWidth < request.geometry.validTileWidth
                        ? remainingWidth
                        : request.geometry.validTileWidth;
                const StageKernelLayout kernelLayout{
                    request.layout.width,
                    request.layout.height,
                    request.layout.physicalRealRowFloats,
                    request.fullFrame.width,
                    request.fullFrame.height,
                    static_cast<std::uint64_t>(request.planes->rowStrideFloats),
                    originX,
                    originY,
                    request.geometry.radiusPixels,
                    outputWidth,
                    outputHeight,
                    static_cast<float>(request.geometry.scatterFraction)};

                load_reflect_tile_kernel<<<
                    physicalBlocks,
                    kThreadsPerBlock,
                    0,
                    request.stream>>>(
                    request.execution.transformBuffer,
                    input,
                    kernelLayout);
                auto launch = check_kernel_launch("launch_load_reflect_tile");
                if (!launch.ok()) {
                    return launch;
                }

                const cufftResult r2cResult = cufftExecR2C(
                    request.execution.r2cPlan,
                    request.execution.transformBuffer,
                    reinterpret_cast<cufftComplex*>(
                        request.execution.transformBuffer));
                if (r2cResult != CUFFT_SUCCESS) {
                    return cufft_failure("cufftExecR2C_stage_tile", r2cResult);
                }

                multiply_spectrum_kernel<<<
                    complexBlocks,
                    kThreadsPerBlock,
                    0,
                    request.stream>>>(
                    reinterpret_cast<cufftComplex*>(
                        request.execution.transformBuffer),
                    spectrum,
                    SpectrumMultiplyParameters{complexCount, normalization});
                launch = check_kernel_launch("launch_multiply_spectrum");
                if (!launch.ok()) {
                    return launch;
                }

                const cufftResult c2rResult = cufftExecC2R(
                    request.execution.c2rPlan,
                    reinterpret_cast<cufftComplex*>(
                        request.execution.transformBuffer),
                    request.execution.transformBuffer);
                if (c2rResult != CUFFT_SUCCESS) {
                    return cufft_failure("cufftExecC2R_stage_tile", c2rResult);
                }

                const std::uint64_t outputCount =
                    static_cast<std::uint64_t>(outputWidth) *
                    static_cast<std::uint64_t>(outputHeight);
                unsigned int outputBlocks = 0;
                if (!kernel_block_count(outputCount, outputBlocks)) {
                    return validation_failure("validate_stage_output_extent", 26);
                }
                crop_mix_tile_kernel<<<
                    outputBlocks,
                    kThreadsPerBlock,
                    0,
                    request.stream>>>(
                    input,
                    output,
                    request.execution.transformBuffer,
                    kernelLayout);
                launch = check_kernel_launch("launch_crop_mix_tile");
                if (!launch.ok()) {
                    return launch;
                }
            }
        }
        return success();
    }

} // namespace JuicerCuda::Diffusion::Detail

namespace JuicerCuda::Diffusion {

    using Detail::build_wrapped_psf_kernel;
    using Detail::check_kernel_launch;
    using Detail::clear_psf_padding_kernel;
    using Detail::cuda_failure;
    using Detail::cufft_failure;
    using Detail::kernel_components;
    using Detail::kThreadsPerBlock;
    using Detail::launch_all_tiles;
    using Detail::normalize_wrapped_psf_kernel;
    using Detail::PsfKernelLayout;
    using Detail::reduce_psf_sum_kernel;
    using Detail::success;
    using Detail::validate_package_request;
    using Detail::validate_plane_request;
    using Detail::validate_stage_request;
    using Detail::validation_failure;

    LaunchResult build_psf_plane(
        const SpectrumBuildRequest& request,
        SemanticRgbChannel channel) noexcept {
        const LaunchResult validation = validate_plane_request(request, channel);
        if (!validation.ok()) {
            return validation;
        }

        const cudaError_t clearResult = cudaMemsetAsync(
            request.transformBuffer,
            0,
            static_cast<std::size_t>(request.layout.transformBytes),
            request.stream);
        if (clearResult != cudaSuccess) {
            return cuda_failure("cudaMemsetAsync_psf_transform", clearResult);
        }

        const int diameter = 2 * request.radiusPixels + 1;
        const std::uint64_t sampleCount =
            static_cast<std::uint64_t>(diameter) *
            static_cast<std::uint64_t>(diameter);
        const std::uint64_t blockCount64 =
            (sampleCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
        if (blockCount64 >
            static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
            return validation_failure("validate_psf_launch_extent", 13);
        }
        const auto blockCount = static_cast<unsigned int>(blockCount64);
        const PsfKernelLayout kernelLayout{
            request.layout.width,
            request.layout.height,
            request.layout.physicalRealRowFloats,
            request.radiusPixels};
        build_wrapped_psf_kernel<<<blockCount, kThreadsPerBlock, 0, request.stream>>>(
            request.transformBuffer,
            kernelLayout,
            kernel_components(request.components, channel),
            request.normalizationScratch);
        LaunchResult launch = check_kernel_launch("launch_build_wrapped_psf");
        if (!launch.ok()) {
            return launch;
        }

        reduce_psf_sum_kernel<<<1, kThreadsPerBlock, 0, request.stream>>>(
            request.normalizationScratch,
            blockCount64);
        launch = check_kernel_launch("launch_reduce_psf_sum");
        if (!launch.ok()) {
            return launch;
        }

        normalize_wrapped_psf_kernel<<<
            blockCount,
            kThreadsPerBlock,
            0,
            request.stream>>>(
            request.transformBuffer,
            kernelLayout,
            request.normalizationScratch);
        launch = check_kernel_launch("launch_normalize_wrapped_psf");
        if (!launch.ok()) {
            return launch;
        }

        const unsigned int paddingBlocks = static_cast<unsigned int>(
            (request.layout.height + kThreadsPerBlock - 1) / kThreadsPerBlock);
        clear_psf_padding_kernel<<<
            paddingBlocks,
            kThreadsPerBlock,
            0,
            request.stream>>>(
            request.transformBuffer,
            kernelLayout);
        return check_kernel_launch("launch_clear_psf_padding");
    }

    LaunchResult build_spectrum_package(
        const SpectrumBuildRequest& request) noexcept {
        const LaunchResult validation = validate_package_request(request);
        if (!validation.ok()) {
            return validation;
        }

        constexpr SemanticRgbChannel kChannels[]{
            SemanticRgbChannel::Red,
            SemanticRgbChannel::Green,
            SemanticRgbChannel::Blue};
        cufftComplex* const destinations[]{
            request.destination.red,
            request.destination.green,
            request.destination.blue};
        constexpr const char* kTransformStages[]{
            "cufftExecR2C_psf_red",
            "cufftExecR2C_psf_green",
            "cufftExecR2C_psf_blue"};
        constexpr const char* kCopyStages[]{
            "cudaMemcpyAsync_spectrum_red",
            "cudaMemcpyAsync_spectrum_green",
            "cudaMemcpyAsync_spectrum_blue"};

        for (std::size_t channel = 0; channel < 3; ++channel) {
            SpectrumBuildRequest planeRequest = request;
            planeRequest.normalizationScratch =
                reinterpret_cast<double*>(destinations[channel]);
            planeRequest.normalizationScratchBytes =
                static_cast<std::size_t>(request.layout.transformBytes);
            const LaunchResult plane =
                build_psf_plane(planeRequest, kChannels[channel]);
            if (!plane.ok()) {
                return plane;
            }
            const cufftResult transformResult = cufftExecR2C(
                request.r2cPlan,
                request.transformBuffer,
                reinterpret_cast<cufftComplex*>(request.transformBuffer));
            if (transformResult != CUFFT_SUCCESS) {
                return cufft_failure(kTransformStages[channel], transformResult);
            }
            const cudaError_t copyResult = cudaMemcpyAsync(
                destinations[channel],
                request.transformBuffer,
                static_cast<std::size_t>(request.layout.transformBytes),
                cudaMemcpyDeviceToDevice,
                request.stream);
            if (copyResult != cudaSuccess) {
                return cuda_failure(kCopyStages[channel], copyResult);
            }
        }
        return success();
    }

    LaunchResult launch_stage(StageLaunchRequest& request) noexcept {
        const LaunchResult validation = validate_stage_request(request);
        if (!validation.ok()) {
            return validation;
        }

        float** current[]{
            &request.planes->redSensitive,
            &request.planes->greenSensitive,
            &request.planes->blueSensitive};
        const cufftComplex* spectra[]{
            request.spectra.red,
            request.spectra.green,
            request.spectra.blue};

        for (int channel = 0; channel < 3; ++channel) {
            const LaunchResult launched = launch_all_tiles(
                request,
                *current[channel],
                request.planes->auxiliary,
                spectra[channel]);
            if (!launched.ok()) {
                return launched;
            }
            std::swap(*current[channel], request.planes->auxiliary);
        }
        return success();
    }

} // namespace JuicerCuda::Diffusion
