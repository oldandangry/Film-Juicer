#include "Cuda/Film/JuicerCudaScatterHalation.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "ScatterHalation.h"

namespace {
    constexpr int kBlockWidth = 16;
    constexpr int kBlockHeight = 16;
    constexpr int kLineThreads = 128;
    constexpr int kYvvTileSize = 32;
    constexpr int kYvvPrefetchRows = 8;

    enum class AccumulationMode : std::uint8_t {
        Initialize,
        Add
    };

    struct FirKernelArguments {
        int radius = 0;
        float weights[19]{};
    };

    // NOLINTBEGIN(bugprone-easily-swappable-parameters) Private CUDA kernels keep fixed named launch bindings.

    __device__ int reflect_index(int index, int extent) {
        if (index >= 0 && index < extent) {
            return index;
        }
        if (index >= -extent && index < 0) {
            return -index - 1;
        }
        if (index >= extent && index < 2 * extent) {
            return 2 * extent - 1 - index;
        }
        const int period = 2 * extent;
        index %= period;
        if (index < 0) {
            index += period;
        }
        return index >= extent ? period - 1 - index : index;
    }

    __device__ float accumulate_weighted(
        float prior,
        float value,
        float weight,
        AccumulationMode mode) {
        const float weighted = weight * value;
        return mode == AccumulationMode::Initialize ? weighted
                                                    : prior + weighted;
    }

    __device__ float finalize_scatter_value(
        float source,
        float core,
        float tail,
        float tailWeight,
        float scatterAmount) {
        const float coreTerm = (1.0f - tailWeight) * core;
        const float tailTerm = tailWeight * tail;
        const float scattered = coreTerm + tailTerm;
        const float sourceTerm = (1.0f - scatterAmount) * source;
        const float scatteredTerm = scatterAmount * scattered;
        return sourceTerm + scatteredTerm;
    }

    __global__ void clear_plane_kernel(float* plane, int width, int height) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x < width && y < height) {
            plane[static_cast<std::size_t>(y) * width + x] = 0.0f;
        }
    }

    __global__ void weighted_identity_kernel(
        const float* source,
        std::size_t sourceStride,
        float* accumulation,
        int width,
        int height,
        float weight,
        AccumulationMode mode) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) {
            return;
        }
        const std::size_t sourceIndex =
            static_cast<std::size_t>(y) * sourceStride + x;
        const std::size_t accumulationIndex =
            static_cast<std::size_t>(y) * width + x;
        accumulation[accumulationIndex] = accumulate_weighted(
            accumulation[accumulationIndex],
            source[sourceIndex],
            weight,
            mode);
    }

    __global__ void fir_vertical_kernel(
        const float* source,
        std::size_t sourceStride,
        float* temporary,
        int width,
        int height,
        FirKernelArguments filter) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) {
            return;
        }

        float sum = 0.0f;
        for (int offset = -filter.radius; offset <= filter.radius; ++offset) {
            const int reflectedY = reflect_index(y - offset, height);
            const float sample =
                source[static_cast<std::size_t>(reflectedY) * sourceStride + x];
            sum += sample * filter.weights[offset + filter.radius];
        }
        temporary[static_cast<std::size_t>(y) * width + x] = sum;
    }

    __global__ void fir_horizontal_accumulate_kernel(
        const float* temporary,
        float* accumulation,
        int width,
        int height,
        FirKernelArguments filter,
        float componentWeight,
        AccumulationMode mode) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) {
            return;
        }

        float sum = 0.0f;
        for (int offset = -filter.radius; offset <= filter.radius; ++offset) {
            const int reflectedX = reflect_index(x + offset, width);
            sum += temporary[static_cast<std::size_t>(y) * width + reflectedX] *
                   filter.weights[offset + filter.radius];
        }
        const std::size_t index = static_cast<std::size_t>(y) * width + x;
        accumulation[index] = accumulate_weighted(
            accumulation[index], sum, componentWeight, mode);
    }

    __global__ void fir_horizontal_scatter_core_kernel(
        const float* temporary,
        float* sourceAndDestination,
        std::size_t sourceStride,
        const float* tail,
        int width,
        int height,
        FirKernelArguments filter,
        float tailWeight,
        float scatterAmount) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) {
            return;
        }

        float core = 0.0f;
        for (int offset = -filter.radius; offset <= filter.radius; ++offset) {
            const int reflectedX = reflect_index(x + offset, width);
            core += temporary[static_cast<std::size_t>(y) * width + reflectedX] *
                    filter.weights[offset + filter.radius];
        }
        const std::size_t sourceIndex =
            static_cast<std::size_t>(y) * sourceStride + x;
        const std::size_t tailIndex = static_cast<std::size_t>(y) * width + x;
        const float source = sourceAndDestination[sourceIndex];
        sourceAndDestination[sourceIndex] = finalize_scatter_value(
            source, core, tail[tailIndex], tailWeight, scatterAmount);
    }

    __global__ void yvv_horizontal_kernel(
        const float* source,
        std::size_t sourceStride,
        float* temporary,
        int width,
        int height,
        float B,
        float B1,
        float B2,
        float B3) {
        __shared__ float tile[kYvvTileSize][kYvvTileSize + 1];
        const int lane = static_cast<int>(threadIdx.x);
        const int firstRow = static_cast<int>(blockIdx.x) * kYvvTileSize;
        const int y = firstRow + lane;
        const float first = y < height
                                ? source[static_cast<std::size_t>(y) * sourceStride]
                                : 0.0f;
        float w1 = first;
        float w2 = first;
        float w3 = first;
        // One full warp cooperatively stages columns, then each lane recurs
        // through one row. Out-of-frame lanes still reach every warp barrier.
        for (int firstColumn = 0; firstColumn < width; firstColumn += kYvvTileSize) {
            const int x = firstColumn + lane;
            for (int row = 0; row < kYvvTileSize; ++row) {
                const int sourceY = firstRow + row;
                tile[row][lane] = sourceY < height && x < width
                                      ? source[static_cast<std::size_t>(sourceY) * sourceStride + x]
                                      : 0.0f;
            }
            __syncwarp();
            const int columns = min(kYvvTileSize, width - firstColumn);
            for (int column = 0; column < columns; ++column) {
                const float w = B * tile[lane][column] + B1 * w1 + B2 * w2 + B3 * w3;
                tile[lane][column] = w;
                w3 = w2;
                w2 = w1;
                w1 = w;
            }
            __syncwarp();
            for (int row = 0; row < kYvvTileSize; ++row) {
                const int destinationY = firstRow + row;
                if (destinationY < height && x < width) {
                    temporary[static_cast<std::size_t>(destinationY) * width + x] = tile[row][lane];
                }
            }
            __syncwarp();
        }

        const float terminal = y < height
                                   ? temporary[static_cast<std::size_t>(y) * width + width - 1]
                                   : 0.0f;
        float y1 = terminal;
        float y2 = terminal;
        float y3 = terminal;
        for (int firstColumn = ((width - 1) / kYvvTileSize) * kYvvTileSize;
             firstColumn >= 0;
             firstColumn -= kYvvTileSize) {
            const int x = firstColumn + lane;
            for (int row = 0; row < kYvvTileSize; ++row) {
                const int sourceY = firstRow + row;
                tile[row][lane] = sourceY < height && x < width
                                      ? temporary[static_cast<std::size_t>(sourceY) * width + x]
                                      : 0.0f;
            }
            __syncwarp();
            const int columns = min(kYvvTileSize, width - firstColumn);
            for (int column = columns - 1; column >= 0; --column) {
                const float value =
                    B * tile[lane][column] + B1 * y1 + B2 * y2 + B3 * y3;
                tile[lane][column] = value;
                y3 = y2;
                y2 = y1;
                y1 = value;
            }
            __syncwarp();
            for (int row = 0; row < kYvvTileSize; ++row) {
                const int destinationY = firstRow + row;
                if (destinationY < height && x < width) {
                    temporary[static_cast<std::size_t>(destinationY) * width + x] = tile[row][lane];
                }
            }
            __syncwarp();
        }
    }

    __global__ void yvv_vertical_accumulate_kernel(
        float* temporary,
        float* accumulation,
        int width,
        int height,
        float B,
        float B1,
        float B2,
        float B3,
        float componentWeight,
        AccumulationMode mode) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (x >= width) {
            return;
        }

        const std::size_t topIndex =
            static_cast<std::size_t>(height - 1) * width + x;
        const float first = temporary[topIndex];
        float w1 = first;
        float w2 = first;
        float w3 = first;
        // Prefetch independent inputs while preserving the ordered recurrence.
        for (int firstRow = height - 1; firstRow >= 0; firstRow -= kYvvPrefetchRows) {
            float values[kYvvPrefetchRows];
#pragma unroll
            for (int row = 0; row < kYvvPrefetchRows; ++row) {
                const int y = firstRow - row;
                values[row] = y >= 0 ? temporary[static_cast<std::size_t>(y) * width + x] : 0.0f;
            }
#pragma unroll
            for (int row = 0; row < kYvvPrefetchRows; ++row) {
                const int y = firstRow - row;
                if (y >= 0) {
                    const std::size_t index = static_cast<std::size_t>(y) * width + x;
                    const float w = B * values[row] + B1 * w1 + B2 * w2 + B3 * w3;
                    temporary[index] = w;
                    w3 = w2;
                    w2 = w1;
                    w1 = w;
                }
            }
        }

        const float terminal = temporary[x];
        float y1 = terminal;
        float y2 = terminal;
        float y3 = terminal;
        for (int firstRow = 0; firstRow < height; firstRow += kYvvPrefetchRows) {
            float values[kYvvPrefetchRows];
#pragma unroll
            for (int row = 0; row < kYvvPrefetchRows; ++row) {
                const int y = firstRow + row;
                values[row] = y < height ? temporary[static_cast<std::size_t>(y) * width + x] : 0.0f;
            }
#pragma unroll
            for (int row = 0; row < kYvvPrefetchRows; ++row) {
                const int y = firstRow + row;
                if (y < height) {
                    const std::size_t index = static_cast<std::size_t>(y) * width + x;
                    const float value =
                        B * values[row] + B1 * y1 + B2 * y2 + B3 * y3;
                    accumulation[index] = accumulate_weighted(
                        accumulation[index], value, componentWeight, mode);
                    y3 = y2;
                    y2 = y1;
                    y1 = value;
                }
            }
        }
    }

    __global__ void yvv_vertical_scatter_core_kernel(
        float* temporary,
        float* sourceAndDestination,
        std::size_t sourceStride,
        const float* tail,
        int width,
        int height,
        float B,
        float B1,
        float B2,
        float B3,
        float tailWeight,
        float scatterAmount) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (x >= width) {
            return;
        }

        const std::size_t topIndex =
            static_cast<std::size_t>(height - 1) * width + x;
        const float first = temporary[topIndex];
        float w1 = first;
        float w2 = first;
        float w3 = first;
        // Prefetch independent inputs while preserving the ordered recurrence.
        for (int firstRow = height - 1; firstRow >= 0; firstRow -= kYvvPrefetchRows) {
            float values[kYvvPrefetchRows];
#pragma unroll
            for (int row = 0; row < kYvvPrefetchRows; ++row) {
                const int y = firstRow - row;
                values[row] = y >= 0 ? temporary[static_cast<std::size_t>(y) * width + x] : 0.0f;
            }
#pragma unroll
            for (int row = 0; row < kYvvPrefetchRows; ++row) {
                const int y = firstRow - row;
                if (y >= 0) {
                    const std::size_t index = static_cast<std::size_t>(y) * width + x;
                    const float w = B * values[row] + B1 * w1 + B2 * w2 + B3 * w3;
                    temporary[index] = w;
                    w3 = w2;
                    w2 = w1;
                    w1 = w;
                }
            }
        }

        const float terminal = temporary[x];
        float y1 = terminal;
        float y2 = terminal;
        float y3 = terminal;
        for (int firstRow = 0; firstRow < height; firstRow += kYvvPrefetchRows) {
            float values[kYvvPrefetchRows];
#pragma unroll
            for (int row = 0; row < kYvvPrefetchRows; ++row) {
                const int y = firstRow + row;
                values[row] = y < height ? temporary[static_cast<std::size_t>(y) * width + x] : 0.0f;
            }
#pragma unroll
            for (int row = 0; row < kYvvPrefetchRows; ++row) {
                const int y = firstRow + row;
                if (y < height) {
                    const std::size_t filterIndex =
                        static_cast<std::size_t>(y) * width + x;
                    const float core =
                        B * values[row] + B1 * y1 + B2 * y2 + B3 * y3;
                    const std::size_t sourceIndex =
                        static_cast<std::size_t>(y) * sourceStride + x;
                    const float source = sourceAndDestination[sourceIndex];
                    sourceAndDestination[sourceIndex] = finalize_scatter_value(
                        source,
                        core,
                        tail[filterIndex],
                        tailWeight,
                        scatterAmount);
                    y3 = y2;
                    y2 = y1;
                    y1 = core;
                }
            }
        }
    }

    __global__ void identity_scatter_core_kernel(
        float* sourceAndDestination,
        std::size_t sourceStride,
        const float* tail,
        int width,
        int height,
        float tailWeight,
        float scatterAmount) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) {
            return;
        }
        const std::size_t sourceIndex =
            static_cast<std::size_t>(y) * sourceStride + x;
        const std::size_t tailIndex = static_cast<std::size_t>(y) * width + x;
        const float source = sourceAndDestination[sourceIndex];
        sourceAndDestination[sourceIndex] = finalize_scatter_value(
            source, source, tail[tailIndex], tailWeight, scatterAmount);
    }

    __global__ void back_reflection_finalize_kernel(
        float* sourceAndDestination,
        std::size_t sourceStride,
        const float* accumulation,
        int width,
        int height,
        float totalStrength) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) {
            return;
        }
        const std::size_t sourceIndex =
            static_cast<std::size_t>(y) * sourceStride + x;
        const std::size_t accumulationIndex =
            static_cast<std::size_t>(y) * width + x;
        const float source = sourceAndDestination[sourceIndex];
        const float added = source + totalStrength * accumulation[accumulationIndex];
        sourceAndDestination[sourceIndex] = added / (1.0f + totalStrength);
    }

    // NOLINTEND(bugprone-easily-swappable-parameters)

    dim3 image_grid(int width, int height) {
        return dim3(
            static_cast<unsigned int>((width + kBlockWidth - 1) / kBlockWidth),
            static_cast<unsigned int>((height + kBlockHeight - 1) / kBlockHeight));
    }

    cudaError_t launch_clear(
        float* plane,
        int width,
        int height,
        cudaStream_t stream) {
        clear_plane_kernel<<<image_grid(width, height),
                             dim3(kBlockWidth, kBlockHeight),
                             0,
                             stream>>>(plane, width, height);
        return cudaGetLastError();
    }

    cudaError_t launch_weighted_gaussian(
        const float* source,
        std::size_t sourceStride,
        float* temporary,
        float* accumulation,
        int width,
        int height,
        const ScatterHalationGaussianDescriptor& gaussian,
        float componentWeight,
        AccumulationMode mode,
        cudaStream_t stream) {
        const dim3 grid = image_grid(width, height);
        const dim3 block(kBlockWidth, kBlockHeight);
        switch (gaussian.kind) {
            case ScatterHalationGaussianKind::Identity:
                weighted_identity_kernel<<<grid, block, 0, stream>>>(
                    source,
                    sourceStride,
                    accumulation,
                    width,
                    height,
                    componentWeight,
                    mode);
                return cudaGetLastError();
            case ScatterHalationGaussianKind::FirReflect: {
                FirKernelArguments filter{};
                filter.radius = gaussian.radius;
                for (std::size_t index = 0; index < gaussian.firWeights.size(); ++index) {
                    filter.weights[index] = gaussian.firWeights[index];
                }
                fir_vertical_kernel<<<grid, block, 0, stream>>>(
                    source,
                    sourceStride,
                    temporary,
                    width,
                    height,
                    filter);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
                    return status;
                }
                fir_horizontal_accumulate_kernel<<<grid, block, 0, stream>>>(
                    temporary,
                    accumulation,
                    width,
                    height,
                    filter,
                    componentWeight,
                    mode);
                return cudaGetLastError();
            }
            case ScatterHalationGaussianKind::YvvReplicate:
                yvv_horizontal_kernel<<<(height + kYvvTileSize - 1) / kYvvTileSize,
                                        kYvvTileSize,
                                        0,
                                        stream>>>(
                    source,
                    sourceStride,
                    temporary,
                    width,
                    height,
                    gaussian.B,
                    gaussian.B1,
                    gaussian.B2,
                    gaussian.B3);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
                    return status;
                }
                yvv_vertical_accumulate_kernel<<<
                    (width + kLineThreads - 1) / kLineThreads,
                    kLineThreads,
                    0,
                    stream>>>(
                    temporary,
                    accumulation,
                    width,
                    height,
                    gaussian.B,
                    gaussian.B1,
                    gaussian.B2,
                    gaussian.B3,
                    componentWeight,
                    mode);
                return cudaGetLastError();
        }
        return cudaErrorInvalidValue;
    }

    cudaError_t launch_scatter_core(
        float* sourceAndDestination,
        std::size_t sourceStride,
        float* temporary,
        const float* tail,
        int width,
        int height,
        const ScatterHalationGaussianDescriptor& gaussian,
        float tailWeight,
        float scatterAmount,
        cudaStream_t stream) {
        const dim3 grid = image_grid(width, height);
        const dim3 block(kBlockWidth, kBlockHeight);
        switch (gaussian.kind) {
            case ScatterHalationGaussianKind::Identity:
                identity_scatter_core_kernel<<<grid, block, 0, stream>>>(
                    sourceAndDestination,
                    sourceStride,
                    tail,
                    width,
                    height,
                    tailWeight,
                    scatterAmount);
                return cudaGetLastError();
            case ScatterHalationGaussianKind::FirReflect: {
                FirKernelArguments filter{};
                filter.radius = gaussian.radius;
                for (std::size_t index = 0; index < gaussian.firWeights.size(); ++index) {
                    filter.weights[index] = gaussian.firWeights[index];
                }
                fir_vertical_kernel<<<grid, block, 0, stream>>>(
                    sourceAndDestination,
                    sourceStride,
                    temporary,
                    width,
                    height,
                    filter);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
                    return status;
                }
                fir_horizontal_scatter_core_kernel<<<grid, block, 0, stream>>>(
                    temporary,
                    sourceAndDestination,
                    sourceStride,
                    tail,
                    width,
                    height,
                    filter,
                    tailWeight,
                    scatterAmount);
                return cudaGetLastError();
            }
            case ScatterHalationGaussianKind::YvvReplicate:
                yvv_horizontal_kernel<<<(height + kYvvTileSize - 1) / kYvvTileSize,
                                        kYvvTileSize,
                                        0,
                                        stream>>>(
                    sourceAndDestination,
                    sourceStride,
                    temporary,
                    width,
                    height,
                    gaussian.B,
                    gaussian.B1,
                    gaussian.B2,
                    gaussian.B3);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
                    return status;
                }
                yvv_vertical_scatter_core_kernel<<<
                    (width + kLineThreads - 1) / kLineThreads,
                    kLineThreads,
                    0,
                    stream>>>(
                    temporary,
                    sourceAndDestination,
                    sourceStride,
                    tail,
                    width,
                    height,
                    gaussian.B,
                    gaussian.B1,
                    gaussian.B2,
                    gaussian.B3,
                    tailWeight,
                    scatterAmount);
                return cudaGetLastError();
        }
        return cudaErrorInvalidValue;
    }

    JuicerCuda::ScatterHalationLaunchResult failure(
        cudaError_t status,
        JuicerCuda::ScatterHalationLaunchStage stage,
        JuicerCuda::ScatterHalationLaunchChannel channel,
        int gaussianIndex = -1) {
        return {status, stage, channel, gaussianIndex};
    }
} // namespace

namespace JuicerCuda {
    ScatterHalationLaunchResult launch_scatter_halation(
        const ScatterHalationPreparedView& view,
        cudaStream_t stream) {
        const CameraFilmLinearExposurePlanes& carrier = view.currentCarrier;
        if (!view.descriptor || view.descriptor->recipeHash == 0 ||
            view.fullFrameWidth <= 0 || view.fullFrameHeight <= 0 ||
            !carrier.redSensitive || !carrier.greenSensitive ||
            !carrier.blueSensitive ||
            carrier.rowStrideFloats <
                static_cast<std::size_t>(view.fullFrameWidth) ||
            !view.filterTemp || !view.weightedAccumulation ||
            view.filterTemp == view.weightedAccumulation ||
            view.filterTemp == carrier.redSensitive ||
            view.filterTemp == carrier.greenSensitive ||
            view.filterTemp == carrier.blueSensitive ||
            view.weightedAccumulation == carrier.redSensitive ||
            view.weightedAccumulation == carrier.greenSensitive ||
            view.weightedAccumulation == carrier.blueSensitive ||
            carrier.redSensitive == carrier.greenSensitive ||
            carrier.redSensitive == carrier.blueSensitive ||
            carrier.greenSensitive == carrier.blueSensitive) {
            return failure(
                cudaErrorInvalidValue,
                ScatterHalationLaunchStage::Binding,
                ScatterHalationLaunchChannel::None);
        }

        const std::array<float*, 3> planes{{carrier.redSensitive,
                                            carrier.greenSensitive,
                                            carrier.blueSensitive}};
        constexpr std::array<ScatterHalationLaunchChannel, 3> channels{{ScatterHalationLaunchChannel::Red,
                                                                        ScatterHalationLaunchChannel::Green,
                                                                        ScatterHalationLaunchChannel::Blue}};
        const ScatterHalationFrameDescriptor& descriptor = *view.descriptor;
        for (std::size_t channelIndex = 0; channelIndex < planes.size();
             ++channelIndex) {
            float* const plane = planes[channelIndex];
            const ScatterHalationLaunchChannel channel = channels[channelIndex];
            const ScatterHalationChannelDescriptor& schedule =
                descriptor.channels[channelIndex];

            const bool skipScatter = descriptor.scatterAmount == 0.0f;
            if (!skipScatter) {
                cudaError_t status = launch_clear(
                    view.weightedAccumulation,
                    view.fullFrameWidth,
                    view.fullFrameHeight,
                    stream);
                if (status != cudaSuccess) {
                    return failure(
                        status,
                        ScatterHalationLaunchStage::ScatterAccumulatorClear,
                        channel);
                }

                for (std::size_t gaussianIndex = 0;
                     gaussianIndex < schedule.tail.size();
                     ++gaussianIndex) {
                    status = launch_weighted_gaussian(
                        plane,
                        carrier.rowStrideFloats,
                        view.filterTemp,
                        view.weightedAccumulation,
                        view.fullFrameWidth,
                        view.fullFrameHeight,
                        schedule.tail[gaussianIndex],
                        Spektrafilm::kScatterHalationExponentialAmplitudes
                            [gaussianIndex],
                        gaussianIndex == 0 ? AccumulationMode::Initialize
                                           : AccumulationMode::Add,
                        stream);
                    if (status != cudaSuccess) {
                        return failure(
                            status,
                            ScatterHalationLaunchStage::ScatterTail,
                            channel,
                            static_cast<int>(gaussianIndex));
                    }
                }

                status = launch_scatter_core(
                    plane,
                    carrier.rowStrideFloats,
                    view.filterTemp,
                    view.weightedAccumulation,
                    view.fullFrameWidth,
                    view.fullFrameHeight,
                    schedule.core,
                    Spektrafilm::kScatterHalationTailWeights[channelIndex],
                    descriptor.scatterAmount,
                    stream);
                if (status != cudaSuccess) {
                    return failure(
                        status,
                        ScatterHalationLaunchStage::ScatterCore,
                        channel);
                }
            }

            if (schedule.totalStrength == 0.0f) {
                continue;
            }

            cudaError_t status = launch_clear(
                view.weightedAccumulation,
                view.fullFrameWidth,
                view.fullFrameHeight,
                stream);
            if (status != cudaSuccess) {
                return failure(
                    status,
                    ScatterHalationLaunchStage::BackReflectionAccumulatorClear,
                    channel);
            }
            for (std::size_t gaussianIndex = 0;
                 gaussianIndex < schedule.bounce.size();
                 ++gaussianIndex) {
                status = launch_weighted_gaussian(
                    plane,
                    carrier.rowStrideFloats,
                    view.filterTemp,
                    view.weightedAccumulation,
                    view.fullFrameWidth,
                    view.fullFrameHeight,
                    schedule.bounce[gaussianIndex],
                    Spektrafilm::kScatterHalationBounceWeights[gaussianIndex],
                    gaussianIndex == 0 ? AccumulationMode::Initialize
                                       : AccumulationMode::Add,
                    stream);
                if (status != cudaSuccess) {
                    return failure(
                        status,
                        ScatterHalationLaunchStage::BackReflectionBounce,
                        channel,
                        static_cast<int>(gaussianIndex));
                }
            }

            back_reflection_finalize_kernel<<<
                image_grid(view.fullFrameWidth, view.fullFrameHeight),
                dim3(kBlockWidth, kBlockHeight),
                0,
                stream>>>(
                plane,
                carrier.rowStrideFloats,
                view.weightedAccumulation,
                view.fullFrameWidth,
                view.fullFrameHeight,
                schedule.totalStrength);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                return failure(
                    status,
                    ScatterHalationLaunchStage::BackReflectionFinalize,
                    channel);
            }
        }
        return {};
    }
} // namespace JuicerCuda
