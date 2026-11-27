#include "Cuda/JuicerCudaBlur.h"

#include "Cuda/JuicerCudaBlur.cuh"
#include "Cuda/JuicerCudaDeviceHelpers.cuh"

namespace JuicerCuda {
    namespace {
        template <typename Accumulator>
        __global__ void optics_blur_horizontal_kernel(
            const float* JUICER_RESTRICT in,
            float* out,
            int width,
            int height,
            const float* JUICER_RESTRICT k,
            int radius) {
            if (!in || !out || !k || radius <= 0 || width <= 0 || height <= 0) {
                return;
            }

            const std::size_t radiusCount = static_cast<std::size_t>(radius);
            const std::size_t kLen = 2U * radiusCount + 1U;
            const std::size_t tileW = static_cast<std::size_t>(blockDim.x) + 2U * radiusCount;

            extern __shared__ float horizontalShared[];
            float* sWeights = horizontalShared;
            float* sTile = horizontalShared + kLen;

            const std::size_t tid = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
            const std::size_t tcount = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
            const std::size_t yLocal = static_cast<std::size_t>(threadIdx.y);

            for (std::size_t i = tid; i < kLen; i += tcount) {
                sWeights[i] = k[i];
            }
            __syncthreads();

            const std::size_t widthCount = static_cast<std::size_t>(width);
            const std::size_t heightCount = static_cast<std::size_t>(height);
            const std::size_t blockYStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
            const std::size_t blockXStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
            for (std::size_t blockY = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y); blockY < heightCount; blockY += blockYStep) {
                for (std::size_t blockX = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x); blockX < widthCount; blockX += blockXStep) {
                    const std::size_t x = blockX + static_cast<std::size_t>(threadIdx.x);
                    const std::size_t y = blockY + static_cast<std::size_t>(threadIdx.y);
                    const bool inBounds = (x < widthCount && y < heightCount);
                    const std::size_t yLoad = blockY + yLocal;
                    if (yLoad < heightCount) {
                        const std::size_t rowBase = yLoad * widthCount;
                        for (std::size_t i = static_cast<std::size_t>(threadIdx.x); i < tileW; i += static_cast<std::size_t>(blockDim.x)) {
                            const std::int64_t xLoad = static_cast<std::int64_t>(blockX) + static_cast<std::int64_t>(i) - static_cast<std::int64_t>(radius);
                            const int xx = reflect_index_half_sample_wide_device(xLoad, width);
                            sTile[yLocal * tileW + i] = in[rowBase + static_cast<size_t>(xx)];
                        }
                    }

                    __syncthreads();

                    if (inBounds) {
                        Accumulator acc = 0;
                        const std::size_t tileRow = static_cast<std::size_t>(threadIdx.y) * tileW;
                        for (std::size_t kernelOffset = 0; kernelOffset < kLen; ++kernelOffset) {
                            const std::size_t tileX = static_cast<std::size_t>(threadIdx.x) + kernelOffset;
                            const float v = sTile[tileRow + tileX];
                            const float w = sWeights[kernelOffset];
                            acc += static_cast<Accumulator>(v) *
                                   static_cast<Accumulator>(w);
                        }

                        out[y * widthCount + x] =
                            (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
                    }

                    __syncthreads();
                }
            }
        }

        __global__ void optics_blur_vertical_kernel(
            const float* JUICER_RESTRICT in,
            float* out,
            int width,
            int height,
            const float* JUICER_RESTRICT k,
            int radius) {
            if (!in || !out || !k || radius <= 0 || width <= 0 || height <= 0) {
                return;
            }

            const std::size_t radiusCount = static_cast<std::size_t>(radius);
            const std::size_t kLen = 2U * radiusCount + 1U;
            const std::size_t tileW = static_cast<std::size_t>(blockDim.x);
            const std::size_t tileH = static_cast<std::size_t>(blockDim.y) + 2U * radiusCount;

            extern __shared__ float verticalShared[];
            float* sWeights = verticalShared;
            float* sTile = verticalShared + kLen;

            const std::size_t tid = static_cast<std::size_t>(threadIdx.y) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
            const std::size_t tcount = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockDim.y);
            const std::size_t xLocal = static_cast<std::size_t>(threadIdx.x);

            for (std::size_t i = tid; i < kLen; i += tcount) {
                sWeights[i] = k[i];
            }
            __syncthreads();

            const std::size_t widthCount = static_cast<std::size_t>(width);
            const std::size_t heightCount = static_cast<std::size_t>(height);
            const std::size_t blockYStep = static_cast<std::size_t>(blockDim.y) * static_cast<std::size_t>(gridDim.y);
            const std::size_t blockXStep = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
            for (std::size_t blockY = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(blockDim.y); blockY < heightCount; blockY += blockYStep) {
                for (std::size_t blockX = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x); blockX < widthCount; blockX += blockXStep) {
                    const std::size_t x = blockX + static_cast<std::size_t>(threadIdx.x);
                    const std::size_t y = blockY + static_cast<std::size_t>(threadIdx.y);
                    const bool inBounds = (x < widthCount && y < heightCount);
                    const std::size_t xLoad = blockX + xLocal;
                    if (xLoad < widthCount) {
                        for (std::size_t i = static_cast<std::size_t>(threadIdx.y); i < tileH; i += static_cast<std::size_t>(blockDim.y)) {
                            const std::int64_t yLoad = static_cast<std::int64_t>(blockY) + static_cast<std::int64_t>(i) - static_cast<std::int64_t>(radius);
                            const int yy = reflect_index_half_sample_wide_device(yLoad, height);
                            sTile[i * tileW + xLocal] =
                                in[static_cast<std::size_t>(yy) * widthCount + xLoad];
                        }
                    }

                    __syncthreads();

                    if (inBounds) {
                        double acc = 0.0;
                        for (std::size_t kernelOffset = 0; kernelOffset < kLen; ++kernelOffset) {
                            const std::size_t tileY = static_cast<std::size_t>(threadIdx.y) + kernelOffset;
                            const float v = sTile[tileY * tileW + xLocal];
                            const float w = sWeights[kernelOffset];
                            acc += static_cast<double>(v) * static_cast<double>(w);
                        }

                        out[y * widthCount + x] =
                            (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
                    }

                    __syncthreads();
                }
            }
        }

    } // namespace

    cudaError_t launch_horizontal_blur(BlurPass pass, BlurLaunch launch, BlurPrecision precision) {
        const auto kernelLength = 2 * static_cast<std::size_t>(pass.radius) + 1;
        const auto sharedBytes = (kernelLength +
                                  static_cast<std::size_t>(launch.threads.y) * (launch.threads.x + 2 * pass.radius)) *
                                 sizeof(float);
        if (precision == BlurPrecision::Float) {
            optics_blur_horizontal_kernel<float><<<launch.blocks, launch.threads, sharedBytes, launch.stream>>>(
                pass.input, pass.output, pass.width, pass.height, pass.weights, pass.radius);
        } else {
            optics_blur_horizontal_kernel<double><<<launch.blocks, launch.threads, sharedBytes, launch.stream>>>(
                pass.input, pass.output, pass.width, pass.height, pass.weights, pass.radius);
        }
        return cudaGetLastError();
    }

    cudaError_t blur_plane_in_place(BlurPass pass, BlurLaunch launch) {
        if (!pass.input || !pass.output || !pass.weights || pass.radius <= 0) {
            return cudaSuccess;
        }
        const auto status = launch_horizontal_blur(pass, launch, BlurPrecision::Double);
        if (status != cudaSuccess) {
            return status;
        }
        const auto kernelLength = 2 * static_cast<std::size_t>(pass.radius) + 1;
        const auto sharedBytes = (kernelLength +
                                  static_cast<std::size_t>(launch.threads.x) * (launch.threads.y + 2 * pass.radius)) *
                                 sizeof(float);
        optics_blur_vertical_kernel<<<launch.blocks, launch.threads, sharedBytes, launch.stream>>>(
            pass.output, pass.input, pass.width, pass.height, pass.weights, pass.radius);
        return cudaGetLastError();
    }

} // namespace JuicerCuda
