#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaPayloads.h"

extern "C" cudaError_t juicer_cuda_apply_visual_grain(
    const JuicerCuda::GrainPayload*, const JuicerCuda::GrainKernelPayload*, int, int, float*, float*, float*, float*, float*, float*, float*, float*, void*);
__global__ void grain_prepare_frame_uniforms_kernel(JuicerCuda::GrainPayload);
__global__ void grain_layer_triplet_kernel(JuicerCuda::GrainPayload, int, int, const float*, float*, int);
__global__ void grain_layer_kernel(JuicerCuda::GrainPayload, int, int, const float*, float*, int, int);
__global__ void grain_apply_simple_kernel(JuicerCuda::GrainPayload, int, int, const float*, float*, int);

namespace {
    void require_cuda(cudaError_t status) {
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    }

    class DeviceBuffer {
    public:
        explicit DeviceBuffer(std::size_t count) {
            require_cuda(cudaMalloc(&data, count * sizeof(float)));
        }
        ~DeviceBuffer() {
            (void)cudaFree(data);
        }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        float* data = nullptr;
    };

    float finite_or_zero(float value) {
        return std::isfinite(value) ? value : 0.0f;
    }

    int reflect(int coordinate, int extent) {
        int wrapped = coordinate % (2 * extent);
        if (wrapped < 0) {
            wrapped += 2 * extent;
        }
        return wrapped < extent ? wrapped : 2 * extent - wrapped - 1;
    }

    // Independent scalar FIR oracle. Explicit fma matches the grain FP32
    // convolution contract; no production blur or fused-store helper is used.
    std::vector<float> blur(const std::vector<float>& source, int width, int height, const std::array<float, 3>& weights) {
        std::vector<float> horizontal(source.size()), result(source.size());
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    sum = std::fma(source[y * width + reflect(x + k - 1, width)], weights[k], sum);
                }
                horizontal[y * width + x] = finite_or_zero(sum);
            }
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    sum = std::fma(horizontal[reflect(y + k - 1, height) * width + x], weights[k], sum);
                }
                result[y * width + x] = finite_or_zero(sum);
            }
        }
        return result;
    }

    enum class Shape : std::uint8_t {
        Triplet,
        SeparateLayers,
        FinalLayerUnblurred,
        Unblurred,
        Simple
    };

    void check_case(Shape shape, int width, int height, float densityMinimum, const std::array<float, 3>& weights, float timeAlpha) {
        const std::size_t count = static_cast<std::size_t>(width) * height;
        const std::size_t bytes = count * sizeof(float);
        DeviceBuffer storage(count * 9), curves(12), kernels(9);
        DeviceBuffer uniforms((sizeof(JuicerCuda::GrainFrameUniforms) + sizeof(float) - 1) / sizeof(float));
        const std::array<float, 12> curveValues{0.0f, 1.0f, 2.0f, 0.0f, 0.2f, 0.7f, 0.0f, 0.3f, 0.6f, 0.0f, 0.5f, 0.7f};
        require_cuda(cudaMemcpy(curves.data, curveValues.data(), sizeof(curveValues), cudaMemcpyHostToDevice));
        for (int layer = 0; layer < 3; ++layer) {
            require_cuda(cudaMemcpy(kernels.data + 3u * static_cast<std::size_t>(layer), weights.data(), sizeof(weights), cudaMemcpyHostToDevice));
        }
        JuicerCuda::GrainPayload grain;
        JuicerCuda::GrainKernelPayload prepared;
        grain.active = 1;
        grain.sublayersActive = shape != Shape::Simple ? 1 : 0;
        grain.pixelSizeUm = 6.48f;
        grain.frameUniforms = reinterpret_cast<JuicerCuda::GrainFrameUniforms*>(uniforms.data);
        grain.seedBase = 73;
        grain.seedBaseNext = 74;
        grain.timeAlpha = timeAlpha;
        for (int channel = 0; channel < 3; ++channel) {
            grain.densityMin[channel] = densityMinimum;
            grain.densityMax[channel] = 2.0f;
            grain.nParticles[channel] = 30.0f;
            grain.odParticle[channel] = 0.05f;
            grain.uniformity[channel] = 0.5f;
            grain.densityCurveCmy[channel] = {curves.data, curves.data, 3, 0, 2};
            for (int layer = 0; layer < 3; ++layer) {
                grain.densityCurvesLayers[layer][channel] = curves.data + 3u * static_cast<std::size_t>(layer + 1);
                grain.densityMaxLayers[layer][channel] = 0.7f;
                grain.nParticlesLayers[layer][channel] = 12.0f + static_cast<float>(layer);
                grain.odParticleLayers[layer][channel] = 0.04f;
                const bool unblurred = shape == Shape::Unblurred ||
                                       (shape == Shape::FinalLayerUnblurred && layer == 2);
                prepared.dyeRadius[layer][channel] = unblurred ? 0 : 1;
                prepared.dyeKernel[layer][channel] = unblurred ? nullptr : kernels.data + (shape == Shape::Triplet ? 0 : 3 * layer);
            }
        }
        std::vector<float> source(count), sampled(count), expected(count), actual(count);
        const std::array<float, 10> edges{0.0f, -0.0f, 0.5f, 1.9f, -0.25f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min()};
        for (std::size_t i = 0; i < count; ++i) {
            source[i] = edges[i % edges.size()];
        }
        for (int channel = 0; channel < 3; ++channel) {
            require_cuda(cudaMemcpy(storage.data + channel * count, source.data(), bytes, cudaMemcpyHostToDevice));
        }
        require_cuda(cudaMemset(storage.data + 3 * count, 0xff, 6 * bytes));
        require_cuda(juicer_cuda_apply_visual_grain(&grain, &prepared, width, height, storage.data, storage.data + count, storage.data + 2 * count, storage.data + 3 * count, storage.data + 4 * count, storage.data + 5 * count, grain.sublayersActive != 0 ? storage.data + 6 * count : nullptr, nullptr, nullptr));

        // Generate identical particles, then evaluate the original separate
        // blur/accumulate/bias/subtract sequence on the host, one channel at a time.
        JuicerCuda::GrainPayload pass = grain;
        pass.seedBase ^= 0xA24BAED4963EE407ULL;
        pass.seedBaseNext ^= 0xA24BAED4963EE407ULL;
        const dim3 threads(32, 8);
        const dim3 blocks((width + 31) / 32, (height + 7) / 8);
        float* input = storage.data + 7 * count;
        float* output = storage.data + 8 * count;
        require_cuda(cudaMemcpy(input, source.data(), bytes, cudaMemcpyHostToDevice));
        grain_prepare_frame_uniforms_kernel<<<1, 1>>>(grain);
        require_cuda(cudaGetLastError());
        for (int channel = 0; channel < 3; ++channel) {
            std::fill(expected.begin(), expected.end(), 0.0f);
            const bool batched = shape == Shape::Triplet || shape == Shape::Unblurred;
            const int layers = batched || shape == Shape::Simple ? 1 : 3;
            for (int layer = 0; layer < layers; ++layer) {
                if (shape == Shape::Simple) {
                    grain_apply_simple_kernel<<<blocks, threads>>>(pass, width, height, input, output, channel);
                } else if (batched) {
                    grain_layer_triplet_kernel<<<blocks, threads>>>(pass, width, height, input, output, channel);
                } else {
                    grain_layer_kernel<<<blocks, threads>>>(pass, width, height, input, output, channel, layer);
                }
                require_cuda(cudaGetLastError());
                require_cuda(cudaMemcpy(sampled.data(), output, bytes, cudaMemcpyDeviceToHost));
                if (shape != Shape::Simple && prepared.dyeRadius[layer][channel] > 0) {
                    sampled = blur(sampled, width, height, weights);
                }
                for (std::size_t i = 0; i < count; ++i) {
                    expected[i] = shape == Shape::Simple ? sampled[i] : finite_or_zero(expected[i] + sampled[i]);
                }
            }
            for (std::size_t i = 0; i < count; ++i) {
                float value = expected[i];
                if (densityMinimum != 0.0f) {
                    value = finite_or_zero(value - densityMinimum);
                }
                const float delta = finite_or_zero(value - source[i]);
                const float accumulated = finite_or_zero(std::fma(1.0f, delta, 0.0f));
                expected[i] = finite_or_zero(std::fma(1.0f, accumulated, source[i]));
            }
            require_cuda(cudaMemcpy(actual.data(), storage.data + channel * count, bytes, cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < count; ++i) {
                if (!std::isfinite(actual[i]) ||
                    std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i])) {
                    std::cerr << "shape=" << static_cast<int>(shape) << " extent=" << width << 'x' << height
                              << " minimum=" << densityMinimum << " time=" << timeAlpha
                              << " channel=" << channel << " pixel=" << i << " actual=" << actual[i]
                              << " expected=" << expected[i] << '\n';
                    throw std::runtime_error("Grain differs from the unfused numerical oracle");
                }
            }
        }
    }
} // namespace

int main() {
    try {
        require_cuda(cudaSetDevice(0));
        int cases = 0;
        for (Shape shape : {Shape::Triplet, Shape::SeparateLayers, Shape::FinalLayerUnblurred, Shape::Unblurred, Shape::Simple}) {
            for (const auto extent : {std::array{1, 1}, std::array{1, 11}, std::array{33, 9}}) {
                for (float minimum : {0.0f, -0.0f, 0.08f, -0.08f, -std::numeric_limits<float>::max()}) {
                    for (float time : {0.0f, 0.375f}) {
                        check_case(shape, extent[0], extent[1], minimum, {0.25f, 0.5f, 0.25f}, time);
                        ++cases;
                    }
                }
            }
        }
        for (Shape shape : {Shape::Triplet, Shape::SeparateLayers}) {
            for (float weight : {std::numeric_limits<float>::max(), std::numeric_limits<float>::quiet_NaN()}) {
                check_case(shape, 33, 9, 0.08f, {weight, weight, weight}, 0.0f);
                ++cases;
            }
        }
        std::cout << "PASS: " << cases << " unfused grain numerical cases\n";
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
