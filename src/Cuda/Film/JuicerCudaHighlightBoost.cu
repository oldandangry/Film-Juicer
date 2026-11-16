#include "Cuda/Film/JuicerCudaHighlightBoost.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>

#include <cuda_runtime.h>

namespace {
    using JuicerCuda::CameraFilmLinearExposurePlanes;
    using JuicerCuda::HighlightBoostDescriptor;
    using JuicerCuda::HighlightBoostDisposition;
    using JuicerCuda::HighlightBoostWorkspace;

    constexpr int kThreads = 256;
    static_assert(sizeof(HighlightBoostWorkspace) <= 64ULL * 1024);

    struct HighlightFrame {
        CameraFilmLinearExposurePlanes carrier;
        std::size_t width;
        std::size_t pixels;
    };

    __device__ void include_sample(HighlightBoostWorkspace::Partial& partial, float sample) {
        if (isfinite(sample) && sample >= 0.0f) {
            partial.maximum = fmaxf(partial.maximum, sample);
        } else {
            partial.invalid = 1;
        }
    }

    __device__ void reduce_block(HighlightBoostWorkspace::Partial* shared, HighlightBoostWorkspace::Partial partial) {
        const auto thread = threadIdx.x;
        shared[thread] = partial;
        __syncthreads();
        for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
            if (thread < offset) {
                shared[thread].maximum = fmaxf(shared[thread].maximum, shared[thread + offset].maximum);
                shared[thread].invalid |= shared[thread + offset].invalid;
            }
            __syncthreads();
        }
    }

    __device__ std::size_t carrier_index(std::size_t index, std::size_t width, std::size_t stride) {
        const auto row = index / width;
        return row * stride + (index - row * width);
    }

    __global__ void reduce_highlight_blocks(HighlightFrame frame, HighlightBoostWorkspace* workspace) {
        __shared__ HighlightBoostWorkspace::Partial shared[kThreads];
        HighlightBoostWorkspace::Partial partial{};
        const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < frame.pixels; i += step) {
            const auto index = carrier_index(i, frame.width, frame.carrier.rowStrideFloats);
            include_sample(partial, frame.carrier.redSensitive[index]);
            include_sample(partial, frame.carrier.greenSensitive[index]);
            include_sample(partial, frame.carrier.blueSensitive[index]);
        }
        reduce_block(shared, partial);
        if (threadIdx.x == 0) {
            workspace->partials[blockIdx.x] = shared[0];
        }
    }

    // (exp(t) - t - 1) / t^2, extended at zero by its Taylor series.
    __device__ double normalized_exponential(double t) {
        if (t >= 0.03125) {
            return (expm1(t) - t) / (t * t);
        }
        return 0.5 + t * (1.0 / 6.0 + t * (1.0 / 24.0 + t * (1.0 / 120.0 + t * (1.0 / 720.0 +
                                                                                t * (1.0 / 5040.0 + t * (1.0 / 40320.0 + t * (1.0 / 362880.0 + t / 3628800.0)))))));
    }

    __global__ void finish_highlight_reduction(HighlightBoostDescriptor descriptor, int blocks, HighlightBoostWorkspace* workspace) {
        __shared__ HighlightBoostWorkspace::Partial shared[kThreads];
        HighlightBoostWorkspace::Partial partial{};
        for (auto i = static_cast<int>(threadIdx.x); i < blocks; i += kThreads) {
            const auto entry = workspace->partials[i];
            partial.maximum = fmaxf(partial.maximum, entry.maximum);
            partial.invalid |= entry.invalid;
        }
        reduce_block(shared, partial);
        if (threadIdx.x != 0) {
            return;
        }
        const double maximum = shared[0].maximum;
        workspace->maximum = shared[0].maximum;
        if (shared[0].invalid != 0) {
            workspace->disposition = HighlightBoostDisposition::InvalidCarrier;
        } else if (maximum <= descriptor.protection) {
            workspace->disposition = HighlightBoostDisposition::Bypass;
        } else if (const double peak = maximum * descriptor.peakGain; peak > static_cast<double>(FLT_MAX)) {
            workspace->disposition = HighlightBoostDisposition::PeakOverflow;
        } else {
            const double range = maximum - descriptor.protection;
            workspace->disposition = HighlightBoostDisposition::Applied;
            workspace->peak = peak;
            workspace->inverseRange = 1.0 / range;
            workspace->shapeAtPeak = descriptor.sharpness * range / maximum;
            workspace->scaledAmplitude = maximum * (descriptor.peakGain - 1.0) /
                                         normalized_exponential(workspace->shapeAtPeak);
        }
    }

    __device__ void boost_sample(float& sample, double protection, const HighlightBoostWorkspace& workspace) {
        const double exposure = sample;
        if (exposure <= protection) {
            return;
        }
        if (sample == workspace.maximum) {
            sample = __double2float_rn(workspace.peak);
            return;
        }
        const double fraction = (exposure - protection) * workspace.inverseRange;
        const double normalized = fraction * fraction * normalized_exponential(workspace.shapeAtPeak * fraction);
        sample = __double2float_rn(exposure + workspace.scaledAmplitude * normalized);
    }

    __global__ void apply_highlight_boost(
        HighlightFrame frame,
        double protection,
        const HighlightBoostWorkspace* workspace) {
        if (workspace->disposition != HighlightBoostDisposition::Applied) {
            return;
        }
        const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < frame.pixels; i += step) {
            const auto index = carrier_index(i, frame.width, frame.carrier.rowStrideFloats);
            boost_sample(frame.carrier.redSensitive[index], protection, *workspace);
            boost_sample(frame.carrier.greenSensitive[index], protection, *workspace);
            boost_sample(frame.carrier.blueSensitive[index], protection, *workspace);
        }
    }
} // namespace

namespace JuicerCuda {
    HighlightBoostLaunchResult launch_highlight_boost(
        const HighlightBoostDescriptor& descriptor,
        CameraFilmLinearExposurePlanes carrier,
        int width,
        int height,
        HighlightBoostWorkspace* workspace,
        cudaStream_t stream,
        int blockLimit) {
        if (descriptor.recipeHash == 0) {
            return {};
        }
        if (width <= 0 || height <= 0 || !workspace || !carrier.redSensitive || !carrier.greenSensitive ||
            !carrier.blueSensitive || carrier.redSensitive == carrier.greenSensitive ||
            carrier.redSensitive == carrier.blueSensitive || carrier.greenSensitive == carrier.blueSensitive ||
            carrier.rowStrideFloats < static_cast<std::size_t>(width) ||
            static_cast<std::size_t>(height) > std::numeric_limits<std::size_t>::max() / sizeof(float) / carrier.rowStrideFloats ||
            blockLimit <= 0 || blockLimit > kHighlightBoostMaxBlocks) {
            return {cudaErrorInvalidValue, HighlightBoostLaunchStage::Binding};
        }
        const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        const HighlightFrame frame{carrier, static_cast<std::size_t>(width), pixels};
        const auto blocks = static_cast<int>(std::min((pixels - 1) / kThreads + 1, static_cast<std::size_t>(blockLimit)));
#if defined(JUICER_CONTEXT_DRAIN_TEST_HOOK)
        using TestSupport::HighlightBoostFault;
        if (const auto fault = TestSupport::consume_highlight_boost_fault(HighlightBoostFault::PartialReduction); fault != cudaSuccess) {
            return {fault, HighlightBoostLaunchStage::PartialReduction};
        }
#endif
        reduce_highlight_blocks<<<blocks, kThreads, 0, stream>>>(frame, workspace);
        if (const auto status = cudaGetLastError(); status != cudaSuccess) {
            return {status, HighlightBoostLaunchStage::PartialReduction};
        }
#if defined(JUICER_CONTEXT_DRAIN_TEST_HOOK)
        if (const auto fault = TestSupport::consume_highlight_boost_fault(HighlightBoostFault::FinalReduction); fault != cudaSuccess) {
            return {fault, HighlightBoostLaunchStage::FinalReduction};
        }
#endif
        finish_highlight_reduction<<<1, kThreads, 0, stream>>>(descriptor, blocks, workspace);
        if (const auto status = cudaGetLastError(); status != cudaSuccess) {
            return {status, HighlightBoostLaunchStage::FinalReduction};
        }
#if defined(JUICER_CONTEXT_DRAIN_TEST_HOOK)
        if (const auto fault = TestSupport::consume_highlight_boost_fault(HighlightBoostFault::Apply); fault != cudaSuccess) {
            return {fault, HighlightBoostLaunchStage::Apply};
        }
#endif
        apply_highlight_boost<<<blocks, kThreads, 0, stream>>>(frame, descriptor.protection, workspace);
        if (const auto status = cudaGetLastError(); status != cudaSuccess) {
            return {status, HighlightBoostLaunchStage::Apply};
        }
        return {};
    }
} // namespace JuicerCuda
