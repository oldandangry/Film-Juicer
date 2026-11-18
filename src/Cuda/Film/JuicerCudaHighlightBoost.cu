#include "Cuda/Film/JuicerCudaHighlightBoost.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>

#include <cuda_runtime.h>
#include <math_constants.h>

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
            const double shape = descriptor.sharpness * range / maximum;
            const double normalization = normalized_exponential(shape);
            workspace->disposition = HighlightBoostDisposition::Applied;
            workspace->protection = __double2float_rn(descriptor.protection);
            workspace->protectionResidual = __double2float_rn(descriptor.protection - static_cast<double>(workspace->protection));
            // Strict onset is the first carrier float greater than the original double.
            workspace->onset = static_cast<double>(workspace->protection) <= descriptor.protection
                                   ? nextafterf(workspace->protection, CUDART_INF_F)
                                   : workspace->protection;
            workspace->peak = __double2float_rn(peak);
            workspace->peakResidual = __double2float_rn(peak - static_cast<double>(workspace->peak));
            workspace->range = __double2float_rn(range);
            workspace->rangeResidual = __double2float_rn(range - static_cast<double>(workspace->range));
            workspace->shape = __double2float_rn(shape);
            workspace->shapeResidual = __double2float_rn(shape - static_cast<double>(workspace->shape));
            // These factors are only used for shape >= 0.5, where the denominator is well conditioned.
            if (workspace->shape >= 0.5f) {
                const double denominator = shape * shape * normalization;
                workspace->tailExpFactor = __double2float_rn(exp(shape) / denominator);
                workspace->tailLinearFactor = __double2float_rn(1.0 / denominator);
            }
            workspace->delta = __double2float_rn(maximum * (descriptor.peakGain - 1.0));
            workspace->inverseShapeNormalization = __double2float_rn(1.0 / normalization);
        }
    }

    __device__ float normalized_float(float t, float residual) {
        // Series avoids subtracting nearly equal floats.
        if (t < 0.5f) {
            return 0.5f + t * (1.0f / 6.0f + t * (1.0f / 24.0f + t * (1.0f / 120.0f + t * (1.0f / 720.0f +
                                                                                           t * (1.0f / 5040.0f + t * (1.0f / 40320.0f + t * (1.0f / 362880.0f + t / 3628800.0f)))))));
        }
        // Preserve the low argument word through the exponential. Its square and
        // higher terms are below FP32 significance for the bounded shape.
        const float numerator = fmaf(expf(t), residual, expm1f(t) - t) - residual;
        return numerator / (t * t);
    }

    __device__ void boost_sample(float& sample, const HighlightBoostWorkspace& workspace) {
        if (sample < workspace.onset) {
            return;
        }
        if (sample == workspace.maximum) {
            sample = workspace.peak;
            return;
        }
        const float distance = sample - workspace.protection;
        const float virtualProtection = distance - sample;
        const float distanceResidual = ((sample - (distance - virtualProtection)) +
                                        (-workspace.protection - virtualProtection)) -
                                       workspace.protectionResidual;
        // Division by the split range avoids a subnormal reciprocal for very large legal carriers.
        const float quotient = distance / workspace.range;
        const float correction = (fmaf(-quotient, workspace.range, distance) + distanceResidual -
                                  quotient * workspace.rangeResidual) /
                                 workspace.range;
        // Renormalize both words: near onset the correction may dominate
        // the quotient, including when the rounded protection equals sample.
        const float fraction = quotient + correction;
        const float virtualCorrection = fraction - quotient;
        const float fractionResidual = (quotient - (fraction - virtualCorrection)) +
                                       (correction - virtualCorrection);
        const float t = workspace.shape * fraction;
        const float tResidual = fmaf(workspace.shape, fraction, -t) +
                                workspace.shape * fractionResidual + workspace.shapeResidual * fraction;
        const float fractionSquared = fmaf(2.0f * fraction, fractionResidual, fraction * fraction);
        const float ratio = fractionSquared * (normalized_float(t, tResidual) * workspace.inverseShapeNormalization);
        if (workspace.shape >= 0.5f && ratio >= 0.5f) {
            // The endpoint form subtracts a nonnegative deficit from the
            // representable peak; it cannot overflow by adding to a near-peak sample.
            const float gap = workspace.maximum - sample;
            const float gapFraction = gap / workspace.range;
            const float gapResidual = (fmaf(-gapFraction, workspace.range, gap) -
                                       gapFraction * workspace.rangeResidual) /
                                      workspace.range;
            const float d = workspace.shape * gapFraction;
            const float dResidual = fmaf(workspace.shape, gapFraction, -d) +
                                    workspace.shape * gapResidual + workspace.shapeResidual * gapFraction;
            const float oneMinusExp = fmaf(expf(-d), dResidual, -expm1f(-d));
            const float tail = fmaf(oneMinusExp, workspace.tailExpFactor, -(d + dResidual) * workspace.tailLinearFactor);
            // TwoDiff retains the subtraction's rounding error, and the
            // stored peak residual retains the double-prepared endpoint.
            const float endpoint = workspace.peak - gap;
            const float virtualGap = workspace.peak - endpoint;
            const float endpointResidual = (workspace.peak - (endpoint + virtualGap)) + (virtualGap - gap);
            sample = endpoint + ((endpointResidual + workspace.peakResidual) - workspace.delta * tail);
        } else {
            sample = sample + workspace.delta * ratio;
        }
    }

    __global__ void apply_highlight_boost(
        HighlightFrame frame,
        const HighlightBoostWorkspace* workspace) {
        if (workspace->disposition != HighlightBoostDisposition::Applied) {
            return;
        }
        const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < frame.pixels; i += step) {
            const auto index = carrier_index(i, frame.width, frame.carrier.rowStrideFloats);
            boost_sample(frame.carrier.redSensitive[index], *workspace);
            boost_sample(frame.carrier.greenSensitive[index], *workspace);
            boost_sample(frame.carrier.blueSensitive[index], *workspace);
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
        apply_highlight_boost<<<blocks, kThreads, 0, stream>>>(frame, workspace);
        if (const auto status = cudaGetLastError(); status != cudaSuccess) {
            return {status, HighlightBoostLaunchStage::Apply};
        }
        return {};
    }
} // namespace JuicerCuda
