#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaPayloads.h"

struct ScatterHalationFrameDescriptor;

namespace JuicerCuda {

    enum class ScatterHalationCarrierSource : std::uint8_t {
        DedicatedPreparedPlanes,
        CameraDiffusionStagePlanes
    };

    struct ScatterHalationPreparedView {
        const ScatterHalationFrameDescriptor* descriptor = nullptr;
        int fullFrameWidth = 0;
        int fullFrameHeight = 0;
        ScatterHalationCarrierSource carrierSource{};
        CameraFilmLinearExposurePlanes currentCarrier{};
        float* filterTemp = nullptr;
        float* weightedAccumulation = nullptr;
    };

    enum class ScatterHalationLaunchStage : std::uint8_t {
        None,
        Binding,
        ScatterAccumulatorClear,
        ScatterTail,
        ScatterCore,
        BackReflectionAccumulatorClear,
        BackReflectionBounce,
        BackReflectionFinalize
    };

    enum class ScatterHalationLaunchChannel : std::uint8_t {
        None,
        Red,
        Green,
        Blue
    };

    struct ScatterHalationLaunchResult {
        cudaError_t status = cudaSuccess;
        ScatterHalationLaunchStage stage = ScatterHalationLaunchStage::None;
        ScatterHalationLaunchChannel channel =
            ScatterHalationLaunchChannel::None;
        int gaussianIndex = -1;
    };

    ScatterHalationLaunchResult launch_scatter_halation(
        const ScatterHalationPreparedView& view,
        cudaStream_t stream);

} // namespace JuicerCuda
