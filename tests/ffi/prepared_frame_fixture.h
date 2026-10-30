#pragma once

#include <cstdint>
#include <string>

#include "PreparedState.h"
#include "ProcessRoot.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaFilmPayloads.h"
#include "juicer_cuda_descriptors.h"

namespace JuicerCudaTest::PreparedFrameFixture {

    // Literal frame facts intentionally permit the small synthetic extents used
    // by operator/resource tests. This fixture does not claim image inspection.
    struct FrameFacts {
        FjExtent renderExtent{};
        FjExtent fullFrameExtent{};
        FjExtent pixelDefinition{};
        float pixelSizeUm = 0.0f;
        float filmFormatLongEdgeMm = 0.0f;
        double canonicalX = 0.0;
        double canonicalY = 0.0;
        double canonicalWidth = 0.0;
        double canonicalHeight = 0.0;
        double scaleX = 0.0;
        double scaleY = 0.0;
        double pixelAspectRatio = 1.0;
        double frameTime = 0.0;
        double frameRate = 0.0;
        std::uint64_t sessionSeed = 1;
        std::uint64_t clipToken = 1;
    };

    struct Inputs {
        const FjPreparedStateView& state;
        const JuicerCuda::PreparedDescriptors& descriptors;
        const JuicerProcess::Root::PreparedFrameInput& preparation;
        const JuicerCuda::FilmPayloadInput& filmPayload;
        const JuicerCuda::VisualGrainPayloadInput& grainPayload;
        const FjHostPreparation* hostPreparation;
        float filmRouteCorrectionScale;
        const PrintExposureRecipe& printExposure;
    };

    using Consumer = bool (*)(void*, const Inputs&, std::string& diagnostic);

    // Every view, raw frame record, descriptor and span in Inputs is call-local.
    // Consumers must finish synchronously and must not retain any reference.
    bool with_prepared_frame_inputs(
        const PreparedState& state,
        const FrameFacts& facts,
        void* user,
        Consumer consume,
        std::string& diagnostic);

} // namespace JuicerCudaTest::PreparedFrameFixture
