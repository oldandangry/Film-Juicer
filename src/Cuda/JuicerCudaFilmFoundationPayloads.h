#pragma once

#include <string>

#include "Cuda/JuicerCudaPayloads.h"
#include "ProcessRoot.h"
#include "RenderRecipe.h"

namespace JuicerCuda {

    bool pack_visual_grain_payload(
        const Spektrafilm::VisualGrainRecipe& recipe,
        const JuicerProcess::Root::PreparedCudaFrame::PreparedVisualGrainView& prepared,
        GrainPayload& outGrain,
        GrainKernelPayload& outKernels,
        std::string& diagnostic);

} // namespace JuicerCuda
