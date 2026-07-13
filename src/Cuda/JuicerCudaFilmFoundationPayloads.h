#pragma once

#include <string>

#include "Cuda/JuicerCudaPayloads.h"
#include "FilmJuicerEffectsFrameDescriptor.h"
#include "ProcessRoot.h"
#include "RenderRecipe.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
namespace JuicerCuda {

    bool pack_visual_grain_payload(
        const Spektrafilm::VisualGrainRecipe& recipe,
        const JuicerProcess::Root::PreparedCudaFrame::PreparedVisualGrainView& prepared,
        GrainPayload& outGrain,
        GrainKernelPayload& outKernels,
        std::string& diagnostic);

    bool pack_film_juicer_effects_payload(
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor& descriptor,
        GrainPayload& outDefects,
        GateWeavePayload& outWeave,
        std::string& diagnostic);

} // namespace JuicerCuda
#endif
