#pragma once

#include "Cuda/JuicerCudaExecutor.h"

namespace JuicerCuda {
    class NativeCall;

    // Temporary C++ producer boundary; replaced with Rust preparation in S4.
    // Borrows all owners through the immediate admitted C-record render call.
    // No projected record or host pointer escapes; asynchronous copies are native.
    FjRenderOutcome project_and_render(
        NativeCall& call, const RenderRecipe& recipe, const FocusedRenderPayload& payload, const ExecutionFrame& frame, const FjFrame& rawFrame, const ResourceManager::SubmissionSnapshot& snapshot, PendingContextLossRecovery& recovery, FjAbortCallback abortCallback, std::string& diagnostic);
} // namespace JuicerCuda
