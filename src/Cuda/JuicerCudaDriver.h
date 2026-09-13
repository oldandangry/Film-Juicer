#pragma once

#include <string>

namespace JuicerCuda {

    // Queries only; never creates, switches, or retains a CUDA context.
    bool query_current_cuda_context(void*& outContextOpaque, std::string& outError);

} // namespace JuicerCuda
