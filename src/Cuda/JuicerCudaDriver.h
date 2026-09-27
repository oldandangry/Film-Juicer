#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace JuicerCuda {

    // Queries only; never creates, switches, or retains a CUDA context.
    bool query_current_cuda_context(void*& outContextOpaque, std::string& outError, int* nativeCode = nullptr);

    struct CudaAllocation {
        std::uintptr_t base = 0;
        std::size_t bytes = 0;
    };

    bool query_cuda_pointer_context(std::uintptr_t address, void*& outContext, int& nativeCode, std::string& outError);
    bool query_cuda_allocation(std::uintptr_t address, CudaAllocation& allocation, int& nativeCode, std::string& outError);

} // namespace JuicerCuda
