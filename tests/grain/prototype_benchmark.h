#pragma once

#include <cuda_runtime.h>

#include <filesystem>
#include <iosfwd>

namespace GrainPrototype {
    // The candidate and its bank stay alive until this synchronous call returns.
    using FilmLaunch = cudaError_t (*)(void* owner, float* c, float* m, float* y, cudaStream_t stream);

    // Capture and benchmark are deliberately separate invocations: the raw planar
    // float32 input must be frozen and identified before the first measurement.
    void capture_developed_input(const std::filesystem::path& resources,
                                 const std::filesystem::path& output);
    void capture_research(const std::filesystem::path& resources,
                          const std::filesystem::path& request,
                          std::ostream& results);
    void benchmark(const std::filesystem::path& resources,
                   const std::filesystem::path& input,
                   FilmLaunch film,
                   void* filmOwner,
                   std::ostream& results);
} // namespace GrainPrototype
