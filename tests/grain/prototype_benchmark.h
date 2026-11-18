#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <iosfwd>

namespace GrainPrototype {
    struct FilmTime {
        std::int64_t frame;
        float fraction;
        std::uint32_t seed;
        std::uint32_t nextSeed;
    };

    // Bounded signed frame indices use modulo-2^64 conversion before mixing.
    FilmTime resolve_film_time(double time, std::uint64_t sessionSeed, std::uint64_t clipToken);

    // The candidate and its bank stay alive until this synchronous call returns.
    using FilmLaunch = cudaError_t (*)(void* owner, float* c, float* m, float* y, cudaStream_t stream);

    // Capture and benchmark are deliberately separate invocations: the raw planar
    // float32 input must be frozen and identified before the first measurement.
    void capture_developed_input(const std::filesystem::path& resources,
                                 const std::filesystem::path& output);
    void capture_research(const std::filesystem::path& resources,
                          const std::filesystem::path& request,
                          std::ostream& results);
    void develop_g3(const std::filesystem::path& resources, const std::filesystem::path& request);
    void stage(const std::filesystem::path& resources, const std::filesystem::path& request, bool check = false, bool verify = false);
    void benchmark(const std::filesystem::path& resources,
                   const std::filesystem::path& input,
                   FilmLaunch film,
                   void* filmOwner,
                   std::ostream& results);
} // namespace GrainPrototype
