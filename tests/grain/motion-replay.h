// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#include "Cuda/JuicerCudaFilmGrain.h"

namespace JuicerTests::GrainMotion {
    enum class MotionPass : std::uint8_t {
        Capture,
        Film,
        Replay,
        GrainOff
    };
    using MotionSeedPair = std::array<std::uint32_t, 2>;

    struct MotionDensity {
        MotionPass pass = MotionPass::Capture;
        std::span<float> density;
        std::optional<MotionSeedPair> seeds;
        bool captureFilm = false;
        bool poisonProxyWorkspace = false;
        JuicerCuda::FilmGrain::FilmFrame admitted{};
        float samplePitch = 0.F;
        bool visited = false;
        bool filmLaunched = false;
        bool filmCaptured = false;
    };

    // The tap borrows one capture on the synchronous OFX submission thread.
    class MotionDensityScope {
    public:
        explicit MotionDensityScope(MotionDensity& density) noexcept;
        ~MotionDensityScope();
        MotionDensityScope(const MotionDensityScope&) = delete;
        MotionDensityScope& operator=(const MotionDensityScope&) = delete;

    private:
        MotionDensity* previous_;
    };

    void project_motion_density(std::span<const float> input, std::span<float> grained, double fraction);
} // namespace JuicerTests::GrainMotion
