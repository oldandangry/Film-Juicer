// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "Cuda/JuicerCudaFilmGrainStage.h"
#include "motion-replay.h"

namespace {
    using JuicerTests::GrainMotion::MotionDensity;
    using JuicerTests::GrainMotion::MotionPass;

    constexpr JuicerCuda::GrainFailureSite kDensityTransfer{"motion_density_transfer", "grain motion density transfer failed"};
    constexpr JuicerCuda::GrainFailureSite kDensityReadback{"motion_film_density_readback", "grain motion Film density readback failed"};

    thread_local MotionDensity* motionDensity = nullptr;

    cudaError_t copy_density(std::span<float> density, JuicerCuda::FilmGrain::FilmPlanes planes, cudaStream_t stream, cudaMemcpyKind direction) noexcept {
        const auto size = density.size() / 3;
        cudaError_t copied = cudaSuccess;
        std::size_t offset = 0;
        for (float* plane : {planes.cyan, planes.magenta, planes.yellow}) {
            float* host = density.data() + offset;
            copied = direction == cudaMemcpyDeviceToHost
                         ? cudaMemcpyAsync(host, plane, size * sizeof(float), direction, stream)
                         : cudaMemcpyAsync(plane, host, size * sizeof(float), direction, stream);
            if (copied != cudaSuccess)
                break;
            offset += size;
        }
        // Drain any earlier plane copies even if a later enqueue failed.
        const auto completed = cudaStreamSynchronize(stream);
        return copied == cudaSuccess ? completed : copied;
    }
} // namespace

namespace JuicerTests::GrainMotion {
    MotionDensityScope::MotionDensityScope(MotionDensity& density) noexcept
        : previous_(std::exchange(motionDensity, &density)) {}

    MotionDensityScope::~MotionDensityScope() {
        motionDensity = previous_;
    }

    void project_motion_density(std::span<const float> input, std::span<float> grained, double fraction) {
        // Use the double-precision oracle in grain_prototype.cpp's chroma projection witness.
        constexpr std::array weights{.2126, .7152, .0722};
        const auto planeWords = input.size() / weights.size();
        for (std::size_t pixel = 0; pixel < planeWords; ++pixel) {
            std::array<double, weights.size()> before{}, after{};
            double beforeY = 0, afterY = 0;
            for (std::size_t record = 0; record < weights.size(); ++record) {
                before[record] = std::pow(10.0, -static_cast<double>(input[record * planeWords + pixel]));
                after[record] = std::pow(10.0, -static_cast<double>(grained[record * planeWords + pixel]));
                beforeY += weights[record] * before[record];
                afterY += weights[record] * after[record];
            }
            for (std::size_t record = 0; record < weights.size(); ++record) {
                const double gray = before[record] * (afterY / beforeY);
                const double expected = -std::log10(std::lerp(gray, after[record], fraction));
                grained[record * planeWords + pixel] = static_cast<float>(expected);
            }
        }
        if (!std::all_of(grained.begin(), grained.end(), [](float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("nonfinite projected motion density");
        }
    }
} // namespace JuicerTests::GrainMotion

namespace JuicerCuda::FilmGrain {
    LaunchResult launch_developed(const FilmQuad* bank, const FilmTables& tables, const FilmFrame& frame, FilmPlanes planes, cudaStream_t stream) {
        if (!motionDensity)
            return {apply(bank, tables, frame, planes, stream)};

        auto& tap = *motionDensity;
        const auto size = static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height);
        if (tap.visited || tap.pass == MotionPass::GrainOff || frame.stride != frame.width || tap.density.size() != size * 3)
            return {cudaErrorInvalidValue, &kDensityTransfer};
        tap.visited = true;
        tap.admitted = frame;
        if (tap.seeds) {
            tap.admitted.seed = (*tap.seeds)[0];
            tap.admitted.nextSeed = (*tap.seeds)[1];
        }
        tap.samplePitch = tables.footprint;
        if (tap.pass != MotionPass::Film) {
            const auto direction = tap.pass == MotionPass::Capture ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            const auto status = copy_density(tap.density, planes, stream, direction);
            return {status, status == cudaSuccess ? nullptr : &kDensityTransfer};
        }

        if (tap.poisonProxyWorkspace) {
            if (!frame.proxyWorkspace)
                return {cudaErrorInvalidValue, &kDensityTransfer};
            // Every double starts as NaN: even the empty pixel must publish
            // fresh zero partials before the batched combine consumes them.
            const auto status = cudaMemsetAsync(frame.proxyWorkspace, 0xff, sizeof(ProxyWorkspace), stream);
            if (status != cudaSuccess)
                return {status, &kDensityTransfer};
        }
        const auto launched = apply(bank, tables, tap.admitted, planes, stream);
        if (launched != cudaSuccess)
            return {launched};
        tap.filmLaunched = true;
        if (!tap.captureFilm)
            return {cudaSuccess};

        const auto status = copy_density(tap.density, planes, stream, cudaMemcpyDeviceToHost);
        tap.filmCaptured = status == cudaSuccess;
        return {status, status == cudaSuccess ? nullptr : &kDensityReadback};
    }
} // namespace JuicerCuda::FilmGrain
