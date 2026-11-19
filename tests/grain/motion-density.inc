// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
// Test-only pre-print density tap used by the linked OFX witnesses.

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
    JuicerCuda::FilmGrain::FilmFrame admitted{};
    float samplePitch = 0.F;
    bool visited = false;
    bool filmCaptured = false;

    cudaError_t transfer(JuicerCuda::FilmGrain::FilmFrame& frame,
                         float pitch,
                         JuicerCuda::FilmGrain::FilmPlanes planes,
                         cudaStream_t stream,
                         bool& applyGrain) noexcept {
        const auto size = static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height);
        if (visited || pass == MotionPass::GrainOff || frame.stride != frame.width || density.size() != size * 3)
            return cudaErrorInvalidValue;
        visited = true;
        if (seeds) {
            frame.seed = (*seeds)[0];
            frame.nextSeed = (*seeds)[1];
        }
        admitted = frame;
        samplePitch = pitch;
        applyGrain = pass == MotionPass::Film;
        if (applyGrain)
            return cudaSuccess;

        return copy_density(planes, stream, pass == MotionPass::Capture ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice);
    }

    cudaError_t capture_film_density(JuicerCuda::FilmGrain::FilmPlanes planes, cudaStream_t stream) noexcept {
        if (!captureFilm)
            return cudaSuccess;
        if (!visited || pass != MotionPass::Film || filmCaptured)
            return cudaErrorInvalidValue;
        const auto result = copy_density(planes, stream, cudaMemcpyDeviceToHost);
        filmCaptured = result == cudaSuccess;
        return result;
    }

    cudaError_t copy_density(JuicerCuda::FilmGrain::FilmPlanes planes, cudaStream_t stream, cudaMemcpyKind direction) noexcept {
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
        const auto completed = cudaStreamSynchronize(stream);
        return copied == cudaSuccess ? completed : copied;
    }
};

thread_local MotionDensity* motionDensity = nullptr;
struct MotionDensityScope {
    explicit MotionDensityScope(MotionDensity& density) noexcept {
        motionDensity = &density;
    }
    ~MotionDensityScope() {
        motionDensity = nullptr;
    }
};

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
    require(std::all_of(grained.begin(), grained.end(), [](float value) {
                return std::isfinite(value);
            }),
            "nonfinite projected motion density");
}
