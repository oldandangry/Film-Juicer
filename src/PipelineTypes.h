#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "AgxNanSemantics.h"
#include "SpectralTypes.h"

namespace Print {
    struct Runtime;
}

namespace Pipeline {

    struct RgbLinear {
        float v[3] = { 0.0f, 0.0f, 0.0f };
    };

    struct FilmRaw {
        float v[3] = { 0.0f, 0.0f, 0.0f }; // B/G/R order
    };

    struct FilmLogRaw {
        float v[3] = { 0.0f, 0.0f, 0.0f }; // B/G/R order
    };

    struct NegativeDensityCMY {
        float v[3] = { 0.0f, 0.0f, 0.0f }; // C/M/Y order
    };

    struct PrintRaw {
        float v[3] = { 0.0f, 0.0f, 0.0f }; // C/M/Y order
    };

    struct PrintLogRaw {
        float v[3] = { 0.0f, 0.0f, 0.0f }; // C/M/Y order
    };

    struct PrintDensityCMY {
        float v[3] = { 0.0f, 0.0f, 0.0f }; // C/M/Y order
    };

    enum class DensityMedium {
        Negative,
        Print
    };

    struct NormalizedDensityCMY {
        DensityMedium medium = DensityMedium::Negative;
        float v[3] = { 0.0f, 0.0f, 0.0f }; // C/M/Y order
    };

    struct SpectralDensity {
        std::array<float, static_cast<size_t>(Spectral::kNumSamples)> v{};
    };

    struct Xyz {
        double v[3] = { 0.0, 0.0, 0.0 };
    };

    struct LogXyz {
        double v[3] = { 0.0, 0.0, 0.0 };
    };

    // Scratch buffers reused across the print exposure stage (per-thread).
    struct PrintPipelineScratch {
        std::vector<float> Tneg;
        std::vector<float> Ee_expose;
        std::vector<float> Ee_filtered;
        std::vector<float> Tprint;
        std::vector<float> Ee_viewed;
        std::vector<float> Tpreflash;
        std::vector<float> Ee_preflash;

        // Cached per-render spectral constants (avoid rebuilding 81-sample tables per pixel).
        bool enlargerIlluminantFilteredValid = false;
        std::uint64_t enlargerIlluminantWsBuildCounter = 0;
        std::uint64_t enlargerIlluminantNeutralFilterHash = 0;
        float enlargerIlluminantYShiftSteps = 0.0f;
        float enlargerIlluminantMShiftSteps = 0.0f;
        float enlargerIlluminantCShiftSteps = 0.0f;
        int enlargerIlluminantShapeK = 0;

        bool preflashRawValid = false;
        std::uint64_t preflashWsBuildCounter = 0;
        std::uint64_t preflashNeutralFilterHash = 0;
        int preflashShapeK = 0;
        float preflashRaw[3] = { 0.0f, 0.0f, 0.0f };
    };

} // namespace Pipeline
