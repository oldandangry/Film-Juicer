#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "AgxNanSemantics.h"
#include "SpectralTypes.h"

struct WorkingState;
namespace Print {
    struct Params;
    struct Runtime;
}
namespace Couplers {
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

    // Shared print preflash helper used by the CPU print stage and CUDA resource validation hooks.
    bool compute_preflash_raw(
        const WorkingState& ws,
        const Print::Runtime& printRuntime,
        float outRaw[3],
        int& outShapeK);

    // Shared print-stage seam used by both pipeline orchestration and CUDA-side validation hooks.
    // Keep this centralized so the stage declarations cannot drift across TUs.
    struct ExposePrintInputs {
        const Print::Runtime* printRuntime = nullptr;
        const Print::Params* printParams = nullptr;
        NegativeDensityCMY negativeDensity;
        float midgrayFactor = 1.0f;
    };

    struct ExposePrintOutputs {
        PrintRaw printRaw;
        PrintLogRaw printLogRaw;
    };

    class ExposePrintStage {
    public:
        static bool run(
            const WorkingState& ws,
            const ExposePrintInputs& in,
            ExposePrintOutputs& out,
            PrintPipelineScratch& scratch);

        static float compute_midgray_factor(
            const WorkingState& ws,
            const Print::Runtime& printRuntime,
            const Print::Params& printParams,
            const Couplers::Runtime& dirRT,
            float exposureCompScale);
    };

    struct DevelopPrintInputs {
        const Print::Runtime* printRuntime = nullptr;
        PrintLogRaw printLogRaw;
    };

    struct DevelopPrintOutputs {
        PrintDensityCMY printDensity;
    };

    class DevelopPrintStage {
    public:
        static bool run(const DevelopPrintInputs& in, DevelopPrintOutputs& out);
    };

    // Shared density-pipeline orchestration seam used by the CPU render path and spatial DIR helpers.
    struct PipelineRunnerConfig {
        bool enablePrint = true;
    };

    struct DensityPixelInputs {
        RgbLinear rgb;
        float exposureScale = 1.0f;

        const Couplers::Runtime* dirRuntime = nullptr;
        bool applyDirRuntime = true;

        bool useFilmRawOverride = false;
        FilmRaw filmRawOverride;

        bool useSpatialDIR = false;
        float spatialLogECorrectionsYMC[3] = { 0.0f, 0.0f, 0.0f };

        const Print::Runtime* printRuntime = nullptr;
        const Print::Params* printParams = nullptr;
        float midgrayFactor = 1.0f;
        PrintPipelineScratch* printScratch = nullptr;
    };

    struct DensityPixelOutputs {
        FilmRaw filmRaw;
        FilmLogRaw filmLogRaw;
        NegativeDensityCMY negativeDensity;
        PrintDensityCMY printDensity;
        DensityMedium medium = DensityMedium::Negative;
    };

    class PipelineRunner {
    public:
        explicit PipelineRunner(const PipelineRunnerConfig& cfg);

        bool run_density_pixel(const WorkingState& ws, const DensityPixelInputs& in, DensityPixelOutputs& out) const;

        static float compute_midgray_factor(
            const WorkingState& ws,
            const Print::Runtime& printRuntime,
            const Print::Params& printParams,
            const Couplers::Runtime& dirRT,
            float exposureCompScale);

    private:
        PipelineRunnerConfig cfg_;
    };

} // namespace Pipeline
