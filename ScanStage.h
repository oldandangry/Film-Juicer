#pragma once

#include "PipelineTypes.h"

struct WorkingState;

namespace Scanner {
    struct ScannerMediumRuntime;
}

namespace Pipeline {

    struct ScanInputs {
        DensityMedium medium = DensityMedium::Negative;
        NegativeDensityCMY negativeDensity;
        PrintDensityCMY printDensity;
    };

    struct ScanOutputs {
        LogXyz logXyz;
        Xyz xyz;
    };

    class ScanStage {
    public:
        static bool run(const WorkingState& ws, const ScanInputs& in, ScanOutputs& out);

        // Canonical scanner spectral core (agx-emulsion parity):
        // - Accepts normalized density (0..1), denormalizes per medium range
        // - Converts dyes -> XYZ under the medium's spectral tables
        // - Applies log10(xyz + 1e-10) without clamping
        static void spectral_to_log_xyz(const Scanner::ScannerMediumRuntime& medium, const double D_norm[3], double logXYZ[3]);

        // Canonical normalization for LUT coordinates (mirrors agx _normalize_* semantics).
        static void normalize_density(const Scanner::ScannerMediumRuntime& medium, const float D_cmy[3], double D_norm[3]);
    };

} // namespace Pipeline
