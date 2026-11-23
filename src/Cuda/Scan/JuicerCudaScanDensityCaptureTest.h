#pragma once

#if defined(JUICER_SCAN_DENSITY_CAPTURE_TEST)

#include <cstddef>

namespace JuicerCuda::ScanDensityCaptureTest {

    struct ScanSample {
        float density[3];
        double xyz[3];
        double correctedXYZ[3];
        double adaptedXYZ[3];
        double signedRGB[3];
    };

    // Host-owned device storage with one row-major sample per pixel.
    struct DensityCapture {
        ScanSample* samples = nullptr;
        std::size_t capacity = 0;
        int width = 0;
        int height = 0;
    };

    // Installs one observation for the calling host thread and returns the prior
    // descriptor. The caller owns allocation, stream ordering, and context checks.
    DensityCapture swap_density_capture(DensityCapture capture) noexcept;

} // namespace JuicerCuda::ScanDensityCaptureTest

#endif
