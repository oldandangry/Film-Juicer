#pragma once

#if defined(JUICER_SCAN_DENSITY_CAPTURE_TEST)

#include <cstddef>
#include <cstdint>

namespace JuicerCuda::ScanDensityCaptureTest {

    struct ScanSample {
        float density[3];
        float glarePercent;
        double xyz[3];
        double correctedXYZ[3];
        double adaptedXYZ[3];
        double signedRGB[3];
    };

    struct GlareLaunch {
        std::uint64_t seed = 0;
        int originX = 0;
        int originY = 0;
        int medium = 0;
        int blurRadius = 0;
        unsigned count = 0;
    };

    // Host-owned device storage with one row-major sample per pixel.
    struct DensityCapture {
        ScanSample* samples = nullptr;
        std::size_t capacity = 0;
        int width = 0;
        int height = 0;
        GlareLaunch* glare = nullptr;
        float* printCyan = nullptr;
    };

    // Installs one observation for the calling host thread and returns the prior
    // descriptor. The caller owns allocation, stream ordering, and context checks.
    DensityCapture swap_density_capture(DensityCapture capture) noexcept;

} // namespace JuicerCuda::ScanDensityCaptureTest

#endif
