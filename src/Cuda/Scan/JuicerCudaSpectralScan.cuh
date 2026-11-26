#pragma once

#include <cuda_runtime.h>

#include <cfloat>

#include "Cuda/JuicerCudaPayloads.h"

namespace JuicerCuda {
    __device__ inline bool spectral_xyz(
        const JuicerCuda::ScannerSpectra& spectra,
        const float (&density)[3],
        double (&xyz)[3]) {
        if (!isfinite(density[0]) || !isfinite(density[1]) || !isfinite(density[2])) {
            return false;
        }
        float x = 0.0F;
        float y = 0.0F;
        float z = 0.0F;
        for (const auto& sample : spectra.wavelengths) {
            const bool activeX = isfinite(sample.weightX);
            const bool activeY = isfinite(sample.weightY);
            const bool activeZ = isfinite(sample.weightZ);
            if ((!activeX && !activeY && !activeZ) ||
                isnan(sample.dyeC) || isnan(sample.dyeM) || isnan(sample.dyeY) || isnan(sample.baseDensity)) {
                continue;
            }
            const float spectralDensity = density[0] * sample.dyeC +
                                          density[1] * sample.dyeM +
                                          density[2] * sample.dyeY +
                                          sample.baseDensity;
            if (!isfinite(spectralDensity) || spectralDensity < -FLT_MAX) {
                return false;
            }
            const float transmission = spectralDensity > FLT_MAX ? 0.0F : __exp10f(-spectralDensity);
            if (!isfinite(transmission)) {
                return false;
            }
            if (activeX) {
                x += transmission * sample.weightX;
            }
            if (activeY) {
                y += transmission * sample.weightY;
            }
            if (activeZ) {
                z += transmission * sample.weightZ;
            }
        }
        const float scaledX = x * spectra.inverseYNormalization;
        const float scaledY = y * spectra.inverseYNormalization;
        const float scaledZ = z * spectra.inverseYNormalization;
        if (!isfinite(scaledX) || !isfinite(scaledY) || !isfinite(scaledZ)) {
            return false;
        }
        xyz[0] = static_cast<double>(fmaxf(scaledX, 0.0F)) + 1e-10;
        xyz[1] = static_cast<double>(fmaxf(scaledY, 0.0F)) + 1e-10;
        xyz[2] = static_cast<double>(fmaxf(scaledZ, 0.0F)) + 1e-10;
        return true;
    }
} // namespace JuicerCuda
