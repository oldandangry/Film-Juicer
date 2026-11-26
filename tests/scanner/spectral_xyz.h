#pragma once

#include <array>

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaPayloads.h"

namespace JuicerTests {
    struct SpectralSample {
        bool success;
        double xyz[3];
    };

    cudaError_t launch_spectral_sample(
        const JuicerCuda::ScannerSpectra& spectra,
        std::array<float, 3> density,
        SpectralSample* output,
        cudaStream_t stream);
} // namespace JuicerTests
