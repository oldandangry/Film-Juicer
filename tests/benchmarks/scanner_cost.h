#pragma once

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaPayloads.h"

namespace JuicerTests {
    struct ScannerCostWavelengths {
        std::uint32_t activeWeights[81]{};
    };

    struct ScannerCostFrame {
        JuicerCuda::ScannerSpectra spectra{};
        JuicerCuda::ScanStagePayload scan{};
        ScannerCostWavelengths wavelengths{};
        float* pixels = nullptr;
        int width = 0;
        int height = 0;
    };

    using ScannerCostLaunch = cudaError_t (*)(const ScannerCostFrame&, cudaStream_t);

    struct ScannerCostVariant {
        ScannerCostLaunch launch;
        bool doubleDot;
        bool checkResults;
        bool preclassified;
    };

    std::array<ScannerCostVariant, 8> scanner_cost_variants();
} // namespace JuicerTests
