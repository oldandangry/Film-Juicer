// Cuda/JuicerCudaPayloads.h
//
// Phase 3 scaffolding: compact POD payloads for device kernels.
//
// Intentionally avoids CUDA headers so it can be included from both host and NVCC code.
#pragma once

#include <cstdint>

namespace JuicerCuda {

    // Mirrors the subset of Spectral::FilmRawConfig needed by CUDA kernels.
    struct FilmRawPayload {
        int inputColorSpaceIndex = 0;
        int applyCctfDecoding = 0;
        int applyInputChromaticAdapt = 0;
        int spectralUpsamplingMode = 0; // 0=PreferHanatos, 1=ForceMallett (Spectral::SpectralUpsamplingMode)

        float inputRGBToXYZ[9] = {
            1,0,0,
            0,1,0,
            0,0,1
        };
        float inputXYZAdapt[9] = {
            1,0,0,
            0,1,0,
            0,0,1
        };

        float midgrayScale = 1.0f;
        float refIllumWhiteXYZ[3] = { 0.950455f, 1.0f, 1.089058f };
    };

    // Minimal output encoding payload (Phase 3 will flesh this out as kernels land).
    struct OutputEncodingPayload {
        int outputColorSpaceIndex = 0;
        int applyCctfEncoding = 1;
        int preserveLinearRange = 0;
    };

    // Scanner color runtime (CAT + XYZ->RGB) plus output encoding selection.
    struct ScanColorPayload {
        float cat02[9] = { 0.0f };
        float xyzToRgb[9] = { 0.0f };
        float illuminantXYZ[3] = { 0.0f, 0.0f, 0.0f };
        OutputEncodingPayload encoding{};
    };

} // namespace JuicerCuda

