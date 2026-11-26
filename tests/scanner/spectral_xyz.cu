#include "spectral_xyz.h"

#include "Cuda/Scan/JuicerCudaSpectralScan.cuh"

namespace {
    __global__ void spectral_sample(
        JuicerCuda::ScannerSpectra spectra,
        float cyan,
        float magenta,
        float yellow,
        JuicerTests::SpectralSample* output) {
        const float density[]{cyan, magenta, yellow};
        JuicerTests::SpectralSample result{};
        result.success = JuicerCuda::spectral_xyz(spectra, density, result.xyz);
        *output = result;
    }
} // namespace

namespace JuicerTests {
    cudaError_t launch_spectral_sample(
        const JuicerCuda::ScannerSpectra& spectra,
        std::array<float, 3> density,
        SpectralSample* output,
        cudaStream_t stream) {
        spectral_sample<<<1, 1, 0, stream>>>(spectra, density[0], density[1], density[2], output);
        return cudaGetLastError();
    }
} // namespace JuicerTests
