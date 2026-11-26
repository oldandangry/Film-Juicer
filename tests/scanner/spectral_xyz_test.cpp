#include "spectral_xyz.h"

#include <array>
#include <cmath>
#include <limits>
#include <span>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "Cuda/Scan/JuicerCudaScannerBinding.h"

namespace {
    class SpectralScan : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&output, sizeof(*output)), cudaSuccess);
        }

        void TearDown() override {
            if (output) {
                EXPECT_EQ(cudaFree(output), cudaSuccess);
            }
            if (stream) {
                EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
            }
        }

        void compare(const JuicerCuda::ScannerSpectra& spectra, std::array<float, 3> density, bool success, const std::array<double, 3>& expected) {
            ASSERT_EQ(JuicerTests::launch_spectral_sample(spectra, density, output, stream), cudaSuccess);
            JuicerTests::SpectralSample actual{};
            ASSERT_EQ(cudaMemcpyAsync(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost, stream), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            EXPECT_EQ(actual.success, success);
            for (std::size_t channel = 0; channel < expected.size(); ++channel) {
                EXPECT_NEAR(actual.xyz[channel], expected[channel], 1e-10 + 0.01 * std::abs(expected[channel]));
            }
        }

        cudaStream_t stream = nullptr;
        JuicerTests::SpectralSample* output = nullptr;
    };

    TEST_F(SpectralScan, PreGlareXyzKeepsAllThreeDoublesAndSuccess) {
        std::array<float, 81> cyan{}, magenta{}, yellow{}, base{}, x{}, y{}, z{};
        cyan[0] = magenta[1] = yellow[2] = 1.0F;
        base[1] = 0.25F;
        x[0] = 0.25F;
        y[1] = 0.50F;
        z[2] = 0.75F;
        constexpr float inverseY = 2.0F;
        const auto spectra = JuicerCuda::bind_scanner_spectra({cyan, magenta, yellow, x, y, z, base, inverseY});
        constexpr std::array rows{
            std::array{0.0F, 0.0F, 0.0F},
            std::array{0.75F, 0.75F, 0.75F},
            std::array{0.04F, 0.12F, 0.19F},
            std::array{-0.15F, 0.08F, 0.41F},
            std::array{1.81F, 0.57F, 0.20F},
            std::array{1.2F, 2.2F, 1.7F},
            std::array{3.0F, 3.0F, 3.0F}};
        for (const auto& density : rows) {
            compare(spectra, density, true, {0.5 * std::pow(10.0, -static_cast<double>(density[0])) + 1e-10, std::pow(10.0, -static_cast<double>(density[1]) - 0.25) + 1e-10, 1.5 * std::pow(10.0, -static_cast<double>(density[2])) + 1e-10});
        }
        compare(spectra, {std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F}, false, {});
    }
} // namespace
