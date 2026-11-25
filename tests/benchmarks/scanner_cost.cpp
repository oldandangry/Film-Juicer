#include "scanner_cost.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

namespace {
    constexpr int kWidth = 4832;
    constexpr int kHeight = 2016;
    constexpr auto kWords = static_cast<std::size_t>(kWidth) * kHeight * 3;
    constexpr auto kBytes = kWords * sizeof(float);
    constexpr std::size_t kWarmups = 4;
    constexpr std::size_t kSamples = 32;

    struct ScannerCostInputs {
        const char* density;
        const char* packet;
    };

    void require_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }

    struct DeviceBuffers {
        cudaStream_t stream = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        float* source = nullptr;
        float* scratch = nullptr;
        int* flag = nullptr;

        ~DeviceBuffers() {
            if (stream)
                (void)cudaStreamSynchronize(stream);
            (void)cudaFree(flag);
            (void)cudaFree(scratch);
            (void)cudaFree(source);
            if (stop)
                (void)cudaEventDestroy(stop);
            if (start)
                (void)cudaEventDestroy(start);
            if (stream)
                (void)cudaStreamDestroy(stream);
        }
    };

    void read_exact(std::ifstream& source, void* destination, std::size_t bytes) {
        if (!source.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes)))
            throw std::runtime_error("short scanner input");
    }

    void require_end(std::ifstream& source) {
        if (source.peek() != std::char_traits<char>::eof() || source.bad())
            throw std::runtime_error("scanner input has unexpected trailing data");
    }

    void restage(const DeviceBuffers& device) {
        require_cuda(cudaMemcpyAsync(device.scratch, device.source, kBytes, cudaMemcpyDeviceToDevice, device.stream),
                     "restage held density");
        require_cuda(cudaMemsetAsync(device.flag, 0, sizeof(int), device.stream), "reset scan error");
    }

    void read_scan(const DeviceBuffers& device, std::vector<float>& output) {
        int flag = -1;
        require_cuda(cudaMemcpyAsync(&flag, device.flag, sizeof(flag), cudaMemcpyDeviceToHost, device.stream), "read scan error");
        require_cuda(cudaMemcpyAsync(output.data(), device.scratch, kBytes, cudaMemcpyDeviceToHost, device.stream), "read RGB words");
        require_cuda(cudaStreamSynchronize(device.stream), "finish scan readback");
        if (flag != 0 || !std::ranges::all_of(output, [](float value) {
                return std::isfinite(value);
            }))
            throw std::runtime_error("held scan produced an invalid result");
    }

    nlohmann::json measure(const ScannerCostInputs& inputs) {
        JuicerTests::ScannerCostFrame frame{};
        frame.width = kWidth;
        frame.height = kHeight;
        std::ifstream packet(inputs.packet, std::ios::binary);
        read_exact(packet, &frame.spectra, sizeof(frame.spectra));
        auto& color = frame.scan.scanColor;
        read_exact(packet, color.cat02, sizeof(color.cat02));
        read_exact(packet, color.xyzToRgb, sizeof(color.xyzToRgb));
        require_end(packet);
        std::ifstream raster(inputs.density, std::ios::binary);
        std::vector<float> held(kWords);
        read_exact(raster, held.data(), kBytes);
        require_end(raster);
        if (!std::ranges::all_of(held, [](float value) {
                return std::isfinite(value);
            }))
            throw std::runtime_error("held density must be finite");

        for (std::size_t index = 0; index < std::size(frame.spectra.wavelengths); ++index) {
            const auto& wave = frame.spectra.wavelengths[index];
            if (std::isnan(wave.dyeC) || std::isnan(wave.dyeM) || std::isnan(wave.dyeY) || std::isnan(wave.baseDensity))
                continue;
            frame.wavelengths.activeWeights[index] = (std::isfinite(wave.weightX) ? 1U : 0U) |
                                                     (std::isfinite(wave.weightY) ? 2U : 0U) |
                                                     (std::isfinite(wave.weightZ) ? 4U : 0U);
        }

        require_cuda(cudaSetDevice(0), "select CUDA device");
        cudaDeviceProp deviceProperties{};
        int driverVersion = 0;
        require_cuda(cudaGetDeviceProperties(&deviceProperties, 0), "read device");
        require_cuda(cudaDriverGetVersion(&driverVersion), "read driver version");
        DeviceBuffers device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create supplied stream");
        require_cuda(cudaEventCreate(&device.start), "create start event");
        require_cuda(cudaEventCreate(&device.stop), "create stop event");
        require_cuda(cudaMalloc(&device.source, kBytes), "allocate held density");
        require_cuda(cudaMalloc(&device.scratch, kBytes), "allocate scan scratch");
        require_cuda(cudaMalloc(&device.flag, sizeof(int)), "allocate scan error flag");
        require_cuda(cudaMemcpyAsync(device.source, held.data(), kBytes, cudaMemcpyHostToDevice, device.stream), "upload held density");
        require_cuda(cudaStreamSynchronize(device.stream), "finish held upload");
        frame.pixels = device.scratch;
        frame.scan.scanErrorFlag = device.flag;

        const auto variants = JuicerTests::scanner_cost_variants();
        std::array<std::vector<float>, 8> samples;
        for (std::size_t round = 0; round < kWarmups + kSamples; ++round) {
            for (std::size_t offset = 0; offset < variants.size(); ++offset) {
                // Rotate the start position and alternate direction between rounds.
                const auto index = (round + (round % 2 == 0 ? offset : variants.size() - offset)) % variants.size();
                restage(device);
                require_cuda(cudaEventRecord(device.start, device.stream), "record scan start");
                require_cuda(variants[index].launch(frame, device.stream), "launch measured scan");
                require_cuda(cudaEventRecord(device.stop, device.stream), "record scan stop");
                require_cuda(cudaEventSynchronize(device.stop), "finish measured scan");
                float milliseconds = 0.0F;
                require_cuda(cudaEventElapsedTime(&milliseconds, device.start, device.stop), "read scan events");
                if (round >= kWarmups)
                    samples[index].push_back(milliseconds);
            }
        }

        std::vector<float> baseline(kWords);
        std::vector<float> output(kWords);
        auto rows = nlohmann::json::array();
        for (std::size_t index = 0; index < variants.size(); ++index) {
            restage(device);
            const auto& variant = variants[index];
            require_cuda(variant.launch(frame, device.stream), "launch output verification");
            read_scan(device, output);
            if (index == 0)
                baseline = output;
            std::size_t changed = 0;
            double maxDifference = 0.0;
            for (std::size_t word = 0; word < output.size(); ++word) {
                changed += output[word] != baseline[word] ? 1U : 0U;
                maxDifference = std::max(maxDifference, std::abs(static_cast<double>(output[word]) - baseline[word]));
            }
            rows.push_back({{"double_dot", variant.doubleDot},
                            {"check_results", variant.checkResults},
                            {"preclassified", variant.preclassified},
                            {"milliseconds", samples[index]},
                            {"changed_rgb_values", changed},
                            {"max_rgb_difference", maxDifference}});
        }
        return {{"schema", "juicer-scanner-cost-v1"},
                {"device", deviceProperties.name},
                {"driver", driverVersion},
                {"width", kWidth},
                {"height", kHeight},
                {"timed_samples", kSamples},
                {"warmups", kWarmups},
                {"variants", std::move(rows)}};
    }
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::runtime_error("expected planar density, scanner packet, and result path");
        const auto result = measure({.density = argv[1], .packet = argv[2]});
        std::ofstream output(argv[3], std::ios::binary);
        if (!(output << result.dump(2) << '\n'))
            throw std::runtime_error("cannot write scanner cost receipt");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
