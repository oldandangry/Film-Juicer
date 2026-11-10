#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "instance_host.h"

namespace ofx_test {
    inline constexpr int kProductionPitchPadding = 5;
    inline constexpr float kProductionSourceCanary = -7.0f;
    struct Case {
        const char* id = nullptr;
        const char* routeName = nullptr;
        const char* route = nullptr;
        int reconstruction = 0;
        int components = 0;
        int width = 7;
        int height = 5;
        int grain = 0;
        double time = 37.0;
    };
    struct ImageBuffers {
        cudaStream_t stream = nullptr;
        float* source = nullptr;
        float* destination = nullptr;
    };
    std::vector<float> initial_source(const Case& test);
    bool canaries_preserved(const std::vector<float>& initial, const std::vector<float>& actual, const Case& test);
    bool alpha_preserved(const std::vector<float>& source, const std::vector<float>& output, const Case& test);
    void author_parameters(InstanceHost& host, OfxImageEffectHandle effect, const Case& test);
    void configure(InstanceHost& host, OfxImageEffectHandle effect, const Case& test, const ImageBuffers& device);
} // namespace ofx_test
