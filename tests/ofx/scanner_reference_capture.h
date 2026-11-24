#pragma once

#include "Cuda/Scan/JuicerCudaScanDensityCaptureTest.h"
#include "juicer_cuda_api.h"

#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <span>

extern "C" {
struct OfxImageEffectStruct;
typedef struct FjInstanceObservation FjInstanceObservation;
typedef struct FjPreparedStateView FjPreparedStateView;
FjStatus fj_test_ofx_instance_observe(OfxImageEffectStruct*, FjInstanceObservation**, FjErrorBuffer*);
}

namespace ofx_test {
    class ScanReferenceCapture {
    public:
        static constexpr std::size_t kPixels = 35;
        using Sample = JuicerCuda::ScanDensityCaptureTest::ScanSample;
        using Samples = std::array<Sample, kPixels>;

        explicit ScanReferenceCapture(cudaStream_t stream);
        ~ScanReferenceCapture();
        ScanReferenceCapture(const ScanReferenceCapture&) = delete;
        ScanReferenceCapture& operator=(const ScanReferenceCapture&) = delete;
        Samples read() const;

    private:
        cudaStream_t stream_;
        CUcontext context_ = nullptr;
        Sample* device_ = nullptr;
        JuicerCuda::ScanDensityCaptureTest::DensityCapture previous_{};
    };

    nlohmann::json scan_float_words(std::span<const float> values);
    std::array<float, ScanReferenceCapture::kPixels> scan_glare_field(const ScanReferenceCapture::Samples& samples);
    nlohmann::json scan_reference_stages(const ScanReferenceCapture::Samples& samples);
    nlohmann::json scan_reference_inputs(const FjPreparedStateView& state);
} // namespace ofx_test
