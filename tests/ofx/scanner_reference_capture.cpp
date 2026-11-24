#include "scanner_reference_capture.h"

#include "../ffi/juicer_test_api.h"
#include "GamutCompression.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }

    template <typename T>
    nlohmann::json stage_words(const ofx_test::ScanReferenceCapture::Samples& samples, T (ofx_test::ScanReferenceCapture::Sample::*field)[3]) {
        using Word = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
        auto words = nlohmann::json::array();
        words.get_ref<nlohmann::json::array_t&>().reserve(samples.size() * 3U);
        for (const auto& sample : samples)
            for (const T value : sample.*field) {
                require(std::isfinite(value), "scan observation is missing a finite stage value");
                words.push_back(std::bit_cast<Word>(value));
            }
        return words;
    }
} // namespace

namespace ofx_test {
    ScanReferenceCapture::ScanReferenceCapture(cudaStream_t stream, bool capturePrint) : stream_(stream) {
        CUcontext streamContext = nullptr;
        require(cuCtxGetCurrent(&context_) == CUDA_SUCCESS && context_ && cuStreamGetCtx(stream_, &streamContext) == CUDA_SUCCESS &&
                    streamContext == context_,
                "scan observation requires the active frame context");
        check_cuda(cudaMalloc(&device_, sizeof(Samples)), "allocate scan observation");
        if (capturePrint) {
            const auto allocation = cudaMalloc(&printCyanDevice_, sizeof(float));
            if (allocation != cudaSuccess) {
                (void)cudaFree(device_);
                check_cuda(allocation, "allocate Print cyan observation");
            }
        }
        const auto status = cudaMemsetAsync(device_, 255, sizeof(Samples), stream_);
        const auto cyanStatus = capturePrint ? cudaMemsetAsync(printCyanDevice_, 255, sizeof(float), stream_) : cudaSuccess;
        if (status != cudaSuccess || cyanStatus != cudaSuccess) {
            (void)cudaFree(device_);
            (void)cudaFree(printCyanDevice_);
            check_cuda(status != cudaSuccess ? status : cyanStatus, "clear scan observations");
        }
        previous_ = JuicerCuda::ScanDensityCaptureTest::swap_density_capture({device_, kPixels, 7, 5, &glare_, printCyanDevice_});
    }

    ScanReferenceCapture::~ScanReferenceCapture() {
        JuicerCuda::ScanDensityCaptureTest::swap_density_capture(previous_);
        if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
            (void)cudaStreamSynchronize(stream_);
            (void)cudaFree(device_);
            (void)cudaFree(printCyanDevice_);
            CUcontext restored = nullptr;
            (void)cuCtxPopCurrent(&restored);
        }
    }

    ScanReferenceCapture::Samples ScanReferenceCapture::read() const {
        CUcontext current = nullptr;
        require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == context_, "scan observation context changed");
        Samples samples{};
        check_cuda(cudaMemcpyAsync(samples.data(), device_, sizeof(samples), cudaMemcpyDeviceToHost, stream_), "read scan observation");
        check_cuda(cudaStreamSynchronize(stream_), "complete scan observation");
        return samples;
    }

    float ScanReferenceCapture::print_cyan() const {
        CUcontext current = nullptr;
        require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == context_, "print observation context changed");
        require(printCyanDevice_ != nullptr, "Print cyan observation was not requested");
        float value = 0.0f;
        check_cuda(cudaMemcpyAsync(&value, printCyanDevice_, sizeof(value), cudaMemcpyDeviceToHost, stream_), "read Print cyan observation");
        check_cuda(cudaStreamSynchronize(stream_), "complete Print cyan observation");
        require(std::isfinite(value), "Print cyan observation did not reach the selected sample");
        return value;
    }

    const JuicerCuda::ScanDensityCaptureTest::GlareLaunch& ScanReferenceCapture::glare_launch() const noexcept {
        return glare_;
    }

    nlohmann::json scan_float_words(std::span<const float> values) {
        auto words = nlohmann::json::array();
        words.get_ref<nlohmann::json::array_t&>().reserve(values.size());
        for (const float value : values)
            words.push_back(std::bit_cast<std::uint32_t>(value));
        return words;
    }

    std::array<float, ScanReferenceCapture::kPixels> scan_glare_field(const ScanReferenceCapture::Samples& samples) {
        std::array<float, ScanReferenceCapture::kPixels> field{};
        std::ranges::transform(samples, field.begin(), &ScanReferenceCapture::Sample::glarePercent);
        return field;
    }

    nlohmann::json scan_reference_stages(const ScanReferenceCapture::Samples& samples) {
        using Sample = ScanReferenceCapture::Sample;
        return {{"developed_cmy_bits", stage_words(samples, &Sample::density)},
                {"glare_field_bits", scan_float_words(scan_glare_field(samples))},
                {"device_xyz_bits", stage_words(samples, &Sample::xyz)},
                {"device_corrected_xyz_bits", stage_words(samples, &Sample::correctedXYZ)},
                {"device_adapted_xyz_bits", stage_words(samples, &Sample::adaptedXYZ)},
                {"device_signed_rgb_bits", stage_words(samples, &Sample::signedRGB)}};
    }

    nlohmann::json scan_reference_arithmetic(const ScanReferenceCapture::Samples& samples) {
        auto pixels = nlohmann::json::array();
        pixels.get_ref<nlohmann::json::array_t&>().reserve(samples.size());
        for (const auto& sample : samples) {
            auto steps = nlohmann::json::array();
            steps.get_ref<nlohmann::json::array_t&>().reserve(std::size(sample.wavelengths));
            for (const auto& step : sample.wavelengths) {
                require(step.activeWeights <= 7, "scan arithmetic did not reach a wavelength sample");
                if (step.activeWeights == 0) {
                    steps.push_back(nullptr);
                    continue;
                }
                steps.push_back({step.activeWeights, std::bit_cast<std::uint64_t>(step.density), std::bit_cast<std::uint32_t>(step.exponent), std::bit_cast<std::uint32_t>(step.transmission), scan_float_words(step.accumulated)});
            }
            pixels.push_back({{"steps", std::move(steps)}, {"scaled_xyz_bits", scan_float_words(sample.scaledXYZ)}});
        }
        return pixels;
    }

    nlohmann::json scan_reference_inputs(const FjPreparedStateView& state) {
        const auto lane = [](const FjFloatSpan& values) {
            require(values.data && values.count == 81, "missing held scanner spectrum");
            return scan_float_words({values.data, values.count});
        };
        const auto& scanner = state.scanner_tables;
        const auto& color = state.output_color;
        const auto& correction = state.scanner_correction;
        const auto& encoding = color.encoding;
        const auto& boundary = color.gamut_cmax;
        const std::array coefficients{encoding.gamma, encoding.a, encoding.b, encoding.c, encoding.d, encoding.linear_cutoff};
        const bool gamut = (color.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0;
        require(gamut ? boundary.data && boundary.count == Gamut::kOutputBoundaryValueCount : !boundary.data && boundary.count == 0,
                "incomplete held scanner gamut boundary");
        nlohmann::json input{
            {"dye_bits", {lane(scanner.dyes_cmy[0]), lane(scanner.dyes_cmy[1]), lane(scanner.dyes_cmy[2])}},
            {"weight_bits", {lane(scanner.weighted_xyz[0]), lane(scanner.weighted_xyz[1]), lane(scanner.weighted_xyz[2])}},
            {"base_bits", scanner.has_baseline ? lane(scanner.baseline_min) : nlohmann::json::array()},
            {"inverse_y_bits", std::bit_cast<std::uint32_t>(scanner.inv_yn)},
            {"white_bits", scan_float_words(color.illuminant_xyz)},
            {"cat02_bits", scan_float_words(color.cat02)},
            {"xyz_to_rgb_bits", scan_float_words(color.xyz_to_rgb)},
            {"output_color_space", color.color_space},
            {"output_color_flags", color.flags},
            {"cctf_kind", encoding.cctf_kind},
            {"cctf_bits", scan_float_words(coefficients)},
            {"correction_active", correction.active},
            {"slope_bits", std::bit_cast<std::uint32_t>(correction.xyz_slope)},
            {"offset_bits", std::bit_cast<std::uint32_t>(correction.xyz_offset)},
            {"recipe_hash", state.recipe_hash},
            {"scanner_hash", state.scanner_hash},
            {"print_glare_hash", state.print.glare_hash}};
        if (gamut) {
            input.update({{"gamut_hash", color.gamut_recipe_hash},
                          {"gamut_table_hash", color.gamut_table_hash},
                          {"gamut_cmax_bits", scan_float_words({boundary.data, boundary.count})},
                          {"gamut_lightness_bits", scan_float_words(color.lightness_knee)},
                          {"gamut_chroma_bits", scan_float_words(color.chroma_knee)},
                          {"native_to_d65_bits", scan_float_words(color.native_rgb_to_d65_xyz)},
                          {"d65_to_native_bits", scan_float_words(color.d65_xyz_to_native_rgb)},
                          {"oklab_matrices_bits", {scan_float_words(Gamut::kOklabXyzToLms), scan_float_words(Gamut::kOklabLmsToXyz), scan_float_words(Gamut::kOklabLmsRootToLab), scan_float_words(Gamut::kOklabLabToLmsRoot)}}});
        }
        return input;
    }
} // namespace ofx_test
