#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "nlohmann/json.hpp"
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cuda/JuicerCudaFailure.h"
#include "gtest/gtest.h"

#include "ColorTransforms.h"
#include "CudaRenderProjection.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "output_encoding_oracle.h"

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(const JuicerCuda::DirectPipelineRunParams*, void*);
extern "C" cudaError_t juicer_cuda_print_focused_pipeline(const JuicerCuda::PrintPipelineRunParams*, void*);
extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_density_rgb(const JuicerCuda::DirectPipelineRunParams*, const float*, const float*, const float*, float*, float*, float*, const float*, void*);
extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_density_rgb(const JuicerCuda::PrintPipelineRunParams*, const float*, const float*, const float*, float*, float*, float*, float*, float*, int, int, std::uint64_t, float, float, const float*, int, void*);
extern "C" cudaError_t juicer_cuda_direct_focused_scanner_post_output(const JuicerCuda::DirectPipelineRunParams*, float*, float*, float*, float*, const float*, int, const float*, int, float, const JuicerCuda::FilmDefectsPayload*, const JuicerCuda::GateWeavePayload*, const float*, int, int, void*);
extern "C" cudaError_t juicer_cuda_print_focused_scanner_post_output(const JuicerCuda::PrintPipelineRunParams*, float*, float*, float*, float*, const float*, int, const float*, int, float, const JuicerCuda::FilmDefectsPayload*, const JuicerCuda::GateWeavePayload*, const float*, int, int, void*);

namespace {
    constexpr double kMaximumError = 1.0e-6;
    constexpr double kMeanError = 1.0e-7;
    constexpr std::uint32_t kPadding = 0xa5a5a5a5u;

    void require(bool ok, const std::string& message) {
        if (!ok) {
            throw std::runtime_error(message);
        }
    }
    void check_cuda(cudaError_t status) {
        require(status == cudaSuccess, cudaGetErrorString(status));
    }

    class Buffer {
    public:
        explicit Buffer(std::size_t count) : count_(count) {
            check_cuda(cudaMalloc(&data_, count * sizeof(float)));
        }
        ~Buffer() {
            (void)cudaFree(data_);
        }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        float* get() const {
            return data_;
        }
        void upload(const std::vector<float>& values) {
            require(values.size() == count_, "Upload extent mismatch");
            check_cuda(cudaMemcpy(data_, values.data(), count_ * sizeof(float), cudaMemcpyHostToDevice));
        }
        std::vector<float> download() const {
            std::vector<float> values(count_);
            check_cuda(cudaMemcpy(values.data(), data_, count_ * sizeof(float), cudaMemcpyDeviceToHost));
            return values;
        }

    private:
        float* data_ = nullptr;
        std::size_t count_;
    };

    JuicerCuda::OutputEncodingPayload encoding(const OutputEncoding::Params& settings) {
        JuicerCuda::OutputEncodingPayload result;
        result.outputColorSpaceIndex = OutputEncoding::toIndex(settings.colorSpace);
        result.applyCctfEncoding = settings.applyCctfEncoding ? 1 : 0;
        result.inputIsOutputSpace = settings.inputIsOutputSpace ? 1 : 0;
        FjOutputColorPreparation prepared{};
        require(fj_test_output_color(static_cast<std::uint32_t>(settings.colorSpace), &prepared).category == FJ_STATUS_SUCCESS, "Complete output-color fixture preparation");
        const auto& c = prepared.encoding;
        result.cctf = {static_cast<int>(c.cctf_kind), c.gamma, c.a, c.b, c.c, c.d, c.linear_cutoff};
        std::copy_n(c.dwg_to_output, 9, result.dwgToOutput);
        return result;
    }

    template <typename Params>
    void launch_pipeline(const Params& params) {
        if constexpr (std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>) {
            check_cuda(juicer_cuda_negative_direct_pipeline(&params, nullptr));
        } else {
            check_cuda(juicer_cuda_print_focused_pipeline(&params, nullptr));
        }
    }

    template <typename Params>
    void launch_density_scan(
        const Params& params,
        const float* densityC,
        const float* densityM,
        const float* densityY,
        float* rgbR,
        float* rgbG,
        float* rgbB) {
        if constexpr (std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>) {
            check_cuda(juicer_cuda_direct_focused_scan_linear_density_rgb(
                &params,
                densityC,
                densityM,
                densityY,
                rgbR,
                rgbG,
                rgbB,
                nullptr,
                nullptr));
        } else {
            check_cuda(juicer_cuda_print_focused_scan_linear_density_rgb(
                &params,
                densityC,
                densityM,
                densityY,
                rgbR,
                rgbG,
                rgbB,
                nullptr,
                nullptr,
                0,
                0,
                0,
                0.0f,
                0.0f,
                nullptr,
                0,
                nullptr));
        }
    }

    int read_device_flag(const int* flag) {
        int result = 0;
        check_cuda(cudaMemcpy(&result, flag, sizeof(result), cudaMemcpyDeviceToHost));
        return result;
    }

    void clear_device_flag(int* flag) {
        check_cuda(cudaMemset(flag, 0, sizeof(*flag)));
    }

    std::vector<std::uint32_t> bit_patterns(const std::vector<float>& values) {
        std::vector<std::uint32_t> result(values.size());
        std::transform(
            values.begin(),
            values.end(),
            result.begin(),
            [](float value) {
                return std::bit_cast<std::uint32_t>(value);
            });
        return result;
    }

    struct PostEffects {
        float* temporary = nullptr;
        const float* kernel = nullptr;
        JuicerCuda::FilmDefectsPayload defects;
        JuicerCuda::GateWeavePayload weave;
        const float* mask = nullptr;
    };

    template <typename Params>
    void launch_post(const Params& params, float* planes, const PostEffects& effects = {}) {
        const std::size_t count = static_cast<std::size_t>(params.width) * params.height;
        const auto launch = [&]() {
            if constexpr (std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>) {
                return juicer_cuda_direct_focused_scanner_post_output;
            } else {
                return juicer_cuda_print_focused_scanner_post_output;
            }
        }();
        check_cuda(launch(&params, planes, planes + count, planes + 2 * count, effects.temporary, effects.kernel, effects.kernel ? 1 : 0, effects.kernel, effects.kernel ? 1 : 0, 0.7f, &effects.defects, &effects.weave, effects.mask, effects.mask ? 1 : 0, effects.mask ? 1 : 0, nullptr));
    }

    struct Image {
        int width;
        int height;
        int components;
        std::size_t count;
        std::size_t stride;
        std::vector<float> source;
        Buffer input;
        Buffer output;
        Buffer planes;

        Image(std::array<int, 2> extent, int channels)
            : width(extent[0]), height(extent[1]), components(channels),
              count(static_cast<std::size_t>(width) * height),
              stride(static_cast<std::size_t>(width) * components + 7),
              source(stride * height, std::bit_cast<float>(kPadding)),
              input(source.size()), output(source.size()), planes(3 * count) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t base = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * components;
                    for (int channel = 0; channel < 3; ++channel) {
                        source[base + channel] = static_cast<float>((x * 17 + y * 31 + channel * 89) % 997) / 500.0f;
                    }
                    if (components == 4) {
                        source[base + 3] = x % 17 == 0 ? std::bit_cast<float>(0x7fc01234u) : 0.37f;
                    }
                }
            }
            input.upload(source);
            clear_output();
        }
        void clear_output() const {
            check_cuda(cudaMemset(output.get(), 0xa5, source.size() * sizeof(float)));
        }
        template <typename Params>
        Params params() const {
            Params p;
            p.width = width;
            p.height = height;
            p.nComponents = components;
            p.src = input.get();
            p.dst = output.get();
            p.srcRowBytes = p.dstRowBytes = stride * sizeof(float);
            return p;
        }
        void compare(const std::vector<float>& actual, const std::vector<float>& reference, bool exact) const {
            double maximum = 0.0, sum = 0.0;
            std::size_t finite = 0;
            for (int y = 0; y < height; ++y) {
                for (std::size_t x = 0; x < stride; ++x) {
                    const std::size_t i = static_cast<std::size_t>(y) * stride + x;
                    if (x >= static_cast<std::size_t>(width) * components || (components == 4 && x % 4 == 3)) {
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(source[i])) << i;
                    } else if (!std::isfinite(reference[i])) {
                        ASSERT_TRUE((std::isnan(actual[i]) && std::isnan(reference[i])) || actual[i] == reference[i]) << i;
                    } else {
                        ASSERT_TRUE(std::isfinite(actual[i])) << i;
                        const double error = std::abs(static_cast<double>(actual[i]) - reference[i]);
                        maximum = std::max(maximum, error);
                        sum += error;
                        ++finite;
                        if (exact) {
                            ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(reference[i])) << i;
                        }
                    }
                }
            }
            EXPECT_LE(maximum, kMaximumError);
            ASSERT_GT(finite, 0u);
            EXPECT_LE(sum / static_cast<double>(finite), kMeanError);
        }
        std::vector<float> oracle(const std::vector<float>& linear, const OutputEncoding::Params& settings) const {
            std::vector<float> result = source;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t i = static_cast<std::size_t>(y) * width + x;
                    double rgb[3] = {linear[i], linear[count + i], linear[2 * count + i]};
                    ScannerOracle::apply_encoding(settings, rgb);
                    const std::size_t base = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * components;
                    for (int c = 0; c < 3; ++c) {
                        result[base + c] = static_cast<float>(rgb[c]);
                    }
                }
            }
            return result;
        }
    };

    std::vector<float> transfer_inputs(std::size_t count) {
        std::vector<float> special{-0.0f, 0.0f, 1.0f, -1.0f, 65504.0f, -65504.0f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
        for (float boundary : {0.0031308f, 0.01805397f, 0.001953125f, 0.00262409f, 1.0f}) {
            special.push_back(boundary);
            special.push_back(std::nextafter(boundary, 0.0f));
            special.push_back(std::nextafter(boundary, 2.0f));
        }
        std::vector<float> result(3 * count);
        for (std::size_t i = 0; i < result.size(); ++i) {
            const float unit = static_cast<float>((i * 7919u) % 65536u) / 65536.0f;
            result[i] = i % 4 == 0 ? std::exp2(-150.0f + unit * 166.0f) : unit * 2.2f - 0.2f;
            if (i % count < special.size()) {
                result[i] = special[i % count];
            }
        }
        // Cancellation-sensitive matrix case that rejects narrowing the matrix itself.
        result[100] = 0.7565f;
        result[count + 100] = 1.7618f;
        result[2 * count + 100] = 0.3812f;
        return result;
    }

    template <typename Params>
    void check_transfers(int components) {
        Image image({257, 33}, components);
        const auto values = transfer_inputs(image.count);
        image.planes.upload(values);
        auto p = image.params<Params>();
        for (int space = 0; space < static_cast<int>(OutputEncoding::ColorSpace::Count); ++space) {
            for (bool cctf : {false, true}) {
                for (bool inputIsOutput : {false, true}) {
                    SCOPED_TRACE(testing::Message() << "space=" << space << " cctf=" << cctf << " identity=" << inputIsOutput << " components=" << components);
                    const OutputEncoding::Params settings{OutputEncoding::colorSpaceFromIndex(space), cctf, inputIsOutput};
                    p.scanStage.scanColor.encoding = encoding(settings);
                    image.clear_output();
                    launch_post(p, image.planes.get());
                    image.compare(image.output.download(), image.oracle(values, settings), !cctf && inputIsOutput);
                }
            }
        }
    }

    TEST(ScannerOutput, TransferFunctionsMatchDoubleOracle) {
        for (int components : {3, 4}) {
            check_transfers<JuicerCuda::DirectPipelineRunParams>(components);
            check_transfers<JuicerCuda::PrintPipelineRunParams>(components);
        }
    }

    TEST(ScannerOutput, EncodingFollowsBlurUnsharpWeaveAndGateAttenuation) {
        Image image({129, 67}, 4);
        std::vector<float> values(3 * image.count);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.15f + static_cast<float>(i % 97) / 200.0f;
        }
        Buffer temporary(image.count), kernel(3), mask(1);
        kernel.upload({0.25f, 0.5f, 0.25f});
        mask.upload({0.75f});
        PostEffects effects;
        effects.temporary = temporary.get();
        effects.kernel = kernel.get();
        effects.mask = mask.get();
        effects.weave.active = 1;
        effects.weave.dxPx = 0.35f;
        effects.weave.dyPx = -0.45f;
        effects.weave.cosRot = std::cos(0.001f);
        effects.weave.sinRot = std::sin(0.001f);
        auto p = image.params<JuicerCuda::DirectPipelineRunParams>();
        p.scanStage.scanColor.encoding = encoding({OutputEncoding::ColorSpace::sRGB, false, true});
        image.planes.upload(values);
        launch_post(p, image.planes.get(), effects);
        const auto linear = image.output.download();
        // The bounded pattern must stay away from clipping, so CCTF-off readback
        // is a valid observation of the unchanged spatial stages' output.
        std::vector<float> observed(3 * image.count);
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const float value = linear[static_cast<std::size_t>(y) * image.stride + static_cast<std::size_t>(x) * 4 + c];
                    ASSERT_GT(value, 0.01f);
                    ASSERT_LT(value, 0.99f);
                    observed[static_cast<std::size_t>(c) * image.count + static_cast<std::size_t>(y) * image.width + x] = value;
                }
            }
        }
        for (int space = 0; space < static_cast<int>(OutputEncoding::ColorSpace::Count); ++space) {
            const OutputEncoding::Params settings{OutputEncoding::colorSpaceFromIndex(space), true, true};
            p.scanStage.scanColor.encoding = encoding(settings);
            image.planes.upload(values);
            image.clear_output();
            launch_post(p, image.planes.get(), effects);
            image.compare(image.output.download(), image.oracle(observed, settings), false);
        }
    }

    struct RouteInputs {
        FocusedRenderStateBuildProduct product;
        Scanner::ScannerSpectralLutDescriptor scanner;
        static ParamSnapshot controls_for_route(Spektrafilm::ScanRoute route) {
            ParamSnapshot controls;
            controls.scanRoute = route;
            return controls;
        }
        RouteInputs(Spektrafilm::ScanRoute route, int space, bool gamut)
            : RouteInputs(controls_for_route(route), space, gamut) {}
        RouteInputs(ParamSnapshot controls, int space, bool gamut) {
            const Spektrafilm::ScanRoute route = controls.scanRoute;
            controls.grainControls.active = false;
            controls.dirCouplers.active = false;
            controls.cameraAutoExposureEnabled = 0;
            controls.outputColorSpace = space;
            controls.outputGamutCompressionEnabled = gamut ? 1 : 0;
            if (Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Positive) {
                controls.filmProfileKey = "fujifilm_provia_100f";
            }
            const bool print = Spektrafilm::scan_route_is_print(route);
            std::string error;
            require(print ? build_print_render_state_product(controls, product, error) : build_direct_render_state_product(controls, product, error), error);
            require(print ? Scanner::build_print_scanner_spectral_lut_descriptor({&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput}, scanner, error)
                          : Scanner::build_direct_scanner_spectral_lut_descriptor({&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput}, scanner, error),
                    error);
        }
        JuicerProcess::Root::CudaFramePreparationRequest request(const Image& image) const {
            JuicerProcess::Root::CudaFramePreparationRequest r;
            r.recipe = &product.recipe;
            r.exposureTables = &product.payload.exposureTables;
            r.filmRawConfig = &product.payload.filmRawConfig;
            r.filmTcLut = product.payload.filmTcLut ? &*product.payload.filmTcLut : nullptr;
            r.printMainIlluminant = product.payload.printMainIlluminant ? &*product.payload.printMainIlluminant : nullptr;
            r.scannerTables = &product.payload.scannerTables;
            r.scannerColor = &product.payload.scannerColor;
            r.scannerLutDescriptor = &scanner;
            r.outputBoundaryTable = product.payload.outputBoundaryTable.get();
            r.requestedWidth = image.width;
            r.requestedHeight = image.height;
            return r;
        }
    };

    void bind_scanner(JuicerCuda::ScanStagePayload& scan, const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared, const OutputGamutRecipe& gamut) {
        const auto& range = *prepared.scanRange;
        scan.densityRange.mediumIsNegative = range.mediumIsNegative;
        std::copy_n(range.min_cmy, 3, scan.densityRange.min_cmy);
        std::copy_n(range.inv_max_cmy, 3, scan.densityRange.inv_max_cmy);
        const auto& lut = *prepared.scanLut;
        scan.scanLutLog2PchipXYZ = lut.log2PchipXYZ;
        scan.scanLutPchipSlopeC = lut.slopeC;
        scan.scanLutPchipSlopeM = lut.slopeM;
        scan.scanLutPchipSlopeY = lut.slopeY;
        scan.scanLutPchipCellMin = lut.cellMin;
        scan.scanLutPchipCellMax = lut.cellMax;
        scan.scanLutRes = static_cast<int>(lut.res);
        auto& color = scan.scanColor;
        std::copy_n(prepared.scannerColor->cat02, 9, color.cat02);
        std::copy_n(prepared.scannerColor->xyzToRgb, 9, color.xyzToRgb);
        std::copy_n(prepared.scannerColor->illuminantXYZ, 3, color.illuminantXYZ);
        color.encoding = encoding(prepared.scannerColor->encoding);
        if (gamut.enabled) {
            require(prepared.outputGamutTransform && prepared.outputGamutCmax, "Missing gamut resources");
            color.outputGamutActive = 1;
            color.outputGamutCmax = prepared.outputGamutCmax;
            std::copy_n(prepared.outputGamutTransform->nativeRgbToD65Xyz.data(), 9, color.outputGamutNativeRgbToD65Xyz);
            std::copy_n(prepared.outputGamutTransform->d65XyzToNativeRgb.data(), 9, color.outputGamutD65XyzToNativeRgb);
            std::copy_n(Gamut::kOklabXyzToLms.data(), 9, color.outputGamutOklabXyzToLms);
            std::copy_n(Gamut::kOklabLmsToXyz.data(), 9, color.outputGamutOklabLmsToXyz);
            std::copy_n(Gamut::kOklabLmsRootToLab.data(), 9, color.outputGamutOklabLmsRootToLab);
            std::copy_n(Gamut::kOklabLabToLmsRoot.data(), 9, color.outputGamutOklabLabToLmsRoot);
            color.outputGamutLightnessKnee[0] = gamut.lightnessKneeThreshold;
            color.outputGamutLightnessKnee[1] = gamut.lightnessKneeLimit;
            color.outputGamutLightnessKnee[2] = gamut.lightnessKneePower;
            color.outputGamutChromaKnee[0] = gamut.chromaKneeThreshold;
            color.outputGamutChromaKnee[1] = gamut.chromaKneeLimit;
            color.outputGamutChromaKnee[2] = gamut.chromaKneePower;
        }
    }

    template <typename Params>
    Params make_bound_params(
        const RouteInputs& inputs,
        JuicerProcess::Root::PreparedCudaFrame& frame,
        const Image& image) {
        const auto prepared = frame.focused_resources();
        require(prepared.active, "Focused scanner resources are not active");
        const auto& recipe = inputs.product.recipe;
        std::string error;
        JuicerCuda::FilmPayloadPack film;
        require(
            JuicerCuda::pack_film_payloads(
                recipe.filmRaw,
                recipe.filmDevelop,
                recipe.dirCouplers,
                recipe.densityBounds,
                prepared.film,
                nullptr,
                1.0f,
                film,
                error),
            error);
        Params params = image.params<Params>();
        params.filmRaw = film.filmRaw;
        params.filmExpose = film.filmExposure;
        params.filmDevelop = film.filmDevelop;
        if constexpr (std::is_same_v<Params, JuicerCuda::PrintPipelineRunParams>) {
            JuicerCuda::PrintCudaPayloadPack print;
            require(
                JuicerCuda::pack_print_cuda_payloads(
                    recipe.print,
                    frame.print_resources(),
                    1.0f,
                    print,
                    error),
                error);
            params.printExpose = print.expose;
            params.printDevelop = print.develop;
        }
        bind_scanner(
            params.scanStage,
            prepared,
            recipe.scannerOutput.outputGamut);
        return params;
    }

    struct LutCapture {
        std::uint32_t resolution = 0;
        std::uint64_t hash = 0;
        std::array<std::uintptr_t, 6> addresses{};
        std::vector<std::uint32_t> content;
        std::size_t bytes = 0;
    };

    LutCapture capture_lut(const JuicerCuda::Resources::DeviceSpectralLut& lut) {
        require(lut.canonical_ready(), "Scanner LUT is not ready");
        LutCapture result;
        result.resolution = lut.res;
        result.hash = lut.hash;
        result.addresses = {
            reinterpret_cast<std::uintptr_t>(lut.log2PchipXYZ),
            reinterpret_cast<std::uintptr_t>(lut.slopeC),
            reinterpret_cast<std::uintptr_t>(lut.slopeM),
            reinterpret_cast<std::uintptr_t>(lut.slopeY),
            reinterpret_cast<std::uintptr_t>(lut.cellMin),
            reinterpret_cast<std::uintptr_t>(lut.cellMax)};
        const std::size_t voxelValues =
            static_cast<std::size_t>(lut.res) * lut.res * lut.res * 3u;
        const std::size_t cellResolution = static_cast<std::size_t>(lut.res - 1u);
        const std::size_t cellValues =
            cellResolution * cellResolution * cellResolution * 3u;
        result.bytes = (4u * voxelValues + 2u * cellValues) * sizeof(float);
        result.content.reserve(4u * voxelValues + 2u * cellValues);
        auto append = [&](const float* source, std::size_t count) {
            std::vector<float> values(count);
            check_cuda(cudaMemcpy(
                values.data(),
                source,
                count * sizeof(float),
                cudaMemcpyDeviceToHost));
            const std::vector<std::uint32_t> patterns = bit_patterns(values);
            result.content.insert(
                result.content.end(),
                patterns.begin(),
                patterns.end());
        };
        append(lut.log2PchipXYZ, voxelValues);
        append(lut.slopeC, voxelValues);
        append(lut.slopeM, voxelValues);
        append(lut.slopeY, voxelValues);
        append(lut.cellMin, cellValues);
        append(lut.cellMax, cellValues);
        return result;
    }

    class ScannerRoutes : public testing::Test {
    protected:
        void SetUp() override {
            check_cuda(cudaSetDevice(0));
            check_cuda(cudaFree(nullptr));
            CUcontext context = nullptr;
            require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context, "Missing CUDA context");
            key = {0, context};
        }
        void TearDown() override {
            check_cuda(cudaDeviceSynchronize());
            std::string error;
            require(JuicerProcess::root().retire_idle_context(0, key.contextOpaque, error), error);
        }
        JuicerProcess::Root::PreparedCudaFrame prepare_route(
            const RouteInputs& inputs,
            const Image& image) {
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x5343414e4e4552ull;
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.product.payload.uploadCoreHash,
                inputs.product.recipe.dirCouplers.hash,
                inputs.product.payload.scannerHash,
                0);
            JuicerCuda::Failure error;
            auto frame = JuicerProcess::root().prepare_cuda_frame(
                key,
                snapshot,
                inputs.request(image),
                {},
                nullptr,
                error);
            require(frame.active(), error.diagnostic);
            return frame;
        }
        template <typename Params>
        void check_route(const RouteInputs& inputs) {
            Image image({65, 33}, 4);
            JuicerCuda::Failure error;
            auto frame = prepare_route(inputs, image);
            auto p = make_bound_params<Params>(inputs, frame, image);
            require(frame.prepare_scan_error_stage(p.scanStage.scanErrorFlag, nullptr, error) == JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready, error.diagnostic);
            auto staged = p;
            staged.scanStage.linearRgbR = image.planes.get();
            staged.scanStage.linearRgbG = image.planes.get() + image.count;
            staged.scanStage.linearRgbB = image.planes.get() + 2 * image.count;
            launch_pipeline(staged);
            const auto linear = image.planes.download();
            for (float v : linear) {
                ASSERT_TRUE(std::isfinite(v));
            }
            for (bool cctf : {false, true}) {
                const OutputEncoding::Params settings{inputs.product.payload.scannerColor.encoding.colorSpace, cctf, true};
                p.scanStage.scanColor.encoding = encoding(settings);
                image.clear_output();
                launch_pipeline(p);
                const auto fused = image.output.download();
                image.clear_output();
                launch_post(p, image.planes.get());
                const auto separate = image.output.download();
                const auto reference = image.oracle(linear, settings);
                image.compare(separate, reference, !cctf);
                image.compare(fused, reference, false);
                image.compare(fused, separate, cctf);
            }
            EXPECT_EQ(read_device_flag(p.scanStage.scanErrorFlag), 0);
            require(frame.finalize_scan_error_stage(p.scanStage.scanErrorFlag, nullptr, error), error.diagnostic);
            require(frame.finish(nullptr, error), error.diagnostic);
        }
        void expect_fused_failure_output(const Image& image) {
            const std::vector<float> actual = image.output.download();
            for (int y = 0; y < image.height; ++y) {
                for (std::size_t x = 0; x < image.stride; ++x) {
                    const std::size_t index =
                        static_cast<std::size_t>(y) * image.stride + x;
                    const bool rgb =
                        x < static_cast<std::size_t>(image.width) *
                                image.components &&
                        x % static_cast<std::size_t>(image.components) < 3u;
                    if (rgb) {
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[index]), 0u)
                            << index;
                    } else {
                        EXPECT_EQ(
                            std::bit_cast<std::uint32_t>(actual[index]),
                            std::bit_cast<std::uint32_t>(image.source[index]))
                            << index;
                    }
                }
            }
        }
        template <typename Params>
        void check_required_lut_failures(const RouteInputs& inputs) {
            Image image({17, 9}, 4);
            auto frame = prepare_route(inputs, image);
            auto params = make_bound_params<Params>(inputs, frame, image);
            JuicerCuda::Failure error;
            require(
                frame.prepare_scan_error_stage(
                    params.scanStage.scanErrorFlag,
                    nullptr,
                    error) == JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready,
                error.diagnostic);

            Buffer densities(3u * image.count);
            Buffer scanOutput(3u * image.count);
            densities.upload(std::vector<float>(3u * image.count, 0.25f));
            const std::array<const char*, 7> cases = {
                "log2_xyz",
                "slope_c",
                "slope_m",
                "slope_y",
                "cell_min",
                "cell_max",
                "resolution"};
            for (std::size_t failure = 0; failure < cases.size(); ++failure) {
                SCOPED_TRACE(cases[failure]);
                auto broken = params;
                switch (failure) {
                    case 0:
                        broken.scanStage.scanLutLog2PchipXYZ = nullptr;
                        break;
                    case 1:
                        broken.scanStage.scanLutPchipSlopeC = nullptr;
                        break;
                    case 2:
                        broken.scanStage.scanLutPchipSlopeM = nullptr;
                        break;
                    case 3:
                        broken.scanStage.scanLutPchipSlopeY = nullptr;
                        break;
                    case 4:
                        broken.scanStage.scanLutPchipCellMin = nullptr;
                        break;
                    case 5:
                        broken.scanStage.scanLutPchipCellMax = nullptr;
                        break;
                    case 6:
                        broken.scanStage.scanLutRes = 1;
                        break;
                    default:
                        throw std::logic_error("Unhandled LUT failure case");
                }

                clear_device_flag(broken.scanStage.scanErrorFlag);
                image.clear_output();
                launch_pipeline(broken);
                EXPECT_EQ(read_device_flag(broken.scanStage.scanErrorFlag), 1);
                expect_fused_failure_output(image);

                clear_device_flag(broken.scanStage.scanErrorFlag);
                scanOutput.upload(std::vector<float>(3u * image.count, 1.0f));
                launch_density_scan(
                    broken,
                    densities.get(),
                    densities.get() + image.count,
                    densities.get() + 2u * image.count,
                    scanOutput.get(),
                    scanOutput.get() + image.count,
                    scanOutput.get() + 2u * image.count);
                EXPECT_EQ(read_device_flag(broken.scanStage.scanErrorFlag), 1);
                for (float value : scanOutput.download()) {
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0u);
                }
            }

            clear_device_flag(params.scanStage.scanErrorFlag);
            require(
                frame.finalize_scan_error_stage(
                    params.scanStage.scanErrorFlag,
                    nullptr,
                    error),
                error.diagnostic);
            require(frame.finish(nullptr, error), error.diagnostic);
        }
        struct RouteCapture {
            LutCapture lut;
            std::vector<std::uint32_t> output;
        };
        template <typename Params>
        RouteCapture capture_route(const RouteInputs& inputs) {
            Image image({31, 19}, 4);
            auto frame = prepare_route(inputs, image);
            auto params = make_bound_params<Params>(inputs, frame, image);
            JuicerCuda::Failure error;
            require(
                frame.prepare_scan_error_stage(
                    params.scanStage.scanErrorFlag,
                    nullptr,
                    error) == JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready,
                error.diagnostic);
            launch_pipeline(params);
            RouteCapture result;
            result.output = bit_patterns(image.output.download());
            EXPECT_EQ(read_device_flag(params.scanStage.scanErrorFlag), 0);
            result.lut = capture_lut(*frame.focused_resources().scanLut);
            require(
                frame.finalize_scan_error_stage(
                    params.scanStage.scanErrorFlag,
                    nullptr,
                    error),
                error.diagnostic);
            require(frame.finish(nullptr, error), error.diagnostic);
            return result;
        }
        struct CorrectionCapture {
            LutCapture lut;
            Scanner::ScannerColorCorrectionDescriptor correction;
            std::array<std::uint64_t, 7> printHashes{};
            std::array<std::uintptr_t, 15> printAddresses{};
            std::array<std::uint32_t, 4> printValues{};
        };
        CorrectionCapture correction_resources(const RouteInputs& inputs, const Image& image, bool derive = true) {
            auto frame = prepare_route(inputs, image);
            CorrectionCapture result;
            result.lut = capture_lut(*frame.focused_resources().scanLut);
            const auto& recipe = inputs.product.recipe;
            std::string diagnostic;
            if (Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute)) {
                const auto print = frame.print_resources();
                require(print.active && print.mainIlluminantHost, "prepared print resources remain available for correction");
                result.printHashes = {print.filmDensityTablesHash, print.profileTablesHash, print.mainIlluminantHash, print.preflashIlluminantHash, print.preflashRawHash, print.balanceHash, print.preparationHash};
                std::size_t index = 0;
                const auto address = [&](const float* value) {
                    result.printAddresses[index++] = reinterpret_cast<std::uintptr_t>(value);
                };
                for (const auto& curve : {print.printSensC, print.printSensM, print.printSensY, print.printDcC, print.printDcM, print.printDcY}) {
                    address(curve.x);
                    address(curve.y);
                }
                address(print.mainIlluminant);
                address(print.preflashIlluminant);
                address(print.filmDensityTables.epsC);
                result.printValues = {std::bit_cast<std::uint32_t>(print.preflashRawCmy[0]), std::bit_cast<std::uint32_t>(print.preflashRawCmy[1]), std::bit_cast<std::uint32_t>(print.preflashRawCmy[2]), std::bit_cast<std::uint32_t>(print.normalizer)};
                if (derive)
                    require(Scanner::build_print_scanner_color_correction_descriptor({&recipe, &inputs.product.payload.scannerTables, print.mainIlluminantHost, print.spectralSampleCount, print.preflashRawCmy, print.normalizer}, result.correction, diagnostic), diagnostic);
            } else if (derive) {
                require(Scanner::build_direct_scanner_color_correction_descriptor(recipe, inputs.product.payload.scannerTables, result.correction, diagnostic), diagnostic);
            }
            JuicerCuda::Failure failure;
            require(frame.finish(nullptr, failure), failure.diagnostic);
            return result;
        }
        FjStatus execute_correction(const RouteInputs& inputs, Image& image, bool projection, std::string& diagnostic) {
            std::cout << "scanner correction caller route=" << static_cast<int>(inputs.product.recipe.profileRoute.scanRoute)
                      << " projection=" << projection << " correction=" << inputs.product.recipe.scannerOutput.blackCorrection
                      << "/" << inputs.product.recipe.scannerOutput.whiteCorrection << '\n';
            image.clear_output();
            auto preparation = JuicerProcess::root().begin_frame_preparation();
            require(preparation.active(), "scanner actual-caller frame preparation admission");
            const auto& recipe = inputs.product.recipe;
            const auto& payload = inputs.product.payload;
            const JuicerCuda::FrameRect bounds{0, 0, image.width, image.height};
            const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(bounds, bounds, recipe.filmRaw.autoExposureMethod);
            const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
            const std::optional<ScatterHalationFrameDescriptor> halation;
            const Spektrafilm::FilmJuicerEffectsGeometry geometry{};
            const auto rowBytes = static_cast<std::ptrdiff_t>(image.stride * sizeof(float));
            const JuicerCuda::ExecutionFrame frame{bounds, bounds, bounds, reinterpret_cast<const unsigned char*>(image.input.get()), reinterpret_cast<const unsigned char*>(image.input.get()), reinterpret_cast<unsigned char*>(image.output.get()), rowBytes, rowBytes, image.components, nullptr, diffusion, halation, geometry, 100.0f, 0.0, 24.0, 0x534334, 0x5343414e, meter, false, false};
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x5343414e4e4552ull;
            snapshot.frameToken.value = 0; // floor(frame.timeFrames), independent of snapshot identity.
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(payload.uploadCoreHash, recipe.dirCouplers.hash, payload.scannerHash, meter.hash);
            JuicerCuda::PendingContextLossRecovery recovery;
            FjStatus status{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            diagnostic.clear();
            if (projection) {
                FjFrame raw{};
                raw.source = {reinterpret_cast<std::uintptr_t>(image.input.get()), {0, 0, image.width, image.height}, rowBytes, static_cast<std::uint32_t>(image.components), FJ_DEPTH_FLOAT32};
                raw.destination = raw.source;
                raw.destination.address = reinterpret_cast<std::uintptr_t>(image.output.get());
                raw.render_window = raw.full_frame_extent = raw.source.bounds;
                raw.flags = FJ_FRAME_STREAM_PRESENT;
                JuicerCuda::NativeCall call(JuicerCuda::borrowed_owner());
                FjCudaContext context{};
                require(call.inspect(&raw, &context, nullptr).category == FJ_STATUS_SUCCESS && context.device_id == key.deviceId && reinterpret_cast<void*>(context.context) == key.contextOpaque, "scanner projection exact context");
                const auto outcome = JuicerCuda::project_and_render(call, recipe, payload, frame, raw, snapshot, recovery, {}, nullptr, {}, diagnostic);
                status = outcome.status;
                EXPECT_EQ(outcome.flags, 0u);
            } else {
                try {
                    if (Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute))
                        JuicerCuda::execute_print({recipe, payload, frame, snapshot}, recovery);
                    else
                        JuicerCuda::execute_direct({recipe, payload, frame, snapshot}, recovery);
                } catch (const JuicerCuda::ExecutionFailure& failure) {
                    status = failure.failure.status;
                    diagnostic = failure.failure.diagnostic;
                }
            }
            EXPECT_FALSE(recovery.pending);
            check_cuda(cudaDeviceSynchronize());
            return status;
        }
        void check_correction_output(const Image& image, const std::vector<float>& output) {
            for (int y = 0; y < image.height; ++y)
                for (int x = 0; x < image.width; ++x) {
                    const auto index = static_cast<std::size_t>(y) * image.stride + static_cast<std::size_t>(x) * image.components;
                    for (int channel = 0; channel < 3; ++channel) {
                        EXPECT_TRUE(std::isfinite(output[index + channel]));
                        EXPECT_GE(output[index + channel], 0.0f);
                        EXPECT_LE(output[index + channel], 1.0f);
                    }
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(output[index + 3]), std::bit_cast<std::uint32_t>(image.source[index + 3]));
                }
        }
        JuicerCuda::ResourceManager::DeviceContextKey key;
        inline static std::uint64_t nextIdentity = 1;
    };

    TEST_F(ScannerRoutes, CorrectionRoutes) {
        for (const auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route));
            auto controls = RouteInputs::controls_for_route(route);
            controls.glareActive = false;
            controls.scannerLutResolution = 17;
            RouteInputs off(controls, 0, true);
            controls.scannerBlackCorrection = controls.scannerWhiteCorrection = 1;
            RouteInputs held(controls, 0, true);
            const auto heldHash = held.product.recipe.hash;
            const auto heldScannerHash = held.product.recipe.scannerOutput.hash;
            Image image({31, 19}, 4);
            const auto before = correction_resources(off, image);
            const auto active = correction_resources(held, image);
            EXPECT_EQ(before.lut.hash, active.lut.hash);
            EXPECT_EQ(before.lut.addresses, active.lut.addresses);
            EXPECT_EQ(before.lut.content, active.lut.content);
            EXPECT_EQ(before.printHashes, active.printHashes);
            EXPECT_EQ(before.printAddresses, active.printAddresses);
            EXPECT_EQ(before.printValues, active.printValues);
            const bool correctionActive = route != Spektrafilm::ScanRoute::NegativeDirectScan;
            EXPECT_EQ(active.correction.active, correctionActive);
            EXPECT_EQ(active.correction.hash != 0, correctionActive);
            EXPECT_NE(off.product.recipe.hash, heldHash);
            EXPECT_NE(off.product.recipe.scannerOutput.hash, heldScannerHash);
            std::string diagnostic;
            require(execute_correction(off, image, false, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            const auto offWords = bit_patterns(image.output.download());
            require(execute_correction(held, image, false, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            const auto legacy = image.output.download();
            check_correction_output(image, legacy);
            require(execute_correction(held, image, true, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            EXPECT_EQ(bit_patterns(legacy), bit_patterns(image.output.download()));
            require(execute_correction(held, image, false, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            EXPECT_EQ(bit_patterns(legacy), bit_patterns(image.output.download()));
            if (correctionActive)
                EXPECT_NE(offWords, bit_patterns(legacy));
            else
                EXPECT_EQ(offWords, bit_patterns(legacy));

            if (correctionActive) {
                controls.scannerBlackLevel = controls.scannerWhiteLevel = 0.0;
                RouteInputs failed(controls, 0, true);
                for (bool projection : {false, true}) {
                    const auto status = execute_correction(failed, image, projection, diagnostic);
                    EXPECT_EQ(status.category, FJ_STATUS_PREPARATION_FAILURE);
                    EXPECT_EQ(status.api, FJ_API_NONE);
                    EXPECT_EQ(status.native_code, 0);
                    EXPECT_EQ(diagnostic, Spektrafilm::scan_route_is_print(route) ? "MalformedRequiredProfileData phase=8B print corrected midgray" : "MalformedRequiredProfileData phase=8B direct corrected midgray");
                    for (const auto word : bit_patterns(image.output.download()))
                        EXPECT_EQ(word, kPadding); // No output submission after the computed failure.
                }
                const auto operation = Spektrafilm::scan_route_is_print(route) ? 1u : 0u;
                require(fj_test_scanner_correction_fault(operation, 1).category == FJ_STATUS_SUCCESS, "arm independent scanner constructor fault");
                (void)correction_resources(held, image, false);
                EXPECT_EQ(execute_correction(held, image, false, diagnostic).category, FJ_STATUS_INTERNAL_FAILURE);
                EXPECT_EQ(diagnostic, "scanner correction contained panic");
                for (const auto word : bit_patterns(image.output.download()))
                    EXPECT_EQ(word, kPadding);
            }
            require(execute_correction(held, image, true, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            EXPECT_EQ(bit_patterns(legacy), bit_patterns(image.output.download()));
            const auto restored = correction_resources(held, image);
            EXPECT_EQ(restored.correction.hash, active.correction.hash);
            EXPECT_EQ(restored.lut.addresses, active.lut.addresses);
            EXPECT_EQ(restored.lut.content, active.lut.content);
            EXPECT_EQ(restored.printHashes, active.printHashes);
            EXPECT_EQ(restored.printAddresses, active.printAddresses);
            EXPECT_EQ(restored.printValues, active.printValues);
            EXPECT_EQ(held.product.recipe.hash, heldHash);
            EXPECT_EQ(held.product.recipe.scannerOutput.hash, heldScannerHash);
        }
    }

    TEST_F(ScannerRoutes, GamutSwitchKeepsPreparedAndLegacyRenderingEquivalent) {
        for (const auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            auto controls = RouteInputs::controls_for_route(route);
            controls.glareActive = false;
            controls.scannerLutResolution = 17;
            controls.scannerBlackCorrection = controls.scannerWhiteCorrection = 1;
            RouteInputs disabled(controls, 0, false);
            RouteInputs enabled(controls, 0, true);
            const auto disabledHash = disabled.product.recipe.scannerOutput.outputGamut.hash;
            const auto enabledHash = enabled.product.recipe.scannerOutput.outputGamut.hash;
            ASSERT_NE(disabledHash, 0u);
            ASSERT_NE(enabledHash, 0u);
            EXPECT_NE(disabledHash, enabledHash);

            Image image({31, 19}, 4);
            const auto initial = correction_resources(disabled, image);
            std::array<std::vector<std::uint32_t>, 2> expected;
            for (const auto* inputs : {&disabled, &enabled, &disabled, &enabled}) {
                const auto& gamut = inputs->product.recipe.scannerOutput.outputGamut;
                SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route) << " gamut=" << gamut.enabled);
                std::cout << "prepared gamut route=" << static_cast<int>(route) << " enabled=" << gamut.enabled << '\n';
                {
                    auto frame = prepare_route(*inputs, image);
                    const auto view = frame.focused_resources();
                    ASSERT_TRUE(view.active);
                    ASSERT_NE(view.scannerColor, nullptr);
                    EXPECT_EQ(view.scannerColor->outputGamutRecipeHash, gamut.enabled ? gamut.hash : 0);
                    EXPECT_EQ(view.outputGamutTransform != nullptr, gamut.enabled);
                    EXPECT_EQ(view.outputGamutCmax != nullptr, gamut.enabled);
                    JuicerCuda::Failure failure;
                    require(frame.finish(nullptr, failure), failure.diagnostic);
                }
                std::string diagnostic;
                require(execute_correction(*inputs, image, false, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                const auto legacy = image.output.download();
                check_correction_output(image, legacy);
                require(execute_correction(*inputs, image, true, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                const auto projected = bit_patterns(image.output.download());
                EXPECT_EQ(projected, bit_patterns(legacy));
                auto& previous = expected[gamut.enabled ? 1u : 0u];
                if (previous.empty()) {
                    previous = projected;
                } else {
                    EXPECT_EQ(projected, previous);
                }
                const auto reused = correction_resources(*inputs, image);
                EXPECT_EQ(reused.correction.hash, initial.correction.hash);
                EXPECT_EQ(reused.lut.hash, initial.lut.hash);
                EXPECT_EQ(reused.lut.addresses, initial.lut.addresses);
                EXPECT_EQ(reused.lut.content, initial.lut.content);
                EXPECT_EQ(reused.printHashes, initial.printHashes);
                EXPECT_EQ(reused.printAddresses, initial.printAddresses);
                EXPECT_EQ(reused.printValues, initial.printValues);
            }
            EXPECT_EQ(disabled.product.recipe.scannerOutput.outputGamut.hash, disabledHash);
            EXPECT_EQ(enabled.product.recipe.scannerOutput.outputGamut.hash, enabledHash);
        }
    }

    TEST_F(ScannerRoutes, FusedAndSeparateOutputMatchOnAllFourRoutes) {
        for (auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            for (bool gamut : {false, true}) {
                for (int space : {0, 7}) {
                    SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route) << " space=" << space << " gamut=" << gamut);
                    RouteInputs inputs(route, space, gamut);
                    if (Spektrafilm::scan_route_is_print(route)) {
                        check_route<JuicerCuda::PrintPipelineRunParams>(inputs);
                    } else {
                        check_route<JuicerCuda::DirectPipelineRunParams>(inputs);
                    }
                }
            }
        }
    }

    TEST_F(ScannerRoutes, RequiredLutFailuresAreReportedByBothScannerCallers) {
        RouteInputs direct(
            Spektrafilm::ScanRoute::NegativeDirectScan,
            0,
            false);
        check_required_lut_failures<JuicerCuda::DirectPipelineRunParams>(direct);

        RouteInputs print(
            Spektrafilm::ScanRoute::PositivePrintScan,
            0,
            false);
        check_required_lut_failures<JuicerCuda::PrintPipelineRunParams>(print);
    }

    TEST_F(ScannerRoutes, DeferredDirFailureComesFromReadbackBits) {
        for (const auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            RouteInputs inputs(route, 0, false);
            Image image({17, 9}, 4);
            // Product readback contract: bit 0 is scanner RGB; bit 8 is DIR,
            // with the failed DIR channels in bits 9-11.
            for (const int bits : {0, 1, (1 << 8) | (2 << 9), 1 | (1 << 8) | (7 << 9)}) {
                SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route) << " bits=" << bits);
                JuicerCuda::Failure error;
                auto submitted = prepare_route(inputs, image);
                int* flag = nullptr;
                ASSERT_EQ(submitted.prepare_scan_error_stage(flag, nullptr, error), JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready);
                check_cuda(cudaMemcpy(flag, &bits, sizeof(bits), cudaMemcpyHostToDevice));
                require(submitted.finalize_scan_error_stage(flag, nullptr, error), error.diagnostic);
                require(submitted.finish(nullptr, error), error.diagnostic);
                check_cuda(cudaDeviceSynchronize());

                auto next = prepare_route(inputs, image);
                const auto result = next.prepare_scan_error_stage(flag, nullptr, error);
                if (bits == 0) {
                    EXPECT_EQ(result, JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready);
                    EXPECT_NE(flag, nullptr);
                    require(next.finalize_scan_error_stage(flag, nullptr, error), error.diagnostic);
                    require(next.finish(nullptr, error), error.diagnostic);
                } else {
                    EXPECT_EQ(result, bits == 1 ? JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed : JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::DeferredDirFailure);
                    EXPECT_EQ(flag, nullptr);
                    EXPECT_EQ(error.status.category, FJ_STATUS_PREPARATION_FAILURE);
                    EXPECT_EQ(error.status.api, FJ_API_NONE);
                    EXPECT_EQ(error.status.native_code, 0);
                    next.abort();
                }
                // Inactive-frame failure cannot retain the prior DIR disposition.
                EXPECT_EQ(next.prepare_scan_error_stage(flag, nullptr, error), JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed);
                EXPECT_EQ(flag, nullptr);
            }
        }
    }

    TEST_F(ScannerRoutes, ResolutionTransitionRestoresLutContentAndOutput) {
        for (const auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route));
            const bool print = Spektrafilm::scan_route_is_print(route);
            ParamSnapshot controls17 = RouteInputs::controls_for_route(route);
            controls17.scannerLutResolution = 17;
            ParamSnapshot controls33 = controls17;
            controls33.scannerLutResolution = 33;
            RouteInputs inputs17(controls17, 0, false);
            RouteInputs inputs33(controls33, 0, false);

            ASSERT_EQ(inputs17.scanner.lutResolution, 17u);
            ASSERT_EQ(inputs33.scanner.lutResolution, 33u);
            ASSERT_NE(inputs17.scanner.hash, 0u);
            ASSERT_NE(inputs33.scanner.hash, 0u);
            EXPECT_NE(inputs17.scanner.hash, inputs33.scanner.hash);

            const RouteCapture initial =
                (print ? capture_route<JuicerCuda::PrintPipelineRunParams>(inputs17) : capture_route<JuicerCuda::DirectPipelineRunParams>(inputs17));
            const RouteCapture retained =
                (print ? capture_route<JuicerCuda::PrintPipelineRunParams>(inputs17) : capture_route<JuicerCuda::DirectPipelineRunParams>(inputs17));
            const RouteCapture changed =
                (print ? capture_route<JuicerCuda::PrintPipelineRunParams>(inputs33) : capture_route<JuicerCuda::DirectPipelineRunParams>(inputs33));
            const RouteCapture restored =
                (print ? capture_route<JuicerCuda::PrintPipelineRunParams>(inputs17) : capture_route<JuicerCuda::DirectPipelineRunParams>(inputs17));

            EXPECT_EQ(initial.lut.resolution, 17u);
            EXPECT_EQ(changed.lut.resolution, 33u);
            EXPECT_EQ(restored.lut.resolution, 17u);
            EXPECT_EQ(initial.lut.hash, inputs17.scanner.hash);
            EXPECT_EQ(changed.lut.hash, inputs33.scanner.hash);
            EXPECT_EQ(restored.lut.hash, inputs17.scanner.hash);
            EXPECT_EQ(initial.lut.addresses, retained.lut.addresses);
            EXPECT_EQ(initial.lut.content, retained.lut.content);
            EXPECT_GT(changed.lut.bytes, initial.lut.bytes);
            EXPECT_EQ(initial.lut.bytes, restored.lut.bytes);
            EXPECT_EQ(initial.lut.content, restored.lut.content);
            EXPECT_EQ(initial.output, retained.output);
            EXPECT_EQ(initial.output, restored.output);
        }
    }

    TEST_F(ScannerRoutes, PrintResourceCacheAcceptsPublishedValuesAcrossDescriptorChanges) {
        ParamSnapshot defaultControls =
            RouteInputs::controls_for_route(Spektrafilm::ScanRoute::NegativePrintScan);
        ParamSnapshot gammaControls = defaultControls;
        gammaControls.printGammaFactor = 1.1;
        RouteInputs retainedDefault(defaultControls, 0, false);
        RouteInputs retainedGamma(gammaControls, 0, false);

        auto check_print = [&](const char* scenario, ParamSnapshot controls) {
            SCOPED_TRACE(scenario);
            RouteInputs inputs(std::move(controls), 0, false);
            check_route<JuicerCuda::PrintPipelineRunParams>(inputs);
        };

        {
            SCOPED_TRACE("retained default publication");
            check_route<JuicerCuda::PrintPipelineRunParams>(retainedDefault);
            check_route<JuicerCuda::PrintPipelineRunParams>(retainedDefault);
        }
        {
            SCOPED_TRACE("retained gamma publication");
            check_route<JuicerCuda::PrintPipelineRunParams>(retainedGamma);
        }

        ParamSnapshot illuminantControls = defaultControls;
        illuminantControls.enlIll = 0;
        check_print("illuminant change", illuminantControls);

        ParamSnapshot filterControls = defaultControls;
        filterControls.printUiYmcCc = {4.0, -3.0, 2.0};
        check_print("main filter change", filterControls);

        ParamSnapshot preflashControls = defaultControls;
        preflashControls.printPreflashExposure = 0.1;
        preflashControls.preflashMFilterCc = 3.0;
        preflashControls.preflashYFilterCc = -2.0;
        check_print("preflash enabled", preflashControls);
        check_print("preflash disabled", defaultControls);

        ParamSnapshot unnormalizedControls = defaultControls;
        unnormalizedControls.normalizePrintExposure = 0;
        check_print("normalization disabled", unnormalizedControls);
        check_print("normalization enabled", defaultControls);

        ParamSnapshot profileControls = defaultControls;
        profileControls.printProfileKey = "kodak_supra_endura";
        check_print("print profile change", profileControls);
    }
    TEST_F(ScannerRoutes, LutConstructionFailuresPreserveCategoryAndCache) {
        for (const bool projection : {false, true}) {
            SCOPED_TRACE(testing::Message() << "projection=" << projection);
            for (const auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
                SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route));
                auto controls = RouteInputs::controls_for_route(route);
                controls.scannerLutResolution = 17;
                RouteInputs inputs(controls, 0, false);
                Image image({31, 19}, 4);
                std::string diagnostic;
                const auto reject = [&](const RouteInputs& selected, std::uint32_t category, const char* message) {
                    const auto result = execute_correction(selected, image, projection, diagnostic);
                    EXPECT_EQ(result.category, category);
                    EXPECT_EQ(result.api, FJ_API_NONE);
                    EXPECT_EQ(result.native_code, 0);
                    EXPECT_EQ(diagnostic, std::string(message) + " route=" + Spektrafilm::scan_route_metadata(route).key);
                    EXPECT_EQ(fj_test_scanner_lut_live_owners(), 0u);
                    for (const auto word : bit_patterns(image.output.download()))
                        EXPECT_EQ(word, kPadding);
                };
                for (unsigned buffer = 1; buffer <= 10; ++buffer) {
                    SCOPED_TRACE(testing::Message() << "buffer=" << buffer);
                    ASSERT_EQ(fj_test_scanner_lut_fault(0, buffer + 1).category, FJ_STATUS_SUCCESS);
                    reject(inputs, FJ_STATUS_ALLOCATION_FAILURE, "scanner LUT host allocation failed");
                }
                require(execute_correction(inputs, image, projection, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                const auto original = bit_patterns(image.output.download());
                ASSERT_EQ(fj_test_scanner_lut_fault(0, 2).category, FJ_STATUS_SUCCESS);
                require(execute_correction(inputs, image, projection, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                EXPECT_EQ(bit_patterns(image.output.download()), original);
                controls.scannerLutResolution = 33;
                RouteInputs replacement(controls, 0, false);
                // A cache hit must leave the armed constructor fault for this miss.
                reject(replacement, FJ_STATUS_ALLOCATION_FAILURE, "scanner LUT host allocation failed");
                for (unsigned operation : {0u, 1u}) {
                    ASSERT_EQ(fj_test_scanner_lut_fault(operation, 1).category, FJ_STATUS_SUCCESS);
                    reject(replacement, FJ_STATUS_INTERNAL_FAILURE, "scanner LUT preparation panicked");
                }
                require(execute_correction(inputs, image, projection, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                EXPECT_EQ(bit_patterns(image.output.download()), original);
                require(execute_correction(replacement, image, projection, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                require(execute_correction(inputs, image, projection, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                EXPECT_EQ(bit_patterns(image.output.download()), original);
                EXPECT_EQ(fj_test_scanner_lut_live_owners(), 0u);
            }
        }
    }

    TEST_F(ScannerRoutes, LutIdentityTracksSelectedProfileBoundsAndIlluminant) {
        for (const auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route));
            auto controls = RouteInputs::controls_for_route(route);
            controls.filmProfileKey = "kodak_portra_400";
            controls.printProfileKey = "kodak_portra_endura";
            controls.scannerLutResolution = 17;
            RouteInputs original(controls, 0, false);
            const bool print = Spektrafilm::scan_route_is_print(route);
            if (print)
                controls.printProfileKey = "kodak_2383";
            else
                controls.filmProfileKey = "kodak_portra_800";
            RouteInputs changed(controls, 0, false);
            EXPECT_NE(original.scanner.channelDensityHash, changed.scanner.channelDensityHash);
            EXPECT_NE(original.scanner.densityBoundsHash, changed.scanner.densityBoundsHash);
            if (print) {
                EXPECT_NE(original.scanner.scanIlluminantHash, changed.scanner.scanIlluminantHash);
            }
            EXPECT_NE(original.scanner.hash, changed.scanner.hash);
            const auto before = print ? capture_route<JuicerCuda::PrintPipelineRunParams>(original) : capture_route<JuicerCuda::DirectPipelineRunParams>(original);
            const auto after = print ? capture_route<JuicerCuda::PrintPipelineRunParams>(changed) : capture_route<JuicerCuda::DirectPipelineRunParams>(changed);
            const auto restored = print ? capture_route<JuicerCuda::PrintPipelineRunParams>(original) : capture_route<JuicerCuda::DirectPipelineRunParams>(original);
            EXPECT_EQ(before.lut.hash, original.scanner.hash);
            EXPECT_EQ(after.lut.hash, changed.scanner.hash);
            EXPECT_NE(before.lut.content, after.lut.content);
            EXPECT_EQ(restored.lut.hash, before.lut.hash);
            EXPECT_EQ(restored.lut.content, before.lut.content);
            EXPECT_EQ(restored.output, before.output);
        }
    }

    TEST_F(ScannerRoutes, WholeNativeLutProductsSurviveActualPreparedUpload) {
#if defined(_WIN32)
        const std::string platform = "windows-clang-";
#else
        const std::string platform = "linux-";
#endif
#if defined(NDEBUG)
        const auto preset = platform + "release";
#else
        const auto preset = platform + "debug";
#endif
        const auto directory = std::filesystem::path(__FILE__).parent_path() / "lut-fixtures";
        std::ifstream stream(directory / (preset + ".json"));
        const auto frozen = nlohmann::json::parse(stream);
        for (const auto& row : frozen.at("products")) {
            const auto route = static_cast<Spektrafilm::ScanRoute>(row.at("route").get<unsigned>());
            ParamSnapshot controls;
            controls.scanRoute = route;
            controls.filmProfileKey = row.at("route").get<unsigned>() < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            controls.printProfileKey = "kodak_portra_endura";
            controls.cameraAutoExposureEnabled = 0;
            controls.inputCompressionEnabled = 0;
            controls.outputGamutCompressionEnabled = 0;
            controls.printPreflashExposure = 0.125;
            controls.scannerLutResolution = 17;
            RouteInputs inputs(controls, 0, false);
            std::string diagnostic;
            const bool print = Spektrafilm::scan_route_is_print(route);
            require(print ? build_print_render_state_product(controls, inputs.product, diagnostic)
                          : build_direct_render_state_product(controls, inputs.product, diagnostic),
                    diagnostic);
            const auto& recipe = inputs.product.recipe;
            require(print ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, inputs.scanner, diagnostic)
                          : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, inputs.scanner, diagnostic),
                    diagnostic);
            ASSERT_EQ(inputs.scanner.hash, row.at("descriptor_hash").get<std::uint64_t>());
            const auto capture = print ? capture_route<JuicerCuda::PrintPipelineRunParams>(inputs) : capture_route<JuicerCuda::DirectPipelineRunParams>(inputs);
            std::vector<std::uint32_t> expected;
            for (const auto* name : {"log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max"})
                for (const auto& word : row.at("native").at(name))
                    expected.push_back(word.get<std::uint32_t>());
            EXPECT_EQ(capture.lut.content, expected);
            EXPECT_EQ(capture.lut.bytes, expected.size() * sizeof(float));
        }
    }
} // namespace

int main(int argc, char** argv) {
    JuicerCuda::Owner cudaOwner;
    cudaOwner.create(JuicerProcess::data_directory());
    testing::InitGoogleTest(&argc, argv);
    JuicerProcess::root().ensure_bootstrap();
    const int result = RUN_ALL_TESTS();
    const auto closed = fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
    return closed.category == FJ_STATUS_SUCCESS ? result : 1;
}
