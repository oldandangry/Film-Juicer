#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

#include "SpectralProcessing.h"
#include "JuicerState.h"
#include "ProcessRoot.h"

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
        const auto& c = GeneratedColorSpaces::get(settings.colorSpace).cctf;
        result.cctf = {static_cast<int>(c.kind), c.gamma, c.a, c.b, c.c, c.d, c.linearCutoff};
        const auto matrix = OutputEncoding::dwg_to_output_matrix(settings.colorSpace);
        std::copy_n(matrix.m, 9, result.dwgToOutput);
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
                    OutputEncoding::applyEncoding(settings, rgb);
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
            snapshot.contextEpoch = 1;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.product.payload.uploadCoreHash,
                inputs.product.recipe.dirCouplers.hash,
                inputs.product.payload.scannerHash,
                0);
            std::string error;
            auto frame = JuicerProcess::root().prepare_cuda_frame(
                key,
                snapshot,
                inputs.request(image),
                {},
                nullptr,
                error);
            require(frame.active(), error);
            return frame;
        }
        template <typename Params>
        void check_route(const RouteInputs& inputs) {
            Image image({65, 33}, 4);
            std::string error;
            auto frame = prepare_route(inputs, image);
            auto p = make_bound_params<Params>(inputs, frame, image);
            require(frame.prepare_scan_error_stage(p.scanStage.scanErrorFlag, nullptr, error), error);
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
            require(frame.finalize_scan_error_stage(p.scanStage.scanErrorFlag, nullptr, error), error);
            require(frame.finish(nullptr, error), error);
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
            std::string error;
            require(
                frame.prepare_scan_error_stage(
                    params.scanStage.scanErrorFlag,
                    nullptr,
                    error),
                error);

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
                error);
            require(frame.finish(nullptr, error), error);
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
            std::string error;
            require(
                frame.prepare_scan_error_stage(
                    params.scanStage.scanErrorFlag,
                    nullptr,
                    error),
                error);
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
                error);
            require(frame.finish(nullptr, error), error);
            return result;
        }
        JuicerCuda::ResourceManager::DeviceContextKey key;
        inline static std::uint64_t nextIdentity = 1;
    };

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

    TEST_F(ScannerRoutes, ResolutionTransitionRestoresLutContentAndOutput) {
        ParamSnapshot controls17 = RouteInputs::controls_for_route(
            Spektrafilm::ScanRoute::NegativeDirectScan);
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
            capture_route<JuicerCuda::DirectPipelineRunParams>(inputs17);
        const RouteCapture retained =
            capture_route<JuicerCuda::DirectPipelineRunParams>(inputs17);
        const RouteCapture changed =
            capture_route<JuicerCuda::DirectPipelineRunParams>(inputs33);
        const RouteCapture restored =
            capture_route<JuicerCuda::DirectPipelineRunParams>(inputs17);

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
} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    JuicerProcess::root().ensure_bootstrap();
    const int result = RUN_ALL_TESTS();
    JuicerProcess::shutdown_if_initialized();
    return result;
}
