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
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

#include "SpectralProcessing.h"
#include "JuicerState.h"
#include "ProcessRoot.h"

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(const JuicerCuda::DirectPipelineRunParams*, void*);
extern "C" cudaError_t juicer_cuda_print_focused_pipeline(const JuicerCuda::PrintPipelineRunParams*, void*);
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
        RouteInputs(Spektrafilm::ScanRoute route, int space, bool gamut) {
            ParamSnapshot controls;
            controls.scanRoute = route;
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
            r.outputGamutTransform = product.payload.outputBoundaryTable ? &product.payload.outputGamutTransform : nullptr;
            r.outputBoundaryTable = product.payload.outputBoundaryTable.get();
            r.requestedWidth = image.width;
            r.requestedHeight = image.height;
            return r;
        }
    };

    void bind_scanner(JuicerCuda::ScanStagePayload& scan, const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared, const OutputGamutRecipe& gamut) {
        const auto& medium = *prepared.scanMedium;
        scan.scanTables.mediumIsNegative = medium.mediumIsNegative;
        std::copy_n(medium.min_cmy, 3, scan.scanTables.min_cmy);
        std::copy_n(medium.inv_max_cmy, 3, scan.scanTables.inv_max_cmy);
        scan.scannerUseLut = 1;
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
        template <typename Params>
        void check_route(const RouteInputs& inputs) {
            Image image({65, 33}, 4);
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x5343414e4e4552ull;
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.contextEpoch = 1;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(inputs.product.payload.uploadCoreHash, inputs.product.recipe.dirCouplers.hash, inputs.product.payload.scannerHash, 0);
            std::string error;
            auto frame = JuicerProcess::root().prepare_cuda_frame(key, snapshot, inputs.request(image), {}, nullptr, error);
            require(frame.active(), error);
            const auto& recipe = inputs.product.recipe;
            JuicerCuda::FilmPayloadPack film;
            require(JuicerCuda::pack_film_payloads(recipe.filmRaw, recipe.filmDevelop, recipe.dirCouplers, recipe.densityBounds, frame.focused_resources().film, nullptr, 1.0f, film, error), error);
            auto p = image.params<Params>();
            p.filmRaw = film.filmRaw;
            p.filmExpose = film.filmExposure;
            p.filmDevelop = film.filmDevelop;
            if constexpr (std::is_same_v<Params, JuicerCuda::PrintPipelineRunParams>) {
                JuicerCuda::PrintCudaPayloadPack print;
                require(JuicerCuda::pack_print_cuda_payloads(recipe.print, frame.print_resources(), 1.0f, print, error), error);
                p.printExpose = print.expose;
                p.printDevelop = print.develop;
            }
            bind_scanner(p.scanStage, frame.focused_resources(), recipe.scannerOutput.outputGamut);
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
            require(frame.finalize_scan_error_stage(p.scanStage.scanErrorFlag, nullptr, error), error);
            require(frame.finish(nullptr, error), error);
        }
        JuicerCuda::ResourceManager::DeviceContextKey key;
        inline static std::uint64_t nextIdentity = 1;
    };

    TEST_F(ScannerRoutes, FusedAndSeparateOutputMatchOnAllFourRoutes) {
        for (auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            for (int space : {0, 7}) {
                for (bool gamut : {false, true}) {
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
} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    JuicerProcess::root().ensure_bootstrap();
    const int result = RUN_ALL_TESTS();
    JuicerProcess::shutdown_if_initialized();
    return result;
}
