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

#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "output_encoding_oracle.h"
#include "../ffi/exposure_fixture.h"
#include "../ffi/prepared_frame_fixture.h"
#include "../ffi/prepared_product_fixture.h"
#include "../ffi/case_results.h"
#include "../ffi/gtest_cases.h"

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

    std::string route_case_key(Spektrafilm::ScanRoute route) {
        return "route-" + std::to_string(static_cast<int>(route));
    }

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
        const std::string pipeline = std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>
                                         ? "direct"
                                         : "print";
        std::vector<std::string> ids;
        ids.reserve(static_cast<std::size_t>(OutputEncoding::ColorSpace::Count) * 4);
        for (int space = 0; space < static_cast<int>(OutputEncoding::ColorSpace::Count); ++space)
            for (const bool cctf : {false, true})
                for (const bool inputIsOutput : {false, true})
                    ids.push_back("components-" + std::to_string(components) + "/" + pipeline +
                                  "/space-" + std::to_string(space) + "/cctf-" + (cctf ? "on" : "off") +
                                  "/identity-" + (inputIsOutput ? "on" : "off"));
        JuicerTests::GtestCohort cases("transfer-" + pipeline + "-components-" +
                                           std::to_string(components),
                                       ids);
        for (int space = 0; space < static_cast<int>(OutputEncoding::ColorSpace::Count); ++space) {
            for (bool cctf : {false, true}) {
                for (bool inputIsOutput : {false, true}) {
                    auto row = cases.case_event("components-" + std::to_string(components) + "/" + pipeline +
                                                "/space-" + std::to_string(space) + "/cctf-" + (cctf ? "on" : "off") +
                                                "/identity-" + (inputIsOutput ? "on" : "off"));
                    SCOPED_TRACE(testing::Message() << "space=" << space << " cctf=" << cctf << " identity=" << inputIsOutput << " components=" << components);
                    const OutputEncoding::Params settings{OutputEncoding::colorSpaceFromIndex(space), cctf, inputIsOutput};
                    p.scanStage.scanColor.encoding = encoding(settings);
                    image.clear_output();
                    launch_post(p, image.planes.get());
                    image.compare(image.output.download(), image.oracle(values, settings), !cctf && inputIsOutput);
                    row.passed();
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
        std::vector<std::string> ids;
        ids.reserve(static_cast<std::size_t>(OutputEncoding::ColorSpace::Count));
        for (int space = 0; space < static_cast<int>(OutputEncoding::ColorSpace::Count); ++space)
            ids.push_back("space-" + std::to_string(space));
        JuicerTests::GtestCohort cases("output-space", ids);
        for (int space = 0; space < static_cast<int>(OutputEncoding::ColorSpace::Count); ++space) {
            auto row = cases.case_event("space-" + std::to_string(space));
            const OutputEncoding::Params settings{OutputEncoding::colorSpaceFromIndex(space), true, true};
            p.scanStage.scanColor.encoding = encoding(settings);
            image.planes.upload(values);
            image.clear_output();
            launch_post(p, image.planes.get(), effects);
            image.compare(image.output.download(), image.oracle(observed, settings), false);
            row.passed();
        }
    }

    using Frame = JuicerProcess::Root::PreparedCudaFrame;
    namespace FrameFixture = JuicerCudaTest::PreparedFrameFixture;

    struct RouteInputs {
        JuicerTestSupport::RuntimeFixture& runtime;
        std::optional<JuicerTestSupport::PreparedStateHold> state;
        JuicerTestSupport::PreparationInput controls;
        Scanner::ScannerSpectralLutDescriptor scanner;
        Scanner::ColorRuntime scannerColor;
        Scanner::ScannerColorCorrectionDescriptor correction;
        OutputGamutRecipe outputGamut;
        std::optional<Scanner::ScannerPostEffectsDescriptor> post;
        JuicerCuda::FilmPayloadInput filmPayload;
        float printExposure = 0.0f;
        float preflashExposure = 0.0f;
        float filmRouteCorrectionScale = 1.0f;
        std::uint32_t route = 0;
        std::uint64_t recipeHash = 0;
        std::uint64_t scannerRecipeHash = 0;
        std::uint64_t uploadCoreHash = 0;
        std::uint64_t scannerHash = 0;
        std::uint64_t dirHash = 0;

        static JuicerTestSupport::PreparationInput controls_for_route(Spektrafilm::ScanRoute route) {
            JuicerTestSupport::PreparationInput controls;
            controls.values.scan_route = static_cast<std::uint32_t>(route);
            return controls;
        }
        RouteInputs(JuicerTestSupport::RuntimeFixture& processRuntime, Spektrafilm::ScanRoute route, int space, bool gamut)
            : RouteInputs(processRuntime, controls_for_route(route), space, gamut) {}
        RouteInputs(JuicerTestSupport::RuntimeFixture& processRuntime, JuicerTestSupport::PreparationInput controls, int space, bool gamut) : runtime(processRuntime) {
            const Spektrafilm::ScanRoute route = static_cast<Spektrafilm::ScanRoute>(controls.values.scan_route);
            controls.values.grain.active = false;
            controls.values.dir_couplers.active = false;
            controls.values.camera_auto_exposure_enabled = 0;
            controls.values.output_color_space = space;
            controls.values.output_gamut_compression_enabled = gamut ? 1 : 0;
            if (Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Positive) {
                controls.film_key = "fujifilm_provia_100f";
            }
            this->controls = controls;
            state.emplace(JuicerTestSupport::prepare_product(runtime.borrow().assets, this->controls));

            FrameFixture::FrameFacts facts;
            facts.renderExtent = {0, 0, 17, 9};
            facts.fullFrameExtent = facts.renderExtent;
            facts.pixelDefinition = facts.renderExtent;
            facts.filmFormatLongEdgeMm = this->controls.values.camera_film_format_long_edge_mm;
            facts.pixelSizeUm = facts.filmFormatLongEdgeMm * 1000.0f / 17.0f;
            facts.canonicalWidth = facts.filmFormatLongEdgeMm;
            facts.canonicalHeight = facts.filmFormatLongEdgeMm * 9.0 / 17.0;
            facts.scaleX = 1.0;
            facts.scaleY = 1.0;
            facts.frameRate = 24.0;
            std::string diagnostic;
            require(FrameFixture::with_prepared_frame_inputs(
                        runtime.borrow().assets, *state, facts, this, &capture_metadata, diagnostic),
                    diagnostic);
        }

    private:
        static bool capture_metadata(void* user,
                                     const FrameFixture::Inputs& prepared,
                                     std::string&) {
            auto& out = *static_cast<RouteInputs*>(user);
            out.scanner = prepared.descriptors.scanner;
            out.scannerColor = prepared.descriptors.color;
            out.correction = prepared.descriptors.correction;
            out.outputGamut = prepared.descriptors.outputGamut;
            if (prepared.descriptors.post.hash != 0)
                out.post = prepared.descriptors.post;
            out.filmPayload = prepared.filmPayload;
            out.printExposure = prepared.printExposure;
            out.preflashExposure = prepared.preflashExposure;
            out.filmRouteCorrectionScale = prepared.filmRouteCorrectionScale;
            out.route = prepared.state.route;
            out.recipeHash = prepared.state.recipe_hash;
            out.scannerRecipeHash = prepared.state.scanner_recipe_hash;
            out.uploadCoreHash = prepared.state.upload_core_hash;
            out.scannerHash = prepared.state.scanner_hash;
            out.dirHash = prepared.state.dir_recipe.hash;
            return true;
        }
    };

    struct RootFrameCall {
        JuicerProcess::Root& root;
        const JuicerCuda::ResourceManager::DeviceContextKey* key = nullptr;
        const JuicerCuda::ResourceManager::SubmissionSnapshot* snapshot = nullptr;
        cudaStream_t stream = nullptr;
        JuicerCuda::Failure failure;
        Frame frame;
    };

    bool prepare_root_frame(void* user,
                            const FrameFixture::Inputs& prepared,
                            std::string& diagnostic) {
        auto& call = *static_cast<RootFrameCall*>(user);
        call.frame = call.root.prepare_cuda_frame(
            *call.key, *call.snapshot, prepared.preparation, {}, call.stream, call.failure, prepared.hostPreparation);
        if (!call.frame.active()) {
            diagnostic = call.failure.diagnostic;
            return false;
        }
        if (prepared.descriptors.post.hash != 0 &&
            !call.frame.stage_optical_workspace(
                call.frame.workspace_lease(), call.stream, call.failure)) {
            diagnostic = call.failure.diagnostic;
            return false;
        }
        return true;
    }

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
        std::string error;
        JuicerCuda::FilmPayloadPack film;
        require(
            JuicerCuda::pack_film_payloads(
                inputs.filmPayload,
                prepared.film,
                nullptr,
                inputs.filmRouteCorrectionScale,
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
                    inputs.printExposure,
                    inputs.preflashExposure,
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
            inputs.outputGamut);
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
        inline static JuicerTestSupport::RuntimeFixture runtime;

        static void SetUpTestSuite() {
            runtime.create(JuicerProcess::data_directory());
        }
        static void TearDownTestSuite() {
            require(runtime.close().category == FJ_STATUS_SUCCESS, "scanner Runtime close failed");
        }

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
            require(JuicerCuda::borrow_root(runtime.borrow().cuda).retire_idle_context(0, key.contextOpaque, error), error);
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
                inputs.uploadCoreHash,
                inputs.dirHash,
                inputs.scannerHash,
                0);
            RootFrameCall call{JuicerCuda::borrow_root(inputs.runtime.borrow().cuda), &key, &snapshot, nullptr, {}, {}};
            FrameFixture::FrameFacts facts;
            facts.renderExtent = {0, 0, image.width, image.height};
            facts.fullFrameExtent = facts.renderExtent;
            facts.pixelDefinition = facts.renderExtent;
            facts.filmFormatLongEdgeMm = inputs.controls.values.camera_film_format_long_edge_mm;
            facts.pixelSizeUm = facts.filmFormatLongEdgeMm * 1000.0f /
                                static_cast<float>(std::max(image.width, image.height));
            facts.canonicalWidth = facts.filmFormatLongEdgeMm;
            facts.canonicalHeight = facts.filmFormatLongEdgeMm *
                                    static_cast<double>(image.height) /
                                    static_cast<double>(image.width);
            facts.scaleX = 1.0;
            facts.scaleY = 1.0;
            facts.frameRate = 24.0;
            std::string diagnostic;
            require(FrameFixture::with_prepared_frame_inputs(
                        inputs.runtime.borrow().assets, *inputs.state, facts, &call, &prepare_root_frame, diagnostic),
                    diagnostic.empty() ? call.failure.diagnostic : diagnostic);
            return std::move(call.frame);
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
                const OutputEncoding::Params settings{inputs.scannerColor.encoding.colorSpace, cctf, true};
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
        void check_required_lut_failures(const RouteInputs& inputs, const std::string& caller) {
            const std::array<const char*, 7> faultIds = {
                "log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max", "resolution"};
            std::vector<std::string> ids;
            ids.reserve(faultIds.size());
            for (const auto* id : faultIds)
                ids.emplace_back(id);
            JuicerTests::GtestCohort cases("required-lut-failure-" + caller, ids);
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
            for (std::size_t failure = 0; failure < faultIds.size(); ++failure) {
                auto row = cases.case_event(faultIds[failure]);
                SCOPED_TRACE(faultIds[failure]);
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
                row.passed();
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
            if (Spektrafilm::scan_route_is_print(
                    static_cast<Spektrafilm::ScanRoute>(inputs.route))) {
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
            }
            if (derive)
                result.correction = inputs.correction;
            JuicerCuda::Failure failure;
            require(frame.finish(nullptr, failure), failure.diagnostic);
            return result;
        }
        struct PreparedExecutionCall {
            JuicerProcess::Root& root;
            JuicerCuda::ResourceManager::DeviceContextKey* key = nullptr;
            Image* image = nullptr;
            FrameFixture::FrameFacts facts{};
            std::uint64_t snapshotId = 0;
            FjStatus status{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            std::string diagnostic{}; // NOLINT(readability-redundant-member-init): GCC 13 requires explicit aggregate initialization.
            bool recoveryPending = false;
        };

        static bool execute_prepared_scanner(void* user,
                                             const FrameFixture::Inputs& prepared,
                                             std::string&) {
            auto& call = *static_cast<PreparedExecutionCall*>(user);
            auto& image = *call.image;
            const JuicerCuda::FrameRect bounds{
                0, 0, image.width, image.height};
            const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                bounds,
                bounds,
                static_cast<Spektrafilm::AutoExposureMethod>(prepared.state.input_policy.meter));
            const auto rowBytes = static_cast<std::ptrdiff_t>(image.stride * sizeof(float));
            const Spektrafilm::FilmJuicerEffectsGeometry geometry{};
            const JuicerCuda::ExecutionFrame frame{
                bounds,
                bounds,
                bounds,
                reinterpret_cast<const unsigned char*>(image.input.get()),
                reinterpret_cast<const unsigned char*>(image.input.get()),
                reinterpret_cast<unsigned char*>(image.output.get()),
                rowBytes,
                rowBytes,
                image.components,
                nullptr,
                prepared.descriptors.diffusion,
                prepared.descriptors.halation,
                geometry,
                call.facts.pixelSizeUm,
                call.facts.frameTime,
                call.facts.frameRate,
                call.facts.sessionSeed,
                static_cast<std::uintptr_t>(call.facts.clipToken),
                meter,
                false,
                false};
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x5343414e4e4552ull;
            snapshot.frameToken.value = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(std::floor(call.facts.frameTime)));
            snapshot.snapshotId = call.snapshotId;
            snapshot.deviceContextKey = *call.key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                prepared.state.upload_core_hash,
                prepared.state.dir_recipe.hash,
                prepared.state.scanner_hash,
                meter.hash);
            const JuicerCuda::PreparedExecutionInput input{
                prepared.preparation,
                prepared.descriptors,
                prepared.filmPayload,
                prepared.filmRouteCorrectionScale,
                prepared.printExposure,
                prepared.preflashExposure,
                prepared.state.recipe_hash,
                (prepared.state.film_exposure.flags & FJ_FILM_AUTO_EXPOSURE) != 0,
                frame,
                snapshot,
                prepared.hostPreparation};
            JuicerCuda::PendingContextLossRecovery recovery;
            try {
                JuicerCuda::execute_prepared(call.root, input, recovery);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                call.status = failure.failure.status;
                call.diagnostic = failure.failure.diagnostic;
            }
            call.recoveryPending = recovery.pending;
            check_cuda(cudaDeviceSynchronize());
            return true;
        }

        FjStatus execute_correction(const RouteInputs& inputs, Image& image, std::string& diagnostic) {
            image.clear_output();
            auto preparation = JuicerCuda::borrow_root(runtime.borrow().cuda).begin_frame_preparation();
            require(preparation.active(), "scanner actual-caller frame preparation admission");
            PreparedExecutionCall call{JuicerCuda::borrow_root(inputs.runtime.borrow().cuda)};
            call.key = &key;
            call.image = &image;
            call.snapshotId = nextIdentity++;
            call.facts.renderExtent = {0, 0, image.width, image.height};
            call.facts.fullFrameExtent = call.facts.renderExtent;
            call.facts.pixelDefinition = call.facts.renderExtent;
            call.facts.pixelSizeUm = 100.0f;
            call.facts.filmFormatLongEdgeMm = inputs.controls.values.camera_film_format_long_edge_mm;
            call.facts.canonicalWidth = inputs.controls.values.camera_film_format_long_edge_mm;
            call.facts.canonicalHeight =
                call.facts.canonicalWidth * static_cast<double>(image.height) / image.width;
            call.facts.scaleX = 1.0;
            call.facts.scaleY = 1.0;
            call.facts.frameTime = 0.0;
            call.facts.frameRate = 24.0;
            call.facts.sessionSeed = 0x534334;
            call.facts.clipToken = 0x5343414e;
            require(FrameFixture::with_prepared_frame_inputs(
                        inputs.runtime.borrow().assets, *inputs.state, call.facts, &call, &execute_prepared_scanner, diagnostic),
                    diagnostic);
            diagnostic = call.diagnostic;
            EXPECT_FALSE(call.recoveryPending);
            return call.status;
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
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size());
        for (const auto route : routes)
            ids.push_back(route_case_key(route));
        JuicerTests::GtestCohort cases("route", ids);
        for (const auto route : routes) {
            auto row = cases.case_event(route_case_key(route));
            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route));
            auto controls = RouteInputs::controls_for_route(route);
            controls.values.glare_active = false;
            controls.values.scanner_lut_resolution = 17;
            RouteInputs off(runtime, controls, 0, true);
            controls.values.scanner_black_correction = controls.values.scanner_white_correction = 1;
            RouteInputs held(runtime, controls, 0, true);
            const auto heldHash = held.recipeHash;
            const auto heldScannerHash = held.scannerRecipeHash;
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
            EXPECT_NE(off.recipeHash, heldHash);
            EXPECT_NE(off.scannerRecipeHash, heldScannerHash);
            std::string diagnostic;
            require(execute_correction(off, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            const auto offWords = bit_patterns(image.output.download());
            require(execute_correction(held, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            const auto corrected = image.output.download();
            check_correction_output(image, corrected);
            if (correctionActive)
                EXPECT_NE(offWords, bit_patterns(corrected));
            else
                EXPECT_EQ(offWords, bit_patterns(corrected));

            if (correctionActive) {
                auto failedControls = held.controls;
                failedControls.values.scanner_black_level = failedControls.values.scanner_white_level = 0.0;

                std::optional<JuicerTestSupport::PreparedStateHold> rejectedState;
                std::array<char, 1024> text{};
                FjErrorBuffer error{text.data(), text.size(), 0};
                const auto* const assets = runtime.borrow().assets;
                ASSERT_NE(assets, nullptr);
                const auto status = JuicerTestSupport::prepare_state(
                    assets,
                    failedControls,
                    rejectedState,
                    error);
                EXPECT_EQ(status.category, FJ_STATUS_PREPARATION_FAILURE);
                EXPECT_EQ(status.api, FJ_API_NONE);
                EXPECT_EQ(status.native_code, 0);
                EXPECT_FALSE(rejectedState.has_value());
                const std::string preparedDiagnostic(
                    text.data(), std::min(error.length, text.size()));
                EXPECT_EQ(preparedDiagnostic,
                          Spektrafilm::scan_route_is_print(route)
                              ? "MalformedRequiredProfileData phase=8B print corrected midgray"
                              : "MalformedRequiredProfileData phase=8B direct corrected midgray");
            }
            require(execute_correction(held, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            EXPECT_EQ(bit_patterns(corrected), bit_patterns(image.output.download()));
            const auto restored = correction_resources(held, image);
            EXPECT_EQ(restored.correction.hash, active.correction.hash);
            EXPECT_EQ(restored.lut.addresses, active.lut.addresses);
            EXPECT_EQ(restored.lut.content, active.lut.content);
            EXPECT_EQ(restored.printHashes, active.printHashes);
            EXPECT_EQ(restored.printAddresses, active.printAddresses);
            EXPECT_EQ(restored.printValues, active.printValues);
            EXPECT_EQ(held.recipeHash, heldHash);
            EXPECT_EQ(held.scannerRecipeHash, heldScannerHash);
            row.passed();
        }
    }

    TEST_F(ScannerRoutes, GamutSwitchKeepsPreparedRenderingStable) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size() * 4);
        for (const auto route : routes) {
            for (std::size_t pass = 0; pass < 4; ++pass) {
                const bool enabled = pass % 2 == 1;
                ids.push_back(route_case_key(route) + "/pass-" + std::to_string(pass) +
                              "/gamut-" + (enabled ? "on" : "off"));
            }
        }
        JuicerTests::GtestCohort cases("route-gamut-pass", ids);
        for (const auto route : routes) {
            auto controls = RouteInputs::controls_for_route(route);
            controls.values.glare_active = false;
            controls.values.scanner_lut_resolution = 17;
            controls.values.scanner_black_correction = controls.values.scanner_white_correction = 1;
            RouteInputs disabled(runtime, controls, 0, false);
            RouteInputs enabled(runtime, controls, 0, true);
            const auto disabledHash = disabled.outputGamut.hash;
            const auto enabledHash = enabled.outputGamut.hash;
            ASSERT_NE(disabledHash, 0u);
            ASSERT_NE(enabledHash, 0u);
            EXPECT_NE(disabledHash, enabledHash);

            Image image({31, 19}, 4);
            const auto initial = correction_resources(disabled, image);
            std::array<std::vector<std::uint32_t>, 2> expected;
            const std::array inputsByPass{&disabled, &enabled, &disabled, &enabled};
            for (std::size_t pass = 0; pass < inputsByPass.size(); ++pass) {
                const auto* inputs = inputsByPass[pass];
                auto row = cases.case_event(route_case_key(route) + "/pass-" +
                                            std::to_string(pass) + "/gamut-" +
                                            (inputs->outputGamut.enabled ? "on" : "off"));
                const auto& gamut = inputs->outputGamut;
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
                require(execute_correction(*inputs, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                const auto projected = bit_patterns(image.output.download());
                check_correction_output(image, image.output.download());
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
                if (pass + 1 == inputsByPass.size()) {
                    EXPECT_EQ(disabled.outputGamut.hash, disabledHash);
                    EXPECT_EQ(enabled.outputGamut.hash, enabledHash);
                }
                row.passed();
            }
        }
    }

    TEST_F(ScannerRoutes, PostConstructorResourcesReuseRebuildRejectAndRecover) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size());
        for (const auto route : routes)
            ids.push_back(route_case_key(route));
        JuicerTests::GtestCohort cases("route", ids);
        for (const auto route : routes) {
            auto row = cases.case_event(route_case_key(route));
            auto controls = RouteInputs::controls_for_route(route);
            controls.values.glare_active = true;
            controls.values.glare_percent = 2.0f;
            controls.values.glare_roughness = 0.3f;
            controls.values.glare_blur_sigma_px = 0.75f;
            controls.values.scanner_lens_blur_sigma_px = 0.75f;
            controls.values.scanner_unsharp_mask[0] = 0.75;
            controls.values.scanner_unsharp_mask[1] = 0.8;
            RouteInputs inputs(runtime, controls, 0, false);
            Image image({17, 9}, 4);
            ASSERT_TRUE(inputs.post);
            const auto original = *inputs.post;
            auto first = prepare_route(inputs, image);
            auto view = first.scanner_post_effects_resources(first.workspace_lease(), original.hash);
            ASSERT_TRUE(view.active);
            ASSERT_NE(view.lensBlur.weights, nullptr);
            EXPECT_EQ(view.lensBlur.radius, 2);
            EXPECT_EQ(view.lensBlur.sigma, 0.75f);
            EXPECT_EQ(view.lensBlur.weights, view.unsharp.weights);
            if (Spektrafilm::scan_route_is_print(route)) {
                EXPECT_EQ(view.glare.weights, view.lensBlur.weights);
            } else {
                EXPECT_EQ(view.glare.weights, nullptr);
            }
            EXPECT_FALSE(first.scanner_post_effects_resources(first.workspace_lease(), original.hash ^ 1).active);
            const auto check = [](const auto& kernel) {
                if (!kernel.weights)
                    return;
                const auto expected = gaussian_kernel_fixture_words(JUICER_GAUSSIAN_FIXTURE_PATH, kernel.radius, kernel.sigma);
                std::vector<float> actual(expected.size());
                require(cudaMemcpy(actual.data(), kernel.weights, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "scanner Gaussian download");
                for (std::size_t i = 0; i < actual.size(); ++i)
                    require(std::bit_cast<std::uint32_t>(actual[i]) == expected[i], "every scanner Gaussian coefficient matches frozen words");
            };
            check(view.lensBlur);
            check(view.unsharp);
            check(view.glare);
            const auto lut = capture_lut(*first.focused_resources().scanLut);
            const auto* retained = view.lensBlur.weights;
            JuicerCuda::Failure failure;
            require(first.finish(nullptr, failure), failure.diagnostic);
            const auto acquisitions = fj_test_gaussian_kernel_acquires();
            auto warm = prepare_route(inputs, image);
            EXPECT_EQ(fj_test_gaussian_kernel_acquires(), acquisitions);
            view = warm.scanner_post_effects_resources(warm.workspace_lease(), original.hash);
            ASSERT_TRUE(view.active);
            EXPECT_EQ(view.lensBlur.weights, retained);
            require(warm.finish(nullptr, failure), failure.diagnostic);
            auto changedControls = controls;
            changedControls.values.scanner_lens_blur_sigma_px = 1.25f;
            RouteInputs changedInputs(runtime, changedControls, 0, false);
            ASSERT_TRUE(changedInputs.post);
            auto changed = prepare_route(changedInputs, image);
            view = changed.scanner_post_effects_resources(changed.workspace_lease(), changedInputs.post->hash);
            ASSERT_TRUE(view.active);
            EXPECT_EQ(view.lensBlur.radius, 4);
            check(view.lensBlur);
            EXPECT_NE(view.lensBlur.weights, retained);
            EXPECT_EQ(view.unsharp.weights, retained);
            EXPECT_EQ(capture_lut(*changed.focused_resources().scanLut).addresses, lut.addresses);
            EXPECT_EQ(capture_lut(*changed.focused_resources().scanLut).content, lut.content);
            // A failed descriptor cannot replace resources bound by a live frame.
            ASSERT_EQ(fj_test_scanner_post_fault(1).category, FJ_STATUS_SUCCESS);
            const FjScannerPostEffectsInput raw{
                static_cast<std::uint32_t>(route), 1u, 2.0f, 0.3f, 0.75f, 1.25f, 0.75f, 0.8f};
            FjScannerPostEffectsDescriptor rejected{};
            FjScannerPostEffectsFailure rejectedFailure{};
            std::array<char, 256> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            const auto status = fj_test_scanner_post_describe(
                &raw, &rejected, &rejectedFailure, &error);
            EXPECT_EQ(status.category, FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_EQ(status.api, FJ_API_NONE);
            EXPECT_EQ(status.native_code, 0);
            EXPECT_EQ(rejectedFailure.reason, 4u);
            EXPECT_EQ(rejectedFailure.lane, 0u);
            EXPECT_EQ(rejected.hash, 0u);
            EXPECT_TRUE(changed.scanner_post_effects_resources(
                                   changed.workspace_lease(), changedInputs.post->hash)
                            .active);
            require(changed.finish(nullptr, failure), failure.diagnostic);
            EXPECT_FALSE(changed.scanner_post_effects_resources(
                                    changed.workspace_lease(), changedInputs.post->hash)
                             .active);
            auto restored = prepare_route(inputs, image);
            view = restored.scanner_post_effects_resources(restored.workspace_lease(), original.hash);
            ASSERT_TRUE(view.active);
            EXPECT_EQ(view.lensBlur.weights, retained);
            require(restored.finish(nullptr, failure), failure.diagnostic);
            auto offControls = controls;
            offControls.values.glare_active = false;
            offControls.values.glare_percent = 0.0f;
            offControls.values.glare_roughness = 0.0f;
            offControls.values.glare_blur_sigma_px = 0.0f;
            offControls.values.scanner_lens_blur_sigma_px = 0.0f;
            offControls.values.scanner_unsharp_mask[0] = 0.0;
            offControls.values.scanner_unsharp_mask[1] = 0.0;
            RouteInputs offInputs(runtime, offControls, 0, false);
            EXPECT_FALSE(offInputs.post);
            const auto offAcquires = fj_test_gaussian_kernel_acquires();
            auto off = prepare_route(offInputs, image);
            EXPECT_EQ(fj_test_gaussian_kernel_acquires(), offAcquires);
            EXPECT_FALSE(off.scanner_post_effects_resources(off.workspace_lease(), 0).active);
            EXPECT_FALSE(off.scanner_workspace(off.workspace_lease()).active);
            require(off.finish(nullptr, failure), failure.diagnostic);
            row.passed();
        }
    }

    TEST_F(ScannerRoutes, FusedAndSeparateOutputMatchOnAllFourRoutes) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size() * 2 * 2);
        for (const auto route : routes)
            for (const bool gamut : {false, true})
                for (const int space : {0, 7})
                    ids.push_back(route_case_key(route) + "/gamut-" + (gamut ? "on" : "off") +
                                  "/space-" + std::to_string(space));
        JuicerTests::GtestCohort cases("route-gamut-space", ids);
        for (auto route : routes) {
            for (bool gamut : {false, true}) {
                for (int space : {0, 7}) {
                    auto row = cases.case_event(route_case_key(route) + "/gamut-" +
                                                (gamut ? "on" : "off") + "/space-" + std::to_string(space));
                    SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route) << " space=" << space << " gamut=" << gamut);
                    RouteInputs inputs(runtime, route, space, gamut);
                    if (Spektrafilm::scan_route_is_print(route)) {
                        check_route<JuicerCuda::PrintPipelineRunParams>(inputs);
                    } else {
                        check_route<JuicerCuda::DirectPipelineRunParams>(inputs);
                    }
                    row.passed();
                }
            }
        }
    }

    TEST_F(ScannerRoutes, RequiredLutFailuresAreReportedByBothScannerCallers) {
        JuicerTests::GtestCohort cases("caller", {"direct", "print"});
        auto directRow = cases.case_event("direct");
        RouteInputs direct(runtime,
                           Spektrafilm::ScanRoute::NegativeDirectScan,
                           0,
                           false);
        check_required_lut_failures<JuicerCuda::DirectPipelineRunParams>(direct, "direct");
        directRow.passed();

        auto printRow = cases.case_event("print");
        RouteInputs print(runtime,
                          Spektrafilm::ScanRoute::PositivePrintScan,
                          0,
                          false);
        check_required_lut_failures<JuicerCuda::PrintPipelineRunParams>(print, "print");
        printRow.passed();
    }

    TEST_F(ScannerRoutes, DeferredDirFailureComesFromReadbackBits) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        const std::array bitsCases{0, 1, (1 << 8) | (2 << 9), 1 | (1 << 8) | (7 << 9)};
        std::vector<std::string> ids;
        ids.reserve(routes.size() * bitsCases.size());
        for (const auto route : routes)
            for (const int bits : bitsCases)
                ids.push_back(route_case_key(route) + "/readback-" + std::to_string(bits));
        JuicerTests::GtestCohort cases("route-readback-bits", ids);
        for (const auto route : routes) {
            RouteInputs inputs(runtime, route, 0, false);
            Image image({17, 9}, 4);
            // Product readback contract: bit 0 is scanner RGB; bit 8 is DIR,
            // with the failed DIR channels in bits 9-11.
            for (const int bits : bitsCases) {
                auto row = cases.case_event(route_case_key(route) + "/readback-" + std::to_string(bits));
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
                row.passed();
            }
        }
    }

    TEST_F(ScannerRoutes, ResolutionTransitionRestoresLutContentAndOutput) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size());
        for (const auto route : routes)
            ids.push_back(route_case_key(route));
        JuicerTests::GtestCohort cases("route", ids);
        for (const auto route : routes) {
            auto row = cases.case_event(route_case_key(route));
            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route));
            const bool print = Spektrafilm::scan_route_is_print(route);
            JuicerTestSupport::PreparationInput controls17 = RouteInputs::controls_for_route(route);
            controls17.values.scanner_lut_resolution = 17;
            JuicerTestSupport::PreparationInput controls33 = controls17;
            controls33.values.scanner_lut_resolution = 33;
            RouteInputs inputs17(runtime, controls17, 0, false);
            RouteInputs inputs33(runtime, controls33, 0, false);

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
            row.passed();
        }
    }

    TEST_F(ScannerRoutes, PrintResourceCacheAcceptsPublishedValuesAcrossDescriptorChanges) {
        const std::vector<std::string> scenarios{
            "retained-default", "retained-gamma", "illuminant-change", "main-filter-change", "preflash-enabled", "preflash-disabled", "normalization-disabled", "normalization-enabled", "print-profile-change"};
        JuicerTests::GtestCohort cases("print-cache-scenario", scenarios);
        JuicerTestSupport::PreparationInput defaultControls =
            RouteInputs::controls_for_route(Spektrafilm::ScanRoute::NegativePrintScan);
        JuicerTestSupport::PreparationInput gammaControls = defaultControls;
        gammaControls.values.print_gamma_factor = 1.1;
        RouteInputs retainedDefault(runtime, defaultControls, 0, false);
        RouteInputs retainedGamma(runtime, gammaControls, 0, false);

        auto check_print = [&](const char* scenario, JuicerTestSupport::PreparationInput controls) {
            SCOPED_TRACE(scenario);
            auto row = cases.case_event(scenario);
            RouteInputs inputs(runtime, std::move(controls), 0, false);
            check_route<JuicerCuda::PrintPipelineRunParams>(inputs);
            row.passed();
        };

        {
            SCOPED_TRACE("retained default publication");
            auto row = cases.case_event("retained-default");
            check_route<JuicerCuda::PrintPipelineRunParams>(retainedDefault);
            check_route<JuicerCuda::PrintPipelineRunParams>(retainedDefault);
            row.passed();
        }
        {
            SCOPED_TRACE("retained gamma publication");
            auto row = cases.case_event("retained-gamma");
            check_route<JuicerCuda::PrintPipelineRunParams>(retainedGamma);
            row.passed();
        }

        JuicerTestSupport::PreparationInput illuminantControls = defaultControls;
        illuminantControls.values.enl_ill = 0;
        check_print("illuminant-change", illuminantControls);

        JuicerTestSupport::PreparationInput filterControls = defaultControls;
        filterControls.values.print_ui_ymc_cc[0] = 4.0;
        filterControls.values.print_ui_ymc_cc[1] = -3.0;
        filterControls.values.print_ui_ymc_cc[2] = 2.0;
        check_print("main-filter-change", filterControls);

        JuicerTestSupport::PreparationInput preflashControls = defaultControls;
        preflashControls.values.print_preflash_exposure = 0.1;
        preflashControls.values.preflash_m_filter_cc = 3.0;
        preflashControls.values.preflash_y_filter_cc = -2.0;
        check_print("preflash-enabled", preflashControls);
        check_print("preflash-disabled", defaultControls);

        JuicerTestSupport::PreparationInput unnormalizedControls = defaultControls;
        unnormalizedControls.values.normalize_print_exposure = 0;
        check_print("normalization-disabled", unnormalizedControls);
        check_print("normalization-enabled", defaultControls);

        JuicerTestSupport::PreparationInput profileControls = defaultControls;
        profileControls.print_key = "kodak_supra_endura";
        check_print("print-profile-change", profileControls);
    }
    TEST_F(ScannerRoutes, LutConstructionFailuresPreserveCategoryAndCache) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size() * 13);
        for (const auto route : routes) {
            for (unsigned buffer = 1; buffer <= 10; ++buffer)
                ids.push_back(route_case_key(route) + "/buffer-" + std::to_string(buffer));
            ids.push_back(route_case_key(route) + "/cache-hit-armed-fault");
            for (const unsigned operation : {0u, 1u})
                ids.push_back(route_case_key(route) + "/operation-" + std::to_string(operation));
        }
        JuicerTests::GtestCohort cases("route-fault", ids);
        for (const auto route : routes) {
            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route));
            auto controls = RouteInputs::controls_for_route(route);
            controls.values.scanner_lut_resolution = 17;
            RouteInputs inputs(runtime, controls, 0, false);
            Image image({31, 19}, 4);
            std::string diagnostic;
            const auto reject = [&](const RouteInputs& selected, std::uint32_t category, const char* message) {
                const auto result = execute_correction(selected, image, diagnostic);
                EXPECT_EQ(result.category, category);
                EXPECT_EQ(result.api, FJ_API_NONE);
                EXPECT_EQ(result.native_code, 0);
                EXPECT_EQ(diagnostic, std::string(message) + " route=" + Spektrafilm::scan_route_metadata(route).key);
                EXPECT_EQ(fj_test_scanner_lut_live_owners(), 0u);
                for (const auto word : bit_patterns(image.output.download()))
                    EXPECT_EQ(word, kPadding);
            };
            for (unsigned buffer = 1; buffer <= 10; ++buffer) {
                auto row = cases.case_event(route_case_key(route) + "/buffer-" + std::to_string(buffer));
                SCOPED_TRACE(testing::Message() << "buffer=" << buffer);
                ASSERT_EQ(fj_test_scanner_lut_fault(0, buffer + 1).category, FJ_STATUS_SUCCESS);
                reject(inputs, FJ_STATUS_ALLOCATION_FAILURE, "scanner LUT host allocation failed");
                row.passed();
            }
            require(execute_correction(inputs, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            const auto original = bit_patterns(image.output.download());
            controls.values.scanner_lut_resolution = 33;
            RouteInputs replacement(runtime, controls, 0, false);
            {
                auto row = cases.case_event(route_case_key(route) + "/cache-hit-armed-fault");
                ASSERT_EQ(fj_test_scanner_lut_fault(0, 2).category, FJ_STATUS_SUCCESS);
                require(execute_correction(inputs, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
                EXPECT_EQ(bit_patterns(image.output.download()), original);
                // A cache hit must leave the armed constructor fault for this miss.
                reject(replacement, FJ_STATUS_ALLOCATION_FAILURE, "scanner LUT host allocation failed");
                row.passed();
            }
            for (const unsigned operation : {0u, 1u}) {
                auto row = cases.case_event(route_case_key(route) + "/operation-" + std::to_string(operation));
                ASSERT_EQ(fj_test_scanner_lut_fault(operation, 1).category, FJ_STATUS_SUCCESS);
                reject(replacement, FJ_STATUS_INTERNAL_FAILURE, "scanner LUT preparation panicked");
                row.passed();
            }
            require(execute_correction(inputs, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            EXPECT_EQ(bit_patterns(image.output.download()), original);
            require(execute_correction(replacement, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            require(execute_correction(inputs, image, diagnostic).category == FJ_STATUS_SUCCESS, diagnostic);
            EXPECT_EQ(bit_patterns(image.output.download()), original);
            EXPECT_EQ(fj_test_scanner_lut_live_owners(), 0u);
        }
    }

    TEST_F(ScannerRoutes, LutIdentityTracksSelectedProfileBoundsAndIlluminant) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size());
        for (const auto route : routes)
            ids.push_back(route_case_key(route));
        JuicerTests::GtestCohort cases("route", ids);
        for (const auto route : routes) {
            auto row = cases.case_event(route_case_key(route));
            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route));
            auto controls = RouteInputs::controls_for_route(route);
            controls.film_key = "kodak_portra_400";
            controls.print_key = "kodak_portra_endura";
            controls.values.scanner_lut_resolution = 17;
            RouteInputs original(runtime, controls, 0, false);
            const bool print = Spektrafilm::scan_route_is_print(route);
            if (print)
                controls.print_key = "kodak_2383";
            else
                controls.film_key = "kodak_portra_800";
            RouteInputs changed(runtime, controls, 0, false);
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
            row.passed();
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
        const auto frozen = load_shared_fixture(directory / (preset + ".json"), "manifest.json");
        ASSERT_EQ(frozen.at("preset"), preset);
        std::vector<std::string> ids;
        ids.reserve(frozen.at("products").size());
        for (std::size_t index = 0; index < frozen.at("products").size(); ++index) {
            const auto& row = frozen.at("products").at(index);
            ids.push_back("product-" + std::to_string(index) + "/route-" +
                          std::to_string(row.at("route").get<unsigned>()));
        }
        JuicerTests::GtestCohort products("native-lut-product", ids);
        for (std::size_t index = 0; index < frozen.at("products").size(); ++index) {
            const auto& row = frozen.at("products").at(index);
            auto product = products.case_event(ids[index]);
            const auto route = static_cast<Spektrafilm::ScanRoute>(row.at("route").get<unsigned>());
            JuicerTestSupport::PreparationInput controls;
            controls.values.scan_route = static_cast<std::uint32_t>(route);
            controls.film_key = row.at("route").get<unsigned>() < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            controls.print_key = "kodak_portra_endura";
            controls.values.camera_auto_exposure_enabled = 0;
            controls.values.input_compression_enabled = 0;
            controls.values.output_gamut_compression_enabled = 0;
            controls.values.print_preflash_exposure = 0.125;
            controls.values.scanner_lut_resolution = 17;
            RouteInputs inputs(runtime, controls, 0, false);
            const bool print = Spektrafilm::scan_route_is_print(route);
            ASSERT_EQ(inputs.scanner.hash, row.at("descriptor_hash").get<std::uint64_t>());
            const auto capture = print ? capture_route<JuicerCuda::PrintPipelineRunParams>(inputs) : capture_route<JuicerCuda::DirectPipelineRunParams>(inputs);
            std::vector<std::uint32_t> expected;
            for (const auto* name : {"log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max"})
                for (const auto& word : row.at("native").at(name))
                    expected.push_back(word.get<std::uint32_t>());
            EXPECT_EQ(capture.lut.content, expected);
            EXPECT_EQ(capture.lut.bytes, expected.size() * sizeof(float));
            product.passed();
        }
    }
} // namespace

int main(int argc, char** argv) try {
    testing::InitGoogleTest(&argc, argv);
    if (testing::GTEST_FLAG(list_tests))
        return RUN_ALL_TESTS();
    JuicerTests::CaseLog case_log;
    const int result = RUN_ALL_TESTS();
    case_log.finish();
    return result;
} catch (...) {
    return 1;
}
