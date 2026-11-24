#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <span>
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

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(const JuicerCuda::DirectPipelineRunParams*, const JuicerCuda::ScannerSpectra&, void*, JuicerCuda::CameraFilmLinearExposurePlanes);
extern "C" cudaError_t juicer_cuda_print_focused_pipeline(const JuicerCuda::PrintPipelineRunParams*, const JuicerCuda::ScannerSpectra&, void*, JuicerCuda::CameraFilmLinearExposurePlanes);
extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_density_rgb(const JuicerCuda::DirectPipelineRunParams*, const JuicerCuda::ScannerSpectra&, const float*, const float*, const float*, float*, float*, float*, const float*, void*);
extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_density_rgb(const JuicerCuda::PrintPipelineRunParams*, const JuicerCuda::ScannerSpectra&, const float*, const float*, const float*, float*, float*, float*, float*, float*, int, int, std::uint64_t, float, float, const float*, int, void*);
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
    void launch_pipeline(const Params& params, const JuicerCuda::ScannerSpectra& spectra) {
        if constexpr (std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>) {
            check_cuda(juicer_cuda_negative_direct_pipeline(&params, spectra, nullptr, {}));
        } else {
            check_cuda(juicer_cuda_print_focused_pipeline(&params, spectra, nullptr, {}));
        }
    }

    template <typename Params>
    void launch_density_scan(
        const Params& params,
        const JuicerCuda::ScannerSpectra& spectra,
        const float* densityC,
        const float* densityM,
        const float* densityY,
        float* rgbR,
        float* rgbG,
        float* rgbB) {
        if constexpr (std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>) {
            check_cuda(juicer_cuda_direct_focused_scan_linear_density_rgb(
                &params,
                spectra,
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
                spectra,
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
        int maskWidth = 0;
        int maskHeight = 0;
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
        check_cuda(launch(&params, planes, planes + count, planes + 2 * count, effects.temporary, effects.kernel, effects.kernel ? 1 : 0, effects.kernel, effects.kernel ? 1 : 0, 0.7f, &effects.defects, &effects.weave, effects.mask, effects.maskWidth, effects.maskHeight, nullptr));
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
        effects.maskWidth = 1;
        effects.maskHeight = 1;
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

        Image offsetImage({5, 5}, 4);
        Buffer offsetMask(16);
        std::vector<float> maskValues(16);
        for (std::size_t i = 0; i < maskValues.size(); ++i) {
            maskValues[i] = static_cast<float>(i) / 16.0F;
        }
        offsetMask.upload(maskValues);
        offsetImage.planes.upload(std::vector<float>(3 * offsetImage.count, 0.5F));
        PostEffects offsets;
        offsets.mask = offsetMask.get();
        offsets.maskWidth = 4;
        offsets.maskHeight = 4;
        JuicerTests::GtestCohort offsetCases("gate-offsets", {"direct", "print"});
        const auto verifyOffsets = [&]<typename Params>(std::string_view id) {
            auto row = offsetCases.case_event(id);
            const auto params = offsetImage.params<Params>();
            const auto check = [&](std::array<int, 2> origin, float expected) {
                offsets.defects.roiOffsetX = origin[0];
                offsets.defects.roiOffsetY = origin[1];
                offsetImage.clear_output();
                launch_post(params, offsetImage.planes.get(), offsets);
                const auto actual = offsetImage.output.download();
                const std::size_t pixel = 2 * offsetImage.stride + 8u;
                const auto activeWords = static_cast<std::size_t>(offsetImage.width) * 4u;
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    EXPECT_EQ(actual[pixel + channel], expected);
                }
                for (int scanline = 0; scanline < offsetImage.height; ++scanline) {
                    const std::size_t start = static_cast<std::size_t>(scanline) * offsetImage.stride;
                    for (std::size_t column = 0; column < offsetImage.stride; ++column) {
                        if (column % 4 == 3 || column >= activeWords) {
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[start + column]),
                                      std::bit_cast<std::uint32_t>(offsetImage.source[start + column]));
                        }
                    }
                }
            };
            check({1, 3}, 41.0F / 128.0F);
            check({3, 1}, 29.0F / 128.0F);
            check({0, 3}, 39.0F / 128.0F);
            check({1, 0}, 17.0F / 128.0F);
            check({0, 0}, 15.0F / 128.0F);
            row.passed();
        };
        verifyOffsets.template operator()<JuicerCuda::DirectPipelineRunParams>("direct");
        verifyOffsets.template operator()<JuicerCuda::PrintPipelineRunParams>("print");
    }

    using Frame = JuicerProcess::Root::PreparedCudaFrame;
    namespace FrameFixture = JuicerCudaTest::PreparedFrameFixture;

    struct RouteInputs {
        JuicerTestSupport::RuntimeFixture& runtime;
        std::optional<JuicerTestSupport::PreparedStateHold> state;
        JuicerTestSupport::PreparationInput controls;
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
        require(JuicerCuda::pack_scan_color(scan.scanColor, prepared, gamut), "Missing scanner colour resources");
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

    using SpectralWords = std::array<std::uint32_t, 7 * Spectral::kNumSamples + 1>;

    SpectralWords capture_spectra(const JuicerCuda::ScannerSpectra& spectra) {
        return std::bit_cast<SpectralWords>(spectra);
    }

    bool scan_color_matches(
        const JuicerCuda::ScanColorPayload& actual,
        const FjOutputColor& expected,
        const float* deviceCmax) {
        const auto word = [](float value) {
            return std::bit_cast<std::uint32_t>(value);
        };
        const auto words_equal = [&](std::span<const float> lhs, std::span<const float> rhs) {
            return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), [&](float a, float b) {
                return word(a) == word(b);
            });
        };
        const bool gamut = (expected.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0;
        const auto& cctf = actual.encoding.cctf;
        const auto& encoding = expected.encoding;
        if (actual.outputGamutActive != static_cast<int>(gamut) ||
            !words_equal(actual.cat02, expected.cat02) ||
            !words_equal(actual.xyzToRgb, expected.xyz_to_rgb) ||
            !words_equal(actual.illuminantXYZ, expected.illuminant_xyz) ||
            actual.encoding.applyCctfEncoding != static_cast<int>((expected.flags & FJ_COLOR_ENCODE_CCTF) != 0) ||
            actual.encoding.inputIsOutputSpace != static_cast<int>((expected.flags & FJ_COLOR_INPUT_IS_OUTPUT_SPACE) != 0) ||
            cctf.kind != static_cast<int>(encoding.cctf_kind) ||
            word(cctf.gamma) != word(encoding.gamma) || word(cctf.a) != word(encoding.a) ||
            word(cctf.b) != word(encoding.b) || word(cctf.c) != word(encoding.c) ||
            word(cctf.d) != word(encoding.d) || word(cctf.linearCutoff) != word(encoding.linear_cutoff) ||
            !words_equal(actual.encoding.dwgToOutput, encoding.dwg_to_output)) {
            return false;
        }
        if (!gamut) {
            return actual.outputGamutCmax == nullptr;
        }
        return actual.outputGamutCmax && actual.outputGamutCmax == deviceCmax &&
               words_equal(actual.outputGamutNativeRgbToD65Xyz, expected.native_rgb_to_d65_xyz) &&
               words_equal(actual.outputGamutD65XyzToNativeRgb, expected.d65_xyz_to_native_rgb) &&
               words_equal(actual.outputGamutLightnessKnee, expected.lightness_knee) &&
               words_equal(actual.outputGamutChromaKnee, expected.chroma_knee);
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
            FjPreparedStateView original{};
            std::array<char, 256> viewDiagnostic{};
            FjErrorBuffer errorBuffer{viewDiagnostic.data(), viewDiagnostic.size(), 0};
            ASSERT_EQ(inputs.state->view(original, errorBuffer).category, FJ_STATUS_SUCCESS);
            const auto* cmax = frame.focused_resources().outputGamutCmax;
            const auto& packed = p.scanStage.scanColor;
            const auto& expected = original.output_color;
            ASSERT_TRUE(scan_color_matches(packed, expected, cmax));
            if ((expected.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0 && expected.color_space == FJ_OUTPUT_SRGB) {
                ASSERT_FALSE(std::equal(std::begin(expected.native_rgb_to_d65_xyz), std::end(expected.native_rgb_to_d65_xyz), std::begin(expected.d65_xyz_to_native_rgb)));
                ASSERT_EQ(expected.encoding.gamma, 2.4F);
                ASSERT_EQ(expected.encoding.a, 0.0F);
                auto swapped = packed;
                std::swap(swapped.outputGamutNativeRgbToD65Xyz, swapped.outputGamutD65XyzToNativeRgb);
                EXPECT_FALSE(scan_color_matches(swapped, expected, cmax));
                auto wrongGamma = packed;
                wrongGamma.encoding.cctf.gamma = wrongGamma.encoding.cctf.a;
                EXPECT_FALSE(scan_color_matches(wrongGamma, expected, cmax));
            }
            require(frame.prepare_scan_error_stage(p.scanStage.scanErrorFlag, nullptr, error) == JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready, error.diagnostic);
            auto staged = p;
            staged.scanStage.linearRgbR = image.planes.get();
            staged.scanStage.linearRgbG = image.planes.get() + image.count;
            staged.scanStage.linearRgbB = image.planes.get() + 2 * image.count;
            launch_pipeline(staged, *frame.focused_resources().scannerSpectra);
            const auto linear = image.planes.download();
            for (float v : linear) {
                ASSERT_TRUE(std::isfinite(v));
            }
            for (bool cctf : {false, true}) {
                const OutputEncoding::Params settings{inputs.scannerColor.encoding.colorSpace, cctf, true};
                p.scanStage.scanColor.encoding = encoding(settings);
                image.clear_output();
                launch_pipeline(p, *frame.focused_resources().scannerSpectra);
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
        void check_nonfinite_spectral_output(const RouteInputs& inputs, const std::string& caller) {
            JuicerTests::GtestCohort cases(
                "spectral-transmission-" + caller,
                {"nonfinite-normalization", "omitted-overflow", "active-overflow", "zero-weight-overflow", "partial-weights"});
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
            auto nonfinite = cases.case_event("nonfinite-normalization");
            auto spectra = *frame.focused_resources().scannerSpectra;
            spectra.inverseYNormalization = std::numeric_limits<float>::infinity();
            SCOPED_TRACE(caller);
            clear_device_flag(params.scanStage.scanErrorFlag);
            image.clear_output();
            launch_pipeline(params, spectra);
            EXPECT_EQ(read_device_flag(params.scanStage.scanErrorFlag), 1);
            expect_fused_failure_output(image);

            clear_device_flag(params.scanStage.scanErrorFlag);
            scanOutput.upload(std::vector<float>(3u * image.count, 1.0f));
            launch_density_scan(
                params,
                spectra,
                densities.get(),
                densities.get() + image.count,
                densities.get() + 2u * image.count,
                scanOutput.get(),
                scanOutput.get() + image.count,
                scanOutput.get() + 2u * image.count);
            EXPECT_EQ(read_device_flag(params.scanStage.scanErrorFlag), 1);
            for (float value : scanOutput.download()) {
                EXPECT_EQ(std::bit_cast<std::uint32_t>(value), 0u);
            }
            nonfinite.passed();

            auto& color = params.scanStage.scanColor;
            params.scanStage.correctionActive = 0;
            color.outputGamutActive = 0;
            constexpr std::array identity{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
            std::copy(identity.begin(), identity.end(), color.cat02);
            std::copy(identity.begin(), identity.end(), color.xyzToRgb);
            const auto evaluate = [&](float cyan, const JuicerCuda::ScannerSpectra& samples, int expectedFlag, const std::array<double, 3>& xyz) {
                std::vector<float> input(3u * image.count);
                std::fill_n(input.begin(), image.count, cyan);
                densities.upload(input);
                clear_device_flag(params.scanStage.scanErrorFlag);
                launch_density_scan(
                    params,
                    samples,
                    densities.get(),
                    densities.get() + image.count,
                    densities.get() + 2u * image.count,
                    scanOutput.get(),
                    scanOutput.get() + image.count,
                    scanOutput.get() + 2u * image.count);
                EXPECT_EQ(read_device_flag(params.scanStage.scanErrorFlag), expectedFlag);
                const auto output = scanOutput.download();
                for (std::size_t channel = 0; channel < xyz.size(); ++channel) {
                    for (std::size_t i = 0; i < image.count; ++i) {
                        const float actual = output[channel * image.count + i];
                        if (expectedFlag == 0) {
                            EXPECT_NEAR(actual, xyz[channel], 1e-7);
                        } else {
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), 0u);
                        }
                    }
                }
            };
            spectra = {};
            spectra.inverseYNormalization = 1.0F;
            const float omitted = std::numeric_limits<float>::quiet_NaN();
            spectra.wavelengths[0] = {1.0F, 0.0F, 0.0F, 0.0F, omitted, omitted, omitted};
            spectra.wavelengths[1] = {0.0F, 0.0F, 0.0F, 0.0F, 0.25F, 0.5F, 0.75F};
            auto omittedOverflow = cases.case_event("omitted-overflow");
            evaluate(-64.0F, spectra, 0, {0.25 + 1e-10, 0.5 + 1e-10, 0.75 + 1e-10});
            omittedOverflow.passed();

            auto activeOverflow = cases.case_event("active-overflow");
            spectra.wavelengths[0].weightX = 0.25F;
            evaluate(-64.0F, spectra, 1, {});
            activeOverflow.passed();

            auto zeroWeightOverflow = cases.case_event("zero-weight-overflow");
            spectra.wavelengths[0].weightX = 0.0F;
            evaluate(-64.0F, spectra, 1, {});
            zeroWeightOverflow.passed();

            auto partialWeights = cases.case_event("partial-weights");
            spectra.wavelengths[0].weightX = 0.25F;
            spectra.wavelengths[0].weightZ = 0.5F;
            const double transmission = std::pow(10.0, -0.5);
            evaluate(0.5F, spectra, 0, {0.25 + 0.25 * transmission + 1e-10, 0.5 + 1e-10, 0.75 + 0.5 * transmission + 1e-10});
            partialWeights.passed();

            clear_device_flag(params.scanStage.scanErrorFlag);
            require(
                frame.finalize_scan_error_stage(
                    params.scanStage.scanErrorFlag,
                    nullptr,
                    error),
                error.diagnostic);
            require(frame.finish(nullptr, error), error.diagnostic);
        }

        template <typename Params>
        void check_spectral_reference(const RouteInputs& inputs) {
            constexpr std::array densities{
                std::array{0.0F, 0.0F, 0.0F},
                std::array{0.75F, 0.75F, 0.75F},
                std::array{0.04F, 0.12F, 0.19F},
                std::array{-0.15F, 0.08F, 0.41F},
                std::array{1.81F, 0.57F, 0.20F},
                std::array{1.2F, 2.2F, 1.7F},
                std::array{3.0F, 3.0F, 3.0F}};
            Image image({static_cast<int>(densities.size()), 1}, 4);
            auto frame = prepare_route(inputs, image);
            auto params = make_bound_params<Params>(inputs, frame, image);
            FjPreparedStateView state{};
            std::array<char, 256> diagnostic{};
            FjErrorBuffer statusText{diagnostic.data(), diagnostic.size(), 0};
            ASSERT_EQ(inputs.state->view(state, statusText).category, FJ_STATUS_SUCCESS);
            JuicerCuda::Failure error;
            require(frame.prepare_scan_error_stage(params.scanStage.scanErrorFlag, nullptr, error) == Frame::ScanErrorStageResult::Ready, error.diagnostic);
            const auto& spectra = *frame.focused_resources().scannerSpectra;
            params.scanStage.correctionActive = 0;
            ASSERT_EQ(params.scanStage.scanColor.outputGamutActive, 0);
            const auto originalColor = params.scanStage.scanColor;
            constexpr std::array identity{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
            std::copy(identity.begin(), identity.end(), params.scanStage.scanColor.cat02);
            std::copy(identity.begin(), identity.end(), params.scanStage.scanColor.xyzToRgb);
            Buffer carrier(3 * image.count);
            std::vector<float> packed(3 * image.count);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                for (std::size_t i = 0; i < densities.size(); ++i) {
                    packed[channel * image.count + i] = densities[i][channel];
                }
            }
            carrier.upload(packed);
            const auto scan = [&] {
                launch_density_scan(
                    params, spectra, carrier.get(), carrier.get() + image.count, carrier.get() + 2 * image.count, image.planes.get(), image.planes.get() + image.count, image.planes.get() + 2 * image.count);
                EXPECT_EQ(read_device_flag(params.scanStage.scanErrorFlag), 0);
                return image.planes.download();
            };
            const auto deviceXyz = scan();
            params.scanStage.scanColor = originalColor;
            const auto deviceRgb = scan();
            const auto& tables = state.scanner_tables;
            const auto& color = state.output_color;
            const auto sample = [](const FjFloatSpan& lane, std::size_t wavelength) {
                return static_cast<double>(lane.data[wavelength]);
            };
            for (const auto& lane : tables.dyes_cmy) {
                ASSERT_EQ(lane.count, Spectral::kNumSamples);
                ASSERT_NE(lane.data, nullptr);
            }
            for (const auto& lane : tables.weighted_xyz) {
                ASSERT_EQ(lane.count, Spectral::kNumSamples);
                ASSERT_NE(lane.data, nullptr);
            }
            if (tables.has_baseline) {
                ASSERT_EQ(tables.baseline_min.count, Spectral::kNumSamples);
                ASSERT_NE(tables.baseline_min.data, nullptr);
            }
            const double invY = tables.inv_yn;
            for (std::size_t i = 0; i < densities.size(); ++i) {
                SCOPED_TRACE(testing::Message() << "sample " << i << " CMY "
                                                << densities[i][0] << ' ' << densities[i][1] << ' ' << densities[i][2]);
                std::array<double, 3> xyz{};
                for (std::size_t wavelength = 0; wavelength < Spectral::kNumSamples; ++wavelength) {
                    const double base = tables.has_baseline ? sample(tables.baseline_min, wavelength) : 0.0;
                    const double absorption =
                        static_cast<double>(densities[i][0]) * sample(tables.dyes_cmy[0], wavelength) +
                        static_cast<double>(densities[i][1]) * sample(tables.dyes_cmy[1], wavelength) +
                        static_cast<double>(densities[i][2]) * sample(tables.dyes_cmy[2], wavelength) + base;
                    if (std::isnan(absorption)) {
                        continue;
                    }
                    const double transmission = std::pow(10.0, -absorption);
                    for (std::size_t channel = 0; channel < xyz.size(); ++channel) {
                        const double weight = sample(tables.weighted_xyz[channel], wavelength);
                        if (std::isfinite(weight)) {
                            xyz[channel] += transmission * weight;
                        }
                    }
                }
                for (std::size_t channel = 0; channel < xyz.size(); ++channel) {
                    xyz[channel] = std::max(0.0, xyz[channel] * invY) + 1e-10;
                    EXPECT_NEAR(static_cast<double>(deviceXyz[channel * image.count + i]), xyz[channel], 1e-10 + 0.01 * std::abs(xyz[channel]));
                }
                std::array<double, 3> adapted{};
                for (std::size_t channel = 0; channel < adapted.size(); ++channel) {
                    const auto* row = color.cat02 + channel * 3;
                    adapted[channel] = static_cast<double>(row[0]) * xyz[0] + static_cast<double>(row[1]) * xyz[1] + static_cast<double>(row[2]) * xyz[2];
                }
                for (std::size_t channel = 0; channel < adapted.size(); ++channel) {
                    const auto* row = color.xyz_to_rgb + channel * 3;
                    const double a = static_cast<double>(row[0]) * adapted[0];
                    const double b = static_cast<double>(row[1]) * adapted[1];
                    const double c = static_cast<double>(row[2]) * adapted[2];
                    EXPECT_NEAR(static_cast<double>(deviceRgb[channel * image.count + i]), a + b + c, 1e-6 + 1e-4 * (std::abs(a) + std::abs(b) + std::abs(c)));
                }
            }
            require(frame.finalize_scan_error_stage(params.scanStage.scanErrorFlag, nullptr, error), error.diagnostic);
            require(frame.finish(nullptr, error), error.diagnostic);
        }
        struct RouteCapture {
            SpectralWords spectra{};
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
            launch_pipeline(params, *frame.focused_resources().scannerSpectra);
            RouteCapture result;
            result.output = bit_patterns(image.output.download());
            EXPECT_EQ(read_device_flag(params.scanStage.scanErrorFlag), 0);
            result.spectra = capture_spectra(*frame.focused_resources().scannerSpectra);
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
            SpectralWords spectra{};
            Scanner::ScannerColorCorrectionDescriptor correction;
            std::array<std::uint64_t, 4> printHashes{};
            std::array<std::uintptr_t, 15> printAddresses{};
            std::array<std::uint32_t, 4> printValues{};
        };
        CorrectionCapture correction_resources(const RouteInputs& inputs, const Image& image, bool derive = true) {
            auto frame = prepare_route(inputs, image);
            CorrectionCapture result;
            result.spectra = capture_spectra(*frame.focused_resources().scannerSpectra);
            if (Spektrafilm::scan_route_is_print(
                    static_cast<Spektrafilm::ScanRoute>(inputs.route))) {
                const auto print = frame.print_resources();
                require(print.active && print.mainIlluminant, "prepared print resources remain available for correction");
                result.printHashes = {print.filmDensityTablesHash, print.profileTablesHash, print.mainIlluminantHash, print.preflashIlluminantHash};
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
                prepared.state.print.glare_hash,
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
            RouteInputs off(runtime, controls, 0, true);
            controls.values.scanner_black_correction = controls.values.scanner_white_correction = 1;
            RouteInputs held(runtime, controls, 0, true);
            const auto heldHash = held.recipeHash;
            const auto heldScannerHash = held.scannerRecipeHash;
            Image image({31, 19}, 4);
            const auto before = correction_resources(off, image);
            const auto active = correction_resources(held, image);
            EXPECT_EQ(off.scannerHash, held.scannerHash);
            EXPECT_EQ(before.spectra, active.spectra);
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
            EXPECT_EQ(restored.spectra, active.spectra);
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
                EXPECT_EQ(reused.spectra, initial.spectra);
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
            const auto spectra = capture_spectra(*first.focused_resources().scannerSpectra);
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
            EXPECT_EQ(capture_spectra(*changed.focused_resources().scannerSpectra), spectra);
            EXPECT_EQ(changedInputs.scannerHash, inputs.scannerHash);
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

    TEST_F(ScannerRoutes, NonfiniteSpectralOutputIsReportedByBothScannerCallers) {
        JuicerTests::GtestCohort cases("caller", {"direct", "print"});
        auto directRow = cases.case_event("direct");
        RouteInputs direct(runtime,
                           Spektrafilm::ScanRoute::NegativeDirectScan,
                           0,
                           false);
        check_nonfinite_spectral_output<JuicerCuda::DirectPipelineRunParams>(direct, "direct");
        directRow.passed();

        auto printRow = cases.case_event("print");
        RouteInputs print(runtime,
                          Spektrafilm::ScanRoute::PositivePrintScan,
                          0,
                          false);
        check_nonfinite_spectral_output<JuicerCuda::PrintPipelineRunParams>(print, "print");
        printRow.passed();
    }

    TEST_F(ScannerRoutes, SpectralPlaneMatchesIndependentFloat64Reference) {
        using Route = Spektrafilm::ScanRoute;
        struct Case {
            const char* id;
            Route route;
            const char* filmKey;
            const char* paperKey;
        };
        constexpr std::array rows{
            Case{"negative-direct-portra", Route::NegativeDirectScan, "kodak_portra_400", ""},
            Case{"negative-direct-ektar", Route::NegativeDirectScan, "kodak_ektar_100", ""},
            Case{"positive-direct-provia", Route::PositiveDirectScan, "fujifilm_provia_100f", ""},
            Case{"negative-print-endura", Route::NegativePrintScan, "kodak_portra_400", "kodak_portra_endura"},
            Case{"positive-print-crystal", Route::PositivePrintScan, "fujifilm_provia_100f", "fujifilm_crystal_archive_typeii"}};
        std::vector<std::string> ids;
        ids.reserve(rows.size());
        for (const auto& row : rows) {
            ids.emplace_back(row.id);
        }
        JuicerTests::GtestCohort cases("medium", ids);
        for (const auto& row : rows) {
            auto event = cases.case_event(row.id);
            SCOPED_TRACE(row.id);
            auto controls = RouteInputs::controls_for_route(row.route);
            controls.film_key = row.filmKey;
            if (Spektrafilm::scan_route_is_print(row.route)) {
                controls.print_key = row.paperKey;
                RouteInputs inputs(runtime, controls, 0, false);
                check_spectral_reference<JuicerCuda::PrintPipelineRunParams>(inputs);
            } else {
                RouteInputs inputs(runtime, controls, 0, false);
                check_spectral_reference<JuicerCuda::DirectPipelineRunParams>(inputs);
            }
            event.passed();
        }
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
            const auto otherRoute = static_cast<Spektrafilm::ScanRoute>(static_cast<unsigned>(route) ^ 1U);
            RouteInputs reporterInputs(runtime, otherRoute, 0, false);
            FjPreparedStateView producingState{};
            std::array<char, 256> text{};
            FjErrorBuffer diagnostic{text.data(), text.size(), 0};
            ASSERT_EQ(inputs.state->view(producingState, diagnostic).category, FJ_STATUS_SUCCESS);
            const bool paper = Spektrafilm::scan_route_is_print(route);
            const auto profile = paper ? producingState.print_profile_key : producingState.film_profile_key;
            const auto assetVersion = paper ? producingState.print_profile_asset_version : producingState.film_profile_asset_version;
            Image image({17, 9}, 4);
            // Product readback contract: bit 0 is scanner RGB; bit 8 is DIR,
            // with the failed DIR channels in bits 9-11.
            for (const int bits : bitsCases) {
                auto row = cases.case_event(route_case_key(route) + "/readback-" + std::to_string(bits));
                SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route) << " bits=" << bits);
                JuicerCuda::Failure error;
                auto submitted = prepare_route(inputs, image);
                const auto producing = submitted.admitted_snapshot();
                const auto epoch = submitted.admitted_context_epoch();
                int* flag = nullptr;
                ASSERT_EQ(submitted.prepare_scan_error_stage(flag, nullptr, error), JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready);
                check_cuda(cudaMemcpy(flag, &bits, sizeof(bits), cudaMemcpyHostToDevice));
                require(submitted.finalize_scan_error_stage(flag, nullptr, error), error.diagnostic);
                require(submitted.finish(nullptr, error), error.diagnostic);
                check_cuda(cudaDeviceSynchronize());

                auto next = prepare_route(reporterInputs, image);
                ASSERT_EQ(next.admitted_context_epoch(), epoch);
                ASSERT_NE(next.admitted_snapshot().snapshotId, producing.snapshotId);
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
                    const auto routeKey = std::string(Spektrafilm::scan_route_key(route));
                    if (bits == 1) {
                        const std::string expected = "earlier CUDA scan produced non-finite RGB route=" + routeKey +
                                                     (paper ? " medium=paper profile=" : " medium=film profile=") +
                                                     std::string(profile.data, profile.count) +
                                                     " profile_asset_version=" + std::to_string(assetVersion) +
                                                     " submission_id=" + std::to_string(producing.snapshotId) +
                                                     " frame_token=" + std::to_string(producing.frameToken.value) +
                                                     " scanner_hash=" + std::to_string(inputs.scannerHash) +
                                                     " context_epoch=" + std::to_string(epoch);
                        EXPECT_EQ(error.diagnostic, expected);
                    } else {
                        const std::string provenance = "component=dir route=" + routeKey + " profile=" +
                                                       std::string(producingState.film_profile_key.data, producingState.film_profile_key.count);
                        EXPECT_NE(error.diagnostic.find(provenance), std::string::npos);
                        EXPECT_NE(error.diagnostic.find("channel_mask=" + std::to_string((bits >> 9) & 7)), std::string::npos);
                    }
                    next.abort();
                    auto retry = prepare_route(reporterInputs, image);
                    require(retry.prepare_scan_error_stage(flag, nullptr, error) == Frame::ScanErrorStageResult::Ready, error.diagnostic);
                    require(retry.finalize_scan_error_stage(flag, nullptr, error), error.diagnostic);
                    require(retry.finish(nullptr, error), error.diagnostic);
                }
                // Inactive-frame failure cannot retain the prior DIR disposition.
                EXPECT_EQ(next.prepare_scan_error_stage(flag, nullptr, error), JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed);
                EXPECT_EQ(flag, nullptr);
                row.passed();
            }
        }
    }

    TEST_F(ScannerRoutes, HeldOutputEncodingsRestoreSpectralBindingAndPixels) {
        constexpr std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                    Spektrafilm::ScanRoute::NegativePrintScan,
                                    Spektrafilm::ScanRoute::PositiveDirectScan,
                                    Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size());
        for (const auto route : routes)
            ids.push_back(route_case_key(route));
        JuicerTests::GtestCohort cases("encoding-pair", ids);
        for (const auto route : routes) {
            auto row = cases.case_event(route_case_key(route));
            auto controls = RouteInputs::controls_for_route(route);
            controls.values.output_cctf_encoding = 1;
            RouteInputs encoded(runtime, controls, 0, false);
            controls.values.output_cctf_encoding = 0;
            RouteInputs linear(runtime, controls, 0, false);
            ASSERT_NE(encoded.scannerHash, linear.scannerHash);
            ASSERT_NE(encoded.scannerRecipeHash, linear.scannerRecipeHash);
            const auto render = [&](const RouteInputs& inputs) {
                return Spektrafilm::scan_route_is_print(route)
                           ? capture_route<JuicerCuda::PrintPipelineRunParams>(inputs)
                           : capture_route<JuicerCuda::DirectPipelineRunParams>(inputs);
            };
            const auto encodedBefore = render(encoded);
            const auto linearBefore = render(linear);
            const auto encodedAfter = render(encoded);
            const auto linearAfter = render(linear);
            EXPECT_EQ(encodedBefore.spectra, linearBefore.spectra);
            EXPECT_EQ(encodedAfter.spectra, encodedBefore.spectra);
            EXPECT_EQ(linearAfter.spectra, linearBefore.spectra);
            EXPECT_NE(encodedBefore.output, linearBefore.output);
            EXPECT_EQ(encodedAfter.output, encodedBefore.output);
            EXPECT_EQ(linearAfter.output, linearBefore.output);
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
    TEST_F(ScannerRoutes, SpectralBindingTracksTheSelectedMedium) {
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
            RouteInputs original(runtime, controls, 0, false);
            const bool print = Spektrafilm::scan_route_is_print(route);
            if (print)
                controls.print_key = "kodak_2383";
            else
                controls.film_key = "kodak_portra_800";
            RouteInputs changed(runtime, controls, 0, false);
            EXPECT_NE(original.scannerHash, changed.scannerHash);
            const auto before = print ? capture_route<JuicerCuda::PrintPipelineRunParams>(original) : capture_route<JuicerCuda::DirectPipelineRunParams>(original);
            const auto after = print ? capture_route<JuicerCuda::PrintPipelineRunParams>(changed) : capture_route<JuicerCuda::DirectPipelineRunParams>(changed);
            const auto restored = print ? capture_route<JuicerCuda::PrintPipelineRunParams>(original) : capture_route<JuicerCuda::DirectPipelineRunParams>(original);
            EXPECT_NE(before.spectra, after.spectra);
            EXPECT_NE(before.output, after.output);
            EXPECT_EQ(restored.spectra, before.spectra);
            EXPECT_EQ(restored.output, before.output);
            row.passed();
        }
    }

    TEST_F(ScannerRoutes, WholeSpectralPacketMatchesTheHeldPreparedMedium) {
        constexpr std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                    Spektrafilm::ScanRoute::NegativePrintScan,
                                    Spektrafilm::ScanRoute::PositiveDirectScan,
                                    Spektrafilm::ScanRoute::PositivePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size());
        for (const auto route : routes)
            ids.push_back(route_case_key(route));
        JuicerTests::GtestCohort cases("spectral-packet", ids);
        for (const auto route : routes) {
            auto row = cases.case_event(route_case_key(route));
            const RouteInputs inputs(runtime, route, 0, false);
            FjPreparedStateView held{};
            std::array<char, 256> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            ASSERT_EQ(inputs.state->view(held, error).category, FJ_STATUS_SUCCESS);
            const auto& tables = held.scanner_tables;
            for (const auto& lane : tables.dyes_cmy)
                ASSERT_EQ(lane.count, Spectral::kNumSamples);
            for (const auto& lane : tables.weighted_xyz)
                ASSERT_EQ(lane.count, Spectral::kNumSamples);
            ASSERT_EQ(tables.baseline_min.count, tables.has_baseline ? Spectral::kNumSamples : 0);
            SpectralWords expected{};
            const auto word = [](float value) {
                return std::bit_cast<std::uint32_t>(value);
            };
            for (std::size_t sample = 0; sample < Spectral::kNumSamples; ++sample) {
                const std::size_t first = 7 * sample;
                expected[first] = word(tables.dyes_cmy[0].data[sample]);
                expected[first + 1] = word(tables.dyes_cmy[1].data[sample]);
                expected[first + 2] = word(tables.dyes_cmy[2].data[sample]);
                expected[first + 3] = tables.has_baseline ? word(tables.baseline_min.data[sample]) : 0;
                expected[first + 4] = word(tables.weighted_xyz[0].data[sample]);
                expected[first + 5] = word(tables.weighted_xyz[1].data[sample]);
                expected[first + 6] = word(tables.weighted_xyz[2].data[sample]);
            }
            expected.back() = word(tables.inv_yn);
            const auto actual = Spektrafilm::scan_route_is_print(route)
                                    ? capture_route<JuicerCuda::PrintPipelineRunParams>(inputs)
                                    : capture_route<JuicerCuda::DirectPipelineRunParams>(inputs);
            EXPECT_EQ(actual.spectra, expected);
            row.passed();
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
