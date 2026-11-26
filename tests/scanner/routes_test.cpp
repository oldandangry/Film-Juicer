#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "../resource/observed_operations.h"

#include "Cuda/Film/JuicerCudaDensityStages.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Hash.h"
#include "juicer_cuda_call_scope.h"
#include "juicer_cuda_prepared.h"

namespace {
    constexpr int kWidth = 8;
    constexpr int kHeight = 6;
    constexpr int kPitch = kWidth * 4 + 3;
    constexpr float kCanary = -9.F;
    constexpr std::array kFilmDensity{0.25F, 0.5F, 0.75F};
    constexpr std::array kPaperDensity{0.4F, 0.25F, 0.6F};
    constexpr std::array kDirStrength{0.25F, 0.5F, 0.75F};
    constexpr float kFilmSlope = 0.125F;
    constexpr float kPaperSlope = 0.0625F;
    constexpr std::array kD50{0.96422F, 1.F, 0.82521F};
    constexpr std::array kD65{0.95047F, 1.F, 1.08883F};
    constexpr std::array kIdentity{1.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 1.F};
    constexpr float kGlare = 4.F;
    constexpr std::size_t kArea = std::size_t{kWidth} * kHeight;
    constexpr std::size_t kFrameWords = std::size_t{kPitch} * kHeight;
    constexpr std::size_t kSpectralSamples = 81;
    constexpr std::size_t kDensityInput = 2 + 6 * 2 + 2 * kSpectralSamples + 3 * kArea;

    enum class Schedule : std::uint8_t {
        Source,
        Planes,
        FusedDir
    };

    struct ScanExpectation {
        std::array<float, 3> density;
        std::array<float, 3> white;
        float glare;
    };

    template <std::size_t N>
    FjFloatSpan span(const std::array<float, N>& values) {
        return {values.data(), values.size()};
    }

    FjStatus gaussian(void*, const FjGaussianKernelInput* input, FjGaussianKernelView* output, FjGaussianKernelFailure* failure, FjErrorBuffer*) {
        static constexpr std::array weights{0.25F, 0.5F, 0.25F};
        if (!input || input->radius != 1) {
            return {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        }
        *failure = {};
        *output = {input->radius, input->sigma_px, span(weights)};
        return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
    }

    struct Input {
        std::array<float, 81> cyan{1.F}, magenta{0.F, 1.F}, yellow{0.F, 0.F, 1.F};
        std::array<float, 81> light{}, zero{};
        std::array<float, 243> spectralRgb{};
        std::array<float, 2> axis{-6.F, 6.F};
        std::array<float, 6> filmCurves{}, paperCurves{};

        explicit Input(bool varying = false) {
            light.fill(1.F);
            spectralRgb.fill(1.F);
            std::ranges::copy(kFilmDensity, filmCurves.begin());
            std::ranges::copy(kFilmDensity, filmCurves.begin() + 3);
            std::ranges::copy(kPaperDensity, paperCurves.begin());
            std::ranges::copy(kPaperDensity, paperCurves.begin() + 3);
            if (varying) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    filmCurves[channel] -= 6.F * kFilmSlope;
                    filmCurves[channel + 3] += 6.F * kFilmSlope;
                    paperCurves[channel] -= 6.F * kPaperSlope;
                    paperCurves[channel + 3] += 6.F * kPaperSlope;
                }
            }
        }

        FjPreparedHostData prepared(std::uint32_t route, Schedule schedule, bool coupling = false) const {
            const bool paper = (route & 1U) != 0;
            const bool positive = (route & 2U) != 0;
            FjPreparedHostData value{};
            value.route = route;
            value.capture_polarity = positive ? FJ_POLARITY_POSITIVE : FJ_POLARITY_NEGATIVE;
            value.film_profile_key = {"producer-film", 13};
            value.film_profile_asset_version = 41;
            value.recipe_hash = 25 + route;
            value.film_exposure.input_color_space = FJ_INPUT_SRGB_REC709;
            value.film_exposure.method = FJ_RAW_MALLETT_2019;
            value.film_exposure.route_correction_scale = 1.F;
            value.film_exposure.mallett_green_midgray_scale = 1.F;
            std::ranges::copy(kIdentity, value.film_exposure.input_rgb_to_xyz);
            std::ranges::copy(kIdentity, value.film_exposure.input_xyz_adapt);
            std::ranges::copy(kIdentity, value.film_exposure.xyz_to_linear_srgb);
            value.film_exposure.sensitivity_rgb = span(spectralRgb);
            value.film_exposure.mallett_basis = span(spectralRgb);
            value.film_exposure.mallett_illuminant = span(light);
            value.film_exposure.sensitivity_hash = 7;
            value.film_development.log_exposure = span(axis);
            value.film_development.density_rgb = span(filmCurves);
            std::ranges::fill(value.film_development.gamma_rgb, 1.F);
            value.film_development.density_curves_hash = 8;
            value.film_development.hash = 9;
            value.scanner_spectra = {span(yellow), span(magenta), span(cyan), span(cyan), span(magenta), span(yellow), paper ? span(zero) : FjFloatSpan{}, 1.F};
            value.scanner_correction = {FJ_CORRECTION_ACTIVE, 0.75F, 0.10F, 1.F};
            value.output_color.flags = FJ_COLOR_ENCODE_CCTF | FJ_COLOR_INPUT_IS_OUTPUT_SPACE;
            value.output_color.encoding.schema = FJ_OUTPUT_ENCODING_SCHEMA;
            value.output_color.encoding.cctf_kind = FJ_CCTF_GAMMA;
            value.output_color.encoding.gamma = 2.F;
            std::ranges::copy(kIdentity, value.output_color.cat02);
            std::ranges::copy(kIdentity, value.output_color.xyz_to_rgb);
            std::ranges::copy(kD50, value.output_color.illuminant_xyz);
            if (paper) {
                value.print_profile_key = {"producer-paper", 14};
                value.print_profile_asset_version = 51;
                auto& print = value.print;
                print.film_density_cmy = span(spectralRgb);
                print.film_base_density = span(zero);
                print.sensitivity_cmy = span(spectralRgb);
                print.log_exposure = span(axis);
                print.density_cmy = span(paperCurves);
                print.main_illuminant = span(light);
                print.normalizer = 1.F;
                print.exposure = 1.F;
                print.film_density_hash = 10;
                print.profile_tables_hash = 11;
                print.main_illuminant_hash = 12;
            }
            if (schedule != Schedule::Source) {
                value.scanner_post.hash = 13;
                value.scanner_post.lens_blur_sigma_px = 0.5F;
                value.scanner_post.lens_blur_radius = 1;
                if (paper) {
                    value.scanner_post.flags = FJ_POST_GLARE;
                    value.scanner_post.glare_percent = kGlare;
                    value.print.glare_hash = 14;
                }
            }
            if (schedule != Schedule::Source) {
                auto& dir = value.dir_couplers;
                dir.mode = positive ? FJ_DIR_POSITIVE_RECEIVER_LANGMUIR : FJ_DIR_NEGATIVE_DONOR_LANGMUIR;
                std::ranges::fill(dir.density_max_rgb, 2.F);
                std::ranges::fill(dir.density_ref_rgb, 1.F);
                std::ranges::fill(dir.donor_k_rgb, 1.F);
                std::ranges::fill(dir.receiver_kr_rgb, 1.F);
                if (coupling) {
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        dir.matrix_rgb[4 * channel] = kDirStrength[channel];
                        dir.receiver_c_ref_rgb[channel] = 0.5F;
                    }
                }
                std::ranges::fill(dir.compensated_axes_rgb, span(axis));
                dir.compensated_axes_hash = 15;
                dir.hash = 16;
                auto& spatial = value.spatial_dir;
                spatial.hash = 17;
                spatial.recipe_hash = dir.hash;
                spatial.render_extent = spatial.full_frame_extent = spatial.filter_domain_extent = {0, 0, kWidth, kHeight};
                spatial.component_count = 1;
                auto& filter = spatial.components[0];
                filter.reference_operator = FJ_DIR_OPERATOR_FIR_REFLECT;
                filter.sigma_px = 0.5F;
                filter.radius = 1;
                filter.weight = 1.F;
            }
            if (schedule == Schedule::Planes) {
                // A film density consumer forces materialization even with spatial DIR.
                // Zero defect probabilities leave the developed density unchanged.
                auto& effects = value.effects;
                effects.flags = FJ_EFFECTS_FILM;
                effects.render_extent = effects.full_frame_extent = {0, 0, kWidth, kHeight};
                effects.sample_step_x_mm = effects.sample_step_y_mm = 1.F;
                effects.gate_width = kWidth;
                effects.gate_height = kHeight;
                effects.weave_cos_rot = 1.F;
                effects.recipe_hash = effects.hash = 18;
            }
            return value;
        }
    };

    class ProductionScan : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&source, sizeof(pixels)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&output, sizeof(pixels)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&densityStage, (3 * kArea + kDensityInput) * sizeof(float)), cudaSuccess);
            ASSERT_EQ(fj_cuda_create(&owner, nullptr).category, FJ_STATUS_SUCCESS);
            pixels.fill(0.18F);
            ASSERT_EQ(cudaMemcpy(source, pixels.data(), sizeof(pixels), cudaMemcpyHostToDevice), cudaSuccess);
            frame.source = {reinterpret_cast<std::uintptr_t>(source), {0, 0, kWidth, kHeight}, kPitch * sizeof(float), FJ_COMPONENTS_RGBA, FJ_DEPTH_FLOAT32};
            frame.destination = {reinterpret_cast<std::uintptr_t>(output), frame.source.bounds, frame.source.row_bytes, FJ_COMPONENTS_RGBA, FJ_DEPTH_FLOAT32};
            frame.render_window = frame.full_frame_extent = frame.source.bounds;
            frame.pixel_size_um = 1000.F;
            frame.frame_rate = 24.;
            frame.stream = reinterpret_cast<std::uintptr_t>(stream);
            frame.flags = FJ_FRAME_STREAM_PRESENT;
            const auto preview = JuicerCuda::make_auto_exposure_preview_descriptor({0, 0, kWidth, kHeight}, {0, 0, kWidth, kHeight}, {});
            meter = {frame.source.bounds, frame.source.bounds, preview.previewWidth, preview.previewHeight, 0, preview.hash};
            reset();
        }

        void TearDown() override {
            if (stream) {
                EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            }
            if (owner) {
                EXPECT_EQ(fj_cuda_destroy(owner, nullptr).category, FJ_STATUS_SUCCESS);
            }
            if (source) {
                EXPECT_EQ(cudaFree(source), cudaSuccess);
            }
            if (output) {
                EXPECT_EQ(cudaFree(output), cudaSuccess);
            }
            if (densityStage) {
                EXPECT_EQ(cudaFree(densityStage), cudaSuccess);
            }
            if (stream) {
                EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
            }
        }

        FjSubmission submission(const FjPreparedHostData& value, std::uint64_t id) const {
            const auto binding = Hash::hash_uint64_values({JuicerCuda::spectral_binding_digest(value), 0});
            const auto hashes = JuicerCuda::ResourceManager::make_key_digests(value.recipe_hash, value.dir_couplers.hash, binding, meter.hash);
            return {1, 0, id, hashes.uploadCoreHash, hashes.dirHash, hashes.scannerHash, hashes.autoExposureHash};
        }

        FjRenderOutcome render(FjPreparedHostData value, std::uint64_t id, FjAbortCallback abort = {}) {
            value.auto_exposure = meter;
            const auto selected = submission(value, id);
            frame.time_frames = 0.;
            FjCudaContext context{};
            std::array<char, 1024> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            JuicerCuda::NativeCall call(owner);
            const auto inspected = call.inspect(&frame, &context, &error);
            if (inspected.category != FJ_STATUS_SUCCESS) {
                diagnostic.assign(text.data(), error.length);
                return {inspected, 0};
            }
            JuicerCuda::PendingContextLossRecovery recovery{};
            const FjHostPreparation host{this, gaussian};
            return call.render(&context, &frame, &selected, &value, abort, recovery, diagnostic, &host);
        }

        void reset() {
            std::array<float, kFrameWords> initial{};
            initial.fill(kCanary);
            ASSERT_EQ(cudaMemcpyAsync(output, initial.data(), sizeof(initial), cudaMemcpyHostToDevice, stream), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        }

        void compare(const ScanExpectation& expectation, bool compareDensity = false) {
            const auto& [density, white, glare] = expectation;
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            std::array<float, kFrameWords> actual{};
            ASSERT_EQ(cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
            std::array<double, 3> xyz{};
            for (std::size_t channel = 0; channel < xyz.size(); ++channel) {
                xyz[channel] = std::pow(10., -static_cast<double>(density[channel])) + 1e-10;
            }
            const auto ratio = std::clamp(0.75 * xyz[1] + 0.10, 0., 1.) / (xyz[1] + 1e-10);
            for (std::size_t channel = 0; channel < xyz.size(); ++channel) {
                const auto linear = xyz[channel] * ratio + 0.01 * glare * white[channel];
                xyz[channel] = linear * linear;
            }
            for (int y = 0; y < kHeight; ++y) {
                const auto row = static_cast<std::size_t>(y) * kPitch;
                for (int lane = 0; lane < kPitch; ++lane) {
                    const auto index = row + lane;
                    if (lane >= kWidth * 4) {
                        EXPECT_EQ(actual[index], kCanary);
                    } else if (lane % 4 == 3) {
                        EXPECT_EQ(actual[index], pixels[index]);
                    } else {
                        EXPECT_NEAR(actual[index], xyz[lane % 4], 2e-5) << lane << ' ' << y;
                        if (compareDensity) {
                            // Undo the witness's gamma, uniform glare and
                            // affine luminance correction to observe density.
                            const auto green = index - lane % 4 + 1;
                            const auto greenLinear = std::sqrt(actual[green]) - 0.01 * glare * white[1];
                            const auto recoveredY = (greenLinear - 0.10) / 0.75;
                            const auto correction = greenLinear / recoveredY;
                            const auto measured = (std::sqrt(actual[index]) - 0.01 * glare * white[lane % 4]) / correction;
                            EXPECT_NEAR(-std::log10(measured - 1e-10), density[lane % 4], 5e-5) << lane << ' ' << y;
                        }
                    }
                }
            }
        }

        void compare_development(bool paper) {
            auto* c = densityStage;
            auto* m = c + kArea;
            auto* y = m + kArea;
            auto* axis = y + kArea;
            auto* film = axis + 2;
            auto* print = film + 6;
            auto* ones = print + 6;
            auto* zero = ones + kSpectralSamples;
            auto* camera = zero + kSpectralSamples;
            std::array<float, 3 * kArea + kDensityInput> backing{};
            backing.fill(kCanary);
            auto* values = backing.data() + 3 * kArea;
            values[0] = -6.F;
            values[1] = 6.F;
            for (std::size_t channel = 0; channel < kFilmDensity.size(); ++channel) {
                std::fill_n(values + 2 + 2 * channel, 2, kFilmDensity[channel]);
                std::fill_n(values + 8 + 2 * channel, 2, kPaperDensity[channel]);
            }
            std::fill_n(values + 14, kSpectralSamples, 1.F);
            std::fill_n(values + 14 + kSpectralSamples, kSpectralSamples, 0.F);
            std::fill_n(values + 14 + 2 * kSpectralSamples, 3 * kArea, 1.F);
            ASSERT_EQ(cudaMemcpyAsync(densityStage, backing.data(), sizeof(backing), cudaMemcpyHostToDevice, stream), cudaSuccess);
            const JuicerCuda::CameraFilmLinearExposurePlanes exposure{camera, camera + kArea, camera + 2 * kArea, kWidth};
            JuicerCuda::DirectPipelineRunParams direct{};
            direct.width = kWidth;
            direct.height = kHeight;
            direct.filmDevelop.densR = {axis, film, 2, 0, 1};
            direct.filmDevelop.densG = {axis, film + 2, 2, 0, 1};
            direct.filmDevelop.densB = {axis, film + 4, 2, 0, 1};
            JuicerCuda::PrintPipelineRunParams printRun{};
            if (paper) {
                printRun.width = kWidth;
                printRun.height = kHeight;
                printRun.filmDevelop = direct.filmDevelop;
                ASSERT_EQ(juicer_cuda_print_focused_capture_density_from_camera_film_linear(&printRun, exposure, c, m, y, stream), cudaSuccess);
            } else {
                ASSERT_EQ(juicer_cuda_direct_focused_capture_density_from_camera_film_linear(&direct, exposure, c, m, y, stream), cudaSuccess);
            }
            const auto compare = [&](const std::array<float, 3>& expected) {
                std::array<float, 3 * kArea> actual{};
                ASSERT_EQ(cudaMemcpyAsync(actual.data(), densityStage, sizeof(actual), cudaMemcpyDeviceToHost, stream), cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                for (std::size_t channel = 0; channel < expected.size(); ++channel) {
                    const auto plane = std::span(actual).subspan(channel * kArea, kArea);
                    EXPECT_TRUE(std::ranges::all_of(plane, [&](float value) {
                        return value == expected[channel];
                    })) << "density channel "
                        << channel;
                }
            };
            compare(kFilmDensity);
            if (paper) {
                printRun.printExpose.active = 1;
                printRun.printExpose.negTables = {ones, ones, ones, zero, 81};
                printRun.printExpose.printIllumFiltered = ones;
                printRun.printExpose.printIllumK = 81;
                printRun.printExpose.printSensC = printRun.printExpose.printSensM = printRun.printExpose.printSensY =
                    {nullptr, ones, 81, 0, 80};
                printRun.printDevelop.printDcC = {axis, print, 2, 0, 1};
                printRun.printDevelop.printDcM = {axis, print + 2, 2, 0, 1};
                printRun.printDevelop.printDcY = {axis, print + 4, 2, 0, 1};
                ASSERT_EQ(juicer_cuda_print_focused_continue_from_capture_density(&printRun, c, m, y, nullptr, stream), cudaSuccess);
                compare(kPaperDensity);
            }
        }

        cudaStream_t stream = nullptr;
        float* source = nullptr;
        float* output = nullptr;
        float* densityStage = nullptr;
        FjCuda* owner = nullptr;
        FjFrame frame{};
        FjAutoExposure meter{};
        std::string diagnostic;
        std::array<float, kFrameWords> pixels{};
    };

    TEST_F(ProductionScan, ExposedFilmAndPaperDevelopment) {
        compare_development(false);
        compare_development(true);
    }

    TEST_F(ProductionScan, SourcePlanesAndFusedDirForBothPolarities) {
        const Input backing(true);
        std::uint64_t id = 1;
        for (std::uint32_t route = 0; route <= FJ_ROUTE_POSITIVE_PRINT; ++route) {
            const bool paper = (route & 1U) != 0;
            const bool positive = (route & 2U) != 0;
            for (const auto schedule : {Schedule::Source, Schedule::Planes, Schedule::FusedDir}) {
                SCOPED_TRACE(testing::Message() << "route=" << route << " schedule=" << static_cast<int>(schedule));
                auto prepared = backing.prepared(route, schedule, true);
                // A uniform 0.18 RGB patch and unit spectra give 81 * 0.54
                // at each film layer. The reflected FIR preserves this patch.
                const double filmLogExposure = std::log10(81. * 0.54);
                std::array<float, 3> density{};
                for (std::size_t channel = 0; channel < density.size(); ++channel) {
                    const double uncoupled = kFilmDensity[channel] + kFilmSlope * filmLogExposure;
                    double correction = 0.;
                    if (schedule != Schedule::Source) {
                        if (positive) {
                            const auto matrixOutput = kDirStrength[channel] * (2. - uncoupled);
                            correction = 1.5 * matrixOutput / (1. + matrixOutput);
                        } else {
                            correction = kDirStrength[channel] * 2. * uncoupled / (1. + uncoupled);
                        }
                    }
                    density[channel] = static_cast<float>(uncoupled - kFilmSlope * correction);
                }
                if (paper) {
                    // Every print wavelength sees the sum of the three dye
                    // densities, so log10(81 * transmission) reduces to this.
                    const auto printLogExposure = std::log10(81.) - density[0] - density[1] - density[2];
                    for (std::size_t channel = 0; channel < density.size(); ++channel) {
                        density[channel] = static_cast<float>(kPaperDensity[channel] + kPaperSlope * printLogExposure);
                    }
                }
                const bool glare = paper && schedule != Schedule::Source;
                const auto bound = JuicerCuda::spectral_binding_digest(prepared);
                for (const auto& white : std::array{kD50, glare ? kD65 : kD50, kD50}) {
                    std::ranges::copy(white, prepared.output_color.illuminant_xyz);
                    EXPECT_EQ(JuicerCuda::spectral_binding_digest(prepared) == bound, white == kD50);
                    reset();
                    ASSERT_EQ(render(prepared, id++).status.category, FJ_STATUS_SUCCESS) << diagnostic;
                    compare({density, white, glare ? kGlare : 0.F}, true);
                    if (!glare) {
                        break;
                    }
                }
            }
        }
    }

    TEST_F(ProductionScan, DeferredFailuresReportProducingSubmissionAndRecover) {
        const Input backing;
        std::uint64_t id = 1;
        for (const bool dir : {false, true}) {
            auto producer = backing.prepared(FJ_ROUTE_POSITIVE_DIRECT, Schedule::FusedDir);
            if (dir) {
                producer.dir_couplers.receiver_c_ref_rgb[1] = 0.1F;
                producer.dir_couplers.receiver_kr_rgb[1] = 0.F;
            } else {
                producer.scanner_spectra.inverse_y_normalization = std::numeric_limits<float>::infinity();
            }
            const auto producingId = id++;
            ASSERT_EQ(render(producer, producingId).status.category, FJ_STATUS_SUCCESS) << diagnostic;
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            auto reporter = backing.prepared(FJ_ROUTE_POSITIVE_DIRECT, Schedule::FusedDir);
            reporter.film_profile_key = {"reporting-film", 14};
            reporter.film_profile_asset_version = 42;
            const auto result = render(reporter, id++);
            EXPECT_EQ(result.status.category, FJ_STATUS_PREPARATION_FAILURE) << diagnostic;
            EXPECT_EQ(result.status.api, FJ_API_NONE);
            EXPECT_EQ(result.status.native_code, 0);
            EXPECT_EQ(result.flags, dir ? FJ_RENDER_DEFERRED_DIR_ERROR : 0U);
            EXPECT_NE(diagnostic.find("profile=producer-film"), std::string::npos) << diagnostic;
            EXPECT_EQ(diagnostic.find("reporting-film"), std::string::npos) << diagnostic;
            EXPECT_NE(diagnostic.find("submission_id=" + std::to_string(producingId)), std::string::npos) << diagnostic;
            if (dir) {
                EXPECT_NE(diagnostic.find("channel_mask=2"), std::string::npos) << diagnostic;
            }
            EXPECT_EQ(render(reporter, id++).status.category, FJ_STATUS_SUCCESS) << diagnostic;
            compare({kFilmDensity, kD50, 0.F});
        }
        EXPECT_EQ(fj_cuda_retire_instance(owner, 1, nullptr).category, FJ_STATUS_SUCCESS);
        auto recovered = backing.prepared(FJ_ROUTE_POSITIVE_DIRECT, Schedule::Source);
        ASSERT_EQ(render(recovered, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        compare({kFilmDensity, kD50, 0.F});
    }

    TEST_F(ProductionScan, MaterializedDirFiltersNonuniformDensity) {
        const Input backing(true);
        auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Planes, true);
        input.scanner_post = {};
        input.scanner_correction = {0, 1.F, 0.F, 1.F};
        // Repeat a two-column bright band vertically to isolate the reflected
        // horizontal FIR. The film effects consumer still requires density planes.
        const auto exposure = [](int x) -> float {
            return x == 2 || x == 3 ? 0.65F : 0.02F;
        };
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const auto index = static_cast<std::size_t>(y) * kPitch + static_cast<std::size_t>(x) * 4;
                std::fill_n(pixels.data() + index, 3, exposure(x));
            }
        }
        ASSERT_EQ(cudaMemcpyAsync(source, pixels.data(), sizeof(pixels), cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(render(input, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        std::array<float, kFrameWords> actual{};
        ASSERT_EQ(cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
        const auto uncoupled = [&](int x, std::size_t channel) {
            return kFilmDensity[channel] + kFilmSlope * std::log10(243. * exposure(x));
        };
        const auto local = [&](int x, std::size_t channel) {
            const auto density = uncoupled(x, channel);
            return kDirStrength[channel] * 2. * density / (1. + density);
        };
        for (int y = 0; y < kHeight; ++y) {
            const auto row = std::span(actual).subspan(static_cast<std::size_t>(y) * kPitch, kPitch);
            for (int x = 0; x < kWidth; ++x) {
                const auto index = static_cast<std::size_t>(x) * 4;
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const auto filtered = 0.25 * local(std::max(x - 1, 0), channel) +
                                          0.5 * local(x, channel) +
                                          0.25 * local(std::min(x + 1, kWidth - 1), channel);
                    const auto expected = uncoupled(x, channel) - kFilmSlope * filtered;
                    const auto measured = -std::log10(std::sqrt(static_cast<double>(row[index + channel])) - 1e-10);
                    EXPECT_NEAR(measured, expected, 5e-5) << x << ' ' << y << ' ' << channel;
                    if (x == 2 || x == 3) {
                        const auto unfiltered = uncoupled(x, channel) - kFilmSlope * local(x, channel);
                        EXPECT_GT(std::abs(measured - unfiltered), 1e-3) << x << ' ' << y << ' ' << channel;
                    }
                }
                EXPECT_EQ(row[index + 3], 0.18F);
            }
            for (const auto guard : row.subspan(std::size_t{kWidth} * 4)) {
                EXPECT_EQ(guard, kCanary);
            }
        }
    }

    TEST_F(ProductionScan, CancelsAfterScanAndFencesTheAbortedFrame) {
        const Input backing;
        const auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        ASSERT_EQ(render(input, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        reset();
        unsigned queries = 0;
        const FjAbortCallback abort{
            [](void* user) -> std::uint32_t {
                return ++*static_cast<unsigned*>(user) == 3 ? FJ_ABORT_REQUESTED : FJ_ABORT_CONTINUE;
            },
            &queries};
        const auto records = JuicerTest::calls.eventRecords;
        const auto outcome = render(input, 2, abort);
        EXPECT_EQ(queries, 3U);
        ASSERT_EQ(outcome.status.category, FJ_STATUS_CANCELLED) << diagnostic;
        EXPECT_EQ(JuicerTest::calls.eventRecords, records + 1);
        compare({kFilmDensity, kD50, 0.F});
        ASSERT_EQ(render(input, 3).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        compare({kFilmDensity, kD50, 0.F});
    }

    TEST_F(ProductionScan, ImplicitStreamCompletionReportsFailureAfterCancellation) {
        const Input backing;
        const auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        frame.flags = 0;
        frame.stream = 0;
        std::uint64_t id = 1;
        ASSERT_EQ(render(input, id++).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        struct CompletionAttempt {
            cudaError_t fault = cudaSuccess;
            unsigned queries = 0;
        };
        for (auto attempt : {CompletionAttempt{}, CompletionAttempt{cudaErrorInvalidValue}}) {
            SCOPED_TRACE(attempt.fault);
            reset();
            const FjAbortCallback abort{
                [](void* user) -> std::uint32_t {
                    auto& attempt = *static_cast<CompletionAttempt*>(user);
                    if (++attempt.queries == 3) {
                        JuicerTest::calls.nextSynchronization = attempt.fault;
                        return FJ_ABORT_REQUESTED;
                    }
                    return FJ_ABORT_CONTINUE;
                },
                &attempt};
            const auto records = JuicerTest::calls.eventRecords;
            const auto syncs = JuicerTest::calls.synchronizations;
            const auto outcome = render(input, id++, abort);
            EXPECT_EQ(attempt.queries, 3U);
            EXPECT_EQ(JuicerTest::calls.eventRecords, records + 1);
            EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 1);
            EXPECT_EQ(JuicerTest::calls.nextSynchronization, cudaSuccess);
            if (attempt.fault == cudaSuccess) {
                EXPECT_EQ(outcome.status.category, FJ_STATUS_CANCELLED) << diagnostic;
                EXPECT_EQ(cudaStreamQuery(nullptr), cudaSuccess);
            } else {
                EXPECT_EQ(outcome.status.category, FJ_STATUS_CUDA_FAILURE) << diagnostic;
                EXPECT_EQ(outcome.status.api, FJ_API_CUDA_RUNTIME);
                EXPECT_EQ(outcome.status.native_code, cudaErrorInvalidValue);
                EXPECT_EQ(diagnostic, cudaGetErrorString(cudaErrorInvalidValue));
            }
            // An injected synchronization failure does not drain the real stream.
            ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            compare({kFilmDensity, kD50, 0.F});
        }
        reset();
        ASSERT_EQ(render(input, id).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        compare({kFilmDensity, kD50, 0.F});
    }

    TEST_F(ProductionScan, FrameUseFailureSurvivesAbortCleanup) {
        const Input backing;
        const auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        ASSERT_EQ(render(input, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        reset();
        unsigned queries = 0;
        const FjAbortCallback injectAfterScan{
            [](void* user) -> std::uint32_t {
                auto& count = *static_cast<unsigned*>(user);
                if (++count == 3) {
                    JuicerTest::calls.nextRecord = cudaErrorInvalidResourceHandle;
                    JuicerTest::calls.nextSynchronization = cudaErrorInvalidValue;
                }
                return 0;
            },
            &queries};
        const auto syncs = JuicerTest::calls.synchronizations;
        const auto result = render(input, 2, injectAfterScan);
        ASSERT_EQ(queries, 3U);
        EXPECT_EQ(result.status.category, FJ_STATUS_CUDA_FAILURE) << diagnostic;
        EXPECT_EQ(result.status.api, FJ_API_CUDA_RUNTIME);
        EXPECT_EQ(result.status.native_code, cudaErrorInvalidResourceHandle);
        EXPECT_NE(diagnostic.find("cudaEventRecord(frame use)"), std::string::npos) << diagnostic;
        EXPECT_NE(diagnostic.find("cudaStreamSynchronize(frame use)"), std::string::npos) << diagnostic;
        EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 2);
        EXPECT_EQ(cudaStreamQuery(stream), cudaSuccess);
        compare({kFilmDensity, kD50, 0.F});
        ASSERT_EQ(fj_cuda_retire_instance(owner, 1, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(fj_cuda_shutdown(owner, nullptr).category, FJ_STATUS_SUCCESS);
        const auto destroyed = fj_cuda_destroy(owner, nullptr);
        owner = nullptr;
        ASSERT_EQ(destroyed.category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(fj_cuda_create(&owner, nullptr).category, FJ_STATUS_SUCCESS);
        reset();
        ASSERT_EQ(render(input, 3).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        compare({kFilmDensity, kD50, 0.F});
    }

    TEST_F(ProductionScan, SyncContextLossRetiresOnlyTheFaultedEpoch) {
        namespace Registry = JuicerCuda::ResourceManager;
        const Input backing;
        auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        auto id = std::uint64_t{1};
        ASSERT_EQ(render(input, id++).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        CUcontext current = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
        ASSERT_NE(current, nullptr);
        CUcontext other = nullptr;
        ASSERT_EQ(cuCtxCreate(&other, nullptr, 0, 0), CUDA_SUCCESS);
        const auto destroyOther = [](CUcontext context) {
            EXPECT_EQ(cuCtxDestroy(context), CUDA_SUCCESS);
        };
        const std::unique_ptr<std::remove_pointer_t<CUcontext>, decltype(destroyOther)> otherContext(other, destroyOther);
        ASSERT_EQ(cuCtxSetCurrent(current), CUDA_SUCCESS);
        const Registry::DeviceContextKey key{0, current}, otherKey{0, other};
        using Fixture = std::remove_pointer_t<decltype(this)>;
        Registry::RegistryContextSnapshot otherEpoch{};
        ASSERT_TRUE(Registry::registry_begin_submission(otherKey, otherEpoch));
        ASSERT_TRUE(Registry::registry_note_submission_end(otherKey));

        enum class LossSite : std::uint8_t {
            FrameUse,
            AbortFence,
            ImplicitCompletion,
            FrameUseThenImplicitCompletion,
            PinnedOverwrite
        };
        for (const auto site : {LossSite::FrameUse, LossSite::AbortFence, LossSite::ImplicitCompletion, LossSite::FrameUseThenImplicitCompletion, LossSite::PinnedOverwrite}) {
            SCOPED_TRACE(static_cast<int>(site));
            Registry::RegistryContextSnapshot previous{};
            ASSERT_TRUE(Registry::registry_begin_submission(key, previous));
            ASSERT_TRUE(Registry::registry_note_submission_end(key));
            if (site == LossSite::PinnedOverwrite) {
                ++input.film_development.density_curves_hash;
                ++input.film_development.hash;
                ++input.recipe_hash;
            }
            reset();
            input.auto_exposure = meter;
            struct Attempt {
                Fixture* fixture = nullptr;
                const FjPreparedHostData* input = nullptr;
                FjSubmission selected{};
                std::uint64_t probeId = 0;
                LossSite site{};
                std::uint32_t retired = 0;
                void* quarantinedStaging = nullptr;
            } attempt{this, &input, submission(input, id++), id++, site};
            const auto continueCall = [](void* user, FjCudaCallAccess access) -> FjCallCompletionResult {
                auto& pending = *static_cast<Attempt*>(user);
                auto& fixture = *pending.fixture;
                std::array<char, 2048> text{};
                FjErrorBuffer error{text.data(), text.size(), 0};
                const auto admission = access.admit(access.user, &error);
                if (admission.category != FJ_STATUS_SUCCESS) {
                    return {{admission, 0}, 1};
                }
                FjCudaContext context{};
                auto frame = fixture.frame;
                const bool implicit = pending.site == LossSite::ImplicitCompletion ||
                                      pending.site == LossSite::FrameUseThenImplicitCompletion;
                if (implicit) {
                    frame.flags = 0;
                    frame.stream = 0;
                }
                const auto inspected = access.inspect(access.user, &frame, &context, &error);
                if (inspected.category != FJ_STATUS_SUCCESS) {
                    access.release(access.user);
                    return {{inspected, 0}, 1};
                }
                struct Query {
                    LossSite site = LossSite::FrameUse;
                    unsigned count = 0;
                } query{pending.site};
                FjAbortCallback abort{};
                const auto copies = JuicerTest::calls.copies;
                const auto syncs = JuicerTest::calls.synchronizations;
                const auto records = JuicerTest::calls.eventRecords;
                if (pending.site == LossSite::PinnedOverwrite) {
                    JuicerTest::calls.nextRecord = cudaErrorInvalidResourceHandle;
                    JuicerTest::calls.nextSynchronization = cudaErrorContextIsDestroyed;
                } else {
                    abort = {[](void* user) -> std::uint32_t {
                                 auto& query = *static_cast<Query*>(user);
                                 if (++query.count != 3) {
                                     return FJ_ABORT_CONTINUE;
                                 }
                                 if (query.site != LossSite::ImplicitCompletion) {
                                     JuicerTest::calls.nextRecord = cudaErrorInvalidResourceHandle;
                                 }
                                 if (query.site == LossSite::FrameUseThenImplicitCompletion) {
                                     // Rollback waits between the first fence and final completion.
                                     JuicerTest::calls.nextSynchronization = cudaErrorInvalidValue;
                                     JuicerTest::calls.scheduledSynchronization =
                                         JuicerTest::SynchronizationFault{JuicerTest::calls.synchronizations + 3, cudaErrorContextIsDestroyed};
                                     return FJ_ABORT_CONTINUE;
                                 }
                                 JuicerTest::calls.nextSynchronization = cudaErrorContextIsDestroyed;
                                 return query.site == LossSite::FrameUse ? FJ_ABORT_CONTINUE : FJ_ABORT_REQUESTED;
                             },
                             &query};
                }
                const FjHostPreparation host{&fixture, gaussian};
                const auto outcome = access.render(access.user, &context, &frame, &pending.selected, pending.input, &host, abort, &error);
                EXPECT_EQ(JuicerTest::calls.nextRecord, cudaSuccess);
                EXPECT_EQ(JuicerTest::calls.nextSynchronization, cudaSuccess);
                EXPECT_FALSE(JuicerTest::calls.scheduledSynchronization);
                if (pending.site == LossSite::PinnedOverwrite) {
                    EXPECT_NE(std::string_view(text.data(), error.length).find("pinned staging"), std::string_view::npos) << text.data();
                    EXPECT_EQ(JuicerTest::calls.copies, copies + 1);
                    pending.quarantinedStaging = JuicerTest::calls.lastStaging;
                } else {
                    EXPECT_EQ(query.count, 3U);
                    if (pending.site == LossSite::FrameUse || pending.site == LossSite::FrameUseThenImplicitCompletion) {
                        EXPECT_NE(std::string_view(text.data(), error.length).find("frame use"), std::string_view::npos) << text.data();
                    } else {
                        EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 1);
                        EXPECT_EQ(JuicerTest::calls.eventRecords, records + (implicit ? 1 : 0));
                        if (implicit) {
                            EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
                        }
                        fixture.compare({kFilmDensity, kD50, 0.F});
                    }
                }
                if (pending.site == LossSite::FrameUseThenImplicitCompletion) {
                    EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 3);
                    EXPECT_EQ(cudaStreamQuery(nullptr), cudaSuccess);
                    fixture.compare({kFilmDensity, kD50, 0.F});
                }
                access.release(access.user);
                // Ask the same root for another frame before the pending
                // completion gets the opportunity to retire this context.
                if (pending.site != LossSite::ImplicitCompletion) {
                    EXPECT_NE(fixture.render(*pending.input, pending.probeId).status.category, FJ_STATUS_SUCCESS);
                    EXPECT_NE(fixture.diagnostic.find("quarantined"), std::string::npos) << fixture.diagnostic;
                }
                const FjCallCompletionInput completion{outcome, {}, 1, 0, 1, 1, context, 0};
                return access.complete(access.user, &completion);
            };
            // The flat completion ABI fixes the order of the two opaque arguments.
            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
            const auto completeCall = [](void* user, void* scope, const FjCallCompletionInput* completion) -> FjCudaCompletion {
                auto& pending = *static_cast<Attempt*>(user);
                pending.retired = fj_cuda_call_retire_context(scope, completion);
                return {{completion->outcome, 1}, pending.retired, {}, 0};
            };
            const auto result = fj_cuda_call(owner, input.route, continueCall, completeCall, &attempt);
            const auto status = result.outcome.status;
            if (site == LossSite::AbortFence) {
                EXPECT_EQ(status.category, FJ_STATUS_CANCELLED);
                EXPECT_EQ(status.api, FJ_API_NONE);
                EXPECT_EQ(status.native_code, 0);
            } else if (site == LossSite::ImplicitCompletion) {
                EXPECT_EQ(status.category, FJ_STATUS_CONTEXT_LOSS);
                EXPECT_EQ(status.api, FJ_API_CUDA_RUNTIME);
                EXPECT_EQ(status.native_code, cudaErrorContextIsDestroyed);
            } else {
                EXPECT_EQ(status.category, FJ_STATUS_CUDA_FAILURE);
                EXPECT_EQ(status.api, FJ_API_CUDA_RUNTIME);
                EXPECT_EQ(status.native_code, cudaErrorInvalidResourceHandle);
            }
            ASSERT_EQ(attempt.retired, 3U);
            std::vector<Registry::DeviceContextKey> live;
            Registry::registry_snapshot_context_keys(live);
            EXPECT_EQ(std::ranges::find(live, key), live.end());
            EXPECT_NE(std::ranges::find(live, otherKey), live.end());
            Registry::RegistryContextSnapshot preserved{};
            ASSERT_TRUE(Registry::registry_begin_submission(otherKey, preserved));
            EXPECT_EQ(preserved.contextEpoch, otherEpoch.contextEpoch);
            ASSERT_TRUE(Registry::registry_note_submission_end(otherKey));
            if (site == LossSite::PinnedOverwrite) {
                // The fault is injected. The real stream can finish and the
                // test can free the detached block the owner had to abandon.
                ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                ASSERT_NE(attempt.quarantinedStaging, nullptr);
                EXPECT_EQ(cudaFreeHost(attempt.quarantinedStaging), cudaSuccess);
            }
            reset();
            ASSERT_EQ(render(input, id++).status.category, FJ_STATUS_SUCCESS) << diagnostic;
            compare({kFilmDensity, kD50, 0.F});
            Registry::RegistryContextSnapshot recovered{};
            ASSERT_TRUE(Registry::registry_begin_submission(key, recovered));
            EXPECT_NE(recovered.contextEpoch, previous.contextEpoch);
            ASSERT_TRUE(Registry::registry_note_submission_end(key));
        }
        Registry::RegistryContextSnapshot retiring{};
        ASSERT_TRUE(Registry::registry_begin_owner_retire(otherKey, retiring));
        EXPECT_TRUE(Registry::registry_retire(otherKey));
    }
} // namespace
