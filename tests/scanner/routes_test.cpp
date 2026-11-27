#include "Cuda/Scan/PrintDevelop.h"
#include "Cuda/Film/FilmLaunchers.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <array>
#include <bit>
#include <fstream>
#include <future>
#include <numbers>
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
#include <nlohmann/json.hpp>

#include "../resource/allocation_fault.h"
#include "../resource/observed_operations.h"

#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Hash.h"
#include "juicer_cuda_api.h"
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
            frame.effects_geometry.pixel_definition = {0, 0, kWidth, kHeight};
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
                std::array<char, 512> message{};
                FjErrorBuffer error{message.data(), message.size(), 0};
                EXPECT_EQ(fj_cuda_destroy(owner, &error).category, FJ_STATUS_SUCCESS) << message.data();
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
            const auto inspected = fj_cuda_inspect(owner, &frame, &context, &error);
            if (inspected.category != FJ_STATUS_SUCCESS) {
                diagnostic.assign(text.data(), error.length);
                return {inspected, 0, {}};
            }
            const FjHostPreparation host{this, gaussian, nullptr};
            const auto outcome = fj_cuda_render(owner, &context, &frame, &selected, &value, &host, abort, &error);
            diagnostic.assign(text.data(), error.length);
            return outcome;
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
            JuicerCuda::FilmScanRunParams direct{};
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
                ASSERT_EQ(juicer_cuda_capture_density_from_camera_film_linear(&printRun, exposure, c, m, y, stream), cudaSuccess);
            } else {
                ASSERT_EQ(juicer_cuda_capture_density_from_camera_film_linear(&direct, exposure, c, m, y, stream), cudaSuccess);
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
            EXPECT_EQ(result.flags, FJ_RENDER_ADMITTED | (dir ? FJ_RENDER_DEFERRED_DIR_ERROR : 0U));
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

    void set_camera_diffusion(FjPreparedHostData& input, FjFrame& frame) {
        const auto fixture = nlohmann::json::parse(std::ifstream(JUICER_DIFFUSION_FRAME_FIXTURE));
        const auto& cases = fixture.at("products");
        const auto row = std::ranges::find_if(cases, [](const nlohmann::json& value) {
            return value.at("input").at("name") == "product/0/family/3";
        });
        ASSERT_NE(row, cases.end());
        const auto scalar = [](const nlohmann::json& word) {
            return std::bit_cast<double>(word.get<std::uint64_t>());
        };
        const auto& camera = row->at("frame").at("camera");
        const auto& lanes = row->at("samples").at(0).at("components");
        const auto readLanes = [&](const char* name, std::span<double, 9> out) {
            std::ranges::transform(lanes.at(name), out.begin(), scalar);
        };
        input.optics.flags = FJ_OPTICS_CAMERA_DIFFUSION;
        input.optics.full_frame = {0, 0, kWidth, kHeight};
        input.optics.diffusion_hash = row->at("frame").at("hash").get<std::uint64_t>();
        auto& wire = input.optics.camera;
        wire.scatter_fraction = scalar(camera.at("scatter"));
        wire.radius_px = camera.at("sample").at("radius").get<int>();
        ASSERT_EQ(wire.radius_px, 1);
        wire.sample_hash = camera.at("sample").at("hash").get<std::uint64_t>();
        wire.hash = camera.at("hash").get<std::uint64_t>();
        readLanes("lambda", wire.lambda_px);
        readLanes("red", wire.red_weights);
        readLanes("green", wire.green_weights);
        readLanes("blue", wire.blue_weights);
        frame.pixel_size_um = static_cast<float>(scalar(camera.at("sample").at("pixel")));
    }

    TEST_F(ProductionScan, CameraDiffusionMatchesConvolutionOfTaggedComponentWords) {
        const Input backing(true);
        auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        input.scanner_correction = {0, 1.F, 0.F, 1.F};
        ASSERT_NO_FATAL_FAILURE(set_camera_diffusion(input, frame));
        const auto& wire = input.optics.camera;
        const auto exposure = [](int x, int y) -> float {
            return x == 3 && y == 2 ? 0.8F : 0.02F;
        };
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const auto index = static_cast<std::size_t>(y) * kPitch + static_cast<std::size_t>(x) * 4;
                std::fill_n(pixels.data() + index, 3, exposure(x, y));
            }
        }
        ASSERT_EQ(cudaMemcpy(source, pixels.data(), sizeof(pixels), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(render(input, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        std::array<float, kFrameWords> actual{};
        ASSERT_EQ(cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);

        // Scalar nine-tap convolution of the archived C++ component words.
        // At this radius half-sample reflection clamps a one-pixel border.
        const std::array<std::span<const double, 9>, 3> weights{
            wire.red_weights, wire.green_weights, wire.blue_weights};
        for (std::size_t channel = 0; channel < weights.size(); ++channel) {
            std::array<double, 9> taps{};
            double sum = 0.;
            for (int index = 0; index < 9; ++index) {
                const auto radius = std::hypot(index % 3 - 1, index / 3 - 1);
                double value = 0.;
                for (std::size_t lane = 0; lane < 9; ++lane) {
                    const auto lambda = wire.lambda_px[lane];
                    value += weights[channel][lane] * std::exp(-radius / lambda) /
                             (2. * std::numbers::pi * lambda * lambda);
                }
                const auto tap = static_cast<double>(static_cast<float>(value));
                taps[static_cast<std::size_t>(index)] = tap;
                sum += tap;
            }
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    double filtered = 0.;
                    for (int index = 0; index < 9; ++index) {
                        const int xx = std::clamp(x + index % 3 - 1, 0, kWidth - 1);
                        const int yy = std::clamp(y + index / 3 - 1, 0, kHeight - 1);
                        filtered += taps[static_cast<std::size_t>(index)] / sum * exposure(xx, yy);
                    }
                    const auto mixed = std::lerp(static_cast<double>(exposure(x, y)), filtered, wire.scatter_fraction);
                    const auto density = kFilmDensity[channel] + kFilmSlope * std::log10(243. * mixed);
                    const auto index = static_cast<std::size_t>(y) * kPitch + static_cast<std::size_t>(x) * 4;
                    const auto observed = -std::log10(std::sqrt(static_cast<double>(actual[index + channel])) - 1e-10);
                    EXPECT_NEAR(observed, density, 5e-5) << x << ' ' << y << ' ' << channel;
                    EXPECT_EQ(actual[index + 3], 0.18F);
                }
            }
        }
    }

    TEST_F(ProductionScan, DiffusionRollbackKeepsCufftStatusAndClosesAdmission) {
        const Input backing;
        auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        ASSERT_NO_FATAL_FAILURE(set_camera_diffusion(input, frame));
        JuicerTest::calls.nextSpectrumTransform = CUFFT_EXEC_FAILED;
        JuicerTest::calls.nextSynchronization = cudaErrorContextIsDestroyed;
        const auto failed = render(input, 1);
        EXPECT_EQ(failed.status.category, FJ_STATUS_CUFFT_FAILURE) << diagnostic;
        EXPECT_EQ(failed.status.api, FJ_API_CUFFT);
        EXPECT_EQ(failed.status.native_code, CUFFT_EXEC_FAILED);
        ASSERT_EQ(failed.flags, FJ_RENDER_ADMITTED | FJ_RENDER_CONTEXT_LOSS_PENDING) << diagnostic;
        EXPECT_EQ(failed.retirement.submission_id, 1U);
        CUcontext current = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
        EXPECT_EQ(failed.retirement.context.context, reinterpret_cast<std::uintptr_t>(current));
        const auto rejected = render(input, 2);
        EXPECT_EQ(rejected.status.category, FJ_STATUS_PREPARATION_FAILURE) << diagnostic;
        EXPECT_EQ(rejected.flags, 0U);
        EXPECT_NE(diagnostic.find("retiring"), std::string::npos);
        ASSERT_EQ(fj_cuda_retire_context(owner, &failed.retirement, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(render(input, 3).status.category, FJ_STATUS_SUCCESS) << diagnostic;
    }

    class DiffusionTrimFailure : public ProductionScan, public testing::WithParamInterface<bool> {};

    TEST_P(DiffusionTrimFailure, TrimPreservesCudaStatusAndClosesTheEpoch) {
        const Input backing;
        auto active = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        ASSERT_NO_FATAL_FAILURE(set_camera_diffusion(active, frame));
        ASSERT_EQ(render(active, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const auto inactive = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        JuicerTest::calls.deviceFreeFaultIndex = 0;
        JuicerTest::calls.deviceFreeFaults = {cudaErrorInvalidValue, cudaErrorContextIsDestroyed};
        if (GetParam()) {
            JuicerTest::calls.beforeLastDeviceFreeFault = []() noexcept {
                JuicerTest::fail_allocation_after(1);
            };
        }
        const auto failed = render(inactive, 2);
        const bool allocationFailed = JuicerTest::take_allocation_fault();
        EXPECT_EQ(JuicerTest::calls.deviceFreeFaultIndex, 2U);
        EXPECT_EQ(allocationFailed, GetParam());
        EXPECT_EQ(failed.status.category, FJ_STATUS_CUDA_FAILURE) << diagnostic;
        EXPECT_EQ(failed.status.api, FJ_API_CUDA_RUNTIME);
        EXPECT_EQ(failed.status.native_code, cudaErrorInvalidValue);
        EXPECT_NE(diagnostic.find("CUDA device buffer free failed"), std::string::npos);
        EXPECT_EQ(diagnostic.find("route=") == std::string::npos, GetParam()) << diagnostic;
        ASSERT_EQ(failed.flags, FJ_RENDER_ADMITTED | FJ_RENDER_CONTEXT_LOSS_PENDING) << diagnostic;
        EXPECT_EQ(failed.retirement.submission_id, 2U);
        const auto rejected = render(inactive, 3);
        EXPECT_EQ(rejected.flags, 0U);
        EXPECT_EQ(rejected.status.category, FJ_STATUS_PREPARATION_FAILURE) << diagnostic;
        EXPECT_NE(diagnostic.find("retiring"), std::string::npos);
        ASSERT_EQ(fj_cuda_retire_context(owner, &failed.retirement, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(render(inactive, 4).status.category, FJ_STATUS_SUCCESS) << diagnostic;
    }

    INSTANTIATE_TEST_SUITE_P(Annotations, DiffusionTrimFailure, testing::Bool());

    TEST_F(ProductionScan, CancelsBeforeAdmissionAndLeavesAnIdleEpoch) {
        const Input backing;
        const auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        frame.flags = 0;
        frame.stream = 0;
        FjCudaContext context{};
        ASSERT_EQ(fj_cuda_inspect(owner, &frame, &context, nullptr).category, FJ_STATUS_SUCCESS);
        unsigned queries = 0;
        const FjAbortCallback abort{
            [](void* user) -> std::uint32_t {
                ++*static_cast<unsigned*>(user);
                return FJ_ABORT_REQUESTED;
            },
            &queries};
        const auto copies = JuicerTest::calls.copies;
        const auto records = JuicerTest::calls.eventRecords;
        const auto syncs = JuicerTest::calls.synchronizations;
        const auto outcome = render(input, 1, abort);
        EXPECT_EQ(outcome.status.category, FJ_STATUS_CANCELLED) << diagnostic;
        EXPECT_EQ(outcome.flags, 0U);
        EXPECT_EQ(queries, 1U);
        EXPECT_EQ(JuicerTest::calls.copies, copies);
        EXPECT_EQ(JuicerTest::calls.eventRecords, records);
        EXPECT_EQ(JuicerTest::calls.synchronizations, syncs);
        const FjCudaRetirement idle{context, 0, 0};
        EXPECT_EQ(fj_cuda_retire_context(owner, &idle, nullptr).category, FJ_STATUS_SUCCESS);
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
        EXPECT_NE(outcome.flags & FJ_RENDER_ADMITTED, 0U);
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

    TEST_F(ProductionScan, ReportsDeferredDirAndTheProducingEpochWhenImplicitCompletionFails) {
        const Input backing;
        auto producer = backing.prepared(FJ_ROUTE_POSITIVE_DIRECT, Schedule::FusedDir);
        producer.dir_couplers.receiver_c_ref_rgb[1] = 0.1F;
        producer.dir_couplers.receiver_kr_rgb[1] = 0.F;
        frame.flags = 0;
        frame.stream = 0;
        ASSERT_EQ(render(producer, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        auto reporter = backing.prepared(FJ_ROUTE_POSITIVE_DIRECT, Schedule::FusedDir);
        reporter.film_profile_key = {"reporting-film", 14};
        reporter.film_profile_asset_version = 42;
        FjCudaContext epoch{};
        ASSERT_EQ(fj_cuda_inspect(owner, &frame, &epoch, nullptr).category, FJ_STATUS_SUCCESS);
        const FjAbortCallback failCompletion{
            [](void*) -> std::uint32_t {
                JuicerTest::calls.nextSynchronization = cudaErrorContextIsDestroyed;
                return FJ_ABORT_CONTINUE;
            },
            nullptr};
        const auto result = render(reporter, 2, failCompletion);
        EXPECT_EQ(result.status.category, FJ_STATUS_CONTEXT_LOSS) << diagnostic;
        EXPECT_EQ(result.status.api, FJ_API_CUDA_RUNTIME);
        EXPECT_EQ(result.status.native_code, cudaErrorContextIsDestroyed);
        EXPECT_EQ(result.flags, FJ_RENDER_ADMITTED | FJ_RENDER_DEFERRED_DIR_ERROR | FJ_RENDER_CONTEXT_LOSS_PENDING);
        EXPECT_EQ(result.retirement.context.context, epoch.context);
        EXPECT_EQ(result.retirement.context.epoch, epoch.epoch);
        EXPECT_EQ(result.retirement.instance_token, 1U);
        EXPECT_EQ(result.retirement.submission_id, 2U);
        EXPECT_EQ(render(reporter, 3).status.category, FJ_STATUS_PREPARATION_FAILURE) << diagnostic;
        ASSERT_EQ(fj_cuda_retire_context(owner, &result.retirement, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(render(reporter, 4).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        compare({kFilmDensity, kD50, 0.F});
    }

    struct StreamWork {
        std::mutex mutex;
        std::condition_variable wake;
        bool allowed = false;
        std::atomic<bool> finished{false};
        unsigned queries = 0;
        cudaStream_t stream = nullptr;
        bool cancel = false;
        bool implicit = false;

        void release() {
            {
                const std::lock_guard lock(mutex);
                allowed = true;
            }
            wake.notify_all();
        }

        ~StreamWork() {
            release();
            (void)cudaStreamSynchronize(stream);
        }

        static void CUDART_CB finish(void* user) {
            auto& work = *static_cast<StreamWork*>(user);
            std::unique_lock lock(work.mutex);
            work.finished.store(work.wake.wait_for(lock, std::chrono::seconds(10), [&] {
                return work.allowed;
            }));
        }

        static std::uint32_t after_scan(void* user) {
            auto& work = *static_cast<StreamWork*>(user);
            if (++work.queries != 3) {
                return FJ_ABORT_CONTINUE;
            }
            EXPECT_EQ(cudaLaunchHostFunc(work.stream, finish, &work), cudaSuccess);
            if (work.implicit) {
                JuicerTest::calls.beforeSynchronization = [](void* user) {
                    static_cast<StreamWork*>(user)->release();
                };
                JuicerTest::calls.beforeSynchronizationUser = &work;
            }
            return work.cancel ? FJ_ABORT_REQUESTED : FJ_ABORT_CONTINUE;
        }
    };

    TEST_F(ProductionScan, StreamPresenceControlsWhetherEnqueuedWorkMayRemainPending) {
        const Input backing;
        const auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        std::uint64_t id = 1;
        for (const auto mode : {0, 1, 2}) {
            frame.flags = mode == 0 ? 0U : FJ_FRAME_STREAM_PRESENT;
            auto* selectedStream = mode == 2 ? stream : nullptr;
            frame.stream = reinterpret_cast<std::uintptr_t>(selectedStream);
            ASSERT_EQ(render(input, id++).status.category, FJ_STATUS_SUCCESS) << diagnostic;
            ASSERT_EQ(cudaStreamSynchronize(selectedStream), cudaSuccess);
            for (const bool cancel : {false, true}) {
                reset();
                StreamWork work;
                work.stream = selectedStream;
                work.cancel = cancel;
                work.implicit = mode == 0;
                const auto waits = JuicerTest::calls.synchronizations;
                const auto outcome = render(input, id++, {StreamWork::after_scan, &work});
                EXPECT_EQ(outcome.status.category, cancel ? FJ_STATUS_CANCELLED : FJ_STATUS_SUCCESS) << diagnostic;
                EXPECT_EQ(work.queries, 3U);
                EXPECT_EQ(work.finished.load(), work.implicit);
                EXPECT_EQ(cudaStreamQuery(selectedStream), work.implicit ? cudaSuccess : cudaErrorNotReady);
                EXPECT_EQ(JuicerTest::calls.synchronizations, waits + (work.implicit ? 1 : 0));
                EXPECT_EQ(JuicerTest::calls.beforeSynchronization, nullptr);
                work.release();
                ASSERT_EQ(cudaStreamSynchronize(selectedStream), cudaSuccess);
                EXPECT_TRUE(work.finished.load());
                compare({kFilmDensity, kD50, 0.F});
            }
        }
    }

    TEST_F(ProductionScan, SameContextFramesOverlapOnDistinctStreams) {
        const Input backing;
        const auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        ASSERT_EQ(render(input, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        reset();

        cudaStream_t otherStream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&otherStream, cudaStreamNonBlocking), cudaSuccess);
        const auto destroyStream = [](cudaStream_t value) {
            EXPECT_EQ(cudaStreamDestroy(value), cudaSuccess);
        };
        const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(destroyStream)> ownedStream(otherStream, destroyStream);
        float* otherOutput = nullptr;
        ASSERT_EQ(cudaMalloc(&otherOutput, sizeof(pixels)), cudaSuccess);
        const auto freeOutput = [](float* value) {
            EXPECT_EQ(cudaFree(value), cudaSuccess);
        };
        const std::unique_ptr<float, decltype(freeOutput)> ownedOutput(otherOutput, freeOutput);
        std::array<float, kFrameWords> initial{};
        initial.fill(kCanary);
        ASSERT_EQ(cudaMemcpy(otherOutput, initial.data(), sizeof(initial), cudaMemcpyHostToDevice), cudaSuccess);

        StreamWork pending;
        pending.stream = stream;
        ASSERT_EQ(render(input, 2, {StreamWork::after_scan, &pending}).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_EQ(pending.queries, 3U);
        ASSERT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);

        const auto firstFrame = frame;
        frame.stream = reinterpret_cast<std::uintptr_t>(otherStream);
        frame.destination.address = reinterpret_cast<std::uintptr_t>(otherOutput);
        const auto second = render(input, 3);
        frame = firstFrame;
        EXPECT_EQ(second.status.category, FJ_STATUS_SUCCESS) << diagnostic;
        EXPECT_NE(second.flags & FJ_RENDER_ADMITTED, 0U);
        EXPECT_FALSE(pending.finished.load());
        EXPECT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);

        pending.release();
        ASSERT_EQ(cudaStreamSynchronize(otherStream), cudaSuccess);
        compare({kFilmDensity, kD50, 0.F});
        std::array<float, kFrameWords> first{}, next{};
        ASSERT_EQ(cudaMemcpy(first.data(), output, sizeof(first), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(next.data(), otherOutput, sizeof(next), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(first, next);
    }

    TEST_F(ProductionScan, RejectedFrameCannotDischargeAnEarlierUnfencedUse) {
        const Input backing;
        const auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        ASSERT_EQ(render(input, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        reset();

        CUcontext context = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&context), CUDA_SUCCESS);
        cudaStream_t other = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), cudaSuccess);
        const auto destroyStream = [](cudaStream_t value) {
            EXPECT_EQ(cudaStreamDestroy(value), cudaSuccess);
        };
        const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(destroyStream)> ownedStream(other, destroyStream);
        StreamWork pending;
        pending.stream = stream;
        JuicerTest::calls.failedStream = stream;
        JuicerTest::calls.streamWaitFault = cudaErrorInvalidValue;
        const FjAbortCallback faultFence{[](void* user) -> std::uint32_t {
                                             const auto next = StreamWork::after_scan(user);
                                             if (static_cast<StreamWork*>(user)->queries == 3) {
                                                 JuicerTest::calls.nextRecord = cudaErrorInvalidResourceHandle;
                                             }
                                             return next;
                                         },
                                         &pending};
        ASSERT_EQ(render(input, 2, faultFence).status.native_code, cudaErrorInvalidResourceHandle) << diagnostic;
        ASSERT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);

        frame.stream = reinterpret_cast<std::uintptr_t>(other);
        JuicerTest::calls.failedStream = other;
        const auto waits = JuicerTest::calls.synchronizations;
        const auto rejected = render(input, 3);
        EXPECT_EQ(rejected.status.category, FJ_STATUS_PREPARATION_FAILURE) << diagnostic;
        EXPECT_NE(diagnostic.find("quarantined"), std::string::npos);
        EXPECT_GT(JuicerTest::calls.synchronizations, waits) << "the rejected frame did not attempt abort fencing";
        EXPECT_EQ(cudaStreamSynchronize(other), cudaSuccess);
        EXPECT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);
        JuicerTest::calls.streamWaitFault = cudaSuccess;

        std::promise<void> barrierEntered;
        auto barrier = barrierEntered.get_future();
        auto retiring = std::async(std::launch::async, [&] {
            if (cuCtxSetCurrent(context) != CUDA_SUCCESS) {
                return false;
            }
            JuicerTest::calls.beforeContextSynchronization = [](void* user) {
                static_cast<std::promise<void>*>(user)->set_value();
            };
            JuicerTest::calls.beforeContextSynchronizationUser = &barrierEntered;
            return fj_cuda_shutdown(owner, nullptr).category == FJ_STATUS_SUCCESS;
        });
        const bool stillPending = barrier.wait_for(std::chrono::seconds(2)) == std::future_status::ready &&
                                  retiring.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout &&
                                  !pending.finished.load();
        pending.release();
        EXPECT_TRUE(retiring.get()) << "could not retire after the original stream completed";
        EXPECT_TRUE(stillPending) << "the completed rejected stream displaced the original completion obligation";
    }

    TEST_F(ProductionScan, AdmissionRevalidatesCurrentContextAndStream) {
        const Input backing;
        auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        input.auto_exposure = meter;
        const auto selected = submission(input, 1);
        FjCudaContext context{};
        ASSERT_EQ(fj_cuda_inspect(owner, &frame, &context, nullptr).category, FJ_STATUS_SUCCESS);
        CUcontext original = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&original), CUDA_SUCCESS);
        CUcontext foreign = nullptr;
        ASSERT_EQ(cuCtxCreate(&foreign, nullptr, 0, 0), CUDA_SUCCESS);
        cudaStream_t foreignStream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&foreignStream, cudaStreamNonBlocking), cudaSuccess);
        const auto copies = JuicerTest::calls.copies;
        const auto moved = fj_cuda_render(owner, &context, &frame, &selected, &input, nullptr, {}, nullptr);
        EXPECT_NE(moved.status.category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(moved.flags, 0U);
        ASSERT_EQ(cuCtxSetCurrent(original), CUDA_SUCCESS);
        auto wrong = frame;
        wrong.stream = reinterpret_cast<std::uintptr_t>(foreignStream);
        const auto result = fj_cuda_render(owner, &context, &wrong, &selected, &input, nullptr, {}, nullptr);
        EXPECT_EQ(result.status.category, FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(result.flags, 0U);
        EXPECT_EQ(JuicerTest::calls.copies, copies);
        ASSERT_EQ(cuCtxSetCurrent(foreign), CUDA_SUCCESS);
        EXPECT_EQ(cudaStreamDestroy(foreignStream), cudaSuccess);
        EXPECT_EQ(cuCtxDestroy(foreign), CUDA_SUCCESS);
        ASSERT_EQ(cuCtxSetCurrent(original), CUDA_SUCCESS);
        ASSERT_EQ(render(input, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
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

    class GrainOwnerFence : public ProductionScan, public testing::WithParamInterface<bool> {};

    TEST_P(GrainOwnerFence, PersistentFrameFenceFailureStillFencesTheGrainOwnerBeforeInstanceRetirement) {
        const Input backing;
        auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        const std::array<std::uint8_t, 1> noise{0};
        input.noise = {{noise.data(), noise.size()}, 1, 1, 1, {noise.data(), noise.size()}, {noise.data(), noise.size()}, 1, 1, 1, 1};
        auto& grain = input.grain;
        grain.flags = FJ_GRAIN_ACTIVE;
        grain.recipe_hash = 91;
        grain.render_extent = grain.full_frame_extent = {0, 0, kWidth, kHeight};
        grain.pixel_size_um = frame.pixel_size_um;
        grain.pitch_px = kHeight;
        grain.sublayer_count = 1;
        grain.debug_scale = grain.size_mix_gain = grain.size_mix_scale = grain.amplitude = 1.F;
        grain.chroma_mix = grain.chroma_independent_weight = 1.F;
        grain.size_mix_weights[0] = 1.F;
        grain.wang_cell_mm = 1.F;
        std::ranges::fill(grain.density_max_cmy, 1.F);
        std::ranges::fill(grain.n_particles_cmy, 1.F);
        std::ranges::fill(grain.od_particle_cmy, 0.1F);

        struct Injection {
            bool cancel = false;
            cudaStream_t stream = nullptr;
            unsigned queries = 0;
            std::size_t recordsAtFailure = 0;
        } injection{GetParam(), stream};
        const FjAbortCallback callback{[](void* user) -> std::uint32_t {
                                           auto& timing = *static_cast<Injection*>(user);
                                           if (++timing.queries != 3) {
                                               return FJ_ABORT_CONTINUE;
                                           }
                                           timing.recordsAtFailure = JuicerTest::calls.eventRecords;
                                           JuicerTest::calls.nextRecord = cudaErrorInvalidResourceHandle;
                                           JuicerTest::calls.failedStream = timing.stream;
                                           JuicerTest::calls.streamWaitFault = cudaErrorInvalidValue;
                                           return timing.cancel ? FJ_ABORT_REQUESTED : FJ_ABORT_CONTINUE;
                                       },
                                       &injection};
        const auto result = render(input, 1, callback);
        ASSERT_EQ(injection.queries, 3U) << diagnostic;
        EXPECT_EQ(result.status.category, injection.cancel ? FJ_STATUS_CANCELLED : FJ_STATUS_CUDA_FAILURE) << diagnostic;
        if (!injection.cancel) {
            EXPECT_EQ(result.status.native_code, cudaErrorInvalidResourceHandle);
        }
        EXPECT_EQ(JuicerTest::calls.eventRecords, injection.recordsAtFailure + 1)
            << "grain static owner has no completion event";
        EXPECT_EQ(fj_cuda_retire_instance(owner, 1, nullptr).category, FJ_STATUS_SUCCESS);
        const auto contextWaits = JuicerTest::calls.contextSynchronizations;
        EXPECT_EQ(fj_cuda_shutdown(owner, nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(JuicerTest::calls.contextSynchronizations, contextWaits + 1);
        JuicerTest::calls.streamWaitFault = cudaSuccess;
    }

    INSTANTIATE_TEST_SUITE_P(Completion, GrainOwnerFence, testing::Bool());

    TEST_F(ProductionScan, DestroySkipsAnInspectionOnlyEpochWhoseContextWasDestroyed) {
        const Input backing;
        const auto input = backing.prepared(FJ_ROUTE_NEGATIVE_DIRECT, Schedule::Source);
        ASSERT_EQ(render(input, 1).status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        CUcontext current = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
        ASSERT_NE(current, nullptr);

        CUcontext other = nullptr;
        ASSERT_EQ(cuCtxCreate(&other, nullptr, 0, 0), CUDA_SUCCESS);
        const auto destroyOther = [current](CUcontext context) {
            EXPECT_EQ(cuCtxSetCurrent(current), CUDA_SUCCESS);
            EXPECT_EQ(cuCtxDestroy(context), CUDA_SUCCESS);
        };
        std::unique_ptr<std::remove_pointer_t<CUcontext>, decltype(destroyOther)> otherContext(other, destroyOther);
        CUdeviceptr otherSource = 0, otherOutput = 0;
        ASSERT_EQ(cuMemAlloc(&otherSource, sizeof(pixels)), CUDA_SUCCESS);
        ASSERT_EQ(cuMemAlloc(&otherOutput, sizeof(pixels)), CUDA_SUCCESS);
        auto otherFrame = frame;
        otherFrame.source.address = otherSource;
        otherFrame.destination.address = otherOutput;
        otherFrame.stream = 0;
        otherFrame.flags = 0;
        FjCudaContext inspected{};
        ASSERT_EQ(fj_cuda_inspect(owner, &otherFrame, &inspected, nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(inspected.context, reinterpret_cast<std::uintptr_t>(other));
        EXPECT_NE(inspected.epoch, 0U);
        ASSERT_EQ(cuMemFree(otherOutput), CUDA_SUCCESS);
        ASSERT_EQ(cuMemFree(otherSource), CUDA_SUCCESS);
        otherContext.reset();
        CUcontext restored = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&restored), CUDA_SUCCESS);
        ASSERT_EQ(restored, current);

        std::array<char, 512> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        EXPECT_EQ(fj_cuda_destroy(std::exchange(owner, nullptr), &error).category, FJ_STATUS_SUCCESS) << message.data();
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
        const auto otherEpoch = Registry::registry_inspect_context(otherKey);
        ASSERT_NE(otherEpoch, 0U);

        enum class LossSite : std::uint8_t {
            FrameUse,
            AbortFence,
            ImplicitCompletion,
            FrameUseThenImplicitCompletion,
            PinnedOverwrite
        };
        for (const auto site : {LossSite::FrameUse, LossSite::AbortFence, LossSite::ImplicitCompletion, LossSite::FrameUseThenImplicitCompletion, LossSite::PinnedOverwrite}) {
            SCOPED_TRACE(static_cast<int>(site));
            const auto previous = Registry::registry_inspect_context(key);
            ASSERT_NE(previous, 0U);
            if (site == LossSite::PinnedOverwrite) {
                ++input.film_development.density_curves_hash;
                ++input.film_development.hash;
                ++input.recipe_hash;
            }
            reset();
            input.auto_exposure = meter;
            auto inputFrame = frame;
            const bool implicit = site == LossSite::ImplicitCompletion || site == LossSite::FrameUseThenImplicitCompletion;
            if (implicit) {
                inputFrame.flags = 0;
                inputFrame.stream = 0;
            }
            std::array<char, 2048> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            FjCudaContext context{};
            ASSERT_EQ(fj_cuda_inspect(owner, &inputFrame, &context, &error).category, FJ_STATUS_SUCCESS);
            const auto selected = submission(input, id++);
            struct Query {
                LossSite site = LossSite::FrameUse;
                unsigned count = 0;
            } query{site};
            FjAbortCallback abort{};
            const auto copies = JuicerTest::calls.copies;
            const auto syncs = JuicerTest::calls.synchronizations;
            const auto records = JuicerTest::calls.eventRecords;
            if (site == LossSite::PinnedOverwrite) {
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
            const FjHostPreparation host{this, gaussian, nullptr};
            const auto outcome = fj_cuda_render(owner, &context, &inputFrame, &selected, &input, &host, abort, &error);
            EXPECT_EQ(JuicerTest::calls.nextRecord, cudaSuccess);
            EXPECT_EQ(JuicerTest::calls.nextSynchronization, cudaSuccess);
            EXPECT_FALSE(JuicerTest::calls.scheduledSynchronization);
            if (site == LossSite::PinnedOverwrite) {
                EXPECT_NE(std::string_view(text.data(), error.length).find("pinned staging"), std::string_view::npos) << text.data();
                EXPECT_EQ(JuicerTest::calls.copies, copies + 1);
                EXPECT_NE(JuicerTest::calls.lastStaging, nullptr);
            } else {
                EXPECT_EQ(query.count, 3U);
                if (site == LossSite::FrameUse || site == LossSite::FrameUseThenImplicitCompletion) {
                    EXPECT_NE(std::string_view(text.data(), error.length).find("frame use"), std::string_view::npos) << text.data();
                } else {
                    EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 1);
                    EXPECT_EQ(JuicerTest::calls.eventRecords, records + (implicit ? 1 : 0));
                    if (implicit) {
                        EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
                    }
                    compare({kFilmDensity, kD50, 0.F});
                }
            }
            if (site == LossSite::FrameUseThenImplicitCompletion) {
                EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 3);
                EXPECT_EQ(cudaStreamQuery(nullptr), cudaSuccess);
                compare({kFilmDensity, kD50, 0.F});
            }
            ASSERT_NE(outcome.flags & FJ_RENDER_CONTEXT_LOSS_PENDING, 0U);
            EXPECT_NE(outcome.flags & FJ_RENDER_ADMITTED, 0U);
            EXPECT_EQ(outcome.retirement.context.device_id, context.device_id);
            EXPECT_EQ(outcome.retirement.context.context, context.context);
            EXPECT_EQ(outcome.retirement.context.epoch, context.epoch);
            EXPECT_EQ(outcome.retirement.instance_token, selected.instance_token);
            EXPECT_EQ(outcome.retirement.submission_id, selected.submission_id);
            // Admission is already closed for this epoch, before retirement.
            EXPECT_EQ(render(input, id++).status.category, FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_NE(diagnostic.find("retiring"), std::string::npos) << diagnostic;
            const FjCudaRetirement wrongId{context, selected.instance_token, selected.submission_id + 1};
            EXPECT_NE(fj_cuda_retire_context(owner, &wrongId, nullptr).category, FJ_STATUS_SUCCESS);
            ASSERT_EQ(fj_cuda_retire_context(owner, &outcome.retirement, nullptr).category, FJ_STATUS_SUCCESS);
            const auto status = outcome.status;
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
            std::vector<Registry::DeviceContextKey> live;
            Registry::registry_snapshot_context_keys(live);
            EXPECT_EQ(std::ranges::find(live, key), live.end());
            EXPECT_NE(std::ranges::find(live, otherKey), live.end());
            EXPECT_EQ(Registry::registry_inspect_context(otherKey), otherEpoch);
            reset();
            ASSERT_EQ(render(input, id++).status.category, FJ_STATUS_SUCCESS) << diagnostic;
            compare({kFilmDensity, kD50, 0.F});
            const auto recovered = Registry::registry_inspect_context(key);
            EXPECT_NE(recovered, previous);
            const auto before = JuicerTest::calls.copies;
            const auto lateSelection = submission(input, id++);
            for (const auto* delayed : {&selected, &lateSelection}) {
                const auto stale = fj_cuda_render(owner, &context, &inputFrame, delayed, &input, &host, {}, nullptr);
                EXPECT_EQ(stale.status.category, FJ_STATUS_PREPARATION_FAILURE);
                EXPECT_EQ(stale.flags, 0U);
            }
            EXPECT_EQ(JuicerTest::calls.copies, before);
            for (int retry = 0; retry < 2; ++retry) {
                EXPECT_NE(fj_cuda_retire_context(owner, &outcome.retirement, nullptr).category, FJ_STATUS_SUCCESS);
            }
            EXPECT_EQ(Registry::registry_inspect_context(key), recovered);
            ASSERT_EQ(render(input, id - 2).status.category, FJ_STATUS_SUCCESS) << diagnostic;
            compare({kFilmDensity, kD50, 0.F});
        }
        const FjCudaRetirement otherIdentity{{0, reinterpret_cast<std::uintptr_t>(other), otherEpoch}, 0, 0};
        EXPECT_EQ(fj_cuda_retire_context(owner, &otherIdentity, nullptr).category, FJ_STATUS_SUCCESS);
    }
} // namespace
