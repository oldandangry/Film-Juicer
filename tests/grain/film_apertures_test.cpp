#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "Cuda/JuicerCudaFilmGrainStage.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Hash.h"
#include "film_oracle.h"
#include "juicer_cuda_call_scope.h"
#include "juicer_cuda_prepared.h"

namespace {
    namespace Film = JuicerCuda::FilmGrain;
    constexpr std::uint32_t kSeed = 1764135136U;
    constexpr std::uint32_t kNextSeed = 195565897U;
    constexpr float kFraction = 0.375F;
    constexpr std::array kNet{0.15F, 0.40F, 0.8F};
    constexpr std::array kIdentity{1.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 1.F};
    constexpr float kCanary = -99.F;

    struct Aperture {
        std::string name;
        bool print = false;
        std::array<double, 4> canonical{};
        std::array<int, 4> bounds{};
        std::array<double, 2> scale{};
        bool integrated = false;
        float samplePitch = 0.F;
        std::array<double, 2> extent{};
        std::array<double, 2> step{};
        std::array<double, 2> origin{};
        std::array<int, 4> border{};
    };

    template <typename T, std::size_t N>
    void read_array(std::istream& stream, std::array<T, N>& values) {
        for (auto& value : values) {
            stream >> value;
        }
    }

    std::array<double, 2> aperture_edges(const Aperture& row, std::size_t axis, int pixel) {
        const auto low = std::floor(row.canonical[axis]);
        const auto high = std::ceil(row.canonical[axis + 2]);
        const auto index = pixel + row.bounds[axis];
        return {std::clamp(index / row.scale[axis], low, high) - low,
                std::clamp((index + 1.) / row.scale[axis], low, high) - low};
    }

    double oracle_delta(const GrainPrototype::FilmBank& bank, int x, int y, int channel) {
        const auto sample = [&](std::uint32_t seed) {
            return GrainPrototype::evaluate_sample(bank, {bank.tables.footprint, x, y, seed, channel, kNet[channel], true}).delta;
        };
        const auto time = static_cast<double>(kFraction);
        return ((1. - time) * sample(kSeed) + time * sample(kNextSeed)) / std::hypot(1. - time, time);
    }

    double expected_density(const GrainPrototype::FilmBank& bank, const Aperture& row, int x, int y, int channel) {
        if (!row.integrated) {
            const auto [left, top, right, bottom] = row.border;
            if (x < left || y < top || x >= row.bounds[2] - row.bounds[0] - right ||
                y >= row.bounds[3] - row.bounds[1] - bottom) {
                return 0.;
            }
            return oracle_delta(bank, x - left, y - top, channel);
        }
        const auto [left, right] = aperture_edges(row, 0, x);
        const auto [top, bottom] = aperture_edges(row, 1, y);
        double light = 0.;
        for (int iy = static_cast<int>(top); iy < static_cast<int>(std::ceil(bottom)); ++iy) {
            for (int ix = static_cast<int>(left); ix < static_cast<int>(std::ceil(right)); ++ix) {
                const double area = (std::min(bottom, iy + 1.) - std::max(top, static_cast<double>(iy))) *
                                    (std::min(right, ix + 1.) - std::max(left, static_cast<double>(ix)));
                light += area * std::pow(10., -oracle_delta(bank, ix, iy, channel));
            }
        }
        return -std::log10(light / ((bottom - top) * (right - left)));
    }

    FjFilmGrain wire(const Aperture& row, const GrainPrototype::FilmBank& bank) {
        FjFilmGrain input{};
        input.bank_hash = 1;
        input.sums = {bank.sums.data(), bank.sums.size()};
        input.seed = kSeed;
        input.next_seed = kNextSeed;
        input.pitch = 37.5F;
        input.temporal_fraction = kFraction;
        input.gain = 1.F;
        input.chroma_fraction = 1.F;
        input.sample_pitch = row.samplePitch;
        std::ranges::copy(row.extent, input.sample_extent);
        std::ranges::copy(row.step, input.sample_step);
        std::ranges::copy(row.origin, input.sample_origin);
        std::ranges::transform(row.border, input.calibrated_border, [](int value) {
            return static_cast<std::uint8_t>(value);
        });
        for (int record = 0; record < Film::kRecords; ++record) {
            const auto& source = bank.tables.records[record];
            auto& output = input.records[record];
            output.minimum_gross = source.minimumGross;
            output.maximum_gross = source.maximumGross;
            output.offset = source.offset;
            for (int level = 0; level < Film::kLevels; ++level) {
                output.mean_light[level] = source.levels[level].meanLight;
                output.mean_density[level] = source.levels[level].meanDensity;
                output.correlation[level] = source.levels[level].correlation;
            }
        }
        return input;
    }

    template <std::size_t N>
    FjFloatSpan span(const std::array<float, N>& values) {
        return {values.data(), N};
    }

    struct CaptureInputs {
        std::array<float, 81> cyan{1.F}, magenta{0.F, 1.F}, yellow{0.F, 0.F, 1.F}, zero{};
        std::array<float, 81> light{};
        std::array<float, 243> camera{}, dyes{};
        std::array<float, 2> axis{-6.F, 6.F};
        std::array<float, 6> film{kNet[0], kNet[1], kNet[2], kNet[0], kNet[1], kNet[2]};
        // The three separated spectral lanes give paper log exposure -filmDensity.
        std::array<float, 6> paper{0.F, 0.F, 0.F, 1.5F, 1.5F, 1.5F};

        CaptureInputs() {
            light.fill(1.F);
            camera.fill(1.F);
            dyes[0] = dyes[4] = dyes[8] = 1.F;
        }

        FjPreparedHostData prepare(const Aperture& row, const FjFilmGrain& grain, FjAutoExposure meter) const {
            FjPreparedHostData value{};
            value.route = row.print ? FJ_ROUTE_NEGATIVE_PRINT : FJ_ROUTE_NEGATIVE_DIRECT;
            value.capture_polarity = FJ_POLARITY_NEGATIVE;
            value.film_profile_key = {"portra-400-density-oracle", 25};
            value.film_profile_asset_version = 1;
            value.recipe_hash = 2 + value.route;
            value.film_exposure.input_color_space = FJ_INPUT_SRGB_REC709;
            value.film_exposure.method = FJ_RAW_MALLETT_2019;
            value.film_exposure.route_correction_scale = 1.F;
            value.film_exposure.mallett_green_midgray_scale = 1.F;
            std::ranges::copy(kIdentity, value.film_exposure.input_rgb_to_xyz);
            std::ranges::copy(kIdentity, value.film_exposure.input_xyz_adapt);
            std::ranges::copy(kIdentity, value.film_exposure.xyz_to_linear_srgb);
            value.film_exposure.sensitivity_rgb = span(camera);
            value.film_exposure.mallett_basis = span(camera);
            value.film_exposure.mallett_illuminant = span(light);
            value.film_exposure.sensitivity_hash = 4;
            value.film_development.log_exposure = span(axis);
            value.film_development.density_rgb = span(film);
            std::ranges::fill(value.film_development.gamma_rgb, 1.F);
            value.film_development.density_curves_hash = 5;
            value.film_development.hash = 6;
            value.scanner_spectra = {span(yellow), span(magenta), span(cyan), span(cyan), span(magenta), span(yellow), row.print ? span(zero) : FjFloatSpan{}, 1.F};
            value.scanner_correction = {0, 1.F, 0.F, 1.F};
            value.output_color.flags = FJ_COLOR_INPUT_IS_OUTPUT_SPACE;
            value.output_color.encoding.schema = FJ_OUTPUT_ENCODING_SCHEMA;
            std::ranges::copy(kIdentity, value.output_color.cat02);
            std::ranges::copy(kIdentity, value.output_color.xyz_to_rgb);
            std::ranges::fill(value.output_color.illuminant_xyz, 1.F);
            if (row.print) {
                value.print_profile_key = {"separated-paper", 15};
                value.print_profile_asset_version = 1;
                value.print.film_density_cmy = span(dyes);
                value.print.film_base_density = span(zero);
                value.print.sensitivity_cmy = span(dyes);
                value.print.log_exposure = span(axis);
                value.print.density_cmy = span(paper);
                value.print.main_illuminant = span(light);
                value.print.normalizer = 1.F;
                value.print.exposure = 1.F;
                value.print.film_density_hash = 7;
                value.print.profile_tables_hash = 8;
                value.print.main_illuminant_hash = 9;
            }
            value.auto_exposure = meter;
            value.film_grain = &grain;
            return value;
        }
    };

    TEST(FilmApertures, ProductionBindingAndSignedDensity) {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        const auto destroyOwner = [](FjCuda* cuda) {
            EXPECT_EQ(fj_cuda_destroy(cuda, nullptr).category, FJ_STATUS_SUCCESS);
        };
        FjCuda* rawOwner = nullptr;
        ASSERT_EQ(fj_cuda_create(&rawOwner, nullptr).category, FJ_STATUS_SUCCESS);
        const std::unique_ptr<FjCuda, decltype(destroyOwner)> owner(rawOwner, destroyOwner);
        cudaStream_t rawStream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&rawStream, cudaStreamNonBlocking), cudaSuccess);
        const auto destroyStream = [](cudaStream_t stream) {
            (void)cudaStreamDestroy(stream);
        };
        const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(destroyStream)> stream(rawStream, destroyStream);

        const auto calibrated = GrainPrototype::read_film_bank(JUICER_GRAIN_TEST_BANK, 37.5F, JUICER_GRAIN_TEST_TABLES);
        const auto full = GrainPrototype::read_film_bank(JUICER_GRAIN_TEST_BANK, 9.375F, JUICER_GRAIN_TEST_TABLES);
        std::ifstream input(JUICER_GRAIN_APERTURE_WIRE);
        std::string schema;
        ASSERT_TRUE(static_cast<bool>(input >> schema));
        ASSERT_EQ(schema, "film-aperture-geometry-v1");
        const CaptureInputs inputs;
        std::vector<float> beforeCutoff;
        std::size_t cases = 0;
        for (Aperture row; input >> row.name; ++cases) {
            SCOPED_TRACE(row.name);
            input >> row.print;
            read_array(input, row.canonical);
            read_array(input, row.bounds);
            read_array(input, row.scale);
            input >> row.integrated >> row.samplePitch;
            read_array(input, row.extent);
            read_array(input, row.step);
            read_array(input, row.origin);
            ASSERT_TRUE(static_cast<bool>(input));
            std::string trailing;
            std::getline(input, trailing);
            if (!trailing.empty()) {
                std::istringstream tail(trailing);
                read_array(tail, row.border);
                ASSERT_TRUE(static_cast<bool>(tail));
            }
            const int width = row.bounds[2] - row.bounds[0];
            const int height = row.bounds[3] - row.bounds[1];
            const auto pitch = static_cast<std::size_t>(width) * 4 + 5;
            const auto words = pitch * height;
            const auto& bank = row.integrated ? full : calibrated;
            const auto descriptor = wire(row, bank);
            constexpr float sourceWord = 0.18F;
            const std::vector<float> source(words, sourceWord);
            std::vector<float> host(words + 2, kCanary);
            float* raw = nullptr;
            const auto bytes = words * sizeof(float);
            const auto guardedBytes = host.size() * sizeof(float);
            ASSERT_EQ(cudaMalloc(&raw, bytes + guardedBytes), cudaSuccess);
            const auto deviceFree = [](float* data) {
                (void)cudaFree(data);
            };
            const std::unique_ptr<float, decltype(deviceFree)> device(raw, deviceFree);
            auto* guarded = raw + words;
            ASSERT_EQ(cudaMemcpyAsync(raw, source.data(), bytes, cudaMemcpyHostToDevice, rawStream), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(guarded, host.data(), guardedBytes, cudaMemcpyHostToDevice, rawStream), cudaSuccess);

            const auto [x1, y1, x2, y2] = row.bounds;
            const FjRect bounds{x1, y1, x2, y2};
            FjFrame frame{};
            const auto rowBytes = static_cast<std::ptrdiff_t>(pitch * sizeof(float));
            frame.source = {reinterpret_cast<std::uintptr_t>(raw), bounds, rowBytes, FJ_COMPONENTS_RGBA, FJ_DEPTH_FLOAT32};
            frame.destination = {reinterpret_cast<std::uintptr_t>(guarded + 1), bounds, frame.source.row_bytes, FJ_COMPONENTS_RGBA, FJ_DEPTH_FLOAT32};
            frame.render_window = frame.full_frame_extent = bounds;
            frame.pixel_size_um = 37.5F;
            frame.time_frames = 37.375;
            frame.frame_rate = 24.;
            frame.stream = reinterpret_cast<std::uintptr_t>(rawStream);
            frame.flags = FJ_FRAME_STREAM_PRESENT;
            const auto preview = JuicerCuda::make_auto_exposure_preview_descriptor({x1, y1, x2, y2}, {x1, y1, x2, y2}, {});
            const FjAutoExposure meter{bounds, bounds, preview.previewWidth, preview.previewHeight, 0, preview.hash};
            const auto prepared = inputs.prepare(row, descriptor, meter);
            const auto binding = Hash::hash_uint64_values({JuicerCuda::spectral_binding_digest(prepared), 0});
            const auto keys = JuicerCuda::ResourceManager::make_key_digests(prepared.recipe_hash, 0, binding, meter.hash);
            const FjSubmission submission{1, 37, cases + 1, keys.uploadCoreHash, keys.dirHash, keys.scannerHash, keys.autoExposureHash};
            FjCudaContext context{};
            std::array<char, 1024> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            std::string diagnostic;
            {
                JuicerCuda::NativeCall call(owner.get());
                ASSERT_EQ(call.inspect(&frame, &context, &error).category, FJ_STATUS_SUCCESS) << text.data();
                JuicerCuda::PendingContextLossRecovery recovery{};
                ASSERT_EQ(call.render(&context, &frame, &submission, &prepared, {}, recovery, diagnostic).status.category,
                          FJ_STATUS_SUCCESS)
                    << diagnostic;
            }
            ASSERT_EQ(cudaMemcpyAsync(host.data(), guarded, guardedBytes, cudaMemcpyDeviceToHost, rawStream), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(rawStream), cudaSuccess);
            EXPECT_EQ(host.front(), kCanary);
            EXPECT_EQ(host.back(), kCanary);
            bool changed = false;
            for (int y = 0; y < height; ++y) {
                const auto line = std::span(host).subspan(1 + static_cast<std::size_t>(y) * pitch, pitch);
                for (int x = 0; x < width; ++x) {
                    EXPECT_EQ(line[static_cast<std::size_t>(x) * 4 + 3], sourceWord);
                }
                for (const auto value : line.subspan(static_cast<std::size_t>(width) * 4)) {
                    EXPECT_EQ(value, kCanary);
                }
            }
            for (int channel = 0; channel < 3; ++channel) {
                for (int y = 0; y < height; ++y) {
                    for (int x : {0, 1, 3, 127, 255, width - 1}) {
                        const auto index = 1 + static_cast<std::size_t>(y) * pitch + 4 * static_cast<std::size_t>(x) + channel;
                        const double scannerDensity = -std::log10(static_cast<double>(host[index]) - 1e-10);
                        const double filmDensity = row.print ? (0.75 - scannerDensity) * 8. : scannerDensity;
                        const double actual = filmDensity - kNet[channel];
                        EXPECT_NEAR(actual, expected_density(bank, row, x, y, channel), 5e-5)
                            << channel << ' ' << x << ' ' << y;
                        changed |= std::abs(actual) > 5e-5;
                    }
                }
            }
            EXPECT_TRUE(changed);
            if (row.name == "cutoff-print") {
                beforeCutoff = host;
            } else if (row.name == "above-print") {
                EXPECT_NE(host, beforeCutoff);
            } else if (row.name == "restored-print") {
                EXPECT_EQ(host, beforeCutoff);
            }
        }
        EXPECT_EQ(cases, 8U);
    }
} // namespace
