#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>

#include <gtest/gtest.h>

#include "juicer_cuda_prepared.h"
#include "Cuda/JuicerCudaHostViews.h"

namespace {

    const std::array<float, 243> kSamples{};
    const std::array<float, 2> kAxis{};
    const std::array<std::uint8_t, 16> kNoise{};

    FjPreparedHostData direct_input() {
        FjPreparedHostData source{};
        source.route = FJ_ROUTE_NEGATIVE_DIRECT;
        source.capture_polarity = FJ_POLARITY_NEGATIVE;
        source.film_exposure.method = FJ_RAW_MALLETT_2019;
        source.film_exposure.sensitivity_rgb = {kSamples.data(), 243};
        source.film_exposure.mallett_illuminant = {kSamples.data(), 81};
        source.film_exposure.mallett_basis = {kSamples.data(), 243};
        source.film_development.log_exposure = {kAxis.data(), 2};
        source.film_development.density_rgb = {kSamples.data(), 6};
        source.scanner_spectra = {{kSamples.data(), 81}, {kSamples.data(), 81}, {kSamples.data(), 81}, {kSamples.data(), 81}, {kSamples.data(), 81}, {kSamples.data(), 81}, {}, 1.0f};
        return source;
    }

    FjPreparedHostData print_input() {
        FjPreparedHostData source = direct_input();
        source.route = FJ_ROUTE_NEGATIVE_PRINT;
        auto& print = source.print;
        print.film_density_cmy = {kSamples.data(), 243};
        print.film_base_density = {kSamples.data(), 81};
        print.sensitivity_cmy = {kSamples.data(), 243};
        print.log_exposure = {kAxis.data(), 2};
        print.density_cmy = {kSamples.data(), 6};
        print.main_illuminant = {kSamples.data(), 81};
        print.flags = FJ_PRINT_PREFLASH;
        print.preflash_illuminant = {kSamples.data(), 81};
        return source;
    }

    void enable_grain(FjPreparedHostData& source, bool layers) {
        source.grain.flags = FJ_GRAIN_ACTIVE | (layers ? FJ_GRAIN_SUBLAYERS : 0u);
        source.grain.hash = 11;
        source.grain.recipe_hash = 12;
        source.noise = {{kNoise.data(), 8}, 2, 2, 2, {kNoise.data(), 4}, {kNoise.data(), 16}, 2, 2, 1, 2};
        if (layers) {
            source.film_development.density_layers_hash = 13;
            source.grain.density_layers_hash = 13;
            for (auto& layer : source.film_development.density_layers) {
                for (auto& channel : layer) {
                    channel = {kAxis.data(), 2};
                }
            }
        }
    }

    void expect_rejected(const FjPreparedHostData& source, const char* field) {
        JuicerCuda::PreparedDescriptors descriptors;
        std::string diagnostic;
        ASSERT_TRUE(JuicerCuda::decode_prepared_descriptors(source, descriptors, diagnostic)) << diagnostic;
        const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
        const std::optional<ScatterHalationFrameDescriptor> halation;
        const Spektrafilm::FilmJuicerEffectsGeometry geometry;
        const JuicerCuda::AutoExposurePreviewDescriptor meter;
        const JuicerCuda::ExecutionFrame frame{
            {}, {}, {}, nullptr, nullptr, nullptr, 0, 0, 3, nullptr, diffusion, halation, geometry, 0, 0, 0, 1, 0, meter, false, false};
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
        JuicerCuda::PendingContextLossRecovery recovery;
        // No Root, inspected context, image, or GPU is created. Every case must
        // fail span admission before any sample read or executor invocation.
        EXPECT_FALSE(JuicerCuda::execute_prepared_host_data(source, frame, snapshot, recovery, {}, diagnostic));
        EXPECT_EQ(diagnostic, std::string("MalformedPreparedHostData field=") + field);
        EXPECT_FALSE(recovery.pending);
    }

    TEST(PreparedSpans, NativeRowsAndFlatCStoragePreserveChannelOrderAcrossRows) {
        const std::array<std::array<float, 3>, 4> rows{{{1.0f, 2.0f, 3.0f},
                                                        {11.0f, 12.0f, 13.0f},
                                                        {21.0f, 22.0f, 23.0f},
                                                        {31.0f, 32.0f, 33.0f}}};
        const std::array<float, 12> scalars{
            1.0f, 2.0f, 3.0f, 11.0f, 12.0f, 13.0f, 21.0f, 22.0f, 23.0f, 31.0f, 32.0f, 33.0f};
        const JuicerCuda::ThreeChannelSamplesView native{std::span<const std::array<float, 3>>(rows)};
        const JuicerCuda::ThreeChannelSamplesView foreign{std::span<const float>(scalars)};
        // The C pointer carries the row object representation without forming or
        // indexing a float span across separate native row subobjects.
        const FjFloatSpan transported{reinterpret_cast<const float*>(native.object_bytes().data()), native.scalar_count()};
        const JuicerCuda::ThreeChannelSamplesView foreignRows{std::span<const std::byte>{
            reinterpret_cast<const std::byte*>(transported.data), transported.count * sizeof(float)}};
        EXPECT_EQ(native.sample_count(), 4u);
        EXPECT_EQ(native.scalar_count(), 12u);
        EXPECT_EQ(foreign.sample_count(), 4u);
        EXPECT_EQ(foreign.scalar_count(), 12u);
        EXPECT_EQ(foreignRows.sample_count(), 4u);
        EXPECT_EQ(foreignRows.scalar_count(), 12u);
        EXPECT_EQ(foreignRows.object_bytes().data(), native.object_bytes().data());
        EXPECT_EQ(native.sample(1), (std::array<float, 3>{11.0f, 12.0f, 13.0f}));
        EXPECT_EQ(native.sample(3), (std::array<float, 3>{31.0f, 32.0f, 33.0f}));
        EXPECT_EQ(foreign.sample(1), native.sample(1));
        EXPECT_EQ(foreign.sample(3), native.sample(3));
        EXPECT_EQ(foreignRows.sample(1), native.sample(1));
        EXPECT_EQ(foreignRows.sample(3), native.sample(3));
    }

    TEST(PreparedSpans, FilmAndScannerRequireExactCountsAndNonNullStorage) {
        struct Case {
            const char* field;
            FjFloatSpan& (*select)(FjPreparedHostData&);
        };
        const Case cases[] = {
            {"film_exposure.sensitivity_rgb", [](auto& value) -> FjFloatSpan& {
                 return value.film_exposure.sensitivity_rgb;
             }},
            {"film_exposure.mallett_illuminant", [](auto& value) -> FjFloatSpan& {
                 return value.film_exposure.mallett_illuminant;
             }},
            {"film_exposure.mallett_basis", [](auto& value) -> FjFloatSpan& {
                 return value.film_exposure.mallett_basis;
             }},
            {"film_development.log_exposure", [](auto& value) -> FjFloatSpan& {
                 return value.film_development.log_exposure;
             }},
            {"film_development.density_rgb", [](auto& value) -> FjFloatSpan& {
                 return value.film_development.density_rgb;
             }},
            {"scanner_spectra", [](auto& value) -> FjFloatSpan& {
                 return value.scanner_spectra.dye_c;
             }},
            {"scanner_spectra", [](auto& value) -> FjFloatSpan& {
                 return value.scanner_spectra.dye_m;
             }},
            {"scanner_spectra", [](auto& value) -> FjFloatSpan& {
                 return value.scanner_spectra.dye_y;
             }},
            {"scanner_spectra", [](auto& value) -> FjFloatSpan& {
                 return value.scanner_spectra.weighted_x;
             }},
            {"scanner_spectra", [](auto& value) -> FjFloatSpan& {
                 return value.scanner_spectra.weighted_y;
             }},
            {"scanner_spectra", [](auto& value) -> FjFloatSpan& {
                 return value.scanner_spectra.weighted_z;
             }}};
        for (const auto& item : cases) {
            SCOPED_TRACE(item.field);
            auto source = direct_input();
            item.select(source).data = nullptr;
            expect_rejected(source, item.field);
            source = direct_input();
            if (&item.select(source) == &source.film_development.log_exposure) {
                source.film_development.density_rgb.count = 9;
                expect_rejected(source, "film_development.density_rgb");
            } else {
                --item.select(source).count;
                expect_rejected(source, item.field);
            }
        }
    }

    TEST(PreparedSpans, LargeSelectedTablesRejectShapeWithoutReadingStorage) {
        auto source = direct_input();
        source.film_exposure.method = FJ_RAW_HANATOS_2025;
        source.film_exposure.mallett_basis = {};
        source.film_exposure.mallett_illuminant = {};
        // The deliberately wrong counts are rejected before reading. No realistic
        // TC/gamut allocation is needed to exercise the metadata contract.
        source.film_exposure.tc_lut_rgba = {kSamples.data(), std::size_t{192} * 192u * 4u - 1u};
        expect_rejected(source, "film_exposure.tc_lut_rgba");
        source.film_exposure.tc_lut_rgba = {nullptr, std::size_t{192} * 192u * 4u};
        expect_rejected(source, "film_exposure.tc_lut_rgba");
        source = direct_input();
        source.output_color.flags = FJ_COLOR_GAMUT_COMPRESSION;
        source.output_color.gamut_cmax = {kSamples.data(), std::size_t{64} * 720u - 1u};
        expect_rejected(source, "output_color.gamut_cmax");
        source.output_color.gamut_cmax = {nullptr, std::size_t{64} * 720u};
        expect_rejected(source, "output_color.gamut_cmax");
    }

    TEST(PreparedSpans, EmptySpansRequireNullAndOptionalBaselineRequires81Samples) {
        auto source = direct_input();
        source.film_exposure.tc_lut_rgba = {kSamples.data(), 0};
        expect_rejected(source, "film_exposure.tc_lut_rgba");
        source = direct_input();
        source.scanner_spectra.base_density = {kSamples.data(), 0};
        expect_rejected(source, "scanner_spectra.base_density");
        source.scanner_spectra.base_density = {kSamples.data(), 80};
        expect_rejected(source, "scanner_spectra.base_density");
        source.scanner_spectra.base_density = {nullptr, 81};
        expect_rejected(source, "scanner_spectra.base_density");
        source = direct_input();
        source.output_color.gamut_cmax = {kSamples.data(), 0};
        expect_rejected(source, "output_color.gamut_cmax");
        source = direct_input();
        source.film_profile_key = {"", 0};
        expect_rejected(source, "film_profile_key");
        source.film_profile_key = {nullptr, 1};
        expect_rejected(source, "film_profile_key");
        source = direct_input();
        source.print_profile_key = {"", 0};
        expect_rejected(source, "print_profile_key");
    }

    TEST(PreparedSpans, ExposureAxisCountsStayPositiveAndRepresentable) {
        auto source = direct_input();
        source.film_development.log_exposure = {};
        expect_rejected(source, "film_development.log_exposure.count");
        source.film_development.log_exposure = {kAxis.data(), std::numeric_limits<std::size_t>::max()};
        expect_rejected(source, "film_development.log_exposure.count");
        source = print_input();
        source.print.log_exposure = {};
        expect_rejected(source, "print.log_exposure.count");
        source.print.log_exposure = {kAxis.data(), std::numeric_limits<std::size_t>::max()};
        expect_rejected(source, "print.log_exposure.count");
    }

    TEST(PreparedSpans, EveryLayerAndDirAxisObeysItsActivationAndSampleCount) {
        for (std::size_t layer = 0; layer < 3; ++layer) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                SCOPED_TRACE(layer * 3u + channel);
                auto source = direct_input();
                source.film_development.density_layers[layer][channel] = {kAxis.data(), 2};
                expect_rejected(source, "film_development.density_layers");
                source = direct_input();
                enable_grain(source, true);
                source.film_development.density_layers[layer][channel].count = 1;
                expect_rejected(source, "film_development.density_layers");
                source.film_development.density_layers[layer][channel] = {nullptr, 2};
                expect_rejected(source, "film_development.density_layers");
            }
        }
        for (std::size_t channel = 0; channel < 3; ++channel) {
            auto source = direct_input();
            source.dir_couplers.compensated_axes_rgb[channel] = {kAxis.data(), 2};
            expect_rejected(source, "dir_couplers.compensated_axes_rgb");
            source = direct_input();
            source.dir_couplers.mode = FJ_DIR_NEGATIVE_DONOR_LANGMUIR;
            for (auto& axis : source.dir_couplers.compensated_axes_rgb) {
                axis = {kAxis.data(), 2};
            }
            source.dir_couplers.compensated_axes_rgb[channel].count = 1;
            expect_rejected(source, "dir_couplers.compensated_axes_rgb");
            source.dir_couplers.compensated_axes_rgb[channel] = {nullptr, 2};
            expect_rejected(source, "dir_couplers.compensated_axes_rgb");
        }
    }

    TEST(PreparedSpans, PrintTablesAndIlluminantsRequireExactStorage) {
        struct Case {
            const char* field;
            FjFloatSpan FjPrint::* member;
        };
        const Case cases[] = {
            {"print.film_density_cmy", &FjPrint::film_density_cmy},
            {"print.film_base_density", &FjPrint::film_base_density},
            {"print.sensitivity_cmy", &FjPrint::sensitivity_cmy},
            {"print.log_exposure", &FjPrint::log_exposure},
            {"print.density_cmy", &FjPrint::density_cmy},
            {"print.main_illuminant", &FjPrint::main_illuminant},
            {"print.preflash_illuminant", &FjPrint::preflash_illuminant}};
        for (const auto& item : cases) {
            SCOPED_TRACE(item.field);
            auto source = print_input();
            (source.print.*item.member).data = nullptr;
            expect_rejected(source, item.field);
            source = print_input();
            if (item.member == &FjPrint::log_exposure) {
                source.print.density_cmy.count = 9;
                expect_rejected(source, "print.density_cmy");
            } else {
                --(source.print.*item.member).count;
                expect_rejected(source, item.field);
            }
            source = direct_input();
            source.print.*item.member = {kSamples.data(), 1};
            expect_rejected(source, item.field);
        }
    }

    TEST(PreparedSpans, NoiseRequiresExactCheckedProductsAndPointers) {
        struct Case {
            const char* field;
            FjByteSpan FjStaticNoise::* member;
        };
        const Case cases[] = {{"noise.stbn", &FjStaticNoise::stbn},
                              {"noise.wang_tiles", &FjStaticNoise::wang_tiles},
                              {"noise.wang_lut", &FjStaticNoise::wang_lut}};
        for (const auto& item : cases) {
            auto source = direct_input();
            enable_grain(source, false);
            --(source.noise.*item.member).count;
            expect_rejected(source, item.field);
            source = direct_input();
            enable_grain(source, false);
            (source.noise.*item.member).data = nullptr;
            expect_rejected(source, item.field);
            source = direct_input();
            source.noise.*item.member = {kNoise.data(), 0};
            expect_rejected(source, item.field);
        }
        auto source = direct_input();
        enable_grain(source, false);
        source.noise.stbn_width = std::numeric_limits<int>::max();
        source.noise.stbn_height = std::numeric_limits<int>::max();
        source.noise.stbn_frames = std::numeric_limits<int>::max();
        expect_rejected(source, "noise.dimensions");
        source = direct_input();
        enable_grain(source, false);
        source.noise.wang_colors = std::numeric_limits<int>::max();
        expect_rejected(source, "noise.dimensions");
        source = direct_input();
        enable_grain(source, false);
        source.noise.wang_tile_count = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1u;
        expect_rejected(source, "noise.dimensions");
    }

    TEST(PreparedSpans, InactiveDependenciesAndUnknownFilmTagsAreRejected) {
        auto source = direct_input();
        source.dir_couplers.hash = 1;
        expect_rejected(source, "inactive_family");
        source = direct_input();
        source.dir_couplers.compensated_axes_hash = 1;
        expect_rejected(source, "inactive_family");
        source = direct_input();
        source.film_development.density_layers_hash = 1;
        expect_rejected(source, "inactive_family");
        source = direct_input();
        source.print.flags = FJ_PRINT_PREFLASH;
        expect_rejected(source, "inactive_family");
        source = direct_input();
        source.noise.stbn_width = 1;
        expect_rejected(source, "inactive_noise");
        source = direct_input();
        source.film_exposure.flags = 8;
        expect_rejected(source, "film_or_print_tag");
        source = direct_input();
        source.film_exposure.method = FJ_RAW_ARCTIC_2026_BETA04 + 1u;
        expect_rejected(source, "film_or_print_tag");
        source = direct_input();
        source.film_exposure.input_color_space = FJ_INPUT_SRGB_REC709 + 1u;
        expect_rejected(source, "film_or_print_tag");
        source = direct_input();
        source.dir_couplers.mode = FJ_DIR_POSITIVE_RECEIVER_LANGMUIR + 1u;
        expect_rejected(source, "film_or_print_tag");
    }

} // namespace
