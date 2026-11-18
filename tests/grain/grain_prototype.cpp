#include "film_oracle.h"
#include "film_reference.h"
#include "prototype_benchmark.h"
#include "Cuda/JuicerCudaFilmGrain.h"

#include "../ffi/case_results.h"
#include "../ffi/juicer_test_api.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace GrainPrototype {
    namespace {
        constexpr std::string_view kCaseCaller = "film-reference";
        constexpr float kUhdPitch = 9.375F;
        using Clock = std::chrono::steady_clock;

        void require(bool condition, std::string_view message) {
            if (!condition)
                throw std::runtime_error(std::string(message));
        }

        void check_cuda(cudaError_t status) {
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
        }

        void open_cuda() {
            check_cuda(cudaSetDevice(0));
            check_cuda(cudaFree(nullptr));
        }

        // Only the foreign-context GPU witness needs these versioned driver entry points.
        template <typename Function>
        Function context_entry(const char* name, unsigned int version) {
            void* address = nullptr;
            cudaDriverEntryPointQueryResult query{};
            check_cuda(cudaGetDriverEntryPointByVersion(name, &address, version, cudaEnableDefault, &query));
            require(query == cudaDriverEntryPointSuccess && address != nullptr, name);
            return reinterpret_cast<Function>(address);
        }

        enum class SyntheticBank : std::uint8_t {
            Rectangles,
            Dense,
            Constant
        };

        FilmBank make_synthetic_bank(SyntheticBank pattern, float footprint) {
            FilmBank bank{};
            bank.sums.resize(kBankEntries);
            bank.tables.footprint = footprint;
            const bool dense = pattern == SyntheticBank::Dense;
            const bool constant = pattern == SyntheticBank::Constant;
            const float scale = dense ? 1.F / 16384.F : 1.F / 64.F;
            for (int record = 0; record < kRecords; ++record) {
                auto& table = bank.tables.records[record];
                table.minimumGross = 0.125F;
                table.maximumGross = 2.125F;
                table.offset = 0.125F;
                for (int level = 0; level < kLevels; ++level) {
                    const float mean = dense ? 1.F / 1024.F : 0.25F;
                    const float step = static_cast<float>(record * kLevels + level) * mean / 128;
                    table.levels[level] = constant ? FilmLevel{1.F, 0.F, 1.F}
                                                   : FilmLevel{mean - step, dense ? 3.F : 0.625F, 0.75F};
                    if (constant)
                        continue;
                    // Analytic prefix of X(x)+Y(y). X is +1 then -1 at
                    // x=128; Y is +3 then -1 at y=64. Each sums to zero.
                    const auto offset = (static_cast<std::size_t>(record) * kLevels + static_cast<std::size_t>(level)) * kTileEntries;
                    auto* sums = bank.sums.data() + offset;
                    for (int y = 1; y <= kTileSide; ++y) {
                        const int column = 4 * std::min(y, 64) - y;
                        for (int x = 1; x <= kTileSide; ++x) {
                            const int row = 2 * std::min(x, 128) - x;
                            sums[y * kTileStride + x] = scale * static_cast<float>(row * y + column * x);
                        }
                    }
                }
            }
            return bank;
        }

        bool same_words(float a, float b) {
            return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
        }

        class DeviceCarrier {
        public:
            explicit DeviceCarrier(std::size_t planeWords) : planeWords_(planeWords) {
                check_cuda(cudaMalloc(&data_, 4 * planeWords * sizeof(float)));
            }
            DeviceCarrier(const DeviceCarrier&) = delete;
            DeviceCarrier& operator=(const DeviceCarrier&) = delete;
            ~DeviceCarrier() {
                static_cast<void>(cudaFree(data_));
            }

            FilmPlanes planes() const {
                return {data_, data_ + planeWords_, data_ + 2 * planeWords_};
            }

            void write(const std::vector<float>& host) const {
                require(host.size() == 4 * planeWords_, "unexpected carrier size");
                check_cuda(cudaMemcpy(data_, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
            }

            std::vector<float> read() const {
                std::vector<float> host(4 * planeWords_);
                check_cuda(cudaDeviceSynchronize());
                check_cuda(cudaMemcpy(host.data(), data_, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
                return host;
            }

        private:
            float* data_ = nullptr;
            std::size_t planeWords_;
        };

        // Input and reference are the before/after carriers for the same frame.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        void compare_variants(const DeviceFilmBank& resident, const FilmBank& bank, FilmFrame frame, const DeviceCarrier& carrier, const std::vector<float>& input, const std::vector<float>& reference) {
            const auto& tables = bank.tables;
            for (const auto variant : {FilmVariant::Scalar, FilmVariant::Reuse, FilmVariant::Paired})
                for (const auto block : {FilmBlock::Block32x4, FilmBlock::Block16x8, FilmBlock::Block32x8}) {
                    carrier.write(input);
                    check_cuda(apply_film_variant(resident, tables, frame, carrier.planes(), nullptr, variant, block));
                    const auto actual = carrier.read();
                    require(std::equal(actual.begin(), actual.end(), reference.begin(), same_words), "Film variant differs from reference");
                }
            DeviceFilmQuads quads;
            if (frame.enabled)
                check_cuda(quads.upload(pack_film_quads(bank)));
            for (const auto block : {FilmBlock::Block32x4, FilmBlock::Block16x8, FilmBlock::Block32x8}) {
                carrier.write(input);
                check_cuda(apply_film_quads(quads, tables, frame, carrier.planes(), nullptr, block));
                const auto actual = carrier.read();
                require(std::equal(actual.begin(), actual.end(), reference.begin(), same_words), "quad Film variant differs from reference");
            }
            require(apply_film_variant(resident, tables, frame, carrier.planes(), nullptr, FilmVariant::Quad, FilmBlock::Block32x8) == cudaErrorInvalidValue, "canonical owner accepted quad layout");
        }

        std::vector<float> render_production_quad(const DeviceFilmQuads& quads, const FilmTables& sourceTables, const FilmFrame& reference, const DeviceCarrier& carrier) {
            JuicerCuda::FilmGrain::FilmTables tables{};
            tables.footprint = sourceTables.footprint;
            for (int record = 0; record < kRecords; ++record) {
                const auto& source = sourceTables.records[record];
                auto& target = tables.records[record];
                target.minimumGross = source.minimumGross;
                target.maximumGross = source.maximumGross;
                target.offset = source.offset;
                for (int level = 0; level < kLevels; ++level) {
                    target.levels[level] = {source.levels[level].meanLight, source.levels[level].meanDensity, source.levels[level].correlation};
                }
            }
            JuicerCuda::FilmGrain::FilmFrame frame{};
            frame.pitch = reference.pitch;
            frame.originX = reference.originX;
            frame.originY = reference.originY;
            frame.width = reference.width;
            frame.height = reference.height;
            frame.stride = reference.stride;
            frame.seed = reference.seed;
            frame.nextSeed = reference.nextSeed;
            frame.enabled = reference.enabled;
            frame.temporalFraction = reference.temporalFraction;
            frame.chromaFraction = reference.chromaFraction;
            std::copy_n(reference.amounts, kRecords, frame.amounts);
            const auto planes = carrier.planes();
            check_cuda(JuicerCuda::FilmGrain::apply(
                reinterpret_cast<const JuicerCuda::FilmGrain::FilmQuad*>(quads.quads()), tables, frame, {planes.cyan, planes.magenta, planes.yellow}, nullptr));
            return carrier.read();
        }

        void compare_production_quad(const FilmBank& bank, float gain) {
            DeviceFilmQuads quads;
            check_cuda(quads.upload(pack_film_quads(bank)));
            FilmFrame reference{bank.tables.footprint, -17, -9, 37, 11, 43, 0x62345678U, true};
            reference.nextSeed = 0x90123456U;
            reference.chromaFraction = 0.2406350076198578F;
            std::fill_n(reference.amounts, kRecords, gain);
            const auto planeWords = static_cast<std::size_t>(reference.stride) * static_cast<std::size_t>(reference.height);
            std::vector<float> input(4 * planeWords, -17.F);
            for (int record = 0; record < kRecords; ++record) {
                for (int y = 0; y < reference.height; ++y) {
                    for (int x = 0; x < reference.width; ++x) {
                        const auto index = static_cast<std::size_t>(record) * planeWords +
                                           static_cast<std::size_t>(y * reference.stride + x);
                        input[index] = -0.25F + static_cast<float>((record * 5 + x + 3 * y) % 17) * 0.125F;
                    }
                }
            }
            DeviceCarrier carrier(planeWords);
            for (const float time : {0.F, 0.5F}) {
                reference.temporalFraction = time;
                carrier.write(input);
                check_cuda(apply_film_quads(quads, bank.tables, reference, carrier.planes(), nullptr, FilmBlock::Block32x8));
                const auto expected = carrier.read();
                carrier.write(input);
                const auto actual = render_production_quad(quads, bank.tables, reference, carrier);
                require(std::equal(actual.begin(), actual.end(), expected.begin(), same_words),
                        "production Film Quad differs from G3, including padding or alpha");
                require(!std::equal(actual.begin(), actual.end(), input.begin(), same_words),
                        "production Film Quad did not change the carrier");
            }
        }

        void production_quad_matches_g3() {
            compare_production_quad(make_synthetic_bank(SyntheticBank::Rectangles, kUhdPitch), 1.F);
            const std::filesystem::path bankDirectory = JUICER_GRAIN_TEST_BANK;
            const std::filesystem::path tableDirectory = JUICER_GRAIN_TEST_TABLES;
            compare_production_quad(read_film_bank(bankDirectory, kUhdPitch, tableDirectory), 1.F);
            compare_production_quad(read_film_bank(bankDirectory, 37.5F, tableDirectory), 16.F);
        }

        struct Errors {
            double largestError = 0;
            double largestBudget = 0;
            double largestFraction = 0;
            std::size_t samples = 0;

            void measure(double actual, double expected, double budget, int x, int y, int record) {
                const double error = std::abs(actual - expected);
                if (!(error <= budget)) {
                    std::ostringstream message;
                    message << std::setprecision(10) << "sample (" << x << ',' << y << ") record " << record
                            << " actual " << actual << " expected " << expected << " error " << error
                            << " budget " << budget << " (density)";
                    throw std::runtime_error(message.str());
                }
                largestError = std::max(largestError, error);
                largestBudget = std::max(largestBudget, budget);
                largestFraction = std::max(largestFraction, error / budget);
                ++samples;
            }

            void report(std::string_view name) const {
                std::cout << name << " samples=" << samples << " max_error_density=" << largestError
                          << " max_budget_density=" << largestBudget << " max_budget_fraction=" << largestFraction << '\n';
            }
        };

        void production_proxy_footprint() {
            const std::filesystem::path bankDirectory = JUICER_GRAIN_TEST_BANK;
            const std::filesystem::path tableDirectory = JUICER_GRAIN_TEST_TABLES;
            auto bank = read_film_bank(bankDirectory, 37.5F, tableDirectory);
            // Fix the density reference across widths to isolate the placed
            // box average. Rust tests the production width-specific tables.
            for (auto& record : bank.tables.records)
                for (auto& level : record.levels)
                    level.meanDensity = -std::log10(level.meanLight);
            DeviceFilmQuads quads;
            check_cuda(quads.upload(pack_film_quads(bank)));
            constexpr float net = 0.2F;
            FilmFrame frame{0.F, -3, -2, 9, 6, 13, 1179208781U, true};
            const auto stride = static_cast<std::size_t>(frame.stride);
            const auto planeWords = stride * static_cast<std::size_t>(frame.height);
            std::vector<float> input(4 * planeWords, -17.F);
            for (int record = 0; record < kRecords; ++record)
                for (int y = 0; y < frame.height; ++y)
                    std::fill_n(input.data() + static_cast<std::size_t>(record) * planeWords + static_cast<std::size_t>(y) * stride,
                                frame.width,
                                net);
            DeviceCarrier carrier(planeWords);
            constexpr float custom8Width = static_cast<float>(8000.0 / 4832.0 / 4.0);
            constexpr float super16Width = static_cast<float>(static_cast<double>(11.66F) * 1000.0 / 4832.0 / 4.0);
            const std::array footprints{
                0.25F,
                custom8Width,
                super16Width,
                std::nextafter(1.0F, 0.0F),
                1.0F,
                std::nextafter(1.0F, 2.0F),
                7.450331211090088F,
                std::nextafter(37.5F, 0.F),
                37.5F,
                std::nextafter(37.5F, std::numeric_limits<float>::infinity()),
                static_cast<float>(36000.0 / (1208 * 0.5)),
                static_cast<float>(36000.0 / (604 * 0.5)),
                static_cast<float>(36000.0 / (302 * 0.5)),
                JuicerCuda::FilmGrain::kSingleWrapMaxFootprint,
                std::nextafter(255.0F, 256.0F),
                256.0F,
                static_cast<float>(36'000.0 / 92.0),
                512.0F,
                static_cast<float>(240'000.0 / 92.0),
            };
            for (const float footprint : footprints) {
                bank.tables.footprint = footprint;
                frame.pitch = footprint;
                carrier.write(input);
                const auto actual = render_production_quad(quads, bank.tables, frame, carrier);
                // The ROI crosses the placement seams at film origin and the
                // placed rectangles straddle the periodic tile boundary.
                Errors errors;
                std::size_t crossings = 0;
                double minDelta = 0;
                double maxDelta = 0;
                for (int record = 0; record <= kRecords; ++record) {
                    const auto plane = static_cast<std::size_t>(record) * planeWords;
                    for (int y = 0; y < frame.height; ++y) {
                        for (int x = 0; x < frame.stride; ++x) {
                            const auto at = plane + static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);
                            if (record == kRecords || x >= frame.width) {
                                require(same_words(actual[at], input[at]), "proxy Film grain changed padding or alpha");
                                continue;
                            }
                            const int column = frame.originX + x;
                            const int row = frame.originY + y;
                            const auto expected = evaluate_sample(bank, {
                                                                            .pitch = footprint,
                                                                            .x = column,
                                                                            .y = row,
                                                                            .seed = frame.seed,
                                                                            .record = record,
                                                                            .net = net,
                                                                            .productionSubtexel = true,
                                                                        });
                            crossings += static_cast<std::size_t>(expected.crossesTexel);
                            const double delta = static_cast<double>(actual[at]) - net;
                            errors.measure(delta, expected.delta, expected.roundingBound, column, row, record);
                            minDelta = std::min(minDelta, delta);
                            maxDelta = std::max(maxDelta, delta);
                        }
                    }
                }
                if (std::fmod(footprint, 256.0F) == 0.0F) {
                    for (int record = 0; record < kRecords; ++record) {
                        const auto start = static_cast<std::size_t>(record) * planeWords;
                        for (int y = 0; y < frame.height; ++y)
                            for (int x = 0; x < frame.width; ++x)
                                require(std::abs(actual[start + static_cast<std::size_t>(y) * stride + x] - actual[start]) <=
                                            8.0F * std::numeric_limits<float>::epsilon(),
                                        "a whole-period average varied across the image");
                    }
                } else {
                    require(minDelta < 0.0 && maxDelta > 0.0, "proxy Film grain did not produce signed density variation");
                }
                if (footprint == custom8Width || footprint == super16Width)
                    require(crossings != 0, "selected sub-texel reference did not cross a texel edge");
                if (footprint < 1.0F) {
                    const auto seed = frame.seed;
                    frame.seed = seed + 1;
                    carrier.write(input);
                    require(render_production_quad(quads, bank.tables, frame, carrier) != actual,
                            "sub-texel Film seed did not change signed density on static input");
                    frame.seed = seed;
                }
                std::cout << std::setprecision(9) << "footprint_texels=" << footprint
                          << " min_delta_density=" << minDelta << " max_delta_density=" << maxDelta
                          << " subtexel_crossing_samples=" << crossings << ' ';
                errors.report("production-proxy-footprint");
            }
        }

        void known_rectangles() {
            const auto bank = make_synthetic_bank(SyntheticBank::Rectangles, 1.F);
            const auto sample = evaluate_sample(bank, {1.F, 10, 10, 0, 0, 0.F});
            // At (10,10), field 0 reads four entire one-texel rectangles
            // with light {5/16,5/16,9/32,1/4}. Its two-axis weights are
            // the products of 11/64 and 53/64. Field 1 reads 9/32.
            constexpr double near = 11.0 / 64;
            constexpr double far = 53.0 / 64;
            const auto density = [](double light) {
                return -std::log10(light) - 0.625;
            };
            const double first = (near * near * density(5.0 / 16) + near * far * density(5.0 / 16) +
                                  far * near * density(9.0 / 32) + far * far * density(1.0 / 4)) /
                                 (near * near + far * far);
            const double expected = (first + density(9.0 / 32)) / std::sqrt(2.0);
            require(std::abs(sample.delta - expected) <= 16 * std::numeric_limits<double>::epsilon(),
                    "independent rectangle integral differs from analytic light");
            const auto constant = make_synthetic_bank(SyntheticBank::Constant, 32.F);
            require(evaluate_sample(constant, {128.F, -1, 0, 0, 2, 0.625F}).delta == 0,
                    "constant-light oracle changed density");
        }

        // Both numerical agreement and bitwise storage identity are checked
        // on the actual in-place device carrier.
        std::vector<float> check_carrier(const FilmBank& bank, FilmFrame frame, bool identity, Errors& errors) {
            const auto stride = static_cast<std::size_t>(frame.stride);
            const auto planeWords = stride * static_cast<std::size_t>(frame.height);
            std::vector<float> initial(4 * planeWords, -17.F);
            constexpr std::array densities{-0.25F, -0.F, 0.125F, 0.1875F, 0.9375F, 1.9999F, 2.F, 2.5F};
            for (int record = 0; record < kRecords; ++record) {
                const auto plane = planeWords * static_cast<std::size_t>(record);
                for (int y = 0; y < frame.height; ++y) {
                    for (int x = 0; x < frame.width; ++x) {
                        const auto pixel = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);
                        initial[plane + pixel] = densities.at(static_cast<std::size_t>(x + y + record) % densities.size());
                    }
                }
            }
            DeviceFilmBank resident;
            if (frame.enabled)
                check_cuda(resident.upload(bank));
            DeviceCarrier carrier(planeWords);
            carrier.write(initial);
            check_cuda(apply_film_reference(resident, frame, carrier.planes(), nullptr));
            auto output = carrier.read();
            compare_variants(resident, bank, frame, carrier, initial, output);
            if (!frame.enabled)
                require(!resident.resident(), "disabled frame made bank resident");
            for (int channel = 0; channel <= kRecords; ++channel) {
                for (int y = 0; y < frame.height; ++y) {
                    for (int x = 0; x < frame.stride; ++x) {
                        const auto at = static_cast<std::size_t>(channel) * planeWords +
                                        static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);
                        if (identity || channel == kRecords || x >= frame.width) {
                            require(same_words(initial[at], output[at]), "identity, padding or alpha changed");
                        } else {
                            const int column = frame.originX + x;
                            const int row = frame.originY + y;
                            const auto expected = evaluate_sample(bank, {frame.pitch, column, row, frame.seed, channel, initial[at]});
                            double delta = expected.delta;
                            double bound = expected.roundingBound;
                            if (frame.temporalFraction != 0.F) {
                                const auto next = evaluate_sample(bank, {frame.pitch, column, row, frame.nextSeed, channel, initial[at]});
                                const double t = frame.temporalFraction;
                                const double norm = std::sqrt((1 - t) * (1 - t) + t * t);
                                delta = ((1 - t) * delta + t * next.delta) / norm;
                                bound = ((1 - t) * bound + t * next.roundingBound) / norm;
                            }
                            delta *= frame.amounts[channel];
                            bound = bound * frame.amounts[channel] + 8 * std::numeric_limits<float>::epsilon() * (1 + std::abs(delta) + std::abs(initial[at]));
                            errors.measure(output[at], initial[at] + delta, bound, column, row, channel);
                        }
                    }
                }
            }
            return output;
        }

        void rectangles_and_seams() {
            Errors errors;
            const auto rectangles = make_synthetic_bank(SyntheticBank::Rectangles, 32.F);
            // Pitch 128 places the second field exactly on block crossings.
            static_cast<void>(check_carrier(rectangles, {128, -3, -3, 7, 7, 11, 0, true}, false, errors));
            // Pitch 32 places both fields at exact seam-band ends.
            static_cast<void>(check_carrier(rectangles, {32, 30, 31, 9, 6, 9, 91, true}, false, errors));
            errors.report("rectangles-and-seams");
        }

        void dense_footprints() {
            Errors errors;
            const auto dense = make_synthetic_bank(SyntheticBank::Dense, 1.F);
            static_cast<void>(check_carrier(dense, {1, 3830, 2143, 11, 10, 13, 0x07e114f0U, true}, false, errors));
            errors.report("dense-footprints");
        }

        void g2_geometry_and_temporal() {
            Errors errors;
            for (const float footprint : {0.25F, 0.78125F, 3.17F, 37.5F}) {
                const auto bank = make_synthetic_bank(SyntheticBank::Rectangles, footprint);
                FilmFrame frame{footprint, -17, 197, 9, 7, 12, 0x62345678U, true};
                frame.nextSeed = 0x90123456U;
                frame.temporalFraction = 0.5F;
                frame.amounts[0] = 0.125F;
                frame.amounts[1] = 2.F;
                frame.amounts[2] = 8.F;
                static_cast<void>(check_carrier(bank, frame, false, errors));
            }
            errors.report("g2-geometry-and-temporal");
        }

        void chroma_projection() {
            const auto bank = make_synthetic_bank(SyntheticBank::Rectangles, kUhdPitch);
            DeviceFilmBank resident;
            check_cuda(resident.upload(bank));
            FilmFrame frame{kUhdPitch, -17, 197, 9, 7, 12, 0x62345678U, true};
            frame.nextSeed = 0x90123456U;
            const auto stride = static_cast<std::size_t>(frame.stride);
            const auto planeWords = stride * static_cast<std::size_t>(frame.height);
            DeviceCarrier carrier(planeWords);
            std::vector<float> input(4 * planeWords, -17.F);
            for (std::size_t record = 0; record < kRecords; ++record)
                for (int row = 0; row < frame.height; ++row)
                    for (int column = 0; column < frame.width; ++column)
                        input[record * planeWords + static_cast<std::size_t>(row) * stride + static_cast<std::size_t>(column)] =
                            -.25F + static_cast<float>(record * 3 + static_cast<std::size_t>(column % 3)) * .375F;
            const auto render = [&](float chroma) {
                auto request = frame;
                request.chromaFraction = chroma;
                carrier.write(input);
                check_cuda(apply_film_reference(resident, request, carrier.planes(), nullptr));
                auto output = carrier.read();
                compare_variants(resident, bank, request, carrier, input, output);
                return output;
            };
            Errors errors;
            for (float time : {0.F, .5F}) {
                frame.temporalFraction = time;
                const auto reference = render(1.F);
                for (float fraction : {0.F, .240635F}) {
                    const auto revised = render(fraction);
                    require(!std::equal(revised.begin(), revised.end(), reference.begin(), same_words), "reduced chroma had no effect");
                    constexpr std::array weights{.2126, .7152, .0722};
                    for (int row = 0; row < frame.height; ++row) {
                        for (int column = 0; column < frame.stride; ++column) {
                            const auto pixel = static_cast<std::size_t>(row) * stride + static_cast<std::size_t>(column);
                            require(same_words(input[3 * planeWords + pixel], revised[3 * planeWords + pixel]), "chroma changed alpha");
                            if (column >= frame.width) {
                                for (std::size_t record = 0; record < kRecords; ++record)
                                    require(same_words(input[record * planeWords + pixel], revised[record * planeWords + pixel]), "chroma changed padding");
                                continue;
                            }
                            std::array<double, kRecords> before{}, after{};
                            double beforeY = 0, afterY = 0;
                            for (std::size_t record = 0; record < kRecords; ++record) {
                                before[record] = std::pow(10.0, -static_cast<double>(input[record * planeWords + pixel]));
                                after[record] = std::pow(10.0, -static_cast<double>(reference[record * planeWords + pixel]));
                                beforeY += weights[record] * before[record];
                                afterY += weights[record] * after[record];
                            }
                            for (std::size_t record = 0; record < kRecords; ++record) {
                                const double gray = before[record] * (afterY / beforeY);
                                const double expected = -std::log10(std::lerp(gray, after[record], static_cast<double>(fraction)));
                                // For these finite densities, the pivot differences and
                                // positive weighted sums admit a 32-u density bound.
                                const double budget = 16 * std::numeric_limits<float>::epsilon() * (1 + std::abs(expected));
                                errors.measure(revised[record * planeWords + pixel], expected, budget, column, row, static_cast<int>(record));
                            }
                        }
                    }
                }
            }
            errors.report("chroma-projection");
        }

        void repeat_padded_roi() {
            Errors errors;
            const auto bank = make_synthetic_bank(SyntheticBank::Rectangles, kUhdPitch);
            const FilmFrame roi{kUhdPitch, -14, 219, 15, 11, 20, 0x6c801534U, true};
            const auto first = check_carrier(bank, roi, false, errors);
            const auto second = check_carrier(bank, roi, false, errors);
            require(std::equal(first.begin(), first.end(), second.begin(), same_words), "same-seed ROI words differ");
            FilmFrame shiftedSeed = roi;
            ++shiftedSeed.seed;
            const auto other = check_carrier(bank, shiftedSeed, false, errors);
            require(!std::equal(first.begin(), first.end(), other.begin(), same_words), "changed seed produced identical ROI");
            errors.report("repeat-padded-roi");
        }

        void zero_noise() {
            Errors errors;
            const auto bank = make_synthetic_bank(SyntheticBank::Constant, 32.F);
            static_cast<void>(check_carrier(bank, {128, -1, 0, 8, 7, 13, 0, true}, true, errors));
            FilmFrame reduced{128, -1, 0, 8, 7, 13, 0, true};
            reduced.chromaFraction = .240635F;
            static_cast<void>(check_carrier(bank, reduced, true, errors));
        }

        void disabled() {
            Errors errors;
            const DeviceFilmBank neverUploaded;
            check_cuda(apply_film_reference(neverUploaded, {}, {}, nullptr));
            static_cast<void>(check_carrier({}, {kUhdPitch, 0, 0, 8, 7, 13, 0, false}, true, errors));
        }

        void unsupported_frame() {
            const auto bank = make_synthetic_bank(SyntheticBank::Constant, 1.F);
            DeviceFilmBank resident;
            check_cuda(resident.upload(bank));
            DeviceFilmQuads quads;
            check_cuda(quads.upload(pack_film_quads(bank)));
            require(quads.resident() && quads.context() != nullptr, "quad context was not retained");
            require(quads.upload(pack_film_quads(bank)) == cudaErrorInvalidValue, "quad owner accepted second upload");
            const DeviceCarrier carrier(16);
            auto frame = FilmFrame{kUhdPitch, 4096, 0, 2, 2, 4, 0, true};
            require(apply_film_reference(resident, frame, carrier.planes(), nullptr) == cudaErrorInvalidValue,
                    "out-of-range ROI was admitted");
            require(apply_film_quads(quads, bank.tables, frame, carrier.planes(), nullptr, FilmBlock::Block32x8) == cudaErrorInvalidValue, "quad admitted out-of-range ROI");
            frame.originX = 0;
            frame.pitch = 0.125F;
            require(apply_film_reference(resident, frame, carrier.planes(), nullptr) == cudaErrorInvalidValue,
                    "unsupported pitch was admitted");
            require(apply_film_quads(quads, bank.tables, frame, carrier.planes(), nullptr, FilmBlock::Block32x8) == cudaErrorInvalidValue, "quad admitted unsupported pitch");
            frame.pitch = kUhdPitch;
            for (float chroma : {-0.1F, 1.1F, std::numeric_limits<float>::quiet_NaN()}) {
                frame.chromaFraction = chroma;
                require(apply_film_reference(resident, frame, carrier.planes(), nullptr) == cudaErrorInvalidValue,
                        "invalid chroma fraction was admitted");
            }
            frame.chromaFraction = 1.F;
            const auto getCurrent = context_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
            const auto create = context_entry<PFN_cuCtxCreate_v12050>("cuCtxCreate", 12050);
            const auto setCurrent = context_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
            const auto destroy = context_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
            CUcontext original = nullptr;
            CUcontext foreign = nullptr;
            require(getCurrent(&original) == CUDA_SUCCESS && original != nullptr, "missing original CUDA context");
            require(create(&foreign, nullptr, 0, 0) == CUDA_SUCCESS, "cannot create foreign CUDA context");
            const auto foreignStatus = apply_film_quads(quads, bank.tables, frame, carrier.planes(), nullptr, FilmBlock::Block32x8);
            const auto restoreStatus = setCurrent(original);
            const auto destroyStatus = destroy(foreign);
            require(restoreStatus == CUDA_SUCCESS && destroyStatus == CUDA_SUCCESS, "cannot restore CUDA context");
            require(foreignStatus == cudaErrorInvalidResourceHandle, "quad accepted foreign CUDA context");
            require(apply_film_quads({}, bank.tables, frame, carrier.planes(), nullptr, FilmBlock::Block32x8) == cudaErrorInvalidValue, "quad admitted missing bank");
            require(apply_film_reference({}, frame, carrier.planes(), nullptr) == cudaErrorInvalidValue,
                    "enabled frame accepted missing bank");
        }

        struct Case {
            std::string_view id;
            void (*run)();
        };

        void signed_film_time() {
            const auto negative = resolve_film_time(-1.0, 17, 23);
            const auto half = resolve_film_time(-0.5, 17, 23);
            const auto zero = resolve_film_time(0.0, 17, 23);
            const auto positiveHalf = resolve_film_time(0.5, 17, 23);
            const auto one = resolve_film_time(1.0, 17, 23);
            require(zero.seed == 3099808472U && one.seed == 1854463680U && negative.seed == 1701653862U, "Film seed mapping changed");
            require(negative.frame == -1 && negative.fraction == 0.F && negative.nextSeed == 0, "negative integer frame");
            require(half.frame == -1 && half.fraction == 0.5F && half.seed == negative.seed && half.nextSeed == zero.seed, "negative half frame transition");
            require(positiveHalf.frame == 0 && positiveHalf.seed == zero.seed && positiveHalf.nextSeed == one.seed, "positive half frame transition");
            const auto replay = resolve_film_time(-0.5, 17, 23);
            require(replay.seed == half.seed && replay.nextSeed == half.nextSeed, "frame request order changed seeds");
        }

        constexpr std::array kHostCases{Case{"known-rectangles-and-constant", known_rectangles}, Case{"signed-film-time", signed_film_time}};
        constexpr std::array kGpuCases{
            Case{"production-quad-g3-parity", production_quad_matches_g3},
            Case{"production-proxy-footprint", production_proxy_footprint},
            Case{"g2-geometry-and-temporal", g2_geometry_and_temporal},
            Case{"chroma-projection", chroma_projection},
            Case{"rectangles-wrap-and-seams", rectangles_and_seams},
            Case{"dense-minimum-footprint", dense_footprints},
            Case{"endpoints-seed-roi-alpha-padding", repeat_padded_roi},
            Case{"zero-noise-identity", zero_noise},
            Case{"disabled-no-bank", disabled},
            Case{"unsupported-frame", unsupported_frame}};

        void run_cases(const auto& cases, bool list) {
            if (list) {
                for (const auto& test : cases)
                    JuicerTests::list_case(kCaseCaller, test.id);
                return;
            }
            JuicerTests::CaseLog log;
            for (const auto& test : cases) {
                JuicerTests::CaseEvent eventStart(kCaseCaller, test.id);
                // Preserve case identity in the console before any fatal CUDA failure.
                std::cout << "Film reference " << test.id << '\n'
                          << std::flush;
                test.run();
                eventStart.passed();
            }
            log.finish();
        }

        struct ReferenceProbe {
            float pitch, footprint;
            int x, y;
            std::uint32_t seed;
            int record;
            float net;
            double expected;
        };

        std::vector<ReferenceProbe> read_probes(const std::filesystem::path& path, bool g2 = false) {
            std::ifstream input(path);
            require(static_cast<bool>(input), "cannot open pinned reference probes");
            std::vector<ReferenceProbe> probes;
            for (std::string line; std::getline(input, line);) {
                std::istringstream row(line);
                ReferenceProbe p{};
                require(static_cast<bool>(row >> p.pitch >> p.footprint >> p.x >> p.y >> p.seed >> p.record >> p.net >> p.expected),
                        "invalid pinned reference probe");
                row >> std::ws;
                require(row.eof() && p.record >= 0 && p.record < kRecords && std::isfinite(p.net) &&
                            std::isfinite(p.expected) && std::isfinite(p.pitch) &&
                            (g2 ? (p.footprint == 0.25F || p.footprint == 0.78125F || p.footprint == 3.17F || p.footprint == 37.5F)
                                : (p.footprint == 1 || p.footprint == kUhdPitch || p.footprint == 32)),
                        "pinned reference probe outside the supported domain");
                probes.push_back(p);
            }
            require(input.eof() && !probes.empty(), "missing pinned reference probes");
            return probes;
        }

        void compare_capture(const std::filesystem::path& directory, bool compareVariant, FilmVariant variant) {
            const auto probes = read_probes(directory / "reference.tsv");
            require(probes.size() == 3360, "incomplete G1 pinned probe capture");
            for (const float footprint : {1.F, kUhdPitch, 32.F}) {
                const auto bank = read_film_bank(directory, footprint);
                DeviceFilmBank device;
                check_cuda(device.upload(bank));
                DeviceFilmQuads quads;
                if (compareVariant && variant == FilmVariant::Quad)
                    check_cuda(quads.upload(pack_film_quads(bank)));
                DeviceCarrier carrier(1);
                Errors oracleErrors;
                Errors upstreamErrors;
                Errors oracleUpstreamErrors;
                for (const auto& probe : probes) {
                    if (probe.footprint != footprint)
                        continue;
                    carrier.write(std::vector<float>(4, probe.net));
                    const FilmFrame frame{probe.pitch, probe.x, probe.y, 1, 1, 1, probe.seed, true};
                    check_cuda(apply_film_reference(device, frame, carrier.planes(), nullptr));
                    const auto reference = carrier.read();
                    if (compareVariant) {
                        carrier.write(std::vector<float>(4, probe.net));
                        check_cuda(variant == FilmVariant::Quad ? apply_film_quads(quads, bank.tables, frame, carrier.planes(), nullptr, FilmBlock::Block32x8)
                                                                : apply_film_variant(device, bank.tables, frame, carrier.planes(), nullptr, variant, FilmBlock::Block32x4));
                        const auto candidate = carrier.read();
                        require(std::equal(candidate.begin(), candidate.end(), reference.begin(), same_words), "Film variant pinned probe differs");
                    }
                    const double output = reference.at(static_cast<std::size_t>(probe.record));
                    const auto oracle = evaluate_sample(bank, {probe.pitch, probe.x, probe.y, probe.seed, probe.record, probe.net});
                    const double delta = output - probe.net;
                    oracleErrors.measure(delta, oracle.delta, oracle.roundingBound, probe.x, probe.y, probe.record);
                    oracleUpstreamErrors.measure(probe.expected, oracle.delta, oracle.roundingBound, probe.x, probe.y, probe.record);
                    upstreamErrors.measure(delta, probe.expected, 2 * oracle.roundingBound, probe.x, probe.y, probe.record);
                    std::cout << "probe," << probe.pitch << ',' << footprint << ',' << probe.x << ',' << probe.y << ','
                              << probe.seed << ',' << probe.record << ',' << probe.net << ',' << probe.expected << ','
                              << oracle.delta << ',' << delta << ',' << oracle.roundingBound << '\n';
                }
                require(oracleErrors.samples != 0, "missing probes for supported footprint");
                std::cout << "footprint_um=" << footprint << '\n';
                oracleErrors.report("CUDA/oracle");
                oracleUpstreamErrors.report("Swift/oracle");
                upstreamErrors.report("CUDA/Swift");
            }
        }

        // The immutable SAT and the independently captured footprint tables may
        // reside in separate directories. Each probe is compared three ways.
        // The arguments follow the explicit bank, tables, probes command order.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        void compare_g2_capture(const std::filesystem::path& bankDirectory,
                                const std::filesystem::path& tableDirectory,
                                const std::filesystem::path& probeFile,
                                bool compareVariant,
                                FilmVariant variant) {
            const auto probes = read_probes(probeFile, true);
            require(probes.size() == 3360, "incomplete G2 pinned probe capture");
            auto bank = read_film_bank(bankDirectory, 1.F);
            DeviceFilmBank device;
            check_cuda(device.upload(bank));
            DeviceFilmQuads quads;
            if (compareVariant && variant == FilmVariant::Quad)
                check_cuda(quads.upload(pack_film_quads(bank)));
            DeviceCarrier carrier(1);
            for (const float footprint : {0.25F, 0.78125F, 3.17F, 37.5F}) {
                bank.tables = read_film_tables(tableDirectory, footprint);
                Errors oracleErrors;
                Errors upstreamErrors;
                Errors oracleUpstreamErrors;
                for (const auto& probe : probes) {
                    if (probe.footprint != footprint)
                        continue;
                    carrier.write(std::vector<float>(4, probe.net));
                    const FilmFrame frame{probe.pitch, probe.x, probe.y, 1, 1, 1, probe.seed, true};
                    check_cuda(apply_film_reference(device, bank.tables, frame, carrier.planes(), nullptr));
                    const auto reference = carrier.read();
                    if (compareVariant) {
                        carrier.write(std::vector<float>(4, probe.net));
                        check_cuda(variant == FilmVariant::Quad ? apply_film_quads(quads, bank.tables, frame, carrier.planes(), nullptr, FilmBlock::Block32x8)
                                                                : apply_film_variant(device, bank.tables, frame, carrier.planes(), nullptr, variant, FilmBlock::Block32x4));
                        const auto candidate = carrier.read();
                        require(std::equal(candidate.begin(), candidate.end(), reference.begin(), same_words), "Film variant pinned probe differs");
                    }
                    const double output = reference.at(static_cast<std::size_t>(probe.record));
                    const auto oracle = evaluate_sample(bank, {probe.pitch, probe.x, probe.y, probe.seed, probe.record, probe.net});
                    const double delta = output - probe.net;
                    oracleErrors.measure(delta, oracle.delta, oracle.roundingBound, probe.x, probe.y, probe.record);
                    oracleUpstreamErrors.measure(probe.expected, oracle.delta, oracle.roundingBound, probe.x, probe.y, probe.record);
                    upstreamErrors.measure(delta, probe.expected, 2 * oracle.roundingBound, probe.x, probe.y, probe.record);
                    std::cout << "probe," << probe.pitch << ',' << footprint << ',' << probe.x << ',' << probe.y << ','
                              << probe.seed << ',' << probe.record << ',' << probe.net << ',' << probe.expected << ','
                              << oracle.delta << ',' << delta << ',' << oracle.roundingBound << '\n';
                }
                require(oracleErrors.samples == 840, "incomplete G2 footprint probes");
                std::cout << "footprint_um=" << footprint << '\n';
                oracleErrors.report("CUDA/oracle");
                oracleUpstreamErrors.report("Swift/oracle");
                upstreamErrors.report("CUDA/Swift");
            }
        }

        // Positional resource root and stock ID in the response command.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        void capture_response(std::string_view resources, std::string_view key) {
            constexpr auto release = [](FjFilmProfile* value) {
                static_cast<void>(fj_test_film_profile_release(value, nullptr));
            };
            FjFilmProfile* profile = nullptr;
            const auto status = fj_test_film_profile_acquire({resources.data(), resources.size()},
                                                             {key.data(), key.size()},
                                                             &profile,
                                                             nullptr);
            const std::unique_ptr<FjFilmProfile, decltype(release)> owner(profile, release);
            require(status.category == FJ_STATUS_SUCCESS && owner != nullptr, "cannot acquire stock response");
            FjFilmFixtureView view{};
            require(fj_test_film_profile_view(owner.get(), &view, nullptr).category == FJ_STATUS_SUCCESS,
                    "cannot read stock response");
            const auto count = view.log_exposure.count;
            require(count == 256 && view.density_curves_cmy.count == count * kRecords,
                    "incomplete stock response");
            auto axis = nlohmann::json::array();
            auto density = nlohmann::json::array();
            for (std::size_t i = 0; i < count; ++i) {
                axis.push_back(view.log_exposure.data[i]);
                const float* cmy = view.density_curves_cmy.data + i * kRecords;
                density.push_back({cmy[0], cmy[1], cmy[2]});
            }
            std::cout << nlohmann::json{{"data", {{"log_exposure", axis}, {"density_curves", density}}}}.dump() << '\n';
        }

        cudaError_t launch_uhd(void* owner, float* c, float* m, float* y, cudaStream_t stream) {
            constexpr FilmFrame frame{kUhdPitch, 0, 0, 3840, 2160, 3840, 0, true};
            return apply_film_reference(*static_cast<const DeviceFilmBank*>(owner), frame, {c, m, y}, stream);
        }

        double elapsed_ms(Clock::time_point start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }

        // Path order follows the benchmark command's bank, resources, carrier operands.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        void time_uhd(const std::filesystem::path& directory, const std::filesystem::path& resources, const std::filesystem::path& input) {
            const auto read = Clock::now();
            const auto bank = read_film_bank(directory, kUhdPitch);
            std::cout << "film_read_validate_ms," << elapsed_ms(read) << '\n';
            const auto setup = Clock::now();
            DeviceFilmBank device;
            check_cuda(device.upload(bank));
            check_cuda(cudaDeviceSynchronize());
            std::cout << "film_allocate_upload_ms," << elapsed_ms(setup) << "\nfilm_bank_host_bytes,"
                      << bank.sums.capacity() * sizeof(float) + sizeof(bank) << "\nfilm_bank_device_bytes,"
                      << bank.sums.size() * sizeof(float) << "\nfilm_kernel_table_argument_bytes," << sizeof(FilmTables)
                      << "\nfilm_seed,0\nfilm_workspace_plane_bytes,0\n";
            benchmark(resources, input, launch_uhd, &device, std::cout);
        }

        // Command operands are the bank directory followed by the output report.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        void verify_quads(const std::filesystem::path& directory, const std::filesystem::path& reportPath) {
            const auto bank = read_film_bank(directory, 1.F);
            const auto packed = pack_film_quads(bank);
            DeviceFilmQuads device;
            check_cuda(device.upload(packed));
            std::vector<FilmQuad> copied(kQuadEntries);
            check_cuda(cudaMemcpy(copied.data(), device.quads(), copied.size() * sizeof(FilmQuad), cudaMemcpyDeviceToHost));
            std::size_t mismatches = 0;
            nlohmann::json first = nullptr;
            for (int record = 0; record < kRecords; ++record)
                for (int level = 0; level < kLevels; ++level)
                    for (int y = 0; y < kTileSide; ++y)
                        for (int x = 0; x < kTileSide; ++x) {
                            const auto levelIndex = static_cast<std::size_t>(record) * kLevels + static_cast<std::size_t>(level);
                            const auto source = levelIndex * kTileEntries + static_cast<std::size_t>(y * kTileStride + x);
                            const auto& q = copied[levelIndex * kTileSide * kTileSide + static_cast<std::size_t>(y * kTileSide + x)];
                            const std::array actual{q.topLeft, q.topRight, q.bottomLeft, q.bottomRight};
                            const std::array expected{bank.sums[source], bank.sums[source + 1], bank.sums[source + kTileStride], bank.sums[source + kTileStride + 1]};
                            for (std::size_t word = 0; word < actual.size(); ++word)
                                if (!same_words(actual[word], expected[word])) {
                                    ++mismatches;
                                    if (first.is_null())
                                        first = {{"record", record}, {"level", level}, {"x", x}, {"y", y}, {"word", word}, {"expected", std::bit_cast<std::uint32_t>(expected[word])}, {"actual", std::bit_cast<std::uint32_t>(actual[word])}};
                                }
                        }
            require(!std::filesystem::exists(reportPath), "quad verification report exists");
            std::ofstream output(reportPath);
            output << nlohmann::json{{"bank", directory.string()}, {"quad_count", kQuadEntries}, {"checked_words", kQuadEntries * 4}, {"bit_mismatches", mismatches}, {"device_quad_bytes", kQuadEntries * sizeof(FilmQuad)}, {"first_mismatch", first}}.dump(2) << '\n';
            output.close();
            require(output.good(), "cannot write quad verification report");
            require(mismatches == 0, "quad packing differs from bank");
        }

        void run_main(int argc, char** argv) {
            require(argc >= 2, "expected host, gpu, response, reference, capture or benchmark mode");
            const std::string_view mode(argv[1]);
            const bool list = JuicerTests::list_requested(argc, argv);
            if (mode == "host" && (argc == 2 || (argc == 3 && list))) {
                run_cases(kHostCases, list);
                return;
            }
            if (mode == "gpu" && (argc == 2 || (argc == 3 && list))) {
                if (!list)
                    open_cuda();
                run_cases(kGpuCases, list);
                return;
            }
            if (mode == "develop-g3" && argc == 4) {
                develop_g3(argv[2], argv[3]);
                return;
            }
            if ((mode == "stage" || mode == "stage-check" || mode == "stage-verify") && argc == 4) {
                stage(argv[2], argv[3], mode == "stage-check", mode == "stage-verify");
                return;
            }
            if (mode == "verify-quads" && argc == 4) {
                open_cuda();
                verify_quads(argv[2], argv[3]);
                return;
            }
            if (mode == "research" && argc == 4) {
                capture_research(argv[2], argv[3], std::cout);
                return;
            }
            if ((mode == "reference-g2" || mode == "reference-g2-scalar" || mode == "reference-g2-reuse" || mode == "reference-g2-paired" || mode == "reference-g2-quad") && argc == 5) {
                open_cuda();
                compare_g2_capture(argv[2], argv[3], argv[4], mode != "reference-g2", mode == "reference-g2-reuse" ? FilmVariant::Reuse : (mode == "reference-g2-paired" ? FilmVariant::Paired : (mode == "reference-g2-quad" ? FilmVariant::Quad : FilmVariant::Scalar)));
                return;
            }
            if ((mode == "reference" || mode == "reference-scalar" || mode == "reference-reuse" || mode == "reference-paired" || mode == "reference-quad") && argc == 3) {
                open_cuda();
                compare_capture(argv[2], mode != "reference", mode == "reference-reuse" ? FilmVariant::Reuse : (mode == "reference-paired" ? FilmVariant::Paired : (mode == "reference-quad" ? FilmVariant::Quad : FilmVariant::Scalar)));
                return;
            }
            if (mode == "response" && (argc == 3 || argc == 4)) {
                capture_response(argv[2], argc == 4 ? argv[3] : "kodak_portra_400");
                return;
            }
            if (mode == "capture" && argc == 4) {
                capture_developed_input(argv[2], argv[3]);
                return;
            }
            if (mode == "benchmark" && argc == 5) {
                open_cuda();
                time_uhd(argv[2], argv[3], argv[4]);
                return;
            }
            throw std::invalid_argument("invalid Film reference command");
        }
    } // namespace
} // namespace GrainPrototype

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(10);
        GrainPrototype::run_main(argc, argv);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Film reference: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("Film reference: unknown execution failure\n", stderr);
        return 1;
    }
}
