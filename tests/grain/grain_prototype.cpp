#include "film_oracle.h"
#include "film_reference.h"
#include "prototype_benchmark.h"

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
                            errors.measure(output[at], initial[at] + expected.delta, expected.roundingBound, column, row, channel);
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
            const DeviceCarrier carrier(16);
            auto frame = FilmFrame{kUhdPitch, 4096, 0, 2, 2, 4, 0, true};
            require(apply_film_reference(resident, frame, carrier.planes(), nullptr) == cudaErrorInvalidValue,
                    "out-of-range ROI was admitted");
            frame.originX = 0;
            frame.pitch = 0.5F;
            require(apply_film_reference(resident, frame, carrier.planes(), nullptr) == cudaErrorInvalidValue,
                    "unsupported pitch was admitted");
            frame.pitch = kUhdPitch;
            require(apply_film_reference({}, frame, carrier.planes(), nullptr) == cudaErrorInvalidValue,
                    "enabled frame accepted missing bank");
        }

        struct Case {
            std::string_view id;
            void (*run)();
        };

        constexpr std::array kHostCases{Case{"known-rectangles-and-constant", known_rectangles}};
        constexpr std::array kGpuCases{
            Case{"rectangles-wrap-and-seams", rectangles_and_seams}, Case{"dense-minimum-footprint", dense_footprints}, Case{"endpoints-seed-roi-alpha-padding", repeat_padded_roi}, Case{"zero-noise-identity", zero_noise}, Case{"disabled-no-bank", disabled}, Case{"unsupported-frame", unsupported_frame}};

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

        std::vector<ReferenceProbe> read_probes(const std::filesystem::path& path) {
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
                            std::isfinite(p.expected) && (p.footprint == 1 || p.footprint == kUhdPitch || p.footprint == 32),
                        "pinned reference probe outside the supported domain");
                probes.push_back(p);
            }
            require(input.eof() && !probes.empty(), "missing pinned reference probes");
            return probes;
        }

        void compare_capture(const std::filesystem::path& directory) {
            const auto probes = read_probes(directory / "reference.tsv");
            require(probes.size() == 3360, "incomplete G1 pinned probe capture");
            for (const float footprint : {1.F, kUhdPitch, 32.F}) {
                const auto bank = read_film_bank(directory, footprint);
                DeviceFilmBank device;
                check_cuda(device.upload(bank));
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
                    const double output = carrier.read().at(static_cast<std::size_t>(probe.record));
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

        void capture_response(std::string_view resources) {
            constexpr std::string_view key = "kodak_portra_400";
            constexpr auto release = [](FjFilmProfile* value) {
                static_cast<void>(fj_test_film_profile_release(value, nullptr));
            };
            FjFilmProfile* profile = nullptr;
            const auto status = fj_test_film_profile_acquire({resources.data(), resources.size()},
                                                             {key.data(), key.size()},
                                                             &profile,
                                                             nullptr);
            const std::unique_ptr<FjFilmProfile, decltype(release)> owner(profile, release);
            require(status.category == FJ_STATUS_SUCCESS && owner != nullptr, "cannot acquire Portra 400 response");
            FjFilmFixtureView view{};
            require(fj_test_film_profile_view(owner.get(), &view, nullptr).category == FJ_STATUS_SUCCESS,
                    "cannot read Portra 400 response");
            const auto count = view.log_exposure.count;
            require(count == 256 && view.density_curves_cmy.count == count * kRecords,
                    "incomplete Portra 400 response");
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
            if (mode == "reference" && argc == 3) {
                open_cuda();
                compare_capture(argv[2]);
                return;
            }
            if (mode == "response" && argc == 3) {
                capture_response(argv[2]);
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
