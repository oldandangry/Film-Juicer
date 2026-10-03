#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "ResourceAssetLibrary.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "SpectralProcessing.h"
#include "Illuminants.h"
#include "Hash.h"
#include "Cuda/JuicerCudaResources.h"
#include "juicer_cuda_owner.h"
#include "RustAssetBridge.h"
#include "juicer_test_api.h"

extern "C" const std::size_t* fj_test_production_profile_abi_c_facts(std::size_t* count);
extern "C" const std::size_t* fj_test_production_profile_abi_facts(std::size_t* count);
#define FJ_ABI_TYPE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_ABI_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#define FJ_ABI_VALUE(tag, value) static_assert(tag == value);
#include "production_profile_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE

static_assert(std::is_same_v<decltype(&fj_legacy_film_profile_acquire), FjStatus (*)(const FjAssets*, FjStringView, FjFilmProfile**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_print_profile_acquire), FjStatus (*)(const FjAssets*, FjStringView, FjPrintProfile**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_film_profile_view), FjStatus (*)(const FjFilmProfile*, FjFilmProfileView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_print_profile_view), FjStatus (*)(const FjPrintProfile*, FjPrintProfileView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_print_profile_sample_density), FjStatus (*)(const FjPrintProfile*, double, FjPrintDensityCurves**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_print_density_view), FjStatus (*)(const FjPrintDensityCurves*, FjPrintDensityView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_film_profile_release), FjStatus (*)(FjFilmProfile*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_print_profile_release), FjStatus (*)(FjPrintProfile*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_print_density_release), FjStatus (*)(FjPrintDensityCurves*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_assets_release_cached_payloads), FjStatus (*)(const FjAssets*, FjErrorBuffer*)>);

namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    fs::path resources;
    fs::path fixtures;
    fs::path scratch;
    thread_local unsigned conversionFault = 0;
    std::barrier<>* publicationBarrier = nullptr;
    std::barrier<>* catalogPublicationBarrier = nullptr;
    std::atomic<bool> failedCandidateReady{false};
    std::atomic<bool> winnerPublished{false};
    std::atomic<std::size_t> peakProfileOwners{0};

    FjStringView text(const std::string& value) {
        return {value.empty() ? nullptr : value.data(), value.size()};
    }
    void expect_status(FjStatus status, std::uint32_t expected) {
        EXPECT_EQ(status.category, expected);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
    }
    void require_release(FjStatus status) noexcept {
        if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
            std::fprintf(stderr, "unexpected destructor release status=%u/%u/%d\n", status.category, status.api, status.native_code);
            std::abort();
        }
    }
    Json read_json(const fs::path& path) {
        std::ifstream file(path);
        return Json::parse(file);
    }
    struct Assets {
        FjAssets* owner = nullptr;
        explicit Assets(const fs::path& root) {
            const JuicerAssets::NativePathArgument path(root);
            expect_status(fj_legacy_assets_create(path.view(), &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Assets() {
            require_release(fj_legacy_assets_destroy(owner, nullptr));
        }
    };
    struct Film {
        FjFilmProfile* owner = nullptr;
        Film(const Assets& assets, const std::string& key) {
            expect_status(fj_legacy_film_profile_acquire(assets.owner, text(key), &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Film() {
            require_release(fj_legacy_film_profile_release(owner, nullptr));
        }
        FjFilmProfileView view() const {
            FjFilmProfileView out{};
            expect_status(fj_legacy_film_profile_view(owner, &out, nullptr), FJ_STATUS_SUCCESS);
            return out;
        }
    };
    struct Print {
        FjPrintProfile* owner = nullptr;
        Print(const Assets& assets, const std::string& key) {
            expect_status(fj_legacy_print_profile_acquire(assets.owner, text(key), &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Print() {
            require_release(fj_legacy_print_profile_release(owner, nullptr));
        }
        FjPrintProfileView view() const {
            FjPrintProfileView out{};
            expect_status(fj_legacy_print_profile_view(owner, &out, nullptr), FJ_STATUS_SUCCESS);
            return out;
        }
    };
    struct Density {
        FjPrintDensityCurves* owner = nullptr;
        Density(const Print& print, double gamma) {
            expect_status(fj_legacy_print_profile_sample_density(print.owner, gamma, &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Density() {
            require_release(fj_legacy_print_density_release(owner, nullptr));
        }
        FjPrintDensityView view() const {
            FjPrintDensityView out{};
            expect_status(fj_legacy_print_density_view(owner, &out, nullptr), FJ_STATUS_SUCCESS);
            return out;
        }
    };
    void bits(FjFloatSpan span, const Json& expected) {
        ASSERT_EQ(span.count, expected.size());
        for (std::size_t i = 0; i < span.count; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(span.data[i]), expected[i].get<std::uint32_t>());
        }
    }
    std::vector<unsigned char> density_payload() {
        std::ifstream file(fixtures / "density-samples.bin", std::ios::binary);
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }
    std::uint32_t word(const std::vector<unsigned char>& bytes, std::size_t offset) {
        return std::uint32_t(bytes[offset]) | std::uint32_t(bytes[offset + 1]) << 8 | std::uint32_t(bytes[offset + 2]) << 16 | std::uint32_t(bytes[offset + 3]) << 24;
    }
    void density_bits(FjFloatSpan span, const Json& capture, const std::vector<unsigned char>& bytes) {
        const auto start = capture["sample_start"].get<std::size_t>();
        for (std::size_t i = 0; i < span.count / 3; ++i) {
            ASSERT_EQ(bytes[(start + i) * 49], 1);
            for (std::size_t c = 0; c < 3; ++c) {
                EXPECT_EQ(std::bit_cast<std::uint32_t>(span.data[3 * i + c]), word(bytes, (start + i) * 49 + 1 + c * 4));
            }
        }
    }
    void tables(FjProfileTablesView view, const Json& expected) {
        bits(view.channel_density_cmy, expected["fields"]["channel_density"]);
        bits(view.base_density, expected["fields"]["base_density"]);
        bits(view.log_exposure, expected["fields"]["log_exposure"]);
#if defined(_WIN32)
        bits(view.linear_sensitivity_rgb, expected["linear_windows"]);
#else
        bits(view.linear_sensitivity_rgb, expected["linear_linux"]);
#endif
    }
    const Json& capture(const Json& density, const std::string& id) {
        for (const auto& row : density["cases"]) {
            if (row["id"] == id) {
                return row;
            }
        }
        throw std::runtime_error("missing independent capture");
    }

    TEST(ProductionProfileOwner, ActualCAndRustLayoutsMatchEveryField) {
        const std::size_t cpp[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#define FJ_ABI_VALUE(tag, value) tag,
#include "production_profile_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
        };
        std::size_t cCount = 0, rustCount = 0;
        const auto* c = fj_test_production_profile_abi_c_facts(&cCount);
        const auto* rust = fj_test_production_profile_abi_facts(&rustCount);
        ASSERT_EQ(cCount, std::size(cpp));
        ASSERT_EQ(rustCount, std::size(cpp));
        for (std::size_t i = 0; i < cCount; ++i) {
            EXPECT_EQ(c[i], cpp[i]);
            EXPECT_EQ(rust[i], cpp[i]);
        }
    }

    TEST(ProductionProfileOwner, All28TokensTablesDigestsAndGammaMatchIndependentCaptures) {
        const auto completed = read_json(fixtures / "completed.json");
        const auto density = read_json(fixtures / "density.json");
        const auto gamma = read_json(fixtures / "print-gamma.json");
        const auto payload = density_payload();
        Assets assets(resources);
        FjCatalog* catalog = nullptr;
        FjCatalogCounts counts{};
        expect_status(fj_legacy_catalog_acquire(assets.owner, &catalog, &counts, nullptr), FJ_STATUS_SUCCESS);
        ASSERT_EQ(counts.film_count + counts.print_count, 28u);
        for (std::uint32_t role : {FJ_PROFILE_ROLE_FILM, FJ_PROFILE_ROLE_PRINT}) {
            const auto count = role == FJ_PROFILE_ROLE_FILM ? counts.film_count : counts.print_count;
            for (std::size_t i = 0; i < count; ++i) {
                FjCatalogEntryView entry{};
                expect_status(fj_legacy_catalog_entry(catalog, role, i, &entry, nullptr), FJ_STATUS_SUCCESS);
                const std::string key(entry.key.data, entry.key.count);
                SCOPED_TRACE(key);
                const auto& expected = completed["profiles"][key];
                const std::string baselineId = key + "/gamma-1.0";
                const auto& baseline = capture(density, baselineId);
                if (role == FJ_PROFILE_ROLE_FILM) {
                    Film film(assets, key);
                    const auto view = film.view();
                    EXPECT_EQ(view.asset_token, expected["asset_token"].get<std::uint64_t>());
                    tables(view.tables, expected);
                    bits(view.wavelengths, expected["fields"]["wavelengths"]);
                    density_bits(view.tables.density_curves_cmy, baseline, payload);
                    for (std::size_t layer = 0; layer < 3; ++layer) {
                        for (std::size_t channel = 0; channel < 3; ++channel) {
                            const auto span = view.density_curves_layers[layer][channel];
                            ASSERT_EQ(span.count, view.tables.log_exposure.count);
                            for (std::size_t row = 0; row < span.count; ++row) {
                                EXPECT_EQ(std::bit_cast<std::uint32_t>(span.data[row]), word(payload, (baseline["sample_start"].get<std::size_t>() + row) * 49 + 1 + (3 + layer * 3 + channel) * 4));
                            }
                        }
                    }
                    bits({view.digest.gamma_samelayer_rgb, 3}, expected["digest"]["gamma_samelayer_rgb"]);
                    bits({view.digest.gamma_interlayer_r_to_gb, 2}, expected["digest"]["gamma_interlayer_r_to_gb"]);
                    bits({view.digest.gamma_interlayer_g_to_rb, 2}, expected["digest"]["gamma_interlayer_g_to_rb"]);
                    bits({view.digest.gamma_interlayer_b_to_rg, 2}, expected["digest"]["gamma_interlayer_b_to_rg"]);
                    bits({view.digest.halation_first_sigma_um, 3}, expected["digest"]["halation_first_sigma_um"]);
                    bits({view.digest.halation_primary_amount, 3}, expected["digest"]["halation_primary_amount"]);
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(view.digest.hanatos_spectral_gaussian_blur_default), expected["digest"]["hanatos_spectral_gaussian_blur_default"].get<std::uint32_t>());
                    if (!expected["fields"]["window"].is_null()) {
                        bits(view.hanatos_window, expected["fields"]["window"]);
                    } else {
                        EXPECT_EQ(view.hanatos_window.count, 0u);
                        EXPECT_EQ(view.hanatos_window.data, nullptr);
                    }
                    if (!expected["fields"]["surface"].is_null()) {
                        bits(view.hanatos_surface_rgb, expected["fields"]["surface"]);
                    }
                } else {
                    Print print(assets, key);
                    const auto view = print.view();
                    EXPECT_EQ(view.asset_token, expected["asset_token"].get<std::uint64_t>());
                    tables(view.tables, expected);
                    density_bits(view.tables.density_curves_cmy, baseline, payload);
                    for (const auto& item : gamma["captured"]) {
                        const auto id = item["id"].get<std::string>();
                        if (!id.starts_with(key + "/gamma-")) {
                            continue;
                        }
                        const auto& row = capture(density, id);
                        Density curves(print, std::bit_cast<double>(row["gamma_bits"].get<std::uint64_t>()));
                        const auto sampled = curves.view();
                        EXPECT_EQ(sampled.hash, row["identities"]["print_hash"].get<std::uint64_t>());
                        ASSERT_EQ(sampled.totals_cmy.count, view.tables.log_exposure.count * 3);
                        density_bits(sampled.totals_cmy, row, payload);
                    }
                }
            }
        }
        expect_status(fj_legacy_catalog_release(catalog, nullptr), FJ_STATUS_SUCCESS);
    }

    TEST(ProductionProfileOwner, BorrowedViewsAndOwnedGammaSurviveCacheAndAssetsDrop) {
        Assets assets(resources);
        Film film(assets, "kodak_portra_400");
        Print print(assets, "kodak_portra_endura");
        const auto fv = film.view();
        const auto pv = print.view();
        Density density(print, 1.1);
        const auto dv = density.view();
        const auto first = dv.totals_cmy.data[0];
        const auto hash = dv.hash;
        expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_assets_destroy(std::exchange(assets.owner, nullptr), nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(film.view().tables.log_exposure.data, fv.tables.log_exposure.data);
        EXPECT_EQ(print.view().tables.log_exposure.data, pv.tables.log_exposure.data);
        std::array<std::thread, 8> readers;
        std::array<bool, 8> passed{};
        for (std::size_t i = 0; i < readers.size(); ++i) {
            readers[i] = std::thread([&, i] {
                FjPrintDensityCurves* curves = nullptr;
                FjPrintDensityView view{};
                FjFilmProfileView filmView{};
                const auto a = fj_legacy_print_profile_sample_density(print.owner, 1.1, &curves, nullptr);
                const auto b = fj_legacy_print_density_view(curves, &view, nullptr);
                const auto c = fj_legacy_film_profile_view(film.owner, &filmView, nullptr);
                passed[i] = a.category == FJ_STATUS_SUCCESS && b.category == FJ_STATUS_SUCCESS && c.category == FJ_STATUS_SUCCESS && view.hash == hash && filmView.tables.log_exposure.data == fv.tables.log_exposure.data;
                passed[i] = passed[i] && fj_legacy_print_density_release(curves, nullptr).category == FJ_STATUS_SUCCESS;
            });
        }
        for (auto& reader : readers) {
            reader.join();
        }
        for (bool pass : passed) {
            EXPECT_TRUE(pass);
        }
        expect_status(fj_legacy_print_profile_release(std::exchange(print.owner, nullptr), nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(density.view().hash, hash);
        EXPECT_EQ(density.view().totals_cmy.data[0], first);
    }

    TEST(ProductionProfileOwner, ForeignFailuresClearOutputsAndConsumeMalformedReleases) {
        Assets assets(resources);
        for (auto key : {FjStringView{nullptr, 1}, FjStringView{"x", std::numeric_limits<std::size_t>::max()}, FjStringView{"\xff", 1}}) {
            FjFilmProfile* film = nullptr;
            FjPrintProfile* print = nullptr;
            expect_status(fj_legacy_film_profile_acquire(assets.owner, key, &film, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(fj_legacy_print_profile_acquire(assets.owner, key, &print, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(film, nullptr);
            EXPECT_EQ(print, nullptr);
        }
        for (const std::string& key : {std::string(), std::string("missing"), std::string("kodak_portra_400\0", 17), std::string("kodak_portra_endura")}) {
            FjFilmProfile* film = nullptr;
            expect_status(fj_legacy_film_profile_acquire(assets.owner, text(key), &film, nullptr), FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_EQ(film, nullptr);
        }
        FjFilmProfileView fv{};
        fv.asset_token = 42;
        expect_status(fj_legacy_film_profile_view(nullptr, &fv, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fv.asset_token, 0u);
        EXPECT_EQ(fv.tables.log_exposure.data, nullptr);
        Print print(assets, "kodak_portra_endura");
        for (double gamma : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            FjPrintDensityCurves* curves = nullptr;
            expect_status(fj_legacy_print_profile_sample_density(print.owner, gamma, &curves, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(curves, nullptr);
        }
        for (unsigned fault : {1u, 2u}) {
            fj_test_profile_fault(fault);
            FjFilmProfile* out = nullptr;
            expect_status(fj_legacy_film_profile_acquire(assets.owner, text("kodak_portra_400"), &out, nullptr), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE);
            EXPECT_EQ(out, nullptr);
            fj_test_profile_fault(fault);
            FjPrintDensityCurves* curves = nullptr;
            expect_status(fj_legacy_print_profile_sample_density(print.owner, 1.1, &curves, nullptr), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE);
            EXPECT_EQ(curves, nullptr);
        }
        Film film(assets, "kodak_portra_400");
        FjErrorBuffer malformedFilm{nullptr, 1, 99};
        expect_status(fj_legacy_film_profile_release(std::exchange(film.owner, nullptr), &malformedFilm), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_test_profile_live_owners(), 1u);
        for (bool gammaResult : {false, true}) {
            if (gammaResult) {
                Density released(print, 1.1);
                fj_test_profile_fault(3);
                expect_status(fj_legacy_print_density_release(std::exchange(released.owner, nullptr), nullptr), FJ_STATUS_INTERNAL_FAILURE);
                EXPECT_EQ(fj_test_print_density_live_owners(), 0u);
            } else {
                Film released(assets, "kodak_portra_400");
                fj_test_profile_fault(3);
                expect_status(fj_legacy_film_profile_release(std::exchange(released.owner, nullptr), nullptr), FJ_STATUS_INTERNAL_FAILURE);
                EXPECT_EQ(fj_test_profile_live_owners(), 1u);
            }
        }
        FjErrorBuffer malformed{nullptr, 1, 99};
        Density density(print, 1.1);
        expect_status(fj_legacy_print_density_release(std::exchange(density.owner, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_test_print_density_live_owners(), 0u);
        EXPECT_EQ(malformed.length, 0u);
        expect_status(fj_legacy_print_profile_release(std::exchange(print.owner, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_test_profile_live_owners(), 0u);
        std::array<char, 4> bytes{};
        FjErrorBuffer tiny{bytes.data(), bytes.size(), 99};
        expect_status(fj_legacy_print_density_view(nullptr, nullptr, &tiny), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(tiny.length, 3u);
        EXPECT_EQ(bytes.back(), '\0');
        expect_status(fj_legacy_print_density_release(nullptr, nullptr), FJ_STATUS_SUCCESS);
    }

    TEST(ProductionProfileOwner, TerminalHostCloseReportsFirstFailureAndConsumesEveryFamily) {
        JuicerAssets::Library library(resources, resources.string());
        {
            auto selected = library.selected_profiles_for_route({"kodak_portra_400", "kodak_portra_endura", Spektrafilm::ScanRoute::NegativePrintScan});
            ASSERT_TRUE(selected.valid);
        }
        fj_test_profile_fault(3);
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        expect_status(library.close(&error), FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_NE(std::string(message.data(), error.length).find("after consume"), std::string::npos);
        EXPECT_EQ(fj_test_profile_live_owners(), 0u);
        EXPECT_EQ(fj_test_catalog_live_owners(), 0u);
        EXPECT_EQ(fj_test_assets_live_owners(), 0u);
        expect_status(library.close(nullptr), FJ_STATUS_SUCCESS);
    }

    TEST(ProductionProfileOwner, NativeColdCandidatesPublishOneWinnerAndWarmCallsCopyNothing) {
        for (bool print : {false, true}) {
            JuicerAssets::Library library(resources, resources.string());
            (void)library.spektrafilm_profile_catalog();
            peakProfileOwners = 0;
            std::barrier barrier(8);
            publicationBarrier = &barrier;
            std::array<std::shared_ptr<const Profiles::FilmProfile>, 8> films;
            std::array<std::shared_ptr<const JuicerAssets::PrintProfileSource>, 8> prints;
            std::array<std::thread, 8> workers;
            for (std::size_t i = 0; i < workers.size(); ++i) {
                workers[i] = std::thread([&, i] {
                    const auto selected = library.selected_profiles_for_route({"kodak_portra_400", "kodak_portra_endura", print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan});
                    films[i] = selected.filmProfile;
                    prints[i] = selected.printSource;
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
            publicationBarrier = nullptr;
            for (std::size_t i = 1; i < workers.size(); ++i) {
                EXPECT_EQ(films[i], films[0]);
                EXPECT_EQ(prints[i], prints[0]);
            }
            EXPECT_EQ(fj_test_profile_live_owners(), print ? 2u : 1u);
            EXPECT_EQ(peakProfileOwners.load(), print ? 9u : 8u);
            const auto& data = films[0]->data;
            std::size_t filmCapacity = sizeof(*films[0]) + data.logExposure.capacity() * sizeof(float) + data.densityCurves.capacity() * sizeof(std::array<float, 3>);
            for (const auto& layer : data.densityCurvesLayers) {
                for (const auto& channel : layer) {
                    filmCapacity += channel.capacity() * sizeof(float);
                }
            }
            std::printf("Native film fixed+vector capacity=%zu; simultaneous cold sources=%zu; film-only cold-copy overlap=%zu; losers reclaimed to=%zu; metadata strings/control blocks/allocator excluded\n", filmCapacity, peakProfileOwners.load(), print ? filmCapacity : 8 * filmCapacity, fj_test_profile_live_owners());
            for (unsigned i = 0; i < 32; ++i) {
                const auto selected = library.selected_profiles_for_route({"kodak_portra_400", "kodak_portra_endura", print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan});
                EXPECT_EQ(selected.filmProfile, films[0]);
                EXPECT_EQ(selected.printSource, prints[0]);
            }
        }
        EXPECT_EQ(fj_test_profile_live_owners(), 0u);
    }

    TEST(ProductionProfileOwner, PartialNativeCopiesCleanUpAndRetryTheRetainedSource) {
        for (unsigned fault : {1u, 2u, 3u}) {
            JuicerAssets::Library library(resources, resources.string());
            (void)library.spektrafilm_profile_catalog();
            conversionFault = fault;
            bool failed = false;
            try {
                (void)library.selected_film_profile_for_key("kodak_portra_400");
            } catch (const std::bad_alloc&) {
                failed = fault == 3;
            } catch (const JuicerCuda::ExecutionFailure& error) {
                failed = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            EXPECT_TRUE(failed);
            EXPECT_EQ(fj_test_profile_live_owners(), 0u);
            EXPECT_NE(library.selected_film_profile_for_key("kodak_portra_400"), nullptr);
        }
    }


    TEST(ProductionProfileOwner, FailedColdAttemptReturnsConcurrentCompleteWinner) {
        JuicerAssets::Library library(resources, resources.string());
        (void)library.spektrafilm_profile_catalog();
        failedCandidateReady = false;
        winnerPublished = false;
        std::shared_ptr<const Profiles::FilmProfile> failedResult;
        std::thread failed([&] {
            conversionFault = 4;
            failedResult = library.selected_film_profile_for_key("kodak_portra_400");
        });
        while (!failedCandidateReady.load()) {
            failedCandidateReady.wait(false);
        }
        const auto winner = library.selected_film_profile_for_key("kodak_portra_400");
        winnerPublished = true;
        winnerPublished.notify_all();
        failed.join();
        EXPECT_EQ(failedResult, winner);
        EXPECT_EQ(fj_test_profile_live_owners(), 1u);
    }

    TEST(ProductionProfileOwner, CacheReleaseOverlapsFirstCatalogPublication) {
        for (unsigned attempt = 0; attempt < 8; ++attempt) {
            JuicerAssets::Library library(resources, resources.string());
            expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
            EXPECT_EQ(fj_test_catalog_live_owners(), 0u);
            std::barrier barrier(2);
            catalogPublicationBarrier = &barrier;
            JuicerAssets::SelectedProfileResult selected;
            std::exception_ptr failure;
            std::thread loader([&] {
                try {
                    selected = library.selected_profiles_for_route({"kodak_portra_400", "kodak_portra_endura", Spektrafilm::ScanRoute::NegativePrintScan});
                } catch (...) {
                    failure = std::current_exception();
                }
            });
            barrier.arrive_and_wait();
            // Release must finish while the first catalog is still unpublished.
            expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
            barrier.arrive_and_wait();
            for (unsigned release = 0; release < 32; ++release) {
                expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
            }
            loader.join();
            catalogPublicationBarrier = nullptr;
            if (failure) {
                std::rethrow_exception(failure);
            }
            ASSERT_TRUE(selected.valid);
            ASSERT_NE(selected.filmProfile, nullptr);
            ASSERT_NE(selected.printSource, nullptr);
            const auto density = selected.printSource->sample_density_curves(1.1);
            auto native = selected.printSource->profile();
            expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
            EXPECT_EQ(selected.printSource->sample_density_curves(1.1).hash, density.hash);
            selected = {};
            EXPECT_EQ(fj_test_profile_live_owners(), 0u);
            expect_status(library.close(), FJ_STATUS_SUCCESS);
            EXPECT_EQ(fj_test_catalog_live_owners(), 0u);
            EXPECT_EQ(fj_test_assets_live_owners(), 0u);
            EXPECT_FALSE(native->data.logExposure.empty());
        }
    }

    TEST(ProductionProfileOwner, CacheReleasePreservesExactPrintSourceAndIndependentNativeCopy) {
        const auto root = scratch / "same-source";
        fs::create_directories(root / "profiles");
        fs::copy_file(resources / "profiles/kodak_portra_400.json", root / "profiles/kodak_portra_400.json", fs::copy_options::overwrite_existing);
        fs::copy_file(resources / "profiles/kodak_portra_endura.json", root / "profiles/kodak_portra_endura.json", fs::copy_options::overwrite_existing);
        JuicerAssets::Library library(root, root.string());
        auto selected = library.selected_profiles_for_route({"kodak_portra_400", "kodak_portra_endura", Spektrafilm::ScanRoute::NegativePrintScan});
        ASSERT_TRUE(selected.valid);
        const auto old = selected.printSource->sample_density_curves(1.1);
        auto native = selected.printSource->profile();
        auto modified = read_json(root / "profiles/kodak_portra_endura.json");
        modified["data"]["density_curves_model"]["amplitudes"][0][0] = 0.123;
        {
            std::ofstream file(root / "profiles/kodak_portra_endura.json");
            file << modified;
        }
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(selected.printSource->sample_density_curves(1.1).hash, old.hash);
        auto newer = library.selected_profiles_for_route({"kodak_portra_400", "kodak_portra_endura", Spektrafilm::ScanRoute::NegativePrintScan});
        ASSERT_TRUE(newer.valid);
        EXPECT_NE(newer.printSource->profile()->assetVersionToken, native->assetVersionToken);
        EXPECT_NE(newer.printSource->sample_density_curves(1.1).hash, old.hash);
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        selected = {};
        newer = {};
        EXPECT_EQ(fj_test_profile_live_owners(), 0u);
        EXPECT_FALSE(native->data.logExposure.empty());
        std::size_t nativeCapacity = sizeof(*native) + native->data.logExposure.capacity() * sizeof(float) + native->data.densityCurves.capacity() * sizeof(std::array<float, 3>);
        std::printf("Print independent native cold copy capacity=%zu; gamma totals copy capacity=%zu; source lease inline=%zu; warm copies=0; transient gamma Rust/native overlap logical=%zu (allocator/RSS excluded)\n", nativeCapacity, old.totals.capacity() * sizeof(std::array<float, 3>), sizeof(JuicerAssets::PrintProfileSource), 2 * old.totals.size() * sizeof(std::array<float, 3>));
    }
    fs::path consumer_resources(const char* name) {
        const auto root = scratch / name;
        fs::create_directories(root);
        fs::copy(resources, root, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        return root;
    }
    void save_film(const fs::path& root, const Json& document) {
        {
            std::ofstream file(root / "profiles/kodak_portra_400.json");
            file << document;
        }
        expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
    }
    ParamSnapshot direct_controls() {
        ParamSnapshot controls;
        controls.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        controls.spectralUpsamplingMode = 1;
        controls.dirCouplers.active = false;
        controls.outputGamutCompressionEnabled = 0;
        return controls;
    }

    TEST(ProductionProfileConsumer, TypedOriginalBbReachesEveryProfileIlluminantConsumer) {
        const auto root = consumer_resources("bb-consumers");
        const auto original = read_json(root / "profiles/kodak_portra_400.json");
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        for (const auto& item : std::array<std::pair<const char*, double>, 3>{{{" BB+3200.5 ", 3200.5}, {"bb3.2005e3", 3200.5}, {"BB6500.25", 6500.25}}}) {
            SCOPED_TRACE(item.first);
            auto changed = original;
            changed["info"]["reference_illuminant"] = item.first;
            changed["info"]["viewing_illuminant"] = item.first;
            save_film(root, changed);
            auto controls = direct_controls();
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            ASSERT_TRUE(build_direct_render_state_product(controls, product, diagnostic)) << diagnostic;
            EXPECT_EQ(product.recipe.filmRaw.referenceIlluminant, item.first);
            EXPECT_EQ(product.recipe.scannerOutput.viewingIlluminant, item.first);
            const auto& reference = product.recipe.profileRoute.filmProfile->info.referenceIlluminant;
            ASSERT_TRUE(std::holds_alternative<Profiles::BlackbodyIlluminant>(reference.kind));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(std::get<Profiles::BlackbodyIlluminant>(reference.kind).temperatureKelvin), std::bit_cast<std::uint64_t>(item.second));
            std::vector<float> expected(81);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                expected[i] = Spectral::planck_blackbody({380.0f + 5.0f * static_cast<float>(i), static_cast<float>(item.second)});
            }
            Spectral::mean_power_normalize(expected);
            EXPECT_EQ(product.payload.exposureTables.illum, expected);
            EXPECT_EQ(product.payload.scannerTables.illum, expected);
            EXPECT_NE(product.recipe.scannerOutput.syntheticFilmReference.hash, 0u);
            // The print scanner's authored viewing label also binds a typed owner.
            auto printDocument = read_json(resources / "profiles/kodak_portra_endura.json");
            printDocument["info"]["viewing_illuminant"] = item.first;
            {
                std::ofstream file(root / "profiles/kodak_portra_endura.json");
                file << printDocument;
            }
            expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            controls.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
            ASSERT_TRUE(build_print_render_state_product(controls, product, diagnostic)) << diagnostic;
            EXPECT_EQ(product.payload.scannerTables.illum, expected);
        }
        for (const auto* label : {"BB0", "BB-0.0", "BB-3200", "BB1e-400"}) {
            auto changed = original;
            changed["info"]["reference_illuminant"] = label;
            save_film(root, changed);
            const auto selected = JuicerProcess::root().assets().selected_film_profile_for_key("kodak_portra_400");
            ASSERT_NE(selected, nullptr);
            ASSERT_TRUE(std::holds_alternative<Profiles::BlackbodyIlluminant>(selected->info.referenceIlluminant.kind));
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            EXPECT_FALSE(build_direct_render_state_product(direct_controls(), product, diagnostic));
            EXPECT_FALSE(diagnostic.empty());
        }
    }

    TEST(ProductionProfileConsumer, CanonicalAxesIgnoreAuthoredLabelsWhileIdentityRetainsThem) {
        const auto root = consumer_resources("canonical-consumers");
        const auto original = read_json(root / "profiles/kodak_portra_400.json");
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto controls = direct_controls();
        controls.printProfileKey = "missing-unselected-print";
        {
            std::ofstream file(root / "profiles/kodak_portra_endura.json");
            file << R"({"info":{"stock":"kodak_portra_endura","stage":"printing"},"data":{"wavelengths":[]}})";
        }
        FocusedRenderStateBuildProduct baseline;
        std::string diagnostic;
        ASSERT_TRUE(build_direct_render_state_product(controls, baseline, diagnostic)) << diagnostic;
        auto changed = original;
        for (std::size_t i = 0; i < 81; ++i) {
            changed["data"]["wavelengths"][i] = 381.0 + 5.0 * static_cast<double>(i);
        }
        save_film(root, changed);
        FocusedRenderStateBuildProduct shifted;
        ASSERT_TRUE(build_direct_render_state_product(controls, shifted, diagnostic)) << diagnostic;
        EXPECT_NE(shifted.recipe.profileRoute.filmProfileAssetVersionToken, baseline.recipe.profileRoute.filmProfileAssetVersionToken);
        EXPECT_EQ(shifted.payload.exposureTables.lambda, baseline.payload.exposureTables.lambda);
        EXPECT_EQ(shifted.payload.scannerTables.lambda, baseline.payload.scannerTables.lambda);
        EXPECT_EQ(std::memcmp(shifted.payload.exposureTables.epsC.data(), baseline.payload.exposureTables.epsC.data(), baseline.payload.exposureTables.epsC.size() * sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(shifted.payload.scannerTables.epsC.data(), baseline.payload.scannerTables.epsC.data(), baseline.payload.scannerTables.epsC.size() * sizeof(float)), 0);
        EXPECT_EQ(shifted.recipe.filmRaw.finalSensitivity, baseline.recipe.filmRaw.finalSensitivity);
        EXPECT_EQ(shifted.recipe.profileRoute.filmProfile->data.wavelengths.front(), 381.0f);
        changed["data"]["wavelengths"][0] = 1e40;
        save_film(root, changed);
        ASSERT_TRUE(build_direct_render_state_product(controls, shifted, diagnostic)) << diagnostic;
        EXPECT_TRUE(std::isinf(shifted.recipe.profileRoute.filmProfile->data.wavelengths[0]));
        EXPECT_EQ(shifted.payload.exposureTables.lambda, baseline.payload.exposureTables.lambda);
        EXPECT_EQ(std::memcmp(shifted.payload.scannerTables.epsC.data(), baseline.payload.scannerTables.epsC.data(), baseline.payload.scannerTables.epsC.size() * sizeof(float)), 0);
    }

    TEST(ProductionProfileConsumer, HanatosWindowBroaderValuesReachAcceptedFormula) {
        const auto root = consumer_resources("hanatos-consumers");
        const auto original = read_json(root / "profiles/kodak_portra_400.json");
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto controls = direct_controls();
        controls.spectralUpsamplingMode = 0;
        controls.hanatos2025AdaptationWindow = 1;
        controls.hanatos2025AdaptationSurface = 0;
        for (double width : {-100.0, 1e40, 0.0}) {
            SCOPED_TRACE(width);
            auto changed = original;
            changed["data"]["hanatos2025_adaptation_window_params"] = {380.0, width, 780.0, 100.0};
            save_film(root, changed);
            const auto source = JuicerProcess::root().assets().selected_film_profile_for_key("kodak_portra_400");
            ASSERT_NE(source, nullptr);
            EXPECT_EQ(source->data.hanatos2025AdaptationWindowParams[1], static_cast<float>(width));
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            const bool built = build_direct_render_state_product(controls, product, diagnostic);
            if (width == 0.0) {
                EXPECT_FALSE(built);
                EXPECT_NE(diagnostic.find("sensitivity"), std::string::npos) << diagnostic;
            } else {
                ASSERT_TRUE(built) << diagnostic;
                EXPECT_TRUE(product.recipe.filmRaw.hanatos.applyWindow);
                EXPECT_NE(product.recipe.filmRaw.finalSensitivityHash, 0u);
            }
        }
    }

    TEST(ProductionProfileConsumer, BroaderAxesAreCopiedAndReachLocalComputations) {
        const auto root = consumer_resources("axis-consumers");
        const auto original = read_json(root / "profiles/kodak_portra_400.json");
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        const std::array<Json, 4> axes{Json::array({0.0}), Json::array({-1.0, 0.0, 0.0, 1.0}), Json::array({1.00000004, 1.00000003}), Json::array({-1e40, 0.0, 1e40})};
        for (const auto& axis : axes) {
            SCOPED_TRACE(axis.dump());
            auto changed = original;
            changed["data"]["log_exposure"] = axis;
            save_film(root, changed);
            const auto selected = JuicerProcess::root().assets().selected_film_profile_for_key("kodak_portra_400");
            ASSERT_NE(selected, nullptr);
            ASSERT_EQ(selected->data.logExposure.size(), axis.size());
            ASSERT_EQ(selected->data.densityCurves.size(), axis.size());
            for (std::size_t i = 0; i < axis.size(); ++i) {
                EXPECT_EQ(selected->data.logExposure[i], static_cast<float>(axis[i].get<double>()));
            }
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            if (build_direct_render_state_product(direct_controls(), product, diagnostic)) {
                EXPECT_EQ(product.recipe.profileRoute.filmProfile, selected);
                EXPECT_EQ(product.recipe.filmDevelop.normalizedDensityCurves.size(), axis.size());
            } else {
                EXPECT_FALSE(diagnostic.empty());
                EXPECT_NE(diagnostic.find("field="), std::string::npos) << diagnostic;
            }
            auto active = direct_controls();
            active.dirCouplers.active = true;
            if (axis.size() == 1 || std::isinf(selected->data.logExposure.front())) {
                EXPECT_FALSE(build_direct_render_state_product(active, product, diagnostic));
                EXPECT_FALSE(diagnostic.empty());
            }
        }
    }

    TEST(ProductionProfileConsumer, AuthoredHashEncodesSpecialValuesWithoutChangingOtherLanes) {
        const auto root = consumer_resources("hash-consumers");
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto controls = direct_controls();
        controls.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        ASSERT_TRUE(build_print_render_state_product(controls, product, diagnostic)) << diagnostic;
        auto film = std::make_shared<Profiles::FilmProfile>(*product.recipe.profileRoute.filmProfile);
        film->data.wavelengths.fill(0.0f);
        const std::array<std::uint32_t, 6> special{0, 0x80000000u, 0x7fc00001u, 0xff800123u, 0x7f800000u, 0xff800000u};
        for (std::size_t i = 0; i < special.size(); ++i) {
            film->data.wavelengths[i] = std::bit_cast<float>(special[i]);
        }
        product.recipe.profileRoute.filmProfile = film;
        JuicerCuda::PrintResourceDescriptors descriptor;
        ASSERT_TRUE(JuicerCuda::build_print_resource_descriptors(product.recipe, descriptor, diagnostic)) << diagnostic;
        const auto channel = Hash::hash_float_span_with_nan_mask(&film->data.channelDensity[0][0], 243);
        const auto base = Hash::hash_float_span_with_nan_mask(film->data.baseDensity.data(), 81);
        // Independent fixed byte-stream expectation for these 81 authored labels.
        constexpr std::uint64_t kAuthoredLabelsHash = 18245953401382371285ull;
        EXPECT_EQ(descriptor.filmDensityTables.densityTablesHash, Hash::hash_uint64_values({kAuthoredLabelsHash, channel.valueHash, channel.nanMaskHash, base.valueHash, base.nanMaskHash}));
        const auto& print = product.recipe.profileRoute.printProfile->data;
        const auto sensitivity = Hash::hash_float_span_with_nan_mask(&print.linearSensitivity[0][0], 243);
        EXPECT_EQ(descriptor.profileTables.sensitivitiesHash, Hash::hash_uint64_values({sensitivity.valueHash, sensitivity.nanMaskHash}));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(film->data.wavelengths[3]), special[3]);
    }

} // namespace

namespace JuicerAssets::ProfileTest {
    void before_catalog_publication() {
        if (catalogPublicationBarrier) {
            catalogPublicationBarrier->arrive_and_wait();
            catalogPublicationBarrier->arrive_and_wait();
        }
    }
    void film_view(FjFilmProfileView& view) {
        if (conversionFault == 1) {
            conversionFault = 0;
            view.tables.density_curves_cmy.count -= 1;
        }
        if (conversionFault == 2) {
            conversionFault = 0;
            view.polarity = 99;
        }
    }
    void print_view(FjPrintProfileView& view) {
        (void)view;
    }
    void after_tables() {
        if (conversionFault == 4) {
            conversionFault = 0;
            failedCandidateReady = true;
            failedCandidateReady.notify_all();
            while (!winnerPublished.load()) {
                winnerPublished.wait(false);
            }
            throw std::bad_alloc();
        }
        if (std::exchange(conversionFault, 0) == 3) {
            throw std::bad_alloc();
        }
    }
    void before_publication() {
        if (publicationBarrier) {
            const auto owners = fj_test_profile_live_owners();
            auto peak = peakProfileOwners.load();
            while (peak < owners && !peakProfileOwners.compare_exchange_weak(peak, owners)) {
            }
            publicationBarrier->arrive_and_wait();
        }
    }
} // namespace JuicerAssets::ProfileTest

int main(int argc, char** argv) {
    try {
        ::testing::InitGoogleTest(&argc, argv);
        if (argc != 4) {
            std::fprintf(stderr, "resources, fixtures and scratch paths required\n");
            return 2;
        }
        resources = argv[1];
        fixtures = argv[2];
        scratch = argv[3];
        fs::create_directories(scratch);
        return RUN_ALL_TESTS();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "production profile fixture: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("production profile fixture: unknown exception\n", stderr);
        return 1;
    }
}
