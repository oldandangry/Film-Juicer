#include <array>
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
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "ResourceAssetLibrary.h"
#include "test_asset_workspace.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ColorTransforms.h"
#include "Hash.h"
#include "Cuda/JuicerCudaResources.h"
#include "juicer_cuda_owner.h"
#include "RustAssetBridge.h"
#include "PreparedState.h"
#include "juicer_test_api.h"
#include "exposure_fixture.h"

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
static_assert(std::is_same_v<decltype(&fj_legacy_assets_release_derived_payloads), FjStatus (*)(const FjAssets*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_assets_release_source_payloads), FjStatus (*)(const FjAssets*, FjErrorBuffer*)>);

namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    fs::path resources;
    fs::path fixtures;
    fs::path scratch;
    std::barrier<>* catalogPublicationBarrier = nullptr;

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
        Film(const FjAssets* assets, const std::string& key) {
            expect_status(fj_legacy_film_profile_acquire(assets, text(key), &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        Film(const Assets& assets, const std::string& key)
            : Film(assets.owner, key) {}
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
        Print(const FjAssets* assets, const std::string& key) {
            expect_status(fj_legacy_print_profile_acquire(assets, text(key), &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        Print(const Assets& assets, const std::string& key)
            : Print(assets.owner, key) {}
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
    void expect_span_bits(FjFloatSpan actual, const std::vector<float>& expected) {
        ASSERT_EQ(actual.count, expected.size());
        for (std::size_t i = 0; i < actual.count; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.data[i]),
                      std::bit_cast<std::uint32_t>(expected[i]));
        }
    }
    std::vector<float> copy_span(FjFloatSpan span) {
        return span.count == 0 ? std::vector<float>{}
                               : std::vector<float>(span.data, span.data + span.count);
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


    void check_cleanup_precedence(bool earlier) {
        JuicerAssets::AssetBridge bridge(resources);
        ASSERT_TRUE(bridge.load_catalog().valid);
        Film original(bridge.borrow_assets(), "kodak_portra_400");
        Film warm(bridge.borrow_assets(), "kodak_portra_400");
        const auto before = original.view();
        ASSERT_EQ(before.tables.log_exposure.data, warm.view().tables.log_exposure.data);
        FjIlluminantsOwner* illuminants = nullptr;
        expect_status(fj_test_assets_illuminants_acquire(bridge.borrow_assets(), &illuminants, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_illuminants_release(illuminants, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_cache_clear_failures(bridge.borrow_assets(), earlier, 1, nullptr), FJ_STATUS_SUCCESS);
        std::array<char, 256> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const auto result = bridge.release_cached_payloads(&error);
        expect_status(result, earlier ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_STREQ(message.data(), earlier ? "host cache cleanup allocation failed" : "injected post-release cache failure");
        EXPECT_EQ(error.length, std::strlen(message.data()));
        FjCacheMetrics metrics{};
        expect_status(fj_test_cache_metrics(bridge.borrow_assets(), FJ_TEST_CACHE_ILLUMINANTS, &metrics, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(metrics.destructions, earlier ? 0u : 1u);
        EXPECT_EQ(original.view().tables.log_exposure.data, before.tables.log_exposure.data);
        Film fresh(bridge.borrow_assets(), "kodak_portra_400");
        const auto after = fresh.view();
        EXPECT_NE(before.tables.log_exposure.data, after.tables.log_exposure.data);
        ASSERT_EQ(before.tables.log_exposure.count, after.tables.log_exposure.count);
        EXPECT_EQ(std::memcmp(before.tables.log_exposure.data, after.tables.log_exposure.data, before.tables.log_exposure.count * sizeof(float)), 0);
        // Source release still runs after an earlier derived-cache failure; a
        // second call observes the already-consumed injected source failure.
        expect_status(fj_legacy_assets_release_source_payloads(bridge.borrow_assets(), nullptr), FJ_STATUS_SUCCESS);
        expect_status(bridge.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(original.view().tables.log_exposure.data, before.tables.log_exposure.data);
    }

    TEST(ProductionProfileOwner, RawHoldersSurviveCacheReleaseAndPreserveFirstError) {
        check_cleanup_precedence(false);
        check_cleanup_precedence(true);
    }

    TEST(ProductionProfileOwner, CacheReleaseOverlapsFirstCatalogPublication) {
        for (unsigned attempt = 0; attempt < 8; ++attempt) {
            JuicerAssets::Library library(resources);
            expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
            EXPECT_EQ(fj_test_catalog_live_owners(), 0u);
            std::barrier barrier(2);
            catalogPublicationBarrier = &barrier;
            const Spektrafilm::ProfileCatalog* selected = nullptr;
            std::exception_ptr failure;
            std::thread loader([&] {
                try {
                    selected = &library.spektrafilm_profile_catalog();
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
            ASSERT_NE(selected, nullptr);
            ASSERT_TRUE(selected->valid);
            ASSERT_FALSE(selected->filmProfiles.empty());
            ASSERT_FALSE(selected->printProfiles.empty());
            const auto* publication = selected;
            const auto* firstKey = selected->filmProfiles.front().key.data();
            expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
            EXPECT_EQ(&library.spektrafilm_profile_catalog(), publication);
            EXPECT_EQ(publication->filmProfiles.front().key.data(), firstKey);
            EXPECT_EQ(fj_test_catalog_live_owners(), 1u);
            expect_status(library.close(), FJ_STATUS_SUCCESS);
            EXPECT_EQ(fj_test_catalog_live_owners(), 0u);
            EXPECT_EQ(fj_test_assets_live_owners(), 0u);
        }
    }

    TEST(ProductionProfileOwner, CacheReleasePreservesRawPrintAndDensityOwners) {
        const auto root = JuicerTestSupport::test_asset_workspaces.create(scratch / "same-source");
        fs::create_directories(root / "profiles");
        fs::copy_file(resources / "profiles/kodak_portra_400.json", root / "profiles/kodak_portra_400.json", fs::copy_options::overwrite_existing);
        fs::copy_file(resources / "profiles/kodak_portra_endura.json", root / "profiles/kodak_portra_endura.json", fs::copy_options::overwrite_existing);
        Assets assets(root);
        Print oldPrint(assets, "kodak_portra_endura");
        const auto oldView = oldPrint.view();
        Density oldDensity(oldPrint, 1.1);
        const auto oldDensityView = oldDensity.view();
        auto modified = read_json(root / "profiles/kodak_portra_endura.json");
        modified["data"]["density_curves_model"]["amplitudes"][0][0] = 0.123;
        {
            std::ofstream file(root / "profiles/kodak_portra_endura.json");
            file << modified;
        }
        expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(oldPrint.view().asset_token, oldView.asset_token);
        EXPECT_EQ(oldDensity.view().hash, oldDensityView.hash);
        EXPECT_EQ(oldDensity.view().totals_cmy.data, oldDensityView.totals_cmy.data);
        Print newer(assets, "kodak_portra_endura");
        const auto newView = newer.view();
        EXPECT_NE(newView.asset_token, oldView.asset_token);
        Density newDensity(newer, 1.1);
        EXPECT_NE(newDensity.view().hash, oldDensityView.hash);
        EXPECT_EQ(oldPrint.view().tables.log_exposure.data, oldView.tables.log_exposure.data);
    }
    fs::path consumer_resources(const char* name) {
        const auto root = JuicerTestSupport::test_asset_workspaces.create(scratch / name);
        JuicerTestSupport::copy_consumer_assets(resources, root);
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

    bool prepared_view(
        const ParamSnapshot& controls,
        std::optional<PreparedState>& owner,
        FjPreparedStateView& view,
        std::string& diagnostic) {
        diagnostic.clear();
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjStatus status = prepare_state(
            JuicerProcess::root().assets().borrow_rust_assets(),
            controls,
            owner,
            error);
        if (status.category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(message.data(), error.length);
            return false;
        }
        status = owner->view(view, error);
        if (status.category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(message.data(), error.length);
            return false;
        }
        return true;
    }

    TEST(ProductionProfileOwner, FilmDefaultsCopiesFixedDigestAndClearsMissingResult) {
        JuicerAssets::Library library(resources);
        FjFilmDigest defaults{};
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        expect_status(library.film_defaults("kodak_portra_400", defaults, error), FJ_STATUS_SUCCESS);
        Film source(library.borrow_rust_assets(), "kodak_portra_400");
        const FjFilmProfileView view = source.view();
        EXPECT_EQ(std::memcmp(&defaults, &view.digest, sizeof(defaults)), 0);

        std::memset(&defaults, 0xa5, sizeof(defaults));
        error.length = 0;
        const FjStatus missing = library.film_defaults("missing-film-profile", defaults, error);
        EXPECT_EQ(missing.category, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(missing.api, FJ_API_NONE);
        EXPECT_EQ(missing.native_code, 0);
        const FjFilmDigest cleared{};
        EXPECT_EQ(std::memcmp(&defaults, &cleared, sizeof(defaults)), 0);
    }

    TEST(ProductionProfileConsumer, TypedOriginalBbReachesEveryProfileIlluminantConsumer) {
        const auto root = consumer_resources("bb-consumers");
        const auto original = read_json(root / "profiles/kodak_portra_400.json");
        JuicerCuda::Owner owner;
        owner.create(root);
        for (const auto& item : std::array<std::pair<const char*, double>, 3>{{{" BB+3200.5 ", 3200.5}, {"bb3.2005e3", 3200.5}, {"BB6500.25", 6500.25}}}) {
            SCOPED_TRACE(item.first);
            auto changed = original;
            changed["info"]["reference_illuminant"] = item.first;
            changed["info"]["viewing_illuminant"] = item.first;
            save_film(root, changed);
            auto controls = direct_controls();
            std::string diagnostic;
            std::optional<PreparedState> prepared;
            FjPreparedStateView preparedState{};
            ASSERT_TRUE(prepared_view(controls, prepared, preparedState, diagnostic)) << diagnostic;
            Film filmSource(JuicerProcess::root().assets().borrow_rust_assets(), "kodak_portra_400");
            const FjFilmProfileView filmView = filmSource.view();
            EXPECT_EQ(filmView.reference_illuminant.kind, FJ_ILLUMINANT_BLACKBODY);
            EXPECT_EQ(filmView.viewing_illuminant.kind, FJ_ILLUMINANT_BLACKBODY);
            EXPECT_EQ(
                std::string(filmView.reference_illuminant.label.data, filmView.reference_illuminant.label.count),
                item.first);
            EXPECT_EQ(
                std::string(filmView.viewing_illuminant.label.data, filmView.viewing_illuminant.label.count),
                item.first);
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(filmView.reference_illuminant.temperature_kelvin),
                std::bit_cast<std::uint64_t>(item.second));
            EXPECT_EQ(preparedState.film_profile_asset_version, filmView.asset_token);
#if defined(_WIN32)
            const std::string platform = "windows-clang";
#else
            const std::string platform = "linux";
#endif
#if defined(NDEBUG)
            const std::string configuration = "release";
#else
            const std::string configuration = "debug";
#endif
            std::string captureName = "construction-";
            captureName.append(platform).append("-").append(configuration).append(".json");
            const auto path = fixtures.parent_path().parent_path() / "ffi/fixtures/illuminants" / captureName;
            const auto expectations = load_shared_fixture(path, "construction-manifest.json");
            const auto temperature = std::bit_cast<std::uint32_t>(static_cast<float>(item.second));
            const auto& rows = expectations["leaf"]["blackbody"];
            const auto row = std::find_if(rows.begin(), rows.end(), [&](const auto& value) {
                return value["temperature"].template get<std::uint32_t>() == temperature;
            });
            ASSERT_NE(row, rows.end());
            std::vector<float> expected;
            for (const auto& word : (*row)["normalized"]) {
                expected.push_back(std::bit_cast<float>(word.template get<std::uint32_t>()));
            }
            expect_span_bits(preparedState.exposure_tables.illuminant, expected);
            expect_span_bits(preparedState.scanner_tables.illuminant, expected);
            EXPECT_NE(preparedState.film_reference.hash, 0u);
            // The print scanner's authored viewing label also binds a typed owner.
            auto printDocument = read_json(resources / "profiles/kodak_portra_endura.json");
            printDocument["info"]["viewing_illuminant"] = item.first;
            {
                std::ofstream file(root / "profiles/kodak_portra_endura.json");
                file << printDocument;
            }
            expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            controls.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
            prepared.reset();
            ASSERT_TRUE(prepared_view(controls, prepared, preparedState, diagnostic)) << diagnostic;
            expect_span_bits(preparedState.scanner_tables.illuminant, expected);
            Print printSource(JuicerProcess::root().assets().borrow_rust_assets(), "kodak_portra_endura");
            const FjPrintProfileView printView = printSource.view();
            EXPECT_EQ(printView.viewing_illuminant.kind, FJ_ILLUMINANT_BLACKBODY);
            EXPECT_EQ(
                std::string(printView.viewing_illuminant.label.data, printView.viewing_illuminant.label.count),
                item.first);
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(printView.viewing_illuminant.temperature_kelvin),
                std::bit_cast<std::uint64_t>(item.second));
            EXPECT_EQ(preparedState.print_profile_asset_version, printView.asset_token);
        }
        for (const auto* label : {"BB0", "BB-0.0", "BB-3200", "BB1e-400"}) {
            auto changed = original;
            changed["info"]["reference_illuminant"] = label;
            save_film(root, changed);
            Film selected(JuicerProcess::root().assets().borrow_rust_assets(), "kodak_portra_400");
            const FjFilmProfileView selectedView = selected.view();
            EXPECT_EQ(selectedView.reference_illuminant.kind, FJ_ILLUMINANT_BLACKBODY);
            std::optional<PreparedState> rejected;
            FjPreparedStateView rejectedView{};
            std::string diagnostic;
            EXPECT_FALSE(prepared_view(direct_controls(), rejected, rejectedView, diagnostic));
            EXPECT_FALSE(rejected.has_value());
            EXPECT_FALSE(diagnostic.empty());
        }
    }

    TEST(ProductionProfileConsumer, CanonicalAxesIgnoreAuthoredLabelsWhileIdentityRetainsThem) {
        const auto root = consumer_resources("canonical-consumers");
        const auto original = read_json(root / "profiles/kodak_portra_400.json");
        JuicerCuda::Owner owner;
        owner.create(root);
        auto controls = direct_controls();
        controls.printProfileKey = "missing-unselected-print";
        {
            std::ofstream file(root / "profiles/kodak_portra_endura.json");
            file << R"({"info":{"stock":"kodak_portra_endura","stage":"printing"},"data":{"wavelengths":[]}})";
        }
        std::string diagnostic;
        std::optional<PreparedState> prepared;
        FjPreparedStateView preparedState{};
        ASSERT_TRUE(prepared_view(controls, prepared, preparedState, diagnostic)) << diagnostic;
        const auto baselineFilmVersion = preparedState.film_profile_asset_version;
        const auto baselineExposureLambda = copy_span(preparedState.exposure_tables.lambda_nm);
        const auto baselineScannerLambda = copy_span(preparedState.scanner_tables.lambda_nm);
        const auto baselineExposureEpsC = copy_span(preparedState.exposure_tables.dyes_cmy[0]);
        const auto baselineScannerEpsC = copy_span(preparedState.scanner_tables.dyes_cmy[0]);
        const auto baselineSensitivity = copy_span(preparedState.film_exposure.sensitivity_rgb);
        auto changed = original;
        for (std::size_t i = 0; i < 81; ++i) {
            changed["data"]["wavelengths"][i] = 381.0 + 5.0 * static_cast<double>(i);
        }
        save_film(root, changed);
        ASSERT_TRUE(prepared_view(controls, prepared, preparedState, diagnostic)) << diagnostic;
        EXPECT_NE(preparedState.film_profile_asset_version, baselineFilmVersion);
        expect_span_bits(preparedState.exposure_tables.lambda_nm, baselineExposureLambda);
        expect_span_bits(preparedState.scanner_tables.lambda_nm, baselineScannerLambda);
        expect_span_bits(preparedState.exposure_tables.dyes_cmy[0], baselineExposureEpsC);
        expect_span_bits(preparedState.scanner_tables.dyes_cmy[0], baselineScannerEpsC);
        expect_span_bits(preparedState.film_exposure.sensitivity_rgb, baselineSensitivity);
        Film shiftedSource(JuicerProcess::root().assets().borrow_rust_assets(), "kodak_portra_400");
        const FjFilmProfileView shiftedView = shiftedSource.view();
        EXPECT_EQ(preparedState.film_profile_asset_version, shiftedView.asset_token);
        EXPECT_EQ(shiftedView.wavelengths.data[0], 381.0f);
        changed["data"]["wavelengths"][0] = 1e40;
        save_film(root, changed);
        ASSERT_TRUE(prepared_view(controls, prepared, preparedState, diagnostic)) << diagnostic;
        Film nonfiniteSource(JuicerProcess::root().assets().borrow_rust_assets(), "kodak_portra_400");
        const FjFilmProfileView nonfiniteView = nonfiniteSource.view();
        EXPECT_TRUE(std::isinf(nonfiniteView.wavelengths.data[0]));
        expect_span_bits(preparedState.exposure_tables.lambda_nm, baselineExposureLambda);
        expect_span_bits(preparedState.scanner_tables.lambda_nm, baselineScannerLambda);
        expect_span_bits(preparedState.scanner_tables.dyes_cmy[0], baselineScannerEpsC);
    }

    TEST(ProductionProfileConsumer, HanatosWindowBroaderValuesReachAcceptedFormula) {
        const auto root = consumer_resources("hanatos-consumers");
        const auto original = read_json(root / "profiles/kodak_portra_400.json");
        JuicerCuda::Owner owner;
        owner.create(root);
        auto controls = direct_controls();
        controls.spectralUpsamplingMode = 0;
        controls.hanatos2025AdaptationWindow = 1;
        controls.hanatos2025AdaptationSurface = 0;
        for (double width : {-100.0, 1e40, 0.0}) {
            SCOPED_TRACE(width);
            auto changed = original;
            changed["data"]["hanatos2025_adaptation_window_params"] = {380.0, width, 780.0, 100.0};
            save_film(root, changed);
            Film source(JuicerProcess::root().assets().borrow_rust_assets(), "kodak_portra_400");
            const FjFilmProfileView sourceView = source.view();
            ASSERT_EQ(sourceView.hanatos_window.count, 4u);
            EXPECT_EQ(sourceView.hanatos_window.data[1], static_cast<float>(width));
            std::optional<PreparedState> prepared;
            FjPreparedStateView preparedState{};
            std::string preparedDiagnostic;
            const bool preparedBuilt = prepared_view(
                controls, prepared, preparedState, preparedDiagnostic);
            if (width == 0.0) {
                EXPECT_FALSE(preparedBuilt);
                EXPECT_FALSE(prepared.has_value());
                EXPECT_NE(preparedDiagnostic.find("sensitivity"), std::string::npos)
                    << preparedDiagnostic;
            } else {
                ASSERT_TRUE(preparedBuilt) << preparedDiagnostic;
                EXPECT_EQ(preparedState.input_policy.apply_hanatos_window, 1u);
                EXPECT_NE(preparedState.film_exposure.sensitivity_hash, 0u);
            }
        }
    }

    TEST(ProductionProfileConsumer, BroaderAxesAreCopiedAndReachLocalComputations) {
        const auto root = consumer_resources("axis-consumers");
        const auto original = read_json(root / "profiles/kodak_portra_400.json");
        JuicerCuda::Owner owner;
        owner.create(root);
        const std::array<Json, 4> axes{Json::array({0.0}), Json::array({-1.0, 0.0, 0.0, 1.0}), Json::array({1.00000004, 1.00000003}), Json::array({-1e40, 0.0, 1e40})};
        for (const auto& axis : axes) {
            SCOPED_TRACE(axis.dump());
            auto changed = original;
            changed["data"]["log_exposure"] = axis;
            save_film(root, changed);
            Film source(JuicerProcess::root().assets().borrow_rust_assets(), "kodak_portra_400");
            const FjFilmProfileView sourceView = source.view();
            ASSERT_EQ(sourceView.tables.log_exposure.count, axis.size());
            ASSERT_EQ(sourceView.tables.density_curves_cmy.count, axis.size() * 3);
            for (std::size_t i = 0; i < axis.size(); ++i) {
                EXPECT_EQ(sourceView.tables.log_exposure.data[i], static_cast<float>(axis[i].get<double>()));
            }
            std::optional<PreparedState> prepared;
            FjPreparedStateView preparedState{};
            std::string preparedDiagnostic;
            const bool preparedBuilt = prepared_view(
                direct_controls(), prepared, preparedState, preparedDiagnostic);
            if (preparedBuilt) {
                EXPECT_EQ(std::string(preparedState.film_profile_key.data,
                                      preparedState.film_profile_key.count),
                          "kodak_portra_400");
                EXPECT_EQ(preparedState.film_profile_asset_version, sourceView.asset_token);
                EXPECT_EQ(preparedState.film_development.density_rgb.count,
                          axis.size() * 3);
            } else {
                EXPECT_FALSE(prepared.has_value());
                EXPECT_FALSE(preparedDiagnostic.empty());
            }
            auto active = direct_controls();
            active.dirCouplers.active = true;
            if (axis.size() == 1 || std::isinf(sourceView.tables.log_exposure.data[0])) {
                std::optional<PreparedState> activePrepared;
                EXPECT_FALSE(prepared_view(active, activePrepared, preparedState, preparedDiagnostic));
                EXPECT_FALSE(activePrepared.has_value());
                EXPECT_FALSE(preparedDiagnostic.empty());
            }
        }
    }


} // namespace

namespace JuicerAssets::ProfileTest {
    void before_catalog_publication() {
        if (catalogPublicationBarrier) {
            catalogPublicationBarrier->arrive_and_wait();
            catalogPublicationBarrier->arrive_and_wait();
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
        const int result = RUN_ALL_TESTS();
        const bool ownersDrained = fj_test_profile_live_owners() == 0 && fj_test_print_density_live_owners() == 0 && fj_test_catalog_live_owners() == 0 && fj_test_assets_live_owners() == 0;
        return JuicerTestSupport::test_asset_workspaces.finish(result, ownersDrained);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "production profile fixture: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("production profile fixture: unknown exception\n", stderr);
        return 1;
    }
}
