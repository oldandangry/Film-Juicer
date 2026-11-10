#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "native_path.h"
#include "owner_fixture.h"
#include "../ofx/context_host.h"
#include "juicer_ofx_api.h"
#include "juicer_test_api.h"
#include "case_results.h"

static_assert(sizeof(FjCatalogEntryView) == 40 && alignof(FjCatalogEntryView) == 8);
static_assert(offsetof(FjCatalogEntryView, key) == 0 && offsetof(FjCatalogEntryView, label) == 16 &&
              offsetof(FjCatalogEntryView, polarity) == 32);
static_assert(sizeof(FjCatalogCounts) == 16 && alignof(FjCatalogCounts) == 8);
static_assert(offsetof(FjCatalogCounts, film_count) == 0 && offsetof(FjCatalogCounts, print_count) == 8);
static_assert(std::is_same_v<decltype(&fj_test_assets_create), FjStatus (*)(FjPathView, FjAssets**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_assets_destroy), FjStatus (*)(FjAssets*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_catalog_acquire), FjStatus (*)(const FjAssets*, FjCatalog**, FjCatalogCounts*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_catalog_entry), FjStatus (*)(const FjCatalog*, std::uint32_t, std::size_t, FjCatalogEntryView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_catalog_release), FjStatus (*)(FjCatalog*, FjErrorBuffer*)>);

namespace {
    namespace fs = std::filesystem;

    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }
    void require_status(FjStatus result, std::uint32_t category) {
        require(result.category == category && result.api == FJ_API_NONE && result.native_code == 0, "wrong catalog status origin");
    }
    struct Assets {
        FjAssets* handle = nullptr;
        explicit Assets(const fs::path& root) {
            const JuicerAssets::NativePathArgument path(root);
            require_status(fj_test_assets_create(path.view(), &handle, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Assets() {
            (void)fj_test_assets_destroy(handle, nullptr);
        }
        Assets(const Assets&) = delete;
        Assets& operator=(const Assets&) = delete;
    };
    struct Catalog {
        FjCatalog* handle = nullptr;
        FjCatalogCounts counts{};
        explicit Catalog(const FjAssets* assets) {
            require_status(fj_test_catalog_acquire(assets, &handle, &counts, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Catalog() {
            (void)fj_test_catalog_release(handle, nullptr);
        }
        Catalog(const Catalog&) = delete;
        Catalog& operator=(const Catalog&) = delete;
        FjCatalogEntryView entry(std::uint32_t role, std::size_t index) const {
            FjCatalogEntryView view{};
            require_status(fj_test_catalog_entry(handle, role, index, &view, nullptr), FJ_STATUS_SUCCESS);
            return view;
        }
    };
    std::string text(FjStringView view) {
        return view.count ? std::string(view.data, view.count) : std::string();
    }
    void write(const fs::path& path, const std::string& bytes) {
        fs::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        require(bool(file), "catalog fixture write failed");
    }
    void minimal_catalog(const fs::path& root) {
        write(root / "profiles/film.json", R"({"info":{"stock":"kodak_portra_400","name":"","type":"positive"},"data":{"wavelengths":[]}})");
        write(root / "profiles/print.json", R"({"info":{"stock":"kodak_portra_endura","name":"A\u0000B","stage":"printing","support":"paper"},"data":{"wavelengths":[]}})");
        write(root / "profiles/ignored.json", "{}");
        write(root / "profiles/malformed.json", "{");
        write(root / "profiles/unsupported.json", R"({"info":{"stock":"unavailable","type":"other"},"data":{"wavelengths":[]}})");
    }

    void input_and_lifetime(const fs::path& root) {
        Assets assets(root);
        Catalog first(assets.handle);
        auto film = first.entry(FJ_PROFILE_ROLE_FILM, 0);
        auto print = first.entry(FJ_PROFILE_ROLE_PRINT, 0);
        JuicerTests::Cohort descriptors("catalog_owner.entry_descriptors", {"film/positive-empty-label", "print/embedded-nul-label"});
        {
            auto event = descriptors.case_event("film/positive-empty-label");
            require(first.counts.film_count == 1 && !film.label.data && film.label.count == 0 && film.polarity == FJ_POLARITY_POSITIVE, "classification/empty film label/polarity changed");
            event.passed();
        }
        {
            auto event = descriptors.case_event("print/embedded-nul-label");
            require(first.counts.print_count == 1 && text(print.label) == std::string("A\0B", 3), "embedded NUL print label was truncated");
            event.passed();
        }
        FjCatalog* cleared = first.handle;
        FjCatalogCounts counts{999, 999};
        require_status(fj_test_catalog_acquire(assets.handle, &cleared, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(!cleared, "other valid acquire slot not cleared");
        require_status(fj_test_catalog_acquire(assets.handle, nullptr, &counts, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(counts.film_count == 0 && counts.print_count == 0, "other valid counts not cleared");
        require_status(fj_test_catalog_acquire(nullptr, &cleared, &counts, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        const std::array invalid_entry_ids{"unknown-role/index-zero", "film/count-index"};
        JuicerTests::Cohort invalid_entries("catalog_owner.invalid_entry", {invalid_entry_ids[0], invalid_entry_ids[1]});
        const std::array invalid_entry_requests{std::pair{99u, std::size_t{0}}, std::pair{FJ_PROFILE_ROLE_FILM, first.counts.film_count}};
        for (std::size_t i = 0; i < invalid_entry_requests.size(); ++i) {
            auto event = invalid_entries.case_event(invalid_entry_ids[i]);
            const auto [role, index] = invalid_entry_requests[i];
            auto out = film;
            require_status(fj_test_catalog_entry(first.handle, role, index, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(!out.key.data && !out.label.data && !out.key.count && out.polarity == 0, "entry failure not cleared");
            event.passed();
        }
        require_status(fj_test_catalog_entry(nullptr, 0, 0, &film, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        film = first.entry(0, 0);
        std::array<std::thread, 8> readers;
        std::array<bool, 8> passed{};
        for (std::size_t i = 0; i < readers.size(); ++i) {
            readers[i] = std::thread([&, i] {
                Catalog other(assets.handle);
                const auto repeated = other.entry(0, 0);
                const auto sharedRead = first.entry(0, 0);
                passed[i] = repeated.key.data == film.key.data && sharedRead.key.data == film.key.data && text(repeated.key) == "kodak_portra_400";
            });
        }
        for (auto& reader : readers) {
            reader.join();
        }
        for (bool pass : passed) {
            require(pass, "concurrent acquisition changed source publication");
        }

        Catalog independent(assets.handle);
        FjErrorBuffer malformed{nullptr, 1, 99};
        require_status(fj_test_assets_destroy(std::exchange(assets.handle, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 2, "destroy revoked retained catalog");
        require(text(first.entry(0, 0).key) == "kodak_portra_400" && text(independent.entry(1, 0).label) == std::string("A\0B", 3), "views expired with Assets");
        require_status(fj_test_catalog_release(std::exchange(independent.handle, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        require(text(first.entry(0, 0).key) == "kodak_portra_400", "independent release revoked first handle");
        require_status(fj_test_catalog_release(nullptr, nullptr), FJ_STATUS_SUCCESS);
        require_status(fj_test_assets_destroy(nullptr, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        FjErrorBuffer zero{nullptr, 0, 99};
        require_status(fj_test_assets_destroy(nullptr, &zero), FJ_STATUS_SUCCESS);
        require(zero.length == 0, "zero-capacity diagnostic length not cleared");
        std::array<char, 4> bytes{};
        FjErrorBuffer tiny{bytes.data(), bytes.size(), 99};
        require_status(fj_test_catalog_entry(nullptr, 0, 0, &print, &tiny), FJ_STATUS_UNSUPPORTED_INPUT);
        require(tiny.length == 3 && bytes.back() == '\0', "truncated diagnostic changed status or termination");
    }

    void root_shapes(const fs::path& root) {
        const JuicerAssets::NativePathArgument argument(root);
        const auto valid = argument.view();
        FjAssets* out = nullptr;
        const std::vector<std::string> invalid_path_ids{"null-data", "empty-path", "unknown-encoding-zero", "wrong-platform-encoding", "maximum-count"};
        JuicerTests::Cohort invalid_paths("catalog_owner.root_path", invalid_path_ids);
        const std::array invalid_path_views{FjPathView{nullptr, 1, valid.encoding}, FjPathView{valid.data, 0, valid.encoding}, FjPathView{valid.data, valid.count, 0}, FjPathView{valid.data, valid.count, valid.encoding == 1 ? 2u : 1u}, FjPathView{valid.data, std::numeric_limits<std::size_t>::max(), valid.encoding}};
        for (std::size_t i = 0; i < invalid_path_views.size(); ++i) {
            auto event = invalid_paths.case_event(invalid_path_ids[i]);
            const auto invalid = invalid_path_views[i];
            require_status(fj_test_assets_create(invalid, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(!out, "invalid root published Assets");
            event.passed();
        }
        const fs::path nulPath(fs::path::string_type(3, 0));
        const JuicerAssets::NativePathArgument nul(nulPath);
        JuicerTests::Cohort path_edges("catalog_owner.root_path_edge", {"embedded-nul", "malformed-diagnostic"});
        {
            auto event = path_edges.case_event("embedded-nul");
            require_status(fj_test_assets_create(nul.view(), &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            event.passed();
        }
        {
            auto event = path_edges.case_event("malformed-diagnostic");
            FjErrorBuffer oversized{nullptr, std::numeric_limits<std::size_t>::max(), 99};
            require_status(fj_test_assets_create(valid, &out, &oversized), FJ_STATUS_UNSUPPORTED_INPUT);
            require(!out && oversized.length == 0, "malformed diagnostics did not precede operation");
            event.passed();
        }
#if defined(_WIN32)
        const std::array<std::uint16_t, 2> units{65, 66};
        JuicerTests::Cohort alignment("catalog_owner.root_path_alignment", {"unaligned-wide"});
        auto event = alignment.case_event("unaligned-wide");
        require_status(fj_test_assets_create({reinterpret_cast<const char*>(units.data()) + 1, 1, FJ_PATH_WINDOWS_WIDE}, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        event.passed();
#endif
    }

    void sticky_source_and_retry(const fs::path& scratch) {
        const auto root = scratch / "retry";
        minimal_catalog(root);
        {
            JuicerTestSupport::RuntimeFixture runtime;
            runtime.create(root);
            const auto* const assets = runtime.borrow().assets;
            JuicerTests::Cohort source_faults("catalog_owner.source_fault", {"allocation", "panic"});
            for (unsigned fault : {2u, 1u}) {
                auto event = source_faults.case_event(fault == 2 ? "allocation" : "panic");
                fj_test_catalog_fault(fault);
                FjCatalog* out = nullptr;
                FjCatalogCounts counts{999, 999};
                require_status(fj_test_catalog_acquire(assets, &out, &counts, nullptr),
                               fault == 2 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE);
                require(!out && counts.film_count == 0 && counts.print_count == 0 &&
                            fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 0,
                        "incomplete catalog acquisition published or leaked");
                event.passed();
            }
            fs::remove_all(root / "profiles");
            {
                Catalog catalog(assets);
                require(catalog.counts.film_count == 1 && catalog.counts.print_count == 1,
                        "catalog retry rediscovered source");
                const auto film = catalog.entry(FJ_PROFILE_ROLE_FILM, 0);
                const auto print = catalog.entry(FJ_PROFILE_ROLE_PRINT, 0);
                std::uintptr_t backing = 0;
                require_status(fj_test_catalog_backing_identity(catalog.handle, &backing, nullptr), FJ_STATUS_SUCCESS);
                require(backing != 0, "missing catalog backing identity");
                Catalog warm(assets);
                std::uintptr_t repeated = 0;
                require_status(fj_test_catalog_backing_identity(warm.handle, &repeated, nullptr), FJ_STATUS_SUCCESS);
                require(repeated == backing && warm.entry(FJ_PROFILE_ROLE_FILM, 0).key.data == film.key.data &&
                            warm.entry(FJ_PROFILE_ROLE_PRINT, 0).label.data == print.label.data &&
                            fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 2,
                        "warm catalog did not reuse actual source backing");
            }
            require(fj_test_catalog_live_owners() == 0 && runtime.close().category == FJ_STATUS_SUCCESS &&
                        fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0,
                    "catalog retry owner did not close cleanly");
        }

        const auto missing = scratch / "sticky-error";
        {
            JuicerTestSupport::RuntimeFixture runtime;
            runtime.create(missing);
            const auto* const assets = runtime.borrow().assets;
            fj_test_catalog_fault(3);
            FjCatalog* out = nullptr;
            FjCatalogCounts counts{999, 999};
            require_status(fj_test_catalog_acquire(assets, &out, &counts, nullptr), FJ_STATUS_ALLOCATION_FAILURE);
            require(!out && counts.film_count == 0 && counts.print_count == 0 &&
                        fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 0,
                    "source diagnostic allocation published an incomplete catalog");
            minimal_catalog(missing);
            std::array<char, 1024> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            const FjStatus status = fj_test_catalog_acquire(assets, &out, &counts, &error);
            require(status.category == FJ_STATUS_PREPARATION_FAILURE && status.api == FJ_API_NONE &&
                        status.native_code == 0 && !out && counts.film_count == 0 && counts.print_count == 0 &&
                        error.length != 0 && fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 0,
                    "source error lost stickiness or retained a catalog handle");
            require(runtime.close().category == FJ_STATUS_SUCCESS && fj_test_assets_live_owners() == 0 &&
                        fj_test_catalog_live_owners() == 0,
                    "sticky-error Runtime did not close cleanly");
        }

        JuicerTests::Cohort catalog_rejections("catalog_owner.key_rejection", {"duplicate-film-key", "missing-print-default"});
        const auto duplicate = scratch / "duplicate";
        minimal_catalog(duplicate);
        fs::copy_file(duplicate / "profiles/film.json", duplicate / "profiles/duplicate.json");
        const auto absentDefault = scratch / "missing-default";
        minimal_catalog(absentDefault);
        fs::remove(absentDefault / "profiles/print.json");
        for (const auto& [id, path] : {std::pair{"duplicate-film-key", duplicate}, std::pair{"missing-print-default", absentDefault}}) {
            JuicerTestSupport::RuntimeFixture runtime;
            runtime.create(path);
            auto event = catalog_rejections.case_event(id);
            FjCatalog* out = nullptr;
            FjCatalogCounts counts{};
            require_status(fj_test_catalog_acquire(runtime.borrow().assets, &out, &counts, nullptr), FJ_STATUS_PREPARATION_FAILURE);
            require(!out && counts.film_count == 0 && counts.print_count == 0 &&
                        fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 0,
                    "invalid catalog accepted or retained a handle");
            event.passed();
            require(runtime.close().category == FJ_STATUS_SUCCESS, "invalid catalog Runtime did not close cleanly");
        }
    }

    struct CatalogPaths {
        fs::path scratch;
        fs::path bundled;
    };

    void exact_paths_and_consumers(const CatalogPaths& paths) {
        const auto& [scratch, bundled] = paths;
        fs::path::string_type unusual;
#if defined(_WIN32)
        unusual = L"native-\u00e5-\U0001f600-";
        unusual.push_back(static_cast<wchar_t>(0xd800));
#else
        unusual = "native-\xc3\xa5-\xf0\x9f\x98\x80-\xff";
#endif
#if defined(_WIN32)
        const auto root = scratch / fs::path(unusual);
#else
        // WSL's Windows-mounted volume can replace invalid UTF-8 names. Use
        // the native Linux volume for this bounded OS-open fixture and remove
        // it at scope exit; reports stay in the configured artifact directory.
        std::array<char, 40> pattern{};
        const std::string prefix = (fs::temp_directory_path() / "juicer-catalog-XXXXXX").native();
        require(prefix.size() < pattern.size(), "native temporary path too long");
        std::copy(prefix.begin(), prefix.end(), pattern.begin());
        const char* created = mkdtemp(pattern.data());
        require(created != nullptr, "could not create native Linux path fixture");
        struct NativeScratch {
            fs::path path;
            ~NativeScratch() {
                std::error_code ec;
                fs::remove_all(path, ec);
            }
        } nativeScratch{fs::path(created)};
        const auto root = nativeScratch.path / fs::path(unusual);
        (void)scratch;
#endif
        const auto filmPath = root / "profiles" / fs::path(unusual + fs::path("-film.json").native());
        const auto printPath = root / "profiles" / fs::path(unusual + fs::path("-print.json").native());
        const JuicerAssets::NativePathArgument argument(root);
#if defined(_WIN32)
        require(argument.view().count == root.native().size(), "root native unit count changed");
#else
        require(std::string(static_cast<const char*>(argument.view().data), argument.view().count) == root.native(), "root native units changed");
#endif
        std::error_code ec;
        fs::create_directories(root / "profiles", ec);
        if (ec) {
            std::printf("LIMITATION unusual-name fixture rejected by filesystem: %s; exact root-unit round trip passed\n", ec.message().c_str());
            return;
        }
        fs::copy_file(bundled / "profiles/kodak_portra_400.json", filmPath);
        fs::copy_file(bundled / "profiles/kodak_portra_endura.json", printPath);
        JuicerTestSupport::RuntimeFixture runtime;
        runtime.create(root);
        std::string filmKey, filmLabel, printKey;
        {
            Catalog catalog(runtime.borrow().assets);
            const auto film = catalog.entry(FJ_PROFILE_ROLE_FILM, 0);
            const auto print = catalog.entry(FJ_PROFILE_ROLE_PRINT, 0);
            filmKey = text(film.key);
            filmLabel = text(film.label);
            printKey = text(print.key);
            require(catalog.counts.film_count == 1 && catalog.counts.print_count == 1 &&
                        filmKey == "kodak_portra_400" && printKey == "kodak_portra_endura",
                    "catalog changed native file units");
            FjFilmDigest defaults{};
            const FjStringView key{filmKey.data(), filmKey.size()};
            require_status(fj_test_assets_film_defaults(runtime.borrow().assets, key, &defaults, nullptr), FJ_STATUS_SUCCESS);
            FjFilmProfile* rawFilm = nullptr;
            require_status(fj_test_assets_film_acquire(runtime.borrow().assets, key, &rawFilm, nullptr), FJ_STATUS_SUCCESS);
            const auto releaseFilm = [](FjFilmProfile* value) noexcept {
                if (value)
                    (void)fj_test_assets_film_release(value, nullptr);
            };
            std::unique_ptr<FjFilmProfile, decltype(releaseFilm)> filmOwner(rawFilm, releaseFilm);
            require(filmOwner != nullptr, "film profile owner missing");
            FjFilmProfileView filmView{};
            require_status(fj_test_assets_film_view(filmOwner.get(), &filmView, nullptr), FJ_STATUS_SUCCESS);
            const auto same = [](const auto& left, const auto& right) {
                return std::equal(std::begin(left), std::end(left), std::begin(right));
            };
            require(text(filmView.stock) == filmKey &&
                        same(defaults.gamma_samelayer_rgb, filmView.digest.gamma_samelayer_rgb) &&
                        same(defaults.gamma_interlayer_r_to_gb, filmView.digest.gamma_interlayer_r_to_gb) &&
                        same(defaults.gamma_interlayer_g_to_rb, filmView.digest.gamma_interlayer_g_to_rb) &&
                        same(defaults.gamma_interlayer_b_to_rg, filmView.digest.gamma_interlayer_b_to_rg) &&
                        same(defaults.halation_first_sigma_um, filmView.digest.halation_first_sigma_um) &&
                        same(defaults.halation_primary_amount, filmView.digest.halation_primary_amount) &&
                        defaults.hanatos_spectral_gaussian_blur_default == filmView.digest.hanatos_spectral_gaussian_blur_default,
                    "film defaults differ from the acquired profile");
            filmOwner.reset();
            require(runtime.shutdown().category == FJ_STATUS_SUCCESS && fj_test_assets_live_owners() == 1 &&
                        fj_test_catalog_live_owners() == 1,
                    "borrowed shutdown consumed Runtime Assets or catalog hold");
            require(text(catalog.entry(FJ_PROFILE_ROLE_FILM, 0).key) == filmKey, "borrowed shutdown expired catalog options");
        }
        require(runtime.close().category == FJ_STATUS_SUCCESS && fj_test_assets_live_owners() == 0 &&
                    fj_test_catalog_live_owners() == 0,
                "terminal cleanup did not consume Runtime owners");
        ofx_test::ContextHost host;
        auto* plugin = OfxGetPlugin(0);
        require(plugin && plugin->setHost && plugin->mainEntry, "standard OFX descriptor missing");
        plugin->setHost(host.binding());
        require(plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) == kOfxStatOK, "OFX catalog module load");
        struct ModuleGuard {
            OfxPlugin* plugin;
            ~ModuleGuard() {
                if (plugin)
                    (void)plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr);
            }
            OfxStatus close() {
                auto* owner = std::exchange(plugin, nullptr);
                return owner->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr);
            }
        } module{plugin};
        require(plugin->mainEntry(kOfxImageEffectActionDescribeInContext, host.effect_handle(), host.context_handle(), nullptr) == kOfxStatOK,
                "OFX catalog descriptor");
        const auto verify_option = [&](const char* name, const std::string& key, const std::string* label) {
            const auto definition = std::find_if(host.definitions.begin(), host.definitions.end(), [&](const auto& item) {
                return item.name == name;
            });
            require(definition != host.definitions.end(), "catalog option descriptor missing");
            const auto& keys = definition->properties.at(kOfxParamPropChoiceEnum);
            const auto& labels = definition->properties.at(kOfxParamPropChoiceOption);
            require(keys.size() == labels.size(), "catalog option labels differ in count");
            const auto found = std::find(keys.begin(), keys.end(), ofx_test::Value{key});
            require(found != keys.end(), "OFX catalog option key missing");
            const auto index = static_cast<std::size_t>(std::distance(keys.begin(), found));
            if (label)
                require(labels[index] == ofx_test::Value{*label}, "OFX catalog film label differs");
        };
        verify_option("FilmProfileKey", filmKey, &filmLabel);
        verify_option("PrintProfileKey", printKey, nullptr);
        require(module.close() == kOfxStatOK && !host.callback_error, "OFX catalog module unload");
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "OFX descriptor retained catalog owners");
    }

    void bundled_cold_and_warm_publication(const fs::path& root) {
        JuicerTestSupport::RuntimeFixture runtime;
        runtime.create(root);
        {
            Catalog catalog(runtime.borrow().assets);
            require(catalog.counts.film_count != 0 && catalog.counts.print_count != 0, "bundled production catalog unavailable");
            std::uintptr_t backing = 0;
            require_status(fj_test_catalog_backing_identity(catalog.handle, &backing, nullptr), FJ_STATUS_SUCCESS);
            const auto* const label = catalog.entry(FJ_PROFILE_ROLE_FILM, 0).label.data;
            require(backing != 0, "bundled catalog backing missing");
            struct Reentry {
                const FjAssets* assets = nullptr;
                FjCatalog* nested = nullptr;
                FjCatalogCounts counts{};
                FjStatus status{};
                ~Reentry() {
                    if (nested)
                        (void)fj_test_catalog_release(nested, nullptr);
                }
            } reentry{runtime.borrow().assets};
            struct HookGuard {
                ~HookGuard() {
                    fj_test_catalog_publication_hook(nullptr, nullptr);
                }
            } hookGuard;
            fj_test_catalog_publication_hook([](void* user) noexcept {
                auto& item = *static_cast<Reentry*>(user);
                item.status = fj_test_catalog_acquire(item.assets, &item.nested, &item.counts, nullptr);
            },
                                             &reentry);
            {
                Catalog outer(runtime.borrow().assets);
                const auto release = [](FjCatalog* value) noexcept {
                    if (value)
                        (void)fj_test_catalog_release(value, nullptr);
                };
                std::unique_ptr<FjCatalog, decltype(release)> nested(std::exchange(reentry.nested, nullptr), release);
                require_status(reentry.status, FJ_STATUS_SUCCESS);
                require(nested && reentry.counts.film_count == catalog.counts.film_count &&
                            reentry.counts.print_count == catalog.counts.print_count,
                        "reentrant catalog acquisition did not complete");
                std::uintptr_t inner = 0, outerIdentity = 0;
                require_status(fj_test_catalog_backing_identity(nested.get(), &inner, nullptr), FJ_STATUS_SUCCESS);
                require_status(fj_test_catalog_backing_identity(outer.handle, &outerIdentity, nullptr), FJ_STATUS_SUCCESS);
                require(inner == backing && outerIdentity == backing, "reentrant acquisition did not share real catalog backing");
            }
            for (unsigned i = 0; i < 32; ++i) {
                Catalog warm(runtime.borrow().assets);
                std::uintptr_t repeated = 0;
                require_status(fj_test_catalog_backing_identity(warm.handle, &repeated, nullptr), FJ_STATUS_SUCCESS);
                require(repeated == backing && warm.entry(FJ_PROFILE_ROLE_FILM, 0).label.data == label &&
                            fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 2,
                        "warm catalog did not retain the same backing and key bytes");
            }
            require(fj_test_catalog_live_owners() == 1, "warm catalog handles leaked");
        }
        require(runtime.close().category == FJ_STATUS_SUCCESS && fj_test_assets_live_owners() == 0 &&
                    fj_test_catalog_live_owners() == 0,
                "bundled Runtime publication did not close cleanly");
    }
} // namespace

int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            for (const char* id : {"root_shapes", "input_and_lifetime", "sticky_source_and_retry", "exact_paths_and_consumers", "bundled_publication"})
                JuicerTests::list_case("catalog_owner", id);
            return 0;
        }
        require(argc == 3, "catalog fixture needs scratch and bundled Resources paths");
        const fs::path scratch(argv[1]);
        fs::remove_all(scratch);
        minimal_catalog(scratch / "minimal");
        {
            JuicerTests::CaseEvent event("catalog_owner", "root_shapes");
            root_shapes(scratch / "minimal");
            event.passed();
        }
        {
            JuicerTests::CaseEvent event("catalog_owner", "input_and_lifetime");
            input_and_lifetime(scratch / "minimal");
            event.passed();
        }
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "raw ownership leaked");
        {
            JuicerTests::CaseEvent event("catalog_owner", "sticky_source_and_retry");
            sticky_source_and_retry(scratch);
            event.passed();
        }
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "native publication leaked");
        {
            JuicerTests::CaseEvent event("catalog_owner", "exact_paths_and_consumers");
            exact_paths_and_consumers({.scratch = scratch, .bundled = argv[2]});
            event.passed();
        }
        {
            JuicerTests::CaseEvent event("catalog_owner", "bundled_publication");
            bundled_cold_and_warm_publication(argv[2]);
            event.passed();
        }
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "bundled publication owners accumulated after drop");
        std::puts("PASS catalog ABI/production options, publication/retry, source stickiness, concurrent acquisition, retained borrows and native opens");
        case_log.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL catalog: %s\n", error.what());
        return 1;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        std::fprintf(stderr, "FAIL catalog boundary: %s\n", error.failure.diagnostic.c_str());
        return 1;
    }
}
