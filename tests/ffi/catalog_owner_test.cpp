#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
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
#include "JuicerState.h"
#include "SpectralProcessing.h"
#include "ProcessRoot.h"
#include "RustAssetBridge.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"

static_assert(sizeof(FjCatalogEntryView) == 40 && alignof(FjCatalogEntryView) == 8);
static_assert(offsetof(FjCatalogEntryView, key) == 0 && offsetof(FjCatalogEntryView, label) == 16 &&
              offsetof(FjCatalogEntryView, polarity) == 32);
static_assert(sizeof(FjCatalogCounts) == 16 && alignof(FjCatalogCounts) == 8);
static_assert(offsetof(FjCatalogCounts, film_count) == 0 && offsetof(FjCatalogCounts, print_count) == 8);
static_assert(std::is_same_v<decltype(&fj_legacy_assets_create), FjStatus (*)(FjPathView, FjAssets**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_assets_destroy), FjStatus (*)(FjAssets*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_catalog_acquire), FjStatus (*)(const FjAssets*, FjCatalog**, FjCatalogCounts*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_catalog_entry), FjStatus (*)(const FjCatalog*, std::uint32_t, std::size_t, FjCatalogEntryView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_catalog_release), FjStatus (*)(FjCatalog*, FjErrorBuffer*)>);

namespace {
    namespace fs = std::filesystem;
    unsigned conversionFault = 0;
    bool diagnosticFault = false;

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
            require_status(fj_legacy_assets_create(path.view(), &handle, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Assets() {
            (void)fj_legacy_assets_destroy(handle, nullptr);
        }
        Assets(const Assets&) = delete;
        Assets& operator=(const Assets&) = delete;
    };
    struct Catalog {
        FjCatalog* handle = nullptr;
        FjCatalogCounts counts{};
        explicit Catalog(FjAssets* assets) {
            require_status(fj_legacy_catalog_acquire(assets, &handle, &counts, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Catalog() {
            (void)fj_legacy_catalog_release(handle, nullptr);
        }
        Catalog(const Catalog&) = delete;
        Catalog& operator=(const Catalog&) = delete;
        FjCatalogEntryView entry(std::uint32_t role, std::size_t index) const {
            FjCatalogEntryView view{};
            require_status(fj_legacy_catalog_entry(handle, role, index, &view, nullptr), FJ_STATUS_SUCCESS);
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
        require(first.counts.film_count == 1 && first.counts.print_count == 1, "classification changed");
        require(!film.label.data && film.label.count == 0 && film.polarity == FJ_POLARITY_POSITIVE, "empty label/polarity lost");
        require(text(print.label) == std::string("A\0B", 3), "embedded NUL label was truncated");
        FjCatalog* cleared = first.handle;
        FjCatalogCounts counts{999, 999};
        require_status(fj_legacy_catalog_acquire(assets.handle, &cleared, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(!cleared, "other valid acquire slot not cleared");
        require_status(fj_legacy_catalog_acquire(assets.handle, nullptr, &counts, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(counts.film_count == 0 && counts.print_count == 0, "other valid counts not cleared");
        require_status(fj_legacy_catalog_acquire(nullptr, &cleared, &counts, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        for (const auto& [role, index] : {std::pair{99u, std::size_t{0}}, std::pair{FJ_PROFILE_ROLE_FILM, first.counts.film_count}}) {
            auto out = film;
            require_status(fj_legacy_catalog_entry(first.handle, role, index, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(!out.key.data && !out.label.data && !out.key.count && out.polarity == 0, "entry failure not cleared");
        }
        require_status(fj_legacy_catalog_entry(nullptr, 0, 0, &film, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
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
        require_status(fj_legacy_assets_destroy(std::exchange(assets.handle, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 2, "destroy revoked retained catalog");
        require(text(first.entry(0, 0).key) == "kodak_portra_400" && text(independent.entry(1, 0).label) == std::string("A\0B", 3), "views expired with Assets");
        require_status(fj_legacy_catalog_release(std::exchange(independent.handle, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        require(text(first.entry(0, 0).key) == "kodak_portra_400", "independent release revoked first handle");
        require_status(fj_legacy_catalog_release(nullptr, nullptr), FJ_STATUS_SUCCESS);
        require_status(fj_legacy_assets_destroy(nullptr, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        FjErrorBuffer zero{nullptr, 0, 99};
        require_status(fj_legacy_assets_destroy(nullptr, &zero), FJ_STATUS_SUCCESS);
        require(zero.length == 0, "zero-capacity diagnostic length not cleared");
        std::array<char, 4> bytes{};
        FjErrorBuffer tiny{bytes.data(), bytes.size(), 99};
        require_status(fj_legacy_catalog_entry(nullptr, 0, 0, &print, &tiny), FJ_STATUS_UNSUPPORTED_INPUT);
        require(tiny.length == 3 && bytes.back() == '\0', "truncated diagnostic changed status or termination");
    }

    void root_shapes(const fs::path& root) {
        const JuicerAssets::NativePathArgument argument(root);
        const auto valid = argument.view();
        FjAssets* out = nullptr;
        for (auto invalid : {FjPathView{nullptr, 1, valid.encoding}, FjPathView{valid.data, 0, valid.encoding}, FjPathView{valid.data, valid.count, 0}, FjPathView{valid.data, valid.count, valid.encoding == 1 ? 2u : 1u}, FjPathView{valid.data, std::numeric_limits<std::size_t>::max(), valid.encoding}}) {
            require_status(fj_legacy_assets_create(invalid, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(!out, "invalid root published Assets");
        }
        const fs::path nulPath(fs::path::string_type(3, 0));
        const JuicerAssets::NativePathArgument nul(nulPath);
        require_status(fj_legacy_assets_create(nul.view(), &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        FjErrorBuffer oversized{nullptr, std::numeric_limits<std::size_t>::max(), 99};
        require_status(fj_legacy_assets_create(valid, &out, &oversized), FJ_STATUS_UNSUPPORTED_INPUT);
        require(!out && oversized.length == 0, "malformed diagnostics did not precede operation");
#if defined(_WIN32)
        const std::array<std::uint16_t, 2> units{65, 66};
        require_status(fj_legacy_assets_create({reinterpret_cast<const char*>(units.data()) + 1, 1, FJ_PATH_WINDOWS_WIDE}, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
#endif
    }

    void sticky_source_and_retry(const fs::path& scratch) {
        const auto root = scratch / "retry";
        minimal_catalog(root);
        JuicerAssets::Library library(root);
        for (unsigned fault : {1u, 2u}) {
            conversionFault = fault;
            bool rejected = false;
            try {
                (void)library.spektrafilm_profile_catalog();
            } catch (const std::bad_alloc&) {
                rejected = fault == 1;
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = fault == 2 && error.failure.status.category == FJ_STATUS_PREPARATION_FAILURE;
            }
            require(rejected && fj_test_catalog_live_owners() == 0, "incomplete native candidate published or leaked");
        }
        fs::remove_all(root / "profiles");
        const auto& catalog = library.spektrafilm_profile_catalog();
        require(catalog.valid && catalog.filmProfiles.size() == 1 && catalog.printProfiles.size() == 1, "native retry rediscovered source");
        std::size_t nativeSize = 0;
        std::size_t nativeCapacity = 0;
        for (const auto* entries : {&catalog.filmProfiles, &catalog.printProfiles}) {
            nativeSize += entries->size() * sizeof(Spektrafilm::ProfileCatalogEntry);
            nativeCapacity += entries->capacity() * sizeof(Spektrafilm::ProfileCatalogEntry);
            for (const auto& entry : *entries) {
                nativeSize += entry.key.size() + entry.label.size();
                nativeCapacity += entry.key.capacity() + entry.label.capacity();
            }
        }
        std::printf("Catalog cold native copy logical=%zu bytes capacity=%zu bytes; one retained handle, two discarded cold candidates, warm publication reused\n", nativeSize, nativeCapacity);
        const auto* label = catalog.printProfiles.front().label.data();
        require(library.spektrafilm_profile_catalog().printProfiles.front().label.data() == label, "warm native catalog copied");

        const auto missing = scratch / "sticky-error";
        JuicerAssets::Library failed(missing);
        diagnosticFault = true;
        bool rejected = false;
        try {
            (void)failed.spektrafilm_profile_catalog();
        } catch (const std::bad_alloc&) {
            rejected = true;
        }
        require(rejected, "source diagnostic allocation published incomplete unavailable catalog");
        minimal_catalog(missing);
        require(!failed.spektrafilm_profile_catalog().valid && !failed.spektrafilm_profile_catalog().failure.empty(), "source error lost stickiness");
        const auto duplicate = scratch / "duplicate";
        minimal_catalog(duplicate);
        fs::copy_file(duplicate / "profiles/film.json", duplicate / "profiles/duplicate.json");
        JuicerAssets::Library duplicates(duplicate);
        require(!duplicates.spektrafilm_profile_catalog().valid, "duplicate key accepted");
        const auto absentDefault = scratch / "missing-default";
        minimal_catalog(absentDefault);
        fs::remove(absentDefault / "profiles/print.json");
        JuicerAssets::Library noDefault(absentDefault);
        require(!noDefault.spektrafilm_profile_catalog().valid, "missing default accepted");
    }

    void exact_paths_and_consumers(const fs::path& scratch) {
        const auto bundled = JuicerProcess::data_directory();
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
        JuicerCuda::Owner owner;
        owner.create(root);
        auto& library = JuicerProcess::root().assets();
        const auto& catalog = library.spektrafilm_profile_catalog();
        require(catalog.valid && catalog.filmProfiles.front().key == "kodak_portra_400" && catalog.printProfiles.front().key == "kodak_portra_endura", "catalog changed native file units");
        require(film_profile_option_count() == 1 && print_profile_option_count() == 1 &&
                    std::string(film_profile_option_key(0)) == "kodak_portra_400" && std::string(print_profile_option_key(0)) == "kodak_portra_endura" &&
                    std::string(film_profile_option_label(0)) == catalog.filmProfiles.front().label,
                "production OFX option accessors did not consume Rust publication");
        require(library.selected_film_profile_for_key("kodak_portra_400") != nullptr, "Rust selected film opener failed exact path");
        auto selected = library.selected_profiles_for_route({"kodak_portra_400", "kodak_portra_endura", Spektrafilm::ScanRoute::NegativePrintScan});
        require(selected.valid && selected.filmProfile && selected.printSource, "Rust selected print opener failed exact path");
        std::printf("PASS native root and Rust film/print opens: %s\n", JuicerAssets::path_diagnostic(root).c_str());
        require(fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr).category == FJ_STATUS_SUCCESS && fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 1, "borrowed shutdown consumed host/catalog");
        require(catalog.filmProfiles.front().key == "kodak_portra_400", "borrowed shutdown expired options");
        selected = {};
        require(owner.close().category == FJ_STATUS_SUCCESS && fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "terminal cleanup did not consume host owners");
    }

    void bundled_cold_and_warm_publication() {
        const auto root = JuicerProcess::data_directory();
        JuicerAssets::Library library(root);
        const auto& catalog = library.spektrafilm_profile_catalog();
        require(catalog.valid, "bundled production catalog unavailable");
        std::size_t logical = 0;
        std::size_t capacity = 0;
        for (const auto* entries : {&catalog.filmProfiles, &catalog.printProfiles}) {
            logical += entries->size() * sizeof(Spektrafilm::ProfileCatalogEntry);
            capacity += entries->capacity() * sizeof(Spektrafilm::ProfileCatalogEntry);
            for (const auto& entry : *entries) {
                logical += entry.key.size() + entry.label.size();
                capacity += entry.key.capacity() + entry.label.capacity();
            }
        }
        const auto* publication = &catalog;
        const auto* label = catalog.filmProfiles.front().label.data();
        for (unsigned i = 0; i < 32; ++i) {
            const auto& warm = library.spektrafilm_profile_catalog();
            require(&warm == publication && warm.filmProfiles.front().label.data() == label &&
                        fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 1,
                    "warm publication converted or accumulated owners");
        }
        std::printf("Catalog bundled cold native record-plus-field logical=%zu bytes capacity=%zu bytes; transient native candidate peak=one complete copy; cold conversions=1 warm calls=32 retained owners=1/1\n", logical, capacity);
    }
} // namespace

namespace JuicerAssets::CatalogTest {
    void before_conversion(FjCatalogCounts& counts) {
        switch (std::exchange(conversionFault, 0)) {
            case 1:
                throw std::bad_alloc();
            case 2:
                counts.film_count = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
                break;
            default:
                break;
        }
    }
    void before_source_diagnostic() {
        if (std::exchange(diagnosticFault, false)) {
            throw std::bad_alloc();
        }
    }
} // namespace JuicerAssets::CatalogTest

int main(int argc, char** argv) {
    try {
        require(argc == 2, "catalog fixture needs scratch path");
        const fs::path scratch(argv[1]);
        fs::remove_all(scratch);
        minimal_catalog(scratch / "minimal");
        root_shapes(scratch / "minimal");
        input_and_lifetime(scratch / "minimal");
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "raw ownership leaked");
        sticky_source_and_retry(scratch);
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "native publication leaked");
        exact_paths_and_consumers(scratch);
        bundled_cold_and_warm_publication();
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "bundled publication owners accumulated after drop");
        std::puts("PASS catalog ABI/production options, publication/retry, source stickiness, concurrent acquisition, retained borrows and native opens");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL catalog: %s\n", error.what());
        return 1;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        std::fprintf(stderr, "FAIL catalog boundary: %s\n", error.failure.diagnostic.c_str());
        return 1;
    }
}
