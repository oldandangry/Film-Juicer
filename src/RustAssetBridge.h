#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ProfileCatalog.h"
#include "ProfileAssets.h"
#include "juicer_legacy_api.h"

namespace JuicerAssets {

    // Synchronous argument: native path bytes borrow the path; Windows units
    // are copied by value, never aliased between wchar_t and uint16_t.
    class NativePathArgument final {
    public:
        explicit NativePathArgument(const std::filesystem::path& path);
        FjPathView view() const noexcept;

    private:
#if defined(_WIN32)
        std::vector<std::uint16_t> _units;
#else
        const std::string& _units;
#endif
    };

    std::filesystem::path copy_native_path(FjPathView view);
    std::string path_diagnostic(const std::filesystem::path& path);

#if defined(JUICER_CATALOG_TEST_HOOK)
    namespace CatalogTest {
        void before_conversion(FjCatalogCounts& counts);
        void before_source_diagnostic();
    } // namespace CatalogTest
#endif

    class PrintProfileSource final {
    public:
        ~PrintProfileSource();
        PrintProfileSource(const PrintProfileSource&) = delete;
        PrintProfileSource& operator=(const PrintProfileSource&) = delete;
        const std::shared_ptr<const Profiles::PrintProfile>& profile() const noexcept;
        Profiles::PrintDensityCurves sample_density_curves(double gamma) const;

    private:
        friend class AssetBridge;
        FjStatus close(FjErrorBuffer* error = nullptr) const noexcept;
        PrintProfileSource(FjPrintProfile* owner, std::shared_ptr<const Profiles::PrintProfile> profile);
        mutable FjPrintProfile* _owner = nullptr;
        std::shared_ptr<const Profiles::PrintProfile> _profile;
    };

#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
    namespace ProfileTest {
        void film_view(FjFilmProfileView& view);
        void print_view(FjPrintProfileView& view);
        void after_tables();
        void before_publication();
        void before_catalog_publication();
    } // namespace ProfileTest
#endif

    // FJ_TEMP_BRIDGE: asset conversion; remove S4.E.
    // All retained foreign owners nest inside the detachable native Library.
    class AssetBridge final {
    public:
        explicit AssetBridge(const std::filesystem::path& resourceRoot);
        ~AssetBridge();
        AssetBridge(const AssetBridge&) = delete;
        AssetBridge& operator=(const AssetBridge&) = delete;

        Spektrafilm::ProfileCatalog load_catalog();
        std::shared_ptr<const Profiles::FilmProfile> film(const std::string& key);
        std::shared_ptr<const PrintProfileSource> print(const std::string& key);
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;

    private:
        struct ConversionSlots;
        // Catalog readers use Library's call_once; cache release uses this lock.
        std::mutex _slotsMutex;
        std::unique_ptr<ConversionSlots> _slots;
        FjAssets* _assets = nullptr;
        FjCatalog* _catalog = nullptr;
    };

} // namespace JuicerAssets
