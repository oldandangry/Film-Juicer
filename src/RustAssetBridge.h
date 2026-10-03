#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "ProfileCatalog.h"
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

    // FJ_TEMP_BRIDGE: asset conversion; remove S4.E.
    // All retained foreign owners nest inside the detachable native Library.
    class AssetBridge final {
    public:
        explicit AssetBridge(const std::filesystem::path& resourceRoot);
        ~AssetBridge();
        AssetBridge(const AssetBridge&) = delete;
        AssetBridge& operator=(const AssetBridge&) = delete;

        Spektrafilm::ProfileCatalog load_catalog();
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;

    private:
        FjAssets* _assets = nullptr;
        FjCatalog* _catalog = nullptr;
    };

} // namespace JuicerAssets
