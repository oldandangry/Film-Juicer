#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ProfileCatalog.h"
#include "juicer_legacy_api.h"

namespace Gamut {
    class InputHullHold;
    class OutputBoundaryHold;
    struct OutputGamutTransform;
} // namespace Gamut

namespace JuicerAssets {
    enum class NeutralPrintCalibrationStatus : unsigned char {
        MissingFile,
        MissingEntry,
        Found,
        Malformed
    };

    struct NeutralPrintCalibrationResult {
        NeutralPrintCalibrationStatus status =
            NeutralPrintCalibrationStatus::MissingEntry;
        std::array<float, 3> cmyCc{};
        std::string diagnostic;
    };

#if defined(JUICER_ILLUMINANT_TEST_HOOK)
    namespace IlluminantTest {
        void calibration_result(FjNeutralCalibrationResult& result);
        void frame_abort() noexcept;
    } // namespace IlluminantTest
#endif

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

#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
    namespace ProfileTest {
        void before_catalog_publication();
    } // namespace ProfileTest
#endif

    // Native host forwarding for the Rust Assets and retained catalog owners.
    class AssetBridge final {
    public:
        explicit AssetBridge(const std::filesystem::path& resourceRoot);
        ~AssetBridge();
        AssetBridge(const AssetBridge&) = delete;
        AssetBridge& operator=(const AssetBridge&) = delete;

        std::shared_ptr<const Gamut::InputHullHold> input_hull();
        std::shared_ptr<const Gamut::OutputBoundaryHold> output_boundary(const Gamut::OutputGamutTransform&, std::string& diagnostic);
        NeutralPrintCalibrationResult neutral_print_calibration(const std::string& printStock,
                                                                const std::string& illuminant,
                                                                const std::string& filmStock);
        Spektrafilm::ProfileCatalog load_catalog();
        FjStatus film_defaults(const std::string& key, FjFilmDigest& out, FjErrorBuffer& error);
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;
        // SF_TEMP_BRIDGE_PREPARED_ASSET_BORROW: synchronous existing-owner borrow;
        // remove with the remaining process-owner forwarding at S5.C.
        FjAssets* borrow_assets() const noexcept;

    private:
        FjAssets* _assets = nullptr;
        // Library publishes once; terminal close excludes host readers.
        FjCatalog* _catalog = nullptr;
    };

} // namespace JuicerAssets
