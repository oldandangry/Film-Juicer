#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "ProfileAssets.h"
#include "ProfileCatalog.h"
#include "ScanRoute.h"
#include "SpectralData.h"

namespace Profiles {
    struct SpektrafilmProfileJson;
} // namespace Profiles

namespace JuicerAssets {

    struct SelectedFilmProfileAsset {
        std::string optionLabel;
        std::string jsonKey;
        std::uint64_t version = 0;
    };

    struct SelectedPrintProfileAsset {
        std::string optionLabel;
        std::string jsonKey;
        std::uint64_t version = 0;
    };

    struct NeutralFilterDatabaseAsset {
        std::uint32_t databaseId = 0;
        std::uint64_t version = 0;
    };

    enum class NeutralFilterLookupThread : unsigned char {
        Control = 0,
        RenderWorker = 1
    };

    struct NeutralFilterLookupResult {
        bool found = false;
        std::tuple<float, float, float> cmyCc;
        std::string selectedDbVersionHash;
    };

    struct StbnNoisePayload {
        std::vector<std::uint8_t> data;
        int width = 512;
        int height = 512;
        int frames = 256;
        bool valid = false;
        std::string error;
        std::uint64_t version = 0;
    };

    struct WangNoisePayload {
        std::vector<std::uint8_t> tiles;
        std::vector<std::uint8_t> lut;
        int width = 0;
        int height = 0;
        int count = 0;
        int colors = 0;
        bool valid = false;
        std::string error;
        std::uint64_t version = 0;
    };

    struct StaticNoisePayloadSet {
        StbnNoisePayload stbn;
        WangNoisePayload wang;
        std::uint64_t version = 0;
    };

    struct IlluminantFilterCurveSet {
        Spectral::Curve d65;
        Spectral::Curve d55;
        Spectral::Curve d50;
        Spectral::Curve tungsten;
        Spectral::Curve kinoton75P;
        Spectral::Curve tungstenKg3;
        Spectral::Curve tungstenKg3Lens;
        std::uint64_t version = 0;
    };

    struct PrintRuntimeAssetSet {
        SelectedFilmProfileAsset filmStock;
        SelectedPrintProfileAsset printPaper;
        NeutralFilterDatabaseAsset neutralFilters;
    };

    struct NeutralFilterLookupKey {
        std::string printProfileKey;
        std::string illuminantKey;
        std::string filmProfileKey;
    };

    struct PrintRuntimeProfileKeyChoices {
        std::string filmProfileKey;
        std::string printProfileKey;
        int dichroicSetChoice = 0;
    };

    struct MeasuredDichroicResourceIdentity {
        std::string setKey;
        std::array<std::string, 3> resourcePathsCmy;
        std::array<std::uint64_t, 3> resourceHashesCmy{};
        std::uint64_t hash = 0;
        bool valid = false;
        std::string diagnostic;
    };

    struct MeasuredDichroicCurveResult {
        std::array<std::array<float, 81>, 3> transmittanceCmy{};
        std::array<std::uint64_t, 3> resourceHashesCmy{};
        std::uint64_t hash = 0;
        bool valid = false;
        std::string diagnostic;
    };

    enum class NeutralPrintCalibrationStatus : unsigned char {
        MissingFile,
        MissingEntry,
        Found,
        Malformed
    };

    struct NeutralPrintCalibrationResult {
        NeutralPrintCalibrationStatus status = NeutralPrintCalibrationStatus::MissingEntry;
        std::array<float, 3> cmyCc{};
        std::string diagnostic;
    };

    using SelectedProfileRequest = Profiles::SelectedProfileRequest;
    using SelectedProfileResult = Profiles::SelectedProfileResult;

    class Library {
    public:
        struct StaticNoiseAssetSet;
        struct IlluminantFilterAssetSet;

        // Identity for pinned process-owned spectral assets. Bump when their source set or
        // derivation changes; focused descriptors enforce this token before consuming curves.
        static constexpr std::uint64_t kProcessAssetVersion = 1ull;

        explicit Library(std::string dataDir);
        ~Library();

        const Spektrafilm::ProfileCatalog& spektrafilm_profile_catalog();
        const JuicerAssets::SelectedFilmProfileAsset& film_profile_for_key(const std::string& key);
        const JuicerAssets::SelectedPrintProfileAsset& print_profile_for_key(const std::string& key);
        std::shared_ptr<const Profiles::ValidatedFilmProfile> selected_film_profile_for_key(
            const std::string& key);
        std::shared_ptr<const Profiles::ValidatedPrintProfile> selected_print_profile_for_key(
            const std::string& key);
        SelectedProfileResult selected_profiles_for_route(const SelectedProfileRequest& request);
        const NeutralFilterDatabaseAsset& neutral_filter_database_for_dichroic_set(int dichroicSetChoice);
        NeutralFilterLookupResult lookup_neutral_filters(
            const NeutralFilterDatabaseAsset& database,
            const NeutralFilterLookupKey& lookupKey,
            NeutralFilterLookupThread threadClass);
        std::shared_ptr<const StaticNoisePayloadSet> static_noise_payloads();
        const IlluminantFilterCurveSet& illuminant_filter_curves();
        MeasuredDichroicResourceIdentity measured_dichroic_resource_identity(const std::string& setKey);
        MeasuredDichroicCurveResult measured_dichroic_curves(const std::string& setKey);
        NeutralPrintCalibrationResult neutral_print_calibration(
            const std::string& printProfileKey,
            const std::string& printIlluminantKey,
            const std::string& filmProfileKey);
        PrintRuntimeAssetSet print_runtime_assets_for_profile_keys(const PrintRuntimeProfileKeyChoices& choices);
        bool load_spektrafilm_film_profile(const SelectedFilmProfileAsset& asset, Profiles::SpektrafilmProfileJson& outProfile);
        bool load_spektrafilm_print_profile(const SelectedPrintProfileAsset& asset, Profiles::SpektrafilmProfileJson& outProfile);
        void release_cached_payloads() noexcept;

        int film_stock_count();
        int print_paper_count();

    private:
        void ensure_catalogs();
        void ensure_neutral_filter_databases();
        void ensure_static_noise_assets();
        void ensure_illuminant_filter_assets();
        void load_catalogs();
        void load_neutral_filter_databases();
        void load_static_noise_assets();
        void load_illuminant_filter_assets();
        struct NeutralFilterPathLookup {
            const std::string& jsonPath;
            const NeutralFilterLookupKey& lookupKey;
            NeutralFilterLookupThread threadClass;
        };

        NeutralFilterLookupResult lookup_neutral_filter_path(const NeutralFilterPathLookup& lookup);
        bool load_spektrafilm_profile_path(const std::string& jsonPath, Profiles::SpektrafilmProfileJson& outProfile);

        struct NeutralFilterCacheState;
        struct NeutralFilterDatabasePathSet;
        struct StaticNoisePayloadCacheState;
        struct IlluminantFilterCurveCacheState;
        struct ProfileCacheState;

        std::once_flag _catalogOnce;
        std::once_flag _neutralFilterOnce;
        std::once_flag _staticNoiseOnce;
        std::once_flag _illuminantFilterOnce;
        std::string _dataDir;
        std::vector<SelectedFilmProfileAsset> _filmStocks;
        std::vector<SelectedPrintProfileAsset> _printPapers;
        Spektrafilm::ProfileCatalog _spektrafilmProfileCatalog;
        std::array<NeutralFilterDatabaseAsset, 3> _neutralFilterDatabases{};
        std::unique_ptr<NeutralFilterDatabasePathSet[]> _neutralFilterDatabasePaths;
        std::unique_ptr<StaticNoiseAssetSet> _staticNoiseAssets;
        std::unique_ptr<IlluminantFilterAssetSet> _illuminantFilterAssets;
        std::unique_ptr<NeutralFilterCacheState> _neutralFilterCache;
        std::unique_ptr<StaticNoisePayloadCacheState> _staticNoisePayloadCache;
        std::unique_ptr<IlluminantFilterCurveCacheState> _illuminantFilterCurveCache;
        std::unique_ptr<ProfileCacheState> _profileCache;
        std::unique_ptr<Profiles::ProfileAssetStore> _selectedProfileAssets;
    };

} // namespace JuicerAssets
