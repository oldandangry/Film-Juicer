#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "SpectralData.h"

namespace Profiles {
    struct AgxFilmProfile;
}

namespace JuicerAssets {

    struct FilmStockAsset {
        std::string optionLabel;
        std::string jsonKey;
        std::uint64_t version = 0;
    };

    struct PrintPaperAsset {
        std::string optionLabel;
        std::string folderName;
        std::string jsonKey;
        std::uint64_t version = 0;
    };

    struct PrintPaperFolderProfilePayload {
        std::vector<std::pair<float, float>> dyeC;
        std::vector<std::pair<float, float>> dyeM;
        std::vector<std::pair<float, float>> dyeY;
        std::vector<std::pair<float, float>> logSensR;
        std::vector<std::pair<float, float>> logSensG;
        std::vector<std::pair<float, float>> logSensB;
        std::vector<std::pair<float, float>> baseMin;
        std::vector<std::pair<float, float>> baseMid;
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
        std::tuple<float, float, float> ymc{};
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

    struct DichroicFilterCurveSet {
        Spectral::Curve filterY;
        Spectral::Curve filterM;
        Spectral::Curve filterC;
        std::uint64_t version = 0;
    };

    struct IlluminantFilterCurveSet {
        Spectral::Curve d65;
        Spectral::Curve d55;
        Spectral::Curve d50;
        Spectral::Curve tungsten;
        Spectral::Curve kinoton75P;
        Spectral::Curve tungstenKg3Lens;
        std::uint64_t version = 0;
    };

    struct PrintRuntimeAssetSet {
        FilmStockAsset filmStock;
        PrintPaperAsset printPaper;
        NeutralFilterDatabaseAsset neutralFilters;
    };

    class Library {
    public:
        struct StaticNoiseAssetSet;
        struct DichroicFilterAssetSet;
        struct IlluminantFilterAssetSet;

        static constexpr std::uint64_t kProcessAssetVersion = 1ull;

        explicit Library(std::string dataDir);
        ~Library();

        const FilmStockAsset& film_stock_for_index(int index);
        const PrintPaperAsset& print_paper_for_index(int index);
        std::shared_ptr<const PrintPaperFolderProfilePayload> print_paper_folder_profile_payload(
            const PrintPaperAsset& asset);
        const NeutralFilterDatabaseAsset& neutral_filter_database_for_dichroic_set(int dichroicSetChoice);
        NeutralFilterLookupResult lookup_neutral_filters(
            const NeutralFilterDatabaseAsset& database,
            const std::string& paperKey,
            const std::string& illuminantKey,
            const std::string& negativeKey,
            NeutralFilterLookupThread threadClass);
        std::shared_ptr<const StaticNoisePayloadSet> static_noise_payloads();
        const DichroicFilterCurveSet& dichroic_filter_curves_for_choice(int dichroicSetChoice);
        const IlluminantFilterCurveSet& illuminant_filter_curves();
        PrintRuntimeAssetSet print_runtime_assets_for_choices(int filmIndex, int printPaperIndex, int dichroicSetChoice);
        bool load_agx_film_profile(const FilmStockAsset& asset, Profiles::AgxFilmProfile& outProfile);
        bool load_agx_print_profile(const PrintPaperAsset& asset, Profiles::AgxFilmProfile& outProfile);
        void release_cached_payloads() noexcept;

        int film_stock_count();
        int print_paper_count();

    private:
        void ensure_catalogs();
        void ensure_neutral_filter_databases();
        void ensure_static_noise_assets();
        void ensure_dichroic_filter_sets();
        void ensure_illuminant_filter_assets();
        void load_catalogs();
        void load_neutral_filter_databases();
        void load_static_noise_assets();
        void load_dichroic_filter_sets();
        void load_illuminant_filter_assets();
        NeutralFilterLookupResult lookup_neutral_filter_path(
            const std::string& jsonPath,
            const std::string& paperKey,
            const std::string& illuminantKey,
            const std::string& negativeKey,
            NeutralFilterLookupThread threadClass);
        bool load_agx_profile_path(const std::string& jsonPath, Profiles::AgxFilmProfile& outProfile);

        struct NeutralFilterCacheState;
        struct NeutralFilterDatabasePathSet;
        struct PrintPaperFolderProfilePayloadCacheState;
        struct StaticNoisePayloadCacheState;
        struct DichroicFilterCurveCacheState;
        struct IlluminantFilterCurveCacheState;
        struct ProfileCacheState;

        std::once_flag _catalogOnce;
        std::once_flag _neutralFilterOnce;
        std::once_flag _staticNoiseOnce;
        std::once_flag _dichroicFilterOnce;
        std::once_flag _illuminantFilterOnce;
        std::string _dataDir;
        std::vector<FilmStockAsset> _filmStocks;
        std::vector<PrintPaperAsset> _printPapers;
        std::array<NeutralFilterDatabaseAsset, 3> _neutralFilterDatabases{};
        std::unique_ptr<NeutralFilterDatabasePathSet[]> _neutralFilterDatabasePaths;
        std::unique_ptr<StaticNoiseAssetSet> _staticNoiseAssets;
        std::unique_ptr<DichroicFilterAssetSet[]> _dichroicFilterSets;
        std::unique_ptr<IlluminantFilterAssetSet> _illuminantFilterAssets;
        std::unique_ptr<NeutralFilterCacheState> _neutralFilterCache;
        std::unique_ptr<PrintPaperFolderProfilePayloadCacheState> _printPaperFolderProfilePayloadCache;
        std::unique_ptr<StaticNoisePayloadCacheState> _staticNoisePayloadCache;
        std::unique_ptr<DichroicFilterCurveCacheState> _dichroicFilterCurveCache;
        std::unique_ptr<IlluminantFilterCurveCacheState> _illuminantFilterCurveCache;
        std::unique_ptr<ProfileCacheState> _profileCache;
    };

} // namespace JuicerAssets
