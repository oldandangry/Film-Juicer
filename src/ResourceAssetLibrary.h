#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace JuicerAssets {

    struct FilmStockAsset {
        std::string optionLabel;
        std::string jsonKey;
        std::string profileJsonPath;
        std::uint64_t version = 0;
    };

    struct PrintPaperAsset {
        std::string optionLabel;
        std::string folderName;
        std::string jsonKey;
        std::string profileJsonPath;
        std::string paperDir;
        std::uint64_t version = 0;
    };

    struct NeutralFilterDatabaseAsset {
        std::string selectedPath;
        std::string defaultPath;
        std::uint64_t version = 0;
    };

    struct StaticNoiseAssetSet {
        std::string stbnPath;
        std::string wangTilesPath;
        std::string wangMetadataPath;
        std::uint64_t version = 0;
    };

    struct DichroicFilterAssetSet {
        std::string directory;
        std::uint64_t version = 0;
    };

    struct IlluminantFilterAssetSet {
        std::string d65Path;
        std::string d55Path;
        std::string d50Path;
        std::string tungstenPath;
        std::string kinoton75PPath;
        std::string kg3Path;
        std::string lensTransmissionPath;
        std::uint64_t version = 0;
    };

    class Library {
    public:
        static constexpr std::uint64_t kProcessAssetVersion = 1ull;

        const FilmStockAsset& film_stock_for_index(int index);
        const PrintPaperAsset& print_paper_for_index(int index);
        const NeutralFilterDatabaseAsset& neutral_filter_database_for_dichroic_set(int dichroicSetChoice);
        const StaticNoiseAssetSet& static_noise_assets();
        const DichroicFilterAssetSet& dichroic_filter_set_for_choice(int dichroicSetChoice);
        const IlluminantFilterAssetSet& illuminant_filter_assets();

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

        std::once_flag _catalogOnce;
        std::once_flag _neutralFilterOnce;
        std::once_flag _staticNoiseOnce;
        std::once_flag _dichroicFilterOnce;
        std::once_flag _illuminantFilterOnce;
        std::vector<FilmStockAsset> _filmStocks;
        std::vector<PrintPaperAsset> _printPapers;
        std::array<NeutralFilterDatabaseAsset, 3> _neutralFilterDatabases{};
        StaticNoiseAssetSet _staticNoiseAssets;
        std::array<DichroicFilterAssetSet, 3> _dichroicFilterSets{};
        IlluminantFilterAssetSet _illuminantFilterAssets;
    };

} // namespace JuicerAssets
