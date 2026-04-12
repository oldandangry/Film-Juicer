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

    class Library {
    public:
        static constexpr std::uint64_t kProcessAssetVersion = 1ull;

        const FilmStockAsset& film_stock_for_index(int index);
        const PrintPaperAsset& print_paper_for_index(int index);
        const NeutralFilterDatabaseAsset& neutral_filter_database_for_dichroic_set(int dichroicSetChoice);
        const StaticNoiseAssetSet& static_noise_assets();

        int film_stock_count();
        int print_paper_count();

    private:
        void ensure_catalogs();
        void ensure_neutral_filter_databases();
        void ensure_static_noise_assets();
        void load_catalogs();
        void load_neutral_filter_databases();
        void load_static_noise_assets();

        std::once_flag _catalogOnce;
        std::once_flag _neutralFilterOnce;
        std::once_flag _staticNoiseOnce;
        std::vector<FilmStockAsset> _filmStocks;
        std::vector<PrintPaperAsset> _printPapers;
        std::array<NeutralFilterDatabaseAsset, 3> _neutralFilterDatabases{};
        StaticNoiseAssetSet _staticNoiseAssets;
    };

} // namespace JuicerAssets
