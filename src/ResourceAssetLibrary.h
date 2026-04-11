#pragma once

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

    class Library {
    public:
        static constexpr std::uint64_t kProcessAssetVersion = 1ull;

        const FilmStockAsset& film_stock_for_index(int index);
        const PrintPaperAsset& print_paper_for_index(int index);

        int film_stock_count();
        int print_paper_count();

    private:
        void ensure_catalogs();
        void load_catalogs();

        std::once_flag _catalogOnce;
        std::vector<FilmStockAsset> _filmStocks;
        std::vector<PrintPaperAsset> _printPapers;
    };

} // namespace JuicerAssets
