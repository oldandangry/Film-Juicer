#include "ResourceAssetLibrary.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Logging.h"
#include "ProfileJSONLoader.h"
#include "nlohmann/json.hpp"

extern const std::string gDataDir;

namespace JuicerAssets {

    namespace {
        namespace fs = std::filesystem;

        struct FilmStockSeed {
            const char* optionLabel;
            const char* jsonKey;
        };

        struct PrintPaperSeed {
            const char* optionLabel;
            const char* folderName;
            const char* jsonKey;
        };

        static const std::array<FilmStockSeed, 5> kDefaultFilmStocks{{{"Vision3 250D", "kodak_vision3_250d_uc"},
                                                                      {"Vision3 50D", "kodak_vision3_50d_uc"},
                                                                      {"Vision3 200T", "kodak_vision3_200t_uc"},
                                                                      {"Vision3 500T", "kodak_vision3_500t_uc"},
                                                                      {"Portra 400", "kodak_portra_400_auc"}}};

        static const std::array<PrintPaperSeed, 2> kDefaultPrintPapers{{{"2383", "kodak_2383", "kodak_2383_uc"},
                                                                        {"2393", "kodak_2393", "kodak_2393_uc"}}};

        struct FilterCatalog {
            std::vector<std::string> paperKeys;
            std::vector<std::string> filmKeys;
            std::unordered_set<std::string> paperKeySet;
            std::unordered_set<std::string> filmKeySet;
        };

        std::string data_path_string(std::initializer_list<const char*> segments) {
            fs::path path(gDataDir);
            for (const char* segment : segments) {
                if (segment && *segment) {
                    path /= segment;
                }
            }
            path.make_preferred();
            return path.string();
        }

        std::string profile_json_path_for_key(const std::string& jsonKey) {
            if (jsonKey.empty()) {
                return {};
            }
            std::string fileName = jsonKey;
            fileName += ".json";
            return data_path_string({"profiles", fileName.c_str()});
        }

        std::string profile_asset_path(const char* fileName) {
            return data_path_string({"profiles", fileName});
        }

        std::string paper_dir_for_folder(const std::string& folderName) {
            if (folderName.empty() || gDataDir.empty()) {
                return {};
            }
            fs::path dir = fs::path(gDataDir) / "paper" / folderName;
            dir.make_preferred();
            std::string result = dir.string();
#ifdef _WIN32
            const char separator = '\\';
#else
            const char separator = '/';
#endif
            if (!result.empty() && result.back() != separator) {
                result.push_back(separator);
            }
            return result;
        }

        FilmStockAsset make_film_stock(std::string optionLabel, std::string jsonKey) {
            FilmStockAsset asset;
            asset.optionLabel = std::move(optionLabel);
            asset.jsonKey = std::move(jsonKey);
            asset.profileJsonPath = profile_json_path_for_key(asset.jsonKey);
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        PrintPaperAsset make_print_paper(
            std::string optionLabel,
            std::string folderName,
            std::string jsonKey) {
            PrintPaperAsset asset;
            asset.optionLabel = std::move(optionLabel);
            asset.folderName = std::move(folderName);
            asset.jsonKey = std::move(jsonKey);
            asset.profileJsonPath = profile_json_path_for_key(asset.jsonKey);
            asset.paperDir = paper_dir_for_folder(asset.folderName);
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        NeutralFilterDatabaseAsset make_neutral_filter_database(const char* selectedFileName) {
            NeutralFilterDatabaseAsset asset;
            asset.selectedPath = profile_asset_path(selectedFileName);
            asset.defaultPath = profile_asset_path("enlarger_neutral_ymc_filters.json");
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        void append_default_film_stocks(std::vector<FilmStockAsset>& out) {
            out.clear();
            out.reserve(kDefaultFilmStocks.size());
            for (const FilmStockSeed& seed : kDefaultFilmStocks) {
                out.emplace_back(make_film_stock(seed.optionLabel, seed.jsonKey));
            }
        }

        void append_default_print_papers(std::vector<PrintPaperAsset>& out) {
            out.clear();
            out.reserve(kDefaultPrintPapers.size());
            for (const PrintPaperSeed& seed : kDefaultPrintPapers) {
                out.emplace_back(make_print_paper(seed.optionLabel, seed.folderName, seed.jsonKey));
            }
        }

        std::string sanitize_identifier(const std::string& value) {
            std::string out;
            out.reserve(value.size());
            const char* inData = value.data();
            const char* const inEnd = inData + value.size();
            for (; inData < inEnd; ++inData) {
                unsigned char uc = static_cast<unsigned char>(*inData);
                if (std::isalnum(uc)) {
                    out.push_back(static_cast<char>(std::tolower(uc)));
                }
            }
            return out;
        }

        bool equals_ignore_case(const std::string& a, const std::string& b) {
            if (a.size() != b.size()) {
                return false;
            }
            const char* aData = a.data();
            const char* bData = b.data();
            const char* const aEnd = aData + a.size();
            for (; aData < aEnd; ++aData, ++bData) {
                if (std::tolower(static_cast<unsigned char>(*aData)) !=
                    std::tolower(static_cast<unsigned char>(*bData))) {
                    return false;
                }
            }
            return true;
        }

        FilterCatalog load_filter_catalog(const fs::path& filterPath) {
            FilterCatalog catalog;
            std::error_code ec;
            if (!fs::exists(filterPath, ec) || fs::is_directory(filterPath, ec)) {
                return catalog;
            }

            std::ifstream file(filterPath, std::ios::binary);
            if (!file.is_open()) {
                return catalog;
            }

            nlohmann::json root = nlohmann::json::parse(file, nullptr, false);
            if (root.is_discarded() || !root.is_object()) {
                return catalog;
            }
            const size_t paperCount = root.size();
            catalog.paperKeys.reserve(paperCount);
            catalog.paperKeySet.reserve(paperCount);
            catalog.filmKeys.reserve(paperCount * 4);
            catalog.filmKeySet.reserve(paperCount * 4);

            for (auto it = root.begin(); it != root.end(); ++it) {
                if (!it.value().is_object()) {
                    continue;
                }
                const std::string paperKey = it.key();
                if (catalog.paperKeySet.insert(paperKey).second) {
                    catalog.paperKeys.emplace_back(paperKey);
                }
                for (auto illumIt = it.value().begin(); illumIt != it.value().end(); ++illumIt) {
                    if (!illumIt.value().is_object()) {
                        continue;
                    }
                    for (auto filmIt = illumIt.value().begin(); filmIt != illumIt.value().end(); ++filmIt) {
                        const std::string filmKey = filmIt.key();
                        if (catalog.filmKeySet.insert(filmKey).second) {
                            catalog.filmKeys.emplace_back(filmKey);
                        }
                    }
                }
            }
            return catalog;
        }

        struct PrintFolderInfo {
            std::string name;
            std::string sanitized;
            bool used = false;
        };

        std::string claim_print_folder(
            const Profiles::ProfileInfoSummary& info,
            const std::string& key,
            std::vector<PrintFolderInfo>& folders) {
            std::string keySan = sanitize_identifier(key);
            std::string nameSan = sanitize_identifier(info.name);
            size_t bestScore = 0;
            int bestIndex = -1;
            PrintFolderInfo* folderData = folders.data();
            const size_t folderCount = folders.size();
            PrintFolderInfo* folderIt = folderData;
            for (size_t i = 0; i < folderCount; ++i, ++folderIt) {
                if (folderIt->used) {
                    continue;
                }
                const std::string& folderSan = folderIt->sanitized;
                if (folderSan.empty()) {
                    continue;
                }
                size_t score = 0;
                bool match = false;
                if (!keySan.empty() && keySan.find(folderSan) != std::string::npos) {
                    match = true;
                    score = folderSan.size() * 4;
                }
                if (!match && !nameSan.empty() && nameSan.find(folderSan) != std::string::npos) {
                    match = true;
                    score = folderSan.size() * 3;
                }
                if (!match && !keySan.empty() && folderSan.find(keySan) != std::string::npos) {
                    match = true;
                    score = keySan.size() * 2;
                }
                if (!match && !nameSan.empty() && folderSan.find(nameSan) != std::string::npos) {
                    match = true;
                    score = nameSan.size();
                }
                if (match && score > bestScore) {
                    bestScore = score;
                    bestIndex = static_cast<int>(i);
                }
            }
            if (bestIndex >= 0) {
                folderData[bestIndex].used = true;
                return folderData[bestIndex].name;
            }
            return {};
        }

    } // namespace

    void Library::ensure_catalogs() {
        std::call_once(_catalogOnce, [this]() {
            load_catalogs();
        });
    }

    void Library::ensure_neutral_filter_databases() {
        std::call_once(_neutralFilterOnce, [this]() {
            load_neutral_filter_databases();
        });
    }

    void Library::load_catalogs() {
        const bool traceCatalog = JTRACE_ENABLED(1);
        _filmStocks.clear();
        _printPapers.clear();

        fs::path base = fs::path(gDataDir);
        fs::path profilesDir = base / "profiles";
        fs::path paperDir = base / "paper";

        std::unordered_map<std::string, Profiles::ProfileInfoSummary> infoByKey;
        std::vector<std::string> missingFilmKeys;
        std::vector<std::string> missingPaperKeys;
        std::error_code ec;
        if (!gDataDir.empty() && fs::exists(profilesDir, ec) && fs::is_directory(profilesDir, ec)) {
            for (fs::directory_iterator it(profilesDir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
                if (!it->is_regular_file(ec)) {
                    continue;
                }
                if (it->path().extension() != ".json") {
                    continue;
                }
                Profiles::ProfileInfoSummary info;
                const std::string jsonPath = it->path().string();
                if (Profiles::load_profile_info(jsonPath, info)) {
                    infoByKey[info.stock] = std::move(info);
                }
            }
        }
        _filmStocks.reserve(infoByKey.size());
        _printPapers.reserve(infoByKey.size());

        FilterCatalog filters = load_filter_catalog(profilesDir / "enlarger_neutral_ymc_filters.json");
        if (traceCatalog) {
            missingFilmKeys.reserve(filters.filmKeys.size());
            missingPaperKeys.reserve(filters.paperKeys.size());
        }

        auto pushFilm = [&](const std::string& key) {
            auto it = infoByKey.find(key);
            if (it == infoByKey.end()) {
                if (traceCatalog) {
                    missingFilmKeys.push_back(key + " (profile missing)");
                }
                return;
            }
            if (!equals_ignore_case(it->second.type, "negative")) {
                if (traceCatalog) {
                    std::string reason = key + " (type='" + it->second.type + "')";
                    missingFilmKeys.push_back(std::move(reason));
                }
                return;
            }
            std::string label = it->second.name.empty() ? it->second.stock : it->second.name;
            _filmStocks.emplace_back(make_film_stock(std::move(label), it->second.stock));
        };

        for (const std::string& key : filters.filmKeys) {
            pushFilm(key);
        }

        if (_filmStocks.empty()) {
            for (const auto& pair : infoByKey) {
                if (!equals_ignore_case(pair.second.type, "negative")) {
                    continue;
                }
                std::string label = pair.second.name.empty() ? pair.second.stock : pair.second.name;
                _filmStocks.emplace_back(make_film_stock(std::move(label), pair.second.stock));
            }
            std::sort(_filmStocks.begin(), _filmStocks.end(), [](const FilmStockAsset& a, const FilmStockAsset& b) {
                return a.optionLabel < b.optionLabel;
            });
        }

        if (_filmStocks.empty()) {
            if (traceCatalog) {
                if (!missingFilmKeys.empty()) {
                    std::ostringstream oss;
                    oss << "catalog default: film profiles unavailable for keys: ";
                    const size_t missingCount = missingFilmKeys.size();
                    const std::string* missingData = missingFilmKeys.data();
                    for (size_t i = 0; i < missingCount; ++i, ++missingData) {
                        if (i > 0) {
                            oss << ", ";
                        }
                        oss << *missingData;
                    }
                    JTRACE("CATALOG", oss.str());
                } else {
                    JTRACE("CATALOG", "catalog default: no film profiles discovered; using defaults");
                }
            }
            append_default_film_stocks(_filmStocks);
        }

        std::vector<PrintFolderInfo> folders;
        if (!gDataDir.empty() && fs::exists(paperDir, ec) && fs::is_directory(paperDir, ec)) {
            for (fs::directory_iterator it(paperDir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
                if (it->is_directory(ec)) {
                    std::string folder = it->path().filename().string();
                    if (!folder.empty()) {
                        folders.emplace_back(PrintFolderInfo{folder, sanitize_identifier(folder), false});
                    }
                }
            }
        }

        auto pushPaper = [&](const std::string& key) {
            auto it = infoByKey.find(key);
            if (it == infoByKey.end()) {
                if (traceCatalog) {
                    missingPaperKeys.push_back(key + " (profile missing)");
                }
                return;
            }
            const auto& info = it->second;
            if (!equals_ignore_case(info.type, "paper")) {
                if (traceCatalog) {
                    std::string reason = key + " (type='" + info.type + "')";
                    missingPaperKeys.push_back(std::move(reason));
                }
                return;
            }
            std::string folder = claim_print_folder(info, key, folders);
            std::string label = info.name.empty() ? key : info.name;
            _printPapers.emplace_back(make_print_paper(std::move(label), std::move(folder), key));
        };

        for (const std::string& key : filters.paperKeys) {
            pushPaper(key);
        }

        if (_printPapers.empty()) {
            for (auto& folderInfo : folders) {
                if (folderInfo.used || folderInfo.sanitized.empty()) {
                    continue;
                }
                std::string bestKey;
                size_t bestScore = 0;
                for (const auto& pair : infoByKey) {
                    if (!equals_ignore_case(pair.second.type, "paper")) {
                        continue;
                    }
                    std::string keySan = sanitize_identifier(pair.first);
                    std::string nameSan = sanitize_identifier(pair.second.name);
                    size_t score = 0;
                    bool match = false;
                    if (!keySan.empty() && keySan.find(folderInfo.sanitized) != std::string::npos) {
                        match = true;
                        score = folderInfo.sanitized.size() * 4;
                    }
                    if (!match && !nameSan.empty() && nameSan.find(folderInfo.sanitized) != std::string::npos) {
                        match = true;
                        score = folderInfo.sanitized.size() * 3;
                    }
                    if (!match && !keySan.empty() && folderInfo.sanitized.find(keySan) != std::string::npos) {
                        match = true;
                        score = keySan.size() * 2;
                    }
                    if (!match && !nameSan.empty() && folderInfo.sanitized.find(nameSan) != std::string::npos) {
                        match = true;
                        score = nameSan.size();
                    }
                    if (match && score > bestScore) {
                        bestScore = score;
                        bestKey = pair.first;
                    }
                }
                if (!bestKey.empty()) {
                    folderInfo.used = true;
                    std::string label = folderInfo.name;
                    _printPapers.emplace_back(make_print_paper(std::move(label), folderInfo.name, bestKey));
                }
            }
        }

        if (_printPapers.empty()) {
            if (traceCatalog) {
                if (!missingPaperKeys.empty()) {
                    std::ostringstream oss;
                    oss << "catalog default: print profiles unavailable for keys: ";
                    const size_t missingCount = missingPaperKeys.size();
                    const std::string* missingData = missingPaperKeys.data();
                    for (size_t i = 0; i < missingCount; ++i, ++missingData) {
                        if (i > 0) {
                            oss << ", ";
                        }
                        oss << *missingData;
                    }
                    JTRACE("CATALOG", oss.str());
                } else {
                    JTRACE("CATALOG", "catalog default: no print profiles discovered; using defaults");
                }
            }
            append_default_print_papers(_printPapers);
        }
    }

    void Library::load_neutral_filter_databases() {
        _neutralFilterDatabases[0] = make_neutral_filter_database("enlarger_neutral_ymc_filters.json");
        _neutralFilterDatabases[1] = make_neutral_filter_database("enlarger_neutral_ymc_filters_thorlabs.json");
        _neutralFilterDatabases[2] = make_neutral_filter_database("enlarger_neutral_ymc_filters_edmund.json");
    }

    const FilmStockAsset& Library::film_stock_for_index(int index) {
        ensure_catalogs();
        static const FilmStockAsset empty{};
        if (_filmStocks.empty()) {
            return empty;
        }
        if (index < 0 || index >= static_cast<int>(_filmStocks.size())) {
            index = 0;
        }
        return _filmStocks[static_cast<size_t>(index)];
    }

    const PrintPaperAsset& Library::print_paper_for_index(int index) {
        ensure_catalogs();
        static const PrintPaperAsset empty{};
        if (_printPapers.empty()) {
            return empty;
        }
        if (index < 0 || index >= static_cast<int>(_printPapers.size())) {
            index = 0;
        }
        return _printPapers[static_cast<size_t>(index)];
    }

    const NeutralFilterDatabaseAsset& Library::neutral_filter_database_for_dichroic_set(int dichroicSetChoice) {
        ensure_neutral_filter_databases();
        if (dichroicSetChoice < 0 || dichroicSetChoice >= static_cast<int>(_neutralFilterDatabases.size())) {
            dichroicSetChoice = 0;
        }
        return _neutralFilterDatabases[static_cast<size_t>(dichroicSetChoice)];
    }

    int Library::film_stock_count() {
        ensure_catalogs();
        return static_cast<int>(_filmStocks.size());
    }

    int Library::print_paper_count() {
        ensure_catalogs();
        return static_cast<int>(_printPapers.size());
    }

} // namespace JuicerAssets
