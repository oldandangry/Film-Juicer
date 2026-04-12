#include "ResourceAssetLibrary.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <system_error>
#include <tuple>
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
        using Clock = std::chrono::steady_clock;
        using Json = nlohmann::json;

        constexpr std::int64_t kNeutralFilterDiagnosticsReloadCheckMs = 1000;
        constexpr std::uint64_t kFnvOffsetBasis64 = 1469598103934665603ull;
        constexpr std::uint64_t kFnvPrime64 = 1099511628211ull;

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

        struct NeutralFilterFileStamp {
            bool valid = false;
            std::uint64_t sizeBytes = 0;
            std::int64_t writeTimeTicks = 0;
        };

        struct ParsedNeutralFilterDb {
            std::unordered_map<std::string, std::tuple<float, float, float>> lookup;
            NeutralFilterFileStamp stamp;
            std::string versionHash;
        };

        struct NeutralFilterCacheEntry {
            std::shared_ptr<const ParsedNeutralFilterDb> db;
            Clock::time_point lastDiagnosticsReloadCheck{};
            bool hasDiagnosticsReloadCheck = false;
        };

        struct FilterDbRead {
            std::shared_ptr<const ParsedNeutralFilterDb> db;
            bool stop = false;
        };

        std::string to_lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return s;
        }

        const char* thread_class_name(NeutralFilterLookupThread threadClass) {
            switch (threadClass) {
                case NeutralFilterLookupThread::Control:
                    return "control";
                case NeutralFilterLookupThread::RenderWorker:
                    return "render_worker";
                default:
                    return "unknown";
            }
        }

        std::uint64_t fnv1a_append(std::uint64_t hash, const void* data, size_t sizeBytes) {
            const unsigned char* bytes = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < sizeBytes; ++i) {
                hash ^= static_cast<std::uint64_t>(bytes[i]);
                hash *= kFnvPrime64;
            }
            return hash;
        }

        std::string hash_to_hex(std::uint64_t hash) {
            std::ostringstream oss;
            oss << std::hex << hash;
            return oss.str();
        }

        std::string make_lookup_key(const std::string& paperKey, const std::string& illuminantKey, const std::string& negativeKey) {
            std::string key;
            key.reserve(paperKey.size() + illuminantKey.size() + negativeKey.size() + 2);
            key += to_lower(paperKey);
            key.push_back('\x1f');
            key += to_lower(illuminantKey);
            key.push_back('\x1f');
            key += to_lower(negativeKey);
            return key;
        }

        std::string normalize_path_for_cache_key(const std::string& jsonPath) {
            fs::path path(jsonPath);
            path.make_preferred();
            return to_lower(path.lexically_normal().string());
        }

        bool same_file_stamp(const NeutralFilterFileStamp& a, const NeutralFilterFileStamp& b) {
            return a.valid && b.valid && a.sizeBytes == b.sizeBytes && a.writeTimeTicks == b.writeTimeTicks;
        }

        NeutralFilterFileStamp read_file_stamp(const std::string& jsonPath) {
            fs::path path(jsonPath);
            std::error_code ec;
            const auto sizeBytes = fs::file_size(path, ec);
            if (ec) {
                return {};
            }
            const auto writeTime = fs::last_write_time(path, ec);
            if (ec) {
                return {};
            }

            NeutralFilterFileStamp stamp;
            stamp.valid = true;
            stamp.sizeBytes = static_cast<std::uint64_t>(sizeBytes);
            stamp.writeTimeTicks = static_cast<std::int64_t>(writeTime.time_since_epoch().count());
            return stamp;
        }

        bool parse_array_triplet(const Json& arrNode, std::tuple<float, float, float>& outYMC) {
            if (!arrNode.is_array() || arrNode.size() < 3) {
                return false;
            }
            float vals[3] = {};
            for (size_t i = 0; i < 3; ++i) {
                const Json& element = arrNode[i];
                if (!(element.is_number_float() || element.is_number_integer())) {
                    return false;
                }
                float v = static_cast<float>(element.get<double>());
                if (!std::isfinite(v)) {
                    return false;
                }
                v = std::clamp(v, 0.0f, 1.0f);
                vals[i] = v;
            }

            outYMC = std::make_tuple(vals[0], vals[1], vals[2]);
            return true;
        }

        bool build_neutral_filter_lookup_table(
            const Json& root,
            std::unordered_map<std::string, std::tuple<float, float, float>>& outLookup) {
            if (!root.is_object()) {
                return false;
            }

            size_t validEntryCount = 0;
            for (auto paperIt = root.cbegin(); paperIt != root.cend(); ++paperIt) {
                if (!paperIt->is_object()) {
                    continue;
                }
                const std::string paperKey = paperIt.key();
                for (auto illuminantIt = paperIt->cbegin(); illuminantIt != paperIt->cend(); ++illuminantIt) {
                    if (!illuminantIt->is_object()) {
                        continue;
                    }
                    const std::string illuminantKey = illuminantIt.key();
                    for (auto negativeIt = illuminantIt->cbegin(); negativeIt != illuminantIt->cend(); ++negativeIt) {
                        std::tuple<float, float, float> ymc{};
                        if (!parse_array_triplet(*negativeIt, ymc)) {
                            continue;
                        }
                        outLookup[make_lookup_key(paperKey, illuminantKey, negativeIt.key())] = ymc;
                        ++validEntryCount;
                    }
                }
            }

            return validEntryCount > 0;
        }

        std::string compute_version_hash(const Json& root, const NeutralFilterFileStamp& stamp) {
            std::uint64_t hash = kFnvOffsetBasis64;
            const std::uint8_t validByte = stamp.valid ? 1u : 0u;
            hash = fnv1a_append(hash, &validByte, sizeof(validByte));
            hash = fnv1a_append(hash, &stamp.sizeBytes, sizeof(stamp.sizeBytes));
            hash = fnv1a_append(hash, &stamp.writeTimeTicks, sizeof(stamp.writeTimeTicks));
            const std::string jsonBlob = root.dump();
            hash = fnv1a_append(hash, jsonBlob.data(), jsonBlob.size());
            return hash_to_hex(hash);
        }

        bool parse_neutral_filter_db_from_disk(
            const std::string& jsonPath,
            ParsedNeutralFilterDb& outDb,
            std::string& outReason) {
            std::ifstream file(jsonPath, std::ios::binary);
            if (!file.is_open()) {
                outReason = "open_failed";
                return false;
            }

            Json root = Json::parse(file, nullptr, false);
            if (root.is_discarded()) {
                outReason = "parse_failed";
                return false;
            }

            std::unordered_map<std::string, std::tuple<float, float, float>> lookup;
            if (!build_neutral_filter_lookup_table(root, lookup)) {
                outReason = "schema_or_entries_invalid";
                return false;
            }

            outDb.lookup = std::move(lookup);
            outDb.stamp = read_file_stamp(jsonPath);
            outDb.versionHash = compute_version_hash(root, outDb.stamp);
            return true;
        }

        bool diagnostics_reload_enabled() {
            return JTRACE_ENABLED(3);
        }

        void trace_neutral_filter_event(
            const char* operation,
            const std::string& selectedDbVersionHash,
            NeutralFilterLookupThread threadClass,
            const char* reason = nullptr,
            const std::string* path = nullptr) {
            if (!JTRACE_ENABLED(1)) {
                return;
            }
            std::ostringstream oss;
            oss << "operation=" << (operation ? operation : "unknown")
                << " selected_db_version_hash=" << (selectedDbVersionHash.empty() ? "none" : selectedDbVersionHash)
                << " check_interval_ms=" << kNeutralFilterDiagnosticsReloadCheckMs
                << " thread_class=" << thread_class_name(threadClass);
            if (reason && *reason) {
                oss << " reason=" << reason;
            }
            if (path && !path->empty()) {
                oss << " path=" << *path;
            }
            JTRACE("MSNFD", oss.str());
        }

        void refresh_filter_db(
            NeutralFilterCacheEntry& cacheEntry,
            std::shared_ptr<const ParsedNeutralFilterDb>& dbSnapshot,
            const std::string& jsonPath,
            NeutralFilterLookupThread threadClass,
            Clock::time_point now) {
            bool shouldCheckReload = false;
            if (!cacheEntry.hasDiagnosticsReloadCheck) {
                shouldCheckReload = true;
            } else {
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                         now - cacheEntry.lastDiagnosticsReloadCheck)
                                         .count();
                shouldCheckReload = elapsed >= kNeutralFilterDiagnosticsReloadCheckMs;
            }

            if (!shouldCheckReload) {
                trace_neutral_filter_event("reload_skip", dbSnapshot->versionHash, threadClass, "interval_not_elapsed", &jsonPath);
                return;
            }

            cacheEntry.lastDiagnosticsReloadCheck = now;
            cacheEntry.hasDiagnosticsReloadCheck = true;
            trace_neutral_filter_event("reload_check", dbSnapshot->versionHash, threadClass, nullptr, &jsonPath);

            const NeutralFilterFileStamp newStamp = read_file_stamp(jsonPath);
            if (!newStamp.valid) {
                trace_neutral_filter_event("reload_failed", dbSnapshot->versionHash, threadClass, "stamp_unavailable", &jsonPath);
                return;
            }
            if (!dbSnapshot->stamp.valid) {
                trace_neutral_filter_event("reload_failed", dbSnapshot->versionHash, threadClass, "cached_stamp_unavailable", &jsonPath);
                return;
            }
            if (same_file_stamp(newStamp, dbSnapshot->stamp)) {
                trace_neutral_filter_event("reload_skip", dbSnapshot->versionHash, threadClass, "file_stamp_unchanged", &jsonPath);
                return;
            }

            ParsedNeutralFilterDb parsedDb;
            std::string parseReason;
            if (!parse_neutral_filter_db_from_disk(jsonPath, parsedDb, parseReason)) {
                trace_neutral_filter_event("reload_failed", dbSnapshot->versionHash, threadClass, parseReason.c_str(), &jsonPath);
                return;
            }

            dbSnapshot = std::make_shared<ParsedNeutralFilterDb>(std::move(parsedDb));
            cacheEntry.db = dbSnapshot;
            trace_neutral_filter_event("reload_commit", dbSnapshot->versionHash, threadClass, nullptr, &jsonPath);
        }

        FilterDbRead get_filter_db(
            std::mutex& cacheMutex,
            std::unordered_map<std::string, NeutralFilterCacheEntry>& cacheEntries,
            const std::string& cacheKey,
            const std::string& jsonPath,
            NeutralFilterLookupThread threadClass,
            bool diagnosticsReload,
            Clock::time_point now) {
            FilterDbRead read;

            std::lock_guard<std::mutex> lock(cacheMutex);
            auto cacheIt = cacheEntries.find(cacheKey);
            NeutralFilterCacheEntry* cacheEntry = (cacheIt != cacheEntries.end())
                                                      ? &cacheIt->second
                                                      : nullptr;
            read.db = cacheEntry ? cacheEntry->db : nullptr;

            if (!read.db) {
                if (threadClass == NeutralFilterLookupThread::RenderWorker) {
                    trace_neutral_filter_event("miss", "none", threadClass, "cache_cold_render_worker", &jsonPath);
                    read.stop = true;
                    return read;
                }

                ParsedNeutralFilterDb parsedDb;
                std::string parseReason;
                if (!parse_neutral_filter_db_from_disk(jsonPath, parsedDb, parseReason)) {
                    trace_neutral_filter_event("reload_failed", "none", threadClass, parseReason.c_str(), &jsonPath);
                    read.stop = true;
                    return read;
                }

                read.db = std::make_shared<ParsedNeutralFilterDb>(std::move(parsedDb));
                cacheIt = cacheEntries.emplace(cacheKey, NeutralFilterCacheEntry{}).first;
                cacheEntry = &cacheIt->second;
                cacheEntry->db = read.db;
                if (diagnosticsReload) {
                    cacheEntry->lastDiagnosticsReloadCheck = now;
                    cacheEntry->hasDiagnosticsReloadCheck = true;
                }
                trace_neutral_filter_event("load", read.db->versionHash, threadClass, nullptr, &jsonPath);
            }

            if (diagnosticsReload && threadClass == NeutralFilterLookupThread::Control && read.db) {
                if (!cacheEntry) {
                    trace_neutral_filter_event("reload_failed", read.db->versionHash, threadClass, "cache_entry_missing", &jsonPath);
                    read.db.reset();
                    read.stop = true;
                    return read;
                }
                refresh_filter_db(*cacheEntry, read.db, jsonPath, threadClass, now);
            }

            return read;
        }

        NeutralFilterLookupResult lookup_filter_ymc(
            const ParsedNeutralFilterDb& db,
            const std::string& lookupKey,
            NeutralFilterLookupThread threadClass) {
            NeutralFilterLookupResult result;
            const auto it = db.lookup.find(lookupKey);
            if (it == db.lookup.end()) {
                trace_neutral_filter_event("miss", db.versionHash, threadClass, "entry_not_found");
                return result;
            }

            result.found = true;
            result.ymc = it->second;
            result.selectedDbVersionHash = db.versionHash;
            trace_neutral_filter_event("hit", db.versionHash, threadClass);
            return result;
        }

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

        std::string noise_asset_path(std::initializer_list<const char*> segments) {
            if (gDataDir.empty()) {
                return {};
            }
            return data_path_string(segments);
        }

        std::string data_directory_path(std::initializer_list<const char*> segments) {
            if (gDataDir.empty()) {
                return {};
            }
            std::string result = data_path_string(segments);
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

        StaticNoiseAssetSet make_static_noise_assets() {
            StaticNoiseAssetSet asset;
            asset.stbnPath = noise_asset_path({"Noise", "stbn_scalar_512x512x256_u8.bin"});
            asset.wangTilesPath = noise_asset_path({"Noise", "Wang", "wang_tiles_256x256x16_u8.bin"});
            asset.wangMetadataPath = noise_asset_path({"Noise", "Wang", "tiles.json"});
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        DichroicFilterAssetSet make_dichroic_filter_set(const char* folderName) {
            DichroicFilterAssetSet asset;
            asset.directory = data_directory_path({"filters", "dichroics", folderName});
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        IlluminantFilterAssetSet make_illuminant_filter_assets() {
            IlluminantFilterAssetSet asset;
            asset.d65Path = data_path_string({"illuminants", "D65.csv"});
            asset.d55Path = data_path_string({"illuminants", "D55.csv"});
            asset.d50Path = data_path_string({"illuminants", "D50.csv"});
            asset.tungstenPath = data_path_string({"illuminants", "T.csv"});
            asset.kinoton75PPath = data_path_string({"illuminants", "K75P.csv"});
            asset.kg3Path = data_path_string({"filters", "heat_absorbing", "schott", "KG3.csv"});
            asset.lensTransmissionPath = data_path_string({"filters", "lens_transmission", "canon", "canon_24_f28_is.csv"});
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

    struct Library::NeutralFilterCacheState {
        std::mutex mutex;
        std::unordered_map<std::string, NeutralFilterCacheEntry> entries;
    };

    Library::Library()
        : _neutralFilterCache(std::make_unique<NeutralFilterCacheState>()) {
    }

    Library::~Library() = default;

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

    void Library::ensure_static_noise_assets() {
        std::call_once(_staticNoiseOnce, [this]() {
            load_static_noise_assets();
        });
    }

    void Library::ensure_dichroic_filter_sets() {
        std::call_once(_dichroicFilterOnce, [this]() {
            load_dichroic_filter_sets();
        });
    }

    void Library::ensure_illuminant_filter_assets() {
        std::call_once(_illuminantFilterOnce, [this]() {
            load_illuminant_filter_assets();
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

    NeutralFilterLookupResult Library::lookup_neutral_filters(
        const std::string& jsonPath,
        const std::string& paperKey,
        const std::string& illuminantKey,
        const std::string& negativeKey,
        NeutralFilterLookupThread threadClass) {
        NeutralFilterLookupResult result;

        if (paperKey.empty() || illuminantKey.empty() || negativeKey.empty()) {
            trace_neutral_filter_event("miss", "none", threadClass, "missing_lookup_key", &jsonPath);
            return result;
        }

        const std::string lookupKey = make_lookup_key(paperKey, illuminantKey, negativeKey);
        const std::string cacheKey = normalize_path_for_cache_key(jsonPath);
        if (cacheKey.empty()) {
            trace_neutral_filter_event("miss", "none", threadClass, "empty_path", &jsonPath);
            return result;
        }

        const bool diagnosticsReload = diagnostics_reload_enabled();
        const Clock::time_point now = Clock::now();

        FilterDbRead dbRead = get_filter_db(
            _neutralFilterCache->mutex,
            _neutralFilterCache->entries,
            cacheKey,
            jsonPath,
            threadClass,
            diagnosticsReload,
            now);

        if (!dbRead.db) {
            if (!dbRead.stop) {
                trace_neutral_filter_event("miss", "none", threadClass, "cache_unavailable", &jsonPath);
            }
            return result;
        }

        return lookup_filter_ymc(*dbRead.db, lookupKey, threadClass);
    }

    void Library::load_static_noise_assets() {
        _staticNoiseAssets = make_static_noise_assets();
    }

    void Library::load_dichroic_filter_sets() {
        _dichroicFilterSets[0] = make_dichroic_filter_set("durst_digital_light");
        _dichroicFilterSets[1] = make_dichroic_filter_set("thorlabs");
        _dichroicFilterSets[2] = make_dichroic_filter_set("edmund_optics");
    }

    void Library::load_illuminant_filter_assets() {
        _illuminantFilterAssets = make_illuminant_filter_assets();
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

    const StaticNoiseAssetSet& Library::static_noise_assets() {
        ensure_static_noise_assets();
        return _staticNoiseAssets;
    }

    const DichroicFilterAssetSet& Library::dichroic_filter_set_for_choice(int dichroicSetChoice) {
        ensure_dichroic_filter_sets();
        if (dichroicSetChoice < 0 || dichroicSetChoice >= static_cast<int>(_dichroicFilterSets.size())) {
            dichroicSetChoice = 0;
        }
        return _dichroicFilterSets[static_cast<size_t>(dichroicSetChoice)];
    }

    const IlluminantFilterAssetSet& Library::illuminant_filter_assets() {
        ensure_illuminant_filter_assets();
        return _illuminantFilterAssets;
    }

    PrintRuntimeAssetSet Library::print_runtime_assets_for_choices(int filmIndex, int printPaperIndex, int dichroicSetChoice) {
        PrintRuntimeAssetSet assets;
        assets.filmStock = film_stock_for_index(filmIndex);
        assets.printPaper = print_paper_for_index(printPaperIndex);
        assets.neutralFilters = neutral_filter_database_for_dichroic_set(dichroicSetChoice);
        assets.dichroicFilters = dichroic_filter_set_for_choice(dichroicSetChoice);
        assets.illuminantFilters = illuminant_filter_assets();
        return assets;
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
