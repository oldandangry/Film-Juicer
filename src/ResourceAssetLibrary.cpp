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
#include "Illuminants.h"
#include "ProfileJSONLoader.h"
#include "nlohmann/json.hpp"

namespace JuicerAssets {

    struct Library::StaticNoiseAssetSet {
        std::string stbnPath;
        std::string wangTilesPath;
        std::string wangMetadataPath;
        std::uint64_t version = 0;
    };

    struct Library::DichroicFilterAssetSet {
        std::string directory;
        std::uint64_t version = 0;
    };

    struct Library::IlluminantFilterAssetSet {
        std::string d65Path;
        std::string d55Path;
        std::string d50Path;
        std::string tungstenPath;
        std::string kinoton75PPath;
        std::string kg3Path;
        std::string lensTransmissionPath;
        std::uint64_t version = 0;
    };

    struct Library::NeutralFilterDatabasePathSet {
        std::string selectedPath;
        std::string defaultPath;
        std::uint64_t version = 0;
    };

    namespace {
        namespace fs = std::filesystem;
        using Clock = std::chrono::steady_clock;
        using Json = nlohmann::json;

        constexpr std::int64_t kNeutralFilterDiagnosticsReloadCheckMs = 1000;
        constexpr std::uint64_t kFnvOffsetBasis64 = 1469598103934665603ull;
        constexpr std::uint64_t kFnvPrime64 = 1099511628211ull;
        constexpr int kNeutralFilterDatabaseCount = 3;
        constexpr int kDichroicFilterSetCount = 3;

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
            FilterCatalog() = default;
            FilterCatalog(const FilterCatalog&) = delete;
            FilterCatalog& operator=(const FilterCatalog&) = delete;
            FilterCatalog(FilterCatalog&&) = delete;
            FilterCatalog& operator=(FilterCatalog&&) = delete;

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

        struct ProfileFileStamp {
            bool valid = false;
            std::uint64_t sizeBytes = 0;
            std::int64_t writeTimeTicks = 0;
        };

        struct ParsedNeutralFilterDb {
            ParsedNeutralFilterDb() = default;
            ParsedNeutralFilterDb(const ParsedNeutralFilterDb&) = delete;
            ParsedNeutralFilterDb& operator=(const ParsedNeutralFilterDb&) = delete;
            ParsedNeutralFilterDb(ParsedNeutralFilterDb&&) = delete;
            ParsedNeutralFilterDb& operator=(ParsedNeutralFilterDb&&) = delete;

            std::unordered_map<std::string, std::tuple<float, float, float>> lookup;
            NeutralFilterFileStamp stamp;
            std::string versionHash;
        };

        struct NeutralFilterCacheEntry {
            std::shared_ptr<const ParsedNeutralFilterDb> db;
            Clock::time_point lastDiagnosticsReloadCheck;
            bool hasDiagnosticsReloadCheck = false;
        };

        struct DichroicFilterCurveCacheEntry {
            DichroicFilterCurveSet curves;
            bool ready = false;
        };

        struct IlluminantFilterCurveCacheEntry {
            IlluminantFilterCurveSet curves;
            bool ready = false;
        };

        struct PrintPaperFolderProfilePayloadCacheEntry {
            std::shared_ptr<const PrintPaperFolderProfilePayload> payload;
        };

        struct FilterDbRead {
            std::shared_ptr<const ParsedNeutralFilterDb> db;
            bool stop = false;
        };

        struct ProfileCacheEntry {
            std::string cacheKey;
            ProfileFileStamp stamp;
            Profiles::AgxFilmProfile profile;
        };

        constexpr std::size_t kProfileCacheCapacity = 2;

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

        bool same_file_stamp(const ProfileFileStamp& a, const ProfileFileStamp& b) {
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

        ProfileFileStamp read_profile_file_stamp(const std::string& jsonPath) {
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

            ProfileFileStamp stamp;
            stamp.valid = true;
            stamp.sizeBytes = static_cast<std::uint64_t>(sizeBytes);
            stamp.writeTimeTicks = static_cast<std::int64_t>(writeTime.time_since_epoch().count());
            return stamp;
        }

        bool get_profile(
            std::vector<ProfileCacheEntry>& cache,
            const std::string& cacheKey,
            const ProfileFileStamp& stamp,
            Profiles::AgxFilmProfile& outProfile) {
            if (cacheKey.empty() || !stamp.valid) {
                return false;
            }

            for (std::size_t i = 0; i < cache.size(); ++i) {
                ProfileCacheEntry& entry = cache[i];
                if (entry.cacheKey != cacheKey || !same_file_stamp(entry.stamp, stamp)) {
                    continue;
                }

                if (i != 0) {
                    std::swap(cache[0], cache[i]);
                }
                outProfile = cache[0].profile;
                return true;
            }

            return false;
        }

        void store_profile(
            std::vector<ProfileCacheEntry>& cache,
            std::string cacheKey,
            const ProfileFileStamp& stamp,
            const Profiles::AgxFilmProfile& profile) {
            if (cacheKey.empty() || !stamp.valid) {
                return;
            }

            for (std::size_t i = 0; i < cache.size(); ++i) {
                if (cache[i].cacheKey == cacheKey) {
                    cache.erase(cache.begin() + static_cast<std::ptrdiff_t>(i));
                    break;
                }
            }

            cache.insert(
                cache.begin(),
                ProfileCacheEntry{
                    std::move(cacheKey),
                    stamp,
                    profile});
            if (cache.size() > kProfileCacheCapacity) {
                cache.resize(kProfileCacheCapacity);
            }
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
                const std::string& paperKey = paperIt.key();
                for (auto illuminantIt = paperIt->cbegin(); illuminantIt != paperIt->cend(); ++illuminantIt) {
                    if (!illuminantIt->is_object()) {
                        continue;
                    }
                    const std::string& illuminantKey = illuminantIt.key();
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

            std::shared_ptr<ParsedNeutralFilterDb> parsedDb = std::make_shared<ParsedNeutralFilterDb>();
            std::string parseReason;
            if (!parse_neutral_filter_db_from_disk(jsonPath, *parsedDb, parseReason)) {
                trace_neutral_filter_event("reload_failed", dbSnapshot->versionHash, threadClass, parseReason.c_str(), &jsonPath);
                return;
            }

            dbSnapshot = std::move(parsedDb);
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

                std::shared_ptr<ParsedNeutralFilterDb> parsedDb = std::make_shared<ParsedNeutralFilterDb>();
                std::string parseReason;
                if (!parse_neutral_filter_db_from_disk(jsonPath, *parsedDb, parseReason)) {
                    trace_neutral_filter_event("reload_failed", "none", threadClass, parseReason.c_str(), &jsonPath);
                    read.stop = true;
                    return read;
                }

                read.db = std::move(parsedDb);
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

        std::string data_path_string(const std::string& dataDir, std::initializer_list<const char*> segments) {
            fs::path path(dataDir);
            for (const char* segment : segments) {
                if (segment && *segment) {
                    path /= segment;
                }
            }
            path.make_preferred();
            return path.string();
        }

        struct ProfileJsonPathRequest {
            const std::string& dataDir;
            const std::string& jsonKey;
        };

        std::string profile_json_path_for_key(const ProfileJsonPathRequest& request) {
            if (request.jsonKey.empty()) {
                return {};
            }
            std::string fileName = request.jsonKey;
            fileName += ".json";
            return data_path_string(request.dataDir, {"profiles", fileName.c_str()});
        }

        std::string profile_asset_path(const std::string& dataDir, const char* fileName) {
            return data_path_string(dataDir, {"profiles", fileName});
        }

        std::string noise_asset_path(const std::string& dataDir, std::initializer_list<const char*> segments) {
            if (dataDir.empty()) {
                return {};
            }
            return data_path_string(dataDir, segments);
        }

        std::string data_directory_path(const std::string& dataDir, std::initializer_list<const char*> segments) {
            if (dataDir.empty()) {
                return {};
            }
            std::string result = data_path_string(dataDir, segments);
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

        std::string paper_dir_for_folder(const std::string& dataDir, const std::string& folderName) {
            if (folderName.empty() || dataDir.empty()) {
                return {};
            }
            fs::path dir = fs::path(dataDir) / "paper" / folderName;
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

        struct FilmStockAssetRequest {
            std::string optionLabel;
            std::string jsonKey;
        };

        FilmStockAsset make_film_stock(FilmStockAssetRequest request) {
            FilmStockAsset asset;
            asset.optionLabel = std::move(request.optionLabel);
            asset.jsonKey = std::move(request.jsonKey);
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        PrintPaperAsset make_print_paper(
            std::string optionLabel,
            std::string jsonKey) {
            PrintPaperAsset asset;
            asset.optionLabel = std::move(optionLabel);
            asset.jsonKey = std::move(jsonKey);
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        void add_print_paper(
            std::vector<PrintPaperAsset>& assets,
            std::vector<std::string>& folderNames,
            std::string optionLabel,
            std::string folderName,
            std::string jsonKey) {
            assets.emplace_back(make_print_paper(std::move(optionLabel), std::move(jsonKey)));
            folderNames.emplace_back(std::move(folderName));
        }

        NeutralFilterDatabaseAsset make_neutral_filter_database(std::uint32_t databaseId) {
            NeutralFilterDatabaseAsset asset;
            asset.databaseId = databaseId;
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        Library::StaticNoiseAssetSet make_static_noise_assets(const std::string& dataDir) {
            Library::StaticNoiseAssetSet asset;
            asset.stbnPath = noise_asset_path(dataDir, {"Noise", "stbn_scalar_512x512x256_u8.bin"});
            asset.wangTilesPath = noise_asset_path(dataDir, {"Noise", "Wang", "wang_tiles_256x256x16_u8.bin"});
            asset.wangMetadataPath = noise_asset_path(dataDir, {"Noise", "Wang", "tiles.json"});
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        StbnNoisePayload load_stbn_noise_payload(const Library::StaticNoiseAssetSet& assets) {
            StbnNoisePayload payload;
            payload.width = 512;
            payload.height = 512;
            payload.frames = 256;
            payload.version = assets.version;

            if (assets.stbnPath.empty()) {
                payload.error = "STBN load failed: data directory missing";
                return payload;
            }

            fs::path path = fs::path(assets.stbnPath);
            path.make_preferred();

            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) {
                payload.error = "STBN load failed: cannot open logical noise asset";
                return payload;
            }

            const std::streamsize size = file.tellg();
            if (size <= 0) {
                payload.error = "STBN load failed: logical noise asset is empty";
                return payload;
            }

            const std::size_t expected = static_cast<std::size_t>(payload.width) *
                                         static_cast<std::size_t>(payload.height) *
                                         static_cast<std::size_t>(payload.frames);
            if (static_cast<std::size_t>(size) != expected) {
                payload.error = "STBN load failed: logical noise asset has unexpected size";
                return payload;
            }

            payload.data.resize(expected);
            file.seekg(0, std::ios::beg);
            if (!file.read(reinterpret_cast<char*>(payload.data.data()), size)) {
                payload.error = "STBN load failed: logical noise asset read error";
                payload.data.clear();
                return payload;
            }

            payload.valid = true;
            return payload;
        }

        struct WangTileEdges {
            int left = 0;
            int right = 0;
            int top = 0;
            int bottom = 0;
        };

        std::size_t wang_lut_index(const WangTileEdges& edges, int colors) {
            const std::size_t c = static_cast<std::size_t>(colors);
            return (((static_cast<std::size_t>(edges.left) * c + static_cast<std::size_t>(edges.right)) * c +
                     static_cast<std::size_t>(edges.top)) *
                        c +
                    static_cast<std::size_t>(edges.bottom));
        }

        WangNoisePayload load_wang_noise_payload(const Library::StaticNoiseAssetSet& assets) {
            WangNoisePayload payload;
            payload.version = assets.version;

            if (assets.wangTilesPath.empty() || assets.wangMetadataPath.empty()) {
                payload.error = "Wang tiles load failed: data directory missing";
                return payload;
            }

            fs::path binPath = fs::path(assets.wangTilesPath);
            fs::path jsonPath = fs::path(assets.wangMetadataPath);
            binPath.make_preferred();
            jsonPath.make_preferred();

            if (!fs::exists(binPath) || !fs::exists(jsonPath)) {
                payload.error = "Wang tiles load failed: logical noise asset set is incomplete";
                return payload;
            }

            std::ifstream jf(jsonPath);
            if (!jf) {
                payload.error = "Wang tiles load failed: cannot open logical metadata asset";
                return payload;
            }

            Json root;
            try {
                jf >> root;
            } catch (const std::exception& e) {
                payload.error = std::string("Wang tiles load failed: invalid JSON ") + e.what();
                return payload;
            }

            if (!root.contains("resolution") || !root.contains("tiles") || !root.contains("colors") ||
                !root.contains("mapping")) {
                payload.error = "Wang tiles load failed: tiles.json missing required fields";
                return payload;
            }

            const int width = root.value("resolution", 0);
            const int height = width;
            const int count = root.value("tiles", 0);
            const int colors = root.value("colors", 0);
            if (width <= 0 || height <= 0 || count <= 0 || colors <= 0) {
                payload.error = "Wang tiles load failed: invalid metadata in tiles.json";
                return payload;
            }

            const std::size_t lutSize = static_cast<std::size_t>(colors) *
                                        static_cast<std::size_t>(colors) *
                                        static_cast<std::size_t>(colors) *
                                        static_cast<std::size_t>(colors);
            std::vector<std::uint8_t> lut(lutSize, 0);

            const auto& mapping = root["mapping"];
            if (!mapping.is_array()) {
                payload.error = "Wang tiles load failed: mapping is not an array";
                return payload;
            }

            for (const auto& entry : mapping) {
                if (!entry.contains("index") || !entry.contains("labels")) {
                    continue;
                }
                const int idx = entry.value("index", 0);
                const auto& labels = entry["labels"];
                const int l = labels.value("L", 0);
                const int r = labels.value("R", 0);
                const int t = labels.value("T", 0);
                const int b = labels.value("B", 0);
                if (l < 0 || r < 0 || t < 0 || b < 0 ||
                    l >= colors || r >= colors || t >= colors || b >= colors) {
                    continue;
                }
                const std::size_t lutIndex = wang_lut_index(WangTileEdges{l, r, t, b}, colors);
                if (lutIndex < lut.size() && idx >= 0 && idx < count) {
                    lut[lutIndex] = static_cast<std::uint8_t>(idx);
                }
            }

            std::ifstream bin(binPath, std::ios::binary | std::ios::ate);
            if (!bin) {
                payload.error = "Wang tiles load failed: cannot open logical tile asset";
                return payload;
            }
            const std::streamsize size = bin.tellg();
            if (size <= 0) {
                payload.error = "Wang tiles load failed: logical tile asset is empty";
                return payload;
            }
            const std::size_t expected = static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(height) *
                                         static_cast<std::size_t>(count);
            if (static_cast<std::size_t>(size) != expected) {
                payload.error = "Wang tiles load failed: logical tile asset has unexpected size";
                return payload;
            }

            std::vector<std::uint8_t> tiles(expected);
            bin.seekg(0, std::ios::beg);
            if (!bin.read(reinterpret_cast<char*>(tiles.data()), size)) {
                payload.error = "Wang tiles load failed: logical tile asset read error";
                return payload;
            }

            payload.tiles = std::move(tiles);
            payload.lut = std::move(lut);
            payload.width = width;
            payload.height = height;
            payload.count = count;
            payload.colors = colors;
            payload.valid = true;
            return payload;
        }

        StaticNoisePayloadSet load_static_noise_payloads(const Library::StaticNoiseAssetSet& assets) {
            StaticNoisePayloadSet payloads;
            payloads.stbn = load_stbn_noise_payload(assets);
            payloads.wang = load_wang_noise_payload(assets);
            payloads.version = assets.version;
            return payloads;
        }

        Library::DichroicFilterAssetSet make_dichroic_filter_set(const std::string& dataDir, const char* folderName) {
            Library::DichroicFilterAssetSet asset;
            asset.directory = data_directory_path(dataDir, {"filters", "dichroics", folderName});
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        std::vector<std::pair<float, float>> load_pairs_silent(const std::string& path) {
            try {
                return Spectral::load_csv_pairs(path);
            } catch (...) {
                return {};
            }
        }

        std::string print_paper_file_path(const std::string& dataDir, const std::string& folderName, const char* fileName) {
            const std::string paperDir = paper_dir_for_folder(dataDir, folderName);
            if (paperDir.empty()) {
                return {};
            }
            fs::path path(paperDir);
            path /= fileName;
            path.make_preferred();
            return path.string();
        }

        std::vector<std::pair<float, float>> load_print_paper_pairs(
            const std::string& dataDir,
            const std::string& folderName,
            const char* fileName) {
            const std::string path = print_paper_file_path(dataDir, folderName, fileName);
            if (path.empty()) {
                return {};
            }
            return load_pairs_silent(path);
        }

        PrintPaperFolderProfilePayload load_print_paper_folder_profile_payload(
            const std::string& dataDir,
            const PrintPaperAsset& asset,
            const std::string& folderName) {
            PrintPaperFolderProfilePayload payload;
            payload.dyeC = load_print_paper_pairs(dataDir, folderName, "dye_density_c.csv");
            payload.dyeM = load_print_paper_pairs(dataDir, folderName, "dye_density_m.csv");
            payload.dyeY = load_print_paper_pairs(dataDir, folderName, "dye_density_y.csv");
            payload.logSensR = load_print_paper_pairs(dataDir, folderName, "log_sensitivity_r.csv");
            payload.logSensG = load_print_paper_pairs(dataDir, folderName, "log_sensitivity_g.csv");
            payload.logSensB = load_print_paper_pairs(dataDir, folderName, "log_sensitivity_b.csv");
            payload.baseMin = load_print_paper_pairs(dataDir, folderName, "dye_density_min.csv");
            payload.baseMid = load_print_paper_pairs(dataDir, folderName, "dye_density_mid.csv");
            payload.version = asset.version;
            return payload;
        }

        std::string print_paper_folder_profile_payload_key(const PrintPaperAsset& asset, const std::string& folderName) {
            std::ostringstream key;
            key << asset.version << '\n'
                << asset.jsonKey << '\n'
                << folderName;
            return key.str();
        }

        void prepare_identity_dichroic_curve(Spectral::Curve& curve) {
            Spectral::assign_reference_axis(curve.lambda_nm);
            curve.linear.assign(static_cast<size_t>(Spectral::gShape.K), 1.0f);
        }

        void apply_dichroic_channel(
            const std::vector<std::pair<float, float>>& pairs,
            Spectral::Curve& dst) {
            if (pairs.size() < 2) {
                return;
            }

            // Match agx-emulsion's Akima path: no extrapolation outside measured samples.
            const std::vector<std::pair<float, float>> resampled =
                Spectral::resample_pairs_akima_to_reference_axis(pairs);
            if (resampled.empty() || resampled.size() != static_cast<size_t>(Spectral::gShape.K)) {
                return;
            }

            for (size_t i = 0; i < resampled.size(); ++i) {
                dst.linear[i] = resampled[i].second * 0.01f;
            }
        }

        DichroicFilterCurveSet load_dichroic_filter_curves(const Library::DichroicFilterAssetSet& asset) {
            DichroicFilterCurveSet curves;
            curves.version = asset.version;
            prepare_identity_dichroic_curve(curves.filterY);
            prepare_identity_dichroic_curve(curves.filterM);
            prepare_identity_dichroic_curve(curves.filterC);

            if (asset.directory.empty()) {
                return curves;
            }

            const std::string yPath = asset.directory + "filter_y.csv";
            const std::string mPath = asset.directory + "filter_m.csv";
            const std::string cPath = asset.directory + "filter_c.csv";

            apply_dichroic_channel(load_pairs_silent(yPath), curves.filterY);
            apply_dichroic_channel(load_pairs_silent(mPath), curves.filterM);
            apply_dichroic_channel(load_pairs_silent(cPath), curves.filterC);
            return curves;
        }

        Library::IlluminantFilterAssetSet make_illuminant_filter_assets(const std::string& dataDir) {
            Library::IlluminantFilterAssetSet asset;
            asset.d65Path = data_path_string(dataDir, {"illuminants", "D65.csv"});
            asset.d55Path = data_path_string(dataDir, {"illuminants", "D55.csv"});
            asset.d50Path = data_path_string(dataDir, {"illuminants", "D50.csv"});
            asset.tungstenPath = data_path_string(dataDir, {"illuminants", "T.csv"});
            asset.kinoton75PPath = data_path_string(dataDir, {"illuminants", "K75P.csv"});
            asset.kg3Path = data_path_string(dataDir, {"filters", "heat_absorbing", "schott", "KG3.csv"});
            asset.lensTransmissionPath = data_path_string(
                dataDir,
                {"filters", "lens_transmission", "canon", "canon_24_f28_is.csv"});
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        IlluminantFilterCurveSet load_illuminant_filter_curves(const Library::IlluminantFilterAssetSet& asset) {
            IlluminantFilterCurveSet curves;
            curves.d65 = Spectral::build_curve_D65_pinned(asset.d65Path);
            curves.d55 = Spectral::build_curve_D55_pinned(asset.d55Path);
            curves.d50 = Spectral::build_curve_D50_pinned(asset.d50Path);
            curves.tungsten = Spectral::build_curve_T_pinned(asset.tungstenPath);
            curves.kinoton75P = Spectral::build_curve_K75P_pinned(asset.kinoton75PPath);
            curves.tungstenKg3Lens = Spectral::build_curve_TH_KG3_L_pinned(
                asset.kg3Path,
                asset.lensTransmissionPath);
            curves.version = asset.version;
            return curves;
        }

        bool curve_is_on_reference_axis(const Spectral::Curve& curve) {
            const size_t expected = static_cast<size_t>(Spectral::gShape.K);
            return curve.lambda_nm.size() == expected && curve.linear.size() == expected;
        }

        bool illuminant_filter_curves_complete(const IlluminantFilterCurveSet& curves) {
            return curve_is_on_reference_axis(curves.d65) &&
                   curve_is_on_reference_axis(curves.d55) &&
                   curve_is_on_reference_axis(curves.d50) &&
                   curve_is_on_reference_axis(curves.tungsten) &&
                   curve_is_on_reference_axis(curves.kinoton75P) &&
                   curve_is_on_reference_axis(curves.tungstenKg3Lens);
        }

        void append_default_film_stocks(std::vector<FilmStockAsset>& out) {
            out.clear();
            out.reserve(kDefaultFilmStocks.size());
            for (const FilmStockSeed& seed : kDefaultFilmStocks) {
                out.emplace_back(make_film_stock(FilmStockAssetRequest{seed.optionLabel, seed.jsonKey}));
            }
        }

        void append_default_print_papers(std::vector<PrintPaperAsset>& out, std::vector<std::string>& outFolderNames) {
            out.clear();
            outFolderNames.clear();
            out.reserve(kDefaultPrintPapers.size());
            outFolderNames.reserve(kDefaultPrintPapers.size());
            for (const PrintPaperSeed& seed : kDefaultPrintPapers) {
                add_print_paper(out, outFolderNames, seed.optionLabel, seed.folderName, seed.jsonKey);
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

        void clear_filter_catalog(FilterCatalog& catalog) {
            catalog.paperKeys.clear();
            catalog.filmKeys.clear();
            catalog.paperKeySet.clear();
            catalog.filmKeySet.clear();
        }

        bool load_filter_catalog(const fs::path& filterPath, FilterCatalog& catalog) {
            clear_filter_catalog(catalog);
            std::error_code ec;
            if (!fs::exists(filterPath, ec) || fs::is_directory(filterPath, ec)) {
                return false;
            }

            std::ifstream file(filterPath, std::ios::binary);
            if (!file.is_open()) {
                return false;
            }

            nlohmann::json root = nlohmann::json::parse(file, nullptr, false);
            if (root.is_discarded() || !root.is_object()) {
                return false;
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
                const std::string& paperKey = it.key();
                if (catalog.paperKeySet.insert(paperKey).second) {
                    catalog.paperKeys.emplace_back(paperKey);
                }
                for (auto illumIt = it.value().begin(); illumIt != it.value().end(); ++illumIt) {
                    if (!illumIt.value().is_object()) {
                        continue;
                    }
                    for (auto filmIt = illumIt.value().begin(); filmIt != illumIt.value().end(); ++filmIt) {
                        const std::string& filmKey = filmIt.key();
                        if (catalog.filmKeySet.insert(filmKey).second) {
                            catalog.filmKeys.emplace_back(filmKey);
                        }
                    }
                }
            }
            return true;
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

    struct Library::PrintPaperFolderProfilePayloadCacheState {
        std::mutex mutex;
        std::unordered_map<std::string, PrintPaperFolderProfilePayloadCacheEntry> entries;
    };

    struct Library::StaticNoisePayloadCacheState {
        std::mutex mutex;
        std::shared_ptr<const StaticNoisePayloadSet> payloads;
    };

    struct Library::DichroicFilterCurveCacheState {
        std::mutex mutex;
        std::array<DichroicFilterCurveCacheEntry, 3> entries{};
    };

    struct Library::IlluminantFilterCurveCacheState {
        std::mutex mutex;
        IlluminantFilterCurveCacheEntry entry;
    };

    struct Library::ProfileCacheState {
        std::mutex mutex;
        std::vector<ProfileCacheEntry> profiles;
    };

    Library::Library(std::string dataDir)
        : _dataDir(std::move(dataDir)),
          _neutralFilterDatabasePaths(std::make_unique<NeutralFilterDatabasePathSet[]>(kNeutralFilterDatabaseCount)),
          _staticNoiseAssets(std::make_unique<StaticNoiseAssetSet>()),
          _dichroicFilterSets(std::make_unique<DichroicFilterAssetSet[]>(kDichroicFilterSetCount)),
          _illuminantFilterAssets(std::make_unique<IlluminantFilterAssetSet>()),
          _neutralFilterCache(std::make_unique<NeutralFilterCacheState>()),
          _printPaperFolderProfilePayloadCache(std::make_unique<PrintPaperFolderProfilePayloadCacheState>()),
          _staticNoisePayloadCache(std::make_unique<StaticNoisePayloadCacheState>()),
          _dichroicFilterCurveCache(std::make_unique<DichroicFilterCurveCacheState>()),
          _illuminantFilterCurveCache(std::make_unique<IlluminantFilterCurveCacheState>()),
          _profileCache(std::make_unique<ProfileCacheState>()) {
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
        _printPaperFolderNames.clear();

        const std::string& dataDir = _dataDir;
        fs::path base = fs::path(dataDir);
        fs::path profilesDir = base / "profiles";
        fs::path paperDir = base / "paper";

        std::unordered_map<std::string, Profiles::ProfileInfoSummary> infoByKey;
        std::vector<std::string> missingFilmKeys;
        std::vector<std::string> missingPaperKeys;
        std::error_code ec;
        if (!dataDir.empty() && fs::exists(profilesDir, ec) && fs::is_directory(profilesDir, ec)) {
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

        FilterCatalog filters;
        (void)load_filter_catalog(profilesDir / "enlarger_neutral_ymc_filters.json", filters);
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
            _filmStocks.emplace_back(make_film_stock(FilmStockAssetRequest{std::move(label), it->second.stock}));
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
                _filmStocks.emplace_back(make_film_stock(FilmStockAssetRequest{std::move(label), pair.second.stock}));
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
        if (!dataDir.empty() && fs::exists(paperDir, ec) && fs::is_directory(paperDir, ec)) {
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
            add_print_paper(_printPapers, _printPaperFolderNames, std::move(label), std::move(folder), key);
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
                    add_print_paper(_printPapers, _printPaperFolderNames, std::move(label), folderInfo.name, bestKey);
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
            append_default_print_papers(_printPapers, _printPaperFolderNames);
        }
    }

    void Library::load_neutral_filter_databases() {
        auto makePaths = [this](const char* selectedFileName) {
            NeutralFilterDatabasePathSet paths;
            paths.selectedPath = profile_asset_path(_dataDir, selectedFileName);
            paths.defaultPath = profile_asset_path(_dataDir, "enlarger_neutral_ymc_filters.json");
            paths.version = Library::kProcessAssetVersion;
            return paths;
        };

        _neutralFilterDatabases[0] = make_neutral_filter_database(0);
        _neutralFilterDatabasePaths[0] = makePaths("enlarger_neutral_ymc_filters.json");
        _neutralFilterDatabases[1] = make_neutral_filter_database(1);
        _neutralFilterDatabasePaths[1] = makePaths("enlarger_neutral_ymc_filters_thorlabs.json");
        _neutralFilterDatabases[2] = make_neutral_filter_database(2);
        _neutralFilterDatabasePaths[2] = makePaths("enlarger_neutral_ymc_filters_edmund.json");
    }

    NeutralFilterLookupResult Library::lookup_neutral_filter_path(
        const NeutralFilterPathLookup& lookup) {
        NeutralFilterLookupResult result;
        const NeutralFilterLookupKey& key = lookup.lookupKey;

        if (key.paperKey.empty() || key.illuminantKey.empty() || key.negativeKey.empty()) {
            trace_neutral_filter_event("miss", "none", lookup.threadClass, "missing_lookup_key", &lookup.jsonPath);
            return result;
        }

        const std::string lookupKey = make_lookup_key(key.paperKey, key.illuminantKey, key.negativeKey);
        const std::string cacheKey = normalize_path_for_cache_key(lookup.jsonPath);
        if (cacheKey.empty()) {
            trace_neutral_filter_event("miss", "none", lookup.threadClass, "empty_path", &lookup.jsonPath);
            return result;
        }

        const bool diagnosticsReload = diagnostics_reload_enabled();
        const Clock::time_point now = Clock::now();

        FilterDbRead dbRead = get_filter_db(
            _neutralFilterCache->mutex,
            _neutralFilterCache->entries,
            cacheKey,
            lookup.jsonPath,
            lookup.threadClass,
            diagnosticsReload,
            now);

        if (!dbRead.db) {
            if (!dbRead.stop) {
                trace_neutral_filter_event("miss", "none", lookup.threadClass, "cache_unavailable", &lookup.jsonPath);
            }
            return result;
        }

        return lookup_filter_ymc(*dbRead.db, lookupKey, lookup.threadClass);
    }

    NeutralFilterLookupResult Library::lookup_neutral_filters(
        const NeutralFilterDatabaseAsset& database,
        const NeutralFilterLookupKey& lookupKey,
        NeutralFilterLookupThread threadClass) {
        ensure_neutral_filter_databases();
        const std::size_t databaseIndex = static_cast<std::size_t>(database.databaseId);
        if (!_neutralFilterDatabasePaths || database.version == 0 || databaseIndex >= _neutralFilterDatabases.size()) {
            return {};
        }

        const NeutralFilterDatabasePathSet& paths = _neutralFilterDatabasePaths[databaseIndex];
        if (paths.version != database.version) {
            return {};
        }

        NeutralFilterLookupResult result = lookup_neutral_filter_path(
            NeutralFilterPathLookup{paths.selectedPath, lookupKey, threadClass});
        if (result.found || paths.selectedPath == paths.defaultPath) {
            return result;
        }
        return lookup_neutral_filter_path(
            NeutralFilterPathLookup{paths.defaultPath, lookupKey, threadClass});
    }

    void Library::load_static_noise_assets() {
        *_staticNoiseAssets = make_static_noise_assets(_dataDir);
    }

    void Library::load_dichroic_filter_sets() {
        _dichroicFilterSets[0] = make_dichroic_filter_set(_dataDir, "durst_digital_light");
        _dichroicFilterSets[1] = make_dichroic_filter_set(_dataDir, "thorlabs");
        _dichroicFilterSets[2] = make_dichroic_filter_set(_dataDir, "edmund_optics");
    }

    void Library::load_illuminant_filter_assets() {
        *_illuminantFilterAssets = make_illuminant_filter_assets(_dataDir);
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

    std::string Library::print_paper_folder_name_for_asset(const PrintPaperAsset& asset) {
        ensure_catalogs();
        const size_t count = std::min(_printPapers.size(), _printPaperFolderNames.size());
        for (size_t i = 0; i < count; ++i) {
            const PrintPaperAsset& candidate = _printPapers[i];
            if (candidate.version == asset.version && candidate.jsonKey == asset.jsonKey) {
                return _printPaperFolderNames[i];
            }
        }
        return {};
    }

    std::shared_ptr<const PrintPaperFolderProfilePayload> Library::print_paper_folder_profile_payload(
        const PrintPaperAsset& asset) {
        const std::string folderName = print_paper_folder_name_for_asset(asset);
        const std::string cacheKey = print_paper_folder_profile_payload_key(asset, folderName);
        std::lock_guard<std::mutex> lock(_printPaperFolderProfilePayloadCache->mutex);
        PrintPaperFolderProfilePayloadCacheEntry& entry =
            _printPaperFolderProfilePayloadCache->entries[cacheKey];
        if (!entry.payload) {
            entry.payload =
                std::make_shared<PrintPaperFolderProfilePayload>(load_print_paper_folder_profile_payload(_dataDir, asset, folderName));
        }
        return entry.payload;
    }

    const NeutralFilterDatabaseAsset& Library::neutral_filter_database_for_dichroic_set(int dichroicSetChoice) {
        ensure_neutral_filter_databases();
        if (dichroicSetChoice < 0 || dichroicSetChoice >= static_cast<int>(_neutralFilterDatabases.size())) {
            dichroicSetChoice = 0;
        }
        return _neutralFilterDatabases[static_cast<size_t>(dichroicSetChoice)];
    }

    std::shared_ptr<const StaticNoisePayloadSet> Library::static_noise_payloads() {
        ensure_static_noise_assets();
        std::lock_guard<std::mutex> lock(_staticNoisePayloadCache->mutex);
        if (!_staticNoisePayloadCache->payloads) {
            _staticNoisePayloadCache->payloads =
                std::make_shared<StaticNoisePayloadSet>(load_static_noise_payloads(*_staticNoiseAssets));
        }
        return _staticNoisePayloadCache->payloads;
    }

    const DichroicFilterCurveSet& Library::dichroic_filter_curves_for_choice(int dichroicSetChoice) {
        ensure_dichroic_filter_sets();
        if (dichroicSetChoice < 0 || dichroicSetChoice >= kDichroicFilterSetCount) {
            dichroicSetChoice = 0;
        }

        const size_t index = static_cast<size_t>(dichroicSetChoice);
        std::lock_guard<std::mutex> lock(_dichroicFilterCurveCache->mutex);
        DichroicFilterCurveCacheEntry& entry = _dichroicFilterCurveCache->entries[index];
        if (!entry.ready) {
            entry.curves = load_dichroic_filter_curves(_dichroicFilterSets[index]);
            entry.ready = true;
        }
        return entry.curves;
    }

    const IlluminantFilterCurveSet& Library::illuminant_filter_curves() {
        ensure_illuminant_filter_assets();
        std::lock_guard<std::mutex> lock(_illuminantFilterCurveCache->mutex);
        IlluminantFilterCurveCacheEntry& entry = _illuminantFilterCurveCache->entry;
        if (!entry.ready) {
            entry.curves = load_illuminant_filter_curves(*_illuminantFilterAssets);
            entry.ready = illuminant_filter_curves_complete(entry.curves);
        }
        return entry.curves;
    }

    PrintRuntimeAssetSet Library::print_runtime_assets_for_choices(const PrintRuntimeChoices& choices) {
        PrintRuntimeAssetSet assets;
        assets.filmStock = film_stock_for_index(choices.filmIndex);
        assets.printPaper = print_paper_for_index(choices.printPaperIndex);
        assets.neutralFilters = neutral_filter_database_for_dichroic_set(choices.dichroicSetChoice);
        return assets;
    }

    bool Library::load_agx_film_profile(const FilmStockAsset& asset, Profiles::AgxFilmProfile& outProfile) {
        const std::string jsonPath = profile_json_path_for_key(ProfileJsonPathRequest{_dataDir, asset.jsonKey});
        if (jsonPath.empty()) {
            outProfile = Profiles::AgxFilmProfile{};
            return false;
        }
        return load_agx_profile_path(jsonPath, outProfile);
    }

    bool Library::load_agx_print_profile(const PrintPaperAsset& asset, Profiles::AgxFilmProfile& outProfile) {
        const std::string jsonPath = profile_json_path_for_key(ProfileJsonPathRequest{_dataDir, asset.jsonKey});
        if (jsonPath.empty()) {
            outProfile = Profiles::AgxFilmProfile{};
            return false;
        }
        return load_agx_profile_path(jsonPath, outProfile);
    }

    bool Library::load_agx_profile_path(const std::string& jsonPath, Profiles::AgxFilmProfile& outProfile) {
        outProfile = Profiles::AgxFilmProfile{};

        const std::string cacheKey = normalize_path_for_cache_key(jsonPath);
        const ProfileFileStamp stamp = read_profile_file_stamp(jsonPath);
        {
            std::lock_guard<std::mutex> lock(_profileCache->mutex);
            if (get_profile(_profileCache->profiles, cacheKey, stamp, outProfile)) {
                return true;
            }
        }

        Profiles::AgxFilmProfile parsedProfile;
        if (!Profiles::load_agx_film_profile_json(jsonPath, parsedProfile)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(_profileCache->mutex);
            store_profile(_profileCache->profiles, cacheKey, stamp, parsedProfile);
        }
        outProfile = std::move(parsedProfile);
        return true;
    }

    void Library::release_cached_payloads() noexcept {
        try {
            if (_neutralFilterCache) {
                std::lock_guard<std::mutex> lock(_neutralFilterCache->mutex);
                _neutralFilterCache->entries.clear();
            }
            if (_printPaperFolderProfilePayloadCache) {
                std::lock_guard<std::mutex> lock(_printPaperFolderProfilePayloadCache->mutex);
                _printPaperFolderProfilePayloadCache->entries.clear();
            }
            if (_staticNoisePayloadCache) {
                std::lock_guard<std::mutex> lock(_staticNoisePayloadCache->mutex);
                _staticNoisePayloadCache->payloads.reset();
            }
            if (_dichroicFilterCurveCache) {
                std::lock_guard<std::mutex> lock(_dichroicFilterCurveCache->mutex);
                for (DichroicFilterCurveCacheEntry& entry : _dichroicFilterCurveCache->entries) {
                    entry = DichroicFilterCurveCacheEntry{};
                }
            }
            if (_illuminantFilterCurveCache) {
                std::lock_guard<std::mutex> lock(_illuminantFilterCurveCache->mutex);
                _illuminantFilterCurveCache->entry = IlluminantFilterCurveCacheEntry{};
            }
            if (_profileCache) {
                std::lock_guard<std::mutex> lock(_profileCache->mutex);
                _profileCache->profiles.clear();
            }
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
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
