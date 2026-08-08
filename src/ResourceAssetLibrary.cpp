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

        struct IlluminantFilterCurveCacheEntry {
            IlluminantFilterCurveSet curves;
            bool ready = false;
        };

        struct FilterDbRead {
            std::shared_ptr<const ParsedNeutralFilterDb> db;
            bool stop = false;
        };

        struct ProfileCacheEntry {
            std::string cacheKey;
            ProfileFileStamp stamp;
            Profiles::SpektrafilmProfileJson profile;
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

        void hash_string(std::uint64_t& hash, const std::string& value) {
            hash = fnv1a_append(hash, value.data(), value.size());
        }

        template <typename T>
        void hash_value(std::uint64_t& hash, const T& value) {
            hash = fnv1a_append(hash, &value, sizeof(value));
        }

        bool read_file_bytes(
            const std::string& path,
            std::string& out,
            std::uint64_t* outHash = nullptr) {
            out.clear();
            if (outHash) {
                *outHash = 0;
            }
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) {
                return false;
            }
            const std::streamsize size = file.tellg();
            if (size < 0) {
                return false;
            }
            out.resize(static_cast<std::size_t>(size));
            file.seekg(0, std::ios::beg);
            if (size > 0 && !file.read(out.data(), size)) {
                out.clear();
                return false;
            }
            if (outHash) {
                *outHash = fnv1a_append(kFnvOffsetBasis64, out.data(), out.size());
            }
            return true;
        }

        std::string make_lookup_key(const std::string& printProfileKey, const std::string& illuminantKey, const std::string& filmProfileKey) {
            std::string key;
            key.reserve(printProfileKey.size() + illuminantKey.size() + filmProfileKey.size() + 2);
            key += to_lower(printProfileKey);
            key.push_back('\x1f');
            key += to_lower(illuminantKey);
            key.push_back('\x1f');
            key += to_lower(filmProfileKey);
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
            Profiles::SpektrafilmProfileJson& outProfile) {
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
            const Profiles::SpektrafilmProfileJson& profile) {
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

        bool parse_array_triplet(const Json& arrNode, std::tuple<float, float, float>& outCmyCc) {
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

            outCmyCc = std::make_tuple(vals[0], vals[1], vals[2]);
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
                const std::string& printProfileKey = paperIt.key();
                for (auto illuminantIt = paperIt->cbegin(); illuminantIt != paperIt->cend(); ++illuminantIt) {
                    if (!illuminantIt->is_object()) {
                        continue;
                    }
                    const std::string& illuminantKey = illuminantIt.key();
                    for (auto negativeIt = illuminantIt->cbegin(); negativeIt != illuminantIt->cend(); ++negativeIt) {
                        std::tuple<float, float, float> cmyCc{};
                        if (!parse_array_triplet(*negativeIt, cmyCc)) {
                            continue;
                        }
                        outLookup[make_lookup_key(printProfileKey, illuminantKey, negativeIt.key())] = cmyCc;
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
            const std::string& jsonPath,
            NeutralFilterLookupThread threadClass,
            bool diagnosticsReload,
            Clock::time_point now) {
            FilterDbRead read;
            const std::string cacheKey = normalize_path_for_cache_key(jsonPath);
            if (cacheKey.empty()) {
                trace_neutral_filter_event("miss", "none", threadClass, "empty_path", &jsonPath);
                read.stop = true;
                return read;
            }

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

        NeutralFilterLookupResult lookup_filter_cmy_cc(
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
            result.cmyCc = it->second;
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

        struct ProfilePathRequest {
            const std::string& dataDir;
            const std::string& jsonKey;
        };

        std::string profile_path_for_key(const ProfilePathRequest& request) {
            if (request.jsonKey.empty()) {
                return {};
            }
            std::string fileName = request.jsonKey;
            fileName += ".json";
            return data_path_string(request.dataDir, {"profiles", fileName.c_str()});
        }

        std::string neutral_print_calibration_path(const std::string& dataDir) {
            return data_path_string(dataDir, {"filters", "neutral_print_filters.json"});
        }

        std::string measured_dichroic_relative_path(const std::string& setKey, const char* channel) {
            return "Resources/filters/dichroics/" + setKey + "/filter_" + channel + ".csv";
        }

        struct MeasuredDichroicChannelRequest {
            const std::string& path;
            const std::string& relativePath;
        };

        bool parse_measured_dichroic_channel(
            const MeasuredDichroicChannelRequest& request,
            std::uint64_t& outHash,
            std::array<float, 81>* outTransmittance,
            std::string& diagnostic) {
            std::string bytes;
            std::uint64_t fileHash = 0;
            if (!read_file_bytes(request.path, bytes, &fileHash)) {
                diagnostic = "SelectedDichroicResourceMissing phase=4A resource=" + request.relativePath;
                return false;
            }

            std::vector<std::pair<float, float>> pairs;
            std::istringstream input(bytes);
            std::string line;
            std::size_t lineNumber = 0;
            while (std::getline(input, line)) {
                ++lineNumber;
                const std::size_t comment = line.find('#');
                if (comment != std::string::npos) {
                    line.erase(comment);
                }
                const std::size_t first = line.find_first_not_of(" \t\r\n");
                if (first == std::string::npos) {
                    continue;
                }

                std::istringstream row(line.substr(first));
                float wavelength = 0.0f;
                float percentTransmittance = 0.0f;
                if (!(row >> wavelength)) {
                    diagnostic = "MalformedSelectedDichroicResource phase=4A line=" + std::to_string(lineNumber);
                    return false;
                }
                while (row.peek() == ',' || row.peek() == ';') {
                    row.get();
                }
                if (!(row >> percentTransmittance) ||
                    !std::isfinite(wavelength) ||
                    !std::isfinite(percentTransmittance)) {
                    diagnostic = "MalformedSelectedDichroicResource phase=4A line=" + std::to_string(lineNumber);
                    return false;
                }
                row >> std::ws;
                if (!row.eof()) {
                    diagnostic = "MalformedSelectedDichroicResource phase=4A line=" + std::to_string(lineNumber);
                    return false;
                }
                pairs.emplace_back(wavelength, percentTransmittance);
            }

            const std::vector<std::pair<float, float>> resampled =
                Spectral::resample_pairs_akima_to_reference_axis(pairs);
            if (resampled.size() != 81u) {
                diagnostic = "MalformedSelectedDichroicResource phase=4A field=akima_reference_axis";
                return false;
            }

            std::array<float, 81> transmittance{};
            for (std::size_t i = 0; i < resampled.size(); ++i) {
                const float value = resampled[i].second * 0.01f;
                if (!std::isfinite(value)) {
                    diagnostic = "MalformedSelectedDichroicResource phase=4A field=canonical_axis_coverage";
                    return false;
                }
                transmittance[i] = value;
            }

            std::uint64_t hash = kFnvOffsetBasis64;
            constexpr std::uint32_t kSchemaVersion = 1u;
            hash_value(hash, kSchemaVersion);
            hash_string(hash, request.relativePath);
            hash_value(hash, fileHash);
            hash = fnv1a_append(hash, transmittance.data(), sizeof(transmittance));
            outHash = hash;
            if (outTransmittance) {
                *outTransmittance = transmittance;
            }
            return outHash != 0;
        }

        std::string noise_asset_path(const std::string& dataDir, std::initializer_list<const char*> segments) {
            if (dataDir.empty()) {
                return {};
            }
            return data_path_string(dataDir, segments);
        }

        struct FilmProfileAssetRequest {
            std::string optionLabel;
            std::string jsonKey;
        };

        SelectedFilmProfileAsset make_film_profile_asset(FilmProfileAssetRequest request) {
            SelectedFilmProfileAsset asset;
            asset.optionLabel = std::move(request.optionLabel);
            asset.jsonKey = std::move(request.jsonKey);
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        SelectedPrintProfileAsset make_print_profile_asset(
            std::string optionLabel,
            std::string jsonKey) {
            SelectedPrintProfileAsset asset;
            asset.optionLabel = std::move(optionLabel);
            asset.jsonKey = std::move(jsonKey);
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        void add_print_profile(
            std::vector<SelectedPrintProfileAsset>& assets,
            std::string optionLabel,
            std::string jsonKey) {
            assets.emplace_back(make_print_profile_asset(std::move(optionLabel), std::move(jsonKey)));
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
            curves.tungstenKg3 = Spectral::build_curve_TH_KG3_pinned(asset.kg3Path);
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
                   curve_is_on_reference_axis(curves.tungstenKg3) &&
                   curve_is_on_reference_axis(curves.tungstenKg3Lens);
        }

    } // namespace

    struct Library::NeutralFilterCacheState {
        std::mutex mutex;
        std::unordered_map<std::string, NeutralFilterCacheEntry> entries;
    };

    struct Library::StaticNoisePayloadCacheState {
        std::mutex mutex;
        std::shared_ptr<const StaticNoisePayloadSet> payloads;
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
          _illuminantFilterAssets(std::make_unique<IlluminantFilterAssetSet>()),
          _neutralFilterCache(std::make_unique<NeutralFilterCacheState>()),
          _staticNoisePayloadCache(std::make_unique<StaticNoisePayloadCacheState>()),
          _illuminantFilterCurveCache(std::make_unique<IlluminantFilterCurveCacheState>()),
          _profileCache(std::make_unique<ProfileCacheState>()),
          _selectedProfileAssets(std::make_unique<Profiles::ProfileAssetStore>()) {
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

    void Library::ensure_illuminant_filter_assets() {
        std::call_once(_illuminantFilterOnce, [this]() {
            load_illuminant_filter_assets();
        });
    }

    void Library::load_catalogs() {
        const bool traceCatalog = JTRACE_ENABLED(1);
        _filmStocks.clear();
        _printPapers.clear();

        _spektrafilmProfileCatalog = Spektrafilm::build_profile_catalog(_dataDir);
        if (!_spektrafilmProfileCatalog.valid) {
            if (traceCatalog) {
                JTRACE("CATALOG", "spektrafilm profile catalog unavailable: " + _spektrafilmProfileCatalog.failure);
            }
            return;
        }

        _filmStocks.reserve(_spektrafilmProfileCatalog.filmProfiles.size());
        for (const Spektrafilm::ProfileCatalogEntry& entry : _spektrafilmProfileCatalog.filmProfiles) {
            _filmStocks.emplace_back(make_film_profile_asset(FilmProfileAssetRequest{entry.label, entry.key}));
            _filmStocks.back().version = entry.sourceVersion;
        }

        _printPapers.reserve(_spektrafilmProfileCatalog.printProfiles.size());
        for (const Spektrafilm::ProfileCatalogEntry& entry : _spektrafilmProfileCatalog.printProfiles) {
            add_print_profile(_printPapers, entry.label, entry.key);
            _printPapers.back().version = entry.sourceVersion;
        }

        if (traceCatalog) {
            std::ostringstream oss;
            oss << "spektrafilm profile catalog film=" << _filmStocks.size()
                << " print=" << _printPapers.size()
                << " defaultFilm=" << (_spektrafilmProfileCatalog.defaultFilmPresent ? 1 : 0)
                << " defaultPrint=" << (_spektrafilmProfileCatalog.defaultPrintPresent ? 1 : 0);
            JTRACE("CATALOG", oss.str());
        }
    }

    void Library::load_neutral_filter_databases() {
        auto makePaths = [this]() {
            NeutralFilterDatabasePathSet paths;
            paths.selectedPath = neutral_print_calibration_path(_dataDir);
            paths.version = Library::kProcessAssetVersion;
            return paths;
        };

        _neutralFilterDatabases[0] = make_neutral_filter_database(0);
        _neutralFilterDatabasePaths[0] = makePaths();
        _neutralFilterDatabases[1] = make_neutral_filter_database(1);
        _neutralFilterDatabasePaths[1] = makePaths();
        _neutralFilterDatabases[2] = make_neutral_filter_database(2);
        _neutralFilterDatabasePaths[2] = makePaths();
    }

    NeutralFilterLookupResult Library::lookup_neutral_filter_path(
        const NeutralFilterPathLookup& lookup) {
        NeutralFilterLookupResult result;
        const NeutralFilterLookupKey& key = lookup.lookupKey;

        if (key.printProfileKey.empty() || key.illuminantKey.empty() || key.filmProfileKey.empty()) {
            trace_neutral_filter_event("miss", "none", lookup.threadClass, "missing_lookup_key", &lookup.jsonPath);
            return result;
        }

        const std::string lookupKey = make_lookup_key(key.printProfileKey, key.illuminantKey, key.filmProfileKey);
        const bool diagnosticsReload = diagnostics_reload_enabled();
        const Clock::time_point now = Clock::now();

        FilterDbRead dbRead = get_filter_db(
            _neutralFilterCache->mutex,
            _neutralFilterCache->entries,
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

        return lookup_filter_cmy_cc(*dbRead.db, lookupKey, lookup.threadClass);
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

        return lookup_neutral_filter_path(
            NeutralFilterPathLookup{paths.selectedPath, lookupKey, threadClass});
    }

    void Library::load_static_noise_assets() {
        *_staticNoiseAssets = make_static_noise_assets(_dataDir);
    }

    void Library::load_illuminant_filter_assets() {
        *_illuminantFilterAssets = make_illuminant_filter_assets(_dataDir);
    }

    const Spektrafilm::ProfileCatalog& Library::spektrafilm_profile_catalog() {
        ensure_catalogs();
        return _spektrafilmProfileCatalog;
    }

    const SelectedFilmProfileAsset& Library::film_profile_for_key(const std::string& key) {
        ensure_catalogs();
        static const SelectedFilmProfileAsset empty{};
        for (const SelectedFilmProfileAsset& asset : _filmStocks) {
            if (asset.jsonKey == key) {
                return asset;
            }
        }
        return empty;
    }

    const SelectedPrintProfileAsset& Library::print_profile_for_key(const std::string& key) {
        ensure_catalogs();
        static const SelectedPrintProfileAsset empty{};
        for (const SelectedPrintProfileAsset& asset : _printPapers) {
            if (asset.jsonKey == key) {
                return asset;
            }
        }
        return empty;
    }

    std::shared_ptr<const Profiles::ValidatedFilmProfile> Library::selected_film_profile_for_key(
        const std::string& key) {
        ensure_catalogs();
        return _selectedProfileAssets->load_film_profile_by_key(_spektrafilmProfileCatalog, key);
    }

    std::shared_ptr<const Profiles::ValidatedPrintProfile> Library::selected_print_profile_for_key(
        const std::string& key) {
        ensure_catalogs();
        return _selectedProfileAssets->load_print_profile_by_key(_spektrafilmProfileCatalog, key);
    }

    SelectedProfileResult Library::selected_profiles_for_route(const SelectedProfileRequest& request) {
        ensure_catalogs();
        return _selectedProfileAssets->selected_profiles_for_route(_spektrafilmProfileCatalog, request);
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

    MeasuredDichroicResourceIdentity Library::measured_dichroic_resource_identity(const std::string& setKey) {
        MeasuredDichroicResourceIdentity result;
        result.setKey = setKey;
        constexpr std::array<const char*, 3> kChannelsCmy{{"c", "m", "y"}};
        for (std::size_t channel = 0; channel < kChannelsCmy.size(); ++channel) {
            const char* channelKey = kChannelsCmy[channel];
            result.resourcePathsCmy[channel] = measured_dichroic_relative_path(setKey, channelKey);
            const std::string fileName = std::string("filter_") + channelKey + ".csv";
            const std::string path =
                data_path_string(_dataDir, {"filters", "dichroics", setKey.c_str(), fileName.c_str()});
            if (!parse_measured_dichroic_channel(
                    MeasuredDichroicChannelRequest{path, result.resourcePathsCmy[channel]},
                    result.resourceHashesCmy[channel],
                    nullptr,
                    result.diagnostic)) {
                return result;
            }
        }

        std::uint64_t hash = kFnvOffsetBasis64;
        constexpr std::uint32_t kSchemaVersion = 1u;
        hash_value(hash, kSchemaVersion);
        hash_string(hash, result.setKey);
        for (std::size_t channel = 0; channel < result.resourcePathsCmy.size(); ++channel) {
            hash_string(hash, result.resourcePathsCmy[channel]);
            hash_value(hash, result.resourceHashesCmy[channel]);
        }
        result.hash = hash;
        result.valid = result.hash != 0;
        return result;
    }

    MeasuredDichroicCurveResult Library::measured_dichroic_curves(const std::string& setKey) {
        MeasuredDichroicCurveResult result;
        constexpr std::array<const char*, 3> kChannelsCmy{{"c", "m", "y"}};
        std::uint64_t hash = kFnvOffsetBasis64;
        constexpr std::uint32_t kSchemaVersion = 1u;
        hash_value(hash, kSchemaVersion);
        hash_string(hash, setKey);
        for (std::size_t channel = 0; channel < kChannelsCmy.size(); ++channel) {
            const char* channelKey = kChannelsCmy[channel];
            const std::string relativePath = measured_dichroic_relative_path(setKey, channelKey);
            const std::string fileName = std::string("filter_") + channelKey + ".csv";
            const std::string path =
                data_path_string(_dataDir, {"filters", "dichroics", setKey.c_str(), fileName.c_str()});
            if (!parse_measured_dichroic_channel(
                    MeasuredDichroicChannelRequest{path, relativePath},
                    result.resourceHashesCmy[channel],
                    &result.transmittanceCmy[channel],
                    result.diagnostic)) {
                return result;
            }
            hash_string(hash, relativePath);
            hash_value(hash, result.resourceHashesCmy[channel]);
        }
        result.hash = hash;
        result.valid = result.hash != 0;
        return result;
    }

    NeutralPrintCalibrationResult Library::neutral_print_calibration(
        const std::string& printProfileKey,
        const std::string& printIlluminantKey,
        const std::string& filmProfileKey) {
        NeutralPrintCalibrationResult result;
        const std::string path = neutral_print_calibration_path(_dataDir);
        std::string bytes;
        if (!read_file_bytes(path, bytes)) {
            std::error_code ec;
            if (fs::exists(path, ec) && !ec) {
                result.status = NeutralPrintCalibrationStatus::Malformed;
                result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=resource_read";
            } else {
                result.status = NeutralPrintCalibrationStatus::MissingFile;
            }
        } else {
            const Json root = Json::parse(bytes, nullptr, false);
            if (root.is_discarded() || !root.is_object()) {
                result.status = NeutralPrintCalibrationStatus::Malformed;
                result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=root";
            } else {
                const auto printIt = root.find(printProfileKey);
                if (printIt == root.end()) {
                    result.status = NeutralPrintCalibrationStatus::MissingEntry;
                } else if (!printIt->is_object()) {
                    result.status = NeutralPrintCalibrationStatus::Malformed;
                    result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=print_profile";
                } else {
                    const auto illuminantIt = printIt->find(printIlluminantKey);
                    if (illuminantIt == printIt->end()) {
                        result.status = NeutralPrintCalibrationStatus::MissingEntry;
                    } else if (!illuminantIt->is_object()) {
                        result.status = NeutralPrintCalibrationStatus::Malformed;
                        result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=print_illuminant";
                    } else {
                        const auto filmIt = illuminantIt->find(filmProfileKey);
                        if (filmIt == illuminantIt->end()) {
                            result.status = NeutralPrintCalibrationStatus::MissingEntry;
                        } else if (!filmIt->is_array() || filmIt->size() != result.cmyCc.size()) {
                            result.status = NeutralPrintCalibrationStatus::Malformed;
                            result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=cmy_cc";
                        } else {
                            result.status = NeutralPrintCalibrationStatus::Found;
                            for (std::size_t channel = 0; channel < result.cmyCc.size(); ++channel) {
                                const Json& value = (*filmIt)[channel];
                                if (!value.is_number()) {
                                    result.status = NeutralPrintCalibrationStatus::Malformed;
                                    result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=cmy_cc";
                                    break;
                                }
                                const double cc = value.get<double>();
                                if (!std::isfinite(cc)) {
                                    result.status = NeutralPrintCalibrationStatus::Malformed;
                                    result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=cmy_cc";
                                    break;
                                }
                                result.cmyCc[channel] = static_cast<float>(cc);
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    PrintRuntimeAssetSet Library::print_runtime_assets_for_profile_keys(const PrintRuntimeProfileKeyChoices& choices) {
        PrintRuntimeAssetSet assets;
        assets.filmStock = film_profile_for_key(choices.filmProfileKey);
        assets.printPaper = print_profile_for_key(choices.printProfileKey);
        assets.neutralFilters = neutral_filter_database_for_dichroic_set(choices.dichroicSetChoice);
        return assets;
    }

    bool Library::load_spektrafilm_film_profile(const SelectedFilmProfileAsset& asset, Profiles::SpektrafilmProfileJson& outProfile) {
        const std::string jsonPath = profile_path_for_key(ProfilePathRequest{_dataDir, asset.jsonKey});
        if (jsonPath.empty()) {
            outProfile = Profiles::SpektrafilmProfileJson{};
            return false;
        }
        return load_spektrafilm_profile_path(jsonPath, outProfile);
    }

    bool Library::load_spektrafilm_print_profile(const SelectedPrintProfileAsset& asset, Profiles::SpektrafilmProfileJson& outProfile) {
        const std::string jsonPath = profile_path_for_key(ProfilePathRequest{_dataDir, asset.jsonKey});
        if (jsonPath.empty()) {
            outProfile = Profiles::SpektrafilmProfileJson{};
            return false;
        }
        return load_spektrafilm_profile_path(jsonPath, outProfile);
    }

    bool Library::load_spektrafilm_profile_path(const std::string& jsonPath, Profiles::SpektrafilmProfileJson& outProfile) {
        outProfile = Profiles::SpektrafilmProfileJson{};

        const std::string cacheKey = normalize_path_for_cache_key(jsonPath);
        const ProfileFileStamp stamp = read_profile_file_stamp(jsonPath);
        {
            std::lock_guard<std::mutex> lock(_profileCache->mutex);
            if (get_profile(_profileCache->profiles, cacheKey, stamp, outProfile)) {
                return true;
            }
        }

        Profiles::SpektrafilmProfileJson parsedProfile;
        if (!Profiles::load_spektrafilm_profile_json(jsonPath, parsedProfile)) {
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
            if (_staticNoisePayloadCache) {
                std::lock_guard<std::mutex> lock(_staticNoisePayloadCache->mutex);
                _staticNoisePayloadCache->payloads.reset();
            }
            if (_illuminantFilterCurveCache) {
                std::lock_guard<std::mutex> lock(_illuminantFilterCurveCache->mutex);
                _illuminantFilterCurveCache->entry = IlluminantFilterCurveCacheEntry{};
            }
            if (_profileCache) {
                std::lock_guard<std::mutex> lock(_profileCache->mutex);
                _profileCache->profiles.clear();
            }
            if (_selectedProfileAssets) {
                _selectedProfileAssets->release_cached_payloads();
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
