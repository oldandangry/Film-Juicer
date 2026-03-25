#include "NeutralFilters.h"

#include "Logging.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "nlohmann/json.hpp"

namespace {
    using Json = nlohmann::json;
    using Clock = std::chrono::steady_clock;

    constexpr std::int64_t kNeutralFilterDiagnosticsReloadCheckMs = 1000;
    constexpr std::uint64_t kFnvOffsetBasis64 = 1469598103934665603ull;
    constexpr std::uint64_t kFnvPrime64 = 1099511628211ull;

    struct FileStamp {
        bool valid = false;
        std::uint64_t sizeBytes = 0;
        std::int64_t writeTimeTicks = 0;
    };

    struct ParsedNeutralFilterDb {
        std::unordered_map<std::string, std::tuple<float, float, float>> lookup;
        FileStamp stamp;
        std::string versionHash;
    };

    struct NeutralFilterCacheEntry {
        std::shared_ptr<const ParsedNeutralFilterDb> db;
        Clock::time_point lastDiagnosticsReloadCheck{};
        bool hasDiagnosticsReloadCheck = false;
    };

    std::mutex gNeutralFilterCacheMutex;
    std::unordered_map<std::string, NeutralFilterCacheEntry> gNeutralFilterCache;

    std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
            });
        return s;
    }

    const char* thread_class_name(NeutralFilterThreadClass threadClass) {
        switch (threadClass) {
        case NeutralFilterThreadClass::Control:
            return "control";
        case NeutralFilterThreadClass::RenderWorker:
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
        std::filesystem::path path(jsonPath);
        path.make_preferred();
        return to_lower(path.lexically_normal().string());
    }

    bool same_file_stamp(const FileStamp& a, const FileStamp& b) {
        return a.valid && b.valid
            && a.sizeBytes == b.sizeBytes
            && a.writeTimeTicks == b.writeTimeTicks;
    }

    FileStamp read_file_stamp(const std::string& jsonPath) {
        std::filesystem::path path(jsonPath);
        std::error_code ec;
        const auto sizeBytes = std::filesystem::file_size(path, ec);
        if (ec) {
            return {};
        }
        const auto writeTime = std::filesystem::last_write_time(path, ec);
        if (ec) {
            return {};
        }

        FileStamp stamp;
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

    bool build_lookup_table(
        const Json& root,
        std::unordered_map<std::string, std::tuple<float, float, float>>& outLookup)
    {
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

    std::string compute_version_hash(const Json& root, const FileStamp& stamp) {
        std::uint64_t hash = kFnvOffsetBasis64;
        const std::uint8_t validByte = stamp.valid ? 1u : 0u;
        hash = fnv1a_append(hash, &validByte, sizeof(validByte));
        hash = fnv1a_append(hash, &stamp.sizeBytes, sizeof(stamp.sizeBytes));
        hash = fnv1a_append(hash, &stamp.writeTimeTicks, sizeof(stamp.writeTimeTicks));
        const std::string jsonBlob = root.dump();
        hash = fnv1a_append(hash, jsonBlob.data(), jsonBlob.size());
        return hash_to_hex(hash);
    }

    bool parse_db_from_disk(
        const std::string& jsonPath,
        ParsedNeutralFilterDb& outDb,
        std::string& outReason)
    {
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
        if (!build_lookup_table(root, lookup)) {
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
        NeutralFilterThreadClass threadClass,
        const char* reason = nullptr,
        const std::string* path = nullptr)
    {
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

    bool lookup_triplet(
        const ParsedNeutralFilterDb& db,
        const std::string& lookupKey,
        std::tuple<float, float, float>& outYMC)
    {
        const auto it = db.lookup.find(lookupKey);
        if (it == db.lookup.end()) {
            return false;
        }
        outYMC = it->second;
        return true;
    }
}

bool load_enlarger_neutral_filters(
    const std::string& jsonPath,
    const std::string& paperKey,
    const std::string& illuminantKey,
    const std::string& negativeKey,
    std::tuple<float, float, float>& outYMC,
    NeutralFilterThreadClass threadClass,
    std::string* outSelectedDbVersionHash)
{
    if (outSelectedDbVersionHash) {
        outSelectedDbVersionHash->clear();
    }

    if (paperKey.empty() || illuminantKey.empty() || negativeKey.empty()) {
        trace_neutral_filter_event("miss", "none", threadClass, "missing_lookup_key", &jsonPath);
        return false;
    }

    const std::string lookupKey = make_lookup_key(paperKey, illuminantKey, negativeKey);
    const std::string cacheKey = normalize_path_for_cache_key(jsonPath);
    if (cacheKey.empty()) {
        trace_neutral_filter_event("miss", "none", threadClass, "empty_path", &jsonPath);
        return false;
    }

    const bool diagnosticsReload = diagnostics_reload_enabled();
    const Clock::time_point now = Clock::now();

    std::shared_ptr<const ParsedNeutralFilterDb> dbSnapshot;
    {
        std::lock_guard<std::mutex> lock(gNeutralFilterCacheMutex);
        auto cacheIt = gNeutralFilterCache.find(cacheKey);
        NeutralFilterCacheEntry* cacheEntry = (cacheIt != gNeutralFilterCache.end())
            ? &cacheIt->second
            : nullptr;
        dbSnapshot = cacheEntry ? cacheEntry->db : nullptr;

        if (!dbSnapshot) {
            if (threadClass == NeutralFilterThreadClass::RenderWorker) {
                trace_neutral_filter_event("miss", "none", threadClass, "cache_cold_render_worker", &jsonPath);
                return false;
            }

            ParsedNeutralFilterDb parsedDb;
            std::string parseReason;
            if (!parse_db_from_disk(jsonPath, parsedDb, parseReason)) {
                trace_neutral_filter_event("reload_failed", "none", threadClass, parseReason.c_str(), &jsonPath);
                return false;
            }

            dbSnapshot = std::make_shared<ParsedNeutralFilterDb>(std::move(parsedDb));
            cacheIt = gNeutralFilterCache.emplace(cacheKey, NeutralFilterCacheEntry{}).first;
            cacheEntry = &cacheIt->second;
            cacheEntry->db = dbSnapshot;
            if (diagnosticsReload) {
                cacheEntry->lastDiagnosticsReloadCheck = now;
                cacheEntry->hasDiagnosticsReloadCheck = true;
            }
            trace_neutral_filter_event("load", dbSnapshot->versionHash, threadClass, nullptr, &jsonPath);
        }

        if (diagnosticsReload && threadClass == NeutralFilterThreadClass::Control && dbSnapshot) {
            if (!cacheEntry) {
                trace_neutral_filter_event("reload_failed", dbSnapshot->versionHash, threadClass, "cache_entry_missing", &jsonPath);
                return false;
            }
            bool shouldCheckReload = false;
            if (!cacheEntry->hasDiagnosticsReloadCheck) {
                shouldCheckReload = true;
            }
            else {
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - cacheEntry->lastDiagnosticsReloadCheck).count();
                shouldCheckReload = elapsed >= kNeutralFilterDiagnosticsReloadCheckMs;
            }

            if (!shouldCheckReload) {
                trace_neutral_filter_event("reload_skip", dbSnapshot->versionHash, threadClass, "interval_not_elapsed", &jsonPath);
            }
            else {
                cacheEntry->lastDiagnosticsReloadCheck = now;
                cacheEntry->hasDiagnosticsReloadCheck = true;
                trace_neutral_filter_event("reload_check", dbSnapshot->versionHash, threadClass, nullptr, &jsonPath);

                const FileStamp newStamp = read_file_stamp(jsonPath);
                if (!newStamp.valid) {
                    trace_neutral_filter_event("reload_failed", dbSnapshot->versionHash, threadClass, "stamp_unavailable", &jsonPath);
                }
                else if (!dbSnapshot->stamp.valid) {
                    trace_neutral_filter_event("reload_failed", dbSnapshot->versionHash, threadClass, "cached_stamp_unavailable", &jsonPath);
                }
                else if (same_file_stamp(newStamp, dbSnapshot->stamp)) {
                    trace_neutral_filter_event("reload_skip", dbSnapshot->versionHash, threadClass, "file_stamp_unchanged", &jsonPath);
                }
                else {
                    ParsedNeutralFilterDb parsedDb;
                    std::string parseReason;
                    if (!parse_db_from_disk(jsonPath, parsedDb, parseReason)) {
                        trace_neutral_filter_event("reload_failed", dbSnapshot->versionHash, threadClass, parseReason.c_str(), &jsonPath);
                    }
                    else {
                        dbSnapshot = std::make_shared<ParsedNeutralFilterDb>(std::move(parsedDb));
                        cacheEntry->db = dbSnapshot;
                        trace_neutral_filter_event("reload_commit", dbSnapshot->versionHash, threadClass, nullptr, &jsonPath);
                    }
                }
            }
        }
    }

    if (!dbSnapshot) {
        trace_neutral_filter_event("miss", "none", threadClass, "cache_unavailable", &jsonPath);
        return false;
    }

    if (!lookup_triplet(*dbSnapshot, lookupKey, outYMC)) {
        trace_neutral_filter_event("miss", dbSnapshot->versionHash, threadClass, "entry_not_found", &jsonPath);
        return false;
    }

    if (outSelectedDbVersionHash) {
        *outSelectedDbVersionHash = dbSnapshot->versionHash;
    }
    trace_neutral_filter_event("hit", dbSnapshot->versionHash, threadClass, nullptr, &jsonPath);
    return true;
}
