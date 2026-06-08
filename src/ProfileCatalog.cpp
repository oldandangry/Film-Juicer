#include "ProfileCatalog.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <unordered_set>

#include "nlohmann/json.hpp"

namespace Spektrafilm {
    namespace {
        namespace fs = std::filesystem;
        using Json = nlohmann::json;

        constexpr std::uint64_t kFnvOffsetBasis64 = 1469598103934665603ull;
        constexpr std::uint64_t kFnvPrime64 = 1099511628211ull;

        std::uint64_t hash_string(std::uint64_t h, const std::string& value) {
            for (const unsigned char c : value) {
                h ^= static_cast<std::uint64_t>(c);
                h *= kFnvPrime64;
            }
            return h;
        }

        std::uint64_t source_version_for_path(const fs::path& path) {
            std::uint64_t h = hash_string(kFnvOffsetBasis64, path.generic_string());
            std::error_code ec;
            const std::uintmax_t size = fs::file_size(path, ec);
            if (!ec) {
                h ^= static_cast<std::uint64_t>(size);
                h *= kFnvPrime64;
            }
            return h;
        }

        std::string json_string_member_or(const Json& object, const char* key, std::string fallback) {
            const auto it = object.find(key);
            if (it == object.end() || !it->is_string()) {
                return fallback;
            }
            return it->get<std::string>();
        }

        ProfileSupport parse_support(const std::string& value) {
            if (value == "film") {
                return ProfileSupport::Film;
            }
            if (value == "paper") {
                return ProfileSupport::Paper;
            }
            return ProfileSupport::Unsupported;
        }

        ProfileStage parse_stage(const std::string& value) {
            if (value == "filming") {
                return ProfileStage::Filming;
            }
            if (value == "printing") {
                return ProfileStage::Printing;
            }
            return ProfileStage::Unsupported;
        }

        ProfilePolarity parse_polarity(const std::string& value) {
            if (value == "negative") {
                return ProfilePolarity::Negative;
            }
            if (value == "positive") {
                return ProfilePolarity::Positive;
            }
            return ProfilePolarity::Unsupported;
        }

        enum class EntryReadDisposition {
            Profile,
            NonProfile,
            Unavailable
        };

        bool has_profile_shaped_data(const Json& root) {
            const auto dataIt = root.find("data");
            if (dataIt == root.end() || !dataIt->is_object()) {
                return false;
            }
            static constexpr const char* kProfileDataMembers[] = {
                "wavelengths",
                "log_sensitivity",
                "channel_density",
                "base_density",
                "log_exposure",
                "density_curves"};
            return std::any_of(
                std::begin(kProfileDataMembers),
                std::end(kProfileDataMembers),
                [dataIt](const char* key) {
                    return dataIt->contains(key);
                });
        }

        template <typename ValueT, typename ParseFn>
        ValueT read_bounded_metadata(
            const Json& info,
            const char* key,
            const char* defaultValue,
            ValueT unsupportedValue,
            bool& defaulted,
            ParseFn&& parse) {
            const auto it = info.find(key);
            defaulted = it == info.end();
            if (defaulted) {
                return parse(defaultValue);
            }
            if (!it->is_string()) {
                return unsupportedValue;
            }
            return parse(it->get<std::string>());
        }

        EntryReadDisposition read_profile_entry(
            const fs::path& path,
            ProfileCatalogEntry& entry,
            std::string& error) {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                error = "open failed";
                return EntryReadDisposition::Unavailable;
            }

            Json root;
            try {
                file >> root;
            } catch (const Json::exception& ex) {
                error = ex.what();
                return EntryReadDisposition::Unavailable;
            }

            if (!has_profile_shaped_data(root)) {
                return EntryReadDisposition::NonProfile;
            }

            const Json empty = Json::object();
            const Json& info = root.contains("info") && root["info"].is_object() ? root["info"] : empty;

            entry.key = json_string_member_or(info, "stock", std::string());
            if (entry.key.empty()) {
                error = "missing info.stock";
                return EntryReadDisposition::Unavailable;
            }

            entry.label = json_string_member_or(info, "name", entry.key);
            entry.sourcePath = path.generic_string();
            entry.support = read_bounded_metadata(
                info,
                "support",
                "film",
                ProfileSupport::Unsupported,
                entry.supportDefaulted,
                parse_support);
            entry.stage = read_bounded_metadata(
                info,
                "stage",
                "filming",
                ProfileStage::Unsupported,
                entry.stageDefaulted,
                parse_stage);
            entry.polarity = read_bounded_metadata(
                info,
                "type",
                "negative",
                ProfilePolarity::Unsupported,
                entry.polarityDefaulted,
                parse_polarity);
            entry.sourceVersion = source_version_for_path(path);
            return EntryReadDisposition::Profile;
        }

        bool role_key_inserted(
            std::unordered_set<std::string>& keys,
            const ProfileCatalogEntry& entry,
            const char* role,
            ProfileCatalog& catalog) {
            if (keys.insert(entry.key).second) {
                return true;
            }
            catalog.failure = std::string("duplicate ") + role + " profile key: " + entry.key;
            catalog.valid = false;
            return false;
        }

        void sort_by_label(std::vector<ProfileCatalogEntry>& entries) {
            std::sort(entries.begin(), entries.end(), [](const ProfileCatalogEntry& a, const ProfileCatalogEntry& b) {
                if (a.label == b.label) {
                    return a.key < b.key;
                }
                return a.label < b.label;
            });
        }
    } // namespace

    ProfileCatalog build_profile_catalog(const std::string& dataDir) {
        ProfileCatalog catalog;
        catalog.ignoredResources.emplace_back("Resources/filters/neutral_print_filters.json");
        catalog.ignoredResources.emplace_back("Resources/profiles/enlarger_neutral*.json");

        const fs::path profilesDir = fs::path(dataDir) / "profiles";
        std::error_code ec;
        if (!fs::exists(profilesDir, ec) || !fs::is_directory(profilesDir, ec)) {
            catalog.failure = "Resources/profiles is missing";
            return catalog;
        }

        std::unordered_set<std::string> filmKeys;
        std::unordered_set<std::string> printKeys;

        for (fs::directory_iterator it(profilesDir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec) || it->path().extension() != ".json") {
                continue;
            }

            ProfileCatalogEntry entry;
            std::string error;
            const EntryReadDisposition disposition = read_profile_entry(it->path(), entry, error);
            if (disposition == EntryReadDisposition::NonProfile) {
                catalog.ignoredResources.emplace_back(it->path().generic_string());
                continue;
            }
            if (disposition == EntryReadDisposition::Unavailable) {
                catalog.unavailableProfiles.emplace_back(it->path().generic_string() + ": " + error);
                continue;
            }

            const bool visibleAsFilm = entry.support == ProfileSupport::Film &&
                                       entry.stage == ProfileStage::Filming &&
                                       entry.polarity != ProfilePolarity::Unsupported;
            const bool visibleAsPrint = entry.support != ProfileSupport::Unsupported &&
                                        entry.stage == ProfileStage::Printing &&
                                        entry.polarity != ProfilePolarity::Unsupported;

            if (visibleAsFilm) {
                if (!role_key_inserted(filmKeys, entry, "film", catalog)) {
                    return catalog;
                }
                catalog.filmProfiles.push_back(entry);
            }
            if (visibleAsPrint) {
                if (!role_key_inserted(printKeys, entry, "print", catalog)) {
                    return catalog;
                }
                catalog.printProfiles.push_back(entry);
            }
            if (!visibleAsFilm && !visibleAsPrint) {
                catalog.unavailableProfiles.emplace_back(entry.sourcePath + ": unsupported Phase 1A role metadata");
            }
        }

        sort_by_label(catalog.filmProfiles);
        sort_by_label(catalog.printProfiles);

        catalog.defaultFilmPresent = filmKeys.find(kDefaultFilmProfileKey) != filmKeys.end();
        catalog.defaultPrintPresent = printKeys.find(kDefaultPrintProfileKey) != printKeys.end();

        if (catalog.filmProfiles.empty()) {
            catalog.failure = "no visible filming profiles";
            return catalog;
        }
        if (catalog.printProfiles.empty()) {
            catalog.failure = "no visible printing profiles";
            return catalog;
        }
        if (!catalog.defaultFilmPresent) {
            catalog.failure = std::string("missing default film profile key: ") + kDefaultFilmProfileKey;
            return catalog;
        }
        if (!catalog.defaultPrintPresent) {
            catalog.failure = std::string("missing default print profile key: ") + kDefaultPrintProfileKey;
            return catalog;
        }

        catalog.valid = true;
        return catalog;
    }

} // namespace Spektrafilm
