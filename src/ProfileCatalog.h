#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Spektrafilm {

    inline constexpr const char kDefaultFilmProfileKey[] = "kodak_portra_400";
    inline constexpr const char kDefaultPrintProfileKey[] = "kodak_portra_endura";

    enum class ProfileSupport : std::uint8_t {
        Film,
        Paper,
        Unsupported
    };

    enum class ProfileStage : std::uint8_t {
        Filming,
        Printing,
        Unsupported
    };

    enum class ProfilePolarity : std::uint8_t {
        Negative,
        Positive,
        Unsupported
    };

    struct ProfileCatalogEntry {
        std::string key;
        std::string label;
        std::string sourcePath;
        ProfileSupport support = ProfileSupport::Film;
        ProfileStage stage = ProfileStage::Filming;
        ProfilePolarity polarity = ProfilePolarity::Negative;
        bool supportDefaulted = false;
        bool stageDefaulted = false;
        bool polarityDefaulted = false;
        std::uint64_t sourceVersion = 0;
    };

    struct ProfileCatalog {
        std::vector<ProfileCatalogEntry> filmProfiles;
        std::vector<ProfileCatalogEntry> printProfiles;
        std::vector<std::string> ignoredResources;
        std::vector<std::string> unavailableProfiles;
        bool valid = false;
        bool defaultFilmPresent = false;
        bool defaultPrintPresent = false;
        std::string failure;
    };

    ProfileCatalog build_profile_catalog(const std::string& dataDir);

} // namespace Spektrafilm
