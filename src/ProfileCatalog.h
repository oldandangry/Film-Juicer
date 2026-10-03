#pragma once

#include <cstdint>
#include <filesystem>
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
        std::filesystem::path sourcePath;
        ProfilePolarity polarity = ProfilePolarity::Negative;
    };

    struct ProfileCatalog {
        std::vector<ProfileCatalogEntry> filmProfiles;
        std::vector<ProfileCatalogEntry> printProfiles;
        bool valid = false;
        bool defaultFilmPresent = false;
        bool defaultPrintPresent = false;
        std::string failure;
    };

} // namespace Spektrafilm
