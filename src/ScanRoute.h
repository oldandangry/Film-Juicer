#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "ProfileCatalog.h"

namespace Spektrafilm {

    enum class ScanRoute : std::uint8_t {
        NegativeDirectScan = 0,
        NegativePrintScan = 1,
        PositiveDirectScan = 2,
        PositivePrintScan = 3
    };

    // clang-format off
struct ScanRouteMetadata {
    ScanRoute route = ScanRoute::NegativePrintScan;
    ProfilePolarity capturePolarity = ProfilePolarity::Negative;
    bool printRoute = true;
    const char* key = "NegativePrintScan";
    const char* label = "Negative print scan";
};
    // clang-format on

    inline constexpr ScanRoute kDefaultScanRoute = ScanRoute::NegativePrintScan;

    inline constexpr std::array<ScanRouteMetadata, 4> kScanRouteMatrix{{{ScanRoute::NegativeDirectScan, ProfilePolarity::Negative, false, "NegativeDirectScan", "Negative direct scan"},
                                                                        {ScanRoute::NegativePrintScan, ProfilePolarity::Negative, true, "NegativePrintScan", "Negative print scan"},
                                                                        {ScanRoute::PositiveDirectScan, ProfilePolarity::Positive, false, "PositiveDirectScan", "Positive direct scan"},
                                                                        {ScanRoute::PositivePrintScan, ProfilePolarity::Positive, true, "PositivePrintScan", "Positive print scan"}}};

    inline int scan_route_option_count() {
        return static_cast<int>(kScanRouteMatrix.size());
    }

    inline const ScanRouteMetadata& scan_route_metadata(ScanRoute route) {
        const std::uint8_t index = static_cast<std::uint8_t>(route);
        if (index < kScanRouteMatrix.size()) {
            return kScanRouteMatrix[index];
        }
        return kScanRouteMatrix[static_cast<std::uint8_t>(kDefaultScanRoute)];
    }

    inline const char* scan_route_key(ScanRoute route) {
        return scan_route_metadata(route).key;
    }

    inline const char* scan_route_label(ScanRoute route) {
        return scan_route_metadata(route).label;
    }

    inline const char* scan_route_option_key(int index) {
        return (index >= 0 && index < scan_route_option_count())
                   ? kScanRouteMatrix[static_cast<std::size_t>(index)].key
                   : "";
    }

    inline const char* scan_route_option_label(int index) {
        return (index >= 0 && index < scan_route_option_count())
                   ? kScanRouteMatrix[static_cast<std::size_t>(index)].label
                   : "";
    }

    inline bool scan_route_is_print(ScanRoute route) {
        return scan_route_metadata(route).printRoute;
    }

    inline ScanRoute scan_route_from_key_or(const std::string& key, ScanRoute fallback) {
        for (const ScanRouteMetadata& metadata : kScanRouteMatrix) {
            if (key == metadata.key) {
                return metadata.route;
            }
        }
        return fallback;
    }

    inline ScanRoute default_scan_route_for_polarity(ProfilePolarity polarity) {
        return (polarity == ProfilePolarity::Positive)
                   ? ScanRoute::PositiveDirectScan
                   : ScanRoute::NegativePrintScan;
    }

    inline ScanRoute resolve_scan_route(ProfilePolarity capturePolarity, ScanRoute userRouteSelection) {
        const bool printRoute = scan_route_is_print(userRouteSelection);
        if (capturePolarity == ProfilePolarity::Positive) {
            return printRoute ? ScanRoute::PositivePrintScan : ScanRoute::PositiveDirectScan;
        }
        return printRoute ? ScanRoute::NegativePrintScan : ScanRoute::NegativeDirectScan;
    }

} // namespace Spektrafilm
