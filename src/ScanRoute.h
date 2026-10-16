#pragma once

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

    struct ScanRouteMetadata {
        ScanRoute route = ScanRoute::NegativePrintScan;
        ProfilePolarity capturePolarity = ProfilePolarity::Negative;
        bool printRoute = true;
        const char* key = "NegativePrintScan";
        const char* label = "Negative print scan";
    };

    inline constexpr ScanRoute kDefaultScanRoute = ScanRoute::NegativePrintScan;

    int scan_route_option_count();
    const ScanRouteMetadata& scan_route_metadata(ScanRoute route);
    const char* scan_route_key(ScanRoute route);
    const char* scan_route_label(ScanRoute route);
    const char* scan_route_option_key(int index);
    const char* scan_route_option_label(int index);
    bool scan_route_is_print(ScanRoute route);
    ScanRoute scan_route_from_key_or(const std::string& key, ScanRoute fallback);
    ScanRoute default_scan_route_for_polarity(ProfilePolarity polarity);

    bool resolve_scan_route(
        ProfilePolarity capturePolarity,
        ScanRoute userRouteSelection,
        ScanRoute& outRoute,
        std::string& outDiagnostic);

} // namespace Spektrafilm
