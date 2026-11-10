#pragma once

#include <cstdint>
#include <string>


namespace Spektrafilm {

    enum class ProfilePolarity : std::uint8_t {
        Negative,
        Positive,
        Unsupported
    };


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

    const ScanRouteMetadata& scan_route_metadata(ScanRoute route);
    const char* scan_route_key(ScanRoute route);
    const char* scan_route_label(ScanRoute route);
    bool scan_route_is_print(ScanRoute route);


} // namespace Spektrafilm
