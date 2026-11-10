#include "ScanRoute.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>


namespace Spektrafilm {

    namespace {
        constexpr std::array<ScanRouteMetadata, 4> kScanRouteMatrix{{{ScanRoute::NegativeDirectScan, ProfilePolarity::Negative, false, "NegativeDirectScan", "Negative direct scan"},
                                                                     {ScanRoute::NegativePrintScan, ProfilePolarity::Negative, true, "NegativePrintScan", "Negative print scan"},
                                                                     {ScanRoute::PositiveDirectScan, ProfilePolarity::Positive, false, "PositiveDirectScan", "Positive direct scan"},
                                                                     {ScanRoute::PositivePrintScan, ProfilePolarity::Positive, true, "PositivePrintScan", "Positive print scan"}}};
    } // namespace


    const ScanRouteMetadata& scan_route_metadata(ScanRoute route) {
        return kScanRouteMatrix.at(static_cast<std::size_t>(route));
    }

    const char* scan_route_key(ScanRoute route) {
        return scan_route_metadata(route).key;
    }

    const char* scan_route_label(ScanRoute route) {
        return scan_route_metadata(route).label;
    }


    bool scan_route_is_print(ScanRoute route) {
        return scan_route_metadata(route).printRoute;
    }


} // namespace Spektrafilm
