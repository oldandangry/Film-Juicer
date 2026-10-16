#include "ScanRoute.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "juicer_legacy_api.h"

namespace Spektrafilm {

    namespace {
        constexpr std::array<ScanRouteMetadata, 4> kScanRouteMatrix{{{ScanRoute::NegativeDirectScan, ProfilePolarity::Negative, false, "NegativeDirectScan", "Negative direct scan"},
                                                                     {ScanRoute::NegativePrintScan, ProfilePolarity::Negative, true, "NegativePrintScan", "Negative print scan"},
                                                                     {ScanRoute::PositiveDirectScan, ProfilePolarity::Positive, false, "PositiveDirectScan", "Positive direct scan"},
                                                                     {ScanRoute::PositivePrintScan, ProfilePolarity::Positive, true, "PositivePrintScan", "Positive print scan"}}};
    } // namespace

    int scan_route_option_count() {
        return static_cast<int>(kScanRouteMatrix.size());
    }

    const ScanRouteMetadata& scan_route_metadata(ScanRoute route) {
        return kScanRouteMatrix.at(static_cast<std::size_t>(route));
    }

    const char* scan_route_key(ScanRoute route) {
        return scan_route_metadata(route).key;
    }

    const char* scan_route_label(ScanRoute route) {
        return scan_route_metadata(route).label;
    }

    const char* scan_route_option_key(int index) {
        return (index >= 0 && index < scan_route_option_count())
                   ? kScanRouteMatrix[static_cast<std::size_t>(index)].key
                   : "";
    }

    const char* scan_route_option_label(int index) {
        return (index >= 0 && index < scan_route_option_count())
                   ? kScanRouteMatrix[static_cast<std::size_t>(index)].label
                   : "";
    }

    bool scan_route_is_print(ScanRoute route) {
        return scan_route_metadata(route).printRoute;
    }

    ScanRoute scan_route_from_key_or(const std::string& key, ScanRoute fallback) {
        for (const ScanRouteMetadata& metadata : kScanRouteMatrix) {
            if (key == metadata.key) {
                return metadata.route;
            }
        }
        return fallback;
    }

    ScanRoute default_scan_route_for_polarity(ProfilePolarity polarity) {
        return (polarity == ProfilePolarity::Positive)
                   ? ScanRoute::PositiveDirectScan
                   : ScanRoute::NegativePrintScan;
    }

    bool resolve_scan_route(
        ProfilePolarity capturePolarity,
        ScanRoute userRouteSelection,
        ScanRoute& outRoute,
        std::string& outDiagnostic) {
        std::uint8_t routeTag = 0;
        const std::uint32_t status = fj_legacy_resolve_route(
            static_cast<std::uint8_t>(capturePolarity),
            static_cast<std::uint8_t>(userRouteSelection),
            &routeTag);
        if (status != FJ_LEGACY_ROUTE_SUCCESS || routeTag > 3) {
            const std::uint32_t failureStatus =
                status == FJ_LEGACY_ROUTE_SUCCESS
                    ? static_cast<std::uint32_t>(FJ_LEGACY_ROUTE_INTERNAL_PANIC)
                    : status;
            outDiagnostic = "RouteResolutionFailure status=" + std::to_string(failureStatus) +
                            " polarity=" + std::to_string(static_cast<std::uint8_t>(capturePolarity)) +
                            " selection=" + std::to_string(static_cast<std::uint8_t>(userRouteSelection));
            return false;
        }
        outRoute = static_cast<ScanRoute>(routeTag);
        outDiagnostic.clear();
        return true;
    }

} // namespace Spektrafilm
