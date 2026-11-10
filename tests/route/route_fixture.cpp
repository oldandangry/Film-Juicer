#include "route_fixture.h"

#include <cstdint>
#include <string>

namespace JuicerTestFixture::Route {

    int scan_route_option_count() {
        return static_cast<int>(Spektrafilm::ScanRoute::PositivePrintScan) + 1;
    }

    const char* scan_route_option_key(int index) {
        return (index >= 0 && index < scan_route_option_count())
                   ? Spektrafilm::scan_route_key(static_cast<Spektrafilm::ScanRoute>(index))
                   : "";
    }

    const char* scan_route_option_label(int index) {
        return (index >= 0 && index < scan_route_option_count())
                   ? Spektrafilm::scan_route_label(static_cast<Spektrafilm::ScanRoute>(index))
                   : "";
    }

    Spektrafilm::ScanRoute scan_route_from_key_or(const std::string& key, Spektrafilm::ScanRoute fallback) {
        for (int index = 0; index < scan_route_option_count(); ++index) {
            const auto route = static_cast<Spektrafilm::ScanRoute>(index);
            if (key == Spektrafilm::scan_route_key(route)) {
                return route;
            }
        }
        return fallback;
    }

    Spektrafilm::ScanRoute default_scan_route_for_polarity(Spektrafilm::ProfilePolarity polarity) {
        return (polarity == Spektrafilm::ProfilePolarity::Positive)
                   ? Spektrafilm::ScanRoute::PositiveDirectScan
                   : Spektrafilm::ScanRoute::NegativePrintScan;
    }

    bool resolve_scan_route(
        Spektrafilm::ProfilePolarity capturePolarity,
        Spektrafilm::ScanRoute userRouteSelection,
        Spektrafilm::ScanRoute& outRoute,
        std::string& outDiagnostic) {
        std::uint8_t routeTag = 0;
        const std::uint32_t status = fj_test_resolve_route(
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
        outRoute = static_cast<Spektrafilm::ScanRoute>(routeTag);
        outDiagnostic.clear();
        return true;
    }

} // namespace JuicerTestFixture::Route
