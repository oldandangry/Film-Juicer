#include "ScanRoute.h"

#include <cstdint>
#include <string>

#include "juicer_legacy_api.h"

namespace Spektrafilm {

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
