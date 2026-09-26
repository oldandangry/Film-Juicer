#include <cstdint>
#include <string>

#include "ScanRoute.h"
#include "juicer_legacy_api.h"

namespace RouteFaultTest {

    int remainingCalls = 0;

    void fail_on_call(int call) {
        remainingCalls = call;
    }

} // namespace RouteFaultTest

namespace Spektrafilm {

    // This definition replaces the production adapter only in the OFX fixture.
    bool resolve_scan_route(
        ProfilePolarity capturePolarity,
        ScanRoute selection,
        ScanRoute& outRoute,
        std::string& outDiagnostic) {
        if (RouteFaultTest::remainingCalls > 0 &&
            --RouteFaultTest::remainingCalls == 0) {
            outDiagnostic = "RouteResolutionFailure status=fixture-injected";
            return false;
        }
        std::uint8_t route = 0;
        const std::uint32_t status = fj_legacy_resolve_route(
            static_cast<std::uint8_t>(capturePolarity),
            static_cast<std::uint8_t>(selection),
            &route);
        if (status != FJ_LEGACY_ROUTE_SUCCESS || route > 3) {
            outDiagnostic = "RouteResolutionFailure status=" + std::to_string(status);
            return false;
        }
        outRoute = static_cast<ScanRoute>(route);
        outDiagnostic.clear();
        return true;
    }

} // namespace Spektrafilm
