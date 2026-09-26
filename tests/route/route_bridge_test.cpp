#include <array>
#include <cstdio>
#include <cstdint>
#include <string>
#include <type_traits>

#include "ScanRoute.h"
#include "juicer_legacy_api.h"

namespace {

    using ResolveSignature = std::uint32_t (*)(std::uint8_t, std::uint8_t, std::uint8_t*);
    static_assert(std::is_same_v<decltype(&fj_legacy_resolve_route), ResolveSignature>);
    static_assert(sizeof(Spektrafilm::ScanRoute) == 1);
    static_assert(sizeof(Spektrafilm::ProfilePolarity) == 1);
    static_assert(static_cast<std::uint8_t>(Spektrafilm::ProfilePolarity::Negative) ==
                  FJ_LEGACY_POLARITY_NEGATIVE);
    static_assert(static_cast<std::uint8_t>(Spektrafilm::ProfilePolarity::Positive) ==
                  FJ_LEGACY_POLARITY_POSITIVE);
    static_assert(static_cast<std::uint8_t>(Spektrafilm::ProfilePolarity::Unsupported) ==
                  FJ_LEGACY_POLARITY_UNSUPPORTED);
    static_assert(static_cast<std::uint8_t>(Spektrafilm::ScanRoute::NegativeDirectScan) ==
                  FJ_LEGACY_ROUTE_NEGATIVE_DIRECT);
    static_assert(static_cast<std::uint8_t>(Spektrafilm::ScanRoute::NegativePrintScan) ==
                  FJ_LEGACY_ROUTE_NEGATIVE_PRINT);
    static_assert(static_cast<std::uint8_t>(Spektrafilm::ScanRoute::PositiveDirectScan) ==
                  FJ_LEGACY_ROUTE_POSITIVE_DIRECT);
    static_assert(static_cast<std::uint8_t>(Spektrafilm::ScanRoute::PositivePrintScan) ==
                  FJ_LEGACY_ROUTE_POSITIVE_PRINT);

    bool verify(bool condition, const char* message) {
        if (!condition) {
            std::fprintf(stderr, "%s\n", message);
        }
        return condition;
    }

} // namespace

int main() {
    try {
        constexpr std::array<std::array<std::uint8_t, 4>, 3> expected{{
            {{0, 1, 0, 1}},
            {{2, 3, 2, 3}},
            {{0, 1, 0, 1}},
        }};
        for (std::uint8_t polarity = 0; polarity < expected.size(); ++polarity) {
            for (std::uint8_t selection = 0; selection < expected[0].size(); ++selection) {
                std::uint8_t route = 99;
                if (!verify(
                        fj_legacy_resolve_route(polarity, selection, &route) ==
                                FJ_LEGACY_ROUTE_SUCCESS &&
                            route == expected[polarity][selection],
                        "route ABI table mismatch")) {
                    return 1;
                }
                Spektrafilm::ScanRoute nativeRoute{};
                std::string diagnostic;
                if (!verify(
                        Spektrafilm::resolve_scan_route(
                            static_cast<Spektrafilm::ProfilePolarity>(polarity),
                            static_cast<Spektrafilm::ScanRoute>(selection),
                            nativeRoute,
                            diagnostic) &&
                            static_cast<std::uint8_t>(nativeRoute) == route &&
                            diagnostic.empty(),
                        "native route adapter mismatch")) {
                    return 1;
                }
            }
        }

        std::uint8_t route = 99;
        if (!verify(
                fj_legacy_resolve_route(3, 0, &route) == FJ_LEGACY_ROUTE_INVALID_POLARITY &&
                    fj_legacy_resolve_route(0, 4, &route) == FJ_LEGACY_ROUTE_INVALID_SELECTION &&
                    fj_legacy_resolve_route(0, 0, nullptr) == FJ_LEGACY_ROUTE_NULL_OUTPUT &&
                    route == 99,
                "route ABI failure contract mismatch")) {
            return 1;
        }

        Spektrafilm::ScanRoute nativeRoute =
            Spektrafilm::ScanRoute::PositivePrintScan;
        std::string diagnostic;
        if (!verify(
                !Spektrafilm::resolve_scan_route(
                    static_cast<Spektrafilm::ProfilePolarity>(3),
                    Spektrafilm::ScanRoute::NegativeDirectScan,
                    nativeRoute,
                    diagnostic) &&
                    nativeRoute == Spektrafilm::ScanRoute::PositivePrintScan &&
                    diagnostic ==
                        "RouteResolutionFailure status=1 polarity=3 selection=0",
                "native invalid-polarity mapping mismatch")) {
            return 1;
        }
        diagnostic.clear();
        if (!verify(
                !Spektrafilm::resolve_scan_route(
                    Spektrafilm::ProfilePolarity::Negative,
                    static_cast<Spektrafilm::ScanRoute>(4),
                    nativeRoute,
                    diagnostic) &&
                    nativeRoute == Spektrafilm::ScanRoute::PositivePrintScan &&
                    diagnostic ==
                        "RouteResolutionFailure status=2 polarity=0 selection=4",
                "native invalid-selection mapping mismatch")) {
            return 1;
        }

        const auto negativeDefault = Spektrafilm::default_scan_route_for_polarity(
            Spektrafilm::ProfilePolarity::Negative);
        const auto positiveDefault = Spektrafilm::default_scan_route_for_polarity(
            Spektrafilm::ProfilePolarity::Positive);
        if (!verify(
                negativeDefault == Spektrafilm::ScanRoute::NegativePrintScan &&
                    positiveDefault == Spektrafilm::ScanRoute::PositiveDirectScan &&
                    Spektrafilm::scan_route_from_key_or("", negativeDefault) == negativeDefault &&
                    Spektrafilm::scan_route_from_key_or("unknown", positiveDefault) == positiveDefault,
                "route default and key fallback mismatch")) {
            return 1;
        }
        return 0;
    } catch (...) {
        std::fprintf(stderr, "route bridge test threw unexpectedly\n");
        return 1;
    }
}
