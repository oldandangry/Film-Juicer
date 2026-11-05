#include <array>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ScanRoute.h"
#include "juicer_legacy_api.h"
#include "../ffi/case_results.h"

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

    bool verify_metadata() {
        using Spektrafilm::ProfilePolarity;
        using Spektrafilm::ScanRoute;
        constexpr std::array<Spektrafilm::ScanRouteMetadata, 4> expected{{{ScanRoute::NegativeDirectScan, ProfilePolarity::Negative, false, "NegativeDirectScan", "Negative direct scan"},
                                                                          {ScanRoute::NegativePrintScan, ProfilePolarity::Negative, true, "NegativePrintScan", "Negative print scan"},
                                                                          {ScanRoute::PositiveDirectScan, ProfilePolarity::Positive, false, "PositiveDirectScan", "Positive direct scan"},
                                                                          {ScanRoute::PositivePrintScan, ProfilePolarity::Positive, true, "PositivePrintScan", "Positive print scan"}}};
        if (!verify(Spektrafilm::scan_route_option_count() == static_cast<int>(expected.size()), "route option count mismatch")) {
            return false;
        }
        for (int index = 0; index < static_cast<int>(expected.size()); ++index) {
            const auto& row = expected[static_cast<std::size_t>(index)];
            JuicerTests::CaseEvent event("route_bridge.metadata", row.key);
            const auto& actual = Spektrafilm::scan_route_metadata(row.route);
            const auto fallback = row.route == ScanRoute::NegativeDirectScan ? ScanRoute::PositivePrintScan : ScanRoute::NegativeDirectScan;
            if (!verify(
                    actual.route == row.route && actual.capturePolarity == row.capturePolarity &&
                        actual.printRoute == row.printRoute && std::string(actual.key) == row.key &&
                        std::string(actual.label) == row.label && Spektrafilm::scan_route_is_print(row.route) == row.printRoute &&
                        std::string(Spektrafilm::scan_route_option_key(index)) == row.key &&
                        std::string(Spektrafilm::scan_route_option_label(index)) == row.label &&
                        Spektrafilm::scan_route_from_key_or(row.key, fallback) == row.route &&
                        Spektrafilm::scan_route_from_key_or("", row.route) == row.route &&
                        Spektrafilm::scan_route_from_key_or("unknown", row.route) == row.route,
                    "route metadata/key/index/fallback mismatch")) {
                return false;
            }
            if (!verify(
                    &actual == &Spektrafilm::scan_route_metadata(row.route) &&
                        actual.key == Spektrafilm::scan_route_key(row.route) &&
                        actual.label == Spektrafilm::scan_route_label(row.route),
                    "route metadata reference lifetime mismatch")) {
                return false;
            }
            event.passed();
        }
        for (const int index : {std::numeric_limits<int>::min(), -1, 4, std::numeric_limits<int>::max()}) {
            JuicerTests::CaseEvent event("route_bridge.invalid_option_index", std::to_string(index));
            if (!verify(std::string(Spektrafilm::scan_route_option_key(index)).empty() &&
                            std::string(Spektrafilm::scan_route_option_label(index)).empty(),
                        "invalid route option must be empty")) {
                return false;
            }
            event.passed();
        }
        for (const auto route : {static_cast<ScanRoute>(4), static_cast<ScanRoute>(255)}) {
            const auto route_id = std::to_string(static_cast<unsigned>(route));
            JuicerTests::CaseEvent event("route_bridge.invalid_metadata", route_id);
            for (int operation = 0; operation < 4; ++operation) {
                bool rejected = false;
                try {
                    switch (operation) {
                        case 0:
                            (void)Spektrafilm::scan_route_metadata(route);
                            break;
                        case 1:
                            (void)Spektrafilm::scan_route_key(route);
                            break;
                        case 2:
                            (void)Spektrafilm::scan_route_label(route);
                            break;
                        default:
                            (void)Spektrafilm::scan_route_is_print(route);
                            break;
                    }
                } catch (const std::out_of_range&) {
                    rejected = true;
                }
                if (!verify(rejected, "invalid route metadata must throw out_of_range")) {
                    return false;
                }
            }
            event.passed();
        }
        JuicerTests::CaseEvent default_event("route_bridge", "default_metadata");
        const Spektrafilm::ScanRouteMetadata defaults;
        const bool defaults_valid = verify(
            defaults.route == ScanRoute::NegativePrintScan && defaults.capturePolarity == ProfilePolarity::Negative &&
                defaults.printRoute && std::string(defaults.key) == "NegativePrintScan" &&
                std::string(defaults.label) == "Negative print scan" && Spektrafilm::kDefaultScanRoute == defaults.route &&
                Spektrafilm::default_scan_route_for_polarity(ProfilePolarity::Unsupported) == defaults.route &&
                Spektrafilm::default_scan_route_for_polarity(static_cast<ProfilePolarity>(255)) == defaults.route,
            "route default metadata/polarity fallback mismatch");
        default_event.complete(defaults_valid);
        return defaults_valid;
    }

} // namespace

int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            for (const char* id : {"NegativeDirectScan", "NegativePrintScan", "PositiveDirectScan", "PositivePrintScan"})
                JuicerTests::list_case("route_bridge.metadata", id);
            for (int index : {std::numeric_limits<int>::min(), -1, 4, std::numeric_limits<int>::max()})
                JuicerTests::list_case("route_bridge.invalid_option_index", std::to_string(index));
            for (unsigned route : {4u, 255u})
                JuicerTests::list_case("route_bridge.invalid_metadata", std::to_string(route));
            JuicerTests::list_case("route_bridge", "default_metadata");
            for (const char* id : {"resolve.negative.0", "resolve.negative.1", "resolve.negative.2", "resolve.negative.3", "resolve.positive.0", "resolve.positive.1", "resolve.positive.2", "resolve.positive.3", "resolve.unsupported.0", "resolve.unsupported.1", "resolve.unsupported.2", "resolve.unsupported.3", "abi.failures", "native.invalid-polarity", "native.invalid-selection", "defaults"})
                JuicerTests::list_case("route_bridge", id);
            return 0;
        }
        if (!verify_metadata()) {
            return 1;
        }
        constexpr std::array<std::array<std::uint8_t, 4>, 3> expected{{
            {{0, 1, 0, 1}},
            {{2, 3, 2, 3}},
            {{0, 1, 0, 1}},
        }};
        for (std::uint8_t polarity = 0; polarity < expected.size(); ++polarity) {
            for (std::uint8_t selection = 0; selection < expected[0].size(); ++selection) {
                std::uint8_t route = 99;
                JuicerTests::CaseEvent event("route_bridge", std::string("resolve.") + (polarity == 0 ? "negative." : polarity == 1 ? "positive."
                                                                                                                                    : "unsupported.") +
                                                                 std::to_string(selection));
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
                event.passed();
            }
        }

        {
            JuicerTests::CaseEvent event("route_bridge", "abi.failures");
            std::uint8_t route = 99;
            if (!verify(
                    fj_legacy_resolve_route(3, 0, &route) == FJ_LEGACY_ROUTE_INVALID_POLARITY &&
                        fj_legacy_resolve_route(0, 4, &route) == FJ_LEGACY_ROUTE_INVALID_SELECTION &&
                        fj_legacy_resolve_route(0, 0, nullptr) == FJ_LEGACY_ROUTE_NULL_OUTPUT &&
                        route == 99,
                    "route ABI failure contract mismatch")) {
                return 1;
            }
            event.passed();
        }

        Spektrafilm::ScanRoute nativeRoute = Spektrafilm::ScanRoute::PositivePrintScan;
        std::string diagnostic;
        {
            JuicerTests::CaseEvent event("route_bridge", "native.invalid-polarity");
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
            event.passed();
        }
        {
            JuicerTests::CaseEvent event("route_bridge", "native.invalid-selection");
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
            event.passed();
        }

        {
            JuicerTests::CaseEvent event("route_bridge", "defaults");
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
            event.passed();
            case_log.finish();
            return 0;
        }
    } catch (...) {
        std::fprintf(stderr, "route bridge test threw unexpectedly\n");
        return 1;
    }
}
