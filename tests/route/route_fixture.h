#pragma once

#include <string>

#include "ScanRoute.h"
#include "../ffi/juicer_test_api.h"

namespace JuicerTestFixture::Route {
    int scan_route_option_count();
    const char* scan_route_option_key(int index);
    const char* scan_route_option_label(int index);
    Spektrafilm::ScanRoute scan_route_from_key_or(const std::string& key, Spektrafilm::ScanRoute fallback);
    Spektrafilm::ScanRoute default_scan_route_for_polarity(Spektrafilm::ProfilePolarity polarity);
    bool resolve_scan_route(Spektrafilm::ProfilePolarity capturePolarity,
                            Spektrafilm::ScanRoute userRouteSelection,
                            Spektrafilm::ScanRoute& outRoute,
                            std::string& outDiagnostic);
} // namespace JuicerTestFixture::Route
