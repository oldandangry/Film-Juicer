#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FJ_LEGACY_ROUTE_SUCCESS UINT32_C(0)
#define FJ_LEGACY_ROUTE_INVALID_POLARITY UINT32_C(1)
#define FJ_LEGACY_ROUTE_INVALID_SELECTION UINT32_C(2)
#define FJ_LEGACY_ROUTE_NULL_OUTPUT UINT32_C(3)
#define FJ_LEGACY_ROUTE_INTERNAL_PANIC UINT32_C(4)

#define FJ_LEGACY_POLARITY_NEGATIVE UINT8_C(0)
#define FJ_LEGACY_POLARITY_POSITIVE UINT8_C(1)
#define FJ_LEGACY_POLARITY_UNSUPPORTED UINT8_C(2)
#define FJ_LEGACY_ROUTE_NEGATIVE_DIRECT UINT8_C(0)
#define FJ_LEGACY_ROUTE_NEGATIVE_PRINT UINT8_C(1)
#define FJ_LEGACY_ROUTE_POSITIVE_DIRECT UINT8_C(2)
#define FJ_LEGACY_ROUTE_POSITIVE_PRINT UINT8_C(3)

// FJ_TEMP_BRIDGE: route resolution; remove S6.D.
// A nonnull out_route must point to one writable byte; failures leave it unchanged.
uint32_t fj_legacy_resolve_route(uint8_t capture_polarity, uint8_t selected_route, uint8_t* out_route);

#ifdef __cplusplus
}
#endif
