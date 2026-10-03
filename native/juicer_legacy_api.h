#pragma once

#include <stdint.h>

#include "juicer_cuda_api.h"

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

/* FJ_TEMP_BRIDGE: asset conversion; remove S4.E.
 * Handles own immutable Rust storage; only matching create/acquire values are
 * legal. Caller excludes destruction from operations and release from every
 * entry read/outstanding view. Concurrent reads require disjoint outputs.
 * All outputs/diagnostics are exclusive, aligned and mutually disjoint, and
 * disjoint from initialized inputs/owner storage. Valid outputs are cleared
 * before input/diagnostic checks. NULL required slots/owners are unsupported.
 * Text is length-delimited UTF-8; empty text is (NULL,0), embedded NUL survives.
 * Destroy/release consume once even with malformed diagnostics; NULL is a no-op
 * subject to diagnostic validation. No CUDA/OFX callbacks occur here.
 * Diagnostics: NULL allowed; length initialized to zero; capacity zero never
 * reads data. Otherwise nonnull backing, capacity <= PTRDIFF_MAX, trailing NUL.
 * Categories: success, unsupported input, preparation, allocation, internal;
 * api/native_code are zero. Diagnostic truncation never changes the status. */
typedef struct FjAssets FjAssets;
typedef struct FjCatalog FjCatalog;
#define FJ_PROFILE_ROLE_FILM UINT32_C(0)
#define FJ_PROFILE_ROLE_PRINT UINT32_C(1)
typedef struct FjCatalogCounts {
    size_t film_count;
    size_t print_count;
} FjCatalogCounts;
typedef struct FjCatalogEntryView {
    FjStringView key;
    FjStringView label;
    FjPathView source_path; /* Temporary: removed with native profile opens in C8 Slice 2. */
    uint32_t polarity;      /* FJ_POLARITY_NEGATIVE/POSITIVE, mapped explicitly. */
} FjCatalogEntryView;
FjStatus fj_legacy_assets_create(FjPathView resource_root, FjAssets** out_assets, FjErrorBuffer* error);
FjStatus fj_legacy_assets_destroy(FjAssets* assets, FjErrorBuffer* error);
FjStatus fj_legacy_catalog_acquire(const FjAssets* assets, FjCatalog** out_catalog, FjCatalogCounts* out_counts, FjErrorBuffer* error);
FjStatus fj_legacy_catalog_entry(const FjCatalog* catalog, uint32_t role, size_t index, FjCatalogEntryView* out_entry, FjErrorBuffer* error);
FjStatus fj_legacy_catalog_release(FjCatalog* catalog, FjErrorBuffer* error);

#ifdef __cplusplus
}
#endif
