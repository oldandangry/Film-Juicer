#include <stddef.h>

#include "juicer_legacy_api.h"

_Static_assert(sizeof(FjCatalogCounts) == 16 && _Alignof(FjCatalogCounts) == 8, "catalog counts layout");
_Static_assert(offsetof(FjCatalogCounts, film_count) == 0 && offsetof(FjCatalogCounts, print_count) == 8, "catalog count offsets");
_Static_assert(sizeof(FjCatalogEntryView) == 40 && _Alignof(FjCatalogEntryView) == 8, "catalog entry layout");
_Static_assert(offsetof(FjCatalogEntryView, key) == 0 && offsetof(FjCatalogEntryView, label) == 16 &&
                   offsetof(FjCatalogEntryView, polarity) == 32,
               "catalog entry offsets");
_Static_assert(FJ_PROFILE_ROLE_FILM == 0 && FJ_PROFILE_ROLE_PRINT == 1 && FJ_POLARITY_NEGATIVE == 0 && FJ_POLARITY_POSITIVE == 1, "catalog tags");
_Static_assert(_Generic(&fj_legacy_assets_create, FjStatus (*)(FjPathView, FjAssets**, FjErrorBuffer*): 1, default: 0), "assets create signature");
_Static_assert(_Generic(&fj_legacy_assets_destroy, FjStatus (*)(FjAssets*, FjErrorBuffer*): 1, default: 0), "assets destroy signature");
_Static_assert(_Generic(&fj_legacy_catalog_acquire, FjStatus (*)(const FjAssets*, FjCatalog**, FjCatalogCounts*, FjErrorBuffer*): 1, default: 0), "catalog acquire signature");
_Static_assert(_Generic(&fj_legacy_catalog_entry, FjStatus (*)(const FjCatalog*, uint32_t, size_t, FjCatalogEntryView*, FjErrorBuffer*): 1, default: 0), "catalog entry signature");
_Static_assert(_Generic(&fj_legacy_catalog_release, FjStatus (*)(FjCatalog*, FjErrorBuffer*): 1, default: 0), "catalog release signature");
