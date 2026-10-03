#include <stddef.h>
#include "juicer_legacy_api.h"
#define FJ_ABI_TYPE(type, size, alignment) _Static_assert(sizeof(type) == size && _Alignof(type) == alignment, #type);
#define FJ_ABI_FIELD(type, field, offset) _Static_assert(offsetof(type, field) == offset, #field);
#define FJ_ABI_VALUE(tag, value) _Static_assert(tag == value, #tag);
#include "production_profile_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
const size_t* fj_test_production_profile_abi_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), _Alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#define FJ_ABI_VALUE(tag, value) tag,
#include "production_profile_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}

#define FJ_SIGNATURE(function, type) _Static_assert(_Generic(&(function), type: 1, default: 0), #function)
FJ_SIGNATURE(fj_legacy_film_profile_acquire, FjStatus (*)(const FjAssets*, FjStringView, FjFilmProfile**, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_print_profile_acquire, FjStatus (*)(const FjAssets*, FjStringView, FjPrintProfile**, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_film_profile_view, FjStatus (*)(const FjFilmProfile*, FjFilmProfileView*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_print_profile_view, FjStatus (*)(const FjPrintProfile*, FjPrintProfileView*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_print_profile_sample_density, FjStatus (*)(const FjPrintProfile*, double, FjPrintDensityCurves**, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_print_density_view, FjStatus (*)(const FjPrintDensityCurves*, FjPrintDensityView*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_film_profile_release, FjStatus (*)(FjFilmProfile*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_print_profile_release, FjStatus (*)(FjPrintProfile*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_print_density_release, FjStatus (*)(FjPrintDensityCurves*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_assets_release_cached_payloads, FjStatus (*)(const FjAssets*, FjErrorBuffer*));
#undef FJ_SIGNATURE
