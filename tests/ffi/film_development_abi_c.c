#include <stddef.h>
#include "juicer_legacy_api.h"

typedef FjStatus (*Acquire)(const FjFilmProfile*, uint32_t, float, uint32_t, FjFilmDevelopmentOwner**, FjErrorBuffer*);
typedef FjStatus (*View)(const FjFilmDevelopmentOwner*, FjFilmDevelopmentView*, FjErrorBuffer*);
typedef FjStatus (*Release)(FjFilmDevelopmentOwner*, FjErrorBuffer*);
_Static_assert(_Generic(&fj_legacy_film_development_acquire, Acquire: 1, default: 0), "C11 acquire signature");
_Static_assert(_Generic(&fj_legacy_film_development_view, View: 1, default: 0), "C11 view signature");
_Static_assert(_Generic(&fj_legacy_film_development_release, Release: 1, default: 0), "C11 release signature");

const size_t* fj_test_film_development_abi_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_DEVELOPMENT_TYPE(type) sizeof(type), _Alignof(type),
#define FJ_DEVELOPMENT_FIELD(type, field) offsetof(type, field),
#define FJ_DEVELOPMENT_EXTENT(field, element) sizeof(((FjFilmDevelopmentView*)0)->field) / sizeof(element),
#define FJ_DEVELOPMENT_VALUE(value) value,
#include "film_development_abi_facts.inc"
#undef FJ_DEVELOPMENT_TYPE
#undef FJ_DEVELOPMENT_FIELD
#undef FJ_DEVELOPMENT_EXTENT
#undef FJ_DEVELOPMENT_VALUE
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}
