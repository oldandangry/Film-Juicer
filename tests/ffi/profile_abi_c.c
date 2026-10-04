#include <stddef.h>
#include <stdint.h>

#include "juicer_test_api.h"

#define FJ_ABI_TYPE(type, size, alignment)                \
    _Static_assert(sizeof(type) == size, "size: " #type); \
    _Static_assert(_Alignof(type) == alignment, "alignment: " #type);
#define FJ_ABI_FIELD(type, field, offset) \
    _Static_assert(offsetof(type, field) == offset, "offset: " #type "." #field);
#define FJ_ABI_VALUE(tag, value) _Static_assert(tag == value, "tag: " #tag);
#include "profile_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE

_Static_assert(_Generic(&fj_test_film_profile_acquire, FjStatus (*)(FjStringView, FjStringView, FjFilmProfile**, FjErrorBuffer*): 1, default: 0), "acquire signature");
_Static_assert(_Generic(&fj_test_film_profile_view, FjStatus (*)(const FjFilmProfile*, FjFilmFixtureView*, FjErrorBuffer*): 1, default: 0), "view signature");
_Static_assert(_Generic(&fj_test_film_profile_release, FjStatus (*)(FjFilmProfile*, FjErrorBuffer*): 1, default: 0), "release signature");
_Static_assert(_Generic(&fj_test_profile_abi_facts, const size_t* (*)(size_t*): 1, default: 0), "facts signature");

const size_t* fj_test_profile_abi_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), _Alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#define FJ_ABI_VALUE(tag, value) tag,
#include "profile_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}
