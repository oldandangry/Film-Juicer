#include <stddef.h>
#include "juicer_legacy_api.h"

#define FJ_HULL_SIZE(type, size, alignment) _Static_assert(sizeof(type) == size && _Alignof(type) == alignment, #type " layout");
#define FJ_HULL_FIELD(type, field, offset) _Static_assert(offsetof(type, field) == offset, #type " " #field);
#include "input_hull_abi_facts.inc"
#undef FJ_HULL_SIZE
#undef FJ_HULL_FIELD

FjStatus fj_test_input_hull_abi_c(const FjInputHullSource* source, FjInputHull* out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjInputHullSource*, FjInputHull*, FjErrorBuffer*) = fj_legacy_gamut_input_hull;
    return call(source, out, error);
}
