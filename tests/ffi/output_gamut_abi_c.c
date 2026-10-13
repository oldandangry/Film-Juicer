#include <stddef.h>
#include "juicer_legacy_api.h"

#define FJ_GAMUT_SIZE(type, size, alignment) _Static_assert(sizeof(type) == size && _Alignof(type) == alignment, #type " layout");
#define FJ_GAMUT_FIELD(type, field, offset) _Static_assert(offsetof(type, field) == offset, #type " " #field);
#include "output_gamut_abi_facts.inc"
#undef FJ_GAMUT_SIZE
#undef FJ_GAMUT_FIELD

FjStatus fj_test_output_transform_abi_c(const FjOutputGamutSource* source, FjOutputGamutTransform* out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjOutputGamutSource*, FjOutputGamutTransform*, FjErrorBuffer*) = fj_legacy_gamut_output_transform;
    return call(source, out, error);
}
FjStatus fj_test_output_boundary_abi_c(const FjOutputBoundarySource* source, float* cmax, size_t count, FjOutputBoundaryIdentity* identity, FjOutputBoundaryFailure* failure, FjErrorBuffer* error) {
    FjStatus (*call)(const FjOutputBoundarySource*, float*, size_t, FjOutputBoundaryIdentity*, FjOutputBoundaryFailure*, FjErrorBuffer*) = fj_legacy_gamut_output_boundary;
    return call(source, cmax, count, identity, failure, error);
}
/* Actual C11 calls separately exercise null record and present zero capacity. */
FjStatus fj_test_output_transform_status_c(const FjOutputGamutSource* source, FjOutputGamutTransform* out, unsigned zero_capacity, size_t* length) {
    FjErrorBuffer error;
    error.capacity = 0;
    error.length = 73;
    FjStatus status = fj_legacy_gamut_output_transform(source, out, zero_capacity ? &error : NULL);
    *length = zero_capacity ? error.length : 0;
    return status;
}
FjStatus fj_test_output_boundary_status_c(const FjOutputBoundarySource* source, float* cmax, size_t count, FjOutputBoundaryIdentity* identity, FjOutputBoundaryFailure* failure, unsigned zero_capacity, size_t* length) {
    FjErrorBuffer error;
    error.capacity = 0;
    error.length = 73;
    FjStatus status = fj_legacy_gamut_output_boundary(source, cmax, count, identity, failure, zero_capacity ? &error : NULL);
    *length = zero_capacity ? error.length : 0;
    return status;
}
