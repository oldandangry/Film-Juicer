#include <stddef.h>
#include "juicer_legacy_api.h"

#define FJ_EXPOSURE_SIZE(type, size, alignment) _Static_assert(sizeof(type) == size && _Alignof(type) == alignment, #type " size/alignment");
#define FJ_EXPOSURE_FIELD(type, field, offset) _Static_assert(offsetof(type, field) == offset, #type " " #field " offset");
#include "exposure_abi_facts.inc"
#undef FJ_EXPOSURE_SIZE
#undef FJ_EXPOSURE_FIELD

int fj_test_exposure_abi_c(const FjReferenceWhiteInput* reference, FjReferenceWhite* white, const FjSensitivityInput* sensitivity, FjSensitivity* out, FjSensitivityFailure* failure, FjErrorBuffer* error) {
    FjStatus (*reference_call)(const FjReferenceWhiteInput*, FjReferenceWhite*, FjErrorBuffer*) = fj_legacy_reconstruction_reference_white;
    FjStatus (*sensitivity_call)(const FjSensitivityInput*, FjSensitivity*, FjSensitivityFailure*, FjErrorBuffer*) = fj_legacy_exposure_sensitivity;
    FjStatus a = reference_call(reference, white, error);
    FjStatus b = sensitivity_call(sensitivity, out, failure, error);
    return a.category == FJ_STATUS_SUCCESS && b.category == FJ_STATUS_SUCCESS;
}
