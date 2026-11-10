#include "juicer_test_api.h"
#include <stddef.h>
#define FJ_ABI_TYPE(type, size, alignment) _Static_assert(sizeof(type) == size && _Alignof(type) == alignment, #type);
#define FJ_ABI_FIELD(type, field, offset) _Static_assert(offsetof(type, field) == offset, #field);
#include "spectral_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#define FJ_SIGNATURE(function, type) _Static_assert(_Generic(&(function), type: 1, default: 0), #function)
FJ_SIGNATURE(fj_test_hanatos_acquire, FjStatus (*)(const FjAssets*, FjSpectraLut**, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_arctic_acquire, FjStatus (*)(const FjAssets*, FjSpectraLut**, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_spectra_lut_view, FjStatus (*)(const FjSpectraLut*, FjSpectraLutView*, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_spectra_lut_release, FjStatus (*)(FjSpectraLut*, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_mallett_acquire, FjStatus (*)(const FjAssets*, FjMallettBasis**, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_mallett_view, FjStatus (*)(const FjMallettBasis*, FjFloatSpan*, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_mallett_release, FjStatus (*)(FjMallettBasis*, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_cmf_acquire, FjStatus (*)(const FjAssets*, FjCmf**, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_cmf_view, FjStatus (*)(const FjCmf*, FjFloatSpan*, FjErrorBuffer*));
FJ_SIGNATURE(fj_test_cmf_release, FjStatus (*)(FjCmf*, FjErrorBuffer*));

const size_t* fj_test_spectral_abi_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), _Alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#include "spectral_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}
