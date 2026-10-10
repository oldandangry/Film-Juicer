#include <stddef.h>
#include <stdint.h>
#include "juicer_test_api.h"

#define FJ_MALLETT_SIZE(type, size, alignment) _Static_assert(sizeof(type) == size && _Alignof(type) == alignment, #type);
#define FJ_MALLETT_FIELD(type, field, offset) _Static_assert(offsetof(type, field) == offset, #type "." #field);
#include "mallett_exposure_abi_facts.inc"
#undef FJ_MALLETT_SIZE
#undef FJ_MALLETT_FIELD
_Static_assert(FJ_INPUT_DWG == 0 && FJ_INPUT_BT2020 == 1 && FJ_INPUT_ACES2065_1 == 2 && FJ_INPUT_SRGB_REC709 == 3, "input tags");

int fj_test_mallett_exposure_abi_c(const FjMallettMidgrayInput* focused,
                                   FjMallettMidgray* focused_out,
                                   float green,
                                   FjMidgrayNormalization* normalization,
                                   float ev,
                                   float* source,
                                   const FjMallettReferenceInput* reference,
                                   FjReferenceRaw* raw,
                                   FjErrorBuffer* error) {
    FjStatus (*midgray_fn)(const FjMallettMidgrayInput*, FjMallettMidgray*, FjErrorBuffer*) = fj_legacy_exposure_mallett_midgray;
    FjStatus (*tc_fn)(float, FjMidgrayNormalization*, FjErrorBuffer*) = fj_legacy_exposure_tc_midgray;
    FjStatus (*source_fn)(float, float*, FjErrorBuffer*) = fj_legacy_exposure_reference_source;
    FjStatus (*reference_fn)(const FjMallettReferenceInput*, FjReferenceRaw*, FjErrorBuffer*) = fj_legacy_exposure_mallett_reference_raw;
    FjStatus a = midgray_fn(focused, focused_out, error);
    FjStatus b = tc_fn(green, normalization, error);
    FjStatus c = source_fn(ev, source, error);
    FjStatus d = reference_fn(reference, raw, error);
    return a.category != FJ_STATUS_SUCCESS || b.category != FJ_STATUS_SUCCESS ||
           c.category != FJ_STATUS_SUCCESS || d.category != FJ_STATUS_SUCCESS ||
           a.api != FJ_API_NONE || b.api != FJ_API_NONE || c.api != FJ_API_NONE || d.api != FJ_API_NONE ||
           a.native_code != 0 || b.native_code != 0 || c.native_code != 0 || d.native_code != 0;
}

int fj_test_spectrum_abi_c(const FjHanatosSpectrumInput* hanatos, FjSpectrumFixture* hanatos_out, const FjTablesSpectrumInput* tables, FjSpectrumFixture* tables_out, const FjMallettRawInput* mallett, float bgr[3], FjErrorBuffer* error) {
    FjStatus (*hanatos_fn)(const FjHanatosSpectrumInput*, FjSpectrumFixture*, FjErrorBuffer*) = fj_test_exposure_hanatos_spectrum;
    FjStatus (*tables_fn)(const FjTablesSpectrumInput*, FjSpectrumFixture*, FjErrorBuffer*) = fj_test_exposure_tables_spectrum;
    FjStatus (*mallett_fn)(const FjMallettRawInput*, float*, FjErrorBuffer*) = fj_test_exposure_mallett_raw;
    FjStatus a = hanatos_fn(hanatos, hanatos_out, error), b = tables_fn(tables, tables_out, error), c = mallett_fn(mallett, bgr, error);
    return a.category != FJ_STATUS_SUCCESS || b.category != FJ_STATUS_SUCCESS || c.category != FJ_STATUS_SUCCESS ||
           a.api != FJ_API_NONE || b.api != FJ_API_NONE || c.api != FJ_API_NONE || a.native_code != 0 || b.native_code != 0 || c.native_code != 0;
}
