#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "juicer_legacy_api.h"
#include "juicer_test_api.h"

_Static_assert(_Alignof(float) == 4, "color float alignment");
_Static_assert(sizeof(float) == 4, "color requires 32-bit float");
_Static_assert(sizeof(float[3]) == 12 && sizeof(float[9]) == 36, "color fixed-array extents");
_Static_assert(sizeof(FjStatus) == 12, "status ABI");
_Static_assert(offsetof(FjStatus, category) == 0, "category ABI");
_Static_assert(offsetof(FjStatus, api) == 4, "api ABI");
_Static_assert(offsetof(FjStatus, native_code) == 8, "native code ABI");

static int has_status(FjStatus status, uint32_t category) {
    return status.category == category && status.api == FJ_API_NONE && status.native_code == 0;
}

typedef FjStatus (*MatrixOperation)(const float*, const float*, float*);
typedef FjStatus (*ScalarOperation)(const float*, const float*, const float*, float*);

static int color_abi_c(const MatrixOperation matrices[2], const ScalarOperation adapters[2], uint32_t matrix_tag) {
    const float white[3] = {0.95045593f, 1.0f, 1.08905775f};
    const float xyz[3] = {2.0f, -0.25f, 0.5f};
    const uint32_t special_bits[3] = {UINT32_C(0x7f800000), UINT32_C(0xff800000), UINT32_C(0x7fc00000)};
    float output[11];
    int failures = 0;
    for (int boundary = 0; boundary < 2; ++boundary) {
        FjStatus (*matrix)(const float*, const float*, float*) = matrices[boundary];
        FjStatus (*adapt)(const float*, const float*, const float*, float*) = adapters[boundary];
        for (int operation = 1; operation <= 2; ++operation) {
            const int extent = operation == 1 ? 9 : 3;
            const int pointers = operation == 1 ? 3 : 4;
            for (int null_slot = 0; null_slot < pointers; ++null_slot) {
                for (int i = 0; i < 11; ++i) {
                    output[i] = 42.0f;
                }
                const FjStatus result = operation == 1 ? matrix(null_slot == 0 ? NULL : white, null_slot == 1 ? NULL : white, null_slot == 2 ? NULL : output + 1)
                                                       : adapt(null_slot == 0 ? NULL : xyz, null_slot == 1 ? NULL : white, null_slot == 2 ? NULL : white, null_slot == 3 ? NULL : output + 1);
                failures += !has_status(result, FJ_STATUS_UNSUPPORTED_INPUT);
                for (int i = 0; i < 11; ++i) {
                    const float expected = null_slot != pointers - 1 && i > 0 && i <= extent ? 0.0f : 42.0f;
                    failures += output[i] != expected;
                    if (expected == 0.0f) {
                        const float zero = 0.0f;
                        failures += memcmp(&output[i], &zero, sizeof(float)) != 0;
                    }
                }
            }
            for (int fault = 1; boundary == 0 && fault <= 2; ++fault) {
                for (int i = 0; i < 11; ++i) {
                    output[i] = 42.0f;
                }
                failures += !has_status(fj_test_color_arm_fault(matrix_tag + (uint32_t)operation - 1, 1, (uint32_t)fault), FJ_STATUS_SUCCESS);
                const FjStatus result = operation == 1 ? matrix(white, white, output + 1) : adapt(xyz, white, white, output + 1);
                failures += !has_status(result, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                for (int i = 0; i < 11; ++i) {
                    failures += output[i] != (i > 0 && i <= extent ? 0.0f : 42.0f);
                    if (i > 0 && i <= extent) {
                        const float zero = 0.0f;
                        failures += memcmp(&output[i], &zero, sizeof(float)) != 0;
                    }
                }
                failures += !has_status(operation == 1 ? matrix(white, white, output + 1) : adapt(xyz, white, white, output + 1), FJ_STATUS_SUCCESS);
                failures += !has_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
            }
            for (int special = 0; special < 3; ++special) {
                float nonfinite[3];
                memcpy(nonfinite, white, sizeof(nonfinite));
                memcpy(&nonfinite[special], &special_bits[special], sizeof(float));
                failures += !has_status(operation == 1 ? matrix(nonfinite, white, output + 1) : adapt(nonfinite, white, white, output + 1), FJ_STATUS_SUCCESS);
                failures += !has_status(operation == 1 ? matrix(white, nonfinite, output + 1) : adapt(xyz, nonfinite, white, output + 1), FJ_STATUS_SUCCESS);
                if (operation == 2) {
                    failures += !has_status(adapt(xyz, white, nonfinite, output + 1), FJ_STATUS_SUCCESS);
                }
                failures += output[0] != 42.0f || output[extent + 1] != 42.0f;
            }
            if (operation == 2) {
                float separate[3][3];
                float reference[3];
                for (int i = 0; i < 3; ++i) {
                    memcpy(separate[i], white, sizeof(white));
                }
                failures += !has_status(adapt(separate[0], separate[1], separate[2], reference), FJ_STATUS_SUCCESS);
                const float* aliases[4][3] = {{white, white, separate[0]}, {separate[0], white, white}, {white, separate[0], white}, {white, white, white}};
                for (int alias = 0; alias < 4; ++alias) {
                    failures += !has_status(adapt(aliases[alias][0], aliases[alias][1], aliases[alias][2], output + 1), FJ_STATUS_SUCCESS);
                    failures += memcmp(reference, output + 1, sizeof(reference)) != 0;
                    failures += output[0] != 42.0f || output[4] != 42.0f;
                }
            }
            for (int i = 0; i < 11; ++i) {
                output[i] = 42.0f;
            }
            failures += !has_status(operation == 1 ? matrix(white, white, output + 1) : adapt(white, white, white, output + 1), FJ_STATUS_SUCCESS);
            for (int i = 0; i < 11; ++i) {
                if (i == 0 || i > extent) {
                    failures += output[i] != 42.0f;
                } else {
                    failures += output[i] == 42.0f;
                }
            }
        }
    }
    return failures;
}

int fj_test_color_abi_c(void) {
    const MatrixOperation matrices[2] = {fj_legacy_cat16_matrix, fj_test_cat16_matrix};
    const ScalarOperation adapters[2] = {fj_legacy_adapt_cat16, fj_test_adapt_cat16};
    return color_abi_c(matrices, adapters, FJ_TEST_CAT16_MATRIX);
}

int fj_test_cat02_abi_c(void) {
    const MatrixOperation matrices[2] = {fj_legacy_cat02_matrix, fj_test_cat02_matrix};
    const ScalarOperation adapters[2] = {fj_legacy_adapt_cat02, fj_test_adapt_cat02};
    return color_abi_c(matrices, adapters, FJ_TEST_CAT02_MATRIX);
}
