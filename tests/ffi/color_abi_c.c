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

_Static_assert(sizeof(FjInputColorMatrices) == 96 && _Alignof(FjInputColorMatrices) == 4, "input matrices ABI");
_Static_assert(offsetof(FjInputColorMatrices, rgb_to_xyz) == 0 && offsetof(FjInputColorMatrices, nominal_white_xyz) == 36, "input matrix/white offsets");
_Static_assert(offsetof(FjInputColorMatrices, xyz_to_linear_srgb) == 48 && offsetof(FjInputColorMatrices, d65_white_xyz) == 84, "input inverse/D65 offsets");
_Static_assert(sizeof(FjInputColorConversion) == 84 && _Alignof(FjInputColorConversion) == 4, "input conversion ABI");
_Static_assert(offsetof(FjInputColorConversion, input_space) == 0 && offsetof(FjInputColorConversion, decode_cctf) == 4 && offsetof(FjInputColorConversion, adapt_xyz) == 8, "input conversion flag offsets");
_Static_assert(offsetof(FjInputColorConversion, rgb_to_xyz) == 12 && offsetof(FjInputColorConversion, xyz_adapt) == 48, "input conversion matrix offsets");
_Static_assert(FJ_INPUT_DWG == 0 && FJ_INPUT_BT2020 == 1 && FJ_INPUT_ACES2065_1 == 2 && FJ_INPUT_SRGB_REC709 == 3, "input space tags");

typedef FjStatus (*InputMatricesOperation)(uint32_t, FjInputColorMatrices*);
typedef FjStatus (*InputDwgOperation)(const FjInputColorConversion*, const float*, uint32_t, float*, float*);
typedef FjStatus (*InputSrgbOperation)(const FjInputColorConversion*, const float*, const float*, float*, float*);
typedef FjStatus (*InputLeafOperation)(const float*, float*);
typedef FjStatus (*InputProjectionOperation)(const float*, const float*, const float*, float*);

static void seed_triplet(float output[5]) {
    for (int i = 0; i < 5; ++i) {
        output[i] = 42.0f;
    }
}

static int triplet_canaries(const float output[5], int cleared) {
    const float zero[3] = {0, 0, 0};
    return output[0] == 42.0f && output[4] == 42.0f && (!cleared || memcmp(output + 1, zero, sizeof(zero)) == 0);
}

static int matrices_cleared(const FjInputColorMatrices* output) {
    const FjInputColorMatrices zero = {{0}, {0}, {0}, {0}};
    return memcmp(output, &zero, sizeof(zero)) == 0;
}

int fj_test_input_color_abi_c(void) {
    const InputMatricesOperation matrices[2] = {fj_legacy_input_matrices, fj_test_input_matrices};
    const InputDwgOperation dwg[2] = {fj_legacy_input_to_dwg, fj_test_input_to_dwg};
    const InputSrgbOperation srgb[2] = {fj_legacy_input_to_linear_srgb, fj_test_input_to_linear_srgb};
    const InputLeafOperation leaves[2][2] = {{fj_legacy_linear_srgb_to_xyz, fj_legacy_dwg_to_xyz}, {fj_test_linear_srgb_to_xyz, fj_test_dwg_to_xyz}};
    const InputProjectionOperation projection[2] = {fj_legacy_project_linear_rgb_to_xyz, fj_test_project_linear_rgb_to_xyz};
    const float rgb[3] = {0.184f, -0.25f, 2.0f};
    const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    FjInputColorConversion input = {FJ_INPUT_DWG, 0, 0, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}};
    float out_rgb[5], out_xyz[5];
    int failures = 0;
    for (int boundary = 0; boundary < 2; ++boundary) {
        struct {
            uint32_t before;
            FjInputColorMatrices output;
            uint32_t after;
        } bounded = {UINT32_C(0x12345678), {{0}, {0}, {0}, {0}}, UINT32_C(0x87654321)};
        for (uint32_t space = 0; space < 4; ++space) {
            failures += !has_status(matrices[boundary](space, &bounded.output), FJ_STATUS_SUCCESS);
            failures += bounded.output.rgb_to_xyz[0] == 0.0f;
            failures += bounded.before != UINT32_C(0x12345678) || bounded.after != UINT32_C(0x87654321);
        }
        failures += !has_status(matrices[boundary](4, &bounded.output), FJ_STATUS_UNSUPPORTED_INPUT);
        failures += !matrices_cleared(&bounded.output);
        failures += !has_status(matrices[boundary](0, NULL), FJ_STATUS_UNSUPPORTED_INPUT);
        failures += bounded.before != UINT32_C(0x12345678) || bounded.after != UINT32_C(0x87654321);
        for (int target = 0; target < 2; ++target) {
            const int pointers = target == 0 ? 4 : 5;
            for (int slot = 0; slot < pointers; ++slot) {
                seed_triplet(out_rgb);
                seed_triplet(out_xyz);
                const int rgb_slot = target == 0 ? 2 : 3;
                const int xyz_slot = target == 0 ? 3 : 4;
                const FjStatus result = target == 0
                                            ? dwg[boundary](slot == 0 ? NULL : &input, slot == 1 ? NULL : rgb, 1, slot == rgb_slot ? NULL : out_rgb + 1, slot == xyz_slot ? NULL : out_xyz + 1)
                                            : srgb[boundary](slot == 0 ? NULL : &input, slot == 1 ? NULL : rgb, slot == 2 ? NULL : identity, slot == rgb_slot ? NULL : out_rgb + 1, slot == xyz_slot ? NULL : out_xyz + 1);
                failures += !has_status(result, FJ_STATUS_UNSUPPORTED_INPUT);
                failures += !triplet_canaries(out_rgb, slot != rgb_slot) || !triplet_canaries(out_xyz, slot != xyz_slot);
                if (slot == rgb_slot) {
                    failures += out_rgb[1] != 42.0f || out_rgb[2] != 42.0f || out_rgb[3] != 42.0f;
                }
                if (slot == xyz_slot) {
                    failures += out_xyz[1] != 42.0f || out_xyz[2] != 42.0f || out_xyz[3] != 42.0f;
                }
            }
            for (int invalid = 0; invalid < 4; ++invalid) {
                input.input_space = invalid == 0 ? UINT32_MAX : FJ_INPUT_DWG;
                input.decode_cctf = invalid == 1 ? 2u : 0u;
                input.adapt_xyz = invalid == 2 ? 2u : 0u;
                if (target == 1 && invalid == 3) {
                    continue;
                }
                seed_triplet(out_rgb);
                seed_triplet(out_xyz);
                const FjStatus result = target == 0 ? dwg[boundary](&input, rgb, invalid == 3 ? 2u : 0u, out_rgb + 1, out_xyz + 1)
                                                    : srgb[boundary](&input, rgb, identity, out_rgb + 1, out_xyz + 1);
                failures += !has_status(result, FJ_STATUS_UNSUPPORTED_INPUT);
                failures += !triplet_canaries(out_rgb, 1) || !triplet_canaries(out_xyz, 1);
            }
            input.input_space = FJ_INPUT_DWG;
            input.decode_cctf = 0;
            input.adapt_xyz = 0;
            /* Complete initialized read-only storage may legally overlap. */
            seed_triplet(out_rgb);
            seed_triplet(out_xyz);
            const FjStatus aliased = target == 0 ? dwg[boundary](&input, input.rgb_to_xyz, 0, out_rgb + 1, out_xyz + 1)
                                                 : srgb[boundary](&input, input.rgb_to_xyz, input.rgb_to_xyz, out_rgb + 1, out_xyz + 1);
            failures += !has_status(aliased, FJ_STATUS_SUCCESS);
            failures += !triplet_canaries(out_rgb, 0) || !triplet_canaries(out_xyz, 0);
            /* Disabled adaptation ignores initialized scientific nonfinite content. */
            const uint32_t nan = UINT32_C(0x7fc00000);
            memcpy(input.xyz_adapt, &nan, sizeof(nan));
            failures += !has_status(target == 0 ? dwg[boundary](&input, rgb, 0, out_rgb + 1, out_xyz + 1)
                                                : srgb[boundary](&input, rgb, identity, out_rgb + 1, out_xyz + 1),
                                    FJ_STATUS_SUCCESS);
            memcpy(input.xyz_adapt, identity, sizeof(identity));
        }
        for (int leaf = 0; leaf < 2; ++leaf) {
            for (int slot = 0; slot < 2; ++slot) {
                seed_triplet(out_xyz);
                failures += !has_status(leaves[boundary][leaf](slot == 0 ? NULL : rgb, slot == 1 ? NULL : out_xyz + 1), FJ_STATUS_UNSUPPORTED_INPUT);
                failures += !triplet_canaries(out_xyz, slot == 0);
            }
            const uint32_t infinite[3] = {UINT32_C(0x7f800000), UINT32_C(0x3f800000), UINT32_C(0x3f000000)};
            float special[3];
            memcpy(special, infinite, sizeof(special));
            failures += !has_status(leaves[boundary][leaf](special, out_xyz + 1), FJ_STATUS_SUCCESS);
            failures += !triplet_canaries(out_xyz, 0);
        }
        for (int slot = 0; slot < 4; ++slot) {
            seed_triplet(out_xyz);
            failures += !has_status(projection[boundary](slot == 0 ? NULL : rgb, slot == 1 ? NULL : identity, slot == 2 ? NULL : identity, slot == 3 ? NULL : out_xyz + 1), FJ_STATUS_UNSUPPORTED_INPUT);
            failures += !triplet_canaries(out_xyz, slot != 3);
        }
        failures += !has_status(projection[boundary](identity, identity, identity, out_xyz + 1), FJ_STATUS_SUCCESS);
    }
    for (uint32_t operation = 5; operation <= 10; ++operation) {
        for (uint32_t fault = 1; fault <= 2; ++fault) {
            FjInputColorMatrices matrix;
            memset(&matrix, 42, sizeof(matrix));
            seed_triplet(out_rgb);
            seed_triplet(out_xyz);
            failures += !has_status(fj_test_color_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
            FjStatus result;
            switch (operation) {
                case FJ_TEST_INPUT_MATRICES:
                    result = matrices[0](0, &matrix);
                    break;
                case FJ_TEST_INPUT_TO_DWG:
                    result = dwg[0](&input, rgb, 1, out_rgb + 1, out_xyz + 1);
                    break;
                case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                    result = srgb[0](&input, rgb, identity, out_rgb + 1, out_xyz + 1);
                    break;
                case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                    result = leaves[0][0](rgb, out_xyz + 1);
                    break;
                case FJ_TEST_DWG_TO_XYZ:
                    result = leaves[0][1](rgb, out_xyz + 1);
                    break;
                case FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ:
                    result = projection[0](rgb, identity, identity, out_xyz + 1);
                    break;
                default:
                    return failures + 1;
            }
            failures += !has_status(result, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
            failures += operation == 5 ? !matrices_cleared(&matrix) : !triplet_canaries(out_xyz, 1);
            if (operation == 6 || operation == 7) {
                failures += !triplet_canaries(out_rgb, 1);
            }
            failures += !has_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
        }
    }
    for (uint32_t space = 0; space < 4; ++space) {
        for (uint32_t decode = 0; decode < 2; ++decode) {
            seed_triplet(out_rgb);
            failures += !has_status(fj_test_decode_input(space, decode, rgb, out_rgb + 1), FJ_STATUS_SUCCESS);
            failures += !triplet_canaries(out_rgb, 0);
        }
    }
    for (int invalid = 0; invalid < 4; ++invalid) {
        seed_triplet(out_rgb);
        failures += !has_status(fj_test_decode_input(invalid == 0 ? 4u : 0u, invalid == 1 ? 2u : 0u, invalid == 2 ? NULL : rgb, invalid == 3 ? NULL : out_rgb + 1), FJ_STATUS_UNSUPPORTED_INPUT);
        failures += !triplet_canaries(out_rgb, invalid != 3);
    }
    return failures;
}
