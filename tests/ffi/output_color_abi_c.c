#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "juicer_legacy_api.h"
#include "juicer_test_api.h"

_Static_assert(sizeof(FjOutputEncoding) == 68 && _Alignof(FjOutputEncoding) == 4, "encoding layout");
_Static_assert(offsetof(FjOutputEncoding, schema) == 0 && offsetof(FjOutputEncoding, cctf_kind) == 4, "encoding tags");
_Static_assert(offsetof(FjOutputEncoding, gamma) == 8 && offsetof(FjOutputEncoding, a) == 12, "encoding gamma/a");
_Static_assert(offsetof(FjOutputEncoding, b) == 16 && offsetof(FjOutputEncoding, c) == 20, "encoding b/c");
_Static_assert(offsetof(FjOutputEncoding, d) == 24 && offsetof(FjOutputEncoding, linear_cutoff) == 28, "encoding d/cutoff");
_Static_assert(offsetof(FjOutputEncoding, dwg_to_output) == 32, "encoding matrix");
_Static_assert(sizeof(FjOutputColorPreparation) == 168 && _Alignof(FjOutputColorPreparation) == 8, "output preparation layout");
_Static_assert(offsetof(FjOutputColorPreparation, output_space) == 0, "output tag");
_Static_assert(offsetof(FjOutputColorPreparation, rgb_to_xyz) == 4, "output forward");
_Static_assert(offsetof(FjOutputColorPreparation, xyz_to_rgb) == 40, "output inverse");
_Static_assert(offsetof(FjOutputColorPreparation, white_xyz) == 76, "output white");
_Static_assert(offsetof(FjOutputColorPreparation, encoding) == 88, "output encoding");
_Static_assert(offsetof(FjOutputColorPreparation, entry_hash) == 160, "output identity");
_Static_assert(_Generic(&fj_legacy_output_color, FjStatus (*)(uint32_t, FjOutputColorPreparation*): 1, default: 0), "output signature");
_Static_assert(_Generic(&fj_test_output_color, FjStatus (*)(uint32_t, FjOutputColorPreparation*): 1, default: 0), "facade signature");

static int category(FjStatus result, uint32_t expected) {
    return result.category == expected && result.api == FJ_API_NONE && result.native_code == 0;
}
int output_color_abi_c(void) {
    typedef FjStatus (*Prepare)(uint32_t, FjOutputColorPreparation*);
    const Prepare operations[2] = {fj_legacy_output_color, fj_test_output_color};
    int failures = 0;
    struct Guarded {
        uint64_t before;
        FjOutputColorPreparation out;
        uint64_t after;
    } guarded;
    const unsigned char zeros[sizeof(FjOutputColorPreparation)] = {0};
    for (int facade = 0; facade < 2; ++facade) {
        Prepare prepare = operations[facade];
        for (uint32_t tag = 0; tag < 9; ++tag) {
            guarded.before = UINT64_C(0x1122334455667788);
            guarded.after = UINT64_C(0x8877665544332211);
            failures += !category(prepare(tag, &guarded.out), FJ_STATUS_SUCCESS);
            failures += guarded.out.output_space != tag || guarded.out.encoding.schema != 1 || guarded.out.encoding.cctf_kind > 5 || guarded.out.entry_hash == 0;
            failures += guarded.before != UINT64_C(0x1122334455667788) || guarded.after != UINT64_C(0x8877665544332211);
        }
        failures += !category(prepare(0, NULL), FJ_STATUS_UNSUPPORTED_INPUT);
        _Alignas(8) unsigned char misaligned[sizeof(FjOutputColorPreparation) + 8];
        memset(misaligned, 0xa5, sizeof(misaligned));
        failures += !category(prepare(0, (FjOutputColorPreparation*)(void*)(misaligned + 1)), FJ_STATUS_UNSUPPORTED_INPUT);
        for (size_t i = 0; i < sizeof(misaligned); ++i)
            failures += misaligned[i] != 0xa5;
        failures += !category(prepare(0, (FjOutputColorPreparation*)(uintptr_t)(UINTPTR_MAX - 7)), FJ_STATUS_UNSUPPORTED_INPUT);
        const uint32_t invalid[3] = {9, 255, UINT32_MAX};
        for (size_t i = 0; i < 3; ++i) {
            memset(&guarded.out, 0xa5, sizeof(guarded.out));
            failures += !category(prepare(invalid[i], &guarded.out), FJ_STATUS_UNSUPPORTED_INPUT);
            failures += memcmp(&guarded.out, zeros, sizeof(guarded.out)) != 0;
        }
        for (uint32_t fault = 1; fault <= 2; ++fault) {
            FjStatus armed = facade ? fj_test_color_arm_facade_fault(12, 2, fault) : fj_test_color_arm_fault(11, 2, fault);
            failures += !category(armed, FJ_STATUS_SUCCESS);
            failures += !category(operations[1 - facade](0, &guarded.out), FJ_STATUS_SUCCESS);
            failures += !category(prepare(0, &guarded.out), FJ_STATUS_SUCCESS);
            failures += !category(prepare(0, &guarded.out), fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
            failures += memcmp(&guarded.out, zeros, sizeof(guarded.out)) != 0;
            if (facade)
                failures += fj_test_color_facade_fault_consumed(12, fault) != 1;
            failures += !category(prepare(0, &guarded.out), FJ_STATUS_SUCCESS);
            (void)fj_test_color_clear_fault();
            (void)fj_test_color_clear_facade_fault();
        }
    }
    return failures;
}

_Static_assert(sizeof(FjScannerColorInput) == 40 && _Alignof(FjScannerColorInput) == 8, "scanner input layout");
_Static_assert(offsetof(FjScannerColorInput, medium) == 0 && offsetof(FjScannerColorInput, output_space) == 4 && offsetof(FjScannerColorInput, apply_cctf) == 8 && offsetof(FjScannerColorInput, illuminant_xyz) == 12 && offsetof(FjScannerColorInput, illuminant_hash) == 24 && offsetof(FjScannerColorInput, active_gamut_recipe_hash) == 32, "scanner input offsets");
_Static_assert(sizeof(FjScannerColor) == 176 && _Alignof(FjScannerColor) == 8, "scanner result layout");
_Static_assert(offsetof(FjScannerColor, cat02) == 0 && offsetof(FjScannerColor, xyz_to_rgb) == 36 && offsetof(FjScannerColor, illuminant_xyz) == 72 && offsetof(FjScannerColor, output_space) == 84 && offsetof(FjScannerColor, flags) == 88 && offsetof(FjScannerColor, encoding) == 92 && offsetof(FjScannerColor, active_gamut_recipe_hash) == 160 && offsetof(FjScannerColor, hash) == 168, "scanner result offsets");
_Static_assert(sizeof(FjScannerColorFailure) == 8 && _Alignof(FjScannerColorFailure) == 4 && offsetof(FjScannerColorFailure, reason) == 0 && offsetof(FjScannerColorFailure, stage) == 4, "scanner failure layout");
typedef FjStatus (*ScannerColorPrepare)(const FjScannerColorInput*, FjScannerColor*, FjScannerColorFailure*, FjErrorBuffer*);
_Static_assert(_Generic(&fj_legacy_scanner_color_prepare, ScannerColorPrepare: 1, default: 0), "scanner legacy signature");
_Static_assert(_Generic(&fj_test_scanner_color_prepare, ScannerColorPrepare: 1, default: 0), "scanner facade signature");
int scanner_color_abi_c(void) {
    const ScannerColorPrepare operations[2] = {fj_legacy_scanner_color_prepare, fj_test_scanner_color_prepare};
    int failures = 0;
    const unsigned char zero[sizeof(FjScannerColor)] = {0};
    for (int facade = 0; facade < 2; ++facade) {
        ScannerColorPrepare prepare = operations[facade];
        FjScannerColorInput in = {255, 255, 0, {0.950455f, 1.0f, 1.089058f}, 1234567, 0};
        struct Guarded {
            uint64_t before;
            FjScannerColor out;
            uint64_t after;
        } g;
        struct FailureGuard {
            uint32_t before;
            FjScannerColorFailure out;
            uint32_t after;
        } f = {0x12345678, {99, 99}, 0x87654321};
        char text[128];
        FjErrorBuffer error = {text, sizeof(text), 99};
        g.before = UINT64_C(0x1122334455667788);
        g.after = UINT64_C(0x8877665544332211);
        failures += !category(prepare(&in, &g.out, &f.out, &error), 0) || f.out.reason != 0 || f.out.stage != 0 || g.out.output_space != 255 || g.out.flags != 2 || g.out.encoding.schema != 1 || error.length != 0 || text[0] != 0;
        for (int field = 0; field < 3; ++field)
            for (int wide = 0; wide < 2; ++wide) {
                FjScannerColorInput bad = in;
                uint32_t value = wide ? UINT32_MAX : 256;
                if (field == 0)
                    bad.medium = value;
                else if (field == 1)
                    bad.output_space = value;
                else
                    bad.apply_cctf = value;
                memset(&g.out, 0xa5, sizeof(g.out));
                f.out = (FjScannerColorFailure){99, 99};
                failures += !category(prepare(&bad, &g.out, &f.out, &error), 2) || f.out.reason != 1 || f.out.stage != 0 || memcmp(&g.out, zero, sizeof(g.out)) != 0 || strcmp(text, "invalid scanner color arguments") != 0;
            }
        in.illuminant_hash = 0;
        memset(&g.out, 0xa5, sizeof(g.out));
        failures += !category(prepare(&in, &g.out, &f.out, &error), 3) || f.out.reason != 2 || f.out.stage != 0 || memcmp(&g.out, zero, sizeof(g.out)) != 0 || strcmp(text, "scanner illuminant hash invalid for color runtime") != 0;
        in.illuminant_hash = 1234567;
        memset(&g.out, 0xa5, sizeof(g.out));
        failures += !category(prepare(NULL, &g.out, &f.out, &error), 2) || f.out.reason != 1 || memcmp(&g.out, zero, sizeof(g.out)) != 0;
        memset(&g.out, 0xa5, sizeof(g.out));
        failures += !category(prepare(&in, &g.out, NULL, &error), 2) || memcmp(&g.out, zero, sizeof(g.out)) != 0;
        f.out = (FjScannerColorFailure){99, 99};
        failures += !category(prepare(&in, NULL, &f.out, &error), 2) || f.out.reason != 1 || f.out.stage != 0;
        memset(&g.out, 0xa5, sizeof(g.out));
        failures += !category(prepare(&in, &g.out, &f.out, NULL), 2) || f.out.reason != 1 || memcmp(&g.out, zero, sizeof(g.out)) != 0;
        FjErrorBuffer malformed = {NULL, 1, 99};
        memset(&g.out, 0xa5, sizeof(g.out));
        failures += !category(prepare(&in, &g.out, &f.out, &malformed), 2) || malformed.length != 0 || f.out.reason != 1 || memcmp(&g.out, zero, sizeof(g.out)) != 0;
        _Alignas(8) unsigned char unaligned[sizeof(FjScannerColor) + 8];
        memset(unaligned, 0xa5, sizeof(unaligned));
        failures += !category(prepare(&in, (FjScannerColor*)(void*)(unaligned + 1), &f.out, &error), 2) || f.out.reason != 1;
        for (size_t i = 0; i < sizeof(unaligned); ++i)
            failures += unaligned[i] != 0xa5;
        memset(&g.out, 0xa5, sizeof(g.out));
        failures += !category(prepare((const FjScannerColorInput*)(const void*)(unaligned + 1), &g.out, &f.out, &error), 2) || memcmp(&g.out, zero, sizeof(g.out)) != 0;
        failures += !category(prepare(&in, (FjScannerColor*)(uintptr_t)(UINTPTR_MAX - 7), &f.out, &error), 2) || f.out.reason != 1;
        FjErrorBuffer empty = {NULL, 0, 99};
        failures += !category(prepare(&in, &g.out, &f.out, &empty), 0) || empty.length != 0;
        char short_text[2] = {99, 99};
        FjErrorBuffer truncated = {short_text, 2, 99};
        failures += !category(prepare(NULL, &g.out, &f.out, &truncated), 2) || truncated.length != 1 || short_text[1] != 0;
        for (uint32_t operation = 3; operation <= 11; operation += 8)
            for (uint32_t fault = 1; fault <= 2; ++fault) {
                failures += !category(fj_test_color_arm_fault(operation, 2, fault), 0);
                in.illuminant_hash = 0;
                failures += !category(prepare(&in, &g.out, &f.out, &error), 3) || f.out.reason != 2;
                in.illuminant_hash = 1234567;
                failures += !category(prepare(&in, &g.out, &f.out, &error), 0);
                memset(&g.out, 0xa5, sizeof(g.out));
                failures += !category(prepare(&in, &g.out, &f.out, &error), fault == 1 ? 2 : 8) || f.out.reason != (fault == 1 ? 4u : 5u) || f.out.stage != (fault == 1 ? (operation == 11 ? 1u : 2u) : 0u) || memcmp(&g.out, zero, sizeof(g.out)) != 0;
                failures += strcmp(text, fault == 2 ? "scanner color preparation panicked" : operation == 11 ? "Output color preparation failed"
                                                                                                             : "CAT02 matrix preparation failed") != 0;
                failures += !category(prepare(&in, &g.out, &f.out, &error), 0);
                (void)fj_test_color_clear_fault();
            }
        failures += g.before != UINT64_C(0x1122334455667788) || g.after != UINT64_C(0x8877665544332211) || f.before != 0x12345678 || f.after != 0x87654321;
    }
    return failures;
}
