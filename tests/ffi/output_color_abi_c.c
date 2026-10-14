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
