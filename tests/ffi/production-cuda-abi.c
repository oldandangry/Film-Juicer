#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "juicer_cuda_api.h"
#include "juicer_dir_boundary_api.h"
#include "juicer_diffusion_api.h"

#define FJ_ABI_TYPE(type, size, alignment)                \
    _Static_assert(sizeof(type) == size, "size: " #type); \
    _Static_assert(_Alignof(type) == alignment, "alignment: " #type);
#define FJ_ABI_FIELD(type, field, offset) \
    _Static_assert(offsetof(type, field) == offset, "offset: " #type "." #field);
#define FJ_ABI_VALUE(tag, value) _Static_assert(tag == value, "value: " #tag);
#include "cuda_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE

typedef FjStatus (*Describe)(const FjDirBoundaryInput*, uint64_t*, FjDirBoundaryFailure*, FjErrorBuffer*);
typedef FjStatus (*Acquire)(const FjDirBoundaryInput*, FjDirBoundaryOwner**, FjDirBoundaryFailure*, FjErrorBuffer*);
typedef FjStatus (*View)(const FjDirBoundaryOwner*, FjDirBoundaryView*, FjDirBoundaryFailure*, FjErrorBuffer*);
typedef FjStatus (*Release)(FjDirBoundaryOwner*, FjErrorBuffer*);
typedef FjStatus (*Components)(const FjDiffusionSample*, FjDiffusionComponents*, FjDiffusionFailure*, FjErrorBuffer*);
_Static_assert(_Generic(&fj_dir_boundary_describe, Describe: 1, default: 0), "DIR describe signature");
_Static_assert(_Generic(&fj_dir_boundary_acquire, Acquire: 1, default: 0), "DIR acquire signature");
_Static_assert(_Generic(&fj_dir_boundary_view, View: 1, default: 0), "DIR view signature");
_Static_assert(_Generic(&fj_dir_boundary_release, Release: 1, default: 0), "DIR release signature");
_Static_assert(_Generic(&fj_diffusion_components_prepare, Components: 1, default: 0), "diffusion signature");
_Static_assert(sizeof(FjDirBoundaryInput) == 120 && _Alignof(FjDirBoundaryInput) == 8, "DIR input layout");
_Static_assert(sizeof(FjDirBoundaryView) == 192 && _Alignof(FjDirBoundaryView) == 8, "DIR view layout");
_Static_assert(sizeof(FjDirBoundaryFailure) == 40 && _Alignof(FjDirBoundaryFailure) == 8, "DIR failure layout");
_Static_assert(sizeof(FjDiffusionComponents) == 288 && _Alignof(FjDiffusionComponents) == 8, "diffusion components layout");

#define CHECK(condition)                                       \
    do {                                                       \
        if (!(condition)) {                                    \
            fprintf(stderr, "DIR boundary: %s\n", #condition); \
            return 1;                                          \
        }                                                      \
    } while (0)

static int status_is(FjStatus status, uint32_t category) {
    return status.category == category && status.api == 0 && status.native_code == 0;
}

static uint64_t word(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int check_dir_boundary(void) {
    uint64_t key = 99;
    char text[6] = {'x', 'x', 'x', 'x', 'x', 'x'};
    FjErrorBuffer error = {text, 5, 99};
    FjDirBoundaryFailure failure = {.reason = FJ_DIR_BOUNDARY_NONE};
    CHECK(status_is(fj_dir_boundary_describe(NULL, &key, &failure, &error), FJ_STATUS_UNSUPPORTED_INPUT));
    CHECK(key == 0 && failure.reason == FJ_DIR_BOUNDARY_RAW);
    CHECK(error.length == 4 && memcmp(text, "spat\0x", sizeof(text)) == 0);

    FjDirBoundaryOwner* first = (FjDirBoundaryOwner*)(uintptr_t)1;
    failure.reason = FJ_DIR_BOUNDARY_NONE;
    CHECK(status_is(fj_dir_boundary_acquire(NULL, &first, &failure, NULL), FJ_STATUS_UNSUPPORTED_INPUT));
    CHECK(!first && failure.reason == FJ_DIR_BOUNDARY_RAW);
    const double sentinel = 1.0;
    FjDirBoundaryView a = {
        .weights = &sentinel,
        .count = 1,
        .identity = 1,
        .width = 1,
        .height = 1,
        .horizontal_length = 1,
        .vertical_length = 1,
        .horizontal_terminal_size = 1,
        .vertical_terminal_size = 1,
        .horizontal_terminal = {1, 1, 1, 1, 1, 1, 1, 1, 1},
        .vertical_terminal = {1, 1, 1, 1, 1, 1, 1, 1, 1},
    };
    failure.reason = FJ_DIR_BOUNDARY_NONE;
    CHECK(status_is(fj_dir_boundary_view(NULL, &a, &failure, NULL), FJ_STATUS_UNSUPPORTED_INPUT));
    CHECK(failure.reason == FJ_DIR_BOUNDARY_RAW);
    CHECK(!a.weights && a.count == 0 && a.identity == 0);
    CHECK(a.width == 0 && a.height == 0 && a.horizontal_length == 0 && a.vertical_length == 0);
    CHECK(a.horizontal_terminal_size == 0 && a.vertical_terminal_size == 0);
    for (size_t i = 0; i < sizeof(a.horizontal_terminal) / sizeof(a.horizontal_terminal[0]); ++i) {
        CHECK(word(a.horizontal_terminal[i]) == 0 && word(a.vertical_terminal[i]) == 0);
    }

    // The four preset captures in tests/dir/boundary-fixtures supply the
    // literal-9x6 numerical words. checked.json supplies the lifetime key and count.
    const FjDirBoundaryInput input = {
        .component = {
            .sigma_px = 4.0f,
            .weight = 1.0f,
            .radius = 12,
            .reference_operator = FJ_DIR_OPERATOR_YVV_REFLECT,
            .feedforward = 0.5,
            .feedback = {0.5, 0.0, 0.0},
            .normalization_denominator = 2.0,
            .feedforward_numerator = 1.0,
            .feedback_numerators = {1.0, 0.0, 0.0},
            .boundary_truncation_accuracy = 1e-11,
            .boundary_certification_tolerance = 1e-6,
            .boundary_derivation_version = 1,
        },
        .width = 9,
        .height = 6,
    };
    CHECK(status_is(fj_dir_boundary_describe(&input, &key, &failure, NULL), FJ_STATUS_SUCCESS));
    CHECK(key == UINT64_C(11586933080928547342) && failure.reason == FJ_DIR_BOUNDARY_NONE);
    FjDirBoundaryOwner* second = NULL;
    CHECK(status_is(fj_dir_boundary_acquire(&input, &first, &failure, NULL), FJ_STATUS_SUCCESS));
    CHECK(status_is(fj_dir_boundary_acquire(&input, &second, &failure, NULL), FJ_STATUS_SUCCESS));
    CHECK(first && second && first != second);
    FjDirBoundaryView b = {0};
    CHECK(status_is(fj_dir_boundary_view(first, &a, &failure, NULL), FJ_STATUS_SUCCESS));
    CHECK(status_is(fj_dir_boundary_view(second, &b, &failure, NULL), FJ_STATUS_SUCCESS));
    CHECK(a.weights && b.weights && a.weights != b.weights && a.identity == key && b.identity == key);
    CHECK(a.count == 45 && b.count == 45 && word(a.weights[0]) == UINT64_C(4602678870712451073));
    CHECK(status_is(fj_dir_boundary_release(first, NULL), FJ_STATUS_SUCCESS));
    FjDirBoundaryView surviving = {0};
    CHECK(status_is(fj_dir_boundary_view(second, &surviving, &failure, NULL), FJ_STATUS_SUCCESS));
    CHECK(surviving.weights == b.weights && surviving.identity == key && surviving.count == 45);
    CHECK(word(surviving.weights[0]) == UINT64_C(4602678870712451073));
    CHECK(word(surviving.weights[44]) == UINT64_C(4591421469774413848));
    CHECK(word(surviving.horizontal_terminal[1]) == UINT64_C(4598175219545276416));
    CHECK(word(surviving.vertical_terminal[8]) == UINT64_C(4607182418800017408));
    FjErrorBuffer malformed = {NULL, 1, 99};
    CHECK(status_is(fj_dir_boundary_release(second, &malformed), FJ_STATUS_UNSUPPORTED_INPUT));
    CHECK(malformed.length == 0);
    CHECK(status_is(fj_dir_boundary_release(NULL, &error), FJ_STATUS_SUCCESS) && error.length == 0);
    return 0;
}

int main(void) {
    FjCuda* owner = NULL;
    FjErrorBuffer error = {NULL, 0, 99};
    FjStatus result = fj_cuda_create(&owner, &error);
    if (result.category != FJ_STATUS_SUCCESS || !owner || error.length != 0)
        return 1;
    result = fj_cuda_shutdown(owner, &error);
    if (result.category != FJ_STATUS_SUCCESS || error.length != 0)
        return 2;
    result = fj_cuda_destroy(owner, &error);
    owner = NULL;
    if (result.category != FJ_STATUS_SUCCESS || error.length != 0)
        return 3;
    if (check_dir_boundary() != 0)
        return 4;
    FjDiffusionComponents components;
    FjDiffusionFailure expansion;
    const unsigned char zero[sizeof(components)] = {0};
    memset(&components, 0xa5, sizeof(components));
    result = fj_diffusion_components_prepare(NULL, &components, &expansion, &error);
    if (result.category != FJ_STATUS_UNSUPPORTED_INPUT || expansion.reason != FJ_DIFFUSION_REASON_RAW ||
        memcmp(&components, zero, sizeof(components)) != 0)
        return 6;
    return 0;
}
