#include <stddef.h>
#include <stdint.h>
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
    uint64_t identity = 99;
    FjDirBoundaryFailure failure;
    memset(&failure, 0xa5, sizeof(failure));
    result = fj_dir_boundary_describe(NULL, &identity, &failure, &error);
    if (result.category != FJ_STATUS_UNSUPPORTED_INPUT || identity != 0 || failure.reason != FJ_DIR_BOUNDARY_RAW)
        return 4;
    result = fj_dir_boundary_release(NULL, &error);
    if (result.category != FJ_STATUS_SUCCESS || error.length != 0)
        return 5;
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
