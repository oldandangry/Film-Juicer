#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "juicer_cuda_api.h"

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

_Static_assert(_Generic(((FjFilmGrain*)0)->records, FjFilmGrainRecord*: 1, default: 0), "Film record type");
_Static_assert(sizeof(((FjFilmGrain*)0)->records) == 3 * sizeof(FjFilmGrainRecord), "Film records");
_Static_assert(sizeof(((FjFilmGrainRecord*)0)->mean_light) == 17 * sizeof(float), "Film mean-light levels");
_Static_assert(sizeof(((FjFilmGrainRecord*)0)->mean_density) == 17 * sizeof(float), "Film mean-density levels");
_Static_assert(sizeof(((FjFilmGrainRecord*)0)->correlation) == 17 * sizeof(float), "Film correlation levels");

_Static_assert(_Generic((FjAbortQuery)0, uint32_t (*)(void*): 1, default: 0), "abort signature");
_Static_assert(_Generic(&fj_cuda_create, FjStatus (*)(FjCuda**, FjErrorBuffer*): 1, default: 0), "create signature");
_Static_assert(_Generic(&fj_cuda_inspect, FjStatus (*)(FjCuda*, const FjFrame*, FjCudaContext*, FjErrorBuffer*): 1, default: 0), "inspect signature");
_Static_assert(_Generic(&fj_cuda_render, FjRenderOutcome (*)(FjCuda*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, const FjHostPreparation*, FjAbortCallback, FjErrorBuffer*): 1, default: 0), "render signature");
_Static_assert(_Generic(&fj_cuda_retire_instance, FjStatus (*)(FjCuda*, uint64_t, FjErrorBuffer*): 1, default: 0), "retire signature");
_Static_assert(_Generic(&fj_cuda_shutdown, FjStatus (*)(FjCuda*, FjErrorBuffer*): 1, default: 0), "shutdown signature");
_Static_assert(_Generic(&fj_cuda_destroy, FjStatus (*)(FjCuda*, FjErrorBuffer*): 1, default: 0), "destroy signature");
_Static_assert(_Generic(&fj_cuda_check_terminal, FjStatus (*)(FjCuda*, FjErrorBuffer*): 1, default: 0), "check-terminal signature");

typedef FjStatus (*DirQuery)(void*, const FjDirBoundaryInput*, FjDirBoundaryView*, FjDirBoundaryFailure*, FjErrorBuffer*);
typedef FjStatus (*GaussianQuery)(void*, const FjGaussianKernelInput*, FjGaussianKernelView*, FjGaussianKernelFailure*, FjErrorBuffer*);
_Static_assert(_Generic(((FjHostPreparation*)0)->dir_boundary, DirQuery: 1, default: 0), "DIR query signature");
_Static_assert(_Generic(((FjHostPreparation*)0)->gaussian, GaussianQuery: 1, default: 0), "Gaussian query signature");
_Static_assert(_Generic(&fj_cuda_retire_context, FjStatus (*)(FjCuda*, const FjCudaRetirement*, FjErrorBuffer*): 1, default: 0), "retirement signature");
_Static_assert(_Generic(&fj_cuda_metering, FjStatus (*)(const FjRect*, uint32_t, FjAutoExposure*, FjErrorBuffer*): 1, default: 0), "metering signature");
_Static_assert(sizeof(((FjDiffusion*)0)->lambda_px) == 9 * sizeof(double), "diffusion lane count");
_Static_assert(sizeof(((FjDiffusion*)0)->red_weights) == 9 * sizeof(double), "diffusion red lane count");
_Static_assert(sizeof(((FjDiffusion*)0)->green_weights) == 9 * sizeof(double), "diffusion green lane count");
_Static_assert(sizeof(((FjDiffusion*)0)->blue_weights) == 9 * sizeof(double), "diffusion blue lane count");

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
    const FjCudaRetirement invalid = {0};
    if (fj_cuda_retire_context(NULL, &invalid, &error).category != FJ_STATUS_UNSUPPORTED_INPUT)
        return 4;
    FjAutoExposure metering = {0};
    const FjRect bounds = {-2, -3, 6, 5};
    result = fj_cuda_metering(&bounds, FJ_METER_AVERAGE, &metering, &error);
    if (result.category != FJ_STATUS_SUCCESS || metering.hash == 0 ||
        metering.preview_width != 8 || metering.preview_height != 8)
        return 5;
    return 0;
}
