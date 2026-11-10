#include <stddef.h>
#include <stdint.h>

#include "juicer_cuda_api.h"
#include "juicer_test_api.h"
#include "cuda_abi_test.h"

#define FJ_ABI_TYPE(type, size, alignment)                \
    _Static_assert(sizeof(type) == size, "size: " #type); \
    _Static_assert(_Alignof(type) == alignment, "alignment: " #type);
#define FJ_ABI_FIELD(type, field, offset) \
    _Static_assert(offsetof(type, field) == offset, "offset: " #type "." #field);
#define FJ_ABI_VALUE(tag, value) _Static_assert(tag == value, "tag: " #tag);
#include "cuda_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE

_Static_assert(_Generic((FjAbortQuery)0, uint32_t (*)(void*): 1, default: 0), "abort signature");
_Static_assert(_Generic(&fj_cuda_create, FjStatus (*)(FjCuda**, FjErrorBuffer*): 1, default: 0), "create signature");
_Static_assert(_Generic(&fj_cuda_inspect, FjStatus (*)(FjCuda*, const FjFrame*, FjCudaContext*, FjErrorBuffer*): 1, default: 0), "inspect signature");
_Static_assert(_Generic(&fj_cuda_render, FjRenderOutcome (*)(FjCuda*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, FjAbortCallback, FjErrorBuffer*): 1, default: 0), "render signature");
_Static_assert(_Generic(&fj_cuda_retire_instance, FjStatus (*)(FjCuda*, uint64_t, FjErrorBuffer*): 1, default: 0), "retire signature");
_Static_assert(_Generic(&fj_cuda_shutdown, FjStatus (*)(FjCuda*, FjErrorBuffer*): 1, default: 0), "shutdown signature");
_Static_assert(_Generic(&fj_cuda_destroy, FjStatus (*)(FjCuda*, FjErrorBuffer*): 1, default: 0), "destroy signature");
_Static_assert(_Generic(&fj_cuda_check_terminal, FjStatus (*)(FjCuda*, FjErrorBuffer*): 1, default: 0), "check-terminal signature");

_Static_assert(_Generic(&fj_test_runtime_create, FjStatus (*)(FjPathView, FjRuntime**, FjErrorBuffer*): 1, default: 0), "Runtime create signature");
_Static_assert(_Generic(&fj_test_runtime_borrow, FjStatus (*)(const FjRuntime*, FjRuntimeView*, FjErrorBuffer*): 1, default: 0), "Runtime borrow signature");
_Static_assert(_Generic(&fj_test_runtime_check_close, FjStatus (*)(const FjRuntime*, FjErrorBuffer*): 1, default: 0), "Runtime check-close signature");
_Static_assert(_Generic(&fj_test_runtime_shutdown, FjStatus (*)(const FjRuntime*, FjErrorBuffer*): 1, default: 0), "Runtime shutdown signature");
_Static_assert(_Generic(&fj_test_runtime_destroy, FjStatus (*)(FjRuntime*, FjErrorBuffer*): 1, default: 0), "Runtime destroy signature");
_Static_assert(sizeof(FjRuntimeView) == sizeof(const FjAssets*) + sizeof(FjCuda*), "Runtime view size");
_Static_assert(_Alignof(FjRuntimeView) == _Alignof(const FjAssets*), "Runtime view alignment");
_Static_assert(offsetof(FjRuntimeView, assets) == 0, "Runtime view Assets offset");
_Static_assert(offsetof(FjRuntimeView, cuda) >= sizeof(const FjAssets*), "Runtime view CUDA offset");

_Static_assert(_Generic(&fj_test_assets_noise_acquire, FjStatus (*)(const FjAssets*, FjNoise**, FjErrorBuffer*): 1, default: 0), "noise acquire signature");
_Static_assert(_Generic(&fj_test_noise_view, FjStatus (*)(const FjNoise*, FjStaticNoise*, FjErrorBuffer*): 1, default: 0), "noise view signature");
_Static_assert(_Generic(&fj_test_noise_release, FjStatus (*)(FjNoise*, FjErrorBuffer*): 1, default: 0), "noise release signature");
_Static_assert(_Generic(&fj_test_noise_acquire, FjStatus (*)(FjStringView, FjNoise**, FjErrorBuffer*): 1, default: 0), "fixture noise acquire signature");
_Static_assert(_Generic(&fj_test_noise_view, FjStatus (*)(const FjNoise*, FjStaticNoise*, FjErrorBuffer*): 1, default: 0), "fixture noise view signature");
_Static_assert(_Generic(&fj_test_noise_release, FjStatus (*)(FjNoise*, FjErrorBuffer*): 1, default: 0), "fixture noise release signature");

const size_t* fj_test_cuda_abi_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), _Alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#define FJ_ABI_VALUE(tag, value) tag,
#include "cuda_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}
