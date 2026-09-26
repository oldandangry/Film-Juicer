#include <stddef.h>
#include <stdint.h>

#include "juicer_cuda_api.h"
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
_Static_assert(_Generic(&fj_cuda_create, FjStatus (*)(FjStringView, FjCuda**, FjErrorBuffer*): 1, default: 0), "create signature");
_Static_assert(_Generic(&fj_cuda_inspect, FjStatus (*)(FjCuda*, const FjFrame*, FjCudaContext*, FjErrorBuffer*): 1, default: 0), "inspect signature");
_Static_assert(_Generic(&fj_cuda_render, FjStatus (*)(FjCuda*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, FjAbortCallback, FjErrorBuffer*): 1, default: 0), "render signature");
_Static_assert(_Generic(&fj_cuda_retire_instance, FjStatus (*)(FjCuda*, uint64_t, FjErrorBuffer*): 1, default: 0), "retire signature");
_Static_assert(_Generic(&fj_cuda_shutdown, FjStatus (*)(FjCuda*, FjErrorBuffer*): 1, default: 0), "shutdown signature");
_Static_assert(_Generic(&fj_cuda_destroy, FjStatus (*)(FjCuda*, FjErrorBuffer*): 1, default: 0), "destroy signature");

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
