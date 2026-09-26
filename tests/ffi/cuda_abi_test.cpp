#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <type_traits>

#include "juicer_cuda_api.h"
#include "cuda_abi_test.h"

#define FJ_ABI_TYPE(type, size, alignment)     \
    static_assert(sizeof(type) == size);       \
    static_assert(alignof(type) == alignment); \
    static_assert(std::is_standard_layout_v<type>);
#define FJ_ABI_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#define FJ_ABI_VALUE(tag, value) static_assert(tag == value);
#include "cuda_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE

static_assert(std::is_same_v<FjAbortQuery, std::uint32_t (*)(void*)>);
static_assert(std::is_same_v<decltype(&fj_cuda_create), FjStatus (*)(FjStringView, FjCuda**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_cuda_inspect), FjStatus (*)(FjCuda*, const FjFrame*, FjCudaContext*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_cuda_render), FjStatus (*)(FjCuda*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, FjAbortCallback, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_cuda_retire_instance), FjStatus (*)(FjCuda*, std::uint64_t, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_cuda_shutdown), FjStatus (*)(FjCuda*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_cuda_destroy), FjStatus (*)(FjCuda*, FjErrorBuffer*)>);

int main() {
    const std::size_t cppFacts[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#define FJ_ABI_VALUE(tag, value) tag,
#include "cuda_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
    };
    const char* names[] = {
#define FJ_ABI_TYPE(type, size, alignment) "sizeof " #type, "alignof " #type,
#define FJ_ABI_FIELD(type, field, offset) "offsetof " #type "." #field,
#define FJ_ABI_VALUE(tag, value) #tag,
#include "cuda_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
    };
    std::size_t cCount = 0;
    std::size_t rustCount = 0;
    const std::size_t* cFacts = fj_test_cuda_abi_c_facts(&cCount);
    const std::size_t* rustFacts = fj_test_cuda_abi_facts(&rustCount);
    if (cCount != std::size(cppFacts) || rustCount != cCount) {
        std::fprintf(stderr, "ABI fact count mismatch: C=%zu C++=%zu Rust=%zu\n", cCount, std::size(cppFacts), rustCount);
        return 1;
    }
    for (std::size_t index = 0; index < cCount; ++index) {
        if (cppFacts[index] != cFacts[index] || cFacts[index] != rustFacts[index]) {
            std::fprintf(stderr, "%s: C=%zu C++=%zu Rust=%zu\n", names[index], cFacts[index], cppFacts[index], rustFacts[index]);
            return 1;
        }
    }
    std::printf("C/C++/Rust agree on %zu ABI layout/tag/flag facts; all signatures compiled.\n", cCount);
    return 0;
}
