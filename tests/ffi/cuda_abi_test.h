#ifndef FJ_CUDA_ABI_TEST_H
#define FJ_CUDA_ABI_TEST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Immutable size_t[*count] layout facts; each callee owns static storage.
 * count points to one writable size_t. Spans last for the test process lifetime. */
const size_t* fj_test_cuda_abi_facts(size_t* count);
const size_t* fj_test_cuda_abi_c_facts(size_t* count);

#ifdef __cplusplus
}
#endif

#endif
