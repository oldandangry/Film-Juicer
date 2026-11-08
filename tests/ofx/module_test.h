#pragma once

#include "juicer_cuda_api.h"
#include "juicer_ofx_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Nondefault test-support observations; never a shipping OFX interface.
 * Borrowed identities expire at terminal unload. Path modes: 0 current module,
 * 1 lexical input, 2 lookup failure, 3 Windows native-buffer growth. Output
 * capacity/return length count native path units, without a terminator. */
void fj_test_ofx_panic(uint32_t stage);
/* Worker TLS pause bits: 1 boundary return, 2 admitted Describe panic,
 * 4 complete unpublished construction, 8 invalidated Runtime rollback.
 * One worker at a time; waits are bounded. Release before joining the worker;
 * its return is zero if a bounded pause expired and the overlap is unproved. */
void fj_test_ofx_arm_pause(uint32_t stages);
int fj_test_ofx_wait_pause(uint32_t stage);
int fj_test_ofx_release_pause(void);
FjCuda* fj_test_ofx_native(void);
const void* fj_test_ofx_assets(void);
int fj_test_ofx_platform_abi(const size_t* facts, size_t count);
size_t fj_test_ofx_path(uint32_t mode, FjPathView input, void* output, size_t capacity);
int fj_test_ofx_platform_c(void);

#ifdef __cplusplus
}
#endif
