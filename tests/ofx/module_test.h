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
/* One-shot worker TLS descriptor faults: 1 catalog readiness result, 2 early
 * recoverable allocation, 3 owned catalog string allocation, 4 declaration panic. */
void fj_test_ofx_context_fault(uint32_t mode);
/* Worker TLS pause bits: 1 boundary return, 2 admitted Describe panic,
 * 4 complete unpublished construction, 8 invalidated Runtime rollback,
 * 16 invalidated DescribeInContext with its Admission still held.
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

/* Render fault kinds: 1 Film copy, 2 Print copy, 3 diagnostic reservation,
 * 4 action panic, 5 abort panic, 6 release panic (after the actual release).
 * Tokens are fixture-only TLS; arm before Create for identity and each Render
 * for frame facts; all zeros clears. No production archive exports these. */
void fj_test_ofx_render_control(uint32_t kind);
void fj_test_ofx_render_tokens(uint64_t session, uint64_t instance, uint64_t clip);
typedef struct FjOfxRenderObservation {
    double frame_time, frame_rate;
    uint64_t frame_token, session_seed, instance_token, clip_token;
    uint64_t native_entry_count, submission_id, input_identity, build_number;
    size_t backing_identity;
    int32_t full_bounds[4], render_bounds[4];
    float pixel_size_um;
    uint32_t status, fatal;
} FjOfxRenderObservation;
int fj_test_ofx_render_observe(OfxImageEffectHandle, FjOfxRenderObservation*);

#ifdef __cplusplus
}
#endif
