#pragma once

#include "juicer_cuda_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Rust owns each completed reflected boundary. Input and admitted writable peers
 * are live, initialized, aligned and disjoint, including diagnostic backing.
 * Each output clears independently before admission. NULL diagnostics are valid;
 * zero capacity ignores backing; positive capacity reserves NUL and length counts
 * written bytes. Truncation does not alter status or typed failure metadata.
 * Describe precedes the native warm check; acquire is cold-only. weights contains
 * immutable row-major f64 horizontal 3*horizontal_length, then vertical
 * 3*vertical_length. Terminal matrices use their terminal-size stride. Native
 * staging owns all queued upload bytes before release. Views expire at consuming
 * release, which excludes readers and consumes even on diagnostic/panic failure.
 * Calls are synchronous on the calling thread, contain panics, retain no input
 * pointer and permit no reentry while a mutable Rust borrow exists. */
#define FJ_DIR_BOUNDARY_NONE UINT32_C(0)
#define FJ_DIR_BOUNDARY_RAW UINT32_C(1)
#define FJ_DIR_BOUNDARY_REQUEST UINT32_C(2)
#define FJ_DIR_BOUNDARY_AXIS_SIZE UINT32_C(3)
#define FJ_DIR_BOUNDARY_POLICY UINT32_C(4)
#define FJ_DIR_BOUNDARY_CAUSAL_SINGULAR UINT32_C(5)
#define FJ_DIR_BOUNDARY_NONFINITE_WEIGHTS UINT32_C(6)
#define FJ_DIR_BOUNDARY_CAUSAL_CERTIFICATION UINT32_C(7)
#define FJ_DIR_BOUNDARY_TAIL_ESCAPE UINT32_C(8)
#define FJ_DIR_BOUNDARY_TERMINAL_SINGULAR UINT32_C(9)
#define FJ_DIR_BOUNDARY_TERMINAL_CERTIFICATION UINT32_C(10)
#define FJ_DIR_BOUNDARY_STORAGE_SIZE UINT32_C(11)
#define FJ_DIR_BOUNDARY_ALLOCATION UINT32_C(12)
#define FJ_DIR_BOUNDARY_PANIC UINT32_C(13)
#define FJ_DIR_BOUNDARY_NO_AXIS UINT32_C(0)
#define FJ_DIR_BOUNDARY_HORIZONTAL UINT32_C(1)
#define FJ_DIR_BOUNDARY_VERTICAL UINT32_C(2)
#define FJ_DIR_BOUNDARY_NO_ROW UINT32_C(3)
typedef struct FjDirBoundaryInput {
    FjDirGaussian component;
    int32_t width, height;
} FjDirBoundaryInput;
typedef struct FjDirBoundaryView {
    const double* weights;
    size_t count;
    uint64_t identity;
    int32_t width, height, horizontal_length, vertical_length;
    int32_t horizontal_terminal_size, vertical_terminal_size;
    double horizontal_terminal[9], vertical_terminal[9];
} FjDirBoundaryView;
typedef struct FjDirBoundaryFailure {
    uint32_t reason, axis, row;
    int32_t length;
    float sigma_px;
    double sum, tolerance;
} FjDirBoundaryFailure;
typedef struct FjDirBoundaryOwner FjDirBoundaryOwner;
FjStatus fj_dir_boundary_describe(const FjDirBoundaryInput*, uint64_t*, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_dir_boundary_acquire(const FjDirBoundaryInput*, FjDirBoundaryOwner**, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_dir_boundary_view(const FjDirBoundaryOwner*, FjDirBoundaryView*, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_dir_boundary_release(FjDirBoundaryOwner*, FjErrorBuffer*);

#ifdef __cplusplus
}
#endif
