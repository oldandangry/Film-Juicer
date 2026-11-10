#pragma once

#include "juicer_cuda_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Synchronous expansion of a completed sample into nine lambda/R/G/B lanes.
 * Readable input is initialized, aligned and live until return. Output, failure,
 * optional error record and nonempty diagnostic backing are exclusive, mutually
 * disjoint and disjoint from input. Each admitted output clears independently.
 * No pointer, owner or cache survives return; no foreign reentry occurs under a
 * mutable Rust borrow. The calling-thread operation contains panics and preserves
 * binary64 order. Diagnostics reserve NUL and report written bytes; zero capacity
 * ignores backing, and truncation changes neither status nor typed failure. */
typedef struct FjDiffusionSample {
    uint32_t family;
    double group_weights[3], group_lambda_um[3];
    double warmth, spatial_scale, pixel_size_um;
    int32_t radius_px;
    uint64_t hash;
} FjDiffusionSample;

typedef struct FjDiffusionComponents {
    double lambda_px[9], red_weights[9], green_weights[9], blue_weights[9];
} FjDiffusionComponents;

typedef struct FjDiffusionFailure {
    uint32_t reason, stage, field;
} FjDiffusionFailure;

#define FJ_DIFFUSION_REASON_NONE 0U
#define FJ_DIFFUSION_REASON_RAW 1U
#define FJ_DIFFUSION_REASON_PREPARATION 2U
#define FJ_DIFFUSION_REASON_PANIC 3U
#define FJ_DIFFUSION_STAGE_NONE 0U
#define FJ_DIFFUSION_STAGE_CAMERA 1U
#define FJ_DIFFUSION_STAGE_ENLARGER 2U
#define FJ_DIFFUSION_FIELD_NONE 0U
#define FJ_DIFFUSION_FIELD_PIXEL_SIZE_UM 1U
#define FJ_DIFFUSION_FIELD_FULL_FRAME_EXTENT 2U
#define FJ_DIFFUSION_FIELD_FAMILY 3U
#define FJ_DIFFUSION_FIELD_RADIUS_PIXELS 4U
#define FJ_DIFFUSION_FIELD_SAMPLE_DESCRIPTOR_HASH 5U
#define FJ_DIFFUSION_FIELD_SAMPLE_DESCRIPTOR_INACTIVE 6U
#define FJ_DIFFUSION_FIELD_GROUP_WEIGHTS 7U
#define FJ_DIFFUSION_FIELD_GROUP_CENTERS_UM 8U
#define FJ_DIFFUSION_FIELD_EFFECTIVE_WARMTH 9U
#define FJ_DIFFUSION_FIELD_SPATIAL_SCALE 10U
#define FJ_DIFFUSION_FIELD_COMPONENT_EXPANSION 11U
#define FJ_DIFFUSION_FIELD_COMPONENT_COUNT 12U
#define FJ_DIFFUSION_FIELD_HALO_CHANNEL_WEIGHTS 13U
#define FJ_DIFFUSION_FIELD_COMPONENT_OUTPUT 14U
#define FJ_DIFFUSION_FIELD_ROUTE 15U
#define FJ_DIFFUSION_FIELD_FULL_FRAME_DOMAIN 16U
#define FJ_DIFFUSION_FIELD_ENLARGER_ROUTE 17U
#define FJ_DIFFUSION_FIELD_LINEAR_DOMAIN 18U
#define FJ_DIFFUSION_FIELD_STAGE_SAMPLE 19U
#define FJ_DIFFUSION_FIELD_STAGE_HASH 20U
#define FJ_DIFFUSION_FIELD_FRAME_SET_HASH 21U

FjStatus fj_diffusion_components_prepare(const FjDiffusionSample*, FjDiffusionComponents*, FjDiffusionFailure*, FjErrorBuffer*);

#ifdef __cplusplus
}
#endif
