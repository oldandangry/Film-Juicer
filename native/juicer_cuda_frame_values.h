#ifndef FJ_CUDA_FRAME_VALUES_H
#define FJ_CUDA_FRAME_VALUES_H

#include "juicer_legacy_api.h"

/* Temporary S4.E preparation boundary. All access and backing expire at return. */
typedef struct FjPreparedFramePolicy {
    FjFrameGeometry geometry;
    FjOptics optics;
    uint64_t recipe_hash;
} FjPreparedFramePolicy;

typedef struct FjGrainCoefficients {
    uint32_t flags;
    uint32_t debug_view;
    size_t sublayer_count;
    float micro_structure[2];
    float clump_temporal_mix;
    float size_mix_scale;
    float amplitude;
    float chroma_mix;
    float chroma_shared_weight;
    float chroma_independent_weight;
    float particle_density_min_cmy[3];
    float uniformity_cmy[3];
    float layer_axis_block_prefix_max[3][16];
} FjGrainCoefficients;

typedef struct FjFrameMathInput {
    FjDirFrameMath dir;
    FjGrainFrameMath grain;
    FjEffectsFrameMath effects;
    FjGrainCoefficients grain_coefficients;
    FjDust film_dust;
    FjDust gate_dust;
    FjScratch film_scratch;
    FjScratch gate_scratch;
} FjFrameMathInput;

typedef struct FjNativeFrameDescriptors {
    FjSpatialDir spatial_dir;
    FjVisualGrain grain;
    FjFilmEffects effects;
} FjNativeFrameDescriptors;

#endif
