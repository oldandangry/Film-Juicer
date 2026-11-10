#ifndef FJ_CUDA_FRAME_VALUES_H
#define FJ_CUDA_FRAME_VALUES_H

#include "juicer_cuda_api.h"

/* Completed Rust values consumed by the synchronous native call. Every span
 * borrows that call; native staging owns queued reads before return. These
 * records contain no native preparation or application-policy owner. */
#define FJ_SCANNER_LUT_REASON_NONE UINT32_C(0)
#define FJ_SCANNER_LUT_REASON_RAW UINT32_C(1)
#define FJ_SCANNER_LUT_REASON_RESOLUTION UINT32_C(2)
#define FJ_SCANNER_LUT_REASON_SIZE UINT32_C(3)
#define FJ_SCANNER_LUT_REASON_ALLOCATION UINT32_C(4)
#define FJ_SCANNER_LUT_REASON_NONFINITE UINT32_C(5)
#define FJ_SCANNER_LUT_REASON_PANIC UINT32_C(6)
#define FJ_GAUSSIAN_KERNEL_NONE UINT32_C(0)
#define FJ_GAUSSIAN_KERNEL_RAW UINT32_C(1)
#define FJ_GAUSSIAN_KERNEL_RADIUS UINT32_C(2)
#define FJ_GAUSSIAN_KERNEL_SIGMA UINT32_C(3)
#define FJ_GAUSSIAN_KERNEL_ALLOCATION UINT32_C(4)
#define FJ_GAUSSIAN_KERNEL_PANIC UINT32_C(5)

typedef struct FjScannerLutInput {
    FjFloatSpan eps_c, eps_m, eps_y;
    FjFloatSpan weighted_x, weighted_y, weighted_z, baseline_min;
    float range_min_cmy[3], inverse_span_cmy[3];
    int32_t sample_count;
    float inv_yn;
    uint32_t has_baseline, medium, resolution;
} FjScannerLutInput;

typedef struct FjScannerLutView {
    uint32_t resolution;
    FjFloatSpan log2_xyz, slope_c, slope_m, slope_y, cell_min, cell_max;
} FjScannerLutView;

typedef struct FjScannerLutFailure {
    uint32_t reason, channel;
    uint64_t index;
    uint32_t buffer;
} FjScannerLutFailure;

typedef struct FjGaussianKernelInput {
    int32_t radius;
    float sigma_px;
} FjGaussianKernelInput;

typedef struct FjGaussianKernelView {
    int32_t radius;
    float sigma_px;
    FjFloatSpan weights;
} FjGaussianKernelView;

typedef struct FjGaussianKernelFailure {
    uint32_t reason;
} FjGaussianKernelFailure;

typedef struct FjFrameGeometry {
    FjExtent full_frame;
    float pixel_size_um;
} FjFrameGeometry;

typedef struct FjDirFrameMath {
    uint32_t present, component_count;
    uint64_t recipe_hash;
    FjExtent render_extent, full_frame_extent, filter_domain_extent;
    FjDirGaussian components[4];
} FjDirFrameMath;

typedef struct FjGrainFrameMath {
    uint32_t present;
    uint32_t capture_polarity;
    FjExtent render_extent;
    FjExtent full_frame_extent;
    float pixel_size_um;
    int64_t frame0;
    float frame_alpha;
    uint64_t seed_base;
    uint64_t seed_base_next;
    uint64_t session_seed;
    uint64_t clip_token;
    int32_t pitch_px;
    int32_t breathing_period_frames;
    int32_t clump_morph_period_frames;
    float wang_cell_mm;
    float breathing_amplitude;
    float breathing_cell_um_small;
    float breathing_cell_um_large;
    float breathing_mix;
    float breathing_drift_um_per_frame;
    float debug_scale;
    float density_max_cmy[3];
    float n_particles_cmy[3];
    float od_particle_cmy[3];
    float density_min_layers[3][3];
    float density_max_layers[3][3];
    float n_particles_layers[3][3];
    float od_particle_layers[3][3];
    FjGrainGaussian correlation[3];
    FjGrainGaussian dye_cloud[3][3];
    float size_mix_weights[3];
    float size_mix_gain;
    uint32_t requires_full_frame;
    uint64_t recipe_hash;
    uint64_t density_layers_hash;
} FjGrainFrameMath;

typedef struct FjEffectsFrameMath {
    uint32_t present;
    uint32_t flags;
    FjExtent render_extent;
    FjExtent full_frame_extent;
    FjDefectOrigin origins[4];
    float sample_step_x_mm;
    float sample_step_y_mm;
    int32_t roi_offset_x;
    int32_t roi_offset_y;
    int32_t gate_width;
    int32_t gate_height;
    uint64_t session_seed;
    uint64_t clip_token;
    float weave_dx_px;
    float weave_dy_px;
    float weave_cos_rot;
    float weave_sin_rot;
    uint64_t recipe_hash;
    uint64_t hash;
} FjEffectsFrameMath;


/* All access and backing expire at synchronous call return. */
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
