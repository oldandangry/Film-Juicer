#ifndef FJ_CUDA_API_H
#define FJ_CUDA_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private in-process ABI; C and Rust ship together. No packing or version negotiation.
 * All records are non-owning values except the opaque FjCuda handle. No padding is
 * hashed. Every host span expires at native-call return: native code must copy into
 * its own staging before returning if a GPU transfer remains outstanding.
 * Empty spans are (NULL, 0); nonempty counts denote initialized elements, not bytes.
 * Arrays are row-major unless an individual field states otherwise. */
typedef struct FjCuda FjCuda;

#define FJ_STATUS_SUCCESS 0U
#define FJ_STATUS_CANCELLED 1U
#define FJ_STATUS_UNSUPPORTED_INPUT 2U
#define FJ_STATUS_PREPARATION_FAILURE 3U
#define FJ_STATUS_ALLOCATION_FAILURE 4U
#define FJ_STATUS_CUDA_FAILURE 5U
#define FJ_STATUS_CUFFT_FAILURE 6U
#define FJ_STATUS_CONTEXT_LOSS 7U
#define FJ_STATUS_INTERNAL_FAILURE 8U
#define FJ_API_NONE 0U
#define FJ_API_CUDA_RUNTIME 1U
#define FJ_API_CUDA_DRIVER 2U
#define FJ_API_CUFFT 3U

typedef struct FjStatus {
    uint32_t category; /* FJ_STATUS_* */
    uint32_t api;      /* FJ_API_*; NONE requires native_code == 0. */
    int32_t native_code;
} FjStatus;

typedef struct FjErrorBuffer {
    /* char[capacity], diagnostic bytes produced by the native entry for its caller.
     * No units/channel order. Writable until native-call return; never retained.
     * capacity == 0 permits NULL. Otherwise reserve and write a trailing NUL.
     * length excludes that NUL; truncation changes text only, never FjStatus. */
    char* data;
    size_t capacity;
    size_t length;
} FjErrorBuffer;

typedef struct FjStringView {
    /* char[count], length-delimited UTF-8 bytes (no terminator required).
     * Producer: module bootstrap or selected profile identity. Consumer: native
     * create/diagnostics. No physical units; expires at native-call return. */
    const char* data;
    size_t count;
} FjStringView;

typedef struct FjFloatSpan {
    /* float[count]; each containing field specifies shape/order/units and its
     * producer/consumer. Borrowed and immutable until native-call return. */
    const float* data;
    size_t count;
} FjFloatSpan;

typedef struct FjByteSpan {
    /* uint8_t[count]; each containing field specifies shape/order and its
     * producer/consumer. Borrowed and immutable until native-call return. */
    const uint8_t* data;
    size_t count;
} FjByteSpan;

/* Device/context/stream addresses are integer tokens, never host spans or owners. */
typedef struct FjCudaContext {
    int32_t device_id;
    uintptr_t context;
} FjCudaContext;

typedef struct FjRect {
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
} FjRect;

typedef struct FjExtent {
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
} FjExtent;

#define FJ_COMPONENTS_UNKNOWN 0U
#define FJ_COMPONENTS_RGB 3U
#define FJ_COMPONENTS_RGBA 4U
#define FJ_DEPTH_UNKNOWN 0U
#define FJ_DEPTH_FLOAT32 1U
#define FJ_FRAME_STREAM_PRESENT 1U
#define FJ_FRAME_TRACE_INFO 2U
#define FJ_FRAME_TRACE_VERBOSE 4U

typedef struct FjImage {
    uintptr_t address; /* Image base at its own bounds origin; device storage. */
    FjRect bounds;     /* Pixel coordinates; exclusive x2/y2. */
    ptrdiff_t row_bytes;
    uint32_t components; /* FJ_COMPONENTS_* */
    uint32_t depth;      /* FJ_DEPTH_* */
} FjImage;

typedef struct FjEffectsGeometry {
    FjExtent pixel_definition;
    double canonical_x;
    double canonical_y;
    double canonical_width;
    double canonical_height;
    double scale_x;
    double scale_y;
    double pixel_aspect_ratio;
} FjEffectsGeometry;

typedef struct FjFrame {
    FjImage source;
    FjImage destination;
    FjRect render_window;
    FjRect full_frame_extent;
    uintptr_t stream; /* Zero value is independent of FJ_FRAME_STREAM_PRESENT. */
    uint32_t flags;   /* FJ_FRAME_* */
    float pixel_size_um;
    double time_frames;
    double frame_rate;
    uint64_t session_seed;
    uint64_t clip_token;
    FjEffectsGeometry effects_geometry;
} FjFrame;

typedef struct FjSubmission {
    uint64_t instance_token;
    uint64_t frame_token;
    uint64_t submission_id;
    /* ResourceManager::KeyDigests in its existing order; epoch stays native. */
    uint64_t upload_core_hash;
    uint64_t dir_hash;
    uint64_t scanner_hash;
    uint64_t auto_exposure_hash;
} FjSubmission;

#define FJ_ABORT_CONTINUE 0U
#define FJ_ABORT_REQUESTED 1U
/* Nullable query; non-null is invoked synchronously on the calling thread only.
 * user is solely callback-local abort access, never a model or resource pointer.
 * Both expire at native-call return. No reentry, retention, exception or unwind.
 * A Rust panic must be caught in the callback and recorded locally as failure. */
typedef uint32_t (*FjAbortQuery)(void* user);
typedef struct FjAbortCallback {
    FjAbortQuery query;
    void* user;
} FjAbortCallback;

#define FJ_ROUTE_NEGATIVE_DIRECT 0U
#define FJ_ROUTE_NEGATIVE_PRINT 1U
#define FJ_ROUTE_POSITIVE_DIRECT 2U
#define FJ_ROUTE_POSITIVE_PRINT 3U
#define FJ_POLARITY_NEGATIVE 0U
#define FJ_POLARITY_POSITIVE 1U
#define FJ_POLARITY_UNSUPPORTED 2U
#define FJ_RAW_HANATOS_2025 0U
#define FJ_RAW_MALLETT_2019 1U
#define FJ_RAW_ARCTIC_2026_BETA04 2U
#define FJ_INPUT_DWG 0U
#define FJ_INPUT_BT2020 1U
#define FJ_INPUT_ACES2065_1 2U
#define FJ_INPUT_SRGB_REC709 3U
#define FJ_FILM_DECODE_CCTF 1U
#define FJ_FILM_ADAPT_INPUT 2U
#define FJ_FILM_AUTO_EXPOSURE 4U
#define FJ_METER_CENTER_WEIGHTED 0U
#define FJ_METER_AVERAGE 1U
#define FJ_METER_MEDIAN 2U /* Retains the current deliberate rejection. */
#define FJ_METER_PARTIAL 3U
#define FJ_METER_MATRIX 4U
#define FJ_METER_MULTI_ZONE 5U
#define FJ_METER_HIGHLIGHT_WEIGHTED 6U

typedef struct FjAutoExposure {
    FjRect source_bounds;
    FjRect meter_bounds;
    int32_t preview_width;
    int32_t preview_height;
    uint32_t method; /* FJ_METER_* */
    uint64_t hash;
} FjAutoExposure;

typedef struct FjFilmExposure {
    uint32_t input_color_space; /* FJ_INPUT_* */
    uint32_t method;            /* FJ_RAW_* */
    uint32_t flags;             /* FJ_FILM_* */
    float manual_exposure_ev;
    float route_correction_scale;
    float mallett_green_midgray_scale;
    float input_rgb_to_xyz[9];
    float input_xyz_adapt[9];
    float xyz_to_linear_srgb[9];
    /* float[81*3], wavelength-major RGB, relative linear sensitivity.
     * FilmRawRecipe::finalSensitivity -> prepare_focused_route_resources (B/G/R
     * curve split); expires at native-call return. */
    FjFloatSpan sensitivity_rgb;
    /* float[192*192*4] or empty, FilmTcLut::rgba C-order [tc0][tc1][R,G,B,padding].
     * FocusedRenderPayload -> film TC-LUT upload; integrated reconstruction
     * values, no physical units; expires at native-call return. */
    FjFloatSpan tc_lut_rgba;
    /* float[81] or empty, ascending 380..780 nm in 5 nm steps, relative SPD.
     * FocusedRenderPayload::exposureTables.illum -> Mallett upload;
     * expires at native-call return. */
    FjFloatSpan mallett_illuminant;
    /* float[81*3] or empty, [wavelength][basis R/G/B], relative basis weights.
     * SpectralContext::mallettBasis -> Mallett upload;
     * expires at native-call return. */
    FjFloatSpan mallett_basis;
    uint64_t sensitivity_hash;
    uint64_t tc_lut_hash;
    uint64_t hash;
} FjFilmExposure;

typedef struct FjFilmDevelopment {
    /* float[N], ascending log10 exposure; FilmDevelopRecipe::logExposure ->
     * focused density-curve upload; expires at native-call return. */
    FjFloatSpan log_exposure;
    /* float[N*3], [sample][RGB] optical density; normalizedDensityCurves ->
     * focused density-curve B/G/R split; expires at native-call return. */
    FjFloatSpan density_rgb;
    /* Nine separate float[N] spans, [layer 0..2][channel 0..2], optical density.
     * Channel indices are unchanged: R/G/B sensitivity, C/M/Y at grain consumption.
     * FilmDevelopRecipe::densityCurvesLayers -> grain layer upload, without
     * flattening the nine owners; empty when unused; expires at native-call return. */
    FjFloatSpan density_layers[3][3];
    float gamma_rgb[3];
    uint64_t density_curves_hash;
    uint64_t density_layers_hash;
    uint64_t hash;
} FjFilmDevelopment;

#define FJ_DIR_INACTIVE 0U
#define FJ_DIR_NEGATIVE_DONOR_LANGMUIR 1U
#define FJ_DIR_POSITIVE_RECEIVER_LANGMUIR 2U
typedef struct FjDirCouplers {
    uint32_t mode;       /* FJ_DIR_*; inactive means no compensated spans. */
    float matrix_rgb[9]; /* [donor RGB][receiver RGB]; packer maps to BGR. */
    float density_max_rgb[3];
    float density_ref_rgb[3];
    float donor_k_rgb[3];
    float receiver_c_ref_rgb[3];
    float receiver_kr_rgb[3];
    /* Three float[N] spans, [RGB][sample], compensated log10 exposure axes.
     * DirCouplersRecipe -> focused DIR curve upload (B/G/R split).
     * Empty when inactive; expires at native-call return. */
    FjFloatSpan compensated_axes_rgb[3];
    uint64_t compensated_axes_hash;
    uint64_t hash;
} FjDirCouplers;

/* DensityBoundsRecipe -> scanner/enlarger payload; unmodified optical-density
 * minima and inverse spans in CMY. Direct-route sign convention is applied by
 * native packing, exactly as in prepare_focused_route_resources. */
typedef struct FjDensityBounds {
    float min_cmy[3];
    float inv_span_cmy[3];
    uint64_t hash;
} FjDensityBounds;

typedef struct FjScannerSpectra {
    /* Each float[81], wavelength order 380..780 nm, 5 nm steps, Y/M/C optical
     * extinction. FocusedRenderPayload::scannerTables.epsY/M/C ->
     * Scanner::spectral_to_log_xyz; expires at native-call return. */
    FjFloatSpan dye_y;
    FjFloatSpan dye_m;
    FjFloatSpan dye_c;
    /* Each float[81], same wavelength order, illuminant-weighted XYZ CMFs.
     * scannerTables.Ax/Ay/Az -> canonical scan-LUT host build;
     * relative tristimulus weights; expires at native-call return. */
    FjFloatSpan weighted_x;
    FjFloatSpan weighted_y;
    FjFloatSpan weighted_z;
    /* float[81] optical density in wavelength order, or empty for no baseline.
     * scannerTables.baseDensityMin -> Scanner::spectral_to_log_xyz;
     * expires at native-call return. */
    FjFloatSpan base_density;
    float inverse_y_normalization;
} FjScannerSpectra;

/* Consumed ScannerSpectralLutDescriptor identity and resolution. Route/medium/
 * polarity come from FjPreparedHostData; hash constituent metadata stays upstream. */
typedef struct FjScannerLut {
    uint64_t density_bounds_hash;
    uint32_t resolution;
    uint64_t hash;
} FjScannerLut;

#define FJ_CORRECTION_ACTIVE 1U
typedef struct FjScannerCorrection {
    uint32_t flags; /* FJ_CORRECTION_* */
    float xyz_slope;
    float xyz_offset;
    float exposure_scale;
} FjScannerCorrection;

#define FJ_POST_GLARE 1U
typedef struct FjScannerPostEffects {
    uint32_t flags; /* FJ_POST_* */
    float glare_percent;
    float glare_roughness;
    float glare_blur_sigma_px;
    int32_t glare_blur_radius;
    float lens_blur_sigma_px;
    int32_t lens_blur_radius;
    float unsharp_sigma_px;
    int32_t unsharp_radius;
    float unsharp_amount;
    uint64_t hash;
} FjScannerPostEffects;

#define FJ_OUTPUT_SRGB 0U
#define FJ_OUTPUT_DCI_P3 1U
#define FJ_OUTPUT_DISPLAY_P3 2U
#define FJ_OUTPUT_ADOBE_RGB 3U
#define FJ_OUTPUT_BT2020 4U
#define FJ_OUTPUT_PROPHOTO_RGB 5U
#define FJ_OUTPUT_ACES2065_1 6U
#define FJ_OUTPUT_DWG_INTERMEDIATE 7U
#define FJ_OUTPUT_REC709 8U
#define FJ_COLOR_ENCODE_CCTF 1U
#define FJ_COLOR_INPUT_IS_OUTPUT_SPACE 2U
#define FJ_COLOR_GAMUT_COMPRESSION 4U
typedef struct FjOutputColor {
    float cat02[9];
    float xyz_to_rgb[9];
    float illuminant_xyz[3];
    uint32_t color_space; /* FJ_OUTPUT_* */
    uint32_t flags;       /* FJ_COLOR_* */
    float native_rgb_to_d65_xyz[9];
    float d65_xyz_to_native_rgb[9];
    float lightness_knee[3]; /* threshold, limit, power */
    float chroma_knee[3];    /* threshold, limit, power */
    /* float[64*720] or empty, [lightness][hue], maximum OkLab chroma.
     * Gamut::OutputBoundaryTable::cmax -> output gamut upload;
     * expires at native-call return. */
    FjFloatSpan gamut_cmax;
    uint64_t gamut_table_hash;
    uint64_t gamut_recipe_hash;
} FjOutputColor;

#define FJ_PRINT_PREFLASH 1U
typedef struct FjPrint {
    /* float[81*3], [wavelength][CMY] optical extinction, and float[81] base
     * optical density, ascending canonical wavelength order. ValidatedFilmProfile
     * -> prepare_print_resources film-density upload; expires at native-call return. */
    FjFloatSpan film_density_cmy;
    FjFloatSpan film_base_density;
    /* float[81*3], [wavelength][CMY], relative linear sensitivity.
     * ValidatedPrintProfile::linearSensitivity -> print sensitivity upload;
     * expires at native-call return. */
    FjFloatSpan sensitivity_cmy;
    /* float[N], ascending log10 exposure, and float[N*3], [sample][CMY] density.
     * ValidatedPrintProfile::logExposure / PrintRecipe::develop.densityCurves ->
     * print development upload; expires at native-call return. */
    FjFloatSpan log_exposure;
    FjFloatSpan density_cmy;
    /* Each float[81], canonical wavelength order, relative filtered SPD.
     * FocusedRenderPayload / print illuminant derivation -> print illuminant
     * upload. Preflash is empty when disabled; expires at native-call return. */
    FjFloatSpan main_illuminant;
    FjFloatSpan preflash_illuminant;
    uint32_t flags; /* FJ_PRINT_* */
    float main_cc_cmy[3];
    float preflash_cc_cmy[3];
    float exposure;
    float preflash_exposure;
    float preflash_raw_cmy[3]; /* derive_preflash_raw result; relative exposure. */
    float normalizer;          /* PrintRecipe::balance.normalizer, dimensionless. */
    uint64_t film_density_hash;
    uint64_t profile_tables_hash;
    uint64_t main_illuminant_hash;
    uint64_t preflash_illuminant_hash;
    uint64_t preflash_raw_hash;
    uint64_t balance_hash;
    uint64_t hash;
} FjPrint;

#define FJ_DIFFUSION_GLIMMERGLASS 0U
#define FJ_DIFFUSION_BLACK_PRO_MIST 1U
#define FJ_DIFFUSION_PRO_MIST 2U
#define FJ_DIFFUSION_CINEBLOOM 3U
#define FJ_OPTICS_CAMERA_DIFFUSION 1U
#define FJ_OPTICS_ENLARGER_DIFFUSION 2U
#define FJ_OPTICS_SCATTER_HALATION 4U

typedef struct FjDiffusion {
    uint32_t family; /* FJ_DIFFUSION_* */
    double scatter_fraction;
    double group_weights[3];   /* core, halo, bloom */
    double group_lambda_um[3]; /* core, halo, bloom */
    double warmth;
    double spatial_scale;
    double pixel_size_um;
    int32_t radius_px;
    uint64_t sample_hash;
    uint64_t hash;
} FjDiffusion;

#define FJ_HALATION_IDENTITY 0U
#define FJ_HALATION_FIR_REFLECT 1U
#define FJ_HALATION_YVV_REPLICATE 2U
typedef struct FjHalationGaussian {
    uint32_t kind; /* FJ_HALATION_* */
    int32_t radius;
    float fir_weights[19];
    float feedforward;
    float feedback[3]; /* B1, B2, B3 */
} FjHalationGaussian;

typedef struct FjHalationChannel {
    FjHalationGaussian core;
    FjHalationGaussian tail[3];
    FjHalationGaussian bounce[3];
    float total_strength;
} FjHalationChannel;

typedef struct FjScatterHalation {
    uint64_t recipe_hash;
    float scatter_amount;
    FjHalationChannel channels[3]; /* RGB, matching the frame descriptor. */
} FjScatterHalation;

typedef struct FjOptics {
    uint32_t flags; /* FJ_OPTICS_* */
    FjExtent full_frame;
    FjDiffusion camera;
    FjDiffusion enlarger;
    uint64_t diffusion_hash;
    FjScatterHalation scatter_halation;
} FjOptics;

#define FJ_DIR_OPERATOR_NONE 0U
#define FJ_DIR_OPERATOR_IDENTITY 1U
#define FJ_DIR_OPERATOR_FIR_REFLECT 2U
#define FJ_DIR_OPERATOR_YVV_REFLECT 3U
typedef struct FjDirGaussian {
    float sigma_px;
    float weight;
    int32_t radius;
    uint32_t reference_operator; /* FJ_DIR_OPERATOR_* */
    double feedforward;
    double feedback[3];
    double normalization_denominator;
    double feedforward_numerator;
    double feedback_numerators[3];
    double boundary_truncation_accuracy;
    double boundary_certification_tolerance;
    uint32_t boundary_derivation_version;
} FjDirGaussian;

/* Resolved host operator only. Scratch tiers, roles, aliases and allocation
 * choices are deliberately absent; they remain native execution policy. */
typedef struct FjSpatialDir {
    FjExtent render_extent;
    FjExtent full_frame_extent;
    FjExtent filter_domain_extent;
    size_t component_count; /* 0..4 initialized entries in components. */
    FjDirGaussian components[4];
    uint64_t recipe_hash;
    uint64_t hash;
} FjSpatialDir;

#define FJ_GRAIN_ACTIVE 1U
#define FJ_GRAIN_SUBLAYERS 2U
#define FJ_GRAIN_FULL_FRAME 4U
#define FJ_GRAIN_AXIS_FINITE_C 8U
#define FJ_GRAIN_AXIS_FINITE_M 16U
#define FJ_GRAIN_AXIS_FINITE_Y 32U
#define FJ_GRAIN_DEBUG_OFF 0U
#define FJ_GRAIN_DEBUG_DELTA_MIX 1U
#define FJ_GRAIN_DEBUG_DELTA_FINE 2U
#define FJ_GRAIN_DEBUG_DELTA_COARSE 3U
#define FJ_GRAIN_DEBUG_DELTA_FINE_RAW 4U
#define FJ_GRAIN_DEBUG_DELTA_COARSE_RAW 5U
#define FJ_GRAIN_DEBUG_MEAN_DENSITY 6U

typedef struct FjGrainGaussian {
    float sigma_px;
    int32_t radius;
    uint64_t hash;
} FjGrainGaussian;

/* VisualGrainFrameDescriptor and consumed VisualGrainRecipe coefficients ->
 * pack_visual_grain_payload. Layer arrays keep [layer][CMY]; no BGR conversion.
 * Scratch shape is native-derived, never a foreign allocation instruction. */
typedef struct FjVisualGrain {
    uint32_t flags;      /* FJ_GRAIN_* except DEBUG tags. */
    uint32_t debug_view; /* FJ_GRAIN_DEBUG_* */
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
    FjGrainGaussian correlation[3]; /* fine, mid, coarse */
    FjGrainGaussian dye_cloud[3][3];
    float size_mix_weights[3]; /* fine, mid, coarse */
    float size_mix_gain;
    size_t sublayer_count;
    float micro_structure[2]; /* clump cell size (um), clump sigma (x1e-3) */
    float clump_temporal_mix;
    float size_mix_scale;
    float amplitude;
    float chroma_mix;
    float chroma_shared_weight;
    float chroma_independent_weight;
    float particle_density_min_cmy[3];
    float uniformity_cmy[3];
    float layer_axis_block_prefix_max[3][16]; /* CMY, axis block */
    uint64_t recipe_hash;
    uint64_t density_layers_hash;
    uint64_t hash;
} FjVisualGrain;

/* Fixed resolved defect coefficients consumed by pack_film_juicer_effects_payload.
 * Fractions/probabilities/strengths are dimensionless; lengths are millimeters,
 * angles radians. Geometry remains on the virtual film strip or fixed gate. */

typedef struct FjDust {
    float cell_width_mm;
    float cell_height_mm;
    float slot_probability;
    float softness_min_mm;
    float softness_max_mm;
    float softness_size_cap_fraction;
    float support_x_mm;
    float support_y_mm;
    float fiber_fraction;
    float fiber_drift_fraction;
    float fiber_first_knot_min;
    float fiber_first_knot_max;
    float fiber_second_knot_min;
    float fiber_second_knot_max;
    float fiber_interior_width_min_fraction;
    float fiber_interior_width_max_fraction;
    float diameter_min_mm;
    float diameter_bulk_max_mm;
    float diameter_max_mm;
    float diameter_tail_fraction;
    float fiber_length_min_mm;
    float fiber_length_max_mm;
    float fiber_width_min_mm;
    float fiber_width_max_mm;
    float opacity_faint_cumulative;
    float opacity_intermediate_cumulative;
    float compact_opacity_min;
    float compact_opacity_faint_end;
    float compact_opacity_intermediate_end;
    float compact_opacity_max;
    float fiber_opacity_min;
    float fiber_opacity_faint_end;
    float fiber_opacity_intermediate_end;
    float fiber_opacity_max;
    float compact_dominant_aspect_min;
    float compact_dominant_aspect_max;
    float compact_subsidiary_scale_min;
    float compact_subsidiary_scale_max;
    float compact_subsidiary_aspect_min;
    float compact_subsidiary_aspect_max;
    float compact_subsidiary_offset_max;
    float compact_subsidiary_angle_max_radians;
} FjDust;

typedef struct FjScratch {
    float cell_width_mm;
    float cell_height_mm;
    float slot_probability;
    float softness_min_mm;
    float softness_max_mm;
    float softness_size_cap_fraction;
    float support_x_mm;
    float support_y_mm;
    float length_min_mm;
    float length_bulk_max_mm;
    float length_max_mm;
    float length_tail_fraction;
    float width_min_mm;
    float width_bulk_max_mm;
    float width_max_mm;
    float width_tail_fraction;
    float drift_fraction;
    float first_knot_min;
    float first_knot_max;
    float second_knot_min;
    float second_knot_max;
    float interior_width_min_fraction;
    float interior_width_max_fraction;
    float interior_depth_min_fraction;
    float interior_depth_max_fraction;
    float endpoint_abrupt_probability;
    float interruption_probability;
    float gap_center_min;
    float gap_center_max;
    float gap_span_min;
    float gap_span_max;
    float scuff_probability;
    float scuff_length_max_mm;
    float scuff_angle_max_radians;
    float strength_min;
    float strength_max;
} FjScratch;

typedef struct FjDefectOrigin {
    int64_t cell_x;
    int64_t cell_y;
    float local_x_mm;
    float local_y_mm;
} FjDefectOrigin;

#define FJ_EFFECTS_WEAVE 1U
#define FJ_EFFECTS_FILM 2U
#define FJ_EFFECTS_GATE_TRANSMITTANCE 4U
#define FJ_EFFECTS_GATE_OUTPUT 8U
#define FJ_EFFECTS_FULL_FRAME 16U
typedef struct FjFilmEffects {
    FjExtent render_extent;
    FjExtent full_frame_extent;
    FjDust film_dust;
    FjScratch film_scratch;
    FjDust gate_dust;
    FjScratch gate_scratch;
    FjDefectOrigin origins[4]; /* film dust, film scratch, gate dust, gate scratch */
    float sample_step_x_mm;
    float sample_step_y_mm;
    int32_t roi_offset_x;
    int32_t roi_offset_y;
    int32_t gate_width;
    int32_t gate_height;
    uint64_t session_seed;
    uint64_t clip_token;
    uint32_t flags; /* FJ_EFFECTS_* */
    float weave_dx_px;
    float weave_dy_px;
    float weave_cos_rot;
    float weave_sin_rot;
    uint64_t recipe_hash;
    uint64_t hash;
} FjFilmEffects;

typedef struct FjStaticNoise {
    /* uint8_t[width*height*frames], [frame][y][x], dimensionless STBN samples.
     * ResourceAssetLibrary::static_noise_payloads -> static-grain upload;
     * empty when grain is inactive; expires at native-call return. */
    FjByteSpan stbn;
    int32_t stbn_width;
    int32_t stbn_height;
    int32_t stbn_frames;
    /* uint8_t[width*height*tile_count], [tile][y][x], dimensionless noise.
     * ResourceAssetLibrary::static_noise_payloads -> Wang tile upload;
     * empty when grain is inactive; expires at native-call return. */
    FjByteSpan wang_tiles;
    /* uint8_t[colors^4], index (((L*colors+R)*colors+T)*colors+B), tile indices.
     * ResourceAssetLibrary::static_noise_payloads -> Wang LUT upload;
     * empty when grain is inactive; expires at native-call return. */
    FjByteSpan wang_lut;
    int32_t wang_width;
    int32_t wang_height;
    size_t wang_tile_count;
    int32_t wang_colors;
} FjStaticNoise;

/* Complete invocation-local host input. Fixed records copy resolved coefficients;
 * spans borrow immutable producer storage. Unselected families are zero/empty.
 * Presence is explicit in family flags (print is selected by route); no model
 * pointer, native resource handle, or backend workspace policy crosses here. */
typedef struct FjPreparedHostData {
    uint32_t route;            /* FJ_ROUTE_* */
    uint32_t capture_polarity; /* FJ_POLARITY_* */
    /* UTF-8 byte spans from ProfileRoute -> native failure diagnostics;
     * no physical units; expires at native-call return. */
    FjStringView film_profile_key;
    FjStringView print_profile_key;
    uint64_t film_profile_asset_version;
    uint64_t print_profile_asset_version;
    uint64_t recipe_hash;
    FjFilmExposure film_exposure;
    FjAutoExposure auto_exposure;
    FjFilmDevelopment film_development;
    FjDirCouplers dir_couplers;
    FjDensityBounds scanner_bounds;
    FjDensityBounds enlarger_film_bounds;
    FjScannerSpectra scanner_spectra;
    FjScannerLut scanner_lut;
    FjScannerCorrection scanner_correction;
    FjScannerPostEffects scanner_post;
    FjOutputColor output_color;
    FjPrint print;
    FjOptics optics;
    FjSpatialDir spatial_dir;
    FjVisualGrain grain;
    FjFilmEffects effects;
    FjStaticNoise noise;
} FjPreparedHostData;

/* Declarations only until their implementation packages are accepted.
 * Calls serialize on one owner gate. inspect through render stays on the same
 * callback thread with its host context current and images/stream still leased.
 * No function resets a live host-owned CUDA context; no exception may escape.
 * Non-null record pointers denote one live initialized record for this call. */

/* data_directory: module bootstrap -> native Root bootstrap, copied if retained.
 * Host metadata only: no CUDA discovery. Success publishes exactly one owner in
 * out_cuda; failure publishes none. A duplicate active runtime is rejected. */
FjStatus fj_cuda_create(FjStringView data_directory, FjCuda** out_cuda, FjErrorBuffer* error);

/* Borrows owner/frame; writes exact discovered identity on success only.
 * Coverage, supported strides/aliases and stream completion are qualified by
 * the inspection package before this boundary can be used for production. */
FjStatus fj_cuda_inspect(FjCuda* cuda, const FjFrame* frame, FjCudaContext* out_context, FjErrorBuffer* error);

/* All prepared spans and abort access expire at return, even on failure.
 * Enqueue/finish/abort and outstanding native-owned staging remain native.
 * This declaration does not establish that the old direct-upload fallback
 * already satisfies foreign-span expiry; that must be closed before cutover. */
FjStatus fj_cuda_render(FjCuda* cuda, const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, FjAbortCallback abort_callback, FjErrorBuffer* error);

/* Retires only this instance's static-grain membership; borrows the owner. */
FjStatus fj_cuda_retire_instance(FjCuda* cuda, uint64_t instance_token, FjErrorBuffer* error);

/* Borrows: success leaves a closed caller-owned handle, failure a blocked
 * caller-owned handle retaining the graph. Already-closed shutdown is harmless. */
FjStatus fj_cuda_shutdown(FjCuda* cuda, FjErrorBuffer* error);

/* Consumes exactly once on EVERY outcome. Clear the owning pointer before the
 * call. An accepting owner gets one close attempt; blocked/uncertain graphs
 * remain under native quarantine without retry, allocation or destructor walk.
 * Failed native retention never extends host image/context/module lifetime. */
FjStatus fj_cuda_destroy(FjCuda* cuda, FjErrorBuffer* error);

#ifdef __cplusplus
}
#endif

#endif
