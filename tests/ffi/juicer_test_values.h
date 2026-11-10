#pragma once

#include <stdint.h>

#include "juicer_cuda_api.h"
#include "juicer_cuda_frame_values.h"
#include "juicer_dir_boundary_api.h"
#include "juicer_diffusion_api.h"

/* Test-only C values for the Rust FFI fixtures. Native frame, CUDA, DIR
 * boundary and diffusion records come from the included native headers. Raw
 * values are used synchronously; owner and borrowed-view expiry is specified
 * by the corresponding test facade declarations. This header declares no API. */
#ifdef __cplusplus
extern "C" {
#endif

#define FJ_LEGACY_ROUTE_SUCCESS UINT32_C(0)

#define FJ_LEGACY_ROUTE_INVALID_POLARITY UINT32_C(1)

#define FJ_LEGACY_ROUTE_INVALID_SELECTION UINT32_C(2)

#define FJ_LEGACY_ROUTE_NULL_OUTPUT UINT32_C(3)

#define FJ_LEGACY_ROUTE_INTERNAL_PANIC UINT32_C(4)

#define FJ_LEGACY_POLARITY_NEGATIVE UINT8_C(0)

#define FJ_LEGACY_POLARITY_POSITIVE UINT8_C(1)

#define FJ_LEGACY_POLARITY_UNSUPPORTED UINT8_C(2)

#define FJ_LEGACY_ROUTE_NEGATIVE_DIRECT UINT8_C(0)

#define FJ_LEGACY_ROUTE_NEGATIVE_PRINT UINT8_C(1)

#define FJ_LEGACY_ROUTE_POSITIVE_DIRECT UINT8_C(2)

#define FJ_LEGACY_ROUTE_POSITIVE_PRINT UINT8_C(3)

typedef struct FjInputHullSource {
    FjFloatSpan x, y, z, d65;
} FjInputHullSource;

typedef struct FjInputHull {
    float center_xy[2];
    float xy[2050];
    uint64_t hash;
} FjInputHull;

typedef struct FjOutputGamutSource {
    uint32_t output_space;
    float rgb_to_xyz[9], xyz_to_rgb[9], white_xyz[3], d65_white_xyz[3];
} FjOutputGamutSource;

typedef struct FjOutputGamutTransform {
    float native_rgb_to_d65_xyz[9], d65_xyz_to_native_rgb[9];
    uint32_t output_space;
    uint64_t hash, contract_hash;
} FjOutputGamutTransform;

typedef struct FjOutputBoundarySource {
    uint32_t output_space;
    float d65_xyz_to_native_rgb[9];
    uint64_t transform_hash;
} FjOutputBoundarySource;

typedef struct FjOutputBoundaryIdentity {
    uint64_t transform_hash, contract_hash, hash;
} FjOutputBoundaryIdentity;

typedef struct FjOutputBoundaryFailure {
    uint32_t requirement, lightness_index, hue_index;
} FjOutputBoundaryFailure;

typedef struct FjSpectralObserver {
    FjFloatSpan x, y, z;
} FjSpectralObserver;

typedef struct FjSpectralInput {
    FjFloatSpan dyes_cmy;
    FjSpectralObserver observer;
    FjFloatSpan illuminant;
    FjFloatSpan baseline_min, baseline_mid;
    uint64_t illuminant_hash;
} FjSpectralInput;

typedef struct FjSpectralTables {
    float lambda_nm[81], illuminant[81];
    float observer_xyz[3][81], weighted_xyz[3][81], dyes_cmy[3][81];
    float baseline_min[81], baseline_mid[81];
    float delta_lambda, inv_yn, white_xyz[3], reference_white_xyz[3];
    uint32_t has_baseline;
    uint64_t illuminant_hash, tables_hash;
} FjSpectralTables;

typedef struct FjSpectralWhiteInput {
    FjSpectralObserver observer;
    FjFloatSpan illuminant;
} FjSpectralWhiteInput;

typedef struct FjSpectralWhite {
    float normalization, white_xyz[3], white_xy[2];
    uint64_t hash;
} FjSpectralWhite;

typedef struct FjSpectralWhiteFailure {
    uint32_t reason;
    double luminance_sum;
} FjSpectralWhiteFailure;

typedef struct FjSpectralSInput {
    FjFloatSpan weighted_x, weighted_y, weighted_z;
} FjSpectralSInput;

typedef struct FjSpectralInverse {
    float matrix[9];
} FjSpectralInverse;

typedef struct FjReferenceWhiteInput {
    FjFloatSpan spectra;
    float white_xyz[3];
    float spectral_blur;
} FjReferenceWhiteInput;

typedef struct FjReferenceWhite {
    float samples[81];
} FjReferenceWhite;

typedef struct FjSensitivityInput {
    FjFloatSpan linear_sensitivity_rgb;
    FjFloatSpan reference_illuminant;
    uint32_t method, band_pass_active, apply_window;
    float uv[3], ir[3];
    FjFloatSpan window_params, reconstructed_reference_white;
} FjSensitivityInput;

typedef struct FjSensitivity {
    float values_rgb[81][3];
    float mallett_green_scale;
    uint64_t hash;
} FjSensitivity;

typedef struct FjSensitivityFailure {
    uint32_t hash_failure, hash_sample_index;
} FjSensitivityFailure;

typedef struct FjOutputColorPreparation {
    uint32_t output_space;
    float rgb_to_xyz[9];
    float xyz_to_rgb[9];
    float white_xyz[3];
    FjOutputEncoding encoding;
    uint64_t entry_hash;
} FjOutputColorPreparation;

typedef struct FjInputColorMatrices {
    float rgb_to_xyz[9];
    float nominal_white_xyz[3];
    float xyz_to_linear_srgb[9];
    float d65_white_xyz[3];
} FjInputColorMatrices;

typedef struct FjInputColorConversion {
    uint32_t input_space;
    uint32_t decode_cctf;
    uint32_t adapt_xyz;
    float rgb_to_xyz[9];
    float xyz_adapt[9];
} FjInputColorConversion;

typedef struct FjMallettMidgrayInput {
    FjInputColorConversion color;
    float xyz_to_linear_srgb[9];
    FjFloatSpan basis_rgb, illuminant, sensitivity_rgb;
} FjMallettMidgrayInput;

typedef struct FjMallettMidgray {
    float midgray_dwg_rgb[3], raw_midgray_bgr[3];
    float raw_green, scale;
} FjMallettMidgray;

typedef struct FjMidgrayNormalization {
    float raw_green, scale;
} FjMidgrayNormalization;

typedef struct FjMallettReferenceInput {
    FjFloatSpan basis_rgb, illuminant, sensitivity_rgb;
    float source, green_scale;
} FjMallettReferenceInput;

typedef struct FjReferenceRaw {
    float rgb[3];
} FjReferenceRaw;

typedef struct FjAssets FjAssets;

typedef struct FjRuntime FjRuntime;

typedef struct FjRuntimeView {
    const FjAssets* assets;
    FjCuda* cuda;
} FjRuntimeView;

typedef struct FjCatalog FjCatalog;

#define FJ_PROFILE_ROLE_FILM UINT32_C(0)

#define FJ_PROFILE_ROLE_PRINT UINT32_C(1)

typedef struct FjCatalogCounts {
    size_t film_count;
    size_t print_count;
} FjCatalogCounts;

typedef struct FjCatalogEntryView {
    FjStringView key;
    FjStringView label;
    uint32_t polarity; /* FJ_POLARITY_NEGATIVE/POSITIVE, mapped explicitly. */
} FjCatalogEntryView;

typedef struct FjNoise FjNoise;

typedef struct FjFilmProfile FjFilmProfile;

#define FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE UINT32_C(0)

#define FJ_FILM_INPUT_ILLUMINANT_D65 UINT32_C(1)

#define FJ_FILM_INPUT_ORIGIN_NONE UINT32_C(0)

#define FJ_FILM_INPUT_ORIGIN_RAW UINT32_C(1)

#define FJ_FILM_INPUT_ORIGIN_RESOURCE UINT32_C(2)

#define FJ_FILM_INPUT_ORIGIN_COLOR UINT32_C(3)

#define FJ_FILM_INPUT_ORIGIN_SENSITIVITY UINT32_C(4)

#define FJ_FILM_INPUT_ORIGIN_IDENTITY UINT32_C(5)

#define FJ_FILM_INPUT_ORIGIN_INTERNAL UINT32_C(6)

#define FJ_FILM_INPUT_REASON_NONE UINT32_C(0)

#define FJ_FILM_INPUT_RAW_RECORDS UINT32_C(1)

#define FJ_FILM_INPUT_RAW_DIAGNOSTIC UINT32_C(2)

#define FJ_FILM_INPUT_RAW_INPUT_SPACE UINT32_C(3)

#define FJ_FILM_INPUT_RAW_METHOD UINT32_C(4)

#define FJ_FILM_INPUT_RAW_METER UINT32_C(5)

#define FJ_FILM_INPUT_RAW_FLAGS UINT32_C(6)

#define FJ_FILM_INPUT_RAW_SPANS UINT32_C(7)

#define FJ_FILM_INPUT_RESOURCE_HANATOS_WINDOW UINT32_C(1)

#define FJ_FILM_INPUT_RESOURCE_HANATOS_SURFACE UINT32_C(2)

#define FJ_FILM_INPUT_RESOURCE_HANATOS_BLUR UINT32_C(3)

#define FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE UINT32_C(4)

#define FJ_FILM_INPUT_RESOURCE_TC_SOURCE UINT32_C(5)

#define FJ_FILM_INPUT_RESOURCE_INPUT_HULL UINT32_C(6)

#define FJ_FILM_INPUT_COLOR_MATRICES UINT32_C(1)

#define FJ_FILM_INPUT_COLOR_CAT16 UINT32_C(2)

#define FJ_FILM_INPUT_COLOR_CAT02 UINT32_C(3)

#define FJ_FILM_INPUT_SENSITIVITY_COMPUTED UINT32_C(1)

#define FJ_FILM_INPUT_SENSITIVITY_BOUNDARY UINT32_C(2)

#define FJ_FILM_INPUT_IDENTITY_REQUIRED UINT32_C(1)

#define FJ_FILM_INPUT_INTERNAL_PANIC UINT32_C(1)

#define FJ_FILM_INPUT_INTERNAL_INTEGRITY UINT32_C(2)

#define FJ_FILM_INPUT_INTERNAL_MATRICES_PANIC UINT32_C(3)

#define FJ_FILM_INPUT_INTERNAL_CAT16_PANIC UINT32_C(4)

#define FJ_FILM_INPUT_INTERNAL_CAT02_PANIC UINT32_C(5)

#define FJ_FILM_INPUT_INTERNAL_SENSITIVITY_PANIC UINT32_C(6)

typedef struct FjFilmInputParams {
    uint32_t input_space;
    uint32_t decode_cctf;
    uint32_t method;
    uint32_t auto_exposure_enabled;
    uint32_t meter;
    float manual_exposure_ev;
    float film_format_long_edge_mm;
    uint32_t filter_override;
    double uv[3];
    double ir[3];
    uint32_t apply_hanatos_window;
    uint32_t apply_hanatos_surface;
    uint32_t input_compression;
} FjFilmInputParams;

typedef struct FjFilmInputResources {
    FjFloatSpan reference_illuminant;
    FjFloatSpan reference_white_xyz;
    FjFloatSpan integrated_d65_xyz;
    FjFloatSpan reconstructed_reference_white;
    uint64_t tc_source_hash;
    uint64_t input_hull_hash;
} FjFilmInputResources;

typedef struct FjFilmInput {
    FjInputColorConversion color;
    float xyz_to_linear_srgb[9];
    float nominal_white_xyz[3];
    float projection_white_xyz[3];
    uint32_t method;
    uint32_t auto_exposure_enabled;
    uint32_t meter;
    float manual_exposure_ev;
    float film_format_long_edge_mm;
    uint32_t band_pass_active;
    float uv[3];
    float ir[3];
    uint32_t apply_hanatos_window;
    uint32_t apply_hanatos_surface;
    float hanatos_spectral_blur;
    float hanatos_window[4];
    float hanatos_surface_rgb[3][15];
    uint32_t projection_illuminant;
    uint32_t input_compression_active;
    uint64_t input_hull_hash;
    uint64_t tc_source_hash;
    float sensitivity_rgb[81][3];
    uint64_t sensitivity_hash;
    float recipe_mallett_green_scale;
    uint64_t tc_hash;
    uint64_t hash;
} FjFilmInput;

typedef struct FjFilmInputFailure {
    uint32_t origin;
    uint32_t reason;
    FjSensitivityFailure sensitivity;
} FjFilmInputFailure;

typedef struct FjPrintProfile FjPrintProfile;

typedef struct FjPrintDensityCurves FjPrintDensityCurves;

#define FJ_ILLUMINANT_NAMED UINT32_C(0)

#define FJ_ILLUMINANT_BLACKBODY UINT32_C(1)

#define FJ_PROFILE_SUPPORT_FILM UINT32_C(0)

#define FJ_PROFILE_SUPPORT_PAPER UINT32_C(1)

#define FJ_PROFILE_STAGE_FILMING UINT32_C(0)

#define FJ_PROFILE_STAGE_PRINTING UINT32_C(1)

typedef struct FjIlluminantView {
    FjStringView label;
    double temperature_kelvin;
    uint32_t kind;
} FjIlluminantView;

typedef struct FjProfileTablesView {
    FjFloatSpan linear_sensitivity_rgb;
    FjFloatSpan channel_density_cmy;
    FjFloatSpan base_density;
    FjFloatSpan log_exposure;
    FjFloatSpan density_curves_cmy;
} FjProfileTablesView;

typedef struct FjFilmDigest {
    float gamma_samelayer_rgb[3];
    float gamma_interlayer_r_to_gb[2];
    float gamma_interlayer_g_to_rb[2];
    float gamma_interlayer_b_to_rg[2];
    float halation_first_sigma_um[3];
    float halation_primary_amount[3];
    float hanatos_spectral_gaussian_blur_default;
} FjFilmDigest;

typedef struct FjFilmProfileView {
    FjStringView stock;
    FjIlluminantView reference_illuminant;
    FjIlluminantView viewing_illuminant;
    FjProfileTablesView tables;
    FjFloatSpan wavelengths;
    FjFloatSpan density_curves_layers[3][3];
    FjFilmDigest digest;
    FjFloatSpan hanatos_window;
    FjFloatSpan hanatos_surface_rgb;
    uint64_t asset_token;
    uint32_t support;
    uint32_t stage;
    uint32_t polarity;
} FjFilmProfileView;

typedef struct FjPrintProfileView {
    FjStringView stock;
    FjIlluminantView viewing_illuminant;
    FjProfileTablesView tables;
    uint64_t asset_token;
    uint32_t stage;
} FjPrintProfileView;

typedef struct FjPrintDensityView {
    FjFloatSpan totals_cmy;
    uint64_t hash;
} FjPrintDensityView;

typedef struct FjFilmDevelopmentOwner FjFilmDevelopmentOwner;

typedef struct FjFilmDevelopmentView {
    uint32_t route;
    uint32_t polarity;
    uint32_t layers_required;
    FjFloatSpan log_exposure;
    FjFloatSpan authored_curves_cmy;
    FjFloatSpan normalized_curves_cmy;
    FjFloatSpan density_curves_layers[3][3];
    float gamma_rgb[3];
    float minimum_cmy[3];
    float maximum_cmy[3];
    uint64_t normalized_curves_hash;
    uint64_t layers_hash;
    uint64_t hash;
} FjFilmDevelopmentView;

#define FJ_FILM_DIR_INACTIVE UINT32_C(0)

#define FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR UINT32_C(1)

#define FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR UINT32_C(2)

#define FJ_FILM_RECIPE_REASON_NONE UINT32_C(0)

typedef struct FjFilmReferenceInput {
    uint32_t method, auto_exposure_enabled;
    float manual_exposure_ev, mallett_green_scale;
    float projection_white_xyz[3];
    FjFloatSpan basis_rgb, film_illuminant, sensitivity_rgb, tc_lut;
} FjFilmReferenceInput;

typedef struct FjFilmReference {
    float baseline_raw_rgb[3], compensated_raw_rgb[3];
    float baseline_density_cmy[3], compensated_density_cmy[3];
    float meter_scale;
    uint64_t hash;
} FjFilmReference;

typedef struct FjFilmReferenceFailure {
    uint32_t reason, exposure, operation, channel;
} FjFilmReferenceFailure;

#define FJ_FILM_REFERENCE_REASON_NONE UINT32_C(0)

#define FJ_FILM_REFERENCE_RAW UINT32_C(1)

#define FJ_FILM_REFERENCE_RECONSTRUCTION UINT32_C(2)

#define FJ_FILM_REFERENCE_METER UINT32_C(3)

#define FJ_FILM_REFERENCE_DENSITY_LOOKUP UINT32_C(4)

#define FJ_FILM_REFERENCE_DEPENDENCY UINT32_C(5)

#define FJ_FILM_REFERENCE_PANIC UINT32_C(6)

#define FJ_FILM_RECIPE_RAW UINT32_C(1)

#define FJ_FILM_RECIPE_MINIMUM UINT32_C(2)

#define FJ_FILM_RECIPE_MATRIX UINT32_C(3)

#define FJ_FILM_RECIPE_DENSITY_MAX UINT32_C(4)

#define FJ_FILM_RECIPE_RECEIVER_K UINT32_C(5)

#define FJ_FILM_RECIPE_DONOR_K UINT32_C(6)

#define FJ_FILM_RECIPE_DONOR_LANGMUIR UINT32_C(7)

#define FJ_FILM_RECIPE_RECEIVER_LANGMUIR UINT32_C(8)

#define FJ_FILM_RECIPE_COMPENSATED_AXIS UINT32_C(9)

#define FJ_FILM_RECIPE_IDENTITY UINT32_C(10)

#define FJ_FILM_RECIPE_BOUNDS_SPAN UINT32_C(11)

#define FJ_FILM_RECIPE_SIZE UINT32_C(12)

#define FJ_FILM_RECIPE_ALLOCATION UINT32_C(13)

#define FJ_FILM_RECIPE_PANIC UINT32_C(14)

#define FJ_FILM_RECIPE_INTEGRITY UINT32_C(15)

#define FJ_FILM_RECIPE_LOG_EXPOSURE UINT32_C(16)

#define FJ_FILM_RECIPE_GRAIN_CONTROLS UINT32_C(17)

#define FJ_FILM_RECIPE_GRAIN_LAYERS UINT32_C(18)

#define FJ_FILM_RECIPE_EFFECTS_PROBABILITY UINT32_C(19)

typedef struct FjFilmRecipeFailure {
    uint32_t reason;
    uint32_t channel;
    uint64_t sample;
} FjFilmRecipeFailure;

#define FJ_OPTICS_REASON_NONE UINT32_C(0)

#define FJ_OPTICS_REASON_RAW UINT32_C(1)

#define FJ_OPTICS_REASON_DIFFUSION UINT32_C(2)

#define FJ_OPTICS_REASON_LENS_BLUR UINT32_C(3)

#define FJ_OPTICS_REASON_HALATION UINT32_C(4)

#define FJ_OPTICS_REASON_PANIC UINT32_C(5)

#define FJ_OPTICS_COMPONENT_NONE UINT32_C(0)

#define FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION UINT32_C(1)

#define FJ_OPTICS_COMPONENT_CAMERA_LENS_BLUR UINT32_C(2)

#define FJ_OPTICS_COMPONENT_SCATTER_HALATION UINT32_C(3)

#define FJ_OPTICS_COMPONENT_ENLARGER_DIFFUSION UINT32_C(4)

#define FJ_OPTICS_FIELD_NONE UINT32_C(0)

#define FJ_OPTICS_FIELD_FAMILY UINT32_C(1)

#define FJ_OPTICS_FIELD_STRENGTH UINT32_C(2)

#define FJ_OPTICS_FIELD_SPATIAL_SCALE UINT32_C(3)

#define FJ_OPTICS_FIELD_HALO_WARMTH UINT32_C(4)

#define FJ_OPTICS_FIELD_CORE_INTENSITY UINT32_C(5)

#define FJ_OPTICS_FIELD_CORE_SIZE UINT32_C(6)

#define FJ_OPTICS_FIELD_HALO_INTENSITY UINT32_C(7)

#define FJ_OPTICS_FIELD_HALO_SIZE UINT32_C(8)

#define FJ_OPTICS_FIELD_BLOOM_INTENSITY UINT32_C(9)

#define FJ_OPTICS_FIELD_BLOOM_SIZE UINT32_C(10)

#define FJ_OPTICS_FIELD_SCATTER_FRACTION UINT32_C(11)

#define FJ_OPTICS_FIELD_GROUP_WEIGHTS UINT32_C(12)

#define FJ_OPTICS_FIELD_GROUP_CENTERS UINT32_C(13)

#define FJ_OPTICS_FIELD_RESOLVED_HASH UINT32_C(14)

#define FJ_OPTICS_FIELD_SIGMA_UM UINT32_C(15)

#define FJ_OPTICS_FIELD_TOTAL_STRENGTH UINT32_C(16)

typedef struct FjDiffusionParams {
    uint32_t active;
    uint32_t family;
    double strength;
    double spatial_scale;
    double halo_warmth;
    double core_intensity;
    double core_size;
    double halo_intensity;
    double halo_size;
    double bloom_intensity;
    double bloom_size;
} FjDiffusionParams;

typedef struct FjHalationParams {
    uint32_t active;
    float scatter_amount;
    float scatter_spatial_scale;
    float halation_amount;
    float halation_spatial_scale;
    float first_sigma_um[3];
    float primary_amount[3];
} FjHalationParams;

typedef struct FjSpatialOpticsParams {
    FjDiffusionParams camera_diffusion;
    float camera_lens_blur_um;
    FjHalationParams scatter_halation;
    FjDiffusionParams enlarger_diffusion;
    uint32_t print_domain;
} FjSpatialOpticsParams;

typedef struct FjDiffusionRecipe {
    uint32_t domain;
    uint32_t active;
    uint32_t family;
    double scatter_fraction;
    double group_weights[3];
    double group_lambda_um[3];
    double warmth;
    double spatial_scale;
    uint64_t resolved_hash;
    uint64_t hash;
} FjDiffusionRecipe;

typedef struct FjDiffusionSampleInput {
    uint32_t family;
    double group_weights[3], group_lambda_um[3];
    double warmth, spatial_scale, pixel_size_um;
    int32_t width, height;
} FjDiffusionSampleInput;

typedef struct FjDiffusionFrameInput {
    uint32_t route;
    FjExtent full_frame;
    double pixel_size_um;
    FjDiffusionRecipe camera, enlarger;
} FjDiffusionFrameInput;

typedef struct FjDiffusionFrameSet {
    uint32_t route, flags; /* camera=1, enlarger=2 */
    FjExtent full_frame;
    FjDiffusion camera, enlarger;
    uint64_t hash;
} FjDiffusionFrameSet;

typedef struct FjLensBlurRecipe {
    float sigma_um;
    uint64_t hash;
} FjLensBlurRecipe;

typedef struct FjHalationRecipe {
    uint32_t scatter_active;
    uint32_t back_reflection_active;
    float scatter_amount;
    float scatter_spatial_scale;
    float halation_spatial_scale;
    float first_sigma_um[3];
    float total_strength[3];
    uint64_t hash;
} FjHalationRecipe;

typedef struct FjSpatialOpticsRecipe {
    FjDiffusionRecipe camera_diffusion;
    FjLensBlurRecipe camera_lens_blur;
    FjHalationRecipe scatter_halation;
    FjDiffusionRecipe enlarger_diffusion;
    uint64_t hash;
} FjSpatialOpticsRecipe;

typedef struct FjSpatialOpticsFailure {
    uint32_t reason;
    uint32_t component;
    uint32_t field;
    uint32_t channel;
} FjSpatialOpticsFailure;

typedef struct FjPhysicalEffectsParams {
    float film_dust_amount;
    float film_scratch_amount;
    float gate_dust_amount;
    float gate_scratch_amount;
    double gate_weave_amount;
    uint32_t grain_active;
    int32_t grain_debug_view;
} FjPhysicalEffectsParams;

typedef struct FjPhysicalEffectsRecipe {
    FjDust film_dust;
    FjScratch film_scratch;
    FjDust gate_dust;
    FjScratch gate_scratch;
    double gate_weave_amount;
    uint32_t active;
    uint64_t hash;
} FjPhysicalEffectsRecipe;

typedef struct FjFilmDensityMinimum {
    float cmy[3];
} FjFilmDensityMinimum;

typedef struct FjFilmDirParams {
    uint32_t active;
    float amount;
    float inhibition_same_layer;
    float inhibition_interlayer;
    float diffusion_size_um;
    float diffusion_tail_um;
    float diffusion_tail_weight;
    uint32_t gamma_use_stock;
    float gamma_same_layer_rgb[3];
    float gamma_interlayer_r_to_gb[2];
    float gamma_interlayer_g_to_rb[2];
    float gamma_interlayer_b_to_rg[2];
    float langmuir_donor_k_rgb[3];
    float langmuir_receiver_k_rgb[3];
} FjFilmDirParams;

typedef struct FjFilmDirOwner FjFilmDirOwner;

typedef struct FjFilmDirView {
    uint32_t polarity;
    uint32_t mode;
    float matrix_rgb[9];
    float diffusion_size_um;
    float diffusion_tail_um;
    float diffusion_tail_weight;
    float density_max_rgb[3];
    float density_ref_rgb[3];
    float donor_k_rgb[3];
    float receiver_c_ref_rgb[3];
    float receiver_kr_rgb[3];
    FjFloatSpan compensated_axes_rgb[3];
    uint64_t compensated_axes_hash;
    uint64_t hash;
} FjFilmDirView;

typedef struct FjFilmDensityBounds {
    uint32_t route;
    uint32_t medium;
    uint32_t polarity;
    uint32_t source;
    float minimum_cmy[3];
    float maximum_cmy[3];
    float inv_span_cmy[3];
    float authored_minimum_cmy[3];
    float authored_maximum_cmy[3];
    uint64_t hash;
} FjFilmDensityBounds;

typedef struct FjVisualGrainParams {
    uint32_t active;
    uint32_t sublayers_active;
    float particle_area_um2;
    float particle_scale_cmy[3];
    float particle_scale_layers[3];
    float particle_density_min_cmy[3];
    float uniformity_cmy[3];
    float correlation_sigma_px;
    float dye_cloud_blur_um;
    float micro_structure[2];
    int32_t sublayer_count;
    float amplitude;
    float chroma_mix;
    float chroma_shared_weight;
    float chroma_independent_weight;
    float coarse_weight;
    float mid_weight;
    float size_mix_scale;
    float clump_temporal_mix;
    float clump_morph_period_sec;
    int32_t debug_view;
} FjVisualGrainParams;

typedef struct FjVisualGrainRecipe {
    uint32_t active;
    uint32_t sublayers_active;
    uint32_t axis_finite_cmy[3];
    float axis_block_prefix_max_cmy[3][16];
    float particle_area_um2;
    float particle_scale_cmy[3];
    float particle_scale_layers[3];
    float particle_density_min_cmy[3];
    float uniformity_cmy[3];
    float correlation_sigma_px;
    float dye_cloud_blur_um;
    float micro_structure[2];
    int32_t sublayer_count;
    float amplitude;
    float chroma_mix;
    float chroma_shared_weight;
    float chroma_independent_weight;
    float fine_weight;
    float mid_weight;
    float coarse_weight;
    float size_mix_scale;
    float clump_temporal_mix;
    float clump_morph_period_sec;
    int32_t debug_view;
    uint64_t layers_hash;
    uint64_t hash;
} FjVisualGrainRecipe;

typedef struct FjSpectraLut FjSpectraLut;

typedef struct FjMallettBasis FjMallettBasis;

typedef struct FjCmf FjCmf;

typedef struct FjSpectraLutView {
    FjFloatSpan samples;
    uint64_t asset_hash;
} FjSpectraLutView;

typedef struct FjCsvPairs FjCsvPairs;

#define FJ_CSV_D65 UINT32_C(1)

#define FJ_CSV_D55 UINT32_C(2)

#define FJ_CSV_D50 UINT32_C(3)

#define FJ_CSV_T UINT32_C(4)

#define FJ_CSV_K75P UINT32_C(5)

#define FJ_CSV_KG3 UINT32_C(6)

#define FJ_CSV_CANON_24_F28_IS UINT32_C(7)

#define FJ_CALIBRATION_FOUND UINT32_C(1)

#define FJ_CALIBRATION_MISSING_FILE UINT32_C(2)

#define FJ_CALIBRATION_MISSING_ENTRY UINT32_C(3)

#define FJ_CALIBRATION_MALFORMED UINT32_C(4)

#define FJ_CALIBRATION_FIELD_NONE UINT32_C(0)

#define FJ_CALIBRATION_FIELD_RESOURCE_READ UINT32_C(1)

#define FJ_CALIBRATION_FIELD_ROOT UINT32_C(2)

#define FJ_CALIBRATION_FIELD_PRINT_PROFILE UINT32_C(3)

#define FJ_CALIBRATION_FIELD_PRINT_ILLUMINANT UINT32_C(4)

#define FJ_CALIBRATION_FIELD_CMY_CC UINT32_C(5)

typedef struct FjNeutralCalibrationResult {
    uint32_t outcome;
    uint32_t field;
    float cmy_cc[3];
} FjNeutralCalibrationResult;

typedef struct FjIlluminant {
    float samples[81];
} FjIlluminant;

typedef struct FjPrintIlluminantParams {
    float edges_nm[4];
    float transitions_nm[4];
    float cmy_cc[3];
} FjPrintIlluminantParams;

typedef struct FjPrintIlluminantFailure {
    uint32_t reason;
    uint32_t sample;
} FjPrintIlluminantFailure;

#define FJ_PRINT_ILLUMINANT_REASON_NONE UINT32_C(0)

#define FJ_PRINT_ILLUMINANT_REASON_RAW UINT32_C(1)

#define FJ_PRINT_ILLUMINANT_REASON_SAMPLE UINT32_C(2)

#define FJ_PRINT_ILLUMINANT_REASON_ENERGY UINT32_C(3)

#define FJ_PRINT_ILLUMINANT_REASON_PANIC UINT32_C(4)

typedef struct FjPrintPreflashInput {
    FjFloatSpan film_base_density;
    FjFloatSpan filtered_illuminant;
    FjFloatSpan print_sensitivity_cmy;
} FjPrintPreflashInput;

typedef struct FjPrintPreflashRaw {
    float cmy[3];
} FjPrintPreflashRaw;

typedef struct FjPrintPreflashFailure {
    uint32_t reason;
    uint32_t channel;
} FjPrintPreflashFailure;

#define FJ_PRINT_PREFLASH_REASON_NONE UINT32_C(0)

#define FJ_PRINT_PREFLASH_REASON_RAW UINT32_C(1)

#define FJ_PRINT_PREFLASH_REASON_CHANNEL UINT32_C(2)

#define FJ_PRINT_PREFLASH_REASON_PANIC UINT32_C(3)

typedef struct FjPrintBalanceInput {
    FjFloatSpan film_base_density;
    FjFloatSpan film_channel_density_cmy;
    FjFloatSpan print_sensitivity_rgb;
    FjFloatSpan filtered_main_illuminant;
    float baseline_density_cmy[3];
    float compensated_density_cmy[3];
    uint64_t reference_hash;
    uint32_t normalization_mode;
} FjPrintBalanceInput;

typedef struct FjPrintBalance {
    float baseline_raw_rgb[3];
    float compensated_raw_rgb[3];
    float factor_midgray;
    float factor_midgray_comp;
    float normalizer;
    uint64_t filtered_main_illuminant_hash;
    uint64_t hash;
} FjPrintBalance;

typedef struct FjPrintBalanceFailure {
    uint32_t reason;
    uint32_t channel;
} FjPrintBalanceFailure;

#define FJ_PRINT_BALANCE_REASON_NONE UINT32_C(0)

#define FJ_PRINT_BALANCE_REASON_RAW UINT32_C(1)

#define FJ_PRINT_BALANCE_REASON_BASELINE_RAW UINT32_C(2)

#define FJ_PRINT_BALANCE_REASON_COMPENSATED_RAW UINT32_C(3)

#define FJ_PRINT_BALANCE_REASON_BALANCE UINT32_C(4)

#define FJ_PRINT_BALANCE_REASON_PANIC UINT32_C(5)

typedef struct FjIlluminantCoverage {
    uint32_t warnings;
    float min_nm;
    float max_nm;
} FjIlluminantCoverage;

typedef struct FjIlluminantLens FjIlluminantLens;

typedef struct FjFilmTcLutInput {
    FjFloatSpan spectra, sensitivity_rgb, reference_illuminant;
    float projection_white_xyz[3];
    float spectral_blur;
    uint32_t method, apply_surface;
    FjFloatSpan surface_rgb;
    uint32_t compression_active, hull_available;
    float hull_center_xy[2];
    FjFloatSpan hull_xy;
} FjFilmTcLutInput;

typedef struct FjOwnedFilmTcLut {
    FjFloatSpan samples; /* 147456 readonly f32 RGB/padding values, native host/CUDA staging consumer */
    size_t capacity;     /* Actual original allocator capacity, not a recomputed count. */
} FjOwnedFilmTcLut;

typedef struct FjScannerCorrectionParams {
    uint32_t route, flags; /* black=1, white=2 */
    float black_level, white_level;
} FjScannerCorrectionParams;

typedef struct FjScannerCorrectionTables {
    FjFloatSpan eps_c, eps_m, eps_y, weighted_y, baseline_min;
    int32_t sample_count;
    float inv_yn;
    uint32_t has_baseline;
} FjScannerCorrectionTables;

typedef struct FjScannerDirectCorrectionInput {
    const FjScannerCorrectionTables* tables;
    FjFloatSpan film_base_density, film_log_exposure, film_authored_density_curves_cmy;
    float minimum_density_cmy[3], maximum_density_cmy[3], inverse_span_cmy[3];
    float baseline_density_cmy[3];
    uint64_t reference_hash;
    uint32_t medium;
} FjScannerDirectCorrectionInput;

typedef struct FjScannerPrintCorrectionInput {
    const FjScannerCorrectionTables* tables;
    FjFloatSpan film_base_density, film_channel_density_cmy;
    FjFloatSpan print_sensitivity_cmy, filtered_main_illuminant;
    FjFloatSpan print_base_density, print_log_exposure, print_density_curves_cmy;
    float minimum_density_cmy[3], inverse_span_cmy[3];
    float film_minimum_density_cmy[3], film_maximum_density_cmy[3];
    float preflash_raw_cmy[3], normalizer, preflash_exposure;
    uint32_t medium;
} FjScannerPrintCorrectionInput;

typedef struct FjScannerColorCorrection {
    uint32_t route, active, black_correction, white_correction;
    float target_black_linear, target_white_linear;
    float reference_black_y, reference_white_y, xyz_slope, xyz_offset, exposure_scale;
    uint32_t schema_version;
    uint64_t hash;
} FjScannerColorCorrection;

typedef struct FjScannerCorrectionFailure {
    uint32_t reason;
} FjScannerCorrectionFailure;

typedef struct FjScannerCorrectionTrace {
    uint32_t outcome;
    float reference_black_y, reference_white_y, target_black_linear, target_white_linear;
    float xyz_slope, xyz_offset, corrected_midgray, exposure_scale;
    uint64_t hash;
} FjScannerCorrectionTrace;

#define FJ_SCANNER_CORRECTION_REASON_NONE UINT32_C(0)

#define FJ_SCANNER_CORRECTION_REASON_RAW UINT32_C(1)

#define FJ_SCANNER_CORRECTION_REASON_DIRECT_INPUTS UINT32_C(2)

#define FJ_SCANNER_CORRECTION_REASON_DIRECT_REFERENCE UINT32_C(3)

#define FJ_SCANNER_CORRECTION_REASON_DIRECT_SCAN UINT32_C(4)

#define FJ_SCANNER_CORRECTION_REASON_DIRECT_MIDGRAY UINT32_C(5)

#define FJ_SCANNER_CORRECTION_REASON_DIRECT_LOOKUP UINT32_C(6)

#define FJ_SCANNER_CORRECTION_REASON_PRINT_INPUTS UINT32_C(7)

#define FJ_SCANNER_CORRECTION_REASON_PRINT_REFERENCE_LOOKUP UINT32_C(8)

#define FJ_SCANNER_CORRECTION_REASON_PRINT_SCAN UINT32_C(9)

#define FJ_SCANNER_CORRECTION_REASON_PRINT_MIDGRAY UINT32_C(10)

#define FJ_SCANNER_CORRECTION_REASON_PRINT_LOOKUP UINT32_C(11)

#define FJ_SCANNER_CORRECTION_REASON_INVALID UINT32_C(12)

#define FJ_SCANNER_CORRECTION_REASON_IDENTITY UINT32_C(13)

#define FJ_SCANNER_CORRECTION_REASON_PANIC UINT32_C(14)

#define FJ_SCANNER_CORRECTION_TRACE_NONE UINT32_C(0)

#define FJ_SCANNER_CORRECTION_TRACE_ACCEPTED UINT32_C(1)

#define FJ_SCANNER_CORRECTION_TRACE_MIDGRAY UINT32_C(2)

#define FJ_SCANNER_CORRECTION_TRACE_MALFORMED UINT32_C(3)

#define FJ_SCANNER_CORRECTION_TRACE_ZERO_HASH UINT32_C(4)

typedef struct FjScannerLutOwner FjScannerLutOwner;

typedef struct FjScannerColorInput {
    uint32_t medium, output_space, apply_cctf;
    float illuminant_xyz[3];
    uint64_t illuminant_hash, active_gamut_recipe_hash;
} FjScannerColorInput;

typedef struct FjScannerColor {
    float cat02[9], xyz_to_rgb[9], illuminant_xyz[3];
    uint32_t output_space, flags;
    FjOutputEncoding encoding;
    uint64_t active_gamut_recipe_hash, hash;
} FjScannerColor;

typedef struct FjScannerColorFailure {
    uint32_t reason, stage;
} FjScannerColorFailure;

#define FJ_SCANNER_COLOR_REASON_NONE UINT32_C(0)

#define FJ_SCANNER_COLOR_REASON_RAW UINT32_C(1)

#define FJ_SCANNER_COLOR_REASON_ILLUMINANT UINT32_C(2)

#define FJ_SCANNER_COLOR_REASON_OUTPUT UINT32_C(3)

#define FJ_SCANNER_COLOR_REASON_BOUNDARY UINT32_C(4)

#define FJ_SCANNER_COLOR_REASON_PANIC UINT32_C(5)

typedef struct FjScannerLutDescriptorInput {
    uint32_t mode, route, capture_polarity, bounds_medium;
    uint32_t bounds_source, output_medium, profile_present, resolution;
    uint64_t density_bounds_hash, print_profile_token;
    FjFloatSpan channel_density_cmy, base_density;
    FjByteSpan viewing_illuminant, profile_viewing_illuminant;
    FjByteSpan observer_identity, print_profile_key;
} FjScannerLutDescriptorInput;

typedef struct FjScannerLutDescriptor {
    uint32_t route, medium, capture_polarity, resolution;
    uint64_t density_bounds_hash, channel_density_hash, base_density_hash;
    uint64_t viewing_illuminant_hash, observer_hash, hash;
} FjScannerLutDescriptor;

typedef struct FjScannerLutDescriptorFailure {
    uint32_t reason, lane;
} FjScannerLutDescriptorFailure;

#define FJ_SCANNER_LUT_DESCRIPTOR_NONE UINT32_C(0)

#define FJ_SCANNER_LUT_DESCRIPTOR_RAW UINT32_C(1)

#define FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_RECIPE UINT32_C(2)

#define FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_BOUNDS UINT32_C(3)

#define FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_OBSERVER UINT32_C(4)

#define FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_ILLUMINANT UINT32_C(5)

#define FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_HASH UINT32_C(6)

#define FJ_SCANNER_LUT_DESCRIPTOR_PRINT_RECIPE UINT32_C(7)

#define FJ_SCANNER_LUT_DESCRIPTOR_PRINT_HASH UINT32_C(8)

#define FJ_SCANNER_LUT_DESCRIPTOR_PANIC UINT32_C(9)

typedef struct FjScannerPostEffectsInput {
    uint32_t route, glare_active;
    float glare_percent, glare_roughness, glare_blur_sigma_px;
    float lens_blur_sigma_px, unsharp_sigma_px, unsharp_amount;
} FjScannerPostEffectsInput;

typedef struct FjScannerPostEffectsDescriptor {
    uint32_t route, glare_active;
    float glare_percent, glare_roughness, glare_blur_sigma_px;
    int32_t glare_blur_radius;
    float lens_blur_sigma_px;
    int32_t lens_blur_radius;
    float unsharp_sigma_px;
    int32_t unsharp_radius;
    float unsharp_amount;
    uint32_t schema;
    uint64_t hash;
} FjScannerPostEffectsDescriptor;

typedef struct FjScannerPostEffectsFailure {
    uint32_t reason, lane;
    FjScannerPostEffectsDescriptor rejected;
} FjScannerPostEffectsFailure;

#define FJ_SCANNER_POST_NONE UINT32_C(0)

#define FJ_SCANNER_POST_RAW UINT32_C(1)

#define FJ_SCANNER_POST_VALUES UINT32_C(2)

#define FJ_SCANNER_POST_RADIUS UINT32_C(3)

#define FJ_SCANNER_POST_HASH UINT32_C(4)

#define FJ_SCANNER_POST_PANIC UINT32_C(5)

typedef struct FjGaussianKernelOwner FjGaussianKernelOwner;

#define FJ_FRAME_GEOMETRY_NONE UINT32_C(0)

#define FJ_FRAME_GEOMETRY_RAW UINT32_C(1)

#define FJ_FRAME_GEOMETRY_DOMAIN_RANGE UINT32_C(2)

#define FJ_FRAME_GEOMETRY_PANIC UINT32_C(3)

typedef struct FjFrameGeometryInput {
    FjExtent full_frame;
    float film_format_long_edge_mm;
} FjFrameGeometryInput;

typedef struct FjFrameGeometryFailure {
    uint32_t reason;
} FjFrameGeometryFailure;

#define FJ_HALATION_FRAME_NONE UINT32_C(0)

#define FJ_HALATION_FRAME_RAW UINT32_C(1)

#define FJ_HALATION_FRAME_PIXEL_SIZE UINT32_C(2)

#define FJ_HALATION_FRAME_GAUSSIAN UINT32_C(3)

#define FJ_HALATION_FRAME_PANIC UINT32_C(4)

#define FJ_HALATION_FRAME_FIELD_NONE UINT32_C(0)

#define FJ_HALATION_FRAME_FIELD_PIXEL_SIZE UINT32_C(1)

#define FJ_HALATION_FRAME_FIELD_SCATTER_CORE UINT32_C(2)

#define FJ_HALATION_FRAME_FIELD_SCATTER_TAIL UINT32_C(3)

#define FJ_HALATION_FRAME_FIELD_BOUNCE UINT32_C(4)

typedef struct FjHalationFrameInput {
    FjHalationRecipe recipe;
    float pixel_size_um;
} FjHalationFrameInput;

typedef struct FjHalationFrameFailure {
    uint32_t reason, field, channel, index;
    double sigma_px;
} FjHalationFrameFailure;

#define FJ_DIR_FRAME_NONE UINT32_C(0)

#define FJ_DIR_FRAME_RAW UINT32_C(1)

#define FJ_DIR_FRAME_PREPARATION UINT32_C(2)

#define FJ_DIR_FRAME_PANIC UINT32_C(3)

#define FJ_DIR_FRAME_FIELD_NONE UINT32_C(0)

#define FJ_DIR_FRAME_FIELD_PIXEL_SIZE_UM UINT32_C(1)

#define FJ_DIR_FRAME_FIELD_RENDER_EXTENT UINT32_C(2)

#define FJ_DIR_FRAME_FIELD_FULL_FRAME_EXTENT UINT32_C(3)

#define FJ_DIR_FRAME_FIELD_PARTIAL_RENDER_WINDOW UINT32_C(4)

#define FJ_DIR_FRAME_FIELD_CORE UINT32_C(5)

#define FJ_DIR_FRAME_FIELD_TAIL UINT32_C(6)

#define FJ_DIR_FRAME_FIELD_COMPONENT_VALUES UINT32_C(7)

#define FJ_DIR_FRAME_FIELD_COMPONENT_CAPACITY UINT32_C(8)

#define FJ_DIR_FRAME_FIELD_RADIUS UINT32_C(9)

#define FJ_DIR_FRAME_FIELD_IIR_DENOMINATOR UINT32_C(10)

#define FJ_DIR_FRAME_FIELD_IIR_OUTPUT UINT32_C(11)

#define FJ_DIR_FRAME_FIELD_EMPTY_PLAN UINT32_C(12)

#define FJ_DIR_FRAME_NO_COMPONENT UINT32_C(4)

typedef struct FjDirFrameInput {
    uint32_t active;
    float diffusion_size_um, diffusion_tail_um, diffusion_tail_weight, pixel_size_um;
    FjExtent render_extent, full_frame_extent;
    uint64_t recipe_hash;
} FjDirFrameInput;

typedef struct FjDirFrameFailure {
    uint32_t reason, field, component;
    FjDirFrameMath trace;
} FjDirFrameFailure;

typedef struct FjGrainFrameInput {
    uint32_t active;
    uint32_t sublayers_active;
    uint32_t layer_curves_required;
    uint32_t capture_polarity;
    uint32_t film_polarity;
    int32_t sublayer_count;
    float pixel_size_um;
    float particle_area_um2;
    float particle_scale_cmy[3];
    float particle_scale_layers[3];
    float particle_density_min_cmy[3];
    float authored_max_cmy[3];
    float correlation_sigma_px;
    float dye_cloud_blur_um;
    float size_mix_scale;
    float fine_weight;
    float mid_weight;
    float coarse_weight;
    float clump_morph_period_sec;
    FjExtent render_extent;
    FjExtent full_frame_extent;
    double frame_time;
    double frame_rate;
    uint64_t session_seed;
    uint64_t clip_token;
    uint64_t recipe_hash;
    uint64_t recipe_layers_hash;
    uint64_t film_layers_hash;
    FjFloatSpan layer_curves[3][3];
} FjGrainFrameInput;

typedef struct FjGrainFrameFailure {
    uint32_t reason;
    uint32_t field;
    uint32_t layer;
    uint32_t channel;
} FjGrainFrameFailure;

typedef struct FjEffectsFrameInput {
    uint32_t active;
    FjExtent render_extent;
    FjExtent pixel_definition;
    float pixel_size_um;
    float film_format_long_edge_mm;
    double canonical_x;
    double canonical_y;
    double canonical_width;
    double canonical_height;
    double scale_x;
    double scale_y;
    double pixel_aspect_ratio;
    double frame_time;
    double frame_rate;
    double gate_weave_amount;
    uint64_t session_seed;
    uint64_t clip_token;
    uint64_t recipe_hash;
    FjDust film_dust;
    FjScratch film_scratch;
    FjDust gate_dust;
    FjScratch gate_scratch;
} FjEffectsFrameInput;

typedef struct FjEffectsFrameFailure {
    uint32_t reason;
} FjEffectsFrameFailure;

typedef struct FjIlluminantsOwner FjIlluminantsOwner;

typedef struct FjInputHullOwner FjInputHullOwner;

typedef struct FjOutputBoundaryOwner FjOutputBoundaryOwner;

typedef struct FjIlluminantsView {
    FjFloatSpan d65, d55, d50, tungsten, k75p, tungsten_kg3, tungsten_kg3_lens;
    uintptr_t entry_identity;
} FjIlluminantsView;

typedef struct FjInputHullView {
    float center_xy[2];
    FjFloatSpan xy; /* 2050 f32, 1025 ordered xy points, closed polygon. */
    uint64_t hash;
    uintptr_t entry_identity;
} FjInputHullView;

typedef struct FjOutputBoundaryView {
    FjFloatSpan cmax; /* 46080 f32, lightness-major [64][720]. */
    uint64_t transform_hash, contract_hash, hash;
    uintptr_t entry_identity;
} FjOutputBoundaryView;

typedef struct FjPreparationHalation {
    uint32_t active;
    float scatter_amount, scatter_spatial_scale, halation_amount, halation_spatial_scale;
} FjPreparationHalation;

typedef struct FjPreparationParams {
    double print_exposure, print_preflash_exposure, print_gamma_factor;
    double preflash_m_filter_cc, preflash_y_filter_cc;
    double print_shadow_compensation_factor, print_shadow_compensation_density, print_shadow_compensation_transition;
    double glare_percent, glare_roughness, glare_blur_sigma_px;
    double camera_exposure_compensation_ev; /* EV; print exposure/preflash are linear. */
    float camera_film_format_long_edge_mm, film_gamma_factor;
    double scanner_lens_blur_sigma_px, scanner_black_level, scanner_white_level, gate_weave_amount;
    double scanner_unsharp_mask[2], print_ui_ymc_cc[3], camera_filter_uv[3], camera_filter_ir[3];
    FjDiffusionParams camera_diffusion, enlarger_diffusion;
    FjFilmDirParams dir_couplers;
    FjStringView film_profile_key, print_profile_key;
    int32_t spectral_upsampling_mode, ref_ill, enl_ill;
    int32_t normalize_print_exposure, print_exposure_compensation;
    int32_t input_color_space, input_cctf_decoding, input_compression_enabled;
    int32_t hanatos2025_adaptation_window, hanatos2025_adaptation_surface;
    int32_t camera_auto_exposure_enabled, camera_metering_method;
    FjPreparationHalation scatter_halation;
    int32_t scanner_black_correction, scanner_white_correction, scanner_lut_resolution;
    int32_t output_color_space, output_cctf_encoding, output_gamut_compression_enabled;
    FjVisualGrainParams grain;
    float film_dust_amount, film_scratch_amount, gate_dust_amount, gate_scratch_amount;
    uint32_t scan_route, glare_active, camera_filter_override;
} FjPreparationParams;

typedef struct FjPreparedSpectralView {
    FjFloatSpan lambda_nm, illuminant;
    FjFloatSpan observer_xyz[3], weighted_xyz[3], dyes_cmy[3];
    FjFloatSpan baseline_min, baseline_mid;
    float inv_yn, white_xyz[3];
    uint32_t has_baseline;
    uint64_t illuminant_hash, hash;
} FjPreparedSpectralView;

typedef struct FjPreparedInputPolicy {
    uint32_t meter, band_pass_active, apply_hanatos_window, apply_hanatos_surface;
    uint32_t projection_illuminant, input_compression_active;
    float film_format_long_edge_mm, uv[3], ir[3], hanatos_spectral_blur;
    float nominal_white_xyz[3], projection_white_xyz[3];
    FjFloatSpan hanatos_window, hanatos_surface_rgb;
    uint64_t input_hull_hash, tc_source_hash;
} FjPreparedInputPolicy;

typedef struct FjPreparedScannerPolicy {
    uint32_t black_correction, white_correction, glare_active;
    float black_level, white_level, glare_percent, glare_roughness, glare_blur_sigma_px;
    float lens_blur_sigma_px, unsharp_sigma_px, unsharp_amount;
} FjPreparedScannerPolicy;

typedef struct FjPreparedStateView {
    FjStringView film_profile_key, print_profile_key;
    uint64_t film_profile_asset_version, print_profile_asset_version, profile_route_hash;
    uint64_t recipe_hash, upload_core_hash, scanner_hash;
    uint32_t route, polarity;
    FjPreparedInputPolicy input_policy;
    FjPreparedScannerPolicy scanner_policy;
    FjFilmExposure film_exposure;
    FjFilmDevelopment film_development;
    FjFilmDirView dir_recipe;
    FjFilmDensityMinimum grain_minimum;
    FjFilmDensityBounds scanner_bounds, enlarger_film_bounds;
    FjFilmReference film_reference;
    FjSpatialOpticsRecipe optics;
    FjVisualGrainRecipe grain;
    FjPhysicalEffectsRecipe effects;
    FjPreparedSpectralView exposure_tables, scanner_tables;
    float s_inverse[9], midgray_dwg_rgb[3], raw_midgray[3], raw_green, midgray_scale;
    uint32_t raw_midgray_bgr; /* 1 Mallett B/G/R; 0 TC R/G/B. */
    FjInputHullView input_hull;
    FjOutputBoundaryView output_boundary;
    FjScannerLutDescriptor scanner_lut;
    FjScannerColor scanner_color;
    FjScannerColorCorrection scanner_correction;
    FjScannerPostEffectsDescriptor scanner_post_effects;
    FjOutputColor output_color;
    FjPrint print;
    double print_gamma;
    float print_filter_main_cmy_cc[3], print_filter_preflash_cmy_cc[3];
    uint32_t print_normalization_mode, normalize_print_exposure, print_exposure_compensation;
    uint32_t calibration; /* 0 MissingFile, 1 MissingEntry, 2 Calibrated. */
    float neutral_cmy_cc[3];
    FjFloatSpan print_source_illuminant;
    FjPrintBalance print_balance;
    uint64_t scanner_recipe_hash, print_recipe_hash, print_filters_hash, print_exposure_hash, print_illuminant_hash, print_curves_hash;
} FjPreparedStateView;

typedef struct FjPreparedState FjPreparedState;

typedef struct FjInstance FjInstance;

typedef struct FjHostWrite FjHostWrite;

typedef struct FjCaptureStart {
    uint64_t capture;
    uint32_t kind;
} FjCaptureStart;

typedef struct FjSnapshotStart {
    uint64_t write_revision;
    uint32_t kind;
} FjSnapshotStart;

typedef struct FjAdmission {
    FjPreparedState* state;
    uint64_t input_identity;
    uint64_t build;
    uint32_t kind;
    uint32_t route;
} FjAdmission;

#define FJ_CAPTURE_INITIAL UINT32_C(0)

#define FJ_CAPTURE_CHANGE UINT32_C(1)

#define FJ_CAPTURE_REFRESH UINT32_C(2)

#define FJ_CAPTURE_START_NONE UINT32_C(0)

#define FJ_CAPTURE_STARTED UINT32_C(1)

#define FJ_CAPTURE_NOT_STARTED UINT32_C(2)

#define FJ_HOST_WRITE_NONE UINT32_C(0)

#define FJ_HOST_WRITE_STARTED UINT32_C(1)

#define FJ_HOST_WRITE_IGNORED UINT32_C(2)

#define FJ_HOST_WRITE_SUCCEEDED UINT32_C(0)

#define FJ_SNAPSHOT_NONE UINT32_C(0)

#define FJ_SNAPSHOT_READY UINT32_C(1)

#define FJ_SNAPSHOT_DEFERRED UINT32_C(2)

#define FJ_SNAPSHOT_IGNORED UINT32_C(3)

#define FJ_UPDATE_NONE UINT32_C(0)

#define FJ_UPDATE_APPLIED UINT32_C(1)

#define FJ_UPDATE_IGNORED UINT32_C(2)

#define FJ_UPDATE_DEFERRED UINT32_C(3)

#define FJ_CAPTURE_INVALID_CONTROLS UINT32_C(1)

#define FJ_CAPTURE_HOST_FAILURE UINT32_C(2)

#define FJ_CAPTURE_ALLOCATION_FAILURE UINT32_C(3)

#define FJ_ADMISSION_NONE UINT32_C(0)

#define FJ_ADMISSION_READY UINT32_C(1)

#define FJ_ADMISSION_NEEDS_CAPTURE UINT32_C(2)

#define FJ_ADMISSION_CAPTURING UINT32_C(3)

#define FJ_ADMISSION_CAPTURE_FAILED UINT32_C(4)

#define FJ_ADMISSION_PREPARATION_FAILED UINT32_C(5)

#define FJ_ADMISSION_POISONED UINT32_C(6)

#define FJ_ADMISSION_NEEDS_REFRESH UINT32_C(7)

#ifdef __cplusplus
}
#endif
