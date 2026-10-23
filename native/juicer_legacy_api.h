#pragma once

#include <stdint.h>

#include "juicer_cuda_api.h"

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

// FJ_TEMP_BRIDGE: route resolution; remove S6.D.
// A nonnull out_route must point to one writable byte; failures leave it unchanged.
uint32_t fj_legacy_resolve_route(uint8_t capture_polarity, uint8_t selected_route, uint8_t* out_route);

/* FJ_TEMP_BRIDGE: completed input hull value; remove S4.E.
 * Required records authorize initialized, aligned storage. Each source is
 * exactly 81 positional f32 values, immutable/live until synchronous return.
 * Immutable inputs may alias each other. Output, error record and nonempty
 * diagnostic backing are exclusive, mutually disjoint and disjoint from inputs.
 * No source pointer escapes. Authorized output values clear before fallible work;
 * padding is not a value. No heap result owner or release operation exists.
 * Malformed structure -> UnsupportedInput; computed hull -> PreparationFailure;
 * contained panic -> InternalFailure. Fixed math has no AllocationFailure.
 * Library cache and recipe ownership remain native until S4.E.
 */
typedef struct FjInputHullSource {
    FjFloatSpan x, y, z, d65;
} FjInputHullSource;
typedef struct FjInputHull {
    float center_xy[2];
    float xy[2050];
    uint64_t hash;
} FjInputHull;
FjStatus fj_legacy_gamut_input_hull(const FjInputHullSource* source, FjInputHull* out, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: completed output gamut values; remove S4.E.
 * Source records are initialized, aligned, addressable and immutable until
 * synchronous return. Registry literals are borrowed; no source is retained.
 * Writable records, Cmax, error record/backing are mutually disjoint and disjoint
 * from inputs; immutable inputs may alias. Address checks do not prove lifetime
 * or the actual allocation extent. Closed output_space tags match native 0..8.
 * Cmax is exactly 46080 f32 words, lightness-major [64][720]. Only an aligned,
 * non-overflowing, isize-bounded exact-count extent authorizes clearing/copying.
 * Independently valid outputs clear before diagnostic/input/fault rejection.
 * Failure requirement: 0 none, 1 transform, 2 c2_outside, 3 finite_first_exit,
 * 4 identity. Only 2 carries lightness/hue indices; failed values are cleared.
 * Optional error=NULL requests status only. A present live/exclusive error record
 * must have initialized capacity and writable length. At zero capacity data need
 * not be initialized and is never accessed. Positive capacity requires initialized
 * non-null writable data, capacity <= isize::MAX and non-overflowing address extent.
 * Malformed diagnostics -> UnsupportedInput without borrowing source/visiting
 * faults; valid outputs still clear, valid record length becomes zero, malformed
 * backing remains untouched. Valid text truncates with length excluding the NUL.
 * Computed failures -> PreparationFailure; actual reserve -> AllocationFailure;
 * panic -> InternalFailure. No partial table or retained foreign owner is returned.
 * Complete Rust heap construction copies 184320 bytes once on successful return.
 */
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
FjStatus fj_legacy_gamut_output_transform(const FjOutputGamutSource* source, FjOutputGamutTransform* out, FjErrorBuffer* error);
FjStatus fj_legacy_gamut_output_boundary(const FjOutputBoundarySource* source, float* cmax, size_t count, FjOutputBoundaryIdentity* identity, FjOutputBoundaryFailure* failure, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: spectral value preparation; remove S4.E.
 * Required pointers authorize complete initialized aligned records. Input spans
 * remain immutable and live through synchronous return; outputs/error backing
 * are exclusive and disjoint. No pointer or result owner is retained.
 * Fixed observer/illuminant arrays contain 81 positional samples on 380..780 nm
 * at 5 nm. Dyes contain 243 wavelength-major C/M/Y floats from profile storage.
 * Baselines are null/zero absent or exactly 81; midpoint requires minimum.
 * Tables project into native FocusedRenderPayload; family math/hashes are Rust.
 * Defined writable outputs clear before fallible work (padding is not a value).
 * Malformed structures -> UnsupportedInput; computed white -> PreparationFailure;
 * contained panic/integrity -> InternalFailure. Fixed array math has no allocator.
 */
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
/* Closed diagnostic reasons: 0 none, 1 nonfinite sample, 2 invalid luminance,
 * 3 invalid white sum, 4 nonfinite hash operand, 5 zero identity. Only reason 2
 * carries the observed f64 luminance sum; no failure record confers readiness. */
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
FjStatus fj_legacy_spectral_tables(const FjSpectralInput* input, FjSpectralTables* out, FjErrorBuffer* error);
FjStatus fj_legacy_spectral_white(const FjSpectralWhiteInput* input, FjSpectralWhite* out, FjSpectralWhiteFailure* out_failure, FjErrorBuffer* error);
FjStatus fj_legacy_spectral_s_inverse(const FjSpectralSInput* input, FjSpectralInverse* out, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: reference reconstruction/final sensitivity; remove S4.E.
 * Required pointers authorize initialized aligned records. Every consumed span
 * remains immutable/live through synchronous return. Output, failure, error record
 * and nonempty error backing are exclusive, mutually disjoint and disjoint from
 * all inputs. No foreign pointer is retained. Defined writable outputs clear
 * before fallible work; padding is neither a value nor an identity input.
 * spectra borrows the existing C-order [C=192][M=192][wavelength=81] Hanatos
 * irradiance tensor (2,985,984 f32s), never an Arctic/TC-LUT or a copied owner.
 * linear sensitivity is 243 wavelength-major R/G/B f32s already prepared by the
 * profile; illuminant is 81 positional samples at 380..780 nm, 5 nm steps.
 * method: 0 Hanatos2025, 1 Mallett2019, 2 Arctic2026beta04; flags are 0/1.
 * Only active Hanatos window spans are consumed: params must contain 4, white
 * is absent (NULL,0) or exactly 81. Absence fails at the late window stage.
 * Inactive window spans are ignored, including their pointer/count values.
 * UV/IR controls are already narrowed amplitude/center_nm/width_nm triplets;
 * band_pass_active retains native cameraFilterOverride activation policy.
 * Complete sensitivity/hash/recipe Mallett scale bind once into native FilmRawRecipe;
 * reference white is invocation-local. TC/enclosing keys remain native.
 * Malformed structure -> UnsupportedInput; computed/size failure -> PreparationFailure;
 * real kernel reservation failure -> AllocationFailure; panic -> InternalFailure.
 * The owner-approved A5 numerical policy selects pinned libm erff; no native math
 * callback, scientific tensor rescan, durable mirror or fallback exists here.
 */
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
/* Diagnostic-only hash failure: 0 none, 1 nonfinite operand, 2 zero identity.
 * Index is meaningful only for 1. Earlier metadata survives a later scale failure;
 * a failure record never confers readiness or supplies an identity substitute. */
typedef struct FjSensitivityFailure {
    uint32_t hash_failure, hash_sample_index;
} FjSensitivityFailure;
FjStatus fj_legacy_reconstruction_reference_white(const FjReferenceWhiteInput* input, FjReferenceWhite* out, FjErrorBuffer* error);
FjStatus fj_legacy_exposure_sensitivity(const FjSensitivityInput* input, FjSensitivity* out, FjSensitivityFailure* out_failure, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: CAT02/CAT16 host preparation; remove S4.E.
 * Inputs are initialized aligned XYZ triplets (12 bytes), read-only until
 * return and may share storage. Outputs are exclusive, aligned and disjoint
 * from every input: 9 row-major floats (36 bytes) or 3 XYZ floats (12 bytes).
 * No pointer/owner is retained. Valid output clears before input checks.
 * NULL required pointers -> UnsupportedInput; contained panic -> InternalFailure;
 * normally returned arrays, including nonfinite mathematics, -> Success.
 * Every result has API=None/native_code=0. Completed output belongs to caller. */
FjStatus fj_legacy_cat16_matrix(const float source_white_xyz[3], const float destination_white_xyz[3], float out_row_major[9]);
FjStatus fj_legacy_adapt_cat16(const float xyz[3], const float source_white_xyz[3], const float destination_white_xyz[3], float out_xyz[3]);

FjStatus fj_legacy_cat02_matrix(const float source_white_xyz[3], const float destination_white_xyz[3], float out_row_major[9]);

/* FJ_TEMP_BRIDGE: fixed output-color preparation; remove S4.E.
 * Strict raw FJ_OUTPUT_* tag, distinct from the native selector fallback.
 * Output authorizes its complete initialized live aligned/addressable exclusive
 * 168-byte extent through return. Valid output clears before tag/fault handling;
 * invalid output is never written. No allocation or foreign pointer is retained.
 * Unsupported tag/extent -> UnsupportedInput; contained panic -> InternalFailure. */
typedef struct FjOutputColorPreparation {
    uint32_t output_space;
    float rgb_to_xyz[9];
    float xyz_to_rgb[9];
    float white_xyz[3];
    FjOutputEncoding encoding;
    uint64_t entry_hash;
} FjOutputColorPreparation;
FjStatus fj_legacy_output_color(uint32_t output_space, FjOutputColorPreparation* out);

/* FJ_TEMP_BRIDGE: input-color host preparation; remove S4.E.
 * Inputs are complete initialized aligned records/arrays, read-only until return;
 * they may share storage. Disabled adaptation is also initialized. Outputs are
 * exclusive, mutually disjoint and disjoint from every input. Concurrent calls
 * require disjoint outputs; no input mutation/release during a call.
 * Every nonnull output clears its full extent inside panic containment before
 * pointer/tag/flag checks, even when another required output is NULL. All outputs
 * are required. No pointer or owner is retained. Unknown tags/non-0/1 flags/NULL
 * required pointers -> UnsupportedInput; panic -> InternalFailure; mathematical
 * nonfinite results -> Success. All statuses have API=None/native_code=0.
 * Matrices record is 96 bytes/align4, conversion 84 bytes/align4; triplets 12 bytes,
 * row-major matrices 36 bytes. Completed values are copied into native owners. */
/* input_space uses the FJ_INPUT_* uint32 tags declared in juicer_cuda_api.h. */
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
FjStatus fj_legacy_input_matrices(uint32_t input_space, FjInputColorMatrices* out);
FjStatus fj_legacy_linear_srgb_to_xyz(const float rgb[3], float out_xyz[3]);
FjStatus fj_legacy_project_linear_rgb_to_xyz(const float rgb[3], const float rgb_to_xyz[9], const float xyz_adapt[9], float out_xyz[3]);

/* Fixed exposure calls borrow initialized, aligned, immutable inputs only until
 * return. Basis/final-sensitivity spans contain exactly 243 wavelength-major RGB
 * floats; illuminant has 81 canonical positional samples. Current process/state
 * owners retain those arrays. No source ownership or pointer escapes the call.
 * Complete fixed results bind once into FilmRawConfig or synthetic references.
 * Output/error storage is exclusive, mutually disjoint and disjoint from inputs.
 * Valid outputs clear before fallible work. Unsupported structure/tags/flags ->
 * UnsupportedInput; computed source/reference failure -> PreparationFailure;
 * contained panic -> InternalFailure. Fixed mathematics allocates no heap.
 * Diagnostics require an initialized FjErrorBuffer; zero capacity is status-only.
 * Focused raw fields are BGR and may preserve finite-f64 narrowing to infinity;
 * raw_green is the resolved denominator. Reference output is RGB. */
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
/* FJ_TEMP_BRIDGE: focused Mallett mid-gray binding; remove S4.E. */
FjStatus fj_legacy_exposure_mallett_midgray(const FjMallettMidgrayInput* input, FjMallettMidgray* out, FjErrorBuffer* error);
/* FJ_TEMP_BRIDGE: TC green normalization binding; remove S4.E. */
FjStatus fj_legacy_exposure_tc_midgray(float green, FjMidgrayNormalization* out, FjErrorBuffer* error);
/* FJ_TEMP_BRIDGE: shared synthetic reference source binding; remove S4.E. */
FjStatus fj_legacy_exposure_reference_source(float exposure_ev, float* out, FjErrorBuffer* error);
/* FJ_TEMP_BRIDGE: synthetic Mallett reference reduction binding; remove S4.E. */
FjStatus fj_legacy_exposure_mallett_reference_raw(const FjMallettReferenceInput* input, FjReferenceRaw* out, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: asset conversion; remove S4.E.
 * Handles own immutable Rust storage; only matching create/acquire values are
 * legal. Caller excludes destruction from operations and release from every
 * entry read/outstanding view. Concurrent reads require disjoint outputs.
 * All outputs/diagnostics are exclusive, aligned and mutually disjoint, and
 * disjoint from initialized inputs/owner storage. Valid outputs are cleared
 * before input/diagnostic checks. NULL required slots/owners are unsupported.
 * Text is length-delimited UTF-8; empty text is (NULL,0), embedded NUL survives.
 * Destroy/release consume once even with malformed diagnostics; NULL is a no-op
 * subject to diagnostic validation. No CUDA/OFX callbacks occur here.
 * Diagnostics: NULL allowed; length initialized to zero; capacity zero never
 * reads data. Otherwise nonnull backing, capacity <= PTRDIFF_MAX, trailing NUL.
 * Categories: success, unsupported input, preparation, allocation, internal;
 * api/native_code are zero. Diagnostic truncation never changes the status. */
typedef struct FjAssets FjAssets;
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
FjStatus fj_legacy_assets_create(FjPathView resource_root, FjAssets** out_assets, FjErrorBuffer* error);
FjStatus fj_legacy_assets_destroy(FjAssets* assets, FjErrorBuffer* error);
FjStatus fj_legacy_catalog_acquire(const FjAssets* assets, FjCatalog** out_catalog, FjCatalogCounts* out_counts, FjErrorBuffer* error);
FjStatus fj_legacy_catalog_entry(const FjCatalog* catalog, uint32_t role, size_t index, FjCatalogEntryView* out_entry, FjErrorBuffer* error);
FjStatus fj_legacy_catalog_release(FjCatalog* catalog, FjErrorBuffer* error);

/* Complete STBN/Wang source; views reuse FjStaticNoise and borrow their owner.
 * Owners survive Assets/cache release. No byte copy, runtime identity or CUDA
 * work occurs here. Capacity and reported Open/Read OOM -> AllocationFailure;
 * other decoder failures -> PreparationFailure; poison/panic -> InternalFailure.
 * Capacity is nonsticky; reported I/O OOM retains the ordinary error-cache policy.
 * Shared output/diagnostic/consume-once obligations above apply on every outcome.
 * Incidental Box/Arc/path/parser allocator aborts are not recoverable. */
typedef struct FjNoise FjNoise;
FjStatus fj_legacy_noise_acquire(const FjAssets* assets, FjNoise** out_owner, FjErrorBuffer* error);
FjStatus fj_legacy_noise_view(const FjNoise* owner, FjStaticNoise* out_view, FjErrorBuffer* error);
FjStatus fj_legacy_noise_release(FjNoise* owner, FjErrorBuffer* error);

/* Profile views borrow their matching owner. Gamma results own totals and
 * survive print release. Sampling uses the retained original f64 source axis.
 * Span counts are elements; empty spans are (NULL,0). Release excludes every
 * use/outstanding borrow; acquired owners survive Assets release/destruction. */
typedef struct FjFilmProfile FjFilmProfile;
/* SF_TEMP_BRIDGE_FILM_INPUT: synchronous fixed values; remove S4.E.
 * All records except error are required, complete, initialized and aligned.
 * profile is a matching retained handle; release/mutation cannot overlap this call.
 * Inputs and present sample extents remain immutable until return and may alias.
 * out_input, out_failure, optional error and positive-capacity diagnostic backing
 * are exclusive, mutually disjoint and disjoint from all inputs/reachable profile
 * storage. The caller supplies actual extent/lifetime/disjointness; checked numeric
 * extents do not establish these properties. No input pointer is retained.
 * reference_illuminant is nonnull/81. Optional whites are NULL/0 or nonnull/3;
 * reconstructed_reference_white is NULL/0 or nonnull/81. All supplied spans have
 * canonical counts, checked multiplication, PTRDIFF_MAX bounds and address extent
 * before borrowing, including inactive spans. No authored-content scan is added.
 * Independently authorized outputs clear every semantic field to integer/+float
 * zero before admission; padding is unspecified and is never evidence. Success
 * commits the entire result, including three row-major matrices and XYZ whites.
 * error=NULL is supported. Authorized length becomes zero first. Capacity zero
 * never reads data. Positive backing must be nonnull/addressable, <=PTRDIFF_MAX;
 * valid text uses capacity-1 bytes plus NUL, length excludes NUL. Invalid backing
 * is untouched. Diagnostic absence/truncation never changes status or metadata.
 * Raw rejection order: diagnostics; required outputs/profile/params/resources;
 * input-space/method/meter tags; declaration-order flags; declaration-order spans.
 * Raw rejection is UnsupportedInput/RAW with its closed reason. Semantic resource,
 * sensitivity and identity errors are PreparationFailure; caught panic is
 * InternalFailure with narrow leaf provenance. All statuses use API_NONE/code=0.
 * Hash metadata: 0 none/index0, 1 nonfinite/index0..242, 2 zero identity/index0.
 * Failure fields use origin-scoped closed reason codes below; success has zero
 * metadata. Failure clears the result while preserving consumed hash metadata.
 * Test-only category/fault injection is absent from testing-OFF products.
 */
#define FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE UINT32_C(0)
#define FJ_FILM_INPUT_ILLUMINANT_D65 UINT32_C(1)

#define FJ_FILM_INPUT_ORIGIN_NONE UINT32_C(0)
#define FJ_FILM_INPUT_ORIGIN_RAW UINT32_C(1)
#define FJ_FILM_INPUT_ORIGIN_RESOURCE UINT32_C(2)
#define FJ_FILM_INPUT_ORIGIN_COLOR UINT32_C(3)
#define FJ_FILM_INPUT_ORIGIN_SENSITIVITY UINT32_C(4)
#define FJ_FILM_INPUT_ORIGIN_IDENTITY UINT32_C(5)
#define FJ_FILM_INPUT_ORIGIN_INTERNAL UINT32_C(6)

/* reason is a closed code within origin, not a global enum. */
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

FjStatus fj_legacy_film_input_prepare(
    const FjFilmProfile* profile,
    const FjFilmInputParams* params,
    const FjFilmInputResources* resources,
    FjFilmInput* out_input,
    FjFilmInputFailure* out_failure,
    FjErrorBuffer* error);

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

FjStatus fj_legacy_film_profile_acquire(const FjAssets* assets, FjStringView key, FjFilmProfile** out_profile, FjErrorBuffer* error);
FjStatus fj_legacy_film_profile_view(const FjFilmProfile* profile,
                                     FjFilmProfileView* out_view,
                                     FjErrorBuffer* error);
FjStatus fj_legacy_film_profile_release(FjFilmProfile* profile, FjErrorBuffer* error);

/* SF_TEMP_BRIDGE_FILM_DEVELOPMENT: completed capture-film family; remove S4.E.
 * The opaque owner retains its completed profile independently of the input
 * profile handle/cache. Views expire at consuming release; no pointer escapes
 * synchronous native copying into FilmDevelopRecipe or its async staging.
 * log_exposure has N elements; authored/normalized CMY are exposure-major 3N;
 * selected layers have nine N-element [layer][channel][sample] lanes. Omitted
 * layers are (NULL,0) and layers_hash=0. Route tags are 0..3, polarity/layers 0/1.
 * Live/aligned/disjoint storage and bounded diagnostic preconditions are the
 * existing profile API contract. Authorized output slots/semantic fields clear
 * before fallible work. Release takes the handle exactly once even with malformed
 * diagnostics or a contained failure; NULL is harmless. Invalid pointers/tags/
 * controls and route mismatch -> UnsupportedInput; size/zero identity ->
 * PreparationFailure; explicit normalized reserve -> AllocationFailure; panic ->
 * InternalFailure. API=None/native_code=0. Incidental Box/Arc/diagnostic allocation
 * may abort: unwind containment does not promise physical OOM recovery.
 */
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
FjStatus fj_legacy_film_development_acquire(const FjFilmProfile* profile, uint32_t requested_route, float gamma_factor, uint32_t layers_required, FjFilmDevelopmentOwner** out_development, FjErrorBuffer* error);
FjStatus fj_legacy_film_development_view(const FjFilmDevelopmentOwner* development,
                                         FjFilmDevelopmentView* out_view,
                                         FjErrorBuffer* error);
FjStatus fj_legacy_film_development_release(FjFilmDevelopmentOwner* development, FjErrorBuffer* error);

/* SF_TEMP_BRIDGE_FILM_DIR_DENSITY: synchronous film construction; remove S4.E.
 * Required result/failure storage is initialized, aligned, live and disjoint
 * from inputs, owners and optional diagnostics. Numeric extent checks do not
 * establish allocation lifetime, object initialization or non-overlap.
 * Diagnostic capacity zero never reads data. Every admitted result is cleared
 * independently before rejection; failure starts {RAW,3,UINT64_MAX}.
 * Success commits {NONE,3,UINT64_MAX}. DIR views expire on consuming release.
 * Development retain returns the same address with an independent release
 * obligation; every release consumes its hold even on diagnostic failure.
 * Original requests to minimum and bounds must be identical; the safe core
 * borrows a completed Minimum while this foreign boundary reconstructs it.
 */
#define FJ_FILM_DIR_INACTIVE UINT32_C(0)
#define FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR UINT32_C(1)
#define FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR UINT32_C(2)
#define FJ_FILM_RECIPE_REASON_NONE UINT32_C(0)

/* SF_TEMP_BRIDGE_FILM_REFERENCE: synchronous complete value; remove S4.E.
 * Inputs borrow completed Development and fixed float spans until return.
 * Active spans: Mallett wavelength-major RGB basis243, illuminant81,
 * sensitivity RGB243; TC192*192*4. Missing active spans and every unused
 * span are canonical {NULL,0}. No ownership or pointer escapes.
 * Initialized live storage is required. Result/failure/error and diagnostic
 * backing are exclusive and disjoint from each other and all input storage.
 * Each valid writable result/failure is independently cleared on failure. */
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
FjStatus fj_legacy_film_reference_prepare(const FjFilmDevelopmentOwner*, const FjFilmReferenceInput*, FjFilmReference*, FjFilmReferenceFailure*, FjErrorBuffer*);
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


/* SF_TEMP_BRIDGE_SPATIAL_OPTICS: complete fixed recipe transport; remove S4.E.
 * Required records are initialized/live/aligned; every writable output, optional
 * diagnostic record and positive-capacity backing is exclusive, mutually disjoint
 * and disjoint from input storage until synchronous return. No pointer is retained.
 * Checked size/address/PTRDIFF_MAX/alignment admission does not prove actual
 * allocation validity, initialization or disjointness. Rejected numeric pointer
 * representations are never dereferenced. Every independently admitted result
 * clears every semantic field to integer/+float zero before other admission;
 * padding is unspecified. Admitted failure starts NONE/NONE/NONE/channel=3.
 * error=NULL is supported; admitted length initializes to zero. Capacity zero
 * never reads backing. Positive capacity requires bounded valid backing and emits
 * at most capacity-1 bytes plus NUL; length excludes NUL. Absent/truncated text
 * changes neither status nor typed failure. All statuses use API_NONE/code=0.
 * Admission order: independent output clearing; diagnostics/required records;
 * declaration-order input booleans (all four are 0/1); owning recipe stages.
 * Family/numeric semantics are constructor-owned, including direct-route omission
 * of enlarger content. Wire uint32 flags/tags are NOT the native one-byte hash
 * inputs. Inactive successful diffusion retains family=1/domain=0; raw clearing
 * uses family=0. A successful inactive lens may retain negative zero.
 * Success: SUCCESS/NONE/NONE/NONE/3. Raw: UNSUPPORTED_INPUT/RAW/NONE/NONE/3.
 * Construction failure: PREPARATION_FAILURE with closed component/field metadata.
 * Panic: INTERNAL_FAILURE/PANIC/NONE/NONE/3. Only TOTAL_STRENGTH has RGB channel
 * 0..2; all other fields use 3. Fixed construction has no allocation status.
 */
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

FjStatus fj_legacy_spatial_optics_prepare(const FjSpatialOpticsParams* params, FjSpatialOpticsRecipe* out_recipe, FjSpatialOpticsFailure* out_failure, FjErrorBuffer* error);

/* SF_TEMP_BRIDGE_PHYSICAL_EFFECTS: completed fixed recipe transport; remove S4.E.
 * Call records/backing must be initialized, live, aligned, mutually disjoint and
 * exclusive where writable. Extent admission cannot prove allocation or lifetime.
 * Every independently admitted output clears before input/diagnostic rejection.
 * Numeric controls are sanitized by the safe constructor; grain_active is 0/1. */
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

FjStatus fj_legacy_physical_effects_prepare(
    const FjPhysicalEffectsParams* params,
    FjPhysicalEffectsRecipe* out_recipe,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);

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

/* SF_TEMP_BRIDGE_VISUAL_GRAIN: fixed completed grain projection; remove S4.E.
 * The matching live Development is borrowed only through return, without a hold.
 * Initialized aligned readable params and exclusive writable result/failure and
 * optional diagnostics are live and mutually disjoint from every input/owner.
 * Booleans are exactly 0/1. Extent checks do not prove pointer/owner provenance.
 * Failure clears every admitted result field to zero; inactive success has scale 1.
 */
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

FjStatus fj_legacy_visual_grain_prepare(
    const FjFilmDevelopmentOwner* development,
    const FjVisualGrainParams* params,
    FjVisualGrainRecipe* out_recipe,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_legacy_film_development_retain(
    const FjFilmDevelopmentOwner* development,
    FjFilmDevelopmentOwner** out_development,
    FjErrorBuffer* error);
FjStatus fj_legacy_film_density_minimum_prepare(
    const FjFilmDensityMinimum* requested,
    FjFilmDensityMinimum* out_minimum,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_legacy_film_dir_acquire(
    const FjFilmDevelopmentOwner* development,
    const FjFilmDirParams* params,
    FjFilmDirOwner** out_dir,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_legacy_film_dir_view(
    const FjFilmDirOwner* dir,
    FjFilmDirView* out_view,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_legacy_film_dir_release(
    FjFilmDirOwner* dir, FjErrorBuffer* error);
FjStatus fj_legacy_film_density_bounds_prepare(
    const FjFilmDevelopmentOwner* development,
    const FjFilmDensityMinimum* requested,
    FjFilmDensityBounds* out_bounds,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_legacy_print_profile_acquire(const FjAssets* assets, FjStringView key, FjPrintProfile** out_profile, FjErrorBuffer* error);
FjStatus fj_legacy_print_profile_view(const FjPrintProfile* profile,
                                      FjPrintProfileView* out_view,
                                      FjErrorBuffer* error);
FjStatus fj_legacy_print_profile_release(FjPrintProfile* profile, FjErrorBuffer* error);
FjStatus fj_legacy_print_profile_sample_density(const FjPrintProfile* profile,
                                                double gamma,
                                                FjPrintDensityCurves** out_curves,
                                                FjErrorBuffer* error);
FjStatus fj_legacy_print_density_view(const FjPrintDensityCurves* curves,
                                      FjPrintDensityView* out_view,
                                      FjErrorBuffer* error);
FjStatus fj_legacy_print_density_release(FjPrintDensityCurves* curves, FjErrorBuffer* error);
FjStatus fj_legacy_assets_release_cached_payloads(const FjAssets* assets, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: spectral source conversion; remove S4.E.
 * Source views borrow their matching family owner, independently of Assets.
 * LUT samples: 192*192*81 f32 C-order [i][j][wavelength], 380..780 nm @ 5nm.
 * asset_hash is the C6 identity of these actual samples. Mallett: 243 samples,
 * wavelength-major RGB. CMF: 4*N in file order [nm,x_bar,y_bar,z_bar]; N may be
 * zero, with (NULL,0). Native CMF construction owns its distinct axis contract.
 * View/acquire failures clear valid outputs, including hashes. Matching release
 * consumes once even with malformed diagnostics. No allocation/I/O/hash on view.
 * Reconstruction/CMF read capacity -> AllocationFailure, other read failures ->
 * PreparationFailure, contained panic -> InternalFailure; API=None/code=0.
 * Explicit large-buffer capacity failures are recoverable; incidental Box/Arc/
 * path/parser allocation aborts are not. The shared pointer/diagnostic contract
 * above applies; spans expire on family release, excluded from all active reads. */
typedef struct FjSpectraLut FjSpectraLut;
typedef struct FjMallettBasis FjMallettBasis;
typedef struct FjCmf FjCmf;
typedef struct FjSpectraLutView {
    FjFloatSpan samples;
    uint64_t asset_hash;
} FjSpectraLutView;
FjStatus fj_legacy_hanatos_acquire(const FjAssets* assets, FjSpectraLut** out_lut, FjErrorBuffer* error);
FjStatus fj_legacy_arctic_acquire(const FjAssets* assets, FjSpectraLut** out_lut, FjErrorBuffer* error);
FjStatus fj_legacy_spectra_lut_view(const FjSpectraLut* lut, FjSpectraLutView* out_view, FjErrorBuffer* error);
FjStatus fj_legacy_spectra_lut_release(FjSpectraLut* lut, FjErrorBuffer* error);
FjStatus fj_legacy_mallett_acquire(const FjAssets* assets, FjMallettBasis** out_basis, FjErrorBuffer* error);
FjStatus fj_legacy_mallett_view(const FjMallettBasis* basis, FjFloatSpan* out_samples_rgb, FjErrorBuffer* error);
FjStatus fj_legacy_mallett_release(FjMallettBasis* basis, FjErrorBuffer* error);
FjStatus fj_legacy_cmf_acquire(const FjAssets* assets, FjCmf** out_cmf, FjErrorBuffer* error);
FjStatus fj_legacy_cmf_view(const FjCmf* cmf, FjFloatSpan* out_rows, FjErrorBuffer* error);
FjStatus fj_legacy_cmf_release(FjCmf* cmf, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: CSV source conversion and selected calibration; remove S4.E.
 * CSV source is one of the seven explicit tags below. Rows are 2*N f32 values
 * in authored [wavelength_nm,value] order; empty success is (NULL,0). A view
 * borrows its CsvPairs owner until matching release, independently of Assets
 * cache release/destruction. Native borrows checked rows through synchronous construction; no rows
 * reach CUDA/retirement state. Capacity failure is AllocationFailure, ordinary
 * CSV read failure PreparationFailure, poison/panic InternalFailure.
 * Calibration keys are exact length-delimited UTF-8 stock/illuminant strings;
 * empty and embedded NUL are preserved. Its result is owned by value. All four
 * domain outcomes return Success; malformed is still a recipe failure. Reported
 * read/probe OutOfMemory returns AllocationFailure, never a domain outcome.
 * Non-Found coefficients are zero; only Malformed has a known nonzero field.
 * Valid outputs clear before other validation. Matching CSV release consumes
 * once even for malformed diagnostics; NULL release succeeds. The shared
 * exclusive/disjoint-storage and optional bounded diagnostic contract applies.
 * Incidental infallible allocation can abort; no allocator recovery is promised. */
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
FjStatus fj_legacy_csv_acquire(const FjAssets* assets, uint32_t source, FjCsvPairs** out_pairs, FjErrorBuffer* error);
FjStatus fj_legacy_csv_view(const FjCsvPairs* pairs, FjFloatSpan* out_rows, FjErrorBuffer* error);
FjStatus fj_legacy_csv_release(FjCsvPairs* pairs, FjErrorBuffer* error);
FjStatus fj_legacy_neutral_calibration_lookup(const FjAssets* assets, FjStringView print_stock, FjStringView illuminant, FjStringView film_stock, FjNeutralCalibrationResult* out_result, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: synchronous illuminant values/source/lens binding; remove S4.E.
 * Samples are 81 relative-linear SPD values on 380..780 nm, 5 nm positions.
 * Rows count interleaved f32 scalars [wavelength_nm,value,...], not pairs.
 * Sources borrow immutable caller storage through return; no pointer is retained.
 * All declared output/error locations are required, aligned, writable and disjoint
 * from each other and inputs. Error backing storage follows FjErrorBuffer's bound.
 * Curves and coverage clear before fallible work. Coverage warnings are low=1,
 * high=2 with finite observed min/max; no finite range yields zero metadata.
 * Finish/release take and clear a valid io_lens before other validation, consuming
 * once even on malformed output/error or failure. NULL incoming release succeeds.
 * Stale/dangling handles, concurrent use and invalid storage violate the caller
 * contract. Prepare returns no handle on failure. Native retains sources only
 * through synchronous calls. Checked scratch failure is AllocationFailure;
 * incidental Box/Arc allocator abort is not recoverable by panic containment. */
typedef struct FjIlluminant {
    float samples[81];
} FjIlluminant;

/* SF_TEMP_BRIDGE_PRINT_ILLUMINANT: complete fixed samples; remove S4.E.
 * Edges are Y/M-lower/M-upper/C and transitions are in nm; cmy_cc is C/M/Y
 * Kodak CC. Params, failure and present error records are initialized/live;
 * source is exactly 81 initialized f32 elements. Records/source are aligned,
 * and byte extents must not overflow PTRDIFF_MAX or address arithmetic.
 * Writable outputs/error backing are exclusive and mutually disjoint from
 * each other and all inputs; immutable input aliases are allowed. No pointer
 * escapes synchronous return. Numerical address checks do not establish those
 * allocation, lifetime or initialization obligations.
 * Every independently writable result clears to +0 on failure. Sample errors
 * identify the first index; all other outcomes use UINT32_MAX. Optional error
 * NULL is status-only; capacity zero ignores backing; positive backing receives
 * bounded text/NUL and length excluding NUL. Malformed diagnostics yield RAW.
 */
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
FjStatus fj_legacy_print_illuminant_prepare(
    const FjPrintIlluminantParams* params,
    FjFloatSpan source,
    FjIlluminant* out_illuminant,
    FjPrintIlluminantFailure* out_failure,
    FjErrorBuffer* error);

/* SF_TEMP_BRIDGE_PRINT_PREFLASH; remove S4.E. Synchronous borrowed f32 inputs:
 * 81 film base-density samples, 81 already filtered light samples and 243 print
 * sensitivities, sample-major C/M/Y. Rust print::PreflashRaw produces the owned
 * triplet consumed by native print resource preparation before exposure scaling.
 * Counts, alignment and address arithmetic are checked before access. Callers
 * provide initialized live backing; immutable inputs may alias. All writable
 * result/failure/error records and diagnostic backing are exclusive, mutually
 * disjoint and disjoint from inputs. No pointer survives return.
 * Independently writable output triplets clear to +0 on every failure. Channel
 * is UINT32_MAX except CHANNEL (first failing f64 total, 0..2). Finite f64 totals
 * may successfully narrow to f32 infinity; native payload admission is later.
 * Error NULL is status-only; zero capacity ignores backing; positive backing
 * receives bounded text/NUL with length excluding NUL. Invalid diagnostics are RAW.
 */
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
FjStatus fj_legacy_print_preflash_prepare(
    const FjPrintPreflashInput* input,
    FjPrintPreflashRaw* out_raw,
    FjPrintPreflashFailure* out_failure,
    FjErrorBuffer* error);

/* SF_TEMP_BRIDGE_PRINT_BALANCE, retirement S4.E. Synchronous safe-core producer;
 * native recipe completion consumes the finished copy. Four immutable spans of
 * 81/243/243/81 f32 elements: film base density, sample-major C/M/Y dye density,
 * sample-major RGB print sensitivity and filtered main light. Source allocations
 * are initialized/aligned/live through return and may alias each other. No
 * scientific admission scan; reference_hash is opaque, including zero.
 * Mode wire 0..255 preserves native u8 behavior; wider values are malformed.
 * Output/failure/error record and positive-capacity diagnostic backing are live,
 * exclusive, mutually disjoint and disjoint from all input storage. Raw admission
 * checks extents/alignment/arithmetic, not allocation/initialization/lifetime.
 * Every independently writable result field clears to +0/0 on failure. Channel
 * is UINT32_MAX except the first failed baseline/compensated raw channel 0..2.
 * Optional error NULL is status-only; capacity zero never accesses backing;
 * positive capacities write bounded NUL-terminated text, length excludes NUL.
 */
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
FjStatus fj_legacy_print_balance_prepare(
    const FjPrintBalanceInput* input, FjPrintBalance* out_balance, FjPrintBalanceFailure* out_failure, FjErrorBuffer* error);

typedef struct FjIlluminantCoverage {
    uint32_t warnings;
    float min_nm;
    float max_nm;
} FjIlluminantCoverage;
typedef struct FjIlluminantLens FjIlluminantLens;
FjStatus fj_legacy_illuminant_from_samples(FjFloatSpan rows, FjIlluminant* out_curve, FjErrorBuffer* error);
FjStatus fj_legacy_illuminant_blackbody(float temperature_kelvin, FjIlluminant* out_curve, FjErrorBuffer* error);
FjStatus fj_legacy_illuminant_equal_energy(FjIlluminant* out_curve, FjErrorBuffer* error);
FjStatus fj_legacy_illuminant_tungsten_kg3(FjFloatSpan rows, FjIlluminant* out_curve, FjIlluminantCoverage* out_coverage, FjErrorBuffer* error);
FjStatus fj_legacy_illuminant_lens_prepare(FjFloatSpan rows, FjIlluminantLens** out_lens, FjIlluminantCoverage* out_coverage, FjErrorBuffer* error);
FjStatus fj_legacy_illuminant_lens_finish(FjIlluminantLens** io_lens, FjFloatSpan lens_rows, FjIlluminant* out_curve, FjIlluminantCoverage* out_coverage, FjErrorBuffer* error);
FjStatus fj_legacy_illuminant_lens_release(FjIlluminantLens** io_lens, FjErrorBuffer* error);

/* FJ_TEMP_BRIDGE: film TC preparation, sampling and allocation ownership; remove S4.E.
 * Input/consumed spans are initialized, aligned, live immutable storage through
 * synchronous return. Outputs, ownership/error records and nonempty error bytes
 * are exclusive/disjoint from each other and inputs. Inactive surface/hull spans
 * are ignored. Borrow the existing [192][192][81] source (C/M/wavelength), final
 * [81][3] RGB sensitivity and [81] SPD. Active Hanatos surface is [3][15], and
 * available active input hull is the native A8 center and closed [1025][2] polygon.
 * Only method 0 Hanatos irradiance and 2 Arctic reflectance construct a TC LUT;
 * all flags are 0/1. Missing active hull fails after integration. No source or
 * hull borrow is retained by the completed RGB/padding [192][192][4] result.
 * Build output must initially be all-zero empty; never overwrite a live token.
 * Success transfers the original Vec pointer/length/capacity with no extra
 * allocation/copy. Samples are readonly while the single owner is live. Release
 * consumes the exact unmodified token in the same loaded Rust module/allocator,
 * clears it, and accepts all-zero empty as a no-op. Copying a live record does
 * not create another owner. Corrupt/stale/duplicate/foreign tokens are outside
 * the contract. Final release may cross threads after all use ends. There is no
 * release fault, retry/retirement policy or diagnostic allocation. The module
 * must outlive every holder. Failed build/sample leaves no ready result.
 * Structural failures are UnsupportedInput; computed failures PreparationFailure;
 * failed reservation AllocationFailure; contained panic InternalFailure. Native
 * undefined coordinate conversions are checked failures, never zero substitutes.
 */
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
FjStatus fj_legacy_reconstruction_tc_lut(const FjFilmTcLutInput* input, FjOwnedFilmTcLut* out, FjErrorBuffer* error);
FjStatus fj_legacy_reconstruction_sample_tc_lut(FjFloatSpan lut, const float projected_xyz[3], float out_rgb[3], FjErrorBuffer* error);
FjStatus fj_legacy_reconstruction_release_tc_lut(FjOwnedFilmTcLut* owned);

/* SF_TEMP_BRIDGE_SCANNER_CORRECTION: complete synchronous host values; remove S4.E.
 * Immutable inputs may alias. Live initialized input allocations and exclusive,
 * mutually disjoint writable records/backing are caller obligations. Writable
 * storage must also be disjoint from inputs. No pointer escapes. Extent checks
 * cannot establish these lifetime/allocation obligations. Required Params/result/
 * failure records are admitted independently; every admitted result is cleared
 * on failure. Optional trace/error may be NULL. Zero diagnostic capacity touches
 * no backing; nonzero capacity reserves a terminating NUL. Input demand precedes
 * all active-storage access. Route/medium preserve the full native u8 domain;
 * medium 1 means print, every other admitted byte means film. Wire values above
 * 255 and unknown flags are RAW. Curves are sample-major C/M/Y triples; empty or
 * axis/row mismatch is left to the ordinary owning lookup. Spectral spans contain
 * exactly K consumed floats; baseline is consumed only when has_baseline=1.
 * Inactive success has canonical no-op values/schema 1/hash 0, distinct from the
 * all-zero semantic failure result. This full host record does not change the
 * compact permanent FjScannerCorrection CUDA execution record.
 */
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
FjStatus fj_legacy_scanner_direct_correction_prepare(
    const FjScannerCorrectionParams*, const FjScannerDirectCorrectionInput*, FjScannerColorCorrection*, FjScannerCorrectionFailure*, FjScannerCorrectionTrace*, FjErrorBuffer*);
FjStatus fj_legacy_scanner_print_correction_prepare(
    const FjScannerCorrectionParams*, const FjScannerPrintCorrectionInput*, FjScannerColorCorrection*, FjScannerCorrectionFailure*, FjScannerCorrectionTrace*, FjErrorBuffer*);
/* operation 0 direct / 1 print. Query does not consume the prepare fault selector.
 * required/failure are required, independently cleared. No active input is read.
 */
FjStatus fj_legacy_scanner_correction_input_demand(
    const FjScannerCorrectionParams*, uint32_t operation, uint32_t* required, FjScannerCorrectionFailure*, FjErrorBuffer*);

/* SF_TEMP_BRIDGE_SCANNER_LUT: remove S4.E after the last native preparation consumer.
 * A successful acquire owns all six immutable arrays; inputs expire at return.
 * Every aligned writable output clears independently before peer admission.
 * Records/backing must be live, initialized and disjoint; view excludes release.
 * Handles have unique live provenance. Release consumes a nonnull admitted owner
 * even on diagnostic rejection or contained panic; NULL release is a no-op.
 * Empty consumed spans are (NULL,0); ignored backing is never dereferenced.
 * Counts are elements, never bytes. No retained input, retain call or CUDA owner.
 */
typedef struct FjScannerLutOwner FjScannerLutOwner;
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
#define FJ_SCANNER_LUT_REASON_NONE UINT32_C(0)
#define FJ_SCANNER_LUT_REASON_RAW UINT32_C(1)
#define FJ_SCANNER_LUT_REASON_RESOLUTION UINT32_C(2)
#define FJ_SCANNER_LUT_REASON_SIZE UINT32_C(3)
#define FJ_SCANNER_LUT_REASON_ALLOCATION UINT32_C(4)
#define FJ_SCANNER_LUT_REASON_NONFINITE UINT32_C(5)
#define FJ_SCANNER_LUT_REASON_PANIC UINT32_C(6)
FjStatus fj_legacy_scanner_lut_acquire(const FjScannerLutInput*, FjScannerLutOwner**, FjScannerLutFailure*, FjErrorBuffer*);
FjStatus fj_legacy_scanner_lut_view(const FjScannerLutOwner*, FjScannerLutView*, FjScannerLutFailure*, FjErrorBuffer*);
FjStatus fj_legacy_scanner_lut_release(FjScannerLutOwner*, FjErrorBuffer*);

/* SF_TEMP_BRIDGE_SCANNER_COLOR: complete synchronous copied value; remove S4.E.
 * Input/result/failure/error and nonempty error backing are live, aligned,
 * addressable and disjoint; writable records are exclusive through return.
 * No pointer is retained. White words are supplied XYZ, matrices row-major.
 * Every present valid result/failure clears independently before admission.
 * Raw output bytes 0..255 retain identity; bytes 9..255 select sRGB constants.
 * This does not widen the permanent executor or strict catalogue export.
 */
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
FjStatus fj_legacy_scanner_color_prepare(const FjScannerColorInput*, FjScannerColor*, FjScannerColorFailure*, FjErrorBuffer*);

#ifdef __cplusplus
}
#endif
