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

#ifdef __cplusplus
}
#endif
