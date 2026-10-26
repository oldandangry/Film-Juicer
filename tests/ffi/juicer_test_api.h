#ifndef FJ_JUICER_TEST_API_H
#define FJ_JUICER_TEST_API_H

#include "juicer_legacy_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Feature-only film-input facade; identical synchronous storage rules and safe constructor/projector. */
FjStatus fj_test_film_input_prepare(const FjFilmProfile*, const FjFilmInputParams*, const FjFilmInputResources*, FjFilmInput*, FjFilmInputFailure*, FjErrorBuffer*);
const size_t* fj_test_film_input_abi_facts(size_t* count);
/* 1 projection, 2 diagnostic formatting, 3 success cleanup; one-shot current-thread panic. */
void fj_test_film_input_arm_boundary_fault(uint32_t stage);

/* Direct-core exposure facade; same synchronous storage rules as production.
 * Closed raw fault slots: 1 reference, 2 sensitivity, 3 Mallett mid-gray,
 * 4 TC normalization, 5 reference source, 6 Mallett reference; fault 1 unsupported,
 * 2 panic, 3 allocation category, 4 preparation. One-based matching call count.
 * Read the consume-once same-thread witness immediately, before another call or
 * cleanup. The independent facade slot has its own consume-once witness.
 * Category injection is distinct from the core's real kernel reservation test.
 * Allocation fault 3 is supported only by operations 1/2, never fixed A7 math. */
FjStatus fj_test_exposure_arm_fault(uint32_t operation, uint32_t call_index, uint32_t fault);
void fj_test_exposure_clear_fault(void);
uint32_t fj_test_exposure_fault_consumed(uint32_t operation, uint32_t fault);
void fj_test_exposure_arm_facade_fault(void);
uint32_t fj_test_exposure_facade_fault_consumed(void);
FjStatus fj_test_reconstruction_reference_white(const FjReferenceWhiteInput* input, FjReferenceWhite* out, FjErrorBuffer* error);
FjStatus fj_test_exposure_sensitivity(const FjSensitivityInput* input, FjSensitivity* out, FjSensitivityFailure* failure, FjErrorBuffer* error);
FjStatus fj_test_exposure_window_sample(float wavelength, FjFloatSpan params, float* out, FjErrorBuffer* error);
FjStatus fj_test_exposure_mallett_midgray(const FjMallettMidgrayInput* input, FjMallettMidgray* out, FjErrorBuffer* error);
FjStatus fj_test_exposure_tc_midgray(float green, FjMidgrayNormalization* out, FjErrorBuffer* error);
FjStatus fj_test_exposure_reference_source(float exposure_ev, float* out, FjErrorBuffer* error);
FjStatus fj_test_exposure_mallett_reference_raw(const FjMallettReferenceInput* input, FjReferenceRaw* out, FjErrorBuffer* error);

/* Feature-only fixed-axis spectrum witnesses, independent of legacy color faults.
 * Borrow initialized aligned immutable inputs/spans only until return; exclusive
 * aligned output/error storage is mutually disjoint and disjoint from inputs.
 * Hanatos consumes exactly 192*192*81 samples, tables consume three 81-sample
 * weights. All outputs clear before fallible work. No source pointer is retained,
 * heap scratch, method selector or production fallback is supplied by this facade.
 * Closed fixture slots: 1 Hanatos spectrum, 2 tables spectrum, 3 Mallett BGR leaf;
 * faults 1 unsupported and 2 panic, with one-based matching-call counts and
 * consume-once same-thread witnesses read before cleanup. */
typedef struct FjHanatosSpectrumInput {
    float rgb_dwg[3], reference_white[3];
    FjFloatSpan spectra;
} FjHanatosSpectrumInput;
typedef struct FjTablesSpectrumInput {
    float rgb_dwg[3], reference_white[3], s_inverse[9];
    FjFloatSpan ax, ay, az;
} FjTablesSpectrumInput;
typedef struct FjMallettRawInput {
    float linear_srgb[3];
    FjFloatSpan basis_rgb, illuminant, sensitivity_rgb;
} FjMallettRawInput;
typedef struct FjSpectrumFixture {
    float pre_xyz[3], consumer_white[3], post_adapt_xyz[3], spectrum[81];
} FjSpectrumFixture;
FjStatus fj_test_exposure_hanatos_spectrum(const FjHanatosSpectrumInput* input, FjSpectrumFixture* out, FjErrorBuffer* error);
FjStatus fj_test_exposure_tables_spectrum(const FjTablesSpectrumInput* input, FjSpectrumFixture* out, FjErrorBuffer* error);
FjStatus fj_test_exposure_mallett_raw(const FjMallettRawInput* input, float out_bgr[3], FjErrorBuffer* error);
FjStatus fj_test_exposure_fixture_arm_fault(uint32_t operation, uint32_t index, uint32_t fault);
void fj_test_exposure_fixture_clear_fault(void);
uint32_t fj_test_exposure_fixture_fault_consumed(uint32_t operation, uint32_t fault);

/* Fixed spectral direct-core facade; same synchronous storage contract as production.
 * Independent TLS fault slots: raw operations 1 tables, 2 white, 3 inverse;
 * fault 1 unsupported, 2 panic, 3 allocation category, 4 preparation category.
 * Fixed math allocates no Rust heap: category injection does not test an allocator.
 * One-based matching call count is consumed before the numerical action. */
FjStatus fj_test_spectral_arm_fault(uint32_t operation, uint32_t call_index, uint32_t fault);
void fj_test_spectral_clear_fault(void);
void fj_test_spectral_arm_facade_fault(void);
FjStatus fj_test_spectral_tables(const FjSpectralInput* input, FjSpectralTables* out, FjErrorBuffer* error);
FjStatus fj_test_spectral_white(const FjSpectralWhiteInput* input, FjSpectralWhite* out, FjSpectralWhiteFailure* failure, FjErrorBuffer* error);
FjStatus fj_test_spectral_s_inverse(const FjSpectralSInput* input, FjSpectralInverse* out, FjErrorBuffer* error);

/* BUILD_TESTING's nondefault Rust test-support facade; never an installed API. */
/* Closed production math selectors; 1=unsupported, 2=panic, 3=real checked
 * scratch capacity, 4=preparation faults. TLS one-shot matching raw-call count.
 * The curve facade invokes core directly: 1..4 as below, 5 composed lens,
 * 8 private resampling projection. Its one-shot fault slot is independent. */
#define FJ_TEST_ILLUMINANT_FROM_SAMPLES UINT32_C(1)
#define FJ_TEST_ILLUMINANT_BLACKBODY UINT32_C(2)
#define FJ_TEST_ILLUMINANT_EQUAL_ENERGY UINT32_C(3)
#define FJ_TEST_ILLUMINANT_TUNGSTEN_KG3 UINT32_C(4)
#define FJ_TEST_ILLUMINANT_LENS_PREPARE UINT32_C(5)
#define FJ_TEST_ILLUMINANT_LENS_FINISH UINT32_C(6)
#define FJ_TEST_ILLUMINANT_LENS_RELEASE UINT32_C(7)
FjStatus fj_test_illuminant_arm_fault(uint32_t operation, uint32_t call_index, uint32_t fault);
void fj_test_illuminant_clear_fault(void);
size_t fj_test_illuminant_live_lenses(void);
void fj_test_illuminant_arm_facade_fault(void);
/* Ten initialized writable size_t elements; returns actual successful Vec count.
 * Values are byte capacities, cumulative within the last resampling call, not
 * simultaneous live bytes, allocator metadata or RSS. OFF builds omit probes. */
size_t fj_test_illuminant_scratch_capacities(size_t* out_bytes);
FjStatus fj_test_illuminant_curve(uint32_t operation, FjFloatSpan rows, FjFloatSpan lens_rows, float temperature_kelvin, FjIlluminant* out_curve, FjIlluminantCoverage* out_coverage, FjErrorBuffer* error);
/* CAT02/CAT16 value facade uses the production fixed-array pointer/extent contract.
 * Faults affect only the calling thread's actual production exports: one-based
 * matching call count, consumed before action. Invalid arm disarms; clear is
 * idempotent. Fault storage/branches/exports are absent without test-support. */
#define FJ_TEST_CAT16_MATRIX UINT32_C(1)
#define FJ_TEST_CAT16_ADAPT UINT32_C(2)
#define FJ_TEST_CAT02_MATRIX UINT32_C(3)
#define FJ_TEST_CAT02_ADAPT UINT32_C(4)
#define FJ_TEST_INPUT_MATRICES UINT32_C(5)
#define FJ_TEST_INPUT_TO_DWG UINT32_C(6)
#define FJ_TEST_INPUT_TO_LINEAR_SRGB UINT32_C(7)
#define FJ_TEST_LINEAR_SRGB_TO_XYZ UINT32_C(8)
#define FJ_TEST_DWG_TO_XYZ UINT32_C(9)
#define FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ UINT32_C(10)
#define FJ_TEST_COLOR_UNSUPPORTED_INPUT UINT32_C(1)
#define FJ_TEST_COLOR_PANIC UINT32_C(2)
FjStatus fj_test_cat16_matrix(const float source_white_xyz[3], const float destination_white_xyz[3], float out_row_major[9]);
FjStatus fj_test_adapt_cat16(const float xyz[3], const float source_white_xyz[3], const float destination_white_xyz[3], float out_xyz[3]);
FjStatus fj_test_cat02_matrix(const float source_white_xyz[3], const float destination_white_xyz[3], float out_row_major[9]);
FjStatus fj_test_adapt_cat02(const float xyz[3], const float source_white_xyz[3], const float destination_white_xyz[3], float out_xyz[3]);
FjStatus fj_test_input_matrices(uint32_t input_space, FjInputColorMatrices* out);
FjStatus fj_test_input_to_dwg(const FjInputColorConversion* input, const float rgb[3], uint32_t clamp_nonnegative, float out_rgb[3], float out_xyz[3]);
FjStatus fj_test_input_to_linear_srgb(const FjInputColorConversion* input, const float rgb[3], const float xyz_to_linear_srgb[9], float out_rgb[3], float out_xyz[3]);
FjStatus fj_test_linear_srgb_to_xyz(const float rgb[3], float out_xyz[3]);
FjStatus fj_test_dwg_to_xyz(const float rgb[3], float out_xyz[3]);
FjStatus fj_test_project_linear_rgb_to_xyz(const float rgb[3], const float rgb_to_xyz[9], const float xyz_adapt[9], float out_xyz[3]);

FjStatus fj_test_decode_input(uint32_t input_space, uint32_t decode_cctf, const float rgb[3], float out_rgb[3]);
FjStatus fj_test_color_arm_fault(uint32_t operation, uint32_t call_index, uint32_t fault);
FjStatus fj_test_color_clear_fault(void);
/* Direct current color-facade slots 4/6/7/9, independent of live production
 * export faults and never visited by pure core composition. Read the consumed
 * witness before cleanup; an armed/cleared slot is not consumption evidence. */
FjStatus fj_test_color_arm_facade_fault(uint32_t operation, uint32_t index, uint32_t fault);
FjStatus fj_test_color_clear_facade_fault(void);
uint32_t fj_test_color_facade_fault_consumed(uint32_t operation, uint32_t fault);

typedef struct FjFilmProfile FjFilmProfile;

/* SF_TEMP_BRIDGE_FILM_DEVELOPMENT: same internal constructor/projection as production;
 * opaque owner/view lifetime is identical; remove native transport at S4.E.
 * Fault operation 0 acquire, 1 view, 2 consuming release. Fault 0 clears,
 * 1 panics, acquire-only 2 rejects normalized reservation, 3/4 reject normalized/
 * final zero identities, 5 supplies the active-grain zero layer identity seam. */
FjStatus fj_test_film_development_acquire(const FjFilmProfile* profile, uint32_t requested_route, float gamma_factor, uint32_t layers_required, FjFilmDevelopmentOwner** out_development, FjErrorBuffer* error);
FjStatus fj_test_film_development_view(const FjFilmDevelopmentOwner* development,
                                       FjFilmDevelopmentView* out_view,
                                       FjErrorBuffer* error);
FjStatus fj_test_film_development_release(FjFilmDevelopmentOwner* development, FjErrorBuffer* error);
FjStatus fj_test_film_development_fault(uint32_t operation, uint32_t fault);
size_t fj_test_film_development_live_owners(void);
const size_t* fj_test_film_development_abi_facts(size_t* count);

#define FJ_PROFILE_USE_STILL UINT32_C(0)
#define FJ_PROFILE_USE_CINE UINT32_C(1)
#define FJ_PROFILE_ANTIHALATION_STRONG UINT32_C(0)
#define FJ_PROFILE_ANTIHALATION_WEAK UINT32_C(1)
#define FJ_PROFILE_ANTIHALATION_NO UINT32_C(2)

typedef struct FjDoubleSpan {
    const double* data;
    size_t count;
} FjDoubleSpan;

/* Counts are scalar elements. N = source_log_exposure.count = log_exposure.count.
 * Totals are 3*N exposure-major CMY; layers are [layer][channel][exposure], N each.
 * channel_density_cmy is 81*3 wavelength-major CMY; base_density is 81.
 * All spans borrow the immutable owner, independent of Assets/cache lifetime.
 * C copies of this record do not retain it. Empty spans are (NULL,0).
 * No full render/prepared-host record or unconsumed profile graph is exposed. */
typedef struct FjFilmFixtureView {
    uint32_t use;
    uint32_t antihalation;
    uint64_t asset_token;
    float halation_first_sigma_um[3];
    float halation_primary_amount[3];
    FjDoubleSpan source_log_exposure;
    FjFloatSpan log_exposure;
    FjFloatSpan density_curves_cmy;
    FjFloatSpan density_curves_layers[3][3];
    FjFloatSpan channel_density_cmy;
    FjFloatSpan base_density;
} FjFilmFixtureView;

/* Input strings are nonempty UTF-8 bytes without NUL, count <= PTRDIFF_MAX.
 * The caller authorizes readable initialized bytes until return. The root must
 * satisfy the real catalog's film/print role and default-key requirements.
 * Outputs/error storage are aligned, exclusively writable, mutually disjoint,
 * and do not overlap inputs or owner storage. NULL required outputs are errors.
 * Every valid output is cleared before input/diagnostic validation. Acquisition
 * publishes one complete owner or NULL. View failure leaves all fields zero and
 * spans empty; success reports the actual Rust token/arrays without recomputation.
 *
 * error may be NULL. Otherwise length resets to 0; capacity=0 never accesses data.
 * capacity (and data for nonzero capacity) are initialized input fields. Output
 * length and writable bytes need no prior initialization.
 * Nonzero capacity <= PTRDIFF_MAX requires writable data and reserves/emits a NUL.
 * length excludes NUL; truncation changes only text. Nonzero capacity with NULL
 * data or capacity > PTRDIFF_MAX returns UnsupportedInput. API=None/native_code=0 on all statuses.
 * Resource/decode/completion failures are PreparationFailure; explicit profile
 * capacity errors AllocationFailure; poisoned cache/contained panic InternalFailure.
 * Small Box/Arc/path/parser allocation aborts are not recoverable panic failures.
 *
 * Reads may run concurrently with disjoint outputs while the owner stays live.
 * No mutation/deletion of Rust buffers or release concurrent with any use/borrow.
 * Spans expire on consuming release. Native invocations still obey their stricter
 * return/staging-copy expiry. Expired pointers must never be used or released.
 * Each operation contains its own Rust panics; no foreign callbacks occur. */
FjStatus fj_test_film_profile_acquire(FjStringView resource_root, FjStringView key, FjFilmProfile** out_owner, FjErrorBuffer* error);
FjStatus fj_test_film_profile_view(const FjFilmProfile* owner,
                                   FjFilmFixtureView* out_view,
                                   FjErrorBuffer* error);

/* Consumes a nonnull live owner exactly once on EVERY outcome, including malformed
 * error output. NULL is a successful no-op unless the error buffer is malformed,
 * which takes precedence as UnsupportedInput and does not access its data. */
FjStatus fj_test_film_profile_release(FjFilmProfile* owner, FjErrorBuffer* error);

/* Static process-lifetime size_t facts. count is NULL or writable size_t storage; NULL returns NULL. */
const size_t* fj_test_profile_abi_facts(size_t* count);

/* Feature-only process/catalog ownership and one-shot projection fault probes. */
/* Classifier faults: 1-3 read OOM with true/false/failed probe, 4 probe OOM;
 * 5-7 ordinary read with true/false/failed probe. Zero clears injection.
 * Thread-local counts observe actual core loading and existence probing. */
void fj_test_calibration_read_fault(uint32_t mode);
size_t fj_test_calibration_reads(void);
size_t fj_test_calibration_probes(void);
size_t fj_test_csv_live_owners(void);
/* One-shot ABI classification: 1 contained panic, 2 CSV reader capacity, 3 CSV poison. */
void fj_test_csv_fault(uint32_t fault);
void fj_test_spectral_fault(uint32_t fault);
size_t fj_test_spectral_live_owners(void);
void fj_test_catalog_fault(uint32_t fault);
void fj_test_profile_fault(uint32_t fault);
size_t fj_test_profile_live_owners(void);
size_t fj_test_print_density_live_owners(void);
void fj_test_cache_release_failure(void);
size_t fj_test_assets_live_owners(void);
size_t fj_test_catalog_live_owners(void);

/* Fixture-only UTF-8, nonempty, NUL-free root; same complete safe producer and
 * borrowed FjStaticNoise as production. Pair with this facade's consuming release.
 * Shared owner/output/diagnostic/panic obligations apply, including malformed
 * diagnostic consuming release. No catalog/profile discovery is performed. */
typedef struct FjNoise FjNoise;
FjStatus fj_test_noise_acquire(FjStringView resource_root, FjNoise** out_owner, FjErrorBuffer* error);
FjStatus fj_test_noise_view(const FjNoise* owner, FjStaticNoise* out_view, FjErrorBuffer* error);
FjStatus fj_test_noise_release(FjNoise* owner, FjErrorBuffer* error);
/* Calling-thread bounded probes: 1=acquire panic, 2=view panic, 3=release panic. */
void fj_test_noise_fault(uint32_t fault);
size_t fj_test_noise_acquisition_count(void);
size_t fj_test_live_noise_owners(void);

/* Direct-core TC facade: same live-span and one-allocation ownership contract
 * as juicer_legacy_api.h; no legacy export is called. Separate facade fault
 * observes actual direct-core entry. Release has no injected work/failure policy. */
FjStatus fj_test_reconstruction_tc_lut(const FjFilmTcLutInput*, FjOwnedFilmTcLut*, FjErrorBuffer*);
FjStatus fj_test_reconstruction_sample_tc_lut(FjFloatSpan, const float[3], float[3], FjErrorBuffer*);
FjStatus fj_test_reconstruction_release_tc_lut(FjOwnedFilmTcLut*);
FjStatus fj_test_tc_lut_arm_fault(uint32_t operation, uint32_t index, uint32_t fault);
void fj_test_tc_lut_clear_fault(void);
uint32_t fj_test_tc_lut_fault_consumed(uint32_t operation, uint32_t fault);
FjStatus fj_test_tc_lut_allocation_counts(size_t out_counts[2]);
void fj_test_tc_lut_arm_facade_fault(void);
uint32_t fj_test_tc_lut_facade_fault_consumed(void);

/* Direct checked core hull facade; never calls the production legacy export.
 * facade 0 = production, 1 = fixture. Closed one-call faults: 1 structure,
 * 2 panic, 3 locus/center, 4 rays, 5 scale, 6 identity. No allocation fault.
 * Consume observation before clear; these are category witnesses, not preimages. */
FjStatus fj_test_gamut_input_hull(const FjInputHullSource*, FjInputHull*, FjErrorBuffer*);
FjStatus fj_test_gamut_arm_fault(uint32_t facade, uint32_t fault);
uint32_t fj_test_gamut_fault_consumed(uint32_t facade, uint32_t fault);
void fj_test_gamut_clear_fault(void);

/* Operation-specific A9 visits: 1 transform, 2 boundary; facade 0/1.
 * Fault 1 unsupported, 2 panic, 3 allocation transport; transform 4 finite,
 * 5 identity; boundary 4 transform, 5 c2_outside(3,7), 6 finite_exit, 7 identity. */
FjStatus fj_test_output_gamut_arm_fault(uint32_t operation, uint32_t facade, uint32_t fault);
void fj_test_output_gamut_clear_fault(void);
uint32_t fj_test_output_gamut_fault_consumed(uint32_t operation, uint32_t facade, uint32_t fault);
uint64_t fj_test_output_gamut_visits(uint32_t operation);
/* Calling-thread successful cold candidates: count, reserved bytes, copied bytes,
 * construction nanoseconds, copy nanoseconds; reset 0/1. No product profiling. */
FjStatus fj_test_output_gamut_copy_metrics(uint64_t out_metrics[5], uint32_t reset);
FjStatus fj_test_gamut_output_transform(const FjOutputGamutSource* source, FjOutputGamutTransform* out, FjErrorBuffer* error);
FjStatus fj_test_gamut_output_boundary(const FjOutputBoundarySource* source, float* cmax, size_t count, FjOutputBoundaryIdentity* identity, FjOutputBoundaryFailure* failure, FjErrorBuffer* error);

#define FJ_TEST_OUTPUT_COLOR 11U
#define FJ_TEST_OUTPUT_COLOR_FACADE 12U
FjStatus fj_test_output_color(uint32_t output_space, FjOutputColorPreparation* out);


FjStatus fj_test_film_density_minimum_prepare(
    const FjFilmDensityMinimum* requested,
    FjFilmDensityMinimum* out_minimum,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_test_film_dir_acquire(
    const FjFilmDevelopmentOwner* development,
    const FjFilmDirParams* params,
    FjFilmDirOwner** out_dir,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_test_film_dir_view(
    const FjFilmDirOwner* dir,
    FjFilmDirView* out_view,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_test_film_dir_release(
    FjFilmDirOwner* dir, FjErrorBuffer* error);
/* Identical fixed spatial-optics contract; forwards to the checked producer. */
FjStatus fj_test_spatial_optics_prepare(const FjSpatialOpticsParams* params, FjSpatialOpticsRecipe* out_recipe, FjSpatialOpticsFailure* out_failure, FjErrorBuffer* error);
FjStatus fj_test_physical_effects_prepare(
    const FjPhysicalEffectsParams* params,
    FjPhysicalEffectsRecipe* out_recipe,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);

FjStatus fj_test_visual_grain_prepare(
    const FjFilmDevelopmentOwner* development,
    const FjVisualGrainParams* params,
    FjVisualGrainRecipe* out_recipe,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_test_film_density_bounds_prepare(
    const FjFilmDevelopmentOwner* development,
    const FjFilmDensityMinimum* requested,
    FjFilmDensityBounds* out_bounds,
    FjFilmRecipeFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_test_film_development_retain(const FjFilmDevelopmentOwner*, FjFilmDevelopmentOwner**, FjErrorBuffer*);
/* Bounded scalar witnesses in the native domain primitive mode. out is five
 * exclusive doubles (power, transmitted, NaN-adjusted light, product, sum),
 * narrowed is one exclusive float. All storage lives through synchronous return. */
/* Primitive-only witness: density has seven f32 values (base, C/M/Y reference,
 * C/M/Y dye), raw three RGB values. out holds 12 doubles, narrowed two floats
 * (factor, ratio to denominator). Exclusive outputs expire at synchronous return. */
void fj_test_print_balance_primitive(const float* density, const float* raw, float denominator, double* out, float* narrowed);
/* Primitive-only scanner witness: six float and three double outputs. */
void fj_test_scanner_correction_primitive(float value, float accumulator, float multiplier, float* out, double* out_double);
FjStatus fj_test_film_reference_prepare(const FjFilmDevelopmentOwner*, const FjFilmReferenceInput*, FjFilmReference*, FjFilmReferenceFailure*, FjErrorBuffer*);
FjStatus fj_test_film_reference_density_lookup(float, FjFloatSpan, FjFloatSpan, uint32_t, float*, uint32_t*);
void fj_test_print_preflash_primitive(double density, double illuminant, double sensitivity, double accumulator, double* out, float* narrowed);
FjStatus fj_test_print_balance_prepare(
    const FjPrintBalanceInput* input, FjPrintBalance* out_balance, FjPrintBalanceFailure* out_failure, FjErrorBuffer* error);
FjStatus fj_test_print_preflash_prepare(
    const FjPrintPreflashInput* input, FjPrintPreflashRaw* out_raw, FjPrintPreflashFailure* out_failure, FjErrorBuffer* error);
FjStatus fj_test_print_illuminant_prepare(
    const FjPrintIlluminantParams* params,
    FjFloatSpan source,
    FjIlluminant* out_illuminant,
    FjPrintIlluminantFailure* out_failure,
    FjErrorBuffer* error);
FjStatus fj_test_film_recipe_fault(uint32_t stage, uint32_t fault);
size_t fj_test_film_dir_live_owners(void);
const size_t* fj_test_film_recipe_abi_facts(size_t* count);
FjStatus fj_test_scanner_direct_correction_prepare(
    const FjScannerCorrectionParams*, const FjScannerDirectCorrectionInput*, FjScannerColorCorrection*, FjScannerCorrectionFailure*, FjScannerCorrectionTrace*, FjErrorBuffer*);
FjStatus fj_test_scanner_print_correction_prepare(
    const FjScannerCorrectionParams*, const FjScannerPrintCorrectionInput*, FjScannerColorCorrection*, FjScannerCorrectionFailure*, FjScannerCorrectionTrace*, FjErrorBuffer*);
FjStatus fj_test_scanner_correction_fault(uint32_t operation, uint32_t fault);
FjStatus fj_test_scanner_lookup(float query, FjFloatSpan axis, FjFloatSpan values, float* out, uint32_t* present);
FjStatus fj_test_scanner_density_curve(float query, FjFloatSpan axis, FjFloatSpan curves_cmy, uint32_t channel, float* out, uint32_t* present);
FjStatus fj_test_scanner_lut_acquire(const FjScannerLutInput*, FjScannerLutOwner**, FjScannerLutFailure*, FjErrorBuffer*);
FjStatus fj_test_scanner_lut_view(const FjScannerLutOwner*, FjScannerLutView*, FjScannerLutFailure*, FjErrorBuffer*);
FjStatus fj_test_scanner_lut_release(FjScannerLutOwner*, FjErrorBuffer*);
FjStatus fj_test_scanner_lut_fault(uint32_t operation, uint32_t fault);
size_t fj_test_scanner_lut_live_owners(void);
FjStatus fj_test_scanner_color_prepare(const FjScannerColorInput*, FjScannerColor*, FjScannerColorFailure*, FjErrorBuffer*);

FjStatus fj_test_scanner_lut_describe(const FjScannerLutDescriptorInput*, FjScannerLutDescriptor*, FjScannerLutDescriptorFailure*, FjErrorBuffer*);
FjStatus fj_test_scanner_lut_descriptor_fault(uint32_t fault);

/* Same synchronous scalar-copy contract as production. */
FjStatus fj_test_scanner_post_describe(const FjScannerPostEffectsInput*, FjScannerPostEffectsDescriptor*, FjScannerPostEffectsFailure*, FjErrorBuffer*);
/* Thread-local one-shot: 0 clear; 1 active hash0; 2 panic. Early errors retain it. */
FjStatus fj_test_scanner_post_fault(uint32_t);
/* One-shot per-operation Gaussian faults: 0 clears; 1 panics; acquire 2
 * rejects its coefficient reservation; view 2/3/4 corrupt count/sigma/alignment.
 * Other combinations are unsupported. Counters are monotonic, never reset. */
FjStatus fj_test_gaussian_kernel_acquire(const FjGaussianKernelInput*, FjGaussianKernelOwner**, FjGaussianKernelFailure*, FjErrorBuffer*);
FjStatus fj_test_gaussian_kernel_view(const FjGaussianKernelOwner*, FjGaussianKernelView*, FjGaussianKernelFailure*, FjErrorBuffer*);
FjStatus fj_test_gaussian_kernel_release(FjGaussianKernelOwner*, FjErrorBuffer*);
FjStatus fj_test_gaussian_kernel_fault(uint32_t operation, uint32_t fault);
size_t fj_test_gaussian_kernel_acquires(void);
size_t fj_test_gaussian_kernel_live_owners(void);

FjStatus fj_test_frame_domain_prepare(const FjRect*, FjExtent*, FjFrameGeometryFailure*, FjErrorBuffer*);
FjStatus fj_test_frame_geometry_prepare(const FjFrameGeometryInput*, FjFrameGeometry*, FjFrameGeometryFailure*, FjErrorBuffer*);
FjStatus fj_test_frame_geometry_fault(uint32_t operation, uint32_t fault);
FjStatus fj_test_frame_domain_c_prepare(const FjRect*, FjExtent*, FjFrameGeometryFailure*, FjErrorBuffer*);
FjStatus fj_test_frame_geometry_c_prepare(const FjFrameGeometryInput*, FjFrameGeometry*, FjFrameGeometryFailure*, FjErrorBuffer*);

FjStatus fj_test_scatter_halation_frame_prepare(const FjHalationFrameInput*, FjScatterHalation*, FjHalationFrameFailure*, FjErrorBuffer*);
FjStatus fj_test_scatter_halation_frame_fault(uint32_t fault);
FjStatus fj_test_scatter_halation_frame_c_prepare(const FjHalationFrameInput*, FjScatterHalation*, FjHalationFrameFailure*, FjErrorBuffer*);
int fj_test_scatter_halation_frame_c_check(const FjHalationFrameInput*, const FjScatterHalation*, const FjHalationFrameFailure*, uint32_t category, const char* diagnostic);

FjStatus fj_test_diffusion_sample_prepare(const FjDiffusionSampleInput*, FjDiffusionSample*, FjDiffusionFailure*, FjErrorBuffer*);
FjStatus fj_test_diffusion_sample_c_prepare(const FjDiffusionSampleInput*, FjDiffusionSample*, FjDiffusionFailure*, FjErrorBuffer*);
FjStatus fj_test_diffusion_components_prepare(const FjDiffusionSample*, FjDiffusionComponents*, FjDiffusionFailure*, FjErrorBuffer*);
FjStatus fj_test_diffusion_components_c_prepare(const FjDiffusionSample*, FjDiffusionComponents*, FjDiffusionFailure*, FjErrorBuffer*);
FjStatus fj_test_diffusion_frame_prepare(const FjDiffusionFrameInput*, FjDiffusionFrameSet*, FjDiffusionFailure*, FjErrorBuffer*);
FjStatus fj_test_diffusion_frame_c_prepare(const FjDiffusionFrameInput*, FjDiffusionFrameSet*, FjDiffusionFailure*, FjErrorBuffer*);
FjStatus fj_test_diffusion_frame_fault(uint32_t operation, uint32_t fault);

FjStatus fj_test_dir_frame_prepare(const FjDirFrameInput*, FjDirFrameMath*, FjDirFrameFailure*, FjErrorBuffer*);
FjStatus fj_test_dir_frame_fault(uint32_t fault);
FjStatus fj_test_dir_frame_c_prepare(const FjDirFrameInput*, FjDirFrameMath*, FjDirFrameFailure*, FjErrorBuffer*);
int fj_test_dir_frame_c_check(const FjDirFrameInput*, const FjDirFrameMath*, const FjDirFrameFailure*, uint32_t category, const char* diagnostic);

/* Same live storage, owner/view expiry and consume-once contract as the legacy
 * DIR boundary edge. Fault operation: describe0/acquire1/view2/release3.
 * Fault0 clears, panic1; acquire2..8 reservations, view9..12 metadata only.
 * Counts are acquire/view/release calls and current live owners respectively. */
FjStatus fj_test_dir_boundary_describe(const FjDirBoundaryInput*, uint64_t*, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_test_dir_boundary_acquire(const FjDirBoundaryInput*, FjDirBoundaryOwner**, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_test_dir_boundary_view(const FjDirBoundaryOwner*, FjDirBoundaryView*, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_test_dir_boundary_release(FjDirBoundaryOwner*, FjErrorBuffer*);
FjStatus fj_test_dir_boundary_fault(uint32_t operation, uint32_t fault);
uint64_t fj_test_dir_boundary_count(uint32_t index);


FjStatus fj_test_dir_boundary_c_describe(const FjDirBoundaryInput*, uint64_t*, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_test_dir_boundary_c_acquire(const FjDirBoundaryInput*, FjDirBoundaryOwner**, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_test_dir_boundary_c_view(const FjDirBoundaryOwner*, FjDirBoundaryView*, FjDirBoundaryFailure*, FjErrorBuffer*);
FjStatus fj_test_dir_boundary_c_release(FjDirBoundaryOwner*, FjErrorBuffer*);
#ifdef __cplusplus
}
#endif

#endif
