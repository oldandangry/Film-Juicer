#ifndef FJ_JUICER_TEST_API_H
#define FJ_JUICER_TEST_API_H

#include "juicer_legacy_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Direct-core exposure facade; same synchronous storage rules as production.
 * Closed raw fault slots: 1 reference, 2 sensitivity; fault 1 unsupported,
 * 2 panic, 3 allocation category, 4 preparation. One-based matching call count.
 * Read the consume-once same-thread witness immediately, before another call or
 * cleanup. The independent facade slot has its own consume-once witness.
 * Category injection is distinct from the core's real kernel reservation test. */
FjStatus fj_test_exposure_arm_fault(uint32_t operation, uint32_t call_index, uint32_t fault);
void fj_test_exposure_clear_fault(void);
uint32_t fj_test_exposure_fault_consumed(uint32_t operation, uint32_t fault);
void fj_test_exposure_arm_facade_fault(void);
uint32_t fj_test_exposure_facade_fault_consumed(void);
FjStatus fj_test_reconstruction_reference_white(const FjReferenceWhiteInput* input, FjReferenceWhite* out, FjErrorBuffer* error);
FjStatus fj_test_exposure_sensitivity(const FjSensitivityInput* input, FjSensitivity* out, FjSensitivityFailure* failure, FjErrorBuffer* error);
FjStatus fj_test_exposure_window_sample(float wavelength, FjFloatSpan params, float* out, FjErrorBuffer* error);

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

typedef struct FjFilmProfile FjFilmProfile;

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

#ifdef __cplusplus
}
#endif

#endif
