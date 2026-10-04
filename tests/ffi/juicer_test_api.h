#ifndef FJ_JUICER_TEST_API_H
#define FJ_JUICER_TEST_API_H

#include "juicer_cuda_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* BUILD_TESTING's nondefault Rust test-support facade; never an installed API. */
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
