#ifndef FJ_JUICER_TEST_API_H
#define FJ_JUICER_TEST_API_H

#include "juicer_test_values.h"
#include "juicer_cuda_call_scope.h"

#define FJ_UNAVAILABLE_STATE 0U
#define FJ_UNAVAILABLE_INSTANCE 1U

#ifdef __cplusplus
extern "C" {
#endif

/* Checked mathematical dependency boundaries preserve the original fault
 * controls independently of the direct-core numerical fixture calls below.
 * Inputs/owned holds and exclusive outputs follow the same synchronous,
 * disjoint, non-retaining facade contract. No legacy export is called. */
FjStatus fj_test_boundary_cat16_matrix(const float source_white_xyz[3], const float destination_white_xyz[3], float out_row_major[9]);
FjStatus fj_test_boundary_adapt_cat16(const float xyz[3], const float source_white_xyz[3], const float destination_white_xyz[3], float out_xyz[3]);
FjStatus fj_test_boundary_cat02_matrix(const float source_white_xyz[3], const float destination_white_xyz[3], float out_row_major[9]);
FjStatus fj_test_boundary_input_matrices(uint32_t input_space, FjInputColorMatrices* out);
FjStatus fj_test_boundary_linear_srgb_to_xyz(const float rgb[3], float out_xyz[3]);
FjStatus fj_test_boundary_project_linear_rgb_to_xyz(const float rgb[3], const float rgb_to_xyz[9], const float xyz_adapt[9], float out_xyz[3]);
FjStatus fj_test_boundary_output_color(uint32_t output_space, FjOutputColorPreparation* out);
FjStatus fj_test_boundary_reconstruction_tc_lut(const FjFilmTcLutInput* input, FjOwnedFilmTcLut* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_reconstruction_sample_tc_lut(FjFloatSpan lut, const float projected_xyz[3], float out_rgb[3], FjErrorBuffer* error);
FjStatus fj_test_boundary_reconstruction_release_tc_lut(FjOwnedFilmTcLut* owned);
FjStatus fj_test_boundary_reconstruction_reference_white(const FjReferenceWhiteInput* input, FjReferenceWhite* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_exposure_sensitivity(const FjSensitivityInput* input, FjSensitivity* out, FjSensitivityFailure* out_failure, FjErrorBuffer* error);
FjStatus fj_test_boundary_exposure_mallett_midgray(const FjMallettMidgrayInput* input, FjMallettMidgray* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_exposure_tc_midgray(float green, FjMidgrayNormalization* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_exposure_reference_source(float exposure_ev, float* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_exposure_mallett_reference_raw(const FjMallettReferenceInput* input, FjReferenceRaw* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_spectral_tables(const FjSpectralInput* input, FjSpectralTables* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_spectral_white(const FjSpectralWhiteInput* input, FjSpectralWhite* out, FjSpectralWhiteFailure* out_failure, FjErrorBuffer* error);
FjStatus fj_test_boundary_spectral_s_inverse(const FjSpectralSInput* input, FjSpectralInverse* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_gamut_input_hull(const FjInputHullSource* source, FjInputHull* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_gamut_output_transform(const FjOutputGamutSource* source, FjOutputGamutTransform* out, FjErrorBuffer* error);
FjStatus fj_test_boundary_gamut_output_boundary(const FjOutputBoundarySource* source, float* cmax, size_t count, FjOutputBoundaryIdentity* identity, FjOutputBoundaryFailure* failure, FjErrorBuffer* error);

/* Fixture-only ownership uses the same checked operations as complete
 * preparation. Inputs are initialized, aligned and live through synchronous
 * return. Outputs/error backing are exclusive and disjoint from owners/inputs.
 * Returned owners are consumed once, excluding readers; views expire on release.
 * Cache release drops cache holds while separately held results remain live.
 * Calls run on the calling thread, retain no raw input and contain panics. */
FjStatus fj_test_assets_create(FjPathView, FjAssets**, FjErrorBuffer*);
FjStatus fj_test_assets_destroy(FjAssets*, FjErrorBuffer*);
FjStatus fj_test_assets_release_cached_payloads(const FjAssets*, FjErrorBuffer*);
FjStatus fj_test_assets_release_derived_payloads(const FjAssets*, FjErrorBuffer*);
FjStatus fj_test_assets_release_source_payloads(const FjAssets*, FjErrorBuffer*);
FjStatus fj_test_assets_film_acquire(const FjAssets*, FjStringView, FjFilmProfile**, FjErrorBuffer*);
FjStatus fj_test_assets_film_view(const FjFilmProfile*, FjFilmProfileView*, FjErrorBuffer*);
FjStatus fj_test_assets_film_release(FjFilmProfile*, FjErrorBuffer*);
FjStatus fj_test_assets_film_defaults(const FjAssets*, FjStringView, FjFilmDigest*, FjErrorBuffer*);
FjStatus fj_test_assets_noise_acquire(const FjAssets*, FjNoise**, FjErrorBuffer*);
FjStatus fj_test_csv_acquire(const FjAssets*, uint32_t, FjCsvPairs**, FjErrorBuffer*);
FjStatus fj_test_csv_view(const FjCsvPairs*, FjFloatSpan*, FjErrorBuffer*);
FjStatus fj_test_csv_release(FjCsvPairs*, FjErrorBuffer*);
FjStatus fj_test_illuminant_from_samples(FjFloatSpan, FjIlluminant*, FjErrorBuffer*);
/* A non-null output is one exclusive writable byte; invalid input leaves it unchanged. */
uint32_t fj_test_resolve_route(uint8_t polarity, uint8_t selection, uint8_t* out_route);
FjStatus fj_test_catalog_acquire(const FjAssets* assets, FjCatalog** out_catalog, FjCatalogCounts* out_counts, FjErrorBuffer* error);
FjStatus fj_test_catalog_entry(const FjCatalog* catalog, uint32_t role, size_t index, FjCatalogEntryView* out_entry, FjErrorBuffer* error);
FjStatus fj_test_catalog_release(FjCatalog* catalog, FjErrorBuffer* error);
/* One-shot hook runs before the next catalog output publication on this thread,
 * with no catalog lock held. Callback/user stay live until invocation or explicit
 * disarm (null callback), contain exceptions and retain no Rust borrow. */
void fj_test_catalog_publication_hook(void (*callback)(void*), void* user);
/* Numeric shared-backing identity is observational; it authorizes no dereference
 * and expires with the held owner. */
FjStatus fj_test_catalog_backing_identity(const FjCatalog*, uintptr_t*, FjErrorBuffer*);
FjStatus fj_test_print_profile_acquire(const FjAssets* assets, FjStringView key, FjPrintProfile** out_profile, FjErrorBuffer* error);
FjStatus fj_test_print_profile_view(const FjPrintProfile* profile,
                                    FjPrintProfileView* out_view,
                                    FjErrorBuffer* error);
FjStatus fj_test_print_profile_release(FjPrintProfile* profile, FjErrorBuffer* error);
FjStatus fj_test_print_profile_sample_density(const FjPrintProfile* profile,
                                              double gamma,
                                              FjPrintDensityCurves** out_curves,
                                              FjErrorBuffer* error);
FjStatus fj_test_print_density_view(const FjPrintDensityCurves* curves,
                                    FjPrintDensityView* out_view,
                                    FjErrorBuffer* error);
FjStatus fj_test_print_density_release(FjPrintDensityCurves* curves, FjErrorBuffer* error);
FjStatus fj_test_hanatos_acquire(const FjAssets* assets, FjSpectraLut** out_lut, FjErrorBuffer* error);
FjStatus fj_test_arctic_acquire(const FjAssets* assets, FjSpectraLut** out_lut, FjErrorBuffer* error);
FjStatus fj_test_mallett_acquire(const FjAssets* assets, FjMallettBasis** out_basis, FjErrorBuffer* error);
FjStatus fj_test_cmf_acquire(const FjAssets* assets, FjCmf** out_cmf, FjErrorBuffer* error);
FjStatus fj_test_spectra_lut_view(const FjSpectraLut* lut, FjSpectraLutView* out_view, FjErrorBuffer* error);
FjStatus fj_test_spectra_lut_release(FjSpectraLut* lut, FjErrorBuffer* error);
FjStatus fj_test_mallett_view(const FjMallettBasis* basis, FjFloatSpan* out_samples_rgb, FjErrorBuffer* error);
FjStatus fj_test_mallett_release(FjMallettBasis* basis, FjErrorBuffer* error);
FjStatus fj_test_cmf_view(const FjCmf* cmf, FjFloatSpan* out_rows, FjErrorBuffer* error);
FjStatus fj_test_cmf_release(FjCmf* cmf, FjErrorBuffer* error);
FjStatus fj_test_neutral_calibration_lookup(const FjAssets* assets, FjStringView print_stock, FjStringView illuminant, FjStringView film_stock, FjNeutralCalibrationResult* out_result, FjErrorBuffer* error);
FjStatus fj_test_illuminant_blackbody(float temperature_kelvin, FjIlluminant* out_curve, FjErrorBuffer* error);
FjStatus fj_test_illuminant_equal_energy(FjIlluminant* out_curve, FjErrorBuffer* error);
FjStatus fj_test_illuminant_tungsten_kg3(FjFloatSpan rows, FjIlluminant* out_curve, FjIlluminantCoverage* out_coverage, FjErrorBuffer* error);
/* Finish and release take the exclusive lens slot, clear it before validating
 * other inputs, and consume its matching owner even on failure. No reader may
 * overlap that consumption. Input rows are borrowed only through return. */
FjStatus fj_test_illuminant_lens_prepare(FjFloatSpan rows, FjIlluminantLens** out_lens, FjIlluminantCoverage* out_coverage, FjErrorBuffer* error);
FjStatus fj_test_illuminant_lens_finish(FjIlluminantLens** io_lens, FjFloatSpan lens_rows, FjIlluminant* out_curve, FjIlluminantCoverage* out_coverage, FjErrorBuffer* error);
FjStatus fj_test_illuminant_lens_release(FjIlluminantLens** io_lens, FjErrorBuffer* error);
FjStatus fj_test_scanner_correction_input_demand(const FjScannerCorrectionParams*, uint32_t, uint32_t*, FjScannerCorrectionFailure*, FjErrorBuffer*);

/* Complete owners and call tables stay live through every synchronous callback;
 * all callbacks contain exceptions, run serially on the calling thread and
 * retain nothing. Native staging owns asynchronous reads before return. No
 * owner is consumed by policy/render. The pointer-free policy belongs to the
 * same held state/full frame. Unavailable admits only missing-state/null or
 * missing-instance/held-state pairs. Disjoint diagnostics never alias inputs. */
FjStatus fj_test_prepared_state_frame_policy(const FjPreparedState*, const FjExtent*, FjPreparedFramePolicy*, FjErrorBuffer*);
FjCallCompletionResult fj_test_prepared_state_render(const FjAssets*, const FjPreparedState*, FjCudaCallAccess, const FjFrame*, const FjPreparedFramePolicy*, const FjInstance*, uint64_t, FjAbortCallback, FjErrorBuffer*);
FjCallCompletionResult fj_test_prepared_unavailable(const FjAssets*, const FjPreparedState*, uint32_t, FjCudaCallAccess, const FjFrame*, FjAbortCallback, FjErrorBuffer*);

/* Exact capture/write/snapshot/admission and Runtime protocols. Owners remain
 * Rust-owned; release consumes once with callback/read exclusion. No host call
 * holds an exclusive Rust borrow or a publication/UI lock. All outputs and
 * diagnostics are live, initialized, aligned, exclusive and disjoint. */
FjStatus fj_test_runtime_create(FjPathView resource_dir, FjRuntime** out, FjErrorBuffer* error);
FjStatus fj_test_runtime_borrow(const FjRuntime* runtime, FjRuntimeView* out, FjErrorBuffer* error);
FjStatus fj_test_runtime_check_close(const FjRuntime* runtime, FjErrorBuffer* error);
FjStatus fj_test_runtime_shutdown(const FjRuntime* runtime, FjErrorBuffer* error);
FjStatus fj_test_runtime_destroy(FjRuntime* runtime, FjErrorBuffer* error);
FjStatus fj_test_instance_create(FjInstance** out, FjErrorBuffer* error);
FjStatus fj_test_instance_bind_identity(const FjInstance*, uint64_t session_seed, uint64_t instance_token, FjErrorBuffer* error);
FjStatus fj_test_instance_identity(const FjInstance*, uint32_t* out_initialized, uint64_t* out_seed, uint64_t* out_token, FjErrorBuffer* error);
FjStatus fj_test_instance_invalidate_submission(const FjInstance*, int32_t device_id, uintptr_t context, uint32_t* out_cleared, FjErrorBuffer* error);
FjStatus fj_test_instance_begin_capture(FjInstance*, uint32_t mode, FjCaptureStart* out, FjErrorBuffer* error);
FjStatus fj_test_instance_capture_is_current(const FjInstance*, uint64_t capture, uint32_t* out_current, FjErrorBuffer* error);
FjStatus fj_test_instance_begin_host_write(FjInstance*, uint64_t capture, FjHostWrite** out, uint32_t* out_start, FjErrorBuffer* error);
FjStatus fj_test_host_write_finish(FjHostWrite*, uint32_t failure_kind, FjStringView diagnostic, uint32_t* out_refresh, FjErrorBuffer* error);
FjStatus fj_test_instance_begin_snapshot(FjInstance*, uint64_t capture, FjSnapshotStart* out, FjErrorBuffer* error);
FjStatus fj_test_instance_install(FjInstance*, uint64_t capture, uint64_t write_revision, const FjPreparationParams*, uint64_t input_identity, uint32_t* out_update, FjErrorBuffer* error);
FjStatus fj_test_instance_invalidate(FjInstance*, uint64_t capture, uint32_t failure_kind, FjStringView diagnostic, uint32_t* out_update, FjErrorBuffer* error);
FjStatus fj_test_instance_admit(FjInstance*, const FjAssets*, FjAdmission* out, FjErrorBuffer* error);
FjStatus fj_test_instance_destroy(FjInstance*, FjErrorBuffer* error);

typedef struct FjHostPreparation FjHostPreparation;
typedef FjStatus (*FjHostPreparationConsumer)(void*, const FjHostPreparation*, FjErrorBuffer*);
/* Test-support only: backing expires after this synchronous consumer returns.
 * The consumer must contain all exceptions and retain no pointer or callback.
 * Preparation callbacks run serially on the calling thread. */
FjStatus fj_test_with_host_preparation(void*, FjHostPreparationConsumer, FjErrorBuffer*);
FjStatus fj_test_mallett_source_override(FjPathView, FjErrorBuffer*);
void fj_test_mallett_source_clear(void);
uint32_t fj_test_mallett_source_consumed(void);
FjStatus fj_test_film_reference_lookup_fault(uint32_t mode);
uint32_t fj_test_film_reference_lookup_fault_consumed(void);

/* Owner-bound complete preparation/cache facade, sharing private construction
 * and projection with production. Same synchronous and consuming contracts as
 * the sole held-state section in juicer_legacy_api.h. */
FjStatus fj_test_assets_illuminants_acquire(const FjAssets*, FjIlluminantsOwner**, FjErrorBuffer*);
FjStatus fj_test_illuminants_view(const FjIlluminantsOwner*, FjIlluminantsView*, FjErrorBuffer*);
FjStatus fj_test_illuminants_release(FjIlluminantsOwner*, FjErrorBuffer*);
FjStatus fj_test_assets_input_hull_acquire(const FjAssets*, FjInputHullOwner**, FjErrorBuffer*);
FjStatus fj_test_input_hull_view(const FjInputHullOwner*, FjInputHullView*, FjErrorBuffer*);
FjStatus fj_test_input_hull_release(FjInputHullOwner*, FjErrorBuffer*);
FjStatus fj_test_assets_output_boundary_acquire(const FjAssets*, const FjOutputBoundarySource*, uint64_t expected_contract_hash, FjOutputBoundaryOwner**, FjErrorBuffer*);
FjStatus fj_test_output_boundary_view(const FjOutputBoundaryOwner*, FjOutputBoundaryView*, FjErrorBuffer*);
FjStatus fj_test_output_boundary_release(FjOutputBoundaryOwner*, FjErrorBuffer*);
FjStatus fj_test_prepared_state_create(const FjAssets*, const FjPreparationParams*, FjPreparedState**, FjErrorBuffer*);
/* Copies Rust capture defaults with both key views empty. Output and diagnostic
 * storage are live, aligned, exclusive, and disjoint through this synchronous
 * call. No pointer is retained and no foreign callback is invoked. */
FjStatus fj_test_preparation_defaults(FjPreparationParams* out, FjErrorBuffer* error);
FjStatus fj_test_capture_gamma(uint32_t route, double film, double print, float*, double*, FjErrorBuffer*);
FjStatus fj_test_prepared_state_view(const FjPreparedState*, FjPreparedStateView*, FjErrorBuffer*);
FjStatus fj_test_prepared_state_release(FjPreparedState*, FjErrorBuffer*);
/* Test-only borrowed observation immediately before a complete prepared state
 * is released. The callback runs outside the observer lock, receives the
 * state's TC LUT samples or null, and must not retain the pointer. Passing null
 * clears the callback. */
void fj_test_prepared_state_release_observer(void (*)(const float*));
/* One-shot, arming-thread observation after actual profile/hull selection and
 * before complete preparation. Spans are borrowed only during the callback;
 * it contains all foreign exceptions, runs outside cache locks and supplies no
 * policy. Null clears the observation. Address identities may be copied for
 * comparison with a subsequently admitted held state, never dereferenced alone. */
void fj_test_prepared_input_observer(void (*)(FjFloatSpan profile_axis, uint64_t profile_token, FjFloatSpan hull_xy, uint64_t hull_hash));
const size_t* fj_test_prepared_state_abi_facts(size_t* count);
size_t fj_test_prepared_live_states(void);

/* Owner cache observations are non-default test controls. The two callbacks
 * borrow their table only during acquire, run outside owner locks, contain all
 * C++ exceptions, and return an FJ_STATUS category. No callback is retained. */
typedef struct FjOutputCacheHooks {
    uint32_t (*after_miss)(uint64_t generation);
    uint32_t (*after_selection)(uint32_t warm, uint64_t generation);
} FjOutputCacheHooks;
typedef struct FjIlluminantCacheHooks {
    /* source uses FJ_CSV_*; lens_prepared is 0/1 at the actual source visit. */
    uint32_t (*before_source)(uint32_t source, uint32_t lens_prepared);
    uint32_t (*before_publication)(uintptr_t entry, size_t backing_bytes);
} FjIlluminantCacheHooks;
typedef struct FjInputHullCacheHooks {
    uint32_t (*before_build)(uintptr_t entry, size_t backing_bytes);
} FjInputHullCacheHooks;
typedef struct FjCacheObservationOwner FjCacheObservationOwner;
typedef struct FjCacheMetrics {
    uint64_t allocations, derivations, destructions, live_bytes, peak_bytes;
    uint64_t reentries, reentry_failures;
} FjCacheMetrics;
#define FJ_TEST_CACHE_ILLUMINANTS 0U
#define FJ_TEST_CACHE_INPUT_HULL 1U
#define FJ_TEST_CACHE_OUTPUT_BOUNDARY 2U
FjStatus fj_test_assets_output_boundary_acquire_observed(const FjAssets*, const FjOutputBoundarySource*, uint64_t expected_contract, const FjOutputCacheHooks*, FjOutputBoundaryOwner**, FjErrorBuffer*);
FjStatus fj_test_assets_illuminants_acquire_observed(const FjAssets*, const FjIlluminantCacheHooks*, FjIlluminantsOwner**, FjErrorBuffer*);
FjStatus fj_test_assets_input_hull_acquire_observed(const FjAssets*, const FjInputHullCacheHooks*, FjInputHullOwner**, FjErrorBuffer*);
FjStatus fj_test_input_hull_entry_fault(const FjAssets*, uint32_t active, FjErrorBuffer*);
FjStatus fj_test_input_hull_entry_fault_consumed(const FjAssets*, uint32_t*, FjErrorBuffer*);
FjStatus fj_test_input_hull_source_tail(const FjAssets*, uint32_t active, float add, FjErrorBuffer*);
/* Observation holds retain no Assets or entry. Every release consumes once. */
FjStatus fj_test_cache_observation_acquire(const FjAssets*, FjCacheObservationOwner**, FjErrorBuffer*);
FjStatus fj_test_cache_observation_metrics(const FjCacheObservationOwner*, uint32_t family, FjCacheMetrics*, FjErrorBuffer*);
FjStatus fj_test_cache_observation_entry_live(const FjCacheObservationOwner*, uint32_t family, uintptr_t entry, uint32_t*, FjErrorBuffer*);
FjStatus fj_test_cache_observation_release(FjCacheObservationOwner*, FjErrorBuffer*);
FjStatus fj_test_output_cache_generation(const FjAssets*, uint64_t*, FjErrorBuffer*);
FjStatus fj_test_output_cache_set_generation(const FjAssets*, uint64_t, FjErrorBuffer*);
FjStatus fj_test_cache_metrics(const FjAssets*, uint32_t family, FjCacheMetrics*, FjErrorBuffer*);
/* Faults: 0 clear, 1 unsupported, 2 panic, 3 actual failed reserve, 4/5 preparation.
 * Fault selection/consumption belongs to this owner and the arming thread. */
FjStatus fj_test_cache_fault(const FjAssets*, uint32_t family, uint32_t fault, FjErrorBuffer*);
FjStatus fj_test_cache_fault_consumed(const FjAssets*, uint32_t family, uint64_t*, FjErrorBuffer*);
FjStatus fj_test_cache_clear_failures(const FjAssets*, uint32_t illuminants, uint32_t late, FjErrorBuffer*);
FjStatus fj_test_cache_drop_reentry(const FjAssets*, uint32_t family, FjErrorBuffer*);
FjStatus fj_test_cache_entry_live(const FjAssets*, uint32_t family, uintptr_t entry, uint32_t*, FjErrorBuffer*);
/* Quiescent actual-owner observations. Calibration comes from the source owner's
 * real decoded file. Curve faults are one-shot on D65/D55/D50/T/K75P derived
 * acquisition: 0 clear, 1 real unaddressable scratch reservation, 2 panic,
 * 3 decoded-row integrity failure.
 * Warm entry hits skip the fault. These controls never enter production. */
FjStatus fj_test_assets_calibration_source(const FjAssets* destination, const FjAssets* source, FjErrorBuffer*);
/* Actual Assets hull owner; existing faults 1 unsupported, 2 panic,
 * 3 locus/center, 4 ray intersections, 5 inscribed scale, 6 identity; 0 clears.
 * Warm retained hulls leave faults pending. No cross-owner/global consumption. */
FjStatus fj_test_assets_input_hull_fault(const FjAssets*, uint32_t fault, FjErrorBuffer*);
FjStatus fj_test_assets_input_hull_fault_pending(const FjAssets*, uint32_t*, FjErrorBuffer*);
/* Existing cache-owned leaf operations 1/4/5/6; faults 0 clear, 1 unsupported,
 * 2 panic, 3 actual unaddressable scratch reserve. Owner and arming thread scoped. */
FjStatus fj_test_assets_illuminant_operation_fault(const FjAssets*, uint32_t operation, uint32_t fault, FjErrorBuffer*);
FjStatus fj_test_assets_illuminant_operation_consumed(const FjAssets*, uint32_t*, FjErrorBuffer*);
FjStatus fj_test_assets_curve_fault(const FjAssets*, uint32_t source, uint32_t fault, FjErrorBuffer*);
FjStatus fj_test_assets_curve_fault_pending(const FjAssets*, uint32_t* pending, FjErrorBuffer*);
/* Last synchronous core preparation on this thread; observation only. Raw
 * record rejection does not enter/reset that constructor. No stage injects policy. */
typedef struct FjPreparationTrace {
    uint32_t count, stages[24];
} FjPreparationTrace;
FjPreparationTrace fj_test_prepared_trace(void);
/* One-shot zero at the actual completed identity check: 0 clear, 1 recipe,
 * 2 scanner resource. Unreached targets remain armed; trace reset does not clear. */
FjStatus fj_test_prepared_identity_fault(uint32_t target);
#define FJ_PREPARE_SELECTED_FILM 1U
#define FJ_PREPARE_SELECTED_PRINT 2U
#define FJ_PREPARE_ILLUMINANTS 3U
#define FJ_PREPARE_DEVELOPMENT 4U
#define FJ_PREPARE_OBSERVER 5U
#define FJ_PREPARE_RECONSTRUCTION_SOURCE 6U
#define FJ_PREPARE_INPUT_HULL 7U
#define FJ_PREPARE_CALIBRATION 8U
#define FJ_PREPARE_ROUTE_PRECHECKS 9U
#define FJ_PREPARE_FOUNDATION 10U
#define FJ_PREPARE_FILM_PAYLOAD 11U
#define FJ_PREPARE_PRINT_MAIN_ILLUMINANT 12U
#define FJ_PREPARE_FILM_REFERENCE 13U
#define FJ_PREPARE_PRINT_BALANCE 14U
#define FJ_PREPARE_SCANNER_PAYLOAD 15U
#define FJ_PREPARE_OUTPUT_BOUNDARY 16U
#define FJ_PREPARE_PRINT_PREFLASH 17U
#define FJ_PREPARE_SCANNER_CORRECTION 18U
#define FJ_PREPARE_FINAL_IDENTITY 19U

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
/* Arm checked Film::Development::new. Faults 2 and 3 reject normalized/final identity. */
FjStatus fj_test_development_construction_fault(uint32_t fault);
/* Actual Rust normalized-curve observation: 0 live, 1 successful reservations,
 * 2 drops. Sample only outside concurrent construction/destruction. */
size_t fj_test_development_curve_count(uint32_t metric);
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
/* Consume-once Runtime faults: 1/2 before/after native construction allocation
 * failures, 3 post-terminal host cleanup failure, 4/5 corresponding panics. */
void fj_test_runtime_fault(uint32_t stage);
void fj_test_runtime_cache_hold(uint32_t hold);
size_t fj_test_runtime_cache_attempts(void);
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

/* Calling-thread observation of the real Rust prepared-render CallGuard.
 * Counters reset independently of Assets cache state. The call-local source
 * result is sampled at completion before it is dropped. */
typedef struct FjPreparedNoiseObservation {
    uint32_t sequence;
    uint32_t precheck_order;
    uint32_t acquisition_order;
    uint32_t admission_order;
    uint32_t inspection_order;
    uint32_t release_order;
    uint32_t complete_order;
    uint32_t drop_order;
    uint32_t native_render_order;
    uint32_t acquisition_attempts;
    uint32_t successful_results;
    uint32_t preparation_failures;
    uint32_t allocation_failures;
    uint32_t internal_failures;
    uint32_t held_at_complete;
    uint32_t complete_calls;
    uint32_t drops_after_complete;
    uint32_t native_render_calls;
    size_t arc_references_at_complete;
} FjPreparedNoiseObservation;
void fj_test_prepared_noise_observation_reset(void);
FjStatus fj_test_prepared_noise_observation(FjPreparedNoiseObservation* out);
/* Calling-thread one-shot at the real noise decoder reservation step. */
FjStatus fj_test_noise_reservation_failure(uint32_t active);
uint32_t fj_test_noise_reservation_consumed(void);
/* Permanently poison only an isolated, quiescent scratch Assets noise cache. */
FjStatus fj_test_assets_poison_noise(const FjAssets*, FjErrorBuffer*);
/* Shorten only the call-local literal STBN view; the real bundle stays held. */
FjStatus fj_test_prepared_noise_short_view(uint32_t active);
uint32_t fj_test_prepared_noise_short_view_consumed(void);

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
/* Calls the actual render gate with a completed sigma in micrometers. Optional
 * initialized diagnostics/backing are exclusive and live until synchronous
 * return; no pointer is retained and unwind is contained. */
FjStatus fj_test_camera_lens_blur_preflight(float sigma_um, FjErrorBuffer* error);
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
/* Arm one-shot failures in the actual Rust recipe producer, not its legacy C facade. */
FjStatus fj_test_prepared_effects_probability_fault(void);
FjStatus fj_test_prepared_optics_resolved_hash_fault(void);
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
/* Same checked body and synchronous storage contract as the production grain frame edge. */
FjStatus fj_test_grain_frame_prepare(const FjGrainFrameInput*, FjGrainFrameMath*, FjGrainFrameFailure*, FjErrorBuffer*);
/* Closed one-shot faults: 0 clear, 1 entry panic, 2 reached layer panic,
 * 3 presence, 4 association, 5 Gaussian radius, 6 full-frame flag. */
FjStatus fj_test_grain_frame_fault(uint32_t);
int fj_test_grain_frame_c_check(uint32_t, const FjGrainFrameInput*, const FjGrainFrameMath*, FjStatus, FjGrainFrameFailure, const char*);
FjStatus fj_test_effects_frame_prepare(const FjEffectsFrameInput*, FjEffectsFrameMath*, FjEffectsFrameFailure*, FjErrorBuffer*);
FjStatus fj_test_effects_frame_fault(uint32_t);
int fj_test_effects_frame_c_check(uint32_t, const FjEffectsFrameInput*, const FjEffectsFrameMath*, FjStatus, FjEffectsFrameFailure, const char*);
int fj_test_effects_frame_c_raw(uint32_t, const FjEffectsFrameInput*, uint32_t, const FjEffectsFrameMath*);

/* Test-support-only immutable observations from the actual publication owner.
 * Borrowed snapshot strings/spans expire when observation is released. This
 * does not prepare/admit/recover state. Acquire/release consume one owner. */
const size_t* fj_test_instance_abi_facts(size_t* count);
size_t fj_test_prepared_state_identity(const FjPreparedState*);
typedef struct FjInstanceLifetime FjInstanceLifetime;
FjStatus fj_test_instance_lifetime_acquire(const FjInstance*, FjInstanceLifetime**, FjErrorBuffer*);
FjStatus fj_test_instance_lifetime_live(const FjInstanceLifetime*, uint32_t* out_live, FjErrorBuffer*);
FjStatus fj_test_instance_lifetime_release(FjInstanceLifetime*, FjErrorBuffer*);
typedef struct FjInstanceObservation FjInstanceObservation;
typedef struct FjInstanceObservationView {
    uint32_t pending_kind; /* 0 uninitialized, 1 capturing, 2 awaiting writes,
                            3 needs refresh, 4 valid, 5 failed, 6 blocked */
    uint32_t failure_kind;
    uint64_t issued;
    uint64_t write_revision;
    uint64_t outstanding_writes;
    uint64_t input_identity;
    uint64_t published_identity;
    uint64_t build;
    size_t backing_identity;
    FjStringView diagnostic;
    FjPreparationParams captured;
    FjPreparedStateView published;
} FjInstanceObservationView;
FjStatus fj_test_instance_submission(const FjInstance*, uint32_t* out_valid, FjSubmission* out, FjCudaContext* out_context, FjErrorBuffer*);
FjStatus fj_test_instance_observation_acquire(const FjInstance*, FjInstanceObservation**, FjErrorBuffer*);
FjStatus fj_test_instance_observation_view(const FjInstanceObservation*, FjInstanceObservationView*, FjErrorBuffer*);
FjStatus fj_test_instance_observation_release(FjInstanceObservation*, FjErrorBuffer*);
/* Real OFX owner observations. The caller retains the live effect/module and
 * excludes destruction through each synchronous call. Outputs/diagnostics are
 * initialized, aligned, exclusive and disjoint; Ready transfers one state hold.
 * No operation fabricates an FjInstance wrapper for the inline OFX owner. */
struct OfxImageEffectStruct;
typedef struct FjOfxLifetime FjOfxLifetime;
FjStatus fj_test_ofx_lifetime_acquire(struct OfxImageEffectStruct*, FjOfxLifetime**, FjErrorBuffer*);
FjStatus fj_test_ofx_lifetime_live(const FjOfxLifetime*, uint32_t*, FjErrorBuffer*);
FjStatus fj_test_ofx_lifetime_release(FjOfxLifetime*, FjErrorBuffer*);
int32_t fj_test_ofx_host_call_active(struct OfxImageEffectStruct*);
/* Independent read of the actual host controls, without ticket/publication/UI
 * mutation. Return is the actual OFX status; successful key views borrow the
 * returned input owner until consuming release. The module/effect and callbacks
 * remain live during acquisition; output/error storage is exclusive/disjoint. */
typedef struct FjOfxControlRead FjOfxControlRead;
int32_t fj_test_ofx_capture_host_input(struct OfxImageEffectStruct*, FjOfxControlRead**, FjErrorBuffer*);
FjStatus fj_test_ofx_control_input_view(const FjOfxControlRead*, FjPreparationParams*, FjErrorBuffer*);
FjStatus fj_test_ofx_control_input_release(FjOfxControlRead*, FjErrorBuffer*);
typedef struct FjOfxRetirementObservation {
    uint32_t count, identity_live;
    uint64_t token;
    FjStatus status;
} FjOfxRetirementObservation;
void fj_test_ofx_retirement_reset(void);
int32_t fj_test_ofx_retirement_observe(FjOfxRetirementObservation*);
/* The calling fixture owns this image's active standard-discovery session.
 * Invoke while Loaded retains Runtime; shutdown follows its actual native-
 * before-host cleanup protocol and retains no caller data. */
FjStatus fj_test_ofx_runtime_shutdown(void);
/* Existing render-control tag 7 relays a contained foreign abort-delivery
 * exception after the real host invocation. Tag 5 remains the terminal Rust
 * abort panic. Neither allows a C++ exception to unwind through Rust/C. */
/* Copied observations at the actual full-domain, accepted-policy/scalar and
 * effects-geometry callback stages. Raw mathematical transport counts detect
 * unintended facade round trips; the callback uses safe preparation directly. */
typedef struct FjOfxFrameObservation {
    uint32_t phases[3], count, full_count, policy_count, effects_count;
    uint32_t diffusion_transport_calls, halation_transport_calls;
    FjExtent full_frame;
    FjPreparedFramePolicy policy;
    FjEffectsGeometry effects;
    uint32_t route;
} FjOfxFrameObservation;
void fj_test_ofx_frame_reset(void);
int32_t fj_test_ofx_frame_observe(FjOfxFrameObservation*);
FjStatus fj_test_ofx_instance_admission(struct OfxImageEffectStruct*, FjAdmission*, FjErrorBuffer*);
FjStatus fj_test_ofx_instance_submission(struct OfxImageEffectStruct*, uint32_t*, FjSubmission*, FjCudaContext*, FjErrorBuffer*);
FjStatus fj_test_ofx_capture_is_current(struct OfxImageEffectStruct*, uint64_t, uint32_t*, FjErrorBuffer*);
/* Same synchronous held-render contract as fj_test_prepared_state_render;
 * the live OFX effect supplies its actual Instance and retained Runtime. */
FjCallCompletionResult fj_test_ofx_prepared_state_render(struct OfxImageEffectStruct*, const FjPreparedState*, FjCudaCallAccess, const FjFrame*, const FjPreparedFramePolicy*, uint64_t, FjAbortCallback, FjErrorBuffer*);
/* Thread-local hooks run after input selection / after complete preparation,
 * outside the state lock. Fixture callbacks contain exceptions; their context
 * stays live until cleared. Hooks are never present in a production archive. */
void fj_test_instance_prepare_hooks(void (*before_prepare)(void*), void (*after_prepare)(void*), void* context);
void fj_test_prepared_contents_reservation_failure(void);
uint32_t fj_test_prepared_contents_reservation_failure_consumed(void);

#ifdef __cplusplus
}
#endif


#endif
