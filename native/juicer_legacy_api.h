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

#ifdef __cplusplus
}
#endif
