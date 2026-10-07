#include <stddef.h>
#include "juicer_legacy_api.h"

_Static_assert(sizeof(FjSpectralObserver) == 48 && _Alignof(FjSpectralObserver) == 8, "observer layout");
_Static_assert(offsetof(FjSpectralObserver, x) == 0 && offsetof(FjSpectralObserver, y) == 16 && offsetof(FjSpectralObserver, z) == 32, "observer offsets");
_Static_assert(sizeof(FjSpectralInput) == 120 && _Alignof(FjSpectralInput) == 8, "table input layout");
_Static_assert(offsetof(FjSpectralInput, dyes_cmy) == 0 && offsetof(FjSpectralInput, observer) == 16 && offsetof(FjSpectralInput, illuminant) == 64 && offsetof(FjSpectralInput, baseline_min) == 80 && offsetof(FjSpectralInput, baseline_mid) == 96 && offsetof(FjSpectralInput, illuminant_hash) == 112, "input offsets");
_Static_assert(sizeof(FjSpectralTables) == 4264 && _Alignof(FjSpectralTables) == 8, "tables layout");
_Static_assert(offsetof(FjSpectralTables, lambda_nm) == 0 && offsetof(FjSpectralTables, illuminant) == 324 && offsetof(FjSpectralTables, observer_xyz) == 648 && offsetof(FjSpectralTables, weighted_xyz) == 1620 && offsetof(FjSpectralTables, dyes_cmy) == 2592, "plane offsets");
_Static_assert(offsetof(FjSpectralTables, baseline_min) == 3564 && offsetof(FjSpectralTables, baseline_mid) == 3888 && offsetof(FjSpectralTables, delta_lambda) == 4212 && offsetof(FjSpectralTables, inv_yn) == 4216 && offsetof(FjSpectralTables, white_xyz) == 4220 && offsetof(FjSpectralTables, reference_white_xyz) == 4232 && offsetof(FjSpectralTables, has_baseline) == 4244 && offsetof(FjSpectralTables, illuminant_hash) == 4248 && offsetof(FjSpectralTables, tables_hash) == 4256, "table scalar offsets");
_Static_assert(sizeof(FjSpectralWhiteInput) == 64 && _Alignof(FjSpectralWhiteInput) == 8 && offsetof(FjSpectralWhiteInput, illuminant) == 48, "white input layout");
_Static_assert(sizeof(FjSpectralWhite) == 32 && _Alignof(FjSpectralWhite) == 8 && offsetof(FjSpectralWhite, normalization) == 0 && offsetof(FjSpectralWhite, white_xyz) == 4 && offsetof(FjSpectralWhite, white_xy) == 16 && offsetof(FjSpectralWhite, hash) == 24, "white layout");
_Static_assert(sizeof(FjSpectralWhiteFailure) == 16 && _Alignof(FjSpectralWhiteFailure) == 8 && offsetof(FjSpectralWhiteFailure, reason) == 0 && offsetof(FjSpectralWhiteFailure, luminance_sum) == 8, "failure layout");
_Static_assert(sizeof(FjSpectralSInput) == 48 && _Alignof(FjSpectralSInput) == 8 && offsetof(FjSpectralSInput, weighted_x) == 0 && offsetof(FjSpectralSInput, weighted_y) == 16 && offsetof(FjSpectralSInput, weighted_z) == 32, "inverse input layout");
_Static_assert(sizeof(FjSpectralInverse) == 36 && _Alignof(FjSpectralInverse) == 4 && offsetof(FjSpectralInverse, matrix) == 0, "inverse layout");

int fj_test_spectral_preparation_abi_c(void) {
    FjStatus (*tables)(const FjSpectralInput*, FjSpectralTables*, FjErrorBuffer*) = fj_legacy_spectral_tables;
    FjStatus (*white)(const FjSpectralWhiteInput*, FjSpectralWhite*, FjSpectralWhiteFailure*, FjErrorBuffer*) = fj_legacy_spectral_white;
    FjStatus (*inverse)(const FjSpectralSInput*, FjSpectralInverse*, FjErrorBuffer*) = fj_legacy_spectral_s_inverse;
    return tables != NULL && white != NULL && inverse != NULL;
}
