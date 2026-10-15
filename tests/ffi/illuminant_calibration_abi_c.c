#include <stddef.h>
#include "juicer_legacy_api.h"
_Static_assert(sizeof(FjNeutralCalibrationResult) == 20 && _Alignof(FjNeutralCalibrationResult) == 4, "calibration layout");
_Static_assert(offsetof(FjNeutralCalibrationResult, outcome) == 0 && offsetof(FjNeutralCalibrationResult, field) == 4 && offsetof(FjNeutralCalibrationResult, cmy_cc) == 8, "calibration offsets");
_Static_assert(sizeof(FjFloatSpan) == 16 && _Alignof(FjFloatSpan) == 8, "CSV span layout");
#define FJ_SIGNATURE(function, type) _Static_assert(_Generic(&(function), type: 1, default: 0), #function)
FJ_SIGNATURE(fj_legacy_csv_acquire, FjStatus (*)(const FjAssets*, uint32_t, FjCsvPairs**, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_csv_view, FjStatus (*)(const FjCsvPairs*, FjFloatSpan*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_csv_release, FjStatus (*)(FjCsvPairs*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_neutral_calibration_lookup, FjStatus (*)(const FjAssets*, FjStringView, FjStringView, FjStringView, FjNeutralCalibrationResult*, FjErrorBuffer*));
_Static_assert(FJ_CSV_D65 == 1 && FJ_CSV_D55 == 2 && FJ_CSV_D50 == 3 && FJ_CSV_T == 4 && FJ_CSV_K75P == 5 && FJ_CSV_KG3 == 6 && FJ_CSV_CANON_24_F28_IS == 7, "CSV tags");
_Static_assert(FJ_CALIBRATION_FOUND == 1 && FJ_CALIBRATION_MISSING_FILE == 2 && FJ_CALIBRATION_MISSING_ENTRY == 3 && FJ_CALIBRATION_MALFORMED == 4, "outcomes");
_Static_assert(FJ_CALIBRATION_FIELD_NONE == 0 && FJ_CALIBRATION_FIELD_RESOURCE_READ == 1 && FJ_CALIBRATION_FIELD_ROOT == 2 && FJ_CALIBRATION_FIELD_PRINT_PROFILE == 3 && FJ_CALIBRATION_FIELD_PRINT_ILLUMINANT == 4 && FJ_CALIBRATION_FIELD_CMY_CC == 5, "fields");

_Static_assert(sizeof(FjIlluminant) == 324 && _Alignof(FjIlluminant) == 4, "illuminant layout");
_Static_assert(sizeof(FjIlluminantCoverage) == 12 && _Alignof(FjIlluminantCoverage) == 4, "coverage layout");
_Static_assert(offsetof(FjIlluminantCoverage, warnings) == 0 && offsetof(FjIlluminantCoverage, min_nm) == 4 && offsetof(FjIlluminantCoverage, max_nm) == 8, "coverage offsets");
FJ_SIGNATURE(fj_legacy_illuminant_from_samples, FjStatus (*)(FjFloatSpan, FjIlluminant*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_illuminant_blackbody, FjStatus (*)(float, FjIlluminant*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_illuminant_equal_energy, FjStatus (*)(FjIlluminant*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_illuminant_tungsten_kg3, FjStatus (*)(FjFloatSpan, FjIlluminant*, FjIlluminantCoverage*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_illuminant_lens_prepare, FjStatus (*)(FjFloatSpan, FjIlluminantLens**, FjIlluminantCoverage*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_illuminant_lens_finish, FjStatus (*)(FjIlluminantLens**, FjFloatSpan, FjIlluminant*, FjIlluminantCoverage*, FjErrorBuffer*));
FJ_SIGNATURE(fj_legacy_illuminant_lens_release, FjStatus (*)(FjIlluminantLens**, FjErrorBuffer*));
