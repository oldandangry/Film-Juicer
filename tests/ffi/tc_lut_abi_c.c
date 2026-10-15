#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "juicer_legacy_api.h"

#define FJ_TC_SIZE(type, size, alignment) _Static_assert(sizeof(type) == size && _Alignof(type) == alignment, #type " layout");
#define FJ_TC_FIELD(type, field, offset) _Static_assert(offsetof(type, field) == offset, #type " " #field);
#include "tc_lut_abi_facts.inc"
#undef FJ_TC_SIZE
#undef FJ_TC_FIELD

typedef FjStatus (*Build)(const FjFilmTcLutInput*, FjOwnedFilmTcLut*, FjErrorBuffer*);
typedef FjStatus (*Sample)(FjFloatSpan, const float*, float*, FjErrorBuffer*);
typedef FjStatus (*Release)(FjOwnedFilmTcLut*);
_Static_assert(_Generic(&fj_legacy_reconstruction_tc_lut, Build: 1, default: 0), "build prototype");
_Static_assert(_Generic(&fj_legacy_reconstruction_sample_tc_lut, Sample: 1, default: 0), "sample by-value span and array pointers");
_Static_assert(_Generic(&fj_legacy_reconstruction_release_tc_lut, Release: 1, default: 0), "release mutable ownership pointer");

int fj_test_tc_lut_abi_c(void) {
    float* spectra = (float*)malloc((size_t)192 * 192u * 81u * sizeof(float));
    float sensitivity[243], spd[81], xyz[3] = {1.0f, 0.0f, 0.0f}, out[3] = {-1.0f, -1.0f, -1.0f};
    FjFilmTcLutInput input;
    FjOwnedFilmTcLut owned = {{NULL, 0}, 0};
    FjErrorBuffer error = {NULL, 0, 77};
    size_t i;
    if (!spectra)
        return 0;
    for (i = 0; i < (size_t)192 * 192u * 81u; ++i)
        spectra[i] = 1.0f;
    for (i = 0; i < 243; ++i)
        sensitivity[i] = 1.0f;
    for (i = 0; i < 81; ++i)
        spd[i] = 1.0f;
    memset(&input, 0, sizeof(input));
    input.spectra.data = spectra;
    input.spectra.count = (size_t)192 * 192u * 81u;
    input.sensitivity_rgb.data = sensitivity;
    input.sensitivity_rgb.count = 243;
    input.reference_illuminant.data = spd;
    input.reference_illuminant.count = 81;
    input.projection_white_xyz[0] = input.projection_white_xyz[1] = input.projection_white_xyz[2] = 1.0f;
    input.method = 2;
    if (fj_legacy_reconstruction_tc_lut(&input, &owned, &error).category != FJ_STATUS_SUCCESS) {
        free(spectra);
        return 0;
    }
    free(spectra);
    if (!owned.samples.data || owned.samples.count != 147456 || owned.capacity < owned.samples.count) {
        fj_legacy_reconstruction_release_tc_lut(&owned);
        return 0;
    }
    for (i = 0; i < owned.samples.count; ++i) {
        const float expected = i % 4 == 3 ? 0.0f : 1.0f;
        if (owned.samples.data[i] != expected) {
            fj_legacy_reconstruction_release_tc_lut(&owned);
            return 0;
        }
    }
    if (fj_legacy_reconstruction_sample_tc_lut(owned.samples, xyz, out, &error).category != FJ_STATUS_SUCCESS || out[0] != 1.0f || out[1] != 1.0f || out[2] != 1.0f) {
        fj_legacy_reconstruction_release_tc_lut(&owned);
        return 0;
    }
    if (fj_legacy_reconstruction_release_tc_lut(&owned).category != FJ_STATUS_SUCCESS || owned.samples.data || owned.samples.count || owned.capacity)
        return 0;
    return fj_legacy_reconstruction_release_tc_lut(&owned).category == FJ_STATUS_SUCCESS;
}
