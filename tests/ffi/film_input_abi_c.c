#include <stddef.h>
#include "juicer_legacy_api.h"
typedef FjStatus (*Prepare)(const FjFilmProfile*, const FjFilmInputParams*, const FjFilmInputResources*, FjFilmInput*, FjFilmInputFailure*, FjErrorBuffer*);
_Static_assert(_Generic(&fj_legacy_film_input_prepare, Prepare: 1, default: 0), "C11 film-input signature");
typedef uint32_t FjFilmInputParams_input_space_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->input_space, FjFilmInputParams_input_space_type*: 1, default: 0), "FjFilmInputParams.input_space type");
typedef uint32_t FjFilmInputParams_decode_cctf_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->decode_cctf, FjFilmInputParams_decode_cctf_type*: 1, default: 0), "FjFilmInputParams.decode_cctf type");
typedef uint32_t FjFilmInputParams_method_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->method, FjFilmInputParams_method_type*: 1, default: 0), "FjFilmInputParams.method type");
typedef uint32_t FjFilmInputParams_auto_exposure_enabled_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->auto_exposure_enabled, FjFilmInputParams_auto_exposure_enabled_type*: 1, default: 0), "FjFilmInputParams.auto_exposure_enabled type");
typedef uint32_t FjFilmInputParams_meter_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->meter, FjFilmInputParams_meter_type*: 1, default: 0), "FjFilmInputParams.meter type");
typedef float FjFilmInputParams_manual_exposure_ev_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->manual_exposure_ev, FjFilmInputParams_manual_exposure_ev_type*: 1, default: 0), "FjFilmInputParams.manual_exposure_ev type");
typedef float FjFilmInputParams_film_format_long_edge_mm_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->film_format_long_edge_mm, FjFilmInputParams_film_format_long_edge_mm_type*: 1, default: 0), "FjFilmInputParams.film_format_long_edge_mm type");
typedef uint32_t FjFilmInputParams_filter_override_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->filter_override, FjFilmInputParams_filter_override_type*: 1, default: 0), "FjFilmInputParams.filter_override type");
typedef double FjFilmInputParams_uv_type[3];
_Static_assert(_Generic(&((FjFilmInputParams*)0)->uv, FjFilmInputParams_uv_type*: 1, default: 0), "FjFilmInputParams.uv type");
typedef double FjFilmInputParams_ir_type[3];
_Static_assert(_Generic(&((FjFilmInputParams*)0)->ir, FjFilmInputParams_ir_type*: 1, default: 0), "FjFilmInputParams.ir type");
typedef uint32_t FjFilmInputParams_apply_hanatos_window_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->apply_hanatos_window, FjFilmInputParams_apply_hanatos_window_type*: 1, default: 0), "FjFilmInputParams.apply_hanatos_window type");
typedef uint32_t FjFilmInputParams_apply_hanatos_surface_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->apply_hanatos_surface, FjFilmInputParams_apply_hanatos_surface_type*: 1, default: 0), "FjFilmInputParams.apply_hanatos_surface type");
typedef uint32_t FjFilmInputParams_input_compression_type;
_Static_assert(_Generic(&((FjFilmInputParams*)0)->input_compression, FjFilmInputParams_input_compression_type*: 1, default: 0), "FjFilmInputParams.input_compression type");
typedef FjFloatSpan FjFilmInputResources_reference_illuminant_type;
_Static_assert(_Generic(&((FjFilmInputResources*)0)->reference_illuminant, FjFilmInputResources_reference_illuminant_type*: 1, default: 0), "FjFilmInputResources.reference_illuminant type");
typedef FjFloatSpan FjFilmInputResources_reference_white_xyz_type;
_Static_assert(_Generic(&((FjFilmInputResources*)0)->reference_white_xyz, FjFilmInputResources_reference_white_xyz_type*: 1, default: 0), "FjFilmInputResources.reference_white_xyz type");
typedef FjFloatSpan FjFilmInputResources_integrated_d65_xyz_type;
_Static_assert(_Generic(&((FjFilmInputResources*)0)->integrated_d65_xyz, FjFilmInputResources_integrated_d65_xyz_type*: 1, default: 0), "FjFilmInputResources.integrated_d65_xyz type");
typedef FjFloatSpan FjFilmInputResources_reconstructed_reference_white_type;
_Static_assert(_Generic(&((FjFilmInputResources*)0)->reconstructed_reference_white, FjFilmInputResources_reconstructed_reference_white_type*: 1, default: 0), "FjFilmInputResources.reconstructed_reference_white type");
typedef uint64_t FjFilmInputResources_tc_source_hash_type;
_Static_assert(_Generic(&((FjFilmInputResources*)0)->tc_source_hash, FjFilmInputResources_tc_source_hash_type*: 1, default: 0), "FjFilmInputResources.tc_source_hash type");
typedef uint64_t FjFilmInputResources_input_hull_hash_type;
_Static_assert(_Generic(&((FjFilmInputResources*)0)->input_hull_hash, FjFilmInputResources_input_hull_hash_type*: 1, default: 0), "FjFilmInputResources.input_hull_hash type");
typedef FjInputColorConversion FjFilmInput_color_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->color, FjFilmInput_color_type*: 1, default: 0), "FjFilmInput.color type");
typedef float FjFilmInput_xyz_to_linear_srgb_type[9];
_Static_assert(_Generic(&((FjFilmInput*)0)->xyz_to_linear_srgb, FjFilmInput_xyz_to_linear_srgb_type*: 1, default: 0), "FjFilmInput.xyz_to_linear_srgb type");
typedef float FjFilmInput_nominal_white_xyz_type[3];
_Static_assert(_Generic(&((FjFilmInput*)0)->nominal_white_xyz, FjFilmInput_nominal_white_xyz_type*: 1, default: 0), "FjFilmInput.nominal_white_xyz type");
typedef float FjFilmInput_projection_white_xyz_type[3];
_Static_assert(_Generic(&((FjFilmInput*)0)->projection_white_xyz, FjFilmInput_projection_white_xyz_type*: 1, default: 0), "FjFilmInput.projection_white_xyz type");
typedef uint32_t FjFilmInput_method_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->method, FjFilmInput_method_type*: 1, default: 0), "FjFilmInput.method type");
typedef uint32_t FjFilmInput_auto_exposure_enabled_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->auto_exposure_enabled, FjFilmInput_auto_exposure_enabled_type*: 1, default: 0), "FjFilmInput.auto_exposure_enabled type");
typedef uint32_t FjFilmInput_meter_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->meter, FjFilmInput_meter_type*: 1, default: 0), "FjFilmInput.meter type");
typedef float FjFilmInput_manual_exposure_ev_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->manual_exposure_ev, FjFilmInput_manual_exposure_ev_type*: 1, default: 0), "FjFilmInput.manual_exposure_ev type");
typedef float FjFilmInput_film_format_long_edge_mm_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->film_format_long_edge_mm, FjFilmInput_film_format_long_edge_mm_type*: 1, default: 0), "FjFilmInput.film_format_long_edge_mm type");
typedef uint32_t FjFilmInput_band_pass_active_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->band_pass_active, FjFilmInput_band_pass_active_type*: 1, default: 0), "FjFilmInput.band_pass_active type");
typedef float FjFilmInput_uv_type[3];
_Static_assert(_Generic(&((FjFilmInput*)0)->uv, FjFilmInput_uv_type*: 1, default: 0), "FjFilmInput.uv type");
typedef float FjFilmInput_ir_type[3];
_Static_assert(_Generic(&((FjFilmInput*)0)->ir, FjFilmInput_ir_type*: 1, default: 0), "FjFilmInput.ir type");
typedef uint32_t FjFilmInput_apply_hanatos_window_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->apply_hanatos_window, FjFilmInput_apply_hanatos_window_type*: 1, default: 0), "FjFilmInput.apply_hanatos_window type");
typedef uint32_t FjFilmInput_apply_hanatos_surface_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->apply_hanatos_surface, FjFilmInput_apply_hanatos_surface_type*: 1, default: 0), "FjFilmInput.apply_hanatos_surface type");
typedef float FjFilmInput_hanatos_spectral_blur_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->hanatos_spectral_blur, FjFilmInput_hanatos_spectral_blur_type*: 1, default: 0), "FjFilmInput.hanatos_spectral_blur type");
typedef float FjFilmInput_hanatos_window_type[4];
_Static_assert(_Generic(&((FjFilmInput*)0)->hanatos_window, FjFilmInput_hanatos_window_type*: 1, default: 0), "FjFilmInput.hanatos_window type");
typedef float FjFilmInput_hanatos_surface_rgb_type[3][15];
_Static_assert(_Generic(&((FjFilmInput*)0)->hanatos_surface_rgb, FjFilmInput_hanatos_surface_rgb_type*: 1, default: 0), "FjFilmInput.hanatos_surface_rgb type");
typedef uint32_t FjFilmInput_projection_illuminant_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->projection_illuminant, FjFilmInput_projection_illuminant_type*: 1, default: 0), "FjFilmInput.projection_illuminant type");
typedef uint32_t FjFilmInput_input_compression_active_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->input_compression_active, FjFilmInput_input_compression_active_type*: 1, default: 0), "FjFilmInput.input_compression_active type");
typedef uint64_t FjFilmInput_input_hull_hash_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->input_hull_hash, FjFilmInput_input_hull_hash_type*: 1, default: 0), "FjFilmInput.input_hull_hash type");
typedef uint64_t FjFilmInput_tc_source_hash_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->tc_source_hash, FjFilmInput_tc_source_hash_type*: 1, default: 0), "FjFilmInput.tc_source_hash type");
typedef float FjFilmInput_sensitivity_rgb_type[81][3];
_Static_assert(_Generic(&((FjFilmInput*)0)->sensitivity_rgb, FjFilmInput_sensitivity_rgb_type*: 1, default: 0), "FjFilmInput.sensitivity_rgb type");
typedef uint64_t FjFilmInput_sensitivity_hash_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->sensitivity_hash, FjFilmInput_sensitivity_hash_type*: 1, default: 0), "FjFilmInput.sensitivity_hash type");
typedef float FjFilmInput_recipe_mallett_green_scale_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->recipe_mallett_green_scale, FjFilmInput_recipe_mallett_green_scale_type*: 1, default: 0), "FjFilmInput.recipe_mallett_green_scale type");
typedef uint64_t FjFilmInput_tc_hash_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->tc_hash, FjFilmInput_tc_hash_type*: 1, default: 0), "FjFilmInput.tc_hash type");
typedef uint64_t FjFilmInput_hash_type;
_Static_assert(_Generic(&((FjFilmInput*)0)->hash, FjFilmInput_hash_type*: 1, default: 0), "FjFilmInput.hash type");
typedef uint32_t FjFilmInputFailure_origin_type;
_Static_assert(_Generic(&((FjFilmInputFailure*)0)->origin, FjFilmInputFailure_origin_type*: 1, default: 0), "FjFilmInputFailure.origin type");
typedef uint32_t FjFilmInputFailure_reason_type;
_Static_assert(_Generic(&((FjFilmInputFailure*)0)->reason, FjFilmInputFailure_reason_type*: 1, default: 0), "FjFilmInputFailure.reason type");
typedef FjSensitivityFailure FjFilmInputFailure_sensitivity_type;
_Static_assert(_Generic(&((FjFilmInputFailure*)0)->sensitivity, FjFilmInputFailure_sensitivity_type*: 1, default: 0), "FjFilmInputFailure.sensitivity type");
const size_t* fj_test_film_input_abi_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_INPUT_TYPE(type) sizeof(type), _Alignof(type),
#define FJ_INPUT_FIELD(type, field) offsetof(type, field),
#define FJ_INPUT_EXTENT(type, field, element) sizeof(((type*)0)->field) / sizeof(element),
#define FJ_INPUT_VALUE(value) value,
#include "film_input_abi_facts.inc"
#undef FJ_INPUT_TYPE
#undef FJ_INPUT_FIELD
#undef FJ_INPUT_EXTENT
#undef FJ_INPUT_VALUE
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}
FjStatus fj_test_film_input_c_call(const FjFilmProfile* profile, const FjFilmInputParams* params, const FjFilmInputResources* resources, FjFilmInput* out, FjFilmInputFailure* failure, FjErrorBuffer* error) {
    return fj_legacy_film_input_prepare(profile, params, resources, out, failure, error);
}
