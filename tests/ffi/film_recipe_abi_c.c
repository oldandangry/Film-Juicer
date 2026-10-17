#include <stddef.h>
#include "juicer_legacy_api.h"
#include "juicer_test_api.h"
_Static_assert(_Generic(((FjVisualGrainParams*)0)->active, uint32_t: 1, default: 0), "FjVisualGrainParams.active type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->sublayers_active, uint32_t: 1, default: 0), "FjVisualGrainParams.sublayers_active type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->particle_area_um2, float: 1, default: 0), "FjVisualGrainParams.particle_area_um2 type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->particle_scale_cmy, float*: 1, default: 0), "FjVisualGrainParams.particle_scale_cmy type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->particle_scale_layers, float*: 1, default: 0), "FjVisualGrainParams.particle_scale_layers type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->particle_density_min_cmy, float*: 1, default: 0), "FjVisualGrainParams.particle_density_min_cmy type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->uniformity_cmy, float*: 1, default: 0), "FjVisualGrainParams.uniformity_cmy type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->correlation_sigma_px, float: 1, default: 0), "FjVisualGrainParams.correlation_sigma_px type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->dye_cloud_blur_um, float: 1, default: 0), "FjVisualGrainParams.dye_cloud_blur_um type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->micro_structure, float*: 1, default: 0), "FjVisualGrainParams.micro_structure type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->sublayer_count, int32_t: 1, default: 0), "FjVisualGrainParams.sublayer_count type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->amplitude, float: 1, default: 0), "FjVisualGrainParams.amplitude type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->chroma_mix, float: 1, default: 0), "FjVisualGrainParams.chroma_mix type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->chroma_shared_weight, float: 1, default: 0), "FjVisualGrainParams.chroma_shared_weight type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->chroma_independent_weight, float: 1, default: 0), "FjVisualGrainParams.chroma_independent_weight type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->coarse_weight, float: 1, default: 0), "FjVisualGrainParams.coarse_weight type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->mid_weight, float: 1, default: 0), "FjVisualGrainParams.mid_weight type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->size_mix_scale, float: 1, default: 0), "FjVisualGrainParams.size_mix_scale type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->clump_temporal_mix, float: 1, default: 0), "FjVisualGrainParams.clump_temporal_mix type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->clump_morph_period_sec, float: 1, default: 0), "FjVisualGrainParams.clump_morph_period_sec type");
_Static_assert(_Generic(((FjVisualGrainParams*)0)->debug_view, int32_t: 1, default: 0), "FjVisualGrainParams.debug_view type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->active, uint32_t: 1, default: 0), "FjVisualGrainRecipe.active type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->sublayers_active, uint32_t: 1, default: 0), "FjVisualGrainRecipe.sublayers_active type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->axis_finite_cmy, uint32_t*: 1, default: 0), "FjVisualGrainRecipe.axis_finite_cmy type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->axis_block_prefix_max_cmy, float (*)[16]: 1, default: 0), "FjVisualGrainRecipe.axis_block_prefix_max_cmy type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->particle_area_um2, float: 1, default: 0), "FjVisualGrainRecipe.particle_area_um2 type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->particle_scale_cmy, float*: 1, default: 0), "FjVisualGrainRecipe.particle_scale_cmy type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->particle_scale_layers, float*: 1, default: 0), "FjVisualGrainRecipe.particle_scale_layers type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->particle_density_min_cmy, float*: 1, default: 0), "FjVisualGrainRecipe.particle_density_min_cmy type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->uniformity_cmy, float*: 1, default: 0), "FjVisualGrainRecipe.uniformity_cmy type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->correlation_sigma_px, float: 1, default: 0), "FjVisualGrainRecipe.correlation_sigma_px type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->dye_cloud_blur_um, float: 1, default: 0), "FjVisualGrainRecipe.dye_cloud_blur_um type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->micro_structure, float*: 1, default: 0), "FjVisualGrainRecipe.micro_structure type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->sublayer_count, int32_t: 1, default: 0), "FjVisualGrainRecipe.sublayer_count type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->amplitude, float: 1, default: 0), "FjVisualGrainRecipe.amplitude type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->chroma_mix, float: 1, default: 0), "FjVisualGrainRecipe.chroma_mix type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->chroma_shared_weight, float: 1, default: 0), "FjVisualGrainRecipe.chroma_shared_weight type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->chroma_independent_weight, float: 1, default: 0), "FjVisualGrainRecipe.chroma_independent_weight type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->fine_weight, float: 1, default: 0), "FjVisualGrainRecipe.fine_weight type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->mid_weight, float: 1, default: 0), "FjVisualGrainRecipe.mid_weight type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->coarse_weight, float: 1, default: 0), "FjVisualGrainRecipe.coarse_weight type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->size_mix_scale, float: 1, default: 0), "FjVisualGrainRecipe.size_mix_scale type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->clump_temporal_mix, float: 1, default: 0), "FjVisualGrainRecipe.clump_temporal_mix type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->clump_morph_period_sec, float: 1, default: 0), "FjVisualGrainRecipe.clump_morph_period_sec type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->debug_view, int32_t: 1, default: 0), "FjVisualGrainRecipe.debug_view type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->layers_hash, uint64_t: 1, default: 0), "FjVisualGrainRecipe.layers_hash type");
_Static_assert(_Generic(((FjVisualGrainRecipe*)0)->hash, uint64_t: 1, default: 0), "FjVisualGrainRecipe.hash type");
_Static_assert(_Generic(&fj_legacy_visual_grain_prepare, FjStatus (*)(const FjFilmDevelopmentOwner*, const FjVisualGrainParams*, FjVisualGrainRecipe*, FjFilmRecipeFailure*, FjErrorBuffer*): 1, default: 0), "fj_legacy_visual_grain_prepare signature");
_Static_assert(_Generic(&fj_test_visual_grain_prepare, FjStatus (*)(const FjFilmDevelopmentOwner*, const FjVisualGrainParams*, FjVisualGrainRecipe*, FjFilmRecipeFailure*, FjErrorBuffer*): 1, default: 0), "fj_test_visual_grain_prepare signature");
_Static_assert(sizeof(FjVisualGrainParams) == 120 && _Alignof(FjVisualGrainParams) == 4, "grain params x64 layout");
_Static_assert(sizeof(FjVisualGrainRecipe) == 344 && _Alignof(FjVisualGrainRecipe) == 8 && offsetof(FjVisualGrainRecipe, layers_hash) == 328 && offsetof(FjVisualGrainRecipe, hash) == 336, "grain recipe x64 layout");
_Static_assert(_Generic(((FjFilmRecipeFailure*)0)->reason, uint32_t: 1, default: 0), "FjFilmRecipeFailure.reason type");
_Static_assert(_Generic(((FjFilmRecipeFailure*)0)->channel, uint32_t: 1, default: 0), "FjFilmRecipeFailure.channel type");
_Static_assert(_Generic(((FjFilmRecipeFailure*)0)->sample, uint64_t: 1, default: 0), "FjFilmRecipeFailure.sample type");
_Static_assert(_Generic(((FjFilmDensityMinimum*)0)->cmy, float*: 1, default: 0), "FjFilmDensityMinimum.cmy type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->active, uint32_t: 1, default: 0), "FjFilmDirParams.active type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->amount, float: 1, default: 0), "FjFilmDirParams.amount type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->inhibition_same_layer, float: 1, default: 0), "FjFilmDirParams.inhibition_same_layer type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->inhibition_interlayer, float: 1, default: 0), "FjFilmDirParams.inhibition_interlayer type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->diffusion_size_um, float: 1, default: 0), "FjFilmDirParams.diffusion_size_um type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->diffusion_tail_um, float: 1, default: 0), "FjFilmDirParams.diffusion_tail_um type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->diffusion_tail_weight, float: 1, default: 0), "FjFilmDirParams.diffusion_tail_weight type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->gamma_use_stock, uint32_t: 1, default: 0), "FjFilmDirParams.gamma_use_stock type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->gamma_same_layer_rgb, float*: 1, default: 0), "FjFilmDirParams.gamma_same_layer_rgb type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->gamma_interlayer_r_to_gb, float*: 1, default: 0), "FjFilmDirParams.gamma_interlayer_r_to_gb type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->gamma_interlayer_g_to_rb, float*: 1, default: 0), "FjFilmDirParams.gamma_interlayer_g_to_rb type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->gamma_interlayer_b_to_rg, float*: 1, default: 0), "FjFilmDirParams.gamma_interlayer_b_to_rg type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->langmuir_donor_k_rgb, float*: 1, default: 0), "FjFilmDirParams.langmuir_donor_k_rgb type");
_Static_assert(_Generic(((FjFilmDirParams*)0)->langmuir_receiver_k_rgb, float*: 1, default: 0), "FjFilmDirParams.langmuir_receiver_k_rgb type");
_Static_assert(_Generic(((FjFilmDirView*)0)->polarity, uint32_t: 1, default: 0), "FjFilmDirView.polarity type");
_Static_assert(_Generic(((FjFilmDirView*)0)->mode, uint32_t: 1, default: 0), "FjFilmDirView.mode type");
_Static_assert(_Generic(((FjFilmDirView*)0)->matrix_rgb, float*: 1, default: 0), "FjFilmDirView.matrix_rgb type");
_Static_assert(_Generic(((FjFilmDirView*)0)->diffusion_size_um, float: 1, default: 0), "FjFilmDirView.diffusion_size_um type");
_Static_assert(_Generic(((FjFilmDirView*)0)->diffusion_tail_um, float: 1, default: 0), "FjFilmDirView.diffusion_tail_um type");
_Static_assert(_Generic(((FjFilmDirView*)0)->diffusion_tail_weight, float: 1, default: 0), "FjFilmDirView.diffusion_tail_weight type");
_Static_assert(_Generic(((FjFilmDirView*)0)->density_max_rgb, float*: 1, default: 0), "FjFilmDirView.density_max_rgb type");
_Static_assert(_Generic(((FjFilmDirView*)0)->density_ref_rgb, float*: 1, default: 0), "FjFilmDirView.density_ref_rgb type");
_Static_assert(_Generic(((FjFilmDirView*)0)->donor_k_rgb, float*: 1, default: 0), "FjFilmDirView.donor_k_rgb type");
_Static_assert(_Generic(((FjFilmDirView*)0)->receiver_c_ref_rgb, float*: 1, default: 0), "FjFilmDirView.receiver_c_ref_rgb type");
_Static_assert(_Generic(((FjFilmDirView*)0)->receiver_kr_rgb, float*: 1, default: 0), "FjFilmDirView.receiver_kr_rgb type");
_Static_assert(_Generic(((FjFilmDirView*)0)->compensated_axes_rgb, FjFloatSpan*: 1, default: 0), "FjFilmDirView.compensated_axes_rgb type");
_Static_assert(_Generic(((FjFilmDirView*)0)->compensated_axes_hash, uint64_t: 1, default: 0), "FjFilmDirView.compensated_axes_hash type");
_Static_assert(_Generic(((FjFilmDirView*)0)->hash, uint64_t: 1, default: 0), "FjFilmDirView.hash type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->route, uint32_t: 1, default: 0), "FjFilmDensityBounds.route type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->medium, uint32_t: 1, default: 0), "FjFilmDensityBounds.medium type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->polarity, uint32_t: 1, default: 0), "FjFilmDensityBounds.polarity type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->source, uint32_t: 1, default: 0), "FjFilmDensityBounds.source type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->minimum_cmy, float*: 1, default: 0), "FjFilmDensityBounds.minimum_cmy type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->maximum_cmy, float*: 1, default: 0), "FjFilmDensityBounds.maximum_cmy type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->inv_span_cmy, float*: 1, default: 0), "FjFilmDensityBounds.inv_span_cmy type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->authored_minimum_cmy, float*: 1, default: 0), "FjFilmDensityBounds.authored_minimum_cmy type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->authored_maximum_cmy, float*: 1, default: 0), "FjFilmDensityBounds.authored_maximum_cmy type");
_Static_assert(_Generic(((FjFilmDensityBounds*)0)->hash, uint64_t: 1, default: 0), "FjFilmDensityBounds.hash type");
_Static_assert(_Generic(&fj_legacy_film_development_retain, FjStatus (*)(const FjFilmDevelopmentOwner*, FjFilmDevelopmentOwner**, FjErrorBuffer*): 1, default: 0), "fj_legacy_film_development_retain signature");
_Static_assert(_Generic(&fj_legacy_film_density_minimum_prepare, FjStatus (*)(const FjFilmDensityMinimum*, FjFilmDensityMinimum*, FjFilmRecipeFailure*, FjErrorBuffer*): 1, default: 0), "fj_legacy_film_density_minimum_prepare signature");
_Static_assert(_Generic(&fj_legacy_film_dir_acquire, FjStatus (*)(const FjFilmDevelopmentOwner*, const FjFilmDirParams*, FjFilmDirOwner**, FjFilmRecipeFailure*, FjErrorBuffer*): 1, default: 0), "fj_legacy_film_dir_acquire signature");
_Static_assert(_Generic(&fj_legacy_film_dir_view, FjStatus (*)(const FjFilmDirOwner*, FjFilmDirView*, FjFilmRecipeFailure*, FjErrorBuffer*): 1, default: 0), "fj_legacy_film_dir_view signature");
_Static_assert(_Generic(&fj_legacy_film_dir_release, FjStatus (*)(FjFilmDirOwner*, FjErrorBuffer*): 1, default: 0), "fj_legacy_film_dir_release signature");
_Static_assert(_Generic(&fj_legacy_film_density_bounds_prepare, FjStatus (*)(const FjFilmDevelopmentOwner*, const FjFilmDensityMinimum*, FjFilmDensityBounds*, FjFilmRecipeFailure*, FjErrorBuffer*): 1, default: 0), "fj_legacy_film_density_bounds_prepare signature");
const size_t* fj_test_film_recipe_abi_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_RECIPE_TYPE(t) sizeof(t), _Alignof(t),
#define FJ_RECIPE_FIELD(t, f) offsetof(t, f),
#define FJ_RECIPE_EXTENT(t, f, e) sizeof(((t*)0)->f) / sizeof(e),
#define FJ_RECIPE_VALUE(v) v,
#include "film_recipe_abi_facts.inc"
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}
