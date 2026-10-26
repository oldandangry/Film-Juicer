#include "RustGrainFrameBridge.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>

namespace JuicerGrain {
    namespace {
        std::uint32_t polarity(Spektrafilm::ProfilePolarity value) {
            switch (value) {
                case Spektrafilm::ProfilePolarity::Negative:
                    return 0;
                case Spektrafilm::ProfilePolarity::Positive:
                    return 1;
                default:
                    return 2;
            }
        }
        bool zero(float value) {
            return std::bit_cast<std::uint32_t>(value) == 0;
        }
        bool empty(const FjExtent& v) {
            return v.x == 0 && v.y == 0 && v.width == 0 && v.height == 0;
        }
        bool empty(const FjGrainGaussian& v) {
            return zero(v.sigma_px) && v.radius == 0 && v.hash == 0;
        }
        bool empty(const FjGrainFrameMath& v) {
            return v.present == 0 && v.capture_polarity == 0 && empty(v.render_extent) && empty(v.full_frame_extent) && zero(v.pixel_size_um) && v.frame0 == 0 && zero(v.frame_alpha) && v.seed_base == 0 && v.seed_base_next == 0 && v.session_seed == 0 && v.clip_token == 0 && v.pitch_px == 0 && v.breathing_period_frames == 0 && v.clump_morph_period_frames == 0 && zero(v.wang_cell_mm) && zero(v.breathing_amplitude) && zero(v.breathing_cell_um_small) && zero(v.breathing_cell_um_large) && zero(v.breathing_mix) && zero(v.breathing_drift_um_per_frame) && zero(v.debug_scale) && std::all_of(std::begin(v.density_max_cmy), std::end(v.density_max_cmy), [](const auto& row) {
                       return zero(row);
                   }) &&
                   std::all_of(std::begin(v.n_particles_cmy), std::end(v.n_particles_cmy), [](const auto& row) {
                       return zero(row);
                   }) &&
                   std::all_of(std::begin(v.od_particle_cmy), std::end(v.od_particle_cmy), [](const auto& row) {
                       return zero(row);
                   }) &&
                   std::all_of(std::begin(v.density_min_layers), std::end(v.density_min_layers), [](const auto& row) {
                       return std::all_of(std::begin(row), std::end(row), [](const auto& value) {
                           return zero(value);
                       });
                   }) &&
                   std::all_of(std::begin(v.density_max_layers), std::end(v.density_max_layers), [](const auto& row) {
                       return std::all_of(std::begin(row), std::end(row), [](const auto& value) {
                           return zero(value);
                       });
                   }) &&
                   std::all_of(std::begin(v.n_particles_layers), std::end(v.n_particles_layers), [](const auto& row) {
                       return std::all_of(std::begin(row), std::end(row), [](const auto& value) {
                           return zero(value);
                       });
                   }) &&
                   std::all_of(std::begin(v.od_particle_layers), std::end(v.od_particle_layers), [](const auto& row) {
                       return std::all_of(std::begin(row), std::end(row), [](const auto& value) {
                           return zero(value);
                       });
                   }) &&
                   std::all_of(std::begin(v.correlation), std::end(v.correlation), [](const auto& row) {
                       return empty(row);
                   }) &&
                   std::all_of(std::begin(v.dye_cloud), std::end(v.dye_cloud), [](const auto& row) {
                       return std::all_of(std::begin(row), std::end(row), [](const auto& value) {
                           return empty(value);
                       });
                   }) &&
                   std::all_of(std::begin(v.size_mix_weights), std::end(v.size_mix_weights), [](const auto& row) {
                       return zero(row);
                   }) &&
                   zero(v.size_mix_gain) && v.requires_full_frame == 0 && v.recipe_hash == 0 && v.density_layers_hash == 0;
        }
        bool gaussian_shape(const FjGrainGaussian& v) {
            return v.radius == 0 ? empty(v) : v.radius >= 1 && v.radius <= 75 && std::isfinite(v.sigma_px) && v.sigma_px > 0.0f;
        }
        template <typename Extent>
        bool same_extent(const FjExtent& a, const Extent& b) {
            return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
        }
        bool failure_shape(FjStatus status, const FjGrainFrameFailure& v) {
            if (status.category == 0)
                return v.reason == 0 && v.field == 0 && v.layer == 3 && v.channel == 3;
            if (status.category == 2)
                return v.reason == 1 && ((v.field == 0 && v.layer == 3 && v.channel == 3) || (v.field == 9 && v.layer < 3 && v.channel < 3));
            if (status.category == 8)
                return v.reason == 3 && v.field == 0 && v.layer == 3 && v.channel == 3;
            if (status.category != 3 || v.reason != 2 || v.field < 1 || v.field > 13)
                return false;
            switch (v.field) {
                case 6:
                case 7:
                case 10:
                    return v.layer == 3 && v.channel < 3;
                case 9:
                case 11:
                case 12:
                    return v.layer < 3 && v.channel < 3;
                default:
                    return v.layer == 3 && v.channel == 3;
            }
        }
    } // namespace

    bool project_input(const Spektrafilm::VisualGrainFrameDescriptorInput& input, FjGrainFrameInput& out, std::string& diagnostic) {
        diagnostic.clear();
        out = {};
        if (!input.recipe || !input.filmDevelop) {
            diagnostic = "MissingRequiredResource phase=grain_descriptor field=recipe_or_film";
            return false;
        }
        if (!input.recipe->active)
            return true;
        const auto& recipe = *input.recipe;
        const auto& film = *input.filmDevelop;
        out.active = 1;
        out.sublayers_active = recipe.sublayersActive;
        out.layer_curves_required = film.densityCurvesLayersRequired;
        out.capture_polarity = polarity(input.capturePolarity);
        out.film_polarity = polarity(film.polarity);
        out.sublayer_count = recipe.nSubLayers;
        out.pixel_size_um = input.pixelSizeUm;
        out.particle_area_um2 = recipe.particleAreaUm2;
        std::copy(recipe.particleScaleCmy.begin(), recipe.particleScaleCmy.end(), out.particle_scale_cmy);
        std::copy(recipe.particleScaleLayers.begin(), recipe.particleScaleLayers.end(), out.particle_scale_layers);
        std::copy(recipe.visualParticleDensityMinCmy.begin(), recipe.visualParticleDensityMinCmy.end(), out.particle_density_min_cmy);
        std::copy(film.authoredMaxCmy.begin(), film.authoredMaxCmy.end(), out.authored_max_cmy);
        out.correlation_sigma_px = recipe.correlationSigmaPx;
        out.dye_cloud_blur_um = recipe.dyeCloudBlurUm;
        out.size_mix_scale = recipe.sizeMixScale;
        out.fine_weight = recipe.fineWeight;
        out.mid_weight = recipe.midWeight;
        out.coarse_weight = recipe.coarseWeight;
        out.clump_morph_period_sec = recipe.clumpMorphPeriodSec;
        out.render_extent = {input.renderExtent.x, input.renderExtent.y, input.renderExtent.width, input.renderExtent.height};
        out.full_frame_extent = {input.fullFrameExtent.x, input.fullFrameExtent.y, input.fullFrameExtent.width, input.fullFrameExtent.height};
        out.frame_time = input.frameTime;
        out.frame_rate = input.frameRate;
        out.session_seed = input.sessionSeed;
        out.clip_token = input.clipToken;
        out.recipe_hash = recipe.hash;
        out.recipe_layers_hash = recipe.densityCurvesLayersHash;
        out.film_layers_hash = film.densityCurvesLayersHash;
        if (recipe.sublayersActive) {
            for (std::size_t layer = 0; layer < 3; ++layer) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const auto& curve = film.densityCurvesLayers[layer][channel];
                    out.layer_curves[layer][channel] = {curve.empty() ? nullptr : curve.data(), curve.size()};
                }
            }
        }
        return true;
    }

    bool admit_math(const Spektrafilm::VisualGrainFrameDescriptorInput& input, const FjGrainFrameMath& v) {
        if (!input.recipe || !input.filmDevelop)
            return false;
        if (!input.recipe->active)
            return empty(v);
        if (v.present != 1 || v.capture_polarity > 1 || v.capture_polarity != polarity(input.capturePolarity) || v.capture_polarity != polarity(input.filmDevelop->polarity) || (input.recipe->sublayersActive && (!input.filmDevelop->densityCurvesLayersRequired || input.filmDevelop->densityCurvesLayersHash == 0 || input.recipe->densityCurvesLayersHash != input.filmDevelop->densityCurvesLayersHash)) || v.recipe_hash != input.recipe->hash || v.recipe_hash == 0 || v.density_layers_hash != (input.recipe->sublayersActive ? input.filmDevelop->densityCurvesLayersHash : 0) || !same_extent(v.render_extent, input.renderExtent) || !same_extent(v.full_frame_extent, input.fullFrameExtent) || std::bit_cast<std::uint32_t>(v.pixel_size_um) != std::bit_cast<std::uint32_t>(input.pixelSizeUm) || v.session_seed != input.sessionSeed || v.clip_token != input.clipToken || v.pitch_px != input.fullFrameExtent.height || v.requires_full_frame > 1)
            return false;
        bool fullFrame = false;
        for (const auto& g : v.correlation) {
            if (!gaussian_shape(g))
                return false;
            fullFrame = fullFrame || g.radius > 0;
        }
        for (const auto& row : v.dye_cloud) {
            for (const auto& g : row) {
                if (!gaussian_shape(g))
                    return false;
                fullFrame = fullFrame || g.radius > 0;
            }
        }
        return v.requires_full_frame == static_cast<unsigned>(fullFrame) && (!fullFrame || same_extent(v.render_extent, v.full_frame_extent));
    }

    bool accept_result(const Spektrafilm::VisualGrainFrameDescriptorInput& input, FjStatus status, const FjGrainFrameMath& math, const FjGrainFrameFailure& failure, const FjErrorBuffer& error, const std::array<char, 256>& text, Spektrafilm::VisualGrainFrameDescriptor& out, std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (status.api != FJ_API_NONE || status.native_code != 0 || !failure_shape(status, failure) || error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0' || (status.category != 0 && !empty(math))) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result";
            return false;
        }
        if (status.category != 0) {
            diagnostic.assign(text.data(), error.length);
            return false;
        }
        if (error.length != 0) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result";
            return false;
        }
        return Spektrafilm::compose_visual_grain_frame_descriptor(input, math, out, diagnostic);
    }

    bool prepare_frame(const Spektrafilm::VisualGrainFrameDescriptorInput& input, Spektrafilm::VisualGrainFrameDescriptor& out, std::string& diagnostic) {
        out = {};
        FjGrainFrameInput raw{};
        if (!project_input(input, raw, diagnostic))
            return false;
        FjGrainFrameMath math{};
        FjGrainFrameFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_grain_frame_prepare(&raw, &math, &failure, &error);
        return accept_result(input, status, math, failure, error, text, out, diagnostic);
    }
} // namespace JuicerGrain
