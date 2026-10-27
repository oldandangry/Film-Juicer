#include "FilmEffectsFrameDescriptors.h"

#include <algorithm>

#include "Hash.h"
#include "RustGrainFrameBridge.h"
#include "RustEffectsFrameBridge.h"


namespace {

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    Spektrafilm::VisualGrainScratchShape select_scratch_shape(
        bool sublayers,
        bool sharedChroma) {
        if (sublayers) {
            return sharedChroma
                       ? Spektrafilm::VisualGrainScratchShape::StreamedLayersShared
                       : Spektrafilm::VisualGrainScratchShape::StreamedLayers;
        }
        return sharedChroma
                   ? Spektrafilm::VisualGrainScratchShape::StreamedShared
                   : Spektrafilm::VisualGrainScratchShape::Streamed;
    }

    template <typename Extent>
    void hash_extent(std::uint64_t& hash, const Extent& extent) {
        hash_value(hash, extent.x);
        hash_value(hash, extent.y);
        hash_value(hash, extent.width);
        hash_value(hash, extent.height);
    }

    std::uint64_t hash_descriptor(
        const Spektrafilm::VisualGrainFrameDescriptor& descriptor) {
        if (!descriptor.active) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, descriptor.active);
        hash_value(hash, descriptor.capturePolarity);
        hash_extent(hash, descriptor.renderExtent);
        hash_extent(hash, descriptor.fullFrameExtent);
        hash_value(hash, descriptor.pixelSizeUm);
        hash_value(hash, descriptor.frame0);
        hash_value(hash, descriptor.frameAlpha);
        hash_value(hash, descriptor.seedBase);
        hash_value(hash, descriptor.seedBaseNext);
        hash_value(hash, descriptor.sessionSeed);
        hash_value(hash, descriptor.clipToken);
        hash_value(hash, descriptor.pitchPx);
        hash_value(hash, descriptor.breathingPeriodFrames);
        hash_value(hash, descriptor.clumpMorphPeriodFrames);
        hash_value(hash, descriptor.wangCellMm);
        hash_value(hash, descriptor.breathingAmplitude);
        hash_value(hash, descriptor.breathingCellUmSmall);
        hash_value(hash, descriptor.breathingCellUmLarge);
        hash_value(hash, descriptor.breathingMix);
        hash_value(hash, descriptor.breathingDriftUmPerFrame);
        hash_value(hash, descriptor.debugScale);
        Hash::hash_bytes_update(
            hash,
            descriptor.densityMaxCmy.data(),
            sizeof(descriptor.densityMaxCmy));
        Hash::hash_bytes_update(
            hash,
            descriptor.nParticlesCmy.data(),
            sizeof(descriptor.nParticlesCmy));
        Hash::hash_bytes_update(
            hash,
            descriptor.odParticleCmy.data(),
            sizeof(descriptor.odParticleCmy));
        Hash::hash_bytes_update(
            hash,
            descriptor.densityMinLayers.data(),
            sizeof(descriptor.densityMinLayers));
        Hash::hash_bytes_update(
            hash,
            descriptor.densityMaxLayers.data(),
            sizeof(descriptor.densityMaxLayers));
        Hash::hash_bytes_update(
            hash,
            descriptor.nParticlesLayers.data(),
            sizeof(descriptor.nParticlesLayers));
        Hash::hash_bytes_update(
            hash,
            descriptor.odParticleLayers.data(),
            sizeof(descriptor.odParticleLayers));
        for (const auto& gaussian : descriptor.correlation) {
            hash_value(hash, gaussian.hash);
        }
        for (const auto& layer : descriptor.dyeCloud) {
            for (const auto& gaussian : layer) {
                hash_value(hash, gaussian.hash);
            }
        }
        hash_value(hash, descriptor.effectiveFineWeight);
        hash_value(hash, descriptor.effectiveMidWeight);
        hash_value(hash, descriptor.effectiveCoarseWeight);
        hash_value(hash, descriptor.sizeMixGain);
        hash_value(hash, descriptor.scratchShape);
        hash_value(hash, descriptor.requiresFullFrame);
        hash_value(hash, descriptor.recipeHash);
        hash_value(hash, descriptor.densityCurvesLayersHash);
        return hash;
    }

} // namespace


namespace Spektrafilm {

    VisualGrainScratchShape visual_grain_scratch_shape(const VisualGrainRecipe& recipe) noexcept {
        if (!recipe.active) {
            return VisualGrainScratchShape::None;
        }
        const bool sharedChroma =
            (recipe.debugView == 0 || recipe.debugView == 1) &&
            recipe.chromaMix < 0.999f &&
            recipe.chromaSharedWeight > 0.0f;
        return select_scratch_shape(recipe.sublayersActive, sharedChroma);
    }

    bool compose_visual_grain_frame_descriptor(
        const VisualGrainFrameDescriptorInput& input, const FjGrainFrameMath& math, VisualGrainFrameDescriptor& out, std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (!JuicerGrain::admit_math(input, math)) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result";
            return false;
        }
        if (!math.present)
            return true;
        VisualGrainFrameDescriptor descriptor;
        descriptor.active = true;
        descriptor.capturePolarity = input.capturePolarity;
        descriptor.renderExtent = {math.render_extent.x, math.render_extent.y, math.render_extent.width, math.render_extent.height};
        descriptor.fullFrameExtent = {math.full_frame_extent.x, math.full_frame_extent.y, math.full_frame_extent.width, math.full_frame_extent.height};
        descriptor.pixelSizeUm = math.pixel_size_um;
        descriptor.frame0 = math.frame0;
        descriptor.frameAlpha = math.frame_alpha;
        descriptor.seedBase = math.seed_base;
        descriptor.seedBaseNext = math.seed_base_next;
        descriptor.sessionSeed = math.session_seed;
        descriptor.clipToken = math.clip_token;
        descriptor.pitchPx = math.pitch_px;
        descriptor.breathingPeriodFrames = math.breathing_period_frames;
        descriptor.clumpMorphPeriodFrames = math.clump_morph_period_frames;
        descriptor.wangCellMm = math.wang_cell_mm;
        descriptor.breathingAmplitude = math.breathing_amplitude;
        descriptor.breathingCellUmSmall = math.breathing_cell_um_small;
        descriptor.breathingCellUmLarge = math.breathing_cell_um_large;
        descriptor.breathingMix = math.breathing_mix;
        descriptor.breathingDriftUmPerFrame = math.breathing_drift_um_per_frame;
        descriptor.debugScale = math.debug_scale;
        std::copy_n(math.density_max_cmy, 3, descriptor.densityMaxCmy.begin());
        std::copy_n(math.n_particles_cmy, 3, descriptor.nParticlesCmy.begin());
        std::copy_n(math.od_particle_cmy, 3, descriptor.odParticleCmy.begin());
        for (std::size_t layer = 0; layer < 3; ++layer)
            std::copy_n(math.density_min_layers[layer], 3, descriptor.densityMinLayers[layer].begin());
        for (std::size_t layer = 0; layer < 3; ++layer)
            std::copy_n(math.density_max_layers[layer], 3, descriptor.densityMaxLayers[layer].begin());
        for (std::size_t layer = 0; layer < 3; ++layer)
            std::copy_n(math.n_particles_layers[layer], 3, descriptor.nParticlesLayers[layer].begin());
        for (std::size_t layer = 0; layer < 3; ++layer)
            std::copy_n(math.od_particle_layers[layer], 3, descriptor.odParticleLayers[layer].begin());
        descriptor.sizeMixGain = math.size_mix_gain;
        descriptor.requiresFullFrame = math.requires_full_frame != 0;
        descriptor.recipeHash = math.recipe_hash;
        descriptor.densityCurvesLayersHash = math.density_layers_hash;
        const auto gaussian = [](const FjGrainGaussian& v) -> VisualGrainGaussian {
            return {v.sigma_px, v.radius, v.hash};
        };
        for (std::size_t index = 0; index < 3; ++index) {
            descriptor.correlation[index] = gaussian(math.correlation[index]);
            for (std::size_t channel = 0; channel < 3; ++channel)
                descriptor.dyeCloud[index][channel] = gaussian(math.dye_cloud[index][channel]);
        }
        descriptor.effectiveFineWeight = math.size_mix_weights[0];
        descriptor.effectiveMidWeight = math.size_mix_weights[1];
        descriptor.effectiveCoarseWeight = math.size_mix_weights[2];
        descriptor.scratchShape = visual_grain_scratch_shape(*input.recipe);
        descriptor.hash = hash_descriptor(descriptor);
        if (descriptor.hash == 0) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=hash";
            return false;
        }
        out = descriptor;
        return true;
    }

    bool build_visual_grain_frame_descriptor(
        const VisualGrainFrameDescriptorInput& input,
        VisualGrainFrameDescriptor& out,
        std::string& diagnostic) {
        return JuicerGrain::prepare_frame(input, out, diagnostic);
    }

} // namespace Spektrafilm

namespace Spektrafilm {
    bool build_film_juicer_effects_frame_descriptor(
        const FilmJuicerEffectsFrameDescriptorInput& input,
        FilmJuicerEffectsFrameDescriptor& out,
        std::string& diagnostic) {
        return JuicerEffects::prepare_frame(input, out, diagnostic);
    }
} // namespace Spektrafilm
