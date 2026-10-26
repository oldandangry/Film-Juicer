#include "FilmEffectsFrameDescriptors.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Hash.h"
#include "RustGrainFrameBridge.h"


namespace {

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    template <typename Extent>
    bool valid_extent(const Extent& extent) {
        return extent.width > 0 && extent.height > 0;
    }

    template <typename Extent>
    bool same_extent(
        const Extent& a,
        const Extent& b) {
        return a.x == b.x && a.y == b.y &&
               a.width == b.width && a.height == b.height;
    }

    template <typename Extent>
    bool extent_contains(
        const Extent& outer,
        const Extent& inner) {
        const std::int64_t outerRight =
            static_cast<std::int64_t>(outer.x) + outer.width;
        const std::int64_t outerBottom =
            static_cast<std::int64_t>(outer.y) + outer.height;
        const std::int64_t innerRight =
            static_cast<std::int64_t>(inner.x) + inner.width;
        const std::int64_t innerBottom =
            static_cast<std::int64_t>(inner.y) + inner.height;
        return inner.x >= outer.x && inner.y >= outer.y &&
               innerRight <= outerRight && innerBottom <= outerBottom;
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

namespace {

    constexpr std::uint64_t kSeedPassWeave = 3;

    double hash_to_unit(std::uint64_t hash) {
        constexpr double kInverseMantissaRange =
            1.0 / 9007199254740992.0;
        return static_cast<double>(hash >> 11) * kInverseMantissaRange;
    }

    double phase_from_seed(
        std::uint64_t sessionSeed,
        std::uint64_t passId,
        int axis,
        int component) {
        const std::uint64_t fields[4] = {
            sessionSeed,
            passId,
            static_cast<std::uint64_t>(axis),
            static_cast<std::uint64_t>(component)};
        std::uint64_t hash = Hash::hash_bytes(fields, sizeof(fields));
        if (hash == 0) {
            hash = 1;
        }
        constexpr double kTwoPi = 6.28318530717958647692;
        return hash_to_unit(hash) * kTwoPi;
    }

    // Argument order mirrors the reviewed Archive signal formula.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    double sin_sum(
        const double* frequencies,
        int count,
        std::uint64_t sessionSeed,
        std::uint64_t passId,
        int axis,
        int componentOffset,
        double timeSeconds) {
        constexpr double kTwoPi = 6.28318530717958647692;
        double sum = 0.0;
        for (int index = 0; index < count; ++index) {
            const double phase = phase_from_seed(
                sessionSeed,
                passId,
                axis,
                componentOffset + index);
            sum += std::sin(
                kTwoPi * frequencies[index] * timeSeconds + phase);
        }
        return sum;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    struct GateWeaveSignal final {
        float dxPx = 0.0f;
        float dyPx = 0.0f;
        float cosRot = 1.0f;
        float sinRot = 0.0f;
    };

    // Units are named and passed once from the descriptor builder.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    GateWeaveSignal compute_gate_weave(
        std::uint64_t sessionSeed,
        double timeSeconds,
        double translateRmsUm,
        double rotateRmsDeg,
        double pixelSizeUm,
        double amount) {
        GateWeaveSignal out{};
        if (!(amount > 0.0) || !std::isfinite(pixelSizeUm) ||
            !(pixelSizeUm > 0.0)) {
            return out;
        }

        const double translateRms = translateRmsUm * amount;
        const double rotateRms = rotateRmsDeg * amount;
        if (!(translateRms > 0.0 || rotateRms > 0.0)) {
            return out;
        }

        constexpr double kDriftFrequencies[] = {0.15, 0.35, 0.80};
        constexpr double kJitterFrequencies[] = {6.0, 12.0};
        constexpr int kDriftCount =
            static_cast<int>(std::size(kDriftFrequencies));
        constexpr int kJitterCount =
            static_cast<int>(std::size(kJitterFrequencies));
        const double driftNorm =
            1.0 / std::sqrt(0.5 * static_cast<double>(kDriftCount));
        const double jitterNorm =
            1.0 / std::sqrt(0.5 * static_cast<double>(kJitterCount));
        constexpr double kDriftWeight = 0.85;
        constexpr double kJitterWeight = 0.15;
        const double weightNorm = 1.0 / std::sqrt(
                                            kDriftWeight * kDriftWeight +
                                            kJitterWeight * kJitterWeight);

        for (int axis = 0; axis < 2; ++axis) {
            const double drift = sin_sum(
                                     kDriftFrequencies,
                                     kDriftCount,
                                     sessionSeed,
                                     kSeedPassWeave,
                                     axis,
                                     0,
                                     timeSeconds) *
                                 driftNorm;
            const double jitter = sin_sum(
                                      kJitterFrequencies,
                                      kJitterCount,
                                      sessionSeed,
                                      kSeedPassWeave,
                                      axis,
                                      10,
                                      timeSeconds) *
                                  jitterNorm;
            const double composite =
                (kDriftWeight * drift + kJitterWeight * jitter) *
                weightNorm;
            const float deltaPx = static_cast<float>(
                composite * translateRms / pixelSizeUm);
            if (axis == 0) {
                out.dxPx = deltaPx;
            } else {
                out.dyPx = deltaPx;
            }
        }

        if (rotateRms > 0.0) {
            const double rotationSignal = sin_sum(
                                              kDriftFrequencies,
                                              kDriftCount,
                                              sessionSeed,
                                              kSeedPassWeave,
                                              2,
                                              0,
                                              timeSeconds) *
                                          driftNorm;
            const double rotationDegrees = rotationSignal * rotateRms;
            const double rotationRadians =
                rotationDegrees * (3.14159265358979323846 / 180.0);
            out.cosRot = static_cast<float>(std::cos(rotationRadians));
            out.sinRot = static_cast<float>(std::sin(rotationRadians));
        }
        return out;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

} // namespace

namespace Spektrafilm {

    std::uint64_t hash_film_juicer_effects_descriptor(const FilmJuicerEffectsFrameDescriptor& d) {
        if (!d.filmActive && !d.gateOutputActive) {
            return 0;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_extent(hash, d.renderExtent);
        hash_extent(hash, d.fullFrameExtent);
        hash_value(hash, d.sampleStepXMm);
        hash_value(hash, d.sampleStepYMm);
        hash_value(hash, d.roiOffsetX);
        hash_value(hash, d.roiOffsetY);
        hash_value(hash, d.gateWidth);
        hash_value(hash, d.gateHeight);
        hash_value(hash, d.sessionSeed);
        hash_value(hash, d.clipToken);
        hash_value(hash, d.weaveActive);
        hash_value(hash, d.weaveDxPx);
        hash_value(hash, d.weaveDyPx);
        hash_value(hash, d.weaveCosRot);
        hash_value(hash, d.weaveSinRot);
        hash_value(hash, d.filmActive);
        hash_value(hash, d.gateTransmittanceActive);
        hash_value(hash, d.gateOutputActive);
        hash_value(hash, d.requiresFullFrame);
        hash_value(hash, d.recipeHash);
        if (d.filmDust.slotProbability > 0.0f) {
            hash_value(hash, d.filmDust.cellWidthMm);
            hash_value(hash, d.filmDust.cellHeightMm);
            hash_value(hash, d.filmDust.slotProbability);
            hash_value(hash, d.filmDust.softnessMinMm);
            hash_value(hash, d.filmDust.softnessMaxMm);
            hash_value(hash, d.filmDust.softnessSizeCapFraction);
            hash_value(hash, d.filmDust.supportXMm);
            hash_value(hash, d.filmDust.supportYMm);
            hash_value(hash, d.filmDust.fiberFraction);
            hash_value(hash, d.filmDust.fiberDriftFraction);
            hash_value(hash, d.filmDust.fiberFirstKnotMin);
            hash_value(hash, d.filmDust.fiberFirstKnotMax);
            hash_value(hash, d.filmDust.fiberSecondKnotMin);
            hash_value(hash, d.filmDust.fiberSecondKnotMax);
            hash_value(hash, d.filmDust.fiberInteriorWidthMinFraction);
            hash_value(hash, d.filmDust.fiberInteriorWidthMaxFraction);
            hash_value(hash, d.filmDust.diameterMinMm);
            hash_value(hash, d.filmDust.diameterBulkMaxMm);
            hash_value(hash, d.filmDust.diameterMaxMm);
            hash_value(hash, d.filmDust.diameterTailFraction);
            hash_value(hash, d.filmDust.fiberLengthMinMm);
            hash_value(hash, d.filmDust.fiberLengthMaxMm);
            hash_value(hash, d.filmDust.fiberWidthMinMm);
            hash_value(hash, d.filmDust.fiberWidthMaxMm);
            hash_value(hash, d.filmDust.opacityFaintCumulative);
            hash_value(hash, d.filmDust.opacityIntermediateCumulative);
            hash_value(hash, d.filmDust.compactOpacityMin);
            hash_value(hash, d.filmDust.compactOpacityFaintEnd);
            hash_value(hash, d.filmDust.compactOpacityIntermediateEnd);
            hash_value(hash, d.filmDust.compactOpacityMax);
            hash_value(hash, d.filmDust.fiberOpacityMin);
            hash_value(hash, d.filmDust.fiberOpacityFaintEnd);
            hash_value(hash, d.filmDust.fiberOpacityIntermediateEnd);
            hash_value(hash, d.filmDust.fiberOpacityMax);
            hash_value(hash, d.filmDust.compactDominantAspectMin);
            hash_value(hash, d.filmDust.compactDominantAspectMax);
            hash_value(hash, d.filmDust.compactSubsidiaryScaleMin);
            hash_value(hash, d.filmDust.compactSubsidiaryScaleMax);
            hash_value(hash, d.filmDust.compactSubsidiaryAspectMin);
            hash_value(hash, d.filmDust.compactSubsidiaryAspectMax);
            hash_value(hash, d.filmDust.compactSubsidiaryOffsetMax);
            hash_value(hash, d.filmDust.compactSubsidiaryAngleMaxRadians);
            hash_value(hash, d.origins[0].cellX);
            hash_value(hash, d.origins[0].cellY);
            hash_value(hash, d.origins[0].localXMm);
            hash_value(hash, d.origins[0].localYMm);
        }
        if (d.filmScratch.slotProbability > 0.0f) {
            hash_value(hash, d.filmScratch.cellWidthMm);
            hash_value(hash, d.filmScratch.cellHeightMm);
            hash_value(hash, d.filmScratch.slotProbability);
            hash_value(hash, d.filmScratch.softnessMinMm);
            hash_value(hash, d.filmScratch.softnessMaxMm);
            hash_value(hash, d.filmScratch.softnessSizeCapFraction);
            hash_value(hash, d.filmScratch.supportXMm);
            hash_value(hash, d.filmScratch.supportYMm);
            hash_value(hash, d.filmScratch.lengthMinMm);
            hash_value(hash, d.filmScratch.lengthBulkMaxMm);
            hash_value(hash, d.filmScratch.lengthMaxMm);
            hash_value(hash, d.filmScratch.lengthTailFraction);
            hash_value(hash, d.filmScratch.widthMinMm);
            hash_value(hash, d.filmScratch.widthBulkMaxMm);
            hash_value(hash, d.filmScratch.widthMaxMm);
            hash_value(hash, d.filmScratch.widthTailFraction);
            hash_value(hash, d.filmScratch.driftFraction);
            hash_value(hash, d.filmScratch.firstKnotMin);
            hash_value(hash, d.filmScratch.firstKnotMax);
            hash_value(hash, d.filmScratch.secondKnotMin);
            hash_value(hash, d.filmScratch.secondKnotMax);
            hash_value(hash, d.filmScratch.interiorWidthMinFraction);
            hash_value(hash, d.filmScratch.interiorWidthMaxFraction);
            hash_value(hash, d.filmScratch.interiorDepthMinFraction);
            hash_value(hash, d.filmScratch.interiorDepthMaxFraction);
            hash_value(hash, d.filmScratch.endpointAbruptProbability);
            hash_value(hash, d.filmScratch.interruptionProbability);
            hash_value(hash, d.filmScratch.gapCenterMin);
            hash_value(hash, d.filmScratch.gapCenterMax);
            hash_value(hash, d.filmScratch.gapSpanMin);
            hash_value(hash, d.filmScratch.gapSpanMax);
            hash_value(hash, d.filmScratch.scuffProbability);
            hash_value(hash, d.filmScratch.scuffLengthMaxMm);
            hash_value(hash, d.filmScratch.scuffAngleMaxRadians);
            hash_value(hash, d.filmScratch.strengthMin);
            hash_value(hash, d.filmScratch.strengthMax);
            hash_value(hash, d.origins[1].cellX);
            hash_value(hash, d.origins[1].cellY);
            hash_value(hash, d.origins[1].localXMm);
            hash_value(hash, d.origins[1].localYMm);
        }
        if (d.gateDust.slotProbability > 0.0f) {
            hash_value(hash, d.gateDust.cellWidthMm);
            hash_value(hash, d.gateDust.cellHeightMm);
            hash_value(hash, d.gateDust.slotProbability);
            hash_value(hash, d.gateDust.softnessMinMm);
            hash_value(hash, d.gateDust.softnessMaxMm);
            hash_value(hash, d.gateDust.softnessSizeCapFraction);
            hash_value(hash, d.gateDust.supportXMm);
            hash_value(hash, d.gateDust.supportYMm);
            hash_value(hash, d.gateDust.fiberFraction);
            hash_value(hash, d.gateDust.fiberDriftFraction);
            hash_value(hash, d.gateDust.fiberFirstKnotMin);
            hash_value(hash, d.gateDust.fiberFirstKnotMax);
            hash_value(hash, d.gateDust.fiberSecondKnotMin);
            hash_value(hash, d.gateDust.fiberSecondKnotMax);
            hash_value(hash, d.gateDust.fiberInteriorWidthMinFraction);
            hash_value(hash, d.gateDust.fiberInteriorWidthMaxFraction);
            hash_value(hash, d.gateDust.diameterMinMm);
            hash_value(hash, d.gateDust.diameterBulkMaxMm);
            hash_value(hash, d.gateDust.diameterMaxMm);
            hash_value(hash, d.gateDust.diameterTailFraction);
            hash_value(hash, d.gateDust.fiberLengthMinMm);
            hash_value(hash, d.gateDust.fiberLengthMaxMm);
            hash_value(hash, d.gateDust.fiberWidthMinMm);
            hash_value(hash, d.gateDust.fiberWidthMaxMm);
            hash_value(hash, d.gateDust.opacityFaintCumulative);
            hash_value(hash, d.gateDust.opacityIntermediateCumulative);
            hash_value(hash, d.gateDust.compactOpacityMin);
            hash_value(hash, d.gateDust.compactOpacityFaintEnd);
            hash_value(hash, d.gateDust.compactOpacityIntermediateEnd);
            hash_value(hash, d.gateDust.compactOpacityMax);
            hash_value(hash, d.gateDust.fiberOpacityMin);
            hash_value(hash, d.gateDust.fiberOpacityFaintEnd);
            hash_value(hash, d.gateDust.fiberOpacityIntermediateEnd);
            hash_value(hash, d.gateDust.fiberOpacityMax);
            hash_value(hash, d.gateDust.compactDominantAspectMin);
            hash_value(hash, d.gateDust.compactDominantAspectMax);
            hash_value(hash, d.gateDust.compactSubsidiaryScaleMin);
            hash_value(hash, d.gateDust.compactSubsidiaryScaleMax);
            hash_value(hash, d.gateDust.compactSubsidiaryAspectMin);
            hash_value(hash, d.gateDust.compactSubsidiaryAspectMax);
            hash_value(hash, d.gateDust.compactSubsidiaryOffsetMax);
            hash_value(hash, d.gateDust.compactSubsidiaryAngleMaxRadians);
            hash_value(hash, d.origins[2].cellX);
            hash_value(hash, d.origins[2].cellY);
            hash_value(hash, d.origins[2].localXMm);
            hash_value(hash, d.origins[2].localYMm);
        }
        if (d.gateScratch.slotProbability > 0.0f) {
            hash_value(hash, d.gateScratch.cellWidthMm);
            hash_value(hash, d.gateScratch.cellHeightMm);
            hash_value(hash, d.gateScratch.slotProbability);
            hash_value(hash, d.gateScratch.softnessMinMm);
            hash_value(hash, d.gateScratch.softnessMaxMm);
            hash_value(hash, d.gateScratch.softnessSizeCapFraction);
            hash_value(hash, d.gateScratch.supportXMm);
            hash_value(hash, d.gateScratch.supportYMm);
            hash_value(hash, d.gateScratch.lengthMinMm);
            hash_value(hash, d.gateScratch.lengthBulkMaxMm);
            hash_value(hash, d.gateScratch.lengthMaxMm);
            hash_value(hash, d.gateScratch.lengthTailFraction);
            hash_value(hash, d.gateScratch.widthMinMm);
            hash_value(hash, d.gateScratch.widthBulkMaxMm);
            hash_value(hash, d.gateScratch.widthMaxMm);
            hash_value(hash, d.gateScratch.widthTailFraction);
            hash_value(hash, d.gateScratch.driftFraction);
            hash_value(hash, d.gateScratch.firstKnotMin);
            hash_value(hash, d.gateScratch.firstKnotMax);
            hash_value(hash, d.gateScratch.secondKnotMin);
            hash_value(hash, d.gateScratch.secondKnotMax);
            hash_value(hash, d.gateScratch.interiorWidthMinFraction);
            hash_value(hash, d.gateScratch.interiorWidthMaxFraction);
            hash_value(hash, d.gateScratch.interiorDepthMinFraction);
            hash_value(hash, d.gateScratch.interiorDepthMaxFraction);
            hash_value(hash, d.gateScratch.endpointAbruptProbability);
            hash_value(hash, d.gateScratch.interruptionProbability);
            hash_value(hash, d.gateScratch.gapCenterMin);
            hash_value(hash, d.gateScratch.gapCenterMax);
            hash_value(hash, d.gateScratch.gapSpanMin);
            hash_value(hash, d.gateScratch.gapSpanMax);
            hash_value(hash, d.gateScratch.scuffProbability);
            hash_value(hash, d.gateScratch.scuffLengthMaxMm);
            hash_value(hash, d.gateScratch.scuffAngleMaxRadians);
            hash_value(hash, d.gateScratch.strengthMin);
            hash_value(hash, d.gateScratch.strengthMax);
            hash_value(hash, d.origins[3].cellX);
            hash_value(hash, d.origins[3].cellY);
            hash_value(hash, d.origins[3].localXMm);
            hash_value(hash, d.origins[3].localYMm);
        }
        return hash;
    }

    namespace {
        struct DefectAxisSplit {
            std::int64_t cell = 0;
            float localMm = 0;
        };

        bool split_defect_origin(const std::array<double, 2>& product, float cellMm, DefectAxisSplit& out) {
            const double size = static_cast<double>(cellMm);
            const double estimate = std::floor(product[0] / size);
            constexpr double kCellLimit = 4503599627370496.0;
            if (!std::isfinite(estimate) || std::abs(estimate) >= kCellLimit) {
                return false;
            }
            double remainder = std::fma(-estimate, size, product[0]) + product[1];
            const double correction = std::floor(remainder / size);
            const double index = estimate + correction;
            remainder = std::fma(-correction, size, remainder);
            if (!std::isfinite(index) || std::abs(index) >= kCellLimit ||
                !std::isfinite(remainder) || remainder < 0.0 || remainder >= size) {
                return false;
            }
            out.cell = static_cast<std::int64_t>(index);
            out.localMm = static_cast<float>(remainder);
            if (out.localMm >= cellMm) {
                ++out.cell;
                out.localMm = 0.0f;
            }
            return std::abs(static_cast<double>(out.cell)) < kCellLimit;
        }

        template <typename Policy>
        bool valid_defect_grid(const Policy& p, const DefectCellOrigin& o, const FilmJuicerEffectsFrameDescriptor& d) {
            const double nx = std::ceil((static_cast<double>(d.fullFrameExtent.width) * d.sampleStepXMm +
                                         2.0 * (p.supportXMm + p.softnessMaxMm + d.sampleStepXMm)) /
                                        p.cellWidthMm) +
                              2.0;
            const double ny = std::ceil((static_cast<double>(d.fullFrameExtent.height) * d.sampleStepYMm +
                                         2.0 * (p.supportYMm + p.softnessMaxMm + d.sampleStepYMm)) /
                                        p.cellHeightMm) +
                              2.0;
            constexpr double kCellLimit = 4503599627370496.0;
            return std::isfinite(nx) && std::isfinite(ny) && nx > 0 && ny > 0 &&
                   nx * ny * 2.0 < static_cast<double>(std::numeric_limits<int>::max()) &&
                   std::abs(static_cast<double>(o.cellX)) + nx < kCellLimit &&
                   std::abs(static_cast<double>(o.cellY)) + ny < kCellLimit;
        }
    } // namespace

    bool build_film_juicer_effects_frame_descriptor(
        const FilmJuicerEffectsFrameDescriptorInput& input,
        FilmJuicerEffectsFrameDescriptor& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        const auto fail = [&]() {
            out = {};
            diagnostic = "ResourceDescriptorMismatch phase=effects_descriptor field=geometry_or_policy";
            return false;
        };
        if (!input.recipe) {
            return fail();
        }
        if (!input.recipe->active) {
            return input.recipe->hash == 0;
        }
        const auto& g = input.geometry;
        const double x0 = g.canonicalX / g.pixelAspectRatio;
        const double w0 = g.canonicalWidth / g.pixelAspectRatio;
        const double h0 = g.canonicalHeight;
        if (!valid_extent(g.pixelDefinition) || !valid_extent(input.renderExtent) ||
            !extent_contains(g.pixelDefinition, input.renderExtent) ||
            g.pixelDefinition.width > std::numeric_limits<int>::max() - 64 ||
            g.pixelDefinition.height > std::numeric_limits<int>::max() - 64 ||
            !std::isfinite(x0) || !std::isfinite(g.canonicalY) ||
            !std::isfinite(w0) || !(w0 > 0) || !std::isfinite(h0) || !(h0 > 0) ||
            !std::isfinite(g.pixelAspectRatio) || !(g.pixelAspectRatio > 0) ||
            !std::isfinite(g.scaleX) || !(g.scaleX > 0) || !std::isfinite(g.scaleY) || !(g.scaleY > 0) ||
            !std::isfinite(input.filmFormatLongEdgeMm) || !(input.filmFormatLongEdgeMm > 0)) {
            return fail();
        }
        out.renderExtent = input.renderExtent;
        out.fullFrameExtent = g.pixelDefinition;
        out.sessionSeed = input.sessionSeed;
        out.clipToken = input.clipToken;
        out.filmDust = input.recipe->filmDust;
        out.filmScratch = input.recipe->filmScratch;
        out.gateDust = input.recipe->gateDust;
        out.gateScratch = input.recipe->gateScratch;
        out.filmActive = out.filmDust.slotProbability > 0 || out.filmScratch.slotProbability > 0;
        out.gateTransmittanceActive = out.gateDust.slotProbability > 0 || out.gateScratch.slotProbability > 0;
        out.weaveActive = input.recipe->gateWeaveAmount > 0;
        out.gateOutputActive = out.weaveActive || out.gateTransmittanceActive;
        out.requiresFullFrame = out.weaveActive;
        if (out.weaveActive && !same_extent(out.renderExtent, out.fullFrameExtent)) {
            return fail();
        }
        const std::int64_t roiOffsetX =
            static_cast<std::int64_t>(out.renderExtent.x) - out.fullFrameExtent.x;
        const std::int64_t roiOffsetY =
            static_cast<std::int64_t>(out.renderExtent.y) - out.fullFrameExtent.y;
        if (roiOffsetX < std::numeric_limits<int>::min() ||
            roiOffsetX > std::numeric_limits<int>::max() ||
            roiOffsetY < std::numeric_limits<int>::min() ||
            roiOffsetY > std::numeric_limits<int>::max()) {
            return fail();
        }
        out.roiOffsetX = static_cast<int>(roiOffsetX);
        out.roiOffsetY = static_cast<int>(roiOffsetY);
        const double mm = static_cast<double>(input.filmFormatLongEdgeMm) / std::max(w0, h0);
        if (!std::isfinite(mm) || !(mm > 0.0)) {
            return fail();
        }
        const double phaseX = (static_cast<double>(g.pixelDefinition.x) / g.scaleX - x0) * mm;
        const double phaseY = (static_cast<double>(g.pixelDefinition.y) / g.scaleY - g.canonicalY) * mm;
        if (out.filmActive || out.gateTransmittanceActive) {
            out.sampleStepXMm = static_cast<float>(mm / g.scaleX);
            out.sampleStepYMm = static_cast<float>(mm / g.scaleY);
            if (!std::isfinite(out.sampleStepXMm) || !(out.sampleStepXMm > 0.0f) ||
                !std::isfinite(out.sampleStepYMm) || !(out.sampleStepYMm > 0.0f)) {
                return fail();
            }
        }
        if (out.gateTransmittanceActive) {
            out.gateWidth = out.fullFrameExtent.width / 2 + out.fullFrameExtent.width % 2;
            out.gateHeight = out.fullFrameExtent.height / 2 + out.fullFrameExtent.height % 2;
        }
        double high = 0.0;
        double low = 0.0;
        if (out.filmActive) {
            const double advance = mm * h0;
            high = input.frameTime * advance;
            low = std::fma(input.frameTime, advance, -high);
            if (!std::isfinite(high) || !std::isfinite(low)) {
                return fail();
            }
        }
        if (out.filmDust.slotProbability > 0) {
            auto& o = out.origins[0];
            DefectAxisSplit x{}, y{};
            if (!split_defect_origin({phaseX, 0.0}, out.filmDust.cellWidthMm, x) ||
                !split_defect_origin({high, low + phaseY}, out.filmDust.cellHeightMm, y)) {
                return fail();
            }
            o = {x.cell, y.cell, x.localMm, y.localMm};
        }
        if (out.filmScratch.slotProbability > 0) {
            auto& o = out.origins[1];
            DefectAxisSplit x{}, y{};
            if (!split_defect_origin({phaseX, 0.0}, out.filmScratch.cellWidthMm, x) ||
                !split_defect_origin({high, low + phaseY}, out.filmScratch.cellHeightMm, y)) {
                return fail();
            }
            o = {x.cell, y.cell, x.localMm, y.localMm};
        }
        if (out.gateDust.slotProbability > 0) {
            auto& o = out.origins[2];
            DefectAxisSplit x{}, y{};
            if (!split_defect_origin({phaseX, 0.0}, out.gateDust.cellWidthMm, x) ||
                !split_defect_origin({phaseY, 0.0}, out.gateDust.cellHeightMm, y)) {
                return fail();
            }
            o = {x.cell, y.cell, x.localMm, y.localMm};
        }
        if (out.gateScratch.slotProbability > 0) {
            auto& o = out.origins[3];
            DefectAxisSplit x{}, y{};
            if (!split_defect_origin({phaseX, 0.0}, out.gateScratch.cellWidthMm, x) ||
                !split_defect_origin({phaseY, 0.0}, out.gateScratch.cellHeightMm, y)) {
                return fail();
            }
            o = {x.cell, y.cell, x.localMm, y.localMm};
        }
        if ((out.filmDust.slotProbability > 0 &&
             !valid_defect_grid(out.filmDust, out.origins[0], out)) ||
            (out.filmScratch.slotProbability > 0 &&
             !valid_defect_grid(out.filmScratch, out.origins[1], out)) ||
            (out.gateDust.slotProbability > 0 &&
             !valid_defect_grid(out.gateDust, out.origins[2], out)) ||
            (out.gateScratch.slotProbability > 0 &&
             !valid_defect_grid(out.gateScratch, out.origins[3], out))) {
            return fail();
        }
        if (out.weaveActive) {
            if (!std::isfinite(input.frameTime) || !std::isfinite(input.frameRate) || !(input.frameRate > 0) ||
                !std::isfinite(input.pixelSizeUm) || !(input.pixelSizeUm > 0)) {
                return fail();
            }
            const auto weave = compute_gate_weave(out.sessionSeed, input.frameTime / input.frameRate, 6.0, 0.005, static_cast<double>(input.pixelSizeUm), input.recipe->gateWeaveAmount);
            out.weaveDxPx = weave.dxPx;
            out.weaveDyPx = weave.dyPx;
            out.weaveCosRot = weave.cosRot;
            out.weaveSinRot = weave.sinRot;
            if (!std::isfinite(out.weaveDxPx) || !std::isfinite(out.weaveDyPx) ||
                !std::isfinite(out.weaveCosRot) || !std::isfinite(out.weaveSinRot)) {
                return fail();
            }
        }
        out.recipeHash = input.recipe->hash;
        out.hash = hash_film_juicer_effects_descriptor(out);
        return true;
    }
} // namespace Spektrafilm
