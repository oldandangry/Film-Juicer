#include "FilmEffectsFrameDescriptors.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "GaussianSciPy.h"
#include "Hash.h"

namespace {

    constexpr std::uint64_t kSeedPassGrain = 1;
    constexpr int kMaxGrainGaussianRadius = 75;

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

    bool nanmax(const std::vector<float>& values, float& outMax) {
        double maximum = -std::numeric_limits<double>::infinity();
        bool found = false;
        for (float value : values) {
            if (std::isfinite(value)) {
                maximum = std::max(maximum, static_cast<double>(value));
                found = true;
            }
        }
        if (!found || !std::isfinite(maximum)) {
            return false;
        }
        outMax = static_cast<float>(maximum);
        return std::isfinite(outMax);
    }

    Spektrafilm::VisualGrainGaussian make_gaussian(float sigma) {
        Spektrafilm::VisualGrainGaussian gaussian{};
        if (!std::isfinite(sigma) || !(sigma > 0.0f)) {
            return gaussian;
        }
        gaussian.sigmaPx = sigma;
        gaussian.radius = std::min(
            JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f),
            kMaxGrainGaussianRadius);
        if (gaussian.radius <= 0) {
            return Spektrafilm::VisualGrainGaussian{};
        }
        gaussian.hash = Hash::kFnvOffset;
        hash_value(gaussian.hash, gaussian.sigmaPx);
        hash_value(gaussian.hash, gaussian.radius);
        return gaussian;
    }

    float kernel_energy_2d(float sigma) {
        const Spektrafilm::VisualGrainGaussian gaussian = make_gaussian(sigma);
        if (gaussian.radius <= 0) {
            return 1.0f;
        }

        const double sigmaSquared =
            static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double weightSum = 0.0;
        std::array<double, (kMaxGrainGaussianRadius * 2) + 1> weights{};
        const int kernelWidth = (gaussian.radius * 2) + 1;
        for (int index = 0; index < kernelWidth; ++index) {
            const int offset = index - gaussian.radius;
            const double distance = static_cast<double>(offset);
            const double weight = std::exp(-(distance * distance) / sigmaSquared);
            weights[static_cast<std::size_t>(index)] = weight;
            weightSum += weight;
        }
        if (!(weightSum > 0.0) || !std::isfinite(weightSum)) {
            return 1.0f;
        }

        const double inverseWeightSum = 1.0 / weightSum;
        double squaredSum = 0.0;
        for (int index = 0; index < kernelWidth; ++index) {
            const double normalized =
                weights[static_cast<std::size_t>(index)] *
                inverseWeightSum;
            squaredSum += normalized * normalized;
        }
        return static_cast<float>(std::max(1.0e-12, squaredSum * squaredSum));
    }

    std::uint64_t make_seed_base(
        std::uint64_t clipToken,
        std::int64_t frameIndex,
        std::uint64_t sessionSeed) {
        const std::uint64_t fields[4] = {
            clipToken,
            static_cast<std::uint64_t>(frameIndex),
            sessionSeed,
            kSeedPassGrain};
        const std::uint64_t hash = Hash::hash_bytes(fields, sizeof(fields));
        return hash != 0 ? hash : 1;
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

    bool build_visual_grain_frame_descriptor(
        const VisualGrainFrameDescriptorInput& input,
        VisualGrainFrameDescriptor& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = VisualGrainFrameDescriptor{};
        if (!input.recipe || !input.filmDevelop) {
            diagnostic = "MissingRequiredResource phase=grain_descriptor field=recipe_or_film";
            return false;
        }
        if (!input.recipe->active) {
            return true;
        }
        if (input.recipe->hash == 0 ||
            (input.capturePolarity != ProfilePolarity::Negative &&
             input.capturePolarity != ProfilePolarity::Positive) ||
            input.filmDevelop->polarity != input.capturePolarity ||
            !valid_extent(input.renderExtent) ||
            !valid_extent(input.fullFrameExtent) ||
            !extent_contains(input.fullFrameExtent, input.renderExtent) ||
            !std::isfinite(input.pixelSizeUm) || !(input.pixelSizeUm > 0.0f) ||
            !std::isfinite(input.frameTime) ||
            !std::isfinite(input.frameRate) || !(input.frameRate > 0.0)) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=frame_input";
            return false;
        }

        const VisualGrainRecipe& recipe = *input.recipe;
        const FilmDevelopRecipe& film = *input.filmDevelop;
        if (recipe.sublayersActive &&
            (!film.densityCurvesLayersRequired ||
             film.densityCurvesLayersHash == 0 ||
             recipe.densityCurvesLayersHash != film.densityCurvesLayersHash)) {
            diagnostic =
                "ResourceDescriptorMismatch phase=grain_descriptor field=density_layers";
            return false;
        }

        const double frameFloor = std::floor(input.frameTime);
        if (frameFloor < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
            frameFloor >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            diagnostic = "UnsupportedMode phase=grain_descriptor field=frame_time";
            return false;
        }

        out.active = true;
        out.capturePolarity = input.capturePolarity;
        out.renderExtent = input.renderExtent;
        out.fullFrameExtent = input.fullFrameExtent;
        out.pixelSizeUm = input.pixelSizeUm;
        out.frame0 = static_cast<std::int64_t>(frameFloor);
        out.frameAlpha = static_cast<float>(
            std::clamp(input.frameTime - frameFloor, 0.0, 1.0));
        const std::int64_t frame1 = out.frame0 + 1;
        out.sessionSeed = input.sessionSeed;
        out.clipToken = input.clipToken;
        out.seedBase = make_seed_base(out.clipToken, out.frame0, out.sessionSeed);
        out.seedBaseNext =
            make_seed_base(out.clipToken, frame1, out.sessionSeed);
        out.pitchPx = input.fullFrameExtent.height;
        out.breathingPeriodFrames = std::max(
            1,
            static_cast<int>(std::llround(input.frameRate * 2.5)));
        out.clumpMorphPeriodFrames = std::max(
            1,
            static_cast<int>(
                std::llround(input.frameRate * recipe.clumpMorphPeriodSec)));
        const float filmFormatMm =
            input.pixelSizeUm *
            static_cast<float>(std::max(
                input.fullFrameExtent.width,
                input.fullFrameExtent.height)) /
            1000.0f;
        const float filmScale = filmFormatMm / 10.0f;
        out.wangCellMm = 2.0f;
        out.breathingAmplitude = 0.01902219f;
        out.breathingCellUmSmall = 2500.0f * filmScale;
        out.breathingCellUmLarge = 5000.0f * filmScale;
        out.breathingMix = 0.30f;
        out.breathingDriftUmPerFrame = 1.0f;
        out.recipeHash = recipe.hash;
        out.densityCurvesLayersHash =
            recipe.sublayersActive ? film.densityCurvesLayersHash : 0;

        const float pixelAreaUm2 = input.pixelSizeUm * input.pixelSizeUm;
        if (!std::isfinite(pixelAreaUm2) || !(pixelAreaUm2 > 0.0f)) {
            diagnostic = "ResourceDescriptorMismatch phase=grain_descriptor field=pixel_area";
            return false;
        }

        constexpr std::array<float, 3> kDefaultParticleScale{
            1.10f,
            1.27f,
            2.08f};
        constexpr float kDefaultParticleAreaUm2 = 0.335f;
        float blurAreaRatioSum = 0.0f;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const float densityMax =
                film.authoredMaxCmy[channel] +
                recipe.visualParticleDensityMinCmy[channel];
            const float particleArea =
                recipe.particleAreaUm2 * recipe.particleScaleCmy[channel];
            const float particleAreaReference =
                kDefaultParticleAreaUm2 * kDefaultParticleScale[channel];
            if (!std::isfinite(densityMax) || !(densityMax > 0.0f) ||
                !std::isfinite(particleArea) || !(particleArea > 0.0f) ||
                !std::isfinite(particleAreaReference) ||
                !(particleAreaReference > 0.0f)) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=grain_descriptor field=particle_model";
                return false;
            }
            float nParticles = pixelAreaUm2 / particleArea;
            if (recipe.nSubLayers > 1) {
                nParticles /= static_cast<float>(recipe.nSubLayers);
            }
            if (!std::isfinite(nParticles) || !(nParticles > 0.0f)) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=grain_descriptor field=particle_count";
                return false;
            }
            out.densityMaxCmy[channel] = densityMax;
            out.nParticlesCmy[channel] = nParticles;
            out.odParticleCmy[channel] = densityMax / nParticles;
            blurAreaRatioSum += particleArea / particleAreaReference;
        }
        const float densityMaxAverage =
            (out.densityMaxCmy[0] + out.densityMaxCmy[1] +
             out.densityMaxCmy[2]) /
            3.0f;
        out.debugScale =
            0.25f / std::max(1.0e-6f, densityMaxAverage);

        const float blurAreaRatio = blurAreaRatioSum / 3.0f;
        float fineSigma =
            recipe.correlationSigmaPx * std::sqrt(std::max(blurAreaRatio, 0.0f));
        const bool blurEnabled = recipe.sublayersActive
                                     ? fineSigma > 0.0f
                                     : fineSigma > 0.4f;
        if (!blurEnabled || !std::isfinite(fineSigma)) {
            fineSigma = 0.0f;
        }

        bool mixActive =
            fineSigma > 0.0f &&
            recipe.sizeMixScale > 1.0f &&
            (recipe.midWeight > 0.0f || recipe.coarseWeight > 0.0f);
        float midSigma = 0.0f;
        float coarseSigma = 0.0f;
        if (mixActive) {
            const float coarseRaw =
                fineSigma * std::sqrt(recipe.sizeMixScale);
            coarseSigma = std::max(
                fineSigma,
                std::min(coarseRaw, fineSigma * 4.0f));
            midSigma = std::sqrt(std::max(0.0f, fineSigma * coarseSigma));
            mixActive =
                std::isfinite(midSigma) && midSigma > 0.0f &&
                std::isfinite(coarseSigma) && coarseSigma > 0.0f;
        }

        out.correlation[0] = make_gaussian(fineSigma);
        if (mixActive) {
            out.correlation[1] = make_gaussian(midSigma);
            out.correlation[2] = make_gaussian(coarseSigma);
            out.effectiveFineWeight = recipe.fineWeight;
            out.effectiveMidWeight = recipe.midWeight;
            out.effectiveCoarseWeight = recipe.coarseWeight;
            const float energyFine = kernel_energy_2d(fineSigma);
            const float energyMid = kernel_energy_2d(midSigma);
            const float energyCoarse = kernel_energy_2d(coarseSigma);
            const float midScale = std::sqrt(recipe.sizeMixScale);
            const float ratioMid =
                midScale * energyMid / std::max(1.0e-12f, energyFine);
            const float ratioCoarse =
                recipe.sizeMixScale * energyCoarse /
                std::max(1.0e-12f, energyFine);
            const float denominator =
                out.effectiveFineWeight * out.effectiveFineWeight +
                out.effectiveMidWeight * out.effectiveMidWeight * ratioMid +
                out.effectiveCoarseWeight * out.effectiveCoarseWeight *
                    ratioCoarse;
            if (!std::isfinite(denominator) || !(denominator > 1.0e-12f)) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=grain_descriptor field=size_mix";
                return false;
            }
            out.sizeMixGain = 1.0f / std::sqrt(denominator);
        }

        if (recipe.sublayersActive) {
            std::array<std::array<float, 3>, 3> layerMax{};
            for (std::size_t layer = 0; layer < 3; ++layer) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    if (!nanmax(
                            film.densityCurvesLayers[layer][channel],
                            layerMax[layer][channel])) {
                        diagnostic =
                            "MalformedRequiredProfileData phase=grain_descriptor field=density_layers";
                        return false;
                    }
                }
            }
            for (std::size_t channel = 0; channel < 3; ++channel) {
                float total = 0.0f;
                for (std::size_t layer = 0; layer < 3; ++layer) {
                    total += layerMax[layer][channel];
                }
                if (!std::isfinite(total) || !(total > 0.0f)) {
                    diagnostic =
                        "MalformedRequiredProfileData phase=grain_descriptor field=density_layer_total";
                    return false;
                }
                for (std::size_t layer = 0; layer < 3; ++layer) {
                    const float fraction = layerMax[layer][channel] / total;
                    const float densityMin =
                        fraction *
                        recipe.visualParticleDensityMinCmy[channel];
                    const float densityMax =
                        layerMax[layer][channel] + densityMin;
                    const float particleArea =
                        recipe.particleAreaUm2 *
                        recipe.particleScaleCmy[channel] *
                        recipe.particleScaleLayers[layer];
                    if (!std::isfinite(particleArea) ||
                        !(particleArea > 0.0f)) {
                        diagnostic =
                            "ResourceDescriptorMismatch phase=grain_descriptor field=layer_particle_area";
                        return false;
                    }
                    const float nParticles =
                        pixelAreaUm2 * fraction / particleArea;
                    if (!std::isfinite(nParticles) ||
                        !(nParticles > 0.0f)) {
                        diagnostic =
                            "ResourceDescriptorMismatch phase=grain_descriptor field=layer_particle_count";
                        return false;
                    }
                    const float odParticle = densityMax / nParticles;
                    out.densityMinLayers[layer][channel] = densityMin;
                    out.densityMaxLayers[layer][channel] = densityMax;
                    out.nParticlesLayers[layer][channel] = nParticles;
                    out.odParticleLayers[layer][channel] = odParticle;
                    const float dyeSigma =
                        recipe.dyeCloudBlurUm *
                        std::sqrt(std::max(0.0f, odParticle));
                    out.dyeCloud[layer][channel] =
                        make_gaussian(dyeSigma);
                }
            }
        }

        const bool sharedChroma =
            (recipe.debugView == 0 || recipe.debugView == 1) &&
            recipe.chromaMix < 0.999f &&
            recipe.chromaSharedWeight > 0.0f;
        out.scratchShape =
            select_scratch_shape(recipe.sublayersActive, sharedChroma);
        out.requiresFullFrame = std::any_of(
            out.correlation.begin(),
            out.correlation.end(),
            [](const VisualGrainGaussian& gaussian) {
                return gaussian.radius > 0;
            });
        for (const auto& layer : out.dyeCloud) {
            out.requiresFullFrame =
                out.requiresFullFrame ||
                std::any_of(
                    layer.begin(),
                    layer.end(),
                    [](const VisualGrainGaussian& gaussian) {
                        return gaussian.radius > 0;
                    });
        }
        if (out.requiresFullFrame &&
            !same_extent(out.renderExtent, out.fullFrameExtent)) {
            diagnostic =
                "UnsupportedMode phase=grain_descriptor field=partial_frame";
            return false;
        }

        out.hash = hash_descriptor(out);
        if (out.hash == 0) {
            diagnostic =
                "ResourceDescriptorMismatch phase=grain_descriptor field=hash";
            return false;
        }
        return true;
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
