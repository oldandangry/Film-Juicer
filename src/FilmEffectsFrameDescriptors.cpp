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
        hash_value(hash, descriptor.staticNoiseVersion);
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
            !std::isfinite(input.frameRate) || !(input.frameRate > 0.0) ||
            input.staticNoiseVersion == 0) {
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
        out.sessionSeed = input.sessionSeed != 0 ? input.sessionSeed : 1;
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
        out.staticNoiseVersion = input.staticNoiseVersion;

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
            if (p.slotProbability == 0.0f) {
                return true;
            }
            if (!(p.slotProbability > 0.0f && p.slotProbability < 0.25f) ||
                !(p.cellWidthMm > 0.0f && p.cellHeightMm > 0.0f) ||
                !(p.softnessMinMm >= 0.0f && p.softnessMaxMm >= p.softnessMinMm &&
                  p.softnessSizeCapFraction > 0.0f && p.supportXMm > 0.0f && p.supportYMm > 0.0f) ||
                !(o.localXMm >= 0.0f && o.localXMm < p.cellWidthMm &&
                  o.localYMm >= 0.0f && o.localYMm < p.cellHeightMm)) {
                return false;
            }
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

        bool valid_opacity_boundaries(float minimum, float faintEnd, float intermediateEnd, float maximum) {
            return minimum >= 0.0f && minimum < faintEnd && faintEnd < intermediateEnd &&
                   intermediateEnd < maximum && maximum < 1.0f;
        }

        bool valid_dust_policy(const DefectDustRecipe& p) {
            const float values[] = {
                p.cellWidthMm,
                p.cellHeightMm,
                p.slotProbability,
                p.softnessMinMm,
                p.softnessMaxMm,
                p.softnessSizeCapFraction,
                p.supportXMm,
                p.supportYMm,
                p.fiberFraction,
                p.fiberDriftFraction,
                p.fiberFirstKnotMin,
                p.fiberFirstKnotMax,
                p.fiberSecondKnotMin,
                p.fiberSecondKnotMax,
                p.fiberInteriorWidthMinFraction,
                p.fiberInteriorWidthMaxFraction,
                p.diameterMinMm,
                p.diameterBulkMaxMm,
                p.diameterMaxMm,
                p.diameterTailFraction,
                p.fiberLengthMinMm,
                p.fiberLengthMaxMm,
                p.fiberWidthMinMm,
                p.fiberWidthMaxMm,
                p.opacityFaintCumulative,
                p.opacityIntermediateCumulative,
                p.compactOpacityMin,
                p.compactOpacityFaintEnd,
                p.compactOpacityIntermediateEnd,
                p.compactOpacityMax,
                p.fiberOpacityMin,
                p.fiberOpacityFaintEnd,
                p.fiberOpacityIntermediateEnd,
                p.fiberOpacityMax,
                p.compactDominantAspectMin,
                p.compactDominantAspectMax,
                p.compactSubsidiaryScaleMin,
                p.compactSubsidiaryScaleMax,
                p.compactSubsidiaryAspectMin,
                p.compactSubsidiaryAspectMax,
                p.compactSubsidiaryOffsetMax,
                p.compactSubsidiaryAngleMaxRadians};
            if (!std::all_of(std::begin(values), std::end(values), [](float value) {
                    return std::isfinite(value);
                })) {
                return false;
            }
            const float fiberSupport =
                std::fma(p.fiberLengthMaxMm, 0.5f + p.fiberDriftFraction, p.fiberWidthMaxMm);
            const float geometrySupport = std::max(p.diameterMaxMm * 0.5f, fiberSupport);
            return p.cellWidthMm > 0.0f && p.cellHeightMm > 0.0f &&
                   p.slotProbability > 0.0f && p.slotProbability < 0.25f &&
                   p.softnessMinMm >= 0.0f && p.softnessMaxMm >= p.softnessMinMm &&
                   p.softnessSizeCapFraction > 0.0f && p.softnessSizeCapFraction <= 0.5f &&
                   p.supportXMm >= geometrySupport && p.supportYMm >= geometrySupport &&
                   p.fiberFraction >= 0.0f && p.fiberFraction <= 1.0f &&
                   p.fiberDriftFraction >= 0.0f && p.fiberDriftFraction <= 0.1f &&
                   p.fiberFirstKnotMin > 0.0f && p.fiberFirstKnotMin <= p.fiberFirstKnotMax &&
                   p.fiberFirstKnotMax < p.fiberSecondKnotMin &&
                   p.fiberSecondKnotMin <= p.fiberSecondKnotMax && p.fiberSecondKnotMax < 1.0f &&
                   p.fiberInteriorWidthMinFraction > 0.0f &&
                   p.fiberInteriorWidthMinFraction <= p.fiberInteriorWidthMaxFraction &&
                   p.fiberInteriorWidthMaxFraction <= 1.0f &&
                   p.diameterMinMm > 0.0f && p.diameterBulkMaxMm >= p.diameterMinMm &&
                   p.diameterMaxMm >= p.diameterBulkMaxMm &&
                   p.diameterTailFraction >= 0.0f && p.diameterTailFraction <= 1.0f &&
                   p.fiberLengthMinMm > 0.0f && p.fiberLengthMaxMm >= p.fiberLengthMinMm &&
                   p.fiberWidthMinMm > 0.0f && p.fiberWidthMaxMm >= p.fiberWidthMinMm &&
                   p.opacityFaintCumulative > 0.0f &&
                   p.opacityFaintCumulative < p.opacityIntermediateCumulative &&
                   p.opacityIntermediateCumulative < 1.0f &&
                   valid_opacity_boundaries(p.compactOpacityMin,
                                            p.compactOpacityFaintEnd,
                                            p.compactOpacityIntermediateEnd,
                                            p.compactOpacityMax) &&
                   valid_opacity_boundaries(p.fiberOpacityMin,
                                            p.fiberOpacityFaintEnd,
                                            p.fiberOpacityIntermediateEnd,
                                            p.fiberOpacityMax) &&
                   p.compactDominantAspectMin > 0.0f &&
                   p.compactDominantAspectMin <= p.compactDominantAspectMax &&
                   p.compactDominantAspectMax <= 1.0f &&
                   p.compactSubsidiaryScaleMin > 0.0f &&
                   p.compactSubsidiaryScaleMin <= p.compactSubsidiaryScaleMax &&
                   p.compactSubsidiaryScaleMax <= 1.0f &&
                   p.compactSubsidiaryAspectMin > 0.0f &&
                   p.compactSubsidiaryAspectMin <= p.compactSubsidiaryAspectMax &&
                   p.compactSubsidiaryAspectMax <= 1.0f &&
                   p.compactSubsidiaryOffsetMax >= 0.0f && p.compactSubsidiaryOffsetMax < 1.0f &&
                   p.compactSubsidiaryAngleMaxRadians > 0.0f &&
                   p.compactSubsidiaryAngleMaxRadians <= 3.14159265359f;
        }

        bool valid_scratch_policy(const DefectScratchRecipe& p) {
            const float values[] = {
                p.cellWidthMm,
                p.cellHeightMm,
                p.slotProbability,
                p.softnessMinMm,
                p.softnessMaxMm,
                p.softnessSizeCapFraction,
                p.supportXMm,
                p.supportYMm,
                p.lengthMinMm,
                p.lengthBulkMaxMm,
                p.lengthMaxMm,
                p.lengthTailFraction,
                p.widthMinMm,
                p.widthBulkMaxMm,
                p.widthMaxMm,
                p.widthTailFraction,
                p.driftFraction,
                p.firstKnotMin,
                p.firstKnotMax,
                p.secondKnotMin,
                p.secondKnotMax,
                p.interiorWidthMinFraction,
                p.interiorWidthMaxFraction,
                p.interiorDepthMinFraction,
                p.interiorDepthMaxFraction,
                p.endpointAbruptProbability,
                p.interruptionProbability,
                p.gapCenterMin,
                p.gapCenterMax,
                p.gapSpanMin,
                p.gapSpanMax,
                p.scuffProbability,
                p.scuffLengthMaxMm,
                p.scuffAngleMaxRadians,
                p.strengthMin,
                p.strengthMax};
            if (!std::all_of(std::begin(values), std::end(values), [](float value) {
                    return std::isfinite(value);
                })) {
                return false;
            }
            const float transportSupportX = p.lengthMaxMm * p.driftFraction + p.widthMaxMm;
            const float scuffSupportX = 0.5f * p.scuffLengthMaxMm * std::sin(p.scuffAngleMaxRadians) +
                                        p.scuffLengthMaxMm * p.driftFraction + p.widthMaxMm;
            const float transportSupportY = p.lengthMaxMm * 0.5f + p.widthMaxMm;
            const float scuffSupportY = 0.5f * p.scuffLengthMaxMm * std::cos(p.scuffAngleMaxRadians) +
                                        p.scuffLengthMaxMm * p.driftFraction + p.widthMaxMm;
            return p.cellWidthMm > 0.0f && p.cellHeightMm > 0.0f &&
                   p.slotProbability > 0.0f && p.slotProbability < 0.25f &&
                   p.softnessMinMm >= 0.0f && p.softnessMaxMm >= p.softnessMinMm &&
                   p.softnessSizeCapFraction > 0.0f && p.softnessSizeCapFraction <= 0.5f &&
                   p.supportXMm >= std::max(transportSupportX, scuffSupportX) &&
                   p.supportYMm >= std::max(transportSupportY, scuffSupportY) &&
                   p.lengthMinMm > 0.0f && p.lengthBulkMaxMm >= p.lengthMinMm &&
                   p.lengthMaxMm >= p.lengthBulkMaxMm &&
                   p.lengthTailFraction >= 0.0f && p.lengthTailFraction <= 1.0f &&
                   p.widthMinMm > 0.0f && p.widthBulkMaxMm >= p.widthMinMm &&
                   p.widthMaxMm >= p.widthBulkMaxMm &&
                   p.widthTailFraction >= 0.0f && p.widthTailFraction <= 1.0f &&
                   p.driftFraction >= 0.0f &&
                   p.firstKnotMin > 0.0f && p.firstKnotMin <= p.firstKnotMax &&
                   p.firstKnotMax < p.secondKnotMin &&
                   p.secondKnotMin <= p.secondKnotMax && p.secondKnotMax < 1.0f &&
                   p.interiorWidthMinFraction > 0.0f &&
                   p.interiorWidthMinFraction <= p.interiorWidthMaxFraction &&
                   p.interiorWidthMaxFraction <= 1.0f &&
                   p.interiorDepthMinFraction > 0.0f &&
                   p.interiorDepthMinFraction <= p.interiorDepthMaxFraction &&
                   p.interiorDepthMaxFraction <= 1.0f &&
                   p.endpointAbruptProbability >= 0.0f && p.endpointAbruptProbability <= 1.0f &&
                   p.interruptionProbability >= 0.0f && p.interruptionProbability <= 1.0f &&
                   p.gapCenterMin > 0.0f && p.gapCenterMin <= p.gapCenterMax &&
                   p.gapCenterMax < 1.0f && p.gapSpanMin > 0.0f &&
                   p.gapSpanMin <= p.gapSpanMax &&
                   p.gapCenterMin - 0.5f * p.gapSpanMax > 0.0f &&
                   p.gapCenterMax + 0.5f * p.gapSpanMax < 1.0f &&
                   p.scuffProbability >= 0.0f && p.scuffProbability <= 1.0f &&
                   p.scuffLengthMaxMm >= p.lengthMinMm && p.scuffLengthMaxMm <= p.lengthMaxMm &&
                   p.scuffAngleMaxRadians > 0.0f && p.scuffAngleMaxRadians <= 1.57079632679f &&
                   p.strengthMin >= 0.0f && p.strengthMax >= p.strengthMin && p.strengthMax <= 1.0f;
        }
    } // namespace

    bool validate_film_juicer_effects_frame_descriptor(const FilmJuicerEffectsFrameDescriptor& d) {
        if (d.filmDust.slotProbability == 0.0f) {
            if (d.filmDust.cellWidthMm != 0.0f ||
                d.filmDust.cellHeightMm != 0.0f ||
                d.filmDust.slotProbability != 0.0f ||
                d.filmDust.softnessMinMm != 0.0f || d.filmDust.softnessMaxMm != 0.0f ||
                d.filmDust.softnessSizeCapFraction != 0.0f ||
                d.filmDust.supportXMm != 0.0f ||
                d.filmDust.supportYMm != 0.0f ||
                d.filmDust.fiberFraction != 0.0f ||
                d.filmDust.fiberDriftFraction != 0.0f ||
                d.filmDust.fiberFirstKnotMin != 0.0f || d.filmDust.fiberFirstKnotMax != 0.0f ||
                d.filmDust.fiberSecondKnotMin != 0.0f || d.filmDust.fiberSecondKnotMax != 0.0f ||
                d.filmDust.fiberInteriorWidthMinFraction != 0.0f || d.filmDust.fiberInteriorWidthMaxFraction != 0.0f ||
                d.filmDust.diameterMinMm != 0.0f ||
                d.filmDust.diameterBulkMaxMm != 0.0f ||
                d.filmDust.diameterMaxMm != 0.0f ||
                d.filmDust.diameterTailFraction != 0.0f ||
                d.filmDust.fiberLengthMinMm != 0.0f ||
                d.filmDust.fiberLengthMaxMm != 0.0f ||
                d.filmDust.fiberWidthMinMm != 0.0f ||
                d.filmDust.fiberWidthMaxMm != 0.0f ||
                d.filmDust.opacityFaintCumulative != 0.0f || d.filmDust.opacityIntermediateCumulative != 0.0f ||
                d.filmDust.compactOpacityMin != 0.0f || d.filmDust.compactOpacityFaintEnd != 0.0f ||
                d.filmDust.compactOpacityIntermediateEnd != 0.0f || d.filmDust.compactOpacityMax != 0.0f ||
                d.filmDust.fiberOpacityMin != 0.0f || d.filmDust.fiberOpacityFaintEnd != 0.0f ||
                d.filmDust.fiberOpacityIntermediateEnd != 0.0f || d.filmDust.fiberOpacityMax != 0.0f ||
                d.filmDust.compactDominantAspectMin != 0.0f || d.filmDust.compactDominantAspectMax != 0.0f ||
                d.filmDust.compactSubsidiaryScaleMin != 0.0f || d.filmDust.compactSubsidiaryScaleMax != 0.0f ||
                d.filmDust.compactSubsidiaryAspectMin != 0.0f || d.filmDust.compactSubsidiaryAspectMax != 0.0f ||
                d.filmDust.compactSubsidiaryOffsetMax != 0.0f || d.filmDust.compactSubsidiaryAngleMaxRadians != 0.0f ||
                d.origins[0].cellX != 0 || d.origins[0].cellY != 0 ||
                d.origins[0].localXMm != 0 || d.origins[0].localYMm != 0) {
                return false;
            }
        }
        if (d.filmScratch.slotProbability == 0.0f) {
            if (d.filmScratch.cellWidthMm != 0.0f ||
                d.filmScratch.cellHeightMm != 0.0f ||
                d.filmScratch.slotProbability != 0.0f ||
                d.filmScratch.softnessMinMm != 0.0f || d.filmScratch.softnessMaxMm != 0.0f ||
                d.filmScratch.softnessSizeCapFraction != 0.0f ||
                d.filmScratch.supportXMm != 0.0f ||
                d.filmScratch.supportYMm != 0.0f ||
                d.filmScratch.lengthMinMm != 0.0f ||
                d.filmScratch.lengthBulkMaxMm != 0.0f ||
                d.filmScratch.lengthMaxMm != 0.0f ||
                d.filmScratch.lengthTailFraction != 0.0f ||
                d.filmScratch.widthMinMm != 0.0f ||
                d.filmScratch.widthBulkMaxMm != 0.0f ||
                d.filmScratch.widthMaxMm != 0.0f ||
                d.filmScratch.widthTailFraction != 0.0f ||
                d.filmScratch.driftFraction != 0.0f ||
                d.filmScratch.firstKnotMin != 0.0f || d.filmScratch.firstKnotMax != 0.0f ||
                d.filmScratch.secondKnotMin != 0.0f || d.filmScratch.secondKnotMax != 0.0f ||
                d.filmScratch.interiorWidthMinFraction != 0.0f || d.filmScratch.interiorWidthMaxFraction != 0.0f ||
                d.filmScratch.interiorDepthMinFraction != 0.0f || d.filmScratch.interiorDepthMaxFraction != 0.0f ||
                d.filmScratch.endpointAbruptProbability != 0.0f || d.filmScratch.interruptionProbability != 0.0f ||
                d.filmScratch.gapCenterMin != 0.0f || d.filmScratch.gapCenterMax != 0.0f ||
                d.filmScratch.gapSpanMin != 0.0f || d.filmScratch.gapSpanMax != 0.0f ||
                d.filmScratch.scuffProbability != 0.0f || d.filmScratch.scuffLengthMaxMm != 0.0f ||
                d.filmScratch.scuffAngleMaxRadians != 0.0f ||
                d.filmScratch.strengthMin != 0.0f ||
                d.filmScratch.strengthMax != 0.0f ||
                d.origins[1].cellX != 0 || d.origins[1].cellY != 0 ||
                d.origins[1].localXMm != 0 || d.origins[1].localYMm != 0) {
                return false;
            }
        }
        if (d.gateDust.slotProbability == 0.0f) {
            if (d.gateDust.cellWidthMm != 0.0f ||
                d.gateDust.cellHeightMm != 0.0f ||
                d.gateDust.slotProbability != 0.0f ||
                d.gateDust.softnessMinMm != 0.0f || d.gateDust.softnessMaxMm != 0.0f ||
                d.gateDust.softnessSizeCapFraction != 0.0f ||
                d.gateDust.supportXMm != 0.0f ||
                d.gateDust.supportYMm != 0.0f ||
                d.gateDust.fiberFraction != 0.0f ||
                d.gateDust.fiberDriftFraction != 0.0f ||
                d.gateDust.fiberFirstKnotMin != 0.0f || d.gateDust.fiberFirstKnotMax != 0.0f ||
                d.gateDust.fiberSecondKnotMin != 0.0f || d.gateDust.fiberSecondKnotMax != 0.0f ||
                d.gateDust.fiberInteriorWidthMinFraction != 0.0f || d.gateDust.fiberInteriorWidthMaxFraction != 0.0f ||
                d.gateDust.diameterMinMm != 0.0f ||
                d.gateDust.diameterBulkMaxMm != 0.0f ||
                d.gateDust.diameterMaxMm != 0.0f ||
                d.gateDust.diameterTailFraction != 0.0f ||
                d.gateDust.fiberLengthMinMm != 0.0f ||
                d.gateDust.fiberLengthMaxMm != 0.0f ||
                d.gateDust.fiberWidthMinMm != 0.0f ||
                d.gateDust.fiberWidthMaxMm != 0.0f ||
                d.gateDust.opacityFaintCumulative != 0.0f || d.gateDust.opacityIntermediateCumulative != 0.0f ||
                d.gateDust.compactOpacityMin != 0.0f || d.gateDust.compactOpacityFaintEnd != 0.0f ||
                d.gateDust.compactOpacityIntermediateEnd != 0.0f || d.gateDust.compactOpacityMax != 0.0f ||
                d.gateDust.fiberOpacityMin != 0.0f || d.gateDust.fiberOpacityFaintEnd != 0.0f ||
                d.gateDust.fiberOpacityIntermediateEnd != 0.0f || d.gateDust.fiberOpacityMax != 0.0f ||
                d.gateDust.compactDominantAspectMin != 0.0f || d.gateDust.compactDominantAspectMax != 0.0f ||
                d.gateDust.compactSubsidiaryScaleMin != 0.0f || d.gateDust.compactSubsidiaryScaleMax != 0.0f ||
                d.gateDust.compactSubsidiaryAspectMin != 0.0f || d.gateDust.compactSubsidiaryAspectMax != 0.0f ||
                d.gateDust.compactSubsidiaryOffsetMax != 0.0f || d.gateDust.compactSubsidiaryAngleMaxRadians != 0.0f ||
                d.origins[2].cellX != 0 || d.origins[2].cellY != 0 ||
                d.origins[2].localXMm != 0 || d.origins[2].localYMm != 0) {
                return false;
            }
        }
        if (d.gateScratch.slotProbability == 0.0f) {
            if (d.gateScratch.cellWidthMm != 0.0f ||
                d.gateScratch.cellHeightMm != 0.0f ||
                d.gateScratch.slotProbability != 0.0f ||
                d.gateScratch.softnessMinMm != 0.0f || d.gateScratch.softnessMaxMm != 0.0f ||
                d.gateScratch.softnessSizeCapFraction != 0.0f ||
                d.gateScratch.supportXMm != 0.0f ||
                d.gateScratch.supportYMm != 0.0f ||
                d.gateScratch.lengthMinMm != 0.0f ||
                d.gateScratch.lengthBulkMaxMm != 0.0f ||
                d.gateScratch.lengthMaxMm != 0.0f ||
                d.gateScratch.lengthTailFraction != 0.0f ||
                d.gateScratch.widthMinMm != 0.0f ||
                d.gateScratch.widthBulkMaxMm != 0.0f ||
                d.gateScratch.widthMaxMm != 0.0f ||
                d.gateScratch.widthTailFraction != 0.0f ||
                d.gateScratch.driftFraction != 0.0f ||
                d.gateScratch.firstKnotMin != 0.0f || d.gateScratch.firstKnotMax != 0.0f ||
                d.gateScratch.secondKnotMin != 0.0f || d.gateScratch.secondKnotMax != 0.0f ||
                d.gateScratch.interiorWidthMinFraction != 0.0f || d.gateScratch.interiorWidthMaxFraction != 0.0f ||
                d.gateScratch.interiorDepthMinFraction != 0.0f || d.gateScratch.interiorDepthMaxFraction != 0.0f ||
                d.gateScratch.endpointAbruptProbability != 0.0f || d.gateScratch.interruptionProbability != 0.0f ||
                d.gateScratch.gapCenterMin != 0.0f || d.gateScratch.gapCenterMax != 0.0f ||
                d.gateScratch.gapSpanMin != 0.0f || d.gateScratch.gapSpanMax != 0.0f ||
                d.gateScratch.scuffProbability != 0.0f || d.gateScratch.scuffLengthMaxMm != 0.0f ||
                d.gateScratch.scuffAngleMaxRadians != 0.0f ||
                d.gateScratch.strengthMin != 0.0f ||
                d.gateScratch.strengthMax != 0.0f ||
                d.origins[3].cellX != 0 || d.origins[3].cellY != 0 ||
                d.origins[3].localXMm != 0 || d.origins[3].localYMm != 0) {
                return false;
            }
        }
        if (!d.filmActive && !d.gateOutputActive) {
            if (d.renderExtent.x != 0 || d.renderExtent.y != 0 || d.renderExtent.width != 0 || d.renderExtent.height != 0 ||
                d.fullFrameExtent.x != 0 || d.fullFrameExtent.y != 0 || d.fullFrameExtent.width != 0 || d.fullFrameExtent.height != 0 ||
                d.sampleStepXMm != 0 || d.sampleStepYMm != 0 || d.roiOffsetX != 0 || d.roiOffsetY != 0 ||
                d.gateWidth != 0 || d.gateHeight != 0 || d.sessionSeed != 0 || d.clipToken != 0 ||
                d.weaveDxPx != 0 || d.weaveDyPx != 0 || d.weaveCosRot != 1 || d.weaveSinRot != 0 || d.requiresFullFrame) {
                return false;
            }
            return d.hash == 0 && d.recipeHash == 0 && !d.weaveActive && !d.gateTransmittanceActive &&
                   d.filmDust.slotProbability == 0.0f && d.filmScratch.slotProbability == 0.0f &&
                   d.gateDust.slotProbability == 0.0f && d.gateScratch.slotProbability == 0.0f;
        }
        if (d.fullFrameExtent.width > std::numeric_limits<int>::max() - 64 ||
            d.fullFrameExtent.height > std::numeric_limits<int>::max() - 64) {
            return false;
        }
        if (!valid_extent(d.renderExtent) || !valid_extent(d.fullFrameExtent) ||
            !extent_contains(d.fullFrameExtent, d.renderExtent) ||
            d.sessionSeed == 0 || d.recipeHash == 0 || d.hash == 0 ||
            d.hash != hash_film_juicer_effects_descriptor(d) ||
            d.filmActive != (d.filmDust.slotProbability > 0 || d.filmScratch.slotProbability > 0) ||
            d.gateTransmittanceActive != (d.gateDust.slotProbability > 0 || d.gateScratch.slotProbability > 0) ||
            d.gateOutputActive != (d.weaveActive || d.gateTransmittanceActive) ||
            d.requiresFullFrame != d.weaveActive ||
            (d.requiresFullFrame && !same_extent(d.renderExtent, d.fullFrameExtent)) ||
            d.roiOffsetX != static_cast<std::int64_t>(d.renderExtent.x) - d.fullFrameExtent.x ||
            d.roiOffsetY != static_cast<std::int64_t>(d.renderExtent.y) - d.fullFrameExtent.y ||
            !std::isfinite(d.weaveDxPx) || !std::isfinite(d.weaveDyPx) ||
            !std::isfinite(d.weaveCosRot) || !std::isfinite(d.weaveSinRot)) {
            return false;
        }
        if (d.filmActive || d.gateTransmittanceActive) {
            if (!(d.sampleStepXMm > 0.0f && d.sampleStepYMm > 0.0f) ||
                !std::isfinite(d.sampleStepXMm) || !std::isfinite(d.sampleStepYMm)) {
                return false;
            }
        }
        if (d.gateTransmittanceActive && (d.gateWidth != d.fullFrameExtent.width / 2 + d.fullFrameExtent.width % 2 ||
                                          d.gateHeight != d.fullFrameExtent.height / 2 + d.fullFrameExtent.height % 2)) {
            return false;
        }
        if (d.filmDust.slotProbability != 0.0f) {
            const auto& p = d.filmDust;
            if (!valid_dust_policy(p) || !valid_defect_grid(p, d.origins[0], d)) {
                return false;
            }
        }
        if (d.filmScratch.slotProbability != 0.0f) {
            const auto& p = d.filmScratch;
            if (!valid_scratch_policy(p) || !valid_defect_grid(p, d.origins[1], d)) {
                return false;
            }
        }
        if (d.gateDust.slotProbability != 0.0f) {
            const auto& p = d.gateDust;
            if (!valid_dust_policy(p) || !valid_defect_grid(p, d.origins[2], d)) {
                return false;
            }
        }
        if (d.gateScratch.slotProbability != 0.0f) {
            const auto& p = d.gateScratch;
            if (!valid_scratch_policy(p) || !valid_defect_grid(p, d.origins[3], d)) {
                return false;
            }
        }
        return true;
    }

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
            !std::isfinite(x0) || !std::isfinite(g.canonicalY) ||
            !std::isfinite(w0) || !(w0 > 0) || !std::isfinite(h0) || !(h0 > 0) ||
            !std::isfinite(g.pixelAspectRatio) || !(g.pixelAspectRatio > 0) ||
            !std::isfinite(g.scaleX) || !(g.scaleX > 0) || !std::isfinite(g.scaleY) || !(g.scaleY > 0) ||
            !std::isfinite(input.filmFormatLongEdgeMm) || !(input.filmFormatLongEdgeMm > 0)) {
            return fail();
        }
        out.renderExtent = input.renderExtent;
        out.fullFrameExtent = g.pixelDefinition;
        out.sessionSeed = input.sessionSeed ? input.sessionSeed : 1;
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
        out.roiOffsetX = static_cast<int>(static_cast<std::int64_t>(out.renderExtent.x) - out.fullFrameExtent.x);
        out.roiOffsetY = static_cast<int>(static_cast<std::int64_t>(out.renderExtent.y) - out.fullFrameExtent.y);
        const double mm = static_cast<double>(input.filmFormatLongEdgeMm) / std::max(w0, h0);
        const double phaseX = (static_cast<double>(g.pixelDefinition.x) / g.scaleX - x0) * mm;
        const double phaseY = (static_cast<double>(g.pixelDefinition.y) / g.scaleY - g.canonicalY) * mm;
        if (out.filmActive || out.gateTransmittanceActive) {
            out.sampleStepXMm = static_cast<float>(mm / g.scaleX);
            out.sampleStepYMm = static_cast<float>(mm / g.scaleY);
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
        }
        out.recipeHash = input.recipe->hash;
        out.hash = hash_film_juicer_effects_descriptor(out);
        return validate_film_juicer_effects_frame_descriptor(out) || fail();
    }
} // namespace Spektrafilm
