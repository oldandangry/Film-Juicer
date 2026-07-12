#include "VisualGrainFrameDescriptor.h"

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

    bool valid_extent(const Spektrafilm::VisualGrainFrameExtent& extent) {
        return extent.width > 0 && extent.height > 0;
    }

    bool same_extent(
        const Spektrafilm::VisualGrainFrameExtent& a,
        const Spektrafilm::VisualGrainFrameExtent& b) {
        return a.x == b.x && a.y == b.y &&
               a.width == b.width && a.height == b.height;
    }

    bool extent_contains(
        const Spektrafilm::VisualGrainFrameExtent& outer,
        const Spektrafilm::VisualGrainFrameExtent& inner) {
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

    void hash_extent(
        std::uint64_t& hash,
        const Spektrafilm::VisualGrainFrameExtent& extent) {
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
