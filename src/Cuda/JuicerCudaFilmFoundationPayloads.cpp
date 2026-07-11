#include "Cuda/JuicerCudaFilmFoundationPayloads.h"

#include <cstddef>
#include <cstring>

#include "Hash.h"

namespace {

    // NOLINTBEGIN(bugprone-easily-swappable-parameters) STBN helpers mirror the reviewed seed formula.
    int stbn_offset(
        std::uint64_t sessionSeed,
        int dimension,
        std::uint64_t salt) {
        if (dimension <= 0) {
            return 0;
        }
        const std::uint64_t hash =
            Hash::hash_uint64_values({sessionSeed, salt});
        return static_cast<int>(
            hash % static_cast<std::uint64_t>(dimension));
    }

    int stbn_frame_index(
        std::int64_t frameIndex,
        int frameCount,
        std::uint64_t sessionSeed) {
        if (frameCount <= 0) {
            return 0;
        }
        const std::int64_t phase = static_cast<std::int64_t>(
            sessionSeed % static_cast<std::uint64_t>(frameCount));
        const std::int64_t temporalIndex = frameIndex + phase;
        int frame = static_cast<int>(temporalIndex % frameCount);
        if (frame < 0) {
            frame += frameCount;
        }
        return frame;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    bool gaussian_binding_ready(
        const Spektrafilm::VisualGrainGaussian& descriptor,
        const JuicerProcess::Root::PreparedCudaFrame::PreparedGaussianView& prepared) {
        if (descriptor.radius <= 0) {
            return descriptor.hash == 0 && !prepared.active &&
                   !prepared.kernel.weights && prepared.kernel.radius == 0 &&
                   prepared.descriptorHash == 0;
        }
        return descriptor.hash != 0 && prepared.active &&
               prepared.descriptorHash == descriptor.hash &&
               prepared.kernel.weights &&
               prepared.kernel.radius == descriptor.radius &&
               prepared.kernel.sigma == descriptor.sigmaPx;
    }

} // namespace

namespace JuicerCuda {

    bool pack_visual_grain_payload(
        const Spektrafilm::VisualGrainRecipe& recipe,
        const JuicerProcess::Root::PreparedCudaFrame::PreparedVisualGrainView& prepared,
        GrainPayload& outGrain,
        GrainKernelPayload& outKernels,
        std::string& diagnostic) {
        std::memset(&outGrain, 0, sizeof(outGrain));
        std::memset(&outKernels, 0, sizeof(outKernels));
        diagnostic.clear();

        if (!recipe.active) {
            if (recipe.hash != 0 || prepared.active || prepared.descriptor) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=grain_payload field=inactive_state";
                return false;
            }
            return true;
        }
        if (recipe.hash == 0 || !prepared.active || !prepared.descriptor) {
            diagnostic =
                "MissingRequiredResource phase=grain_payload field=prepared_view";
            return false;
        }

        const Spektrafilm::VisualGrainFrameDescriptor& descriptor =
            *prepared.descriptor;
        if (!descriptor.active || descriptor.hash == 0 ||
            descriptor.recipeHash != recipe.hash ||
            (descriptor.capturePolarity !=
                 Spektrafilm::ProfilePolarity::Negative &&
             descriptor.capturePolarity !=
                 Spektrafilm::ProfilePolarity::Positive) ||
            descriptor.staticNoiseVersion == 0 ||
            descriptor.renderExtent.width <= 0 ||
            descriptor.renderExtent.height <= 0 ||
            !(descriptor.pixelSizeUm > 0.0f) || descriptor.pitchPx <= 0) {
            diagnostic =
                "ResourceDescriptorMismatch phase=grain_payload field=descriptor";
            return false;
        }

        const auto& noise = prepared.staticNoise;
        if (!noise.stbn || noise.stbnWidth <= 0 || noise.stbnHeight <= 0 ||
            noise.stbnFrames <= 0) {
            diagnostic =
                "MissingRequiredResource phase=grain_payload field=stbn";
            return false;
        }
        if (!noise.wangTiles || !noise.wangLut || noise.wangWidth <= 0 ||
            noise.wangHeight <= 0 || noise.wangCount <= 0 ||
            noise.wangColors <= 0 ||
            noise.version != descriptor.staticNoiseVersion) {
            diagnostic =
                "ResourceDescriptorMismatch phase=grain_payload field=wang_identity";
            return false;
        }

        for (std::size_t index = 0;
             index < descriptor.correlation.size();
             ++index) {
            if (!gaussian_binding_ready(
                    descriptor.correlation[index],
                    prepared.correlation[index])) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=grain_payload field=correlation_gaussian";
                return false;
            }
        }
        for (std::size_t layer = 0; layer < 3; ++layer) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                if (!gaussian_binding_ready(
                        descriptor.dyeCloud[layer][channel],
                        prepared.dyeCloud[layer][channel])) {
                    diagnostic =
                        "ResourceDescriptorMismatch phase=grain_payload field=dye_cloud_gaussian";
                    return false;
                }
            }
        }

        if (recipe.sublayersActive) {
            if (descriptor.densityCurvesLayersHash == 0 ||
                descriptor.densityCurvesLayersHash !=
                    recipe.densityCurvesLayersHash ||
                !prepared.densityLayers.active ||
                prepared.densityLayers.hash !=
                    descriptor.densityCurvesLayersHash) {
                diagnostic =
                    "MissingRequiredResource phase=grain_payload field=density_layers";
                return false;
            }
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const DeviceCurveView& baseCurve =
                    prepared.densityLayers.baseCurvesCmy[channel];
                if (!baseCurve.x || !baseCurve.y || baseCurve.n <= 0 ||
                    baseCurve.domainBegin < 0 ||
                    baseCurve.domainEnd < baseCurve.domainBegin ||
                    baseCurve.domainEnd >= baseCurve.n) {
                    diagnostic =
                        "MissingRequiredResource phase=grain_payload field=base_density_curve";
                    return false;
                }
            }
            for (std::size_t layer = 0; layer < 3; ++layer) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    if (!prepared.densityLayers.curves[layer][channel]) {
                        diagnostic =
                            "MissingRequiredResource phase=grain_payload field=density_layer_curve";
                        return false;
                    }
                }
            }
        } else if (descriptor.densityCurvesLayersHash != 0 ||
                   prepared.densityLayers.active ||
                   prepared.densityLayers.hash != 0) {
            diagnostic =
                "ResourceDescriptorMismatch phase=grain_payload field=inactive_density_layers";
            return false;
        }

        outGrain.active = 1;
        outGrain.sublayersActive = recipe.sublayersActive ? 1 : 0;
        outGrain.nSubLayers = recipe.nSubLayers;
        outGrain.originX = descriptor.renderExtent.x;
        outGrain.originY = descriptor.renderExtent.y;
        outGrain.seedBase = descriptor.seedBase;
        outGrain.seedBaseNext = descriptor.seedBaseNext;
        outGrain.stbn = noise.stbn;
        outGrain.stbnWidth = noise.stbnWidth;
        outGrain.stbnHeight = noise.stbnHeight;
        outGrain.stbnFrames = noise.stbnFrames;
        outGrain.stbnOffsetX =
            stbn_offset(descriptor.sessionSeed, noise.stbnWidth, 0xA5u);
        outGrain.stbnOffsetY =
            stbn_offset(descriptor.sessionSeed, noise.stbnHeight, 0x5Au);
        outGrain.stbnFrame = stbn_frame_index(
            descriptor.frame0,
            noise.stbnFrames,
            descriptor.sessionSeed);
        outGrain.frameIndex = descriptor.frame0;
        outGrain.timeAlpha = descriptor.frameAlpha;
        outGrain.stbnSessionSeed = descriptor.sessionSeed;
        outGrain.clipToken = descriptor.clipToken;
        outGrain.wangTiles = noise.wangTiles;
        outGrain.wangLut = noise.wangLut;
        outGrain.wangWidth = noise.wangWidth;
        outGrain.wangHeight = noise.wangHeight;
        outGrain.wangCount = noise.wangCount;
        outGrain.wangColors = noise.wangColors;
        outGrain.wangCellMm = descriptor.wangCellMm;
        outGrain.breathingPeriodFrames = descriptor.breathingPeriodFrames;
        outGrain.breathingAmplitude = descriptor.breathingAmplitude;
        outGrain.breathingCellUmSmall = descriptor.breathingCellUmSmall;
        outGrain.breathingCellUmLarge = descriptor.breathingCellUmLarge;
        outGrain.breathingMix = descriptor.breathingMix;
        outGrain.breathingDriftUmPerFrame =
            descriptor.breathingDriftUmPerFrame;
        outGrain.breathingDebug = recipe.breathingDebug ? 1 : 0;
        outGrain.debugView = recipe.debugView;
        outGrain.pixelSizeUm = descriptor.pixelSizeUm;
        outGrain.pitchPx = descriptor.pitchPx;
        outGrain.blurSigmaPx = recipe.correlationSigmaPx;
        outGrain.blurDyeCloudsUm = recipe.dyeCloudBlurUm;
        outGrain.microStructure[0] = recipe.microStructure[0];
        outGrain.microStructure[1] = recipe.microStructure[1];
        outGrain.clumpTemporalMix = recipe.clumpTemporalMix;
        outGrain.clumpMorphPeriodFrames =
            descriptor.clumpMorphPeriodFrames;
        outGrain.sizeMixWeightFine = descriptor.effectiveFineWeight;
        outGrain.sizeMixWeight = descriptor.effectiveCoarseWeight;
        outGrain.sizeMixWeightMid = descriptor.effectiveMidWeight;
        outGrain.sizeMixScale = recipe.sizeMixScale;
        outGrain.sizeMixGain = descriptor.sizeMixGain;
        outGrain.amplitude = recipe.amplitude;
        outGrain.chromaMix = recipe.chromaMix;
        outGrain.chromaSharedWeight = recipe.chromaSharedWeight;
        outGrain.chromaIndWeight = recipe.chromaIndependentWeight;
        outGrain.debugScale = descriptor.debugScale;

        for (std::size_t channel = 0; channel < 3; ++channel) {
            outGrain.densityMin[channel] =
                recipe.visualParticleDensityMinCmy[channel];
            outGrain.uniformity[channel] = recipe.uniformityCmy[channel];
            outGrain.densityMax[channel] = descriptor.densityMaxCmy[channel];
            outGrain.nParticles[channel] = descriptor.nParticlesCmy[channel];
            outGrain.odParticle[channel] = descriptor.odParticleCmy[channel];
            outGrain.densityCurveCmy[channel] =
                recipe.sublayersActive
                    ? prepared.densityLayers.baseCurvesCmy[channel]
                    : DeviceCurveView{};
        }
        for (std::size_t layer = 0; layer < 3; ++layer) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                outGrain.densityMinLayers[layer][channel] =
                    descriptor.densityMinLayers[layer][channel];
                outGrain.densityMaxLayers[layer][channel] =
                    descriptor.densityMaxLayers[layer][channel];
                outGrain.nParticlesLayers[layer][channel] =
                    descriptor.nParticlesLayers[layer][channel];
                outGrain.odParticleLayers[layer][channel] =
                    descriptor.odParticleLayers[layer][channel];
                outGrain.densityCurvesLayers[layer][channel] =
                    recipe.sublayersActive
                        ? prepared.densityLayers.curves[layer][channel]
                        : nullptr;
                outKernels.dyeKernel[layer][channel] =
                    prepared.dyeCloud[layer][channel].kernel.weights;
                outKernels.dyeRadius[layer][channel] =
                    prepared.dyeCloud[layer][channel].kernel.radius;
            }
        }

        outKernels.blurKernel = prepared.correlation[0].kernel.weights;
        outKernels.blurRadius = prepared.correlation[0].kernel.radius;
        outKernels.blurKernelMid = prepared.correlation[1].kernel.weights;
        outKernels.blurRadiusMid = prepared.correlation[1].kernel.radius;
        outKernels.blurKernelCoarse =
            prepared.correlation[2].kernel.weights;
        outKernels.blurRadiusCoarse =
            prepared.correlation[2].kernel.radius;
        return true;
    }

} // namespace JuicerCuda
