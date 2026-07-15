#include "Cuda/JuicerCudaFilmFoundationPayloads.h"

#include <cstddef>
#include <cstring>

#include "Hash.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    void hash_effects_extent(
        std::uint64_t& hash,
        const Spektrafilm::FilmJuicerEffectsFrameExtent& extent) {
        hash_value(hash, extent.x);
        hash_value(hash, extent.y);
        hash_value(hash, extent.width);
        hash_value(hash, extent.height);
    }

    std::uint64_t hash_effects_descriptor(
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor& descriptor) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_effects_extent(hash, descriptor.renderExtent);
        hash_effects_extent(hash, descriptor.fullFrameExtent);
        hash_value(hash, descriptor.pixelSizeUm);
        hash_value(hash, descriptor.frame0);
        hash_value(hash, descriptor.frameAlpha);
        hash_value(hash, descriptor.sessionSeed);
        hash_value(hash, descriptor.clipToken);
        hash_value(hash, descriptor.pitchPx);
        hash_value(hash, descriptor.filmDustAmount);
        hash_value(hash, descriptor.filmScratchAmount);
        hash_value(hash, descriptor.gateDustAmount);
        hash_value(hash, descriptor.gateScratchAmount);
        hash_value(hash, descriptor.weaveActive);
        hash_value(hash, descriptor.weaveDxPx);
        hash_value(hash, descriptor.weaveDyPx);
        hash_value(hash, descriptor.weaveCosRot);
        hash_value(hash, descriptor.weaveSinRot);
        hash_value(hash, descriptor.filmActive);
        hash_value(hash, descriptor.gateMaskActive);
        hash_value(hash, descriptor.gateOutputActive);
        hash_value(hash, descriptor.requiresFullFrame);
        hash_value(hash, descriptor.recipeHash);
        return hash;
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
        outGrain.positiveFilm =
            descriptor.capturePolarity ==
                    Spektrafilm::ProfilePolarity::Positive
                ? 1
                : 0;
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
        outGrain.debugView = recipe.debugView;
        outGrain.pixelSizeUm = descriptor.pixelSizeUm;
        outGrain.pitchPx = descriptor.pitchPx;
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
            outGrain.densityLayerAxisFinite[channel] =
                recipe.sublayersActive &&
                        recipe.grainLayerAxisFinite[channel]
                    ? 1
                    : 0;
            if (recipe.sublayersActive) {
                for (std::size_t block = 0; block < 16; ++block) {
                    outGrain.densityLayerAxisBlockPrefixMax[channel][block] =
                        recipe.grainLayerAxisBlockPrefixMax[channel][block];
                }
            }
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

    bool pack_film_juicer_effects_payload(
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor& descriptor,
        GrainPayload& outDefects,
        GateWeavePayload& outWeave,
        std::string& diagnostic) {
        std::memset(&outDefects, 0, sizeof(outDefects));
        std::memset(&outWeave, 0, sizeof(outWeave));
        diagnostic.clear();

        if (!descriptor.filmActive && !descriptor.gateOutputActive) {
            if (descriptor.hash != 0 || descriptor.recipeHash != 0) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=effects_payload field=inactive_descriptor";
                return false;
            }
            return true;
        }
        const bool predicatesMatch =
            descriptor.filmActive ==
                (descriptor.filmDustAmount > 0.0f ||
                 descriptor.filmScratchAmount > 0.0f) &&
            descriptor.gateMaskActive ==
                (descriptor.gateDustAmount > 0.0f ||
                 descriptor.gateScratchAmount > 0.0f) &&
            descriptor.gateOutputActive ==
                (descriptor.weaveActive || descriptor.gateMaskActive) &&
            descriptor.requiresFullFrame == descriptor.gateOutputActive;
        if (descriptor.hash == 0 || descriptor.recipeHash == 0 ||
            descriptor.renderExtent.width <= 0 ||
            descriptor.renderExtent.height <= 0 ||
            descriptor.fullFrameExtent.width <= 0 ||
            descriptor.fullFrameExtent.height <= 0 ||
            !(descriptor.pixelSizeUm > 0.0f) || descriptor.pitchPx <= 0 ||
            descriptor.sessionSeed == 0 || !predicatesMatch ||
            descriptor.hash != hash_effects_descriptor(descriptor)) {
            diagnostic =
                "ResourceDescriptorMismatch phase=effects_payload field=descriptor";
            return false;
        }

        outDefects.originX = descriptor.renderExtent.x;
        outDefects.originY = descriptor.renderExtent.y;
        outDefects.frameIndex = descriptor.frame0;
        outDefects.timeAlpha = descriptor.frameAlpha;
        outDefects.stbnSessionSeed = descriptor.sessionSeed;
        outDefects.clipToken = descriptor.clipToken;
        outDefects.pixelSizeUm = descriptor.pixelSizeUm;
        outDefects.pitchPx = descriptor.pitchPx;
        outDefects.filmDustAmount = descriptor.filmDustAmount;
        outDefects.filmScratchAmount = descriptor.filmScratchAmount;
        outDefects.gateDustAmount = descriptor.gateDustAmount;
        outDefects.gateScratchAmount = descriptor.gateScratchAmount;

        outWeave.active = descriptor.weaveActive ? 1 : 0;
        outWeave.dxPx = descriptor.weaveDxPx;
        outWeave.dyPx = descriptor.weaveDyPx;
        outWeave.cosRot = descriptor.weaveCosRot;
        outWeave.sinRot = descriptor.weaveSinRot;
        return true;
    }

} // namespace JuicerCuda
#endif
