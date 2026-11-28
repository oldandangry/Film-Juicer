#include "FilmEffectsFrameDescriptors.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include "Hash.h"
#include "Cuda/JuicerCudaFilmPayloads.h"


namespace {

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    Spektrafilm::LegacyGrainScratchShape select_scratch_shape(
        bool sublayers,
        bool sharedChroma) {
        if (sublayers) {
            return sharedChroma
                       ? Spektrafilm::LegacyGrainScratchShape::StreamedLayersShared
                       : Spektrafilm::LegacyGrainScratchShape::StreamedLayers;
        }
        return sharedChroma
                   ? Spektrafilm::LegacyGrainScratchShape::StreamedShared
                   : Spektrafilm::LegacyGrainScratchShape::Streamed;
    }

    template <typename Extent>
    void hash_extent(std::uint64_t& hash, const Extent& extent) {
        hash_value(hash, extent.x);
        hash_value(hash, extent.y);
        hash_value(hash, extent.width);
        hash_value(hash, extent.height);
    }

    std::uint64_t hash_descriptor(
        const Spektrafilm::LegacyGrainFrameDescriptor& descriptor) {
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

    bool finish_legacy_grain_descriptor(LegacyGrainFrameDescriptor& descriptor) {
        descriptor.hash = hash_descriptor(descriptor);
        return descriptor.hash != 0;
    }

    LegacyGrainScratchShape legacy_grain_scratch_shape(const FjGrainCoefficients& coefficients) noexcept {
        if ((coefficients.flags & FJ_GRAIN_ACTIVE) == 0) {
            return LegacyGrainScratchShape::None;
        }
        const bool sharedChroma =
            (coefficients.debug_view == FJ_GRAIN_DEBUG_OFF || coefficients.debug_view == FJ_GRAIN_DEBUG_DELTA_MIX) &&
            coefficients.chroma_mix < 0.999f && coefficients.chroma_shared_weight > 0.0f;
        return select_scratch_shape((coefficients.flags & FJ_GRAIN_SUBLAYERS) != 0, sharedChroma);
    }

} // namespace Spektrafilm
