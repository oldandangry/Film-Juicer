#include "Cuda/JuicerCudaFilmPayloads.h"

#include <cmath>
#include <cstddef>
#include <memory>

#include "Hash.h"

namespace {

    void copy_film_floats(float* dst, const float* src, int count) {
        for (int index = 0; index < count; ++index) {
            dst[index] = src[index];
        }
    }

    bool film_curve_ready(const JuicerCuda::DeviceCurveView& curve, int expectedSamples) {
        return curve.x && curve.y && curve.n == expectedSamples &&
               curve.domainBegin >= 0 && curve.domainEnd >= curve.domainBegin &&
               curve.domainEnd < curve.n;
    }

} // namespace

namespace JuicerCuda {

    bool pack_film_payloads(
        const FilmRawRecipe& filmRaw,
        const FilmDevelopRecipe& filmDevelop,
        const DirCouplersRecipe& dirCouplers,
        const DensityBoundsRecipe& densityBounds,
        const FilmPreparedView& prepared,
        const float* autoExposureScaleDevice,
        float routeCorrectionScale,
        FilmPayloadPack& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = FilmPayloadPack{};
        if (filmRaw.finalSensitivityHash == 0 ||
            prepared.finalSensitivityHash != filmRaw.finalSensitivityHash) {
            diagnostic = "ResourceDescriptorMismatch phase=3B field=final_sensitivity";
            return false;
        }
        if (filmDevelop.normalizedDensityCurvesHash == 0 ||
            prepared.normalizedDensityCurvesHash != filmDevelop.normalizedDensityCurvesHash) {
            diagnostic = "ResourceDescriptorMismatch phase=3B field=normalized_density_curves";
            return false;
        }
        if (densityBounds.hash == 0) {
            diagnostic = "ResourceDescriptorMismatch phase=3B field=density_bounds";
            return false;
        }
        const int densitySamples = static_cast<int>(filmDevelop.logExposure.size());
        if (dirCouplers.active) {
            if (dirCouplers.hash == 0 ||
                dirCouplers.precorrectedDensityCurvesHash == 0 ||
                prepared.dirCouplersHash != dirCouplers.hash) {
                diagnostic = "ResourceDescriptorMismatch phase=3D-3 field=dirCouplers";
                return false;
            }
            if (!film_curve_ready(prepared.dirDensB, densitySamples) ||
                !film_curve_ready(prepared.dirDensG, densitySamples) ||
                !film_curve_ready(prepared.dirDensR, densitySamples)) {
                diagnostic = "MissingRequiredResource phase=3D-3 field=dir_density_device_curves";
                return false;
            }
        } else if (prepared.dirCouplersHash != 0) {
            diagnostic = "ResourceDescriptorMismatch phase=3D-3 field=disabled_dir_resources";
            return false;
        }
        if (!film_curve_ready(prepared.finalSensB, 81) ||
            !film_curve_ready(prepared.finalSensG, 81) ||
            !film_curve_ready(prepared.finalSensR, 81)) {
            diagnostic = "MissingRequiredResource phase=3B field=final_sensitivity_device_curves";
            return false;
        }
        if (densitySamples <= 0 ||
            !film_curve_ready(prepared.normalizedDensB, densitySamples) ||
            !film_curve_ready(prepared.normalizedDensG, densitySamples) ||
            !film_curve_ready(prepared.normalizedDensR, densitySamples)) {
            diagnostic = "MissingRequiredResource phase=3B field=normalized_density_device_curves";
            return false;
        }
        if (!prepared.tablesAx || !prepared.tablesAy || !prepared.tablesAz ||
            !prepared.tablesIllum || prepared.tablesK != 81) {
            diagnostic = "MissingRequiredResource phase=3B field=spectral_tables";
            return false;
        }
        if (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025) {
            if (!prepared.hanatosLut || prepared.hanatosN <= 0) {
                diagnostic = "MissingRequiredResource phase=3B field=hanatos_lut";
                return false;
            }
        } else if (!prepared.mallettBasis || prepared.mallettBasisK != 81) {
            diagnostic = "MissingRequiredResource phase=3B field=mallett_basis";
            return false;
        }

        const double manualScale64 =
            std::exp2(static_cast<double>(filmRaw.manualExposureCompensationEv));
        const float manualScale = static_cast<float>(manualScale64);
        if (!std::isfinite(filmRaw.manualExposureCompensationEv) ||
            !std::isfinite(manualScale64) || !std::isfinite(manualScale) ||
            !(manualScale > 0.0f) ||
            !std::isfinite(routeCorrectionScale) ||
            !(routeCorrectionScale > 0.0f)) {
            diagnostic =
                "ResourceDescriptorMismatch phase=3B field=film_exposure_scale";
            return false;
        }
        if (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019 &&
            (!std::isfinite(filmRaw.mallettGreenMidgrayScale) ||
             !(filmRaw.mallettGreenMidgrayScale > 0.0f))) {
            diagnostic =
                "ResourceDescriptorMismatch phase=3B field=mallett_midgray_scale";
            return false;
        }
        for (float gamma : filmDevelop.densityCurveGamma) {
            if (!std::isfinite(gamma) || !(gamma > 0.0f)) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=3B field=density_curve_gamma";
                return false;
            }
        }
        out.filmRaw.inputColorSpaceIndex = filmRaw.inputColorSpace;
        out.filmRaw.applyCctfDecoding = filmRaw.inputCctfDecoding ? 1 : 0;
        out.filmRaw.applyInputChromaticAdapt = prepared.applyInputChromaticAdapt;
        out.filmRaw.spectralUpsamplingMode =
            filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019 ? 1 : 0;
        out.filmRaw.mallettGreenMidgrayScale = filmRaw.mallettGreenMidgrayScale;
        copy_film_floats(out.filmRaw.inputRGBToXYZ, prepared.inputRGBToXYZ, 9);
        copy_film_floats(out.filmRaw.inputXYZAdapt, prepared.inputXYZAdapt, 9);
        copy_film_floats(out.filmRaw.refIllumWhiteXYZ, prepared.refIllumWhiteXYZ, 3);

        out.filmExposure.manualExposureScale = manualScale;
        out.filmExposure.routeCorrectionScale = routeCorrectionScale;
        out.filmExposure.exposureScaleDevice = autoExposureScaleDevice;
        out.filmExposure.sensB = prepared.finalSensB;
        out.filmExposure.sensG = prepared.finalSensG;
        out.filmExposure.sensR = prepared.finalSensR;
        out.filmExposure.tablesAx = prepared.tablesAx;
        out.filmExposure.tablesAy = prepared.tablesAy;
        out.filmExposure.tablesAz = prepared.tablesAz;
        out.filmExposure.tablesIllum = prepared.tablesIllum;
        out.filmExposure.tablesK = prepared.tablesK;
        copy_film_floats(out.filmExposure.spdSInv, prepared.spdSInv, 9);
        if (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025) {
            out.filmExposure.hanatosLut = prepared.hanatosLut;
            out.filmExposure.hanatosN = prepared.hanatosN;
            out.filmExposure.hanatosLutIntegrated = prepared.hanatosLutIntegrated;
            out.filmExposure.hanatosNIntegrated = prepared.hanatosNIntegrated;
        } else {
            out.filmExposure.mallettBasis = prepared.mallettBasis;
            out.filmExposure.mallettBasisK = prepared.mallettBasisK;
        }

        out.filmDevelop.gammaFactorB = filmDevelop.densityCurveGamma[2];
        out.filmDevelop.gammaFactorG = filmDevelop.densityCurveGamma[1];
        out.filmDevelop.gammaFactorR = filmDevelop.densityCurveGamma[0];
        out.filmDevelop.densB = prepared.normalizedDensB;
        out.filmDevelop.densG = prepared.normalizedDensG;
        out.filmDevelop.densR = prepared.normalizedDensR;
        if (dirCouplers.active) {
            out.filmDevelop.dirPrecorrected = 1;
            out.filmDevelop.dir.active = 1;
            out.filmDevelop.dir.positive =
                dirCouplers.polarity == Spektrafilm::ProfilePolarity::Positive ? 1 : 0;
            for (int donorBgr = 0; donorBgr < 3; ++donorBgr) {
                for (int receiverBgr = 0; receiverBgr < 3; ++receiverBgr) {
                    out.filmDevelop.dir.M[donorBgr * 3 + receiverBgr] =
                        dirCouplers.matrixRgb[2 - donorBgr][2 - receiverBgr];
                }
                out.filmDevelop.dir.dMax[donorBgr] =
                    dirCouplers.densityMaxRgb[2 - donorBgr];
            }
            out.filmDevelop.dirDensB = prepared.dirDensB;
            out.filmDevelop.dirDensG = prepared.dirDensG;
            out.filmDevelop.dirDensR = prepared.dirDensR;
        }
        out.densityBoundsHash = densityBounds.hash;
        return true;
    }

} // namespace JuicerCuda

namespace {

    template <typename Payload>
    void reset_payload_to_defaults(Payload& payload) noexcept {
        // Clang cannot synthesize assignment for payloads containing arrays
        // of restrict-qualified pointers, so reconstruct the aggregate in place.
        std::destroy_at(std::addressof(payload));
        std::construct_at(std::addressof(payload));
    }

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
        const JuicerCuda::VisualGrainPreparedGaussianView& prepared) {
        if (descriptor.radius <= 0) {
            return descriptor.hash == 0 && !prepared.active &&
                   !prepared.weights && prepared.radius == 0 &&
                   prepared.descriptorHash == 0;
        }
        return descriptor.hash != 0 && prepared.active &&
               prepared.descriptorHash == descriptor.hash &&
               prepared.weights &&
               prepared.radius == descriptor.radius &&
               prepared.sigma == descriptor.sigmaPx;
    }

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }


} // namespace

namespace JuicerCuda {

    bool pack_visual_grain_payload(
        const Spektrafilm::VisualGrainRecipe& recipe,
        const PreparedVisualGrainView& prepared,
        GrainPayload& outGrain,
        GrainKernelPayload& outKernels,
        std::string& diagnostic) {
        reset_payload_to_defaults(outGrain);
        reset_payload_to_defaults(outKernels);
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
                    prepared.dyeCloud[layer][channel].weights;
                outKernels.dyeRadius[layer][channel] =
                    prepared.dyeCloud[layer][channel].radius;
            }
        }

        outKernels.blurKernel = prepared.correlation[0].weights;
        outKernels.blurRadius = prepared.correlation[0].radius;
        outKernels.blurKernelMid = prepared.correlation[1].weights;
        outKernels.blurRadiusMid = prepared.correlation[1].radius;
        outKernels.blurKernelCoarse =
            prepared.correlation[2].weights;
        outKernels.blurRadiusCoarse =
            prepared.correlation[2].radius;
        return true;
    }

    bool pack_film_juicer_effects_payload(
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor& descriptor,
        FilmDefectsPayload& outDefects,
        GateWeavePayload& outWeave,
        std::string& diagnostic) {
        outDefects = {};
        outWeave = {};
        diagnostic.clear();
        if (!Spektrafilm::validate_film_juicer_effects_frame_descriptor(descriptor)) {
            diagnostic = "ResourceDescriptorMismatch phase=effects_payload field=descriptor";
            return false;
        }
        if (!descriptor.filmActive && !descriptor.gateOutputActive) {
            return true;
        }
        outDefects.filmDust.cellWidthMm = descriptor.filmDust.cellWidthMm;
        outDefects.filmDust.cellHeightMm = descriptor.filmDust.cellHeightMm;
        outDefects.filmDust.slotProbability = descriptor.filmDust.slotProbability;
        outDefects.filmDust.softnessMinMm = descriptor.filmDust.softnessMinMm;
        outDefects.filmDust.softnessMaxMm = descriptor.filmDust.softnessMaxMm;
        outDefects.filmDust.softnessSizeCapFraction = descriptor.filmDust.softnessSizeCapFraction;
        outDefects.filmDust.supportXMm = descriptor.filmDust.supportXMm;
        outDefects.filmDust.supportYMm = descriptor.filmDust.supportYMm;
        outDefects.filmDust.fiberFraction = descriptor.filmDust.fiberFraction;
        outDefects.filmDust.fiberDriftFraction = descriptor.filmDust.fiberDriftFraction;
        outDefects.filmDust.fiberFirstKnotMin = descriptor.filmDust.fiberFirstKnotMin;
        outDefects.filmDust.fiberFirstKnotMax = descriptor.filmDust.fiberFirstKnotMax;
        outDefects.filmDust.fiberSecondKnotMin = descriptor.filmDust.fiberSecondKnotMin;
        outDefects.filmDust.fiberSecondKnotMax = descriptor.filmDust.fiberSecondKnotMax;
        outDefects.filmDust.fiberInteriorWidthMinFraction = descriptor.filmDust.fiberInteriorWidthMinFraction;
        outDefects.filmDust.fiberInteriorWidthMaxFraction = descriptor.filmDust.fiberInteriorWidthMaxFraction;
        outDefects.filmDust.diameterMinMm = descriptor.filmDust.diameterMinMm;
        outDefects.filmDust.diameterBulkMaxMm = descriptor.filmDust.diameterBulkMaxMm;
        outDefects.filmDust.diameterMaxMm = descriptor.filmDust.diameterMaxMm;
        outDefects.filmDust.diameterTailFraction = descriptor.filmDust.diameterTailFraction;
        outDefects.filmDust.fiberLengthMinMm = descriptor.filmDust.fiberLengthMinMm;
        outDefects.filmDust.fiberLengthMaxMm = descriptor.filmDust.fiberLengthMaxMm;
        outDefects.filmDust.fiberWidthMinMm = descriptor.filmDust.fiberWidthMinMm;
        outDefects.filmDust.fiberWidthMaxMm = descriptor.filmDust.fiberWidthMaxMm;
        outDefects.filmDust.opacityFaintCumulative = descriptor.filmDust.opacityFaintCumulative;
        outDefects.filmDust.opacityIntermediateCumulative = descriptor.filmDust.opacityIntermediateCumulative;
        outDefects.filmDust.compactOpacityMin = descriptor.filmDust.compactOpacityMin;
        outDefects.filmDust.compactOpacityFaintEnd = descriptor.filmDust.compactOpacityFaintEnd;
        outDefects.filmDust.compactOpacityIntermediateEnd = descriptor.filmDust.compactOpacityIntermediateEnd;
        outDefects.filmDust.compactOpacityMax = descriptor.filmDust.compactOpacityMax;
        outDefects.filmDust.fiberOpacityMin = descriptor.filmDust.fiberOpacityMin;
        outDefects.filmDust.fiberOpacityFaintEnd = descriptor.filmDust.fiberOpacityFaintEnd;
        outDefects.filmDust.fiberOpacityIntermediateEnd = descriptor.filmDust.fiberOpacityIntermediateEnd;
        outDefects.filmDust.fiberOpacityMax = descriptor.filmDust.fiberOpacityMax;
        outDefects.filmDust.compactDominantAspectMin = descriptor.filmDust.compactDominantAspectMin;
        outDefects.filmDust.compactDominantAspectMax = descriptor.filmDust.compactDominantAspectMax;
        outDefects.filmDust.compactSubsidiaryScaleMin = descriptor.filmDust.compactSubsidiaryScaleMin;
        outDefects.filmDust.compactSubsidiaryScaleMax = descriptor.filmDust.compactSubsidiaryScaleMax;
        outDefects.filmDust.compactSubsidiaryAspectMin = descriptor.filmDust.compactSubsidiaryAspectMin;
        outDefects.filmDust.compactSubsidiaryAspectMax = descriptor.filmDust.compactSubsidiaryAspectMax;
        outDefects.filmDust.compactSubsidiaryOffsetMax = descriptor.filmDust.compactSubsidiaryOffsetMax;
        outDefects.filmDust.compactSubsidiaryAngleMaxRadians = descriptor.filmDust.compactSubsidiaryAngleMaxRadians;
        outDefects.filmScratch.cellWidthMm = descriptor.filmScratch.cellWidthMm;
        outDefects.filmScratch.cellHeightMm = descriptor.filmScratch.cellHeightMm;
        outDefects.filmScratch.slotProbability = descriptor.filmScratch.slotProbability;
        outDefects.filmScratch.softnessMinMm = descriptor.filmScratch.softnessMinMm;
        outDefects.filmScratch.softnessMaxMm = descriptor.filmScratch.softnessMaxMm;
        outDefects.filmScratch.softnessSizeCapFraction = descriptor.filmScratch.softnessSizeCapFraction;
        outDefects.filmScratch.supportXMm = descriptor.filmScratch.supportXMm;
        outDefects.filmScratch.supportYMm = descriptor.filmScratch.supportYMm;
        outDefects.filmScratch.lengthMinMm = descriptor.filmScratch.lengthMinMm;
        outDefects.filmScratch.lengthBulkMaxMm = descriptor.filmScratch.lengthBulkMaxMm;
        outDefects.filmScratch.lengthMaxMm = descriptor.filmScratch.lengthMaxMm;
        outDefects.filmScratch.lengthTailFraction = descriptor.filmScratch.lengthTailFraction;
        outDefects.filmScratch.widthMinMm = descriptor.filmScratch.widthMinMm;
        outDefects.filmScratch.widthBulkMaxMm = descriptor.filmScratch.widthBulkMaxMm;
        outDefects.filmScratch.widthMaxMm = descriptor.filmScratch.widthMaxMm;
        outDefects.filmScratch.widthTailFraction = descriptor.filmScratch.widthTailFraction;
        outDefects.filmScratch.driftFraction = descriptor.filmScratch.driftFraction;
        outDefects.filmScratch.firstKnotMin = descriptor.filmScratch.firstKnotMin;
        outDefects.filmScratch.firstKnotMax = descriptor.filmScratch.firstKnotMax;
        outDefects.filmScratch.secondKnotMin = descriptor.filmScratch.secondKnotMin;
        outDefects.filmScratch.secondKnotMax = descriptor.filmScratch.secondKnotMax;
        outDefects.filmScratch.interiorWidthMinFraction = descriptor.filmScratch.interiorWidthMinFraction;
        outDefects.filmScratch.interiorWidthMaxFraction = descriptor.filmScratch.interiorWidthMaxFraction;
        outDefects.filmScratch.interiorDepthMinFraction = descriptor.filmScratch.interiorDepthMinFraction;
        outDefects.filmScratch.interiorDepthMaxFraction = descriptor.filmScratch.interiorDepthMaxFraction;
        outDefects.filmScratch.endpointAbruptProbability = descriptor.filmScratch.endpointAbruptProbability;
        outDefects.filmScratch.interruptionProbability = descriptor.filmScratch.interruptionProbability;
        outDefects.filmScratch.gapCenterMin = descriptor.filmScratch.gapCenterMin;
        outDefects.filmScratch.gapCenterMax = descriptor.filmScratch.gapCenterMax;
        outDefects.filmScratch.gapSpanMin = descriptor.filmScratch.gapSpanMin;
        outDefects.filmScratch.gapSpanMax = descriptor.filmScratch.gapSpanMax;
        outDefects.filmScratch.scuffProbability = descriptor.filmScratch.scuffProbability;
        outDefects.filmScratch.scuffLengthMaxMm = descriptor.filmScratch.scuffLengthMaxMm;
        outDefects.filmScratch.scuffAngleMaxRadians = descriptor.filmScratch.scuffAngleMaxRadians;
        outDefects.filmScratch.strengthMin = descriptor.filmScratch.strengthMin;
        outDefects.filmScratch.strengthMax = descriptor.filmScratch.strengthMax;
        outDefects.gateDust.cellWidthMm = descriptor.gateDust.cellWidthMm;
        outDefects.gateDust.cellHeightMm = descriptor.gateDust.cellHeightMm;
        outDefects.gateDust.slotProbability = descriptor.gateDust.slotProbability;
        outDefects.gateDust.softnessMinMm = descriptor.gateDust.softnessMinMm;
        outDefects.gateDust.softnessMaxMm = descriptor.gateDust.softnessMaxMm;
        outDefects.gateDust.softnessSizeCapFraction = descriptor.gateDust.softnessSizeCapFraction;
        outDefects.gateDust.supportXMm = descriptor.gateDust.supportXMm;
        outDefects.gateDust.supportYMm = descriptor.gateDust.supportYMm;
        outDefects.gateDust.fiberFraction = descriptor.gateDust.fiberFraction;
        outDefects.gateDust.fiberDriftFraction = descriptor.gateDust.fiberDriftFraction;
        outDefects.gateDust.fiberFirstKnotMin = descriptor.gateDust.fiberFirstKnotMin;
        outDefects.gateDust.fiberFirstKnotMax = descriptor.gateDust.fiberFirstKnotMax;
        outDefects.gateDust.fiberSecondKnotMin = descriptor.gateDust.fiberSecondKnotMin;
        outDefects.gateDust.fiberSecondKnotMax = descriptor.gateDust.fiberSecondKnotMax;
        outDefects.gateDust.fiberInteriorWidthMinFraction = descriptor.gateDust.fiberInteriorWidthMinFraction;
        outDefects.gateDust.fiberInteriorWidthMaxFraction = descriptor.gateDust.fiberInteriorWidthMaxFraction;
        outDefects.gateDust.diameterMinMm = descriptor.gateDust.diameterMinMm;
        outDefects.gateDust.diameterBulkMaxMm = descriptor.gateDust.diameterBulkMaxMm;
        outDefects.gateDust.diameterMaxMm = descriptor.gateDust.diameterMaxMm;
        outDefects.gateDust.diameterTailFraction = descriptor.gateDust.diameterTailFraction;
        outDefects.gateDust.fiberLengthMinMm = descriptor.gateDust.fiberLengthMinMm;
        outDefects.gateDust.fiberLengthMaxMm = descriptor.gateDust.fiberLengthMaxMm;
        outDefects.gateDust.fiberWidthMinMm = descriptor.gateDust.fiberWidthMinMm;
        outDefects.gateDust.fiberWidthMaxMm = descriptor.gateDust.fiberWidthMaxMm;
        outDefects.gateDust.opacityFaintCumulative = descriptor.gateDust.opacityFaintCumulative;
        outDefects.gateDust.opacityIntermediateCumulative = descriptor.gateDust.opacityIntermediateCumulative;
        outDefects.gateDust.compactOpacityMin = descriptor.gateDust.compactOpacityMin;
        outDefects.gateDust.compactOpacityFaintEnd = descriptor.gateDust.compactOpacityFaintEnd;
        outDefects.gateDust.compactOpacityIntermediateEnd = descriptor.gateDust.compactOpacityIntermediateEnd;
        outDefects.gateDust.compactOpacityMax = descriptor.gateDust.compactOpacityMax;
        outDefects.gateDust.fiberOpacityMin = descriptor.gateDust.fiberOpacityMin;
        outDefects.gateDust.fiberOpacityFaintEnd = descriptor.gateDust.fiberOpacityFaintEnd;
        outDefects.gateDust.fiberOpacityIntermediateEnd = descriptor.gateDust.fiberOpacityIntermediateEnd;
        outDefects.gateDust.fiberOpacityMax = descriptor.gateDust.fiberOpacityMax;
        outDefects.gateDust.compactDominantAspectMin = descriptor.gateDust.compactDominantAspectMin;
        outDefects.gateDust.compactDominantAspectMax = descriptor.gateDust.compactDominantAspectMax;
        outDefects.gateDust.compactSubsidiaryScaleMin = descriptor.gateDust.compactSubsidiaryScaleMin;
        outDefects.gateDust.compactSubsidiaryScaleMax = descriptor.gateDust.compactSubsidiaryScaleMax;
        outDefects.gateDust.compactSubsidiaryAspectMin = descriptor.gateDust.compactSubsidiaryAspectMin;
        outDefects.gateDust.compactSubsidiaryAspectMax = descriptor.gateDust.compactSubsidiaryAspectMax;
        outDefects.gateDust.compactSubsidiaryOffsetMax = descriptor.gateDust.compactSubsidiaryOffsetMax;
        outDefects.gateDust.compactSubsidiaryAngleMaxRadians = descriptor.gateDust.compactSubsidiaryAngleMaxRadians;
        outDefects.gateScratch.cellWidthMm = descriptor.gateScratch.cellWidthMm;
        outDefects.gateScratch.cellHeightMm = descriptor.gateScratch.cellHeightMm;
        outDefects.gateScratch.slotProbability = descriptor.gateScratch.slotProbability;
        outDefects.gateScratch.softnessMinMm = descriptor.gateScratch.softnessMinMm;
        outDefects.gateScratch.softnessMaxMm = descriptor.gateScratch.softnessMaxMm;
        outDefects.gateScratch.softnessSizeCapFraction = descriptor.gateScratch.softnessSizeCapFraction;
        outDefects.gateScratch.supportXMm = descriptor.gateScratch.supportXMm;
        outDefects.gateScratch.supportYMm = descriptor.gateScratch.supportYMm;
        outDefects.gateScratch.lengthMinMm = descriptor.gateScratch.lengthMinMm;
        outDefects.gateScratch.lengthBulkMaxMm = descriptor.gateScratch.lengthBulkMaxMm;
        outDefects.gateScratch.lengthMaxMm = descriptor.gateScratch.lengthMaxMm;
        outDefects.gateScratch.lengthTailFraction = descriptor.gateScratch.lengthTailFraction;
        outDefects.gateScratch.widthMinMm = descriptor.gateScratch.widthMinMm;
        outDefects.gateScratch.widthBulkMaxMm = descriptor.gateScratch.widthBulkMaxMm;
        outDefects.gateScratch.widthMaxMm = descriptor.gateScratch.widthMaxMm;
        outDefects.gateScratch.widthTailFraction = descriptor.gateScratch.widthTailFraction;
        outDefects.gateScratch.driftFraction = descriptor.gateScratch.driftFraction;
        outDefects.gateScratch.firstKnotMin = descriptor.gateScratch.firstKnotMin;
        outDefects.gateScratch.firstKnotMax = descriptor.gateScratch.firstKnotMax;
        outDefects.gateScratch.secondKnotMin = descriptor.gateScratch.secondKnotMin;
        outDefects.gateScratch.secondKnotMax = descriptor.gateScratch.secondKnotMax;
        outDefects.gateScratch.interiorWidthMinFraction = descriptor.gateScratch.interiorWidthMinFraction;
        outDefects.gateScratch.interiorWidthMaxFraction = descriptor.gateScratch.interiorWidthMaxFraction;
        outDefects.gateScratch.interiorDepthMinFraction = descriptor.gateScratch.interiorDepthMinFraction;
        outDefects.gateScratch.interiorDepthMaxFraction = descriptor.gateScratch.interiorDepthMaxFraction;
        outDefects.gateScratch.endpointAbruptProbability = descriptor.gateScratch.endpointAbruptProbability;
        outDefects.gateScratch.interruptionProbability = descriptor.gateScratch.interruptionProbability;
        outDefects.gateScratch.gapCenterMin = descriptor.gateScratch.gapCenterMin;
        outDefects.gateScratch.gapCenterMax = descriptor.gateScratch.gapCenterMax;
        outDefects.gateScratch.gapSpanMin = descriptor.gateScratch.gapSpanMin;
        outDefects.gateScratch.gapSpanMax = descriptor.gateScratch.gapSpanMax;
        outDefects.gateScratch.scuffProbability = descriptor.gateScratch.scuffProbability;
        outDefects.gateScratch.scuffLengthMaxMm = descriptor.gateScratch.scuffLengthMaxMm;
        outDefects.gateScratch.scuffAngleMaxRadians = descriptor.gateScratch.scuffAngleMaxRadians;
        outDefects.gateScratch.strengthMin = descriptor.gateScratch.strengthMin;
        outDefects.gateScratch.strengthMax = descriptor.gateScratch.strengthMax;
        for (int i = 0; i < 4; ++i) {
            const auto& o = descriptor.origins[static_cast<std::size_t>(i)];
            outDefects.origins[i] = {o.cellX, o.cellY, o.localXMm, o.localYMm};
        }
        outDefects.sampleStepXMm = descriptor.sampleStepXMm;
        outDefects.sampleStepYMm = descriptor.sampleStepYMm;
        outDefects.roiOffsetX = descriptor.roiOffsetX;
        outDefects.roiOffsetY = descriptor.roiOffsetY;
        outDefects.sessionSeed = descriptor.sessionSeed;
        outDefects.clipToken = descriptor.clipToken;
        outWeave.active = descriptor.weaveActive ? 1 : 0;
        outWeave.dxPx = descriptor.weaveDxPx;
        outWeave.dyPx = descriptor.weaveDyPx;
        outWeave.cosRot = descriptor.weaveCosRot;
        outWeave.sinRot = descriptor.weaveSinRot;
        return true;
    }
} // namespace JuicerCuda
