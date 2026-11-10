#include "production_images.h"

#include <bit>
#include <stdexcept>
#include <utility>

#include "ofxImageEffectExt.h"

namespace ofx_test {
    namespace {
        void require(bool condition, const std::string& message) {
            if (!condition)
                throw std::runtime_error(message);
        }
    } // namespace
    std::vector<float> initial_source(const Case& test) {
        const int pitch = test.width * test.components + kProductionPitchPadding;
        std::vector<float> source(static_cast<std::size_t>(pitch) * static_cast<std::size_t>(test.height), kProductionSourceCanary);
        for (int y = 0; y < test.height; ++y) {
            for (int x = 0; x < test.width; ++x) {
                const auto pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    source[pixel + static_cast<std::size_t>(c)] =
                        c == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f
                               : static_cast<float>((x + 3 * y + 7 * c) % 19) / 19.0f;
                }
            }
        }
        return source;
    }

    bool canaries_preserved(const std::vector<float>& initial, const std::vector<float>& actual, const Case& test) {
        const int pitch = test.width * test.components + kProductionPitchPadding;
        for (int y = 0; y < test.height; ++y) {
            for (int x = test.width * test.components; x < pitch; ++x) {
                const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x);
                if (std::bit_cast<std::uint32_t>(actual[index]) != std::bit_cast<std::uint32_t>(initial[index])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool alpha_preserved(const std::vector<float>& source, const std::vector<float>& output, const Case& test) {
        if (test.components != 4) {
            return true;
        }
        const int pitch = test.width * test.components + kProductionPitchPadding;
        for (int y = 0; y < test.height; ++y) {
            for (int x = 0; x < test.width; ++x) {
                const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components) + 3u;
                if (std::bit_cast<std::uint32_t>(source[index]) != std::bit_cast<std::uint32_t>(output[index])) {
                    return false;
                }
            }
        }
        return true;
    }

    struct ParameterValues {
        std::vector<InstanceValue> uiDefault;
        std::vector<InstanceValue> authored;
    };
    void author(InstanceHost& host, OfxImageEffectHandle effect, const char* name, ParameterValues values) {
        require(host.value(effect, name) == values.uiDefault, std::string("unexpected UI default: ") + name);
        host.set_value(effect, name, std::move(values.authored));
    }
    void author_parameters(InstanceHost& host, OfxImageEffectHandle effect, const Case& test) {
        author(host, effect, "Exposure", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "CameraMeteringMethod", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "CameraAutoExposure", {.uiDefault = {1}, .authored = {0}});
        author(host, effect, "CameraFilmFormatPreset", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "CameraFilmFormatMm", {.uiDefault = {35.0}, .authored = {35.0}});
        author(host, effect, "FilmProfileKey", {.uiDefault = {std::string("kodak_portra_400")}, .authored = {std::string("kodak_portra_400")}});
        author(host, effect, "SpectralUpsampling", {.uiDefault = {0}, .authored = {test.reconstruction}});
        author(host, effect, "InputColorSpace", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "InputCctfDecoding", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "InputCompression", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "Hanatos2025AdaptationWindow", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "Hanatos2025AdaptationSurface", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "PrintProfileKey", {.uiDefault = {std::string("kodak_portra_endura")}, .authored = {std::string("kodak_portra_endura")}});
        author(host, effect, "ScanRoute", {.uiDefault = {std::string("NegativePrintScan")}, .authored = {std::string(test.route)}});
        author(host, effect, "PrintExposure", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintPreflash", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "PrintExposureCompensation", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "EnlargerY", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "EnlargerM", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "EnlargerC", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "FilmGammaFactor", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintGammaFactor", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CouplersActive", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "CouplersAmount", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CouplersInhibitionSameLayer", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CouplersInhibitionInterlayer", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CouplersDiffusionSizeUm", {.uiDefault = {20.0}, .authored = {20.0}});
        author(host, effect, "CouplersLangmuirDonorKRgb", {.uiDefault = {1.0, 1.0, 1.0}, .authored = {1.0, 1.0, 1.0}});
        author(host, effect, "CouplersLangmuirReceiverKRgb", {.uiDefault = {1.0, 1.0, 1.0}, .authored = {1.0, 1.0, 1.0}});
        author(host, effect, "CouplersDiffusionTailUm", {.uiDefault = {200.0}, .authored = {200.0}});
        author(host, effect, "CouplersDiffusionTailWeight", {.uiDefault = {0.03}, .authored = {0.03}});
        author(host, effect, "CouplersGammaUseStock", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "CouplersGammaSameLayerRgb", {.uiDefault = {0.341, 0.324, 0.273}, .authored = {0.341, 0.324, 0.273}});
        author(host, effect, "CouplersGammaInterlayerRToGb", {.uiDefault = {0.355, 0.305}, .authored = {0.355, 0.305}});
        author(host, effect, "CouplersGammaInterlayerGToRb", {.uiDefault = {0.154, 0.358}, .authored = {0.154, 0.358}});
        author(host, effect, "CouplersGammaInterlayerBToRg", {.uiDefault = {0.171, 0.225}, .authored = {0.171, 0.225}});
        author(host, effect, "ScannerLensBlurSigmaPx", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "ScannerUnsharpMask", {.uiDefault = {0.7, 0.7}, .authored = {0.0, 0.0}});
        author(host, effect, "ScannerBlackCorrection", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "ScannerWhiteCorrection", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "ScannerBlackLevel", {.uiDefault = {0.01}, .authored = {0.01}});
        author(host, effect, "ScannerWhiteLevel", {.uiDefault = {0.98}, .authored = {0.98}});
        author(host, effect, "ScannerLutResolution", {.uiDefault = {17}, .authored = {17}});
        author(host, effect, "HalationActive", {.uiDefault = {1}, .authored = {0}});
        author(host, effect, "HalationScatterAmount", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "HalationScatterSpatialScale", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "HalationAmount", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "HalationSpatialScale", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "GrainActive", {.uiDefault = {0}, .authored = {test.grain}});
        author(host, effect, "GrainSublayersActive", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "GrainPreset", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "GrainAmplitude", {.uiDefault = {-1.20}, .authored = {0.0}});
        author(host, effect, "GrainBlur", {.uiDefault = {0.50}, .authored = {0.65}});
        author(host, effect, "GrainSharpness", {.uiDefault = {0.5}, .authored = {0.5}});
        author(host, effect, "GrainChroma", {.uiDefault = {0.3}, .authored = {0.0}});
        author(host, effect, "GrainTexture", {.uiDefault = {0.55}, .authored = {0.55}});
        author(host, effect, "GrainParticleAreaUm2", {.uiDefault = {0.25}, .authored = {0.2}});
        author(host, effect, "GrainParticleScaleMaster", {.uiDefault = {1.48}, .authored = {1.48}});
        author(host, effect, "GrainParticleScaleLayersMaster", {.uiDefault = {1.922}, .authored = {1.922}});
        author(host, effect, "GrainDensityMinMaster", {.uiDefault = {0.08}, .authored = {0.08}});
        author(host, effect, "GrainUniformityMaster", {.uiDefault = {0.97}, .authored = {0.97}});
        author(host, effect, "GrainBlurDyeCloudsUm", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "GrainSizeMixWeight", {.uiDefault = {0.226}, .authored = {0.0}});
        author(host, effect, "GrainSizeMixWeightMid", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "GrainSizeMixScale", {.uiDefault = {19.0}, .authored = {1.0}});
        author(host, effect, "GrainMicroStructure", {.uiDefault = {60.0, 170.0}, .authored = {0.2, 30.0}});
        author(host, effect, "GrainDebugView", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "GrainParticleScale", {.uiDefault = {1.48, 1.48, 1.48}, .authored = {0.8, 1.0, 2.0}});
        author(host, effect, "GrainParticleScaleLayers", {.uiDefault = {1.922, 1.922, 1.922}, .authored = {2.5, 1.0, 0.5}});
        author(host, effect, "GrainDensityMin", {.uiDefault = {0.08, 0.08, 0.08}, .authored = {0.07, 0.08, 0.12}});
        author(host, effect, "GrainUniformity", {.uiDefault = {0.97, 0.97, 0.97}, .authored = {0.97, 0.97, 0.99}});
        author(host, effect, "GrainClumpTemporalMix", {.uiDefault = {0.30}, .authored = {0.0}});
        author(host, effect, "GrainClumpMorphPeriodSec", {.uiDefault = {8.0}, .authored = {8.0}});
        author(host, effect, "GateWeaveAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "FilmDustAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "GateDustAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "FilmScratchAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "GateScratchAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "CameraDiffusionEnabled", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "CameraDiffusionFamily", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "CameraDiffusionStrength", {.uiDefault = {0.5}, .authored = {0.5}});
        author(host, effect, "CameraDiffusionSpatialScale", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionHaloWarmth", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "CameraDiffusionCoreIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionCoreSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionHaloIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionHaloSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionBloomIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionBloomSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionEnabled", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "PrintDiffusionFamily", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "PrintDiffusionStrength", {.uiDefault = {0.5}, .authored = {0.5}});
        author(host, effect, "PrintDiffusionSpatialScale", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionHaloWarmth", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "PrintDiffusionCoreIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionCoreSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionHaloIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionHaloSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionBloomIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionBloomSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "GlareActive", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "GlarePercent", {.uiDefault = {0.03}, .authored = {0.03}});
        author(host, effect, "GlareRoughness", {.uiDefault = {0.7}, .authored = {0.7}});
        author(host, effect, "GlareBlurSigmaPx", {.uiDefault = {0.5}, .authored = {0.5}});
        author(host, effect, "PrintShadowCompensationFactor", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "PrintShadowCompensationDensity", {.uiDefault = {1.2}, .authored = {1.2}});
        author(host, effect, "PrintShadowCompensationTransition", {.uiDefault = {0.3}, .authored = {0.3}});
        author(host, effect, "ReferenceIlluminant", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "EnlargerIlluminant", {.uiDefault = {3}, .authored = {3}});
        author(host, effect, "OutputColorSpace", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "OutputCctfEncoding", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "OutputGamutCompression", {.uiDefault = {1}, .authored = {1}});
    }
    void configure(InstanceHost& host, OfxImageEffectHandle effect, const Case& test, const ImageBuffers& device) {
        host.set_effect_property(effect, kOfxImageEffectPropFrameRate, {24.0});
        auto* args = host.action_args();
        host.set_property(args, kOfxPropTime, {test.time});
        host.set_property(args, kOfxImageEffectPropRenderScale, {1.0, 1.0});
        host.set_property(args, kOfxImageEffectPropRenderWindow, {0, 0, test.width, test.height});
        host.set_property(args, kOfxImageEffectPropCudaEnabled, {1});
        host.set_property(args, kOfxImageEffectPropCudaStream, {static_cast<void*>(device.stream)});
        host.set_property(args, kOfxImageEffectPropFieldToRender, {std::string(kOfxImageFieldNone)});
        const int rowBytes = (test.width * test.components + kProductionPitchPadding) * static_cast<int>(sizeof(float));
        for (const auto* clip : {"Source", "Output"}) {
            const bool source = std::string_view(clip) == "Source";
            auto set = [&](const char* key, std::vector<InstanceValue> values) {
                host.set_image_property(effect, clip, key, std::move(values));
            };
            set(kOfxImagePropRowBytes, {rowBytes});
            set(kOfxImagePropPixelAspectRatio, {1.0});
            set(kOfxImageEffectPropComponents, {std::string(test.components == 3 ? kOfxImageComponentRGB : kOfxImageComponentRGBA)});
            set(kOfxImageEffectPropPixelDepth, {std::string(kOfxBitDepthFloat)});
            set(kOfxImageEffectPropPreMultiplication, {std::string(kOfxImageUnPreMultiplied)});
            set(kOfxImagePropBounds, {0, 0, test.width, test.height});
            set(kOfxImagePropRegionOfDefinition, {0, 0, test.width, test.height});
            set(kOfxImagePropField, {std::string(kOfxImageFieldNone)});
            set(kOfxImagePropUniqueIdentifier, {std::string(clip)});
            set(kOfxImageEffectPropRenderScale, {1.0, 1.0});
            set(kOfxImagePropData, {static_cast<void*>(source ? device.source : device.destination)});
            host.set_clip_region(effect, clip, OfxRectD{0.0, 0.0, static_cast<double>(test.width), static_cast<double>(test.height)});
        }
    }

} // namespace ofx_test
