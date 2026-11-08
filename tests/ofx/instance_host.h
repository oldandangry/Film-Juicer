#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "ofxImageEffect.h"
#include "ofxParam.h"

namespace ofx_test {
    enum class InstanceParamKind : std::uint8_t {
        Double,
        Boolean,
        Choice,
        StringChoice,
        Integer,
        Double2D,
        Double3D,
        PushButton
    };

    struct InstanceParamDefinition {
        std::string_view name;
        InstanceParamKind kind;
        bool operator==(const InstanceParamDefinition&) const = default;
    };

    // Frozen from the independent C++ constructor, in handle acquisition order.
    inline constexpr std::array<InstanceParamDefinition, 113> kInstanceParameters{{{"Exposure", InstanceParamKind::Double},
                                                                                   {"CameraAutoExposure", InstanceParamKind::Boolean},
                                                                                   {"CameraFilmFormatPreset", InstanceParamKind::Choice},
                                                                                   {"CameraFilmFormatMm", InstanceParamKind::Double},
                                                                                   {"CameraMeteringMethod", InstanceParamKind::Choice},
                                                                                   {"FilmProfileKey", InstanceParamKind::StringChoice},
                                                                                   {"SpectralUpsampling", InstanceParamKind::Choice},
                                                                                   {"PrintProfileKey", InstanceParamKind::StringChoice},
                                                                                   {"ReferenceIlluminant", InstanceParamKind::Choice},
                                                                                   {"EnlargerIlluminant", InstanceParamKind::Choice},
                                                                                   {"InputColorSpace", InstanceParamKind::Choice},
                                                                                   {"InputCctfDecoding", InstanceParamKind::Boolean},
                                                                                   {"InputCompression", InstanceParamKind::Boolean},
                                                                                   {"Hanatos2025AdaptationWindow", InstanceParamKind::Boolean},
                                                                                   {"Hanatos2025AdaptationSurface", InstanceParamKind::Boolean},
                                                                                   {"ScanRoute", InstanceParamKind::StringChoice},
                                                                                   {"FilmGammaFactor", InstanceParamKind::Double},
                                                                                   {"PrintGammaFactor", InstanceParamKind::Double},
                                                                                   {"OutputColorSpace", InstanceParamKind::Choice},
                                                                                   {"OutputCctfEncoding", InstanceParamKind::Boolean},
                                                                                   {"OutputGamutCompression", InstanceParamKind::Boolean},
                                                                                   {"CouplersActive", InstanceParamKind::Boolean},
                                                                                   {"CouplersAmount", InstanceParamKind::Double},
                                                                                   {"CouplersInhibitionSameLayer", InstanceParamKind::Double},
                                                                                   {"CouplersInhibitionInterlayer", InstanceParamKind::Double},
                                                                                   {"CouplersDiffusionSizeUm", InstanceParamKind::Double},
                                                                                   {"CouplersLangmuirDonorKRgb", InstanceParamKind::Double3D},
                                                                                   {"CouplersLangmuirReceiverKRgb", InstanceParamKind::Double3D},
                                                                                   {"CouplersDiffusionTailUm", InstanceParamKind::Double},
                                                                                   {"CouplersDiffusionTailWeight", InstanceParamKind::Double},
                                                                                   {"CouplersGammaUseStock", InstanceParamKind::Boolean},
                                                                                   {"CouplersGammaSameLayerRgb", InstanceParamKind::Double3D},
                                                                                   {"CouplersGammaInterlayerRToGb", InstanceParamKind::Double2D},
                                                                                   {"CouplersGammaInterlayerGToRb", InstanceParamKind::Double2D},
                                                                                   {"CouplersGammaInterlayerBToRg", InstanceParamKind::Double2D},
                                                                                   {"ScannerLensBlurSigmaPx", InstanceParamKind::Double},
                                                                                   {"ScannerUnsharpMask", InstanceParamKind::Double2D},
                                                                                   {"ScannerBlackCorrection", InstanceParamKind::Boolean},
                                                                                   {"ScannerWhiteCorrection", InstanceParamKind::Boolean},
                                                                                   {"ScannerBlackLevel", InstanceParamKind::Double},
                                                                                   {"ScannerWhiteLevel", InstanceParamKind::Double},
                                                                                   {"ScannerLutResolution", InstanceParamKind::Integer},
                                                                                   {"PrintExposure", InstanceParamKind::Double},
                                                                                   {"PrintPreflash", InstanceParamKind::Double},
                                                                                   {"PrintExposureCompensation", InstanceParamKind::Boolean},
                                                                                   {"EnlargerY", InstanceParamKind::Double},
                                                                                   {"EnlargerM", InstanceParamKind::Double},
                                                                                   {"EnlargerC", InstanceParamKind::Double},
                                                                                   {"HalationActive", InstanceParamKind::Boolean},
                                                                                   {"HalationScatterAmount", InstanceParamKind::Double},
                                                                                   {"HalationScatterSpatialScale", InstanceParamKind::Double},
                                                                                   {"HalationAmount", InstanceParamKind::Double},
                                                                                   {"HalationSpatialScale", InstanceParamKind::Double},
                                                                                   {"GrainActive", InstanceParamKind::Boolean},
                                                                                   {"GrainSublayersActive", InstanceParamKind::Boolean},
                                                                                   {"GrainPreset", InstanceParamKind::Choice},
                                                                                   {"GrainParticleAreaUm2", InstanceParamKind::Double},
                                                                                   {"GrainAmplitude", InstanceParamKind::Double},
                                                                                   {"GrainSharpness", InstanceParamKind::Double},
                                                                                   {"GrainChroma", InstanceParamKind::Double},
                                                                                   {"GrainTexture", InstanceParamKind::Double},
                                                                                   {"GrainParticleScaleMaster", InstanceParamKind::Double},
                                                                                   {"GrainParticleScaleLayersMaster", InstanceParamKind::Double},
                                                                                   {"GrainDensityMinMaster", InstanceParamKind::Double},
                                                                                   {"GrainUniformityMaster", InstanceParamKind::Double},
                                                                                   {"GrainParticleScale", InstanceParamKind::Double3D},
                                                                                   {"GrainParticleScaleLayers", InstanceParamKind::Double3D},
                                                                                   {"GrainDensityMin", InstanceParamKind::Double3D},
                                                                                   {"GrainUniformity", InstanceParamKind::Double3D},
                                                                                   {"GrainBlur", InstanceParamKind::Double},
                                                                                   {"GrainBlurDyeCloudsUm", InstanceParamKind::Double},
                                                                                   {"GrainSizeMixWeight", InstanceParamKind::Double},
                                                                                   {"GrainSizeMixWeightMid", InstanceParamKind::Double},
                                                                                   {"GrainSizeMixScale", InstanceParamKind::Double},
                                                                                   {"GrainClumpTemporalMix", InstanceParamKind::Double},
                                                                                   {"GrainClumpMorphPeriodSec", InstanceParamKind::Double},
                                                                                   {"GrainDebugView", InstanceParamKind::Choice},
                                                                                   {"GrainMicroStructure", InstanceParamKind::Double2D},
                                                                                   {"GrainResetAdvanced", InstanceParamKind::PushButton},
                                                                                   {"GateWeaveAmount", InstanceParamKind::Double},
                                                                                   {"FilmDustAmount", InstanceParamKind::Double},
                                                                                   {"GateDustAmount", InstanceParamKind::Double},
                                                                                   {"FilmScratchAmount", InstanceParamKind::Double},
                                                                                   {"GateScratchAmount", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionEnabled", InstanceParamKind::Boolean},
                                                                                   {"CameraDiffusionFamily", InstanceParamKind::Choice},
                                                                                   {"CameraDiffusionStrength", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionSpatialScale", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionHaloWarmth", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionCoreIntensity", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionCoreSize", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionHaloIntensity", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionHaloSize", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionBloomIntensity", InstanceParamKind::Double},
                                                                                   {"CameraDiffusionBloomSize", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionEnabled", InstanceParamKind::Boolean},
                                                                                   {"PrintDiffusionFamily", InstanceParamKind::Choice},
                                                                                   {"PrintDiffusionStrength", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionSpatialScale", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionHaloWarmth", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionCoreIntensity", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionCoreSize", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionHaloIntensity", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionHaloSize", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionBloomIntensity", InstanceParamKind::Double},
                                                                                   {"PrintDiffusionBloomSize", InstanceParamKind::Double},
                                                                                   {"GlareActive", InstanceParamKind::Boolean},
                                                                                   {"GlarePercent", InstanceParamKind::Double},
                                                                                   {"GlareRoughness", InstanceParamKind::Double},
                                                                                   {"GlareBlurSigmaPx", InstanceParamKind::Double},
                                                                                   {"PrintShadowCompensationFactor", InstanceParamKind::Double},
                                                                                   {"PrintShadowCompensationDensity", InstanceParamKind::Double},
                                                                                   {"PrintShadowCompensationTransition", InstanceParamKind::Double}}};

    inline constexpr std::array<std::string_view, 2> kInstanceClips{{"Source", "Output"}};
    using InstanceValue = std::variant<int, double, std::string, void*>;
    struct InstanceCall {
        std::string operation;
        std::string target;
        std::string property;
        int index = 0;
        std::vector<InstanceValue> values;
        std::string scope;
        OfxStatus status = kOfxStatOK;
        bool operator==(const InstanceCall&) const = default;
    };

    struct InstanceChange {
        const char* reason;
        const char* type;
        const char* name;
    };

    // Each host owns its descriptor, argument bags and all effects until destruction.
    class InstanceHost {
    public:
        InstanceHost();
        ~InstanceHost();
        InstanceHost(const InstanceHost&) = delete;
        InstanceHost& operator=(const InstanceHost&) = delete;
        OfxHost* binding() noexcept;
        OfxImageEffectHandle descriptor_handle() noexcept;
        OfxPropertySetHandle context_args() noexcept;
        OfxImageEffectHandle create_effect();
        OfxPropertySetHandle change_args(const InstanceChange& change, double time = 0.0);
        void set_value(OfxImageEffectHandle effect, std::string_view name, std::vector<InstanceValue> values);
        std::vector<InstanceValue> value(OfxImageEffectHandle effect,
                                         std::string_view name) const;
        void clear_calls();
        std::vector<InstanceCall>& calls() noexcept;
        bool callback_error() const noexcept;
        // Invoked once at the selected operation; status applies to that operation.
        void fail_next(std::string operation, OfxStatus status);
        void fail_on_nth(int occurrence, std::string operation, OfxStatus status);
        void null_on_nth(std::string operation, int occurrence);
        void null_next(std::string operation);
        void store_then_fail_next(std::string operation, OfxStatus status);
        void set_parameter_type(OfxImageEffectHandle effect, std::string_view name, std::string type);
        void* instance_data(OfxImageEffectHandle effect) const;
        std::string effect_scope(OfxImageEffectHandle effect) const;
        void set_effect_context(OfxImageEffectHandle effect, std::string context);
        void set_parameter_presentation(OfxImageEffectHandle effect, std::string_view name, std::string_view property, std::vector<InstanceValue> values);
        void return_string_on_failure_next(std::string operation, OfxStatus status);
        void reuse_string_getter_storage(bool enabled) noexcept;
        std::size_t string_getter_reuses() const noexcept;
        void reenter_next(std::string operation, std::function<void()> callback);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace ofx_test
