#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ofxImageEffectExt.h"
#include "ofxParametricParam.h"

#include "context_host.h"
#include "host_reference.h"

#include "juicer_ofx_api.h"
#include "../ffi/case_results.h"

namespace {
    using ofx_test::ContextHost;
    using ofx_test::Definition;
    constexpr std::string_view kCaller = "ofx.context";
    constexpr std::array kCases{"reference", "filter", "contexts", "acquisitions", "definitions", "properties", "catalog", "reentry"};
    constexpr std::array<std::pair<std::string_view, std::string_view>, 128> kExpected{{
        {"InputGroup", kOfxParamTypeGroup},
        {"Exposure", kOfxParamTypeDouble},
        {"CameraMeteringMethod", kOfxParamTypeChoice},
        {"CameraAutoExposure", kOfxParamTypeBoolean},
        {"CameraFilmFormatPreset", kOfxParamTypeChoice},
        {"CameraFilmFormatMm", kOfxParamTypeDouble},
        {"FilmProfileKey", kOfxParamTypeStrChoice},
        {"SpectralUpsampling", kOfxParamTypeChoice},
        {"InputColorSpace", kOfxParamTypeChoice},
        {"InputCctfDecoding", kOfxParamTypeBoolean},
        {"InputAdvancedGroup", kOfxParamTypeGroup},
        {"InputCompression", kOfxParamTypeBoolean},
        {"Hanatos2025AdaptationWindow", kOfxParamTypeBoolean},
        {"Hanatos2025AdaptationSurface", kOfxParamTypeBoolean},
        {"PrintGroup", kOfxParamTypeGroup},
        {"PrintProfileKey", kOfxParamTypeStrChoice},
        {"ScanRoute", kOfxParamTypeStrChoice},
        {"PrintExposure", kOfxParamTypeDouble},
        {"PrintPreflash", kOfxParamTypeDouble},
        {"PrintExposureCompensation", kOfxParamTypeBoolean},
        {"EnlargerY", kOfxParamTypeDouble},
        {"EnlargerM", kOfxParamTypeDouble},
        {"EnlargerC", kOfxParamTypeDouble},
        {"TuningGroup", kOfxParamTypeGroup},
        {"FilmGammaFactor", kOfxParamTypeDouble},
        {"PrintGammaFactor", kOfxParamTypeDouble},
        {"Couplers", kOfxParamTypeGroup},
        {"CouplersActive", kOfxParamTypeBoolean},
        {"CouplersAmount", kOfxParamTypeDouble},
        {"CouplersInhibitionSameLayer", kOfxParamTypeDouble},
        {"CouplersInhibitionInterlayer", kOfxParamTypeDouble},
        {"CouplersDiffusionSizeUm", kOfxParamTypeDouble},
        {"CouplersAdvancedGroup", kOfxParamTypeGroup},
        {"CouplersLangmuirDonorKRgb", kOfxParamTypeDouble3D},
        {"CouplersLangmuirReceiverKRgb", kOfxParamTypeDouble3D},
        {"CouplersDiffusionTailUm", kOfxParamTypeDouble},
        {"CouplersDiffusionTailWeight", kOfxParamTypeDouble},
        {"CouplersGammaUseStock", kOfxParamTypeBoolean},
        {"CouplersGammaSameLayerRgb", kOfxParamTypeDouble3D},
        {"CouplersGammaInterlayerRToGb", kOfxParamTypeDouble2D},
        {"CouplersGammaInterlayerGToRb", kOfxParamTypeDouble2D},
        {"CouplersGammaInterlayerBToRg", kOfxParamTypeDouble2D},
        {"ScannerOptics", kOfxParamTypeGroup},
        {"ScannerMath", kOfxParamTypeGroup},
        {"ScannerLensBlurSigmaPx", kOfxParamTypeDouble},
        {"ScannerUnsharpMask", kOfxParamTypeDouble2D},
        {"ScannerBlackCorrection", kOfxParamTypeBoolean},
        {"ScannerWhiteCorrection", kOfxParamTypeBoolean},
        {"ScannerBlackLevel", kOfxParamTypeDouble},
        {"ScannerWhiteLevel", kOfxParamTypeDouble},
        {"ScannerLutResolution", kOfxParamTypeInteger},
        {"HalationGroup", kOfxParamTypeGroup},
        {"HalationActive", kOfxParamTypeBoolean},
        {"HalationScatterAmount", kOfxParamTypeDouble},
        {"HalationScatterSpatialScale", kOfxParamTypeDouble},
        {"HalationAmount", kOfxParamTypeDouble},
        {"HalationSpatialScale", kOfxParamTypeDouble},
        {"GrainGroup", kOfxParamTypeGroup},
        {"GrainActive", kOfxParamTypeBoolean},
        {"GrainSublayersActive", kOfxParamTypeBoolean},
        {"GrainPreset", kOfxParamTypeChoice},
        {"GrainAmplitude", kOfxParamTypeDouble},
        {"GrainBlur", kOfxParamTypeDouble},
        {"GrainSharpness", kOfxParamTypeDouble},
        {"GrainChroma", kOfxParamTypeDouble},
        {"GrainTexture", kOfxParamTypeDouble},
        {"GrainAdvancedGroup", kOfxParamTypeGroup},
        {"GrainResetAdvanced", kOfxParamTypePushButton},
        {"GrainParticleAreaUm2", kOfxParamTypeDouble},
        {"GrainParticleScaleMaster", kOfxParamTypeDouble},
        {"GrainParticleScaleLayersMaster", kOfxParamTypeDouble},
        {"GrainDensityMinMaster", kOfxParamTypeDouble},
        {"GrainUniformityMaster", kOfxParamTypeDouble},
        {"GrainBlurDyeCloudsUm", kOfxParamTypeDouble},
        {"GrainSizeMixWeight", kOfxParamTypeDouble},
        {"GrainSizeMixWeightMid", kOfxParamTypeDouble},
        {"GrainSizeMixScale", kOfxParamTypeDouble},
        {"GrainMicroStructure", kOfxParamTypeDouble2D},
        {"GrainDebugView", kOfxParamTypeChoice},
        {"GrainParticleScale", kOfxParamTypeDouble3D},
        {"GrainParticleScaleLayers", kOfxParamTypeDouble3D},
        {"GrainDensityMin", kOfxParamTypeDouble3D},
        {"GrainUniformity", kOfxParamTypeDouble3D},
        {"GrainClumpTemporalMix", kOfxParamTypeDouble},
        {"GrainClumpMorphPeriodSec", kOfxParamTypeDouble},
        {"EffectsGroup", kOfxParamTypeGroup},
        {"GateWeaveAmount", kOfxParamTypeDouble},
        {"FilmDustAmount", kOfxParamTypeDouble},
        {"GateDustAmount", kOfxParamTypeDouble},
        {"FilmScratchAmount", kOfxParamTypeDouble},
        {"GateScratchAmount", kOfxParamTypeDouble},
        {"DiffusionGroup", kOfxParamTypeGroup},
        {"CameraDiffusionEnabled", kOfxParamTypeBoolean},
        {"CameraDiffusionFamily", kOfxParamTypeChoice},
        {"CameraDiffusionStrength", kOfxParamTypeDouble},
        {"CameraDiffusionSpatialScale", kOfxParamTypeDouble},
        {"CameraDiffusionHaloWarmth", kOfxParamTypeDouble},
        {"CameraDiffusionCoreIntensity", kOfxParamTypeDouble},
        {"CameraDiffusionCoreSize", kOfxParamTypeDouble},
        {"CameraDiffusionHaloIntensity", kOfxParamTypeDouble},
        {"CameraDiffusionHaloSize", kOfxParamTypeDouble},
        {"CameraDiffusionBloomIntensity", kOfxParamTypeDouble},
        {"CameraDiffusionBloomSize", kOfxParamTypeDouble},
        {"PrintDiffusionEnabled", kOfxParamTypeBoolean},
        {"PrintDiffusionFamily", kOfxParamTypeChoice},
        {"PrintDiffusionStrength", kOfxParamTypeDouble},
        {"PrintDiffusionSpatialScale", kOfxParamTypeDouble},
        {"PrintDiffusionHaloWarmth", kOfxParamTypeDouble},
        {"PrintDiffusionCoreIntensity", kOfxParamTypeDouble},
        {"PrintDiffusionCoreSize", kOfxParamTypeDouble},
        {"PrintDiffusionHaloIntensity", kOfxParamTypeDouble},
        {"PrintDiffusionHaloSize", kOfxParamTypeDouble},
        {"PrintDiffusionBloomIntensity", kOfxParamTypeDouble},
        {"PrintDiffusionBloomSize", kOfxParamTypeDouble},
        {"GlareGroup", kOfxParamTypeGroup},
        {"GlareActive", kOfxParamTypeBoolean},
        {"GlarePercent", kOfxParamTypeDouble},
        {"GlareRoughness", kOfxParamTypeDouble},
        {"GlareBlurSigmaPx", kOfxParamTypeDouble},
        {"PrintShadowCompensationFactor", kOfxParamTypeDouble},
        {"PrintShadowCompensationDensity", kOfxParamTypeDouble},
        {"PrintShadowCompensationTransition", kOfxParamTypeDouble},
        {"ReferenceIlluminant", kOfxParamTypeChoice},
        {"EnlargerIlluminant", kOfxParamTypeChoice},
        {"OutputEncodingGroup", kOfxParamTypeGroup},
        {"OutputColorSpace", kOfxParamTypeChoice},
        {"OutputCctfEncoding", kOfxParamTypeBoolean},
        {"OutputGamutCompression", kOfxParamTypeBoolean},
    }};
    void require(bool value, const std::string& message) {
        if (!value)
            throw std::runtime_error(message);
    }
    OfxStatus call(OfxPlugin* plugin, const char* action, const void* handle = nullptr, OfxPropertySetHandle input = nullptr) {
        return plugin->mainEntry(action, handle, input, nullptr);
    }
    OfxStatus describe(OfxPlugin* plugin, ContextHost& host) {
        return call(plugin, kOfxImageEffectActionDescribeInContext, host.effect_handle(), host.context_handle());
    }
    bool accepted(int status) {
        return status == kOfxStatOK || status == kOfxStatReplyYes || status == kOfxStatReplyNo || status == kOfxStatReplyDefault;
    }
    int property_result(int status) {
        if (status == kOfxStatErrUnknown || status == kOfxStatErrUnsupported)
            return kOfxStatErrMissingHostFeature;
        if (status == kOfxStatErrValue)
            return kOfxStatFailed;
        return status;
    }
    bool same(const ContextHost& a, const ContextHost& b) {
        if (ofx_test::reference::context(a)[3] != ofx_test::reference::context(b)[3])
            return false;
        auto actual = b.calls;
#ifdef NDEBUG
        // OFXS omits this read when assertions are disabled.
        std::erase_if(actual, [](const auto& c) {
            return c.operation == "dimension" && c.property == kOfxParamPropChoiceEnum;
        });
#endif
        return ofx_test::reference::calls(a.calls) == ofx_test::reference::calls(actual);
    }
    void check_census(const ContextHost& host) {
        require(host.definitions.size() == kExpected.size() + 2, "definition count");
        require(host.definitions[0].name == "Source" && host.definitions[1].name == "Output" &&
                    host.definitions[0].type == kOfxTypeClip && host.definitions[1].type == kOfxTypeClip,
                "clip order");
        for (std::size_t i = 0; i < kExpected.size(); ++i)
            require(host.definitions[i + 2].name == kExpected[i].first && host.definitions[i + 2].type == kExpected[i].second,
                    "declaration census at " + std::to_string(i));
    }
    void check_choice_pairs(const ContextHost& host) {
        for (std::size_t i = 0; i < host.calls.size(); ++i) {
            const auto& c = host.calls[i];
            if (c.operation == "string" && c.property == kOfxParamPropChoiceEnum) {
                require(i >= 2 && i + 1 < host.calls.size(), "string choice trace length");
                require(host.calls[i - 2].operation == "dimension" && host.calls[i - 2].property == kOfxParamPropChoiceOption &&
                            host.calls[i - 1].operation == "dimension" && host.calls[i - 1].property == kOfxParamPropChoiceEnum &&
                            host.calls[i + 1].operation == "string" && host.calls[i + 1].property == kOfxParamPropChoiceOption &&
                            host.calls[i + 1].target == c.target && host.calls[i + 1].index == c.index,
                        "string choice append order");
            }
        }
    }
    std::vector<std::string> matrix_rows(std::string_view prefix, const std::vector<std::string>& names) {
        std::vector<std::string> rows;
        for (const auto& name : names)
            for (int status = 0; status < 15; ++status)
                rows.push_back(std::string(prefix) + "/" + name + "/" + std::to_string(status));
        return rows;
    }
    std::size_t call_end(const std::vector<ofx_test::ContextCall>& expected, std::string_view key, std::size_t start = 0) {
        const auto slash = key.find('/');
        const auto operation = key.substr(0, slash);
        const auto property = slash == std::string_view::npos ? std::string_view{} : key.substr(slash + 1);
        require(start <= expected.size(), "expected trace search range");
        const auto found = std::find_if(expected.begin() + static_cast<std::ptrdiff_t>(start), expected.end(), [&](const auto& c) {
            return c.operation == operation && c.property == property;
        });
        require(found != expected.end(), "missing expected operation " + std::string(key));
        return static_cast<std::size_t>(found - expected.begin()) + 1;
    }
    bool matches_trace(const ContextHost& actual, const std::vector<ofx_test::ContextCall>& expected, std::size_t count) {
        return !actual.callback_error && !expected.empty() && count <= expected.size() &&
               actual.calls.size() == count && std::equal(actual.calls.begin(), actual.calls.end(), expected.begin());
    }
    const Definition& definition(const ContextHost& host, std::string_view name) {
        const auto found = std::find_if(host.definitions.begin(), host.definitions.end(), [&](const auto& d) {
            return d.name == name;
        });
        require(found != host.definitions.end(), "missing definition " + std::string(name));
        return *found;
    }
    void check_startup(const std::vector<ofx_test::ContextCall>& calls) {
        const std::vector<ofx_test::ContextCall> expected{
            {"fetch", "OfxImageEffectSuite", {}, 1, 0, 0}, {"fetch", "OfxPropertySuite", {}, 1, 0, 0}, {"fetch", "OfxParameterSuite", {}, 1, 0, 0}, {"fetch", "OfxMemorySuite", {}, 1, 0, 0}, {"fetch", "OfxMultiThreadSuite", {}, 1, 0, 0}, {"fetch", "OfxMessageSuite", {}, 1, 0, 0}, {"fetch", "OfxProgressSuite", {}, 1, 0, 0}, {"fetch", "OfxProgressSuite", {}, 2, 0, 0}, {"fetch", "OfxTimeLineSuite", {}, 1, 0, 0}, {"get_int", kOfxPropAPIVersion, {}, 0, 0, 0}, {"get_int", kOfxPropAPIVersion, {}, 1, 0, 0}, {"get_string", kOfxPropName, {}, 0, 0, 0}, {"get_string", kOfxPropLabel, {}, 0, 0, 0}, {"get_int", kOfxPropVersion, {}, 0, 0, 0}, {"get_int", kOfxPropVersion, {}, 1, 0, 0}, {"get_int", kOfxPropVersion, {}, 2, 0, 0}, {"get_string", kOfxPropVersionLabel, {}, 0, 0, 0}, {"get_int", kOfxImageEffectHostPropIsBackground, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropSupportsOverlays, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropSupportsMultiResolution, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropSupportsTiles, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropTemporalClipAccess, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropSupportsMultipleClipDepths, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropSupportsMultipleClipPARs, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropSetableFrameRate, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropSetableFielding, {}, 0, 0, 0}, {"get_int", kOfxImageEffectInstancePropSequentialRender, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropSupportsStringAnimation, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropSupportsCustomInteract, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropSupportsChoiceAnimation, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropSupportsStrChoiceAnimation, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropSupportsBooleanAnimation, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropSupportsCustomAnimation, {}, 0, 0, 0}, {"get_pointer", kOfxPropHostOSHandle, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropSupportsParametricAnimation, {}, 0, 0, 0}, {"get_string", kOfxImageEffectPropOpenCLRenderSupported, {}, 0, 0, 0}, {"get_string", kOfxImageEffectPropCudaRenderSupported, {}, 0, 0, 0}, {"get_string", kOfxImageEffectPropCudaStreamSupported, {}, 0, 0, 0}, {"get_string", kOfxImageEffectPropMetalRenderSupported, {}, 0, 0, 0}, {"get_int", kOfxImageEffectPropRenderQualityDraft, {}, 0, 0, 0}, {"get_string", kOfxImageEffectHostPropNativeOrigin, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropMaxParameters, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropMaxPages, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropPageRowColumnCount, {}, 0, 0, 0}, {"get_int", kOfxParamHostPropPageRowColumnCount, {}, 1, 0, 0}, {"dimension", kOfxImageEffectPropSupportedComponents, {}, 0, 0, 0}, {"dimension", kOfxImageEffectPropSupportedContexts, {}, 0, 0, 0}, {"dimension", kOfxImageEffectPropSupportedPixelDepths, {}, 0, 0, 0}};
        require(calls == expected, "OFXS startup sequence");
    }
    void reference_case(const ContextHost& host, const std::vector<ofx_test::ContextCall>& startup) {
        JuicerTests::CaseEvent test(kCaller, "reference");
        const JuicerTests::Cohort rows("ofx.context.reference", {"census", "startup", "missing", "extra", "type", "default", "option_order", "target", "premature_success", "late_stop"});
        {
            auto row = rows.case_event("census");
            check_census(host);
            require(!host.callback_error, "reference host callback");
            row.passed();
        }
        {
            auto row = rows.case_event("startup");
            check_startup(startup);
            row.passed();
        }
        auto mutation = [&](const char* id, auto change) {
            auto row = rows.case_event(id);
            ContextHost copy;
            copy.calls = host.calls;
            copy.definitions = host.definitions;
            change(copy);
            require(!same(host, copy), std::string("comparator missed ") + id);
            row.passed();
        };
        mutation("missing", [](auto& h) {
            h.definitions.pop_back();
        });
        mutation("extra", [](auto& h) {
            h.definitions.push_back(h.definitions.back());
        });
        mutation("type", [](auto& h) {
            h.definitions[2].type = "wrong";
        });
        mutation("default", [](auto& h) {
            h.definitions[3].properties[kOfxParamPropDefault] = {99.0};
        });
        mutation("option_order", [](auto& h) {
            auto& v = h.definitions[4].properties[kOfxParamPropChoiceOption];
            require(v.size() > 1, "option mutation control");
            std::swap(v[0], v[1]);
        });
        mutation("target", [](auto& h) {
            require(!h.calls.empty(), "target mutation control");
            h.calls.back().target += 1;
        });
        {
            auto row = rows.case_event("premature_success");
            require(matches_trace(host, host.calls, host.calls.size()), "complete trace positive control");
            ContextHost premature;
            premature.calls = host.calls;
            const auto first = call_end(host.calls, "string/OfxPropLabel");
            require(first < host.calls.size(), "premature trace control boundary");
            premature.calls.resize(first);
            require(!matches_trace(premature, host.calls, host.calls.size()), "premature success was accepted");
            row.passed();
        }
        {
            auto row = rows.case_event("late_stop");
            const auto first = call_end(host.calls, "string/OfxPropLabel");
            const auto second = call_end(host.calls, "string/OfxPropLabel", first);
            require(second > first && host.calls[first - 1].target != host.calls[second - 1].target, "distinct failure control boundaries");
            ContextHost stopped;
            stopped.calls = host.calls;
            stopped.calls.resize(first);
            require(matches_trace(stopped, host.calls, first), "first failure positive control");
            stopped.calls = host.calls;
            stopped.calls.resize(second);
            require(!matches_trace(stopped, host.calls, first), "late matching failure was accepted");
            row.passed();
        }
        test.passed();
    }
    void filter_case(OfxPlugin* plugin, const ContextHost& reference) {
        JuicerTests::CaseEvent test(kCaller, "filter");
        const JuicerTests::Cohort rows("ofx.context.filter", {"equal", "repeat", "string_choice_sequence"});
        ContextHost first, second;
        {
            auto row = rows.case_event("equal");
            require(describe(plugin, first) == kOfxStatOK && !first.callback_error, "candidate filter");
            check_census(first);
            require(same(reference, first), "candidate differs from reference");
            auto altered = ofx_test::reference::context(reference);
            altered[3][0][0] = "altered";
            ContextHost changed;
            ofx_test::reference::context(altered, changed);
            require(!same(changed, first), "candidate accepted altered reference definition");
            altered = ofx_test::reference::context(reference);
            std::swap(altered[3][0], altered[3][1]);
            ContextHost reordered;
            ofx_test::reference::context(altered, reordered);
            require(!same(reordered, first), "candidate accepted reordered reference definitions");
            row.passed();
        }
        {
            auto row = rows.case_event("repeat");
            require(describe(plugin, second) == kOfxStatOK && !second.callback_error && first.definitions == second.definitions && first.calls == second.calls, "fresh filter repeat");
            row.passed();
        }
        {
            auto row = rows.case_event("string_choice_sequence");
            check_choice_pairs(first);
            row.passed();
        }
        test.passed();
    }
    void contexts_case(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "contexts");
        const JuicerTests::Cohort rows("ofx.context.contexts", {"generator", "transition", "paint", "general", "retimer", "unknown", "null_effect", "null_input", "advertised"});
        const std::array pairs{std::pair{"generator", kOfxImageEffectContextGenerator}, std::pair{"transition", kOfxImageEffectContextTransition}, std::pair{"paint", kOfxImageEffectContextPaint}, std::pair{"general", kOfxImageEffectContextGeneral}, std::pair{"retimer", kOfxImageEffectContextRetimer}};
        for (const auto& [id, name] : pairs) {
            auto row = rows.case_event(id);
            ContextHost host;
            host.context = name;
            require(describe(plugin, host) == kOfxStatOK && host.definitions.empty() && host.calls.size() == 5 && !host.callback_error, "nonfilter context");
            row.passed();
        }
        {
            auto row = rows.case_event("unknown");
            ContextHost host;
            host.context = "unknown";
            require(describe(plugin, host) == kOfxStatFailed && host.calls.size() == 4 && host.definitions.empty(), "unknown context");
            row.passed();
        }
        {
            auto row = rows.case_event("null_effect");
            ContextHost host;
            require(call(plugin, kOfxImageEffectActionDescribeInContext, nullptr, host.context_handle()) == kOfxStatErrBadHandle && host.calls.empty(), "null effect");
            row.passed();
        }
        {
            auto row = rows.case_event("null_input");
            ContextHost host;
            require(call(plugin, kOfxImageEffectActionDescribeInContext, host.effect_handle()) == kOfxStatErrBadHandle && host.calls.empty(), "null input");
            row.passed();
        }
        {
            auto row = rows.case_event("advertised");
            ContextHost host;
            require(call(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatOK && host.effect_values[kOfxImageEffectPropSupportedContexts] == std::vector<ofx_test::Value>{std::string(kOfxImageEffectContextFilter)}, "advertised contexts");
            row.passed();
        }
        test.passed();
    }
    void acquisitions_case(OfxPlugin* plugin, const ContextHost& baseline) {
        JuicerTests::CaseEvent test(kCaller, "acquisitions");
        const std::vector<std::string> names{"effect_first", "param_set", "param_props", "get_string/OfxImageEffectPropContext", "effect_second"};
        auto ids = matrix_rows("status", names);
        for (const auto* id : {"null_first", "null_params_filter", "null_params_nonfilter", "null_param_props", "null_second", "distinct_second", "null_context", "failed_null_context", "failed_null_first"})
            ids.emplace_back(id);
        const JuicerTests::Cohort rows("ofx.context.acquisitions", ids);
        for (const auto& name : names)
            for (int status = 0; status < 15; ++status) {
                auto row = rows.case_event("status/" + name + "/" + std::to_string(status));
                ContextHost host;
                host.statuses[name] = status;
                const auto result = describe(plugin, host);
                require(result == (accepted(status) ? kOfxStatOK : name.starts_with("get_string/") ? property_result(status)
                                                                                                   : status),
                        "acquisition status " + name);
                const auto count = accepted(status) ? baseline.calls.size() : call_end(baseline.calls, name);
                require(matches_trace(host, baseline.calls, count), "exact acquisition trace " + name);
                row.passed();
            }
        {
            auto row = rows.case_event("null_first");
            ContextHost h;
            h.null_outputs["effect_first"] = true;
            require(describe(plugin, h) == kOfxStatErrBadHandle, "null first");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "effect_first")), "null first trace");
            row.passed();
        }
        {
            auto row = rows.case_event("null_params_filter");
            ContextHost h;
            h.null_outputs["param_set"] = true;
            auto expected = baseline.calls;
            expected.erase(expected.begin() + static_cast<std::ptrdiff_t>(call_end(expected, "param_props") - 1));
            const auto count = call_end(expected, "param_define/InputGroup") - 1;
            require(describe(plugin, h) == kOfxStatErrBadHandle && h.definitions.size() == 2 && h.definitions[0].name == "Source" && h.definitions[1].name == "Output", "null params filter");
            require(matches_trace(h, expected, count), "null params filter trace");
            row.passed();
        }
        {
            auto row = rows.case_event("null_params_nonfilter");
            ContextHost h;
            h.null_outputs["param_set"] = true;
            h.context = kOfxImageEffectContextGeneral;
            auto expected = baseline.calls;
            expected.erase(expected.begin() + static_cast<std::ptrdiff_t>(call_end(expected, "param_props") - 1));
            expected[call_end(expected, "get_string/OfxImageEffectPropContext") - 1].text = kOfxImageEffectContextGeneral;
            const auto count = call_end(expected, "effect_second");
            require(describe(plugin, h) == kOfxStatOK && h.definitions.empty(), "null params nonfilter");
            require(matches_trace(h, expected, count), "null params nonfilter trace");
            row.passed();
        }
        {
            auto row = rows.case_event("null_param_props");
            ContextHost h;
            h.null_outputs["param_props"] = true;
            require(describe(plugin, h) == kOfxStatOK && h.definitions == baseline.definitions, "unused params properties");
            require(matches_trace(h, baseline.calls, baseline.calls.size()), "unused params properties trace");
            row.passed();
        }
        {
            auto row = rows.case_event("null_second");
            ContextHost h;
            h.null_outputs["effect_second"] = true;
            require(describe(plugin, h) == kOfxStatOK && h.definitions == baseline.definitions, "unused second properties");
            require(matches_trace(h, baseline.calls, baseline.calls.size()), "unused second properties trace");
            row.passed();
        }
        {
            auto row = rows.case_event("distinct_second");
            ContextHost h;
            h.distinct_second = true;
            require(describe(plugin, h) == kOfxStatOK && h.definitions == baseline.definitions && h.calls == baseline.calls, "second properties target");
            row.passed();
        }
        {
            auto row = rows.case_event("null_context");
            ContextHost h;
            h.null_outputs["get_string"] = true;
            require(describe(plugin, h) == kOfxStatErrBadHandle, "null context");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "get_string/OfxImageEffectPropContext")), "null context trace");
            row.passed();
        }
        {
            auto row = rows.case_event("failed_null_context");
            ContextHost h;
            h.null_outputs["get_string"] = true;
            h.statuses["get_string/OfxImageEffectPropContext"] = kOfxStatErrMemory;
            require(describe(plugin, h) == kOfxStatErrMemory, "failed null context precedence");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "get_string/OfxImageEffectPropContext")), "failed null context trace");
            row.passed();
        }
        {
            auto row = rows.case_event("failed_null_first");
            ContextHost h;
            h.null_outputs["effect_first"] = true;
            h.statuses["effect_first"] = kOfxStatErrMemory;
            require(describe(plugin, h) == kOfxStatErrMemory, "failed null first precedence");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "effect_first")), "failed null first trace");
            row.passed();
        }
        test.passed();
    }
    void definitions_case(OfxPlugin* plugin, const ContextHost& baseline) {
        JuicerTests::CaseEvent test(kCaller, "definitions");
        const std::vector<std::string> kinds{"InputGroup", "Exposure", "CameraMeteringMethod", "CameraAutoExposure", "FilmProfileKey", "CouplersLangmuirDonorKRgb", "CouplersGammaInterlayerRToGb", "ScannerLutResolution", "GrainResetAdvanced"};
        auto ids = matrix_rows("param", {"InputGroup"});
        for (const auto& name : kinds)
            ids.push_back("kind/" + name);
        for (int status = 0; status < 15; ++status)
            ids.push_back("clip/" + std::to_string(status));
        for (const auto* id : {"null_param", "null_clip", "failed_null_param", "failed_null_clip"})
            ids.emplace_back(id);
        const JuicerTests::Cohort rows("ofx.context.definitions", ids);
        for (int status = 0; status < 15; ++status) {
            auto row = rows.case_event("param/InputGroup/" + std::to_string(status));
            ContextHost h;
            h.statuses["param_define/InputGroup"] = status;
            require(describe(plugin, h) == (accepted(status) ? kOfxStatOK : status), "paramDefine status");
            const auto count = accepted(status) ? baseline.calls.size() : call_end(baseline.calls, "param_define/InputGroup");
            require(matches_trace(h, baseline.calls, count), "exact paramDefine trace");
            row.passed();
        }
        for (const auto& name : kinds) {
            auto row = rows.case_event("kind/" + name);
            ContextHost h;
            h.statuses["param_define/" + name] = kOfxStatErrMemory;
            require(describe(plugin, h) == kOfxStatErrMemory, "param kind failure");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "param_define/" + name)), "exact parameter kind trace");
            row.passed();
        }
        {
            auto row = rows.case_event("null_param");
            ContextHost h;
            h.null_outputs["param_define"] = true;
            require(describe(plugin, h) == kOfxStatErrBadHandle, "null parameter");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "param_define/InputGroup")), "null parameter trace");
            row.passed();
        }
        {
            auto row = rows.case_event("null_clip");
            ContextHost h;
            h.null_outputs["clip_define"] = true;
            require(describe(plugin, h) == kOfxStatErrBadHandle, "null clip");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "clip_define/Source")), "null clip trace");
            row.passed();
        }
        {
            auto row = rows.case_event("failed_null_param");
            ContextHost h;
            h.statuses["param_define/InputGroup"] = kOfxStatErrMemory;
            h.null_outputs["param_define"] = true;
            require(describe(plugin, h) == kOfxStatErrMemory, "paramDefine status before null");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "param_define/InputGroup")), "failed null parameter trace");
            row.passed();
        }
        {
            auto row = rows.case_event("failed_null_clip");
            ContextHost h;
            h.statuses["clip_define/Source"] = kOfxStatErrMemory;
            h.null_outputs["clip_define"] = true;
            require(describe(plugin, h) == kOfxStatErrBadHandle, "clipDefine ignores status before null");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "clip_define/Source")), "failed null clip trace");
            row.passed();
        }
        for (int status = 0; status < 15; ++status) {
            auto row = rows.case_event("clip/" + std::to_string(status));
            ContextHost h;
            h.statuses["clip_define/Source"] = status;
            require(describe(plugin, h) == kOfxStatOK && h.definitions == baseline.definitions && h.calls == baseline.calls, "clip status ignored");
            row.passed();
        }
        test.passed();
    }
    void properties_case(OfxPlugin* plugin, const ContextHost& baseline) {
        JuicerTests::CaseEvent test(kCaller, "properties");
        const std::vector<std::string> operations{"string/OfxPropLabel", "int/OfxImageEffectPropSupportsTiles", "double/OfxParamPropDefault", "dimension/OfxImageEffectPropSupportedComponents"};
        auto ids = matrix_rows("required", operations);
        for (int status = 0; status < 15; ++status) {
            ids.push_back("hint/" + std::to_string(status));
            ids.push_back("open/" + std::to_string(status));
        }
        for (const auto* id : {"hint_then_required", "open_then_required", "append_nonzero", "negative_dimension", "negative_choice", "negative_enum", "unpaired_dimensions", "exact_double"})
            ids.emplace_back(id);
        const JuicerTests::Cohort rows("ofx.context.properties", ids);
        for (const auto& operation : operations)
            for (int status = 0; status < 15; ++status) {
                auto row = rows.case_event("required/" + operation + "/" + std::to_string(status));
                ContextHost h;
                h.statuses[operation] = status;
                require(describe(plugin, h) == (accepted(status) ? kOfxStatOK : property_result(status)), "property status " + operation);
                const auto count = accepted(status) ? baseline.calls.size() : call_end(baseline.calls, operation);
                require(matches_trace(h, baseline.calls, count), "exact required property trace " + operation);
                row.passed();
            }
        for (int status = 0; status < 15; ++status)
            for (const auto& [id, key] : {std::pair{"hint", "string/OfxParamPropHint"}, std::pair{"open", "int/OfxParamPropGroupOpen"}}) {
                auto row = rows.case_event(std::string(id) + "/" + std::to_string(status));
                ContextHost h;
                h.statuses[key] = status;
                require(describe(plugin, h) == kOfxStatOK && !h.callback_error && h.calls == baseline.calls, "suppressed property status");
                row.passed();
            }
        for (const auto& [id, key] : {std::pair{"hint_then_required", "string/OfxParamPropHint"}, std::pair{"open_then_required", "int/OfxParamPropGroupOpen"}}) {
            auto row = rows.case_event(id);
            ContextHost h;
            h.statuses[key] = kOfxStatErrMemory;
            h.status_after[key] = {"double/OfxParamPropDefault", kOfxStatErrMemory};
            const auto trigger = call_end(baseline.calls, key);
            const auto count = call_end(baseline.calls, "double/OfxParamPropDefault", trigger);
            require(describe(plugin, h) == kOfxStatErrMemory, "later required failure precedence");
            require(matches_trace(h, baseline.calls, count), "exact later required failure trace");
            row.passed();
        }
        {
            auto row = rows.case_event("append_nonzero");
            ContextHost h;
            h.dimensions[kOfxParamPropChoiceOption] = 2;
            h.dimensions[kOfxParamPropChoiceEnum] = 2;
            auto expected = baseline.calls;
            for (auto& c : expected)
                if (c.operation == "string" && (c.property == kOfxParamPropChoiceOption || c.property == kOfxParamPropChoiceEnum))
                    c.index = 2;
            require(describe(plugin, h) == kOfxStatOK, "nonzero append");
            require(matches_trace(h, expected, expected.size()), "complete nonzero append trace");
            require(std::any_of(h.calls.begin(), h.calls.end(), [](const auto& c) {
                        return c.operation == "string" && c.property == kOfxParamPropChoiceEnum && c.index == 2;
                    }),
                    "append index");
            check_choice_pairs(h);
            row.passed();
        }
        {
            auto row = rows.case_event("negative_dimension");
            ContextHost h;
            h.dimensions[kOfxImageEffectPropSupportedComponents] = -1;
            require(describe(plugin, h) == kOfxStatErrBadIndex, "negative dimension");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "dimension/OfxImageEffectPropSupportedComponents")), "negative component dimension trace");
            row.passed();
        }
        {
            auto row = rows.case_event("negative_choice");
            ContextHost h;
            h.dimensions[kOfxParamPropChoiceOption] = -1;
            require(describe(plugin, h) == kOfxStatErrBadIndex, "negative choice dimension");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "dimension/OfxParamPropChoiceOption")), "negative choice dimension trace");
            row.passed();
        }
        {
            auto row = rows.case_event("negative_enum");
            ContextHost h;
            h.dimensions[kOfxParamPropChoiceEnum] = -1;
            require(describe(plugin, h) == kOfxStatErrBadIndex, "negative enum dimension");
            require(matches_trace(h, baseline.calls, call_end(baseline.calls, "dimension/OfxParamPropChoiceEnum")), "negative enum dimension trace");
            row.passed();
        }
        {
            auto row = rows.case_event("unpaired_dimensions");
            ContextHost h;
            h.dimensions[kOfxParamPropChoiceOption] = 1;
            h.dimensions[kOfxParamPropChoiceEnum] = 0;
            auto expected = baseline.calls;
            const auto count = call_end(expected, "dimension/OfxParamPropChoiceEnum");
            for (std::size_t i = 0; i < count; ++i)
                if (expected[i].operation == "string" && expected[i].property == kOfxParamPropChoiceOption)
                    expected[i].index = 1;
            require(describe(plugin, h) == kOfxStatFailed, "mismatched dimensions");
            require(matches_trace(h, expected, count), "mismatched dimensions trace");
            row.passed();
        }
        {
            auto row = rows.case_event("exact_double");
            require(std::any_of(baseline.calls.begin(), baseline.calls.end(), [](const auto& c) {
                        return c.operation == "double" && std::holds_alternative<double>(c.value) && std::get<double>(c.value) != 0.0;
                    }),
                    "double positive control");
            row.passed();
        }
        test.passed();
    }
    void catalog_case(const ContextHost& host, const ContextHost& reference) {
        JuicerTests::CaseEvent test(kCaller, "catalog");
        const JuicerTests::Cohort rows("ofx.context.catalog", {"film", "print", "routes"});
        for (const auto& [id, name, value] : {std::tuple{"film", "FilmProfileKey", "kodak_portra_400"}, std::tuple{"print", "PrintProfileKey", "kodak_portra_endura"}}) {
            auto row = rows.case_event(id);
            const auto& d = definition(host, name);
            const auto& r = definition(reference, name);
            require(d.properties == r.properties && d.properties.at(kOfxParamPropDefault) == std::vector<ofx_test::Value>{std::string(value)}, "catalog default/equality");
            const auto& keys = d.properties.at(kOfxParamPropChoiceEnum);
            const auto& labels = d.properties.at(kOfxParamPropChoiceOption);
            require(!keys.empty() && keys.size() == labels.size(), "catalog options");
            row.passed();
        }
        {
            auto row = rows.case_event("routes");
            const auto& p = definition(host, "ScanRoute").properties;
            require(p.at(kOfxParamPropChoiceEnum) == std::vector<ofx_test::Value>{std::string("NegativeDirectScan"), std::string("NegativePrintScan"), std::string("PositiveDirectScan"), std::string("PositivePrintScan")}, "route keys");
            require(p.at(kOfxParamPropChoiceOption) == std::vector<ofx_test::Value>{std::string("Negative direct scan"), std::string("Negative print scan"), std::string("Positive direct scan"), std::string("Positive print scan")}, "route labels");
            row.passed();
        }
        test.passed();
    }
    void reentry_case(OfxPlugin* plugin, const ContextHost& baseline) {
        JuicerTests::CaseEvent test(kCaller, "reentry");
        const JuicerTests::Cohort rows("ofx.context.reentry", {"unload", "rebind", "recovery"});
        {
            auto row = rows.case_event("unload");
            ContextHost h;
            int nested = -1;
            h.reentry = [&] {
                nested = call(plugin, kOfxActionUnload);
            };
            require(describe(plugin, h) == kOfxStatOK && nested == kOfxStatErrFatal && h.calls == baseline.calls && h.definitions == baseline.definitions, "reentrant unload");
            row.passed();
        }
        ContextHost replacement;
        {
            auto row = rows.case_event("rebind");
            ContextHost h;
            h.reentry = [&] {
                plugin->setHost(replacement.binding());
            };
            require(describe(plugin, h) == kOfxStatOK && h.calls == baseline.calls && h.definitions == baseline.definitions, "reentrant setHost held borrow");
            ContextHost next;
            require(describe(plugin, next) == kOfxStatFailed && next.calls.empty(), "rebind disposition");
            require(call(plugin, kOfxActionUnload) == kOfxStatOK, "rebind cleanup");
            row.passed();
        }
        {
            auto row = rows.case_event("recovery");
            plugin->setHost(replacement.binding());
            require(call(plugin, kOfxActionLoad) == kOfxStatOK, "rebind reload");
            ContextHost h;
            require(describe(plugin, h) == kOfxStatOK && h.definitions == baseline.definitions && h.calls == baseline.calls, "rebind recovery");
            require(call(plugin, kOfxActionUnload) == kOfxStatOK, "rebind final unload");
            row.passed();
        }
        test.passed();
    }
} // namespace

int main(int argc, char** argv) try {
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto* id : kCases)
            JuicerTests::list_case(kCaller, id);
        return 0;
    }
    require(argc == 3 && std::string_view(argv[1]) == "--reference", "usage: context fixture --reference file");
    JuicerTests::CaseLog log;
    auto* candidate = OfxGetPlugin(0);
    require(candidate, "candidate descriptor");
    ContextHost oracle;
    ofx_test::reference::context(ofx_test::reference::read(argv[2], "context"), oracle);
    reference_case(oracle, oracle.startup);
    ContextHost binding;
    candidate->setHost(binding.binding());
    require(call(candidate, kOfxActionLoad) == kOfxStatOK, "candidate load");
    ContextHost baseline;
    require(describe(candidate, baseline) == kOfxStatOK && !baseline.callback_error, "candidate baseline");
    filter_case(candidate, oracle);
    contexts_case(candidate);
    acquisitions_case(candidate, baseline);
    definitions_case(candidate, baseline);
    properties_case(candidate, baseline);
    catalog_case(baseline, oracle);
    reentry_case(candidate, baseline);
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "context fixture: %s\n", error.what());
    return 1;
}
