#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "juicer_ofx_api.h"
#include "juicer_cuda_api.h"
#include "ofxImageEffectExt.h"
#include "module_host.h"

#include <nlohmann/json.hpp>

namespace {
    using ofx_test::Call;
    using ofx_test::Host;
    constexpr std::array kAcquisition{"effect_first", "param_set", "param_props"};
    constexpr std::array kRequired{"label", "grouping", "context_dimension", "context_write", "depth_dimension", "depth_write", "single_instance", "host_threading", "multiresolution", "tiles", "render_safety", "cuda_render", "cuda_stream"};
    constexpr std::array kOptional{"short_label", "long_label"};
    constexpr std::array kRequiredIndices{3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    constexpr std::array kOptionalIndices{4, 5};

    void require(bool condition, const std::string& detail) {
        if (!condition)
            throw std::runtime_error(detail);
    }
    OfxStatus action(OfxPlugin* plugin, const char* name, const void* handle = nullptr) {
        return plugin->mainEntry(name, handle, nullptr, nullptr);
    }
    void require_native_available() {
        FjCuda* native = nullptr;
        require(fj_cuda_create(&native, nullptr).category == FJ_STATUS_SUCCESS && native, "native owner unexpectedly present");
        require(fj_cuda_destroy(native, nullptr).category == FJ_STATUS_SUCCESS, "native probe close");
    }
    bool accepted(int status) {
        return status == 0 || status == 12 || status == 13 || status == 14;
    }
    int property_result(int status) {
        if (status == kOfxStatErrUnknown || status == kOfxStatErrUnsupported)
            return kOfxStatErrMissingHostFeature;
        if (status == kOfxStatErrValue)
            return kOfxStatFailed;
        return status;
    }
    std::vector<Call> expected(const char* label, int context = 0, int depth = 0) {
        return {
            {"effect_first", {}, 0, {}, 0, 0}, {"param_set", {}, 0, {}, 0, 0}, {"param_props", {}, 0, {}, 0, 0}, {"string", kOfxPropLabel, 0, label, 0, 1}, {"string", kOfxPropShortLabel, 0, label, 0, 1}, {"string", kOfxPropLongLabel, 0, label, 0, 1}, {"string", kOfxImageEffectPluginPropGrouping, 0, "Negative-juice", 0, 1}, {"dimension", kOfxImageEffectPropSupportedContexts, 0, {}, 0, 1}, {"string", kOfxImageEffectPropSupportedContexts, context, kOfxImageEffectContextFilter, 0, 1}, {"dimension", kOfxImageEffectPropSupportedPixelDepths, 0, {}, 0, 1}, {"string", kOfxImageEffectPropSupportedPixelDepths, depth, kOfxBitDepthFloat, 0, 1}, {"int", kOfxImageEffectPluginPropSingleInstance, 0, {}, 0, 1}, {"int", kOfxImageEffectPluginPropHostFrameThreading, 0, {}, 0, 1}, {"int", kOfxImageEffectPropSupportsMultiResolution, 0, {}, 1, 1}, {"int", kOfxImageEffectPropSupportsTiles, 0, {}, 0, 1}, {"string", kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderUnsafe, 0, 1}, {"string", kOfxImageEffectPropCudaRenderSupported, 0, "true", 0, 1}, {"string", kOfxImageEffectPropCudaStreamSupported, 0, "true", 0, 1}};
    }
    std::string key(const Call& call) {
        return call.property.empty() ? call.operation : call.operation + "/" + call.property;
    }
    void check_calls(const Host& host, const std::vector<Call>& expected_calls, std::string_view id) {
        require(!host.callback_error, std::string(id) + ": callback exception");
        require(host.calls.size() == expected_calls.size(), std::string(id) + ": wrong call count");
        for (std::size_t i = 0; i < expected_calls.size(); ++i) {
            const auto& a = host.calls[i];
            const auto& b = expected_calls[i];
            require(a.operation == b.operation && a.property == b.property && a.index == b.index && a.text == b.text && a.integer == b.integer && a.target == b.target,
                    std::string(id) + ": call mismatch at " + std::to_string(i));
        }
    }
    std::vector<std::string> describe_ids() {
        std::vector<std::string> ids;
        const auto add = [&](std::string_view category, const auto& names) {
            for (const auto* name : names)
                for (int status = 0; status < 15; ++status)
                    ids.push_back(std::string(category) + "/" + name + "/status_" + std::to_string(status));
        };
        add("acquire", kAcquisition);
        add("required", kRequired);
        add("optional", kOptional);
        ids.push_back("manifest");
        for (const auto* id : {"null_effect", "null_first", "null_param", "null_param_props", "failed_null/effect_first", "failed_null/param_set", "failed_null/param_props", "both_optional_fail", "short_then_value", "long_then_unknown", "append_dimensions"})
            ids.emplace_back(id);
        return ids;
    }
    struct DescriptorExpectation {
        const char* label;
        const char* manifest_path;
    };

    void describe(OfxPlugin* plugin, Host& host, const DescriptorExpectation& descriptor) {
        const auto ids = describe_ids();
        require(ids.size() == 282, "Describe membership size");
        const auto baseline = expected(descriptor.label);
        const auto run_matrix = [&](std::string_view group, const auto& names, const auto& indices, bool is_optional, bool is_acquisition) {
            for (std::size_t n = 0; n < names.size(); ++n)
                for (int status = 0; status < 15; ++status) {
                    const std::string id = std::string(group) + "/" + names[n] + "/status_" + std::to_string(status);
                    host.reset_trace();
                    host.statuses.clear();
                    host.null_outputs.clear();
                    const auto index = static_cast<std::size_t>(indices[n]);
                    host.statuses[key(baseline[index])] = status;
                    const int result = action(plugin, kOfxActionDescribe, host.effect_handle());
                    const bool stop = !accepted(status) && !is_optional;
                    const int expected_result = stop ? (is_acquisition ? status : property_result(status)) : kOfxStatOK;
                    require(result == expected_result, id + ": status");
                    auto calls = baseline;
                    if (stop)
                        calls.erase(calls.begin() + static_cast<std::ptrdiff_t>(index + 1), calls.end());
                    check_calls(host, calls, id);
                }
        };
        run_matrix("acquire", kAcquisition, std::array{0, 1, 2}, false, true);
        run_matrix("required", kRequired, kRequiredIndices, false, false);
        run_matrix("optional", kOptional, kOptionalIndices, true, false);
        host.statuses.clear();
        host.null_outputs.clear();
        host.reset_trace();
        {
            require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatOK, "manifest Describe");
            check_calls(host, baseline, "manifest");
            nlohmann::json actual = {{"OfxPropType", nlohmann::json::array({"OfxTypeImageEffect"})}};
            for (const auto& call : host.calls) {
                if (call.operation == "string")
                    actual[call.property] = nlohmann::json::array({call.text});
                if (call.operation == "int")
                    actual[call.property] = nlohmann::json::array({call.integer});
            }
            std::ifstream in(descriptor.manifest_path);
            require(bool(in), "manifest open");
            nlohmann::json oracle;
            in >> oracle;
            require(actual == oracle.at("properties"), "manifest mismatch");
        }
        const auto run = [&](const std::string& id, auto configure, int result, const std::vector<Call>& calls, bool null_effect = false) {
            host.reset_trace();
            host.statuses.clear();
            host.null_outputs.clear();
            host.context_dimension = 0;
            host.depth_dimension = 0;
            configure();
            require(action(plugin, kOfxActionDescribe, null_effect ? nullptr : host.effect_handle()) == result, id + ": status");
            check_calls(host, calls, id);
        };
        run("null_effect", [] {}, kOfxStatErrBadHandle, {}, true);
        run("null_first", [&] {
            host.null_outputs["effect_first"] = true;
        },
            kOfxStatErrBadHandle,
            {baseline[0]});
        auto no_param = baseline;
        no_param.erase(no_param.begin() + 2);
        run("null_param", [&] {
            host.null_outputs["param_set"] = true;
        },
            kOfxStatOK,
            no_param);
        run("null_param_props", [&] {
            host.null_outputs["param_props"] = true;
        },
            kOfxStatOK,
            baseline);
        for (std::size_t i = 0; i < kAcquisition.size(); ++i) {
            auto prefix = baseline;
            prefix.erase(prefix.begin() + static_cast<std::ptrdiff_t>(i + 1), prefix.end());
            run(std::string("failed_null/") + kAcquisition[i], [&] {
                host.null_outputs[kAcquisition[i]] = true;
                host.statuses[kAcquisition[i]] = kOfxStatErrValue;
            },
                kOfxStatErrValue,
                prefix);
        }
        run("both_optional_fail", [&] {
            host.statuses[key(baseline[4])] = kOfxStatErrMemory;
            host.statuses[key(baseline[5])] = kOfxStatErrUnsupported;
        },
            kOfxStatOK,
            baseline);
        auto short_prefix = baseline;
        short_prefix.erase(short_prefix.begin() + 7, short_prefix.end());
        run("short_then_value", [&] {
            host.statuses[key(baseline[4])] = kOfxStatErrMemory;
            host.statuses[key(baseline[6])] = kOfxStatErrValue;
        },
            kOfxStatFailed,
            short_prefix);
        run("long_then_unknown", [&] {
            host.statuses[key(baseline[5])] = kOfxStatErrMemory;
            host.statuses[key(baseline[6])] = kOfxStatErrUnknown;
        },
            kOfxStatErrMissingHostFeature,
            short_prefix);
        run("append_dimensions", [&] {
            host.context_dimension = 2;
            host.depth_dimension = 3;
        },
            kOfxStatOK,
            expected(descriptor.label, 2, 3));
        host.statuses.clear();
        host.null_outputs.clear();
        host.context_dimension = host.depth_dimension = 0;
    }
} // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "expected descriptor manifest path");
    const DescriptorExpectation descriptor{"Juicer", argv[1]};
    const char* identifier = "com.juicer.Juicer";
    OfxPlugin* plugin = nullptr;
    {
        require(OfxGetNumberOfPlugins() == 1 && fj_ofx_plugin_count() == 1, "plugin count");
        plugin = fj_ofx_plugin(0);
        require(plugin && plugin == OfxGetPlugin(0) && plugin == fj_ofx_plugin(0) &&
                    !OfxGetPlugin(-1) && !OfxGetPlugin(1) && !OfxGetPlugin(std::numeric_limits<int>::min()) && !OfxGetPlugin(std::numeric_limits<int>::max()) &&
                    !fj_ofx_plugin(-1) && !fj_ofx_plugin(1) && !fj_ofx_plugin(std::numeric_limits<int>::min()) && !fj_ofx_plugin(std::numeric_limits<int>::max()),
                "discovery bounds or identity");
        require(plugin->pluginApi && std::string_view(plugin->pluginApi) == kOfxImageEffectPluginApi && plugin->apiVersion == 1 && plugin->pluginIdentifier && std::string_view(plugin->pluginIdentifier) == identifier && plugin->pluginVersionMajor == 1 && plugin->pluginVersionMinor == 0 && plugin->setHost && plugin->mainEntry, "plugin descriptor");
        {
            require_native_available();
        }
        {
            require(action(plugin, kOfxActionUnload) == kOfxStatOK, "initial unload");
        }
        {
            require_native_available();
        }
    }
    Host host;
    {
        plugin->setHost(nullptr);
        require(action(plugin, kOfxActionLoad) == kOfxStatErrBadHandle, "null binding status");
        OfxHost invalid = *host.binding();
        invalid.fetchSuite = nullptr;
        plugin->setHost(&invalid);
        require(action(plugin, kOfxActionLoad) == kOfxStatErrBadHandle, "null fetch status");
        plugin->setHost(host.binding());
        require(host.calls.empty(), "setHost made host call");
    }
    {
        const std::vector<std::string> load_ids{"property_suite", "effect_suite", "parameter_suite", "message_suite", "effect_properties", "effect_params", "parameter_properties", "property_dimension", "property_string", "property_int", "property_get_string", "property_double", "clip_define", "param_define", "property_get_pointer", "property_set_pointer", "property_get_double", "clip_handle", "param_handle", "param_property", "param_get", "param_set", "empty_message", "property_get_int", "clip_image", "release_image", "clip_region", "abort", "valid_message"};
        for (int row = 0; row < 28; ++row) {
            Host missing;
            switch (row) {
                case 0:
                    missing.missing_suite = kOfxPropertySuite;
                    break;
                case 1:
                    missing.missing_suite = kOfxImageEffectSuite;
                    break;
                case 2:
                    missing.missing_suite = kOfxParameterSuite;
                    break;
                case 3:
                    missing.missing_suite = kOfxMessageSuite;
                    break;
                case 4:
                    missing.effect.getPropertySet = nullptr;
                    break;
                case 5:
                    missing.effect.getParamSet = nullptr;
                    break;
                case 6:
                    missing.parameter.paramSetGetPropertySet = nullptr;
                    break;
                case 7:
                    missing.property.propGetDimension = nullptr;
                    break;
                case 8:
                    missing.property.propSetString = nullptr;
                    break;
                case 9:
                    missing.property.propSetInt = nullptr;
                    break;
                case 10:
                    missing.property.propGetString = nullptr;
                    break;
                case 11:
                    missing.property.propSetDouble = nullptr;
                    break;
                case 12:
                    missing.effect.clipDefine = nullptr;
                    break;
                case 13:
                    missing.parameter.paramDefine = nullptr;
                    break;
                case 14:
                    missing.property.propGetPointer = nullptr;
                    break;
                case 15:
                    missing.property.propSetPointer = nullptr;
                    break;
                case 16:
                    missing.property.propGetDouble = nullptr;
                    break;
                case 17:
                    missing.effect.clipGetHandle = nullptr;
                    break;
                case 18:
                    missing.parameter.paramGetHandle = nullptr;
                    break;
                case 19:
                    missing.parameter.paramGetPropertySet = nullptr;
                    break;
                case 20:
                    missing.parameter.paramGetValue = nullptr;
                    break;
                case 21:
                    missing.parameter.paramSetValue = nullptr;
                    break;
                case 22:
                    missing.message.message = nullptr;
                    break;
                case 23:
                    missing.property.propGetInt = nullptr;
                    break;
                case 24:
                    missing.effect.clipGetImage = nullptr;
                    break;
                case 25:
                    missing.effect.clipReleaseImage = nullptr;
                    break;
                case 26:
                    missing.effect.clipGetRegionOfDefinition = nullptr;
                    break;
                case 27:
                    missing.effect.abort = nullptr;
                    break;
            }
            plugin->setHost(missing.binding());
            require(action(plugin, kOfxActionLoad) == kOfxStatErrMissingHostFeature, "missing suite/slot row " + std::to_string(row));
            require(!missing.callback_error, "unexpected suite call");
            require_native_available();
            require(action(plugin, kOfxActionUnload) == kOfxStatOK, "failed load cleanup");
        }
        int nested_load = -1;
        host.reentry = [&] {
            nested_load = action(plugin, kOfxActionLoad);
        };
        plugin->setHost(host.binding());
        require(action(plugin, kOfxActionLoad) == kOfxStatOK, "load with message suite");
        require(nested_load == kOfxStatFailed, "reentrant load status");
        check_calls(host, {{"fetch", kOfxPropertySuite, 1, {}, 0, 0}, {"fetch", kOfxImageEffectSuite, 1, {}, 0, 0}, {"fetch", kOfxParameterSuite, 1, {}, 0, 0}, {"fetch", kOfxMessageSuite, 1, {}, 0, 0}}, "suite fetches");
        require(action(plugin, kOfxActionLoad) == kOfxStatFailed, "duplicate load");
    }
    {
        describe(plugin, host, descriptor);
    }
    {
        host.reset_trace();
        require(action(plugin, "UnknownAction") == kOfxStatReplyDefault, "unknown action");
        require(action(plugin, kOfxImageEffectActionRender) == kOfxStatErrBadHandle, "render requires effect and input");
        require(action(plugin, kOfxImageEffectActionRender, host.effect_handle()) == kOfxStatErrBadHandle, "render requires input");
        require(action(plugin, nullptr) == kOfxStatErrBadHandle, "null action");
        require(host.calls.empty(), "default action called host");
    }
    {
        host.reset_trace();
        int nested = -1;
        host.reentry = [&] {
            nested = action(plugin, kOfxActionUnload);
        };
        require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatOK && nested == kOfxStatErrFatal, "reentrant unload");
        check_calls(host, expected(descriptor.label), "reentry");
    }
    {
        require(action(plugin, kOfxActionUnload) == kOfxStatOK, "unload for reload");
        host.reset_trace();
        require(action(plugin, kOfxActionLoad) == kOfxStatOK && host.calls.size() == 4, "same host reload");
        {
            auto original = host.property.propSetString;
            host.property.propSetString = nullptr;
            host.reset_trace();
            const int status = action(plugin, kOfxActionDescribe, host.effect_handle());
            host.property.propSetString = original;
            require(status == kOfxStatOK, "loaded suite table was not copied");
            check_calls(host, expected(descriptor.label), "suite copy");
        }
        {
            Host rejected;
            plugin->setHost(rejected.binding());
            host.reset_trace();
            require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatFailed, "live rebind did not block ordinary action");
            require(host.calls.empty() && rejected.calls.empty(), "live rebind made host call");
            require(action(plugin, kOfxActionUnload) == kOfxStatOK, "live rebind lost cleanup owner");
        }
        Host fresh;
        plugin->setHost(fresh.binding());
        require(fresh.calls.empty(), "rebind called host");
        require(action(plugin, kOfxActionLoad) == kOfxStatOK && fresh.calls.size() == 4, "fresh host reload");
        require(action(plugin, kOfxActionUnload) == kOfxStatOK, "fresh host cleanup");
    }
    {
        require(action(plugin, kOfxActionUnload) == kOfxStatOK && action(plugin, kOfxActionUnload) == kOfxStatOK, "empty unload");
        require_native_available();
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "module fixture: %s\n", error.what());
    return 1;
}
