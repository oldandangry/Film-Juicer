#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "instance_host.h"
#include "juicer_ofx_api.h"
#include "ofxsImageEffect.h"
#include "../ffi/case_results.h"

namespace {
    constexpr std::string_view kCaller = "ofx.instance";
    constexpr const char* kCases[]{"reference", "create", "handles", "actions", "parameters", "edits", "reentry", "destroy"};
    using ofx_test::InstanceCall;
    using ofx_test::InstanceHost;
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    OfxStatus call(OfxPlugin* plugin, const char* action, const void* handle = nullptr, OfxPropertySetHandle input = nullptr) {
        return plugin->mainEntry(action, handle, input, nullptr);
    }
    std::vector<std::string> handles(const std::vector<InstanceCall>& calls, std::string_view operation) {
        std::vector<std::string> result;
        for (const auto& entry : calls)
            if (entry.operation == operation)
                result.push_back(entry.property);
        return result;
    }
    bool constructor_census(const std::vector<InstanceCall>& calls) {
        const auto params = handles(calls, "param_handle");
        const auto clips = handles(calls, "clip_handle");
        if (params.size() != ofx_test::kInstanceParameters.size() || clips.size() != ofx_test::kInstanceClips.size())
            return false;
        for (std::size_t i = 0; i < params.size(); ++i)
            if (params[i] != ofx_test::kInstanceParameters[i].name)
                return false;
        for (std::size_t i = 0; i < clips.size(); ++i)
            if (clips[i] != ofx_test::kInstanceClips[i])
                return false;
        return true;
    }
    bool create_equivalent(std::vector<InstanceCall> actual, std::vector<InstanceCall> expected) {
        const auto publication = [](const InstanceCall& call) {
            return call.operation == "set_pointer" && call.property == kOfxPropInstanceData &&
                   call.values.size() == 1 && std::holds_alternative<void*>(call.values[0]) &&
                   std::get<void*>(call.values[0]) != nullptr;
        };
        if (std::count_if(actual.begin(), actual.end(), publication) != 1 ||
            std::count_if(expected.begin(), expected.end(), publication) != 1)
            return false;
        const auto actual_write = std::find_if(actual.begin(), actual.end(), publication);
        const auto expected_write = std::find_if(expected.begin(), expected.end(), publication);
        const auto actual_position = static_cast<std::size_t>(actual_write - actual.begin());
        const auto expected_position = static_cast<std::size_t>(expected_write - expected.begin());
        if (expected_position >= actual_position)
            return false;
        actual.erase(actual_write);
        expected.erase(expected_write);
        if (actual == expected)
            return true;
        // Candidate checks the existing instance pointer before creating an owner.
        const auto is_read = [](const InstanceCall& call) {
            return call.operation == "get_pointer" && call.property == kOfxPropInstanceData;
        };
        if (actual.size() != expected.size() + 1 || std::count_if(actual.begin(), actual.end(), is_read) != 1 ||
            std::any_of(expected.begin(), expected.end(), is_read))
            return false;
        const auto read = std::find_if(actual.begin(), actual.end(), is_read);
        if (static_cast<std::size_t>(read - actual.begin()) >= actual_position)
            return false;
        actual.erase(read);
        return actual == expected;
    }
    bool failed_create_equivalent(std::vector<InstanceCall> actual, std::vector<InstanceCall> expected) {
        const auto is_write = [](const InstanceCall& call) {
            return call.operation == "set_pointer" && call.property == kOfxPropInstanceData;
        };
        const auto is_read = [](const InstanceCall& call) {
            return call.operation == "get_pointer" && call.property == kOfxPropInstanceData;
        };
        if (std::any_of(actual.begin(), actual.end(), is_write) || std::count_if(expected.begin(), expected.end(), is_write) != 2)
            return false;
        auto publish = std::find_if(expected.begin(), expected.end(), is_write);
        if (publish->values.size() != 1 || !std::holds_alternative<void*>(publish->values[0]) || !std::get<void*>(publish->values[0]))
            return false;
        auto clear = std::find_if(std::next(publish), expected.end(), is_write);
        if (clear == expected.end() || clear->values.size() != 1 || !std::holds_alternative<void*>(clear->values[0]) || std::get<void*>(clear->values[0]))
            return false;
        if (std::any_of(expected.begin(), publish, [](const auto& call) {
                return call.operation == "param_handle";
            }) ||
            std::any_of(std::next(clear), expected.end(), [](const auto& call) {
                return call.operation == "param_handle";
            }))
            return false;
        expected.erase(clear);
        expected.erase(publish);
        if (actual == expected)
            return true;
        if (actual.size() != expected.size() + 1 || std::count_if(actual.begin(), actual.end(), is_read) != 1 ||
            std::any_of(expected.begin(), expected.end(), is_read))
            return false;
        auto read = std::find_if(actual.begin(), actual.end(), is_read);
        if (std::any_of(actual.begin(), read, [](const auto& call) {
                return call.operation == "param_handle";
            }))
            return false;
        actual.erase(read);
        return actual == expected;
    }
    struct HandleFault {
        std::string name;
        int injected;
        OfxStatus status;
        std::vector<std::string> params;
        std::vector<InstanceCall> calls;
    };
    template <class F>
    void with_row(const JuicerTests::Cohort* rows, const char* id, F&& body) {
        if (rows) {
            auto row = rows->case_event(id);
            body();
            row.passed();
        } else
            body();
    }
    std::vector<HandleFault> handle_faults(OfxPlugin* plugin, InstanceHost& host, bool include_null, const JuicerTests::Cohort* rows = nullptr, const std::vector<HandleFault>* expected = nullptr) {
        std::vector<HandleFault> results;
        auto append = [&](HandleFault value) {
            if (expected) {
                const auto& r = expected->at(results.size());
                require(value.name == r.name && value.injected == r.injected && value.status == r.status && value.params == r.params, "handle fault membership");
                require(value.status == kOfxStatOK ? create_equivalent(value.calls, r.calls) : failed_create_equivalent(value.calls, r.calls), "handle fault full continuation");
                auto shortened = value.calls;
                require(!shortened.empty(), "handle continuation boundary");
                shortened.pop_back();
                require(!(value.status == kOfxStatOK ? create_equivalent(shortened, r.calls) : failed_create_equivalent(shortened, r.calls)), "handle premature success negative control");
                auto extended = value.calls;
                extended.push_back(value.calls.back());
                require(!(value.status == kOfxStatOK ? create_equivalent(extended, r.calls) : failed_create_equivalent(extended, r.calls)), "handle late stop negative control");
            }
            results.push_back(std::move(value));
        };
        const std::array names{"Exposure", "GrainParticleScale", "PrintShadowCompensationTransition"};
        const std::array ids{"first", "middle", "last"};
        for (std::size_t index = 0; index < names.size(); ++index)
            with_row(rows, ids[index], [&] {
                const char* name = names[index];
                for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status) {
                    auto* effect = host.create_effect();
                    host.clear_calls();
                    host.fail_next(std::string("param_handle/") + name, status);
                    const auto actual = call(plugin, kOfxActionCreateInstance, effect);
                    auto seen = handles(host.calls(), "param_handle");
                    auto calls = host.calls();
                    require(!seen.empty() && std::find(seen.begin(), seen.end(), name) != seen.end(), "fault boundary reached");
                    if (actual == kOfxStatOK)
                        require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "fault success cleanup");
                    else
                        require(host.instance_data(effect) == nullptr, "fault rollback");
                    append({name, status, actual, std::move(seen), std::move(calls)});
                }
            });
        const std::array acquisition_ids{"source", "output", "parameter_properties", "parameter_type"};
        const std::array acquisition_faults{"clip_handle/Source", "clip_handle/Output", "param_properties/Exposure", "get_string/OfxParamPropType"};
        for (std::size_t index = 0; index < acquisition_ids.size(); ++index)
            with_row(rows, acquisition_ids[index], [&] {
                const char* operation = acquisition_faults[index];
                for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status) {
                    auto* effect = host.create_effect();
                    host.clear_calls();
                    host.fail_next(operation, status);
                    const auto actual = call(plugin, kOfxActionCreateInstance, effect);
                    auto seen = handles(host.calls(), "param_handle");
                    auto calls = host.calls();
                    const auto slash = std::string_view(operation).find('/');
                    require(std::any_of(calls.begin(), calls.end(), [&](const auto& c) {
                                return c.operation == std::string_view(operation).substr(0, slash) &&
                                       (c.property == std::string_view(operation).substr(slash + 1) || c.target == std::string_view(operation).substr(slash + 1)) && c.status == status;
                            }),
                            "acquisition fault boundary reached");
                    if (actual == kOfxStatOK)
                        require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "acquisition fault cleanup");
                    else
                        require(host.instance_data(effect) == nullptr, "acquisition fault rollback");
                    append({operation, status, actual, std::move(seen), std::move(calls)});
                }
            });
        const std::array typed_names{"Exposure", "CameraAutoExposure", "CameraFilmFormatPreset", "FilmProfileKey", "ScannerLutResolution", "ScannerUnsharpMask", "CouplersLangmuirDonorKRgb", "GrainResetAdvanced"};
        const std::array typed_ids{"double", "boolean", "choice", "string_choice", "integer", "double2", "double3", "push_button"};
        with_row(rows, "type", [&] {
            for (std::size_t index = 0; index < typed_names.size(); ++index)
                with_row(rows, typed_ids[index], [&] {
                    const char* name = typed_names[index];
                    auto* effect = host.create_effect();
                    host.set_parameter_type(effect, name, "InvalidParameterType");
                    host.clear_calls();
                    const auto actual = call(plugin, kOfxActionCreateInstance, effect);
                    auto seen = handles(host.calls(), "param_handle");
                    auto calls = host.calls();
                    require(actual != kOfxStatOK && host.instance_data(effect) == nullptr, "type mismatch rollback");
                    append({name, -1, actual, std::move(seen), std::move(calls)});
                });
        });
        if (include_null)
            with_row(rows, "null", [&] {
                for (const auto* operation : {"param_handle/Exposure", "param_handle/PrintShadowCompensationTransition", "clip_handle/Source", "clip_handle/Output", "param_properties/Exposure"}) {
                    auto* effect = host.create_effect();
                    host.clear_calls();
                    host.null_next(operation);
                    const auto actual = call(plugin, kOfxActionCreateInstance, effect);
                    auto seen = handles(host.calls(), "param_handle");
                    auto calls = host.calls();
                    const auto slash = std::string_view(operation).find('/');
                    require(actual == kOfxStatErrBadHandle && host.instance_data(effect) == nullptr, "null handle rollback");
                    require(!calls.empty() && calls.back().operation == std::string_view(operation).substr(0, slash) &&
                                (calls.back().property == std::string_view(operation).substr(slash + 1) || calls.back().target == std::string_view(operation).substr(slash + 1)),
                            "null handle exact terminal call");
                    results.push_back({operation, -2, actual, std::move(seen), std::move(calls)});
                }
            });
        return results;
    }
    struct EditScenario {
        const char* id;
        const char* name;
        std::vector<ofx_test::InstanceValue> authored;
        const char* observed;
    };
    const std::vector<EditScenario>& edit_scenarios() {
        static const std::vector<EditScenario> cases{
            {"camera_manual", "CameraFilmFormatMm", {40.0}, "CameraFilmFormatPreset"},
            {"film_route", "FilmProfileKey", {std::string()}, "ScanRoute"},
            {"gamma_manual", "CouplersGammaSameLayerRgb", {0.2, 0.3, 0.4}, "CouplersGammaUseStock"},
            {"stock_defaults", "CouplersGammaUseStock", {1}, "CouplersGammaSameLayerRgb"},
            {"grain_preset", "GrainPreset", {1}, "GrainAmplitude"},
            {"grain_preset_fine", "GrainPreset", {0}, "GrainAmplitude"},
            {"grain_preset_coarse", "GrainPreset", {2}, "GrainAmplitude"},
            {"grain_reset", "GrainResetAdvanced", {}, "GrainParticleScaleMaster"},
            {"grain_masters", "GrainParticleScaleMaster", {1.5}, "GrainParticleScale"},
            {"grain_layers_master", "GrainParticleScaleLayersMaster", {1.6}, "GrainParticleScaleLayers"},
            {"grain_density_master", "GrainDensityMinMaster", {0.2}, "GrainDensityMin"},
            {"grain_uniformity_master", "GrainUniformityMaster", {0.8}, "GrainUniformity"},
            {"grain_linked", "GrainSharpness", {0.5}, "GrainTexture"}};
        return cases;
    }
    struct ActionScenario {
        const char* id;
        const char* reason;
        const char* type;
        const char* name;
        double time;
        bool null_args;
        bool fail_reason;
    };
    const std::array kActionScenarios{
        ActionScenario{"unrelated", kOfxChangeUserEdited, kOfxTypeClip, "Source", 0.0, false, false},
        ActionScenario{"plugin_edit", kOfxChangePluginEdited, kOfxTypeParameter, "Exposure", 0.0, false, false},
        ActionScenario{"time", kOfxChangeTime, kOfxTypeParameter, "Exposure", 123.0, false, false},
        ActionScenario{"null_args", kOfxChangeUserEdited, kOfxTypeParameter, "Exposure", 0.0, true, false},
        ActionScenario{"reason", "unknown-change-reason", kOfxTypeParameter, "Exposure", 0.0, false, false},
        ActionScenario{"name", kOfxChangeUserEdited, kOfxTypeParameter, "UnrelatedParameter", 0.0, false, false},
        ActionScenario{"status", kOfxChangeUserEdited, kOfxTypeParameter, "Exposure", 0.0, false, true}};
    struct ActionObservation {
        OfxStatus status;
        std::vector<InstanceCall> calls;
    };
    bool same_action(const ActionObservation& actual, const ActionObservation& expected) {
        return actual.status == expected.status && actual.calls == expected.calls;
    }
    ActionObservation run_action(OfxPlugin* plugin, InstanceHost& host, OfxImageEffectHandle effect, const ActionScenario& scenario) {
        auto* args = scenario.null_args ? nullptr : host.change_args({scenario.reason, scenario.type, scenario.name}, scenario.time);
        host.clear_calls();
        if (scenario.fail_reason)
            host.fail_next("get_string/OfxPropChangeReason", kOfxStatErrMemory);
        const auto status = call(plugin, kOfxActionInstanceChanged, effect, args);
        return {status, host.calls()};
    }
    struct EditObservation {
        OfxStatus status;
        std::vector<InstanceCall> calls;
        std::vector<ofx_test::InstanceValue> value;
    };
    OfxStatus change(OfxPlugin* plugin, InstanceHost& host, OfxImageEffectHandle effect, const char* reason, const char* name);
    struct ControlFault {
        const char* operation;
        int injected;
        OfxStatus status;
        std::string scope;
        std::vector<InstanceCall> calls;
    };
    bool same_effect_trace(const ControlFault& actual, const ControlFault& expected);
    struct ControlScenario {
        const char* id;
        const char* operation;
        const char* changed;
    };
    const std::array kControlScenarios{
        ControlScenario{"get", "param_get/Exposure", "Exposure"},
        ControlScenario{"set", "param_set/CameraFilmFormatMm", "CameraFilmFormatPreset"},
        ControlScenario{"get_boolean", "param_get/CameraAutoExposure", "Exposure"},
        ControlScenario{"get_choice", "param_get/CameraMeteringMethod", "Exposure"},
        ControlScenario{"get_integer", "param_get/ScannerLutResolution", "Exposure"},
        ControlScenario{"get_pair", "param_get/ScannerUnsharpMask", "Exposure"},
        ControlScenario{"get_triplet", "param_get/GrainParticleScale", "Exposure"},
        ControlScenario{"get_string", "param_get/FilmProfileKey", "Exposure"},
        ControlScenario{"set_boolean", "param_set/CouplersGammaUseStock", "CouplersGammaSameLayerRgb"},
        ControlScenario{"set_integer", "param_set/CameraFilmFormatPreset", "CameraFilmFormatMm"},
        ControlScenario{"set_pair", "param_set/GrainMicroStructure", "GrainResetAdvanced"},
        ControlScenario{"set_triplet", "param_set/GrainParticleScale", "GrainParticleScaleMaster"},
        ControlScenario{"set_string", "param_set/ScanRoute", "FilmProfileKey"},
        ControlScenario{"property_enabled", "set_int/OfxParamPropEnabled", "Exposure"},
        ControlScenario{"property_secret", "set_int/OfxParamPropSecret", "Exposure"},
        ControlScenario{"hint_set", "set_string/OfxParamPropHint", "Exposure"}};
    std::vector<std::string> control_fault_ids() {
        std::vector<std::string> ids;
        for (const auto& scenario : kControlScenarios)
            for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status)
                ids.push_back(std::string(scenario.id) + "/status_" + std::to_string(status));
        return ids;
    }
    std::vector<ControlFault> control_faults(OfxPlugin* plugin, InstanceHost& host, const JuicerTests::Cohort* rows = nullptr, const std::vector<ControlFault>* expected = nullptr) {
        std::vector<ControlFault> observations;
        for (const auto& scenario : kControlScenarios) {
            auto* effect = host.create_effect();
            require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "control fault create");
            host.set_value(effect, "GrainActive", {1});
            host.set_value(effect, "CameraAutoExposure", {1});
            for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status) {
                const auto id = std::string(scenario.id) + "/status_" + std::to_string(status);
                with_row(rows, id.c_str(), [&] {
                    if (std::string_view(scenario.changed) == "CameraFilmFormatPreset")
                        host.set_value(effect, scenario.changed, {1});
                    host.clear_calls();
                    host.fail_next(scenario.operation, status);
                    const auto result = change(plugin, host, effect, kOfxChangeUserEdited, scenario.changed);
                    ControlFault observation{scenario.operation, status, result, host.effect_scope(effect), host.calls()};
                    const auto operation = std::string_view(scenario.operation);
                    const auto slash = operation.find('/');
                    require(std::any_of(observation.calls.begin(), observation.calls.end(), [&](const auto& c) {
                                return c.operation == operation.substr(0, slash) && (c.target == operation.substr(slash + 1) || c.property == operation.substr(slash + 1)) && c.status == status;
                            }),
                            "control fault targeted operation");
                    if (expected)
                        require(same_effect_trace(observation, expected->at(observations.size())), "actual callback get/set full status continuation");
                    observations.push_back(std::move(observation));
                });
            }
            require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "control fault destroy");
        }
        return observations;
    }
    bool same_effect_trace(const ControlFault& actual, const ControlFault& expected) {
        if (std::string_view(actual.operation) != expected.operation || actual.injected != expected.injected || actual.status != expected.status || actual.calls.size() != expected.calls.size())
            return false;
        for (std::size_t i = 0; i < actual.calls.size(); ++i) {
            auto a = actual.calls[i];
            const auto& e = expected.calls[i];
            if (a.scope == actual.scope)
                a.scope = expected.scope;
            if (a.target == actual.scope)
                a.target = expected.scope;
            else if (a.target == actual.scope + ".params")
                a.target = expected.scope + ".params";
            if (a != e)
                return false;
        }
        return true;
    }
    struct TypedScenario {
        const char* id;
        const char* operation;
        const char* target;
        const char* changed;
        bool fail;
    };
    const std::array kTypedScenarios{
        TypedScenario{"get_scalar", "param_get", "Exposure", "Exposure", false},
        TypedScenario{"get_integer", "param_get", "ScannerLutResolution", "Exposure", false},
        TypedScenario{"get_boolean", "param_get", "CameraAutoExposure", "Exposure", false},
        TypedScenario{"get_choice", "param_get", "CameraMeteringMethod", "Exposure", false},
        TypedScenario{"get_pair", "param_get", "ScannerUnsharpMask", "Exposure", false},
        TypedScenario{"get_triplet", "param_get", "GrainParticleScale", "Exposure", false},
        TypedScenario{"get_string_copy", "param_get", "FilmProfileKey", "Exposure", false},
        TypedScenario{"set_scalar", "param_set", "CameraFilmFormatMm", "CameraFilmFormatPreset", false},
        TypedScenario{"set_integer", "param_set", "CameraFilmFormatPreset", "CameraFilmFormatMm", false},
        TypedScenario{"set_boolean", "param_set", "CouplersGammaUseStock", "CouplersGammaSameLayerRgb", false},
        TypedScenario{"set_pair", "param_set", "GrainMicroStructure", "GrainResetAdvanced", false},
        TypedScenario{"set_triplet", "param_set", "GrainParticleScale", "GrainParticleScaleMaster", false},
        TypedScenario{"set_string_copy", "param_set", "ScanRoute", "FilmProfileKey", false},
        TypedScenario{"fail", "param_get", "Exposure", "Exposure", true}};
    std::vector<ControlFault> typed_controls(OfxPlugin* plugin, InstanceHost& host, const JuicerTests::Cohort* rows = nullptr, const std::vector<ControlFault>* expected = nullptr) {
        std::vector<ControlFault> observations;
        auto* effect = host.create_effect();
        require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "typed consumer create");
        host.set_value(effect, "GrainActive", {1});
        host.set_value(effect, "CameraAutoExposure", {1});
        for (const auto& scenario : kTypedScenarios)
            with_row(rows, scenario.id, [&] {
                if (std::string_view(scenario.changed) == "CameraFilmFormatPreset")
                    host.set_value(effect, scenario.changed, {1});
                host.clear_calls();
                if (scenario.fail)
                    host.fail_next("param_get/Exposure", kOfxStatErrMemory);
                const auto result = change(plugin, host, effect, kOfxChangeUserEdited, scenario.changed);
                ControlFault observation{scenario.id, scenario.fail ? kOfxStatErrMemory : kOfxStatOK, result, host.effect_scope(effect), host.calls()};
                auto found = std::find_if(observation.calls.begin(), observation.calls.end(), [&](const auto& c) {
                    return c.operation == scenario.operation && c.target == scenario.target;
                });
                require(found != observation.calls.end() && found->status == observation.injected, "typed candidate callback operation");
                if (!scenario.fail) {
                    const auto* const kind = std::find_if(ofx_test::kInstanceParameters.data(), ofx_test::kInstanceParameters.data() + ofx_test::kInstanceParameters.size(), [&](const auto& entry) {
                        return entry.name == scenario.target;
                    });
                    require(kind != ofx_test::kInstanceParameters.data() + ofx_test::kInstanceParameters.size(), "typed census lookup");
                    const auto arity = kind->kind == ofx_test::InstanceParamKind::Double2D ? 2u : kind->kind == ofx_test::InstanceParamKind::Double3D ? 3u
                                                                                                                                                      : 1u;
                    require(found->values.size() == arity, "typed callback arity");
                    if (kind->kind == ofx_test::InstanceParamKind::StringChoice)
                        require(std::holds_alternative<std::string>(found->values.front()), "typed string bytes");
                    else if (kind->kind == ofx_test::InstanceParamKind::Boolean || kind->kind == ofx_test::InstanceParamKind::Choice || kind->kind == ofx_test::InstanceParamKind::Integer)
                        require(std::holds_alternative<int>(found->values.front()), "typed integer value");
                    else
                        require(std::all_of(found->values.begin(), found->values.end(), [](const auto& value) {
                                    return std::holds_alternative<double>(value);
                                }),
                                "typed floating values");
                }
                if (std::string_view(scenario.id) == "set_string_copy")
                    require(host.value(effect, scenario.target) == found->values, "candidate string setter retained copied bytes");
                if (expected)
                    require(same_effect_trace(observation, expected->at(observations.size())), "typed actual callback continuation");
                observations.push_back(std::move(observation));
            });
        require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "typed consumer destroy");
        return observations;
    }
    struct ContextScenario {
        const char* id;
        const char* context;
    };
    const std::array kContextScenarios{
        ContextScenario{"context_filter", kOfxImageEffectContextFilter},
        ContextScenario{"context_generator", kOfxImageEffectContextGenerator},
        ContextScenario{"context_transition", kOfxImageEffectContextTransition},
        ContextScenario{"context_paint", kOfxImageEffectContextPaint},
        ContextScenario{"context_general", kOfxImageEffectContextGeneral},
        ContextScenario{"context_retimer", kOfxImageEffectContextRetimer},
        ContextScenario{"context_unknown", "unknown-context"}};
    struct ContextObservation {
        OfxStatus status;
        std::string scope;
        std::vector<InstanceCall> calls;
    };
    std::vector<ContextObservation> context_cases(OfxPlugin* plugin, InstanceHost& host, const JuicerTests::Cohort* rows = nullptr, const std::vector<ContextObservation>* expected = nullptr) {
        std::vector<ContextObservation> observations;
        for (const auto& scenario : kContextScenarios)
            with_row(rows, scenario.id, [&] {
                auto* effect = host.create_effect();
                host.set_effect_context(effect, scenario.context);
                host.clear_calls();
                const auto status = call(plugin, kOfxActionCreateInstance, effect);
                ContextObservation actual{status, host.effect_scope(effect), host.calls()};
                const bool unknown = std::string_view(scenario.id) == "context_unknown";
                require(status == (unknown ? kOfxStatFailed : kOfxStatOK), "context create status");
                if (expected) {
                    const auto& ref = expected->at(observations.size());
                    auto calls = actual.calls;
                    for (auto& c : calls) {
                        if (c.scope == actual.scope)
                            c.scope = ref.scope;
                        if (c.target == actual.scope)
                            c.target = ref.scope;
                        else if (c.target == actual.scope + ".params")
                            c.target = ref.scope + ".params";
                    }
                    require(actual.status == ref.status && (unknown ? calls == ref.calls : create_equivalent(calls, ref.calls)), "context full continuation");
                }
                if (unknown)
                    require(host.instance_data(effect) == nullptr && actual.calls.size() == 2 && actual.calls.back().operation == "get_string" && actual.calls.back().property == kOfxImageEffectPropContext, "unknown context stop");
                else
                    require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "context destroy");
                observations.push_back(std::move(actual));
            });
        if (rows)
            with_row(rows, "context_null", [&] {
                auto* effect = host.create_effect();
                host.null_next("get_string/OfxImageEffectPropContext");
                host.clear_calls();
                require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatErrBadHandle && host.instance_data(effect) == nullptr, "null context status");
                require(host.calls().size() == 2 && host.calls().back().operation == "get_string" && host.calls().back().property == kOfxImageEffectPropContext, "null context stop");
            });
        return observations;
    }
    struct AcquisitionScenario {
        const char* id;
        const char* operation;
        int occurrence;
    };
    const std::array kAcquisitionScenarios{
        AcquisitionScenario{"effect_first", "effect_properties", 1},
        AcquisitionScenario{"effect_second", "effect_properties", 2},
        AcquisitionScenario{"param_set", "param_set", 1},
        AcquisitionScenario{"param_set_properties", "param_set_properties", 1}};
    std::vector<std::string> create_ids() {
        std::vector<std::string> ids{"happy", "duplicate", "before_publication", "after_publication", "null_effect", "null_properties", "status", "context_filter", "context_generator", "context_transition", "context_paint", "context_general", "context_retimer", "context_unknown", "context_null"};
        for (const auto& scenario : kAcquisitionScenarios)
            for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status)
                ids.push_back(std::string(scenario.id) + "/status_" + std::to_string(status));
        for (const auto& scenario : kAcquisitionScenarios)
            ids.push_back(std::string(scenario.id) + "/null");
        return ids;
    }
    std::vector<InstanceCall> expected_prepublication(const std::vector<InstanceCall>& native) {
        auto expected = native;
        const auto publication = [](const InstanceCall& c) {
            return c.operation == "set_pointer" && c.property == kOfxPropInstanceData;
        };
        require(std::count_if(expected.begin(), expected.end(), publication) == 1, "native publication membership");
        auto write = std::find_if(expected.begin(), expected.end(), publication);
        require(write->values.size() == 1 && std::holds_alternative<void*>(write->values[0]) && std::get<void*>(write->values[0]), "native publication value");
        expected.erase(write);
        require(expected.size() >= 4 && expected[0].operation == "effect_properties" && expected[1].operation == "get_string" && expected[1].property == kOfxImageEffectPropContext && expected[2].operation == "effect_properties" && expected[3].operation == "get_string" && expected[3].property == kOfxImageEffectPropContext, "native constructor acquisition prefix");
        expected.insert(expected.begin() + 2, InstanceCall{"get_pointer", expected[0].target, kOfxPropInstanceData, 0, {}, {}});
        return expected;
    }
    std::size_t acquisition_index(const std::vector<InstanceCall>& calls, const AcquisitionScenario& scenario) {
        int remaining = scenario.occurrence;
        for (std::size_t i = 0; i < calls.size(); ++i)
            if (calls[i].operation == scenario.operation && --remaining == 0)
                return i;
        throw std::runtime_error("missing acquisition in native constructor");
    }
    void acquisition_diagnostic(const AcquisitionScenario& scenario, int injected, OfxStatus result, std::vector<InstanceCall> actual, std::vector<InstanceCall> expected, bool successful) {
        if (successful) {
            const auto publication = [](const auto& c) {
                return c.operation == "set_pointer" && c.property == kOfxPropInstanceData;
            };
            const auto duplicate_read = [](const auto& c) {
                return c.operation == "get_pointer" && c.property == kOfxPropInstanceData;
            };
            auto a = std::find_if(actual.begin(), actual.end(), publication);
            if (a != actual.end())
                actual.erase(a);
            auto e = std::find_if(expected.begin(), expected.end(), publication);
            if (e != expected.end())
                expected.erase(e);
            a = std::find_if(actual.begin(), actual.end(), duplicate_read);
            if (a != actual.end())
                actual.erase(a);
        }
        std::size_t i = 0;
        while (i < actual.size() && i < expected.size() && actual[i] == expected[i])
            ++i;
        std::fprintf(stderr, "acquisition %s operation=%s occurrence=%d injected=%d result=%d first_difference=%zu actual_count=%zu expected_count=%zu\n", scenario.id, scenario.operation, scenario.occurrence, injected, result, i, actual.size(), expected.size());
        const auto print = [](const char* side, const InstanceCall& c) {
            std::fprintf(stderr, "%s operation=%s target=%s property=%s index=%d status=%d scope=%s values=%zu\n", side, c.operation.c_str(), c.target.c_str(), c.property.c_str(), c.index, c.status, c.scope.c_str(), c.values.size());
        };
        if (i < actual.size())
            print("actual", actual[i]);
        if (i < expected.size())
            print("expected", expected[i]);
    }
    void acquisition_cases(OfxPlugin* plugin, InstanceHost& host, const JuicerTests::Cohort& rows, const std::vector<InstanceCall>& native) {
        const auto prefix = expected_prepublication(native);
        const std::string native_scope = native.front().target;
        for (const auto& scenario : kAcquisitionScenarios) {
            for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status) {
                const auto id = std::string(scenario.id) + "/status_" + std::to_string(status);
                auto row = rows.case_event(id);
                auto* effect = host.create_effect();
                const auto scope = host.effect_scope(effect);
                host.clear_calls();
                host.fail_on_nth(scenario.occurrence, scenario.operation, status);
                const auto result = call(plugin, kOfxActionCreateInstance, effect);
                auto actual = host.calls();
                for (auto& c : actual) {
                    if (c.scope == scope)
                        c.scope = native_scope;
                    if (c.target == scope)
                        c.target = native_scope;
                    else if (c.target == scope + ".params")
                        c.target = native_scope + ".params";
                }
                const bool accepted = status == kOfxStatOK || status == kOfxStatReplyYes || status == kOfxStatReplyNo || status == kOfxStatReplyDefault;
                if (accepted) {
                    auto full = native;
                    full[acquisition_index(full, scenario)].status = status;
                    const bool equal = result == kOfxStatOK && create_equivalent(actual, full);
                    if (!equal)
                        acquisition_diagnostic(scenario, status, result, actual, full, true);
                    require(equal, "accepted acquisition full continuation");
                    require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "accepted acquisition destroy");
                } else {
                    auto stopped = prefix;
                    const auto index = acquisition_index(stopped, scenario);
                    stopped[index].status = status;
                    stopped.resize(index + 1);
                    const bool equal = result == status && host.instance_data(effect) == nullptr && actual == stopped;
                    if (!equal)
                        acquisition_diagnostic(scenario, status, result, actual, stopped, false);
                    require(equal, "failed acquisition exact prefix");
                }
                row.passed();
            }
        }
        for (const auto& scenario : kAcquisitionScenarios) {
            const auto id = std::string(scenario.id) + "/null";
            auto row = rows.case_event(id);
            auto* effect = host.create_effect();
            const auto scope = host.effect_scope(effect);
            host.clear_calls();
            host.null_on_nth(scenario.operation, scenario.occurrence);
            const auto status = call(plugin, kOfxActionCreateInstance, effect);
            auto actual = host.calls();
            for (auto& c : actual) {
                if (c.scope == scope)
                    c.scope = native_scope;
                if (c.target == scope)
                    c.target = native_scope;
                else if (c.target == scope + ".params")
                    c.target = native_scope + ".params";
            }
            auto stopped = prefix;
            stopped.resize(acquisition_index(stopped, scenario) + 1);
            const bool equal = status == kOfxStatErrBadHandle && host.instance_data(effect) == nullptr && actual == stopped;
            if (!equal)
                acquisition_diagnostic(scenario, -1, status, actual, stopped, false);
            require(equal, "null acquisition exact prefix");
            row.passed();
        }
    }
    struct Reference {
        std::vector<HandleFault> faults;
        std::vector<ControlFault> controls;
        std::vector<ControlFault> typed;
        std::vector<ContextObservation> contexts;
        std::vector<EditObservation> edits;
        std::vector<ActionObservation> actions;
        std::vector<ActionObservation> nested_changes;
        std::vector<InstanceCall> create, ordinary_create, begin, change, end, edit, reentry, destroy;
        std::vector<ofx_test::InstanceValue> edited_value;
        OfxStatus begin_status{}, change_status{}, end_status{}, edit_status{}, reentry_status{};
    };
    OfxStatus change(OfxPlugin* plugin, InstanceHost& host, OfxImageEffectHandle effect, const char* reason, const char* name) {
        return call(plugin, kOfxActionInstanceChanged, effect, host.change_args({reason, kOfxTypeParameter, name}));
    }
    Reference reference_cohort(OfxPlugin* plugin, InstanceHost& host) {
        JuicerTests::CaseEvent test(kCaller, "reference");
        const JuicerTests::Cohort rows("ofx.instance.reference", {"census", "clip_order", "action_order", "missing", "extra", "type", "premature_success", "late_stop"});
        Reference result;
        plugin->setHost(host.binding());
        require(call(plugin, kOfxActionLoad) == kOfxStatOK, "reference load");
        require(call(plugin, kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK, "reference context");
        auto* effect = host.create_effect();
        host.set_parameter_presentation(effect, "GrainChroma", kOfxParamPropHint, {std::string("returned hint")});
        host.return_string_on_failure_next("get_string/OfxParamPropHint", kOfxStatFailed);
        host.clear_calls();
        require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "reference create");
        result.create = host.calls();
        {
            auto row = rows.case_event("census");
            require(!host.callback_error() && constructor_census(result.create), "reference constructor census");
            row.passed();
        }
        {
            auto row = rows.case_event("clip_order");
            require(handles(result.create, "clip_handle") == std::vector<std::string>{"Source", "Output"}, "reference clip order");
            row.passed();
        }
        auto missing = result.create;
        const auto first = std::find_if(missing.begin(), missing.end(), [](const auto& c) {
            return c.operation == "param_handle";
        });
        require(first != missing.end(), "constructor missing-control boundary");
        missing.erase(first);
        {
            auto row = rows.case_event("missing");
            require(!constructor_census(missing), "missing parameter negative control");
            row.passed();
        }
        auto extra = result.create;
        extra.push_back({"param_handle", {}, "extra", 0, {}, {}});
        {
            auto row = rows.case_event("extra");
            require(!constructor_census(extra), "extra parameter negative control");
            row.passed();
        }
        host.clear_calls();
        result.begin_status = call(plugin, kOfxActionBeginInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "Exposure"}));
        result.begin = host.calls();
        host.clear_calls();
        result.change_status = change(plugin, host, effect, kOfxChangeUserEdited, "Exposure");
        result.change = host.calls();
        host.clear_calls();
        result.end_status = call(plugin, kOfxActionEndInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "Exposure"}));
        result.end = host.calls();
        {
            auto row = rows.case_event("action_order");
            std::vector<std::string> properties;
            for (const auto& c : result.change)
                if (c.target == "change")
                    properties.push_back(c.property);
            require(properties == std::vector<std::string>{kOfxPropChangeReason, kOfxPropTime, kOfxImageEffectPropRenderScale, kOfxImageEffectPropRenderScale, kOfxPropType, kOfxPropName}, "reference action property order");
            row.passed();
        }
        {
            auto row = rows.case_event("premature_success");
            auto shortened = result.change;
            require(shortened.size() > 1, "change trace boundary");
            shortened.pop_back();
            require(same_action({result.change_status, result.change}, {result.change_status, result.change}), "action verifier positive control");
            require(!same_action({result.change_status, shortened}, {result.change_status, result.change}), "premature success negative control");
            row.passed();
        }
        {
            auto row = rows.case_event("late_stop");
            auto extended = result.change;
            extended.push_back(result.change.back());
            require(!same_action({result.change_status, extended}, {result.change_status, result.change}), "late stop negative control");
            row.passed();
        }
        host.set_value(effect, "CameraFilmFormatPreset", {1});
        host.clear_calls();
        result.edit_status = change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset");
        result.edit = host.calls();
        result.edited_value = host.value(effect, "CameraFilmFormatMm");
        host.set_value(effect, "CameraFilmFormatPreset", {2});
        host.clear_calls();
        host.reenter_next("param_set/CameraFilmFormatMm", [&] {
            require(change(plugin, host, effect, kOfxChangePluginEdited, "CameraFilmFormatMm") == result.change_status, "reference nested change");
        });
        result.reentry_status = change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset");
        result.reentry = host.calls();
        for (const auto& scenario : edit_scenarios()) {
            if (!scenario.authored.empty())
                host.set_value(effect, scenario.name, scenario.authored);
            host.clear_calls();
            const auto status = change(plugin, host, effect, kOfxChangeUserEdited, scenario.name);
            result.edits.push_back({status, host.calls(), host.value(effect, scenario.observed)});
        }
        for (const auto& scenario : kActionScenarios)
            result.actions.push_back(run_action(plugin, host, effect, scenario));
        for (const auto* reason : {kOfxChangeUserEdited, kOfxChangeTime}) {
            host.set_value(effect, "CameraFilmFormatPreset", {1});
            host.clear_calls();
            int nested = -1;
            host.reenter_next("param_set/CameraFilmFormatMm", [&] {
                nested = change(plugin, host, effect, reason, "Exposure");
            });
            const auto status = change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset");
            require(nested == result.change_status, "reference nested user/time status");
            result.nested_changes.push_back({status, host.calls()});
        }
        host.clear_calls();
        require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "reference destroy");
        result.destroy = host.calls();
        require(host.instance_data(effect) == nullptr, "reference detached");
        result.faults = handle_faults(plugin, host, false);
        result.controls = control_faults(plugin, host);
        result.typed = typed_controls(plugin, host);
        result.contexts = context_cases(plugin, host);
        {
            auto* ordinary = host.create_effect();
            host.clear_calls();
            require(call(plugin, kOfxActionCreateInstance, ordinary) == kOfxStatOK, "ordinary reference create");
            result.ordinary_create = host.calls();
            require(call(plugin, kOfxActionDestroyInstance, ordinary) == kOfxStatOK, "ordinary reference destroy");
        }
        {
            auto row = rows.case_event("type");
            require(result.faults.size() == 113 && std::all_of(result.faults.begin() + 105, result.faults.begin() + 113, [](const auto& fault) {
                        return fault.injected == -1 && fault.status != kOfxStatOK;
                    }),
                    "reference type failures");
            row.passed();
        }
        require(call(plugin, kOfxActionUnload) == kOfxStatOK, "reference unload");
        require(!host.callback_error(), "reference host callback");
        test.passed();
        return result;
    }
    void candidate_cohort(OfxPlugin* plugin, InstanceHost& host, const Reference& reference) {
        const JuicerTests::Cohort create_rows("ofx.instance.create", create_ids());
        const JuicerTests::Cohort action_rows("ofx.instance.actions", {"change", "unrelated", "plugin_edit", "time", "null_args", "reason", "name", "status"});
        const JuicerTests::Cohort parameter_rows("ofx.instance.parameters", {"get_scalar", "get_integer", "get_boolean", "get_choice", "get_pair", "get_triplet", "get_string_copy", "set_scalar", "set_integer", "set_boolean", "set_pair", "set_triplet", "set_string_copy", "fail"});
        const JuicerTests::Cohort reentry_rows("ofx.instance.reentry", {"same_plugin", "same_user", "same_time", "other_instance", "before_publication", "after_publication"});
        const JuicerTests::Cohort destroy_rows("ofx.instance.destroy", {"normal", "null", "repeat", "detach", "terminal_order"});
        JuicerTests::CaseEvent create_test(kCaller, "create");
        JuicerTests::CaseEvent action_test(kCaller, "actions");
        JuicerTests::CaseEvent parameter_test(kCaller, "parameters");
        JuicerTests::CaseEvent reentry_test(kCaller, "reentry");
        plugin->setHost(host.binding());
        require(call(plugin, kOfxActionLoad) == kOfxStatOK, "candidate load");
        require(call(plugin, kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK, "candidate context");
        auto* effect = host.create_effect();
        host.set_parameter_presentation(effect, "GrainChroma", kOfxParamPropHint, {std::string("returned hint")});
        host.return_string_on_failure_next("get_string/OfxParamPropHint", kOfxStatFailed);
        {
            auto row = create_rows.case_event("happy");
            host.clear_calls();
            require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "candidate create");
            require(host.instance_data(effect) != nullptr, "candidate publication");
            require(create_equivalent(host.calls(), reference.create), "constructor continuation");
            auto shortened = host.calls();
            require(!shortened.empty(), "constructor trace boundary");
            shortened.pop_back();
            require(!create_equivalent(shortened, reference.create), "constructor premature success negative control");
            auto extended = host.calls();
            extended.push_back(host.calls().back());
            require(!create_equivalent(extended, reference.create), "constructor late stop negative control");
            require(std::count_if(host.calls().begin(), host.calls().end(), [](const auto& c) {
                        return c.operation == "param_properties";
                    }) == 113,
                    "second parameter properties acquired");
            require(std::any_of(host.calls().begin(), host.calls().end(), [](const auto& c) {
                        return c.operation == "get_string" && c.target == "GrainChroma" && c.property == kOfxParamPropType;
                    }),
                    "type property identity");
            require(std::any_of(host.calls().begin(), host.calls().end(), [](const auto& c) {
                        return c.operation == "get_string" && c.target == "GrainChroma.presentation" && c.property == kOfxParamPropHint && c.status == kOfxStatFailed;
                    }),
                    "optional hint property identity and failure");
            row.passed();
        }
        {
            JuicerTests::CaseEvent test(kCaller, "handles");
            require(constructor_census(host.calls()), "candidate constructor census");
            const JuicerTests::Cohort rows("ofx.instance.handles", {"first", "middle", "last", "source", "output", "parameter_properties", "parameter_type", "null", "type", "double", "boolean", "choice", "string_choice", "integer", "double2", "double3", "push_button"});
            auto actual = handle_faults(plugin, host, true, &rows, &reference.faults);
            require(actual.size() == reference.faults.size() + 5, "handle fault membership");
            test.passed();
        }
        {
            auto row = create_rows.case_event("duplicate");
            void* original = host.instance_data(effect);
            require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatErrFatal && host.instance_data(effect) == original, "duplicate create preserves owner");
            row.passed();
        }
        {
            auto row = create_rows.case_event("null_effect");
            require(call(plugin, kOfxActionCreateInstance) == kOfxStatErrBadHandle, "null create effect");
            row.passed();
        }
        {
            auto row = create_rows.case_event("null_properties");
            auto* probe = host.create_effect();
            host.null_next("effect_properties");
            require(call(plugin, kOfxActionCreateInstance, probe) == kOfxStatErrBadHandle && host.instance_data(probe) == nullptr, "null effect properties");
            row.passed();
        }
        {
            auto row = create_rows.case_event("status");
            auto* probe = host.create_effect();
            host.fail_next("effect_properties", kOfxStatErrMemory);
            require(call(plugin, kOfxActionCreateInstance, probe) == kOfxStatErrMemory && host.instance_data(probe) == nullptr, "create acquisition failure");
            row.passed();
        }
        {
            auto row = create_rows.case_event("before_publication");
            auto reentry_row = reentry_rows.case_event("before_publication");
            auto* probe = host.create_effect();
            int nested = -1;
            host.reenter_next("param_handle/Exposure", [&] {
                nested = change(plugin, host, probe, kOfxChangePluginEdited, "Exposure");
            });
            require(call(plugin, kOfxActionCreateInstance, probe) == kOfxStatOK && nested == kOfxStatErrBadHandle, "nested change before publication");
            require(call(plugin, kOfxActionDestroyInstance, probe) == kOfxStatOK, "before publication cleanup");
            row.passed();
            reentry_row.passed();
        }
        {
            auto row = action_rows.case_event("change");
            host.clear_calls();
            require(call(plugin, kOfxActionBeginInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "Exposure"})) == reference.begin_status && host.calls() == reference.begin, "begin action");
            host.clear_calls();
            require(change(plugin, host, effect, kOfxChangeUserEdited, "Exposure") == reference.change_status && host.calls() == reference.change, "change action");
            require(std::any_of(host.calls().begin(), host.calls().end(), [](const auto& c) {
                        return c.operation == "set_string" && c.target == "GrainChroma.presentation" && c.property == kOfxParamPropHint && c.values == std::vector<ofx_test::InstanceValue>{std::string("returned hint")};
                    }),
                    "failed optional hint returned string retained");
            host.clear_calls();
            require(call(plugin, kOfxActionEndInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "Exposure"})) == reference.end_status && host.calls() == reference.end, "end action");
            row.passed();
        }
        {
            const auto& calls = reference.change;
            require(std::any_of(calls.begin(), calls.end(), [](const auto& c) {
                        return c.operation == "param_get";
                    }),
                    "current value getters");
            require(std::none_of(calls.begin(), calls.end(), [](const auto& c) {
                        return c.operation == "param_get_time";
                    }),
                    "change used animated getter");
        }
        const JuicerTests::Cohort edit_rows("ofx.instance.edits", {"camera_preset", "camera_manual", "film_route", "gamma_manual", "stock_defaults", "grain_preset", "grain_preset_fine", "grain_preset_coarse", "grain_reset", "grain_masters", "grain_layers_master", "grain_density_master", "grain_uniformity_master", "grain_linked"});
        JuicerTests::CaseEvent edit_test(kCaller, "edits");
        {
            auto row = edit_rows.case_event("camera_preset");
            host.set_value(effect, "CameraFilmFormatPreset", {1});
            host.clear_calls();
            require(change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset") == reference.edit_status && host.calls() == reference.edit, "camera preset continuation");
            require(host.value(effect, "CameraFilmFormatMm") == reference.edited_value, "camera preset value");
            row.passed();
        }
        {
            auto row = reentry_rows.case_event("same_plugin");
            host.set_value(effect, "CameraFilmFormatPreset", {2});
            host.clear_calls();
            host.reenter_next("param_set/CameraFilmFormatMm", [&] {
                require(change(plugin, host, effect, kOfxChangePluginEdited, "CameraFilmFormatMm") == reference.change_status, "candidate nested change");
            });
            require(change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset") == reference.reentry_status && host.calls() == reference.reentry, "nested edit continuation");
            row.passed();
        }
        require(reference.edits.size() == edit_scenarios().size(), "edit membership");
        for (std::size_t i = 0; i < edit_scenarios().size(); ++i) {
            const auto& scenario = edit_scenarios()[i];
            const auto& expected = reference.edits[i];
            auto row = edit_rows.case_event(scenario.id);
            if (!scenario.authored.empty())
                host.set_value(effect, scenario.name, scenario.authored);
            host.clear_calls();
            require(change(plugin, host, effect, kOfxChangeUserEdited, scenario.name) == expected.status, "edit status");
            require(host.calls() == expected.calls && host.value(effect, scenario.observed) == expected.value, "edit continuation");
            row.passed();
        }
        edit_test.passed();
        require(reference.actions.size() == kActionScenarios.size(), "action membership");
        for (std::size_t i = 0; i < kActionScenarios.size(); ++i) {
            const auto& scenario = kActionScenarios[i];
            auto row = action_rows.case_event(scenario.id);
            const auto actual = run_action(plugin, host, effect, scenario);
            require(same_action(actual, reference.actions[i]), "action continuation");
            row.passed();
        }
        action_test.passed();
        require(reference.nested_changes.size() == 2, "nested change membership");
        for (std::size_t i = 0; i < reference.nested_changes.size(); ++i) {
            auto row = reentry_rows.case_event(i == 0 ? "same_user" : "same_time");
            const char* reason = i == 0 ? kOfxChangeUserEdited : kOfxChangeTime;
            host.set_value(effect, "CameraFilmFormatPreset", {1});
            host.clear_calls();
            int nested = -1;
            host.reenter_next("param_set/CameraFilmFormatMm", [&] {
                nested = change(plugin, host, effect, reason, "Exposure");
            });
            const auto status = change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset");
            require(nested == reference.change_status && status == reference.nested_changes[i].status && host.calls() == reference.nested_changes[i].calls, "nested user/time continuation");
            row.passed();
        }
        {
            auto row = reentry_rows.case_event("other_instance");
            auto* other = host.create_effect();
            require(call(plugin, kOfxActionCreateInstance, other) == kOfxStatOK, "second instance create");
            host.set_value(effect, "CameraFilmFormatPreset", {1});
            host.clear_calls();
            int nested = -1;
            host.reenter_next("param_set/CameraFilmFormatMm", [&] {
                nested = change(plugin, host, other, kOfxChangePluginEdited, "Exposure");
            });
            require(change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset") == reference.change_status && nested == reference.change_status, "other instance nested change");
            require(std::any_of(host.calls().begin(), host.calls().end(), [&](const auto& c) {
                        return c.operation == "param_get" && c.scope == host.effect_scope(other);
                    }),
                    "other instance was suppressed");
            require(call(plugin, kOfxActionDestroyInstance, other) == kOfxStatOK, "second instance destroy");
            row.passed();
        }
        {
            auto row = create_rows.case_event("after_publication");
            auto reentry_row = reentry_rows.case_event("after_publication");
            int nested = -1;
            void* original = host.instance_data(effect);
            host.reenter_next("param_get/Exposure", [&] {
                nested = change(plugin, host, effect, kOfxChangePluginEdited, "Exposure");
            });
            require(change(plugin, host, effect, kOfxChangeUserEdited, "Exposure") == reference.change_status && nested == reference.change_status && host.instance_data(effect) == original, "nested change after publication");
            row.passed();
            reentry_row.passed();
        }
        reentry_test.passed();
        {
            JuicerTests::CaseEvent test(kCaller, "destroy");
            auto normal = destroy_rows.case_event("normal");
            auto detached = destroy_rows.case_event("detach");
            auto order = destroy_rows.case_event("terminal_order");
            host.clear_calls();
            require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "candidate destroy");
            require(host.instance_data(effect) == nullptr, "candidate detached");
            require(host.calls() == reference.destroy, "destroy continuation");
            normal.passed();
            detached.passed();
            const auto clear = std::find_if(host.calls().begin(), host.calls().end(), [](const auto& c) {
                return c.operation == "set_pointer" && c.property == kOfxPropInstanceData;
            });
            require(clear != host.calls().end() && std::next(clear) == host.calls().end(), "terminal pointer clear order");
            order.passed();
            {
                auto row = destroy_rows.case_event("null");
                require(call(plugin, kOfxActionDestroyInstance) == kOfxStatErrBadHandle, "null destroy");
                row.passed();
            }
            {
                auto row = destroy_rows.case_event("repeat");
                require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatErrBadHandle && host.instance_data(effect) == nullptr, "repeat destroy");
                row.passed();
            }
            test.passed();
        }
        const JuicerTests::Cohort control_rows("ofx.instance.control_status", control_fault_ids());
        const auto controls = control_faults(plugin, host, &control_rows, &reference.controls);
        require(controls.size() == reference.controls.size(), "control fault membership");
        const auto typed = typed_controls(plugin, host, &parameter_rows, &reference.typed);
        require(typed.size() == reference.typed.size(), "typed callback membership");
        parameter_test.passed();
        const auto contexts = context_cases(plugin, host, &create_rows, &reference.contexts);
        require(contexts.size() == reference.contexts.size(), "context membership");
        acquisition_cases(plugin, host, create_rows, reference.ordinary_create);
        create_test.passed();
        require(call(plugin, kOfxActionUnload) == kOfxStatOK && !host.callback_error(), "candidate unload");
    }
} // namespace

int main(int argc, char** argv) try {
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto* id : kCases)
            JuicerTests::list_case(kCaller, id);
        return 0;
    }
    JuicerTests::CaseLog log;
    auto* reference = OfxGetPlugin(0);
    auto* candidate = fj_ofx_plugin(0);
    require(reference && candidate && reference->mainEntry != candidate->mainEntry, "independent descriptors");
    InstanceHost reference_host;
    const auto expected = reference_cohort(reference, reference_host);
    reference = nullptr;
    InstanceHost candidate_host;
    candidate_cohort(candidate, candidate_host, expected);
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "instance fixture: %s\n", error.what());
    return 1;
}
