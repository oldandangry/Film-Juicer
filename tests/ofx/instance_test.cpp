#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "instance_host.h"
#include "host_reference.h"

#include "juicer_ofx_api.h"

namespace {
    using ofx_test::InstanceCall;
    using ofx_test::InstanceHost;
    void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
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
    using ofx_test::reference::advanced_halation_parameter;
    constexpr std::string_view kArchivedScannerGrid = "ScannerLutResolution";
    constexpr std::string_view kArchivedScannerGridGetter = "param_get/ScannerLutResolution";
    constexpr std::string_view kIntegerControl = "HalationBounceCount";
    std::vector<InstanceCall> historical_instance_calls(std::vector<InstanceCall> calls, bool create = false) {
        std::erase_if(calls, [&](const InstanceCall& c) {
            if (c.operation == "param_handle")
                return advanced_halation_parameter(c.property);
            if (c.operation == "param_properties" || c.operation == "param_get" ||
                (c.operation == "get_string" && c.property == kOfxParamPropType))
                return advanced_halation_parameter(c.target) ||
                       (c.operation == "param_get" && c.target == (create ? "HalationActive" : "GrainModel"));
            constexpr std::string_view presentation = ".presentation";
            return c.operation == "set_int" && c.property == kOfxParamPropEnabled && c.target.ends_with(presentation) &&
                   advanced_halation_parameter(std::string_view(c.target).substr(0, c.target.size() - presentation.size()));
        });
        return calls;
    }
    constexpr std::array<std::string_view, 4> kFilmParameters{
        "GrainModel", "FilmGrainAmountEv", "FilmGrainTexelPitchUm", "FilmGrainChromaFraction"};
    constexpr std::array<std::string_view, 21> kLegacyVisibility{
        "GrainSublayersActive", "GrainPreset", "GrainAmplitude", "GrainBlur", "GrainSharpness", "GrainChroma", "GrainTexture", "GrainResetAdvanced", "GrainParticleAreaUm2", "GrainParticleScaleMaster", "GrainParticleScaleLayersMaster", "GrainDensityMinMaster", "GrainUniformityMaster", "GrainBlurDyeCloudsUm", "GrainSizeMixWeight", "GrainSizeMixWeightMid", "GrainSizeMixScale", "GrainMicroStructure", "GrainDebugView", "GrainClumpTemporalMix", "GrainClumpMorphPeriodSec"};
    bool grain_v2_parameter(std::string_view name) {
        return std::find(kFilmParameters.begin(), kFilmParameters.end(), name) != kFilmParameters.end();
    }
    void project_legacy_grain_calls(std::vector<InstanceCall>& calls) {
        std::erase_if(calls, [](const InstanceCall& c) {
            if (c.operation == "param_handle")
                return grain_v2_parameter(c.property);
            if (c.operation == "param_get" || c.operation == "param_properties" ||
                (c.operation == "get_string" && c.property == kOfxParamPropType))
                return grain_v2_parameter(c.target);
            constexpr std::string_view presentation = ".presentation";
            if (c.operation != "set_int" || c.property != kOfxParamPropSecret || !c.target.ends_with(presentation))
                return false;
            const auto name = std::string_view(c.target).substr(0, c.target.size() - presentation.size());
            return grain_v2_parameter(name) ||
                   std::find(kLegacyVisibility.begin(), kLegacyVisibility.end(), name) != kLegacyVisibility.end();
        });
    }
    bool constructor_census(const std::vector<InstanceCall>& calls, bool legacy = false) {
        auto params = handles(calls, "param_handle");
        if (!legacy) {
            const auto first = std::find(params.begin(), params.end(), kFilmParameters.front());
            if (params.end() - first <= static_cast<std::ptrdiff_t>(kFilmParameters.size()))
                return false;
            const auto next = first + static_cast<std::ptrdiff_t>(kFilmParameters.size());
            if (*next != "GrainSublayersActive" || !std::equal(kFilmParameters.begin(), kFilmParameters.end(), first))
                return false;
            params.erase(first, next);
        }
        const auto clips = handles(calls, "clip_handle");
        if (clips.size() != ofx_test::kInstanceClips.size())
            return false;
        std::size_t i = 0;
        for (const auto& p : ofx_test::kInstanceParameters) {
            if (legacy && advanced_halation_parameter(p.name))
                continue;
            if (i == params.size() || params[i++] != p.name)
                return false;
        }
        return i == params.size() && std::equal(clips.begin(), clips.end(), ofx_test::kInstanceClips.begin());
    }
    bool create_equivalent(std::vector<InstanceCall> actual, std::vector<InstanceCall> expected, bool full_grain = false) {
        actual = historical_instance_calls(std::move(actual), true);
        if (!full_grain)
            project_legacy_grain_calls(actual);
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
        if (ofx_test::reference::calls(actual) == ofx_test::reference::calls(expected))
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
        return ofx_test::reference::calls(actual) == ofx_test::reference::calls(expected);
    }
    bool failed_create_equivalent(std::vector<InstanceCall> actual, std::vector<InstanceCall> expected) {
        actual = historical_instance_calls(std::move(actual), true);
        project_legacy_grain_calls(actual);
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
        if (ofx_test::reference::calls(actual) == ofx_test::reference::calls(expected))
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
        return ofx_test::reference::calls(actual) == ofx_test::reference::calls(expected);
    }
    struct HandleFault {
        std::string name;
        int injected;
        OfxStatus status;
        std::vector<std::string> params;
        std::vector<InstanceCall> calls;
    };
    std::vector<HandleFault> handle_faults(OfxPlugin* plugin, InstanceHost& host, bool include_null, const std::vector<HandleFault>* expected = nullptr) {
        std::vector<HandleFault> results;
        auto append = [&](HandleFault value) {
            if (expected) {
                const auto& r = expected->at(results.size());
                if (value.name == kIntegerControl) {
                    require(r.name == kArchivedScannerGrid && value.injected == r.injected && value.status == r.status &&
                                !value.calls.empty() && value.calls.back().operation == "get_string" &&
                                value.calls.back().target == kIntegerControl && value.calls.back().property == kOfxParamPropType &&
                                !value.params.empty() && value.params.back() == kIntegerControl,
                            "integer type fault stops at the live type query");
                    results.push_back(std::move(value));
                    return;
                }
                auto legacy_params = value.params;
                std::erase_if(legacy_params, [](const auto& name) {
                    return advanced_halation_parameter(name) || grain_v2_parameter(name);
                });
                require(value.name == r.name && value.injected == r.injected && value.status == r.status && legacy_params == r.params, "handle fault membership");
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
        for (std::size_t index = 0; index < names.size(); ++index)
            ([&] {
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
            })();
        const std::array acquisition_ids{"source", "output", "parameter_properties", "parameter_type"};
        const std::array acquisition_faults{"clip_handle/Source", "clip_handle/Output", "param_properties/Exposure", "get_string/OfxParamPropType"};
        for (std::size_t index = 0; index < acquisition_ids.size(); ++index)
            ([&] {
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
            })();
        const std::array typed_names{"Exposure", "CameraAutoExposure", "CameraFilmFormatPreset", "FilmProfileKey", "HalationBounceCount", "ScannerUnsharpMask", "CouplersLangmuirDonorKRgb", "GrainResetAdvanced"};
        ([&] {
            for (std::size_t index = 0; index < typed_names.size(); ++index)
                ([&] {
                    const char* name = typed_names[index];
                    auto* effect = host.create_effect();
                    host.set_parameter_type(effect, name, "InvalidParameterType");
                    host.clear_calls();
                    const auto actual = call(plugin, kOfxActionCreateInstance, effect);
                    auto seen = handles(host.calls(), "param_handle");
                    auto calls = host.calls();
                    require(actual != kOfxStatOK && host.instance_data(effect) == nullptr, "type mismatch rollback");
                    append({name, -1, actual, std::move(seen), std::move(calls)});
                })();
        })();
        if (include_null)
            ([&] {
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
            })();
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
        return actual.status == expected.status && ofx_test::reference::calls(historical_instance_calls(actual.calls)) == ofx_test::reference::calls(expected.calls);
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
        std::string operation;
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
        ControlScenario{"get_integer", "param_get/HalationBounceCount", "Exposure"},
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
    std::vector<ControlFault> control_faults(OfxPlugin* plugin, InstanceHost& host, const std::vector<ControlFault>* expected = nullptr) {
        std::vector<ControlFault> observations;
        for (const auto& scenario : kControlScenarios) {
            auto* effect = host.create_effect();
            require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "control fault create");
            host.set_value(effect, "GrainActive", {1});
            host.set_value(effect, "CameraAutoExposure", {1});
            const bool halation_integer = std::string_view(scenario.id) == "get_integer";
            if (halation_integer) {
                host.set_value(effect, "HalationActive", {1});
                host.set_value(effect, "HalationBounceCount", {4});
            }
            for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status) {
                ([&] {
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
                    if (expected) {
                        const auto& archived = expected->at(observations.size());
                        if (halation_integer) {
                            require(archived.operation == kArchivedScannerGridGetter && archived.injected == status &&
                                        result == archived.status,
                                    "live integer getter returns the exact OFX status");
                            const auto bounce = std::find_if(observation.calls.begin(), observation.calls.end(), [](const auto& call) {
                                return call.operation == "param_get" && call.target == kIntegerControl;
                            });
                            const bool continued = std::any_of(std::next(bounce), observation.calls.end(), [](const auto& call) {
                                return call.operation == "param_get" && call.target == "HalationBounceDecay";
                            });
                            require(continued == (result == kOfxStatReplyDefault && (status == kOfxStatOK || status >= kOfxStatReplyYes)),
                                    "live integer getter continuation");
                        } else {
                            require(same_effect_trace(observation, archived), "actual callback get/set full status continuation");
                        }
                    }
                    observations.push_back(std::move(observation));
                })();
            }
            require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "control fault destroy");
        }
        return observations;
    }
    bool same_effect_trace(const ControlFault& actual, const ControlFault& expected) {
        const auto legacy = historical_instance_calls(actual.calls);
        if (std::string_view(actual.operation) != expected.operation || actual.injected != expected.injected || actual.status != expected.status || legacy.size() != expected.calls.size())
            return false;
        for (std::size_t i = 0; i < legacy.size(); ++i) {
            auto a = legacy[i];
            const auto& e = expected.calls[i];
            if (a.scope == actual.scope)
                a.scope = expected.scope;
            if (a.target == actual.scope)
                a.target = expected.scope;
            else if (a.target == actual.scope + ".params")
                a.target = expected.scope + ".params";
            if (ofx_test::reference::calls(std::vector<InstanceCall>{a}) != ofx_test::reference::calls(std::vector<InstanceCall>{e}))
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
        TypedScenario{"get_integer", "param_get", "HalationBounceCount", "Exposure", false},
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
    std::vector<ControlFault> typed_controls(OfxPlugin* plugin, InstanceHost& host, const std::vector<ControlFault>* expected = nullptr) {
        std::vector<ControlFault> observations;
        auto* effect = host.create_effect();
        require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "typed consumer create");
        host.set_value(effect, "GrainActive", {1});
        host.set_value(effect, "CameraAutoExposure", {1});
        const auto original_active = host.value(effect, "HalationActive");
        const auto original_bounces = host.value(effect, kIntegerControl);
        for (const auto& scenario : kTypedScenarios)
            ([&] {
                const bool halation_integer = std::string_view(scenario.target) == kIntegerControl;
                if (halation_integer) {
                    host.set_value(effect, "HalationActive", {1});
                    host.set_value(effect, scenario.target, {4});
                }
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
                if (expected) {
                    const auto& archived = expected->at(observations.size());
                    if (halation_integer) {
                        require(archived.operation == scenario.id && result == archived.status &&
                                    found->values == std::vector<ofx_test::InstanceValue>{4},
                                "live integer getter preserves the authored value");
                    } else {
                        require(same_effect_trace(observation, archived), "typed actual callback continuation");
                    }
                }
                if (halation_integer) {
                    host.set_value(effect, "HalationActive", original_active);
                    host.set_value(effect, kIntegerControl, original_bounces);
                }
                observations.push_back(std::move(observation));
            })();
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
    std::vector<ContextObservation> context_cases(OfxPlugin* plugin, InstanceHost& host, const std::vector<ContextObservation>* expected = nullptr) {
        std::vector<ContextObservation> observations;
        for (const auto& scenario : kContextScenarios)
            ([&] {
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
                    require(actual.status == ref.status && (unknown ? ofx_test::reference::calls(calls) == ofx_test::reference::calls(ref.calls) : create_equivalent(calls, ref.calls)), "context full continuation");
                }
                if (unknown)
                    require(host.instance_data(effect) == nullptr && actual.calls.size() == 2 && actual.calls.back().operation == "get_string" && actual.calls.back().property == kOfxImageEffectPropContext, "unknown context stop");
                else
                    require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "context destroy");
                observations.push_back(std::move(actual));
            })();
        if (expected)
            ([&] {
                auto* effect = host.create_effect();
                host.null_next("get_string/OfxImageEffectPropContext");
                host.clear_calls();
                require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatErrBadHandle && host.instance_data(effect) == nullptr, "null context status");
                require(host.calls().size() == 2 && host.calls().back().operation == "get_string" && host.calls().back().property == kOfxImageEffectPropContext, "null context stop");
            })();
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
    void acquisition_cases(OfxPlugin* plugin, InstanceHost& host, const std::vector<InstanceCall>& native) {
        const auto prefix = expected_prepublication(native);
        const std::string native_scope = native.front().target;
        for (const auto& scenario : kAcquisitionScenarios) {
            for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status) {
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
            }
        }
        for (const auto& scenario : kAcquisitionScenarios) {
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
    using ofx_test::reference::Json;
    template <class T, class F>
    Json encode_list(const std::vector<T>& items, F f) {
        auto out = Json::array();
        for (const auto& item : items)
            out.push_back(f(item));
        return out;
    }
    Json encode(const Reference& r) {
        using namespace ofx_test::reference;
        auto faults = encode_list(r.faults, [](const auto& x) {
            return Json::array({x.name, x.injected, x.status, x.params, calls(x.calls)});
        });
        auto controls = [](const auto& items) {
            return encode_list(items, [](const auto& x) {
                return Json::array({x.operation, x.injected, x.status, x.scope, calls(x.calls)});
            });
        };
        auto contexts = encode_list(r.contexts, [](const auto& x) {
            return Json::array({x.status, x.scope, calls(x.calls)});
        });
        auto edits = encode_list(r.edits, [](const auto& x) {
            return Json::array({x.status, calls(x.calls), values(x.value)});
        });
        auto actions = [](const auto& items) {
            return encode_list(items, [](const auto& x) {
                return Json::array({x.status, calls(x.calls)});
            });
        };
        return Json::array({faults, controls(r.controls), controls(r.typed), contexts, edits, actions(r.actions), actions(r.nested_changes), calls(r.create), calls(r.ordinary_create), calls(r.begin), calls(r.change), calls(r.end), calls(r.edit), calls(r.reentry), calls(r.destroy), values(r.edited_value), r.begin_status, r.change_status, r.end_status, r.edit_status, r.reentry_status});
    }
    Reference decode(const Json& record) {
        using namespace ofx_test::reference;
        if (!record.is_array() || record.size() != 21)
            throw std::runtime_error("instance reference shape");
        for (std::size_t i = 0; i < 7; ++i)
            if (!record[i].is_array())
                throw std::runtime_error("instance reference collection");
        auto shape = [](const Json& x, std::size_t n) {
            if (!x.is_array() || x.size() != n)
                throw std::runtime_error("instance reference row");
        };
        Reference r;
        for (const auto& x : record[0]) {
            shape(x, 5);
            r.faults.push_back({x[0].get<std::string>(), x[1].get<int>(), x[2].get<int>(), x[3].get<std::vector<std::string>>(), instance_calls(x[4])});
        }
        auto controls = [&](const Json& rows, auto& out) {
            for (const auto& x : rows) {
                shape(x, 5);
                out.push_back({x[0].get<std::string>(), x[1].get<int>(), x[2].get<int>(), x[3].get<std::string>(), instance_calls(x[4])});
            }
        };
        controls(record[1], r.controls);
        controls(record[2], r.typed);
        for (const auto& x : record[3]) {
            shape(x, 3);
            r.contexts.push_back({x[0].get<int>(), x[1].get<std::string>(), instance_calls(x[2])});
        }
        for (const auto& x : record[4]) {
            shape(x, 3);
            r.edits.push_back({x[0].get<int>(), instance_calls(x[1]), instance_values(x[2])});
        }
        auto actions = [&](const Json& rows, auto& out) {
            for (const auto& x : rows) {
                shape(x, 2);
                out.push_back({x[0].get<int>(), instance_calls(x[1])});
            }
        };
        actions(record[5], r.actions);
        actions(record[6], r.nested_changes);
        r.create = instance_calls(record[7]);
        r.ordinary_create = instance_calls(record[8]);
        r.begin = instance_calls(record[9]);
        r.change = instance_calls(record[10]);
        r.end = instance_calls(record[11]);
        r.edit = instance_calls(record[12]);
        r.reentry = instance_calls(record[13]);
        r.destroy = instance_calls(record[14]);
        r.edited_value = instance_values(record[15]);
        r.begin_status = record[16].get<int>();
        r.change_status = record[17].get<int>();
        r.end_status = record[18].get<int>();
        r.edit_status = record[19].get<int>();
        r.reentry_status = record[20].get<int>();
        if (r.faults.size() != 113 || r.controls.size() != kControlScenarios.size() * 15 || r.contexts.size() != kContextScenarios.size() || r.actions.size() != kActionScenarios.size() || r.nested_changes.size() != 2 || encode(r) != record)
            throw std::runtime_error("instance reference membership");
        const auto retire_grid_calls = [](auto& calls) {
            return std::erase_if(calls, [](const auto& call) {
                return (call.operation == "param_handle" && call.property == kArchivedScannerGrid) ||
                       call.target == kArchivedScannerGrid || call.target == "ScannerLutResolution.presentation";
            });
        };
        require(retire_grid_calls(r.create) == 3 && retire_grid_calls(r.ordinary_create) == 3,
                "archived scanner grid constructor boundaries");
        for (auto& fault : r.faults) {
            std::erase(fault.params, kArchivedScannerGrid);
            retire_grid_calls(fault.calls);
        }
        for (auto* calls : {&r.begin, &r.change, &r.end, &r.edit, &r.reentry, &r.destroy})
            retire_grid_calls(*calls);
        for (auto* group : {&r.controls, &r.typed})
            for (auto& observation : *group)
                retire_grid_calls(observation.calls);
        for (auto& context : r.contexts)
            retire_grid_calls(context.calls);
        for (auto& edit : r.edits)
            retire_grid_calls(edit.calls);
        for (auto* group : {&r.actions, &r.nested_changes})
            for (auto& observation : *group)
                retire_grid_calls(observation.calls);
        return r;
    }
    void reference_record_case(const Reference& r) {
        {
            require(constructor_census(r.create, true), "reference constructor census");
        }
        {
            require(handles(r.create, "clip_handle") == std::vector<std::string>{"Source", "Output"}, "reference clip order");
        }
        auto missing = r.create;
        const auto first = std::find_if(missing.begin(), missing.end(), [](const auto& c) {
            return c.operation == "param_handle";
        });
        require(first != missing.end(), "constructor missing-control boundary");
        missing.erase(first);
        {
            require(!constructor_census(missing, true), "missing parameter negative control");
        }
        auto extra = r.create;
        extra.push_back({"param_handle", {}, "extra", 0, {}, {}});
        {
            require(!constructor_census(extra, true), "extra parameter negative control");
        }
        {
            std::vector<std::string> properties;
            for (const auto& c : r.change)
                if (c.target == "change")
                    properties.push_back(c.property);
            require(properties == std::vector<std::string>{kOfxPropChangeReason, kOfxPropTime, kOfxImageEffectPropRenderScale, kOfxImageEffectPropRenderScale, kOfxPropType, kOfxPropName}, "reference action property order");
        }
        {
            auto shortened = r.change;
            require(shortened.size() > 1, "change trace boundary");
            shortened.pop_back();
            require(same_action({r.change_status, r.change}, {r.change_status, r.change}) && !same_action({r.change_status, shortened}, {r.change_status, r.change}), "premature success negative control");
        }
        {
            auto extended = r.change;
            extended.push_back(r.change.back());
            require(!same_action({r.change_status, extended}, {r.change_status, r.change}), "late stop negative control");
        }
        {
            require(r.faults.size() == 113 && std::all_of(r.faults.begin() + 105, r.faults.end(), [](const auto& fault) {
                        return fault.injected == -1 && fault.status != kOfxStatOK;
                    }),
                    "reference type failures");
        }
    }
    OfxStatus change(OfxPlugin* plugin, InstanceHost& host, OfxImageEffectHandle effect, const char* reason, const char* name) {
        return call(plugin, kOfxActionInstanceChanged, effect, host.change_args({reason, kOfxTypeParameter, name}));
    }
    void saved_control_getters(OfxPlugin* plugin, InstanceHost& host) {
        using ofx_test::InstanceValue;
        const std::array saved{
            std::pair{"ReferenceIlluminant", InstanceValue{2}},
            std::pair{"PrintShadowCompensationFactor", InstanceValue{0.5}},
            std::pair{"PrintShadowCompensationDensity", InstanceValue{1.5}},
            std::pair{"PrintShadowCompensationTransition", InstanceValue{0.75}},
        };
        auto* effect = host.create_effect();
        for (const auto& [name, value] : saved)
            host.set_value(effect, name, {value});
        const auto getters = [&] {
            std::vector<InstanceCall> reads;
            for (const auto& entry : host.calls())
                if (entry.operation == "param_get")
                    reads.push_back(entry);
            return reads;
        };
        require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "create saved-control instance");
        const auto* owner = host.instance_data(effect);
        const auto capture = [&] {
            return change(plugin, host, effect, kOfxChangeUserEdited, "Exposure");
        };
        require(capture() == kOfxStatReplyDefault, "initialize saved controls");
        host.clear_calls();
        require(capture() == kOfxStatReplyDefault, "load saved controls");
        const auto expected = getters();
        const auto require_order = [&](const auto& names) {
            require(std::ranges::search(expected, names, {}, &InstanceCall::target).begin() != expected.end(),
                    "saved capture getter order");
        };
        require_order(std::array{"SpectralUpsampling", "ReferenceIlluminant", "EnlargerIlluminant"});
        require_order(std::array{"PrintShadowCompensationFactor", "PrintShadowCompensationDensity", "PrintShadowCompensationTransition", "GlareActive"});
        for (const auto& [name, value] : saved) {
            const auto target = std::ranges::find_if(expected, [name](const auto& entry) {
                return entry.target == name;
            });
            require(target != expected.end() && target->values == std::vector{value},
                    std::string("saved getter value: ") + name);
            host.clear_calls();
            host.fail_next(std::string("param_get/") + name, kOfxStatErrMemory);
            require(capture() == kOfxStatErrMemory && host.instance_data(effect) == owner,
                    std::string("saved getter status and instance preservation: ") + name);
            const auto actual = getters();
            require(actual.size() == static_cast<std::size_t>(target - expected.begin()) + 1 &&
                        std::equal(actual.begin(), actual.end() - 1, expected.begin()) &&
                        actual.back().target == name && actual.back().status == kOfxStatErrMemory,
                    std::string("first failing saved getter: ") + name);
            host.clear_calls();
            require(capture() == kOfxStatReplyDefault && getters() == expected,
                    std::string("saved getter recovery: ") + name);
        }
        require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "release recovered instance");
    }

    void candidate_cohort(OfxPlugin* plugin, InstanceHost& host, const Reference& reference) {
        plugin->setHost(host.binding());
        require(call(plugin, kOfxActionLoad) == kOfxStatOK, "candidate load");
        require(call(plugin, kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK, "candidate context");
        auto* effect = host.create_effect();
        host.set_parameter_presentation(effect, "GrainChroma", kOfxParamPropHint, {std::string("returned hint")});
        host.return_string_on_failure_next("get_string/OfxParamPropHint", kOfxStatFailed);
        {
            host.clear_calls();
            require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "candidate create");
            require(host.instance_data(effect) != nullptr, "candidate publication");
            require(create_equivalent(host.calls(), reference.create), "constructor continuation");
            auto changed = reference.create;
            require(!changed.empty(), "reference constructor trace boundary");
            changed.front().operation = "altered";
            require(!create_equivalent(host.calls(), changed), "candidate accepted altered constructor reference");
            auto reordered = reference.create;
            const auto first = std::find_if(reordered.begin(), reordered.end(), [](const auto& c) {
                return c.operation == "param_handle";
            });
            require(first != reordered.end(), "reference constructor first parameter");
            const auto second = std::find_if(std::next(first), reordered.end(), [](const auto& c) {
                return c.operation == "param_handle";
            });
            require(second != reordered.end(), "reference constructor second parameter");
            std::iter_swap(first, second);
            require(!create_equivalent(host.calls(), reordered), "candidate accepted reordered constructor reference");
            auto shortened = host.calls();
            require(!shortened.empty(), "constructor trace boundary");
            shortened.pop_back();
            require(!create_equivalent(shortened, reference.create), "constructor premature success negative control");
            auto extended = host.calls();
            extended.push_back(host.calls().back());
            require(!create_equivalent(extended, reference.create), "constructor late stop negative control");
            require(std::count_if(host.calls().begin(), host.calls().end(), [](const auto& c) {
                        return c.operation == "param_properties";
                    }) == static_cast<std::ptrdiff_t>(ofx_test::kInstanceParameters.size() + kFilmParameters.size()),
                    "second parameter properties acquired");
            require(std::any_of(host.calls().begin(), host.calls().end(), [](const auto& c) {
                        return c.operation == "get_string" && c.target == "GrainChroma" && c.property == kOfxParamPropType;
                    }),
                    "type property identity");
            require(std::any_of(host.calls().begin(), host.calls().end(), [](const auto& c) {
                        return c.operation == "get_string" && c.target == "GrainChroma.presentation" && c.property == kOfxParamPropHint && c.status == kOfxStatFailed;
                    }),
                    "optional hint property identity and failure");
        }
        {
            require(constructor_census(host.calls()), "candidate constructor census");
            auto actual = handle_faults(plugin, host, true, &reference.faults);
            require(actual.size() == reference.faults.size() + 5, "handle fault membership");
        }
        {
            void* original = host.instance_data(effect);
            require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatErrFatal && host.instance_data(effect) == original, "duplicate create preserves owner");
        }
        {
            require(call(plugin, kOfxActionCreateInstance) == kOfxStatErrBadHandle, "null create effect");
        }
        {
            auto* probe = host.create_effect();
            host.null_next("effect_properties");
            require(call(plugin, kOfxActionCreateInstance, probe) == kOfxStatErrBadHandle && host.instance_data(probe) == nullptr, "null effect properties");
        }
        {
            auto* probe = host.create_effect();
            host.fail_next("effect_properties", kOfxStatErrMemory);
            require(call(plugin, kOfxActionCreateInstance, probe) == kOfxStatErrMemory && host.instance_data(probe) == nullptr, "create acquisition failure");
        }
        {
            auto* probe = host.create_effect();
            int nested = -1;
            host.reenter_next("param_handle/Exposure", [&] {
                nested = change(plugin, host, probe, kOfxChangePluginEdited, "Exposure");
            });
            require(call(plugin, kOfxActionCreateInstance, probe) == kOfxStatOK && nested == kOfxStatErrBadHandle, "nested change before publication");
            require(call(plugin, kOfxActionDestroyInstance, probe) == kOfxStatOK, "before publication cleanup");
        }
        {
            host.clear_calls();
            require(call(plugin, kOfxActionBeginInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "Exposure"})) == reference.begin_status && ofx_test::reference::calls(host.calls()) == ofx_test::reference::calls(reference.begin), "begin action");
            host.clear_calls();
            require(change(plugin, host, effect, kOfxChangeUserEdited, "Exposure") == reference.change_status && ofx_test::reference::calls(historical_instance_calls(host.calls())) == ofx_test::reference::calls(reference.change), "change action");
            require(std::any_of(host.calls().begin(), host.calls().end(), [](const auto& c) {
                        return c.operation == "set_string" && c.target == "GrainChroma.presentation" && c.property == kOfxParamPropHint && c.values == std::vector<ofx_test::InstanceValue>{std::string("returned hint")};
                    }),
                    "failed optional hint returned string retained");
            host.clear_calls();
            require(call(plugin, kOfxActionEndInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "Exposure"})) == reference.end_status && ofx_test::reference::calls(host.calls()) == ofx_test::reference::calls(reference.end), "end action");
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
        {
            host.set_value(effect, "CameraFilmFormatPreset", {1});
            host.clear_calls();
            require(change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset") == reference.edit_status && ofx_test::reference::calls(historical_instance_calls(host.calls())) == ofx_test::reference::calls(reference.edit), "camera preset continuation");
            require(host.value(effect, "CameraFilmFormatMm") == reference.edited_value, "camera preset value");
        }
        {
            host.set_value(effect, "CameraFilmFormatPreset", {2});
            host.clear_calls();
            host.reenter_next("param_set/CameraFilmFormatMm", [&] {
                require(change(plugin, host, effect, kOfxChangePluginEdited, "CameraFilmFormatMm") == reference.change_status, "candidate nested change");
            });
            require(change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset") == reference.reentry_status && ofx_test::reference::calls(historical_instance_calls(host.calls())) == ofx_test::reference::calls(reference.reentry), "nested edit continuation");
        }
        require(reference.edits.size() == edit_scenarios().size(), "edit membership");
        for (std::size_t i = 0; i < edit_scenarios().size(); ++i) {
            const auto& scenario = edit_scenarios()[i];
            const auto& expected = reference.edits[i];
            if (!scenario.authored.empty())
                host.set_value(effect, scenario.name, scenario.authored);
            host.clear_calls();
            require(change(plugin, host, effect, kOfxChangeUserEdited, scenario.name) == expected.status, "edit status");
            require(ofx_test::reference::calls(historical_instance_calls(host.calls())) == ofx_test::reference::calls(expected.calls) && ofx_test::reference::values(host.value(effect, scenario.observed)) == ofx_test::reference::values(expected.value), "edit continuation");
        }
        require(reference.actions.size() == kActionScenarios.size(), "action membership");
        for (std::size_t i = 0; i < kActionScenarios.size(); ++i) {
            const auto& scenario = kActionScenarios[i];
            const auto actual = run_action(plugin, host, effect, scenario);
            require(same_action(actual, reference.actions[i]), "action continuation");
        }
        require(reference.nested_changes.size() == 2, "nested change membership");
        for (std::size_t i = 0; i < reference.nested_changes.size(); ++i) {
            const char* reason = i == 0 ? kOfxChangeUserEdited : kOfxChangeTime;
            host.set_value(effect, "CameraFilmFormatPreset", {1});
            host.clear_calls();
            int nested = -1;
            host.reenter_next("param_set/CameraFilmFormatMm", [&] {
                nested = change(plugin, host, effect, reason, "Exposure");
            });
            const auto status = change(plugin, host, effect, kOfxChangeUserEdited, "CameraFilmFormatPreset");
            require(nested == reference.change_status && status == reference.nested_changes[i].status && ofx_test::reference::calls(historical_instance_calls(host.calls())) == ofx_test::reference::calls(reference.nested_changes[i].calls), "nested user/time continuation");
        }
        {
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
        }
        {
            int nested = -1;
            void* original = host.instance_data(effect);
            host.reenter_next("param_get/Exposure", [&] {
                nested = change(plugin, host, effect, kOfxChangePluginEdited, "Exposure");
            });
            require(change(plugin, host, effect, kOfxChangeUserEdited, "Exposure") == reference.change_status && nested == reference.change_status && host.instance_data(effect) == original, "nested change after publication");
        }
        {
            host.clear_calls();
            require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "candidate destroy");
            require(host.instance_data(effect) == nullptr, "candidate detached");
            require(ofx_test::reference::calls(host.calls()) == ofx_test::reference::calls(reference.destroy), "destroy continuation");
            const auto clear = std::find_if(host.calls().begin(), host.calls().end(), [](const auto& c) {
                return c.operation == "set_pointer" && c.property == kOfxPropInstanceData;
            });
            require(clear != host.calls().end() && std::next(clear) == host.calls().end(), "terminal pointer clear order");
            {
                require(call(plugin, kOfxActionDestroyInstance) == kOfxStatErrBadHandle, "null destroy");
            }
            {
                require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatErrBadHandle && host.instance_data(effect) == nullptr, "repeat destroy");
            }
        }
        const auto controls = control_faults(plugin, host, &reference.controls);
        require(controls.size() == reference.controls.size(), "control fault membership");
        const auto typed = typed_controls(plugin, host, &reference.typed);
        require(typed.size() == reference.typed.size(), "typed callback membership");
        const auto contexts = context_cases(plugin, host, &reference.contexts);
        require(contexts.size() == reference.contexts.size(), "context membership");
        acquisition_cases(plugin, host, reference.ordinary_create);
        saved_control_getters(plugin, host);
        require(call(plugin, kOfxActionUnload) == kOfxStatOK && !host.callback_error(), "candidate unload");
    }

    void grain_v2_constructor(OfxPlugin* plugin, InstanceHost& host, const Reference& reference) {
        constexpr std::array parameters{
            std::pair{"GrainModel", kOfxParamTypeStrChoice},
            std::pair{"FilmGrainAmountEv", kOfxParamTypeDouble},
            std::pair{"FilmGrainTexelPitchUm", kOfxParamTypeDouble},
            std::pair{"FilmGrainChromaFraction", kOfxParamTypeDouble},
        };
        auto expected = reference.create;
        require(!expected.empty() && expected.back().operation == "effect_properties", "Grain v2 initialization boundary");
        const auto scope = expected.back().target;
        const auto next = std::find_if(expected.begin(), expected.end(), [](const auto& entry) {
            return entry.operation == "param_handle" && entry.property == "GrainSublayersActive";
        });
        require(next != expected.end(), "Grain v2 constructor insertion boundary");
        const auto param_set = next->target;
        std::vector<InstanceCall> grain_v2_calls;
        for (const auto& parameter : parameters) {
            const auto* name = parameter.first;
            grain_v2_calls.push_back({"param_handle", param_set, name, 0, {}, {}});
            grain_v2_calls.push_back({"get_string", name, kOfxParamPropType, 0, {}, scope});
            grain_v2_calls.push_back({"param_properties", name, {}, 0, {}, scope});
        }
        expected.insert(next, grain_v2_calls.begin(), grain_v2_calls.end());
        // Initialize the Legacy model before the final instance check.
        std::vector<InstanceCall> visibility{{"param_get", "GrainModel", {}, 0, {std::string("current")}, scope}};
        for (const char* name : {"FilmGrainAmountEv", "FilmGrainTexelPitchUm", "FilmGrainChromaFraction"})
            visibility.push_back({"set_int", std::string(name) + ".presentation", kOfxParamPropSecret, 0, {1}, scope});
        for (const auto name : kLegacyVisibility)
            visibility.push_back({"set_int", std::string(name) + ".presentation", kOfxParamPropSecret, 0, {0}, scope});
        expected.insert(std::prev(expected.end()), visibility.begin(), visibility.end());

        plugin->setHost(host.binding());
        require(call(plugin, kOfxActionLoad) == kOfxStatOK, "Grain v2 load");
        host.clear_calls();
        require(call(plugin, kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK, "Grain v2 context");
        constexpr std::array controls{
            std::array{0.0, -3.0, 3.0, 0.1},
            std::array{1.0, 0.5, 4.0, 0.01},
            std::array{1.0, 0.0, 1.0, 0.01},
        };
        for (std::size_t i = 0; i < controls.size(); ++i) {
            const auto* name = parameters[i + 1].first;
            const auto [value, min, max, increment] = controls[i];
            for (const auto& [property, expected_value] : std::array{
                     std::pair{kOfxParamPropDefault, value},
                     std::pair{kOfxParamPropMin, min},
                     std::pair{kOfxParamPropMax, max},
                     std::pair{kOfxParamPropDisplayMin, min},
                     std::pair{kOfxParamPropDisplayMax, max},
                     std::pair{kOfxParamPropIncrement, increment},
                 }) {
                const auto setter = [&](const auto& entry) {
                    return entry.operation == "set_double" && entry.target == name &&
                           entry.property == property && entry.index == 0;
                };
                const auto entry = std::ranges::find_if(host.calls(), setter);
                require(entry != host.calls().end() && entry->status == kOfxStatOK &&
                            entry->values == std::vector<ofx_test::InstanceValue>{expected_value} &&
                            std::ranges::count_if(host.calls(), setter) == 1,
                        (std::string(name) + " unexpected descriptor property " + property).c_str());
            }
        }
        auto* effect = host.create_effect();
        {
            for (const auto& [name, type] : parameters)
                host.set_parameter_type(effect, name, type);
            host.set_parameter_presentation(effect, "GrainChroma", kOfxParamPropHint, {std::string("returned hint")});
            host.return_string_on_failure_next("get_string/OfxParamPropHint", kOfxStatFailed);
            host.clear_calls();
            require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK && host.instance_data(effect), "Grain v2 create and publish");
            require(create_equivalent(host.calls(), expected, true), "Grain v2 constructor continuation");
            auto altered = host.calls();
            const auto first = std::find(altered.begin(), altered.end(), grain_v2_calls.front());
            require(first != altered.end() && first != altered.begin(), "Grain v2 constructor call boundary");
            const auto index = first - altered.begin();
            altered.erase(first);
            require(!create_equivalent(altered, expected, true), "constructor accepted a missing Grain v2 call");
            altered = host.calls();
            std::iter_swap(altered.begin() + index, altered.begin() + index - 1);
            require(!create_equivalent(altered, expected, true), "constructor accepted a misplaced Grain v2 call");
            altered = host.calls();
            altered.front().operation = "altered";
            require(!create_equivalent(altered, expected, true), "constructor accepted a changed ordinary call");
        }
        {
            host.clear_calls();
            require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK && !host.instance_data(effect), "Grain v2 destroy and detach");
            require(ofx_test::reference::calls(host.calls()) == ofx_test::reference::calls(reference.destroy), "Grain v2 destroy continuation");
        }
        {
            for (const auto& parameter : parameters) {
                host.set_parameter_type(effect, parameter.first, kOfxParamTypeInteger);
                host.clear_calls();
                require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatFailed && !host.instance_data(effect), "Grain v2 parameter type failure and rollback");
                require(!host.calls().empty() && host.calls().back() == InstanceCall{"get_string", parameter.first, kOfxParamPropType, 0, {}, scope}, "Grain v2 parameter type terminal");
                host.set_parameter_type(effect, parameter.first, parameter.second);
            }
        }
        require(call(plugin, kOfxActionUnload) == kOfxStatOK && !host.callback_error(), "Grain v2 unload");
    }
} // namespace

int main(int argc, char** argv) try {
    const bool grain_v2 = argc > 1 && std::string_view(argv[1]) == "--grain-v2";
    require(argc == (grain_v2 ? 4 : 3) && std::string_view(argv[argc - 2]) == "--reference", "usage: instance fixture [--grain-v2] --reference file");
    auto* candidate = OfxGetPlugin(0);
    require(candidate && candidate->pluginIdentifier &&
                std::string_view(candidate->pluginIdentifier) == "com.juicer.Juicer",
            "instance descriptor variant");
    const auto expected = decode(ofx_test::reference::read(argv[argc - 1], "instance"));
    InstanceHost candidate_host;
    if (grain_v2) {
        grain_v2_constructor(candidate, candidate_host, expected);
    } else {
        reference_record_case(expected);
        candidate_cohort(candidate, candidate_host, expected);
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "instance fixture: %s\n", error.what());
    return 1;
}
