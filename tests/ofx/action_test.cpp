#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "instance_host.h"
#include "juicer_ofx_api.h"
#include "ofxsImageEffect.h"
#include "../ffi/case_results.h"

namespace {
    using ofx_test::InstanceCall;
    using ofx_test::InstanceHost;
    constexpr std::string_view kCaller = "ofx.actions";
    struct Getter {
        const char* operation;
        const char* property;
        int index;
        bool optional;
    };
    struct Action {
        const char* id;
        const char* name;
        bool input;
        bool output;
        std::vector<Getter> getters;
    };
    const std::vector<Getter> kSequenceTail{
        {"get_double", kOfxImageEffectPropRenderScale, 0, false}, {"get_double", kOfxImageEffectPropRenderScale, 1, false}, {"get_int", "OfxImageEffectPropOpenCLEnabled", 0, true}, {"get_int", "OfxImageEffectPropCudaEnabled", 0, true}, {"get_int", "OfxImageEffectPropMetalEnabled", 0, true}, {"get_pointer", "OfxImageEffectPropOpenCLCommandQueue", 0, true}, {"get_pointer", "OfxImageEffectPropCudaStream", 0, true}, {"get_pointer", "OfxImageEffectPropMetalCommandQueue", 0, true}, {"get_int", kOfxPropIsInteractive, 0, false}, {"get_int", kOfxImageEffectPropSequentialRenderStatus, 0, true}, {"get_int", kOfxImageEffectPropInteractiveRenderStatus, 0, true}};
    std::vector<Getter> begin_getters() {
        std::vector<Getter> result{{"get_double", kOfxImageEffectPropFrameRange, 0, false},
                                   {"get_double", kOfxImageEffectPropFrameRange, 1, false},
                                   {"get_double", kOfxImageEffectPropFrameStep, 0, false}};
        result.insert(result.end(), kSequenceTail.begin(), kSequenceTail.end());
        return result;
    }
    const std::vector<Action> kActions{
        {"begin_sequence", kOfxImageEffectActionBeginSequenceRender, true, false, begin_getters()},
        {"end_sequence", kOfxImageEffectActionEndSequenceRender, true, false, kSequenceTail},
        {"identity", kOfxImageEffectActionIsIdentity, true, true, {{"get_double", kOfxPropTime, 0, false}, {"get_double", kOfxImageEffectPropRenderScale, 0, false}, {"get_double", kOfxImageEffectPropRenderScale, 1, false}, {"get_int", kOfxImageEffectPropRenderWindow, 0, false}, {"get_int", kOfxImageEffectPropRenderWindow, 1, false}, {"get_int", kOfxImageEffectPropRenderWindow, 2, false}, {"get_int", kOfxImageEffectPropRenderWindow, 3, false}, {"get_string", kOfxImageEffectPropFieldToRender, 0, false}}},
        {"region_of_definition", kOfxImageEffectActionGetRegionOfDefinition, true, true, {{"get_double", kOfxImageEffectPropRenderScale, 0, false}, {"get_double", kOfxImageEffectPropRenderScale, 1, false}, {"get_double", kOfxPropTime, 0, false}}},
        {"regions_of_interest", kOfxImageEffectActionGetRegionsOfInterest, true, true, {{"get_double", kOfxImageEffectPropRenderScale, 0, false}, {"get_double", kOfxImageEffectPropRenderScale, 1, false}, {"get_double", kOfxImageEffectPropRegionOfInterest, 0, false}, {"get_double", kOfxImageEffectPropRegionOfInterest, 1, false}, {"get_double", kOfxImageEffectPropRegionOfInterest, 2, false}, {"get_double", kOfxImageEffectPropRegionOfInterest, 3, false}, {"get_double", kOfxPropTime, 0, false}}},
        {"frames_needed", kOfxImageEffectActionGetFramesNeeded, true, true, {{"get_double", kOfxPropTime, 0, false}}},
        {"clip_preferences", kOfxImageEffectActionGetClipPreferences, false, true, {}},
        {"time_domain", kOfxImageEffectActionGetTimeDomain, false, true, {}},
        {"purge_caches", kOfxActionPurgeCaches, false, false, {}},
        {"sync_private_data", kOfxActionSyncPrivateData, false, false, {}},
        {"begin_edit", kOfxActionBeginInstanceEdit, false, false, {}},
        {"end_edit", kOfxActionEndInstanceEdit, false, false, {}},
        {"unknown", "OfxActionUnknownFixture", false, false, {}}};
    constexpr std::array kStatuses{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

    std::string getter_id(const Action& action, const Getter& getter, int status) {
        return std::string(action.id) + "/get/" + getter.operation + "/" + getter.property + "/" +
               std::to_string(getter.index) + "/" + std::to_string(status);
    }
    std::vector<std::string> row_ids() {
        std::vector<std::string> ids;
        for (const auto& action : kActions) {
            const std::string prefix = std::string(action.id) + "/";
            ids.push_back(prefix + "valid");
            if (prefix == "unknown/") {
                ids.push_back(prefix + "null_action");
                continue;
            }
            ids.push_back(prefix + "missing_effect");
            if (action.input)
                ids.push_back(prefix + "missing_input");
            if (action.output)
                ids.push_back(prefix + "missing_output");
            if (prefix == "identity/")
                ids.push_back(prefix + "null_field");
            for (const char* acquisition : {"effect", "instance"})
                for (int status : kStatuses)
                    ids.push_back(prefix + acquisition + "/" + std::to_string(status));
            for (const auto& getter : action.getters)
                for (int status : kStatuses)
                    ids.push_back(getter_id(action, getter, status));
        }
        return ids;
    }
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    bool accepted(int status) {
        return status == kOfxStatOK || status == kOfxStatReplyYes || status == kOfxStatReplyNo || status == kOfxStatReplyDefault;
    }
    OfxStatus property_status(int status) {
        if (accepted(status))
            return kOfxStatReplyDefault;
        if (status == kOfxStatErrUnknown || status == kOfxStatErrUnsupported)
            return kOfxStatErrMissingHostFeature;
        if (status == kOfxStatErrValue)
            return kOfxStatFailed;
        return status;
    }
    struct Observation {
        OfxStatus status;
        std::vector<InstanceCall> calls;
        std::vector<ofx_test::InstanceValue> output;
        bool operator==(const Observation&) const = default;
    };
    bool trace_matches(const std::vector<InstanceCall>& calls, const Action& action, std::size_t count) {
        if (std::string_view(action.id) == "unknown")
            return calls.empty();
        if (calls.size() != count + 2 || calls[0].operation != "effect_properties" ||
            calls[1].operation != "get_pointer" || calls[1].property != kOfxPropInstanceData)
            return false;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& actual = calls[i + 2];
            const auto& expected = action.getters[i];
            if (actual.operation != expected.operation || actual.property != expected.property || actual.index != expected.index)
                return false;
        }
        return true;
    }
    void populate(InstanceHost& host, OfxPropertySetHandle input, OfxPropertySetHandle output) {
        host.set_property(input, kOfxPropTime, {1.25});
        host.set_property(input, kOfxImageEffectPropRenderScale, {1.0, 0.5});
        host.set_property(input, kOfxImageEffectPropFrameRange, {0.0, 10.0});
        host.set_property(input, kOfxImageEffectPropFrameStep, {1.0});
        host.set_property(input, kOfxImageEffectPropRenderWindow, {0, 1, 20, 30});
        host.set_property(input, kOfxImageEffectPropRegionOfInterest, {0.0, 1.0, 20.0, 30.0});
        host.set_property(input, kOfxImageEffectPropFieldToRender, {std::string(kOfxImageFieldNone)});
        host.set_property(input, kOfxPropIsInteractive, {0});
        host.set_property(output, "fixture.sentinel", {42});
    }
    struct Fixture {
        OfxPlugin* plugin;
        InstanceHost& host;
        OfxImageEffectHandle effect;
        OfxPropertySetHandle input;
        OfxPropertySetHandle output;
        Fixture(OfxPlugin* p, InstanceHost& h) : plugin(p), host(h), effect(nullptr), input(h.action_args()), output(h.output_args()) {
            populate(host, input, output);
            plugin->setHost(host.binding());
            require(plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) == kOfxStatOK, "load");
            require(plugin->mainEntry(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args(), nullptr) == kOfxStatOK, "context");
            effect = host.create_effect();
            require(plugin->mainEntry(kOfxActionCreateInstance, effect, nullptr, nullptr) == kOfxStatOK, "create");
        }
        Observation call(const Action& action) {
            host.clear_calls();
            const auto status = plugin->mainEntry(action.name, effect, action.input ? input : nullptr, action.output ? output : nullptr);
            return {status, host.calls(), host.property(output, "fixture.sentinel")};
        }
        void close() {
            require(plugin->mainEntry(kOfxActionDestroyInstance, effect, nullptr, nullptr) == kOfxStatOK, "destroy");
            require(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatOK, "unload");
            require(!host.callback_error(), "host callback");
        }
    };
    using Observations = std::vector<std::vector<Observation>>;
    Observations exercise(Fixture& fixture, const Observations* expected) {
        Observations all;
        std::unique_ptr<JuicerTests::Cohort> cohort;
        if (expected) {
            const auto ids = row_ids();
            require(ids.size() == 1059, "action internal row count");
            cohort = std::make_unique<JuicerTests::Cohort>("ofx.actions.rows", ids);
        }
        for (const auto& action : kActions) {
            std::unique_ptr<JuicerTests::CaseEvent> row;
            if (expected)
                row = std::make_unique<JuicerTests::CaseEvent>(kCaller, action.id);
            const std::string prefix = std::string(action.id) + "/";
            auto begin_internal = [&](const std::string& id) {
                return cohort ? std::make_unique<JuicerTests::CaseEvent>("ofx.actions.rows", id) : nullptr;
            };
            auto pass_internal = [](const std::unique_ptr<JuicerTests::CaseEvent>& event) {
                if (event)
                    event->passed();
            };
            if (expected) {
                if (prefix == "unknown/") {
                    auto event = begin_internal(prefix + "null_action");
                    fixture.host.clear_calls();
                    require(fixture.plugin->mainEntry(nullptr, fixture.effect, nullptr, nullptr) == kOfxStatErrBadHandle && fixture.host.calls().empty(), "null action boundary");
                    pass_internal(event);
                } else {
                    auto event = begin_internal(prefix + "missing_effect");
                    fixture.host.clear_calls();
                    require(fixture.plugin->mainEntry(action.name, nullptr, action.input ? fixture.input : nullptr, action.output ? fixture.output : nullptr) == kOfxStatErrBadHandle && fixture.host.calls().empty(), "missing effect boundary");
                    pass_internal(event);
                    if (action.input) {
                        auto input_event = begin_internal(prefix + "missing_input");
                        fixture.host.clear_calls();
                        require(fixture.plugin->mainEntry(action.name, fixture.effect, nullptr, action.output ? fixture.output : nullptr) == kOfxStatErrBadHandle && fixture.host.calls().empty(), "missing input boundary");
                        pass_internal(input_event);
                    }
                    if (action.output) {
                        auto output_event = begin_internal(prefix + "missing_output");
                        fixture.host.clear_calls();
                        require(fixture.plugin->mainEntry(action.name, fixture.effect, action.input ? fixture.input : nullptr, nullptr) == kOfxStatErrBadHandle && fixture.host.calls().empty(), "missing output boundary");
                        pass_internal(output_event);
                    }
                }
            }
            std::vector<Observation> observations;
            auto check = [&](Observation actual) {
                if (expected)
                    require(actual == expected->at(all.size()).at(observations.size()), "native action observation");
                observations.push_back(std::move(actual));
            };
            auto valid_event = begin_internal(prefix + "valid");
            auto valid = fixture.call(action);
            require(valid.status == kOfxStatReplyDefault && valid.output == std::vector<ofx_test::InstanceValue>{42} && trace_matches(valid.calls, action, action.getters.size()), "literal valid action trace");
            auto bad = valid;
            bad.status = kOfxStatOK;
            require(!(bad == valid), "status negative control");
            bad = valid;
            bad.output = {43};
            require(!(bad == valid), "output negative control");
            if (!valid.calls.empty()) {
                bad = valid;
                bad.calls.pop_back();
                require(!(bad == valid) && !trace_matches(bad.calls, action, action.getters.size()), "missing getter negative control");
                bad = valid;
                bad.calls.push_back(bad.calls.back());
                require(!(bad == valid) && !trace_matches(bad.calls, action, action.getters.size()), "extra getter negative control");
            }
            if (action.getters.size() >= 2) {
                bad = valid;
                std::swap(bad.calls[2], bad.calls[3]);
                require(!(bad == valid) && !trace_matches(bad.calls, action, action.getters.size()), "reordered getter negative control");
            }
            check(std::move(valid));
            pass_internal(valid_event);
            if (std::string_view(action.id) == "identity") {
                auto event = begin_internal(prefix + "null_field");
                fixture.host.null_next("get_string/OfxImageEffectPropFieldToRender");
                auto null_field = fixture.call(action);
                require(null_field.status == kOfxStatReplyDefault && trace_matches(null_field.calls, action, action.getters.size()), "null field continuation");
                check(std::move(null_field));
                pass_internal(event);
            }
            if (std::string_view(action.id) != "unknown") {
                for (const char* op : {"effect_properties", "get_pointer/OfxPropInstanceData"}) {
                    for (int status : kStatuses) {
                        const auto* acquisition = std::string_view(op) == "effect_properties" ? "effect" : "instance";
                        auto event = begin_internal(prefix + acquisition + "/" + std::to_string(status));
                        fixture.host.fail_next(op, status);
                        auto result = fixture.call(action);
                        const auto expected_status = std::string_view(op) == "effect_properties"
                                                         ? (accepted(status) ? kOfxStatReplyDefault : status)
                                                         : property_status(status);
                        require(result.status == expected_status, "instance acquisition status");
                        check(std::move(result));
                        pass_internal(event);
                    }
                }
                for (std::size_t i = 0; i < action.getters.size(); ++i) {
                    const auto& getter = action.getters[i];
                    int occurrence = 1;
                    for (std::size_t j = 0; j < i; ++j)
                        if (std::string_view(action.getters[j].operation) == getter.operation && std::string_view(action.getters[j].property) == getter.property)
                            ++occurrence;
                    for (int status : kStatuses) {
                        auto event = begin_internal(getter_id(action, getter, status));
                        fixture.host.fail_on_nth(occurrence, std::string(getter.operation) + "/" + getter.property, status);
                        auto result = fixture.call(action);
                        const auto count = getter.optional || accepted(status) ? action.getters.size() : i + 1;
                        require(result.status == (getter.optional ? kOfxStatReplyDefault : property_status(status)), "getter status");
                        require(trace_matches(result.calls, action, count), "getter failure prefix or continuation");
                        check(std::move(result));
                        pass_internal(event);
                    }
                }
            }
            all.push_back(std::move(observations));
            if (row)
                row->passed();
        }
        return all;
    }
} // namespace

int main(int argc, char** argv) try {
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto& action : kActions)
            JuicerTests::list_case(kCaller, action.id);
        return 0;
    }
    JuicerTests::CaseLog log;
    auto* reference = OfxGetPlugin(0);
    auto* candidate = fj_ofx_plugin(0);
    require(reference && candidate && reference->mainEntry != candidate->mainEntry, "independent descriptors");
    InstanceHost reference_host;
    Fixture native(reference, reference_host);
    const auto expected = exercise(native, nullptr);
    native.close();
    reference = nullptr;
    InstanceHost candidate_host;
    Fixture rust(candidate, candidate_host);
    exercise(rust, &expected);
    rust.close();
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "action fixture: %s\n", error.what());
    return 1;
}
