#include <algorithm>
#include <cstring>
#include <string_view>
#include <utility>

#include "ofxImageEffect.h"
#include "ofxImageEffectExt.h"
#include "ofxOpenGLRender.h"
#include "ofxParametricParam.h"
#include "ofxProgress.h"
#include "ofxTimeLine.h"

#include "context_host.h"

namespace {
    OfxStatus unused_set_pointer(OfxPropertySetHandle, const char*, int, void*) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_get_double(OfxPropertySetHandle, const char*, int, double*) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_clip_image(OfxImageClipHandle, OfxTime, const OfxRectD*, OfxPropertySetHandle*) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_release_image(OfxPropertySetHandle) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_clip_region(OfxImageClipHandle, OfxTime, OfxRectD*) noexcept {
        return kOfxStatFailed;
    }
    int unused_abort(OfxImageEffectHandle) noexcept {
        return 0;
    }
    OfxStatus unused_clip_handle(OfxImageEffectHandle, const char*, OfxImageClipHandle*, OfxPropertySetHandle*) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_param_handle(OfxParamSetHandle, const char*, OfxParamHandle*, OfxPropertySetHandle*) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_param_property(OfxParamHandle, OfxPropertySetHandle*) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_param_get(OfxParamHandle, ...) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_param_set(OfxParamHandle, ...) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_message(void*, const char*, const char*, const char*, ...) noexcept {
        return kOfxStatFailed;
    }
} // namespace

namespace ofx_test {
    ContextHost::ContextHost() {
        host_.host = reinterpret_cast<OfxPropertySetHandle>(&host_token_);
        host_.fetchSuite = fetch;
        effect_.getPropertySet = effect_properties;
        effect_.getParamSet = params;
        effect_.clipDefine = clip_define;
        parameter_.paramSetGetPropertySet = param_properties;
        parameter_.paramDefine = param_define;
        property_.propSetString = set_string;
        property_.propSetInt = set_int;
        property_.propSetDouble = set_double;
        property_.propGetString = get_string;
        property_.propGetInt = get_int;
        property_.propGetPointer = get_pointer;
        property_.propGetDimension = dimension;
        property_.propSetPointer = unused_set_pointer;
        property_.propGetDouble = unused_get_double;
        effect_.clipGetHandle = unused_clip_handle;
        effect_.clipGetImage = unused_clip_image;
        effect_.clipReleaseImage = unused_release_image;
        effect_.clipGetRegionOfDefinition = unused_clip_region;
        effect_.abort = unused_abort;
        parameter_.paramGetHandle = unused_param_handle;
        parameter_.paramGetPropertySet = unused_param_property;
        parameter_.paramGetValue = unused_param_get;
        parameter_.paramSetValue = unused_param_set;
        message_.message = unused_message;
    }
    OfxImageEffectHandle ContextHost::effect_handle() {
        return reinterpret_cast<OfxImageEffectHandle>(&effect_token_);
    }
    OfxPropertySetHandle ContextHost::context_handle() {
        return reinterpret_cast<OfxPropertySetHandle>(&input_);
    }
    void ContextHost::reset_trace() {
        calls.clear();
        effect_fetches_ = 0;
        callback_error = false;
    }
    OfxStatus ContextHost::record(ContextCall call) {
        const std::string key = call.property.empty() ? call.operation : call.operation + "/" + call.property;
        calls.push_back(std::move(call));
        if (reentry) {
            auto callback = std::move(reentry);
            callback();
        }
        const auto found = statuses.find(key);
        const OfxStatus result = found == statuses.end() ? kOfxStatOK : found->second;
        const auto later = status_after.find(key);
        if (later != status_after.end())
            statuses[later->second.first] = later->second.second;
        return result;
    }
    const void* ContextHost::fetch(OfxPropertySetHandle raw, const char* name, int version) noexcept {
        auto& self = *reinterpret_cast<Token*>(raw)->owner;
        try {
            self.startup.push_back({"fetch", name, {}, version, 0, 0});
        } catch (...) {
            self.callback_error = true;
            return nullptr;
        }
        if ((std::strcmp(name, kOfxProgressSuite) == 0 && (version == 1 || version == 2)) || (std::strcmp(name, kOfxTimeLineSuite) == 0 && version == 1) || (std::strcmp(name, kOfxOpenGLRenderSuite) == 0 && version == 1))
            return nullptr;
        if (version != 1) {
            self.callback_error = true;
            return nullptr;
        }
        if (std::strcmp(name, kOfxImageEffectSuite) == 0)
            return &self.effect_;
        if (std::strcmp(name, kOfxPropertySuite) == 0)
            return &self.property_;
        if (std::strcmp(name, kOfxParameterSuite) == 0)
            return &self.parameter_;
        if (std::strcmp(name, kOfxMemorySuite) == 0)
            return &self.memory_;
        if (std::strcmp(name, kOfxMultiThreadSuite) == 0)
            return &self.thread_;
        if (std::strcmp(name, kOfxMessageSuite) == 0)
            return &self.message_;
        self.callback_error = true;
        return nullptr;
    }
    OfxStatus ContextHost::effect_properties(OfxImageEffectHandle raw, OfxPropertySetHandle* out) noexcept {
        auto& s = *reinterpret_cast<Token*>(raw)->owner;
        try {
            bool second = s.effect_fetches_++ != 0;
            const char* key = second ? "effect_second" : "effect_first";
            auto status = s.record({key, {}, {}, 0, 0, 0});
            *out = s.null_outputs[key] ? nullptr : reinterpret_cast<OfxPropertySetHandle>(second && s.distinct_second ? &s.second_ : &s.first_);
            return status;
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::params(OfxImageEffectHandle raw, OfxParamSetHandle* out) noexcept {
        auto& s = *reinterpret_cast<Token*>(raw)->owner;
        try {
            auto status = s.record({"param_set", {}, {}, 0, 0, 0});
            *out = s.null_outputs["param_set"] ? nullptr : reinterpret_cast<OfxParamSetHandle>(&s.params_);
            return status;
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::param_properties(OfxParamSetHandle raw, OfxPropertySetHandle* out) noexcept {
        auto& s = *reinterpret_cast<Token*>(raw)->owner;
        try {
            auto status = s.record({"param_props", {}, {}, 0, 0, 0});
            *out = s.null_outputs["param_props"] ? nullptr : reinterpret_cast<OfxPropertySetHandle>(&s.param_props_);
            return status;
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::define(const char* op, const char* type, const char* name, OfxPropertySetHandle* out) {
        auto status = record({op, name, type, 0, 0, 0});
        if (null_outputs[op]) {
            *out = nullptr;
            return status;
        }
        auto& definition = definitions.emplace_back();
        definition.name = name;
        definition.type = type;
        definition.properties[kOfxPropType] = {std::string(std::strcmp(type, kOfxTypeClip) == 0 ? kOfxTypeClip : kOfxTypeParameter)};
        if (std::strcmp(type, kOfxTypeClip) == 0)
            definition.properties[kOfxImageClipPropOptional] = {1};
        else {
            definition.properties[kOfxParamPropEnabled] = {1};
            definition.properties[kOfxParamPropSecret] = {0};
            definition.properties[kOfxParamPropAnimates] = {1};
        }
        tokens_.push_back(std::make_unique<Token>(this, static_cast<int>(definitions.size()) + 6));
        *out = reinterpret_cast<OfxPropertySetHandle>(tokens_.back().get());
        return status;
    }
    OfxStatus ContextHost::clip_define(OfxImageEffectHandle raw, const char* name, OfxPropertySetHandle* out) noexcept {
        auto& s = *reinterpret_cast<Token*>(raw)->owner;
        try {
            return s.define("clip_define", kOfxTypeClip, name, out);
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::param_define(OfxParamSetHandle raw, const char* type, const char* name, OfxPropertySetHandle* out) noexcept {
        auto& s = *reinterpret_cast<Token*>(raw)->owner;
        try {
            return s.define("param_define", type, name, out);
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::set(Token& token, const char* property, int index, Value value, const char* operation) {
        auto status = record({operation, property, {}, index, token.target, value});
        if (status != kOfxStatOK && status != kOfxStatReplyYes && status != kOfxStatReplyNo && status != kOfxStatReplyDefault)
            return status;
        if (index < 0 || (token.target < 7 && token.target != 2))
            return kOfxStatErrBadHandle;
        auto& properties = token.target == 2 ? effect_values : definitions.at(static_cast<std::size_t>(token.target - 7)).properties;
        auto& values = properties[property];
        if (values.size() <= static_cast<std::size_t>(index))
            values.resize(static_cast<std::size_t>(index) + 1);
        values[static_cast<std::size_t>(index)] = std::move(value);
        return status;
    }
    OfxStatus ContextHost::set_string(OfxPropertySetHandle raw, const char* name, int index, const char* value) noexcept {
        auto& t = *reinterpret_cast<Token*>(raw);
        try {
            return t.owner->set(t, name, index, std::string(value), "string");
        } catch (...) {
            t.owner->callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::set_int(OfxPropertySetHandle raw, const char* name, int index, int value) noexcept {
        auto& t = *reinterpret_cast<Token*>(raw);
        try {
            return t.owner->set(t, name, index, value, "int");
        } catch (...) {
            t.owner->callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::set_double(OfxPropertySetHandle raw, const char* name, int index, double value) noexcept {
        auto& t = *reinterpret_cast<Token*>(raw);
        try {
            return t.owner->set(t, name, index, value, "double");
        } catch (...) {
            t.owner->callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::get_string(OfxPropertySetHandle raw, const char* name, int index, char** out) noexcept {
        auto& t = *reinterpret_cast<Token*>(raw);
        auto& s = *t.owner;
        try {
            if (t.target == 6 && index == 0 && std::strcmp(name, kOfxImageEffectPropContext) == 0) {
                auto status = s.record({"get_string", name, s.context, index, t.target, 0});
                *out = s.null_outputs["get_string"] ? nullptr : s.context.data();
                return status;
            }
            if (t.target == 0) {
                s.startup.push_back({"get_string", name, {}, index, 0, 0});
                constexpr std::string_view allowed[]{kOfxImageEffectHostPropNativeOrigin, kOfxImageEffectPropCudaRenderSupported, kOfxImageEffectPropCudaStreamSupported, kOfxImageEffectPropMetalRenderSupported, kOfxImageEffectPropOpenCLRenderSupported, kOfxImageEffectPropOpenGLRenderSupported, kOfxPropLabel, kOfxPropName, kOfxPropVersionLabel};
                if (index != 0 || std::find(std::begin(allowed), std::end(allowed), name) == std::end(allowed)) {
                    s.callback_error = true;
                    return kOfxStatFailed;
                }
                static char host[] = "Context test host";
                static char empty[] = "";
                *out = (std::strcmp(name, kOfxPropName) == 0 || std::strcmp(name, kOfxPropLabel) == 0) ? host : empty;
                return kOfxStatOK;
            }
            s.callback_error = true;
            return kOfxStatFailed;
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::get_int(OfxPropertySetHandle raw, const char* name, int index, int* out) noexcept {
        auto& t = *reinterpret_cast<Token*>(raw);
        auto& s = *t.owner;
        try {
            constexpr std::string_view allowed[]{kOfxImageEffectHostPropIsBackground, kOfxImageEffectInstancePropSequentialRender, kOfxImageEffectPropRenderQualityDraft, kOfxImageEffectPropSetableFielding, kOfxImageEffectPropSetableFrameRate, kOfxImageEffectPropSupportsMultiResolution, kOfxImageEffectPropSupportsMultipleClipDepths, kOfxImageEffectPropSupportsMultipleClipPARs, kOfxImageEffectPropSupportsOverlays, kOfxImageEffectPropSupportsTiles, kOfxImageEffectPropTemporalClipAccess, kOfxParamHostPropMaxPages, kOfxParamHostPropMaxParameters, kOfxParamHostPropPageRowColumnCount, kOfxParamHostPropSupportsBooleanAnimation, kOfxParamHostPropSupportsChoiceAnimation, kOfxParamHostPropSupportsCustomAnimation, kOfxParamHostPropSupportsCustomInteract, kOfxParamHostPropSupportsParametricAnimation, kOfxParamHostPropSupportsStrChoiceAnimation, kOfxParamHostPropSupportsStringAnimation, kOfxPropAPIVersion, kOfxPropVersion};
            int max_index = 0;
            if (std::strcmp(name, kOfxPropAPIVersion) == 0 || std::strcmp(name, kOfxParamHostPropPageRowColumnCount) == 0)
                max_index = 1;
            if (std::strcmp(name, kOfxPropVersion) == 0)
                max_index = 2;
            if (t.target != 0 || index < 0 || index > max_index || std::find(std::begin(allowed), std::end(allowed), name) == std::end(allowed)) {
                s.callback_error = true;
                return kOfxStatFailed;
            }
            s.startup.push_back({"get_int", name, {}, index, 0, 0});
            *out = 0;
            return kOfxStatOK;
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::get_pointer(OfxPropertySetHandle raw, const char* name, int index, void** out) noexcept {
        auto& t = *reinterpret_cast<Token*>(raw);
        auto& s = *t.owner;
        try {
            if (t.target != 0 || index != 0 || std::strcmp(name, kOfxPropHostOSHandle) != 0) {
                s.callback_error = true;
                return kOfxStatFailed;
            }
            s.startup.push_back({"get_pointer", name, {}, index, 0, 0});
            *out = nullptr;
            return kOfxStatOK;
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus ContextHost::dimension(OfxPropertySetHandle raw, const char* name, int* out) noexcept {
        auto& t = *reinterpret_cast<Token*>(raw);
        auto& s = *t.owner;
        try {
            if (t.target == 0) {
                if (std::strcmp(name, kOfxImageEffectPropSupportedComponents) != 0 && std::strcmp(name, kOfxImageEffectPropSupportedContexts) != 0 && std::strcmp(name, kOfxImageEffectPropSupportedPixelDepths) != 0) {
                    s.callback_error = true;
                    return kOfxStatFailed;
                }
                s.startup.push_back({"dimension", name, {}, 0, 0, 0});
                *out = 0;
                return kOfxStatOK;
            }
            auto status = s.record({"dimension", name, {}, 0, t.target, 0});
            const auto override = s.dimensions.find(name);
            *out = override != s.dimensions.end() ? override->second : (t.target == 2 ? static_cast<int>(s.effect_values[name].size()) : t.target >= 7 ? static_cast<int>(s.definitions.at(static_cast<std::size_t>(t.target - 7)).properties[name].size())
                                                                                                                                                       : 0);
            return status;
        } catch (...) {
            s.callback_error = true;
            return kOfxStatFailed;
        }
    }
} // namespace ofx_test
