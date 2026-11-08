#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "ofxImageEffectExt.h"

#include "module_host.h"

namespace {
    OfxStatus unused_set_pointer(OfxPropertySetHandle, const char*, int, void*) noexcept {
        return kOfxStatFailed;
    }
    OfxStatus unused_get_double(OfxPropertySetHandle, const char*, int, double*) noexcept {
        return kOfxStatFailed;
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
    OfxStatus unused_get_pointer(OfxPropertySetHandle, const char*, int, void**) noexcept {
        return kOfxStatFailed;
    }
} // namespace

namespace ofx_test {
    Host::Host() {
        host.host = reinterpret_cast<OfxPropertySetHandle>(&effect_token_);
        host.fetchSuite = fetch;
        effect.getPropertySet = get_effect_properties;
        effect.getParamSet = get_params;
        parameter.paramSetGetPropertySet = get_param_properties;
        property.propSetString = set_string;
        property.propSetInt = set_int;
        property.propGetDimension = get_dimension;
        property.propGetString = get_string;
        property.propSetDouble = set_double;
        effect.clipDefine = define_clip;
        parameter.paramDefine = define_param;
        property.propSetPointer = unused_set_pointer;
        property.propGetDouble = unused_get_double;
        effect.clipGetHandle = unused_clip_handle;
        parameter.paramGetHandle = unused_param_handle;
        parameter.paramGetPropertySet = unused_param_property;
        parameter.paramGetValue = unused_param_get;
        parameter.paramSetValue = unused_param_set;
        message.message = unused_message;
        property.propGetPointer = unused_get_pointer;
    }

    OfxImageEffectHandle Host::effect_handle() {
        return reinterpret_cast<OfxImageEffectHandle>(&effect_token_);
    }
    void Host::reset_trace() {
        calls.clear();
        effect_fetches_ = 0;
        callback_error = false;
    }

    OfxStatus Host::record(Call call) {
        const std::string key = call.property.empty() ? call.operation : call.operation + "/" + call.property;
        calls.push_back(std::move(call));
        if (reentry) {
            auto callback = std::move(reentry);
            callback();
        }
        const auto found = statuses.find(key);
        return found == statuses.end() ? kOfxStatOK : found->second;
    }

    const void* Host::fetch(OfxPropertySetHandle raw, const char* name, int version) noexcept {
        auto& self = *reinterpret_cast<Token*>(raw)->owner;
        try {
            self.calls.push_back({"fetch", name, version, {}, 0, 0});
            if (self.reentry) {
                auto callback = std::move(self.reentry);
                callback();
            }
            if (version != 1 || self.missing_suite == name)
                return nullptr;
            if (std::strcmp(name, kOfxPropertySuite) == 0)
                return &self.property;
            if (std::strcmp(name, kOfxImageEffectSuite) == 0)
                return &self.effect;
            if (std::strcmp(name, kOfxParameterSuite) == 0)
                return &self.parameter;
            if (std::strcmp(name, kOfxMessageSuite) == 0)
                return &self.message;
            self.callback_error = true;
        } catch (...) {
            self.callback_error = true;
        }
        return nullptr;
    }

    OfxStatus Host::get_effect_properties(OfxImageEffectHandle raw, OfxPropertySetHandle* out) noexcept {
        auto& self = *reinterpret_cast<Token*>(raw)->owner;
        try {
            const bool second = self.effect_fetches_++ != 0;
            const std::string key = second ? "effect_second" : "effect_first";
            const auto status = self.record({key, {}, 0, {}, 0, 0});
            *out = self.null_outputs[key] ? nullptr : reinterpret_cast<OfxPropertySetHandle>(second && self.distinct_second ? &self.second_ : &self.first_);
            return status;
        } catch (...) {
            self.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus Host::get_params(OfxImageEffectHandle raw, OfxParamSetHandle* out) noexcept {
        auto& self = *reinterpret_cast<Token*>(raw)->owner;
        try {
            const auto status = self.record({"param_set", {}, 0, {}, 0, 0});
            *out = self.null_outputs["param_set"] ? nullptr : reinterpret_cast<OfxParamSetHandle>(&self.params_);
            return status;
        } catch (...) {
            self.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus Host::get_param_properties(OfxParamSetHandle raw, OfxPropertySetHandle* out) noexcept {
        auto& self = *reinterpret_cast<Token*>(raw)->owner;
        try {
            const auto status = self.record({"param_props", {}, 0, {}, 0, 0});
            *out = self.null_outputs["param_props"] ? nullptr : reinterpret_cast<OfxPropertySetHandle>(&self.param_props_);
            return status;
        } catch (...) {
            self.callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus Host::set_string(OfxPropertySetHandle raw, const char* property_name, int index, const char* value) noexcept {
        auto& token = *reinterpret_cast<Token*>(raw);
        try {
            constexpr std::array allowed{kOfxPropLabel, kOfxPropShortLabel, kOfxPropLongLabel, kOfxImageEffectPluginPropGrouping, kOfxImageEffectPropSupportedContexts, kOfxImageEffectPropSupportedPixelDepths, kOfxImageEffectPluginRenderThreadSafety, kOfxImageEffectPropCudaRenderSupported, kOfxImageEffectPropCudaStreamSupported};
            if (std::find(allowed.begin(), allowed.end(), std::string_view(property_name)) == allowed.end())
                throw std::runtime_error("unexpected string property");
            return token.owner->record({"string", property_name, index, value, 0, token.target});
        } catch (...) {
            token.owner->callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus Host::set_int(OfxPropertySetHandle raw, const char* property_name, int index, int value) noexcept {
        auto& token = *reinterpret_cast<Token*>(raw);
        try {
            constexpr std::array allowed{kOfxImageEffectPluginPropSingleInstance, kOfxImageEffectPluginPropHostFrameThreading, kOfxImageEffectPropSupportsMultiResolution, kOfxImageEffectPropSupportsTiles};
            if (std::find(allowed.begin(), allowed.end(), std::string_view(property_name)) == allowed.end())
                throw std::runtime_error("unexpected integer property");
            return token.owner->record({"int", property_name, index, {}, value, token.target});
        } catch (...) {
            token.owner->callback_error = true;
            return kOfxStatFailed;
        }
    }
    OfxStatus Host::get_string(OfxPropertySetHandle raw, const char*, int, char**) noexcept {
        reinterpret_cast<Token*>(raw)->owner->callback_error = true;
        return kOfxStatFailed;
    }
    OfxStatus Host::set_double(OfxPropertySetHandle raw, const char*, int, double) noexcept {
        reinterpret_cast<Token*>(raw)->owner->callback_error = true;
        return kOfxStatFailed;
    }
    OfxStatus Host::define_clip(OfxImageEffectHandle raw, const char*, OfxPropertySetHandle*) noexcept {
        reinterpret_cast<Token*>(raw)->owner->callback_error = true;
        return kOfxStatFailed;
    }
    OfxStatus Host::define_param(OfxParamSetHandle raw, const char*, const char*, OfxPropertySetHandle*) noexcept {
        reinterpret_cast<Token*>(raw)->owner->callback_error = true;
        return kOfxStatFailed;
    }
    OfxStatus Host::get_dimension(OfxPropertySetHandle raw, const char* property_name, int* out) noexcept {
        auto& token = *reinterpret_cast<Token*>(raw);
        try {
            const std::string_view name(property_name);
            if (name != kOfxImageEffectPropSupportedContexts && name != kOfxImageEffectPropSupportedPixelDepths)
                throw std::runtime_error("unexpected dimension property");
            *out = name == kOfxImageEffectPropSupportedContexts ? token.owner->context_dimension : token.owner->depth_dimension;
            return token.owner->record({"dimension", property_name, 0, {}, 0, token.target});
        } catch (...) {
            token.owner->callback_error = true;
            return kOfxStatFailed;
        }
    }
} // namespace ofx_test
