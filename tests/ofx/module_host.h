#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProperty.h"

namespace ofx_test {
    struct Call {
        std::string operation;
        std::string property;
        int index = 0;
        std::string text;
        int integer = 0;
        int target = 0;
    };

    class Host {
    public:
        Host();
        Host(const Host&) = delete;
        Host& operator=(const Host&) = delete;
        OfxHost* binding() {
            return &host;
        }
        OfxImageEffectHandle effect_handle();
        void reset_trace();

        OfxHost host{};
        OfxPropertySuiteV1 property{};
        OfxImageEffectSuiteV1 effect{};
        OfxParameterSuiteV1 parameter{};
        OfxMessageSuiteV1 message{};
        std::vector<Call> calls;
        std::map<std::string, OfxStatus> statuses;
        std::map<std::string, bool> null_outputs;
        std::string missing_suite;
        std::function<void()> reentry;
        bool callback_error = false;
        int context_dimension = 0;
        int depth_dimension = 0;
        bool distinct_second = false;

    private:
        struct Token {
            Host* owner;
            int target;
        };
        Token effect_token_{this, 0};
        Token first_{this, 1};
        Token second_{this, 2};
        Token params_{this, 3};
        Token param_props_{this, 4};
        int effect_fetches_ = 0;

        OfxStatus record(Call call);
        static const void* fetch(OfxPropertySetHandle, const char*, int) noexcept;
        static OfxStatus get_effect_properties(OfxImageEffectHandle, OfxPropertySetHandle*) noexcept;
        static OfxStatus get_params(OfxImageEffectHandle, OfxParamSetHandle*) noexcept;
        static OfxStatus get_param_properties(OfxParamSetHandle, OfxPropertySetHandle*) noexcept;
        static OfxStatus set_string(OfxPropertySetHandle, const char*, int, const char*) noexcept;
        static OfxStatus set_int(OfxPropertySetHandle, const char*, int, int) noexcept;
        static OfxStatus get_string(OfxPropertySetHandle, const char*, int, char**) noexcept;
        static OfxStatus set_double(OfxPropertySetHandle, const char*, int, double) noexcept;
        static OfxStatus define_clip(OfxImageEffectHandle, const char*, OfxPropertySetHandle*) noexcept;
        static OfxStatus define_param(OfxParamSetHandle, const char*, const char*, OfxPropertySetHandle*) noexcept;
        static OfxStatus get_dimension(OfxPropertySetHandle, const char*, int*) noexcept;
    };
} // namespace ofx_test
