#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "ofxImageEffect.h"
#include "ofxMemory.h"
#include "ofxMessage.h"
#include "ofxMultiThread.h"
#include "ofxParam.h"
#include "ofxParamExt.h"
#include "ofxProperty.h"

namespace ofx_test {
    using Value = std::variant<int, double, std::string>;
    struct ContextCall {
        std::string operation, property, text;
        int index = 0, target = 0;
        Value value = 0;
        bool operator==(const ContextCall&) const = default;
    };
    struct Definition {
        Definition() = default;
        Definition(const Definition&) = default;
        Definition& operator=(const Definition&) = default;
        // MSVC's map move can allocate its sentinel.
        Definition(Definition&&) noexcept(false) = default;
        std::string name, type;
        std::map<std::string, std::vector<Value>> properties;
        bool operator==(const Definition&) const = default;
    };
    class ContextHost {
    public:
        ContextHost();
        ContextHost(const ContextHost&) = delete;
        ContextHost& operator=(const ContextHost&) = delete;
        OfxHost* binding() {
            return &host_;
        }
        OfxImageEffectHandle effect_handle();
        OfxPropertySetHandle context_handle();
        void reset_trace();
        std::vector<ContextCall> calls;
        std::vector<ContextCall> startup;
        std::map<std::string, std::vector<Value>> effect_values;
        std::vector<Definition> definitions;
        std::map<std::string, OfxStatus> statuses;
        std::map<std::string, std::pair<std::string, OfxStatus>> status_after;
        std::map<std::string, bool> null_outputs;
        std::map<std::string, int> dimensions;
        std::function<void()> reentry;
        std::string context = kOfxImageEffectContextFilter;
        bool callback_error = false;
        bool distinct_second = false;

    private:
        struct Token {
            ContextHost* owner;
            int target;
        };
        Token host_token_{this, 0}, effect_token_{this, 1}, first_{this, 2}, second_{this, 3}, params_{this, 4}, param_props_{this, 5}, input_{this, 6};
        std::vector<std::unique_ptr<Token>> tokens_;
        OfxHost host_{};
        OfxPropertySuiteV1 property_{};
        OfxImageEffectSuiteV1 effect_{};
        OfxParameterSuiteV1 parameter_{};
        OfxMessageSuiteV1 message_{};
        OfxMemorySuiteV1 memory_{};
        OfxMultiThreadSuiteV1 thread_{};
        int effect_fetches_ = 0;
        OfxStatus record(ContextCall call);
        static const void* fetch(OfxPropertySetHandle, const char*, int) noexcept;
        static OfxStatus effect_properties(OfxImageEffectHandle, OfxPropertySetHandle*) noexcept;
        static OfxStatus params(OfxImageEffectHandle, OfxParamSetHandle*) noexcept;
        static OfxStatus param_properties(OfxParamSetHandle, OfxPropertySetHandle*) noexcept;
        static OfxStatus clip_define(OfxImageEffectHandle, const char*, OfxPropertySetHandle*) noexcept;
        static OfxStatus param_define(OfxParamSetHandle, const char*, const char*, OfxPropertySetHandle*) noexcept;
        static OfxStatus set_string(OfxPropertySetHandle, const char*, int, const char*) noexcept;
        static OfxStatus set_int(OfxPropertySetHandle, const char*, int, int) noexcept;
        static OfxStatus set_double(OfxPropertySetHandle, const char*, int, double) noexcept;
        static OfxStatus get_string(OfxPropertySetHandle, const char*, int, char**) noexcept;
        static OfxStatus get_int(OfxPropertySetHandle, const char*, int, int*) noexcept;
        static OfxStatus get_pointer(OfxPropertySetHandle, const char*, int, void**) noexcept;
        static OfxStatus dimension(OfxPropertySetHandle, const char*, int*) noexcept;
        OfxStatus define(const char*, const char*, const char*, OfxPropertySetHandle*);
        OfxStatus set(Token&, const char*, int, Value, const char*);
    };
} // namespace ofx_test
