#include "instance_host.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

#include "ofxImageEffectExt.h"
#include "ofxMemory.h"
#include "ofxMessage.h"
#include "ofxMultiThread.h"
#include "ofxParamExt.h"
#include "ofxParametricParam.h"
#include "ofxOpenGLRender.h"
#include "ofxProperty.h"

namespace ofx_test {
    struct InstanceHost::Impl {
        struct Bag {
            Bag(Impl* owner, std::string name) : owner(owner), name(std::move(name)) {}
            Bag(Bag&&) = delete;
            Impl* owner;
            std::string name;
            std::map<std::string, std::vector<InstanceValue>> properties;
            std::string scope;
        };
        struct Param {
            Param(Impl* owner, const std::string& name, InstanceParamKind kind)
                : bag(owner, name), presentation(owner, name + ".presentation"), kind(kind) {}
            Param(Param&&) = delete;
            Bag bag;
            Bag presentation;
            InstanceParamKind kind;
            std::vector<InstanceValue> values;
        };
        struct Clip {
            Clip(Impl* owner, std::string name) : bag(owner, std::move(name)) {
                bag.properties[kOfxImageEffectPropPixelDepth] = {std::string(kOfxBitDepthFloat)};
                bag.properties[kOfxImageEffectPropSupportedComponents] = {std::string(kOfxImageComponentRGBA)};
                bag.properties[kOfxImageEffectPropFrameRate] = {24.0};
                image_properties[kOfxImagePropData] = {static_cast<void*>(nullptr)};
                image_properties[kOfxImagePropBounds] = {0, 0, 0, 0};
                image_properties[kOfxImagePropRowBytes] = {0};
                image_properties[kOfxImageEffectPropPixelDepth] = {std::string(kOfxBitDepthFloat)};
                image_properties[kOfxImageEffectPropComponents] = {std::string(kOfxImageComponentRGBA)};
                image_properties[kOfxImageEffectPropPreMultiplication] = {std::string("OfxImageAlphaUnPremultiplied")};
                image_properties[kOfxImagePropField] = {std::string(kOfxImageFieldNone)};
                image_properties[kOfxImagePropUniqueIdentifier] = {std::string("instance-host-image")};
                image_properties[kOfxImagePropRegionOfDefinition] = {0, 0, 0, 0};
                image_properties[kOfxImageEffectPropRenderScale] = {1.0, 1.0};
                image_properties[kOfxImagePropPixelAspectRatio] = {1.0};
            }
            Clip(Clip&&) = delete;
            Bag bag;
            OfxRectD region{0.0, 0.0, 0.0, 0.0};
            OfxStatus image_status = kOfxStatOK;
            OfxStatus release_status = kOfxStatOK;
            bool image_available = true;
            std::size_t releases = 0;
            std::map<std::string, std::vector<InstanceValue>> image_properties;
            struct Image {
                Image(Impl* owner, std::string name) : bag(owner, std::move(name)) {}
                Bag bag;
                std::string clip_name;
            };
            std::vector<std::unique_ptr<Image>> images;
        };
        struct Effect {
            Effect(Impl* owner, const std::string& name) : bag(owner, name), param_set(owner, name + ".params") {}
            Effect(Effect&&) = delete;
            Bag bag;
            Bag param_set;
            std::map<std::string, std::unique_ptr<Param>> params;
            std::map<std::string, std::unique_ptr<Clip>> clips;
        };
        OfxHost host{};
        OfxPropertySuiteV1 property{};
        OfxImageEffectSuiteV1 effect_suite{};
        OfxParameterSuiteV1 param_suite{};
        OfxMessageSuiteV1 message{};
        OfxMemorySuiteV1 memory{};
        OfxMultiThreadSuiteV1 threads{};
        Bag host_bag{this, "host"}, context{this, "context"}, action{this, "action"}, output{this, "output"};
        std::vector<std::unique_ptr<Bag>> changes;
        Effect descriptor{this, "descriptor"};
        std::vector<std::unique_ptr<Effect>> effects;
        std::vector<InstanceCall> trace;
        std::string fail_operation, reentry_operation, null_operation;
        OfxStatus fail_status = kOfxStatOK;
        int fail_remaining = 1;
        int null_remaining = 1;
        std::function<void()> reentry;
        bool error = false;
        bool fail_after_store = false;
        bool current_store_failure = false;
        bool return_failed_string = false;
        bool reuse_string_storage = false;
        bool string_storage_used = false;
        bool abort_requested = false;
        std::size_t string_reuses = 0;
        std::array<char, 4096> string_storage{};

        static Bag& bag(void* p) {
            return *reinterpret_cast<Bag*>(p);
        }
        static Effect& effect(void* p) {
            return *reinterpret_cast<Effect*>(p);
        }
        static Param& param(void* p) {
            return *reinterpret_cast<Param*>(p);
        }
        static Clip& clip(void* p) {
            return *reinterpret_cast<Clip*>(p);
        }
        static Impl& owner(void* p) {
            return *bag(p).owner;
        }
        OfxStatus record(InstanceCall call, std::string_view scope = {}) {
            call.scope = scope;
            current_store_failure = false;
            const std::string operation = call.operation;
            const std::string qualified = operation + "/" + call.property;
            const std::string targeted = operation + "/" + call.target;
            const auto trace_index = trace.size();
            trace.push_back(std::move(call));
            OfxStatus result = kOfxStatOK;
            bool store_failure = false;
            if ((operation == fail_operation || qualified == fail_operation || targeted == fail_operation) && --fail_remaining == 0) {
                fail_operation.clear();
                result = fail_status;
                store_failure = fail_after_store;
                fail_after_store = false;
            }
            if (operation == reentry_operation || qualified == reentry_operation || targeted == reentry_operation) {
                reentry_operation.clear();
                auto callback = std::move(reentry);
                if (callback)
                    callback();
            }
            current_store_failure = store_failure;
            trace[trace_index].status = result;
            return result;
        }
        bool take_null(const std::string& operation) {
            if (null_operation != operation || --null_remaining != 0)
                return false;
            null_operation.clear();
            return true;
        }
        static bool accepted(OfxStatus status) noexcept {
            return status == kOfxStatOK || status == kOfxStatReplyYes ||
                   status == kOfxStatReplyNo || status == kOfxStatReplyDefault;
        }
        template <class F>
        static OfxStatus guarded(Impl& self, F&& f) noexcept {
            try {
                return f();
            } catch (...) {
                self.error = true;
                return kOfxStatFailed;
            }
        }
        static const char* type(InstanceParamKind kind) {
            switch (kind) {
                case InstanceParamKind::Double:
                    return kOfxParamTypeDouble;
                case InstanceParamKind::Boolean:
                    return kOfxParamTypeBoolean;
                case InstanceParamKind::Choice:
                    return kOfxParamTypeChoice;
                case InstanceParamKind::StringChoice:
                    return kOfxParamTypeStrChoice;
                case InstanceParamKind::Integer:
                    return kOfxParamTypeInteger;
                case InstanceParamKind::Double2D:
                    return kOfxParamTypeDouble2D;
                case InstanceParamKind::Double3D:
                    return kOfxParamTypeDouble3D;
                case InstanceParamKind::PushButton:
                    return kOfxParamTypePushButton;
            }
            throw std::runtime_error("unknown parameter kind");
        }
        static InstanceParamKind kind(const char* name) {
            for (auto entry : kInstanceParameters)
                if (entry.name == name)
                    return entry.kind;
            throw std::runtime_error("unexpected instance parameter");
        }
        static const void* fetch(OfxPropertySetHandle raw, const char* name, int version) noexcept {
            auto& self = owner(raw);
            try {
                if (self.record({"fetch", "host", name, version, {}, {}}) != kOfxStatOK)
                    return nullptr;
                if (version != 1)
                    return nullptr;
                if (!std::strcmp(name, kOfxPropertySuite))
                    return &self.property;
                if (!std::strcmp(name, kOfxImageEffectSuite))
                    return &self.effect_suite;
                if (!std::strcmp(name, kOfxParameterSuite))
                    return &self.param_suite;
                if (!std::strcmp(name, kOfxMessageSuite))
                    return &self.message;
                if (!std::strcmp(name, kOfxMemorySuite))
                    return &self.memory;
                if (!std::strcmp(name, kOfxMultiThreadSuite))
                    return &self.threads;
                return nullptr;
            } catch (...) {
                self.error = true;
                return nullptr;
            }
        }
        static OfxStatus set(Bag& b, const char* key, int index, InstanceValue value, const char* op) {
            if (!key || index < 0)
                return kOfxStatErrBadIndex;
            auto status = b.owner->record({op, b.name, key, index, {value}, {}}, b.scope);
            if (!accepted(status) && !b.owner->current_store_failure)
                return status;
            auto& values = b.properties[key];
            if (values.size() <= static_cast<std::size_t>(index))
                values.resize(static_cast<std::size_t>(index) + 1, value);
            values[static_cast<std::size_t>(index)] = std::move(value);
            return status;
        }
        template <class T>
        static OfxStatus get(Bag& b, const char* key, int index, T* out, const char* op) {
            if (!key || !out || index < 0)
                return kOfxStatErrBadIndex;
            auto status = b.owner->record({op, b.name, key, index, {}, {}}, b.scope);
            if (!accepted(status))
                return status;
            auto found = b.properties.find(key);
            if (found == b.properties.end())
                return kOfxStatErrUnknown;
            if (static_cast<std::size_t>(index) >= found->second.size())
                return kOfxStatErrBadIndex;
            auto* value = std::get_if<T>(&found->second[static_cast<std::size_t>(index)]);
            if (!value)
                return kOfxStatErrValue;
            *out = *value;
            return status;
        }
#define PROP_SET(NAME, TYPE, VALUE)                                                        \
    static OfxStatus NAME(OfxPropertySetHandle h, const char* n, int i, TYPE v) noexcept { \
        if (!h)                                                                            \
            return kOfxStatErrBadHandle;                                                   \
        auto& b = bag(h);                                                                  \
        return guarded(*b.owner, [&] {                                                     \
            return set(b, n, i, VALUE, #NAME);                                             \
        });                                                                                \
    }
        PROP_SET(set_int, int, v)
        PROP_SET(set_double, double, v)
        PROP_SET(set_pointer, void*, v)
        PROP_SET(set_string, const char*, std::string(v ? v : ""))
#undef PROP_SET
#define PROP_GET(NAME, TYPE)                                                                \
    static OfxStatus NAME(OfxPropertySetHandle h, const char* n, int i, TYPE* v) noexcept { \
        if (!h)                                                                             \
            return kOfxStatErrBadHandle;                                                    \
        auto& b = bag(h);                                                                   \
        return guarded(*b.owner, [&] {                                                      \
            return get(b, n, i, v, #NAME);                                                  \
        });                                                                                 \
    }
        PROP_GET(get_int, int)
        PROP_GET(get_double, double)
        PROP_GET(get_pointer, void*)
#undef PROP_GET
        static OfxStatus get_string(OfxPropertySetHandle h, const char* n, int i, char** out) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& b = bag(h);
            return guarded(*b.owner, [&] {
                auto status = b.owner->record({"get_string", b.name, n, i, {}, {}}, b.scope);
                const bool return_value = b.owner->return_failed_string && std::strcmp(n, kOfxParamPropHint) == 0;
                if (return_value)
                    b.owner->return_failed_string = false;
                if (!accepted(status) && !return_value)
                    return status;
                if (b.owner->take_null(std::string("get_string/") + n)) {
                    *out = nullptr;
                    return status;
                }
                auto it = b.properties.find(n);
                if (it == b.properties.end())
                    return kOfxStatErrUnknown;
                if (i < 0 || static_cast<std::size_t>(i) >= it->second.size())
                    return kOfxStatErrBadIndex;
                auto* value = std::get_if<std::string>(&it->second[static_cast<std::size_t>(i)]);
                if (!value)
                    return kOfxStatErrValue;
                *out = value->data();
                return status;
            });
        }
        template <class T, class F>
        static OfxStatus array_get(OfxPropertySetHandle h, const char* n, int count, T* values, F get_one) noexcept {
            for (int i = 0; i < count; ++i) {
                auto status = get_one(h, n, i, &values[i]);
                if (!accepted(status))
                    return status;
            }
            return kOfxStatOK;
        }
        template <class T, class F>
        static OfxStatus array_set(OfxPropertySetHandle h, const char* n, int count, const T* values, F set_one) noexcept {
            for (int i = 0; i < count; ++i) {
                auto status = set_one(h, n, i, values[i]);
                if (!accepted(status))
                    return status;
            }
            return kOfxStatOK;
        }
#define ARRAY_PROP(NAME, TYPE)                                                                                       \
    static OfxStatus get_##NAME##_n(OfxPropertySetHandle h, const char* n, int count, TYPE* values) noexcept {       \
        return array_get(h, n, count, values, get_##NAME);                                                           \
    }                                                                                                                \
    static OfxStatus set_##NAME##_n(OfxPropertySetHandle h, const char* n, int count, const TYPE* values) noexcept { \
        return array_set(h, n, count, values, set_##NAME);                                                           \
    }
        ARRAY_PROP(int, int)
        ARRAY_PROP(double, double)
#undef ARRAY_PROP
        static OfxStatus reset(OfxPropertySetHandle h, const char* n) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& b = bag(h);
            return guarded(*b.owner, [&] {
                auto status = b.owner->record({"reset", b.name, n, 0, {}, {}}, b.scope);
                if (accepted(status))
                    b.properties.erase(n);
                return status;
            });
        }
        static OfxStatus dimension(OfxPropertySetHandle h, const char* n, int* out) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& b = bag(h);
            return guarded(*b.owner, [&] {
                auto status = b.owner->record({"dimension", b.name, n, 0, {}, {}}, b.scope);
                if (!accepted(status))
                    return status;
                auto it = b.properties.find(n);
                if (it == b.properties.end()) {
                    if (std::strcmp(n, kOfxParamPropChoiceOption) == 0 || std::strcmp(n, kOfxParamPropChoiceEnum) == 0) {
                        *out = 0;
                        return kOfxStatOK;
                    }
                    return kOfxStatErrUnknown;
                }
                *out = static_cast<int>(it->second.size());
                return status;
            });
        }
        static OfxStatus effect_props(OfxImageEffectHandle h, OfxPropertySetHandle* out) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& e = effect(h);
            return guarded(*e.bag.owner, [&] {
                auto status = e.bag.owner->record({"effect_properties", e.bag.name, {}, 0, {}, {}});
                const bool null = e.bag.owner->take_null("effect_properties");
                *out = null ? nullptr : reinterpret_cast<OfxPropertySetHandle>(&e.bag);
                return status;
            });
        }
        static OfxStatus effect_params(OfxImageEffectHandle h, OfxParamSetHandle* out) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& e = effect(h);
            return guarded(*e.bag.owner, [&] {
                auto status = e.bag.owner->record({"param_set", e.bag.name, {}, 0, {}, {}});
                const bool null = e.bag.owner->take_null("param_set");
                *out = null ? nullptr : reinterpret_cast<OfxParamSetHandle>(&e.param_set);
                return status;
            });
        }
        Effect& effect_for_set(Bag& set) {
            if (&descriptor.param_set == &set)
                return descriptor;
            for (auto& e : effects)
                if (&e->param_set == &set)
                    return *e;
            throw std::runtime_error("unknown parameter set");
        }
        static OfxStatus clip_define(OfxImageEffectHandle h, const char* n, OfxPropertySetHandle* out) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& e = effect(h);
            return guarded(*e.bag.owner, [&] {
                auto status = e.bag.owner->record({"clip_define", e.bag.name, n, 0, {}, {}});
                if (!accepted(status))
                    return status;
                auto c = std::make_unique<Clip>(e.bag.owner, n);
                c->bag.properties[kOfxPropType] = {std::string(kOfxTypeClip)};
                c->bag.properties[kOfxImageClipPropOptional] = {1};
                c->bag.properties[kOfxImageEffectPropSupportedComponents] = {};
                auto* ptr = c.get();
                e.clips.emplace(n, std::move(c));
                *out = reinterpret_cast<OfxPropertySetHandle>(&ptr->bag);
                return status;
            });
        }
        static OfxStatus param_define(OfxParamSetHandle h, const char* t, const char* n, OfxPropertySetHandle* out) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& b = bag(h);
            return guarded(*b.owner, [&] {
                auto status = b.owner->record({"param_define", b.name, n, 0, {std::string(t)}, {}});
                if (!accepted(status))
                    return status;
                auto& e = b.owner->effect_for_set(b);
                auto p = std::make_unique<Param>(b.owner, n, InstanceParamKind::Double);
                p->bag.properties[kOfxPropType] = {std::string(kOfxTypeParameter)};
                p->bag.scope = b.scope;
                p->bag.properties[kOfxParamPropType] = {std::string(t)};
                p->bag.properties[kOfxParamPropEnabled] = {1};
                p->bag.properties[kOfxParamPropSecret] = {0};
                p->bag.properties[kOfxParamPropAnimates] = {1};
                p->bag.properties[kOfxParamPropChoiceOption] = {};
                p->bag.properties[kOfxParamPropChoiceEnum] = {};
                p->bag.properties[kOfxParamPropPageChild] = {};
                auto* ptr = p.get();
                e.params.emplace(n, std::move(p));
                *out = reinterpret_cast<OfxPropertySetHandle>(&ptr->bag);
                return status;
            });
        }
        static OfxStatus param_set_props(OfxParamSetHandle h, OfxPropertySetHandle* out) noexcept {
            if (!h || !out)
                return kOfxStatErrBadHandle;
            auto& b = bag(h);
            return guarded(*b.owner, [&] {
                auto status = b.owner->record({"param_set_properties", b.name, {}, 0, {}, {}}, b.scope);
                if (!accepted(status))
                    return status;
                *out = b.owner->take_null("param_set_properties") ? nullptr : reinterpret_cast<OfxPropertySetHandle>(h);
                return status;
            });
        }
        static OfxStatus clip_handle(OfxImageEffectHandle h, const char* n, OfxImageClipHandle* out, OfxPropertySetHandle* props) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& e = effect(h);
            return guarded(*e.bag.owner, [&] {
                auto status = e.bag.owner->record({"clip_handle", e.bag.name, n, 0, {}, {}});
                if (!accepted(status))
                    return status;
                auto it = e.clips.find(n);
                if (it == e.clips.end())
                    return kOfxStatErrUnknown;
                const bool null = e.bag.owner->take_null("clip_handle/" + std::string(n));
                *out = null ? nullptr : reinterpret_cast<OfxImageClipHandle>(it->second.get());
                if (props)
                    *props = null ? nullptr : reinterpret_cast<OfxPropertySetHandle>(&it->second->bag);
                return status;
            });
        }
        static OfxStatus param_handle(OfxParamSetHandle h, const char* n, OfxParamHandle* out, OfxPropertySetHandle* props) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& b = bag(h);
            return guarded(*b.owner, [&] {
                auto status = b.owner->record({"param_handle", b.name, n, 0, {}, {}});
                if (!accepted(status))
                    return status;
                auto& e = b.owner->effect_for_set(b);
                auto it = e.params.find(n);
                if (it == e.params.end())
                    return kOfxStatErrUnknown;
                const bool null = b.owner->take_null("param_handle/" + std::string(n));
                *out = null ? nullptr : reinterpret_cast<OfxParamHandle>(it->second.get());
                if (props)
                    *props = null ? nullptr : reinterpret_cast<OfxPropertySetHandle>(&it->second->bag);
                return status;
            });
        }
        static OfxStatus param_props(OfxParamHandle h, OfxPropertySetHandle* out) noexcept {
            if (!h || !out)
                return kOfxStatErrBadHandle;
            auto& p = param(h);
            return guarded(*p.bag.owner, [&] {
                auto status = p.bag.owner->record({"param_properties", p.bag.name, {}, 0, {}, {}}, p.bag.scope);
                if (!accepted(status))
                    return status;
                const bool null = p.bag.owner->take_null("param_properties/" + p.bag.name);
                *out = null ? nullptr : reinterpret_cast<OfxPropertySetHandle>(&p.presentation);
                return status;
            });
        }
        static OfxStatus clip_props(OfxImageClipHandle h, OfxPropertySetHandle* out) noexcept {
            if (!h || !out)
                return kOfxStatErrBadHandle;
            *out = reinterpret_cast<OfxPropertySetHandle>(&clip(h).bag);
            return kOfxStatOK;
        }
        static OfxStatus clip_get_image(OfxImageClipHandle raw, OfxTime time, const OfxRectD* region, OfxPropertySetHandle* out) noexcept {
            if (!raw || !out)
                return kOfxStatErrBadHandle;
            auto& c = clip(raw);
            return guarded(*c.bag.owner, [&] {
                std::vector<InstanceValue> values{time};
                if (region)
                    values.insert(values.end(), {region->x1, region->y1, region->x2, region->y2});
                const auto injected = c.bag.owner->record({"clip_get_image", c.bag.name, {}, 0, values, c.bag.scope});
                if (!accepted(injected)) {
                    *out = nullptr;
                    return injected;
                }
                if (!accepted(c.image_status) || !c.image_available) {
                    *out = nullptr;
                    c.bag.owner->trace.back().status = !c.image_available && accepted(c.image_status)
                                                           ? kOfxStatFailed
                                                           : c.image_status;
                    return c.bag.owner->trace.back().status;
                }
                auto image = std::make_unique<Clip::Image>(c.bag.owner, c.bag.name + ".image." + std::to_string(c.images.size()));
                image->clip_name = c.bag.name;
                image->bag.scope = c.bag.scope;
                image->bag.properties = c.image_properties;
                auto* ptr = image.get();
                c.images.push_back(std::move(image));
                *out = reinterpret_cast<OfxPropertySetHandle>(&ptr->bag);
                c.bag.owner->trace.back().status = c.image_status;
                return c.image_status;
            });
        }
        static OfxStatus clip_release_image(OfxPropertySetHandle raw) noexcept {
            if (!raw)
                return kOfxStatErrBadHandle;
            auto& b = bag(raw);
            return guarded(*b.owner, [&] {
                const auto injected = b.owner->record({"clip_release_image", b.name, {}, 0, {}, b.scope});
                auto separator = b.name.find(".image.");
                if (separator != std::string::npos) {
                    const auto clip_name = b.name.substr(0, separator);
                    for (auto& e : b.owner->effects) {
                        if (e->bag.name != b.scope)
                            continue;
                        auto found = e->clips.find(clip_name);
                        if (found != e->clips.end()) {
                            ++found->second->releases;
                            if (injected == kOfxStatOK && found->second->release_status != kOfxStatOK) {
                                b.owner->trace.back().status = found->second->release_status;
                                return found->second->release_status;
                            }
                            break;
                        }
                    }
                }
                return injected;
            });
        }
        static OfxStatus clip_region(OfxImageClipHandle raw, OfxTime time, OfxRectD* out) noexcept {
            if (!raw || !out)
                return kOfxStatErrBadHandle;
            auto& c = clip(raw);
            return guarded(*c.bag.owner, [&] {
                auto status = c.bag.owner->record({"clip_get_region", c.bag.name, {}, 0, {time}, c.bag.scope});
                if (!accepted(status))
                    return status;
                *out = c.region;
                c.bag.owner->trace.back().values.insert(c.bag.owner->trace.back().values.end(), {out->x1, out->y1, out->x2, out->y2});
                return status;
            });
        }
        static int abort(OfxImageEffectHandle raw) noexcept {
            if (!raw)
                return 1;
            auto& e = effect(raw);
            try {
                const auto status = e.bag.owner->record({"abort", e.bag.name, {}, 0, {e.bag.owner->abort_requested ? 1 : 0}, {}});
                return accepted(status) ? static_cast<int>(e.bag.owner->abort_requested) : 1;
            } catch (...) {
                e.bag.owner->error = true;
                return 1;
            }
        }
        static OfxStatus read_value(Param& p, va_list args, const char* operation) {
            auto status = p.bag.owner->record({operation, p.bag.name, {}, 0, p.values, {}}, p.bag.scope);
            if (!accepted(status))
                return status;
            switch (p.kind) {
                case InstanceParamKind::Double:
                case InstanceParamKind::Double2D:
                case InstanceParamKind::Double3D:
                    for (auto& v : p.values)
                        *va_arg(args, double*) = std::get<double>(v);
                    break;
                case InstanceParamKind::StringChoice: {
                    auto& value = std::get<std::string>(p.values.at(0));
                    auto& owner = *p.bag.owner;
                    if (owner.reuse_string_storage) {
                        if (value.size() >= owner.string_storage.size())
                            return kOfxStatErrValue;
                        if (owner.string_storage_used)
                            ++owner.string_reuses;
                        owner.string_storage.fill('\0');
                        std::memcpy(owner.string_storage.data(), value.data(), value.size());
                        owner.string_storage_used = true;
                        *va_arg(args, char**) = owner.string_storage.data();
                    } else {
                        *va_arg(args, char**) = value.data();
                    }
                    break;
                }
                case InstanceParamKind::Boolean:
                case InstanceParamKind::Choice:
                case InstanceParamKind::Integer:
                    *va_arg(args, int*) = std::get<int>(p.values.at(0));
                    break;
                case InstanceParamKind::PushButton:
                    return kOfxStatErrUnsupported;
            }
            return status;
        }
        static OfxStatus write_value(Param& p, va_list args, const char* operation) {
            std::vector<InstanceValue> values;
            switch (p.kind) {
                case InstanceParamKind::Double:
                    values.emplace_back(va_arg(args, double));
                    break;
                case InstanceParamKind::Double2D:
                    values.emplace_back(va_arg(args, double));
                    values.emplace_back(va_arg(args, double));
                    break;
                case InstanceParamKind::Double3D:
                    values.emplace_back(va_arg(args, double));
                    values.emplace_back(va_arg(args, double));
                    values.emplace_back(va_arg(args, double));
                    break;
                case InstanceParamKind::StringChoice: {
                    const auto* s = va_arg(args, const char*);
                    values.emplace_back(std::string(s ? s : ""));
                    break;
                }
                case InstanceParamKind::Boolean:
                case InstanceParamKind::Choice:
                case InstanceParamKind::Integer:
                    values.emplace_back(va_arg(args, int));
                    break;
                case InstanceParamKind::PushButton:
                    return kOfxStatErrUnsupported;
            }
            auto status = p.bag.owner->record({operation, p.bag.name, {}, 0, values, {}}, p.bag.scope);
            if (accepted(status) || p.bag.owner->current_store_failure)
                p.values = std::move(values);
            return status;
        }
        static OfxStatus param_get(OfxParamHandle h, ...) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& p = param(h);
            va_list args;
            va_start(args, h);
            auto status = guarded(*p.bag.owner, [&] {
                return read_value(p, args, "param_get");
            });
            va_end(args);
            return status;
        }
        static OfxStatus param_get_time(OfxParamHandle h, OfxTime time, ...) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& p = param(h);
            va_list args;
            va_start(args, time);
            auto status = guarded(*p.bag.owner, [&] {
                return read_value(p, args, "param_get_time");
            });
            va_end(args);
            return status;
        }
        static OfxStatus param_set(OfxParamHandle h, ...) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& p = param(h);
            va_list args;
            va_start(args, h);
            auto status = guarded(*p.bag.owner, [&] {
                return write_value(p, args, "param_set");
            });
            va_end(args);
            return status;
        }
        static OfxStatus param_set_time(OfxParamHandle h, OfxTime time, ...) noexcept {
            if (!h)
                return kOfxStatErrBadHandle;
            auto& p = param(h);
            va_list args;
            va_start(args, time);
            auto status = guarded(*p.bag.owner, [&] {
                return write_value(p, args, "param_set_time");
            });
            va_end(args);
            return status;
        }
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): OFX mandates this variadic callback signature.
        static OfxStatus host_message(void* raw, const char* message_type, const char* message_id, const char* message_format, ...) noexcept {
            if (!raw || !message_format)
                return kOfxStatErrBadHandle;
            auto& e = effect(raw);
            va_list args;
            va_start(args, message_format);
            OfxStatus status = guarded(*e.bag.owner, [&] {
                va_list copy;
                va_copy(copy, args);
                const int size = std::vsnprintf(nullptr, 0, message_format, copy);
                va_end(copy);
                if (size < 0)
                    return kOfxStatFailed;
                std::vector<char> text(static_cast<std::size_t>(size) + 1);
                std::vsnprintf(text.data(), text.size(), message_format, args);
                return e.bag.owner->record({"message", e.bag.name, message_type ? message_type : "", 0, {std::string(message_id ? message_id : ""), std::string(text.data())}, {}});
            });
            va_end(args);
            return status;
        }
        Impl() {
            message.message = host_message;
            host.host = reinterpret_cast<OfxPropertySetHandle>(&host_bag);
            host.fetchSuite = fetch;
            property.propSetInt = set_int;
            property.propSetDouble = set_double;
            property.propSetString = set_string;
            property.propSetPointer = set_pointer;
            property.propSetIntN = set_int_n;
            property.propSetDoubleN = set_double_n;
            property.propGetIntN = get_int_n;
            property.propGetDoubleN = get_double_n;
            property.propReset = reset;
            property.propGetInt = get_int;
            property.propGetDouble = get_double;
            property.propGetString = get_string;
            property.propGetPointer = get_pointer;
            property.propGetDimension = dimension;
            effect_suite.getPropertySet = effect_props;
            effect_suite.getParamSet = effect_params;
            effect_suite.clipDefine = clip_define;
            effect_suite.clipGetHandle = clip_handle;
            effect_suite.clipGetPropertySet = clip_props;
            effect_suite.clipGetImage = clip_get_image;
            effect_suite.clipReleaseImage = clip_release_image;
            effect_suite.clipGetRegionOfDefinition = clip_region;
            effect_suite.abort = abort;
            param_suite.paramDefine = param_define;
            param_suite.paramSetGetPropertySet = param_set_props;
            param_suite.paramGetHandle = param_handle;
            param_suite.paramGetPropertySet = param_props;
            param_suite.paramGetValue = param_get;
            param_suite.paramGetValueAtTime = param_get_time;
            param_suite.paramSetValue = param_set;
            param_suite.paramSetValueAtTime = param_set_time;
            auto& h = host_bag.properties;
            h[kOfxPropType] = {std::string(kOfxTypeImageEffectHost)};
            h[kOfxPropName] = {std::string("FilmJuicerInstanceHost")};
            h[kOfxPropLabel] = {std::string("FilmJuicerInstanceHost")};
            h[kOfxPropVersionLabel] = {std::string("1")};
            h[kOfxPropAPIVersion] = {1, 4};
            h[kOfxPropVersion] = {1, 0, 0};
            h[kOfxImageEffectPropSupportedContexts] = {std::string(kOfxImageEffectContextFilter)};
            h[kOfxImageEffectPropSupportedComponents] = {std::string(kOfxImageComponentRGBA)};
            h[kOfxImageEffectPropSupportedPixelDepths] = {std::string(kOfxBitDepthFloat)};
            h[kOfxImageEffectPropCudaRenderSupported] = {std::string("true")};
            h[kOfxImageEffectPropCudaStreamSupported] = {std::string("true")};
            h[kOfxImageEffectHostPropNativeOrigin] = {std::string(kOfxHostNativeOriginBottomLeft)};
            for (const char* n : {kOfxImageEffectPropOpenCLRenderSupported, kOfxImageEffectPropMetalRenderSupported, kOfxImageEffectPropOpenGLRenderSupported})
                h[n] = {std::string("false")};
            for (const char* n : {kOfxImageEffectHostPropIsBackground, kOfxImageEffectPropSupportsOverlays, kOfxImageEffectPropSupportsTiles, kOfxImageEffectPropTemporalClipAccess, kOfxImageEffectPropSupportsMultipleClipDepths, kOfxImageEffectPropSupportsMultipleClipPARs, kOfxImageEffectPropSetableFrameRate, kOfxImageEffectPropSetableFielding, kOfxParamHostPropSupportsCustomInteract, kOfxParamHostPropSupportsCustomAnimation, kOfxParamHostPropMaxPages})
                h[n] = {0};
            for (const char* n : {kOfxImageEffectPropSupportsMultiResolution, kOfxParamHostPropSupportsStringAnimation, kOfxParamHostPropSupportsChoiceAnimation, kOfxParamHostPropSupportsStrChoiceAnimation, kOfxParamHostPropSupportsBooleanAnimation})
                h[n] = {1};
            h[kOfxParamHostPropSupportsParametricAnimation] = {0};
            h[kOfxImageEffectPropRenderQualityDraft] = {0};
            h[kOfxImageEffectInstancePropSequentialRender] = {0};
            h[kOfxPropHostOSHandle] = {static_cast<void*>(nullptr)};
            h[kOfxParamHostPropMaxParameters] = {1024};
            h[kOfxParamHostPropPageRowColumnCount] = {0, 0};
            context.properties[kOfxImageEffectPropContext] = {std::string(kOfxImageEffectContextFilter)};
            descriptor.bag.properties[kOfxPropType] = {std::string(kOfxTypeImageEffect)};
            descriptor.bag.properties[kOfxPluginPropFilePath] = {std::string("FilmJuicerInstanceHost")};
            descriptor.bag.properties[kOfxImageEffectPropSupportedContexts] = {};
            descriptor.bag.properties[kOfxImageEffectPropSupportedPixelDepths] = {};
            descriptor.bag.properties[kOfxImageEffectPropClipPreferencesSlaveParam] = {};
            descriptor.param_set.properties[kOfxPluginPropParamPageOrder] = {};
        }
        Effect& add_effect() {
            const std::string effect_name = "effect." + std::to_string(effects.size());
            auto e = std::make_unique<Effect>(this, effect_name);
            e->bag.properties[kOfxPropType] = {std::string(kOfxTypeImageEffectInstance)};
            e->bag.properties[kOfxImageEffectPropContext] = {std::string(kOfxImageEffectContextFilter)};
            e->bag.properties[kOfxPropInstanceData] = {static_cast<void*>(nullptr)};
            for (auto entry : kInstanceParameters) {
                auto p = std::make_unique<Param>(this, std::string(entry.name), entry.kind);
                p->bag.scope = effect_name;
                p->bag.properties[kOfxParamPropType] = {std::string(type(entry.kind))};
                p->bag.properties[kOfxParamPropEnabled] = {1};
                p->bag.properties[kOfxParamPropSecret] = {0};
                auto found = descriptor.params.find(std::string(entry.name));
                if (found != descriptor.params.end()) {
                    p->bag.properties = found->second->bag.properties;
                    auto d = found->second->bag.properties.find(kOfxParamPropDefault);
                    if (d != found->second->bag.properties.end())
                        p->values = d->second;
                }
                p->presentation.properties = p->bag.properties;
                p->presentation.scope = effect_name;
                if (p->values.empty()) {
                    switch (entry.kind) {
                        case InstanceParamKind::Double:
                            p->values = {0.0};
                            break;
                        case InstanceParamKind::Double2D:
                            p->values = {0.0, 0.0};
                            break;
                        case InstanceParamKind::Double3D:
                            p->values = {0.0, 0.0, 0.0};
                            break;
                        case InstanceParamKind::StringChoice:
                            p->values = {std::string()};
                            break;
                        case InstanceParamKind::PushButton:
                            break;
                        default:
                            p->values = {0};
                            break;
                    }
                }
                e->params.emplace(std::string(entry.name), std::move(p));
            }
            for (auto name : kInstanceClips) {
                auto c = std::make_unique<Clip>(this, std::string(name));
                c->bag.scope = effect_name;
                auto found = descriptor.clips.find(std::string(name));
                if (found != descriptor.clips.end())
                    c->bag.properties = found->second->bag.properties;
                c->bag.properties[kOfxPropType] = {std::string(kOfxTypeClip)};
                c->bag.properties[kOfxPropName] = {std::string(name)};
                if (!c->bag.properties.contains(kOfxImageEffectPropPixelDepth))
                    c->bag.properties[kOfxImageEffectPropPixelDepth] = {std::string(kOfxBitDepthFloat)};
                if (!c->bag.properties.contains(kOfxImageEffectPropSupportedComponents))
                    c->bag.properties[kOfxImageEffectPropSupportedComponents] = {std::string(kOfxImageComponentRGBA)};
                if (!c->bag.properties.contains(kOfxImageEffectPropFrameRate))
                    c->bag.properties[kOfxImageEffectPropFrameRate] = {24.0};
                e->clips.emplace(std::string(name), std::move(c));
            }
            auto* ptr = e.get();
            effects.push_back(std::move(e));
            return *ptr;
        }
    };
    InstanceHost::InstanceHost()
        : impl_(std::make_unique<Impl>()) {}
    InstanceHost::~InstanceHost() = default;
    OfxHost* InstanceHost::binding() noexcept {
        return &impl_->host;
    }
    OfxImageEffectHandle InstanceHost::descriptor_handle() noexcept {
        return reinterpret_cast<OfxImageEffectHandle>(&impl_->descriptor);
    }
    OfxPropertySetHandle InstanceHost::context_args() noexcept {
        return reinterpret_cast<OfxPropertySetHandle>(&impl_->context);
    }
    OfxPropertySetHandle InstanceHost::action_args() noexcept {
        return reinterpret_cast<OfxPropertySetHandle>(&impl_->action);
    }
    OfxPropertySetHandle InstanceHost::output_args() noexcept {
        return reinterpret_cast<OfxPropertySetHandle>(&impl_->output);
    }
    OfxImageEffectHandle InstanceHost::create_effect() {
        return reinterpret_cast<OfxImageEffectHandle>(&impl_->add_effect());
    }
    OfxPropertySetHandle InstanceHost::change_args(const InstanceChange& change, double time) {
        auto args = std::make_unique<Impl::Bag>(impl_.get(), "change");
        auto& p = args->properties;
        p[kOfxPropChangeReason] = {std::string(change.reason)};
        p[kOfxPropType] = {std::string(change.type)};
        p[kOfxPropName] = {std::string(change.name)};
        p[kOfxPropTime] = {time};
        p[kOfxImageEffectPropRenderScale] = {1.0, 1.0};
        auto* result = args.get();
        impl_->changes.push_back(std::move(args));
        return reinterpret_cast<OfxPropertySetHandle>(result);
    }
    void InstanceHost::set_property(OfxPropertySetHandle handle, std::string_view key, std::vector<InstanceValue> values) {
        if (!handle)
            throw std::invalid_argument("property handle is null");
        auto& properties = Impl::bag(handle).properties;
        if (values.empty()) {
            properties.erase(std::string(key));
        } else {
            properties[std::string(key)] = std::move(values);
        }
    }
    std::vector<InstanceValue> InstanceHost::property(OfxPropertySetHandle handle, std::string_view key) const {
        if (!handle)
            throw std::invalid_argument("property handle is null");
        const auto& properties = Impl::bag(handle).properties;
        auto found = properties.find(std::string(key));
        return found == properties.end() ? std::vector<InstanceValue>{} : found->second;
    }
    void InstanceHost::set_value(OfxImageEffectHandle effect, std::string_view name, std::vector<InstanceValue> values) {
        Impl::effect(effect).params.at(std::string(name))->values = std::move(values);
    }
    std::vector<InstanceValue> InstanceHost::value(OfxImageEffectHandle effect, std::string_view name) const {
        return Impl::effect(effect).params.at(std::string(name))->values;
    }
    void InstanceHost::set_effect_property(OfxImageEffectHandle effect, std::string_view key, std::vector<InstanceValue> values) {
        Impl::effect(effect).bag.properties[std::string(key)] = std::move(values);
    }
    void InstanceHost::set_clip_property(OfxImageEffectHandle effect, std::string_view clip_name, const char* key, std::vector<InstanceValue> values) {
        auto& clip = *Impl::effect(effect).clips.at(std::string(clip_name));
        clip.bag.properties[std::string(key)] = std::move(values);
    }
    void InstanceHost::set_image_property(OfxImageEffectHandle effect, std::string_view clip_name, const char* key, std::vector<InstanceValue> values) {
        auto& clip = *Impl::effect(effect).clips.at(std::string(clip_name));
        clip.image_properties[std::string(key)] = std::move(values);
    }
    void InstanceHost::set_clip_region(OfxImageEffectHandle effect, std::string_view clip_name, OfxRectD bounds) {
        Impl::effect(effect).clips.at(std::string(clip_name))->region = bounds;
    }
    void InstanceHost::set_image_available(OfxImageEffectHandle effect, std::string_view clip_name, bool available) {
        Impl::effect(effect).clips.at(std::string(clip_name))->image_available = available;
    }
    void InstanceHost::set_image_get_status(OfxImageEffectHandle effect, std::string_view clip_name, OfxStatus status) {
        Impl::effect(effect).clips.at(std::string(clip_name))->image_status = status;
    }
    void InstanceHost::set_image_release_status(OfxImageEffectHandle effect, std::string_view clip_name, OfxStatus status) {
        Impl::effect(effect).clips.at(std::string(clip_name))->release_status = status;
    }
    std::size_t InstanceHost::image_release_count(OfxImageEffectHandle effect, std::string_view clip_name) const {
        return Impl::effect(effect).clips.at(std::string(clip_name))->releases;
    }
    void InstanceHost::set_abort(bool abort) noexcept {
        impl_->abort_requested = abort;
    }
    void InstanceHost::clear_calls() {
        impl_->trace.clear();
    }
    std::vector<InstanceCall>& InstanceHost::calls() noexcept {
        return impl_->trace;
    }
    bool InstanceHost::callback_error() const noexcept {
        return impl_->error;
    }
    void InstanceHost::fail_next(std::string operation, OfxStatus status) {
        impl_->fail_operation = std::move(operation);
        impl_->fail_remaining = 1;
        impl_->fail_status = status;
        impl_->fail_after_store = false;
    }
    void InstanceHost::store_then_fail_next(std::string operation, OfxStatus status) {
        impl_->fail_operation = std::move(operation);
        impl_->fail_status = status;
        impl_->fail_remaining = 1;
        impl_->fail_after_store = true;
    }
    void InstanceHost::null_next(std::string operation) {
        null_on_nth(std::move(operation), 1);
    }
    void InstanceHost::null_on_nth(std::string operation, int occurrence) {
        impl_->null_operation = std::move(operation);
        impl_->null_remaining = occurrence;
    }
    void InstanceHost::fail_on_nth(int occurrence, std::string operation, OfxStatus status) {
        impl_->fail_operation = std::move(operation);
        impl_->fail_remaining = occurrence;
        impl_->fail_status = status;
        impl_->fail_after_store = false;
    }
    void InstanceHost::set_parameter_type(OfxImageEffectHandle effect, std::string_view name, std::string type) {
        Impl::effect(effect).params.at(std::string(name))->bag.properties[kOfxParamPropType] = {std::move(type)};
    }
    void* InstanceHost::instance_data(OfxImageEffectHandle effect) const {
        return std::get<void*>(Impl::effect(effect).bag.properties.at(kOfxPropInstanceData).at(0));
    }
    void InstanceHost::set_parameter_presentation(OfxImageEffectHandle effect, std::string_view name, std::string_view property, std::vector<InstanceValue> values) {
        Impl::effect(effect).params.at(std::string(name))->presentation.properties[std::string(property)] = std::move(values);
    }
    void InstanceHost::reuse_string_getter_storage(bool enabled) noexcept {
        impl_->reuse_string_storage = enabled;
        impl_->string_storage_used = false;
        impl_->string_reuses = 0;
    }
    std::size_t InstanceHost::string_getter_reuses() const noexcept {
        return impl_->string_reuses;
    }
    void InstanceHost::return_string_on_failure_next(std::string operation, OfxStatus status) {
        impl_->fail_operation = std::move(operation);
        impl_->fail_status = status;
        impl_->fail_remaining = 1;
        impl_->return_failed_string = true;
    }
    void InstanceHost::set_effect_context(OfxImageEffectHandle effect, std::string context) {
        Impl::effect(effect).bag.properties[kOfxImageEffectPropContext] = {std::move(context)};
    }
    std::string InstanceHost::effect_scope(OfxImageEffectHandle effect) const {
        return Impl::effect(effect).bag.name;
    }
    void InstanceHost::reenter_next(std::string operation, std::function<void()> callback) {
        impl_->reentry_operation = std::move(operation);
        impl_->reentry = std::move(callback);
    }
} // namespace ofx_test
