#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>

#include "ofx_abi_test.h"

#define FJ_OFX_TYPE(type, size, alignment)                \
    _Static_assert(sizeof(type) == size, "size: " #type); \
    _Static_assert(_Alignof(type) == alignment, "alignment: " #type);
#define FJ_OFX_FIELD(type, field, offset, expected)                               \
    _Static_assert(offsetof(type, field) == offset, "offset: " #type "." #field); \
    _Static_assert(_Generic(((type*)0)->field, expected: 1, default: 0), "type: " #type "." #field);
#define FJ_OFX_VALUE(name, expected) _Static_assert(name == expected, "value: " #name);
#include "ofx_abi_facts.inc"
#undef FJ_OFX_VALUE
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD
_Static_assert(_Generic((OfxStatus)0, int: 1, default: 0), "signed status");
_Static_assert(_Generic((OfxTime)0, double: 1, default: 0), "time");
_Static_assert(_Generic((OfxPropertySetHandle)0, struct OfxPropertySetStruct*: 1, default: 0), "property handle");
_Static_assert(_Generic((OfxParamHandle)0, struct OfxParamStruct*: 1, default: 0), "parameter handle");
_Static_assert(_Generic((OfxParamSetHandle)0, struct OfxParamSetStruct*: 1, default: 0), "parameter set handle");
_Static_assert(_Generic((OfxImageEffectHandle)0, struct OfxImageEffectStruct*: 1, default: 0), "effect handle");
_Static_assert(_Generic((OfxImageClipHandle)0, struct OfxImageClipStruct*: 1, default: 0), "clip handle");
_Static_assert(_Generic((OfxImageMemoryHandle)0, struct OfxImageMemoryStruct*: 1, default: 0), "memory handle");

const size_t* fj_test_ofx_abi_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_OFX_TYPE(type, size, alignment) sizeof(type), _Alignof(type),
#define FJ_OFX_FIELD(type, field, offset, expected) offsetof(type, field),
#include "ofx_abi_facts.inc"
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}

static const void* fetch_suite(OfxPropertySetHandle host, const char* name, int version) {
    return (uintptr_t)host == 1 && strcmp(name, "fixture.suite") == 0 && version == 7 ? (const void*)(uintptr_t)2 : NULL;
}
static void set_host(OfxHost* host) {
    host->host = (OfxPropertySetHandle)(uintptr_t)6;
}
static OfxStatus main_entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    return strcmp(action, "fixture.action") == 0 && (uintptr_t)handle == 3 && (uintptr_t)in == 4 && (uintptr_t)out == 5 ? 17 : 1;
}
const OfxHost* fj_test_ofx_host(void) {
    static const OfxHost host = {(OfxPropertySetHandle)(uintptr_t)1, fetch_suite};
    return &host;
}
const OfxPlugin* fj_test_ofx_plugin(void) {
    static const OfxPlugin plugin = {"fixture", 1, "fixture", 1, 0, set_host, main_entry};
    return &plugin;
}
int fj_test_ofx_call_rust(const OfxHost* host, const OfxPlugin* plugin) {
    if (!host || !plugin || !host->fetchSuite || !plugin->setHost || !plugin->mainEntry ||
        (uintptr_t)host->host != 1 || (uintptr_t)host->fetchSuite(host->host, "fixture.suite", 7) != 2)
        return 1;
    OfxHost copy = *host;
    plugin->setHost(&copy);
    if ((uintptr_t)copy.host != 6)
        return 1;
    return plugin->mainEntry("fixture.action", (void*)(uintptr_t)3, (OfxPropertySetHandle)(uintptr_t)4, (OfxPropertySetHandle)(uintptr_t)5) == 17 ? 0 : 1;
}

static int parameter_values(OfxParamHandle handle, va_list args, int output) {
    uintptr_t shape = (uintptr_t)handle;
    if (shape == 1 || shape == 6 || shape == 7) {
        int value = shape == 1 ? 23 : (shape == 6 ? 1 : 2);
        if (output)
            *va_arg(args, int*) = value;
        else if (va_arg(args, int) != value)
            return 1;
    } else if (shape == 2) {
        if (output)
            *va_arg(args, char**) = (char*)"fixture-value";
        else if (strcmp(va_arg(args, const char*), "fixture-value") != 0)
            return 1;
    } else if (shape >= 3 && shape <= 5) {
        const double values[] = {1.25, 2.5, 3.75};
        size_t count = shape - 2;
        for (size_t i = 0; i < count; ++i) {
            if (output)
                *va_arg(args, double*) = values[i];
            else if (va_arg(args, double) != values[i])
                return 1;
        }
    } else
        return 1;
    return 0;
}
static OfxStatus get_value(OfxParamHandle handle, ...) {
    va_list args;
    va_start(args, handle);
    int result = parameter_values(handle, args, 1);
    va_end(args);
    return result;
}
static OfxStatus get_value_at_time(OfxParamHandle handle, OfxTime time, ...) {
    va_list args;
    va_start(args, time);
    int result = time == 12.5 ? parameter_values(handle, args, 1) : 1;
    va_end(args);
    return result;
}
static OfxStatus set_value(OfxParamHandle handle, ...) {
    va_list args;
    va_start(args, handle);
    int result = parameter_values(handle, args, 0);
    va_end(args);
    return result;
}
static OfxStatus set_value_at_time(OfxParamHandle handle, OfxTime time, ...) {
    va_list args;
    va_start(args, time);
    int result = time == 12.5 ? parameter_values(handle, args, 0) : 1;
    va_end(args);
    return result;
}
const OfxParameterSuiteV1* fj_test_ofx_parameter_suite(void) {
    static const OfxParameterSuiteV1 suite = {.paramGetValue = get_value, .paramGetValueAtTime = get_value_at_time, .paramSetValue = set_value, .paramSetValueAtTime = set_value_at_time};
    return &suite;
}
static OfxStatus message(void* handle, const char* type, const char* id, const char* format, ...) {
    va_list args;
    va_start(args, format);
    const char* text = va_arg(args, const char*);
    va_end(args);
    return (uintptr_t)handle == 7 && strcmp(type, "fixture.type") == 0 && strcmp(id, "fixture.id") == 0 && strcmp(format, "%s") == 0 && strcmp(text, "100% fixture") == 0 ? 17 : 1;
}
const OfxMessageSuiteV1* fj_test_ofx_message_suite(void) {
    static const OfxMessageSuiteV1 suite = {message};
    return &suite;
}

const size_t* fj_test_ofx_abi_c_values(size_t* count) {
    static const size_t values[] = {
#define FJ_OFX_TYPE(type, size, alignment)
#define FJ_OFX_FIELD(type, field, offset, expected)
#define FJ_OFX_VALUE(name, expected) name,
#include "ofx_abi_facts.inc"
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD
#undef FJ_OFX_VALUE
    };
    *count = sizeof(values) / sizeof(values[0]);
    return values;
}
const unsigned char* fj_test_ofx_abi_c_string(size_t index, size_t* count) {
    static const struct {
        const char* bytes;
        size_t size;
    } strings[] = {
#define FJ_OFX_TYPE(type, size, alignment)
#define FJ_OFX_FIELD(type, field, offset, expected)
#define FJ_OFX_STRING(name, expected) {name, sizeof(name)},
#include "ofx_abi_facts.inc"
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD
#undef FJ_OFX_STRING
    };
    if (index >= sizeof(strings) / sizeof(strings[0])) {
        *count = 0;
        return NULL;
    }
    *count = strings[index].size;
    return (const unsigned char*)strings[index].bytes;
}
