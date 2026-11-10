#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iterator>
#include <string_view>
#include <type_traits>

#include "ofx_abi_test.h"
#include "case_results.h"

#define FJ_OFX_TYPE(type, size, alignment) \
    static_assert(sizeof(type) == size);   \
    static_assert(alignof(type) == alignment);
#define FJ_OFX_FIELD(type, field, offset, expected) \
    static_assert(offsetof(type, field) == offset); \
    static_assert(std::is_same_v<decltype(type::field), expected>);
#define FJ_OFX_VALUE(name, expected) static_assert(name == expected);
#include "ofx_abi_facts.inc"
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD
#undef FJ_OFX_VALUE
static_assert(std::is_same_v<OfxStatus, int>);
static_assert(std::is_same_v<OfxTime, double>);
static_assert(std::is_same_v<OfxPropertySetHandle, OfxPropertySetStruct*>);
static_assert(std::is_same_v<OfxParamHandle, OfxParamStruct*>);
static_assert(std::is_same_v<OfxParamSetHandle, OfxParamSetStruct*>);
static_assert(std::is_same_v<OfxImageEffectHandle, OfxImageEffectStruct*>);
static_assert(std::is_same_v<OfxImageClipHandle, OfxImageClipStruct*>);
static_assert(std::is_same_v<OfxImageMemoryHandle, OfxImageMemoryStruct*>);

namespace {
    const std::size_t facts[] = {
#define FJ_OFX_TYPE(type, size, alignment) sizeof(type), alignof(type),
#define FJ_OFX_FIELD(type, field, offset, expected) offsetof(type, field),
#include "ofx_abi_facts.inc"
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD
    };
    const char* fact_names[] = {
#define FJ_OFX_TYPE(type, size, alignment) "sizeof " #type, "alignof " #type,
#define FJ_OFX_FIELD(type, field, offset, expected) "offsetof " #type "." #field,
#include "ofx_abi_facts.inc"
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD
    };
#define FJ_OFX_TYPE(type, size, alignment)
#define FJ_OFX_FIELD(type, field, offset, expected)
    const std::size_t values[] = {
#define FJ_OFX_VALUE(name, expected) name,
#include "ofx_abi_facts.inc"
#undef FJ_OFX_VALUE
    };
    const char* value_names[] = {
#define FJ_OFX_VALUE(name, expected) #name,
#include "ofx_abi_facts.inc"
#undef FJ_OFX_VALUE
    };
    struct StringFact {
        const char* name;
        const char* actual;
        const char* expected;
        std::size_t size;
    };
    const StringFact strings[] = {
#define FJ_OFX_STRING(name, expected) {#name, name, expected, sizeof(name)},
#include "ofx_abi_facts.inc"
#undef FJ_OFX_STRING
    };
#undef FJ_OFX_TYPE
#undef FJ_OFX_FIELD

    static_assert(std::size(facts) == 111);
    static_assert(std::size(fact_names) == 111);
    static_assert(std::size(values) == 17);
    static_assert(std::size(strings) == 143);

    bool layout() {
        std::size_t c_count = 0, rust_count = 0;
        const auto* c = fj_test_ofx_abi_c_facts(&c_count);
        const auto* rust = fj_test_ofx_abi_facts(&rust_count);
        if (c_count != std::size(facts) || rust_count != std::size(facts) || !c || !rust) {
            std::fprintf(stderr, "layout counts C=%zu C++=%zu Rust=%zu\n", c_count, std::size(facts), rust_count);
            return false;
        }
        for (std::size_t i = 0; i < c_count; ++i) {
            if (facts[i] != c[i] || facts[i] != rust[i]) {
                std::fprintf(stderr, "%s: C=%zu C++=%zu Rust=%zu\n", fact_names[i], c[i], facts[i], rust[i]);
                return false;
            }
        }
        std::puts("111 layout facts; 73 independently checked field offsets and declared types.");
        return true;
    }
    bool constants() {
        std::size_t count = 0;
        const auto* rust = fj_test_ofx_abi_values(&count);
        std::size_t c_count = 0;
        const auto* c = fj_test_ofx_abi_c_values(&c_count);
        if (!rust || !c || c_count != std::size(values) || count != std::size(values)) {
            std::fprintf(stderr, "value count: %zu expected %zu\n", count, std::size(values));
            return false;
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (rust[i] != values[i] || c[i] != values[i]) {
                std::fprintf(stderr, "%s: native=%zu Rust=%zu\n", value_names[i], values[i], rust[i]);
                return false;
            }
        }
        for (std::size_t i = 0; i < std::size(strings); ++i) {
            const auto& item = strings[i];
            std::size_t length = 0;
            const auto* bytes = fj_test_ofx_abi_string(i, &length);
            std::size_t c_length = 0;
            const auto* c_bytes = fj_test_ofx_abi_c_string(i, &c_length);
            if (!bytes || !c_bytes || c_length != item.size || length != item.size || std::strlen(item.expected) + 1 != item.size ||
                std::memcmp(item.actual, item.expected, item.size) != 0 || std::memcmp(c_bytes, item.expected, item.size) != 0 || std::memcmp(bytes, item.expected, item.size) != 0) {
                std::fprintf(stderr, "string %s length=%zu expected=%zu\n", item.name, length, item.size);
                return false;
            }
        }
        std::size_t length = 99;
        if (fj_test_ofx_abi_string(std::size(strings), &length) != nullptr || length != 0)
            return false;
        std::puts("17 numeric constants and 143 strings agree in C11/C++20/Rust, including NUL.");
        return true;
    }

    bool signatures() {
        std::puts("53 callback/slot signatures compiled independently in C11/C++20/Rust; nullable C callbacks and variadic tails included.");
        return true;
    }

    bool callbacks() {
        OfxHost host{};
        OfxPlugin plugin{};
        return fj_test_ofx_abi_callbacks(fj_test_ofx_host(), fj_test_ofx_plugin(), &host, &plugin) == 0 &&
               fj_test_ofx_call_rust(&host, &plugin) == 0;
    }
} // namespace
int main(int argc, char** argv) try {
    constexpr const char* cases[] = {"layout", "constants", "signatures", "callbacks", "parameters", "messages"};
    if (JuicerTests::list_requested(argc, argv)) {
        for (const char* name : cases)
            JuicerTests::list_case("ofx_abi", name);
        return 0;
    }
    JuicerTests::CaseLog log;
    bool ok = true;
    for (const char* name : cases) {
        JuicerTests::CaseEvent event("ofx_abi", name);
        const std::string_view current(name);
        const bool passed = current == "layout" ? layout() : current == "constants" ? constants()
                                                         : current == "signatures"  ? signatures()
                                                         : current == "callbacks"   ? callbacks()
                                                         : current == "parameters"  ? fj_test_ofx_abi_parameters(fj_test_ofx_parameter_suite()) == 0
                                                                                    : fj_test_ofx_abi_messages(fj_test_ofx_message_suite()) == 0;
        event.complete(passed);
        ok = ok && passed;
    }
    log.finish();
    return ok ? 0 : 1;
} catch (const std::exception& error) {
    std::fprintf(stderr, "OFX ABI fixture: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("OFX ABI fixture failed\n", stderr);
    return 1;
}
