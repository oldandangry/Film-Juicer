#include "host_reference.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#ifndef JUICER_TEST_PRESET_NAME
#define JUICER_TEST_PRESET_NAME "unspecified"
#endif

namespace ofx_test::reference {
    namespace {
        void shape(const Json& value, std::size_t size) {
            if (!value.is_array() || value.size() != size)
                throw std::runtime_error("invalid OFX reference record shape");
        }
        std::string bits(double value) {
            std::ostringstream out;
            out << std::hex << std::setfill('0') << std::setw(16) << std::bit_cast<std::uint64_t>(value);
            return out.str();
        }
        double from_bits(const Json& value) {
            const auto text = value.get<std::string>();
            if (text.size() != 16 || text.find_first_not_of("0123456789abcdef") != std::string::npos)
                throw std::runtime_error("invalid OFX reference floating bits");
            return std::bit_cast<double>(std::stoull(text, nullptr, 16));
        }
        int pointer_token;
        template <class Variant>
        Json encode_value(const Variant& item) {
            return std::visit([](const auto& v) -> Json {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, int>)
                    return Json::array({"i", v});
                else if constexpr (std::is_same_v<T, double>)
                    return Json::array({"d", bits(v)});
                else if constexpr (std::is_same_v<T, std::string>)
                    return Json::array({"s", v});
                else
                    return Json::array({"p", v != nullptr});
            },
                              item);
        }
        template <class T, class F>
        Json encode_list(const std::vector<T>& items, F encode) {
            auto out = Json::array();
            for (const auto& item : items)
                out.push_back(encode(item));
            return out;
        }
    } // namespace
    Json value(const InstanceValue& item) {
        return encode_value(item);
    }
    Json value(const Value& item) {
        return encode_value(item);
    }
    InstanceValue instance_value(const Json& item) {
        shape(item, 2);
        const auto tag = item[0].get<std::string>();
        if (tag == "i")
            return item[1].get<int>();
        if (tag == "d")
            return from_bits(item[1]);
        if (tag == "s")
            return item[1].get<std::string>();
        if (tag == "p")
            return item[1].get<bool>() ? static_cast<void*>(&pointer_token) : nullptr;
        throw std::runtime_error("invalid OFX reference value tag");
    }
    Value context_value(const Json& item) {
        const auto decoded = instance_value(item);
        if (const auto* v = std::get_if<int>(&decoded))
            return *v;
        if (const auto* v = std::get_if<double>(&decoded))
            return *v;
        if (const auto* v = std::get_if<std::string>(&decoded))
            return *v;
        throw std::runtime_error("pointer in OFX context value");
    }
    Json values(const std::vector<InstanceValue>& items) {
        return encode_list(items, [](const auto& item) {
            return value(item);
        });
    }
    std::vector<InstanceValue> instance_values(const Json& items) {
        if (!items.is_array())
            throw std::runtime_error("invalid OFX reference values");
        std::vector<InstanceValue> out;
        for (const auto& item : items)
            out.push_back(instance_value(item));
        return out;
    }
    Json calls(const std::vector<InstanceCall>& items) {
        return encode_list(items, [](const auto& c) {
            return Json::array({c.operation, c.target, c.property, c.index, values(c.values), c.scope, c.status});
        });
    }
    std::vector<InstanceCall> instance_calls(const Json& items) {
        if (!items.is_array())
            throw std::runtime_error("invalid OFX instance calls");
        std::vector<InstanceCall> out;
        for (const auto& c : items) {
            shape(c, 7);
            out.push_back({c[0].get<std::string>(), c[1].get<std::string>(), c[2].get<std::string>(), c[3].get<int>(), instance_values(c[4]), c[5].get<std::string>(), c[6].get<int>()});
        }
        return out;
    }
    Json calls(const std::vector<ContextCall>& items) {
        return encode_list(items, [](const auto& c) {
            return Json::array({c.operation, c.property, c.text, c.index, c.target, value(c.value)});
        });
    }
    std::vector<ContextCall> context_calls(const Json& items) {
        if (!items.is_array())
            throw std::runtime_error("invalid OFX context calls");
        std::vector<ContextCall> out;
        for (const auto& c : items) {
            shape(c, 6);
            out.push_back({c[0].get<std::string>(), c[1].get<std::string>(), c[2].get<std::string>(), c[3].get<int>(), c[4].get<int>(), context_value(c[5])});
        }
        return out;
    }
    Json context(const ContextHost& host) {
        auto properties = [](const auto& map) {
            auto out = Json::array();
            for (const auto& [key, items] : map)
                out.push_back(Json::array({key, encode_list(items, [](const auto& v) {
                                               return value(v);
                                           })}));
            return out;
        };
        auto definitions = Json::array();
        for (const auto& d : host.definitions)
            definitions.push_back(Json::array({d.name, d.type, properties(d.properties)}));
        return Json::array({calls(host.calls), calls(host.startup), properties(host.effect_values), definitions});
    }
    void context(const Json& record, ContextHost& host) {
        shape(record, 4);
        host.calls = context_calls(record[0]);
        host.startup = context_calls(record[1]);
        auto properties = [](const Json& input, auto& output) {
            if (!input.is_array())
                throw std::runtime_error("invalid OFX context properties");
            for (const auto& pair : input) {
                shape(pair, 2);
                std::vector<Value> values;
                if (!pair[1].is_array())
                    throw std::runtime_error("invalid OFX context property values");
                for (const auto& item : pair[1])
                    values.push_back(context_value(item));
                if (!output.emplace(pair[0].get<std::string>(), std::move(values)).second)
                    throw std::runtime_error("duplicate OFX context property");
            }
        };
        properties(record[2], host.effect_values);
        if (!record[3].is_array())
            throw std::runtime_error("invalid OFX context definitions");
        for (const auto& item : record[3]) {
            shape(item, 3);
            Definition d;
            d.name = item[0].get<std::string>();
            d.type = item[1].get<std::string>();
            properties(item[2], d.properties);
            host.definitions.push_back(std::move(d));
        }
    }
    Json read(const std::filesystem::path& path, const char* owner) {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("cannot open OFX reference");
        Json record;
        in >> record;
        in >> std::ws;
        if (!in.eof())
            throw std::runtime_error("extra OFX reference data");
        shape(record, 5);
        const auto presets = Json::array({"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"});
        const bool v1 = record[0] == "film-juicer-ofx-reference-v1" && record[2] == JUICER_TEST_PRESET_NAME;
        const bool v2 = record[0] == "film-juicer-ofx-reference-v2" && record[2] == presets &&
                        std::find(presets.begin(), presets.end(), JUICER_TEST_PRESET_NAME) != presets.end();
        if ((!v1 && !v2) || record[1] != owner || record[3] != "d4bc2135d02ed88c91403040c4429a4fdb802a2b")
            throw std::runtime_error("OFX reference provenance mismatch");
        return record[4];
    }
} // namespace ofx_test::reference
