#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "instance_host.h"
#include "context_host.h"

namespace ofx_test::reference {
    using Json = nlohmann::json;
    Json value(const InstanceValue& item);
    InstanceValue instance_value(const Json& item);
    Json value(const Value& item);
    Value context_value(const Json& item);
    Json calls(const std::vector<InstanceCall>& items);
    std::vector<InstanceCall> instance_calls(const Json& items);
    Json calls(const std::vector<ContextCall>& items);
    std::vector<ContextCall> context_calls(const Json& items);
    Json values(const std::vector<InstanceValue>& items);
    std::vector<InstanceValue> instance_values(const Json& items);
    Json context(const ContextHost& host);
    void context(const Json& record, ContextHost& host);
    Json read(const std::filesystem::path& path, const char* owner);
} // namespace ofx_test::reference
