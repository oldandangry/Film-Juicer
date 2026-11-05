#pragma once

#include <nlohmann/json.hpp>

struct FjPreparedStateView;
struct FjPreparationParams;

namespace PreparedStateOracle {

    using Json = nlohmann::json;

    Json view(const FjPreparedStateView& value);
    Json params(const FjPreparationParams& value);
    void compare(const Json& expected, const Json& actual);

} // namespace PreparedStateOracle
