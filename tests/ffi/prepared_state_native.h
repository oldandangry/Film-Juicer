#pragma once

#include <array>
#include <string>

#include "juicer_legacy_api.h"

struct FocusedRenderStateBuildProduct;
struct ParamSnapshot;

// SF_TEMP_BRIDGE_NATIVE_PREPARED_VIEW: independent native expected projection
// for the S4E prepared-state oracle. Remove with the I7 native-model deletion.
// Borrowed output spans refer to the product or these two fixed backing arrays;
// both the product and backing must outlive the immediate oracle serialization.
namespace PreparedStateOracle {
    struct NativeBacking {
        std::array<float, 81> preflash{};
        std::array<float, 81> print_source_illuminant{};
    };

    bool SF_TEMP_BRIDGE_native_prepared_view(
        const FocusedRenderStateBuildProduct& product,
        const ParamSnapshot& params,
        FjPreparedStateView& out,
        NativeBacking& backing,
        std::string& diagnostic);
} // namespace PreparedStateOracle
