#pragma once

#include <optional>

#include "ScatterHalation.h"

namespace JuicerCuda {
    struct Failure;
}

struct ScatterHalationRawControls {
    bool active = false;
    double scatterAmount = 1.0;
    double scatterSpatialScale = 1.0;
    double halationAmount = 1.0;
    double halationSpatialScale = 1.0;
};

struct ScatterHalationControls {
    bool active = false;
    float scatterAmount = 0.0f;
    float scatterSpatialScale = 0.0f;
    float halationAmount = 0.0f;
    float halationSpatialScale = 0.0f;
};


namespace Spektrafilm {
    using ::ScatterHalationControls;
    using ::ScatterHalationRawControls;
    bool build_scatter_halation_controls(
        const ScatterHalationRawControls& raw,
        ScatterHalationControls& out,
        std::string& outDiagnostic);


    bool build_scatter_halation_frame_descriptor(
        const ScatterHalationOpticsRecipe& recipe,
        float pixelSizeUm,
        std::optional<ScatterHalationFrameDescriptor>& out,
        std::string& outDiagnostic);
} // namespace Spektrafilm

namespace JuicerHalation {
    bool prepare_frame(const ScatterHalationOpticsRecipe&, float pixelSizeUm, std::optional<ScatterHalationFrameDescriptor>&, JuicerCuda::Failure&);
} // namespace JuicerHalation
