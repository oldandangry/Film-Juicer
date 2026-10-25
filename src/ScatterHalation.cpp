#include "ScatterHalation.h"

#include <cmath>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "RustHalationBridge.h"

namespace {
    bool convert_authored_control(
        double raw,
        const char* field,
        float& out,
        std::string& outDiagnostic) {
        const char* reason = nullptr;
        if (!std::isfinite(raw)) {
            reason = "non_finite";
        } else if (raw < 0.0) {
            reason = "below_minimum";
        } else if (raw > 2.0) {
            reason = "above_maximum";
        }
        if (reason) {
            outDiagnostic = "InvalidAuthoredControl component=scatter_halation field=";
            outDiagnostic += field;
            outDiagnostic += " reason=";
            outDiagnostic += reason;
            return false;
        }

        const float narrowed = static_cast<float>(raw);
        out = narrowed == 0.0f ? 0.0f : narrowed;
        return true;
    }

} // namespace

namespace Spektrafilm {
    bool build_scatter_halation_controls(
        const ScatterHalationRawControls& raw,
        ScatterHalationControls& out,
        std::string& outDiagnostic) {
        out = ScatterHalationControls{};
        outDiagnostic.clear();
        if (!raw.active) {
            return true;
        }

        ScatterHalationControls converted{};
        converted.active = true;
        if (!convert_authored_control(
                raw.scatterAmount,
                "scatter_amount",
                converted.scatterAmount,
                outDiagnostic) ||
            !convert_authored_control(
                raw.scatterSpatialScale,
                "scatter_spatial_scale",
                converted.scatterSpatialScale,
                outDiagnostic) ||
            !convert_authored_control(
                raw.halationAmount,
                "halation_amount",
                converted.halationAmount,
                outDiagnostic) ||
            !convert_authored_control(
                raw.halationSpatialScale,
                "halation_spatial_scale",
                converted.halationSpatialScale,
                outDiagnostic)) {
            return false;
        }

        out = converted;
        return true;
    }

    bool build_scatter_halation_frame_descriptor(
        const ScatterHalationOpticsRecipe& recipe,
        float pixelSizeUm,
        std::optional<ScatterHalationFrameDescriptor>& out,
        std::string& outDiagnostic) {
        JuicerCuda::Failure failure;
        const bool success = JuicerHalation::prepare_frame(recipe, pixelSizeUm, out, failure);
        outDiagnostic = std::move(failure.diagnostic);
        return success;
    }
} // namespace Spektrafilm
