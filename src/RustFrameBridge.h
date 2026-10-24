#pragma once

#include <cstdint>

#include "Cuda/JuicerCudaFailure.h"

namespace JuicerFrame {
    struct Bounds {
        int x1, y1, x2, y2;
    };
    struct Domain {
        int x, y, width, height;
    };
    struct Geometry {
        Domain fullFrame;
        float pixelSizeUm;
    };
    enum class FailureReason : std::uint8_t {
        None,
        Raw,
        DomainRange,
        Panic,
        Integrity
    };
    struct Failure {
        FailureReason reason = FailureReason::None;
        JuicerCuda::Failure delivery;
    };

    // SF_TEMP_BRIDGE_FRAME_GEOMETRY: remove at the final native callback consumer,
    // no later than S6.D. These signed counts do not admit allocation coverage.
    bool domain_from_bounds(Bounds, Domain&, Failure&);
    bool prepare_geometry(Domain fullFrame, float filmFormatLongEdgeMm, Geometry&, Failure&);
} // namespace JuicerFrame
