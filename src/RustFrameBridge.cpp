#include "RustFrameBridge.h"

#include <array>
#include <bit>
#include <cstdint>
#include <string_view>

#include "juicer_legacy_api.h"

namespace JuicerFrame {
    namespace {
        bool envelope(FjStatus status, const FjFrameGeometryFailure& failure, const FjErrorBuffer& error, const std::array<char, 128>& text, bool domain, Failure& rejected) {
            constexpr std::array categories{FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE};
            constexpr std::array messages{"", "invalid frame geometry arguments", "frame descriptor dimensions exceed signed integer range", "frame geometry preparation panicked"};
            if (failure.reason >= categories.size() || (!domain && failure.reason == FJ_FRAME_GEOMETRY_DOMAIN_RANGE) ||
                status.category != categories[failure.reason] || status.api != FJ_API_NONE || status.native_code != 0 ||
                error.data != text.data() || error.capacity != text.size() || error.length >= text.size() ||
                text[error.length] != '\0' || std::string_view(text.data(), error.length) != messages[failure.reason]) {
                rejected.reason = FailureReason::Integrity;
                JuicerCuda::set_failure(rejected.delivery, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "frame geometry projection invalid");
                return false;
            }
            rejected.reason = static_cast<FailureReason>(failure.reason);
            if (failure.reason != FJ_FRAME_GEOMETRY_NONE)
                JuicerCuda::set_failure(rejected.delivery, status, std::string_view(text.data(), error.length));
            return true;
        }
        bool cleared(FjExtent extent) {
            return extent.x == 0 && extent.y == 0 && extent.width == 0 && extent.height == 0;
        }
        bool invalid(Failure& failure) {
            failure.reason = FailureReason::Integrity;
            JuicerCuda::set_failure(failure.delivery, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "frame geometry projection invalid");
            return false;
        }
    } // namespace

    bool domain_from_bounds(Bounds bounds, Domain& out, Failure& rejected) {
        out = {};
        rejected = {};
        const FjRect input{bounds.x1, bounds.y1, bounds.x2, bounds.y2};
        FjExtent wire{};
        FjFrameGeometryFailure failure{};
        std::array<char, 128> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_frame_domain_prepare(&input, &wire, &failure, &error);
        if (!envelope(status, failure, error, text, true, rejected))
            return false;
        if (failure.reason != FJ_FRAME_GEOMETRY_NONE)
            return cleared(wire) ? false : invalid(rejected);
        out = {wire.x, wire.y, wire.width, wire.height};
        return true;
    }

    bool prepare_geometry(Domain fullFrame, float filmFormatLongEdgeMm, Geometry& out, Failure& rejected) {
        out = {};
        rejected = {};
        const FjFrameGeometryInput input{{fullFrame.x, fullFrame.y, fullFrame.width, fullFrame.height}, filmFormatLongEdgeMm};
        FjFrameGeometry wire{};
        FjFrameGeometryFailure failure{};
        std::array<char, 128> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_frame_geometry_prepare(&input, &wire, &failure, &error);
        if (!envelope(status, failure, error, text, false, rejected))
            return false;
        if (failure.reason != FJ_FRAME_GEOMETRY_NONE)
            return cleared(wire.full_frame) && std::bit_cast<std::uint32_t>(wire.pixel_size_um) == 0 ? false : invalid(rejected);
        out = {{wire.full_frame.x, wire.full_frame.y, wire.full_frame.width, wire.full_frame.height}, wire.pixel_size_um};
        return true;
    }
} // namespace JuicerFrame
