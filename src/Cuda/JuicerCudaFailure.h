#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "juicer_cuda_api.h"

namespace JuicerCuda {

    // Native preparation/execution failures use the ABI vocabulary at their origin.
    // A diagnostic-only helper has no external API code.
    struct Failure {
        FjStatus status{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        std::string diagnostic;
        // Cleanup can prove context loss without replacing the operation's status.
        std::optional<FjStatus> cleanupContextLoss = std::nullopt;
    };

    FjStatus runtime_failure_status(std::int32_t code) noexcept;
    FjStatus driver_failure_status(std::int32_t code) noexcept;
    FjStatus cufft_failure_status(std::int32_t code) noexcept;
    bool allocation_capacity_exhausted(const Failure& failure) noexcept;
    bool context_loss(const Failure& failure) noexcept;
    std::optional<FjStatus> context_loss_status(const Failure& failure) noexcept;
    void record_cleanup_context_loss(Failure& failure, FjStatus cleanupStatus) noexcept;
    // Diagnostic storage cannot replace an already selected failure origin.
    void set_failure(Failure& failure, FjStatus status, std::string_view diagnostic) noexcept;
    FjStatus write_status(FjStatus status, std::string_view diagnostic, FjErrorBuffer* error) noexcept;

} // namespace JuicerCuda
