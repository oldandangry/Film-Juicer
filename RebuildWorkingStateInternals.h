#pragma once

#include "WorkingState.h"
#include "Couplers.h"

#include <cstdint>

namespace RebuildWorkingState {

    struct NegativeReuseContext {
        std::uint64_t activeBuildCounter = 0;
        std::uint64_t lastHash = 0;
        int lastFilmStock = 0;
        int lastViewIll = 0;
        int lastEnlargerIll = 0;
    };

    inline bool dir_runtime_equivalent(const Couplers::Runtime& a, const Couplers::Runtime& b) {
        if (a.active != b.active) return false;
        if (a.highShift != b.highShift) return false;
        if (a.spatialSigmaMicrometers != b.spatialSigmaMicrometers) return false;
        if (a.spatialSigmaPixels != b.spatialSigmaPixels) return false;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (a.M[r][c] != b.M[r][c]) {
                    return false;
                }
            }
        }
        return true;
    }

    inline bool negative_inputs_match(const NegativeReuseContext& ctx,
        int filmStockIndex,
        int viewIll,
        int enlargerIll)
    {
        return (ctx.lastHash != 0) &&
            (ctx.lastFilmStock == filmStockIndex) &&
            (ctx.lastViewIll == viewIll) &&
            (ctx.lastEnlargerIll == enlargerIll);
    }

    inline bool can_reuse_negative_params(const NegativeReuseContext& ctx,
        const WorkingState* prev,
        const WorkingState& candidate,
        int filmStockIndex,
        int viewIll,
        int enlargerIll)
    {
        if (!prev) {
            return false;
        }
        if (prev->buildCounter == 0) {
            return false;
        }
        if (prev->buildCounter != ctx.activeBuildCounter) {
            return false;
        }
        if (!negative_inputs_match(ctx, filmStockIndex, viewIll, enlargerIll)) {
            return false;
        }
        if (!dir_runtime_equivalent(prev->dirRT, candidate.dirRT)) {
            return false;
        }
        if (prev->dirPrecorrected != candidate.dirPrecorrected) {
            return false;
        }
        return true;
    }

} // namespace RebuildWorkingState