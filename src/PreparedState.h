#pragma once

#include "juicer_legacy_api.h"

#include <optional>

struct ParamSnapshot;
struct FjCudaCallAccess;
struct FjPreparedFramePolicy;
struct FjSubmissionAccess;
struct FjCallCompletionResult;

// Profile-key strings borrow the snapshot through the synchronous create call.
FjPreparationParams preparation_params(const ParamSnapshot&);

class PreparedState {
public:
    PreparedState(const PreparedState&) = delete;
    PreparedState& operator=(const PreparedState&) = delete;

    PreparedState(PreparedState&& other) noexcept;
    PreparedState& operator=(PreparedState&& other) noexcept;
    ~PreparedState();

    // The pointer-bearing view is valid only while this holder remains alive.
    FjStatus view(FjPreparedStateView& out, FjErrorBuffer& error) const;

    FjStatus frame_policy(const FjExtent&, FjPreparedFramePolicy&, FjErrorBuffer&) const;
    FjCallCompletionResult render(
        FjAssets*, FjCudaCallAccess, const FjFrame&, const FjPreparedFramePolicy&,
        FjSubmissionAccess, FjAbortCallback, FjErrorBuffer&) const;
    FjCallCompletionResult reject_missing_instance(
        FjAssets*, FjCudaCallAccess, const FjFrame&, FjAbortCallback, FjErrorBuffer&) const;

private:
    struct AdoptTag {};

    PreparedState(FjPreparedState* owner, AdoptTag) noexcept;
    void release() noexcept;

    FjPreparedState* owner_;

    friend FjStatus prepare_state(
        FjAssets*, const ParamSnapshot&, std::optional<PreparedState>&, FjErrorBuffer&);
};

FjStatus prepare_state(
    FjAssets* assets,
    const ParamSnapshot& snapshot,
    std::optional<PreparedState>& out,
    FjErrorBuffer& error);

FjCallCompletionResult reject_missing_render_state(
    FjAssets*, FjCudaCallAccess, const FjFrame&, FjAbortCallback, FjErrorBuffer&);
