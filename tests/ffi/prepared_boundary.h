#ifndef FJ_TEST_PREPARED_BOUNDARY_H
#define FJ_TEST_PREPARED_BOUNDARY_H

#include "juicer_legacy_api.h"
#include "juicer_cuda_call_scope.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A C translation unit exercises the production render signature. */
FjRenderOutcome fj_test_execute_prepared_c(const FjCudaCallAccess* access, const FjHostPreparation* host_preparation, const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjAbortCallback abort, FjErrorBuffer* error);

#ifdef __cplusplus
}

#include <string>
#include <memory>
#include <optional>

#include "Cuda/JuicerCudaExecutor.h"
#include "PreparedState.h"

struct ParamSnapshot;

namespace JuicerCudaTest {

    struct RawPreparedStateDeleter {
        void operator()(FjPreparedState* owner) const noexcept;
    };
    using RawPreparedStateHold = std::unique_ptr<FjPreparedState, RawPreparedStateDeleter>;
    RawPreparedStateHold prepare_raw_state(FjAssets*, const ParamSnapshot&, FjStatus&, std::string&);
    std::optional<PreparedState> prepare_state(FjAssets*, const ParamSnapshot&, FjStatus&, std::string&);

    enum class StateLifetimeCase : std::uint8_t {
        SuppliedStream,
        PresentNullStream,
        AbsentStream,
        CancelAfterPrepare,
        CancelAfterExecute,
        PinnedAllocation,
        PinnedEventCreation,
        PinnedCopy,
        PinnedEventRecord,
        PinnedSynchronization,
        IdleRetirement,
        DeferredDestruction,
        Count
    };

    bool execute_prepared_state(FjAssets*, const ParamSnapshot&, const FjFrame&, std::uint64_t instanceToken, bool contract, bool lifetime, std::string&, StateLifetimeCase = StateLifetimeCase::SuppliedStream);
    void check_prepared_state_lifetime(std::optional<PreparedState>&, FjAssets*, FjCuda*, const FjCudaContext&, const FjFrame&, const FjSubmission&, FjPreparedHostData&, FjErrorBuffer*, StateLifetimeCase);

    // The retained grain fixtures need a recipe; DIR-only held-state callers do not.
    void check_frame_bindings(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, const RenderRecipe* recipe = nullptr);

    FjStatus check_render_contract(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, FjErrorBuffer* error);


    bool execute_boundary(
        const PreparedState& owner,
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const JuicerCuda::ExecutionFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const JuicerCuda::PreparedDescriptors& descriptors,
        std::string& diagnostic);

} // namespace JuicerCudaTest
#endif

#endif
