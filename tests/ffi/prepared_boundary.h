#ifndef FJ_TEST_PREPARED_BOUNDARY_H
#define FJ_TEST_PREPARED_BOUNDARY_H

#include "juicer_test_api.h"
#include "juicer_cuda_call_scope.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A C translation unit exercises the production render signature. */
FjRenderOutcome fj_test_execute_prepared_c(const FjCudaCallAccess* access, const FjHostPreparation* host_preparation, const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjAbortCallback abort, FjErrorBuffer* error);

#ifdef __cplusplus
}

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "Cuda/JuicerCudaExecutor.h"
#include "owner_fixture.h"
#include "preparation_fixture.h"


namespace JuicerCudaTest {

    struct RawPreparedStateDeleter {
        void operator()(FjPreparedState* owner) const noexcept;
    };
    using RawPreparedStateHold = std::unique_ptr<FjPreparedState, RawPreparedStateDeleter>;
    RawPreparedStateHold prepare_raw_state(const FjAssets*, const JuicerTestSupport::PreparationInput&, FjStatus&, std::string&);
    std::optional<JuicerTestSupport::PreparedStateHold> prepare_state(const FjAssets*, const JuicerTestSupport::PreparationInput&, FjStatus&, std::string&);

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
        SpectralPacketOverlap,
        Count
    };

    bool execute_prepared_state(FjRuntimeView, const JuicerTestSupport::PreparationInput&, const FjFrame&, std::uint64_t instanceToken, bool contract, bool lifetime, std::string&, StateLifetimeCase = StateLifetimeCase::SuppliedStream);
    void check_prepared_state_lifetime(std::optional<JuicerTestSupport::PreparedStateHold>&, const FjAssets*, FjCuda*, const FjCudaContext&, const FjFrame&, const FjSubmission&, FjPreparedHostData&, FjErrorBuffer*, StateLifetimeCase);
    struct BoundScannerPacket {
        FjPreparedHostData prepared;
        FjSubmission submission;
    };
    void check_spectral_packet_overlap(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const std::array<BoundScannerPacket, 2>& packets);

    // Temporal grain fixtures borrow the complete state; DIR-only callers need no hold here.
    void check_frame_bindings(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, const FjPreparedStateView* state, const FjHostPreparation* hostPreparation);

    FjStatus check_render_contract(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, FjErrorBuffer* error);
    enum class ReadbackProbe : std::uint8_t {
        Inline,
        Plane,
        FusedDir
    };
    enum class ScanReadbackFault : std::uint8_t {
        Existing,
        ScratchOverflow
    };
    void check_scan_readback(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, ReadbackProbe probe, const BoundScannerPacket* switchedReporter = nullptr, ScanReadbackFault fault = ScanReadbackFault::Existing);

    void check_grain_dispatch(FjCuda* cuda, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& current, const FjHostPreparation* hostPreparation);

    bool execute_boundary(
        FjRuntimeView runtime,
        const JuicerTestSupport::PreparedStateHold& owner,
        const FjFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::string& diagnostic,
        const std::string& invocationPhase,
        bool grainDispatchContract,
        const JuicerTestSupport::PreparationInput* sourceParams = nullptr);

    void check_shipping_print_preflash(const std::filesystem::path& resources, const JuicerTestSupport::PreparationInput& sourceParams);

} // namespace JuicerCudaTest
#endif

#endif
