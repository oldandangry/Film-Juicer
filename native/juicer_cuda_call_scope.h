#ifndef FJ_CUDA_CALL_SCOPE_H
#define FJ_CUDA_CALL_SCOPE_H

#include "juicer_cuda_frame_values.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed from one Rust call. Invoke serially on the calling thread; the table,
 * user and candidate views expire when the immediate native render returns. */
typedef struct FjHostPreparation {
    void* user;
    FjStatus (*scanner_lut)(void*, const FjScannerLutInput*, FjScannerLutView*, FjScannerLutFailure*, FjErrorBuffer*);
    FjStatus (*gaussian)(void*, const FjGaussianKernelInput*, FjGaussianKernelView*, FjGaussianKernelFailure*, FjErrorBuffer*);
} FjHostPreparation;

typedef struct FjCallCompletionInput {
    FjRenderOutcome outcome;
    FjStringView diagnostic;
    uint32_t render_outcome_valid;
    uint32_t abort_failed;
    uint32_t recovery_scope_entered;
    uint32_t has_inspected_context;
    FjCudaContext inspected_context;
    uint32_t force_fatal;
} FjCallCompletionInput;

typedef struct FjCallCompletionResult {
    FjRenderOutcome outcome;
    uint32_t fatal;
} FjCallCompletionResult;

typedef struct FjCudaCallAccess {
    void* user;
    FjStatus (*precheck)(void*, FjErrorBuffer*);
    FjStatus (*admit)(void*, FjErrorBuffer*);
    void (*release)(void*);
    FjStatus (*inspect)(void*, const FjFrame*, FjCudaContext*, FjErrorBuffer*);
    FjStatus (*preview)(void*, const FjRect*, const FjRect*, uint32_t, FjAutoExposure*, FjErrorBuffer*);
    FjStatus (*frame_descriptors)(void*, const FjFrameMathInput*, FjNativeFrameDescriptors*, FjErrorBuffer*);
    FjRenderOutcome (*render)(void*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, const FjHostPreparation*, FjAbortCallback, FjErrorBuffer*);
    FjCallCompletionResult (*complete)(void*, const FjCallCompletionInput*);
} FjCudaCallAccess;

typedef struct FjSubmissionAccess {
    uint64_t instance_token;
    uint64_t frame_token;
    void* user;
    FjStatus (*select)(void*, const FjCudaContext*, const FjSubmission*, FjSubmission*, FjErrorBuffer*);
} FjSubmissionAccess;

#define FJ_UNAVAILABLE_STATE 0U
#define FJ_UNAVAILABLE_INSTANCE 1U

FjStatus fj_legacy_prepared_state_frame_policy(const FjPreparedState*, const FjExtent*, FjPreparedFramePolicy*, FjErrorBuffer*);
FjCallCompletionResult fj_legacy_prepared_state_render(const FjAssets*, const FjPreparedState*, FjCudaCallAccess, const FjFrame*, const FjPreparedFramePolicy*, FjSubmissionAccess, FjAbortCallback, FjErrorBuffer*);
FjCallCompletionResult fj_legacy_prepared_unavailable(const FjAssets*, const FjPreparedState*, uint32_t, FjCudaCallAccess, const FjFrame*, FjAbortCallback, FjErrorBuffer*);

#ifdef __cplusplus
}

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "Cuda/JuicerCudaExecutor.h"
#include "juicer_cuda_owner.h"

namespace JuicerCuda {

    using CallCompletionContinuation = FjCallCompletionResult (*)(void*, const FjCallCompletionInput*) noexcept;

    class CallScope final {
    public:
        CallScope(FjCuda* owner, std::optional<uint32_t> route, CallCompletionContinuation completion, void* completionUser) noexcept;
        CallScope(const CallScope&) = delete;
        CallScope& operator=(const CallScope&) = delete;

        FjCudaCallAccess access() noexcept;
        PendingContextLossRecovery& recovery() noexcept;
        const std::string& diagnostic() const noexcept;
        void set_diagnostic(std::string_view value) noexcept;

    private:
        static FjStatus precheck(void*, FjErrorBuffer*) noexcept;
        static FjStatus admit(void*, FjErrorBuffer*) noexcept;
        static void release(void*) noexcept;
        static FjStatus inspect(void*, const FjFrame*, FjCudaContext*, FjErrorBuffer*) noexcept;
        static FjStatus preview(void*, const FjRect*, const FjRect*, uint32_t, FjAutoExposure*, FjErrorBuffer*) noexcept;
        static FjStatus frame_descriptors(void*, const FjFrameMathInput*, FjNativeFrameDescriptors*, FjErrorBuffer*) noexcept;
        static FjRenderOutcome render(void*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, const FjHostPreparation*, FjAbortCallback, FjErrorBuffer*) noexcept;
        static FjCallCompletionResult complete(void*, const FjCallCompletionInput*) noexcept;

        FjStatus inspect(const FjFrame*, FjCudaContext*, FjErrorBuffer*) noexcept;
        FjStatus preview(const FjRect*, const FjRect*, uint32_t, FjAutoExposure*, FjErrorBuffer*) noexcept;
        FjStatus frame_descriptors(const FjFrameMathInput*, FjNativeFrameDescriptors*, FjErrorBuffer*) noexcept;
        FjRenderOutcome render(const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, const FjHostPreparation*, FjAbortCallback, FjErrorBuffer*) noexcept;
        FjCallCompletionResult complete(const FjCallCompletionInput*) noexcept;

        FjCuda* _owner = nullptr;
        std::optional<uint32_t> _route;
        CallCompletionContinuation _completion = nullptr;
        void* _completionUser = nullptr;
        std::optional<NativeCall> _call;
        PendingContextLossRecovery _recovery{};
        std::string _diagnostic;
        std::optional<FjFrame> _frame;
        FjCudaContext _inspectedContext{};
        float _pixelSizeUm = 0.0f;
        bool _hasInspectedContext = false;
        bool _admissionAttempted = false;
        bool _released = false;
        bool _completed = false;
    };

} // namespace JuicerCuda
#endif

#endif
