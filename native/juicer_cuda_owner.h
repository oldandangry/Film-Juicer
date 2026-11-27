#pragma once

#include <mutex>
#include <string>
#include <string_view>

#include "juicer_cuda_api.h"

struct FjHostPreparation;
namespace JuicerProcess {
    class Root;
}

#if defined(JUICER_TEST_SEAMS)
namespace JuicerTest {
    void before_render_admission(FjCuda* owner, const FjCudaContext* context, const FjSubmission* submission);
}
#endif

namespace JuicerCuda {

    struct PendingContextLossRecovery;

    // Callback-local binding from the supplied registered handle; never creates.
    JuicerProcess::Root& borrow_root(FjCuda* cuda);

    // Registration/reentry precheck only: no lock, reservation or lifecycle admission.
    void check_native_call_admission(FjCuda* cuda);

    // Private callback-local admission. Owns no runtime or CUDA context. The
    // host lifetime excludes owner destruction until this borrow ends.
    class NativeCall final {
    public:
        explicit NativeCall(FjCuda* cuda);
        ~NativeCall();
        NativeCall(const NativeCall&) = delete;
        NativeCall& operator=(const NativeCall&) = delete;

        JuicerProcess::Root& root() const noexcept;

        FjStatus inspect(const FjFrame* frame, FjCudaContext* outContext, FjErrorBuffer* error);

        // Same admitted body as fj_cuda_render. Inputs expire at return; the
        // adapter retains full diagnostic/recovery detail until host delivery.
        FjRenderOutcome render(
            const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, FjAbortCallback abortCallback, PendingContextLossRecovery& recovery, std::string& diagnostic, const FjHostPreparation* hostPreparation = nullptr);

    private:
        std::unique_lock<std::mutex> _lock;
        FjCuda* _cuda;
    };


} // namespace JuicerCuda
