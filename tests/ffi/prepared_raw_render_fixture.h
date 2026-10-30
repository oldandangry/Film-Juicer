#pragma once

#include "juicer_cuda_call_scope.h"
#include "juicer_test_api.h"
#include "prepared_boundary.h"

namespace JuicerCudaTest {

    namespace PreparedRawRenderFixture {

        struct Consumer {
            FjCudaCallAccess* access = nullptr;
            const FjPreparedHostData* prepared = nullptr;
            const FjFrame* frame = nullptr;
            const FjCudaContext* context = nullptr;
            const FjSubmission* submission = nullptr;
            FjAbortCallback abort{};
            FjRenderOutcome outcome{{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, 0};
            bool invoked = false;
        };

        extern "C" inline FjStatus render_with_borrowed_preparation(void* user, const FjHostPreparation* host, FjErrorBuffer* error) noexcept {
            auto* consumer = static_cast<Consumer*>(user);
            if (!consumer || !host)
                return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
            consumer->invoked = true;
            consumer->outcome = fj_test_execute_prepared_c(consumer->access, host, consumer->prepared, consumer->frame, consumer->context, consumer->submission, consumer->abort, error);
            return consumer->outcome.status;
        }

        inline FjCallCompletionResult complete_without_application_recovery(void*, const FjCallCompletionInput* input) noexcept {
            if (!input)
                return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
            return {input->outcome, 0};
        }

    } // namespace PreparedRawRenderFixture

    inline FjRenderOutcome render_raw_prepared(
        FjCuda* owner,
        const FjFrame& frame,
        const FjSubmission& submission,
        const FjPreparedHostData& prepared,
        FjAbortCallback abort = {},
        FjErrorBuffer* error = nullptr) {
        using namespace PreparedRawRenderFixture;
        FjErrorBuffer emptyError{nullptr, 0, 0};
        FjErrorBuffer* const effectiveError = error ? error : &emptyError;
        const FjRenderOutcome unsupported{{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, 0};
        JuicerCuda::CallScope scope(owner, prepared.route, &complete_without_application_recovery, nullptr);
        FjCudaCallAccess access = scope.access();
        bool admitted = false;
        bool released = false;
        FjCudaContext context{};
        bool inspected = false;
        Consumer consumer{&access, &prepared, &frame, &context, &submission, abort};
        FjRenderOutcome outcome = unsupported;

        const auto finish = [&](bool renderOutcomeValid) {
            if (admitted && !released) {
                access.release(access.user);
                released = true;
            }
            const FjCallCompletionInput input{outcome,
                                              {},
                                              renderOutcomeValid ? 1u : 0u,
                                              0,
                                              0,
                                              inspected ? 1u : 0u,
                                              inspected ? context : FjCudaContext{},
                                              0};
            const FjCallCompletionResult completed = access.complete(access.user, &input);
            return completed.outcome;
        };

        FjStatus status = access.precheck(access.user, effectiveError);
        if (status.category != FJ_STATUS_SUCCESS) {
            outcome = {status, 0};
            return finish(false);
        }
        status = access.admit(access.user, effectiveError);
        if (status.category != FJ_STATUS_SUCCESS) {
            outcome = {status, 0};
            return finish(false);
        }
        admitted = true;
        status = access.inspect(access.user, &frame, &context, effectiveError);
        if (status.category != FJ_STATUS_SUCCESS) {
            outcome = {status, 0};
            return finish(false);
        }
        inspected = true;

        const FjStatus hostStatus = fj_test_with_host_preparation(&consumer, &render_with_borrowed_preparation, effectiveError);
        outcome = consumer.invoked ? consumer.outcome : FjRenderOutcome{hostStatus, 0};
        return finish(consumer.invoked);
    }

} // namespace JuicerCudaTest
