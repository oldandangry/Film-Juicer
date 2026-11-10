#include "prepared_boundary.h"

FjRenderOutcome fj_test_execute_prepared_c(const FjCudaCallAccess* access, const FjHostPreparation* host_preparation, const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjAbortCallback abort, FjErrorBuffer* error) {
    FjRenderOutcome rejected = {{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, 0};
    if (access == 0 || access->render == 0 || host_preparation == 0 || prepared == 0 || frame == 0 || context == 0 || submission == 0) {
        return rejected;
    }
    return access->render(access->user, context, frame, submission, prepared, host_preparation, abort, error);
}
