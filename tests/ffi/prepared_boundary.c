#include "prepared_boundary.h"

int fj_test_execute_prepared_c(FjCuda* cuda, const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjErrorBuffer* error) {
    const FjAbortCallback callback = {0, 0};
    return fj_cuda_render(cuda, context, frame, submission, prepared, callback, error).status.category == FJ_STATUS_SUCCESS;
}
