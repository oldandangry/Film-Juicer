#include "prepared_boundary.h"

int fj_test_execute_prepared_c(const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjErrorBuffer* error) {
    return fj_test_execute_prepared_cpp(prepared, frame, context, submission, error);
}
