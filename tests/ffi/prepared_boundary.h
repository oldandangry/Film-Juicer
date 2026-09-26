#ifndef FJ_TEST_PREPARED_BOUNDARY_H
#define FJ_TEST_PREPARED_BOUNDARY_H

#include "juicer_cuda_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fixture-only bridge for already-inspected disjoint float RGB/RGBA images
 * with positive pitches. It is not the final frame-admission or status API. */
int fj_test_execute_prepared_cpp(const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjErrorBuffer* error);
int fj_test_execute_prepared_c(const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjErrorBuffer* error);

#ifdef __cplusplus
}

#include <string>

#include "Cuda/JuicerCudaExecutor.h"

namespace JuicerCudaTest {

    bool execute_boundary(
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const JuicerCuda::ExecutionFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const JuicerCuda::PreparedDescriptors& descriptors,
        std::string& diagnostic);

} // namespace JuicerCudaTest
#endif

#endif
