#ifndef FJ_TEST_PREPARED_BOUNDARY_H
#define FJ_TEST_PREPARED_BOUNDARY_H

#include "juicer_cuda_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A C translation unit exercises the production render signature. */
int fj_test_execute_prepared_c(FjCuda* cuda, const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjErrorBuffer* error);

#ifdef __cplusplus
}

#include <string>

#include "Cuda/JuicerCudaExecutor.h"

namespace JuicerCudaTest {

    void check_frame_bindings(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, const RenderRecipe& recipe);

    FjStatus check_render_contract(FjCuda* cuda, const FjCudaContext& context, const FjFrame& frame, const FjSubmission& submission, const FjPreparedHostData& prepared, FjErrorBuffer* error);


    bool execute_boundary(
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const JuicerCuda::ExecutionFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const JuicerCuda::PreparedDescriptors& descriptors,
        std::string& diagnostic,
        bool checkContract = false);

} // namespace JuicerCudaTest
#endif

#endif
