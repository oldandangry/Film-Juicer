#include <stddef.h>
#include <stdint.h>

#include "juicer_test_api.h"
#include "juicer_cuda_call_scope.h"

#define FJ_ABI_TYPE(type, size, alignment)                \
    _Static_assert(sizeof(type) == size, "size: " #type); \
    _Static_assert(_Alignof(type) == alignment, "alignment: " #type);
#define FJ_ABI_FIELD(type, field, offset) \
    _Static_assert(offsetof(type, field) == offset, "offset: " #type "." #field);
#define FJ_ABI_VALUE(tag, value) _Static_assert(tag == value, "value: " #tag);
#include "prepared_call_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE

_Static_assert(_Generic(((FjGrainCoefficients*)0)->flags, uint32_t: 1, default: 0), "grain flags width");
_Static_assert(_Generic(((FjGrainCoefficients*)0)->debug_view, uint32_t: 1, default: 0), "grain debug width");
_Static_assert(_Generic(((FjGrainCoefficients*)0)->sublayer_count, size_t: 1, default: 0), "grain layer count width");
_Static_assert(_Generic(&((FjGrainCoefficients*)0)->micro_structure, float (*)[2]: 1, default: 0), "grain micro shape");
_Static_assert(_Generic(&((FjGrainCoefficients*)0)->particle_density_min_cmy, float (*)[3]: 1, default: 0), "grain density shape");
_Static_assert(_Generic(&((FjGrainCoefficients*)0)->uniformity_cmy, float (*)[3]: 1, default: 0), "grain uniformity shape");
_Static_assert(_Generic(&((FjGrainCoefficients*)0)->layer_axis_block_prefix_max, float (*)[3][16]: 1, default: 0), "grain axis shape");
_Static_assert(_Generic(((FjCallCompletionInput*)0)->render_outcome_valid, uint32_t: 1, default: 0), "completion validity width");
_Static_assert(_Generic(((FjCallCompletionInput*)0)->abort_failed, uint32_t: 1, default: 0), "completion abort width");
_Static_assert(_Generic(((FjCallCompletionInput*)0)->recovery_scope_entered, uint32_t: 1, default: 0), "completion recovery width");
_Static_assert(_Generic(((FjCallCompletionInput*)0)->has_inspected_context, uint32_t: 1, default: 0), "completion context flag width");
_Static_assert(_Generic(((FjCallCompletionInput*)0)->force_fatal, uint32_t: 1, default: 0), "completion fatal width");
_Static_assert(_Generic(((FjCallCompletionResult*)0)->fatal, uint32_t: 1, default: 0), "completion result width");

typedef FjStatus (*FjCallPrecheckFn)(void*, FjErrorBuffer*);
typedef FjStatus (*FjCallAdmitFn)(void*, FjErrorBuffer*);
typedef void (*FjCallReleaseFn)(void*);
typedef FjStatus (*FjCallInspectFn)(void*, const FjFrame*, FjCudaContext*, FjErrorBuffer*);
typedef FjStatus (*FjCallPreviewFn)(void*, const FjRect*, const FjRect*, uint32_t, FjAutoExposure*, FjErrorBuffer*);
typedef FjStatus (*FjCallFrameDescriptorsFn)(void*, const FjFrameMathInput*, FjNativeFrameDescriptors*, FjErrorBuffer*);
typedef FjRenderOutcome (*FjCallRenderFn)(void*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, const FjHostPreparation*, FjAbortCallback, FjErrorBuffer*);
typedef FjCallCompletionResult (*FjCallCompleteFn)(void*, const FjCallCompletionInput*);
typedef FjStatus (*FjHostScannerLutFn)(void*, const FjScannerLutInput*, FjScannerLutView*, FjScannerLutFailure*, FjErrorBuffer*);
typedef FjStatus (*FjHostGaussianFn)(void*, const FjGaussianKernelInput*, FjGaussianKernelView*, FjGaussianKernelFailure*, FjErrorBuffer*);
typedef FjStatus (*FjSubmissionSelectFn)(void*, const FjCudaContext*, const FjSubmission*, FjSubmission*, FjErrorBuffer*);
typedef FjStatus (*FjPolicyEntryFn)(const FjPreparedState*, const FjExtent*, FjPreparedFramePolicy*, FjErrorBuffer*);
typedef FjCallCompletionResult (*FjRenderEntryFn)(const FjAssets*, const FjPreparedState*, FjCudaCallAccess, const FjFrame*, const FjPreparedFramePolicy*, FjSubmissionAccess, FjAbortCallback, FjErrorBuffer*);
typedef FjCallCompletionResult (*FjUnavailableEntryFn)(const FjAssets*, const FjPreparedState*, uint32_t, FjCudaCallAccess, const FjFrame*, FjAbortCallback, FjErrorBuffer*);
typedef FjStatus (*FjHostPreparationScopeFn)(void*, FjHostPreparationConsumer, FjErrorBuffer*);

_Static_assert(_Generic(((FjCudaCallAccess*)0)->precheck, FjCallPrecheckFn: 1, default: 0), "precheck callback signature");
_Static_assert(_Generic(((FjCudaCallAccess*)0)->admit, FjCallAdmitFn: 1, default: 0), "admit callback signature");
_Static_assert(_Generic(((FjCudaCallAccess*)0)->release, FjCallReleaseFn: 1, default: 0), "release callback signature");
_Static_assert(_Generic(((FjCudaCallAccess*)0)->inspect, FjCallInspectFn: 1, default: 0), "inspect callback signature");
_Static_assert(_Generic(((FjCudaCallAccess*)0)->preview, FjCallPreviewFn: 1, default: 0), "preview callback signature");
_Static_assert(_Generic(((FjCudaCallAccess*)0)->frame_descriptors, FjCallFrameDescriptorsFn: 1, default: 0), "frame descriptor callback signature");
_Static_assert(_Generic(((FjCudaCallAccess*)0)->render, FjCallRenderFn: 1, default: 0), "render callback signature");
_Static_assert(_Generic(((FjCudaCallAccess*)0)->complete, FjCallCompleteFn: 1, default: 0), "complete callback signature");
_Static_assert(_Generic(((FjHostPreparation*)0)->scanner_lut, FjHostScannerLutFn: 1, default: 0), "scanner LUT callback signature");
_Static_assert(_Generic(((FjHostPreparation*)0)->gaussian, FjHostGaussianFn: 1, default: 0), "Gaussian callback signature");
_Static_assert(_Generic(((FjSubmissionAccess*)0)->select, FjSubmissionSelectFn: 1, default: 0), "submission selector signature");
_Static_assert(_Generic(&fj_legacy_prepared_state_frame_policy, FjPolicyEntryFn: 1, default: 0), "frame policy entry signature");
_Static_assert(_Generic(&fj_legacy_prepared_state_render, FjRenderEntryFn: 1, default: 0), "prepared render entry signature");
_Static_assert(_Generic(&fj_legacy_prepared_unavailable, FjUnavailableEntryFn: 1, default: 0), "unavailable entry signature");
_Static_assert(_Generic(&fj_test_with_host_preparation, FjHostPreparationScopeFn: 1, default: 0), "scoped preparation facade signature");

static FjStatus consume_host_preparation(void* user, const FjHostPreparation* preparation, FjErrorBuffer* error) {
    size_t* calls = (size_t*)user;
    ++*calls;
    if (!preparation || !preparation->user || !preparation->gaussian || !preparation->scanner_lut)
        return (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    const FjGaussianKernelInput input = {3, 1.0f};
    FjGaussianKernelView view = {0};
    FjGaussianKernelFailure failure = {0};
    const FjStatus status = preparation->gaussian(preparation->user, &input, &view, &failure, error);
    if (status.category != FJ_STATUS_SUCCESS || failure.reason || view.radius != 3 ||
        view.sigma_px != 1.0f || !view.weights.data || view.weights.count != 7)
        return (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    float retained[7];
    for (size_t i = 0; i < 7; ++i)
        retained[i] = view.weights.data[i];
    const FjGaussianKernelInput next_input = {2, 2.0f};
    FjGaussianKernelView next_view = {0};
    if (preparation->gaussian(preparation->user, &next_input, &next_view, &failure, error).category != FJ_STATUS_SUCCESS ||
        next_view.weights.count != 5 || fj_test_gaussian_kernel_live_owners() < 2)
        return (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    for (size_t i = 0; i < 7; ++i)
        if (retained[i] != view.weights.data[i])
            return (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    if (fj_test_gaussian_kernel_fault(0, 1).category != FJ_STATUS_SUCCESS ||
        preparation->gaussian(preparation->user, &input, NULL, &failure, error).category != FJ_STATUS_UNSUPPORTED_INPUT ||
        failure.reason != FJ_GAUSSIAN_KERNEL_RAW)
        return (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    if (preparation->gaussian(preparation->user, &input, &next_view, &failure, error).category != FJ_STATUS_SUCCESS)
        return (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    FjScannerLutFailure scanner_failure = {0};
    FjScannerLutView scanner_view = {0};
    if (fj_test_scanner_lut_fault(0, 1).category != FJ_STATUS_SUCCESS ||
        preparation->scanner_lut(preparation->user, NULL, NULL, &scanner_failure, error).category != FJ_STATUS_UNSUPPORTED_INPUT ||
        scanner_failure.reason != FJ_SCANNER_LUT_REASON_RAW)
        return (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    if (preparation->scanner_lut(preparation->user, NULL, &scanner_view, &scanner_failure, error).category != FJ_STATUS_UNSUPPORTED_INPUT)
        return (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    return status;
}

int fj_test_host_preparation_c_call(void) {
    size_t calls = 0;
    if (fj_test_with_host_preparation(&calls, consume_host_preparation, NULL).category != FJ_STATUS_SUCCESS || calls != 1)
        return 0;
    if (fj_test_with_host_preparation(NULL, consume_host_preparation, NULL).category != FJ_STATUS_UNSUPPORTED_INPUT || calls != 1)
        return 0;
    if (fj_test_with_host_preparation(&calls, NULL, NULL).category != FJ_STATUS_UNSUPPORTED_INPUT || calls != 1)
        return 0;
    FjErrorBuffer malformed = {NULL, 1, 99};
    return fj_test_with_host_preparation(&calls, consume_host_preparation, &malformed).category == FJ_STATUS_UNSUPPORTED_INPUT &&
           malformed.length == 0 && calls == 1;
}

const size_t* fj_test_prepared_call_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), _Alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#define FJ_ABI_VALUE(tag, value) tag,
#include "prepared_call_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}

const size_t* fj_test_prepared_state_c_facts(size_t* count) {
    static const size_t facts[] = {
#define FJ_PREPARED_TYPE(type) sizeof(type), _Alignof(type),
#define FJ_PREPARED_FIELD(type, field) offsetof(type, field),
#include "prepared_state_abi_facts.inc"
#undef FJ_PREPARED_TYPE
#undef FJ_PREPARED_FIELD
    };
    *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}

FjStatus fj_test_prepared_state_create_c(FjAssets* assets, const FjPreparationParams* params, FjPreparedState** out, FjErrorBuffer* error) {
    FjStatus (*call)(FjAssets*, const FjPreparationParams*, FjPreparedState**, FjErrorBuffer*) = fj_legacy_prepared_state_create;
    return call(assets, params, out, error);
}
FjStatus fj_test_prepared_state_view_c(const FjPreparedState* state, FjPreparedStateView* out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjPreparedState*, FjPreparedStateView*, FjErrorBuffer*) = fj_legacy_prepared_state_view;
    return call(state, out, error);
}
FjStatus fj_test_prepared_state_release_c(FjPreparedState* state, FjErrorBuffer* error) {
    FjStatus (*call)(FjPreparedState*, FjErrorBuffer*) = fj_legacy_prepared_state_release;
    return call(state, error);
}

FjStatus fj_test_prepared_illuminants_acquire_c(const FjAssets* assets, FjIlluminantsOwner** out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjAssets*, FjIlluminantsOwner**, FjErrorBuffer*) = fj_legacy_assets_illuminants_acquire;
    return call(assets, out, error);
}
FjStatus fj_test_prepared_illuminants_view_c(const FjIlluminantsOwner* owner, FjIlluminantsView* out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjIlluminantsOwner*, FjIlluminantsView*, FjErrorBuffer*) = fj_legacy_illuminants_view;
    return call(owner, out, error);
}
FjStatus fj_test_prepared_illuminants_release_c(FjIlluminantsOwner* owner, FjErrorBuffer* error) {
    FjStatus (*call)(FjIlluminantsOwner*, FjErrorBuffer*) = fj_legacy_illuminants_release;
    return call(owner, error);
}
FjStatus fj_test_prepared_hull_acquire_c(const FjAssets* assets, FjInputHullOwner** out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjAssets*, FjInputHullOwner**, FjErrorBuffer*) = fj_legacy_assets_input_hull_acquire;
    return call(assets, out, error);
}
FjStatus fj_test_prepared_hull_view_c(const FjInputHullOwner* owner, FjInputHullView* out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjInputHullOwner*, FjInputHullView*, FjErrorBuffer*) = fj_legacy_input_hull_view;
    return call(owner, out, error);
}
FjStatus fj_test_prepared_hull_release_c(FjInputHullOwner* owner, FjErrorBuffer* error) {
    FjStatus (*call)(FjInputHullOwner*, FjErrorBuffer*) = fj_legacy_input_hull_release;
    return call(owner, error);
}
FjStatus fj_test_prepared_boundary_acquire_c(const FjAssets* assets, const FjOutputBoundarySource* source, uint64_t expected, FjOutputBoundaryOwner** out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjAssets*, const FjOutputBoundarySource*, uint64_t, FjOutputBoundaryOwner**, FjErrorBuffer*) = fj_legacy_assets_output_boundary_acquire;
    return call(assets, source, expected, out, error);
}
FjStatus fj_test_prepared_boundary_view_c(const FjOutputBoundaryOwner* owner, FjOutputBoundaryView* out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjOutputBoundaryOwner*, FjOutputBoundaryView*, FjErrorBuffer*) = fj_legacy_output_boundary_view;
    return call(owner, out, error);
}
FjStatus fj_test_prepared_boundary_release_c(FjOutputBoundaryOwner* owner, FjErrorBuffer* error) {
    FjStatus (*call)(FjOutputBoundaryOwner*, FjErrorBuffer*) = fj_legacy_output_boundary_release;
    return call(owner, error);
}
