#include "runtime_c.h"

FjStatus fj_test_runtime_c_consumer(FjPathView resource_dir) {
    FjRuntime* runtime = NULL;
    FjStatus result = fj_legacy_runtime_create(resource_dir, &runtime, NULL);
    if (result.category != FJ_STATUS_SUCCESS || !runtime) {
        return result.category == FJ_STATUS_SUCCESS ? (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0} : result;
    }

    FjRuntimeView first = {0};
    FjRuntimeView second = {0};
    result = fj_legacy_runtime_borrow(runtime, &first, NULL);
    if (result.category == FJ_STATUS_SUCCESS) {
        result = fj_legacy_runtime_borrow(runtime, &second, NULL);
    }
    if (result.category == FJ_STATUS_SUCCESS &&
        (!first.assets || !first.cuda || first.assets != second.assets || first.cuda != second.cuda)) {
        result = (FjStatus){FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
    }
    if (result.category == FJ_STATUS_SUCCESS) {
        result = fj_legacy_runtime_check_close(runtime, NULL);
    }
    if (result.category == FJ_STATUS_SUCCESS) {
        result = fj_legacy_runtime_shutdown(runtime, NULL);
    }

    FjRuntime* consumed = runtime;
    runtime = NULL;
    const FjStatus destroyed = fj_legacy_runtime_destroy(consumed, NULL);
    if (result.category == FJ_STATUS_SUCCESS) {
        result = destroyed;
    }
    return result;
}
