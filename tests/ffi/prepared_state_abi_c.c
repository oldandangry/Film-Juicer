#include <stddef.h>
#include <stdint.h>

#include "juicer_test_api.h"
#include "juicer_cuda_call_scope.h"

typedef FjStatus (*FjInstanceCreateFn)(FjInstance**, FjErrorBuffer*);
typedef FjStatus (*FjBeginCaptureFn)(FjInstance*, uint32_t, FjCaptureStart*, FjErrorBuffer*);
typedef FjStatus (*FjCaptureCurrentFn)(const FjInstance*, uint64_t, uint32_t*, FjErrorBuffer*);
typedef FjStatus (*FjBeginHostWriteFn)(FjInstance*, uint64_t, FjHostWrite**, uint32_t*, FjErrorBuffer*);
typedef FjStatus (*FjHostWriteFinishFn)(FjHostWrite*, uint32_t, FjStringView, uint32_t*, FjErrorBuffer*);
typedef FjStatus (*FjBeginSnapshotFn)(FjInstance*, uint64_t, FjSnapshotStart*, FjErrorBuffer*);
typedef FjStatus (*FjInstallFn)(FjInstance*, uint64_t, uint64_t, const FjPreparationParams*, uint64_t, uint32_t*, FjErrorBuffer*);
typedef FjStatus (*FjInvalidateFn)(FjInstance*, uint64_t, uint32_t, FjStringView, uint32_t*, FjErrorBuffer*);
typedef FjStatus (*FjAdmitFn)(FjInstance*, const FjAssets*, FjAdmission*, FjErrorBuffer*);
typedef FjStatus (*FjInstanceDestroyFn)(FjInstance*, FjErrorBuffer*);

_Static_assert(_Generic(&fj_legacy_instance_create, FjInstanceCreateFn: 1, default: 0), "instance create signature");
_Static_assert(_Generic(&fj_legacy_instance_begin_capture, FjBeginCaptureFn: 1, default: 0), "begin capture signature");
_Static_assert(_Generic(&fj_legacy_instance_capture_is_current, FjCaptureCurrentFn: 1, default: 0), "capture current signature");
_Static_assert(_Generic(&fj_legacy_instance_begin_host_write, FjBeginHostWriteFn: 1, default: 0), "begin host write signature");
_Static_assert(_Generic(&fj_legacy_host_write_finish, FjHostWriteFinishFn: 1, default: 0), "host write finish signature");
_Static_assert(_Generic(&fj_legacy_instance_begin_snapshot, FjBeginSnapshotFn: 1, default: 0), "begin snapshot signature");
_Static_assert(_Generic(&fj_legacy_instance_install, FjInstallFn: 1, default: 0), "install signature");
_Static_assert(_Generic(&fj_legacy_instance_invalidate, FjInvalidateFn: 1, default: 0), "invalidate signature");
_Static_assert(_Generic(&fj_legacy_instance_admit, FjAdmitFn: 1, default: 0), "instance admit signature");
_Static_assert(_Generic(&fj_legacy_instance_destroy, FjInstanceDestroyFn: 1, default: 0), "instance destroy signature");

_Static_assert(_Generic(((FjCaptureStart*)0)->capture, uint64_t: 1, default: 0), "capture ticket width");
_Static_assert(_Generic(((FjCaptureStart*)0)->kind, uint32_t: 1, default: 0), "capture tag width");
_Static_assert(_Generic(((FjSnapshotStart*)0)->write_revision, uint64_t: 1, default: 0), "snapshot revision width");
_Static_assert(_Generic(((FjSnapshotStart*)0)->kind, uint32_t: 1, default: 0), "snapshot tag width");
_Static_assert(_Generic(((FjAdmission*)0)->state, FjPreparedState*: 1, default: 0), "admission owner type");
_Static_assert(_Generic(((FjAdmission*)0)->input_identity, uint64_t: 1, default: 0), "admission identity width");
_Static_assert(_Generic(((FjAdmission*)0)->build, uint64_t: 1, default: 0), "admission build width");
_Static_assert(_Generic(((FjAdmission*)0)->kind, uint32_t: 1, default: 0), "admission tag width");
_Static_assert(_Generic(((FjAdmission*)0)->route, uint32_t: 1, default: 0), "admission route width");

_Static_assert(FJ_CAPTURE_INITIAL == 0 && FJ_CAPTURE_CHANGE == 1 && FJ_CAPTURE_REFRESH == 2, "capture mode tags");
_Static_assert(FJ_CAPTURE_START_NONE == 0 && FJ_CAPTURE_STARTED == 1 && FJ_CAPTURE_NOT_STARTED == 2, "capture result tags");
_Static_assert(FJ_HOST_WRITE_NONE == 0 && FJ_HOST_WRITE_STARTED == 1 && FJ_HOST_WRITE_IGNORED == 2 && FJ_HOST_WRITE_SUCCEEDED == 0, "host write tags");
_Static_assert(FJ_SNAPSHOT_NONE == 0 && FJ_SNAPSHOT_READY == 1 && FJ_SNAPSHOT_DEFERRED == 2 && FJ_SNAPSHOT_IGNORED == 3, "snapshot tags");
_Static_assert(FJ_UPDATE_NONE == 0 && FJ_UPDATE_APPLIED == 1 && FJ_UPDATE_IGNORED == 2 && FJ_UPDATE_DEFERRED == 3, "update tags");
_Static_assert(FJ_CAPTURE_INVALID_CONTROLS == 1 && FJ_CAPTURE_HOST_FAILURE == 2 && FJ_CAPTURE_ALLOCATION_FAILURE == 3, "capture failure tags");
_Static_assert(FJ_ADMISSION_NONE == 0 && FJ_ADMISSION_READY == 1 && FJ_ADMISSION_NEEDS_CAPTURE == 2 && FJ_ADMISSION_CAPTURING == 3 && FJ_ADMISSION_CAPTURE_FAILED == 4 && FJ_ADMISSION_PREPARATION_FAILED == 5 && FJ_ADMISSION_POISONED == 6 && FJ_ADMISSION_NEEDS_REFRESH == 7, "admission tags");

_Static_assert(sizeof(FjCaptureStart) == 16 && _Alignof(FjCaptureStart) == 8, "capture start layout");
_Static_assert(offsetof(FjCaptureStart, capture) == 0 && offsetof(FjCaptureStart, kind) == 8, "capture start fields");
_Static_assert(sizeof(FjSnapshotStart) == 16 && _Alignof(FjSnapshotStart) == 8, "snapshot start layout");
_Static_assert(offsetof(FjSnapshotStart, write_revision) == 0 && offsetof(FjSnapshotStart, kind) == 8, "snapshot start fields");
_Static_assert(sizeof(FjAdmission) == 32 && _Alignof(FjAdmission) == 8, "admission layout");
_Static_assert(offsetof(FjAdmission, state) == 0 && offsetof(FjAdmission, input_identity) == 8 &&
                   offsetof(FjAdmission, build) == 16 && offsetof(FjAdmission, kind) == 24 &&
                   offsetof(FjAdmission, route) == 28,
               "admission fields");

const size_t* fj_test_instance_c_facts(size_t* count) {
    static const size_t facts[] = {
        sizeof(FjCaptureStart), _Alignof(FjCaptureStart), offsetof(FjCaptureStart, capture), offsetof(FjCaptureStart, kind), sizeof(FjSnapshotStart), _Alignof(FjSnapshotStart), offsetof(FjSnapshotStart, write_revision), offsetof(FjSnapshotStart, kind), sizeof(FjAdmission), _Alignof(FjAdmission), offsetof(FjAdmission, state), offsetof(FjAdmission, input_identity), offsetof(FjAdmission, build), offsetof(FjAdmission, kind), offsetof(FjAdmission, route), FJ_CAPTURE_INITIAL, FJ_CAPTURE_CHANGE, FJ_CAPTURE_REFRESH, FJ_CAPTURE_START_NONE, FJ_CAPTURE_STARTED, FJ_CAPTURE_NOT_STARTED, FJ_HOST_WRITE_NONE, FJ_HOST_WRITE_STARTED, FJ_HOST_WRITE_IGNORED, FJ_HOST_WRITE_SUCCEEDED, FJ_SNAPSHOT_NONE, FJ_SNAPSHOT_READY, FJ_SNAPSHOT_DEFERRED, FJ_SNAPSHOT_IGNORED, FJ_UPDATE_NONE, FJ_UPDATE_APPLIED, FJ_UPDATE_IGNORED, FJ_UPDATE_DEFERRED, FJ_CAPTURE_INVALID_CONTROLS, FJ_CAPTURE_HOST_FAILURE, FJ_CAPTURE_ALLOCATION_FAILURE, FJ_ADMISSION_NONE, FJ_ADMISSION_READY, FJ_ADMISSION_NEEDS_CAPTURE, FJ_ADMISSION_CAPTURING, FJ_ADMISSION_CAPTURE_FAILED, FJ_ADMISSION_PREPARATION_FAILED, FJ_ADMISSION_POISONED, FJ_ADMISSION_NEEDS_REFRESH};
    if (count)
        *count = sizeof(facts) / sizeof(facts[0]);
    return facts;
}

/* Returns zero on success; nonzero values identify the failed protocol row. */
uint32_t fj_test_instance_c_witness(const FjAssets* assets, const FjPreparationParams* params, uint32_t scenario) {
    FjInstance* instance = NULL;
    FjHostWrite* write = NULL;
    FjPreparedState* prepared = NULL;
    uint32_t failure = 0;
    uint32_t update = FJ_UPDATE_NONE;
    uint32_t current = 0;
    uint32_t start = FJ_HOST_WRITE_NONE;
    uint32_t refresh = 0;
    FjHostWrite* rejected_write = NULL;
    FjCaptureStart capture = {0, FJ_CAPTURE_START_NONE};
    FjCaptureStart changed = {0, FJ_CAPTURE_START_NONE};
    FjCaptureStart replacement = {0, FJ_CAPTURE_START_NONE};
    FjSnapshotStart snapshot = {0, FJ_SNAPSHOT_NONE};
    FjAdmission admission = {0};
    FjErrorBuffer error = {NULL, 0, 0};
    FjStringView empty = {NULL, 0};
    static const char host_failure[] = "C11 host capture failed";
    static const char malformed_failure[] = "malformed host-write completion";
    FjStringView host_diagnostic = {host_failure, sizeof(host_failure) - 1};
    FjStringView malformed_diagnostic = {malformed_failure, sizeof(malformed_failure) - 1};
    FjStatus status;
    uint64_t ticket = 0;
    uint64_t revision = 0;

#define FJ_INSTANCE_CHECK(expression, code) \
    do {                                    \
        if (!(expression)) {                \
            failure = (code);               \
            goto cleanup;                   \
        }                                   \
    } while (0)

    status = fj_legacy_instance_create(&instance, &error);
    FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && instance != NULL, 1);
    status = fj_legacy_instance_admit(instance, assets, &admission, &error);
    FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_NEEDS_CAPTURE &&
                          admission.state == NULL && admission.input_identity == 0 && admission.build == 0 && admission.route == 0,
                      2);

    if (scenario == 0) {
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_REFRESH, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_NOT_STARTED && capture.capture == 0, 28);
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED && capture.capture != 0, 3);
        ticket = capture.capture;
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &changed, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && changed.kind == FJ_CAPTURE_NOT_STARTED && changed.capture == 0, 4);
        status = fj_legacy_instance_capture_is_current(instance, ticket + 1, &current, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && current == 0, 5);
        status = fj_legacy_instance_capture_is_current(instance, ticket, &current, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && current == 1, 25);
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_CHANGE, &changed, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && changed.kind == FJ_CAPTURE_STARTED && changed.capture > ticket, 6);
        status = fj_legacy_instance_capture_is_current(instance, ticket, &current, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && current == 0, 7);
        ticket = changed.capture;

        status = fj_legacy_instance_begin_host_write(instance, ticket, &write, &start, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && start == FJ_HOST_WRITE_STARTED && write != NULL, 8);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_PREPARATION_FAILURE && admission.kind == FJ_ADMISSION_CAPTURING && admission.state == NULL, 26);
        status = fj_legacy_instance_begin_snapshot(instance, ticket, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_DEFERRED && snapshot.write_revision == 0, 9);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_PREPARATION_FAILURE && admission.kind == FJ_ADMISSION_CAPTURING && admission.state == NULL, 27);
        status = fj_legacy_instance_begin_host_write(instance, ticket, &rejected_write, &start, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && start == FJ_HOST_WRITE_IGNORED && rejected_write == NULL, 10);
        status = fj_legacy_host_write_finish(write, FJ_HOST_WRITE_SUCCEEDED, empty, &refresh, &error);
        write = NULL;
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && refresh == 1, 11);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_NEEDS_REFRESH && admission.state == NULL, 12);
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_REFRESH, &replacement, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && replacement.kind == FJ_CAPTURE_STARTED && replacement.capture > ticket, 13);
        ticket = replacement.capture;
        status = fj_legacy_instance_begin_host_write(instance, ticket, &write, &start, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && start == FJ_HOST_WRITE_NONE && write == NULL, 14);
        status = fj_legacy_instance_begin_snapshot(instance, ticket, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_READY, 15);
        revision = snapshot.write_revision;
        status = fj_legacy_instance_begin_snapshot(instance, ticket, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && snapshot.kind == FJ_SNAPSHOT_NONE && snapshot.write_revision == 0, 16);
        status = fj_legacy_instance_begin_host_write(instance, ticket, &write, &start, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && start == FJ_HOST_WRITE_NONE && write == NULL, 17);
        status = fj_legacy_instance_install(instance, ticket, revision + 1, params, 0, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && update == FJ_UPDATE_NONE, 18);
        status = fj_legacy_instance_install(instance, ticket, revision, params, 0, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 19);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_READY && admission.state != NULL &&
                              admission.input_identity == 0 && admission.build == 1 && admission.route == FJ_LEGACY_ROUTE_NEGATIVE_DIRECT,
                          20);
        prepared = admission.state;
        admission.state = NULL;
        status = fj_legacy_instance_destroy(instance, &error);
        instance = NULL;
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS, 21);
        status = fj_legacy_instance_create(&instance, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && instance != NULL, 22);
        status = fj_legacy_instance_destroy(instance, &error);
        instance = NULL;
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS, 23);
        {
            FjPreparedStateView view = {0};
            status = fj_legacy_prepared_state_view(prepared, &view, &error);
            FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && view.recipe_hash != 0, 24);
        }
    } else if (scenario == 1) {
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 31);
        status = fj_legacy_instance_invalidate(instance, capture.capture, FJ_CAPTURE_HOST_FAILURE, host_diagnostic, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 32);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_PREPARATION_FAILURE && admission.kind == FJ_ADMISSION_CAPTURE_FAILED && admission.state == NULL, 33);
    } else if (scenario == 2) {
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 41);
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_CHANGE, &changed, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && changed.kind == FJ_CAPTURE_STARTED, 42);
        status = fj_legacy_instance_invalidate(instance, capture.capture, FJ_CAPTURE_HOST_FAILURE, host_diagnostic, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_IGNORED, 43);
        snapshot.write_revision = UINT64_C(99);
        snapshot.kind = UINT32_C(99);
        status = fj_legacy_instance_begin_snapshot(instance, capture.capture, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_IGNORED && snapshot.write_revision == 0, 47);
        status = fj_legacy_instance_begin_snapshot(instance, changed.capture, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_READY, 44);
        status = fj_legacy_instance_install(instance, changed.capture, snapshot.write_revision, params, 7, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 45);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_READY &&
                              admission.input_identity == 7 && admission.state != NULL,
                          46);
        prepared = admission.state;
        admission.state = NULL;
    } else if (scenario == 3) {
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 51);
        status = fj_legacy_instance_begin_host_write(instance, capture.capture, &write, &start, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && start == FJ_HOST_WRITE_STARTED && write != NULL, 52);
        status = fj_legacy_host_write_finish(write, UINT32_C(99), malformed_diagnostic, &refresh, &error);
        write = NULL; /* The consuming call owns/frees this pointer for every result. */
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && refresh == 0, 53);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_PREPARATION_FAILURE && admission.kind == FJ_ADMISSION_CAPTURE_FAILED && admission.state == NULL, 54);
    } else if (scenario == 4) {
        capture.capture = UINT64_C(99);
        capture.kind = UINT32_C(99);
        status = fj_legacy_instance_begin_capture(instance, UINT32_C(99), &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && capture.capture == 0 && capture.kind == FJ_CAPTURE_START_NONE, 61);
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 62);
        ticket = capture.capture;
        current = UINT32_C(99);
        status = fj_legacy_instance_capture_is_current(instance, ticket + 1, &current, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && current == 0, 63);
        snapshot.write_revision = UINT64_C(99);
        snapshot.kind = UINT32_C(99);
        status = fj_legacy_instance_begin_snapshot(instance, ticket + 100, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && snapshot.write_revision == 0 && snapshot.kind == FJ_SNAPSHOT_NONE, 64);
        status = fj_legacy_instance_begin_snapshot(instance, ticket, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_READY, 65);
        revision = snapshot.write_revision;
        update = UINT32_C(99);
        status = fj_legacy_instance_install(instance, ticket, revision + 1, params, 0, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && update == FJ_UPDATE_NONE, 66);
        status = fj_legacy_instance_install(instance, ticket, revision, params, 0, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 67);
        admission.state = (FjPreparedState*)(uintptr_t)1;
        admission.input_identity = UINT64_C(99);
        admission.build = UINT64_C(99);
        admission.kind = UINT32_C(99);
        admission.route = UINT32_C(99);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_READY && admission.state != NULL && admission.input_identity == 0 && admission.build == 1, 68);
    } else if (scenario == 5) {
        FjPreparationParams malformed = *params;
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 71);
        ticket = capture.capture;
        status = fj_legacy_instance_begin_snapshot(instance, ticket, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_READY, 72);
        malformed.scan_route = UINT32_C(99);
        status = fj_legacy_instance_install(instance, ticket, snapshot.write_revision, &malformed, UINT64_C(71), &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && update == FJ_UPDATE_APPLIED, 73);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_PREPARATION_FAILURE && admission.kind == FJ_ADMISSION_CAPTURE_FAILED && admission.state == NULL, 74);
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_CHANGE, &changed, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && changed.kind == FJ_CAPTURE_STARTED, 75);
        status = fj_legacy_instance_install(instance, ticket, snapshot.write_revision, &malformed, UINT64_C(72), &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_IGNORED, 76);
        status = fj_legacy_instance_begin_snapshot(instance, changed.capture, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_READY, 77);
        status = fj_legacy_instance_install(instance, changed.capture, snapshot.write_revision, params, UINT64_C(73), &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 78);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_READY && admission.input_identity == UINT64_C(73), 79);
    } else if (scenario == 6) {
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 81);
        status = fj_legacy_instance_invalidate(instance, capture.capture, FJ_CAPTURE_INVALID_CONTROLS, empty, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 82);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_PREPARATION_FAILURE && admission.kind == FJ_ADMISSION_CAPTURE_FAILED && admission.state == NULL, 83);
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_CHANGE, &changed, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && changed.kind == FJ_CAPTURE_STARTED, 84);
        status = fj_legacy_instance_invalidate(instance, changed.capture, FJ_CAPTURE_ALLOCATION_FAILURE, empty, &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 85);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_ALLOCATION_FAILURE && admission.kind == FJ_ADMISSION_CAPTURE_FAILED && admission.state == NULL, 86);
    } else if (scenario == 7) {
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 91);
        status = fj_legacy_instance_begin_snapshot(instance, capture.capture, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_READY, 92);
        status = fj_legacy_instance_install(instance, capture.capture, snapshot.write_revision, params, UINT64_C(91), &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 93);
        fj_test_prepared_contents_reservation_failure();
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_ALLOCATION_FAILURE && admission.kind == FJ_ADMISSION_PREPARATION_FAILED && admission.state == NULL, 94);
        FJ_INSTANCE_CHECK(fj_test_prepared_contents_reservation_failure_consumed() == 1, 95);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && admission.kind == FJ_ADMISSION_READY && admission.state != NULL && admission.input_identity == UINT64_C(91) && admission.build == 1, 96);
    } else if (scenario == 8) {
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 101);
        status = fj_legacy_instance_begin_snapshot(instance, capture.capture, &snapshot, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && snapshot.kind == FJ_SNAPSHOT_READY, 102);
        status = fj_legacy_instance_install(instance, capture.capture, snapshot.write_revision, params, UINT64_C(101), &update, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && update == FJ_UPDATE_APPLIED, 103);
        FJ_INSTANCE_CHECK(fj_test_spectral_arm_fault(3, 1, 2).category == FJ_STATUS_SUCCESS, 106);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_INTERNAL_FAILURE && admission.kind == FJ_ADMISSION_POISONED && admission.state == NULL, 104);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_INTERNAL_FAILURE && admission.kind == FJ_ADMISSION_POISONED && admission.state == NULL, 105);
    } else if (scenario == 9) {
        static const char invalid_utf8[] = {(char)0xff};
        FjStringView malformed_text = {invalid_utf8, sizeof(invalid_utf8)};
        status = fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && capture.kind == FJ_CAPTURE_STARTED, 111);
        status = fj_legacy_instance_begin_host_write(instance, capture.capture, &write, &start, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_SUCCESS && start == FJ_HOST_WRITE_STARTED && write != NULL, 112);
        status = fj_legacy_host_write_finish(write, FJ_CAPTURE_HOST_FAILURE, malformed_text, &refresh, &error);
        write = NULL; /* Invalid diagnostics still consume and fail the safe guard as Host. */
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_UNSUPPORTED_INPUT && refresh == 0, 113);
        status = fj_legacy_instance_admit(instance, assets, &admission, &error);
        FJ_INSTANCE_CHECK(status.category == FJ_STATUS_PREPARATION_FAILURE && admission.kind == FJ_ADMISSION_CAPTURE_FAILED && admission.state == NULL, 114);
    } else {
        failure = 90;
    }

cleanup:
    if (write) {
        (void)fj_legacy_host_write_finish(write, FJ_CAPTURE_HOST_FAILURE, host_diagnostic, &refresh, &error);
        write = NULL;
    }
    if (rejected_write)
        (void)fj_legacy_host_write_finish(rejected_write, FJ_CAPTURE_HOST_FAILURE, host_diagnostic, &refresh, &error);
    if (admission.state)
        (void)fj_legacy_prepared_state_release(admission.state, &error);
    if (prepared)
        (void)fj_legacy_prepared_state_release(prepared, &error);
    if (instance)
        (void)fj_legacy_instance_destroy(instance, &error);
#undef FJ_INSTANCE_CHECK
    return failure;
}

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

FjStatus fj_test_prepared_state_create_c(const FjAssets* assets, const FjPreparationParams* params, FjPreparedState** out, FjErrorBuffer* error) {
    FjStatus (*call)(const FjAssets*, const FjPreparationParams*, FjPreparedState**, FjErrorBuffer*) = fj_legacy_prepared_state_create;
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
