#include <stddef.h>

#include "juicer_test_api.h"

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
