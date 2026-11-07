//! Exact vendored OpenFX declarations and their native ABI fixture.

#[allow(
    unsafe_code,
    dead_code,
    nonstandard_style,
    clippy::allow_attributes_without_reason,
    reason = "bindgen preserves foreign declarations and emits layout assertions with generated lint attributes; complete suites precede host-adapter consumers"
)]
pub(super) mod sys;

#[cfg(feature = "test-support")]
#[allow(
    unsafe_code,
    reason = "the native ABI fixture borrows immutable fact tables and exercises synchronous foreign callbacks and variadic fields"
)]
mod abi_tests;
