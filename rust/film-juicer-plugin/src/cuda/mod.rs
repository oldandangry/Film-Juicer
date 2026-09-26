//! Raw CUDA declarations and their bounded ABI evidence.

#[allow(
    unsafe_code,
    dead_code,
    clippy::allow_attributes_without_reason,
    reason = "bindgen emits foreign declarations, layout assertions with generated lint attributes, and records exercised by ABI tests before production cutover"
)]
mod sys;

#[cfg(any(test, feature = "test-support"))]
#[allow(
    unsafe_code,
    reason = "the CMake-linked ABI fixture exports a borrowed static layout table and checks raw signatures"
)]
mod abi_tests;
