//! Raw CUDA declarations and their bounded ABI evidence.

#[allow(
    unsafe_code,
    dead_code,
    clippy::allow_attributes_without_reason,
    reason = "bindgen emits foreign declarations, layout assertions with generated lint attributes, and records exercised by ABI tests before production cutover"
)]
pub(super) mod sys;

#[allow(
    unsafe_code,
    reason = "the held-render edge admits foreign storage and dispatches synchronous scoped callbacks"
)]
pub(crate) mod edge;

mod frame;
mod frame_policy;
#[allow(
    unsafe_code,
    reason = "call-local scanner and Gaussian candidates back the synchronous native preparation callbacks"
)]
mod host_preparation;
mod projection;
mod render;

pub(crate) use edge::{
    Abort, Frame, FramePolicy, OwnerBorrow, RenderError, RenderOutcome, SubmissionLatch,
};
pub(crate) use render::render;

#[cfg(feature = "test-support")]
pub(crate) use render::{
    NoiseCallObservation, noise_call_observation, reset_noise_call_observation,
    set_short_noise_view, short_noise_view_consumed,
};

#[cfg(any(test, feature = "test-support"))]
#[allow(
    unsafe_code,
    reason = "the CMake-linked ABI fixture exports a borrowed static layout table and checks raw signatures"
)]
mod abi_tests;
