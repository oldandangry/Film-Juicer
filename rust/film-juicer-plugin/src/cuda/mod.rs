//! CUDA status values, foreign declarations and their bounded ABI evidence.

use std::fmt::{self, Write};

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

#[allow(
    unsafe_code,
    reason = "the sole native owning handle confines construction, terminal FFI and justified thread traits to this edge"
)]
pub(crate) mod owner;
pub(crate) use owner::Owner;

#[derive(Clone, Copy)]
pub(crate) struct Failure {
    pub(crate) status: sys::FjStatus,
    diagnostic: [u8; 1024],
    length: usize,
}

impl Failure {
    pub(crate) fn new(category: u32, message: impl fmt::Display) -> Self {
        let mut failure = Self {
            status: sys::FjStatus {
                category,
                api: sys::FJ_API_NONE,
                native_code: 0,
            },
            diagnostic: [0; 1024],
            length: 0,
        };
        let _ = write!(failure, "{message}");
        failure
    }

    pub(crate) fn diagnostic(&self) -> &str {
        // Formatting preserves UTF-8; native diagnostics stay raw bytes at the C edge.
        std::str::from_utf8(&self.diagnostic[..self.length])
            .unwrap_or("native terminal diagnostic is not UTF-8")
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.diagnostic[..self.length]
    }
}

impl Write for Failure {
    fn write_str(&mut self, message: &str) -> fmt::Result {
        let mut count = message.len().min(1023 - self.length);
        while !message.is_char_boundary(count) {
            count -= 1;
        }
        self.diagnostic[self.length..self.length + count]
            .copy_from_slice(&message.as_bytes()[..count]);
        self.length += count;
        Ok(())
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.diagnostic())
    }
}

impl fmt::Debug for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Failure")
            .field("category", &self.status.category)
            .field("api", &self.status.api)
            .field("native_code", &self.status.native_code)
            .field("diagnostic", &self.diagnostic())
            .finish()
    }
}

pub(crate) use edge::{Abort, Frame, FramePolicy, OwnerBorrow, RenderError, RenderOutcome};
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
