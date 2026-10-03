//! Film-Juicer's Rust plug-in integration boundary.

#![deny(unsafe_code)]

#[allow(
    unsafe_code,
    reason = "the temporary C ABI validates its foreign input at this boundary"
)]
mod legacy_bridge;

mod cuda;

mod asset_catalog;

#[allow(
    unsafe_code,
    reason = "the private production asset edge checks foreign extents and owns consume-once handles"
)]
mod asset_bridge;

mod asset_profile;

#[cfg(feature = "test-support")]
#[allow(
    unsafe_code,
    reason = "the test facade checks foreign inputs and consumes retained profile owners at the C boundary"
)]
mod test_support;
