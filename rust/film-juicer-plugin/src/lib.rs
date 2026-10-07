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

mod asset_spectral;

mod asset_noise;

mod asset_calibration;

#[allow(
    unsafe_code,
    reason = "the illuminant C edge validates spans and consumes call-local lens owners"
)]
mod asset_illuminant;

#[cfg(feature = "test-support")]
#[allow(
    unsafe_code,
    reason = "the test facade checks foreign inputs and consumes retained profile owners at the C boundary"
)]
mod test_support;

#[allow(
    unsafe_code,
    reason = "the fixed spectral edge checks foreign extents and scopes immutable borrows"
)]
mod spectral_bridge;
