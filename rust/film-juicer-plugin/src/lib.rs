//! Film-Juicer's Rust plug-in integration boundary.

#![deny(unsafe_code)]

#[cfg(test)]
#[path = "../../film-juicer-core/tests/support/case_results.rs"]
mod test_cases;

#[allow(
    unsafe_code,
    reason = "checked color fixture operations validate initialized, disjoint foreign storage at this private raw edge"
)]
#[cfg(any(test, feature = "test-support"))]
mod color_values;

mod cuda;
mod ofx;

mod asset_catalog;

mod instance;
mod submission;

#[cfg(feature = "test-support")]
mod frame_observer;

mod runtime;

#[allow(
    unsafe_code,
    reason = "module lookup and the characterized clock use native OS declarations with exact path units and output widths"
)]
mod platform;

#[allow(
    unsafe_code,
    reason = "the private process edge admits native paths and projects synchronous borrows and consume-once Runtime ownership"
)]
#[cfg(any(test, feature = "test-support"))]
mod runtime_bridge;

#[allow(
    unsafe_code,
    reason = "the instance C edge validates foreign records and consumes host-write and admission handles"
)]
#[cfg(any(test, feature = "test-support"))]
mod instance_bridge;

#[allow(
    unsafe_code,
    reason = "the private production asset edge checks foreign extents and owns consume-once handles"
)]
mod asset_bridge;

#[allow(
    unsafe_code,
    reason = "held preparation and cache views admit foreign storage and keep raw projections at their owner-bounded C edge"
)]
mod preparation;

mod asset_profile;

#[allow(
    unsafe_code,
    reason = "the temporary film development edge scopes profile borrows and consumes result owners once"
)]
#[cfg(any(test, feature = "test-support"))]
mod film_bridge;

#[allow(
    unsafe_code,
    reason = "temporary film recipe transport admits foreign records and scopes consuming owners"
)]
mod film_recipe_bridge;

#[allow(
    unsafe_code,
    reason = "the film-input value boundary checks fixed extents and scopes immutable profile/resource borrows"
)]
#[cfg(any(test, feature = "test-support"))]
mod film_input_bridge;

mod asset_spectral;

mod asset_noise;

mod asset_calibration;

#[allow(
    unsafe_code,
    reason = "the illuminant C edge validates spans and consumes call-local lens owners"
)]
mod asset_illuminant;

#[cfg(any(test, feature = "test-support"))]
#[allow(
    unsafe_code,
    reason = "the test facade checks foreign inputs and consumes retained profile owners at the C boundary"
)]
mod test_support;

#[allow(
    unsafe_code,
    reason = "the fixed spectral edge checks foreign extents and scopes immutable borrows"
)]
#[cfg(any(test, feature = "test-support"))]
mod spectral_bridge;

#[allow(
    unsafe_code,
    reason = "the exposure edge checks fixed foreign spans and scopes immutable tensor borrows"
)]
#[cfg(any(test, feature = "test-support"))]
mod exposure_bridge;

#[allow(
    unsafe_code,
    reason = "the TC edge validates spans and transfers/releases an exact Vec allocation"
)]
#[cfg(any(test, feature = "test-support"))]
mod reconstruction_bridge;

#[allow(
    unsafe_code,
    reason = "the gamut edge checks fixed foreign operands, optional diagnostics and synchronous result copies"
)]
#[cfg(any(test, feature = "test-support"))]
mod gamut_bridge;

#[allow(
    unsafe_code,
    reason = "fixed optics transport admits foreign extents and commits complete synchronous stack values"
)]
mod optics_bridge;

#[allow(
    unsafe_code,
    reason = "fixed print illumination transport admits foreign extents and copies complete synchronous values"
)]
mod print_bridge;

#[allow(
    unsafe_code,
    reason = "scanner transport checks foreign extents and scopes synchronous immutable input borrows"
)]
mod scanner_bridge;

#[allow(
    unsafe_code,
    reason = "the fixed frame edge admits foreign records and contains whole-operation unwind"
)]
mod frame_bridge;

#[allow(
    unsafe_code,
    reason = "fixed diffusion transport admits foreign records and contains whole-operation unwind"
)]
mod diffusion_bridge;

#[allow(
    unsafe_code,
    reason = "fixed DIR frame transport admits foreign storage and contains whole-operation unwind"
)]
mod dir_frame_bridge;

#[allow(
    unsafe_code,
    reason = "the temporary reflected-boundary edge admits foreign records and consumes scoped host owners once"
)]
mod dir_boundary_bridge;

#[allow(
    unsafe_code,
    reason = "the grain frame edge admits synchronous foreign storage and contains whole-operation unwind"
)]
mod grain_frame_bridge;
#[allow(
    unsafe_code,
    reason = "the physical effects frame edge admits fixed foreign records and contains whole-operation unwind"
)]
mod effects_frame_bridge;
