//! Film-Juicer's Rust plug-in integration boundary.

#![deny(unsafe_code)]

#[allow(
    unsafe_code,
    reason = "the temporary C ABI validates its foreign input at this boundary"
)]
mod legacy_bridge;

mod cuda;
