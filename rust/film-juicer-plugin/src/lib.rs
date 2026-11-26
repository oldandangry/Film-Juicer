//! Film-Juicer's Rust plug-in integration boundary.

#![deny(unsafe_code)]

mod asset_noise;
mod cuda;
mod instance;
mod ofx;
#[allow(
    unsafe_code,
    reason = "the OS edge resolves the loaded module and reads the native clock"
)]
mod platform;
mod runtime;
mod submission;
