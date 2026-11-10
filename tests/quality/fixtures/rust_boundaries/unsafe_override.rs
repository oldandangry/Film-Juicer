#[allow(unsafe_code, reason = "negative control: a safe owner must reject this override")]
pub fn foreign_read(address: *const u8) -> u8 {
    // SAFETY: deliberately invalid for a safe API; this fixture must not compile.
    unsafe { *address }
}
