//! FNV-1a primitives with the little-endian and floating-point rules of Hash.h.

use std::fmt;

pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

pub fn update_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

pub fn bytes(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    update_bytes(&mut hash, bytes);
    hash
}

pub fn u64s(values: &[u64]) -> u64 {
    let mut hash = FNV_OFFSET;
    for value in values {
        update_bytes(&mut hash, &value.to_le_bytes());
    }
    hash
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NonfiniteSample {
    pub index: usize,
}

impl fmt::Display for NonfiniteSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "nonfinite hash sample at index {}", self.index)
    }
}
impl std::error::Error for NonfiniteSample {}

/// Hash finite samples, canonicalizing either zero sign to positive zero.
pub fn finite_f32s(samples: &[f32]) -> Result<u64, NonfiniteSample> {
    let mut hash = FNV_OFFSET;
    for (index, &sample) in samples.iter().enumerate() {
        if !sample.is_finite() {
            return Err(NonfiniteSample { index });
        }
        let bits = if sample == 0.0 { 0 } else { sample.to_bits() };
        update_bytes(&mut hash, &bits.to_le_bytes());
    }
    Ok(hash)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatSpanHash {
    pub values: u64,
    pub nan_mask: u64,
}

/// NaNs contribute +0 to values and a set bit to the LSB-first 64-bit mask.
/// Infinities retain their bits; either zero sign contributes +0.
pub fn f32s_with_nan_mask(samples: &[f32]) -> FloatSpanHash {
    let mut hashes = FloatSpanHash {
        values: FNV_OFFSET,
        nan_mask: FNV_OFFSET,
    };
    for chunk in samples.chunks(64) {
        let mut word = 0_u64;
        for (bit, &sample) in chunk.iter().enumerate() {
            let bits = if sample.is_nan() {
                word |= 1_u64 << bit;
                0
            } else if sample == 0.0 {
                0
            } else {
                sample.to_bits()
            };
            update_bytes(&mut hashes.values, &bits.to_le_bytes());
        }
        update_bytes(&mut hashes.nan_mask, &word.to_le_bytes());
    }
    hashes
}
