//! Independent native primitive vectors; see tests/profile/fixtures/manifest.json.

#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;

use serde::Deserialize;

use film_juicer_core::hash;

#[derive(Deserialize)]
struct Vector {
    id: String,
    bits: Vec<u32>,
    expected: [u64; 4],
}

#[test]
fn native_float_and_mask_vectors_match() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/profile/fixtures/hash.json");
    let vectors: Vec<Vector> = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(vectors.len(), 16);
    for vector in vectors {
        let samples: Vec<f32> = vector.bits.into_iter().map(f32::from_bits).collect();
        let finite = hash::finite_f32s(&samples);
        if let Some(index) = samples.iter().position(|v| !v.is_finite()) {
            assert_eq!(
                finite.unwrap_err(),
                hash::NonfiniteSample { index },
                "{}",
                vector.id
            );
            assert_eq!(vector.expected[0], 0); // Native failure sentinel, not a Rust hash.
        } else {
            assert_eq!(finite.unwrap(), vector.expected[0], "{}", vector.id);
        }
        let pair = hash::f32s_with_nan_mask(&samples);
        assert_eq!(
            [
                pair.values,
                pair.nan_mask,
                hash::u64s(&[pair.values, pair.nan_mask])
            ],
            vector.expected[1..],
            "{}",
            vector.id
        );
    }
}

#[test]
fn byte_and_word_order_retain_wrapping_arithmetic() {
    // Existing native tests/hash/hash_contract_test.cpp independently pins these.
    assert_eq!(hash::bytes(&[0, 1, 0xfe, 0xff]), 0x4651de7f9a7611e9);
    assert_eq!(
        hash::u64s(&[0x0123456789abcdef, u64::MAX]),
        0x000c8698d4e9656d
    );
    let mut incremental = hash::FNV_OFFSET;
    hash::update_bytes(&mut incremental, &[0, 1]);
    hash::update_bytes(&mut incremental, &[]);
    hash::update_bytes(&mut incremental, &[0xfe, 0xff]);
    assert_eq!(incremental, 0x4651de7f9a7611e9);
    assert_eq!(hash::bytes(&[]), hash::FNV_OFFSET);
    assert_eq!(hash::u64s(&[]), hash::FNV_OFFSET);
    // Primitives do not perform the remapping owned by specific final identities.
    let mut zero = 0;
    hash::update_bytes(&mut zero, &[]);
    assert_eq!(zero, 0);
}

#[test]
fn signed_zero_and_nan_payloads_follow_their_distinct_encodings() {
    assert_eq!(hash::finite_f32s(&[0.0, -0.0]).unwrap(), 0xa8c7f832281a39c5);
    assert_ne!(
        hash::bytes(&0.0_f32.to_le_bytes()),
        hash::bytes(&(-0.0_f32).to_le_bytes())
    );
    let positive_nan = f32::from_bits(0x7fc00001);
    let negative_nan = f32::from_bits(0xffc54321);
    assert_eq!(
        hash::f32s_with_nan_mask(&[positive_nan]),
        hash::f32s_with_nan_mask(&[negative_nan])
    );
    assert_ne!(
        hash::f32s_with_nan_mask(&[positive_nan]),
        hash::f32s_with_nan_mask(&[0.0])
    );
    assert_ne!(
        hash::f32s_with_nan_mask(&[f32::INFINITY]),
        hash::f32s_with_nan_mask(&[f32::NEG_INFINITY])
    );
    assert_ne!(hash::u64s(&[1, 2]), hash::u64s(&[2, 1]));
}

#[test]
fn native_mask_vectors_pin_the_word_boundary() {
    let mut samples = [0.0; 65];
    samples[63] = f32::NAN;
    let first = hash::f32s_with_nan_mask(&samples[..64]);
    assert_eq!(
        (first.values, first.nan_mask),
        (0xd80ac658736bb725, 0xa8c7783228196045)
    );
    samples[64] = f32::NAN;
    let second = hash::f32s_with_nan_mask(&samples);
    assert_eq!(
        (second.values, second.nan_mask),
        (0xde65f6d7d32ae7f5, 0xb3faa9e6089510c4)
    );
}

#[derive(Deserialize)]
struct ResourceVector {
    id: String,
    bits: Vec<u32>,
    repeat: usize,
    expected: u64,
}

#[test]
fn independent_resource_vectors_match_without_changing_samples() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/profile/fixtures/resource-hash.json");
    let vectors: Vec<ResourceVector> = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(vectors.len(), 13);
    for vector in vectors {
        let bits = vector.bits.repeat(vector.repeat);
        let samples: Vec<f32> = bits.iter().copied().map(f32::from_bits).collect();
        assert_eq!(
            hash::resource_f32s(&samples),
            vector.expected,
            "{}",
            vector.id
        );
        for (sample, bits) in samples.iter().zip(bits) {
            assert_eq!(sample.to_bits(), bits, "{}", vector.id);
        }
    }
}

#[test]
fn resource_hash_preserves_existing_native_finite_vectors() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/profile/fixtures/hash.json");
    let vectors: Vec<Vector> = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    for vector in vectors {
        let samples: Vec<f32> = vector.bits.into_iter().map(f32::from_bits).collect();
        if samples.iter().all(|sample| sample.is_finite()) {
            assert_eq!(
                hash::resource_f32s(&samples),
                vector.expected[0],
                "{}",
                vector.id
            );
        }
    }
}

#[test]
fn resource_equivalence_keeps_nan_zero_and_infinity_distinctions() {
    assert_eq!(hash::resource_f32s(&[0.0]), hash::resource_f32s(&[-0.0]));
    let nan = hash::resource_f32s(&[f32::from_bits(0x7fc00001)]);
    for bits in [0x7f800001, 0x7fffffff, 0xff800001, 0xffc54321] {
        assert_eq!(nan, hash::resource_f32s(&[f32::from_bits(bits)]));
    }
    assert_ne!(nan, hash::resource_f32s(&[0.0]));
    assert_ne!(
        hash::resource_f32s(&[f32::INFINITY]),
        hash::resource_f32s(&[f32::NEG_INFINITY])
    );
    assert_ne!(
        hash::resource_f32s(&[1.0, 2.0]),
        hash::resource_f32s(&[2.0, 1.0])
    );
    assert_ne!(
        hash::resource_f32s(&[1.0]),
        hash::resource_f32s(&[f32::from_bits(0x3f800001)])
    );
}
