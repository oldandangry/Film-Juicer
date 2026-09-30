//! Exact approved Rust samples, with independent native characterization/replay.
//! Provenance, the owner-approved exception and binary layout are in
//! tests/profile/fixtures/README.md and manifest.json.

#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;

use serde::Deserialize;

use film_juicer_core::hash;
use film_juicer_core::profile::{
    DensityCurveModel, DensityCurveSample, DensitySampleError, Polarity, Requirement,
    load_film_source, load_print_source,
};

#[derive(Deserialize)]
struct ModelBits {
    centers: [[u64; 3]; 3],
    amplitudes: [[u64; 3]; 3],
    sigmas: [[u64; 3]; 3],
}

#[derive(Deserialize)]
struct Identities {
    total_hash: u64,
    layer_hash: u64,
    print_hash: u64,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    profile: String,
    positive: bool,
    gamma_bits: u64,
    axis: usize,
    model: ModelBits,
    supported: bool,
    sample_start: Option<usize>,
    identities: Option<Identities>,
}

impl Case {
    fn polarity(&self) -> Polarity {
        if self.positive {
            Polarity::Positive
        } else {
            Polarity::Negative
        }
    }
}

#[derive(Deserialize)]
struct NativeWitness {
    case: String,
    platform: String,
    mode: String,
    sample: usize,
    native: [u32; 12],
}

#[derive(Deserialize)]
struct Fixture {
    axes: Vec<Vec<u64>>,
    cases: Vec<Case>,
    native_witnesses: Vec<NativeWitness>,
    #[serde(skip)]
    samples: Vec<u8>,
}

fn fixture() -> Fixture {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/profile/fixtures");
    let mut fixture: Fixture =
        serde_json::from_slice(&fs::read(root.join("density.json")).unwrap()).unwrap();
    fixture.samples = fs::read(root.join("density-samples.bin")).unwrap();
    fixture
}

fn coefficients(bits: [[u64; 3]; 3]) -> [[f64; 3]; 3] {
    bits.map(|row| row.map(f64::from_bits))
}

fn sample_bits(sample: &DensityCurveSample) -> [u32; 12] {
    std::array::from_fn(|i| {
        if i < 3 {
            sample.total[i].to_bits()
        } else {
            sample.layers[(i - 3) / 3][(i - 3) % 3].to_bits()
        }
    })
}

fn expected_sample(samples: &[u8], index: usize) -> Option<[u32; 12]> {
    let row = &samples[index * 49..(index + 1) * 49];
    match row[0] {
        0 => None,
        1 => Some(std::array::from_fn(|i| {
            u32::from_le_bytes(row[1 + i * 4..5 + i * 4].try_into().unwrap())
        })),
        _ => panic!("invalid fixture validity byte"),
    }
}

fn combined_hash(samples: &[f32]) -> u64 {
    let pair = hash::f32s_with_nan_mask(samples);
    hash::u64s(&[pair.values, pair.nan_mask])
}

#[test]
fn samples_and_admission_match_the_frozen_cohort() {
    let fixture = fixture();
    assert_eq!(fixture.cases.len(), 138);
    assert_eq!(fixture.samples.len(), 19_247 * 49);
    let mut valid_count = 0;
    let mut invalid_count = 0;
    let mut rejected_count = 0;
    for case in &fixture.cases {
        let model = DensityCurveModel::new(
            coefficients(case.model.centers),
            coefficients(case.model.amplitudes),
            coefficients(case.model.sigmas),
        );
        assert_eq!(model.is_ok(), case.supported, "{} admission", case.id);
        let Ok(model) = model else {
            assert!(case.sample_start.is_none());
            rejected_count += 1;
            continue;
        };
        // Admission retains source f64 precision; no f32 coefficient cache.
        assert_eq!(
            model.centers().map(|r| r.map(f64::to_bits)),
            case.model.centers
        );
        assert_eq!(
            model.amplitudes().map(|r| r.map(f64::to_bits)),
            case.model.amplitudes
        );
        assert_eq!(
            model.sigmas().map(|r| r.map(f64::to_bits)),
            case.model.sigmas
        );
        let mut actual = Vec::new();
        for (i, &bits) in fixture.axes[case.axis].iter().enumerate() {
            let sample = model.sample(case.polarity(), f64::from_bits(bits));
            let expected = expected_sample(&fixture.samples, case.sample_start.unwrap() + i);
            assert_eq!(
                sample.as_ref().ok().map(sample_bits),
                expected,
                "{} sample {i}",
                case.id
            );
            match sample {
                Ok(sample) => {
                    valid_count += 1;
                    actual.push(sample);
                }
                Err(_) => invalid_count += 1,
            }
        }
        if let Some(expected) = &case.identities {
            // Test-only composition mirrors the consuming native order. It does
            // not introduce profile/recipe identity APIs ahead of their slices.
            assert_eq!(actual.len(), fixture.axes[case.axis].len());
            let totals: Vec<f32> = actual.iter().flat_map(|s| s.total).collect();
            assert_eq!(
                combined_hash(&totals),
                expected.total_hash,
                "{} totals",
                case.id
            );
            let mut layer_hashes = Vec::new();
            for layer in 0..3 {
                for channel in 0..3 {
                    let values: Vec<f32> =
                        actual.iter().map(|s| s.layers[layer][channel]).collect();
                    layer_hashes.push(combined_hash(&values));
                }
            }
            assert_eq!(
                hash::u64s(&layer_hashes),
                expected.layer_hash,
                "{} layers",
                case.id
            );
            let mut print_hash = hash::FNV_OFFSET;
            hash::update_bytes(&mut print_hash, &(actual.len() as u64).to_le_bytes());
            for &bits in &fixture.axes[case.axis] {
                hash::update_bytes(
                    &mut print_hash,
                    &(f64::from_bits(bits) as f32).to_le_bytes(),
                );
            }
            for value in totals {
                hash::update_bytes(&mut print_hash, &value.to_le_bytes());
            }
            assert_eq!(
                print_hash, expected.print_hash,
                "{} raw print encoding",
                case.id
            );
        }
    }
    assert_eq!(
        (valid_count, invalid_count, rejected_count),
        (19_235, 12, 16)
    );
}

#[test]
fn shipped_sources_match_frozen_model_and_axis_bits() {
    let fixture = fixture();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources/profiles");
    let mut count = 0;
    for case in &fixture.cases {
        if case.profile.is_empty() || case.id != format!("{}/gamma-1.0", case.profile) {
            continue;
        }
        assert_eq!(case.gamma_bits, 1.0_f64.to_bits());
        let path = root.join(format!("{}.json", case.profile));
        let source = load_film_source(&path)
            .or_else(|_| load_print_source(&path))
            .unwrap();
        assert_eq!(source.info().polarity(), case.polarity());
        assert_eq!(
            source
                .density_model()
                .centers()
                .map(|r| r.map(f64::to_bits)),
            case.model.centers
        );
        assert_eq!(
            source
                .density_model()
                .amplitudes()
                .map(|r| r.map(f64::to_bits)),
            case.model.amplitudes
        );
        assert_eq!(
            source.density_model().sigmas().map(|r| r.map(f64::to_bits)),
            case.model.sigmas
        );
        let axis: Vec<u64> = source
            .samples()
            .log_exposure()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(axis, fixture.axes[case.axis]);
        count += 1;
    }
    assert_eq!(count, 28);
}

#[test]
fn native_characterization_is_distinct_from_approved_rust_expectations() {
    let fixture = fixture();
    assert_eq!(fixture.native_witnesses.len(), 84);
    for witness in &fixture.native_witnesses {
        let case = fixture.cases.iter().find(|c| c.id == witness.case).unwrap();
        let approved = expected_sample(
            &fixture.samples,
            case.sample_start.unwrap() + witness.sample,
        )
        .unwrap();
        assert_ne!(
            approved, witness.native,
            "{} {} {}",
            witness.case, witness.platform, witness.mode
        );
        assert!(
            witness
                .native
                .iter()
                .all(|&bits| f32::from_bits(bits).is_finite())
        );
    }
}

#[test]
fn failures_identify_the_construction_or_sample_boundary() {
    let mut sigmas = [[1.0; 3]; 3];
    sigmas[2][1] = f64::MIN_POSITIVE;
    let error = DensityCurveModel::new([[0.0; 3]; 3], [[1.0; 3]; 3], sigmas).unwrap_err();
    assert_eq!(
        (error.field, error.channel, error.layer, error.requirement),
        ("sigmas", 2, 1, Requirement::PositiveSigma)
    );
    let mut amplitudes = [[1.0; 3]; 3];
    amplitudes[1][2] = f64::MAX;
    let error = DensityCurveModel::new([[0.0; 3]; 3], amplitudes, [[1.0; 3]; 3]).unwrap_err();
    assert_eq!(
        (error.field, error.channel, error.layer, error.requirement),
        ("amplitudes", 1, 2, Requirement::FloatRange)
    );
    let model = DensityCurveModel::new([[0.0; 3]; 3], [[f64::from(f32::MAX); 3]; 3], [[1.0; 3]; 3])
        .unwrap();
    for exposure in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            model.sample(Polarity::Negative, exposure).unwrap_err(),
            DensitySampleError::NonfiniteExposure
        );
    }
    for exposure in [f64::MAX, -f64::MAX] {
        assert_eq!(
            model.sample(Polarity::Negative, exposure).unwrap_err(),
            DensitySampleError::ExposureRange
        );
    }
    assert_eq!(
        model.sample(Polarity::Negative, 0.0).unwrap_err(),
        DensitySampleError::NonfiniteTotal { channel: 0 }
    );
}

#[test]
fn integer_negative_zero_retains_owned_bits_and_raw_identity() {
    let negative: f64 = serde_json::from_str("-0").unwrap();
    let decimal: f64 = serde_json::from_str("-0.0").unwrap();
    let positive: f64 = serde_json::from_str("0").unwrap();
    let model =
        DensityCurveModel::new([[negative; 3]; 3], [[negative; 3]; 3], [[1.0; 3]; 3]).unwrap();
    assert_eq!(model.centers()[0][0].to_bits(), negative.to_bits());
    assert_eq!(model.amplitudes()[0][0].to_bits(), negative.to_bits());
    let sample = model.sample(Polarity::Negative, negative).unwrap();
    assert_eq!(
        sample.layers.map(|r| r.map(f32::to_bits)),
        [[(-0.0_f32).to_bits(); 3]; 3]
    );
    assert_eq!(sample.total.map(f32::to_bits), [0; 3]);
    assert_eq!(
        hash::bytes(&negative.to_le_bytes()),
        hash::bytes(&decimal.to_le_bytes())
    );
    assert_ne!(
        hash::bytes(&negative.to_le_bytes()),
        hash::bytes(&positive.to_le_bytes())
    );
    assert_eq!(
        hash::finite_f32s(&[negative as f32]),
        hash::finite_f32s(&[positive as f32])
    );
}
