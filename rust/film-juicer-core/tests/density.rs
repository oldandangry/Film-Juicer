//! Exact approved Rust samples, with independent native characterization/replay.
//! Provenance, the owner-approved exception and binary layout are in
//! tests/profile/fixtures/README.md and manifest.json.

#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;

use serde::Deserialize;

use film_juicer_core::hash;
#[path = "support/case_results.rs"]
mod test_cases;
use test_cases::RowCases;
use film_juicer_core::profile::{
    DensityCurveModel, DensityCurveSample, DensitySampleError, Polarity, load_film_source,
    load_print_source,
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
    #[serde(rename = "supported")]
    historically_supported: bool,
    sample_start: Option<usize>,
    identities: Option<Identities>,
}

impl Case {
    fn model(&self) -> DensityCurveModel {
        DensityCurveModel::new(
            coefficients(self.model.centers),
            coefficients(self.model.amplitudes),
            coefficients(self.model.sigmas),
        )
    }

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
fn valid_samples_and_replay_identities_match_the_frozen_cohort() {
    let fixture = fixture();
    assert_eq!(fixture.cases.len(), 138);
    assert_eq!(fixture.samples.len(), 19_247 * 49);
    let mut valid_count = 0;
    let mut historical_failure_count = 0;
    let mut identity_count = 0;
    let cases: Vec<_> = fixture
        .cases
        .iter()
        .filter(|c| c.historically_supported)
        .collect();
    let mut rows = RowCases::new("supported-density-cases", cases.iter().map(|case| &case.id));
    for case in cases {
        let row = rows.case(&case.id);
        let model = case.model();
        let mut actual = Vec::new();
        for (i, &bits) in fixture.axes[case.axis].iter().enumerate() {
            let Some(expected) = expected_sample(&fixture.samples, case.sample_start.unwrap() + i)
            else {
                // Every historical failure is asserted separately at its C1 boundary.
                historical_failure_count += 1;
                continue;
            };
            let sample = model.sample(case.polarity(), f64::from_bits(bits)).unwrap();
            assert_eq!(sample_bits(&sample), expected, "{} sample {i}", case.id);
            valid_count += 1;
            actual.push(sample);
        }
        if let Some(expected) = &case.identities {
            identity_count += 1;
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
        row.passed();
    }
    rows.finish();
    assert_eq!(
        (valid_count, historical_failure_count, identity_count),
        (19_235, 12, 119)
    );
}

#[test]
fn construction_retains_all_frozen_coefficient_bits() {
    let fixture = fixture();
    let mut historically_rejected_count = 0;
    let mut rows = RowCases::new(
        "all-density-cases",
        fixture.cases.iter().map(|case| &case.id),
    );
    for case in &fixture.cases {
        let row = rows.case(&case.id);
        let model = case.model();
        assert_eq!(
            model.centers().map(|r| r.map(f64::to_bits)),
            case.model.centers,
            "{} centers",
            case.id
        );
        assert_eq!(
            model.amplitudes().map(|r| r.map(f64::to_bits)),
            case.model.amplitudes,
            "{} amplitudes",
            case.id
        );
        assert_eq!(
            model.sigmas().map(|r| r.map(f64::to_bits)),
            case.model.sigmas,
            "{} sigmas",
            case.id
        );
        if !case.historically_supported {
            assert!(case.sample_start.is_none());
            assert!(case.identities.is_none());
            // These records contain no approved sample expectations.
            historically_rejected_count += 1;
        }
        row.passed();
    }
    rows.finish();
    assert_eq!(historically_rejected_count, 16);
}

#[test]
fn historical_sample_failures_reach_their_current_numerical_boundary() {
    let fixture = fixture();
    let mut endpoint_count = 0;
    let mut computed_failure_count = 0;
    let cases: Vec<_> = fixture
        .cases
        .iter()
        .filter(|c| c.historically_supported)
        .collect();
    let mut rows = RowCases::new(
        "supported-density-failure-boundaries",
        cases.iter().map(|case| &case.id),
    );
    for case in cases {
        let row = rows.case(&case.id);
        let model = case.model();
        for (i, &bits) in fixture.axes[case.axis].iter().enumerate() {
            if expected_sample(&fixture.samples, case.sample_start.unwrap() + i).is_some() {
                continue;
            }
            let sample = model.sample(case.polarity(), f64::from_bits(bits));
            match (case.id.as_str(), i) {
                ("exposure/narrowing-special", 0 | 1 | 12..=14) => {
                    // Frozen unit amplitudes/sigmas and zero centers give exact
                    // CDF endpoints. Invalid-record padding is not an oracle.
                    assert_eq!(case.model.centers, [[0; 3]; 3]);
                    assert_eq!(case.model.amplitudes, [[1.0_f64.to_bits(); 3]; 3]);
                    assert_eq!(case.model.sigmas, [[1.0_f64.to_bits(); 3]; 3]);
                    assert_eq!(case.polarity(), Polarity::Negative);
                    let mut expected = [0; 12];
                    if i >= 12 {
                        expected[..3].fill(3.0_f32.to_bits());
                        expected[3..].fill(1.0_f32.to_bits());
                    }
                    assert_eq!(
                        sample_bits(&sample.unwrap()),
                        expected,
                        "{} row {i}",
                        case.id
                    );
                    endpoint_count += 1;
                }
                ("exposure/narrowing-special", 15) => {
                    assert_eq!(
                        sample.unwrap_err(),
                        DensitySampleError::NonfiniteLayer {
                            channel: 0,
                            layer: 0
                        }
                    );
                    computed_failure_count += 1;
                }
                ("overflow/total-positive" | "overflow/total-negative", 0..=2) => {
                    assert_eq!(
                        sample.unwrap_err(),
                        DensitySampleError::NonfiniteTotal { channel: 0 }
                    );
                    computed_failure_count += 1;
                }
                _ => panic!("unclassified historical failure: {} row {i}", case.id),
            }
        }
        row.passed();
    }
    rows.finish();
    assert_eq!((endpoint_count, computed_failure_count), (5, 7));
    assert_eq!(19_235 + endpoint_count + computed_failure_count, 19_247);
}

#[test]
fn shipped_sources_match_frozen_model_and_axis_bits() {
    let fixture = fixture();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources/profiles");
    let mut count = 0;
    let cases: Vec<_> = fixture
        .cases
        .iter()
        .filter(|case| !case.profile.is_empty() && case.id == format!("{}/gamma-1.0", case.profile))
        .collect();
    let mut rows = RowCases::new(
        "shipped-profile-density-models",
        cases.iter().map(|case| &case.id),
    );
    for case in cases {
        let row = rows.case(&case.id);
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
        row.passed();
    }
    rows.finish();
    assert_eq!(count, 28);
}

#[test]
fn native_characterization_is_distinct_from_approved_rust_expectations() {
    let fixture = fixture();
    assert_eq!(fixture.native_witnesses.len(), 84);
    let ids: Vec<String> = fixture
        .native_witnesses
        .iter()
        .map(|w| format!("{}/{}/{}/{}", w.case, w.platform, w.mode, w.sample))
        .collect();
    let mut rows = RowCases::new("native-density-characterization-witnesses", &ids);
    for (witness, id) in fixture.native_witnesses.iter().zip(&ids) {
        let row = rows.case(id);
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
        row.passed();
    }
    rows.finish();
}

#[test]
fn unusual_coefficients_reach_density_arithmetic() {
    // Infinite sigma uses finite exposures to produce z = signed zero.
    let parameters: [(f64, f64, f32, f32); 7] = [
        (-1.0, 1e300_f64, 1.0_f32, 0.0_f32),
        (0.0, 1e300, 0.0, 1.0),
        (-0.0, 1e300, 1.0, 0.0),
        (1e-50, 1e300, 0.0, 1.0),
        (-1e-50, 1e300, 1.0, 0.0),
        (1e40, 1.0, 0.5, 0.5),
        (-1e40, 1.0, 0.5, 0.5),
    ];
    let ids: Vec<String> = parameters
        .iter()
        .map(|(sigma, _, _, _)| format!("sigma/{:016x}", sigma.to_bits()))
        .collect();
    let mut rows = RowCases::new("unusual-density-sigmas", &ids);
    let evaluation_names = [
        "negative/negative-exposure",
        "negative/positive-exposure",
        "positive/negative-exposure",
        "positive/positive-exposure",
    ];
    let evaluation_ids: Vec<String> = parameters
        .iter()
        .flat_map(|(sigma, _, _, _)| {
            evaluation_names
                .iter()
                .map(move |name| format!("sigma/{:016x}/{name}", sigma.to_bits()))
                .chain(if *sigma == -1.0 {
                    Some(format!(
                        "sigma/{:016x}/negative/zero-exposure",
                        sigma.to_bits()
                    ))
                } else if sigma.abs() < 1e-40 {
                    Some(format!(
                        "sigma/{:016x}/negative/zero-exposure-error",
                        sigma.to_bits()
                    ))
                } else {
                    None
                })
        })
        .collect();
    let mut evaluation_cases = RowCases::new("unusual-sigma-polarity-evaluations", &evaluation_ids);
    for ((sigma, exposure, low, high), id) in parameters.into_iter().zip(&ids) {
        let row = rows.case(id);
        let model = DensityCurveModel::new([[0.0; 3]; 3], [[1.0; 3]; 3], [[sigma; 3]; 3]);
        for (name, polarity, sample_exposure, expected) in [
            (evaluation_names[0], Polarity::Negative, -exposure, low),
            (evaluation_names[1], Polarity::Negative, exposure, high),
            (evaluation_names[2], Polarity::Positive, -exposure, high),
            (evaluation_names[3], Polarity::Positive, exposure, low),
        ] {
            let evaluation =
                evaluation_cases.case(format!("sigma/{:016x}/{name}", sigma.to_bits()));
            let sample = model.sample(polarity, sample_exposure).unwrap();
            assert_eq!(
                sample.layers.map(|r| r.map(f32::to_bits)),
                [[expected.to_bits(); 3]; 3]
            );
            assert_eq!(
                sample.total.map(f32::to_bits),
                [(expected * 3.0).to_bits(); 3]
            );
            evaluation.passed();
        }
        if sigma == -1.0 {
            let evaluation = evaluation_cases.case(format!(
                "sigma/{:016x}/negative/zero-exposure",
                sigma.to_bits()
            ));
            let sample = model.sample(Polarity::Negative, 0.0).unwrap();
            assert_eq!(
                sample.layers.map(|r| r.map(f32::to_bits)),
                [[0.5_f32.to_bits(); 3]; 3]
            );
            assert_eq!(sample.total.map(f32::to_bits), [1.5_f32.to_bits(); 3]);
            evaluation.passed();
        } else if sigma.abs() < 1e-40 {
            let evaluation = evaluation_cases.case(format!(
                "sigma/{:016x}/negative/zero-exposure-error",
                sigma.to_bits()
            ));
            assert_eq!(
                model.sample(Polarity::Negative, 0.0).unwrap_err(),
                DensitySampleError::NonfiniteLayer {
                    channel: 0,
                    layer: 0
                }
            );
            evaluation.passed();
        }
        row.passed();
    }
    evaluation_cases.finish();
    rows.finish();
    let centers = [(1e40, 0.0_f32, 1.0_f32), (-1e40, 1.0, 0.0)];
    let center_ids = ["center/positive-overflow", "center/negative-overflow"];
    let mut center_rows = RowCases::new("unusual-density-centers", center_ids);
    let center_evaluation_ids: Vec<String> = center_ids
        .iter()
        .flat_map(|id| ["negative", "positive"].map(|polarity| format!("{id}/{polarity}")))
        .collect();
    let mut center_evaluations = RowCases::new("unusual-center-polarities", &center_evaluation_ids);
    for ((center, negative, positive), id) in centers.into_iter().zip(center_ids) {
        let row = center_rows.case(id);
        let model = DensityCurveModel::new([[center; 3]; 3], [[1.0; 3]; 3], [[1.0; 3]; 3]);
        for (name, polarity, expected) in [
            ("negative", Polarity::Negative, negative),
            ("positive", Polarity::Positive, positive),
        ] {
            let evaluation = center_evaluations.case(format!("{id}/{name}"));
            let sample = model.sample(polarity, 0.0).unwrap();
            assert_eq!(
                sample.layers.map(|r| r.map(f32::to_bits)),
                [[expected.to_bits(); 3]; 3]
            );
            assert_eq!(
                sample.total.map(f32::to_bits),
                [(expected * 3.0).to_bits(); 3]
            );
            evaluation.passed();
        }
        row.passed();
    }
    center_evaluations.finish();
    center_rows.finish();
}

#[test]
fn scalar_exposure_narrowing_produces_exact_endpoints() {
    let model = DensityCurveModel::new([[0.0; 3]; 3], [[1.0; 3]; 3], [[1.0; 3]; 3]);
    let exposures = [
        (-1e40, 0.0_f32, 1.0_f32),
        (1e40, 1.0, 0.0),
        (f64::NEG_INFINITY, 0.0, 1.0),
        (f64::INFINITY, 1.0, 0.0),
    ];
    let ids: Vec<String> = exposures
        .iter()
        .map(|(exposure, _, _)| format!("exposure/{:016x}", exposure.to_bits()))
        .collect();
    let mut rows = RowCases::new("scalar-exposure-endpoints", &ids);
    let endpoint_ids: Vec<String> = ids
        .iter()
        .flat_map(|id| ["negative", "positive"].map(|polarity| format!("{id}/{polarity}")))
        .collect();
    let mut endpoint_cases = RowCases::new("scalar-exposure-polarity-endpoints", &endpoint_ids);
    for ((exposure, negative, positive), id) in exposures.into_iter().zip(&ids) {
        let row = rows.case(id);
        for (name, polarity, expected) in [
            ("negative", Polarity::Negative, negative),
            ("positive", Polarity::Positive, positive),
        ] {
            let endpoint_case = endpoint_cases.case(format!("{id}/{name}"));
            let sample = model.sample(polarity, exposure).unwrap();
            assert_eq!(
                sample.layers.map(|r| r.map(f32::to_bits)),
                [[expected.to_bits(); 3]; 3]
            );
            assert_eq!(
                sample.total.map(f32::to_bits),
                [(expected * 3.0).to_bits(); 3]
            );
            endpoint_case.passed();
        }
        row.passed();
    }
    endpoint_cases.finish();
    rows.finish();
    let mut nan_rows = RowCases::new("scalar-exposure-invalid", ["nan"]);
    let nan_row = nan_rows.case("nan");
    assert_eq!(
        model.sample(Polarity::Negative, f64::NAN).unwrap_err(),
        DensitySampleError::NonfiniteLayer {
            channel: 0,
            layer: 0
        }
    );
    nan_row.passed();
    nan_rows.finish();
}

#[test]
fn computed_failures_identify_channel_and_layer() {
    let mut rows = RowCases::new(
        "computed-density-failures",
        [
            "zero-sigma-layer",
            "overflow-amplitude-negative-zero",
            "overflow-amplitude-negative-infinity",
            "nonfinite-total-channel",
        ],
    );
    let zero_sigma = rows.case("zero-sigma-layer");
    let mut sigmas = [[1.0; 3]; 3];
    sigmas[2][1] = 0.0;
    let model = DensityCurveModel::new([[0.0; 3]; 3], [[1.0; 3]; 3], sigmas);
    assert_eq!(
        model.sample(Polarity::Negative, 0.0).unwrap_err(),
        DensitySampleError::NonfiniteLayer {
            channel: 2,
            layer: 1
        }
    );
    zero_sigma.passed();
    let mut amplitudes = [[1.0; 3]; 3];
    amplitudes[1][2] = f64::MAX;
    let model = DensityCurveModel::new([[0.0; 3]; 3], amplitudes, [[1.0; 3]; 3]);
    for exposure in [0.0, f64::NEG_INFINITY] {
        let id = if exposure == 0.0 {
            "overflow-amplitude-negative-zero"
        } else {
            "overflow-amplitude-negative-infinity"
        };
        let row = rows.case(id);
        assert_eq!(
            model.sample(Polarity::Negative, exposure).unwrap_err(),
            DensitySampleError::NonfiniteLayer {
                channel: 1,
                layer: 2
            }
        );
        row.passed();
    }
    amplitudes = [[1.0; 3]; 3];
    amplitudes[2] = [f64::from(f32::MAX); 3];
    let model = DensityCurveModel::new([[0.0; 3]; 3], amplitudes, [[1.0; 3]; 3]);
    assert_eq!(
        model.sample(Polarity::Negative, 0.0).unwrap_err(),
        DensitySampleError::NonfiniteTotal { channel: 2 }
    );
    rows.case("nonfinite-total-channel").passed();
    rows.finish();
}

#[test]
fn integer_negative_zero_retains_owned_bits_and_raw_identity() {
    let mut rows = RowCases::new("integer-negative-zero-identity", ["integer-negative-zero"]);
    let row = rows.case("integer-negative-zero");
    let negative: f64 = serde_json::from_str("-0").unwrap();
    let decimal: f64 = serde_json::from_str("-0.0").unwrap();
    let positive: f64 = serde_json::from_str("0").unwrap();
    let model = DensityCurveModel::new([[negative; 3]; 3], [[negative; 3]; 3], [[1.0; 3]; 3]);
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
    row.passed();
    rows.finish();
}
