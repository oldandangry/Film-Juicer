//! Frozen native reference and owner-approved pinned-erff expectations. The
//! independent candidate executable predates the core implementation.

use std::path::{Path, PathBuf};

use film_juicer_core::{data_io, exposure, reconstruction};
use serde_json::Value;

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(repository().join("tests/ffi/fixtures/exposure/numerical.json")).unwrap(),
    )
    .unwrap()
}
fn word(value: &Value) -> f32 {
    f32::from_bits(value.as_u64().unwrap() as u32)
}
fn spectrum(value: &Value) -> [f32; 81] {
    std::array::from_fn(|i| word(&value[i]))
}
fn rgb(value: &Value) -> [[f32; 3]; 81] {
    std::array::from_fn(|i| std::array::from_fn(|c| word(&value[i][c])))
}
fn fixed<const N: usize>(value: &Value) -> [f32; N] {
    std::array::from_fn(|i| word(&value[i]))
}

#[test]
fn sensitivity_matches_frozen_leaves_and_actual_producer_inputs() {
    for row in fixture()["sensitivity"].as_array().unwrap() {
        let source = &row["input"];
        let expected = &row["expected"];
        let linear = rgb(&source["linear"]);
        let illuminant = spectrum(&source["illuminant"]);
        let reference = spectrum(&source["reference"]);
        let window = fixed(&source["window"]);
        let method = match source["method"].as_u64().unwrap() {
            0 => exposure::Method::Hanatos {
                window: source["apply_window"]
                    .as_bool()
                    .unwrap()
                    .then_some(exposure::Window {
                        params: &window,
                        reference_white: source["reference_valid"]
                            .as_bool()
                            .unwrap()
                            .then_some(&reference),
                    }),
            },
            1 => exposure::Method::Mallett,
            2 => exposure::Method::Arctic,
            _ => unreachable!(),
        };
        let input = exposure::Input {
            linear_sensitivity_rgb: &linear,
            reference_illuminant: &illuminant,
            band_pass: source["active"]
                .as_bool()
                .unwrap()
                .then_some(exposure::BandPass {
                    uv: fixed(&source["uv"]),
                    ir: fixed(&source["ir"]),
                }),
            method,
        };
        let result = exposure::prepare_sensitivity(input);
        assert_eq!(
            result.is_ok(),
            expected["built"].as_bool().unwrap(),
            "case {}",
            source["id"]
        );
        if let Ok(result) = result {
            let words = result.values_rgb().map(|rgb| rgb.map(f32::to_bits));
            let expected_words = rgb(&expected["expected"]).map(|rgb| rgb.map(f32::to_bits));
            assert_eq!(words, expected_words, "case {}", source["id"]);
            assert_eq!(result.hash(), expected["hash"].as_u64().unwrap());
            assert_eq!(
                result.mallett_green_scale().to_bits(),
                expected["scale"].as_u64().unwrap() as u32
            );
        } else {
            let error = result.err().unwrap();
            let reason = match expected["failure"].as_str().unwrap() {
                "reference_illuminant" => exposure::ErrorKind::ReferenceIlluminant,
                "band_pass_response" => exposure::ErrorKind::BandPassResponse,
                "missing_reference" => exposure::ErrorKind::MissingReference,
                "window_sample" => exposure::ErrorKind::WindowSample,
                "window_response" => exposure::ErrorKind::WindowResponse,
                "adapted_sensitivity" => exposure::ErrorKind::AdaptedSensitivity,
                "finite_hash" => exposure::ErrorKind::Hash,
                "mallett_response" => exposure::ErrorKind::MallettResponse,
                "mallett_scale" => exposure::ErrorKind::MallettScale,
                _ => unreachable!(),
            };
            assert_eq!(error.kind(), reason, "case {}", source["id"]);
            let index = expected["hash_failure_index"].as_u64().map(|i| i as usize);
            assert_eq!(
                error.hash_failure(),
                index.map(exposure::HashFailure::NonfiniteOperand)
            );
        }
    }
}

#[test]
fn reference_matches_native_sampling_reflection_and_failure_priority() {
    let mut procedural: Vec<f32> = (0..reconstruction::HANATOS_SAMPLE_COUNT)
        .map(|i| f32::from_bits(0x3e00_0123 + (i % 4093) as u32 * 173))
        .collect();
    let hanatos = data_io::load_spectra_lut(
        repository().join("Resources/luts/spectral_upsampling/irradiance_xy_tc.npy"),
    )
    .unwrap();
    for row in fixture()["reference"].as_array().unwrap() {
        let index = row["source_index"].as_u64().map(|v| v as usize);
        let saved = index.map(|i| procedural[i]);
        if let Some(index) = index {
            procedural[index] = word(&row["source_value"]);
        }
        let source = if row["source"] == "accepted_hanatos" {
            &hanatos
        } else {
            &procedural
        };
        let result = reconstruction::reference_white(
            source.as_slice().try_into().unwrap(),
            word(&row["blur"]),
            fixed(&row["white"]),
        );
        assert_eq!(result.is_ok(), row["built"].as_bool().unwrap(), "{row}");
        if let Ok(result) = result {
            assert_eq!(
                result.samples().map(f32::to_bits),
                spectrum(&row["expected"]).map(f32::to_bits)
            );
        } else {
            assert_eq!(
                result.err().unwrap().to_string(),
                row["diagnostic"].as_str().unwrap()
            );
        }
        if let (Some(index), Some(saved)) = (index, saved) {
            procedural[index] = saved;
        }
    }
}
