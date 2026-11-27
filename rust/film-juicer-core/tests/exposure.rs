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
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/exposure/numerical.json"),
        )
        .unwrap(),
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
    let rows = fixture()["sensitivity"].as_array().unwrap().to_vec();
    let ids: Vec<String> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row["input"]["id"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("sensitivity/{i}"))
        })
        .collect();
    let cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "exposure-sensitivity-inputs"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "exposure-sensitivity-inputs"
    );
    for (row, id) in rows.iter().zip(&ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
}

#[test]
fn reference_matches_native_sampling_reflection_and_failure_priority() {
    let mut procedural: Vec<f32> = (0..reconstruction::SPECTRA_SAMPLE_COUNT)
        .map(|i| f32::from_bits(0x3e00_0123 + (i % 4093) as u32 * 173))
        .collect();
    let hanatos = data_io::load_spectra_lut(
        repository().join("Resources/luts/spectral_upsampling/irradiance_xy_tc.npy"),
    )
    .unwrap();
    let rows = fixture()["reference"].as_array().unwrap().to_vec();
    let ids: Vec<String> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row["id"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("reference/{i}"))
        })
        .collect();
    let cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "exposure-reference-white-rows"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "exposure-reference-white-rows"
    );
    for (row, id) in rows.iter().zip(&ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
}

#[path = "support/capture.rs"]
mod capture;

fn mallett_fixture() -> Value {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/exposure");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(directory.join("mallett-manifest.json")).unwrap())
            .unwrap();
    let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
        (false, true) => "linux-debug",
        (false, false) => "linux-release",
        (true, true) => "windows-clang-debug",
        (true, false) => "windows-clang-release",
    };
    capture::read_parts(&directory, &manifest["presets"][preset]["numerical"])
}
fn exact_words(actual: &[f32], expected: &Value) {
    assert_eq!(actual.len(), expected.as_array().unwrap().len());
    for (actual, expected) in actual.iter().zip(expected.as_array().unwrap()) {
        assert!(
            actual.to_bits() == word(expected).to_bits()
                || (actual.is_nan() && word(expected).is_nan())
        );
    }
}
#[test]
fn focused_mallett_uses_independent_color_composition_and_bgr_expectations() {
    use film_juicer_core::color;
    let rows = mallett_fixture()["focused"].as_array().unwrap().to_vec();
    let ids: Vec<String> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row["id"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("focused/{i}"))
        })
        .collect();
    let cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "focused-mallett-cases"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "focused-mallett-cases"
    );
    for (row, id) in rows.iter().zip(&ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        let input = &row["input"];
        let basis = rgb(&input["basis"]);
        let illuminant = spectrum(&input["illuminant"]);
        let sensitivity = rgb(&input["sensitivity"]);
        let conversion = color::InputConversion {
            space: match input["space"].as_u64().unwrap() {
                0 => color::InputSpace::DavinciWideGamut,
                1 => color::InputSpace::Bt2020,
                2 => color::InputSpace::Aces2065_1,
                3 => color::InputSpace::SrgbRec709,
                _ => unreachable!(),
            },
            decode_cctf: input["decode"].as_bool().unwrap(),
            rgb_to_xyz: fixed(&input["rgb_to_xyz"]),
            xyz_adaptation: input["adapt"]
                .as_bool()
                .unwrap()
                .then(|| fixed(&input["xyz_adapt"])),
        };
        let result = exposure::mallett_midgray(exposure::MallettInput {
            color: conversion,
            xyz_to_linear_srgb: fixed(&input["xyz_to_srgb"]),
            basis_rgb: &basis,
            illuminant: &illuminant,
            sensitivity_rgb: &sensitivity,
        });
        let expected = &row["expected"];
        exact_words(result.midgray_dwg_rgb(), &expected["dwg"]);
        exact_words(result.raw_midgray_bgr(), &expected["raw_bgr"]);
        exact_words(
            &[
                result.normalization().raw_green(),
                result.normalization().scale(),
            ],
            &expected["normalization"],
        );
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
}
#[test]
fn reference_source_and_distinct_rgb_reduction_preserve_frozen_outcomes() {
    let fixture = mallett_fixture();
    let source_rows = fixture["sources"].as_array().unwrap();
    let source_ids: Vec<String> = (0..source_rows.len())
        .map(|i| format!("source/{i}"))
        .collect();
    let sources_ids: Vec<_> = IntoIterator::into_iter(&source_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !sources_ids.is_empty(),
        "empty case set: {}",
        "mallett-reference-source-ev"
    );
    let sources_count = sources_ids.len();
    let mut sources_remaining = sources_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        sources_remaining.len(),
        sources_count,
        "duplicate case IDs: {}",
        "mallett-reference-source-ev"
    );
    for (row, id) in source_rows.iter().zip(&source_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        let result = exposure::reference_source(word(&row["ev"]));
        assert_eq!(result.is_ok(), row["built"].as_bool().unwrap());
        match result {
            Ok(value) => assert_eq!(
                value.value().to_bits(),
                word(&row["exp2_source"][1]).to_bits()
            ),
            Err(error) => assert_eq!(error.kind(), exposure::ErrorKind::ReferenceSource),
        }
        assert!(
            sources_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        sources_remaining.is_empty(),
        "missing comparisons: {sources_remaining:?}"
    );
    let reference_rows = fixture["references"].as_array().unwrap();
    let reference_ids: Vec<String> = (0..reference_rows.len())
        .map(|i| format!("reference/{i}"))
        .collect();
    let references_ids: Vec<_> = IntoIterator::into_iter(&reference_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !references_ids.is_empty(),
        "empty case set: {}",
        "mallett-reference-raw-cases"
    );
    let references_count = references_ids.len();
    let mut references_remaining = references_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        references_remaining.len(),
        references_count,
        "duplicate case IDs: {}",
        "mallett-reference-raw-cases"
    );
    for (row, id) in reference_rows.iter().zip(&reference_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        let basis = rgb(&row["basis"]);
        let illuminant = spectrum(&row["illuminant"]);
        let sensitivity = rgb(&row["sensitivity"]);
        let result = exposure::mallett_reference_raw(exposure::ReferenceInput {
            basis_rgb: &basis,
            illuminant: &illuminant,
            sensitivity_rgb: &sensitivity,
            source: word(&row["source_scale"][0]),
            green_scale: word(&row["source_scale"][1]),
        });
        assert_eq!(result.is_ok(), row["built"].as_bool().unwrap());
        match result {
            Ok(value) => exact_words(value.rgb(), &row["expected"]),
            Err(error) => assert_eq!(error.kind(), exposure::ErrorKind::ReferenceRaw),
        }
        assert!(
            references_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        references_remaining.is_empty(),
        "missing comparisons: {references_remaining:?}"
    );
}
#[test]
fn tc_green_threshold_zero_sign_and_reciprocal_match_independent_expectations() {
    let rows = mallett_fixture()["normalizations"]
        .as_array()
        .unwrap()
        .to_vec();
    let ids: Vec<String> = (0..rows.len())
        .map(|i| format!("normalization/{i}"))
        .collect();
    let cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "mallett-normalization-inputs"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "mallett-normalization-inputs"
    );
    for (row, id) in rows.iter().zip(&ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        let result = exposure::tc_midgray(word(&row["green"]));
        assert_eq!(
            result.raw_green().to_bits(),
            word(&row["expected"][0]).to_bits()
        );
        assert_eq!(
            result.scale().to_bits(),
            word(&row["expected"][2]).to_bits()
        );
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
}
