//! Print gamma uses pre-C5 model/sample captures and independent native identity
//! replay. New unusual models use exact Gaussian endpoints and half responses.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use film_juicer_core::hash;
use film_juicer_core::profile::{
    DensitySampleError, PrintDensityError, PrintProfile, load_print_source,
};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture(name: &str) -> Value {
    serde_json::from_slice(
        &fs::read(repository().join("tests/profile/fixtures").join(name)).unwrap(),
    )
    .unwrap()
}
fn number(node: &Value) -> u64 {
    node.as_u64().unwrap()
}
fn doubles(node: &Value) -> Vec<f64> {
    node.as_array()
        .unwrap()
        .iter()
        .map(|n| f64::from_bits(number(n)))
        .collect()
}
fn coefficients(node: &Value) -> [[f64; 3]; 3] {
    serde_json::from_value::<[[u64; 3]; 3]>(node.clone())
        .unwrap()
        .map(|row| row.map(f64::from_bits))
}
fn model_bits(profile: &PrintProfile) -> [[[u64; 3]; 3]; 3] {
    [
        profile.density_model().centers(),
        profile.density_model().amplitudes(),
        profile.density_model().sigmas(),
    ]
    .map(|m| m.map(|r| r.map(f64::to_bits)))
}
fn totals_bits(totals: &[[f32; 3]]) -> Vec<[u32; 3]> {
    totals.iter().map(|r| r.map(f32::to_bits)).collect()
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::var_os("JUICER_TEST_ARTIFACT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| repository().join("out/validation/rust-core"));
        let path = root.join(format!(
            "print-gamma-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn profile(&self, document: &Value) -> PrintProfile {
        let path = self.0.join("source.json");
        fs::write(&path, document.to_string()).unwrap();
        PrintProfile::new(load_print_source(&path).unwrap()).unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn document(model: &Value, axis: &[f64]) -> Value {
    json!({
        "info": {"stock":"gamma-contract", "stage":"printing", "type":"negative"},
        "data": {
            "wavelengths": ([380.0;81].as_slice()),
            "log_sensitivity": ([[0.0;3];81].as_slice()),
            "channel_density": ([[0.0;3];81].as_slice()),
            "base_density": ([0.0;81].as_slice()),
            "log_exposure": axis,
            "density_curves_model": {
                "centers":coefficients(&model["centers"]),
                "amplitudes":coefficients(&model["amplitudes"]),
                "sigmas":coefficients(&model["sigmas"]),
            }
        }
    })
}
fn uniform_model(center: f64, amplitude: f64, sigma: f64) -> Value {
    json!({"centers": ([[center.to_bits(); 3]; 3]), "amplitudes": ([[amplitude.to_bits(); 3]; 3]), "sigmas": ([[sigma.to_bits(); 3]; 3])})
}

// This fixed-field composition is only a test consumer of the curve identity.
// The independent native replay supplies all expectations; no recipe is ported.
fn downstream(curve_hash: u64, gamma: u64, expected: &Value) -> [u64; 3] {
    let print = hash::u64s(&[23, 29, 31, curve_hash, gamma, 37]);
    let recipe = hash::u64s(&[
        number(&expected["asset_token"]),
        11,
        number(&expected["normalized_hash"]),
        13,
        17,
        19,
        print,
    ]);
    [print, recipe, hash::u64s(&[recipe, 41, 43, 47])]
}

#[test]
fn bundled_and_floor_gamma_samples_and_downstream_identities_match_captures() {
    let gamma_fixture = fixture("print-gamma.json");
    let density = fixture("density.json");
    let samples =
        fs::read(repository().join("tests/profile/fixtures/density-samples.bin")).unwrap();
    let directory = Directory::new();
    let mut bundled = 0;
    let mut floors = 0;
    let mut sampled = 0;
    for captured in gamma_fixture["captured"].as_array().unwrap() {
        let id = captured["id"].as_str().unwrap();
        let case = density["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap();
        let axis = doubles(&density["axes"][number(&case["axis"]) as usize]);
        let key = case["profile"].as_str().unwrap();
        let profile = if key.is_empty() {
            floors += 1;
            directory.profile(&document(&captured["source_model"], &axis))
        } else {
            bundled += 1;
            PrintProfile::new(
                load_print_source(&repository().join(format!("Resources/profiles/{key}.json")))
                    .unwrap(),
            )
            .unwrap()
        };
        assert_eq!(
            profile
                .tables()
                .source_log_exposure()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            axis.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "{id}"
        );
        let before = model_bits(&profile);
        let token = profile.asset_token();
        let baseline = totals_bits(profile.tables().density_curves());
        let gamma_bits = number(&case["gamma_bits"]);
        let gamma = f64::from_bits(gamma_bits);
        let curves = profile.sample_density_curves(gamma).unwrap();
        let start = number(&case["sample_start"]) as usize;
        let expected: Vec<[u32; 3]> = (start..start + axis.len())
            .map(|i| {
                let row = &samples[i * 49..(i + 1) * 49];
                assert_eq!(row[0], 1);
                std::array::from_fn(|c| {
                    u32::from_le_bytes(row[1 + c * 4..5 + c * 4].try_into().unwrap())
                })
            })
            .collect();
        assert_eq!(totals_bits(curves.totals()), expected, "{id}");
        assert_eq!(
            curves.hash(),
            number(&case["identities"]["print_hash"]),
            "{id}"
        );
        let repeated = profile.sample_density_curves(gamma).unwrap();
        assert_eq!(repeated.hash(), curves.hash(), "{id} reuse");
        assert_eq!(totals_bits(repeated.totals()), expected);
        assert_eq!(model_bits(&profile), before);
        assert_eq!(profile.asset_token(), token);
        assert_eq!(totals_bits(profile.tables().density_curves()), baseline);
        if gamma == 1.0 {
            assert_eq!(expected, baseline, "{id} identity branch");
        }
        let identities = &captured["downstream"];
        assert_eq!(
            downstream(curves.hash(), gamma_bits, identities),
            [
                number(&identities["print_recipe_hash"]),
                number(&identities["recipe_hash"]),
                number(&identities["glare_seed"])
            ],
            "{id} downstream"
        );
        sampled += axis.len();
    }
    assert_eq!((bundled, floors, sampled), (48, 30, 12_438));
}

#[test]
fn unusual_coefficients_succeed_through_print_gamma_without_readmission() {
    let fixtures = fixture("print-gamma.json");
    let directory = Directory::new();
    for case in fixtures["analytical"].as_array().unwrap() {
        let profile = directory.profile(&document(&case["model"], &doubles(&case["axis"])));
        let curves = profile
            .sample_density_curves(f64::from_bits(number(&case["gamma_bits"])))
            .unwrap();
        assert_eq!(
            totals_bits(curves.totals()),
            serde_json::from_value::<Vec<[u32; 3]>>(case["totals"].clone()).unwrap(),
            "{}",
            case["id"]
        );
        assert_eq!(curves.hash(), number(&case["print_hash"]), "{}", case["id"]);
        assert!(curves.totals().iter().flatten().all(|v| v.is_finite()));
    }
}

#[test]
fn gamma_control_boundary_requires_only_finite_positive_values() {
    let directory = Directory::new();
    let profile = directory.profile(&document(&uniform_model(0., 1., 1.), &[0.]));
    for gamma in [0., -0., -1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            profile.sample_density_curves(gamma).unwrap_err(),
            PrintDensityError::InvalidGamma
        );
    }
    // Range admission stays with the recipe, and even an infinite adjusted sigma
    // can produce a finite half response through the approved sampler.
    for gamma in [0.25, 4., f64::from_bits(1), f64::MAX] {
        assert_eq!(
            profile.sample_density_curves(gamma).unwrap().totals(),
            &[[1.5; 3]]
        );
    }
}

#[test]
fn gamma_computed_failures_return_no_partial_curves_and_leave_baseline_usable() {
    let directory = Directory::new();
    let profile = directory.profile(&document(&uniform_model(2e38, 1., 2e38), &[0., 1.]));
    assert_eq!(
        profile.sample_density_curves(0.5).unwrap_err(),
        PrintDensityError::Density {
            index: 0,
            source: DensitySampleError::NonfiniteLayer {
                channel: 0,
                layer: 0
            }
        }
    );
    assert!(profile.sample_density_curves(1.).is_ok());

    let profile = directory.profile(&document(
        &uniform_model(-100., f32::MAX as f64, 1.),
        &[-400., -200.],
    ));
    assert_eq!(
        profile.sample_density_curves(0.5).unwrap_err(),
        PrintDensityError::Density {
            index: 1,
            source: DensitySampleError::NonfiniteTotal { channel: 0 }
        }
    );
    assert_eq!(
        profile.sample_density_curves(1.).unwrap().totals(),
        &[[0.; 3]; 2]
    );
}

#[test]
fn curve_identity_preserves_raw_axis_bits_count_and_sample_dependencies() {
    let directory = Directory::new();
    let model = uniform_model(0., 1., 1.);
    let base_document = document(&model, &[-0., 0.]);
    let base = directory.profile(&base_document);
    let base_curves = base.sample_density_curves(1.).unwrap();
    let changed_axis = directory.profile(&document(&model, &[0., 0.]));
    let changed_count = directory.profile(&document(&model, &[0.]));
    for profile in [&changed_axis, &changed_count] {
        assert_ne!(
            profile.sample_density_curves(1.).unwrap().hash(),
            base_curves.hash()
        );
    }
    let changed_model = directory.profile(&document(&uniform_model(0., 2., 1.), &[-0., 0.]));
    assert_ne!(
        changed_model.sample_density_curves(1.).unwrap().hash(),
        base_curves.hash()
    );
    let mut excluded = base_document.clone();
    excluded["info"]["name"] = json!("Another display label");
    excluded["data"]["channel_density"][0][0] = json!(2.);
    let excluded = directory.profile(&excluded);
    assert_ne!(excluded.asset_token(), base.asset_token());
    assert_eq!(
        excluded.sample_density_curves(1.).unwrap().hash(),
        base_curves.hash()
    );
    // A change of gamma can leave totals and the content identity equal while
    // still changing the downstream recipe/glare seed through its own gamma lane.
    let second = base.sample_density_curves(2.).unwrap();
    assert_eq!(second.hash(), base_curves.hash());
    let fixed = json!({"asset_token":11,"normalized_hash":13});
    let first = downstream(base_curves.hash(), 1_f64.to_bits(), &fixed);
    let second = downstream(second.hash(), 2_f64.to_bits(), &fixed);
    assert!(first.iter().zip(second).all(|(a, b)| *a != b));
    // The opposite polarity changes actual samples away from the half response.
    let mut positive = document(&model, &[-1., 1.]);
    let negative = directory.profile(&positive);
    positive["info"]["type"] = json!("positive");
    let positive = directory.profile(&positive);
    assert_ne!(
        positive.sample_density_curves(1.).unwrap().hash(),
        negative.sample_density_curves(1.).unwrap().hash()
    );
}
