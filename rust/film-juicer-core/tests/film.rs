//! Exact native development captures, including retained profile ownership.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use film_juicer_core::assets::Assets;
use film_juicer_core::film::{Development, Error, Params};
use film_juicer_core::route::ScanRoute;
use serde_json::{Value, json};

fn bits(values: &[f32]) -> Value {
    json!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    )
}
fn rows(values: &[[f32; 3]]) -> Value {
    json!(
        values
            .iter()
            .map(|row| row.map(f32::to_bits))
            .collect::<Vec<_>>()
    )
}
fn route(tag: u64) -> ScanRoute {
    match tag {
        0 => ScanRoute::NegativeDirect,
        1 => ScanRoute::NegativePrint,
        2 => ScanRoute::PositiveDirect,
        3 => ScanRoute::PositivePrint,
        _ => panic!("capture tag"),
    }
}
fn development_case_id(case: &Value) -> String {
    format!(
        "{}/route-{}/gamma-{:016x}/grain-{}",
        case["key"].as_str().unwrap(),
        case["route"].as_u64().unwrap(),
        case["gamma_double"].as_u64().unwrap(),
        case["grain_policy"].as_u64().unwrap()
    )
}
#[test]
fn matches_native_captures_and_retains_the_profile() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
        (true, true) => "windows-clang-debug",
        (true, false) => "windows-clang-release",
        (false, true) => "linux-debug",
        (false, false) => "linux-release",
    };
    let args = [
        PathBuf::new(),
        root.join("Resources"),
        root.join("tests/film/fixtures")
            .join(format!("{preset}.json")),
    ];
    let assets = Assets::new(PathBuf::from(&args[1]));
    let fixture: Value = serde_json::from_slice(&fs::read(&args[2]).unwrap()).unwrap();
    let mut count = 0;
    let cases = fixture["cases"].as_array().unwrap();
    let ledger_cases_ids: Vec<_> = IntoIterator::into_iter(cases.iter().map(development_case_id))
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !ledger_cases_ids.is_empty(),
        "empty case set: {}",
        "native-development-cases"
    );
    let ledger_cases_count = ledger_cases_ids.len();
    let mut ledger_cases_remaining = ledger_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        ledger_cases_remaining.len(),
        ledger_cases_count,
        "duplicate case IDs: {}",
        "native-development-cases"
    );
    for case in cases {
        let key = case["key"].as_str().unwrap();
        let row = AsRef::<str>::as_ref(&(development_case_id(case))).to_owned();
        let profile = assets.film(key).unwrap();
        let expected = &case["development"];
        let table = &fixture["profiles"][key];
        let params = Params {
            gamma_factor: f64::from_bits(case["gamma_double"].as_u64().unwrap()) as f32,
            layers_required: expected["layers_required"].as_bool().unwrap(),
            requested_route: route(case["route"].as_u64().unwrap()),
        };
        let development = Development::new(profile, params).unwrap();
        assert_eq!(
            bits(development.log_exposure()),
            table["axis"],
            "axis {key}"
        );
        assert_eq!(
            rows(development.authored_curves_cmy()),
            table["authored"],
            "authored {key}"
        );
        assert_eq!(
            rows(development.normalized_curves_cmy()),
            table["normalized"],
            "normalized {key}"
        );
        assert_eq!(
            bits(development.minimum_cmy()),
            table["minimum"],
            "minimum {key}"
        );
        assert_eq!(
            bits(development.maximum_cmy()),
            table["maximum"],
            "maximum {key}"
        );
        assert_eq!(
            bits(development.gamma_rgb()),
            expected["gamma"],
            "gamma {key}"
        );
        assert_eq!(
            json!(development.normalized_curves_hash()),
            table["normalized_hash"],
            "normalized identity {key}"
        );
        assert_eq!(
            json!(development.layers_hash()),
            expected["layers_hash"],
            "layer identity {key}"
        );
        assert_eq!(
            json!(development.hash()),
            expected["hash"],
            "development identity {key}"
        );
        assert_eq!(development.route(), route(case["route"].as_u64().unwrap()));
        if let Some(layers) = development.density_curves_layers() {
            for layer in 0..3 {
                for channel in 0..3 {
                    assert_eq!(
                        bits(&layers[layer][channel]),
                        table["layers"][layer][channel],
                        "layer lane {key}"
                    );
                }
            }
        } else {
            assert!(!development.layers_required());
            assert_eq!(development.layers_hash(), 0);
        }
        count += 1;
        assert!(
            ledger_cases_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        ledger_cases_remaining.is_empty(),
        "missing comparisons: {ledger_cases_remaining:?}"
    );
    assert_eq!(count, 480);
    let profile = assets.film("kodak_portra_400").unwrap();
    let invalid_gammas = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        0.049,
        4.001,
    ];
    let gamma_ids = [
        "nan",
        "+infinity",
        "-infinity",
        "zero",
        "below-min",
        "above-max",
    ];
    let invalid_rows_ids: Vec<_> = IntoIterator::into_iter(gamma_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !invalid_rows_ids.is_empty(),
        "empty case set: {}",
        "invalid-gamma-controls"
    );
    let invalid_rows_count = invalid_rows_ids.len();
    let mut invalid_rows_remaining = invalid_rows_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        invalid_rows_remaining.len(),
        invalid_rows_count,
        "duplicate case IDs: {}",
        "invalid-gamma-controls"
    );
    for (gamma, id) in invalid_gammas.into_iter().zip(gamma_ids) {
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        assert!(matches!(
            Development::new(
                profile.clone(),
                Params {
                    gamma_factor: gamma,
                    layers_required: false,
                    requested_route: ScanRoute::NegativeDirect
                }
            ),
            Err(Error::InvalidGamma)
        ));
        assert!(
            invalid_rows_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        invalid_rows_remaining.is_empty(),
        "missing comparisons: {invalid_rows_remaining:?}"
    );
    let route_row_ids: Vec<_> = IntoIterator::into_iter(["positive-route-on-negative-film"])
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !route_row_ids.is_empty(),
        "empty case set: {}",
        "route-polarity-mismatch"
    );
    let route_row_count = route_row_ids.len();
    let mut route_row_remaining = route_row_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        route_row_remaining.len(),
        route_row_count,
        "duplicate case IDs: {}",
        "route-polarity-mismatch"
    );
    let mismatch = AsRef::<str>::as_ref(&("positive-route-on-negative-film")).to_owned();
    assert!(matches!(
        Development::new(
            profile,
            Params {
                gamma_factor: 1.0,
                layers_required: false,
                requested_route: ScanRoute::PositiveDirect
            }
        ),
        Err(Error::RouteMismatch)
    ));
    assert!(
        route_row_remaining.remove(&mismatch),
        "unexpected or repeated comparison: {mismatch}"
    );
    assert!(
        route_row_remaining.is_empty(),
        "missing comparisons: {route_row_remaining:?}"
    );
    let retained = {
        let temporary = Assets::new(PathBuf::from(&args[1]));
        Development::new(
            temporary.film("kodak_portra_400").unwrap(),
            Params {
                gamma_factor: 1.0,
                layers_required: true,
                requested_route: ScanRoute::NegativeDirect,
            },
        )
        .unwrap()
    };
    assert_eq!(
        rows(retained.normalized_curves_cmy()),
        fixture["profiles"]["kodak_portra_400"]["normalized"]
    );
    println!(
        "PASS {count} independent native development cases and invalid-control/owner-survival witnesses"
    );
}

#[test]
fn variable_axes_and_reservation_identity_failures() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
        (true, true) => "windows-clang-debug",
        (true, false) => "windows-clang-release",
        (false, true) => "linux-debug",
        (false, false) => "linux-release",
    };
    let fixtures = root.join("tests/film/fixtures");
    let captured: Value = serde_json::from_slice(
        &fs::read(fixtures.join(format!("{preset}-synthetic.json"))).unwrap(),
    )
    .unwrap();
    let cases = captured.as_array().unwrap();
    let ids: Vec<_> = cases
        .iter()
        .map(|case| case["case"].as_str().unwrap().to_owned())
        .collect();
    let ledger_cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !ledger_cases_ids.is_empty(),
        "empty case set: {}",
        "synthetic-development-cases"
    );
    let ledger_cases_count = ledger_cases_ids.len();
    let mut ledger_cases_remaining = ledger_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        ledger_cases_remaining.len(),
        ledger_cases_count,
        "duplicate case IDs: {}",
        "synthetic-development-cases"
    );
    #[cfg(feature = "test-support")]
    let mut fault_ids = Vec::new();
    #[cfg(feature = "test-support")]
    for id in &ids {
        for fault in 1..=3 {
            fault_ids.push(format!("{id}/reservation-fault/{fault}"));
        }
    }
    #[cfg(feature = "test-support")]
    let mut fault_rows_remaining = {
        let expected = fault_ids.len();
        let remaining = fault_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(remaining.len(), expected, "duplicate reservation fault IDs");
        remaining
    };
    for case in cases {
        let id = case["case"].as_str().unwrap();
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        let assets = Assets::new(
            fixtures
                .join("synthetic")
                .join(case["case"].as_str().unwrap()),
        );
        let profile = assets.film("kodak_portra_400").unwrap();
        let params = || Params {
            gamma_factor: 1.0,
            layers_required: true,
            requested_route: ScanRoute::NegativeDirect,
        };
        let development = Development::new(profile.clone(), params()).unwrap();
        assert_eq!(bits(development.log_exposure()), case["axis"]);
        assert_eq!(rows(development.authored_curves_cmy()), case["authored"]);
        assert_eq!(
            rows(development.normalized_curves_cmy()),
            case["normalized"]
        );
        assert_eq!(bits(development.minimum_cmy()), case["minimum"]);
        assert_eq!(bits(development.maximum_cmy()), case["maximum"]);
        assert_eq!(json!(development.normalized_curves_hash()), case["hash"]);
        #[cfg(feature = "test-support")]
        for fault in 1..=3 {
            let fault_row =
                AsRef::<str>::as_ref(&(format!("{id}/reservation-fault/{fault}"))).to_owned();
            film_juicer_core::film::test_support::reject_next(fault);
            let error = Development::new(profile.clone(), params()).err().unwrap();
            assert_eq!(
                error,
                if fault == 1 {
                    Error::Allocation
                } else {
                    Error::ZeroIdentity
                }
            );
            assert!(
                fault_rows_remaining.remove(&fault_row),
                "unexpected or repeated comparison: {fault_row}"
            );
        }
        assert!(
            ledger_cases_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        ledger_cases_remaining.is_empty(),
        "missing comparisons: {ledger_cases_remaining:?}"
    );
    #[cfg(feature = "test-support")]
    assert!(
        fault_rows_remaining.is_empty(),
        "missing comparisons: {fault_rows_remaining:?}"
    );
}
