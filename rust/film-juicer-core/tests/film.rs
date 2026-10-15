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
    for case in fixture["cases"].as_array().unwrap() {
        let key = case["key"].as_str().unwrap();
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
    }
    assert_eq!(count, 480);
    let profile = assets.film("kodak_portra_400").unwrap();
    for gamma in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        0.049,
        4.001,
    ] {
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
    }
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
    for case in captured.as_array().unwrap() {
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
        }
    }
}
