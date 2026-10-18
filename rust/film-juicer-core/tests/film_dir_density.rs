#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use film_juicer_core::assets::Assets;
use film_juicer_core::film::{self, density, dir};
use film_juicer_core::profile::Polarity;
use film_juicer_core::route::ScanRoute;
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture(name: &str) -> Value {
    let platform = if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };
    serde_json::from_slice(
        &fs::read(
            root()
                .join("tests/film/dir-fixtures")
                .join(format!("{platform}{name}.json")),
        )
        .unwrap(),
    )
    .unwrap()
}
fn bits(v: &[f32]) -> Value {
    json!(v.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
}
fn floats<const N: usize>(v: &Value) -> [f32; N] {
    std::array::from_fn(|i| f32::from_bits(v[i].as_u64().unwrap() as u32))
}
fn params(v: &Value) -> dir::Params {
    dir::Params {
        active: v["active"].as_bool().unwrap(),
        amount: floats::<1>(&v["amount"])[0],
        inhibition_same_layer: floats::<2>(&v["inhibition"])[0],
        inhibition_interlayer: floats::<2>(&v["inhibition"])[1],
        diffusion_size_um: floats::<3>(&v["diffusion"])[0],
        diffusion_tail_um: floats::<3>(&v["diffusion"])[1],
        diffusion_tail_weight: floats::<3>(&v["diffusion"])[2],
        gamma_use_stock: v["stock"].as_bool().unwrap(),
        gamma_same_layer_rgb: floats(&v["same"]),
        gamma_interlayer_r_to_gb: floats(&v["red"]),
        gamma_interlayer_g_to_rb: floats(&v["green"]),
        gamma_interlayer_b_to_rg: floats(&v["blue"]),
        langmuir_donor_k_rgb: floats(&v["donor_k"]),
        langmuir_receiver_k_rgb: floats(&v["receiver_k"]),
    }
}
fn route(tag: u64) -> ScanRoute {
    match tag {
        0 => ScanRoute::NegativeDirect,
        1 => ScanRoute::NegativePrint,
        2 => ScanRoute::PositiveDirect,
        3 => ScanRoute::PositivePrint,
        _ => panic!("fixture route"),
    }
}
fn project(recipe: &dir::Recipe) -> Value {
    json!({"polarity":match recipe.polarity() { Some(Polarity::Negative)=>0,Some(Polarity::Positive)=>1,None=>2 },
        "active":recipe.active(),"mode":match recipe.mode() {dir::Mode::Inactive=>0,dir::Mode::NegativeDonorLangmuir=>1,dir::Mode::PositiveReceiverLangmuir=>2},
        "matrix":recipe.matrix_rgb().iter().map(|v|bits(v)).collect::<Vec<_>>(),
        "diffusion":bits(&[recipe.diffusion_size_um(),recipe.diffusion_tail_um(),recipe.diffusion_tail_weight()]),
        "maximum":bits(recipe.density_max_rgb()),"reference":bits(recipe.density_ref_rgb()),"donor_k":bits(recipe.donor_k_rgb()),
        "receiver_c_ref":bits(recipe.receiver_c_ref_rgb()),"receiver_kr":bits(recipe.receiver_kr_rgb()),
        "axes":recipe.compensated_axes_rgb().iter().map(|v|bits(v)).collect::<Vec<_>>(),
        "axes_hash":recipe.compensated_axes_hash(),"hash":recipe.hash()})
}
#[test]
fn exact_native_dir_and_bounds() {
    let assets = Assets::new(root().join("Resources"));
    let captured = fixture("");
    let cases = captured["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 64);
    for case in cases {
        let development = film::Development::new(
            assets.film(case["key"].as_str().unwrap()).unwrap(),
            film::Params {
                gamma_factor: floats::<1>(&case["gamma"])[0],
                layers_required: false,
                requested_route: route(case["route"].as_u64().unwrap()),
            },
        )
        .unwrap();
        let recipe = dir::Recipe::new(&development, params(&case["controls"])).unwrap();
        assert_eq!(project(&recipe), case["dir"], "{}", case["key"]);
        let minimum = density::Minimum::new([0.07, 0.08, 0.12]).unwrap();
        assert_eq!(bits(minimum.cmy()), case["grain_minimum"]);
        let bounds = density::Bounds::new(&development, &minimum).unwrap();
        let v = json!({"route":case["route"],"medium":0,"polarity":if bounds.polarity()==Polarity::Negative{0}else{1},
            "source":if bounds.source()==density::Source::DirectFilm{0}else{1},"minimum":bits(bounds.minimum_cmy()),
            "maximum":bits(bounds.maximum_cmy()),"inverse":bits(bounds.inv_span_cmy()),
            "authored_minimum":bits(bounds.authored_minimum_cmy()),"authored_maximum":bits(bounds.authored_maximum_cmy()),"hash":bounds.hash()});
        assert_eq!(v, case["bounds"]);
        assert_eq!(bounds.route(), development.route());
    }
}

#[test]
fn original_exposure_spans_and_bypasses() {
    let default_controls = fixture("")["cases"][2]["controls"].clone();
    let captured = fixture("-f1");
    for case in captured.as_array().unwrap() {
        let name = case["case"].as_str().unwrap();
        let resources = if matches!(name, "duplicates" | "singleton") {
            root().join("tests/film/fixtures/synthetic").join(name)
        } else {
            root().join("tests/film/dir-fixtures/synthetic").join(name)
        };
        let assets = Assets::new(resources);
        let development = film::Development::new(
            assets.film("kodak_portra_400").unwrap(),
            film::Params {
                gamma_factor: 1.0,
                layers_required: false,
                requested_route: ScanRoute::NegativeDirect,
            },
        )
        .unwrap();
        assert_eq!(bits(development.log_exposure()), case["axis"]);
        let mut controls = params(&default_controls);
        match case["kind"].as_u64().unwrap() {
            1 => controls.active = false,
            2 => controls.amount = 0.0,
            3 => {
                controls.inhibition_same_layer = 0.0;
                controls.inhibition_interlayer = 0.0;
            }
            4 => {
                controls.gamma_use_stock = false;
                controls.gamma_same_layer_rgb[0] = f32::INFINITY;
            }
            5 => controls.langmuir_donor_k_rgb[0] = 0.0,
            6 => controls.amount = -1.0,
            _ => {}
        }
        let result = dir::Recipe::new(&development, controls);
        assert_eq!(result.is_ok(), case["success"].as_bool().unwrap(), "{name}");
        if let Ok(recipe) = result {
            let view = project(&recipe);
            for field in ["polarity", "mode", "axes", "axes_hash", "hash"] {
                assert_eq!(view[field], case[field], "{name} {field}");
            }
        } else {
            let error = result.err().unwrap();
            match case["kind"].as_u64().unwrap() {
                4 => assert!(matches!(error, dir::Error::Matrix)),
                _ if name == "singleton" => assert!(matches!(error, dir::Error::DensityMaximum)),
                5 if name == "finite-wide" => {
                    assert!(matches!(error, dir::Error::DonorK { channel: 0 }))
                }
                _ => assert!(matches!(error, dir::Error::LogExposure { sample: 1 })),
            }
        }
    }
}

#[test]
fn minimum_keeps_zero_sign_and_rejects_before_clamp() {
    assert_eq!(
        bits(density::Minimum::new([-0.0, 0.0, 2.0]).unwrap().cmy()),
        json!([2147483648_u32, 0, 1065353216])
    );
    for value in [-0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            density::Minimum::new([value, 0.0, 0.0]),
            Err(density::Error::Minimum)
        ));
    }
}

#[test]
fn focused_native_conditionals_and_minima() {
    let assets = Assets::new(root().join("Resources"));
    let base = fixture("");
    for case in fixture("-focused").as_array().unwrap() {
        let key = case["key"].as_str().unwrap();
        let selected = if key == "kodak_portra_400" {
            ScanRoute::NegativeDirect
        } else {
            ScanRoute::PositiveDirect
        };
        let development = film::Development::new(
            assets.film(key).unwrap(),
            film::Params {
                gamma_factor: 1.0,
                layers_required: false,
                requested_route: selected,
            },
        )
        .unwrap();
        if let Some(kind) = case["kind"].as_u64() {
            let mut controls = params(&base["cases"][2]["controls"]);
            match kind {
                0 => controls.diffusion_size_um = 0.0,
                1 => controls.diffusion_size_um = -1.0,
                2 => controls.diffusion_tail_weight = 0.0,
                3 => controls.diffusion_tail_weight = -1.0,
                4 => controls.diffusion_tail_um = 123.4567,
                5 => {
                    controls.active = false;
                    controls.amount = f32::NAN;
                }
                6 => controls.amount = f32::NAN,
                7 => controls.langmuir_donor_k_rgb[0] = 0.0,
                8 => controls.langmuir_receiver_k_rgb[0] = 0.0,
                9 => {
                    controls.gamma_use_stock = false;
                    controls.gamma_same_layer_rgb = [0.0; 3];
                    controls.gamma_interlayer_r_to_gb = [0.0; 2];
                    controls.gamma_interlayer_g_to_rb = [0.0; 2];
                    controls.gamma_interlayer_b_to_rg = [0.0; 2];
                }
                10 => {
                    controls.gamma_use_stock = false;
                    controls.gamma_same_layer_rgb[0] = 0.0;
                    controls.gamma_interlayer_g_to_rb[0] = 0.0;
                    controls.gamma_interlayer_b_to_rg[0] = 0.0;
                }
                11 => controls.langmuir_receiver_k_rgb[0] = f32::NAN,
                12 => controls.langmuir_donor_k_rgb[0] = f32::NAN,
                _ => panic!("focused fixture kind"),
            }
            let result = dir::Recipe::new(&development, controls);
            assert_eq!(result.is_ok(), case["success"].as_bool().unwrap());
            if let Ok(recipe) = result {
                let projected = project(&recipe);
                for field in ["matrix", "axes", "hash", "axes_hash", "diffusion"] {
                    assert_eq!(
                        projected[field], case[field],
                        "{key} kind={kind} field={field}"
                    );
                }
            }
        } else {
            let mut requested = [0.07, 0.08, 0.12];
            match case["minimum_kind"].as_u64().unwrap() {
                1 => requested = [-0.0, 0.0, 1.0],
                2 => requested = [2.0, 3.0, 4.0],
                3 => requested[0] = -0.1,
                4 => requested[0] = f32::INFINITY,
                5 => requested[0] = f32::NAN,
                6 => requested = [0.0; 3],
                _ => {}
            }
            let result = density::Minimum::new(requested);
            assert_eq!(result.is_ok(), case["minimum_success"].as_bool().unwrap());
            if let Ok(minimum) = result {
                assert_eq!(bits(minimum.cmy()), case["minimum"]);
                let bounds = density::Bounds::new(&development, &minimum).unwrap();
                assert_eq!(bits(bounds.minimum_cmy()), case["low"]);
                assert_eq!(bits(bounds.maximum_cmy()), case["high"]);
                assert_eq!(bits(bounds.inv_span_cmy()), case["inverse"]);
                assert_eq!(json!(bounds.hash()), case["hash"]);
            }
        }
    }
}

#[cfg(feature = "test-support")]
#[test]
fn spans_precede_knees_and_axis_reservation() {
    let resources = root().join("tests/film/fixtures/synthetic/duplicates");
    let duplicate_assets = Assets::new(resources);
    let duplicate = film::Development::new(
        duplicate_assets.film("kodak_portra_400").unwrap(),
        film::Params {
            gamma_factor: 1.0,
            layers_required: false,
            requested_route: ScanRoute::NegativeDirect,
        },
    )
    .unwrap();
    let assets = Assets::new(root().join("Resources"));
    let ordinary = film::Development::new(
        assets.film("kodak_portra_400").unwrap(),
        film::Params {
            gamma_factor: 1.0,
            layers_required: false,
            requested_route: ScanRoute::NegativeDirect,
        },
    )
    .unwrap();
    let controls = fixture("")["cases"][2]["controls"].clone();
    dir::test_support::reject_next(2);
    let mut matrix = params(&controls);
    matrix.gamma_use_stock = false;
    matrix.gamma_same_layer_rgb[0] = f32::INFINITY;
    assert!(matches!(
        dir::Recipe::new(&duplicate, matrix),
        Err(dir::Error::Matrix)
    ));
    let mut bad_k = params(&controls);
    bad_k.langmuir_donor_k_rgb[0] = 0.0;
    assert!(matches!(
        dir::Recipe::new(&duplicate, bad_k),
        Err(dir::Error::LogExposure { sample: 1 })
    ));
    assert!(matches!(
        dir::Recipe::new(&ordinary, params(&controls)),
        Err(dir::Error::Allocation)
    ));
    for fault in [1, 2, 3] {
        dir::test_support::reject_next(fault);
        let result = dir::Recipe::new(&ordinary, params(&controls));
        assert!(matches!(
            (fault, result),
            (1, Err(dir::Error::Size))
                | (2, Err(dir::Error::Allocation))
                | (3, Err(dir::Error::ZeroIdentity))
        ));
    }
}
