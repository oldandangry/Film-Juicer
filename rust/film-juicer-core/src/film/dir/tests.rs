use super::{self as dir};
use crate::{assets::Assets, film, route::ScanRoute};
use std::path::Path;

#[test]
fn matrix_and_axis_errors_precede_knees() {
    let resources = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/film/development/synthetic/duplicates");
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
    let controls = dir::capture_defaults(true, false);
    let mut matrix = controls;
    matrix.gamma_same_layer_rgb[0] = f32::INFINITY;
    assert!(matches!(
        dir::Recipe::new(&duplicate, matrix),
        Err(dir::Error::Matrix)
    ));

    let mut bad_k = controls;
    bad_k.langmuir_donor_k_rgb[0] = 0.0;
    assert!(matches!(
        dir::Recipe::new(&duplicate, bad_k),
        Err(dir::Error::LogExposure { sample: 1 })
    ));
}
