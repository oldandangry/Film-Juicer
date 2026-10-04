//! Complete-profile contracts use frozen native fields/processing defaults, qualified native
//! sensitivity and approved density bits with independent native identity replay.
//! Synthetic token expectations were captured before the completion encoder.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;
use serde_json::{Value, json};

use film_juicer_core::profile::{
    DensityCurveModel, DensitySampleError, FilmProcessingDefaults, FilmProfile, PrintProfile,
    ProfileCompletionError, ProfileCompletionErrorKind, ProfileInfo, ProfileSource, ProfileTables,
    Role, load_catalog, load_film_source, load_print_source,
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

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::var_os("JUICER_TEST_ARTIFACT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| repository().join("out/validation/rust-core"));
        let path = root.join(format!(
            "sampled-profile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn source(&self, document: &Value, role: Role) -> ProfileSource {
        self.source_text(&document.to_string(), role)
    }
    fn source_text(&self, text: &str, role: Role) -> ProfileSource {
        let path = self.0.join("source.json");
        fs::write(&path, text).unwrap();
        match role {
            Role::Film => load_film_source(&path),
            Role::Print => load_print_source(&path),
        }
        .unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

enum Completed {
    Film(FilmProfile),
    Print(PrintProfile),
}
impl Completed {
    fn new(source: ProfileSource, role: Role) -> Result<Self, ProfileCompletionError> {
        match role {
            Role::Film => FilmProfile::new(source).map(Self::Film),
            Role::Print => PrintProfile::new(source).map(Self::Print),
        }
    }
    fn tables(&self) -> &ProfileTables {
        match self {
            Self::Film(p) => p.tables(),
            Self::Print(p) => p.tables(),
        }
    }
    fn info(&self) -> &ProfileInfo {
        match self {
            Self::Film(p) => p.info(),
            Self::Print(p) => p.info(),
        }
    }
    fn model(&self) -> &DensityCurveModel {
        match self {
            Self::Film(p) => p.density_model(),
            Self::Print(p) => p.density_model(),
        }
    }
    fn token(&self) -> u64 {
        match self {
            Self::Film(p) => p.asset_token(),
            Self::Print(p) => p.asset_token(),
        }
    }
}

fn bits(samples: impl IntoIterator<Item = f32>) -> Vec<u32> {
    samples.into_iter().map(f32::to_bits).collect()
}
fn expected_bits(node: &Value) -> Vec<u32> {
    serde_json::from_value(node.clone()).unwrap()
}
fn assert_processing_defaults(actual: &FilmProcessingDefaults, expected: &Value) {
    for (name, samples) in [
        ("gamma_samelayer_rgb", actual.gamma_samelayer_rgb.as_slice()),
        (
            "gamma_interlayer_r_to_gb",
            actual.gamma_interlayer_r_to_gb.as_slice(),
        ),
        (
            "gamma_interlayer_g_to_rb",
            actual.gamma_interlayer_g_to_rb.as_slice(),
        ),
        (
            "gamma_interlayer_b_to_rg",
            actual.gamma_interlayer_b_to_rg.as_slice(),
        ),
        (
            "halation_first_sigma_um",
            actual.halation_first_sigma_um.as_slice(),
        ),
        (
            "halation_primary_amount",
            actual.halation_primary_amount.as_slice(),
        ),
    ] {
        assert_eq!(
            bits(samples.iter().copied()),
            expected_bits(&expected[name]),
            "{name}"
        );
    }
    assert_eq!(
        u64::from(actual.hanatos_spectral_gaussian_blur_default.to_bits()),
        expected["hanatos_spectral_gaussian_blur_default"]
            .as_u64()
            .unwrap()
    );
}

#[derive(Deserialize)]
struct ModelBits {
    centers: [[u64; 3]; 3],
    amplitudes: [[u64; 3]; 3],
    sigmas: [[u64; 3]; 3],
}
#[derive(Deserialize)]
struct DensityCase {
    id: String,
    axis: usize,
    model: ModelBits,
    sample_start: Option<usize>,
}
#[derive(Deserialize)]
struct DensityFixture {
    axes: Vec<Vec<u64>>,
    cases: Vec<DensityCase>,
}

#[test]
fn all_bundled_profiles_match_independent_completed_fields_and_approved_density() {
    let expected = fixture("completed.json");
    let density: DensityFixture = serde_json::from_value(fixture("density.json")).unwrap();
    let payload =
        fs::read(repository().join("tests/profile/fixtures/density-samples.bin")).unwrap();
    let catalog = load_catalog(&repository().join("Resources")).unwrap();
    let mut count = 0;
    for (role, entries) in [
        (Role::Film, catalog.films()),
        (Role::Print, catalog.prints()),
    ] {
        for entry in entries {
            let key = entry.key();
            let source = match role {
                Role::Film => load_film_source(entry.source_path()),
                Role::Print => load_print_source(entry.source_path()),
            }
            .unwrap();
            let axis_ptr = source.samples().log_exposure().as_ptr();
            let stock_ptr = source.info().stock().as_ptr();
            let completed = Completed::new(source, role).unwrap();
            assert_eq!(completed.info().stock(), key);
            assert_eq!(completed.info().name(), entry.label());
            assert_eq!(
                (
                    completed.info().support(),
                    completed.info().stage(),
                    completed.info().polarity()
                ),
                (entry.support(), entry.stage(), entry.polarity())
            );
            assert_eq!(
                completed.info().stock().as_ptr(),
                stock_ptr,
                "stock moves without allocation"
            );
            let tables = completed.tables();
            assert_eq!(
                tables.authored_log_exposure().as_ptr(),
                axis_ptr,
                "f64 axis moves without allocation"
            );
            let fields = &expected["profiles"][key]["fields"];
            for (name, samples) in [
                ("wavelengths", bits(*tables.wavelengths())),
                (
                    "log_sensitivity",
                    bits(tables.log_sensitivity().iter().flatten().copied()),
                ),
                (
                    "channel_density",
                    bits(tables.channel_density().iter().flatten().copied()),
                ),
                ("base_density", bits(*tables.base_density())),
                (
                    "log_exposure",
                    bits(tables.interpolation_log_exposure().iter().copied()),
                ),
            ] {
                assert_eq!(samples, expected_bits(&fields[name]), "{key} {name}");
            }
            let linear = if cfg!(windows) {
                "linear_windows"
            } else {
                "linear_linux"
            };
            assert_eq!(
                bits(tables.linear_sensitivity().iter().flatten().copied()),
                expected_bits(&expected["profiles"][key][linear]),
                "{key} sensitivity"
            );
            let window = tables
                .hanatos2025_adaptation_window_params()
                .map(|v| bits(*v));
            let surface = tables
                .hanatos2025_adaptation_surface_params()
                .map(|v| bits(v.iter().flatten().copied()));
            assert_eq!(
                window,
                serde_json::from_value::<Option<Vec<u32>>>(fields["window"].clone()).unwrap(),
                "{key} window"
            );
            assert_eq!(
                surface,
                serde_json::from_value::<Option<Vec<u32>>>(fields["surface"].clone()).unwrap(),
                "{key} surface"
            );
            let case = density
                .cases
                .iter()
                .find(|c| c.id == format!("{key}/gamma-1.0"))
                .unwrap();
            assert_eq!(
                tables
                    .authored_log_exposure()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                density.axes[case.axis],
                "{key} f64 axis"
            );
            assert_eq!(
                completed.model().centers().map(|r| r.map(f64::to_bits)),
                case.model.centers
            );
            assert_eq!(
                completed.model().amplitudes().map(|r| r.map(f64::to_bits)),
                case.model.amplitudes
            );
            assert_eq!(
                completed.model().sigmas().map(|r| r.map(f64::to_bits)),
                case.model.sigmas
            );
            assert_eq!(
                tables.density_curves().len(),
                tables.interpolation_log_exposure().len()
            );
            for (i, total) in tables.density_curves().iter().enumerate() {
                let offset = (case.sample_start.unwrap() + i) * 49;
                assert_eq!(payload[offset], 1);
                let row: [u32; 12] = std::array::from_fn(|j| {
                    u32::from_le_bytes(
                        payload[offset + 1 + j * 4..offset + 5 + j * 4]
                            .try_into()
                            .unwrap(),
                    )
                });
                assert_eq!(total.map(f32::to_bits), row[..3], "{key} total {i}");
                for layer in 0..3 {
                    for channel in 0..3 {
                        let curve = &tables.density_curves_layers()[layer][channel];
                        assert_eq!(curve.len(), tables.interpolation_log_exposure().len());
                        assert_eq!(
                            curve[i].to_bits(),
                            row[3 + layer * 3 + channel],
                            "{key} layer {layer} channel {channel} row {i}"
                        );
                    }
                }
            }
            assert_eq!(
                completed.token(),
                expected["profiles"][key]["asset_token"].as_u64().unwrap(),
                "{key} native replay token"
            );
            if let Completed::Film(film) = &completed {
                assert_processing_defaults(
                    film.processing_defaults(),
                    &expected["profiles"][key]["digest"],
                );
            }
            count += 1;
        }
    }
    assert_eq!(count, 28);
}

fn case_document(fixture: &Value, case: &Value) -> Value {
    let mut document = fixture["base"].clone();
    for change in case["changes"].as_array().unwrap() {
        let path = change["path"].as_str().unwrap();
        if document.pointer(path).is_some() {
            *document.pointer_mut(path).unwrap() = change["value"].clone();
        } else {
            let (parent, key) = path.rsplit_once('/').unwrap();
            document.pointer_mut(parent).unwrap()[key] = change["value"].clone();
        }
    }
    document
}

#[test]
fn synthetic_completions_match_frozen_identity_composition_and_digests() {
    let expected = fixture("completed.json");
    let directory = Directory::new();
    for case in expected["cases"].as_array().unwrap() {
        let document = case_document(&expected, case);
        let role = if document["info"]["stage"] == "printing" {
            Role::Print
        } else {
            Role::Film
        };
        let completed = Completed::new(directory.source(&document, role), role).unwrap();
        assert_eq!(
            completed.token(),
            case["asset_token"].as_u64().unwrap(),
            "{} token",
            case["id"]
        );
        assert_eq!(
            bits(
                completed
                    .tables()
                    .density_curves()
                    .iter()
                    .flatten()
                    .copied()
            ),
            expected_bits(&case["totals"]),
            "{} exact analytic totals",
            case["id"]
        );
        if let Completed::Film(film) = &completed {
            assert_processing_defaults(film.processing_defaults(), &case["digest"]);
        }
    }
}

#[test]
fn interpolation_structure_is_established_only_after_f32_narrowing() {
    let base = fixture("completed.json")["base"].clone();
    let directory = Directory::new();
    for role in [Role::Film, Role::Print] {
        for axis in [
            vec![0.0_f64],
            vec![-1.0, -0.0, 0.0, 0.0, 1.0],
            vec![1.00000002, 1.00000001],
            vec![-1e40, -1e40, 0.0, 1e40, 1e40],
        ] {
            let mut document = base.clone();
            if role == Role::Print {
                document["info"]["stage"] = json!("printing");
            }
            document["data"]["log_exposure"] = json!(axis);
            let profile = Completed::new(directory.source(&document, role), role).unwrap();
            let tables = profile.tables();
            assert_eq!(
                tables
                    .authored_log_exposure()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                axis.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                bits(tables.interpolation_log_exposure().iter().copied()),
                bits(axis.iter().map(|&v| v as f32))
            );
            assert_eq!(tables.density_curves().len(), axis.len());
            assert!(
                tables
                    .density_curves_layers()
                    .iter()
                    .flatten()
                    .all(|c| c.len() == axis.len())
            );
            if axis[0] < -1e39 {
                assert_eq!(tables.density_curves()[0], [0.0; 3]);
                assert_eq!(tables.density_curves()[axis.len() - 1], [6.0; 3]);
            }
        }
        let mut document = base.clone();
        if role == Role::Print {
            document["info"]["stage"] = json!("printing");
        }
        document["data"]["log_exposure"] = json!([1.0, 0.0]);
        let source = directory.source(&document, role);
        assert_eq!(source.samples().log_exposure(), [1.0, 0.0]);
        let error = Completed::new(source, role).err().unwrap();
        assert_eq!(
            error.kind,
            ProfileCompletionErrorKind::DescendingAxis { index: 1 }
        );
        assert_eq!(error.stock, "authored");
        assert_eq!(error.role, role);
    }
}

#[test]
fn unusual_models_reach_complete_film_and_print_sampling() {
    let base = fixture("completed.json")["base"].clone();
    let directory = Directory::new();
    for role in [Role::Film, Role::Print] {
        for (field, value, exposure, total) in [
            ("sigmas", -1.0_f64, 0.0, 3.0),
            ("sigmas", 0.0, 1.0, 6.0),
            ("sigmas", 1e-50, 1.0, 6.0),
            ("sigmas", 1e40, 0.0, 3.0),
            ("centers", 1e40, 0.0, 0.0),
        ] {
            let mut document = base.clone();
            if role == Role::Print {
                document["info"]["stage"] = json!("printing");
            }
            document["data"]["log_exposure"] = json!([exposure]);
            document["data"]["density_curves_model"][field] = json!(([[value; 3]; 3]));
            let profile = Completed::new(directory.source(&document, role), role).unwrap();
            assert_eq!(profile.tables().density_curves(), [[total; 3]]);
            let coefficients = if field == "sigmas" {
                profile.model().sigmas()
            } else {
                profile.model().centers()
            };
            assert!(
                coefficients
                    .iter()
                    .flatten()
                    .all(|v| v.to_bits() == value.to_bits())
            );
        }
    }
}

#[test]
fn completion_reports_computed_failure_after_prior_successful_samples() {
    let directory = Directory::new();
    for role in [Role::Film, Role::Print] {
        for (field, value, axis, expected) in [
            (
                "sigmas",
                0.0,
                vec![-1.0, 0.0, 1.0],
                ProfileCompletionErrorKind::Density {
                    index: 1,
                    source: DensitySampleError::NonfiniteLayer {
                        channel: 0,
                        layer: 0,
                    },
                },
            ),
            (
                "amplitudes",
                1e40,
                vec![0.0],
                ProfileCompletionErrorKind::Density {
                    index: 0,
                    source: DensitySampleError::NonfiniteLayer {
                        channel: 0,
                        layer: 0,
                    },
                },
            ),
            (
                "amplitudes",
                3e38,
                vec![-1e40, 1e40],
                ProfileCompletionErrorKind::Density {
                    index: 1,
                    source: DensitySampleError::NonfiniteTotal { channel: 0 },
                },
            ),
        ] {
            let mut document = fixture("completed.json")["base"].clone();
            if role == Role::Print {
                document["info"]["stage"] = json!("printing");
            }
            document["data"]["log_exposure"] = json!(axis);
            document["data"]["density_curves_model"][field] = json!(([[value; 3]; 3]));
            let error = Completed::new(directory.source(&document, role), role)
                .err()
                .unwrap();
            assert_eq!(error.kind, expected);
        }
    }
}

#[test]
fn optional_adaptation_and_null_samples_survive_completion() {
    let directory = Directory::new();
    for role in [Role::Film, Role::Print] {
        let mut document = fixture("completed.json")["base"].clone();
        if role == Role::Print {
            document["info"]["stage"] = json!("printing");
        }
        let baseline = Completed::new(directory.source(&document, role), role)
            .unwrap()
            .token();
        for missing in [Value::Array(vec![]), Value::Null] {
            if role == Role::Film && missing.is_null() {
                continue;
            }
            for key in [
                "hanatos2025_adaptation_window_params",
                "hanatos2025_adaptation_surface_params",
            ] {
                document["data"][key] = missing.clone();
            }
            let completed = Completed::new(directory.source(&document, role), role).unwrap();
            assert!(
                completed
                    .tables()
                    .hanatos2025_adaptation_window_params()
                    .is_none()
            );
            assert!(
                completed
                    .tables()
                    .hanatos2025_adaptation_surface_params()
                    .is_none()
            );
            assert_eq!(completed.token(), baseline);
            assert!(completed.tables().log_sensitivity()[0][0].is_nan());
            assert!(completed.tables().base_density().iter().all(|v| v.is_nan()));
            assert_eq!(completed.tables().linear_sensitivity()[0], [0.0, 1.0, 10.0]);
        }
        document["data"]["hanatos2025_adaptation_window_params"] =
            json!([380.0, -100.0, 780.0, 1e40]);
        document["data"]["hanatos2025_adaptation_surface_params"] = json!(([[1e40; 15]; 3]));
        document["data"]["log_sensitivity"][0] = json!([1e40, -1e40, 40.0]);
        let completed = Completed::new(directory.source(&document, role), role).unwrap();
        assert_eq!(
            completed
                .tables()
                .hanatos2025_adaptation_window_params()
                .unwrap()[1],
            -100.0
        );
        assert!(
            completed
                .tables()
                .hanatos2025_adaptation_window_params()
                .unwrap()[3]
                .is_infinite()
        );
        assert!(
            completed
                .tables()
                .hanatos2025_adaptation_surface_params()
                .unwrap()
                .iter()
                .flatten()
                .all(|v| v.is_infinite())
        );
        assert_eq!(completed.tables().linear_sensitivity()[0], [0.0; 3]);
    }
}

#[test]
fn equivalent_serializations_preserve_identity_and_authored_labels_do_not_reorder_samples() {
    let directory = Directory::new();
    let mut base = fixture("completed.json")["base"].clone();
    // Distinct positions make a resampling/reordering regression observable.
    base["data"]["channel_density"][0] = json!([1.0, 2.0, 3.0]);
    base["data"]["channel_density"][80] = json!([4.0, 5.0, 6.0]);
    base["data"]["log_sensitivity"][0] = json!([0.0, 1.0, 2.0]);
    base["data"]["log_sensitivity"][80] = json!([2.0, 1.0, 0.0]);
    let first = FilmProfile::new(directory.source(&base, Role::Film)).unwrap();
    let reordered = format!(
        "{{\n\"data\": {},\n\"info\": {}\n}}",
        serde_json::to_string_pretty(&base["data"]).unwrap(),
        serde_json::to_string_pretty(&base["info"]).unwrap()
    );
    let second = FilmProfile::new(directory.source_text(&reordered, Role::Film)).unwrap();
    assert_eq!(first.asset_token(), second.asset_token());
    let mut shifted = base.clone();
    shifted["data"]["wavelengths"] = json!((381..=781).step_by(5).collect::<Vec<_>>());
    let shifted = FilmProfile::new(directory.source(&shifted, Role::Film)).unwrap();
    assert_ne!(first.asset_token(), shifted.asset_token());
    assert_eq!(shifted.tables().wavelengths()[0], 381.0);
    assert_eq!(shifted.tables().wavelengths()[80], 781.0);
    assert_eq!(
        bits(first.tables().channel_density().iter().flatten().copied()),
        bits(shifted.tables().channel_density().iter().flatten().copied())
    );
    assert_eq!(
        first.tables().linear_sensitivity(),
        shifted.tables().linear_sensitivity()
    );
    // The first and last samples still occupy computation's 380/780 nm positions.
    // Spectral preparation itself remains with its native owner in this slice.
    assert_eq!(shifted.tables().channel_density()[0], [1.0, 2.0, 3.0]);
    assert_eq!(shifted.tables().channel_density()[80], [4.0, 5.0, 6.0]);
    assert_eq!(shifted.tables().linear_sensitivity()[0], [1.0, 10.0, 100.0]);
    assert_eq!(
        shifted.tables().linear_sensitivity()[80],
        [100.0, 10.0, 1.0]
    );
}

#[test]
fn identity_retains_f64_distinctions_and_canonicalizes_only_f32_zero_signs() {
    let expected = fixture("completed.json");
    let directory = Directory::new();
    let baseline = FilmProfile::new(directory.source(&expected["base"], Role::Film)).unwrap();
    for id in [
        "axis-f64",
        "axis-minus-zero",
        "center-f64",
        "center-minus-zero",
        "amplitude-f64",
        "sigma-f64",
    ] {
        let case = expected["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap();
        let changed =
            FilmProfile::new(directory.source(&case_document(&expected, case), Role::Film))
                .unwrap();
        assert_ne!(changed.asset_token(), baseline.asset_token(), "{id}");
        assert_eq!(
            changed.tables().interpolation_log_exposure(),
            baseline.tables().interpolation_log_exposure()
        );
        assert_eq!(
            changed.tables().density_curves(),
            baseline.tables().density_curves()
        );
        assert_eq!(
            changed
                .density_model()
                .centers()
                .map(|r| r.map(|v| v as f32)),
            baseline
                .density_model()
                .centers()
                .map(|r| r.map(|v| v as f32))
        );
        assert_eq!(
            changed
                .density_model()
                .amplitudes()
                .map(|r| r.map(|v| v as f32)),
            baseline
                .density_model()
                .amplitudes()
                .map(|r| r.map(|v| v as f32))
        );
        assert_eq!(
            changed
                .density_model()
                .sigmas()
                .map(|r| r.map(|v| v as f32)),
            baseline
                .density_model()
                .sigmas()
                .map(|r| r.map(|v| v as f32))
        );
    }
    let mut changed = expected["base"].clone();
    changed["data"]["log_sensitivity"][0][1] = json!(0.0);
    changed["data"]["channel_density"][0][0] = json!(-0.0);
    let zero = FilmProfile::new(directory.source(&changed, Role::Film)).unwrap();
    assert_eq!(zero.asset_token(), baseline.asset_token());
    assert_eq!(
        zero.tables().channel_density()[0][0].to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        baseline.tables().log_sensitivity()[0][1].to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        zero.tables().log_sensitivity()[0][1].to_bits(),
        0.0_f32.to_bits()
    );
    changed["data"]["log_sensitivity"][0][1] = Value::Null;
    let missing = FilmProfile::new(directory.source(&changed, Role::Film)).unwrap();
    assert_ne!(missing.asset_token(), zero.asset_token());
    changed["data"]["log_sensitivity"][0][1] = json!(-1e40);
    let negative_infinity = FilmProfile::new(directory.source(&changed, Role::Film)).unwrap();
    changed["data"]["log_sensitivity"][0][1] = json!(1e40);
    let positive_infinity = FilmProfile::new(directory.source(&changed, Role::Film)).unwrap();
    assert_ne!(
        negative_infinity.asset_token(),
        positive_infinity.asset_token()
    );
    assert_eq!(negative_infinity.tables().linear_sensitivity()[0][1], 0.0);
    assert_eq!(positive_infinity.tables().linear_sensitivity()[0][1], 0.0);
}

#[test]
fn completed_kind_rejects_a_source_loaded_for_the_other_role() {
    let directory = Directory::new();
    let mut document = fixture("completed.json")["base"].clone();
    let source = directory.source(&document, Role::Film);
    assert_eq!(
        PrintProfile::new(source).unwrap_err().kind,
        ProfileCompletionErrorKind::SelectedRole
    );
    document["info"]["stage"] = json!("printing");
    let source = directory.source(&document, Role::Print);
    assert_eq!(
        FilmProfile::new(source).unwrap_err().kind,
        ProfileCompletionErrorKind::SelectedRole
    );
}
