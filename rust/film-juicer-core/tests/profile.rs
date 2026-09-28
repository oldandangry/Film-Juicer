//! Film-Juicer characterization of authored input, before density evaluation.
//! Expected bits/order were captured from the accepted C++ readers at
//! ed3b759a5e3c419ef09b3db694cb220e53e0844e before the Rust translation.
//! Bundle digest: deaddd9e79004008ff0bce40ba48de616d23456cf026c554bb9b111224744773.
//! Spektrafilm provenance: 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc.
//! Ordinary tests use only shipped resources and the independent constants here.
//! Product-contract exceptions: JSON negative zero retains its sign; blackbody
//! temperatures use the original finite positive decimal suffix, without hex.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use film_juicer_core::profile::{
    Antihalation, CatalogEntryErrorKind, CatalogError, ChannelModel, Polarity, ProfileError,
    ProfileErrorKind, ProfileSource, ProfileUse, Requirement, Role, Stage, Support, load_catalog,
    load_film_source, load_print_source,
};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::var_os("JUICER_TEST_ARTIFACT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| repository().join("out/validation/rust-core"));
        let path = root.join(format!(
            "profile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, text: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, text).unwrap();
        path
    }
    fn source(&self, document: &Value, role: Role) -> Result<ProfileSource, ProfileError> {
        let path = self.write("source.json", serde_json::to_vec(document).unwrap());
        match role {
            Role::Film => load_film_source(&path),
            Role::Print => load_print_source(&path),
        }
    }
    fn entry(&self, filename: &str, info: Value) {
        self.write(
            &format!("profiles/{filename}"),
            json!({"info":info,"data":{"density_curves":null}}).to_string(),
        );
    }
    fn defaults(&self) {
        self.entry("film.json", json!({"stock":"kodak_portra_400"}));
        self.entry(
            "print.json",
            json!({"stock":"kodak_portra_endura","stage":"printing"}),
        );
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove profile test directory");
    }
}

fn authored(role: Role) -> Value {
    let mut document = json!({
        "info": {"stock":"authored"},
        "data": {
            "wavelengths": (380..=780).step_by(5).collect::<Vec<_>>(),
            "log_sensitivity": vec![vec![Value::Null,json!(-0.0),json!(1.0000000596046448_f64)];81],
            "channel_density": vec![[0.1,0.2,0.3];81],
            "base_density": vec![Value::Null;81],
            "log_exposure": [-1.0,-0.0,0.0,1.0000000596046448_f64],
            "density_curves_model": {
                "centers": ([[0.1,0.2,0.3];3]),
                "amplitudes": ([[1.0,2.0,3.0];3]),
                "sigmas": ([[0.1,0.2,0.3];3])
            }
        }
    });
    if role == Role::Print {
        document["info"]["stage"] = json!("printing");
        document["info"]["support"] = json!("paper");
    }
    document
}

fn assert_field(error: ProfileError, field: &str, expected: Requirement) {
    assert!(error.path.ends_with("source.json"));
    assert!(error.to_string().contains(field));
    match error.kind {
        ProfileErrorKind::Field {
            field: actual,
            requirement,
        } => {
            assert_eq!(actual, field);
            assert_eq!(requirement, expected);
        }
        other => panic!("expected field error for {field}, got {other:?}"),
    }
}

const FILM_CATALOG: &[(&str, &str)] = &[
    ("fujifilm_c200", "Fujifilm C200"),
    ("fujifilm_pro_400h", "Fujifilm Pro 400H"),
    ("fujifilm_provia_100f", "Fujifilm Provia 100F"),
    ("fujifilm_velvia_100", "Fujifilm Velvia 100"),
    ("fujifilm_xtra_400", "Fujifilm X-Tra 400"),
    ("kodak_ektachrome_100", "Kodak Ektachrome 100"),
    ("kodak_ektar_100", "Kodak Ektar 100"),
    ("kodak_gold_200", "Kodak Gold 200"),
    ("kodak_kodachrome_64", "Kodak Kodachrome 64"),
    ("kodak_portra_160", "Kodak Portra 160"),
    ("kodak_portra_400", "Kodak Portra 400"),
    ("kodak_portra_800", "Kodak Portra 800"),
    ("kodak_portra_800_push1", "Kodak Portra 800 (Push 1)"),
    ("kodak_portra_800_push2", "Kodak Portra 800 (Push 2)"),
    ("kodak_ultramax_400", "Kodak Ultramax 400"),
    ("kodak_verita_200d", "Kodak Verita 200D"),
    ("kodak_vision3_200t", "Kodak Vision3 200T"),
    ("kodak_vision3_250d", "Kodak Vision3 250D"),
    ("kodak_vision3_500t", "Kodak Vision3 500T"),
    ("kodak_vision3_50d", "Kodak Vision3 50D"),
];

const PRINT_CATALOG: &[(&str, &str)] = &[
    (
        "fujifilm_crystal_archive_typeii",
        "Fujifilm Crystal Archive Type II",
    ),
    ("kodak_ektacolor_edge", "Kodak Ektacolor Edge"),
    ("kodak_endura_premier", "Kodak Professional Endura Premier"),
    ("kodak_portra_endura", "Kodak Professional Portra Endura"),
    ("kodak_supra_endura", "Kodak Professional Supra Endura"),
    ("kodak_ultra_endura", "Kodak Professional Ultra Endura"),
    ("kodak_2383", "Kodak Vision 2383"),
    ("kodak_2393", "Kodak Vision Premier 2393"),
];

#[test]
fn bundled_catalog_order_labels_defaults_and_all_sources_match_cpp() {
    let catalog = load_catalog(&repository().join("Resources")).unwrap();
    assert_eq!(catalog.films().len(), 20);
    assert_eq!(catalog.prints().len(), 8);
    assert!(catalog.unavailable().is_empty());
    assert_eq!(
        catalog.ignored(),
        [
            PathBuf::from("Resources/filters/neutral_print_filters.json"),
            PathBuf::from("Resources/profiles/enlarger_neutral*.json")
        ]
    );
    assert_eq!(catalog.default_film().key(), "kodak_portra_400");
    assert_eq!(catalog.default_print().key(), "kodak_portra_endura");
    assert!(catalog.film("missing").is_none());
    assert!(catalog.print("missing").is_none());
    for (role, entries, expected) in [
        (Role::Film, catalog.films(), FILM_CATALOG),
        (Role::Print, catalog.prints(), PRINT_CATALOG),
    ] {
        for (entry, &(key, label)) in entries.iter().zip(expected) {
            assert_eq!((entry.key(), entry.label()), (key, label));
            let source = match role {
                Role::Film => load_film_source(entry.source_path()),
                Role::Print => load_print_source(entry.source_path()),
            }
            .unwrap_or_else(|error| panic!("{key}: {error}"));
            let info = source.info();
            assert_eq!((info.stock(), info.name()), (key, label));
            assert_eq!(
                (info.support(), info.stage(), info.polarity()),
                (entry.support(), entry.stage(), entry.polarity())
            );
            assert_eq!(source.samples().log_exposure().len(), 256);
            assert_eq!(source.samples().wavelengths()[0].to_bits(), 0x43be0000);
            assert_eq!(source.samples().wavelengths()[80].to_bits(), 0x44430000);
        }
    }
}

#[test]
fn representative_metadata_and_source_bits_match_cpp() {
    {
        let source =
            load_film_source(&repository().join("Resources/profiles/kodak_portra_400.json"))
                .unwrap();
        let info = source.info();
        assert_eq!(info.polarity(), Polarity::Negative);
        assert_eq!(info.usage(), ProfileUse::Still);
        assert_eq!(info.antihalation(), Antihalation::Strong);
        assert_eq!(info.channel_model(), ChannelModel::Color);
        assert_eq!(
            (info.reference_illuminant(), info.viewing_illuminant()),
            ("D55", "D50")
        );
        let samples = source.samples();
        assert_eq!(
            samples.log_sensitivity()[40].map(f32::to_bits),
            [0xbf7b7fd0, 0xbf8051d2, 0xc08c254c]
        );
        assert_eq!(
            samples.channel_density()[40].map(f32::to_bits),
            [0x3deeb13e, 0x3f1db5a6, 0x3cab1d50]
        );
        assert_eq!(samples.base_density()[40].to_bits(), 0x3ed9806f);
        assert_eq!(samples.log_exposure()[128].to_bits(), 0x3fe0706f69446738);
        assert_eq!((samples.log_exposure()[128] as f32).to_bits(), 0x3f03837b);
        assert_eq!(
            source.density_model().centers()[0][0].to_bits(),
            0xbfe6c9fb6134ce3e
        );
        assert_eq!(
            source.density_model().amplitudes()[0][0].to_bits(),
            0x3fe31b1d92b7fe09
        );
        assert_eq!(
            source.density_model().sigmas()[0][0].to_bits(),
            0x3fdc7ced916872b0
        );
        assert_eq!(
            samples
                .hanatos2025_adaptation_window_params()
                .unwrap()
                .map(f32::to_bits),
            [0x43d8bf05, 0x41fc78dc, 0x44144895, 0x428986c3]
        );
        assert_eq!(
            samples.hanatos2025_adaptation_surface_params().unwrap()[2][14].to_bits(),
            0xc083b839
        );
    }
    {
        let source =
            load_film_source(&repository().join("Resources/profiles/fujifilm_provia_100f.json"))
                .unwrap();
        let info = source.info();
        assert_eq!(info.polarity(), Polarity::Positive);
        assert_eq!(info.usage(), ProfileUse::Still);
        assert_eq!(info.antihalation(), Antihalation::Strong);
        assert_eq!(info.channel_model(), ChannelModel::Color);
        assert_eq!(
            (info.reference_illuminant(), info.viewing_illuminant()),
            ("D55", "D50")
        );
        let samples = source.samples();
        assert_eq!(
            samples.log_sensitivity()[40].map(f32::to_bits),
            [0xbfc0b9dc, 0xbedbb85f, 0xc099851f]
        );
        assert_eq!(
            samples.channel_density()[40].map(f32::to_bits),
            [0x3f09b7d0, 0x3f207ef1, 0x3d088d6d]
        );
        assert_eq!(samples.base_density()[40].to_bits(), 0x3db90646);
        assert_eq!(samples.log_exposure()[128].to_bits(), 0x3fe0706f69446738);
        assert_eq!((samples.log_exposure()[128] as f32).to_bits(), 0x3f03837b);
        assert_eq!(
            source.density_model().centers()[0][0].to_bits(),
            0xbff02492bc2fc697
        );
        assert_eq!(
            source.density_model().amplitudes()[0][0].to_bits(),
            0x3febfecbb7f9d6f1
        );
        assert_eq!(
            source.density_model().sigmas()[0][0].to_bits(),
            0x3fdb5f809917939a
        );
        assert_eq!(
            samples
                .hanatos2025_adaptation_window_params()
                .unwrap()
                .map(f32::to_bits),
            [0x43c6584d, 0x420a763c, 0x441656c4, 0x421e40c1]
        );
        assert_eq!(
            samples.hanatos2025_adaptation_surface_params().unwrap()[2][14].to_bits(),
            0xbf18edf6
        );
    }
    {
        let source =
            load_film_source(&repository().join("Resources/profiles/kodak_vision3_500t.json"))
                .unwrap();
        let info = source.info();
        assert_eq!(info.polarity(), Polarity::Negative);
        assert_eq!(info.usage(), ProfileUse::Cine);
        assert_eq!(info.antihalation(), Antihalation::Strong);
        assert_eq!(info.channel_model(), ChannelModel::Color);
        assert_eq!(
            (info.reference_illuminant(), info.viewing_illuminant()),
            ("T", "D50")
        );
        let samples = source.samples();
        assert_eq!(
            samples.log_sensitivity()[40].map(f32::to_bits),
            [0xbfc3804e, 0xbf24e619, 0xc0aa074a]
        );
        assert_eq!(
            samples.channel_density()[40].map(f32::to_bits),
            [0x3e2aaeb0, 0x3f1b42dd, 0x3d3caa76]
        );
        assert_eq!(samples.base_density()[40].to_bits(), 0x3f05c769);
        assert_eq!(samples.log_exposure()[128].to_bits(), 0x3fe0706f69446738);
        assert_eq!((samples.log_exposure()[128] as f32).to_bits(), 0x3f03837b);
        assert_eq!(
            source.density_model().centers()[0][0].to_bits(),
            0xbfecb6848beb5b2d
        );
        assert_eq!(
            source.density_model().amplitudes()[0][0].to_bits(),
            0x3fe1f991bc558644
        );
        assert_eq!(
            source.density_model().sigmas()[0][0].to_bits(),
            0x3fe47ca643cc07ab
        );
        assert_eq!(
            samples
                .hanatos2025_adaptation_window_params()
                .unwrap()
                .map(f32::to_bits),
            [0x43d23323, 0x41db3f66, 0x4419e648, 0x4221b671]
        );
        assert_eq!(
            samples.hanatos2025_adaptation_surface_params().unwrap()[2][14].to_bits(),
            0xc0951e71
        );
    }
    {
        let source =
            load_print_source(&repository().join("Resources/profiles/kodak_portra_endura.json"))
                .unwrap();
        let info = source.info();
        assert_eq!(info.polarity(), Polarity::Negative);
        assert_eq!(info.usage(), ProfileUse::Still);
        assert_eq!(info.antihalation(), Antihalation::Strong);
        assert_eq!(info.channel_model(), ChannelModel::Color);
        assert_eq!(
            (info.reference_illuminant(), info.viewing_illuminant()),
            ("TH-KG3", "D50")
        );
        let samples = source.samples();
        assert_eq!(
            samples.log_sensitivity()[40].map(f32::to_bits),
            [0xbf832631, 0xbfafb8cc, 0xc0bec956]
        );
        assert_eq!(
            samples.channel_density()[40].map(f32::to_bits),
            [0x3eeeb9e5, 0x3f0c8d09, 0x3be6ea85]
        );
        assert_eq!(samples.base_density()[40].to_bits(), 0x3db5579b);
        assert_eq!(samples.log_exposure()[128].to_bits(), 0x3fe0706f69446738);
        assert_eq!((samples.log_exposure()[128] as f32).to_bits(), 0x3f03837b);
        assert_eq!(
            source.density_model().centers()[0][0].to_bits(),
            0xbfb3bff8a8f3a9b0
        );
        assert_eq!(
            source.density_model().amplitudes()[0][0].to_bits(),
            0x3fe949b62c77574f
        );
        assert_eq!(
            source.density_model().sigmas()[0][0].to_bits(),
            0x3fd0a5f84cad57bc
        );
        assert!(samples.hanatos2025_adaptation_window_params().is_none());
        assert!(samples.hanatos2025_adaptation_surface_params().is_none());
    }
    {
        let source =
            load_print_source(&repository().join("Resources/profiles/kodak_2383.json")).unwrap();
        let info = source.info();
        assert_eq!(info.polarity(), Polarity::Negative);
        assert_eq!(info.usage(), ProfileUse::Cine);
        assert_eq!(info.antihalation(), Antihalation::Strong);
        assert_eq!(info.channel_model(), ChannelModel::Color);
        assert_eq!(
            (info.reference_illuminant(), info.viewing_illuminant()),
            ("TH-KG3", "K75P")
        );
        let samples = source.samples();
        assert_eq!(
            samples.log_sensitivity()[40].map(f32::to_bits),
            [0xc0294ecf, 0xbf2aa3df, 0xc0de4817]
        );
        assert_eq!(
            samples.channel_density()[40].map(f32::to_bits),
            [0x3ee40faa, 0x3f0fef6b, 0x3cf2de44]
        );
        assert_eq!(samples.base_density()[40].to_bits(), 0x3d5ee137);
        assert_eq!(samples.log_exposure()[128].to_bits(), 0x3fe0706f69446738);
        assert_eq!((samples.log_exposure()[128] as f32).to_bits(), 0x3f03837b);
        assert_eq!(
            source.density_model().centers()[0][0].to_bits(),
            0x3f9526527a20578e
        );
        assert_eq!(
            source.density_model().amplitudes()[0][0].to_bits(),
            0x3fef36ac647778dd
        );
        assert_eq!(
            source.density_model().sigmas()[0][0].to_bits(),
            0x3fd5dab191dde376
        );
        assert!(samples.hanatos2025_adaptation_window_params().is_none());
        assert!(samples.hanatos2025_adaptation_surface_params().is_none());
    }
}

#[test]
fn metadata_defaults_and_nullable_samples_are_complete() {
    let dir = Directory::new();
    for role in [Role::Film, Role::Print] {
        let source = dir.source(&authored(role), role).unwrap();
        let info = source.info();
        assert_eq!((info.stock(), info.name()), ("authored", "authored"));
        assert_eq!(
            info.support(),
            if role == Role::Film {
                Support::Film
            } else {
                Support::Paper
            }
        );
        assert_eq!(
            info.stage(),
            if role == Role::Film {
                Stage::Filming
            } else {
                Stage::Printing
            }
        );
        assert_eq!(info.polarity(), Polarity::Negative);
        assert_eq!(info.usage(), ProfileUse::Still);
        assert_eq!(info.antihalation(), Antihalation::Weak);
        assert_eq!(info.channel_model(), ChannelModel::Color);
        assert_eq!(
            (info.reference_illuminant(), info.viewing_illuminant()),
            ("D55", "D50")
        );
        let samples = source.samples();
        assert_eq!(
            samples.log_sensitivity()[0].map(f32::to_bits),
            [0x7fc00000, 0x80000000, 0x3f800000]
        );
        assert!(
            samples
                .base_density()
                .iter()
                .all(|sample| sample.to_bits() == 0x7fc00000)
        );
        assert_eq!(
            samples
                .log_exposure()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            [
                0xbff0000000000000,
                0x8000000000000000,
                0,
                0x3ff0000010000000
            ]
        );
        assert!(samples.hanatos2025_adaptation_window_params().is_none());
        assert!(samples.hanatos2025_adaptation_surface_params().is_none());
    }
}

#[test]
fn source_metadata_rejects_null_types_unknown_values_and_role_mismatch() {
    let dir = Directory::new();
    for key in [
        "stock",
        "name",
        "support",
        "stage",
        "type",
        "use",
        "antihalation",
        "channel_model",
        "reference_illuminant",
        "viewing_illuminant",
    ] {
        for value in [Value::Null, json!(1), json!(true), json!([]), json!({})] {
            let mut input = authored(Role::Film);
            input["info"][key] = value;
            assert_field(
                dir.source(&input, Role::Film).unwrap_err(),
                &format!("info.{key}"),
                Requirement::String,
            );
        }
    }
    for key in [
        "support",
        "stage",
        "type",
        "use",
        "antihalation",
        "channel_model",
    ] {
        let mut input = authored(Role::Film);
        input["info"][key] = json!("unknown");
        assert_field(
            dir.source(&input, Role::Film).unwrap_err(),
            &format!("info.{key}"),
            Requirement::MetadataValue,
        );
    }
    let mut input = authored(Role::Film);
    input["info"].as_object_mut().unwrap().remove("stock");
    assert_field(
        dir.source(&input, Role::Film).unwrap_err(),
        "info.stock",
        Requirement::String,
    );
    input["info"]["stock"] = json!("");
    assert_eq!(dir.source(&input, Role::Film).unwrap().info().stock(), "");
    input["info"]["support"] = json!("paper");
    assert_field(
        dir.source(&input, Role::Film).unwrap_err(),
        "info",
        Requirement::SelectedRole,
    );
    input["info"]["support"] = json!("film");
    assert_field(
        dir.source(&input, Role::Print).unwrap_err(),
        "info.stage",
        Requirement::SelectedRole,
    );
    input["info"]["stage"] = json!("printing");
    assert_field(
        dir.source(&input, Role::Film).unwrap_err(),
        "info",
        Requirement::SelectedRole,
    );
    let source = dir.source(&input, Role::Print).unwrap();
    assert_eq!(source.info().support(), Support::Film);
}

#[test]
fn authored_metadata_variants_and_original_illuminant_spelling_are_retained() {
    let dir = Directory::new();
    let mut input = authored(Role::Film);
    input["info"]["use"] = json!("cine");
    input["info"]["antihalation"] = json!("no");
    input["info"]["channel_model"] = json!("bw");
    input["info"]["type"] = json!("positive");
    let source = dir.source(&input, Role::Film).unwrap();
    assert_eq!(
        (
            source.info().usage(),
            source.info().antihalation(),
            source.info().channel_model(),
            source.info().polarity()
        ),
        (
            ProfileUse::Cine,
            Antihalation::No,
            ChannelModel::Bw,
            Polarity::Positive
        )
    );
    for key in ["reference_illuminant", "viewing_illuminant"] {
        for accepted in [
            " d55 ",
            "th_kg3_l",
            "KINOTON75P",
            "D1",
            "BB3200",
            "BB+3200.5",
            "BB3.2e3",
            "BB1e-3",
            " bb3200.5 ",
        ] {
            input["info"][key] = json!(accepted);
            let source = dir.source(&input, Role::Film).unwrap();
            let actual = if key == "reference_illuminant" {
                source.info().reference_illuminant()
            } else {
                source.info().viewing_illuminant()
            };
            assert_eq!(actual, accepted);
        }
        for rejected in ["D", "D-50", "BB0", "BB-3", "BB1e309", "BB1e-999", "unknown"] {
            input["info"][key] = json!(rejected);
            assert_field(
                dir.source(&input, Role::Film).unwrap_err(),
                &format!("info.{key}"),
                Requirement::Illuminant,
            );
        }
        input["info"][key] = json!("D55");
    }
}

#[test]
fn hanatos_absence_empty_null_and_shapes_follow_the_selected_role() {
    let dir = Directory::new();
    for role in [Role::Film, Role::Print] {
        for key in [
            "hanatos2025_adaptation_window_params",
            "hanatos2025_adaptation_surface_params",
        ] {
            let mut input = authored(role);
            input["data"][key] = Value::Null;
            if role == Role::Print {
                assert!(dir.source(&input, role).is_ok());
            } else {
                assert_field(
                    dir.source(&input, role).unwrap_err(),
                    &format!("data.{key}"),
                    Requirement::AdaptationArray,
                );
            }
            input["data"][key] = json!([]);
            assert!(dir.source(&input, role).is_ok());
            input["data"][key] = json!(false);
            assert_field(
                dir.source(&input, role).unwrap_err(),
                &format!("data.{key}"),
                Requirement::AdaptationArray,
            );
            let window = key.contains("window");
            input["data"][key] = json!([1, 2]);
            assert_field(
                dir.source(&input, role).unwrap_err(),
                &format!("data.{key}"),
                Requirement::ArrayLength(if window { 4 } else { 3 }),
            );
            input["data"][key] = if window {
                json!([1, 2, 3, 4])
            } else {
                json!(([[0.25; 15]; 3]))
            };
            let source = dir.source(&input, role).unwrap();
            if window {
                assert_eq!(
                    source
                        .samples()
                        .hanatos2025_adaptation_window_params()
                        .unwrap(),
                    &[1.0, 2.0, 3.0, 4.0]
                );
            } else {
                assert_eq!(
                    source
                        .samples()
                        .hanatos2025_adaptation_surface_params()
                        .unwrap(),
                    &[[0.25; 15]; 3]
                );
            }
            if window {
                input["data"][key][0] = Value::Null;
            } else {
                input["data"][key][0][0] = Value::Null;
            }
            let field = if window {
                format!("data.{key}[0]")
            } else {
                format!("data.{key}[0][0]")
            };
            assert_field(
                dir.source(&input, role).unwrap_err(),
                &field,
                Requirement::FiniteNumber,
            );
            input["data"][key] = if window {
                json!(([1e40; 4]))
            } else {
                json!(([[1e40; 15]; 3]))
            };
            let source = dir.source(&input, role).unwrap();
            if window {
                assert!(
                    source
                        .samples()
                        .hanatos2025_adaptation_window_params()
                        .unwrap()[0]
                        .is_infinite()
                );
            } else {
                assert!(
                    source
                        .samples()
                        .hanatos2025_adaptation_surface_params()
                        .unwrap()[0][0]
                        .is_infinite()
                );
            }
        }
    }
}

#[test]
fn scientific_sample_shape_axis_and_token_checks_match_cpp() {
    let dir = Directory::new();
    for key in [
        "wavelengths",
        "log_sensitivity",
        "channel_density",
        "base_density",
    ] {
        let mut input = authored(Role::Film);
        input["data"][key] = json!([1]);
        assert_field(
            dir.source(&input, Role::Film).unwrap_err(),
            &format!("data.{key}"),
            Requirement::ArrayLength(81),
        );
    }
    for key in ["log_sensitivity", "channel_density"] {
        let mut input = authored(Role::Film);
        input["data"][key][0] = json!([1, 2]);
        assert_field(
            dir.source(&input, Role::Film).unwrap_err(),
            &format!("data.{key}[0]"),
            Requirement::ArrayLength(3),
        );
        input["data"][key][0] = json!(["1", 2, 3]);
        assert_field(
            dir.source(&input, Role::Film).unwrap_err(),
            &format!("data.{key}[0][0]"),
            Requirement::FiniteNumber,
        );
        input["data"][key][0] = json!([1e40, null, -1e40]);
        assert_eq!(
            if key == "log_sensitivity" {
                dir.source(&input, Role::Film)
                    .unwrap()
                    .samples()
                    .log_sensitivity()[0]
                    .map(f32::to_bits)
            } else {
                dir.source(&input, Role::Film)
                    .unwrap()
                    .samples()
                    .channel_density()[0]
                    .map(f32::to_bits)
            },
            [0x7f800000, 0x7fc00000, 0xff800000]
        );
    }
    let mut input = authored(Role::Film);
    input["data"]["wavelengths"][0] = json!(380.000001);
    assert_eq!(
        dir.source(&input, Role::Film)
            .unwrap()
            .samples()
            .wavelengths()[0],
        380.0
    );
    input["data"]["wavelengths"][0] = json!(381);
    assert_field(
        dir.source(&input, Role::Film).unwrap_err(),
        "data.wavelengths[0]",
        Requirement::CanonicalWavelength(380),
    );
    input["data"]["wavelengths"][0] = Value::Null;
    assert_field(
        dir.source(&input, Role::Film).unwrap_err(),
        "data.wavelengths[0]",
        Requirement::FiniteNumber,
    );
    input = authored(Role::Film);
    input["data"]["base_density"][0] = json!(true);
    assert_field(
        dir.source(&input, Role::Film).unwrap_err(),
        "data.base_density[0]",
        Requirement::FiniteNumber,
    );
}

#[test]
fn exposure_checks_both_precisions_and_accepts_short_repeated_axes() {
    let dir = Directory::new();
    let mut input = authored(Role::Film);
    for accepted in [json!([1]), json!([0, 0]), json!([1.00000001, 1.00000002])] {
        input["data"]["log_exposure"] = accepted;
        assert!(dir.source(&input, Role::Film).is_ok());
    }
    for (array, field, requirement) in [
        (json!([]), "data.log_exposure", Requirement::ExposureCount),
        (
            json!([null]),
            "data.log_exposure[0]",
            Requirement::FiniteNumber,
        ),
        (
            json!([true]),
            "data.log_exposure[0]",
            Requirement::FiniteNumber,
        ),
        (
            json!(["1"]),
            "data.log_exposure[0]",
            Requirement::FiniteNumber,
        ),
        (
            json!([1.00000002, 1.00000001]),
            "data.log_exposure[1]",
            Requirement::NondecreasingExposure,
        ),
        (
            json!([1e40]),
            "data.log_exposure[0]",
            Requirement::FloatRange,
        ),
    ] {
        input["data"]["log_exposure"] = array;
        let error = dir.source(&input, Role::Film).unwrap_err();
        assert_eq!(error.role, Role::Film);
        assert_eq!(error.stock.as_deref(), Some("authored"));
        assert_field(error, field, requirement);
    }
}

#[test]
fn density_model_is_source_only_and_enforces_coefficient_shapes_and_sigma_range() {
    let dir = Directory::new();
    for key in ["centers", "amplitudes", "sigmas"] {
        let mut input = authored(Role::Film);
        input["data"]["density_curves_model"][key] = json!([[1, 2, 3]]);
        assert_field(
            dir.source(&input, Role::Film).unwrap_err(),
            "data.density_curves_model",
            Requirement::ModelCoefficients,
        );
        for invalid in [Value::Null, json!(true), json!("1")] {
            input = authored(Role::Film);
            input["data"]["density_curves_model"][key][0][0] = invalid;
            assert_field(
                dir.source(&input, Role::Film).unwrap_err(),
                "data.density_curves_model",
                Requirement::ModelCoefficients,
            );
        }
        input = authored(Role::Film);
        input["data"]["density_curves_model"][key][0][0] = json!(1e40);
        assert_field(
            dir.source(&input, Role::Film).unwrap_err(),
            &format!("data.density_curves_model.{key}[0][0]"),
            Requirement::FloatRange,
        );
    }
    for sigma in [0.0, -1.0, 1e-50] {
        let mut input = authored(Role::Film);
        input["data"]["density_curves_model"]["sigmas"][0][0] = json!(sigma);
        assert_field(
            dir.source(&input, Role::Film).unwrap_err(),
            "data.density_curves_model.sigmas[0][0]",
            Requirement::PositiveSigma,
        );
    }
    let mut input = authored(Role::Film);
    input["data"]["density_curves"] = json!("unused authored curves");
    input["data"]["density_curves_layers"] = json!(false);
    input["data"]["density_curves_model"]["model_type"] = json!("unused");
    input["data"]["density_curves_model"]["amplitudes"][0][0] = json!(-1.0);
    assert_eq!(
        dir.source(&input, Role::Film)
            .unwrap()
            .density_model()
            .amplitudes()[0][0],
        -1.0
    );
}

#[test]
fn file_and_json_errors_do_not_publish_a_partial_source() {
    let dir = Directory::new();
    let error = load_film_source(&dir.0.join("missing.json")).unwrap_err();
    assert!(matches!(error.kind, ProfileErrorKind::Read(_)));
    for text in ["{", "true trailing", "{\"unused\":1e999}"] {
        let path = dir.write("source.json", text);
        assert!(matches!(
            load_film_source(&path).unwrap_err().kind,
            ProfileErrorKind::Json(_)
        ));
    }
    for text in ["[]", "null", "1"] {
        let path = dir.write("source.json", text);
        assert_field(
            load_film_source(&path).unwrap_err(),
            "info",
            Requirement::Object,
        );
    }
    let mut input = authored(Role::Film);
    input.as_object_mut().unwrap().remove("data");
    assert_field(
        dir.source(&input, Role::Film).unwrap_err(),
        "data",
        Requirement::Object,
    );
    input = authored(Role::Film);
    input["data"]
        .as_object_mut()
        .unwrap()
        .remove("density_curves_model");
    assert_field(
        dir.source(&input, Role::Film).unwrap_err(),
        "data.density_curves_model",
        Requirement::Object,
    );
}

#[test]
fn decimal_tokens_use_double_rounding_and_preserve_negative_zero() {
    let dir = Directory::new();
    let mut input = authored(Role::Film);
    input["data"]["log_exposure"] = json!("TOKEN");
    let text = serde_json::to_string(&input).unwrap().replace(
        "\"TOKEN\"",
        "[-0,-0.0,1.0000000596046448,18446744073709551616]",
    );
    let path = dir.write("source.json", text);
    let source = load_film_source(&path).unwrap();
    let axis = source.samples().log_exposure();
    assert_eq!(
        axis.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [
            0x8000000000000000,
            0x8000000000000000,
            0x3ff0000010000000,
            0x43f0000000000000
        ]
    );
    assert_eq!(
        axis.iter()
            .map(|x| (*x as f32).to_bits())
            .collect::<Vec<_>>(),
        [0x80000000, 0x80000000, 0x3f800000, 0x5f800000]
    );
    let text = serde_json::to_string(&input)
        .unwrap()
        .replace("\"TOKEN\"", "[1e-999]");
    let path = dir.write("source.json", text);
    assert_eq!(
        load_film_source(&path).unwrap().samples().log_exposure()[0].to_bits(),
        0
    );
}

#[test]
fn bom_and_duplicate_members_follow_cpp_json_decoding() {
    let dir = Directory::new();
    let text = serde_json::to_string(&authored(Role::Film))
        .unwrap()
        .replace(
            "\"stock\":\"authored\"",
            "\"stock\":1,\"stock\":\"authored\"",
        );
    let path = dir.write("source.json", format!("\u{feff}{text}"));
    assert_eq!(load_film_source(&path).unwrap().info().stock(), "authored");
}

#[test]
fn catalog_is_shallow_and_classifies_defaults_ignored_and_unavailable_entries() {
    let dir = Directory::new();
    dir.defaults();
    dir.write("profiles/ignored.json", "{}");
    dir.write("profiles/bad.json", "{");
    dir.write(
        "profiles/model-only.json",
        r#"{"data":{"density_curves_model":{}}}"#,
    );
    dir.entry("no-stock.json", json!({}));
    dir.entry("null-support.json", json!({"stock":"null","support":null}));
    dir.entry(
        "paper-filming.json",
        json!({"stock":"paper","support":"paper"}),
    );
    dir.entry("uppercase-only.JSON", json!({"stock":"wrong-extension"}));
    dir.entry("nested/child.json", json!({"stock":"nested"}));
    dir.entry("a.json", json!({"stock":"a","name":"Same"}));
    dir.entry("b.json", json!({"stock":"b","name":"Same"}));
    dir.entry("label-number.json", json!({"stock":"label","name":3}));
    dir.entry("empty-label.json", json!({"stock":"empty","name":""}));
    dir.entry(
        "same-key-print.json",
        json!({"stock":"a","stage":"printing"}),
    );
    dir.write(
        "profiles/trailing.json",
        r#"{"info":{"stock":"trailing"},"data":{"base_density":null}} trailing"#,
    );
    let catalog = load_catalog(&dir.0).unwrap();
    assert_eq!(
        catalog.films().iter().map(|e| e.key()).collect::<Vec<_>>(),
        ["empty", "a", "b", "kodak_portra_400", "label", "trailing"]
    );
    assert_eq!(
        catalog.prints().iter().map(|e| e.key()).collect::<Vec<_>>(),
        ["a", "kodak_portra_endura"]
    );
    assert_eq!(catalog.ignored().len(), 4);
    assert_eq!(catalog.unavailable().len(), 4);
    for entry in catalog.unavailable() {
        match entry.path.file_name().unwrap().to_str().unwrap() {
            "bad.json" => assert!(matches!(entry.kind, CatalogEntryErrorKind::Json(_))),
            "no-stock.json" => assert!(matches!(entry.kind, CatalogEntryErrorKind::MissingStock)),
            _ => assert!(matches!(entry.kind, CatalogEntryErrorKind::UnsupportedRole)),
        }
    }
    let film = catalog.default_film();
    assert_eq!(
        (film.key(), film.label()),
        ("kodak_portra_400", "kodak_portra_400")
    );
    assert_eq!(
        (film.support(), film.stage(), film.polarity()),
        (Support::Film, Stage::Filming, Polarity::Negative)
    );
}

#[test]
fn catalog_duplicate_rejection_is_per_role() {
    for role in [Role::Film, Role::Print] {
        let dir = Directory::new();
        dir.defaults();
        let (key, stage) = if role == Role::Film {
            ("kodak_portra_400", "filming")
        } else {
            ("kodak_portra_endura", "printing")
        };
        dir.entry("duplicate.json", json!({"stock":key,"stage":stage}));
        match load_catalog(&dir.0).unwrap_err() {
            CatalogError::DuplicateKey {
                role: actual,
                key: actual_key,
                ..
            } => {
                assert_eq!(actual, role);
                assert_eq!(actual_key, key);
            }
            error => panic!("{error}"),
        }
    }
    let dir = Directory::new();
    dir.defaults();
    dir.entry(
        "invisible-duplicate.json",
        json!({"stock":"kodak_portra_400","support":"paper"}),
    );
    assert_eq!(load_catalog(&dir.0).unwrap().unavailable().len(), 1);
}

#[test]
fn catalog_requires_both_roles_and_both_default_keys() {
    let dir = Directory::new();
    assert!(matches!(
        load_catalog(&dir.0),
        Err(CatalogError::Directory { .. })
    ));
    fs::create_dir_all(dir.0.join("profiles")).unwrap();
    assert!(matches!(
        load_catalog(&dir.0),
        Err(CatalogError::EmptyRole {
            role: Role::Film,
            ..
        })
    ));
    dir.entry("film.json", json!({"stock":"film"}));
    assert!(matches!(
        load_catalog(&dir.0),
        Err(CatalogError::EmptyRole {
            role: Role::Print,
            ..
        })
    ));
    dir.entry("print.json", json!({"stock":"print","stage":"printing"}));
    assert!(matches!(
        load_catalog(&dir.0),
        Err(CatalogError::MissingDefault {
            role: Role::Film,
            ..
        })
    ));
    dir.entry("film.json", json!({"stock":"kodak_portra_400"}));
    assert!(matches!(
        load_catalog(&dir.0),
        Err(CatalogError::MissingDefault {
            role: Role::Print,
            ..
        })
    ));
    dir.entry(
        "print.json",
        json!({"stock":"kodak_portra_endura","stage":"printing"}),
    );
    assert!(load_catalog(&dir.0).is_ok());
}

#[test]
fn blackbody_temperatures_require_an_unmodified_positive_decimal_suffix() {
    let dir = Directory::new();
    for role in [Role::Film, Role::Print] {
        let mut input = authored(role);
        for key in ["reference_illuminant", "viewing_illuminant"] {
            // Product-contract expectations, including cases whose validity would
            // change if decimal points, exponent signs or punctuation were removed.
            for (illuminant, accepted) in [
                ("BB3200", true),
                (" bb+3200.5\t", true),
                ("Bb3.2e+3", true),
                ("BB1e-0", true),
                ("BB1.5e308", true),
                ("BB5e-324", true),
                ("BB0.1e-323", false),
                ("BB1e309", false),
                ("BB0", false),
                ("BB-0", false),
                ("BB-3200", false),
                ("BBNaN", false),
                ("BBinf", false),
                ("BBInfinity", false),
                ("BB", false),
                ("BB1e-", false),
                ("BB3.2.0", false),
                ("BB3_200", false),
                ("BB3,200", false),
                ("BB3200!", false),
                ("BB 3200", false),
                ("B.B3200", false),
                ("éBB3200", false),
                ("☀BB3200", false),
                ("BB0x1p4", false),
                ("BB0X1000", false),
                ("BB0x1p-1074", false),
                ("BB0x1fffffffffffffp971", false),
                ("BB0x1fffffffffffff8p967", false),
                ("BB0x1fffffffffffff80p963", false),
            ] {
                input["info"][key] = json!(illuminant);
                let result = dir.source(&input, role);
                assert_eq!(result.is_ok(), accepted, "{role:?} {key} {illuminant}");
                if let Err(error) = result {
                    assert_eq!(error.role, role);
                    assert_field(error, &format!("info.{key}"), Requirement::Illuminant);
                }
            }
            input["info"][key] = json!("D55");
        }
    }
}

#[test]
fn json_zero_and_exponent_tokens_preserve_bits_and_strings_in_both_roles() {
    let dir = Directory::new();
    dir.defaults();
    for role in [Role::Film, Role::Print] {
        let mut input = authored(role);
        input["info"]["stock"] = json!(r#"quoted " -0, slash \ 1e-0]"#);
        input["data"]["log_exposure"] = json!(["TOKEN"]);
        input["data"]["log_sensitivity"][0][0] = json!("TOKEN");
        input["data"]["channel_density"][0][0] = json!("TOKEN");
        input["data"]["base_density"][0] = json!("TOKEN");
        input["data"]["density_curves_model"]["centers"][0][0] = json!("TOKEN");
        input["data"]["density_curves_model"]["amplitudes"][0][0] = json!("TOKEN");
        input["data"]["hanatos2025_adaptation_window_params"] = json!(["TOKEN", 1, 2, 3]);
        input["unused"] = json!({"$serde_json::private::Number":"not a number"});
        for (token, bits_f64, bits_f32) in [
            ("-0", 0x8000000000000000, 0x80000000),
            ("-0.0", 0x8000000000000000, 0x80000000),
            ("-0e0", 0x8000000000000000, 0x80000000),
            ("-0e-0", 0x8000000000000000, 0x80000000),
            ("-0e+0", 0x8000000000000000, 0x80000000),
            ("0", 0, 0),
            ("0e-0", 0, 0),
            ("1e-0", 0x3ff0000000000000, 0x3f800000),
            ("1E-0", 0x3ff0000000000000, 0x3f800000),
            ("1e-00", 0x3ff0000000000000, 0x3f800000),
            ("1e+0", 0x3ff0000000000000, 0x3f800000),
        ] {
            let text = serde_json::to_string(&input)
                .unwrap()
                .replace("\"TOKEN\"", token);
            let path = dir.write("source.json", &text);
            let source = match role {
                Role::Film => load_film_source(&path),
                Role::Print => load_print_source(&path),
            }
            .unwrap_or_else(|error| panic!("{role:?} {token}: {error}"));
            let stock = input["info"]["stock"].as_str().unwrap();
            assert_eq!(source.info().stock(), stock);
            assert_eq!(source.samples().log_exposure()[0].to_bits(), bits_f64);
            assert_eq!(source.density_model().centers()[0][0].to_bits(), bits_f64);
            assert_eq!(
                source.density_model().amplitudes()[0][0].to_bits(),
                bits_f64
            );
            for sample in [
                source.samples().log_sensitivity()[0][0],
                source.samples().channel_density()[0][0],
                source.samples().base_density()[0],
                source
                    .samples()
                    .hanatos2025_adaptation_window_params()
                    .unwrap()[0],
            ] {
                assert_eq!(sample.to_bits(), bits_f32, "{role:?} {token}");
            }
            dir.write("profiles/numbers.json", text);
            let catalog = load_catalog(&dir.0).unwrap();
            assert!(catalog.unavailable().is_empty(), "{role:?} {token}");
            let entry = match role {
                Role::Film => catalog.film(stock),
                Role::Print => catalog.print(stock),
            }
            .unwrap();
            assert_eq!(entry.label(), stock);
        }
    }
}

#[test]
fn malformed_json_numbers_fail_source_loading_and_make_catalog_entries_unavailable() {
    let dir = Directory::new();
    dir.defaults();
    for role in [Role::Film, Role::Print] {
        let mut input = authored(role);
        input["data"]["log_exposure"] = json!(["TOKEN"]);
        for token in ["--0", "- 0", "01", "1e", "1e-", "NaN", "Infinity", "1e999"] {
            let text = serde_json::to_string(&input)
                .unwrap()
                .replace("\"TOKEN\"", token);
            let path = dir.write("source.json", &text);
            let error = match role {
                Role::Film => load_film_source(&path),
                Role::Print => load_print_source(&path),
            }
            .unwrap_err();
            assert!(matches!(error.kind, ProfileErrorKind::Json(_)), "{token}");
            dir.write("profiles/numbers.json", text);
            let catalog = load_catalog(&dir.0).unwrap();
            assert_eq!(catalog.films().len(), 1);
            assert_eq!(catalog.prints().len(), 1);
            assert_eq!(catalog.unavailable().len(), 1);
            assert!(matches!(
                catalog.unavailable()[0].kind,
                CatalogEntryErrorKind::Json(_)
            ));
        }
    }
}

#[test]
fn model_coefficients_keep_channel_then_layer_order() {
    let source =
        load_film_source(&repository().join("Resources/profiles/kodak_portra_400.json")).unwrap();
    assert_eq!(
        source
            .density_model()
            .centers()
            .map(|row| row.map(f64::to_bits)),
        [
            [0xbfe6c9fb6134ce3e, 0x3fe01fd36f7e3d1d, 0x3ffac85c24c404a7],
            [0xbfeac9abb01c92de, 0x3fd64b1ee2435697, 0x3ff8b0cdc8754f37],
            [0xbfed8a0f4d7add16, 0x3fd27625204af923, 0x3ff87ac6045baf53]
        ]
    );
    assert_eq!(
        source
            .density_model()
            .amplitudes()
            .map(|row| row.map(f64::to_bits)),
        [
            [0x3fe31b1d92b7fe09, 0x3fe61df97aaac109, 0x3fe67d21ff2e48e9],
            [0x3fe1fae7924af0bf, 0x3fe3bf7ced916873, 0x3fe3c38088509bfa],
            [0x3fe4fa4e7ab75643, 0x3fe7ee525892684d, 0x3fe854bad7d7c2ca]
        ]
    );
    assert_eq!(
        source
            .density_model()
            .sigmas()
            .map(|row| row.map(f64::to_bits)),
        [
            [0x3fdc7ced916872b0, 0x3fe1068123810e88, 0x3fe0d912556d19df],
            [0x3fdc64d7f0ed3d86, 0x3fe108765ba6efc3, 0x3fe0e5a57646ae3a],
            [0x3fdc7585be1a8262, 0x3fe107314ca925ff, 0x3fe0dc6195464dc2]
        ]
    );
}

#[test]
fn catalog_recognizes_each_authored_shape_field_without_loading_samples() {
    let dir = Directory::new();
    dir.defaults();
    for key in [
        "wavelengths",
        "log_sensitivity",
        "channel_density",
        "base_density",
        "log_exposure",
        "density_curves",
    ] {
        let mut data = serde_json::Map::new();
        data.insert(key.to_owned(), Value::Null);
        dir.write(
            &format!("profiles/{key}.json"),
            json!({"info":{"stock":key},"data":data}).to_string(),
        );
    }
    for key in ["support", "stage", "type"] {
        let mut info = json!({"stock":key});
        info[key] = Value::Null;
        dir.entry(&format!("null-{key}.json"), info);
    }
    let catalog = load_catalog(&dir.0).unwrap();
    assert_eq!(catalog.films().len(), 7);
    assert_eq!(catalog.unavailable().len(), 3);
    assert!(
        catalog
            .unavailable()
            .iter()
            .all(|entry| matches!(entry.kind, CatalogEntryErrorKind::UnsupportedRole))
    );
}
