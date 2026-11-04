//! Film-Juicer characterization of authored input, before density evaluation.
//! Expected bits/order were captured from the accepted C++ readers at
//! ed3b759a5e3c419ef09b3db694cb220e53e0844e before the Rust translation.
//! Bundle digest: deaddd9e79004008ff0bce40ba48de616d23456cf026c554bb9b111224744773.
//! Spektrafilm provenance: 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc.
//! Ordinary tests use only shipped resources and the independent constants here.
//! Product-contract exceptions: JSON negative zero retains its sign; blackbody
//! temperatures use the original finite decimal suffix, without hex or positivity
//! admission. Source labels and f64 exposures retain authored order and conversions.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

#[path = "support/case_results.rs"]
mod test_cases;
use test_cases::RowCases;
use film_juicer_core::profile::{
    IlluminantKind, Antihalation, CatalogEntryErrorKind, CatalogError, ChannelModel, Polarity,
    ProfileError, ProfileErrorKind, ProfileSource, ProfileUse, Requirement, Role, Stage, Support,
    load_catalog, load_film_source, load_print_source,
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
    fn load_source(&self, document: &Value, role: Role) -> Result<ProfileSource, ProfileError> {
        let path = self.write("source.json", serde_json::to_vec(document).unwrap());
        match role {
            Role::Film => load_film_source(&path),
            Role::Print => load_print_source(&path),
        }
    }
    fn write_entry(&self, filename: &str, info: Value) {
        self.write(
            &format!("profiles/{filename}"),
            json!({"info":info,"data":{"density_curves":null}}).to_string(),
        );
    }
    fn write_defaults(&self) {
        self.write_entry("film.json", json!({"stock":"kodak_portra_400"}));
        self.write_entry(
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

fn profile_document(role: Role) -> Value {
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

fn assert_field_error(error: ProfileError, field: &str, expected: Requirement) {
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
fn bundled_catalog_matches_reference() {
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
    let mut rows = RowCases::new("bundled-catalog-role-membership", ["film", "print"]);
    for (role, entries, expected) in [
        ("film", catalog.films(), FILM_CATALOG),
        ("print", catalog.prints(), PRINT_CATALOG),
    ] {
        let row = rows.case(role);
        for (entry, &(key, label)) in entries.iter().zip(expected) {
            assert_eq!((entry.key(), entry.label()), (key, label));
        }
        row.passed();
    }
    rows.finish();
}

#[test]
fn loads_bundled_profiles() {
    let catalog = load_catalog(&repository().join("Resources")).unwrap();
    let entries: Vec<_> = catalog
        .films()
        .iter()
        .map(|entry| (Role::Film, entry))
        .chain(catalog.prints().iter().map(|entry| (Role::Print, entry)))
        .collect();
    let ids: Vec<String> = entries
        .iter()
        .map(|(role, entry)| {
            format!(
                "{}/{}",
                if *role == Role::Film { "film" } else { "print" },
                entry.key()
            )
        })
        .collect();
    let mut rows = RowCases::new("bundled-profile-source-cases", &ids);
    for ((role, entry), id) in entries.iter().zip(&ids) {
        let row = rows.case(id);
        let (key, label) = (entry.key(), entry.label());
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
        row.passed();
    }
    rows.finish();
}

#[test]
fn profile_sources_match_reference() {
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
fn loads_default_metadata() {
    let dir = Directory::new();
    let mut role_cases = RowCases::new("profile-loads_default_metadata", ["film", "print"]);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let source = dir.load_source(&profile_document(role), role).unwrap();
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
        assert!(samples.hanatos2025_adaptation_window_params().is_none());
        assert!(samples.hanatos2025_adaptation_surface_params().is_none());

        row_case.passed();
    }
    role_cases.finish();
}

#[test]
fn preserves_sample_bits() {
    let dir = Directory::new();
    let mut role_cases = RowCases::new("profile-preserves_sample_bits", ["film", "print"]);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let source = dir.load_source(&profile_document(role), role).unwrap();
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

        row_case.passed();
    }
    role_cases.finish();
}

#[test]
fn rejects_invalid_metadata() {
    let dir = Directory::new();
    let string_fields = [
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
    ];
    let invalid_values = [
        ("null", Value::Null),
        ("number", json!(1)),
        ("boolean", json!(true)),
        ("array", json!([])),
        ("object", json!({})),
    ];
    let enum_fields = [
        "support",
        "stage",
        "type",
        "use",
        "antihalation",
        "channel_model",
    ];
    let mut ids: Vec<String> = string_fields
        .iter()
        .flat_map(|key| {
            invalid_values
                .iter()
                .map(move |(kind, _)| format!("info/{key}/wrong-type/{kind}"))
        })
        .collect();
    ids.extend(
        enum_fields
            .iter()
            .map(|key| format!("info/{key}/unknown-metadata")),
    );
    ids.push("info/stock/missing".to_owned());
    let mut cases = RowCases::new("invalid-profile-metadata", &ids);
    for key in string_fields {
        for (kind, value) in &invalid_values {
            let row = cases.case(format!("info/{key}/wrong-type/{kind}"));
            let mut input = profile_document(Role::Film);
            input["info"][key] = value.clone();
            assert_field_error(
                dir.load_source(&input, Role::Film).unwrap_err(),
                &format!("info.{key}"),
                Requirement::String,
            );
            row.passed();
        }
    }
    for key in enum_fields {
        let row = cases.case(format!("info/{key}/unknown-metadata"));
        let mut input = profile_document(Role::Film);
        input["info"][key] = json!("unknown");
        assert_field_error(
            dir.load_source(&input, Role::Film).unwrap_err(),
            &format!("info.{key}"),
            Requirement::MetadataValue,
        );
        row.passed();
    }
    let row = cases.case("info/stock/missing");
    let mut input = profile_document(Role::Film);
    input["info"].as_object_mut().unwrap().remove("stock");
    assert_field_error(
        dir.load_source(&input, Role::Film).unwrap_err(),
        "info.stock",
        Requirement::String,
    );
    row.passed();
    cases.finish();
}

#[test]
fn loads_empty_stock() {
    let dir = Directory::new();
    let mut input = profile_document(Role::Film);
    input["info"]["stock"] = json!("");
    assert_eq!(
        dir.load_source(&input, Role::Film).unwrap().info().stock(),
        ""
    );
}

#[test]
fn enforces_profile_role() {
    let dir = Directory::new();
    let mut input = profile_document(Role::Film);
    input["info"]["stock"] = json!("");
    input["info"]["support"] = json!("paper");
    assert_field_error(
        dir.load_source(&input, Role::Film).unwrap_err(),
        "info",
        Requirement::SelectedRole,
    );
    input["info"]["support"] = json!("film");
    assert_field_error(
        dir.load_source(&input, Role::Print).unwrap_err(),
        "info.stage",
        Requirement::SelectedRole,
    );
    input["info"]["stage"] = json!("printing");
    assert_field_error(
        dir.load_source(&input, Role::Film).unwrap_err(),
        "info",
        Requirement::SelectedRole,
    );
    let source = dir.load_source(&input, Role::Print).unwrap();
    assert_eq!(source.info().support(), Support::Film);
}

#[test]
fn loads_metadata_variants() {
    let dir = Directory::new();
    let mut input = profile_document(Role::Film);
    input["info"]["use"] = json!("cine");
    input["info"]["antihalation"] = json!("no");
    input["info"]["channel_model"] = json!("bw");
    input["info"]["type"] = json!("positive");
    let source = dir.load_source(&input, Role::Film).unwrap();
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
}

#[test]
fn preserves_illuminant_spelling() {
    let dir = Directory::new();
    let mut input = profile_document(Role::Film);
    let keys = ["reference_illuminant", "viewing_illuminant"];
    let accepted = [
        " d55 ",
        "th_kg3_l",
        "KINOTON75P",
        "D1",
        "BB3200",
        "BB+3200.5",
        "BB3.2e3",
        "BB1e-3",
        " bb3200.5 ",
    ];
    let ids: Vec<String> = keys
        .iter()
        .flat_map(|key| {
            accepted
                .iter()
                .map(move |value| format!("info/{key}/accepted/{}", value.trim()))
        })
        .collect();
    let mut cases = RowCases::new("accepted-illuminant-spellings", &ids);
    for key in keys {
        for accepted in accepted {
            let row = cases.case(format!("info/{key}/accepted/{}", accepted.trim()));
            input["info"][key] = json!(accepted);
            let source = dir.load_source(&input, Role::Film).unwrap();
            let actual = if key == "reference_illuminant" {
                source.info().reference_illuminant()
            } else {
                source.info().viewing_illuminant()
            };
            assert_eq!(actual, accepted);
            row.passed();
        }
        input["info"][key] = json!("D55");
    }
    cases.finish();
}

#[test]
fn rejects_invalid_illuminants() {
    let dir = Directory::new();
    let mut input = profile_document(Role::Film);
    let keys = ["reference_illuminant", "viewing_illuminant"];
    let rejected_values = ["D", "D-50", "BB1e309", "unknown"];
    let ids: Vec<String> = keys
        .iter()
        .flat_map(|key| {
            rejected_values
                .iter()
                .map(move |value| format!("info/{key}/rejected/{value}"))
        })
        .collect();
    let mut cases = RowCases::new("rejected-illuminant-spellings", &ids);
    for key in keys {
        for rejected in rejected_values {
            let row = cases.case(format!("info/{key}/rejected/{rejected}"));
            input["info"][key] = json!(rejected);
            assert_field_error(
                dir.load_source(&input, Role::Film).unwrap_err(),
                &format!("info.{key}"),
                Requirement::Illuminant,
            );
            row.passed();
        }
        input["info"][key] = json!("D55");
    }
    cases.finish();
}

#[test]
fn adaptation_nullability_follows_role() {
    let dir = Directory::new();
    let mut role_cases = RowCases::new(
        "profile-adaptation_nullability_follows_role",
        ["film", "print"],
    );
    let keys = [
        "hanatos2025_adaptation_window_params",
        "hanatos2025_adaptation_surface_params",
    ];
    let inner_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| keys.iter().map(move |key| format!("{role}/{key}")))
        .collect();
    let mut adaptation_cases = RowCases::new("adaptation-nullability-fields", &inner_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        for key in keys {
            let field_case = adaptation_cases.case(format!(
                "{}/{key}",
                if role == Role::Film { "film" } else { "print" }
            ));
            let mut input = profile_document(role);
            input["data"][key] = Value::Null;
            if role == Role::Print {
                assert!(dir.load_source(&input, role).is_ok());
            } else {
                assert_field_error(
                    dir.load_source(&input, role).unwrap_err(),
                    &format!("data.{key}"),
                    Requirement::AdaptationArray,
                );
            }
            input["data"][key] = json!([]);
            assert!(dir.load_source(&input, role).is_ok());
            field_case.passed();
        }

        row_case.passed();
    }
    adaptation_cases.finish();
    role_cases.finish();
}

#[test]
fn validates_adaptation_samples() {
    let dir = Directory::new();
    let mut role_cases = RowCases::new("profile-validates_adaptation_samples", ["film", "print"]);
    let keys = [
        "hanatos2025_adaptation_window_params",
        "hanatos2025_adaptation_surface_params",
    ];
    let inner_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| keys.iter().map(move |key| format!("{role}/{key}")))
        .collect();
    let mut adaptation_cases = RowCases::new("adaptation-sample-fields", &inner_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        for key in keys {
            let field_case = adaptation_cases.case(format!(
                "{}/{key}",
                if role == Role::Film { "film" } else { "print" }
            ));
            let mut input = profile_document(role);
            input["data"][key] = json!(false);
            assert_field_error(
                dir.load_source(&input, role).unwrap_err(),
                &format!("data.{key}"),
                Requirement::AdaptationArray,
            );
            let window = key.contains("window");
            input["data"][key] = json!([1, 2]);
            assert_field_error(
                dir.load_source(&input, role).unwrap_err(),
                &format!("data.{key}"),
                Requirement::ArrayLength(if window { 4 } else { 3 }),
            );
            input["data"][key] = if window {
                json!([1, 2, 3, 4])
            } else {
                json!(([[0.25; 15]; 3]))
            };
            let source = dir.load_source(&input, role).unwrap();
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
            assert_field_error(
                dir.load_source(&input, role).unwrap_err(),
                &field,
                Requirement::FiniteNumber,
            );
            input["data"][key] = if window {
                json!(([1e40; 4]))
            } else {
                json!(([[1e40; 15]; 3]))
            };
            let source = dir.load_source(&input, role).unwrap();
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
            field_case.passed();
        }

        row_case.passed();
    }
    adaptation_cases.finish();
    role_cases.finish();
}

#[test]
fn requires_consumed_sample_fields() {
    let dir = Directory::new();
    let mut role_cases =
        RowCases::new("profile-requires_consumed_sample_fields", ["film", "print"]);
    let keys = [
        "wavelengths",
        "log_sensitivity",
        "channel_density",
        "base_density",
    ];
    let inner_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| keys.iter().map(move |key| format!("{role}/{key}")))
        .collect();
    let mut sample_cases = RowCases::new("required-sample-fields", &inner_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        for key in keys {
            let field_case = sample_cases.case(format!(
                "{}/{key}",
                if role == Role::Film { "film" } else { "print" }
            ));
            let mut input = profile_document(role);
            input["data"].as_object_mut().unwrap().remove(key);
            assert_field_error(
                dir.load_source(&input, role).unwrap_err(),
                &format!("data.{key}"),
                Requirement::ArrayLength(81),
            );
            field_case.passed();
        }

        row_case.passed();
    }
    sample_cases.finish();
    role_cases.finish();
}

#[test]
fn validates_sample_dimensions() {
    let dir = Directory::new();
    let outer_fields = [
        "wavelengths",
        "log_sensitivity",
        "channel_density",
        "base_density",
    ];
    let inner_fields = ["log_sensitivity", "channel_density"];
    let ids: Vec<String> = outer_fields
        .iter()
        .map(|key| format!("outer/{key}/wrong-length"))
        .chain(
            inner_fields
                .iter()
                .map(|key| format!("inner/{key}/wrong-length")),
        )
        .collect();
    let mut cases = RowCases::new("sample-dimension-errors", &ids);
    for key in outer_fields {
        let row = cases.case(format!("outer/{key}/wrong-length"));
        let mut input = profile_document(Role::Film);
        input["data"][key] = json!([1]);
        assert_field_error(
            dir.load_source(&input, Role::Film).unwrap_err(),
            &format!("data.{key}"),
            Requirement::ArrayLength(81),
        );
        row.passed();
    }
    for key in inner_fields {
        let row = cases.case(format!("inner/{key}/wrong-length"));
        let mut input = profile_document(Role::Film);
        input["data"][key][0] = json!([1, 2]);
        assert_field_error(
            dir.load_source(&input, Role::Film).unwrap_err(),
            &format!("data.{key}[0]"),
            Requirement::ArrayLength(3),
        );
        row.passed();
    }
    cases.finish();
}

#[test]
fn preserves_sample_conversion() {
    let dir = Directory::new();
    let fields = ["log_sensitivity", "channel_density"];
    let ids: Vec<String> = fields
        .iter()
        .flat_map(|key| {
            ["reject-string", "f32-overflow-null"].map(|variant| format!("{key}/{variant}"))
        })
        .chain(std::iter::once("base_density/reject-boolean".to_owned()))
        .collect();
    let mut cases = RowCases::new("sample-conversion-inputs", &ids);
    for key in fields {
        let string_row = cases.case(format!("{key}/reject-string"));
        let mut input = profile_document(Role::Film);
        input["data"][key][0] = json!(["1", 2, 3]);
        assert_field_error(
            dir.load_source(&input, Role::Film).unwrap_err(),
            &format!("data.{key}[0][0]"),
            Requirement::FiniteNumber,
        );
        string_row.passed();
        let conversion_row = cases.case(format!("{key}/f32-overflow-null"));
        input["data"][key][0] = json!([1e40, null, -1e40]);
        assert_eq!(
            if key == "log_sensitivity" {
                dir.load_source(&input, Role::Film)
                    .unwrap()
                    .samples()
                    .log_sensitivity()[0]
                    .map(f32::to_bits)
            } else {
                dir.load_source(&input, Role::Film)
                    .unwrap()
                    .samples()
                    .channel_density()[0]
                    .map(f32::to_bits)
            },
            [0x7f800000, 0x7fc00000, 0xff800000]
        );
        conversion_row.passed();
    }
    let base_density_row = cases.case("base_density/reject-boolean");
    let mut input = profile_document(Role::Film);
    input["data"]["base_density"][0] = json!(true);
    assert_field_error(
        dir.load_source(&input, Role::Film).unwrap_err(),
        "data.base_density[0]",
        Requirement::FiniteNumber,
    );
    base_density_row.passed();
    cases.finish();
}

#[test]
fn preserves_authored_wavelength_positions() {
    let dir = Directory::new();
    let mut role_cases = RowCases::new(
        "profile-preserves_authored_wavelength_positions",
        ["film", "print"],
    );
    let invalid_values = [
        ("null", Value::Null),
        ("boolean", json!(true)),
        ("string", json!("381")),
    ];
    let invalid_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| {
            invalid_values
                .iter()
                .map(move |(kind, _)| format!("{role}/{kind}/wavelengths/0"))
        })
        .collect();
    let mut invalid_cases = RowCases::new("invalid-wavelength-samples", &invalid_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let mut input = profile_document(role);
        let labels: [f32; 81] = std::array::from_fn(|index| 381.0 + index as f32 * 5.0);
        input["data"]["wavelengths"] = json!(labels.as_slice());
        for index in 0..81 {
            input["data"]["log_sensitivity"][index] = json!([index, index + 81, index + 162]);
            input["data"]["channel_density"][index] =
                json!([index + 243, index + 324, index + 405]);
            input["data"]["base_density"][index] = json!(index + 486);
        }
        let source = dir.load_source(&input, role).unwrap();
        let samples = source.samples();
        assert_eq!(
            samples.wavelengths().map(f32::to_bits),
            labels.map(f32::to_bits)
        );
        for index in 0..81 {
            assert_eq!(
                samples.log_sensitivity()[index],
                [index as f32, (index + 81) as f32, (index + 162) as f32]
            );
            assert_eq!(
                samples.channel_density()[index],
                [
                    (index + 243) as f32,
                    (index + 324) as f32,
                    (index + 405) as f32
                ]
            );
            assert_eq!(samples.base_density()[index], (index + 486) as f32);
        }
        input["data"]["wavelengths"][0] = json!(380.000001);
        input["data"]["wavelengths"][1] = json!(1e40);
        input["data"]["wavelengths"][2] = json!(-0.0);
        let source = dir.load_source(&input, role).unwrap();
        assert_eq!(
            source.samples().wavelengths()[..3]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0x43be0000, 0x7f800000, 0x80000000]
        );
        for (kind, invalid) in &invalid_values {
            let invalid_case = invalid_cases.case(format!(
                "{}/{kind}/wavelengths/0",
                if role == Role::Film { "film" } else { "print" }
            ));
            input["data"]["wavelengths"][0] = invalid.clone();
            assert_field_error(
                dir.load_source(&input, role).unwrap_err(),
                "data.wavelengths[0]",
                Requirement::FiniteNumber,
            );
            invalid_case.passed();
        }

        row_case.passed();
    }
    invalid_cases.finish();
    role_cases.finish();
}

#[test]
fn preserves_authored_exposure_sequence() {
    let dir = Directory::new();
    // Inputs are the oracle: decoding must preserve every authored f64 bit and position.
    let sequences: [(&str, &[f64]); 8] = [
        ("single", &[1.0][..]),
        ("equal-zeroes", &[0.0, 0.0]),
        ("descending", &[2.0, 1.0, -1.0]),
        ("ascending-near-equal", &[1.00000001, 1.00000002]),
        ("descending-near-equal", &[1.00000002, 1.00000001]),
        ("signed-zero", &[1.0, -0.0, 0.0, -1.0]),
        ("wide-finite", &[-1e40, -1.0, 0.0, 1.0, 1e40]),
        ("opposite-wide", &[1e40, -1e40]),
    ];
    let mut role_cases = RowCases::new(
        "profile-preserves_authored_exposure_sequence",
        ["film", "print"],
    );
    let sequence_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| {
            sequences
                .iter()
                .map(move |(name, _)| format!("{role}/sequence/{name}"))
                .chain([
                    format!("{role}/narrowing/descending-double"),
                    format!("{role}/narrowing/f32-overflow"),
                ])
        })
        .collect();
    let mut sequence_cases = RowCases::new("authored-exposure-sequences", &sequence_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let mut input = profile_document(role);
        for (name, authored) in sequences {
            let sequence_case = sequence_cases.case(format!(
                "{}/sequence/{name}",
                if role == Role::Film { "film" } else { "print" }
            ));
            input["data"]["log_exposure"] = json!(authored);
            let source = dir.load_source(&input, role).unwrap();
            let exposure = source.samples().log_exposure();
            assert_eq!(
                exposure.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                authored.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{role:?} {authored:?}"
            );
            sequence_case.passed();
        }
        let descending_case = sequence_cases.case(format!(
            "{}/narrowing/descending-double",
            if role == Role::Film { "film" } else { "print" }
        ));
        input["data"]["log_exposure"] = json!([1.00000002, 1.00000001]);
        let source = dir.load_source(&input, role).unwrap();
        let exposure = source.samples().log_exposure();
        assert!(exposure[0] > exposure[1]);
        assert_eq!((exposure[0] as f32).to_bits(), 0x3f800000);
        assert_eq!((exposure[1] as f32).to_bits(), 0x3f800000);
        descending_case.passed();
        let overflow_case = sequence_cases.case(format!(
            "{}/narrowing/f32-overflow",
            if role == Role::Film { "film" } else { "print" }
        ));
        input["data"]["log_exposure"] = json!([-1e40, 1e40]);
        let source = dir.load_source(&input, role).unwrap();
        assert_eq!(
            source
                .samples()
                .log_exposure()
                .iter()
                .map(|v| (*v as f32).to_bits())
                .collect::<Vec<_>>(),
            [0xff800000, 0x7f800000]
        );
        overflow_case.passed();

        row_case.passed();
    }
    sequence_cases.finish();
    role_cases.finish();
}

#[test]
fn requires_exposure_array_and_numbers() {
    let dir = Directory::new();
    let mut role_cases = RowCases::new(
        "profile-requires_exposure_array_and_numbers",
        ["film", "print"],
    );
    let failure_cases = [
        ("null-array", Value::Null),
        ("number-array", json!(1)),
        ("empty-array", json!([])),
        ("null-sample", json!([null])),
        ("boolean-sample", json!([true])),
        ("string-sample", json!(["1"])),
        ("nested-array-sample", json!([[]])),
    ];
    let inner_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| {
            failure_cases
                .iter()
                .map(move |(id, _)| format!("{role}/{id}"))
                .chain(std::iter::once(format!("{role}/missing-array")))
        })
        .collect();
    let mut exposure_cases = RowCases::new("exposure-array-errors", &inner_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let mut input = profile_document(role);
        for (id, array) in &failure_cases {
            let error_case = exposure_cases.case(format!(
                "{}/{id}",
                if role == Role::Film { "film" } else { "print" }
            ));
            input["data"]["log_exposure"] = array.clone();
            let error = dir.load_source(&input, role).unwrap_err();
            assert_eq!(error.role, role);
            assert_eq!(error.stock.as_deref(), Some("authored"));
            let (field, requirement) = if id.ends_with("array") {
                ("data.log_exposure", Requirement::ExposureCount)
            } else {
                ("data.log_exposure[0]", Requirement::FiniteNumber)
            };
            assert_field_error(error, field, requirement);
            error_case.passed();
        }
        let missing_case = exposure_cases.case(format!(
            "{}/missing-array",
            if role == Role::Film { "film" } else { "print" }
        ));
        input["data"]
            .as_object_mut()
            .unwrap()
            .remove("log_exposure");
        assert_field_error(
            dir.load_source(&input, role).unwrap_err(),
            "data.log_exposure",
            Requirement::ExposureCount,
        );
        missing_case.passed();

        row_case.passed();
    }
    exposure_cases.finish();
    role_cases.finish();
}

#[test]
fn decodes_model_structure_and_numbers() {
    let dir = Directory::new();
    let fields = ["centers", "amplitudes", "sigmas"];
    let invalid_values = ["null", "boolean", "string"];
    let ids: Vec<String> = fields
        .iter()
        .flat_map(|key| {
            std::iter::once(format!("{key}/outer-shape"))
                .chain(
                    invalid_values
                        .iter()
                        .map(move |kind| format!("{key}/coefficient/{kind}")),
                )
                .chain([format!("{key}/row-width"), format!("{key}/missing-field")])
        })
        .collect();
    let mut cases = RowCases::new("density-model-structure-inputs", &ids);
    for key in fields {
        let outer_shape = cases.case(format!("{key}/outer-shape"));
        let mut input = profile_document(Role::Film);
        input["data"]["density_curves_model"][key] = json!([[1, 2, 3]]);
        assert_field_error(
            dir.load_source(&input, Role::Film).unwrap_err(),
            "data.density_curves_model",
            Requirement::ModelCoefficients,
        );
        outer_shape.passed();
        for (kind, invalid) in [
            ("null", Value::Null),
            ("boolean", json!(true)),
            ("string", json!("1")),
        ] {
            let row = cases.case(format!("{key}/coefficient/{kind}"));
            input = profile_document(Role::Film);
            input["data"]["density_curves_model"][key][0][0] = invalid;
            assert_field_error(
                dir.load_source(&input, Role::Film).unwrap_err(),
                "data.density_curves_model",
                Requirement::ModelCoefficients,
            );
            row.passed();
        }
        let row_width = cases.case(format!("{key}/row-width"));
        input = profile_document(Role::Film);
        input["data"]["density_curves_model"][key][0] = json!([1, 2]);
        assert_field_error(
            dir.load_source(&input, Role::Film).unwrap_err(),
            "data.density_curves_model",
            Requirement::ModelCoefficients,
        );
        row_width.passed();
        let missing_field = cases.case(format!("{key}/missing-field"));
        input = profile_document(Role::Film);
        input["data"]["density_curves_model"]
            .as_object_mut()
            .unwrap()
            .remove(key);
        assert_field_error(
            dir.load_source(&input, Role::Film).unwrap_err(),
            "data.density_curves_model",
            Requirement::ModelCoefficients,
        );
        missing_field.passed();
    }
    cases.finish();
}

#[test]
fn preserves_unusual_authored_coefficient_bits() {
    let dir = Directory::new();
    let mut role_cases = RowCases::new(
        "profile-preserves_unusual_authored_coefficient_bits",
        ["film", "print"],
    );
    let fields = ["centers", "amplitudes", "sigmas"];
    let raw_values = [
        ("positive-overflow", 1e40_f64),
        ("negative-overflow", -1e40),
        ("positive-underflow", 1e-50),
        ("negative-underflow", -1e-50),
        ("negative-one", -1.0),
        ("positive-zero", 0.0),
        ("negative-zero", -0.0),
    ];
    let mut coefficient_ids = Vec::new();
    for role in ["film", "print"] {
        for field in fields {
            for (value, _) in raw_values {
                coefficient_ids.push(format!("{role}/{field}/{value}"));
            }
        }
    }
    let mut coefficient_cases = RowCases::new("authored-density-coefficients", &coefficient_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        for key in fields {
            for (value_name, raw) in raw_values {
                let coefficient_case = coefficient_cases.case(format!(
                    "{}/{key}/{value_name}",
                    if role == Role::Film { "film" } else { "print" }
                ));
                let mut input = profile_document(role);
                input["data"]["density_curves_model"][key][2][1] = json!(raw);
                let source = dir.load_source(&input, role).unwrap();
                let model = source.density_model();
                let coefficients = match key {
                    "centers" => model.centers(),
                    "amplitudes" => model.amplitudes(),
                    "sigmas" => model.sigmas(),
                    _ => unreachable!(),
                };
                assert_eq!(
                    coefficients[2][1].to_bits(),
                    raw.to_bits(),
                    "{role:?} {key} {raw}"
                );
                coefficient_case.passed();
            }
        }

        row_case.passed();
    }
    coefficient_cases.finish();
    role_cases.finish();
}

#[test]
fn ignores_sampled_density_curves() {
    let dir = Directory::new();
    let mut input = profile_document(Role::Film);
    input["data"]["density_curves"] = json!("unused authored curves");
    input["data"]["density_curves_layers"] = json!(false);
    input["data"]["density_curves_model"]["model_type"] = json!("unused");
    input["data"]["density_curves_model"]["amplitudes"][0][0] = json!(-1.0);
    assert_eq!(
        dir.load_source(&input, Role::Film)
            .unwrap()
            .density_model()
            .amplitudes()[0][0],
        -1.0
    );
}

#[test]
fn reports_source_errors() {
    let dir = Directory::new();
    let json_inputs = ["malformed-object", "trailing-value", "unused-overflow"];
    let root_inputs = ["array", "null", "number"];
    let ids: Vec<String> = std::iter::once("missing-file".to_owned())
        .chain(json_inputs.iter().map(|id| format!("json/{id}")))
        .chain(root_inputs.iter().map(|id| format!("root/{id}")))
        .chain([
            "missing-data-object".to_owned(),
            "missing-density-model".to_owned(),
        ])
        .collect();
    let mut cases = RowCases::new("profile-source-error-inputs", &ids);
    let missing_file = cases.case("missing-file");
    let error = load_film_source(&dir.0.join("missing.json")).unwrap_err();
    assert!(matches!(error.kind, ProfileErrorKind::Read(_)));
    missing_file.passed();
    for (id, text) in json_inputs
        .into_iter()
        .zip(["{", "true trailing", "{\"unused\":1e999}"])
    {
        let row = cases.case(format!("json/{id}"));
        let path = dir.write("source.json", text);
        assert!(matches!(
            load_film_source(&path).unwrap_err().kind,
            ProfileErrorKind::Json(_)
        ));
        row.passed();
    }
    for (id, text) in root_inputs.into_iter().zip(["[]", "null", "1"]) {
        let row = cases.case(format!("root/{id}"));
        let path = dir.write("source.json", text);
        assert_field_error(
            load_film_source(&path).unwrap_err(),
            "info",
            Requirement::Object,
        );
        row.passed();
    }
    let missing_data = cases.case("missing-data-object");
    let mut input = profile_document(Role::Film);
    input.as_object_mut().unwrap().remove("data");
    assert_field_error(
        dir.load_source(&input, Role::Film).unwrap_err(),
        "data",
        Requirement::Object,
    );
    missing_data.passed();
    let missing_model = cases.case("missing-density-model");
    input = profile_document(Role::Film);
    input["data"]
        .as_object_mut()
        .unwrap()
        .remove("density_curves_model");
    assert_field_error(
        dir.load_source(&input, Role::Film).unwrap_err(),
        "data.density_curves_model",
        Requirement::Object,
    );
    missing_model.passed();
    cases.finish();
}

#[test]
fn preserves_decimal_conversion_bits() {
    let dir = Directory::new();
    let mut input = profile_document(Role::Film);
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
fn accepts_json_bom() {
    let dir = Directory::new();
    let text = serde_json::to_string(&profile_document(Role::Film)).unwrap();
    let path = dir.write("source.json", format!("\u{feff}{text}"));
    assert_eq!(load_film_source(&path).unwrap().info().stock(), "authored");
}

#[test]
fn last_json_member_wins() {
    let dir = Directory::new();
    let text = serde_json::to_string(&profile_document(Role::Film))
        .unwrap()
        .replace(
            "\"stock\":\"authored\"",
            "\"stock\":1,\"stock\":\"authored\"",
        );
    let variants = [
        ("plain", text.clone()),
        ("bom-prefixed", format!("\u{feff}{text}")),
    ];
    let mut rows = RowCases::new(
        "last-json-member-variants",
        variants.iter().map(|(id, _)| *id),
    );
    for (id, text) in variants {
        let row = rows.case(id);
        let path = dir.write("source.json", text);
        assert_eq!(load_film_source(&path).unwrap().info().stock(), "authored");
        row.passed();
    }
    rows.finish();
}

#[test]
fn classifies_catalog_entries() {
    let dir = Directory::new();
    dir.write_defaults();
    dir.write("profiles/ignored.json", "{}");
    dir.write("profiles/bad.json", "{");
    dir.write(
        "profiles/model-only.json",
        r#"{"data":{"density_curves_model":{}}}"#,
    );
    dir.write_entry("no-stock.json", json!({}));
    dir.write_entry("null-support.json", json!({"stock":"null","support":null}));
    dir.write_entry(
        "paper-filming.json",
        json!({"stock":"paper","support":"paper"}),
    );
    dir.write_entry("uppercase-only.JSON", json!({"stock":"wrong-extension"}));
    dir.write_entry("nested/child.json", json!({"stock":"nested"}));
    dir.write_entry("a.json", json!({"stock":"a","name":"Same"}));
    dir.write_entry("b.json", json!({"stock":"b","name":"Same"}));
    dir.write_entry("label-number.json", json!({"stock":"label","name":3}));
    dir.write_entry("empty-label.json", json!({"stock":"empty","name":""}));
    dir.write_entry(
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
        ["empty", "a", "b", "kodak_portra_400", "label"]
    );
    assert_eq!(
        catalog.prints().iter().map(|e| e.key()).collect::<Vec<_>>(),
        ["a", "kodak_portra_endura"]
    );
    assert_eq!(catalog.ignored().len(), 4);
    assert_eq!(catalog.unavailable().len(), 5);
    let ids: Vec<String> = catalog
        .unavailable()
        .iter()
        .map(|entry| {
            entry
                .path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    let mut rows = RowCases::new("unavailable-catalog-entries", &ids);
    for (entry, id) in catalog.unavailable().iter().zip(&ids) {
        let row = rows.case(id);
        match entry.path.file_name().unwrap().to_str().unwrap() {
            "bad.json" | "trailing.json" => {
                assert!(matches!(entry.kind, CatalogEntryErrorKind::Json(_)))
            }
            "no-stock.json" => assert!(matches!(entry.kind, CatalogEntryErrorKind::MissingStock)),
            _ => assert!(matches!(entry.kind, CatalogEntryErrorKind::UnsupportedRole)),
        }
        row.passed();
    }
    rows.finish();
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
fn catalog_defers_sample_decoding_until_selection() {
    let dir = Directory::new();
    dir.write_defaults();
    let mut role_cases = RowCases::new(
        "profile-catalog_defers_sample_decoding_until_selection",
        ["film", "print"],
    );
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let mut input = profile_document(role);
        let key = if role == Role::Film {
            "trusted-film"
        } else {
            "trusted-print"
        };
        input["info"]["stock"] = json!(key);
        input["info"]["reference_illuminant"] = json!("BB-0");
        input["data"]["wavelengths"] = json!((381..=781).step_by(5).collect::<Vec<_>>());
        input["data"]["log_exposure"] = json!([1e40, -1e40]);
        dir.write(&format!("profiles/{key}.json"), input.to_string());
        let malformed_key = format!("malformed-{key}");
        input["info"]["stock"] = json!(malformed_key);
        input["data"]["log_exposure"] = json!([]);
        dir.write(&format!("profiles/{malformed_key}.json"), input.to_string());

        let catalog = load_catalog(&dir.0).unwrap();
        assert!(catalog.unavailable().is_empty());
        let (entry, malformed) = match role {
            Role::Film => (catalog.film(key), catalog.film(&malformed_key)),
            Role::Print => (catalog.print(key), catalog.print(&malformed_key)),
        };
        let load = match role {
            Role::Film => load_film_source,
            Role::Print => load_print_source,
        };
        let source = load(entry.unwrap().source_path()).unwrap();
        assert_eq!(source.info().stock(), key);
        assert_eq!(source.samples().log_exposure(), &[1e40, -1e40]);
        let error = load(malformed.unwrap().source_path()).unwrap_err();
        assert_eq!(error.stock.as_deref(), Some(malformed_key.as_str()));
        assert!(matches!(
            error.kind,
            ProfileErrorKind::Field {
                requirement: Requirement::ExposureCount,
                ..
            }
        ));
        assert_eq!(catalog.default_film().key(), "kodak_portra_400");
        assert_eq!(catalog.default_print().key(), "kodak_portra_endura");

        row_case.passed();
    }
    role_cases.finish();
}

#[test]
fn rejects_catalog_duplicates_per_role() {
    let mut role_cases = RowCases::new(
        "profile-rejects_catalog_duplicates_per_role",
        ["film", "print"],
    );
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let dir = Directory::new();
        dir.write_defaults();
        let (key, stage) = if role == Role::Film {
            ("kodak_portra_400", "filming")
        } else {
            ("kodak_portra_endura", "printing")
        };
        dir.write_entry("duplicate.json", json!({"stock":key,"stage":stage}));
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

        row_case.passed();
    }
    role_cases.finish();
    let dir = Directory::new();
    dir.write_defaults();
    dir.write_entry(
        "invisible-duplicate.json",
        json!({"stock":"kodak_portra_400","support":"paper"}),
    );
    assert_eq!(load_catalog(&dir.0).unwrap().unavailable().len(), 1);
}

#[test]
fn catalog_requires_roles_and_defaults() {
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
    dir.write_entry("film.json", json!({"stock":"film"}));
    assert!(matches!(
        load_catalog(&dir.0),
        Err(CatalogError::EmptyRole {
            role: Role::Print,
            ..
        })
    ));
    dir.write_entry("print.json", json!({"stock":"print","stage":"printing"}));
    assert!(matches!(
        load_catalog(&dir.0),
        Err(CatalogError::MissingDefault {
            role: Role::Film,
            ..
        })
    ));
    dir.write_entry("film.json", json!({"stock":"kodak_portra_400"}));
    assert!(matches!(
        load_catalog(&dir.0),
        Err(CatalogError::MissingDefault {
            role: Role::Print,
            ..
        })
    ));
    dir.write_entry(
        "print.json",
        json!({"stock":"kodak_portra_endura","stage":"printing"}),
    );
    assert!(load_catalog(&dir.0).is_ok());
}

#[test]
fn blackbody_requires_finite_decimal() {
    let dir = Directory::new();
    let labels = [
        ("BB3200", true),
        (" bb+3200.5\t", true),
        ("Bb3.2e+3", true),
        ("BB1e-0", true),
        ("BB1.5e308", true),
        ("BB5e-324", true),
        ("BB0.1e-323", true),
        ("BB1e309", false),
        ("BB0", true),
        ("BB-0", true),
        ("BB-3200", true),
        (" \tBb-3200.5\r\n", true),
        ("BB-1.5e-3", true),
        ("BB1e-999", true),
        ("BB-1e-999", true),
        ("BB-1e309", false),
        ("BBNaN", false),
        ("BBinf", false),
        ("BBInfinity", false),
        ("BB-inf", false),
        ("BB+NaN", false),
        ("BB+", false),
        ("BB-", false),
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
    ];
    let mut role_cases = RowCases::new(
        "profile-blackbody_requires_finite_decimal",
        ["film", "print"],
    );
    let mut inner_ids = Vec::new();
    for role in ["film", "print"] {
        for key in ["reference_illuminant", "viewing_illuminant"] {
            for (label, _) in labels {
                inner_ids.push(format!("{role}/{key}/label-{label:?}"));
            }
        }
    }
    let mut illuminant_cases = RowCases::new("blackbody-illuminant-labels", &inner_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let mut input = profile_document(role);
        for key in ["reference_illuminant", "viewing_illuminant"] {
            // Product-contract expectations, including cases whose validity would
            // change if decimal points, exponent signs or punctuation were removed.
            for (illuminant, accepted) in labels {
                let illuminant_case = illuminant_cases.case(format!(
                    "{}/{key}/label-{illuminant:?}",
                    if role == Role::Film { "film" } else { "print" }
                ));
                input["info"][key] = json!(illuminant);
                let result = dir.load_source(&input, role);
                assert_eq!(result.is_ok(), accepted, "{role:?} {key} {illuminant}");
                match result {
                    Ok(source) => {
                        let actual = if key == "reference_illuminant" {
                            source.info().reference_illuminant()
                        } else {
                            source.info().viewing_illuminant()
                        };
                        assert_eq!(actual, illuminant);
                    }
                    Err(error) => {
                        assert_eq!(error.role, role);
                        assert_field_error(error, &format!("info.{key}"), Requirement::Illuminant);
                    }
                }
                illuminant_case.passed();
            }
            input["info"][key] = json!("D55");
        }

        row_case.passed();
    }
    illuminant_cases.finish();
    role_cases.finish();
}

#[test]
fn preserves_json_number_bits() {
    let dir = Directory::new();
    dir.write_defaults();
    let number_tokens = [
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
    ];
    let mut role_cases = RowCases::new("profile-preserves_json_number_bits", ["film", "print"]);
    let token_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| {
            number_tokens
                .iter()
                .map(move |(token, _, _)| format!("{role}/{token}"))
        })
        .collect();
    let mut token_cases = RowCases::new("preserved-json-number-tokens", &token_ids);
    let sample_names = [
        "log-sensitivity",
        "channel-density",
        "base-density",
        "adaptation-window",
    ];
    let mut sample_ids = Vec::new();
    for role in ["film", "print"] {
        for (token, _, _) in number_tokens {
            for sample in sample_names {
                sample_ids.push(format!("{role}/{token}/{sample}"));
            }
        }
    }
    let mut sample_cases = RowCases::new("preserved-json-f32-samples", &sample_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let mut input = profile_document(role);
        input["info"]["stock"] = json!(r#"quoted " -0, slash \ 1e-0]"#);
        input["data"]["log_exposure"] = json!(["TOKEN"]);
        input["data"]["log_sensitivity"][0][0] = json!("TOKEN");
        input["data"]["channel_density"][0][0] = json!("TOKEN");
        input["data"]["base_density"][0] = json!("TOKEN");
        input["data"]["density_curves_model"]["centers"][0][0] = json!("TOKEN");
        input["data"]["density_curves_model"]["amplitudes"][0][0] = json!("TOKEN");
        input["data"]["hanatos2025_adaptation_window_params"] = json!(["TOKEN", 1, 2, 3]);
        input["unused"] = json!({"$serde_json::private::Number":"not a number"});
        for &(token, bits_f64, bits_f32) in &number_tokens {
            let token_case = token_cases.case(format!(
                "{}/{token}",
                if role == Role::Film { "film" } else { "print" }
            ));
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
            let sample_values = [
                ("log-sensitivity", source.samples().log_sensitivity()[0][0]),
                ("channel-density", source.samples().channel_density()[0][0]),
                ("base-density", source.samples().base_density()[0]),
                (
                    "adaptation-window",
                    source
                        .samples()
                        .hanatos2025_adaptation_window_params()
                        .unwrap()[0],
                ),
            ];
            for (sample_name, sample) in sample_values {
                let sample_case = sample_cases.case(format!(
                    "{}/{token}/{sample_name}",
                    if role == Role::Film { "film" } else { "print" }
                ));
                assert_eq!(sample.to_bits(), bits_f32, "{role:?} {token}");
                sample_case.passed();
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
            token_case.passed();
        }

        row_case.passed();
    }
    sample_cases.finish();
    token_cases.finish();
    role_cases.finish();
}

#[test]
fn requires_complete_json_documents() {
    let dir = Directory::new();
    dir.write_defaults();
    let invalid_documents = [
        ("unfinished-exponent", "1e"),
        ("overflow-exponent", "1e309"),
        ("nan-token", "NaN"),
        ("infinity-token", "Infinity"),
        ("truncated-document", "truncate"),
        ("trailing-token", "trailing"),
        ("second-document", "second-document"),
    ];
    let mut role_cases = RowCases::new(
        "profile-requires_complete_json_documents",
        ["film", "print"],
    );
    let document_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| {
            invalid_documents
                .iter()
                .map(move |(id, _)| format!("{role}/{id}"))
                .chain(std::iter::once(format!("{role}/trailing-whitespace")))
        })
        .collect();
    let mut document_cases = RowCases::new("complete-json-document-cases", &document_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        let mut input = profile_document(role);
        input["data"]["log_exposure"] = json!(["TOKEN"]);
        let text = serde_json::to_string(&input).unwrap();
        let complete = text.replace("\"TOKEN\"", "0");
        for (id, replacement) in invalid_documents {
            let document_case = document_cases.case(format!(
                "{}/{id}",
                if role == Role::Film { "film" } else { "print" }
            ));
            let malformed = match replacement {
                "truncate" => complete[..complete.len() - 1].to_owned(),
                "trailing" => format!("{complete} trailing"),
                "second-document" => format!("{complete} {{}}"),
                token => text.replace("\"TOKEN\"", token),
            };
            let path = dir.write("source.json", &malformed);
            let error = match role {
                Role::Film => load_film_source(&path),
                Role::Print => load_print_source(&path),
            }
            .unwrap_err();
            assert!(matches!(error.kind, ProfileErrorKind::Json(_)));
            dir.write("profiles/numbers.json", malformed);
            let catalog = load_catalog(&dir.0).unwrap();
            assert_eq!(catalog.films().len(), 1);
            assert_eq!(catalog.prints().len(), 1);
            assert_eq!(catalog.unavailable().len(), 1);
            assert!(matches!(
                catalog.unavailable()[0].kind,
                CatalogEntryErrorKind::Json(_)
            ));
            document_case.passed();
        }
        let whitespace_case = document_cases.case(format!(
            "{}/trailing-whitespace",
            if role == Role::Film { "film" } else { "print" }
        ));
        let text = format!("{complete} \t\r\n");
        let path = dir.write("source.json", &text);
        match role {
            Role::Film => load_film_source(&path),
            Role::Print => load_print_source(&path),
        }
        .unwrap();
        dir.write("profiles/numbers.json", text);
        let catalog = load_catalog(&dir.0).unwrap();
        assert!(catalog.unavailable().is_empty());
        assert_eq!(catalog.films().len() + catalog.prints().len(), 3);
        whitespace_case.passed();

        row_case.passed();
    }
    document_cases.finish();
    role_cases.finish();
}

#[test]
fn preserves_model_channel_layer_order() {
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
fn catalog_recognizes_profile_fields() {
    let dir = Directory::new();
    dir.write_defaults();
    let fields = [
        "wavelengths",
        "log_sensitivity",
        "channel_density",
        "base_density",
        "log_exposure",
        "density_curves",
    ];
    let mut cases = RowCases::new("catalog-profile-field-recognition", fields);
    for key in fields {
        let mut data = serde_json::Map::new();
        data.insert(key.to_owned(), Value::Null);
        dir.write(
            &format!("profiles/{key}.json"),
            json!({"info":{"stock":key},"data":data}).to_string(),
        );
    }
    let catalog = load_catalog(&dir.0).unwrap();
    assert_eq!(catalog.films().len(), 7);
    assert!(catalog.unavailable().is_empty());
    for key in fields {
        let row = cases.case(key);
        assert!(
            catalog.film(key).is_some(),
            "catalog omitted field fixture {key}"
        );
        row.passed();
    }
    cases.finish();
}

#[test]
fn catalog_rejects_null_metadata() {
    let dir = Directory::new();
    dir.write_defaults();
    let fields = ["support", "stage", "type"];
    let mut cases = RowCases::new("catalog-null-metadata-fields", fields);
    for key in fields {
        let mut info = json!({"stock":key});
        info[key] = Value::Null;
        dir.write_entry(&format!("null-{key}.json"), info);
    }
    let catalog = load_catalog(&dir.0).unwrap();
    assert_eq!(catalog.unavailable().len(), 3);
    assert!(
        catalog
            .unavailable()
            .iter()
            .all(|entry| matches!(entry.kind, CatalogEntryErrorKind::UnsupportedRole))
    );
    for key in fields {
        let row = cases.case(key);
        let filename = format!("null-{key}.json");
        let entry = catalog
            .unavailable()
            .iter()
            .find(|entry| {
                entry.path.file_name().and_then(|name| name.to_str()) == Some(filename.as_str())
            })
            .expect("catalog omitted null-metadata field fixture");
        assert!(matches!(entry.kind, CatalogEntryErrorKind::UnsupportedRole));
        row.passed();
    }
    cases.finish();
}

#[test]
fn illuminant_kinds_preserve_original_decimal_bits_and_named_precedence() {
    let directory = Directory::new();
    let blackbody_labels = [
        (" BB+3200.5 ", 3200.5_f64),
        ("bb3.2005e3", 3200.5),
        ("BB0", 0.0),
        ("BB-0.0", -0.0),
        ("BB-3200", -3200.0),
        ("BB1e-400", 0.0),
    ];
    let named_labels = ["D65", "TH_KG3", "D1234", "KINOTON75P"];
    let mut role_cases = RowCases::new(
        "profile-illuminant_kinds_preserve_original_decimal_bits_and_named_precedence",
        ["film", "print"],
    );
    let mut blackbody_ids = Vec::new();
    for role in ["film", "print"] {
        for key in ["reference_illuminant", "viewing_illuminant"] {
            for (label, _) in blackbody_labels {
                blackbody_ids.push(format!("{role}/{key}/label-{label:?}"));
            }
        }
    }
    let mut blackbody_cases = RowCases::new("illuminant-blackbody-label-kinds", &blackbody_ids);
    let named_ids: Vec<String> = ["film", "print"]
        .into_iter()
        .flat_map(|role| {
            named_labels
                .iter()
                .map(move |label| format!("{role}/viewing/{label}"))
        })
        .collect();
    let mut named_cases = RowCases::new("illuminant-named-label-kinds", &named_ids);
    for role in [Role::Film, Role::Print] {
        let row_case = role_cases.case(if role == Role::Film { "film" } else { "print" });
        for (label, expected) in blackbody_labels {
            for key in ["reference_illuminant", "viewing_illuminant"] {
                let label_case = blackbody_cases.case(format!(
                    "{}/{key}/label-{label:?}",
                    if role == Role::Film { "film" } else { "print" }
                ));
                let mut document = profile_document(role);
                document["info"]["reference_illuminant"] = json!(label);
                document["info"]["viewing_illuminant"] = json!(label);
                let source = directory.load_source(&document, role).unwrap();
                let (original, kind) = if key == "reference_illuminant" {
                    (
                        source.info().reference_illuminant(),
                        source.info().reference_illuminant_kind(),
                    )
                } else {
                    (
                        source.info().viewing_illuminant(),
                        source.info().viewing_illuminant_kind(),
                    )
                };
                assert_eq!(original, label);
                match kind {
                    IlluminantKind::Blackbody { temperature_kelvin } => {
                        assert_eq!(temperature_kelvin.to_bits(), expected.to_bits())
                    }
                    IlluminantKind::Named => panic!("BB must retain its decimal interpretation"),
                }
                label_case.passed();
            }
        }
        for label in named_labels {
            let label_case = named_cases.case(format!(
                "{}/viewing/{label}",
                if role == Role::Film { "film" } else { "print" }
            ));
            let mut document = profile_document(role);
            document["info"]["viewing_illuminant"] = json!(label);
            let source = directory.load_source(&document, role).unwrap();
            assert_eq!(
                source.info().viewing_illuminant_kind(),
                IlluminantKind::Named
            );
            assert_eq!(source.info().viewing_illuminant(), label);
            label_case.passed();
        }

        row_case.passed();
    }
    blackbody_cases.finish();
    named_cases.finish();
    role_cases.finish();
}
