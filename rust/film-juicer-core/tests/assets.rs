//! Asset ownership contracts; numerical identities use the existing frozen C4/C5 captures.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use film_juicer_core::assets::{AssetError, Assets};
use film_juicer_core::profile::{CatalogError, ProfileCompletionErrorKind, ProfileErrorKind, Role};

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
            "assets-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn write(&self, file: &str, document: &Value) {
        fs::create_dir_all(self.0.join("profiles")).unwrap();
        fs::write(self.0.join("profiles").join(file), document.to_string()).unwrap();
    }

    fn defaults(&self) {
        self.write("film.json", &document("kodak_portra_400", Role::Film));
        self.write("print.json", &document("kodak_portra_endura", Role::Print));
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn document(key: &str, role: Role) -> Value {
    json!({
        "info": {
            "stock": key,
            "stage": if role == Role::Film { "filming" } else { "printing" },
        },
        "data": {
            "wavelengths": ([380.0; 81].as_slice()),
            "log_sensitivity": ([[0.0; 3]; 81].as_slice()),
            "channel_density": ([[0.0; 3]; 81].as_slice()),
            "base_density": ([0.0; 81].as_slice()),
            "log_exposure": [-1.0, 0.0, 1.0],
            "density_curves_model": {
                "centers": ([[0.0; 3]; 3]),
                "amplitudes": ([[1.0; 3]; 3]),
                "sigmas": ([[1.0; 3]; 3]),
            },
        },
    })
}

#[test]
fn construction_and_unused_release_do_no_discovery() {
    let directory = Directory::new();
    let assets = Assets::new(directory.0.clone());
    assets.release_cached_payloads().unwrap();
    directory.defaults();
    let catalog = assets.catalog().unwrap();
    assert_eq!(catalog.default_film().key(), "kodak_portra_400");
    assert_eq!(catalog.default_print().key(), "kodak_portra_endura");
}

#[test]
fn catalog_and_film_do_not_complete_unselected_print() {
    let directory = Directory::new();
    directory.defaults();
    // Discoverable metadata with an unusable payload preserves catalog policy.
    directory.write(
        "print.json",
        &json!({
            "info": {"stock": "kodak_portra_endura", "stage": "printing"},
            "data": {"wavelengths": []},
        }),
    );
    let assets = Assets::new(directory.0.clone());
    assert_eq!(assets.catalog().unwrap().prints().len(), 1);
    assert!(assets.film("kodak_portra_400").is_ok());
    assert!(matches!(
        assets.print("kodak_portra_endura"),
        Err(AssetError::ProfileDecode(_))
    ));
    // No NPY, calibration, CSV or noise files are present in this root.
    assert!(assets.film("kodak_portra_400").is_ok());
}

#[test]
fn role_and_authored_key_select_the_resolved_path() {
    let directory = Directory::new();
    directory.defaults();
    directory.write("unrelated-film-name.json", &document("shared", Role::Film));
    directory.write(
        "unrelated-print-name.json",
        &document("shared", Role::Print),
    );
    let assets = Assets::new(directory.0.clone());
    let film = assets.film("shared").unwrap();
    let print = assets.print("shared").unwrap();
    assert_eq!(film.info().stock(), "shared");
    assert_eq!(print.info().stock(), "shared");
    for (role, key) in [
        (Role::Film, "kodak_portra_endura"),
        (Role::Print, "kodak_portra_400"),
        (Role::Film, "film"),
        (Role::Print, "SHARED"),
    ] {
        let error = match role {
            Role::Film => assets.film(key).unwrap_err(),
            Role::Print => assets.print(key).unwrap_err(),
        };
        assert!(
            matches!(error, AssetError::MissingProfile { role: actual_role, key: actual_key } if actual_role == role && actual_key == key)
        );
    }
}

#[test]
fn successful_hits_share_payloads_without_reparsing() {
    let directory = Directory::new();
    directory.defaults();
    let assets = Assets::new(directory.0.clone());
    let film = assets.film("kodak_portra_400").unwrap();
    let print = assets.print("kodak_portra_endura").unwrap();
    fs::remove_dir_all(directory.0.join("profiles")).unwrap();
    let reused_film = assets.film("kodak_portra_400").unwrap();
    let reused_print = assets.print("kodak_portra_endura").unwrap();
    assert!(Arc::ptr_eq(&film, &reused_film));
    assert!(Arc::ptr_eq(&print, &reused_print));
    assert_eq!(
        film.tables().source_log_exposure().as_ptr(),
        reused_film.tables().source_log_exposure().as_ptr()
    );
    assert_eq!(
        print.tables().density_curves().as_ptr(),
        reused_print.tables().density_curves().as_ptr()
    );
}

#[test]
fn decode_failure_publishes_nothing_and_explicit_requests_retry() {
    let directory = Directory::new();
    directory.defaults();
    let assets = Assets::new(directory.0.clone());
    assets.catalog().unwrap();
    fs::write(directory.0.join("profiles/film.json"), "{").unwrap();
    fs::remove_file(directory.0.join("profiles/print.json")).unwrap();
    assert!(
        matches!(assets.film("kodak_portra_400"), Err(AssetError::ProfileDecode(error)) if error.role == Role::Film && matches!(error.kind, ProfileErrorKind::Json(_)))
    );
    assert!(
        matches!(assets.print("kodak_portra_endura"), Err(AssetError::ProfileDecode(error)) if error.role == Role::Print && matches!(error.kind, ProfileErrorKind::Read(_)))
    );
    directory.defaults();
    assert!(assets.film("kodak_portra_400").is_ok());
    assert!(assets.print("kodak_portra_endura").is_ok());
}

#[test]
fn failed_completion_is_not_cached_or_replaced_by_another_key() {
    let directory = Directory::new();
    directory.defaults();
    let mut film = document("broken-film", Role::Film);
    film["data"]["log_exposure"] = json!([1.0, -1.0]);
    directory.write("broken-film.json", &film);
    let mut print = document("broken-print", Role::Print);
    print["data"]["density_curves_model"]["amplitudes"][0][0] = json!(1e40);
    directory.write("broken-print.json", &print);
    let assets = Assets::new(directory.0.clone());
    assert!(assets.film("kodak_portra_400").is_ok());
    assert!(assets.print("kodak_portra_endura").is_ok());
    assert!(
        matches!(assets.film("broken-film"), Err(AssetError::ProfileCompletion(error)) if error.role == Role::Film && error.stock == "broken-film" && error.kind == ProfileCompletionErrorKind::DescendingAxis { index: 1 })
    );
    assert!(
        matches!(assets.print("broken-print"), Err(AssetError::ProfileCompletion(error)) if error.role == Role::Print && error.stock == "broken-print" && matches!(error.kind, ProfileCompletionErrorKind::Density { .. }))
    );
    directory.write("broken-film.json", &document("broken-film", Role::Film));
    directory.write("broken-print.json", &document("broken-print", Role::Print));
    assert_eq!(
        assets.film("broken-film").unwrap().info().stock(),
        "broken-film"
    );
    assert_eq!(
        assets.print("broken-print").unwrap().info().stock(),
        "broken-print"
    );
}

#[test]
fn catalog_success_is_fixed_across_release_and_directory_changes() {
    let directory = Directory::new();
    directory.defaults();
    let assets = Assets::new(directory.0.clone());
    let catalog = assets.catalog().unwrap();
    directory.write("later.json", &document("later", Role::Film));
    assets.release_cached_payloads().unwrap();
    assert!(Arc::ptr_eq(&catalog, &assets.catalog().unwrap()));
    assert!(matches!(
        assets.film("later"),
        Err(AssetError::MissingProfile { .. })
    ));
    assert!(Assets::new(directory.0.clone()).film("later").is_ok());
    drop(assets);
    assert_eq!(catalog.default_film().key(), "kodak_portra_400");
}

#[test]
fn catalog_error_is_shared_and_survives_repair_and_release() {
    let directory = Directory::new();
    let assets = Assets::new(directory.0.clone());
    let first = match assets.catalog().unwrap_err() {
        AssetError::Catalog(error) => error,
        error => panic!("{error}"),
    };
    assert!(matches!(first.as_ref(), CatalogError::Directory { .. }));
    directory.defaults();
    assets.release_cached_payloads().unwrap();
    for error in [
        assets.catalog().unwrap_err(),
        assets.film("kodak_portra_400").unwrap_err(),
        assets.print("kodak_portra_endura").unwrap_err(),
    ] {
        assert!(matches!(error, AssetError::Catalog(error) if Arc::ptr_eq(&first, &error)));
    }
    assert!(Assets::new(directory.0.clone()).catalog().is_ok());
    drop(assets);
    assert!(!first.to_string().is_empty());
}

#[test]
fn catalog_still_requires_default_print_availability() {
    let directory = Directory::new();
    directory.write("film.json", &document("kodak_portra_400", Role::Film));
    directory.write("print.json", &document("other-print", Role::Print));
    let assets = Assets::new(directory.0.clone());
    assert!(
        matches!(assets.film("kodak_portra_400"), Err(AssetError::Catalog(error)) if matches!(error.as_ref(), CatalogError::MissingDefault { role: Role::Print, key: "kodak_portra_endura", .. }))
    );
}

#[test]
fn handles_survive_release_and_owner_drop_with_fresh_reacquisition() {
    let directory = Directory::new();
    directory.defaults();
    let assets = Assets::new(directory.0.clone());
    let film = assets.film("kodak_portra_400").unwrap();
    let print = assets.print("kodak_portra_endura").unwrap();
    let tokens = (film.asset_token(), print.asset_token());
    assets.release_cached_payloads().unwrap();
    assert_eq!(Arc::strong_count(&film), 1);
    assert_eq!(Arc::strong_count(&print), 1);
    let new_film = assets.film("kodak_portra_400").unwrap();
    let new_print = assets.print("kodak_portra_endura").unwrap();
    assert!(!Arc::ptr_eq(&film, &new_film));
    assert!(!Arc::ptr_eq(&print, &new_print));
    assert_eq!(tokens, (new_film.asset_token(), new_print.asset_token()));
    let released_film = Arc::downgrade(&new_film);
    let released_print = Arc::downgrade(&new_print);
    drop(new_film);
    drop(new_print);
    assets.release_cached_payloads().unwrap();
    assert!(released_film.upgrade().is_none());
    assert!(released_print.upgrade().is_none());
    drop(assets);
    assert_eq!(tokens, (film.asset_token(), print.asset_token()));
    assert_eq!(film.tables().source_log_exposure(), &[-1.0, 0.0, 1.0]);
    assert_eq!(
        print.sample_density_curves(1.0).unwrap().totals()[1],
        [1.5; 3]
    );
}

#[test]
fn bundled_handles_preserve_c4_tokens_and_c5_samples_after_release_and_drop() {
    let assets = Assets::new(repository().join("Resources"));
    let catalog = assets.catalog().unwrap();
    let completed = fixture("completed.json");
    for entry in catalog.films() {
        let film = assets.film(entry.key()).unwrap();
        let expected = &completed["profiles"][entry.key()];
        assert_eq!(
            film.asset_token(),
            expected["asset_token"].as_u64().unwrap(),
            "{}",
            entry.key()
        );
    }
    let prints: Vec<_> = catalog
        .prints()
        .iter()
        .map(|entry| assets.print(entry.key()).unwrap())
        .collect();
    assets.release_cached_payloads().unwrap();
    drop(assets);
    let gamma = fixture("print-gamma.json");
    let density = fixture("density.json");
    let samples =
        fs::read(repository().join("tests/profile/fixtures/density-samples.bin")).unwrap();
    let mut cases = 0;
    for captured in gamma["captured"].as_array().unwrap() {
        let id = captured["id"].as_str().unwrap();
        let case = density["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap();
        let key = case["profile"].as_str().unwrap();
        if key.is_empty() {
            continue;
        }
        let print = prints.iter().find(|p| p.info().stock() == key).unwrap();
        assert_eq!(
            print.asset_token(),
            completed["profiles"][key]["asset_token"].as_u64().unwrap()
        );
        let curves = print
            .sample_density_curves(f64::from_bits(case["gamma_bits"].as_u64().unwrap()))
            .unwrap();
        assert_eq!(
            curves.hash(),
            case["identities"]["print_hash"].as_u64().unwrap(),
            "{id}"
        );
        let start = case["sample_start"].as_u64().unwrap() as usize;
        for (i, total) in curves.totals().iter().enumerate() {
            let row = &samples[(start + i) * 49..(start + i + 1) * 49];
            assert_eq!(row[0], 1);
            let expected: [u32; 3] = std::array::from_fn(|c| {
                u32::from_le_bytes(row[1 + c * 4..5 + c * 4].try_into().unwrap())
            });
            assert_eq!(total.map(f32::to_bits), expected, "{id} row {i}");
        }
        cases += 1;
    }
    assert_eq!(cases, 48);
}
