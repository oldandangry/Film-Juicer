//! Asset ownership contracts; numerical identities use the existing frozen C4/C5 captures.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use film_juicer_core::assets::{AssetError, Assets, CsvSource};
use film_juicer_core::data_io::calibration::{ErrorKind as CalibrationErrorKind, Field};
use film_juicer_core::data_io::noise::ErrorKind as NoiseErrorKind;
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

const CSV_SOURCES: &[(CsvSource, &str, usize, u64)] = &[
    (
        CsvSource::D65,
        "illuminants/D65.csv",
        81,
        0xadbdb3d2c250ef10,
    ),
    (
        CsvSource::D55,
        "illuminants/D55.csv",
        81,
        0xbc4634b46165cfff,
    ),
    (
        CsvSource::D50,
        "illuminants/D50.csv",
        81,
        0xf70ca3af96e9a5e8,
    ),
    (CsvSource::T, "illuminants/T.csv", 81, 0x2013ac33b7e2aaec),
    (
        CsvSource::K75p,
        "illuminants/K75P.csv",
        81,
        0x1adba11b770daaa9,
    ),
    (
        CsvSource::Kg3,
        "filters/heat_absorbing/schott/KG3.csv",
        146,
        0xe040af29a7935a51,
    ),
    (
        CsvSource::Canon24F28Is,
        "filters/lens_transmission/canon/canon_24_f28_is.csv",
        107,
        0x74308f00cb53f384,
    ),
];

fn write_resource(directory: &Directory, name: &str, bytes: impl AsRef<[u8]>) {
    let path = directory.0.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn write_noise(directory: &Directory) {
    // Fixed complete zero-filled inputs suffice for owner/release contracts.
    // Independent bundled byte expectations remain in the reader and owner tests.
    for (name, length) in [
        ("Noise/stbn_scalar_512x512x256_u8.bin", 67_108_864),
        ("Noise/Wang/wang_tiles_256x256x16_u8.bin", 1_048_576),
    ] {
        let path = directory.0.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::File::create(path).unwrap().set_len(length).unwrap();
    }
    write_resource(
        directory,
        "Noise/Wang/tiles.json",
        fs::read(repository().join("Resources/Noise/Wang/tiles.json")).unwrap(),
    );
}

fn byte_fingerprint(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

#[test]
fn source_owners_preserve_independent_bundled_csv_bits() {
    // Reuse the pre-translation C++ captures documented in tests/data_io.rs.
    let assets = Assets::new(repository().join("Resources"));
    let cmf = assets.cmf().unwrap();
    assert_eq!(cmf.rows().len(), 81);
    assert_eq!(
        byte_fingerprint(cmf.rows().iter().flatten().flat_map(|v| v.to_le_bytes())),
        0xc0ea4ea30d5fd861
    );
    assert!(Arc::ptr_eq(&cmf, &assets.cmf().unwrap()));
    for &(source, path, count, hash) in CSV_SOURCES {
        let rows = assets.csv_source(source).unwrap();
        assert_eq!(rows.rows().len(), count, "{path}");
        assert_eq!(
            byte_fingerprint(rows.rows().iter().flatten().flat_map(|v| v.to_le_bytes())),
            hash,
            "{path}"
        );
        let reused = assets.csv_source(source).unwrap();
        assert!(Arc::ptr_eq(&rows, &reused));
        assert_eq!(rows.rows().as_ptr(), reused.rows().as_ptr());
    }
}

#[test]
fn closed_csv_sources_retry_failures_and_preserve_raw_rows() {
    let directory = Directory::new();
    let assets = Assets::new(directory.0.clone());
    for &(source, path, _, _) in CSV_SOURCES {
        let error = assets.csv_source(source).unwrap_err();
        let AssetError::CsvSource {
            source: selected,
            error,
        } = error
        else {
            panic!("CSV error expected")
        };
        assert_eq!(selected, source);
        assert_eq!(error.path, directory.0.join(path));
        write_resource(&directory, path, "2,-0\nbad\n1,1e-50\n1,-1e-50\n-4,3\n");
        let decoded = assets.csv_source(source).unwrap();
        assert_eq!(
            decoded
                .rows()
                .iter()
                .map(|row| row.map(f32::to_bits))
                .collect::<Vec<_>>(),
            [
                [2.0f32.to_bits(), 0x8000_0000],
                [1.0f32.to_bits(), 0],
                [1.0f32.to_bits(), 0x8000_0000],
                [(-4.0f32).to_bits(), 3.0f32.to_bits()]
            ]
        );
        fs::remove_file(directory.0.join(path)).unwrap();
        assert!(Arc::ptr_eq(&decoded, &assets.csv_source(source).unwrap()));
        assets.release_cached_payloads().unwrap();
        assert!(matches!(
            assets.csv_source(source),
            Err(AssetError::CsvSource { .. })
        ));
        assert_eq!(decoded.rows().len(), 4);
    }
}

#[test]
fn decoded_sources_do_not_acquire_axis_or_scientific_admission() {
    let directory = Directory::new();
    write_resource(
        &directory,
        "cie1931_2deg.csv",
        "2,-0,3,4\n1,5,6,7\n1,8,9,10\n",
    );
    write_resource(&directory, "illuminants/D65.csv", "header only\n");
    let assets = Assets::new(directory.0.clone());
    let cmf = assets.cmf().unwrap();
    assert_eq!(cmf.rows().len(), 3);
    assert_eq!(
        cmf.rows()[0].map(f32::to_bits),
        [
            2.0f32.to_bits(),
            0x8000_0000,
            3.0f32.to_bits(),
            4.0f32.to_bits()
        ]
    );
    assert_eq!(cmf.rows()[1], [1.0, 5.0, 6.0, 7.0]);
    assert_eq!(cmf.rows()[2], [1.0, 8.0, 9.0, 10.0]);
    assert!(assets.csv_source(CsvSource::D65).unwrap().rows().is_empty());
    let empty = Directory::new();
    write_resource(&empty, "cie1931_2deg.csv", "no decoded rows\n");
    assert!(
        Assets::new(empty.0.clone())
            .cmf()
            .unwrap()
            .rows()
            .is_empty()
    );
}

#[test]
fn cmf_success_and_error_are_retained_across_release_and_owner_drop() {
    let directory = Directory::new();
    let assets = Assets::new(directory.0.clone());
    let Err(AssetError::Cmf(error)) = assets.cmf() else {
        panic!("missing CMF expected")
    };
    write_resource(&directory, "cie1931_2deg.csv", "1,2,3,4\n");
    assets.release_cached_payloads().unwrap();
    let Err(AssetError::Cmf(reused)) = assets.cmf() else {
        panic!("CMF outcome remains fixed")
    };
    assert!(Arc::ptr_eq(&error, &reused));
    let success_owner = Assets::new(directory.0.clone());
    let cmf = success_owner.cmf().unwrap();
    fs::remove_file(directory.0.join("cie1931_2deg.csv")).unwrap();
    success_owner.release_cached_payloads().unwrap();
    assert!(Arc::ptr_eq(&cmf, &success_owner.cmf().unwrap()));
    drop(assets);
    drop(success_owner);
    assert_eq!(cmf.rows(), &[[1.0, 2.0, 3.0, 4.0]]);
    assert_eq!(error.path, directory.0.join("cie1931_2deg.csv"));
}

#[test]
fn calibration_owner_preserves_selected_lookup_and_cmy_cc_order() {
    let directory = Directory::new();
    let path = "filters/neutral_print_filters.json";
    write_resource(&directory, path, br#"{"paper":{"D65":{"film":[-0,2.25,1e40],"bad":[1,2]}},"bad_print":4,"bad_illuminant":{"D65":false}}"#);
    let assets = Assets::new(directory.0.clone());
    let calibration = assets.neutral_calibration().unwrap();
    let cmy = calibration.lookup("paper", "D65", "film").unwrap().unwrap();
    assert_eq!(
        cmy.map(f32::to_bits),
        [0x8000_0000, 2.25f32.to_bits(), f32::INFINITY.to_bits()]
    );
    for selection in [
        ("absent", "D65", "film"),
        ("paper", "absent", "film"),
        ("paper", "D65", "absent"),
    ] {
        assert_eq!(
            calibration
                .lookup(selection.0, selection.1, selection.2)
                .unwrap(),
            None
        );
    }
    for (selection, field) in [
        (("bad_print", "D65", "film"), Field::PrintProfile),
        (("bad_illuminant", "D65", "film"), Field::PrintIlluminant),
        (("paper", "D65", "bad"), Field::CmyCc),
    ] {
        let error = calibration
            .lookup(selection.0, selection.1, selection.2)
            .unwrap_err();
        assert_eq!(error.kind(), CalibrationErrorKind::Malformed(field));
        assert_eq!(
            error.selection(),
            Some([selection.0, selection.1, selection.2])
        );
        assert_eq!(error.path(), directory.0.join(path));
        assert!(calibration.lookup("paper", "D65", "film").is_ok());
    }
    fs::remove_file(directory.0.join(path)).unwrap();
    assert!(Arc::ptr_eq(
        &calibration,
        &assets.neutral_calibration().unwrap()
    ));
}

#[test]
fn calibration_load_errors_remain_snapshots_until_release() {
    for case in ["missing", "read", "root"] {
        let directory = Directory::new();
        let path = directory.0.join("filters/neutral_print_filters.json");
        match case {
            "read" => fs::create_dir_all(&path).unwrap(),
            "root" => write_resource(&directory, "filters/neutral_print_filters.json", "[]"),
            _ => (),
        }
        let assets = Assets::new(directory.0.clone());
        let Err(AssetError::Calibration(error)) = assets.neutral_calibration() else {
            panic!("load error expected")
        };
        match case {
            "missing" => assert_eq!(error.kind(), CalibrationErrorKind::MissingFile),
            "read" => assert!(matches!(error.kind(), CalibrationErrorKind::Read(_))),
            _ => assert_eq!(error.kind(), CalibrationErrorKind::Malformed(Field::Root)),
        }
        assert_eq!(error.selection(), None);
        assert_eq!(error.path(), path);
        if path.is_dir() {
            fs::remove_dir(&path).unwrap();
        }
        write_resource(
            &directory,
            "filters/neutral_print_filters.json",
            r#"{"p":{"i":{"f":[1,2,3]}}}"#,
        );
        let Err(AssetError::Calibration(reused)) = assets.neutral_calibration() else {
            panic!("error is sticky")
        };
        assert!(Arc::ptr_eq(&error, &reused));
        assets.release_cached_payloads().unwrap();
        let loaded = assets.neutral_calibration().unwrap();
        drop(assets);
        assert_eq!(loaded.lookup("p", "i", "f").unwrap(), Some([1.0, 2.0, 3.0]));
        assert_eq!(error.path(), path);
    }
}

#[test]
fn complete_noise_owner_preserves_independent_bundled_bytes() {
    let assets = Assets::new(repository().join("Resources"));
    let bundle = assets.noise().unwrap();
    assert_eq!(bundle.stbn().dimensions(), [512, 512, 256]);
    assert_eq!(bundle.wang().dimensions(), [256, 256, 16]);
    assert_eq!(bundle.wang().colors(), 2);
    assert_eq!(
        byte_fingerprint(bundle.stbn().bytes().iter().copied()),
        16781862639737418621
    );
    assert_eq!(
        byte_fingerprint(bundle.wang().tiles().iter().copied()),
        10252691589253770981
    );
    assert_eq!(
        bundle.wang().lut(),
        [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    );
    let reused = assets.noise().unwrap();
    assert!(Arc::ptr_eq(&bundle, &reused));
    assert_eq!(
        bundle.stbn().bytes().as_ptr(),
        reused.stbn().bytes().as_ptr()
    );
    assert_eq!(
        bundle.wang().tiles().as_ptr(),
        reused.wang().tiles().as_ptr()
    );
}

#[test]
fn ordinary_noise_errors_are_cached_until_release() {
    for case in [
        "missing_stbn",
        "malformed_stbn",
        "missing_wang",
        "malformed_wang",
        "malformed_json",
        "malformed_metadata",
    ] {
        let directory = Directory::new();
        write_noise(&directory);
        let stbn = directory.0.join("Noise/stbn_scalar_512x512x256_u8.bin");
        let wang = directory.0.join("Noise/Wang/wang_tiles_256x256x16_u8.bin");
        let metadata = directory.0.join("Noise/Wang/tiles.json");
        let path = match case {
            "missing_stbn" => {
                fs::remove_file(&stbn).unwrap();
                &stbn
            }
            "malformed_stbn" => {
                fs::write(&stbn, [0u8]).unwrap();
                &stbn
            }
            "missing_wang" => {
                fs::remove_file(&wang).unwrap();
                &wang
            }
            "malformed_wang" => {
                fs::write(&wang, [0u8]).unwrap();
                &wang
            }
            "malformed_json" => {
                fs::write(&metadata, "{").unwrap();
                &metadata
            }
            _ => {
                fs::write(&metadata, "{}").unwrap();
                &metadata
            }
        };
        let assets = Assets::new(directory.0.clone());
        let Err(AssetError::Noise(error)) = assets.noise() else {
            panic!("complete bundle cannot be published")
        };
        assert_eq!(error.path(), path);
        match case {
            "missing_stbn" | "missing_wang" => assert_eq!(error.kind(), NoiseErrorKind::Missing),
            "malformed_stbn" | "malformed_wang" => {
                assert!(matches!(error.kind(), NoiseErrorKind::Length { .. }))
            }
            _ => assert_eq!(error.kind(), NoiseErrorKind::Json),
        }
        write_noise(&directory);
        let Err(AssetError::Noise(reused)) = assets.noise() else {
            panic!("ordinary error is sticky")
        };
        assert!(Arc::ptr_eq(&error, &reused));
        assets.release_cached_payloads().unwrap();
        let bundle = assets.noise().unwrap();
        drop(assets);
        assert_eq!(bundle.stbn().bytes().len(), 67_108_864);
        assert_eq!(bundle.wang().tiles().len(), 1_048_576);
        assert_eq!(error.path(), path);
    }
}

#[test]
fn release_detaches_new_families_even_with_absent_or_failed_catalog() {
    for failed_catalog in [false, true] {
        let directory = Directory::new();
        write_noise(&directory);
        write_resource(
            &directory,
            "filters/neutral_print_filters.json",
            r#"{"p":{"i":{"f":[1,2,3]}}}"#,
        );
        write_resource(&directory, "cie1931_2deg.csv", "1,2,3,4\n");
        for &(_, path, _, _) in CSV_SOURCES {
            write_resource(&directory, path, "1,2\n");
        }
        let assets = Assets::new(directory.0.clone());
        if failed_catalog {
            assert!(assets.catalog().is_err());
        }
        let cmf = assets.cmf().unwrap();
        let noise = assets.noise().unwrap();
        let calibration = assets.neutral_calibration().unwrap();
        let sources = CSV_SOURCES
            .iter()
            .map(|(source, _, _, _)| assets.csv_source(*source).unwrap())
            .collect::<Vec<_>>();
        let reclaimed = Arc::downgrade(&assets.csv_source(CsvSource::D65).unwrap());
        for &(_, path, _, _) in CSV_SOURCES {
            write_resource(&directory, path, "3,4\n");
        }
        write_resource(&directory, "cie1931_2deg.csv", "9,8,7,6\n");
        write_resource(
            &directory,
            "filters/neutral_print_filters.json",
            r#"{"p":{"i":{"f":[4,5,6]}}}"#,
        );
        assets.release_cached_payloads().unwrap();
        assert!(Arc::ptr_eq(&cmf, &assets.cmf().unwrap()));
        let new_calibration = assets.neutral_calibration().unwrap();
        let new_noise = assets.noise().unwrap();
        assert!(!Arc::ptr_eq(&calibration, &new_calibration));
        assert!(!Arc::ptr_eq(&noise, &new_noise));
        for (&(source, _, _, _), old) in CSV_SOURCES.iter().zip(&sources) {
            let new = assets.csv_source(source).unwrap();
            assert!(!Arc::ptr_eq(old, &new));
            assert_eq!(new.rows(), &[[3.0, 4.0]]);
        }
        drop(assets);
        assert_eq!(cmf.rows(), &[[1.0, 2.0, 3.0, 4.0]]);
        assert_eq!(
            calibration.lookup("p", "i", "f").unwrap(),
            Some([1.0, 2.0, 3.0])
        );
        assert_eq!(
            new_calibration.lookup("p", "i", "f").unwrap(),
            Some([4.0, 5.0, 6.0])
        );
        assert_eq!(noise.stbn().bytes()[0], 0);
        assert_eq!(new_noise.wang().tiles()[0], 0);
        for old in &sources {
            assert_eq!(old.rows(), &[[1.0, 2.0]]);
        }
        drop(sources);
        assert!(reclaimed.upgrade().is_none());
    }
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
        film.tables().authored_log_exposure().as_ptr(),
        reused_film.tables().authored_log_exposure().as_ptr()
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
    assert_eq!(film.tables().authored_log_exposure(), &[-1.0, 0.0, 1.0]);
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

const HANATOS: &str = "irradiance_xy_tc.npy";
const ARCTIC: &str = "arctic2026beta04_reflectance_xy_tc.npy";
const MALLETT: &str = "mallett2019_basis.npy";
const LUT_COUNT: usize = 192 * 192 * 81;

fn lut_path(directory: &Directory, name: &str) -> PathBuf {
    directory.0.join("luts/spectral_upsampling").join(name)
}

fn copy_reconstruction(directory: &Directory, name: &str) {
    let target = lut_path(directory, name);
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::copy(
        repository()
            .join("Resources/luts/spectral_upsampling")
            .join(name),
        target,
    )
    .unwrap();
}

fn npy(directory: &Directory, name: &str, header: &str, pattern: &[u8], repeat: usize) {
    let path = lut_path(directory, name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&pattern.repeat(repeat));
    fs::write(path, bytes).unwrap();
}

fn reconstruction_error(error: AssetError) -> Arc<film_juicer_core::data_io::ReadError> {
    match error {
        AssetError::Reconstruction(source) => source,
        other => panic!("expected reconstruction failure, got {other:?}"),
    }
}

#[test]
fn reconstruction_selection_is_lazy_and_independent_of_profiles_and_other_families() {
    let directory = Directory::new();
    let assets = Assets::new(directory.0.clone());
    assets.release_cached_payloads().unwrap();
    // Catalog failure and unavailable unrelated spectra do not block Mallett.
    assert!(matches!(assets.catalog(), Err(AssetError::Catalog(_))));
    copy_reconstruction(&directory, MALLETT);
    assert_eq!(assets.mallett().unwrap()[0][0].to_bits(), 0x3ea7a880);
    // Loading Mallett must not have snapped missing Hanatos/Arctic outcomes.
    copy_reconstruction(&directory, HANATOS);
    assert_eq!(assets.hanatos().unwrap().asset_hash(), 0x81ebefd4e4cc9926);
    copy_reconstruction(&directory, ARCTIC);
    assert_eq!(assets.arctic().unwrap().asset_hash(), 0x9262ffb765e3289e);

    let profiles = Directory::new();
    profiles.defaults();
    let assets = Assets::new(profiles.0.clone());
    assert!(assets.hanatos().is_err());
    assert!(assets.film("kodak_portra_400").is_ok());
    assert!(assets.print("kodak_portra_endura").is_ok());
    copy_reconstruction(&profiles, ARCTIC);
    assert!(assets.arctic().is_ok());
    copy_reconstruction(&profiles, MALLETT);
    assert!(assets.mallett().is_ok());
}

#[test]
fn reconstruction_handles_and_buffers_survive_profile_release_and_owner_drop() {
    let directory = Directory::new();
    directory.defaults();
    for name in [HANATOS, ARCTIC, MALLETT] {
        copy_reconstruction(&directory, name);
    }
    let assets = Assets::new(directory.0.clone());
    let film = assets.film("kodak_portra_400").unwrap();
    let hanatos = assets.hanatos().unwrap();
    let arctic = assets.arctic().unwrap();
    let mallett = assets.mallett().unwrap();
    let pointers = (
        hanatos.samples().as_ptr(),
        arctic.samples().as_ptr(),
        mallett.as_ptr(),
    );
    fs::remove_dir_all(directory.0.join("luts")).unwrap();
    assets.release_cached_payloads().unwrap();
    assert!(!Arc::ptr_eq(
        &film,
        &assets.film("kodak_portra_400").unwrap()
    ));
    assert!(Arc::ptr_eq(&hanatos, &assets.hanatos().unwrap()));
    assert!(Arc::ptr_eq(&arctic, &assets.arctic().unwrap()));
    assert!(Arc::ptr_eq(&mallett, &assets.mallett().unwrap()));
    assert_eq!(
        pointers,
        (
            hanatos.samples().as_ptr(),
            arctic.samples().as_ptr(),
            mallett.as_ptr()
        )
    );
    drop(assets);
    assert_eq!(hanatos.samples().len(), LUT_COUNT);
    assert_eq!(hanatos.samples()[0].to_bits(), 0x40bb0000);
    assert_eq!(hanatos.asset_hash(), 0x81ebefd4e4cc9926);
    assert_eq!(arctic.samples().len(), LUT_COUNT);
    assert_eq!(arctic.asset_hash(), 0x9262ffb765e3289e);
    assert_eq!(mallett[80][2].to_bits(), 0x3eaa8d89);
}

#[test]
fn reconstruction_failures_remain_fixed_across_release_and_file_repair() {
    use film_juicer_core::data_io::ReadErrorKind;
    let directory = Directory::new();
    let assets = Assets::new(directory.0.clone());
    let hanatos = reconstruction_error(assets.hanatos().unwrap_err());
    let arctic = reconstruction_error(assets.arctic().unwrap_err());
    let mallett = reconstruction_error(assets.mallett().unwrap_err());
    for (name, error) in [(HANATOS, &hanatos), (ARCTIC, &arctic), (MALLETT, &mallett)] {
        assert_eq!(error.path, lut_path(&directory, name));
        assert_eq!(
            error.kind,
            ReadErrorKind::Open(std::io::ErrorKind::NotFound)
        );
        copy_reconstruction(&directory, name);
    }
    assets.release_cached_payloads().unwrap();
    assert!(Arc::ptr_eq(
        &hanatos,
        &reconstruction_error(assets.hanatos().unwrap_err())
    ));
    assert!(Arc::ptr_eq(
        &arctic,
        &reconstruction_error(assets.arctic().unwrap_err())
    ));
    assert!(Arc::ptr_eq(
        &mallett,
        &reconstruction_error(assets.mallett().unwrap_err())
    ));
    drop(assets);
    assert_eq!(hanatos.path, lut_path(&directory, HANATOS));
    let fresh = Assets::new(directory.0.clone());
    assert!(fresh.hanatos().is_ok());
    assert!(fresh.arctic().is_ok());
    assert!(fresh.mallett().is_ok());
}

#[test]
fn supported_npy_headers_and_widths_share_identities_and_preserve_sample_bits() {
    let vectors = fixture("resource-hash.json");
    let vector = vectors
        .as_array()
        .unwrap()
        .iter()
        .find(|vector| vector["id"] == "lut-specials")
        .unwrap();
    let expected = vector["expected"].as_u64().unwrap();
    let bits: Vec<u32> = vector["bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|bits| u32::try_from(bits.as_u64().unwrap()).unwrap())
        .collect();
    let halves: [u16; 8] = [0, 0x8000, 0x3c00, 0xc000, 0x7c00, 0xfc00, 0x7e23, 0xfe35];
    for width in [2, 4, 8] {
        let pattern: Vec<u8> = match width {
            2 => halves.iter().flat_map(|bits| bits.to_le_bytes()).collect(),
            4 => bits.iter().flat_map(|bits| bits.to_le_bytes()).collect(),
            8 => bits
                .iter()
                .flat_map(|&bits| f64::from(f32::from_bits(bits)).to_le_bytes())
                .collect(),
            _ => unreachable!(),
        };
        for header in [
            format!("{{'descr':'<f{width}','fortran_order':False,'shape':(192,192,81)}}\n"),
            format!(
                " {{\"shape\":(192,192,81,),\"descr\":\"<f{width}\",\"fortran_order\":False,}}   \n"
            ),
        ] {
            let directory = Directory::new();
            npy(
                &directory,
                HANATOS,
                &header,
                &pattern,
                LUT_COUNT / bits.len(),
            );
            npy(
                &directory,
                ARCTIC,
                &header,
                &pattern,
                LUT_COUNT / bits.len(),
            );
            let assets = Assets::new(directory.0.clone());
            for lut in [assets.hanatos().unwrap(), assets.arctic().unwrap()] {
                assert_eq!(lut.asset_hash(), expected, "width {width}: {header}");
                assert_eq!(lut.samples().len(), LUT_COUNT);
                for (i, sample) in lut.samples().iter().enumerate() {
                    assert_eq!(sample.to_bits(), bits[i % bits.len()], "sample {i}");
                }
            }
        }
    }
}

#[test]
fn resource_equivalence_and_changed_content_use_actual_lut_owners() {
    use std::io::{Seek, SeekFrom, Write};
    let directory = Directory::new();
    let header = "{'descr':'<f4','fortran_order':False,'shape':(192,192,81)}\n";
    let bits: [u32; 8] = [
        0, 0x80000000, 0x3f800000, 0xc0000000, 0x7f800000, 0xff800000, 0x7f800001, 0xffc54321,
    ];
    let equivalent: [u32; 8] = [
        0x80000000, 0, 0x3f800000, 0xc0000000, 0x7f800000, 0xff800000, 0xffc54321, 0x7fc00001,
    ];
    let pattern: Vec<u8> = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
    let other: Vec<u8> = equivalent
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    npy(&directory, HANATOS, header, &pattern, LUT_COUNT / 8);
    npy(&directory, ARCTIC, header, &other, LUT_COUNT / 8);
    let assets = Assets::new(directory.0.clone());
    let hanatos = assets.hanatos().unwrap();
    let arctic = assets.arctic().unwrap();
    assert_eq!(hanatos.asset_hash(), arctic.asset_hash());
    assert_eq!(hanatos.samples()[6].to_bits(), bits[6]);
    assert_eq!(arctic.samples()[6].to_bits(), equivalent[6]);
    // The held snapshot stays fixed; a fresh owner sees changed contributing bits.
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(lut_path(&directory, ARCTIC))
        .unwrap();
    file.seek(SeekFrom::Start(
        u64::try_from(10 + header.len() + 2 * 4).unwrap(),
    ))
    .unwrap();
    file.write_all(&0x3f800001u32.to_le_bytes()).unwrap();
    drop(file);
    let fresh = Assets::new(directory.0.clone()).arctic().unwrap();
    assert_ne!(fresh.asset_hash(), arctic.asset_hash());
    assert_eq!(fresh.samples()[2].to_bits(), 0x3f800001);
    assert!(Arc::ptr_eq(&arctic, &assets.arctic().unwrap()));
}

#[test]
fn truncated_reconstruction_outcomes_do_not_publish_partial_assets_or_retry_after_repair() {
    use film_juicer_core::data_io::{ReadErrorKind, ReadPart};
    let directory = Directory::new();
    for name in [HANATOS, ARCTIC, MALLETT] {
        let path = lut_path(&directory, name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"\x93NUMPY\x01\x00").unwrap();
    }
    let assets = Assets::new(directory.0.clone());
    let hanatos = reconstruction_error(assets.hanatos().unwrap_err());
    let arctic = reconstruction_error(assets.arctic().unwrap_err());
    let mallett = reconstruction_error(assets.mallett().unwrap_err());
    for (name, error) in [(HANATOS, &hanatos), (ARCTIC, &arctic), (MALLETT, &mallett)] {
        assert_eq!(error.kind, ReadErrorKind::ShortRead(ReadPart::Header));
        copy_reconstruction(&directory, name);
    }
    assets.release_cached_payloads().unwrap();
    assert!(Arc::ptr_eq(
        &hanatos,
        &reconstruction_error(assets.hanatos().unwrap_err())
    ));
    assert!(Arc::ptr_eq(
        &arctic,
        &reconstruction_error(assets.arctic().unwrap_err())
    ));
    assert!(Arc::ptr_eq(
        &mallett,
        &reconstruction_error(assets.mallett().unwrap_err())
    ));
}
