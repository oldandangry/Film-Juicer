//! Film-Juicer static-noise characterization from unchanged C++ at 4c848d2f35ca8861b4e322a3891febfc807dc620.
//! Bundle deaddd9e79004008ff0bce40ba48de616d23456cf026c554bb9b111224744773;
//! spektrafilm 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc. FNV-1a expectations
//! cover every native payload byte. Metadata rejection cases enforce the approved
//! bundled integer/layout/mapping contract; they do not emulate C++ coercions.

#![forbid(unsafe_code)]

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use film_juicer_core::data_io::noise::{ErrorKind, load_stbn, load_wang};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tiles_path() -> PathBuf {
    repository().join("Resources/Noise/Wang/wang_tiles_256x256x16_u8.bin")
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::var_os("JUICER_TEST_ARTIFACT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| repository().join("out/validation/rust-core"));
        let path = root.join(format!(
            "noise-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn metadata(&self, root: &Value) -> PathBuf {
        self.write(&serde_json::to_vec(root).unwrap())
    }
    fn write(&self, bytes: &[u8]) -> PathBuf {
        let path = self.0.join("tiles.json");
        fs::write(&path, bytes).unwrap();
        path
    }
    fn payload(&self, length: u64) -> PathBuf {
        let path = self.0.join("noise.bin");
        File::create(&path).unwrap().set_len(length).unwrap();
        path
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove noise test directory");
    }
}

fn metadata() -> Value {
    serde_json::from_slice(&fs::read(repository().join("Resources/Noise/Wang/tiles.json")).unwrap())
        .unwrap()
}

fn identity(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

#[test]
fn bundled_stbn() {
    let noise =
        load_stbn(repository().join("Resources/Noise/stbn_scalar_512x512x256_u8.bin")).unwrap();
    assert_eq!(noise.dimensions(), [512, 512, 256]);
    assert_eq!(noise.bytes().len(), 67_108_864);
    assert_eq!(identity(noise.bytes()), 16781862639737418621);
    for (index, value) in [(0, 158), (1, 220), (33_554_432, 32), (67_108_863, 55)] {
        assert_eq!(noise.bytes()[index], value);
    }
}

#[test]
fn bundled_wang() {
    let noise = load_wang(
        tiles_path(),
        repository().join("Resources/Noise/Wang/tiles.json"),
    )
    .unwrap();
    assert_eq!(noise.dimensions(), [256, 256, 16]);
    assert_eq!(noise.colors(), 2);
    assert_eq!(noise.tiles().len(), 1_048_576);
    assert_eq!(identity(noise.tiles()), 10252691589253770981);
    assert_eq!(
        noise.lut(),
        [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    );
    for (index, value) in [(0, 71), (1, 243), (524_288, 71), (1_048_575, 193)] {
        assert_eq!(noise.tiles()[index], value);
    }
}

#[test]
fn stbn_missing_and_wrong_size() {
    let directory = Directory::new();
    let path = directory.0.join("missing.bin");
    let error = load_stbn(&path).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Missing);
    assert_eq!(error.path(), path);
    assert!(error.to_string().contains("missing.bin"));
    for actual in [0, 1, 67_108_863, 67_108_865] {
        let error = load_stbn(directory.payload(actual)).unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::Length {
                expected: 67_108_864,
                actual
            }
        );
    }
}

#[test]
fn wang_incomplete_assets_and_wrong_size() {
    let directory = Directory::new();
    let absent = directory.0.join("absent");
    let metadata = directory.metadata(&metadata());
    for (tiles, metadata, missing) in [
        (absent.clone(), metadata.clone(), absent.clone()),
        (tiles_path(), absent.clone(), absent.clone()),
        (absent.clone(), absent.clone(), absent.clone()),
    ] {
        let error = load_wang(tiles, metadata).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Missing);
        assert_eq!(error.path(), missing);
    }
    for actual in [0, 1_048_575, 1_048_577] {
        let path = directory.payload(actual);
        let error = load_wang(&path, &metadata).unwrap_err();
        assert_eq!(error.path(), path);
        assert_eq!(
            error.kind(),
            ErrorKind::Length {
                expected: 1_048_576,
                actual
            }
        );
    }
}

#[test]
fn mapping_order_and_unused_provenance() {
    let directory = Directory::new();
    let mut root = metadata();
    root["mapping"].as_array_mut().unwrap().reverse();
    root["seed"] = json!("unused");
    root["edge_band"] = Value::Null;
    root["blend_band"] = json!(false);
    root["mapping"][0]["name"] = json!([]);
    let noise = load_wang(tiles_path(), directory.metadata(&root)).unwrap();
    assert_eq!(
        noise.lut(),
        [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    );
}

#[test]
fn rejects_invalid_metadata_fields() {
    let directory = Directory::new();
    for (field, value, expected) in [
        ("/resolution", json!(true), ErrorKind::Json),
        ("/resolution", json!(256.0), ErrorKind::Json),
        ("/resolution", json!("256"), ErrorKind::Json),
        ("/resolution", json!(-1), ErrorKind::Json),
        ("/tiles", Value::Null, ErrorKind::Json),
        ("/colors", json!(1e30), ErrorKind::Json),
        ("/resolution", json!(255), ErrorKind::Dimensions),
        ("/tiles", json!(15), ErrorKind::Dimensions),
        ("/colors", json!(3), ErrorKind::Dimensions),
        ("/colors", json!(0), ErrorKind::Dimensions),
        ("/mapping/0/index", json!(-1), ErrorKind::Json),
        ("/mapping/0/index", json!(4294967296u64), ErrorKind::Json),
        ("/mapping/0/index", json!(1.75), ErrorKind::Json),
        ("/mapping/0/index", json!(16), ErrorKind::Metadata),
        ("/mapping/0/labels/L", json!(2), ErrorKind::Metadata),
        ("/mapping/0/labels/R", json!(-1), ErrorKind::Json),
        ("/mapping/0/labels/T", json!(true), ErrorKind::Json),
        ("/mapping/0/labels/B", json!(0.75), ErrorKind::Json),
        ("/mapping/0", json!({}), ErrorKind::Json),
        (
            "/mapping/0/labels",
            json!({"L":0,"R":0,"T":0}),
            ErrorKind::Json,
        ),
    ] {
        let mut root = metadata();
        *root.pointer_mut(field).unwrap() = value;
        let path = directory.metadata(&root);
        let error = load_wang(tiles_path(), &path).unwrap_err();
        assert_eq!(error.kind(), expected, "{field}");
        assert_eq!(error.path(), path);
        assert!(error.to_string().contains("tiles.json"));
    }
}

#[test]
fn requires_object_metadata_records() {
    let directory = Directory::new();
    let cases: [(&str, &[&str]); 3] = [
        ("", &["resolution", "tiles", "colors", "mapping"]),
        ("/mapping/0", &["index", "labels"]),
        ("/mapping/0/labels", &["L", "R", "T", "B"]),
    ];
    for (field, members) in cases {
        let mut root = metadata();
        let record = root.pointer_mut(field).unwrap();
        *record = Value::Array(
            members
                .iter()
                .map(|member| record[*member].clone())
                .collect(),
        );
        let path = directory.metadata(&root);
        let result = load_wang(tiles_path(), &path);
        assert!(result.is_err(), "accepted positional record at {field:?}");
        let error = result.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Json, "{field}");
        assert_eq!(error.path(), path);
    }
}

#[test]
fn requires_complete_unique_mapping() {
    let directory = Directory::new();
    let mut missing = metadata();
    missing.as_object_mut().unwrap().remove("mapping");
    let mut incomplete = metadata();
    incomplete["mapping"].as_array_mut().unwrap().pop();
    let mut extra = metadata();
    let entry = extra["mapping"][0].clone();
    extra["mapping"].as_array_mut().unwrap().push(entry);
    let mut duplicate = metadata();
    duplicate["mapping"][15]["labels"] = duplicate["mapping"][0]["labels"].clone();
    for (root, expected) in [
        (missing, ErrorKind::Json),
        (incomplete, ErrorKind::Json),
        (extra, ErrorKind::Json),
        (duplicate, ErrorKind::Metadata),
    ] {
        let error = load_wang(tiles_path(), directory.metadata(&root)).unwrap_err();
        assert_eq!(error.kind(), expected);
    }
}

#[test]
fn metadata_requires_complete_json() {
    let directory = Directory::new();
    let text = serde_json::to_string(&metadata()).unwrap();
    let noise = load_wang(
        tiles_path(),
        directory.write(format!("\u{feff}{text} \t\r\n").as_bytes()),
    )
    .unwrap();
    assert_eq!(
        noise.lut(),
        [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    );
    for invalid in [
        "{".to_owned(),
        format!("{text} trailing"),
        format!("{text} {{}}"),
    ] {
        let path = directory.write(invalid.as_bytes());
        let error = load_wang(tiles_path(), &path).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Json);
        assert_eq!(error.path(), path);
    }
}
