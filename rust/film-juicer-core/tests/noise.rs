//! Film-Juicer static-noise characterization from unchanged C++ at 4c848d2f35ca8861b4e322a3891febfc807dc620.
//! Bundle deaddd9e79004008ff0bce40ba48de616d23456cf026c554bb9b111224744773;
//! spektrafilm 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc. FNV-1a expectations
//! cover every native payload byte. Synthetic overflow expectations follow the
//! owner-approved rule: unrepresentable float mappings skip; header fields fail.

#![forbid(unsafe_code)]

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use film_juicer_core::data_io::noise::{ErrorKind, MetadataField, Wang, load_stbn, load_wang};

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
    fn wang(&self, mapping: Value) -> Wang {
        load_wang(tiles_path(), self.metadata(&metadata(mapping))).unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove noise test directory");
    }
}

fn metadata(mapping: Value) -> Value {
    json!({"resolution":256,"tiles":16,"colors":2,"mapping":mapping})
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
    let metadata = directory.metadata(&metadata(json!([])));
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
fn mapping_defaults_and_edge_order() {
    let directory = Directory::new();
    let noise = directory.wang(json!([
        {"index":5,"labels":{}},
        {"index":1,"labels":{"B":1}},
        {"index":2,"labels":{"T":1}},
        {"index":3,"labels":{"R":1}},
        {"index":4,"labels":{"L":1}}
    ]));
    assert_eq!(
        noise.lut(),
        [5, 1, 2, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(directory.wang(json!([])).lut(), [0; 16]);
}

#[test]
fn incomplete_and_out_of_range_mappings() {
    let directory = Directory::new();
    let noise = directory.wang(json!([
        {}, {"index":4}, {"labels":{"L":1}}, null, 3, [], false,
        {"index":-1,"labels":{}}, {"index":16,"labels":{}},
        {"index":2,"labels":{"L":2}}, {"index":3,"labels":{"R":-1}},
        {"index":4,"labels":{"T":2}}, {"index":5,"labels":{"B":-1}}
    ]));
    assert_eq!(noise.lut(), [0; 16]);
}

#[test]
fn duplicate_mappings() {
    let directory = Directory::new();
    let noise = directory.wang(json!([
        {"index":4,"labels":{}}, {"index":7,"labels":{}},
        {"index":16,"labels":{}}, {"index":5,"labels":{"L":2}}
    ]));
    assert_eq!(
        noise.lut(),
        [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
}

#[test]
fn native_numeric_conversions() {
    let directory = Directory::new();
    let noise = directory.wang(json!([
        {"index":4294967301u64,"labels":{"L":4294967296u64}},
        {"index":3.75,"labels":{"L":-0.75,"R":1.75}},
        {"index":true,"labels":{"L":true,"R":false}}
    ]));
    assert_eq!(
        noise.lut(),
        [5, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    );
    let mut root = metadata(json!([]));
    root["resolution"] = json!(256.9);
    root["tiles"] = json!(16.75);
    root["colors"] = json!(true);
    let noise = load_wang(tiles_path(), directory.metadata(&root)).unwrap();
    assert_eq!(noise.dimensions(), [256, 256, 16]);
    assert_eq!(noise.colors(), 1);
    assert_eq!(noise.lut(), [0]);
}

#[test]
fn unrepresentable_mappings_skip() {
    let directory = Directory::new();
    for number in [1e30, -1e30, 2147483648.0, -2147483649.0] {
        for field in ["index", "L", "R", "T", "B"] {
            let mut entry = json!({"index":5,"labels":{}});
            if field == "index" {
                entry[field] = json!(number);
            } else {
                entry["labels"][field] = json!(number);
            }
            let noise = directory.wang(json!([{"index":7,"labels":{}},entry]));
            assert_eq!(
                noise.lut(),
                [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "{field}={number}"
            );
        }
    }
}

#[test]
fn unrepresentable_dimensions_fail() {
    let directory = Directory::new();
    for (name, field) in [
        ("resolution", MetadataField::Resolution),
        ("tiles", MetadataField::Tiles),
        ("colors", MetadataField::Colors),
    ] {
        for value in [1e30, -1e30] {
            let mut root = metadata(json!([]));
            root[name] = json!(value);
            let error = load_wang(tiles_path(), directory.metadata(&root)).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Metadata(field));
        }
    }
}

#[test]
fn wrong_dimensions_and_lut_overflow() {
    let directory = Directory::new();
    for (field, value, expected) in [
        ("resolution", 0, ErrorKind::Dimensions),
        ("resolution", 255, ErrorKind::Dimensions),
        ("tiles", 15, ErrorKind::Dimensions),
        ("colors", 0, ErrorKind::Dimensions),
        ("colors", 65_536, ErrorKind::Size),
        ("colors", 60_000, ErrorKind::Capacity),
    ] {
        let mut root = metadata(json!([]));
        root[field] = json!(value);
        let error = load_wang(tiles_path(), directory.metadata(&root)).unwrap_err();
        assert_eq!(error.kind(), expected);
        assert!(error.path().ends_with("tiles.json"));
    }
    let mut root = metadata(json!([]));
    root["colors"] = json!(3);
    let noise = load_wang(tiles_path(), directory.metadata(&root)).unwrap();
    assert_eq!(noise.colors(), 3);
    assert_eq!(noise.lut(), [0; 81]);
}

#[test]
fn malformed_metadata() {
    let directory = Directory::new();
    for bytes in [b"".as_slice(), b"{", b"{\"resolution\":1e999}"] {
        assert_eq!(
            load_wang(tiles_path(), directory.write(bytes))
                .unwrap_err()
                .kind(),
            ErrorKind::Json
        );
    }
    for (field, expected) in [
        ("resolution", MetadataField::Resolution),
        ("tiles", MetadataField::Tiles),
        ("colors", MetadataField::Colors),
        ("mapping", MetadataField::Mapping),
    ] {
        let mut root = metadata(json!([]));
        root.as_object_mut().unwrap().remove(field);
        assert_eq!(
            load_wang(tiles_path(), directory.metadata(&root))
                .unwrap_err()
                .kind(),
            ErrorKind::Metadata(expected)
        );
    }
    for (root, expected) in [
        (json!([]), MetadataField::Root),
        (
            json!({"resolution":"256","tiles":16,"colors":2,"mapping":[]}),
            MetadataField::Resolution,
        ),
        (
            json!({"resolution":256,"tiles":16,"colors":null,"mapping":[]}),
            MetadataField::Colors,
        ),
        (metadata(json!({})), MetadataField::Mapping),
        (
            metadata(json!([{"index":1,"labels":null}])),
            MetadataField::Labels,
        ),
        (
            metadata(json!([{"index":1,"labels":[]}])),
            MetadataField::Labels,
        ),
        (
            metadata(json!([{"index":null,"labels":{}}])),
            MetadataField::Index,
        ),
        (
            metadata(json!([{"index":1,"labels":{"T":null}}])),
            MetadataField::Top,
        ),
        (
            metadata(json!([{"index":1,"labels":{"R":"1"}}])),
            MetadataField::Right,
        ),
        (
            metadata(json!([{"index":16,"labels":{"T":null}}])),
            MetadataField::Top,
        ),
    ] {
        assert_eq!(
            load_wang(tiles_path(), directory.metadata(&root))
                .unwrap_err()
                .kind(),
            ErrorKind::Metadata(expected)
        );
    }
}

#[test]
fn metadata_bom_and_trailing_input() {
    let directory = Directory::new();
    let bytes =
        b"\xef\xbb\xbf{\"resolution\":256,\"tiles\":16,\"colors\":2,\"mapping\":[]} trailing";
    let noise = load_wang(tiles_path(), directory.write(bytes)).unwrap();
    assert_eq!(noise.lut(), [0; 16]);
}
