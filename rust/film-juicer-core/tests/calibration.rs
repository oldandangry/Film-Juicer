//! Neutral calibration characterization from unchanged C++ at
//! 4c848d2f35ca8861b4e322a3891febfc807dc620. The FNV-1a expectation covers all
//! 160 bundled CMY triplets in sorted print/illuminant/film order, LE f32 bits.
//! Bundle deaddd9e79004008ff0bce40ba48de616d23456cf026c554bb9b111224744773;
//! spektrafilm 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc. Bare -0 follows the
//! previously approved direct-Serde input contract, not native integer-zero bits.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use film_juicer_core::data_io::calibration::{ErrorKind, Field, load_neutral_calibration};

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
            "calibration-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn write(&self, bytes: &[u8]) -> PathBuf {
        let path = self.0.join("neutral.json");
        fs::write(&path, bytes).unwrap();
        path
    }
    fn json(&self, value: Value) -> PathBuf {
        self.write(&serde_json::to_vec(&value).unwrap())
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove calibration test directory");
    }
}

const PRINTS: &[&str] = &[
    "fujifilm_crystal_archive_typeii",
    "kodak_2383",
    "kodak_2393",
    "kodak_ektacolor_edge",
    "kodak_endura_premier",
    "kodak_portra_endura",
    "kodak_supra_endura",
    "kodak_ultra_endura",
];
const FILMS: &[&str] = &[
    "fujifilm_c200",
    "fujifilm_pro_400h",
    "fujifilm_provia_100f",
    "fujifilm_velvia_100",
    "fujifilm_xtra_400",
    "kodak_ektachrome_100",
    "kodak_ektar_100",
    "kodak_gold_200",
    "kodak_kodachrome_64",
    "kodak_portra_160",
    "kodak_portra_400",
    "kodak_portra_800",
    "kodak_portra_800_push1",
    "kodak_portra_800_push2",
    "kodak_ultramax_400",
    "kodak_verita_200d",
    "kodak_vision3_200t",
    "kodak_vision3_250d",
    "kodak_vision3_500t",
    "kodak_vision3_50d",
];

#[test]
fn bundled_calibration() {
    let calibration =
        load_neutral_calibration(repository().join("Resources/filters/neutral_print_filters.json"))
            .unwrap();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for print in PRINTS {
        for film in FILMS {
            let cmy = calibration.lookup(print, "TH-KG3", film).unwrap().unwrap();
            for byte in cmy.into_iter().flat_map(f32::to_le_bytes) {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    assert_eq!(PRINTS.len() * FILMS.len(), 160);
    assert_eq!(hash, 0x3ef4_3845_b118_f2f9);
    let cmy = calibration
        .lookup("kodak_portra_endura", "TH-KG3", "kodak_portra_400")
        .unwrap()
        .unwrap();
    assert_eq!(cmy.map(f32::to_bits), [0, 1112497324, 1112343655]);
}

#[test]
fn missing_file_and_entry() {
    let directory = Directory::new();
    let path = directory.0.join("missing.json");
    let error = load_neutral_calibration(&path).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MissingFile);
    assert_eq!(error.path(), path);
    assert_eq!(error.selection(), None);
    assert!(error.to_string().contains("missing.json"));
    let calibration =
        load_neutral_calibration(directory.json(json!({"paper":{"light":{"film":[1,2,3]}}})))
            .unwrap();
    for keys in [
        ["missing", "light", "film"],
        ["paper", "missing", "film"],
        ["paper", "light", "missing"],
        ["Paper", "light", "film"],
    ] {
        assert_eq!(calibration.lookup(keys[0], keys[1], keys[2]).unwrap(), None);
    }
}

#[test]
fn selected_branch_errors() {
    let directory = Directory::new();
    for (root, field) in [
        (json!({"paper":[]}), Field::PrintProfile),
        (json!({"paper":{"light":null}}), Field::PrintIlluminant),
        (json!({"paper":{"light":{"film":{}}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,2]}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,2,3,4]}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,null,3]}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,true,3]}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,"2",3]}}}), Field::CmyCc),
    ] {
        let path = directory.json(root);
        let calibration = load_neutral_calibration(&path).unwrap();
        let error = calibration.lookup("paper", "light", "film").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Malformed(field));
        assert_eq!(error.path(), path);
        assert_eq!(error.selection(), Some(["paper", "light", "film"]));
        assert!(error.to_string().contains("paper/light/film"));
    }
}

#[test]
fn unused_malformed_branches() {
    let directory = Directory::new();
    let path = directory.json(json!({
        "paper":{"light":{"film":[0,88.428,105.3677],"bad":[null]},"bad":null},
        "bad":[]
    }));
    let calibration = load_neutral_calibration(path).unwrap();
    for keys in [
        ["bad", "light", "film"],
        ["paper", "bad", "film"],
        ["paper", "light", "bad"],
    ] {
        assert!(calibration.lookup(keys[0], keys[1], keys[2]).is_err());
        let cmy = calibration
            .lookup("paper", "light", "film")
            .unwrap()
            .unwrap();
        assert_eq!(cmy.map(f32::to_bits), [0, 1118886691, 1121107011]);
    }
}

#[test]
fn coefficient_conversion_bits() {
    let directory = Directory::new();
    for (bytes, expected) in [
        // Literal just above the f32 midpoint first rounds to that midpoint in f64.
        (
            b"{\"p\":{\"i\":{\"f\":[-0,-0.0,1.000000059604644775390626]}}}".as_slice(),
            [0x80000000, 0x80000000, 0x3f800000],
        ),
        (
            b"{\"p\":{\"i\":{\"f\":[1e300,-1e300,1e-300]}}}",
            [0x7f800000, 0xff800000, 0],
        ),
        (
            b"{\"p\":{\"i\":{\"f\":[-0e0,1e-0,1.000000059604645]}}}",
            [0x80000000, 0x3f800000, 0x3f800001],
        ),
    ] {
        let calibration = load_neutral_calibration(directory.write(bytes)).unwrap();
        assert_eq!(
            calibration
                .lookup("p", "i", "f")
                .unwrap()
                .unwrap()
                .map(f32::to_bits),
            expected
        );
    }
}

#[test]
fn malformed_root() {
    let directory = Directory::new();
    for bytes in [
        b"".as_slice(),
        b"{",
        b"[]",
        b"null",
        b"{} trailing",
        b"{\"unused\":1e999}",
        b"{\"unused\":NaN}",
    ] {
        let error = load_neutral_calibration(directory.write(bytes)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Malformed(Field::Root));
        assert_eq!(error.selection(), None);
    }
    let calibration =
        load_neutral_calibration(directory.write(b"\xef\xbb\xbf{\"p\":{\"i\":{\"f\":[1,2,3]}}}"))
            .unwrap();
    assert_eq!(
        calibration.lookup("p", "i", "f").unwrap(),
        Some([1.0, 2.0, 3.0])
    );
}

#[test]
fn existing_unreadable_resource() {
    let directory = Directory::new();
    let error = load_neutral_calibration(&directory.0).unwrap_err();
    assert!(matches!(error.kind(), ErrorKind::Read(_)));
    assert_eq!(error.path(), directory.0);
}
