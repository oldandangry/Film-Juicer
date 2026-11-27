//! Neutral calibration characterization from unchanged C++ at
//! 4c848d2f35ca8861b4e322a3891febfc807dc620. The FNV-1a expectation covers all
//! 160 bundled CMY triplets in sorted print/illuminant/film order, LE f32 bits.
//! Bundle deaddd9e79004008ff0bce40ba48de616d23456cf026c554bb9b111224744773;
//! spektrafilm 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc. Bare -0 follows the
//! previously approved direct-Serde input contract, not native integer-zero bits.

#![forbid(unsafe_code)]

use std::error::Error as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use film_juicer_core::assets::Assets;
use film_juicer_core::data_io::calibration::{self, ErrorKind, Field, load_neutral_calibration};
use film_juicer_core::recipe::{self, Calibration, CalibrationError, PrepareError, PreparedRoute};
use film_juicer_core::{capture, route::ScanRoute};

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
fn print_preparation_retains_calibration_cause() {
    let directory = Directory::new();
    let resources = repository().join("Resources");
    for filename in [
        "cie1931_2deg.csv",
        "profiles/kodak_portra_400.json",
        "profiles/kodak_portra_endura.json",
        "luts/spectral_upsampling/mallett2019_basis.npy",
        "illuminants/D65.csv",
        "illuminants/D55.csv",
        "illuminants/D50.csv",
        "illuminants/T.csv",
        "illuminants/K75P.csv",
        "filters/heat_absorbing/schott/KG3.csv",
        "filters/lens_transmission/canon/canon_24_f28_is.csv",
    ] {
        let target = directory.0.join(filename);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(resources.join(filename), target).unwrap();
    }
    let path = directory.0.join("filters/neutral_print_filters.json");
    let mut params = capture::defaults();
    params.film_profile_key = "kodak_portra_400".into();
    params.print_profile_key = "kodak_portra_endura".into();
    params.scan_route = ScanRoute::NegativePrint;
    params.spectral_upsampling_mode = 1;
    params.enl_ill = 3;

    for (input, field, diagnostic, selection) in [
        ("[]", Field::Root, "root", None),
        (
            r#"{"kodak_portra_endura":{"TH-KG3":{"kodak_portra_400":[1,2]}}}"#,
            Field::CmyCc,
            "cmy_cc",
            Some(["kodak_portra_endura", "TH-KG3", "kodak_portra_400"]),
        ),
    ] {
        fs::write(&path, input).unwrap();
        // Calibration is evaluated before Print gamma.
        params.print_gamma_factor = 0.0;
        let assets = Assets::new(directory.0.clone());
        let Err(error @ PrepareError::Calibration(_)) = recipe::prepare(&assets, &params) else {
            panic!("expected calibration to be the first failure");
        };
        assert_eq!(
            error.to_string(),
            format!("MalformedNeutralPrintCalibration phase=4A field={diagnostic}")
        );
        let wrapper = error
            .source()
            .unwrap()
            .downcast_ref::<CalibrationError>()
            .unwrap();
        assert!(matches!(
            (wrapper, field),
            (CalibrationError::Resource(_), Field::Root)
                | (CalibrationError::Entry(_), Field::CmyCc)
        ));
        let cause = wrapper
            .source()
            .unwrap()
            .downcast_ref::<calibration::Error>()
            .unwrap();
        assert_eq!(cause.path(), path);
        assert_eq!(cause.selection(), selection);
        assert_eq!(cause.kind(), ErrorKind::Malformed(field));
    }

    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    let Err(error @ PrepareError::Calibration(_)) =
        recipe::prepare(&Assets::new(directory.0.clone()), &params)
    else {
        panic!("expected calibration resource read failure");
    };
    assert_eq!(
        error.to_string(),
        "MalformedNeutralPrintCalibration phase=4A field=resource_read"
    );
    let wrapper = error
        .source()
        .unwrap()
        .downcast_ref::<CalibrationError>()
        .unwrap();
    assert!(matches!(wrapper, CalibrationError::Resource(_)));
    let cause = wrapper
        .source()
        .unwrap()
        .downcast_ref::<calibration::Error>()
        .unwrap();
    assert_eq!(cause.path(), path);
    assert_eq!(cause.selection(), None);
    assert!(matches!(cause.kind(), ErrorKind::Read(_)));
    fs::remove_dir(&path).unwrap();

    params.print_gamma_factor = 1.0;
    for expected in [Calibration::MissingFile, Calibration::MissingEntry] {
        if expected == Calibration::MissingEntry {
            fs::write(&path, "{}").unwrap();
        }
        let state = recipe::prepare(&Assets::new(directory.0.clone()), &params).unwrap();
        let PreparedRoute::Print(print) = state.route() else {
            panic!("expected Print preparation");
        };
        assert_eq!(print.calibration(), expected);
    }
}

#[test]
fn bundled_calibration() {
    let calibration =
        load_neutral_calibration(repository().join("Resources/filters/neutral_print_filters.json"))
            .unwrap();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let ids: Vec<String> = PRINTS
        .iter()
        .flat_map(|print| {
            FILMS
                .iter()
                .map(move |film| format!("{print}/{film}/TH-KG3"))
        })
        .collect();
    let cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "bundled-print-film-lookups"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "bundled-print-film-lookups"
    );
    for print in PRINTS {
        for film in FILMS {
            let row = AsRef::<str>::as_ref(&(format!("{print}/{film}/TH-KG3"))).to_owned();
            let cmy = calibration.lookup(print, "TH-KG3", film).unwrap().unwrap();
            for byte in cmy.into_iter().flat_map(f32::to_le_bytes) {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
            }
            assert!(
                cases_remaining.remove(&row),
                "unexpected or repeated comparison: {row}"
            );
        }
    }
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
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
    let selections = [
        ["missing", "light", "film"],
        ["paper", "missing", "film"],
        ["paper", "light", "missing"],
        ["Paper", "light", "film"],
    ];
    let ids: Vec<String> = selections
        .iter()
        .enumerate()
        .map(|(i, _)| format!("missing-selection/{i}"))
        .collect();
    let rows_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !rows_ids.is_empty(),
        "empty case set: {}",
        "missing-calibration-selection-paths"
    );
    let rows_count = rows_ids.len();
    let mut rows_remaining = rows_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        rows_remaining.len(),
        rows_count,
        "duplicate case IDs: {}",
        "missing-calibration-selection-paths"
    );
    for (keys, id) in selections.into_iter().zip(&ids) {
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        assert_eq!(calibration.lookup(keys[0], keys[1], keys[2]).unwrap(), None);
        assert!(
            rows_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        rows_remaining.is_empty(),
        "missing comparisons: {rows_remaining:?}"
    );
}

#[test]
fn selected_branch_errors() {
    let directory = Directory::new();
    let cases = [
        (json!({"paper":[]}), Field::PrintProfile),
        (json!({"paper":{"light":null}}), Field::PrintIlluminant),
        (json!({"paper":{"light":{"film":{}}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,2]}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,2,3,4]}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,null,3]}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,true,3]}}}), Field::CmyCc),
        (json!({"paper":{"light":{"film":[1,"2",3]}}}), Field::CmyCc),
    ];
    let ids: Vec<String> = (0..cases.len())
        .map(|i| format!("selected-malformed-branch/{i}"))
        .collect();
    let rows_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !rows_ids.is_empty(),
        "empty case set: {}",
        "selected-calibration-branch-errors"
    );
    let rows_count = rows_ids.len();
    let mut rows_remaining = rows_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        rows_remaining.len(),
        rows_count,
        "duplicate case IDs: {}",
        "selected-calibration-branch-errors"
    );
    for ((root, field), id) in cases.into_iter().zip(&ids) {
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        let path = directory.json(root);
        let calibration = load_neutral_calibration(&path).unwrap();
        let error = calibration.lookup("paper", "light", "film").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Malformed(field));
        assert_eq!(error.path(), path);
        assert_eq!(error.selection(), Some(["paper", "light", "film"]));
        assert!(error.to_string().contains("paper/light/film"));
        assert!(
            rows_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        rows_remaining.is_empty(),
        "missing comparisons: {rows_remaining:?}"
    );
}

#[test]
fn unused_malformed_branches() {
    let directory = Directory::new();
    let path = directory.json(json!({
        "paper":{"light":{"film":[0,88.428,105.3677],"bad":[null]},"bad":null},
        "bad":[]
    }));
    let calibration = load_neutral_calibration(path).unwrap();
    let selections = [
        ["bad", "light", "film"],
        ["paper", "bad", "film"],
        ["paper", "light", "bad"],
    ];
    let ids: Vec<String> = (0..selections.len())
        .map(|i| format!("unused-malformed-branch/{i}"))
        .collect();
    let rows_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !rows_ids.is_empty(),
        "empty case set: {}",
        "unused-malformed-calibration-branches"
    );
    let rows_count = rows_ids.len();
    let mut rows_remaining = rows_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        rows_remaining.len(),
        rows_count,
        "duplicate case IDs: {}",
        "unused-malformed-calibration-branches"
    );
    for (keys, id) in selections.into_iter().zip(&ids) {
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        assert!(calibration.lookup(keys[0], keys[1], keys[2]).is_err());
        let cmy = calibration
            .lookup("paper", "light", "film")
            .unwrap()
            .unwrap();
        assert_eq!(cmy.map(f32::to_bits), [0, 1118886691, 1121107011]);
        assert!(
            rows_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        rows_remaining.is_empty(),
        "missing comparisons: {rows_remaining:?}"
    );
}

#[test]
fn coefficient_conversion_bits() {
    let directory = Directory::new();
    let cases = [
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
    ];
    let ids = [
        "negative-zero-midpoint",
        "f64-overflow-underflow",
        "decimal-midpoint-neighbor",
    ];
    let rows_ids: Vec<_> = IntoIterator::into_iter(ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !rows_ids.is_empty(),
        "empty case set: {}",
        "calibration-coefficient-conversion"
    );
    let rows_count = rows_ids.len();
    let mut rows_remaining = rows_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        rows_remaining.len(),
        rows_count,
        "duplicate case IDs: {}",
        "calibration-coefficient-conversion"
    );
    for ((bytes, expected), id) in cases.into_iter().zip(ids) {
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        let calibration = load_neutral_calibration(directory.write(bytes)).unwrap();
        assert_eq!(
            calibration
                .lookup("p", "i", "f")
                .unwrap()
                .unwrap()
                .map(f32::to_bits),
            expected
        );
        assert!(
            rows_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        rows_remaining.is_empty(),
        "missing comparisons: {rows_remaining:?}"
    );
}

#[test]
fn malformed_root() {
    let directory = Directory::new();
    let documents = [
        b"".as_slice(),
        b"{",
        b"[]",
        b"null",
        b"{} trailing",
        b"{\"unused\":1e999}",
        b"{\"unused\":NaN}",
    ];
    let ids: Vec<String> = (0..documents.len())
        .map(|i| format!("malformed-json/{i}"))
        .collect();
    let rows_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !rows_ids.is_empty(),
        "empty case set: {}",
        "malformed-calibration-root-documents"
    );
    let rows_count = rows_ids.len();
    let mut rows_remaining = rows_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        rows_remaining.len(),
        rows_count,
        "duplicate case IDs: {}",
        "malformed-calibration-root-documents"
    );
    for (bytes, id) in documents.into_iter().zip(&ids) {
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        let error = load_neutral_calibration(directory.write(bytes)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Malformed(Field::Root));
        assert_eq!(error.selection(), None);
        assert!(
            rows_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    assert!(
        rows_remaining.is_empty(),
        "missing comparisons: {rows_remaining:?}"
    );
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
