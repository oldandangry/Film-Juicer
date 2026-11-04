//! Film-Juicer raw-reader characterization, captured from unchanged C++ at
//! dece4c21bc9043eb64b009346294550bb721da84 before this Rust translation.
//! Bundle: deaddd9e79004008ff0bce40ba48de616d23456cf026c554bb9b111224744773.
//! Spektrafilm: 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc.
//! Counts, representative bits and FNV-1a of every decoded LE f32 byte are
//! independent C++ expectations. Tests read only tracked Resources and literals.
//! Approved exceptions: shared decimal CSV conversion keeps underflow zeros and
//! skips hex fields. NPY 1.0 decoding preserves values in the supported layouts;
//! bundled fingerprints remain independent regression evidence only.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use film_juicer_core::data_io::{
    ReadError, ReadErrorKind, ReadPart, load_csv_pairs, load_cmf_csv, load_mallett_basis,
    load_spectra_lut,
};
#[path = "support/case_results.rs"]
mod test_cases;
use test_cases::RowCases;

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
            "data-io-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn write(&self, bytes: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join("input");
        fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove data I/O test directory");
    }
}

fn fingerprint(values: impl IntoIterator<Item = f32>) -> u64 {
    values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        })
}

fn assert_invalid(error: ReadError, path: &Path, shape: &str) {
    assert_eq!(error.path, path);
    assert!(
        matches!(error.kind, ReadErrorKind::InvalidNpy { expected, .. } if expected.contains(shape))
    );
    let diagnostic = error.to_string();
    assert!(diagnostic.contains(&path.display().to_string()));
    assert!(diagnostic.contains(shape));
}

type BundleCase = (&'static str, usize, u64, &'static [(usize, u32)]);
const BUNDLE: &[BundleCase] = &[
    (
        "cie1931_2deg.csv",
        81,
        0xc0ea4ea30d5fd861,
        &[
            (0, 0x43be0000),
            (1, 0x3ab34e77),
            (162, 0x3f5eb852),
            (323, 0x00000000),
        ],
    ),
    (
        "illuminants/D50.csv",
        81,
        0xf70ca3af96e9a5e8,
        &[
            (0, 0x43be0000),
            (1, 0x3e91ecb3),
            (81, 0x3f935d1c),
            (161, 0x3f6937e3),
        ],
    ),
    (
        "illuminants/D55.csv",
        81,
        0xbc4634b46165cfff,
        &[
            (0, 0x43be0000),
            (1, 0x3ec23162),
            (81, 0x3f91a3db),
            (161, 0x3f560243),
        ],
    ),
    (
        "illuminants/D65.csv",
        81,
        0xadbdb3d2c250ef10,
        &[
            (0, 0x43be0000),
            (1, 0x3f121b3b),
            (81, 0x3f8c0560),
            (161, 0x3f394da5),
        ],
    ),
    (
        "illuminants/K75P.csv",
        81,
        0x1adba11b770daaa9,
        &[
            (0, 0x43be0000),
            (1, 0x3e4d29dc),
            (81, 0x3fbd542a),
            (161, 0x3cd273d6),
        ],
    ),
    (
        "illuminants/T.csv",
        81,
        0x2013ac33b7e2aaec,
        &[
            (0, 0x43be0000),
            (1, 0x3d356a7b),
            (81, 0x3f784029),
            (161, 0x4000302b),
        ],
    ),
    (
        "filters/heat_absorbing/schott/KG3.csv",
        146,
        0xe040af29a7935a51,
        &[
            (0, 0x4388624d),
            (1, 0x3af43bc9),
            (146, 0x4404b518),
            (291, 0x3b4b8728),
        ],
    ),
    (
        "filters/lens_transmission/canon/canon_24_f28_is.csv",
        107,
        0x74308f00cb53f384,
        &[
            (0, 0x43af01ac),
            (1, 0x42be190c),
            (107, 0x42b0805b),
            (213, 0x4271deee),
        ],
    ),
    (
        "luts/spectral_upsampling/irradiance_xy_tc.npy",
        0,
        0x81ebefd4e4cc9926,
        &[
            (0, 0x40bb0000),
            (1, 0x40bae000),
            (1492992, 0x4031a000),
            (2985983, 0x36240000),
        ],
    ),
    (
        "luts/spectral_upsampling/arctic2026beta04_reflectance_xy_tc.npy",
        0,
        0x9262ffb765e3289e,
        &[
            (0, 0x00000000),
            (1, 0x00000000),
            (1492992, 0x00000000),
            (2985983, 0x00000000),
        ],
    ),
    (
        "luts/spectral_upsampling/mallett2019_basis.npy",
        0,
        0xb2ce998405c55a48,
        &[
            (0, 0x3ea7a880),
            (1, 0x3ea9e9c7),
            (121, 0x3f747714),
            (242, 0x3eaa8d89),
        ],
    ),
];

#[test]
fn bundled_csv_bits() {
    let mut cases = RowCases::new(
        "bundled-csv-resources",
        BUNDLE[..8].iter().map(|(resource, _, _, _)| *resource),
    );
    for &(resource, rows, hash, samples) in &BUNDLE[..8] {
        let row = cases.case(resource);
        let path = repository().join("Resources").join(resource);
        let values: Vec<f32> = if resource == "cie1931_2deg.csv" {
            let csv = load_cmf_csv(&path).unwrap();
            assert_eq!(csv.rows().len(), rows);
            csv.rows().iter().flatten().copied().collect()
        } else {
            let csv = load_csv_pairs(&path).unwrap();
            assert_eq!(csv.rows().len(), rows);
            csv.rows().iter().flatten().copied().collect()
        };
        assert_eq!(fingerprint(values.iter().copied()), hash, "{resource}");
        for &(index, expected) in samples {
            assert_eq!(values[index].to_bits(), expected, "{resource}[{index}]");
        }
        row.passed();
    }
    cases.finish();
}

#[test]
fn bundled_npy_bits() {
    let mut cases = RowCases::new(
        "bundled-npy-resources",
        BUNDLE[8..].iter().map(|(resource, _, _, _)| *resource),
    );
    for &(resource, _, hash, samples) in &BUNDLE[8..] {
        let row = cases.case(resource);
        let path = repository().join("Resources").join(resource);
        let basis = resource.ends_with("mallett2019_basis.npy");
        let values = if basis {
            load_mallett_basis(path).unwrap().as_flattened().to_vec()
        } else {
            load_spectra_lut(path).unwrap()
        };
        assert_eq!(values.len(), if basis { 243 } else { 2_985_984 });
        assert_eq!(fingerprint(values.iter().copied()), hash, "{resource}");
        for &(index, expected) in samples {
            assert_eq!(values[index].to_bits(), expected, "{resource}[{index}]");
        }
        if resource.contains("arctic") {
            assert_eq!(values.iter().copied().fold(0.0_f32, f32::max), 41.0625);
        }
        row.passed();
    }
    cases.finish();
}

#[test]
fn csv_pair_extraction() {
    let directory = Directory::new();
    let cases: &[(&str, &[u32])] = &[
        ("", &[]),
        ("   ", &[]),
        ("# comment", &[]),
        ("; comment", &[]),
        ("1,2", &[0x3f800000, 0x40000000]),
        ("1,,2", &[0x3f800000, 0x40000000]),
        ("1 2", &[0x3f800000, 0x40000000]),
        ("1\t2", &[0x3f800000, 0x40000000]),
        ("1 ,2", &[]),
        ("1, 2", &[0x3f800000, 0x40000000]),
        ("1;2", &[]),
        ("1,2; comment", &[0x3f800000, 0x40000000]),
        ("1,2# comment", &[0x3f800000, 0x40000000]),
        ("1,2,3", &[0x3f800000, 0x40000000]),
        ("1,2tail", &[0x3f800000, 0x40000000]),
        ("1,2e", &[]),
        ("1,2e+", &[]),
        ("1,2e-0", &[0x3f800000, 0x40000000]),
        ("1,2.3.4", &[0x3f800000, 0x40133333]),
        ("1,2-3", &[0x3f800000, 0x40000000]),
        ("1,NaN", &[]),
        ("1,inf", &[]),
        ("1,-inf", &[]),
        ("1,INFINITY", &[]),
        ("NaN,2", &[]),
        ("1,0x1p0", &[]),
        ("0x1,2", &[]),
        ("1", &[]),
        ("broken", &[]),
        ("1,bad", &[]),
        ("-0,+0", &[0x80000000, 0x00000000]),
        ("1,-0e-0", &[0x3f800000, 0x80000000]),
        ("1,1e39", &[]),
        (
            "1,3.40282346638528859811704183484516925440e38",
            &[0x3f800000, 0x7f7fffff],
        ),
        ("1,3.40282356779733661637539395458142568448e38", &[]),
        ("1,1e-50", &[0x3f800000, 0x00000000]),
        ("1,-1e-50", &[0x3f800000, 0x80000000]),
        ("1,1.401298464324817e-45", &[0x3f800000, 0x00000001]),
        ("1,1.000000059604644775390625", &[0x3f800000, 0x3f800000]),
        ("1,1.000000059604644775390626", &[0x3f800000, 0x3f800001]),
        ("1,1.0000000596046448", &[0x3f800000, 0x3f800001]),
        ("+1.,.5", &[0x3f800000, 0x3f000000]),
        ("1,2\0tail", &[0x3f800000, 0x40000000]),
        ("1\x0b2", &[0x3f800000, 0x40000000]),
        ("1\x0c2", &[0x3f800000, 0x40000000]),
        ("1\r2", &[0x3f800000, 0x40000000]),
        ("1,2 4", &[0x3f800000, 0x40000000]),
        ("\u{feff}1,2", &[]),
        (
            "2,3\r\n1,4",
            &[0x40000000, 0x40400000, 0x3f800000, 0x40800000],
        ),
    ];
    let ids: Vec<String> = cases
        .iter()
        .enumerate()
        .map(|(i, _)| format!("csv-line/{i}"))
        .collect();
    let mut rows = RowCases::new("csv-pair-extraction-lines", &ids);
    for ((line, expected), id) in cases.iter().zip(&ids) {
        let row = rows.case(id);
        let csv = load_csv_pairs(directory.write(format!("{line}\n"))).unwrap();
        let actual: Vec<u32> = csv.rows().iter().flatten().map(|v| v.to_bits()).collect();
        assert_eq!(actual, *expected, "{line:?}");
        row.passed();
    }
    rows.finish();
}

#[test]
fn cmf_csv_extraction() {
    let directory = Directory::new();
    let accepted = [
        "1,2,3,4",
        "1 2 3 4",
        "1,,2,,,3,4tail",
        "1,2,3,4,5",
        "1,2,3,4#comment",
        "1,2,3,4;comment",
    ];
    let accepted_ids: Vec<String> = (0..accepted.len())
        .map(|i| format!("accepted/{i}"))
        .collect();
    let mut accepted_rows = RowCases::new("cmf-csv-accepted-forms", &accepted_ids);
    for (line, id) in accepted.into_iter().zip(&accepted_ids) {
        let row = accepted_rows.case(id);
        let csv = load_cmf_csv(directory.write(line)).unwrap();
        assert_eq!(csv.rows(), &[[1.0, 2.0, 3.0, 4.0]], "{line}");
        row.passed();
    }
    accepted_rows.finish();
    let rejected = [
        "1,2,3",
        "1,2,bad,4",
        "1,2,3,NaN",
        "1 ,2,3,4",
        "1,2 ,3,4",
        "1,2,3 ,4",
        "1,2,3,4e+",
        "1;2;3;4",
    ];
    let rejected_ids: Vec<String> = (0..rejected.len())
        .map(|i| format!("rejected/{i}"))
        .collect();
    let mut rejected_rows = RowCases::new("cmf-csv-rejected-forms", &rejected_ids);
    for (line, id) in rejected.into_iter().zip(&rejected_ids) {
        let row = rejected_rows.case(id);
        assert!(
            load_cmf_csv(directory.write(line))
                .unwrap()
                .rows()
                .is_empty(),
            "{line}"
        );
        row.passed();
    }
    rejected_rows.finish();
    let csv = load_cmf_csv(directory.write("-0,+0,-0e0,1e-50")).unwrap();
    assert_eq!(
        csv.rows()[0].map(f32::to_bits),
        [0x8000_0000, 0, 0x8000_0000, 0]
    );
}

#[test]
fn csv_preserves_raw_rows() {
    let directory = Directory::new();
    let csv = load_csv_pairs(directory.write("2,3\nbad\n1,4\n1,5\n\n")).unwrap();
    assert_eq!(csv.rows(), &[[2.0, 3.0], [1.0, 4.0], [1.0, 5.0]]);
    let csv = load_cmf_csv(directory.write("2,3,4,5\n1,2,3,4")).unwrap();
    assert_eq!(csv.rows(), &[[2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0]]);
    assert!(
        load_csv_pairs(directory.write(""))
            .unwrap()
            .rows()
            .is_empty()
    );
}

#[test]
fn csv_line_limit() {
    let directory = Directory::new();
    let mut rows = RowCases::new(
        "csv-line-limit-boundaries",
        ["over-limit-second-line", "at-limit-valid-row"],
    );
    let too_long = rows.case("over-limit-second-line");
    let mut source = b"1,2\n".to_vec();
    source.extend(vec![b'x'; 65_536]);
    let error = load_csv_pairs(directory.write(source)).unwrap_err();
    assert_eq!(error.kind, ReadErrorKind::CsvLineTooLong { line: 2 });
    assert!(error.path.ends_with("input"));
    too_long.passed();
    let at_limit = rows.case("at-limit-valid-row");
    let source = format!("1,2{}", " ".repeat(65_532));
    assert_eq!(
        load_csv_pairs(directory.write(source)).unwrap().rows(),
        &[[1.0, 2.0]]
    );
    at_limit.passed();
    rows.finish();
}

#[test]
fn missing_files() {
    let directory = Directory::new();
    let path = directory.0.join("absent");
    let ids = ["csv-pairs", "cmf-csv", "spectral-lut", "mallett-basis"];
    let mut rows = RowCases::new("missing-resource-files", ids);
    let row = rows.case("csv-pairs");
    let error = load_csv_pairs(&path).unwrap_err();
    assert_eq!(error.path, path);
    assert_eq!(
        error.kind,
        ReadErrorKind::Open(std::io::ErrorKind::NotFound)
    );
    row.passed();
    let row = rows.case("cmf-csv");
    let error = load_cmf_csv(&path).unwrap_err();
    assert_eq!(error.path, path);
    assert_eq!(
        error.kind,
        ReadErrorKind::Open(std::io::ErrorKind::NotFound)
    );
    row.passed();
    let row = rows.case("spectral-lut");
    let error = load_spectra_lut(&path).unwrap_err();
    assert_eq!(error.path, path);
    assert_eq!(
        error.kind,
        ReadErrorKind::Open(std::io::ErrorKind::NotFound)
    );
    row.passed();
    let row = rows.case("mallett-basis");
    let error = load_mallett_basis(&path).unwrap_err();
    assert_eq!(error.path, path);
    assert_eq!(
        error.kind,
        ReadErrorKind::Open(std::io::ErrorKind::NotFound)
    );
    row.passed();
    rows.finish();
}

fn npy(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend(u16::try_from(header.len()).unwrap().to_le_bytes());
    bytes.extend(header.as_bytes());
    bytes.extend(payload);
    bytes
}

fn load_npy(path: &Path, basis: bool) -> Result<Vec<f32>, ReadError> {
    if basis {
        load_mallett_basis(path).map(|basis| basis.as_flattened().to_vec())
    } else {
        load_spectra_lut(path)
    }
}

#[test]
fn equivalent_npy_headers() {
    let directory = Directory::new();
    let ids: Vec<String> = BUNDLE[8..]
        .iter()
        .flat_map(|(resource, _, _, _)| (0..6).map(move |i| format!("{resource}/header/{i}")))
        .collect();
    let mut cases = RowCases::new("equivalent-npy-header-forms", &ids);
    for &(resource, _, hash, _) in &BUNDLE[8..] {
        let bytes = fs::read(repository().join("Resources").join(resource)).unwrap();
        let basis = resource.ends_with("mallett2019_basis.npy");
        let dtype = if basis { "<f4" } else { "<f2" };
        let shape = if basis { "81,3" } else { "192,192,81" };
        for (header_index, header) in [
            format!("{{'descr':'{dtype}','fortran_order':False,'shape':({shape})}}\n"),
            format!("{{\"descr\":\"{dtype}\",\"shape\":({shape},),\"fortran_order\":False,}}\n"),
            format!(" {{'fortran_order':False,'descr':\"{dtype}\",'shape':({shape})}}    \n"),
            format!(
                "{{'fortran_order':False,'shape':({shape},),'descr':'{dtype}'}}{}\n",
                " ".repeat(300)
            ),
            format!(
                "\t{{\n\"shape\" : ( {shape} , ), 'descr' : '{dtype}',\n'fortran_order' : False ,\n}}\t \n"
            ),
            format!("{{'shape':({shape}),'fortran_order':False,'descr':'{dtype}'}}\r\n"),
        ].into_iter().enumerate() {
            let id = format!("{resource}/header/{header_index}");
            let row = cases.case(id);
            let values = load_npy(&directory.write(npy(&header, &bytes[128..])), basis).unwrap();
            assert_eq!(fingerprint(values), hash, "{resource}: {header}");
            row.passed();
        }
    }
    cases.finish();
}

#[test]
fn supported_float_widths_preserve_authored_samples() {
    let directory = Directory::new();
    // Literal conversion expectations include subnormals, both zero signs,
    // negative/out-of-range samples, infinities and NaNs. NaN payloads are exact
    // for f16/f32; the quiet f64 NaNs narrow to distinctive signed payloads.
    let half: [u16; 10] = [
        0, 0x8000, 0x3c00, 0xc000, 0x7bff, 1, 0x7c00, 0xfc00, 0x7e23, 0xfe35,
    ];
    let half_bits = [
        0,
        0x8000_0000,
        0x3f80_0000,
        0xc000_0000,
        0x477f_e000,
        0x3380_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc4_6000,
        0xffc6_a000,
    ];
    let single: [u32; 10] = [
        0,
        0x8000_0000,
        0x3f80_0000,
        0xc000_0000,
        0x7f7f_ffff,
        1,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
    ];
    let double = [
        0.0f64,
        -0.0,
        1.0,
        -2.0,
        1e40,
        -1e40,
        1e-300,
        -1e-300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_2468_a000_0000),
        f64::from_bits(0xfff8_a864_2000_0000),
        f64::from_bits(0x3ff0_0000_1000_0000),
        f64::from_bits(0x3ff0_0000_1000_0001),
    ];
    let double_bits = [
        0,
        0x8000_0000,
        0x3f80_0000,
        0xc000_0000,
        0x7f80_0000,
        0xff80_0000,
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
        0x3f80_0000,
        0x3f80_0001,
    ];
    let types = [
        (
            "<f2",
            half.into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
            half_bits.to_vec(),
        ),
        (
            "<f4",
            single.into_iter().flat_map(u32::to_le_bytes).collect(),
            single.to_vec(),
        ),
        (
            "<f8",
            double.into_iter().flat_map(f64::to_le_bytes).collect(),
            double_bits.to_vec(),
        ),
    ];
    let ids: Vec<String> = ["<f2", "<f4", "<f8"]
        .into_iter()
        .flat_map(|dtype| {
            ["spectra-lut", "mallett-basis"].map(move |asset| format!("{dtype}/{asset}"))
        })
        .collect();
    let mut cases = RowCases::new("npy-float-width-samples", &ids);
    for (dtype, pattern, expected) in types {
        for (basis, shape, count, asset) in [
            (false, "192,192,81", 2_985_984, "spectra-lut"),
            (true, "81,3", 243, "mallett-basis"),
        ] {
            let row = cases.case(format!("{dtype}/{asset}"));
            let width = pattern.len() / expected.len();
            let mut payload = Vec::with_capacity(count * width);
            for index in 0..count {
                let start = (index % expected.len()) * width;
                payload.extend(&pattern[start..start + width]);
            }
            let header =
                format!("{{'shape':({shape},),\"descr\":'{dtype}','fortran_order':False}}\n");
            let values = load_npy(&directory.write(npy(&header, &payload)), basis).unwrap();
            assert_eq!(values.len(), count);
            for (index, sample) in values.into_iter().enumerate() {
                assert_eq!(
                    sample.to_bits(),
                    expected[index % expected.len()],
                    "{dtype} basis={basis} [{index}]"
                );
            }
            row.passed();
        }
    }
    cases.finish();
}

#[test]
fn changed_bundled_contents_load() {
    let directory = Directory::new();
    let mut cases = RowCases::new(
        "changed-bundled-npy-content",
        BUNDLE[8..].iter().map(|(resource, _, _, _)| *resource),
    );
    for &(resource, _, hash, _) in &BUNDLE[8..] {
        let row = cases.case(resource);
        let mut bytes = fs::read(repository().join("Resources").join(resource)).unwrap();
        let basis = resource.ends_with("mallett2019_basis.npy");
        // Change a finite sample without changing the representation.
        if basis {
            bytes[128..132].copy_from_slice(&(-1234.5f32).to_le_bytes());
        } else {
            bytes[128..130].copy_from_slice(&0xc000u16.to_le_bytes());
        }
        let values = load_npy(&directory.write(&bytes), basis).unwrap();
        assert_eq!(values[0], if basis { -1234.5 } else { -2.0 });
        assert_ne!(fingerprint(values), hash);
        if basis {
            bytes[128..132].copy_from_slice(&0xffc1_2345u32.to_le_bytes());
        } else {
            bytes[128..130].copy_from_slice(&0xfe23u16.to_le_bytes());
        }
        let values = load_npy(&directory.write(&bytes), basis).unwrap();
        assert_eq!(
            values[0].to_bits(),
            if basis { 0xffc1_2345 } else { 0xffc4_6000 }
        );
        row.passed();
    }
    cases.finish();
}

#[test]
fn rejects_malformed_or_unsupported_npy_representations() {
    let directory = Directory::new();
    let layouts = [
        (false, "192,192,81", "spectra-lut"),
        (true, "81,3", "mallett-basis"),
    ];
    let mut layout_cases = RowCases::new(
        "npy-malformed-layouts",
        layouts.iter().map(|(_, _, id)| *id),
    );
    for (basis, shape, layout_id) in layouts {
        let layout_case = layout_cases.case(layout_id);
        let layout = if basis { "(81, 3)" } else { "(192, 192, 81)" };
        let valid = format!("{{'descr':'<f4','fortran_order':False,'shape':({shape})}}\n");
        let mut headers = vec![
            "{}\n".to_owned(),
            format!("{{'fortran_order':False,'shape':({shape})}}\n"),
            format!("{{'descr':'<f4','shape':({shape})}}\n"),
            "{'descr':'<f4','fortran_order':False}\n".to_owned(),
            valid.replace("'descr':'<f4'", "'descr':'<f4','descr':'<f4'"),
            valid.replace(
                "'fortran_order':False",
                "'fortran_order':False,\"fortran_order\":False",
            ),
            valid.replace(
                &format!("'shape':({shape})"),
                &format!("'shape':({shape}),'shape':({shape})"),
            ),
            valid.replace("False", "True"),
            valid.replace("False", "false"),
            valid.replace("False", "0"),
            valid.replace("False", "'False'"),
            valid.replace("False", "Falsehood"),
            valid.replace("'<f4'", "[('field','<f4')]"),
            valid.replace("'descr'", "'de\\x73cr'"),
            valid.replace("'descr'", "descr"),
            valid.replace("'descr':'<f4'", "'descr':'<f4', 'extra':0"),
            valid.replace("'descr':'<f4'", "'descr':'<f4"),
            valid.replace(':', "="),
            valid.replace("'descr':'<f4',", "'descr':'<f4'"),
            valid.replace("'descr':'<f4',", "'descr':'<f4',,"),
            valid.replace('}', ""),
            valid.replace('}', "} garbage"),
            valid.trim_end().to_owned(),
            valid.replace("<f4", "<f\u{e9}"),
        ];
        for dtype in [
            ">f2", ">f4", ">f8", "|f4", "=f4", "f4", "<i4", "<u4", "<c8", "|O", "<f16",
        ] {
            headers.push(valid.replace("<f4", dtype));
        }
        for wrong_shape in [
            "()",
            "(81)",
            "(81,)",
            "(3,81)",
            "(81,3,1)",
            "(192,192,80)",
            "(192,192,81,1)",
            "(191,192,81)",
            "(0,3)",
            "(-81,3)",
            "(81.0,3)",
            "('81',3)",
            "[81,3]",
            "(81,,3)",
            "(81 3)",
            "(81,3,,)",
            "(081,3)",
            "(81*1,3)",
        ] {
            headers.push(valid.replace(&format!("({shape})"), wrong_shape));
        }
        let header_ids: Vec<String> = (0..headers.len()).map(|i| format!("header/{i}")).collect();
        let mut header_cases =
            RowCases::new(format!("npy-invalid-header/{layout_id}"), &header_ids);
        for (header, id) in headers.into_iter().zip(&header_ids) {
            let row = header_cases.case(id);
            let path = directory.write(npy(&header, &[]));
            assert_invalid(load_npy(&path, basis).unwrap_err(), &path, layout);
            row.passed();
        }
        header_cases.finish();
        let versions = [(0, 0), (1, 1), (2, 0), (3, 0), (255, 255)];
        let version_ids: Vec<String> = versions
            .iter()
            .map(|(major, minor)| format!("version/{major}/{minor}"))
            .collect();
        let mut version_cases =
            RowCases::new(format!("npy-invalid-versions/{layout_id}"), &version_ids);
        for ((major, minor), id) in versions.into_iter().zip(&version_ids) {
            let row = version_cases.case(id);
            let mut bytes = npy(&valid, &[]);
            bytes[6] = major;
            bytes[7] = minor;
            let path = directory.write(bytes);
            assert_invalid(load_npy(&path, basis).unwrap_err(), &path, layout);
            row.passed();
        }
        version_cases.finish();
        let mut magic_cases =
            RowCases::new(format!("npy-invalid-magic/{layout_id}"), ["bad-magic"]);
        let magic = magic_cases.case("bad-magic");
        let mut bytes = npy(&valid, &[]);
        bytes[0] = 0;
        let path = directory.write(bytes);
        assert_invalid(load_npy(&path, basis).unwrap_err(), &path, layout);
        magic.passed();
        magic_cases.finish();
        layout_case.passed();
    }
    layout_cases.finish();
}

#[test]
fn rejects_npy_size_overflow() {
    let directory = Directory::new();
    let layouts = [
        (false, ",192,81", "spectra-lut"),
        (true, ",3", "mallett-basis"),
    ];
    let ids = ["spectra-lut", "mallett-basis"];
    let mut layout_cases = RowCases::new("npy-size-overflow-layouts", ids);
    for (basis, tail, name) in layouts {
        let layout = layout_cases.case(name);
        let dimensions = [
            format!("{}0", usize::MAX),
            usize::MAX.to_string(),
            (usize::MAX / 8).to_string(),
        ];
        let ids: Vec<String> = (0..dimensions.len())
            .map(|i| format!("dimension/{i}"))
            .collect();
        let mut cases = RowCases::new(format!("npy-overflow-dimensions/{name}"), &ids);
        for (dimension, id) in dimensions.into_iter().zip(&ids) {
            let row = cases.case(id);
            let header =
                format!("{{'descr':'<f8','fortran_order':False,'shape':({dimension}{tail})}}\n");
            let path = directory.write(npy(&header, &[]));
            let error = load_npy(&path, basis).unwrap_err();
            assert_eq!(error.path, path);
            assert_eq!(error.kind, ReadErrorKind::Size);
            row.passed();
        }
        cases.finish();
        layout.passed();
    }
    layout_cases.finish();
}

#[test]
fn rejects_truncated_npy_headers_and_payloads() {
    let directory = Directory::new();
    let layouts = [
        (false, "192,192,81", 2_985_984, "spectra-lut"),
        (true, "81,3", 243, "mallett-basis"),
    ];
    let mut layouts_cases = RowCases::new(
        "npy-truncated-layouts",
        layouts.iter().map(|(_, _, _, id)| *id),
    );
    for (basis, shape, count, name) in layouts {
        let layout_case = layouts_cases.case(name);
        let dtypes = [("<f2", 2), ("<f4", 4), ("<f8", 8)];
        let dtype_ids = ["<f2", "<f4", "<f8"];
        let mut dtype_cases = RowCases::new(format!("npy-truncated-dtypes/{name}"), dtype_ids);
        for ((dtype, width), dtype_id) in dtypes.into_iter().zip(dtype_ids) {
            let dtype_case = dtype_cases.case(dtype_id);
            let header = format!("{{'descr':'{dtype}','fortran_order':False,'shape':({shape})}}\n");
            let mut bytes = npy(&header, &[]);
            let offset = bytes.len();
            bytes.resize(offset + count * width, 0);
            let cuts = [
                (0, ReadPart::Header),
                (6, ReadPart::Header),
                (9, ReadPart::Header),
                (10, ReadPart::Header),
                (offset - 1, ReadPart::Header),
                (offset, ReadPart::Payload),
                (bytes.len() - width, ReadPart::Payload),
                (bytes.len() - 1, ReadPart::Payload),
            ];
            let ids: Vec<String> = cuts
                .iter()
                .enumerate()
                .map(|(i, (length, part))| format!("cut/{i}/{part:?}/{length}"))
                .collect();
            let mut cut_cases = RowCases::new(format!("npy-truncated-cuts/{name}/{dtype}"), &ids);
            for ((length, part), id) in cuts.into_iter().zip(&ids) {
                let row = cut_cases.case(id);
                let path = directory.write(&bytes[..length]);
                let error = load_npy(&path, basis).unwrap_err();
                assert_eq!(error.path, path);
                assert_eq!(
                    error.kind,
                    ReadErrorKind::ShortRead(part),
                    "{dtype} length={length}"
                );
                row.passed();
            }
            cut_cases.finish();
            dtype_case.passed();
        }
        dtype_cases.finish();
        layout_case.passed();
    }
    layouts_cases.finish();
}

#[test]
fn rejects_trailing_npy_payload_bytes() {
    let directory = Directory::new();
    let mut rows = RowCases::new(
        "npy-trailing-payload-bytes",
        BUNDLE[8..].iter().map(|(resource, _, _, _)| *resource),
    );
    for &(resource, _, _, _) in &BUNDLE[8..] {
        let row = rows.case(resource);
        let mut bytes = fs::read(repository().join("Resources").join(resource)).unwrap();
        let basis = resource.ends_with("mallett2019_basis.npy");
        bytes.push(0);
        let path = directory.write(bytes);
        assert_invalid(
            load_npy(&path, basis).unwrap_err(),
            &path,
            if basis { "(81, 3)" } else { "(192, 192, 81)" },
        );
        row.passed();
    }
    rows.finish();
}
