//! Film-Juicer raw-reader characterization, captured from unchanged C++ at
//! dece4c21bc9043eb64b009346294550bb721da84 before this Rust translation.
//! Bundle: deaddd9e79004008ff0bce40ba48de616d23456cf026c554bb9b111224744773.
//! Spektrafilm: 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc.
//! Counts, representative bits and FNV-1a of every decoded LE f32 byte are
//! independent C++ expectations. Tests read only tracked Resources and literals.
//! Approved exceptions: shared decimal CSV conversion keeps underflow zeros and
//! skips hex fields; NPY construction rejects malformed headers/versions/sizes.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use film_juicer_core::data_io::{
    HeaderError, NpyArray, NpyByteOrder, NpyDtype, ReadError, ReadErrorKind, ReadPart,
    load_csv_pairs, load_csv_triplets, load_mallett_basis, load_spectra_lut,
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
    fn lut(&self, header: &str, payload: &[u8]) -> Result<NpyArray, ReadError> {
        load_spectra_lut(self.write(npy([1, 0], header, payload)))
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove data I/O test directory");
    }
}

fn npy(version: [u8; 2], header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.extend(version);
    if version[0] == 1 {
        bytes.extend(u16::try_from(header.len()).unwrap().to_le_bytes());
    } else {
        bytes.extend(u32::try_from(header.len()).unwrap().to_le_bytes());
    }
    bytes.extend(header.as_bytes());
    bytes.extend(payload);
    bytes
}

fn header(dtype: &str, shape: &str) -> String {
    format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}\n")
}

fn identity(values: impl IntoIterator<Item = f32>) -> u64 {
    values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        })
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn assert_error(result: Result<NpyArray, ReadError>, expected: ReadErrorKind) {
    let error = result.unwrap_err();
    assert!(error.path.ends_with("input"));
    assert_eq!(error.kind, expected);
    assert!(error.to_string().contains("input"));
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
    for &(resource, rows, hash, samples) in &BUNDLE[..8] {
        let path = repository().join("Resources").join(resource);
        let values: Vec<f32> = if resource == "cie1931_2deg.csv" {
            let csv = load_csv_triplets(&path).unwrap();
            assert_eq!(csv.rows().len(), rows);
            csv.rows().iter().flatten().copied().collect()
        } else {
            let csv = load_csv_pairs(&path).unwrap();
            assert_eq!(csv.rows().len(), rows);
            csv.rows().iter().flatten().copied().collect()
        };
        assert_eq!(identity(values.iter().copied()), hash, "{resource}");
        for &(index, expected) in samples {
            assert_eq!(values[index].to_bits(), expected, "{resource}[{index}]");
        }
    }
}

#[test]
fn bundled_npy_bits() {
    for &(resource, _, hash, samples) in &BUNDLE[8..] {
        let path = repository().join("Resources").join(resource);
        let basis = resource.ends_with("mallett2019_basis.npy");
        let array = if basis {
            load_mallett_basis(path)
        } else {
            load_spectra_lut(path)
        }
        .unwrap();
        assert_eq!(array.version(), [1, 0]);
        assert_eq!(array.byte_order(), NpyByteOrder::Little);
        assert_eq!(
            array.dtype(),
            if basis { NpyDtype::F32 } else { NpyDtype::F16 }
        );
        assert_eq!(
            array.shape(),
            if basis {
                &[81, 3][..]
            } else {
                &[192, 192, 81][..]
            }
        );
        assert_eq!(array.source_shape(), array.shape());
        assert_eq!(array.values().len(), if basis { 243 } else { 2_985_984 });
        assert_eq!(identity(array.values().iter().copied()), hash, "{resource}");
        for &(index, expected) in samples {
            assert_eq!(
                array.values()[index].to_bits(),
                expected,
                "{resource}[{index}]"
            );
        }
        if resource.contains("arctic") {
            assert_eq!(
                array.values().iter().copied().fold(0.0_f32, f32::max),
                41.0625
            );
        }
    }
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
    for &(line, expected) in cases {
        let csv = load_csv_pairs(directory.write(format!("{line}\n"))).unwrap();
        let actual: Vec<u32> = csv.rows().iter().flatten().map(|v| v.to_bits()).collect();
        assert_eq!(actual, expected, "{line:?}");
    }
}

#[test]
fn csv_triplet_extraction() {
    let directory = Directory::new();
    for line in [
        "1,2,3,4",
        "1 2 3 4",
        "1,,2,,,3,4tail",
        "1,2,3,4,5",
        "1,2,3,4#comment",
        "1,2,3,4;comment",
    ] {
        let csv = load_csv_triplets(directory.write(line)).unwrap();
        assert_eq!(csv.rows(), &[[1.0, 2.0, 3.0, 4.0]], "{line}");
    }
    for line in [
        "1,2,3",
        "1,2,bad,4",
        "1,2,3,NaN",
        "1 ,2,3,4",
        "1,2 ,3,4",
        "1,2,3 ,4",
        "1,2,3,4e+",
        "1;2;3;4",
    ] {
        assert!(
            load_csv_triplets(directory.write(line))
                .unwrap()
                .rows()
                .is_empty(),
            "{line}"
        );
    }
    let csv = load_csv_triplets(directory.write("-0,+0,-0e0,1e-50")).unwrap();
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
    let csv = load_csv_triplets(directory.write("2,3,4,5\n1,2,3,4")).unwrap();
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
    let mut source = b"1,2\n".to_vec();
    source.extend(vec![b'x'; 65_536]);
    let error = load_csv_pairs(directory.write(source)).unwrap_err();
    assert_eq!(error.kind, ReadErrorKind::CsvLineTooLong { line: 2 });
    assert!(error.path.ends_with("input"));
    let source = format!("1,2{}", " ".repeat(65_532));
    assert_eq!(
        load_csv_pairs(directory.write(source)).unwrap().rows(),
        &[[1.0, 2.0]]
    );
}

#[test]
fn missing_files() {
    let directory = Directory::new();
    let path = directory.0.join("absent");
    for error in [
        load_csv_pairs(&path).unwrap_err(),
        load_csv_triplets(&path).unwrap_err(),
        load_spectra_lut(&path).unwrap_err(),
        load_mallett_basis(&path).unwrap_err(),
    ] {
        assert_eq!(error.path, path);
        assert_eq!(
            error.kind,
            ReadErrorKind::Open(std::io::ErrorKind::NotFound)
        );
    }
}

#[test]
fn half_conversion_bits() {
    let directory = Directory::new();
    let source = [
        0u16, 0x8000, 1, 0x8001, 0x3ff, 0x400, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7c01, 0x7e00,
        0xffff,
    ];
    let payload: Vec<u8> = source.into_iter().flat_map(u16::to_le_bytes).collect();
    let array = directory.lut(&header("<f2", "(1,1,13)"), &payload).unwrap();
    assert_eq!(
        bits(array.values()),
        [
            0,
            0x8000_0000,
            0x3380_0000,
            0xb380_0000,
            0x387f_c000,
            0x3880_0000,
            0x3f80_0000,
            0x477f_e000,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_2000,
            0x7fc0_0000,
            0xffff_e000
        ]
    );
    let payload: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let array = directory
        .lut(&header("<f2", "(1,1,65536)"), &payload)
        .unwrap();
    assert_eq!(
        identity(array.values().iter().copied()),
        12_710_733_099_726_876_997
    );
}

#[test]
fn f32_preserves_payload_bits() {
    let directory = Directory::new();
    let expected = [
        0u32,
        0x8000_0000,
        1,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0x7fc1_2345,
        0xffc1_2345,
        0x3f80_0001,
    ];
    let payload: Vec<u8> = expected.into_iter().flat_map(u32::to_le_bytes).collect();
    let array = directory.lut(&header("<f4", "(1,1,9)"), &payload).unwrap();
    assert_eq!(bits(array.values()), expected);
}

#[test]
fn f64_narrowing() {
    let directory = Directory::new();
    let source = [
        0.0_f64,
        -0.0,
        1.0000000596046448,
        1.000000059604645,
        1e300,
        -1e300,
        1e-300,
        -1e-300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let payload: Vec<u8> = source.into_iter().flat_map(f64::to_le_bytes).collect();
    let array = directory.lut(&header("<f8", "(1,1,11)"), &payload).unwrap();
    assert_eq!(
        bits(array.values()),
        [
            0,
            0x8000_0000,
            0x3f80_0000,
            0x3f80_0001,
            0x7f80_0000,
            0xff80_0000,
            0,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000
        ]
    );
}

#[test]
fn mallett_transpose() {
    let directory = Directory::new();
    let payload: Vec<u8> = (0..243).flat_map(|v| (v as f32).to_le_bytes()).collect();
    for (shape, hash) in [
        ("(81,3)", 7_430_929_967_373_113_414),
        ("(3,81)", 16_790_069_341_578_716_178),
    ] {
        let array =
            load_mallett_basis(directory.write(npy([1, 0], &header("<f4", shape), &payload)))
                .unwrap();
        assert_eq!(array.shape(), [81, 3]);
        assert_eq!(
            array.source_shape(),
            if shape == "(81,3)" { [81, 3] } else { [3, 81] }
        );
        assert_eq!(identity(array.values().iter().copied()), hash);
        if shape == "(3,81)" {
            assert_eq!(&array.values()[..6], &[0.0, 81.0, 162.0, 1.0, 82.0, 163.0]);
        }
    }
}

#[test]
fn supported_versions_and_headers() {
    let directory = Directory::new();
    for version in [[1, 0], [2, 0], [3, 0]] {
        let text = "{\"shape\": (+1, 1, 1,), \"descr\": \"<f4\", \"fortran_order\": False}   ";
        let array =
            load_spectra_lut(directory.write(npy(version, text, &2.0_f32.to_le_bytes()))).unwrap();
        assert_eq!(array.version(), version);
        assert_eq!(array.values(), [2.0]);
    }
}

#[test]
fn accepted_byte_order_spellings() {
    let directory = Directory::new();
    for (dtype, bytes) in [
        ("f2", 0x3c00_u16.to_le_bytes().to_vec()),
        ("f4", 1.0_f32.to_le_bytes().to_vec()),
        ("f8", 1.0_f64.to_le_bytes().to_vec()),
    ] {
        for (prefix, order) in [
            ("<", NpyByteOrder::Little),
            ("|", NpyByteOrder::NotApplicable),
        ] {
            let array = directory
                .lut(&header(&format!("{prefix}{dtype}"), "(1,1,1)"), &bytes)
                .unwrap();
            assert_eq!(array.byte_order(), order);
            assert_eq!(array.values(), [1.0]);
        }
    }
}

#[test]
fn rejects_magic_and_versions() {
    let directory = Directory::new();
    let mut bytes = npy([1, 0], &header("<f4", "(1,1,1)"), &[0; 4]);
    bytes[0] = 0;
    assert_error(
        load_spectra_lut(directory.write(bytes)),
        ReadErrorKind::Magic,
    );
    for version in [[0, 0], [1, 1], [2, 1], [3, 1], [4, 0], [255, 255]] {
        assert_error(
            load_spectra_lut(directory.write(npy(version, "", &[]))),
            ReadErrorKind::Version(version),
        );
    }
}

#[test]
fn rejects_header_substitutions() {
    let directory = Directory::new();
    for text in [
        "{'descr':'<f4','shape':(1,1,1)}",
        "{'foo':'<f4','fortran_order':False,'shape':(1,1,1)}",
        "{'descr':'<f4','descr':'<f8','fortran_order':False,'shape':(1,1,1)}",
        "{'descr':'<f4','fortran_order':False,'shape':(1,1,1),'note':'True'}",
        "{'descr':'<f4','fortran_order':False}",
        "{'descr':'<f4','fortran_order':False,'fortran_order':False,'shape':(1,1,1)}",
        "{'descr':'<f4','fortran_order':False,'shape':(1,1,1),'shape':(1,1,1)}",
    ] {
        assert_error(
            directory.lut(text, &[0; 8]),
            ReadErrorKind::Header(HeaderError::Fields),
        );
    }
    for text in [
        "<f4 (1,1,1)",
        "{'descr':'<f4', 'fortran_order':False, 'shape':(1,1,1)}junk",
        "{'descr':'<f4' 'fortran_order':False,'shape':(1,1,1)}",
        "{'descr':'<f4','fortran_order':false,'shape':(1,1,1)}",
        "{'descr':'<f4\\n','fortran_order':False,'shape':(1,1,1)}",
        "{'descr':'<f4','fortran_order':False,'shape':(1,1,1)",
        "é",
    ] {
        assert_error(
            directory.lut(text, &[0; 8]),
            ReadErrorKind::Header(HeaderError::Syntax),
        );
    }
}

#[test]
fn rejects_dtype_endian_and_layout() {
    let directory = Directory::new();
    for dtype in [">f4", "=f4", "f4", ""] {
        assert_error(
            directory.lut(&header(dtype, "(1,1,1)"), &[0; 8]),
            ReadErrorKind::ByteOrder,
        );
    }
    for dtype in ["<i4", "<f16", "<f4junk", "<", "|u2"] {
        assert_error(
            directory.lut(&header(dtype, "(1,1,1)"), &[0; 8]),
            ReadErrorKind::Dtype,
        );
    }
    assert_error(
        directory.lut(&header("<f4", "(1,1,1)").replace("False", "True"), &[0; 4]),
        ReadErrorKind::FortranOrder,
    );
    assert_error(
        directory.lut(
            "{'foo':'<f4','fortran_order':False,'shape':(1,1,1),'descr':'>i4'}",
            &[0; 8],
        ),
        ReadErrorKind::Header(HeaderError::Fields),
    );
}

#[test]
fn rejects_shapes() {
    let directory = Directory::new();
    for shape in [
        "(0,0,1)",
        "(-1,1,1)",
        "(1,2,1)",
        "(1,1,0)",
        "(1,1)",
        "(1,1,1,1)",
        "(1x,1,1)",
        "(1,,1,1)",
        "(1 1 1)",
        "()",
        "(2147483648,2147483648,1)",
    ] {
        assert_error(
            directory.lut(&header("<f4", shape), &[0; 8]),
            ReadErrorKind::Shape,
        );
    }
    for shape in ["(80,3)", "(81,4)", "(1,243)", "(3,3,27)"] {
        assert_error(
            load_mallett_basis(directory.write(npy([1, 0], &header("<f4", shape), &[]))),
            ReadErrorKind::Shape,
        );
    }
}

#[test]
fn rejects_size_overflow() {
    let directory = Directory::new();
    for (dtype, shape) in [
        ("<f4", "(4194304,4194304,4194304)"),
        ("<f8", "(2147483647,2147483647,1)"),
        ("<f4", "(18446744073709551616,1,1)"),
    ] {
        assert_error(
            directory.lut(&header(dtype, shape), &[]),
            ReadErrorKind::Size,
        );
    }
    assert_error(
        directory.lut(&header("<f2", "(2147483647,2147483647,1)"), &[]),
        ReadErrorKind::Capacity,
    );
}

#[test]
fn rejects_header_lengths() {
    let directory = Directory::new();
    for count in [0u32, 65536, u32::MAX] {
        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend(count.to_le_bytes());
        assert_error(
            load_spectra_lut(directory.write(bytes)),
            ReadErrorKind::Header(HeaderError::Length),
        );
    }
    let mut text = header("<f4", "(1,1,1)");
    text.push_str(&" ".repeat(65_535 - text.len()));
    assert_eq!(text.len(), 65_535);
    assert_eq!(directory.lut(&text, &[0; 4]).unwrap().values(), [0.0]);
}

#[test]
fn rejects_every_truncation() {
    let directory = Directory::new();
    for version in [[1, 0], [2, 0], [3, 0]] {
        for (dtype, width) in [("<f2", 2), ("<f4", 4), ("<f8", 8)] {
            let text = header(dtype, "(2,2,3)");
            let bytes = npy(version, &text, &vec![0; 12 * width]);
            let preamble = if version == [1, 0] { 10 } else { 12 };
            for length in 0..bytes.len() {
                let part = if length < preamble {
                    ReadPart::Preamble
                } else if length < preamble + text.len() {
                    ReadPart::Header
                } else {
                    ReadPart::Payload
                };
                assert_error(
                    load_spectra_lut(directory.write(&bytes[..length])),
                    ReadErrorKind::ShortRead(part),
                );
            }
        }
    }
}

#[test]
fn ignores_trailing_payload() {
    let directory = Directory::new();
    let mut payload = 2.0_f32.to_le_bytes().to_vec();
    payload.extend(b"trailing");
    let array = directory.lut(&header("<f4", "(1,1,1)"), &payload).unwrap();
    assert_eq!(array.values(), [2.0]);
    let payload: Vec<u8> = (0..324).flat_map(|v| (v as f32).to_le_bytes()).collect();
    let array =
        load_mallett_basis(directory.write(npy([1, 0], &header("<f4", "(81,3)"), &payload)))
            .unwrap();
    assert_eq!(array.values().len(), 243);
    assert_eq!(
        identity(array.values().iter().copied()),
        7_430_929_967_373_113_414
    );
}
