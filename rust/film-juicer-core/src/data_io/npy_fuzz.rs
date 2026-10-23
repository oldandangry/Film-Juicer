//! Bounded structural mutations; authored sample bits are trusted content.

#![forbid(unsafe_code)]

use super::{ReadErrorKind, ReadPart, read_mallett_basis};
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};

const SEEDS: [u64; 4] = [0x4e50_5901, 0x81_03, 0xdead_beef, 0x5eed_cafe];
const CASES: usize = 512;
const SAMPLE_BITS: [u32; 12] = [
    0,
    0x8000_0000,
    1,
    0x007f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0xc000_0000,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc4_6000,
    0xffc6_a000,
];

fn next(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state & 0xffff_ffff) as usize
}

fn npy(header: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend(u16::try_from(header.len()).unwrap().to_le_bytes());
    bytes.extend(header);
    bytes.extend(payload);
    bytes
}

fn source() -> Vec<u8> {
    let payload: Vec<_> = (0..243)
        .flat_map(|index| SAMPLE_BITS[index % SAMPLE_BITS.len()].to_le_bytes())
        .collect();
    npy(
        b"{'descr': '<f4', 'fortran_order': False, 'shape': (81,3)}\n",
        &payload,
    )
}

struct Fragmented<'a> {
    input: Cursor<&'a [u8]>,
    width: usize,
}

impl Read for Fragmented<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let length = bytes.len().min(self.width);
        self.input.read(&mut bytes[..length])
    }
}

fn bits(basis: [[f32; 3]; 81]) -> Vec<u32> {
    basis.into_iter().flatten().map(f32::to_bits).collect()
}

fn check(bytes: &mut [u8], case: usize, expected: Option<Result<Vec<u32>, ReadErrorKind>>) -> bool {
    let mut basis = read_mallett_basis(&mut Cursor::new(&*bytes));
    if case == 0 && std::env::var("JUICER_NPY_FUZZ_CONTROL").as_deref() == Ok("nonfinite-clamp") {
        // Qualification deliberately simulates forbidden scientific-content repair.
        for sample in basis.as_mut().unwrap().as_flattened_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }
    let observed = basis.map(bits);
    let accepted = observed.is_ok();
    if let Some(expected) = expected {
        assert_eq!(observed, expected, "independent structural/sample oracle");
    }
    let fragmented = read_mallett_basis(&mut Fragmented {
        input: Cursor::new(&*bytes),
        width: 1 + case % 17,
    })
    .map(bits);
    assert_eq!(observed, fragmented, "fragmentation changed decode");
    if let Err(error) = observed {
        assert!(matches!(
            error,
            ReadErrorKind::InvalidNpy { .. }
                | ReadErrorKind::Size
                | ReadErrorKind::ShortRead(ReadPart::Header | ReadPart::Payload)
        ));
    }
    // The result owns its storage; source mutation and later failures cannot alter it.
    let retained = basis.ok();
    bytes.fill(0xa5);
    assert!(read_mallett_basis(&mut Cursor::new(&*bytes)).is_err());
    if let Some(retained) = retained {
        assert_eq!(bits(retained), observed.unwrap());
    }
    accepted
}

fn failure_root() -> PathBuf {
    std::env::var_os("JUICER_TEST_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/validation/rust-core")
        })
}

fn retain_failure(root: &Path, seed: u64, case: usize, bytes: &[u8]) -> io::Result<PathBuf> {
    retain_failure_with(root, seed, case, bytes, |file, bytes| file.write_all(bytes))
}

fn retain_failure_with(
    root: &Path,
    seed: u64,
    case: usize,
    bytes: &[u8],
    mut write: impl FnMut(&mut std::fs::File, &[u8]) -> io::Result<()>,
) -> io::Result<PathBuf> {
    std::fs::create_dir_all(root)?;
    let mut attempt = 0_u64;
    loop {
        let suffix = if attempt == 0 {
            String::new()
        } else {
            format!("-attempt-{attempt}")
        };
        let path = root.join(format!("npy-{seed:016x}-{case:04}{suffix}.npy"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                if let Err(error) = write(&mut file, bytes) {
                    drop(file);
                    let cleanup = std::fs::remove_file(&path);
                    return Err(io::Error::new(
                        error.kind(),
                        format!(
                            "writing {}: {error}; incomplete-file removal: {cleanup:?}",
                            path.display()
                        ),
                    ));
                }
                return Ok(path);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                if std::fs::read(&path)? == bytes {
                    return Ok(path);
                }
                attempt = attempt
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("NPY attempt names exhausted"))?;
            }
            Err(error) => return Err(error),
        }
    }
}

fn resume_failure(
    root: &Path,
    seed: u64,
    case: usize,
    bytes: &[u8],
    panic: Box<dyn std::any::Any + Send>,
    diagnostic: &mut impl Write,
) -> ! {
    match retain_failure(root, seed, case, bytes) {
        Ok(path) => {
            let _ = writeln!(
                diagnostic,
                "NPY failure seed={seed:016x} case={case} bytes={} artifact={}",
                bytes.len(),
                path.display()
            );
        }
        Err(error) => {
            let _ = writeln!(
                diagnostic,
                "NPY failure seed={seed:016x} case={case} bytes={} sample could not be saved under {}: {error}",
                bytes.len(),
                root.display()
            );
        }
    }
    std::panic::resume_unwind(panic)
}

struct FailureDirectory(PathBuf);

impl FailureDirectory {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = failure_root();
        std::fs::create_dir_all(&parent).unwrap();
        loop {
            let path = parent.join(format!(
                "npy-retention-control-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("NPY retention control setup: {error}"),
            }
        }
    }
}

impl Drop for FailureDirectory {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[test]
fn failure_first_save_and_identical_repeat() {
    let directory = FailureDirectory::new();
    let bytes = b"tiny NPY failure control";
    let first = retain_failure(&directory.0, 0x81, 3, bytes).unwrap();
    assert_eq!(first.file_name().unwrap(), "npy-0000000000000081-0003.npy");
    assert_eq!(std::fs::read(&first).unwrap(), bytes);
    let repeated = retain_failure(&directory.0, 0x81, 3, bytes).unwrap();
    assert_eq!(repeated, first);
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 1);
}

#[test]
fn failure_different_sample_preserves_occupied_names() {
    let directory = FailureDirectory::new();
    let first = retain_failure(&directory.0, 1, 2, b"first").unwrap();
    let second = retain_failure(&directory.0, 1, 2, b"other").unwrap();
    assert_ne!(first, second);
    assert_eq!(std::fs::read(&first).unwrap(), b"first");
    assert_eq!(std::fs::read(&second).unwrap(), b"other");
    assert_eq!(
        retain_failure(&directory.0, 1, 2, b"other").unwrap(),
        second
    );
}

#[test]
fn failure_partial_write_preserves_existing_sample_and_allows_retry() {
    let directory = FailureDirectory::new();
    let first = retain_failure(&directory.0, 1, 2, b"first").unwrap();
    let error = retain_failure_with(&directory.0, 1, 2, b"other", |file, bytes| {
        file.write_all(&bytes[..2])?;
        Err(io::Error::other("partial-write control"))
    })
    .unwrap_err();
    assert!(error.to_string().contains("partial-write control"));
    assert_eq!(std::fs::read(&first).unwrap(), b"first");
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 1);
    let second = retain_failure(&directory.0, 1, 2, b"other").unwrap();
    assert_eq!(std::fs::read(second).unwrap(), b"other");
}

#[test]
fn failure_retention_resumes_original_payload_on_save_error() {
    let directory = FailureDirectory::new();
    for writable in [true, false] {
        let root = directory
            .0
            .join(if writable { "directory" } else { "file" });
        if !writable {
            std::fs::write(&root, b"occupied regular file").unwrap();
        }
        let original = Box::new(String::from("original sample assertion"));
        let identity = std::ptr::from_ref(original.as_ref());
        let mut diagnostic = Vec::new();
        let resumed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            resume_failure(&root, 0x81, 3, b"sample", original, &mut diagnostic);
        }))
        .unwrap_err()
        .downcast::<String>()
        .unwrap();
        assert!(std::ptr::eq(identity, std::ptr::from_ref(resumed.as_ref())));
        assert_eq!(*resumed, "original sample assertion");
        let diagnostic = String::from_utf8(diagnostic).unwrap();
        assert!(diagnostic.contains("seed=0000000000000081 case=3 bytes=6"));
        if writable {
            assert_eq!(
                std::fs::read(root.join("npy-0000000000000081-0003.npy")).unwrap(),
                b"sample"
            );
            assert!(diagnostic.contains("artifact="));
        } else {
            assert!(diagnostic.contains("sample could not be saved"));
            assert_eq!(std::fs::read(&root).unwrap(), b"occupied regular file");
        }
    }
}

#[test]
fn failure_diagnostic_write_error_preserves_original_payload() {
    struct Unwritable;
    impl Write for Unwritable {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("diagnostic write control"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let directory = FailureDirectory::new();
    let original = Box::new(String::from("original decode failure"));
    let identity = std::ptr::from_ref(original.as_ref());
    let resumed = std::panic::catch_unwind(|| {
        resume_failure(&directory.0, 1, 0, b"sample", original, &mut Unwritable);
    })
    .unwrap_err()
    .downcast::<String>()
    .unwrap();
    assert!(std::ptr::eq(identity, std::ptr::from_ref(resumed.as_ref())));
}

#[test]
fn structural_mutations() {
    let source = source();
    let payload = 10 + usize::from(u16::from_le_bytes([source[8], source[9]]));
    let expected: Vec<_> = (0..243)
        .map(|index| SAMPLE_BITS[index % SAMPLE_BITS.len()])
        .collect();
    let mut accepted = 0;
    let mut rejected = 0;
    for seed in SEEDS {
        let mut state = seed;
        for case in 0..CASES {
            let mut bytes = source.clone();
            let mut oracle = None;
            match case % 8 {
                0 => oracle = Some(Ok(expected.clone())),
                1 => {
                    let end = next(&mut state) % bytes.len();
                    bytes.truncate(end);
                    oracle = Some(Err(ReadErrorKind::ShortRead(if end < payload {
                        ReadPart::Header
                    } else {
                        ReadPart::Payload
                    })));
                }
                2 => {
                    bytes.push(next(&mut state) as u8);
                    oracle = Some(Err(super::invalid_npy(
                        super::MALLETT_LAYOUT,
                        "extra payload",
                    )));
                }
                3 => {
                    let offset = next(&mut state) % bytes.len();
                    bytes[offset] ^= 1 << (next(&mut state) % 8);
                }
                4 => {
                    let offset = next(&mut state) % payload;
                    bytes.remove(offset);
                }
                5 => {
                    let offset = next(&mut state) % payload;
                    bytes.insert(offset, next(&mut state) as u8);
                }
                6 => {
                    let length = (next(&mut state) as u16).to_le_bytes();
                    bytes[8..10].copy_from_slice(&length);
                }
                _ => {
                    let offset = payload + next(&mut state) % (bytes.len() - payload);
                    bytes[offset] = next(&mut state) as u8;
                    oracle = Some(Ok(bytes[payload..]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| u32::from_le_bytes(*chunk))
                        .collect()));
                }
            }
            let original = bytes.clone();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check(&mut bytes, case, oracle)
            })) {
                Ok(true) => accepted += 1,
                Ok(false) => rejected += 1,
                Err(panic) => {
                    resume_failure(
                        &failure_root(),
                        seed,
                        case,
                        &original,
                        panic,
                        &mut io::stderr().lock(),
                    );
                }
            }
        }
    }
    assert!(accepted >= 512 && rejected >= 512);
    println!(
        "NPY corpus seeds={SEEDS:x?} cases={} accepted={accepted} rejected={rejected}",
        SEEDS.len() * CASES
    );
}

#[test]
fn representations_and_shape_overflow() {
    // Independent expected conversions, including nonfinite samples.
    for (dtype, payload, expected) in [
        ("<f2", vec![0x23, 0x7e], 0x7fc4_6000),
        ("<f4", 0xff80_0000u32.to_le_bytes().to_vec(), 0xff80_0000),
        ("<f8", f64::MAX.to_le_bytes().to_vec(), 0x7f80_0000),
    ] {
        for header in [
            format!("{{'descr':'{dtype}', 'fortran_order':False, 'shape':(81,3)}}\n"),
            format!("{{\"shape\": (81, 3,), \"fortran_order\": False, \"descr\": \"{dtype}\",}}\n"),
        ] {
            let mut bytes = npy(header.as_bytes(), &payload.repeat(243));
            check(&mut bytes, 1, Some(Ok(vec![expected; 243])));
        }
    }
    for shape in ["(184467440737095516160,3)", "(18446744073709551615,3)"] {
        let header = format!("{{'descr':'<f8','fortran_order':False,'shape':{shape}}}\n");
        let mut bytes = npy(header.as_bytes(), &[]);
        check(&mut bytes, 1, Some(Err(ReadErrorKind::Size)));
    }
}
