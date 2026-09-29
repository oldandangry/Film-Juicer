//! Bounded resource decoding, before spectral preparation or asset ownership.
//! Static noise and lazy neutral calibration have focused child modules.
//! Constructors return complete values. Bundled spectral LUTs are C-order f32;
//! the Mallett basis has 81 wavelength rows and three RGB columns.
//! Explicit sample and noise buffers reserve fallibly. Incidental path and parser
//! allocations can still abort; this module does not provide general OOM recovery.

pub mod noise;
pub mod calibration;

use std::collections::TryReserveError;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

const MAX_CSV_LINE_BYTES: usize = 65_535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPart {
    Header,
    Payload,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadErrorKind {
    Open(io::ErrorKind),
    Io {
        part: ReadPart,
        kind: io::ErrorKind,
    },
    ShortRead(ReadPart),
    InvalidNpy {
        expected: &'static str,
        reason: &'static str,
    },
    Size,
    Capacity,
    CsvLineTooLong {
        line: usize,
    },
}

/// Path storage is acquired before construction, so a capacity error needs no
/// new string allocation. The failure kind contains only bounded typed facts.
#[derive(Debug)]
pub struct ReadError {
    pub path: PathBuf,
    pub kind: ReadErrorKind,
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:?}", self.path.display(), self.kind)
    }
}

impl std::error::Error for ReadError {}

/// Source wavelength/value pairs in file order. No axis or normalization policy.
#[derive(Debug)]
pub struct CsvPairs {
    rows: Vec<[f32; 2]>,
}

impl CsvPairs {
    pub fn rows(&self) -> &[[f32; 2]] {
        &self.rows
    }
}

/// Rows are [wavelength, x_bar, y_bar, z_bar], in file order.
#[derive(Debug)]
pub struct CsvTriplets {
    rows: Vec<[f32; 4]>,
}

impl CsvTriplets {
    pub fn rows(&self) -> &[[f32; 4]] {
        &self.rows
    }
}

pub fn load_csv_pairs(path: impl AsRef<Path>) -> Result<CsvPairs, ReadError> {
    let path = path.as_ref().to_path_buf();
    let result = File::open(&path)
        .map_err(|error| ReadErrorKind::Open(error.kind()))
        .and_then(|file| read_csv(BufReader::new(file)))
        .map(|rows| CsvPairs { rows });
    result.map_err(|kind| ReadError { path, kind })
}

pub fn load_csv_triplets(path: impl AsRef<Path>) -> Result<CsvTriplets, ReadError> {
    let path = path.as_ref().to_path_buf();
    let result = File::open(&path)
        .map_err(|error| ReadErrorKind::Open(error.kind()))
        .and_then(|file| read_csv(BufReader::new(file)))
        .map(|rows| CsvTriplets { rows });
    result.map_err(|kind| ReadError { path, kind })
}

fn read_csv<const N: usize>(mut reader: impl BufRead) -> Result<Vec<[f32; N]>, ReadErrorKind> {
    let mut rows = Vec::new();
    let mut line = Vec::new();
    let mut line_number = 0usize;
    loop {
        line.clear();
        let count = reader
            .by_ref()
            .take((MAX_CSV_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
            .map_err(|error| read_error(ReadPart::Csv, error))?;
        if count == 0 {
            return Ok(rows);
        }
        line_number = line_number.checked_add(1).ok_or(ReadErrorKind::Size)?;
        if count > MAX_CSV_LINE_BYTES {
            return Err(ReadErrorKind::CsvLineTooLong { line: line_number });
        }
        let end = line
            .iter()
            .position(|c| matches!(c, b'#' | b';'))
            .unwrap_or(line.len());
        let mut remaining = &line[..end];
        let mut row = [0.0; N];
        let mut complete = true;
        for sample in &mut row {
            if let Some(value) = csv_number(&mut remaining) {
                *sample = value;
                while remaining.first() == Some(&b',') {
                    remaining = &remaining[1..];
                }
            } else {
                complete = false;
                break;
            }
        }
        if complete {
            rows.len()
                .checked_add(1)
                .and_then(|n| n.checked_mul(size_of::<[f32; N]>()))
                .ok_or(ReadErrorKind::Size)?;
            rows.try_reserve(1).map_err(|_| ReadErrorKind::Capacity)?;
            rows.push(row);
        }
    }
}

// This is decimal stream extraction, not token splitting: the final number may
// have ignored trailing text, but a consumed exponent must be complete. Convert
// directly to f32; an f64 intermediate changes rounding just above midpoints.
fn csv_number(input: &mut &[u8]) -> Option<f32> {
    let mut bytes = *input;
    while matches!(
        bytes.first(),
        Some(b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
    ) {
        bytes = &bytes[1..];
    }
    let mut end = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    if bytes
        .get(end..end + 2)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"0x"))
    {
        return None;
    }
    let start = end;
    while bytes.get(end).is_some_and(u8::is_ascii_digit) {
        end += 1;
    }
    let mut digits = end - start;
    if bytes.get(end) == Some(&b'.') {
        end += 1;
        let start = end;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        digits += end - start;
    }
    if digits == 0 {
        return None;
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        end += 1;
        if matches!(bytes.get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        let start = end;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        if start == end {
            return None;
        }
    }
    let value = std::str::from_utf8(&bytes[..end])
        .ok()?
        .parse::<f32>()
        .ok()?;
    if !value.is_finite() {
        return None;
    }
    *input = &bytes[end..];
    Some(value)
}

// These are the two exact installed NPY 1.0 prefixes, including padding.
const LUT_PREFIX: &[u8; 128] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f2', 'fortran_order': False, 'shape': (192, 192, 81), }                                                  \n";
const MALLETT_PREFIX: &[u8; 128] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f4', 'fortran_order': False, 'shape': (81, 3), }                                                         \n";
const LUT_SAMPLES: usize = 192 * 192 * 81;
const LUT_FILE_BYTES: u64 = (LUT_PREFIX.len() + LUT_SAMPLES * 2) as u64;
const MALLETT_FILE_BYTES: u64 = (MALLETT_PREFIX.len() + 81 * 3 * 4) as u64;
const MALLETT_ASSET: &str = "mallett2019_basis.npy";
const FINGERPRINT_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// The two bundled LUTs share a layout but have distinct decoded contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectraLutAsset {
    Hanatos,
    Arctic,
}

impl SpectraLutAsset {
    fn name(self) -> &'static str {
        match self {
            Self::Hanatos => "irradiance_xy_tc.npy",
            Self::Arctic => "arctic2026beta04_reflectance_xy_tc.npy",
        }
    }

    fn fingerprint(self) -> u64 {
        match self {
            Self::Hanatos => 0x81eb_efd4_e4cc_9926,
            Self::Arctic => 0x9262_ffb7_65e3_289e,
        }
    }
}

/// Load the selected bundled 192x192x81 f16 LUT as C-order f32 samples.
pub fn load_spectra_lut(
    path: impl AsRef<Path>,
    asset: SpectraLutAsset,
) -> Result<Vec<f32>, ReadError> {
    let path = path.as_ref().to_path_buf();
    let result = (|| {
        let (mut file, length) = open_npy(&path)?;
        read_spectra_lut(&mut file, length, asset, Vec::try_reserve_exact)
    })();
    result.map_err(|kind| ReadError { path, kind })
}

/// Load the bundled f32 basis in wavelength-major RGB order.
pub fn load_mallett_basis(path: impl AsRef<Path>) -> Result<[[f32; 3]; 81], ReadError> {
    let path = path.as_ref().to_path_buf();
    let result = (|| {
        let (mut file, length) = open_npy(&path)?;
        read_npy_prefix(
            &mut file,
            length,
            MALLETT_FILE_BYTES,
            MALLETT_PREFIX,
            MALLETT_ASSET,
        )?;
        let mut basis = [[0.0; 3]; 81];
        let mut fingerprint = FINGERPRINT_OFFSET;
        for row in &mut basis {
            let mut bytes = [0; 12];
            read_npy_bytes(&mut file, &mut bytes, ReadPart::Payload, MALLETT_ASSET)?;
            for (sample, bytes) in row.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *sample = f32::from_le_bytes(*bytes);
                if !sample.is_finite() {
                    return Err(invalid_npy(MALLETT_ASSET, "nonfinite sample"));
                }
                fingerprint = fingerprint_sample(fingerprint, *sample);
            }
        }
        if fingerprint != 0xb2ce_9984_05c5_5a48 {
            return Err(invalid_npy(MALLETT_ASSET, "decoded fingerprint mismatch"));
        }
        require_npy_eof(&mut file, MALLETT_ASSET)?;
        Ok(basis)
    })();
    result.map_err(|kind| ReadError { path, kind })
}

fn open_npy(path: &Path) -> Result<(File, u64), ReadErrorKind> {
    let file = File::open(path).map_err(|error| ReadErrorKind::Open(error.kind()))?;
    let length = file
        .metadata()
        .map_err(|error| read_error(ReadPart::Header, error))?
        .len();
    Ok((file, length))
}

fn invalid_npy(expected: &'static str, reason: &'static str) -> ReadErrorKind {
    ReadErrorKind::InvalidNpy { expected, reason }
}

fn read_error(part: ReadPart, error: io::Error) -> ReadErrorKind {
    if error.kind() == io::ErrorKind::UnexpectedEof {
        ReadErrorKind::ShortRead(part)
    } else {
        ReadErrorKind::Io {
            part,
            kind: error.kind(),
        }
    }
}

fn read_npy_bytes(
    reader: &mut impl Read,
    bytes: &mut [u8],
    part: ReadPart,
    asset: &'static str,
) -> Result<(), ReadErrorKind> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            invalid_npy(asset, "short read")
        } else {
            read_error(part, error)
        }
    })
}

fn read_npy_prefix(
    reader: &mut impl Read,
    file_bytes: u64,
    expected_bytes: u64,
    expected_prefix: &[u8; 128],
    asset: &'static str,
) -> Result<(), ReadErrorKind> {
    if file_bytes != expected_bytes {
        return Err(invalid_npy(asset, "wrong file length"));
    }
    let mut prefix = [0; 128];
    read_npy_bytes(reader, &mut prefix, ReadPart::Header, asset)?;
    if &prefix != expected_prefix {
        return Err(invalid_npy(asset, "wrong bundled NPY prefix"));
    }
    Ok(())
}

fn require_npy_eof(reader: &mut impl Read, asset: &'static str) -> Result<(), ReadErrorKind> {
    let mut byte = [0];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(invalid_npy(asset, "extra payload")),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(read_error(ReadPart::Payload, error)),
        }
    }
}

// FNV-1a over decoded little-endian f32 bits, matching the independent bundled
// expectations. This detects installed-content corruption, not authenticity.
fn fingerprint_sample(hash: u64, sample: f32) -> u64 {
    sample.to_le_bytes().into_iter().fold(hash, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

// The private reservation seam exercises the one large final allocation.
fn read_spectra_lut(
    reader: &mut impl Read,
    file_bytes: u64,
    asset: SpectraLutAsset,
    reserve: impl FnOnce(&mut Vec<f32>, usize) -> Result<(), TryReserveError>,
) -> Result<Vec<f32>, ReadErrorKind> {
    read_npy_prefix(reader, file_bytes, LUT_FILE_BYTES, LUT_PREFIX, asset.name())?;
    let mut samples = Vec::new();
    reserve(&mut samples, LUT_SAMPLES).map_err(|_| ReadErrorKind::Capacity)?;
    let mut buffer = [0; 8192];
    let mut fingerprint = FINGERPRINT_OFFSET;
    while samples.len() < LUT_SAMPLES {
        let count = (LUT_SAMPLES - samples.len()).min(buffer.len() / 2);
        let bytes = &mut buffer[..count * 2];
        read_npy_bytes(reader, bytes, ReadPart::Payload, asset.name())?;
        for bytes in bytes.as_chunks::<2>().0 {
            let sample = half_to_float(u16::from_le_bytes(*bytes));
            if !sample.is_finite() {
                return Err(invalid_npy(asset.name(), "nonfinite sample"));
            }
            fingerprint = fingerprint_sample(fingerprint, sample);
            samples.push(sample);
        }
    }
    if fingerprint != asset.fingerprint() {
        return Err(invalid_npy(asset.name(), "decoded fingerprint mismatch"));
    }
    require_npy_eof(reader, asset.name())?;
    Ok(samples)
}

fn half_to_float(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 31;
    let mut fraction = u32::from(bits & 0x3ff);
    let magnitude = match exponent {
        31 => 0x7f80_0000 | (fraction << 13),
        0 if fraction == 0 => 0,
        0 => {
            let mut shift = 0;
            while fraction & 0x400 == 0 {
                fraction <<= 1;
                shift += 1;
            }
            ((127 - 14 - shift) << 23) | ((fraction & 0x3ff) << 13)
        }
        _ => ((u32::from(exponent) + 127 - 15) << 23) | (fraction << 13),
    };
    f32::from_bits(sign | magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn bundled_lut() -> Vec<u8> {
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../Resources/luts/spectral_upsampling/irradiance_xy_tc.npy"),
        )
        .unwrap()
    }

    #[test]
    fn half_conversion_bits() {
        for (half, bits) in [
            (0, 0),
            (0x8000, 0x8000_0000),
            (1, 0x3380_0000),
            (0x3ff, 0x387f_c000),
            (0x400, 0x3880_0000),
            (0x3c00, 0x3f80_0000),
            (0x7bff, 0x477f_e000),
        ] {
            assert_eq!(half_to_float(half).to_bits(), bits);
        }
    }

    #[test]
    fn reservation_failure() {
        let mut reader = Cursor::new(LUT_PREFIX);
        let error = read_spectra_lut(
            &mut reader,
            LUT_FILE_BYTES,
            SpectraLutAsset::Hanatos,
            |samples, count| {
                assert!(samples.is_empty());
                assert_eq!(count, LUT_SAMPLES);
                samples.try_reserve_exact(usize::MAX)
            },
        )
        .unwrap_err();
        assert_eq!(error, ReadErrorKind::Capacity);
        assert_eq!(reader.position(), 128);
    }

    #[test]
    fn file_length_precedes_reservation() {
        let error = read_spectra_lut(
            &mut io::empty(),
            LUT_FILE_BYTES - 1,
            SpectraLutAsset::Hanatos,
            |_, _| panic!("wrong file length must not reserve"),
        )
        .unwrap_err();
        assert_eq!(
            error,
            invalid_npy(SpectraLutAsset::Hanatos.name(), "wrong file length")
        );
    }

    struct ChunkReader {
        input: Cursor<Vec<u8>>,
        max_read: usize,
        fail_after: Option<u64>,
    }
    impl Read for ChunkReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.max_read = self.max_read.max(buffer.len());
            if self
                .fail_after
                .is_some_and(|limit| self.input.position() >= limit)
            {
                return Err(io::Error::from(io::ErrorKind::PermissionDenied));
            }
            let length = buffer.len().min(3);
            self.input.read(&mut buffer[..length])
        }
    }

    #[test]
    fn bounded_partial_reads() {
        let mut reader = ChunkReader {
            input: Cursor::new(bundled_lut()),
            max_read: 0,
            fail_after: None,
        };
        let samples = read_spectra_lut(
            &mut reader,
            LUT_FILE_BYTES,
            SpectraLutAsset::Hanatos,
            Vec::try_reserve_exact,
        )
        .unwrap();
        assert_eq!(samples.len(), LUT_SAMPLES);
        assert_eq!(reader.max_read, 8192);
        assert_eq!(reader.input.position(), LUT_FILE_BYTES);
    }

    #[test]
    fn late_payload_failures() {
        let mut bytes = bundled_lut();
        bytes.truncate(9000);
        for (fail_after, expected) in [
            (
                Some(9000),
                ReadErrorKind::Io {
                    part: ReadPart::Payload,
                    kind: io::ErrorKind::PermissionDenied,
                },
            ),
            (
                None,
                invalid_npy(SpectraLutAsset::Hanatos.name(), "short read"),
            ),
        ] {
            let mut reader = ChunkReader {
                input: Cursor::new(bytes.clone()),
                max_read: 0,
                fail_after,
            };
            let error = read_spectra_lut(
                &mut reader,
                LUT_FILE_BYTES,
                SpectraLutAsset::Hanatos,
                Vec::try_reserve_exact,
            )
            .unwrap_err();
            assert_eq!(error, expected);
            assert_eq!(reader.input.position(), 9000);
        }
    }

    #[test]
    fn growth_after_length_check() {
        let mut bytes = bundled_lut();
        bytes.push(0);
        let error = read_spectra_lut(
            &mut Cursor::new(bytes),
            LUT_FILE_BYTES,
            SpectraLutAsset::Hanatos,
            Vec::try_reserve_exact,
        )
        .unwrap_err();
        assert_eq!(
            error,
            invalid_npy(SpectraLutAsset::Hanatos.name(), "extra payload")
        );
    }

    #[test]
    fn csv_read_failure() {
        let reader = ChunkReader {
            input: Cursor::new(b"1,2\n3,4\n".to_vec()),
            max_read: 0,
            fail_after: Some(6),
        };
        let result = read_csv::<2>(BufReader::new(reader));
        assert_eq!(
            result.unwrap_err(),
            ReadErrorKind::Io {
                part: ReadPart::Csv,
                kind: io::ErrorKind::PermissionDenied
            }
        );
    }
}
