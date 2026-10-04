//! Bounded resource decoding, before spectral preparation or asset ownership.
//! Static noise and lazy neutral calibration have focused child modules.
//! Constructors return complete values. Spectral LUTs decode to C-order f32;
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

const LUT_SHAPE: [usize; 3] = [192, 192, 81];
const MALLETT_SHAPE: [usize; 2] = [81, 3];
const LUT_LAYOUT: &str = "NPY 1.0, little-endian f16/f32/f64, C-order (192, 192, 81)";
const MALLETT_LAYOUT: &str = "NPY 1.0, little-endian f16/f32/f64, C-order (81, 3) RGB";
const NPY_CHUNK_BYTES: usize = 8192;

/// Load a 192x192x81 C-order spectral LUT, converting supported floats to f32.
pub fn load_spectra_lut(path: impl AsRef<Path>) -> Result<Vec<f32>, ReadError> {
    let path = path.as_ref().to_path_buf();
    let result = File::open(&path)
        .map_err(|error| ReadErrorKind::Open(error.kind()))
        .and_then(|mut file| read_spectra_lut(&mut file, Vec::try_reserve_exact));
    result.map_err(|kind| ReadError { path, kind })
}

/// Load an 81x3 C-order Mallett basis in wavelength-major RGB order.
pub fn load_mallett_basis(path: impl AsRef<Path>) -> Result<[[f32; 3]; 81], ReadError> {
    let path = path.as_ref().to_path_buf();
    let result = File::open(&path)
        .map_err(|error| ReadErrorKind::Open(error.kind()))
        .and_then(|mut file| read_mallett_basis(&mut file));
    result.map_err(|kind| ReadError { path, kind })
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
) -> Result<(), ReadErrorKind> {
    reader
        .read_exact(bytes)
        .map_err(|error| read_error(part, error))
}

#[derive(Clone, Copy)]
enum NpyDtype {
    F16,
    F32,
    F64,
}

impl NpyDtype {
    fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    // The payload reader supplies one exact-width chunk from this dtype.
    fn decode(self, bytes: &[u8]) -> f32 {
        match self {
            Self::F16 => half_to_float(u16::from_le_bytes([bytes[0], bytes[1]])),
            Self::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Self::F64 => f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as f32,
        }
    }
}

// Only the three supported header fields are parsed; no Python evaluation,
// escaped strings, nested metadata, or arbitrary literal grammar is needed.
struct NpyHeaderParser<'a> {
    remaining: &'a [u8],
    expected: &'static str,
}

impl<'a> NpyHeaderParser<'a> {
    fn whitespace(&mut self) {
        while self.remaining.first().is_some_and(u8::is_ascii_whitespace) {
            self.remaining = &self.remaining[1..];
        }
    }

    fn consume(&mut self, byte: u8) -> bool {
        self.whitespace();
        if self.remaining.first() != Some(&byte) {
            return false;
        }
        self.remaining = &self.remaining[1..];
        true
    }

    fn require(&mut self, byte: u8) -> Result<(), ReadErrorKind> {
        if self.consume(byte) {
            Ok(())
        } else {
            Err(invalid_npy(self.expected, "malformed header syntax"))
        }
    }

    fn string(&mut self) -> Result<&'a [u8], ReadErrorKind> {
        self.whitespace();
        let Some(&quote @ (b'\'' | b'"')) = self.remaining.first() else {
            return Err(invalid_npy(self.expected, "expected quoted header string"));
        };
        self.remaining = &self.remaining[1..];
        let Some(end) = self.remaining.iter().position(|&byte| byte == quote) else {
            return Err(invalid_npy(self.expected, "unterminated header string"));
        };
        let string = &self.remaining[..end];
        if string
            .iter()
            .any(|&byte| byte == b'\\' || !byte.is_ascii() || byte.is_ascii_control())
        {
            return Err(invalid_npy(self.expected, "unsupported header string"));
        }
        self.remaining = &self.remaining[end + 1..];
        Ok(string)
    }

    fn dimension(&mut self) -> Result<usize, ReadErrorKind> {
        self.whitespace();
        let end = self
            .remaining
            .iter()
            .position(|byte| !byte.is_ascii_digit())
            .unwrap_or(self.remaining.len());
        if end == 0 || (end > 1 && self.remaining[0] == b'0') {
            return Err(invalid_npy(
                self.expected,
                "expected decimal shape dimension",
            ));
        }
        let mut dimension = 0usize;
        for &digit in &self.remaining[..end] {
            dimension = dimension
                .checked_mul(10)
                .and_then(|n| n.checked_add(usize::from(digit - b'0')))
                .ok_or(ReadErrorKind::Size)?;
        }
        self.remaining = &self.remaining[end..];
        Ok(dimension)
    }

    fn shape<const N: usize>(&mut self) -> Result<[usize; N], ReadErrorKind> {
        self.require(b'(')?;
        let mut shape = [0; N];
        for (index, dimension) in shape.iter_mut().enumerate() {
            if index != 0 {
                self.require(b',')?;
            }
            *dimension = self.dimension()?;
        }
        self.consume(b',');
        self.require(b')')?;
        Ok(shape)
    }
}

fn read_npy_header<const N: usize>(
    reader: &mut impl Read,
    required_shape: [usize; N],
    expected: &'static str,
) -> Result<(NpyDtype, usize), ReadErrorKind> {
    let mut prefix = [0; 10];
    read_npy_bytes(reader, &mut prefix, ReadPart::Header)?;
    if &prefix[..6] != b"\x93NUMPY" {
        return Err(invalid_npy(expected, "wrong NPY magic"));
    }
    if prefix[6..8] != [1, 0] {
        return Err(invalid_npy(expected, "unsupported NPY version"));
    }
    let header_bytes = usize::from(u16::from_le_bytes([prefix[8], prefix[9]]));
    let mut header = vec![0; header_bytes];
    for chunk in header.chunks_mut(NPY_CHUNK_BYTES) {
        read_npy_bytes(reader, chunk, ReadPart::Header)?;
    }
    if header.last() != Some(&b'\n') {
        return Err(invalid_npy(expected, "header must end with a newline"));
    }
    let mut parser = NpyHeaderParser {
        remaining: &header,
        expected,
    };
    let mut dtype = None;
    let mut c_order = false;
    let mut shape = None;
    parser.require(b'{')?;
    if !parser.consume(b'}') {
        loop {
            let key = parser.string()?;
            parser.require(b':')?;
            match key {
                b"descr" if dtype.is_none() => {
                    dtype = Some(match parser.string()? {
                        b"<f2" => NpyDtype::F16,
                        b"<f4" => NpyDtype::F32,
                        b"<f8" => NpyDtype::F64,
                        _ => return Err(invalid_npy(expected, "unsupported dtype or byte order")),
                    });
                }
                b"fortran_order" if !c_order => {
                    parser.whitespace();
                    let Some(remaining) = parser.remaining.strip_prefix(b"False") else {
                        return Err(invalid_npy(expected, "C-order False required"));
                    };
                    parser.remaining = remaining;
                    c_order = true;
                }
                b"shape" if shape.is_none() => shape = Some(parser.shape::<N>()?),
                b"descr" | b"fortran_order" | b"shape" => {
                    return Err(invalid_npy(expected, "duplicate header field"));
                }
                _ => return Err(invalid_npy(expected, "unsupported header field")),
            }
            if parser.consume(b'}') {
                break;
            }
            parser.require(b',')?;
            if parser.consume(b'}') {
                break;
            }
        }
    }
    parser.whitespace();
    if !parser.remaining.is_empty() {
        return Err(invalid_npy(expected, "extra header syntax"));
    }
    let (Some(dtype), true, Some(shape)) = (dtype, c_order, shape) else {
        return Err(invalid_npy(expected, "missing required header field"));
    };
    let samples = shape.into_iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(dimension).ok_or(ReadErrorKind::Size)
    })?;
    samples
        .checked_mul(dtype.byte_width())
        .ok_or(ReadErrorKind::Size)?;
    samples
        .checked_mul(size_of::<f32>())
        .ok_or(ReadErrorKind::Size)?;
    if shape != required_shape {
        return Err(invalid_npy(expected, "incompatible shape"));
    }
    Ok((dtype, samples))
}

fn require_npy_eof(reader: &mut impl Read, expected: &'static str) -> Result<(), ReadErrorKind> {
    let mut byte = [0];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(invalid_npy(expected, "extra payload")),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(read_error(ReadPart::Payload, error)),
        }
    }
}

fn read_npy_samples(
    reader: &mut impl Read,
    dtype: NpyDtype,
    samples: &mut [f32],
) -> Result<(), ReadErrorKind> {
    let mut buffer = [0; NPY_CHUNK_BYTES];
    let width = dtype.byte_width();
    for chunk in samples.chunks_mut(buffer.len() / width) {
        let bytes = &mut buffer[..chunk.len() * width];
        read_npy_bytes(reader, bytes, ReadPart::Payload)?;
        for (sample, bytes) in chunk.iter_mut().zip(bytes.chunks_exact(width)) {
            *sample = dtype.decode(bytes);
        }
    }
    Ok(())
}

// The private reservation seam exercises the one large final allocation.
fn read_spectra_lut(
    reader: &mut impl Read,
    reserve: impl FnOnce(&mut Vec<f32>, usize) -> Result<(), TryReserveError>,
) -> Result<Vec<f32>, ReadErrorKind> {
    let (dtype, count) = read_npy_header(reader, LUT_SHAPE, LUT_LAYOUT)?;
    let mut samples = Vec::new();
    reserve(&mut samples, count).map_err(|_| ReadErrorKind::Capacity)?;
    samples.resize(count, 0.0);
    read_npy_samples(reader, dtype, &mut samples)?;
    require_npy_eof(reader, LUT_LAYOUT)?;
    Ok(samples)
}

fn read_mallett_basis(reader: &mut impl Read) -> Result<[[f32; 3]; 81], ReadErrorKind> {
    let (dtype, _) = read_npy_header(reader, MALLETT_SHAPE, MALLETT_LAYOUT)?;
    let mut basis = [[0.0; 3]; 81];
    read_npy_samples(reader, dtype, basis.as_flattened_mut())?;
    require_npy_eof(reader, MALLETT_LAYOUT)?;
    Ok(basis)
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
    fn cmf_source_capacity_receipt() {
        let rows = load_csv_triplets(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources/cie1931_2deg.csv"),
        )
        .unwrap();
        assert_eq!(rows.rows.len(), 81);
        println!(
            "source cmf: length={} capacity={} requested_row_bytes={} inline_payload_bytes={}",
            rows.rows.len(),
            rows.rows.capacity(),
            rows.rows.capacity() * size_of::<[f32; 4]>(),
            size_of::<CsvTriplets>()
        );
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
            (0xc000, 0xc000_0000),
            (0x7c00, 0x7f80_0000),
            (0xfc00, 0xff80_0000),
            (0x7e23, 0x7fc4_6000),
            (0xfe35, 0xffc6_a000),
        ] {
            assert_eq!(half_to_float(half).to_bits(), bits);
        }
    }

    fn npy_header(header: &[u8]) -> Vec<u8> {
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend(u16::try_from(header.len()).unwrap().to_le_bytes());
        bytes.extend(header);
        bytes
    }

    #[test]
    fn reservation_failure() {
        let bytes =
            npy_header(b"{'shape': (192,192,81), 'descr': '<f8', 'fortran_order': False}\n");
        let mut reader = Cursor::new(&bytes);
        let error = read_spectra_lut(&mut reader, |samples, count| {
            assert!(samples.is_empty());
            assert_eq!(count, 192 * 192 * 81);
            samples.try_reserve_exact(usize::MAX)
        })
        .unwrap_err();
        assert_eq!(error, ReadErrorKind::Capacity);
        assert_eq!(reader.position(), bytes.len() as u64);
    }

    #[test]
    fn representation_precedes_reservation() {
        for header in [
            "{'descr': '<f4', 'shape': (81,3), 'fortran_order': False}\n".to_owned(),
            format!(
                "{{'descr': '<f8', 'shape': ({},192,81), 'fortran_order': False}}\n",
                usize::MAX
            ),
        ] {
            let error =
                read_spectra_lut(&mut Cursor::new(npy_header(header.as_bytes())), |_, _| {
                    panic!("invalid representation must not reserve")
                })
                .unwrap_err();
            assert!(matches!(
                error,
                ReadErrorKind::InvalidNpy { .. } | ReadErrorKind::Size
            ));
        }
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
        let bytes = bundled_lut();
        let length = bytes.len() as u64;
        let mut reader = ChunkReader {
            input: Cursor::new(bytes),
            max_read: 0,
            fail_after: None,
        };
        let samples = read_spectra_lut(&mut reader, Vec::try_reserve_exact).unwrap();
        assert_eq!(samples.len(), 192 * 192 * 81);
        assert_eq!(reader.max_read, NPY_CHUNK_BYTES);
        assert_eq!(reader.input.position(), length);

        for dtype in ["<f2", "<f4", "<f8"] {
            let mut header =
                format!("{{'descr': '{dtype}', 'shape': (81,3), 'fortran_order': False}}")
                    .into_bytes();
            header.resize(usize::from(u16::MAX) - 1, b' ');
            header.push(b'\n');
            let mut bytes = npy_header(&header);
            let width = match dtype {
                "<f2" => 2,
                "<f4" => 4,
                _ => 8,
            };
            bytes.resize(bytes.len() + 81 * 3 * width, 0);
            let length = bytes.len() as u64;
            let mut reader = ChunkReader {
                input: Cursor::new(bytes),
                max_read: 0,
                fail_after: None,
            };
            assert_eq!(read_mallett_basis(&mut reader).unwrap(), [[0.0; 3]; 81]);
            assert_eq!(reader.max_read, NPY_CHUNK_BYTES);
            assert_eq!(reader.input.position(), length);
        }
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
            (None, ReadErrorKind::ShortRead(ReadPart::Payload)),
        ] {
            let mut reader = ChunkReader {
                input: Cursor::new(bytes.clone()),
                max_read: 0,
                fail_after,
            };
            let error = read_spectra_lut(&mut reader, Vec::try_reserve_exact).unwrap_err();
            assert_eq!(error, expected);
            assert_eq!(reader.input.position(), 9000);
        }
    }

    #[test]
    fn basis_read_failures() {
        let prefix = npy_header(b"{'descr': '<f8', 'fortran_order': False, 'shape': (81,3)}\n");
        let payload_end = prefix.len() + 81 * 3 * 8;
        for (end, part) in [
            (0, ReadPart::Header),
            (15, ReadPart::Header),
            (prefix.len() + 30, ReadPart::Payload),
            (payload_end, ReadPart::Payload),
        ] {
            let mut bytes = prefix.clone();
            bytes.resize(payload_end, 0);
            bytes.truncate(end);
            for fail_after in [None, Some(end as u64)] {
                let mut reader = ChunkReader {
                    input: Cursor::new(bytes.clone()),
                    max_read: 0,
                    fail_after,
                };
                let result = read_mallett_basis(&mut reader);
                if fail_after.is_some() {
                    assert_eq!(
                        result.unwrap_err(),
                        ReadErrorKind::Io {
                            part,
                            kind: io::ErrorKind::PermissionDenied
                        }
                    );
                } else if end < payload_end {
                    assert_eq!(result.unwrap_err(), ReadErrorKind::ShortRead(part));
                } else {
                    assert_eq!(result.unwrap(), [[0.0; 3]; 81]);
                }
            }
        }
    }

    #[test]
    fn lut_eof_failure_does_not_return_samples() {
        let bytes = bundled_lut();
        let end = bytes.len() as u64;
        let mut reader = ChunkReader {
            input: Cursor::new(bytes),
            max_read: 0,
            fail_after: Some(end),
        };
        assert_eq!(
            read_spectra_lut(&mut reader, Vec::try_reserve_exact).unwrap_err(),
            ReadErrorKind::Io {
                part: ReadPart::Payload,
                kind: io::ErrorKind::PermissionDenied
            }
        );
        assert_eq!(reader.input.position(), end);
    }

    #[test]
    fn trailing_payload_does_not_return_samples() {
        let mut bytes = bundled_lut();
        bytes.push(0);
        let error = read_spectra_lut(&mut Cursor::new(bytes), Vec::try_reserve_exact).unwrap_err();
        assert_eq!(error, invalid_npy(LUT_LAYOUT, "extra payload"));
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
