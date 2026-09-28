//! Raw CSV and reconstruction NPY input, before spectral preparation or asset policy.
//! Constructors return complete immutable values. NPY arrays are C-order f32;
//! Mallett input is normalized to 81 wavelength rows and three basis columns.
//! Explicit sample buffers reserve fallibly. Incidental path, header and parser
//! allocations can still abort; this module does not provide general OOM recovery.

use std::collections::TryReserveError;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

const MAX_HEADER_BYTES: usize = 65_535;
const MAX_CSV_LINE_BYTES: usize = 65_535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPart {
    Preamble,
    Header,
    Payload,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderError {
    Length,
    Syntax,
    Fields,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadErrorKind {
    Open(io::ErrorKind),
    Io { part: ReadPart, kind: io::ErrorKind },
    ShortRead(ReadPart),
    Magic,
    Version([u8; 2]),
    Header(HeaderError),
    Dtype,
    ByteOrder,
    FortranOrder,
    Shape,
    Size,
    Capacity,
    CsvLineTooLong { line: usize },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    F16,
    F32,
    F64,
}

impl NpyDtype {
    fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// The accepted `|` spelling retains the C++ little-endian interpretation even
/// for multi-byte floats. Native (`=`) and big-endian (`>`) sources are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyByteOrder {
    Little,
    NotApplicable,
}

/// A complete C-order reconstruction array; nonfinite samples remain raw input.
/// Only the two loaders below construct this type. No cache or asset identity.
#[derive(Debug)]
pub struct NpyArray {
    version: [u8; 2],
    dtype: NpyDtype,
    byte_order: NpyByteOrder,
    source_shape: Vec<usize>,
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl NpyArray {
    pub fn version(&self) -> [u8; 2] {
        self.version
    }
    pub fn dtype(&self) -> NpyDtype {
        self.dtype
    }
    pub fn byte_order(&self) -> NpyByteOrder {
        self.byte_order
    }
    pub fn source_shape(&self) -> &[usize] {
        &self.source_shape
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Load a positive square N x N x K array. Trailing payload bytes are ignored,
/// matching the accepted C++ reader; resource-specific dimensions belong later.
pub fn load_spectra_lut(path: impl AsRef<Path>) -> Result<NpyArray, ReadError> {
    load_npy(path.as_ref(), ArrayKind::SpectraLut)
}

/// Accept 81x3 or 3x81 input, returning row-major 81x3 without a second buffer.
pub fn load_mallett_basis(path: impl AsRef<Path>) -> Result<NpyArray, ReadError> {
    load_npy(path.as_ref(), ArrayKind::MallettBasis)
}

#[derive(Clone, Copy)]
enum ArrayKind {
    SpectraLut,
    MallettBasis,
}

fn load_npy(path: &Path, kind: ArrayKind) -> Result<NpyArray, ReadError> {
    let path = path.to_path_buf();
    let result = (|| {
        let mut file = File::open(&path).map_err(|error| ReadErrorKind::Open(error.kind()))?;
        let length = file
            .metadata()
            .map_err(|error| read_error(ReadPart::Preamble, error))?
            .len();
        read_npy(&mut file, length, kind, Vec::try_reserve_exact)
    })();
    result.map_err(|kind| ReadError { path, kind })
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

fn read_exact(
    reader: &mut impl Read,
    bytes: &mut [u8],
    part: ReadPart,
) -> Result<(), ReadErrorKind> {
    reader
        .read_exact(bytes)
        .map_err(|error| read_error(part, error))
}

// The private reservation argument is used only to exercise this failure path.
// It is not a stored allocator, public configuration, or global fault injection.
fn read_npy(
    reader: &mut impl Read,
    file_bytes: u64,
    kind: ArrayKind,
    reserve: impl FnOnce(&mut Vec<f32>, usize) -> Result<(), TryReserveError>,
) -> Result<NpyArray, ReadErrorKind> {
    let mut magic = [0; 6];
    read_exact(reader, &mut magic, ReadPart::Preamble)?;
    if &magic != b"\x93NUMPY" {
        return Err(ReadErrorKind::Magic);
    }
    let mut version = [0; 2];
    read_exact(reader, &mut version, ReadPart::Preamble)?;
    let (header_bytes, preamble_bytes) = match version {
        [1, 0] => {
            let mut bytes = [0; 2];
            read_exact(reader, &mut bytes, ReadPart::Preamble)?;
            (usize::from(u16::from_le_bytes(bytes)), 10usize)
        }
        [2 | 3, 0] => {
            let mut bytes = [0; 4];
            read_exact(reader, &mut bytes, ReadPart::Preamble)?;
            (
                usize::try_from(u32::from_le_bytes(bytes)).map_err(|_| ReadErrorKind::Size)?,
                12usize,
            )
        }
        _ => return Err(ReadErrorKind::Version(version)),
    };
    if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
        return Err(ReadErrorKind::Header(HeaderError::Length));
    }
    let payload_offset = preamble_bytes
        .checked_add(header_bytes)
        .ok_or(ReadErrorKind::Size)?;
    if file_bytes < payload_offset as u64 {
        return Err(ReadErrorKind::ShortRead(ReadPart::Header));
    }
    let mut header = vec![0; header_bytes];
    read_exact(reader, &mut header, ReadPart::Header)?;
    let (dtype, byte_order, source_shape) = parse_header(&header)?;
    match (kind, source_shape.as_slice()) {
        (ArrayKind::SpectraLut, [n, m, _]) if n == m => {}
        (ArrayKind::MallettBasis, [81, 3] | [3, 81]) => {}
        _ => return Err(ReadErrorKind::Shape),
    }
    let count = source_shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or(ReadErrorKind::Size)?;
    let payload_bytes = count
        .checked_mul(dtype.bytes())
        .ok_or(ReadErrorKind::Size)?;
    let decoded_bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or(ReadErrorKind::Size)?;
    if decoded_bytes > isize::MAX as usize {
        return Err(ReadErrorKind::Capacity);
    }
    let end = payload_offset
        .checked_add(payload_bytes)
        .ok_or(ReadErrorKind::Size)?;
    if file_bytes < end as u64 {
        return Err(ReadErrorKind::ShortRead(ReadPart::Payload));
    }
    let mut values = Vec::new();
    reserve(&mut values, count).map_err(|_| ReadErrorKind::Capacity)?;
    let mut buffer = [0u8; 8192];
    let width = dtype.bytes();
    while values.len() < count {
        let samples = (count - values.len()).min(buffer.len() / width);
        // samples is bounded by the fixed chunk capacity; all total products
        // above were checked before reservation or any payload access.
        let bytes = &mut buffer[..samples * width];
        read_exact(reader, bytes, ReadPart::Payload)?;
        for sample in bytes.chunks_exact(width) {
            let value = match dtype {
                NpyDtype::F16 => half_to_float(u16::from_le_bytes(
                    sample.try_into().expect("two-byte chunk"),
                )),
                NpyDtype::F32 => f32::from_le_bytes(sample.try_into().expect("four-byte chunk")),
                NpyDtype::F64 => {
                    f64::from_le_bytes(sample.try_into().expect("eight-byte chunk")) as f32
                }
            };
            values.push(value);
        }
    }
    let shape = if matches!(kind, ArrayKind::MallettBasis) {
        if source_shape == [3, 81] {
            transpose_basis(&mut values);
        }
        vec![81, 3]
    } else {
        source_shape.clone()
    };
    Ok(NpyArray {
        version,
        dtype,
        byte_order,
        source_shape,
        shape,
        values,
    })
}

fn transpose_basis(values: &mut [f32]) {
    // The constructor established exactly 3*81 elements. Move each permutation
    // cycle in place, preserving bits and using only fixed bookkeeping storage.
    let mut visited = [false; 243];
    for start in 0..243 {
        if visited[start] {
            continue;
        }
        let mut index = start;
        let mut value = values[index];
        loop {
            visited[index] = true;
            let next = (index % 81) * 3 + index / 81;
            std::mem::swap(&mut value, &mut values[next]);
            index = next;
            if index == start {
                break;
            }
        }
    }
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

fn parse_header(bytes: &[u8]) -> Result<(NpyDtype, NpyByteOrder, Vec<usize>), ReadErrorKind> {
    if !bytes.is_ascii() {
        return Err(ReadErrorKind::Header(HeaderError::Syntax));
    }
    let mut text =
        std::str::from_utf8(bytes).map_err(|_| ReadErrorKind::Header(HeaderError::Syntax))?;
    consume(&mut text, "{")?;
    let mut dtype = None;
    let mut order = None;
    let mut shape = None;
    loop {
        let key = quoted(&mut text)?;
        consume(&mut text, ":")?;
        match key {
            "descr" if dtype.is_none() => dtype = Some(parse_dtype(quoted(&mut text)?)?),
            "fortran_order" if order.is_none() => {
                text = text.trim_ascii_start();
                if let Some(rest) = text.strip_prefix("False") {
                    text = rest;
                    order = Some(false);
                } else if let Some(rest) = text.strip_prefix("True") {
                    text = rest;
                    order = Some(true);
                } else {
                    return Err(ReadErrorKind::Header(HeaderError::Syntax));
                }
            }
            "shape" if shape.is_none() => shape = Some(parse_shape(&mut text)?),
            _ => return Err(ReadErrorKind::Header(HeaderError::Fields)),
        }
        text = text.trim_ascii_start();
        if let Some(rest) = text.strip_prefix(',') {
            text = rest.trim_ascii_start();
        } else if !text.starts_with('}') {
            return Err(ReadErrorKind::Header(HeaderError::Syntax));
        }
        if text.starts_with('}') {
            consume(&mut text, "}")?;
            break;
        }
    }
    if !text.trim_ascii().is_empty() {
        return Err(ReadErrorKind::Header(HeaderError::Syntax));
    }
    let (dtype, byte_order) = dtype.ok_or(ReadErrorKind::Header(HeaderError::Fields))?;
    if order.ok_or(ReadErrorKind::Header(HeaderError::Fields))? {
        return Err(ReadErrorKind::FortranOrder);
    }
    Ok((
        dtype,
        byte_order,
        shape.ok_or(ReadErrorKind::Header(HeaderError::Fields))?,
    ))
}

fn consume(text: &mut &str, token: &str) -> Result<(), ReadErrorKind> {
    *text = text
        .trim_ascii_start()
        .strip_prefix(token)
        .ok_or(ReadErrorKind::Header(HeaderError::Syntax))?;
    Ok(())
}

fn quoted<'a>(text: &mut &'a str) -> Result<&'a str, ReadErrorKind> {
    let bytes = text.trim_ascii_start();
    let quote = bytes
        .as_bytes()
        .first()
        .filter(|c| matches!(c, b'\'' | b'"'))
        .ok_or(ReadErrorKind::Header(HeaderError::Syntax))?;
    let end = bytes[1..]
        .find(char::from(*quote))
        .ok_or(ReadErrorKind::Header(HeaderError::Syntax))?
        + 1;
    let value = &bytes[1..end];
    if value.contains('\\') {
        return Err(ReadErrorKind::Header(HeaderError::Syntax));
    }
    *text = &bytes[end + 1..];
    Ok(value)
}

fn parse_dtype(text: &str) -> Result<(NpyDtype, NpyByteOrder), ReadErrorKind> {
    let order = match text.as_bytes().first() {
        Some(b'<') => NpyByteOrder::Little,
        Some(b'|') => NpyByteOrder::NotApplicable,
        _ => return Err(ReadErrorKind::ByteOrder),
    };
    let dtype = match &text[1..] {
        "f2" => NpyDtype::F16,
        "f4" => NpyDtype::F32,
        "f8" => NpyDtype::F64,
        _ => return Err(ReadErrorKind::Dtype),
    };
    Ok((dtype, order))
}

fn parse_shape(text: &mut &str) -> Result<Vec<usize>, ReadErrorKind> {
    consume(text, "(")?;
    let mut shape = Vec::new();
    loop {
        *text = text.trim_ascii_start();
        if let Some(rest) = text.strip_prefix('+') {
            *text = rest;
        }
        let digits = text.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return Err(ReadErrorKind::Shape);
        }
        let dimension = text[..digits]
            .parse::<usize>()
            .map_err(|_| ReadErrorKind::Size)?;
        if dimension == 0 || dimension > i32::MAX as usize || shape.len() == 3 {
            return Err(ReadErrorKind::Shape);
        }
        shape.push(dimension);
        *text = text[digits..].trim_ascii_start();
        if let Some(rest) = text.strip_prefix(',') {
            *text = rest.trim_ascii_start();
        } else if !text.starts_with(')') {
            return Err(ReadErrorKind::Shape);
        }
        if text.starts_with(')') {
            consume(text, ")")?;
            return Ok(shape);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn input(dtype: &str, count: usize, width: usize) -> Vec<u8> {
        let header = format!("{{'descr':'{dtype}','fortran_order':False,'shape':(1,1,{count})}}\n");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend(u16::try_from(header.len()).unwrap().to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.resize(bytes.len() + count * width, 0);
        bytes
    }

    #[test]
    fn reservation_failure() {
        for (dtype, width) in [("<f2", 2), ("<f4", 4), ("<f8", 8)] {
            let bytes = input(dtype, 10, width);
            let length = bytes.len() as u64;
            let mut reader = Cursor::new(bytes);
            let result = read_npy(
                &mut reader,
                length,
                ArrayKind::SpectraLut,
                |values, count| {
                    assert!(values.is_empty());
                    assert_eq!(count, 10);
                    values.try_reserve_exact(usize::MAX)
                },
            );
            assert_eq!(result.unwrap_err(), ReadErrorKind::Capacity);
            assert_eq!(reader.position(), length - (10 * width) as u64);
        }
    }

    #[test]
    fn payload_size_precedes_reservation() {
        let bytes = input("<f4", 10, 4);
        let length = bytes.len() as u64 - 1;
        let result = read_npy(
            &mut Cursor::new(bytes),
            length,
            ArrayKind::SpectraLut,
            |_, _| panic!("truncated payload must not reserve"),
        );
        assert_eq!(
            result.unwrap_err(),
            ReadErrorKind::ShortRead(ReadPart::Payload)
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
        for (dtype, width) in [("<f2", 2), ("<f4", 4), ("<f8", 8)] {
            let bytes = input(dtype, 10_000, width);
            let length = bytes.len() as u64;
            let mut reader = ChunkReader {
                input: Cursor::new(bytes),
                max_read: 0,
                fail_after: None,
            };
            let array = read_npy(
                &mut reader,
                length,
                ArrayKind::SpectraLut,
                Vec::try_reserve_exact,
            )
            .unwrap();
            assert_eq!(array.values().len(), 10_000);
            assert!(array.values().iter().all(|v| v.to_bits() == 0));
            assert_eq!(reader.max_read, 8192);
            assert_eq!(reader.input.position(), length);
        }
    }

    #[test]
    fn late_payload_failures() {
        for (dtype, width) in [("<f2", 2), ("<f4", 4), ("<f8", 8)] {
            let bytes = input(dtype, 10_000, width);
            let length = bytes.len() as u64;
            let mut reader = ChunkReader {
                input: Cursor::new(bytes.clone()),
                max_read: 0,
                fail_after: Some(9_000),
            };
            let error = read_npy(
                &mut reader,
                length,
                ArrayKind::SpectraLut,
                Vec::try_reserve_exact,
            )
            .unwrap_err();
            assert_eq!(
                error,
                ReadErrorKind::Io {
                    part: ReadPart::Payload,
                    kind: io::ErrorKind::PermissionDenied
                }
            );
            // Simulate file truncation after metadata preflight, after at least
            // one complete chunk was converted into the unpublished vector.
            let mut reader = Cursor::new(bytes[..9_000].to_vec());
            let error = read_npy(
                &mut reader,
                length,
                ArrayKind::SpectraLut,
                Vec::try_reserve_exact,
            )
            .unwrap_err();
            assert_eq!(error, ReadErrorKind::ShortRead(ReadPart::Payload));
        }
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
