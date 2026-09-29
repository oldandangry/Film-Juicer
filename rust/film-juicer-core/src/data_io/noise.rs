//! Static STBN/Wang decoding. Values own complete buffers; no cache or upload.
//! Explicit byte buffers reserve fallibly and are filled in place. JSON/parser,
//! path and incidental small allocations can still abort.

use std::collections::TryReserveError;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

const STBN_DIMENSIONS: [usize; 3] = [512, 512, 256];
const WANG_DIMENSIONS: [usize; 3] = [256, 256, 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataField {
    Root,
    Resolution,
    Tiles,
    Colors,
    Mapping,
    Index,
    Labels,
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Missing,
    Open(io::ErrorKind),
    Read(io::ErrorKind),
    ShortRead,
    Length { expected: usize, actual: u64 },
    Size,
    Capacity,
    Json,
    Metadata(MetadataField),
    Dimensions,
}

/// Paths are acquired before buffer construction. Capacity failures carry no
/// newly allocated diagnostic string or partially constructed payload.
#[derive(Debug)]
pub struct Error {
    path: PathBuf,
    kind: ErrorKind,
}

impl Error {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:?}", self.path.display(), self.kind)
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub struct Stbn {
    bytes: Vec<u8>,
}

impl Stbn {
    pub fn dimensions(&self) -> [usize; 3] {
        STBN_DIMENSIONS
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug)]
pub struct Wang {
    tiles: Vec<u8>,
    lut: Vec<u8>,
    colors: usize,
}

impl Wang {
    pub fn dimensions(&self) -> [usize; 3] {
        WANG_DIMENSIONS
    }

    pub fn tiles(&self) -> &[u8] {
        &self.tiles
    }

    /// Flat LUT indexed in left, right, top, bottom order, with bottom fastest.
    pub fn lut(&self) -> &[u8] {
        &self.lut
    }

    pub fn colors(&self) -> usize {
        self.colors
    }
}

pub fn load_stbn(path: impl AsRef<Path>) -> Result<Stbn, Error> {
    let path = path.as_ref().to_path_buf();
    let result = (|| {
        let (mut file, length) = open_payload(&path)?;
        read_stbn(&mut file, length, Vec::try_reserve_exact)
    })();
    result.map_err(|kind| Error { path, kind })
}

pub fn load_wang(
    tiles_path: impl AsRef<Path>,
    metadata_path: impl AsRef<Path>,
) -> Result<Wang, Error> {
    let tiles_path = tiles_path.as_ref().to_path_buf();
    let metadata_path = metadata_path.as_ref().to_path_buf();
    enum Input {
        Tiles,
        Metadata,
    }
    let result = (|| {
        for (path, input) in [
            (&tiles_path, Input::Tiles),
            (&metadata_path, Input::Metadata),
        ] {
            match fs::exists(path) {
                Ok(true) => (),
                Ok(false) => return Err((input, ErrorKind::Missing)),
                Err(error) => return Err((input, ErrorKind::Open(error.kind()))),
            }
        }
        let metadata =
            File::open(&metadata_path).map_err(|e| (Input::Metadata, ErrorKind::Open(e.kind())))?;
        let root =
            read_metadata(BufReader::new(metadata)).map_err(|kind| (Input::Metadata, kind))?;
        let (colors, lut) = build_lut(&root, &mut Vec::try_reserve_exact)
            .map_err(|kind| (Input::Metadata, kind))?;
        let (mut file, length) = open_payload(&tiles_path).map_err(|kind| (Input::Tiles, kind))?;
        read_wang(&mut file, length, colors, lut, Vec::try_reserve_exact)
            .map_err(|kind| (Input::Tiles, kind))
    })();
    result.map_err(|(input, kind)| Error {
        path: match input {
            Input::Tiles => tiles_path,
            Input::Metadata => metadata_path,
        },
        kind,
    })
}

fn open_payload(path: &Path) -> Result<(File, u64), ErrorKind> {
    let file = File::open(path).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            ErrorKind::Missing
        } else {
            ErrorKind::Open(e.kind())
        }
    })?;
    let length = file
        .metadata()
        .map_err(|e| ErrorKind::Read(e.kind()))?
        .len();
    Ok((file, length))
}

fn read_stbn(
    reader: &mut impl Read,
    length: u64,
    reserve: impl FnOnce(&mut Vec<u8>, usize) -> Result<(), TryReserveError>,
) -> Result<Stbn, ErrorKind> {
    let bytes = read_bytes(reader, length, &STBN_DIMENSIONS, reserve)?;
    Ok(Stbn { bytes })
}

fn read_wang(
    reader: &mut impl Read,
    length: u64,
    colors: usize,
    lut: Vec<u8>,
    reserve: impl FnOnce(&mut Vec<u8>, usize) -> Result<(), TryReserveError>,
) -> Result<Wang, ErrorKind> {
    let tiles = read_bytes(reader, length, &WANG_DIMENSIONS, reserve)?;
    Ok(Wang { tiles, lut, colors })
}

fn byte_count(dimensions: &[usize]) -> Result<usize, ErrorKind> {
    let count = dimensions.iter().try_fold(1usize, |n, &dimension| {
        n.checked_mul(dimension).ok_or(ErrorKind::Size)
    })?;
    if count > isize::MAX as usize {
        return Err(ErrorKind::Capacity);
    }
    Ok(count)
}

fn read_bytes(
    reader: &mut impl Read,
    length: u64,
    dimensions: &[usize],
    reserve: impl FnOnce(&mut Vec<u8>, usize) -> Result<(), TryReserveError>,
) -> Result<Vec<u8>, ErrorKind> {
    let count = byte_count(dimensions)?;
    if length != count as u64 {
        return Err(ErrorKind::Length {
            expected: count,
            actual: length,
        });
    }
    let mut bytes = Vec::new();
    reserve(&mut bytes, count).map_err(|_| ErrorKind::Capacity)?;
    bytes.resize(count, 0);
    for chunk in bytes.chunks_mut(8192) {
        reader.read_exact(chunk).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                ErrorKind::ShortRead
            } else {
                ErrorKind::Read(error.kind())
            }
        })?;
    }
    Ok(bytes)
}

fn read_metadata(mut reader: impl Read) -> Result<Value, ErrorKind> {
    let mut prefix = [0; 3];
    let mut count = 0;
    while count < prefix.len() {
        match reader.read(&mut prefix[count..]) {
            Ok(0) => break,
            Ok(n) => count += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(ErrorKind::Read(error.kind())),
        }
    }
    let prefix = if prefix == *b"\xef\xbb\xbf" {
        &[][..]
    } else {
        &prefix[..count]
    };
    // Native stream extraction consumes the first JSON value, including a BOM,
    // and does not require EOF after that value.
    let reader = prefix.chain(reader);
    Value::deserialize(&mut serde_json::Deserializer::from_reader(reader)).map_err(|error| {
        error
            .io_error_kind()
            .map_or(ErrorKind::Json, ErrorKind::Read)
    })
}

// Keep native bool conversion, integer narrowing and representable fraction
// truncation. None denotes the approved out-of-i32-range floating input rule.
fn metadata_integer(value: &Value, field: MetadataField) -> Result<Option<i32>, ErrorKind> {
    match value {
        Value::Bool(value) => Ok(Some(i32::from(*value))),
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                Ok(Some(value as i32))
            } else if let Some(value) = number.as_i64() {
                Ok(Some(value as i32))
            } else {
                let value = number.as_f64().expect("JSON number is an f64").trunc();
                if value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX) {
                    Ok(Some(value as i32))
                } else {
                    Ok(None)
                }
            }
        }
        _ => Err(ErrorKind::Metadata(field)),
    }
}

fn build_lut(
    root: &Value,
    reserve: &mut impl FnMut(&mut Vec<u8>, usize) -> Result<(), TryReserveError>,
) -> Result<(usize, Vec<u8>), ErrorKind> {
    let root = root
        .as_object()
        .ok_or(ErrorKind::Metadata(MetadataField::Root))?;
    let mut sizes = [0; 3];
    for (destination, (name, field)) in sizes.iter_mut().zip([
        ("resolution", MetadataField::Resolution),
        ("tiles", MetadataField::Tiles),
        ("colors", MetadataField::Colors),
    ]) {
        let value = root.get(name).ok_or(ErrorKind::Metadata(field))?;
        *destination = metadata_integer(value, field)?.ok_or(ErrorKind::Metadata(field))?;
    }
    let [resolution, count, colors] = sizes;
    if resolution != WANG_DIMENSIONS[0] as i32 || count != WANG_DIMENSIONS[2] as i32 || colors <= 0
    {
        return Err(ErrorKind::Dimensions);
    }
    let colors = colors as usize;
    let size = byte_count(&[colors; 4])?;
    let mapping = root
        .get("mapping")
        .and_then(Value::as_array)
        .ok_or(ErrorKind::Metadata(MetadataField::Mapping))?;
    let mut lut = Vec::new();
    reserve(&mut lut, size).map_err(|_| ErrorKind::Capacity)?;
    lut.resize(size, 0);
    for entry in mapping {
        let (Some(index), Some(labels)) = (entry.get("index"), entry.get("labels")) else {
            continue;
        };
        let Some(index) = metadata_integer(index, MetadataField::Index)? else {
            continue;
        };
        let labels = labels
            .as_object()
            .ok_or(ErrorKind::Metadata(MetadataField::Labels))?;
        let mut edges = [0i32; 4];
        let mut representable = true;
        for (edge, (name, field)) in edges.iter_mut().zip([
            ("L", MetadataField::Left),
            ("R", MetadataField::Right),
            ("T", MetadataField::Top),
            ("B", MetadataField::Bottom),
        ]) {
            if let Some(value) = labels.get(name) {
                if let Some(value) = metadata_integer(value, field)? {
                    *edge = value;
                } else {
                    representable = false;
                    break;
                }
            }
        }
        if !representable
            || edges
                .iter()
                .any(|&edge| edge < 0 || edge as usize >= colors)
            || index < 0
            || index >= count
        {
            continue;
        }
        // Checked colors^4 bounds every L/R/T/B multiply/add below.
        let [left, right, top, bottom] = edges.map(|edge| edge as usize);
        let offset = ((left * colors + right) * colors + top) * colors + bottom;
        lut[offset] = index as u8;
    }
    Ok((colors, lut))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn reservation_failure(bytes: &mut Vec<u8>, _: usize) -> Result<(), TryReserveError> {
        bytes.try_reserve_exact(usize::MAX)
    }

    #[test]
    fn rejects_size_and_capacity_overflow() {
        for (dimensions, expected) in [
            (vec![usize::MAX, 2], ErrorKind::Size),
            (vec![isize::MAX as usize + 1], ErrorKind::Capacity),
        ] {
            let result = read_bytes(&mut io::empty(), 0, &dimensions, |_, _| {
                panic!("invalid products must not reach reservation")
            });
            assert_eq!(result.unwrap_err(), expected);
        }
        for (colors, expected) in [(65_536, ErrorKind::Size), (60_000, ErrorKind::Capacity)] {
            let root =
                serde_json::json!({"resolution":256,"tiles":16,"colors":colors,"mapping":[]});
            let result = build_lut(&root, &mut |_, _| {
                panic!("invalid LUT product must not reserve")
            });
            assert_eq!(result.unwrap_err(), expected);
        }
    }

    #[test]
    fn rejects_wrong_lengths_before_reservation() {
        for dimensions in [STBN_DIMENSIONS, WANG_DIMENSIONS] {
            let count = byte_count(&dimensions).unwrap();
            for actual in [0, count as u64 - 1, count as u64 + 1] {
                let result = read_bytes(&mut io::empty(), actual, &dimensions, |_, _| {
                    panic!("wrong length must not reserve")
                });
                assert_eq!(
                    result.unwrap_err(),
                    ErrorKind::Length {
                        expected: count,
                        actual
                    }
                );
            }
        }
    }

    #[test]
    fn reservation_failure_publishes_no_noise() {
        let mut reader = Cursor::new([9u8; 1]);
        assert_eq!(
            read_stbn(&mut reader, 67_108_864, reservation_failure).unwrap_err(),
            ErrorKind::Capacity
        );
        assert_eq!(reader.position(), 0);
        assert_eq!(
            read_wang(&mut reader, 1_048_576, 2, vec![0; 16], reservation_failure).unwrap_err(),
            ErrorKind::Capacity
        );
        assert_eq!(reader.position(), 0);
        let root = serde_json::json!({"resolution":256,"tiles":16,"colors":2,"mapping":[]});
        assert_eq!(
            build_lut(&root, &mut reservation_failure).unwrap_err(),
            ErrorKind::Capacity
        );
    }

    struct ChunkReader {
        input: Cursor<Vec<u8>>,
        chunk_size: usize,
        fail_after: Option<u64>,
    }

    impl Read for ChunkReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self
                .fail_after
                .is_some_and(|position| self.input.position() >= position)
            {
                return Err(io::Error::from(io::ErrorKind::PermissionDenied));
            }
            let count = self.chunk_size.min(buffer.len());
            self.input.read(&mut buffer[..count])
        }
    }

    #[test]
    fn fills_the_reserved_buffer() {
        let input: Vec<u8> = (0..20_000).map(|i| (i % 251) as u8).collect();
        let mut reader = ChunkReader {
            input: Cursor::new(input.clone()),
            chunk_size: 17,
            fail_after: None,
        };
        let mut address = 0;
        let bytes = read_bytes(&mut reader, 20_000, &[20_000], |bytes, count| {
            assert!(bytes.is_empty());
            bytes.try_reserve_exact(count)?;
            address = bytes.as_ptr() as usize;
            Ok(())
        })
        .unwrap();
        assert_eq!(bytes, input);
        assert_eq!(bytes.as_ptr() as usize, address);
    }

    #[test]
    fn read_failure_publishes_no_noise() {
        for (fail_after, expected) in [
            (Some(9000), ErrorKind::Read(io::ErrorKind::PermissionDenied)),
            (None, ErrorKind::ShortRead),
        ] {
            let reader = || ChunkReader {
                input: Cursor::new(vec![7; 9000]),
                chunk_size: 3,
                fail_after,
            };
            assert_eq!(
                read_stbn(&mut reader(), 67_108_864, Vec::try_reserve_exact).unwrap_err(),
                expected
            );
            assert_eq!(
                read_wang(
                    &mut reader(),
                    1_048_576,
                    2,
                    vec![0; 16],
                    Vec::try_reserve_exact
                )
                .unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn metadata_short_reads_and_errors() {
        let bytes = b"\xef\xbb\xbf{\"mapping\":[]} trailing";
        let reader = || ChunkReader {
            input: Cursor::new(bytes.to_vec()),
            chunk_size: 1,
            fail_after: None,
        };
        assert_eq!(
            read_metadata(reader()).unwrap(),
            serde_json::json!({"mapping":[]})
        );
        for position in [0, 2, 10] {
            let mut reader = reader();
            reader.fail_after = Some(position);
            assert_eq!(
                read_metadata(reader).unwrap_err(),
                ErrorKind::Read(io::ErrorKind::PermissionDenied)
            );
        }
    }

    #[test]
    fn float_integer_boundaries() {
        for (number, expected) in [
            (2147483647.9, Some(i32::MAX)),
            (-2147483648.9, Some(i32::MIN)),
            (2147483648.0, None),
            (-2147483649.0, None),
            (1e30, None),
            (-1e30, None),
            (-0.75, Some(0)),
        ] {
            assert_eq!(
                metadata_integer(&serde_json::json!(number), MetadataField::Index).unwrap(),
                expected
            );
        }
    }
}
