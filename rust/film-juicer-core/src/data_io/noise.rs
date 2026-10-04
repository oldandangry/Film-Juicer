//! Static STBN/Wang decoding. Values own complete buffers; no cache or upload.
//! Explicit byte buffers reserve fallibly and are filled in place. JSON/parser,
//! path and incidental small allocations can still abort.

use std::collections::TryReserveError;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Deserializer};

const STBN_DIMENSIONS: [usize; 3] = [512, 512, 256];
const WANG_DIMENSIONS: [usize; 3] = [256, 256, 16];

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
    Metadata,
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
    // Unit tests observe this hold only through Weak, after the bytes are dropped.
    #[cfg(test)]
    drop_hold: Option<std::sync::Arc<()>>,
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
    lut: [u8; 16],
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
        2
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
        let metadata =
            read_metadata(BufReader::new(metadata)).map_err(|kind| (Input::Metadata, kind))?;
        let lut = build_lut(metadata).map_err(|kind| (Input::Metadata, kind))?;
        let (mut file, length) = open_payload(&tiles_path).map_err(|kind| (Input::Tiles, kind))?;
        read_wang(&mut file, length, lut, Vec::try_reserve_exact)
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
    Ok(Stbn {
        bytes,
        #[cfg(test)]
        drop_hold: None,
    })
}

fn read_wang(
    reader: &mut impl Read,
    length: u64,
    lut: [u8; 16],
    reserve: impl FnOnce(&mut Vec<u8>, usize) -> Result<(), TryReserveError>,
) -> Result<Wang, ErrorKind> {
    let tiles = read_bytes(reader, length, &WANG_DIMENSIONS, reserve)?;
    Ok(Wang { tiles, lut })
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

// Derived records also accept sequences; Wang records require named members.
#[derive(Debug)]
struct JsonObject<T>(T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for JsonObject<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ObjectVisitor<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for ObjectVisitor<T> {
            type Value = JsonObject<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                T::deserialize(MapAccessDeserializer::new(map)).map(JsonObject)
            }
        }

        deserializer.deserialize_map(ObjectVisitor(PhantomData))
    }
}

#[derive(Debug, Deserialize)]
struct Metadata {
    resolution: usize,
    tiles: usize,
    colors: usize,
    mapping: [JsonObject<Mapping>; 16],
}

#[derive(Debug, Deserialize)]
struct Mapping {
    index: u8,
    labels: JsonObject<Labels>,
}

#[derive(Debug, Deserialize)]
struct Labels {
    #[serde(rename = "L")]
    left: u8,
    #[serde(rename = "R")]
    right: u8,
    #[serde(rename = "T")]
    top: u8,
    #[serde(rename = "B")]
    bottom: u8,
}

fn read_metadata(mut reader: impl Read) -> Result<Metadata, ErrorKind> {
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
    serde_json::from_reader::<_, JsonObject<Metadata>>(prefix.chain(reader))
        .map(|JsonObject(metadata)| metadata)
        .map_err(|error| {
            error
                .io_error_kind()
                .map_or(ErrorKind::Json, ErrorKind::Read)
        })
}

fn build_lut(metadata: Metadata) -> Result<[u8; 16], ErrorKind> {
    if metadata.resolution != WANG_DIMENSIONS[0]
        || metadata.tiles != WANG_DIMENSIONS[2]
        || metadata.colors != 2
    {
        return Err(ErrorKind::Dimensions);
    }
    let mut lut = [0; 16];
    let mut seen = [false; 16];
    // Exactly 16 distinct two-color combinations also establishes completeness.
    for JsonObject(Mapping { index, labels }) in metadata.mapping {
        let JsonObject(Labels {
            left,
            right,
            top,
            bottom,
        }) = labels;
        if index >= 16 || [left, right, top, bottom].iter().any(|&edge| edge > 1) {
            return Err(ErrorKind::Metadata);
        }
        let offset = usize::from(((left * 2 + right) * 2 + top) * 2 + bottom);
        if seen[offset] {
            return Err(ErrorKind::Metadata);
        }
        seen[offset] = true;
        lut[offset] = index;
    }
    Ok(lut)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::{Arc, Weak};

    pub(crate) fn track_stbn_drop(mut stbn: Stbn) -> (Stbn, Weak<()>) {
        let hold = Arc::new(());
        let probe = Arc::downgrade(&hold);
        stbn.drop_hold = Some(hold);
        (stbn, probe)
    }

    pub(crate) fn io_oom_error(path: &Path, open: bool) -> Error {
        Error {
            path: path.to_owned(),
            kind: if open {
                ErrorKind::Open(io::ErrorKind::OutOfMemory)
            } else {
                ErrorKind::Read(io::ErrorKind::OutOfMemory)
            },
        }
    }

    pub(crate) fn capacity_error(path: &Path) -> Error {
        let kind = read_stbn(&mut io::empty(), 67_108_864, |bytes, _| {
            bytes.try_reserve_exact(usize::MAX)
        })
        .unwrap_err();
        assert_eq!(kind, ErrorKind::Capacity);
        Error {
            path: path.to_owned(),
            kind,
        }
    }
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
            read_wang(&mut reader, 1_048_576, [0; 16], reservation_failure).unwrap_err(),
            ErrorKind::Capacity
        );
        assert_eq!(reader.position(), 0);
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
                read_wang(&mut reader(), 1_048_576, [0; 16], Vec::try_reserve_exact).unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn metadata_short_reads_and_errors() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources/Noise/Wang/tiles.json"),
        )
        .unwrap();
        let mut with_bom = b"\xef\xbb\xbf".to_vec();
        with_bom.extend(bytes);
        let reader = || ChunkReader {
            input: Cursor::new(with_bom.clone()),
            chunk_size: 1,
            fail_after: None,
        };
        assert_eq!(
            build_lut(read_metadata(reader()).unwrap()).unwrap(),
            [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
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
    fn metadata_reader_preserves_reported_oom_before_and_after_prefix() {
        struct OomReader {
            remaining: usize,
        }
        impl Read for OomReader {
            fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
                if self.remaining == 0 {
                    return Err(io::ErrorKind::OutOfMemory.into());
                }
                let count = out.len().min(self.remaining);
                out[..count].fill(b' ');
                self.remaining -= count;
                Ok(count)
            }
        }
        for remaining in [0, 1, 3, 64] {
            assert_eq!(
                read_metadata(OomReader { remaining }).unwrap_err(),
                ErrorKind::Read(io::ErrorKind::OutOfMemory)
            );
        }
    }

    #[test]
    fn bundled_noise_requested_capacities() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources/Noise");
        let stbn = load_stbn(root.join("stbn_scalar_512x512x256_u8.bin")).unwrap();
        let wang = load_wang(
            root.join("Wang/wang_tiles_256x256x16_u8.bin"),
            root.join("Wang/tiles.json"),
        )
        .unwrap();
        assert_eq!(stbn.bytes.capacity(), 67_108_864);
        assert_eq!(wang.tiles.capacity(), 1_048_576);
        println!(
            "noise actual capacities: STBN={} Wang={} LUT={} container sizes STBN={} Wang={}",
            stbn.bytes.capacity(),
            wang.tiles.capacity(),
            wang.lut.len(),
            size_of::<Stbn>(),
            size_of::<Wang>()
        );
    }
}
