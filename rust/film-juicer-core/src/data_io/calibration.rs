//! Neutral-print calibration input and lazy selected-entry validation.
//! Parsing establishes an object root; only the selected print/illuminant/film
//! branch is structurally validated. No cache or recipe policy lives here.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Root,
    PrintProfile,
    PrintIlluminant,
    CmyCc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    MissingFile,
    Read(io::ErrorKind),
    Malformed(Field),
}

#[derive(Debug)]
pub struct Error {
    path: PathBuf,
    kind: ErrorKind,
    selection: Option<[String; 3]>,
}

impl Error {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Selected print profile, print illuminant and film profile, in lookup order.
    pub fn selection(&self) -> Option<[&str; 3]> {
        self.selection
            .as_ref()
            .map(|keys| keys.each_ref().map(String::as_str))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:?}", self.path.display(), self.kind)?;
        if let Some(keys) = self.selection() {
            write!(f, " ({}/{}/{})", keys[0], keys[1], keys[2])?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub struct NeutralCalibration {
    path: PathBuf,
    root: Map<String, Value>,
}

pub fn load_neutral_calibration(path: impl AsRef<Path>) -> Result<NeutralCalibration, Error> {
    let path = path.as_ref().to_path_buf();
    let result = (|| {
        let bytes = read_resource(&path)
            .map_err(|error| classify_read_error(error.kind(), || probe_resource(&path)))?;
        decode_root(&bytes)
    })();
    match result {
        Ok(root) => Ok(NeutralCalibration { path, root }),
        Err(kind) => Err(Error {
            path,
            kind,
            selection: None,
        }),
    }
}

// Preserve recoverable allocation before the ordinary existence-based policy.
fn classify_read_error(
    kind: io::ErrorKind,
    exists: impl FnOnce() -> io::Result<bool>,
) -> ErrorKind {
    if kind == io::ErrorKind::OutOfMemory {
        return ErrorKind::Read(kind);
    }
    match exists() {
        Err(error) if error.kind() == io::ErrorKind::OutOfMemory => {
            ErrorKind::Read(io::ErrorKind::OutOfMemory)
        }
        Ok(true) => ErrorKind::Read(kind),
        _ => ErrorKind::MissingFile,
    }
}

fn read_resource(path: &Path) -> io::Result<Vec<u8>> {
    #[cfg(any(test, feature = "test-support"))]
    if let Some(kind) = test_support::read_error() {
        return Err(kind.into());
    }
    fs::read(path)
}

fn probe_resource(path: &Path) -> io::Result<bool> {
    #[cfg(any(test, feature = "test-support"))]
    if let Some(result) = test_support::probe_result() {
        return result.map_err(io::Error::from);
    }
    fs::exists(path)
}

/// Deterministic target-local faults at the real read/classification boundary.
/// These functions and their state are absent from production configurations.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use std::cell::Cell;
    use std::io;
    #[derive(Clone, Copy)]
    struct Fault {
        read: io::ErrorKind,
        probe: Result<bool, io::ErrorKind>,
    }
    thread_local! {
        static FAULT: Cell<Option<Fault>> = const { Cell::new(None) };
        static READS: Cell<usize> = const { Cell::new(0) };
        static PROBES: Cell<usize> = const { Cell::new(0) };
    }
    pub fn set_read_fault(read: io::ErrorKind, probe: Result<bool, io::ErrorKind>) {
        FAULT.set(Some(Fault { read, probe }));
        READS.set(0);
        PROBES.set(0);
    }
    pub fn clear_read_fault() {
        FAULT.set(None);
    }
    pub fn counts() -> (usize, usize) {
        (READS.get(), PROBES.get())
    }
    pub(super) fn read_error() -> Option<io::ErrorKind> {
        READS.set(READS.get() + 1);
        FAULT.get().map(|fault| fault.read)
    }
    pub(super) fn probe_result() -> Option<Result<bool, io::ErrorKind>> {
        PROBES.set(PROBES.get() + 1);
        FAULT.get().map(|fault| fault.probe)
    }
}

fn decode_root(bytes: &[u8]) -> Result<Map<String, Value>, ErrorKind> {
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    match serde_json::from_slice(bytes) {
        Ok(Value::Object(root)) => Ok(root),
        _ => Err(ErrorKind::Malformed(Field::Root)),
    }
}

impl NeutralCalibration {
    /// None means the selected entry is absent, distinct from a missing file
    /// at load and a malformed selected branch. Coefficients are in CMY CC order.
    pub fn lookup(
        &self,
        print_profile: &str,
        print_illuminant: &str,
        film_profile: &str,
    ) -> Result<Option<[f32; 3]>, Error> {
        self.select(print_profile, print_illuminant, film_profile)
            .map_err(|field| Error {
                path: self.path.clone(),
                kind: ErrorKind::Malformed(field),
                selection: Some([
                    print_profile.into(),
                    print_illuminant.into(),
                    film_profile.into(),
                ]),
            })
    }

    fn select(
        &self,
        print_profile: &str,
        print_illuminant: &str,
        film_profile: &str,
    ) -> Result<Option<[f32; 3]>, Field> {
        let Some(print) = self.root.get(print_profile) else {
            return Ok(None);
        };
        let print = print.as_object().ok_or(Field::PrintProfile)?;
        let Some(illuminant) = print.get(print_illuminant) else {
            return Ok(None);
        };
        let illuminant = illuminant.as_object().ok_or(Field::PrintIlluminant)?;
        let Some(film) = illuminant.get(film_profile) else {
            return Ok(None);
        };
        let coefficients = film
            .as_array()
            .filter(|values| values.len() == 3)
            .ok_or(Field::CmyCc)?;
        let mut cmy_cc = [0.0; 3];
        for (cc, value) in cmy_cc.iter_mut().zip(coefficients) {
            let value = value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or(Field::CmyCc)?;
            // Native validates f64 finiteness before narrowing, including finite
            // inputs that overflow f32. This is decoding, not recipe validation.
            *cc = value as f32;
        }
        Ok(Some(cmy_cc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_precedes_existence_and_ordinary_policy_is_preserved() {
        assert_eq!(
            classify_read_error(io::ErrorKind::OutOfMemory, || panic!("must not probe")),
            ErrorKind::Read(io::ErrorKind::OutOfMemory)
        );
        for (probe, expected) in [
            (Ok(true), ErrorKind::Read(io::ErrorKind::PermissionDenied)),
            (Ok(false), ErrorKind::MissingFile),
            (Err(io::ErrorKind::PermissionDenied), ErrorKind::MissingFile),
            (
                Err(io::ErrorKind::OutOfMemory),
                ErrorKind::Read(io::ErrorKind::OutOfMemory),
            ),
        ] {
            assert_eq!(
                classify_read_error(io::ErrorKind::PermissionDenied, || probe
                    .map_err(io::Error::from)),
                expected
            );
        }
    }
    #[test]
    fn assets_cache_allocation_until_release() {
        let assets = crate::assets::Assets::new(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"),
        );
        for probe in [Ok(true), Ok(false), Err(io::ErrorKind::PermissionDenied)] {
            test_support::set_read_fault(io::ErrorKind::OutOfMemory, probe);
            for _ in 0..2 {
                let Err(crate::assets::AssetError::Calibration(error)) =
                    assets.neutral_calibration()
                else {
                    panic!("allocation must remain an asset error")
                };
                assert_eq!(error.kind(), ErrorKind::Read(io::ErrorKind::OutOfMemory));
            }
            assert_eq!(test_support::counts(), (1, 0));
            test_support::clear_read_fault();
            assert!(assets.neutral_calibration().is_err());
            assets.release_cached_payloads().unwrap();
            assert!(assets.neutral_calibration().is_ok());
            assets.release_cached_payloads().unwrap();
        }
        test_support::set_read_fault(
            io::ErrorKind::PermissionDenied,
            Err(io::ErrorKind::OutOfMemory),
        );
        let Err(crate::assets::AssetError::Calibration(error)) = assets.neutral_calibration()
        else {
            panic!("probe allocation must remain an asset error")
        };
        assert_eq!(error.kind(), ErrorKind::Read(io::ErrorKind::OutOfMemory));
        assert_eq!(test_support::counts(), (1, 1));
        test_support::clear_read_fault();
    }
}
