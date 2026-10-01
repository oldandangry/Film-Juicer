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
        let bytes = fs::read(&path).map_err(|error| {
            // Native read failure is MissingFile when existence cannot be
            // established, and malformed resource_read when it can.
            if fs::exists(&path).unwrap_or(false) {
                ErrorKind::Read(error.kind())
            } else {
                ErrorKind::MissingFile
            }
        })?;
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
