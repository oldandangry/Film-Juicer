//! Selected calibration outcomes, separate from private raw projection and recipe policy.
#![forbid(unsafe_code)]

use film_juicer_core::assets::{AssetError, Assets};
use film_juicer_core::data_io::calibration::{ErrorKind, Field};
use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CalibrationField {
    ResourceRead,
    Root,
    PrintProfile,
    PrintIlluminant,
    CmyCc,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CalibrationLookup {
    Found([f32; 3]),
    MissingFile,
    MissingEntry,
    Malformed(CalibrationField),
}
fn classify(error: AssetError) -> Result<CalibrationLookup, AssetError> {
    let AssetError::Calibration(detail) = &error else {
        return Err(error);
    };
    match detail.kind() {
        ErrorKind::Read(io::ErrorKind::OutOfMemory) => Err(error),
        ErrorKind::MissingFile => Ok(CalibrationLookup::MissingFile),
        ErrorKind::Read(_) => Ok(CalibrationLookup::Malformed(CalibrationField::ResourceRead)),
        ErrorKind::Malformed(field) => Ok(CalibrationLookup::Malformed(match field {
            Field::Root => CalibrationField::Root,
            Field::PrintProfile => CalibrationField::PrintProfile,
            Field::PrintIlluminant => CalibrationField::PrintIlluminant,
            Field::CmyCc => CalibrationField::CmyCc,
        })),
    }
}
pub(crate) fn lookup(
    assets: &Assets,
    print_stock: &str,
    illuminant: &str,
    film_stock: &str,
) -> Result<CalibrationLookup, AssetError> {
    let calibration = match assets.neutral_calibration() {
        Ok(calibration) => calibration,
        Err(error) => return classify(error),
    };
    match calibration.lookup(print_stock, illuminant, film_stock) {
        Ok(Some(cmy_cc)) => Ok(CalibrationLookup::Found(cmy_cc)),
        Ok(None) => Ok(CalibrationLookup::MissingEntry),
        Err(error) => classify(AssetError::Calibration(std::sync::Arc::new(error))),
    }
}
