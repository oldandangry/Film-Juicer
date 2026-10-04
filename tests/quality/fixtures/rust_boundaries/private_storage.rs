use crate::asset_profile::{FilmOwner, PrintOwner};
use film_juicer_core::profile::{FilmDigest, FilmProfile, ProfileTables};

pub fn replace_digest(film: &mut FilmProfile, digest: FilmDigest) {
    film.digest = digest;
}

pub fn replace_axis(tables: &mut ProfileTables) {
    tables.log_exposure.clear();
}

pub fn bypass_film_owner(film_owner: &FilmOwner) {
    let _ = &film_owner.profile;
}

pub fn bypass_print_owner(print_owner: &PrintOwner) {
    let _ = &print_owner.profile;
}

use crate::asset_spectral::{SpectraOwner, MallettOwner, CmfOwner};
pub fn private_spectra(spectra_owner: SpectraOwner) { let _ = spectra_owner.lut; }
pub fn private_mallett(mallett_owner: MallettOwner) { let _ = mallett_owner.basis; }
pub fn private_cmf(cmf_owner: CmfOwner) { let _ = cmf_owner.rows; }

use crate::asset_spectral::CsvPairsOwner;
pub fn private_csv(csv_owner: CsvPairsOwner) { let _ = csv_owner.rows; }
