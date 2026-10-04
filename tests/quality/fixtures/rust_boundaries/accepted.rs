use crate::asset_profile::{FilmOwner, FilmView, PrintOwner, PrintView};
use film_juicer_core::profile::{FilmProfile, PrintProfile, ProfileCompletionError, ProfileSource};

pub fn complete_film(source: ProfileSource) -> Result<FilmProfile, ProfileCompletionError> {
    FilmProfile::new(source)
}

pub fn complete_print(source: ProfileSource) -> Result<PrintProfile, ProfileCompletionError> {
    PrintProfile::new(source)
}

pub fn film_view(owner: &FilmOwner) -> FilmView<'_> {
    owner.view()
}

pub fn print_view(owner: &PrintOwner) -> PrintView<'_> {
    owner.view()
}

pub fn consume_film(owner: FilmOwner) -> usize {
    let count = owner.view().tables.interpolation_log_exposure.len();
    drop(owner);
    count
}

pub fn consume_print(owner: PrintOwner) -> usize {
    let count = owner.view().tables.interpolation_log_exposure.len();
    drop(owner);
    count
}

use crate::asset_spectral::{SpectraOwner, SpectraView, MallettOwner, CmfOwner};
use film_juicer_core::assets::SpectraLut;
use film_juicer_core::data_io::CmfRows;
use std::sync::Arc;
pub fn spectra_owner(lut: Arc<SpectraLut>) -> SpectraOwner { SpectraOwner::new(lut) }
pub fn spectra_view(owner: &SpectraOwner) -> SpectraView<'_> { owner.view() }
pub fn mallett_owner(basis: Arc<[[f32;3];81]>) -> MallettOwner { MallettOwner::new(basis) }
pub fn mallett_samples(owner: &MallettOwner) -> &[[f32;3];81] { owner.samples() }
pub fn cmf_owner(rows: Arc<CmfRows>) -> CmfOwner { CmfOwner::new(rows) }
pub fn cmf_rows(owner: &CmfOwner) -> &[[f32;4]] { owner.rows() }

use crate::asset_spectral::CsvPairsOwner;
use film_juicer_core::data_io::CsvPairs;
pub fn csv_owner(rows: Arc<CsvPairs>) -> CsvPairsOwner { CsvPairsOwner::new(rows) }
pub fn csv_rows(owner: &CsvPairsOwner) -> &[[f32;2]] { owner.rows() }
