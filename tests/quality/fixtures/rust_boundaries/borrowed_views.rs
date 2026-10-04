use crate::asset_profile::{FilmOwner, FilmView, PrintOwner, PrintView};

pub fn escape_film(film_owner: FilmOwner) -> FilmView<'static> {
    film_owner.view()
}

pub fn escape_print(print_owner: PrintOwner) -> PrintView<'static> {
    print_owner.view()
}

pub fn release_film_while_borrowed(film_owner: FilmOwner) -> usize {
    let view = film_owner.view();
    drop(film_owner);
    view.tables.interpolation_log_exposure.len()
}

pub fn release_print_while_borrowed(print_owner: PrintOwner) -> usize {
    let view = print_owner.view();
    drop(print_owner);
    view.tables.interpolation_log_exposure.len()
}

use crate::asset_spectral::{SpectraOwner, SpectraView, MallettOwner, CmfOwner};
pub fn escape_spectra(spectra_owner: SpectraOwner) -> SpectraView<'static> { spectra_owner.view() }
pub fn escape_mallett(mallett_owner: MallettOwner) -> &'static [[f32;3];81] { mallett_owner.samples() }
pub fn escape_cmf(cmf_owner: CmfOwner) -> &'static [[f32;4]] { cmf_owner.rows() }
pub fn drop_spectra(spectra_owner: SpectraOwner) -> usize {
    let view = spectra_owner.view();
    drop(spectra_owner);
    view.samples.len()
}
pub fn drop_mallett(mallett_owner: MallettOwner) -> usize {
    let view = mallett_owner.samples();
    drop(mallett_owner);
    view.len()
}
pub fn drop_cmf(cmf_owner: CmfOwner) -> usize {
    let view = cmf_owner.rows();
    drop(cmf_owner);
    view.len()
}

use crate::asset_spectral::CsvPairsOwner;
pub fn escape_csv(csv_owner: CsvPairsOwner) -> &'static [[f32;2]] { csv_owner.rows() }
pub fn drop_csv(csv_owner: CsvPairsOwner) -> usize {
    let rows = csv_owner.rows();
    drop(csv_owner);
    rows.len()
}
