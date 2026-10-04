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
    let count = owner.view().tables.log_exposure.len();
    drop(owner);
    count
}

pub fn consume_print(owner: PrintOwner) -> usize {
    let count = owner.view().tables.log_exposure.len();
    drop(owner);
    count
}
