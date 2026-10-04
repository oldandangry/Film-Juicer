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
    view.tables.log_exposure.len()
}

pub fn release_print_while_borrowed(print_owner: PrintOwner) -> usize {
    let view = print_owner.view();
    drop(print_owner);
    view.tables.log_exposure.len()
}
