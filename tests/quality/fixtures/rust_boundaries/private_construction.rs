use film_juicer_core::profile::{FilmProfile, PrintProfile};

pub fn bypass_film_construction(film: FilmProfile) -> FilmProfile {
    FilmProfile { ..film }
}

pub fn bypass_print_construction(print: PrintProfile) -> PrintProfile {
    PrintProfile { ..print }
}
