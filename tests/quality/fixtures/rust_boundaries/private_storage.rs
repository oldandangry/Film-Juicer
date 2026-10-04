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
