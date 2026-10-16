use std::sync::Arc;
use film_juicer_core::film::{Development, Params};
use film_juicer_core::profile::FilmProfile;
use film_juicer_core::route::ScanRoute;
fn consume(profile: Arc<FilmProfile>) {
    let development = Development::new(profile, Params { gamma_factor: 1.0, layers_required: true, requested_route: ScanRoute::NegativeDirect }).unwrap();
    let _ = (development.route(), development.polarity(), development.gamma_rgb(), development.minimum_cmy(), development.maximum_cmy(), development.layers_required(), development.log_exposure(), development.authored_curves_cmy(), development.normalized_curves_cmy(), development.density_curves_layers(), development.normalized_curves_hash(), development.layers_hash(), development.hash());
}
