//! Retained film payload and borrow-tied projection shared by asset boundaries.

#![forbid(unsafe_code)]

use std::sync::Arc;

use film_juicer_core::profile::{
    FilmProcessingDefaults, FilmProfile, IlluminantKind, Polarity, PrintDensityCurves,
    PrintDensityError, PrintProfile, ProfileTables, Stage, Support,
};
#[cfg(feature = "test-support")]
use film_juicer_core::profile::{Antihalation, ProfileUse};

pub(crate) struct FilmOwner {
    profile: Arc<FilmProfile>,
}

#[cfg(feature = "test-support")]
pub(crate) struct FilmFixtureView<'a> {
    pub usage: ProfileUse,
    pub antihalation: Antihalation,
    pub asset_token: u64,
    pub halation_first_sigma_um: [f32; 3],
    pub halation_primary_amount: [f32; 3],
    pub authored_log_exposure: &'a [f64],
    pub interpolation_log_exposure: &'a [f32],
    pub density_curves_cmy: &'a [f32],
    pub density_curves_layers: [[&'a [f32]; 3]; 3],
    pub channel_density_cmy: &'a [f32],
    pub base_density: &'a [f32],
}

impl FilmOwner {
    pub(crate) fn new(profile: Arc<FilmProfile>) -> Self {
        Self { profile }
    }

    #[cfg(feature = "test-support")]
    pub(crate) fn fixture_view(&self) -> FilmFixtureView<'_> {
        let tables = self.profile.tables();
        let processing_defaults = self.profile.processing_defaults();
        FilmFixtureView {
            usage: self.profile.info().usage(),
            antihalation: self.profile.info().antihalation(),
            asset_token: self.profile.asset_token(),
            halation_first_sigma_um: processing_defaults.halation_first_sigma_um,
            halation_primary_amount: processing_defaults.halation_primary_amount,
            authored_log_exposure: tables.authored_log_exposure(),
            interpolation_log_exposure: tables.interpolation_log_exposure(),
            density_curves_cmy: tables.density_curves().as_flattened(),
            density_curves_layers: std::array::from_fn(|layer| {
                std::array::from_fn(|channel| {
                    tables.density_curves_layers()[layer][channel].as_slice()
                })
            }),
            channel_density_cmy: tables.channel_density().as_flattened(),
            base_density: tables.base_density(),
        }
    }
}

pub(crate) struct IlluminantView<'a> {
    pub label: &'a str,
    pub kind: IlluminantKind,
}

pub(crate) struct ProfileTablesView<'a> {
    pub linear_sensitivity_rgb: &'a [f32],
    pub channel_density_cmy: &'a [f32],
    pub base_density: &'a [f32],
    pub interpolation_log_exposure: &'a [f32],
    pub density_curves_cmy: &'a [f32],
}

impl<'a> ProfileTablesView<'a> {
    fn new(tables: &'a ProfileTables) -> Self {
        Self {
            linear_sensitivity_rgb: tables.linear_sensitivity().as_flattened(),
            channel_density_cmy: tables.channel_density().as_flattened(),
            base_density: tables.base_density(),
            interpolation_log_exposure: tables.interpolation_log_exposure(),
            density_curves_cmy: tables.density_curves().as_flattened(),
        }
    }
}

pub(crate) struct FilmView<'a> {
    pub stock: &'a str,
    pub reference_illuminant: IlluminantView<'a>,
    pub viewing_illuminant: IlluminantView<'a>,
    pub tables: ProfileTablesView<'a>,
    pub wavelengths: &'a [f32],
    pub density_curves_layers: [[&'a [f32]; 3]; 3],
    pub processing_defaults: &'a FilmProcessingDefaults,
    pub hanatos_window: &'a [f32],
    pub hanatos_surface_rgb: &'a [f32],
    pub asset_token: u64,
    pub support: Support,
    pub stage: Stage,
    pub polarity: Polarity,
}

impl FilmOwner {
    pub(crate) fn view(&self) -> FilmView<'_> {
        let info = self.profile.info();
        let tables = self.profile.tables();
        FilmView {
            stock: info.stock(),
            reference_illuminant: IlluminantView {
                label: info.reference_illuminant(),
                kind: info.reference_illuminant_kind(),
            },
            viewing_illuminant: IlluminantView {
                label: info.viewing_illuminant(),
                kind: info.viewing_illuminant_kind(),
            },
            tables: ProfileTablesView::new(tables),
            wavelengths: tables.wavelengths(),
            density_curves_layers: std::array::from_fn(|layer| {
                std::array::from_fn(|channel| {
                    tables.density_curves_layers()[layer][channel].as_slice()
                })
            }),
            processing_defaults: self.profile.processing_defaults(),
            hanatos_window: tables
                .hanatos2025_adaptation_window_params()
                .map_or(&[], |values| values.as_slice()),
            hanatos_surface_rgb: tables
                .hanatos2025_adaptation_surface_params()
                .map_or(&[], |values| values.as_flattened()),
            asset_token: self.profile.asset_token(),
            support: info.support(),
            stage: info.stage(),
            polarity: info.polarity(),
        }
    }
}

pub(crate) struct PrintOwner {
    profile: Arc<PrintProfile>,
}

pub(crate) struct PrintView<'a> {
    pub stock: &'a str,
    pub viewing_illuminant: IlluminantView<'a>,
    pub tables: ProfileTablesView<'a>,
    pub asset_token: u64,
    pub stage: Stage,
}

impl PrintOwner {
    pub(crate) fn new(profile: Arc<PrintProfile>) -> Self {
        Self { profile }
    }

    pub(crate) fn view(&self) -> PrintView<'_> {
        let info = self.profile.info();
        PrintView {
            stock: info.stock(),
            viewing_illuminant: IlluminantView {
                label: info.viewing_illuminant(),
                kind: info.viewing_illuminant_kind(),
            },
            tables: ProfileTablesView::new(self.profile.tables()),
            asset_token: self.profile.asset_token(),
            stage: info.stage(),
        }
    }

    pub(crate) fn sample_density_curves(
        &self,
        gamma: f64,
    ) -> Result<PrintDensityCurves, PrintDensityError> {
        self.profile.sample_density_curves(gamma)
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use std::path::Path;

    use film_juicer_core::assets::Assets;

    use super::*;

    #[test]
    fn projection_borrows_exact_completed_fields_through_cache_release() {
        let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let profile = assets.film("kodak_portra_400").unwrap();
        let weak = Arc::downgrade(&profile);
        let owner = FilmOwner::new(Arc::clone(&profile));
        assets.release_cached_payloads().unwrap();
        drop(assets);
        let view = owner.fixture_view();
        let tables = profile.tables();
        assert_eq!(view.usage, ProfileUse::Still);
        assert_eq!(view.antihalation, Antihalation::Strong);
        assert_eq!(view.asset_token, profile.asset_token());
        assert_eq!(
            view.halation_first_sigma_um.map(f32::to_bits),
            profile
                .processing_defaults()
                .halation_first_sigma_um
                .map(f32::to_bits)
        );
        assert_eq!(
            view.halation_primary_amount.map(f32::to_bits),
            profile
                .processing_defaults()
                .halation_primary_amount
                .map(f32::to_bits)
        );
        assert_eq!(
            view.authored_log_exposure.as_ptr(),
            tables.authored_log_exposure().as_ptr()
        );
        assert_eq!(
            view.authored_log_exposure.len(),
            tables.authored_log_exposure().len()
        );
        for (span, expected) in [
            (
                view.interpolation_log_exposure,
                tables.interpolation_log_exposure(),
            ),
            (
                view.density_curves_cmy,
                tables.density_curves().as_flattened(),
            ),
            (
                view.channel_density_cmy,
                tables.channel_density().as_flattened(),
            ),
            (view.base_density, tables.base_density()),
        ] {
            assert_eq!(span.as_ptr(), expected.as_ptr());
            assert_eq!(span.len(), expected.len());
        }
        for layer in 0..3 {
            for channel in 0..3 {
                let expected = &tables.density_curves_layers()[layer][channel];
                let span = view.density_curves_layers[layer][channel];
                assert_eq!(span.as_ptr(), expected.as_ptr());
                assert_eq!(span.len(), expected.len());
            }
        }
        drop(profile);
        assert_eq!(weak.strong_count(), 1);
        drop(owner);
        assert!(weak.upgrade().is_none());
    }
}

#[cfg(test)]
mod production_tests {
    use std::path::Path;
    use film_juicer_core::assets::Assets;
    use super::*;

    #[test]
    fn production_projection_shares_source_and_gamma_owns_its_result() {
        let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let film = assets.film("kodak_portra_400").unwrap();
        let print = assets.print("kodak_portra_endura").unwrap();
        let film_weak = Arc::downgrade(&film);
        let print_weak = Arc::downgrade(&print);
        let film_owner = FilmOwner::new(Arc::clone(&film));
        let print_owner = PrintOwner::new(Arc::clone(&print));
        assert_eq!(
            film_owner.view().tables.interpolation_log_exposure.as_ptr(),
            film.tables().interpolation_log_exposure().as_ptr()
        );
        assert_eq!(
            print_owner
                .view()
                .tables
                .interpolation_log_exposure
                .as_ptr(),
            print.tables().interpolation_log_exposure().as_ptr()
        );
        let curves = print_owner.sample_density_curves(1.1).unwrap();
        assert_ne!(
            curves.totals().as_ptr(),
            print.tables().density_curves().as_ptr()
        );
        let hash = curves.hash();
        assets.release_cached_payloads().unwrap();
        drop(assets);
        drop(film);
        drop(print);
        assert_eq!(film_weak.strong_count(), 1);
        assert_eq!(print_weak.strong_count(), 1);
        assert!(
            !film_owner
                .view()
                .tables
                .interpolation_log_exposure
                .is_empty()
        );
        drop(film_owner);
        drop(print_owner);
        assert!(film_weak.upgrade().is_none());
        assert!(print_weak.upgrade().is_none());
        assert_eq!(curves.hash(), hash);
        assert!(!curves.totals().is_empty());
        println!(
            "Safe retained owner inline: film={} print={}; Arc payload shared, zero warm table copy",
            std::mem::size_of::<FilmOwner>(),
            std::mem::size_of::<PrintOwner>()
        );
    }
}
