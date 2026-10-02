//! Retained film payload and borrow-tied projection shared by asset boundaries.

#![forbid(unsafe_code)]

use std::sync::Arc;

use film_juicer_core::profile::{Antihalation, FilmProfile, ProfileUse};

pub(crate) struct FilmOwner {
    profile: Arc<FilmProfile>,
}

pub(crate) struct FilmView<'a> {
    pub usage: ProfileUse,
    pub antihalation: Antihalation,
    pub asset_token: u64,
    pub halation_first_sigma_um: [f32; 3],
    pub halation_primary_amount: [f32; 3],
    pub source_log_exposure: &'a [f64],
    pub log_exposure: &'a [f32],
    pub density_curves_cmy: &'a [f32],
    pub density_curves_layers: [[&'a [f32]; 3]; 3],
    pub channel_density_cmy: &'a [f32],
    pub base_density: &'a [f32],
}

impl FilmOwner {
    pub(crate) fn new(profile: Arc<FilmProfile>) -> Self {
        Self { profile }
    }

    pub(crate) fn view(&self) -> FilmView<'_> {
        let tables = self.profile.tables();
        let digest = self.profile.digest();
        FilmView {
            usage: self.profile.info().usage(),
            antihalation: self.profile.info().antihalation(),
            asset_token: self.profile.asset_token(),
            halation_first_sigma_um: digest.halation_first_sigma_um,
            halation_primary_amount: digest.halation_primary_amount,
            source_log_exposure: tables.source_log_exposure(),
            log_exposure: tables.log_exposure(),
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

#[cfg(test)]
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
        let view = owner.view();
        let tables = profile.tables();
        assert_eq!(view.usage, ProfileUse::Still);
        assert_eq!(view.antihalation, Antihalation::Strong);
        assert_eq!(view.asset_token, profile.asset_token());
        assert_eq!(
            view.halation_first_sigma_um.map(f32::to_bits),
            profile.digest().halation_first_sigma_um.map(f32::to_bits)
        );
        assert_eq!(
            view.halation_primary_amount.map(f32::to_bits),
            profile.digest().halation_primary_amount.map(f32::to_bits)
        );
        assert_eq!(
            view.source_log_exposure.as_ptr(),
            tables.source_log_exposure().as_ptr()
        );
        assert_eq!(
            view.source_log_exposure.len(),
            tables.source_log_exposure().len()
        );
        for (span, expected) in [
            (view.log_exposure, tables.log_exposure()),
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
