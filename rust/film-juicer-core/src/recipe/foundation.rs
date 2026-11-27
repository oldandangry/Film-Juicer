//! Profile-derived resources and the spectral observer shared by both routes.

use std::sync::Arc;

use super::{ControlError, PrepareError};
use crate::assets::{AssetError, Assets, Illuminants, InputHull, SpectraLut};
use crate::capture::{Params, current_grain};
use crate::profile::{FilmProfile, IlluminantKind};
use crate::{film, illuminant, reconstruction, spectral};

pub(super) struct Observer {
    xyz: [[f32; 81]; 3],
}
impl Observer {
    fn acquire(assets: &Assets) -> Result<Self, PrepareError> {
        let rows = assets.cmf()?;
        if rows.rows().len() != 81 {
            return Err(ControlError::ExposureTables.into());
        }
        Ok(Self {
            xyz: std::array::from_fn(|c| std::array::from_fn(|i| rows.rows()[i][c + 1])),
        })
    }
    pub(super) fn borrow(&self) -> spectral::Observer<'_> {
        spectral::Observer {
            x: &self.xyz[0],
            y: &self.xyz[1],
            z: &self.xyz[2],
        }
    }
}

pub(super) fn named_illuminant(
    curves: &Illuminants,
    label: &str,
) -> Result<Option<[f32; 81]>, PrepareError> {
    let key = normalize_illuminant(label);
    Ok(match key.as_str() {
        "D65" => curves.d65().copied(),
        "D55" => curves.d55().copied(),
        "D50" => curves.d50().copied(),
        "TH-KG3-L" | "THKG3L" | "TH-KG3L" => curves.tungsten_kg3_lens().copied(),
        "TH-KG3" | "THKG3" => curves.tungsten_kg3().copied(),
        "T" | "INCANDESCENT" => curves.tungsten().copied(),
        "K75P" | "KINOTON75P" => curves.k75p().copied(),
        "EQUAL" | "EQUALENERGY" | "EQUAL-ENERGY" => Some(illuminant::equal_energy()),
        _ => None,
    })
}
pub(super) fn profile_illuminant(
    curves: &Illuminants,
    label: &str,
    kind: IlluminantKind,
) -> Result<Option<[f32; 81]>, PrepareError> {
    match kind {
        IlluminantKind::Blackbody { temperature_kelvin } => {
            Ok(Some(illuminant::blackbody(temperature_kelvin as f32)))
        }
        IlluminantKind::Named => named_illuminant(curves, label),
    }
}
fn foundation_reference(curves: &Illuminants, profile: &FilmProfile) -> Option<[f32; 81]> {
    if let IlluminantKind::Blackbody { temperature_kelvin } =
        profile.info().reference_illuminant_kind()
    {
        return Some(illuminant::blackbody(temperature_kelvin as f32));
    }
    let label = normalize_illuminant(profile.info().reference_illuminant());
    match label.as_str() {
        "D65" => curves.d65().copied(),
        "D55" => curves.d55().copied(),
        "D50" => curves.d50().copied(),
        "T" | "TUNGSTEN" => curves.tungsten().copied(),
        "TH-KG3" | "TUNGSTEN-KG3" => curves.tungsten_kg3_lens().copied(),
        _ => None,
    }
}

fn normalize_illuminant(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut hyphen = true;
    for byte in raw.bytes() {
        if byte.is_ascii_alphanumeric() {
            result.push(byte.to_ascii_uppercase() as char);
            hyphen = false;
        } else if (byte == b'-' || byte == b'_' || byte.is_ascii_whitespace())
            && !result.is_empty()
            && !hyphen
        {
            result.push('-');
            hyphen = true;
        }
    }
    while result.ends_with('-') {
        result.pop();
    }
    if result.is_empty() {
        result = raw
            .chars()
            .filter(|c| !c.is_ascii_whitespace())
            .map(|c| c.to_ascii_uppercase())
            .collect();
    }
    result
}

pub(super) struct FoundationInput {
    pub(super) illuminants: Arc<Illuminants>,
    pub(super) development: film::Development,
    pub(super) observer: Observer,
    pub(super) reference_samples: Option<[f32; 81]>,
    pub(super) reference_white: Option<[f32; 3]>,
    pub(super) integrated_d65: Option<[f32; 3]>,
    pub(super) reconstructed_white: Option<reconstruction::ReferenceWhite>,
    pub(super) tc_source: Option<Arc<SpectraLut>>,
    pub(super) mallett: Option<Arc<[[f32; 3]; 81]>>,
    pub(super) hull: Option<Arc<InputHull>>,
}
pub(super) fn acquire_foundation(
    assets: &Assets,
    params: &Params,
    profile: Arc<FilmProfile>,
) -> Result<FoundationInput, PrepareError> {
    let illuminants = assets.illuminants()?;
    let development = film::Development::new(
        profile,
        film::Params {
            gamma_factor: params.film_gamma_factor,
            layers_required: current_grain(params).active && params.grain.sublayers_active,
            requested_route: params.scan_route,
        },
    )
    .map_err(PrepareError::Development)?;
    let profile = development.profile();
    let reference_samples = foundation_reference(&illuminants, profile);
    let observer = Observer::acquire(assets)?;
    let reference_white_samples = profile_illuminant(
        &illuminants,
        profile.info().reference_illuminant(),
        profile.info().reference_illuminant_kind(),
    )?;
    let reference_white = reference_white_samples
        .map(|samples| optional_white(observer.borrow(), &samples))
        .transpose()?
        .flatten();
    let mut integrated_d65 = None;
    let mut reconstructed_white = None;
    let mut tc_source = None;
    let mut mallett = None;
    match params.spectral_upsampling_mode {
        0 => {
            let source = assets.hanatos()?;
            if params.hanatos2025_adaptation_window != 0
                && let Some(white) = reference_white
            {
                let prepared = reconstruction::reference_white(
                    source
                        .samples()
                        .try_into()
                        .expect("Assets establishes reconstruction shape"),
                    profile
                        .processing_defaults()
                        .hanatos_spectral_gaussian_blur_default,
                    white,
                );
                reconstructed_white = match prepared {
                    Ok(white) => Some(white),
                    Err(error @ reconstruction::Error::AllocationFailure) => {
                        return Err(PrepareError::ReferenceWhite(error));
                    }
                    Err(_) => None,
                };
            }
            tc_source = Some(source);
        }
        1 => mallett = Some(assets.mallett()?),
        2 => {
            integrated_d65 = illuminants
                .d65()
                .map(|samples| optional_white(observer.borrow(), samples))
                .transpose()?
                .flatten();
            tc_source = Some(assets.arctic()?);
        }
        _ => (),
    }
    let hull = if matches!(params.spectral_upsampling_mode, 0 | 2)
        && params.input_compression_enabled != 0
    {
        match assets.input_hull() {
            Ok(hull) => Some(hull),
            Err(AssetError::Derived(
                crate::assets::DerivedError::Hull(_) | crate::assets::DerivedError::Resource(_),
            )) => None,
            Err(error) => return Err(error.into()),
        }
    } else {
        None
    };
    Ok(FoundationInput {
        illuminants,
        development,
        observer,
        reference_samples,
        reference_white,
        integrated_d65,
        reconstructed_white,
        tc_source,
        mallett,
        hull,
    })
}

pub(super) fn build_tables(
    profile: &crate::profile::ProfileTables,
    samples: &[f32; 81],
    white: &spectral::White,
    observer: &Observer,
) -> spectral::Tables {
    spectral::build_tables(spectral::Input {
        dyes_cmy: profile.channel_density(),
        observer: observer.borrow(),
        illuminant: samples,
        baseline: Some(spectral::Baseline {
            minimum: profile.base_density(),
            midpoint: None,
        }),
        illuminant_hash: white.hash(),
    })
}

fn optional_white(
    observer: spectral::Observer<'_>,
    illuminant: &[f32; 81],
) -> Result<Option<[f32; 3]>, PrepareError> {
    Ok(spectral::integrate_white(observer, illuminant)
        .ok()
        .map(|white| *white.xyz()))
}
