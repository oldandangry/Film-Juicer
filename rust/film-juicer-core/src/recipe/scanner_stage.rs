//! Spectral scanner construction and route-dependent correction inputs.

use super::film_stage::FilmStage;
use super::foundation::{FoundationInput, build_tables, profile_illuminant};
use super::{
    ControlError, DirectState, OutputGamut, PrepareError, ScannerOutput, ScannerParams,
    polarity_tag, print_route, route_tag, update_floats,
};
use crate::assets::Assets;
use crate::capture::Params;
use crate::film::{density, reference};
use crate::profile::{Polarity, PrintProfile};
use crate::route::ScanRoute;
use crate::{color, gamut, hash, print, scanner, spectral};

pub(super) const OUTPUT_KNEE: [f32; 3] = [0.95, 1.0, 1.6];

pub(super) fn output_policy(
    params: &Params,
) -> Result<(gamut::OutputTransform, u64), PrepareError> {
    let space = color::OutputSpace::try_from(params.output_color_space as u32)
        .map_err(|_| ControlError::OutputColorSpace)?;
    let output = color::output_color(space);
    let transform = gamut::output_transform(gamut::TransformInput {
        space,
        rgb_to_xyz: output.rgb_to_xyz(),
        xyz_to_rgb: output.xyz_to_rgb(),
        white_xyz: output.white_xyz(),
        d65_white_xyz: &color::input_matrices(color::InputSpace::SrgbRec709).d65_white_xyz,
    })
    .map_err(PrepareError::OutputTransform)?;
    let mut hash = hash::FNV_OFFSET;
    hash::update_bytes(
        &mut hash,
        &[u8::from(params.output_gamut_compression_enabled != 0)],
    );
    update_floats(&mut hash, &OUTPUT_KNEE);
    update_floats(&mut hash, &OUTPUT_KNEE);
    hash::update_bytes(&mut hash, &params.output_color_space.to_le_bytes());
    hash::update_bytes(&mut hash, &transform.contract_hash().to_le_bytes());
    Ok((transform, hash))
}
pub(super) fn scanner_params(
    params: &Params,
    space: color::OutputSpace,
    correction: scanner::Params,
) -> ScannerParams {
    let print = print_route(params.scan_route);
    ScannerParams {
        output_space: space,
        encode_cctf: params.output_cctf_encoding != 0,
        black_correction: correction.black_correction,
        white_correction: correction.white_correction,
        black_level: correction.black_level,
        white_level: correction.white_level,
        glare_active: print && params.glare_active,
        glare_percent: if print {
            params.glare_percent as f32
        } else {
            0.0
        },
        glare_roughness: if print {
            params.glare_roughness as f32
        } else {
            0.0
        },
        glare_blur_sigma_px: if print {
            params.glare_blur_sigma_px as f32
        } else {
            0.0
        },
        lens_blur_sigma_px: params.scanner_lens_blur_sigma_px as f32,
        unsharp_sigma_px: params.scanner_unsharp_mask[0] as f32,
        unsharp_amount: params.scanner_unsharp_mask[1] as f32,
    }
}
fn scanner_recipe_hash(
    params: ScannerParams,
    route: ScanRoute,
    polarity: Polarity,
    viewing: &str,
    raw_digest: u64,
    reference_hash: u64,
    gamut_hash: u64,
) -> u64 {
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(
        &mut identity,
        &[
            route_tag(route),
            u8::from(print_route(route)),
            polarity_tag(polarity),
        ],
    );
    hash::update_bytes(&mut identity, viewing.as_bytes());
    hash::update_bytes(&mut identity, &raw_digest.to_le_bytes());
    hash::update_bytes(&mut identity, &(params.output_space as i32).to_le_bytes());
    hash::update_bytes(
        &mut identity,
        &[
            u8::from(params.encode_cctf),
            u8::from(params.black_correction),
            u8::from(params.white_correction),
        ],
    );
    update_floats(&mut identity, &[params.black_level, params.white_level]);
    hash::update_bytes(&mut identity, &[u8::from(params.glare_active)]);
    update_floats(
        &mut identity,
        &[
            params.glare_percent,
            params.glare_roughness,
            params.glare_blur_sigma_px,
            params.lens_blur_sigma_px,
            params.unsharp_sigma_px,
            params.unsharp_amount,
        ],
    );
    hash::update_bytes(&mut identity, &reference_hash.to_le_bytes());
    hash::update_bytes(&mut identity, &gamut_hash.to_le_bytes());
    identity
}

pub(super) fn correction_tables(tables: &spectral::Tables) -> scanner::ReferenceTables<'_> {
    scanner::ReferenceTables {
        eps_c: &tables.dyes_cmy()[0],
        eps_m: &tables.dyes_cmy()[1],
        eps_y: &tables.dyes_cmy()[2],
        weighted_y: &tables.weighted_xyz()[1],
        baseline_min: tables
            .has_baseline()
            .then(|| tables.baseline_min().as_slice()),
        sample_count: 81,
        inv_yn: tables.inv_yn(),
    }
}

pub(super) struct CorrectionBounds<'a> {
    required: bool,
    pub(super) params: scanner::Params,
    grain_v2: Option<&'a density::Bounds>,
    correction_film: Option<density::Bounds>,
    pub(super) print: Option<print::Bounds>,
}

impl CorrectionBounds<'_> {
    pub(super) fn film(&self) -> Option<&density::Bounds> {
        self.grain_v2.or(self.correction_film.as_ref())
    }
}

pub(super) fn correction_bounds<'a>(
    acquired: &FoundationInput,
    params: &Params,
    film: &'a FilmStage,
    selected_print: Option<&PrintProfile>,
) -> Result<CorrectionBounds<'a>, PrepareError> {
    // Route composition follows foundation, before focused payload construction.
    let correction_params = scanner::Params {
        route: route_tag(params.scan_route),
        black_correction: params.scanner_black_correction != 0,
        white_correction: params.scanner_white_correction != 0,
        black_level: params.scanner_black_level as f32,
        white_level: params.scanner_white_level as f32,
    };
    let correction_required = if selected_print.is_some() {
        correction_params.print_inputs_required()
    } else {
        correction_params.direct_inputs_required()
    }
    .map_err(PrepareError::ScannerCorrection)?;
    let correction_film_bounds = (correction_required && film.grain_v2_bounds.is_none())
        .then(|| density::Bounds::new(&acquired.development, &film.minimum))
        .transpose()
        .map_err(PrepareError::Bounds)?;
    let print_bounds = if correction_required {
        selected_print
            .map(print::Bounds::new)
            .transpose()
            .map_err(PrepareError::PrintComposition)?
    } else {
        None
    };

    Ok(CorrectionBounds {
        required: correction_required,
        params: correction_params,
        grain_v2: film.grain_v2_bounds.as_ref(),
        correction_film: correction_film_bounds,
        print: print_bounds,
    })
}

pub(super) struct ScannerStage {
    pub(super) params: ScannerParams,
    pub(super) tables: spectral::Tables,
    pub(super) post_effects: scanner::PostEffects,
    color: scanner::Color,
    gamut: OutputGamut,
    raw_digest: u64,
    active_gamut_hash: u64,
}

pub(super) fn prepare(
    assets: &Assets,
    params: &Params,
    acquired: &FoundationInput,
    print: Option<&PrintProfile>,
    transform: gamut::OutputTransform,
    gamut_hash: u64,
    scanner_params: ScannerParams,
) -> Result<ScannerStage, PrepareError> {
    let film = acquired.development.profile();
    let (scanner_profile, viewing, viewing_kind, scanner_key, scanner_asset_token) = match &print {
        Some(print) => (
            print.tables(),
            print.info().viewing_illuminant(),
            print.info().viewing_illuminant_kind(),
            params.print_profile_key.as_str(),
            print.asset_token(),
        ),
        None => (
            film.tables(),
            film.info().viewing_illuminant(),
            film.info().viewing_illuminant_kind(),
            params.film_profile_key.as_str(),
            film.asset_token(),
        ),
    };
    let scanner_illuminant = profile_illuminant(&acquired.illuminants, viewing, viewing_kind)?
        .ok_or(ControlError::ScannerPayload(params.scan_route))?;
    let scanner_white =
        match spectral::integrate_white(acquired.observer.borrow(), &scanner_illuminant) {
            Ok(white) => white,
            Err(_) => {
                return Err(ControlError::ScannerPayload(params.scan_route).into());
            }
        };
    let scanner_tables = build_tables(
        scanner_profile,
        &scanner_illuminant,
        &scanner_white,
        &acquired.observer,
    );
    let active_gamut_hash = if params.output_gamut_compression_enabled != 0 {
        gamut_hash
    } else {
        0
    };
    let scanner_color = scanner::Color::new(scanner::ColorInput {
        medium: u8::from(print.is_some()),
        output_space: scanner_params.output_space as u8,
        apply_cctf: scanner_params.encode_cctf,
        illuminant_xyz: *scanner_white.xyz(),
        illuminant_hash: scanner_white.hash(),
        active_gamut_recipe_hash: active_gamut_hash,
    })
    .map_err(PrepareError::ScannerColor)?;
    let raw_scanner_digest = scanner_color.spectral_binding_digest(
        route_tag(params.scan_route),
        scanner_key,
        scanner_asset_token,
        &scanner_tables,
        params.output_gamut_compression_enabled != 0,
    );
    let boundary = if params.output_gamut_compression_enabled != 0 {
        Some(assets.output_boundary(
            gamut::BoundaryInput {
                space: transform.space(),
                d65_xyz_to_native_rgb: transform.d65_xyz_to_native_rgb(),
                transform_hash: transform.hash(),
            },
            transform.contract_hash(),
        )?)
    } else {
        None
    };
    let gamut = OutputGamut {
        enabled: params.output_gamut_compression_enabled != 0,
        transform,
        boundary,
        hash: gamut_hash,
    };
    let scanner_post_effects = scanner::describe_post_effects(scanner::PostEffectsParams {
        route: params.scan_route,
        glare_active: scanner_params.glare_active,
        glare_percent: scanner_params.glare_percent,
        glare_roughness: scanner_params.glare_roughness,
        glare_blur_sigma_px: scanner_params.glare_blur_sigma_px,
        lens_blur_sigma_px: scanner_params.lens_blur_sigma_px,
        unsharp_sigma_px: scanner_params.unsharp_sigma_px,
        unsharp_amount: scanner_params.unsharp_amount,
    })
    .map_err(PrepareError::ScannerPostEffects)?;

    Ok(ScannerStage {
        params: scanner_params,
        tables: scanner_tables,
        post_effects: scanner_post_effects,
        color: scanner_color,
        gamut,
        raw_digest: raw_scanner_digest,
        active_gamut_hash,
    })
}

pub(super) fn direct(
    acquired: &FoundationInput,
    reference: &reference::Reference,
    scanner: &ScannerStage,
    bounds: &CorrectionBounds<'_>,
) -> Result<DirectState, PrepareError> {
    let film = acquired.development.profile();
    let film_bounds = bounds.film();
    let correction = scanner::Correction::direct(
        bounds.params,
        film_bounds
            .filter(|_| bounds.required)
            .map(|film_bounds| scanner::DirectInput {
                tables: correction_tables(&scanner.tables),
                film_base_density: film.tables().base_density(),
                film_log_exposure: acquired.development.log_exposure(),
                film_authored_density_curves: acquired.development.authored_curves_cmy(),
                minimum_density_cmy: *film_bounds.minimum_cmy(),
                maximum_density_cmy: *film_bounds.maximum_cmy(),
                inverse_span_cmy: *film_bounds.inv_span_cmy(),
                baseline_density_cmy: *reference.baseline_density_cmy(),
                reference_hash: reference.hash(),
                medium: 0,
            }),
    )
    .map_err(PrepareError::ScannerCorrection)?;
    Ok(DirectState { correction })
}

pub(super) fn finish(
    stage: ScannerStage,
    route: ScanRoute,
    polarity: crate::profile::Polarity,
    viewing: &str,
    reference: &reference::Reference,
) -> (ScannerOutput, u64) {
    let ScannerStage {
        params: scanner_params,
        tables: scanner_tables,
        post_effects: scanner_post_effects,
        color: scanner_color,
        gamut,
        raw_digest: raw_scanner_digest,
        active_gamut_hash,
    } = stage;
    let gamut_hash = gamut.hash;
    let scanner_recipe_hash = scanner_recipe_hash(
        scanner_params,
        route,
        polarity,
        viewing,
        raw_scanner_digest,
        reference.hash(),
        gamut_hash,
    );
    let spectral_hash = hash::u64s(&[raw_scanner_digest, active_gamut_hash]);
    let output = ScannerOutput {
        params: scanner_params,
        tables: scanner_tables,
        color: scanner_color,
        post_effects: scanner_post_effects,
        gamut,
        hash: scanner_recipe_hash,
    };

    (output, spectral_hash)
}
