//! Film policy, spectral reconstruction and synthetic exposure reference.

use std::sync::Arc;

use super::foundation::{FoundationInput, build_tables, profile_illuminant};
use super::{
    ControlError, FilmGrainFootprintKind, FilmGrainFrame, FilmReconstruction, PrepareError,
    ReconstructionStorage,
};
use crate::assets::{Assets, FilmGrainStock, InputHull, SpectraLut};
use crate::capture::{HalationReflectionSource, Params, current_grain};
use crate::film::{density, dir, grain, input, reference};
use crate::{color, effects, exposure, gamut, optics, reconstruction, spectral};

fn input_params(params: &Params) -> Result<input::Params, PrepareError> {
    let input_space = match params.input_color_space {
        0 => color::InputSpace::DavinciWideGamut,
        1 => color::InputSpace::Bt2020,
        2 => color::InputSpace::Aces2065_1,
        3 => color::InputSpace::SrgbRec709,
        _ => return Err(ControlError::InputColorSpace.into()),
    };
    let method = match params.spectral_upsampling_mode {
        0 => input::Method::Hanatos2025,
        1 => input::Method::Mallett2019,
        2 => input::Method::Arctic2026Beta04,
        _ => return Err(ControlError::ReconstructionMethod.into()),
    };
    let meter = match params.camera_metering_method {
        0 => input::Meter::CenterWeighted,
        1 => input::Meter::Average,
        2 => input::Meter::Median,
        3 => input::Meter::Partial,
        4 => input::Meter::Matrix,
        5 => input::Meter::MultiZone,
        6 => input::Meter::HighlightWeighted,
        _ => return Err(ControlError::Meter.into()),
    };
    Ok(input::Params {
        input_space,
        method,
        meter,
        decode_cctf: params.input_cctf_decoding != 0,
        auto_exposure_enabled: params.camera_auto_exposure_enabled != 0,
        manual_exposure_ev: params.camera_exposure_compensation_ev as f32,
        film_format_long_edge_mm: params.camera_film_format_long_edge_mm,
        filter_override: params.camera_filter_override,
        uv: params.camera_filter_uv,
        ir: params.camera_filter_ir,
        apply_hanatos_window: params.hanatos2025_adaptation_window != 0,
        apply_hanatos_surface: params.hanatos2025_adaptation_surface != 0,
        input_compression: params.input_compression_enabled != 0,
    })
}

fn complete_reconstruction(
    input: &input::Recipe,
    tables: &spectral::Tables,
    tc_source: Option<Arc<SpectraLut>>,
    mallett: Option<Arc<[[f32; 3]; 81]>>,
    hull: Option<&InputHull>,
) -> Result<FilmReconstruction, PrepareError> {
    let storage = if matches!(input.method(), input::Method::Mallett2019) {
        let basis = mallett.expect("Mallett source acquired before composition");
        let midgray = exposure::mallett_midgray(exposure::MallettInput {
            color: input.color().to_conversion(),
            xyz_to_linear_srgb: *input.color().xyz_to_linear_srgb(),
            basis_rgb: &basis,
            illuminant: tables.illuminant(),
            sensitivity_rgb: input.sensitivity().values_rgb(),
        });
        ReconstructionStorage::Mallett { basis, midgray }
    } else {
        let source = tc_source.expect("TC source acquired before composition");
        let lut = reconstruction::build_tc_lut(reconstruction::Input {
            spectra: source
                .samples()
                .try_into()
                .expect("Assets establishes reconstruction shape"),
            sensitivity_rgb: input.sensitivity().values_rgb(),
            reference_illuminant: tables.illuminant(),
            projection_white_xyz: *input.color().projection_white_xyz(),
            method: if matches!(input.method(), input::Method::Arctic2026Beta04) {
                reconstruction::Method::Arctic
            } else {
                reconstruction::Method::Hanatos {
                    spectral_blur: input.hanatos_spectral_blur(),
                    surface_rgb: input
                        .apply_hanatos_surface()
                        .then(|| input.hanatos_surface_rgb()),
                }
            },
            input_compression_active: input.input_compression_active(),
            input_hull: hull
                .map(|entry| gamut::InputHull::new(*entry.hull().center(), entry.hull().xy())),
        })
        .map_err(PrepareError::Tc)?;
        let xyz = color::project_linear_rgb_to_xyz(
            [0.184; 3],
            *input.color().rgb_to_xyz(),
            *input.color().xyz_adapt(),
        );
        let raw_midgray_rgb =
            reconstruction::sample_tc_lut(lut.samples(), xyz).map_err(PrepareError::Tc)?;
        let normalization = exposure::tc_midgray(raw_midgray_rgb[1]);
        ReconstructionStorage::Tc {
            source,
            lut,
            raw_midgray_rgb,
            normalization,
        }
    };
    Ok(FilmReconstruction { storage })
}

pub(super) fn prepare_grain(
    assets: &Assets,
    params: &Params,
    frame: Option<FilmGrainFrame>,
) -> Result<Option<grain::film_v1::Recipe>, PrepareError> {
    let Some(controls) = params.film_grain.controls() else {
        return Ok(None);
    };
    let stock =
        FilmGrainStock::for_profile(&params.film_profile_key).ok_or(ControlError::FilmGrain)?;
    let long_edge_mm = params.camera_film_format_long_edge_mm;
    if !controls.valid(long_edge_mm, params.scan_route) {
        return Err(ControlError::FilmGrain.into());
    }
    // Keep the authored selection for current_grain(), which disables
    // Current while Film is selected, including on a grain-free thumbnail.
    let raster = match frame {
        Some(FilmGrainFrame::Thumbnail) => return Ok(None),
        Some(FilmGrainFrame::Raster(raster)) => raster,
        None => return Err(PrepareError::FilmGrainGeometry),
    };
    let (rendered_texels, sample_texels, sampling) = raster.sampling(controls, long_edge_mm);
    for (kind, width_texels, minimum) in [
        (
            FilmGrainFootprintKind::Rendered,
            rendered_texels,
            crate::assets::grain::MIN_RENDERED_FOOTPRINT,
        ),
        (
            FilmGrainFootprintKind::Sample,
            sample_texels,
            crate::assets::grain::MIN_PROXY_SAMPLE_FOOTPRINT,
        ),
    ] {
        let supported_texels = minimum..=crate::assets::grain::MAX_FOOTPRINT;
        if !supported_texels.contains(&width_texels) {
            return Err(PrepareError::FilmGrainFootprint {
                kind,
                long_edge_mm,
                texel_pitch_um: controls.texel_pitch_um,
                width_texels,
                supported_texels,
            });
        }
    }
    let bank = assets.film_grain_bank(stock)?;
    let table = assets.film_grain_table(&bank, sample_texels)?;
    Ok(Some(grain::film_v1::Recipe::new(
        bank,
        table,
        controls,
        raster,
        rendered_texels,
        sampling,
    )))
}

pub(super) struct FilmStage {
    pub(super) input: input::Recipe,
    pub(super) optics: optics::Recipe,
    pub(super) minimum: density::Minimum,
    pub(super) grain: grain::Recipe,
    pub(super) effects: effects::Recipe,
    pub(super) dir: dir::Recipe,
    pub(super) film_v1_bounds: Option<density::Bounds>,
}

pub(super) fn prepare(
    acquired: &FoundationInput,
    params: &Params,
    print_domain: bool,
    film_grain: Option<&grain::film_v1::Recipe>,
) -> Result<FilmStage, PrepareError> {
    let film = acquired.development.profile();
    let reference_samples = acquired
        .reference_samples
        .as_ref()
        .ok_or(ControlError::ReferenceIlluminant)?;
    let input = input::Recipe::new(
        film,
        input_params(params)?,
        input::Resources {
            reference_illuminant: reference_samples,
            reference_white_xyz: acquired.reference_white.as_ref(),
            integrated_d65_xyz: acquired.integrated_d65.as_ref(),
            reconstructed_reference_white: acquired
                .reconstructed_white
                .as_ref()
                .map(|white| white.samples()),
            tc_source_hash: acquired
                .tc_source
                .as_ref()
                .map_or(0, |source| source.asset_hash()),
            input_hull_hash: acquired.hull.as_ref().map_or(0, |hull| hull.hull().hash()),
        },
    )
    .map_err(PrepareError::Input)?;
    let defaults = film.processing_defaults();
    let controls = params.scatter_halation;
    let reflection = if controls.active {
        controls.reflection
    } else {
        HalationReflectionSource::Profile
    };
    let (first_sigma_um, primary_amount) = reflection
        .resolve(
            defaults.halation_first_sigma_um,
            defaults.halation_primary_amount,
        )
        .map_err(PrepareError::Optics)?;
    let optics = optics::Recipe::new(optics::Params {
        camera_diffusion: params.camera_diffusion,
        camera_lens_blur_um: 0.0,
        scatter_halation: optics::HalationParams {
            active: controls.active,
            scatter_amount: controls.scatter_amount,
            scatter_spatial_scale: controls.scatter_spatial_scale,
            halation_amount: controls.halation_amount,
            halation_spatial_scale: controls.halation_spatial_scale,
            first_sigma_um,
            primary_amount,
            shape: controls.shape,
            legacy_reflection: crate::profile::is_legacy_halation(first_sigma_um, primary_amount),
        },
        highlight_boost: optics::highlight_boost::Recipe::new(params.highlight_boost)
            .map_err(PrepareError::HighlightBoost)?,
        enlarger_diffusion: params.enlarger_diffusion,
        print_domain,
    })
    .map_err(PrepareError::Optics)?;
    let minimum = density::Minimum::new([0.07, 0.08, 0.12]).map_err(PrepareError::Minimum)?;
    let grain = grain::Recipe::new(&acquired.development, current_grain(params))
        .map_err(PrepareError::Grain)?;
    let effects = effects::Recipe::new(effects::Params {
        film_dust_amount: params.film_dust_amount,
        film_scratch_amount: params.film_scratch_amount,
        gate_dust_amount: params.gate_dust_amount,
        gate_scratch_amount: params.gate_scratch_amount,
        gate_weave_amount: params.gate_weave_amount,
        grain_active: grain.active(),
        grain_debug_view: grain.debug_view(),
    })
    .map_err(PrepareError::Effects)?;
    let dir = dir::Recipe::new(&acquired.development, params.dir_couplers).map_err(|error| {
        PrepareError::Dir {
            error,
            route: params.scan_route,
            stock: film.info().stock().to_owned(),
        }
    })?;
    let film_v1_bounds = film_grain
        .map(|grain| grain.density_bounds(&acquired.development))
        .transpose()
        .map_err(PrepareError::Bounds)?;

    Ok(FilmStage {
        input,
        optics,
        minimum,
        grain,
        effects,
        dir,
        film_v1_bounds,
    })
}

pub(super) struct ExposureStage {
    pub(super) tables: spectral::Tables,
    pub(super) s_inverse: [f32; 9],
    pub(super) reconstruction: FilmReconstruction,
}

pub(super) fn prepare_exposure(
    acquired: &mut FoundationInput,
    input: &input::Recipe,
) -> Result<ExposureStage, PrepareError> {
    let film = acquired.development.profile();
    let film_illuminant = profile_illuminant(
        &acquired.illuminants,
        film.info().reference_illuminant(),
        film.info().reference_illuminant_kind(),
    )?
    .ok_or(ControlError::FilmReferenceIlluminant)?;
    let film_white = match spectral::integrate_white(acquired.observer.borrow(), &film_illuminant) {
        Ok(white) => white,
        Err(_) => {
            return Err(ControlError::FilmReferenceIlluminant.into());
        }
    };
    let exposure_tables = build_tables(
        film.tables(),
        &film_illuminant,
        &film_white,
        &acquired.observer,
    );
    if exposure_tables.hash() == 0
        || !exposure_tables.white_xyz().iter().all(|v| v.is_finite())
        || exposure_tables.white_xyz()[1] <= 0.0
    {
        return Err(ControlError::ExposureTables.into());
    }
    let weights = exposure_tables.weighted_xyz();
    let s_inverse = spectral::s_inverse([&weights[0], &weights[1], &weights[2]]);
    let reconstruction = complete_reconstruction(
        input,
        &exposure_tables,
        acquired.tc_source.take(),
        acquired.mallett.take(),
        acquired.hull.as_deref(),
    )?;

    Ok(ExposureStage {
        tables: exposure_tables,
        s_inverse,
        reconstruction,
    })
}

pub(super) fn prepare_reference(
    development: &crate::film::Development,
    input: &input::Recipe,
    exposure: &ExposureStage,
) -> Result<reference::Reference, PrepareError> {
    let reconstruction = &exposure.reconstruction;
    let film_illuminant = exposure.tables.illuminant();
    let reference_input = match &reconstruction.storage {
        ReconstructionStorage::Tc { lut, .. } => reference::Reconstruction::Tc {
            samples: lut.samples(),
            projection_white_xyz: *input.color().projection_white_xyz(),
        },
        ReconstructionStorage::Mallett { basis, .. } => reference::Reconstruction::Mallett {
            basis_rgb: basis,
            illuminant: film_illuminant,
            sensitivity_rgb: input.sensitivity().values_rgb(),
            green_scale: input.sensitivity().mallett_green_scale(),
        },
    };
    reference::Reference::new(
        development,
        reference::Params {
            manual_exposure_ev: input.manual_exposure_ev(),
            auto_exposure_enabled: input.auto_exposure_enabled(),
        },
        Some(reference_input),
    )
    .map_err(PrepareError::Reference)
}
