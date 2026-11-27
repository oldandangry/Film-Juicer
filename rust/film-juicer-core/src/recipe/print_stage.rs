//! Print-route calibration, filtered illumination, balance and binding.

use std::sync::Arc;

use super::foundation::named_illuminant;
use super::scanner_stage::{ScannerStage, correction_tables};
use super::{
    Calibration, CalibrationError, ControlError, Preflash, PrepareError, PrintState, route_tag,
    update_floats,
};
use crate::assets::{AssetError, Assets, Illuminants};
use crate::capture::Params;
use crate::data_io::calibration;
use crate::film::reference;
use crate::profile::{FilmProfile, PrintDensityCurves, PrintProfile};
use crate::route::ScanRoute;
use crate::{hash, print, scanner};

pub(super) fn enlarger_illuminant(choice: i32) -> &'static str {
    match choice {
        0 => "D65",
        1 => "D55",
        2 => "D50",
        4 => "TH-KG3-L",
        5 => "T",
        6 => "K75P",
        7 => "EQUAL",
        _ => "TH-KG3",
    }
}
pub(super) fn acquire_calibration(
    assets: &Assets,
    film: &FilmProfile,
    print: &PrintProfile,
    label: &str,
) -> Result<Calibration, PrepareError> {
    let outcome = match assets.neutral_calibration() {
        Ok(calibration) => calibration.lookup(print.info().stock(), label, film.info().stock()),
        Err(AssetError::Calibration(error)) => match error.kind() {
            calibration::ErrorKind::MissingFile => return Ok(Calibration::MissingFile),
            calibration::ErrorKind::Read(std::io::ErrorKind::OutOfMemory) => {
                return Err(AssetError::Calibration(error).into());
            }
            _ => return Err(PrepareError::Calibration(CalibrationError::Resource(error))),
        },
        Err(error) => return Err(error.into()),
    };
    match outcome {
        Ok(Some(cc)) => Ok(Calibration::Calibrated(cc)),
        Ok(None) => Ok(Calibration::MissingEntry),
        Err(error) => Err(PrepareError::Calibration(CalibrationError::Entry(error))),
    }
}

pub(super) fn route_prechecks(
    params: &Params,
    print: Option<&PrintProfile>,
    calibration: Option<Calibration>,
) -> Result<(), PrepareError> {
    let scanner = [
        params.scanner_black_level,
        params.scanner_white_level,
        params.scanner_lens_blur_sigma_px,
        params.scanner_unsharp_mask[0],
        params.scanner_unsharp_mask[1],
    ];
    if let Some(print) = print {
        if !params.print_gamma_factor.is_finite()
            || !(0.5..=2.0).contains(&params.print_gamma_factor)
        {
            return Err(ControlError::PrintGamma.into());
        }
        if print.info().stage() != crate::profile::Stage::Printing {
            return Err(ControlError::PrintStage.into());
        }
        let neutral = match calibration {
            Some(Calibration::Calibrated(cc)) => cc,
            _ => print::CURRENT_NEUTRAL_CMY_CC,
        };
        if neutral.iter().any(|v| !v.is_finite())
            || scanner
                .iter()
                .chain(&params.print_ui_ymc_cc)
                .chain(&[
                    params.preflash_m_filter_cc,
                    params.preflash_y_filter_cc,
                    params.print_exposure,
                    params.print_preflash_exposure,
                    params.camera_exposure_compensation_ev,
                    params.glare_percent,
                    params.glare_roughness,
                    params.glare_blur_sigma_px,
                ])
                .any(|v| !(*v as f32).is_finite())
        {
            return Err(ControlError::PrintInput.into());
        }
    } else if scanner.iter().any(|v| !(*v as f32).is_finite()) {
        return Err(ControlError::DirectScanner.into());
    }
    Ok(())
}

pub(super) struct PrintAssembly {
    pub(super) profile: Arc<PrintProfile>,
    curves: PrintDensityCurves,
    filters: print::Filters,
    exposure: print::Exposure,
    calibration: Calibration,
    neutral_cmy_cc: [f32; 3],
    illuminant_key: &'static str,
    illuminant_hash: u64,
}

pub(super) fn prepare(
    params: &Params,
    profile: Arc<PrintProfile>,
    calibration: Calibration,
) -> Result<PrintAssembly, PrepareError> {
    let curves = profile
        .sample_density_curves(params.print_gamma_factor)
        .map_err(PrepareError::PrintDevelopment)?;
    let neutral = match calibration {
        Calibration::Calibrated(cc) => cc,
        _ => print::CURRENT_NEUTRAL_CMY_CC,
    };
    let filters = print::Filters::new(
        neutral,
        params.print_ui_ymc_cc.map(|value| value as f32),
        params.preflash_m_filter_cc as f32,
        params.preflash_y_filter_cc as f32,
    )
    .map_err(PrepareError::PrintComposition)?;
    let exposure = print::Exposure::new(print::ExposureParams {
        print_exposure: params.print_exposure as f32,
        preflash_exposure: params.print_preflash_exposure as f32,
        camera_exposure_ev: params.camera_exposure_compensation_ev as f32,
        normalize: params.normalize_print_exposure != 0,
        compensate: params.print_exposure_compensation != 0,
    })
    .map_err(PrepareError::PrintComposition)?;
    let illuminant_key = enlarger_illuminant(params.enl_ill);
    Ok(PrintAssembly {
        profile,
        curves,
        filters,
        exposure,
        calibration,
        neutral_cmy_cc: neutral,
        illuminant_key,
        illuminant_hash: hash::bytes(illuminant_key.as_bytes()),
    })
}

pub(super) struct PrintIllumination {
    source: [f32; 81],
    main: print::Illuminant,
}

pub(super) fn illuminate(
    illuminants: &Illuminants,
    print: &PrintAssembly,
) -> Result<PrintIllumination, PrepareError> {
    let source = named_illuminant(illuminants, print.illuminant_key)?
        .ok_or(ControlError::PrintIlluminant)?;
    let main = print::Illuminant::new(print.filters.illumination_params(false), &source)
        .map_err(PrepareError::PrintIlluminant)?;
    Ok(PrintIllumination { source, main })
}

pub(super) fn balance(
    film: &FilmProfile,
    print: &PrintAssembly,
    illumination: &PrintIllumination,
    reference: &reference::Reference,
) -> Result<print::Balance, PrepareError> {
    print::Balance::new(
        print::BalanceParams {
            baseline_density_cmy: *reference.baseline_density_cmy(),
            compensated_density_cmy: *reference.compensated_density_cmy(),
            reference_hash: reference.hash(),
            normalization_mode: print.exposure.normalization_mode(),
        },
        film.tables().base_density(),
        film.tables().channel_density(),
        print.profile.tables().linear_sensitivity(),
        illumination.main.samples(),
    )
    .map_err(PrepareError::PrintBalance)
}

pub(super) fn finish(
    params: &Params,
    film: &FilmProfile,
    print: PrintAssembly,
    illumination: PrintIllumination,
    balance: print::Balance,
    scanner: &ScannerStage,
    bounds: &super::scanner_stage::CorrectionBounds<'_>,
) -> Result<PrintState, PrepareError> {
    let PrintIllumination {
        source: source_illuminant,
        main: main_illuminant,
    } = illumination;
    let print_bounds = bounds.print.as_ref();
    let film_bounds = bounds.film();
    let resource_descriptors = print::ResourceDescriptors::new(print::ResourceDescriptorInput {
        film,
        profile: &print.profile,
        curves: &print.curves,
        filters: &print.filters,
        exposure: &print.exposure,
        illuminant_hash: print.illuminant_hash,
    })
    .map_err(PrepareError::PrintComposition)?;
    let preflash = if print.exposure.params().preflash_exposure > 0.0 {
        let illuminant =
            print::Illuminant::new(print.filters.illumination_params(true), &source_illuminant)
                .map_err(PrepareError::PrintIlluminant)?;
        let raw = Preflash::prepare_raw(
            film.tables().base_density(),
            illuminant.samples(),
            print.profile.tables().linear_sensitivity(),
        )?;
        Some(Preflash { illuminant, raw })
    } else {
        None
    };
    let correction = scanner::Correction::print(
        bounds.params,
        print_bounds.map(|paper_bounds| {
            let film_bounds = film_bounds.expect("active Print correction has film bounds");
            scanner::PrintInput {
                tables: correction_tables(&scanner.tables),
                film_base_density: film.tables().base_density(),
                film_channel_density_cmy: film.tables().channel_density(),
                print_sensitivity_cmy: print.profile.tables().linear_sensitivity(),
                filtered_main_illuminant: main_illuminant.samples(),
                print_base_density: print.profile.tables().base_density(),
                print_log_exposure: print.profile.tables().interpolation_log_exposure(),
                print_density_curves: print.profile.tables().density_curves(),
                minimum_density_cmy: *paper_bounds.minimum_cmy(),
                inverse_span_cmy: *paper_bounds.inv_span_cmy(),
                film_minimum_density_cmy: *film_bounds.minimum_cmy(),
                film_maximum_density_cmy: *film_bounds.maximum_cmy(),
                preflash_raw_cmy: preflash
                    .as_ref()
                    .map_or([0.0; 3], |preflash| *preflash.raw.cmy()),
                normalizer: balance.normalizer(),
                preflash_exposure: print.exposure.params().preflash_exposure,
                medium: 1,
            }
        }),
    )
    .map_err(PrepareError::ScannerCorrection)?;
    let glare_hash = print_glare_hash(
        params.scan_route,
        &params.print_profile_key,
        &scanner.post_effects,
    );
    if scanner.post_effects.fields().glare_active() && glare_hash == 0 {
        return Err(ControlError::Identity(params.scan_route).into());
    }
    let hash = hash::u64s(&[
        print.filters.hash(),
        print.exposure.hash(),
        print.illuminant_hash,
        print.curves.hash(),
        params.print_gamma_factor.to_bits(),
        balance.hash(),
    ]);
    Ok(PrintState {
        key: params.print_profile_key.clone(),
        profile: print.profile,
        gamma: params.print_gamma_factor,
        curves: print.curves,
        calibration: print.calibration,
        neutral_cmy_cc: print.neutral_cmy_cc,
        filters: print.filters,
        exposure: print.exposure,
        illuminant_key: print.illuminant_key,
        illuminant_hash: print.illuminant_hash,
        source_illuminant,
        main_illuminant,
        preflash,
        balance,
        resource_descriptors,
        correction,
        glare_hash,
        hash,
    })
}

pub(super) fn print_glare_hash(route: ScanRoute, key: &str, effects: &scanner::PostEffects) -> u64 {
    let fields = effects.fields();
    if !fields.glare_active() {
        return 0;
    }
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(&mut identity, b"print-glare-v1");
    hash::update_bytes(&mut identity, &[route_tag(route)]);
    hash::update_bytes(&mut identity, &(key.len() as u64).to_le_bytes());
    hash::update_bytes(&mut identity, key.as_bytes());
    hash::update_bytes(&mut identity, &[1]);
    update_floats(&mut identity, &fields.values()[..3]);
    identity
}
