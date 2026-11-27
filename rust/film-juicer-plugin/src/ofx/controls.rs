//! Current-value control capture and host UI edits.

#![forbid(unsafe_code)]

use film_juicer_core::film::grain;
use film_juicer_core::optics::highlight_boost;
use film_juicer_core::profile::{self, Polarity, Role};
use film_juicer_core::capture::{HalationControls, HalationReflectionSource};
use film_juicer_core::{capture, exposure, film, optics, print};
use film::dir::DirField;

use super::param_id::ParamId;
use super::host::Status;
use super::instance::Controls;

pub(super) enum CaptureError {
    Host(Status),
    Invalid(String),
}
impl From<Status> for CaptureError {
    fn from(status: Status) -> Self {
        Self::Host(status)
    }
}

fn invalid_diagnostic(text: &str) -> CaptureError {
    let mut diagnostic = String::new();
    if diagnostic.try_reserve_exact(text.len()).is_err() {
        return CaptureError::Host(Status::MEMORY);
    }
    diagnostic.push_str(text);
    CaptureError::Invalid(diagnostic)
}

fn diffusion(controls: &Controls<'_>, print: bool) -> Result<optics::DiffusionParams, Status> {
    let names = if print {
        [
            ParamId::PrintDiffusionEnabled,
            ParamId::PrintDiffusionFamily,
            ParamId::PrintDiffusionStrength,
            ParamId::PrintDiffusionSpatialScale,
            ParamId::PrintDiffusionHaloWarmth,
            ParamId::PrintDiffusionCoreIntensity,
            ParamId::PrintDiffusionCoreSize,
            ParamId::PrintDiffusionHaloIntensity,
            ParamId::PrintDiffusionHaloSize,
            ParamId::PrintDiffusionBloomIntensity,
            ParamId::PrintDiffusionBloomSize,
        ]
    } else {
        [
            ParamId::CameraDiffusionEnabled,
            ParamId::CameraDiffusionFamily,
            ParamId::CameraDiffusionStrength,
            ParamId::CameraDiffusionSpatialScale,
            ParamId::CameraDiffusionHaloWarmth,
            ParamId::CameraDiffusionCoreIntensity,
            ParamId::CameraDiffusionCoreSize,
            ParamId::CameraDiffusionHaloIntensity,
            ParamId::CameraDiffusionHaloSize,
            ParamId::CameraDiffusionBloomIntensity,
            ParamId::CameraDiffusionBloomSize,
        ]
    };
    let active = controls.integer(names[0])? != 0;
    let family = controls.integer(names[1])?;
    let mut values = [0.0; 9];
    for (value, name) in values.iter_mut().zip(&names[2..]) {
        *value = controls.double(*name)?;
    }
    Ok(capture::diffusion(active, family, values))
}

fn capture_dir(
    controls: &Controls<'_>,
    positive: bool,
) -> Result<film_juicer_core::film::dir::Params, CaptureError> {
    let active = controls.integer(ParamId::CouplersActive)? != 0;
    let stock = controls.integer(ParamId::CouplersGammaUseStock)? != 0;
    let mut dir = film::dir::capture_defaults(active, stock);
    if !active {
        return Ok(dir);
    }
    let narrow = |field: DirField, raw| field.narrow(raw).map_err(invalid_diagnostic);
    dir.amount = narrow(DirField::Amount, controls.double(ParamId::CouplersAmount)?)?;
    if dir.amount == 0.0 {
        return Ok(dir);
    }
    dir.inhibition_same_layer = narrow(
        DirField::InhibitionSame,
        controls.double(ParamId::CouplersInhibitionSameLayer)?,
    )?;
    dir.inhibition_interlayer = narrow(
        DirField::InhibitionInter,
        controls.double(ParamId::CouplersInhibitionInterlayer)?,
    )?;
    dir.diffusion_size_um = narrow(
        DirField::DiffusionCoreUm,
        controls.double(ParamId::CouplersDiffusionSizeUm)?,
    )?;
    if !stock {
        let same = controls.double3(ParamId::CouplersGammaSameLayerRgb)?;
        let r = controls.double2(ParamId::CouplersGammaInterlayerRToGb)?;
        let g = controls.double2(ParamId::CouplersGammaInterlayerGToRb)?;
        let b = controls.double2(ParamId::CouplersGammaInterlayerBToRg)?;
        for (dst, value) in dir.gamma_same_layer_rgb.iter_mut().zip(same) {
            *dst = narrow(DirField::GammaSame, value)?;
        }
        for i in 0..2 {
            dir.gamma_interlayer_r_to_gb[i] = narrow(DirField::GammaRToGb, r[i])?;
            dir.gamma_interlayer_g_to_rb[i] = narrow(DirField::GammaGToRb, g[i])?;
            dir.gamma_interlayer_b_to_rg[i] = narrow(DirField::GammaBToRg, b[i])?;
        }
    }
    let (name, field, destination) = if positive {
        (
            ParamId::CouplersLangmuirReceiverKRgb,
            DirField::ReceiverK,
            &mut dir.langmuir_receiver_k_rgb,
        )
    } else {
        (
            ParamId::CouplersLangmuirDonorKRgb,
            DirField::DonorK,
            &mut dir.langmuir_donor_k_rgb,
        )
    };
    for (dst, value) in destination.iter_mut().zip(controls.double3(name)?) {
        *dst = narrow(field, value)?;
    }
    if dir.diffusion_size_um > 0.0 {
        dir.diffusion_tail_weight = narrow(
            DirField::DiffusionTailWeight,
            controls.double(ParamId::CouplersDiffusionTailWeight)?,
        )?;
        if dir.diffusion_tail_weight > 0.0 {
            dir.diffusion_tail_um = narrow(
                DirField::DiffusionTailUm,
                controls.double(ParamId::CouplersDiffusionTailUm)?,
            )?;
        }
    }
    Ok(dir)
}

fn invalid_halation(error: optics::Error) -> CaptureError {
    const PREFIX: &str = "InvalidAuthoredControl component=scatter_halation field=";
    let field = error.field().name();
    let channel = match error.channel() {
        Some(0) => " channel=R",
        Some(1) => " channel=G",
        Some(2) => " channel=B",
        _ => "",
    };
    let mut diagnostic = String::new();
    if diagnostic
        .try_reserve_exact(PREFIX.len() + field.len() + channel.len())
        .is_err()
    {
        return CaptureError::Host(Status::MEMORY);
    }
    diagnostic.push_str(PREFIX);
    diagnostic.push_str(field);
    diagnostic.push_str(channel);
    CaptureError::Invalid(diagnostic)
}

fn capture_halation(controls: &Controls<'_>) -> Result<HalationControls, CaptureError> {
    if controls.integer(ParamId::HalationActive)? == 0 {
        return Ok(HalationControls::default());
    }
    let mut halation = capture::halation(
        true,
        [
            controls.double(ParamId::HalationScatterAmount)?,
            controls.double(ParamId::HalationScatterSpatialScale)?,
            controls.double(ParamId::HalationAmount)?,
            controls.double(ParamId::HalationSpatialScale)?,
        ],
    )
    .map_err(invalid_diagnostic)?;
    halation.shape = optics::HalationShape {
        scatter_core_um: controls.double3(ParamId::HalationScatterCoreSigmaUm)?,
        scatter_tail_um: controls.double3(ParamId::HalationScatterTailScaleUm)?,
        scatter_tail_weight: controls.double3(ParamId::HalationScatterTailWeight)?,
        bounce_count: controls.integer(ParamId::HalationBounceCount)?,
        bounce_decay: controls.double(ParamId::HalationBounceDecay)?,
        renormalize: controls.integer(ParamId::HalationRenormalize)? != 0,
    }
    .validate()
    .map_err(invalid_halation)?;
    halation.reflection = match controls.integer(ParamId::HalationReflectionSource)? {
        0 => HalationReflectionSource::Profile,
        1 => HalationReflectionSource::Custom {
            strength: controls.double3(ParamId::HalationStrengthRgb)?,
            first_sigma_um: controls.double3(ParamId::HalationFirstBounceSigmaUm)?,
        },
        _ => {
            return Err(invalid_diagnostic(
                "InvalidAuthoredControl component=scatter_halation field=reflection_source",
            ));
        }
    }
    .validate()
    .map_err(invalid_halation)?;
    Ok(halation)
}

fn capture_highlight_boost(
    controls: &Controls<'_>,
) -> Result<highlight_boost::Controls, CaptureError> {
    let boost_ev =
        highlight_boost::Controls::validate_ev(controls.double(ParamId::HalationHighlightBoostEv)?)
            .map_err(|error| invalid_diagnostic(error.diagnostic()))?;
    if boost_ev == 0.0 {
        return Ok(highlight_boost::Controls::default());
    }
    highlight_boost::Controls {
        boost_ev,
        protection_ev: controls.double(ParamId::HalationHighlightProtectionEv)?,
        range: controls.double(ParamId::HalationHighlightBoostRange)?,
    }
    .validate()
    .map_err(|error| invalid_diagnostic(error.diagnostic()))
}

pub(super) fn capture(controls: &Controls<'_>) -> Result<capture::Captured, CaptureError> {
    let mut p = capture::defaults();
    p.scatter_halation = capture_halation(controls)?;
    p.highlight_boost = capture_highlight_boost(controls)?;
    p.camera_film_format_long_edge_mm =
        film::input::film_format_mm(controls.double(ParamId::CameraFilmFormatMm)?)
            .map_err(invalid_diagnostic)?;
    let film = film_key(controls)?;
    let print = profile::capture_key(
        Role::Print,
        controls.string_bytes(ParamId::PrintProfileKey)?,
    )
    .map_err(|_| Status::MEMORY)?;
    let polarity = positive(controls, &film)?;
    let route = controls.string_bytes(ParamId::ScanRoute)?;
    p.scan_route = capture::route(polarity, &route);
    p.spectral_upsampling_mode = controls.integer(ParamId::SpectralUpsampling)?;
    let reference_illuminant = controls.integer(ParamId::ReferenceIlluminant)?;
    p.enl_ill = controls.integer(ParamId::EnlargerIlluminant)?;
    let film_gamma = controls.double(ParamId::FilmGammaFactor)?;
    let print_gamma = controls.double(ParamId::PrintGammaFactor)?;
    p.film_gamma_factor = film::capture_gamma(film_gamma).map_err(invalid_diagnostic)?;
    p.print_gamma_factor = print::capture_gamma(
        print_gamma,
        matches!(
            p.scan_route,
            film_juicer_core::route::ScanRoute::NegativePrint
                | film_juicer_core::route::ScanRoute::PositivePrint
        ),
    )
    .map_err(invalid_diagnostic)?;
    p.camera_diffusion = diffusion(controls, false)?;
    p.enlarger_diffusion = diffusion(controls, true)?;
    capture::PrintControls {
        exposure: controls.double(ParamId::PrintExposure)?,
        preflash: controls.double(ParamId::PrintPreflash)?,
        exposure_compensation: controls.integer(ParamId::PrintExposureCompensation)? != 0,
        ymc: [
            controls.double(ParamId::EnlargerY)?,
            controls.double(ParamId::EnlargerM)?,
            controls.double(ParamId::EnlargerC)?,
        ],
    }
    .complete(&mut p);
    let shadow = capture::ShadowControls {
        factor: controls.double(ParamId::PrintShadowCompensationFactor)?,
        density: controls.double(ParamId::PrintShadowCompensationDensity)?,
        transition: controls.double(ParamId::PrintShadowCompensationTransition)?,
    };
    capture::GlareControls {
        active: controls.integer(ParamId::GlareActive)? != 0,
        percent: controls.double(ParamId::GlarePercent)?,
        roughness: controls.double(ParamId::GlareRoughness)?,
        blur: controls.double(ParamId::GlareBlurSigmaPx)?,
    }
    .complete(&mut p);
    p.input_color_space = controls.integer(ParamId::InputColorSpace)?;
    p.input_cctf_decoding = i32::from(controls.integer(ParamId::InputCctfDecoding)? != 0);
    p.input_compression_enabled = i32::from(controls.integer(ParamId::InputCompression)? != 0);
    p.hanatos2025_adaptation_window =
        i32::from(controls.integer(ParamId::Hanatos2025AdaptationWindow)? != 0);
    p.hanatos2025_adaptation_surface =
        i32::from(controls.integer(ParamId::Hanatos2025AdaptationSurface)? != 0);
    p.camera_exposure_compensation_ev = exposure::authored_ev(controls.double(ParamId::Exposure)?);
    p.camera_auto_exposure_enabled = i32::from(controls.integer(ParamId::CameraAutoExposure)? != 0);
    p.camera_metering_method = controls.integer(ParamId::CameraMeteringMethod)?;
    p.grain = capture_grain(controls)?;
    {
        use grain::film_v1::{Controls as FilmControls, Selection};
        let model = controls.string_bytes(ParamId::GrainModel)?;
        p.film_grain = match model.as_slice() {
            b"current" => Selection::Current,
            b"film_v1" if !p.grain.active => Selection::FilmV1Disabled,
            b"film_v1" => Selection::FilmV1(FilmControls {
                amount_ev: controls.double(ParamId::FilmGrainAmountEv)? as f32,
                texel_pitch_um: controls.double(ParamId::FilmGrainTexelPitchUm)? as f32,
                chroma_fraction: controls.double(ParamId::FilmGrainChromaFraction)? as f32,
            }),
            _ => return Err(invalid_diagnostic("unsupported grain model")),
        };
    }
    capture::EffectsControls {
        film_dust: controls.double(ParamId::FilmDustAmount)?,
        gate_dust: controls.double(ParamId::GateDustAmount)?,
        film_scratch: controls.double(ParamId::FilmScratchAmount)?,
        gate_scratch: controls.double(ParamId::GateScratchAmount)?,
        gate_weave: controls.double(ParamId::GateWeaveAmount)?,
    }
    .complete(&mut p);
    p.dir_couplers = capture_dir(controls, polarity)?;
    p.scanner_lens_blur_sigma_px = controls.double(ParamId::ScannerLensBlurSigmaPx)?;
    p.scanner_unsharp_mask = controls.double2(ParamId::ScannerUnsharpMask)?;
    p.scanner_black_correction = i32::from(controls.integer(ParamId::ScannerBlackCorrection)? != 0);
    p.scanner_white_correction = i32::from(controls.integer(ParamId::ScannerWhiteCorrection)? != 0);
    let black = controls.double(ParamId::ScannerBlackLevel)?;
    let white = controls.double(ParamId::ScannerWhiteLevel)?;
    (p.scanner_black_level, p.scanner_white_level) = capture::capture_levels(black, white);
    p.output_color_space = controls.integer(ParamId::OutputColorSpace)?;
    p.output_cctf_encoding = i32::from(controls.integer(ParamId::OutputCctfEncoding)? != 0);
    p.output_gamut_compression_enabled =
        i32::from(controls.integer(ParamId::OutputGamutCompression)? != 0);
    Ok(capture::Captured::new(
        p,
        film,
        print,
        reference_illuminant,
        shadow,
    ))
}

fn film_key(controls: &Controls<'_>) -> Result<Vec<u8>, Status> {
    profile::capture_key(Role::Film, controls.string_bytes(ParamId::FilmProfileKey)?)
        .map_err(|_| Status::MEMORY)
}

fn positive(controls: &Controls<'_>, key: &[u8]) -> Result<bool, Status> {
    let catalog = controls.catalog()?;
    Ok(catalog
        .films()
        .iter()
        .find(|entry| entry.key().as_bytes() == key)
        .is_some_and(|entry| entry.polarity() == Polarity::Positive))
}

fn control_route(controls: &Controls<'_>) -> Result<(bool, bool), Status> {
    let key = film_key(controls)?;
    let positive = positive(controls, &key)?;
    let selection = controls.string_bytes(ParamId::ScanRoute)?;
    let print = matches!(
        capture::route(positive, &selection),
        film_juicer_core::route::ScanRoute::NegativePrint
            | film_juicer_core::route::ScanRoute::PositivePrint
    );
    Ok((print, positive))
}

pub(super) fn initialize(controls: &Controls<'_>) -> Result<(), Status> {
    refresh_route_ui(controls)?;
    update_halation(controls)?;
    update_highlight_boost(controls)?;
    update_grain_model(controls)?;
    Ok(())
}

fn refresh_route_ui(controls: &Controls<'_>) -> Result<(), Status> {
    let diffusion_route = control_route(controls)?;
    let gamma_route = control_route(controls)?;
    update_diffusion(controls, diffusion_route.0)?;
    update_gamma(controls, gamma_route.0, gamma_route.1)?;
    update_spectral(controls)
}

fn update_halation(controls: &Controls<'_>) -> Result<(), Status> {
    let active = controls.integer(ParamId::HalationActive)? != 0;
    let custom = active && controls.integer(ParamId::HalationReflectionSource)? == 1;
    for name in [
        ParamId::HalationScatterCoreSigmaUm,
        ParamId::HalationScatterTailScaleUm,
        ParamId::HalationScatterTailWeight,
        ParamId::HalationReflectionSource,
        ParamId::HalationBounceCount,
        ParamId::HalationBounceDecay,
        ParamId::HalationRenormalize,
    ] {
        controls.set_enabled(name, active)?;
    }
    controls.set_enabled(ParamId::HalationStrengthRgb, custom)?;
    controls.set_enabled(ParamId::HalationFirstBounceSigmaUm, custom)
}

fn update_highlight_boost(controls: &Controls<'_>) -> Result<(), Status> {
    let active = controls.double(ParamId::HalationHighlightBoostEv)? > 0.0;
    controls.set_enabled(ParamId::HalationHighlightProtectionEv, active)?;
    controls.set_enabled(ParamId::HalationHighlightBoostRange, active)
}
fn update_grain_model(controls: &Controls<'_>) -> Result<(), Status> {
    let model = controls.string_bytes(ParamId::GrainModel)?;
    let film = match model.as_slice() {
        b"current" => false,
        b"film_v1" => true,
        _ => return Err(Status::FAILED),
    };
    for name in [
        ParamId::FilmGrainAmountEv,
        ParamId::FilmGrainTexelPitchUm,
        ParamId::FilmGrainChromaFraction,
    ] {
        controls.set_secret(name, !film)?;
    }
    for name in [
        ParamId::GrainSublayersActive,
        ParamId::GrainPreset,
        ParamId::GrainAmplitude,
        ParamId::GrainBlur,
        ParamId::GrainSharpness,
        ParamId::GrainChroma,
        ParamId::GrainTexture,
        ParamId::GrainResetAdvanced,
        ParamId::GrainParticleAreaUm2,
        ParamId::GrainParticleScaleMaster,
        ParamId::GrainParticleScaleLayersMaster,
        ParamId::GrainDensityMinMaster,
        ParamId::GrainUniformityMaster,
        ParamId::GrainBlurDyeCloudsUm,
        ParamId::GrainSizeMixWeight,
        ParamId::GrainSizeMixWeightMid,
        ParamId::GrainSizeMixScale,
        ParamId::GrainMicroStructure,
        ParamId::GrainDebugView,
        ParamId::GrainClumpTemporalMix,
        ParamId::GrainClumpMorphPeriodSec,
    ] {
        controls.set_secret(name, film)?;
    }
    Ok(())
}

fn update_grain_chroma(controls: &Controls<'_>) -> Result<(), Status> {
    controls.set_enabled(ParamId::GrainChroma, true)?;
    controls.set_hint(ParamId::GrainChroma, controls.grain_chroma_hint())
}

fn update_grain_label(controls: &Controls<'_>, custom: bool) -> Result<(), Status> {
    let suffix: &[u8] = if custom { b" (Custom)" } else { b"" };
    let mut label = Vec::new();
    let length = controls
        .base_grain_label()
        .len()
        .checked_add(suffix.len())
        .ok_or(Status::MEMORY)?;
    label
        .try_reserve_exact(length)
        .map_err(|_| Status::MEMORY)?;
    label.extend_from_slice(controls.base_grain_label());
    if custom {
        label.extend_from_slice(suffix);
    }
    controls.set_label(ParamId::GrainPreset, &label)
}

pub(super) fn apply_dir_defaults(controls: &Controls<'_>) -> Result<(), Status> {
    if controls.integer(ParamId::CouplersGammaUseStock)? == 0 {
        return Ok(());
    }
    let key = film_key(controls)?;
    if let Some(defaults) = controls.film_defaults(&key)? {
        controls.set_double3(
            ParamId::CouplersGammaSameLayerRgb,
            defaults.gamma_samelayer_rgb.map(f64::from),
        )?;
        controls.set_double2(
            ParamId::CouplersGammaInterlayerRToGb,
            defaults.gamma_interlayer_r_to_gb.map(f64::from),
        )?;
        controls.set_double2(
            ParamId::CouplersGammaInterlayerGToRb,
            defaults.gamma_interlayer_g_to_rb.map(f64::from),
        )?;
        controls.set_double2(
            ParamId::CouplersGammaInterlayerBToRg,
            defaults.gamma_interlayer_b_to_rg.map(f64::from),
        )?;
    }
    Ok(())
}

// Returns true for preset/reset edits, whose native path captures immediately.
pub(super) fn change(
    controls: &Controls<'_>,
    name: &[u8],
    user: bool,
    read_before_ui: bool,
) -> Result<bool, Status> {
    if user && name == b"CameraFilmFormatPreset" {
        if let Some(mm) =
            film::input::film_format_preset(controls.integer(ParamId::CameraFilmFormatPreset)?)
        {
            controls.set_double(ParamId::CameraFilmFormatMm, mm)?;
        }
    } else if user && name == b"CameraFilmFormatMm" {
        controls.set_integer(ParamId::CameraFilmFormatPreset, 0)?;
    }
    if user
        && [
            b"CouplersGammaSameLayerRgb".as_slice(),
            b"CouplersGammaInterlayerRToGb",
            b"CouplersGammaInterlayerGToRb",
            b"CouplersGammaInterlayerBToRg",
        ]
        .contains(&name)
    {
        controls.set_integer(ParamId::CouplersGammaUseStock, 0)?;
    }
    if name == b"FilmProfileKey" {
        let key = film_key(controls)?;
        let route = capture::route_key(capture::default_route(positive(controls, &key)?));
        controls.set_string_bytes(ParamId::ScanRoute, route)?;
    }
    if user && name == b"GrainPreset" {
        let index = controls.integer(ParamId::GrainPreset)?;
        apply_grain_preset(controls, index)?;
        if read_before_ui {
            controls.begin_read()?;
        }
        update_grain_label(controls, false)?;
        update_grain_chroma(controls)?;
        return Ok(true);
    }
    if user && name == b"GrainResetAdvanced" {
        reset_grain_advanced(controls)?;
        if read_before_ui {
            controls.begin_read()?;
        }
        update_grain_chroma(controls)?;
        return Ok(true);
    }
    let linked_grain_write = user
        && [
            b"GrainParticleScaleMaster".as_slice(),
            b"GrainParticleScaleLayersMaster",
            b"GrainDensityMinMaster",
            b"GrainUniformityMaster",
            b"GrainSharpness",
            b"GrainTexture",
        ]
        .contains(&name);
    if read_before_ui && !linked_grain_write {
        controls.begin_read()?;
    }
    if user
        && [
            b"GrainAmplitude".as_slice(),
            b"GrainBlur",
            b"GrainSharpness",
            b"GrainChroma",
            b"GrainTexture",
            b"GrainSublayersActive",
            b"GrainParticleAreaUm2",
            b"GrainParticleScaleMaster",
            b"GrainParticleScaleLayersMaster",
            b"GrainDensityMinMaster",
            b"GrainUniformityMaster",
            b"GrainParticleScale",
            b"GrainParticleScaleLayers",
            b"GrainDensityMin",
            b"GrainUniformity",
            b"GrainBlurDyeCloudsUm",
            b"GrainSizeMixWeight",
            b"GrainSizeMixWeightMid",
            b"GrainSizeMixScale",
            b"GrainMicroStructure",
            b"GrainClumpTemporalMix",
            b"GrainClumpMorphPeriodSec",
        ]
        .contains(&name)
    {
        update_grain_label(controls, true)?;
    }
    update_grain_chroma(controls)?;
    if user {
        for (master_name, triplet_name, control) in [
            (
                ParamId::GrainParticleScaleMaster,
                ParamId::GrainParticleScale,
                grain::GrainRatioControl::ParticleScale,
            ),
            (
                ParamId::GrainParticleScaleLayersMaster,
                ParamId::GrainParticleScaleLayers,
                grain::GrainRatioControl::ParticleScaleLayers,
            ),
            (
                ParamId::GrainDensityMinMaster,
                ParamId::GrainDensityMin,
                grain::GrainRatioControl::DensityMin,
            ),
            (
                ParamId::GrainUniformityMaster,
                ParamId::GrainUniformity,
                grain::GrainRatioControl::Uniformity,
            ),
        ] {
            if name == master_name.name().to_bytes() {
                let master = controls.double(master_name)?;
                if master.is_finite() {
                    let values = controls.double3(triplet_name)?;
                    if let Some(values) = control.values(master, values) {
                        controls.set_double3(triplet_name, values)?;
                    }
                }
                break;
            }
        }
        if name == b"GrainSharpness" {
            let value = controls.double(ParamId::GrainSharpness)?;
            if let Some(blur) = grain::sharpness_link(value) {
                controls.set_double(ParamId::GrainBlurDyeCloudsUm, blur)?;
            }
        }
        if name == b"GrainTexture" {
            let value = controls.double(ParamId::GrainTexture)?;
            if let Some((weight, micro)) = grain::texture_link(value) {
                controls.set_double(ParamId::GrainSizeMixWeight, weight)?;
                controls.set_double2(ParamId::GrainMicroStructure, micro)?;
            }
        }
    }
    if read_before_ui {
        controls.begin_read()?;
    }
    refresh_route_ui(controls)?;
    if name == b"GrainModel" {
        update_grain_model(controls)?;
    }
    if matches!(name, b"HalationActive" | b"HalationReflectionSource") {
        update_halation(controls)?;
    }
    if name == b"HalationHighlightBoostEv" {
        update_highlight_boost(controls)?;
    }
    Ok(false)
}

fn update_diffusion(controls: &Controls<'_>, print_route: bool) -> Result<(), Status> {
    let camera_enabled = controls.integer(ParamId::CameraDiffusionEnabled)? != 0;
    let print_enabled = controls.integer(ParamId::PrintDiffusionEnabled)? != 0;
    for (names, visible, enabled) in [
        (
            [
                ParamId::CameraDiffusionEnabled,
                ParamId::CameraDiffusionFamily,
                ParamId::CameraDiffusionStrength,
                ParamId::CameraDiffusionSpatialScale,
                ParamId::CameraDiffusionHaloWarmth,
                ParamId::CameraDiffusionCoreIntensity,
                ParamId::CameraDiffusionCoreSize,
                ParamId::CameraDiffusionHaloIntensity,
                ParamId::CameraDiffusionHaloSize,
                ParamId::CameraDiffusionBloomIntensity,
                ParamId::CameraDiffusionBloomSize,
            ],
            true,
            camera_enabled,
        ),
        (
            [
                ParamId::PrintDiffusionEnabled,
                ParamId::PrintDiffusionFamily,
                ParamId::PrintDiffusionStrength,
                ParamId::PrintDiffusionSpatialScale,
                ParamId::PrintDiffusionHaloWarmth,
                ParamId::PrintDiffusionCoreIntensity,
                ParamId::PrintDiffusionCoreSize,
                ParamId::PrintDiffusionHaloIntensity,
                ParamId::PrintDiffusionHaloSize,
                ParamId::PrintDiffusionBloomIntensity,
                ParamId::PrintDiffusionBloomSize,
            ],
            print_route,
            print_enabled,
        ),
    ] {
        for (index, name) in names.into_iter().enumerate() {
            controls.set_secret(name, !visible)?;
            controls.set_enabled(name, visible && (index == 0 || enabled))?;
        }
    }
    Ok(())
}

fn update_gamma(controls: &Controls<'_>, print_route: bool, positive: bool) -> Result<(), Status> {
    controls.set_enabled(ParamId::PrintGammaFactor, print_route)?;
    let active = controls.integer(ParamId::CouplersActive)? != 0;
    let core = controls.double(ParamId::CouplersDiffusionSizeUm)?;
    let tail_weight = controls.double(ParamId::CouplersDiffusionTailWeight)?;
    controls.set_enabled(ParamId::CouplersLangmuirDonorKRgb, active && !positive)?;
    controls.set_enabled(ParamId::CouplersLangmuirReceiverKRgb, active && positive)?;
    controls.set_enabled(ParamId::CouplersDiffusionTailWeight, active && core > 0.0)?;
    controls.set_enabled(
        ParamId::CouplersDiffusionTailUm,
        active && core > 0.0 && tail_weight > 0.0,
    )
}

fn update_spectral(controls: &Controls<'_>) -> Result<(), Status> {
    let mode = controls.integer(ParamId::SpectralUpsampling)?;
    controls.set_enabled(ParamId::InputCompression, mode != 1)?;
    controls.set_enabled(ParamId::Hanatos2025AdaptationWindow, mode == 0)?;
    controls.set_enabled(ParamId::Hanatos2025AdaptationSurface, mode == 0)
}

fn write_grain_advanced(
    controls: &Controls<'_>,
    preset: grain::PresetDefaults,
    advanced: grain::AdvancedDefaults,
    edit: grain::GrainDefaultsEdit,
) -> Result<(), Status> {
    let values = grain::advanced_writes(preset, advanced, edit);
    controls.set_double(ParamId::GrainParticleAreaUm2, preset.particle_area_um2)?;
    for (master_name, triplet_name, master, triplet) in [
        (
            ParamId::GrainParticleScaleMaster,
            ParamId::GrainParticleScale,
            preset.particle_scale_master,
            values.triplets[0],
        ),
        (
            ParamId::GrainParticleScaleLayersMaster,
            ParamId::GrainParticleScaleLayers,
            preset.particle_scale_layers_master,
            values.triplets[1],
        ),
        (
            ParamId::GrainDensityMinMaster,
            ParamId::GrainDensityMin,
            preset.density_min_master,
            values.triplets[2],
        ),
        (
            ParamId::GrainUniformityMaster,
            ParamId::GrainUniformity,
            preset.uniformity_master,
            values.triplets[3],
        ),
    ] {
        controls.set_double(master_name, master)?;
        controls.set_double3(triplet_name, triplet)?;
    }
    controls.set_double(ParamId::GrainBlurDyeCloudsUm, values.blur_dye_clouds)?;
    controls.set_double(ParamId::GrainSizeMixWeight, values.size_mix_weight)?;
    controls.set_double(ParamId::GrainSizeMixWeightMid, values.mid_weight)?;
    controls.set_double(ParamId::GrainSizeMixScale, values.size_mix_scale)?;
    controls.set_double2(ParamId::GrainMicroStructure, values.micro_structure)?;
    controls.set_double(ParamId::GrainClumpTemporalMix, values.clump_temporal_mix)?;
    controls.set_double(
        ParamId::GrainClumpMorphPeriodSec,
        values.clump_morph_period_sec,
    )
}

fn apply_grain_preset(controls: &Controls<'_>, index: i32) -> Result<(), Status> {
    let preset = grain::preset_defaults(index);
    let advanced = grain::advanced_defaults(preset, preset.sharpness, preset.texture);
    controls.set_double(ParamId::GrainAmplitude, preset.amount_ev)?;
    controls.set_double(ParamId::GrainBlur, preset.size_px)?;
    controls.set_double(ParamId::GrainSharpness, preset.sharpness)?;
    controls.set_double(ParamId::GrainChroma, preset.chroma)?;
    controls.set_double(ParamId::GrainTexture, preset.texture)?;
    controls.set_integer(
        ParamId::GrainSublayersActive,
        i32::from(preset.sublayers_active),
    )?;
    write_grain_advanced(controls, preset, advanced, grain::GrainDefaultsEdit::Preset)
}

fn reset_grain_advanced(controls: &Controls<'_>) -> Result<(), Status> {
    let preset = grain::preset_defaults(controls.integer(ParamId::GrainPreset)?);
    let sharpness = controls.double(ParamId::GrainSharpness)?;
    let texture = controls.double(ParamId::GrainTexture)?;
    let advanced = grain::reset_advanced_defaults(preset, sharpness, texture);
    write_grain_advanced(controls, preset, advanced, grain::GrainDefaultsEdit::Reset)
}

fn capture_grain(controls: &Controls<'_>) -> Result<grain::Params, Status> {
    let authored = grain::CaptureControls {
        active: controls.integer(ParamId::GrainActive)? != 0,
        preset: controls.integer(ParamId::GrainPreset)?,
        sublayers_active: controls.integer(ParamId::GrainSublayersActive)? != 0,
        amount_ev: controls.double(ParamId::GrainAmplitude)?,
        size_px: controls.double(ParamId::GrainBlur)?,
        sharpness: controls.double(ParamId::GrainSharpness)?,
        chroma: controls.double(ParamId::GrainChroma)?,
        texture: controls.double(ParamId::GrainTexture)?,
        particle_area_um2: controls.double(ParamId::GrainParticleAreaUm2)?,
        size_mix_scale: controls.double(ParamId::GrainSizeMixScale)?,
        coarse_weight: controls.double(ParamId::GrainSizeMixWeight)?,
        mid_weight: controls.double(ParamId::GrainSizeMixWeightMid)?,
        dye_cloud_blur_um: controls.double(ParamId::GrainBlurDyeCloudsUm)?,
        particle_scale_cmy: controls.double3(ParamId::GrainParticleScale)?,
        particle_scale_layers: controls.double3(ParamId::GrainParticleScaleLayers)?,
        particle_density_min_cmy: controls.double3(ParamId::GrainDensityMin)?,
        uniformity_cmy: controls.double3(ParamId::GrainUniformity)?,
        clump_temporal_mix: controls.double(ParamId::GrainClumpTemporalMix)?,
        clump_morph_period_sec: controls.double(ParamId::GrainClumpMorphPeriodSec)?,
        micro_structure: controls.double2(ParamId::GrainMicroStructure)?,
        debug_view: controls.integer(ParamId::GrainDebugView)?,
    };
    Ok(authored.complete())
}
