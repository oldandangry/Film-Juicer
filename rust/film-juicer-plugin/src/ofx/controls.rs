//! Current-value control capture and host UI edits.

#![forbid(unsafe_code)]

use film_juicer_core::film::grain;
use film_juicer_core::profile::{self, Polarity, Role};
use film_juicer_core::{capture, effects, exposure, film, optics, print, scanner};
use film::dir::DirField;

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
            c"PrintDiffusionEnabled",
            c"PrintDiffusionFamily",
            c"PrintDiffusionStrength",
            c"PrintDiffusionSpatialScale",
            c"PrintDiffusionHaloWarmth",
            c"PrintDiffusionCoreIntensity",
            c"PrintDiffusionCoreSize",
            c"PrintDiffusionHaloIntensity",
            c"PrintDiffusionHaloSize",
            c"PrintDiffusionBloomIntensity",
            c"PrintDiffusionBloomSize",
        ]
    } else {
        [
            c"CameraDiffusionEnabled",
            c"CameraDiffusionFamily",
            c"CameraDiffusionStrength",
            c"CameraDiffusionSpatialScale",
            c"CameraDiffusionHaloWarmth",
            c"CameraDiffusionCoreIntensity",
            c"CameraDiffusionCoreSize",
            c"CameraDiffusionHaloIntensity",
            c"CameraDiffusionHaloSize",
            c"CameraDiffusionBloomIntensity",
            c"CameraDiffusionBloomSize",
        ]
    };
    let active = controls.integer(names[0])? != 0;
    let family = controls.integer(names[1])?;
    let mut values = [0.0; 9];
    for (value, name) in values.iter_mut().zip(&names[2..]) {
        *value = controls.double(name)?;
    }
    Ok(optics::diffusion(active, family, values))
}

fn capture_dir(
    controls: &Controls<'_>,
    positive: bool,
) -> Result<film_juicer_core::film::dir::Params, CaptureError> {
    let active = controls.integer(c"CouplersActive")? != 0;
    let stock = controls.integer(c"CouplersGammaUseStock")? != 0;
    let mut dir = film::dir::capture_defaults(active, stock);
    if !active {
        return Ok(dir);
    }
    let narrow = |field: DirField, raw| field.narrow(raw).map_err(invalid_diagnostic);
    dir.amount = narrow(DirField::Amount, controls.double(c"CouplersAmount")?)?;
    if dir.amount == 0.0 {
        return Ok(dir);
    }
    dir.inhibition_same_layer = narrow(
        DirField::InhibitionSame,
        controls.double(c"CouplersInhibitionSameLayer")?,
    )?;
    dir.inhibition_interlayer = narrow(
        DirField::InhibitionInter,
        controls.double(c"CouplersInhibitionInterlayer")?,
    )?;
    dir.diffusion_size_um = narrow(
        DirField::DiffusionCoreUm,
        controls.double(c"CouplersDiffusionSizeUm")?,
    )?;
    if !stock {
        let same = controls.double3(c"CouplersGammaSameLayerRgb")?;
        let r = controls.double2(c"CouplersGammaInterlayerRToGb")?;
        let g = controls.double2(c"CouplersGammaInterlayerGToRb")?;
        let b = controls.double2(c"CouplersGammaInterlayerBToRg")?;
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
            c"CouplersLangmuirReceiverKRgb",
            DirField::ReceiverK,
            &mut dir.langmuir_receiver_k_rgb,
        )
    } else {
        (
            c"CouplersLangmuirDonorKRgb",
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
            controls.double(c"CouplersDiffusionTailWeight")?,
        )?;
        if dir.diffusion_tail_weight > 0.0 {
            dir.diffusion_tail_um = narrow(
                DirField::DiffusionTailUm,
                controls.double(c"CouplersDiffusionTailUm")?,
            )?;
        }
    }
    Ok(dir)
}

pub(super) fn capture(controls: &Controls<'_>) -> Result<capture::Captured, CaptureError> {
    let mut p = capture::defaults();
    let active = controls.integer(c"HalationActive")? != 0;
    let values = if active {
        [
            controls.double(c"HalationScatterAmount")?,
            controls.double(c"HalationScatterSpatialScale")?,
            controls.double(c"HalationAmount")?,
            controls.double(c"HalationSpatialScale")?,
        ]
    } else {
        [0.0; 4]
    };
    p.scatter_halation = optics::halation(active, values).map_err(invalid_diagnostic)?;
    p.camera_film_format_long_edge_mm =
        film::input::film_format_mm(controls.double(c"CameraFilmFormatMm")?)
            .map_err(invalid_diagnostic)?;
    let film = film_key(controls)?;
    let print = profile::capture_key(Role::Print, controls.string_bytes(c"PrintProfileKey")?)
        .map_err(|_| Status::MEMORY)?;
    let polarity = positive(controls, &film)?;
    let route = controls.string_bytes(c"ScanRoute")?;
    p.scan_route = capture::route(polarity, &route);
    p.spectral_upsampling_mode = controls.integer(c"SpectralUpsampling")?;
    p.ref_ill = controls.integer(c"ReferenceIlluminant")?;
    p.enl_ill = controls.integer(c"EnlargerIlluminant")?;
    let film_gamma = controls.double(c"FilmGammaFactor")?;
    let print_gamma = controls.double(c"PrintGammaFactor")?;
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
    print::CaptureControls {
        exposure: controls.double(c"PrintExposure")?,
        preflash: controls.double(c"PrintPreflash")?,
        exposure_compensation: controls.integer(c"PrintExposureCompensation")? != 0,
        ymc: [
            controls.double(c"EnlargerY")?,
            controls.double(c"EnlargerM")?,
            controls.double(c"EnlargerC")?,
        ],
    }
    .complete(&mut p);
    scanner::GlareControls {
        factor: controls.double(c"PrintShadowCompensationFactor")?,
        density: controls.double(c"PrintShadowCompensationDensity")?,
        transition: controls.double(c"PrintShadowCompensationTransition")?,
        active: controls.integer(c"GlareActive")? != 0,
        percent: controls.double(c"GlarePercent")?,
        roughness: controls.double(c"GlareRoughness")?,
        blur: controls.double(c"GlareBlurSigmaPx")?,
    }
    .complete(&mut p);
    p.input_color_space = controls.integer(c"InputColorSpace")?;
    p.input_cctf_decoding = i32::from(controls.integer(c"InputCctfDecoding")? != 0);
    p.input_compression_enabled = i32::from(controls.integer(c"InputCompression")? != 0);
    p.hanatos2025_adaptation_window =
        i32::from(controls.integer(c"Hanatos2025AdaptationWindow")? != 0);
    p.hanatos2025_adaptation_surface =
        i32::from(controls.integer(c"Hanatos2025AdaptationSurface")? != 0);
    p.camera_exposure_compensation_ev = exposure::authored_ev(controls.double(c"Exposure")?);
    p.camera_auto_exposure_enabled = i32::from(controls.integer(c"CameraAutoExposure")? != 0);
    p.camera_metering_method = controls.integer(c"CameraMeteringMethod")?;
    p.grain = capture_grain(controls)?;
    effects::CaptureControls {
        film_dust: controls.double(c"FilmDustAmount")?,
        gate_dust: controls.double(c"GateDustAmount")?,
        film_scratch: controls.double(c"FilmScratchAmount")?,
        gate_scratch: controls.double(c"GateScratchAmount")?,
        gate_weave: controls.double(c"GateWeaveAmount")?,
    }
    .complete(&mut p);
    p.dir_couplers = capture_dir(controls, polarity)?;
    p.scanner_lens_blur_sigma_px = controls.double(c"ScannerLensBlurSigmaPx")?;
    p.scanner_unsharp_mask = controls.double2(c"ScannerUnsharpMask")?;
    p.scanner_black_correction = i32::from(controls.integer(c"ScannerBlackCorrection")? != 0);
    p.scanner_white_correction = i32::from(controls.integer(c"ScannerWhiteCorrection")? != 0);
    let black = controls.double(c"ScannerBlackLevel")?;
    let white = controls.double(c"ScannerWhiteLevel")?;
    (p.scanner_black_level, p.scanner_white_level) = scanner::capture_levels(black, white);
    p.scanner_lut_resolution = controls.integer(c"ScannerLutResolution")?;
    p.output_color_space = controls.integer(c"OutputColorSpace")?;
    p.output_cctf_encoding = i32::from(controls.integer(c"OutputCctfEncoding")? != 0);
    p.output_gamut_compression_enabled =
        i32::from(controls.integer(c"OutputGamutCompression")? != 0);
    Ok(capture::Captured::new(p, film, print))
}

fn film_key(controls: &Controls<'_>) -> Result<Vec<u8>, Status> {
    profile::capture_key(Role::Film, controls.string_bytes(c"FilmProfileKey")?)
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
    let selection = controls.string_bytes(c"ScanRoute")?;
    let print = matches!(
        capture::route(positive, &selection),
        film_juicer_core::route::ScanRoute::NegativePrint
            | film_juicer_core::route::ScanRoute::PositivePrint
    );
    Ok((print, positive))
}

pub(super) fn initialize(controls: &Controls<'_>) -> Result<(), Status> {
    let diffusion_route = control_route(controls)?;
    let gamma_route = control_route(controls)?;
    update_diffusion(controls, diffusion_route.0)?;
    update_gamma(controls, gamma_route.0, gamma_route.1)?;
    update_spectral(controls)
}

fn update_grain_chroma(controls: &Controls<'_>) -> Result<(), Status> {
    controls.set_enabled(c"GrainChroma", true)?;
    controls.set_hint(c"GrainChroma", controls.grain_chroma_hint())
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
    controls.set_label(c"GrainPreset", &label)
}

pub(super) fn apply_dir_defaults(controls: &Controls<'_>) -> Result<(), Status> {
    if controls.integer(c"CouplersGammaUseStock")? == 0 {
        return Ok(());
    }
    let key = film_key(controls)?;
    if let Some(defaults) = controls.film_defaults(&key)? {
        controls.set_double3(
            c"CouplersGammaSameLayerRgb",
            defaults.gamma_samelayer_rgb.map(f64::from),
        )?;
        controls.set_double2(
            c"CouplersGammaInterlayerRToGb",
            defaults.gamma_interlayer_r_to_gb.map(f64::from),
        )?;
        controls.set_double2(
            c"CouplersGammaInterlayerGToRb",
            defaults.gamma_interlayer_g_to_rb.map(f64::from),
        )?;
        controls.set_double2(
            c"CouplersGammaInterlayerBToRg",
            defaults.gamma_interlayer_b_to_rg.map(f64::from),
        )?;
    }
    Ok(())
}

// Returns true for preset/reset edits, whose native path captures immediately.
pub(super) fn change(controls: &Controls<'_>, name: &[u8], user: bool) -> Result<bool, Status> {
    if user && name == b"CameraFilmFormatPreset" {
        if let Some(mm) =
            film::input::film_format_preset(controls.integer(c"CameraFilmFormatPreset")?)
        {
            controls.set_double(c"CameraFilmFormatMm", mm)?;
        }
    } else if user && name == b"CameraFilmFormatMm" {
        controls.set_integer(c"CameraFilmFormatPreset", 0)?;
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
        controls.set_integer(c"CouplersGammaUseStock", 0)?;
    }
    if name == b"FilmProfileKey" {
        let key = film_key(controls)?;
        let route = capture::route_key(capture::default_route(positive(controls, &key)?));
        controls.set_string_bytes(c"ScanRoute", route)?;
    }
    if user && name == b"GrainPreset" {
        let index = controls.integer(c"GrainPreset")?;
        apply_grain_preset(controls, index)?;
        update_grain_label(controls, false)?;
        update_grain_chroma(controls)?;
        return Ok(true);
    }
    if user && name == b"GrainResetAdvanced" {
        reset_grain_advanced(controls)?;
        update_grain_chroma(controls)?;
        return Ok(true);
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
                c"GrainParticleScaleMaster",
                c"GrainParticleScale",
                grain::GrainRatioControl::ParticleScale,
            ),
            (
                c"GrainParticleScaleLayersMaster",
                c"GrainParticleScaleLayers",
                grain::GrainRatioControl::ParticleScaleLayers,
            ),
            (
                c"GrainDensityMinMaster",
                c"GrainDensityMin",
                grain::GrainRatioControl::DensityMin,
            ),
            (
                c"GrainUniformityMaster",
                c"GrainUniformity",
                grain::GrainRatioControl::Uniformity,
            ),
        ] {
            if name == master_name.to_bytes() {
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
            let value = controls.double(c"GrainSharpness")?;
            if let Some(blur) = grain::sharpness_link(value) {
                controls.set_double(c"GrainBlurDyeCloudsUm", blur)?;
            }
        }
        if name == b"GrainTexture" {
            let value = controls.double(c"GrainTexture")?;
            if let Some((weight, micro)) = grain::texture_link(value) {
                controls.set_double(c"GrainSizeMixWeight", weight)?;
                controls.set_double2(c"GrainMicroStructure", micro)?;
            }
        }
    }
    initialize(controls)?;
    Ok(false)
}

fn update_diffusion(controls: &Controls<'_>, print_route: bool) -> Result<(), Status> {
    let camera_enabled = controls.integer(c"CameraDiffusionEnabled")? != 0;
    let print_enabled = controls.integer(c"PrintDiffusionEnabled")? != 0;
    for (names, visible, enabled) in [
        (
            [
                c"CameraDiffusionEnabled",
                c"CameraDiffusionFamily",
                c"CameraDiffusionStrength",
                c"CameraDiffusionSpatialScale",
                c"CameraDiffusionHaloWarmth",
                c"CameraDiffusionCoreIntensity",
                c"CameraDiffusionCoreSize",
                c"CameraDiffusionHaloIntensity",
                c"CameraDiffusionHaloSize",
                c"CameraDiffusionBloomIntensity",
                c"CameraDiffusionBloomSize",
            ],
            true,
            camera_enabled,
        ),
        (
            [
                c"PrintDiffusionEnabled",
                c"PrintDiffusionFamily",
                c"PrintDiffusionStrength",
                c"PrintDiffusionSpatialScale",
                c"PrintDiffusionHaloWarmth",
                c"PrintDiffusionCoreIntensity",
                c"PrintDiffusionCoreSize",
                c"PrintDiffusionHaloIntensity",
                c"PrintDiffusionHaloSize",
                c"PrintDiffusionBloomIntensity",
                c"PrintDiffusionBloomSize",
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
    controls.set_enabled(c"PrintGammaFactor", print_route)?;
    let active = controls.integer(c"CouplersActive")? != 0;
    let core = controls.double(c"CouplersDiffusionSizeUm")?;
    let tail_weight = controls.double(c"CouplersDiffusionTailWeight")?;
    controls.set_enabled(c"CouplersLangmuirDonorKRgb", active && !positive)?;
    controls.set_enabled(c"CouplersLangmuirReceiverKRgb", active && positive)?;
    controls.set_enabled(c"CouplersDiffusionTailWeight", active && core > 0.0)?;
    controls.set_enabled(
        c"CouplersDiffusionTailUm",
        active && core > 0.0 && tail_weight > 0.0,
    )
}

fn update_spectral(controls: &Controls<'_>) -> Result<(), Status> {
    let mode = controls.integer(c"SpectralUpsampling")?;
    controls.set_enabled(c"InputCompression", mode != 1)?;
    controls.set_enabled(c"Hanatos2025AdaptationWindow", mode == 0)?;
    controls.set_enabled(c"Hanatos2025AdaptationSurface", mode == 0)
}

fn write_grain_advanced(
    controls: &Controls<'_>,
    preset: grain::PresetDefaults,
    advanced: grain::AdvancedDefaults,
    edit: grain::GrainDefaultsEdit,
) -> Result<(), Status> {
    let values = grain::advanced_writes(preset, advanced, edit);
    controls.set_double(c"GrainParticleAreaUm2", preset.particle_area_um2)?;
    for (master_name, triplet_name, master, triplet) in [
        (
            c"GrainParticleScaleMaster",
            c"GrainParticleScale",
            preset.particle_scale_master,
            values.triplets[0],
        ),
        (
            c"GrainParticleScaleLayersMaster",
            c"GrainParticleScaleLayers",
            preset.particle_scale_layers_master,
            values.triplets[1],
        ),
        (
            c"GrainDensityMinMaster",
            c"GrainDensityMin",
            preset.density_min_master,
            values.triplets[2],
        ),
        (
            c"GrainUniformityMaster",
            c"GrainUniformity",
            preset.uniformity_master,
            values.triplets[3],
        ),
    ] {
        controls.set_double(master_name, master)?;
        controls.set_double3(triplet_name, triplet)?;
    }
    controls.set_double(c"GrainBlurDyeCloudsUm", values.blur_dye_clouds)?;
    controls.set_double(c"GrainSizeMixWeight", values.size_mix_weight)?;
    controls.set_double(c"GrainSizeMixWeightMid", values.mid_weight)?;
    controls.set_double(c"GrainSizeMixScale", values.size_mix_scale)?;
    controls.set_double2(c"GrainMicroStructure", values.micro_structure)?;
    controls.set_double(c"GrainClumpTemporalMix", values.clump_temporal_mix)?;
    controls.set_double(c"GrainClumpMorphPeriodSec", values.clump_morph_period_sec)
}

fn apply_grain_preset(controls: &Controls<'_>, index: i32) -> Result<(), Status> {
    let preset = grain::preset_defaults(index);
    let advanced = grain::advanced_defaults(preset, preset.sharpness, preset.texture);
    controls.set_double(c"GrainAmplitude", preset.amount_ev)?;
    controls.set_double(c"GrainBlur", preset.size_px)?;
    controls.set_double(c"GrainSharpness", preset.sharpness)?;
    controls.set_double(c"GrainChroma", preset.chroma)?;
    controls.set_double(c"GrainTexture", preset.texture)?;
    controls.set_integer(c"GrainSublayersActive", i32::from(preset.sublayers_active))?;
    write_grain_advanced(controls, preset, advanced, grain::GrainDefaultsEdit::Preset)
}

fn reset_grain_advanced(controls: &Controls<'_>) -> Result<(), Status> {
    let preset = grain::preset_defaults(controls.integer(c"GrainPreset")?);
    let sharpness = controls.double(c"GrainSharpness")?;
    let texture = controls.double(c"GrainTexture")?;
    let advanced = grain::reset_advanced_defaults(preset, sharpness, texture);
    write_grain_advanced(controls, preset, advanced, grain::GrainDefaultsEdit::Reset)
}

fn capture_grain(controls: &Controls<'_>) -> Result<grain::Params, Status> {
    let authored = grain::CaptureControls {
        active: controls.integer(c"GrainActive")? != 0,
        preset: controls.integer(c"GrainPreset")?,
        sublayers_active: controls.integer(c"GrainSublayersActive")? != 0,
        amount_ev: controls.double(c"GrainAmplitude")?,
        size_px: controls.double(c"GrainBlur")?,
        sharpness: controls.double(c"GrainSharpness")?,
        chroma: controls.double(c"GrainChroma")?,
        texture: controls.double(c"GrainTexture")?,
        particle_area_um2: controls.double(c"GrainParticleAreaUm2")?,
        size_mix_scale: controls.double(c"GrainSizeMixScale")?,
        coarse_weight: controls.double(c"GrainSizeMixWeight")?,
        mid_weight: controls.double(c"GrainSizeMixWeightMid")?,
        dye_cloud_blur_um: controls.double(c"GrainBlurDyeCloudsUm")?,
        particle_scale_cmy: controls.double3(c"GrainParticleScale")?,
        particle_scale_layers: controls.double3(c"GrainParticleScaleLayers")?,
        particle_density_min_cmy: controls.double3(c"GrainDensityMin")?,
        uniformity_cmy: controls.double3(c"GrainUniformity")?,
        clump_temporal_mix: controls.double(c"GrainClumpTemporalMix")?,
        clump_morph_period_sec: controls.double(c"GrainClumpMorphPeriodSec")?,
        micro_structure: controls.double2(c"GrainMicroStructure")?,
        debug_view: controls.integer(c"GrainDebugView")?,
    };
    Ok(authored.complete())
}
