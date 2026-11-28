//! Identity of completed host controls before preparation admission.
use crate::film::{dir, grain};
use crate::{hash, optics, route::ScanRoute};

struct Mixer(u64);
impl Mixer {
    fn add(&mut self, v: u64) {
        self.0 ^= v
            .wrapping_add(0x9e3779b97f4a7c15)
            .wrapping_add(self.0 << 6)
            .wrapping_add(self.0 >> 2);
    }
    fn int(&mut self, v: i32) {
        self.add(v as u64);
    }
    fn string(&mut self, v: &[u8]) {
        self.add(v.len() as u64);
        if !v.is_empty() {
            self.add(hash::bytes(v));
        }
    }
    fn float(&mut self, v: f32) {
        self.add(u64::from(if v == 0.0 { 0 } else { v.to_bits() }));
    }
    fn double(&mut self, v: f64) {
        self.add(if v == 0.0 { 0 } else { v.to_bits() });
    }
    fn retained(&mut self, v: f64) {
        if v.is_finite() {
            self.add(u64::from((v as f32).to_bits()));
        }
    }
    fn canonical_retained(&mut self, v: f64) {
        if v.is_finite() {
            self.float(v as f32);
        }
    }
}
/// Hash copied key bytes before UTF-8 admission, matching the native snapshot stream.
/// Captured recipe inputs. Integer choices stay unadmitted until their
/// owning construction stage, preserving competing-failure precedence.
pub struct Params {
    pub print_exposure: f64,
    pub print_preflash_exposure: f64,
    pub print_gamma_factor: f64,
    pub preflash_m_filter_cc: f64,
    pub preflash_y_filter_cc: f64,
    pub glare_percent: f64,
    pub glare_roughness: f64,
    pub glare_blur_sigma_px: f64,
    pub camera_exposure_compensation_ev: f64,
    pub camera_film_format_long_edge_mm: f32,
    pub film_gamma_factor: f32,
    pub scanner_lens_blur_sigma_px: f64,
    pub scanner_black_level: f64,
    pub scanner_white_level: f64,
    pub gate_weave_amount: f64,
    pub scanner_unsharp_mask: [f64; 2],
    pub print_ui_ymc_cc: [f64; 3],
    pub camera_filter_uv: [f64; 3],
    pub camera_filter_ir: [f64; 3],
    pub camera_diffusion: optics::DiffusionParams,
    pub enlarger_diffusion: optics::DiffusionParams,
    pub dir_couplers: dir::Params,
    pub film_profile_key: String,
    pub print_profile_key: String,
    pub spectral_upsampling_mode: i32,
    pub enl_ill: i32,
    pub normalize_print_exposure: i32,
    pub print_exposure_compensation: i32,
    pub input_color_space: i32,
    pub input_cctf_decoding: i32,
    pub input_compression_enabled: i32,
    pub hanatos2025_adaptation_window: i32,
    pub hanatos2025_adaptation_surface: i32,
    pub camera_auto_exposure_enabled: i32,
    pub camera_metering_method: i32,
    pub scatter_halation: HalationControls,
    pub highlight_boost: optics::highlight_boost::Controls,
    pub scanner_black_correction: i32,
    pub scanner_white_correction: i32,
    pub output_color_space: i32,
    pub output_cctf_encoding: i32,
    pub output_gamut_compression_enabled: i32,
    pub grain: grain::Params,
    pub grain_v2: grain::grain_v2::Selection,
    pub film_dust_amount: f32,
    pub film_scratch_amount: f32,
    pub gate_dust_amount: f32,
    pub gate_scratch_amount: f32,
    pub scan_route: ScanRoute,
    pub glare_active: bool,
    pub camera_filter_override: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HalationControls {
    pub active: bool,
    pub scatter_amount: f32,
    pub scatter_spatial_scale: f32,
    pub halation_amount: f32,
    pub halation_spatial_scale: f32,
    pub shape: optics::HalationShape,
    pub reflection: HalationReflectionSource,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum HalationReflectionSource {
    #[default]
    Profile,
    Custom {
        strength: [f64; 3],
        first_sigma_um: [f64; 3],
    },
}

impl HalationReflectionSource {
    pub fn validate(self) -> Result<Self, optics::Error> {
        match self {
            Self::Profile => Ok(self),
            Self::Custom {
                strength,
                first_sigma_um,
            } => Ok(Self::Custom {
                strength: optics::halation_triplet(
                    strength,
                    optics::HALATION_MAX_AUTHORED,
                    optics::Field::ReflectionStrength,
                )?,
                first_sigma_um: optics::halation_triplet(
                    first_sigma_um,
                    optics::HALATION_MAX_AUTHORED,
                    optics::Field::FirstBounceSigma,
                )?,
            }),
        }
    }

    pub(crate) fn resolve(
        self,
        profile_sigma_um: [f32; 3],
        profile_strength: [f32; 3],
    ) -> Result<([f32; 3], [f32; 3]), optics::Error> {
        match self.validate()? {
            Self::Profile => Ok((profile_sigma_um, profile_strength)),
            Self::Custom {
                strength,
                first_sigma_um,
            } => Ok((first_sigma_um.map(|v| v as f32), strength.map(|v| v as f32))),
        }
    }
}

pub(crate) fn legacy_grain(params: &Params) -> grain::Params {
    if params.grain_v2.controls().is_some() {
        return grain::Params {
            active: false,
            ..params.grain
        };
    }
    params.grain
}

#[cfg(test)]
pub(crate) fn hash_params(p: &Params, film_key: &[u8], print_key: &[u8]) -> u64 {
    capture_identity(p, film_key, print_key, 0, ShadowControls::default())
}

fn capture_identity(
    p: &Params,
    film_key: &[u8],
    print_key: &[u8],
    reference_illuminant: i32,
    shadow: ShadowControls,
) -> u64 {
    let mut h = Mixer(0);
    h.string(film_key);
    let route = match p.scan_route {
        ScanRoute::NegativeDirect => 0,
        ScanRoute::NegativePrint => 1,
        ScanRoute::PositiveDirect => 2,
        ScanRoute::PositivePrint => 3,
    };
    h.add(route);
    if route == 1 || route == 3 {
        h.string(print_key);
        h.int(p.enl_ill);
        h.retained(p.print_exposure);
        h.retained(p.print_preflash_exposure);
        h.add(p.print_gamma_factor.to_bits());
        h.int(p.normalize_print_exposure);
        h.int(p.print_exposure_compensation);
        for v in p.print_ui_ymc_cc {
            h.canonical_retained(v);
        }
        h.canonical_retained(p.preflash_m_filter_cc);
        h.canonical_retained(p.preflash_y_filter_cc);
    }
    h.int(p.spectral_upsampling_mode);
    h.int(reference_illuminant);
    for v in [
        bounded(shadow.factor, 0.0, 0.0, 1.0),
        bounded(shadow.density, 1.2, 0.0, 3.0),
        bounded(shadow.transition, 0.3, 0.0, 2.0),
    ] {
        h.add((v * 10000.0) as u64);
    }
    h.add(p.glare_active as u64);
    for v in [p.glare_percent, p.glare_roughness, p.glare_blur_sigma_px] {
        h.retained(v);
    }
    let d = &p.dir_couplers;
    h.add(d.active as u64);
    for v in [
        d.amount,
        d.inhibition_same_layer,
        d.inhibition_interlayer,
        d.diffusion_size_um,
        d.diffusion_tail_um,
        d.diffusion_tail_weight,
    ] {
        h.float(v);
    }
    h.add(d.gamma_use_stock as u64);
    for v in d
        .gamma_same_layer_rgb
        .into_iter()
        .chain(d.gamma_interlayer_r_to_gb)
        .chain(d.gamma_interlayer_g_to_rb)
        .chain(d.gamma_interlayer_b_to_rg)
        .chain(d.langmuir_donor_k_rgb)
        .chain(d.langmuir_receiver_k_rgb)
    {
        h.float(v);
    }
    for v in [
        p.input_color_space,
        p.input_cctf_decoding,
        p.output_color_space,
        p.output_cctf_encoding,
        p.output_gamut_compression_enabled,
    ] {
        h.int(v);
    }
    if p.spectral_upsampling_mode == 0 {
        h.int(p.hanatos2025_adaptation_window);
        h.int(p.hanatos2025_adaptation_surface);
    }
    if p.spectral_upsampling_mode == 0 || p.spectral_upsampling_mode == 2 {
        h.int(p.input_compression_enabled);
    }
    h.add(p.camera_filter_override as u64);
    if p.camera_filter_override {
        for v in p.camera_filter_uv.into_iter().chain(p.camera_filter_ir) {
            h.retained(v);
        }
    }
    h.int(p.camera_auto_exposure_enabled);
    h.int(p.camera_metering_method);
    h.retained(p.camera_exposure_compensation_ev);
    h.float(p.camera_film_format_long_edge_mm);
    h.float(p.film_gamma_factor);
    let s = &p.scatter_halation;
    h.add(s.active as u64);
    if s.active {
        for v in [
            s.scatter_amount,
            s.scatter_spatial_scale,
            s.halation_amount,
            s.halation_spatial_scale,
        ] {
            h.float(v);
        }
        let shape = s.shape.canonicalize_defaults();
        if shape != optics::HalationShape::default() {
            h.add(hash::bytes(b"halation-shape"));
            for v in shape
                .scatter_core_um
                .into_iter()
                .chain(shape.scatter_tail_um)
                .chain(shape.scatter_tail_weight)
            {
                h.double(v);
            }
            h.int(shape.bounce_count);
            h.double(shape.bounce_decay);
            h.add(shape.renormalize as u64);
        }
        if let HalationReflectionSource::Custom {
            strength,
            first_sigma_um,
        } = s.reflection
        {
            h.add(hash::bytes(b"halation-custom"));
            for v in strength.into_iter().chain(first_sigma_um) {
                h.double(v);
            }
        }
    }
    for v in [
        p.scanner_lens_blur_sigma_px,
        p.scanner_unsharp_mask[0],
        p.scanner_unsharp_mask[1],
    ] {
        h.retained(v);
    }
    h.int(p.scanner_black_correction);
    h.int(p.scanner_white_correction);
    h.retained(p.scanner_black_level);
    h.retained(p.scanner_white_level);
    h.add(0);
    h.add(optics::hash_diffusion_authored_controls(
        &p.camera_diffusion,
    ));
    h.add(3);
    h.add(optics::hash_diffusion_authored_controls(
        &p.enlarger_diffusion,
    ));
    // With grain disabled the prepared photographic state is shared. OFX
    // resolves per-frame dust and scratch geometry from the captured selection.
    if let Some(controls) = p.grain_v2.controls() {
        // Preserve the capture identity across the model rename.
        h.add(hash::bytes(b"film-v1-grain"));
        for value in [
            controls.amount_ev,
            controls.texel_pitch_um,
            controls.chroma_fraction,
        ] {
            h.add(u64::from(value.to_bits()));
        }
    }
    let g = legacy_grain(p);
    if g.active {
        h.add(1);
        h.add(g.sublayers_active as u64);
        for v in [g.particle_area_um2]
            .into_iter()
            .chain(g.particle_scale_cmy)
            .chain(g.particle_scale_layers)
            .chain(g.particle_density_min_cmy)
            .chain(g.uniformity_cmy)
            .chain([g.correlation_sigma_px, g.dye_cloud_blur_um])
            .chain(g.micro_structure)
        {
            h.retained(v as f64);
        }
        h.int(g.sublayer_count);
        for v in [
            g.amplitude,
            g.chroma_shared_weight,
            g.chroma_independent_weight,
            g.coarse_weight,
            g.mid_weight,
            g.size_mix_scale,
            g.clump_temporal_mix,
            g.clump_morph_period_sec,
        ] {
            h.retained(v as f64);
        }
        h.int(g.debug_view);
    }
    if [
        p.film_dust_amount as f64,
        p.film_scratch_amount as f64,
        p.gate_dust_amount as f64,
        p.gate_scratch_amount as f64,
        p.gate_weave_amount,
    ]
    .into_iter()
    .any(|v| v.is_finite() && v > 0.0)
    {
        h.add(1);
        for v in [
            p.film_dust_amount,
            p.film_scratch_amount,
            p.gate_dust_amount,
            p.gate_scratch_amount,
        ] {
            h.canonical_retained(v as f64);
        }
        if p.gate_weave_amount.is_finite() {
            h.add(p.gate_weave_amount.to_bits());
        }
    }
    if p.highlight_boost.boost_ev != 0.0 {
        h.add(hash::bytes(b"highlight-boost"));
        h.double(p.highlight_boost.boost_ev);
        h.double(p.highlight_boost.protection_ev);
        h.double(p.highlight_boost.range);
    }
    h.0
}

/// Completed capture retains host key bytes until the ticket is admitted.
pub struct Captured {
    params: Params,
    film_key: Vec<u8>,
    print_key: Vec<u8>,
    identity: u64,
}
impl Captured {
    pub fn new(
        params: Params,
        film_key: Vec<u8>,
        print_key: Vec<u8>,
        reference_illuminant: i32,
        shadow: ShadowControls,
    ) -> Self {
        let identity =
            capture_identity(&params, &film_key, &print_key, reference_illuminant, shadow);
        Self {
            params,
            film_key,
            print_key,
            identity,
        }
    }
    pub fn admit(mut self) -> Result<(Params, u64), std::string::FromUtf8Error> {
        self.params.film_profile_key = String::from_utf8(self.film_key)?;
        self.params.print_profile_key = String::from_utf8(self.print_key)?;
        Ok((self.params, self.identity))
    }
}
pub fn route(positive: bool, key: &[u8]) -> ScanRoute {
    let print = match key {
        b"NegativeDirectScan" | b"PositiveDirectScan" => false,
        b"NegativePrintScan" | b"PositivePrintScan" => true,
        _ => !positive,
    };
    match (positive, print) {
        (false, false) => ScanRoute::NegativeDirect,
        (false, true) => ScanRoute::NegativePrint,
        (true, false) => ScanRoute::PositiveDirect,
        (true, true) => ScanRoute::PositivePrint,
    }
}
pub fn route_key(route: ScanRoute) -> &'static [u8] {
    match route {
        ScanRoute::NegativeDirect => b"NegativeDirectScan",
        ScanRoute::NegativePrint => b"NegativePrintScan",
        ScanRoute::PositiveDirect => b"PositiveDirectScan",
        ScanRoute::PositivePrint => b"PositivePrintScan",
    }
}
pub fn default_route(positive: bool) -> ScanRoute {
    if positive {
        ScanRoute::PositiveDirect
    } else {
        ScanRoute::NegativePrint
    }
}
/// Initial snapshot values used before each ordered family capture completes.
pub fn defaults() -> Params {
    let diffusion = diffusion(false, 1, [0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    Params {
        print_exposure: 1.0,
        print_preflash_exposure: 0.0,
        print_gamma_factor: 1.0,
        preflash_m_filter_cc: 0.0,
        preflash_y_filter_cc: 0.0,
        glare_percent: 0.03,
        glare_roughness: 0.7,
        glare_blur_sigma_px: 0.5,
        camera_exposure_compensation_ev: 0.0,
        camera_film_format_long_edge_mm: 35.0,
        film_gamma_factor: 1.0,
        scanner_lens_blur_sigma_px: 0.0,
        scanner_black_level: 0.01,
        scanner_white_level: 0.98,
        gate_weave_amount: 0.0,
        scanner_unsharp_mask: [0.7, 0.7],
        print_ui_ymc_cc: [0.0; 3],
        camera_filter_uv: [1.0, 410.0, 8.0],
        camera_filter_ir: [1.0, 675.0, 15.0],
        camera_diffusion: diffusion,
        enlarger_diffusion: diffusion,
        dir_couplers: crate::film::dir::capture_defaults(true, true),
        film_profile_key: String::new(),
        print_profile_key: String::new(),
        spectral_upsampling_mode: 0,
        enl_ill: 3,
        normalize_print_exposure: 1,
        print_exposure_compensation: 1,
        input_color_space: 0,
        input_cctf_decoding: 0,
        input_compression_enabled: 1,
        hanatos2025_adaptation_window: 1,
        hanatos2025_adaptation_surface: 0,
        camera_auto_exposure_enabled: 1,
        camera_metering_method: 0,
        scatter_halation: HalationControls::default(),
        highlight_boost: optics::highlight_boost::Controls::default(),
        scanner_black_correction: 0,
        scanner_white_correction: 0,
        output_color_space: 0,
        output_cctf_encoding: 1,
        output_gamut_compression_enabled: 1,
        grain: crate::film::grain::Params {
            active: false,
            sublayers_active: true,
            particle_area_um2: 0.2,
            particle_scale_cmy: [0.8, 1.0, 2.0],
            particle_scale_layers: [2.5, 1.0, 0.5],
            particle_density_min_cmy: [0.07, 0.08, 0.12],
            uniformity_cmy: [0.97, 0.97, 0.99],
            correlation_sigma_px: 0.65,
            dye_cloud_blur_um: 1.0,
            micro_structure: [0.2, 30.0],
            sublayer_count: 1,
            amplitude: 1.0,
            chroma_mix: 0.0,
            chroma_shared_weight: 1.0,
            chroma_independent_weight: 0.0,
            coarse_weight: 0.0,
            mid_weight: 0.0,
            size_mix_scale: 1.0,
            clump_temporal_mix: 0.0,
            clump_morph_period_sec: 8.0,
            debug_view: 0,
        },
        grain_v2: crate::film::grain::grain_v2::Selection::Legacy,
        film_dust_amount: 0.0,
        film_scratch_amount: 0.0,
        gate_dust_amount: 0.0,
        gate_scratch_amount: 0.0,
        scan_route: ScanRoute::NegativePrint,
        glare_active: true,
        camera_filter_override: false,
    }
}

pub struct PrintControls {
    pub exposure: f64,
    pub preflash: f64,
    pub exposure_compensation: bool,
    pub ymc: [f64; 3],
}
impl PrintControls {
    pub fn complete(self, p: &mut Params) {
        p.print_exposure = crate::exposure::finite_or(self.exposure, 1.0);
        p.print_preflash_exposure = crate::exposure::finite_or(self.preflash, 0.0);
        p.print_exposure_compensation = i32::from(self.exposure_compensation);
        p.normalize_print_exposure = 1;
        p.print_ui_ymc_cc = self.ymc.map(|v| crate::exposure::finite_or(v, 0.0));
    }
}

fn bounded(value: f64, fallback: f64, lo: f64, hi: f64) -> f64 {
    if value.is_finite() {
        value.clamp(lo, hi)
    } else {
        fallback
    }
}
fn bounded_float(value: f64, fallback: f32, lo: f64, hi: f64) -> f32 {
    bounded(value, f64::from(fallback), lo, hi) as f32
}

pub struct GlareControls {
    pub active: bool,
    pub percent: f64,
    pub roughness: f64,
    pub blur: f64,
}
impl GlareControls {
    pub fn complete(self, p: &mut Params) {
        p.glare_active = self.active;
        p.glare_percent = f64::from(bounded_float(self.percent, 0.03, 0.0, 1.0));
        p.glare_roughness = f64::from(bounded_float(self.roughness, 0.7, 0.0, 1.0));
        p.glare_blur_sigma_px = f64::from(bounded_float(self.blur, 0.5, 0.0, 10.0));
    }
}

pub fn capture_levels(black: f64, white: f64) -> (f64, f64) {
    (
        bounded(black, 0.01, 0.0, 1.0),
        bounded(white, 0.98, 0.0, 1.0),
    )
}

/// Saved shadow controls contribute to capture reuse, even on Direct routes.
pub struct ShadowControls {
    pub factor: f64,
    pub density: f64,
    pub transition: f64,
}
#[cfg(test)]
impl Default for ShadowControls {
    fn default() -> Self {
        Self {
            factor: 0.0,
            density: 1.2,
            transition: 0.3,
        }
    }
}

pub struct EffectsControls {
    pub film_dust: f64,
    pub gate_dust: f64,
    pub film_scratch: f64,
    pub gate_scratch: f64,
    pub gate_weave: f64,
}
impl EffectsControls {
    pub fn complete(self, p: &mut Params) {
        p.film_dust_amount = bounded_float(self.film_dust, 0.0, 0.0, 10.0);
        p.gate_dust_amount = bounded_float(self.gate_dust, 0.0, 0.0, 10.0);
        p.film_scratch_amount = bounded_float(self.film_scratch, 0.0, 0.0, 10.0);
        p.gate_scratch_amount = bounded_float(self.gate_scratch, 0.0, 0.0, 10.0);
        p.gate_weave_amount = bounded(self.gate_weave, 0.0, 0.0, 10.0);
    }
}

pub fn halation(active: bool, values: [f64; 4]) -> Result<HalationControls, &'static str> {
    let mut result = HalationControls {
        active,
        ..Default::default()
    };
    if !active {
        return Ok(result);
    }
    const DIAGNOSTICS: [[&str; 3]; 4] = [
        [
            "InvalidAuthoredControl component=scatter_halation field=scatter_amount reason=non_finite",
            "InvalidAuthoredControl component=scatter_halation field=scatter_amount reason=below_minimum",
            "InvalidAuthoredControl component=scatter_halation field=scatter_amount reason=above_maximum",
        ],
        [
            "InvalidAuthoredControl component=scatter_halation field=scatter_spatial_scale reason=non_finite",
            "InvalidAuthoredControl component=scatter_halation field=scatter_spatial_scale reason=below_minimum",
            "InvalidAuthoredControl component=scatter_halation field=scatter_spatial_scale reason=above_maximum",
        ],
        [
            "InvalidAuthoredControl component=scatter_halation field=halation_amount reason=non_finite",
            "InvalidAuthoredControl component=scatter_halation field=halation_amount reason=below_minimum",
            "InvalidAuthoredControl component=scatter_halation field=halation_amount reason=above_maximum",
        ],
        [
            "InvalidAuthoredControl component=scatter_halation field=halation_spatial_scale reason=non_finite",
            "InvalidAuthoredControl component=scatter_halation field=halation_spatial_scale reason=below_minimum",
            "InvalidAuthoredControl component=scatter_halation field=halation_spatial_scale reason=above_maximum",
        ],
    ];
    let mut out = [0.0; 4];
    for (i, raw) in values.into_iter().enumerate() {
        let reason = if !raw.is_finite() {
            Some(0)
        } else if raw < 0.0 {
            Some(1)
        } else if raw > 2.0 {
            Some(2)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(DIAGNOSTICS[i][reason]);
        }
        let v = raw as f32;
        out[i] = if v == 0.0 { 0.0 } else { v };
    }
    [
        result.scatter_amount,
        result.scatter_spatial_scale,
        result.halation_amount,
        result.halation_spatial_scale,
    ] = out;
    Ok(result)
}

pub fn diffusion(active: bool, family: i32, values: [f64; 9]) -> optics::DiffusionParams {
    optics::DiffusionParams {
        active,
        family: family.clamp(0, 3) as u32,
        strength: values[0],
        spatial_scale: values[1],
        halo_warmth: values[2],
        core_intensity: values[3],
        core_size: values[4],
        halo_intensity: values[5],
        halo_size: values[6],
        bloom_intensity: values[7],
        bloom_size: values[8],
    }
}
