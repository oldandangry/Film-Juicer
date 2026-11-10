//! Identity of completed host controls before preparation admission.
use crate::{hash, optics, recipe::Params, route::ScanRoute};

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
pub fn hash_params(p: &Params, film_key: &[u8], print_key: &[u8]) -> u64 {
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
    h.int(p.ref_ill);
    for v in [
        p.print_shadow_compensation_factor,
        p.print_shadow_compensation_density,
        p.print_shadow_compensation_transition,
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
        p.scanner_lut_resolution,
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
    let g = &p.grain;
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
    pub fn new(params: Params, film_key: Vec<u8>, print_key: Vec<u8>) -> Self {
        let identity = hash_params(&params, &film_key, &print_key);
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
    let diffusion = optics::diffusion(false, 1, [0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    Params {
        print_exposure: 1.0,
        print_preflash_exposure: 0.0,
        print_gamma_factor: 1.0,
        preflash_m_filter_cc: 0.0,
        preflash_y_filter_cc: 0.0,
        print_shadow_compensation_factor: 0.0,
        print_shadow_compensation_density: 1.2,
        print_shadow_compensation_transition: 0.3,
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
        ref_ill: 0,
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
        scatter_halation: crate::recipe::HalationControls {
            active: false,
            scatter_amount: 0.0,
            scatter_spatial_scale: 0.0,
            halation_amount: 0.0,
            halation_spatial_scale: 0.0,
        },
        scanner_black_correction: 0,
        scanner_white_correction: 0,
        scanner_lut_resolution: 17,
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
        film_dust_amount: 0.0,
        film_scratch_amount: 0.0,
        gate_dust_amount: 0.0,
        gate_scratch_amount: 0.0,
        scan_route: ScanRoute::NegativePrint,
        glare_active: true,
        camera_filter_override: false,
    }
}
