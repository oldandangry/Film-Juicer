//! Callback-local visual-grain mathematics. Static recipe policy, CUDA scratch
//! selection and the final descriptor identity have separate existing owners.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    fn scalar(v: &Value) -> f32 {
        f32::from_bits(v.as_u64().unwrap() as u32)
    }
    fn wide(v: &Value) -> f64 {
        f64::from_bits(v.as_u64().unwrap())
    }
    fn floats(v: &Value) -> [f32; 3] {
        std::array::from_fn(|c| scalar(&v[c]))
    }
    fn domain(v: &Value) -> Domain {
        Domain {
            x: v[0].as_i64().unwrap() as i32,
            y: v[1].as_i64().unwrap() as i32,
            width: v[2].as_i64().unwrap() as i32,
            height: v[3].as_i64().unwrap() as i32,
        }
    }
    fn polarity(v: &Value) -> Option<Polarity> {
        match v.as_u64().unwrap() {
            0 => Some(Polarity::Negative),
            1 => Some(Polarity::Positive),
            _ => None,
        }
    }
    fn input(v: &Value) -> Input {
        Input {
            active: v["active"].as_bool().unwrap(),
            sublayers_active: v["sublayers_active"].as_bool().unwrap(),
            layer_curves_required: v["layer_curves_required"].as_bool().unwrap(),
            capture_polarity: polarity(&v["capture_polarity"]),
            film_polarity: polarity(&v["film_polarity"]),
            sublayer_count: v["sublayer_count"].as_i64().unwrap() as i32,
            pixel_size_um: scalar(&v["pixel_size_um"]),
            particle_area_um2: scalar(&v["particle_area_um2"]),
            particle_scale_cmy: floats(&v["particle_scale_cmy"]),
            particle_scale_layers: floats(&v["particle_scale_layers"]),
            particle_density_min_cmy: floats(&v["particle_density_min_cmy"]),
            authored_max_cmy: floats(&v["authored_max_cmy"]),
            correlation_sigma_px: scalar(&v["correlation_sigma_px"]),
            dye_cloud_blur_um: scalar(&v["dye_cloud_blur_um"]),
            size_mix_scale: scalar(&v["size_mix_scale"]),
            fine_weight: scalar(&v["fine_weight"]),
            mid_weight: scalar(&v["mid_weight"]),
            coarse_weight: scalar(&v["coarse_weight"]),
            clump_morph_period_sec: scalar(&v["clump_morph_period_sec"]),
            render_extent: domain(&v["render_extent"]),
            full_frame_extent: domain(&v["full_frame_extent"]),
            frame_time: wide(&v["frame_time"]),
            frame_rate: wide(&v["frame_rate"]),
            session_seed: v["session_seed"].as_u64().unwrap(),
            clip_token: v["clip_token"].as_u64().unwrap(),
            recipe_hash: v["recipe_hash"].as_u64().unwrap(),
            recipe_layers_hash: v["recipe_layers_hash"].as_u64().unwrap(),
            film_layers_hash: v["film_layers_hash"].as_u64().unwrap(),
        }
    }
    fn gaussian(g: &Gaussian) -> Value {
        json!([g.sigma_px().to_bits(), g.radius(), g.hash()])
    }
    fn fields(p: &Plan) -> Value {
        let mut out = json!({"active": true});
        out["capture_polarity"] = json!(match p.capture_polarity() {
            Polarity::Negative => 0,
            Polarity::Positive => 1,
        });
        out["render_extent"] = {
            let d = p.render_extent();
            json!([d.x, d.y, d.width, d.height])
        };
        out["full_frame_extent"] = {
            let d = p.full_frame_extent();
            json!([d.x, d.y, d.width, d.height])
        };
        out["pixel_size_um"] = json!(p.pixel_size_um().to_bits());
        out["frame0"] = json!(p.frame0());
        out["frame_alpha"] = json!(p.frame_alpha().to_bits());
        out["seed_base"] = json!(p.seed_base());
        out["seed_base_next"] = json!(p.seed_base_next());
        out["session_seed"] = json!(p.session_seed());
        out["clip_token"] = json!(p.clip_token());
        out["pitch_px"] = json!(p.pitch_px());
        out["breathing_period_frames"] = json!(p.breathing_period_frames());
        out["clump_morph_period_frames"] = json!(p.clump_morph_period_frames());
        out["wang_cell_mm"] = json!(p.wang_cell_mm().to_bits());
        out["breathing_amplitude"] = json!(p.breathing_amplitude().to_bits());
        out["breathing_cell_um_small"] = json!(p.breathing_cell_um_small().to_bits());
        out["breathing_cell_um_large"] = json!(p.breathing_cell_um_large().to_bits());
        out["breathing_mix"] = json!(p.breathing_mix().to_bits());
        out["breathing_drift_um_per_frame"] = json!(p.breathing_drift_um_per_frame().to_bits());
        out["debug_scale"] = json!(p.debug_scale().to_bits());
        out["density_max_cmy"] = json!(p.density_max_cmy().map(f32::to_bits));
        out["n_particles_cmy"] = json!(p.n_particles_cmy().map(f32::to_bits));
        out["od_particle_cmy"] = json!(p.od_particle_cmy().map(f32::to_bits));
        out["density_min_layers"] = json!(p.density_min_layers().map(|row| row.map(f32::to_bits)));
        out["density_max_layers"] = json!(p.density_max_layers().map(|row| row.map(f32::to_bits)));
        out["n_particles_layers"] = json!(p.n_particles_layers().map(|row| row.map(f32::to_bits)));
        out["od_particle_layers"] = json!(p.od_particle_layers().map(|row| row.map(f32::to_bits)));
        out["correlation"] = Value::Array(p.correlation().iter().map(gaussian).collect());
        out["dye_cloud"] = Value::Array(
            p.dye_cloud()
                .iter()
                .map(|row| Value::Array(row.iter().map(gaussian).collect()))
                .collect(),
        );
        out["size_mix_weights"] = json!(p.size_mix_weights().map(f32::to_bits));
        out["size_mix_gain"] = json!(p.size_mix_gain().to_bits());
        out["requires_full_frame"] = json!(p.requires_full_frame());
        out["recipe_hash"] = json!(p.recipe_hash());
        out["density_layers_hash"] = json!(p.density_layers_hash());
        out
    }
    fn captured() -> Value {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/grain/frame-fixtures");
        let preset = match (cfg!(windows), cfg!(debug_assertions)) {
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
        };
        serde_json::from_slice(&std::fs::read(root.join(format!("{preset}.json"))).unwrap())
            .unwrap()
    }
    #[test]
    fn complete_native_values_and_ordered_failures() {
        let captured = captured();
        assert_eq!(captured["cases"].as_array().unwrap().len(), 153);
        for row in captured["cases"].as_array().unwrap() {
            let v = &row["input"];
            if v["missing_recipe"] == true || v["missing_film"] == true {
                continue;
            }
            let curves: Vec<Vec<Vec<f32>>> = v["layer_curves"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|curve| curve.as_array().unwrap().iter().map(scalar).collect())
                        .collect()
                })
                .collect();
            let mut reads = Vec::new();
            let result = Plan::new(input(v), |l, c| {
                reads.push((l, c));
                Ok(&curves[l][c])
            });
            assert_eq!(
                result.is_ok(),
                row["success"].as_bool().unwrap(),
                "{}",
                v["name"]
            );
            if let Ok(Some(plan)) = result {
                let mut expected = row["native"].clone();
                expected.as_object_mut().unwrap().remove("scratch_shape");
                expected.as_object_mut().unwrap().remove("hash");
                assert_eq!(fields(&plan), expected, "{}", v["name"]);
            } else if let Err(error) = result {
                assert_eq!(
                    error.field() as u32,
                    v["expected_field"].as_u64().unwrap() as u32,
                    "{}",
                    v["name"]
                );
                assert_eq!(
                    error.layer(),
                    v["expected_layer"].as_u64().unwrap() as usize,
                    "{}",
                    v["name"]
                );
                assert_eq!(
                    error.channel(),
                    v["expected_channel"].as_u64().unwrap() as usize,
                    "{}",
                    v["name"]
                );
                assert!(!error.layer_access());
            }
            if v["name"] == "S4D6-R1" {
                assert_eq!(
                    reads,
                    (0..3)
                        .flat_map(|l| (0..3).map(move |c| (l, c)))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn independently_captured_primitives() {
        let captured = captured();
        assert_eq!(captured["primitives"].as_array().unwrap().len(), 50);
        for row in captured["primitives"].as_array().unwrap() {
            let v = &row["input"];
            let expected = &row["result"];
            let actual = match v["op"].as_str().unwrap() {
                "gaussian" => gaussian(&Gaussian::new(scalar(&v["input"]))),
                "energy" => json!(energy(scalar(&v["input"])).to_bits()),
                "sqrt" => json!(scalar(&v["input"]).sqrt().to_bits()),
                "period" => json!(wide(&v["input"]).round() as i64),
                "exp" => {
                    let bits = wide(&v["input"]).exp().to_bits();
                    if json!(bits) != *expected {
                        eprintln!(
                            "internal f64 exp observation: input={} native={} Rust={}",
                            v["input"], expected, bits
                        );
                    }
                    continue;
                }
                _ => panic!("closed primitive cohort"),
            };
            assert_eq!(actual, *expected, "{v}");
        }
    }
    #[test]
    fn bounded_conversion_and_deferred_borrowing() {
        let captured = captured();
        let mut v = input(&captured["cases"][0]["input"]);
        for rate in [f64::MAX, 2.0_f64.powi(63) / 2.5] {
            v.frame_rate = rate;
            let error = Plan::new(v, |_, _| panic!("period failure must precede layer access"))
                .unwrap_err();
            assert_eq!(error.field(), Field::PeriodFrames);
        }
        v = input(&captured["cases"][0]["input"]);
        v.sublayers_active = true;
        let error = Plan::new(v, |l, c| {
            assert_eq!((l, c), (0, 0));
            Err(LayerAccessError::Invalid)
        })
        .unwrap_err();
        assert_eq!(
            (
                error.field(),
                error.layer(),
                error.channel(),
                error.layer_access()
            ),
            (Field::LayerCurve, 0, 0, true)
        );
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/grain/frame-fixtures/checked.json");
        let checked: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for row in checked["gaussian_safety"].as_array().unwrap() {
            assert_eq!(
                gaussian(&Gaussian::new(scalar(&row["input"]))),
                row["expected"]
            );
        }
        assert_eq!(maximum(-0.0, 0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(minimum(-0.0, 0.0).to_bits(), (-0.0_f32).to_bits());
        assert!(maximum(f32::NAN, 1.0).is_nan());
        assert!(minimum(f32::NAN, 1.0).is_nan());
    }
}

use crate::frame::Domain;
use crate::hash;
use crate::profile::Polarity;

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub active: bool,
    pub sublayers_active: bool,
    pub layer_curves_required: bool,
    pub capture_polarity: Option<Polarity>,
    pub film_polarity: Option<Polarity>,
    pub sublayer_count: i32,
    pub pixel_size_um: f32,
    pub particle_area_um2: f32,
    pub particle_scale_cmy: [f32; 3],
    pub particle_scale_layers: [f32; 3],
    pub particle_density_min_cmy: [f32; 3],
    pub authored_max_cmy: [f32; 3],
    pub correlation_sigma_px: f32,
    pub dye_cloud_blur_um: f32,
    pub size_mix_scale: f32,
    pub fine_weight: f32,
    pub mid_weight: f32,
    pub coarse_weight: f32,
    pub clump_morph_period_sec: f32,
    pub render_extent: Domain,
    pub full_frame_extent: Domain,
    pub frame_time: f64,
    pub frame_rate: f64,
    pub session_seed: u64,
    pub clip_token: u64,
    pub recipe_hash: u64,
    pub recipe_layers_hash: u64,
    pub film_layers_hash: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Gaussian {
    sigma_px: f32,
    radius: i32,
    hash: u64,
}

impl Gaussian {
    const EMPTY: Self = Self {
        sigma_px: 0.0,
        radius: 0,
        hash: 0,
    };
    fn new(sigma: f32) -> Self {
        if !positive(sigma) {
            return Self::EMPTY;
        }
        let expression = 4.0_f32 * sigma + 0.5_f32;
        if !positive(expression) {
            return Self::EMPTY;
        }
        // Cap before conversion: the uncapped finite expression may exceed i32.
        let radius = if expression >= 75.0 {
            75
        } else {
            expression as i32
        };
        if radius <= 0 {
            return Self::EMPTY;
        }
        let mut hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut hash, &sigma.to_bits().to_le_bytes());
        hash::update_bytes(&mut hash, &radius.to_le_bytes());
        Self {
            sigma_px: sigma,
            radius,
            hash,
        }
    }
    pub fn sigma_px(&self) -> f32 {
        self.sigma_px
    }
    pub fn radius(&self) -> i32 {
        self.radius
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerAccessError {
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Field {
    FrameInput = 1,
    DensityLayers = 2,
    FrameTime = 3,
    PeriodFrames = 4,
    PixelArea = 5,
    ParticleModel = 6,
    ParticleCount = 7,
    SizeMix = 8,
    LayerCurve = 9,
    DensityLayerTotal = 10,
    LayerParticleArea = 11,
    LayerParticleCount = 12,
    PartialFrame = 13,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error {
    field: Field,
    layer: usize,
    channel: usize,
    layer_access: bool,
}
impl Error {
    fn new(field: Field) -> Self {
        Self {
            field,
            layer: 3,
            channel: 3,
            layer_access: false,
        }
    }
    fn for_channel(field: Field, channel: usize) -> Self {
        Self {
            channel,
            ..Self::new(field)
        }
    }
    fn cell(field: Field, layer: usize, channel: usize) -> Self {
        Self {
            layer,
            channel,
            ..Self::new(field)
        }
    }
    pub fn field(&self) -> Field {
        self.field
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn channel(&self) -> usize {
        self.channel
    }
    pub fn layer_access(&self) -> bool {
        self.layer_access
    }
}

#[derive(Debug)]
pub struct Plan {
    capture_polarity: Polarity,
    render_extent: Domain,
    full_frame_extent: Domain,
    pixel_size_um: f32,
    frame0: i64,
    frame_alpha: f32,
    seed_base: u64,
    seed_base_next: u64,
    session_seed: u64,
    clip_token: u64,
    pitch_px: i32,
    breathing_period_frames: i32,
    clump_morph_period_frames: i32,
    wang_cell_mm: f32,
    breathing_amplitude: f32,
    breathing_cell_um_small: f32,
    breathing_cell_um_large: f32,
    breathing_mix: f32,
    breathing_drift_um_per_frame: f32,
    debug_scale: f32,
    density_max_cmy: [f32; 3],
    n_particles_cmy: [f32; 3],
    od_particle_cmy: [f32; 3],
    density_min_layers: [[f32; 3]; 3],
    density_max_layers: [[f32; 3]; 3],
    n_particles_layers: [[f32; 3]; 3],
    od_particle_layers: [[f32; 3]; 3],
    correlation: [Gaussian; 3],
    dye_cloud: [[Gaussian; 3]; 3],
    size_mix_weights: [f32; 3],
    size_mix_gain: f32,
    requires_full_frame: bool,
    recipe_hash: u64,
    density_layers_hash: u64,
}

fn positive(v: f32) -> bool {
    v.is_finite() && v > 0.0
}
// Native std::min/max keep the first operand on an unordered comparison or tie.
fn maximum(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}
fn minimum(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

fn contains(outer: Domain, inner: Domain) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && i64::from(inner.x) + i64::from(inner.width)
            <= i64::from(outer.x) + i64::from(outer.width)
        && i64::from(inner.y) + i64::from(inner.height)
            <= i64::from(outer.y) + i64::from(outer.height)
}

fn period(value: f64) -> Result<i32, Error> {
    let rounded = value.round();
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded >= i64::MAX as f64 {
        return Err(Error::new(Field::PeriodFrames));
    }
    Ok(((rounded as i64) as i32).max(1))
}

fn seed(clip: u64, frame: i64, session: u64) -> u64 {
    let hash = hash::u64s(&[clip, frame as u64, session, 1]);
    if hash == 0 { 1 } else { hash }
}

fn energy(sigma: f32) -> f32 {
    let gaussian = Gaussian::new(sigma);
    if gaussian.radius <= 0 {
        return 1.0;
    }
    let sigma_squared = f64::from(sigma) * f64::from(sigma) * 2.0;
    let mut weights = [0.0_f64; 151];
    let width = (gaussian.radius * 2 + 1) as usize;
    let mut sum = 0.0;
    for (index, weight) in weights[..width].iter_mut().enumerate() {
        let offset = index as i32 - gaussian.radius;
        let distance = f64::from(offset);
        *weight = (-(distance * distance) / sigma_squared).exp();
        sum += *weight;
    }
    if !(sum > 0.0 && sum.is_finite()) {
        return 1.0;
    }
    let inverse_sum = 1.0 / sum;
    let mut squared_sum = 0.0;
    for &weight in &weights[..width] {
        // Grain energy uses the original f64 samples, unlike upload coefficients.
        let normalized = weight * inverse_sum;
        squared_sum += normalized * normalized;
    }
    let energy = squared_sum * squared_sum;
    (if 1.0e-12 < energy { energy } else { 1.0e-12 }) as f32
}

impl Plan {
    pub fn new<'a>(
        input: Input,
        mut layers: impl FnMut(usize, usize) -> Result<&'a [f32], LayerAccessError>,
    ) -> Result<Option<Self>, Error> {
        if !input.active {
            return Ok(None);
        }
        let render = input.render_extent;
        let full = input.full_frame_extent;
        let Some(polarity) = input.capture_polarity else {
            return Err(Error::new(Field::FrameInput));
        };
        if input.recipe_hash == 0
            || input.film_polarity != Some(polarity)
            || render.width <= 0
            || render.height <= 0
            || full.width <= 0
            || full.height <= 0
            || !contains(full, render)
            || !positive(input.pixel_size_um)
            || !input.frame_time.is_finite()
            || !input.frame_rate.is_finite()
            || input.frame_rate <= 0.0
        {
            return Err(Error::new(Field::FrameInput));
        }
        if input.sublayers_active
            && (!input.layer_curves_required
                || input.film_layers_hash == 0
                || input.recipe_layers_hash != input.film_layers_hash)
        {
            return Err(Error::new(Field::DensityLayers));
        }
        let floor = input.frame_time.floor();
        if floor < i64::MIN as f64 || floor >= i64::MAX as f64 {
            return Err(Error::new(Field::FrameTime));
        }
        let frame0 = floor as i64;
        let frame_alpha = (input.frame_time - floor).clamp(0.0, 1.0) as f32;
        let breathing_period_frames = period(input.frame_rate * 2.5)?;
        let clump_morph_period_frames =
            period(input.frame_rate * f64::from(input.clump_morph_period_sec))?;
        let film_format_mm = input.pixel_size_um * full.width.max(full.height) as f32 / 1000.0;
        let film_scale = film_format_mm / 10.0;
        let pixel_area = input.pixel_size_um * input.pixel_size_um;
        if !positive(pixel_area) {
            return Err(Error::new(Field::PixelArea));
        }
        let mut density_max_cmy = [0.0; 3];
        let mut n_particles_cmy = [0.0; 3];
        let mut od_particle_cmy = [0.0; 3];
        let mut area_ratio_sum = 0.0;
        for channel in 0..3 {
            let max = input.authored_max_cmy[channel] + input.particle_density_min_cmy[channel];
            let area = input.particle_area_um2 * input.particle_scale_cmy[channel];
            let reference_area = 0.335_f32 * [1.10_f32, 1.27, 2.08][channel];
            if !positive(max) || !positive(area) || !positive(reference_area) {
                return Err(Error::for_channel(Field::ParticleModel, channel));
            }
            let mut count = pixel_area / area;
            if input.sublayer_count > 1 {
                count /= input.sublayer_count as f32;
            }
            if !positive(count) {
                return Err(Error::for_channel(Field::ParticleCount, channel));
            }
            density_max_cmy[channel] = max;
            n_particles_cmy[channel] = count;
            od_particle_cmy[channel] = max / count;
            area_ratio_sum += area / reference_area;
        }
        let mean = (density_max_cmy[0] + density_max_cmy[1] + density_max_cmy[2]) / 3.0;
        let debug_scale = 0.25 / maximum(1.0e-6, mean);
        let mut fine = input.correlation_sigma_px * maximum(area_ratio_sum / 3.0, 0.0).sqrt();
        let enabled = if input.sublayers_active {
            fine > 0.0
        } else {
            fine > 0.4
        };
        if !enabled || !fine.is_finite() {
            fine = 0.0;
        }
        let mut mixed = fine > 0.0
            && input.size_mix_scale > 1.0
            && (input.mid_weight > 0.0 || input.coarse_weight > 0.0);
        let mut mid = 0.0;
        let mut coarse = 0.0;
        if mixed {
            let raw = fine * input.size_mix_scale.sqrt();
            coarse = maximum(fine, minimum(raw, fine * 4.0));
            mid = maximum(0.0, fine * coarse).sqrt();
            mixed = positive(mid) && positive(coarse);
        }
        let mut correlation = [Gaussian::EMPTY; 3];
        correlation[0] = Gaussian::new(fine);
        let mut size_mix_weights = [1.0, 0.0, 0.0];
        let mut size_mix_gain = 1.0;
        if mixed {
            correlation[1] = Gaussian::new(mid);
            correlation[2] = Gaussian::new(coarse);
            size_mix_weights = [input.fine_weight, input.mid_weight, input.coarse_weight];
            let fine_energy = energy(fine);
            let mid_energy = energy(mid);
            let coarse_energy = energy(coarse);
            let mid_scale = input.size_mix_scale.sqrt();
            let ratio_mid = mid_scale * mid_energy / maximum(1.0e-12, fine_energy);
            let ratio_coarse = input.size_mix_scale * coarse_energy / maximum(1.0e-12, fine_energy);
            let denominator = input.fine_weight * input.fine_weight
                + input.mid_weight * input.mid_weight * ratio_mid
                + input.coarse_weight * input.coarse_weight * ratio_coarse;
            if !denominator.is_finite() || denominator <= 1.0e-12 {
                return Err(Error::new(Field::SizeMix));
            }
            size_mix_gain = 1.0 / denominator.sqrt();
        }
        let mut density_min_layers = [[0.0; 3]; 3];
        let mut density_max_layers = [[0.0; 3]; 3];
        let mut n_particles_layers = [[0.0; 3]; 3];
        let mut od_particle_layers = [[0.0; 3]; 3];
        let mut dye_cloud = [[Gaussian::EMPTY; 3]; 3];
        if input.sublayers_active {
            let mut maxima = [[0.0; 3]; 3];
            for (layer, row) in maxima.iter_mut().enumerate() {
                for (channel, max) in row.iter_mut().enumerate() {
                    let curve = layers(layer, channel).map_err(|_| Error {
                        layer_access: true,
                        ..Error::cell(Field::LayerCurve, layer, channel)
                    })?;
                    let mut maximum = f64::NEG_INFINITY;
                    let mut found = false;
                    for &sample in curve {
                        if sample.is_finite() {
                            if maximum < f64::from(sample) {
                                maximum = f64::from(sample);
                            }
                            found = true;
                        }
                    }
                    *max = maximum as f32;
                    if !found || !max.is_finite() {
                        return Err(Error::cell(Field::LayerCurve, layer, channel));
                    }
                }
            }
            // S4D6-R1: all nine reductions, then each CMY total immediately
            // followed by that channel's three layers, before the next total.
            for channel in 0..3 {
                let mut total = 0.0;
                for row in &maxima {
                    total += row[channel];
                }
                if !positive(total) {
                    return Err(Error::for_channel(Field::DensityLayerTotal, channel));
                }
                for layer in 0..3 {
                    let fraction = maxima[layer][channel] / total;
                    let min = fraction * input.particle_density_min_cmy[channel];
                    let max = maxima[layer][channel] + min;
                    let area = input.particle_area_um2
                        * input.particle_scale_cmy[channel]
                        * input.particle_scale_layers[layer];
                    if !positive(area) {
                        return Err(Error::cell(Field::LayerParticleArea, layer, channel));
                    }
                    let count = pixel_area * fraction / area;
                    if !positive(count) {
                        return Err(Error::cell(Field::LayerParticleCount, layer, channel));
                    }
                    let od = max / count;
                    density_min_layers[layer][channel] = min;
                    density_max_layers[layer][channel] = max;
                    n_particles_layers[layer][channel] = count;
                    od_particle_layers[layer][channel] = od;
                    dye_cloud[layer][channel] =
                        Gaussian::new(input.dye_cloud_blur_um * maximum(0.0, od).sqrt());
                }
            }
        }
        let requires_full_frame = correlation
            .iter()
            .chain(dye_cloud.iter().flatten())
            .any(|g| g.radius > 0);
        if requires_full_frame && render != full {
            return Err(Error::new(Field::PartialFrame));
        }
        Ok(Some(Self {
            capture_polarity: polarity,
            render_extent: render,
            full_frame_extent: full,
            pixel_size_um: input.pixel_size_um,
            frame0,
            frame_alpha,
            seed_base: seed(input.clip_token, frame0, input.session_seed),
            seed_base_next: seed(input.clip_token, frame0 + 1, input.session_seed),
            session_seed: input.session_seed,
            clip_token: input.clip_token,
            pitch_px: full.height,
            breathing_period_frames,
            clump_morph_period_frames,
            wang_cell_mm: 2.0,
            breathing_amplitude: 0.01902219,
            breathing_cell_um_small: 2500.0 * film_scale,
            breathing_cell_um_large: 5000.0 * film_scale,
            breathing_mix: 0.30,
            breathing_drift_um_per_frame: 1.0,
            debug_scale,
            density_max_cmy,
            n_particles_cmy,
            od_particle_cmy,
            density_min_layers,
            density_max_layers,
            n_particles_layers,
            od_particle_layers,
            correlation,
            dye_cloud,
            size_mix_weights,
            size_mix_gain,
            requires_full_frame,
            recipe_hash: input.recipe_hash,
            density_layers_hash: if input.sublayers_active {
                input.film_layers_hash
            } else {
                0
            },
        }))
    }

    pub fn capture_polarity(&self) -> Polarity {
        self.capture_polarity
    }
    pub fn render_extent(&self) -> Domain {
        self.render_extent
    }
    pub fn full_frame_extent(&self) -> Domain {
        self.full_frame_extent
    }
    pub fn pixel_size_um(&self) -> f32 {
        self.pixel_size_um
    }
    pub fn frame0(&self) -> i64 {
        self.frame0
    }
    pub fn frame_alpha(&self) -> f32 {
        self.frame_alpha
    }
    pub fn seed_base(&self) -> u64 {
        self.seed_base
    }
    pub fn seed_base_next(&self) -> u64 {
        self.seed_base_next
    }
    pub fn session_seed(&self) -> u64 {
        self.session_seed
    }
    pub fn clip_token(&self) -> u64 {
        self.clip_token
    }
    pub fn pitch_px(&self) -> i32 {
        self.pitch_px
    }
    pub fn breathing_period_frames(&self) -> i32 {
        self.breathing_period_frames
    }
    pub fn clump_morph_period_frames(&self) -> i32 {
        self.clump_morph_period_frames
    }
    pub fn wang_cell_mm(&self) -> f32 {
        self.wang_cell_mm
    }
    pub fn breathing_amplitude(&self) -> f32 {
        self.breathing_amplitude
    }
    pub fn breathing_cell_um_small(&self) -> f32 {
        self.breathing_cell_um_small
    }
    pub fn breathing_cell_um_large(&self) -> f32 {
        self.breathing_cell_um_large
    }
    pub fn breathing_mix(&self) -> f32 {
        self.breathing_mix
    }
    pub fn breathing_drift_um_per_frame(&self) -> f32 {
        self.breathing_drift_um_per_frame
    }
    pub fn debug_scale(&self) -> f32 {
        self.debug_scale
    }
    pub fn density_max_cmy(&self) -> &[f32; 3] {
        &self.density_max_cmy
    }
    pub fn n_particles_cmy(&self) -> &[f32; 3] {
        &self.n_particles_cmy
    }
    pub fn od_particle_cmy(&self) -> &[f32; 3] {
        &self.od_particle_cmy
    }
    pub fn density_min_layers(&self) -> &[[f32; 3]; 3] {
        &self.density_min_layers
    }
    pub fn density_max_layers(&self) -> &[[f32; 3]; 3] {
        &self.density_max_layers
    }
    pub fn n_particles_layers(&self) -> &[[f32; 3]; 3] {
        &self.n_particles_layers
    }
    pub fn od_particle_layers(&self) -> &[[f32; 3]; 3] {
        &self.od_particle_layers
    }
    pub fn correlation(&self) -> &[Gaussian; 3] {
        &self.correlation
    }
    pub fn dye_cloud(&self) -> &[[Gaussian; 3]; 3] {
        &self.dye_cloud
    }
    pub fn size_mix_weights(&self) -> &[f32; 3] {
        &self.size_mix_weights
    }
    pub fn size_mix_gain(&self) -> f32 {
        self.size_mix_gain
    }
    pub fn requires_full_frame(&self) -> bool {
        self.requires_full_frame
    }
    pub fn recipe_hash(&self) -> u64 {
        self.recipe_hash
    }
    pub fn density_layers_hash(&self) -> u64 {
        self.density_layers_hash
    }
}
