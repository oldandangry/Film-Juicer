//! Completed spatial-optics recipes from captured controls and copied profile defaults.

use crate::hash;

pub mod diffusion_frame;

mod gaussian;
pub use gaussian::{GaussianKernel, GaussianKernelError};

mod halation_frame;
pub use halation_frame::{
    HalationChannel, HalationFrame, HalationFrameError, HalationFrameField, HalationFrameInput,
    HalationGaussian, HalationGaussianKind,
};

// Matches the frozen native v1 resolved-hash contract. Native authored-control
// and PSF-sample hashes are distinct identities; their schema is
// kDiffusionHostBehaviorSchemaVersion in DiffusionHostBehavior.h.
const DIFFUSION_RESOLVED_HASH_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub struct DiffusionParams {
    pub active: bool,
    pub family: u32,
    pub strength: f64,
    pub spatial_scale: f64,
    pub halo_warmth: f64,
    pub core_intensity: f64,
    pub core_size: f64,
    pub halo_intensity: f64,
    pub halo_size: f64,
    pub bloom_intensity: f64,
    pub bloom_size: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct HalationParams {
    pub active: bool,
    pub scatter_amount: f32,
    pub scatter_spatial_scale: f32,
    pub halation_amount: f32,
    pub halation_spatial_scale: f32,
    pub first_sigma_um: [f32; 3],
    pub primary_amount: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub camera_diffusion: DiffusionParams,
    pub camera_lens_blur_um: f32,
    pub scatter_halation: HalationParams,
    pub enlarger_diffusion: DiffusionParams,
    pub print_domain: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    Glimmerglass,
    BlackProMist,
    ProMist,
    Cinebloom,
}
impl Family {
    fn hash_tag(self) -> u8 {
        match self {
            Self::Glimmerglass => 0,
            Self::BlackProMist => 1,
            Self::ProMist => 2,
            Self::Cinebloom => 3,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Domain {
    FilmLinearExposure,
    PrintLinearExposure,
}
impl Domain {
    fn hash_tag(self) -> u8 {
        match self {
            Self::FilmLinearExposure => 0,
            Self::PrintLinearExposure => 1,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Component {
    CameraDiffusion,
    CameraLensBlur,
    ScatterHalation,
    EnlargerDiffusion,
}
impl Component {
    fn hash_tag(self) -> u8 {
        match self {
            Self::CameraDiffusion => 0,
            Self::CameraLensBlur => 1,
            Self::ScatterHalation => 2,
            Self::EnlargerDiffusion => 3,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Field {
    Family,
    Strength,
    SpatialScale,
    HaloWarmth,
    CoreIntensity,
    CoreSize,
    HaloIntensity,
    HaloSize,
    BloomIntensity,
    BloomSize,
    ScatterFraction,
    GroupWeights,
    GroupCenters,
    ResolvedHash,
    SigmaUm,
    TotalStrength,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error {
    component: Component,
    field: Field,
    channel: Option<usize>,
}
impl Error {
    fn new(component: Component, field: Field) -> Self {
        Self {
            component,
            field,
            channel: None,
        }
    }
    pub fn component(self) -> Component {
        self.component
    }
    pub fn field(self) -> Field {
        self.field
    }
    pub fn channel(self) -> Option<usize> {
        self.channel
    }
}

#[derive(Debug)]
pub struct Diffusion {
    domain: Domain,
    active: bool,
    family: Family,
    scatter_fraction: f64,
    group_weights: [f64; 3],
    group_lambda_um: [f64; 3],
    warmth: f64,
    spatial_scale: f64,
    resolved_hash: u64,
    hash: u64,
}
#[derive(Debug)]
pub struct LensBlur {
    sigma_um: f32,
    hash: u64,
}
#[derive(Debug)]
pub struct Halation {
    scatter_active: bool,
    back_reflection_active: bool,
    scatter_amount: f32,
    scatter_spatial_scale: f32,
    halation_spatial_scale: f32,
    first_sigma_um: [f32; 3],
    total_strength: [f32; 3],
    hash: u64,
}
#[derive(Debug)]
pub struct Recipe {
    camera_diffusion: Diffusion,
    camera_lens_blur: LensBlur,
    scatter_halation: Halation,
    enlarger_diffusion: Diffusion,
    hash: u64,
}

struct FamilyConfig {
    lambda_um: [f64; 3],
    weights: [f64; 3],
    warmth: f64,
    gain: f64,
}
const FAMILIES: [FamilyConfig; 4] = [
    FamilyConfig {
        lambda_um: [10.0, 50.0, 260.0],
        weights: [0.60, 0.30, 0.10],
        warmth: 0.0,
        gain: 0.65,
    },
    FamilyConfig {
        lambda_um: [16.0, 95.0, 380.0],
        weights: [0.40, 0.47, 0.13],
        warmth: 0.65,
        gain: 0.75,
    },
    FamilyConfig {
        lambda_um: [14.0, 150.0, 650.0],
        weights: [0.28, 0.42, 0.30],
        warmth: 0.40,
        gain: 1.05,
    },
    FamilyConfig {
        lambda_um: [20.0, 200.0, 1000.0],
        weights: [0.22, 0.30, 0.48],
        warmth: 0.85,
        gain: 1.0,
    },
];
const MINIMUM_SCALE: f64 = 1.0e-6;

fn ordered_max(value: f64, minimum: f64) -> f64 {
    // Keep the first operand's zero sign, as in the native ordered comparison.
    if value < minimum { minimum } else { value }
}
fn scatter_fraction(strength: f64, gain: f64) -> f64 {
    const BREAKPOINTS: [f64; 5] = [0.125, 0.25, 0.5, 1.0, 2.0];
    const FRACTIONS: [f64; 5] = [0.10, 0.20, 0.35, 0.55, 0.75];
    let log_strength = ordered_max(strength, MINIMUM_SCALE).log2();
    let first_break = BREAKPOINTS[0].log2();
    let last_break = BREAKPOINTS[4].log2();
    let mut fraction = FRACTIONS[0];
    if log_strength >= last_break {
        fraction = FRACTIONS[4];
    } else if log_strength > first_break {
        for index in 0..4 {
            let lower = BREAKPOINTS[index].log2();
            let upper = BREAKPOINTS[index + 1].log2();
            if log_strength <= upper {
                let t = (log_strength - lower) / (upper - lower);
                fraction = FRACTIONS[index] + t * (FRACTIONS[index + 1] - FRACTIONS[index]);
                break;
            }
        }
    }
    (fraction * gain).clamp(0.0, 0.99)
}
fn hash_double(hash: &mut u64, value: f64) {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    hash::update_bytes(hash, &bits.to_le_bytes());
}
fn canonical_float(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

impl Diffusion {
    fn inactive() -> Self {
        Self {
            domain: Domain::FilmLinearExposure,
            active: false,
            family: Family::BlackProMist,
            scatter_fraction: 0.0,
            group_weights: [0.0; 3],
            group_lambda_um: [0.0; 3],
            warmth: 0.0,
            spatial_scale: 0.0,
            resolved_hash: 0,
            hash: 0,
        }
    }
    fn new(
        params: DiffusionParams,
        domain: Domain,
        component: Component,
        #[cfg(any(test, feature = "test-support"))] zero_hash: bool,
    ) -> Result<Self, Error> {
        let family = match params.family {
            0 => Family::Glimmerglass,
            1 => Family::BlackProMist,
            2 => Family::ProMist,
            3 => Family::Cinebloom,
            _ => return Err(Error::new(component, Field::Family)),
        };
        for (value, field) in [
            (params.strength, Field::Strength),
            (params.spatial_scale, Field::SpatialScale),
            (params.halo_warmth, Field::HaloWarmth),
            (params.core_intensity, Field::CoreIntensity),
            (params.core_size, Field::CoreSize),
            (params.halo_intensity, Field::HaloIntensity),
            (params.halo_size, Field::HaloSize),
            (params.bloom_intensity, Field::BloomIntensity),
            (params.bloom_size, Field::BloomSize),
        ] {
            if !value.is_finite() {
                return Err(Error::new(component, field));
            }
        }
        if !params.active || params.strength <= 0.0 || params.spatial_scale <= 0.0 {
            return Ok(Self::inactive());
        }
        let config = &FAMILIES[usize::from(family.hash_tag())];
        let scatter_fraction = scatter_fraction(params.strength, config.gain);
        if !(scatter_fraction.is_finite() && scatter_fraction > 0.0) {
            return Err(Error::new(component, Field::ScatterFraction));
        }
        let intensities = [
            params.core_intensity,
            params.halo_intensity,
            params.bloom_intensity,
        ];
        let mut group_weights = [0.0; 3];
        let mut total = 0.0;
        for group in 0..3 {
            group_weights[group] = config.weights[group] * ordered_max(intensities[group], 0.0);
            total += group_weights[group];
        }
        let mut group_lambda_um = config.lambda_um;
        if total <= 0.0 {
            group_weights = config.weights;
        } else {
            if !total.is_finite() {
                return Err(Error::new(component, Field::GroupWeights));
            }
            let sizes = [params.core_size, params.halo_size, params.bloom_size];
            for group in 0..3 {
                group_weights[group] /= total;
                group_lambda_um[group] =
                    config.lambda_um[group] * ordered_max(sizes[group], MINIMUM_SCALE);
            }
        }
        for center in group_lambda_um {
            if !(center.is_finite() && center > 0.0) {
                return Err(Error::new(component, Field::GroupCenters));
            }
        }
        let warmth = (config.warmth + params.halo_warmth).clamp(-1.5, 1.5);
        let spatial_scale = params.spatial_scale;
        let mut resolved_hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut resolved_hash, b"diffusion-resolved\0");
        hash::update_bytes(
            &mut resolved_hash,
            &DIFFUSION_RESOLVED_HASH_SCHEMA_VERSION.to_le_bytes(),
        );
        hash::update_bytes(&mut resolved_hash, &[family.hash_tag()]);
        for value in [
            scatter_fraction,
            group_weights[0],
            group_weights[1],
            group_weights[2],
            group_lambda_um[0],
            group_lambda_um[1],
            group_lambda_um[2],
            warmth,
            spatial_scale,
        ] {
            hash_double(&mut resolved_hash, value);
        }
        #[cfg(any(test, feature = "test-support"))]
        let resolved_hash = if zero_hash { 0 } else { resolved_hash };
        if resolved_hash == 0 {
            return Err(Error::new(component, Field::ResolvedHash));
        }
        let mut hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut hash, &[domain.hash_tag()]);
        hash::update_bytes(&mut hash, &resolved_hash.to_le_bytes());
        Ok(Self {
            domain,
            active: true,
            family,
            scatter_fraction,
            group_weights,
            group_lambda_um,
            warmth,
            spatial_scale,
            resolved_hash,
            hash,
        })
    }
}
impl LensBlur {
    fn new(sigma_um: f32) -> Result<Self, Error> {
        if !sigma_um.is_finite() {
            return Err(Error::new(Component::CameraLensBlur, Field::SigmaUm));
        }
        let sigma_um = if sigma_um < 0.0 { 0.0 } else { sigma_um };
        let hash = if sigma_um > 0.0 {
            hash::bytes(&sigma_um.to_le_bytes())
        } else {
            0
        };
        Ok(Self { sigma_um, hash })
    }
}
impl Halation {
    fn new(params: HalationParams) -> Result<Self, Error> {
        let mut result = Self {
            scatter_active: false,
            back_reflection_active: false,
            scatter_amount: 0.0,
            scatter_spatial_scale: 0.0,
            halation_spatial_scale: 0.0,
            first_sigma_um: [0.0; 3],
            total_strength: [0.0; 3],
            hash: 0,
        };
        if !params.active {
            return Ok(result);
        }
        result.scatter_active = params.scatter_amount > 0.0 && params.scatter_spatial_scale > 0.0;
        if result.scatter_active {
            result.scatter_amount = canonical_float(params.scatter_amount);
            result.scatter_spatial_scale = canonical_float(params.scatter_spatial_scale);
        }
        result.back_reflection_active = params.halation_amount > 0.0
            && params.halation_spatial_scale > 0.0
            && params.primary_amount.iter().any(|&v| v > 0.0)
            && params.first_sigma_um.iter().any(|&v| v > 0.0);
        if result.back_reflection_active {
            result.halation_spatial_scale = canonical_float(params.halation_spatial_scale);
            result.first_sigma_um = params.first_sigma_um;
            for channel in 0..3 {
                let strength =
                    f64::from(params.primary_amount[channel]) * f64::from(params.halation_amount);
                if !strength.is_finite() || strength < 0.0 || strength > f64::from(f32::MAX) {
                    return Err(Error {
                        component: Component::ScatterHalation,
                        field: Field::TotalStrength,
                        channel: Some(channel),
                    });
                }
                let narrowed = strength as f32;
                if !narrowed.is_finite() || narrowed < 0.0 {
                    return Err(Error {
                        component: Component::ScatterHalation,
                        field: Field::TotalStrength,
                        channel: Some(channel),
                    });
                }
                result.total_strength[channel] = canonical_float(narrowed);
            }
            if !result.total_strength.iter().any(|&v| v > 0.0) {
                result.back_reflection_active = false;
                result.halation_spatial_scale = 0.0;
                result.first_sigma_um = [0.0; 3];
                result.total_strength = [0.0; 3];
            }
        }
        if !result.scatter_active && !result.back_reflection_active {
            return Ok(result);
        }
        let mut hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut hash, &[u8::from(result.scatter_active)]);
        if result.scatter_active {
            hash::update_bytes(&mut hash, &result.scatter_amount.to_le_bytes());
            hash::update_bytes(&mut hash, &result.scatter_spatial_scale.to_le_bytes());
        }
        hash::update_bytes(&mut hash, &[u8::from(result.back_reflection_active)]);
        if result.back_reflection_active {
            hash::update_bytes(&mut hash, &result.halation_spatial_scale.to_le_bytes());
            for value in result.first_sigma_um {
                hash::update_bytes(&mut hash, &value.to_le_bytes());
            }
            for value in result.total_strength {
                hash::update_bytes(&mut hash, &value.to_le_bytes());
            }
        }
        result.hash = if hash == 0 { 1 } else { hash };
        Ok(result)
    }
}
impl Recipe {
    pub fn new(params: Params) -> Result<Self, Error> {
        #[cfg(any(test, feature = "test-support"))]
        let zero_hash = test_support::take();
        let camera_diffusion = Diffusion::new(
            params.camera_diffusion,
            Domain::FilmLinearExposure,
            Component::CameraDiffusion,
            #[cfg(any(test, feature = "test-support"))]
            zero_hash,
        )?;
        let camera_lens_blur = LensBlur::new(params.camera_lens_blur_um)?;
        let scatter_halation = Halation::new(params.scatter_halation)?;
        let enlarger_diffusion = if params.print_domain {
            Diffusion::new(
                params.enlarger_diffusion,
                Domain::PrintLinearExposure,
                Component::EnlargerDiffusion,
                #[cfg(any(test, feature = "test-support"))]
                zero_hash,
            )?
        } else {
            Diffusion::inactive()
        };
        let mut hash = hash::FNV_OFFSET;
        let mut contributed = false;
        for (component, identity) in [
            (Component::CameraDiffusion, camera_diffusion.hash),
            (Component::CameraLensBlur, camera_lens_blur.hash),
            (Component::ScatterHalation, scatter_halation.hash),
            (Component::EnlargerDiffusion, enlarger_diffusion.hash),
        ] {
            if identity != 0 {
                hash::update_bytes(&mut hash, &[component.hash_tag()]);
                hash::update_bytes(&mut hash, &identity.to_le_bytes());
                contributed = true;
            }
        }
        Ok(Self {
            camera_diffusion,
            camera_lens_blur,
            scatter_halation,
            enlarger_diffusion,
            hash: if contributed { hash } else { 0 },
        })
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use std::cell::Cell;
    thread_local! { static ZERO_HASH: Cell<bool> = const { Cell::new(false) }; }
    thread_local! { static GAUSSIAN_RESERVATION: Cell<bool> = const { Cell::new(false) }; }
    pub fn reject_gaussian_reservation(reject: bool) {
        GAUSSIAN_RESERVATION.set(reject);
    }
    pub(super) fn take_gaussian_reservation() -> bool {
        GAUSSIAN_RESERVATION.replace(false)
    }
    pub fn zero_next_resolved_hash() {
        ZERO_HASH.set(true);
    }
    pub(super) fn take() -> bool {
        ZERO_HASH.replace(false)
    }
}

impl Diffusion {
    pub fn domain(&self) -> Domain {
        self.domain
    }
    pub fn active(&self) -> bool {
        self.active
    }
    pub fn family(&self) -> Family {
        self.family
    }
    pub fn scatter_fraction(&self) -> f64 {
        self.scatter_fraction
    }
    pub fn group_weights(&self) -> &[f64; 3] {
        &self.group_weights
    }
    pub fn group_lambda_um(&self) -> &[f64; 3] {
        &self.group_lambda_um
    }
    pub fn warmth(&self) -> f64 {
        self.warmth
    }
    pub fn spatial_scale(&self) -> f64 {
        self.spatial_scale
    }
    pub fn resolved_hash(&self) -> u64 {
        self.resolved_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

impl LensBlur {
    pub fn sigma_um(&self) -> f32 {
        self.sigma_um
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

impl Halation {
    pub fn scatter_active(&self) -> bool {
        self.scatter_active
    }
    pub fn back_reflection_active(&self) -> bool {
        self.back_reflection_active
    }
    pub fn scatter_amount(&self) -> f32 {
        self.scatter_amount
    }
    pub fn scatter_spatial_scale(&self) -> f32 {
        self.scatter_spatial_scale
    }
    pub fn halation_spatial_scale(&self) -> f32 {
        self.halation_spatial_scale
    }
    pub fn first_sigma_um(&self) -> &[f32; 3] {
        &self.first_sigma_um
    }
    pub fn total_strength(&self) -> &[f32; 3] {
        &self.total_strength
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

impl Recipe {
    pub fn camera_diffusion(&self) -> &Diffusion {
        &self.camera_diffusion
    }
    pub fn camera_lens_blur(&self) -> &LensBlur {
        &self.camera_lens_blur
    }
    pub fn scatter_halation(&self) -> &Halation {
        &self.scatter_halation
    }
    pub fn enlarger_diffusion(&self) -> &Diffusion {
        &self.enlarger_diffusion
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::path::PathBuf;

    fn fixture(kind: &str) -> Value {
        let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
        };
        let directory =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/film/optics-fixtures");
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        serde_json::from_slice(
            &std::fs::read(directory.join(manifest["presets"][preset][kind].as_str().unwrap()))
                .unwrap(),
        )
        .unwrap()
    }
    fn diffusion_params(v: &Value) -> DiffusionParams {
        let d = |i: usize| f64::from_bits(v["f64"][i].as_u64().unwrap());
        DiffusionParams {
            active: v["active"].as_bool().unwrap(),
            family: v["family"].as_u64().unwrap() as u32,
            strength: d(0),
            spatial_scale: d(1),
            halo_warmth: d(2),
            core_intensity: d(3),
            core_size: d(4),
            halo_intensity: d(5),
            halo_size: d(6),
            bloom_intensity: d(7),
            bloom_size: d(8),
        }
    }
    fn params(v: &Value) -> Params {
        let f = |key: &str, i: usize| f32::from_bits(v[key][i].as_u64().unwrap() as u32);
        Params {
            camera_diffusion: diffusion_params(&v["camera"]),
            camera_lens_blur_um: f32::from_bits(v["lens_f32"].as_u64().unwrap() as u32),
            scatter_halation: HalationParams {
                active: v["halation_active"].as_bool().unwrap(),
                scatter_amount: f("halation_f32", 0),
                scatter_spatial_scale: f("halation_f32", 1),
                halation_amount: f("halation_f32", 2),
                halation_spatial_scale: f("halation_f32", 3),
                first_sigma_um: std::array::from_fn(|i| f("first_sigma_f32", i)),
                primary_amount: std::array::from_fn(|i| f("primary_f32", i)),
            },
            enlarger_diffusion: diffusion_params(&v["enlarger"]),
            print_domain: v["print"].as_bool().unwrap(),
        }
    }
    fn diffusion(v: &Diffusion) -> Value {
        json!({"domain":v.domain().hash_tag(),"active":v.active(),"family":v.family().hash_tag(),
            "f64":([v.scatter_fraction(),v.group_weights()[0],v.group_weights()[1],v.group_weights()[2],
            v.group_lambda_um()[0],v.group_lambda_um()[1],v.group_lambda_um()[2],v.warmth(),v.spatial_scale()].map(f64::to_bits)),
            "resolved_hash":v.resolved_hash(),"hash":v.hash()})
    }
    fn project(v: &Recipe) -> Value {
        let h = v.scatter_halation();
        json!({"camera":diffusion(v.camera_diffusion()),"enlarger":diffusion(v.enlarger_diffusion()),
            "lens_f32":v.camera_lens_blur().sigma_um().to_bits(),"lens_hash":v.camera_lens_blur().hash(),
            "halation":{"scatter_active":h.scatter_active(),"back_reflection_active":h.back_reflection_active(),
            "f32":([h.scatter_amount(),h.scatter_spatial_scale(),h.halation_spatial_scale(),h.first_sigma_um()[0],h.first_sigma_um()[1],h.first_sigma_um()[2],h.total_strength()[0],h.total_strength()[1],h.total_strength()[2]].map(f32::to_bits)),"hash":h.hash()},"hash":v.hash()})
    }
    fn diagnostic(error: Error) -> String {
        let field = match error.field() {
            Field::Family => "family",
            Field::Strength => "strength",
            Field::SpatialScale => "spatial_scale",
            Field::HaloWarmth => "halo_warmth",
            Field::CoreIntensity => "core_intensity",
            Field::CoreSize => "core_size",
            Field::HaloIntensity => "halo_intensity",
            Field::HaloSize => "halo_size",
            Field::BloomIntensity => "bloom_intensity",
            Field::BloomSize => "bloom_size",
            Field::ScatterFraction => "scatter_fraction",
            Field::GroupWeights => "group_weights",
            Field::GroupCenters => "group_centers_um",
            Field::ResolvedHash => "resolved_hash",
            Field::SigmaUm => "sigma_um",
            Field::TotalStrength => "total_strength",
        };
        let component = match error.component() {
            Component::CameraDiffusion => "camera_diffusion",
            Component::CameraLensBlur => "camera_lens_blur",
            Component::ScatterHalation => "scatter_halation",
            Component::EnlargerDiffusion => "enlarger_diffusion",
        };
        if let Some(channel) = error.channel() {
            format!(
                "ResourceDescriptorMismatch component={component} field={field} channel={channel}"
            )
        } else {
            format!("ResourceDescriptorMismatch phase=6A component={component} field={field}")
        }
    }
    #[test]
    fn matches_frozen_native_words_identities_and_failures() {
        let captured = fixture("literal");
        let rows = captured["cases"].as_array().unwrap();
        assert_eq!(rows.len(), 553);
        let mut mismatches = Vec::new();
        for row in rows {
            match Recipe::new(params(&row["input"])) {
                Ok(recipe) => {
                    if row["success"] != true || project(&recipe) != row["recipe"] {
                        mismatches.push(json!({"id":row["id"],"expected":row["recipe"],"actual":project(&recipe)}));
                    }
                }
                Err(error) => {
                    if row["success"] != false
                        || diagnostic(error) != row["diagnostic"].as_str().unwrap()
                    {
                        mismatches.push(json!({"id":row["id"],"error":diagnostic(error),"expected":row["diagnostic"]}));
                    }
                }
            }
        }
        if !mismatches.is_empty() {
            eprintln!(
                "OPTICS_NATIVE_MISMATCHES {}",
                serde_json::to_string(&mismatches).unwrap()
            );
        }
        assert!(
            mismatches.is_empty(),
            "{} frozen native optics mismatches",
            mismatches.len()
        );
    }
    #[test]
    fn completed_profile_defaults_are_copied_and_owned() {
        let captured = fixture("defaults");
        let rows = captured["completed_profile_defaults"].as_array().unwrap();
        assert_eq!(rows.len(), 12);
        for row in rows {
            let recipe = {
                let input = params(&row["input"]);
                Recipe::new(input).unwrap()
            };
            assert_eq!(project(&recipe), row["recipe"], "{}", row["key"]);
        }
    }
    #[test]
    fn real_resolved_hash_guard_is_call_bounded_and_thread_local() {
        let captured = fixture("literal");
        let rows = captured["cases"].as_array().unwrap();
        let active = params(
            &rows
                .iter()
                .find(|r| r["id"] == "optics-active-print")
                .unwrap()["input"],
        );
        let inactive =
            params(&rows.iter().find(|r| r["id"] == "optics-inactive").unwrap()["input"]);
        test_support::zero_next_resolved_hash();
        let error = Recipe::new(active).unwrap_err();
        assert_eq!(
            (error.component(), error.field(), error.channel()),
            (Component::CameraDiffusion, Field::ResolvedHash, None)
        );
        assert!(Recipe::new(active).is_ok());
        test_support::zero_next_resolved_hash();
        assert!(Recipe::new(inactive).is_ok());
        assert!(Recipe::new(active).is_ok());
        let mut invalid = inactive;
        invalid.camera_diffusion.family = 255;
        test_support::zero_next_resolved_hash();
        assert_eq!(Recipe::new(invalid).unwrap_err().field(), Field::Family);
        assert!(Recipe::new(active).is_ok());
        let thread = std::thread::spawn(move || {
            test_support::zero_next_resolved_hash();
            assert_eq!(
                Recipe::new(active).unwrap_err().field(),
                Field::ResolvedHash
            );
            assert!(Recipe::new(active).is_ok());
        });
        assert!(Recipe::new(active).is_ok());
        thread.join().unwrap();
    }
}
