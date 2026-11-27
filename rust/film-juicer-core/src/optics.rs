//! Completed spatial-optics recipes from captured controls and copied profile defaults.

use crate::hash;

pub mod diffusion_frame;
pub mod highlight_boost;

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
    pub shape: HalationShape,
    pub legacy_reflection: bool,
}

pub const HALATION_TAIL_COMPONENTS: usize = 3;
pub const HALATION_MAX_BOUNCES: usize = 5;
const SCATTER_CORE_UM: [f64; 3] = [2.2, 2.0, 1.6];
const SCATTER_TAIL_UM: [f64; 3] = [9.3, 9.7, 9.1];
const SCATTER_TAIL_WEIGHT: [f64; 3] = [0.78, 0.65, 0.67];
const LEGACY_BOUNCE_WEIGHTS: [f32; HALATION_MAX_BOUNCES] =
    [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 0.0, 0.0];
pub const HALATION_MAX_AUTHORED: f64 = 1.0e6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalationShape {
    pub scatter_core_um: [f64; 3],
    pub scatter_tail_um: [f64; 3],
    pub scatter_tail_weight: [f64; 3],
    pub bounce_count: i32,
    pub bounce_decay: f64,
    pub renormalize: bool,
}
impl Default for HalationShape {
    fn default() -> Self {
        Self {
            scatter_core_um: SCATTER_CORE_UM,
            scatter_tail_um: SCATTER_TAIL_UM,
            scatter_tail_weight: SCATTER_TAIL_WEIGHT,
            bounce_count: 3,
            bounce_decay: 0.5,
            renormalize: true,
        }
    }
}
impl HalationShape {
    pub(crate) fn canonicalize_defaults(mut self) -> Self {
        // Resolve can return Double3D defaults after an FP32 round trip. Restore
        // only those exact aliases before identity and legacy schedule selection;
        // nearby authored values retain their original precision.
        for (values, defaults) in [
            (&mut self.scatter_core_um, SCATTER_CORE_UM),
            (&mut self.scatter_tail_um, SCATTER_TAIL_UM),
            (&mut self.scatter_tail_weight, SCATTER_TAIL_WEIGHT),
        ] {
            for (value, default) in values.iter_mut().zip(defaults) {
                if *value == f64::from(default as f32) {
                    *value = default;
                }
            }
        }
        self
    }

    pub fn validate(self) -> Result<Self, Error> {
        if !(1..=HALATION_MAX_BOUNCES as i32).contains(&self.bounce_count) {
            return Err(Error::new(Component::ScatterHalation, Field::BounceCount));
        }
        if !(0.0..=1.0).contains(&self.bounce_decay) {
            return Err(Error::new(Component::ScatterHalation, Field::BounceDecay));
        }
        Ok(Self {
            scatter_core_um: halation_triplet(
                self.scatter_core_um,
                HALATION_MAX_AUTHORED,
                Field::ScatterCore,
            )?,
            scatter_tail_um: halation_triplet(
                self.scatter_tail_um,
                HALATION_MAX_AUTHORED,
                Field::ScatterTail,
            )?,
            scatter_tail_weight: halation_triplet(
                self.scatter_tail_weight,
                1.0,
                Field::ScatterTailWeight,
            )?,
            bounce_decay: canonical_double(self.bounce_decay),
            ..self
        }
        .canonicalize_defaults())
    }

    fn weights(self) -> [f32; HALATION_MAX_BOUNCES] {
        let mut powers = [0.0; HALATION_MAX_BOUNCES];
        let mut power = 1.0;
        let mut sum = 0.0;
        for weight in &mut powers[..self.bounce_count as usize] {
            *weight = power;
            sum += power;
            power *= self.bounce_decay;
        }
        powers.map(|weight| (weight / sum) as f32)
    }
}

pub(crate) fn halation_triplet(
    values: [f64; 3],
    maximum: f64,
    field: Field,
) -> Result<[f64; 3], Error> {
    for (channel, value) in values.into_iter().enumerate() {
        if !(0.0..=maximum).contains(&value) {
            return Err(Error {
                component: Component::ScatterHalation,
                field,
                channel: Some(channel),
            });
        }
    }
    Ok(values.map(canonical_double))
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub camera_diffusion: DiffusionParams,
    pub camera_lens_blur_um: f32,
    pub scatter_halation: HalationParams,
    pub highlight_boost: Option<highlight_boost::Recipe>,
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
    ScatterCore,
    ScatterTail,
    ScatterTailWeight,
    FirstBounceSigma,
    ReflectionStrength,
    BounceCount,
    BounceDecay,
}
impl Field {
    pub fn name(self) -> &'static str {
        match self {
            Self::Family => "family",
            Self::Strength => "strength",
            Self::SpatialScale => "spatial_scale",
            Self::HaloWarmth => "halo_warmth",
            Self::CoreIntensity => "core_intensity",
            Self::CoreSize => "core_size",
            Self::HaloIntensity => "halo_intensity",
            Self::HaloSize => "halo_size",
            Self::BloomIntensity => "bloom_intensity",
            Self::BloomSize => "bloom_size",
            Self::ScatterFraction => "scatter_fraction",
            Self::GroupWeights => "group_weights",
            Self::GroupCenters => "group_centers_um",
            Self::ResolvedHash => "resolved_hash",
            Self::SigmaUm => "sigma_um",
            Self::TotalStrength => "total_strength",
            Self::ScatterCore => "scatter_core_um",
            Self::ScatterTail => "scatter_tail_um",
            Self::ScatterTailWeight => "scatter_tail_weight",
            Self::FirstBounceSigma => "first_sigma_um",
            Self::ReflectionStrength => "reflection_strength",
            Self::BounceCount => "bounce_count",
            Self::BounceDecay => "bounce_decay",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error {
    component: Component,
    field: Field,
    channel: Option<usize>,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let field = self.field().name();
        if self.component() == Component::ScatterHalation {
            return write!(
                f,
                "ResourceDescriptorMismatch component=scatter_halation field={field} channel={}",
                self.channel().unwrap_or(3)
            );
        }
        let component = match self.component() {
            Component::CameraDiffusion => "camera_diffusion",
            Component::EnlargerDiffusion => "enlarger_diffusion",
            Component::CameraLensBlur => "camera_lens_blur",
            Component::ScatterHalation => unreachable!(),
        };
        write!(
            f,
            "ResourceDescriptorMismatch phase=6A component={component} field={field}"
        )
    }
}

impl std::error::Error for Error {}

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
    scatter_core_um: [f64; 3],
    scatter_tail_um: [f64; 3],
    scatter_tail_weight: [f32; 3],
    bounce_weights: [f32; HALATION_MAX_BOUNCES],
    bounce_count: usize,
    renormalize: bool,
    legacy_scatter: bool,
    legacy_reflection: bool,
    hash: u64,
}
#[derive(Debug)]
pub struct Recipe {
    camera_diffusion: Diffusion,
    camera_lens_blur: LensBlur,
    scatter_halation: Halation,
    highlight_boost: Option<highlight_boost::Recipe>,
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
fn canonical_double(value: f64) -> f64 {
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
    fn new(params: DiffusionParams, domain: Domain, component: Component) -> Result<Self, Error> {
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
            scatter_core_um: [0.0; 3],
            scatter_tail_um: [0.0; 3],
            scatter_tail_weight: [0.0; 3],
            bounce_weights: [0.0; HALATION_MAX_BOUNCES],
            bounce_count: 0,
            renormalize: false,
            legacy_scatter: false,
            legacy_reflection: false,
            hash: 0,
        };
        if !params.active {
            return Ok(result);
        }
        let shape = params.shape.validate()?;
        result.scatter_active = params.scatter_amount > 0.0
            && params.scatter_spatial_scale > 0.0
            && shape
                .scatter_core_um
                .into_iter()
                .chain(shape.scatter_tail_um)
                .any(|v| v > 0.0);
        if result.scatter_active {
            result.scatter_amount = canonical_float(params.scatter_amount);
            result.scatter_spatial_scale = canonical_float(params.scatter_spatial_scale);
            result.scatter_core_um = shape.scatter_core_um;
            result.scatter_tail_um = shape.scatter_tail_um;
            result.scatter_tail_weight = shape.scatter_tail_weight.map(|value| value as f32);
            result.legacy_scatter = result.scatter_core_um == SCATTER_CORE_UM
                && result.scatter_tail_um == SCATTER_TAIL_UM
                && result.scatter_tail_weight == SCATTER_TAIL_WEIGHT.map(|value| value as f32);
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
            } else {
                result.bounce_weights = shape.weights();
                result.bounce_count = result
                    .bounce_weights
                    .iter()
                    .rposition(|&weight| weight > 0.0)
                    .map_or(0, |index| index + 1);
                result.renormalize = shape.renormalize;
                result.legacy_reflection = params.legacy_reflection
                    && result.renormalize
                    && result.bounce_weights == LEGACY_BOUNCE_WEIGHTS;
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
        if result.scatter_active && !result.legacy_scatter {
            hash::update_bytes(&mut hash, b"halation-scatter\0");
            for value in result
                .scatter_core_um
                .into_iter()
                .chain(result.scatter_tail_um)
            {
                hash_double(&mut hash, value);
            }
            for weight in result.scatter_tail_weight {
                hash::update_bytes(&mut hash, &weight.to_le_bytes());
            }
        }
        if result.back_reflection_active && !result.legacy_reflection {
            hash::update_bytes(&mut hash, b"halation-reflection\0");
            hash::update_bytes(
                &mut hash,
                &[result.bounce_count as u8, u8::from(result.renormalize)],
            );
            for weight in result.bounce_weights {
                hash::update_bytes(&mut hash, &weight.to_le_bytes());
            }
        }
        result.hash = if hash == 0 { 1 } else { hash };
        Ok(result)
    }
}
impl Recipe {
    pub fn new(params: Params) -> Result<Self, Error> {
        let camera_diffusion = Diffusion::new(
            params.camera_diffusion,
            Domain::FilmLinearExposure,
            Component::CameraDiffusion,
        )?;
        let camera_lens_blur = LensBlur::new(params.camera_lens_blur_um)?;
        let scatter_halation = Halation::new(params.scatter_halation)?;
        let enlarger_diffusion = if params.print_domain {
            Diffusion::new(
                params.enlarger_diffusion,
                Domain::PrintLinearExposure,
                Component::EnlargerDiffusion,
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
        if let Some(boost) = params.highlight_boost {
            hash::update_bytes(&mut hash, b"highlight-boost\0");
            hash::update_bytes(&mut hash, &boost.hash().to_le_bytes());
            contributed = true;
        }
        Ok(Self {
            camera_diffusion,
            camera_lens_blur,
            scatter_halation,
            highlight_boost: params.highlight_boost,
            enlarger_diffusion,
            hash: if contributed { hash } else { 0 },
        })
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
    pub fn highlight_boost(&self) -> Option<highlight_boost::Recipe> {
        self.highlight_boost
    }
    pub fn enlarger_diffusion(&self) -> &Diffusion {
        &self.enlarger_diffusion
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

/// Identity of host-authored diffusion controls, distinct from the resolved recipe.
pub fn hash_diffusion_authored_controls(p: &DiffusionParams) -> u64 {
    let mut h = hash::FNV_OFFSET;
    hash::update_bytes(&mut h, b"diffusion-authored\0");
    hash::update_bytes(&mut h, &1_u32.to_le_bytes());
    hash::update_bytes(&mut h, &[u8::from(p.active), p.family as u8]);
    for value in [
        p.strength,
        p.spatial_scale,
        p.halo_warmth,
        p.core_intensity,
        p.core_size,
        p.halo_intensity,
        p.halo_size,
        p.bloom_intensity,
        p.bloom_size,
    ] {
        let bits = if value == 0.0 { 0 } else { value.to_bits() };
        hash::update_bytes(&mut h, &bits.to_le_bytes());
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::path::PathBuf;

    fn active_halation(shape: HalationShape) -> Halation {
        Halation::new(HalationParams {
            active: true,
            scatter_amount: 1.0,
            scatter_spatial_scale: 1.0,
            halation_amount: 1.0,
            halation_spatial_scale: 1.0,
            first_sigma_um: [65.0; 3],
            primary_amount: [0.015, 0.005, 0.0],
            shape,
            legacy_reflection: true,
        })
        .unwrap()
    }

    #[test]
    fn host_rounded_halation_defaults_preserve_identity_and_schedule() {
        let defaults = HalationShape::default();
        let expected = active_halation(defaults);
        let mut params = crate::capture::defaults();
        params.scatter_halation = crate::capture::halation(true, [1.0; 4]).unwrap();
        let identity = crate::capture::hash_params(&params, b"kodak_portra_400", b"print");
        // Every mixture of literal and host-rounded default channels must agree.
        for mask in 0..512 {
            let mut shape = defaults;
            for (index, value) in shape
                .scatter_core_um
                .iter_mut()
                .chain(&mut shape.scatter_tail_um)
                .chain(&mut shape.scatter_tail_weight)
                .enumerate()
            {
                if mask & (1 << index) != 0 {
                    *value = f64::from(*value as f32);
                }
            }
            assert_eq!(shape.validate().unwrap(), defaults);
            // Direct capture callers may hash before validation.
            params.scatter_halation.shape = shape;
            assert_eq!(
                crate::capture::hash_params(&params, b"kodak_portra_400", b"print"),
                identity
            );
            let actual = active_halation(shape);
            assert!(actual.legacy_scatter && actual.legacy_reflection);
            assert_eq!(actual.hash, expected.hash);
            // Include the captured clip geometry and separate FIR/YVV regimes.
            for pixel_size_um in [35000.0_f32 / 6048.0, 100.0, 0.5] {
                let frame = HalationFrame::new((&actual).into(), pixel_size_um)
                    .unwrap()
                    .unwrap();
                let expected_frame = HalationFrame::new((&expected).into(), pixel_size_um)
                    .unwrap()
                    .unwrap();
                assert_eq!(frame.recipe_hash(), expected_frame.recipe_hash());
                assert_eq!(frame.scatter_amount(), expected_frame.scatter_amount());
                assert_eq!(frame.bounce_count(), expected_frame.bounce_count());
                assert_eq!(frame.bounce_weights(), expected_frame.bounce_weights());
                assert_eq!(frame.renormalize(), expected_frame.renormalize());
                for (channel, expected_channel) in
                    frame.channels().iter().zip(expected_frame.channels())
                {
                    assert_eq!(channel.tail_weight(), expected_channel.tail_weight());
                    assert_eq!(channel.total_strength(), expected_channel.total_strength());
                    let gaussians = std::iter::once(channel.core())
                        .chain(channel.tail())
                        .chain(channel.bounce());
                    let expected_gaussians = std::iter::once(expected_channel.core())
                        .chain(expected_channel.tail())
                        .chain(expected_channel.bounce());
                    for (gaussian, expected_gaussian) in gaussians.zip(expected_gaussians) {
                        assert_eq!(gaussian.kind(), expected_gaussian.kind());
                        assert_eq!(gaussian.radius(), expected_gaussian.radius());
                        assert_eq!(
                            gaussian.fir_weights().map(f32::to_bits),
                            expected_gaussian.fir_weights().map(f32::to_bits)
                        );
                        assert_eq!(
                            gaussian.yvv().map(f32::to_bits),
                            expected_gaussian.yvv().map(f32::to_bits)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn halation_default_recognition_preserves_adjacent_custom_values() {
        let defaults = HalationShape::default();
        let mut params = crate::capture::defaults();
        params.scatter_halation = crate::capture::halation(true, [1.0; 4]).unwrap();
        let identity = crate::capture::hash_params(&params, b"film", b"print");
        for (field, values) in [
            defaults.scatter_core_um,
            defaults.scatter_tail_um,
            defaults.scatter_tail_weight,
        ]
        .into_iter()
        .enumerate()
        {
            for (channel, default) in values.into_iter().enumerate() {
                for alias in [default, f64::from(default as f32)] {
                    for custom in [alias.next_down(), alias.next_up()] {
                        let mut shape = HalationShape {
                            scatter_core_um: defaults.scatter_core_um.map(|v| f64::from(v as f32)),
                            scatter_tail_um: defaults.scatter_tail_um.map(|v| f64::from(v as f32)),
                            scatter_tail_weight: defaults
                                .scatter_tail_weight
                                .map(|v| f64::from(v as f32)),
                            ..defaults
                        };
                        let mut expected = defaults;
                        for value in [&mut shape, &mut expected] {
                            let triplet = match field {
                                0 => &mut value.scatter_core_um,
                                1 => &mut value.scatter_tail_um,
                                _ => &mut value.scatter_tail_weight,
                            };
                            triplet[channel] = custom;
                        }
                        assert_eq!(shape.validate().unwrap(), expected);
                        params.scatter_halation.shape = shape;
                        let custom_identity =
                            crate::capture::hash_params(&params, b"film", b"print");
                        assert_ne!(custom_identity, identity);
                        params.scatter_halation.shape = expected;
                        assert_eq!(
                            crate::capture::hash_params(&params, b"film", b"print"),
                            custom_identity
                        );
                        if field < 2 {
                            let resolved = active_halation(shape);
                            assert!(!resolved.legacy_scatter);
                            assert_eq!(resolved.scatter_core_um, expected.scatter_core_um);
                            assert_eq!(resolved.scatter_tail_um, expected.scatter_tail_um);
                        }
                    }
                }
            }
        }
    }

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
                shape: HalationShape::default(),
                legacy_reflection: true,
            },
            highlight_boost: None,
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
        let field = error.field().name();
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
        let cases_ids: Vec<_> =
            IntoIterator::into_iter(rows.iter().map(|row| row["id"].as_str().unwrap()))
                .map(|id| AsRef::<str>::as_ref(&id).to_owned())
                .collect();
        assert!(
            !cases_ids.is_empty(),
            "optics-frozen-native-words-identities-failures"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "optics-frozen-native-words-identities-failures"
        );
        for row in rows {
            let id = row["id"].as_str().unwrap();
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            match Recipe::new(params(&row["input"])) {
                Ok(recipe) => {
                    let actual = project(&recipe);
                    if row["success"] != true || actual != row["recipe"] {
                        mismatches
                            .push(json!({"id":row["id"],"expected":row["recipe"],"actual":actual}));
                    } else {
                        assert!(
                            cases_remaining.remove(&case),
                            "unexpected or repeated comparison: {case}"
                        );
                    }
                }
                Err(error) => {
                    let actual = diagnostic(error);
                    if row["success"] != false || actual != row["diagnostic"].as_str().unwrap() {
                        mismatches.push(
                            json!({"id":row["id"],"error":actual,"expected":row["diagnostic"]}),
                        );
                    } else {
                        assert!(
                            cases_remaining.remove(&case),
                            "unexpected or repeated comparison: {case}"
                        );
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
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
    }
    #[test]
    fn completed_profile_defaults_are_copied_and_owned() {
        let captured = fixture("defaults");
        let rows = captured["completed_profile_defaults"].as_array().unwrap();
        assert_eq!(rows.len(), 12);
        let cases_ids: Vec<_> = IntoIterator::into_iter(rows.iter().map(|row| {
            format!(
                "{}/print={}",
                row["key"].as_str().unwrap(),
                row["input"]["print"].as_bool().unwrap()
            )
        }))
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
        assert!(!cases_ids.is_empty(), "optics-completed-profile-defaults");
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "optics-completed-profile-defaults"
        );
        for row in rows {
            let id = format!(
                "{}/print={}",
                row["key"].as_str().unwrap(),
                row["input"]["print"].as_bool().unwrap()
            );
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            let recipe = {
                let input = params(&row["input"]);
                Recipe::new(input).unwrap()
            };
            assert_eq!(project(&recipe), row["recipe"], "{}", row["key"]);
            assert!(
                cases_remaining.remove(&case),
                "unexpected or repeated comparison: {case}"
            );
        }
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
    }
}
