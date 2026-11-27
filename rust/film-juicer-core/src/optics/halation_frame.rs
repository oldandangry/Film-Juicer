//! Callback-local scatter/halation schedules, distinct from the cached GaussianKernel.

use super::{HALATION_MAX_BOUNCES, HALATION_TAIL_COMPONENTS};

#[derive(Clone, Copy, Debug)]
pub struct HalationFrameInput {
    pub scatter_active: bool,
    pub back_reflection_active: bool,
    pub scatter_amount: f32,
    pub scatter_spatial_scale: f32,
    pub halation_spatial_scale: f32,
    pub first_sigma_um: [f32; 3],
    pub total_strength: [f32; 3],
    pub recipe_hash: u64,
    pub scatter_core_um: [f64; 3],
    pub scatter_tail_um: [f64; 3],
    pub scatter_tail_weight: [f32; 3],
    pub bounce_weights: [f32; HALATION_MAX_BOUNCES],
    pub bounce_count: usize,
    pub renormalize: bool,
    pub legacy_scatter: bool,
    pub legacy_reflection: bool,
}

#[cfg(test)]
impl Default for HalationFrameInput {
    fn default() -> Self {
        Self {
            scatter_active: false,
            back_reflection_active: false,
            scatter_amount: 0.0,
            scatter_spatial_scale: 0.0,
            halation_spatial_scale: 0.0,
            first_sigma_um: [0.0; 3],
            total_strength: [0.0; 3],
            recipe_hash: 0,
            scatter_core_um: super::SCATTER_CORE_UM,
            scatter_tail_um: super::SCATTER_TAIL_UM,
            scatter_tail_weight: super::SCATTER_TAIL_WEIGHT.map(|v| v as f32),
            bounce_weights: super::LEGACY_BOUNCE_WEIGHTS,
            bounce_count: 3,
            renormalize: true,
            legacy_scatter: true,
            legacy_reflection: true,
        }
    }
}

impl From<&super::Halation> for HalationFrameInput {
    fn from(recipe: &super::Halation) -> Self {
        Self {
            scatter_active: recipe.scatter_active(),
            back_reflection_active: recipe.back_reflection_active(),
            scatter_amount: recipe.scatter_amount(),
            scatter_spatial_scale: recipe.scatter_spatial_scale(),
            halation_spatial_scale: recipe.halation_spatial_scale(),
            first_sigma_um: *recipe.first_sigma_um(),
            total_strength: *recipe.total_strength(),
            recipe_hash: recipe.hash(),
            scatter_core_um: recipe.scatter_core_um,
            scatter_tail_um: recipe.scatter_tail_um,
            scatter_tail_weight: recipe.scatter_tail_weight,
            bounce_weights: recipe.bounce_weights,
            bounce_count: recipe.bounce_count,
            renormalize: recipe.renormalize,
            legacy_scatter: recipe.legacy_scatter,
            legacy_reflection: recipe.legacy_reflection,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalationGaussianKind {
    Identity,
    FirReflect,
    YvvReplicate,
    YvvReplicateFactored,
}

#[derive(Clone, Copy, Debug)]
pub struct HalationGaussian {
    kind: HalationGaussianKind,
    radius: i32,
    fir_weights: [f32; 19],
    yvv: [f32; 4],
}

// Factor the normalized third-order recurrence into a real pole and a
// second-order section. The bisection and reconstruction checks match the
// measured precision witness; only the final execution factors are narrowed.
fn factored_coefficients([b, a1, a2, a3]: [f64; 4]) -> Option<[f32; 4]> {
    if ![b, a1, a2, a3].iter().all(|value| value.is_finite()) || b <= 0.0 || a3 <= 0.0 {
        return None;
    }
    let c1 = 3.0 - 2.0 * a1 - a2;
    let c2 = 3.0 - a1;
    let polynomial = |h: f64| ((-h + c2) * h - c1) * h + b;
    if !(polynomial(0.0) > 0.0 && polynomial(1.0) < 0.0) {
        return None;
    }
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..64 {
        let mid = (low + high) * 0.5;
        if polynomial(mid) > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    let h = (low + high) * 0.5;
    let real_pole = 1.0 - h;
    let t = a3 / real_pole;
    let g = b / h;
    let pair_sum = 1.0 + t - g;
    let reconstructed_a1 = real_pole + pair_sum;
    let reconstructed_a2 = -t - real_pole * pair_sum;
    let reconstructed_a3 = real_pole * t;
    const TOLERANCE: f64 = 2.0e-10;
    if !(h > 0.0
        && h < 1.0
        && t > 0.0
        && t < 1.0
        && g > 0.0
        && g < 2.0 * (1.0 + t)
        && g.is_finite())
        || (reconstructed_a1 - a1).abs() > TOLERANCE
        || (reconstructed_a2 - a2).abs() > TOLERANCE
        || (reconstructed_a3 - a3).abs() > TOLERANCE
    {
        return None;
    }
    let factors = [h as f32, g as f32, t as f32, 0.0];
    if !(factors[0] > 0.0
        && factors[0] < 1.0
        && factors[1] > 0.0
        && factors[1].is_finite()
        && factors[2] > 0.0
        && factors[2] < 1.0
        && f64::from(factors[1]) < 2.0 * (1.0 + f64::from(factors[2])))
    {
        return None;
    }
    Some(factors)
}

impl HalationGaussian {
    const IDENTITY: Self = Self {
        kind: HalationGaussianKind::Identity,
        radius: 0,
        fir_weights: [0.0; 19],
        yvv: [0.0; 4],
    };

    fn new(
        sigma_px: f64,
        field: HalationFrameField,
        channel: usize,
        index: Option<usize>,
        legacy: bool,
    ) -> Result<Self, HalationFrameError> {
        let failed = || HalationFrameError::Gaussian {
            field,
            channel,
            index,
            sigma_px,
        };
        if !sigma_px.is_finite() {
            return Err(failed());
        }
        if sigma_px <= 0.0 {
            return Ok(Self::IDENTITY);
        }
        if sigma_px < 3.0 {
            let radius = (3.0 * sigma_px + 0.5) as i32;
            if radius == 0 {
                return Ok(Self::IDENTITY);
            }
            let count = (2 * radius + 1) as usize;
            let mut weights = [0.0; 19];
            let mut sum = 0.0;
            for (i, weight) in weights.iter_mut().enumerate().take(count) {
                let x = f64::from(i as i32 - radius);
                let value = (-(x * x) / (2.0 * sigma_px * sigma_px)).exp();
                if !value.is_finite() {
                    return Err(failed());
                }
                *weight = value;
                sum += value;
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(failed());
            }
            let mut result = Self {
                kind: HalationGaussianKind::FirReflect,
                radius,
                ..Self::IDENTITY
            };
            for (i, weight) in weights.iter().enumerate().take(count) {
                let narrowed = (weight / sum) as f32;
                if !narrowed.is_finite() {
                    return Err(failed());
                }
                result.fir_weights[i] = super::canonical_float(narrowed);
            }
            return Ok(result);
        }

        const MAX_ADVANCED_SIGMA_PX: f64 = 1024.0;
        if !legacy && sigma_px > MAX_ADVANCED_SIGMA_PX {
            return Err(HalationFrameError::AdvancedRadius {
                field,
                channel,
                index,
                sigma_px,
            });
        }
        let q = 0.98711 * sigma_px - 0.96330;
        let q2 = q * q;
        let q3 = q2 * q;
        let b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        let b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
        let b2 = -(1.4281 * q2 + 1.26661 * q3);
        let b3 = 0.422205 * q3;
        let coefficients = [1.0 - (b1 + b2 + b3) / b0, b1 / b0, b2 / b0, b3 / b0];
        if !coefficients.iter().all(|v| v.is_finite()) {
            return Err(failed());
        }
        let (kind, yvv) = if legacy {
            (
                HalationGaussianKind::YvvReplicate,
                coefficients.map(|v| v as f32),
            )
        } else {
            (
                HalationGaussianKind::YvvReplicateFactored,
                factored_coefficients(coefficients).ok_or_else(failed)?,
            )
        };
        let result = Self {
            kind,
            yvv,
            ..Self::IDENTITY
        };
        if !result.yvv.iter().all(|v| v.is_finite()) {
            return Err(failed());
        }
        Ok(result)
    }

    pub fn kind(&self) -> HalationGaussianKind {
        self.kind
    }
    pub fn radius(&self) -> i32 {
        self.radius
    }
    pub fn fir_weights(&self) -> &[f32; 19] {
        &self.fir_weights
    }
    pub fn yvv(&self) -> &[f32; 4] {
        &self.yvv
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HalationChannel {
    core: HalationGaussian,
    tail: [HalationGaussian; HALATION_TAIL_COMPONENTS],
    bounce: [HalationGaussian; HALATION_MAX_BOUNCES],
    tail_weight: f32,
    total_strength: f32,
}
impl HalationChannel {
    pub fn core(&self) -> &HalationGaussian {
        &self.core
    }
    pub fn tail(&self) -> &[HalationGaussian; HALATION_TAIL_COMPONENTS] {
        &self.tail
    }
    pub fn bounce(&self) -> &[HalationGaussian; HALATION_MAX_BOUNCES] {
        &self.bounce
    }
    pub fn total_strength(&self) -> f32 {
        self.total_strength
    }
    pub fn tail_weight(&self) -> f32 {
        self.tail_weight
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HalationFrame {
    recipe_hash: u64,
    scatter_amount: f32,
    bounce_weights: [f32; HALATION_MAX_BOUNCES],
    bounce_count: usize,
    renormalize: bool,
    channels: [HalationChannel; 3],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalationFrameField {
    ScatterCore,
    ScatterTail,
    Bounce,
}
impl HalationFrameField {
    pub fn name(self) -> &'static str {
        match self {
            Self::ScatterCore => "scatter_core",
            Self::ScatterTail => "scatter_tail",
            Self::Bounce => "back_reflection_bounce",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HalationFrameError {
    PixelSize,
    BounceCount,
    Gaussian {
        field: HalationFrameField,
        channel: usize,
        index: Option<usize>,
        sigma_px: f64,
    },
    AdvancedRadius {
        field: HalationFrameField,
        channel: usize,
        index: Option<usize>,
        sigma_px: f64,
    },
}

impl std::fmt::Display for HalationFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PixelSize => f.write_str("halation requires a finite, positive pixel size"),
            Self::BounceCount => {
                f.write_str("halation bounce count is outside the supported range")
            }
            Self::Gaussian {
                field,
                channel,
                index,
                sigma_px,
            } => write!(
                f,
                "invalid halation Gaussian field={field:?} channel={channel} index={index:?} sigma_px={sigma_px}"
            ),
            Self::AdvancedRadius {
                field,
                channel,
                index,
                sigma_px,
            } => write!(
                f,
                "halation Gaussian radius exceeds the frame limit field={field:?} channel={channel} index={index:?} sigma_px={sigma_px}"
            ),
        }
    }
}

impl std::error::Error for HalationFrameError {}

impl HalationFrame {
    /// Resolved scalars do not authenticate recipe provenance or admit CUDA storage.
    pub fn new(
        input: HalationFrameInput,
        pixel_size_um: f32,
    ) -> Result<Option<Self>, HalationFrameError> {
        if input.recipe_hash == 0 {
            return Ok(None);
        }
        if !pixel_size_um.is_finite() || pixel_size_um <= 0.0 {
            return Err(HalationFrameError::PixelSize);
        }
        if input.back_reflection_active && !(1..=HALATION_MAX_BOUNCES).contains(&input.bounce_count)
        {
            return Err(HalationFrameError::BounceCount);
        }
        const SIGMA_RATIOS: [f64; HALATION_TAIL_COMPONENTS] = [0.5360, 1.5236, 2.7684];
        #[expect(
            clippy::approx_constant,
            clippy::excessive_precision,
            reason = "preserve the native f64 bounce literal spellings and exact rounded values"
        )]
        const BOUNCE_MULTIPLIERS: [f64; HALATION_MAX_BOUNCES] = [
            1.0,
            1.4142135623730950488,
            1.7320508075688772935,
            2.0,
            2.23606797749979,
        ];
        let mut result = Self {
            recipe_hash: input.recipe_hash,
            scatter_amount: if input.scatter_active {
                super::canonical_float(input.scatter_amount)
            } else {
                0.0
            },
            bounce_weights: if input.back_reflection_active {
                input.bounce_weights
            } else {
                [0.0; HALATION_MAX_BOUNCES]
            },
            bounce_count: if input.back_reflection_active {
                input.bounce_count
            } else {
                0
            },
            renormalize: input.back_reflection_active && input.renormalize,
            channels: [HalationChannel {
                core: HalationGaussian::IDENTITY,
                tail: [HalationGaussian::IDENTITY; HALATION_TAIL_COMPONENTS],
                bounce: [HalationGaussian::IDENTITY; HALATION_MAX_BOUNCES],
                tail_weight: 0.0,
                total_strength: 0.0,
            }; 3],
        };
        let pixel = f64::from(pixel_size_um);
        for (channel, output) in result.channels.iter_mut().enumerate() {
            if input.scatter_active {
                let scale = f64::from(input.scatter_spatial_scale);
                output.tail_weight = input.scatter_tail_weight[channel];
                if output.tail_weight != 1.0 {
                    let sigma =
                        super::ordered_max(input.scatter_core_um[channel] * scale / pixel, 1.0e-6);
                    output.core = HalationGaussian::new(
                        sigma,
                        HalationFrameField::ScatterCore,
                        channel,
                        None,
                        input.legacy_scatter,
                    )?;
                }
                for (index, tail) in output
                    .tail
                    .iter_mut()
                    .enumerate()
                    .filter(|_| output.tail_weight != 0.0)
                {
                    let sigma = super::ordered_max(
                        input.scatter_tail_um[channel] * SIGMA_RATIOS[index] * scale / pixel,
                        1.0e-6,
                    );
                    *tail = HalationGaussian::new(
                        sigma,
                        HalationFrameField::ScatterTail,
                        channel,
                        Some(index),
                        input.legacy_scatter,
                    )?;
                }
            }
            output.total_strength = if input.back_reflection_active {
                super::canonical_float(input.total_strength[channel])
            } else {
                0.0
            };
            if input.back_reflection_active
                && (input.legacy_reflection || output.total_strength > 0.0)
            {
                let first_sigma = f64::from(input.first_sigma_um[channel]);
                let scale = f64::from(input.halation_spatial_scale);
                for (index, bounce) in output.bounce[..result.bounce_count].iter_mut().enumerate() {
                    let sigma = super::ordered_max(
                        first_sigma * BOUNCE_MULTIPLIERS[index] * scale / pixel,
                        1.0e-6,
                    );
                    *bounce = HalationGaussian::new(
                        sigma,
                        HalationFrameField::Bounce,
                        channel,
                        Some(index),
                        input.legacy_reflection,
                    )?;
                }
            }
        }
        Ok(Some(result))
    }
    pub fn recipe_hash(&self) -> u64 {
        self.recipe_hash
    }
    pub fn scatter_amount(&self) -> f32 {
        self.scatter_amount
    }
    pub fn bounce_weights(&self) -> &[f32; HALATION_MAX_BOUNCES] {
        &self.bounce_weights
    }
    pub fn bounce_count(&self) -> usize {
        self.bounce_count
    }
    pub fn renormalize(&self) -> bool {
        self.renormalize
    }
    pub fn channels(&self) -> &[HalationChannel; 3] {
        &self.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn resolved_recipe_copy_does_not_reapply_amount_or_retain_a_borrow() {
        let input = {
            let recipe = super::super::Halation::new(super::super::HalationParams {
                active: true,
                scatter_amount: 0.75,
                scatter_spatial_scale: 1.25,
                halation_amount: 2.0,
                halation_spatial_scale: 0.5,
                first_sigma_um: [65.0, 32.0, 9.0],
                primary_amount: [0.3, 0.0, 0.015],
                shape: super::super::HalationShape::default(),
                legacy_reflection: true,
            })
            .unwrap();
            let input = HalationFrameInput::from(&recipe);
            assert_eq!(input.recipe_hash, recipe.hash());
            assert_eq!(
                input.total_strength.map(f32::to_bits),
                recipe.total_strength().map(f32::to_bits)
            );
            assert_eq!(input.first_sigma_um, *recipe.first_sigma_um());
            input
        };
        let frame = HalationFrame::new(input, 10.0).unwrap().unwrap();
        assert_eq!(frame.recipe_hash(), input.recipe_hash);
        assert_eq!(
            frame.scatter_amount().to_bits(),
            input.scatter_amount.to_bits()
        );
        for (channel, expected) in frame.channels().iter().zip(input.total_strength) {
            assert_eq!(channel.total_strength().to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn gaussian_identity_has_only_positive_zero_fields() {
        let sigmas = [
            ("negative", -1.0),
            ("negative-zero", -0.0),
            ("positive-zero", 0.0),
            ("small-positive", 1.0e-6),
        ];
        let cases_ids: Vec<_> = IntoIterator::into_iter(sigmas.iter().map(|(id, _)| *id))
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "halation-gaussian-identity"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "halation-gaussian-identity"
        );
        for (id, sigma) in sigmas {
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            let g = HalationGaussian::new(sigma, HalationFrameField::ScatterCore, 0, None, true)
                .unwrap();
            assert_eq!(g.kind(), HalationGaussianKind::Identity);
            assert_eq!(g.radius(), 0);
            assert_eq!(g.fir_weights().map(f32::to_bits), [0; 19]);
            assert_eq!(g.yvv().map(f32::to_bits), [0; 4]);
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

    #[test]
    fn advanced_factors_reconstruct_the_original_recurrence() {
        for sigma in [3.0, 8.0, 64.0, 1024.0] {
            let gaussian =
                HalationGaussian::new(sigma, HalationFrameField::ScatterCore, 0, None, false)
                    .unwrap();
            assert_eq!(gaussian.kind(), HalationGaussianKind::YvvReplicateFactored);
            let [h, g, t, reserved] = gaussian.yvv().map(f64::from);
            assert!(h > 0.0 && h < 1.0 && g > 0.0 && t > 0.0 && t < 1.0);
            assert!(g < 2.0 * (1.0 + t));
            assert_eq!(reserved.to_bits(), 0);

            let q = 0.98711 * sigma - 0.96330;
            let q2 = q * q;
            let q3 = q2 * q;
            let b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
            let expected = [
                (2.44413 * q + 2.85619 * q2 + 1.26661 * q3) / b0,
                -(1.4281 * q2 + 1.26661 * q3) / b0,
                0.422205 * q3 / b0,
            ];
            let pole = 1.0 - h;
            let pair_sum = 1.0 + t - g;
            let actual = [pole + pair_sum, -t - pole * pair_sum, pole * t];
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 2.0e-7, "sigma={sigma}");
            }
        }
        for (sigma, expected) in [
            (3.0, [0x3ebcc461, 0x3e8d20e0, 0x3eafdc3e, 0]),
            (8.0, [0x3e138802, 0x3d44ee67, 0x3f38af36, 0]),
        ] {
            let gaussian =
                HalationGaussian::new(sigma, HalationFrameField::Bounce, 2, Some(1), false)
                    .unwrap();
            assert_eq!(gaussian.yvv().map(f32::to_bits), expected);
        }
    }

    #[test]
    fn legacy_yvv_bits_and_advanced_boundaries_are_preserved() {
        let legacy =
            HalationGaussian::new(3.0, HalationFrameField::Bounce, 1, Some(2), true).unwrap();
        assert_eq!(legacy.kind(), HalationGaussianKind::YvvReplicate);
        assert_eq!(
            legacy.yvv().map(f32::to_bits),
            [0x3dd020cf, 0x3fd97dbf, 0xbf824147, 0x3e5e0bd7]
        );
        let below = f64::from_bits(3.0f64.to_bits() - 1);
        assert_eq!(
            HalationGaussian::new(below, HalationFrameField::ScatterTail, 1, Some(2), false)
                .unwrap()
                .kind(),
            HalationGaussianKind::FirReflect
        );
        let above = f64::from_bits(1024.0f64.to_bits() + 1);
        assert_eq!(
            HalationGaussian::new(above, HalationFrameField::ScatterTail, 1, Some(2), false)
                .unwrap_err(),
            HalationFrameError::AdvancedRadius {
                field: HalationFrameField::ScatterTail,
                channel: 1,
                index: Some(2),
                sigma_px: above,
            }
        );
        assert!(matches!(
            HalationGaussian::new(f64::INFINITY, HalationFrameField::Bounce, 1, Some(2), false),
            Err(HalationFrameError::Gaussian {
                field: HalationFrameField::Bounce,
                channel: 1,
                index: Some(2),
                sigma_px,
            }) if sigma_px.is_infinite()
        ));
        assert!(factored_coefficients([0.0, 0.0, 0.0, 1.0]).is_none());
        assert!(factored_coefficients([1.0, 1.0, 1.0, f64::NAN]).is_none());
        assert!(factored_coefficients([1.0, 0.0, 0.0, 1.0]).is_none());
    }

    fn capture() -> Value {
        if let Ok(path) = std::env::var("FJ_HALATION_NATIVE_CAPTURE") {
            return serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/scatter_halation/frame-fixtures");
        let read = |name: &str| -> Value {
            serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap()
        };
        let manifest = read("manifest.json");
        let preset = match (cfg!(windows), cfg!(debug_assertions)) {
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
        };
        let logical = manifest["presets"][preset]["native"].as_str().unwrap();
        assert_eq!(
            read(manifest["contexts"][logical].as_str().unwrap())["preset"],
            preset
        );
        read(manifest["payloads"][logical].as_str().unwrap())
    }
    fn input(v: &Value) -> HalationFrameInput {
        let f = |key: &str, i: usize| f32::from_bits(v[key][i].as_u64().unwrap() as u32);
        HalationFrameInput {
            scatter_active: v["scatter_active"].as_bool().unwrap(),
            back_reflection_active: v["back_reflection_active"].as_bool().unwrap(),
            scatter_amount: f("scalars", 0),
            scatter_spatial_scale: f("scalars", 1),
            halation_spatial_scale: f("scalars", 2),
            first_sigma_um: std::array::from_fn(|i| f("first_sigma_um", i)),
            total_strength: std::array::from_fn(|i| f("total_strength", i)),
            recipe_hash: v["recipe_hash"].as_u64().unwrap(),
            ..HalationFrameInput::default()
        }
    }
    fn gaussian(v: &HalationGaussian) -> Value {
        let kind = match v.kind() {
            HalationGaussianKind::Identity => 0,
            HalationGaussianKind::FirReflect => 1,
            HalationGaussianKind::YvvReplicate => 2,
            HalationGaussianKind::YvvReplicateFactored => {
                panic!("legacy fixture emitted factored YVV")
            }
        };
        json!({"kind":kind,"radius":v.radius(),"weights":v.fir_weights().map(f32::to_bits),"yvv":v.yvv().map(f32::to_bits)})
    }
    fn project(v: &HalationFrame) -> Value {
        json!({"recipe_hash":v.recipe_hash(),"scatter_amount":v.scatter_amount().to_bits(),
            "channels":v.channels().map(|c| json!({"core":gaussian(c.core()),"tail":c.tail().map(|g|gaussian(&g)),
                "bounce":c.bounce()[..3].iter().map(gaussian).collect::<Vec<_>>(),"strength":c.total_strength().to_bits()}))})
    }

    #[test]
    fn complete_native_capture() {
        let captured = capture();
        let case_count = captured["cases"].as_array().unwrap().len();
        let mut rows = captured["cases"].as_array().unwrap().clone();
        assert_eq!(rows.len(), 52);
        let products = captured["products"].as_array().unwrap();
        assert_eq!(products.len(), 20);
        rows.extend(products.iter().cloned());
        let checked: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/scatter_halation/frame-fixtures/checked.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut mismatches = Vec::new();
        let expected_ids = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let group = if index < case_count {
                    "case"
                } else {
                    "product"
                };
                format!("{group}/{}", row["input"]["name"].as_str().unwrap())
            })
            .collect::<Vec<_>>();
        let cases_ids: Vec<_> = IntoIterator::into_iter(expected_ids)
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "halation-frame"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "halation-frame"
        );
        for (index, row) in rows.iter().enumerate() {
            let group = if index < case_count {
                "case"
            } else {
                "product"
            };
            let row_case = AsRef::<str>::as_ref(
                &(format!("{group}/{}", row["input"]["name"].as_str().unwrap())),
            )
            .to_owned();
            let mismatch_count = mismatches.len();
            let result = HalationFrame::new(
                input(&row["input"]),
                f32::from_bits(row["input"]["pixel_size_um"].as_u64().unwrap() as u32),
            );
            if let Err(error) = result {
                let expected = &checked["cases"][row["input"]["name"].as_str().unwrap()];
                match error {
                    HalationFrameError::BounceCount | HalationFrameError::AdvancedRadius { .. } => {
                        panic!("legacy frame returned an advanced failure: {error:?}")
                    }
                    HalationFrameError::PixelSize => assert_eq!(expected["reason"], 2),
                    HalationFrameError::Gaussian {
                        field,
                        channel,
                        index,
                        sigma_px,
                    } => {
                        assert_eq!(expected["reason"], 3);
                        let tag = match field {
                            HalationFrameField::ScatterCore => 2,
                            HalationFrameField::ScatterTail => 3,
                            HalationFrameField::Bounce => 4,
                        };
                        assert_eq!(expected["field"], tag);
                        assert_eq!(expected["channel"], channel);
                        assert_eq!(expected["index"], index.unwrap_or(3));
                        if expected["sigma_nan"].as_bool().unwrap() {
                            assert!(sigma_px.is_nan());
                        } else {
                            assert_eq!(expected["sigma_bits"], sigma_px.to_bits());
                        }
                    }
                }
            }
            let success = result.is_ok();
            let descriptor = result.ok().flatten();
            let actual = descriptor.as_ref().map_or(Value::Null, project);
            if row["success"] != success
                || row["present"] != descriptor.is_some()
                || row["descriptor"] != actual
            {
                mismatches.push(json!({"case":row["input"]["name"],"actual":actual,"expected":row["descriptor"]}));
            }
            if mismatches.len() == mismatch_count {
                assert!(
                    cases_remaining.remove(&row_case),
                    "unexpected or repeated comparison: {row_case}"
                );
            }
        }
        let primitives = captured["primitives"].as_array().unwrap();
        let primitive_cases_ids: Vec<_> = IntoIterator::into_iter(["exp-sweep"])
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !primitive_cases_ids.is_empty(),
            "empty case set: {}",
            "halation-primitives"
        );
        let primitive_cases_count = primitive_cases_ids.len();
        let mut primitive_cases_remaining = primitive_cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            primitive_cases_remaining.len(),
            primitive_cases_count,
            "duplicate case IDs: {}",
            "halation-primitives"
        );
        let primitive_case = AsRef::<str>::as_ref(&("exp-sweep")).to_owned();
        let mut primitive_count = 0;
        let primitive_differences = primitives
            .iter()
            .filter(|p| {
                primitive_count += 1;
                f64::from_bits(p["argument"].as_u64().unwrap())
                    .exp()
                    .to_bits()
                    != p["exp"].as_u64().unwrap()
            })
            .count();
        assert_eq!(primitive_count, primitives.len());
        assert!(
            primitive_cases_remaining.remove(&primitive_case),
            "unexpected or repeated comparison: {primitive_case}"
        );
        println!(
            "Halation: {} complete records; {primitive_differences} f64 exp differences; {} full-record mismatches",
            rows.len(),
            mismatches.len()
        );
        for mismatch in &mismatches {
            println!("{mismatch}");
        }
        assert!(
            mismatches.is_empty(),
            "exact whole halation schedule gate failed"
        );
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
        assert!(
            primitive_cases_remaining.is_empty(),
            "missing comparisons: {primitive_cases_remaining:?}"
        );
    }
}
