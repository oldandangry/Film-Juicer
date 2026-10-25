//! Callback-local scatter/halation schedules, distinct from the cached GaussianKernel.

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
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalationGaussianKind {
    Identity,
    FirReflect,
    YvvReplicate,
}

#[derive(Clone, Copy, Debug)]
pub struct HalationGaussian {
    kind: HalationGaussianKind,
    radius: i32,
    fir_weights: [f32; 19],
    yvv: [f32; 4],
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
        let result = Self {
            kind: HalationGaussianKind::YvvReplicate,
            yvv: coefficients.map(|v| v as f32),
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
    tail: [HalationGaussian; 3],
    bounce: [HalationGaussian; 3],
    total_strength: f32,
}
impl HalationChannel {
    pub fn core(&self) -> &HalationGaussian {
        &self.core
    }
    pub fn tail(&self) -> &[HalationGaussian; 3] {
        &self.tail
    }
    pub fn bounce(&self) -> &[HalationGaussian; 3] {
        &self.bounce
    }
    pub fn total_strength(&self) -> f32 {
        self.total_strength
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HalationFrame {
    recipe_hash: u64,
    scatter_amount: f32,
    channels: [HalationChannel; 3],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalationFrameField {
    ScatterCore,
    ScatterTail,
    Bounce,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HalationFrameError {
    PixelSize,
    Gaussian {
        field: HalationFrameField,
        channel: usize,
        index: Option<usize>,
        sigma_px: f64,
    },
}
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
        const CORE_UM: [f64; 3] = [2.2, 2.0, 1.6];
        const TAIL_UM: [f64; 3] = [9.3, 9.7, 9.1];
        const SIGMA_RATIOS: [f64; 3] = [0.5360, 1.5236, 2.7684];
        #[expect(
            clippy::approx_constant,
            clippy::excessive_precision,
            reason = "preserve the native f64 bounce literal spellings and exact rounded values"
        )]
        const BOUNCE_MULTIPLIERS: [f64; 3] = [1.0, 1.4142135623730950488, 1.7320508075688772935];
        let mut result = Self {
            recipe_hash: input.recipe_hash,
            scatter_amount: if input.scatter_active {
                super::canonical_float(input.scatter_amount)
            } else {
                0.0
            },
            channels: [HalationChannel {
                core: HalationGaussian::IDENTITY,
                tail: [HalationGaussian::IDENTITY; 3],
                bounce: [HalationGaussian::IDENTITY; 3],
                total_strength: 0.0,
            }; 3],
        };
        let pixel = f64::from(pixel_size_um);
        for (channel, output) in result.channels.iter_mut().enumerate() {
            if input.scatter_active {
                let scale = f64::from(input.scatter_spatial_scale);
                let sigma = super::ordered_max(CORE_UM[channel] * scale / pixel, 1.0e-6);
                output.core =
                    HalationGaussian::new(sigma, HalationFrameField::ScatterCore, channel, None)?;
                for (index, tail) in output.tail.iter_mut().enumerate() {
                    let sigma = super::ordered_max(
                        TAIL_UM[channel] * SIGMA_RATIOS[index] * scale / pixel,
                        1.0e-6,
                    );
                    *tail = HalationGaussian::new(
                        sigma,
                        HalationFrameField::ScatterTail,
                        channel,
                        Some(index),
                    )?;
                }
            }
            output.total_strength = if input.back_reflection_active {
                super::canonical_float(input.total_strength[channel])
            } else {
                0.0
            };
            if input.back_reflection_active {
                let first_sigma = f64::from(input.first_sigma_um[channel]);
                let scale = f64::from(input.halation_spatial_scale);
                for (index, bounce) in output.bounce.iter_mut().enumerate() {
                    let sigma = super::ordered_max(
                        first_sigma * BOUNCE_MULTIPLIERS[index] * scale / pixel,
                        1.0e-6,
                    );
                    *bounce = HalationGaussian::new(
                        sigma,
                        HalationFrameField::Bounce,
                        channel,
                        Some(index),
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
        for sigma in [-1.0, -0.0, 0.0, 1.0e-6] {
            let g = HalationGaussian::new(sigma, HalationFrameField::ScatterCore, 0, None).unwrap();
            assert_eq!(g.kind(), HalationGaussianKind::Identity);
            assert_eq!(g.radius(), 0);
            assert_eq!(g.fir_weights().map(f32::to_bits), [0; 19]);
            assert_eq!(g.yvv().map(f32::to_bits), [0; 4]);
        }
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
        }
    }
    fn gaussian(v: &HalationGaussian) -> Value {
        let kind = match v.kind() {
            HalationGaussianKind::Identity => 0,
            HalationGaussianKind::FirReflect => 1,
            HalationGaussianKind::YvvReplicate => 2,
        };
        json!({"kind":kind,"radius":v.radius(),"weights":v.fir_weights().map(f32::to_bits),"yvv":v.yvv().map(f32::to_bits)})
    }
    fn project(v: &HalationFrame) -> Value {
        json!({"recipe_hash":v.recipe_hash(),"scatter_amount":v.scatter_amount().to_bits(),
            "channels":v.channels().map(|c| json!({"core":gaussian(c.core()),"tail":c.tail().map(|g|gaussian(&g)),
                "bounce":c.bounce().map(|g|gaussian(&g)),"strength":c.total_strength().to_bits()}))})
    }

    #[test]
    fn complete_native_capture() {
        let captured = capture();
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
        for row in &rows {
            let result = HalationFrame::new(
                input(&row["input"]),
                f32::from_bits(row["input"]["pixel_size_um"].as_u64().unwrap() as u32),
            );
            if let Err(error) = result {
                let expected = &checked["cases"][row["input"]["name"].as_str().unwrap()];
                match error {
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
        }
        let primitive_differences = captured["primitives"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| {
                f64::from_bits(p["argument"].as_u64().unwrap())
                    .exp()
                    .to_bits()
                    != p["exp"].as_u64().unwrap()
            })
            .count();
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
    }
}
