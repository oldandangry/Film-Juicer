//! Invocation-local scalar scanner optics policy. No input borrow is retained.

use crate::{hash, route::ScanRoute};
use std::fmt;

const SCHEMA: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct PostEffectsParams {
    pub route: ScanRoute,
    pub glare_active: bool,
    pub glare_percent: f32,
    pub glare_roughness: f32,
    pub glare_blur_sigma_px: f32,
    pub lens_blur_sigma_px: f32,
    pub unsharp_sigma_px: f32,
    pub unsharp_amount: f32,
}

/// Read-only attempted fields, also retained on ordinary native value/hash failure.
/// This record alone is never a completed or cacheable post-effects value.
#[derive(Clone, Copy, Debug)]
pub struct PostEffectsAttempt {
    route: ScanRoute,
    glare_active: bool,
    values: [f32; 6],
    radii: [i32; 3],
}
impl PostEffectsAttempt {
    pub fn route(&self) -> ScanRoute {
        self.route
    }
    pub fn glare_active(&self) -> bool {
        self.glare_active
    }
    /// Glare percent, roughness, glare sigma, lens sigma, unsharp sigma, amount.
    pub fn values(&self) -> &[f32; 6] {
        &self.values
    }
    /// Glare, lens and unsharp radii, respectively.
    pub fn radii(&self) -> &[i32; 3] {
        &self.radii
    }
    pub fn schema(&self) -> u32 {
        SCHEMA
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PostEffects {
    fields: PostEffectsAttempt,
    hash: u64,
}
impl PostEffects {
    pub fn fields(&self) -> &PostEffectsAttempt {
        &self.fields
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn active(&self) -> bool {
        self.hash != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PostEffectsErrorKind {
    Values,
    RadiusRange,
    ZeroHash,
}

#[derive(Clone, Copy, Debug)]
pub struct PostEffectsError {
    kind: PostEffectsErrorKind,
    lane: u32,
    rejected: Option<PostEffectsAttempt>,
}
impl PostEffectsError {
    pub fn kind(&self) -> PostEffectsErrorKind {
        self.kind
    }
    pub fn lane(&self) -> u32 {
        self.lane
    }
    pub fn rejected_attempt(&self) -> Option<&PostEffectsAttempt> {
        self.rejected.as_ref()
    }
}
impl fmt::Display for PostEffectsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self.kind {
            PostEffectsErrorKind::Values => {
                "MalformedRequiredProfileData phase=8C scanner post effects"
            }
            PostEffectsErrorKind::RadiusRange => {
                "scanner post-effects Gaussian radius outside int32 range"
            }
            PostEffectsErrorKind::ZeroHash => "",
        })
    }
}
impl std::error::Error for PostEffectsError {}

fn radius(sigma: f32, lane: u32) -> Result<i32, PostEffectsError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Ok(0);
    }
    // Preserve the native float overload: no f64 promotion, mul_add or early cap.
    let r = 3.0_f32 * sigma + 0.5_f32;
    if r <= 0.0 || !r.is_finite() {
        return Ok(0);
    }
    if r >= 2147483648.0_f32 {
        return Err(PostEffectsError {
            kind: PostEffectsErrorKind::RadiusRange,
            lane,
            rejected: None,
        });
    }
    Ok((r as i32).min(75))
}
fn sigma(sigma: f32, lane: u32) -> Result<f32, PostEffectsError> {
    if !sigma.is_finite() {
        return Ok(sigma);
    }
    if sigma <= 0.0 {
        return Ok(0.0);
    }
    Ok(if radius(sigma, lane)? > 0 { sigma } else { 0.0 })
}

pub fn describe_post_effects(params: PostEffectsParams) -> Result<PostEffects, PostEffectsError> {
    let printing = matches!(
        params.route,
        ScanRoute::NegativePrint | ScanRoute::PositivePrint
    );
    let glare_active = printing && params.glare_active && params.glare_percent > 0.0;
    let glare_percent = if glare_active {
        params.glare_percent
    } else {
        0.0
    };
    let glare_roughness = if glare_active {
        params.glare_roughness
    } else {
        0.0
    };
    let glare_sigma = if glare_active {
        sigma(params.glare_blur_sigma_px, 1)?
    } else {
        0.0
    };
    let glare_radius = radius(glare_sigma, 1)?;
    let lens_sigma = sigma(params.lens_blur_sigma_px, 2)?;
    let lens_radius = radius(lens_sigma, 2)?;
    let unsharp_sigma = sigma(params.unsharp_sigma_px, 3)?;
    let unsharp_active = unsharp_sigma > 0.0 && params.unsharp_amount > 0.0;
    let unsharp_sigma = if unsharp_active { unsharp_sigma } else { 0.0 };
    let unsharp_radius = radius(unsharp_sigma, 3)?;
    let unsharp_amount = if unsharp_active {
        params.unsharp_amount
    } else {
        0.0
    };
    let fields = PostEffectsAttempt {
        route: params.route,
        glare_active,
        values: [
            glare_percent,
            glare_roughness,
            glare_sigma,
            lens_sigma,
            unsharp_sigma,
            unsharp_amount,
        ],
        radii: [glare_radius, lens_radius, unsharp_radius],
    };
    for (index, value) in fields.values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 {
            return Err(PostEffectsError {
                kind: PostEffectsErrorKind::Values,
                lane: index as u32 + 1,
                rejected: Some(fields),
            });
        }
    }
    if !glare_active && lens_sigma <= 0.0 && unsharp_sigma <= 0.0 {
        return Ok(PostEffects { fields, hash: 0 });
    }
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(&mut identity, &[params.route as u8, u8::from(glare_active)]);
    for value in fields.values {
        hash::update_bytes(&mut identity, &value.to_bits().to_le_bytes());
    }
    for value in fields.radii {
        hash::update_bytes(&mut identity, &value.to_le_bytes());
    }
    hash::update_bytes(&mut identity, &SCHEMA.to_le_bytes());
    if identity == 0 {
        return Err(PostEffectsError {
            kind: PostEffectsErrorKind::ZeroHash,
            lane: 0,
            rejected: Some(fields),
        });
    }
    Ok(PostEffects {
        fields,
        hash: identity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn params(route: ScanRoute) -> PostEffectsParams {
        PostEffectsParams {
            route,
            glare_active: true,
            glare_percent: 2.0,
            glare_roughness: 0.3,
            glare_blur_sigma_px: 0.7,
            lens_blur_sigma_px: 1.25,
            unsharp_sigma_px: 0.7,
            unsharp_amount: 0.8,
        }
    }
    fn project(fields: &PostEffectsAttempt, hash: u64) -> Value {
        json!({"route": fields.route() as u32, "glare_active": fields.glare_active(),
            "values": fields.values().map(f32::to_bits), "radii": fields.radii(),
            "schema": fields.schema(), "hash": hash})
    }
    #[test]
    fn complete_native_fields_identity_failures_and_owned_lifetime() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scanner/post-effects");
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
        let fixture = read(manifest["payloads"][logical].as_str().unwrap());
        let context = read(manifest["contexts"][logical].as_str().unwrap());
        assert_eq!(context["preset"], preset);
        assert_eq!(fixture["cases"].as_array().unwrap().len(), 400);
        let rows = fixture["cases"].as_array().unwrap();
        let eligible = rows
            .iter()
            .filter(|row| row["input"]["route"].as_u64().unwrap() <= 3)
            .collect::<Vec<_>>();
        let cases_ids: Vec<_> =
            IntoIterator::into_iter(eligible.iter().map(|row| row["id"].as_str().unwrap()))
                .map(|id| AsRef::<str>::as_ref(&id).to_owned())
                .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "scanner-post-effects"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "scanner-post-effects"
        );
        let successful_ids = eligible
            .iter()
            .filter(|row| row["native"]["success"].as_bool().unwrap())
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let retained_cases_ids: Vec<_> = IntoIterator::into_iter(successful_ids)
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !retained_cases_ids.is_empty(),
            "empty case set: {}",
            "scanner-post-effects-retained"
        );
        let retained_cases_count = retained_cases_ids.len();
        let mut retained_cases_remaining = retained_cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            retained_cases_remaining.len(),
            retained_cases_count,
            "duplicate case IDs: {}",
            "scanner-post-effects-retained"
        );
        let mut held = Vec::new();
        for row in rows {
            let input = &row["input"];
            let route = match input["route"].as_u64().unwrap() {
                0 => ScanRoute::NegativeDirect,
                1 => ScanRoute::NegativePrint,
                2 => ScanRoute::PositiveDirect,
                3 => ScanRoute::PositivePrint,
                _ => continue,
            };
            let case = AsRef::<str>::as_ref(&(row["id"].as_str().unwrap())).to_owned();
            let v: [f32; 6] =
                std::array::from_fn(
                    |i| f32::from_bits(input["values"][i].as_u64().unwrap() as u32),
                );
            let actual = describe_post_effects(PostEffectsParams {
                route,
                glare_active: input["glare_active"].as_bool().unwrap(),
                glare_percent: v[0],
                glare_roughness: v[1],
                glare_blur_sigma_px: v[2],
                lens_blur_sigma_px: v[3],
                unsharp_sigma_px: v[4],
                unsharp_amount: v[5],
            });
            match actual {
                Ok(value) => {
                    assert_eq!(row["native"]["success"], true, "{}", row["id"]);
                    assert_eq!(value.active(), value.hash() != 0);
                    held.push((
                        row["id"].as_str().unwrap().to_owned(),
                        value,
                        row["native"]["descriptor"].clone(),
                    ));
                }
                Err(error) => {
                    assert_eq!(row["native"]["success"], false, "{}", row["id"]);
                    assert_eq!(error.kind(), PostEffectsErrorKind::Values);
                    assert_eq!(
                        error.to_string(),
                        row["native"]["diagnostic"].as_str().unwrap()
                    );
                    let attempt = error.rejected_attempt().unwrap();
                    assert_eq!(
                        project(attempt, 0),
                        row["native"]["descriptor"],
                        "{}",
                        row["id"]
                    );
                    let first = attempt
                        .values()
                        .iter()
                        .position(|v| !v.is_finite() || *v < 0.0)
                        .unwrap();
                    assert_eq!(error.lane(), first as u32 + 1);
                }
            }
            assert!(
                cases_remaining.remove(&case),
                "unexpected or repeated comparison: {case}"
            );
        }
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
        for (id, value, expected) in held {
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            assert_eq!(project(value.fields(), value.hash()), expected);
            assert!(
                retained_cases_remaining.remove(&case),
                "unexpected or repeated comparison: {case}"
            );
        }
        assert!(
            retained_cases_remaining.is_empty(),
            "missing comparisons: {retained_cases_remaining:?}"
        );
    }
    #[test]
    fn radius_range_order_masking_and_overflow_are_representation_contracts() {
        let cases_ids: Vec<_> = IntoIterator::into_iter([
            "below-threshold",
            "at-threshold",
            "glare-radius-lane",
            "lens-radius-lane",
            "unsharp-radius-lane",
            "direct-route-mask",
            "maximum-radius-overflow",
        ])
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "scanner-post-effect-radius-boundaries"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "scanner-post-effect-radius-boundaries"
        );
        let threshold = 2147483648.0_f32 / 3.0;
        let lower = f32::from_bits(threshold.to_bits() - 1);
        let case = AsRef::<str>::as_ref(&("below-threshold")).to_owned();
        assert_eq!(radius(lower, 1).unwrap(), 75);
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
        let case = AsRef::<str>::as_ref(&("at-threshold")).to_owned();
        assert_eq!(
            radius(threshold, 1).unwrap_err().kind(),
            PostEffectsErrorKind::RadiusRange
        );
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
        for lane in 1..=3 {
            let id = match lane {
                1 => "glare-radius-lane",
                2 => "lens-radius-lane",
                _ => "unsharp-radius-lane",
            };
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            let mut p = params(ScanRoute::NegativePrint);
            match lane {
                1 => p.glare_blur_sigma_px = threshold,
                2 => p.lens_blur_sigma_px = threshold,
                _ => {
                    p.unsharp_sigma_px = threshold;
                    p.unsharp_amount = 0.0;
                }
            }
            p.glare_roughness = f32::NAN;
            let error = describe_post_effects(p).unwrap_err();
            assert_eq!(error.kind(), PostEffectsErrorKind::RadiusRange);
            assert_eq!(error.lane(), lane);
            assert!(error.rejected_attempt().is_none());
            assert!(
                cases_remaining.remove(&case),
                "unexpected or repeated comparison: {case}"
            );
        }
        let case = AsRef::<str>::as_ref(&("direct-route-mask")).to_owned();
        let mut p = params(ScanRoute::NegativeDirect);
        p.glare_blur_sigma_px = threshold;
        assert!(describe_post_effects(p).is_ok());
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
        let case = AsRef::<str>::as_ref(&("maximum-radius-overflow")).to_owned();
        p = params(ScanRoute::NegativePrint);
        p.glare_blur_sigma_px = f32::MAX;
        p.lens_blur_sigma_px = f32::MAX;
        p.unsharp_sigma_px = f32::MAX;
        let value = describe_post_effects(p).unwrap();
        assert_eq!(value.fields().radii(), &[0; 3]);
        assert!(
            value.fields().values()[2..5]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
    }
}
