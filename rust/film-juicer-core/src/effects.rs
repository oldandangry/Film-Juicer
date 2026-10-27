//! Film-Juicer physical effects from captured controls and completed grain facts.

use crate::hash;

pub mod frame;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub film_dust_amount: f32,
    pub film_scratch_amount: f32,
    pub gate_dust_amount: f32,
    pub gate_scratch_amount: f32,
    pub gate_weave_amount: f64,
    pub grain_active: bool,
    pub grain_debug_view: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Probability,
}

#[derive(Debug)]
pub struct Dust {
    cell_width_mm: f32,
    cell_height_mm: f32,
    slot_probability: f32,
    softness_min_mm: f32,
    softness_max_mm: f32,
    softness_size_cap_fraction: f32,
    support_x_mm: f32,
    support_y_mm: f32,
    fiber_fraction: f32,
    fiber_drift_fraction: f32,
    fiber_first_knot_min: f32,
    fiber_first_knot_max: f32,
    fiber_second_knot_min: f32,
    fiber_second_knot_max: f32,
    fiber_interior_width_min_fraction: f32,
    fiber_interior_width_max_fraction: f32,
    diameter_min_mm: f32,
    diameter_bulk_max_mm: f32,
    diameter_max_mm: f32,
    diameter_tail_fraction: f32,
    fiber_length_min_mm: f32,
    fiber_length_max_mm: f32,
    fiber_width_min_mm: f32,
    fiber_width_max_mm: f32,
    opacity_faint_cumulative: f32,
    opacity_intermediate_cumulative: f32,
    compact_opacity_min: f32,
    compact_opacity_faint_end: f32,
    compact_opacity_intermediate_end: f32,
    compact_opacity_max: f32,
    fiber_opacity_min: f32,
    fiber_opacity_faint_end: f32,
    fiber_opacity_intermediate_end: f32,
    fiber_opacity_max: f32,
    compact_dominant_aspect_min: f32,
    compact_dominant_aspect_max: f32,
    compact_subsidiary_scale_min: f32,
    compact_subsidiary_scale_max: f32,
    compact_subsidiary_aspect_min: f32,
    compact_subsidiary_aspect_max: f32,
    compact_subsidiary_offset_max: f32,
    compact_subsidiary_angle_max_radians: f32,
}

#[derive(Debug)]
pub struct Scratch {
    cell_width_mm: f32,
    cell_height_mm: f32,
    slot_probability: f32,
    softness_min_mm: f32,
    softness_max_mm: f32,
    softness_size_cap_fraction: f32,
    support_x_mm: f32,
    support_y_mm: f32,
    length_min_mm: f32,
    length_bulk_max_mm: f32,
    length_max_mm: f32,
    length_tail_fraction: f32,
    width_min_mm: f32,
    width_bulk_max_mm: f32,
    width_max_mm: f32,
    width_tail_fraction: f32,
    drift_fraction: f32,
    first_knot_min: f32,
    first_knot_max: f32,
    second_knot_min: f32,
    second_knot_max: f32,
    interior_width_min_fraction: f32,
    interior_width_max_fraction: f32,
    interior_depth_min_fraction: f32,
    interior_depth_max_fraction: f32,
    endpoint_abrupt_probability: f32,
    interruption_probability: f32,
    gap_center_min: f32,
    gap_center_max: f32,
    gap_span_min: f32,
    gap_span_max: f32,
    scuff_probability: f32,
    scuff_length_max_mm: f32,
    scuff_angle_max_radians: f32,
    strength_min: f32,
    strength_max: f32,
}

#[derive(Debug)]
pub struct Recipe {
    film_dust: Dust,
    film_scratch: Scratch,
    gate_dust: Dust,
    gate_scratch: Scratch,
    gate_weave_amount: f64,
    active: bool,
    hash: u64,
}

fn normalize_amount(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let lower = if value < 0.0 { 0.0 } else { value };
    if lower > 10.0 { 10.0 } else { lower }
}
fn normalize_weave(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    let lower = if value < 0.0 { 0.0 } else { value };
    if lower > 10.0 { 10.0 } else { lower }
}
fn ordered_max(first: f32, second: f32) -> f32 {
    if first < second { second } else { first }
}

impl Dust {
    fn zero() -> Self {
        Self {
            cell_width_mm: 0.0,
            cell_height_mm: 0.0,
            slot_probability: 0.0,
            softness_min_mm: 0.0,
            softness_max_mm: 0.0,
            softness_size_cap_fraction: 0.0,
            support_x_mm: 0.0,
            support_y_mm: 0.0,
            fiber_fraction: 0.0,
            fiber_drift_fraction: 0.0,
            fiber_first_knot_min: 0.0,
            fiber_first_knot_max: 0.0,
            fiber_second_knot_min: 0.0,
            fiber_second_knot_max: 0.0,
            fiber_interior_width_min_fraction: 0.0,
            fiber_interior_width_max_fraction: 0.0,
            diameter_min_mm: 0.0,
            diameter_bulk_max_mm: 0.0,
            diameter_max_mm: 0.0,
            diameter_tail_fraction: 0.0,
            fiber_length_min_mm: 0.0,
            fiber_length_max_mm: 0.0,
            fiber_width_min_mm: 0.0,
            fiber_width_max_mm: 0.0,
            opacity_faint_cumulative: 0.0,
            opacity_intermediate_cumulative: 0.0,
            compact_opacity_min: 0.0,
            compact_opacity_faint_end: 0.0,
            compact_opacity_intermediate_end: 0.0,
            compact_opacity_max: 0.0,
            fiber_opacity_min: 0.0,
            fiber_opacity_faint_end: 0.0,
            fiber_opacity_intermediate_end: 0.0,
            fiber_opacity_max: 0.0,
            compact_dominant_aspect_min: 0.0,
            compact_dominant_aspect_max: 0.0,
            compact_subsidiary_scale_min: 0.0,
            compact_subsidiary_scale_max: 0.0,
            compact_subsidiary_aspect_min: 0.0,
            compact_subsidiary_aspect_max: 0.0,
            compact_subsidiary_offset_max: 0.0,
            compact_subsidiary_angle_max_radians: 0.0,
        }
    }
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        clippy::excessive_precision,
        clippy::approx_constant,
        reason = "preserve native comparison and literal rounding in physical policy"
    )]
    fn new(amount: f32, maximum_rate: f32, softness_min_mm: f32, softness_max_mm: f32) -> Self {
        let mut p = Self::zero();
        if !(amount > 0.0) {
            return p;
        }
        p.cell_width_mm = 0.5f32;
        p.cell_height_mm = 0.5f32;
        p.slot_probability =
            maximum_rate * (amount / 10.0f32).powf(2.2f32) * p.cell_width_mm * p.cell_height_mm
                / 2.0f32;
        p.softness_min_mm = softness_min_mm;
        p.softness_max_mm = softness_max_mm;
        p.softness_size_cap_fraction = 0.25f32;
        p.fiber_fraction = 0.25f32;
        p.fiber_drift_fraction = 0.10f32;
        p.fiber_first_knot_min = 0.08f32;
        p.fiber_first_knot_max = 0.35f32;
        p.fiber_second_knot_min = 0.65f32;
        p.fiber_second_knot_max = 0.92f32;
        p.fiber_interior_width_min_fraction = 0.45f32;
        p.fiber_interior_width_max_fraction = 1.0f32;
        p.diameter_min_mm = 0.008f32;
        p.diameter_bulk_max_mm = 0.080f32;
        p.diameter_max_mm = 0.250f32;
        p.diameter_tail_fraction = 0.20f32;
        p.fiber_length_min_mm = 0.04f32;
        p.fiber_length_max_mm = 0.65f32;
        p.fiber_width_min_mm = 0.003f32;
        p.fiber_width_max_mm = 0.025f32;
        p.opacity_faint_cumulative = 0.50f32;
        p.opacity_intermediate_cumulative = 0.85f32;
        p.compact_opacity_min = 0.10f32;
        p.compact_opacity_faint_end = 0.45f32;
        p.compact_opacity_intermediate_end = 0.85f32;
        p.compact_opacity_max = 0.99f32;
        p.fiber_opacity_min = 0.08f32;
        p.fiber_opacity_faint_end = 0.35f32;
        p.fiber_opacity_intermediate_end = 0.75f32;
        p.fiber_opacity_max = 0.97f32;
        p.compact_dominant_aspect_min = 0.52f32;
        p.compact_dominant_aspect_max = 0.95f32;
        p.compact_subsidiary_scale_min = 0.30f32;
        p.compact_subsidiary_scale_max = 0.75f32;
        p.compact_subsidiary_aspect_min = 0.45f32;
        p.compact_subsidiary_aspect_max = 1.0f32;
        p.compact_subsidiary_offset_max = 0.68f32;
        p.compact_subsidiary_angle_max_radians = 3.14159265359f32;
        p.support_x_mm = p
            .fiber_length_max_mm
            .mul_add(0.5f32 + p.fiber_drift_fraction, p.fiber_width_max_mm);
        p.support_y_mm = p.support_x_mm;
        p
    }
    pub fn cell_width_mm(&self) -> f32 {
        self.cell_width_mm
    }
    pub fn cell_height_mm(&self) -> f32 {
        self.cell_height_mm
    }
    pub fn slot_probability(&self) -> f32 {
        self.slot_probability
    }
    pub fn softness_min_mm(&self) -> f32 {
        self.softness_min_mm
    }
    pub fn softness_max_mm(&self) -> f32 {
        self.softness_max_mm
    }
    pub fn softness_size_cap_fraction(&self) -> f32 {
        self.softness_size_cap_fraction
    }
    pub fn support_x_mm(&self) -> f32 {
        self.support_x_mm
    }
    pub fn support_y_mm(&self) -> f32 {
        self.support_y_mm
    }
    pub fn fiber_fraction(&self) -> f32 {
        self.fiber_fraction
    }
    pub fn fiber_drift_fraction(&self) -> f32 {
        self.fiber_drift_fraction
    }
    pub fn fiber_first_knot_min(&self) -> f32 {
        self.fiber_first_knot_min
    }
    pub fn fiber_first_knot_max(&self) -> f32 {
        self.fiber_first_knot_max
    }
    pub fn fiber_second_knot_min(&self) -> f32 {
        self.fiber_second_knot_min
    }
    pub fn fiber_second_knot_max(&self) -> f32 {
        self.fiber_second_knot_max
    }
    pub fn fiber_interior_width_min_fraction(&self) -> f32 {
        self.fiber_interior_width_min_fraction
    }
    pub fn fiber_interior_width_max_fraction(&self) -> f32 {
        self.fiber_interior_width_max_fraction
    }
    pub fn diameter_min_mm(&self) -> f32 {
        self.diameter_min_mm
    }
    pub fn diameter_bulk_max_mm(&self) -> f32 {
        self.diameter_bulk_max_mm
    }
    pub fn diameter_max_mm(&self) -> f32 {
        self.diameter_max_mm
    }
    pub fn diameter_tail_fraction(&self) -> f32 {
        self.diameter_tail_fraction
    }
    pub fn fiber_length_min_mm(&self) -> f32 {
        self.fiber_length_min_mm
    }
    pub fn fiber_length_max_mm(&self) -> f32 {
        self.fiber_length_max_mm
    }
    pub fn fiber_width_min_mm(&self) -> f32 {
        self.fiber_width_min_mm
    }
    pub fn fiber_width_max_mm(&self) -> f32 {
        self.fiber_width_max_mm
    }
    pub fn opacity_faint_cumulative(&self) -> f32 {
        self.opacity_faint_cumulative
    }
    pub fn opacity_intermediate_cumulative(&self) -> f32 {
        self.opacity_intermediate_cumulative
    }
    pub fn compact_opacity_min(&self) -> f32 {
        self.compact_opacity_min
    }
    pub fn compact_opacity_faint_end(&self) -> f32 {
        self.compact_opacity_faint_end
    }
    pub fn compact_opacity_intermediate_end(&self) -> f32 {
        self.compact_opacity_intermediate_end
    }
    pub fn compact_opacity_max(&self) -> f32 {
        self.compact_opacity_max
    }
    pub fn fiber_opacity_min(&self) -> f32 {
        self.fiber_opacity_min
    }
    pub fn fiber_opacity_faint_end(&self) -> f32 {
        self.fiber_opacity_faint_end
    }
    pub fn fiber_opacity_intermediate_end(&self) -> f32 {
        self.fiber_opacity_intermediate_end
    }
    pub fn fiber_opacity_max(&self) -> f32 {
        self.fiber_opacity_max
    }
    pub fn compact_dominant_aspect_min(&self) -> f32 {
        self.compact_dominant_aspect_min
    }
    pub fn compact_dominant_aspect_max(&self) -> f32 {
        self.compact_dominant_aspect_max
    }
    pub fn compact_subsidiary_scale_min(&self) -> f32 {
        self.compact_subsidiary_scale_min
    }
    pub fn compact_subsidiary_scale_max(&self) -> f32 {
        self.compact_subsidiary_scale_max
    }
    pub fn compact_subsidiary_aspect_min(&self) -> f32 {
        self.compact_subsidiary_aspect_min
    }
    pub fn compact_subsidiary_aspect_max(&self) -> f32 {
        self.compact_subsidiary_aspect_max
    }
    pub fn compact_subsidiary_offset_max(&self) -> f32 {
        self.compact_subsidiary_offset_max
    }
    pub fn compact_subsidiary_angle_max_radians(&self) -> f32 {
        self.compact_subsidiary_angle_max_radians
    }
    fn append_hash(&self, identity: &mut u64) {
        let enabled = self.slot_probability > 0.0;
        hash::update_bytes(identity, &[u8::from(enabled)]);
        if enabled {
            for value in [
                self.cell_width_mm,
                self.cell_height_mm,
                self.slot_probability,
                self.softness_min_mm,
                self.softness_max_mm,
                self.softness_size_cap_fraction,
                self.support_x_mm,
                self.support_y_mm,
                self.fiber_fraction,
                self.fiber_drift_fraction,
                self.fiber_first_knot_min,
                self.fiber_first_knot_max,
                self.fiber_second_knot_min,
                self.fiber_second_knot_max,
                self.fiber_interior_width_min_fraction,
                self.fiber_interior_width_max_fraction,
                self.diameter_min_mm,
                self.diameter_bulk_max_mm,
                self.diameter_max_mm,
                self.diameter_tail_fraction,
                self.fiber_length_min_mm,
                self.fiber_length_max_mm,
                self.fiber_width_min_mm,
                self.fiber_width_max_mm,
                self.opacity_faint_cumulative,
                self.opacity_intermediate_cumulative,
                self.compact_opacity_min,
                self.compact_opacity_faint_end,
                self.compact_opacity_intermediate_end,
                self.compact_opacity_max,
                self.fiber_opacity_min,
                self.fiber_opacity_faint_end,
                self.fiber_opacity_intermediate_end,
                self.fiber_opacity_max,
                self.compact_dominant_aspect_min,
                self.compact_dominant_aspect_max,
                self.compact_subsidiary_scale_min,
                self.compact_subsidiary_scale_max,
                self.compact_subsidiary_aspect_min,
                self.compact_subsidiary_aspect_max,
                self.compact_subsidiary_offset_max,
                self.compact_subsidiary_angle_max_radians,
            ] {
                hash::update_bytes(identity, &value.to_bits().to_le_bytes());
            }
        }
    }
}

impl Scratch {
    fn zero() -> Self {
        Self {
            cell_width_mm: 0.0,
            cell_height_mm: 0.0,
            slot_probability: 0.0,
            softness_min_mm: 0.0,
            softness_max_mm: 0.0,
            softness_size_cap_fraction: 0.0,
            support_x_mm: 0.0,
            support_y_mm: 0.0,
            length_min_mm: 0.0,
            length_bulk_max_mm: 0.0,
            length_max_mm: 0.0,
            length_tail_fraction: 0.0,
            width_min_mm: 0.0,
            width_bulk_max_mm: 0.0,
            width_max_mm: 0.0,
            width_tail_fraction: 0.0,
            drift_fraction: 0.0,
            first_knot_min: 0.0,
            first_knot_max: 0.0,
            second_knot_min: 0.0,
            second_knot_max: 0.0,
            interior_width_min_fraction: 0.0,
            interior_width_max_fraction: 0.0,
            interior_depth_min_fraction: 0.0,
            interior_depth_max_fraction: 0.0,
            endpoint_abrupt_probability: 0.0,
            interruption_probability: 0.0,
            gap_center_min: 0.0,
            gap_center_max: 0.0,
            gap_span_min: 0.0,
            gap_span_max: 0.0,
            scuff_probability: 0.0,
            scuff_length_max_mm: 0.0,
            scuff_angle_max_radians: 0.0,
            strength_min: 0.0,
            strength_max: 0.0,
        }
    }
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        clippy::excessive_precision,
        reason = "preserve native comparison and literal rounding in physical policy"
    )]
    fn new(
        amount: f32,
        maximum_length_density: f32,
        strength_max: f32,
        softness_min_mm: f32,
        softness_max_mm: f32,
    ) -> Self {
        let mut p = Self::zero();
        if !(amount > 0.0) {
            return p;
        }
        p.cell_width_mm = 1.0f32;
        p.cell_height_mm = 8.0f32;
        p.length_min_mm = 0.25f32;
        p.length_bulk_max_mm = 6.0f32;
        p.length_max_mm = 24.0f32;
        p.length_tail_fraction = 0.16f32;
        // Each arc-length mixture branch uses E[u^3] = 1/4.
        let first_weight = 1.0f32 - p.length_tail_fraction;
        let first_length = p.length_min_mm + (p.length_bulk_max_mm - p.length_min_mm) * 0.25f32;
        let tail_length = p.length_tail_fraction
            * (p.length_bulk_max_mm + (p.length_max_mm - p.length_bulk_max_mm) * 0.25f32);
        // Preserve the native Windows Release contraction and separate rounding elsewhere.
        let expected_length = if cfg!(all(target_os = "windows", not(debug_assertions))) {
            first_weight.mul_add(first_length, tail_length)
        } else {
            first_weight * first_length + tail_length
        };
        p.slot_probability = maximum_length_density * (amount / 10.0f32).powf(2.2f32)
            / expected_length
            * p.cell_width_mm
            * p.cell_height_mm
            / 2.0f32;
        p.width_min_mm = 0.002f32;
        p.width_bulk_max_mm = 0.025f32;
        p.width_max_mm = 0.080f32;
        p.width_tail_fraction = 0.18f32;
        p.drift_fraction = 0.025f32;
        p.first_knot_min = 0.06f32;
        p.first_knot_max = 0.28f32;
        p.second_knot_min = 0.72f32;
        p.second_knot_max = 0.94f32;
        p.interior_width_min_fraction = 0.35f32;
        p.interior_width_max_fraction = 1.0f32;
        p.interior_depth_min_fraction = 0.35f32;
        p.interior_depth_max_fraction = 1.0f32;
        p.endpoint_abrupt_probability = 0.30f32;
        p.interruption_probability = 0.15f32;
        p.gap_center_min = 0.25f32;
        p.gap_center_max = 0.75f32;
        p.gap_span_min = 0.02f32;
        p.gap_span_max = 0.12f32;
        p.scuff_probability = 0.10f32;
        p.scuff_length_max_mm = 0.75f32;
        p.scuff_angle_max_radians = 0.61086523820f32;
        p.strength_min = strength_max * 0.15f32;
        p.strength_max = strength_max;
        p.softness_min_mm = softness_min_mm;
        p.softness_max_mm = softness_max_mm;
        p.softness_size_cap_fraction = 0.25f32;
        let transport_support_x = p.length_max_mm * p.drift_fraction + p.width_max_mm;
        let scuff_support_x = 0.5f32 * p.scuff_length_max_mm * p.scuff_angle_max_radians.sin()
            + p.scuff_length_max_mm * p.drift_fraction
            + p.width_max_mm;
        let scuff_support_y = 0.5f32 * p.scuff_length_max_mm * p.scuff_angle_max_radians.cos()
            + p.scuff_length_max_mm * p.drift_fraction
            + p.width_max_mm;
        p.support_x_mm = ordered_max(transport_support_x, scuff_support_x);
        p.support_y_mm = ordered_max(p.length_max_mm * 0.5f32 + p.width_max_mm, scuff_support_y);
        p
    }
    pub fn cell_width_mm(&self) -> f32 {
        self.cell_width_mm
    }
    pub fn cell_height_mm(&self) -> f32 {
        self.cell_height_mm
    }
    pub fn slot_probability(&self) -> f32 {
        self.slot_probability
    }
    pub fn softness_min_mm(&self) -> f32 {
        self.softness_min_mm
    }
    pub fn softness_max_mm(&self) -> f32 {
        self.softness_max_mm
    }
    pub fn softness_size_cap_fraction(&self) -> f32 {
        self.softness_size_cap_fraction
    }
    pub fn support_x_mm(&self) -> f32 {
        self.support_x_mm
    }
    pub fn support_y_mm(&self) -> f32 {
        self.support_y_mm
    }
    pub fn length_min_mm(&self) -> f32 {
        self.length_min_mm
    }
    pub fn length_bulk_max_mm(&self) -> f32 {
        self.length_bulk_max_mm
    }
    pub fn length_max_mm(&self) -> f32 {
        self.length_max_mm
    }
    pub fn length_tail_fraction(&self) -> f32 {
        self.length_tail_fraction
    }
    pub fn width_min_mm(&self) -> f32 {
        self.width_min_mm
    }
    pub fn width_bulk_max_mm(&self) -> f32 {
        self.width_bulk_max_mm
    }
    pub fn width_max_mm(&self) -> f32 {
        self.width_max_mm
    }
    pub fn width_tail_fraction(&self) -> f32 {
        self.width_tail_fraction
    }
    pub fn drift_fraction(&self) -> f32 {
        self.drift_fraction
    }
    pub fn first_knot_min(&self) -> f32 {
        self.first_knot_min
    }
    pub fn first_knot_max(&self) -> f32 {
        self.first_knot_max
    }
    pub fn second_knot_min(&self) -> f32 {
        self.second_knot_min
    }
    pub fn second_knot_max(&self) -> f32 {
        self.second_knot_max
    }
    pub fn interior_width_min_fraction(&self) -> f32 {
        self.interior_width_min_fraction
    }
    pub fn interior_width_max_fraction(&self) -> f32 {
        self.interior_width_max_fraction
    }
    pub fn interior_depth_min_fraction(&self) -> f32 {
        self.interior_depth_min_fraction
    }
    pub fn interior_depth_max_fraction(&self) -> f32 {
        self.interior_depth_max_fraction
    }
    pub fn endpoint_abrupt_probability(&self) -> f32 {
        self.endpoint_abrupt_probability
    }
    pub fn interruption_probability(&self) -> f32 {
        self.interruption_probability
    }
    pub fn gap_center_min(&self) -> f32 {
        self.gap_center_min
    }
    pub fn gap_center_max(&self) -> f32 {
        self.gap_center_max
    }
    pub fn gap_span_min(&self) -> f32 {
        self.gap_span_min
    }
    pub fn gap_span_max(&self) -> f32 {
        self.gap_span_max
    }
    pub fn scuff_probability(&self) -> f32 {
        self.scuff_probability
    }
    pub fn scuff_length_max_mm(&self) -> f32 {
        self.scuff_length_max_mm
    }
    pub fn scuff_angle_max_radians(&self) -> f32 {
        self.scuff_angle_max_radians
    }
    pub fn strength_min(&self) -> f32 {
        self.strength_min
    }
    pub fn strength_max(&self) -> f32 {
        self.strength_max
    }
    fn append_hash(&self, identity: &mut u64) {
        let enabled = self.slot_probability > 0.0;
        hash::update_bytes(identity, &[u8::from(enabled)]);
        if enabled {
            for value in [
                self.cell_width_mm,
                self.cell_height_mm,
                self.slot_probability,
                self.softness_min_mm,
                self.softness_max_mm,
                self.softness_size_cap_fraction,
                self.support_x_mm,
                self.support_y_mm,
                self.length_min_mm,
                self.length_bulk_max_mm,
                self.length_max_mm,
                self.length_tail_fraction,
                self.width_min_mm,
                self.width_bulk_max_mm,
                self.width_max_mm,
                self.width_tail_fraction,
                self.drift_fraction,
                self.first_knot_min,
                self.first_knot_max,
                self.second_knot_min,
                self.second_knot_max,
                self.interior_width_min_fraction,
                self.interior_width_max_fraction,
                self.interior_depth_min_fraction,
                self.interior_depth_max_fraction,
                self.endpoint_abrupt_probability,
                self.interruption_probability,
                self.gap_center_min,
                self.gap_center_max,
                self.gap_span_min,
                self.gap_span_max,
                self.scuff_probability,
                self.scuff_length_max_mm,
                self.scuff_angle_max_radians,
                self.strength_min,
                self.strength_max,
            ] {
                hash::update_bytes(identity, &value.to_bits().to_le_bytes());
            }
        }
    }
}

fn recipe_hash(recipe: &Recipe) -> u64 {
    if !recipe.active {
        return 0;
    }
    let mut identity = hash::FNV_OFFSET;
    recipe.film_dust.append_hash(&mut identity);
    recipe.film_scratch.append_hash(&mut identity);
    recipe.gate_dust.append_hash(&mut identity);
    recipe.gate_scratch.append_hash(&mut identity);
    hash::update_bytes(
        &mut identity,
        &recipe.gate_weave_amount.to_bits().to_le_bytes(),
    );
    hash::update_bytes(&mut identity, &[u8::from(recipe.active)]);
    identity
}

impl Recipe {
    fn zero() -> Self {
        Self {
            film_dust: Dust::zero(),
            film_scratch: Scratch::zero(),
            gate_dust: Dust::zero(),
            gate_scratch: Scratch::zero(),
            gate_weave_amount: 0.0,
            active: false,
            hash: 0,
        }
    }
    pub fn new(params: Params) -> Result<Self, Error> {
        #[cfg(any(test, feature = "test-support"))]
        let probability = test_support::take();
        if params.grain_active && params.grain_debug_view != 0 {
            return Ok(Self::zero());
        }
        let film_dust = Dust::new(
            normalize_amount(params.film_dust_amount),
            0.80,
            0.00025,
            0.003,
        );
        let gate_dust = Dust::new(
            normalize_amount(params.gate_dust_amount),
            0.40,
            0.0005,
            0.005,
        );
        let film_scratch = Scratch::new(
            normalize_amount(params.film_scratch_amount),
            0.080,
            0.95,
            0.00015,
            0.0015,
        );
        let gate_scratch = Scratch::new(
            normalize_amount(params.gate_scratch_amount),
            0.040,
            0.85,
            0.00025,
            0.003,
        );
        let gate_weave_amount = normalize_weave(params.gate_weave_amount);
        let mut recipe = Self {
            film_dust,
            film_scratch,
            gate_dust,
            gate_scratch,
            gate_weave_amount,
            active: false,
            hash: 0,
        };
        #[cfg(any(test, feature = "test-support"))]
        if let Some(probability) = probability {
            recipe.film_dust.slot_probability = probability;
        }
        for probability in [
            recipe.film_dust.slot_probability,
            recipe.gate_dust.slot_probability,
            recipe.film_scratch.slot_probability,
            recipe.gate_scratch.slot_probability,
        ] {
            if !probability.is_finite() || probability < 0.0 || probability >= 0.25 {
                return Err(Error::Probability);
            }
        }
        recipe.active = recipe.film_dust.slot_probability > 0.0
            || recipe.film_scratch.slot_probability > 0.0
            || recipe.gate_dust.slot_probability > 0.0
            || recipe.gate_scratch.slot_probability > 0.0
            || recipe.gate_weave_amount > 0.0;
        recipe.hash = recipe_hash(&recipe);
        Ok(recipe)
    }
    pub fn film_dust(&self) -> &Dust {
        &self.film_dust
    }
    pub fn film_scratch(&self) -> &Scratch {
        &self.film_scratch
    }
    pub fn gate_dust(&self) -> &Dust {
        &self.gate_dust
    }
    pub fn gate_scratch(&self) -> &Scratch {
        &self.gate_scratch
    }
    pub fn gate_weave_amount(&self) -> f64 {
        self.gate_weave_amount
    }
    pub fn active(&self) -> bool {
        self.active
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use std::cell::Cell;
    thread_local! {static PROBABILITY: Cell<Option<f32>> = const {Cell::new(None)};}
    /// Substitute one computed probability immediately before the real guard.
    pub fn substitute_next_probability(probability: f32) {
        PROBABILITY.set(Some(probability));
    }
    pub(super) fn take() -> Option<f32> {
        PROBABILITY.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::path::PathBuf;

    fn fixture() -> Value {
        let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
        };
        let directory =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/film/effects-fixtures");
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        serde_json::from_slice(
            &std::fs::read(
                directory.join(manifest["presets"][preset]["literal"].as_str().unwrap()),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn params(value: &Value) -> Params {
        let amount =
            |index: usize| f32::from_bits(value["amounts"][index].as_u64().unwrap() as u32);
        Params {
            film_dust_amount: amount(0),
            film_scratch_amount: amount(1),
            gate_dust_amount: amount(2),
            gate_scratch_amount: amount(3),
            gate_weave_amount: f64::from_bits(value["weave"].as_u64().unwrap()),
            grain_active: value["grain_active"].as_bool().unwrap(),
            grain_debug_view: value["grain_debug_view"].as_i64().unwrap() as i32,
        }
    }
    fn dust(value: &Dust) -> Vec<u32> {
        vec![
            value.cell_width_mm().to_bits(),
            value.cell_height_mm().to_bits(),
            value.slot_probability().to_bits(),
            value.softness_min_mm().to_bits(),
            value.softness_max_mm().to_bits(),
            value.softness_size_cap_fraction().to_bits(),
            value.support_x_mm().to_bits(),
            value.support_y_mm().to_bits(),
            value.fiber_fraction().to_bits(),
            value.fiber_drift_fraction().to_bits(),
            value.fiber_first_knot_min().to_bits(),
            value.fiber_first_knot_max().to_bits(),
            value.fiber_second_knot_min().to_bits(),
            value.fiber_second_knot_max().to_bits(),
            value.fiber_interior_width_min_fraction().to_bits(),
            value.fiber_interior_width_max_fraction().to_bits(),
            value.diameter_min_mm().to_bits(),
            value.diameter_bulk_max_mm().to_bits(),
            value.diameter_max_mm().to_bits(),
            value.diameter_tail_fraction().to_bits(),
            value.fiber_length_min_mm().to_bits(),
            value.fiber_length_max_mm().to_bits(),
            value.fiber_width_min_mm().to_bits(),
            value.fiber_width_max_mm().to_bits(),
            value.opacity_faint_cumulative().to_bits(),
            value.opacity_intermediate_cumulative().to_bits(),
            value.compact_opacity_min().to_bits(),
            value.compact_opacity_faint_end().to_bits(),
            value.compact_opacity_intermediate_end().to_bits(),
            value.compact_opacity_max().to_bits(),
            value.fiber_opacity_min().to_bits(),
            value.fiber_opacity_faint_end().to_bits(),
            value.fiber_opacity_intermediate_end().to_bits(),
            value.fiber_opacity_max().to_bits(),
            value.compact_dominant_aspect_min().to_bits(),
            value.compact_dominant_aspect_max().to_bits(),
            value.compact_subsidiary_scale_min().to_bits(),
            value.compact_subsidiary_scale_max().to_bits(),
            value.compact_subsidiary_aspect_min().to_bits(),
            value.compact_subsidiary_aspect_max().to_bits(),
            value.compact_subsidiary_offset_max().to_bits(),
            value.compact_subsidiary_angle_max_radians().to_bits(),
        ]
    }
    fn scratch(value: &Scratch) -> Vec<u32> {
        vec![
            value.cell_width_mm().to_bits(),
            value.cell_height_mm().to_bits(),
            value.slot_probability().to_bits(),
            value.softness_min_mm().to_bits(),
            value.softness_max_mm().to_bits(),
            value.softness_size_cap_fraction().to_bits(),
            value.support_x_mm().to_bits(),
            value.support_y_mm().to_bits(),
            value.length_min_mm().to_bits(),
            value.length_bulk_max_mm().to_bits(),
            value.length_max_mm().to_bits(),
            value.length_tail_fraction().to_bits(),
            value.width_min_mm().to_bits(),
            value.width_bulk_max_mm().to_bits(),
            value.width_max_mm().to_bits(),
            value.width_tail_fraction().to_bits(),
            value.drift_fraction().to_bits(),
            value.first_knot_min().to_bits(),
            value.first_knot_max().to_bits(),
            value.second_knot_min().to_bits(),
            value.second_knot_max().to_bits(),
            value.interior_width_min_fraction().to_bits(),
            value.interior_width_max_fraction().to_bits(),
            value.interior_depth_min_fraction().to_bits(),
            value.interior_depth_max_fraction().to_bits(),
            value.endpoint_abrupt_probability().to_bits(),
            value.interruption_probability().to_bits(),
            value.gap_center_min().to_bits(),
            value.gap_center_max().to_bits(),
            value.gap_span_min().to_bits(),
            value.gap_span_max().to_bits(),
            value.scuff_probability().to_bits(),
            value.scuff_length_max_mm().to_bits(),
            value.scuff_angle_max_radians().to_bits(),
            value.strength_min().to_bits(),
            value.strength_max().to_bits(),
        ]
    }
    fn project(value: &Recipe) -> Value {
        json!({"film_dust": dust(value.film_dust()), "film_scratch": scratch(value.film_scratch()),
            "gate_dust": dust(value.gate_dust()), "gate_scratch": scratch(value.gate_scratch()),
            "weave": value.gate_weave_amount().to_bits(), "active": value.active(), "hash": value.hash()})
    }
    #[test]
    fn matches_all_frozen_native_words_and_identities() {
        let fixture = fixture();
        let rows = fixture["cases"].as_array().unwrap();
        assert_eq!(rows.len(), 220);
        let mut mismatches = Vec::new();
        for row in rows {
            let actual = project(&Recipe::new(params(&row["controls"])).unwrap());
            if actual != row["recipe"] {
                let mut fields = Vec::new();
                for family in ["film_dust", "film_scratch", "gate_dust", "gate_scratch"] {
                    for (index, (actual, expected)) in actual[family]
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(row["recipe"][family].as_array().unwrap())
                        .enumerate()
                    {
                        if actual != expected {
                            fields.push(format!(
                                "{family}[{index}] actual={actual} expected={expected}"
                            ));
                        }
                    }
                }
                for field in ["weave", "active", "hash"] {
                    if actual[field] != row["recipe"][field] {
                        fields.push(format!(
                            "{field} actual={} expected={}",
                            actual[field], row["recipe"][field]
                        ));
                    }
                }
                mismatches.push(format!("{}: {}", row["id"], fields.join(", ")));
            }
        }
        println!(
            "EFFECTS_NATIVE cases={} mismatches={}",
            rows.len(),
            mismatches.len()
        );
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }
    #[test]
    fn actual_probability_guard_and_entry_consumption_are_thread_local() {
        let fixture = fixture();
        let row = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == "effects-active")
            .unwrap();
        let controls = params(&row["controls"]);
        for probability in [f32::NAN, f32::INFINITY, -1.0, 0.25] {
            test_support::substitute_next_probability(probability);
            assert_eq!(Recipe::new(controls).unwrap_err(), Error::Probability);
            assert!(Recipe::new(controls).is_ok());
            test_support::substitute_next_probability(probability);
            let suppressed = Recipe::new(Params {
                grain_active: true,
                grain_debug_view: i32::MIN,
                ..controls
            })
            .unwrap();
            assert_eq!(project(&suppressed), project(&Recipe::zero()));
            assert!(Recipe::new(controls).is_ok());
        }
        test_support::substitute_next_probability(0.25);
        std::thread::spawn(move || assert!(Recipe::new(controls).is_ok()))
            .join()
            .unwrap();
        assert_eq!(Recipe::new(controls).unwrap_err(), Error::Probability);
        assert!(Recipe::new(controls).is_ok());
    }
    #[test]
    fn owned_result_and_zero_identity_do_not_encode_readiness() {
        assert!(
            !std::mem::needs_drop::<Recipe>()
                && !std::mem::needs_drop::<Dust>()
                && !std::mem::needs_drop::<Scratch>()
        );
        let fixture = fixture();
        let row = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == "effects-active")
            .unwrap();
        let (mut recipe, expected) = {
            let mut controls = params(&row["controls"]);
            let recipe = Recipe::new(controls).unwrap();
            let expected = project(&recipe);
            controls.film_dust_amount = 0.0;
            assert!(recipe.film_dust().slot_probability() > controls.film_dust_amount);
            (recipe, expected)
        };
        assert_eq!(project(&recipe), expected);
        recipe.hash = 0;
        assert!(recipe.active());
        assert_eq!(recipe.hash(), 0);
    }
}
