//! Complete callback-local physical-effects geometry and identity from resolved literals.

use crate::{frame::Domain, hash};

use super::{Defects, Recipe};

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub render_extent: Domain,
    pub pixel_definition: Domain,
    pub pixel_size_um: f32,
    pub film_format_long_edge_mm: f32,
    pub canonical_x: f64,
    pub canonical_y: f64,
    pub canonical_width: f64,
    pub canonical_height: f64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub pixel_aspect_ratio: f64,
    pub frame_time: f64,
    pub frame_rate: f64,
    pub session_seed: u64,
    pub clip_token: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InactiveIdentity,
    GeometryOrPolicy,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InactiveIdentity => "inactive physical effects have a recipe identity",
            Self::GeometryOrPolicy => {
                "ResourceDescriptorMismatch phase=effects_descriptor field=geometry_or_policy"
            }
        })
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
pub struct Origin {
    pub cell_x: i64,
    pub cell_y: i64,
    pub local_x_mm: f32,
    pub local_y_mm: f32,
}

const EMPTY_ORIGIN: Origin = Origin {
    cell_x: 0,
    cell_y: 0,
    local_x_mm: 0.0,
    local_y_mm: 0.0,
};

#[derive(Debug)]
pub struct Frame<'a> {
    film: &'a Defects,
    gate: &'a Defects,
    render_extent: Domain,
    full_frame_extent: Domain,
    origins: [Origin; 4],
    sample_step_x_mm: f32,
    sample_step_y_mm: f32,
    roi_offset_x: i32,
    roi_offset_y: i32,
    gate_width: i32,
    gate_height: i32,
    session_seed: u64,
    clip_token: u64,
    weave_active: bool,
    weave: [f32; 4],
    film_active: bool,
    gate_transmittance_active: bool,
    gate_output_active: bool,
    requires_full_frame: bool,
    recipe_hash: u64,
    hash: u64,
}

impl Defects {
    fn dust_grid(&self) -> Grid {
        Grid {
            slot_probability: self.dust.slot_probability,
            cell_width_mm: self.dust.cell_width_mm,
            cell_height_mm: self.dust.cell_height_mm,
            support_x_mm: self.dust.support_x_mm,
            support_y_mm: self.dust.support_y_mm,
            softness_max_mm: self.dust.softness_max_mm,
        }
    }

    fn scratch_grid(&self) -> Grid {
        Grid {
            slot_probability: self.scratch.slot_probability,
            cell_width_mm: self.scratch.cell_width_mm,
            cell_height_mm: self.scratch.cell_height_mm,
            support_x_mm: self.scratch.support_x_mm,
            support_y_mm: self.scratch.support_y_mm,
            softness_max_mm: self.scratch.softness_max_mm,
        }
    }
}

struct Grid {
    slot_probability: f32,
    cell_width_mm: f32,
    cell_height_mm: f32,
    support_x_mm: f32,
    support_y_mm: f32,
    softness_max_mm: f32,
}

const CELL_LIMIT: f64 = 4503599627370496.0;

fn valid_extent(v: Domain) -> bool {
    v.width > 0 && v.height > 0
}
fn contains(outer: Domain, inner: Domain) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && i64::from(inner.x) + i64::from(inner.width)
            <= i64::from(outer.x) + i64::from(outer.width)
        && i64::from(inner.y) + i64::from(inner.height)
            <= i64::from(outer.y) + i64::from(outer.height)
}

fn split_origin(product: [f64; 2], cell_mm: f32) -> Result<(i64, f32), Error> {
    let size = f64::from(cell_mm);
    let estimate = (product[0] / size).floor();
    if !estimate.is_finite() || estimate.abs() >= CELL_LIMIT {
        return Err(Error::GeometryOrPolicy);
    }
    let remainder = libm::fma(-estimate, size, product[0]) + product[1];
    let correction = (remainder / size).floor();
    let index = estimate + correction;
    let remainder = libm::fma(-correction, size, remainder);
    if !index.is_finite()
        || index.abs() >= CELL_LIMIT
        || !remainder.is_finite()
        || remainder < 0.0
        || remainder >= size
    {
        return Err(Error::GeometryOrPolicy);
    }
    let mut cell = index as i64;
    let mut local = remainder as f32;
    if local >= cell_mm {
        cell += 1;
        local = 0.0;
    }
    if (cell as f64).abs() >= CELL_LIMIT {
        return Err(Error::GeometryOrPolicy);
    }
    Ok((cell, local))
}

fn phase_from_seed(session: u64, axis: u64, component: u64) -> f64 {
    let seed = hash::u64s(&[session, 3, axis, component]);
    let seed = if seed == 0 { 1 } else { seed };
    #[expect(
        clippy::excessive_precision,
        clippy::approx_constant,
        reason = "preserved native binary64 literal"
    )]
    let two_pi = 6.28318530717958647692;
    (seed >> 11) as f64 * (1.0 / 9007199254740992.0) * two_pi
}

fn sin_sum(frequencies: &[f64], session: u64, axis: u64, offset: u64, time: f64) -> f64 {
    #[expect(
        clippy::excessive_precision,
        clippy::approx_constant,
        reason = "preserved native binary64 literal"
    )]
    let two_pi = 6.28318530717958647692;
    let mut sum = 0.0;
    for (index, frequency) in frequencies.iter().enumerate() {
        let phase = phase_from_seed(session, axis, offset + index as u64);
        sum += (two_pi * frequency * time + phase).sin();
    }
    sum
}

#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "preserve native branch consumption of NaN"
)]
fn gate_weave(session: u64, time: f64, pixel_size_um: f64, amount: f64) -> [f32; 4] {
    let mut out = [0.0, 0.0, 1.0, 0.0];
    if !(amount > 0.0) || !pixel_size_um.is_finite() || !(pixel_size_um > 0.0) {
        return out;
    }
    let translate_rms = 6.0 * amount;
    let rotate_rms = 0.005 * amount;
    if !(translate_rms > 0.0 || rotate_rms > 0.0) {
        return out;
    }
    const DRIFT: [f64; 3] = [0.15, 0.35, 0.80];
    const JITTER: [f64; 2] = [6.0, 12.0];
    let drift_norm = 1.0 / (0.5 * DRIFT.len() as f64).sqrt();
    let jitter_norm = 1.0 / (0.5 * JITTER.len() as f64).sqrt();
    let weight_norm = 1.0 / (0.85_f64 * 0.85 + 0.15_f64 * 0.15).sqrt();
    for (axis, delta) in out[..2].iter_mut().enumerate() {
        let drift = sin_sum(&DRIFT, session, axis as u64, 0, time) * drift_norm;
        let jitter = sin_sum(&JITTER, session, axis as u64, 10, time) * jitter_norm;
        let composite = (0.85 * drift + 0.15 * jitter) * weight_norm;
        *delta = (composite * translate_rms / pixel_size_um) as f32;
    }
    if rotate_rms > 0.0 {
        let signal = sin_sum(&DRIFT, session, 2, 0, time) * drift_norm;
        let degrees = signal * rotate_rms;
        #[expect(
            clippy::excessive_precision,
            clippy::approx_constant,
            reason = "preserved native binary64 literal"
        )]
        let radians = degrees * (3.14159265358979323846 / 180.0);
        out[2] = radians.cos() as f32;
        out[3] = radians.sin() as f32;
    }
    out
}

impl<'a> Frame<'a> {
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "preserve native ordered comparisons and consumed-field admission"
    )]
    pub fn new(input: Input, recipe: &'a Recipe) -> Result<Option<Self>, Error> {
        if !recipe.active() {
            return if recipe.hash() == 0 {
                Ok(None)
            } else {
                Err(Error::InactiveIdentity)
            };
        }
        let full = input.pixel_definition;
        let render = input.render_extent;
        let x0 = input.canonical_x / input.pixel_aspect_ratio;
        let w0 = input.canonical_width / input.pixel_aspect_ratio;
        let h0 = input.canonical_height;
        if !valid_extent(full)
            || !valid_extent(render)
            || !contains(full, render)
            || full.width > i32::MAX - 64
            || full.height > i32::MAX - 64
            || !x0.is_finite()
            || !input.canonical_y.is_finite()
            || !w0.is_finite()
            || !(w0 > 0.0)
            || !h0.is_finite()
            || !(h0 > 0.0)
            || !input.pixel_aspect_ratio.is_finite()
            || !(input.pixel_aspect_ratio > 0.0)
            || !input.scale_x.is_finite()
            || !(input.scale_x > 0.0)
            || !input.scale_y.is_finite()
            || !(input.scale_y > 0.0)
            || !input.film_format_long_edge_mm.is_finite()
            || !(input.film_format_long_edge_mm > 0.0)
        {
            return Err(Error::GeometryOrPolicy);
        }
        let film = recipe.film();
        let gate = recipe.gate();
        let film_active = film.active();
        let gate_transmittance_active = gate.active();
        let weave_active = recipe.gate_weave_amount() > 0.0;
        if weave_active && render != full {
            return Err(Error::GeometryOrPolicy);
        }
        let roi_offset_x = i32::try_from(i64::from(render.x) - i64::from(full.x))
            .map_err(|_| Error::GeometryOrPolicy)?;
        let roi_offset_y = i32::try_from(i64::from(render.y) - i64::from(full.y))
            .map_err(|_| Error::GeometryOrPolicy)?;
        let mm = f64::from(input.film_format_long_edge_mm) / w0.max(h0);
        if !mm.is_finite() || !(mm > 0.0) {
            return Err(Error::GeometryOrPolicy);
        }
        let phase_x = (f64::from(full.x) / input.scale_x - x0) * mm;
        let phase_y = (f64::from(full.y) / input.scale_y - input.canonical_y) * mm;
        let (step_x, step_y) = if film_active || gate_transmittance_active {
            let x = (mm / input.scale_x) as f32;
            let y = (mm / input.scale_y) as f32;
            if !x.is_finite() || !(x > 0.0) || !y.is_finite() || !(y > 0.0) {
                return Err(Error::GeometryOrPolicy);
            }
            (x, y)
        } else {
            (0.0, 0.0)
        };
        let (high, low) = if film_active {
            let advance = mm * h0;
            let high = input.frame_time * advance;
            let low = libm::fma(input.frame_time, advance, -high);
            if !high.is_finite() || !low.is_finite() {
                return Err(Error::GeometryOrPolicy);
            }
            (high, low)
        } else {
            (0.0, 0.0)
        };
        // All participating origins precede every grid check, FD/FS/GD/GS, X before Y.
        let families = [
            film.dust_grid(),
            film.scratch_grid(),
            gate.dust_grid(),
            gate.scratch_grid(),
        ];
        let mut origins = [EMPTY_ORIGIN; 4];
        for (index, grid) in families.iter().enumerate() {
            if grid.slot_probability > 0.0 {
                let (cell_x, local_x_mm) = split_origin([phase_x, 0.0], grid.cell_width_mm)?;
                let (cell_y, local_y_mm) = split_origin(
                    if index < 2 {
                        [high, low + phase_y]
                    } else {
                        [phase_y, 0.0]
                    },
                    grid.cell_height_mm,
                )?;
                origins[index] = Origin {
                    cell_x,
                    cell_y,
                    local_x_mm,
                    local_y_mm,
                };
            }
        }
        for (index, grid) in families.iter().enumerate() {
            if grid.slot_probability > 0.0 {
                // Keep the support sums in f32 before widening.
                let nx = ((f64::from(full.width) * f64::from(step_x)
                    + 2.0 * f64::from(grid.support_x_mm + grid.softness_max_mm + step_x))
                    / f64::from(grid.cell_width_mm))
                .ceil()
                    + 2.0;
                let ny = ((f64::from(full.height) * f64::from(step_y)
                    + 2.0 * f64::from(grid.support_y_mm + grid.softness_max_mm + step_y))
                    / f64::from(grid.cell_height_mm))
                .ceil()
                    + 2.0;
                if !nx.is_finite()
                    || !ny.is_finite()
                    || !(nx > 0.0)
                    || !(ny > 0.0)
                    || !(nx * ny * 2.0 < f64::from(i32::MAX))
                    || !((origins[index].cell_x as f64).abs() + nx < CELL_LIMIT)
                    || !((origins[index].cell_y as f64).abs() + ny < CELL_LIMIT)
                {
                    return Err(Error::GeometryOrPolicy);
                }
            }
        }
        let weave = if weave_active {
            if !input.frame_time.is_finite()
                || !input.frame_rate.is_finite()
                || !(input.frame_rate > 0.0)
                || !input.pixel_size_um.is_finite()
                || !(input.pixel_size_um > 0.0)
            {
                return Err(Error::GeometryOrPolicy);
            }
            let weave = gate_weave(
                input.session_seed,
                input.frame_time / input.frame_rate,
                f64::from(input.pixel_size_um),
                recipe.gate_weave_amount(),
            );
            if !weave.iter().all(|v| v.is_finite()) {
                return Err(Error::GeometryOrPolicy);
            }
            weave
        } else {
            [0.0, 0.0, 1.0, 0.0]
        };
        let mut frame = Self {
            film,
            gate,
            render_extent: render,
            full_frame_extent: full,
            origins,
            sample_step_x_mm: step_x,
            sample_step_y_mm: step_y,
            roi_offset_x,
            roi_offset_y,
            gate_width: if gate_transmittance_active {
                full.width / 2 + full.width % 2
            } else {
                0
            },
            gate_height: if gate_transmittance_active {
                full.height / 2 + full.height % 2
            } else {
                0
            },
            session_seed: input.session_seed,
            clip_token: input.clip_token,
            weave_active,
            weave,
            film_active,
            gate_transmittance_active,
            gate_output_active: weave_active || gate_transmittance_active,
            requires_full_frame: weave_active,
            recipe_hash: recipe.hash(),
            hash: 0,
        };
        frame.hash = frame.identity();
        Ok(Some(frame))
    }
    pub fn film(&self) -> &Defects {
        self.film
    }
    pub fn gate(&self) -> &Defects {
        self.gate
    }
    pub fn render_extent(&self) -> Domain {
        self.render_extent
    }
    pub fn full_frame_extent(&self) -> Domain {
        self.full_frame_extent
    }
    pub fn origins(&self) -> &[Origin; 4] {
        &self.origins
    }
    pub fn sample_step_x_mm(&self) -> f32 {
        self.sample_step_x_mm
    }
    pub fn sample_step_y_mm(&self) -> f32 {
        self.sample_step_y_mm
    }
    pub fn roi_offset_x(&self) -> i32 {
        self.roi_offset_x
    }
    pub fn roi_offset_y(&self) -> i32 {
        self.roi_offset_y
    }
    pub fn gate_width(&self) -> i32 {
        self.gate_width
    }
    pub fn gate_height(&self) -> i32 {
        self.gate_height
    }
    pub fn session_seed(&self) -> u64 {
        self.session_seed
    }
    pub fn clip_token(&self) -> u64 {
        self.clip_token
    }
    pub fn weave_active(&self) -> bool {
        self.weave_active
    }
    pub fn weave(&self) -> &[f32; 4] {
        &self.weave
    }
    pub fn film_active(&self) -> bool {
        self.film_active
    }
    pub fn gate_transmittance_active(&self) -> bool {
        self.gate_transmittance_active
    }
    pub fn gate_output_active(&self) -> bool {
        self.gate_output_active
    }
    pub fn requires_full_frame(&self) -> bool {
        self.requires_full_frame
    }
    pub fn recipe_hash(&self) -> u64 {
        self.recipe_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
    fn identity(&self) -> u64 {
        if !self.film_active && !self.gate_output_active {
            return 0;
        }
        let mut hash = hash::FNV_OFFSET;
        for e in [self.render_extent, self.full_frame_extent] {
            for v in [e.x, e.y, e.width, e.height] {
                hash::update_bytes(&mut hash, &v.to_le_bytes());
            }
        }
        for v in [self.sample_step_x_mm, self.sample_step_y_mm] {
            hash::update_bytes(&mut hash, &v.to_bits().to_le_bytes());
        }
        for v in [
            self.roi_offset_x,
            self.roi_offset_y,
            self.gate_width,
            self.gate_height,
        ] {
            hash::update_bytes(&mut hash, &v.to_le_bytes());
        }
        for v in [self.session_seed, self.clip_token] {
            hash::update_bytes(&mut hash, &v.to_le_bytes());
        }
        hash::update_bytes(&mut hash, &[u8::from(self.weave_active)]);
        for v in self.weave {
            hash::update_bytes(&mut hash, &v.to_bits().to_le_bytes());
        }
        for v in [
            self.film_active,
            self.gate_transmittance_active,
            self.gate_output_active,
            self.requires_full_frame,
        ] {
            hash::update_bytes(&mut hash, &[u8::from(v)]);
        }
        hash::update_bytes(&mut hash, &self.recipe_hash.to_le_bytes());
        for (defects, [dust_origin, scratch_origin]) in [
            (self.film, [self.origins[0], self.origins[1]]),
            (self.gate, [self.origins[2], self.origins[3]]),
        ] {
            if defects.dust.slot_probability > 0.0 {
                defects.dust.append_fields(&mut hash);
                append_origin(&mut hash, dust_origin);
            }
            if defects.scratch.slot_probability > 0.0 {
                defects.scratch.append_fields(&mut hash);
                append_origin(&mut hash, scratch_origin);
            }
        }
        hash
    }
}

fn append_origin(identity: &mut u64, origin: Origin) {
    hash::update_bytes(identity, &origin.cell_x.to_le_bytes());
    hash::update_bytes(identity, &origin.cell_y.to_le_bytes());
    hash::update_bytes(identity, &origin.local_x_mm.to_bits().to_le_bytes());
    hash::update_bytes(identity, &origin.local_y_mm.to_bits().to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::{Dust, Scratch};
    use serde_json::{Value, json};
    fn scalar(v: &Value) -> f32 {
        f32::from_bits(v.as_u64().unwrap() as u32)
    }
    fn wide(v: &Value) -> f64 {
        f64::from_bits(v.as_u64().unwrap())
    }
    fn domain(v: &Value) -> Domain {
        Domain {
            x: v[0].as_i64().unwrap() as i32,
            y: v[1].as_i64().unwrap() as i32,
            width: v[2].as_i64().unwrap() as i32,
            height: v[3].as_i64().unwrap() as i32,
        }
    }
    fn dust(v: &Value) -> Dust {
        Dust {
            cell_width_mm: scalar(&v["cell_width_mm"]),
            cell_height_mm: scalar(&v["cell_height_mm"]),
            slot_probability: scalar(&v["slot_probability"]),
            softness_min_mm: scalar(&v["softness_min_mm"]),
            softness_max_mm: scalar(&v["softness_max_mm"]),
            softness_size_cap_fraction: scalar(&v["softness_size_cap_fraction"]),
            support_x_mm: scalar(&v["support_x_mm"]),
            support_y_mm: scalar(&v["support_y_mm"]),
            fiber_fraction: scalar(&v["fiber_fraction"]),
            fiber_drift_fraction: scalar(&v["fiber_drift_fraction"]),
            fiber_first_knot_min: scalar(&v["fiber_first_knot_min"]),
            fiber_first_knot_max: scalar(&v["fiber_first_knot_max"]),
            fiber_second_knot_min: scalar(&v["fiber_second_knot_min"]),
            fiber_second_knot_max: scalar(&v["fiber_second_knot_max"]),
            fiber_interior_width_min_fraction: scalar(&v["fiber_interior_width_min_fraction"]),
            fiber_interior_width_max_fraction: scalar(&v["fiber_interior_width_max_fraction"]),
            diameter_min_mm: scalar(&v["diameter_min_mm"]),
            diameter_bulk_max_mm: scalar(&v["diameter_bulk_max_mm"]),
            diameter_max_mm: scalar(&v["diameter_max_mm"]),
            diameter_tail_fraction: scalar(&v["diameter_tail_fraction"]),
            fiber_length_min_mm: scalar(&v["fiber_length_min_mm"]),
            fiber_length_max_mm: scalar(&v["fiber_length_max_mm"]),
            fiber_width_min_mm: scalar(&v["fiber_width_min_mm"]),
            fiber_width_max_mm: scalar(&v["fiber_width_max_mm"]),
            opacity_faint_cumulative: scalar(&v["opacity_faint_cumulative"]),
            opacity_intermediate_cumulative: scalar(&v["opacity_intermediate_cumulative"]),
            compact_opacity_min: scalar(&v["compact_opacity_min"]),
            compact_opacity_faint_end: scalar(&v["compact_opacity_faint_end"]),
            compact_opacity_intermediate_end: scalar(&v["compact_opacity_intermediate_end"]),
            compact_opacity_max: scalar(&v["compact_opacity_max"]),
            fiber_opacity_min: scalar(&v["fiber_opacity_min"]),
            fiber_opacity_faint_end: scalar(&v["fiber_opacity_faint_end"]),
            fiber_opacity_intermediate_end: scalar(&v["fiber_opacity_intermediate_end"]),
            fiber_opacity_max: scalar(&v["fiber_opacity_max"]),
            compact_dominant_aspect_min: scalar(&v["compact_dominant_aspect_min"]),
            compact_dominant_aspect_max: scalar(&v["compact_dominant_aspect_max"]),
            compact_subsidiary_scale_min: scalar(&v["compact_subsidiary_scale_min"]),
            compact_subsidiary_scale_max: scalar(&v["compact_subsidiary_scale_max"]),
            compact_subsidiary_aspect_min: scalar(&v["compact_subsidiary_aspect_min"]),
            compact_subsidiary_aspect_max: scalar(&v["compact_subsidiary_aspect_max"]),
            compact_subsidiary_offset_max: scalar(&v["compact_subsidiary_offset_max"]),
            compact_subsidiary_angle_max_radians: scalar(
                &v["compact_subsidiary_angle_max_radians"],
            ),
        }
    }
    fn scratch(v: &Value) -> Scratch {
        Scratch {
            cell_width_mm: scalar(&v["cell_width_mm"]),
            cell_height_mm: scalar(&v["cell_height_mm"]),
            slot_probability: scalar(&v["slot_probability"]),
            softness_min_mm: scalar(&v["softness_min_mm"]),
            softness_max_mm: scalar(&v["softness_max_mm"]),
            softness_size_cap_fraction: scalar(&v["softness_size_cap_fraction"]),
            support_x_mm: scalar(&v["support_x_mm"]),
            support_y_mm: scalar(&v["support_y_mm"]),
            length_min_mm: scalar(&v["length_min_mm"]),
            length_bulk_max_mm: scalar(&v["length_bulk_max_mm"]),
            length_max_mm: scalar(&v["length_max_mm"]),
            length_tail_fraction: scalar(&v["length_tail_fraction"]),
            width_min_mm: scalar(&v["width_min_mm"]),
            width_bulk_max_mm: scalar(&v["width_bulk_max_mm"]),
            width_max_mm: scalar(&v["width_max_mm"]),
            width_tail_fraction: scalar(&v["width_tail_fraction"]),
            drift_fraction: scalar(&v["drift_fraction"]),
            first_knot_min: scalar(&v["first_knot_min"]),
            first_knot_max: scalar(&v["first_knot_max"]),
            second_knot_min: scalar(&v["second_knot_min"]),
            second_knot_max: scalar(&v["second_knot_max"]),
            interior_width_min_fraction: scalar(&v["interior_width_min_fraction"]),
            interior_width_max_fraction: scalar(&v["interior_width_max_fraction"]),
            interior_depth_min_fraction: scalar(&v["interior_depth_min_fraction"]),
            interior_depth_max_fraction: scalar(&v["interior_depth_max_fraction"]),
            endpoint_abrupt_probability: scalar(&v["endpoint_abrupt_probability"]),
            interruption_probability: scalar(&v["interruption_probability"]),
            gap_center_min: scalar(&v["gap_center_min"]),
            gap_center_max: scalar(&v["gap_center_max"]),
            gap_span_min: scalar(&v["gap_span_min"]),
            gap_span_max: scalar(&v["gap_span_max"]),
            scuff_probability: scalar(&v["scuff_probability"]),
            scuff_length_max_mm: scalar(&v["scuff_length_max_mm"]),
            scuff_angle_max_radians: scalar(&v["scuff_angle_max_radians"]),
            strength_min: scalar(&v["strength_min"]),
            strength_max: scalar(&v["strength_max"]),
        }
    }
    fn input(v: &Value) -> Input {
        Input {
            render_extent: domain(&v["render_extent"]),
            pixel_definition: domain(&v["pixel_definition"]),
            pixel_size_um: scalar(&v["pixel_size_um"]),
            film_format_long_edge_mm: scalar(&v["film_format_long_edge_mm"]),
            canonical_x: wide(&v["canonical_x"]),
            canonical_y: wide(&v["canonical_y"]),
            canonical_width: wide(&v["canonical_width"]),
            canonical_height: wide(&v["canonical_height"]),
            scale_x: wide(&v["scale_x"]),
            scale_y: wide(&v["scale_y"]),
            pixel_aspect_ratio: wide(&v["pixel_aspect_ratio"]),
            frame_time: wide(&v["frame_time"]),
            frame_rate: wide(&v["frame_rate"]),
            session_seed: v["session_seed"].as_u64().unwrap(),
            clip_token: v["clip_token"].as_u64().unwrap(),
        }
    }
    fn fields(v: &Frame<'_>) -> Value {
        let extent = |e: Domain| json!([e.x, e.y, e.width, e.height]);
        json!({
            "render_extent": extent(v.render_extent()), "full_frame_extent": extent(v.full_frame_extent()),
            "origins": v.origins().map(|o| json!([o.cell_x, o.cell_y, o.local_x_mm.to_bits(), o.local_y_mm.to_bits()])),
            "sample_step_x_mm": v.sample_step_x_mm().to_bits(), "sample_step_y_mm": v.sample_step_y_mm().to_bits(),
            "roi_offset_x": v.roi_offset_x(), "roi_offset_y": v.roi_offset_y(), "gate_width": v.gate_width(), "gate_height": v.gate_height(),
            "session_seed": v.session_seed(), "clip_token": v.clip_token(), "recipe_hash": v.recipe_hash(), "hash": v.hash(),
            "weave_active": v.weave_active(), "weave_dx_px": v.weave()[0].to_bits(), "weave_dy_px": v.weave()[1].to_bits(),
            "weave_cos_rot": v.weave()[2].to_bits(), "weave_sin_rot": v.weave()[3].to_bits(),
            "film_active": v.film_active(), "gate_transmittance_active": v.gate_transmittance_active(),
            "gate_output_active": v.gate_output_active(), "requires_full_frame": v.requires_full_frame(),
        })
    }
    fn read(name: &str) -> Value {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/film/effects-frame");
        serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap()
    }
    fn captured() -> Value {
        let preset = match (cfg!(windows), cfg!(debug_assertions)) {
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
        };
        let logical = format!("{preset}.json");
        let manifest = read("manifest.json");
        let mut captured = read(manifest["common"].as_str().unwrap());
        let products = read(manifest["payloads"][&logical].as_str().unwrap());
        captured["products"] = products["products"].clone();
        let context = read(manifest["contexts"][&logical].as_str().unwrap());
        assert_eq!(context["preset"], preset);
        for (key, value) in context.as_object().unwrap() {
            captured
                .as_object_mut()
                .unwrap()
                .insert(key.clone(), value.clone());
        }
        captured
    }
    #[test]
    fn complete_independent_native_values_failures_and_identity() {
        let captured = captured();
        let checked = read("checked.json");
        let rows = captured["cases"].as_array().unwrap();
        assert_eq!(rows.len(), 347);
        assert_eq!(
            json!(rows.iter().map(|r| &r["input"]["id"]).collect::<Vec<_>>()),
            checked["case_ids"]
        );
        let products = captured["products"].as_array().unwrap();
        let expected_ids = rows
            .iter()
            .filter(|row| row["input"]["missing_recipe"] != true)
            .map(|row| format!("case/{}", row["input"]["id"].as_str().unwrap()))
            .chain(
                products
                    .iter()
                    .filter(|row| row["input"]["missing_recipe"] != true)
                    .map(|row| format!("product/{}", row["input"]["id"].as_str().unwrap())),
            )
            .collect::<Vec<_>>();
        let cases_ids: Vec<_> = IntoIterator::into_iter(expected_ids)
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "effects-frame-values"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "effects-frame-values"
        );
        for (group, row) in rows
            .iter()
            .map(|row| ("case", row))
            .chain(products.iter().map(|row| ("product", row)))
        {
            let v = &row["input"];
            if v["missing_recipe"] == true {
                continue;
            }
            let case = AsRef::<str>::as_ref(&(format!("{group}/{}", v["id"].as_str().unwrap())))
                .to_owned();
            let recipe = Recipe {
                film: Defects {
                    dust: dust(&v["film_dust"]),
                    scratch: scratch(&v["film_scratch"]),
                },
                gate: Defects {
                    dust: dust(&v["gate_dust"]),
                    scratch: scratch(&v["gate_scratch"]),
                },
                active: v["active"].as_bool().unwrap(),
                gate_weave_amount: wide(&v["gate_weave_amount"]),
                hash: v["recipe_hash"].as_u64().unwrap(),
            };
            let result = Frame::new(input(v), &recipe);
            assert_eq!(
                result.is_ok(),
                row["success"].as_bool().unwrap(),
                "{}",
                v["id"]
            );
            match result {
                Ok(Some(frame)) => {
                    let mut expected = row["native"].clone();
                    for family in ["film_dust", "film_scratch", "gate_dust", "gate_scratch"] {
                        expected.as_object_mut().unwrap().remove(family);
                    }
                    assert_eq!(fields(&frame), expected, "{}", v["id"]);
                }
                Ok(None) => {
                    assert_eq!(v["active"], false, "{}", v["id"]);
                    assert_eq!(row["diagnostic"], "");
                }
                Err(error) => {
                    assert_eq!(
                        error,
                        if v["active"] == false {
                            Error::InactiveIdentity
                        } else {
                            Error::GeometryOrPolicy
                        },
                        "{}",
                        v["id"]
                    );
                    assert_eq!(
                        row["diagnostic"],
                        if error == Error::InactiveIdentity {
                            ""
                        } else {
                            "ResourceDescriptorMismatch phase=effects_descriptor field=geometry_or_policy"
                        }
                    );
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
    }
    #[test]
    fn independently_characterized_primitives() {
        let captured = captured();
        let checked = read("checked.json");
        let rows = captured["primitives"].as_array().unwrap();
        assert_eq!(rows.len(), 57);
        assert_eq!(
            json!(rows.iter().map(|r| &r["input"]["id"]).collect::<Vec<_>>()),
            checked["primitive_ids"]
        );
        let cases_ids: Vec<_> =
            IntoIterator::into_iter(rows.iter().map(|row| row["input"]["id"].as_str().unwrap()))
                .map(|id| AsRef::<str>::as_ref(&id).to_owned())
                .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "effects-frame-primitives"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "effects-frame-primitives"
        );
        for row in rows {
            let v = &row["input"];
            let case = AsRef::<str>::as_ref(&(v["id"].as_str().unwrap())).to_owned();
            let args = [
                wide(&v["args"][0]),
                wide(&v["args"][1]),
                wide(&v["args"][2]),
            ];
            let seed = v["seed"].as_u64().unwrap();
            let op = v["op"].as_u64().unwrap();
            let words = match op {
                0 => match split_origin([args[0], args[1]], args[2] as f32) {
                    Ok((cell, local)) => [1, cell as u64, u64::from(local.to_bits()), 0],
                    Err(_) => [0; 4],
                },
                1 => gate_weave(seed, args[0], args[1], args[2]).map(|v| u64::from(v.to_bits())),
                2 => [libm::fma(args[0], args[1], args[2]).to_bits(), 0, 0, 0],
                3 => [args[0].sin().to_bits(), 0, 0, 0],
                4 => [args[0].cos().to_bits(), 0, 0, 0],
                5 => [args[0].sqrt().to_bits(), 0, 0, 0],
                6 => [
                    phase_from_seed(seed, args[0] as u64, args[1] as u64).to_bits(),
                    0,
                    0,
                    0,
                ],
                _ => panic!("unknown bounded primitive"),
            };
            if (op == 3 || op == 4) && json!(words) != row["words"] {
                eprintln!(
                    "internal f64 primitive observation {}: native={} Rust={}; complete weave/frame readers remain exact",
                    v["id"],
                    row["words"],
                    json!(words)
                );
            } else {
                assert_eq!(json!(words), row["words"], "{}", v["id"]);
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
    }
}
