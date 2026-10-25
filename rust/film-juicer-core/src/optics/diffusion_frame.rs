//! Callback-local diffusion support, stage identities and nine-component PSFs.

use crate::hash;

#[derive(Clone, Copy, Debug)]
pub struct SampleInput {
    pub family: u32,
    pub group_weights: [f64; 3],
    pub group_lambda_um: [f64; 3],
    pub warmth: f64,
    pub spatial_scale: f64,
    pub pixel_size_um: f64,
    pub width: i32,
    pub height: i32,
}

/// Copied boundary fields; sample construction does not establish component readiness.
#[derive(Clone, Copy, Debug)]
pub struct SampleFields {
    pub family: u32,
    pub group_weights: [f64; 3],
    pub group_lambda_um: [f64; 3],
    pub warmth: f64,
    pub spatial_scale: f64,
    pub pixel_size_um: f64,
    pub radius_px: i32,
    pub hash: u64,
}

#[derive(Debug)]
pub struct Sample {
    fields: SampleFields,
}

#[derive(Debug)]
pub struct Components {
    lambda_px: [f64; 9],
    red_weights: [f64; 9],
    green_weights: [f64; 9],
    blue_weights: [f64; 9],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Field {
    PixelSizeUm = 1,
    FullFrameExtent = 2,
    Family = 3,
    RadiusPixels = 4,
    SampleDescriptorHash = 5,
    SampleDescriptorInactive = 6,
    GroupWeights = 7,
    GroupCentersUm = 8,
    EffectiveWarmth = 9,
    SpatialScale = 10,
    ComponentExpansion = 11,
    ComponentCount = 12,
    HaloChannelWeights = 13,
    ComponentOutput = 14,
    Route = 15,
    FullFrameDomain = 16,
    EnlargerRoute = 17,
    LinearDomain = 18,
    StageSample = 19,
    StageHash = 20,
    FrameSetHash = 21,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Error(ErrorKind);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErrorKind {
    Sample(Field),
    Frame { stage: u32, field: Field },
}

impl Error {
    fn sample(field: Field) -> Self {
        Self(ErrorKind::Sample(field))
    }
    fn frame(stage: u32, field: Field) -> Self {
        Self(ErrorKind::Frame { stage, field })
    }
    pub fn field(self) -> Field {
        match self.0 {
            ErrorKind::Sample(field) | ErrorKind::Frame { field, .. } => field,
        }
    }

    pub fn stage(self) -> u32 {
        match self.0 {
            ErrorKind::Sample(_) => 0,
            ErrorKind::Frame { stage, .. } => stage,
        }
    }

    pub fn diagnostic(self) -> &'static str {
        match self.0 {
            ErrorKind::Sample(field) => match field {
                Field::PixelSizeUm => "InvalidDiffusionHostBehavior field=pixel_size_um",
                Field::FullFrameExtent => "InvalidDiffusionHostBehavior field=full_frame_extent",
                Field::Family => "InvalidDiffusionHostBehavior field=family",
                Field::RadiusPixels => "InvalidDiffusionHostBehavior field=radius_pixels",
                Field::SampleDescriptorHash => {
                    "InvalidDiffusionHostBehavior field=sample_descriptor_hash"
                }
                Field::SampleDescriptorInactive => {
                    "InvalidDiffusionHostBehavior field=sample_descriptor_inactive"
                }
                Field::GroupWeights => "InvalidDiffusionHostBehavior field=group_weights",
                Field::GroupCentersUm => "InvalidDiffusionHostBehavior field=group_centers_um",
                Field::EffectiveWarmth => "InvalidDiffusionHostBehavior field=effective_warmth",
                Field::SpatialScale => "InvalidDiffusionHostBehavior field=spatial_scale",
                Field::ComponentExpansion => {
                    "InvalidDiffusionHostBehavior field=component_expansion"
                }
                Field::ComponentCount => "InvalidDiffusionHostBehavior field=component_count",
                Field::HaloChannelWeights => {
                    "InvalidDiffusionHostBehavior field=halo_channel_weights"
                }
                Field::ComponentOutput => "InvalidDiffusionHostBehavior field=component_output",
                _ => unreachable!("sample error has a sample field"),
            },
            ErrorKind::Frame { stage, field } => match field {
                Field::PixelSizeUm => "InvalidDiffusionFrameDescriptor field=pixel_size_um",
                Field::Route => "InvalidDiffusionFrameDescriptor field=route",
                Field::FullFrameDomain => "InvalidDiffusionFrameDescriptor field=full_frame_domain",
                Field::EnlargerRoute => "InvalidDiffusionFrameDescriptor field=enlarger_route",
                Field::LinearDomain if stage == 1 => {
                    "InvalidDiffusionFrameDescriptor field=camera_linear_domain"
                }
                Field::LinearDomain => {
                    "InvalidDiffusionFrameDescriptor field=enlarger_linear_domain"
                }
                Field::StageSample if stage == 1 => {
                    "InvalidDiffusionFrameDescriptor field=camera_sample"
                }
                Field::StageSample => "InvalidDiffusionFrameDescriptor field=enlarger_sample",
                Field::StageHash if stage == 1 => {
                    "InvalidDiffusionFrameDescriptor field=camera_hash"
                }
                Field::StageHash => "InvalidDiffusionFrameDescriptor field=enlarger_hash",
                Field::FrameSetHash => "InvalidDiffusionFrameDescriptor field=frame_set_hash",
                _ => unreachable!("frame error has a frame field"),
            },
        }
    }
}

fn ordered_max(value: f64, floor: f64) -> f64 {
    if value < floor { floor } else { value }
}

fn hash_double(hash: &mut u64, value: f64) {
    hash::update_bytes(
        hash,
        &if value == 0.0 { 0 } else { value.to_bits() }.to_le_bytes(),
    );
}

fn sample_hash(fields: &SampleFields) -> u64 {
    let mut result = hash::FNV_OFFSET;
    hash::update_bytes(&mut result, b"diffusion-psf-sample\0");
    hash::update_bytes(&mut result, &1_u32.to_le_bytes());
    hash::update_bytes(&mut result, &[fields.family as u8]);
    for value in fields
        .group_weights
        .into_iter()
        .chain(fields.group_lambda_um)
    {
        hash_double(&mut result, value);
    }
    for value in [fields.warmth, fields.spatial_scale, fields.pixel_size_um] {
        hash_double(&mut result, value);
    }
    hash::update_bytes(&mut result, &fields.radius_px.to_le_bytes());
    result
}

impl Sample {
    pub fn new(input: SampleInput) -> Result<Self, Error> {
        if !(input.pixel_size_um.is_finite() && input.pixel_size_um > 0.0) {
            return Err(Error::sample(Field::PixelSizeUm));
        }
        if input.width <= 0 || input.height <= 0 {
            return Err(Error::sample(Field::FullFrameExtent));
        }
        if input.family > 3 {
            return Err(Error::sample(Field::Family));
        }
        let bloom_maximum_lambda_um = input.group_lambda_um[2] * 2.5;
        let requested_radius = ordered_max(
            8.0 * bloom_maximum_lambda_um * input.spatial_scale / input.pixel_size_um,
            5.0,
        )
        .ceil();
        if !(requested_radius.is_finite() && requested_radius > 0.0) {
            return Err(Error::sample(Field::RadiusPixels));
        }
        let cap = (input.width.min(input.height) / 2 - 1).max(1);
        let radius_px = if requested_radius >= f64::from(cap) {
            cap
        } else {
            requested_radius as i32
        };
        if radius_px <= 0 {
            return Err(Error::sample(Field::RadiusPixels));
        }
        let mut fields = SampleFields {
            family: input.family,
            group_weights: input.group_weights,
            group_lambda_um: input.group_lambda_um,
            warmth: input.warmth,
            spatial_scale: input.spatial_scale,
            pixel_size_um: input.pixel_size_um,
            radius_px,
            hash: 0,
        };
        fields.hash = sample_hash(&fields);
        if fields.hash == 0 {
            return Err(Error::sample(Field::SampleDescriptorHash));
        }
        Ok(Self { fields })
    }

    pub fn fields(&self) -> SampleFields {
        self.fields
    }
}

impl From<&Sample> for SampleFields {
    fn from(sample: &Sample) -> Self {
        sample.fields
    }
}

fn admit_components(fields: &SampleFields) -> Result<(), Error> {
    let reject = |field| Err(Error::sample(field));
    if fields.hash == 0 {
        return reject(Field::SampleDescriptorInactive);
    }
    if fields.family > 3 {
        return reject(Field::Family);
    }
    let mut total = 0.0;
    for weight in fields.group_weights {
        if !weight.is_finite() || weight < 0.0 {
            return reject(Field::GroupWeights);
        }
        total += weight;
    }
    if !(total.is_finite() && total > 0.0) {
        return reject(Field::GroupWeights);
    }
    for center in fields.group_lambda_um {
        if !(center.is_finite() && center > 0.0) {
            return reject(Field::GroupCentersUm);
        }
    }
    if !fields.warmth.is_finite() || fields.warmth < -1.5 || fields.warmth > 1.5 {
        return reject(Field::EffectiveWarmth);
    }
    if !(fields.spatial_scale.is_finite() && fields.spatial_scale > 0.0) {
        return reject(Field::SpatialScale);
    }
    if !(fields.pixel_size_um.is_finite() && fields.pixel_size_um > 0.0) {
        return reject(Field::PixelSizeUm);
    }
    if fields.radius_px <= 0 {
        return reject(Field::RadiusPixels);
    }
    if sample_hash(fields) != fields.hash {
        return reject(Field::SampleDescriptorHash);
    }
    Ok(())
}

struct Group {
    lambda_um: [f64; 4],
    weights: [f64; 4],
    count: usize,
}

fn expand_group(
    center: f64,
    spread: f64,
    count: usize,
    bloom_alpha: Option<f64>,
) -> Result<Group, Error> {
    let mut out = Group {
        lambda_um: [0.0; 4],
        weights: [0.0; 4],
        count,
    };
    let log_low = (center / spread).ln();
    let log_high = (center * spread).ln();
    if !(log_low.is_finite() && log_high.is_finite()) {
        return Err(Error::sample(Field::ComponentExpansion));
    }
    for index in 0..count {
        let log_lambda = if index + 1 == count {
            log_high
        } else if index != 0 {
            let t = index as f64 / (count - 1) as f64;
            log_low + (log_high - log_low) * t
        } else {
            log_low
        };
        out.lambda_um[index] = log_lambda.exp();
        out.weights[index] = match bloom_alpha {
            Some(alpha) => out.lambda_um[index].powf(2.0 - alpha),
            None => 1.0,
        };
        if !(out.lambda_um[index].is_finite()
            && out.lambda_um[index] > 0.0
            && out.weights[index].is_finite()
            && out.weights[index] >= 0.0)
        {
            return Err(Error::sample(Field::ComponentExpansion));
        }
    }
    let mut total = 0.0;
    for index in 0..count {
        total += out.weights[index];
    }
    if !(total.is_finite() && total > 0.0) {
        return Err(Error::sample(Field::ComponentExpansion));
    }
    for index in 0..count {
        out.weights[index] /= total;
    }
    Ok(out)
}

fn halo_weights(halo: &Group, warmth: f64) -> Result<[[f64; 4]; 3], Error> {
    let mut gradient = [0.0; 4];
    let mut target_total = 0.0;
    let mut weighted_gradient = 0.0;
    for (index, entry) in gradient.iter_mut().enumerate().take(halo.count) {
        *entry = -1.0 + 2.0 * index as f64 / (halo.count - 1) as f64;
        target_total += halo.weights[index];
        weighted_gradient += halo.weights[index] * *entry;
    }
    if !(target_total.is_finite() && target_total > 0.0) {
        return Err(Error::sample(Field::HaloChannelWeights));
    }
    weighted_gradient /= target_total;
    for entry in gradient.iter_mut().take(halo.count) {
        *entry -= weighted_gradient;
    }
    let mut out = [[0.0; 4]; 3];
    for (channel, axis) in [1.30, 0.15, -1.45].into_iter().enumerate() {
        let mut total = 0.0;
        for (index, &g) in gradient.iter().enumerate().take(halo.count) {
            out[channel][index] = ordered_max(halo.weights[index] * (1.0 + warmth * axis * g), 0.0);
            total += out[channel][index];
        }
        if total > 0.0 {
            let scale = target_total / total;
            for value in out[channel].iter_mut().take(halo.count) {
                *value *= scale;
            }
        } else {
            out[channel] = halo.weights;
        }
    }
    Ok(out)
}

impl Components {
    pub fn new(fields: SampleFields) -> Result<Self, Error> {
        admit_components(&fields)?;
        let core = expand_group(fields.group_lambda_um[0], 1.5, 2, None)?;
        let halo = expand_group(fields.group_lambda_um[1], 2.0, 3, None)?;
        let alpha = [3.2, 3.5, 2.9, 2.5][fields.family as usize];
        let bloom = expand_group(fields.group_lambda_um[2], 2.5, 4, Some(alpha))?;
        if core.count != 2 || halo.count != 3 || bloom.count != 4 {
            return Err(Error::sample(Field::ComponentCount));
        }
        let halo_rgb = halo_weights(&halo, fields.warmth)?;
        let scale = ordered_max(fields.spatial_scale, 1e-6);
        let mut out = Self {
            lambda_px: [0.0; 9],
            red_weights: [0.0; 9],
            green_weights: [0.0; 9],
            blue_weights: [0.0; 9],
        };
        let mut offset = 0;
        for (group_index, group) in [&core, &halo, &bloom].into_iter().enumerate() {
            for (index, &lambda_um) in group.lambda_um.iter().take(group.count).enumerate() {
                let output = offset + index;
                out.lambda_px[output] = ordered_max(lambda_um * scale / fields.pixel_size_um, 1e-6);
                if group_index == 1 {
                    out.red_weights[output] = fields.group_weights[1] * halo_rgb[0][index];
                    out.green_weights[output] = fields.group_weights[1] * halo_rgb[1][index];
                    out.blue_weights[output] = fields.group_weights[1] * halo_rgb[2][index];
                } else {
                    let weight = fields.group_weights[group_index] * group.weights[index];
                    out.red_weights[output] = weight;
                    out.green_weights[output] = weight;
                    out.blue_weights[output] = weight;
                }
            }
            offset += group.count;
        }
        for index in 0..9 {
            if !(out.lambda_px[index].is_finite() && out.lambda_px[index] > 0.0)
                || [
                    out.red_weights[index],
                    out.green_weights[index],
                    out.blue_weights[index],
                ]
                .iter()
                .any(|w| !w.is_finite() || *w < 0.0)
            {
                return Err(Error::sample(Field::ComponentOutput));
            }
        }
        Ok(out)
    }

    pub fn lambda_px(&self) -> &[f64; 9] {
        &self.lambda_px
    }
    pub fn red_weights(&self) -> &[f64; 9] {
        &self.red_weights
    }
    pub fn green_weights(&self) -> &[f64; 9] {
        &self.green_weights
    }
    pub fn blue_weights(&self) -> &[f64; 9] {
        &self.blue_weights
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StageInput {
    pub domain: u32,
    pub family: u32,
    pub group_weights: [f64; 3],
    pub group_lambda_um: [f64; 3],
    pub warmth: f64,
    pub spatial_scale: f64,
    pub scatter_fraction: f64,
    pub hash: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameInput {
    pub route: u32,
    pub full_frame: [i32; 4],
    pub pixel_size_um: f64,
    pub camera: StageInput,
    pub enlarger: StageInput,
}

#[derive(Debug)]
pub struct Stage {
    stage: u32,
    scatter_fraction: f64,
    sample: Sample,
    hash: u64,
}

impl Stage {
    fn new(input: StageInput, frame: &FrameInput, stage: u32) -> Result<Self, Error> {
        let reject = |field| Error::frame(stage + 1, field);
        if input.domain != stage {
            return Err(reject(Field::LinearDomain));
        }
        let sample = Sample::new(SampleInput {
            family: input.family,
            group_weights: input.group_weights,
            group_lambda_um: input.group_lambda_um,
            warmth: input.warmth,
            spatial_scale: input.spatial_scale,
            pixel_size_um: frame.pixel_size_um,
            width: frame.full_frame[2],
            height: frame.full_frame[3],
        })
        .map_err(|_| reject(Field::StageSample))?;
        let mut hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut hash, b"diffusion-stage-frame\0");
        hash::update_bytes(&mut hash, &2_u32.to_le_bytes());
        hash::update_bytes(&mut hash, &[stage as u8]);
        hash_double(&mut hash, input.scatter_fraction);
        hash::update_bytes(&mut hash, &sample.fields.hash.to_le_bytes());
        if hash == 0 {
            return Err(reject(Field::StageHash));
        }
        Ok(Self {
            stage,
            scatter_fraction: input.scatter_fraction,
            sample,
            hash,
        })
    }
    pub fn stage(&self) -> u32 {
        self.stage
    }
    pub fn scatter_fraction(&self) -> f64 {
        self.scatter_fraction
    }
    pub fn sample(&self) -> &Sample {
        &self.sample
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Debug)]
pub struct FrameSet {
    route: u32,
    full_frame: [i32; 4],
    camera: Option<Stage>,
    enlarger: Option<Stage>,
    hash: u64,
}

impl FrameSet {
    pub fn new(input: FrameInput) -> Result<Option<Self>, Error> {
        let camera_active = input.camera.hash != 0;
        let enlarger_active = input.enlarger.hash != 0;
        if !camera_active && !enlarger_active {
            return Ok(None);
        }
        let reject = |field| Error::frame(0, field);
        if input.route > 3 {
            return Err(reject(Field::Route));
        }
        if input.full_frame[2] < 2 || input.full_frame[3] < 2 {
            return Err(reject(Field::FullFrameDomain));
        }
        if !(input.pixel_size_um.is_finite() && input.pixel_size_um > 0.0) {
            return Err(reject(Field::PixelSizeUm));
        }
        if enlarger_active && input.route.is_multiple_of(2) {
            return Err(reject(Field::EnlargerRoute));
        }
        let camera = if camera_active {
            Some(Stage::new(input.camera, &input, 0)?)
        } else {
            None
        };
        let enlarger = if enlarger_active {
            Some(Stage::new(input.enlarger, &input, 1)?)
        } else {
            None
        };
        let mut hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut hash, b"diffusion-frame-set\0");
        hash::update_bytes(&mut hash, &2_u32.to_le_bytes());
        hash::update_bytes(&mut hash, &[input.route as u8]);
        for value in input.full_frame {
            hash::update_bytes(&mut hash, &value.to_le_bytes());
        }
        for stage in [&camera, &enlarger] {
            hash::update_bytes(&mut hash, &[u8::from(stage.is_some())]);
            if let Some(stage) = stage {
                hash::update_bytes(&mut hash, &stage.hash.to_le_bytes());
            }
        }
        if hash == 0 {
            return Err(reject(Field::FrameSetHash));
        }
        Ok(Some(Self {
            route: input.route,
            full_frame: input.full_frame,
            camera,
            enlarger,
            hash,
        }))
    }
    pub fn route(&self) -> u32 {
        self.route
    }
    pub fn full_frame(&self) -> [i32; 4] {
        self.full_frame
    }
    pub fn camera(&self) -> Option<&Stage> {
        self.camera.as_ref()
    }
    pub fn enlarger(&self) -> Option<&Stage> {
        self.enlarger.as_ref()
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn scalar(value: &Value) -> f64 {
        f64::from_bits(value.as_u64().unwrap())
    }
    fn doubles<const N: usize>(value: &Value) -> [f64; N] {
        assert_eq!(value.as_array().unwrap().len(), N);
        std::array::from_fn(|i| scalar(&value[i]))
    }
    fn sample_input(input: &Value) -> SampleInput {
        SampleInput {
            family: input["family"].as_u64().unwrap() as u32,
            group_weights: doubles(&input["weights"]),
            group_lambda_um: doubles(&input["centers"]),
            warmth: scalar(&input["warmth"]),
            spatial_scale: scalar(&input["scale"]),
            pixel_size_um: scalar(&input["pixel"]),
            width: input["width"].as_i64().unwrap() as i32,
            height: input["height"].as_i64().unwrap() as i32,
        }
    }
    fn stage_input(input: &Value) -> StageInput {
        StageInput {
            domain: input["domain"].as_u64().unwrap() as u32,
            family: input["family"].as_u64().unwrap() as u32,
            group_weights: doubles(&input["weights"]),
            group_lambda_um: doubles(&input["centers"]),
            warmth: scalar(&input["warmth"]),
            spatial_scale: scalar(&input["scale"]),
            scatter_fraction: scalar(&input["scatter"]),
            hash: input["hash"].as_u64().unwrap(),
        }
    }
    fn sample_json(fields: SampleFields) -> Value {
        json!({"family": fields.family, "weights": fields.group_weights.map(f64::to_bits),
            "centers": fields.group_lambda_um.map(f64::to_bits), "warmth": fields.warmth.to_bits(),
            "scale": fields.spatial_scale.to_bits(), "pixel": fields.pixel_size_um.to_bits(),
            "radius": fields.radius_px, "hash": fields.hash})
    }
    fn components_json(c: &Components) -> Value {
        json!({"lambda": c.lambda_px.map(f64::to_bits), "red": c.red_weights.map(f64::to_bits),
            "green": c.green_weights.map(f64::to_bits), "blue": c.blue_weights.map(f64::to_bits)})
    }
    fn stage_json(stage: Option<&Stage>) -> Value {
        match stage {
            None => Value::Null,
            Some(s) => json!({"stage": s.stage, "scatter": s.scatter_fraction.to_bits(),
                "sample": sample_json(s.sample.fields()), "hash": s.hash}),
        }
    }
    fn frame_json(frame: Option<&FrameSet>) -> Value {
        match frame {
            None => Value::Null,
            Some(f) => json!({"route": f.route, "extent": f.full_frame,
                "camera": stage_json(f.camera()), "enlarger": stage_json(f.enlarger()), "hash": f.hash}),
        }
    }

    #[test]
    fn native_capture() {
        let captured: Value = {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/diffusion/frame-fixtures");
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
            let mut captured = read(manifest["payloads"][logical].as_str().unwrap());
            let edge_logical = manifest["presets"][preset]["edges"].as_str().unwrap();
            assert_eq!(
                read(manifest["contexts"][edge_logical].as_str().unwrap())["preset"],
                preset
            );
            let edges = read(manifest["payloads"][edge_logical].as_str().unwrap());
            captured["samples"]
                .as_array_mut()
                .unwrap()
                .extend(edges["samples"].as_array().unwrap().iter().cloned());
            captured
        };
        let mut mismatches = Vec::new();
        let mut compare = |label: String, actual: Value, expected: &Value| {
            if &actual != expected {
                mismatches.push(format!("{label}: actual={actual} expected={expected}"));
            }
        };
        let samples: Vec<&Value> = captured["samples"]
            .as_array()
            .unwrap()
            .iter()
            .chain(
                captured["products"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row["samples"].as_array().unwrap()),
            )
            .collect();
        for row in samples {
            let input = &row["input"];
            let label = input["name"].as_str().unwrap();
            match Sample::new(sample_input(input)) {
                Ok(sample) => {
                    compare(format!("{label}/success"), json!(true), &row["success"]);
                    compare(
                        format!("{label}/sample"),
                        sample_json(sample.fields()),
                        &row["sample"],
                    );
                    match Components::new((&sample).into()) {
                        Ok(components) => {
                            compare(format!("{label}/expanded"), json!(true), &row["expanded"]);
                            compare(
                                format!("{label}/components"),
                                components_json(&components),
                                &row["components"],
                            );
                        }
                        Err(error) => {
                            compare(format!("{label}/expanded"), json!(false), &row["expanded"]);
                            compare(
                                format!("{label}/component_diagnostic"),
                                json!(error.diagnostic()),
                                &row["component_diagnostic"],
                            );
                            compare(
                                format!("{label}/cleared_components"),
                                json!({"lambda":vec![0; 9],"red":vec![0; 9],"green":vec![0; 9],"blue":vec![0; 9]}),
                                &row["components"],
                            );
                        }
                    }
                }
                Err(error) => {
                    compare(format!("{label}/success"), json!(false), &row["success"]);
                    compare(
                        format!("{label}/diagnostic"),
                        json!(error.diagnostic()),
                        &row["diagnostic"],
                    );
                    compare(
                        format!("{label}/native_default"),
                        json!({"family":1,"weights":vec![0; 3],"centers":vec![0; 3],"warmth":0,"scale":0,"pixel":0,"radius":0,"hash":0}),
                        &row["sample"],
                    );
                }
            }
        }
        for row in captured["frames"]
            .as_array()
            .unwrap()
            .iter()
            .chain(captured["products"].as_array().unwrap())
        {
            let input = &row["input"];
            let label = input["name"].as_str().unwrap();
            let result = FrameSet::new(FrameInput {
                route: input["route"].as_u64().unwrap() as u32,
                full_frame: std::array::from_fn(|i| input["extent"][i].as_i64().unwrap() as i32),
                pixel_size_um: scalar(&input["pixel"]),
                camera: stage_input(&input["camera"]),
                enlarger: stage_input(&input["enlarger"]),
            });
            compare(
                format!("{label}/success"),
                json!(result.is_ok()),
                &row["success"],
            );
            let (frame, diagnostic) = match &result {
                Ok(f) => (frame_json(f.as_ref()), ""),
                Err(e) => (Value::Null, e.diagnostic()),
            };
            compare(format!("{label}/frame"), frame, &row["frame"]);
            compare(
                format!("{label}/diagnostic"),
                json!(diagnostic),
                &row["diagnostic"],
            );
        }
        for row in captured["primitives"].as_array().unwrap() {
            let value = scalar(&row["input"]["value"]);
            let exponent = scalar(&row["input"]["exponent"]);
            for (name, value) in [
                ("log", value.ln()),
                ("exp", value.exp()),
                ("pow", value.powf(exponent)),
            ] {
                compare(
                    format!("primitive/{name}/{}", row["input"]["value"]),
                    json!(value.to_bits()),
                    &row[name],
                );
            }
        }
        assert_eq!(captured["samples"].as_array().unwrap().len(), 96);
        assert_eq!(captured["frames"].as_array().unwrap().len(), 35);
        assert_eq!(captured["primitives"].as_array().unwrap().len(), 8);
        assert_eq!(captured["products"].as_array().unwrap().len(), 92);
        assert!(
            mismatches.is_empty(),
            "{} exact differences:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }
}
