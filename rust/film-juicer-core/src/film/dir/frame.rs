//! Current-frame DIR mathematics. Workspace policy and the final descriptor
//! identity remain with the native composer, including ordinary failed progress.

use crate::frame::Domain;

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub active: bool,
    pub diffusion_size_um: f32,
    pub diffusion_tail_um: f32,
    pub diffusion_tail_weight: f32,
    pub pixel_size_um: f32,
    pub render_extent: Domain,
    pub full_frame_extent: Domain,
    pub recipe_hash: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operator {
    None,
    Identity,
    FirReflect,
    YvvReflect,
}

#[derive(Clone, Copy, Debug)]
pub struct Component {
    sigma_px: f32,
    weight: f32,
    radius: i32,
    operator: Operator,
    // feedforward, feedback[3], D, N, feedback numerators[3].
    coefficients: [f64; 9],
}

impl Component {
    const EMPTY: Self = Self {
        sigma_px: 0.0,
        weight: 0.0,
        radius: 0,
        operator: Operator::None,
        coefficients: [0.0; 9],
    };

    pub fn sigma_px(&self) -> f32 {
        self.sigma_px
    }
    pub fn weight(&self) -> f32 {
        self.weight
    }
    pub fn radius(&self) -> i32 {
        self.radius
    }
    pub fn operator(&self) -> Operator {
        self.operator
    }
    pub fn coefficients(&self) -> &[f64; 9] {
        &self.coefficients
    }
    pub fn boundary_accuracy(&self) -> f64 {
        1.0e-11
    }
    pub fn boundary_tolerance(&self) -> f64 {
        5.0e-7
    }
    pub fn boundary_version(&self) -> u32 {
        1
    }

    fn build_iir(&mut self) -> Result<(), Field> {
        let q = 0.98711 * f64::from(self.sigma_px) - 0.96330;
        let q2 = q * q;
        let q3 = q2 * q;
        let denominator = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        self.coefficients[4] = denominator;
        self.coefficients[5] = 1.57825;
        self.coefficients[6] = 2.44413 * q + 2.0 * 1.4281 * q2 + 3.0 * 0.422205 * q3;
        self.coefficients[7] = -(1.4281 * q2 + 3.0 * 0.422205 * q3);
        self.coefficients[8] = 0.422205 * q3;
        if !(denominator > 0.0 && denominator.is_finite()) {
            return Err(Field::IirDenominator);
        }
        self.coefficients[0] = self.coefficients[5] / denominator;
        for i in 0..3 {
            self.coefficients[i + 1] = self.coefficients[i + 6] / denominator;
        }
        let dc = self.coefficients[0]
            + self.coefficients[1]
            + self.coefficients[2]
            + self.coefficients[3];
        if !(self.coefficients[0].is_finite()
            && self.coefficients[0] > 0.0
            && self.coefficients[1..4].iter().all(|v| v.is_finite())
            && (dc - 1.0).abs() <= 32.0 * f64::EPSILON)
        {
            return Err(Field::IirOutput);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Field {
    PixelSizeUm = 1,
    RenderExtent = 2,
    FullFrameExtent = 3,
    PartialRenderWindow = 4,
    Core = 5,
    Tail = 6,
    ComponentValues = 7,
    ComponentCapacity = 8,
    Radius = 9,
    IirDenominator = 10,
    IirOutput = 11,
    EmptyPlan = 12,
}

impl Field {
    pub fn name(self) -> &'static str {
        match self {
            Self::PixelSizeUm => "pixel_size_um",
            Self::RenderExtent => "render_extent",
            Self::FullFrameExtent => "full_frame_extent",
            Self::PartialRenderWindow => "partial_render_window",
            Self::Core => "core",
            Self::Tail => "tail",
            Self::ComponentValues => "component_values",
            Self::ComponentCapacity => "component_capacity",
            Self::Radius => "radius",
            Self::IirDenominator => "iir_denominator",
            Self::IirOutput => "iir_output",
            Self::EmptyPlan => "empty_plan",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Progress {
    recipe_hash: u64,
    render_extent: Domain,
    full_frame_extent: Domain,
    component_count: usize,
    components: [Component; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    progress: Progress,
}

#[derive(Clone, Copy, Debug)]
pub struct FailureTrace {
    progress: Progress,
}

#[derive(Clone, Copy, Debug)]
pub struct Error {
    field: Field,
    component: usize,
    trace: Option<FailureTrace>,
}

impl Error {
    pub fn field(&self) -> Field {
        self.field
    }
    pub fn component(&self) -> usize {
        self.component
    }
    pub fn trace(&self) -> Option<&FailureTrace> {
        self.trace.as_ref()
    }
}

impl Progress {
    fn rejected(self, field: Field, component: usize) -> Error {
        Error {
            field,
            component,
            trace: Some(FailureTrace { progress: self }),
        }
    }

    fn append(&mut self, sigma_px: f32, weight: f32) -> Result<(), Field> {
        if self.component_count >= self.components.len() {
            return Err(Field::ComponentCapacity);
        }
        let component = &mut self.components[self.component_count];
        component.sigma_px = sigma_px;
        component.weight = weight;
        let radius = 3.0 * f64::from(sigma_px) + 0.5;
        if !radius.is_finite() || radius > f64::from(i32::MAX) {
            return Err(Field::Radius);
        }
        component.radius = radius as i32;
        component.operator = if sigma_px <= 0.0 || component.radius == 0 {
            Operator::Identity
        } else if sigma_px >= 3.0 {
            Operator::YvvReflect
        } else {
            Operator::FirReflect
        };
        if component.operator == Operator::YvvReflect {
            component.build_iir()?;
        }
        self.component_count += 1;
        Ok(())
    }
}

impl Plan {
    #[expect(
        clippy::result_large_err,
        reason = "ordinary failure preserves a bounded four-slot diagnostic trace without a heap owner"
    )]
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "the native inactivity comparison deliberately masks NaN before consuming frame facts"
    )]
    pub fn new(input: Input) -> Result<Option<Self>, Error> {
        if !input.active || !(input.diffusion_size_um > 0.0) {
            return Ok(None);
        }
        let reject = |field| Error {
            field,
            component: 4,
            trace: None,
        };
        if !(input.pixel_size_um.is_finite() && input.pixel_size_um > 0.0) {
            return Err(reject(Field::PixelSizeUm));
        }
        if input.render_extent.width <= 0 || input.render_extent.height <= 0 {
            return Err(reject(Field::RenderExtent));
        }
        if input.full_frame_extent.width <= 0 || input.full_frame_extent.height <= 0 {
            return Err(reject(Field::FullFrameExtent));
        }
        let mut progress = Progress {
            recipe_hash: input.recipe_hash,
            render_extent: input.render_extent,
            full_frame_extent: input.full_frame_extent,
            component_count: 0,
            components: [Component::EMPTY; 4],
        };
        let core_sigma = input.diffusion_size_um / input.pixel_size_um;
        let core_weight = 1.0_f32 - input.diffusion_tail_weight;
        if !(core_sigma.is_finite()
            && core_sigma > 0.0
            && core_weight.is_finite()
            && core_weight >= 0.0)
        {
            return Err(progress.rejected(Field::Core, 0));
        }
        if input.render_extent != input.full_frame_extent {
            return Err(progress.rejected(Field::PartialRenderWindow, 4));
        }
        if let Err(field) = progress.append(core_sigma, core_weight) {
            return Err(progress.rejected(field, 0));
        }
        if input.diffusion_tail_weight > 0.0 {
            let tail_sigma = input.diffusion_tail_um / input.pixel_size_um;
            if !(tail_sigma.is_finite() && tail_sigma >= 0.0) {
                return Err(progress.rejected(Field::Tail, 4));
            }
            const AMPLITUDES: [f32; 3] = [0.1633, 0.6496, 0.1870];
            const RATIOS: [f32; 3] = [0.5360, 1.5236, 2.7684];
            let sum = AMPLITUDES[0] + AMPLITUDES[1] + AMPLITUDES[2];
            if !(sum.is_finite() && sum > 0.0) {
                return Err(progress.rejected(Field::Tail, 4));
            }
            for i in 0..3 {
                let amplitude = AMPLITUDES[i] / sum;
                let sigma = tail_sigma * RATIOS[i];
                let weight = input.diffusion_tail_weight * amplitude;
                if !(sigma.is_finite() && sigma >= 0.0 && weight.is_finite() && weight >= 0.0) {
                    return Err(progress.rejected(Field::ComponentValues, i + 1));
                }
                if let Err(field) = progress.append(sigma, weight) {
                    return Err(progress.rejected(field, i + 1));
                }
            }
        }
        if progress.component_count == 0
            || !progress.components[..progress.component_count]
                .iter()
                .any(|c| c.weight > 0.0)
        {
            return Err(progress.rejected(Field::EmptyPlan, 4));
        }
        Ok(Some(Self { progress }))
    }

    pub fn recipe_hash(&self) -> u64 {
        self.progress.recipe_hash
    }
    pub fn render_extent(&self) -> Domain {
        self.progress.render_extent
    }
    pub fn full_frame_extent(&self) -> Domain {
        self.progress.full_frame_extent
    }
    pub fn component_count(&self) -> usize {
        self.progress.component_count
    }
    pub fn components(&self) -> &[Component; 4] {
        &self.progress.components
    }
}

impl FailureTrace {
    pub fn recipe_hash(&self) -> u64 {
        self.progress.recipe_hash
    }
    pub fn render_extent(&self) -> Domain {
        self.progress.render_extent
    }
    pub fn full_frame_extent(&self) -> Domain {
        self.progress.full_frame_extent
    }
    pub fn component_count(&self) -> usize {
        self.progress.component_count
    }
    pub fn components(&self) -> &[Component; 4] {
        &self.progress.components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn scalar(input: &Value) -> f32 {
        f32::from_bits(input.as_u64().unwrap() as u32)
    }
    fn domain(input: &Value) -> Domain {
        let a: [i32; 4] = serde_json::from_value(input.clone()).unwrap();
        Domain {
            x: a[0],
            y: a[1],
            width: a[2],
            height: a[3],
        }
    }
    fn extent(domain: Domain) -> [i32; 4] {
        [domain.x, domain.y, domain.width, domain.height]
    }
    fn input(row: &Value) -> Input {
        Input {
            active: row["active"].as_bool().unwrap(),
            diffusion_size_um: scalar(&row["size"]),
            diffusion_tail_um: scalar(&row["tail"]),
            diffusion_tail_weight: scalar(&row["weight"]),
            pixel_size_um: scalar(&row["pixel"]),
            render_extent: domain(&row["render"]),
            full_frame_extent: domain(&row["full"]),
            recipe_hash: row["hash"].as_u64().unwrap(),
        }
    }
    fn compare(progress: &Progress, expected: &Value) {
        assert_eq!(
            progress.recipe_hash,
            expected["recipe_hash"].as_u64().unwrap()
        );
        assert_eq!(
            serde_json::json!(extent(progress.render_extent)),
            expected["render"]
        );
        assert_eq!(
            serde_json::json!(extent(progress.full_frame_extent)),
            expected["full"]
        );
        assert_eq!(expected["filter"], expected["full"]);
        assert_eq!(
            progress.component_count,
            expected["count"].as_u64().unwrap() as usize
        );
        for (c, row) in progress
            .components
            .iter()
            .zip(expected["components"].as_array().unwrap())
        {
            assert_eq!(
                u64::from(c.sigma_px.to_bits()),
                row["sigma"].as_u64().unwrap()
            );
            assert_eq!(
                u64::from(c.weight.to_bits()),
                row["weight"].as_u64().unwrap()
            );
            assert_eq!(i64::from(c.radius), row["radius"].as_i64().unwrap());
            assert_eq!(c.operator as u64, row["operator"].as_u64().unwrap());
            let coefficients = [
                row["feedforward"].clone(),
                row["feedback"][0].clone(),
                row["feedback"][1].clone(),
                row["feedback"][2].clone(),
                row["denominator"].clone(),
                row["numerator"].clone(),
                row["numerators"][0].clone(),
                row["numerators"][1].clone(),
                row["numerators"][2].clone(),
            ];
            for (actual, expected) in c.coefficients.iter().zip(coefficients) {
                assert_eq!(actual.to_bits(), expected.as_u64().unwrap());
            }
            assert_eq!(
                c.boundary_accuracy().to_bits(),
                row["accuracy"].as_u64().unwrap()
            );
            assert_eq!(
                c.boundary_tolerance().to_bits(),
                row["tolerance"].as_u64().unwrap()
            );
            assert_eq!(
                u64::from(c.boundary_version()),
                row["version"].as_u64().unwrap()
            );
        }
    }

    #[test]
    fn unchanged_whole_native_words_and_failure_progress() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/dir/frame-fixtures");
        let preset = match (cfg!(windows), cfg!(debug_assertions)) {
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
        };
        let captured: Value =
            serde_json::from_slice(&std::fs::read(root.join(format!("{preset}.json"))).unwrap())
                .unwrap();
        let checked: Value =
            serde_json::from_slice(&std::fs::read(root.join("checked.json")).unwrap()).unwrap();
        assert_eq!(captured["preset"], preset);
        assert_eq!(captured["cases"].as_array().unwrap().len(), 74);
        assert_eq!(captured["products"].as_array().unwrap().len(), 24);
        for row in captured["cases"]
            .as_array()
            .unwrap()
            .iter()
            .chain(captured["products"].as_array().unwrap())
        {
            let name = row["input"]["name"].as_str().unwrap();
            let result = Plan::new(input(&row["input"]));
            assert_eq!(result.is_ok(), row["success"].as_bool().unwrap(), "{name}");
            match result {
                Ok(Some(plan)) => compare(&plan.progress, &row["native"]),
                Ok(None) => assert_eq!(row["native"]["support"], 1, "{name}"),
                Err(error) => {
                    assert_eq!(
                        error.field as u64,
                        checked["ordinary"][name][0].as_u64().unwrap(),
                        "{name}"
                    );
                    assert_eq!(
                        error.component as u64,
                        checked["ordinary"][name][1].as_u64().unwrap(),
                        "{name}"
                    );
                    if let Some(trace) = error.trace {
                        compare(&trace.progress, &row["native"]);
                    } else {
                        assert_eq!(row["native"]["support"], 1, "{name}");
                    }
                }
            }
        }
    }
}
