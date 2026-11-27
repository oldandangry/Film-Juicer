//! Complete synthetic film reference from a borrowed Development and reconstruction.

use crate::{color, exposure, hash, reconstruction};
use super::Development;

pub struct Params {
    pub manual_exposure_ev: f32,
    pub auto_exposure_enabled: bool,
}

pub enum Reconstruction<'a> {
    Mallett {
        basis_rgb: &'a [[f32; 3]; 81],
        illuminant: &'a [f32; 81],
        sensitivity_rgb: &'a [[f32; 3]; 81],
        green_scale: f32,
    },
    Tc {
        samples: &'a [f32; reconstruction::TC_LUT_SAMPLE_COUNT],
        projection_white_xyz: [f32; 3],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Exposure {
    Baseline = 1,
    Compensated = 2,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Operation {
    Source = 1,
    InputMatrices = 2,
    LinearSrgb = 3,
    Cat16 = 4,
    TcSample = 5,
    MallettRaw = 6,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Reconstruction {
        exposure: Exposure,
        operation: Option<Operation>,
    },
    Meter,
    DensityLookup {
        exposure: Exposure,
        channel: usize,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Reconstruction { .. } => {
                "MalformedRequiredResource phase=3B field=synthetic_reference_raw"
            }
            Self::Meter => "MalformedRequiredResource phase=3B field=synthetic_reference_meter",
            Self::DensityLookup { .. } => {
                "ResourceDescriptorMismatch phase=3B field=synthetic_reference_density_lookup"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for Error {}

pub struct Reference {
    baseline_raw_rgb: [f32; 3],
    compensated_raw_rgb: [f32; 3],
    baseline_density_cmy: [f32; 3],
    compensated_density_cmy: [f32; 3],
    meter_scale: f32,
    hash: u64,
}

fn sample_density(query: f32, axis: &[f32], curves: &[[f32; 3]], channel: usize) -> Option<f32> {
    if query.is_nan() {
        return None;
    }
    if query <= axis[0] {
        return Some(curves[0][channel]);
    }
    if query >= axis[axis.len() - 1] {
        return Some(curves[curves.len() - 1][channel]);
    }
    // Development establishes this nondecreasing axis once. Interior ties select
    // the last equal row, while endpoint checks above retain the first/last row.
    let hi = axis.partition_point(|value| *value <= query);
    if hi == 0 || hi == axis.len() {
        return None;
    }
    let lo = hi - 1;
    let span = axis[hi] - axis[lo];
    let t = if span > 0.0 {
        (query - axis[lo]) / span
    } else {
        0.0
    };
    Some(curves[lo][channel] + t * (curves[hi][channel] - curves[lo][channel]))
}

fn visit(exposure: Exposure, operation: Operation) -> Result<(), Error> {
    let _ = (exposure, operation);
    Ok(())
}

fn reconstruct(
    ev: f32,
    exposure: Exposure,
    reconstruction: Option<&Reconstruction<'_>>,
) -> Result<[f32; 3], Error> {
    let failure = |operation| Error::Reconstruction {
        exposure,
        operation,
    };
    visit(exposure, Operation::Source)?;
    let source = exposure::reference_source(ev)
        .map_err(|_| failure(Some(Operation::Source)))?
        .value();
    let rgb = match reconstruction.ok_or_else(|| failure(None))? {
        Reconstruction::Mallett {
            basis_rgb,
            illuminant,
            sensitivity_rgb,
            green_scale,
        } => {
            visit(exposure, Operation::MallettRaw)?;
            *exposure::mallett_reference_raw(exposure::ReferenceInput {
                basis_rgb,
                illuminant,
                sensitivity_rgb,
                source,
                green_scale: *green_scale,
            })
            .map_err(|_| failure(Some(Operation::MallettRaw)))?
            .rgb()
        }
        Reconstruction::Tc {
            samples,
            projection_white_xyz,
        } => {
            visit(exposure, Operation::InputMatrices)?;
            let matrices = color::input_matrices(color::InputSpace::SrgbRec709);
            visit(exposure, Operation::LinearSrgb)?;
            let xyz = color::linear_srgb_to_xyz([source; 3]);
            let projected = color::adapt_cat16(
                xyz,
                color::Whites {
                    source_xyz: matrices.d65_white_xyz,
                    destination_xyz: *projection_white_xyz,
                },
            );
            visit(exposure, Operation::TcSample)?;
            reconstruction::sample_tc_lut(samples, projected)
                .map_err(|_| failure(Some(Operation::TcSample)))?
        }
    };
    if !rgb.iter().all(|value| value.is_finite() && *value >= 0.0) {
        return Err(failure(Some(match reconstruction {
            Some(Reconstruction::Mallett { .. }) => Operation::MallettRaw,
            _ => Operation::TcSample,
        })));
    }
    Ok(rgb)
}

impl Reference {
    pub fn new(
        development: &Development,
        params: Params,
        reconstruction: Option<Reconstruction<'_>>,
    ) -> Result<Self, Error> {
        let baseline_raw_rgb = reconstruct(0.0, Exposure::Baseline, reconstruction.as_ref())?;
        let compensated_raw_rgb = reconstruct(
            params.manual_exposure_ev,
            Exposure::Compensated,
            reconstruction.as_ref(),
        )?;
        let mut meter_scale = 1.0;
        if params.auto_exposure_enabled {
            let mean =
                ((baseline_raw_rgb[0] + baseline_raw_rgb[1]) + baseline_raw_rgb[2]) / 3.0_f32;
            if !mean.is_finite() || mean <= 0.0 {
                return Err(Error::Meter);
            }
            // A positive subnormal mean may overflow here. Its downstream NaN
            // query belongs to density lookup, not meter admission.
            meter_scale = 1.0_f32 / mean;
        }
        let mut baseline_density_cmy = [0.0; 3];
        let mut compensated_density_cmy = [0.0; 3];
        for channel in 0..3 {
            let baseline_log = (baseline_raw_rgb[channel] * meter_scale + 1.0e-10_f32).log10();
            let compensated_log =
                (compensated_raw_rgb[channel] * meter_scale + 1.0e-10_f32).log10();
            let gamma = development.gamma_rgb()[channel];
            let baseline_query = baseline_log * gamma;
            let baseline_density = sample_density(
                baseline_query,
                development.log_exposure(),
                development.authored_curves_cmy(),
                channel,
            );
            let compensated_density = sample_density(
                compensated_log * gamma,
                development.log_exposure(),
                development.authored_curves_cmy(),
                channel,
            );
            baseline_density_cmy[channel] = baseline_density.ok_or(Error::DensityLookup {
                exposure: Exposure::Baseline,
                channel,
            })?;
            compensated_density_cmy[channel] = compensated_density.ok_or(Error::DensityLookup {
                exposure: Exposure::Compensated,
                channel,
            })?;
        }
        let mut identity = hash::FNV_OFFSET;
        for value in baseline_raw_rgb
            .iter()
            .chain(&compensated_raw_rgb)
            .chain(&baseline_density_cmy)
            .chain(&compensated_density_cmy)
            .chain(std::iter::once(&meter_scale))
        {
            hash::update_bytes(&mut identity, &value.to_bits().to_le_bytes());
        }
        Ok(Self {
            baseline_raw_rgb,
            compensated_raw_rgb,
            baseline_density_cmy,
            compensated_density_cmy,
            meter_scale,
            hash: identity,
        })
    }
    pub fn baseline_raw_rgb(&self) -> &[f32; 3] {
        &self.baseline_raw_rgb
    }
    pub fn compensated_raw_rgb(&self) -> &[f32; 3] {
        &self.compensated_raw_rgb
    }
    pub fn baseline_density_cmy(&self) -> &[f32; 3] {
        &self.baseline_density_cmy
    }
    pub fn compensated_density_cmy(&self) -> &[f32; 3] {
        &self.compensated_density_cmy
    }
    pub fn meter_scale(&self) -> f32 {
        self.meter_scale
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_lookup_preserves_endpoint_and_upper_bound_selection() {
        let curves = [[0.0; 3], [4.0; 3], [8.0; 3], [12.0; 3]];
        let axis = [0.0, 2.0, 2.0, 4.0];
        assert_eq!(sample_density(2.0, &axis, &curves, 0), Some(8.0));
        assert_eq!(sample_density(3.0, &axis, &curves, 2), Some(10.0));
        assert_eq!(sample_density(2.0, &[2.0, 2.0], &curves[..2], 0), Some(0.0));
        assert_eq!(
            sample_density(-0.0, &[-0.0, 0.0], &curves[..2], 0)
                .unwrap()
                .to_bits(),
            0
        );
        assert_eq!(
            sample_density(0.0, &[-1.0, -0.0, 0.0, 1.0], &curves, 0),
            Some(8.0)
        );
        let queries = [
            ("negative-infinity", f32::NEG_INFINITY),
            ("zero", 0.0),
            ("positive-infinity", f32::INFINITY),
        ];
        let cases_ids: Vec<_> = IntoIterator::into_iter(queries.iter().map(|(id, _)| *id))
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "film-reference-singleton-query"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "film-reference-singleton-query"
        );
        for (id, query) in queries {
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            assert_eq!(
                sample_density(query, &[2.0], &[[7.0, 8.0, 9.0]], 2),
                Some(9.0)
            );
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
    fn ordered_infinity_arithmetic_is_a_present_density() {
        let axis = [f32::NEG_INFINITY, 0.0, f32::INFINITY];
        let curves = [[1.0; 3], [2.0; 3], [3.0; 3]];
        assert!(sample_density(-1.0, &axis, &curves, 0).unwrap().is_nan());
        assert_eq!(sample_density(1.0, &axis, &curves, 0), Some(2.0));
        assert_eq!(sample_density(f32::NAN, &axis, &curves, 0), None);
    }
}
