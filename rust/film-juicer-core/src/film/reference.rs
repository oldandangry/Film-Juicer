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
#[cfg(feature = "test-support")]
pub enum Category {
    Unsupported,
    Allocation,
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
    #[cfg(feature = "test-support")]
    Dependency {
        exposure: Exposure,
        operation: Operation,
        category: Category,
    },
}

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
    #[cfg(feature = "test-support")]
    {
        use super::input::testing;
        let fault = match operation {
            Operation::Source => testing::exposure_fault(5),
            Operation::MallettRaw => testing::exposure_fault(6),
            Operation::InputMatrices => testing::color_fault(5).err().map(|_| 1),
            Operation::LinearSrgb => testing::color_fault(8).err().map(|_| 1),
            Operation::Cat16 => testing::color_fault(2).err().map(|_| 1),
            Operation::TcSample => reconstruction::testing::take(2),
        };
        match fault {
            Some(2) => panic!("film reference dependency boundary fault"),
            Some(1) => {
                return Err(Error::Dependency {
                    exposure,
                    operation,
                    category: Category::Unsupported,
                });
            }
            Some(3) => {
                return Err(Error::Dependency {
                    exposure,
                    operation,
                    category: Category::Allocation,
                });
            }
            Some(4) => {
                return Err(Error::Reconstruction {
                    exposure,
                    operation: Some(operation),
                });
            }
            _ => (),
        }
    }
    #[cfg(not(feature = "test-support"))]
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
            visit(exposure, Operation::Cat16)?;
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
            let baseline_density = sample_density(
                baseline_log * gamma,
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

#[cfg(feature = "test-support")]
pub mod testing {
    /// Uses the production lookup with caller-supplied test-ingress shape.
    pub fn sample_density(
        query: f32,
        axis: &[f32],
        curves: &[[f32; 3]],
        channel: usize,
    ) -> Option<f32> {
        if axis.is_empty() || axis.len() != curves.len() || channel >= 3 {
            return None;
        }
        super::sample_density(query, axis, curves, channel)
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
        for query in [f32::NEG_INFINITY, 0.0, f32::INFINITY] {
            assert_eq!(
                sample_density(query, &[2.0], &[[7.0, 8.0, 9.0]], 2),
                Some(9.0)
            );
        }
    }

    #[test]
    fn ordered_infinity_arithmetic_is_a_present_density() {
        let axis = [f32::NEG_INFINITY, 0.0, f32::INFINITY];
        let curves = [[1.0; 3], [2.0; 3], [3.0; 3]];
        assert!(sample_density(-1.0, &axis, &curves, 0).unwrap().is_nan());
        assert_eq!(sample_density(1.0, &axis, &curves, 0), Some(2.0));
        assert_eq!(sample_density(f32::NAN, &axis, &curves, 0), None);
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn existing_exposure_hook_preserves_typed_allocation_and_consumption() {
        use crate::{film, profile, route};
        let source = profile::load_film_source(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../Resources/profiles/kodak_portra_400.json"),
        )
        .unwrap();
        let development = Development::new(
            std::sync::Arc::new(profile::FilmProfile::new(source).unwrap()),
            film::Params {
                gamma_factor: 1.0,
                layers_required: false,
                requested_route: route::ScanRoute::NegativeDirect,
            },
        )
        .unwrap();
        let basis = [[1.0; 3]; 81];
        let light = [1.0; 81];
        let sensitivity = [[1.0; 3]; 81];
        let prepare = || {
            Reference::new(
                &development,
                Params {
                    manual_exposure_ev: 0.0,
                    auto_exposure_enabled: true,
                },
                Some(Reconstruction::Mallett {
                    basis_rgb: &basis,
                    illuminant: &light,
                    sensitivity_rgb: &sensitivity,
                    green_scale: 1.0,
                }),
            )
        };
        for (selector, operation) in [(5, Operation::Source), (6, Operation::MallettRaw)] {
            for (index, exposure) in [(1, Exposure::Baseline), (2, Exposure::Compensated)] {
                film::input::testing::arm_exposure(selector, index, 3);
                assert_eq!(
                    prepare().err(),
                    Some(Error::Dependency {
                        exposure,
                        operation,
                        category: Category::Allocation
                    })
                );
                assert!(film::input::testing::exposure_consumed(selector, 3));
                assert!(!film::input::testing::exposure_consumed(selector, 3));
                assert!(prepare().is_ok());
            }
        }
    }
}
