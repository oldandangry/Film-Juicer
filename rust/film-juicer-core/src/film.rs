//! Completed capture-film development; source tables remain in the retained profile.

pub mod input;
pub mod dir;
pub mod density;
pub mod grain;
pub mod reference;

use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use crate::hash;
use crate::profile::{FilmProfile, Polarity};
use crate::route::{self, CapturePolarity, RouteSelection, ScanRoute};

pub struct Params {
    pub gamma_factor: f32,
    pub layers_required: bool,
    pub requested_route: ScanRoute,
}

pub struct Development {
    profile: Arc<FilmProfile>,
    normalized_curves_cmy: Vec<[f32; 3]>,
    route: ScanRoute,
    polarity: Polarity,
    gamma_rgb: [f32; 3],
    minimum_cmy: [f32; 3],
    maximum_cmy: [f32; 3],
    layers_required: bool,
    normalized_curves_hash: u64,
    layers_hash: u64,
    hash: u64,
    #[cfg(any(test, feature = "test-support"))]
    _curve_observation: test_support::CurveObservation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidGamma,
    RouteMismatch,
    Size,
    Allocation,
    ZeroIdentity,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidGamma => {
                "InvalidAuthoredControl component=film_development field=film_gamma_factor"
            }
            Self::RouteMismatch => {
                "UnsupportedMode phase=3A selected capture profile route mismatch"
            }
            Self::Size => "film development normalized table size exceeds addressable storage",
            Self::Allocation => "film development normalized table reservation failed",
            Self::ZeroIdentity => {
                "MalformedRequiredProfileData phase=3A field=data.density_curves hash"
            }
        })
    }
}
impl std::error::Error for Error {}

fn curve_hash(samples: &[f32]) -> u64 {
    let pair = hash::f32s_with_nan_mask(samples);
    hash::u64s(&[pair.values, pair.nan_mask])
}

fn normalize_curves(curves: &mut [[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    // Complete profiles establish nonempty tables. Keep the first operand on
    // ordered ties and unordered comparisons, including either zero sign.
    let (mut minimum, mut maximum) = (curves[0], curves[0]);
    for row in curves.iter() {
        for (channel, &value) in row.iter().enumerate() {
            if value < minimum[channel] {
                minimum[channel] = value;
            }
            if maximum[channel] < value {
                maximum[channel] = value;
            }
        }
    }
    for row in curves {
        for (channel, value) in row.iter_mut().enumerate() {
            *value -= minimum[channel];
        }
    }
    (minimum, maximum)
}

impl Development {
    pub fn profile(&self) -> &FilmProfile {
        &self.profile
    }

    pub fn new(profile: Arc<FilmProfile>, params: Params) -> Result<Self, Error> {
        if !params.gamma_factor.is_finite() || !(0.05..=4.0).contains(&params.gamma_factor) {
            return Err(Error::InvalidGamma);
        }
        let polarity = profile.info().polarity();
        let capture_polarity = match polarity {
            Polarity::Negative => CapturePolarity::Negative,
            Polarity::Positive => CapturePolarity::Positive,
        };
        let selection = match params.requested_route {
            ScanRoute::NegativeDirect | ScanRoute::PositiveDirect => RouteSelection::Direct,
            ScanRoute::NegativePrint | ScanRoute::PositivePrint => RouteSelection::Print,
        };
        let route = route::resolve(capture_polarity, selection);
        if route != params.requested_route {
            return Err(Error::RouteMismatch);
        }
        let authored = profile.tables().density_curves();
        let bytes = authored
            .len()
            .checked_mul(3)
            .and_then(|count| count.checked_mul(size_of::<f32>()))
            .ok_or(Error::Size)?;
        if bytes > isize::MAX as usize {
            return Err(Error::Size);
        }
        #[cfg(any(test, feature = "test-support"))]
        let fault = test_support::take();
        let capacity = authored.len();
        #[cfg(any(test, feature = "test-support"))]
        let capacity = if fault == 1 { usize::MAX } else { capacity };
        let mut normalized_curves_cmy = Vec::new();
        normalized_curves_cmy
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Allocation)?;
        #[cfg(any(test, feature = "test-support"))]
        let curve_observation = test_support::CurveObservation::new();
        normalized_curves_cmy.extend_from_slice(authored);
        let (minimum_cmy, maximum_cmy) = normalize_curves(&mut normalized_curves_cmy);
        let normalized_curves_hash = curve_hash(normalized_curves_cmy.as_flattened());
        #[cfg(any(test, feature = "test-support"))]
        let normalized_curves_hash = if fault == 2 {
            0
        } else {
            normalized_curves_hash
        };
        let layers_hash = if params.layers_required {
            let mut identity = hash::FNV_OFFSET;
            for layer in profile.tables().density_curves_layers() {
                for channel in layer {
                    hash::update_bytes(&mut identity, &curve_hash(channel).to_le_bytes());
                }
            }
            identity
        } else {
            0
        };
        #[cfg(any(test, feature = "test-support"))]
        let layers_hash = if fault == 4 { 0 } else { layers_hash };
        let gamma_rgb = [params.gamma_factor; 3];
        let mut identity = hash::FNV_OFFSET;
        hash::update_bytes(
            &mut identity,
            &[match polarity {
                Polarity::Negative => 0,
                Polarity::Positive => 1,
            }],
        );
        hash::update_bytes(&mut identity, &normalized_curves_hash.to_le_bytes());
        hash::update_bytes(&mut identity, &[u8::from(params.layers_required)]);
        if params.layers_required {
            hash::update_bytes(&mut identity, &layers_hash.to_le_bytes());
        }
        for gamma in gamma_rgb {
            hash::update_bytes(&mut identity, &gamma.to_bits().to_le_bytes());
        }
        #[cfg(any(test, feature = "test-support"))]
        let identity = if fault == 3 { 0 } else { identity };
        if normalized_curves_hash == 0 || identity == 0 {
            return Err(Error::ZeroIdentity);
        }
        Ok(Self {
            profile,
            normalized_curves_cmy,
            route,
            polarity,
            gamma_rgb,
            minimum_cmy,
            maximum_cmy,
            layers_required: params.layers_required,
            normalized_curves_hash,
            layers_hash,
            hash: identity,
            #[cfg(any(test, feature = "test-support"))]
            _curve_observation: curve_observation,
        })
    }

    pub fn route(&self) -> ScanRoute {
        self.route
    }
    pub fn polarity(&self) -> Polarity {
        self.polarity
    }
    pub fn gamma_rgb(&self) -> &[f32; 3] {
        &self.gamma_rgb
    }
    pub fn minimum_cmy(&self) -> &[f32; 3] {
        &self.minimum_cmy
    }
    pub fn maximum_cmy(&self) -> &[f32; 3] {
        &self.maximum_cmy
    }
    pub fn layers_required(&self) -> bool {
        self.layers_required
    }
    pub fn log_exposure(&self) -> &[f32] {
        self.profile.tables().interpolation_log_exposure()
    }
    pub fn authored_curves_cmy(&self) -> &[[f32; 3]] {
        self.profile.tables().density_curves()
    }
    pub fn normalized_curves_cmy(&self) -> &[[f32; 3]] {
        &self.normalized_curves_cmy
    }
    pub fn density_curves_layers(&self) -> Option<&[[Vec<f32>; 3]; 3]> {
        self.layers_required
            .then(|| self.profile.tables().density_curves_layers())
    }
    pub fn normalized_curves_hash(&self) -> u64 {
        self.normalized_curves_hash
    }
    pub fn layers_hash(&self) -> u64 {
        self.layers_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    //! Bounded reservation and rare zero-identity rejection witnesses, absent in production.
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LIVE_CURVES: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATED_CURVES: AtomicUsize = AtomicUsize::new(0);
    static DROPPED_CURVES: AtomicUsize = AtomicUsize::new(0);

    // Field order drops this observation after a completed Development's curve
    // vector; failed local construction drops both before its caller can sample.
    pub(super) struct CurveObservation;
    impl CurveObservation {
        pub(super) fn new() -> Self {
            LIVE_CURVES.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_CURVES.fetch_add(1, Ordering::Relaxed);
            Self
        }
    }
    impl Drop for CurveObservation {
        fn drop(&mut self) {
            LIVE_CURVES.fetch_sub(1, Ordering::Relaxed);
            DROPPED_CURVES.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn curve_counts() -> [usize; 3] {
        [
            LIVE_CURVES.load(Ordering::Relaxed),
            ALLOCATED_CURVES.load(Ordering::Relaxed),
            DROPPED_CURVES.load(Ordering::Relaxed),
        ]
    }
    thread_local! { static FAULT: Cell<u32> = const { Cell::new(0) }; }
    pub fn reject_next(fault: u32) {
        FAULT.set(fault);
    }
    pub(super) fn take() -> u32 {
        FAULT.replace(0)
    }
}

#[cfg(test)]
mod tests {
    use super::{curve_hash, normalize_curves};
    use crate::hash;
    use serde_json::Value;
    use std::path::PathBuf;

    #[test]
    fn native_primitive_extrema_subtraction_and_mask_boundaries() {
        let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
        };
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/film/fixtures")
            .join(format!("{preset}-primitive.json"));
        let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for row in fixture["normalization"].as_array().unwrap() {
            let mut curves: Vec<[f32; 3]> = row["authored"]
                .as_array()
                .unwrap()
                .iter()
                .map(|sample| {
                    std::array::from_fn(|channel| {
                        f32::from_bits(sample[channel].as_u64().unwrap() as u32)
                    })
                })
                .collect();
            let (minimum, maximum) = normalize_curves(&mut curves);
            assert_eq!(serde_json::json!(minimum.map(f32::to_bits)), row["minimum"]);
            assert_eq!(serde_json::json!(maximum.map(f32::to_bits)), row["maximum"]);
            assert_eq!(
                serde_json::json!(
                    curves
                        .iter()
                        .map(|samples| samples.map(f32::to_bits))
                        .collect::<Vec<_>>()
                ),
                row["normalized"]
            );
            assert_eq!(
                curve_hash(curves.as_flattened()),
                row["hash"].as_u64().unwrap()
            );
        }
        for row in fixture["hash"].as_array().unwrap() {
            let samples: Vec<f32> = row["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|word| f32::from_bits(word.as_u64().unwrap() as u32))
                .collect();
            let pair = hash::f32s_with_nan_mask(&samples);
            assert_eq!(pair.values, row["value_hash"].as_u64().unwrap());
            assert_eq!(pair.nan_mask, row["mask_hash"].as_u64().unwrap());
            assert_eq!(curve_hash(&samples), row["hash"].as_u64().unwrap());
        }
    }
}
