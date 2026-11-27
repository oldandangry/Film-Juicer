//! Asset-bound derived entries. Construction and destruction run outside map locks.

use std::fmt;
use std::sync::Arc;

use crate::{gamut, illuminant};
use crate::data_io::ReadErrorKind;
use super::{AssetError, Assets, CsvSource};

#[derive(Debug)]
pub enum DerivedError {
    Allocation(&'static str),
    Internal(&'static str),
    Resource(&'static str),
    Unsupported(&'static str),
    Hull(gamut::HullError),
    Boundary(gamut::BoundaryError),
    Illuminant {
        label: &'static str,
        error: illuminant::Error,
    },
}
impl fmt::Display for DerivedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocation(detail)
            | Self::Internal(detail)
            | Self::Resource(detail)
            | Self::Unsupported(detail) => f.write_str(detail),
            Self::Hull(error) => error.fmt(f),
            Self::Boundary(error) => error.fmt(f),
            Self::Illuminant { label, error } => {
                write!(f, "Illuminant preparation ({label}): {error}")
            }
        }
    }
}
impl std::error::Error for DerivedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Hull(error) => Some(error),
            Self::Boundary(error) => Some(error),
            _ => None,
        }
    }
}
impl From<DerivedError> for AssetError {
    fn from(error: DerivedError) -> Self {
        Self::Derived(error)
    }
}

/// Seven canonical-position curves, allowing the established partial result.
/// The substantial allocation is fallible and remains stable in every live hold.
pub struct Illuminants {
    curves: Vec<Option<[f32; 81]>>,
}

/// A fallibly reserved substantial entry, shared without copying its coordinates.
/// The small Arc allocation owns only this Vec header.
pub struct InputHull {
    hull: Vec<gamut::Hull>,
}

/// Asset cache hold around the accepted mathematical boundary. Its Cmax buffer
/// remains the leaf producer's allocation; acquisition never clones that buffer.
pub struct OutputBoundary {
    boundary: gamut::OutputBoundary,
}
impl OutputBoundary {
    pub fn boundary(&self) -> &gamut::OutputBoundary {
        &self.boundary
    }
    pub fn cmax(&self) -> &[f32] {
        self.boundary.cmax()
    }
    pub fn transform_hash(&self) -> u64 {
        self.boundary.transform_hash()
    }
    pub fn contract_hash(&self) -> u64 {
        self.boundary.contract_hash()
    }
    pub fn hash(&self) -> u64 {
        self.boundary.hash()
    }
}
impl InputHull {
    pub fn hull(&self) -> &gamut::Hull {
        &self.hull[0]
    }
}
impl Illuminants {
    pub fn d65(&self) -> Option<&[f32; 81]> {
        self.curves[0].as_ref()
    }
    pub fn d55(&self) -> Option<&[f32; 81]> {
        self.curves[1].as_ref()
    }
    pub fn d50(&self) -> Option<&[f32; 81]> {
        self.curves[2].as_ref()
    }
    pub fn tungsten(&self) -> Option<&[f32; 81]> {
        self.curves[3].as_ref()
    }
    pub fn k75p(&self) -> Option<&[f32; 81]> {
        self.curves[4].as_ref()
    }
    pub fn tungsten_kg3(&self) -> Option<&[f32; 81]> {
        self.curves[5].as_ref()
    }
    pub fn tungsten_kg3_lens(&self) -> Option<&[f32; 81]> {
        self.curves[6].as_ref()
    }
    fn complete(&self) -> bool {
        self.curves.iter().all(Option::is_some)
    }
}

pub(super) struct OutputCache {
    pub(super) generation: u64,
    slots: [Option<Arc<OutputBoundary>>; 9],
}

fn reserve_hull_entry(count: usize) -> Result<Vec<gamut::Hull>, AssetError> {
    let mut hull = Vec::new();
    hull.try_reserve_exact(count)
        .map_err(|_| DerivedError::Allocation("input hull allocation failed"))?;
    Ok(hull)
}
impl OutputCache {
    pub(super) fn new() -> Self {
        Self {
            generation: 0,
            slots: std::array::from_fn(|_| None),
        }
    }
}

fn available_rows(
    assets: &Assets,
    source: CsvSource,
) -> Result<Option<Arc<crate::data_io::CsvPairs>>, AssetError> {
    match assets.csv_source(source) {
        Ok(rows) => Ok(Some(rows)),
        // Ordinary resource failure permits a partial curve set. Capacity and
        // integrity failures retain precedence even for an unselected curve.
        Err(AssetError::CsvSource { error, .. })
            if !matches!(error.kind, ReadErrorKind::Capacity) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
fn available_curve(
    result: Result<[f32; 81], illuminant::Error>,
    label: &'static str,
) -> Result<Option<[f32; 81]>, AssetError> {
    match result {
        Ok(curve) => Ok(Some(curve)),
        Err(error) if error.kind == illuminant::ErrorKind::Preparation => Ok(None),
        Err(error) => Err(DerivedError::Illuminant { label, error }.into()),
    }
}

impl Assets {
    pub fn illuminants(&self) -> Result<Arc<Illuminants>, AssetError> {
        self.illuminants_observed_inner()
    }
    fn illuminants_observed_inner(&self) -> Result<Arc<Illuminants>, AssetError> {
        self.illuminants_with(|| {
            let mut curves = Vec::new();
            curves
                .try_reserve_exact(7)
                .map_err(|_| DerivedError::Allocation("illuminant curve allocation failed"))?;
            let mut candidate = Illuminants { curves };
            for (source, label) in [
                (CsvSource::D65, "D65"),
                (CsvSource::D55, "D55"),
                (CsvSource::D50, "D50"),
                (CsvSource::T, "T"),
                (CsvSource::K75p, "K75P"),
            ] {
                candidate.curves.push(match available_rows(self, source)? {
                    Some(rows) => available_curve(illuminant::from_samples(rows.rows()), label)?,
                    None => None,
                });
            }
            let mut kg3 = available_rows(self, CsvSource::Kg3)?;
            let filtered = match &kg3 {
                Some(rows) => available_curve(
                    illuminant::tungsten_kg3(rows.rows()).map(|(curve, _)| curve),
                    "KG3",
                )?,
                None => {
                    kg3 = available_rows(self, CsvSource::Kg3)?;
                    None
                }
            };
            candidate.curves.push(filtered);
            let mut lens_curve = None;
            if let Some(rows) = kg3 {
                match illuminant::prepare_lens(rows.rows()) {
                    Ok((input, _)) => {
                        if let Some(lens) = available_rows(self, CsvSource::Canon24F28Is)? {
                            lens_curve = available_curve(
                                illuminant::finish_lens(input, lens.rows()).map(|(curve, _)| curve),
                                "Canon 24 F2.8 IS",
                            )?;
                        }
                    }
                    Err(error) if error.kind == illuminant::ErrorKind::Preparation => (),
                    Err(error) => {
                        return Err(DerivedError::Illuminant {
                            label: "KG3",
                            error,
                        }
                        .into());
                    }
                }
            }
            candidate.curves.push(lens_curve);
            Ok(candidate)
        })
    }

    fn illuminants_with(
        &self,
        build: impl FnOnce() -> Result<Illuminants, AssetError>,
    ) -> Result<Arc<Illuminants>, AssetError> {
        let cached = self
            .illuminants
            .lock()
            .map_err(|_| DerivedError::Internal("illuminant cache poisoned"))?
            .as_ref()
            .map(Arc::clone);
        if let Some(cached) = cached {
            return Ok(cached);
        }
        let candidate = Arc::new(build()?);
        let selected = {
            let mut cache = self
                .illuminants
                .lock()
                .map_err(|_| DerivedError::Internal("illuminant cache poisoned"))?;
            if cache.is_none() && candidate.complete() {
                *cache = Some(Arc::clone(&candidate));
            }
            cache.as_ref().map(Arc::clone)
        };
        Ok(selected.unwrap_or(candidate))
    }

    pub fn input_hull(&self) -> Result<Arc<InputHull>, AssetError> {
        self.input_hull_observed_inner()
    }
    fn input_hull_observed_inner(&self) -> Result<Arc<InputHull>, AssetError> {
        self.input_hull_with(|| {
            let curves = self.illuminants()?;
            let d65 = curves.d65().ok_or(DerivedError::Resource("MalformedRequiredResource component=input_gamut_hull requirement=cie1931_380_700_and_d65"))?;
            let cmf = self.cmf()?;
            if cmf.rows().len() != 81 {
                return Err(DerivedError::Resource("MalformedRequiredResource component=input_gamut_hull requirement=cie1931_380_700_and_d65").into());
            }
            let x = std::array::from_fn(|i| cmf.rows()[i][1]);
            let y = std::array::from_fn(|i| cmf.rows()[i][2]);
            let z = std::array::from_fn(|i| cmf.rows()[i][3]);
            gamut::build_input_hull(gamut::HullInput { x: &x, y: &y, z: &z, d65 }).map_err(|error| DerivedError::Hull(error).into())
        })
    }

    fn input_hull_with(
        &self,
        build: impl FnOnce() -> Result<gamut::Hull, AssetError>,
    ) -> Result<Arc<InputHull>, AssetError> {
        let cached = self
            .input_hull
            .lock()
            .map_err(|_| DerivedError::Internal("input hull cache poisoned"))?
            .as_ref()
            .map(Arc::clone);
        if let Some(cached) = cached {
            return Ok(cached);
        }
        let hull = reserve_hull_entry(1)?;
        let mut value = InputHull { hull };
        value.hull.push(build()?);
        let candidate = Arc::new(value);
        let selected = {
            let mut cache = self
                .input_hull
                .lock()
                .map_err(|_| DerivedError::Internal("input hull cache poisoned"))?;
            if cache.is_none() {
                *cache = Some(Arc::clone(&candidate));
            }
            Arc::clone(cache.as_ref().expect("complete hull published"))
        };
        Ok(selected)
    }

    pub fn output_boundary(
        &self,
        source: gamut::BoundaryInput<'_>,
        expected_contract_hash: u64,
    ) -> Result<Arc<OutputBoundary>, AssetError> {
        self.output_boundary_with(source, expected_contract_hash, || {
            gamut::build_output_boundary(source)
        })
    }

    fn output_boundary_with(
        &self,
        source: gamut::BoundaryInput<'_>,
        expected_contract_hash: u64,
        build: impl FnOnce() -> Result<gamut::OutputBoundary, gamut::BoundaryError>,
    ) -> Result<Arc<OutputBoundary>, AssetError> {
        if source.transform_hash == 0 || expected_contract_hash == 0 {
            return Err(DerivedError::Resource(
                "ResourceDescriptorMismatch component=output_boundary_cache field=transform",
            )
            .into());
        }
        let index = source.space as usize;
        let matches = |entry: &OutputBoundary| -> Result<(), AssetError> {
            if entry.transform_hash() != source.transform_hash {
                return Err(DerivedError::Resource("ResourceDescriptorMismatch component=output_boundary_cache field=published_identity").into());
            }
            if entry.contract_hash() != expected_contract_hash {
                return Err(DerivedError::Resource("ResourceDescriptorMismatch component=output_boundary_cache field=contract_identity").into());
            }
            Ok(())
        };
        let (selected, generation) = {
            let cache = self
                .output_boundaries
                .lock()
                .map_err(|_| DerivedError::Internal("output boundary cache poisoned"))?;
            (
                cache.slots[index].as_ref().map(Arc::clone),
                cache.generation,
            )
        };
        if let Some(selected) = selected {
            matches(&selected)?;
            return Ok(selected);
        }
        // Own failure wins even when a contender has published meanwhile.
        let candidate = Arc::new(OutputBoundary {
            boundary: build().map_err(DerivedError::Boundary)?,
        });
        let selected = {
            let mut cache = self
                .output_boundaries
                .lock()
                .map_err(|_| DerivedError::Internal("output boundary cache poisoned"))?;
            if cache.generation != generation {
                None
            } else {
                let slot = &mut cache.slots[index];
                if slot.is_none() {
                    *slot = Some(Arc::clone(&candidate));
                }
                slot.as_ref().map(Arc::clone)
            }
        };
        // Stale success remains private, independent of any newer winner.
        let Some(selected) = selected else {
            return Ok(candidate);
        };
        matches(&selected)?;
        Ok(selected)
    }

    pub(super) fn release_illuminants(&self) -> Result<(), AssetError> {
        let illuminants = self
            .illuminants
            .lock()
            .map_err(|_| DerivedError::Internal("illuminant cache poisoned"))?
            .take();
        drop(illuminants);
        Ok(())
    }

    pub(super) fn release_output_boundaries(&self) -> Result<(), AssetError> {
        let detached = {
            let mut cache = self
                .output_boundaries
                .lock()
                .map_err(|_| DerivedError::Internal("output boundary cache poisoned"))?;
            if cache.generation == u64::MAX {
                return Err(
                    DerivedError::Internal("output boundary cache generation exhausted").into(),
                );
            }
            cache.generation += 1;
            std::mem::replace(&mut cache.slots, std::array::from_fn(|_| None))
        };
        drop(detached);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{self, InputSpace, OutputSpace};
    use std::path::Path;
    use std::sync::Barrier;
    use std::thread;

    fn assets() -> Assets {
        Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"))
    }

    fn complete() -> Illuminants {
        Illuminants {
            curves: vec![Some([1.0; 81]); 7],
        }
    }
    fn transform() -> gamut::OutputTransform {
        let output = color::output_color(OutputSpace::Srgb);
        gamut::output_transform(gamut::TransformInput {
            space: OutputSpace::Srgb,
            rgb_to_xyz: output.rgb_to_xyz(),
            xyz_to_rgb: output.xyz_to_rgb(),
            white_xyz: output.white_xyz(),
            d65_white_xyz: &color::input_matrices(InputSpace::SrgbRec709).d65_white_xyz,
        })
        .unwrap()
    }
    fn source(transform: &gamut::OutputTransform) -> gamut::BoundaryInput<'_> {
        gamut::BoundaryInput {
            space: transform.space(),
            d65_xyz_to_native_rgb: transform.d65_xyz_to_native_rgb(),
            transform_hash: transform.hash(),
        }
    }

    #[test]
    fn partial_illuminants_retry_and_live_holds_survive_release() {
        let assets = assets();
        let partial = assets
            .illuminants_with(|| {
                let mut curves = complete();
                curves.curves[1] = None;
                Ok(curves)
            })
            .unwrap();
        assert!(partial.d55().is_none());
        assert!(assets.illuminants.lock().unwrap().is_none());
        let held = assets.illuminants().unwrap();
        assert!(held.complete());
        let pointer = held.d65().unwrap().as_ptr();
        for _ in 0..100 {
            let warm = assets
                .illuminants_with(|| panic!("warm acquisition rebuilt curves"))
                .unwrap();
            assert!(Arc::ptr_eq(&held, &warm));
            assert_eq!(pointer, warm.d65().unwrap().as_ptr());
        }
        let weak = Arc::downgrade(&held);
        assets.release_cached_payloads().unwrap();
        assert_eq!(held.d65().unwrap().as_ptr(), pointer);
        let replacement = assets.illuminants().unwrap();
        assert!(!Arc::ptr_eq(&held, &replacement));
        assert_eq!(held.d65(), replacement.d65());
        drop(assets);
        drop(held);
        assert!(weak.upgrade().is_none());
        assert!(replacement.complete());
    }

    #[test]
    fn cold_illuminant_contenders_select_one_entry_outside_locks() {
        let assets = assets();
        let barrier = Barrier::new(2);
        thread::scope(|scope| {
            let build = || {
                assets
                    .illuminants_with(|| {
                        barrier.wait();
                        // A release during either build must not deadlock or prevent
                        // later publication in this family's distinct protocol.
                        assets.release_cached_payloads()?;
                        Ok(complete())
                    })
                    .unwrap()
            };
            let first = scope.spawn(build);
            let second = scope.spawn(build);
            let first = first.join().unwrap();
            let second = second.join().unwrap();
            // A clear between publications may select a new entry. Both complete
            // holds remain readable and the final published entry is reused.
            assert!(first.complete() && second.complete());
            let published = assets
                .illuminants_with(|| panic!("complete publication absent"))
                .unwrap();
            assert!(Arc::ptr_eq(&first, &published) || Arc::ptr_eq(&second, &published));
        });
    }

    #[test]
    fn hull_warm_acquisition_skips_sources_and_release_retains_backing() {
        let assets = assets();
        let held = assets.input_hull().unwrap();
        let pointer = held.hull().xy().as_ptr();
        let weak = Arc::downgrade(&held);
        assets.release_cached_payloads().unwrap();
        let warm = assets
            .input_hull_with(|| panic!("warm hull visited source or mathematics"))
            .unwrap();
        assert!(Arc::ptr_eq(&held, &warm));
        assert_eq!(warm.hull().xy().as_ptr(), pointer);
        assert_eq!(warm.hull().xy()[0], warm.hull().xy()[1024]);
        drop(assets);
        drop(held);
        assert!(weak.upgrade().is_some());
        let address = pointer as usize;
        thread::spawn(move || {
            assert_eq!(warm.hull().xy().as_ptr() as usize, address);
            drop(warm);
        })
        .join()
        .unwrap();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn output_warm_identity_order_and_own_failure_do_not_substitute() {
        let assets = assets();
        let transform = transform();
        let source = source(&transform);
        let held = assets
            .output_boundary(source, transform.contract_hash())
            .unwrap();
        let pointer = held.cmax().as_ptr();
        for _ in 0..100 {
            let warm = assets
                .output_boundary_with(source, transform.contract_hash(), || {
                    panic!("warm boundary allocated or derived")
                })
                .unwrap();
            assert!(Arc::ptr_eq(&held, &warm));
            assert_eq!(pointer, warm.cmax().as_ptr());
        }
        let wrong = gamut::BoundaryInput {
            transform_hash: source.transform_hash ^ 1,
            ..source
        };
        let mismatch = assets.output_boundary(wrong, transform.contract_hash() ^ 1);
        assert!(matches!(
            mismatch,
            Err(AssetError::Derived(DerivedError::Resource(
                "ResourceDescriptorMismatch component=output_boundary_cache field=published_identity"
            )))
        ));
        let mismatch = assets.output_boundary(source, transform.contract_hash() ^ 1);
        assert!(matches!(
            mismatch,
            Err(AssetError::Derived(DerivedError::Resource(
                "ResourceDescriptorMismatch component=output_boundary_cache field=contract_identity"
            )))
        ));
        assets.release_cached_payloads().unwrap();
        let failed = assets.output_boundary_with(source, transform.contract_hash(), || {
            assets
                .output_boundary(source, transform.contract_hash())
                .unwrap();
            Err(gamut::BoundaryError::AllocationFailure)
        });
        assert!(matches!(
            failed,
            Err(AssetError::Derived(DerivedError::Boundary(
                gamut::BoundaryError::AllocationFailure
            )))
        ));
        assert_eq!(pointer, held.cmax().as_ptr());
    }

    #[test]
    fn stale_output_success_is_private_and_generation_exhaustion_preserves_slots() {
        let assets = assets();
        let transform = transform();
        let source = source(&transform);
        let stale = assets
            .output_boundary_with(source, transform.contract_hash(), || {
                assets.release_cached_payloads().unwrap();
                assets.release_cached_payloads().unwrap();
                assets
                    .output_boundary(source, transform.contract_hash())
                    .unwrap();
                gamut::build_output_boundary(source)
            })
            .unwrap();
        let winner = assets
            .output_boundary(source, transform.contract_hash())
            .unwrap();
        assert!(!Arc::ptr_eq(&stale, &winner));
        assert_eq!(stale.hash(), winner.hash());
        assert_eq!(stale.cmax(), winner.cmax());
        assets.output_boundaries.lock().unwrap().generation = u64::MAX;
        let profile = assets.film("kodak_portra_400").unwrap();
        let result = assets.release_cached_payloads();
        assert!(matches!(
            result,
            Err(AssetError::Derived(DerivedError::Internal(
                "output boundary cache generation exhausted"
            )))
        ));
        assert_eq!(
            assets.output_boundaries.lock().unwrap().generation,
            u64::MAX
        );
        assert!(Arc::ptr_eq(
            &winner,
            &assets
                .output_boundary(source, transform.contract_hash())
                .unwrap()
        ));
        assert!(!Arc::ptr_eq(
            &profile,
            &assets.film("kodak_portra_400").unwrap()
        ));
    }

    #[test]
    fn earlier_cleanup_failure_wins_while_output_and_sources_still_release() {
        let assets = assets();
        let transform = transform();
        let held = assets
            .output_boundary(source(&transform), transform.contract_hash())
            .unwrap();
        let profile = assets.film("kodak_portra_400").unwrap();
        let _ = std::panic::catch_unwind(|| {
            let _guard = assets.illuminants.lock().unwrap();
            panic!("bounded illuminant cleanup poison");
        });
        let result = assets.release_cached_payloads();
        assert!(matches!(
            result,
            Err(AssetError::Derived(DerivedError::Internal(
                "illuminant cache poisoned"
            )))
        ));
        assert_eq!(assets.output_boundaries.lock().unwrap().generation, 1);
        assert!(
            assets
                .output_boundaries
                .lock()
                .unwrap()
                .slots
                .iter()
                .all(Option::is_none)
        );
        assert!(!Arc::ptr_eq(
            &profile,
            &assets.film("kodak_portra_400").unwrap()
        ));
        assert_eq!(held.cmax().len(), gamut::OUTPUT_BOUNDARY_VALUE_COUNT);
    }
}
