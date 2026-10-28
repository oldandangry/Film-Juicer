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
impl std::error::Error for DerivedError {}
impl From<DerivedError> for AssetError {
    fn from(error: DerivedError) -> Self {
        Self::Derived(error)
    }
}

/// Seven canonical-position curves, allowing the established partial result.
/// The substantial allocation is fallible and remains stable in every live hold.
pub struct Illuminants {
    curves: Vec<Option<[f32; 81]>>,
    #[cfg(feature = "test-support")]
    _lifetime: Option<super::testing::EntryObservation>,
}

/// A fallibly reserved substantial entry, shared without copying its coordinates.
/// The small Arc allocation owns only this Vec header.
pub struct InputHull {
    hull: Vec<gamut::Hull>,
    #[cfg(feature = "test-support")]
    _lifetime: Option<super::testing::EntryObservation>,
}

/// Asset cache hold around the accepted mathematical boundary. Its Cmax buffer
/// remains the leaf producer's allocation; acquisition never clones that buffer.
pub struct OutputBoundary {
    boundary: gamut::OutputBoundary,
    #[cfg(feature = "test-support")]
    _lifetime: Option<super::testing::EntryObservation>,
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
    #[cfg(feature = "test-support")] observer: Option<&dyn super::testing::IlluminantObserver>,
    #[cfg(feature = "test-support")] lens_prepared: bool,
) -> Result<Option<Arc<crate::data_io::CsvPairs>>, AssetError> {
    #[cfg(feature = "test-support")]
    if let Some(observer) = observer {
        observer.before_source(source, lens_prepared)?;
    }
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

#[cfg(feature = "test-support")]
#[derive(Clone, Copy)]
pub enum CurveFault {
    Capacity,
    Panic,
    RowIntegrity,
}

impl Assets {
    #[cfg(feature = "test-support")]
    pub fn arm_illuminant_curve_fault(
        &self,
        source: CsvSource,
        fault: Option<CurveFault>,
    ) -> Result<(), AssetError> {
        *self
            .curve_fault
            .lock()
            .map_err(|_| DerivedError::Internal("illuminant observation poisoned"))? =
            fault.map(|fault| (source, fault));
        Ok(())
    }
    #[cfg(feature = "test-support")]
    pub fn illuminant_curve_fault_pending(&self) -> Result<bool, AssetError> {
        Ok(self
            .curve_fault
            .lock()
            .map_err(|_| DerivedError::Internal("illuminant observation poisoned"))?
            .is_some())
    }
    #[cfg(feature = "test-support")]
    fn visit_illuminant_curve(
        &self,
        source: CsvSource,
        label: &'static str,
    ) -> Result<(), AssetError> {
        let fault = {
            let mut pending = self
                .curve_fault
                .lock()
                .map_err(|_| DerivedError::Internal("illuminant observation poisoned"))?;
            if pending
                .as_ref()
                .is_some_and(|(selected, _)| *selected == source)
            {
                pending.take().map(|(_, fault)| fault)
            } else {
                None
            }
        };
        match fault {
            Some(CurveFault::Capacity) => {
                let error = illuminant::test_support::capacity_failure()
                    .expect_err("unaddressable scratch reservation must fail");
                Err(DerivedError::Illuminant { label, error }.into())
            }
            Some(CurveFault::Panic) => panic!("illuminant derived-curve acquisition panic"),
            Some(CurveFault::RowIntegrity) => {
                Err(DerivedError::Internal("illuminant decoded-row integrity failure").into())
            }
            None => Ok(()),
        }
    }
    pub fn illuminants(&self) -> Result<Arc<Illuminants>, AssetError> {
        self.illuminants_observed_inner(
            #[cfg(feature = "test-support")]
            None,
        )
    }
    #[cfg(feature = "test-support")]
    pub fn illuminants_observed(
        &self,
        observer: &dyn super::testing::IlluminantObserver,
    ) -> Result<Arc<Illuminants>, AssetError> {
        self.illuminants_observed_inner(Some(observer))
    }
    #[cfg(feature = "test-support")]
    fn visit_illuminant_operation(
        &self,
        operation: super::testing::IlluminantOperation,
        label: &'static str,
    ) -> Result<(), AssetError> {
        use super::testing::IlluminantFault;
        match self.cache_observations.illuminant_fault(operation) {
            None => Ok(()),
            Some(IlluminantFault::Unsupported) => {
                Err(DerivedError::Unsupported("illuminant injected unsupported input").into())
            }
            Some(IlluminantFault::Panic) => panic!("illuminant cache-owned leaf panic"),
            Some(IlluminantFault::Capacity) => illuminant::test_support::capacity_failure()
                .map_err(|error| DerivedError::Illuminant { label, error }.into()),
        }
    }
    fn illuminants_observed_inner(
        &self,
        #[cfg(feature = "test-support")] observer: Option<&dyn super::testing::IlluminantObserver>,
    ) -> Result<Arc<Illuminants>, AssetError> {
        self.illuminants_with(
            || {
                let mut curves = Vec::new();
                curves
                    .try_reserve_exact(7)
                    .map_err(|_| DerivedError::Allocation("illuminant curve allocation failed"))?;
                let mut candidate = Illuminants {
                    #[cfg(feature = "test-support")]
                    _lifetime: Some(self.cache_observations.track(
                        super::testing::CacheFamily::Illuminants,
                        curves.as_ptr() as usize,
                        curves.capacity() * std::mem::size_of::<Option<[f32; 81]>>(),
                    )),
                    curves,
                };
                for (source, label) in [
                    (CsvSource::D65, "D65"),
                    (CsvSource::D55, "D55"),
                    (CsvSource::D50, "D50"),
                    (CsvSource::T, "T"),
                    (CsvSource::K75p, "K75P"),
                ] {
                    candidate.curves.push(
                        match available_rows(
                            self,
                            source,
                            #[cfg(feature = "test-support")]
                            observer,
                            #[cfg(feature = "test-support")]
                            false,
                        )? {
                            Some(rows) => {
                                #[cfg(feature = "test-support")]
                                self.visit_illuminant_curve(source, label)?;
                                #[cfg(feature = "test-support")]
                                self.cache_observations
                                    .derive(super::testing::CacheFamily::Illuminants);
                                #[cfg(feature = "test-support")]
                                self.visit_illuminant_operation(
                                    super::testing::IlluminantOperation::Samples,
                                    label,
                                )?;
                                available_curve(illuminant::from_samples(rows.rows()), label)?
                            }
                            None => None,
                        },
                    );
                }
                let mut kg3 = available_rows(
                    self,
                    CsvSource::Kg3,
                    #[cfg(feature = "test-support")]
                    observer,
                    #[cfg(feature = "test-support")]
                    false,
                )?;
                let filtered = match &kg3 {
                    Some(rows) => {
                        #[cfg(feature = "test-support")]
                        self.cache_observations
                            .derive(super::testing::CacheFamily::Illuminants);
                        #[cfg(feature = "test-support")]
                        self.visit_illuminant_operation(
                            super::testing::IlluminantOperation::TungstenKg3,
                            "KG3",
                        )?;
                        available_curve(
                            illuminant::tungsten_kg3(rows.rows()).map(|(curve, _)| curve),
                            "KG3",
                        )?
                    }
                    None => {
                        kg3 = available_rows(
                            self,
                            CsvSource::Kg3,
                            #[cfg(feature = "test-support")]
                            observer,
                            #[cfg(feature = "test-support")]
                            false,
                        )?;
                        None
                    }
                };
                candidate.curves.push(filtered);
                let mut lens_curve = None;
                if let Some(rows) = kg3 {
                    #[cfg(feature = "test-support")]
                    self.cache_observations
                        .derive(super::testing::CacheFamily::Illuminants);
                    #[cfg(feature = "test-support")]
                    self.visit_illuminant_operation(
                        super::testing::IlluminantOperation::LensPrepare,
                        "KG3",
                    )?;
                    match illuminant::prepare_lens(rows.rows()) {
                        Ok((input, _)) => {
                            if let Some(lens) = available_rows(
                                self,
                                CsvSource::Canon24F28Is,
                                #[cfg(feature = "test-support")]
                                observer,
                                #[cfg(feature = "test-support")]
                                true,
                            )? {
                                #[cfg(feature = "test-support")]
                                self.cache_observations
                                    .derive(super::testing::CacheFamily::Illuminants);
                                #[cfg(feature = "test-support")]
                                self.visit_illuminant_operation(
                                    super::testing::IlluminantOperation::LensFinish,
                                    "Canon 24 F2.8 IS",
                                )?;
                                lens_curve = available_curve(
                                    illuminant::finish_lens(input, lens.rows())
                                        .map(|(curve, _)| curve),
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
            },
            #[cfg(feature = "test-support")]
            observer,
        )
    }

    fn illuminants_with(
        &self,
        build: impl FnOnce() -> Result<Illuminants, AssetError>,
        #[cfg(feature = "test-support")] observer: Option<&dyn super::testing::IlluminantObserver>,
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
        #[cfg(feature = "test-support")]
        let candidate = {
            let mut candidate = candidate;
            let address = Arc::as_ptr(&candidate) as usize;
            let bytes = candidate.curves.capacity() * std::mem::size_of::<Option<[f32; 81]>>();
            let value = Arc::get_mut(&mut candidate).expect("unpublished entry is unique");
            if let Some(lifetime) = &mut value._lifetime {
                lifetime.identify(address);
            } else {
                value._lifetime = Some(self.cache_observations.track(
                    super::testing::CacheFamily::Illuminants,
                    address,
                    bytes,
                ));
            }
            candidate
        };
        #[cfg(feature = "test-support")]
        if let Some(observer) = observer {
            observer.before_publication(
                Arc::as_ptr(&candidate) as usize,
                candidate.curves.capacity() * std::mem::size_of::<Option<[f32; 81]>>(),
            )?;
        }
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
        self.input_hull_observed_inner(
            #[cfg(feature = "test-support")]
            None,
        )
    }
    #[cfg(feature = "test-support")]
    pub fn input_hull_observed(
        &self,
        observer: &dyn super::testing::HullObserver,
    ) -> Result<Arc<InputHull>, AssetError> {
        self.input_hull_observed_inner(Some(observer))
    }
    fn input_hull_observed_inner(
        &self,
        #[cfg(feature = "test-support")] observer: Option<&dyn super::testing::HullObserver>,
    ) -> Result<Arc<InputHull>, AssetError> {
        self.input_hull_with(|| {
            let curves = self.illuminants()?;
            let d65 = curves.d65().ok_or(DerivedError::Resource("MalformedRequiredResource component=input_gamut_hull requirement=cie1931_380_700_and_d65"))?;
            let cmf = self.cmf()?;
            if cmf.rows().len() != 81 {
                return Err(DerivedError::Resource("MalformedRequiredResource component=input_gamut_hull requirement=cie1931_380_700_and_d65").into());
            }
            let x = std::array::from_fn(|i| cmf.rows()[i][1]);
            #[cfg(feature = "test-support")]
            let x = {
                let mut x = x;
                if let Some(add) = self.cache_observations.hull_tail_add() { x[80] += add; }
                x
            };
            let y = std::array::from_fn(|i| cmf.rows()[i][2]);
            let z = std::array::from_fn(|i| cmf.rows()[i][3]);
            #[cfg(feature = "test-support")]
            {
                use super::testing::{CacheFamily, CacheFault};
                match self.cache_observations.hull_fault() {
                    Some(super::testing::HullFault::Unsupported) => return Err(DerivedError::Unsupported("input hull boundary fault").into()),
                    Some(super::testing::HullFault::Panic) => panic!("input hull boundary fault"),
                    Some(super::testing::HullFault::Preparation(error)) => return Err(DerivedError::Hull(error).into()),
                    None => (),
                }
                match self.cache_observations.fault(CacheFamily::InputHull) {
                    Some(CacheFault::Unsupported) => return Err(DerivedError::Unsupported("input hull injected unsupported input").into()),
                    Some(CacheFault::Panic) => panic!("input hull derivation panic"),
                    Some(CacheFault::Capacity) => { reserve_hull_entry(usize::MAX)?; },
                    Some(CacheFault::Preparation) => return Err(DerivedError::Resource("input hull derivation failure").into()),
                    None => (),
                }
                self.cache_observations.derive(CacheFamily::InputHull);
            }
            gamut::build_input_hull(gamut::HullInput { x: &x, y: &y, z: &z, d65 }).map_err(|error| DerivedError::Hull(error).into())
        }, #[cfg(feature = "test-support")] observer)
    }

    fn input_hull_with(
        &self,
        build: impl FnOnce() -> Result<gamut::Hull, AssetError>,
        #[cfg(feature = "test-support")] observer: Option<&dyn super::testing::HullObserver>,
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
        #[cfg(feature = "test-support")]
        if self.cache_observations.hull_entry_reservation_fault() {
            reserve_hull_entry(usize::MAX)?;
        }
        let hull = reserve_hull_entry(1)?;
        let mut value = InputHull {
            hull,
            #[cfg(feature = "test-support")]
            _lifetime: None,
        };
        #[cfg(feature = "test-support")]
        {
            value._lifetime = Some(self.cache_observations.track(
                super::testing::CacheFamily::InputHull,
                value.hull.as_ptr() as usize,
                value.hull.capacity() * std::mem::size_of::<gamut::Hull>(),
            ));
            if let Some(observer) = observer {
                observer.before_build(
                    value.hull.as_ptr() as usize,
                    value.hull.capacity() * std::mem::size_of::<gamut::Hull>(),
                )?;
            }
        }
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
        self.output_boundary_with(
            source,
            expected_contract_hash,
            || gamut::build_output_boundary(source),
            #[cfg(feature = "test-support")]
            None,
        )
    }

    #[cfg(feature = "test-support")]
    pub fn output_boundary_observed(
        &self,
        source: gamut::BoundaryInput<'_>,
        expected_contract_hash: u64,
        observer: &dyn super::testing::OutputObserver,
    ) -> Result<Arc<OutputBoundary>, AssetError> {
        self.output_boundary_with(
            source,
            expected_contract_hash,
            || gamut::build_output_boundary(source),
            Some(observer),
        )
    }

    fn output_boundary_with(
        &self,
        source: gamut::BoundaryInput<'_>,
        expected_contract_hash: u64,
        build: impl FnOnce() -> Result<gamut::OutputBoundary, gamut::BoundaryError>,
        #[cfg(feature = "test-support")] observer: Option<&dyn super::testing::OutputObserver>,
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
            #[cfg(feature = "test-support")]
            if let Some(observer) = observer {
                observer.after_selection(true, generation)?;
            }
            matches(&selected)?;
            return Ok(selected);
        }
        #[cfg(feature = "test-support")]
        {
            use super::testing::{CacheFamily, CacheFault};
            if let Some(observer) = observer {
                observer.after_miss(generation)?;
            }
            match self.cache_observations.fault(CacheFamily::OutputBoundary) {
                Some(CacheFault::Unsupported) => {
                    return Err(DerivedError::Unsupported(
                        "output cache injected unsupported input",
                    )
                    .into());
                }
                Some(CacheFault::Panic) => panic!("output cache derivation panic"),
                Some(CacheFault::Capacity) => {
                    return Err(
                        DerivedError::Boundary(gamut::fixtures::output_reserve_failure()).into(),
                    );
                }
                Some(CacheFault::Preparation) => {
                    return Err(DerivedError::Resource("output cache derivation failure").into());
                }
                None => (),
            }
            self.cache_observations.derive(CacheFamily::OutputBoundary);
        }
        // Own failure wins even when a contender has published meanwhile.
        let candidate = Arc::new(OutputBoundary {
            boundary: build().map_err(DerivedError::Boundary)?,
            #[cfg(feature = "test-support")]
            _lifetime: None,
        });
        #[cfg(feature = "test-support")]
        let candidate = {
            let mut candidate = candidate;
            let lifetime = self.cache_observations.track(
                super::testing::CacheFamily::OutputBoundary,
                candidate.boundary() as *const gamut::OutputBoundary as usize,
                gamut::fixtures::output_allocation_bytes(candidate.boundary()),
            );
            Arc::get_mut(&mut candidate)
                .expect("unpublished entry is unique")
                ._lifetime = Some(lifetime);
            candidate
        };
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
        #[cfg(feature = "test-support")]
        if let Some(observer) = observer {
            observer.after_selection(false, generation)?;
        }
        matches(&selected)?;
        Ok(selected)
    }

    pub(super) fn release_illuminants(&self) -> Result<(), AssetError> {
        #[cfg(feature = "test-support")]
        self.cache_observations.before_illuminant_clear()?;
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

    #[cfg(feature = "test-support")]
    #[test]
    fn hull_admission_faults_preserve_typed_causes_and_warm_owner_isolation() {
        use super::super::testing::{CacheFamily, HullFault};
        for cause in [
            gamut::HullError::LocusAndCenter,
            gamut::HullError::RayIntersections,
            gamut::HullError::InscribedScale,
            gamut::HullError::Identity,
        ] {
            let owner = assets();
            owner.arm_hull_fault(Some(HullFault::Preparation(cause)));
            let error = owner.input_hull().err().expect("actual cold hull visit");
            assert!(
                matches!(error, AssetError::Derived(DerivedError::Hull(actual)) if actual == cause)
            );
            assert_eq!(owner.take_cache_fault_consumed(CacheFamily::InputHull), 1);
            let recovered = owner.input_hull().unwrap();
            owner.arm_hull_fault(Some(HullFault::Unsupported));
            assert!(Arc::ptr_eq(&recovered, &owner.input_hull().unwrap()));
            assert_eq!(owner.take_cache_fault_consumed(CacheFamily::InputHull), 0);
            assert!(owner.hull_fault_pending());
            let other = assets();
            assert!(other.input_hull().is_ok());
            assert!(owner.hull_fault_pending());
            owner.arm_hull_fault(None);
            assert!(!owner.hull_fault_pending());
        }
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn cache_owned_illuminant_faults_are_lazy_owner_scoped_and_consumed_once() {
        use super::super::testing::{CacheFamily, IlluminantFault, IlluminantOperation};
        for operation in [
            IlluminantOperation::Samples,
            IlluminantOperation::TungstenKg3,
            IlluminantOperation::LensPrepare,
            IlluminantOperation::LensFinish,
        ] {
            for fault in [IlluminantFault::Unsupported, IlluminantFault::Capacity] {
                let owner = assets();
                let warm = owner.illuminants().unwrap();
                owner.arm_illuminant_operation(operation, Some(fault));
                assert!(Arc::ptr_eq(&warm, &owner.illuminants().unwrap()));
                assert_eq!(owner.illuminant_operation_fault_consumed(), 0);
                let other = assets();
                assert!(other.illuminants().is_ok());
                assert_eq!(owner.illuminant_operation_fault_consumed(), 0);
                owner.release_cached_payloads().unwrap();
                let error = owner
                    .illuminants()
                    .err()
                    .expect("cold owner visits the armed leaf");
                assert!(matches!(
                    error,
                    AssetError::Derived(DerivedError::Unsupported(_))
                        | AssetError::Derived(DerivedError::Illuminant { .. })
                ));
                assert_eq!(owner.illuminant_operation_fault_consumed(), 1);
                let recovered = owner.illuminants().unwrap();
                assert_eq!(owner.illuminant_operation_fault_consumed(), 1);
                owner.release_cached_payloads().unwrap();
                drop(warm);
                drop(recovered);
                let metrics = owner.cache_metrics(CacheFamily::Illuminants);
                assert_eq!(metrics.allocations, 3);
                assert_eq!(metrics.destructions, 3);
                assert_eq!(metrics.live_bytes, 0);
            }
        }
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn real_backing_drop_reenters_without_retaining_assets() {
        use super::super::testing::CacheFamily;
        let assets = Arc::new(assets());
        assets.observe_cache_owner();
        let observation = assets.cache_observation();
        let held = assets.illuminants().unwrap();
        assets.release_cached_payloads().unwrap();
        assert!(observation.metrics(CacheFamily::Illuminants).live_bytes > 0);
        assets.arm_cache_drop_reentry(CacheFamily::Illuminants);
        drop(held);
        let metrics = observation.metrics(CacheFamily::Illuminants);
        assert_eq!(
            (
                metrics.live_bytes,
                metrics.reentries,
                metrics.reentry_failures
            ),
            (0, 1, 0)
        );
        let hull = assets.input_hull().unwrap();
        let weak = Arc::downgrade(&assets);
        drop(assets);
        assert!(weak.upgrade().is_none());
        assert!(observation.metrics(CacheFamily::InputHull).live_bytes > 0);
        assert_ne!(hull.hull().hash(), 0);
        drop(hull);
        let metrics = observation.metrics(CacheFamily::InputHull);
        assert_eq!((metrics.live_bytes, metrics.destructions), (0, 1));
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn early_curve_fault_reclaims_the_actual_derived_allocation() {
        use super::super::testing::CacheFamily;
        let assets = assets();
        let observation = assets.cache_observation();
        assets
            .arm_illuminant_curve_fault(CsvSource::D55, Some(CurveFault::Capacity))
            .unwrap();
        assert!(assets.illuminants().is_err());
        assert!(!assets.illuminant_curve_fault_pending().unwrap());
        let metrics = observation.metrics(CacheFamily::Illuminants);
        assert_eq!(
            (
                metrics.allocations,
                metrics.destructions,
                metrics.live_bytes
            ),
            (1, 1, 0)
        );
        assert!(metrics.peak_bytes >= 7 * 81 * 4);
        assert!(assets.illuminants().unwrap().complete());
        assets.release_cached_payloads().unwrap();
        let metrics = observation.metrics(CacheFamily::Illuminants);
        assert_eq!(
            (
                metrics.allocations,
                metrics.destructions,
                metrics.live_bytes
            ),
            (2, 2, 0)
        );
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn actual_illuminant_loser_drop_reenters_after_selection() {
        use super::super::testing::{CacheFamily, IlluminantObserver};
        struct Publication(Barrier);
        impl IlluminantObserver for Publication {
            fn before_source(&self, _: CsvSource, _: bool) -> Result<(), DerivedError> {
                Ok(())
            }
            fn before_publication(&self, _: usize, _: usize) -> Result<(), DerivedError> {
                self.0.wait();
                Ok(())
            }
        }
        let assets = Arc::new(assets());
        assets.observe_cache_owner();
        assets.arm_cache_drop_reentry(CacheFamily::Illuminants);
        let observation = assets.cache_observation();
        let gate = Publication(Barrier::new(2));
        let (first, second) = thread::scope(|scope| {
            let first = scope.spawn(|| assets.illuminants_observed(&gate).unwrap());
            let second = scope.spawn(|| assets.illuminants_observed(&gate).unwrap());
            (first.join().unwrap(), second.join().unwrap())
        });
        assert!(Arc::ptr_eq(&first, &second));
        let metrics = observation.metrics(CacheFamily::Illuminants);
        assert_eq!(
            (
                metrics.allocations,
                metrics.destructions,
                metrics.reentries,
                metrics.reentry_failures
            ),
            (2, 1, 1, 0)
        );
        assert!(metrics.peak_bytes > metrics.live_bytes);
        drop(first);
        drop(second);
        assert_eq!(observation.metrics(CacheFamily::Illuminants).live_bytes, 0);
    }
    fn complete() -> Illuminants {
        Illuminants {
            curves: vec![Some([1.0; 81]); 7],
            #[cfg(feature = "test-support")]
            _lifetime: None,
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
            .illuminants_with(
                || {
                    let mut curves = complete();
                    curves.curves[1] = None;
                    Ok(curves)
                },
                #[cfg(feature = "test-support")]
                None,
            )
            .unwrap();
        assert!(partial.d55().is_none());
        assert!(assets.illuminants.lock().unwrap().is_none());
        let held = assets.illuminants().unwrap();
        assert!(held.complete());
        let pointer = held.d65().unwrap().as_ptr();
        for _ in 0..100 {
            let warm = assets
                .illuminants_with(
                    || panic!("warm acquisition rebuilt curves"),
                    #[cfg(feature = "test-support")]
                    None,
                )
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
                    .illuminants_with(
                        || {
                            barrier.wait();
                            // A release during either build must not deadlock or prevent
                            // later publication in this family's distinct protocol.
                            assets.release_cached_payloads()?;
                            Ok(complete())
                        },
                        #[cfg(feature = "test-support")]
                        None,
                    )
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
                .illuminants_with(
                    || panic!("complete publication absent"),
                    #[cfg(feature = "test-support")]
                    None,
                )
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
            .input_hull_with(
                || panic!("warm hull visited source or mathematics"),
                #[cfg(feature = "test-support")]
                None,
            )
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
                .output_boundary_with(
                    source,
                    transform.contract_hash(),
                    || panic!("warm boundary allocated or derived"),
                    #[cfg(feature = "test-support")]
                    None,
                )
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
        let failed = assets.output_boundary_with(
            source,
            transform.contract_hash(),
            || {
                assets
                    .output_boundary(source, transform.contract_hash())
                    .unwrap();
                Err(gamut::BoundaryError::AllocationFailure)
            },
            #[cfg(feature = "test-support")]
            None,
        );
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
            .output_boundary_with(
                source,
                transform.contract_hash(),
                || {
                    assets.release_cached_payloads().unwrap();
                    assets.release_cached_payloads().unwrap();
                    assets
                        .output_boundary(source, transform.contract_hash())
                        .unwrap();
                    gamut::build_output_boundary(source)
                },
                #[cfg(feature = "test-support")]
                None,
            )
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
