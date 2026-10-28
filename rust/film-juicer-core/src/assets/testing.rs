//! Finite observations of the three asset-owned caches. These controls are absent
//! from default builds; counters describe Rust entries, never foreign wrappers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, ThreadId};

use super::{AssetError, Assets, CsvSource, DerivedError};
use crate::gamut::HullError;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheFamily {
    Illuminants,
    InputHull,
    OutputBoundary,
}

#[derive(Clone, Copy)]
pub enum CacheFault {
    Unsupported,
    Panic,
    Capacity,
    Preparation,
}

#[derive(Clone, Copy)]
pub enum IlluminantFault {
    Unsupported,
    Panic,
    Capacity,
}

/// Cache-owned leaf boundaries; blackbody/equal-energy remain native callers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IlluminantOperation {
    Samples,
    TungstenKg3,
    LensPrepare,
    LensFinish,
}

#[derive(Clone, Copy)]
pub enum HullFault {
    Unsupported,
    Panic,
    Preparation(HullError),
}

/// The existing output-cache rendezvous, borrowed only during a synchronous
/// acquire. Implementations must contain foreign callback failures themselves.
pub trait OutputObserver {
    fn after_miss(&self, generation: u64) -> Result<(), DerivedError>;
    fn after_selection(&self, warm: bool, generation: u64) -> Result<(), DerivedError>;
}

/// The existing source-acquisition and candidate-publication observations. Both
/// methods run outside cache locks; the observer is never stored in an entry.
pub trait IlluminantObserver {
    fn before_source(&self, source: CsvSource, lens_prepared: bool) -> Result<(), DerivedError>;
    fn before_publication(&self, entry: usize, bytes: usize) -> Result<(), DerivedError>;
}

pub trait HullObserver {
    fn before_build(&self, entry: usize, bytes: usize) -> Result<(), DerivedError>;
}

#[derive(Clone, Copy, Default)]
pub struct CacheMetrics {
    pub allocations: u64,
    pub derivations: u64,
    pub destructions: u64,
    pub live_bytes: u64,
    pub peak_bytes: u64,
    pub reentries: u64,
    pub reentry_failures: u64,
}

#[derive(Default)]
struct State {
    metrics: [CacheMetrics; 3],
    live: Vec<(usize, CacheFamily)>,
    faults: HashMap<(ThreadId, CacheFamily), CacheFault>,
    consumed: HashMap<(ThreadId, CacheFamily), u64>,
    illuminant_faults: HashMap<ThreadId, (IlluminantOperation, IlluminantFault)>,
    illuminant_consumed: HashMap<ThreadId, u32>,
    owner: Weak<Assets>,
    reenter: Option<CacheFamily>,
    illuminant_clear_failure: bool,
    late_clear_failure: bool,
    hull_entry_faults: HashMap<ThreadId, bool>,
    hull_faults: HashMap<ThreadId, HullFault>,
    hull_tail_add: Option<f32>,
}

#[derive(Default)]
pub(super) struct Observations(Mutex<State>);

/// Retains observation storage only. Its Weak target never extends Assets or
/// any cache entry, so final-hold reclamation can be read after Assets expires.
pub struct CacheObservation(Arc<Observations>);
impl CacheObservation {
    pub fn metrics(&self, family: CacheFamily) -> CacheMetrics {
        self.0.0.lock().expect("cache observation poisoned").metrics[family as usize]
    }
    pub fn entry_live(&self, family: CacheFamily, address: usize) -> bool {
        self.0
            .0
            .lock()
            .expect("cache observation poisoned")
            .live
            .contains(&(address, family))
    }
}

impl Observations {
    pub(super) fn hull_entry_reservation_fault(&self) -> bool {
        let mut state = self.0.lock().expect("cache observation poisoned");
        match state.hull_entry_faults.get_mut(&thread::current().id()) {
            Some(armed) => std::mem::take(armed),
            None => false,
        }
    }
    pub(super) fn hull_fault(&self) -> Option<HullFault> {
        let mut state = self.0.lock().expect("cache observation poisoned");
        let key = thread::current().id();
        let fault = state.hull_faults.remove(&key);
        if fault.is_some() {
            *state
                .consumed
                .entry((key, CacheFamily::InputHull))
                .or_default() += 1;
        }
        fault
    }
    pub(super) fn hull_tail_add(&self) -> Option<f32> {
        self.0
            .lock()
            .expect("cache observation poisoned")
            .hull_tail_add
    }
    pub(super) fn illuminant_fault(
        &self,
        operation: IlluminantOperation,
    ) -> Option<IlluminantFault> {
        let mut state = self.0.lock().expect("cache observation poisoned");
        let key = thread::current().id();
        if state
            .illuminant_faults
            .get(&key)
            .is_some_and(|(selected, _)| *selected == operation)
        {
            let (_, fault) = state.illuminant_faults.remove(&key).unwrap();
            *state.illuminant_consumed.entry(key).or_default() += 1;
            Some(fault)
        } else {
            None
        }
    }
    pub(super) fn derive(&self, family: CacheFamily) {
        self.0.lock().expect("cache observation poisoned").metrics[family as usize].derivations +=
            1;
    }

    pub(super) fn fault(&self, family: CacheFamily) -> Option<CacheFault> {
        let mut state = self.0.lock().expect("cache observation poisoned");
        let key = (thread::current().id(), family);
        let fault = state.faults.remove(&key);
        if fault.is_some() {
            *state.consumed.entry(key).or_default() += 1;
        }
        fault
    }

    pub(super) fn track(
        self: &Arc<Self>,
        family: CacheFamily,
        address: usize,
        bytes: usize,
    ) -> EntryObservation {
        let mut state = self.0.lock().expect("cache observation poisoned");
        let metric = &mut state.metrics[family as usize];
        metric.allocations += 1;
        metric.live_bytes += bytes as u64;
        metric.peak_bytes = metric.peak_bytes.max(metric.live_bytes);
        state.live.push((address, family));
        EntryObservation {
            owner: Arc::clone(self),
            family,
            address,
            bytes,
        }
    }

    pub(super) fn before_illuminant_clear(&self) -> Result<(), AssetError> {
        let fail = std::mem::take(
            &mut self
                .0
                .lock()
                .expect("cache observation poisoned")
                .illuminant_clear_failure,
        );
        if fail {
            Err(DerivedError::Allocation("host cache cleanup allocation failed").into())
        } else {
            Ok(())
        }
    }

    pub(super) fn after_payload_clear(&self) -> Result<(), AssetError> {
        let fail = std::mem::take(
            &mut self
                .0
                .lock()
                .expect("cache observation poisoned")
                .late_clear_failure,
        );
        if fail {
            Err(DerivedError::Internal("injected post-release cache failure").into())
        } else {
            Ok(())
        }
    }
}

pub(super) struct EntryObservation {
    owner: Arc<Observations>,
    family: CacheFamily,
    address: usize,
    bytes: usize,
}

impl EntryObservation {
    pub(super) fn identify(&mut self, address: usize) {
        let mut state = self.owner.0.lock().expect("cache observation poisoned");
        let entry = state
            .live
            .iter_mut()
            .find(|entry| **entry == (self.address, self.family))
            .expect("live unpublished entry observation");
        entry.0 = address;
        self.address = address;
    }
}

impl Drop for EntryObservation {
    fn drop(&mut self) {
        let reenter = {
            let mut state = self.owner.0.lock().expect("cache observation poisoned");
            let metric = &mut state.metrics[self.family as usize];
            metric.destructions += 1;
            metric.live_bytes -= self.bytes as u64;
            state
                .live
                .retain(|&(address, family)| (address, family) != (self.address, self.family));
            if state.reenter == Some(self.family) {
                state.reenter = None;
                state.owner.upgrade()
            } else {
                None
            }
        };
        if let Some(owner) = reenter {
            // This is the actual cache operation, after the observation lock
            // and the cache publication/detachment locks have been released.
            let failed = owner.release_cached_payloads().is_err();
            let mut state = self.owner.0.lock().expect("cache observation poisoned");
            let metric = &mut state.metrics[self.family as usize];
            metric.reentries += 1;
            metric.reentry_failures += u64::from(failed);
        }
    }
}

impl Assets {
    pub fn arm_illuminant_operation(
        &self,
        operation: IlluminantOperation,
        fault: Option<IlluminantFault>,
    ) {
        let mut state = self
            .cache_observations
            .0
            .lock()
            .expect("cache observation poisoned");
        let key = thread::current().id();
        if let Some(fault) = fault {
            state.illuminant_faults.insert(key, (operation, fault));
        } else {
            state.illuminant_faults.remove(&key);
        }
        state.illuminant_consumed.insert(key, 0);
    }
    pub fn illuminant_operation_fault_consumed(&self) -> u32 {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .illuminant_consumed
            .get(&thread::current().id())
            .copied()
            .unwrap_or(0)
    }
    pub fn cache_observation(&self) -> CacheObservation {
        CacheObservation(Arc::clone(&self.cache_observations))
    }
    pub fn arm_hull_fault(&self, fault: Option<HullFault>) {
        let mut state = self
            .cache_observations
            .0
            .lock()
            .expect("cache observation poisoned");
        let key = thread::current().id();
        if let Some(fault) = fault {
            state.hull_faults.insert(key, fault);
        } else {
            state.hull_faults.remove(&key);
        }
        state.consumed.insert((key, CacheFamily::InputHull), 0);
    }
    pub fn hull_fault_pending(&self) -> bool {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .hull_faults
            .contains_key(&thread::current().id())
    }
    pub fn arm_hull_entry_reservation(&self, active: bool) {
        let mut state = self
            .cache_observations
            .0
            .lock()
            .expect("cache observation poisoned");
        let key = thread::current().id();
        if active {
            state.hull_entry_faults.insert(key, true);
        } else {
            state.hull_entry_faults.remove(&key);
        }
    }
    pub fn take_hull_entry_reservation_consumed(&self) -> bool {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .hull_entry_faults
            .remove(&thread::current().id())
            == Some(false)
    }
    /// Vary the cold operation's local source copy, never a published CMF outcome.
    pub fn set_hull_source_tail_add(&self, value: Option<f32>) {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .hull_tail_add = value;
    }
    /// Associate a weak reentry target without making an entry retain Assets.
    pub fn observe_cache_owner(self: &Arc<Self>) {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .owner = Arc::downgrade(self);
    }

    pub fn cache_metrics(&self, family: CacheFamily) -> CacheMetrics {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .metrics[family as usize]
    }

    pub fn cache_entry_live(&self, family: CacheFamily, address: usize) -> bool {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .live
            .contains(&(address, family))
    }

    pub fn arm_cache_fault(&self, family: CacheFamily, fault: Option<CacheFault>) {
        let mut state = self
            .cache_observations
            .0
            .lock()
            .expect("cache observation poisoned");
        let key = (thread::current().id(), family);
        state.consumed.remove(&key);
        if let Some(fault) = fault {
            state.faults.insert(key, fault);
        } else {
            state.faults.remove(&key);
        }
    }

    pub fn take_cache_fault_consumed(&self, family: CacheFamily) -> u64 {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .consumed
            .remove(&(thread::current().id(), family))
            .unwrap_or(0)
    }

    pub fn arm_cache_drop_reentry(&self, family: CacheFamily) {
        self.cache_observations
            .0
            .lock()
            .expect("cache observation poisoned")
            .reenter = Some(family);
    }

    pub fn arm_cache_clear_failures(&self, illuminant: bool, late: bool) {
        let mut state = self
            .cache_observations
            .0
            .lock()
            .expect("cache observation poisoned");
        state.illuminant_clear_failure = illuminant;
        state.late_clear_failure = late;
    }

    pub fn output_cache_generation(&self) -> Result<u64, AssetError> {
        Ok(self
            .output_boundaries
            .lock()
            .map_err(|_| DerivedError::Internal("output boundary cache poisoned"))?
            .generation)
    }

    /// Test callers must exclude every concurrent cache operation while setting.
    pub fn set_output_cache_generation(&self, generation: u64) -> Result<(), AssetError> {
        self.output_boundaries
            .lock()
            .map_err(|_| DerivedError::Internal("output boundary cache poisoned"))?
            .generation = generation;
        Ok(())
    }
}
