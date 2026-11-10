//! Pointer-free observations of real callback stages, available only to fixtures.
#![forbid(unsafe_code)]

use std::cell::Cell;

use crate::cuda::{edge::FjPreparedFramePolicy, sys::FjEffectsGeometry};

#[derive(Clone, Copy)]
pub(crate) struct Observation {
    pub(crate) phases: [u32; 3],
    pub(crate) count: u32,
    pub(crate) full: Option<[i32; 4]>,
    pub(crate) full_count: u32,
    pub(crate) policy: Option<(u32, FjPreparedFramePolicy)>,
    pub(crate) policy_count: u32,
    pub(crate) effects: Option<FjEffectsGeometry>,
    pub(crate) effects_count: u32,
    pub(crate) diffusion_transport_calls: u32,
    pub(crate) halation_transport_calls: u32,
}
const EMPTY: Observation = Observation {
    phases: [0; 3],
    count: 0,
    full: None,
    full_count: 0,
    policy: None,
    policy_count: 0,
    effects: None,
    effects_count: 0,
    diffusion_transport_calls: 0,
    halation_transport_calls: 0,
};
thread_local! { static OBSERVED: Cell<Observation> = const { Cell::new(EMPTY) }; }

fn phase(observed: &mut Observation, value: u32) {
    if let Some(slot) = observed.phases.get_mut(observed.count as usize) {
        *slot = value;
    }
    observed.count = observed.count.saturating_add(1);
}
pub(crate) fn reset() {
    OBSERVED.set(EMPTY);
}
pub(crate) fn read() -> Observation {
    OBSERVED.get()
}
pub(crate) fn record_full_domain(domain: film_juicer_core::frame::Domain) {
    let mut observed = read();
    observed.full = Some([domain.x, domain.y, domain.width, domain.height]);
    observed.full_count = observed.full_count.saturating_add(1);
    phase(&mut observed, 1);
    OBSERVED.set(observed);
}
pub(crate) fn record_policy(value: FjPreparedFramePolicy, route: u32) {
    let mut observed = read();
    observed.policy = Some((route, value));
    observed.policy_count = observed.policy_count.saturating_add(1);
    phase(&mut observed, 2);
    OBSERVED.set(observed);
}
pub(crate) fn record_effects_geometry(value: FjEffectsGeometry) {
    let mut observed = read();
    observed.effects = Some(value);
    observed.effects_count = observed.effects_count.saturating_add(1);
    phase(&mut observed, 3);
    OBSERVED.set(observed);
}
pub(crate) fn record_diffusion_transport() {
    let mut observed = read();
    observed.diffusion_transport_calls = observed.diffusion_transport_calls.saturating_add(1);
    OBSERVED.set(observed);
}
pub(crate) fn record_halation_transport() {
    let mut observed = read();
    observed.halation_transport_calls = observed.halation_transport_calls.saturating_add(1);
    OBSERVED.set(observed);
}
