//! Admitted Film v1 controls and their canonical Portra bank.

use std::sync::Arc;

use crate::assets::{PortraGrainBank, PortraGrainTable};
use crate::{hash, route::ScanRoute};
use super::super::density::portra;

pub const DEFAULT_CHROMA: f64 = 0.2406350076198578;

#[derive(Clone, Copy)]
pub struct Controls {
    pub amount_ev: f32,
    pub texel_pitch_um: f32,
    pub chroma_fraction: f32,
}

pub struct Recipe {
    bank: Arc<PortraGrainBank>,
    texel_pitch_um: f32,
    gain: f32,
    chroma_fraction: f32,
    hash: u64,
}

impl Controls {
    pub(crate) fn valid(self, stock: &str, long_edge_mm: f32, route: ScanRoute) -> bool {
        stock == "kodak_portra_400"
            && long_edge_mm == 36.0
            && matches!(route, ScanRoute::NegativeDirect | ScanRoute::NegativePrint)
            && (-3.0..=3.0).contains(&self.amount_ev)
            && (0.5..=4.0).contains(&self.texel_pitch_um)
            && (0.0..=1.0).contains(&self.chroma_fraction)
    }
}

impl Recipe {
    pub(crate) fn new(bank: Arc<PortraGrainBank>, controls: Controls) -> Self {
        let gain = controls.amount_ev.exp2() / controls.texel_pitch_um;
        let hash = hash::u64s(&[
            bank.identity(),
            u64::from(controls.texel_pitch_um.to_bits()),
            u64::from(gain.to_bits()),
            u64::from(controls.chroma_fraction.to_bits()),
        ]);
        Self {
            bank,
            texel_pitch_um: controls.texel_pitch_um,
            gain,
            chroma_fraction: controls.chroma_fraction,
            hash,
        }
    }

    pub fn bank(&self) -> &Arc<PortraGrainBank> {
        &self.bank
    }

    pub fn texel_pitch_um(&self) -> f32 {
        self.texel_pitch_um
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn chroma_fraction(&self) -> f32 {
        self.chroma_fraction
    }

    pub fn footprint(&self, long_edge_px: i32) -> Option<f32> {
        if !(1..=portra::MAX_RENDER_AXIS).contains(&long_edge_px) {
            return None;
        }
        let footprint = (portra::FILM_LONG_EDGE_UM
            / f64::from(long_edge_px)
            / f64::from(self.texel_pitch_um)) as f32;
        (2.0..=37.5).contains(&footprint).then_some(footprint)
    }

    pub fn verified_normalization(&self, table: &PortraGrainTable) -> bool {
        portra::verify_table(&self.bank, table)
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }
}
