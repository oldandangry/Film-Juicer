//! Admitted Film v1 controls and their canonical Portra bank.

use std::sync::Arc;

use crate::assets::{PortraGrainBank, PortraGrainTable};
use crate::{hash, route::ScanRoute};
use super::super::density::portra;

pub const DEFAULT_CHROMA: f64 = 1.0;

#[derive(Clone, Copy)]
pub struct Controls {
    pub amount_ev: f32,
    pub texel_pitch_um: f32,
    pub chroma_fraction: f32,
}

pub struct Recipe {
    bank: Arc<PortraGrainBank>,
    table: Arc<PortraGrainTable>,
    long_edge_px: i32,
    gain: f32,
    chroma_fraction: f32,
    hash: u64,
}

impl Controls {
    pub(crate) fn valid(self, stock: &str, long_edge_mm: f32, route: ScanRoute) -> bool {
        stock == "kodak_portra_400"
            && super::super::input::film_format_mm(f64::from(long_edge_mm)).is_ok()
            && matches!(route, ScanRoute::NegativeDirect | ScanRoute::NegativePrint)
            && (-3.0..=3.0).contains(&self.amount_ev)
            && (0.5..=4.0).contains(&self.texel_pitch_um)
            && (0.0..=1.0).contains(&self.chroma_fraction)
    }

    pub(crate) fn footprint(self, long_edge_mm: f32, long_edge_px: i32) -> f32 {
        (f64::from(long_edge_mm) * 1000.0
            / f64::from(long_edge_px)
            / f64::from(self.texel_pitch_um)) as f32
    }
}

impl Recipe {
    pub(crate) fn new(
        bank: Arc<PortraGrainBank>,
        table: Arc<PortraGrainTable>,
        controls: Controls,
        long_edge_px: i32,
    ) -> Self {
        let gain = controls.amount_ev.exp2() / controls.texel_pitch_um;
        let hash = hash::u64s(&[
            bank.identity(),
            u64::from(controls.texel_pitch_um.to_bits()),
            u64::from(gain.to_bits()),
            u64::from(controls.chroma_fraction.to_bits()),
            u64::from(table.footprint.to_bits()),
        ]);
        Self {
            bank,
            table,
            long_edge_px,
            gain,
            chroma_fraction: controls.chroma_fraction,
            hash,
        }
    }

    pub fn bank(&self) -> &Arc<PortraGrainBank> {
        &self.bank
    }

    pub fn table(&self) -> &PortraGrainTable {
        &self.table
    }

    pub fn long_edge_px(&self) -> i32 {
        self.long_edge_px
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn chroma_fraction(&self) -> f32 {
        self.chroma_fraction
    }

    pub(crate) fn density_allowances(&self) -> Option<[f32; 2]> {
        portra::allowances(&self.bank, &self.table, self.gain)
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }
}
