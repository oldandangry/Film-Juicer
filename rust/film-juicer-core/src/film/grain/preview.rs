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
    raster: Raster,
    footprint: f32,
    sampling: Option<ProxyGrid>,
    gain: f32,
    chroma_fraction: f32,
    hash: u64,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Raster {
    pub long_edge_px: i32,
    pub full_grid: Option<ProxyGrid>,
}

/// Rendered pixel edges expressed on a sampling lattice anchored to Full.
#[derive(Clone, Copy, PartialEq)]
pub struct ProxyGrid {
    pub extent: [f64; 2],
    pub step: [f64; 2],
    pub origin: [f64; 2],
}

fn widest_clipped_aperture(extent: f64, step: f64, origin: f64) -> f64 {
    // Past the first nonnegative edge, aperture widths are constant until
    // the right edge clips. Compare the first interior and preceding cells.
    let first_inside = (-origin / step).ceil().max(0.0);
    let edge = |index: f64| index.mul_add(step, origin).clamp(0.0, extent);
    let start = edge(first_inside);
    (edge(first_inside + 1.0) - start).max(start - edge((first_inside - 1.0).max(0.0)))
}

impl Raster {
    pub(crate) fn sampling(
        self,
        controls: Controls,
        long_edge_mm: f32,
        footprint: f32,
    ) -> (f32, Option<ProxyGrid>) {
        let Some(grid) = self.full_grid else {
            return (footprint, None);
        };
        let full_pitch = f64::from(long_edge_mm) * 1000.0
            / grid.extent[0].max(grid.extent[1])
            / f64::from(controls.texel_pitch_um);
        // Keep the calibrated decision on square rendered pixels, including
        // when outward-rounded raster edges shift the represented width.
        let widest = if grid.step[0] == grid.step[1] {
            footprint
        } else {
            let coverage_x = widest_clipped_aperture(grid.extent[0], grid.step[0], grid.origin[0]);
            let coverage_y = widest_clipped_aperture(grid.extent[1], grid.step[1], grid.origin[1]);
            (full_pitch * coverage_x.max(coverage_y)) as f32
        };
        if widest > portra::CALIBRATED_MAX_FOOTPRINT {
            (full_pitch as f32, Some(grid))
        } else {
            (footprint, None)
        }
    }
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
        raster: Raster,
        footprint: f32,
        sampling: Option<ProxyGrid>,
    ) -> Self {
        let gain = controls.amount_ev.exp2() / controls.texel_pitch_um;
        let base_hash = hash::u64s(&[
            bank.identity(),
            u64::from(controls.texel_pitch_um.to_bits()),
            u64::from(gain.to_bits()),
            u64::from(controls.chroma_fraction.to_bits()),
            u64::from(table.footprint.to_bits()),
        ]);
        let hash = sampling.map_or(base_hash, |grid| {
            hash::u64s(&[
                base_hash,
                u64::from(footprint.to_bits()),
                grid.extent[0].to_bits(),
                grid.extent[1].to_bits(),
                grid.step[0].to_bits(),
                grid.step[1].to_bits(),
                grid.origin[0].to_bits(),
                grid.origin[1].to_bits(),
            ])
        });
        Self {
            bank,
            table,
            raster,
            footprint,
            sampling,
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

    pub fn raster(&self) -> Raster {
        self.raster
    }

    pub fn footprint(&self) -> f32 {
        self.footprint
    }

    pub fn sampling(&self) -> Option<ProxyGrid> {
        self.sampling
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn chroma_fraction(&self) -> f32 {
        self.chroma_fraction
    }

    pub(crate) fn density_allowances(&self) -> Option<[f32; 2]> {
        // Transmission averaging preserves the density enclosure of its samples.
        portra::allowances(&self.bank, &self.table, self.gain)
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::{Controls, ProxyGrid, Raster};

    #[test]
    fn whole_period_proxy_keeps_the_full_lattice() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 0.5,
            chroma_fraction: 0.24,
        };
        let grid = ProxyGrid {
            extent: [2000.0, 1000.0],
            step: [500.0, 500.0],
            origin: [0.0; 2],
        };
        let raster = Raster {
            long_edge_px: 4,
            full_grid: Some(grid),
        };
        let rendered = controls.footprint(32.0, raster.long_edge_px);
        let (sample_pitch, sampling) = raster.sampling(controls, 32.0, rendered);
        assert_eq!(sample_pitch, controls.footprint(32.0, 2000));
        assert!(sampling == Some(grid));
    }

    #[test]
    fn unequal_proxy_uses_the_wider_full_aperture_in_either_orientation() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let full_pitch = controls.footprint(36.0, 4832);
        let rendered = controls.footprint(36.0, 1208);
        assert!(f64::from(full_pitch) * 4.0 < 37.5);
        assert!(f64::from(full_pitch) * 8.0 > 37.5);
        for (extent, step) in [([4832.0, 96.0], [4.0, 8.0]), ([96.0, 4832.0], [8.0, 4.0])] {
            let grid = ProxyGrid {
                extent,
                step,
                origin: [0.0; 2],
            };
            let raster = Raster {
                long_edge_px: 1208,
                full_grid: Some(grid),
            };
            let (sample_pitch, sampling) = raster.sampling(controls, 36.0, rendered);
            assert_eq!(sample_pitch, full_pitch);
            assert!(sampling == Some(grid));

            let square = Raster {
                full_grid: Some(ProxyGrid {
                    step: [4.0; 2],
                    ..grid
                }),
                ..raster
            };
            assert!(square.sampling(controls, 36.0, rendered) == (rendered, None));
        }
    }

    #[test]
    fn clipped_quarter_uses_the_calibrated_table() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let grid = ProxyGrid {
            extent: [4832.0, 8.0],
            step: [4.0, 8.0],
            origin: [0.0, -4.0],
        };
        let raster = Raster {
            long_edge_px: 1208,
            full_grid: Some(grid),
        };
        let rendered = controls.footprint(36.0, raster.long_edge_px);
        assert!(raster.sampling(controls, 36.0, rendered) == (rendered, None));
        let taller = Raster {
            full_grid: Some(ProxyGrid {
                extent: [4832.0, 10.0],
                ..grid
            }),
            ..raster
        };
        assert!(taller.sampling(controls, 36.0, rendered).1.is_some());
    }
}
