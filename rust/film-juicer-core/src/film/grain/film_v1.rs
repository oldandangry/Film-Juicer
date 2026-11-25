//! Admitted Film v1 controls and their canonical Portra bank.

use std::sync::Arc;

use crate::assets::{PortraGrainBank, PortraGrainTable};
use crate::{hash, route::ScanRoute};
use super::super::{
    Development,
    density::{self, portra},
};

pub const AMOUNT_EV_RANGE: std::ops::RangeInclusive<f32> = -3.0..=3.0;
pub const TEXEL_PITCH_UM_RANGE: std::ops::RangeInclusive<f32> = 0.5..=4.0;
pub const CHROMA_FRACTION_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;
pub const DEFAULT_AMOUNT_EV: f32 = 0.0;
pub const DEFAULT_TEXEL_PITCH_UM: f32 = 1.0;
pub const DEFAULT_CHROMA_FRACTION: f32 = 1.0;

#[derive(Clone, Copy)]
pub struct Controls {
    pub amount_ev: f32,
    pub texel_pitch_um: f32,
    pub chroma_fraction: f32,
}

#[derive(Clone, Copy)]
pub enum Selection {
    Current,
    FilmV1Disabled,
    FilmV1(Controls),
}

impl Selection {
    pub fn selected(self) -> bool {
        !matches!(self, Self::Current)
    }

    pub fn controls(self) -> Option<Controls> {
        match self {
            Self::FilmV1(controls) => Some(controls),
            Self::Current | Self::FilmV1Disabled => None,
        }
    }
}

pub struct Recipe {
    bank: Arc<PortraGrainBank>,
    table: Arc<PortraGrainTable>,
    raster: Raster,
    rendered_footprint_texels: f32,
    sampling: Sampling,
    gain: f32,
    chroma_fraction: f32,
    hash: u64,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Raster {
    /// Complete image long edge at the current render scale.
    pub rendered_long_edge_px: i32,
    /// Full grid considered during sampling selection and publication reuse.
    pub candidate_full_grid: Option<ProxyGrid>,
}

/// Rendered pixel edges expressed on a sampling lattice anchored to Full.
#[derive(Clone, Copy, PartialEq)]
pub struct ProxyGrid {
    /// Full image extent, in Full cells.
    pub extent: [f64; 2],
    /// Full cells per rendered pixel.
    pub step: [f64; 2],
    /// Signed first rendered edge, in Full cells relative to the Full image.
    pub origin: [f64; 2],
    /// Admitted rendered dimensions, including any empty border.
    pub size: [i32; 2],
}

#[derive(Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Empty borders, in left, top, right, bottom order.
    Calibrated([u8; 4]),
    Integrated(ProxyGrid),
}

#[derive(Clone, Copy)]
struct Axis {
    extent: f64,
    step: f64,
    origin: f64,
    size: i32,
}

impl Axis {
    fn edge(self, i: i32) -> f64 {
        f64::from(i)
            .mul_add(self.step, self.origin)
            .clamp(0.0, self.extent)
    }

    fn border(self) -> [u8; 2] {
        // OFX admission proves that each empty end has a positive neighbor.
        [
            u8::from(self.edge(0) == self.edge(1)),
            u8::from(self.edge(self.size - 1) == self.edge(self.size)),
        ]
    }

    fn widest_interval(self, first: i32, end: i32) -> f64 {
        // Independently rounded edges can widen any cell, even when the
        // span, endpoints and nominal spacing all agree. Visit each pair.
        let mut left = self.edge(first);
        let mut widest = 0.0_f64;
        for i in first + 1..=end {
            let right = self.edge(i);
            widest = widest.max(right - left);
            left = right;
        }
        widest
    }

    fn spacing_range(self, first: i32, end: i32, widest: f64) -> [f64; 2] {
        if end - first < 3 {
            return [widest; 2];
        }
        // Interior edges are not clipped. Their rounding bins describe
        // nominal square spacing, not the widest realized aperture.
        let a = self.edge(first + 1);
        let b = self.edge(end - 1);
        let cells = f64::from(end - first - 2);
        let span = b - a;
        let width = span / cells;
        let residual = (b - span) - a;
        let residual = (-width).mul_add(cells, span) + residual;
        let lower = ((b - b.next_down()) + (a.next_up() - a)) * 0.5;
        let upper = ((b.next_up() - b) + (a - a.next_down())) * 0.5;
        [
            width + (residual - lower) / cells,
            width + (residual + upper) / cells,
        ]
    }

    fn rounded_scale_range(cells: i32, edge: impl Fn(i32) -> f64) -> Option<[f64; 2]> {
        if cells < 2 || edge(0) != 0.0 {
            return None;
        }
        let pixels = f64::from(cells);
        let step = edge(1);
        let terminal = edge(cells);
        let scale_range = |dimension| [(pixels - 0.5) / dimension, (pixels + 0.5) / dimension];
        if cells == 2 {
            return (terminal.is_finite()
                && terminal == terminal.trunc()
                && step == terminal * 0.5)
                .then(|| scale_range(terminal));
        }

        // For positive cells with a zero anchor, only the terminal can clip.
        let projection = pixels * step;
        if terminal > projection || !(2..cells).all(|i| edge(i) == f64::from(i) * step) {
            return None;
        }

        // A clipped projection fixes the dimension at the terminal. Otherwise,
        // division (at most 4/3 unit roundoffs), reciprocal and product (one
        // each) bound a matching dimension to four Float64 steps above it.
        let upward_steps = if terminal == projection { 4 } else { 0 };
        let mut dimension = terminal;
        let mut range: Option<[f64; 2]> = None;
        for _ in 0..=upward_steps {
            if dimension.is_finite()
                && dimension == dimension.trunc()
                && (pixels / dimension).recip() == step
                && projection.min(dimension) == terminal
            {
                let [low, high] = scale_range(dimension);
                // The exact candidates' ranges overlap: their relative width
                // is 1/pixels and their centers differ by only a few ULPs.
                range = Some(range.map_or([low, high], |[min, max]| [min.min(low), max.max(high)]));
            }
            dimension = dimension.next_up();
        }
        range
    }
}

impl Raster {
    pub(crate) fn sampling(self, controls: Controls, long_edge_mm: f32) -> (f32, f32, Sampling) {
        let Some(grid) = self.candidate_full_grid else {
            let rendered_texels = controls
                .footprint_texels(long_edge_mm, f64::from(self.rendered_long_edge_px))
                as f32;
            return (
                rendered_texels,
                rendered_texels,
                Sampling::Calibrated([0; 4]),
            );
        };
        let axes: [Axis; 2] = std::array::from_fn(|i| Axis {
            extent: grid.extent[i],
            step: grid.step[i],
            origin: grid.origin[i],
            size: grid.size[i],
        });
        let [[left, right], [top, bottom]] = axes.map(Axis::border);
        let first = [i32::from(left), i32::from(top)];
        let end = [
            grid.size[0] - i32::from(right),
            grid.size[1] - i32::from(bottom),
        ];
        let supported_long_edge_px = (end[0] - first[0]).max(end[1] - first[1]);
        let rendered_texels =
            controls.footprint_texels(long_edge_mm, f64::from(supported_long_edge_px)) as f32;
        let full_cell_texels =
            controls.footprint_texels(long_edge_mm, grid.extent[0].max(grid.extent[1]));
        let widest: [f64; 2] = std::array::from_fn(|i| axes[i].widest_interval(first[i], end[i]));
        if full_cell_texels * widest[0].max(widest[1])
            <= f64::from(portra::CALIBRATED_MAX_FOOTPRINT)
        {
            let [[x_min, x_max], [y_min, y_max]] =
                std::array::from_fn(|i| axes[i].spacing_range(first[i], end[i], widest[i]));
            let square = x_min.max(y_min) <= x_max.min(y_max) || {
                let [x, y] = std::array::from_fn(|i| {
                    Axis::rounded_scale_range(end[i] - first[i], |edge| {
                        axes[i].edge(first[i] + edge)
                    })
                });
                matches!((x, y), (Some([x_min, x_max]), Some([y_min, y_max]))
                    if x_min.max(y_min) < x_max.min(y_max))
            };
            if square {
                return (
                    rendered_texels,
                    rendered_texels,
                    Sampling::Calibrated([left, top, right, bottom]),
                );
            }
        }
        (
            rendered_texels,
            full_cell_texels as f32,
            Sampling::Integrated(grid),
        )
    }
}

impl Controls {
    pub(crate) fn valid(self, stock: &str, long_edge_mm: f32, route: ScanRoute) -> bool {
        stock == "kodak_portra_400"
            && super::super::input::film_format_mm(f64::from(long_edge_mm)).is_ok()
            && matches!(route, ScanRoute::NegativeDirect | ScanRoute::NegativePrint)
            && AMOUNT_EV_RANGE.contains(&self.amount_ev)
            && TEXEL_PITCH_UM_RANGE.contains(&self.texel_pitch_um)
            && CHROMA_FRACTION_RANGE.contains(&self.chroma_fraction)
    }

    fn footprint_texels(self, long_edge_mm: f32, long_edge_cells: f64) -> f64 {
        f64::from(long_edge_mm) * 1000.0 / long_edge_cells / f64::from(self.texel_pitch_um)
    }
}

impl Recipe {
    pub(crate) fn new(
        bank: Arc<PortraGrainBank>,
        table: Arc<PortraGrainTable>,
        controls: Controls,
        raster: Raster,
        rendered_footprint_texels: f32,
        sampling: Sampling,
    ) -> Self {
        let gain = controls.amount_ev.exp2() / controls.texel_pitch_um;
        let base_hash = hash::u64s(&[
            bank.identity(),
            u64::from(controls.texel_pitch_um.to_bits()),
            u64::from(gain.to_bits()),
            u64::from(controls.chroma_fraction.to_bits()),
            u64::from(table.footprint.to_bits()),
        ]);
        let hash = match sampling {
            Sampling::Calibrated([0, 0, 0, 0]) => base_hash,
            Sampling::Calibrated(border) => {
                hash::u64s(&[base_hash, u64::from(u32::from_le_bytes(border))])
            }
            Sampling::Integrated(grid) => hash::u64s(&[
                base_hash,
                u64::from(rendered_footprint_texels.to_bits()),
                grid.extent[0].to_bits(),
                grid.extent[1].to_bits(),
                grid.step[0].to_bits(),
                grid.step[1].to_bits(),
                grid.origin[0].to_bits(),
                grid.origin[1].to_bits(),
            ]),
        };
        Self {
            bank,
            table,
            raster,
            rendered_footprint_texels,
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

    pub fn rendered_footprint_texels(&self) -> f32 {
        self.rendered_footprint_texels
    }

    pub fn sampling(&self) -> Sampling {
        self.sampling
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn chroma_fraction(&self) -> f32 {
        self.chroma_fraction
    }

    pub(crate) fn density_bounds(
        &self,
        development: &Development,
    ) -> Result<density::Bounds, density::Error> {
        // Transmission averaging preserves the density enclosure of its samples.
        density::Bounds::portra_film_v1(
            development,
            &self.bank,
            &self.table,
            self.gain,
            self.chroma_fraction,
        )
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Axis, Controls, DEFAULT_AMOUNT_EV, DEFAULT_CHROMA_FRACTION, DEFAULT_TEXEL_PITCH_UM,
        ProxyGrid, Raster, Sampling, ScanRoute, Selection,
    };

    #[test]
    fn authored_controls_admit_only_finite_values_in_the_inclusive_ranges() {
        let defaults = Controls {
            amount_ev: DEFAULT_AMOUNT_EV,
            texel_pitch_um: DEFAULT_TEXEL_PITCH_UM,
            chroma_fraction: DEFAULT_CHROMA_FRACTION,
        };
        let valid = |controls: Controls| {
            controls.valid("kodak_portra_400", 36.0, ScanRoute::NegativeDirect)
        };
        assert!(valid(defaults));
        type SetControl = fn(Controls, f32) -> Controls;
        let controls: [(SetControl, f32, f32); 3] = [
            (|c, amount_ev| Controls { amount_ev, ..c }, -3.0, 3.0),
            (
                |c, texel_pitch_um| Controls {
                    texel_pitch_um,
                    ..c
                },
                0.5,
                4.0,
            ),
            (
                |c, chroma_fraction| Controls {
                    chroma_fraction,
                    ..c
                },
                0.0,
                1.0,
            ),
        ];
        for (with, min, max) in controls {
            assert!(valid(with(defaults, min)) && valid(with(defaults, max)));
            for outside in [
                min.next_down(),
                max.next_up(),
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
            ] {
                assert!(!valid(with(defaults, outside)));
            }
        }
    }

    fn realized_axis_edges(grid: ProxyGrid, axis: usize) -> Vec<f64> {
        (0..=grid.size[axis])
            .map(|i| {
                f64::from(i)
                    .mul_add(grid.step[axis], grid.origin[axis])
                    .max(0.0)
                    .min(grid.extent[axis])
            })
            .collect()
    }

    fn oracle_max_width(grid: ProxyGrid) -> f64 {
        (0..2)
            .map(|axis| {
                realized_axis_edges(grid, axis)
                    .windows(2)
                    .map(|edges| edges[1] - edges[0])
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max)
    }

    fn check_realized_cutoff(
        extent: [f64; 2],
        size: [i32; 2],
        lower: i32,
        scale: f64,
        rows: &[f64],
        changed: (usize, f64),
    ) {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let step_x = (f64::from(size[0]) / extent[0]).recip();
        let full_pitch = 36_000.0 / extent[0];
        for (scale_y, above) in [(scale, false), (scale.next_down(), true)] {
            let grid = ProxyGrid {
                extent,
                step: [step_x, scale_y.recip()],
                origin: [0.0, f64::from(lower) / scale_y - f64::from(lower) / scale],
                size,
            };
            let mut expected = rows.to_vec();
            if above {
                expected[changed.0] = changed.1;
            }
            assert_eq!(realized_axis_edges(grid, 1), expected);
            let widest = oracle_max_width(grid) * full_pitch;
            assert!(if above { widest > 37.5 } else { widest == 37.5 });
            assert_eq!(widest as f32, 37.5);
            let raster = Raster {
                rendered_long_edge_px: size[0],
                candidate_full_grid: Some(grid),
            };
            let (rendered, pitch, sampling) = raster.sampling(controls, 36.0);
            if above {
                assert!(
                    sampling == Sampling::Integrated(grid),
                    "realized pitch {widest:.17} above 37.5 selected calibrated pitch {pitch}"
                );
                assert_eq!(pitch, full_pitch as f32);
            } else {
                assert!(sampling == Sampling::Calibrated([0; 4]));
                assert_eq!(pitch, rendered);
            }
        }
    }

    #[test]
    fn realized_cutoff_includes_the_unprobed_rounded_interior_cell() {
        check_realized_cutoff(
            [12_480.0, 78.0],
            [961, 6],
            -1,
            6.0 / 78.0,
            &[0.0, 13.0, 26.0, 39.0, 52.0, 65.0, 78.0],
            (4, 52.0_f64.next_up()),
        );
    }

    #[test]
    fn realized_cutoff_is_independent_of_nominal_square_bins() {
        check_realized_cutoff(
            [3840.0, 15.0],
            [960, 4],
            -2,
            0.25,
            &[0.0, 4.0, 8.0, 12.0, 15.0],
            (1, 4.0_f64.next_down().next_down()),
        );
    }

    #[test]
    fn widest_interval_matches_enumeration_with_clipping_and_empty_ends() {
        for (extent, step, origin, size, expected) in [
            (4.0, 8.0, -4.0, 1, 4.0),
            (7.0, 5.0, 0.0, 2, 5.0),
            (11.0, 4.0, 0.0, 3, 4.0),
            (4.0, 4.0, -4.0, 3, 4.0),
            (7.0, 5.0, 0.0, 3, 5.0),
            (15.0, 4.0, -4.0, 5, 4.0),
        ] {
            let grid = ProxyGrid {
                extent: [extent; 2],
                step: [step; 2],
                origin: [origin; 2],
                size: [size; 2],
            };
            let axis = Axis {
                extent,
                step,
                origin,
                size,
            };
            let [leading, trailing] = axis.border();
            assert_eq!(oracle_max_width(grid), expected);
            assert_eq!(
                axis.widest_interval(i32::from(leading), size - i32::from(trailing)),
                expected
            );
        }
    }

    #[test]
    fn realized_cutoff_includes_the_first_and_last_positive_cells_on_either_axis() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        // Neither widening affects the interior span of its three-cell axis.
        let ulp = 4.0_f64.next_up() - 4.0;
        for (extent, step, origin, wide_cell) in [
            (10.0, 4.0_f64.next_up(), ulp * 0.5, 0),
            (12.0, 4.0, -5.0 * ulp, 2),
        ] {
            let mut grid = ProxyGrid {
                extent: [3840.0, extent],
                step: [4.0, step],
                origin: [0.0, origin],
                size: [960, 3],
            };
            for axis in [1, 0] {
                let edges = realized_axis_edges(grid, axis);
                for (cell, pair) in edges.windows(2).enumerate() {
                    assert_eq!((pair[1] - pair[0]) > 4.0, cell == wide_cell);
                }
                let pitch = oracle_max_width(grid) * 9.375;
                assert!(pitch > 37.5);
                assert_eq!(pitch as f32, 37.5);
                let raster = Raster {
                    rendered_long_edge_px: 960,
                    candidate_full_grid: Some(grid),
                };
                assert!(
                    raster.sampling(controls, 36.0) == (37.5, 9.375, Sampling::Integrated(grid))
                );
                grid.extent.swap(0, 1);
                grid.step.swap(0, 1);
                grid.origin.swap(0, 1);
                grid.size.swap(0, 1);
            }
        }
    }

    #[test]
    fn realized_cutoff_keeps_strictly_below_and_exact_neighbors_calibrated() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        for (step, below) in [(4.0 - 2.0_f64.powi(-36), true), (4.0, false)] {
            let grid = ProxyGrid {
                extent: [3840.0, 96.0],
                step: [step; 2],
                origin: [0.0; 2],
                size: [960, 24],
            };
            let pitch = oracle_max_width(grid) * 9.375;
            assert!(if below { pitch < 37.5 } else { pitch == 37.5 });
            assert_eq!(pitch as f32, 37.5);
            let raster = Raster {
                rendered_long_edge_px: 960,
                candidate_full_grid: Some(grid),
            };
            assert!(raster.sampling(controls, 36.0) == (37.5, 37.5, Sampling::Calibrated([0; 4])));
        }
    }

    #[test]
    fn paused_supported_extent_alias_keeps_table_and_bounds() {
        use crate::recipe::{self, FilmGrainFrame, test_support};
        use crate::route::ScanRoute;
        use crate::state::{CaptureMode, CaptureStart, Publication, SnapshotStart};
        use std::sync::Arc;

        let assets = test_support::assets();
        let start = ProxyGrid {
            extent: [4832.0, 7.0],
            step: [4.0, (2.0_f64 / 7.0).recip()],
            origin: [0.0; 2],
            size: [1208, 2],
        };
        let alias = ProxyGrid {
            extent: [4832.0, 8.0],
            ..start
        };
        assert_eq!(7.0 * (2.0 / 7.0), 2.0);
        assert_eq!(7.0_f64.next_up() * (2.0 / 7.0), 2.0);
        for grid in [start, alias] {
            assert!(
                realized_axis_edges(grid, 0)
                    .into_iter()
                    .zip(0..=1208)
                    .all(|(edge, i)| edge.to_bits() == (4.0 * f64::from(i)).to_bits())
            );
            assert_eq!(
                realized_axis_edges(grid, 1)
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                [0.0_f64, 3.5, 7.0].map(f64::to_bits)
            );
        }
        for route in [ScanRoute::NegativeDirect, ScanRoute::NegativePrint] {
            let mut params = test_support::params(route, 1);
            params.camera_film_format_long_edge_mm = 36.0;
            params.film_grain = Selection::FilmV1(Controls {
                amount_ev: 0.0,
                texel_pitch_um: 1.0,
                chroma_fraction: 1.0,
            });
            let publication = Publication::new();
            let CaptureStart::Started { capture } =
                publication.begin_capture(CaptureMode::Change).unwrap()
            else {
                unreachable!()
            };
            let SnapshotStart::Ready { write_revision } =
                publication.begin_snapshot(capture).unwrap()
            else {
                unreachable!()
            };
            publication
                .install(capture, write_revision, params, 1)
                .unwrap();
            let admit = |candidate_full_grid| {
                let frame = FilmGrainFrame::Raster(Raster {
                    rendered_long_edge_px: 1208,
                    candidate_full_grid,
                });
                publication
                    .admit_film_grain_frame(publication.frame_input().unwrap(), frame, |params| {
                        recipe::prepare_for_frame(&assets, params, frame)
                    })
                    .unwrap()
            };
            let oracle = admit(None);
            let initial = admit(Some(start));
            let switched = admit(Some(alias));
            let returned = admit(Some(start));
            assert!(!Arc::ptr_eq(&initial, &switched));
            assert!(!Arc::ptr_eq(&switched, &returned));
            assert!(Arc::ptr_eq(&returned, &admit(Some(start))));
            let expected = oracle.film_grain().unwrap();
            for (state, grid) in [(&initial, start), (&returned, start), (&switched, alias)] {
                let grain = state.film_grain().unwrap();
                assert!(grain.raster().candidate_full_grid == Some(grid));
                assert!(
                    grain.sampling() == Sampling::Calibrated([0; 4]),
                    "identical supported words at extent {} switched the calibrated sampler (table pitch {})",
                    grid.extent[1],
                    grain.table().footprint
                );
                let table = grain.table();
                assert_eq!(
                    table.footprint.to_bits(),
                    expected.table().footprint.to_bits()
                );
                assert_eq!(table.mean_density, expected.table().mean_density);
                assert_eq!(table.correlation, expected.table().correlation);
                assert_eq!(
                    state.film_v1_bounds().unwrap().hash(),
                    oracle.film_v1_bounds().unwrap().hash()
                );
            }
        }
    }

    #[test]
    fn quotient_alignment_checks_the_entire_supported_sequence() {
        let at = Axis {
            extent: 78.0,
            step: 13.0,
            origin: 0.0,
            size: 6,
        };
        let step = 13.0_f64.next_up();
        let changed = Axis {
            step,
            origin: 13.0 - step,
            ..at
        };
        for i in [0, 1, 5, 6] {
            assert_eq!(at.edge(i), changed.edge(i));
        }
        assert_ne!(at.edge(4), changed.edge(4));
        assert!(Axis::rounded_scale_range(6, |i| at.edge(i)).is_some());
        assert!(Axis::rounded_scale_range(6, |i| changed.edge(i)).is_none());
    }

    #[test]
    fn supported_quotients_calibrate_in_either_orientation() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let scale = 3.0 / 11.0;
        let extended = 11.0_f64.next_up();
        assert_eq!(extended * scale, 3.0_f64.next_up());
        assert_eq!(3.0 / extended, scale);
        let rows = [0.0_f64, 3.666_666_666_666_667, 7.333_333_333_333_334, 11.0];
        for (extent, step, size, expected_rows) in [
            ([4832.0, 7.0], [4.0, 3.5], [1208, 2], &[0.0, 3.5, 7.0][..]),
            ([4832.0, 8.0], [4.0, 3.5], [1208, 2], &[0.0, 3.5, 7.0][..]),
            ([4832.0, 11.0], [4.0, scale.recip()], [1208, 3], &rows[..]),
            ([4832.0, 12.0], [4.0, scale.recip()], [1208, 3], &rows[..]),
        ] {
            let mut grid = ProxyGrid {
                extent,
                step,
                origin: [0.0; 2],
                size,
            };
            let pitch = (36_000.0 / 1208.0) as f32;
            for short in [1, 0] {
                assert!(
                    realized_axis_edges(grid, 1 - short)
                        .into_iter()
                        .zip(0..=1208)
                        .all(|(edge, i)| edge.to_bits() == (4.0 * f64::from(i)).to_bits())
                );
                assert!(
                    realized_axis_edges(grid, short)
                        .iter()
                        .zip(expected_rows)
                        .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
                );
                let raster = Raster {
                    rendered_long_edge_px: 1208,
                    candidate_full_grid: Some(grid),
                };
                assert!(
                    raster.sampling(controls, 36.0) == (pitch, pitch, Sampling::Calibrated([0; 4]))
                );
                grid.extent.swap(0, 1);
                grid.step.swap(0, 1);
                grid.size.swap(0, 1);
            }
        }
    }

    #[test]
    fn whole_reference_can_extend_past_the_terminal_word() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let mut grid = ProxyGrid {
            extent: [4830.0, 22.0],
            step: [2.5, (9.0_f64 / 22.0).recip()],
            origin: [0.0; 2],
            size: [1932, 9],
        };
        let rows = realized_axis_edges(grid, 1);
        let terminal = 22.0_f64.next_down();
        assert_eq!(rows[9].to_bits(), terminal.to_bits());
        let mut raster = Raster {
            rendered_long_edge_px: 1932,
            candidate_full_grid: Some(grid),
        };
        let (footprint, pitch, sampling) = raster.sampling(controls, 36.0);
        assert_eq!(footprint, pitch);
        assert!(sampling == Sampling::Calibrated([0; 4]));
        // Same step and every interior word, with a genuinely shortened last cell.
        grid.extent[1] = 21.0;
        let shortened = realized_axis_edges(grid, 1);
        assert_eq!(&shortened[..9], &rows[..9]);
        assert_eq!(shortened[9], 21.0);
        raster.candidate_full_grid = Some(grid);
        assert!(raster.sampling(controls, 36.0).2 == Sampling::Integrated(grid));

        let terminal = 2.0_f64.powi(56);
        let dimension = terminal.next_up();
        let step = (3.0 / dimension).recip();
        assert_eq!(3.0 * step, terminal);
        assert_ne!((3.0 / terminal).recip(), step);
        assert_eq!(
            Axis::rounded_scale_range(3, |i| f64::from(i) * step),
            Some([2.5 / dimension, 3.5 / dimension])
        );

        // Two exact integral references produce the same seven-cell words.
        let lower = f64::from_bits(terminal.to_bits() + 9);
        let upper = lower.next_up();
        let step = (7.0 / upper).recip();
        assert_eq!(7.0 * step, lower);
        assert_eq!((7.0 / lower).recip(), step);
        assert_eq!(
            Axis::rounded_scale_range(7, |i| f64::from(i) * step),
            Some([6.5 / upper, 7.5 / lower])
        );

        let axis = Axis {
            extent: 55.0,
            step: (3.0_f64 / 55.0).recip(),
            origin: 0.0,
            size: 3,
        };
        assert_eq!(axis.step * 3.0, axis.extent.next_up());
        assert_eq!(
            Axis::rounded_scale_range(axis.size, |i| axis.edge(i)),
            Some([2.5 / 55.0, 3.5 / 55.0])
        );
    }

    #[test]
    fn supported_material_rectangles_do_not_gain_square_calibration() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        for (extent, step, expected) in [
            (7.0, 5.0, &[0.0, 5.0, 7.0][..]),
            (5.0, 2.5, &[0.0, 2.5, 5.0][..]),
            (7.0, (2.0_f64 / 6.8).recip(), &[0.0, 3.4, 6.8][..]),
            (2.0, 4.0, &[0.0, 2.0][..]),
        ] {
            let grid = ProxyGrid {
                extent: [4832.0, extent],
                step: [4.0, step],
                origin: [0.0; 2],
                size: [1208, expected.len() as i32 - 1],
            };
            assert_eq!(realized_axis_edges(grid, 1), expected);
            let full_pitch = 36_000.0 / 4832.0;
            assert!(full_pitch * oracle_max_width(grid) < 37.5);
            let raster = Raster {
                rendered_long_edge_px: 1208,
                candidate_full_grid: Some(grid),
            };
            assert!(raster.sampling(controls, 36.0).2 == Sampling::Integrated(grid));
        }
    }

    #[test]
    fn supported_extent_alias_keeps_full_sampling_across_cuda_work_bound() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let scale = 1.0 / 24.0;
        let extended = 24.0_f64.next_up();
        assert_eq!(extended * scale, 1.0);
        let start = ProxyGrid {
            extent: [10_752.0, 24.0],
            step: [168.0, 24.0],
            origin: [0.0; 2],
            size: [64, 1],
        };
        let alternate = ProxyGrid {
            extent: [10_752.0, extended.ceil()],
            ..start
        };
        for axis in [0, 1] {
            assert_eq!(
                realized_axis_edges(start, axis),
                realized_axis_edges(alternate, axis)
            );
        }
        // Mirror the unchanged conservative work count for this witness.
        for (grid, expected_bound) in [(start, 4080), (alternate, 4250)] {
            let bound = (168 + 2) * (grid.extent[1].ceil().min(24.0 + 2.0) as i32);
            assert_eq!(bound, expected_bound);
            assert_eq!(grid.origin, [0.0; 2]);
            let raster = Raster {
                rendered_long_edge_px: 64,
                candidate_full_grid: Some(grid),
            };
            assert!(
                raster.sampling(controls, 36.0)
                    == (
                        562.5,
                        (36_000.0 / 10_752.0) as f32,
                        Sampling::Integrated(grid)
                    )
            );
        }
    }

    #[test]
    fn empty_borders_do_not_change_square_calibration() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 4.0,
            chroma_fraction: 1.0,
        };
        let grid = ProxyGrid {
            extent: [150.0, 95.0],
            step: [2.5; 2],
            origin: [0.0; 2],
            size: [60, 38],
        };
        for (grid, border) in [
            (grid, [0; 4]),
            (
                ProxyGrid {
                    size: [61, 39],
                    ..grid
                },
                [0, 0, 1, 1],
            ),
            (
                ProxyGrid {
                    origin: [-2.5; 2],
                    size: [61, 39],
                    ..grid
                },
                [1, 1, 0, 0],
            ),
        ] {
            let raster = Raster {
                rendered_long_edge_px: grid.size[0],
                candidate_full_grid: Some(grid),
            };
            let pitch = controls.footprint_texels(8.0, f64::from(60)) as f32;
            assert!(raster.sampling(controls, 8.0) == (pitch, pitch, Sampling::Calibrated(border)));
        }
    }

    #[test]
    fn different_steps_with_identical_positive_edges_share_the_cutoff_decision() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        for step in [4.0, 4.0_f64.next_up()] {
            let raster = Raster {
                rendered_long_edge_px: 960,
                candidate_full_grid: Some(ProxyGrid {
                    extent: [3840.0, 11.0],
                    step: [4.0, step],
                    origin: [0.0, 4.0 - step],
                    size: [960, 3],
                }),
            };
            assert_eq!(
                realized_axis_edges(raster.candidate_full_grid.unwrap(), 1),
                [0.0, 4.0, 8.0, 11.0]
            );
            assert!(raster.sampling(controls, 36.0) == (37.5, 37.5, Sampling::Calibrated([0; 4])));
        }
    }

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
            size: [4, 2],
        };
        let raster = Raster {
            rendered_long_edge_px: 4,
            candidate_full_grid: Some(grid),
        };
        let (_, sample_pitch, sampling) = raster.sampling(controls, 32.0);
        assert_eq!(
            sample_pitch,
            controls.footprint_texels(32.0, f64::from(2000)) as f32
        );
        assert!(sampling == Sampling::Integrated(grid));
    }

    #[test]
    fn unequal_proxy_uses_the_wider_full_aperture_in_either_orientation() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let full_pitch = controls.footprint_texels(36.0, f64::from(4832)) as f32;
        let rendered = controls.footprint_texels(36.0, f64::from(1208)) as f32;
        assert!(f64::from(full_pitch) * 4.0 < 37.5);
        assert!(f64::from(full_pitch) * 8.0 > 37.5);
        for (extent, step, size, square_size) in [
            ([4832.0, 96.0], [4.0, 8.0], [1208, 12], [1208, 24]),
            ([96.0, 4832.0], [8.0, 4.0], [12, 1208], [24, 1208]),
        ] {
            let grid = ProxyGrid {
                extent,
                step,
                origin: [0.0; 2],
                size,
            };
            let raster = Raster {
                rendered_long_edge_px: 1208,
                candidate_full_grid: Some(grid),
            };
            let (_, sample_pitch, sampling) = raster.sampling(controls, 36.0);
            assert_eq!(sample_pitch, full_pitch);
            assert!(sampling == Sampling::Integrated(grid));

            let square = Raster {
                candidate_full_grid: Some(ProxyGrid {
                    step: [4.0; 2],
                    size: square_size,
                    ..grid
                }),
                ..raster
            };
            assert!(
                square.sampling(controls, 36.0)
                    == (rendered, rendered, Sampling::Calibrated([0; 4]))
            );
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
            size: [1208, 2],
        };
        let raster = Raster {
            rendered_long_edge_px: 1208,
            candidate_full_grid: Some(grid),
        };
        let rendered =
            controls.footprint_texels(36.0, f64::from(raster.rendered_long_edge_px)) as f32;
        assert!(
            raster.sampling(controls, 36.0) == (rendered, rendered, Sampling::Calibrated([0; 4]))
        );
        let taller = Raster {
            candidate_full_grid: Some(ProxyGrid {
                extent: [4832.0, 10.0],
                ..grid
            }),
            ..raster
        };
        assert!(matches!(
            taller.sampling(controls, 36.0).2,
            Sampling::Integrated(_)
        ));
    }

    #[test]
    fn physical_width_selects_the_full_table_before_narrowing() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        let grid = ProxyGrid {
            extent: [3839.0, 4.0],
            step: [4.0; 2],
            origin: [-1.0, 0.0],
            size: [960, 1],
        };
        let raster = Raster {
            rendered_long_edge_px: 960,
            candidate_full_grid: Some(grid),
        };
        let rendered =
            controls.footprint_texels(36.0, f64::from(raster.rendered_long_edge_px)) as f32;
        assert_eq!(rendered, 37.5);
        for grid in [
            grid,
            ProxyGrid {
                step: [4.0, 8.0],
                origin: [-1.0, -4.0],
                ..grid
            },
        ] {
            let raster = Raster {
                candidate_full_grid: Some(grid),
                ..raster
            };
            let (_, pitch, sampling) = raster.sampling(controls, 36.0);
            assert_eq!(
                pitch,
                controls.footprint_texels(36.0, f64::from(3839)) as f32
            );
            assert!(sampling == Sampling::Integrated(grid));
        }
    }

    #[test]
    fn physical_cutoff_has_no_rounding_allowance() {
        let controls = Controls {
            amount_ev: 0.0,
            texel_pitch_um: 1.0,
            chroma_fraction: 1.0,
        };
        for (step, integrated) in [
            (4.0_f64.next_down(), false),
            (4.0, false),
            (4.0_f64.next_up(), true),
        ] {
            let raster = Raster {
                rendered_long_edge_px: 960,
                candidate_full_grid: Some(ProxyGrid {
                    extent: [3840.0, 96.0],
                    step: [step; 2],
                    origin: [0.0; 2],
                    size: [960, 24],
                }),
            };
            let (_, pitch, sampling) = raster.sampling(controls, 36.0);
            assert_eq!(
                matches!(sampling, Sampling::Integrated(_)),
                integrated,
                "{step:.17}"
            );
            assert_eq!(pitch, if integrated { 9.375 } else { 37.5 });
        }
    }
}
