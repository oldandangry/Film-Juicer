//! Maintenance proof for the selected Portra table and the shipping density guards.

use super::*;
use crate::film::grain::grain_v2::{AMOUNT_EV_RANGE, TEXEL_PITCH_UM_RANGE};
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy)]
struct Guard {
    span: f64,
    floor: f64,
    wavelength_nm: usize,
    first: u32,
    last: u32,
}

impl Guard {
    fn passes(self) -> bool {
        self.span <= 64.0 && self.floor >= -30.0
    }
}

fn guard(development: &film::Development, [lo, hi]: [[f32; 3]; 2]) -> Guard {
    let lo = lo.map(f64::from);
    let hi = hi.map(f64::from);
    let span = hi.into_iter().fold(0.0, f64::max) - lo.into_iter().fold(0.0, f64::min);
    let tables = development.profile().tables();
    let mut floor = f64::INFINITY;
    let mut wavelength_nm = 0;
    for (i, (dyes, &base)) in tables
        .channel_density()
        .iter()
        .zip(tables.base_density())
        .enumerate()
    {
        if base.is_nan() || dyes.iter().any(|dye| dye.is_nan()) {
            continue;
        }
        let density = (0..3).fold(f64::from(base), |sum, k| {
            sum + f64::from(dyes[k]) * if dyes[k] >= 0.0 { lo[k] } else { hi[k] }
        });
        if density < floor {
            (floor, wavelength_nm) = (density, 380 + 5 * i);
        }
    }
    Guard {
        span,
        floor,
        wavelength_nm,
        first: 0,
        last: 0,
    }
}

fn colour_guard(development: &film::Development, at: impl Fn(f32) -> [[f32; 3]; 2]) -> Guard {
    // For fixed density extrema, every light endpoint is affine in Colour.
    // Log density is monotone on each axis, so the union of the two boxes
    // encloses mixed-sign spectral corners throughout the Colour interval.
    let corners = (1..32)
        .map(|n| n as f32 / 32.0)
        .chain((5..=24).map(|n| 1.0 - 2.0_f32.powi(-n)))
        .chain([1.0]);
    let mut left = at(0.0);
    let mut worst = guard(development, left);
    for colour in corners {
        let right = at(colour);
        let union = [
            std::array::from_fn(|k| left[0][k].min(right[0][k]).next_down()),
            std::array::from_fn(|k| left[1][k].max(right[1][k]).next_up()),
        ];
        let tested = guard(development, union);
        if tested.floor < worst.floor {
            (worst.floor, worst.wavelength_nm) = (tested.floor, tested.wavelength_nm);
        }
        worst.span = worst.span.max(tested.span);
        left = right;
    }
    worst
}

fn allowance_guard(development: &film::Development, [lo, hi]: [f32; 2]) -> Guard {
    let maximum = Bounds::grain_v2_clean_maximum(development);
    guard(development, [[lo; 3], maximum.map(|d| (d + hi).next_up())])
}

struct Proof<'a> {
    assets: &'a Assets,
    bank: &'a GrainV2Bank,
    development: &'a film::Development,
    template: GrainV2Table,
    gain: f32,
}

impl Proof<'_> {
    fn exact(&self, word: u32) -> Guard {
        let width = f32::from_bits(word);
        let table = self
            .assets
            .grain_v2_table(self.bank, width)
            .unwrap_or_else(|error| {
                panic!(
                    "selected table width {width} (0x{word:08x}), gain {}: {error}",
                    self.gain
                )
            });
        let original = allowances(self.bank, &table, self.gain).unwrap_or_else(|| {
            panic!(
                "calibrated mean or correlation rejected at {width} (0x{word:08x}), gain {}",
                self.gain
            )
        });
        let retained = allowance_guard(self.development, original);
        if retained.passes() {
            return retained;
        }
        if let Some(periodic) = periodic_allowances(self.bank.stock(), &table, self.gain) {
            let tested = allowance_guard(self.development, periodic);
            assert!(
                tested.passes(),
                "periodic width {width} (0x{word:08x}), gain {}, allowances {periodic:?}: {} D span, {} D at {} nm",
                self.gain,
                tested.span,
                tested.floor,
                tested.wavelength_nm
            );
            return tested;
        }
        let maximum = Bounds::grain_v2_clean_maximum(self.development);
        let projection = projection_range(self.bank, &table, self.gain, maximum, true)
            .unwrap_or_else(|| panic!("no tighter enclosure at {width} (0x{word:08x})"));
        let tested = colour_guard(self.development, |colour| projection.densities(colour));
        assert!(
            tested.passes(),
            "unproved Colour interval at width {width} (0x{word:08x}), gain {}: {} D span, {} D at {} nm",
            self.gain,
            tested.span,
            tested.floor,
            tested.wavelength_nm
        );
        tested
    }

    fn interval(&self, first: u32, last: u32) -> Option<Guard> {
        let [first, last] = [first, last].map(f32::from_bits);
        let stats = calibration_interval::normalize(self.bank, first, last);
        let correlation = stats
            .correlation
            .iter()
            .flat_map(|record| &record[..16])
            .copied()
            .fold(1.0, f32::min);
        if correlation <= -0.9997
            || (first >= 1.0 && last <= CALIBRATED_MAX_FOOTPRINT && correlation < MIN_CORRELATION)
        {
            return None;
        }
        let shift = |r: usize, k: usize, side: usize| {
            f64::from(stats.mean[side][r][k])
                + f64::from(self.bank.records()[r].mean_light[k]).log10()
        };
        let [min_shift, max_shift] = (0..3)
            .flat_map(|r| (0..17).map(move |k| (r, k)))
            .fold([0.0_f64; 2], |[lo, hi], (r, k)| {
                [lo.min(shift(r, k, 0)), hi.max(shift(r, k, 1))]
            });
        if first >= 1.0
            && last <= CALIBRATED_MAX_FOOTPRINT
            && (-min_shift).max(max_shift) * f64::from(last) > MAX_SCALED_MEAN_SHIFT
        {
            return None;
        }

        let [a, b] = [first, last].map(f64::from);
        let division = f64::from(CALIBRATED_MAX_FOOTPRINT);
        let mut delta = [0.0_f64; 2];
        if first < 1.0 {
            delta = SUBTEXEL_DELTAS;
        } else {
            // Clip against both the ceil subdivision boundaries and the
            // scaled SAT bands. Closed endpoints deliberately overlap here.
            for pieces in ((a / division).ceil() as u32)..=((b / division).ceil() as u32) {
                let pieces = f64::from(pieces);
                let least = (a / pieces).max(division * (pieces - 1.0) / pieces);
                let most = (b / pieces).min(division);
                let mut lower = least;
                for &(upper, [bright, dark]) in &SCALED_DELTAS {
                    let upper = f64::from(upper);
                    if lower <= most && lower <= upper {
                        delta = [delta[0].min(bright / lower), delta[1].max(dark / lower)];
                    }
                    lower = lower.max(upper);
                }
            }
        }
        let factor = f64::from(self.gain)
            * 3.012
            * (2.0 / (1.0 + f64::from(correlation) - 8.0 * f64::from(f32::EPSILON))).sqrt();
        let retained = allowance_guard(
            self.development,
            [
                ((factor * (delta[0] - max_shift) - 0.002) as f32).next_down(),
                ((factor * (delta[1] - min_shift) + 0.002) as f32).next_up(),
            ],
        );
        if retained.passes() {
            return Some(retained);
        }
        if last > placement_extrema::BANDS.last().unwrap().0 {
            return None;
        }
        let maximum = Bounds::grain_v2_clean_maximum(self.development);
        let table = |side: usize| {
            let mut table = self.template.clone();
            table.footprint = first;
            table.mean_density = stats.mean[side];
            table.correlation = stats.correlation;
            table
        };
        let [dark, bright] = [table(0), table(1)];
        let tighter = colour_guard(self.development, |colour| {
            [
                density_range(self.bank, &bright, self.gain, colour, maximum).unwrap()[0],
                density_range(self.bank, &dark, self.gain, colour, maximum).unwrap()[1],
            ]
        });
        // The exact passing original boxes are already checked in shipping.
        // This bound proves the tighter path over the whole interval.
        tighter.passes().then_some(tighter)
    }
}

#[test]
fn gold_density_guards_cover_supported_footprints() {
    let assets = assets();
    let bank = assets.grain_v2_bank(GrainV2Stock::Gold200).unwrap();
    let development = film::Development::new(
        assets.film(GrainV2Stock::Gold200.key()).unwrap(),
        film::Params {
            gamma_factor: 1.0,
            layers_required: false,
            requested_route: ScanRoute::NegativeDirect,
        },
    )
    .unwrap();
    let proof = Proof {
        assets: &assets,
        bank: &bank,
        development: &development,
        gain: 16.0,
        template: (*assets
            .grain_v2_table(&bank, MIN_PROXY_SAMPLE_FOOTPRINT)
            .unwrap())
        .clone(),
    };
    let widths = (1..=150)
        .map(|quarter| quarter as f32 / 4.0)
        .chain((100..500).step_by(5).map(|word| word as f32 / 100.0))
        .chain(
            certificate(GrainV2Stock::Gold200)
                .placement
                .iter()
                .flat_map(|&(width, _)| [width.next_down(), width.next_up()]),
        )
        .chain([
            MIN_PROXY_SAMPLE_FOOTPRINT,
            0.999_999_94,
            37.5_f32.next_up(),
            128.0,
            255.0,
        ])
        .chain(
            [
                256.0_f32,
                384.0,
                512.0,
                640.0,
                1536.0,
                2608.6956,
                MAX_FOOTPRINT,
            ]
            .into_iter()
            .flat_map(|width| {
                [width.next_down(), width, width.next_up()]
                    .into_iter()
                    .filter(|&x| x <= MAX_FOOTPRINT)
            }),
        );
    let mut span = 0.0_f64;
    let mut floor = 0.0_f64;
    let mut count = 0;
    for width in widths {
        let tested = proof.exact(width.to_bits());
        span = span.max(tested.span);
        floor = floor.min(tested.floor);
        count += 1;
    }
    println!(
        "Gold gain 16: {count} represented footprints, max span {span} D, min spectrum {floor} D"
    );
}

#[derive(Default)]
struct Coverage {
    count: u32,
    intervals: u32,
    exact: u32,
    span: Option<Guard>,
    floor: Option<Guard>,
    singleton_rechecks: Vec<u32>,
}

impl Coverage {
    fn accept(&mut self, first: u32, last: u32, mut bound: Guard, exact: bool) {
        (bound.first, bound.last) = (first, last);
        self.count += last - first + 1;
        self.exact += u32::from(exact);
        self.intervals += u32::from(!exact);
        self.include(bound, bound);
    }

    fn include(&mut self, span: Guard, floor: Guard) {
        if self.span.is_none_or(|bound| {
            span.span > bound.span || (span.span == bound.span && span.first < bound.first)
        }) {
            self.span = Some(span);
        }
        if self.floor.is_none_or(|bound| {
            floor.floor < bound.floor || (floor.floor == bound.floor && floor.first < bound.first)
        }) {
            self.floor = Some(floor);
        }
    }
}

fn certify(proof: &Proof<'_>, first: u32, last: u32, coverage: &mut Coverage) {
    if let Some(bound) = proof.interval(first, last) {
        coverage.accept(first, last, bound, false);
    } else if first == last {
        coverage.accept(first, last, proof.exact(first), true);
        coverage.singleton_rechecks.push(first);
    } else {
        let middle = first + (last - first) / 2;
        certify(proof, first, middle, coverage);
        certify(proof, middle + 1, last, coverage);
    }
}

fn ranges(first: u32, last: u32) -> impl Iterator<Item = RangeInclusive<u32>> {
    let cuts = [1.0_f32.to_bits() - 1]
        .into_iter()
        .chain(
            placement_extrema::BANDS[1..]
                .iter()
                .map(|(end, _)| end.to_bits()),
        )
        .chain(
            [
                CALIBRATED_MAX_FOOTPRINT,
                SINGLE_WRAP_MAX_FOOTPRINT,
                (PERIODIC_ALLOWANCE_MIN_FOOTPRINT as f32).next_down(),
            ]
            .map(f32::to_bits),
        )
        .chain([last]);
    let mut next = first;
    cuts.filter_map(move |cut| {
        let end = cut.min(last);
        (next <= end).then(|| {
            let range = next..=end;
            next = end + 1;
            range
        })
    })
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Nonconstant,
    Interval,
    Exact,
}

#[derive(serde::Deserialize)]
struct Parent {
    first: u32,
    last: u32,
    words: Option<usize>,
}

#[derive(serde::Deserialize)]
struct Request {
    mode: Mode,
    parents: Vec<Parent>,
}

struct LightCoverage {
    words: u32,
    intervals: u32,
    gap: f64,
    limiting: [u32; 2],
}

fn certify_nonconstant(bank: &GrainV2Bank, first: u32, last: u32, result: &mut LightCoverage) {
    if let Some(gap) = calibration_interval::nonconstant_light_gap(bank, first, last) {
        result.words += last - first + 1;
        result.intervals += 1;
        if gap < result.gap {
            result.gap = gap;
            result.limiting = [first, last];
        }
    } else {
        assert!(
            first < last,
            "unproved nonconstant calibration light at {first:08x}"
        );
        let mid = first + (last - first) / 2;
        certify_nonconstant(bank, first, mid, result);
        certify_nonconstant(bank, mid + 1, last, result);
    }
}

fn nonconstant_report(bank: &GrainV2Bank, parent: &Parent) -> serde_json::Value {
    let Parent { first, last, .. } = *parent;
    assert!(last <= 10.0_f32.to_bits());
    let mut result = LightCoverage {
        words: 0,
        intervals: 0,
        gap: f64::INFINITY,
        limiting: [first, last],
    };
    for range in ranges(first, last) {
        for start in range.clone().step_by(4096) {
            certify_nonconstant(bank, start, (start + 4095).min(*range.end()), &mut result);
        }
    }
    assert_eq!(result.words, last - first + 1);
    serde_json::json!({
        "first": first, "last": last, "count": result.words, "intervals": result.intervals,
        "minimum_light_gap": result.gap, "limiting_words": result.limiting,
    })
}

fn guard_report(proof: &Proof<'_>, parent: &Parent, words: Option<&[u32]>) -> serde_json::Value {
    let Parent { first, last, .. } = *parent;
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(8);
    let mut total = Coverage::default();
    for range in ranges(first, last) {
        let (start, end) = (*range.start(), *range.end());
        let selected = words.map(|words| {
            let from = words.partition_point(|word| *word < start);
            let to = words.partition_point(|word| *word <= end);
            &words[from..to]
        });
        if selected.is_some_and(<[u32]>::is_empty) {
            continue;
        }

        let next = AtomicU32::new(0);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut coverage = Coverage::default();
                        if let Some(words) = selected {
                            while let Some(&word) =
                                words.get(next.fetch_add(1, Ordering::Relaxed) as usize)
                            {
                                coverage.accept(word, word, proof.exact(word), true);
                            }
                        } else if start > SINGLE_WRAP_MAX_FOOTPRINT.to_bits() {
                            loop {
                                let word = start + next.fetch_add(1, Ordering::Relaxed);
                                if word > end {
                                    break;
                                }
                                coverage.accept(word, word, proof.exact(word), true);
                            }
                        } else {
                            loop {
                                let a = start + next.fetch_add(1024, Ordering::Relaxed);
                                if a > end {
                                    break;
                                }
                                certify(proof, a, (a + 1023).min(end), &mut coverage);
                            }
                        }
                        coverage
                    })
                })
                .collect();
            for handle in handles {
                let mut covered = handle.join().unwrap();
                total.count += covered.count;
                total.exact += covered.exact;
                total.intervals += covered.intervals;
                total
                    .singleton_rechecks
                    .append(&mut covered.singleton_rechecks);
                if let (Some(span), Some(floor)) = (covered.span, covered.floor) {
                    total.include(span, floor);
                }
            }
        });
    }
    assert_eq!(
        total.count,
        words.map_or(last - first + 1, |words| words.len() as u32)
    );
    total.singleton_rechecks.sort_unstable();
    serde_json::json!({
        "first": first, "last": last, "count": total.count,
        "intervals": total.intervals, "exact": total.exact,
        "span": total.span.map(|bound| bound.span),
        "span_words": total.span.map(|bound| [bound.first, bound.last]),
        "floor": total.floor.map(|bound| bound.floor),
        "floor_words": total.floor.map(|bound| [bound.first, bound.last]),
        "wavelength_nm": total.floor.map(|bound| bound.wavelength_nm),
        "words": words, "singleton_rechecks": total.singleton_rechecks,
    })
}

pub(super) fn run(assets: &Assets, bank: &GrainV2Bank, development: &film::Development) {
    let Some(bundle) = std::env::var_os("FJ_PORTRA_CERTIFICATE") else {
        return;
    };
    let dir = Path::new(&bundle);
    let request: Request = serde_json::from_reader(
        std::fs::File::open(dir.join("portra-certificate-request.json")).unwrap(),
    )
    .unwrap();
    let bytes = if request.mode == Mode::Exact {
        std::fs::read(dir.join("portra-certificate-input.u32le")).unwrap()
    } else {
        Vec::new()
    };
    let (raw_words, remaining) = bytes.as_chunks::<4>();
    assert!(remaining.is_empty() && !request.parents.is_empty());
    let mut raw_words = raw_words.iter().copied().map(u32::from_le_bytes);
    let gain = AMOUNT_EV_RANGE.end().exp2() / TEXEL_PITCH_UM_RANGE.start();
    let proof = Proof {
        assets,
        bank,
        development,
        gain,
        template: (*assets
            .grain_v2_table(bank, MIN_PROXY_SAMPLE_FOOTPRINT)
            .unwrap())
        .clone(),
    };
    let mut parents = Vec::with_capacity(request.parents.len());
    for parent in &request.parents {
        assert!(
            MIN_PROXY_SAMPLE_FOOTPRINT.to_bits() <= parent.first
                && parent.first <= parent.last
                && parent.last <= MAX_FOOTPRINT.to_bits()
        );
        let outcome = match request.mode {
            Mode::Nonconstant => nonconstant_report(bank, parent),
            Mode::Interval => guard_report(&proof, parent, None),
            Mode::Exact => {
                let count = parent.words.unwrap();
                let mut words: Vec<_> = raw_words.by_ref().take(count).collect();
                assert_eq!(words.len(), count);
                words.sort_unstable();
                assert!(words.windows(2).all(|pair| pair[0] < pair[1]));
                assert!(
                    words
                        .iter()
                        .all(|word| (parent.first..=parent.last).contains(word))
                );
                guard_report(&proof, parent, Some(&words))
            }
        };
        parents.push(outcome);
    }
    assert!(raw_words.next().is_none());
    let report = serde_json::json!({
        "schema": "portra-certificate-report-v1",
        "bank": format!("{:016x}", bank.identity()),
        "gain": gain,
        "mode": request.mode,
        "parents": parents,
    });
    let file = std::fs::File::create_new(dir.join("portra-certificate-report.json")).unwrap();
    serde_json::to_writer(file, &report).unwrap();
}
