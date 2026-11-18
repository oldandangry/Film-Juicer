//! Portra Film preview spectral enclosure witness. The host reader follows the
//! CUDA PCHIP operations and uses the actual Rust-produced six-array LUT.

use super::{Lut, LutInput, LutTables};
use crate::{
    assets::Assets,
    film::{self, density::Bounds},
    route::ScanRoute,
    spectral,
};

fn hermite(a: f32, b: f32, ma: f32, mb: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * a
        + (t3 - 2.0 * t2 + t) * ma
        + (-2.0 * t3 + 3.0 * t2) * b
        + (t3 - t2) * mb
}
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

// Mirrors JuicerCudaDeviceHelpers.cuh's PCHIP reader, including cell limiting.
// This is a host witness; the GPU consumer still needs its own route test.
fn sample(lut: &Lut, normalized: [f32; 3]) -> [f32; 3] {
    let r = lut.resolution() as usize;
    let axis = normalized.map(|x| {
        assert!((0.0..=1.0).contains(&x), "out-of-bounds density {x}");
        let p = x * (r - 1) as f32;
        if p >= (r - 1) as f32 {
            (r - 2, 1.0)
        } else {
            let i = p.floor() as usize;
            (i, p - i as f32)
        }
    });
    let [(c, tc), (m, tm), (y, ty)] = axis;
    std::array::from_fn(|channel| {
        let get = |v: &[f32], dc: usize, dm: usize, dy: usize| {
            v[((((c + dc) * r + m + dm) * r + y + dy) * 3) + channel]
        };
        let along_c = |dm, dy| {
            hermite(
                get(lut.log2_xyz(), 0, dm, dy),
                get(lut.log2_xyz(), 1, dm, dy),
                get(lut.slope_c(), 0, dm, dy),
                get(lut.slope_c(), 1, dm, dy),
                tc,
            )
        };
        let slope_m = |dm, dy| {
            mix(
                get(lut.slope_m(), 0, dm, dy),
                get(lut.slope_m(), 1, dm, dy),
                tc,
            )
        };
        let value = |dy| {
            hermite(
                along_c(0, dy),
                along_c(1, dy),
                slope_m(0, dy),
                slope_m(1, dy),
                tm,
            )
        };
        let slope_y = |dy| {
            mix(
                mix(
                    get(lut.slope_y(), 0, 0, dy),
                    get(lut.slope_y(), 1, 0, dy),
                    tc,
                ),
                mix(
                    get(lut.slope_y(), 0, 1, dy),
                    get(lut.slope_y(), 1, 1, dy),
                    tc,
                ),
                tm,
            )
        };
        let i = (((c * (r - 1) + m) * (r - 1) + y) * 3) + channel;
        hermite(value(0), value(1), slope_y(0), slope_y(1), ty)
            .clamp(lut.cell_min()[i], lut.cell_max()[i])
            .exp2()
    })
}

#[test]
fn portra_film_envelope_scanner_and_enlarger_spectral_witness() {
    // Predeclared criterion: 1% relative, with 1e-10 absolute XYZ for
    // scanner's logarithmic floor, and 1e-8 absolute print exposure.
    const REL: f64 = 0.01;
    let assets =
        Assets::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
    let film = assets.film("kodak_portra_400").unwrap();
    let paper = assets.print("kodak_portra_endura").unwrap();
    let curves = assets.illuminants().unwrap();
    let d50 = curves.d50().unwrap();
    let tungsten = curves.tungsten().unwrap();
    let cmf = assets.cmf().unwrap();
    assert_eq!(cmf.rows().len(), 81);
    let observer: [[f32; 81]; 3] =
        std::array::from_fn(|c| std::array::from_fn(|i| cmf.rows()[i][c + 1]));
    let tables = spectral::build_tables(spectral::Input {
        dyes_cmy: film.tables().channel_density(),
        observer: spectral::Observer {
            x: &observer[0],
            y: &observer[1],
            z: &observer[2],
        },
        illuminant: d50,
        baseline: Some(spectral::Baseline {
            minimum: film.tables().base_density(),
            midpoint: None,
        }),
        illuminant_hash: 1,
    });
    let development = film::Development::new(
        film.clone(),
        film::Params {
            gamma_factor: 1.0,
            layers_required: false,
            requested_route: ScanRoute::NegativeDirect,
        },
    )
    .unwrap();
    let bounds = Bounds::film_v1(&development, 16.0, 0.5).unwrap();
    let lo = *bounds.minimum_cmy();
    let hi = *bounds.maximum_cmy();
    let inv = *bounds.inv_span_cmy();
    let resolution = bounds.film_v1_scan_resolution(17);
    assert!(resolution <= 128);
    let dyes = tables.dyes_cmy();
    let weights = tables.weighted_xyz();
    let lut = Lut::new(LutInput {
        tables: LutTables {
            eps_c: &dyes[0],
            eps_m: &dyes[1],
            eps_y: &dyes[2],
            weighted_x: &weights[0],
            weighted_y: &weights[1],
            weighted_z: &weights[2],
            baseline_min: Some(tables.baseline_min()),
            sample_count: 81,
            inv_yn: tables.inv_yn(),
        },
        range_min_cmy: lo.map(|x| -x),
        inverse_span_cmy: inv,
        medium: 0,
        resolution,
    })
    .unwrap();
    // Corners, faces, non-grid interiors, and ordinary developed densities.
    let coordinates: [Vec<f32>; 3] = std::array::from_fn(|k| {
        [0.0, 0.173, 0.5, 0.831, 1.0]
            .into_iter()
            .chain(
                [0.04, 0.5, 1.0, 2.0]
                    .into_iter()
                    .map(|d| (d - lo[k]) * inv[k]),
            )
            .collect()
    });
    for &a in &coordinates[0] {
        for &b in &coordinates[1] {
            for &c in &coordinates[2] {
                let normalized = [a, b, c];
                let density: [f32; 3] =
                    std::array::from_fn(|k| lo[k] + normalized[k] * (hi[k] - lo[k]));
                let actual = sample(&lut, normalized);
                let mut xyz = [0.0_f64; 3];
                let mut print64 = [0.0_f64; 3];
                let mut print32 = [0.0_f32; 3];
                for i in 0..81 {
                    let spectral64 = (0..3)
                        .map(|k| f64::from(density[k]) * f64::from(dyes[k][i]))
                        .sum::<f64>()
                        + f64::from(tables.baseline_min()[i]);
                    let light64 = 10.0_f64.powf(-spectral64);
                    let spectral32 = density[0] * dyes[0][i]
                        + density[1] * dyes[1][i]
                        + density[2] * dyes[2][i]
                        + tables.baseline_min()[i];
                    let light32 = (-spectral32 * std::f32::consts::LOG2_10).exp2() * tungsten[i];
                    for k in 0..3 {
                        let weighted = light64 * f64::from(weights[k][i]);
                        if !weighted.is_nan() {
                            xyz[k] += weighted;
                        }
                        let sens = paper.tables().linear_sensitivity()[i][k];
                        if !sens.is_nan() {
                            let print_light = light64 * f64::from(tungsten[i]);
                            if !print_light.is_nan() {
                                print64[k] += print_light * f64::from(sens);
                            }
                            if !light32.is_nan() {
                                print32[k] += light32 * sens;
                            }
                        }
                    }
                }
                for k in 0..3 {
                    let expected = (xyz[k] * f64::from(tables.inv_yn())).max(0.0) + 1e-10;
                    assert!(
                        (f64::from(actual[k]) - expected).abs() <= 1e-10 + REL * expected,
                        "scanner {density:?} channel {k}: {} vs {expected}",
                        actual[k]
                    );
                    assert!(print32[k].is_finite() && print64[k].is_finite());
                    assert!(
                        (f64::from(print32[k]) - print64[k]).abs() <= 1e-8 + REL * print64[k].abs(),
                        "enlarger {density:?} channel {k}: {} vs {}",
                        print32[k],
                        print64[k]
                    );
                }
            }
        }
    }
}
