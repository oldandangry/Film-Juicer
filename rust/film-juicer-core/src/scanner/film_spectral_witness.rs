//! Signed Film v1 density and spectral transmission witnesses.

use crate::assets::FilmGrainStock;
use crate::{
    assets::Assets,
    film::{
        self,
        density::{Bounds, film_v1},
    },
    route::ScanRoute,
};

#[test]
fn portra_small_format_high_gain_bounds() {
    let assets =
        Assets::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
    let film = assets.film("kodak_portra_400").unwrap();
    let bank = assets.film_grain_bank(FilmGrainStock::Portra400).unwrap();
    for route in [ScanRoute::NegativeDirect, ScanRoute::NegativePrint] {
        let development = film::Development::new(
            film.clone(),
            film::Params {
                gamma_factor: 1.0,
                layers_required: false,
                requested_route: route,
            },
        )
        .unwrap();
        for (mm, width, amount) in [
            (8.0_f32, 8192.0, 3),
            (8.0_f32, 4832.0, 3),
            (11.66, 4832.0, 3),
            (8.0, 3840.0, 3),
            (11.66, 3840.0, 3),
            (8.0, 1920.0, 3),
            (11.66, 1920.0, 3),
            (8.0, 4832.0, 2),
            (36.0, 4832.0, 3),
        ] {
            let footprint = (f64::from(mm) * 2000.0 / width) as f32;
            let table = assets.film_grain_table(&bank, footprint).unwrap();
            let gain = 2.0_f32.powi(amount) / 0.5;
            let original = Bounds::film_v1(
                &development,
                film_v1::allowances(&bank, &table, gain).unwrap(),
            );
            if width >= 3840.0 && amount == 3 && mm == 8.0 {
                assert!(matches!(
                    original,
                    Err(film::density::Error::FilmProjectionSpan)
                ));
            } else if width >= 3840.0 && amount == 3 && mm == 11.66 {
                assert!(matches!(
                    original,
                    Err(film::density::Error::FilmSpectralRange)
                ));
            } else {
                assert!(original.is_ok());
            }
            for colour in [0.0, 0.37, 1.0] {
                let bounds = Bounds::for_film_grain(&development, &bank, &table, gain, colour)
                    .unwrap_or_else(|error| panic!("{route:?} {mm} mm {width} px {amount} EV, width {footprint}, Colour {colour}: {error}"));
                println!(
                    "{route:?} {mm} mm {width} px {amount} EV Size 0.5 Colour {colour} width {footprint}"
                );
                verify_density_box(&development, &bounds);
                if let Ok(original) = &original {
                    assert_eq!(original.hash(), bounds.hash());
                }
            }
        }
    }
}

fn verify_density_box(development: &film::Development, bounds: &Bounds) {
    let lo = bounds.minimum_cmy().map(f64::from);
    let hi = bounds.maximum_cmy().map(f64::from);
    let span = hi.into_iter().fold(0.0, f64::max) - lo.into_iter().fold(0.0, f64::min);
    assert!((0.0..=64.0).contains(&span));
    let curves = development.profile().tables();
    let mut minimum = (0, f64::INFINITY);
    // Visit every box corner at every active wavelength. Monotonicity of
    // exp10 and linear density then enclose the entire interior.
    for (i, (dyes, &base)) in curves
        .channel_density()
        .iter()
        .zip(curves.base_density())
        .enumerate()
    {
        if base.is_nan() || dyes.iter().any(|dye| dye.is_nan()) {
            continue;
        }
        for corner in 0..8 {
            let density = (0..3).fold(f64::from(base), |d, k| {
                d + f64::from(dyes[k]) * if corner & (1 << k) == 0 { lo[k] } else { hi[k] }
            });
            assert!(
                density.is_finite() && density >= -30.0,
                "{i} {corner} {density}"
            );
            assert!(10.0_f32.powf(-density as f32).is_finite());
            if density < minimum.1 {
                minimum = (i, density);
            }
        }
    }
    assert!(minimum.1.is_finite());
    println!(
        "CMY {lo:?} to {hi:?}, span {span} D, spectral {} D @ {} nm",
        minimum.1,
        380 + minimum.0 * 5
    );
}

#[test]
fn portra_film_envelope_and_enlarger_transmission() {
    // Keep the portrait control and exercise the 4832-pixel sub-texel
    // requests at the maximum Amount for their Size.
    spectral_witness((36_000.0 / 7418.0 / 0.5) as f32, 16.0, 1.0);
    spectral_witness((8_000.0 / 4832.0 / 4.0) as f32, 2.0, 1.0);
    spectral_witness(
        (f64::from(11.66_f32) * 1000.0 / 4832.0 / 4.0) as f32,
        2.0,
        1.0,
    );
    spectral_witness((240_000.0 / 92.0) as f32, 16.0, 1.0);
    for width in [3840.0, 4832.0] {
        for mm in [8.0_f32, 11.66] {
            spectral_witness((f64::from(mm) * 2000.0 / width) as f32, 16.0, 1.0);
        }
    }
    spectral_witness((16_000.0 / 4832.0) as f32, 16.0, 0.37);
    spectral_witness(1.953_125, 16.0, 0.0);
    spectral_witness(1.953_125, 16.0, 1.0);
    spectral_witness(1.25, 16.0, 1.0);
    for edge in [37.5_f32, 255.0, 256.0, 1536.0] {
        for width in [edge.next_down(), edge, edge.next_up()] {
            spectral_witness(width, 16.0, 1.0);
        }
    }
}

#[test]
fn gold_film_envelope_and_enlarger_transmission() {
    for (footprint, gain, colour) in [
        (9.375, 1.0, 1.0),
        (16_000.0 / 4832.0, 16.0, 1.0),
        (16_000.0 / 3840.0, 16.0, 0.37),
        (1.953_125, 16.0, 0.0),
        (0.25, 2.0, 1.0),
        (37.5_f32.next_up(), 16.0, 1.0),
        (255.0_f32.next_up(), 16.0, 1.0),
        (256.0, 16.0, 0.5),
        (240_000.0 / 92.0, 16.0, 1.0),
        (240_000.0, 16.0, 1.0),
    ] {
        stock_spectral_witness(FilmGrainStock::Gold200, footprint, gain, colour);
    }
}

fn spectral_witness(footprint: f32, gain: f32, colour: f32) {
    stock_spectral_witness(FilmGrainStock::Portra400, footprint, gain, colour);
}

fn stock_spectral_witness(stock: FilmGrainStock, footprint: f32, gain: f32, colour: f32) {
    const REL: f64 = 0.01;
    let assets =
        Assets::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
    let film = assets.film(stock.key()).unwrap();
    let paper = assets.print("kodak_portra_endura").unwrap();
    let curves = assets.illuminants().unwrap();
    let tungsten = curves.tungsten().unwrap();
    let d50 = curves.d50().unwrap();
    let cmf = assets.cmf().unwrap();
    let observer: [[f32; 3]; 81] = std::array::from_fn(|i| {
        let wavelength = 380.0 + i as f32 * 5.0;
        let row = cmf.rows().iter().find(|row| row[0] == wavelength).unwrap();
        [row[1], row[2], row[3]]
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
    let bank = assets.film_grain_bank(stock).unwrap();
    let table = assets.film_grain_table(&bank, footprint).unwrap();
    let bounds =
        Bounds::for_film_grain(&development, &bank, &table, gain, colour).unwrap_or_else(|error| {
            panic!("{stock:?}, footprint {footprint}, gain {gain}, Colour {colour}: {error}")
        });
    println!("spectral witness footprint {footprint}, gain {gain}, Colour {colour}");
    verify_density_box(&development, &bounds);
    let lo = *bounds.minimum_cmy();
    let hi = *bounds.maximum_cmy();
    let dyes = film.tables().channel_density();
    let base = film.tables().base_density();
    let densities: [Vec<f32>; 3] = std::array::from_fn(|k| {
        [lo[k], 0.04, 0.5, 1.0, 2.0, hi[k]]
            .into_iter()
            .chain([0.173, 0.5, 0.831].map(|t| lo[k] + t * (hi[k] - lo[k])))
            .filter(|density| (lo[k]..=hi[k]).contains(density))
            .collect()
    });
    for &c in &densities[0] {
        for &m in &densities[1] {
            for &y in &densities[2] {
                let density = [c, m, y];
                let mut print64 = [0.0_f64; 3];
                let mut print32 = [0.0_f32; 3];
                let mut xyz64 = [0.0_f64; 3];
                let mut xyz32 = [0.0_f32; 3];
                for i in 0..81 {
                    let spectral64 = f64::from(density[0]) * f64::from(dyes[i][0])
                        + f64::from(density[1]) * f64::from(dyes[i][1])
                        + f64::from(density[2]) * f64::from(dyes[i][2])
                        + f64::from(base[i]);
                    let transmission64 = 10.0_f64.powf(-spectral64);
                    let print_light = transmission64 * f64::from(tungsten[i]);
                    let spectral32 = density[0] * dyes[i][0]
                        + density[1] * dyes[i][1]
                        + density[2] * dyes[i][2]
                        + base[i];
                    let transmission32 = (-spectral32 * std::f32::consts::LOG2_10).exp2();
                    let light32 = transmission32 * tungsten[i];
                    for k in 0..3 {
                        // Check D50 scanner XYZ using this stock's actual dyes,
                        // with a separate Float64 spectral integration.
                        if transmission64.is_finite() && transmission32.is_finite() {
                            xyz64[k] +=
                                transmission64 * f64::from(d50[i]) * f64::from(observer[i][k]);
                            xyz32[k] += transmission32 * (d50[i] * observer[i][k]);
                        }
                        let sens = paper.tables().linear_sensitivity()[i][k];
                        if !sens.is_nan() {
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
                    assert!(xyz32[k].is_finite() && xyz64[k].is_finite());
                    assert!(
                        (f64::from(xyz32[k]) - xyz64[k]).abs() <= 1e-10 + REL * xyz64[k].abs(),
                        "scanner XYZ {density:?} channel {k}: {} vs {}",
                        xyz32[k],
                        xyz64[k]
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
