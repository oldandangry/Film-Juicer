//! Complete print illumination, balance and raw preflash exposure on the fixed spectral axis.

use crate::hash;
use crate::profile::PrintDensityCurves;

pub(crate) const CURRENT_NEUTRAL_CMY_CC: [f32; 3] = [0.0, 65.0, 55.0];
const DICHROIC_EDGES_NM: [f32; 4] = [516.0, 500.0, 610.0, 607.0];
const DICHROIC_TRANSITIONS_NM: [f32; 4] = [12.0, 8.0, 8.0, 8.0];

/// Remaining print-route composition, separate from the accepted leaf producers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionError {
    DensityBounds,
    RecipeIdentity,
    ResourceDescriptors,
}
impl std::fmt::Display for CompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::DensityBounds => {
                "MalformedRequiredProfileData phase=4C field=print_density_bounds"
            }
            Self::RecipeIdentity => "ResourceDescriptorMismatch phase=4A field=print_recipe_hash",
            Self::ResourceDescriptors => {
                "ResourceDescriptorMismatch phase=4B field=print_resource_descriptors"
            }
        })
    }
}
impl std::error::Error for CompositionError {}

pub struct Bounds {
    minimum_cmy: [f32; 3],
    maximum_cmy: [f32; 3],
    inv_span_cmy: [f32; 3],
    hash: u64,
}
impl Bounds {
    pub fn new(
        profile: &crate::profile::PrintProfile,
        route: crate::route::ScanRoute,
        polarity: crate::profile::Polarity,
    ) -> Result<Self, CompositionError> {
        let curves = profile.tables().density_curves();
        let mut minimum_cmy = curves[0];
        let mut maximum_cmy = curves[0];
        for row in curves {
            for c in 0..3 {
                // Ordered comparisons retain first-operand NaNs and zero signs.
                if row[c] < minimum_cmy[c] {
                    minimum_cmy[c] = row[c];
                }
                if maximum_cmy[c] < row[c] {
                    maximum_cmy[c] = row[c];
                }
            }
        }
        let mut inv_span_cmy = [0.0; 3];
        for c in 0..3 {
            let span = maximum_cmy[c] - minimum_cmy[c];
            if !(span.is_finite() && span > 0.0) {
                return Err(CompositionError::DensityBounds);
            }
            inv_span_cmy[c] = 1.0 / span;
        }
        let route = match route {
            crate::route::ScanRoute::NegativeDirect => 0,
            crate::route::ScanRoute::NegativePrint => 1,
            crate::route::ScanRoute::PositiveDirect => 2,
            crate::route::ScanRoute::PositivePrint => 3,
        };
        let polarity = match polarity {
            crate::profile::Polarity::Negative => 0,
            crate::profile::Polarity::Positive => 1,
        };
        let mut hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut hash, &[route, 1, polarity, 2]);
        for values in [
            &minimum_cmy,
            &maximum_cmy,
            &inv_span_cmy,
            &minimum_cmy,
            &maximum_cmy,
        ] {
            for value in values {
                hash::update_bytes(&mut hash, &value.to_bits().to_le_bytes());
            }
        }
        if hash == 0 {
            return Err(CompositionError::DensityBounds);
        }
        Ok(Self {
            minimum_cmy,
            maximum_cmy,
            inv_span_cmy,
            hash,
        })
    }
    pub fn minimum_cmy(&self) -> &[f32; 3] {
        &self.minimum_cmy
    }
    pub fn maximum_cmy(&self) -> &[f32; 3] {
        &self.maximum_cmy
    }
    pub fn inv_span_cmy(&self) -> &[f32; 3] {
        &self.inv_span_cmy
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

pub struct Filters {
    main_cmy_cc: [f32; 3],
    preflash_cmy_cc: [f32; 3],
    dichroic_hash: u64,
    hash: u64,
}
impl Filters {
    pub fn new(
        neutral_cmy_cc: [f32; 3],
        ui_ymc_cc: [f32; 3],
        preflash_m_cc: f32,
        preflash_y_cc: f32,
    ) -> Result<Self, CompositionError> {
        let main_cmy_cc = std::array::from_fn(|c| neutral_cmy_cc[c] + ui_ymc_cc[2 - c]);
        let preflash_cmy_cc = [
            neutral_cmy_cc[0],
            neutral_cmy_cc[1] + preflash_m_cc,
            neutral_cmy_cc[2] + preflash_y_cc,
        ];
        let mut dichroic_hash = hash::FNV_OFFSET;
        for value in DICHROIC_EDGES_NM.iter().chain(&DICHROIC_TRANSITIONS_NM) {
            hash::update_bytes(&mut dichroic_hash, &value.to_bits().to_le_bytes());
        }
        let mut hash = hash::FNV_OFFSET;
        for value in main_cmy_cc.iter().chain(&preflash_cmy_cc) {
            hash::update_bytes(&mut hash, &value.to_bits().to_le_bytes());
        }
        hash::update_bytes(&mut hash, &dichroic_hash.to_le_bytes());
        if hash == 0 {
            return Err(CompositionError::RecipeIdentity);
        }
        Ok(Self {
            main_cmy_cc,
            preflash_cmy_cc,
            dichroic_hash,
            hash,
        })
    }
    pub fn main_cmy_cc(&self) -> &[f32; 3] {
        &self.main_cmy_cc
    }
    pub fn preflash_cmy_cc(&self) -> &[f32; 3] {
        &self.preflash_cmy_cc
    }
    pub fn dichroic_hash(&self) -> u64 {
        self.dichroic_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn illumination_params(&self, preflash: bool) -> FilterParams {
        FilterParams {
            edges_nm: DICHROIC_EDGES_NM,
            transitions_nm: DICHROIC_TRANSITIONS_NM,
            cmy_cc: if preflash {
                self.preflash_cmy_cc
            } else {
                self.main_cmy_cc
            },
        }
    }
}

#[derive(Clone, Copy)]
pub struct ExposureParams {
    pub print_exposure: f32,
    pub preflash_exposure: f32,
    pub camera_exposure_ev: f32,
    pub normalize: bool,
    pub compensate: bool,
}
pub struct Exposure {
    params: ExposureParams,
    normalization_mode: u8,
    hash: u64,
}
impl Exposure {
    pub fn new(params: ExposureParams) -> Result<Self, CompositionError> {
        let normalization_mode = u8::from(params.normalize) * 2 + u8::from(params.compensate);
        let mut hash = hash::FNV_OFFSET;
        for value in [
            params.print_exposure,
            params.preflash_exposure,
            params.camera_exposure_ev,
        ] {
            hash::update_bytes(&mut hash, &value.to_bits().to_le_bytes());
        }
        hash::update_bytes(
            &mut hash,
            &[
                u8::from(params.normalize),
                u8::from(params.compensate),
                normalization_mode,
            ],
        );
        if hash == 0 {
            return Err(CompositionError::RecipeIdentity);
        }
        Ok(Self {
            params,
            normalization_mode,
            hash,
        })
    }
    pub fn params(&self) -> &ExposureParams {
        &self.params
    }
    pub fn normalization_mode(&self) -> u8 {
        self.normalization_mode
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

pub struct ResourceDescriptorInput<'a> {
    pub film: &'a crate::profile::FilmProfile,
    pub profile: &'a crate::profile::PrintProfile,
    pub curves: &'a PrintDensityCurves,
    pub filters: &'a Filters,
    pub exposure: &'a Exposure,
    pub balance: &'a Balance,
    pub film_input_hash: u64,
    pub film_development_hash: u64,
    pub illuminant_hash: u64,
}

pub struct ResourceDescriptors {
    sensitivities_hash: u64,
    density_tables_hash: u64,
    profile_tables_hash: u64,
    film_density_hash: u64,
    main_illuminant_hash: u64,
    preflash_illuminant_hash: u64,
    preflash_raw_hash: u64,
    balance_hash: u64,
    hash: u64,
}
impl ResourceDescriptors {
    pub fn new(input: ResourceDescriptorInput<'_>) -> Result<Self, CompositionError> {
        let sensitivities =
            hash::f32s_with_nan_mask(input.profile.tables().linear_sensitivity().as_flattened());
        let sensitivities_hash = hash::u64s(&[sensitivities.values, sensitivities.nan_mask]);
        let dyes = hash::f32s_with_nan_mask(input.film.tables().channel_density().as_flattened());
        let base = hash::f32s_with_nan_mask(input.film.tables().base_density());
        let density_tables_hash = hash::u64s(&[
            hash::resource_f32s(input.film.tables().wavelengths()),
            dyes.values,
            dyes.nan_mask,
            base.values,
            base.nan_mask,
        ]);
        let mut profile_tables_hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut profile_tables_hash, &2_u32.to_le_bytes());
        for word in [
            input.profile.asset_token(),
            input.curves.hash(),
            sensitivities_hash,
        ] {
            hash::update_bytes(&mut profile_tables_hash, &word.to_le_bytes());
        }
        hash::update_bytes(
            &mut profile_tables_hash,
            &(input.curves.totals().len() as u32).to_le_bytes(),
        );
        hash::update_bytes(&mut profile_tables_hash, &81_u32.to_le_bytes());
        let mut film_density_hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut film_density_hash, &1_u32.to_le_bytes());
        for word in [input.film.asset_token(), density_tables_hash] {
            hash::update_bytes(&mut film_density_hash, &word.to_le_bytes());
        }
        hash::update_bytes(&mut film_density_hash, &81_u32.to_le_bytes());
        let filtered = |preflash: bool| {
            let mut identity = hash::FNV_OFFSET;
            hash::update_bytes(&mut identity, &1_u32.to_le_bytes());
            for word in [input.illuminant_hash, input.filters.dichroic_hash()] {
                hash::update_bytes(&mut identity, &word.to_le_bytes());
            }
            for value in if preflash {
                input.filters.preflash_cmy_cc()
            } else {
                input.filters.main_cmy_cc()
            } {
                hash::update_bytes(&mut identity, &value.to_bits().to_le_bytes());
            }
            hash::update_bytes(&mut identity, &[u8::from(preflash)]);
            identity
        };
        let main_illuminant_hash = filtered(false);
        let (preflash_illuminant_hash, preflash_raw_hash) =
            if input.exposure.params().preflash_exposure > 0.0 {
                let illuminant = filtered(true);
                let mut raw = hash::FNV_OFFSET;
                hash::update_bytes(&mut raw, &2_u32.to_le_bytes());
                for word in [
                    input.film.asset_token(),
                    input.profile.asset_token(),
                    illuminant,
                ] {
                    hash::update_bytes(&mut raw, &word.to_le_bytes());
                }
                (illuminant, raw)
            } else {
                (0, 0)
            };
        let mut balance_hash = hash::FNV_OFFSET;
        hash::update_bytes(&mut balance_hash, &3_u32.to_le_bytes());
        for word in [
            input.film.asset_token(),
            input.film_input_hash,
            input.film_development_hash,
            input.profile.asset_token(),
            main_illuminant_hash,
        ] {
            hash::update_bytes(&mut balance_hash, &word.to_le_bytes());
        }
        hash::update_bytes(&mut balance_hash, &[input.exposure.normalization_mode()]);
        let camera_ev = if input.exposure.params().compensate {
            input.exposure.params().camera_exposure_ev
        } else {
            0.0
        };
        hash::update_bytes(&mut balance_hash, &camera_ev.to_bits().to_le_bytes());
        hash::update_bytes(&mut balance_hash, &input.balance.hash().to_le_bytes());
        let hash = hash::u64s(&[
            profile_tables_hash,
            film_density_hash,
            main_illuminant_hash,
            preflash_illuminant_hash,
            preflash_raw_hash,
            balance_hash,
        ]);
        if [
            input.curves.hash(),
            sensitivities_hash,
            profile_tables_hash,
            input.film.asset_token(),
            density_tables_hash,
            film_density_hash,
            main_illuminant_hash,
            balance_hash,
            input.film_input_hash,
            input.film_development_hash,
            input.balance.hash(),
            hash,
        ]
        .contains(&0)
            || input.curves.totals().is_empty()
            || (input.exposure.params().preflash_exposure > 0.0
                && (preflash_illuminant_hash == 0 || preflash_raw_hash == 0))
        {
            return Err(CompositionError::ResourceDescriptors);
        }
        Ok(Self {
            sensitivities_hash,
            density_tables_hash,
            profile_tables_hash,
            film_density_hash,
            main_illuminant_hash,
            preflash_illuminant_hash,
            preflash_raw_hash,
            balance_hash,
            hash,
        })
    }
    pub fn sensitivities_hash(&self) -> u64 {
        self.sensitivities_hash
    }
    pub fn density_tables_hash(&self) -> u64 {
        self.density_tables_hash
    }
    pub fn profile_tables_hash(&self) -> u64 {
        self.profile_tables_hash
    }
    pub fn film_density_hash(&self) -> u64 {
        self.film_density_hash
    }
    pub fn main_illuminant_hash(&self) -> u64 {
        self.main_illuminant_hash
    }
    pub fn preflash_illuminant_hash(&self) -> u64 {
        self.preflash_illuminant_hash
    }
    pub fn preflash_raw_hash(&self) -> u64 {
        self.preflash_raw_hash
    }
    pub fn balance_hash(&self) -> u64 {
        self.balance_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BalanceParams {
    pub baseline_density_cmy: [f32; 3],
    pub compensated_density_cmy: [f32; 3],
    pub reference_hash: u64,
    pub normalization_mode: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BalanceError {
    Baseline { channel: usize },
    Compensated { channel: usize },
    Invalid,
}

/// A completed print balance; reference/scanner/whole-state completion is separate.
#[derive(Debug)]
pub struct Balance {
    baseline_raw_rgb: [f32; 3],
    compensated_raw_rgb: [f32; 3],
    factor_midgray: f32,
    factor_midgray_comp: f32,
    normalizer: f32,
    filtered_main_illuminant_hash: u64,
    hash: u64,
}

impl Balance {
    pub fn new(
        params: BalanceParams,
        film_base_density: &[f32; 81],
        film_channel_density_cmy: &[[f32; 3]; 81],
        print_sensitivity_rgb: &[[f32; 3]; 81],
        filtered_main_illuminant: &[f32; 81],
    ) -> Result<Self, BalanceError> {
        let integrate = |density_cmy: &[f32; 3]| -> Result<[f32; 3], usize> {
            let mut accumulated = [0.0_f64; 3];
            for sample in 0..81 {
                let mut density = f64::from(film_base_density[sample]);
                for (channel, reference) in density_cmy.iter().enumerate() {
                    density += f64::from(*reference)
                        * f64::from(film_channel_density_cmy[sample][channel]);
                }
                let light = 10.0_f64.powf(-density) * f64::from(filtered_main_illuminant[sample]);
                if light.is_nan() {
                    continue;
                }
                for (channel, total) in accumulated.iter_mut().enumerate() {
                    let sensitivity = print_sensitivity_rgb[sample][channel];
                    if sensitivity.is_finite() {
                        *total += light * f64::from(sensitivity);
                    }
                }
            }
            let mut raw = [0.0_f32; 3];
            for channel in 0..3 {
                // Balance admits the narrowed value, including negative f64 -> -0.
                raw[channel] = accumulated[channel] as f32;
                if !raw[channel].is_finite() || raw[channel] < 0.0 {
                    return Err(channel);
                }
            }
            Ok(raw)
        };
        let baseline_raw_rgb = integrate(&params.baseline_density_cmy)
            .map_err(|channel| BalanceError::Baseline { channel })?;
        let compensated_raw_rgb = integrate(&params.compensated_density_cmy)
            .map_err(|channel| BalanceError::Compensated { channel })?;
        let factor = |raw: &[f32; 3]| {
            let mut mean_log = 0.0_f64;
            for value in raw {
                mean_log += f64::from(*value).max(1.0e-10).ln();
            }
            (-mean_log / 3.0).exp() as f32
        };
        let factor_midgray = factor(&baseline_raw_rgb);
        let factor_midgray_comp = factor(&compensated_raw_rgb);
        let normalizer = match params.normalization_mode {
            0 => 1.0,
            1 => factor_midgray_comp / factor_midgray,
            2 => factor_midgray,
            3 => factor_midgray_comp,
            _ => f32::NAN,
        };
        let filtered_main_illuminant_hash =
            hash::finite_f32s(filtered_main_illuminant).unwrap_or(0);
        let hash = hash::u64s(&[
            params.reference_hash,
            filtered_main_illuminant_hash,
            hash::finite_f32s(&baseline_raw_rgb).unwrap_or(0),
            hash::finite_f32s(&compensated_raw_rgb).unwrap_or(0),
            u64::from(factor_midgray.to_bits()),
            u64::from(factor_midgray_comp.to_bits()),
            u64::from(normalizer.to_bits()),
        ]);
        if !factor_midgray.is_finite()
            || factor_midgray <= 0.0
            || !factor_midgray_comp.is_finite()
            || factor_midgray_comp <= 0.0
            || !normalizer.is_finite()
            || normalizer <= 0.0
            || filtered_main_illuminant_hash == 0
            || hash == 0
        {
            return Err(BalanceError::Invalid);
        }
        Ok(Self {
            baseline_raw_rgb,
            compensated_raw_rgb,
            factor_midgray,
            factor_midgray_comp,
            normalizer,
            filtered_main_illuminant_hash,
            hash,
        })
    }

    pub fn baseline_raw_rgb(&self) -> &[f32; 3] {
        &self.baseline_raw_rgb
    }
    pub fn compensated_raw_rgb(&self) -> &[f32; 3] {
        &self.compensated_raw_rgb
    }
    pub fn factor_midgray(&self) -> f32 {
        self.factor_midgray
    }
    pub fn factor_midgray_comp(&self) -> f32 {
        self.factor_midgray_comp
    }
    pub fn normalizer(&self) -> f32 {
        self.normalizer
    }
    pub fn filtered_main_illuminant_hash(&self) -> u64 {
        self.filtered_main_illuminant_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreflashError {
    Channel { channel: usize },
}

/// The completed integral before exposure scaling and executable f32 admission.
#[derive(Debug)]
pub struct PreflashRaw {
    cmy: [f32; 3],
}

impl PreflashRaw {
    pub fn new(
        film_base_density: &[f32; 81],
        filtered_illuminant: &[f32; 81],
        print_sensitivity_cmy: &[[f32; 3]; 81],
    ) -> Result<Self, PreflashError> {
        let mut raw = [0.0_f64; 3];
        for sample in 0..81 {
            let density = f64::from(film_base_density[sample]);
            let illuminant = f64::from(filtered_illuminant[sample]);
            let transmitted = 10.0_f64.powf(-density) * illuminant;
            let light = if transmitted.is_nan() {
                0.0
            } else {
                transmitted
            };
            for (channel, total) in raw.iter_mut().enumerate() {
                let sensitivity = print_sensitivity_cmy[sample][channel];
                if sensitivity.is_finite() {
                    *total += light * f64::from(sensitivity);
                }
            }
        }
        let mut cmy = [0.0_f32; 3];
        for channel in 0..3 {
            if !raw[channel].is_finite() || raw[channel] < 0.0 {
                return Err(PreflashError::Channel { channel });
            }
            // Finite f64 totals may narrow to infinity; payload admission is later.
            cmy[channel] = raw[channel] as f32;
        }
        Ok(Self { cmy })
    }

    pub fn cmy(&self) -> &[f32; 3] {
        &self.cmy
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FilterParams {
    pub edges_nm: [f32; 4],
    pub transitions_nm: [f32; 4],
    pub cmy_cc: [f32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Sample { sample: usize },
    Energy,
}

#[derive(Debug)]
pub struct Illuminant {
    samples: [f32; 81],
}

impl Illuminant {
    pub fn new(params: FilterParams, source: &[f32; 81]) -> Result<Self, Error> {
        let mut filters = [[0.0_f32; 3]; 81];
        for (sample, filter) in filters.iter_mut().enumerate() {
            let wavelength = 380.0_f32 + 5.0_f32 * sample as f32;
            let y = 0.5_f32
                * libm::erff((wavelength - params.edges_nm[0]) / params.transitions_nm[0])
                + 0.5_f32;
            let m_erf = if wavelength <= 550.0_f32 {
                -libm::erff((wavelength - params.edges_nm[1]) / params.transitions_nm[1])
            } else {
                libm::erff((wavelength - params.edges_nm[2]) / params.transitions_nm[2])
            };
            let c = -0.5_f32
                * libm::erff((wavelength - params.edges_nm[3]) / params.transitions_nm[3])
                + 0.5_f32;
            filter[0] = c;
            filter[1] = 0.5_f32 * m_erf + 0.5_f32;
            filter[2] = y;
        }
        let transmittance = [
            10.0_f32.powf(-params.cmy_cc[0] / 100.0_f32),
            10.0_f32.powf(-params.cmy_cc[1] / 100.0_f32),
            10.0_f32.powf(-params.cmy_cc[2] / 100.0_f32),
        ];
        let mut samples = [0.0_f32; 81];
        let mut energy = 0.0_f32;
        for sample in 0..81 {
            let mut total = source[sample];
            for channel in 0..3 {
                total *= 1.0_f32
                    - (1.0_f32 - filters[sample][channel]) * (1.0_f32 - transmittance[channel]);
            }
            if !total.is_finite() || total < 0.0_f32 {
                return Err(Error::Sample { sample });
            }
            samples[sample] = total;
            energy += total;
        }
        if !energy.is_finite() || energy <= 0.0_f32 {
            return Err(Error::Energy);
        }
        Ok(Self { samples })
    }

    pub fn samples(&self) -> &[f32; 81] {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, FilterParams, Illuminant};
    use serde_json::Value;

    #[cfg(all(target_os = "linux", debug_assertions))]
    const SELECTED: &str = "linux-debug-rust-literal.json";
    #[cfg(all(target_os = "linux", not(debug_assertions)))]
    const SELECTED: &str = "linux-release-rust-literal.json";
    #[cfg(all(target_os = "windows", debug_assertions))]
    const SELECTED: &str = "windows-clang-debug-rust-literal.json";
    #[cfg(all(target_os = "windows", not(debug_assertions)))]
    const SELECTED: &str = "windows-clang-release-rust-literal.json";

    fn floats<const N: usize>(value: &Value) -> [f32; N] {
        let words = value.as_array().unwrap();
        assert_eq!(words.len(), N);
        std::array::from_fn(|i| f32::from_bits(words[i].as_u64().unwrap() as u32))
    }

    #[test]
    fn selected_characterization() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/print/illuminant-fixtures")
            .join(SELECTED);
        let document: Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut differences = Vec::new();
        for row in document["cases"].as_array().unwrap() {
            let params = FilterParams {
                edges_nm: floats(&row["edges_nm"]),
                transitions_nm: floats(&row["transitions_nm"]),
                cmy_cc: floats(&row["cmy_cc"]),
            };
            let source = floats(&row["source"]);
            let expected = floats::<81>(&row["samples"]);
            let result = Illuminant::new(params, &source);
            let id = row["id"].as_str().unwrap();
            if row["success"].as_bool().unwrap() {
                match result {
                    Ok(value) => {
                        for (sample, (&actual, &selected)) in
                            value.samples().iter().zip(&expected).enumerate()
                        {
                            if actual.to_bits() != selected.to_bits() {
                                differences.push(format!(
                                    "{id} sample={sample} selected={:08x} rust={:08x}",
                                    selected.to_bits(),
                                    actual.to_bits()
                                ));
                            }
                        }
                    }
                    Err(error) => differences.push(format!("{id} unexpected {error:?}")),
                }
            } else {
                // Native failures leave the unvisited suffix at its captured -13
                // seed. Energy failures have written all samples before failing.
                let failure = expected
                    .iter()
                    .position(|value| value.to_bits() == 0xc150_0000)
                    .map_or(Error::Energy, |sample| Error::Sample { sample });
                if result.err() != Some(failure) {
                    differences.push(format!("{id} expected {failure:?}"));
                }
            }
        }
        assert!(
            differences.is_empty(),
            "exact owner-approved print policy gate failed:\n{}",
            differences.join("\n")
        );
    }
}

#[cfg(test)]
mod preflash_tests {
    use super::{PreflashError, PreflashRaw};
    use serde_json::Value;

    #[cfg(all(target_os = "linux", debug_assertions))]
    const PRESET: &str = "linux-debug";
    #[cfg(all(target_os = "linux", not(debug_assertions)))]
    const PRESET: &str = "linux-release";
    #[cfg(all(target_os = "windows", debug_assertions))]
    const PRESET: &str = "windows-clang-debug";
    #[cfg(all(target_os = "windows", not(debug_assertions)))]
    const PRESET: &str = "windows-clang-release";

    fn floats<const N: usize>(value: &Value) -> [f32; N] {
        let words = value.as_array().unwrap();
        assert_eq!(words.len(), N);
        std::array::from_fn(|i| f32::from_bits(words[i].as_u64().unwrap() as u32))
    }

    #[test]
    fn exact_native_literals_and_actual_caller_inputs() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/print/preflash-fixtures");
        let manifest: Value = serde_json::from_str(
            &std::fs::read_to_string(directory.join("manifest.json")).unwrap(),
        )
        .unwrap();
        let captured: Value = serde_json::from_str(
            &std::fs::read_to_string(
                directory.join(manifest["presets"][PRESET]["native"].as_str().unwrap()),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(captured["cases"].as_array().unwrap().len(), 19);
        assert_eq!(captured["products"].as_array().unwrap().len(), 72);
        for row in captured["cases"].as_array().unwrap() {
            let base = floats(&row["film_base_density"]);
            let light = floats(&row["filtered_illuminant"]);
            let flat = floats::<243>(&row["print_sensitivity_cmy"]);
            let sensitivity = std::array::from_fn(|i| std::array::from_fn(|c| flat[i * 3 + c]));
            let result = PreflashRaw::new(&base, &light, &sensitivity);
            let id = row["id"].as_str().unwrap();
            if row["success"].as_bool().unwrap() {
                assert_eq!(
                    result.unwrap().cmy().map(f32::to_bits),
                    floats::<3>(&row["raw"]).map(f32::to_bits),
                    "{id}"
                );
            } else {
                let channel = manifest["failure_channels"][id].as_u64().unwrap() as usize;
                assert_eq!(
                    result.err(),
                    Some(PreflashError::Channel { channel }),
                    "{id}"
                );
            }
        }
        for row in captured["products"].as_array().unwrap() {
            if row["raw"].is_null() {
                continue;
            }
            let base = floats(&row["film_base_density"]);
            let light = floats(&row["light"]);
            let flat = floats::<243>(&row["print_sensitivity_cmy"]);
            let sensitivity = std::array::from_fn(|i| std::array::from_fn(|c| flat[i * 3 + c]));
            assert_eq!(
                PreflashRaw::new(&base, &light, &sensitivity)
                    .unwrap()
                    .cmy()
                    .map(f32::to_bits),
                floats::<3>(&row["raw"]).map(f32::to_bits)
            );
        }
    }

    #[test]
    fn owns_result_and_preserves_computed_failure_boundary() {
        let raw = {
            let base = [0.0; 81];
            let light = [1.0; 81];
            let sensitivity = [[f32::MAX; 3]; 81];
            PreflashRaw::new(&base, &light, &sensitivity).unwrap()
        };
        assert_eq!(raw.cmy().map(f32::to_bits), [f32::INFINITY.to_bits(); 3]);
        let mut base = [0.0; 81];
        base[0] = -309.0;
        let mut sensitivity = [[1.0; 3]; 81];
        sensitivity[0] = [f32::NAN, f32::INFINITY, 0.0];
        // Only transmitted-light NaNs are replaced; infinity times a finite
        // zero sensitivity still participates and fails the final double guard.
        assert_eq!(
            PreflashRaw::new(&base, &[1.0; 81], &sensitivity).err(),
            Some(PreflashError::Channel { channel: 2 })
        );
    }
}

#[cfg(test)]
mod balance_tests {
    use super::{Balance, BalanceError, BalanceParams};

    fn params(mode: u8) -> BalanceParams {
        BalanceParams {
            baseline_density_cmy: [0.0; 3],
            compensated_density_cmy: [1.0, 0.0, 0.0],
            reference_hash: 0x0123_4567_89ab_cdef,
            normalization_mode: mode,
        }
    }

    #[test]
    fn complete_native_words_and_seven_lane_identity() {
        let balance = Balance::new(
            params(0),
            &[0.0; 81],
            &[[1.0, 0.0, 0.0]; 81],
            &[[1.0, 2.0, 3.0]; 81],
            &[1.0; 81],
        )
        .unwrap();
        // Independently adopted unchanged native defaults, all four profiles.
        assert_eq!(
            balance.baseline_raw_rgb().map(f32::to_bits),
            [1117913088, 1126301696, 1131610112]
        );
        assert_eq!(
            balance.compensated_raw_rgb().map(f32::to_bits),
            [1090623898, 1099012506, 1103259238]
        );
        assert_eq!(
            [
                balance.factor_midgray().to_bits(),
                balance.factor_midgray_comp().to_bits(),
                balance.normalizer().to_bits()
            ],
            [1004445939, 1032529048, 1065353216]
        );
        assert_eq!(
            balance.filtered_main_illuminant_hash(),
            10702640014580087064
        );
        assert_eq!(balance.hash(), 2940845408508787789);
    }

    #[test]
    fn narrowed_raw_and_baseline_failure_precede_mode_failure() {
        let p = params(255);
        assert!(matches!(
            Balance::new(
                p,
                &[0.0; 81],
                &[[1.0, 0.0, 0.0]; 81],
                &[[1.0, -1.0, 1.0]; 81],
                &[1.0; 81]
            ),
            Err(BalanceError::Baseline { channel: 1 })
        ));
        assert!(matches!(
            Balance::new(
                p,
                &[0.0; 81],
                &[[1.0, 0.0, 0.0]; 81],
                &[[1.0; 3]; 81],
                &[1.0; 81]
            ),
            Err(BalanceError::Invalid)
        ));
        assert!(matches!(
            Balance::new(
                params(0),
                &[0.0; 81],
                &[[1.0, 0.0, 0.0]; 81],
                &[[f32::MAX; 3]; 81],
                &[1.0; 81]
            ),
            Err(BalanceError::Baseline { channel: 0 })
        ));
        let underflow = Balance::new(
            params(0),
            &[100.0; 81],
            &[[1.0, 0.0, 0.0]; 81],
            &[[-1.0; 3]; 81],
            &[1.0; 81],
        )
        .unwrap();
        assert_eq!(
            underflow.baseline_raw_rgb().map(f32::to_bits),
            [0x8000_0000; 3]
        );
    }

    #[test]
    fn zero_reference_hash_is_opaque_and_inputs_are_not_retained() {
        let balance;
        {
            let mut p = params(3);
            p.reference_hash = 0;
            let mut base = [0.0; 81];
            let mut light = [1.0; 81];
            balance =
                Balance::new(p, &base, &[[1.0, 0.0, 0.0]; 81], &[[1.0; 3]; 81], &light).unwrap();
            base.fill(100.0);
            light.fill(0.0);
        }
        assert_eq!(*balance.baseline_raw_rgb(), [81.0; 3]);
        assert_ne!(balance.hash(), 0);
    }
}
