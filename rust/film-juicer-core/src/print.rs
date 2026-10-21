//! Complete print illumination and raw preflash exposure on the fixed spectral axis.

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
