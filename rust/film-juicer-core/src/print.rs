//! Complete filtered print illumination on the fixed computational spectral axis.

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
