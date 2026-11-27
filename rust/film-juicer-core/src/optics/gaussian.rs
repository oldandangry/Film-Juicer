//! Immutable coefficients for the shared scanner, grain and FIR DIR cache miss.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianKernelError {
    Radius,
    Sigma,
    Allocation,
}

impl std::fmt::Display for GaussianKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Radius => "Gaussian radius is outside 1..=2048 pixels",
            Self::Sigma => "Gaussian sigma must be finite and positive",
            Self::Allocation => "Gaussian weights allocation failed",
        })
    }
}

impl std::error::Error for GaussianKernelError {}

#[derive(Debug)]
pub struct GaussianKernel {
    radius: u32,
    sigma_px: f32,
    weights: Vec<f32>,
}

impl GaussianKernel {
    pub fn new(radius: u32, sigma_px: f32) -> Result<Self, GaussianKernelError> {
        if !(1..=2048).contains(&radius) {
            return Err(GaussianKernelError::Radius);
        }
        if !sigma_px.is_finite() || sigma_px <= 0.0 {
            return Err(GaussianKernelError::Sigma);
        }
        let count = (radius as usize)
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .filter(|n| n.checked_mul(size_of::<f32>()).is_some())
            .ok_or(GaussianKernelError::Allocation)?;
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(count)
            .map_err(|_| GaussianKernelError::Allocation)?;
        let sigma = f64::from(sigma_px);
        let s2 = sigma * sigma * 2.0;
        let mut sum = 0.0;
        for i in -(radius as i32)..=radius as i32 {
            let x = f64::from(i);
            let w = (-(x * x) / s2).exp();
            weights.push(w as f32);
            sum += w;
        }
        let reciprocal = if sum != 0.0 { 1.0 / sum } else { 0.0 };
        // Normalize the first-rounded samples using the unrounded f64 sum.
        for w in &mut weights {
            *w = (f64::from(*w) * reciprocal) as f32;
        }
        Ok(Self {
            radius,
            sigma_px,
            weights,
        })
    }

    pub fn radius(&self) -> u32 {
        self.radius
    }

    pub fn sigma_px(&self) -> f32 {
        self.sigma_px
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_domain_and_order() {
        let radius_cases = [("zero", 0), ("above-limit", 2049), ("max", u32::MAX)];
        let cases_ids: Vec<_> = IntoIterator::into_iter(radius_cases.iter().map(|(id, _)| *id))
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "gaussian-invalid-radius"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "gaussian-invalid-radius"
        );
        for (id, radius) in radius_cases {
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            assert_eq!(
                GaussianKernel::new(radius, f32::NAN).unwrap_err(),
                GaussianKernelError::Radius
            );
            assert!(
                cases_remaining.remove(&case),
                "unexpected or repeated comparison: {case}"
            );
        }
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
        let sigma_cases = [
            ("positive-zero", 0.0),
            ("negative-zero", -0.0),
            ("negative", -1.0),
            ("nan", f32::NAN),
            ("positive-infinity", f32::INFINITY),
            ("negative-infinity", f32::NEG_INFINITY),
        ];
        let cases_ids: Vec<_> = IntoIterator::into_iter(sigma_cases.iter().map(|(id, _)| *id))
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "gaussian-invalid-sigma"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "gaussian-invalid-sigma"
        );
        for (id, sigma) in sigma_cases {
            let case = AsRef::<str>::as_ref(&(id)).to_owned();
            assert_eq!(
                GaussianKernel::new(1, sigma).unwrap_err(),
                GaussianKernelError::Sigma
            );
            assert!(
                cases_remaining.remove(&case),
                "unexpected or repeated comparison: {case}"
            );
        }
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
        let kernel = GaussianKernel::new(2048, f32::from_bits(1)).unwrap();
        assert_eq!(kernel.weights().len(), 4097);
        assert_eq!(kernel.weights()[2048], 1.0);
        assert_eq!(kernel.weights().iter().filter(|&&w| w == 0.0).count(), 4096);
    }

    // Maintenance comparison before independent candidate adoption. No golden
    // generation occurs here; the input comes from the retained native producer.
    #[test]
    fn complete_native_capture() {
        let captured: serde_json::Value =
            if let Ok(path) = std::env::var("FJ_GAUSSIAN_NATIVE_CAPTURE") {
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
            } else {
                let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/optics/gaussian");
                let read = |name: &str| -> serde_json::Value {
                    serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap()
                };
                let manifest = read("manifest.json");
                let preset = match (cfg!(windows), cfg!(debug_assertions)) {
                    (false, true) => "linux-debug",
                    (false, false) => "linux-release",
                    (true, true) => "windows-clang-debug",
                    (true, false) => "windows-clang-release",
                };
                let logical = manifest["presets"][preset]["native"].as_str().unwrap();
                assert_eq!(
                    read(manifest["contexts"][logical].as_str().unwrap())["preset"],
                    preset
                );
                read(manifest["payloads"][logical].as_str().unwrap())
            };
        let mut mismatches = Vec::new();
        let mut primitive_differences = 0;
        let mut samples = 0;
        let rows = captured["cases"].as_array().unwrap();
        let mut ids = rows
            .iter()
            .map(|row| row["name"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        ids.extend(rows.iter().flat_map(|row| {
            row["primitives"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(index, primitive)| {
                    format!(
                        "primitive/{}/query/{index}/i/{}",
                        row["name"].as_str().unwrap(),
                        primitive["i"].as_i64().unwrap()
                    )
                })
        }));
        let cases_ids: Vec<_> = IntoIterator::into_iter(ids)
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(
            !cases_ids.is_empty(),
            "empty case set: {}",
            "gaussian-native-cases-and-primitives"
        );
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            cases_remaining.len(),
            cases_count,
            "duplicate case IDs: {}",
            "gaussian-native-cases-and-primitives"
        );
        for case in rows {
            let name = case["name"].as_str().unwrap();
            let row_case = AsRef::<str>::as_ref(&(name)).to_owned();
            let radius = case["radius"].as_i64().unwrap();
            let sigma_bits = case["sigma_bits"].as_u64().unwrap() as u32;
            let sigma = f32::from_bits(sigma_bits);
            if radius <= 0 || !sigma.is_finite() || sigma <= 0.0 {
                assert_eq!(case["count"], 0);
                assert!(case["weights"].as_array().unwrap().is_empty());
                assert!(
                    cases_remaining.remove(&row_case),
                    "unexpected or repeated comparison: {row_case}"
                );
                continue;
            }
            let kernel = GaussianKernel::new(radius as u32, sigma).unwrap();
            assert_eq!(kernel.radius(), radius as u32);
            assert_eq!(kernel.sigma_px().to_bits(), sigma_bits);
            let expected = case["weights"].as_array().unwrap();
            assert_eq!(kernel.weights().len(), expected.len());
            assert_eq!(case["count"].as_u64().unwrap() as usize, expected.len());
            let mismatch_count = mismatches.len();
            for (index, (actual, word)) in kernel.weights().iter().zip(expected).enumerate() {
                samples += 1;
                if actual.to_bits() != word.as_u64().unwrap() as u32 {
                    mismatches.push(serde_json::json!({"case":case["name"], "radius":radius, "sigma_bits":sigma_bits,
                        "index":index,"native":word,"rust":actual.to_bits()}));
                }
            }
            if mismatches.len() == mismatch_count {
                assert!(
                    cases_remaining.remove(&row_case),
                    "unexpected or repeated comparison: {row_case}"
                );
            } else {
                drop(row_case);
            }
            for (index, primitive) in case["primitives"].as_array().unwrap().iter().enumerate() {
                let primitive_case = AsRef::<str>::as_ref(
                    &(format!(
                        "primitive/{name}/query/{index}/i/{}",
                        primitive["i"].as_i64().unwrap()
                    )),
                )
                .to_owned();
                let x = f64::from(primitive["i"].as_i64().unwrap() as i32);
                let s = f64::from(sigma);
                let argument = -(x * x) / (s * s * 2.0);
                assert_eq!(argument.to_bits(), primitive["argument"].as_u64().unwrap());
                if argument.exp().to_bits() != primitive["exp"].as_u64().unwrap() {
                    primitive_differences += 1;
                }
                assert!(
                    cases_remaining.remove(&primitive_case),
                    "unexpected or repeated comparison: {primitive_case}"
                );
            }
        }
        println!(
            "Gaussian comparison: {samples} complete sample words; {primitive_differences} f64 exp differences; {} f32 mismatches",
            mismatches.len()
        );
        for mismatch in &mismatches {
            println!("{mismatch}");
        }
        assert!(
            mismatches.is_empty(),
            "exact Gaussian coefficient gate failed"
        );
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
    }
}
