//! Complete owned profiles. Source fields move here without reparsing or copying.
//! Completion pairs the selected kind with decoded metadata, establishes the
//! lookup axis and checks table capacity and computed density results. Incidental
//! parser/path allocations can abort.

use std::collections::TryReserveError;
use std::fmt;

use super::{
    Antihalation, ChannelModel, DensityCurveModel, DensitySampleError, Polarity, ProfileInfo,
    ProfileSamples, ProfileSource, ProfileUse, Role, Stage, Support,
};
use crate::hash;

#[derive(Debug)]
pub struct ProfileTables {
    wavelengths: [f32; 81],
    log_sensitivity: [[f32; 3]; 81],
    linear_sensitivity: [[f32; 3]; 81],
    channel_density: [[f32; 3]; 81],
    base_density: [f32; 81],
    source_log_exposure: Vec<f64>,
    log_exposure: Vec<f32>,
    density_curves: Vec<[f32; 3]>,
    density_curves_layers: [[Vec<f32>; 3]; 3],
    hanatos2025_adaptation_window_params: Option<[f32; 4]>,
    hanatos2025_adaptation_surface_params: Option<[[f32; 15]; 3]>,
}

impl ProfileTables {
    /// Authored labels; computation interprets spectral samples positionally on
    /// the fixed 380–780 nm axis, irrespective of these metadata values.
    pub fn wavelengths(&self) -> &[f32; 81] {
        &self.wavelengths
    }
    pub fn log_sensitivity(&self) -> &[[f32; 3]; 81] {
        &self.log_sensitivity
    }
    pub fn linear_sensitivity(&self) -> &[[f32; 3]; 81] {
        &self.linear_sensitivity
    }
    pub fn channel_density(&self) -> &[[f32; 3]; 81] {
        &self.channel_density
    }
    pub fn base_density(&self) -> &[f32; 81] {
        &self.base_density
    }
    /// Original f64 values and order, including distinctions hidden by narrowing.
    pub fn source_log_exposure(&self) -> &[f64] {
        &self.source_log_exposure
    }
    /// Nonempty, NaN-free, nondecreasing axis. Equality and infinities are allowed.
    pub fn log_exposure(&self) -> &[f32] {
        &self.log_exposure
    }
    /// Exposure-major CMY totals; row count matches the interpolation axis.
    pub fn density_curves(&self) -> &[[f32; 3]] {
        &self.density_curves
    }
    /// [layer][channel][exposure], unlike the model's [channel][layer] matrices.
    pub fn density_curves_layers(&self) -> &[[Vec<f32>; 3]; 3] {
        &self.density_curves_layers
    }
    pub fn hanatos2025_adaptation_window_params(&self) -> Option<&[f32; 4]> {
        self.hanatos2025_adaptation_window_params.as_ref()
    }
    pub fn hanatos2025_adaptation_surface_params(&self) -> Option<&[[f32; 15]; 3]> {
        self.hanatos2025_adaptation_surface_params.as_ref()
    }
}

#[derive(Debug)]
pub struct FilmDigest {
    pub gamma_samelayer_rgb: [f32; 3],
    pub gamma_interlayer_r_to_gb: [f32; 2],
    pub gamma_interlayer_g_to_rb: [f32; 2],
    pub gamma_interlayer_b_to_rg: [f32; 2],
    pub halation_first_sigma_um: [f32; 3],
    pub halation_primary_amount: [f32; 3],
    pub hanatos_spectral_gaussian_blur_default: f32,
}

// Shared complete representation for the two immutable owners, never a retained
// ProfileSource or a duplicate set of source arrays.
#[derive(Debug)]
struct CompletedProfile {
    info: ProfileInfo,
    tables: ProfileTables,
    density_model: DensityCurveModel,
    asset_token: u64,
}

#[derive(Debug)]
pub struct FilmProfile {
    profile: CompletedProfile,
    digest: FilmDigest,
}

impl FilmProfile {
    pub fn new(source: ProfileSource) -> Result<Self, ProfileCompletionError> {
        let profile = complete(
            source,
            Role::Film,
            Vec::try_reserve_exact,
            Vec::try_reserve_exact,
        )?;
        let digest = film_digest(&profile.info);
        Ok(Self { profile, digest })
    }
    pub fn info(&self) -> &ProfileInfo {
        &self.profile.info
    }
    pub fn tables(&self) -> &ProfileTables {
        &self.profile.tables
    }
    pub fn density_model(&self) -> &DensityCurveModel {
        &self.profile.density_model
    }
    pub fn asset_token(&self) -> u64 {
        self.profile.asset_token
    }
    pub fn digest(&self) -> &FilmDigest {
        &self.digest
    }
}

#[derive(Debug)]
pub struct PrintProfile {
    profile: CompletedProfile,
}

impl PrintProfile {
    pub fn new(source: ProfileSource) -> Result<Self, ProfileCompletionError> {
        complete(
            source,
            Role::Print,
            Vec::try_reserve_exact,
            Vec::try_reserve_exact,
        )
        .map(|profile| Self { profile })
    }
    pub fn info(&self) -> &ProfileInfo {
        &self.profile.info
    }
    pub fn tables(&self) -> &ProfileTables {
        &self.profile.tables
    }
    pub fn density_model(&self) -> &DensityCurveModel {
        &self.profile.density_model
    }
    pub fn asset_token(&self) -> u64 {
        self.profile.asset_token
    }

    /// Sample gamma-adjusted print totals on the original authored f64 axis.
    /// Gamma must be finite and positive; the recipe owns the user-control range.
    /// The immutable profile and its baseline samples remain unchanged.
    pub fn sample_density_curves(
        &self,
        gamma: f64,
    ) -> Result<PrintDensityCurves, PrintDensityError> {
        sample_print_density(self, gamma, Vec::try_reserve_exact)
    }
}

#[derive(Debug)]
pub struct PrintDensityCurves {
    totals: Vec<[f32; 3]>,
    hash: u64,
}

impl PrintDensityCurves {
    /// Exposure-major CMY totals, matching the source profile's interpolation axis.
    pub fn totals(&self) -> &[[f32; 3]] {
        &self.totals
    }
    /// Raw count/axis/total identity used by print development, without zero canonicalization.
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrintDensityError {
    InvalidGamma,
    Size,
    Capacity,
    Density {
        index: usize,
        source: DensitySampleError,
    },
    ZeroHash,
}

impl fmt::Display for PrintDensityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "print density curves: {self:?}")
    }
}
impl std::error::Error for PrintDensityError {}

fn print_gamma_model(
    baseline: &DensityCurveModel,
    gamma: f64,
) -> Result<DensityCurveModel, PrintDensityError> {
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(PrintDensityError::InvalidGamma);
    }
    let mut adjusted = DensityCurveModel::new(
        *baseline.centers(),
        *baseline.amplitudes(),
        *baseline.sigmas(),
    );
    if gamma != 1.0 {
        for channel in 0..3 {
            for layer in 0..3 {
                adjusted.centers[channel][layer] /= gamma;
                let sigma = adjusted.sigmas[channel][layer] / gamma;
                // std::max retains its first operand on unordered comparison.
                // f64::max would instead replace NaN with the floor.
                adjusted.sigmas[channel][layer] = if sigma < 0.05 { 0.05 } else { sigma };
            }
        }
    }
    Ok(adjusted)
}

fn check_print_density_size(count: usize) -> Result<(), PrintDensityError> {
    count
        .checked_mul(size_of::<[f32; 3]>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(PrintDensityError::Size)?;
    Ok(())
}

fn sample_print_density(
    profile: &PrintProfile,
    gamma: f64,
    reserve: impl FnOnce(&mut Vec<[f32; 3]>, usize) -> Result<(), TryReserveError>,
) -> Result<PrintDensityCurves, PrintDensityError> {
    let adjusted = print_gamma_model(profile.density_model(), gamma)?;
    let tables = profile.tables();
    let count = tables.source_log_exposure.len();
    check_print_density_size(count)?;
    let mut totals = Vec::new();
    reserve(&mut totals, count).map_err(|_| PrintDensityError::Capacity)?;
    for (index, &exposure) in tables.source_log_exposure.iter().enumerate() {
        let sample = adjusted
            .sample(profile.info().polarity, exposure)
            .map_err(|source| PrintDensityError::Density { index, source })?;
        totals.push(sample.total);
    }
    let mut hash = hash::FNV_OFFSET;
    hash::update_bytes(&mut hash, &(count as u64).to_le_bytes());
    for exposure in &tables.log_exposure {
        hash::update_bytes(&mut hash, &exposure.to_le_bytes());
    }
    for total in totals.as_flattened() {
        hash::update_bytes(&mut hash, &total.to_le_bytes());
    }
    if hash == 0 {
        return Err(PrintDensityError::ZeroHash);
    }
    Ok(PrintDensityCurves { totals, hash })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileCompletionErrorKind {
    SelectedRole,
    EmptyAxis,
    NanAxis {
        index: usize,
    },
    DescendingAxis {
        index: usize,
    },
    Size,
    Capacity,
    Density {
        index: usize,
        source: DensitySampleError,
    },
}

#[derive(Debug)]
pub struct ProfileCompletionError {
    pub stock: String,
    pub role: Role,
    pub kind: ProfileCompletionErrorKind,
}

impl fmt::Display for ProfileCompletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "profile {} role={:?} completion: {:?}",
            self.stock, self.role, self.kind
        )
    }
}
impl std::error::Error for ProfileCompletionError {}

fn complete(
    source: ProfileSource,
    role: Role,
    reserve_samples: impl FnMut(&mut Vec<f32>, usize) -> Result<(), TryReserveError>,
    reserve_totals: impl FnOnce(&mut Vec<[f32; 3]>, usize) -> Result<(), TryReserveError>,
) -> Result<CompletedProfile, ProfileCompletionError> {
    let ProfileSource {
        info,
        samples,
        density_model,
    } = source;
    // Both source loaders return ProfileSource, so pairing with a completed kind
    // is the one new role boundary. Decoded metadata needs no further admission.
    let role_matches = match role {
        Role::Film => info.support == Support::Film && info.stage == Stage::Filming,
        Role::Print => info.stage == Stage::Printing,
    };
    let result = if role_matches {
        sample_tables(
            samples,
            &density_model,
            info.polarity,
            reserve_samples,
            reserve_totals,
        )
    } else {
        Err(ProfileCompletionErrorKind::SelectedRole)
    };
    let tables = match result {
        Ok(tables) => tables,
        Err(kind) => {
            return Err(ProfileCompletionError {
                stock: info.stock,
                role,
                kind,
            });
        }
    };
    let asset_token = profile_token(&info, &tables, &density_model);
    Ok(CompletedProfile {
        info,
        tables,
        density_model,
        asset_token,
    })
}

fn check_table_size(count: usize) -> Result<(), ProfileCompletionErrorKind> {
    if count == 0 {
        return Err(ProfileCompletionErrorKind::EmptyAxis);
    }
    // Axis + three totals + nine layer/channel sequences; no unchecked product
    // or infallible second full-size copy on this promised allocation path.
    count
        .checked_mul(13)
        .and_then(|scalars| scalars.checked_mul(size_of::<f32>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(ProfileCompletionErrorKind::Size)?;
    Ok(())
}

fn sample_tables(
    samples: ProfileSamples,
    model: &DensityCurveModel,
    polarity: Polarity,
    mut reserve_samples: impl FnMut(&mut Vec<f32>, usize) -> Result<(), TryReserveError>,
    reserve_totals: impl FnOnce(&mut Vec<[f32; 3]>, usize) -> Result<(), TryReserveError>,
) -> Result<ProfileTables, ProfileCompletionErrorKind> {
    let count = samples.log_exposure.len();
    check_table_size(count)?;
    let mut log_exposure = Vec::new();
    reserve_samples(&mut log_exposure, count).map_err(|_| ProfileCompletionErrorKind::Capacity)?;
    for (index, &authored) in samples.log_exposure.iter().enumerate() {
        let narrowed = authored as f32;
        if narrowed.is_nan() {
            return Err(ProfileCompletionErrorKind::NanAxis { index });
        }
        if log_exposure
            .last()
            .is_some_and(|&previous| previous > narrowed)
        {
            return Err(ProfileCompletionErrorKind::DescendingAxis { index });
        }
        log_exposure.push(narrowed);
    }
    let mut density_curves = Vec::new();
    reserve_totals(&mut density_curves, count).map_err(|_| ProfileCompletionErrorKind::Capacity)?;
    let mut density_curves_layers = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for layer in &mut density_curves_layers {
        for channel in layer {
            reserve_samples(channel, count).map_err(|_| ProfileCompletionErrorKind::Capacity)?;
        }
    }
    for (index, &authored) in samples.log_exposure.iter().enumerate() {
        let sample = model
            .sample(polarity, authored)
            .map_err(|source| ProfileCompletionErrorKind::Density { index, source })?;
        density_curves.push(sample.total);
        for (curves, values) in density_curves_layers.iter_mut().zip(sample.layers) {
            for (curve, value) in curves.iter_mut().zip(values) {
                curve.push(value);
            }
        }
    }
    let linear_sensitivity = samples.log_sensitivity.map(|row| {
        row.map(|authored| {
            let linear = 10.0_f32.powf(authored);
            if linear.is_finite() { linear } else { 0.0 }
        })
    });
    Ok(ProfileTables {
        wavelengths: samples.wavelengths,
        log_sensitivity: samples.log_sensitivity,
        linear_sensitivity,
        channel_density: samples.channel_density,
        base_density: samples.base_density,
        source_log_exposure: samples.log_exposure,
        log_exposure,
        density_curves,
        density_curves_layers,
        hanatos2025_adaptation_window_params: samples.hanatos2025_adaptation_window_params,
        hanatos2025_adaptation_surface_params: samples.hanatos2025_adaptation_surface_params,
    })
}

fn film_digest(info: &ProfileInfo) -> FilmDigest {
    let (same, red, green, blue) = if info.polarity == Polarity::Positive {
        (
            [0.2291, 0.1029, 0.2651],
            [0.1449, 0.1560],
            [0.0496, 0.0874],
            [0.0571, 0.2462],
        )
    } else if info.usage == ProfileUse::Cine && info.reference_illuminant.starts_with('T') {
        (
            [0.4601, 0.4433, 0.4414],
            [0.4313, 0.3157],
            [0.3145, 0.3890],
            [0.2423, 0.2506],
        )
    } else if info.usage == ProfileUse::Cine {
        (
            [0.5456, 0.5434, 0.3766],
            [0.4263, 0.2781],
            [0.2051, 0.5015],
            [0.2950, 0.2449],
        )
    } else {
        (
            [0.5159, 0.5934, 0.2829],
            [0.4032, 0.2488],
            [0.2227, 0.4340],
            [0.1829, 0.1799],
        )
    };
    let (same, red, green, blue) = match info.stock.as_str() {
        "fujifilm_velvia_100" => (
            [0.2398, 0.0662, 0.1440],
            [0.0707, 0.2367],
            [0.0119, 0.0380],
            [0.0111, 0.1413],
        ),
        "kodak_kodachrome_64" => (
            [0.2942, 0.0841, 0.2492],
            [0.1699, 0.2188],
            [0.0625, 0.0692],
            [0.1153, 0.2359],
        ),
        "fujifilm_provia_100f" => (
            [0.2786, 0.1257, 0.3480],
            [0.1448, 0.2003],
            [0.0516, 0.1250],
            [0.0518, 0.3089],
        ),
        _ => (same, red, green, blue),
    };
    FilmDigest {
        gamma_samelayer_rgb: same,
        gamma_interlayer_r_to_gb: red,
        gamma_interlayer_g_to_rb: green,
        gamma_interlayer_b_to_rg: blue,
        halation_first_sigma_um: if info.usage == ProfileUse::Cine {
            [50.0; 3]
        } else {
            [65.0; 3]
        },
        halation_primary_amount: match info.antihalation {
            Antihalation::Strong => [0.015, 0.005, 0.0],
            Antihalation::Weak => [0.08, 0.02, 0.0],
            Antihalation::No => [0.30, 0.10, 0.015],
        },
        hanatos_spectral_gaussian_blur_default: 0.0,
    }
}

fn tagged_samples(token: &mut u64, tag: &str, samples: &[f32]) {
    hash::update_bytes(token, tag.as_bytes());
    let pair = hash::f32s_with_nan_mask(samples);
    hash::update_bytes(token, &pair.values.to_le_bytes());
    hash::update_bytes(token, &pair.nan_mask.to_le_bytes());
}

const DENSITY_CURVE_EVALUATOR_VERSION: u64 = 1;

fn profile_token(info: &ProfileInfo, tables: &ProfileTables, model: &DensityCurveModel) -> u64 {
    let mut token = hash::FNV_OFFSET;
    hash::update_bytes(&mut token, info.stock.as_bytes());
    // These are the external native metadata mappings, not Rust discriminants.
    for tag in [
        match info.support {
            Support::Film => 0_u64,
            Support::Paper => 1,
        },
        match info.stage {
            Stage::Filming => 0,
            Stage::Printing => 1,
        },
        match info.polarity {
            Polarity::Negative => 0,
            Polarity::Positive => 1,
        },
        match info.usage {
            ProfileUse::Still => 0,
            ProfileUse::Cine => 1,
        },
        match info.antihalation {
            Antihalation::Strong => 0,
            Antihalation::Weak => 1,
            Antihalation::No => 2,
        },
        match info.channel_model {
            ChannelModel::Color => 0,
            ChannelModel::Bw => 1,
        },
    ] {
        hash::update_bytes(&mut token, &tag.to_le_bytes());
    }
    hash::update_bytes(&mut token, info.reference_illuminant.as_bytes());
    hash::update_bytes(&mut token, info.viewing_illuminant.as_bytes());
    tagged_samples(&mut token, "data.wavelengths", &tables.wavelengths);
    tagged_samples(
        &mut token,
        "data.log_sensitivity",
        tables.log_sensitivity.as_flattened(),
    );
    tagged_samples(
        &mut token,
        "data.channel_density",
        tables.channel_density.as_flattened(),
    );
    tagged_samples(&mut token, "data.base_density", &tables.base_density);
    tagged_samples(&mut token, "data.log_exposure", &tables.log_exposure);
    tagged_samples(
        &mut token,
        "data.density_curves",
        tables.density_curves.as_flattened(),
    );
    hash::update_bytes(&mut token, b"data.hanatos2025_adaptation_window_params");
    hash::update_bytes(
        &mut token,
        &u64::from(tables.hanatos2025_adaptation_window_params.is_some()).to_le_bytes(),
    );
    if let Some(samples) = &tables.hanatos2025_adaptation_window_params {
        // Presence tags already precede these pairs; add no second tag.
        let pair = hash::f32s_with_nan_mask(samples);
        hash::update_bytes(&mut token, &pair.values.to_le_bytes());
        hash::update_bytes(&mut token, &pair.nan_mask.to_le_bytes());
    }
    hash::update_bytes(&mut token, b"data.hanatos2025_adaptation_surface_params");
    hash::update_bytes(
        &mut token,
        &u64::from(tables.hanatos2025_adaptation_surface_params.is_some()).to_le_bytes(),
    );
    if let Some(samples) = &tables.hanatos2025_adaptation_surface_params {
        let pair = hash::f32s_with_nan_mask(samples.as_flattened());
        hash::update_bytes(&mut token, &pair.values.to_le_bytes());
        hash::update_bytes(&mut token, &pair.nan_mask.to_le_bytes());
    }
    hash::update_bytes(&mut token, b"density-curve-evaluator-version");
    hash::update_bytes(&mut token, &DENSITY_CURVE_EVALUATOR_VERSION.to_le_bytes());
    hash::update_bytes(&mut token, b"data.log_exposure.source-double");
    hash::update_bytes(
        &mut token,
        &(tables.source_log_exposure.len() as u64).to_le_bytes(),
    );
    for sample in &tables.source_log_exposure {
        hash::update_bytes(&mut token, &sample.to_le_bytes());
    }
    for (tag, coefficients) in [
        ("data.density_curves_model.centers", model.centers()),
        ("data.density_curves_model.amplitudes", model.amplitudes()),
        ("data.density_curves_model.sigmas", model.sigmas()),
    ] {
        hash::update_bytes(&mut token, tag.as_bytes());
        for sample in coefficients.iter().flatten() {
            hash::update_bytes(&mut token, &sample.to_le_bytes());
        }
    }
    finish_asset_token(token)
}

fn finish_asset_token(token: u64) -> u64 {
    if token == 0 { 1 } else { token }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn source(role: Role) -> ProfileSource {
        let mut document = json!({
            "info": {"stock": "capacity-case"},
            "data": {
                "wavelengths": ([380.0; 81].as_slice()),
                "log_sensitivity": ([[0.0; 3]; 81].as_slice()),
                "channel_density": ([[0.0; 3]; 81].as_slice()),
                "base_density": ([0.0; 81].as_slice()),
                "log_exposure": [0.0, 0.0, 0.0],
                "density_curves_model": {"centers": ([[0.0; 3]; 3]), "amplitudes": ([[1.0; 3]; 3]), "sigmas": ([[1.0; 3]; 3])}
            }
        });
        if role == Role::Print {
            document["info"]["stage"] = json!("printing");
        }
        super::super::read_source(&document, role).unwrap()
    }

    #[test]
    fn checked_lengths_precede_reservations() {
        assert_eq!(
            check_table_size(0),
            Err(ProfileCompletionErrorKind::EmptyAxis)
        );
        assert!(check_table_size(1).is_ok());
        assert!(check_table_size(i32::MAX as usize).is_ok());
        for count in [usize::MAX, usize::MAX / 13, isize::MAX as usize / 52 + 1] {
            assert_eq!(
                check_table_size(count),
                Err(ProfileCompletionErrorKind::Size)
            );
        }
        let mut source = source(Role::Film);
        source.samples.log_exposure.clear();
        let error = complete(
            source,
            Role::Film,
            |_, _| panic!("empty axis must not reserve"),
            |_, _| panic!("empty axis must not reserve"),
        )
        .unwrap_err();
        assert_eq!(error.kind, ProfileCompletionErrorKind::EmptyAxis);
    }

    #[test]
    fn reservation_failures_return_no_completed_owner() {
        for role in [Role::Film, Role::Print] {
            // Axis first, then all nine layer/channel buffers; fail each actual
            // reservation with a bounded capacity error, including the last one.
            for fail_at in 0..10 {
                let mut call = 0;
                let error = complete(
                    source(role),
                    role,
                    |buffer, count| {
                        assert!(buffer.is_empty());
                        assert_eq!(count, 3);
                        let requested = if call == fail_at { usize::MAX } else { count };
                        call += 1;
                        buffer.try_reserve_exact(requested)
                    },
                    Vec::try_reserve_exact,
                )
                .unwrap_err();
                assert_eq!(error.kind, ProfileCompletionErrorKind::Capacity);
                assert_eq!(error.stock, "capacity-case");
                assert_eq!(error.role, role);
                assert_eq!(call, fail_at + 1);
            }
            let error = complete(
                source(role),
                role,
                Vec::try_reserve_exact,
                |buffer, count| {
                    assert_eq!(count, 3);
                    buffer.try_reserve_exact(usize::MAX)
                },
            )
            .unwrap_err();
            assert_eq!(error.kind, ProfileCompletionErrorKind::Capacity);
        }
    }

    #[test]
    fn nan_axis_fails_before_sampling_or_total_reservation() {
        let mut source = source(Role::Film);
        source.samples.log_exposure[1] = f64::NAN;
        let error = complete(source, Role::Film, Vec::try_reserve_exact, |_, _| {
            panic!("NaN axis must not reserve totals")
        })
        .unwrap_err();
        assert_eq!(error.kind, ProfileCompletionErrorKind::NanAxis { index: 1 });
    }

    #[test]
    fn asset_boundary_maps_only_zero_to_one() {
        assert_eq!(finish_asset_token(0), 1);
        assert_eq!(finish_asset_token(1), 1);
        assert_eq!(finish_asset_token(u64::MAX), u64::MAX);
    }

    #[test]
    fn adjusted_gamma_coefficients_match_independent_native_bits() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/profile/fixtures");
        let gamma: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("print-gamma.json")).unwrap()).unwrap();
        let density: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("density.json")).unwrap()).unwrap();
        let matrix = |node: &serde_json::Value| {
            serde_json::from_value::<[[u64; 3]; 3]>(node.clone())
                .unwrap()
                .map(|row| row.map(f64::from_bits))
        };
        for captured in gamma["captured"].as_array().unwrap() {
            let case = density["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["id"] == captured["id"])
                .unwrap();
            let model = &captured["source_model"];
            let baseline = DensityCurveModel::new(
                matrix(&model["centers"]),
                matrix(&model["amplitudes"]),
                matrix(&model["sigmas"]),
            );
            let adjusted = print_gamma_model(
                &baseline,
                f64::from_bits(case["gamma_bits"].as_u64().unwrap()),
            )
            .unwrap();
            for (name, actual) in [
                ("centers", adjusted.centers()),
                ("amplitudes", adjusted.amplitudes()),
                ("sigmas", adjusted.sigmas()),
            ] {
                let expected: [[u64; 3]; 3] =
                    serde_json::from_value(case["model"][name].clone()).unwrap();
                assert_eq!(
                    actual.map(|r| r.map(f64::to_bits)),
                    expected,
                    "{} {name}",
                    case["id"]
                );
            }
        }
    }

    #[test]
    fn gamma_one_copies_unusual_bits_and_nonunit_floor_keeps_nan() {
        let nan = f64::from_bits(0x7ff8_0000_0000_0123);
        let model = DensityCurveModel::new(
            [[-0.0, 1e40, f64::NEG_INFINITY]; 3],
            [[-1.0, nan, 0.0]; 3],
            [[-0.0, -1.0, nan]; 3],
        );
        let same = print_gamma_model(&model, 1.0).unwrap();
        for (actual, expected) in [
            (same.centers(), model.centers()),
            (same.amplitudes(), model.amplitudes()),
            (same.sigmas(), model.sigmas()),
        ] {
            assert_eq!(
                actual.map(|r| r.map(f64::to_bits)),
                expected.map(|r| r.map(f64::to_bits))
            );
        }
        let adjusted = print_gamma_model(&model, 2.0).unwrap();
        assert_eq!(adjusted.centers()[0][0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(adjusted.centers()[0][1].to_bits(), (5e39_f64).to_bits());
        assert_eq!(adjusted.sigmas()[0][0].to_bits(), 0.05_f64.to_bits());
        assert_eq!(adjusted.sigmas()[0][1].to_bits(), 0.05_f64.to_bits());
        assert_eq!(adjusted.sigmas()[0][2].to_bits(), nan.to_bits());
        assert_eq!(
            adjusted.amplitudes().map(|r| r.map(f64::to_bits)),
            model.amplitudes().map(|r| r.map(f64::to_bits))
        );
        // A NaN sigma reaches computation instead of becoming a floor value.
        let model = DensityCurveModel::new([[0.0; 3]; 3], [[1.0; 3]; 3], [[nan; 3]; 3]);
        assert_eq!(
            print_gamma_model(&model, 2.0)
                .unwrap()
                .sample(Polarity::Negative, 0.0)
                .unwrap_err(),
            DensitySampleError::NonfiniteLayer {
                channel: 0,
                layer: 0
            }
        );
    }

    #[test]
    fn print_gamma_size_and_capacity_failures_return_no_owner() {
        assert!(check_print_density_size(1).is_ok());
        for count in [usize::MAX, isize::MAX as usize / size_of::<[f32; 3]>() + 1] {
            assert_eq!(
                check_print_density_size(count),
                Err(PrintDensityError::Size)
            );
        }
        let profile = PrintProfile::new(source(Role::Print)).unwrap();
        let before = profile.tables().density_curves.as_ptr();
        let error = sample_print_density(&profile, 1.1, |buffer, count| {
            assert!(buffer.is_empty());
            assert_eq!(count, 3);
            buffer.try_reserve_exact(usize::MAX)
        })
        .unwrap_err();
        assert_eq!(error, PrintDensityError::Capacity);
        assert_eq!(profile.tables().density_curves.as_ptr(), before);
        assert!(profile.sample_density_curves(1.1).is_ok());
        assert_eq!(
            sample_print_density(&profile, 0.0, |_, _| panic!(
                "invalid gamma must not reserve"
            ))
            .unwrap_err(),
            PrintDensityError::InvalidGamma
        );
    }
    #[test]
    fn bundled_payload_memory_reports_logical_and_capacity_bytes() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources");
        let catalog = super::super::load_catalog(&root).unwrap();
        for role in [Role::Film, Role::Print] {
            let entries = if role == Role::Film {
                catalog.films()
            } else {
                catalog.prints()
            };
            for entry in entries {
                let source = super::super::load_source(entry.source_path(), role).unwrap();
                let profile =
                    complete(source, role, Vec::try_reserve_exact, Vec::try_reserve_exact).unwrap();
                let tables = &profile.tables;
                let vectors = tables.source_log_exposure.len() * 8
                    + tables.log_exposure.len() * 4
                    + tables.density_curves.len() * 12
                    + tables
                        .density_curves_layers
                        .iter()
                        .flatten()
                        .map(|v| v.len() * 4)
                        .sum::<usize>();
                let capacity = tables.source_log_exposure.capacity() * 8
                    + tables.log_exposure.capacity() * 4
                    + tables.density_curves.capacity() * 12
                    + tables
                        .density_curves_layers
                        .iter()
                        .flatten()
                        .map(|v| v.capacity() * 4)
                        .sum::<usize>();
                let strings = [
                    &profile.info.stock,
                    &profile.info.name,
                    &profile.info.reference_illuminant,
                    &profile.info.viewing_illuminant,
                ];
                let inline = std::mem::size_of::<CompletedProfile>()
                    + if role == Role::Film {
                        std::mem::size_of::<FilmDigest>()
                    } else {
                        0
                    };
                let logical = inline + vectors + strings.iter().map(|s| s.len()).sum::<usize>();
                let retained =
                    inline + capacity + strings.iter().map(|s| s.capacity()).sum::<usize>();
                assert!(retained >= logical);
                println!(
                    "Rust payload {}: logical={logical} capacity={retained}, Arc counters=16, Arc handle=8 (allocator/RSS excluded)",
                    entry.key()
                );
                if role == Role::Print {
                    let print = PrintProfile { profile };
                    let curves = print.sample_density_curves(1.1).unwrap();
                    println!(
                        "Rust gamma {}: inline={} logical={} capacity={}",
                        entry.key(),
                        std::mem::size_of::<PrintDensityCurves>(),
                        curves.totals.len() * 12,
                        curves.totals.capacity() * 12
                    );
                }
            }
        }
    }
}
