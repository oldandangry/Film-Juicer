//! Complete host photographic preparation. Construction is private until every
//! route, payload, reference, balance and identity has succeeded.

use std::fmt;
use std::sync::Arc;

use crate::assets::{AssetError, Assets, Illuminants, InputHull, SpectraLut};
use crate::data_io::calibration;
use crate::film::{self, density, dir, grain, input, reference};
use crate::profile::{FilmProfile, IlluminantKind, Polarity, PrintProfile, PrintDensityCurves};
use crate::route::ScanRoute;
use crate::{
    color, effects, exposure, gamut, hash, illuminant, optics, print, reconstruction, scanner,
    spectral,
};

const OUTPUT_KNEE: [f32; 3] = [0.95, 1.0, 1.6];

#[cfg(feature = "test-support")]
pub mod testing {
    use std::cell::Cell;

    #[derive(Clone, Copy)]
    #[repr(u32)]
    pub enum Stage {
        SelectedFilm = 1,
        SelectedPrint,
        Illuminants,
        Development,
        Observer,
        ReconstructionSource,
        InputHull,
        Calibration,
        RoutePrechecks,
        Foundation,
        FilmPayload,
        PrintMainIlluminant,
        FilmReference,
        PrintBalance,
        ScannerPayload,
        OutputBoundary,
        PrintPreflash,
        ScannerCorrection,
        FinalIdentity,
    }
    #[derive(Clone, Copy)]
    struct Trace {
        stages: [u32; 24],
        count: usize,
    }
    thread_local! {
        static TRACE: Cell<Trace> = const { Cell::new(Trace { stages: [0; 24], count: 0 }) };
        static ZERO_IDENTITY: Cell<Option<ZeroIdentity>> = const { Cell::new(None) };
    }
    #[derive(Clone, Copy, Eq, PartialEq)]
    pub enum ZeroIdentity {
        Recipe,
        Scanner,
    }
    pub fn arm_zero_identity(target: Option<ZeroIdentity>) {
        ZERO_IDENTITY.set(target);
    }
    pub(super) fn identity(target: ZeroIdentity, value: u64) -> u64 {
        if ZERO_IDENTITY.get() == Some(target) {
            ZERO_IDENTITY.set(None);
            0
        } else {
            value
        }
    }
    pub(super) fn reset() {
        TRACE.set(Trace {
            stages: [0; 24],
            count: 0,
        });
    }
    pub(super) fn visit(stage: Stage) {
        let mut trace = TRACE.get();
        assert!(
            trace.count < trace.stages.len(),
            "bounded preparation trace overflow"
        );
        trace.stages[trace.count] = stage as u32;
        trace.count += 1;
        TRACE.set(trace);
    }
    pub fn trace() -> ([u32; 24], u32) {
        let trace = TRACE.get();
        (trace.stages, trace.count as u32)
    }
}
macro_rules! trace_stage {
    ($stage:ident) => {
        #[cfg(feature = "test-support")]
        testing::visit(testing::Stage::$stage);
    };
}

/// Temporary captured snapshot. Integer choices stay unadmitted until their
/// owning construction stage, preserving competing-failure precedence.
pub struct Params {
    pub print_exposure: f64,
    pub print_preflash_exposure: f64,
    pub print_gamma_factor: f64,
    pub preflash_m_filter_cc: f64,
    pub preflash_y_filter_cc: f64,
    pub print_shadow_compensation_factor: f64,
    pub print_shadow_compensation_density: f64,
    pub print_shadow_compensation_transition: f64,
    pub glare_percent: f64,
    pub glare_roughness: f64,
    pub glare_blur_sigma_px: f64,
    pub camera_exposure_compensation_ev: f64,
    pub camera_film_format_long_edge_mm: f32,
    pub film_gamma_factor: f32,
    pub scanner_lens_blur_sigma_px: f64,
    pub scanner_black_level: f64,
    pub scanner_white_level: f64,
    pub gate_weave_amount: f64,
    pub scanner_unsharp_mask: [f64; 2],
    pub print_ui_ymc_cc: [f64; 3],
    pub camera_filter_uv: [f64; 3],
    pub camera_filter_ir: [f64; 3],
    pub camera_diffusion: optics::DiffusionParams,
    pub enlarger_diffusion: optics::DiffusionParams,
    pub dir_couplers: dir::Params,
    pub film_profile_key: String,
    pub print_profile_key: String,
    pub spectral_upsampling_mode: i32,
    pub ref_ill: i32,
    pub enl_ill: i32,
    pub normalize_print_exposure: i32,
    pub print_exposure_compensation: i32,
    pub input_color_space: i32,
    pub input_cctf_decoding: i32,
    pub input_compression_enabled: i32,
    pub hanatos2025_adaptation_window: i32,
    pub hanatos2025_adaptation_surface: i32,
    pub camera_auto_exposure_enabled: i32,
    pub camera_metering_method: i32,
    pub scatter_halation: HalationControls,
    pub scanner_black_correction: i32,
    pub scanner_white_correction: i32,
    pub scanner_lut_resolution: i32,
    pub output_color_space: i32,
    pub output_cctf_encoding: i32,
    pub output_gamut_compression_enabled: i32,
    pub grain: grain::Params,
    pub film_dust_amount: f32,
    pub film_scratch_amount: f32,
    pub gate_dust_amount: f32,
    pub gate_scratch_amount: f32,
    pub scan_route: ScanRoute,
    pub glare_active: bool,
    pub camera_filter_override: bool,
}

#[derive(Clone, Copy)]
pub struct HalationControls {
    pub active: bool,
    pub scatter_amount: f32,
    pub scatter_spatial_scale: f32,
    pub halation_amount: f32,
    pub halation_spatial_scale: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlError {
    DirectScanner,
    PrintGamma,
    PrintStage,
    PrintInput,
    ReferenceIlluminant,
    InputColorSpace,
    ReconstructionMethod,
    Meter,
    OutputColorSpace,
    ExposureTables,
    FilmReferenceIlluminant,
    PrintIlluminant,
    ScannerPayload(ScanRoute),
    Identity(ScanRoute),
}
impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DirectScanner => "ResourceDescriptorMismatch phase=8 field=scanner_output",
            Self::PrintGamma => {
                "InvalidAuthoredControl component=print_development field=print_gamma_factor"
            }
            Self::PrintStage => "UnsupportedMode phase=4A selected profile route mismatch",
            Self::PrintInput => "ResourceDescriptorMismatch phase=4A field=print_recipe_input",
            Self::ReferenceIlluminant => {
                "MissingRequiredResource phase=3B field=reference_illuminant"
            }
            Self::InputColorSpace => "UnsupportedMode phase=3B field=input_color_space",
            Self::ReconstructionMethod => {
                "UnsupportedMode phase=3B field=spectral_upsampling_method"
            }
            Self::Meter => "UnsupportedMode phase=3B field=auto_exposure_method",
            Self::OutputColorSpace => {
                "ResourceDescriptorMismatch component=output_gamut_recipe field=output_color_space"
            }
            Self::ExposureTables => {
                "MalformedRequiredResource component=focused_film_payload field=exposure_tables"
            }
            Self::FilmReferenceIlluminant => {
                "MissingRequiredResource phase=3B field=film_reference_illuminant"
            }
            Self::PrintIlluminant => "MissingRequiredResource phase=4A field=print_illuminant",
            Self::ScannerPayload(route) | Self::Identity(route) => {
                if print_route(*route) { "ResourceDescriptorMismatch phase=4C focused print publication payload build failed" }
                else { "ResourceDescriptorMismatch focused direct publication payload build failed" }
            },
        })
    }
}

#[derive(Debug)]
pub enum PrepareError {
    Asset(AssetError),
    Development(film::Error),
    Control(ControlError),
    Input(input::Error),
    Optics(optics::Error),
    Minimum(density::Error),
    Grain(grain::Error),
    Effects(effects::Error),
    Dir {
        error: dir::Error,
        route: ScanRoute,
        stock: String,
    },
    Bounds(density::Error),
    LutDescriptor(scanner::LutDescriptorError),
    ReferenceWhite(reconstruction::Error),
    Tc(reconstruction::TcError),
    Reference(reference::Error),
    PrintDevelopment(crate::profile::PrintDensityError),
    PrintComposition(print::CompositionError),
    PrintIlluminant(print::Error),
    PrintBalance(print::BalanceError),
    PrintPreflash(print::PreflashError),
    Calibration(calibration::ErrorKind),
    OutputTransform(gamut::TransformError),
    ScannerColor(scanner::ColorError),
    ScannerCorrection(scanner::CorrectionError),
    ScannerPostEffects(scanner::PostEffectsError),
}
impl From<AssetError> for PrepareError {
    fn from(error: AssetError) -> Self {
        Self::Asset(error)
    }
}
impl From<ControlError> for PrepareError {
    fn from(error: ControlError) -> Self {
        Self::Control(error)
    }
}
impl fmt::Display for PrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asset(error) => error.fmt(f),
            Self::Development(error) => error.fmt(f),
            Self::Control(error) => error.fmt(f),
            Self::Input(error) => error.fmt(f),
            Self::Optics(error) => optics_diagnostic(*error, f),
            Self::Minimum(_) => {
                f.write_str("MalformedRequiredProfileData phase=9A field=grain_contract")
            }
            Self::Grain(_) => {
                f.write_str("MalformedRequiredProfileData phase=9A field=visual_grain")
            }
            Self::Effects(_) => {
                f.write_str("ResourceDescriptorMismatch phase=effects_recipe field=physical_policy")
            }
            Self::Dir {
                error,
                route,
                stock,
            } => dir_diagnostic(error, *route, stock, f),
            Self::Bounds(_) => {
                f.write_str("MalformedRequiredProfileData phase=3A field=density_bounds")
            }
            Self::LutDescriptor(error) => error.fmt(f),
            Self::ReferenceWhite(error) => error.fmt(f),
            Self::Tc(error) => f.write_str(error.diagnostic()),
            Self::Reference(error) => reference_diagnostic(*error, f),
            Self::PrintDevelopment(_) => {
                f.write_str("MalformedRequiredProfileData phase=3B field=print_development_table")
            }
            Self::PrintComposition(error) => error.fmt(f),
            Self::PrintIlluminant(_) => {
                f.write_str("MalformedRequiredResource phase=4B field=filtered_print_illuminant")
            }
            Self::PrintBalance(print::BalanceError::Invalid) => {
                f.write_str("MalformedRequiredResource phase=4A field=print_balance")
            }
            Self::PrintBalance(_) => {
                f.write_str("MalformedRequiredResource phase=4A field=print_balance_raw")
            }
            Self::PrintPreflash(_) => {
                f.write_str("MalformedRequiredResource phase=4B field=preflash_raw")
            }
            Self::Calibration(kind) => {
                let field = match kind {
                    calibration::ErrorKind::Read(_) => "resource_read",
                    calibration::ErrorKind::Malformed(calibration::Field::Root) => "root",
                    calibration::ErrorKind::Malformed(calibration::Field::PrintProfile) => {
                        "print_profile"
                    }
                    calibration::ErrorKind::Malformed(calibration::Field::PrintIlluminant) => {
                        "print_illuminant"
                    }
                    calibration::ErrorKind::Malformed(calibration::Field::CmyCc) => "cmy_cc",
                    calibration::ErrorKind::MissingFile => "resource_read",
                };
                write!(f, "MalformedNeutralPrintCalibration phase=4A field={field}")
            }
            Self::OutputTransform(error) => error.fmt(f),
            Self::ScannerColor(error) => error.fmt(f),
            Self::ScannerCorrection(error) => error.fmt(f),
            Self::ScannerPostEffects(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for PrepareError {}

fn optics_diagnostic(error: optics::Error, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use optics::{Component, Field};
    if error.component() == Component::ScatterHalation {
        return write!(
            f,
            "ResourceDescriptorMismatch component=scatter_halation field=total_strength channel={}",
            error.channel().unwrap_or(3)
        );
    }
    let component = match error.component() {
        Component::CameraDiffusion => "camera_diffusion",
        Component::EnlargerDiffusion => "enlarger_diffusion",
        Component::CameraLensBlur => "camera_lens_blur",
        Component::ScatterHalation => unreachable!(),
    };
    let field = match error.field() {
        Field::Family => "family",
        Field::Strength => "strength",
        Field::SpatialScale => "spatial_scale",
        Field::HaloWarmth => "halo_warmth",
        Field::CoreIntensity => "core_intensity",
        Field::CoreSize => "core_size",
        Field::HaloIntensity => "halo_intensity",
        Field::HaloSize => "halo_size",
        Field::BloomIntensity => "bloom_intensity",
        Field::BloomSize => "bloom_size",
        Field::ScatterFraction => "scatter_fraction",
        Field::GroupWeights => "group_weights",
        Field::GroupCenters => "group_centers_um",
        Field::ResolvedHash => "resolved_hash",
        Field::SigmaUm => "sigma_um",
        Field::TotalStrength => "total_strength",
    };
    write!(
        f,
        "ResourceDescriptorMismatch phase=6A component={component} field={field}"
    )
}

fn dir_diagnostic(
    error: &dir::Error,
    route: ScanRoute,
    stock: &str,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    use dir::Error;
    let (field, requirement, channel, sample) = match *error {
        Error::Matrix => ("matrix", "finite", None, None),
        Error::LogExposure { sample } => {
            ("log_exposure", "strictly_increasing", None, Some(sample))
        }
        Error::DensityMaximum => ("density_max", "finite_positive", None, None),
        Error::ReceiverK { channel } => (
            "receiver_k",
            "finite_positive_when_coupled",
            Some(channel),
            None,
        ),
        Error::DonorK { channel } => ("donor_k", "finite_positive", Some(channel), None),
        Error::DonorLangmuir { channel, sample } => (
            "donor_langmuir",
            "finite_defined",
            Some(channel),
            Some(sample),
        ),
        Error::ReceiverLangmuir { channel, sample } => (
            "receiver_langmuir",
            "finite_defined",
            Some(channel),
            Some(sample),
        ),
        Error::CompensatedAxis { channel, sample } => (
            "compensated_axis",
            "finite_float_representable",
            Some(channel),
            Some(sample),
        ),
        Error::Size | Error::Allocation => return write!(f, "{error}"),
        Error::ZeroIdentity => {
            return f.write_str("MalformedRequiredProfileData phase=3D-3 field=dirCouplers");
        }
    };
    write!(
        f,
        "component=dir route={} profile={stock} field={field}",
        route_key(route)
    )?;
    if let Some(channel) = channel {
        write!(f, " channel={channel}")?;
    }
    if let Some(sample) = sample {
        write!(f, " sample={sample}")?;
    }
    write!(f, " requirement={requirement}")
}

fn reference_diagnostic(error: reference::Error, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let message = match error {
        reference::Error::Reconstruction { .. } => {
            "MalformedRequiredResource phase=3B field=synthetic_reference_raw"
        }
        reference::Error::Meter => {
            "MalformedRequiredResource phase=3B field=synthetic_reference_meter"
        }
        reference::Error::DensityLookup { .. } => {
            "ResourceDescriptorMismatch phase=3B field=synthetic_reference_density_lookup"
        }
        #[cfg(feature = "test-support")]
        reference::Error::Dependency {
            category: reference::Category::Allocation,
            ..
        } => "film reference dependency allocation failed",
        #[cfg(feature = "test-support")]
        reference::Error::Dependency { operation, .. } => match operation {
            reference::Operation::Source => {
                "Reference source preparation: exposure production boundary fault"
            }
            reference::Operation::InputMatrices => "Input matrix preparation failed",
            reference::Operation::LinearSrgb => "Linear sRGB to XYZ conversion failed",
            reference::Operation::Cat16 => "CAT16 scalar adaptation failed",
            reference::Operation::TcSample => "Film TC sampling: TC production boundary fault",
            reference::Operation::MallettRaw => {
                "Mallett reference preparation: exposure production boundary fault"
            }
        },
    };
    f.write_str(message)
}

pub struct RenderState {
    film_key: String,
    development: film::Development,
    input: input::Recipe,
    optics: optics::Recipe,
    minimum: density::Minimum,
    grain: grain::Recipe,
    effects: effects::Recipe,
    dir: dir::Recipe,
    film_bounds: density::Bounds,
    exposure_tables: spectral::Tables,
    s_inverse: [f32; 9],
    reconstruction: FilmReconstruction,
    input_hull: Option<Arc<InputHull>>,
    reference: reference::Reference,
    scanner: ScannerOutput,
    route: PreparedRoute,
    profile_route_hash: u64,
    hash: u64,
    scanner_hash: u64,
}

#[allow(
    clippy::large_enum_variant,
    reason = "The complete state has one fallibly reserved backing; inline route storage avoids another substantial allocation and preserves one completion owner."
)]
pub enum PreparedRoute {
    Direct(DirectState),
    Print(PrintState),
}
pub struct DirectState {
    correction: scanner::Correction,
}
pub struct PrintState {
    key: String,
    profile: Arc<PrintProfile>,
    gamma: f64,
    curves: PrintDensityCurves,
    bounds: print::Bounds,
    calibration: Calibration,
    neutral_cmy_cc: [f32; 3],
    filters: print::Filters,
    exposure: print::Exposure,
    illuminant_key: &'static str,
    illuminant_hash: u64,
    source_illuminant: [f32; 81],
    main_illuminant: print::Illuminant,
    preflash: Option<Preflash>,
    balance: print::Balance,
    resource_descriptors: print::ResourceDescriptors,
    correction: scanner::Correction,
    hash: u64,
}
pub struct Preflash {
    illuminant: print::Illuminant,
    raw: print::PreflashRaw,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Calibration {
    MissingFile,
    MissingEntry,
    Calibrated([f32; 3]),
}

pub struct ScannerOutput {
    params: ScannerParams,
    viewing_illuminant: String,
    tables: spectral::Tables,
    color: scanner::Color,
    lut: scanner::LutDescriptor,
    post_effects: scanner::PostEffects,
    gamut: OutputGamut,
    hash: u64,
}
#[derive(Clone, Copy)]
pub struct ScannerParams {
    pub resolution: u32,
    pub output_space: color::OutputSpace,
    pub encode_cctf: bool,
    pub black_correction: bool,
    pub white_correction: bool,
    pub black_level: f32,
    pub white_level: f32,
    pub glare_active: bool,
    pub glare_percent: f32,
    pub glare_roughness: f32,
    pub glare_blur_sigma_px: f32,
    pub lens_blur_sigma_px: f32,
    pub unsharp_sigma_px: f32,
    pub unsharp_amount: f32,
}
pub struct OutputGamut {
    enabled: bool,
    transform: gamut::OutputTransform,
    boundary: Option<Arc<crate::assets::OutputBoundary>>,
    hash: u64,
}

pub struct FilmReconstruction {
    storage: ReconstructionStorage,
}
enum ReconstructionStorage {
    Tc {
        source: Arc<SpectraLut>,
        lut: reconstruction::FilmTcLut,
        raw_midgray_rgb: [f32; 3],
        normalization: exposure::MidgrayNormalization,
    },
    Mallett {
        basis: Arc<[[f32; 3]; 81]>,
        midgray: exposure::MallettMidgray,
    },
}

impl RenderState {
    pub fn film_key(&self) -> &str {
        &self.film_key
    }
    pub fn film_profile(&self) -> &FilmProfile {
        self.development.profile()
    }
    pub fn development(&self) -> &film::Development {
        &self.development
    }
    pub fn input(&self) -> &input::Recipe {
        &self.input
    }
    pub fn optics(&self) -> &optics::Recipe {
        &self.optics
    }
    pub fn minimum(&self) -> &density::Minimum {
        &self.minimum
    }
    pub fn grain(&self) -> &grain::Recipe {
        &self.grain
    }
    pub fn effects(&self) -> &effects::Recipe {
        &self.effects
    }
    pub fn dir(&self) -> &dir::Recipe {
        &self.dir
    }
    pub fn film_bounds(&self) -> &density::Bounds {
        &self.film_bounds
    }
    pub fn exposure_tables(&self) -> &spectral::Tables {
        &self.exposure_tables
    }
    pub fn s_inverse(&self) -> &[f32; 9] {
        &self.s_inverse
    }
    pub fn reconstruction(&self) -> &FilmReconstruction {
        &self.reconstruction
    }
    pub fn input_hull(&self) -> Option<&InputHull> {
        self.input_hull.as_deref()
    }
    pub fn reference(&self) -> &reference::Reference {
        &self.reference
    }
    pub fn scanner(&self) -> &ScannerOutput {
        &self.scanner
    }
    pub fn route(&self) -> &PreparedRoute {
        &self.route
    }
    pub fn profile_route_hash(&self) -> u64 {
        self.profile_route_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn upload_core_hash(&self) -> u64 {
        self.hash
    }
    pub fn scanner_hash(&self) -> u64 {
        self.scanner_hash
    }
}
impl DirectState {
    pub fn correction(&self) -> &scanner::Correction {
        &self.correction
    }
}
impl PrintState {
    pub fn key(&self) -> &str {
        &self.key
    }
    pub fn profile(&self) -> &PrintProfile {
        &self.profile
    }
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
    pub fn curves(&self) -> &PrintDensityCurves {
        &self.curves
    }
    pub fn bounds(&self) -> &print::Bounds {
        &self.bounds
    }
    pub fn calibration(&self) -> Calibration {
        self.calibration
    }
    pub fn neutral_cmy_cc(&self) -> &[f32; 3] {
        &self.neutral_cmy_cc
    }
    pub fn filters(&self) -> &print::Filters {
        &self.filters
    }
    pub fn exposure(&self) -> &print::Exposure {
        &self.exposure
    }
    pub fn illuminant_key(&self) -> &str {
        self.illuminant_key
    }
    pub fn illuminant_hash(&self) -> u64 {
        self.illuminant_hash
    }
    pub fn source_illuminant(&self) -> &[f32; 81] {
        &self.source_illuminant
    }
    pub fn main_illuminant(&self) -> &print::Illuminant {
        &self.main_illuminant
    }
    pub fn preflash(&self) -> Option<&Preflash> {
        self.preflash.as_ref()
    }
    pub fn balance(&self) -> &print::Balance {
        &self.balance
    }
    pub fn resource_descriptors(&self) -> &print::ResourceDescriptors {
        &self.resource_descriptors
    }
    pub fn correction(&self) -> &scanner::Correction {
        &self.correction
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}
impl Preflash {
    pub fn illuminant(&self) -> &print::Illuminant {
        &self.illuminant
    }
    pub fn raw(&self) -> &print::PreflashRaw {
        &self.raw
    }
}
impl ScannerOutput {
    pub fn params(&self) -> &ScannerParams {
        &self.params
    }
    pub fn viewing_illuminant(&self) -> &str {
        &self.viewing_illuminant
    }
    pub fn tables(&self) -> &spectral::Tables {
        &self.tables
    }
    pub fn color(&self) -> &scanner::Color {
        &self.color
    }
    pub fn lut(&self) -> &scanner::LutDescriptor {
        &self.lut
    }
    pub fn post_effects(&self) -> &scanner::PostEffects {
        &self.post_effects
    }
    pub fn gamut(&self) -> &OutputGamut {
        &self.gamut
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}
impl OutputGamut {
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn transform(&self) -> &gamut::OutputTransform {
        &self.transform
    }
    pub fn boundary(&self) -> Option<&gamut::OutputBoundary> {
        self.boundary.as_ref().map(|entry| entry.boundary())
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn lightness_knee(&self) -> &[f32; 3] {
        &OUTPUT_KNEE
    }
    pub fn chroma_knee(&self) -> &[f32; 3] {
        &OUTPUT_KNEE
    }
}
impl FilmReconstruction {
    pub fn tc_lut(&self) -> Option<&reconstruction::FilmTcLut> {
        match &self.storage {
            ReconstructionStorage::Tc { lut, .. } => Some(lut),
            ReconstructionStorage::Mallett { .. } => None,
        }
    }
    pub fn source_lut(&self) -> Option<&SpectraLut> {
        match &self.storage {
            ReconstructionStorage::Tc { source, .. } => Some(source),
            ReconstructionStorage::Mallett { .. } => None,
        }
    }
    pub fn mallett_basis(&self) -> Option<&[[f32; 3]; 81]> {
        match &self.storage {
            ReconstructionStorage::Mallett { basis, .. } => Some(basis),
            ReconstructionStorage::Tc { .. } => None,
        }
    }
    pub fn raw_midgray_rgb(&self) -> Option<&[f32; 3]> {
        match &self.storage {
            ReconstructionStorage::Tc {
                raw_midgray_rgb, ..
            } => Some(raw_midgray_rgb),
            ReconstructionStorage::Mallett { .. } => None,
        }
    }
    pub fn raw_midgray_bgr(&self) -> Option<&[f32; 3]> {
        match &self.storage {
            ReconstructionStorage::Mallett { midgray, .. } => Some(midgray.raw_midgray_bgr()),
            ReconstructionStorage::Tc { .. } => None,
        }
    }
    pub fn midgray_dwg_rgb(&self) -> &[f32; 3] {
        match &self.storage {
            ReconstructionStorage::Mallett { midgray, .. } => midgray.midgray_dwg_rgb(),
            ReconstructionStorage::Tc { .. } => &[0.184; 3],
        }
    }
    pub fn normalization(&self) -> &exposure::MidgrayNormalization {
        match &self.storage {
            ReconstructionStorage::Mallett { midgray, .. } => midgray.normalization(),
            ReconstructionStorage::Tc { normalization, .. } => normalization,
        }
    }
}

fn route_key(route: ScanRoute) -> &'static str {
    match route {
        ScanRoute::NegativeDirect => "NegativeDirectScan",
        ScanRoute::NegativePrint => "NegativePrintScan",
        ScanRoute::PositiveDirect => "PositiveDirectScan",
        ScanRoute::PositivePrint => "PositivePrintScan",
    }
}

fn route_tag(route: ScanRoute) -> u8 {
    match route {
        ScanRoute::NegativeDirect => 0,
        ScanRoute::NegativePrint => 1,
        ScanRoute::PositiveDirect => 2,
        ScanRoute::PositivePrint => 3,
    }
}
fn print_route(route: ScanRoute) -> bool {
    matches!(route, ScanRoute::NegativePrint | ScanRoute::PositivePrint)
}
fn polarity_tag(polarity: Polarity) -> u8 {
    match polarity {
        Polarity::Negative => 0,
        Polarity::Positive => 1,
    }
}
fn update_floats(identity: &mut u64, values: &[f32]) {
    for value in values {
        hash::update_bytes(identity, &value.to_bits().to_le_bytes());
    }
}

struct Observer {
    xyz: [[f32; 81]; 3],
}
impl Observer {
    fn acquire(assets: &Assets) -> Result<Self, PrepareError> {
        let rows = assets.cmf()?;
        if rows.rows().len() != 81 {
            return Err(ControlError::ExposureTables.into());
        }
        Ok(Self {
            xyz: std::array::from_fn(|c| std::array::from_fn(|i| rows.rows()[i][c + 1])),
        })
    }
    fn borrow(&self) -> spectral::Observer<'_> {
        spectral::Observer {
            x: &self.xyz[0],
            y: &self.xyz[1],
            z: &self.xyz[2],
        }
    }
}

fn named_illuminant(curves: &Illuminants, label: &str) -> Option<[f32; 81]> {
    let key = normalize_illuminant(label);
    match key.as_str() {
        "D65" => curves.d65().copied(),
        "D55" => curves.d55().copied(),
        "D50" => curves.d50().copied(),
        "TH-KG3-L" | "THKG3L" | "TH-KG3L" => curves.tungsten_kg3_lens().copied(),
        "TH-KG3" | "THKG3" => curves.tungsten_kg3().copied(),
        "T" | "INCANDESCENT" => curves.tungsten().copied(),
        "K75P" | "KINOTON75P" => curves.k75p().copied(),
        "EQUAL" | "EQUALENERGY" | "EQUAL-ENERGY" => Some(illuminant::equal_energy()),
        _ => None,
    }
}
fn profile_illuminant(
    curves: &Illuminants,
    label: &str,
    kind: IlluminantKind,
) -> Option<[f32; 81]> {
    match kind {
        IlluminantKind::Blackbody { temperature_kelvin } => {
            Some(illuminant::blackbody(temperature_kelvin as f32))
        }
        IlluminantKind::Named => named_illuminant(curves, label),
    }
}
fn foundation_reference(curves: &Illuminants, profile: &FilmProfile) -> Option<[f32; 81]> {
    if let IlluminantKind::Blackbody { temperature_kelvin } =
        profile.info().reference_illuminant_kind()
    {
        return Some(illuminant::blackbody(temperature_kelvin as f32));
    }
    let label = normalize_illuminant(profile.info().reference_illuminant());
    match label.as_str() {
        "D65" => curves.d65().copied(),
        "D55" => curves.d55().copied(),
        "D50" => curves.d50().copied(),
        "T" | "TUNGSTEN" => curves.tungsten().copied(),
        "TH-KG3" | "TUNGSTEN-KG3" => curves.tungsten_kg3_lens().copied(),
        _ => None,
    }
}

fn normalize_illuminant(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut hyphen = true;
    for byte in raw.bytes() {
        if byte.is_ascii_alphanumeric() {
            result.push(byte.to_ascii_uppercase() as char);
            hyphen = false;
        } else if (byte == b'-' || byte == b'_' || byte.is_ascii_whitespace())
            && !result.is_empty()
            && !hyphen
        {
            result.push('-');
            hyphen = true;
        }
    }
    while result.ends_with('-') {
        result.pop();
    }
    if result.is_empty() {
        result = raw
            .chars()
            .filter(|c| !c.is_ascii_whitespace())
            .map(|c| c.to_ascii_uppercase())
            .collect();
    }
    result
}

struct FoundationInput {
    illuminants: Arc<Illuminants>,
    development: film::Development,
    observer: Observer,
    reference_samples: Option<[f32; 81]>,
    reference_white: Option<[f32; 3]>,
    integrated_d65: Option<[f32; 3]>,
    reconstructed_white: Option<reconstruction::ReferenceWhite>,
    tc_source: Option<Arc<SpectraLut>>,
    mallett: Option<Arc<[[f32; 3]; 81]>>,
    hull: Option<Arc<InputHull>>,
}
fn acquire_foundation(
    assets: &Assets,
    params: &Params,
    profile: Arc<FilmProfile>,
) -> Result<FoundationInput, PrepareError> {
    trace_stage!(Illuminants);
    let illuminants = assets.illuminants()?;
    trace_stage!(Development);
    let development = film::Development::new(
        profile,
        film::Params {
            gamma_factor: params.film_gamma_factor,
            layers_required: params.grain.active && params.grain.sublayers_active,
            requested_route: params.scan_route,
        },
    )
    .map_err(PrepareError::Development)?;
    let profile = development.profile();
    let reference_samples = foundation_reference(&illuminants, profile);
    trace_stage!(Observer);
    let observer = Observer::acquire(assets)?;
    let reference_white = profile_illuminant(
        &illuminants,
        profile.info().reference_illuminant(),
        profile.info().reference_illuminant_kind(),
    )
    .and_then(|samples| spectral::integrate_white(observer.borrow(), &samples).ok())
    .map(|white| *white.xyz());
    let mut integrated_d65 = None;
    let mut reconstructed_white = None;
    let mut tc_source = None;
    let mut mallett = None;
    trace_stage!(ReconstructionSource);
    match params.spectral_upsampling_mode {
        0 => {
            let source = assets.hanatos()?;
            if params.hanatos2025_adaptation_window != 0
                && let Some(white) = reference_white
            {
                let prepared = reconstruction::reference_white(
                    source
                        .samples()
                        .try_into()
                        .expect("Assets establishes reconstruction shape"),
                    profile
                        .processing_defaults()
                        .hanatos_spectral_gaussian_blur_default,
                    white,
                );
                reconstructed_white = match prepared {
                    Ok(white) => Some(white),
                    Err(error @ reconstruction::Error::AllocationFailure) => {
                        return Err(PrepareError::ReferenceWhite(error));
                    }
                    Err(_) => None,
                };
            }
            tc_source = Some(source);
        }
        1 => mallett = Some(assets.mallett()?),
        2 => {
            integrated_d65 = illuminants
                .d65()
                .and_then(|samples| spectral::integrate_white(observer.borrow(), samples).ok())
                .map(|white| *white.xyz());
            tc_source = Some(assets.arctic()?);
        }
        _ => (),
    }
    let hull = if matches!(params.spectral_upsampling_mode, 0 | 2)
        && params.input_compression_enabled != 0
    {
        trace_stage!(InputHull);
        match assets.input_hull() {
            Ok(hull) => Some(hull),
            Err(AssetError::Derived(
                crate::assets::DerivedError::Hull(_) | crate::assets::DerivedError::Resource(_),
            )) => None,
            Err(error) => return Err(error.into()),
        }
    } else {
        None
    };
    Ok(FoundationInput {
        illuminants,
        development,
        observer,
        reference_samples,
        reference_white,
        integrated_d65,
        reconstructed_white,
        tc_source,
        mallett,
        hull,
    })
}

fn enlarger_illuminant(choice: i32) -> &'static str {
    match choice {
        0 => "D65",
        1 => "D55",
        2 => "D50",
        4 => "TH-KG3-L",
        5 => "T",
        6 => "K75P",
        7 => "EQUAL",
        _ => "TH-KG3",
    }
}
fn acquire_calibration(
    assets: &Assets,
    film: &FilmProfile,
    print: &PrintProfile,
    label: &str,
) -> Result<Calibration, PrepareError> {
    let outcome = match assets.neutral_calibration() {
        Ok(calibration) => calibration.lookup(print.info().stock(), label, film.info().stock()),
        Err(AssetError::Calibration(error)) => match error.kind() {
            calibration::ErrorKind::MissingFile => return Ok(Calibration::MissingFile),
            calibration::ErrorKind::Read(std::io::ErrorKind::OutOfMemory) => {
                return Err(AssetError::Calibration(error).into());
            }
            kind => return Err(PrepareError::Calibration(kind)),
        },
        Err(error) => return Err(error.into()),
    };
    match outcome {
        Ok(Some(cc)) => Ok(Calibration::Calibrated(cc)),
        Ok(None) => Ok(Calibration::MissingEntry),
        Err(error) => Err(PrepareError::Calibration(error.kind())),
    }
}

fn route_prechecks(
    params: &Params,
    print: Option<&PrintProfile>,
    calibration: Option<Calibration>,
) -> Result<(), PrepareError> {
    let scanner = [
        params.scanner_black_level,
        params.scanner_white_level,
        params.scanner_lens_blur_sigma_px,
        params.scanner_unsharp_mask[0],
        params.scanner_unsharp_mask[1],
    ];
    if let Some(print) = print {
        if !params.print_gamma_factor.is_finite()
            || !(0.5..=2.0).contains(&params.print_gamma_factor)
        {
            return Err(ControlError::PrintGamma.into());
        }
        if print.info().stage() != crate::profile::Stage::Printing {
            return Err(ControlError::PrintStage.into());
        }
        let neutral = match calibration {
            Some(Calibration::Calibrated(cc)) => cc,
            _ => print::CURRENT_NEUTRAL_CMY_CC,
        };
        if neutral.iter().any(|v| !v.is_finite())
            || scanner
                .iter()
                .chain(&params.print_ui_ymc_cc)
                .chain(&[
                    params.preflash_m_filter_cc,
                    params.preflash_y_filter_cc,
                    params.print_exposure,
                    params.print_preflash_exposure,
                    params.camera_exposure_compensation_ev,
                    params.glare_percent,
                    params.glare_roughness,
                    params.glare_blur_sigma_px,
                ])
                .any(|v| !(*v as f32).is_finite())
        {
            return Err(ControlError::PrintInput.into());
        }
    } else if scanner.iter().any(|v| !(*v as f32).is_finite()) {
        return Err(ControlError::DirectScanner.into());
    }
    Ok(())
}

fn input_params(params: &Params) -> Result<input::Params, PrepareError> {
    let input_space = match params.input_color_space {
        0 => color::InputSpace::DavinciWideGamut,
        1 => color::InputSpace::Bt2020,
        2 => color::InputSpace::Aces2065_1,
        3 => color::InputSpace::SrgbRec709,
        _ => return Err(ControlError::InputColorSpace.into()),
    };
    let method = match params.spectral_upsampling_mode {
        0 => input::Method::Hanatos2025,
        1 => input::Method::Mallett2019,
        2 => input::Method::Arctic2026Beta04,
        _ => return Err(ControlError::ReconstructionMethod.into()),
    };
    let meter = match params.camera_metering_method {
        0 => input::Meter::CenterWeighted,
        1 => input::Meter::Average,
        2 => input::Meter::Median,
        3 => input::Meter::Partial,
        4 => input::Meter::Matrix,
        5 => input::Meter::MultiZone,
        6 => input::Meter::HighlightWeighted,
        _ => return Err(ControlError::Meter.into()),
    };
    Ok(input::Params {
        input_space,
        method,
        meter,
        decode_cctf: params.input_cctf_decoding != 0,
        auto_exposure_enabled: params.camera_auto_exposure_enabled != 0,
        manual_exposure_ev: params.camera_exposure_compensation_ev as f32,
        film_format_long_edge_mm: params.camera_film_format_long_edge_mm,
        filter_override: params.camera_filter_override,
        uv: params.camera_filter_uv,
        ir: params.camera_filter_ir,
        apply_hanatos_window: params.hanatos2025_adaptation_window != 0,
        apply_hanatos_surface: params.hanatos2025_adaptation_surface != 0,
        input_compression: params.input_compression_enabled != 0,
    })
}

fn build_tables(
    profile: &crate::profile::ProfileTables,
    samples: &[f32; 81],
    white: &spectral::White,
    observer: &Observer,
) -> spectral::Tables {
    spectral::build_tables(spectral::Input {
        dyes_cmy: profile.channel_density(),
        observer: observer.borrow(),
        illuminant: samples,
        baseline: Some(spectral::Baseline {
            minimum: profile.base_density(),
            midpoint: None,
        }),
        illuminant_hash: white.hash(),
    })
}
fn complete_reconstruction(
    input: &input::Recipe,
    tables: &spectral::Tables,
    tc_source: Option<Arc<SpectraLut>>,
    mallett: Option<Arc<[[f32; 3]; 81]>>,
    hull: Option<&InputHull>,
) -> Result<FilmReconstruction, PrepareError> {
    let storage = if matches!(input.method(), input::Method::Mallett2019) {
        let basis = mallett.expect("Mallett source acquired before composition");
        let midgray = exposure::mallett_midgray(exposure::MallettInput {
            color: input.color().to_conversion(),
            xyz_to_linear_srgb: *input.color().xyz_to_linear_srgb(),
            basis_rgb: &basis,
            illuminant: tables.illuminant(),
            sensitivity_rgb: input.sensitivity().values_rgb(),
        });
        ReconstructionStorage::Mallett { basis, midgray }
    } else {
        let source = tc_source.expect("TC source acquired before composition");
        let lut = reconstruction::build_tc_lut(reconstruction::Input {
            spectra: source
                .samples()
                .try_into()
                .expect("Assets establishes reconstruction shape"),
            sensitivity_rgb: input.sensitivity().values_rgb(),
            reference_illuminant: tables.illuminant(),
            projection_white_xyz: *input.color().projection_white_xyz(),
            method: if matches!(input.method(), input::Method::Arctic2026Beta04) {
                reconstruction::Method::Arctic
            } else {
                reconstruction::Method::Hanatos {
                    spectral_blur: input.hanatos_spectral_blur(),
                    surface_rgb: input
                        .apply_hanatos_surface()
                        .then(|| input.hanatos_surface_rgb()),
                }
            },
            input_compression_active: input.input_compression_active(),
            input_hull: hull
                .map(|entry| gamut::InputHull::new(*entry.hull().center(), entry.hull().xy())),
        })
        .map_err(PrepareError::Tc)?;
        let xyz = color::project_linear_rgb_to_xyz(
            [0.184; 3],
            *input.color().rgb_to_xyz(),
            *input.color().xyz_adapt(),
        );
        let raw_midgray_rgb =
            reconstruction::sample_tc_lut(lut.samples(), xyz).map_err(PrepareError::Tc)?;
        let normalization = exposure::tc_midgray(raw_midgray_rgb[1]);
        ReconstructionStorage::Tc {
            source,
            lut,
            raw_midgray_rgb,
            normalization,
        }
    };
    Ok(FilmReconstruction { storage })
}

fn output_policy(params: &Params) -> Result<(gamut::OutputTransform, u64), PrepareError> {
    let space = color::OutputSpace::try_from(params.output_color_space as u32)
        .map_err(|_| ControlError::OutputColorSpace)?;
    let output = color::output_color(space);
    let transform = gamut::output_transform(gamut::TransformInput {
        space,
        rgb_to_xyz: output.rgb_to_xyz(),
        xyz_to_rgb: output.xyz_to_rgb(),
        white_xyz: output.white_xyz(),
        d65_white_xyz: &color::input_matrices(color::InputSpace::SrgbRec709).d65_white_xyz,
    })
    .map_err(PrepareError::OutputTransform)?;
    let mut hash = hash::FNV_OFFSET;
    hash::update_bytes(
        &mut hash,
        &[u8::from(params.output_gamut_compression_enabled != 0)],
    );
    update_floats(&mut hash, &OUTPUT_KNEE);
    update_floats(&mut hash, &OUTPUT_KNEE);
    hash::update_bytes(&mut hash, &params.output_color_space.to_le_bytes());
    hash::update_bytes(&mut hash, &transform.contract_hash().to_le_bytes());
    Ok((transform, hash))
}
fn scanner_params(params: &Params, space: color::OutputSpace) -> ScannerParams {
    let print = print_route(params.scan_route);
    ScannerParams {
        resolution: params.scanner_lut_resolution.clamp(17, 128) as u32,
        output_space: space,
        encode_cctf: params.output_cctf_encoding != 0,
        black_correction: params.scanner_black_correction != 0,
        white_correction: params.scanner_white_correction != 0,
        black_level: params.scanner_black_level as f32,
        white_level: params.scanner_white_level as f32,
        glare_active: print && params.glare_active,
        glare_percent: if print {
            params.glare_percent as f32
        } else {
            0.0
        },
        glare_roughness: if print {
            params.glare_roughness as f32
        } else {
            0.0
        },
        glare_blur_sigma_px: if print {
            params.glare_blur_sigma_px as f32
        } else {
            0.0
        },
        lens_blur_sigma_px: params.scanner_lens_blur_sigma_px as f32,
        unsharp_sigma_px: params.scanner_unsharp_mask[0] as f32,
        unsharp_amount: params.scanner_unsharp_mask[1] as f32,
    }
}
fn profile_route_hash(params: &Params, film: &FilmProfile, print: Option<&PrintProfile>) -> u64 {
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(&mut identity, params.film_profile_key.as_bytes());
    hash::update_bytes(&mut identity, &film.asset_token().to_le_bytes());
    hash::update_bytes(
        &mut identity,
        &[
            route_tag(params.scan_route),
            polarity_tag(film.info().polarity()),
        ],
    );
    if let Some(print) = print {
        hash::update_bytes(&mut identity, params.print_profile_key.as_bytes());
        hash::update_bytes(&mut identity, &print.asset_token().to_le_bytes());
    }
    identity
}
fn scanner_recipe_hash(
    params: ScannerParams,
    route: ScanRoute,
    polarity: Polarity,
    viewing: &str,
    reference_hash: u64,
    gamut_hash: u64,
) -> u64 {
    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(
        &mut identity,
        &[
            route_tag(route),
            u8::from(print_route(route)),
            polarity_tag(polarity),
        ],
    );
    hash::update_bytes(&mut identity, viewing.as_bytes());
    hash::update_bytes(&mut identity, &params.resolution.to_le_bytes());
    hash::update_bytes(&mut identity, &(params.output_space as i32).to_le_bytes());
    hash::update_bytes(
        &mut identity,
        &[
            u8::from(params.encode_cctf),
            u8::from(params.black_correction),
            u8::from(params.white_correction),
        ],
    );
    update_floats(&mut identity, &[params.black_level, params.white_level]);
    hash::update_bytes(&mut identity, &[u8::from(params.glare_active)]);
    update_floats(
        &mut identity,
        &[
            params.glare_percent,
            params.glare_roughness,
            params.glare_blur_sigma_px,
            params.lens_blur_sigma_px,
            params.unsharp_sigma_px,
            params.unsharp_amount,
        ],
    );
    hash::update_bytes(&mut identity, &reference_hash.to_le_bytes());
    hash::update_bytes(&mut identity, &gamut_hash.to_le_bytes());
    identity
}
fn lut_descriptor(
    params: &Params,
    film: &FilmProfile,
    film_bounds: &density::Bounds,
    print: Option<(&PrintProfile, &print::Bounds)>,
    viewing: &str,
) -> Result<scanner::LutDescriptor, PrepareError> {
    let common = scanner::LutDescriptorParams {
        route: params.scan_route,
        capture_polarity: polarity_tag(film.info().polarity()),
        bounds_medium: u8::from(print.is_some()),
        density_bounds_hash: print.map_or(film_bounds.hash(), |(_, bounds)| bounds.hash()),
        viewing_illuminant: viewing.as_bytes(),
        observer_identity: b"CIE1931_2deg_380_780_5nm",
        resolution: params.scanner_lut_resolution.clamp(17, 128) as u32,
    };
    let input = match print {
        Some((profile, _)) => scanner::LutDescriptorInput::Print {
            params: common,
            print: Some(scanner::LutSpectra {
                channel_density_cmy: profile.tables().channel_density(),
                base_density: profile.tables().base_density(),
            }),
            profile_key: params.print_profile_key.as_bytes(),
            profile_token: profile.asset_token(),
            bounds_source: 2,
            output_medium: 1,
            profile_viewing_illuminant: profile.info().viewing_illuminant().as_bytes(),
        },
        None => scanner::LutDescriptorInput::Direct {
            params: common,
            film: Some(scanner::LutSpectra {
                channel_density_cmy: film.tables().channel_density(),
                base_density: film.tables().base_density(),
            }),
        },
    };
    scanner::describe_lut(input).map_err(PrepareError::LutDescriptor)
}
fn correction_tables(tables: &spectral::Tables) -> scanner::ReferenceTables<'_> {
    scanner::ReferenceTables {
        eps_c: &tables.dyes_cmy()[0],
        eps_m: &tables.dyes_cmy()[1],
        eps_y: &tables.dyes_cmy()[2],
        weighted_y: &tables.weighted_xyz()[1],
        baseline_min: tables
            .has_baseline()
            .then(|| tables.baseline_min().as_slice()),
        sample_count: 81,
        inv_yn: tables.inv_yn(),
    }
}

struct PrintAssembly {
    profile: Arc<PrintProfile>,
    bounds: print::Bounds,
    curves: PrintDensityCurves,
    filters: print::Filters,
    exposure: print::Exposure,
    calibration: Calibration,
    neutral_cmy_cc: [f32; 3],
    illuminant_key: &'static str,
    illuminant_hash: u64,
}

/// Complete one direct or print host state. No foreign record, frame geometry,
/// CUDA residency, publication mutation or intermediate success escapes here.
pub fn prepare(assets: &Assets, params: &Params) -> Result<RenderState, PrepareError> {
    #[cfg(feature = "test-support")]
    testing::reset();
    trace_stage!(SelectedFilm);
    let film = assets.film(&params.film_profile_key)?;
    let selected_print = if print_route(params.scan_route) {
        trace_stage!(SelectedPrint);
        Some(assets.print(&params.print_profile_key)?)
    } else {
        None
    };
    let acquired = acquire_foundation(assets, params, film)?;
    let film = acquired.development.profile();
    let calibration = selected_print
        .as_ref()
        .map(|print| {
            trace_stage!(Calibration);
            acquire_calibration(assets, film, print, enlarger_illuminant(params.enl_ill))
        })
        .transpose()?;
    trace_stage!(RoutePrechecks);
    route_prechecks(params, selected_print.as_deref(), calibration)?;

    trace_stage!(Foundation);
    let reference_samples = acquired
        .reference_samples
        .as_ref()
        .ok_or(ControlError::ReferenceIlluminant)?;
    let input = input::Recipe::new(
        film,
        input_params(params)?,
        input::Resources {
            reference_illuminant: reference_samples,
            reference_white_xyz: acquired.reference_white.as_ref(),
            integrated_d65_xyz: acquired.integrated_d65.as_ref(),
            reconstructed_reference_white: acquired
                .reconstructed_white
                .as_ref()
                .map(|white| white.samples()),
            tc_source_hash: acquired
                .tc_source
                .as_ref()
                .map_or(0, |source| source.asset_hash()),
            input_hull_hash: acquired.hull.as_ref().map_or(0, |hull| hull.hull().hash()),
        },
    )
    .map_err(PrepareError::Input)?;
    let defaults = film.processing_defaults();
    let controls = params.scatter_halation;
    let optics = optics::Recipe::new(optics::Params {
        camera_diffusion: params.camera_diffusion,
        camera_lens_blur_um: 0.0,
        scatter_halation: optics::HalationParams {
            active: controls.active,
            scatter_amount: controls.scatter_amount,
            scatter_spatial_scale: controls.scatter_spatial_scale,
            halation_amount: controls.halation_amount,
            halation_spatial_scale: controls.halation_spatial_scale,
            first_sigma_um: defaults.halation_first_sigma_um,
            primary_amount: defaults.halation_primary_amount,
        },
        enlarger_diffusion: params.enlarger_diffusion,
        print_domain: selected_print.is_some(),
    })
    .map_err(PrepareError::Optics)?;
    let minimum = density::Minimum::new([0.07, 0.08, 0.12]).map_err(PrepareError::Minimum)?;
    let grain =
        grain::Recipe::new(&acquired.development, params.grain).map_err(PrepareError::Grain)?;
    let effects = effects::Recipe::new(effects::Params {
        film_dust_amount: params.film_dust_amount,
        film_scratch_amount: params.film_scratch_amount,
        gate_dust_amount: params.gate_dust_amount,
        gate_scratch_amount: params.gate_scratch_amount,
        gate_weave_amount: params.gate_weave_amount,
        grain_active: grain.active(),
        grain_debug_view: grain.debug_view(),
    })
    .map_err(PrepareError::Effects)?;
    let dir = dir::Recipe::new(&acquired.development, params.dir_couplers).map_err(|error| {
        PrepareError::Dir {
            error,
            route: params.scan_route,
            stock: film.info().stock().to_owned(),
        }
    })?;
    let film_bounds =
        density::Bounds::new(&acquired.development, &minimum).map_err(PrepareError::Bounds)?;
    let profile_route_hash = profile_route_hash(params, film, selected_print.as_deref());

    // Route composition follows foundation, before focused payload construction.
    let print_bounds = selected_print
        .as_ref()
        .map(|profile| print::Bounds::new(profile, params.scan_route, film.info().polarity()))
        .transpose()
        .map_err(PrepareError::PrintComposition)?;
    let (transform, gamut_hash) = output_policy(params)?;
    let scanner_params = scanner_params(params, transform.space());
    let print = match selected_print {
        Some(profile) => {
            let curves = profile
                .sample_density_curves(params.print_gamma_factor)
                .map_err(PrepareError::PrintDevelopment)?;
            let calibration =
                calibration.expect("print calibration acquired before route prechecks");
            let neutral = match calibration {
                Calibration::Calibrated(cc) => cc,
                _ => print::CURRENT_NEUTRAL_CMY_CC,
            };
            let filters = print::Filters::new(
                neutral,
                params.print_ui_ymc_cc.map(|value| value as f32),
                params.preflash_m_filter_cc as f32,
                params.preflash_y_filter_cc as f32,
            )
            .map_err(PrepareError::PrintComposition)?;
            let exposure = print::Exposure::new(print::ExposureParams {
                print_exposure: params.print_exposure as f32,
                preflash_exposure: params.print_preflash_exposure as f32,
                camera_exposure_ev: params.camera_exposure_compensation_ev as f32,
                normalize: params.normalize_print_exposure != 0,
                compensate: params.print_exposure_compensation != 0,
            })
            .map_err(PrepareError::PrintComposition)?;
            let illuminant_key = enlarger_illuminant(params.enl_ill);
            Some(PrintAssembly {
                profile,
                bounds: print_bounds.expect("print bounds composed"),
                curves,
                filters,
                exposure,
                calibration,
                neutral_cmy_cc: neutral,
                illuminant_key,
                illuminant_hash: hash::bytes(illuminant_key.as_bytes()),
            })
        }
        None => None,
    };
    let viewing = print
        .as_ref()
        .map_or_else(
            || film.info().viewing_illuminant(),
            |print| print.profile.info().viewing_illuminant(),
        )
        .to_owned();
    let direct_lut = if print.is_none() {
        Some(lut_descriptor(params, film, &film_bounds, None, &viewing)?)
    } else {
        None
    };

    trace_stage!(FilmPayload);
    let film_illuminant = profile_illuminant(
        &acquired.illuminants,
        film.info().reference_illuminant(),
        film.info().reference_illuminant_kind(),
    )
    .ok_or(ControlError::FilmReferenceIlluminant)?;
    let film_white = spectral::integrate_white(acquired.observer.borrow(), &film_illuminant)
        .map_err(|_| ControlError::FilmReferenceIlluminant)?;
    let exposure_tables = build_tables(
        film.tables(),
        &film_illuminant,
        &film_white,
        &acquired.observer,
    );
    if exposure_tables.hash() == 0
        || !exposure_tables.white_xyz().iter().all(|v| v.is_finite())
        || exposure_tables.white_xyz()[1] <= 0.0
    {
        return Err(ControlError::ExposureTables.into());
    }
    let weights = exposure_tables.weighted_xyz();
    let s_inverse = spectral::s_inverse([&weights[0], &weights[1], &weights[2]]);
    let reconstruction = complete_reconstruction(
        &input,
        &exposure_tables,
        acquired.tc_source,
        acquired.mallett,
        acquired.hull.as_deref(),
    )?;

    // Main print illumination precedes the synthetic reference in native assembly.
    let print_main = print
        .as_ref()
        .map(|print| {
            trace_stage!(PrintMainIlluminant);
            let source = named_illuminant(&acquired.illuminants, print.illuminant_key)
                .ok_or(ControlError::PrintIlluminant)?;
            let main = print::Illuminant::new(print.filters.illumination_params(false), &source)
                .map_err(PrepareError::PrintIlluminant)?;
            Ok::<_, PrepareError>((source, main))
        })
        .transpose()?;
    let reference_input = match &reconstruction.storage {
        ReconstructionStorage::Tc { lut, .. } => reference::Reconstruction::Tc {
            samples: lut.samples(),
            projection_white_xyz: *input.color().projection_white_xyz(),
        },
        ReconstructionStorage::Mallett { basis, .. } => reference::Reconstruction::Mallett {
            basis_rgb: basis,
            illuminant: &film_illuminant,
            sensitivity_rgb: input.sensitivity().values_rgb(),
            green_scale: input.sensitivity().mallett_green_scale(),
        },
    };
    trace_stage!(FilmReference);
    let reference = reference::Reference::new(
        &acquired.development,
        reference::Params {
            manual_exposure_ev: input.manual_exposure_ev(),
            auto_exposure_enabled: input.auto_exposure_enabled(),
        },
        Some(reference_input),
    )
    .map_err(PrepareError::Reference)?;
    let print_balance = match (&print, &print_main) {
        (Some(print), Some((_, main))) => {
            trace_stage!(PrintBalance);
            Some(
                print::Balance::new(
                    print::BalanceParams {
                        baseline_density_cmy: *reference.baseline_density_cmy(),
                        compensated_density_cmy: *reference.compensated_density_cmy(),
                        reference_hash: reference.hash(),
                        normalization_mode: print.exposure.normalization_mode(),
                    },
                    film.tables().base_density(),
                    film.tables().channel_density(),
                    print.profile.tables().linear_sensitivity(),
                    main.samples(),
                )
                .map_err(PrepareError::PrintBalance)?,
            )
        }
        _ => None,
    };

    let (scanner_profile, viewing_kind) = match &print {
        Some(print) => (
            print.profile.tables(),
            print.profile.info().viewing_illuminant_kind(),
        ),
        None => (film.tables(), film.info().viewing_illuminant_kind()),
    };
    trace_stage!(ScannerPayload);
    let scanner_illuminant = profile_illuminant(&acquired.illuminants, &viewing, viewing_kind)
        .ok_or(ControlError::ScannerPayload(params.scan_route))?;
    let scanner_white = spectral::integrate_white(acquired.observer.borrow(), &scanner_illuminant)
        .map_err(|_| ControlError::ScannerPayload(params.scan_route))?;
    let scanner_tables = build_tables(
        scanner_profile,
        &scanner_illuminant,
        &scanner_white,
        &acquired.observer,
    );
    let active_gamut_hash = if params.output_gamut_compression_enabled != 0 {
        gamut_hash
    } else {
        0
    };
    let scanner_color = scanner::Color::new(scanner::ColorInput {
        medium: u8::from(print.is_some()),
        output_space: scanner_params.output_space as u8,
        apply_cctf: scanner_params.encode_cctf,
        illuminant_xyz: *scanner_white.xyz(),
        illuminant_hash: scanner_white.hash(),
        active_gamut_recipe_hash: active_gamut_hash,
    })
    .map_err(PrepareError::ScannerColor)?;
    let boundary = if params.output_gamut_compression_enabled != 0 {
        trace_stage!(OutputBoundary);
        Some(assets.output_boundary(
            gamut::BoundaryInput {
                space: transform.space(),
                d65_xyz_to_native_rgb: transform.d65_xyz_to_native_rgb(),
                transform_hash: transform.hash(),
            },
            transform.contract_hash(),
        )?)
    } else {
        None
    };
    let gamut = OutputGamut {
        enabled: params.output_gamut_compression_enabled != 0,
        transform,
        boundary,
        hash: gamut_hash,
    };
    let scanner_post_effects = scanner::describe_post_effects(scanner::PostEffectsParams {
        route: params.scan_route,
        glare_active: scanner_params.glare_active,
        glare_percent: scanner_params.glare_percent,
        glare_roughness: scanner_params.glare_roughness,
        glare_blur_sigma_px: scanner_params.glare_blur_sigma_px,
        lens_blur_sigma_px: scanner_params.lens_blur_sigma_px,
        unsharp_sigma_px: scanner_params.unsharp_sigma_px,
        unsharp_amount: scanner_params.unsharp_amount,
    })
    .map_err(PrepareError::ScannerPostEffects)?;

    let correction_params = scanner::Params {
        route: route_tag(params.scan_route),
        black_correction: scanner_params.black_correction,
        white_correction: scanner_params.white_correction,
        black_level: scanner_params.black_level,
        white_level: scanner_params.white_level,
    };
    let (route, lut, scanner_bounds_hash) = match (print, print_main, print_balance) {
        (Some(print), Some((source_illuminant, main_illuminant)), Some(balance)) => {
            let lut = lut_descriptor(
                params,
                film,
                &film_bounds,
                Some((&print.profile, &print.bounds)),
                &viewing,
            )?;
            let resource_descriptors =
                print::ResourceDescriptors::new(print::ResourceDescriptorInput {
                    film,
                    profile: &print.profile,
                    curves: &print.curves,
                    filters: &print.filters,
                    exposure: &print.exposure,
                    balance: &balance,
                    film_input_hash: input.hash(),
                    film_development_hash: acquired.development.hash(),
                    illuminant_hash: print.illuminant_hash,
                })
                .map_err(PrepareError::PrintComposition)?;
            let preflash = if print.exposure.params().preflash_exposure > 0.0 {
                trace_stage!(PrintPreflash);
                let illuminant = print::Illuminant::new(
                    print.filters.illumination_params(true),
                    &source_illuminant,
                )
                .map_err(PrepareError::PrintIlluminant)?;
                let raw = print::PreflashRaw::new(
                    film.tables().base_density(),
                    illuminant.samples(),
                    print.profile.tables().linear_sensitivity(),
                )
                .map_err(PrepareError::PrintPreflash)?;
                Some(Preflash { illuminant, raw })
            } else {
                None
            };
            trace_stage!(ScannerCorrection);
            let correction = scanner::Correction::print(
                correction_params,
                Some(scanner::PrintInput {
                    tables: correction_tables(&scanner_tables),
                    film_base_density: film.tables().base_density(),
                    film_channel_density_cmy: film.tables().channel_density(),
                    print_sensitivity_cmy: print.profile.tables().linear_sensitivity(),
                    filtered_main_illuminant: main_illuminant.samples(),
                    print_base_density: print.profile.tables().base_density(),
                    print_log_exposure: print.profile.tables().interpolation_log_exposure(),
                    print_density_curves: print.profile.tables().density_curves(),
                    minimum_density_cmy: *print.bounds.minimum_cmy(),
                    inverse_span_cmy: *print.bounds.inv_span_cmy(),
                    film_minimum_density_cmy: *film_bounds.minimum_cmy(),
                    film_maximum_density_cmy: *film_bounds.maximum_cmy(),
                    preflash_raw_cmy: preflash
                        .as_ref()
                        .map_or([0.0; 3], |preflash| *preflash.raw.cmy()),
                    normalizer: balance.normalizer(),
                    preflash_exposure: print.exposure.params().preflash_exposure,
                    medium: 1,
                }),
            )
            .map_err(PrepareError::ScannerCorrection)?;
            let bounds_hash = print.bounds.hash();
            let hash = hash::u64s(&[
                print.filters.hash(),
                print.exposure.hash(),
                print.illuminant_hash,
                print.curves.hash(),
                params.print_gamma_factor.to_bits(),
                balance.hash(),
            ]);
            (
                PreparedRoute::Print(PrintState {
                    key: params.print_profile_key.clone(),
                    profile: print.profile,
                    gamma: params.print_gamma_factor,
                    curves: print.curves,
                    bounds: print.bounds,
                    calibration: print.calibration,
                    neutral_cmy_cc: print.neutral_cmy_cc,
                    filters: print.filters,
                    exposure: print.exposure,
                    illuminant_key: print.illuminant_key,
                    illuminant_hash: print.illuminant_hash,
                    source_illuminant,
                    main_illuminant,
                    preflash,
                    balance,
                    resource_descriptors,
                    correction,
                    hash,
                }),
                lut,
                bounds_hash,
            )
        }
        (None, None, None) => {
            trace_stage!(ScannerCorrection);
            let correction = scanner::Correction::direct(
                correction_params,
                Some(scanner::DirectInput {
                    tables: correction_tables(&scanner_tables),
                    film_base_density: film.tables().base_density(),
                    film_log_exposure: acquired.development.log_exposure(),
                    film_authored_density_curves: acquired.development.authored_curves_cmy(),
                    minimum_density_cmy: *film_bounds.minimum_cmy(),
                    maximum_density_cmy: *film_bounds.maximum_cmy(),
                    inverse_span_cmy: *film_bounds.inv_span_cmy(),
                    baseline_density_cmy: *reference.baseline_density_cmy(),
                    reference_hash: reference.hash(),
                    medium: 0,
                }),
            )
            .map_err(PrepareError::ScannerCorrection)?;
            (
                PreparedRoute::Direct(DirectState { correction }),
                direct_lut.expect("direct descriptor admitted before payload"),
                film_bounds.hash(),
            )
        }
        _ => unreachable!("print route privately owns all print construction products"),
    };
    trace_stage!(FinalIdentity);
    let scanner_recipe_hash = scanner_recipe_hash(
        scanner_params,
        params.scan_route,
        film.info().polarity(),
        &viewing,
        reference.hash(),
        gamut_hash,
    );
    let scanner_hash = hash::u64s(&[
        scanner_bounds_hash,
        scanner_tables.hash(),
        u64::from(scanner_params.resolution),
        active_gamut_hash,
    ]);
    #[cfg(feature = "test-support")]
    let scanner_hash = testing::identity(testing::ZeroIdentity::Scanner, scanner_hash);
    let scanner = ScannerOutput {
        params: scanner_params,
        viewing_illuminant: viewing,
        tables: scanner_tables,
        color: scanner_color,
        lut,
        post_effects: scanner_post_effects,
        gamut,
        hash: scanner_recipe_hash,
    };
    let mut hash = match (&route, dir.active()) {
        (PreparedRoute::Direct(_), false) => hash::u64s(&[
            profile_route_hash,
            input.hash(),
            acquired.development.hash(),
            scanner_bounds_hash,
            scanner_recipe_hash,
        ]),
        (PreparedRoute::Direct(_), true) => hash::u64s(&[
            profile_route_hash,
            input.hash(),
            acquired.development.hash(),
            dir.hash(),
            scanner_bounds_hash,
            scanner_recipe_hash,
        ]),
        (PreparedRoute::Print(print), false) => hash::u64s(&[
            profile_route_hash,
            input.hash(),
            acquired.development.hash(),
            film_bounds.hash(),
            scanner_bounds_hash,
            scanner_recipe_hash,
            print.hash(),
        ]),
        (PreparedRoute::Print(print), true) => hash::u64s(&[
            profile_route_hash,
            input.hash(),
            acquired.development.hash(),
            dir.hash(),
            film_bounds.hash(),
            scanner_bounds_hash,
            scanner_recipe_hash,
            print.hash(),
        ]),
    };
    for optional in [optics.hash(), grain.hash(), effects.hash()] {
        if optional != 0 {
            hash = hash::u64s(&[hash, optional]);
        }
    }
    #[cfg(feature = "test-support")]
    let hash = testing::identity(testing::ZeroIdentity::Recipe, hash);
    if hash == 0 || scanner_hash == 0 || scanner_recipe_hash == 0 || scanner.color.hash() == 0 {
        return Err(ControlError::Identity(params.scan_route).into());
    }
    Ok(RenderState {
        film_key: params.film_profile_key.clone(),
        development: acquired.development,
        input,
        optics,
        minimum,
        grain,
        effects,
        dir,
        film_bounds,
        exposure_tables,
        s_inverse,
        reconstruction,
        input_hull: acquired.hull,
        reference,
        scanner,
        route,
        profile_route_hash,
        hash,
        scanner_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn assets() -> Assets {
        Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"))
    }
    fn params(route: ScanRoute, method: i32) -> Params {
        let diffusion = optics::DiffusionParams {
            active: false,
            family: 1,
            strength: 0.5,
            spatial_scale: 1.0,
            halo_warmth: 0.0,
            core_intensity: 1.0,
            core_size: 1.0,
            halo_intensity: 1.0,
            halo_size: 1.0,
            bloom_intensity: 1.0,
            bloom_size: 1.0,
        };
        Params {
            print_exposure: 1.0,
            print_preflash_exposure: 0.0,
            print_gamma_factor: 1.0,
            preflash_m_filter_cc: 0.0,
            preflash_y_filter_cc: 0.0,
            print_shadow_compensation_factor: 0.0,
            print_shadow_compensation_density: 1.2,
            print_shadow_compensation_transition: 0.3,
            glare_percent: 0.03,
            glare_roughness: 0.7,
            glare_blur_sigma_px: 0.5,
            camera_exposure_compensation_ev: 0.0,
            camera_film_format_long_edge_mm: 35.0,
            film_gamma_factor: 1.0,
            scanner_lens_blur_sigma_px: 0.0,
            scanner_black_level: 0.01,
            scanner_white_level: 0.98,
            gate_weave_amount: 0.0,
            scanner_unsharp_mask: [0.7, 0.7],
            print_ui_ymc_cc: [0.0; 3],
            camera_filter_uv: [1.0, 410.0, 8.0],
            camera_filter_ir: [1.0, 675.0, 15.0],
            camera_diffusion: diffusion,
            enlarger_diffusion: diffusion,
            dir_couplers: dir::Params {
                active: true,
                amount: 1.0,
                inhibition_same_layer: 1.0,
                inhibition_interlayer: 1.0,
                diffusion_size_um: 20.0,
                diffusion_tail_um: 200.0,
                diffusion_tail_weight: 0.03,
                gamma_use_stock: true,
                gamma_same_layer_rgb: [0.341, 0.324, 0.273],
                gamma_interlayer_r_to_gb: [0.355, 0.305],
                gamma_interlayer_g_to_rb: [0.154, 0.358],
                gamma_interlayer_b_to_rg: [0.171, 0.225],
                langmuir_donor_k_rgb: [1.0; 3],
                langmuir_receiver_k_rgb: [1.0; 3],
            },
            film_profile_key: if matches!(
                route,
                ScanRoute::PositiveDirect | ScanRoute::PositivePrint
            ) {
                "fujifilm_provia_100f"
            } else {
                "kodak_portra_400"
            }
            .into(),
            print_profile_key: "kodak_portra_endura".into(),
            spectral_upsampling_mode: method,
            ref_ill: 0,
            enl_ill: 3,
            normalize_print_exposure: 1,
            print_exposure_compensation: 1,
            input_color_space: 0,
            input_cctf_decoding: 0,
            input_compression_enabled: 1,
            hanatos2025_adaptation_window: 1,
            hanatos2025_adaptation_surface: 0,
            camera_auto_exposure_enabled: 1,
            camera_metering_method: 0,
            scatter_halation: HalationControls {
                active: false,
                scatter_amount: 1.0,
                scatter_spatial_scale: 1.0,
                halation_amount: 1.0,
                halation_spatial_scale: 1.0,
            },
            scanner_black_correction: 0,
            scanner_white_correction: 0,
            scanner_lut_resolution: 17,
            output_color_space: 0,
            output_cctf_encoding: 1,
            output_gamut_compression_enabled: 1,
            grain: grain::Params {
                active: false,
                sublayers_active: true,
                particle_area_um2: 0.2,
                particle_scale_cmy: [0.8, 1.0, 2.0],
                particle_scale_layers: [2.5, 1.0, 0.5],
                particle_density_min_cmy: [0.07, 0.08, 0.12],
                uniformity_cmy: [0.97, 0.97, 0.99],
                correlation_sigma_px: 0.65,
                dye_cloud_blur_um: 1.0,
                micro_structure: [0.2, 30.0],
                sublayer_count: 1,
                amplitude: 1.0,
                chroma_mix: 0.0,
                chroma_shared_weight: 1.0,
                chroma_independent_weight: 0.0,
                coarse_weight: 0.0,
                mid_weight: 0.0,
                size_mix_scale: 1.0,
                clump_temporal_mix: 0.0,
                clump_morph_period_sec: 8.0,
                debug_view: 0,
            },
            film_dust_amount: 0.0,
            film_scratch_amount: 0.0,
            gate_dust_amount: 0.0,
            gate_scratch_amount: 0.0,
            scan_route: route,
            glare_active: true,
            camera_filter_override: false,
        }
    }
    #[test]
    fn all_routes_and_methods_complete_with_active_spatial_dir() {
        let assets = assets();
        let mut count = 0;
        for route in [
            ScanRoute::NegativeDirect,
            ScanRoute::NegativePrint,
            ScanRoute::PositiveDirect,
            ScanRoute::PositivePrint,
        ] {
            for method in 0..3 {
                let state = prepare(&assets, &params(route, method)).unwrap_or_else(|error| {
                    panic!("route={} method={method}: {error}", route_key(route))
                });
                assert_eq!(state.development().route(), route);
                assert!(state.dir().active());
                assert_ne!(state.dir().hash(), 0);
                assert_eq!(
                    (
                        state.optics().hash(),
                        state.grain().hash(),
                        state.effects().hash()
                    ),
                    (0, 0, 0)
                );
                assert_ne!(state.hash(), 0);
                assert_ne!(state.scanner().lut().hash(), 0);
                assert_ne!(state.reference().hash(), 0);
                assert_eq!(state.reconstruction().tc_lut().is_some(), method != 1);
                if let PreparedRoute::Print(print) = state.route() {
                    assert_ne!(print.balance().hash(), 0);
                    assert!(print.preflash().is_none());
                    assert_eq!(
                        print.curves().totals().len(),
                        print.profile().tables().interpolation_log_exposure().len()
                    );
                }
                count += 1;
            }
        }
        assert_eq!(count, 12);
    }
    #[test]
    fn direct_ignores_print_only_sources_and_controls() {
        let assets = assets();
        let mut params = params(ScanRoute::NegativeDirect, 1);
        let expected = prepare(&assets, &params).unwrap();
        params.print_profile_key = "absent-print-profile".into();
        params.print_gamma_factor = f64::NAN;
        params.print_exposure = f64::INFINITY;
        params.glare_percent = f64::NAN;
        let actual = prepare(&assets, &params).unwrap();
        assert_eq!(actual.hash(), expected.hash());
        assert_eq!(actual.scanner_hash(), expected.scanner_hash());
    }
    #[test]
    fn route_prechecks_precede_input_policy_after_development() {
        let assets = assets();
        let mut direct = params(ScanRoute::NegativeDirect, 1);
        direct.scanner_black_level = f64::NAN;
        direct.input_color_space = -1;
        assert!(matches!(
            prepare(&assets, &direct),
            Err(PrepareError::Control(ControlError::DirectScanner))
        ));
        direct.film_gamma_factor = 0.0;
        assert!(matches!(
            prepare(&assets, &direct),
            Err(PrepareError::Development(film::Error::InvalidGamma))
        ));
        let mut print = params(ScanRoute::NegativePrint, 1);
        print.input_color_space = -1;
        print.print_gamma_factor = 0.0;
        assert!(matches!(
            prepare(&assets, &print),
            Err(PrepareError::Control(ControlError::PrintGamma))
        ));
        print.print_gamma_factor = 1.0;
        print.print_exposure = f64::NAN;
        assert!(matches!(
            prepare(&assets, &print),
            Err(PrepareError::Control(ControlError::PrintInput))
        ));
    }
    #[test]
    fn state_holds_derived_backing_after_asset_release_and_drop() {
        let assets = assets();
        let state = prepare(&assets, &params(ScanRoute::NegativeDirect, 0)).unwrap();
        let hull = state.input_hull().unwrap().hull().xy().as_ptr();
        let boundary = state.scanner().gamut().boundary().unwrap().cmax().as_ptr();
        let tc = state.reconstruction().tc_lut().unwrap().samples().as_ptr();
        assets.release_cached_payloads().unwrap();
        drop(assets);
        assert_eq!(state.input_hull().unwrap().hull().xy().as_ptr(), hull);
        assert_eq!(
            state.scanner().gamut().boundary().unwrap().cmax().as_ptr(),
            boundary
        );
        assert_eq!(
            state.reconstruction().tc_lut().unwrap().samples().as_ptr(),
            tc
        );
        assert_eq!(
            state.development().log_exposure().len(),
            state.development().normalized_curves_cmy().len()
        );
    }
}
