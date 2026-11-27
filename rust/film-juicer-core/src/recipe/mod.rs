//! Complete host photographic preparation. Construction is private until every
//! route, payload, reference, balance and identity has succeeded.

use std::fmt;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::{collections::TryReserveError, error::Error as StdError};

use crate::assets::{AssetError, Assets, InputHull, SpectraLut};
use crate::capture::Params;
use crate::data_io::calibration;
use crate::film::{self, density, dir, grain, input, reference};
use crate::profile::{FilmProfile, Polarity, PrintProfile, PrintDensityCurves};
use crate::route::ScanRoute;
use crate::{color, effects, exposure, gamut, hash, optics, print, reconstruction, scanner, spectral};

mod film_stage;
mod foundation;
mod print_stage;
mod scanner_stage;

#[cfg(test)]
use crate::capture::HalationControls;
#[cfg(test)]
use print_stage::print_glare_hash;
use scanner_stage::OUTPUT_KNEE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlError {
    FilmGrain,
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
            Self::FilmGrain => {
                "Film v1 requires Kodak Portra 400, a supported camera film format, a negative route and valid Film grain controls"
            }
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

impl StdError for ControlError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilmGrainFootprintKind {
    Rendered,
    Sample,
}

#[derive(Debug)]
pub enum CalibrationError {
    Resource(Arc<calibration::Error>),
    Entry(calibration::Error),
}

impl CalibrationError {
    fn cause(&self) -> &calibration::Error {
        match self {
            Self::Resource(error) => error,
            Self::Entry(error) => error,
        }
    }
}

#[derive(Debug)]
pub enum PrepareError {
    Asset(AssetError),
    Development(film::Error),
    Control(ControlError),
    FilmGrainGeometry,
    FilmGrainFootprint {
        kind: FilmGrainFootprintKind,
        long_edge_mm: f32,
        texel_pitch_um: f32,
        width_texels: f32,
        supported_texels: RangeInclusive<f32>,
    },
    Input(input::Error),
    Optics(optics::Error),
    HighlightBoost(optics::highlight_boost::Error),
    Minimum(density::Error),
    Grain(grain::Error),
    Effects(effects::Error),
    Dir {
        error: dir::Error,
        route: ScanRoute,
        stock: String,
    },
    Bounds(density::Error),
    ReferenceWhite(reconstruction::Error),
    Tc(reconstruction::TcError),
    Reference(reference::Error),
    PrintDevelopment(crate::profile::PrintDensityError),
    PrintComposition(print::CompositionError),
    PrintIlluminant(print::Error),
    PrintBalance(print::BalanceError),
    PrintPreflash(print::PreflashError),
    Calibration(CalibrationError),
    OutputTransform(gamut::TransformError),
    ScannerColor(scanner::ColorError),
    ScannerCorrection(scanner::CorrectionError),
    ScannerPostEffects(scanner::PostEffectsError),
    ContentsAllocation(TryReserveError),
}
impl PrepareError {
    /// Some film-input leaves contain their unwind before returning to the
    /// complete-state owner. That remains a panic, not an ordinary rejection.
    pub fn is_panic(&self) -> bool {
        matches!(self, Self::Input(input::Error::Panic(_)))
    }
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
            Self::FilmGrainGeometry => f.write_str("Film v1 requires rendered frame geometry"),
            Self::FilmGrainFootprint {
                kind,
                long_edge_mm,
                texel_pitch_um,
                width_texels,
                supported_texels,
            } => {
                let kind = match kind {
                    FilmGrainFootprintKind::Rendered => "rendered",
                    FilmGrainFootprintKind::Sample => "sample",
                };
                write!(
                    f,
                    "Film v1 unsupported {kind} footprint (film_long_edge_mm={long_edge_mm}, grain_size_um={texel_pitch_um}, footprint_texels={}, supported_texels={}..={})",
                    f64::from(*width_texels),
                    f64::from(*supported_texels.start()),
                    f64::from(*supported_texels.end()),
                )
            }
            Self::Input(error) => error.fmt(f),
            Self::Optics(error) => error.fmt(f),
            Self::HighlightBoost(error) => error.fmt(f),
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
            Self::Bounds(
                error @ (density::Error::FilmNormalization
                | density::Error::FilmProjectionSpan
                | density::Error::FilmSpectralRange),
            ) => error.fmt(f),
            Self::Bounds(_) => {
                f.write_str("MalformedRequiredProfileData phase=3A field=density_bounds")
            }
            Self::ReferenceWhite(error) => error.fmt(f),
            Self::Tc(error) => error.fmt(f),
            Self::Reference(error) => error.fmt(f),
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
            Self::Calibration(error) => {
                let field = match error.cause().kind() {
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
            Self::ContentsAllocation(_) => {
                f.write_str("Complete render-state backing allocation failed")
            }
        }
    }
}
impl StdError for PrepareError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Asset(error) => Some(error),
            Self::Development(error) => Some(error),
            Self::Control(error) => Some(error),
            Self::Input(error) => Some(error),
            Self::Optics(error) => Some(error),
            Self::HighlightBoost(error) => Some(error),
            Self::Minimum(error) => Some(error),
            Self::Grain(error) => Some(error),
            Self::Effects(error) => Some(error),
            Self::Bounds(error) => Some(error),
            Self::ReferenceWhite(error) => Some(error),
            Self::Tc(error) => Some(error),
            Self::Reference(error) => Some(error),
            Self::PrintDevelopment(error) => Some(error),
            Self::PrintComposition(error) => Some(error),
            Self::PrintIlluminant(error) => Some(error),
            Self::PrintBalance(error) => Some(error),
            Self::PrintPreflash(error) => Some(error),
            Self::Calibration(error) => Some(error.cause()),
            Self::OutputTransform(error) => Some(error),
            Self::ScannerColor(error) => Some(error),
            Self::ScannerCorrection(error) => Some(error),
            Self::ScannerPostEffects(error) => Some(error),
            Self::ContentsAllocation(error) => Some(error),
            Self::Dir { error, .. } => Some(error),
            _ => None,
        }
    }
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

pub struct RenderState {
    contents: Vec<Contents>,
}

struct Contents {
    film_key: String,
    development: film::Development,
    input: input::Recipe,
    optics: optics::Recipe,
    minimum: density::Minimum,
    grain: grain::Recipe,
    film_grain: Option<grain::film_v1::Recipe>,
    effects: effects::Recipe,
    dir: dir::Recipe,
    #[cfg(test)]
    film_v1_bounds: Option<density::Bounds>,
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
    glare_hash: u64,
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
    tables: spectral::Tables,
    color: scanner::Color,
    post_effects: scanner::PostEffects,
    gamut: OutputGamut,
    hash: u64,
}
#[derive(Clone, Copy)]
pub struct ScannerParams {
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
        &self.contents[0].film_key
    }
    pub fn film_profile(&self) -> &FilmProfile {
        self.contents[0].development.profile()
    }
    pub fn development(&self) -> &film::Development {
        &self.contents[0].development
    }
    pub fn input(&self) -> &input::Recipe {
        &self.contents[0].input
    }
    pub fn optics(&self) -> &optics::Recipe {
        &self.contents[0].optics
    }
    pub fn minimum(&self) -> &density::Minimum {
        &self.contents[0].minimum
    }
    pub fn grain(&self) -> &grain::Recipe {
        &self.contents[0].grain
    }
    pub fn film_grain(&self) -> Option<&grain::film_v1::Recipe> {
        self.contents[0].film_grain.as_ref()
    }
    pub fn effects(&self) -> &effects::Recipe {
        &self.contents[0].effects
    }
    pub fn dir(&self) -> &dir::Recipe {
        &self.contents[0].dir
    }
    #[cfg(test)]
    pub(crate) fn film_v1_bounds(&self) -> Option<&density::Bounds> {
        self.contents[0].film_v1_bounds.as_ref()
    }
    pub fn exposure_tables(&self) -> &spectral::Tables {
        &self.contents[0].exposure_tables
    }
    /// Inverse spectral matrix borrowed from the prepared render state.
    ///
    /// A caller can read the matrix while the state remains in scope.
    ///
    /// ```
    /// use film_juicer_core::recipe::RenderState;
    ///
    /// fn first_coefficient(state: RenderState) -> f32 {
    ///     let matrix = state.s_inverse();
    ///     matrix[0]
    /// }
    /// ```
    ///
    /// ```compile_fail,E0505
    /// use film_juicer_core::recipe::RenderState;
    ///
    /// fn first_coefficient(state: RenderState) -> f32 {
    ///     let matrix = state.s_inverse();
    ///     drop(state);
    ///     matrix[0]
    /// }
    /// ```
    pub fn s_inverse(&self) -> &[f32; 9] {
        &self.contents[0].s_inverse
    }
    pub fn reconstruction(&self) -> &FilmReconstruction {
        &self.contents[0].reconstruction
    }
    pub fn input_hull(&self) -> Option<&InputHull> {
        self.contents[0].input_hull.as_deref()
    }
    pub fn reference(&self) -> &reference::Reference {
        &self.contents[0].reference
    }
    pub fn scanner(&self) -> &ScannerOutput {
        &self.contents[0].scanner
    }
    pub fn route(&self) -> &PreparedRoute {
        &self.contents[0].route
    }
    pub fn profile_route_hash(&self) -> u64 {
        self.contents[0].profile_route_hash
    }
    pub fn hash(&self) -> u64 {
        self.contents[0].hash
    }
    pub fn upload_core_hash(&self) -> u64 {
        self.contents[0].hash
    }
    pub fn scanner_hash(&self) -> u64 {
        self.contents[0].scanner_hash
    }
}
impl DirectState {
    pub fn correction(&self) -> &scanner::Correction {
        &self.correction
    }
}
impl PrintState {
    pub fn glare_hash(&self) -> u64 {
        self.glare_hash
    }
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
    pub(crate) fn prepare_raw(
        film_base_density: &[f32; 81],
        illuminant: &[f32; 81],
        print_sensitivity: &[[f32; 3]; 81],
    ) -> Result<print::PreflashRaw, PrepareError> {
        print::PreflashRaw::new(film_base_density, illuminant, print_sensitivity)
            .map_err(PrepareError::PrintPreflash)
    }

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
    pub fn tables(&self) -> &spectral::Tables {
        &self.tables
    }
    pub fn color(&self) -> &scanner::Color {
        &self.color
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
/// Complete one direct or print host state without frame-dependent grain.
pub fn prepare(assets: &Assets, params: &Params) -> Result<RenderState, PrepareError> {
    prepare_state(assets, params, None)
}

/// Sampling requested for this Film v1 render.
#[derive(Clone, Copy)]
pub enum FilmGrainFrame {
    Raster(grain::film_v1::Raster),
    Thumbnail,
}
pub fn prepare_for_frame(
    assets: &Assets,
    params: &Params,
    frame: FilmGrainFrame,
) -> Result<RenderState, PrepareError> {
    prepare_state(assets, params, Some(frame))
}

fn prepare_state(
    assets: &Assets,
    params: &Params,
    frame: Option<FilmGrainFrame>,
) -> Result<RenderState, PrepareError> {
    let profile = assets.film(&params.film_profile_key)?;
    let film_grain = film_stage::prepare_grain(assets, params, frame)?;
    let selected_print = if print_route(params.scan_route) {
        Some(assets.print(&params.print_profile_key)?)
    } else {
        None
    };
    let mut acquired = foundation::acquire_foundation(assets, params, profile)?;
    let film = acquired.development.profile();
    let calibration = selected_print
        .as_ref()
        .map(|print| {
            print_stage::acquire_calibration(
                assets,
                film,
                print,
                print_stage::enlarger_illuminant(params.enl_ill),
            )
        })
        .transpose()?;
    print_stage::route_prechecks(params, selected_print.as_deref(), calibration)?;
    let film_stage = film_stage::prepare(
        &acquired,
        params,
        selected_print.is_some(),
        film_grain.as_ref(),
    )?;
    let profile_route_hash = profile_route_hash(params, film, selected_print.as_deref());

    let bounds = scanner_stage::correction_bounds(
        &acquired,
        params,
        &film_stage,
        selected_print.as_deref(),
    )?;
    let (transform, gamut_hash) = scanner_stage::output_policy(params)?;
    let scanner_params = scanner_stage::scanner_params(params, transform.space(), bounds.params);
    let print = selected_print
        .zip(calibration)
        .map(|(profile, calibration)| print_stage::prepare(params, profile, calibration))
        .transpose()?;
    let exposure = film_stage::prepare_exposure(&mut acquired, &film_stage.input)?;
    let illumination = print
        .as_ref()
        .map(|print| print_stage::illuminate(&acquired.illuminants, print))
        .transpose()?;
    let reference =
        film_stage::prepare_reference(&acquired.development, &film_stage.input, &exposure)?;
    let film = acquired.development.profile();
    let print = print
        .zip(illumination)
        .map(|(print, illumination)| {
            let balance = print_stage::balance(film, &print, &illumination, &reference)?;
            Ok::<_, PrepareError>((print, illumination, balance))
        })
        .transpose()?;
    let scanner = scanner_stage::prepare(
        assets,
        params,
        &acquired,
        print.as_ref().map(|(print, _, _)| print.profile.as_ref()),
        transform,
        gamut_hash,
        scanner_params,
    )?;
    let route = match print {
        Some((print, illumination, balance)) => PreparedRoute::Print(print_stage::finish(
            params,
            film,
            print,
            illumination,
            balance,
            &scanner,
            &bounds,
        )?),
        None => PreparedRoute::Direct(scanner_stage::direct(
            &acquired, &reference, &scanner, &bounds,
        )?),
    };
    let viewing = match &route {
        PreparedRoute::Direct(_) => film.info().viewing_illuminant(),
        PreparedRoute::Print(print) => print.profile.info().viewing_illuminant(),
    };
    let (scanner, scanner_hash) = scanner_stage::finish(
        scanner,
        params.scan_route,
        film.info().polarity(),
        viewing,
        &reference,
    );
    let scanner_recipe_hash = scanner.hash;
    let film_stage::FilmStage {
        input,
        optics,
        minimum,
        grain,
        effects,
        dir,
        #[cfg(test)]
        film_v1_bounds,
        ..
    } = film_stage;
    let film_stage::ExposureStage {
        tables: exposure_tables,
        s_inverse,
        reconstruction,
    } = exposure;
    let mut hash = match (&route, dir.active()) {
        (PreparedRoute::Direct(_), false) => hash::u64s(&[
            profile_route_hash,
            input.hash(),
            acquired.development.hash(),
            scanner_recipe_hash,
        ]),
        (PreparedRoute::Direct(_), true) => hash::u64s(&[
            profile_route_hash,
            input.hash(),
            acquired.development.hash(),
            dir.hash(),
            scanner_recipe_hash,
        ]),
        (PreparedRoute::Print(print), false) => hash::u64s(&[
            profile_route_hash,
            input.hash(),
            acquired.development.hash(),
            scanner_recipe_hash,
            print.hash(),
        ]),
        (PreparedRoute::Print(print), true) => hash::u64s(&[
            profile_route_hash,
            input.hash(),
            acquired.development.hash(),
            dir.hash(),
            scanner_recipe_hash,
            print.hash(),
        ]),
    };
    for optional in [optics.hash(), grain.hash(), effects.hash()] {
        if optional != 0 {
            hash = hash::u64s(&[hash, optional]);
        }
    }
    if let Some(film_grain) = &film_grain {
        hash = hash::u64s(&[hash, film_grain.hash()]);
    }
    if hash == 0 || scanner_hash == 0 || scanner_recipe_hash == 0 || scanner.color.hash() == 0 {
        return Err(ControlError::Identity(params.scan_route).into());
    }
    let contents = Contents {
        film_key: params.film_profile_key.clone(),
        development: acquired.development,
        input,
        optics,
        minimum,
        grain,
        film_grain,
        effects,
        dir,
        #[cfg(test)]
        film_v1_bounds,
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
    };
    let mut backing = Vec::new();
    backing
        .try_reserve_exact(1)
        .map_err(PrepareError::ContentsAllocation)?;
    backing.push(contents);
    Ok(RenderState { contents: backing })
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::Path;
    use std::sync::Weak;

    use super::*;

    pub(crate) fn assets() -> Assets {
        Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"))
    }
    pub(crate) fn output_boundary_weak(state: &RenderState) -> Weak<crate::assets::OutputBoundary> {
        Arc::downgrade(state.scanner().gamut.boundary.as_ref().unwrap())
    }
    pub(crate) fn params(route: ScanRoute, method: i32) -> Params {
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
                ..HalationControls::default()
            },
            highlight_boost: optics::highlight_boost::Controls::default(),
            scanner_black_correction: 0,
            scanner_white_correction: 0,
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
            film_grain: grain::film_v1::Selection::Current,
            film_dust_amount: 0.0,
            film_scratch_amount: 0.0,
            gate_dust_amount: 0.0,
            gate_scratch_amount: 0.0,
            scan_route: route,
            glare_active: true,
            camera_filter_override: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_support::{assets, params};

    #[test]
    fn scanner_recipe_binds_the_raw_digest_after_the_profile_label() {
        let stock = assets();
        for (route, tags) in [
            (ScanRoute::NegativeDirect, [0_u8, 0, 0]),
            (ScanRoute::NegativePrint, [1_u8, 1, 0]),
            (ScanRoute::PositiveDirect, [2_u8, 0, 1]),
            (ScanRoute::PositivePrint, [3_u8, 1, 1]),
        ] {
            let state = prepare(&stock, &params(route, 1)).unwrap();
            let scanner = state.scanner();
            let (key, token, viewing) = match state.route() {
                PreparedRoute::Direct(_) => (
                    state.film_key(),
                    state.film_profile().asset_token(),
                    state.film_profile().info().viewing_illuminant(),
                ),
                PreparedRoute::Print(paper) => (
                    paper.key(),
                    paper.profile().asset_token(),
                    paper.profile().info().viewing_illuminant(),
                ),
            };
            let raw = scanner.color().spectral_binding_digest(
                tags[0],
                key,
                token,
                scanner.tables(),
                true,
            );
            let gamut = scanner.gamut().hash;
            assert_ne!(gamut, 0);
            let mut combined_stream = [0_u8; 16];
            combined_stream[..8].copy_from_slice(&raw.to_le_bytes());
            combined_stream[8..].copy_from_slice(&gamut.to_le_bytes());
            let combined = hash::bytes(&combined_stream);
            assert_eq!(state.scanner_hash(), combined);
            assert_ne!(combined, raw);

            let controls = scanner.params();
            let mut suffix = Vec::new();
            suffix.extend((controls.output_space as i32).to_le_bytes());
            suffix.extend([
                u8::from(controls.encode_cctf),
                u8::from(controls.black_correction),
                u8::from(controls.white_correction),
            ]);
            suffix.extend(controls.black_level.to_le_bytes());
            suffix.extend(controls.white_level.to_le_bytes());
            suffix.push(u8::from(controls.glare_active));
            for word in [
                controls.glare_percent,
                controls.glare_roughness,
                controls.glare_blur_sigma_px,
                controls.lens_blur_sigma_px,
                controls.unsharp_sigma_px,
                controls.unsharp_amount,
            ] {
                suffix.extend(word.to_le_bytes());
            }
            suffix.extend(state.reference().hash().to_le_bytes());
            suffix.extend(gamut.to_le_bytes());
            let label = viewing.as_bytes();
            let raw_bytes = raw.to_le_bytes();
            let recipe_bytes = [tags.as_slice(), label, &raw_bytes, &suffix].concat();
            let matches = |bytes: &[u8]| hash::bytes(bytes) == scanner.hash();
            assert!(matches(&recipe_bytes));

            let swapped = [tags.as_slice(), &raw_bytes, label, &suffix].concat();
            assert!(!matches(&swapped));
            let substituted = [tags.as_slice(), label, &combined.to_le_bytes(), &suffix].concat();
            assert!(!matches(&substituted));
            let mut mutated = recipe_bytes.clone();
            for index in [tags.len(), tags.len() + label.len()] {
                mutated[index] ^= 1;
                assert!(!matches(&mutated));
                mutated[index] ^= 1;
                assert!(matches(&mutated));
            }
        }
    }

    #[test]
    fn full_route_recipe_folds_the_surviving_owner_words() {
        let stock = assets();
        for (route, dir_enabled) in [
            (ScanRoute::NegativeDirect, true),
            (ScanRoute::NegativePrint, true),
            (ScanRoute::PositiveDirect, false),
            (ScanRoute::PositivePrint, false),
        ] {
            let mut controls = params(route, 1);
            controls.dir_couplers.active = dir_enabled;
            controls.grain.active = dir_enabled;
            assert_eq!(controls.camera_exposure_compensation_ev.to_bits(), 0);
            let mut identities = Vec::new();
            for exposure in [0.0, 0.25, 0.0] {
                controls.camera_exposure_compensation_ev = exposure;
                let state = prepare(&stock, &controls).unwrap();
                assert_eq!(state.dir().active(), dir_enabled);
                assert_eq!(state.grain().hash() != 0, dir_enabled);
                let mut base = Vec::new();
                base.extend(state.profile_route_hash().to_le_bytes());
                base.extend(state.input().hash().to_le_bytes());
                base.extend(state.development().hash().to_le_bytes());
                if dir_enabled {
                    base.extend(state.dir().hash().to_le_bytes());
                }
                base.extend(state.scanner().hash().to_le_bytes());
                if let PreparedRoute::Print(paper) = state.route() {
                    base.extend(paper.hash().to_le_bytes());
                }
                let fold = |bytes: &[u8]| {
                    let optional = [
                        state.optics().hash(),
                        state.grain().hash(),
                        state.effects().hash(),
                    ];
                    let result = optional.into_iter().filter(|&word| word != 0).fold(
                        hash::bytes(bytes),
                        |previous, next| {
                            let mut continuation = previous.to_le_bytes().to_vec();
                            continuation.extend(next.to_le_bytes());
                            hash::bytes(&continuation)
                        },
                    );
                    if let Some(film_grain) = state.film_grain() {
                        return hash::bytes(
                            &[result.to_le_bytes(), film_grain.hash().to_le_bytes()].concat(),
                        );
                    }
                    result
                };
                assert_eq!(fold(&base), state.hash());
                assert_eq!(state.upload_core_hash(), state.hash());
                for index in [0, 8, 16, 24] {
                    base[index] ^= 1;
                    assert_ne!(fold(&base), state.hash());
                    base[index] ^= 1;
                }
                assert_eq!(fold(&base), state.hash());
                identities.push(state.hash());
            }
            assert_ne!(identities[0], identities[1]);
            assert_eq!(identities[0], identities[2]);
        }
    }

    #[test]
    fn all_routes_and_methods_complete_with_active_spatial_dir() {
        let assets = assets();
        let mut count = 0;
        let routes = [
            ScanRoute::NegativeDirect,
            ScanRoute::NegativePrint,
            ScanRoute::PositiveDirect,
            ScanRoute::PositivePrint,
        ];
        let cases_ids: Vec<_> =
            IntoIterator::into_iter(routes.iter().flat_map(|route| {
                (0..3).map(move |method| format!("{}/{method}", route_key(*route)))
            }))
            .map(|id| AsRef::<str>::as_ref(&id).to_owned())
            .collect();
        assert!(!cases_ids.is_empty(), "recipe-route-methods");
        let cases_count = cases_ids.len();
        let mut cases_remaining = cases_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(cases_remaining.len(), cases_count, "recipe-route-methods");
        for route in routes {
            for method in 0..3 {
                let case =
                    AsRef::<str>::as_ref(&(format!("{}/{method}", route_key(route)))).to_owned();
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
                assert_ne!(state.scanner().hash(), 0);
                assert_ne!(state.scanner_hash(), 0);
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
                assert!(
                    cases_remaining.remove(&case),
                    "unexpected or repeated comparison: {case}"
                );
            }
        }
        assert_eq!(count, 12);
        assert!(
            cases_remaining.is_empty(),
            "missing comparisons: {cases_remaining:?}"
        );
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

    #[test]
    fn print_glare_identity_matches_reviewed_known_answers() {
        let mut params = scanner::PostEffectsParams {
            route: ScanRoute::NegativePrint,
            glare_active: true,
            glare_percent: f32::from_bits(0x3cf5_c28f),
            glare_roughness: f32::from_bits(0x3f33_3333),
            glare_blur_sigma_px: f32::from_bits(0x3f00_0000),
            lens_blur_sigma_px: 0.0,
            unsharp_sigma_px: 0.0,
            unsharp_amount: 0.0,
        };
        for (route, key, expected) in [
            (
                ScanRoute::NegativePrint,
                "kodak_portra_endura",
                0xf618_c9a2_7080_fb03,
            ),
            (
                ScanRoute::NegativePrint,
                "kodak_ultra_endura",
                0xf358_9058_de05_8d92,
            ),
            (
                ScanRoute::PositivePrint,
                "kodak_portra_endura",
                0xecf7_7afb_fba9_bf31,
            ),
            (
                ScanRoute::PositivePrint,
                "kodak_ultra_endura",
                0x0135_0cec_7669_ec74,
            ),
        ] {
            params.route = route;
            let original = scanner::describe_post_effects(params).unwrap();
            assert_eq!(print_glare_hash(route, key, &original), expected);

            let sharpened = scanner::describe_post_effects(scanner::PostEffectsParams {
                unsharp_sigma_px: 0.7,
                unsharp_amount: 0.7,
                ..params
            })
            .unwrap();
            assert_ne!(original.hash(), sharpened.hash());
            assert_eq!(print_glare_hash(route, key, &sharpened), expected);

            for inactive in [
                scanner::PostEffectsParams {
                    glare_active: false,
                    ..params
                },
                scanner::PostEffectsParams {
                    glare_percent: 0.0,
                    ..params
                },
            ] {
                let inactive = scanner::describe_post_effects(inactive).unwrap();
                assert_eq!(print_glare_hash(route, key, &inactive), 0);
            }
        }
    }
}
