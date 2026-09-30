//! Catalog discovery, authored profile sources and density-model sampling.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

const DEFAULT_FILM_KEY: &str = "kodak_portra_400";
const DEFAULT_PRINT_KEY: &str = "kodak_portra_endura";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    Film,
    Print,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Support {
    Film,
    Paper,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Filming,
    Printing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Polarity {
    Negative,
    Positive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileUse {
    Still,
    Cine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Antihalation {
    Strong,
    Weak,
    No,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelModel {
    Color,
    Bw,
}

#[derive(Debug)]
pub struct CatalogEntry {
    key: String,
    label: String,
    source_path: PathBuf,
    support: Support,
    stage: Stage,
    polarity: Polarity,
}

impl CatalogEntry {
    pub fn key(&self) -> &str {
        &self.key
    }
    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }
    pub fn support(&self) -> Support {
        self.support
    }
    pub fn stage(&self) -> Stage {
        self.stage
    }
    pub fn polarity(&self) -> Polarity {
        self.polarity
    }
}

#[derive(Debug)]
pub struct Catalog {
    films: Vec<CatalogEntry>,
    prints: Vec<CatalogEntry>,
    ignored: Vec<PathBuf>,
    unavailable: Vec<CatalogEntryError>,
}

impl Catalog {
    pub fn films(&self) -> &[CatalogEntry] {
        &self.films
    }
    pub fn prints(&self) -> &[CatalogEntry] {
        &self.prints
    }
    pub fn ignored(&self) -> &[PathBuf] {
        &self.ignored
    }
    pub fn unavailable(&self) -> &[CatalogEntryError] {
        &self.unavailable
    }
    pub fn film(&self, key: &str) -> Option<&CatalogEntry> {
        self.films.iter().find(|entry| entry.key == key)
    }
    pub fn print(&self, key: &str) -> Option<&CatalogEntry> {
        self.prints.iter().find(|entry| entry.key == key)
    }
    pub fn default_film(&self) -> &CatalogEntry {
        self.film(DEFAULT_FILM_KEY)
            .expect("catalog construction requires the default film")
    }
    pub fn default_print(&self) -> &CatalogEntry {
        self.print(DEFAULT_PRINT_KEY)
            .expect("catalog construction requires the default print")
    }
}

#[derive(Debug)]
pub enum CatalogError {
    Directory {
        path: PathBuf,
        source: io::Error,
    },
    DuplicateKey {
        path: PathBuf,
        role: Role,
        key: String,
    },
    EmptyRole {
        path: PathBuf,
        role: Role,
    },
    MissingDefault {
        path: PathBuf,
        role: Role,
        key: &'static str,
    },
}

#[derive(Debug)]
pub struct CatalogEntryError {
    pub path: PathBuf,
    pub kind: CatalogEntryErrorKind,
}

#[derive(Debug)]
pub enum CatalogEntryErrorKind {
    Read(io::Error),
    Json(serde_json::Error),
    MissingStock,
    UnsupportedRole,
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Directory { path, source } => {
                write!(f, "catalog directory {}: {source}", path.display())
            }
            Self::DuplicateKey { path, role, key } => write!(
                f,
                "duplicate {role:?} profile key {key} at {}",
                path.display()
            ),
            Self::EmptyRole { path, role } => {
                write!(f, "no visible {role:?} profiles in {}", path.display())
            }
            Self::MissingDefault { path, role, key } => write!(
                f,
                "missing default {role:?} profile {key} in {}",
                path.display()
            ),
        }
    }
}
impl std::error::Error for CatalogError {}

fn decode_json(bytes: &[u8]) -> Result<Value, serde_json::Error> {
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    serde_json::from_slice(bytes)
}

fn parse_support(value: &str) -> Option<Support> {
    match value {
        "film" => Some(Support::Film),
        "paper" => Some(Support::Paper),
        _ => None,
    }
}
fn parse_stage(value: &str) -> Option<Stage> {
    match value {
        "filming" => Some(Stage::Filming),
        "printing" => Some(Stage::Printing),
        _ => None,
    }
}
fn parse_polarity(value: &str) -> Option<Polarity> {
    match value {
        "negative" => Some(Polarity::Negative),
        "positive" => Some(Polarity::Positive),
        _ => None,
    }
}

fn load_catalog_entry(path: &Path) -> Result<Option<CatalogEntry>, CatalogEntryErrorKind> {
    let bytes = fs::read(path).map_err(CatalogEntryErrorKind::Read)?;
    let root = decode_json(&bytes).map_err(CatalogEntryErrorKind::Json)?;
    let shaped = root
        .get("data")
        .and_then(Value::as_object)
        .is_some_and(|data| {
            [
                "wavelengths",
                "log_sensitivity",
                "channel_density",
                "base_density",
                "log_exposure",
                "density_curves",
            ]
            .iter()
            .any(|key| data.contains_key(*key))
        });
    if !shaped {
        return Ok(None);
    }
    let info = &root["info"];
    let key = info["stock"]
        .as_str()
        .filter(|key| !key.is_empty())
        .ok_or(CatalogEntryErrorKind::MissingStock)?;
    let label = info["name"].as_str().unwrap_or(key);
    let member = |field, default| match info.get(field) {
        None => Some(default),
        Some(value) => value.as_str(),
    };
    let support = member("support", "film")
        .and_then(parse_support)
        .ok_or(CatalogEntryErrorKind::UnsupportedRole)?;
    let stage = member("stage", "filming")
        .and_then(parse_stage)
        .ok_or(CatalogEntryErrorKind::UnsupportedRole)?;
    let polarity = member("type", "negative")
        .and_then(parse_polarity)
        .ok_or(CatalogEntryErrorKind::UnsupportedRole)?;
    if support == Support::Paper && stage == Stage::Filming {
        return Err(CatalogEntryErrorKind::UnsupportedRole);
    }
    Ok(Some(CatalogEntry {
        key: key.to_owned(),
        label: label.to_owned(),
        source_path: path.to_owned(),
        support,
        stage,
        polarity,
    }))
}

/// Read the shallow `profiles` directory beneath a resource directory.
pub fn load_catalog(resource_dir: &Path) -> Result<Catalog, CatalogError> {
    let path = resource_dir.join("profiles");
    let directory_error = |source| CatalogError::Directory {
        path: path.clone(),
        source,
    };
    let entries = fs::read_dir(&path).map_err(directory_error)?;
    let mut catalog = Catalog {
        films: Vec::new(),
        prints: Vec::new(),
        ignored: vec![
            "Resources/filters/neutral_print_filters.json".into(),
            "Resources/profiles/enlarger_neutral*.json".into(),
        ],
        unavailable: Vec::new(),
    };
    let mut film_keys = HashSet::new();
    let mut print_keys = HashSet::new();
    for entry in entries {
        let entry = entry.map_err(directory_error)?;
        let source_path = entry.path();
        if !source_path.is_file() || source_path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        match load_catalog_entry(&source_path) {
            Ok(Some(entry)) => {
                let (role, keys, profiles) = match entry.stage {
                    Stage::Filming => (Role::Film, &mut film_keys, &mut catalog.films),
                    Stage::Printing => (Role::Print, &mut print_keys, &mut catalog.prints),
                };
                if !keys.insert(entry.key.clone()) {
                    return Err(CatalogError::DuplicateKey {
                        path: source_path,
                        role,
                        key: entry.key,
                    });
                }
                profiles.push(entry);
            }
            Ok(None) => catalog.ignored.push(source_path),
            Err(kind) => catalog.unavailable.push(CatalogEntryError {
                path: source_path,
                kind,
            }),
        }
    }
    for (role, entries) in [
        (Role::Film, &mut catalog.films),
        (Role::Print, &mut catalog.prints),
    ] {
        entries.sort_by(|a, b| a.label.cmp(&b.label).then(a.key.cmp(&b.key)));
        if entries.is_empty() {
            return Err(CatalogError::EmptyRole { path, role });
        }
    }
    for (role, keys, key) in [
        (Role::Film, &film_keys, DEFAULT_FILM_KEY),
        (Role::Print, &print_keys, DEFAULT_PRINT_KEY),
    ] {
        if !keys.contains(key) {
            return Err(CatalogError::MissingDefault { path, role, key });
        }
    }
    Ok(catalog)
}

#[derive(Debug)]
pub struct ProfileInfo {
    stock: String,
    name: String,
    support: Support,
    stage: Stage,
    polarity: Polarity,
    usage: ProfileUse,
    antihalation: Antihalation,
    channel_model: ChannelModel,
    reference_illuminant: String,
    viewing_illuminant: String,
}
impl ProfileInfo {
    pub fn stock(&self) -> &str {
        &self.stock
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn support(&self) -> Support {
        self.support
    }
    pub fn stage(&self) -> Stage {
        self.stage
    }
    pub fn polarity(&self) -> Polarity {
        self.polarity
    }
    pub fn usage(&self) -> ProfileUse {
        self.usage
    }
    pub fn antihalation(&self) -> Antihalation {
        self.antihalation
    }
    pub fn channel_model(&self) -> ChannelModel {
        self.channel_model
    }
    pub fn reference_illuminant(&self) -> &str {
        &self.reference_illuminant
    }
    pub fn viewing_illuminant(&self) -> &str {
        &self.viewing_illuminant
    }
}

#[derive(Debug)]
pub struct ProfileSamples {
    wavelengths: [f32; 81],
    log_sensitivity: [[f32; 3]; 81],
    channel_density: [[f32; 3]; 81],
    base_density: [f32; 81],
    log_exposure: Vec<f64>,
    hanatos2025_adaptation_window_params: Option<[f32; 4]>,
    hanatos2025_adaptation_surface_params: Option<[[f32; 15]; 3]>,
}
impl ProfileSamples {
    pub fn wavelengths(&self) -> &[f32; 81] {
        &self.wavelengths
    }
    pub fn log_sensitivity(&self) -> &[[f32; 3]; 81] {
        &self.log_sensitivity
    }
    pub fn channel_density(&self) -> &[[f32; 3]; 81] {
        &self.channel_density
    }
    pub fn base_density(&self) -> &[f32; 81] {
        &self.base_density
    }
    /// Authored doubles; construction also checks their eventual f32 axis.
    pub fn log_exposure(&self) -> &[f64] {
        &self.log_exposure
    }
    pub fn hanatos2025_adaptation_window_params(&self) -> Option<&[f32; 4]> {
        self.hanatos2025_adaptation_window_params.as_ref()
    }
    pub fn hanatos2025_adaptation_surface_params(&self) -> Option<&[[f32; 15]; 3]> {
        self.hanatos2025_adaptation_surface_params.as_ref()
    }
}

#[derive(Debug)]
pub struct DensityCurveModel {
    centers: [[f64; 3]; 3],
    amplitudes: [[f64; 3]; 3],
    sigmas: [[f64; 3]; 3],
}
impl DensityCurveModel {
    /// Store authored f64 coefficients in [channel][layer] order.
    /// The authored bits are retained, including signed zero and sub-f32 precision.
    pub fn new(centers: [[f64; 3]; 3], amplitudes: [[f64; 3]; 3], sigmas: [[f64; 3]; 3]) -> Self {
        Self {
            centers,
            amplitudes,
            sigmas,
        }
    }

    /// All coefficient matrices use [channel][layer] order.
    pub fn centers(&self) -> &[[f64; 3]; 3] {
        &self.centers
    }
    pub fn amplitudes(&self) -> &[[f64; 3]; 3] {
        &self.amplitudes
    }
    pub fn sigmas(&self) -> &[[f64; 3]; 3] {
        &self.sigmas
    }

    /// Evaluate one exposure, publishing totals and layers only if all are finite.
    pub fn sample(
        &self,
        polarity: Polarity,
        log_exposure: f64,
    ) -> Result<DensityCurveSample, DensitySampleError> {
        let exposure = log_exposure as f32;
        let sign = match polarity {
            Polarity::Negative => 1.0_f32,
            Polarity::Positive => -1.0_f32,
        };
        // Preserve ProfileAssets.cpp's FP32 literal, casts and reduction order.
        #[expect(
            clippy::excessive_precision,
            clippy::approx_constant,
            reason = "retain the qualified native FP32 literal and its rounding"
        )]
        const INVERSE_SQRT_TWO: f32 = 0.7071067811865475244;
        let mut sample = DensityCurveSample {
            total: [0.0; 3],
            layers: [[0.0; 3]; 3],
        };
        for channel in 0..3 {
            let mut total = 0.0_f32;
            for layer in 0..3 {
                let center = self.centers[channel][layer] as f32;
                let amplitude = self.amplitudes[channel][layer] as f32;
                let sigma = self.sigmas[channel][layer] as f32;
                let z = sign * (exposure - center) / sigma;
                let cdf = 0.5_f32 * libm::erfcf(-z * INVERSE_SQRT_TWO);
                let value = amplitude * cdf;
                if !value.is_finite() {
                    return Err(DensitySampleError::NonfiniteLayer { channel, layer });
                }
                sample.layers[layer][channel] = value;
                total += value;
            }
            if !total.is_finite() {
                return Err(DensitySampleError::NonfiniteTotal { channel });
            }
            sample.total[channel] = total;
        }
        Ok(sample)
    }
}

#[derive(Debug)]
pub struct DensityCurveSample {
    pub total: [f32; 3],
    /// Layer-major, then channel; the model coefficients have the opposite order.
    pub layers: [[f32; 3]; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensitySampleError {
    NonfiniteLayer { channel: usize, layer: usize },
    NonfiniteTotal { channel: usize },
}

impl fmt::Display for DensitySampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "density sample: {self:?}")
    }
}
impl std::error::Error for DensitySampleError {}

#[derive(Debug)]
pub struct ProfileSource {
    info: ProfileInfo,
    samples: ProfileSamples,
    density_model: DensityCurveModel,
}
impl ProfileSource {
    pub fn info(&self) -> &ProfileInfo {
        &self.info
    }
    pub fn samples(&self) -> &ProfileSamples {
        &self.samples
    }
    pub fn density_model(&self) -> &DensityCurveModel {
        &self.density_model
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Requirement {
    Object,
    String,
    MetadataValue,
    SelectedRole,
    Illuminant,
    ArrayLength(usize),
    FiniteNumber,
    CanonicalWavelength(u16),
    ExposureCount,
    NondecreasingExposure,
    FloatRange,
    ModelCoefficients,
    AdaptationArray,
}

#[derive(Debug)]
pub enum ProfileErrorKind {
    Read(io::Error),
    Json(serde_json::Error),
    Field {
        field: String,
        requirement: Requirement,
    },
}

#[derive(Debug)]
pub struct ProfileError {
    pub path: PathBuf,
    pub role: Role,
    pub stock: Option<String>,
    pub kind: ProfileErrorKind,
}
impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "profile {} role={:?} stock={:?}: ",
            self.path.display(),
            self.role,
            self.stock
        )?;
        match &self.kind {
            ProfileErrorKind::Read(source) => write!(f, "{source}"),
            ProfileErrorKind::Json(source) => write!(f, "{source}"),
            ProfileErrorKind::Field { field, requirement } => {
                write!(f, "{field} requires {requirement:?}")
            }
        }
    }
}
impl std::error::Error for ProfileError {}

// Private failures acquire path/role/stock once, at the file-loading boundary.
#[derive(Debug)]
struct FieldError {
    field: String,
    requirement: Requirement,
}
fn field_error(field: &str, requirement: Requirement) -> FieldError {
    FieldError {
        field: field.to_owned(),
        requirement,
    }
}
fn read_string<'a>(
    info: &'a Value,
    key: &str,
    default: Option<&'a str>,
) -> Result<&'a str, FieldError> {
    match info.get(key) {
        None => default,
        Some(member) => member.as_str(),
    }
    .ok_or_else(|| field_error(&format!("info.{key}"), Requirement::String))
}

fn normalize_illuminant(raw: &str) -> String {
    let mut normalized = String::new();
    for byte in raw.bytes() {
        if byte.is_ascii_alphanumeric() {
            normalized.push(char::from(byte.to_ascii_uppercase()));
        } else if (byte == b'-' || byte == b'_' || byte.is_ascii_whitespace())
            && !normalized.is_empty()
            && !normalized.ends_with('-')
        {
            normalized.push('-');
        }
    }
    normalized.truncate(normalized.trim_end_matches('-').len());
    normalized
}

fn blackbody_temperature(raw: &str) -> Option<f64> {
    let raw = raw.trim_matches(|character: char| character.is_ascii_whitespace());
    if !raw.get(..2)?.eq_ignore_ascii_case("BB") {
        return None;
    }
    // Numeric punctuation carries meaning and must bypass label normalization.
    raw[2..]
        .parse::<f64>()
        .ok()
        .filter(|temperature| temperature.is_finite() && *temperature > 0.0)
}

fn supported_illuminant(raw: &str) -> bool {
    let normalized = normalize_illuminant(raw);
    if matches!(
        normalized.as_str(),
        "D50"
            | "D55"
            | "D65"
            | "T"
            | "K75P"
            | "KINOTON75P"
            | "TH-KG3"
            | "THKG3"
            | "TH-KG3-L"
            | "THKG3L"
    ) {
        return true;
    }
    if let Some(digits) = normalized.strip_prefix('D') {
        return !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit());
    }
    blackbody_temperature(raw).is_some()
}

fn read_info(root: &Value, role: Role) -> Result<ProfileInfo, FieldError> {
    let info = &root["info"];
    if !info.is_object() {
        return Err(field_error("info", Requirement::Object));
    }
    let stock = read_string(info, "stock", None)?;
    let name = read_string(info, "name", Some(stock))?;
    let support = parse_support(read_string(info, "support", Some("film"))?)
        .ok_or_else(|| field_error("info.support", Requirement::MetadataValue))?;
    let stage = parse_stage(read_string(info, "stage", Some("filming"))?)
        .ok_or_else(|| field_error("info.stage", Requirement::MetadataValue))?;
    let polarity = parse_polarity(read_string(info, "type", Some("negative"))?)
        .ok_or_else(|| field_error("info.type", Requirement::MetadataValue))?;
    if role == Role::Film && (support != Support::Film || stage != Stage::Filming) {
        return Err(field_error("info", Requirement::SelectedRole));
    }
    if role == Role::Print && stage != Stage::Printing {
        return Err(field_error("info.stage", Requirement::SelectedRole));
    }
    let usage = match read_string(info, "use", Some("still"))? {
        "still" => ProfileUse::Still,
        "cine" => ProfileUse::Cine,
        _ => return Err(field_error("info.use", Requirement::MetadataValue)),
    };
    let antihalation = match read_string(info, "antihalation", Some("weak"))? {
        "strong" => Antihalation::Strong,
        "weak" => Antihalation::Weak,
        "no" => Antihalation::No,
        _ => return Err(field_error("info.antihalation", Requirement::MetadataValue)),
    };
    let channel_model = match read_string(info, "channel_model", Some("color"))? {
        "color" => ChannelModel::Color,
        "bw" => ChannelModel::Bw,
        _ => {
            return Err(field_error(
                "info.channel_model",
                Requirement::MetadataValue,
            ));
        }
    };
    let reference_illuminant = read_string(info, "reference_illuminant", Some("D55"))?;
    if !supported_illuminant(reference_illuminant) {
        return Err(field_error(
            "info.reference_illuminant",
            Requirement::Illuminant,
        ));
    }
    let viewing_illuminant = read_string(info, "viewing_illuminant", Some("D50"))?;
    if !supported_illuminant(viewing_illuminant) {
        return Err(field_error(
            "info.viewing_illuminant",
            Requirement::Illuminant,
        ));
    }
    Ok(ProfileInfo {
        stock: stock.to_owned(),
        name: name.to_owned(),
        support,
        stage,
        polarity,
        usage,
        antihalation,
        channel_model,
        reference_illuminant: reference_illuminant.to_owned(),
        viewing_illuminant: viewing_illuminant.to_owned(),
    })
}

fn read_array<'a>(node: &'a Value, len: usize, field: &str) -> Result<&'a [Value], FieldError> {
    node.as_array()
        .filter(|array| array.len() == len)
        .map(Vec::as_slice)
        .ok_or_else(|| field_error(field, Requirement::ArrayLength(len)))
}

fn read_sample(node: &Value, nullable: bool, field: &str) -> Result<f32, FieldError> {
    if nullable && node.is_null() {
        return Ok(f32::NAN);
    }
    // decode_json establishes finite f64 tokens. These authored scientific
    // samples intentionally do not reject overflow introduced by narrowing.
    node.as_f64()
        .map(|raw| raw as f32)
        .ok_or_else(|| field_error(field, Requirement::FiniteNumber))
}
fn read_sample_array<const N: usize>(
    node: &Value,
    nullable: bool,
    field: &str,
) -> Result<[f32; N], FieldError> {
    let rows = read_array(node, N, field)?;
    let mut samples = [0.0; N];
    for (index, (sample_out, node)) in samples.iter_mut().zip(rows).enumerate() {
        *sample_out = read_sample(node, nullable, &format!("{field}[{index}]"))?;
    }
    Ok(samples)
}
fn read_sample_matrix<const R: usize, const C: usize>(
    node: &Value,
    nullable: bool,
    field: &str,
) -> Result<[[f32; C]; R], FieldError> {
    let rows = read_array(node, R, field)?;
    let mut samples = [[0.0; C]; R];
    for (index, (samples_out, row)) in samples.iter_mut().zip(rows).enumerate() {
        *samples_out = read_sample_array(row, nullable, &format!("{field}[{index}]"))?;
    }
    Ok(samples)
}

#[derive(Deserialize)]
struct DensityModelInput {
    centers: [[f64; 3]; 3],
    amplitudes: [[f64; 3]; 3],
    sigmas: [[f64; 3]; 3],
}
fn read_model(data: &Value) -> Result<DensityCurveModel, FieldError> {
    let field = "data.density_curves_model";
    let node = &data["density_curves_model"];
    if !node.is_object() {
        return Err(field_error(field, Requirement::Object));
    }
    let DensityModelInput {
        centers,
        amplitudes,
        sigmas,
    } = DensityModelInput::deserialize(node)
        .map_err(|_| field_error(field, Requirement::ModelCoefficients))?;
    Ok(DensityCurveModel::new(centers, amplitudes, sigmas))
}

fn exposure_count_supported(count: usize) -> bool {
    count > 0 && count <= i32::MAX as usize
}
fn read_log_exposure(data: &Value) -> Result<Vec<f64>, FieldError> {
    let field = "data.log_exposure";
    let rows = data["log_exposure"]
        .as_array()
        .filter(|rows| exposure_count_supported(rows.len()))
        .ok_or_else(|| field_error(field, Requirement::ExposureCount))?;
    let mut exposure = Vec::with_capacity(rows.len());
    for (index, node) in rows.iter().enumerate() {
        let field = format!("{field}[{index}]");
        let raw = node
            .as_f64()
            .ok_or_else(|| field_error(&field, Requirement::FiniteNumber))?;
        if exposure.last().is_some_and(|&previous| raw < previous) {
            return Err(field_error(&field, Requirement::NondecreasingExposure));
        }
        exposure.push(raw);
    }
    Ok(exposure)
}

fn read_adaptation<'a>(
    data: &'a Value,
    key: &str,
    role: Role,
) -> Result<Option<&'a Value>, FieldError> {
    match data.get(key) {
        None => Ok(None),
        Some(Value::Null) if role == Role::Print => Ok(None),
        Some(Value::Array(array)) if array.is_empty() => Ok(None),
        Some(node @ Value::Array(_)) => Ok(Some(node)),
        Some(_) => Err(field_error(
            &format!("data.{key}"),
            Requirement::AdaptationArray,
        )),
    }
}

fn read_source(root: &Value, role: Role) -> Result<ProfileSource, FieldError> {
    let info = read_info(root, role)?;
    let data = &root["data"];
    if !data.is_object() {
        return Err(field_error("data", Requirement::Object));
    }
    let log_exposure = read_log_exposure(data)?;
    let density_model = read_model(data)?;
    let wavelengths = read_sample_array::<81>(&data["wavelengths"], false, "data.wavelengths")?;
    for (index, &wavelength) in wavelengths.iter().enumerate() {
        let expected = 380 + index as u16 * 5;
        if wavelength != f32::from(expected) {
            return Err(field_error(
                &format!("data.wavelengths[{index}]"),
                Requirement::CanonicalWavelength(expected),
            ));
        }
    }
    let log_sensitivity =
        read_sample_matrix(&data["log_sensitivity"], true, "data.log_sensitivity")?;
    let channel_density =
        read_sample_matrix(&data["channel_density"], true, "data.channel_density")?;
    let base_density = read_sample_array(&data["base_density"], true, "data.base_density")?;
    for (index, &raw) in log_exposure.iter().enumerate() {
        let narrowed = raw as f32;
        if !narrowed.is_finite() {
            return Err(field_error(
                &format!("data.log_exposure[{index}]"),
                Requirement::FloatRange,
            ));
        }
        // Finite f64 -> f32 rounding is monotone; the source ordering check
        // therefore also establishes the eventual sampled f32 axis ordering.
    }
    let window_key = "hanatos2025_adaptation_window_params";
    let surface_key = "hanatos2025_adaptation_surface_params";
    let hanatos2025_adaptation_window_params = read_adaptation(data, window_key, role)?
        .map(|node| read_sample_array(node, false, &format!("data.{window_key}")))
        .transpose()?;
    let hanatos2025_adaptation_surface_params = read_adaptation(data, surface_key, role)?
        .map(|node| read_sample_matrix(node, false, &format!("data.{surface_key}")))
        .transpose()?;
    Ok(ProfileSource {
        info,
        samples: ProfileSamples {
            wavelengths,
            log_sensitivity,
            channel_density,
            base_density,
            log_exposure,
            hanatos2025_adaptation_window_params,
            hanatos2025_adaptation_surface_params,
        },
        density_model,
    })
}

fn load_source(path: &Path, role: Role) -> Result<ProfileSource, ProfileError> {
    let file_error = |kind| ProfileError {
        path: path.to_owned(),
        role,
        stock: None,
        kind,
    };
    let bytes = fs::read(path).map_err(|error| file_error(ProfileErrorKind::Read(error)))?;
    let root = decode_json(&bytes).map_err(|error| file_error(ProfileErrorKind::Json(error)))?;
    read_source(&root, role).map_err(|error| ProfileError {
        path: path.to_owned(),
        role,
        stock: root["info"]["stock"].as_str().map(str::to_owned),
        kind: ProfileErrorKind::Field {
            field: error.field,
            requirement: error.requirement,
        },
    })
}

pub fn load_film_source(path: &Path) -> Result<ProfileSource, ProfileError> {
    load_source(path, Role::Film)
}
pub fn load_print_source(path: &Path) -> Result<ProfileSource, ProfileError> {
    load_source(path, Role::Print)
}

#[cfg(test)]
mod tests {
    use super::{blackbody_temperature, exposure_count_supported};

    #[test]
    fn preserves_blackbody_decimal_value() {
        for (authored, kelvin) in [
            ("BB3200.5", 3200.5_f64),
            (" bb+3200.5\t", 3200.5),
            ("Bb3.2e+3", 3200.0),
            ("BB1.5e-3", 0.0015),
            ("BB1e-0", 1.0),
        ] {
            assert_eq!(
                blackbody_temperature(authored).unwrap().to_bits(),
                kelvin.to_bits()
            );
        }
    }

    #[test]
    fn enforces_exposure_count_bounds() {
        assert!(!exposure_count_supported(0));
        assert!(exposure_count_supported(1));
        assert!(exposure_count_supported(i32::MAX as usize));
        assert!(!exposure_count_supported(i32::MAX as usize + 1));
    }
}
