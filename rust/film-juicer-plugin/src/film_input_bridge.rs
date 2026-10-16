//! SF_TEMP_BRIDGE_FILM_INPUT: synchronous fixed values; remove S4.E.

use std::fmt::{self, Write};
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::{color, exposure, film::input};
use crate::asset_profile::FilmOwner;
use crate::film_bridge::FjFilmProfile;
use crate::cuda::sys::{
    FjErrorBuffer, FjFloatSpan, FjStatus, FJ_API_NONE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE,
};
use crate::legacy_bridge::{self, FjInputColorConversion};
use crate::exposure_bridge::FjSensitivityFailure;

pub(crate) const FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE: u32 = 0;
pub(crate) const FJ_FILM_INPUT_ILLUMINANT_D65: u32 = 1;
pub(crate) const FJ_FILM_INPUT_ORIGIN_NONE: u32 = 0;
pub(crate) const FJ_FILM_INPUT_ORIGIN_RAW: u32 = 1;
pub(crate) const FJ_FILM_INPUT_ORIGIN_RESOURCE: u32 = 2;
pub(crate) const FJ_FILM_INPUT_ORIGIN_COLOR: u32 = 3;
pub(crate) const FJ_FILM_INPUT_ORIGIN_SENSITIVITY: u32 = 4;
pub(crate) const FJ_FILM_INPUT_ORIGIN_IDENTITY: u32 = 5;
pub(crate) const FJ_FILM_INPUT_ORIGIN_INTERNAL: u32 = 6;
pub(crate) const FJ_FILM_INPUT_REASON_NONE: u32 = 0;
pub(crate) const FJ_FILM_INPUT_RAW_RECORDS: u32 = 1;
pub(crate) const FJ_FILM_INPUT_RAW_DIAGNOSTIC: u32 = 2;
pub(crate) const FJ_FILM_INPUT_RAW_INPUT_SPACE: u32 = 3;
pub(crate) const FJ_FILM_INPUT_RAW_METHOD: u32 = 4;
pub(crate) const FJ_FILM_INPUT_RAW_METER: u32 = 5;
pub(crate) const FJ_FILM_INPUT_RAW_FLAGS: u32 = 6;
pub(crate) const FJ_FILM_INPUT_RAW_SPANS: u32 = 7;
pub(crate) const FJ_FILM_INPUT_RESOURCE_HANATOS_WINDOW: u32 = 1;
pub(crate) const FJ_FILM_INPUT_RESOURCE_HANATOS_SURFACE: u32 = 2;
pub(crate) const FJ_FILM_INPUT_RESOURCE_HANATOS_BLUR: u32 = 3;
pub(crate) const FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE: u32 = 4;
pub(crate) const FJ_FILM_INPUT_RESOURCE_TC_SOURCE: u32 = 5;
pub(crate) const FJ_FILM_INPUT_RESOURCE_INPUT_HULL: u32 = 6;
pub(crate) const FJ_FILM_INPUT_COLOR_MATRICES: u32 = 1;
pub(crate) const FJ_FILM_INPUT_COLOR_CAT16: u32 = 2;
pub(crate) const FJ_FILM_INPUT_COLOR_CAT02: u32 = 3;
pub(crate) const FJ_FILM_INPUT_SENSITIVITY_COMPUTED: u32 = 1;
pub(crate) const FJ_FILM_INPUT_SENSITIVITY_BOUNDARY: u32 = 2;
pub(crate) const FJ_FILM_INPUT_IDENTITY_REQUIRED: u32 = 1;
pub(crate) const FJ_FILM_INPUT_INTERNAL_PANIC: u32 = 1;
pub(crate) const FJ_FILM_INPUT_INTERNAL_INTEGRITY: u32 = 2;
pub(crate) const FJ_FILM_INPUT_INTERNAL_MATRICES_PANIC: u32 = 3;
pub(crate) const FJ_FILM_INPUT_INTERNAL_CAT16_PANIC: u32 = 4;
pub(crate) const FJ_FILM_INPUT_INTERNAL_CAT02_PANIC: u32 = 5;
pub(crate) const FJ_FILM_INPUT_INTERNAL_SENSITIVITY_PANIC: u32 = 6;

#[repr(C)]
pub(crate) struct FjFilmInputParams {
    pub input_space: u32,
    pub decode_cctf: u32,
    pub method: u32,
    pub auto_exposure_enabled: u32,
    pub meter: u32,
    pub manual_exposure_ev: f32,
    pub film_format_long_edge_mm: f32,
    pub filter_override: u32,
    pub uv: [f64; 3],
    pub ir: [f64; 3],
    pub apply_hanatos_window: u32,
    pub apply_hanatos_surface: u32,
    pub input_compression: u32,
}

#[repr(C)]
pub(crate) struct FjFilmInputResources {
    pub reference_illuminant: FjFloatSpan,
    pub reference_white_xyz: FjFloatSpan,
    pub integrated_d65_xyz: FjFloatSpan,
    pub reconstructed_reference_white: FjFloatSpan,
    pub tc_source_hash: u64,
    pub input_hull_hash: u64,
}

#[repr(C)]
pub(crate) struct FjFilmInput {
    pub color: FjInputColorConversion,
    pub xyz_to_linear_srgb: [f32; 9],
    pub nominal_white_xyz: [f32; 3],
    pub projection_white_xyz: [f32; 3],
    pub method: u32,
    pub auto_exposure_enabled: u32,
    pub meter: u32,
    pub manual_exposure_ev: f32,
    pub film_format_long_edge_mm: f32,
    pub band_pass_active: u32,
    pub uv: [f32; 3],
    pub ir: [f32; 3],
    pub apply_hanatos_window: u32,
    pub apply_hanatos_surface: u32,
    pub hanatos_spectral_blur: f32,
    pub hanatos_window: [f32; 4],
    pub hanatos_surface_rgb: [[f32; 15]; 3],
    pub projection_illuminant: u32,
    pub input_compression_active: u32,
    pub input_hull_hash: u64,
    pub tc_source_hash: u64,
    pub sensitivity_rgb: [[f32; 3]; 81],
    pub sensitivity_hash: u64,
    pub recipe_mallett_green_scale: f32,
    pub tc_hash: u64,
    pub hash: u64,
}

#[repr(C)]
pub(crate) struct FjFilmInputFailure {
    pub origin: u32,
    pub reason: u32,
    pub sensitivity: FjSensitivityFailure,
}

impl FjFilmInput {
    pub(crate) fn empty() -> Self {
        Self {
            color: FjInputColorConversion {
                input_space: 0,
                decode_cctf: 0,
                adapt_xyz: 0,
                rgb_to_xyz: [0.0; 9],
                xyz_adapt: [0.0; 9],
            },
            xyz_to_linear_srgb: [0.0; 9],
            nominal_white_xyz: [0.0; 3],
            projection_white_xyz: [0.0; 3],
            method: 0,
            auto_exposure_enabled: 0,
            meter: 0,
            manual_exposure_ev: 0.0,
            film_format_long_edge_mm: 0.0,
            band_pass_active: 0,
            uv: [0.0; 3],
            ir: [0.0; 3],
            apply_hanatos_window: 0,
            apply_hanatos_surface: 0,
            hanatos_spectral_blur: 0.0,
            hanatos_window: [0.0; 4],
            hanatos_surface_rgb: [[0.0; 15]; 3],
            projection_illuminant: 0,
            input_compression_active: 0,
            input_hull_hash: 0,
            tc_source_hash: 0,
            sensitivity_rgb: [[0.0; 3]; 81],
            sensitivity_hash: 0,
            recipe_mallett_green_scale: 0.0,
            tc_hash: 0,
            hash: 0,
        }
    }
    pub(crate) fn from_recipe(recipe: &input::Recipe) -> Self {
        let color = recipe.color();
        Self {
            color: FjInputColorConversion {
                input_space: match color.space() {
                    color::InputSpace::DavinciWideGamut => 0,
                    color::InputSpace::Bt2020 => 1,
                    color::InputSpace::Aces2065_1 => 2,
                    color::InputSpace::SrgbRec709 => 3,
                },
                decode_cctf: u32::from(color.decode_cctf()),
                adapt_xyz: 1,
                rgb_to_xyz: *color.rgb_to_xyz(),
                xyz_adapt: *color.xyz_adapt(),
            },
            xyz_to_linear_srgb: *color.xyz_to_linear_srgb(),
            nominal_white_xyz: *color.nominal_white_xyz(),
            projection_white_xyz: *color.projection_white_xyz(),
            method: match recipe.method() {
                input::Method::Hanatos2025 => 0,
                input::Method::Mallett2019 => 1,
                input::Method::Arctic2026Beta04 => 2,
            },
            auto_exposure_enabled: u32::from(recipe.auto_exposure_enabled()),
            meter: match recipe.meter() {
                input::Meter::CenterWeighted => 0,
                input::Meter::Average => 1,
                input::Meter::Median => 2,
                input::Meter::Partial => 3,
                input::Meter::Matrix => 4,
                input::Meter::MultiZone => 5,
                input::Meter::HighlightWeighted => 6,
            },
            manual_exposure_ev: recipe.manual_exposure_ev(),
            film_format_long_edge_mm: recipe.film_format_long_edge_mm(),
            band_pass_active: u32::from(recipe.band_pass_active()),
            uv: *recipe.uv(),
            ir: *recipe.ir(),
            apply_hanatos_window: u32::from(recipe.apply_hanatos_window()),
            apply_hanatos_surface: u32::from(recipe.apply_hanatos_surface()),
            hanatos_spectral_blur: recipe.hanatos_spectral_blur(),
            hanatos_window: *recipe.hanatos_window(),
            hanatos_surface_rgb: *recipe.hanatos_surface_rgb(),
            projection_illuminant: match recipe.projection_illuminant() {
                input::ProjectionIlluminant::ProfileReference => {
                    FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE
                }
                input::ProjectionIlluminant::D65 => FJ_FILM_INPUT_ILLUMINANT_D65,
            },
            input_compression_active: u32::from(recipe.input_compression_active()),
            input_hull_hash: recipe.input_hull_hash(),
            tc_source_hash: recipe.tc_source_hash(),
            sensitivity_rgb: *recipe.sensitivity().values_rgb(),
            sensitivity_hash: recipe.sensitivity().hash(),
            recipe_mallett_green_scale: recipe.sensitivity().mallett_green_scale(),
            tc_hash: recipe.tc_hash(),
            hash: recipe.hash(),
        }
    }
}
fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}
fn valid_extent<T>(pointer: *const T, count: usize) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && count.checked_mul(size_of::<T>()).is_some_and(|bytes| {
            bytes <= isize::MAX as usize && (pointer as usize).checked_add(bytes).is_some()
        })
}
fn valid_span<const N: usize>(span: &FjFloatSpan, optional: bool) -> bool {
    (optional && span.data.is_null() && span.count == 0)
        || (span.count == N && valid_extent(span.data, N))
}
struct Diagnostic {
    record: *mut FjErrorBuffer,
    data: *mut u8,
    capacity: usize,
    length: usize,
}
impl Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.capacity != 0 {
            let count = text.len().min(self.capacity - 1 - self.length);
            // SAFETY: Admission established exclusive backing; bounded copies exclude its trailing NUL.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data.add(self.length), count);
                self.length += count;
                self.data.add(self.length).write(0);
            }
        }
        if !self.record.is_null() {
            // SAFETY: This authorized exclusive record remains live until return.
            unsafe {
                (*self.record).length = self.length;
            }
        }
        Ok(())
    }
}
impl Diagnostic {
    /// # Safety
    /// Numerically valid records/backing have the caller's exclusive actual extent.
    unsafe fn admit(record: *mut FjErrorBuffer) -> Result<Self, ()> {
        let mut writer = Self {
            record,
            data: std::ptr::null_mut(),
            capacity: 0,
            length: 0,
        };
        if record.is_null() {
            return Ok(writer);
        }
        if !valid_extent(record, 1) {
            return Err(());
        }
        // SAFETY: The caller authorizes initialized capacity and writable length. At capacity zero data is not read.
        unsafe {
            (*record).length = 0;
            writer.capacity = (*record).capacity;
            if writer.capacity != 0 {
                let data = (*record).data.cast::<u8>();
                if !valid_extent(data, writer.capacity) {
                    return Err(());
                }
                writer.data = data;
                data.write(0);
            }
        }
        Ok(writer)
    }
}
impl FjFilmInputFailure {
    fn empty() -> Self {
        Self {
            origin: FJ_FILM_INPUT_ORIGIN_NONE,
            reason: FJ_FILM_INPUT_REASON_NONE,
            sensitivity: FjSensitivityFailure {
                hash_failure: 0,
                hash_sample_index: 0,
            },
        }
    }
    fn raw(reason: u32) -> Self {
        Self {
            origin: FJ_FILM_INPUT_ORIGIN_RAW,
            reason,
            ..Self::empty()
        }
    }
    fn from_error(error: &input::Error) -> Self {
        use input::{Error, Operation};
        let (origin, reason) = match error {
            #[cfg(feature = "test-support")]
            Error::Boundary(operation, _) => match operation {
                Operation::InputMatrices => {
                    (FJ_FILM_INPUT_ORIGIN_COLOR, FJ_FILM_INPUT_COLOR_MATRICES)
                }
                Operation::Cat16 => (FJ_FILM_INPUT_ORIGIN_COLOR, FJ_FILM_INPUT_COLOR_CAT16),
                Operation::Cat02 => (FJ_FILM_INPUT_ORIGIN_COLOR, FJ_FILM_INPUT_COLOR_CAT02),
                Operation::Sensitivity => (
                    FJ_FILM_INPUT_ORIGIN_SENSITIVITY,
                    FJ_FILM_INPUT_SENSITIVITY_BOUNDARY,
                ),
            },
            Error::MissingHanatosWindow => (
                FJ_FILM_INPUT_ORIGIN_RESOURCE,
                FJ_FILM_INPUT_RESOURCE_HANATOS_WINDOW,
            ),
            Error::MissingHanatosSurface => (
                FJ_FILM_INPUT_ORIGIN_RESOURCE,
                FJ_FILM_INPUT_RESOURCE_HANATOS_SURFACE,
            ),
            Error::InvalidHanatosBlur => (
                FJ_FILM_INPUT_ORIGIN_RESOURCE,
                FJ_FILM_INPUT_RESOURCE_HANATOS_BLUR,
            ),
            Error::MissingProjectionWhite => (
                FJ_FILM_INPUT_ORIGIN_RESOURCE,
                FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE,
            ),
            Error::MissingTcSource(_) => (
                FJ_FILM_INPUT_ORIGIN_RESOURCE,
                FJ_FILM_INPUT_RESOURCE_TC_SOURCE,
            ),
            Error::MissingInputHull => (
                FJ_FILM_INPUT_ORIGIN_RESOURCE,
                FJ_FILM_INPUT_RESOURCE_INPUT_HULL,
            ),
            Error::Sensitivity(_) => (
                FJ_FILM_INPUT_ORIGIN_SENSITIVITY,
                FJ_FILM_INPUT_SENSITIVITY_COMPUTED,
            ),
            Error::ZeroIdentity => (
                FJ_FILM_INPUT_ORIGIN_IDENTITY,
                FJ_FILM_INPUT_IDENTITY_REQUIRED,
            ),
            Error::Panic(operation) => (
                FJ_FILM_INPUT_ORIGIN_INTERNAL,
                match operation {
                    Operation::InputMatrices => FJ_FILM_INPUT_INTERNAL_MATRICES_PANIC,
                    Operation::Cat16 => FJ_FILM_INPUT_INTERNAL_CAT16_PANIC,
                    Operation::Cat02 => FJ_FILM_INPUT_INTERNAL_CAT02_PANIC,
                    Operation::Sensitivity => FJ_FILM_INPUT_INTERNAL_SENSITIVITY_PANIC,
                },
            ),
        };
        let sensitivity = if let Error::Sensitivity(error) = error {
            match error.hash_failure() {
                Some(exposure::HashFailure::NonfiniteOperand(index)) => FjSensitivityFailure {
                    hash_failure: 1,
                    hash_sample_index: index as u32,
                },
                Some(exposure::HashFailure::ZeroIdentity) => FjSensitivityFailure {
                    hash_failure: 2,
                    hash_sample_index: 0,
                },
                None => FjSensitivityFailure::default(),
            }
        } else {
            FjSensitivityFailure::default()
        };
        Self {
            origin,
            reason,
            sensitivity,
        }
    }
}
/// # Safety
/// Required live initialized records and present spans remain immutable until return; profile is a retained FilmOwner.
/// Outputs and diagnostics/backing are exclusive, mutually disjoint and disjoint from every input/reachable profile allocation.
/// Numeric checks cannot prove actual extent, disjointness or lifetime. No pointer is retained.
pub(crate) unsafe fn prepare(
    profile: *const FilmOwner,
    params: *const FjFilmInputParams,
    resources: *const FjFilmInputResources,
    out: *mut FjFilmInput,
    failure: *mut FjFilmInputFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let mut valid_out = false;
    let mut valid_failure = false;
    let result = catch_unwind(AssertUnwindSafe(|| {
        valid_out = valid_extent(out, 1);
        valid_failure = valid_extent(failure, 1);
        // SAFETY: Each independently authorized output is cleared without inspecting its incoming fields or padding.
        unsafe {
            if valid_out {
                out.write(FjFilmInput::empty());
            }
            if valid_failure {
                failure.write(FjFilmInputFailure::empty());
            }
        }
        // SAFETY: The diagnostic record and backing follow the same exclusive call-local contract.
        let diagnostic = unsafe { Diagnostic::admit(error) };
        let raw = if diagnostic.is_err() {
            Some(FJ_FILM_INPUT_RAW_DIAGNOSTIC)
        } else if !valid_out
            || !valid_failure
            || !valid_extent(profile, 1)
            || !valid_extent(params, 1)
            || !valid_extent(resources, 1)
        {
            Some(FJ_FILM_INPUT_RAW_RECORDS)
        } else {
            None
        };
        let mut diagnostic = diagnostic.ok();
        let reject = |reason, diagnostic: &mut Option<Diagnostic>| {
            if valid_failure {
                // SAFETY: Only independently authorized failure storage is written.
                unsafe {
                    failure.write(FjFilmInputFailure::raw(reason));
                }
            }
            if let Some(diagnostic) = diagnostic {
                let _ = diagnostic.write_str("MalformedFilmInput boundary=raw");
            }
            status(FJ_STATUS_UNSUPPORTED_INPUT)
        };
        if let Some(reason) = raw {
            return reject(reason, &mut diagnostic);
        }
        // SAFETY: All complete initialized input record extents passed numeric admission and remain immutable.
        let (params, resources) = unsafe { (&*params, &*resources) };
        let Some(space) = legacy_bridge::input_space(params.input_space) else {
            return reject(FJ_FILM_INPUT_RAW_INPUT_SPACE, &mut diagnostic);
        };
        let method = match params.method {
            0 => input::Method::Hanatos2025,
            1 => input::Method::Mallett2019,
            2 => input::Method::Arctic2026Beta04,
            _ => return reject(FJ_FILM_INPUT_RAW_METHOD, &mut diagnostic),
        };
        let meter = match params.meter {
            0 => input::Meter::CenterWeighted,
            1 => input::Meter::Average,
            2 => input::Meter::Median,
            3 => input::Meter::Partial,
            4 => input::Meter::Matrix,
            5 => input::Meter::MultiZone,
            6 => input::Meter::HighlightWeighted,
            _ => return reject(FJ_FILM_INPUT_RAW_METER, &mut diagnostic),
        };
        if [
            params.decode_cctf,
            params.auto_exposure_enabled,
            params.filter_override,
            params.apply_hanatos_window,
            params.apply_hanatos_surface,
            params.input_compression,
        ]
        .iter()
        .any(|&flag| flag > 1)
        {
            return reject(FJ_FILM_INPUT_RAW_FLAGS, &mut diagnostic);
        }
        if !valid_span::<81>(&resources.reference_illuminant, false)
            || !valid_span::<3>(&resources.reference_white_xyz, true)
            || !valid_span::<3>(&resources.integrated_d65_xyz, true)
            || !valid_span::<81>(&resources.reconstructed_reference_white, true)
        {
            return reject(FJ_FILM_INPUT_RAW_SPANS, &mut diagnostic);
        }
        // SAFETY: Checked fixed spans have complete initialized immutable extents; optional absent pointers are never borrowed.
        let resources = unsafe {
            input::Resources {
                reference_illuminant: &*resources.reference_illuminant.data.cast::<[f32; 81]>(),
                reference_white_xyz: resources
                    .reference_white_xyz
                    .data
                    .cast::<[f32; 3]>()
                    .as_ref(),
                integrated_d65_xyz: resources
                    .integrated_d65_xyz
                    .data
                    .cast::<[f32; 3]>()
                    .as_ref(),
                reconstructed_reference_white: resources
                    .reconstructed_reference_white
                    .data
                    .cast::<[f32; 81]>()
                    .as_ref(),
                tc_source_hash: resources.tc_source_hash,
                input_hull_hash: resources.input_hull_hash,
            }
        };
        let params = input::Params {
            input_space: space,
            decode_cctf: params.decode_cctf != 0,
            method,
            auto_exposure_enabled: params.auto_exposure_enabled != 0,
            meter,
            manual_exposure_ev: params.manual_exposure_ev,
            film_format_long_edge_mm: params.film_format_long_edge_mm,
            filter_override: params.filter_override != 0,
            uv: params.uv,
            ir: params.ir,
            apply_hanatos_window: params.apply_hanatos_window != 0,
            apply_hanatos_surface: params.apply_hanatos_surface != 0,
            input_compression: params.input_compression != 0,
        };
        // SAFETY: Matching retained profile and borrowed resources remain live through synchronous construction.
        let profile = unsafe { &*profile };
        match input::Recipe::new(profile.profile(), params, resources) {
            Ok(recipe) => {
                #[cfg(feature = "test-support")]
                boundary_fault(1);
                let projection = FjFilmInput::from_recipe(&recipe);
                // SAFETY: Commit a complete fixed result only after successful construction/projection.
                unsafe {
                    out.write(projection);
                }
                #[cfg(feature = "test-support")]
                boundary_fault(3);
                status(FJ_STATUS_SUCCESS)
            }
            Err(error) => {
                let metadata = FjFilmInputFailure::from_error(&error);
                let category = if matches!(error, input::Error::Panic(_)) {
                    FJ_STATUS_INTERNAL_FAILURE
                } else {
                    FJ_STATUS_PREPARATION_FAILURE
                };
                #[cfg(feature = "test-support")]
                let category = if let input::Error::Boundary(_, category) = error {
                    category
                } else {
                    category
                };
                // SAFETY: Failure metadata commits before potentially panicking formatting.
                unsafe {
                    failure.write(metadata);
                }
                #[cfg(feature = "test-support")]
                boundary_fault(2);
                if let Some(diagnostic) = &mut diagnostic {
                    let _ = write!(diagnostic, "{error}");
                }
                status(category)
            }
        }
    }));
    match result {
        Ok(status) => status,
        Err(payload) => {
            // SAFETY: Reclear only the authorized result. Retain prior sensitivity metadata if formatting panicked.
            unsafe {
                if valid_out {
                    out.write(FjFilmInput::empty());
                }
                if valid_failure {
                    (*failure).origin = FJ_FILM_INPUT_ORIGIN_INTERNAL;
                    (*failure).reason = FJ_FILM_INPUT_INTERNAL_PANIC;
                }
            }
            let _ = catch_unwind(AssertUnwindSafe(|| {
                // SAFETY: Re-admission never touches rejected diagnostic backing.
                if let Ok(mut diagnostic) = unsafe { Diagnostic::admit(error) } {
                    let _ = diagnostic.write_str("film input boundary panic");
                }
                drop(payload);
            }))
            .map_err(std::mem::forget);
            status(FJ_STATUS_INTERNAL_FAILURE)
        }
    }
}
/// # Safety
/// Same synchronous storage, profile lifetime and disjointness contract as prepare.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_film_input_prepare(
    profile: *const FjFilmProfile,
    params: *const FjFilmInputParams,
    resources: *const FjFilmInputResources,
    out: *mut FjFilmInput,
    failure: *mut FjFilmInputFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The C caller supplies the checked prepare contract.
    unsafe { prepare(profile.cast(), params, resources, out, failure, error) }
}

const _: () = {
    assert!(FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE == 0);
    assert!(FJ_FILM_INPUT_ILLUMINANT_D65 == 1);
    assert!(FJ_FILM_INPUT_ORIGIN_NONE == 0);
    assert!(FJ_FILM_INPUT_ORIGIN_RAW == 1);
    assert!(FJ_FILM_INPUT_ORIGIN_RESOURCE == 2);
    assert!(FJ_FILM_INPUT_ORIGIN_COLOR == 3);
    assert!(FJ_FILM_INPUT_ORIGIN_SENSITIVITY == 4);
    assert!(FJ_FILM_INPUT_ORIGIN_IDENTITY == 5);
    assert!(FJ_FILM_INPUT_ORIGIN_INTERNAL == 6);
    assert!(FJ_FILM_INPUT_REASON_NONE == 0);
    assert!(FJ_FILM_INPUT_RAW_RECORDS == 1);
    assert!(FJ_FILM_INPUT_RAW_DIAGNOSTIC == 2);
    assert!(FJ_FILM_INPUT_RAW_INPUT_SPACE == 3);
    assert!(FJ_FILM_INPUT_RAW_METHOD == 4);
    assert!(FJ_FILM_INPUT_RAW_METER == 5);
    assert!(FJ_FILM_INPUT_RAW_FLAGS == 6);
    assert!(FJ_FILM_INPUT_RAW_SPANS == 7);
    assert!(FJ_FILM_INPUT_RESOURCE_HANATOS_WINDOW == 1);
    assert!(FJ_FILM_INPUT_RESOURCE_HANATOS_SURFACE == 2);
    assert!(FJ_FILM_INPUT_RESOURCE_HANATOS_BLUR == 3);
    assert!(FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE == 4);
    assert!(FJ_FILM_INPUT_RESOURCE_TC_SOURCE == 5);
    assert!(FJ_FILM_INPUT_RESOURCE_INPUT_HULL == 6);
    assert!(FJ_FILM_INPUT_COLOR_MATRICES == 1);
    assert!(FJ_FILM_INPUT_COLOR_CAT16 == 2);
    assert!(FJ_FILM_INPUT_COLOR_CAT02 == 3);
    assert!(FJ_FILM_INPUT_SENSITIVITY_COMPUTED == 1);
    assert!(FJ_FILM_INPUT_SENSITIVITY_BOUNDARY == 2);
    assert!(FJ_FILM_INPUT_IDENTITY_REQUIRED == 1);
    assert!(FJ_FILM_INPUT_INTERNAL_PANIC == 1);
    assert!(FJ_FILM_INPUT_INTERNAL_INTEGRITY == 2);
    assert!(FJ_FILM_INPUT_INTERNAL_MATRICES_PANIC == 3);
    assert!(FJ_FILM_INPUT_INTERNAL_CAT16_PANIC == 4);
    assert!(FJ_FILM_INPUT_INTERNAL_CAT02_PANIC == 5);
    assert!(FJ_FILM_INPUT_INTERNAL_SENSITIVITY_PANIC == 6);
};

#[cfg(feature = "test-support")]
pub(crate) const ABI_FACTS: &[usize] = {
    use std::mem::offset_of;
    &[
        size_of::<FjFilmInputParams>(),
        align_of::<FjFilmInputParams>(),
        offset_of!(FjFilmInputParams, input_space),
        offset_of!(FjFilmInputParams, decode_cctf),
        offset_of!(FjFilmInputParams, method),
        offset_of!(FjFilmInputParams, auto_exposure_enabled),
        offset_of!(FjFilmInputParams, meter),
        offset_of!(FjFilmInputParams, manual_exposure_ev),
        offset_of!(FjFilmInputParams, film_format_long_edge_mm),
        offset_of!(FjFilmInputParams, filter_override),
        offset_of!(FjFilmInputParams, uv),
        3,
        offset_of!(FjFilmInputParams, ir),
        3,
        offset_of!(FjFilmInputParams, apply_hanatos_window),
        offset_of!(FjFilmInputParams, apply_hanatos_surface),
        offset_of!(FjFilmInputParams, input_compression),
        size_of::<FjFilmInputResources>(),
        align_of::<FjFilmInputResources>(),
        offset_of!(FjFilmInputResources, reference_illuminant),
        offset_of!(FjFilmInputResources, reference_white_xyz),
        offset_of!(FjFilmInputResources, integrated_d65_xyz),
        offset_of!(FjFilmInputResources, reconstructed_reference_white),
        offset_of!(FjFilmInputResources, tc_source_hash),
        offset_of!(FjFilmInputResources, input_hull_hash),
        size_of::<FjFilmInput>(),
        align_of::<FjFilmInput>(),
        offset_of!(FjFilmInput, color),
        offset_of!(FjFilmInput, xyz_to_linear_srgb),
        9,
        offset_of!(FjFilmInput, nominal_white_xyz),
        3,
        offset_of!(FjFilmInput, projection_white_xyz),
        3,
        offset_of!(FjFilmInput, method),
        offset_of!(FjFilmInput, auto_exposure_enabled),
        offset_of!(FjFilmInput, meter),
        offset_of!(FjFilmInput, manual_exposure_ev),
        offset_of!(FjFilmInput, film_format_long_edge_mm),
        offset_of!(FjFilmInput, band_pass_active),
        offset_of!(FjFilmInput, uv),
        3,
        offset_of!(FjFilmInput, ir),
        3,
        offset_of!(FjFilmInput, apply_hanatos_window),
        offset_of!(FjFilmInput, apply_hanatos_surface),
        offset_of!(FjFilmInput, hanatos_spectral_blur),
        offset_of!(FjFilmInput, hanatos_window),
        4,
        offset_of!(FjFilmInput, hanatos_surface_rgb),
        45,
        offset_of!(FjFilmInput, projection_illuminant),
        offset_of!(FjFilmInput, input_compression_active),
        offset_of!(FjFilmInput, input_hull_hash),
        offset_of!(FjFilmInput, tc_source_hash),
        offset_of!(FjFilmInput, sensitivity_rgb),
        243,
        offset_of!(FjFilmInput, sensitivity_hash),
        offset_of!(FjFilmInput, recipe_mallett_green_scale),
        offset_of!(FjFilmInput, tc_hash),
        offset_of!(FjFilmInput, hash),
        size_of::<FjFilmInputFailure>(),
        align_of::<FjFilmInputFailure>(),
        offset_of!(FjFilmInputFailure, origin),
        offset_of!(FjFilmInputFailure, reason),
        offset_of!(FjFilmInputFailure, sensitivity),
        FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE as usize,
        FJ_FILM_INPUT_ILLUMINANT_D65 as usize,
        FJ_FILM_INPUT_ORIGIN_NONE as usize,
        FJ_FILM_INPUT_ORIGIN_RAW as usize,
        FJ_FILM_INPUT_ORIGIN_RESOURCE as usize,
        FJ_FILM_INPUT_ORIGIN_COLOR as usize,
        FJ_FILM_INPUT_ORIGIN_SENSITIVITY as usize,
        FJ_FILM_INPUT_ORIGIN_IDENTITY as usize,
        FJ_FILM_INPUT_ORIGIN_INTERNAL as usize,
        FJ_FILM_INPUT_REASON_NONE as usize,
        FJ_FILM_INPUT_RAW_RECORDS as usize,
        FJ_FILM_INPUT_RAW_DIAGNOSTIC as usize,
        FJ_FILM_INPUT_RAW_INPUT_SPACE as usize,
        FJ_FILM_INPUT_RAW_METHOD as usize,
        FJ_FILM_INPUT_RAW_METER as usize,
        FJ_FILM_INPUT_RAW_FLAGS as usize,
        FJ_FILM_INPUT_RAW_SPANS as usize,
        FJ_FILM_INPUT_RESOURCE_HANATOS_WINDOW as usize,
        FJ_FILM_INPUT_RESOURCE_HANATOS_SURFACE as usize,
        FJ_FILM_INPUT_RESOURCE_HANATOS_BLUR as usize,
        FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE as usize,
        FJ_FILM_INPUT_RESOURCE_TC_SOURCE as usize,
        FJ_FILM_INPUT_RESOURCE_INPUT_HULL as usize,
        FJ_FILM_INPUT_COLOR_MATRICES as usize,
        FJ_FILM_INPUT_COLOR_CAT16 as usize,
        FJ_FILM_INPUT_COLOR_CAT02 as usize,
        FJ_FILM_INPUT_SENSITIVITY_COMPUTED as usize,
        FJ_FILM_INPUT_SENSITIVITY_BOUNDARY as usize,
        FJ_FILM_INPUT_IDENTITY_REQUIRED as usize,
        FJ_FILM_INPUT_INTERNAL_PANIC as usize,
        FJ_FILM_INPUT_INTERNAL_INTEGRITY as usize,
        FJ_FILM_INPUT_INTERNAL_MATRICES_PANIC as usize,
        FJ_FILM_INPUT_INTERNAL_CAT16_PANIC as usize,
        FJ_FILM_INPUT_INTERNAL_CAT02_PANIC as usize,
        FJ_FILM_INPUT_INTERNAL_SENSITIVITY_PANIC as usize,
    ]
};

const _: unsafe extern "C" fn(
    *const FjFilmProfile,
    *const FjFilmInputParams,
    *const FjFilmInputResources,
    *mut FjFilmInput,
    *mut FjFilmInputFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_film_input_prepare;

#[cfg(feature = "test-support")]
thread_local! { static BOUNDARY_FAULT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
#[cfg(feature = "test-support")]
pub(crate) fn arm_boundary_fault(stage: u32) {
    BOUNDARY_FAULT.set(stage);
}
#[cfg(feature = "test-support")]
fn boundary_fault(stage: u32) {
    if BOUNDARY_FAULT.get() == stage {
        BOUNDARY_FAULT.set(0);
        panic!("film input boundary fixture panic");
    }
}

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.input_space;

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.decode_cctf;

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.method;

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.auto_exposure_enabled;

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.meter;

const _: fn(&FjFilmInputParams) -> &f32 = |record| &record.manual_exposure_ev;

const _: fn(&FjFilmInputParams) -> &f32 = |record| &record.film_format_long_edge_mm;

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.filter_override;

const _: fn(&FjFilmInputParams) -> &[f64; 3] = |record| &record.uv;

const _: fn(&FjFilmInputParams) -> &[f64; 3] = |record| &record.ir;

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.apply_hanatos_window;

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.apply_hanatos_surface;

const _: fn(&FjFilmInputParams) -> &u32 = |record| &record.input_compression;

const _: fn(&FjFilmInputResources) -> &FjFloatSpan = |record| &record.reference_illuminant;

const _: fn(&FjFilmInputResources) -> &FjFloatSpan = |record| &record.reference_white_xyz;

const _: fn(&FjFilmInputResources) -> &FjFloatSpan = |record| &record.integrated_d65_xyz;

const _: fn(&FjFilmInputResources) -> &FjFloatSpan = |record| &record.reconstructed_reference_white;

const _: fn(&FjFilmInputResources) -> &u64 = |record| &record.tc_source_hash;

const _: fn(&FjFilmInputResources) -> &u64 = |record| &record.input_hull_hash;

const _: fn(&FjFilmInput) -> &FjInputColorConversion = |record| &record.color;

const _: fn(&FjFilmInput) -> &[f32; 9] = |record| &record.xyz_to_linear_srgb;

const _: fn(&FjFilmInput) -> &[f32; 3] = |record| &record.nominal_white_xyz;

const _: fn(&FjFilmInput) -> &[f32; 3] = |record| &record.projection_white_xyz;

const _: fn(&FjFilmInput) -> &u32 = |record| &record.method;

const _: fn(&FjFilmInput) -> &u32 = |record| &record.auto_exposure_enabled;

const _: fn(&FjFilmInput) -> &u32 = |record| &record.meter;

const _: fn(&FjFilmInput) -> &f32 = |record| &record.manual_exposure_ev;

const _: fn(&FjFilmInput) -> &f32 = |record| &record.film_format_long_edge_mm;

const _: fn(&FjFilmInput) -> &u32 = |record| &record.band_pass_active;

const _: fn(&FjFilmInput) -> &[f32; 3] = |record| &record.uv;

const _: fn(&FjFilmInput) -> &[f32; 3] = |record| &record.ir;

const _: fn(&FjFilmInput) -> &u32 = |record| &record.apply_hanatos_window;

const _: fn(&FjFilmInput) -> &u32 = |record| &record.apply_hanatos_surface;

const _: fn(&FjFilmInput) -> &f32 = |record| &record.hanatos_spectral_blur;

const _: fn(&FjFilmInput) -> &[f32; 4] = |record| &record.hanatos_window;

const _: fn(&FjFilmInput) -> &[[f32; 15]; 3] = |record| &record.hanatos_surface_rgb;

const _: fn(&FjFilmInput) -> &u32 = |record| &record.projection_illuminant;

const _: fn(&FjFilmInput) -> &u32 = |record| &record.input_compression_active;

const _: fn(&FjFilmInput) -> &u64 = |record| &record.input_hull_hash;

const _: fn(&FjFilmInput) -> &u64 = |record| &record.tc_source_hash;

const _: fn(&FjFilmInput) -> &[[f32; 3]; 81] = |record| &record.sensitivity_rgb;

const _: fn(&FjFilmInput) -> &u64 = |record| &record.sensitivity_hash;

const _: fn(&FjFilmInput) -> &f32 = |record| &record.recipe_mallett_green_scale;

const _: fn(&FjFilmInput) -> &u64 = |record| &record.tc_hash;

const _: fn(&FjFilmInput) -> &u64 = |record| &record.hash;

const _: fn(&FjFilmInputFailure) -> &u32 = |record| &record.origin;

const _: fn(&FjFilmInputFailure) -> &u32 = |record| &record.reason;

const _: fn(&FjFilmInputFailure) -> &FjSensitivityFailure = |record| &record.sensitivity;

#[cfg(test)]
mod tests {
    use super::*;
    use film_juicer_core::profile;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn consumed_hash_metadata_survives_later_scale_failure_and_formatting_panic() {
        // Complete an actual decoded profile; no private profile fields or admission bypass.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut document =
            std::fs::read_to_string(root.join("Resources/profiles/kodak_portra_400.json")).unwrap();
        let field = document.find("\"log_sensitivity\"").unwrap();
        let start = field + document[field..].find('[').unwrap();
        let mut depth = 0;
        let end = document[start..]
            .char_indices()
            .find_map(|(index, character)| {
                if character == '[' {
                    depth += 1;
                }
                if character == ']' {
                    depth -= 1;
                    if depth == 0 {
                        return Some(start + index + 1);
                    }
                }
                None
            })
            .unwrap();
        let rows = std::iter::repeat_n("[38.0,38.0,38.0]", 81)
            .collect::<Vec<_>>()
            .join(",");
        document.replace_range(start..end, &format!("[{rows}]"));
        let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
        };
        let scratch = root.join("out/validation").join(preset).join("film-input");
        std::fs::create_dir_all(&scratch).unwrap();
        let path = scratch.join(format!("sensitivity-overflow-{}.json", std::process::id()));
        std::fs::write(&path, document).unwrap();
        let source = profile::load_film_source(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let owner = FilmOwner::new(Arc::new(profile::FilmProfile::new(source).unwrap()));
        let reference = [1.0; 81];
        let params = FjFilmInputParams {
            input_space: 0,
            decode_cctf: 0,
            method: 1,
            auto_exposure_enabled: 0,
            meter: 0,
            manual_exposure_ev: 0.0,
            film_format_long_edge_mm: 35.0,
            filter_override: 1,
            uv: [1.0, 780.0, 1.0],
            ir: [0.0, 675.0, 15.0],
            apply_hanatos_window: 1,
            apply_hanatos_surface: 0,
            input_compression: 0,
        };
        let absent = || FjFloatSpan {
            data: std::ptr::null(),
            count: 0,
        };
        let resources = FjFilmInputResources {
            reference_illuminant: FjFloatSpan {
                data: reference.as_ptr(),
                count: 81,
            },
            reference_white_xyz: absent(),
            integrated_d65_xyz: absent(),
            reconstructed_reference_white: absent(),
            tc_source_hash: 0,
            input_hull_hash: 0,
        };
        let mut out = FjFilmInput::empty();
        out.hash = 99;
        let mut failure = FjFilmInputFailure::empty();
        let mut bytes = [0u8; 256];
        let mut diagnostic = FjErrorBuffer {
            data: bytes.as_mut_ptr().cast(),
            capacity: bytes.len(),
            length: 0,
        };
        // SAFETY: The complete live owner, initialized inputs and immutable backing remain alive; all outputs/backing are exclusive and disjoint.
        let status = unsafe {
            prepare(
                &owner,
                &params,
                &resources,
                &mut out,
                &mut failure,
                &mut diagnostic,
            )
        };
        assert_eq!(status.category, FJ_STATUS_PREPARATION_FAILURE);
        assert_eq!(
            (failure.origin, failure.reason),
            (
                FJ_FILM_INPUT_ORIGIN_SENSITIVITY,
                FJ_FILM_INPUT_SENSITIVITY_COMPUTED
            )
        );
        assert_eq!(
            (
                failure.sensitivity.hash_failure,
                failure.sensitivity.hash_sample_index
            ),
            (1, 240)
        );
        assert!(
            std::str::from_utf8(&bytes[..diagnostic.length])
                .unwrap()
                .contains("MallettResponse")
        );
        assert_eq!(out.hash, 0);
        #[cfg(feature = "test-support")]
        {
            arm_boundary_fault(2);
            // SAFETY: The same disjoint live storage is reused only after the preceding synchronous call completed.
            let status = unsafe {
                prepare(
                    &owner,
                    &params,
                    &resources,
                    &mut out,
                    &mut failure,
                    &mut diagnostic,
                )
            };
            assert_eq!(status.category, FJ_STATUS_INTERNAL_FAILURE);
            assert_eq!(
                (failure.origin, failure.reason),
                (FJ_FILM_INPUT_ORIGIN_INTERNAL, FJ_FILM_INPUT_INTERNAL_PANIC)
            );
            assert_eq!(
                (
                    failure.sensitivity.hash_failure,
                    failure.sensitivity.hash_sample_index
                ),
                (1, 240)
            );
            assert_eq!(out.hash, 0);
        }
    }
}
