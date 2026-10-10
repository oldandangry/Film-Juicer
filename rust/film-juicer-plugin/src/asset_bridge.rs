//! Private raw edge for the native process asset/catalog conversion.

#[cfg(any(test, feature = "test-support"))]
use std::collections::TryReserveError;
use std::fmt::{self, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::assets::{AssetError, Assets, CsvSource};
use film_juicer_core::profile::{
    IlluminantKind, Polarity, Role, Stage, Support, ProfileCompletionErrorKind, PrintDensityCurves,
    PrintDensityError,
};
use film_juicer_core::data_io::ReadErrorKind;
use crate::asset_spectral::{CmfOwner, CsvPairsOwner, MallettOwner, SpectraOwner};
use crate::asset_calibration::{self, CalibrationField, CalibrationLookup};

use crate::asset_profile::{
    FilmOwner, FilmView, IlluminantView, PrintOwner, PrintView, ProfileTablesView,
};

use crate::asset_catalog::{CatalogEntryView, CatalogOwner, PathError, PathView};
use crate::cuda::sys::{
    FJ_API_NONE, FJ_POLARITY_NEGATIVE, FJ_POLARITY_POSITIVE, FJ_STATUS_ALLOCATION_FAILURE,
    FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FjErrorBuffer, FjPathView, FjStatus, FjStringView, FjFloatSpan,
};

#[cfg(target_os = "linux")]
use crate::cuda::sys::FJ_PATH_UNIX_BYTES;
#[cfg(target_os = "windows")]
use crate::cuda::sys::FJ_PATH_WINDOWS_WIDE;

const FJ_PROFILE_ROLE_FILM: u32 = 0;
const FJ_PROFILE_ROLE_PRINT: u32 = 1;

#[repr(C)]
struct FjAssets {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FjCatalog {
    _opaque: [u8; 0],
}
#[repr(C)]
#[derive(Default)]
struct FjCatalogCounts {
    film_count: usize,
    print_count: usize,
}
#[repr(C)]
struct FjCatalogEntryView {
    key: FjStringView,
    label: FjStringView,
    polarity: u32,
}

impl FjCatalogEntryView {
    fn empty() -> Self {
        Self {
            key: text(""),
            label: text(""),
            polarity: 0,
        }
    }

    fn from_view(view: CatalogEntryView<'_>) -> Result<Self, Failure> {
        Ok(Self {
            key: text(view.key),
            label: text(view.label),
            polarity: match view.polarity {
                Polarity::Negative => FJ_POLARITY_NEGATIVE,
                Polarity::Positive => FJ_POLARITY_POSITIVE,
            },
        })
    }
}

#[repr(C)]
struct FjFilmProfile {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FjPrintProfile {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FjPrintDensityCurves {
    _opaque: [u8; 0],
}
const FJ_ILLUMINANT_NAMED: u32 = 0;
const FJ_ILLUMINANT_BLACKBODY: u32 = 1;
const FJ_PROFILE_SUPPORT_FILM: u32 = 0;
const FJ_PROFILE_SUPPORT_PAPER: u32 = 1;
const FJ_PROFILE_STAGE_FILMING: u32 = 0;
const FJ_PROFILE_STAGE_PRINTING: u32 = 1;
#[repr(C)]
struct FjIlluminantView {
    label: FjStringView,
    temperature_kelvin: f64,
    kind: u32,
}
#[repr(C)]
struct FjProfileTablesView {
    linear_sensitivity_rgb: FjFloatSpan,
    channel_density_cmy: FjFloatSpan,
    base_density: FjFloatSpan,
    log_exposure: FjFloatSpan,
    density_curves_cmy: FjFloatSpan,
}
#[repr(C)]
struct FjFilmDigest {
    gamma_samelayer_rgb: [f32; 3],
    gamma_interlayer_r_to_gb: [f32; 2],
    gamma_interlayer_g_to_rb: [f32; 2],
    gamma_interlayer_b_to_rg: [f32; 2],
    halation_first_sigma_um: [f32; 3],
    halation_primary_amount: [f32; 3],
    hanatos_spectral_gaussian_blur_default: f32,
}
#[repr(C)]
struct FjFilmProfileView {
    stock: FjStringView,
    reference_illuminant: FjIlluminantView,
    viewing_illuminant: FjIlluminantView,
    tables: FjProfileTablesView,
    wavelengths: FjFloatSpan,
    density_curves_layers: [[FjFloatSpan; 3]; 3],
    digest: FjFilmDigest,
    hanatos_window: FjFloatSpan,
    hanatos_surface_rgb: FjFloatSpan,
    asset_token: u64,
    support: u32,
    stage: u32,
    polarity: u32,
}
#[repr(C)]
struct FjPrintProfileView {
    stock: FjStringView,
    viewing_illuminant: FjIlluminantView,
    tables: FjProfileTablesView,
    asset_token: u64,
    stage: u32,
}
#[repr(C)]
struct FjPrintDensityView {
    totals_cmy: FjFloatSpan,
    hash: u64,
}

fn floats(values: &[f32]) -> FjFloatSpan {
    FjFloatSpan {
        data: if values.is_empty() {
            std::ptr::null()
        } else {
            values.as_ptr()
        },
        count: values.len(),
    }
}
fn illuminant(view: IlluminantView<'_>) -> FjIlluminantView {
    let (kind, temperature_kelvin) = match view.kind {
        IlluminantKind::Named => (FJ_ILLUMINANT_NAMED, 0.0),
        IlluminantKind::Blackbody { temperature_kelvin } => {
            (FJ_ILLUMINANT_BLACKBODY, temperature_kelvin)
        }
    };
    FjIlluminantView {
        label: text(view.label),
        kind,
        temperature_kelvin,
    }
}
fn tables(view: ProfileTablesView<'_>) -> FjProfileTablesView {
    // Fixed ABI `log_exposure` carries the admitted interpolation axis.
    FjProfileTablesView {
        linear_sensitivity_rgb: floats(view.linear_sensitivity_rgb),
        channel_density_cmy: floats(view.channel_density_cmy),
        base_density: floats(view.base_density),
        log_exposure: floats(view.interpolation_log_exposure),
        density_curves_cmy: floats(view.density_curves_cmy),
    }
}
fn stage(stage: Stage) -> u32 {
    match stage {
        Stage::Filming => FJ_PROFILE_STAGE_FILMING,
        Stage::Printing => FJ_PROFILE_STAGE_PRINTING,
    }
}
impl FjFilmProfileView {
    fn from_view(view: FilmView<'_>) -> Self {
        let defaults = view.processing_defaults;
        Self {
            stock: text(view.stock),
            reference_illuminant: illuminant(view.reference_illuminant),
            viewing_illuminant: illuminant(view.viewing_illuminant),
            tables: tables(view.tables),
            wavelengths: floats(view.wavelengths),
            density_curves_layers: view.density_curves_layers.map(|layer| layer.map(floats)),
            // Fixed ABI `digest`/`FjFilmDigest` carries processing defaults.
            digest: FjFilmDigest {
                gamma_samelayer_rgb: defaults.gamma_samelayer_rgb,
                gamma_interlayer_r_to_gb: defaults.gamma_interlayer_r_to_gb,
                gamma_interlayer_g_to_rb: defaults.gamma_interlayer_g_to_rb,
                gamma_interlayer_b_to_rg: defaults.gamma_interlayer_b_to_rg,
                halation_first_sigma_um: defaults.halation_first_sigma_um,
                halation_primary_amount: defaults.halation_primary_amount,
                hanatos_spectral_gaussian_blur_default: defaults
                    .hanatos_spectral_gaussian_blur_default,
            },
            hanatos_window: floats(view.hanatos_window),
            hanatos_surface_rgb: floats(view.hanatos_surface_rgb),
            asset_token: view.asset_token,
            support: match view.support {
                Support::Film => FJ_PROFILE_SUPPORT_FILM,
                Support::Paper => FJ_PROFILE_SUPPORT_PAPER,
            },
            stage: stage(view.stage),
            polarity: match view.polarity {
                Polarity::Negative => FJ_POLARITY_NEGATIVE,
                Polarity::Positive => FJ_POLARITY_POSITIVE,
            },
        }
    }
}
impl FjPrintProfileView {
    fn from_view(view: PrintView<'_>) -> Self {
        Self {
            stock: text(view.stock),
            viewing_illuminant: illuminant(view.viewing_illuminant),
            tables: tables(view.tables),
            asset_token: view.asset_token,
            stage: stage(view.stage),
        }
    }
}

/// # Safety
/// The caller authorizes a live initialized byte extent, disjoint from outputs.
unsafe fn key<'a>(view: FjStringView) -> Result<&'a str, Failure> {
    if view.count > isize::MAX as usize || (view.count != 0 && view.data.is_null()) {
        return Err(Failure::Input("invalid profile key extent"));
    }
    let bytes = if view.count == 0 {
        &[]
    } else {
        // SAFETY: The checked extent belongs to the caller for this operation.
        unsafe { std::slice::from_raw_parts(view.data.cast::<u8>(), view.count) }
    };
    std::str::from_utf8(bytes).map_err(|_| Failure::Input("invalid profile key UTF-8"))
}

fn text(value: &str) -> FjStringView {
    FjStringView {
        data: if value.is_empty() {
            std::ptr::null()
        } else {
            value.as_ptr().cast()
        },
        count: value.len(),
    }
}

pub(crate) enum Failure {
    Input(&'static str),
    Asset(AssetError),
    #[cfg(any(test, feature = "test-support"))]
    Capacity(TryReserveError),
    Gamma(PrintDensityError),
    Illuminant(film_juicer_core::illuminant::Error),
    Spectral(film_juicer_core::spectral::WhiteError),
    Reconstruction(film_juicer_core::reconstruction::Error),
    Exposure(film_juicer_core::exposure::Error),
    FilmTc(film_juicer_core::reconstruction::TcError),
    #[cfg(feature = "test-support")]
    InjectedAllocation,
    #[cfg(feature = "test-support")]
    InjectedPreparation(&'static str),
    #[cfg(any(test, feature = "test-support"))]
    Internal(&'static str),
}
impl Failure {
    fn category(&self) -> u32 {
        match self {
            #[cfg(feature = "test-support")]
            Self::InjectedPreparation(_) => FJ_STATUS_PREPARATION_FAILURE,
            Self::Input(_) => FJ_STATUS_UNSUPPORTED_INPUT,
            Self::Asset(
                AssetError::Catalog(_)
                | AssetError::MissingProfile { .. }
                | AssetError::ProfileDecode(_),
            ) => FJ_STATUS_PREPARATION_FAILURE,
            Self::Asset(AssetError::ProfileCompletion(error)) => match error.kind {
                ProfileCompletionErrorKind::Capacity => FJ_STATUS_ALLOCATION_FAILURE,
                _ => FJ_STATUS_PREPARATION_FAILURE,
            },
            Self::Asset(AssetError::Reconstruction(error) | AssetError::Cmf(error)) => {
                match error.kind {
                    ReadErrorKind::Capacity => FJ_STATUS_ALLOCATION_FAILURE,
                    _ => FJ_STATUS_PREPARATION_FAILURE,
                }
            }
            Self::Asset(AssetError::Calibration(error))
                if error.kind()
                    == film_juicer_core::data_io::calibration::ErrorKind::Read(
                        std::io::ErrorKind::OutOfMemory,
                    ) =>
            {
                FJ_STATUS_ALLOCATION_FAILURE
            }
            Self::Asset(AssetError::CsvSource { error, .. }) => match error.kind {
                ReadErrorKind::Capacity => FJ_STATUS_ALLOCATION_FAILURE,
                _ => FJ_STATUS_PREPARATION_FAILURE,
            },
            Self::Spectral(_) => FJ_STATUS_PREPARATION_FAILURE,
            Self::Exposure(_) => FJ_STATUS_PREPARATION_FAILURE,
            Self::FilmTc(film_juicer_core::reconstruction::TcError::AllocationFailure) => {
                FJ_STATUS_ALLOCATION_FAILURE
            }
            Self::FilmTc(_) => FJ_STATUS_PREPARATION_FAILURE,
            Self::Reconstruction(error) => match error {
                film_juicer_core::reconstruction::Error::AllocationFailure => {
                    FJ_STATUS_ALLOCATION_FAILURE
                }
                _ => FJ_STATUS_PREPARATION_FAILURE,
            },
            #[cfg(feature = "test-support")]
            Self::InjectedAllocation => FJ_STATUS_ALLOCATION_FAILURE,
            Self::Gamma(PrintDensityError::InvalidGamma) => FJ_STATUS_UNSUPPORTED_INPUT,
            Self::Gamma(PrintDensityError::Capacity) => FJ_STATUS_ALLOCATION_FAILURE,
            Self::Gamma(_) => FJ_STATUS_PREPARATION_FAILURE,
            Self::Illuminant(error) => match error.kind {
                film_juicer_core::illuminant::ErrorKind::Capacity => FJ_STATUS_ALLOCATION_FAILURE,
                film_juicer_core::illuminant::ErrorKind::Preparation => {
                    FJ_STATUS_PREPARATION_FAILURE
                }
            },
            #[cfg(any(test, feature = "test-support"))]
            Self::Capacity(_) => FJ_STATUS_ALLOCATION_FAILURE,
            Self::Asset(AssetError::Noise(error)) => noise_category(error.kind()),
            Self::Asset(AssetError::NoiseCachePoisoned) => FJ_STATUS_INTERNAL_FAILURE,
            Self::Asset(_) => FJ_STATUS_INTERNAL_FAILURE,
            #[cfg(any(test, feature = "test-support"))]
            Self::Internal(_) => FJ_STATUS_INTERNAL_FAILURE,
        }
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => formatter.write_str(message),
            #[cfg(any(test, feature = "test-support"))]
            Self::Internal(message) => formatter.write_str(message),
            Self::Asset(error) => write!(formatter, "{error}"),
            #[cfg(any(test, feature = "test-support"))]
            Self::Capacity(error) => write!(formatter, "asset boundary capacity: {error}"),
            Self::Gamma(error) => write!(formatter, "{error}"),
            Self::Illuminant(error) => write!(formatter, "{error}"),
            Self::Spectral(error) => write!(formatter, "{error}"),
            Self::Reconstruction(error) => write!(formatter, "{error}"),
            Self::Exposure(error) => write!(formatter, "{error}"),
            Self::FilmTc(error) => write!(formatter, "{error}"),
            #[cfg(feature = "test-support")]
            Self::InjectedAllocation => {
                formatter.write_str("injected spectral allocation category")
            }
            #[cfg(feature = "test-support")]
            Self::InjectedPreparation(message) => formatter.write_str(message),
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

struct Diagnostic<'a> {
    data: *mut u8,
    capacity: usize,
    length: &'a mut usize,
}
impl Write for Diagnostic<'_> {
    fn write_str(&mut self, message: &str) -> fmt::Result {
        let count = message.len().min(self.capacity - 1 - *self.length);
        // SAFETY: run constructs this writer from disjoint, exclusive foreign
        // bytes. The checked capacity and initialized length bound every write.
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), self.data.add(*self.length), count);
            self.data.add(*self.length + count).write(0);
        }
        *self.length += count;
        Ok(())
    }
}

/// # Safety
/// A nonnull diagnostic is initialized, aligned and exclusively writable; its
/// nonempty backing extent is disjoint from all other call storage.
pub(crate) unsafe fn run<T>(
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<T, Failure>,
) -> Result<T, FjStatus> {
    // Captured consumed owners are dropped inside containment even when the
    // diagnostic is malformed and the ordinary operation must be skipped.
    catch_unwind(AssertUnwindSafe(|| {
        let mut diagnostic = if error.is_null() {
            None
        } else {
            // SAFETY: The caller supplies the record; zero capacity deliberately
            // does not read the data field, which need not be initialized.
            let (capacity, length) = unsafe {
                let capacity = std::ptr::addr_of!((*error).capacity).read();
                let length = std::ptr::addr_of_mut!((*error).length);
                length.write(0);
                (capacity, &mut *length)
            };
            if capacity == 0 {
                None
            } else {
                // SAFETY: Nonzero capacity requires an initialized data field.
                let data = unsafe { std::ptr::addr_of!((*error).data).read() };
                if data.is_null() || capacity > isize::MAX as usize {
                    return Err(status(FJ_STATUS_UNSUPPORTED_INPUT));
                }
                // SAFETY: The caller authorizes capacity writable bytes.
                unsafe { data.write(0) };
                Some(Diagnostic {
                    data: data.cast(),
                    capacity,
                    length,
                })
            }
        };
        let result = catch_unwind(AssertUnwindSafe(|| match operation() {
            Ok(value) => Ok(value),
            Err(failure) => {
                if let Some(diagnostic) = &mut diagnostic {
                    let _ = write!(diagnostic, "{failure}");
                }
                Err(status(failure.category()))
            }
        }));
        result.unwrap_or_else(|_| {
            if let Some(diagnostic) = &mut diagnostic {
                *diagnostic.length = 0;
                let _ = diagnostic.write_str("asset boundary panic");
            }
            Err(status(FJ_STATUS_INTERNAL_FAILURE))
        })
    }))
    .unwrap_or(Err(status(FJ_STATUS_INTERNAL_FAILURE)))
}

/// # Safety
/// The caller authorizes the initialized, platform-native input extent.
unsafe fn path(view: FjPathView) -> Result<std::path::PathBuf, Failure> {
    if view.data.is_null() || view.count == 0 {
        return Err(Failure::Input("empty resource root"));
    }
    #[cfg(target_os = "linux")]
    let units = {
        if view.encoding != FJ_PATH_UNIX_BYTES || view.count > isize::MAX as usize {
            return Err(Failure::Input("invalid Unix path extent or encoding"));
        }
        // SAFETY: Bounds checked before forming the caller-authorized byte slice.
        PathView::Bytes(unsafe { std::slice::from_raw_parts(view.data.cast::<u8>(), view.count) })
    };
    #[cfg(target_os = "windows")]
    let units = {
        if view.encoding != FJ_PATH_WINDOWS_WIDE
            || view.count > isize::MAX as usize / 2
            || !(view.data as usize).is_multiple_of(std::mem::align_of::<u16>())
        {
            return Err(Failure::Input("invalid Windows path extent or encoding"));
        }
        // SAFETY: Checked aligned byte extent precedes the authorized u16 slice.
        PathView::Wide(unsafe { std::slice::from_raw_parts(view.data.cast::<u16>(), view.count) })
    };
    units.to_path_buf().map_err(|error| match error {
        PathError::Empty => Failure::Input("empty resource root"),
        PathError::ContainsNul => Failure::Input("resource root contains NUL"),
    })
}

// FJ_TEMP_BRIDGE: asset conversion; remove S4.E.
/// # Safety
/// Inputs have the documented initialized extent. Nonnull outputs/diagnostics
/// are aligned, exclusive and mutually disjoint. Root borrows expire at return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_assets_create(
    resource_root: FjPathView,
    out_assets: *mut *mut FjAssets,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_assets.is_null() {
        // SAFETY: Caller provides one exclusive output slot.
        unsafe { out_assets.write(std::ptr::null_mut()) };
    }
    // SAFETY: Diagnostic/input/output storage follows the export contract.
    unsafe {
        run(error, || {
            if out_assets.is_null() {
                return Err(Failure::Input("NULL Assets output"));
            }
            let assets = Box::new(Assets::new(path(resource_root)?));
            #[cfg(feature = "test-support")]
            LIVE_ASSETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_assets.write(Box::into_raw(assets).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// assets is NULL or a live matching owner taken once; caller excludes all reads.
/// Diagnostics follow the exclusive disjoint storage contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_assets_destroy(
    assets: *mut FjAssets,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Matching live allocation is consumed, including rejected diagnostics.
    #[cfg(feature = "test-support")]
    if !assets.is_null() {
        LIVE_ASSETS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
    let owner = if assets.is_null() {
        None
    } else {
        // SAFETY: The matching live Box allocation is taken exactly once.
        Some(unsafe { Box::from_raw(assets.cast::<Assets>()) })
    };
    // SAFETY: Captured owner cleanup and diagnostics are contained by run.
    unsafe {
        run(error, move || {
            drop(owner);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// assets is live through return; outputs and diagnostics are exclusive and
/// disjoint. Catalog release is excluded from every returned-view use.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_catalog_acquire(
    assets: *const FjAssets,
    out_catalog: *mut *mut FjCatalog,
    out_counts: *mut FjCatalogCounts,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Each nonnull output is a caller-authorized exclusive slot.
    unsafe {
        if !out_catalog.is_null() {
            out_catalog.write(std::ptr::null_mut());
        }
        if !out_counts.is_null() {
            out_counts.write(FjCatalogCounts::default());
        }
    }
    // SAFETY: The live owner and disjoint output/diagnostic storage are authorized.
    unsafe {
        run(error, || {
            if assets.is_null() || out_catalog.is_null() || out_counts.is_null() {
                return Err(Failure::Input("NULL catalog acquisition input/output"));
            }
            let catalog = (&*assets.cast::<Assets>())
                .catalog()
                .map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            inject()?;
            let owner = Box::new(CatalogOwner::new(catalog));
            let counts = FjCatalogCounts {
                film_count: owner.count(Role::Film),
                print_count: owner.count(Role::Print),
            };
            #[cfg(feature = "test-support")]
            LIVE_CATALOGS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_counts.write(counts);
            out_catalog.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// catalog is live through all returned-view use. Outputs and diagnostics are
/// aligned, exclusive and disjoint from owner storage and other call storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_catalog_entry(
    catalog: *const FjCatalog,
    role: u32,
    index: usize,
    out_entry: *mut FjCatalogEntryView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_entry.is_null() {
        // SAFETY: Caller authorizes this exclusive record.
        unsafe { out_entry.write(FjCatalogEntryView::empty()) };
    }
    // SAFETY: The owner and disjoint output/diagnostics obey the export contract.
    unsafe {
        run(error, || {
            if catalog.is_null() || out_entry.is_null() {
                return Err(Failure::Input("NULL catalog entry input/output"));
            }
            let role = match role {
                FJ_PROFILE_ROLE_FILM => Role::Film,
                FJ_PROFILE_ROLE_PRINT => Role::Print,
                _ => return Err(Failure::Input("unknown profile role")),
            };
            let view = (&*catalog.cast::<CatalogOwner>())
                .entry(role, index)
                .ok_or(Failure::Input("catalog index out of range"))?;
            out_entry.write(FjCatalogEntryView::from_view(view)?);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// catalog is NULL or a matching live owner taken once; release excludes reads
/// and outstanding views. Diagnostics follow the exclusive storage contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_catalog_release(
    catalog: *mut FjCatalog,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Matching live allocation is taken before diagnostic validation.
    #[cfg(feature = "test-support")]
    if !catalog.is_null() {
        LIVE_CATALOGS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
    let owner = if catalog.is_null() {
        None
    } else {
        // SAFETY: The matching live Box allocation is taken exactly once.
        Some(unsafe { Box::from_raw(catalog.cast::<CatalogOwner>()) })
    };
    // SAFETY: Cleanup of the captured owner is contained even on malformed error.
    unsafe {
        run(error, move || {
            drop(owner);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

#[cfg(feature = "test-support")]
static LIVE_ASSETS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "test-support")]
static LIVE_CATALOGS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(any(test, feature = "test-support"))]
thread_local! { static FAULT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
#[cfg(any(test, feature = "test-support"))]
fn inject() -> Result<(), Failure> {
    match FAULT.replace(0) {
        1 => panic!("catalog projection fixture panic"),
        2 => {
            let mut units = Vec::<u16>::new();
            units.try_reserve(usize::MAX).map_err(Failure::Capacity)
        }
        _ => Ok(()),
    }
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_catalog_fault(fault: u32) {
    FAULT.set(fault);
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_assets_live_owners() -> usize {
    LIVE_ASSETS.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_catalog_live_owners() -> usize {
    LIVE_CATALOGS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn layouts_and_explicit_tags() {
        assert_eq!(
            (size_of::<FjCatalogCounts>(), align_of::<FjCatalogCounts>()),
            (16, 8)
        );
        assert_eq!(
            (
                offset_of!(FjCatalogCounts, film_count),
                offset_of!(FjCatalogCounts, print_count)
            ),
            (0, 8)
        );
        assert_eq!(
            (
                size_of::<FjCatalogEntryView>(),
                align_of::<FjCatalogEntryView>()
            ),
            (40, 8)
        );
        assert_eq!(
            (
                offset_of!(FjCatalogEntryView, key),
                offset_of!(FjCatalogEntryView, label),
                offset_of!(FjCatalogEntryView, polarity)
            ),
            (0, 16, 32)
        );
        assert_eq!(
            (
                FJ_PROFILE_ROLE_FILM,
                FJ_PROFILE_ROLE_PRINT,
                FJ_POLARITY_NEGATIVE,
                FJ_POLARITY_POSITIVE
            ),
            (0, 1, 0, 1)
        );
    }

    #[test]
    fn rejected_diagnostics_drop_taken_capture_inside_containment() {
        struct Observed<'a>(&'a std::cell::Cell<bool>);
        impl Drop for Observed<'_> {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }
        let dropped = std::cell::Cell::new(false);
        let owner = Observed(&dropped);
        let mut malformed = FjErrorBuffer {
            data: std::ptr::null_mut(),
            capacity: 1,
            length: 99,
        };
        // SAFETY: Exclusive aligned error record; no backing accessed for malformed shape.
        let result = unsafe {
            run(&raw mut malformed, move || {
                drop(owner);
                Ok(())
            })
        };
        assert!(matches!(
            result,
            Err(FjStatus {
                category: FJ_STATUS_UNSUPPORTED_INPUT,
                ..
            })
        ));
        assert!(dropped.get());
        assert_eq!(malformed.length, 0);
    }

    #[test]
    fn panic_and_capacity_leave_acquisition_empty_and_retry_source() {
        let assets =
            Assets::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let borrowed = (&raw const assets).cast::<FjAssets>();
        for (fault, expected) in [
            (1, FJ_STATUS_INTERNAL_FAILURE),
            (2, FJ_STATUS_ALLOCATION_FAILURE),
        ] {
            FAULT.set(fault);
            let mut owner = std::ptr::null_mut();
            let mut counts = FjCatalogCounts {
                film_count: 999,
                print_count: 999,
            };
            // SAFETY: Live stack owner and exclusive disjoint outputs, null diagnostics.
            let result = unsafe {
                fj_legacy_catalog_acquire(
                    borrowed,
                    &raw mut owner,
                    &raw mut counts,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(result.category, expected);
            assert!(owner.is_null());
            assert_eq!((counts.film_count, counts.print_count), (0, 0));
        }
        let snapshot = assets.catalog().unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &snapshot,
            &assets.catalog().unwrap()
        ));
    }
}

/// # Safety
/// Assets is live through return; key has its documented initialized extent.
/// Outputs/diagnostics are aligned, exclusive and disjoint from all borrows.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_profile_acquire(
    assets: *const FjAssets,
    input_key: FjStringView,
    out_profile: *mut *mut FjFilmProfile,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_profile.is_null() {
        // SAFETY: The caller authorizes one exclusive output slot.
        unsafe { out_profile.write(std::ptr::null_mut()) };
    }
    // SAFETY: Live owner and disjoint caller storage follow the export contract.
    unsafe {
        run(error, || {
            if assets.is_null() || out_profile.is_null() {
                return Err(Failure::Input("NULL film acquisition input/output"));
            }
            let profile = (&*assets.cast::<Assets>())
                .film(key(input_key)?)
                .map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            profile_inject()?;
            let owner = Box::new(FilmOwner::new(profile));
            #[cfg(feature = "test-support")]
            LIVE_PROFILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_profile.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Profile is live through every returned-view use; release excludes borrows.
/// Outputs and diagnostics are aligned, exclusive and mutually disjoint.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_profile_view(
    profile: *const FjFilmProfile,
    out_view: *mut FjFilmProfileView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_view.is_null() {
        // SAFETY: This raw record contains only numeric fields and nullable
        // pointers; all-zero initializes every semantic empty output field.
        unsafe { out_view.write(std::mem::zeroed()) };
    }
    // SAFETY: Matching live owner and authorized disjoint output storage.
    unsafe {
        run(error, || {
            if profile.is_null() || out_view.is_null() {
                return Err(Failure::Input("NULL film view input/output"));
            }
            out_view.write(FjFilmProfileView::from_view(
                (&*profile.cast::<FilmOwner>()).view(),
            ));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Profile is NULL or a matching live allocation consumed once. The caller
/// excludes every use/outstanding view. Diagnostics obey the disjoint contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_profile_release(
    profile: *mut FjFilmProfile,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let owner = if profile.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        LIVE_PROFILES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Matching Box is taken once before diagnostic validation.
        Some(unsafe { Box::from_raw(profile.cast::<FilmOwner>()) })
    };
    // SAFETY: Taken owner drops inside containment even on rejected diagnostics.
    unsafe {
        run(error, move || {
            drop(owner);
            #[cfg(any(test, feature = "test-support"))]
            if PROFILE_FAULT.replace(0) == 3 {
                return Err(Failure::Internal(
                    "profile release fixture failure after consume",
                ));
            }
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Assets is live through return; key has its documented initialized extent.
/// Outputs/diagnostics are aligned, exclusive and disjoint from all borrows.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_print_profile_acquire(
    assets: *const FjAssets,
    input_key: FjStringView,
    out_profile: *mut *mut FjPrintProfile,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_profile.is_null() {
        // SAFETY: The caller authorizes one exclusive output slot.
        unsafe { out_profile.write(std::ptr::null_mut()) };
    }
    // SAFETY: Live owner and disjoint caller storage follow the export contract.
    unsafe {
        run(error, || {
            if assets.is_null() || out_profile.is_null() {
                return Err(Failure::Input("NULL print acquisition input/output"));
            }
            let profile = (&*assets.cast::<Assets>())
                .print(key(input_key)?)
                .map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            profile_inject()?;
            let owner = Box::new(PrintOwner::new(profile));
            #[cfg(feature = "test-support")]
            LIVE_PROFILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_profile.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Profile is live through every returned-view use; release excludes borrows.
/// Outputs and diagnostics are aligned, exclusive and mutually disjoint.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_print_profile_view(
    profile: *const FjPrintProfile,
    out_view: *mut FjPrintProfileView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_view.is_null() {
        // SAFETY: This raw record contains only numeric fields and nullable
        // pointers; all-zero initializes every semantic empty output field.
        unsafe { out_view.write(std::mem::zeroed()) };
    }
    // SAFETY: Matching live owner and authorized disjoint output storage.
    unsafe {
        run(error, || {
            if profile.is_null() || out_view.is_null() {
                return Err(Failure::Input("NULL print view input/output"));
            }
            out_view.write(FjPrintProfileView::from_view(
                (&*profile.cast::<PrintOwner>()).view(),
            ));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Profile is NULL or a matching live allocation consumed once. The caller
/// excludes every use/outstanding view. Diagnostics obey the disjoint contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_print_profile_release(
    profile: *mut FjPrintProfile,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let owner = if profile.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        LIVE_PROFILES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Matching Box is taken once before diagnostic validation.
        Some(unsafe { Box::from_raw(profile.cast::<PrintOwner>()) })
    };
    // SAFETY: Taken owner drops inside containment even on rejected diagnostics.
    unsafe {
        run(error, move || {
            drop(owner);
            #[cfg(any(test, feature = "test-support"))]
            if PROFILE_FAULT.replace(0) == 3 {
                return Err(Failure::Internal(
                    "profile release fixture failure after consume",
                ));
            }
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Print profile is a live matching owner; disjoint output and diagnostics are
/// exclusive. Immutable samples may overlap only with disjoint output storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_print_profile_sample_density(
    profile: *const FjPrintProfile,
    gamma: f64,
    out_curves: *mut *mut FjPrintDensityCurves,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_curves.is_null() {
        // SAFETY: Caller authorizes one exclusive output slot.
        unsafe { out_curves.write(std::ptr::null_mut()) };
    }
    // SAFETY: Matching live owner and disjoint authorized output storage.
    unsafe {
        run(error, || {
            if profile.is_null() || out_curves.is_null() {
                return Err(Failure::Input("NULL print sample input/output"));
            }
            #[cfg(any(test, feature = "test-support"))]
            profile_inject()?;
            let curves = (&*profile.cast::<PrintOwner>())
                .sample_density_curves(gamma)
                .map_err(Failure::Gamma)?;
            #[cfg(feature = "test-support")]
            LIVE_DENSITY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_curves.write(Box::into_raw(Box::new(curves)).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Curves is live through returned-view use; output/diagnostics are disjoint,
/// aligned and exclusive. Release excludes every use and outstanding view.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_print_density_view(
    curves: *const FjPrintDensityCurves,
    out_view: *mut FjPrintDensityView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_view.is_null() {
        // SAFETY: Exclusive raw output; null pointer/zero count and hash are empty.
        unsafe {
            out_view.write(FjPrintDensityView {
                totals_cmy: floats(&[]),
                hash: 0,
            })
        };
    }
    // SAFETY: Matching live result and authorized disjoint output storage.
    unsafe {
        run(error, || {
            if curves.is_null() || out_view.is_null() {
                return Err(Failure::Input("NULL print density view input/output"));
            }
            let owner = &*curves.cast::<PrintDensityCurves>();
            out_view.write(FjPrintDensityView {
                totals_cmy: floats(owner.totals().as_flattened()),
                hash: owner.hash(),
            });
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Curves is NULL or a matching live owner consumed once; caller excludes use
/// and outstanding views. Diagnostics are authorized disjoint writable storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_print_density_release(
    curves: *mut FjPrintDensityCurves,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let owner = if curves.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        LIVE_DENSITY.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Matching Box consumed once before diagnostic validation.
        Some(unsafe { Box::from_raw(curves.cast::<PrintDensityCurves>()) })
    };
    // SAFETY: Taken owner drops within containment including malformed error.
    unsafe {
        run(error, move || {
            drop(owner);
            #[cfg(any(test, feature = "test-support"))]
            if PROFILE_FAULT.replace(0) == 3 {
                return Err(Failure::Internal(
                    "profile release fixture failure after consume",
                ));
            }
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Assets is live throughout this borrow; diagnostics obey the disjoint contract.
/// Release detaches cache holds and does not revoke separately acquired owners.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_assets_release_cached_payloads(
    assets: *const FjAssets,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Live Assets and caller-authorized diagnostics follow the contract.
    unsafe {
        run(error, || {
            if assets.is_null() {
                return Err(Failure::Input("NULL Assets cache release"));
            }
            (&*assets.cast::<Assets>())
                .release_cached_payloads()
                .map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            if CACHE_FAILURE.swap(false, std::sync::atomic::Ordering::Relaxed) {
                return Err(Failure::Internal("injected post-release cache failure"));
            }
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

#[cfg(feature = "test-support")]
static LIVE_PROFILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "test-support")]
static LIVE_DENSITY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(any(test, feature = "test-support"))]
thread_local! { static PROFILE_FAULT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
#[cfg(any(test, feature = "test-support"))]
static CACHE_FAILURE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(any(test, feature = "test-support"))]
fn profile_inject() -> Result<(), Failure> {
    match PROFILE_FAULT.replace(0) {
        1 => panic!("profile boundary fixture panic"),
        2 => Vec::<f32>::new()
            .try_reserve(usize::MAX)
            .map_err(Failure::Capacity),
        _ => Ok(()),
    }
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_profile_fault(fault: u32) {
    PROFILE_FAULT.set(fault);
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_profile_live_owners() -> usize {
    LIVE_PROFILES.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_print_density_live_owners() -> usize {
    LIVE_DENSITY.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_cache_release_failure() {
    CACHE_FAILURE.store(true, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "test-support")]
static PRODUCTION_PROFILE_ABI_FACTS: &[usize] = &[
    std::mem::size_of::<FjIlluminantView>(),
    std::mem::align_of::<FjIlluminantView>(),
    std::mem::offset_of!(FjIlluminantView, label),
    std::mem::offset_of!(FjIlluminantView, temperature_kelvin),
    std::mem::offset_of!(FjIlluminantView, kind),
    std::mem::size_of::<FjProfileTablesView>(),
    std::mem::align_of::<FjProfileTablesView>(),
    std::mem::offset_of!(FjProfileTablesView, linear_sensitivity_rgb),
    std::mem::offset_of!(FjProfileTablesView, channel_density_cmy),
    std::mem::offset_of!(FjProfileTablesView, base_density),
    std::mem::offset_of!(FjProfileTablesView, log_exposure),
    std::mem::offset_of!(FjProfileTablesView, density_curves_cmy),
    std::mem::size_of::<FjFilmDigest>(),
    std::mem::align_of::<FjFilmDigest>(),
    std::mem::offset_of!(FjFilmDigest, gamma_samelayer_rgb),
    std::mem::offset_of!(FjFilmDigest, gamma_interlayer_r_to_gb),
    std::mem::offset_of!(FjFilmDigest, gamma_interlayer_g_to_rb),
    std::mem::offset_of!(FjFilmDigest, gamma_interlayer_b_to_rg),
    std::mem::offset_of!(FjFilmDigest, halation_first_sigma_um),
    std::mem::offset_of!(FjFilmDigest, halation_primary_amount),
    std::mem::offset_of!(FjFilmDigest, hanatos_spectral_gaussian_blur_default),
    std::mem::size_of::<FjFilmProfileView>(),
    std::mem::align_of::<FjFilmProfileView>(),
    std::mem::offset_of!(FjFilmProfileView, stock),
    std::mem::offset_of!(FjFilmProfileView, reference_illuminant),
    std::mem::offset_of!(FjFilmProfileView, viewing_illuminant),
    std::mem::offset_of!(FjFilmProfileView, tables),
    std::mem::offset_of!(FjFilmProfileView, wavelengths),
    std::mem::offset_of!(FjFilmProfileView, density_curves_layers),
    std::mem::offset_of!(FjFilmProfileView, digest),
    std::mem::offset_of!(FjFilmProfileView, hanatos_window),
    std::mem::offset_of!(FjFilmProfileView, hanatos_surface_rgb),
    std::mem::offset_of!(FjFilmProfileView, asset_token),
    std::mem::offset_of!(FjFilmProfileView, support),
    std::mem::offset_of!(FjFilmProfileView, stage),
    std::mem::offset_of!(FjFilmProfileView, polarity),
    std::mem::size_of::<FjPrintProfileView>(),
    std::mem::align_of::<FjPrintProfileView>(),
    std::mem::offset_of!(FjPrintProfileView, stock),
    std::mem::offset_of!(FjPrintProfileView, viewing_illuminant),
    std::mem::offset_of!(FjPrintProfileView, tables),
    std::mem::offset_of!(FjPrintProfileView, asset_token),
    std::mem::offset_of!(FjPrintProfileView, stage),
    std::mem::size_of::<FjPrintDensityView>(),
    std::mem::align_of::<FjPrintDensityView>(),
    std::mem::offset_of!(FjPrintDensityView, totals_cmy),
    std::mem::offset_of!(FjPrintDensityView, hash),
    std::mem::size_of::<FjCatalogEntryView>(),
    std::mem::align_of::<FjCatalogEntryView>(),
    std::mem::offset_of!(FjCatalogEntryView, key),
    std::mem::offset_of!(FjCatalogEntryView, label),
    std::mem::offset_of!(FjCatalogEntryView, polarity),
    FJ_ILLUMINANT_NAMED as usize,
    FJ_ILLUMINANT_BLACKBODY as usize,
    FJ_PROFILE_SUPPORT_FILM as usize,
    FJ_PROFILE_SUPPORT_PAPER as usize,
    FJ_PROFILE_STAGE_FILMING as usize,
    FJ_PROFILE_STAGE_PRINTING as usize,
];
/// # Safety
/// count is one aligned exclusive writable size_t slot.
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_production_profile_abi_facts(count: *mut usize) -> *const usize {
    if count.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller authorizes this exclusive output slot.
    unsafe { count.write(PRODUCTION_PROFILE_ABI_FACTS.len()) };
    PRODUCTION_PROFILE_ABI_FACTS.as_ptr()
}
#[cfg(test)]
#[test]
fn production_profile_layouts() {
    assert_eq!(
        (
            std::mem::size_of::<FjIlluminantView>(),
            std::mem::align_of::<FjIlluminantView>()
        ),
        (32, 8)
    );
    assert_eq!(std::mem::offset_of!(FjIlluminantView, label), 0);
    assert_eq!(
        std::mem::offset_of!(FjIlluminantView, temperature_kelvin),
        16
    );
    assert_eq!(std::mem::offset_of!(FjIlluminantView, kind), 24);
    assert_eq!(
        (
            std::mem::size_of::<FjProfileTablesView>(),
            std::mem::align_of::<FjProfileTablesView>()
        ),
        (80, 8)
    );
    assert_eq!(
        std::mem::offset_of!(FjProfileTablesView, linear_sensitivity_rgb),
        0
    );
    assert_eq!(
        std::mem::offset_of!(FjProfileTablesView, channel_density_cmy),
        16
    );
    assert_eq!(std::mem::offset_of!(FjProfileTablesView, base_density), 32);
    assert_eq!(std::mem::offset_of!(FjProfileTablesView, log_exposure), 48);
    assert_eq!(
        std::mem::offset_of!(FjProfileTablesView, density_curves_cmy),
        64
    );
    assert_eq!(
        (
            std::mem::size_of::<FjFilmDigest>(),
            std::mem::align_of::<FjFilmDigest>()
        ),
        (64, 4)
    );
    assert_eq!(std::mem::offset_of!(FjFilmDigest, gamma_samelayer_rgb), 0);
    assert_eq!(
        std::mem::offset_of!(FjFilmDigest, gamma_interlayer_r_to_gb),
        12
    );
    assert_eq!(
        std::mem::offset_of!(FjFilmDigest, gamma_interlayer_g_to_rb),
        20
    );
    assert_eq!(
        std::mem::offset_of!(FjFilmDigest, gamma_interlayer_b_to_rg),
        28
    );
    assert_eq!(
        std::mem::offset_of!(FjFilmDigest, halation_first_sigma_um),
        36
    );
    assert_eq!(
        std::mem::offset_of!(FjFilmDigest, halation_primary_amount),
        48
    );
    assert_eq!(
        std::mem::offset_of!(FjFilmDigest, hanatos_spectral_gaussian_blur_default),
        60
    );
    assert_eq!(
        (
            std::mem::size_of::<FjFilmProfileView>(),
            std::mem::align_of::<FjFilmProfileView>()
        ),
        (440, 8)
    );
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, stock), 0);
    assert_eq!(
        std::mem::offset_of!(FjFilmProfileView, reference_illuminant),
        16
    );
    assert_eq!(
        std::mem::offset_of!(FjFilmProfileView, viewing_illuminant),
        48
    );
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, tables), 80);
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, wavelengths), 160);
    assert_eq!(
        std::mem::offset_of!(FjFilmProfileView, density_curves_layers),
        176
    );
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, digest), 320);
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, hanatos_window), 384);
    assert_eq!(
        std::mem::offset_of!(FjFilmProfileView, hanatos_surface_rgb),
        400
    );
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, asset_token), 416);
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, support), 424);
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, stage), 428);
    assert_eq!(std::mem::offset_of!(FjFilmProfileView, polarity), 432);
    assert_eq!(
        (
            std::mem::size_of::<FjPrintProfileView>(),
            std::mem::align_of::<FjPrintProfileView>()
        ),
        (144, 8)
    );
    assert_eq!(std::mem::offset_of!(FjPrintProfileView, stock), 0);
    assert_eq!(
        std::mem::offset_of!(FjPrintProfileView, viewing_illuminant),
        16
    );
    assert_eq!(std::mem::offset_of!(FjPrintProfileView, tables), 48);
    assert_eq!(std::mem::offset_of!(FjPrintProfileView, asset_token), 128);
    assert_eq!(std::mem::offset_of!(FjPrintProfileView, stage), 136);
    assert_eq!(
        (
            std::mem::size_of::<FjPrintDensityView>(),
            std::mem::align_of::<FjPrintDensityView>()
        ),
        (24, 8)
    );
    assert_eq!(std::mem::offset_of!(FjPrintDensityView, totals_cmy), 0);
    assert_eq!(std::mem::offset_of!(FjPrintDensityView, hash), 16);
    assert_eq!(
        (
            std::mem::size_of::<FjCatalogEntryView>(),
            std::mem::align_of::<FjCatalogEntryView>()
        ),
        (40, 8)
    );
    assert_eq!(std::mem::offset_of!(FjCatalogEntryView, key), 0);
    assert_eq!(std::mem::offset_of!(FjCatalogEntryView, label), 16);
    assert_eq!(std::mem::offset_of!(FjCatalogEntryView, polarity), 32);
    assert_eq!(FJ_ILLUMINANT_NAMED, 0);
    assert_eq!(FJ_ILLUMINANT_BLACKBODY, 1);
    assert_eq!(FJ_PROFILE_SUPPORT_FILM, 0);
    assert_eq!(FJ_PROFILE_SUPPORT_PAPER, 1);
    assert_eq!(FJ_PROFILE_STAGE_FILMING, 0);
    assert_eq!(FJ_PROFILE_STAGE_PRINTING, 1);
}

#[cfg(test)]
#[test]
fn production_profile_signatures() {
    let _: unsafe extern "C" fn(
        *const FjAssets,
        FjStringView,
        *mut *mut FjFilmProfile,
        *mut FjErrorBuffer,
    ) -> FjStatus = fj_legacy_film_profile_acquire;
    let _: unsafe extern "C" fn(
        *const FjAssets,
        FjStringView,
        *mut *mut FjPrintProfile,
        *mut FjErrorBuffer,
    ) -> FjStatus = fj_legacy_print_profile_acquire;
    let _: unsafe extern "C" fn(
        *const FjFilmProfile,
        *mut FjFilmProfileView,
        *mut FjErrorBuffer,
    ) -> FjStatus = fj_legacy_film_profile_view;
    let _: unsafe extern "C" fn(
        *const FjPrintProfile,
        *mut FjPrintProfileView,
        *mut FjErrorBuffer,
    ) -> FjStatus = fj_legacy_print_profile_view;
    let _: unsafe extern "C" fn(
        *const FjPrintProfile,
        f64,
        *mut *mut FjPrintDensityCurves,
        *mut FjErrorBuffer,
    ) -> FjStatus = fj_legacy_print_profile_sample_density;
    let _: unsafe extern "C" fn(
        *const FjPrintDensityCurves,
        *mut FjPrintDensityView,
        *mut FjErrorBuffer,
    ) -> FjStatus = fj_legacy_print_density_view;
    let _: unsafe extern "C" fn(*mut FjFilmProfile, *mut FjErrorBuffer) -> FjStatus =
        fj_legacy_film_profile_release;
    let _: unsafe extern "C" fn(*mut FjPrintProfile, *mut FjErrorBuffer) -> FjStatus =
        fj_legacy_print_profile_release;
    let _: unsafe extern "C" fn(*mut FjPrintDensityCurves, *mut FjErrorBuffer) -> FjStatus =
        fj_legacy_print_density_release;
    let _: unsafe extern "C" fn(*const FjAssets, *mut FjErrorBuffer) -> FjStatus =
        fj_legacy_assets_release_cached_payloads;
}

// FJ_TEMP_BRIDGE: spectral source conversion; remove S4.E.
#[repr(C)]
struct FjSpectraLut {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FjMallettBasis {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FjCmf {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FjSpectraLutView {
    samples: FjFloatSpan,
    asset_hash: u64,
}

#[cfg(any(test, feature = "test-support"))]
thread_local! { static SPECTRAL_FAULT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
#[cfg(feature = "test-support")]
static LIVE_SPECTRAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(any(test, feature = "test-support"))]
fn spectral_inject() -> Result<(), Failure> {
    match SPECTRAL_FAULT.replace(0) {
        1 => panic!("spectral projection fixture panic"),
        2 => Vec::<f32>::new()
            .try_reserve(usize::MAX)
            .map_err(Failure::Capacity),
        _ => Ok(()),
    }
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_spectral_fault(fault: u32) {
    SPECTRAL_FAULT.set(fault);
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_spectral_live_owners() -> usize {
    LIVE_SPECTRAL.load(std::sync::atomic::Ordering::Relaxed)
}

/// # Safety
/// Assets is live through return. Outputs/diagnostics are aligned, exclusive
/// and disjoint from all input/owner storage. Only success transfers ownership.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_hanatos_acquire(
    assets: *const FjAssets,
    out_owner: *mut *mut FjSpectraLut,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_owner.is_null() {
        // SAFETY: The caller authorizes one exclusive output slot.
        unsafe { out_owner.write(std::ptr::null_mut()) };
    }
    // SAFETY: Live matching owner and disjoint caller storage obey the contract.
    unsafe {
        run(error, || {
            if assets.is_null() || out_owner.is_null() {
                return Err(Failure::Input("NULL hanatos acquisition input/output"));
            }
            let source = (&*assets.cast::<Assets>())
                .hanatos()
                .map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            spectral_inject()?;
            let owner = Box::new(SpectraOwner::new(source));
            #[cfg(feature = "test-support")]
            LIVE_SPECTRAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_owner.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Assets is live through return. Outputs/diagnostics are aligned, exclusive
/// and disjoint from all input/owner storage. Only success transfers ownership.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_arctic_acquire(
    assets: *const FjAssets,
    out_owner: *mut *mut FjSpectraLut,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_owner.is_null() {
        // SAFETY: The caller authorizes one exclusive output slot.
        unsafe { out_owner.write(std::ptr::null_mut()) };
    }
    // SAFETY: Live matching owner and disjoint caller storage obey the contract.
    unsafe {
        run(error, || {
            if assets.is_null() || out_owner.is_null() {
                return Err(Failure::Input("NULL arctic acquisition input/output"));
            }
            let source = (&*assets.cast::<Assets>())
                .arctic()
                .map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            spectral_inject()?;
            let owner = Box::new(SpectraOwner::new(source));
            #[cfg(feature = "test-support")]
            LIVE_SPECTRAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_owner.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Assets is live through return. Outputs/diagnostics are aligned, exclusive
/// and disjoint from all input/owner storage. Only success transfers ownership.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_mallett_acquire(
    assets: *const FjAssets,
    out_owner: *mut *mut FjMallettBasis,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_owner.is_null() {
        // SAFETY: The caller authorizes one exclusive output slot.
        unsafe { out_owner.write(std::ptr::null_mut()) };
    }
    // SAFETY: Live matching owner and disjoint caller storage obey the contract.
    unsafe {
        run(error, || {
            if assets.is_null() || out_owner.is_null() {
                return Err(Failure::Input("NULL mallett acquisition input/output"));
            }
            let source = (&*assets.cast::<Assets>())
                .mallett()
                .map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            spectral_inject()?;
            let owner = Box::new(MallettOwner::new(source));
            #[cfg(feature = "test-support")]
            LIVE_SPECTRAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_owner.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Assets is live through return. Outputs/diagnostics are aligned, exclusive
/// and disjoint from all input/owner storage. Only success transfers ownership.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_cmf_acquire(
    assets: *const FjAssets,
    out_owner: *mut *mut FjCmf,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_owner.is_null() {
        // SAFETY: The caller authorizes one exclusive output slot.
        unsafe { out_owner.write(std::ptr::null_mut()) };
    }
    // SAFETY: Live matching owner and disjoint caller storage obey the contract.
    unsafe {
        run(error, || {
            if assets.is_null() || out_owner.is_null() {
                return Err(Failure::Input("NULL cmf acquisition input/output"));
            }
            let source = (&*assets.cast::<Assets>()).cmf().map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            spectral_inject()?;
            let owner = Box::new(CmfOwner::new(source));
            #[cfg(feature = "test-support")]
            LIVE_SPECTRAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_owner.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// The live matching owner remains held through every returned-view use.
/// Output/diagnostics are aligned, exclusive and disjoint. Release excludes reads.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_spectra_lut_view(
    owner: *const FjSpectraLut,
    out_view: *mut FjSpectraLutView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_view.is_null() {
        // SAFETY: Nullable pointers and integer fields have all-zero empty values.
        unsafe { out_view.write(std::mem::zeroed()) };
    }
    // SAFETY: Matching owner and authorized disjoint output/diagnostic storage.
    unsafe {
        run(error, || {
            if owner.is_null() || out_view.is_null() {
                return Err(Failure::Input("NULL spectra_lut view input/output"));
            }
            #[cfg(any(test, feature = "test-support"))]
            spectral_inject()?;
            let view = (&*owner.cast::<SpectraOwner>()).view();
            out_view.write(FjSpectraLutView {
                samples: floats(view.samples),
                asset_hash: view.asset_hash,
            });
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// owner is NULL or a live matching allocation consumed once. Caller excludes
/// all operations and outstanding borrows; diagnostics obey the disjoint contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_spectra_lut_release(
    owner: *mut FjSpectraLut,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let taken = if owner.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        LIVE_SPECTRAL.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Matching Box is taken exactly once before diagnostic validation.
        Some(unsafe { Box::from_raw(owner.cast::<SpectraOwner>()) })
    };
    // SAFETY: Captured owner drops within containment even on rejected diagnostics.
    unsafe {
        run(error, move || {
            drop(taken);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// The live matching owner remains held through every returned-view use.
/// Output/diagnostics are aligned, exclusive and disjoint. Release excludes reads.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_mallett_view(
    owner: *const FjMallettBasis,
    out_view: *mut FjFloatSpan,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_view.is_null() {
        // SAFETY: Nullable pointers and integer fields have all-zero empty values.
        unsafe { out_view.write(std::mem::zeroed()) };
    }
    // SAFETY: Matching owner and authorized disjoint output/diagnostic storage.
    unsafe {
        run(error, || {
            if owner.is_null() || out_view.is_null() {
                return Err(Failure::Input("NULL mallett view input/output"));
            }
            #[cfg(any(test, feature = "test-support"))]
            spectral_inject()?;
            out_view.write(floats(
                (&*owner.cast::<MallettOwner>()).samples().as_flattened(),
            ));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// owner is NULL or a live matching allocation consumed once. Caller excludes
/// all operations and outstanding borrows; diagnostics obey the disjoint contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_mallett_release(
    owner: *mut FjMallettBasis,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let taken = if owner.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        LIVE_SPECTRAL.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Matching Box is taken exactly once before diagnostic validation.
        Some(unsafe { Box::from_raw(owner.cast::<MallettOwner>()) })
    };
    // SAFETY: Captured owner drops within containment even on rejected diagnostics.
    unsafe {
        run(error, move || {
            drop(taken);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// The live matching owner remains held through every returned-view use.
/// Output/diagnostics are aligned, exclusive and disjoint. Release excludes reads.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_cmf_view(
    owner: *const FjCmf,
    out_view: *mut FjFloatSpan,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_view.is_null() {
        // SAFETY: Nullable pointers and integer fields have all-zero empty values.
        unsafe { out_view.write(std::mem::zeroed()) };
    }
    // SAFETY: Matching owner and authorized disjoint output/diagnostic storage.
    unsafe {
        run(error, || {
            if owner.is_null() || out_view.is_null() {
                return Err(Failure::Input("NULL cmf view input/output"));
            }
            #[cfg(any(test, feature = "test-support"))]
            spectral_inject()?;
            out_view.write(floats((&*owner.cast::<CmfOwner>()).rows().as_flattened()));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// owner is NULL or a live matching allocation consumed once. Caller excludes
/// all operations and outstanding borrows; diagnostics obey the disjoint contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_cmf_release(
    owner: *mut FjCmf,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let taken = if owner.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        LIVE_SPECTRAL.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Matching Box is taken exactly once before diagnostic validation.
        Some(unsafe { Box::from_raw(owner.cast::<CmfOwner>()) })
    };
    // SAFETY: Captured owner drops within containment even on rejected diagnostics.
    unsafe {
        run(error, move || {
            drop(taken);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

#[cfg(test)]
mod spectral_tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};
    #[test]
    fn production_spectral_signatures() {
        let _: unsafe extern "C" fn(
            *const FjAssets,
            *mut *mut FjSpectraLut,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_hanatos_acquire;
        let _: unsafe extern "C" fn(
            *const FjAssets,
            *mut *mut FjSpectraLut,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_arctic_acquire;
        let _: unsafe extern "C" fn(
            *const FjSpectraLut,
            *mut FjSpectraLutView,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_spectra_lut_view;
        let _: unsafe extern "C" fn(*mut FjSpectraLut, *mut FjErrorBuffer) -> FjStatus =
            fj_legacy_spectra_lut_release;
        let _: unsafe extern "C" fn(
            *const FjAssets,
            *mut *mut FjMallettBasis,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_mallett_acquire;
        let _: unsafe extern "C" fn(
            *const FjMallettBasis,
            *mut FjFloatSpan,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_mallett_view;
        let _: unsafe extern "C" fn(*mut FjMallettBasis, *mut FjErrorBuffer) -> FjStatus =
            fj_legacy_mallett_release;
        let _: unsafe extern "C" fn(
            *const FjAssets,
            *mut *mut FjCmf,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_cmf_acquire;
        let _: unsafe extern "C" fn(
            *const FjCmf,
            *mut FjFloatSpan,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_cmf_view;
        let _: unsafe extern "C" fn(*mut FjCmf, *mut FjErrorBuffer) -> FjStatus =
            fj_legacy_cmf_release;
    }
    #[test]
    fn layout_and_source_error_categories() {
        assert_eq!(
            (size_of::<FjFloatSpan>(), align_of::<FjFloatSpan>()),
            (16, 8)
        );
        assert_eq!(
            (
                offset_of!(FjFloatSpan, data),
                offset_of!(FjFloatSpan, count)
            ),
            (0, 8)
        );
        assert_eq!(
            (
                size_of::<FjSpectraLutView>(),
                align_of::<FjSpectraLutView>()
            ),
            (24, 8)
        );
        assert_eq!(
            (
                offset_of!(FjSpectraLutView, samples),
                offset_of!(FjSpectraLutView, asset_hash)
            ),
            (0, 16)
        );
        for kind in [
            ReadErrorKind::Capacity,
            ReadErrorKind::Size,
            ReadErrorKind::Open(std::io::ErrorKind::NotFound),
            ReadErrorKind::Io {
                part: film_juicer_core::data_io::ReadPart::Payload,
                kind: std::io::ErrorKind::UnexpectedEof,
            },
            ReadErrorKind::ShortRead(film_juicer_core::data_io::ReadPart::Header),
            ReadErrorKind::InvalidNpy {
                expected: "fixture",
                reason: "dtype",
            },
            ReadErrorKind::CsvLineTooLong { line: 1 },
        ] {
            for cmf in [false, true] {
                let error = std::sync::Arc::new(film_juicer_core::data_io::ReadError {
                    path: "spectral-source".into(),
                    kind,
                });
                let failure = Failure::Asset(if cmf {
                    AssetError::Cmf(error)
                } else {
                    AssetError::Reconstruction(error)
                });
                assert_eq!(
                    failure.category(),
                    if kind == ReadErrorKind::Capacity {
                        FJ_STATUS_ALLOCATION_FAILURE
                    } else {
                        FJ_STATUS_PREPARATION_FAILURE
                    }
                );
            }
        }
    }
}

// FJ_TEMP_BRIDGE: CSV source conversion and selected calibration; remove S4.E.
const FJ_CSV_D65: u32 = 1;
const FJ_CSV_D55: u32 = 2;
const FJ_CSV_D50: u32 = 3;
const FJ_CSV_T: u32 = 4;
const FJ_CSV_K75P: u32 = 5;
const FJ_CSV_KG3: u32 = 6;
const FJ_CSV_CANON_24_F28_IS: u32 = 7;
const FJ_CALIBRATION_FOUND: u32 = 1;
const FJ_CALIBRATION_MISSING_FILE: u32 = 2;
const FJ_CALIBRATION_MISSING_ENTRY: u32 = 3;
const FJ_CALIBRATION_MALFORMED: u32 = 4;
const FJ_CALIBRATION_FIELD_NONE: u32 = 0;
const FJ_CALIBRATION_FIELD_RESOURCE_READ: u32 = 1;
const FJ_CALIBRATION_FIELD_ROOT: u32 = 2;
const FJ_CALIBRATION_FIELD_PRINT_PROFILE: u32 = 3;
const FJ_CALIBRATION_FIELD_PRINT_ILLUMINANT: u32 = 4;
const FJ_CALIBRATION_FIELD_CMY_CC: u32 = 5;

#[repr(C)]
struct FjCsvPairs {
    _opaque: [u8; 0],
}
#[repr(C)]
#[derive(Default)]
struct FjNeutralCalibrationResult {
    outcome: u32,
    field: u32,
    cmy_cc: [f32; 3],
}
const _: () = {
    assert!(size_of::<FjNeutralCalibrationResult>() == 20);
    assert!(align_of::<FjNeutralCalibrationResult>() == 4);
    assert!(std::mem::offset_of!(FjNeutralCalibrationResult, outcome) == 0);
    assert!(std::mem::offset_of!(FjNeutralCalibrationResult, field) == 4);
    assert!(std::mem::offset_of!(FjNeutralCalibrationResult, cmy_cc) == 8);
};
fn csv_source(tag: u32) -> Result<CsvSource, Failure> {
    match tag {
        FJ_CSV_D65 => Ok(CsvSource::D65),
        FJ_CSV_D55 => Ok(CsvSource::D55),
        FJ_CSV_D50 => Ok(CsvSource::D50),
        FJ_CSV_T => Ok(CsvSource::T),
        FJ_CSV_K75P => Ok(CsvSource::K75p),
        FJ_CSV_KG3 => Ok(CsvSource::Kg3),
        FJ_CSV_CANON_24_F28_IS => Ok(CsvSource::Canon24F28Is),
        _ => Err(Failure::Input("unsupported CSV source tag")),
    }
}
impl FjNeutralCalibrationResult {
    fn from_lookup(result: CalibrationLookup) -> Self {
        let mut raw = Self {
            field: FJ_CALIBRATION_FIELD_NONE,
            ..Self::default()
        };
        match result {
            CalibrationLookup::Found(cmy_cc) => {
                raw.outcome = FJ_CALIBRATION_FOUND;
                raw.cmy_cc = cmy_cc;
            }
            CalibrationLookup::MissingFile => raw.outcome = FJ_CALIBRATION_MISSING_FILE,
            CalibrationLookup::MissingEntry => raw.outcome = FJ_CALIBRATION_MISSING_ENTRY,
            CalibrationLookup::Malformed(field) => {
                raw.outcome = FJ_CALIBRATION_MALFORMED;
                raw.field = match field {
                    CalibrationField::ResourceRead => FJ_CALIBRATION_FIELD_RESOURCE_READ,
                    CalibrationField::Root => FJ_CALIBRATION_FIELD_ROOT,
                    CalibrationField::PrintProfile => FJ_CALIBRATION_FIELD_PRINT_PROFILE,
                    CalibrationField::PrintIlluminant => FJ_CALIBRATION_FIELD_PRINT_ILLUMINANT,
                    CalibrationField::CmyCc => FJ_CALIBRATION_FIELD_CMY_CC,
                };
            }
        }
        raw
    }
}
#[cfg(feature = "test-support")]
static LIVE_CSV: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_csv_live_owners() -> usize {
    LIVE_CSV.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "test-support")]
thread_local! { static CSV_FAULT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_csv_fault(fault: u32) {
    CSV_FAULT.set(fault);
}
#[cfg(feature = "test-support")]
fn csv_inject(source: CsvSource) -> Result<(), Failure> {
    match CSV_FAULT.replace(0) {
        1 => panic!("CSV boundary fixture panic"),
        2 => Err(Failure::Asset(AssetError::CsvSource {
            source,
            error: film_juicer_core::data_io::ReadError {
                path: "CSV capacity fixture".into(),
                kind: ReadErrorKind::Capacity,
            },
        })),
        3 => Err(Failure::Asset(AssetError::CsvCachePoisoned { source })),
        _ => Ok(()),
    }
}

/// # Safety
/// Assets stays live through return; outputs/diagnostics are aligned, exclusive
/// and disjoint. Only success transfers one owner requiring matching release.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_csv_acquire(
    assets: *const FjAssets,
    source: u32,
    out_pairs: *mut *mut FjCsvPairs,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_pairs.is_null() {
        // SAFETY: One valid caller-authorized output slot, cleared before validation.
        unsafe { out_pairs.write(std::ptr::null_mut()) };
    }
    // SAFETY: Live owner and disjoint authorized call storage obey the contract.
    unsafe {
        run(error, || {
            if assets.is_null() || out_pairs.is_null() {
                return Err(Failure::Input("NULL CSV acquisition input/output"));
            }
            let source = csv_source(source)?;
            #[cfg(feature = "test-support")]
            csv_inject(source)?;
            let rows = (&*assets.cast::<Assets>())
                .csv_source(source)
                .map_err(Failure::Asset)?;
            let owner = Box::new(CsvPairsOwner::new(rows));
            #[cfg(feature = "test-support")]
            LIVE_CSV.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_pairs.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// The matching owner remains held through view use; release excludes reads.
/// Output/diagnostics are aligned, exclusive and disjoint from all owner storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_csv_view(
    pairs: *const FjCsvPairs,
    out_rows: *mut FjFloatSpan,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_rows.is_null() {
        // SAFETY: Pointer/count have all-zero empty representations.
        unsafe { out_rows.write(std::mem::zeroed()) };
    }
    // SAFETY: Matching live owner and caller-authorized output/diagnostic storage.
    unsafe {
        run(error, || {
            if pairs.is_null() || out_rows.is_null() {
                return Err(Failure::Input("NULL CSV view input/output"));
            }
            let rows = (&*pairs.cast::<CsvPairsOwner>()).rows();
            if rows.len() > isize::MAX as usize / (2 * size_of::<f32>()) {
                return Err(Failure::Input("CSV span exceeds addressable bytes"));
            }
            out_rows.write(floats(rows.as_flattened()));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// Pairs is NULL or one live matching owner, consumed once with all reads excluded.
/// Diagnostics obey the shared exclusive/disjoint storage contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_csv_release(
    pairs: *mut FjCsvPairs,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let taken = if pairs.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        LIVE_CSV.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Matching Box taken exactly once before diagnostic validation.
        Some(unsafe { Box::from_raw(pairs.cast::<CsvPairsOwner>()) })
    };
    // SAFETY: Captured owner drops inside containment even with bad diagnostics.
    unsafe {
        run(error, move || {
            drop(taken);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// Assets and initialized key extents remain live through return. Output and
/// diagnostics are aligned, exclusive and disjoint from inputs and each other.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_neutral_calibration_lookup(
    assets: *const FjAssets,
    print_stock: FjStringView,
    illuminant: FjStringView,
    film_stock: FjStringView,
    out_result: *mut FjNeutralCalibrationResult,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_result.is_null() {
        // SAFETY: Caller authorizes one complete output, cleared before validation.
        unsafe { out_result.write(FjNeutralCalibrationResult::default()) };
    }
    // SAFETY: Live Assets and initialized disjoint key/output/diagnostic extents.
    unsafe {
        run(error, || {
            if assets.is_null() || out_result.is_null() {
                return Err(Failure::Input("NULL calibration input/output"));
            }
            let result = asset_calibration::lookup(
                &*assets.cast::<Assets>(),
                key(print_stock)?,
                key(illuminant)?,
                key(film_stock)?,
            )
            .map_err(Failure::Asset)?;
            out_result.write(FjNeutralCalibrationResult::from_lookup(result));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_calibration_read_fault(mode: u32) {
    use film_juicer_core::data_io::calibration::test_support;
    use std::io::ErrorKind::{OutOfMemory, PermissionDenied};
    match mode {
        1 => test_support::set_read_fault(OutOfMemory, Ok(true)),
        2 => test_support::set_read_fault(OutOfMemory, Ok(false)),
        3 => test_support::set_read_fault(OutOfMemory, Err(PermissionDenied)),
        4 => test_support::set_read_fault(PermissionDenied, Err(OutOfMemory)),
        5 => test_support::set_read_fault(PermissionDenied, Ok(true)),
        6 => test_support::set_read_fault(PermissionDenied, Ok(false)),
        7 => test_support::set_read_fault(PermissionDenied, Err(PermissionDenied)),
        _ => test_support::clear_read_fault(),
    }
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_calibration_reads() -> usize {
    film_juicer_core::data_io::calibration::test_support::counts().0
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_calibration_probes() -> usize {
    film_juicer_core::data_io::calibration::test_support::counts().1
}

#[cfg(test)]
mod illuminant_calibration_tests {
    use super::*;
    #[test]
    fn source_categories_and_signatures() {
        use film_juicer_core::data_io::ReadError;
        for (kind, category) in [
            (ReadErrorKind::Capacity, FJ_STATUS_ALLOCATION_FAILURE),
            (
                ReadErrorKind::Open(std::io::ErrorKind::NotFound),
                FJ_STATUS_PREPARATION_FAILURE,
            ),
        ] {
            assert_eq!(
                Failure::Asset(AssetError::CsvSource {
                    source: CsvSource::D65,
                    error: ReadError {
                        path: "csv-source".into(),
                        kind
                    }
                })
                .category(),
                category
            );
        }
        assert_eq!(
            Failure::Asset(AssetError::CsvCachePoisoned {
                source: CsvSource::D65
            })
            .category(),
            FJ_STATUS_INTERNAL_FAILURE
        );
        let _: unsafe extern "C" fn(
            *const FjAssets,
            u32,
            *mut *mut FjCsvPairs,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_csv_acquire;
        let _: unsafe extern "C" fn(
            *const FjCsvPairs,
            *mut FjFloatSpan,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_csv_view;
        let _: unsafe extern "C" fn(*mut FjCsvPairs, *mut FjErrorBuffer) -> FjStatus =
            fj_legacy_csv_release;
        let _: unsafe extern "C" fn(
            *const FjAssets,
            FjStringView,
            FjStringView,
            FjStringView,
            *mut FjNeutralCalibrationResult,
            *mut FjErrorBuffer,
        ) -> FjStatus = fj_legacy_neutral_calibration_lookup;
    }
}

// Raw noise projection is shared only by the production and fixture edges.
use crate::asset_noise::{NoiseOwner, NoiseView};
use crate::cuda::sys::{FjByteSpan, FjStaticNoise};
use film_juicer_core::assets::NoiseBundle;
use std::sync::Arc;
use film_juicer_core::data_io::noise;

#[repr(C)]
pub(crate) struct FjNoise {
    _opaque: [u8; 0],
}

pub(crate) fn noise_category(kind: noise::ErrorKind) -> u32 {
    match kind {
        noise::ErrorKind::Capacity
        | noise::ErrorKind::Open(std::io::ErrorKind::OutOfMemory)
        | noise::ErrorKind::Read(std::io::ErrorKind::OutOfMemory) => FJ_STATUS_ALLOCATION_FAILURE,
        noise::ErrorKind::Missing
        | noise::ErrorKind::Open(_)
        | noise::ErrorKind::Read(_)
        | noise::ErrorKind::ShortRead
        | noise::ErrorKind::Length { .. }
        | noise::ErrorKind::Size
        | noise::ErrorKind::Json
        | noise::ErrorKind::Metadata
        | noise::ErrorKind::Dimensions => FJ_STATUS_PREPARATION_FAILURE,
    }
}

fn noise_projection(view: NoiseView<'_>) -> FjStaticNoise {
    let bytes = |samples: &[u8]| FjByteSpan {
        data: samples.as_ptr(),
        count: samples.len(),
    };
    // The complete core representation has fixed dimensions bounded by i32.
    FjStaticNoise {
        stbn: bytes(view.stbn),
        stbn_width: view.stbn_dimensions[0] as i32,
        stbn_height: view.stbn_dimensions[1] as i32,
        stbn_frames: view.stbn_dimensions[2] as i32,
        wang_tiles: bytes(view.wang_tiles),
        wang_lut: bytes(view.wang_lut),
        wang_width: view.wang_dimensions[0] as i32,
        wang_height: view.wang_dimensions[1] as i32,
        wang_tile_count: view.wang_dimensions[2],
        wang_colors: view.wang_colors as i32,
    }
}

#[cfg(feature = "test-support")]
thread_local! {
    static NOISE_ACQUISITIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static NOISE_FAULT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}
#[cfg(feature = "test-support")]
static LIVE_NOISE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// # Safety
/// Output/diagnostics are exclusive aligned disjoint authorized storage.
/// The operation borrows its inputs only until return and publishes complete data.
pub(crate) unsafe fn acquire_noise(
    out_owner: *mut *mut FjNoise,
    error: *mut FjErrorBuffer,
    acquire: impl FnOnce() -> Result<Arc<NoiseBundle>, Failure>,
) -> FjStatus {
    if !out_owner.is_null() {
        // SAFETY: Authorized output slot is cleared before all other validation.
        unsafe { out_owner.write(std::ptr::null_mut()) };
    }
    // SAFETY: Call storage obeys the common foreign diagnostic contract.
    unsafe {
        run(error, || {
            if out_owner.is_null() {
                return Err(Failure::Input("NULL noise acquisition output"));
            }
            #[cfg(feature = "test-support")]
            if NOISE_FAULT.with(|fault| fault.get()) == 1 {
                panic!("noise acquisition fixture panic");
            }
            #[cfg(feature = "test-support")]
            NOISE_ACQUISITIONS.with(|value| value.set(value.get() + 1));
            let owner = Box::new(NoiseOwner::new(acquire()?));
            #[cfg(feature = "test-support")]
            LIVE_NOISE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_owner.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Matching live owner remains held through all view use; output/diagnostics
/// are exclusive aligned disjoint storage. Release excludes all readers.
pub(crate) unsafe fn view_noise(
    owner: *const FjNoise,
    out_view: *mut FjStaticNoise,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_view.is_null() {
        // SAFETY: Numeric and nullable span fields have all-zero empty values.
        unsafe { out_view.write(std::mem::zeroed()) };
    }
    // SAFETY: Live owner and caller-authorized output/diagnostics obey the contract.
    unsafe {
        run(error, || {
            if owner.is_null() || out_view.is_null() {
                return Err(Failure::Input("NULL noise view input/output"));
            }
            #[cfg(feature = "test-support")]
            if NOISE_FAULT.with(|fault| fault.get()) == 2 {
                panic!("noise view fixture panic");
            }
            out_view.write(noise_projection((&*owner.cast::<NoiseOwner>()).view()));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Owner is NULL or one matching live handle consumed once, with all reads and
/// borrows excluded. Diagnostic storage obeys the common disjoint contract.
pub(crate) unsafe fn release_noise(owner: *mut FjNoise, error: *mut FjErrorBuffer) -> FjStatus {
    let taken = if owner.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        LIVE_NOISE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: The matching acquired Box is taken exactly once before validation.
        Some(unsafe { Box::from_raw(owner.cast::<NoiseOwner>()) })
    };
    // SAFETY: Captured cleanup occurs inside containment, even on malformed diagnostics.
    unsafe {
        run(error, move || {
            drop(taken);
            #[cfg(feature = "test-support")]
            if NOISE_FAULT.with(|fault| fault.get()) == 3 {
                panic!("noise release fixture panic");
            }
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Assets is live through return; output/diagnostics are aligned, exclusive and
/// disjoint. Only success transfers a complete source owner requiring release.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_noise_acquire(
    assets: *const FjAssets,
    out_owner: *mut *mut FjNoise,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared implementation contains the operation and validates outputs.
    unsafe {
        acquire_noise(out_owner, error, || {
            if assets.is_null() {
                return Err(Failure::Input("NULL noise Assets"));
            }
            (&*assets.cast::<Assets>()).noise().map_err(Failure::Asset)
        })
    }
}
/// # Safety
/// Matching owner is live through view use; aligned exclusive outputs and
/// diagnostics are disjoint from each other and all owner storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_noise_view(
    owner: *const FjNoise,
    out_view: *mut FjStaticNoise,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Identical owner/borrow/storage obligations to the shared operation.
    unsafe { view_noise(owner, out_view, error) }
}
/// # Safety
/// Matching live owner is consumed once, with every read/borrow excluded.
/// Diagnostics follow the common exclusive/disjoint contract; NULL is allowed.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_noise_release(
    owner: *mut FjNoise,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared cleanup consumes even on diagnostic failure or panic.
    unsafe { release_noise(owner, error) }
}

#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_noise_fault(fault: u32) {
    NOISE_FAULT.with(|value| value.set(fault));
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_live_noise_owners() -> usize {
    LIVE_NOISE.load(std::sync::atomic::Ordering::Relaxed)
}

const _: unsafe extern "C" fn(*const FjAssets, *mut *mut FjNoise, *mut FjErrorBuffer) -> FjStatus =
    fj_legacy_noise_acquire;
const _: unsafe extern "C" fn(*const FjNoise, *mut FjStaticNoise, *mut FjErrorBuffer) -> FjStatus =
    fj_legacy_noise_view;
const _: unsafe extern "C" fn(*mut FjNoise, *mut FjErrorBuffer) -> FjStatus =
    fj_legacy_noise_release;

#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_noise_acquisition_count() -> usize {
    NOISE_ACQUISITIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
mod noise_tests {
    use super::*;
    #[test]
    fn explicit_noise_statuses_do_not_define_cache_retry() {
        for kind in [
            noise::ErrorKind::Capacity,
            noise::ErrorKind::Open(std::io::ErrorKind::OutOfMemory),
            noise::ErrorKind::Read(std::io::ErrorKind::OutOfMemory),
        ] {
            assert_eq!(noise_category(kind), FJ_STATUS_ALLOCATION_FAILURE);
        }
        for kind in [
            noise::ErrorKind::Missing,
            noise::ErrorKind::ShortRead,
            noise::ErrorKind::Json,
            noise::ErrorKind::Metadata,
            noise::ErrorKind::Dimensions,
            noise::ErrorKind::Size,
            noise::ErrorKind::Length {
                expected: 1,
                actual: 0,
            },
            noise::ErrorKind::Open(std::io::ErrorKind::PermissionDenied),
            noise::ErrorKind::Read(std::io::ErrorKind::UnexpectedEof),
        ] {
            assert_eq!(noise_category(kind), FJ_STATUS_PREPARATION_FAILURE);
        }
        assert_eq!(
            Failure::Asset(AssetError::NoiseCachePoisoned).category(),
            FJ_STATUS_INTERNAL_FAILURE
        );
    }
}
