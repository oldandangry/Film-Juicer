//! Feature-only C fixture facade. Raw pointers never enter the safe core.

use crate::asset_illuminant::{self, FjIlluminant, FjIlluminantCoverage};
use film_juicer_core::illuminant;
use crate::scanner_bridge;

thread_local! { static ILLUMINANT_FACADE_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[unsafe(no_mangle)]
extern "C" fn fj_test_illuminant_arm_fault(
    operation: u32,
    call_index: u32,
    fault: u32,
) -> FjStatus {
    asset_illuminant::arm_fault(operation, call_index, fault)
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_illuminant_clear_fault() {
    asset_illuminant::clear_fault();
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_illuminant_live_lenses() -> usize {
    asset_illuminant::live_lenses()
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_illuminant_arm_facade_fault() {
    ILLUMINANT_FACADE_FAULT.set(true);
}
/// # Safety
/// Output authorizes ten exclusive aligned usize elements, with no overlapping call storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_illuminant_scratch_capacities(out: *mut usize) -> usize {
    if out.is_null() || !(out as usize).is_multiple_of(align_of::<usize>()) {
        return 0;
    }
    let (bytes, count) = illuminant::test_support::scratch_capacities();
    // SAFETY: The caller retains ten writable aligned elements through return.
    unsafe {
        out.copy_from_nonoverlapping(bytes.as_ptr(), 10);
    }
    count
}
/// # Safety
/// Required storage follows the illuminant header; rows are immutable until return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_illuminant_curve(
    operation: u32,
    source: FjFloatSpan,
    lens: FjFloatSpan,
    temperature: f32,
    out: *mut FjIlluminant,
    coverage: *mut FjIlluminantCoverage,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Only aligned, caller-authorized outputs are cleared or published;
    // the shared edge checks diagnostics/extents, and the facade invokes core directly.
    unsafe {
        let curve_valid =
            !out.is_null() && (out as usize).is_multiple_of(align_of::<FjIlluminant>());
        let coverage_valid = !coverage.is_null()
            && (coverage as usize).is_multiple_of(align_of::<FjIlluminantCoverage>());
        if curve_valid {
            out.write(FjIlluminant::default());
        }
        if coverage_valid {
            coverage.write(FjIlluminantCoverage::default());
        }
        asset_illuminant::run(error, || {
            if !curve_valid || !coverage_valid {
                return Err(crate::asset_bridge::Failure::Input(
                    "invalid fixture curve output",
                ));
            }
            let source = asset_illuminant::rows(source)?;
            let lens = asset_illuminant::rows(lens)?;
            if ILLUMINANT_FACADE_FAULT.replace(false) {
                return Err(crate::asset_bridge::Failure::Internal(
                    "illuminant fixture-only fault",
                ));
            }
            let result = match operation {
                1 => illuminant::from_samples(source)
                    .map(|curve| (curve, illuminant::Coverage::default())),
                2 => Ok((
                    illuminant::blackbody(temperature),
                    illuminant::Coverage::default(),
                )),
                3 => Ok((illuminant::equal_energy(), illuminant::Coverage::default())),
                4 => illuminant::tungsten_kg3(source),
                5 => illuminant::prepare_lens(source)
                    .and_then(|(input, _)| illuminant::finish_lens(input, lens)),
                8 => illuminant::test_support::resample(source)
                    .map(|curve| (curve, illuminant::Coverage::default())),
                _ => {
                    return Err(crate::asset_bridge::Failure::Input(
                        "invalid fixture illuminant operation",
                    ));
                }
            };
            match result {
                Ok((samples, metadata)) => {
                    coverage.write(metadata.into());
                    out.write(FjIlluminant { samples });
                    Ok(())
                }
                Err(error) => {
                    coverage.write(error.coverage.into());
                    Err(crate::asset_bridge::Failure::Illuminant(error))
                }
            }
        })
    }
}

use std::fmt::{self, Write};
use std::mem::{align_of, offset_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use film_juicer_core::assets::{AssetError, Assets};
use film_juicer_core::profile::{Antihalation, ProfileCompletionErrorKind, ProfileUse};

use crate::asset_profile::{FilmOwner, FilmFixtureView};
use film_juicer_core::color;
use crate::legacy_bridge::{FjInputColorMatrices, FjInputColorConversion, input_space};
use crate::cuda::sys::{FjErrorBuffer, FjFloatSpan, FjStatus, FjStringView};
use crate::cuda::sys::{
    FJ_API_NONE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE,
    FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT,
};

#[repr(C)]
struct FjFilmProfile {
    _opaque: [u8; 0],
}

/// # Safety
/// Matching live profile; initialized, aligned, exclusive and disjoint outputs/diagnostics.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_development_acquire(
    profile: *const FjFilmProfile,
    requested_route: u32,
    gamma_factor: f32,
    layers_required: u32,
    out: *mut *mut crate::film_bridge::FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Same internal constructor and raw lifetime contract, without calling legacy exports.
    unsafe {
        crate::film_bridge::acquire(
            profile.cast(),
            requested_route,
            gamma_factor,
            layers_required,
            out,
            error,
        )
    }
}
/// # Safety
/// Matching live owner; initialized, aligned and disjoint output/diagnostics; views expire on release.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_development_view(
    owner: *const crate::film_bridge::FjFilmDevelopmentOwner,
    out: *mut crate::film_bridge::FjFilmDevelopmentView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Same internal borrow projection; no legacy-export dependency.
    unsafe { crate::film_bridge::view(owner, out, error) }
}
/// # Safety
/// NULL or a matching owner consumed once; release excludes all views; disjoint diagnostics.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_development_release(
    owner: *mut crate::film_bridge::FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Same internal consume-once release, including failure/expiry behavior.
    unsafe { crate::film_bridge::release(owner, error) }
}

/// # Safety
/// Live completed development; initialized/disjoint output and diagnostics.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_development_retain(
    owner: *const crate::film_bridge::FjFilmDevelopmentOwner,
    out: *mut *mut crate::film_bridge::FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade invokes the same internal owner operation, not an export.
    unsafe { crate::film_bridge::retain(owner, out, error) }
}

const FJ_PROFILE_USE_STILL: u32 = 0;
const FJ_PROFILE_USE_CINE: u32 = 1;
const FJ_PROFILE_ANTIHALATION_STRONG: u32 = 0;
const FJ_PROFILE_ANTIHALATION_WEAK: u32 = 1;
const FJ_PROFILE_ANTIHALATION_NO: u32 = 2;

#[repr(C)]
struct FjDoubleSpan {
    data: *const f64,
    count: usize,
}

#[repr(C)]
struct FjFilmFixtureView {
    r#use: u32,
    antihalation: u32,
    asset_token: u64,
    halation_first_sigma_um: [f32; 3],
    halation_primary_amount: [f32; 3],
    source_log_exposure: FjDoubleSpan,
    log_exposure: FjFloatSpan,
    density_curves_cmy: FjFloatSpan,
    density_curves_layers: [[FjFloatSpan; 3]; 3],
    channel_density_cmy: FjFloatSpan,
    base_density: FjFloatSpan,
}

impl FjFilmFixtureView {
    fn empty() -> Self {
        let empty = floats(&[]);
        Self {
            r#use: 0,
            antihalation: 0,
            asset_token: 0,
            halation_first_sigma_um: [0.0; 3],
            halation_primary_amount: [0.0; 3],
            source_log_exposure: FjDoubleSpan {
                data: std::ptr::null(),
                count: 0,
            },
            log_exposure: empty,
            density_curves_cmy: empty,
            density_curves_layers: [[empty; 3]; 3],
            channel_density_cmy: empty,
            base_density: empty,
        }
    }

    // Raw projection stays private to this foreign boundary. The export's caller
    // retains the owner through every subsequent use of the returned spans.
    fn from_view(view: FilmFixtureView<'_>) -> Self {
        Self {
            r#use: match view.usage {
                ProfileUse::Still => FJ_PROFILE_USE_STILL,
                ProfileUse::Cine => FJ_PROFILE_USE_CINE,
            },
            antihalation: match view.antihalation {
                Antihalation::Strong => FJ_PROFILE_ANTIHALATION_STRONG,
                Antihalation::Weak => FJ_PROFILE_ANTIHALATION_WEAK,
                Antihalation::No => FJ_PROFILE_ANTIHALATION_NO,
            },
            asset_token: view.asset_token,
            halation_first_sigma_um: view.halation_first_sigma_um,
            halation_primary_amount: view.halation_primary_amount,
            // Fixed fixture ABI names map authored and interpolation representations.
            source_log_exposure: FjDoubleSpan {
                data: if view.authored_log_exposure.is_empty() {
                    std::ptr::null()
                } else {
                    view.authored_log_exposure.as_ptr()
                },
                count: view.authored_log_exposure.len(),
            },
            log_exposure: floats(view.interpolation_log_exposure),
            density_curves_cmy: floats(view.density_curves_cmy),
            density_curves_layers: view.density_curves_layers.map(|layer| layer.map(floats)),
            channel_density_cmy: floats(view.channel_density_cmy),
            base_density: floats(view.base_density),
        }
    }
}

fn floats(samples: &[f32]) -> FjFloatSpan {
    FjFloatSpan {
        data: if samples.is_empty() {
            std::ptr::null()
        } else {
            samples.as_ptr()
        },
        count: samples.len(),
    }
}

enum Failure {
    Input(&'static str),
    Asset(AssetError),
    Panic,
}

impl Failure {
    fn category(&self) -> u32 {
        match self {
            Self::Input(_) => FJ_STATUS_UNSUPPORTED_INPUT,
            Self::Panic => FJ_STATUS_INTERNAL_FAILURE,
            Self::Asset(AssetError::ProfileCompletion(error))
                if error.kind == ProfileCompletionErrorKind::Capacity =>
            {
                FJ_STATUS_ALLOCATION_FAILURE
            }
            Self::Asset(
                AssetError::ProfileCachePoisoned { .. }
                | AssetError::CsvCachePoisoned { .. }
                | AssetError::CalibrationCachePoisoned
                | AssetError::NoiseCachePoisoned,
            ) => FJ_STATUS_INTERNAL_FAILURE,
            Self::Asset(AssetError::Noise(error)) => {
                crate::asset_bridge::noise_category(error.kind())
            }
            Self::Asset(_) => FJ_STATUS_PREPARATION_FAILURE,
        }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => formatter.write_str(message),
            Self::Asset(error) => write!(formatter, "{error}"),
            Self::Panic => formatter.write_str("film profile boundary panic"),
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
        // SAFETY: Only run constructs this writer from exclusive foreign output.
        // Its nonzero, size-checked capacity bounds every offset; length starts
        // at zero and never exceeds capacity-1. The input text is disjoint. Raw
        // writes permit uninitialized C output bytes without forming references.
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), self.data.add(*self.length), count);
            self.data.add(*self.length + count).write(0);
        }
        *self.length += count;
        Ok(())
    }
}

/// # Safety
/// As for the exports, a nonnull error record and nonempty backing buffer
/// are exclusive writable storage, disjoint from all other inputs/outputs.
unsafe fn run<T>(
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<T, Failure>,
) -> Result<T, FjStatus> {
    let mut diagnostic = if !error.is_null() {
        // SAFETY: The caller authorizes this aligned record. Capacity is an
        // initialized input; length is output and is initialized before borrowing.
        let (capacity, length) = unsafe {
            let capacity = std::ptr::addr_of!((*error).capacity).read();
            let length = std::ptr::addr_of_mut!((*error).length);
            length.write(0);
            (capacity, &mut *length)
        };
        if capacity == 0 {
            None
        } else {
            // SAFETY: The nonempty output contract initializes the pointer field.
            let data = unsafe { std::ptr::addr_of!((*error).data).read() };
            if data.is_null() || capacity > isize::MAX as usize {
                // Dropping the uncalled operation also drops its captured owner.
                return Err(status(FJ_STATUS_UNSUPPORTED_INPUT));
            }
            // SAFETY: Nonempty exclusive writable bytes are caller-authorized.
            unsafe { data.write(0) };
            Some(Diagnostic {
                data: data.cast::<u8>(),
                capacity,
                length,
            })
        }
    } else {
        None
    };
    // This boundary mutates only cleared foreign outputs/diagnostics. Assets owns
    // its own lock/publication invariants; no captured mutable Rust state resumes.
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
            let _ = write!(diagnostic, "{}", Failure::Panic);
        }
        Err(status(FJ_STATUS_INTERNAL_FAILURE))
    })
}

/// # Safety
/// A nonempty view authorizes readable initialized bytes for the returned borrow,
/// which the enclosing export uses only before returning to its foreign caller.
unsafe fn text<'a>(view: FjStringView) -> Result<&'a str, Failure> {
    if view.count == 0 || view.count > isize::MAX as usize || view.data.is_null() {
        return Err(Failure::Input(
            "film profile root/key requires nonempty UTF-8 bytes",
        ));
    }
    // SAFETY: The caller's initialized byte extent is nonempty and size-checked.
    let bytes = unsafe { std::slice::from_raw_parts(view.data.cast::<u8>(), view.count) };
    if bytes.contains(&0) {
        return Err(Failure::Input("film profile root/key contains NUL"));
    }
    std::str::from_utf8(bytes).map_err(|_| Failure::Input("film profile root/key is not UTF-8"))
}

/// Acquire through the real catalog/complete-profile producer.
///
/// # Safety
/// Input spans authorize readable bytes until return. Nonnull outputs/error
/// storage are aligned, exclusive, mutually disjoint and do not overlap inputs.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_profile_acquire(
    resource_root: FjStringView,
    key: FjStringView,
    out_owner: *mut *mut FjFilmProfile,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_owner.is_null() {
        // SAFETY: The caller authorizes this exclusive pointer slot.
        unsafe { out_owner.write(std::ptr::null_mut()) };
    }
    // SAFETY: The foreign caller supplies the documented disjoint error storage.
    let result = unsafe {
        run(error, || {
            if out_owner.is_null() {
                return Err(Failure::Input("film profile owner output is NULL"));
            }
            // SAFETY: Both spans are call-local foreign borrows; neither escapes.
            let root = text(resource_root)?;
            let key = text(key)?;
            let assets = Assets::new(PathBuf::from(root));
            let profile = assets.film(key).map_err(Failure::Asset)?;
            // The returned Arc has left the cache lock. No array conversion or
            // native allocation occurs under it; dropping Assets leaves it live.
            Ok(Box::new(FilmOwner::new(profile)))
        })
    };
    match result {
        Ok(owner) => {
            // SAFETY: Publish only this complete owner into the authorized slot.
            unsafe { out_owner.write(Box::into_raw(owner).cast::<FjFilmProfile>()) };
            status(FJ_STATUS_SUCCESS)
        }
        Err(status) => status,
    }
}

/// Read a family-specific view without transferring its storage.
///
/// # Safety
/// A nonnull owner is a live handle returned by acquisition, retained through
/// every use of the spans. Output/error storage follows the acquire contract and
/// cannot overlap the owner. The caller excludes concurrent release.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_profile_view(
    owner: *const FjFilmProfile,
    out_view: *mut FjFilmFixtureView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_view.is_null() {
        // SAFETY: The foreign caller authorizes this exclusive record.
        unsafe { out_view.write(FjFilmFixtureView::empty()) };
    }
    // SAFETY: Disjoint error storage and live-owner lifetime are caller obligations.
    let result = unsafe {
        run(error, || {
            if out_view.is_null() || owner.is_null() {
                return Err(Failure::Input(
                    "film profile view requires owner and output",
                ));
            }
            // SAFETY: The live handle owns this immutable box; release is excluded.
            let owner = &*owner.cast::<FilmOwner>();
            // Only this foreign boundary detaches the raw record from its borrow;
            // the C contract requires owner retention until the last span use.
            out_view.write(FjFilmFixtureView::from_view(owner.fixture_view()));
            Ok(())
        })
    };
    result.map_or_else(|status| status, |()| status(FJ_STATUS_SUCCESS))
}

/// Consume one owner, even when diagnostic output is malformed.
///
/// # Safety
/// A nonnull owner is acquired, unreleased and has no overlapping access or
/// surviving borrow. It is consumed on every outcome. Error storage is exclusive
/// and disjoint; NULL owner is a no-op subject to diagnostic validation.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_profile_release(
    owner: *mut FjFilmProfile,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let owner = if owner.is_null() {
        None
    } else {
        // SAFETY: Consume the box matching acquisition once; no uses overlap.
        Some(unsafe { Box::from_raw(owner.cast::<FilmOwner>()) })
    };
    // SAFETY: Error storage follows the foreign contract. The closure owns the
    // box, so even error-output rejection consumes it by dropping that closure.
    let result = unsafe {
        run(error, || {
            drop(owner);
            Ok(())
        })
    };
    result.map_or_else(|status| status, |()| status(FJ_STATUS_SUCCESS))
}

static ABI_FACTS: &[usize] = &[
    size_of::<*mut FjFilmProfile>(),
    align_of::<*mut FjFilmProfile>(),
    size_of::<FjDoubleSpan>(),
    align_of::<FjDoubleSpan>(),
    offset_of!(FjDoubleSpan, data),
    offset_of!(FjDoubleSpan, count),
    size_of::<FjFilmFixtureView>(),
    align_of::<FjFilmFixtureView>(),
    offset_of!(FjFilmFixtureView, r#use),
    offset_of!(FjFilmFixtureView, antihalation),
    offset_of!(FjFilmFixtureView, asset_token),
    offset_of!(FjFilmFixtureView, halation_first_sigma_um),
    offset_of!(FjFilmFixtureView, halation_primary_amount),
    offset_of!(FjFilmFixtureView, source_log_exposure),
    offset_of!(FjFilmFixtureView, log_exposure),
    offset_of!(FjFilmFixtureView, density_curves_cmy),
    offset_of!(FjFilmFixtureView, density_curves_layers),
    offset_of!(FjFilmFixtureView, channel_density_cmy),
    offset_of!(FjFilmFixtureView, base_density),
    FJ_PROFILE_USE_STILL as usize,
    FJ_PROFILE_USE_CINE as usize,
    FJ_PROFILE_ANTIHALATION_STRONG as usize,
    FJ_PROFILE_ANTIHALATION_WEAK as usize,
    FJ_PROFILE_ANTIHALATION_NO as usize,
];

/// Static ABI facts for the CMake-linked C/C++ fixture.
///
/// # Safety
/// `count` is NULL or points to one exclusively writable size_t until return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_profile_abi_facts(count: *mut usize) -> *const usize {
    if count.is_null() {
        return std::ptr::null();
    }
    // SAFETY: The caller authorizes the output; no operation here can panic.
    unsafe { count.write(ABI_FACTS.len()) };
    ABI_FACTS.as_ptr()
}

const _: unsafe extern "C" fn(
    FjStringView,
    FjStringView,
    *mut *mut FjFilmProfile,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_film_profile_acquire;
const _: unsafe extern "C" fn(
    *const FjFilmProfile,
    *mut FjFilmFixtureView,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_film_profile_view;
const _: unsafe extern "C" fn(*mut FjFilmProfile, *mut FjErrorBuffer) -> FjStatus =
    fj_test_film_profile_release;
const _: unsafe extern "C" fn(*mut usize) -> *const usize = fj_test_profile_abi_facts;

use crate::asset_bridge::{self, FjNoise};
use crate::cuda::sys::FjStaticNoise;

/// # Safety
/// Root is nonempty initialized UTF-8 without NUL, live through return. Output
/// and diagnostics are aligned, exclusive and disjoint. Success transfers one owner.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_noise_acquire(
    resource_root: FjStringView,
    out_owner: *mut *mut FjNoise,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The common noise edge clears output and contains parsing/cleanup.
    unsafe {
        asset_bridge::acquire_noise(out_owner, error, || {
            let root = text(resource_root)
                .map_err(|_| asset_bridge::Failure::Input("invalid fixture resource root"))?;
            if root.is_empty() || root.contains('\0') {
                return Err(asset_bridge::Failure::Input(
                    "invalid fixture resource root",
                ));
            }
            Assets::new(PathBuf::from(root))
                .noise()
                .map_err(asset_bridge::Failure::Asset)
        })
    }
}
/// # Safety
/// Matching owner remains live through every view use; release excludes readers.
/// Output and diagnostics are exclusive, aligned and disjoint from owner storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_noise_view(
    owner: *const FjNoise,
    out_view: *mut FjStaticNoise,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Same borrow/storage contract as the shared raw projection.
    unsafe { asset_bridge::view_noise(owner, out_view, error) }
}
/// # Safety
/// Owner is NULL or a matching live handle consumed once, excluding reads and
/// borrows. Diagnostics obey the exclusive/disjoint storage contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_noise_release(
    owner: *mut FjNoise,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Same consuming contract, including malformed diagnostics and panic.
    unsafe { asset_bridge::release_noise(owner, error) }
}
const _: unsafe extern "C" fn(FjStringView, *mut *mut FjNoise, *mut FjErrorBuffer) -> FjStatus =
    fj_test_noise_acquire;
const _: unsafe extern "C" fn(*const FjNoise, *mut FjStaticNoise, *mut FjErrorBuffer) -> FjStatus =
    fj_test_noise_view;
const _: unsafe extern "C" fn(*mut FjNoise, *mut FjErrorBuffer) -> FjStatus = fj_test_noise_release;

/// Value-only CAT16 fixture facade, independent of the production export.
///
/// # Safety
/// Read-only inputs authorize three initialized aligned floats each; output
/// authorizes nine exclusive aligned floats disjoint from inputs until return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_cat16_matrix(
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_row_major: *mut f32,
) -> FjStatus {
    catch_unwind(|| {
        if out_row_major.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller authorizes nine exclusive output floats.
        unsafe { out_row_major.write_bytes(0, 9) };
        if source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: Both read-only inputs authorize three initialized floats.
        let whites = unsafe {
            film_juicer_core::color::Whites {
                source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
            }
        };
        let matrix = film_juicer_core::color::cat16_matrix(whites);
        // SAFETY: The local complete array is disjoint from the output extent.
        unsafe { out_row_major.copy_from_nonoverlapping(matrix.as_ptr(), 9) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Value-only scalar CAT16 fixture facade.
///
/// # Safety
/// Read-only inputs authorize three initialized aligned floats each; output
/// authorizes three exclusive aligned floats disjoint from inputs until return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_adapt_cat16(
    xyz: *const f32,
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    catch_unwind(|| {
        if out_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller authorizes three exclusive output floats.
        unsafe { out_xyz.write_bytes(0, 3) };
        if xyz.is_null() || source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: All read-only inputs authorize three initialized floats.
        let (value, whites) = unsafe {
            (
                xyz.cast::<[f32; 3]>().read(),
                film_juicer_core::color::Whites {
                    source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                    destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
                },
            )
        };
        let adapted = film_juicer_core::color::adapt_cat16(value, whites);
        // SAFETY: The local complete array is disjoint from the output extent.
        unsafe { out_xyz.copy_from_nonoverlapping(adapted.as_ptr(), 3) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Value-only CAT02 fixture facade, independent of the production export.
///
/// # Safety
/// Read-only inputs authorize three initialized aligned floats each; output
/// authorizes nine exclusive aligned floats disjoint from inputs until return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_cat02_matrix(
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_row_major: *mut f32,
) -> FjStatus {
    catch_unwind(|| {
        if out_row_major.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller authorizes nine exclusive output floats.
        unsafe { out_row_major.write_bytes(0, 9) };
        if source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: Both read-only inputs authorize three initialized floats.
        let whites = unsafe {
            film_juicer_core::color::Whites {
                source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
            }
        };
        let matrix = film_juicer_core::color::cat02_matrix(whites);
        // SAFETY: The local complete array is disjoint from the output extent.
        unsafe { out_row_major.copy_from_nonoverlapping(matrix.as_ptr(), 9) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Value-only scalar CAT02 fixture facade.
///
/// # Safety
/// Read-only inputs authorize three initialized aligned floats each; output
/// authorizes three exclusive aligned floats disjoint from inputs until return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_adapt_cat02(
    xyz: *const f32,
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    catch_unwind(|| {
        if out_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller authorizes three exclusive output floats.
        unsafe { out_xyz.write_bytes(0, 3) };
        if xyz.is_null() || source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: All read-only inputs authorize three initialized floats.
        let (value, whites) = unsafe {
            (
                xyz.cast::<[f32; 3]>().read(),
                film_juicer_core::color::Whites {
                    source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                    destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
                },
            )
        };
        if let Err(category) = color_facade_fault(4) {
            return status(category);
        }
        let adapted = film_juicer_core::color::adapt_cat02(value, whites);
        // SAFETY: The local complete array is disjoint from the output extent.
        unsafe { out_xyz.copy_from_nonoverlapping(adapted.as_ptr(), 3) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

thread_local! {
    static COLOR_FACADE_FAULT: std::cell::Cell<Option<(u32, u32, u32)>> = const { std::cell::Cell::new(None) };
    static COLOR_FACADE_CONSUMED: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_color_arm_facade_fault(operation: u32, index: u32, fault: u32) -> FjStatus {
    COLOR_FACADE_FAULT.set(None);
    COLOR_FACADE_CONSUMED.set(None);
    let valid = matches!(operation, 4 | 6 | 7 | 9 | 12) && index != 0 && (1..=2).contains(&fault);
    if valid {
        COLOR_FACADE_FAULT.set(Some((operation, index, fault)));
    }
    status(if valid {
        FJ_STATUS_SUCCESS
    } else {
        FJ_STATUS_UNSUPPORTED_INPUT
    })
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_color_clear_facade_fault() -> FjStatus {
    COLOR_FACADE_FAULT.set(None);
    COLOR_FACADE_CONSUMED.set(None);
    status(FJ_STATUS_SUCCESS)
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_color_facade_fault_consumed(operation: u32, fault: u32) -> u32 {
    u32::from(COLOR_FACADE_CONSUMED.take() == Some((operation, fault)))
}
fn color_facade_fault(operation: u32) -> Result<(), u32> {
    if let Some((selected, index, fault)) = COLOR_FACADE_FAULT.get() {
        if selected != operation {
            return Ok(());
        }
        if index > 1 {
            COLOR_FACADE_FAULT.set(Some((selected, index - 1, fault)));
            return Ok(());
        }
        COLOR_FACADE_FAULT.set(None);
        COLOR_FACADE_CONSUMED.set(Some((operation, fault)));
        if fault == 1 {
            return Err(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        panic!("direct color facade fault");
    }
    Ok(())
}

/// Arm a one-shot calling-thread fault on a matching production export.
#[unsafe(no_mangle)]
extern "C" fn fj_test_color_arm_fault(operation: u32, call_index: u32, fault: u32) -> FjStatus {
    catch_unwind(|| crate::legacy_bridge::arm_color_fault(operation, call_index, fault))
        .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Idempotently disarm the calling thread's fault.
#[unsafe(no_mangle)]
extern "C" fn fj_test_color_clear_fault() -> FjStatus {
    catch_unwind(crate::legacy_bridge::clear_color_fault)
        .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Direct-core input matrices fixture facade; independent of production faults.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_input_matrices(
    input_space_tag: u32,
    out: *mut FjInputColorMatrices,
) -> FjStatus {
    catch_unwind(|| {
        if out.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller authorizes the complete exclusive output record.
        unsafe { out.write_bytes(0, 1) };
        let Some(space) = input_space(input_space_tag) else {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        };
        let output = FjInputColorMatrices::from(color::input_matrices(space));
        // SAFETY: The initialized local record is committed to the exclusive output.
        unsafe { out.write(output) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Direct-core input to dwg fixture facade; independent of production faults.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_input_to_dwg(
    input: *const FjInputColorConversion,
    rgb: *const f32,
    clamp_nonnegative: u32,
    out_rgb: *mut f32,
    out_xyz: *mut f32,
) -> FjStatus {
    catch_unwind(|| {
        // SAFETY: Each nonnull output authorizes three exclusive aligned floats.
        unsafe {
            if !out_rgb.is_null() {
                out_rgb.write_bytes(0, 3);
            }
            if !out_xyz.is_null() {
                out_xyz.write_bytes(0, 3);
            }
        }
        if input.is_null()
            || rgb.is_null()
            || out_rgb.is_null()
            || out_xyz.is_null()
            || clamp_nonnegative > 1
        {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The input record is fully initialized, including disabled adaptation.
        let Some(input) = (unsafe { input.read() }).into_core() else {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        };
        // SAFETY: The read-only RGB extent authorizes three initialized aligned floats.
        let rgb = unsafe { rgb.cast::<[f32; 3]>().read() };
        if let Err(category) = color_facade_fault(6) {
            return status(category);
        }
        let output = color::input_to_dwg(input, rgb, clamp_nonnegative != 0);
        // SAFETY: Outputs are mutually disjoint and disjoint from inputs/local arrays.
        unsafe {
            out_rgb.copy_from_nonoverlapping(output.rgb.as_ptr(), 3);
            out_xyz.copy_from_nonoverlapping(output.xyz.as_ptr(), 3);
        }
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Direct-core input to linear srgb fixture facade; independent of production faults.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_input_to_linear_srgb(
    input: *const FjInputColorConversion,
    rgb: *const f32,
    xyz_to_linear_srgb: *const f32,
    out_rgb: *mut f32,
    out_xyz: *mut f32,
) -> FjStatus {
    catch_unwind(|| {
        // SAFETY: Each nonnull output authorizes three exclusive aligned floats.
        unsafe {
            if !out_rgb.is_null() {
                out_rgb.write_bytes(0, 3);
            }
            if !out_xyz.is_null() {
                out_xyz.write_bytes(0, 3);
            }
        }
        if input.is_null()
            || rgb.is_null()
            || xyz_to_linear_srgb.is_null()
            || out_rgb.is_null()
            || out_xyz.is_null()
        {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The input record is fully initialized, including disabled adaptation.
        let Some(input) = (unsafe { input.read() }).into_core() else {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        };
        // SAFETY: Read-only inputs authorize three RGB and nine matrix floats respectively.
        let (rgb, inverse) = unsafe {
            (
                rgb.cast::<[f32; 3]>().read(),
                xyz_to_linear_srgb.cast::<[f32; 9]>().read(),
            )
        };
        if let Err(category) = color_facade_fault(7) {
            return status(category);
        }
        let output = color::input_to_linear_srgb(input, rgb, inverse);
        // SAFETY: Outputs are mutually disjoint and disjoint from inputs/local arrays.
        unsafe {
            out_rgb.copy_from_nonoverlapping(output.rgb.as_ptr(), 3);
            out_xyz.copy_from_nonoverlapping(output.xyz.as_ptr(), 3);
        }
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Direct-core linear srgb to xyz fixture facade; independent of production faults.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_linear_srgb_to_xyz(rgb: *const f32, out_xyz: *mut f32) -> FjStatus {
    catch_unwind(|| {
        if out_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The output authorizes three exclusive aligned floats.
        unsafe { out_xyz.write_bytes(0, 3) };
        if rgb.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The read-only RGB extent authorizes three initialized aligned floats.
        let rgb = unsafe { rgb.cast::<[f32; 3]>().read() };
        let output = color::linear_srgb_to_xyz(rgb);
        // SAFETY: The initialized local triplet is disjoint from the output extent.
        unsafe { out_xyz.copy_from_nonoverlapping(output.as_ptr(), 3) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Direct-core dwg to xyz fixture facade; independent of production faults.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_dwg_to_xyz(rgb: *const f32, out_xyz: *mut f32) -> FjStatus {
    catch_unwind(|| {
        if out_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The output authorizes three exclusive aligned floats.
        unsafe { out_xyz.write_bytes(0, 3) };
        if rgb.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The read-only RGB extent authorizes three initialized aligned floats.
        let rgb = unsafe { rgb.cast::<[f32; 3]>().read() };
        if let Err(category) = color_facade_fault(9) {
            return status(category);
        }
        let output = color::dwg_to_xyz(rgb);
        // SAFETY: The initialized local triplet is disjoint from the output extent.
        unsafe { out_xyz.copy_from_nonoverlapping(output.as_ptr(), 3) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Direct-core project linear rgb to xyz fixture facade; independent of production faults.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_project_linear_rgb_to_xyz(
    rgb: *const f32,
    rgb_to_xyz: *const f32,
    xyz_adapt: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    catch_unwind(|| {
        if out_xyz.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The output authorizes three exclusive aligned floats.
        unsafe { out_xyz.write_bytes(0, 3) };
        if rgb.is_null() || rgb_to_xyz.is_null() || xyz_adapt.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: Inputs authorize an initialized RGB triplet and two nine-float matrices.
        let (rgb, matrix, adaptation) = unsafe {
            (
                rgb.cast::<[f32; 3]>().read(),
                rgb_to_xyz.cast::<[f32; 9]>().read(),
                xyz_adapt.cast::<[f32; 9]>().read(),
            )
        };
        let output = color::project_linear_rgb_to_xyz(rgb, matrix, adaptation);
        // SAFETY: The initialized local triplet is disjoint from the output extent.
        unsafe { out_xyz.copy_from_nonoverlapping(output.as_ptr(), 3) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Direct-core decoder fixture facade; does not consume any production fault.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_decode_input(
    input_space_tag: u32,
    decode_cctf: u32,
    rgb: *const f32,
    out_rgb: *mut f32,
) -> FjStatus {
    catch_unwind(|| {
        if out_rgb.is_null() {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller authorizes three exclusive aligned output floats.
        unsafe { out_rgb.write_bytes(0, 3) };
        if rgb.is_null() || decode_cctf > 1 {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        let Some(space) = input_space(input_space_tag) else {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        };
        // SAFETY: The read-only input authorizes three initialized aligned floats.
        let rgb = unsafe { rgb.cast::<[f32; 3]>().read() };
        let output = color::decode_input(space, decode_cctf != 0, rgb);
        // SAFETY: The complete local result is disjoint from the output.
        unsafe { out_rgb.copy_from_nonoverlapping(output.as_ptr(), 3) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

const _: unsafe extern "C" fn(u32, *mut FjInputColorMatrices) -> FjStatus = fj_test_input_matrices;
const _: unsafe extern "C" fn(
    *const FjInputColorConversion,
    *const f32,
    u32,
    *mut f32,
    *mut f32,
) -> FjStatus = fj_test_input_to_dwg;
const _: unsafe extern "C" fn(
    *const FjInputColorConversion,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
) -> FjStatus = fj_test_input_to_linear_srgb;
const _: unsafe extern "C" fn(*const f32, *mut f32) -> FjStatus = fj_test_linear_srgb_to_xyz;
const _: unsafe extern "C" fn(*const f32, *mut f32) -> FjStatus = fj_test_dwg_to_xyz;
const _: unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32) -> FjStatus =
    fj_test_project_linear_rgb_to_xyz;
const _: unsafe extern "C" fn(u32, u32, *const f32, *mut f32) -> FjStatus = fj_test_decode_input;
#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use film_juicer_core::profile::{ProfileCompletionError, Role};

    use super::*;

    #[test]
    fn foreign_projection_preserves_retained_profile_pointers_and_bits() {
        let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let profile = assets.film("kodak_portra_400").unwrap();
        let owner = FilmOwner::new(Arc::clone(&profile));
        assets.release_cached_payloads().unwrap();
        drop(assets);
        let raw = FjFilmFixtureView::from_view(owner.fixture_view());
        let tables = profile.tables();
        assert_eq!(raw.r#use, FJ_PROFILE_USE_STILL);
        assert_eq!(raw.antihalation, FJ_PROFILE_ANTIHALATION_STRONG);
        assert_eq!(raw.asset_token, profile.asset_token());
        assert_eq!(
            raw.halation_first_sigma_um.map(f32::to_bits),
            profile
                .processing_defaults()
                .halation_first_sigma_um
                .map(f32::to_bits)
        );
        assert_eq!(
            raw.halation_primary_amount.map(f32::to_bits),
            profile
                .processing_defaults()
                .halation_primary_amount
                .map(f32::to_bits)
        );
        assert_eq!(
            raw.source_log_exposure.data,
            tables.authored_log_exposure().as_ptr()
        );
        assert_eq!(
            raw.source_log_exposure.count,
            tables.authored_log_exposure().len()
        );
        for (span, expected) in [
            (raw.log_exposure, tables.interpolation_log_exposure()),
            (
                raw.density_curves_cmy,
                tables.density_curves().as_flattened(),
            ),
            (
                raw.channel_density_cmy,
                tables.channel_density().as_flattened(),
            ),
            (raw.base_density, tables.base_density()),
        ] {
            assert_eq!(span.data, expected.as_ptr());
            assert_eq!(span.count, expected.len());
        }
        for layer in 0..3 {
            for channel in 0..3 {
                let expected = &tables.density_curves_layers()[layer][channel];
                let span = raw.density_curves_layers[layer][channel];
                assert_eq!(span.data, expected.as_ptr());
                assert_eq!(span.count, expected.len());
            }
        }
    }

    #[test]
    fn typed_capacity_poison_and_panic_statuses_do_not_depend_on_text() {
        let error = AssetError::ProfileCompletion(ProfileCompletionError {
            stock: "capacity".into(),
            role: Role::Film,
            kind: ProfileCompletionErrorKind::Capacity,
        });
        assert_eq!(
            Failure::Asset(error).category(),
            FJ_STATUS_ALLOCATION_FAILURE
        );
        let error = AssetError::ProfileCachePoisoned { role: Role::Film };
        assert_eq!(Failure::Asset(error).category(), FJ_STATUS_INTERNAL_FAILURE);
        assert_eq!(Failure::Panic.category(), FJ_STATUS_INTERNAL_FAILURE);
    }

    #[test]
    fn zero_capacity_does_not_read_uninitialized_data_or_length() {
        let mut error = std::mem::MaybeUninit::<FjErrorBuffer>::uninit();
        let error = error.as_mut_ptr();
        // SAFETY: Only capacity is initialized input. run writes length before
        // borrowing it, and zero capacity never reads the uninitialized data field.
        unsafe {
            std::ptr::addr_of_mut!((*error).capacity).write(0);
            let result: Result<(), FjStatus> = run(error, || Ok(()));
            assert!(result.is_ok());
            assert_eq!(std::ptr::addr_of!((*error).length).read(), 0);
        }
    }

    #[test]
    fn consuming_release_reclaims_even_with_malformed_diagnostic() {
        for malformed in 0..3 {
            let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
            let profile = assets.film("kodak_portra_400").unwrap();
            let weak = Arc::downgrade(&profile);
            let owner = Box::into_raw(Box::new(FilmOwner::new(profile))).cast::<FjFilmProfile>();
            assets.release_cached_payloads().unwrap();
            drop(assets);
            assert_eq!(weak.strong_count(), 1);
            let mut byte = 0_i8;
            let mut error = FjErrorBuffer {
                data: if malformed == 2 {
                    (&raw mut byte).cast()
                } else {
                    std::ptr::null_mut()
                },
                capacity: [0, 1, usize::MAX][malformed],
                length: 99,
            };
            // SAFETY: Consume a live box once; the error record is exclusive and
            // the deliberate malformed buffer is rejected without access.
            let result = unsafe { fj_test_film_profile_release(owner, &raw mut error) };
            assert_eq!(
                result.category,
                if malformed != 0 {
                    FJ_STATUS_UNSUPPORTED_INPUT
                } else {
                    FJ_STATUS_SUCCESS
                }
            );
            assert_eq!(error.length, 0);
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn unpublished_owner_is_reclaimed_on_failure_and_contained_panic() {
        for failure in 0..3 {
            let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
            let profile = assets.film("kodak_portra_400").unwrap();
            let weak = Arc::downgrade(&profile);
            let owner = FilmOwner::new(profile);
            assets.release_cached_payloads().unwrap();
            drop(assets);
            let mut bytes = [0_i8; 8];
            let mut error = FjErrorBuffer {
                data: bytes.as_mut_ptr().cast(),
                capacity: bytes.len(),
                length: 0,
            };
            // SAFETY: The disjoint record and initialized buffer stay live; no
            // foreign call/callback runs inside this private cleanup seam.
            let result: Result<(), FjStatus> = unsafe {
                run(&raw mut error, || {
                    let _owner = owner;
                    if failure == 2 {
                        panic!("profile publication fault");
                    }
                    if failure == 1 {
                        return Err(Failure::Asset(AssetError::ProfileCompletion(
                            ProfileCompletionError {
                                stock: "unpublished capacity fault".into(),
                                role: Role::Film,
                                kind: ProfileCompletionErrorKind::Capacity,
                            },
                        )));
                    }
                    Err(Failure::Input("unpublished profile fault"))
                })
            };
            assert_eq!(
                result.err().unwrap().category,
                [
                    FJ_STATUS_UNSUPPORTED_INPUT,
                    FJ_STATUS_ALLOCATION_FAILURE,
                    FJ_STATUS_INTERNAL_FAILURE
                ][failure]
            );
            assert_eq!(error.length, 7);
            assert_eq!(bytes[7], 0);
            assert!(weak.upgrade().is_none());
        }
    }
}

thread_local! { static SPECTRAL_FACADE_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
thread_local! {
    static EXPOSURE_FACADE_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static EXPOSURE_FACADE_CONSUMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_exposure_arm_fault(operation: u32, index: u32, fault: u32) -> FjStatus {
    crate::exposure_bridge::arm_fault(operation, index, fault)
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_exposure_clear_fault() {
    crate::exposure_bridge::clear_fault();
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_exposure_fault_consumed(operation: u32, fault: u32) -> u32 {
    u32::from(crate::exposure_bridge::fault_consumed(operation, fault))
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_exposure_arm_facade_fault() {
    EXPOSURE_FACADE_CONSUMED.set(false);
    EXPOSURE_FACADE_FAULT.set(true);
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_exposure_facade_fault_consumed() -> u32 {
    u32::from(EXPOSURE_FACADE_CONSUMED.replace(false))
}
fn exposure_facade_fault() -> Result<(), crate::asset_bridge::Failure> {
    if EXPOSURE_FACADE_FAULT.replace(false) {
        EXPOSURE_FACADE_CONSUMED.set(true);
        Err(crate::asset_bridge::Failure::Internal(
            "exposure fixture-only fault",
        ))
    } else {
        Ok(())
    }
}
/// # Safety
/// Initialized tensor/input and exclusive disjoint output/error obey the header.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_reconstruction_reference_white(
    input: *const crate::exposure_bridge::FjReferenceWhiteInput,
    out: *mut crate::exposure_bridge::FjReferenceWhite,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared checked projection scopes the tensor borrow; calls core directly.
    unsafe { crate::exposure_bridge::reference_call(input, out, error, exposure_facade_fault) }
}
/// # Safety
/// Required spans/output/failure/error obey the synchronous exposure contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_exposure_sensitivity(
    input: *const crate::exposure_bridge::FjSensitivityInput,
    out: *mut crate::exposure_bridge::FjSensitivity,
    failure: *mut crate::exposure_bridge::FjSensitivityFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The direct-core facade shares only checked call-local projection.
    unsafe {
        crate::exposure_bridge::sensitivity_call(input, out, failure, error, exposure_facade_fault)
    }
}
/// # Safety
/// Initialized four-value parameters and exclusive scalar/error obey the header.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_exposure_window_sample(
    wavelength: f32,
    params: FjFloatSpan,
    out: *mut f32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared checked facade borrows parameters and calls the core leaf.
    unsafe {
        crate::exposure_bridge::window_call(wavelength, params, out, error, exposure_facade_fault)
    }
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_spectral_arm_fault(operation: u32, index: u32, fault: u32) -> FjStatus {
    crate::spectral_bridge::arm_fault(operation, index, fault)
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_spectral_clear_fault() {
    crate::spectral_bridge::clear_fault();
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_spectral_arm_facade_fault() {
    SPECTRAL_FACADE_FAULT.set(true);
}
fn spectral_facade_fault() -> Result<(), crate::asset_bridge::Failure> {
    if SPECTRAL_FACADE_FAULT.replace(false) {
        Err(crate::asset_bridge::Failure::Internal(
            "spectral fixture-only fault",
        ))
    } else {
        Ok(())
    }
}
/// # Safety
/// Initialized input spans and exclusive output/error follow the spectral header.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_spectral_tables(
    input: *const crate::spectral_bridge::FjSpectralInput,
    out: *mut crate::spectral_bridge::FjSpectralTables,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked projection scopes source borrows; core is called directly.
    unsafe { crate::spectral_bridge::tables_call(input, out, error, spectral_facade_fault) }
}
/// # Safety
/// Initialized source and exclusive output/failure/error records follow the header.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_spectral_white(
    input: *const crate::spectral_bridge::FjSpectralWhiteInput,
    out: *mut crate::spectral_bridge::FjSpectralWhite,
    failure: *mut crate::spectral_bridge::FjSpectralWhiteFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Same scoped checked input projection; no production export is called.
    unsafe { crate::spectral_bridge::white_call(input, out, failure, error, spectral_facade_fault) }
}
/// # Safety
/// Weighted source spans and exclusive output/error follow the spectral header.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_spectral_s_inverse(
    input: *const crate::spectral_bridge::FjSpectralSInput,
    out: *mut crate::spectral_bridge::FjSpectralInverse,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade calls core with checked local borrows through shared projection.
    unsafe { crate::spectral_bridge::inverse_call(input, out, error, spectral_facade_fault) }
}

thread_local! {
    static TC_FACADE_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TC_FACADE_CONSUMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_tc_lut_arm_fault(operation: u32, index: u32, fault: u32) -> FjStatus {
    crate::reconstruction_bridge::arm_fault(operation, index, fault)
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_tc_lut_clear_fault() {
    crate::reconstruction_bridge::clear_fault();
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_tc_lut_fault_consumed(operation: u32, fault: u32) -> u32 {
    u32::from(crate::reconstruction_bridge::take_consumed() == Some((operation, fault)))
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_tc_lut_arm_facade_fault() {
    TC_FACADE_FAULT.set(true);
    TC_FACADE_CONSUMED.set(false);
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_tc_lut_facade_fault_consumed() -> u32 {
    u32::from(TC_FACADE_CONSUMED.replace(false))
}
fn tc_facade_fault() -> Result<(), crate::asset_bridge::Failure> {
    if TC_FACADE_FAULT.replace(false) {
        TC_FACADE_CONSUMED.set(true);
        Err(crate::asset_bridge::Failure::Internal(
            "TC fixture-only fault",
        ))
    } else {
        Ok(())
    }
}
/// # Safety
/// Live immutable input spans and exclusive empty output/error obey the TC header.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_reconstruction_tc_lut(
    input: *const crate::reconstruction_bridge::FjFilmTcLutInput,
    out: *mut crate::reconstruction_bridge::FjOwnedFilmTcLut,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared checked edge invokes the core directly, never a legacy export.
    unsafe { crate::reconstruction_bridge::build_call(input, out, error, tc_facade_fault) }
}
/// # Safety
/// Initialized LUT/XYZ and exclusive RGB/error obey the synchronous TC contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_reconstruction_sample_tc_lut(
    lut: FjFloatSpan,
    xyz: *const f32,
    out: *mut f32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared extent checks scope the direct-core borrows and outputs.
    unsafe { crate::reconstruction_bridge::sample_call(lut, xyz, out, error, tc_facade_fault) }
}
/// # Safety
/// Consume the sole unmodified module allocation record with all views ended,
/// or a fully zero empty record. No duplicate/stale/corrupt token is admissible.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_reconstruction_release_tc_lut(
    owned: *mut crate::reconstruction_bridge::FjOwnedFilmTcLut,
) -> FjStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Caller returns its exact live Vec allocation/layout once.
        unsafe { crate::reconstruction_bridge::release_call(owned) }
    })) {
        Ok(status) => status,
        Err(_) => FjStatus {
            category: crate::cuda::sys::FJ_STATUS_INTERNAL_FAILURE,
            api: crate::cuda::sys::FJ_API_NONE,
            native_code: 0,
        },
    }
}

/// # Safety
/// Output authorizes two aligned exclusive usize values, disjoint from all live call storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_tc_lut_allocation_counts(out: *mut usize) -> FjStatus {
    if out.is_null()
        || !(out as usize).is_multiple_of(align_of::<usize>())
        || (out as usize).checked_add(2 * size_of::<usize>()).is_none()
    {
        return FjStatus {
            category: crate::cuda::sys::FJ_STATUS_UNSUPPORTED_INPUT,
            api: crate::cuda::sys::FJ_API_NONE,
            native_code: 0,
        };
    }
    let counts = crate::reconstruction_bridge::allocation_counts();
    // SAFETY: Caller authorizes the full exclusive two-element initialized extent.
    unsafe { out.cast::<[usize; 2]>().write(counts) };
    FjStatus {
        category: crate::cuda::sys::FJ_STATUS_SUCCESS,
        api: crate::cuda::sys::FJ_API_NONE,
        native_code: 0,
    }
}

/// # Safety
/// Initialized fixed-exposure input/spans and exclusive disjoint output/error obey the header.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_exposure_mallett_midgray(
    input: *const crate::exposure_bridge::FjMallettMidgrayInput,
    out: *mut crate::exposure_bridge::FjMallettMidgray,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked projection calls core directly, never a legacy export.
    unsafe {
        crate::exposure_bridge::mallett_midgray_call(input, out, error, exposure_facade_fault)
    }
}
/// # Safety
/// Exclusive aligned fixed output/error records are caller-authorized and disjoint.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_exposure_tc_midgray(
    green: f32,
    out: *mut crate::exposure_bridge::FjMidgrayNormalization,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked projection calls core directly with scoped output.
    unsafe { crate::exposure_bridge::tc_midgray_call(green, out, error, exposure_facade_fault) }
}
/// # Safety
/// The exclusive aligned scalar/error records are caller-authorized and disjoint.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_exposure_reference_source(
    exposure_ev: f32,
    out: *mut f32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked projection calls core directly with scoped output.
    unsafe {
        crate::exposure_bridge::reference_source_call(
            exposure_ev,
            out,
            error,
            exposure_facade_fault,
        )
    }
}
/// # Safety
/// Initialized fixed input/spans and exclusive disjoint output/error obey the header.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_exposure_mallett_reference_raw(
    input: *const crate::exposure_bridge::FjMallettReferenceInput,
    out: *mut crate::exposure_bridge::FjReferenceRaw,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked projection calls core directly, retaining no source pointer.
    unsafe {
        crate::exposure_bridge::mallett_reference_call(input, out, error, exposure_facade_fault)
    }
}

/// Direct completed core output-color facade.
/// # Safety
/// Output authorizes one complete initialized aligned, live/addressable exclusive
/// record through return; no pointer escapes. Invalid output is not written.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_output_color(
    output_space: u32,
    out: *mut crate::legacy_bridge::FjOutputColorPreparation,
) -> FjStatus {
    use crate::legacy_bridge::{FjOutputColorPreparation, output_record_valid};
    let valid = output_record_valid(out);
    if valid {
        // SAFETY: Only independently valid exclusively writable output is cleared.
        unsafe { out.write_bytes(0, 1) };
    }
    catch_unwind(|| {
        if !valid {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        let Ok(space) = film_juicer_core::color::OutputSpace::try_from(output_space) else {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        };
        if let Err(category) = color_facade_fault(12) {
            return status(category);
        }
        let result = FjOutputColorPreparation::prepare(space);
        // SAFETY: The facade commits a complete owned value into the checked output.
        unsafe { out.write(result) };
        status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}
const _: unsafe extern "C" fn(
    u32,
    *mut crate::legacy_bridge::FjOutputColorPreparation,
) -> FjStatus = fj_test_output_color;

/// # Safety
/// Count authorizes one exclusive writable size_t; returned immutable facts are static.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_input_abi_facts(count: *mut usize) -> *const usize {
    if count.is_null() {
        return std::ptr::null();
    }
    // SAFETY: The fixture authorizes this exclusive initialized output.
    unsafe {
        count.write(crate::film_input_bridge::ABI_FACTS.len());
    }
    crate::film_input_bridge::ABI_FACTS.as_ptr()
}
/// # Safety
/// Same call-local initialized storage, retained profile and disjoint outputs contract as the legacy declaration.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_input_prepare(
    profile: *const FjFilmProfile,
    params: *const crate::film_input_bridge::FjFilmInputParams,
    resources: *const crate::film_input_bridge::FjFilmInputResources,
    out: *mut crate::film_input_bridge::FjFilmInput,
    failure: *mut crate::film_input_bridge::FjFilmInputFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade calls the common safe construction/projector, with the same checked raw contract.
    unsafe {
        crate::film_input_bridge::prepare(profile.cast(), params, resources, out, failure, error)
    }
}

#[unsafe(no_mangle)]
extern "C" fn fj_test_film_input_arm_boundary_fault(stage: u32) {
    crate::film_input_bridge::arm_boundary_fault(stage);
}

/// # Safety
/// Matching live owners and the production call's initialized/disjoint record and span obligations.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_density_minimum_prepare(
    requested: *const crate::film_recipe_bridge::FjFilmDensityMinimum,
    out_minimum: *mut crate::film_recipe_bridge::FjFilmDensityMinimum,
    out_failure: *mut crate::film_recipe_bridge::FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade calls the shared constructor/projector, never a legacy export.
    unsafe { crate::film_recipe_bridge::minimum(requested, out_minimum, out_failure, error) }
}

/// # Safety
/// Matching live owners and the production call's initialized/disjoint record and span obligations.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_dir_acquire(
    development: *const crate::film_bridge::FjFilmDevelopmentOwner,
    params: *const crate::film_recipe_bridge::FjFilmDirParams,
    out_dir: *mut *mut crate::film_recipe_bridge::FjFilmDirOwner,
    out_failure: *mut crate::film_recipe_bridge::FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade calls the shared constructor/projector, never a legacy export.
    unsafe { crate::film_recipe_bridge::acquire(development, params, out_dir, out_failure, error) }
}

/// # Safety
/// Matching live owners and the production call's initialized/disjoint record and span obligations.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_dir_view(
    dir: *const crate::film_recipe_bridge::FjFilmDirOwner,
    out_view: *mut crate::film_recipe_bridge::FjFilmDirView,
    out_failure: *mut crate::film_recipe_bridge::FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade calls the shared constructor/projector, never a legacy export.
    unsafe { crate::film_recipe_bridge::view(dir, out_view, out_failure, error) }
}

/// # Safety
/// Matching live owners and the production call's initialized/disjoint record and span obligations.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_dir_release(
    dir: *mut crate::film_recipe_bridge::FjFilmDirOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade calls the shared constructor/projector, never a legacy export.
    unsafe { crate::film_recipe_bridge::release(dir, error) }
}

/// # Safety
/// Matching live owners and the production call's initialized/disjoint record and span obligations.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_density_bounds_prepare(
    development: *const crate::film_bridge::FjFilmDevelopmentOwner,
    requested: *const crate::film_recipe_bridge::FjFilmDensityMinimum,
    out_bounds: *mut crate::film_recipe_bridge::FjFilmDensityBounds,
    out_failure: *mut crate::film_recipe_bridge::FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade calls the shared constructor/projector, never a legacy export.
    unsafe {
        crate::film_recipe_bridge::bounds(development, requested, out_bounds, out_failure, error)
    }
}

/// # Safety
/// Matching live completed Development; initialized aligned readable params and
/// exclusive writable output/failure/diagnostic extents, all mutually disjoint.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_visual_grain_prepare(
    development: *const crate::film_bridge::FjFilmDevelopmentOwner,
    params: *const crate::film_recipe_bridge::FjVisualGrainParams,
    out_recipe: *mut crate::film_recipe_bridge::FjVisualGrainRecipe,
    out_failure: *mut crate::film_recipe_bridge::FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade shares the checked constructor/projector, never the legacy export.
    unsafe {
        crate::film_recipe_bridge::prepare_grain(
            development,
            params,
            out_recipe,
            out_failure,
            error,
        )
    }
}
const _: unsafe extern "C" fn(
    *const crate::film_bridge::FjFilmDevelopmentOwner,
    *const crate::film_recipe_bridge::FjVisualGrainParams,
    *mut crate::film_recipe_bridge::FjVisualGrainRecipe,
    *mut crate::film_recipe_bridge::FjFilmRecipeFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_visual_grain_prepare;

/// # Safety
/// Same initialized, live, aligned, disjoint records as the production call.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_physical_effects_prepare(
    params: *const crate::film_recipe_bridge::FjPhysicalEffectsParams,
    out_recipe: *mut crate::film_recipe_bridge::FjPhysicalEffectsRecipe,
    out_failure: *mut crate::film_recipe_bridge::FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade shares the checked operation and its caller obligations.
    unsafe { crate::film_recipe_bridge::prepare_effects(params, out_recipe, out_failure, error) }
}
const _: unsafe extern "C" fn(
    *const crate::film_recipe_bridge::FjPhysicalEffectsParams,
    *mut crate::film_recipe_bridge::FjPhysicalEffectsRecipe,
    *mut crate::film_recipe_bridge::FjFilmRecipeFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_physical_effects_prepare;

/// # Safety
/// Identical fixed spatial-optics record/diagnostic lifetime and disjointness contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_spatial_optics_prepare(
    params: *const crate::optics_bridge::FjSpatialOpticsParams,
    out: *mut crate::optics_bridge::FjSpatialOpticsRecipe,
    failure: *mut crate::optics_bridge::FjSpatialOpticsFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade forwards to the same checked operation, never a legacy export.
    unsafe { crate::optics_bridge::prepare(params, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const crate::optics_bridge::FjSpatialOpticsParams,
    *mut crate::optics_bridge::FjSpatialOpticsRecipe,
    *mut crate::optics_bridge::FjSpatialOpticsFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_spatial_optics_prepare;

/// # Safety
/// Same initialized, aligned, live and disjoint records/source as the checked operation.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_print_illuminant_prepare(
    params: *const crate::print_bridge::FjPrintIlluminantParams,
    source: FjFloatSpan,
    out: *mut crate::asset_illuminant::FjIlluminant,
    failure: *mut crate::print_bridge::FjPrintIlluminantFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade uses the common checked implementation, not its legacy export.
    unsafe { crate::print_bridge::prepare(params, source, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const crate::print_bridge::FjPrintIlluminantParams,
    FjFloatSpan,
    *mut crate::asset_illuminant::FjIlluminant,
    *mut crate::print_bridge::FjPrintIlluminantFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_print_illuminant_prepare;

/// Scalar primitive witness for the owning preflash domain comparison.
/// # Safety
/// out contains five exclusive writable doubles; narrowed contains one exclusive
/// writable float. Both are aligned, live and disjoint through return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_print_preflash_primitive(
    density: f64,
    illuminant: f64,
    sensitivity: f64,
    accumulator: f64,
    out: *mut f64,
    narrowed: *mut f32,
) {
    let power = 10.0_f64.powf(-density);
    let transmitted = power * illuminant;
    let light = if transmitted.is_nan() {
        0.0
    } else {
        transmitted
    };
    let product = light * sensitivity;
    let sum = accumulator + product;
    // SAFETY: The fixture authorizes the complete fixed, disjoint outputs.
    unsafe {
        std::ptr::copy_nonoverlapping([power, transmitted, light, product, sum].as_ptr(), out, 5);
        narrowed.write(sum as f32);
    }
}

/// Checked preflash fixture facade shares construction without invoking the legacy export.
/// # Safety
/// The same live, initialized, aligned and disjoint synchronous raw contract as
/// print_bridge::prepare_preflash; inputs contain exactly 81/81/243 f32 elements.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_print_preflash_prepare(
    input: *const crate::print_bridge::FjPrintPreflashInput,
    out: *mut crate::print_bridge::FjPrintPreflashRaw,
    failure: *mut crate::print_bridge::FjPrintPreflashFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Identical checked operation, not a call through the legacy export.
    unsafe { crate::print_bridge::prepare_preflash(input, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const crate::print_bridge::FjPrintPreflashInput,
    *mut crate::print_bridge::FjPrintPreflashRaw,
    *mut crate::print_bridge::FjPrintPreflashFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_print_preflash_prepare;

/// Primitive-only witness for early balance feasibility; no balance construction.
/// # Safety
/// density/raw contain seven/three initialized floats. out/narrowed contain
/// twelve/two exclusive writable elements, disjoint from all input storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_print_balance_primitive(
    density: *const f32,
    raw: *const f32,
    denominator: f32,
    out: *mut f64,
    narrowed: *mut f32,
) {
    // SAFETY: The domain fixture provides the documented fixed live extents.
    let (density, raw) = unsafe {
        (
            std::slice::from_raw_parts(density, 7),
            std::slice::from_raw_parts(raw, 3),
        )
    };
    let mut observed = [0.0_f64; 12];
    let mut d = f64::from(density[0]);
    for c in 0..3 {
        d += f64::from(density[c + 1]) * f64::from(density[c + 4]);
        observed[c] = d;
    }
    observed[3] = 10.0_f64.powf(-d);
    let mut mean_log = 0.0_f64;
    for c in 0..3 {
        observed[4 + c] = f64::from(raw[c]).max(1.0e-10).ln();
        mean_log += observed[4 + c];
        observed[7 + c] = mean_log;
    }
    observed[10] = -mean_log / 3.0;
    observed[11] = observed[10].exp();
    let factor = observed[11] as f32;
    // SAFETY: Caller supplies fixed writable extents, exclusive through return.
    unsafe {
        std::ptr::copy_nonoverlapping(observed.as_ptr(), out, 12);
        std::ptr::copy_nonoverlapping([factor, factor / denominator].as_ptr(), narrowed, 2);
    }
}

/// Scalar feasibility witness; no complete correction constructor.
/// # Safety
/// out/out_double contain six/three exclusive initialized writable elements,
/// disjoint from each other and live through synchronous return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_correction_primitive(
    value: f32,
    accumulator: f32,
    multiplier: f32,
    out: *mut f32,
    out_double: *mut f64,
) {
    let clamped = value.clamp(0.0, 1.0);
    let observed = [
        10.0_f32.powf(-value),
        value.log10(),
        value.max(0.0),
        clamped,
        accumulator + value * multiplier,
        if clamped <= 0.04045 {
            clamped / 12.92
        } else {
            ((clamped + 0.055) / 1.055).powf(2.4)
        },
    ];
    let double = f64::from(value);
    let observed_double = [10.0_f64.powf(-double), double.log10(), double.max(0.0)];
    // SAFETY: The fixture supplies the documented disjoint fixed writable extents.
    unsafe {
        std::ptr::copy_nonoverlapping(observed.as_ptr(), out, 6);
        std::ptr::copy_nonoverlapping(observed_double.as_ptr(), out_double, 3);
    }
}

/// Direct fixture facade invokes the checked operation, not the legacy export.
/// # Safety
/// The same initialized live input and exclusive/disjoint writable contract as
/// scanner_bridge::prepare_direct; all borrows expire at synchronous return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_direct_correction_prepare(
    params: *const scanner_bridge::FjScannerCorrectionParams,
    input: *const scanner_bridge::FjScannerDirectCorrectionInput,
    out: *mut scanner_bridge::FjScannerColorCorrection,
    failure: *mut scanner_bridge::FjScannerCorrectionFailure,
    trace: *mut scanner_bridge::FjScannerCorrectionTrace,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked operation admits consumed records and retains none.
    unsafe { scanner_bridge::prepare_direct(params, input, out, failure, trace, error) }
}

/// Print fixture facade invokes the checked operation, not the legacy export.
/// # Safety
/// Same synchronous live/disjoint contract as scanner_bridge::prepare_print.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_print_correction_prepare(
    params: *const scanner_bridge::FjScannerCorrectionParams,
    input: *const scanner_bridge::FjScannerPrintCorrectionInput,
    out: *mut scanner_bridge::FjScannerColorCorrection,
    failure: *mut scanner_bridge::FjScannerCorrectionFailure,
    trace: *mut scanner_bridge::FjScannerCorrectionTrace,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked operation admits consumed records and retains none.
    unsafe { scanner_bridge::prepare_print(params, input, out, failure, trace, error) }
}

#[unsafe(no_mangle)]
extern "C" fn fj_test_scanner_correction_fault(operation: u32, fault: u32) -> FjStatus {
    scanner_bridge::faults::arm(operation, fault)
}

/// # Safety
/// Live immutable spans and two exclusive/disjoint writable output elements.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_lookup(
    query: f32,
    axis: FjFloatSpan,
    values: FjFloatSpan,
    out: *mut f32,
    present: *mut u32,
) -> FjStatus {
    // SAFETY: The same checked scanner primitive preserves optional NaN values.
    unsafe { scanner_bridge::lookup(query, axis, values, None, out, present) }
}

/// # Safety
/// Live immutable axis/packed C/M/Y spans and exclusive/disjoint scalar outputs.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_density_curve(
    query: f32,
    axis: FjFloatSpan,
    curves: FjFloatSpan,
    channel: u32,
    out: *mut f32,
    present: *mut u32,
) -> FjStatus {
    // SAFETY: The checked scanner primitive admits consumed rows before borrowing.
    unsafe { scanner_bridge::lookup(query, axis, curves, Some(channel), out, present) }
}

const _: unsafe extern "C" fn(
    *const scanner_bridge::FjScannerCorrectionParams,
    *const scanner_bridge::FjScannerDirectCorrectionInput,
    *mut scanner_bridge::FjScannerColorCorrection,
    *mut scanner_bridge::FjScannerCorrectionFailure,
    *mut scanner_bridge::FjScannerCorrectionTrace,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_scanner_direct_correction_prepare;
const _: unsafe extern "C" fn(
    *const scanner_bridge::FjScannerCorrectionParams,
    *const scanner_bridge::FjScannerPrintCorrectionInput,
    *mut scanner_bridge::FjScannerColorCorrection,
    *mut scanner_bridge::FjScannerCorrectionFailure,
    *mut scanner_bridge::FjScannerCorrectionTrace,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_scanner_print_correction_prepare;
const _: extern "C" fn(u32, u32) -> FjStatus = fj_test_scanner_correction_fault;
const _: unsafe extern "C" fn(f32, FjFloatSpan, FjFloatSpan, *mut f32, *mut u32) -> FjStatus =
    fj_test_scanner_lookup;
const _: unsafe extern "C" fn(f32, FjFloatSpan, FjFloatSpan, u32, *mut f32, *mut u32) -> FjStatus =
    fj_test_scanner_density_curve;

/// Checked balance fixture facade shares construction without calling the export.
/// # Safety
/// The same fixed live input/output and disjoint synchronous contract as prepare_balance.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_print_balance_prepare(
    input: *const crate::print_bridge::FjPrintBalanceInput,
    out: *mut crate::print_bridge::FjPrintBalance,
    failure: *mut crate::print_bridge::FjPrintBalanceFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Identical checked operation with independent output admission.
    unsafe { crate::print_bridge::prepare_balance(input, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const crate::print_bridge::FjPrintBalanceInput,
    *mut crate::print_bridge::FjPrintBalance,
    *mut crate::print_bridge::FjPrintBalanceFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_print_balance_prepare;

/// # Safety
/// Identical live Development, initialized spans and disjoint synchronous
/// output/diagnostic storage contract as the production reference entry.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_reference_prepare(
    development: *const crate::film_bridge::FjFilmDevelopmentOwner,
    input: *const crate::film_recipe_bridge::FjFilmReferenceInput,
    out: *mut crate::film_recipe_bridge::FjFilmReference,
    failure: *mut crate::film_recipe_bridge::FjFilmReferenceFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Both entries share this checked immediate call; no export nesting.
    unsafe { crate::film_recipe_bridge::prepare_reference(development, input, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const crate::film_bridge::FjFilmDevelopmentOwner,
    *const crate::film_recipe_bridge::FjFilmReferenceInput,
    *mut crate::film_recipe_bridge::FjFilmReference,
    *mut crate::film_recipe_bridge::FjFilmReferenceFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_film_reference_prepare;

/// # Safety
/// Same fixed borrow and disjoint output contract as reference_density_lookup.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_reference_density_lookup(
    query: f32,
    axis: FjFloatSpan,
    curves: FjFloatSpan,
    channel: u32,
    out: *mut f32,
    present: *mut u32,
) -> FjStatus {
    // SAFETY: Forward the identical admitted, synchronous test-ingress contract.
    unsafe {
        crate::film_recipe_bridge::reference_density_lookup(
            query, axis, curves, channel, out, present,
        )
    }
}

/// # Safety
/// Same checked, disjoint synchronous contract as scanner_bridge::lut_acquire.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_lut_acquire(
    input: *const scanner_bridge::FjScannerLutInput,
    out: *mut *mut scanner_bridge::FjScannerLutOwner,
    failure: *mut scanner_bridge::FjScannerLutFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked operation scopes every foreign projection.
    unsafe { scanner_bridge::lut_acquire(input, out, failure, error) }
}
/// # Safety
/// Same live unique-owner and disjoint output contract as scanner_bridge::lut_view.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_lut_view(
    owner: *const scanner_bridge::FjScannerLutOwner,
    out: *mut scanner_bridge::FjScannerLutView,
    failure: *mut scanner_bridge::FjScannerLutFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared checked operation scopes the immutable owner borrow.
    unsafe { scanner_bridge::lut_view(owner, out, failure, error) }
}
/// # Safety
/// Same consuming unique-owner contract as scanner_bridge::lut_release.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_lut_release(
    owner: *mut scanner_bridge::FjScannerLutOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Ownership transfers into the shared consuming checked release.
    unsafe { scanner_bridge::lut_release(owner, error) }
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_scanner_lut_fault(operation: u32, fault: u32) -> FjStatus {
    scanner_bridge::lut_faults::arm(operation, fault)
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_scanner_lut_live_owners() -> usize {
    scanner_bridge::lut_faults::live()
}

/// # Safety
/// Same complete synchronous live/disjoint contract as scanner_bridge::color_prepare.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_color_prepare(
    input: *const scanner_bridge::FjScannerColorInput,
    out: *mut scanner_bridge::FjScannerColor,
    failure: *mut scanner_bridge::FjScannerColorFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Test caller provides the identical checked transport contract.
    unsafe { scanner_bridge::color_prepare(input, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const scanner_bridge::FjScannerColorInput,
    *mut scanner_bridge::FjScannerColor,
    *mut scanner_bridge::FjScannerColorFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_scanner_color_prepare;

/// # Safety
/// Identical synchronous storage contract to the production descriptor entry.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_lut_describe(
    input: *const scanner_bridge::FjScannerLutDescriptorInput,
    out: *mut scanner_bridge::FjScannerLutDescriptor,
    failure: *mut scanner_bridge::FjScannerLutDescriptorFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The foreign caller provides live, disjoint input/output storage.
    unsafe { scanner_bridge::describe_lut(input, out, failure, error) }
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_scanner_lut_descriptor_fault(fault: u32) -> FjStatus {
    if fault > 6 {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    film_juicer_core::scanner::test_support::arm_descriptor_fault(fault);
    status(FJ_STATUS_SUCCESS)
}

/// # Safety
/// Same synchronous live/readable/exclusive/disjoint contract as production.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_scanner_post_describe(
    input: *const scanner_bridge::FjScannerPostEffectsInput,
    out: *mut scanner_bridge::FjScannerPostEffectsDescriptor,
    failure: *mut scanner_bridge::FjScannerPostEffectsFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Test caller supplies the identical checked transport obligations.
    unsafe { scanner_bridge::describe_post(input, out, failure, error) }
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_scanner_post_fault(fault: u32) -> FjStatus {
    if fault > 2 {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    film_juicer_core::scanner::test_support::arm_post_fault(fault);
    status(FJ_STATUS_SUCCESS)
}
const _: unsafe extern "C" fn(
    *const scanner_bridge::FjScannerPostEffectsInput,
    *mut scanner_bridge::FjScannerPostEffectsDescriptor,
    *mut scanner_bridge::FjScannerPostEffectsFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_scanner_post_describe;

const _: unsafe extern "C" fn(
    *const scanner_bridge::FjScannerLutDescriptorInput,
    *mut scanner_bridge::FjScannerLutDescriptor,
    *mut scanner_bridge::FjScannerLutDescriptorFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_scanner_lut_describe;

/// # Safety
/// Same initialized/disjoint ownership-transfer contract as the production acquire.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_gaussian_kernel_acquire(
    input: *const crate::optics_bridge::FjGaussianKernelInput,
    out: *mut *mut crate::optics_bridge::FjGaussianKernelOwner,
    failure: *mut crate::optics_bridge::FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Identical checked implementation; no legacy-export dependency.
    unsafe { crate::optics_bridge::gaussian_acquire(input, out, failure, error) }
}
/// # Safety
/// Legitimate live owner excluding release; initialized disjoint outputs/diagnostics.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_gaussian_kernel_view(
    owner: *const crate::optics_bridge::FjGaussianKernelOwner,
    out: *mut crate::optics_bridge::FjGaussianKernelView,
    failure: *mut crate::optics_bridge::FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Identical checked implementation and borrowed-view expiry.
    unsafe { crate::optics_bridge::gaussian_view(owner, out, failure, error) }
}
/// # Safety
/// NULL or one matching live owner transfers once, excluding all views.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_gaussian_kernel_release(
    owner: *mut crate::optics_bridge::FjGaussianKernelOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Identical checked consuming implementation.
    unsafe { crate::optics_bridge::gaussian_release(owner, error) }
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_gaussian_kernel_fault(operation: u32, fault: u32) -> FjStatus {
    crate::optics_bridge::gaussian_faults::arm(operation, fault)
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_gaussian_kernel_acquires() -> usize {
    crate::optics_bridge::gaussian_faults::counts()[0]
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_gaussian_kernel_live_owners() -> usize {
    crate::optics_bridge::gaussian_faults::counts()[1]
}

#[unsafe(no_mangle)]
/// # Safety
/// Live aligned required input; exclusive/disjoint initialized writable peers and
/// optional bounded diagnostic backing, all borrowed through synchronous return.
unsafe extern "C" fn fj_test_frame_domain_prepare(
    input: *const crate::cuda::sys::FjRect,
    out: *mut crate::cuda::sys::FjExtent,
    failure: *mut crate::frame_bridge::FjFrameGeometryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade shares the production checked body and foreign contract.
    unsafe { crate::frame_bridge::prepare_domain(input, out, failure, error) }
}
#[unsafe(no_mangle)]
/// # Safety
/// Same fixed-record obligations as the frame-domain facade.
unsafe extern "C" fn fj_test_frame_geometry_prepare(
    input: *const crate::frame_bridge::FjFrameGeometryInput,
    out: *mut crate::frame_bridge::FjFrameGeometry,
    failure: *mut crate::frame_bridge::FjFrameGeometryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The facade shares the production checked body and foreign contract.
    unsafe { crate::frame_bridge::prepare_geometry(input, out, failure, error) }
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_frame_geometry_fault(operation: u32, fault: u32) -> FjStatus {
    crate::frame_bridge::faults::arm(operation, fault)
}
