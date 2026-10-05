//! Feature-only C fixture facade. Raw pointers never enter the safe core.

use std::fmt::{self, Write};
use std::mem::{align_of, offset_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use film_juicer_core::assets::{AssetError, Assets};
use film_juicer_core::profile::{Antihalation, ProfileCompletionErrorKind, ProfileUse};

use crate::asset_profile::{FilmOwner, FilmFixtureView};
use crate::cuda::sys::{FjErrorBuffer, FjFloatSpan, FjStatus, FjStringView};
use crate::cuda::sys::{
    FJ_API_NONE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE,
    FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT,
};

#[repr(C)]
struct FjFilmProfile {
    _opaque: [u8; 0],
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

/// Arm a one-shot calling-thread fault on a matching production export.
#[unsafe(no_mangle)]
extern "C" fn fj_test_cat16_arm_fault(operation: u32, call_index: u32, fault: u32) -> FjStatus {
    catch_unwind(|| crate::legacy_bridge::arm_cat16_fault(operation, call_index, fault))
        .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

/// Idempotently disarm the calling thread's fault.
#[unsafe(no_mangle)]
extern "C" fn fj_test_cat16_clear_fault() -> FjStatus {
    catch_unwind(crate::legacy_bridge::clear_cat16_fault)
        .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}

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
