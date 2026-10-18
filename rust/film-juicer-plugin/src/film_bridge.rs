//! SF_TEMP_BRIDGE_FILM_DEVELOPMENT: synchronous completed-family transport; remove S4.E.

use film_juicer_core::film::{Development, Params};
use film_juicer_core::profile::Polarity;
use film_juicer_core::route::ScanRoute;
use std::sync::Arc;

use crate::asset_bridge::{Failure, run};
use crate::asset_profile::FilmOwner;
use crate::cuda::sys::{FJ_API_NONE, FJ_STATUS_SUCCESS, FjErrorBuffer, FjFloatSpan, FjStatus};
#[cfg(feature = "test-support")]
use crate::cuda::sys::FJ_STATUS_UNSUPPORTED_INPUT;

#[repr(C)]
pub(crate) struct FjFilmProfile {
    _opaque: [u8; 0],
}
#[repr(C)]
pub(crate) struct FjFilmDevelopmentOwner {
    _opaque: [u8; 0],
}
#[repr(C)]
pub(crate) struct FjFilmDevelopmentView {
    route: u32,
    polarity: u32,
    layers_required: u32,
    log_exposure: FjFloatSpan,
    authored_curves_cmy: FjFloatSpan,
    normalized_curves_cmy: FjFloatSpan,
    density_curves_layers: [[FjFloatSpan; 3]; 3],
    gamma_rgb: [f32; 3],
    minimum_cmy: [f32; 3],
    maximum_cmy: [f32; 3],
    normalized_curves_hash: u64,
    layers_hash: u64,
    hash: u64,
}

fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}
fn span(values: &[f32]) -> FjFloatSpan {
    FjFloatSpan {
        data: if values.is_empty() {
            std::ptr::null()
        } else {
            values.as_ptr()
        },
        count: values.len(),
    }
}
impl FjFilmDevelopmentView {
    fn new(development: &Development) -> Self {
        Self {
            route: match development.route() {
                ScanRoute::NegativeDirect => 0,
                ScanRoute::NegativePrint => 1,
                ScanRoute::PositiveDirect => 2,
                ScanRoute::PositivePrint => 3,
            },
            polarity: match development.polarity() {
                Polarity::Negative => 0,
                Polarity::Positive => 1,
            },
            layers_required: u32::from(development.layers_required()),
            log_exposure: span(development.log_exposure()),
            authored_curves_cmy: span(development.authored_curves_cmy().as_flattened()),
            normalized_curves_cmy: span(development.normalized_curves_cmy().as_flattened()),
            density_curves_layers: std::array::from_fn(|layer| {
                std::array::from_fn(|channel| {
                    span(
                        development
                            .density_curves_layers()
                            .map_or(&[], |lanes| lanes[layer][channel].as_slice()),
                    )
                })
            }),
            gamma_rgb: *development.gamma_rgb(),
            minimum_cmy: *development.minimum_cmy(),
            maximum_cmy: *development.maximum_cmy(),
            normalized_curves_hash: development.normalized_curves_hash(),
            layers_hash: development.layers_hash(),
            hash: development.hash(),
        }
    }
}

/// # Safety
/// Profile is a matching live FilmOwner. Output/diagnostic records and backing
/// are initialized, aligned, exclusively writable and mutually disjoint, also
/// disjoint from profile storage. All borrows last through synchronous return.
pub(crate) unsafe fn acquire(
    profile: *const FilmOwner,
    requested_route: u32,
    gamma_factor: f32,
    layers_required: u32,
    out: *mut *mut FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out.is_null() {
        // SAFETY: The caller authorizes this exclusive pointer slot.
        unsafe { out.write(std::ptr::null_mut()) };
    }
    // SAFETY: Authorized live profile and disjoint output/diagnostic storage.
    unsafe {
        run(error, || {
            if profile.is_null() || out.is_null() {
                return Err(Failure::Input("NULL film development profile/output"));
            }
            let requested_route = match requested_route {
                0 => ScanRoute::NegativeDirect,
                1 => ScanRoute::NegativePrint,
                2 => ScanRoute::PositiveDirect,
                3 => ScanRoute::PositivePrint,
                _ => return Err(Failure::Input("unsupported film development route")),
            };
            let layers_required = match layers_required {
                0 => false,
                1 => true,
                _ => return Err(Failure::Input("unsupported film development layers tag")),
            };
            #[cfg(feature = "test-support")]
            let fault = faults::take(0);
            #[cfg(feature = "test-support")]
            if (2..=5).contains(&fault) {
                film_juicer_core::film::test_support::reject_next(fault - 1);
            }
            let result = (&*profile).development(Params {
                gamma_factor,
                layers_required,
                requested_route,
            });
            #[cfg(feature = "test-support")]
            film_juicer_core::film::test_support::reject_next(0);
            let development = result.map_err(Failure::Film)?;
            #[cfg(feature = "test-support")]
            if fault == 1 {
                panic!("film development acquire fixture panic");
            }
            let owner = Arc::new(development);
            #[cfg(feature = "test-support")]
            faults::LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out.write(Arc::into_raw(owner).cast_mut().cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Owner is a live Arc raw hold of completed Development. Output is initialized,
/// aligned and exclusively writable, disjoint from owner and optional diagnostics.
/// The returned equal address has a separate consuming release obligation.
pub(crate) unsafe fn retain(
    owner: *const FjFilmDevelopmentOwner,
    out: *mut *mut FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let valid_out = !out.is_null()
        && (out as usize).is_multiple_of(std::mem::align_of::<*mut FjFilmDevelopmentOwner>())
        && (out as usize)
            .checked_add(std::mem::size_of::<*mut FjFilmDevelopmentOwner>())
            .is_some();
    if valid_out {
        // SAFETY: The caller authorizes this independently admitted exclusive slot.
        unsafe { out.write(std::ptr::null_mut()) };
    }
    // SAFETY: The owner remains live, and diagnostic storage follows the owner-family contract.
    unsafe {
        run(error, || {
            if !valid_out
                || owner.is_null()
                || !(owner as usize).is_multiple_of(std::mem::align_of::<Development>())
                || (owner as usize)
                    .checked_add(std::mem::size_of::<Development>())
                    .is_none()
            {
                return Err(Failure::Input(
                    "NULL or malformed film development retain owner/output",
                ));
            }
            Arc::increment_strong_count(owner.cast::<Development>());
            // Guard the provisional hold through all remaining fallible work.
            let retained = Arc::from_raw(owner.cast::<Development>());
            #[cfg(feature = "test-support")]
            if faults::take(3) == 1 {
                panic!("film development retain fixture panic");
            }
            #[cfg(feature = "test-support")]
            faults::LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out.write(Arc::into_raw(retained).cast_mut().cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Owner matches a live Development allocation. Its views expire at release.
/// Output and optional diagnostics obey acquire's aligned/disjoint contract.
pub(crate) unsafe fn view(
    owner: *const FjFilmDevelopmentOwner,
    out: *mut FjFilmDevelopmentView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out.is_null() {
        // SAFETY: Every field is a number or nullable pointer; zero clears semantic fields.
        unsafe { out.write(std::mem::zeroed()) };
    }
    // SAFETY: Owner storage stays live and immutable through the projection borrow.
    unsafe {
        run(error, || {
            if owner.is_null() || out.is_null() {
                return Err(Failure::Input("NULL film development view input/output"));
            }
            #[cfg(feature = "test-support")]
            if faults::take(1) == 1 {
                panic!("film development view fixture panic");
            }
            out.write(FjFilmDevelopmentView::new(&*owner.cast::<Development>()));
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Owner is NULL or a matching allocation consumed exactly once. Release
/// excludes all reads/views. Diagnostics obey the same exclusive contract.
pub(crate) unsafe fn release(
    owner: *mut FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let owner = if owner.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        faults::LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Take the matching allocation before diagnostics can reject the call.
        Some(unsafe { Arc::from_raw(owner.cast::<Development>()) })
    };
    // SAFETY: Consumed storage is dropped inside containment, including malformed diagnostics.
    unsafe {
        run(error, move || {
            #[cfg(feature = "test-support")]
            if faults::take(2) == 1 {
                panic!("film development release fixture panic");
            }
            drop(owner);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// Matching live profile; initialized/aligned/disjoint output and diagnostics.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_development_acquire(
    profile: *const FjFilmProfile,
    requested_route: u32,
    gamma_factor: f32,
    layers_required: u32,
    out: *mut *mut FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export forwards the same raw call contract.
    unsafe {
        acquire(
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
/// Matching live owner; initialized/aligned/disjoint output and diagnostics.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_development_view(
    owner: *const FjFilmDevelopmentOwner,
    out: *mut FjFilmDevelopmentView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export forwards the same raw call contract.
    unsafe { view(owner, out, error) }
}
/// # Safety
/// Matching owner consumed once (or NULL); exclusive/disjoint diagnostics.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_development_release(
    owner: *mut FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export forwards the same consuming call contract.
    unsafe { release(owner, error) }
}

/// # Safety
/// Live shared owner; authorized initialized/disjoint output and diagnostics.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_development_retain(
    owner: *const FjFilmDevelopmentOwner,
    out: *mut *mut FjFilmDevelopmentOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export forwards the same shared-hold contract.
    unsafe { retain(owner, out, error) }
}

#[cfg(feature = "test-support")]
mod faults {
    use std::cell::Cell;
    use std::sync::atomic::AtomicUsize;
    thread_local! { pub(super) static NEXT: Cell<(u32,u32)> = const { Cell::new((0,0)) }; }
    pub(super) static LIVE: AtomicUsize = AtomicUsize::new(0);
    pub(super) fn take(operation: u32) -> u32 {
        let (selected, fault) = NEXT.get();
        if operation == selected {
            NEXT.set((0, 0));
            fault
        } else {
            0
        }
    }
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_film_development_fault(operation: u32, fault: u32) -> FjStatus {
    if operation > 3 || fault > 5 || (operation != 0 && fault > 1) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    faults::NEXT.set((operation, fault));
    status(FJ_STATUS_SUCCESS)
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_film_development_live_owners() -> usize {
    faults::LIVE.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "test-support")]
const ABI_FACTS: &[usize] = &[
    std::mem::size_of::<FjFloatSpan>(),
    std::mem::align_of::<FjFloatSpan>(),
    std::mem::offset_of!(FjFloatSpan, data),
    std::mem::offset_of!(FjFloatSpan, count),
    std::mem::size_of::<*mut FjFilmDevelopmentOwner>(),
    std::mem::align_of::<*mut FjFilmDevelopmentOwner>(),
    std::mem::size_of::<FjFilmDevelopmentView>(),
    std::mem::align_of::<FjFilmDevelopmentView>(),
    std::mem::offset_of!(FjFilmDevelopmentView, route),
    std::mem::offset_of!(FjFilmDevelopmentView, polarity),
    std::mem::offset_of!(FjFilmDevelopmentView, layers_required),
    std::mem::offset_of!(FjFilmDevelopmentView, log_exposure),
    std::mem::offset_of!(FjFilmDevelopmentView, authored_curves_cmy),
    std::mem::offset_of!(FjFilmDevelopmentView, normalized_curves_cmy),
    std::mem::offset_of!(FjFilmDevelopmentView, density_curves_layers),
    std::mem::offset_of!(FjFilmDevelopmentView, gamma_rgb),
    std::mem::offset_of!(FjFilmDevelopmentView, minimum_cmy),
    std::mem::offset_of!(FjFilmDevelopmentView, maximum_cmy),
    std::mem::offset_of!(FjFilmDevelopmentView, normalized_curves_hash),
    std::mem::offset_of!(FjFilmDevelopmentView, layers_hash),
    std::mem::offset_of!(FjFilmDevelopmentView, hash),
    std::mem::size_of::<[[FjFloatSpan; 3]; 3]>() / std::mem::size_of::<FjFloatSpan>(),
    std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
    std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
    std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
    0,
    1,
    2,
    3,
    0,
    1,
];
/// # Safety
/// Nonnull count authorizes one aligned, exclusively writable usize.
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_film_development_abi_facts(count: *mut usize) -> *const usize {
    if !count.is_null() {
        // SAFETY: The caller authorizes the count output; facts are process-lifetime immutable.
        unsafe { count.write(ABI_FACTS.len()) };
    }
    ABI_FACTS.as_ptr()
}

const _: unsafe extern "C" fn(
    *const FjFilmProfile,
    u32,
    f32,
    u32,
    *mut *mut FjFilmDevelopmentOwner,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_film_development_acquire;
const _: unsafe extern "C" fn(
    *const FjFilmDevelopmentOwner,
    *mut FjFilmDevelopmentView,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_film_development_view;
const _: unsafe extern "C" fn(*mut FjFilmDevelopmentOwner, *mut FjErrorBuffer) -> FjStatus =
    fj_legacy_film_development_release;
