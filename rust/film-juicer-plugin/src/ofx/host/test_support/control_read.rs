//! Independent current-control observations; this path never installs state.

use super::*;
use crate::asset_bridge::{Failure, run};
use crate::cuda::sys::{FjErrorBuffer, FjStatus};
use crate::preparation::FjPreparationParams;

struct FjOfxControlRead {
    params: film_juicer_core::recipe::Params,
}

fn valid<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(std::mem::align_of::<T>())
        && (pointer as usize)
            .checked_add(std::mem::size_of::<T>())
            .is_some()
}

/// # Safety
/// Effect/module and its host APIs remain live until return. Output/error and
/// diagnostic backing are initialized, aligned, exclusive and disjoint. Host
/// callbacks contain exceptions, run on this thread and retain no Rust borrow.
/// Success transfers an input observation whose keys live until its release.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_capture_host_input(
    effect: sys::OfxImageEffectHandle,
    out: *mut *mut FjOfxControlRead,
    error: *mut FjErrorBuffer,
) -> i32 {
    let output_valid = valid(out);
    if output_valid {
        // SAFETY: The caller admits the checked exclusive output slot.
        unsafe { out.write(std::ptr::null_mut()) };
    }
    let mut host_status = None;
    // SAFETY: The diagnostic contract is the existing facade's exclusive,
    // initialized extent. Containment covers the complete host read and owner.
    let result = unsafe {
        run(error, || {
            if !output_valid || effect.is_null() {
                return Err(Failure::Input("invalid OFX control observation arguments"));
            }
            let admission = super::super::super::admit_action(true)
                .map_err(|_| Failure::Input("OFX module is unavailable"))?;
            let callback = Callback::new(&admission, effect, std::ptr::null_mut());
            let owner = callback.instance().map_err(|status| {
                host_status = Some(status.code());
                Failure::Input("OFX instance is unavailable")
            })?;
            let captured =
                owner
                    .capture_host_input(&callback)
                    .map_err(|failure| match failure {
                        super::super::super::controls::CaptureError::Host(status) => {
                            host_status = Some(status.code());
                            Failure::Input("host control read failed")
                        }
                        super::super::super::controls::CaptureError::Invalid(text) => {
                            host_status = Some(sys::kOfxStatErrFatal);
                            Failure::CapturedControls(text)
                        }
                    })?;
            let (params, _) = captured.admit().map_err(|_| {
                host_status = Some(sys::kOfxStatErrFatal);
                Failure::Input("preparation key is not UTF-8")
            })?;
            // SAFETY: Output remains exclusive; the box owns all returned key bytes.
            // The observation owns no Runtime, Instance, publication or recipe.
            out.write(Box::into_raw(Box::new(FjOfxControlRead { params })));
            Ok(())
        })
    };
    match result {
        Ok(()) => sys::kOfxStatOK,
        Err(status) => host_status.unwrap_or(
            if status.category == crate::cuda::sys::FJ_STATUS_ALLOCATION_FAILURE {
                sys::kOfxStatErrMemory
            } else if status.category == crate::cuda::sys::FJ_STATUS_UNSUPPORTED_INPUT {
                sys::kOfxStatErrBadHandle
            } else {
                sys::kOfxStatErrFatal
            },
        ),
    }
}

/// # Safety
/// Owner is a matching live input observation, excluded from release. Output
/// and diagnostics are initialized, aligned, exclusive and mutually disjoint.
/// Returned key views expire at consuming release; nothing else is retained.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_control_input_view(
    owner: *const FjOfxControlRead,
    out: *mut FjPreparationParams,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let output_valid = valid(out);
    if output_valid {
        // SAFETY: Clearing is bounded by the caller's live exclusive record.
        unsafe { std::ptr::write_bytes(out, 0, 1) };
    }
    // SAFETY: Caller authorizes live matching owner and disjoint diagnostics;
    // numeric admission precedes dereference and the owner retains every span.
    unsafe {
        run(error, || {
            if !valid(owner) || !output_valid {
                return Err(Failure::Input("invalid OFX input observation"));
            }
            out.write(crate::instance_bridge::captured_projection(
                &(*owner).params,
            ));
            Ok(())
        })
    }
    .map_or_else(
        |status| status,
        |()| FjStatus {
            category: 0,
            api: 0,
            native_code: 0,
        },
    )
}

/// # Safety
/// Matching observation is consumed once with all key readers excluded.
/// Diagnostics are initialized, aligned, exclusive and disjoint through return.
#[cfg_attr(not(test), unsafe(no_mangle))]
unsafe extern "C" fn fj_test_ofx_control_input_release(
    owner: *mut FjOfxControlRead,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The caller supplies matching ownership and exclusive diagnostics.
    unsafe {
        if !valid(owner) {
            return run(error, || Err(Failure::Input("invalid OFX input owner"))).map_or_else(
                |status| status,
                |()| FjStatus {
                    category: 0,
                    api: 0,
                    native_code: 0,
                },
            );
        }
        let owner = Box::from_raw(owner);
        run(error, move || {
            drop(owner);
            Ok(())
        })
        .map_or_else(
            |status| status,
            |()| FjStatus {
                category: 0,
                api: 0,
                native_code: 0,
            },
        )
    }
}

const _: unsafe extern "C" fn(
    sys::OfxImageEffectHandle,
    *mut *mut FjOfxControlRead,
    *mut FjErrorBuffer,
) -> i32 = fj_test_ofx_capture_host_input;
const _: unsafe extern "C" fn(
    *const FjOfxControlRead,
    *mut FjPreparationParams,
    *mut FjErrorBuffer,
) -> FjStatus = fj_test_ofx_control_input_view;
const _: unsafe extern "C" fn(*mut FjOfxControlRead, *mut FjErrorBuffer) -> FjStatus =
    fj_test_ofx_control_input_release;
