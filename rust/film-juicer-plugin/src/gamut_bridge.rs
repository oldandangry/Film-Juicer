//! Synchronous fixed input-hull projection. The native Library owns its cache.

use film_juicer_core::gamut;

use crate::asset_bridge::Failure;
use crate::asset_illuminant;
use crate::cuda::sys::{FjErrorBuffer, FjFloatSpan, FjStatus};

#[repr(C)]
pub(crate) struct FjInputHullSource {
    pub x: FjFloatSpan,
    pub y: FjFloatSpan,
    pub z: FjFloatSpan,
    pub d65: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjInputHull {
    pub center_xy: [f32; 2],
    pub xy: [[f32; 2]; 1025],
    pub hash: u64,
}
impl Default for FjInputHull {
    fn default() -> Self {
        Self {
            center_xy: [0.0; 2],
            xy: [[0.0; 2]; 1025],
            hash: 0,
        }
    }
}
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<FjInputHullSource>() == 64 && align_of::<FjInputHullSource>() == 8);
    assert!(offset_of!(FjInputHullSource, x) == 0);
    assert!(offset_of!(FjInputHullSource, y) == 16);
    assert!(offset_of!(FjInputHullSource, z) == 32);
    assert!(offset_of!(FjInputHullSource, d65) == 48);
    assert!(size_of::<FjInputHull>() == 8216 && align_of::<FjInputHull>() == 8);
    assert!(offset_of!(FjInputHull, center_xy) == 0);
    assert!(offset_of!(FjInputHull, xy) == 8);
    assert!(offset_of!(FjInputHull, hash) == 8208);
};
fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}
/// # Safety
/// The caller retains initialized immutable storage, disjoint from outputs,
/// through synchronous return. A checked address alone cannot prove its lifetime.
unsafe fn spectrum<'a>(span: FjFloatSpan) -> Result<&'a [f32; 81], Failure> {
    let bytes = span
        .count
        .checked_mul(size_of::<f32>())
        .ok_or(Failure::Input("input hull extent overflow"))?;
    if span.count != 81
        || bytes > isize::MAX as usize
        || (span.data as usize).checked_add(bytes).is_none()
        || !aligned(span.data)
    {
        return Err(Failure::Input("invalid input hull spectrum"));
    }
    // SAFETY: Exact fixed layout/addressable extent and caller-proven lifetime.
    Ok(unsafe { &*span.data.cast::<[f32; 81]>() })
}

/// # Safety
/// Required initialized records and spans are live and aligned. Writable output
/// and diagnostic storage are mutually disjoint and disjoint from every input.
pub(crate) unsafe fn hull_call(
    source: *const FjInputHullSource,
    out: *mut FjInputHull,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Only caller-authorized output clears. Diagnostic validation and
    // panic containment precede checked immutable borrows; no pointer escapes.
    unsafe {
        let valid_out = aligned(out);
        if valid_out {
            out.write(FjInputHull::default());
        }
        asset_illuminant::run(error, || {
            if !valid_out || !aligned(source) {
                return Err(Failure::Input("invalid input hull records"));
            }
            let source = &*source;
            let input = gamut::HullInput {
                x: spectrum(source.x)?,
                y: spectrum(source.y)?,
                z: spectrum(source.z)?,
                d65: spectrum(source.d65)?,
            };
            fault()?;
            let hull = gamut::build_input_hull(input).map_err(Failure::Hull)?;
            out.write(FjInputHull {
                center_xy: *hull.center(),
                xy: *hull.xy(),
                hash: hull.hash(),
            });
            Ok(())
        })
    }
}

fn production_fault() -> Result<(), Failure> {
    #[cfg(feature = "test-support")]
    return fixtures::inject(false);
    #[cfg(not(feature = "test-support"))]
    Ok(())
}
// FJ_TEMP_BRIDGE: completed input hull into the sole native cache; remove S4.E.
/// # Safety
/// The required fixed source/output/error storage obeys the legacy header.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_gamut_input_hull(
    source: *const FjInputHullSource,
    out: *mut FjInputHull,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked synchronous operation owns no retained foreign data.
    unsafe { hull_call(source, out, error, production_fault) }
}

#[cfg(feature = "test-support")]
mod fixtures {
    use super::*;
    use crate::cuda::sys::{FJ_API_NONE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT};
    thread_local! {
        static FAULT: std::cell::Cell<Option<(bool,u32)>> = const { std::cell::Cell::new(None) };
        static CONSUMED: std::cell::Cell<Option<(bool,u32)>> = const { std::cell::Cell::new(None) };
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_gamut_arm_fault(facade: u32, fault: u32) -> FjStatus {
        FAULT.set(None);
        CONSUMED.set(None);
        let valid = facade <= 1 && (1..=6).contains(&fault);
        if valid {
            FAULT.set(Some((facade != 0, fault)));
        }
        FjStatus {
            category: if valid {
                FJ_STATUS_SUCCESS
            } else {
                FJ_STATUS_UNSUPPORTED_INPUT
            },
            api: FJ_API_NONE,
            native_code: 0,
        }
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_gamut_clear_fault() {
        FAULT.set(None);
        CONSUMED.set(None);
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_gamut_fault_consumed(facade: u32, fault: u32) -> u32 {
        u32::from(CONSUMED.take() == Some((facade != 0, fault)))
    }
    pub(super) fn inject(facade: bool) -> Result<(), Failure> {
        let Some((selected, fault)) = FAULT.get() else {
            return Ok(());
        };
        if selected != facade {
            return Ok(());
        }
        FAULT.set(None);
        CONSUMED.set(Some((selected, fault)));
        match fault {
            1 => Err(Failure::Input("input hull boundary fault")),
            2 => panic!("input hull boundary fault"),
            3 => Err(Failure::Hull(gamut::HullError::LocusAndCenter)),
            4 => Err(Failure::Hull(gamut::HullError::RayIntersections)),
            5 => Err(Failure::Hull(gamut::HullError::InscribedScale)),
            6 => Err(Failure::Hull(gamut::HullError::Identity)),
            _ => unreachable!("closed input hull fault"),
        }
    }
    /// # Safety
    /// The same synchronous fixed borrow and exclusive output contract applies.
    #[unsafe(no_mangle)]
    unsafe extern "C" fn fj_test_gamut_input_hull(
        source: *const FjInputHullSource,
        out: *mut FjInputHull,
        error: *mut FjErrorBuffer,
    ) -> FjStatus {
        // SAFETY: This invokes the shared checked core operation, not the legacy export.
        unsafe { hull_call(source, out, error, || inject(true)) }
    }
}
