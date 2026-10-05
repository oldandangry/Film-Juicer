//! Temporary CPU boundaries for native route and color preparation.

use film_juicer_core::color::{self, Whites};
use crate::cuda::sys::{
    FjStatus, FJ_API_NONE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT,
    FJ_STATUS_INTERNAL_FAILURE,
};

use film_juicer_core::route::{self, CapturePolarity, RouteSelection, ScanRoute};

const SUCCESS: u32 = 0;
const INVALID_POLARITY: u32 = 1;
const INVALID_SELECTION: u32 = 2;
const NULL_OUTPUT: u32 = 3;
const INTERNAL_PANIC: u32 = 4;

fn resolve_foreign(capture_polarity: u8, selected_route: u8) -> Result<u8, u32> {
    let polarity = match capture_polarity {
        0 | 2 => CapturePolarity::Negative,
        1 => CapturePolarity::Positive,
        _ => return Err(INVALID_POLARITY),
    };
    let selection = match selected_route {
        0 | 2 => RouteSelection::Direct,
        1 | 3 => RouteSelection::Print,
        _ => return Err(INVALID_SELECTION),
    };
    Ok(match route::resolve(polarity, selection) {
        ScanRoute::NegativeDirect => 0,
        ScanRoute::NegativePrint => 1,
        ScanRoute::PositiveDirect => 2,
        ScanRoute::PositivePrint => 3,
    })
}

fn contain_panic(
    resolve: impl FnOnce() -> Result<u8, u32> + std::panic::UnwindSafe,
) -> Result<u8, u32> {
    std::panic::catch_unwind(resolve).unwrap_or(Err(INTERNAL_PANIC))
}

// FJ_TEMP_BRIDGE: route resolution; remove S6.D.
/// Resolve a route for the C++ callers.
///
/// # Safety
/// A nonnull `out_route` must point to one exclusively writable byte until return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_legacy_resolve_route(
    capture_polarity: u8,
    selected_route: u8,
    out_route: *mut u8,
) -> u32 {
    if out_route.is_null() {
        return NULL_OUTPUT;
    }
    match contain_panic(|| resolve_foreign(capture_polarity, selected_route)) {
        Ok(route) => {
            // SAFETY: The foreign caller guarantees one writable byte for this call.
            unsafe { out_route.write(route) };
            SUCCESS
        }
        Err(status) => status,
    }
}

const _: () = assert!(
    std::mem::size_of::<f32>() == 4
        && std::mem::size_of::<[f32; 3]>() == 12
        && std::mem::size_of::<[f32; 9]>() == 36
);

fn color_status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}

#[cfg(feature = "test-support")]
#[derive(Clone, Copy)]
struct Cat16Fault {
    operation: u32,
    remaining: u32,
    fault: u32,
}

#[cfg(feature = "test-support")]
thread_local! {
    static CAT16_FAULT: std::cell::Cell<Option<Cat16Fault>> = const { std::cell::Cell::new(None) };
}

#[cfg(feature = "test-support")]
pub(crate) fn arm_cat16_fault(operation: u32, call_index: u32, fault: u32) -> FjStatus {
    clear_cat16_fault();
    if !(1..=2).contains(&operation) || call_index == 0 || !(1..=2).contains(&fault) {
        return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    CAT16_FAULT.set(Some(Cat16Fault {
        operation,
        remaining: call_index,
        fault,
    }));
    color_status(FJ_STATUS_SUCCESS)
}

#[cfg(feature = "test-support")]
pub(crate) fn clear_cat16_fault() -> FjStatus {
    CAT16_FAULT.set(None);
    color_status(FJ_STATUS_SUCCESS)
}

#[cfg(feature = "test-support")]
fn cat16_fault(operation: u32) -> Result<(), u32> {
    let Some(mut armed) = CAT16_FAULT.get() else {
        return Ok(());
    };
    if armed.operation != operation {
        return Ok(());
    }
    armed.remaining -= 1;
    if armed.remaining != 0 {
        CAT16_FAULT.set(Some(armed));
        return Ok(());
    }
    CAT16_FAULT.set(None);
    if armed.fault == 1 {
        return Err(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    panic!("CAT16 production boundary fault");
}

// FJ_TEMP_BRIDGE: CAT16 host preparation; remove S4.E.
/// Prepare a row-major CAT16 matrix by sampling the three basis vectors.
///
/// # Safety
/// Nonnull inputs authorize three initialized aligned floats each until return;
/// they may alias each other. Nonnull output authorizes nine exclusive aligned
/// floats, disjoint from every input. No storage is retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_legacy_cat16_matrix(
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_row_major: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_row_major.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller provides the exclusive nine-float output extent.
        unsafe { out_row_major.write_bytes(0, 9) };
        if source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = cat16_fault(1) {
            return color_status(category);
        }
        // SAFETY: Both nonnull read-only inputs authorize exactly three floats.
        let whites = unsafe {
            Whites {
                source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
            }
        };
        let matrix = color::cat16_matrix(whites);
        // SAFETY: The complete local result is disjoint from the authorized output.
        unsafe { out_row_major.copy_from_nonoverlapping(matrix.as_ptr(), 9) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: CAT16 host preparation; remove S4.E.
/// Adapt an unsanitized XYZ value between the supplied whites.
///
/// # Safety
/// Nonnull inputs authorize three initialized aligned floats each until return;
/// they may alias each other. Nonnull output authorizes three exclusive aligned
/// floats disjoint from every input. No storage is retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_legacy_adapt_cat16(
    xyz: *const f32,
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller provides the exclusive three-float output extent.
        unsafe { out_xyz.write_bytes(0, 3) };
        if xyz.is_null() || source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = cat16_fault(2) {
            return color_status(category);
        }
        // SAFETY: Each nonnull read-only input authorizes exactly three floats.
        let (value, whites) = unsafe {
            (
                xyz.cast::<[f32; 3]>().read(),
                Whites {
                    source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                    destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
                },
            )
        };
        let adapted = color::adapt_cat16(value, whites);
        // SAFETY: The complete local result is disjoint from the authorized output.
        unsafe { out_xyz.copy_from_nonoverlapping(adapted.as_ptr(), 3) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_tags_preserve_the_reference_table() {
        let expected = [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1]];
        for (polarity, routes) in expected.into_iter().enumerate() {
            for (selection, route) in routes.into_iter().enumerate() {
                assert_eq!(resolve_foreign(polarity as u8, selection as u8), Ok(route));
            }
        }
    }

    #[test]
    fn foreign_failures_leave_output_unchanged() {
        let mut route = 99;
        // SAFETY: `route` is a live, exclusively borrowed output byte.
        unsafe {
            assert_eq!(
                fj_legacy_resolve_route(3, 0, &raw mut route),
                INVALID_POLARITY
            );
            assert_eq!(
                fj_legacy_resolve_route(0, 4, &raw mut route),
                INVALID_SELECTION
            );
            assert_eq!(
                fj_legacy_resolve_route(0, 0, std::ptr::null_mut()),
                NULL_OUTPUT
            );
            assert_eq!(route, 99);
            assert_eq!(fj_legacy_resolve_route(1, 0, &raw mut route), SUCCESS);
            assert_eq!(route, 2);
        }
    }

    #[test]
    fn panic_is_contained_as_status() {
        let result = contain_panic(|| panic!("test route panic"));
        assert_eq!(result, Err(INTERNAL_PANIC));
    }
}
