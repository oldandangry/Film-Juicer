//! Temporary scalar boundary for C++ route consumers.

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
