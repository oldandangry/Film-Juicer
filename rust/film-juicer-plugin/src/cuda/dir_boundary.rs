//! Reflected-boundary projection for a cold native cache miss.

use std::ffi::c_void;
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use film_juicer_core::film::dir::boundary::{Boundary, Error, Input};

use super::{edge, sys::*};

const EMPTY_VIEW: FjDirBoundaryView = FjDirBoundaryView {
    weights: std::ptr::null(),
    count: 0,
    identity: 0,
    width: 0,
    height: 0,
    horizontal_length: 0,
    vertical_length: 0,
    horizontal_terminal_size: 0,
    vertical_terminal_size: 0,
    horizontal_terminal: [0.0; 9],
    vertical_terminal: [0.0; 9],
};

fn failure(reason: u32) -> FjDirBoundaryFailure {
    FjDirBoundaryFailure {
        reason,
        axis: 0,
        row: 3,
        length: 0,
        sigma_px: 0.0,
        sum: 0.0,
        tolerance: 0.0,
    }
}

impl From<Error> for FjDirBoundaryFailure {
    fn from(value: Error) -> Self {
        Self {
            reason: value.reason() as u32,
            axis: value.axis(),
            row: value.row(),
            length: value.length(),
            sigma_px: value.sigma_px(),
            sum: value.sum(),
            tolerance: value.tolerance(),
        }
    }
}

const MESSAGES: [&str; 14] = [
    "",
    "spatial DIR reflected-boundary arguments are invalid",
    "spatial DIR reflected-boundary request is invalid",
    "spatial DIR reflected-boundary axis size is invalid",
    "spatial DIR reflected-boundary policy is invalid",
    "spatial DIR causal reflected-boundary solve is singular",
    "spatial DIR causal reflected-boundary weights are nonfinite",
    "spatial DIR causal reflected-boundary certification failed",
    "spatial DIR reflected terminal system escaped its tail",
    "spatial DIR reflected terminal solve is singular",
    "spatial DIR reflected terminal certification failed",
    "spatial DIR reflected-boundary storage size is invalid",
    "spatial DIR reflected-boundary allocation failed",
    "spatial DIR reflected-boundary preparation panicked",
];

fn status(reason: u32) -> FjStatus {
    FjStatus {
        category: match reason {
            0 => 0,
            1 => 2,
            12 => 4,
            13 => 8,
            _ => 3,
        },
        api: 0,
        native_code: 0,
    }
}

fn valid<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && size_of::<T>() <= isize::MAX as usize
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}

/// # Safety
/// Non-null pointers refer to disjoint caller-owned ABI storage. This function
/// validates alignment and initializes the output records before calling body.
unsafe fn call(
    out: *mut FjDirBoundaryView,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
    body: impl FnOnce() -> Result<FjDirBoundaryView, FjDirBoundaryFailure>,
) -> FjStatus {
    let valid_out = valid(out);
    let valid_details = valid(details);
    // SAFETY: Clear each independently admitted peer before other admission.
    unsafe {
        if valid_out {
            out.write(EMPTY_VIEW);
        }
        if valid_details {
            details.write(failure(0));
        }
    }
    // SAFETY: Optional diagnostics have the same authorized storage/disjointness contract.
    let mut diagnostic = unsafe { edge::Diagnostic::admit(error) };
    let result = if !valid_out || !valid_details || diagnostic.is_err() {
        Err(failure(1))
    } else {
        catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|_| Err(failure(13)))
    };
    match result {
        Ok(value) => {
            // SAFETY: Successful independent admission established exclusive output storage.
            unsafe {
                out.write(value);
            }
            status(0)
        }
        Err(rejected) => {
            if valid_details {
                // SAFETY: This peer was independently admitted, even on another raw failure.
                unsafe {
                    details.write(rejected);
                }
            }
            if let Ok(writer) = &mut diagnostic {
                writer.write(MESSAGES[rejected.reason as usize]);
            }
            status(rejected.reason)
        }
    }
}

pub(super) fn project(raw: &FjDirBoundaryInput) -> Result<Input, FjDirBoundaryFailure> {
    let c = &raw.component;
    if !(c.reference_operator == 3 && c.weight > 0.0) {
        return Err(failure(2));
    }
    Input::new(
        [
            c.feedforward,
            c.feedback[0],
            c.feedback[1],
            c.feedback[2],
            c.normalization_denominator,
            c.feedforward_numerator,
            c.feedback_numerators[0],
            c.feedback_numerators[1],
            c.feedback_numerators[2],
        ],
        raw.width,
        raw.height,
        c.boundary_truncation_accuracy,
        c.boundary_certification_tolerance,
        c.boundary_derivation_version,
        c.sigma_px,
    )
    .map_err(Into::into)
}

/// # Safety
/// The native caller supplies live, disjoint callback records and the
/// outer render's backing. Invocations are synchronous and serial.
pub(super) unsafe extern "C" fn prepare(
    user: *mut c_void,
    input: *const FjDirBoundaryInput,
    out: *mut FjDirBoundaryView,
    details: *mut FjDirBoundaryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The callback contract establishes exclusive outputs and
    // serial access to the render-local backing once the records are checked.
    unsafe {
        call(out, details, error, || {
            if !valid(input) || !valid(user.cast::<super::host_preparation::Backing>()) {
                return Err(failure(FJ_DIR_BOUNDARY_RAW));
            }
            let request = &*input;
            let input = project(request)?;
            if input.key().value() != request.component.boundary_identity {
                return Err(failure(FJ_DIR_BOUNDARY_REQUEST));
            }
            let backing = &mut *user.cast::<super::host_preparation::Backing>();
            backing
                .boundaries
                .try_reserve(1)
                .map_err(|_| failure(FJ_DIR_BOUNDARY_ALLOCATION))?;
            let boundary = Boundary::new(input).map_err(FjDirBoundaryFailure::from)?;
            let view = FjDirBoundaryView {
                weights: boundary.weights().as_ptr(),
                count: boundary.weights().len(),
                identity: boundary.key().value(),
                width: boundary.width(),
                height: boundary.height(),
                horizontal_length: boundary.horizontal().length(),
                vertical_length: boundary.vertical().length(),
                horizontal_terminal_size: boundary.horizontal().terminal_size(),
                vertical_terminal_size: boundary.vertical().terminal_size(),
                horizontal_terminal: *boundary.horizontal().terminal(),
                vertical_terminal: *boundary.vertical().terminal(),
            };
            // Moving Boundary does not move its weight allocation. Subsequent
            // cold misses leave every previous view valid through native return.
            backing.boundaries.push(boundary);
            Ok(view)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_callback_keeps_first_weights_live_across_later_acquisition() {
        // Representative words from the tagged literal-9x6 capture. The native
        // installation witness compares all weights and both matrices.
        let expected_key = 11_586_933_080_928_547_342;
        let mut input = FjDirBoundaryInput {
            width: 9,
            height: 6,
            component: FjDirGaussian {
                sigma_px: 4.0,
                weight: 1.0,
                radius: 12,
                reference_operator: FJ_DIR_OPERATOR_YVV_REFLECT,
                feedforward: 0.5,
                feedback: [0.5, 0.0, 0.0],
                normalization_denominator: 2.0,
                feedforward_numerator: 1.0,
                feedback_numerators: [1.0, 0.0, 0.0],
                boundary_truncation_accuracy: 1e-11,
                boundary_certification_tolerance: 1e-6,
                boundary_derivation_version: 1,
                boundary_identity: 1,
            },
        };
        let call = crate::cuda::host_preparation::Call::new();
        let host = call.raw();
        let acquire = |input: &FjDirBoundaryInput, view: &mut FjDirBoundaryView| {
            let mut failure = failure(FJ_DIR_BOUNDARY_NONE);
            // SAFETY: The call retains backing, and the input and outputs are distinct.
            unsafe {
                host.dir_boundary.unwrap()(
                    host.user,
                    input,
                    view,
                    &mut failure,
                    std::ptr::null_mut(),
                )
            }
        };
        let mut first = EMPTY_VIEW;
        assert_eq!(
            acquire(&input, &mut first).category,
            FJ_STATUS_PREPARATION_FAILURE
        );
        assert!(first.weights.is_null());
        input.component.boundary_identity = project(&input)
            .unwrap_or_else(|_| panic!("invalid boundary"))
            .key()
            .value();
        assert_eq!(acquire(&input, &mut first).category, FJ_STATUS_SUCCESS);
        assert_eq!(first.identity, expected_key);
        let mut second = EMPTY_VIEW;
        std::mem::swap(&mut input.width, &mut input.height);
        input.component.boundary_identity = project(&input)
            .unwrap_or_else(|_| panic!("invalid boundary"))
            .key()
            .value();
        assert_eq!(acquire(&input, &mut second).category, FJ_STATUS_SUCCESS);
        assert_ne!(first.weights, second.weights);
        // SAFETY: Both callback views still borrow the live outer call.
        let actual = unsafe { std::slice::from_raw_parts(first.weights, first.count) };
        assert_eq!(actual.len(), 45);
        assert_eq!(actual[0].to_bits(), 4_602_678_870_712_451_073);
        assert_eq!(actual[44].to_bits(), 4_591_421_469_774_413_848);
        assert_eq!(
            first.horizontal_terminal[1].to_bits(),
            4_598_175_219_545_276_416
        );
        assert_eq!(
            first.vertical_terminal[8].to_bits(),
            4_607_182_418_800_017_408
        );
    }
}
