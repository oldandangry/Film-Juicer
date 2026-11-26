//! Checked optics values and call-local host preparation.

use super::sys::{FjGaussianKernelInput, FjGaussianKernelView, FjGaussianKernelFailure};
use std::fmt::{self, Write};
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use film_juicer_core::optics;
use crate::cuda::sys::{
    FjStatus, FjErrorBuffer, FJ_API_NONE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT,
    FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE,
};

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
struct Diagnostic {
    record: *mut FjErrorBuffer,
    data: *mut u8,
    capacity: usize,
    length: usize,
}
impl Diagnostic {
    /// # Safety
    /// Optional initialized live record and positive backing are exclusive and disjoint.
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
        // SAFETY: The admitted record is live; zero capacity never reads backing.
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
impl Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.capacity != 0 {
            let count = text.len().min(self.capacity - 1 - self.length);
            // SAFETY: Admission and the bounded length maintain the exclusive backing extent.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data.add(self.length), count);
                self.length += count;
                self.data.add(self.length).write(0);
            }
        }
        if !self.record.is_null() {
            // SAFETY: The independently admitted record remains exclusively live.
            unsafe {
                (*self.record).length = self.length;
            }
        }
        Ok(())
    }
}

// Gaussian input and borrowed coefficients for call-local host preparation.
impl FjGaussianKernelView {
    fn empty() -> Self {
        Self {
            radius: 0,
            sigma_px: 0.0,
            weights: crate::cuda::sys::FjFloatSpan {
                data: std::ptr::null(),
                count: 0,
            },
        }
    }
}

/// Clear a callback view when the preceding acquire failed.
///
/// # Safety
/// A nonnull `out` points to one writable, aligned, live view record.
pub(crate) unsafe fn clear_host_gaussian_view(out: *mut FjGaussianKernelView) {
    if valid_extent(out, 1) {
        // SAFETY: The caller supplied the writable view record admitted above.
        unsafe { out.write(FjGaussianKernelView::empty()) };
    }
}

/// Clear a callback output and initialize its failure reason.
///
/// # Safety
/// Each nonnull record points to writable, aligned, live storage.
pub(crate) unsafe fn fail_host_gaussian_view(
    out: *mut FjGaussianKernelView,
    failure: *mut FjGaussianKernelFailure,
    reason: u32,
) {
    // SAFETY: The callback's caller supplies the initialized output records.
    unsafe { clear_host_gaussian_view(out) };
    if valid_extent(failure, 1) {
        // SAFETY: The failure record was admitted above.
        unsafe { failure.write(FjGaussianKernelFailure { reason }) };
    }
}
#[derive(Clone, Copy)]
enum GaussianRejection {
    Raw,
    Core(optics::GaussianKernelError),
    Panic,
    // The callback candidate-vector reserve has a distinct message but uses
    // the existing allocation reason/category in the Gaussian failure record.
    BackingAllocation,
}
impl GaussianRejection {
    fn fields(self) -> (u32, u32, &'static str) {
        match self {
            Self::Raw => (
                1,
                FJ_STATUS_UNSUPPORTED_INPUT,
                "invalid Gaussian kernel arguments",
            ),
            Self::Core(optics::GaussianKernelError::Radius) => (
                2,
                FJ_STATUS_PREPARATION_FAILURE,
                "Gaussian kernel radius outside supported range",
            ),
            Self::Core(optics::GaussianKernelError::Sigma) => (
                3,
                FJ_STATUS_PREPARATION_FAILURE,
                "invalid Gaussian kernel sigma",
            ),
            Self::Core(optics::GaussianKernelError::Allocation) => (
                4,
                crate::cuda::sys::FJ_STATUS_ALLOCATION_FAILURE,
                "Gaussian kernel allocation failed",
            ),
            Self::Panic => (
                5,
                FJ_STATUS_INTERNAL_FAILURE,
                "Gaussian kernel preparation panicked",
            ),
            Self::BackingAllocation => (
                4,
                crate::cuda::sys::FJ_STATUS_ALLOCATION_FAILURE,
                "Gaussian kernel candidate storage allocation failed",
            ),
        }
    }
}
/// # Safety
/// Live initialized writable outputs/diagnostics are exclusive and mutually
/// disjoint, including input storage. Admission does not prove memory provenance.
unsafe fn gaussian_call<T: Copy>(
    out: *mut T,
    cleared: T,
    failure: *mut FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
    body: impl FnOnce() -> Result<T, GaussianRejection>,
) -> FjStatus {
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    // SAFETY: Clear each independently admitted writable peer before any other admission.
    unsafe {
        if valid_out {
            out.write(cleared);
        }
        if valid_failure {
            failure.write(FjGaussianKernelFailure { reason: 0 });
        }
    }
    // SAFETY: Optional diagnostic storage follows the same live/disjoint contract.
    let mut diagnostic = unsafe { Diagnostic::admit(error) };
    let result = if !valid_out || !valid_failure || diagnostic.is_err() {
        Err(GaussianRejection::Raw)
    } else {
        catch_unwind(AssertUnwindSafe(body)).unwrap_or(Err(GaussianRejection::Panic))
    };
    match result {
        Ok(value) => {
            // SAFETY: Successful admission established live exclusive output storage.
            unsafe {
                out.write(value);
            }
            status(FJ_STATUS_SUCCESS)
        }
        Err(rejection) => {
            let (reason, category, text) = rejection.fields();
            if valid_failure {
                // SAFETY: Failure storage was independently admitted before any input access.
                unsafe {
                    failure.write(FjGaussianKernelFailure { reason });
                }
            }
            if let Ok(writer) = &mut diagnostic {
                let _ = writer.write_str(text);
            }
            status(category)
        }
    }
}
/// Build and retain one Gaussian candidate for a native cache miss.
///
/// # Safety
/// The input and outputs follow the initialized, live, disjoint contract of
/// the synchronous native Gaussian callback.
pub(crate) unsafe fn prepare_host_gaussian(
    input: *const FjGaussianKernelInput,
    out: *mut FjGaussianKernelView,
    failure: *mut FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
    candidates: &mut Vec<optics::GaussianKernel>,
) -> FjStatus {
    // SAFETY: The callback provides live and disjoint records. gaussian_call
    // validates their extents before the closure commits any result.
    unsafe {
        gaussian_call(out, FjGaussianKernelView::empty(), failure, error, || {
            if !valid_extent(input, 1) {
                return Err(GaussianRejection::Raw);
            }
            let input = &*input;
            candidates
                .try_reserve(1)
                .map_err(|_| GaussianRejection::BackingAllocation)?;
            let kernel = optics::GaussianKernel::new(input.radius as u32, input.sigma_px)
                .map_err(GaussianRejection::Core)?;
            let view = FjGaussianKernelView {
                radius: kernel.radius() as i32,
                sigma_px: kernel.sigma_px(),
                weights: crate::cuda::sys::FjFloatSpan {
                    data: kernel.weights().as_ptr(),
                    count: kernel.weights().len(),
                },
            };
            candidates.push(kernel);
            Ok(view)
        })
    }
}

const _: fn(
    crate::cuda::sys::FjHalationGaussian,
    crate::cuda::sys::FjHalationChannel,
    crate::cuda::sys::FjScatterHalation,
) = |gaussian, channel, frame| {
    let _: u32 = gaussian.kind;
    let _: i32 = gaussian.radius;
    let _: [f32; 19] = gaussian.fir_weights;
    let _: [f32; 4] = gaussian.yvv;
    let _: crate::cuda::sys::FjHalationGaussian = channel.core;
    let _: [crate::cuda::sys::FjHalationGaussian; 3] = channel.tail;
    let _: [crate::cuda::sys::FjHalationGaussian; optics::HALATION_MAX_BOUNCES] = channel.bounce;
    let _: f32 = channel.total_strength;
    let _: f32 = channel.tail_weight;
    let _: u64 = frame.recipe_hash;
    let _: f32 = frame.scatter_amount;
    let _: u32 = frame.bounce_count;
    let _: [f32; optics::HALATION_MAX_BOUNCES] = frame.bounce_weights;
    let _: u32 = frame.renormalize;
    let _: [crate::cuda::sys::FjHalationChannel; 3] = frame.channels;
};
