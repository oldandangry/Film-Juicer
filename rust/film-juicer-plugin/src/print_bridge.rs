//! SF_TEMP_BRIDGE_PRINT_ILLUMINANT: synchronous fixed values; remove at S4.E.

use std::fmt::{self, Write};
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::print;

use crate::asset_illuminant::FjIlluminant;
use crate::cuda::sys::{
    FJ_API_NONE, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FjErrorBuffer, FjFloatSpan, FjStatus,
};

pub(crate) const FJ_PRINT_ILLUMINANT_REASON_NONE: u32 = 0;
pub(crate) const FJ_PRINT_ILLUMINANT_REASON_RAW: u32 = 1;
pub(crate) const FJ_PRINT_ILLUMINANT_REASON_SAMPLE: u32 = 2;
pub(crate) const FJ_PRINT_ILLUMINANT_REASON_ENERGY: u32 = 3;
pub(crate) const FJ_PRINT_ILLUMINANT_REASON_PANIC: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjPrintIlluminantParams {
    pub(crate) edges_nm: [f32; 4],
    pub(crate) transitions_nm: [f32; 4],
    pub(crate) cmy_cc: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjPrintIlluminantFailure {
    pub(crate) reason: u32,
    pub(crate) sample: u32,
}

const _: () = assert!(size_of::<FjPrintIlluminantParams>() == 44);
const _: () = assert!(align_of::<FjPrintIlluminantParams>() == 4);
const _: () = assert!(size_of::<FjPrintIlluminantFailure>() == 8);
const _: () = assert!(align_of::<FjPrintIlluminantFailure>() == 4);

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
    /// The optional initialized record and its positive-capacity backing are
    /// live, exclusive and disjoint from every other input/output through return.
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
        // SAFETY: Record admission precedes access. Zero capacity never reads
        // backing; positive backing is checked before its first write.
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
            // SAFETY: Admission and the bounded length preserve the exclusive
            // backing extent, including its trailing NUL.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data.add(self.length), count);
                self.length += count;
                self.data.add(self.length).write(0);
            }
        }
        if !self.record.is_null() {
            // SAFETY: The independently admitted diagnostic record remains live.
            unsafe { (*self.record).length = self.length };
        }
        Ok(())
    }
}

struct Failure {
    category: u32,
    details: FjPrintIlluminantFailure,
}

impl Failure {
    fn raw() -> Self {
        Self {
            category: FJ_STATUS_UNSUPPORTED_INPUT,
            details: details(FJ_PRINT_ILLUMINANT_REASON_RAW),
        }
    }

    fn panic() -> Self {
        Self {
            category: FJ_STATUS_INTERNAL_FAILURE,
            details: details(FJ_PRINT_ILLUMINANT_REASON_PANIC),
        }
    }

    fn core(error: print::Error) -> Self {
        Self {
            category: FJ_STATUS_PREPARATION_FAILURE,
            details: match error {
                print::Error::Sample { sample } => FjPrintIlluminantFailure {
                    reason: FJ_PRINT_ILLUMINANT_REASON_SAMPLE,
                    // The owning constructor produces only indices 0..81.
                    sample: sample as u32,
                },
                print::Error::Energy => details(FJ_PRINT_ILLUMINANT_REASON_ENERGY),
            },
        }
    }
}

fn details(reason: u32) -> FjPrintIlluminantFailure {
    FjPrintIlluminantFailure {
        reason,
        sample: u32::MAX,
    }
}

fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}

/// # Safety
/// Records and source are initialized, aligned and live. Source contains exactly
/// 81 f32 elements. Independently writable outputs and optional diagnostic
/// record/backing are exclusive, mutually disjoint and disjoint from all inputs.
/// Immutable input aliases are allowed. No pointer escapes synchronous return.
/// Numeric extent checks do not establish allocation, lifetime or initialization.
pub(crate) unsafe fn prepare(
    params: *const FjPrintIlluminantParams,
    source: FjFloatSpan,
    out: *mut FjIlluminant,
    failure: *mut FjPrintIlluminantFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    #[cfg(feature = "test-support")]
    let fault = crate::film_recipe_bridge::faults::take(10);
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    // SAFETY: Each complete independently admitted output is writable even if
    // its peer or another raw record will subsequently fail admission.
    unsafe {
        if valid_out {
            out.write(FjIlluminant::default());
        }
        if valid_failure {
            failure.write(details(FJ_PRINT_ILLUMINANT_REASON_NONE));
        }
    }
    // SAFETY: The caller authorizes optional diagnostics under the same contract.
    let diagnostic = unsafe { Diagnostic::admit(error) };
    let diagnostic_valid = diagnostic.is_ok();
    let mut diagnostic = diagnostic.ok();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if !valid_out
            || !valid_failure
            || !diagnostic_valid
            || !valid_extent(params, 1)
            || source.count != 81
            || !valid_extent(source.data, source.count)
        {
            return Err(Failure::raw());
        }
        // SAFETY: Structural admission precedes both caller-authorized immutable
        // borrows; these are live initialized inputs and no borrow escapes.
        let (params, source) = unsafe { (&*params, &*source.data.cast::<[f32; 81]>()) };
        #[cfg(feature = "test-support")]
        if fault == 1 {
            panic!("print illuminant fixture panic");
        }
        print::Illuminant::new(
            print::FilterParams {
                edges_nm: params.edges_nm,
                transitions_nm: params.transitions_nm,
                cmy_cc: params.cmy_cc,
            },
            source,
        )
        .map_err(Failure::core)
    }))
    .unwrap_or_else(|_| Err(Failure::panic()));
    match result {
        Ok(value) => {
            // SAFETY: Both outputs were independently admitted; commit only the
            // complete owned result after all construction checks succeed.
            unsafe {
                out.write(FjIlluminant {
                    samples: *value.samples(),
                });
                failure.write(details(FJ_PRINT_ILLUMINANT_REASON_NONE));
            }
            status(FJ_STATUS_SUCCESS)
        }
        Err(rejection) => {
            if let Some(writer) = &mut diagnostic {
                let _ = write!(
                    writer,
                    "print illuminant reason={} sample={}",
                    rejection.details.reason, rejection.details.sample
                );
            }
            if valid_failure {
                // SAFETY: This independently admitted output remains writable
                // even when a peer record failed raw admission.
                unsafe { failure.write(rejection.details) };
            }
            status(rejection.category)
        }
    }
}

/// # Safety
/// The checked operation's initialized, live, aligned and disjoint input/output contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_print_illuminant_prepare(
    params: *const FjPrintIlluminantParams,
    source: FjFloatSpan,
    out: *mut FjIlluminant,
    failure: *mut FjPrintIlluminantFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The same checked synchronous operation retains no foreign pointer.
    unsafe { prepare(params, source, out, failure, error) }
}

const _: unsafe extern "C" fn(
    *const FjPrintIlluminantParams,
    FjFloatSpan,
    *mut FjIlluminant,
    *mut FjPrintIlluminantFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_print_illuminant_prepare;

// SF_TEMP_BRIDGE_PRINT_PREFLASH: fixed synchronous integral; remove S4.E.
pub(crate) const FJ_PRINT_PREFLASH_REASON_NONE: u32 = 0;
pub(crate) const FJ_PRINT_PREFLASH_REASON_RAW: u32 = 1;
pub(crate) const FJ_PRINT_PREFLASH_REASON_CHANNEL: u32 = 2;
pub(crate) const FJ_PRINT_PREFLASH_REASON_PANIC: u32 = 3;

#[repr(C)]
pub(crate) struct FjPrintPreflashInput {
    pub(crate) film_base_density: FjFloatSpan,
    pub(crate) filtered_illuminant: FjFloatSpan,
    pub(crate) print_sensitivity_cmy: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjPrintPreflashRaw {
    pub(crate) cmy: [f32; 3],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjPrintPreflashFailure {
    pub(crate) reason: u32,
    pub(crate) channel: u32,
}

fn preflash_details(reason: u32) -> FjPrintPreflashFailure {
    FjPrintPreflashFailure {
        reason,
        channel: u32::MAX,
    }
}

struct PreflashFailure {
    category: u32,
    details: FjPrintPreflashFailure,
}
impl PreflashFailure {
    fn raw() -> Self {
        Self {
            category: FJ_STATUS_UNSUPPORTED_INPUT,
            details: preflash_details(FJ_PRINT_PREFLASH_REASON_RAW),
        }
    }
    fn panic() -> Self {
        Self {
            category: FJ_STATUS_INTERNAL_FAILURE,
            details: preflash_details(FJ_PRINT_PREFLASH_REASON_PANIC),
        }
    }
    fn core(error: print::PreflashError) -> Self {
        let print::PreflashError::Channel { channel } = error;
        Self {
            category: FJ_STATUS_PREPARATION_FAILURE,
            details: FjPrintPreflashFailure {
                reason: FJ_PRINT_PREFLASH_REASON_CHANNEL,
                channel: channel as u32,
            },
        }
    }
}

/// # Safety
/// Initialized, aligned live input record and backing contain exactly 81/81/243
/// f32 elements, the last sample-major C/M/Y. Immutable inputs may alias.
/// Writable outputs and optional diagnostic record/backing are live, exclusive,
/// mutually disjoint and disjoint from inputs through synchronous return.
/// Address/count admission does not establish allocation, initialization or lifetime.
pub(crate) unsafe fn prepare_preflash(
    input: *const FjPrintPreflashInput,
    out: *mut FjPrintPreflashRaw,
    failure: *mut FjPrintPreflashFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    #[cfg(feature = "test-support")]
    let fault = crate::film_recipe_bridge::faults::take(11);
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    // SAFETY: Independently admitted output records are writable even if a peer
    // or a later input fails admission. No source reference has been formed.
    unsafe {
        if valid_out {
            out.write(FjPrintPreflashRaw { cmy: [0.0; 3] });
        }
        if valid_failure {
            failure.write(preflash_details(FJ_PRINT_PREFLASH_REASON_NONE));
        }
    }
    // SAFETY: The caller authorizes the optional disjoint diagnostic storage.
    let diagnostic = unsafe { Diagnostic::admit(error) };
    let diagnostic_valid = diagnostic.is_ok();
    let mut diagnostic = diagnostic.ok();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if !valid_out || !valid_failure || !diagnostic_valid || !valid_extent(input, 1) {
            return Err(PreflashFailure::raw());
        }
        // SAFETY: The initialized record is structurally admitted before access.
        let input = unsafe { &*input };
        if input.film_base_density.count != 81
            || input.filtered_illuminant.count != 81
            || input.print_sensitivity_cmy.count != 243
            || !valid_extent(input.film_base_density.data, 81)
            || !valid_extent(input.filtered_illuminant.data, 81)
            || !valid_extent(input.print_sensitivity_cmy.data, 243)
        {
            return Err(PreflashFailure::raw());
        }
        #[cfg(feature = "test-support")]
        if fault == 1 {
            panic!("print preflash fixture panic");
        }
        // SAFETY: Exact, checked extents precede references to the live immutable
        // fixed arrays. Nested Rust arrays have the same contiguous f32 layout.
        // These borrows expire before return, including the permitted input aliases.
        let (base, light, sensitivity) = unsafe {
            (
                &*input.film_base_density.data.cast::<[f32; 81]>(),
                &*input.filtered_illuminant.data.cast::<[f32; 81]>(),
                &*input.print_sensitivity_cmy.data.cast::<[[f32; 3]; 81]>(),
            )
        };
        print::PreflashRaw::new(base, light, sensitivity).map_err(PreflashFailure::core)
    }))
    .unwrap_or_else(|_| Err(PreflashFailure::panic()));
    match result {
        Ok(value) => {
            // SAFETY: Both complete writable outputs were admitted; commit only
            // the finished triplet, including permitted f32 narrowing infinity.
            unsafe {
                out.write(FjPrintPreflashRaw { cmy: *value.cmy() });
                failure.write(preflash_details(FJ_PRINT_PREFLASH_REASON_NONE));
            }
            status(FJ_STATUS_SUCCESS)
        }
        Err(rejection) => {
            if let Some(writer) = &mut diagnostic {
                let _ = write!(
                    writer,
                    "print preflash reason={} channel={}",
                    rejection.details.reason, rejection.details.channel
                );
            }
            if valid_failure {
                // SAFETY: This independently admitted record remains writable.
                unsafe { failure.write(rejection.details) };
            }
            status(rejection.category)
        }
    }
}

/// # Safety
/// The same initialized, live, aligned and disjoint synchronous contract as prepare_preflash.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_print_preflash_prepare(
    input: *const FjPrintPreflashInput,
    out: *mut FjPrintPreflashRaw,
    failure: *mut FjPrintPreflashFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked operation retains no pointer and admits each output independently.
    unsafe { prepare_preflash(input, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const FjPrintPreflashInput,
    *mut FjPrintPreflashRaw,
    *mut FjPrintPreflashFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_print_preflash_prepare;
