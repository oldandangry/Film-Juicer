//! SF_TEMP_BRIDGE_SCANNER_CORRECTION: synchronous complete values; remove S4.E.

use std::fmt::{self, Write};
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::scanner;

use crate::cuda::sys::{
    FJ_API_NONE, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FjErrorBuffer, FjFloatSpan, FjStatus,
};

pub(crate) const FJ_SCANNER_CORRECTION_REASON_NONE: u32 = 0;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_RAW: u32 = 1;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_DIRECT_INPUTS: u32 = 2;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_DIRECT_REFERENCE: u32 = 3;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_DIRECT_SCAN: u32 = 4;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_DIRECT_MIDGRAY: u32 = 5;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_DIRECT_LOOKUP: u32 = 6;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_PRINT_INPUTS: u32 = 7;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_PRINT_REFERENCE_LOOKUP: u32 = 8;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_PRINT_SCAN: u32 = 9;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_PRINT_MIDGRAY: u32 = 10;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_PRINT_LOOKUP: u32 = 11;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_INVALID: u32 = 12;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_IDENTITY: u32 = 13;
pub(crate) const FJ_SCANNER_CORRECTION_REASON_PANIC: u32 = 14;
pub(crate) const FJ_SCANNER_CORRECTION_TRACE_NONE: u32 = 0;
pub(crate) const FJ_SCANNER_CORRECTION_TRACE_ACCEPTED: u32 = 1;
pub(crate) const FJ_SCANNER_CORRECTION_TRACE_MIDGRAY: u32 = 2;
pub(crate) const FJ_SCANNER_CORRECTION_TRACE_MALFORMED: u32 = 3;
pub(crate) const FJ_SCANNER_CORRECTION_TRACE_ZERO_HASH: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerCorrectionParams {
    pub(crate) route: u32,
    pub(crate) flags: u32,
    pub(crate) black_level: f32,
    pub(crate) white_level: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerCorrectionTables {
    pub(crate) eps_c: FjFloatSpan,
    pub(crate) eps_m: FjFloatSpan,
    pub(crate) eps_y: FjFloatSpan,
    pub(crate) weighted_y: FjFloatSpan,
    pub(crate) baseline_min: FjFloatSpan,
    pub(crate) sample_count: i32,
    pub(crate) inv_yn: f32,
    pub(crate) has_baseline: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerDirectCorrectionInput {
    pub(crate) tables: *const FjScannerCorrectionTables,
    pub(crate) film_base_density: FjFloatSpan,
    pub(crate) film_log_exposure: FjFloatSpan,
    pub(crate) film_authored_density_curves_cmy: FjFloatSpan,
    pub(crate) minimum_density_cmy: [f32; 3],
    pub(crate) maximum_density_cmy: [f32; 3],
    pub(crate) inverse_span_cmy: [f32; 3],
    pub(crate) baseline_density_cmy: [f32; 3],
    pub(crate) reference_hash: u64,
    pub(crate) medium: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerPrintCorrectionInput {
    pub(crate) tables: *const FjScannerCorrectionTables,
    pub(crate) film_base_density: FjFloatSpan,
    pub(crate) film_channel_density_cmy: FjFloatSpan,
    pub(crate) print_sensitivity_cmy: FjFloatSpan,
    pub(crate) filtered_main_illuminant: FjFloatSpan,
    pub(crate) print_base_density: FjFloatSpan,
    pub(crate) print_log_exposure: FjFloatSpan,
    pub(crate) print_density_curves_cmy: FjFloatSpan,
    pub(crate) minimum_density_cmy: [f32; 3],
    pub(crate) inverse_span_cmy: [f32; 3],
    pub(crate) film_minimum_density_cmy: [f32; 3],
    pub(crate) film_maximum_density_cmy: [f32; 3],
    pub(crate) preflash_raw_cmy: [f32; 3],
    pub(crate) normalizer: f32,
    pub(crate) preflash_exposure: f32,
    pub(crate) medium: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjScannerColorCorrection {
    pub(crate) route: u32,
    pub(crate) active: u32,
    pub(crate) black_correction: u32,
    pub(crate) white_correction: u32,
    pub(crate) target_black_linear: f32,
    pub(crate) target_white_linear: f32,
    pub(crate) reference_black_y: f32,
    pub(crate) reference_white_y: f32,
    pub(crate) xyz_slope: f32,
    pub(crate) xyz_offset: f32,
    pub(crate) exposure_scale: f32,
    pub(crate) schema_version: u32,
    pub(crate) hash: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerCorrectionFailure {
    pub(crate) reason: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerCorrectionTrace {
    pub(crate) outcome: u32,
    pub(crate) reference_black_y: f32,
    pub(crate) reference_white_y: f32,
    pub(crate) target_black_linear: f32,
    pub(crate) target_white_linear: f32,
    pub(crate) xyz_slope: f32,
    pub(crate) xyz_offset: f32,
    pub(crate) corrected_midgray: f32,
    pub(crate) exposure_scale: f32,
    pub(crate) hash: u64,
}
impl Default for FjScannerCorrectionTrace {
    fn default() -> Self {
        Self {
            outcome: FJ_SCANNER_CORRECTION_TRACE_NONE,
            reference_black_y: 0.0,
            reference_white_y: 0.0,
            target_black_linear: 0.0,
            target_white_linear: 0.0,
            xyz_slope: 0.0,
            xyz_offset: 0.0,
            corrected_midgray: 0.0,
            exposure_scale: 0.0,
            hash: 0,
        }
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
    /// The optional initialized record and its nonempty backing are exclusive,
    /// live and disjoint from all other inputs/outputs through synchronous return.
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
        // SAFETY: Record admission precedes access; positive-capacity backing
        // is checked before use. Capacity zero never accesses its pointer.
        unsafe {
            (*record).length = 0;
            writer.capacity = (*record).capacity;
            if writer.capacity != 0 {
                writer.data = (*record).data.cast();
                if !valid_extent(writer.data, writer.capacity) {
                    return Err(());
                }
                writer.data.write(0);
            }
        }
        Ok(writer)
    }
}
impl Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.capacity != 0 {
            let count = text.len().min(self.capacity - 1 - self.length);
            // SAFETY: Admitted backing and bounded length include the final NUL.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data.add(self.length), count);
                self.length += count;
                self.data.add(self.length).write(0);
            }
        }
        if !self.record.is_null() {
            // SAFETY: The independently admitted diagnostic record stays live.
            unsafe {
                (*self.record).length = self.length;
            }
        }
        Ok(())
    }
}

enum Failure {
    Raw,
    Panic,
    Core(scanner::CorrectionError),
}
impl Failure {
    fn category(&self) -> u32 {
        match self {
            Self::Raw => FJ_STATUS_UNSUPPORTED_INPUT,
            Self::Panic => FJ_STATUS_INTERNAL_FAILURE,
            Self::Core(_) => FJ_STATUS_PREPARATION_FAILURE,
        }
    }
    fn reason(&self) -> u32 {
        use scanner::CorrectionError;
        match self {
            Self::Raw => FJ_SCANNER_CORRECTION_REASON_RAW,
            Self::Panic => FJ_SCANNER_CORRECTION_REASON_PANIC,
            Self::Core(error) => match error {
                CorrectionError::DirectInputs => FJ_SCANNER_CORRECTION_REASON_DIRECT_INPUTS,
                CorrectionError::DirectReference => FJ_SCANNER_CORRECTION_REASON_DIRECT_REFERENCE,
                CorrectionError::DirectScan => FJ_SCANNER_CORRECTION_REASON_DIRECT_SCAN,
                CorrectionError::DirectMidgray(_) => FJ_SCANNER_CORRECTION_REASON_DIRECT_MIDGRAY,
                CorrectionError::DirectLookup => FJ_SCANNER_CORRECTION_REASON_DIRECT_LOOKUP,
                CorrectionError::PrintInputs => FJ_SCANNER_CORRECTION_REASON_PRINT_INPUTS,
                CorrectionError::PrintReferenceLookup => {
                    FJ_SCANNER_CORRECTION_REASON_PRINT_REFERENCE_LOOKUP
                }
                CorrectionError::PrintScan => FJ_SCANNER_CORRECTION_REASON_PRINT_SCAN,
                CorrectionError::PrintMidgray(_) => FJ_SCANNER_CORRECTION_REASON_PRINT_MIDGRAY,
                CorrectionError::PrintLookup => FJ_SCANNER_CORRECTION_REASON_PRINT_LOOKUP,
                CorrectionError::Invalid(_) => FJ_SCANNER_CORRECTION_REASON_INVALID,
                CorrectionError::Identity(_) => FJ_SCANNER_CORRECTION_REASON_IDENTITY,
            },
        }
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, writer: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Raw => writer.write_str("scanner correction raw input"),
            Self::Panic => writer.write_str("scanner correction contained panic"),
            Self::Core(error) => error.fmt(writer),
        }
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Direct,
    Print,
}
impl Operation {
    fn demand(self, params: scanner::Params) -> Result<bool, Failure> {
        match self {
            Self::Direct => params.direct_inputs_required(),
            Self::Print => params.print_inputs_required(),
        }
        .map_err(Failure::Core)
    }
    fn inputs_failure(self) -> Failure {
        Failure::Core(match self {
            Self::Direct => scanner::CorrectionError::DirectInputs,
            Self::Print => scanner::CorrectionError::PrintInputs,
        })
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
/// Admitted, live initialized Params are borrowed only through this call.
unsafe fn params(pointer: *const FjScannerCorrectionParams) -> Result<scanner::Params, Failure> {
    if !valid_extent(pointer, 1) {
        return Err(Failure::Raw);
    }
    // SAFETY: Admission precedes this synchronous immutable record borrow.
    let input = unsafe { &*pointer };
    if input.route > 255 || input.flags & !3 != 0 {
        return Err(Failure::Raw);
    }
    Ok(scanner::Params {
        route: input.route as u8,
        black_correction: input.flags & 1 != 0,
        white_correction: input.flags & 2 != 0,
        black_level: input.black_level,
        white_level: input.white_level,
    })
}

/// # Safety
/// Nonempty backing is live initialized immutable f32 storage for this invocation.
/// This private projection never escapes the checked synchronous operation.
unsafe fn span<'a>(input: FjFloatSpan) -> Result<&'a [f32], Failure> {
    if input.count == 0 {
        return Ok(&[]);
    }
    if !valid_extent(input.data, input.count) {
        return Err(Failure::Raw);
    }
    // SAFETY: Extent/alignment admission and the caller's lifetime contract precede the borrow.
    Ok(unsafe { std::slice::from_raw_parts(input.data, input.count) })
}
/// # Safety
/// Same invocation-bounded live immutable storage contract as span.
unsafe fn fixed<'a, const N: usize>(input: FjFloatSpan) -> Result<&'a [f32; N], Failure> {
    if input.count != N {
        return Err(Failure::Raw);
    }
    // SAFETY: span checks the exact extent before constructing the local borrow.
    unsafe { span(input) }?.try_into().map_err(|_| Failure::Raw)
}
/// # Safety
/// Same invocation-bounded live immutable storage contract as span.
unsafe fn rows<'a>(input: FjFloatSpan) -> Result<&'a [[f32; 3]], Failure> {
    if !input.count.is_multiple_of(3) {
        return Err(Failure::Raw);
    }
    // SAFETY: Structural checks precede the local borrow; as_chunks is safe.
    Ok(unsafe { span(input) }?.as_chunks::<3>().0)
}
/// # Safety
/// Same invocation-bounded live immutable storage contract as span.
unsafe fn fixed_rows<'a>(input: FjFloatSpan) -> Result<&'a [[f32; 3]; 81], Failure> {
    if input.count != 243 {
        return Err(Failure::Raw);
    }
    // SAFETY: rows admits the exact sample-major backing before borrowing.
    unsafe { rows(input) }?.try_into().map_err(|_| Failure::Raw)
}
/// # Safety
/// The optional table and consumed spans are initialized and live through return.
unsafe fn tables<'a>(
    pointer: *const FjScannerCorrectionTables,
    operation: Operation,
) -> Result<scanner::ReferenceTables<'a>, Failure> {
    if pointer.is_null() {
        return Err(operation.inputs_failure());
    }
    if !valid_extent(pointer, 1) {
        return Err(Failure::Raw);
    }
    // SAFETY: Table record admission precedes reading its count/tags/spans.
    let input = unsafe { &*pointer };
    if input.sample_count <= 0 {
        return Err(operation.inputs_failure());
    }
    if input.has_baseline > 1 {
        return Err(Failure::Raw);
    }
    let count = input.sample_count as usize;
    let selected = |value: FjFloatSpan| {
        if value.count != count {
            return Err(Failure::Raw);
        }
        // SAFETY: The exact selected extent is live through the checked invocation.
        unsafe { span(value) }
    };
    Ok(scanner::ReferenceTables {
        eps_c: selected(input.eps_c)?,
        eps_m: selected(input.eps_m)?,
        eps_y: selected(input.eps_y)?,
        weighted_y: selected(input.weighted_y)?,
        baseline_min: if input.has_baseline == 1 {
            Some(selected(input.baseline_min)?)
        } else {
            None
        },
        sample_count: input.sample_count,
        inv_yn: input.inv_yn,
    })
}

/// # Safety
/// Demand has succeeded. Active records/consumed spans satisfy the synchronous
/// initialized immutable input contract. No projected borrow escapes prepare.
unsafe fn direct_input<'a>(
    pointer: *const FjScannerDirectCorrectionInput,
) -> Result<Option<scanner::DirectInput<'a>>, Failure> {
    if pointer.is_null() {
        return Ok(None);
    }
    if !valid_extent(pointer, 1) {
        return Err(Failure::Raw);
    }
    // SAFETY: Record admission precedes its first access.
    let input = unsafe { &*pointer };
    if input.medium > 255 {
        return Err(Failure::Raw);
    }
    // SAFETY: Each helper admits consumed extents before creating call-local borrows.
    unsafe {
        Ok(Some(scanner::DirectInput {
            tables: tables(input.tables, Operation::Direct)?,
            film_base_density: fixed(input.film_base_density)?,
            film_log_exposure: span(input.film_log_exposure)?,
            film_authored_density_curves: rows(input.film_authored_density_curves_cmy)?,
            minimum_density_cmy: input.minimum_density_cmy,
            maximum_density_cmy: input.maximum_density_cmy,
            inverse_span_cmy: input.inverse_span_cmy,
            baseline_density_cmy: input.baseline_density_cmy,
            reference_hash: input.reference_hash,
            medium: input.medium as u8,
        }))
    }
}
/// # Safety
/// Same demand-scoped immutable input/lifetime contract as direct_input.
unsafe fn print_input<'a>(
    pointer: *const FjScannerPrintCorrectionInput,
) -> Result<Option<scanner::PrintInput<'a>>, Failure> {
    if pointer.is_null() {
        return Ok(None);
    }
    if !valid_extent(pointer, 1) {
        return Err(Failure::Raw);
    }
    // SAFETY: Record admission precedes its first access.
    let input = unsafe { &*pointer };
    if input.medium > 255 {
        return Err(Failure::Raw);
    }
    // SAFETY: Each helper admits consumed extents before creating call-local borrows.
    unsafe {
        Ok(Some(scanner::PrintInput {
            tables: tables(input.tables, Operation::Print)?,
            film_base_density: fixed(input.film_base_density)?,
            film_channel_density_cmy: fixed_rows(input.film_channel_density_cmy)?,
            print_sensitivity_cmy: fixed_rows(input.print_sensitivity_cmy)?,
            filtered_main_illuminant: fixed(input.filtered_main_illuminant)?,
            print_base_density: fixed(input.print_base_density)?,
            print_log_exposure: span(input.print_log_exposure)?,
            print_density_curves: rows(input.print_density_curves_cmy)?,
            minimum_density_cmy: input.minimum_density_cmy,
            inverse_span_cmy: input.inverse_span_cmy,
            film_minimum_density_cmy: input.film_minimum_density_cmy,
            film_maximum_density_cmy: input.film_maximum_density_cmy,
            preflash_raw_cmy: input.preflash_raw_cmy,
            normalizer: input.normalizer,
            preflash_exposure: input.preflash_exposure,
            medium: input.medium as u8,
        }))
    }
}

fn project(value: scanner::Correction) -> FjScannerColorCorrection {
    FjScannerColorCorrection {
        route: u32::from(value.route()),
        active: u32::from(value.active()),
        black_correction: u32::from(value.black_correction()),
        white_correction: u32::from(value.white_correction()),
        target_black_linear: value.target_black_linear(),
        target_white_linear: value.target_white_linear(),
        reference_black_y: value.reference_black_y(),
        reference_white_y: value.reference_white_y(),
        xyz_slope: value.xyz_slope(),
        xyz_offset: value.xyz_offset(),
        exposure_scale: value.exposure_scale(),
        schema_version: value.schema_version(),
        hash: value.hash(),
    }
}
fn project_trace(trace: &scanner::Trace) -> FjScannerCorrectionTrace {
    let v = trace.values();
    FjScannerCorrectionTrace {
        outcome: match trace.outcome() {
            scanner::TraceOutcome::Accepted => FJ_SCANNER_CORRECTION_TRACE_ACCEPTED,
            scanner::TraceOutcome::Midgray => FJ_SCANNER_CORRECTION_TRACE_MIDGRAY,
            scanner::TraceOutcome::Malformed => FJ_SCANNER_CORRECTION_TRACE_MALFORMED,
            scanner::TraceOutcome::ZeroHash => FJ_SCANNER_CORRECTION_TRACE_ZERO_HASH,
        },
        reference_black_y: v[0],
        reference_white_y: v[1],
        target_black_linear: v[2],
        target_white_linear: v[3],
        xyz_slope: v[4],
        xyz_offset: v[5],
        corrected_midgray: v[6],
        exposure_scale: v[7],
        hash: trace.hash(),
    }
}

/// # Safety
/// Valid writable flags establish which independently admitted records may be
/// written; optional diagnostics were admitted under their exclusive contract.
unsafe fn reject(
    rejection: Failure,
    failure: *mut FjScannerCorrectionFailure,
    valid_failure: bool,
    trace: *mut FjScannerCorrectionTrace,
    valid_trace: bool,
    diagnostic: &mut Option<Diagnostic>,
) -> FjStatus {
    if let Some(writer) = diagnostic {
        let _ = write!(writer, "{rejection}");
    }
    // SAFETY: Only independently admitted exclusive outputs are written.
    unsafe {
        if valid_failure {
            failure.write(FjScannerCorrectionFailure {
                reason: rejection.reason(),
            });
        }
        if valid_trace
            && !trace.is_null()
            && let Failure::Core(error) = &rejection
            && let Some(context) = error.trace()
        {
            trace.write(project_trace(context));
        }
    }
    status(rejection.category())
}

/// # Safety
/// Required/optional records and their initialized storage obey the header's
/// synchronous, mutually disjoint writable and immutable input contract.
unsafe fn prepare(
    params_pointer: *const FjScannerCorrectionParams,
    out: *mut FjScannerColorCorrection,
    failure: *mut FjScannerCorrectionFailure,
    trace: *mut FjScannerCorrectionTrace,
    error: *mut FjErrorBuffer,
    operation: Operation,
    construct: impl FnOnce(scanner::Params, bool) -> Result<scanner::Correction, Failure>,
) -> FjStatus {
    #[cfg(feature = "test-support")]
    let inject = faults::take(operation);
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    let valid_trace = trace.is_null() || valid_extent(trace, 1);
    // SAFETY: Each admitted writable output is independent of invalid peers.
    unsafe {
        if valid_out {
            out.write(FjScannerColorCorrection::default());
        }
        if valid_failure {
            failure.write(FjScannerCorrectionFailure {
                reason: FJ_SCANNER_CORRECTION_REASON_NONE,
            });
        }
        if valid_trace && !trace.is_null() {
            trace.write(FjScannerCorrectionTrace::default());
        }
    }
    // SAFETY: The caller authorizes optional exclusive diagnostics through return.
    let diagnostic = unsafe { Diagnostic::admit(error) };
    let valid_diagnostic = diagnostic.is_ok();
    let mut diagnostic = diagnostic.ok();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if !valid_out || !valid_failure || !valid_trace || !valid_diagnostic {
            return Err(Failure::Raw);
        }
        // SAFETY: Params is independently admitted before forming a reference.
        let params = unsafe { params(params_pointer) }?;
        let required = operation.demand(params)?;
        #[cfg(feature = "test-support")]
        if inject {
            panic!("scanner correction fixture panic");
        }
        construct(params, required)
    }))
    .unwrap_or(Err(Failure::Panic));
    match result {
        Ok(value) => {
            // SAFETY: Commit only complete immutable values to admitted outputs.
            unsafe {
                out.write(project(value));
                if !trace.is_null()
                    && let Some(context) = value.trace()
                {
                    trace.write(project_trace(context));
                }
            }
            status(FJ_STATUS_SUCCESS)
        }
        // SAFETY: The flags precisely describe independent writable admission.
        Err(rejection) => unsafe {
            reject(
                rejection,
                failure,
                valid_failure,
                trace,
                valid_trace,
                &mut diagnostic,
            )
        },
    }
}

/// # Safety
/// The complete header contract: initialized live input/consumed backing,
/// mutually exclusive/disjoint writable records, no borrow beyond return.
pub(crate) unsafe fn prepare_direct(
    params: *const FjScannerCorrectionParams,
    input: *const FjScannerDirectCorrectionInput,
    out: *mut FjScannerColorCorrection,
    failure: *mut FjScannerCorrectionFailure,
    trace: *mut FjScannerCorrectionTrace,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Common admission and shared demand precede all active input access.
    unsafe {
        prepare(
            params,
            out,
            failure,
            trace,
            error,
            Operation::Direct,
            |params, required| {
                let input = if required { direct_input(input)? } else { None };
                scanner::Correction::direct(params, input).map_err(Failure::Core)
            },
        )
    }
}
/// # Safety
/// Same synchronous header contract as prepare_direct, with print-only borrows.
pub(crate) unsafe fn prepare_print(
    params: *const FjScannerCorrectionParams,
    input: *const FjScannerPrintCorrectionInput,
    out: *mut FjScannerColorCorrection,
    failure: *mut FjScannerCorrectionFailure,
    trace: *mut FjScannerCorrectionTrace,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Common admission and shared demand precede all active input access.
    unsafe {
        prepare(
            params,
            out,
            failure,
            trace,
            error,
            Operation::Print,
            |params, required| {
                let input = if required { print_input(input)? } else { None };
                scanner::Correction::print(params, input).map_err(Failure::Core)
            },
        )
    }
}

/// # Safety
/// Params and required/failure/optional error obey the same live/disjoint header
/// contract. The query reads no active storage and consumes no fault selector.
pub(crate) unsafe fn input_demand(
    params_pointer: *const FjScannerCorrectionParams,
    operation: u32,
    required: *mut u32,
    failure: *mut FjScannerCorrectionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let valid_required = valid_extent(required, 1);
    let valid_failure = valid_extent(failure, 1);
    // SAFETY: Independently admitted writable outputs may be cleared before peers fail.
    unsafe {
        if valid_required {
            required.write(0);
        }
        if valid_failure {
            failure.write(FjScannerCorrectionFailure {
                reason: FJ_SCANNER_CORRECTION_REASON_NONE,
            });
        }
    }
    // SAFETY: Optional diagnostics obey the documented exclusive backing contract.
    let diagnostic = unsafe { Diagnostic::admit(error) };
    let valid_diagnostic = diagnostic.is_ok();
    let mut diagnostic = diagnostic.ok();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if !valid_required || !valid_failure || !valid_diagnostic {
            return Err(Failure::Raw);
        }
        // SAFETY: Params is admitted before its immutable borrow is formed.
        let params = unsafe { params(params_pointer) }?;
        let operation = match operation {
            0 => Operation::Direct,
            1 => Operation::Print,
            _ => return Err(Failure::Raw),
        };
        operation.demand(params)
    }))
    .unwrap_or(Err(Failure::Panic));
    match result {
        Ok(value) => {
            // SAFETY: The required output is independently admitted and exclusive.
            unsafe {
                required.write(u32::from(value));
            }
            status(FJ_STATUS_SUCCESS)
        }
        // SAFETY: No trace output is present; admitted failure/diagnostic remain live.
        Err(rejection) => unsafe {
            reject(
                rejection,
                failure,
                valid_failure,
                std::ptr::null_mut(),
                true,
                &mut diagnostic,
            )
        },
    }
}

/// # Safety
/// Same synchronous live/disjoint header contract as prepare_direct.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_scanner_direct_correction_prepare(
    params: *const FjScannerCorrectionParams,
    input: *const FjScannerDirectCorrectionInput,
    out: *mut FjScannerColorCorrection,
    failure: *mut FjScannerCorrectionFailure,
    trace: *mut FjScannerCorrectionTrace,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked operation admits all consumed storage and retains none.
    unsafe { prepare_direct(params, input, out, failure, trace, error) }
}
/// # Safety
/// Same synchronous live/disjoint header contract as prepare_print.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_scanner_print_correction_prepare(
    params: *const FjScannerCorrectionParams,
    input: *const FjScannerPrintCorrectionInput,
    out: *mut FjScannerColorCorrection,
    failure: *mut FjScannerCorrectionFailure,
    trace: *mut FjScannerCorrectionTrace,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked operation admits all consumed storage and retains none.
    unsafe { prepare_print(params, input, out, failure, trace, error) }
}
/// # Safety
/// Same synchronous live/disjoint header contract as input_demand.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_scanner_correction_input_demand(
    params: *const FjScannerCorrectionParams,
    operation: u32,
    required: *mut u32,
    failure: *mut FjScannerCorrectionFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The shared query admits outputs/Params without reading active inputs.
    unsafe { input_demand(params, operation, required, failure, error) }
}

const _: unsafe extern "C" fn(
    *const FjScannerCorrectionParams,
    *const FjScannerDirectCorrectionInput,
    *mut FjScannerColorCorrection,
    *mut FjScannerCorrectionFailure,
    *mut FjScannerCorrectionTrace,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_scanner_direct_correction_prepare;
const _: unsafe extern "C" fn(
    *const FjScannerCorrectionParams,
    *const FjScannerPrintCorrectionInput,
    *mut FjScannerColorCorrection,
    *mut FjScannerCorrectionFailure,
    *mut FjScannerCorrectionTrace,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_scanner_print_correction_prepare;
const _: unsafe extern "C" fn(
    *const FjScannerCorrectionParams,
    u32,
    *mut u32,
    *mut FjScannerCorrectionFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_scanner_correction_input_demand;

#[cfg(feature = "test-support")]
pub(crate) mod faults {
    use super::*;
    use std::cell::Cell;
    thread_local! { static FAULT: Cell<[u32; 2]> = const { Cell::new([0; 2]) }; }
    pub(crate) fn arm(operation: u32, fault: u32) -> FjStatus {
        if operation > 1 || fault > 1 {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        FAULT.with(|state| {
            let mut faults = state.get();
            faults[operation as usize] = fault;
            state.set(faults);
        });
        status(FJ_STATUS_SUCCESS)
    }
    pub(super) fn take(operation: Operation) -> bool {
        FAULT.with(|state| {
            let mut faults = state.get();
            let index = match operation {
                Operation::Direct => 0,
                Operation::Print => 1,
            };
            let fault = faults[index];
            faults[index] = 0;
            state.set(faults);
            fault == 1
        })
    }
}

#[cfg(feature = "test-support")]
/// # Safety
/// All consumed spans and writable outputs obey the same synchronous disjoint
/// contract. Empty spans ignore backing; optional lookup values may contain NaN.
pub(crate) unsafe fn lookup(
    query: f32,
    axis: FjFloatSpan,
    values: FjFloatSpan,
    channel: Option<u32>,
    out: *mut f32,
    present: *mut u32,
) -> FjStatus {
    let valid_out = valid_extent(out, 1);
    let valid_present = valid_extent(present, 1);
    // SAFETY: Clear each independently admitted exclusive output before admission.
    unsafe {
        if valid_out {
            out.write(0.0);
        }
        if valid_present {
            present.write(0);
        }
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if !valid_out || !valid_present {
            return Err(Failure::Raw);
        }
        // SAFETY: Helpers admit each live span before forming call-local borrows.
        unsafe {
            let axis = span(axis)?;
            Ok(match channel {
                None => scanner::test_support::lookup(query, axis, span(values)?),
                Some(channel) => scanner::test_support::density_curve(
                    query,
                    axis,
                    rows(values)?,
                    channel as usize,
                ),
            })
        }
    }))
    .unwrap_or(Err(Failure::Panic));
    match result {
        Ok(value) => {
            if let Some(value) = value {
                // SAFETY: Both outputs are admitted, exclusive and live.
                unsafe {
                    out.write(value);
                    present.write(1);
                }
            }
            status(FJ_STATUS_SUCCESS)
        }
        Err(error) => status(error.category()),
    }
}

#[cfg(feature = "test-support")]
const _: fn(FjScannerCorrectionParams) = |value| {
    let _: u32 = value.route;
    let _: u32 = value.flags;
    let _: f32 = value.black_level;
    let _: f32 = value.white_level;
};
#[cfg(feature = "test-support")]
const _: fn(FjScannerCorrectionTables) = |value| {
    let _: FjFloatSpan = value.eps_c;
    let _: FjFloatSpan = value.eps_m;
    let _: FjFloatSpan = value.eps_y;
    let _: FjFloatSpan = value.weighted_y;
    let _: FjFloatSpan = value.baseline_min;
    let _: i32 = value.sample_count;
    let _: f32 = value.inv_yn;
    let _: u32 = value.has_baseline;
};
#[cfg(feature = "test-support")]
const _: fn(FjScannerDirectCorrectionInput) = |value| {
    let _: *const FjScannerCorrectionTables = value.tables;
    let _: FjFloatSpan = value.film_base_density;
    let _: FjFloatSpan = value.film_log_exposure;
    let _: FjFloatSpan = value.film_authored_density_curves_cmy;
    let _: [f32; 3] = value.minimum_density_cmy;
    let _: [f32; 3] = value.maximum_density_cmy;
    let _: [f32; 3] = value.inverse_span_cmy;
    let _: [f32; 3] = value.baseline_density_cmy;
    let _: u64 = value.reference_hash;
    let _: u32 = value.medium;
};
#[cfg(feature = "test-support")]
const _: fn(FjScannerPrintCorrectionInput) = |value| {
    let _: *const FjScannerCorrectionTables = value.tables;
    let _: FjFloatSpan = value.film_base_density;
    let _: FjFloatSpan = value.film_channel_density_cmy;
    let _: FjFloatSpan = value.print_sensitivity_cmy;
    let _: FjFloatSpan = value.filtered_main_illuminant;
    let _: FjFloatSpan = value.print_base_density;
    let _: FjFloatSpan = value.print_log_exposure;
    let _: FjFloatSpan = value.print_density_curves_cmy;
    let _: [f32; 3] = value.minimum_density_cmy;
    let _: [f32; 3] = value.inverse_span_cmy;
    let _: [f32; 3] = value.film_minimum_density_cmy;
    let _: [f32; 3] = value.film_maximum_density_cmy;
    let _: [f32; 3] = value.preflash_raw_cmy;
    let _: f32 = value.normalizer;
    let _: f32 = value.preflash_exposure;
    let _: u32 = value.medium;
};
#[cfg(feature = "test-support")]
const _: fn(FjScannerColorCorrection) = |value| {
    let _: u32 = value.route;
    let _: u32 = value.active;
    let _: u32 = value.black_correction;
    let _: u32 = value.white_correction;
    let _: f32 = value.target_black_linear;
    let _: f32 = value.target_white_linear;
    let _: f32 = value.reference_black_y;
    let _: f32 = value.reference_white_y;
    let _: f32 = value.xyz_slope;
    let _: f32 = value.xyz_offset;
    let _: f32 = value.exposure_scale;
    let _: u32 = value.schema_version;
    let _: u64 = value.hash;
};
#[cfg(feature = "test-support")]
const _: fn(FjScannerCorrectionFailure) = |value| {
    let _: u32 = value.reason;
};
#[cfg(feature = "test-support")]
const _: fn(FjScannerCorrectionTrace) = |value| {
    let _: u32 = value.outcome;
    let _: f32 = value.reference_black_y;
    let _: f32 = value.reference_white_y;
    let _: f32 = value.target_black_linear;
    let _: f32 = value.target_white_linear;
    let _: f32 = value.xyz_slope;
    let _: f32 = value.xyz_offset;
    let _: f32 = value.corrected_midgray;
    let _: f32 = value.exposure_scale;
    let _: u64 = value.hash;
};

// SF_TEMP_BRIDGE_SCANNER_LUT: immutable owner/view transport; remove S4.E.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerLutInput {
    pub(crate) eps_c: FjFloatSpan,
    pub(crate) eps_m: FjFloatSpan,
    pub(crate) eps_y: FjFloatSpan,
    pub(crate) weighted_x: FjFloatSpan,
    pub(crate) weighted_y: FjFloatSpan,
    pub(crate) weighted_z: FjFloatSpan,
    pub(crate) baseline_min: FjFloatSpan,
    pub(crate) range_min_cmy: [f32; 3],
    pub(crate) inverse_span_cmy: [f32; 3],
    pub(crate) sample_count: i32,
    pub(crate) inv_yn: f32,
    pub(crate) has_baseline: u32,
    pub(crate) medium: u32,
    pub(crate) resolution: u32,
}
#[repr(C)]
pub(crate) struct FjScannerLutOwner {
    _private: [u8; 0],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerLutView {
    pub(crate) resolution: u32,
    pub(crate) log2_xyz: FjFloatSpan,
    pub(crate) slope_c: FjFloatSpan,
    pub(crate) slope_m: FjFloatSpan,
    pub(crate) slope_y: FjFloatSpan,
    pub(crate) cell_min: FjFloatSpan,
    pub(crate) cell_max: FjFloatSpan,
}
impl FjScannerLutView {
    fn empty() -> Self {
        let empty = FjFloatSpan {
            data: std::ptr::null(),
            count: 0,
        };
        Self {
            resolution: 0,
            log2_xyz: empty,
            slope_c: empty,
            slope_m: empty,
            slope_y: empty,
            cell_min: empty,
            cell_max: empty,
        }
    }
    fn new(lut: &scanner::Lut) -> Self {
        let span = |v: &[f32]| FjFloatSpan {
            data: v.as_ptr(),
            count: v.len(),
        };
        Self {
            resolution: lut.resolution(),
            log2_xyz: span(lut.log2_xyz()),
            slope_c: span(lut.slope_c()),
            slope_m: span(lut.slope_m()),
            slope_y: span(lut.slope_y()),
            cell_min: span(lut.cell_min()),
            cell_max: span(lut.cell_max()),
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerLutFailure {
    pub(crate) reason: u32,
    pub(crate) channel: u32,
    pub(crate) index: u64,
    pub(crate) buffer: u32,
}
impl FjScannerLutFailure {
    fn empty() -> Self {
        Self {
            reason: 0,
            channel: u32::MAX,
            index: u64::MAX,
            buffer: 0,
        }
    }
}
enum LutRejection {
    Core(scanner::LutError),
    Panic,
}
impl LutRejection {
    fn fields(&self) -> (FjStatus, FjScannerLutFailure) {
        use scanner::LutError;
        let mut fields = FjScannerLutFailure::empty();
        let category;
        (fields.reason, category) = match self {
            Self::Core(LutError::Raw) => (1, FJ_STATUS_UNSUPPORTED_INPUT),
            Self::Core(LutError::Resolution) => (2, FJ_STATUS_PREPARATION_FAILURE),
            Self::Core(LutError::Size { buffer }) => {
                fields.buffer = *buffer;
                (3, crate::cuda::sys::FJ_STATUS_ALLOCATION_FAILURE)
            }
            Self::Core(LutError::Allocation { buffer }) => {
                fields.buffer = *buffer;
                (4, crate::cuda::sys::FJ_STATUS_ALLOCATION_FAILURE)
            }
            Self::Core(LutError::Nonfinite { index, channel }) => {
                fields.index = *index;
                fields.channel = *channel;
                (5, FJ_STATUS_PREPARATION_FAILURE)
            }
            Self::Panic => (6, FJ_STATUS_INTERNAL_FAILURE),
        };
        (status(category), fields)
    }
}
impl fmt::Display for LutRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(e) => e.fmt(f),
            Self::Panic => f.write_str("scanner LUT preparation panicked"),
        }
    }
}
fn lut_raw() -> LutRejection {
    LutRejection::Core(scanner::LutError::Raw)
}

/// # Safety
/// Live initialized input and consumed backing are immutable through return.
/// No projected reference escapes the immediate complete construction.
unsafe fn lut_input<'a>(
    pointer: *const FjScannerLutInput,
) -> Result<scanner::LutInput<'a>, LutRejection> {
    if !valid_extent(pointer, 1) {
        return Err(lut_raw());
    }
    // SAFETY: The caller guarantees initialized record storage, admitted above.
    let input = unsafe { &*pointer };
    if input.has_baseline > 1 || input.medium > 255 {
        return Err(lut_raw());
    }
    scanner::Lut::check_capacity(input.resolution).map_err(LutRejection::Core)?;
    let bind = |wire: FjFloatSpan| -> Result<&'a [f32], LutRejection> {
        if input.sample_count <= 0 {
            return Ok(&[]);
        }
        if wire.count != input.sample_count as usize || !valid_extent(wire.data, wire.count) {
            return Err(lut_raw());
        }
        // SAFETY: Exact positive length, alignment and addressability are checked;
        // the caller supplies live immutable backing through synchronous return.
        Ok(unsafe { std::slice::from_raw_parts(wire.data, wire.count) })
    };
    Ok(scanner::LutInput {
        resolution: input.resolution,
        medium: input.medium as u8,
        range_min_cmy: input.range_min_cmy,
        inverse_span_cmy: input.inverse_span_cmy,
        tables: scanner::LutTables {
            eps_c: bind(input.eps_c)?,
            eps_m: bind(input.eps_m)?,
            eps_y: bind(input.eps_y)?,
            weighted_x: bind(input.weighted_x)?,
            weighted_y: bind(input.weighted_y)?,
            weighted_z: bind(input.weighted_z)?,
            baseline_min: if input.has_baseline == 1 {
                Some(bind(input.baseline_min)?)
            } else {
                None
            },
            sample_count: input.sample_count,
            inv_yn: input.inv_yn,
        },
    })
}

/// # Safety
/// Each admitted output and diagnostic record/backing is initialized, writable,
/// exclusive and disjoint from peers and consumed inputs until synchronous return.
unsafe fn lut_call<T: Copy>(
    out: *mut T,
    cleared: T,
    failure: *mut FjScannerLutFailure,
    error: *mut FjErrorBuffer,
    body: impl FnOnce() -> Result<T, LutRejection>,
) -> FjStatus {
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    // SAFETY: Independent numeric admission precedes each exclusive output write.
    unsafe {
        if valid_out {
            out.write(cleared);
        }
        if valid_failure {
            failure.write(FjScannerLutFailure::empty());
        }
    }
    // SAFETY: Diagnostic admission follows independent output clearing.
    let mut diagnostic = unsafe { Diagnostic::admit(error) };
    let result = if !valid_out || !valid_failure || diagnostic.is_err() {
        Err(lut_raw())
    } else {
        catch_unwind(AssertUnwindSafe(body)).unwrap_or(Err(LutRejection::Panic))
    };
    match result {
        Ok(value) => {
            // SAFETY: Successful admission established exclusive live output storage.
            unsafe {
                out.write(value);
            }
            status(FJ_STATUS_SUCCESS)
        }
        Err(rejection) => {
            let (status, fields) = rejection.fields();
            if valid_failure {
                // SAFETY: Failure storage was independently admitted above.
                unsafe {
                    failure.write(fields);
                }
            }
            if let Ok(writer) = &mut diagnostic {
                let _ = write!(writer, "{rejection}");
            }
            status
        }
    }
}
fn lut_fault(operation: u32) -> u32 {
    #[cfg(feature = "test-support")]
    {
        lut_faults::take(operation)
    }
    #[cfg(not(feature = "test-support"))]
    {
        let _ = operation;
        0
    }
}
fn lut_panic(fault: u32) {
    #[cfg(feature = "test-support")]
    if fault == 1 {
        panic!("scanner LUT fixture panic");
    }
    #[cfg(not(feature = "test-support"))]
    let _ = fault;
}
/// # Safety
/// Same live/disjoint synchronous contract as lut_call; input backing expires
/// on return. A successful unique handle transfers to its matching release.
pub(crate) unsafe fn lut_acquire(
    input: *const FjScannerLutInput,
    out: *mut *mut FjScannerLutOwner,
    failure: *mut FjScannerLutFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = lut_fault(0);
    // SAFETY: Common admission clears each output before accessing foreign inputs.
    unsafe {
        lut_call(out, std::ptr::null_mut(), failure, error, || {
            lut_panic(fault);
            let input = lut_input(input)?;
            #[cfg(feature = "test-support")]
            scanner::test_support::reject_lut_reservation(if fault >= 2 { fault - 1 } else { 0 });
            let lut = Box::new(scanner::Lut::new(input).map_err(LutRejection::Core)?);
            #[cfg(feature = "test-support")]
            lut_faults::LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Box::into_raw(lut).cast())
        })
    }
}
/// # Safety
/// The caller holds a legitimate live unique owner, excludes release and owns
/// the disjoint output records. Borrowed view storage expires at release.
pub(crate) unsafe fn lut_view(
    owner: *const FjScannerLutOwner,
    out: *mut FjScannerLutView,
    failure: *mut FjScannerLutFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = lut_fault(1);
    // SAFETY: The caller guarantees owner provenance; numeric checks precede borrow.
    unsafe {
        lut_call(out, FjScannerLutView::empty(), failure, error, || {
            if !valid_extent(owner.cast::<scanner::Lut>(), 1) {
                return Err(lut_raw());
            }
            lut_panic(fault);
            Ok(FjScannerLutView::new(&*owner.cast::<scanner::Lut>()))
        })
    }
}
/// # Safety
/// NULL or one legitimate live matching owner transfers exactly once, excluding
/// all views. Invalid diagnostics still consume an admitted nonnull owner.
pub(crate) unsafe fn lut_release(
    owner: *mut FjScannerLutOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = lut_fault(2);
    if !owner.is_null() && !valid_extent(owner.cast::<scanner::Lut>(), 1) {
        let mut fields = FjScannerLutFailure::empty();
        let mut out = 0u32;
        // SAFETY: Local outputs and the caller's diagnostics follow common admission.
        return unsafe { lut_call(&mut out, 0, &mut fields, error, || Err(lut_raw())) };
    }
    let owned = if owner.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        lut_faults::LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: The caller transfers one matching Box before diagnostic admission.
        Some(unsafe { Box::from_raw(owner.cast::<scanner::Lut>()) })
    };
    let mut fields = FjScannerLutFailure::empty();
    let mut out = 0u32;
    // SAFETY: The closure owns and drops the admitted handle even when diagnostic
    // admission rejects before invocation or an injected unwind occurs.
    unsafe {
        lut_call(&mut out, 0, &mut fields, error, move || {
            lut_panic(fault);
            drop(owned);
            Ok(0)
        })
    }
}

/// # Safety
/// Same acquire contract as lut_acquire.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_scanner_lut_acquire(
    input: *const FjScannerLutInput,
    out: *mut *mut FjScannerLutOwner,
    failure: *mut FjScannerLutFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked owner operation scopes all raw access.
    unsafe { lut_acquire(input, out, failure, error) }
}
/// # Safety
/// Same live-owner/view exclusion contract as lut_view.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_scanner_lut_view(
    owner: *const FjScannerLutOwner,
    out: *mut FjScannerLutView,
    failure: *mut FjScannerLutFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked view operation scopes all raw access.
    unsafe { lut_view(owner, out, failure, error) }
}
/// # Safety
/// Same consuming live-owner contract as lut_release.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_scanner_lut_release(
    owner: *mut FjScannerLutOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked release consumes matching ownership on every outcome.
    unsafe { lut_release(owner, error) }
}
const _: unsafe extern "C" fn(
    *const FjScannerLutInput,
    *mut *mut FjScannerLutOwner,
    *mut FjScannerLutFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_scanner_lut_acquire;
const _: unsafe extern "C" fn(
    *const FjScannerLutOwner,
    *mut FjScannerLutView,
    *mut FjScannerLutFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_scanner_lut_view;
const _: unsafe extern "C" fn(*mut FjScannerLutOwner, *mut FjErrorBuffer) -> FjStatus =
    fj_legacy_scanner_lut_release;

#[cfg(feature = "test-support")]
pub(crate) mod lut_faults {
    use super::*;
    use std::cell::Cell;
    pub(super) static LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    thread_local! { static FAULTS: Cell<[u32;3]> = const { Cell::new([0;3]) }; }
    pub(crate) fn arm(operation: u32, fault: u32) -> FjStatus {
        if operation > 2 || fault > if operation == 0 { 11 } else { 1 } {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        FAULTS.with(|v| {
            let mut slots = v.get();
            slots[operation as usize] = fault;
            v.set(slots);
        });
        status(FJ_STATUS_SUCCESS)
    }
    pub(super) fn take(operation: u32) -> u32 {
        FAULTS.with(|v| {
            let mut slots = v.get();
            let fault = slots[operation as usize];
            slots[operation as usize] = 0;
            v.set(slots);
            fault
        })
    }
    pub(crate) fn live() -> usize {
        LIVE.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// SF_TEMP_BRIDGE_SCANNER_COLOR: fixed completed value; remove with native callers in S4.E.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerColorInput {
    pub(crate) medium: u32,
    pub(crate) output_space: u32,
    pub(crate) apply_cctf: u32,
    pub(crate) illuminant_xyz: [f32; 3],
    pub(crate) illuminant_hash: u64,
    pub(crate) active_gamut_recipe_hash: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerColor {
    pub(crate) cat02: [f32; 9],
    pub(crate) xyz_to_rgb: [f32; 9],
    pub(crate) illuminant_xyz: [f32; 3],
    pub(crate) output_space: u32,
    pub(crate) flags: u32,
    pub(crate) encoding: crate::cuda::sys::FjOutputEncoding,
    pub(crate) active_gamut_recipe_hash: u64,
    pub(crate) hash: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjScannerColorFailure {
    pub(crate) reason: u32,
    pub(crate) stage: u32,
}

enum ColorFailure {
    Raw,
    Panic,
    Core(scanner::ColorError),
}
impl ColorFailure {
    fn envelope(&self) -> (u32, u32, u32) {
        use scanner::ColorError;
        match self {
            Self::Raw => (1, 0, FJ_STATUS_UNSUPPORTED_INPUT),
            Self::Panic => (5, 0, FJ_STATUS_INTERNAL_FAILURE),
            Self::Core(ColorError::Illuminant) => (2, 0, FJ_STATUS_PREPARATION_FAILURE),
            Self::Core(ColorError::Output) => (3, 1, FJ_STATUS_PREPARATION_FAILURE),
            #[cfg(feature = "test-support")]
            Self::Core(ColorError::OutputBoundary) => (4, 1, FJ_STATUS_UNSUPPORTED_INPUT),
            #[cfg(feature = "test-support")]
            Self::Core(ColorError::Cat02Boundary) => (4, 2, FJ_STATUS_UNSUPPORTED_INPUT),
        }
    }
}
impl fmt::Display for ColorFailure {
    fn fmt(&self, writer: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Raw => writer.write_str("invalid scanner color arguments"),
            Self::Panic => writer.write_str("scanner color preparation panicked"),
            Self::Core(error) => error.fmt(writer),
        }
    }
}
fn project_color(value: scanner::Color) -> FjScannerColor {
    let c = value.cctf();
    FjScannerColor {
        cat02: *value.cat02(),
        xyz_to_rgb: *value.xyz_to_rgb(),
        illuminant_xyz: *value.illuminant_xyz(),
        output_space: u32::from(value.output_space()),
        flags: 2 | u32::from(value.apply_cctf()),
        encoding: crate::cuda::sys::FjOutputEncoding {
            schema: crate::cuda::sys::FJ_OUTPUT_ENCODING_SCHEMA,
            cctf_kind: c.kind as u32,
            gamma: c.gamma,
            a: c.a,
            b: c.b,
            c: c.c,
            d: c.d,
            linear_cutoff: c.linear_cutoff,
            dwg_to_output: *value.dwg_to_output(),
        },
        active_gamut_recipe_hash: value.active_gamut_recipe_hash(),
        hash: value.hash(),
    }
}

/// # Safety
/// Each present record/backing is live, aligned, addressable and disjoint;
/// outputs/error are exclusive through synchronous return. No pointer is retained.
pub(crate) unsafe fn color_prepare(
    input: *const FjScannerColorInput,
    out: *mut FjScannerColor,
    failure: *mut FjScannerColorFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    // SAFETY: Independently admitted complete writable extents; clear peers even
    // when another envelope is malformed. Caller guarantees disjoint storage.
    unsafe {
        if valid_out {
            out.write_bytes(0, 1);
        }
        if valid_failure {
            failure.write_bytes(0, 1);
        }
    }
    let mut diagnostic = None;
    let result = catch_unwind(AssertUnwindSafe(|| {
        if !valid_extent(error, 1) {
            return Err(ColorFailure::Raw);
        }
        // SAFETY: Valid error record and exclusive backing by caller contract.
        diagnostic = Some(unsafe { Diagnostic::admit(error) }.map_err(|_| ColorFailure::Raw)?);
        if !valid_out || !valid_failure || !valid_extent(input, 1) {
            return Err(ColorFailure::Raw);
        }
        // SAFETY: Extent/alignment admitted; caller supplies initialized readable input.
        let input = unsafe { &*input };
        if input.medium > 255 || input.output_space > 255 || input.apply_cctf > 1 {
            return Err(ColorFailure::Raw);
        }
        scanner::Color::new(scanner::ColorInput {
            medium: input.medium as u8,
            output_space: input.output_space as u8,
            apply_cctf: input.apply_cctf != 0,
            illuminant_xyz: input.illuminant_xyz,
            illuminant_hash: input.illuminant_hash,
            active_gamut_recipe_hash: input.active_gamut_recipe_hash,
        })
        .map(project_color)
        .map_err(ColorFailure::Core)
    }))
    .unwrap_or(Err(ColorFailure::Panic));
    match result {
        Ok(value) => {
            // SAFETY: Admitted exclusive result extent receives only a complete value.
            unsafe { out.write(value) };
            status(FJ_STATUS_SUCCESS)
        }
        Err(rejection) => {
            let (reason, stage, category) = rejection.envelope();
            if valid_failure {
                // SAFETY: Independently admitted writable failure record.
                unsafe { failure.write(FjScannerColorFailure { reason, stage }) };
            }
            if let Some(writer) = &mut diagnostic {
                let _ = write!(writer, "{rejection}");
            }
            status(category)
        }
    }
}
/// # Safety
/// Same live/readable/exclusive/disjoint contract as `color_prepare`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_scanner_color_prepare(
    input: *const FjScannerColorInput,
    out: *mut FjScannerColor,
    failure: *mut FjScannerColorFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The foreign caller supplies the complete checked synchronous contract.
    unsafe { color_prepare(input, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const FjScannerColorInput,
    *mut FjScannerColor,
    *mut FjScannerColorFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_scanner_color_prepare;
const _: () = {
    use std::mem::offset_of;
    assert!(size_of::<FjScannerColorInput>() == 40 && align_of::<FjScannerColorInput>() == 8);
    assert!(offset_of!(FjScannerColorInput, medium) == 0);
    assert!(offset_of!(FjScannerColorInput, output_space) == 4);
    assert!(offset_of!(FjScannerColorInput, apply_cctf) == 8);
    assert!(offset_of!(FjScannerColorInput, illuminant_xyz) == 12);
    assert!(offset_of!(FjScannerColorInput, illuminant_hash) == 24);
    assert!(offset_of!(FjScannerColorInput, active_gamut_recipe_hash) == 32);
    assert!(size_of::<FjScannerColor>() == 176 && align_of::<FjScannerColor>() == 8);
    assert!(offset_of!(FjScannerColor, cat02) == 0);
    assert!(offset_of!(FjScannerColor, xyz_to_rgb) == 36);
    assert!(offset_of!(FjScannerColor, illuminant_xyz) == 72);
    assert!(offset_of!(FjScannerColor, output_space) == 84);
    assert!(offset_of!(FjScannerColor, flags) == 88);
    assert!(offset_of!(FjScannerColor, encoding) == 92);
    assert!(offset_of!(FjScannerColor, active_gamut_recipe_hash) == 160);
    assert!(offset_of!(FjScannerColor, hash) == 168);
    assert!(size_of::<FjScannerColorFailure>() == 8 && align_of::<FjScannerColorFailure>() == 4);
    assert!(
        offset_of!(FjScannerColorFailure, reason) == 0
            && offset_of!(FjScannerColorFailure, stage) == 4
    );
};
