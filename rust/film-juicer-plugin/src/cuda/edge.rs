//! Foreign admission and synchronous native callback dispatch for held rendering.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use crate::dir_frame_bridge::FjDirFrameMath;
use crate::effects_frame_bridge::FjEffectsFrameMath;
use crate::frame_bridge::FjFrameGeometry;
use crate::grain_frame_bridge::FjGrainFrameMath;
use crate::asset_bridge::FjAssets;
use crate::preparation::FjPreparedState;

use super::sys;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjPreparedFramePolicy {
    pub(super) geometry: FjFrameGeometry,
    pub(super) optics: sys::FjOptics,
    pub(super) recipe_hash: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjGrainCoefficients {
    pub(super) flags: u32,
    pub(super) debug_view: u32,
    pub(super) sublayer_count: usize,
    pub(super) micro_structure: [f32; 2],
    pub(super) clump_temporal_mix: f32,
    pub(super) size_mix_scale: f32,
    pub(super) amplitude: f32,
    pub(super) chroma_mix: f32,
    pub(super) chroma_shared_weight: f32,
    pub(super) chroma_independent_weight: f32,
    pub(super) particle_density_min_cmy: [f32; 3],
    pub(super) uniformity_cmy: [f32; 3],
    pub(super) layer_axis_block_prefix_max: [[f32; 16]; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjFrameMathInput {
    pub(super) dir: FjDirFrameMath,
    pub(super) grain: FjGrainFrameMath,
    pub(super) effects: FjEffectsFrameMath,
    pub(super) grain_coefficients: FjGrainCoefficients,
    pub(super) film_dust: sys::FjDust,
    pub(super) gate_dust: sys::FjDust,
    pub(super) film_scratch: sys::FjScratch,
    pub(super) gate_scratch: sys::FjScratch,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjNativeFrameDescriptors {
    pub(super) spatial_dir: sys::FjSpatialDir,
    pub(super) grain: sys::FjVisualGrain,
    pub(super) effects: sys::FjFilmEffects,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjCallCompletionInput {
    pub(super) outcome: sys::FjRenderOutcome,
    pub(super) diagnostic: sys::FjStringView,
    pub(super) render_outcome_valid: u32,
    pub(super) abort_failed: u32,
    pub(super) recovery_scope_entered: u32,
    pub(super) has_inspected_context: u32,
    pub(super) inspected_context: sys::FjCudaContext,
    pub(super) force_fatal: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjCallCompletionResult {
    pub(super) outcome: sys::FjRenderOutcome,
    pub(super) fatal: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjHostPreparation {
    pub(super) user: *mut c_void,
    pub(super) scanner_lut: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const crate::scanner_bridge::FjScannerLutInput,
            *mut crate::scanner_bridge::FjScannerLutView,
            *mut crate::scanner_bridge::FjScannerLutFailure,
            *mut sys::FjErrorBuffer,
        ) -> sys::FjStatus,
    >,
    pub(super) gaussian: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const crate::optics_bridge::FjGaussianKernelInput,
            *mut crate::optics_bridge::FjGaussianKernelView,
            *mut crate::optics_bridge::FjGaussianKernelFailure,
            *mut sys::FjErrorBuffer,
        ) -> sys::FjStatus,
    >,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjCudaCallAccess {
    user: *mut c_void,
    precheck: Option<unsafe extern "C" fn(*mut c_void, *mut sys::FjErrorBuffer) -> sys::FjStatus>,
    admit: Option<unsafe extern "C" fn(*mut c_void, *mut sys::FjErrorBuffer) -> sys::FjStatus>,
    release: Option<unsafe extern "C" fn(*mut c_void)>,
    inspect: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const sys::FjFrame,
            *mut sys::FjCudaContext,
            *mut sys::FjErrorBuffer,
        ) -> sys::FjStatus,
    >,
    preview: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const sys::FjRect,
            *const sys::FjRect,
            u32,
            *mut sys::FjAutoExposure,
            *mut sys::FjErrorBuffer,
        ) -> sys::FjStatus,
    >,
    frame_descriptors: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const FjFrameMathInput,
            *mut FjNativeFrameDescriptors,
            *mut sys::FjErrorBuffer,
        ) -> sys::FjStatus,
    >,
    render: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const sys::FjCudaContext,
            *const sys::FjFrame,
            *const sys::FjSubmission,
            *const sys::FjPreparedHostData,
            *const FjHostPreparation,
            sys::FjAbortCallback,
            *mut sys::FjErrorBuffer,
        ) -> sys::FjRenderOutcome,
    >,
    complete: Option<
        unsafe extern "C" fn(*mut c_void, *const FjCallCompletionInput) -> FjCallCompletionResult,
    >,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FjSubmissionAccess {
    instance_token: u64,
    frame_token: u64,
    user: *mut c_void,
    select: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const sys::FjCudaContext,
            *const sys::FjSubmission,
            *mut sys::FjSubmission,
            *mut sys::FjErrorBuffer,
        ) -> sys::FjStatus,
    >,
}

pub(crate) struct OwnerBorrow<'call> {
    access: FjCudaCallAccess,
    scope: PhantomData<&'call mut ()>,
    thread: PhantomData<Rc<()>>,
}

pub(crate) struct Frame<'call> {
    raw: &'call sys::FjFrame,
    thread: PhantomData<Rc<()>>,
}

pub(crate) struct FramePolicy {
    raw: FjPreparedFramePolicy,
}

pub(crate) struct SubmissionLatch<'call> {
    access: FjSubmissionAccess,
    scope: PhantomData<&'call mut ()>,
    thread: PhantomData<Rc<()>>,
}

pub(crate) struct Abort<'call> {
    raw: sys::FjAbortCallback,
    scope: PhantomData<&'call ()>,
    thread: PhantomData<Rc<()>>,
}

pub(crate) struct RenderOutcome {
    pub(super) completion: FjCallCompletionResult,
    pub(super) diagnostic: String,
}

pub(crate) struct RenderError {
    pub(super) completion: FjCallCompletionResult,
    pub(super) diagnostic: String,
}

pub(super) struct Failure {
    pub(super) status: sys::FjStatus,
    pub(super) diagnostic: String,
}

pub(super) fn status(category: u32) -> sys::FjStatus {
    sys::FjStatus {
        category,
        api: sys::FJ_API_NONE,
        native_code: 0,
    }
}

impl Failure {
    pub(super) fn new(category: u32, diagnostic: impl Into<String>) -> Self {
        Self {
            status: status(category),
            diagnostic: diagnostic.into(),
        }
    }
}

fn valid<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && size_of::<T>() <= isize::MAX as usize
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}

pub(super) struct Diagnostic {
    record: *mut sys::FjErrorBuffer,
    data: *mut u8,
    capacity: usize,
}

impl Diagnostic {
    /// # Safety
    /// Optional initialized record and nonempty writable backing are exclusive,
    /// live and disjoint from all other storage through the foreign call.
    pub(super) unsafe fn admit(record: *mut sys::FjErrorBuffer) -> Result<Self, ()> {
        let mut value = Self {
            record,
            data: std::ptr::null_mut(),
            capacity: 0,
        };
        if record.is_null() {
            return Ok(value);
        }
        if !valid(record) {
            return Err(());
        }
        // SAFETY: The admitted record is writable. Zero capacity deliberately
        // does not read the data field, which the caller need not initialize.
        unsafe {
            std::ptr::addr_of_mut!((*record).length).write(0);
            value.capacity = std::ptr::addr_of!((*record).capacity).read();
            if value.capacity == 0 {
                return Ok(value);
            }
            let data = std::ptr::addr_of!((*record).data).read();
            if data.is_null()
                || value.capacity > isize::MAX as usize
                || (data as usize).checked_add(value.capacity).is_none()
            {
                return Err(());
            }
            value.data = data.cast();
            value.data.write(0);
        }
        Ok(value)
    }

    pub(super) fn write(&mut self, message: &str) {
        if self.capacity == 0 {
            return;
        }
        let count = message.len().min(self.capacity - 1);
        // SAFETY: admit proved the exclusive backing extent; the writer is
        // call-local and retains no borrow after return.
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), self.data, count);
            self.data.add(count).write(0);
            std::ptr::addr_of_mut!((*self.record).length).write(count);
        }
    }
}

fn callback<T>(
    operation: impl FnOnce(*mut sys::FjErrorBuffer) -> (sys::FjStatus, T),
) -> Result<T, Failure> {
    let mut bytes = [0u8; 512];
    let mut error = sys::FjErrorBuffer {
        data: bytes.as_mut_ptr().cast(),
        capacity: bytes.len(),
        length: 0,
    };
    let (result, output) = operation(&mut error);
    if result.category == sys::FJ_STATUS_SUCCESS {
        return Ok(output);
    }
    let count = error.length.min(bytes.len() - 1);
    Err(Failure {
        status: result,
        diagnostic: String::from_utf8_lossy(&bytes[..count]).into_owned(),
    })
}

impl<'call> OwnerBorrow<'call> {
    fn admit_access(access: FjCudaCallAccess) -> Result<Self, Failure> {
        if access.user.is_null()
            || access.precheck.is_none()
            || access.admit.is_none()
            || access.release.is_none()
            || access.inspect.is_none()
            || access.preview.is_none()
            || access.frame_descriptors.is_none()
            || access.render.is_none()
            || access.complete.is_none()
        {
            return Err(Failure::new(
                sys::FJ_STATUS_UNSUPPORTED_INPUT,
                "invalid native call access",
            ));
        }
        Ok(Self {
            access,
            scope: PhantomData,
            thread: PhantomData,
        })
    }

    pub(super) fn precheck(&self) -> Result<(), Failure> {
        callback(|error| {
            // SAFETY: Foreign admission establishes the synchronous callback and user lifetime.
            (
                unsafe {
                    self.access.precheck.expect("admitted call access")(self.access.user, error)
                },
                (),
            )
        })
    }

    pub(super) fn admit(&mut self) -> Result<(), Failure> {
        callback(|error| {
            // SAFETY: One call guard controls this validated native admission.
            (
                unsafe {
                    self.access.admit.expect("admitted call access")(self.access.user, error)
                },
                (),
            )
        })
    }

    pub(super) fn release(&mut self) {
        // SAFETY: The call guard invokes this once after successful admission.
        unsafe { self.access.release.expect("admitted call access")(self.access.user) };
    }

    pub(super) fn inspect(&mut self, frame: &Frame<'_>) -> Result<sys::FjCudaContext, Failure> {
        callback(|error| {
            let mut context = sys::FjCudaContext {
                device_id: 0,
                context: 0,
            };
            // SAFETY: All pointers are initialized, disjoint callback-local records.
            let result = unsafe {
                self.access.inspect.expect("admitted call access")(
                    self.access.user,
                    frame.raw,
                    &mut context,
                    error,
                )
            };
            (result, context)
        })
    }

    pub(super) fn preview(
        &self,
        frame: &Frame<'_>,
        method: u32,
    ) -> Result<sys::FjAutoExposure, Failure> {
        callback(|error| {
            // SAFETY: This pointer-free C output is zero-valid and cleared before callback work.
            let mut preview = unsafe { std::mem::zeroed::<sys::FjAutoExposure>() };
            // SAFETY: Validated callbacks borrow only these initialized stack records.
            let result = unsafe {
                self.access.preview.expect("admitted call access")(
                    self.access.user,
                    &frame.raw.source.bounds,
                    &frame.raw.source.bounds,
                    method,
                    &mut preview,
                    error,
                )
            };
            (result, preview)
        })
    }

    pub(super) fn descriptors(
        &mut self,
        input: &FjFrameMathInput,
    ) -> Result<FjNativeFrameDescriptors, Failure> {
        callback(|error| {
            // SAFETY: The three fixed C records contain only zero-valid scalar values.
            let mut output = unsafe { std::mem::zeroed::<FjNativeFrameDescriptors>() };
            // SAFETY: Complete checked math and writable output remain live through the callback.
            let result = unsafe {
                self.access.frame_descriptors.expect("admitted call access")(
                    self.access.user,
                    input,
                    &mut output,
                    error,
                )
            };
            (result, output)
        })
    }

    pub(super) fn execute(
        &mut self,
        context: &sys::FjCudaContext,
        frame: &Frame<'_>,
        submission: &sys::FjSubmission,
        prepared: &sys::FjPreparedHostData,
        abort: &Abort<'_>,
        host_preparation: &super::host_preparation::Call,
    ) -> (sys::FjRenderOutcome, String) {
        let mut bytes = [0u8; 512];
        let mut error = sys::FjErrorBuffer {
            data: bytes.as_mut_ptr().cast(),
            capacity: bytes.len(),
            length: 0,
        };
        // SAFETY: Raw projection is immediate, all source owners live through native return,
        // and asynchronous reads must move into native staging before this returns.
        let host_preparation = host_preparation.raw();
        // SAFETY: All source owners and callback backing remain live through this synchronous
        // native call; the native renderer copies any data needed by asynchronous work.
        let result = unsafe {
            self.access.render.expect("admitted call access")(
                self.access.user,
                context,
                frame.raw,
                submission,
                prepared,
                &host_preparation,
                abort.raw,
                &mut error,
            )
        };
        let count = error.length.min(bytes.len() - 1);
        (
            result,
            String::from_utf8_lossy(&bytes[..count]).into_owned(),
        )
    }

    pub(super) fn complete(&mut self, input: &FjCallCompletionInput) -> FjCallCompletionResult {
        // SAFETY: The no-throw continuation is synchronous, after native release,
        // while the call guard still retains its noise result.
        unsafe { self.access.complete.expect("admitted call access")(self.access.user, input) }
    }
}

impl Frame<'_> {
    pub(super) fn raw(&self) -> &sys::FjFrame {
        self.raw
    }
}

impl FramePolicy {
    pub(super) fn raw(&self) -> &FjPreparedFramePolicy {
        &self.raw
    }
}

impl SubmissionLatch<'_> {
    pub(super) fn select(
        &mut self,
        context: &sys::FjCudaContext,
        mut requested: sys::FjSubmission,
    ) -> Result<sys::FjSubmission, Failure> {
        requested.instance_token = self.access.instance_token;
        requested.frame_token = self.access.frame_token;
        callback(|error| {
            // SAFETY: A submission consists only of zero-valid fixed integer lanes.
            let mut out = unsafe { std::mem::zeroed::<sys::FjSubmission>() };
            // SAFETY: The validated callback only borrows these live disjoint records.
            let result = unsafe {
                self.access.select.expect("admitted submission access")(
                    self.access.user,
                    context,
                    &requested,
                    &mut out,
                    error,
                )
            };
            (result, out)
        })
    }
}

impl Abort<'_> {
    pub(super) fn query(&self) -> Result<bool, Failure> {
        let Some(query) = self.raw.query else {
            return Ok(false);
        };
        // SAFETY: The foreign edge admits a synchronous exception-containing query.
        match unsafe { query(self.raw.user) } {
            sys::FJ_ABORT_CONTINUE => Ok(false),
            sys::FJ_ABORT_REQUESTED => Ok(true),
            _ => Err(Failure::new(
                sys::FJ_STATUS_INTERNAL_FAILURE,
                "abort callback returned an invalid result",
            )),
        }
    }
}

fn raw_failure(diagnostic: impl Into<String>) -> Failure {
    Failure::new(sys::FJ_STATUS_UNSUPPORTED_INPUT, diagnostic)
}

/// # Safety
/// The matching opaque state allocation is initialized and live through the
/// borrow; the caller excludes release and mutable access until return.
unsafe fn state<'a>(
    pointer: *const FjPreparedState,
) -> Result<&'a film_juicer_core::recipe::RenderState, Failure> {
    if !valid(pointer) {
        return Err(raw_failure("invalid held prepared state"));
    }
    // SAFETY: Numeric admission precedes the caller-authorized opaque owner borrow.
    Ok(unsafe { &*pointer }.render_state())
}

/// # Safety
/// The matching Assets owner is initialized and live through return.
unsafe fn assets<'a>(
    pointer: *const FjAssets,
) -> Result<&'a film_juicer_core::assets::Assets, Failure> {
    // SAFETY: The foreign entry's owner contract authorizes the matching borrow.
    unsafe { crate::preparation::assets(pointer) }.map_err(|error| raw_failure(error.to_string()))
}

/// # Safety
/// The complete initialized input is live and immutable for the call. Image
/// addresses are opaque device integers and are never dereferenced by Rust.
unsafe fn frame<'a>(pointer: *const sys::FjFrame) -> Result<Frame<'a>, Failure> {
    if !valid(pointer) {
        return Err(raw_failure("invalid held render frame"));
    }
    // SAFETY: Only the admitted caller-owned record becomes a reference.
    let raw = unsafe { &*pointer };
    let components = |value| {
        matches!(
            value,
            sys::FJ_COMPONENTS_UNKNOWN | sys::FJ_COMPONENTS_RGB | sys::FJ_COMPONENTS_RGBA
        )
    };
    let depth = |value| matches!(value, sys::FJ_DEPTH_UNKNOWN | sys::FJ_DEPTH_FLOAT32);
    if !components(raw.source.components)
        || !components(raw.destination.components)
        || !depth(raw.source.depth)
        || !depth(raw.destination.depth)
        || raw.flags
            & !(sys::FJ_FRAME_STREAM_PRESENT
                | sys::FJ_FRAME_TRACE_INFO
                | sys::FJ_FRAME_TRACE_VERBOSE)
            != 0
    {
        return Err(raw_failure("invalid held render frame tag"));
    }
    Ok(Frame {
        raw,
        thread: PhantomData,
    })
}

fn abort<'a>(raw: sys::FjAbortCallback) -> Abort<'a> {
    Abort {
        raw,
        scope: PhantomData,
        thread: PhantomData,
    }
}

fn latch<'a>(access: FjSubmissionAccess) -> Result<SubmissionLatch<'a>, Failure> {
    if access.user.is_null() || access.select.is_none() {
        return Err(raw_failure("invalid submission access"));
    }
    Ok(SubmissionLatch {
        access,
        scope: PhantomData,
        thread: PhantomData,
    })
}

/// # Safety
/// The initialized policy is the early checked output for this same held state
/// and frame extent; its live record is immutable and disjoint through return.
unsafe fn policy(
    pointer: *const FjPreparedFramePolicy,
    state: &film_juicer_core::recipe::RenderState,
    frame: &Frame<'_>,
) -> Result<FramePolicy, Failure> {
    if !valid(pointer) {
        return Err(raw_failure("invalid held frame policy"));
    }
    // SAFETY: The record is fixed and pointer-free, and copied after numeric admission.
    let raw = unsafe { *pointer };
    let bounds = frame.raw.full_frame_extent;
    let extent = raw.geometry.full_frame;
    if raw.recipe_hash != state.hash()
        || extent.x != bounds.x1
        || extent.y != bounds.y1
        || bounds.x2.checked_sub(bounds.x1) != Some(extent.width)
        || bounds.y2.checked_sub(bounds.y1) != Some(extent.height)
        || raw.geometry.pixel_size_um.to_bits() != frame.raw.pixel_size_um.to_bits()
        || raw.optics.flags
            & !(sys::FJ_OPTICS_CAMERA_DIFFUSION
                | sys::FJ_OPTICS_ENLARGER_DIFFUSION
                | sys::FJ_OPTICS_SCATTER_HALATION)
            != 0
    {
        return Err(raw_failure("held frame policy differs from state or frame"));
    }
    Ok(FramePolicy { raw })
}

fn rejected(failure: Failure, diagnostic: &mut Diagnostic) -> FjCallCompletionResult {
    diagnostic.write(&failure.diagnostic);
    FjCallCompletionResult {
        outcome: sys::FjRenderOutcome {
            status: failure.status,
            flags: 0,
        },
        fatal: 1,
    }
}

fn returned(
    result: Result<RenderOutcome, RenderError>,
    diagnostic: &mut Diagnostic,
) -> FjCallCompletionResult {
    match result {
        Ok(result) => {
            diagnostic.write(&result.diagnostic);
            result.completion
        }
        Err(result) => {
            diagnostic.write(&result.diagnostic);
            result.completion
        }
    }
}

/// # Safety
/// The matching prepared-state owner and full-frame record are live immutable
/// borrows. `out`, error record and nonempty error backing are exclusive, live,
/// initialized and disjoint. Output expires normally as a pointer-free value.
#[unsafe(no_mangle)]
pub(super) unsafe extern "C" fn fj_legacy_prepared_state_frame_policy(
    owner: *const FjPreparedState,
    full_frame: *const sys::FjExtent,
    out: *mut FjPreparedFramePolicy,
    error: *mut sys::FjErrorBuffer,
) -> sys::FjStatus {
    // SAFETY: The caller authorizes the optional error writer and live inputs.
    let Ok(mut diagnostic) = (unsafe { Diagnostic::admit(error) }) else {
        return status(sys::FJ_STATUS_UNSUPPORTED_INPUT);
    };
    if !valid(out) {
        diagnostic.write("invalid frame policy output");
        return status(sys::FJ_STATUS_UNSUPPORTED_INPUT);
    }
    // SAFETY: Fixed record is zero-valid and admitted writable before all work.
    unsafe { out.write(std::mem::zeroed()) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Matching owner validity is the foreign caller's obligation.
        let state = unsafe { state(owner) }?;
        if !valid(full_frame) {
            return Err(raw_failure("invalid full frame extent"));
        }
        // SAFETY: Numerically admitted initialized fixed input, copied synchronously.
        let full_frame = unsafe { *full_frame };
        super::frame_policy::prepare(state, full_frame)
            .map_err(|error| Failure::new(error.category, error.diagnostic))
    }))
    .unwrap_or_else(|_| {
        Err(Failure::new(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "frame policy panicked",
        ))
    });
    match result {
        Ok(value) => {
            // SAFETY: Complete value commits once to the admitted exclusive output.
            unsafe { out.write(value) };
            status(sys::FJ_STATUS_SUCCESS)
        }
        Err(failure) => {
            diagnostic.write(&failure.diagnostic);
            failure.status
        }
    }
}

/// # Safety
/// Opaque Assets/state are matching live owners; all fixed inputs are initialized
/// immutable disjoint borrows. Tables and their users remain live for this
/// synchronous call, contain foreign exceptions and retain nothing. Native
/// staging owns asynchronous reads before callback return. The error record and
/// nonempty backing are exclusive and disjoint. No owner is consumed here.
#[unsafe(no_mangle)]
pub(super) unsafe extern "C" fn fj_legacy_prepared_state_render(
    asset_owner: *const FjAssets,
    state_owner: *const FjPreparedState,
    access: FjCudaCallAccess,
    raw_frame: *const sys::FjFrame,
    raw_policy: *const FjPreparedFramePolicy,
    submission: FjSubmissionAccess,
    raw_abort: sys::FjAbortCallback,
    error: *mut sys::FjErrorBuffer,
) -> FjCallCompletionResult {
    // SAFETY: The error buffer contract is part of this entry's caller obligations.
    let Ok(mut diagnostic) = (unsafe { Diagnostic::admit(error) }) else {
        return FjCallCompletionResult {
            outcome: sys::FjRenderOutcome {
                status: status(sys::FJ_STATUS_UNSUPPORTED_INPUT),
                flags: 0,
            },
            fatal: 1,
        };
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Each admission checks numeric records before forming the caller-authorized borrow.
        let assets = unsafe { assets(asset_owner) }?;
        // SAFETY: The owner is live and excluded from release for this call.
        let state = unsafe { state(state_owner) }?;
        // SAFETY: The initialized record is live immutable input.
        let frame = unsafe { frame(raw_frame) }?;
        // SAFETY: The pointer-free policy belongs to this same held state and frame.
        let policy = unsafe { policy(raw_policy, state, &frame) }?;
        let mut owner = OwnerBorrow::admit_access(access)?;
        let mut latch = latch(submission)?;
        let abort = abort(raw_abort);
        Ok(super::render(
            &mut owner, assets, state, &frame, &policy, &mut latch, &abort,
        ))
    }))
    .unwrap_or_else(|_| {
        Err(Failure::new(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "held render edge panicked",
        ))
    });
    match result {
        Ok(result) => returned(result, &mut diagnostic),
        Err(failure) => rejected(failure, &mut diagnostic),
    }
}

/// # Safety
/// Same live table/frame/error obligations as held render. Exactly one of
/// `(null state, MissingState)` or `(real held state, MissingInstance)` is valid.
/// No callback or completion occurs when raw admission rejects the combination.
#[unsafe(no_mangle)]
pub(super) unsafe extern "C" fn fj_legacy_prepared_unavailable(
    asset_owner: *const FjAssets,
    state_owner: *const FjPreparedState,
    kind: u32,
    access: FjCudaCallAccess,
    raw_frame: *const sys::FjFrame,
    raw_abort: sys::FjAbortCallback,
    error: *mut sys::FjErrorBuffer,
) -> FjCallCompletionResult {
    // SAFETY: The optional writer is exclusive for this synchronous call.
    let Ok(mut diagnostic) = (unsafe { Diagnostic::admit(error) }) else {
        return FjCallCompletionResult {
            outcome: sys::FjRenderOutcome {
                status: status(sys::FJ_STATUS_UNSUPPORTED_INPUT),
                flags: 0,
            },
            fatal: 1,
        };
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let kind = match (kind, state_owner.is_null()) {
            (0, true) => super::render::UnavailableKind::MissingState,
            (1, false) => super::render::UnavailableKind::MissingInstance,
            _ => return Err(raw_failure("invalid unavailable request kind or state")),
        };
        // SAFETY: Numeric admission precedes the caller-authorized live owner/record borrows.
        let assets = unsafe { assets(asset_owner) }?;
        let state = if state_owner.is_null() {
            None
        } else {
            // SAFETY: The nonnull opaque state is an actual held owner through return.
            Some(unsafe { state(state_owner) }?)
        };
        // SAFETY: The initialized frame record remains live and immutable.
        let frame = unsafe { frame(raw_frame) }?;
        let mut owner = OwnerBorrow::admit_access(access)?;
        let abort = abort(raw_abort);
        Ok(super::render::reject_unavailable(
            &mut owner, assets, state, &frame, &abort, kind,
        ))
    }))
    .unwrap_or_else(|_| {
        Err(Failure::new(
            sys::FJ_STATUS_INTERNAL_FAILURE,
            "unavailable edge panicked",
        ))
    });
    match result {
        Ok(result) => returned(result, &mut diagnostic),
        Err(failure) => rejected(failure, &mut diagnostic),
    }
}

#[cfg(any(test, feature = "test-support"))]
#[allow(
    unsafe_code,
    reason = "test-only prepared-call ABI facts expose a static layout witness"
)]
mod abi_tests;
