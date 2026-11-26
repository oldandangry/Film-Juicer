//! Flat native CUDA calls over borrowed frame and callback inputs.

use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::rc::Rc;

use crate::cuda::geometry::FrameGeometry;

use super::sys;
mod ofx;
pub(crate) use ofx::{render_callback, trace_cutoff, trace_flags};

pub(crate) struct PreparedFramePolicy {
    pub(super) geometry: FrameGeometry,
    pub(super) optics: sys::FjOptics,
}

pub(super) struct FrameDescriptors {
    pub(super) spatial_dir: sys::FjSpatialDir,
    pub(super) grain: sys::FjVisualGrain,
    pub(super) effects: sys::FjFilmEffects,
}

pub(crate) struct Frame<'call> {
    raw: &'call sys::FjFrame,
    frame_token: u64,
    thread: PhantomData<Rc<()>>,
}

pub(crate) struct FramePolicy {
    raw: PreparedFramePolicy,
}

pub(crate) struct Abort<'call> {
    raw: sys::FjAbortCallback,
    scope: PhantomData<&'call ()>,
    thread: PhantomData<Rc<()>>,
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

impl super::Owner {
    pub(super) fn inspect(&self, frame: &Frame<'_>) -> Result<sys::FjCudaContext, Failure> {
        callback(|error| {
            // SAFETY: The C output is zero-valid. The live frame and owner
            // are borrowed for this synchronous call.
            let mut context = unsafe { std::mem::zeroed::<sys::FjCudaContext>() };
            // SAFETY: Owner and frame are live and the output and diagnostic are exclusive.
            let result =
                unsafe { sys::fj_cuda_inspect(self.raw(), frame.raw, &mut context, error) };
            (result, context)
        })
    }

    pub(super) fn execute(
        &self,
        context: &sys::FjCudaContext,
        frame: &Frame<'_>,
        submission: &sys::FjSubmission,
        prepared: &sys::FjPreparedHostData,
        abort: &Abort<'_>,
        preparation: &super::host_preparation::Call,
    ) -> (sys::FjRenderOutcome, String) {
        let mut bytes = [0u8; 2048];
        let mut error = sys::FjErrorBuffer {
            data: bytes.as_mut_ptr().cast(),
            capacity: bytes.len(),
            length: 0,
        };
        let host = preparation.raw();
        // SAFETY: Both image leases, the immutable frame data and callback
        // backing remain live. Native stages asynchronous reads before return.
        let result = unsafe {
            sys::fj_cuda_render(
                self.raw(),
                context,
                frame.raw,
                submission,
                prepared,
                &host,
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

    pub(super) fn retire_context(&self, identity: &sys::FjCudaRetirement) -> Result<(), Failure> {
        callback(|error| {
            // SAFETY: The owner is retained, the producing identity is
            // immutable, and this entry acquires its own native gate.
            (
                unsafe { sys::fj_cuda_retire_context(self.raw(), identity, error) },
                (),
            )
        })
    }
}

pub(super) fn metering(frame: &Frame<'_>, method: u32) -> Result<sys::FjAutoExposure, Failure> {
    callback(|error| {
        // SAFETY: This pointer-free C record is zero-valid. The native
        // calculation borrows only the immutable bounds and writes the output.
        let mut preview = unsafe { std::mem::zeroed::<sys::FjAutoExposure>() };
        // SAFETY: Bounds remain live and the output and diagnostic are disjoint.
        let result =
            unsafe { sys::fj_cuda_metering(&frame.raw.source.bounds, method, &mut preview, error) };
        (result, preview)
    })
}

pub(super) fn trace_submission(frame: &Frame<'_>, submission: &sys::FjSubmission, reused: bool) {
    // SAFETY: The synchronous sink retains neither record.
    unsafe { sys::fj_cuda_trace_submission(frame.raw, submission, u32::from(reused)) };
}

pub(super) fn inactive_grain() -> sys::FjVisualGrain {
    // SAFETY: All fields are zero-valid numeric values or arrays thereof.
    unsafe { std::mem::zeroed() }
}

pub(super) fn inactive_effects() -> sys::FjFilmEffects {
    // SAFETY: All fields are zero-valid numeric values or arrays thereof.
    unsafe { std::mem::zeroed() }
}

pub(super) fn failed_outcome(status: sys::FjStatus) -> sys::FjRenderOutcome {
    // SAFETY: This C record contains only integers and a zero-valid identity.
    sys::FjRenderOutcome {
        status,
        ..unsafe { std::mem::zeroed() }
    }
}

impl Frame<'_> {
    pub(super) fn frame_token(&self) -> u64 {
        self.frame_token
    }

    pub(super) fn raw(&self) -> &sys::FjFrame {
        self.raw
    }
}

impl FramePolicy {
    pub(crate) fn prepare(
        state: &film_juicer_core::recipe::RenderState,
        full: film_juicer_core::frame::Domain,
    ) -> Result<Self, String> {
        super::frame_policy::prepare(
            state,
            sys::FjExtent {
                x: full.x,
                y: full.y,
                width: full.width,
                height: full.height,
            },
        )
        .map(|raw| Self { raw })
        .map_err(|error| error.diagnostic)
    }
    pub(crate) fn pixel_size_um(&self) -> f32 {
        self.raw.geometry.pixel_size_um
    }
    pub(super) fn raw(&self) -> &PreparedFramePolicy {
        &self.raw
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
/// The complete initialized input is live and immutable for the call. Image
/// addresses are opaque device integers and are never dereferenced by Rust.
unsafe fn frame<'a>(pointer: *const sys::FjFrame, frame_token: u64) -> Result<Frame<'a>, Failure> {
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
        frame_token,
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
