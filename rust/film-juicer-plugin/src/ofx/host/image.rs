//! Callback-local OFX image and native render edge. Pixels remain host-owned.

use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use crate::cuda;
use super::{Callback, InstanceHandles, Status, sys};

pub(crate) struct RenderArgs {
    pub(crate) time: f64,
    pub(crate) window: [i32; 4],
    pub(crate) cuda: bool,
    stream: usize,
    stream_present: bool,
}

pub(crate) struct Image<'a> {
    callback: &'a Callback<'a>,
    handle: sys::OfxPropertySetHandle,
    pub(crate) bounds: [i32; 4],
    pub(crate) definition: [i32; 4],
    pub(crate) scale: [f64; 2],
    pub(crate) pixel_aspect_ratio: f64,
    pub(crate) float_path: bool,
    address: usize,
    row_bytes: i32,
    components: u32,
    depth: u32,
    _local: PhantomData<Rc<()>>,
}

impl Drop for Image<'_> {
    fn drop(&mut self) {
        // SAFETY: Successful acquisition established this unique release
        // obligation before metadata decoding. The callback and original suite
        // remain live on this thread; release never retains the handle or retries.
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            let _ = (self.callback.loaded.suites.release_image)(self.handle);
            #[cfg(feature = "test-support")]
            if super::test_support::render_fault(6) {
                panic!("OFX image cleanup panic");
            }
        }));
        if result.is_err() {
            super::super::fail_admission(self.callback.loaded);
        }
    }
}

impl Callback<'_> {
    fn image_string(
        &self,
        handle: sys::OfxPropertySetHandle,
        name: &CStr,
    ) -> Result<Vec<u8>, Status> {
        let mut value = std::ptr::null_mut();
        // SAFETY: This image lease retains the property handle on this callback
        // thread. The host initializes the local output; no lock spans reentry.
        Status::property(unsafe {
            (self.loaded.suites.get_string)(handle, name.as_ptr(), 0, &mut value)
        })?;
        // OFXS treats an accepted null image-property string as empty. Copy a
        // nonnull result before another host call can replace its backing.
        let bytes = if value.is_null() {
            &[]
        } else {
            // SAFETY: An accepted nonnull host string is terminated and remains
            // readable through the next host call. No borrowed bytes escape.
            unsafe { CStr::from_ptr(value) }.to_bytes()
        };
        super::instance_access::copy_bytes(bytes)
    }
    fn image_int(
        &self,
        handle: sys::OfxPropertySetHandle,
        name: &CStr,
        index: i32,
    ) -> Result<i32, Status> {
        let mut value = 0;
        // SAFETY: The callback owns an image lease and this initialized local output.
        Status::property(unsafe {
            (self.loaded.suites.get_int)(handle, name.as_ptr(), index, &mut value)
        })?;
        Ok(value)
    }
    fn image_double(
        &self,
        handle: sys::OfxPropertySetHandle,
        name: &CStr,
        index: i32,
    ) -> Result<f64, Status> {
        let mut value = 0.0;
        // SAFETY: Acquired properties remain callback-live; output is disjoint.
        Status::property(unsafe {
            (self.loaded.suites.get_double)(handle, name.as_ptr(), index, &mut value)
        })?;
        Ok(value)
    }
    pub(crate) fn render_args(&self) -> Result<RenderArgs, Status> {
        let time = self.action_double(c"OfxPropTime", 0)?;
        let _ = self.action_double(c"OfxImageEffectPropRenderScale", 0)?;
        let _ = self.action_double(c"OfxImageEffectPropRenderScale", 1)?;
        let mut window = [0; 4];
        for (index, value) in window.iter_mut().enumerate() {
            *value = self.image_int(self.input, c"OfxImageEffectPropRenderWindow", index as i32)?;
        }
        let _ = self.optional_int(c"OfxImageEffectPropOpenCLEnabled");
        let cuda = self.optional_int(c"OfxImageEffectPropCudaEnabled") != 0;
        let _ = self.optional_int(c"OfxImageEffectPropMetalEnabled");
        let _ = self.optional_pointer(c"OfxImageEffectPropOpenCLCommandQueue");
        let mut dimension = 0;
        // SAFETY: Current action input and local output; this optional raw status
        // intentionally uses exact OK, independently of the pointer getter.
        let stream_present = unsafe {
            (self.loaded.suites.dimension)(
                self.input,
                c"OfxImageEffectPropCudaStream".as_ptr(),
                &mut dimension,
            )
        } == sys::kOfxStatOK
            && dimension > 0;
        let stream = self.optional_pointer(c"OfxImageEffectPropCudaStream") as usize;
        let _ = self.optional_pointer(c"OfxImageEffectPropMetalCommandQueue");
        for name in [
            c"OfxImageEffectPropSequentialRenderStatus",
            c"OfxImageEffectPropInteractiveRenderStatus",
            c"OfxImageEffectPropRenderQualityDraft",
        ] {
            let _ = self.optional_int(name);
        }
        let _ = self.action_field()?;
        Ok(RenderArgs {
            time,
            window,
            cuda,
            stream,
            stream_present,
        })
    }
    pub(crate) fn fetch_image<'a>(
        &'a self,
        handles: &'a InstanceHandles,
        index: usize,
        time: f64,
    ) -> Result<Option<Image<'a>>, Status> {
        let mut handle = std::ptr::null_mut();
        // SAFETY: Stored clip is immutable through Destroy. Original action time
        // and unspecified region are passed unchanged; the host supplies properties.
        let status = unsafe {
            (self.loaded.suites.clip_image)(
                handles.clips[index],
                time,
                std::ptr::null(),
                &mut handle,
            )
        };
        if status == sys::kOfxStatFailed {
            return Ok(None);
        }
        Status::acquisition(status)?;
        if handle.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        let mut image = Image {
            callback: self,
            handle,
            bounds: [0; 4],
            definition: [0; 4],
            scale: [0.0; 2],
            pixel_aspect_ratio: 0.0,
            float_path: false,
            address: 0,
            row_bytes: 0,
            components: 0,
            depth: 0,
            _local: PhantomData,
        };
        image.row_bytes = self.image_int(handle, c"OfxImagePropRowBytes", 0)?;
        image.pixel_aspect_ratio = self.image_double(handle, c"OfxImagePropPixelAspectRatio", 0)?;
        let components = self.image_string(handle, c"OfxImageEffectPropComponents")?;
        image.components = match components.as_slice() {
            b"OfxImageComponentRGB" => 3,
            b"OfxImageComponentRGBA" => 4,
            _ => 0,
        };
        let depth = self.image_string(handle, c"OfxImageEffectPropPixelDepth")?;
        image.depth = u32::from(depth == b"OfxBitDepthFloat");
        image.float_path =
            image.depth == 1 && (image.components != 0 || components == b"OfxImageComponentAlpha");
        let multiplication = self.image_string(handle, c"OfxImageEffectPropPreMultiplication")?;
        if !matches!(
            multiplication.as_slice(),
            b"OfxImageOpaque" | b"OfxImageAlphaPremultiplied" | b"OfxImageAlphaUnPremultiplied"
        ) {
            return Err(Status::FAILED);
        }
        for (index, value) in image.definition.iter_mut().enumerate() {
            *value = self.image_int(handle, c"OfxImagePropRegionOfDefinition", index as i32)?;
        }
        for (index, value) in image.bounds.iter_mut().enumerate() {
            *value = self.image_int(handle, c"OfxImagePropBounds", index as i32)?;
        }
        let _ = self.image_string(handle, c"OfxImagePropField")?;
        let _ = self.image_string(handle, c"OfxImagePropUniqueIdentifier")?;
        image.scale = [
            self.image_double(handle, c"OfxImageEffectPropRenderScale", 0)?,
            self.image_double(handle, c"OfxImageEffectPropRenderScale", 1)?,
        ];
        let mut address = std::ptr::null_mut();
        // SAFETY: The host's device address is copied as an opaque integer only.
        Status::property(unsafe {
            (self.loaded.suites.get_pointer)(handle, c"OfxImagePropData".as_ptr(), 0, &mut address)
        })?;
        image.address = address as usize;
        Ok(Some(image))
    }
    pub(crate) fn frame_rate(&self, handles: &InstanceHandles) -> Result<f64, Status> {
        self.image_double(handles.properties, c"OfxImageEffectPropFrameRate", 0)
    }
    pub(crate) fn canonical_region(
        &self,
        handles: &InstanceHandles,
        time: f64,
    ) -> Result<[f64; 4], Status> {
        let mut region = sys::OfxRectD {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
        };
        // SAFETY: Source clip is retained; host writes this local canonical rectangle.
        Status::acquisition(unsafe {
            (self.loaded.suites.clip_region)(handles.clips[0], time, &mut region)
        })?;
        Ok([region.x1, region.y1, region.x2, region.y2])
    }
    pub(crate) fn message(&self, id: &CStr, text: &str) {
        // This is best-effort completion/admission delivery, after native unlock.
        // Fallible terminated storage avoids letting a diagnostic replace failure.
        if let Ok(bytes) = super::instance_access::terminated(text.as_bytes()) {
            // SAFETY: All strings live through this synchronous variadic call.
            // A fixed %s treats arbitrary diagnostic percent signs as text.
            let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
                (self.loaded.suites.message)(
                    self.effect.cast(),
                    c"OfxMessageError".as_ptr(),
                    id.as_ptr(),
                    c"%s".as_ptr(),
                    bytes.as_ptr().cast::<std::ffi::c_char>(),
                )
            }));
        }
    }
    pub(crate) fn clip_token(&self, handles: &InstanceHandles) -> u64 {
        handles.clips[0] as usize as u64
    }
}

fn rect(bounds: [i32; 4]) -> cuda::sys::FjRect {
    cuda::sys::FjRect {
        x1: bounds[0],
        y1: bounds[1],
        x2: bounds[2],
        y2: bounds[3],
    }
}

impl Image<'_> {
    fn raw(&self) -> cuda::sys::FjImage {
        cuda::sys::FjImage {
            address: self.address,
            bounds: rect(self.bounds),
            row_bytes: self.row_bytes as isize,
            components: match self.components {
                3 => cuda::sys::FJ_COMPONENTS_RGB,
                4 => cuda::sys::FJ_COMPONENTS_RGBA,
                _ => cuda::sys::FJ_COMPONENTS_UNKNOWN,
            },
            depth: if self.depth == 1 {
                cuda::sys::FJ_DEPTH_FLOAT32
            } else {
                cuda::sys::FJ_DEPTH_UNKNOWN
            },
        }
    }
}

struct AbortAccess<'a> {
    callback: &'a Callback<'a>,
    failed: std::cell::Cell<bool>,
}
/// # Safety
/// Native queries serially on the originating callback thread. `user` points to
/// the live stack AbortAccess and retained Callback/host suites. No exclusive
/// Rust borrow or publication lock spans host reentry; native never retains the
/// user pointer or calls it after the synchronous render returns.
unsafe extern "C" fn query_abort(user: *mut c_void) -> u32 {
    // SAFETY: The scoped native call retains this stack access, serially on this
    // callback thread. Only shared host access and a Cell are used; no locks or
    // mutable instance borrow spans reentry, and native never retains user.
    let access = unsafe { &*user.cast::<AbortAccess<'_>>() };
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(feature = "test-support")]
        if super::test_support::render_fault(5) {
            panic!("OFX abort panic");
        }
        // SAFETY: The effect and original suite are retained by the callback.
        if unsafe { (access.callback.loaded.suites.abort)(access.callback.effect) } != 0 {
            cuda::sys::FJ_ABORT_REQUESTED
        } else {
            cuda::sys::FJ_ABORT_CONTINUE
        }
    }))
    .unwrap_or_else(|_| {
        access.failed.set(true);
        super::super::fail_admission(access.callback.loaded);
        2
    })
}

impl Callback<'_> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the immediate edge borrows the existing callback owners and resolved frame facts"
    )]
    pub(crate) fn render_native(
        &self,
        owner: &super::super::instance::Owner,
        state: &film_juicer_core::recipe::RenderState,
        source: &Image<'_>,
        destination: &Image<'_>,
        args: &RenderArgs,
        window: [i32; 4],
        policy: &cuda::FramePolicy,
        time: f64,
        frame_token: u64,
        frame_rate: f64,
        effects: cuda::sys::FjEffectsGeometry,
        trace_flags: u32,
    ) -> Result<(), Status> {
        let identity = owner.state.identity().ok_or(Status::FATAL)?;
        let raw = cuda::sys::FjFrame {
            source: source.raw(),
            destination: destination.raw(),
            render_window: rect(window),
            full_frame_extent: rect(source.bounds),
            stream: args.stream,
            flags: trace_flags
                | if args.stream_present {
                    cuda::sys::FJ_FRAME_STREAM_PRESENT
                } else {
                    0
                },
            pixel_size_um: policy.pixel_size_um(),
            time_frames: time,
            frame_rate,
            session_seed: identity.session_seed,
            clip_token: self.clip_token(&owner.handles),
            effects_geometry: effects,
        };
        #[cfg(feature = "test-support")]
        let raw = if let Some((seed, _, clip)) = super::test_support::render_tokens() {
            cuda::sys::FjFrame {
                session_seed: seed,
                clip_token: clip,
                ..raw
            }
        } else {
            raw
        };
        #[cfg(feature = "test-support")]
        super::test_support::render_frame(&raw, frame_token, &owner.state);
        let access = AbortAccess {
            callback: self,
            failed: std::cell::Cell::new(false),
        };
        let abort = cuda::sys::FjAbortCallback {
            query: Some(query_abort),
            user: std::ptr::from_ref(&access).cast_mut().cast(),
        };
        // SAFETY: Both leases, admitted immutable state, Runtime/native owner,
        // frame and abort backing live on this thread through synchronous return.
        // No raw device address is dereferenced in Rust; native validates device
        // facts, copies asynchronous host inputs and never retains callback tables.
        let fatal = unsafe {
            cuda::edge::render_callback(
                owner.loaded.runtime(),
                state,
                &raw,
                policy,
                &owner.state,
                frame_token,
                abort,
                |completion| {
                    if completion.deferred_dir_error() {
                        self.message(c"FilmJuicerDeferredCudaFailure", completion.diagnostic());
                    }
                    let fatal = completion.fatal() || access.failed.get();
                    if fatal {
                        completion.trace_failure();
                    }
                    #[cfg(feature = "test-support")]
                    super::test_support::render_complete(completion.status(), fatal, &owner.state);
                    completion.recover(&owner.state);
                    fatal
                },
            )
        };
        if fatal { Err(Status::FATAL) } else { Ok(()) }
    }
}
