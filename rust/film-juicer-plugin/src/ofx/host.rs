//! Raw host storage and callback-local capabilities. No foreign call holds MODULE.

use std::ffi::{CStr, c_char, c_void};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::Arc;

use crate::runtime::Runtime;

use super::sys;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Status(i32);

impl Status {
    pub(super) const FAILED: Self = Self(sys::kOfxStatFailed);
    pub(super) const FATAL: Self = Self(sys::kOfxStatErrFatal);
    pub(super) const BAD_HANDLE: Self = Self(sys::kOfxStatErrBadHandle);
    pub(super) const BAD_INDEX: Self = Self(sys::kOfxStatErrBadIndex);
    pub(super) const MEMORY: Self = Self(sys::kOfxStatErrMemory);
    const MISSING_FEATURE: Self = Self(sys::kOfxStatErrMissingHostFeature);

    pub(crate) fn code(self) -> i32 {
        self.0
    }

    fn acquisition(code: i32) -> Result<(), Self> {
        match code {
            sys::kOfxStatOK
            | sys::kOfxStatReplyYes
            | sys::kOfxStatReplyNo
            | sys::kOfxStatReplyDefault => Ok(()),
            _ => Err(Self(code)),
        }
    }

    fn property(code: i32) -> Result<(), Self> {
        Self::acquisition(code).map_err(|error| match error.0 {
            sys::kOfxStatErrUnknown | sys::kOfxStatErrUnsupported => Self::MISSING_FEATURE,
            sys::kOfxStatErrValue => Self::FAILED,
            _ => error,
        })
    }

    pub(super) fn runtime(error: crate::cuda::Failure) -> Self {
        if error.status.category == crate::cuda::sys::FJ_STATUS_ALLOCATION_FAILURE {
            Self(sys::kOfxStatErrMemory)
        } else {
            Self::FAILED
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Binding {
    handle: sys::OfxPropertySetHandle,
    fetch: unsafe extern "C" fn(sys::OfxPropertySetHandle, *const c_char, i32) -> *const c_void,
}

// SAFETY: OfxHost explicitly keeps its host handle and fetched APIs valid until
// binary unload. This copied binding lives only in the serialized module state;
// the lifecycle reserves Load before calling fetchSuite on the host action thread.
// It transfers no callback/image/context capability or permission for worker calls.
unsafe impl Send for Binding {}

#[derive(Clone, Copy)]
pub(super) struct Suites {
    effect_properties:
        unsafe extern "C" fn(sys::OfxImageEffectHandle, *mut sys::OfxPropertySetHandle) -> i32,
    parameters: unsafe extern "C" fn(sys::OfxImageEffectHandle, *mut sys::OfxParamSetHandle) -> i32,
    parameter_properties:
        unsafe extern "C" fn(sys::OfxParamSetHandle, *mut sys::OfxPropertySetHandle) -> i32,
    set_string:
        unsafe extern "C" fn(sys::OfxPropertySetHandle, *const c_char, i32, *const c_char) -> i32,
    set_int: unsafe extern "C" fn(sys::OfxPropertySetHandle, *const c_char, i32, i32) -> i32,
    dimension: unsafe extern "C" fn(sys::OfxPropertySetHandle, *const c_char, *mut i32) -> i32,
    get_string: unsafe extern "C" fn(
        sys::OfxPropertySetHandle,
        *const c_char,
        i32,
        *mut *mut c_char,
    ) -> i32,
    set_double: unsafe extern "C" fn(sys::OfxPropertySetHandle, *const c_char, i32, f64) -> i32,
    define_clip: unsafe extern "C" fn(
        sys::OfxImageEffectHandle,
        *const c_char,
        *mut sys::OfxPropertySetHandle,
    ) -> i32,
    define_parameter: unsafe extern "C" fn(
        sys::OfxParamSetHandle,
        *const c_char,
        *const c_char,
        *mut sys::OfxPropertySetHandle,
    ) -> i32,
}

impl Binding {
    pub(super) fn suites(self) -> Result<Suites, Status> {
        // SAFETY: setHost admitted a binary-lived host handle/fetch function.
        // Load is reserved, but no lock is held across these reentrant calls.
        let (property, effect, parameter, message) = unsafe {
            (
                (self.fetch)(self.handle, c"OfxPropertySuite".as_ptr(), 1),
                (self.fetch)(self.handle, c"OfxImageEffectSuite".as_ptr(), 1),
                (self.fetch)(self.handle, c"OfxParameterSuite".as_ptr(), 1),
                (self.fetch)(self.handle, c"OfxMessageSuite".as_ptr(), 1),
            )
        };
        if property.is_null() || effect.is_null() || parameter.is_null() || message.is_null() {
            return Err(Status::MISSING_FEATURE);
        }
        // SAFETY: Each successful V1 fetch names the corresponding initialized
        // vendored suite. Copy tables locally; no host-storage reference escapes.
        let (property, effect, parameter) = unsafe {
            (
                property.cast::<sys::OfxPropertySuiteV1>().read(),
                effect.cast::<sys::OfxImageEffectSuiteV1>().read(),
                parameter.cast::<sys::OfxParameterSuiteV1>().read(),
            )
        };
        Ok(Suites {
            effect_properties: effect.getPropertySet.ok_or(Status::MISSING_FEATURE)?,
            parameters: effect.getParamSet.ok_or(Status::MISSING_FEATURE)?,
            parameter_properties: parameter
                .paramSetGetPropertySet
                .ok_or(Status::MISSING_FEATURE)?,
            set_string: property.propSetString.ok_or(Status::MISSING_FEATURE)?,
            set_int: property.propSetInt.ok_or(Status::MISSING_FEATURE)?,
            dimension: property.propGetDimension.ok_or(Status::MISSING_FEATURE)?,
            get_string: property.propGetString.ok_or(Status::MISSING_FEATURE)?,
            set_double: property.propSetDouble.ok_or(Status::MISSING_FEATURE)?,
            define_clip: effect.clipDefine.ok_or(Status::MISSING_FEATURE)?,
            define_parameter: parameter.paramDefine.ok_or(Status::MISSING_FEATURE)?,
        })
    }
}

pub(crate) struct Admission {
    suites: Suites,
    // Excludes consuming close across catalog access and reentrant host calls.
    runtime: Arc<Runtime>,
}

impl Admission {
    pub(super) fn new(suites: Suites, runtime: Arc<Runtime>) -> Self {
        Self { suites, runtime }
    }

    pub(super) fn owns(&self, runtime: &Arc<Runtime>) -> bool {
        Arc::ptr_eq(&self.runtime, runtime)
    }
}

pub(crate) struct Callback<'a> {
    loaded: &'a Admission,
    effect: sys::OfxImageEffectHandle,
    input: sys::OfxPropertySetHandle,
    _local: PhantomData<Rc<()>>,
}

impl<'a> Callback<'a> {
    fn new(
        loaded: &'a Admission,
        effect: sys::OfxImageEffectHandle,
        input: sys::OfxPropertySetHandle,
    ) -> Self {
        Self {
            loaded,
            effect,
            input,
            _local: PhantomData,
        }
    }

    fn descriptor_handles(
        &self,
    ) -> Result<(sys::OfxPropertySetHandle, sys::OfxParamSetHandle), Status> {
        let suites = self.loaded.suites;
        let mut first = std::ptr::null_mut();
        // SAFETY: mainEntry authorizes this live Describe effect handle for the
        // whole callback. Output is local; accepted status is checked before null.
        Status::acquisition(unsafe { (suites.effect_properties)(self.effect, &mut first) })?;
        if first.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        let mut parameters = std::ptr::null_mut();
        // SAFETY: Same callback-live effect and disjoint local output.
        Status::acquisition(unsafe { (suites.parameters)(self.effect, &mut parameters) })?;
        if !parameters.is_null() {
            let mut unused = std::ptr::null_mut();
            // SAFETY: Successful acquisition supplies the callback-live param set.
            Status::acquisition(unsafe { (suites.parameter_properties)(parameters, &mut unused) })?;
        }
        Ok((first, parameters))
    }

    fn validate_descriptor(&self) -> Result<(), Status> {
        let mut unused = std::ptr::null_mut();
        // SAFETY: OFXS validates via a second acquisition before authored writes.
        // This unused output may be null or differ; the first handle owns writes.
        Status::acquisition(unsafe {
            (self.loaded.suites.effect_properties)(self.effect, &mut unused)
        })
    }

    pub(crate) fn descriptor_properties(&self) -> Result<Properties<'_>, Status> {
        let (first, _) = self.descriptor_handles()?;
        self.validate_descriptor()?;
        Ok(Properties {
            callback: self,
            handle: first,
        })
    }

    pub(crate) fn context_parameters(&self) -> Result<Option<Parameters<'_>>, Status> {
        let (_, parameters) = self.descriptor_handles()?;
        let mut context = std::ptr::null_mut();
        // SAFETY: DescribeInContext admits a live input property handle. Resolve
        // its borrowed bytes before making another call into the host.
        Status::property(unsafe {
            (self.loaded.suites.get_string)(
                self.input,
                c"OfxImageEffectPropContext".as_ptr(),
                0,
                &mut context,
            )
        })?;
        if context.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        // SAFETY: Accepted nonnull propGetString supplies a terminated string
        // until the next host call. No reference to its storage escapes here.
        let filter = match unsafe { CStr::from_ptr(context) }.to_bytes() {
            b"OfxImageEffectContextFilter" => true,
            b"OfxImageEffectContextGenerator"
            | b"OfxImageEffectContextTransition"
            | b"OfxImageEffectContextPaint"
            | b"OfxImageEffectContextGeneral"
            | b"OfxImageEffectContextRetimer" => false,
            _ => return Err(Status::FAILED),
        };
        self.validate_descriptor()?;
        if !filter {
            return Ok(None);
        }
        Ok(Some(Parameters::new(self, parameters)))
    }

    pub(crate) fn define_clip(&self, name: &CStr) -> Result<Properties<'_>, Status> {
        let mut handle = std::ptr::null_mut();
        // SAFETY: The effect is callback-live, output local and name immutable.
        // OFXS deliberately ignores clipDefine's status; the output is required.
        let _ =
            unsafe { (self.loaded.suites.define_clip)(self.effect, name.as_ptr(), &mut handle) };
        if handle.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        Ok(Properties {
            callback: self,
            handle,
        })
    }

    pub(crate) fn catalog(
        &self,
    ) -> Result<Option<Arc<film_juicer_core::profile::Catalog>>, Status> {
        #[cfg(feature = "test-support")]
        if test_support::context_fault(2) {
            return Err(Status::MEMORY);
        }
        let catalog = self.loaded.runtime.assets().catalog();
        #[cfg(feature = "test-support")]
        if test_support::context_fault(1) {
            return Ok(None);
        }
        match catalog {
            Ok(catalog) => {
                if i32::try_from(catalog.films().len()).is_err()
                    || i32::try_from(catalog.prints().len()).is_err()
                {
                    return Err(Status::FAILED);
                }
                Ok(Some(catalog))
            }
            Err(film_juicer_core::assets::AssetError::Catalog(_)) => Ok(None),
            Err(_) => Err(Status::FAILED),
        }
    }
}

pub(crate) struct Parameters<'a> {
    callback: &'a Callback<'a>,
    handle: sys::OfxParamSetHandle,
}

impl<'a> Parameters<'a> {
    fn new(callback: &'a Callback<'a>, handle: sys::OfxParamSetHandle) -> Self {
        Self { callback, handle }
    }

    pub(crate) fn define(&self, kind: &CStr, name: &CStr) -> Result<Properties<'_>, Status> {
        if self.handle.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        #[cfg(feature = "test-support")]
        if test_support::context_fault(4) {
            panic!("OFX context declaration panic");
        }
        let mut handle = std::ptr::null_mut();
        // SAFETY: The admitted nonnull parameter set and immediate C strings
        // remain live. Check acquisition status before requiring its output.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.define_parameter)(
                self.handle,
                kind.as_ptr(),
                name.as_ptr(),
                &mut handle,
            )
        })?;
        if handle.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        Ok(Properties {
            callback: self.callback,
            handle,
        })
    }
}

pub(crate) struct Properties<'a> {
    callback: &'a Callback<'a>,
    handle: sys::OfxPropertySetHandle,
}

impl Properties<'_> {
    pub(crate) fn set_catalog_string(
        &self,
        name: &CStr,
        index: i32,
        text: &str,
    ) -> Result<(), Status> {
        #[cfg(feature = "test-support")]
        if test_support::context_fault(3) {
            return Err(Status::MEMORY);
        }
        // Preserve std::string::c_str() prefix semantics without changing the
        // catalog's full keys or retaining owned foreign-string storage.
        let bytes = text.as_bytes();
        let end = bytes
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(bytes.len());
        let capacity = end.checked_add(1).ok_or(Status::MEMORY)?;
        let mut terminated = Vec::new();
        terminated
            .try_reserve_exact(capacity)
            .map_err(|_| Status::MEMORY)?;
        terminated.extend_from_slice(&bytes[..end]);
        terminated.push(0);
        let value = CStr::from_bytes_with_nul(&terminated).map_err(|_| Status::FAILED)?;
        self.set_string(name, index, value)
    }

    pub(crate) fn set_double(&self, name: &CStr, index: i32, value: f64) -> Result<(), Status> {
        // SAFETY: Same callback-bound property handle and immediate string borrow.
        Status::property(unsafe {
            (self.callback.loaded.suites.set_double)(self.handle, name.as_ptr(), index, value)
        })
    }
    pub(crate) fn set_string(&self, name: &CStr, index: i32, value: &CStr) -> Result<(), Status> {
        // SAFETY: Required property handle and copied suite remain callback-live;
        // C strings are immutable through this immediate foreign call.
        Status::property(unsafe {
            (self.callback.loaded.suites.set_string)(
                self.handle,
                name.as_ptr(),
                index,
                value.as_ptr(),
            )
        })
    }

    pub(crate) fn set_int(&self, name: &CStr, index: i32, value: i32) -> Result<(), Status> {
        // SAFETY: Same callback-bound property handle and immediate string borrow.
        Status::property(unsafe {
            (self.callback.loaded.suites.set_int)(self.handle, name.as_ptr(), index, value)
        })
    }

    pub(crate) fn dimension(&self, name: &CStr) -> Result<i32, Status> {
        let mut dimension = 0;
        // SAFETY: Output is local initialized storage; no host pointer is retained.
        Status::property(unsafe {
            (self.callback.loaded.suites.dimension)(self.handle, name.as_ptr(), &mut dimension)
        })?;
        Ok(dimension)
    }
}

fn contained(operation: impl FnOnce() -> Result<(), Status>) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(operation));
    #[cfg(feature = "test-support")]
    test_support::pause(1);
    match result {
        Ok(result) => result.map_or_else(Status::code, |()| sys::kOfxStatOK),
        Err(_) => sys::kOfxStatFailed,
    }
}

/// # Safety
/// A nonnull record is initialized and immutable through return. Its host handle
/// and fetched suite APIs remain valid until binary unload, as OfxHost requires.
unsafe extern "C" fn set_host(host: *mut sys::OfxHost) {
    let _ = contained(|| {
        super::bind(|| {
            #[cfg(feature = "test-support")]
            test_support::checkpoint(2);
            if host.is_null() {
                None
            } else {
                // SAFETY: OFX caller supplies a live initialized host struct.
                // Copy only its binary-lived handle/fetch, with no foreign call.
                let host = unsafe { host.read() };
                host.fetchSuite
                    .filter(|_| !host.host.is_null())
                    .map(|fetch| Binding {
                        handle: host.host,
                        fetch,
                    })
            }
        });
        Ok(())
    });
}

/// # Safety
/// A nonnull action is terminated and live through return. Descriptor effects,
/// DescribeInContext input properties and acquired handles are callback-live. The caller
/// excludes callbacks after terminal unload; no borrowed handle may outlive it.
unsafe extern "C" fn main_entry(
    action: *const c_char,
    handle: *const c_void,
    input: sys::OfxPropertySetHandle,
    _output: sys::OfxPropertySetHandle,
) -> i32 {
    contained(|| {
        if action.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        // SAFETY: A nonnull OFX action is a caller-owned terminated string live
        // through return. No numeric pointer test can establish that contract.
        let action = unsafe { CStr::from_ptr(action) };
        match action.to_bytes() {
            b"OfxActionLoad" => super::load(),
            b"OfxActionUnload" => super::unload(),
            b"OfxActionDescribe" | b"OfxImageEffectActionDescribeInContext" => {
                let in_context = action.to_bytes() == b"OfxImageEffectActionDescribeInContext";
                if handle.is_null() || (in_context && input.is_null()) {
                    return Err(Status::BAD_HANDLE);
                }
                let admission = super::admit()?;
                let callback = Callback::new(&admission, handle.cast_mut().cast(), input);
                match catch_unwind(AssertUnwindSafe(|| {
                    #[cfg(feature = "test-support")]
                    test_support::checkpoint(1);
                    if in_context {
                        super::actions::describe_in_context(&callback)
                    } else {
                        super::actions::describe(&callback)
                    }
                })) {
                    Ok(result) => result,
                    Err(_) => {
                        #[cfg(feature = "test-support")]
                        test_support::pause(2);
                        // Keep Admission live until matching-owner invalidation
                        // completes; terminal close cannot pass through this gap.
                        super::fail_admission(&admission);
                        #[cfg(feature = "test-support")]
                        if in_context {
                            test_support::pause(16);
                        }
                        Err(Status::FAILED)
                    }
                }
            }
            b"OfxActionCreateInstance"
            | b"OfxActionDestroyInstance"
            | b"OfxActionBeginInstanceChanged"
            | b"OfxActionInstanceChanged"
            | b"OfxActionEndInstanceChanged"
            | b"OfxImageEffectActionRender" => Err(Status(sys::kOfxStatErrUnsupported)),
            _ => Err(Status(sys::kOfxStatReplyDefault)),
        }
    })
}

struct Descriptor(sys::OfxPlugin);

// SAFETY: This descriptor and its C string storage are immutable for the loaded
// binary lifetime. OFX receives its historical mutable pointer type but may not
// mutate the descriptor. This proof grants no traits to mutable host capabilities.
unsafe impl Sync for Descriptor {}

static DESCRIPTOR: Descriptor = Descriptor(sys::OfxPlugin {
    pluginApi: c"OfxImageEffectPluginAPI".as_ptr(),
    apiVersion: 1,
    pluginIdentifier: c"com.juicer.Juicer".as_ptr(),
    pluginVersionMajor: 1,
    pluginVersionMinor: 0,
    setHost: Some(set_host),
    mainEntry: Some(main_entry),
});

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_ofx_plugin_count() -> i32 {
    // Constant discovery has no allocation, lock, foreign call or panic path.
    1
}

#[cfg_attr(not(test), unsafe(no_mangle))]
extern "C" fn fj_ofx_plugin(index: i32) -> *mut sys::OfxPlugin {
    if index == 0 {
        std::ptr::from_ref(&DESCRIPTOR.0).cast_mut()
    } else {
        std::ptr::null_mut()
    }
}

const _: extern "C" fn() -> i32 = fj_ofx_plugin_count;
const _: extern "C" fn(i32) -> *mut sys::OfxPlugin = fj_ofx_plugin;

#[cfg(feature = "test-support")]
pub(super) mod test_support;
