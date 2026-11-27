//! Typed instance handles and foreign pointer operations, private to the OFX edge.

use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use super::{Callback, Properties, Status, sys};
use super::super::instance::Owner;

use super::super::ofx_str;
use super::super::param_id::{Kind, ParamId};

struct Parameter {
    handle: sys::OfxParamHandle,
    properties: sys::OfxPropertySetHandle,
}

pub(crate) struct InstanceHandles {
    pub(in crate::ofx::host) properties: sys::OfxPropertySetHandle,
    parameters: Vec<Parameter>,
    // Retained source/output handles remain host-owned until terminal Destroy.
    pub(in crate::ofx::host) clips: [sys::OfxImageClipHandle; 2],
    pub(in crate::ofx::host) clip_properties: [sys::OfxPropertySetHandle; 2],
    _local: PhantomData<Rc<()>>,
}

pub(crate) struct ParameterAccess<'a> {
    callback: &'a Callback<'a>,
    handles: &'a InstanceHandles,
}

pub(crate) fn copy_bytes(bytes: &[u8]) -> Result<Vec<u8>, Status> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(bytes.len())
        .map_err(|_| Status::MEMORY)?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}

pub(super) fn terminated(bytes: &[u8]) -> Result<Vec<u8>, Status> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(end.checked_add(1).ok_or(Status::MEMORY)?)
        .map_err(|_| Status::MEMORY)?;
    owned.extend_from_slice(&bytes[..end]);
    owned.push(0);
    Ok(owned)
}

impl Callback<'_> {
    fn instance_pointer(&self) -> Result<*mut Owner, Status> {
        let properties = self.effect_properties()?;
        let mut pointer = std::ptr::null_mut();
        // SAFETY: Host instance data contains either null or this plug-in's
        // unique live Owner from successful publication, never arbitrary bytes.
        Status::property(unsafe {
            (self.loaded.suites.get_pointer)(
                properties,
                ofx_str(sys::kOfxPropInstanceData).as_ptr(),
                0,
                &mut pointer,
            )
        })?;
        if pointer.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        Ok(pointer.cast())
    }

    pub(crate) fn instance(&self) -> Result<&Owner, Status> {
        let pointer = self.instance_pointer()?;
        // SAFETY: OFX excludes overlapping Create/Destroy and forbids callbacks
        // after terminal Destroy. This callback retains the original module,
        // and Owner remains boxed through all reentrant shared callbacks. No
        // exclusive owner reference or transferable capability is constructed.
        Ok(unsafe { &*pointer })
    }

    fn instance_data(&self, handles: &InstanceHandles, pointer: *mut c_void) -> Result<(), Status> {
        // SAFETY: The owner/handles remain live across this potentially reentrant
        // publication or clear. Callers retain or consume according to its result.
        Status::property(unsafe {
            (self.loaded.suites.set_pointer)(
                handles.properties,
                ofx_str(sys::kOfxPropInstanceData).as_ptr(),
                0,
                pointer,
            )
        })
    }

    pub(super) fn create(&self) -> Result<(), Status> {
        let handles = self.instance_handles()?;
        let value = Owner::new(self, handles)?;
        let layout = std::alloc::Layout::new::<Owner>();
        // SAFETY: Allocate the exact nonzero Owner layout once; null remains a
        // recoverable allocation failure before publication. Initialization and
        // Box recovery use that same allocation/layout and unique ownership.
        let pointer = unsafe { std::alloc::alloc(layout) }.cast::<Owner>();
        if pointer.is_null() {
            return Err(Status::MEMORY);
        }
        // SAFETY: Fresh aligned allocation has space for one Owner and no reader.
        unsafe {
            pointer.write(value);
        }
        // SAFETY: Exactly one initialized Owner in the global allocator allocation.
        let owner = unsafe { Box::from_raw(pointer) };
        let bound = owner.bind_address(pointer as usize);
        bound?;
        let publication_attempted = std::cell::Cell::new(false);
        let result = catch_unwind(AssertUnwindSafe(|| {
            owner.host_call(|| {
                publication_attempted.set(true);
                self.instance_data(&owner.handles, pointer.cast())?;
                owner.initialize(self)?;
                self.final_instance_check()
            })
        }));
        match result {
            Ok(Ok(())) => {
                let _ = Box::into_raw(owner);
                Ok(())
            }
            result => {
                owner.fail();
                let original = match result {
                    Ok(Err(status)) => status,
                    _ => {
                        super::super::fail_admission(self.loaded);
                        Status::FATAL
                    }
                };
                if !publication_attempted.get() {
                    drop(owner);
                    return Err(original);
                }
                let detached = catch_unwind(AssertUnwindSafe(|| {
                    owner.host_call(|| self.instance_data(&owner.handles, std::ptr::null_mut()))
                }));
                if matches!(detached, Ok(Ok(()))) {
                    drop(owner);
                    Err(original)
                } else {
                    let _ = Box::into_raw(owner);
                    Err(Status::FATAL)
                }
            }
        }
    }

    pub(super) fn destroy(&self) -> Result<(), Status> {
        let pointer = self.instance_pointer()?;
        // SAFETY: Terminal OFX Destroy excludes all callbacks, and this pointer
        // denotes the unique box published by Create (possibly failed setup).
        // Consume now so every subsequent clear/retire failure still reclaims once.
        let owner = unsafe { Box::from_raw(pointer) };
        let original = Callback::new(&owner.loaded, self.effect, self.input);
        owner.fail();
        let clear = catch_unwind(AssertUnwindSafe(|| {
            owner.host_call(|| original.instance_data(&owner.handles, std::ptr::null_mut()))
        }));
        let retire = catch_unwind(AssertUnwindSafe(|| owner.retire()));
        if clear.is_err() || retire.is_err() {
            super::super::fail_admission(&owner.loaded);
        }
        drop(owner);
        if matches!(clear, Ok(Ok(()))) && matches!(retire, Ok(Ok(()))) {
            Ok(())
        } else {
            Err(Status::FATAL)
        }
    }

    pub(super) fn change_action(&self, changed: bool) -> Result<(), Status> {
        let owner = self.instance()?;
        owner.require_live()?;
        let callback = Callback::new(&owner.loaded, self.effect, self.input);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let reason = callback.action_string(ofx_str(sys::kOfxPropChangeReason))?;
            let (plugin, user) = match reason.as_slice() {
                b"OfxChangePluginEdited" => (true, false),
                b"OfxChangeUserEdited" => (false, true),
                b"OfxChangeTime" => (false, false),
                _ => return Err(Status::FAILED),
            };
            if changed {
                let _time = callback.action_double(ofx_str(sys::kOfxPropTime), 0)?;
                let _scale_x =
                    callback.action_double(ofx_str(sys::kOfxImageEffectPropRenderScale), 0)?;
                let _scale_y =
                    callback.action_double(ofx_str(sys::kOfxImageEffectPropRenderScale), 1)?;
                let kind = callback.action_string(ofx_str(sys::kOfxPropType))?;
                let name = callback.action_string(ofx_str(sys::kOfxPropName))?;
                if kind == b"OfxTypeParameter" {
                    owner.changed(&callback, &name, plugin, user)?;
                }
            }
            Ok(())
        }));
        match result {
            Ok(result) => result,
            Err(_) => {
                owner.fail();
                super::super::fail_admission(&owner.loaded);
                Err(Status::FATAL)
            }
        }
    }
    fn effect_properties(&self) -> Result<sys::OfxPropertySetHandle, Status> {
        let mut properties = std::ptr::null_mut();
        // SAFETY: The action's live effect authorizes the suite call and local output.
        Status::acquisition(unsafe {
            (self.loaded.suites.effect_properties)(self.effect, &mut properties)
        })?;
        if properties.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        Ok(properties)
    }

    pub(super) fn property_bytes(
        &self,
        properties: sys::OfxPropertySetHandle,
        name: &CStr,
    ) -> Result<Vec<u8>, Status> {
        let mut value = std::ptr::null_mut();
        // SAFETY: The admitted property handle is live; copy borrowed host bytes before any host call.
        Status::property(unsafe {
            (self.loaded.suites.get_string)(properties, name.as_ptr(), 0, &mut value)
        })?;
        if value.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        // SAFETY: Accepted nonnull string output is terminated through the next host call.
        copy_bytes(unsafe { CStr::from_ptr(value) }.to_bytes())
    }

    fn context(&self, properties: sys::OfxPropertySetHandle) -> Result<(), Status> {
        match self
            .property_bytes(properties, ofx_str(sys::kOfxImageEffectPropContext))?
            .as_slice()
        {
            b"OfxImageEffectContextFilter"
            | b"OfxImageEffectContextGeneral"
            | b"OfxImageEffectContextGenerator"
            | b"OfxImageEffectContextTransition"
            | b"OfxImageEffectContextPaint"
            | b"OfxImageEffectContextRetimer" => Ok(()),
            _ => Err(Status::FAILED),
        }
    }

    pub(crate) fn instance_handles(&self) -> Result<InstanceHandles, Status> {
        let first = self.effect_properties()?;
        self.context(first)?;
        let mut existing = std::ptr::null_mut();
        // SAFETY: Duplicate Create checks initialized host instance storage without dereferencing it.
        Status::property(unsafe {
            (self.loaded.suites.get_pointer)(
                first,
                ofx_str(sys::kOfxPropInstanceData).as_ptr(),
                0,
                &mut existing,
            )
        })?;
        if !existing.is_null() {
            return Err(Status::FATAL);
        }
        let properties = self.effect_properties()?;
        self.context(properties)?;
        let mut parameters = std::ptr::null_mut();
        // SAFETY: Same live effect and independent local output.
        Status::acquisition(unsafe {
            (self.loaded.suites.parameters)(self.effect, &mut parameters)
        })?;
        if parameters.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        let mut parameter_properties = std::ptr::null_mut();
        // SAFETY: The acquired parameter set remains live through Destroy.
        Status::acquisition(unsafe {
            (self.loaded.suites.parameter_properties)(parameters, &mut parameter_properties)
        })?;
        if parameter_properties.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        let mut clips = [std::ptr::null_mut(); 2];
        let mut clip_properties = [std::ptr::null_mut(); 2];
        for ((clip, properties), name) in clips
            .iter_mut()
            .zip(&mut clip_properties)
            .zip([c"Source", c"Output"])
        {
            // SAFETY: Exact vendored outputs, live effect, immutable clip name.
            Status::acquisition(unsafe {
                (self.loaded.suites.clip_handle)(self.effect, name.as_ptr(), clip, properties)
            })?;
            if clip.is_null() || properties.is_null() {
                return Err(Status::BAD_HANDLE);
            }
        }
        let mut acquired = Vec::new();
        acquired
            .try_reserve_exact(ParamId::ALL.len())
            .map_err(|_| Status::MEMORY)?;
        for &id in ParamId::ALL {
            let name = id.name();
            let mut handle = std::ptr::null_mut();
            let mut type_properties = std::ptr::null_mut();
            // SAFETY: Parameter set/name live; outputs are disjoint local slots.
            Status::acquisition(unsafe {
                (self.loaded.suites.parameter_handle)(
                    parameters,
                    name.as_ptr(),
                    &mut handle,
                    &mut type_properties,
                )
            })?;
            if handle.is_null() || type_properties.is_null() {
                return Err(Status::BAD_HANDLE);
            }
            if self.property_bytes(type_properties, ofx_str(sys::kOfxParamPropType))?
                != id.kind().ofx().to_bytes()
            {
                return Err(Status::FAILED);
            }
            let mut properties = std::ptr::null_mut();
            // SAFETY: Acquired typed parameter remains host-owned through Destroy.
            Status::acquisition(unsafe {
                (self.loaded.suites.parameter_property)(handle, &mut properties)
            })?;
            if properties.is_null() {
                return Err(Status::BAD_HANDLE);
            }
            acquired.push(Parameter { handle, properties });
        }
        Ok(InstanceHandles {
            properties,
            parameters: acquired,
            clips,
            clip_properties,
            _local: PhantomData,
        })
    }

    pub(crate) fn parameter_access<'a>(
        &'a self,
        handles: &'a InstanceHandles,
    ) -> ParameterAccess<'a> {
        ParameterAccess {
            callback: self,
            handles,
        }
    }

    pub(crate) fn final_instance_check(&self) -> Result<(), Status> {
        self.effect_properties().map(|_| ())
    }

    pub(crate) fn action_string(&self, name: &CStr) -> Result<Vec<u8>, Status> {
        self.property_bytes(self.input, name)
    }

    pub(crate) fn action_double(&self, name: &CStr, index: i32) -> Result<f64, Status> {
        let mut value = 0.0;
        // SAFETY: Changed action admits its input properties; output is a local f64.
        Status::property(unsafe {
            (self.loaded.suites.get_double)(self.input, name.as_ptr(), index, &mut value)
        })?;
        Ok(value)
    }
}

impl ParameterAccess<'_> {
    pub(crate) fn label(&self, id: ParamId) -> Result<Vec<u8>, Status> {
        self.presentation(id, ofx_str(sys::kOfxPropLabel), false)
    }
    pub(crate) fn hint(&self, id: ParamId) -> Result<Vec<u8>, Status> {
        self.presentation(id, ofx_str(sys::kOfxParamPropHint), true)
    }
    fn presentation(
        &self,
        id: ParamId,
        property: &CStr,
        optional: bool,
    ) -> Result<Vec<u8>, Status> {
        let mut value = std::ptr::null_mut();
        let parameter = self.parameter(id);
        // SAFETY: Parameter properties are retained host handles. Optional hint
        // preserves OFXS's returned bytes even when its nonthrowing status fails.
        let status = unsafe {
            (self.callback.loaded.suites.get_string)(
                parameter.properties,
                property.as_ptr(),
                0,
                &mut value,
            )
        };
        if !optional {
            Status::property(status)?;
        }
        if value.is_null() {
            return Ok(Vec::new());
        }
        // SAFETY: Any supplied string output remains terminated until the next host call.
        copy_bytes(unsafe { CStr::from_ptr(value) }.to_bytes())
    }
    fn parameter(&self, id: ParamId) -> &Parameter {
        &self.handles.parameters[id as usize]
    }
    fn typed(&self, id: ParamId, kind: Kind) -> Result<&Parameter, Status> {
        let parameter = self.parameter(id);
        if id.kind() != kind {
            return Err(Status::FAILED);
        }
        Ok(parameter)
    }
    pub(crate) fn integer(&self, id: ParamId) -> Result<i32, Status> {
        let parameter = self.parameter(id);
        if !matches!(id.kind(), Kind::Boolean | Kind::Choice | Kind::Integer) {
            return Err(Status::FAILED);
        }
        let mut value = 0i32;
        // SAFETY: Int/Choice/Boolean current getters receive exactly int*.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_get)(parameter.handle, &mut value as *mut i32)
        })?;
        Ok(if id.kind() == Kind::Boolean {
            i32::from(value != 0)
        } else {
            value
        })
    }
    pub(crate) fn double(&self, id: ParamId) -> Result<f64, Status> {
        let parameter = self.typed(id, Kind::Double)?;
        let mut value = 0.0f64;
        // SAFETY: A scalar Double current getter receives exactly double*.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_get)(parameter.handle, &mut value as *mut f64)
        })?;
        Ok(value)
    }
    pub(crate) fn double2(&self, id: ParamId) -> Result<[f64; 2], Status> {
        let parameter = self.typed(id, Kind::Double2D)?;
        let (mut x, mut y) = (0.0f64, 0.0f64);
        // SAFETY: Double2D getter receives two disjoint double* outputs.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_get)(
                parameter.handle,
                &mut x as *mut f64,
                &mut y as *mut f64,
            )
        })?;
        Ok([x, y])
    }
    pub(crate) fn double3(&self, id: ParamId) -> Result<[f64; 3], Status> {
        let parameter = self.typed(id, Kind::Double3D)?;
        let (mut x, mut y, mut z) = (0.0f64, 0.0f64, 0.0f64);
        // SAFETY: Double3D getter receives three disjoint double* outputs.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_get)(
                parameter.handle,
                &mut x as *mut f64,
                &mut y as *mut f64,
                &mut z as *mut f64,
            )
        })?;
        Ok([x, y, z])
    }
    pub(crate) fn string_bytes(&self, id: ParamId) -> Result<Vec<u8>, Status> {
        self.read_string(id, None)
    }
    pub(crate) fn profile_key(&self, id: ParamId, default: &[u8]) -> Result<Vec<u8>, Status> {
        self.read_string(id, Some(default))
    }
    fn read_string(&self, id: ParamId, default: Option<&[u8]>) -> Result<Vec<u8>, Status> {
        let parameter = self.typed(id, Kind::StringChoice)?;
        let mut value = std::ptr::null_mut::<std::ffi::c_char>();
        // SAFETY: StrChoice current getter receives char**; the callback retains
        // the immutable typed handle. No lock/exclusive owner borrow spans this
        // reentrant call. Parameter statuses use acquisition, not property mapping.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_get)(
                parameter.handle,
                &mut value as *mut *mut std::ffi::c_char,
            )
        })?;
        if value.is_null() {
            return Err(Status::BAD_HANDLE);
        }
        // SAFETY: Host bytes are terminated and valid until the next host call.
        // Select the established empty-key default and copy it at this same
        // fallible allocation boundary, before the following parameter getter.
        let bytes = unsafe { CStr::from_ptr(value) }.to_bytes();
        let bytes = if bytes.is_empty() {
            default.unwrap_or(bytes)
        } else {
            bytes
        };
        copy_bytes(bytes)
    }
    pub(crate) fn set_integer(&self, id: ParamId, value: i32) -> Result<(), Status> {
        let parameter = self.parameter(id);
        if !matches!(id.kind(), Kind::Boolean | Kind::Choice | Kind::Integer) {
            return Err(Status::FAILED);
        }
        let value = if id.kind() == Kind::Boolean {
            i32::from(value != 0)
        } else {
            value
        };
        // SAFETY: Exact promoted int setter argument for the admitted kind.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(parameter.handle, value)
        })
    }
    pub(crate) fn set_double(&self, id: ParamId, value: f64) -> Result<(), Status> {
        let parameter = self.typed(id, Kind::Double)?;
        // SAFETY: Exact promoted double setter argument.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(parameter.handle, value)
        })
    }
    pub(crate) fn set_double2(&self, id: ParamId, value: [f64; 2]) -> Result<(), Status> {
        let parameter = self.typed(id, Kind::Double2D)?;
        // SAFETY: Exact ordered promoted double components.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(parameter.handle, value[0], value[1])
        })
    }
    pub(crate) fn set_double3(&self, id: ParamId, value: [f64; 3]) -> Result<(), Status> {
        let parameter = self.typed(id, Kind::Double3D)?;
        // SAFETY: Exact ordered promoted double components.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(
                parameter.handle,
                value[0],
                value[1],
                value[2],
            )
        })
    }
    pub(crate) fn set_string_bytes(&self, id: ParamId, value: &[u8]) -> Result<(), Status> {
        let parameter = self.typed(id, Kind::StringChoice)?;
        let value = terminated(value)?;
        // SAFETY: Terminated immutable bytes remain live through the synchronous setter.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(
                parameter.handle,
                value.as_ptr().cast::<std::ffi::c_char>(),
            )
        })
    }
    fn properties(&self, id: ParamId) -> Properties<'_> {
        Properties {
            callback: self.callback,
            handle: self.parameter(id).properties,
        }
    }
    pub(crate) fn set_enabled(&self, id: ParamId, enabled: bool) -> Result<(), Status> {
        self.properties(id)
            .set_int(ofx_str(sys::kOfxParamPropEnabled), 0, i32::from(enabled))
    }
    pub(crate) fn set_secret(&self, id: ParamId, secret: bool) -> Result<(), Status> {
        self.properties(id)
            .set_int(ofx_str(sys::kOfxParamPropSecret), 0, i32::from(secret))
    }
    pub(crate) fn set_label(&self, id: ParamId, bytes: &[u8]) -> Result<(), Status> {
        self.set_text(id, ofx_str(sys::kOfxPropLabel), bytes)
    }
    pub(crate) fn set_hint(&self, id: ParamId, bytes: &[u8]) -> Result<(), Status> {
        let bytes = terminated(bytes)?;
        let text = CStr::from_bytes_with_nul(&bytes).map_err(|_| Status::FAILED)?;
        // OFXS ignores the optional host status, after constructing the string.
        // Local fallible storage remains an allocation failure for its action.
        let _ = self
            .properties(id)
            .set_string(ofx_str(sys::kOfxParamPropHint), 0, text);
        Ok(())
    }
    fn set_text(&self, id: ParamId, property: &CStr, bytes: &[u8]) -> Result<(), Status> {
        let bytes = terminated(bytes)?;
        let text = CStr::from_bytes_with_nul(&bytes).map_err(|_| Status::FAILED)?;
        self.properties(id).set_string(property, 0, text)
    }
}
