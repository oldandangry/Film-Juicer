//! Typed instance handles and foreign pointer operations, private to the OFX edge.

use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use super::{Callback, Properties, Status, sys};
use super::super::instance::Owner;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Double,
    Boolean,
    Choice,
    StringChoice,
    Integer,
    Double2D,
    Double3D,
    PushButton,
}

impl Kind {
    fn ofx(self) -> &'static CStr {
        match self {
            Self::Double => c"OfxParamTypeDouble",
            Self::Boolean => c"OfxParamTypeBoolean",
            Self::Choice => c"OfxParamTypeChoice",
            Self::StringChoice => c"OfxParamTypeStrChoice",
            Self::Integer => c"OfxParamTypeInteger",
            Self::Double2D => c"OfxParamTypeDouble2D",
            Self::Double3D => c"OfxParamTypeDouble3D",
            Self::PushButton => c"OfxParamTypePushButton",
        }
    }
}

// Constructor acquisitions, distinct from descriptor-only groups.
const REQUIRED: &[(&CStr, Kind)] = &[
    (c"Exposure", Kind::Double),
    (c"CameraAutoExposure", Kind::Boolean),
    (c"CameraFilmFormatPreset", Kind::Choice),
    (c"CameraFilmFormatMm", Kind::Double),
    (c"CameraMeteringMethod", Kind::Choice),
    (c"FilmProfileKey", Kind::StringChoice),
    (c"SpectralUpsampling", Kind::Choice),
    (c"PrintProfileKey", Kind::StringChoice),
    (c"ReferenceIlluminant", Kind::Choice),
    (c"EnlargerIlluminant", Kind::Choice),
    (c"InputColorSpace", Kind::Choice),
    (c"InputCctfDecoding", Kind::Boolean),
    (c"InputCompression", Kind::Boolean),
    (c"Hanatos2025AdaptationWindow", Kind::Boolean),
    (c"Hanatos2025AdaptationSurface", Kind::Boolean),
    (c"ScanRoute", Kind::StringChoice),
    (c"FilmGammaFactor", Kind::Double),
    (c"PrintGammaFactor", Kind::Double),
    (c"OutputColorSpace", Kind::Choice),
    (c"OutputCctfEncoding", Kind::Boolean),
    (c"OutputGamutCompression", Kind::Boolean),
    (c"CouplersActive", Kind::Boolean),
    (c"CouplersAmount", Kind::Double),
    (c"CouplersInhibitionSameLayer", Kind::Double),
    (c"CouplersInhibitionInterlayer", Kind::Double),
    (c"CouplersDiffusionSizeUm", Kind::Double),
    (c"CouplersLangmuirDonorKRgb", Kind::Double3D),
    (c"CouplersLangmuirReceiverKRgb", Kind::Double3D),
    (c"CouplersDiffusionTailUm", Kind::Double),
    (c"CouplersDiffusionTailWeight", Kind::Double),
    (c"CouplersGammaUseStock", Kind::Boolean),
    (c"CouplersGammaSameLayerRgb", Kind::Double3D),
    (c"CouplersGammaInterlayerRToGb", Kind::Double2D),
    (c"CouplersGammaInterlayerGToRb", Kind::Double2D),
    (c"CouplersGammaInterlayerBToRg", Kind::Double2D),
    (c"ScannerLensBlurSigmaPx", Kind::Double),
    (c"ScannerUnsharpMask", Kind::Double2D),
    (c"ScannerBlackCorrection", Kind::Boolean),
    (c"ScannerWhiteCorrection", Kind::Boolean),
    (c"ScannerBlackLevel", Kind::Double),
    (c"ScannerWhiteLevel", Kind::Double),
    (c"ScannerLutResolution", Kind::Integer),
    (c"PrintExposure", Kind::Double),
    (c"PrintPreflash", Kind::Double),
    (c"PrintExposureCompensation", Kind::Boolean),
    (c"EnlargerY", Kind::Double),
    (c"EnlargerM", Kind::Double),
    (c"EnlargerC", Kind::Double),
    (c"HalationActive", Kind::Boolean),
    (c"HalationScatterAmount", Kind::Double),
    (c"HalationScatterSpatialScale", Kind::Double),
    (c"HalationAmount", Kind::Double),
    (c"HalationSpatialScale", Kind::Double),
    (c"GrainActive", Kind::Boolean),
    (c"GrainSublayersActive", Kind::Boolean),
    (c"GrainPreset", Kind::Choice),
    (c"GrainParticleAreaUm2", Kind::Double),
    (c"GrainAmplitude", Kind::Double),
    (c"GrainSharpness", Kind::Double),
    (c"GrainChroma", Kind::Double),
    (c"GrainTexture", Kind::Double),
    (c"GrainParticleScaleMaster", Kind::Double),
    (c"GrainParticleScaleLayersMaster", Kind::Double),
    (c"GrainDensityMinMaster", Kind::Double),
    (c"GrainUniformityMaster", Kind::Double),
    (c"GrainParticleScale", Kind::Double3D),
    (c"GrainParticleScaleLayers", Kind::Double3D),
    (c"GrainDensityMin", Kind::Double3D),
    (c"GrainUniformity", Kind::Double3D),
    (c"GrainBlur", Kind::Double),
    (c"GrainBlurDyeCloudsUm", Kind::Double),
    (c"GrainSizeMixWeight", Kind::Double),
    (c"GrainSizeMixWeightMid", Kind::Double),
    (c"GrainSizeMixScale", Kind::Double),
    (c"GrainClumpTemporalMix", Kind::Double),
    (c"GrainClumpMorphPeriodSec", Kind::Double),
    (c"GrainDebugView", Kind::Choice),
    (c"GrainMicroStructure", Kind::Double2D),
    (c"GrainResetAdvanced", Kind::PushButton),
    (c"GateWeaveAmount", Kind::Double),
    (c"FilmDustAmount", Kind::Double),
    (c"GateDustAmount", Kind::Double),
    (c"FilmScratchAmount", Kind::Double),
    (c"GateScratchAmount", Kind::Double),
    (c"CameraDiffusionEnabled", Kind::Boolean),
    (c"CameraDiffusionFamily", Kind::Choice),
    (c"CameraDiffusionStrength", Kind::Double),
    (c"CameraDiffusionSpatialScale", Kind::Double),
    (c"CameraDiffusionHaloWarmth", Kind::Double),
    (c"CameraDiffusionCoreIntensity", Kind::Double),
    (c"CameraDiffusionCoreSize", Kind::Double),
    (c"CameraDiffusionHaloIntensity", Kind::Double),
    (c"CameraDiffusionHaloSize", Kind::Double),
    (c"CameraDiffusionBloomIntensity", Kind::Double),
    (c"CameraDiffusionBloomSize", Kind::Double),
    (c"PrintDiffusionEnabled", Kind::Boolean),
    (c"PrintDiffusionFamily", Kind::Choice),
    (c"PrintDiffusionStrength", Kind::Double),
    (c"PrintDiffusionSpatialScale", Kind::Double),
    (c"PrintDiffusionHaloWarmth", Kind::Double),
    (c"PrintDiffusionCoreIntensity", Kind::Double),
    (c"PrintDiffusionCoreSize", Kind::Double),
    (c"PrintDiffusionHaloIntensity", Kind::Double),
    (c"PrintDiffusionHaloSize", Kind::Double),
    (c"PrintDiffusionBloomIntensity", Kind::Double),
    (c"PrintDiffusionBloomSize", Kind::Double),
    (c"GlareActive", Kind::Boolean),
    (c"GlarePercent", Kind::Double),
    (c"GlareRoughness", Kind::Double),
    (c"GlareBlurSigmaPx", Kind::Double),
    (c"PrintShadowCompensationFactor", Kind::Double),
    (c"PrintShadowCompensationDensity", Kind::Double),
    (c"PrintShadowCompensationTransition", Kind::Double),
];

struct Parameter {
    name: &'static CStr,
    kind: Kind,
    handle: sys::OfxParamHandle,
    properties: sys::OfxPropertySetHandle,
}

pub(crate) struct InstanceHandles {
    pub(in crate::ofx::host) properties: sys::OfxPropertySetHandle,
    parameters: Vec<Parameter>,
    // Retained source/output handles remain host-owned until terminal Destroy.
    pub(in crate::ofx::host) clips: [sys::OfxImageClipHandle; 2],
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
    #[cfg(feature = "test-support")]
    if super::test_support::instance_fault(8) {
        return Err(Status::MEMORY);
    }
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
                c"OfxPropInstanceData".as_ptr(),
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
                c"OfxPropInstanceData".as_ptr(),
                0,
                pointer,
            )
        })
    }

    pub(super) fn create(&self) -> Result<(), Status> {
        let handles = self.instance_handles()?;
        let value = Owner::new(self, handles)?;
        #[cfg(feature = "test-support")]
        if super::test_support::instance_fault(1) {
            return Err(Status::MEMORY);
        }
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
        #[cfg(feature = "test-support")]
        super::test_support::instance_created();
        let bound = owner.bind_address(pointer as usize);
        #[cfg(feature = "test-support")]
        if bound.is_err() {
            super::test_support::instance_reclaimed();
        }
        bound?;
        let publication_attempted = std::cell::Cell::new(false);
        let result = catch_unwind(AssertUnwindSafe(|| {
            owner.host_call(|| {
                #[cfg(feature = "test-support")]
                if super::test_support::instance_fault(7) {
                    panic!("OFX unpublished construction panic");
                }
                publication_attempted.set(true);
                self.instance_data(&owner.handles, pointer.cast())?;
                #[cfg(feature = "test-support")]
                if super::test_support::instance_fault(3) {
                    panic!("OFX post-publication construction panic");
                }
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
                    #[cfg(feature = "test-support")]
                    super::test_support::instance_reclaimed();
                    drop(owner);
                    return Err(original);
                }
                let detached = catch_unwind(AssertUnwindSafe(|| {
                    owner.host_call(|| self.instance_data(&owner.handles, std::ptr::null_mut()))
                }));
                if matches!(detached, Ok(Ok(()))) {
                    #[cfg(feature = "test-support")]
                    super::test_support::instance_reclaimed();
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
        let retire = catch_unwind(AssertUnwindSafe(|| {
            #[cfg(feature = "test-support")]
            {
                super::test_support::instance_retired();
                if super::test_support::instance_fault(5) {
                    return Err(Status::FATAL);
                }
                if super::test_support::instance_fault(6) {
                    panic!("OFX terminal retirement panic");
                }
            }
            owner.retire()
        }));
        if clear.is_err() || retire.is_err() {
            super::super::fail_admission(&owner.loaded);
        }
        #[cfg(feature = "test-support")]
        super::test_support::instance_reclaimed();
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
            let reason = callback.action_string(c"OfxPropChangeReason")?;
            let (plugin, user) = match reason.as_slice() {
                b"OfxChangePluginEdited" => (true, false),
                b"OfxChangeUserEdited" => (false, true),
                b"OfxChangeTime" => (false, false),
                _ => return Err(Status::FAILED),
            };
            if changed {
                let _time = callback.action_double(c"OfxPropTime", 0)?;
                let _scale_x = callback.action_double(c"OfxImageEffectPropRenderScale", 0)?;
                let _scale_y = callback.action_double(c"OfxImageEffectPropRenderScale", 1)?;
                let kind = callback.action_string(c"OfxPropType")?;
                let name = callback.action_string(c"OfxPropName")?;
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
            .property_bytes(properties, c"OfxImageEffectPropContext")?
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
                c"OfxPropInstanceData".as_ptr(),
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
        for (clip, name) in clips.iter_mut().zip([c"Source", c"Output"]) {
            let mut clip_properties = std::ptr::null_mut();
            // SAFETY: Exact vendored outputs, live effect, immutable clip name.
            Status::acquisition(unsafe {
                (self.loaded.suites.clip_handle)(
                    self.effect,
                    name.as_ptr(),
                    clip,
                    &mut clip_properties,
                )
            })?;
            if clip.is_null() || clip_properties.is_null() {
                return Err(Status::BAD_HANDLE);
            }
        }
        let mut acquired = Vec::new();
        acquired
            .try_reserve_exact(REQUIRED.len())
            .map_err(|_| Status::MEMORY)?;
        for &(name, kind) in REQUIRED {
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
            if self.property_bytes(type_properties, c"OfxParamPropType")? != kind.ofx().to_bytes() {
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
            acquired.push(Parameter {
                name,
                kind,
                handle,
                properties,
            });
        }
        Ok(InstanceHandles {
            properties,
            parameters: acquired,
            clips,
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
    pub(crate) fn label(&self, name: &CStr) -> Result<Vec<u8>, Status> {
        self.presentation(name, c"OfxPropLabel", false)
    }
    pub(crate) fn hint(&self, name: &CStr) -> Result<Vec<u8>, Status> {
        self.presentation(name, c"OfxParamPropHint", true)
    }
    fn presentation(
        &self,
        name: &CStr,
        property: &CStr,
        optional: bool,
    ) -> Result<Vec<u8>, Status> {
        let mut value = std::ptr::null_mut();
        let parameter = self.parameter(name)?;
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
    fn parameter(&self, name: &CStr) -> Result<&Parameter, Status> {
        self.handles
            .parameters
            .iter()
            .find(|parameter| parameter.name == name)
            .ok_or(Status::BAD_HANDLE)
    }
    fn typed(&self, name: &CStr, kind: Kind) -> Result<&Parameter, Status> {
        let parameter = self.parameter(name)?;
        if parameter.kind != kind {
            return Err(Status::FAILED);
        }
        Ok(parameter)
    }
    pub(crate) fn integer(&self, name: &CStr) -> Result<i32, Status> {
        let parameter = self.parameter(name)?;
        if !matches!(parameter.kind, Kind::Boolean | Kind::Choice | Kind::Integer) {
            return Err(Status::FAILED);
        }
        let mut value = 0i32;
        // SAFETY: Int/Choice/Boolean current getters receive exactly int*.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_get)(parameter.handle, &mut value as *mut i32)
        })?;
        Ok(if parameter.kind == Kind::Boolean {
            i32::from(value != 0)
        } else {
            value
        })
    }
    pub(crate) fn double(&self, name: &CStr) -> Result<f64, Status> {
        let parameter = self.typed(name, Kind::Double)?;
        let mut value = 0.0f64;
        // SAFETY: A scalar Double current getter receives exactly double*.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_get)(parameter.handle, &mut value as *mut f64)
        })?;
        Ok(value)
    }
    pub(crate) fn double2(&self, name: &CStr) -> Result<[f64; 2], Status> {
        let parameter = self.typed(name, Kind::Double2D)?;
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
    pub(crate) fn double3(&self, name: &CStr) -> Result<[f64; 3], Status> {
        let parameter = self.typed(name, Kind::Double3D)?;
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
    pub(crate) fn string_bytes(&self, name: &CStr) -> Result<Vec<u8>, Status> {
        self.read_string(name, None)
    }
    pub(crate) fn profile_key(&self, name: &CStr, default: &[u8]) -> Result<Vec<u8>, Status> {
        self.read_string(name, Some(default))
    }
    fn read_string(&self, name: &CStr, default: Option<&[u8]>) -> Result<Vec<u8>, Status> {
        let parameter = self.typed(name, Kind::StringChoice)?;
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
        #[cfg(feature = "test-support")]
        if super::test_support::instance_fault(2) {
            return Err(Status::MEMORY);
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
        #[cfg(feature = "test-support")]
        if default.is_some()
            && ((name == c"FilmProfileKey" && super::test_support::render_fault(1))
                || (name == c"PrintProfileKey" && super::test_support::render_fault(2)))
        {
            return Err(Status::MEMORY);
        }
        copy_bytes(bytes)
    }
    pub(crate) fn set_integer(&self, name: &CStr, value: i32) -> Result<(), Status> {
        let parameter = self.parameter(name)?;
        if !matches!(parameter.kind, Kind::Boolean | Kind::Choice | Kind::Integer) {
            return Err(Status::FAILED);
        }
        let value = if parameter.kind == Kind::Boolean {
            i32::from(value != 0)
        } else {
            value
        };
        // SAFETY: Exact promoted int setter argument for the admitted kind.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(parameter.handle, value)
        })
    }
    pub(crate) fn set_double(&self, name: &CStr, value: f64) -> Result<(), Status> {
        let parameter = self.typed(name, Kind::Double)?;
        // SAFETY: Exact promoted double setter argument.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(parameter.handle, value)
        })
    }
    pub(crate) fn set_double2(&self, name: &CStr, value: [f64; 2]) -> Result<(), Status> {
        let parameter = self.typed(name, Kind::Double2D)?;
        // SAFETY: Exact ordered promoted double components.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(parameter.handle, value[0], value[1])
        })
    }
    pub(crate) fn set_double3(&self, name: &CStr, value: [f64; 3]) -> Result<(), Status> {
        let parameter = self.typed(name, Kind::Double3D)?;
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
    pub(crate) fn set_string_bytes(&self, name: &CStr, value: &[u8]) -> Result<(), Status> {
        let parameter = self.typed(name, Kind::StringChoice)?;
        let value = terminated(value)?;
        // SAFETY: Terminated immutable bytes remain live through the synchronous setter.
        Status::acquisition(unsafe {
            (self.callback.loaded.suites.parameter_set)(
                parameter.handle,
                value.as_ptr().cast::<std::ffi::c_char>(),
            )
        })
    }
    fn properties(&self, name: &CStr) -> Result<Properties<'_>, Status> {
        Ok(Properties {
            callback: self.callback,
            handle: self.parameter(name)?.properties,
        })
    }
    pub(crate) fn set_enabled(&self, name: &CStr, enabled: bool) -> Result<(), Status> {
        self.properties(name)?
            .set_int(c"OfxParamPropEnabled", 0, i32::from(enabled))
    }
    pub(crate) fn set_secret(&self, name: &CStr, secret: bool) -> Result<(), Status> {
        self.properties(name)?
            .set_int(c"OfxParamPropSecret", 0, i32::from(secret))
    }
    pub(crate) fn set_label(&self, name: &CStr, bytes: &[u8]) -> Result<(), Status> {
        self.set_text(name, c"OfxPropLabel", bytes)
    }
    pub(crate) fn set_hint(&self, name: &CStr, bytes: &[u8]) -> Result<(), Status> {
        let bytes = terminated(bytes)?;
        let text = CStr::from_bytes_with_nul(&bytes).map_err(|_| Status::FAILED)?;
        // OFXS ignores the optional host status, after constructing the string.
        // Local fallible storage remains an allocation failure for its action.
        let _ = self
            .properties(name)?
            .set_string(c"OfxParamPropHint", 0, text);
        Ok(())
    }
    fn set_text(&self, name: &CStr, property: &CStr, bytes: &[u8]) -> Result<(), Status> {
        let bytes = terminated(bytes)?;
        let text = CStr::from_bytes_with_nul(&bytes).map_err(|_| Status::FAILED)?;
        self.properties(name)?.set_string(property, 0, text)
    }
}
