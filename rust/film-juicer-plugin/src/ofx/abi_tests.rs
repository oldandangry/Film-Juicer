//! Observations and independently typed consumers of the vendored OFX boundary.

use core::ffi::{c_char, c_int, c_uint, c_void};
use core::mem::{align_of, offset_of, size_of};

use super::sys::*;

// Ordered against the independently frozen native inventory, never bindgen output.
static FACTS: &[usize] = &[
    size_of::<OfxHost>(),
    align_of::<OfxHost>(),
    offset_of!(OfxHost, host),
    offset_of!(OfxHost, fetchSuite),
    size_of::<OfxPlugin>(),
    align_of::<OfxPlugin>(),
    offset_of!(OfxPlugin, pluginApi),
    offset_of!(OfxPlugin, apiVersion),
    offset_of!(OfxPlugin, pluginIdentifier),
    offset_of!(OfxPlugin, pluginVersionMajor),
    offset_of!(OfxPlugin, pluginVersionMinor),
    offset_of!(OfxPlugin, setHost),
    offset_of!(OfxPlugin, mainEntry),
    size_of::<OfxPropertySuiteV1>(),
    align_of::<OfxPropertySuiteV1>(),
    offset_of!(OfxPropertySuiteV1, propSetPointer),
    offset_of!(OfxPropertySuiteV1, propSetString),
    offset_of!(OfxPropertySuiteV1, propSetDouble),
    offset_of!(OfxPropertySuiteV1, propSetInt),
    offset_of!(OfxPropertySuiteV1, propSetPointerN),
    offset_of!(OfxPropertySuiteV1, propSetStringN),
    offset_of!(OfxPropertySuiteV1, propSetDoubleN),
    offset_of!(OfxPropertySuiteV1, propSetIntN),
    offset_of!(OfxPropertySuiteV1, propGetPointer),
    offset_of!(OfxPropertySuiteV1, propGetString),
    offset_of!(OfxPropertySuiteV1, propGetDouble),
    offset_of!(OfxPropertySuiteV1, propGetInt),
    offset_of!(OfxPropertySuiteV1, propGetPointerN),
    offset_of!(OfxPropertySuiteV1, propGetStringN),
    offset_of!(OfxPropertySuiteV1, propGetDoubleN),
    offset_of!(OfxPropertySuiteV1, propGetIntN),
    offset_of!(OfxPropertySuiteV1, propReset),
    offset_of!(OfxPropertySuiteV1, propGetDimension),
    size_of::<OfxParameterSuiteV1>(),
    align_of::<OfxParameterSuiteV1>(),
    offset_of!(OfxParameterSuiteV1, paramDefine),
    offset_of!(OfxParameterSuiteV1, paramGetHandle),
    offset_of!(OfxParameterSuiteV1, paramSetGetPropertySet),
    offset_of!(OfxParameterSuiteV1, paramGetPropertySet),
    offset_of!(OfxParameterSuiteV1, paramGetValue),
    offset_of!(OfxParameterSuiteV1, paramGetValueAtTime),
    offset_of!(OfxParameterSuiteV1, paramGetDerivative),
    offset_of!(OfxParameterSuiteV1, paramGetIntegral),
    offset_of!(OfxParameterSuiteV1, paramSetValue),
    offset_of!(OfxParameterSuiteV1, paramSetValueAtTime),
    offset_of!(OfxParameterSuiteV1, paramGetNumKeys),
    offset_of!(OfxParameterSuiteV1, paramGetKeyTime),
    offset_of!(OfxParameterSuiteV1, paramGetKeyIndex),
    offset_of!(OfxParameterSuiteV1, paramDeleteKey),
    offset_of!(OfxParameterSuiteV1, paramDeleteAllKeys),
    offset_of!(OfxParameterSuiteV1, paramCopy),
    offset_of!(OfxParameterSuiteV1, paramEditBegin),
    offset_of!(OfxParameterSuiteV1, paramEditEnd),
    size_of::<OfxImageEffectSuiteV1>(),
    align_of::<OfxImageEffectSuiteV1>(),
    offset_of!(OfxImageEffectSuiteV1, getPropertySet),
    offset_of!(OfxImageEffectSuiteV1, getParamSet),
    offset_of!(OfxImageEffectSuiteV1, clipDefine),
    offset_of!(OfxImageEffectSuiteV1, clipGetHandle),
    offset_of!(OfxImageEffectSuiteV1, clipGetPropertySet),
    offset_of!(OfxImageEffectSuiteV1, clipGetImage),
    offset_of!(OfxImageEffectSuiteV1, clipReleaseImage),
    offset_of!(OfxImageEffectSuiteV1, clipGetRegionOfDefinition),
    offset_of!(OfxImageEffectSuiteV1, abort),
    offset_of!(OfxImageEffectSuiteV1, imageMemoryAlloc),
    offset_of!(OfxImageEffectSuiteV1, imageMemoryFree),
    offset_of!(OfxImageEffectSuiteV1, imageMemoryLock),
    offset_of!(OfxImageEffectSuiteV1, imageMemoryUnlock),
    size_of::<OfxMessageSuiteV1>(),
    align_of::<OfxMessageSuiteV1>(),
    offset_of!(OfxMessageSuiteV1, message),
    size_of::<OfxRangeD>(),
    align_of::<OfxRangeD>(),
    offset_of!(OfxRangeD, min),
    offset_of!(OfxRangeD, max),
    size_of::<OfxPointI>(),
    align_of::<OfxPointI>(),
    offset_of!(OfxPointI, x),
    offset_of!(OfxPointI, y),
    size_of::<OfxPointD>(),
    align_of::<OfxPointD>(),
    offset_of!(OfxPointD, x),
    offset_of!(OfxPointD, y),
    size_of::<OfxRectI>(),
    align_of::<OfxRectI>(),
    offset_of!(OfxRectI, x1),
    offset_of!(OfxRectI, y1),
    offset_of!(OfxRectI, x2),
    offset_of!(OfxRectI, y2),
    size_of::<OfxRectD>(),
    align_of::<OfxRectD>(),
    offset_of!(OfxRectD, x1),
    offset_of!(OfxRectD, y1),
    offset_of!(OfxRectD, x2),
    offset_of!(OfxRectD, y2),
    size_of::<OfxStatus>(),
    align_of::<OfxStatus>(),
    size_of::<OfxTime>(),
    align_of::<OfxTime>(),
    size_of::<OfxPropertySetHandle>(),
    align_of::<OfxPropertySetHandle>(),
    size_of::<OfxParamHandle>(),
    align_of::<OfxParamHandle>(),
    size_of::<OfxParamSetHandle>(),
    align_of::<OfxParamSetHandle>(),
    size_of::<OfxImageEffectHandle>(),
    align_of::<OfxImageEffectHandle>(),
    size_of::<OfxImageClipHandle>(),
    align_of::<OfxImageClipHandle>(),
    size_of::<OfxImageMemoryHandle>(),
    align_of::<OfxImageMemoryHandle>(),
];

const _: () = assert!(FACTS.len() == 111);

static VALUES: &[usize] = &[
    kOfxStatOK as usize,
    kOfxStatFailed as usize,
    kOfxStatErrFatal as usize,
    kOfxStatErrUnknown as usize,
    kOfxStatErrMissingHostFeature as usize,
    kOfxStatErrUnsupported as usize,
    kOfxStatErrExists as usize,
    kOfxStatErrFormat as usize,
    kOfxStatErrMemory as usize,
    kOfxStatErrBadHandle as usize,
    kOfxStatErrBadIndex as usize,
    kOfxStatErrValue as usize,
    kOfxStatReplyYes as usize,
    kOfxStatReplyNo as usize,
    kOfxStatReplyDefault as usize,
    kOfxStatErrImageFormat as usize,
    kOfxImageEffectPluginApiVersion as usize,
];

const _: () = assert!(VALUES.len() == 17);

static STRINGS: &[&[u8]] = &[
    kOfxImageEffectPluginApi,
    kOfxPropertySuite,
    kOfxParameterSuite,
    kOfxImageEffectSuite,
    kOfxMessageSuite,
    kOfxActionLoad,
    kOfxActionUnload,
    kOfxActionDescribe,
    kOfxActionCreateInstance,
    kOfxActionDestroyInstance,
    kOfxActionPurgeCaches,
    kOfxActionSyncPrivateData,
    kOfxActionBeginInstanceChanged,
    kOfxActionInstanceChanged,
    kOfxActionEndInstanceChanged,
    kOfxActionBeginInstanceEdit,
    kOfxActionEndInstanceEdit,
    kOfxImageEffectActionDescribeInContext,
    kOfxImageEffectActionRender,
    kOfxImageEffectActionBeginSequenceRender,
    kOfxImageEffectActionEndSequenceRender,
    kOfxImageEffectActionIsIdentity,
    kOfxImageEffectActionGetRegionOfDefinition,
    kOfxImageEffectActionGetRegionsOfInterest,
    kOfxImageEffectActionGetFramesNeeded,
    kOfxImageEffectActionGetClipPreferences,
    kOfxImageEffectActionGetTimeDomain,
    kOfxPropType,
    kOfxPropName,
    kOfxPropLabel,
    kOfxPropShortLabel,
    kOfxPropLongLabel,
    kOfxPropVersion,
    kOfxPropVersionLabel,
    kOfxPropPluginDescription,
    kOfxPropInstanceData,
    kOfxPropTime,
    kOfxPropChangeReason,
    kOfxPropIsInteractive,
    kOfxPropAPIVersion,
    kOfxTypeParameter,
    kOfxTypeClip,
    kOfxTypeImageEffect,
    kOfxChangePluginEdited,
    kOfxChangeUserEdited,
    kOfxChangeTime,
    kOfxImageEffectContextFilter,
    kOfxImageEffectSimpleSourceClipName,
    kOfxImageEffectOutputClipName,
    kOfxBitDepthFloat,
    kOfxImageComponentRGBA,
    kOfxImageEffectRenderUnsafe,
    kOfxImageEffectPluginPropGrouping,
    kOfxImageEffectPluginPropHostFrameThreading,
    kOfxImageEffectPluginPropSingleInstance,
    kOfxImageEffectPluginRenderThreadSafety,
    kOfxImageEffectPropSupportedContexts,
    kOfxImageEffectPropSupportedPixelDepths,
    kOfxImageEffectPropSupportsMultiResolution,
    kOfxImageEffectPropSupportsTiles,
    kOfxImageEffectPropSupportedComponents,
    kOfxImageEffectPropTemporalClipAccess,
    kOfxImageClipPropOptional,
    kOfxImageEffectPropCudaRenderSupported,
    kOfxImageEffectPropCudaStreamSupported,
    kOfxImageEffectPropCudaEnabled,
    kOfxImageEffectPropCudaStream,
    kOfxImageEffectPropContext,
    kOfxImageEffectPropRenderScale,
    kOfxImageEffectPropRenderWindow,
    kOfxImageEffectPropFrameRange,
    kOfxImageEffectPropFrameRate,
    kOfxImageEffectPropFrameStep,
    kOfxImageEffectPropRegionOfDefinition,
    kOfxImageEffectPropRegionOfInterest,
    kOfxImageEffectPropPixelDepth,
    kOfxImageEffectPropComponents,
    kOfxImageEffectPropPreMultiplication,
    kOfxImageEffectPropFieldToRender,
    kOfxImageEffectPropSequentialRenderStatus,
    kOfxImageEffectPropInteractiveRenderStatus,
    kOfxImageEffectPropRenderQualityDraft,
    kOfxImageEffectPropProjectSize,
    kOfxImageEffectPropProjectOffset,
    kOfxImageEffectPropProjectExtent,
    kOfxImageEffectPropProjectPixelAspectRatio,
    kOfxImageClipPropConnected,
    kOfxImagePropData,
    kOfxImagePropBounds,
    kOfxImagePropRowBytes,
    kOfxImagePropRegionOfDefinition,
    kOfxImagePropPixelAspectRatio,
    kOfxImagePropField,
    kOfxImagePropUniqueIdentifier,
    kOfxParamTypeBoolean,
    kOfxParamTypeChoice,
    kOfxParamTypeStrChoice,
    kOfxParamTypeDouble,
    kOfxParamTypeDouble2D,
    kOfxParamTypeDouble3D,
    kOfxParamTypeInteger,
    kOfxParamTypeGroup,
    kOfxParamTypePage,
    kOfxParamTypePushButton,
    kOfxParamPropType,
    kOfxParamPropDefault,
    kOfxParamPropMin,
    kOfxParamPropMax,
    kOfxParamPropDisplayMin,
    kOfxParamPropDisplayMax,
    kOfxParamPropIncrement,
    kOfxParamPropDigits,
    kOfxParamPropDoubleType,
    kOfxParamPropDimensionLabel,
    kOfxParamPropChoiceOption,
    kOfxParamPropChoiceEnum,
    kOfxParamHostPropSupportsStrChoiceAnimation,
    kOfxParamPropParent,
    kOfxParamPropPageChild,
    kOfxParamPropGroupOpen,
    kOfxParamPropHint,
    kOfxParamPropScriptName,
    kOfxParamPropAnimates,
    kOfxParamPropEnabled,
    kOfxParamPropSecret,
    kOfxParamPropEvaluateOnChange,
    kOfxParamPropPersistant,
    kOfxParamPropCanUndo,
    kOfxParamPropCacheInvalidation,
    kOfxParamDoubleTypePlain,
    kOfxParamDoubleTypeScale,
    kOfxParamInvalidateValueChange,
    kOfxMessageFatal,
    kOfxMessageError,
    kOfxMessageWarning,
    kOfxMessageMessage,
    kOfxMessageLog,
    kOfxMessageQuestion,
    kOfxImageComponentRGB,
    kOfxImageOpaque,
    kOfxImagePreMultiplied,
    kOfxImageUnPreMultiplied,
    kOfxImageFieldNone,
];

const _: () = assert!(STRINGS.len() == 143);

/// Return the immutable, library-lived layout observations.
///
/// # Safety
/// `count` points to one writable usize for this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_test_ofx_abi_facts(count: *mut usize) -> *const usize {
    // SAFETY: The native fixture supplies a writable, disjoint count element.
    unsafe { count.write(FACTS.len()) };
    FACTS.as_ptr()
}

/// Return the immutable, library-lived numeric constants.
///
/// # Safety
/// `count` points to one writable usize for this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_test_ofx_abi_values(count: *mut usize) -> *const usize {
    // SAFETY: The native fixture supplies a writable, disjoint count element.
    unsafe { count.write(VALUES.len()) };
    VALUES.as_ptr()
}

/// Return one immutable, library-lived string including its terminating NUL.
/// An out-of-range index returns null and count zero.
///
/// # Safety
/// `count` points to one writable usize for this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_test_ofx_abi_string(index: usize, count: *mut usize) -> *const u8 {
    let bytes = STRINGS.get(index).copied().unwrap_or_default();
    // SAFETY: The native fixture supplies a writable, disjoint count element.
    unsafe { count.write(bytes.len()) };
    if bytes.is_empty() {
        core::ptr::null()
    } else {
        bytes.as_ptr()
    }
}

// Referent types cannot coerce mutable pointers to const pointers. These checks
// require exact declared types, including signed coordinates, pointer
// mutability, nullable callbacks, C calling convention and every variadic tail.
const _: fn(&OfxHost) = |record| {
    let _: &OfxPropertySetHandle = &record.host;
};
const _: fn(&OfxHost) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int) -> *const c_void,
    > = &record.fetchSuite;
};
const _: fn(&OfxPlugin) = |record| {
    let _: &*const c_char = &record.pluginApi;
};
const _: fn(&OfxPlugin) = |record| {
    let _: &c_int = &record.apiVersion;
};
const _: fn(&OfxPlugin) = |record| {
    let _: &*const c_char = &record.pluginIdentifier;
};
const _: fn(&OfxPlugin) = |record| {
    let _: &c_uint = &record.pluginVersionMajor;
};
const _: fn(&OfxPlugin) = |record| {
    let _: &c_uint = &record.pluginVersionMinor;
};
const _: fn(&OfxPlugin) = |record| {
    let _: &Option<unsafe extern "C" fn(*mut OfxHost)> = &record.setHost;
};
const _: fn(&OfxPlugin) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            *const c_char,
            *const c_void,
            OfxPropertySetHandle,
            OfxPropertySetHandle,
        ) -> OfxStatus,
    > = &record.mainEntry;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut c_void) -> OfxStatus,
    > = &record.propSetPointer;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxPropertySetHandle,
            *const c_char,
            c_int,
            *const c_char,
        ) -> OfxStatus,
    > = &record.propSetString;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, f64) -> OfxStatus,
    > = &record.propSetDouble;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, c_int) -> OfxStatus,
    > = &record.propSetInt;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxPropertySetHandle,
            *const c_char,
            c_int,
            *const *mut c_void,
        ) -> OfxStatus,
    > = &record.propSetPointerN;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxPropertySetHandle,
            *const c_char,
            c_int,
            *const *const c_char,
        ) -> OfxStatus,
    > = &record.propSetStringN;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *const f64) -> OfxStatus,
    > = &record.propSetDoubleN;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *const c_int) -> OfxStatus,
    > = &record.propSetIntN;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxPropertySetHandle,
            *const c_char,
            c_int,
            *mut *mut c_void,
        ) -> OfxStatus,
    > = &record.propGetPointer;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxPropertySetHandle,
            *const c_char,
            c_int,
            *mut *mut c_char,
        ) -> OfxStatus,
    > = &record.propGetString;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut f64) -> OfxStatus,
    > = &record.propGetDouble;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut c_int) -> OfxStatus,
    > = &record.propGetInt;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxPropertySetHandle,
            *const c_char,
            c_int,
            *mut *mut c_void,
        ) -> OfxStatus,
    > = &record.propGetPointerN;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxPropertySetHandle,
            *const c_char,
            c_int,
            *mut *mut c_char,
        ) -> OfxStatus,
    > = &record.propGetStringN;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut f64) -> OfxStatus,
    > = &record.propGetDoubleN;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut c_int) -> OfxStatus,
    > = &record.propGetIntN;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxPropertySetHandle, *const c_char) -> OfxStatus> =
        &record.propReset;
};
const _: fn(&OfxPropertySuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, *mut c_int) -> OfxStatus,
    > = &record.propGetDimension;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxParamSetHandle,
            *const c_char,
            *const c_char,
            *mut OfxPropertySetHandle,
        ) -> OfxStatus,
    > = &record.paramDefine;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxParamSetHandle,
            *const c_char,
            *mut OfxParamHandle,
            *mut OfxPropertySetHandle,
        ) -> OfxStatus,
    > = &record.paramGetHandle;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxParamSetHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    > = &record.paramSetGetPropertySet;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, *mut OfxPropertySetHandle) -> OfxStatus> =
        &record.paramGetPropertySet;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, ...) -> OfxStatus> = &record.paramGetValue;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, OfxTime, ...) -> OfxStatus> =
        &record.paramGetValueAtTime;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, OfxTime, ...) -> OfxStatus> =
        &record.paramGetDerivative;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, OfxTime, OfxTime, ...) -> OfxStatus> =
        &record.paramGetIntegral;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, ...) -> OfxStatus> = &record.paramSetValue;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, OfxTime, ...) -> OfxStatus> =
        &record.paramSetValueAtTime;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, *mut c_uint) -> OfxStatus> =
        &record.paramGetNumKeys;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, c_uint, *mut OfxTime) -> OfxStatus> =
        &record.paramGetKeyTime;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, OfxTime, c_int, *mut c_int) -> OfxStatus> =
        &record.paramGetKeyIndex;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle, OfxTime) -> OfxStatus> =
        &record.paramDeleteKey;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamHandle) -> OfxStatus> = &record.paramDeleteAllKeys;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxParamHandle,
            OfxParamHandle,
            OfxTime,
            *const OfxRangeD,
        ) -> OfxStatus,
    > = &record.paramCopy;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamSetHandle, *const c_char) -> OfxStatus> =
        &record.paramEditBegin;
};
const _: fn(&OfxParameterSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxParamSetHandle) -> OfxStatus> = &record.paramEditEnd;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxImageEffectHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    > = &record.getPropertySet;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxImageEffectHandle, *mut OfxParamSetHandle) -> OfxStatus,
    > = &record.getParamSet;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxImageEffectHandle,
            *const c_char,
            *mut OfxPropertySetHandle,
        ) -> OfxStatus,
    > = &record.clipDefine;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxImageEffectHandle,
            *const c_char,
            *mut OfxImageClipHandle,
            *mut OfxPropertySetHandle,
        ) -> OfxStatus,
    > = &record.clipGetHandle;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxImageClipHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    > = &record.clipGetPropertySet;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            OfxImageClipHandle,
            OfxTime,
            *const OfxRectD,
            *mut OfxPropertySetHandle,
        ) -> OfxStatus,
    > = &record.clipGetImage;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxPropertySetHandle) -> OfxStatus> =
        &record.clipReleaseImage;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxImageClipHandle, OfxTime, *mut OfxRectD) -> OfxStatus> =
        &record.clipGetRegionOfDefinition;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxImageEffectHandle) -> c_int> = &record.abort;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(OfxImageEffectHandle, usize, *mut OfxImageMemoryHandle) -> OfxStatus,
    > = &record.imageMemoryAlloc;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxImageMemoryHandle) -> OfxStatus> =
        &record.imageMemoryFree;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxImageMemoryHandle, *mut *mut c_void) -> OfxStatus> =
        &record.imageMemoryLock;
};
const _: fn(&OfxImageEffectSuiteV1) = |record| {
    let _: &Option<unsafe extern "C" fn(OfxImageMemoryHandle) -> OfxStatus> =
        &record.imageMemoryUnlock;
};
const _: fn(&OfxMessageSuiteV1) = |record| {
    let _: &Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const c_char,
            *const c_char,
            *const c_char,
            ...
        ) -> OfxStatus,
    > = &record.message;
};
const _: fn(&OfxRangeD) = |record| {
    let _: &f64 = &record.min;
};
const _: fn(&OfxRangeD) = |record| {
    let _: &f64 = &record.max;
};
const _: fn(&OfxPointI) = |record| {
    let _: &c_int = &record.x;
};
const _: fn(&OfxPointI) = |record| {
    let _: &c_int = &record.y;
};
const _: fn(&OfxPointD) = |record| {
    let _: &f64 = &record.x;
};
const _: fn(&OfxPointD) = |record| {
    let _: &f64 = &record.y;
};
const _: fn(&OfxRectI) = |record| {
    let _: &c_int = &record.x1;
};
const _: fn(&OfxRectI) = |record| {
    let _: &c_int = &record.y1;
};
const _: fn(&OfxRectI) = |record| {
    let _: &c_int = &record.x2;
};
const _: fn(&OfxRectI) = |record| {
    let _: &c_int = &record.y2;
};
const _: fn(&OfxRectD) = |record| {
    let _: &f64 = &record.x1;
};
const _: fn(&OfxRectD) = |record| {
    let _: &f64 = &record.y1;
};
const _: fn(&OfxRectD) = |record| {
    let _: &f64 = &record.x2;
};
const _: fn(&OfxRectD) = |record| {
    let _: &f64 = &record.y2;
};

const _: OfxStatus = -1_i32;
const _: OfxTime = 1.0_f64;
const _: fn(&OfxPropertySetHandle) = |value| {
    let _: &*mut OfxPropertySetStruct = value;
};
const _: fn(&OfxParamHandle) = |value| {
    let _: &*mut OfxParamStruct = value;
};
const _: fn(&OfxParamSetHandle) = |value| {
    let _: &*mut OfxParamSetStruct = value;
};
const _: fn(&OfxImageEffectHandle) = |value| {
    let _: &*mut OfxImageEffectStruct = value;
};
const _: fn(&OfxImageClipHandle) = |value| {
    let _: &*mut OfxImageClipStruct = value;
};
const _: fn(&OfxImageMemoryHandle) = |value| {
    let _: &*mut OfxImageMemoryStruct = value;
};
const _: fn(&OfxPluginEntryPoint) = |value| {
    let _: &Option<
        unsafe extern "C" fn(
            *const c_char,
            *const c_void,
            OfxPropertySetHandle,
            OfxPropertySetHandle,
        ) -> c_int,
    > = value;
};

// Native fixture peers arrive as borrowed records, so the shared test-support
// archive introduces no C-stub dependency in its other real consumers.
mod calls {
    use core::ffi::CStr;
    use core::ptr::{null, null_mut};

    use super::*;

    /// # Safety
    /// `name` is a live NUL-terminated string for this synchronous call.
    /// The opaque host token is only compared and is never dereferenced.
    unsafe extern "C" fn fetch_suite(
        host: OfxPropertySetHandle,
        name: *const c_char,
        version: c_int,
    ) -> *const c_void {
        // SAFETY: The synchronous fixture supplies its live NUL-terminated name.
        if host as usize == 1 && unsafe { CStr::from_ptr(name) } == c"fixture.suite" && version == 7
        {
            2_usize as *const c_void
        } else {
            null()
        }
    }

    /// # Safety
    /// `host` lends one exclusive, aligned, writable record for this call.
    unsafe extern "C" fn set_host(host: *mut OfxHost) {
        // SAFETY: The fixture lends one exclusive writable host copy for this call.
        unsafe { (*host).host = 6_usize as OfxPropertySetHandle };
    }

    /// # Safety
    /// `action` is a live NUL-terminated string for this synchronous call.
    /// The other arguments are opaque tokens and are never dereferenced.
    unsafe extern "C" fn main_entry(
        action: *const c_char,
        handle: *const c_void,
        input: OfxPropertySetHandle,
        output: OfxPropertySetHandle,
    ) -> OfxStatus {
        // SAFETY: The synchronous fixture supplies its live NUL-terminated action.
        if unsafe { CStr::from_ptr(action) } == c"fixture.action"
            && handle as usize == 3
            && input as usize == 4
            && output as usize == 5
        {
            17
        } else {
            1
        }
    }

    /// Exercise native and Rust functions through the generated callback fields.
    ///
    /// # Safety
    /// Inputs are live immutable fixture records. Outputs are exclusively writable
    /// and disjoint from both inputs. All calls are synchronous and retain no input.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn fj_test_ofx_abi_callbacks(
        native_host: *const OfxHost,
        native_plugin: *const OfxPlugin,
        rust_host: *mut OfxHost,
        rust_plugin: *mut OfxPlugin,
    ) -> c_int {
        let host = OfxHost {
            host: 1_usize as OfxPropertySetHandle,
            fetchSuite: Some(fetch_suite),
        };
        let plugin = OfxPlugin {
            pluginApi: c"fixture".as_ptr(),
            apiVersion: 1,
            pluginIdentifier: c"fixture".as_ptr(),
            pluginVersionMajor: 1,
            pluginVersionMinor: 0,
            setHost: Some(set_host),
            mainEntry: Some(main_entry),
        };
        // SAFETY: The harness lends disjoint output records and immutable native
        // records. Returned Rust function pointers and literal strings are
        // library-lived; neither direction retains caller-owned storage.
        unsafe {
            rust_host.write(host);
            rust_plugin.write(plugin);
            let mut native_host = *native_host;
            let native_plugin = &*native_plugin;
            let (Some(fetch), Some(set), Some(entry)) = (
                native_host.fetchSuite,
                native_plugin.setHost,
                native_plugin.mainEntry,
            ) else {
                return 1;
            };
            if fetch(native_host.host, c"fixture.suite".as_ptr(), 7) as usize != 2 {
                return 1;
            }
            set(&raw mut native_host);
            if native_host.host as usize != 6 {
                return 1;
            }
            c_int::from(
                entry(
                    c"fixture.action".as_ptr(),
                    3_usize as *const c_void,
                    4_usize as OfxPropertySetHandle,
                    5_usize as OfxPropertySetHandle,
                ) != 17,
            )
        }
    }

    /// Exercise promoted argument and output-pointer shapes through actual slots.
    ///
    /// # Safety
    /// `suite` lends an immutable fixture suite with the documented variadic
    /// contracts. Returned strings remain live through this synchronous call.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn fj_test_ofx_abi_parameters(
        suite: *const OfxParameterSuiteV1,
    ) -> c_int {
        // SAFETY: The harness lends an immutable fixture suite. Every
        // vararg below has the exact promoted C type or writable output-pointer
        // type consumed by va_arg for that shape. All output storage is call-local.
        unsafe {
            let suite = &*suite;
            if suite.paramGetDerivative.is_some() || suite.paramGetIntegral.is_some() {
                return 1;
            }
            let (Some(get), Some(get_at_time), Some(set), Some(set_at_time)) = (
                suite.paramGetValue,
                suite.paramGetValueAtTime,
                suite.paramSetValue,
                suite.paramSetValueAtTime,
            ) else {
                return 1;
            };
            let mut calls = 0;
            for at_time in [false, true] {
                macro_rules! get {
                    ($handle:expr, $($output:expr),+) => {{
                        calls += 1;
                        if at_time { get_at_time($handle, 12.5_f64, $($output),+) }
                        else { get($handle, $($output),+) }
                    }};
                }
                macro_rules! set {
                    ($handle:expr, $($input:expr),+) => {{
                        calls += 1;
                        if at_time { set_at_time($handle, 12.5_f64, $($input),+) }
                        else { set($handle, $($input),+) }
                    }};
                }
                for (token, expected) in [(1_usize, 23_i32), (6, 1), (7, 2)] {
                    let handle = token as OfxParamHandle;
                    let mut integer: c_int = 0;
                    if get!(handle, &raw mut integer) != 0
                        || integer != expected
                        || set!(handle, expected) != 0
                    {
                        return 1;
                    }
                }
                let handle = 2_usize as OfxParamHandle;
                let mut text: *mut c_char = null_mut();
                if get!(handle, &raw mut text) != 0
                    || text.is_null()
                    || CStr::from_ptr(text) != c"fixture-value"
                    || set!(handle, c"fixture-value".as_ptr()) != 0
                {
                    return 1;
                }
                let mut x = 0.0_f64;
                let mut y = 0.0_f64;
                let mut z = 0.0_f64;
                if get!(3_usize as OfxParamHandle, &raw mut x) != 0
                    || x != 1.25
                    || set!(3_usize as OfxParamHandle, 1.25_f64) != 0
                    || get!(4_usize as OfxParamHandle, &raw mut x, &raw mut y) != 0
                    || [x, y] != [1.25, 2.5]
                    || set!(4_usize as OfxParamHandle, 1.25_f64, 2.5_f64) != 0
                    || get!(
                        5_usize as OfxParamHandle,
                        &raw mut x,
                        &raw mut y,
                        &raw mut z
                    ) != 0
                    || [x, y, z] != [1.25, 2.5, 3.75]
                    || set!(5_usize as OfxParamHandle, 1.25_f64, 2.5_f64, 3.75_f64) != 0
                {
                    return 1;
                }
            }
            c_int::from(calls != 28)
        }
    }

    /// Exercise the generated variadic V1 message slot with literal `%s`.
    ///
    /// # Safety
    /// `suite` lends an immutable fixture suite implementing the documented
    /// synchronous V1 message contract.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn fj_test_ofx_abi_messages(suite: *const OfxMessageSuiteV1) -> c_int {
        // SAFETY: The harness lends an immutable fixture suite. Literal C strings stay
        // live through the call; the single vararg is the const char* consumed by
        // va_arg. The opaque handle is only compared, never dereferenced.
        unsafe {
            let suite = &*suite;
            let Some(message) = suite.message else {
                return 1;
            };
            c_int::from(
                message(
                    7_usize as *mut c_void,
                    c"fixture.type".as_ptr(),
                    c"fixture.id".as_ptr(),
                    c"%s".as_ptr(),
                    c"100% fixture".as_ptr(),
                ) != 17,
            )
        }
    }
}
