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

#[test]
fn generated_layout_matches_c11_inventory() {
    let declaration = include_str!("../../../../tests/ffi/ofx_abi_facts.inc");
    let mut layout = Vec::new();
    let mut values = Vec::new();
    let mut strings = Vec::new();
    for line in declaration.lines() {
        let Some((kind, args)) = line.split_once('(') else {
            continue;
        };
        let mut parts = args.trim_end_matches(')').split(',').skip(1).map(str::trim);
        match kind {
            "FJ_OFX_TYPE" => layout.extend(parts.map(|value| value.parse::<usize>().unwrap())),
            "FJ_OFX_FIELD" => layout.push(parts.nth(1).unwrap().parse().unwrap()),
            "FJ_OFX_VALUE" => values.push(parts.next().unwrap().parse::<usize>().unwrap()),
            "FJ_OFX_STRING" => {
                let mut bytes = parts.next().unwrap().trim_matches('"').as_bytes().to_vec();
                bytes.push(0);
                strings.push(bytes);
            }
            _ => {}
        }
    }
    assert_eq!(FACTS, layout);
    assert_eq!(VALUES, values);
    assert_eq!(STRINGS, strings);
}
