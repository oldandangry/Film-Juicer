#ifndef FJ_OFX_ABI_TEST_H
#define FJ_OFX_ABI_TEST_H

#include <stddef.h>
#include "ofx_bindings.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Immutable spans and strings are valid for the test process lifetime.
 * Count pointers must be writable; string indices are checked before access. */
const size_t* fj_test_ofx_abi_facts(size_t* count);
const size_t* fj_test_ofx_abi_c_facts(size_t* count);
const size_t* fj_test_ofx_abi_values(size_t* count);
const size_t* fj_test_ofx_abi_c_values(size_t* count);
const unsigned char* fj_test_ofx_abi_c_string(size_t index, size_t* count);
const unsigned char* fj_test_ofx_abi_string(size_t index, size_t* count);
/* Inputs are live immutable fixture records; outputs are exclusive, disjoint
 * writable records. Returned Rust callbacks/strings last for the library lifetime.
 * The suite callbacks implement the documented synchronous fixture vararg shapes. */
int fj_test_ofx_abi_callbacks(const OfxHost* native_host, const OfxPlugin* native_plugin, OfxHost* rust_host, OfxPlugin* rust_plugin);
int fj_test_ofx_abi_parameters(const OfxParameterSuiteV1* suite);
int fj_test_ofx_abi_messages(const OfxMessageSuiteV1* suite);
const OfxHost* fj_test_ofx_host(void);
const OfxPlugin* fj_test_ofx_plugin(void);
int fj_test_ofx_call_rust(const OfxHost* host, const OfxPlugin* plugin);
const OfxParameterSuiteV1* fj_test_ofx_parameter_suite(void);
const OfxMessageSuiteV1* fj_test_ofx_message_suite(void);

#ifdef __cplusplus
}
#endif

typedef OfxPropertySetHandle FjOfxType_OfxHost_host;
typedef const void* (*FjOfxType_OfxHost_fetchSuite)(OfxPropertySetHandle, const char*, int);
typedef const char* FjOfxType_OfxPlugin_pluginApi;
typedef int FjOfxType_OfxPlugin_apiVersion;
typedef const char* FjOfxType_OfxPlugin_pluginIdentifier;
typedef unsigned int FjOfxType_OfxPlugin_pluginVersionMajor;
typedef unsigned int FjOfxType_OfxPlugin_pluginVersionMinor;
typedef void (*FjOfxType_OfxPlugin_setHost)(OfxHost*);
typedef OfxStatus (*FjOfxType_OfxPlugin_mainEntry)(const char*, const void*, OfxPropertySetHandle, OfxPropertySetHandle);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propSetPointer)(OfxPropertySetHandle, const char*, int, void*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propSetString)(OfxPropertySetHandle, const char*, int, const char*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propSetDouble)(OfxPropertySetHandle, const char*, int, double);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propSetInt)(OfxPropertySetHandle, const char*, int, int);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propSetPointerN)(OfxPropertySetHandle, const char*, int, void* const*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propSetStringN)(OfxPropertySetHandle, const char*, int, const char* const*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propSetDoubleN)(OfxPropertySetHandle, const char*, int, const double*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propSetIntN)(OfxPropertySetHandle, const char*, int, const int*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetPointer)(OfxPropertySetHandle, const char*, int, void**);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetString)(OfxPropertySetHandle, const char*, int, char**);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetDouble)(OfxPropertySetHandle, const char*, int, double*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetInt)(OfxPropertySetHandle, const char*, int, int*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetPointerN)(OfxPropertySetHandle, const char*, int, void**);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetStringN)(OfxPropertySetHandle, const char*, int, char**);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetDoubleN)(OfxPropertySetHandle, const char*, int, double*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetIntN)(OfxPropertySetHandle, const char*, int, int*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propReset)(OfxPropertySetHandle, const char*);
typedef OfxStatus (*FjOfxType_OfxPropertySuiteV1_propGetDimension)(OfxPropertySetHandle, const char*, int*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramDefine)(OfxParamSetHandle, const char*, const char*, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetHandle)(OfxParamSetHandle, const char*, OfxParamHandle*, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramSetGetPropertySet)(OfxParamSetHandle, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetPropertySet)(OfxParamHandle, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetValue)(OfxParamHandle, ...);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetValueAtTime)(OfxParamHandle, OfxTime, ...);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetDerivative)(OfxParamHandle, OfxTime, ...);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetIntegral)(OfxParamHandle, OfxTime, OfxTime, ...);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramSetValue)(OfxParamHandle, ...);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramSetValueAtTime)(OfxParamHandle, OfxTime, ...);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetNumKeys)(OfxParamHandle, unsigned int*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetKeyTime)(OfxParamHandle, unsigned int, OfxTime*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramGetKeyIndex)(OfxParamHandle, OfxTime, int, int*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramDeleteKey)(OfxParamHandle, OfxTime);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramDeleteAllKeys)(OfxParamHandle);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramCopy)(OfxParamHandle, OfxParamHandle, OfxTime, const OfxRangeD*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramEditBegin)(OfxParamSetHandle, const char*);
typedef OfxStatus (*FjOfxType_OfxParameterSuiteV1_paramEditEnd)(OfxParamSetHandle);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_getPropertySet)(OfxImageEffectHandle, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_getParamSet)(OfxImageEffectHandle, OfxParamSetHandle*);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_clipDefine)(OfxImageEffectHandle, const char*, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_clipGetHandle)(OfxImageEffectHandle, const char*, OfxImageClipHandle*, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_clipGetPropertySet)(OfxImageClipHandle, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_clipGetImage)(OfxImageClipHandle, OfxTime, const OfxRectD*, OfxPropertySetHandle*);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_clipReleaseImage)(OfxPropertySetHandle);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_clipGetRegionOfDefinition)(OfxImageClipHandle, OfxTime, OfxRectD*);
typedef int (*FjOfxType_OfxImageEffectSuiteV1_abort)(OfxImageEffectHandle);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_imageMemoryAlloc)(OfxImageEffectHandle, size_t, OfxImageMemoryHandle*);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_imageMemoryFree)(OfxImageMemoryHandle);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_imageMemoryLock)(OfxImageMemoryHandle, void**);
typedef OfxStatus (*FjOfxType_OfxImageEffectSuiteV1_imageMemoryUnlock)(OfxImageMemoryHandle);
typedef OfxStatus (*FjOfxType_OfxMessageSuiteV1_message)(void*, const char*, const char*, const char*, ...);
typedef double FjOfxType_OfxRangeD_min;
typedef double FjOfxType_OfxRangeD_max;
typedef int FjOfxType_OfxPointI_x;
typedef int FjOfxType_OfxPointI_y;
typedef double FjOfxType_OfxPointD_x;
typedef double FjOfxType_OfxPointD_y;
typedef int FjOfxType_OfxRectI_x1;
typedef int FjOfxType_OfxRectI_y1;
typedef int FjOfxType_OfxRectI_x2;
typedef int FjOfxType_OfxRectI_y2;
typedef double FjOfxType_OfxRectD_x1;
typedef double FjOfxType_OfxRectD_y1;
typedef double FjOfxType_OfxRectD_x2;
typedef double FjOfxType_OfxRectD_y2;
#endif
