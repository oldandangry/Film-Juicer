//! Reviewed x86_64 layout and signature witnesses for the prepared-call ABI.

use core::mem::{align_of, offset_of, size_of};

use super::sys;

type Precheck =
    unsafe extern "C" fn(*mut core::ffi::c_void, *mut sys::FjErrorBuffer) -> sys::FjStatus;
type Admit = Precheck;
type Release = unsafe extern "C" fn(*mut core::ffi::c_void);
type Inspect = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const sys::FjFrame,
    *mut sys::FjCudaContext,
    *mut sys::FjErrorBuffer,
) -> sys::FjStatus;
type Preview = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const sys::FjRect,
    *const sys::FjRect,
    u32,
    *mut sys::FjAutoExposure,
    *mut sys::FjErrorBuffer,
) -> sys::FjStatus;
type FrameDescriptors = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const super::FjFrameMathInput,
    *mut super::FjNativeFrameDescriptors,
    *mut sys::FjErrorBuffer,
) -> sys::FjStatus;
type Render = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const sys::FjCudaContext,
    *const sys::FjFrame,
    *const sys::FjSubmission,
    *const sys::FjPreparedHostData,
    *const super::FjHostPreparation,
    sys::FjAbortCallback,
    *mut sys::FjErrorBuffer,
) -> sys::FjRenderOutcome;
type Complete = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const super::FjCallCompletionInput,
) -> super::FjCallCompletionResult;
type ScannerLut = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const crate::scanner_bridge::FjScannerLutInput,
    *mut crate::scanner_bridge::FjScannerLutView,
    *mut crate::scanner_bridge::FjScannerLutFailure,
    *mut sys::FjErrorBuffer,
) -> sys::FjStatus;
type Gaussian = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const crate::optics_bridge::FjGaussianKernelInput,
    *mut crate::optics_bridge::FjGaussianKernelView,
    *mut crate::optics_bridge::FjGaussianKernelFailure,
    *mut sys::FjErrorBuffer,
) -> sys::FjStatus;
type Select = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const sys::FjCudaContext,
    *const sys::FjSubmission,
    *mut sys::FjSubmission,
    *mut sys::FjErrorBuffer,
) -> sys::FjStatus;
type PolicyEntry = unsafe extern "C" fn(
    *const super::FjPreparedState,
    *const sys::FjExtent,
    *mut super::FjPreparedFramePolicy,
    *mut sys::FjErrorBuffer,
) -> sys::FjStatus;
type RenderEntry = unsafe extern "C" fn(
    *const crate::asset_bridge::FjAssets,
    *const super::FjPreparedState,
    super::FjCudaCallAccess,
    *const sys::FjFrame,
    *const super::FjPreparedFramePolicy,
    super::FjSubmissionAccess,
    sys::FjAbortCallback,
    *mut sys::FjErrorBuffer,
) -> super::FjCallCompletionResult;
type UnavailableEntry = unsafe extern "C" fn(
    *const crate::asset_bridge::FjAssets,
    *const super::FjPreparedState,
    u32,
    super::FjCudaCallAccess,
    *const sys::FjFrame,
    sys::FjAbortCallback,
    *mut sys::FjErrorBuffer,
) -> super::FjCallCompletionResult;

const _: PolicyEntry = super::fj_legacy_prepared_state_frame_policy;
const _: RenderEntry = super::fj_legacy_prepared_state_render;
const _: UnavailableEntry = super::fj_legacy_prepared_unavailable;

const _: fn(&super::FjCudaCallAccess) = |access| {
    let _: &Option<Precheck> = &access.precheck;
    let _: &Option<Admit> = &access.admit;
    let _: &Option<Release> = &access.release;
    let _: &Option<Inspect> = &access.inspect;
    let _: &Option<Preview> = &access.preview;
    let _: &Option<FrameDescriptors> = &access.frame_descriptors;
    let _: &Option<Render> = &access.render;
    let _: &Option<Complete> = &access.complete;
};

const _: fn(&super::FjHostPreparation, &super::FjSubmissionAccess) = |host, submission| {
    let _: &Option<ScannerLut> = &host.scanner_lut;
    let _: &Option<Gaussian> = &host.gaussian;
    let _: &Option<Select> = &submission.select;
};

static FACTS: &[usize] = &[
    size_of::<super::FjPreparedFramePolicy>(),
    align_of::<super::FjPreparedFramePolicy>(),
    offset_of!(super::FjPreparedFramePolicy, geometry),
    offset_of!(super::FjPreparedFramePolicy, optics),
    offset_of!(super::FjPreparedFramePolicy, recipe_hash),
    size_of::<super::FjGrainCoefficients>(),
    align_of::<super::FjGrainCoefficients>(),
    offset_of!(super::FjGrainCoefficients, flags),
    offset_of!(super::FjGrainCoefficients, debug_view),
    offset_of!(super::FjGrainCoefficients, sublayer_count),
    offset_of!(super::FjGrainCoefficients, micro_structure),
    offset_of!(super::FjGrainCoefficients, clump_temporal_mix),
    offset_of!(super::FjGrainCoefficients, size_mix_scale),
    offset_of!(super::FjGrainCoefficients, amplitude),
    offset_of!(super::FjGrainCoefficients, chroma_mix),
    offset_of!(super::FjGrainCoefficients, chroma_shared_weight),
    offset_of!(super::FjGrainCoefficients, chroma_independent_weight),
    offset_of!(super::FjGrainCoefficients, particle_density_min_cmy),
    offset_of!(super::FjGrainCoefficients, uniformity_cmy),
    offset_of!(super::FjGrainCoefficients, layer_axis_block_prefix_max),
    size_of::<super::FjFrameMathInput>(),
    align_of::<super::FjFrameMathInput>(),
    offset_of!(super::FjFrameMathInput, dir),
    offset_of!(super::FjFrameMathInput, grain),
    offset_of!(super::FjFrameMathInput, effects),
    offset_of!(super::FjFrameMathInput, grain_coefficients),
    offset_of!(super::FjFrameMathInput, film_dust),
    offset_of!(super::FjFrameMathInput, gate_dust),
    offset_of!(super::FjFrameMathInput, film_scratch),
    offset_of!(super::FjFrameMathInput, gate_scratch),
    size_of::<super::FjNativeFrameDescriptors>(),
    align_of::<super::FjNativeFrameDescriptors>(),
    offset_of!(super::FjNativeFrameDescriptors, spatial_dir),
    offset_of!(super::FjNativeFrameDescriptors, grain),
    offset_of!(super::FjNativeFrameDescriptors, effects),
    size_of::<super::FjHostPreparation>(),
    align_of::<super::FjHostPreparation>(),
    offset_of!(super::FjHostPreparation, user),
    offset_of!(super::FjHostPreparation, scanner_lut),
    offset_of!(super::FjHostPreparation, gaussian),
    size_of::<super::FjCallCompletionInput>(),
    align_of::<super::FjCallCompletionInput>(),
    offset_of!(super::FjCallCompletionInput, outcome),
    offset_of!(super::FjCallCompletionInput, diagnostic),
    offset_of!(super::FjCallCompletionInput, render_outcome_valid),
    offset_of!(super::FjCallCompletionInput, abort_failed),
    offset_of!(super::FjCallCompletionInput, recovery_scope_entered),
    offset_of!(super::FjCallCompletionInput, has_inspected_context),
    offset_of!(super::FjCallCompletionInput, inspected_context),
    offset_of!(super::FjCallCompletionInput, force_fatal),
    size_of::<super::FjCallCompletionResult>(),
    align_of::<super::FjCallCompletionResult>(),
    offset_of!(super::FjCallCompletionResult, outcome),
    offset_of!(super::FjCallCompletionResult, fatal),
    size_of::<super::FjCudaCallAccess>(),
    align_of::<super::FjCudaCallAccess>(),
    offset_of!(super::FjCudaCallAccess, user),
    offset_of!(super::FjCudaCallAccess, precheck),
    offset_of!(super::FjCudaCallAccess, admit),
    offset_of!(super::FjCudaCallAccess, release),
    offset_of!(super::FjCudaCallAccess, inspect),
    offset_of!(super::FjCudaCallAccess, preview),
    offset_of!(super::FjCudaCallAccess, frame_descriptors),
    offset_of!(super::FjCudaCallAccess, render),
    offset_of!(super::FjCudaCallAccess, complete),
    size_of::<super::FjSubmissionAccess>(),
    align_of::<super::FjSubmissionAccess>(),
    offset_of!(super::FjSubmissionAccess, instance_token),
    offset_of!(super::FjSubmissionAccess, frame_token),
    offset_of!(super::FjSubmissionAccess, user),
    offset_of!(super::FjSubmissionAccess, select),
    0, // FJ_UNAVAILABLE_STATE
    1, // FJ_UNAVAILABLE_INSTANCE
];

const _: fn(
    &super::FjGrainCoefficients,
    &super::FjCallCompletionInput,
    &super::FjCallCompletionResult,
) = |grain, completion, result| {
    let _: u32 = grain.flags;
    let _: u32 = grain.debug_view;
    let _: usize = grain.sublayer_count;
    let _: [f32; 2] = grain.micro_structure;
    let _: [f32; 3] = grain.particle_density_min_cmy;
    let _: [f32; 3] = grain.uniformity_cmy;
    let _: [[f32; 16]; 3] = grain.layer_axis_block_prefix_max;
    let _: u32 = completion.render_outcome_valid;
    let _: u32 = completion.abort_failed;
    let _: u32 = completion.recovery_scope_entered;
    let _: u32 = completion.has_inspected_context;
    let _: u32 = completion.force_fatal;
    let _: u32 = result.fatal;
};

/// Test-support facts, compared against the reviewed C/C++ x86_64 constants.
///
/// # Safety
/// `count` must point to one live writable `usize` for the duration of the call.
#[unsafe(no_mangle)]
pub(super) unsafe extern "C" fn fj_test_prepared_call_abi_facts(count: *mut usize) -> *const usize {
    // SAFETY: This test-only ABI getter is called with one live writable count.
    unsafe { count.write(FACTS.len()) };
    FACTS.as_ptr()
}
