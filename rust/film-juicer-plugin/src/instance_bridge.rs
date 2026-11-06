//! Private instance C edge; only this module admits foreign capture records.

#![expect(
    clippy::result_large_err,
    reason = "The instance boundary carries bounded inline capture diagnostics through nonallocating failure cleanup."
)]

use std::fmt::{self, Write};
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use film_juicer_core::state::{self, CaptureFailure, CaptureFailureKind};

use crate::asset_bridge::{Failure as PreparationFailure, FjAssets};
use crate::cuda::sys::{
    FjErrorBuffer, FjStatus, FjStringView, FJ_API_NONE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE,
    FJ_STATUS_INTERNAL_FAILURE,
};
use crate::instance::Instance;
use crate::preparation::{self, FjPreparationParams, FjPreparedState};

pub(crate) struct FjInstance {
    instance: Arc<Instance>,
}
pub(crate) struct FjHostWrite {
    write: Option<state::HostWrite>,
    instance: Arc<Instance>,
}
#[repr(C)]
pub(crate) struct FjCaptureStart {
    capture: u64,
    kind: u32,
}
#[repr(C)]
pub(crate) struct FjSnapshotStart {
    write_revision: u64,
    kind: u32,
}
#[repr(C)]
pub(crate) struct FjAdmission {
    state: *mut FjPreparedState,
    input_identity: u64,
    build: u64,
    kind: u32,
    route: u32,
}

#[expect(
    clippy::large_enum_variant,
    reason = "The foreign edge preserves the inline capture failure without a new allocation during unwind."
)]
enum Failure {
    Input(&'static str),
    State(state::Error),
    Preparation(PreparationFailure),
}
impl From<state::Error> for Failure {
    fn from(error: state::Error) -> Self {
        Self::State(error)
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => f.write_str(message),
            Self::State(error) => error.fmt(f),
            Self::Preparation(error) => error.fmt(f),
        }
    }
}
fn state_category(error: &state::Error) -> u32 {
    match error {
        state::Error::InvalidCaptureTicket | state::Error::InvalidCaptureOperation => {
            FJ_STATUS_UNSUPPORTED_INPUT
        }
        state::Error::CaptureFailed(failure)
            if failure.kind() == CaptureFailureKind::Allocation =>
        {
            FJ_STATUS_ALLOCATION_FAILURE
        }
        state::Error::Preparation(error) => preparation::failure_category(error),
        state::Error::Poisoned => FJ_STATUS_INTERNAL_FAILURE,
        _ => FJ_STATUS_PREPARATION_FAILURE,
    }
}
impl Failure {
    fn category(&self) -> u32 {
        match self {
            Self::Input(_) => FJ_STATUS_UNSUPPORTED_INPUT,
            Self::State(error) => state_category(error),
            Self::Preparation(error) => error.category(),
        }
    }
}
fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}
fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}

// The foreign diagnostic writer is bounded and performs no allocation, including
// capture-guard failure and malformed consuming host-write completion.
struct Diagnostic {
    bytes: [u8; 2048],
    len: usize,
}
impl Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let count = text.len().min(self.bytes.len() - self.len);
        self.bytes[self.len..self.len + count].copy_from_slice(&text.as_bytes()[..count]);
        self.len += count;
        Ok(())
    }
}
/// # Safety
/// The optional aligned record and its nonempty backing are initialized, live,
/// exclusive and disjoint from every other input/output for this call.
unsafe fn diagnostic_valid(error: *mut FjErrorBuffer) -> bool {
    if error.is_null() {
        return true;
    }
    if !aligned(error) {
        return false;
    }
    // SAFETY: The caller authorizes this initialized record, including zero capacity.
    unsafe {
        (*error).length = 0;
        let capacity = (*error).capacity;
        if capacity == 0 {
            return true;
        }
        let data = (*error).data;
        if data.is_null()
            || capacity > isize::MAX as usize
            || (data as usize).checked_add(capacity).is_none()
        {
            return false;
        }
        data.write(0);
    }
    true
}
/// # Safety
/// diagnostic_valid admitted the caller's exclusive record/backing extent.
unsafe fn write_diagnostic(error: *mut FjErrorBuffer, message: impl fmt::Display) {
    if error.is_null() {
        return;
    }
    // SAFETY: Admitted record; capacity zero leaves the data field unread.
    unsafe {
        let capacity = (*error).capacity;
        if capacity == 0 {
            return;
        }
        let mut diagnostic = Diagnostic {
            bytes: [0; 2048],
            len: 0,
        };
        let _ = write!(&mut diagnostic, "{message}");
        let count = diagnostic.len.min(capacity - 1);
        std::ptr::copy_nonoverlapping(diagnostic.bytes.as_ptr(), (*error).data.cast(), count);
        (*error).data.add(count).write(0);
        (*error).length = count;
    }
}
/// # Safety
/// Every input is caller-authorized live storage; diagnostics obey the contract
/// above. Consumed owners captured by operation must drop even on protocol rejection.
unsafe fn run_with(
    instance: Option<&Instance>,
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<(), Failure>,
    on_panic: impl Fn(),
) -> FjStatus {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller authorizes the diagnostic record/backing.
        if !unsafe { diagnostic_valid(error) } {
            drop(operation);
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        match catch_unwind(AssertUnwindSafe(operation)) {
            Ok(Ok(())) => status(FJ_STATUS_SUCCESS),
            Ok(Err(failure)) => {
                // SAFETY: diagnostic_valid admitted this exclusive extent.
                unsafe { write_diagnostic(error, &failure) };
                status(failure.category())
            }
            Err(_) => {
                if let Some(instance) = instance {
                    instance.block();
                }
                on_panic();
                // SAFETY: diagnostic_valid admitted this exclusive extent.
                unsafe { write_diagnostic(error, "instance boundary panic") };
                status(FJ_STATUS_INTERNAL_FAILURE)
            }
        }
    }));
    result.unwrap_or_else(|_| {
        if let Some(instance) = instance {
            instance.block();
        }
        on_panic();
        status(FJ_STATUS_INTERNAL_FAILURE)
    })
}
/// # Safety
/// Same initialized, exclusive foreign storage contract as run_with.
unsafe fn run(
    instance: Option<&Instance>,
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Forward the caller's exact extents and operation lifetime.
    unsafe { run_with(instance, error, operation, || {}) }
}
/// # Safety
/// A matching live instance handle remains borrowed for this synchronous call.
unsafe fn instance<'a>(owner: *const FjInstance) -> Option<&'a Instance> {
    if !aligned(owner) {
        return None;
    }
    // SAFETY: The foreign caller authorizes the live matching handle.
    Some(unsafe { &(*owner).instance })
}
/// # Safety
/// The pointer authorizes exclusive aligned initialized storage for T.
unsafe fn clear<T>(out: *mut T) -> bool {
    if !aligned(out) {
        return false;
    }
    // SAFETY: All output C records/scalars/pointers are zero-valid.
    unsafe { out.write_bytes(0, 1) };
    true
}
fn update_tag(update: state::Update) -> u32 {
    match update {
        state::Update::Applied => 1,
        state::Update::Ignored => 2,
        state::Update::Deferred => 3,
    }
}
fn failure_kind(tag: u32) -> Result<CaptureFailureKind, Failure> {
    match tag {
        1 => Ok(CaptureFailureKind::InvalidControls),
        2 => Ok(CaptureFailureKind::Host),
        3 => Ok(CaptureFailureKind::Allocation),
        _ => Err(Failure::Input("invalid capture failure kind")),
    }
}
/// # Safety
/// The input authorizes its initialized immutable byte extent until return.
unsafe fn diagnostic_text<'a>(text: FjStringView) -> Result<&'a str, Failure> {
    if text.count == 0 {
        return Ok("");
    }
    if text.data.is_null()
        || text.count > isize::MAX as usize
        || (text.data as usize).checked_add(text.count).is_none()
    {
        return Err(Failure::Input("invalid capture diagnostic extent"));
    }
    // SAFETY: The caller authorizes this initialized byte extent.
    let bytes = unsafe { std::slice::from_raw_parts(text.data.cast::<u8>(), text.count) };
    std::str::from_utf8(bytes).map_err(|_| Failure::Input("capture diagnostic is not UTF-8"))
}

macro_rules! export {
    ($name:ident($($arg:ident: $ty:ty),*) $body:block) => {
        /// # Safety
        /// The instance protocol in juicer_legacy_api.h authorizes live handles,
        /// borrowed initialized inputs and disjoint exclusive outputs/diagnostics.
        #[unsafe(no_mangle)]
        pub(crate) unsafe extern "C" fn $name($($arg: $ty),*) -> FjStatus {
            // SAFETY: The caller's documented protocol authorizes all raw accesses.
            unsafe { $body }
        }
    };
}
export!(fj_legacy_instance_create(out: *mut *mut FjInstance, error: *mut FjErrorBuffer) {
    let valid = clear(out);
    run(None, error, || {
        if !valid { return Err(Failure::Input("invalid instance output")); }
        out.write(Box::into_raw(Box::new(FjInstance { instance: Arc::new(Instance::new()) })));
        Ok(())
    })
});
export!(fj_legacy_instance_begin_capture(owner: *mut FjInstance, mode: u32, out: *mut FjCaptureStart, error: *mut FjErrorBuffer) {
    let valid = clear(out);
    let instance = instance(owner);
    run(instance, error, || {
        let instance = instance.ok_or(Failure::Input("invalid instance"))?;
        if !valid { return Err(Failure::Input("invalid capture output")); }
        let mode = match mode { 0 => state::CaptureMode::Initial, 1 => state::CaptureMode::Change,
            2 => state::CaptureMode::Refresh, _ => return Err(Failure::Input("invalid capture mode")) };
        match instance.begin_capture(mode)? {
            state::CaptureStart::Started { capture } => out.write(FjCaptureStart { capture, kind: 1 }),
            state::CaptureStart::NotStarted => (*out).kind = 2,
        }
        Ok(())
    })
});
export!(fj_legacy_instance_capture_is_current(owner: *const FjInstance, capture: u64, out_current: *mut u32, error: *mut FjErrorBuffer) {
    let valid = clear(out_current);
    let instance = instance(owner);
    run(instance, error, || {
        let instance = instance.ok_or(Failure::Input("invalid instance"))?;
        if !valid { return Err(Failure::Input("invalid current output")); }
        out_current.write(u32::from(instance.capture_is_current(capture)?)); Ok(())
    })
});
export!(fj_legacy_instance_begin_host_write(owner: *mut FjInstance, capture: u64, out: *mut *mut FjHostWrite, out_start: *mut u32, error: *mut FjErrorBuffer) {
    let valid = clear(out);
    let start_valid = clear(out_start);
    let instance = instance(owner);
    run(instance, error, || {
        let instance = instance.ok_or(Failure::Input("invalid instance"))?;
        if !valid || !start_valid { return Err(Failure::Input("invalid host write output")); }
        match instance.begin_host_write(capture)? {
            state::HostWriteStart::Started(write) => {
                let owner = FjHostWrite { write: Some(write), instance: Arc::clone(&(*owner).instance) };
                out.write(Box::into_raw(Box::new(owner))); out_start.write(1);
            }
            state::HostWriteStart::Ignored => out_start.write(2),
        }
        Ok(())
    })
});
export!(fj_legacy_host_write_finish(owner: *mut FjHostWrite, failure: u32, diagnostic: FjStringView, out_refresh: *mut u32, error: *mut FjErrorBuffer) {
    let valid = clear(out_refresh);
    if !aligned(owner) {
        return run(None, error, || Err(Failure::Input("invalid host write owner")));
    }
    // Consume before validating any other finish argument. Rejected operations
    // drop the still-armed safe guard, recording bounded Host failure.
    let mut owner = Box::from_raw(owner);
    let instance = Arc::clone(&owner.instance);
    run(Some(&instance), error, move || {
        if !valid { return Err(Failure::Input("invalid refresh output")); }
        let text = diagnostic_text(diagnostic)?;
        let outcome = if failure == 0 {
            if !text.is_empty() { return Err(Failure::Input("successful write has diagnostic")); }
            Ok(())
        } else { Err(CaptureFailure::new(failure_kind(failure)?, text)) };
        let end = owner.write.take().expect("live host write").finish(outcome)?;
        out_refresh.write(u32::from(end == state::WriteEnd::RefreshRequired)); Ok(())
    })
});
export!(fj_legacy_instance_begin_snapshot(owner: *mut FjInstance, capture: u64, out: *mut FjSnapshotStart, error: *mut FjErrorBuffer) {
    let valid = clear(out);
    let instance = instance(owner);
    run(instance, error, || {
        let instance = instance.ok_or(Failure::Input("invalid instance"))?;
        if !valid { return Err(Failure::Input("invalid snapshot output")); }
        match instance.begin_snapshot(capture)? {
            state::SnapshotStart::Ready { write_revision } => out.write(FjSnapshotStart { write_revision, kind: 1 }),
            state::SnapshotStart::Deferred => (*out).kind = 2,
            state::SnapshotStart::Ignored => (*out).kind = 3,
        }
        Ok(())
    })
});
export!(fj_legacy_instance_install(owner: *mut FjInstance, capture: u64, write_revision: u64, params: *const FjPreparationParams, input_identity: u64, out_update: *mut u32, error: *mut FjErrorBuffer) {
    let valid = clear(out_update);
    let instance = instance(owner);
    run(instance, error, || {
        let instance = instance.ok_or(Failure::Input("invalid instance"))?;
        if !valid || !aligned(params) { return Err(Failure::Input("invalid capture input/output")); }
        // Protocol rejection precedes snapshot parsing without mutation. This
        // query does not grant a write or replace install's atomic recheck.
        if !instance.capture_is_current(capture)? { out_update.write(2); return Ok(()); }
        match preparation::params(&*params) {
            Ok(params) => out_update.write(update_tag(instance.install(capture, write_revision, params, input_identity)?)),
            Err(error) => {
                let kind = if error.category() == FJ_STATUS_ALLOCATION_FAILURE { CaptureFailureKind::Allocation }
                    else { CaptureFailureKind::InvalidControls };
                let mut diagnostic = Diagnostic { bytes: [0; 2048], len: 0 };
                let _ = write!(&mut diagnostic, "{error}");
                let mut len = diagnostic.len;
                while std::str::from_utf8(&diagnostic.bytes[..len]).is_err() { len -= 1; }
                let text = std::str::from_utf8(&diagnostic.bytes[..len]).expect("UTF-8 preview");
                let update = instance.fail_snapshot(capture, write_revision, CaptureFailure::new(kind, text))?;
                out_update.write(update_tag(update));
                if update == state::Update::Applied { return Err(Failure::Preparation(error)); }
            }
        }
        Ok(())
    })
});
export!(fj_legacy_instance_invalidate(owner: *mut FjInstance, capture: u64, failure: u32, diagnostic: FjStringView, out_update: *mut u32, error: *mut FjErrorBuffer) {
    let valid = clear(out_update);
    let instance = instance(owner);
    run(instance, error, || {
        let instance = instance.ok_or(Failure::Input("invalid instance"))?;
        if !valid { return Err(Failure::Input("invalid update output")); }
        let failure = CaptureFailure::new(failure_kind(failure)?, diagnostic_text(diagnostic)?);
        out_update.write(update_tag(instance.invalidate(capture, failure)?)); Ok(())
    })
});
export!(fj_legacy_instance_admit(owner: *mut FjInstance, assets: *const FjAssets, out: *mut FjAdmission, error: *mut FjErrorBuffer) {
    let valid = clear(out);
    let instance = instance(owner);
    run_with(instance, error, || {
        let instance = instance.ok_or(Failure::Input("invalid instance"))?;
        if !valid { return Err(Failure::Input("invalid admission output")); }
        let assets = preparation::assets(assets).map_err(Failure::Preparation)?;
        match instance.admit(assets) {
            Ok(admission) => {
                let input_identity = admission.input_identity();
                let build = admission.build();
                let route = admission.state().development().route() as u32;
                let state = Box::into_raw(Box::new(FjPreparedState::from_state(admission.into_state())));
                out.write(FjAdmission { state, input_identity, build, kind: 1, route }); Ok(())
            }
            Err(state::Error::Uninitialized) => { (*out).kind = 2; Ok(()) }
            Err(state::Error::RefreshRequired) => { (*out).kind = 7; Ok(()) }
            Err(state::Error::Preparation(error)) => {
                (*out).kind = if error.is_panic() { 6 } else { 5 };
                Err(Failure::Preparation(match error {
                    film_juicer_core::recipe::PrepareError::Asset(error) => PreparationFailure::Asset(error),
                    error => PreparationFailure::Preparation(error),
                }))
            }
            Err(error) => {
                (*out).kind = match &error { state::Error::Capturing => 3, state::Error::CaptureFailed(_) => 4,
                    state::Error::Preparation(_) => 5, state::Error::Poisoned => 6, _ => 0 };
                Err(Failure::State(error))
            }
        }
    }, || {
        if valid {
            let state = (*out).state;
            clear(out);
            (*out).kind = 6;
            if !state.is_null() { drop(Box::from_raw(state)); }
        }
    })
});
export!(fj_legacy_instance_destroy(owner: *mut FjInstance, error: *mut FjErrorBuffer) {
    if !aligned(owner) { return run(None, error, || Err(Failure::Input("invalid instance owner"))); }
    let owner = Box::from_raw(owner);
    run(None, error, move || {
        // Callback/write exclusion is a caller prerequisite. Admitted states
        // own their independent Arc and neither close nor drop calls host/CUDA.
        match Arc::try_unwrap(owner.instance) {
            Ok(instance) => instance.close(),
            Err(instance) => { instance.block(); }
        }
        Ok(())
    })
});

/// # Safety
/// count authorizes exclusive initialized usize storage; static facts are borrowed.
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_test_instance_abi_facts(count: *mut usize) -> *const usize {
    if !aligned(count) {
        return std::ptr::null();
    }
    use std::mem::offset_of;
    static FACTS: &[usize] = &[
        size_of::<FjCaptureStart>(),
        align_of::<FjCaptureStart>(),
        offset_of!(FjCaptureStart, capture),
        offset_of!(FjCaptureStart, kind),
        size_of::<FjSnapshotStart>(),
        align_of::<FjSnapshotStart>(),
        offset_of!(FjSnapshotStart, write_revision),
        offset_of!(FjSnapshotStart, kind),
        size_of::<FjAdmission>(),
        align_of::<FjAdmission>(),
        offset_of!(FjAdmission, state),
        offset_of!(FjAdmission, input_identity),
        offset_of!(FjAdmission, build),
        offset_of!(FjAdmission, kind),
        offset_of!(FjAdmission, route),
    ];
    // SAFETY: The caller authorizes the checked exclusive scalar slot.
    unsafe { count.write(FACTS.len()) };
    FACTS.as_ptr()
}

#[cfg(feature = "test-support")]
pub(crate) struct FjInstanceObservation {
    observation: state::Observation,
}
#[cfg(feature = "test-support")]
#[repr(C)]
pub(crate) struct FjInstanceObservationView {
    pending_kind: u32,
    failure_kind: u32,
    issued: u64,
    write_revision: u64,
    outstanding_writes: u64,
    input_identity: u64,
    published_identity: u64,
    build: u64,
    backing_identity: usize,
    diagnostic: FjStringView,
    captured: FjPreparationParams,
    published: preparation::FjPreparedStateView,
}

#[cfg(feature = "test-support")]
fn captured_projection(params: &film_juicer_core::recipe::Params) -> FjPreparationParams {
    fn diffusion(
        params: &film_juicer_core::optics::DiffusionParams,
    ) -> crate::optics_bridge::FjDiffusionParams {
        crate::optics_bridge::FjDiffusionParams {
            active: u32::from(params.active),
            family: params.family,
            strength: params.strength,
            spatial_scale: params.spatial_scale,
            halo_warmth: params.halo_warmth,
            core_intensity: params.core_intensity,
            core_size: params.core_size,
            halo_intensity: params.halo_intensity,
            halo_size: params.halo_size,
            bloom_intensity: params.bloom_intensity,
            bloom_size: params.bloom_size,
        }
    }
    let dir = &params.dir_couplers;
    let grain = &params.grain;
    let halation = &params.scatter_halation;
    FjPreparationParams {
        print_exposure: params.print_exposure,
        print_preflash_exposure: params.print_preflash_exposure,
        print_gamma_factor: params.print_gamma_factor,
        preflash_m_filter_cc: params.preflash_m_filter_cc,
        preflash_y_filter_cc: params.preflash_y_filter_cc,
        print_shadow_compensation_factor: params.print_shadow_compensation_factor,
        print_shadow_compensation_density: params.print_shadow_compensation_density,
        print_shadow_compensation_transition: params.print_shadow_compensation_transition,
        glare_percent: params.glare_percent,
        glare_roughness: params.glare_roughness,
        glare_blur_sigma_px: params.glare_blur_sigma_px,
        camera_exposure_compensation_ev: params.camera_exposure_compensation_ev,
        camera_film_format_long_edge_mm: params.camera_film_format_long_edge_mm,
        film_gamma_factor: params.film_gamma_factor,
        scanner_lens_blur_sigma_px: params.scanner_lens_blur_sigma_px,
        scanner_black_level: params.scanner_black_level,
        scanner_white_level: params.scanner_white_level,
        gate_weave_amount: params.gate_weave_amount,
        scanner_unsharp_mask: params.scanner_unsharp_mask,
        print_ui_ymc_cc: params.print_ui_ymc_cc,
        camera_filter_uv: params.camera_filter_uv,
        camera_filter_ir: params.camera_filter_ir,
        spectral_upsampling_mode: params.spectral_upsampling_mode,
        ref_ill: params.ref_ill,
        enl_ill: params.enl_ill,
        normalize_print_exposure: params.normalize_print_exposure,
        print_exposure_compensation: params.print_exposure_compensation,
        input_color_space: params.input_color_space,
        input_cctf_decoding: params.input_cctf_decoding,
        input_compression_enabled: params.input_compression_enabled,
        hanatos2025_adaptation_window: params.hanatos2025_adaptation_window,
        hanatos2025_adaptation_surface: params.hanatos2025_adaptation_surface,
        camera_auto_exposure_enabled: params.camera_auto_exposure_enabled,
        camera_metering_method: params.camera_metering_method,
        scanner_black_correction: params.scanner_black_correction,
        scanner_white_correction: params.scanner_white_correction,
        scanner_lut_resolution: params.scanner_lut_resolution,
        output_color_space: params.output_color_space,
        output_cctf_encoding: params.output_cctf_encoding,
        output_gamut_compression_enabled: params.output_gamut_compression_enabled,
        film_dust_amount: params.film_dust_amount,
        film_scratch_amount: params.film_scratch_amount,
        gate_dust_amount: params.gate_dust_amount,
        gate_scratch_amount: params.gate_scratch_amount,
        camera_diffusion: diffusion(&params.camera_diffusion),
        enlarger_diffusion: diffusion(&params.enlarger_diffusion),
        film_profile_key: FjStringView {
            data: params.film_profile_key.as_ptr().cast(),
            count: params.film_profile_key.len(),
        },
        print_profile_key: FjStringView {
            data: params.print_profile_key.as_ptr().cast(),
            count: params.print_profile_key.len(),
        },
        scan_route: params.scan_route as u32,
        glare_active: u32::from(params.glare_active),
        camera_filter_override: u32::from(params.camera_filter_override),
        scatter_halation: preparation::FjPreparationHalation {
            active: u32::from(halation.active),
            scatter_amount: halation.scatter_amount,
            scatter_spatial_scale: halation.scatter_spatial_scale,
            halation_amount: halation.halation_amount,
            halation_spatial_scale: halation.halation_spatial_scale,
        },
        dir_couplers: crate::film_recipe_bridge::FjFilmDirParams {
            active: u32::from(dir.active),
            amount: dir.amount,
            inhibition_same_layer: dir.inhibition_same_layer,
            inhibition_interlayer: dir.inhibition_interlayer,
            diffusion_size_um: dir.diffusion_size_um,
            diffusion_tail_um: dir.diffusion_tail_um,
            diffusion_tail_weight: dir.diffusion_tail_weight,
            gamma_use_stock: u32::from(dir.gamma_use_stock),
            gamma_same_layer_rgb: dir.gamma_same_layer_rgb,
            gamma_interlayer_r_to_gb: dir.gamma_interlayer_r_to_gb,
            gamma_interlayer_g_to_rb: dir.gamma_interlayer_g_to_rb,
            gamma_interlayer_b_to_rg: dir.gamma_interlayer_b_to_rg,
            langmuir_donor_k_rgb: dir.langmuir_donor_k_rgb,
            langmuir_receiver_k_rgb: dir.langmuir_receiver_k_rgb,
        },
        grain: crate::film_recipe_bridge::FjVisualGrainParams {
            active: u32::from(grain.active),
            sublayers_active: u32::from(grain.sublayers_active),
            particle_area_um2: grain.particle_area_um2,
            particle_scale_cmy: grain.particle_scale_cmy,
            particle_scale_layers: grain.particle_scale_layers,
            particle_density_min_cmy: grain.particle_density_min_cmy,
            uniformity_cmy: grain.uniformity_cmy,
            correlation_sigma_px: grain.correlation_sigma_px,
            dye_cloud_blur_um: grain.dye_cloud_blur_um,
            micro_structure: grain.micro_structure,
            sublayer_count: grain.sublayer_count,
            amplitude: grain.amplitude,
            chroma_mix: grain.chroma_mix,
            chroma_shared_weight: grain.chroma_shared_weight,
            chroma_independent_weight: grain.chroma_independent_weight,
            coarse_weight: grain.coarse_weight,
            mid_weight: grain.mid_weight,
            size_mix_scale: grain.size_mix_scale,
            clump_temporal_mix: grain.clump_temporal_mix,
            clump_morph_period_sec: grain.clump_morph_period_sec,
            debug_view: grain.debug_view,
        },
    }
}

#[cfg(feature = "test-support")]
export!(fj_test_instance_observation_acquire(owner: *const FjInstance, out: *mut *mut FjInstanceObservation, error: *mut FjErrorBuffer) {
    let valid = clear(out);
    let instance = instance(owner);
    run(instance, error, || {
        let instance = instance.ok_or(Failure::Input("invalid observed instance"))?;
        if !valid { return Err(Failure::Input("invalid observation output")); }
        let observation = instance.observe()?;
        out.write(Box::into_raw(Box::new(FjInstanceObservation { observation })));
        Ok(())
    })
});
#[cfg(feature = "test-support")]
export!(fj_test_instance_observation_view(owner: *const FjInstanceObservation, out: *mut FjInstanceObservationView, error: *mut FjErrorBuffer) {
    let valid = clear(out);
    run(None, error, || {
        if !valid || !aligned(owner) { return Err(Failure::Input("invalid observation view")); }
        let observation = &(*owner).observation;
        (*out).issued = observation.issued;
        (*out).write_revision = observation.write_revision;
        (*out).outstanding_writes = observation.outstanding_writes;
        use state::ObservedPending;
        (*out).pending_kind = match &observation.pending {
            ObservedPending::Uninitialized => 0, ObservedPending::Capturing => 1,
            ObservedPending::AwaitingWrites => 2, ObservedPending::NeedsRefresh => 3,
            ObservedPending::Valid { params, identity } => {
                (*out).input_identity = *identity;
                (*out).captured = captured_projection(params); 4
            }
            ObservedPending::Failed(failure) => {
                (*out).failure_kind = match failure.kind() {
                    CaptureFailureKind::InvalidControls => 1, CaptureFailureKind::Host => 2, CaptureFailureKind::Allocation => 3,
                };
                (*out).diagnostic = FjStringView { data: failure.diagnostic().as_ptr().cast(), count: failure.diagnostic().len() }; 5
            }
            ObservedPending::Blocked => 6,
        };
        if let Some(admitted) = &observation.published {
            (*out).published_identity = admitted.input_identity();
            (*out).build = admitted.build();
            (*out).backing_identity = admitted.state() as *const _ as usize;
            let held = FjPreparedState::from_state(Arc::clone(admitted.shared_state()));
            let status = preparation::state_view(&held, &mut (*out).published, std::ptr::null_mut());
            if status.category != FJ_STATUS_SUCCESS { return Err(Failure::Input("cannot project observed state")); }
        }
        Ok(())
    })
});
#[cfg(feature = "test-support")]
export!(fj_test_instance_observation_release(owner: *mut FjInstanceObservation, error: *mut FjErrorBuffer) {
    if !aligned(owner) { return run(None, error, || Err(Failure::Input("invalid observation owner"))); }
    let owner = Box::from_raw(owner);
    run(None, error, move || { drop(owner); Ok(()) })
});

#[cfg(feature = "test-support")]
mod hooks {
    use std::cell::Cell;
    type Callback = Option<unsafe extern "C" fn(*mut std::ffi::c_void)>;
    thread_local! {
        static BEFORE: Cell<Callback> = const { Cell::new(None) };
        static AFTER: Cell<Callback> = const { Cell::new(None) };
        static CONTEXT: Cell<*mut std::ffi::c_void> = const { Cell::new(std::ptr::null_mut()) };
    }
    fn before() {
        if let Some(callback) = BEFORE.get() {
            // SAFETY: The fixture caller keeps the callback/context live on this
            // thread through clearing; callback exceptions never cross this edge.
            unsafe { callback(CONTEXT.get()) };
        }
    }
    fn after() {
        if let Some(callback) = AFTER.get() {
            // SAFETY: Same bounded registration and exception-exclusion contract.
            unsafe { callback(CONTEXT.get()) };
        }
    }
    /// # Safety
    /// Callbacks/context remain live on this thread through clearing. Callbacks
    /// contain native exceptions and are never retained in production safe state.
    #[unsafe(no_mangle)]
    unsafe extern "C" fn fj_test_instance_prepare_hooks(
        before_hook: Callback,
        after_hook: Callback,
        context: *mut std::ffi::c_void,
    ) {
        BEFORE.set(before_hook);
        AFTER.set(after_hook);
        CONTEXT.set(context);
        crate::instance::testing::set_prepare_hooks(
            before_hook.map(|_| before as fn()),
            after_hook.map(|_| after as fn()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;
    use crate::test_cases::RowCases;

    #[test]
    fn instance_layout_and_consuming_write_protocol() {
        assert_eq!(
            (
                size_of::<FjCaptureStart>(),
                align_of::<FjCaptureStart>(),
                offset_of!(FjCaptureStart, capture),
                offset_of!(FjCaptureStart, kind)
            ),
            (16, 8, 0, 8)
        );
        assert_eq!(
            (
                size_of::<FjSnapshotStart>(),
                align_of::<FjSnapshotStart>(),
                offset_of!(FjSnapshotStart, write_revision),
                offset_of!(FjSnapshotStart, kind)
            ),
            (16, 8, 0, 8)
        );
        assert_eq!(
            (
                size_of::<FjAdmission>(),
                align_of::<FjAdmission>(),
                offset_of!(FjAdmission, state),
                offset_of!(FjAdmission, input_identity),
                offset_of!(FjAdmission, build),
                offset_of!(FjAdmission, kind),
                offset_of!(FjAdmission, route)
            ),
            (32, 8, 0, 8, 16, 24, 28)
        );
        let cases = [
            "success-refresh",
            "malformed-consumes",
            "failed-write",
            "poisoned-guard",
        ];
        let mut rows = RowCases::new("instance-write-protocol", cases);
        for id in cases {
            let row = rows.case(id);
            // SAFETY: Local initialized C records are disjoint and every owning
            // handle is consumed once; no raw owner is used after its finish.
            unsafe {
                let mut owner = std::ptr::null_mut();
                assert_eq!(
                    fj_legacy_instance_create(&mut owner, std::ptr::null_mut()).category,
                    FJ_STATUS_SUCCESS
                );
                let mut start = FjCaptureStart {
                    capture: 99,
                    kind: 99,
                };
                assert_eq!(
                    fj_legacy_instance_begin_capture(owner, 0, &mut start, std::ptr::null_mut())
                        .category,
                    FJ_STATUS_SUCCESS
                );
                let first = start.capture;
                let mut write = std::ptr::null_mut();
                let mut kind = 99;
                assert_eq!(
                    fj_legacy_instance_begin_host_write(
                        owner,
                        first,
                        &mut write,
                        &mut kind,
                        std::ptr::null_mut()
                    )
                    .category,
                    FJ_STATUS_SUCCESS
                );
                assert_eq!(kind, 1);
                assert!(!write.is_null());
                assert_eq!(
                    fj_legacy_instance_begin_capture(owner, 1, &mut start, std::ptr::null_mut())
                        .category,
                    FJ_STATUS_SUCCESS
                );
                let mut snapshot = FjSnapshotStart {
                    write_revision: 99,
                    kind: 99,
                };
                assert_eq!(
                    fj_legacy_instance_begin_snapshot(
                        owner,
                        start.capture,
                        &mut snapshot,
                        std::ptr::null_mut()
                    )
                    .category,
                    FJ_STATUS_SUCCESS
                );
                assert_eq!((snapshot.kind, snapshot.write_revision), (2, 0));
                let mut refresh = 99;
                let failure = if id == "failed-write" {
                    3
                } else if id == "malformed-consumes" {
                    99
                } else {
                    0
                };
                if id == "poisoned-guard" {
                    (*owner).instance.block();
                }
                let result = fj_legacy_host_write_finish(
                    write,
                    failure,
                    FjStringView {
                        data: std::ptr::null(),
                        count: 0,
                    },
                    &mut refresh,
                    std::ptr::null_mut(),
                );
                match id {
                    "malformed-consumes" => {
                        assert_eq!((result.category, refresh), (FJ_STATUS_UNSUPPORTED_INPUT, 0));
                        assert!(
                            matches!((*owner).instance.admit(&film_juicer_core::assets::Assets::new(std::path::PathBuf::new())), Err(state::Error::CaptureFailed(failure)) if failure.kind() == CaptureFailureKind::Host)
                        );
                    }
                    "failed-write" => {
                        assert_eq!((result.category, refresh), (FJ_STATUS_SUCCESS, 0))
                    }
                    "poisoned-guard" => {
                        assert_eq!((result.category, refresh), (FJ_STATUS_INTERNAL_FAILURE, 0))
                    }
                    _ => {
                        assert_eq!((result.category, refresh), (FJ_STATUS_SUCCESS, 1));
                        assert_eq!(
                            fj_legacy_instance_begin_capture(
                                owner,
                                2,
                                &mut start,
                                std::ptr::null_mut()
                            )
                            .category,
                            FJ_STATUS_SUCCESS
                        );
                        assert_eq!(start.kind, 1);
                    }
                }
                assert_eq!(
                    fj_legacy_instance_destroy(owner, std::ptr::null_mut()).category,
                    FJ_STATUS_SUCCESS
                );
            }
            row.passed();
        }
        rows.finish();
    }
}
