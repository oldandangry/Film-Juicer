//! SF_TEMP_BRIDGE_SPATIAL_OPTICS: checked fixed completed recipes; remove S4.E.
use std::fmt::{self, Write};
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use film_juicer_core::optics;
use crate::cuda::sys::{
    FjStatus, FjErrorBuffer, FJ_API_NONE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT,
    FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE,
};
pub(crate) const FJ_OPTICS_REASON_NONE: u32 = 0;
pub(crate) const FJ_OPTICS_REASON_RAW: u32 = 1;
pub(crate) const FJ_OPTICS_REASON_DIFFUSION: u32 = 2;
pub(crate) const FJ_OPTICS_REASON_LENS_BLUR: u32 = 3;
pub(crate) const FJ_OPTICS_REASON_HALATION: u32 = 4;
pub(crate) const FJ_OPTICS_REASON_PANIC: u32 = 5;
pub(crate) const FJ_OPTICS_COMPONENT_NONE: u32 = 0;
pub(crate) const FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION: u32 = 1;
pub(crate) const FJ_OPTICS_COMPONENT_CAMERA_LENS_BLUR: u32 = 2;
pub(crate) const FJ_OPTICS_COMPONENT_SCATTER_HALATION: u32 = 3;
pub(crate) const FJ_OPTICS_COMPONENT_ENLARGER_DIFFUSION: u32 = 4;
pub(crate) const FJ_OPTICS_FIELD_NONE: u32 = 0;
pub(crate) const FJ_OPTICS_FIELD_FAMILY: u32 = 1;
pub(crate) const FJ_OPTICS_FIELD_STRENGTH: u32 = 2;
pub(crate) const FJ_OPTICS_FIELD_SPATIAL_SCALE: u32 = 3;
pub(crate) const FJ_OPTICS_FIELD_HALO_WARMTH: u32 = 4;
pub(crate) const FJ_OPTICS_FIELD_CORE_INTENSITY: u32 = 5;
pub(crate) const FJ_OPTICS_FIELD_CORE_SIZE: u32 = 6;
pub(crate) const FJ_OPTICS_FIELD_HALO_INTENSITY: u32 = 7;
pub(crate) const FJ_OPTICS_FIELD_HALO_SIZE: u32 = 8;
pub(crate) const FJ_OPTICS_FIELD_BLOOM_INTENSITY: u32 = 9;
pub(crate) const FJ_OPTICS_FIELD_BLOOM_SIZE: u32 = 10;
pub(crate) const FJ_OPTICS_FIELD_SCATTER_FRACTION: u32 = 11;
pub(crate) const FJ_OPTICS_FIELD_GROUP_WEIGHTS: u32 = 12;
pub(crate) const FJ_OPTICS_FIELD_GROUP_CENTERS: u32 = 13;
pub(crate) const FJ_OPTICS_FIELD_RESOLVED_HASH: u32 = 14;
pub(crate) const FJ_OPTICS_FIELD_SIGMA_UM: u32 = 15;
pub(crate) const FJ_OPTICS_FIELD_TOTAL_STRENGTH: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjDiffusionParams {
    pub(crate) active: u32,
    pub(crate) family: u32,
    pub(crate) strength: f64,
    pub(crate) spatial_scale: f64,
    pub(crate) halo_warmth: f64,
    pub(crate) core_intensity: f64,
    pub(crate) core_size: f64,
    pub(crate) halo_intensity: f64,
    pub(crate) halo_size: f64,
    pub(crate) bloom_intensity: f64,
    pub(crate) bloom_size: f64,
}

const _: () = assert!(size_of::<FjDiffusionParams>() == 80 && align_of::<FjDiffusionParams>() == 8);
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjHalationParams {
    pub(crate) active: u32,
    pub(crate) scatter_amount: f32,
    pub(crate) scatter_spatial_scale: f32,
    pub(crate) halation_amount: f32,
    pub(crate) halation_spatial_scale: f32,
    pub(crate) first_sigma_um: [f32; 3],
    pub(crate) primary_amount: [f32; 3],
}

const _: () = assert!(size_of::<FjHalationParams>() == 44 && align_of::<FjHalationParams>() == 4);
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjSpatialOpticsParams {
    pub(crate) camera_diffusion: FjDiffusionParams,
    pub(crate) camera_lens_blur_um: f32,
    pub(crate) scatter_halation: FjHalationParams,
    pub(crate) enlarger_diffusion: FjDiffusionParams,
    pub(crate) print_domain: u32,
}

const _: () =
    assert!(size_of::<FjSpatialOpticsParams>() == 216 && align_of::<FjSpatialOpticsParams>() == 8);
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjDiffusionRecipe {
    pub(crate) domain: u32,
    pub(crate) active: u32,
    pub(crate) family: u32,
    pub(crate) scatter_fraction: f64,
    pub(crate) group_weights: [f64; 3],
    pub(crate) group_lambda_um: [f64; 3],
    pub(crate) warmth: f64,
    pub(crate) spatial_scale: f64,
    pub(crate) resolved_hash: u64,
    pub(crate) hash: u64,
}

const _: () =
    assert!(size_of::<FjDiffusionRecipe>() == 104 && align_of::<FjDiffusionRecipe>() == 8);
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjLensBlurRecipe {
    pub(crate) sigma_um: f32,
    pub(crate) hash: u64,
}

const _: () = assert!(size_of::<FjLensBlurRecipe>() == 16 && align_of::<FjLensBlurRecipe>() == 8);
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjHalationRecipe {
    pub(crate) scatter_active: u32,
    pub(crate) back_reflection_active: u32,
    pub(crate) scatter_amount: f32,
    pub(crate) scatter_spatial_scale: f32,
    pub(crate) halation_spatial_scale: f32,
    pub(crate) first_sigma_um: [f32; 3],
    pub(crate) total_strength: [f32; 3],
    pub(crate) hash: u64,
}

const _: () = assert!(size_of::<FjHalationRecipe>() == 56 && align_of::<FjHalationRecipe>() == 8);
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct FjSpatialOpticsRecipe {
    pub(crate) camera_diffusion: FjDiffusionRecipe,
    pub(crate) camera_lens_blur: FjLensBlurRecipe,
    pub(crate) scatter_halation: FjHalationRecipe,
    pub(crate) enlarger_diffusion: FjDiffusionRecipe,
    pub(crate) hash: u64,
}

const _: () =
    assert!(size_of::<FjSpatialOpticsRecipe>() == 288 && align_of::<FjSpatialOpticsRecipe>() == 8);
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjSpatialOpticsFailure {
    pub(crate) reason: u32,
    pub(crate) component: u32,
    pub(crate) field: u32,
    pub(crate) channel: u32,
}

const _: () =
    assert!(size_of::<FjSpatialOpticsFailure>() == 16 && align_of::<FjSpatialOpticsFailure>() == 4);

fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}
fn valid_extent<T>(pointer: *const T, count: usize) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && count.checked_mul(size_of::<T>()).is_some_and(|bytes| {
            bytes <= isize::MAX as usize && (pointer as usize).checked_add(bytes).is_some()
        })
}
struct Diagnostic {
    record: *mut FjErrorBuffer,
    data: *mut u8,
    capacity: usize,
    length: usize,
}
impl Diagnostic {
    /// # Safety
    /// Optional initialized live record and positive backing are exclusive and disjoint.
    unsafe fn admit(record: *mut FjErrorBuffer) -> Result<Self, ()> {
        let mut writer = Self {
            record,
            data: std::ptr::null_mut(),
            capacity: 0,
            length: 0,
        };
        if record.is_null() {
            return Ok(writer);
        }
        if !valid_extent(record, 1) {
            return Err(());
        }
        // SAFETY: The admitted record is live; zero capacity never reads backing.
        unsafe {
            (*record).length = 0;
            writer.capacity = (*record).capacity;
            if writer.capacity != 0 {
                let data = (*record).data.cast::<u8>();
                if !valid_extent(data, writer.capacity) {
                    return Err(());
                }
                writer.data = data;
                data.write(0);
            }
        }
        Ok(writer)
    }
}
impl Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.capacity != 0 {
            let count = text.len().min(self.capacity - 1 - self.length);
            // SAFETY: Admission and the bounded length maintain the exclusive backing extent.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data.add(self.length), count);
                self.length += count;
                self.data.add(self.length).write(0);
            }
        }
        if !self.record.is_null() {
            // SAFETY: The independently admitted record remains exclusively live.
            unsafe {
                (*self.record).length = self.length;
            }
        }
        Ok(())
    }
}
fn none() -> FjSpatialOpticsFailure {
    FjSpatialOpticsFailure {
        reason: FJ_OPTICS_REASON_NONE,
        component: FJ_OPTICS_COMPONENT_NONE,
        field: FJ_OPTICS_FIELD_NONE,
        channel: 3,
    }
}
struct Failure {
    category: u32,
    details: FjSpatialOpticsFailure,
}
impl Failure {
    fn raw() -> Self {
        Self {
            category: FJ_STATUS_UNSUPPORTED_INPUT,
            details: FjSpatialOpticsFailure {
                reason: FJ_OPTICS_REASON_RAW,
                ..none()
            },
        }
    }
    fn panic() -> Self {
        Self {
            category: FJ_STATUS_INTERNAL_FAILURE,
            details: FjSpatialOpticsFailure {
                reason: FJ_OPTICS_REASON_PANIC,
                ..none()
            },
        }
    }
    fn core(error: optics::Error) -> Self {
        let (reason, component) = match error.component() {
            optics::Component::CameraDiffusion => (
                FJ_OPTICS_REASON_DIFFUSION,
                FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION,
            ),
            optics::Component::CameraLensBlur => (
                FJ_OPTICS_REASON_LENS_BLUR,
                FJ_OPTICS_COMPONENT_CAMERA_LENS_BLUR,
            ),
            optics::Component::ScatterHalation => (
                FJ_OPTICS_REASON_HALATION,
                FJ_OPTICS_COMPONENT_SCATTER_HALATION,
            ),
            optics::Component::EnlargerDiffusion => (
                FJ_OPTICS_REASON_DIFFUSION,
                FJ_OPTICS_COMPONENT_ENLARGER_DIFFUSION,
            ),
        };
        let field = match error.field() {
            optics::Field::Family => FJ_OPTICS_FIELD_FAMILY,
            optics::Field::Strength => FJ_OPTICS_FIELD_STRENGTH,
            optics::Field::SpatialScale => FJ_OPTICS_FIELD_SPATIAL_SCALE,
            optics::Field::HaloWarmth => FJ_OPTICS_FIELD_HALO_WARMTH,
            optics::Field::CoreIntensity => FJ_OPTICS_FIELD_CORE_INTENSITY,
            optics::Field::CoreSize => FJ_OPTICS_FIELD_CORE_SIZE,
            optics::Field::HaloIntensity => FJ_OPTICS_FIELD_HALO_INTENSITY,
            optics::Field::HaloSize => FJ_OPTICS_FIELD_HALO_SIZE,
            optics::Field::BloomIntensity => FJ_OPTICS_FIELD_BLOOM_INTENSITY,
            optics::Field::BloomSize => FJ_OPTICS_FIELD_BLOOM_SIZE,
            optics::Field::ScatterFraction => FJ_OPTICS_FIELD_SCATTER_FRACTION,
            optics::Field::GroupWeights => FJ_OPTICS_FIELD_GROUP_WEIGHTS,
            optics::Field::GroupCenters => FJ_OPTICS_FIELD_GROUP_CENTERS,
            optics::Field::ResolvedHash => FJ_OPTICS_FIELD_RESOLVED_HASH,
            optics::Field::SigmaUm => FJ_OPTICS_FIELD_SIGMA_UM,
            optics::Field::TotalStrength => FJ_OPTICS_FIELD_TOTAL_STRENGTH,
        };
        Self {
            category: FJ_STATUS_PREPARATION_FAILURE,
            details: FjSpatialOpticsFailure {
                reason,
                component,
                field,
                channel: error.channel().map_or(3, |v| v as u32),
            },
        }
    }
}
fn diffusion_params(v: &FjDiffusionParams) -> optics::DiffusionParams {
    optics::DiffusionParams {
        active: v.active != 0,
        family: v.family,
        strength: v.strength,
        spatial_scale: v.spatial_scale,
        halo_warmth: v.halo_warmth,
        core_intensity: v.core_intensity,
        core_size: v.core_size,
        halo_intensity: v.halo_intensity,
        halo_size: v.halo_size,
        bloom_intensity: v.bloom_intensity,
        bloom_size: v.bloom_size,
    }
}
fn project_diffusion(v: &optics::Diffusion) -> FjDiffusionRecipe {
    FjDiffusionRecipe {
        domain: match v.domain() {
            optics::Domain::FilmLinearExposure => 0,
            optics::Domain::PrintLinearExposure => 1,
        },
        active: u32::from(v.active()),
        family: match v.family() {
            optics::Family::Glimmerglass => 0,
            optics::Family::BlackProMist => 1,
            optics::Family::ProMist => 2,
            optics::Family::Cinebloom => 3,
        },
        scatter_fraction: v.scatter_fraction(),
        group_weights: *v.group_weights(),
        group_lambda_um: *v.group_lambda_um(),
        warmth: v.warmth(),
        spatial_scale: v.spatial_scale(),
        resolved_hash: v.resolved_hash(),
        hash: v.hash(),
    }
}
fn project(v: &optics::Recipe) -> FjSpatialOpticsRecipe {
    let lens = v.camera_lens_blur();
    let halation = v.scatter_halation();
    FjSpatialOpticsRecipe {
        camera_diffusion: project_diffusion(v.camera_diffusion()),
        camera_lens_blur: FjLensBlurRecipe {
            sigma_um: lens.sigma_um(),
            hash: lens.hash(),
        },
        scatter_halation: FjHalationRecipe {
            scatter_active: u32::from(halation.scatter_active()),
            back_reflection_active: u32::from(halation.back_reflection_active()),
            scatter_amount: halation.scatter_amount(),
            scatter_spatial_scale: halation.scatter_spatial_scale(),
            halation_spatial_scale: halation.halation_spatial_scale(),
            first_sigma_um: *halation.first_sigma_um(),
            total_strength: *halation.total_strength(),
            hash: halation.hash(),
        },
        enlarger_diffusion: project_diffusion(v.enlarger_diffusion()),
        hash: v.hash(),
    }
}
/// # Safety
/// Required records are initialized, live and aligned. Exclusive output/failure
/// and optional diagnostic record/backing are mutually disjoint and disjoint
/// from immutable input through synchronous return. No borrow escapes. Numeric
/// pointer admission does not establish those allocation/lifetime obligations.
pub(crate) unsafe fn prepare(
    params: *const FjSpatialOpticsParams,
    out: *mut FjSpatialOpticsRecipe,
    failure: *mut FjSpatialOpticsFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    // SAFETY: Clear each independently admitted numerical record before other admission.
    unsafe {
        if valid_out {
            out.write(FjSpatialOpticsRecipe::default());
        }
        if valid_failure {
            failure.write(none());
        }
    }
    // SAFETY: The caller authorizes optional diagnostics under the same disjoint contract.
    let diagnostic = unsafe { Diagnostic::admit(error) };
    let diagnostic_valid = diagnostic.is_ok();
    let mut diagnostic = diagnostic.ok();
    let result = catch_unwind(AssertUnwindSafe(|| {
        #[cfg(feature = "test-support")]
        let fault = crate::film_recipe_bridge::faults::take(9);
        if !valid_out || !valid_failure || !diagnostic_valid || !valid_extent(params, 1) {
            return Err(Failure::raw());
        }
        // SAFETY: Numeric admission precedes the caller-authorized initialized immutable borrow.
        let input = unsafe { &*params };
        for flag in [
            input.camera_diffusion.active,
            input.scatter_halation.active,
            input.enlarger_diffusion.active,
            input.print_domain,
        ] {
            if flag > 1 {
                return Err(Failure::raw());
            }
        }
        let h = &input.scatter_halation;
        let params = optics::Params {
            camera_diffusion: diffusion_params(&input.camera_diffusion),
            camera_lens_blur_um: input.camera_lens_blur_um,
            scatter_halation: optics::HalationParams {
                active: h.active != 0,
                scatter_amount: h.scatter_amount,
                scatter_spatial_scale: h.scatter_spatial_scale,
                halation_amount: h.halation_amount,
                halation_spatial_scale: h.halation_spatial_scale,
                first_sigma_um: h.first_sigma_um,
                primary_amount: h.primary_amount,
            },
            enlarger_diffusion: diffusion_params(&input.enlarger_diffusion),
            print_domain: input.print_domain != 0,
        };
        #[cfg(feature = "test-support")]
        match fault {
            1 => panic!("spatial optics fixture panic"),
            2 => optics::test_support::zero_next_resolved_hash(),
            _ => {}
        }
        optics::Recipe::new(params)
            .map(|recipe| project(&recipe))
            .map_err(Failure::core)
    }))
    .unwrap_or_else(|_| Err(Failure::panic()));
    match result {
        Ok(recipe) => {
            // SAFETY: Complete output commits only after all required record admissions.
            unsafe {
                out.write(recipe);
                failure.write(none());
            }
            status(FJ_STATUS_SUCCESS)
        }
        Err(rejection) => {
            if let Some(writer) = &mut diagnostic {
                let v = rejection.details;
                let _ = write!(
                    writer,
                    "spatial optics reason={} component={} field={} channel={}",
                    v.reason, v.component, v.field, v.channel
                );
            }
            if valid_failure {
                // SAFETY: Independently admitted failure storage receives the complete closed envelope.
                unsafe {
                    failure.write(rejection.details);
                }
            }
            status(rejection.category)
        }
    }
}
/// # Safety
/// The checked operation's live, initialized, aligned and disjoint record contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_spatial_optics_prepare(
    params: *const FjSpatialOpticsParams,
    out: *mut FjSpatialOpticsRecipe,
    failure: *mut FjSpatialOpticsFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Identical checked synchronous operation with no retained pointer.
    unsafe { prepare(params, out, failure, error) }
}
const _: unsafe extern "C" fn(
    *const FjSpatialOpticsParams,
    *mut FjSpatialOpticsRecipe,
    *mut FjSpatialOpticsFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_spatial_optics_prepare;

// SF_TEMP_BRIDGE_GAUSSIAN_KERNEL: miss-local owned coefficients; remove S4.E.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjGaussianKernelInput {
    pub(crate) radius: i32,
    pub(crate) sigma_px: f32,
}
#[repr(C)]
pub(crate) struct FjGaussianKernelOwner {
    _private: [u8; 0],
}
pub(crate) struct HostGaussianCandidate {
    kernel: optics::GaussianKernel,
}
impl HostGaussianCandidate {
    fn new(kernel: optics::GaussianKernel) -> Self {
        #[cfg(feature = "test-support")]
        gaussian_faults::LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self { kernel }
    }
}
impl Drop for HostGaussianCandidate {
    fn drop(&mut self) {
        #[cfg(feature = "test-support")]
        gaussian_faults::LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjGaussianKernelView {
    pub(crate) radius: i32,
    pub(crate) sigma_px: f32,
    pub(crate) weights: crate::cuda::sys::FjFloatSpan,
}
impl FjGaussianKernelView {
    pub(crate) fn empty() -> Self {
        Self {
            radius: 0,
            sigma_px: 0.0,
            weights: crate::cuda::sys::FjFloatSpan {
                data: std::ptr::null(),
                count: 0,
            },
        }
    }
}

/// Clear a callback view when the preceding acquire failed.
///
/// # Safety
/// A nonnull `out` points to one writable, aligned, live view record.
pub(crate) unsafe fn clear_host_gaussian_view(out: *mut FjGaussianKernelView) {
    if valid_extent(out, 1) {
        // SAFETY: The caller supplied the writable view record admitted above.
        unsafe { out.write(FjGaussianKernelView::empty()) };
    }
}

/// Clear a callback output and initialize its failure reason.
///
/// # Safety
/// Each nonnull record points to writable, aligned, live storage.
pub(crate) unsafe fn fail_host_gaussian_view(
    out: *mut FjGaussianKernelView,
    failure: *mut FjGaussianKernelFailure,
    reason: u32,
) {
    // SAFETY: The callback's caller supplies the initialized output records.
    unsafe { clear_host_gaussian_view(out) };
    if valid_extent(failure, 1) {
        // SAFETY: The failure record was admitted above.
        unsafe { failure.write(FjGaussianKernelFailure { reason }) };
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjGaussianKernelFailure {
    pub(crate) reason: u32,
}
#[derive(Clone, Copy)]
enum GaussianRejection {
    Raw,
    Core(optics::GaussianKernelError),
    Panic,
    // The callback candidate-vector reserve has a distinct message but uses
    // the existing allocation reason/category in the Gaussian failure record.
    BackingAllocation,
}
impl GaussianRejection {
    fn fields(self) -> (u32, u32, &'static str) {
        match self {
            Self::Raw => (
                1,
                FJ_STATUS_UNSUPPORTED_INPUT,
                "invalid Gaussian kernel arguments",
            ),
            Self::Core(optics::GaussianKernelError::Radius) => (
                2,
                FJ_STATUS_PREPARATION_FAILURE,
                "Gaussian kernel radius outside supported range",
            ),
            Self::Core(optics::GaussianKernelError::Sigma) => (
                3,
                FJ_STATUS_PREPARATION_FAILURE,
                "invalid Gaussian kernel sigma",
            ),
            Self::Core(optics::GaussianKernelError::Allocation) => (
                4,
                crate::cuda::sys::FJ_STATUS_ALLOCATION_FAILURE,
                "Gaussian kernel allocation failed",
            ),
            Self::Panic => (
                5,
                FJ_STATUS_INTERNAL_FAILURE,
                "Gaussian kernel preparation panicked",
            ),
            Self::BackingAllocation => (
                4,
                crate::cuda::sys::FJ_STATUS_ALLOCATION_FAILURE,
                "Gaussian kernel candidate storage allocation failed",
            ),
        }
    }
}
/// # Safety
/// Live initialized writable outputs/diagnostics are exclusive and mutually
/// disjoint, including input storage. Admission does not prove memory provenance.
unsafe fn gaussian_call<T: Copy>(
    out: *mut T,
    cleared: T,
    failure: *mut FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
    release: bool,
    body: impl FnOnce() -> Result<T, GaussianRejection>,
) -> FjStatus {
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    // SAFETY: Clear each independently admitted writable peer before any other admission.
    unsafe {
        if valid_out {
            out.write(cleared);
        }
        if valid_failure {
            failure.write(FjGaussianKernelFailure { reason: 0 });
        }
    }
    // SAFETY: Optional diagnostic storage follows the same live/disjoint contract.
    let mut diagnostic = unsafe { Diagnostic::admit(error) };
    let result = if !valid_out || !valid_failure || diagnostic.is_err() {
        Err(GaussianRejection::Raw)
    } else {
        catch_unwind(AssertUnwindSafe(body)).unwrap_or(Err(GaussianRejection::Panic))
    };
    match result {
        Ok(value) => {
            // SAFETY: Successful admission established live exclusive output storage.
            unsafe {
                out.write(value);
            }
            status(FJ_STATUS_SUCCESS)
        }
        Err(rejection) => {
            let (reason, category, text) = rejection.fields();
            if valid_failure {
                // SAFETY: Failure storage was independently admitted before any input access.
                unsafe {
                    failure.write(FjGaussianKernelFailure { reason });
                }
            }
            if let Ok(writer) = &mut diagnostic {
                let text = if release && reason == 5 {
                    "Gaussian kernel release panicked"
                } else {
                    text
                };
                let _ = writer.write_str(text);
            }
            status(category)
        }
    }
}
fn gaussian_fault(operation: usize) -> u32 {
    #[cfg(feature = "test-support")]
    {
        gaussian_faults::take(operation)
    }
    #[cfg(not(feature = "test-support"))]
    {
        let _ = operation;
        0
    }
}
pub(crate) fn take_host_gaussian_acquire_fault() -> u32 {
    let fault = gaussian_fault(0);
    #[cfg(feature = "test-support")]
    gaussian_faults::ACQUIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    fault
}
fn gaussian_panic(fault: u32) {
    #[cfg(feature = "test-support")]
    if fault == 1 {
        panic!("Gaussian kernel fixture panic");
    }
    #[cfg(not(feature = "test-support"))]
    let _ = fault;
}

fn construct_gaussian(
    radius: u32,
    sigma_px: f32,
    fault: u32,
) -> Result<optics::GaussianKernel, GaussianRejection> {
    #[cfg(feature = "test-support")]
    optics::test_support::reject_gaussian_reservation(fault == 2);
    #[cfg(not(feature = "test-support"))]
    let _ = fault;
    optics::GaussianKernel::new(radius, sigma_px).map_err(GaussianRejection::Core)
}

fn view_gaussian(
    kernel: &optics::GaussianKernel,
    fault: u32,
) -> Result<FjGaussianKernelView, GaussianRejection> {
    gaussian_panic(fault);
    let view = FjGaussianKernelView {
        radius: kernel.radius() as i32,
        sigma_px: kernel.sigma_px(),
        weights: crate::cuda::sys::FjFloatSpan {
            data: kernel.weights().as_ptr(),
            count: kernel.weights().len(),
        },
    };
    #[cfg(feature = "test-support")]
    {
        Ok(gaussian_faults::corrupt(view, fault))
    }
    #[cfg(not(feature = "test-support"))]
    {
        let _ = fault;
        Ok(view)
    }
}

/// Build and retain one Gaussian candidate for a native cache miss.
///
/// # Safety
/// The input and outputs follow the initialized, live, disjoint contract of
/// `gaussian_acquire` and `gaussian_view`.
pub(crate) unsafe fn prepare_host_gaussian(
    input: *const FjGaussianKernelInput,
    out: *mut FjGaussianKernelView,
    failure: *mut FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
    candidates: &mut Vec<HostGaussianCandidate>,
    acquire_fault: u32,
) -> FjStatus {
    // Admit the actual callback output and its peers before running optics math.
    // The acquire fault/count were already taken by the callback entry point.
    // SAFETY: `gaussian_call` independently clears and validates caller records.
    let admission = unsafe {
        gaussian_call(
            out,
            FjGaussianKernelView::empty(),
            failure,
            error,
            false,
            || Ok(FjGaussianKernelView::empty()),
        )
    };
    if admission.category != FJ_STATUS_SUCCESS {
        return admission;
    }
    let mut acquired = 0u32;
    // Preserve the legacy acquire/view fault boundary and its separate admission.
    // SAFETY: Admission and acquire use the existing checked Gaussian ABI helper;
    // `input`, output, failure and error storage are supplied by the native callback.
    let acquire_status = unsafe {
        gaussian_call(&mut acquired, 0, failure, error, false, || {
            if !valid_extent(input, 1) {
                return Err(GaussianRejection::Raw);
            }
            gaussian_panic(acquire_fault);
            let input = &*input;
            candidates
                .try_reserve(1)
                .map_err(|_| GaussianRejection::BackingAllocation)?;
            let kernel = construct_gaussian(input.radius as u32, input.sigma_px, acquire_fault)?;
            candidates.push(HostGaussianCandidate::new(kernel));
            Ok(0)
        })
    };
    if acquire_status.category != FJ_STATUS_SUCCESS {
        return acquire_status;
    }
    let view_fault = gaussian_fault(1);
    let candidate = candidates.last().expect("successful Gaussian acquire");
    // SAFETY: Candidate storage remains owned through native return; the view
    // call independently admits and clears its output/failure/error peers.
    unsafe {
        gaussian_call(
            out,
            FjGaussianKernelView::empty(),
            failure,
            error,
            false,
            || view_gaussian(&candidate.kernel, view_fault),
        )
    }
}

/// # Safety
/// Live initialized input and disjoint exclusive output/failure/diagnostics stay
/// valid through return. Successful ownership transfers to matching release.
pub(crate) unsafe fn gaussian_acquire(
    input: *const FjGaussianKernelInput,
    out: *mut *mut FjGaussianKernelOwner,
    failure: *mut FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = take_host_gaussian_acquire_fault();
    // SAFETY: Common admission clears valid peers before borrowing the immutable input.
    unsafe {
        gaussian_call(out, std::ptr::null_mut(), failure, error, false, || {
            if !valid_extent(input, 1) {
                return Err(GaussianRejection::Raw);
            }
            gaussian_panic(fault);
            let input = &*input;
            let kernel = construct_gaussian(input.radius as u32, input.sigma_px, fault)?;
            // Box allocation retains the established infallible allocator-abort limitation.
            let owner = Box::new(kernel);
            #[cfg(feature = "test-support")]
            gaussian_faults::LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Box::into_raw(owner).cast())
        })
    }
}
/// # Safety
/// Borrow one legitimate live owner excluding release. Exclusive initialized
/// output/failure/diagnostic storage is disjoint; all spans expire at release.
pub(crate) unsafe fn gaussian_view(
    owner: *const FjGaussianKernelOwner,
    out: *mut FjGaussianKernelView,
    failure: *mut FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = gaussian_fault(1);
    // SAFETY: Common admission precedes owner borrow; provenance is the caller's obligation.
    unsafe {
        gaussian_call(
            out,
            FjGaussianKernelView::empty(),
            failure,
            error,
            false,
            || {
                if !valid_extent(owner.cast::<optics::GaussianKernel>(), 1) {
                    return Err(GaussianRejection::Raw);
                }
                let kernel = &*owner.cast::<optics::GaussianKernel>();
                view_gaussian(kernel, fault)
            },
        )
    }
}
/// # Safety
/// NULL or one matching live owner transfers exactly once, excluding views.
/// Malformed diagnostics and contained unwind still consume an admitted owner.
pub(crate) unsafe fn gaussian_release(
    owner: *mut FjGaussianKernelOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let fault = gaussian_fault(2);
    let valid_owner = owner.is_null() || valid_extent(owner.cast::<optics::GaussianKernel>(), 1);
    let owned = if !owner.is_null() && valid_owner {
        #[cfg(feature = "test-support")]
        gaussian_faults::LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Caller transfers this exact Box once before fallible diagnostic handling.
        Some(unsafe { Box::from_raw(owner.cast::<optics::GaussianKernel>()) })
    } else {
        None
    };
    let mut out = 0u32;
    let mut failure = FjGaussianKernelFailure { reason: 0 };
    // SAFETY: Local outputs and caller-authorized diagnostics satisfy common admission.
    // The owned closure drops its Box even when rejected before invocation.
    unsafe {
        gaussian_call(&mut out, 0, &mut failure, error, true, move || {
            if !valid_owner {
                return Err(GaussianRejection::Raw);
            }
            gaussian_panic(fault);
            drop(owned);
            Ok(0)
        })
    }
}
/// # Safety
/// Same initialized/disjoint input and ownership transfer contract as gaussian_acquire.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_gaussian_kernel_acquire(
    input: *const FjGaussianKernelInput,
    out: *mut *mut FjGaussianKernelOwner,
    failure: *mut FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical checked ownership contract.
    unsafe { gaussian_acquire(input, out, failure, error) }
}
/// # Safety
/// Same legitimate owner/borrow expiry/disjoint output contract as gaussian_view.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_gaussian_kernel_view(
    owner: *const FjGaussianKernelOwner,
    out: *mut FjGaussianKernelView,
    failure: *mut FjGaussianKernelFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical checked borrowing contract.
    unsafe { gaussian_view(owner, out, failure, error) }
}
/// # Safety
/// Same once-only matching ownership transfer contract as gaussian_release.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_gaussian_kernel_release(
    owner: *mut FjGaussianKernelOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical checked consuming contract.
    unsafe { gaussian_release(owner, error) }
}
#[cfg(feature = "test-support")]
pub(crate) mod gaussian_faults {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    pub(super) static LIVE: AtomicUsize = AtomicUsize::new(0);
    pub(super) static ACQUIRES: AtomicUsize = AtomicUsize::new(0);
    thread_local! { static FAULTS: Cell<[u32; 3]> = const { Cell::new([0; 3]) }; }
    pub(crate) fn arm(operation: u32, fault: u32) -> FjStatus {
        if operation > 2
            || fault
                > match operation {
                    0 => 2,
                    1 => 4,
                    _ => 1,
                }
        {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        FAULTS.with(|v| {
            let mut slots = v.get();
            slots[operation as usize] = fault;
            v.set(slots);
        });
        status(FJ_STATUS_SUCCESS)
    }
    pub(super) fn take(operation: usize) -> u32 {
        FAULTS.with(|v| {
            let mut slots = v.get();
            let fault = slots[operation];
            slots[operation] = 0;
            v.set(slots);
            fault
        })
    }
    pub(crate) fn counts() -> [usize; 2] {
        [
            ACQUIRES.load(Ordering::Relaxed),
            LIVE.load(Ordering::Relaxed),
        ]
    }
    pub(super) fn corrupt(mut view: FjGaussianKernelView, fault: u32) -> FjGaussianKernelView {
        match fault {
            2 => view.weights.count += 1,
            3 => view.sigma_px = f32::NAN,
            4 => view.weights.data = view.weights.data.cast::<u8>().wrapping_add(1).cast(),
            _ => {}
        }
        view
    }
}

const _: unsafe extern "C" fn(
    *const FjGaussianKernelInput,
    *mut *mut FjGaussianKernelOwner,
    *mut FjGaussianKernelFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_gaussian_kernel_acquire;
const _: unsafe extern "C" fn(
    *const FjGaussianKernelOwner,
    *mut FjGaussianKernelView,
    *mut FjGaussianKernelFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_gaussian_kernel_view;
const _: unsafe extern "C" fn(*mut FjGaussianKernelOwner, *mut FjErrorBuffer) -> FjStatus =
    fj_legacy_gaussian_kernel_release;

// SF_TEMP_BRIDGE_HALATION_FRAME: remove at the final native frame-preparation
// caller, no later than S6.D. State integration remains S4.E's responsibility.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjHalationFrameInput {
    pub(crate) recipe: FjHalationRecipe,
    pub(crate) pixel_size_um: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjHalationFrameFailure {
    pub(crate) reason: u32,
    pub(crate) field: u32,
    pub(crate) channel: u32,
    pub(crate) index: u32,
    pub(crate) sigma_px: f64,
}

fn halation_frame_none() -> FjHalationFrameFailure {
    FjHalationFrameFailure {
        reason: 0,
        field: 0,
        channel: 3,
        index: 3,
        sigma_px: 0.0,
    }
}
const HALATION_IDENTITY: crate::cuda::sys::FjHalationGaussian =
    crate::cuda::sys::FjHalationGaussian {
        kind: 0,
        radius: 0,
        fir_weights: [0.0; 19],
        feedforward: 0.0,
        feedback: [0.0; 3],
    };
fn empty_halation_frame() -> crate::cuda::sys::FjScatterHalation {
    crate::cuda::sys::FjScatterHalation {
        recipe_hash: 0,
        scatter_amount: 0.0,
        channels: [crate::cuda::sys::FjHalationChannel {
            core: HALATION_IDENTITY,
            tail: [HALATION_IDENTITY; 3],
            bounce: [HALATION_IDENTITY; 3],
            total_strength: 0.0,
        }; 3],
    }
}
fn halation_gaussian(v: &optics::HalationGaussian) -> crate::cuda::sys::FjHalationGaussian {
    crate::cuda::sys::FjHalationGaussian {
        kind: match v.kind() {
            optics::HalationGaussianKind::Identity => 0,
            optics::HalationGaussianKind::FirReflect => 1,
            optics::HalationGaussianKind::YvvReplicate => 2,
        },
        radius: v.radius(),
        fir_weights: *v.fir_weights(),
        feedforward: v.yvv()[0],
        feedback: [v.yvv()[1], v.yvv()[2], v.yvv()[3]],
    }
}
fn halation_frame_projection(v: &optics::HalationFrame) -> crate::cuda::sys::FjScatterHalation {
    crate::cuda::sys::FjScatterHalation {
        recipe_hash: v.recipe_hash(),
        scatter_amount: v.scatter_amount(),
        channels: std::array::from_fn(|channel| {
            let c = &v.channels()[channel];
            crate::cuda::sys::FjHalationChannel {
                core: halation_gaussian(c.core()),
                tail: std::array::from_fn(|i| halation_gaussian(&c.tail()[i])),
                bounce: std::array::from_fn(|i| halation_gaussian(&c.bounce()[i])),
                total_strength: c.total_strength(),
            }
        }),
    }
}

enum HalationFrameRejection {
    Raw,
    Core(optics::HalationFrameError),
    Panic,
}
impl HalationFrameRejection {
    fn fields(&self) -> (u32, FjHalationFrameFailure, &'static str) {
        let mut failure = halation_frame_none();
        let (category, text) = match *self {
            Self::Raw => {
                failure.reason = 1;
                (
                    FJ_STATUS_UNSUPPORTED_INPUT,
                    "invalid scatter/halation frame arguments",
                )
            }
            Self::Core(optics::HalationFrameError::PixelSize) => {
                failure.reason = 2;
                failure.field = 1;
                (
                    FJ_STATUS_PREPARATION_FAILURE,
                    "ResourceDescriptorMismatch component=scatter_halation field=pixel_size_um",
                )
            }
            Self::Core(optics::HalationFrameError::Gaussian {
                field,
                channel,
                index,
                sigma_px,
            }) => {
                failure.reason = 3;
                failure.channel = channel as u32;
                failure.index = index.map_or(3, |i| i as u32);
                failure.sigma_px = sigma_px;
                let (tag, text) = match field {
                    optics::HalationFrameField::ScatterCore => (
                        2,
                        "ResourceDescriptorMismatch component=scatter_halation field=scatter_core",
                    ),
                    optics::HalationFrameField::ScatterTail => (
                        3,
                        "ResourceDescriptorMismatch component=scatter_halation field=scatter_tail",
                    ),
                    optics::HalationFrameField::Bounce => (
                        4,
                        "ResourceDescriptorMismatch component=scatter_halation field=back_reflection_bounce",
                    ),
                };
                failure.field = tag;
                (FJ_STATUS_PREPARATION_FAILURE, text)
            }
            Self::Panic => {
                failure.reason = 4;
                (
                    FJ_STATUS_INTERNAL_FAILURE,
                    "scatter/halation frame preparation panicked",
                )
            }
        };
        (category, failure, text)
    }
}

/// # Safety
/// Independently numerically valid writable records authorize initialized/live,
/// aligned exclusive storage disjoint from all other records and backing.
unsafe fn clear_halation_frame(
    out: *mut crate::cuda::sys::FjScatterHalation,
    failure: *mut FjHalationFrameFailure,
) {
    // SAFETY: Each peer's numeric admission is independent; provenance is a caller obligation.
    unsafe {
        if valid_extent(out, 1) {
            out.write(empty_halation_frame());
        }
        if valid_extent(failure, 1) {
            failure.write(halation_frame_none());
        }
    }
}

/// # Safety
/// Same initialized/live/exclusive/disjoint obligations as halation_frame_prepare.
unsafe fn halation_frame_body(
    input: *const FjHalationFrameInput,
    out: *mut crate::cuda::sys::FjScatterHalation,
    failure: *mut FjHalationFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Clear each admitted peer before checking input, peers or diagnostic backing.
    unsafe {
        clear_halation_frame(out, failure);
    }
    // SAFETY: Optional initialized record and bounded backing have the entry's exclusive borrow.
    let mut diagnostic = unsafe { Diagnostic::admit(error) };
    let result = (|| -> Result<crate::cuda::sys::FjScatterHalation, HalationFrameRejection> {
        if !valid_extent(out, 1)
            || !valid_extent(failure, 1)
            || !valid_extent(input, 1)
            || diagnostic.is_err()
        {
            return Err(HalationFrameRejection::Raw);
        }
        // SAFETY: Numeric checks precede this caller-authorized live aligned input copy.
        let input = unsafe { input.read() };
        let h = input.recipe;
        if h.scatter_active > 1 || h.back_reflection_active > 1 {
            return Err(HalationFrameRejection::Raw);
        }
        #[cfg(feature = "test-support")]
        let fault = halation_frame_faults::take();
        let input_recipe = optics::HalationFrameInput {
            scatter_active: h.scatter_active != 0,
            back_reflection_active: h.back_reflection_active != 0,
            scatter_amount: h.scatter_amount,
            scatter_spatial_scale: h.scatter_spatial_scale,
            halation_spatial_scale: h.halation_spatial_scale,
            first_sigma_um: h.first_sigma_um,
            total_strength: h.total_strength,
            recipe_hash: h.hash,
        };
        let frame = checked_halation_frame(input_recipe, input.pixel_size_um, {
            #[cfg(feature = "test-support")]
            {
                fault
            }
            #[cfg(not(feature = "test-support"))]
            {
                0
            }
        })
        .map_err(HalationFrameRejection::Core)?;
        let wire = frame
            .as_ref()
            .map_or_else(empty_halation_frame, halation_frame_projection);
        #[cfg(feature = "test-support")]
        let wire = halation_frame_faults::corrupt(wire, fault);
        Ok(wire)
    })();
    match result {
        Ok(wire) => {
            // SAFETY: All required peer admission succeeded and completed value is non-owning.
            unsafe {
                out.write(wire);
            }
            status(FJ_STATUS_SUCCESS)
        }
        Err(rejection) => {
            let (category, details, text) = rejection.fields();
            if valid_extent(failure, 1) {
                // SAFETY: Independently authorized failure slot remains exclusive and live.
                unsafe {
                    failure.write(details);
                }
            }
            if let Ok(writer) = &mut diagnostic {
                let _ = writer.write_str(text);
            }
            status(category)
        }
    }
}

/// Shared checked Hanatos frame construction used by the raw fixed-record
/// adapter and the complete held-state projection. ABI corruption stays raw.
pub(crate) fn checked_halation_frame(
    recipe: optics::HalationFrameInput,
    pixel_size_um: f32,
    fault: u32,
) -> Result<Option<optics::HalationFrame>, optics::HalationFrameError> {
    #[cfg(feature = "test-support")]
    if fault == 1 {
        panic!("scatter/halation frame fixture panic");
    }
    let _ = fault;
    optics::HalationFrame::new(recipe, pixel_size_um)
}

/// # Safety
/// Required records are initialized/aligned/live. Writable records and optional
/// diagnostic backing are exclusive, mutually disjoint and disjoint from input.
/// Numeric checks do not prove arbitrary pointer provenance. No borrow escapes.
pub(crate) unsafe fn halation_frame_prepare(
    input: *const FjHalationFrameInput,
    out: *mut crate::cuda::sys::FjScatterHalation,
    failure: *mut FjHalationFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The checked body shares the full synchronous foreign storage contract.
        unsafe { halation_frame_body(input, out, failure, error) }
    }));
    match result {
        Ok(status) => status,
        Err(_) => {
            let (category, details, text) = HalationFrameRejection::Panic.fields();
            // SAFETY: Recovery uses only independent admission, fixed writes and bounded diagnostics.
            unsafe {
                clear_halation_frame(out, failure);
                if valid_extent(failure, 1) {
                    failure.write(details);
                }
                if let Ok(mut writer) = Diagnostic::admit(error) {
                    let _ = writer.write_str(text);
                }
            }
            status(category)
        }
    }
}

#[unsafe(no_mangle)]
/// # Safety
/// Same initialized/live/aligned, exclusive/disjoint obligations as halation_frame_prepare.
unsafe extern "C" fn fj_legacy_scatter_halation_frame_prepare(
    input: *const FjHalationFrameInput,
    out: *mut crate::cuda::sys::FjScatterHalation,
    failure: *mut FjHalationFrameFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The export delegates to the single whole-operation checked body.
    unsafe { halation_frame_prepare(input, out, failure, error) }
}

#[cfg(feature = "test-support")]
pub(crate) mod halation_frame_faults {
    use super::*;
    use std::cell::Cell;
    thread_local! { static NEXT: Cell<u32> = const { Cell::new(0) }; }
    pub(crate) fn arm(fault: u32) -> FjStatus {
        if fault > 4 {
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        NEXT.set(fault);
        status(FJ_STATUS_SUCCESS)
    }
    pub(crate) fn take() -> u32 {
        NEXT.replace(0)
    }
    pub(super) fn corrupt(
        mut v: crate::cuda::sys::FjScatterHalation,
        fault: u32,
    ) -> crate::cuda::sys::FjScatterHalation {
        match fault {
            2 => v.channels[0].core.kind = 99,
            3 => {
                v.channels[0].core.kind = 1;
                v.channels[0].core.radius = 10;
            }
            4 => v.recipe_hash ^= 1,
            _ => {}
        }
        v
    }
}

const _: () = {
    assert!(size_of::<FjHalationFrameInput>() == 64 && align_of::<FjHalationFrameInput>() == 8);
    assert!(std::mem::offset_of!(FjHalationFrameInput, recipe) == 0);
    assert!(std::mem::offset_of!(FjHalationFrameInput, pixel_size_um) == 56);
    assert!(size_of::<FjHalationFrameFailure>() == 24 && align_of::<FjHalationFrameFailure>() == 8);
    assert!(std::mem::offset_of!(FjHalationFrameFailure, reason) == 0);
    assert!(std::mem::offset_of!(FjHalationFrameFailure, field) == 4);
    assert!(std::mem::offset_of!(FjHalationFrameFailure, channel) == 8);
    assert!(std::mem::offset_of!(FjHalationFrameFailure, index) == 12);
    assert!(std::mem::offset_of!(FjHalationFrameFailure, sigma_px) == 16);
};
const _: unsafe extern "C" fn(
    *const FjHalationFrameInput,
    *mut crate::cuda::sys::FjScatterHalation,
    *mut FjHalationFrameFailure,
    *mut FjErrorBuffer,
) -> FjStatus = fj_legacy_scatter_halation_frame_prepare;
const _: fn(FjHalationFrameInput, FjHalationFrameFailure) = |input, failure| {
    let _: FjHalationRecipe = input.recipe;
    let _: f32 = input.pixel_size_um;
    let _: u32 = failure.reason;
    let _: u32 = failure.field;
    let _: u32 = failure.channel;
    let _: u32 = failure.index;
    let _: f64 = failure.sigma_px;
};

const _: fn(
    crate::cuda::sys::FjHalationGaussian,
    crate::cuda::sys::FjHalationChannel,
    crate::cuda::sys::FjScatterHalation,
) = |gaussian, channel, frame| {
    let _: u32 = gaussian.kind;
    let _: i32 = gaussian.radius;
    let _: [f32; 19] = gaussian.fir_weights;
    let _: f32 = gaussian.feedforward;
    let _: [f32; 3] = gaussian.feedback;
    let _: crate::cuda::sys::FjHalationGaussian = channel.core;
    let _: [crate::cuda::sys::FjHalationGaussian; 3] = channel.tail;
    let _: [crate::cuda::sys::FjHalationGaussian; 3] = channel.bounce;
    let _: f32 = channel.total_strength;
    let _: u64 = frame.recipe_hash;
    let _: f32 = frame.scatter_amount;
    let _: [crate::cuda::sys::FjHalationChannel; 3] = frame.channels;
};
