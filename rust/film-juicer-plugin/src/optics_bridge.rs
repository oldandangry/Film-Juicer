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
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjGaussianKernelView {
    pub(crate) radius: i32,
    pub(crate) sigma_px: f32,
    pub(crate) weights: crate::cuda::sys::FjFloatSpan,
}
impl FjGaussianKernelView {
    fn empty() -> Self {
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
fn gaussian_panic(fault: u32) {
    #[cfg(feature = "test-support")]
    if fault == 1 {
        panic!("Gaussian kernel fixture panic");
    }
    #[cfg(not(feature = "test-support"))]
    let _ = fault;
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
    let fault = gaussian_fault(0);
    #[cfg(feature = "test-support")]
    gaussian_faults::ACQUIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: Common admission clears valid peers before borrowing the immutable input.
    unsafe {
        gaussian_call(out, std::ptr::null_mut(), failure, error, false, || {
            if !valid_extent(input, 1) {
                return Err(GaussianRejection::Raw);
            }
            gaussian_panic(fault);
            let input = &*input;
            #[cfg(feature = "test-support")]
            optics::test_support::reject_gaussian_reservation(fault == 2);
            let kernel = optics::GaussianKernel::new(input.radius as u32, input.sigma_px)
                .map_err(GaussianRejection::Core)?;
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
                gaussian_panic(fault);
                let kernel = &*owner.cast::<optics::GaussianKernel>();
                let view = FjGaussianKernelView {
                    radius: kernel.radius() as i32,
                    sigma_px: kernel.sigma_px(),
                    weights: crate::cuda::sys::FjFloatSpan {
                        data: kernel.weights().as_ptr(),
                        count: kernel.weights().len(),
                    },
                };
                #[cfg(feature = "test-support")]
                let view = gaussian_faults::corrupt(view, fault);
                Ok(view)
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
