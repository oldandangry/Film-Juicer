//! SF_TEMP_BRIDGE_FILM_DIR_DENSITY: synchronous family owners and fixed projections; remove S4.E.

use std::fmt::{self, Write};
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use film_juicer_core::film::{Development, density, dir};
use film_juicer_core::profile::Polarity;
use film_juicer_core::route::ScanRoute;
use crate::film_bridge::FjFilmDevelopmentOwner;
use crate::cuda::sys::{
    FjErrorBuffer, FjFloatSpan, FjStatus, FJ_API_NONE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_ALLOCATION_FAILURE,
    FJ_STATUS_INTERNAL_FAILURE,
};

pub(crate) const FJ_FILM_DIR_INACTIVE: u32 = 0;

pub(crate) const FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR: u32 = 1;

pub(crate) const FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR: u32 = 2;

pub(crate) const FJ_FILM_RECIPE_REASON_NONE: u32 = 0;

pub(crate) const FJ_FILM_RECIPE_RAW: u32 = 1;

pub(crate) const FJ_FILM_RECIPE_MINIMUM: u32 = 2;

pub(crate) const FJ_FILM_RECIPE_MATRIX: u32 = 3;

pub(crate) const FJ_FILM_RECIPE_DENSITY_MAX: u32 = 4;

pub(crate) const FJ_FILM_RECIPE_RECEIVER_K: u32 = 5;

pub(crate) const FJ_FILM_RECIPE_DONOR_K: u32 = 6;

pub(crate) const FJ_FILM_RECIPE_DONOR_LANGMUIR: u32 = 7;

pub(crate) const FJ_FILM_RECIPE_RECEIVER_LANGMUIR: u32 = 8;

pub(crate) const FJ_FILM_RECIPE_COMPENSATED_AXIS: u32 = 9;

pub(crate) const FJ_FILM_RECIPE_IDENTITY: u32 = 10;

pub(crate) const FJ_FILM_RECIPE_BOUNDS_SPAN: u32 = 11;

pub(crate) const FJ_FILM_RECIPE_SIZE: u32 = 12;

pub(crate) const FJ_FILM_RECIPE_ALLOCATION: u32 = 13;

pub(crate) const FJ_FILM_RECIPE_PANIC: u32 = 14;

pub(crate) const FJ_FILM_RECIPE_INTEGRITY: u32 = 15;

pub(crate) const FJ_FILM_RECIPE_LOG_EXPOSURE: u32 = 16;

#[repr(C)]
pub(crate) struct FjFilmRecipeFailure {
    pub(crate) reason: u32,
    pub(crate) channel: u32,
    pub(crate) sample: u64,
}

#[repr(C)]
pub(crate) struct FjFilmDensityMinimum {
    pub(crate) cmy: [f32; 3],
}

#[repr(C)]
pub(crate) struct FjFilmDirParams {
    pub(crate) active: u32,
    pub(crate) amount: f32,
    pub(crate) inhibition_same_layer: f32,
    pub(crate) inhibition_interlayer: f32,
    pub(crate) diffusion_size_um: f32,
    pub(crate) diffusion_tail_um: f32,
    pub(crate) diffusion_tail_weight: f32,
    pub(crate) gamma_use_stock: u32,
    pub(crate) gamma_same_layer_rgb: [f32; 3],
    pub(crate) gamma_interlayer_r_to_gb: [f32; 2],
    pub(crate) gamma_interlayer_g_to_rb: [f32; 2],
    pub(crate) gamma_interlayer_b_to_rg: [f32; 2],
    pub(crate) langmuir_donor_k_rgb: [f32; 3],
    pub(crate) langmuir_receiver_k_rgb: [f32; 3],
}

#[repr(C)]
pub(crate) struct FjFilmDirView {
    pub(crate) polarity: u32,
    pub(crate) mode: u32,
    pub(crate) matrix_rgb: [f32; 9],
    pub(crate) diffusion_size_um: f32,
    pub(crate) diffusion_tail_um: f32,
    pub(crate) diffusion_tail_weight: f32,
    pub(crate) density_max_rgb: [f32; 3],
    pub(crate) density_ref_rgb: [f32; 3],
    pub(crate) donor_k_rgb: [f32; 3],
    pub(crate) receiver_c_ref_rgb: [f32; 3],
    pub(crate) receiver_kr_rgb: [f32; 3],
    pub(crate) compensated_axes_rgb: [FjFloatSpan; 3],
    pub(crate) compensated_axes_hash: u64,
    pub(crate) hash: u64,
}

#[repr(C)]
pub(crate) struct FjFilmDensityBounds {
    pub(crate) route: u32,
    pub(crate) medium: u32,
    pub(crate) polarity: u32,
    pub(crate) source: u32,
    pub(crate) minimum_cmy: [f32; 3],
    pub(crate) maximum_cmy: [f32; 3],
    pub(crate) inv_span_cmy: [f32; 3],
    pub(crate) authored_minimum_cmy: [f32; 3],
    pub(crate) authored_maximum_cmy: [f32; 3],
    pub(crate) hash: u64,
}

#[repr(C)]
pub(crate) struct FjFilmDirOwner {
    _opaque: [u8; 0],
}

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
impl Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.capacity != 0 {
            let count = text.len().min(self.capacity - 1 - self.length);
            // SAFETY: Admission establishes exclusive backing with this checked extent.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), self.data.add(self.length), count);
                self.length += count;
                self.data.add(self.length).write(0);
            }
        }
        if !self.record.is_null() {
            // SAFETY: The exclusively writable admitted record stays live until return.
            unsafe {
                (*self.record).length = self.length;
            }
        }
        Ok(())
    }
}
impl Diagnostic {
    /// # Safety
    /// The optional record/backing are initialized, live, exclusive and disjoint.
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
        // SAFETY: Capacity and writable length are initialized; zero capacity never reads data.
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
struct Failure {
    category: u32,
    record: FjFilmRecipeFailure,
}
impl Failure {
    fn new(category: u32, reason: u32) -> Self {
        Self {
            category,
            record: FjFilmRecipeFailure {
                reason,
                channel: 3,
                sample: u64::MAX,
            },
        }
    }
    fn raw() -> Self {
        Self::new(FJ_STATUS_UNSUPPORTED_INPUT, FJ_FILM_RECIPE_RAW)
    }
    fn dir(error: dir::Error) -> Self {
        let (reason, channel, sample, category) = match error {
            dir::Error::Matrix => (
                FJ_FILM_RECIPE_MATRIX,
                3,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::LogExposure { sample } => (
                FJ_FILM_RECIPE_LOG_EXPOSURE,
                3,
                sample as u64,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::DensityMaximum => (
                FJ_FILM_RECIPE_DENSITY_MAX,
                3,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::ReceiverK { channel } => (
                FJ_FILM_RECIPE_RECEIVER_K,
                channel as u32,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::DonorK { channel } => (
                FJ_FILM_RECIPE_DONOR_K,
                channel as u32,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::DonorLangmuir { channel, sample } => (
                FJ_FILM_RECIPE_DONOR_LANGMUIR,
                channel as u32,
                sample as u64,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::ReceiverLangmuir { channel, sample } => (
                FJ_FILM_RECIPE_RECEIVER_LANGMUIR,
                channel as u32,
                sample as u64,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::CompensatedAxis { channel, sample } => (
                FJ_FILM_RECIPE_COMPENSATED_AXIS,
                channel as u32,
                sample as u64,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
            dir::Error::Size => (
                FJ_FILM_RECIPE_SIZE,
                3,
                u64::MAX,
                FJ_STATUS_ALLOCATION_FAILURE,
            ),
            dir::Error::Allocation => (
                FJ_FILM_RECIPE_ALLOCATION,
                3,
                u64::MAX,
                FJ_STATUS_ALLOCATION_FAILURE,
            ),
            dir::Error::ZeroIdentity => (
                FJ_FILM_RECIPE_IDENTITY,
                3,
                u64::MAX,
                FJ_STATUS_PREPARATION_FAILURE,
            ),
        };
        Self {
            category,
            record: FjFilmRecipeFailure {
                reason,
                channel,
                sample,
            },
        }
    }
    fn density(error: density::Error) -> Self {
        let mut failure = Self::new(
            FJ_STATUS_PREPARATION_FAILURE,
            match error {
                density::Error::Minimum => FJ_FILM_RECIPE_MINIMUM,
                density::Error::Span { .. } => FJ_FILM_RECIPE_BOUNDS_SPAN,
                density::Error::ZeroIdentity => FJ_FILM_RECIPE_IDENTITY,
            },
        );
        if let density::Error::Span { channel } = error {
            failure.record.channel = channel as u32;
        }
        failure
    }
}

/// # Safety
/// T is one of this module's plain numerical/nullable-pointer wire results, valid
/// when zeroed. Caller authorizes live initialized output/failure/diagnostic
/// extents, mutually disjoint and disjoint from all reachable input/owner storage.
unsafe fn execute<T>(
    out: *mut T,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<T, Failure>,
) -> FjStatus {
    let valid_out = valid_extent(out, 1);
    let valid_failure = valid_extent(failure, 1);
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Each independently authorized result is cleared, including nullable spans.
        unsafe {
            if valid_out {
                out.write_bytes(0, 1);
            }
            if valid_failure {
                failure.write(Failure::raw().record);
            }
        }
        // SAFETY: Same exclusive diagnostic contract, admitted before semantic input.
        let diagnostic = unsafe { Diagnostic::admit(error) };
        let rejected = diagnostic.is_err() || !valid_out || !valid_failure;
        let mut diagnostic = diagnostic.ok();
        let value = if rejected {
            Err(Failure::raw())
        } else {
            operation()
        };
        match value {
            Ok(value) => {
                // SAFETY: Complete result and success metadata commit only to admitted outputs.
                unsafe {
                    out.write(value);
                    failure.write(FjFilmRecipeFailure {
                        reason: FJ_FILM_RECIPE_REASON_NONE,
                        channel: 3,
                        sample: u64::MAX,
                    });
                }
                status(FJ_STATUS_SUCCESS)
            }
            Err(rejection) => {
                if let Some(writer) = &mut diagnostic {
                    #[cfg(feature = "test-support")]
                    if faults::take(4) == 1 {
                        panic!("film recipe diagnostic fixture panic");
                    }
                    let _ = write!(
                        writer,
                        "film recipe reason={} channel={} sample={}",
                        rejection.record.reason, rejection.record.channel, rejection.record.sample
                    );
                }
                if valid_failure {
                    // SAFETY: Independent valid failure storage receives the closed typed error.
                    unsafe {
                        failure.write(rejection.record);
                    }
                }
                status(rejection.category)
            }
        }
    }));
    result.unwrap_or_else(|_| {
        // SAFETY: These extents were admitted independently; panic cannot publish partial output.
        unsafe {
            if valid_out {
                out.write_bytes(0, 1);
            }
            if valid_failure {
                failure
                    .write(Failure::new(FJ_STATUS_INTERNAL_FAILURE, FJ_FILM_RECIPE_PANIC).record);
            }
        }
        status(FJ_STATUS_INTERNAL_FAILURE)
    })
}

impl FjFilmDirParams {
    fn params(&self) -> Result<dir::Params, Failure> {
        if self.active > 1 || self.gamma_use_stock > 1 {
            return Err(Failure::raw());
        }
        Ok(dir::Params {
            active: self.active != 0,
            amount: self.amount,
            inhibition_same_layer: self.inhibition_same_layer,
            inhibition_interlayer: self.inhibition_interlayer,
            diffusion_size_um: self.diffusion_size_um,
            diffusion_tail_um: self.diffusion_tail_um,
            diffusion_tail_weight: self.diffusion_tail_weight,
            gamma_use_stock: self.gamma_use_stock != 0,
            gamma_same_layer_rgb: self.gamma_same_layer_rgb,
            gamma_interlayer_r_to_gb: self.gamma_interlayer_r_to_gb,
            gamma_interlayer_g_to_rb: self.gamma_interlayer_g_to_rb,
            gamma_interlayer_b_to_rg: self.gamma_interlayer_b_to_rg,
            langmuir_donor_k_rgb: self.langmuir_donor_k_rgb,
            langmuir_receiver_k_rgb: self.langmuir_receiver_k_rgb,
        })
    }
}
impl FjFilmDirView {
    fn new(recipe: &dir::Recipe) -> Self {
        Self {
            polarity: match recipe.polarity() {
                Some(Polarity::Negative) => 0,
                Some(Polarity::Positive) => 1,
                None => 2,
            },
            mode: match recipe.mode() {
                dir::Mode::Inactive => FJ_FILM_DIR_INACTIVE,
                dir::Mode::NegativeDonorLangmuir => FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR,
                dir::Mode::PositiveReceiverLangmuir => FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR,
            },
            matrix_rgb: std::array::from_fn(|i| recipe.matrix_rgb()[i / 3][i % 3]),
            diffusion_size_um: recipe.diffusion_size_um(),
            diffusion_tail_um: recipe.diffusion_tail_um(),
            diffusion_tail_weight: recipe.diffusion_tail_weight(),
            density_max_rgb: *recipe.density_max_rgb(),
            density_ref_rgb: *recipe.density_ref_rgb(),
            donor_k_rgb: *recipe.donor_k_rgb(),
            receiver_c_ref_rgb: *recipe.receiver_c_ref_rgb(),
            receiver_kr_rgb: *recipe.receiver_kr_rgb(),
            compensated_axes_rgb: std::array::from_fn(|i| {
                let axis = &recipe.compensated_axes_rgb()[i];
                FjFloatSpan {
                    data: if axis.is_empty() {
                        std::ptr::null()
                    } else {
                        axis.as_ptr()
                    },
                    count: axis.len(),
                }
            }),
            compensated_axes_hash: recipe.compensated_axes_hash(),
            hash: recipe.hash(),
        }
    }
}
impl FjFilmDensityBounds {
    fn new(bounds: &density::Bounds) -> Self {
        Self {
            route: match bounds.route() {
                ScanRoute::NegativeDirect => 0,
                ScanRoute::NegativePrint => 1,
                ScanRoute::PositiveDirect => 2,
                ScanRoute::PositivePrint => 3,
            },
            medium: 0,
            polarity: match bounds.polarity() {
                Polarity::Negative => 0,
                Polarity::Positive => 1,
            },
            source: match bounds.source() {
                density::Source::DirectFilm => 0,
                density::Source::EnlargerFilm => 1,
            },
            minimum_cmy: *bounds.minimum_cmy(),
            maximum_cmy: *bounds.maximum_cmy(),
            inv_span_cmy: *bounds.inv_span_cmy(),
            authored_minimum_cmy: *bounds.authored_minimum_cmy(),
            authored_maximum_cmy: *bounds.authored_maximum_cmy(),
            hash: bounds.hash(),
        }
    }
}

/// # Safety
/// Initialized/disjoint live records/backing with caller-authorized extents.
pub(crate) unsafe fn minimum(
    requested: *const FjFilmDensityMinimum,
    out: *mut FjFilmDensityMinimum,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: All records follow the immediate synchronous execute contract.
    unsafe {
        execute(out, failure, error, || {
            if !valid_extent(requested, 1) {
                return Err(Failure::raw());
            }
            let minimum = density::Minimum::new((*requested).cmy).map_err(Failure::density)?;
            Ok(FjFilmDensityMinimum {
                cmy: *minimum.cmy(),
            })
        })
    }
}
/// # Safety
/// Live development hold and initialized/disjoint records/backing until return.
pub(crate) unsafe fn acquire(
    development: *const FjFilmDevelopmentOwner,
    params: *const FjFilmDirParams,
    out: *mut *mut FjFilmDirOwner,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Borrow the completed development only after raw extent admission.
    unsafe {
        execute(out, failure, error, || {
            let development = development.cast::<Development>();
            if !valid_extent(development, 1) || !valid_extent(params, 1) {
                return Err(Failure::raw());
            }
            let params = (*params).params()?;
            #[cfg(feature = "test-support")]
            if faults::take(0) == 1 {
                panic!("film DIR acquire fixture panic");
            }
            let recipe = dir::Recipe::new(&*development, params).map_err(Failure::dir)?;
            let owner = Box::new(recipe);
            #[cfg(feature = "test-support")]
            faults::LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // No fallible work follows raw ownership transfer before execute commits it.
            Ok(Box::into_raw(owner).cast())
        })
    }
}
/// # Safety
/// A matching live immutable DIR owner; exclusive disjoint output/diagnostics.
pub(crate) unsafe fn view(
    owner: *const FjFilmDirOwner,
    out: *mut FjFilmDirView,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The borrow expires at consuming release; no reference escapes this call.
    unsafe {
        execute(out, failure, error, || {
            let recipe = owner.cast::<dir::Recipe>();
            if !valid_extent(recipe, 1) {
                return Err(Failure::raw());
            }
            #[cfg(feature = "test-support")]
            if faults::take(1) == 1 {
                panic!("film DIR view fixture panic");
            }
            Ok(FjFilmDirView::new(&*recipe))
        })
    }
}
/// # Safety
/// Matching NULL/live DIR owner, consumed once, excluding every read/view.
/// Diagnostics obey the existing asset-owner family's optional-buffer contract.
pub(crate) unsafe fn release(owner: *mut FjFilmDirOwner, error: *mut FjErrorBuffer) -> FjStatus {
    if !owner.is_null() && !valid_extent(owner.cast::<dir::Recipe>(), 1) {
        // SAFETY: A numerically malformed non-handle owns nothing. Diagnostics
        // retain the existing owner-family optional-buffer contract.
        return unsafe {
            crate::asset_bridge::run(error, || {
                Err::<(), _>(crate::asset_bridge::Failure::Input(
                    "malformed film DIR release owner",
                ))
            })
        }
        .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS));
    }
    let owned = if owner.is_null() {
        None
    } else {
        #[cfg(feature = "test-support")]
        faults::LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: The caller transfers one matching Box before diagnostic rejection.
        Some(unsafe { Box::from_raw(owner.cast::<dir::Recipe>()) })
    };
    // SAFETY: Captured ownership is destroyed inside containment even when diagnostics reject.
    unsafe {
        crate::asset_bridge::run(error, move || {
            #[cfg(feature = "test-support")]
            if faults::take(2) == 1 {
                panic!("film DIR release fixture panic");
            }
            drop(owned);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}
/// # Safety
/// Live development and unchanged original minimum request; initialized/disjoint
/// exclusive outputs/diagnostics. No owner or input span is retained.
pub(crate) unsafe fn bounds(
    development: *const FjFilmDevelopmentOwner,
    requested: *const FjFilmDensityMinimum,
    out: *mut FjFilmDensityBounds,
    failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Fresh raw request is admitted through the one safe minimum constructor.
    unsafe {
        execute(out, failure, error, || {
            let development = development.cast::<Development>();
            if !valid_extent(development, 1) || !valid_extent(requested, 1) {
                return Err(Failure::raw());
            }
            let minimum = density::Minimum::new((*requested).cmy).map_err(Failure::density)?;
            #[cfg(feature = "test-support")]
            if faults::take(3) == 1 {
                panic!("film density bounds fixture panic");
            }
            let bounds = density::Bounds::new(&*development, &minimum).map_err(Failure::density)?;
            Ok(FjFilmDensityBounds::new(&bounds))
        })
    }
}

#[cfg(feature = "test-support")]
mod faults {
    use std::cell::Cell;
    use std::sync::atomic::AtomicUsize;
    thread_local! {pub(super) static NEXT:Cell<(u32,u32)>=const{Cell::new((0,0))};}
    pub(super) static LIVE: AtomicUsize = AtomicUsize::new(0);
    pub(super) fn take(stage: u32) -> u32 {
        let (selected, fault) = NEXT.get();
        if selected == stage {
            NEXT.set((0, 0));
            fault
        } else {
            0
        }
    }
}

#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_film_recipe_fault(stage: u32, fault: u32) -> FjStatus {
    if stage == 5 && fault <= 3 {
        dir::test_support::reject_next(fault);
        return status(FJ_STATUS_SUCCESS);
    }
    if stage == 6 && fault == 1 {
        density::test_support::zero_next_identity();
        return status(FJ_STATUS_SUCCESS);
    }
    if stage > 4 || fault > 1 {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    faults::NEXT.set((stage, fault));
    status(FJ_STATUS_SUCCESS)
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_film_dir_live_owners() -> usize {
    faults::LIVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_density_minimum_prepare(
    requested: *const FjFilmDensityMinimum,
    out_minimum: *mut FjFilmDensityMinimum,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { minimum(requested, out_minimum, out_failure, error) }
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_dir_acquire(
    development: *const FjFilmDevelopmentOwner,
    params: *const FjFilmDirParams,
    out_dir: *mut *mut FjFilmDirOwner,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { acquire(development, params, out_dir, out_failure, error) }
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_dir_view(
    dir: *const FjFilmDirOwner,
    out_view: *mut FjFilmDirView,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { view(dir, out_view, out_failure, error) }
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_dir_release(
    dir: *mut FjFilmDirOwner,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { release(dir, error) }
}

/// # Safety
/// Matching live owners; caller-authorized initialized/aligned/disjoint input, output and diagnostic extents.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_film_density_bounds_prepare(
    development: *const FjFilmDevelopmentOwner,
    requested: *const FjFilmDensityMinimum,
    out_bounds: *mut FjFilmDensityBounds,
    out_failure: *mut FjFilmRecipeFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Forward the identical immediate raw call contract.
    unsafe { bounds(development, requested, out_bounds, out_failure, error) }
}

// The closed wire contract reserves integrity for detecting impossible envelopes.
const _: () = assert!(FJ_FILM_RECIPE_INTEGRITY == 15);

#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
/// # Safety
/// Count is NULL or a live aligned exclusively writable usize slot. Returned
/// facts have immutable static lifetime and cannot be freed by the caller.
unsafe extern "C" fn fj_test_film_recipe_abi_facts(count: *mut usize) -> *const usize {
    const FACTS: &[usize] = &[
        std::mem::size_of::<FjFilmRecipeFailure>(),
        std::mem::align_of::<FjFilmRecipeFailure>(),
        std::mem::offset_of!(FjFilmRecipeFailure, reason),
        std::mem::offset_of!(FjFilmRecipeFailure, channel),
        std::mem::offset_of!(FjFilmRecipeFailure, sample),
        std::mem::size_of::<FjFilmDensityMinimum>(),
        std::mem::align_of::<FjFilmDensityMinimum>(),
        std::mem::offset_of!(FjFilmDensityMinimum, cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::size_of::<FjFilmDirParams>(),
        std::mem::align_of::<FjFilmDirParams>(),
        std::mem::offset_of!(FjFilmDirParams, active),
        std::mem::offset_of!(FjFilmDirParams, amount),
        std::mem::offset_of!(FjFilmDirParams, inhibition_same_layer),
        std::mem::offset_of!(FjFilmDirParams, inhibition_interlayer),
        std::mem::offset_of!(FjFilmDirParams, diffusion_size_um),
        std::mem::offset_of!(FjFilmDirParams, diffusion_tail_um),
        std::mem::offset_of!(FjFilmDirParams, diffusion_tail_weight),
        std::mem::offset_of!(FjFilmDirParams, gamma_use_stock),
        std::mem::offset_of!(FjFilmDirParams, gamma_same_layer_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, gamma_interlayer_r_to_gb),
        std::mem::size_of::<[f32; 2]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, gamma_interlayer_g_to_rb),
        std::mem::size_of::<[f32; 2]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, gamma_interlayer_b_to_rg),
        std::mem::size_of::<[f32; 2]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, langmuir_donor_k_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirParams, langmuir_receiver_k_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::size_of::<FjFilmDirView>(),
        std::mem::align_of::<FjFilmDirView>(),
        std::mem::offset_of!(FjFilmDirView, polarity),
        std::mem::offset_of!(FjFilmDirView, mode),
        std::mem::offset_of!(FjFilmDirView, matrix_rgb),
        std::mem::size_of::<[f32; 9]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, diffusion_size_um),
        std::mem::offset_of!(FjFilmDirView, diffusion_tail_um),
        std::mem::offset_of!(FjFilmDirView, diffusion_tail_weight),
        std::mem::offset_of!(FjFilmDirView, density_max_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, density_ref_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, donor_k_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, receiver_c_ref_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, receiver_kr_rgb),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDirView, compensated_axes_rgb),
        std::mem::size_of::<[FjFloatSpan; 3]>() / std::mem::size_of::<FjFloatSpan>(),
        std::mem::offset_of!(FjFilmDirView, compensated_axes_hash),
        std::mem::offset_of!(FjFilmDirView, hash),
        std::mem::size_of::<FjFilmDensityBounds>(),
        std::mem::align_of::<FjFilmDensityBounds>(),
        std::mem::offset_of!(FjFilmDensityBounds, route),
        std::mem::offset_of!(FjFilmDensityBounds, medium),
        std::mem::offset_of!(FjFilmDensityBounds, polarity),
        std::mem::offset_of!(FjFilmDensityBounds, source),
        std::mem::offset_of!(FjFilmDensityBounds, minimum_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, maximum_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, inv_span_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, authored_minimum_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, authored_maximum_cmy),
        std::mem::size_of::<[f32; 3]>() / std::mem::size_of::<f32>(),
        std::mem::offset_of!(FjFilmDensityBounds, hash),
        FJ_FILM_DIR_INACTIVE as usize,
        FJ_FILM_DIR_NEGATIVE_DONOR_LANGMUIR as usize,
        FJ_FILM_DIR_POSITIVE_RECEIVER_LANGMUIR as usize,
        FJ_FILM_RECIPE_REASON_NONE as usize,
        FJ_FILM_RECIPE_RAW as usize,
        FJ_FILM_RECIPE_MINIMUM as usize,
        FJ_FILM_RECIPE_MATRIX as usize,
        FJ_FILM_RECIPE_DENSITY_MAX as usize,
        FJ_FILM_RECIPE_RECEIVER_K as usize,
        FJ_FILM_RECIPE_DONOR_K as usize,
        FJ_FILM_RECIPE_DONOR_LANGMUIR as usize,
        FJ_FILM_RECIPE_RECEIVER_LANGMUIR as usize,
        FJ_FILM_RECIPE_COMPENSATED_AXIS as usize,
        FJ_FILM_RECIPE_IDENTITY as usize,
        FJ_FILM_RECIPE_BOUNDS_SPAN as usize,
        FJ_FILM_RECIPE_SIZE as usize,
        FJ_FILM_RECIPE_ALLOCATION as usize,
        FJ_FILM_RECIPE_PANIC as usize,
        FJ_FILM_RECIPE_INTEGRITY as usize,
        FJ_FILM_RECIPE_LOG_EXPOSURE as usize,
    ];
    if !count.is_null() {
        // SAFETY: Facade caller authorizes the exclusive count output.
        unsafe {
            count.write(FACTS.len());
        }
    }
    FACTS.as_ptr()
}
