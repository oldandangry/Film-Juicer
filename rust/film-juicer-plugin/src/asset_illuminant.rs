//! Synchronous illuminant values and consuming call-local lens boundary.
use std::panic::{AssertUnwindSafe, catch_unwind};
use film_juicer_core::illuminant::{self, Coverage, LensInput};
use crate::asset_bridge::{self, Failure};
use crate::cuda::sys::FjFloatSpan;
use crate::cuda::sys::{
    FjErrorBuffer, FjStatus, FJ_API_NONE, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT,
};

#[repr(C)]
pub(crate) struct FjIlluminant {
    pub samples: [f32; 81],
}
impl Default for FjIlluminant {
    fn default() -> Self {
        Self { samples: [0.0; 81] }
    }
}
#[repr(C)]
#[derive(Default)]
pub(crate) struct FjIlluminantCoverage {
    pub warnings: u32,
    pub min_nm: f32,
    pub max_nm: f32,
}
impl From<Coverage> for FjIlluminantCoverage {
    fn from(c: Coverage) -> Self {
        Self {
            warnings: c.warnings,
            min_nm: c.min_nm,
            max_nm: c.max_nm,
        }
    }
}
pub(crate) struct FjIlluminantLens {
    input: Option<LensInput>,
}
impl Drop for FjIlluminantLens {
    fn drop(&mut self) {
        #[cfg(feature = "test-support")]
        LIVE_LENSES.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
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
    !pointer.is_null() && (pointer as usize).is_multiple_of(std::mem::align_of::<T>())
}

/// # Safety
/// Aligned nonnull outputs authorize complete exclusive writable records.
unsafe fn clear<T: Default>(out: *mut T) -> bool {
    if !aligned(out) {
        return false;
    }
    // SAFETY: The caller authorizes the complete writable record.
    unsafe { out.write(T::default()) };
    true
}
/// # Safety
/// Caller authorizes the initialized immutable scalar extent until return.
pub(crate) unsafe fn rows<'a>(span: FjFloatSpan) -> Result<&'a [[f32; 2]], Failure> {
    if !span.count.is_multiple_of(2) {
        return Err(Failure::Input("odd illuminant scalar count"));
    }
    let bytes = span
        .count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(Failure::Input("illuminant extent overflow"))?;
    if bytes > isize::MAX as usize || (span.data as usize).checked_add(bytes).is_none() {
        return Err(Failure::Input("illuminant extent overflow"));
    }
    if span.count == 0 {
        return Ok(&[]);
    }
    if !aligned(span.data) {
        return Err(Failure::Input("invalid illuminant rows"));
    }
    // SAFETY: Checked interleaved floats have the identical [f32;2] alignment
    // and layout. The caller retains this immutable initialized extent.
    Ok(unsafe { std::slice::from_raw_parts(span.data.cast::<[f32; 2]>(), span.count / 2) })
}
/// # Safety
/// Required diagnostic authorizes an initialized exclusive record and its
/// backing extent. Captured consumed owners are destroyed inside containment.
pub(crate) unsafe fn run(
    error: *mut FjErrorBuffer,
    action: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    catch_unwind(AssertUnwindSafe(|| {
        if !aligned(error) {
            drop(action);
            return status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: This aligned required record is initialized and exclusively writable.
        let capacity = unsafe { (*error).capacity };
        if capacity != 0 {
            // SAFETY: Nonzero capacity requires the initialized backing pointer.
            let data = unsafe { (*error).data };
            if (data as usize).checked_add(capacity).is_none() {
                // SAFETY: Only the authorized record is written for invalid backing extent.
                unsafe {
                    (*error).length = 0;
                }
                drop(action);
                return status(FJ_STATUS_UNSUPPORTED_INPUT);
            }
        }
        // SAFETY: Shared containment validates backing storage and formats errors.
        unsafe { asset_bridge::run(error, action) }
            .map_or_else(|s| s, |()| status(FJ_STATUS_SUCCESS))
    }))
    .unwrap_or(status(FJ_STATUS_INTERNAL_FAILURE))
}
/// # Safety
/// A valid aligned location owns at most one live matching lens; no concurrent use.
unsafe fn take(io: *mut *mut FjIlluminantLens) -> Result<Option<Box<FjIlluminantLens>>, Failure> {
    if !aligned(io) {
        return Err(Failure::Input("invalid lens owner location"));
    }
    // SAFETY: The caller exclusively authorizes this location and live handle.
    let pointer = unsafe { io.replace(std::ptr::null_mut()) };
    if pointer.is_null() {
        Ok(None)
    } else {
        // SAFETY: The caller transfers the unique live matching Box exactly once.
        Ok(Some(unsafe { Box::from_raw(pointer) }))
    }
}
/// # Safety
/// Coverage is an aligned exclusive output retained through this synchronous call.
unsafe fn covered<T>(
    value: Result<(T, Coverage), illuminant::Error>,
    out: *mut FjIlluminantCoverage,
) -> Result<T, Failure> {
    match value {
        Ok((value, coverage)) => {
            // SAFETY: This validated output has no overlap with call inputs.
            unsafe { out.write(coverage.into()) };
            Ok(value)
        }
        Err(error) => {
            // SAFETY: Failure metadata is diagnostic only and confers no readiness.
            unsafe { out.write(error.coverage.into()) };
            Err(Failure::Illuminant(error))
        }
    }
}

#[cfg(feature = "test-support")]
static LIVE_LENSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "test-support")]
thread_local! { static FAULT: std::cell::Cell<Option<(u32,u32,u32)>> = const { std::cell::Cell::new(None) }; }
#[cfg(feature = "test-support")]
pub(crate) fn arm_fault(operation: u32, call_index: u32, fault: u32) -> FjStatus {
    FAULT.set(None);
    if !(1..=7).contains(&operation) || call_index == 0 || !(1..=4).contains(&fault) {
        return status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    FAULT.set(Some((operation, call_index, fault)));
    status(FJ_STATUS_SUCCESS)
}
#[cfg(feature = "test-support")]
pub(crate) fn clear_fault() {
    FAULT.set(None);
}
#[cfg(feature = "test-support")]
pub(crate) fn live_lenses() -> usize {
    LIVE_LENSES.load(std::sync::atomic::Ordering::SeqCst)
}
fn inject(operation: u32) -> Result<(), Failure> {
    #[cfg(feature = "test-support")]
    {
        if let Some((selected, remaining, fault)) = FAULT.get()
            && selected == operation
        {
            if remaining > 1 {
                FAULT.set(Some((selected, remaining - 1, fault)));
                return Ok(());
            }
            FAULT.set(None);
            match fault {
                1 => return Err(Failure::Input("illuminant injected unsupported input")),
                2 => panic!("illuminant injected panic"),
                3 => {
                    return illuminant::test_support::capacity_failure()
                        .map_err(Failure::Illuminant);
                }
                4 => {
                    return Err(Failure::Illuminant(
                        illuminant::from_samples(&[]).unwrap_err(),
                    ));
                }
                _ => unreachable!("closed fault selector"),
            }
        }
    }
    #[cfg(not(feature = "test-support"))]
    let _ = operation;
    Ok(())
}

// FJ_TEMP_BRIDGE: illuminant value/source preparation; remove S4.E.
/// # Safety
/// Rows and required outputs/error obey the header's synchronous disjoint-storage contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_illuminant_from_samples(
    span: FjFloatSpan,
    out: *mut FjIlluminant,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Valid writable outputs clear before fallible work.
    let valid = unsafe { clear(out) };
    // SAFETY: Required diagnostics and retained source satisfy run/rows contracts.
    unsafe {
        run(error, || {
            if !valid {
                return Err(Failure::Input("invalid illuminant output"));
            }
            let rows = rows(span)?;
            inject(1)?;
            out.write(FjIlluminant {
                samples: illuminant::from_samples(rows).map_err(Failure::Illuminant)?,
            });
            Ok(())
        })
    }
}
/// # Safety
/// Required outputs/error obey the header's exclusive disjoint-storage contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_illuminant_blackbody(
    temperature: f32,
    out: *mut FjIlluminant,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The caller authorizes writable output and diagnostic storage.
    unsafe {
        let valid = clear(out);
        run(error, || {
            if !valid {
                return Err(Failure::Input("invalid illuminant output"));
            }
            inject(2)?;
            out.write(FjIlluminant {
                samples: illuminant::blackbody(temperature),
            });
            Ok(())
        })
    }
}
/// # Safety
/// Required outputs/error obey the header's exclusive disjoint-storage contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_illuminant_equal_energy(
    out: *mut FjIlluminant,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The caller authorizes writable output and diagnostic storage.
    unsafe {
        let valid = clear(out);
        run(error, || {
            if !valid {
                return Err(Failure::Input("invalid illuminant output"));
            }
            inject(3)?;
            out.write(FjIlluminant {
                samples: illuminant::equal_energy(),
            });
            Ok(())
        })
    }
}
/// # Safety
/// Rows and required outputs/error obey the header's synchronous disjoint-storage contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_illuminant_tungsten_kg3(
    span: FjFloatSpan,
    out: *mut FjIlluminant,
    coverage: *mut FjIlluminantCoverage,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Both validated outputs clear before any fallible work.
    unsafe {
        let curve_valid = clear(out);
        let coverage_valid = clear(coverage);
        run(error, || {
            if !curve_valid || !coverage_valid {
                return Err(Failure::Input("invalid KG3 output"));
            }
            let rows = rows(span)?;
            inject(4)?;
            out.write(FjIlluminant {
                samples: covered(illuminant::tungsten_kg3(rows), coverage)?,
            });
            Ok(())
        })
    }
}
// FJ_TEMP_BRIDGE: consuming call-local illuminant lens; remove S4.E.
/// # Safety
/// Rows and required outputs/error obey the header contract; successful owner transfers to caller.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_illuminant_lens_prepare(
    span: FjFloatSpan,
    out: *mut *mut FjIlluminantLens,
    coverage: *mut FjIlluminantCoverage,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Valid owner/output locations clear before fallible work.
    unsafe {
        let owner_valid = clear(out);
        let coverage_valid = clear(coverage);
        run(error, || {
            if !owner_valid || !coverage_valid {
                return Err(Failure::Input("invalid lens preparation output"));
            }
            let rows = rows(span)?;
            inject(5)?;
            let input = covered(illuminant::prepare_lens(rows), coverage)?;
            let owner = Box::new(FjIlluminantLens { input: Some(input) });
            #[cfg(feature = "test-support")]
            LIVE_LENSES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            out.write(Box::into_raw(owner));
            Ok(())
        })
    }
}
/// # Safety
/// Valid io owns one live lens; required rows/outputs/error obey the header contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_illuminant_lens_finish(
    io: *mut *mut FjIlluminantLens,
    span: FjFloatSpan,
    out: *mut FjIlluminant,
    coverage: *mut FjIlluminantCoverage,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Taking/nullling precedes all other validation; cleanup stays in run.
    unsafe {
        let owner = take(io);
        let curve_valid = clear(out);
        let coverage_valid = clear(coverage);
        run(error, move || {
            let mut owner = owner?.ok_or(Failure::Input("absent lens owner"))?;
            if !curve_valid || !coverage_valid {
                return Err(Failure::Input("invalid lens finish output"));
            }
            let rows = rows(span)?;
            inject(6)?;
            let input = owner.input.take().expect("complete lens owner");
            out.write(FjIlluminant {
                samples: covered(illuminant::finish_lens(input, rows), coverage)?,
            });
            Ok(())
        })
    }
}
/// # Safety
/// Valid io exclusively owns a live matching lens or NULL; required error obeys header contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_illuminant_lens_release(
    io: *mut *mut FjIlluminantLens,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Transfer before diagnostics ensures consume-on-failure semantics.
    unsafe {
        let owner = take(io);
        run(error, move || {
            let _owner = owner?;
            inject(7)
        })
    }
}

const _: () = assert!(
    std::mem::size_of::<FjIlluminant>() == 324
        && std::mem::align_of::<FjIlluminant>() == 4
        && std::mem::size_of::<FjIlluminantCoverage>() == 12
        && std::mem::align_of::<FjIlluminantCoverage>() == 4
);
