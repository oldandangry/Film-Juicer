//! Synchronous TC borrows and allocation-free transfer of the completed Vec.

use film_juicer_core::{gamut, reconstruction};
use crate::asset_bridge::Failure;
use crate::asset_illuminant;
use crate::cuda::sys::{
    FjFloatSpan, FjErrorBuffer, FjStatus, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT,
    FJ_API_NONE,
};

#[cfg(feature = "test-support")]
static TC_TRANSFERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "test-support")]
static TC_RELEASES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "test-support")]
pub(crate) fn allocation_counts() -> [usize; 2] {
    use std::sync::atomic::Ordering;
    [
        TC_TRANSFERS.load(Ordering::SeqCst),
        TC_RELEASES.load(Ordering::SeqCst),
    ]
}

// FJ_TEMP_BRIDGE: native TC preparation/ownership; remove S4.E.
#[repr(C)]
pub(crate) struct FjFilmTcLutInput {
    pub spectra: FjFloatSpan,
    pub sensitivity_rgb: FjFloatSpan,
    pub reference_illuminant: FjFloatSpan,
    pub projection_white_xyz: [f32; 3],
    pub spectral_blur: f32,
    pub method: u32,
    pub apply_surface: u32,
    pub surface_rgb: FjFloatSpan,
    pub compression_active: u32,
    pub hull_available: u32,
    pub hull_center_xy: [f32; 2],
    pub hull_xy: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjOwnedFilmTcLut {
    pub samples: FjFloatSpan,
    pub capacity: usize,
}
impl FjOwnedFilmTcLut {
    pub(crate) fn empty() -> Self {
        Self {
            samples: FjFloatSpan {
                data: std::ptr::null(),
                count: 0,
            },
            capacity: 0,
        }
    }
    fn is_empty(&self) -> bool {
        self.samples.data.is_null() && self.samples.count == 0 && self.capacity == 0
    }
}
fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}
/// # Safety
/// The caller retains initialized immutable live storage for the complete span,
/// disjoint from output/error records and backing until synchronous return.
unsafe fn fixed<'a, const N: usize>(span: FjFloatSpan) -> Result<&'a [f32; N], Failure> {
    let bytes = span
        .count
        .checked_mul(size_of::<f32>())
        .ok_or(Failure::Input("TC span extent overflow"))?;
    if span.count != N
        || bytes > isize::MAX as usize
        || !aligned(span.data)
        || (span.data as usize).checked_add(bytes).is_none()
    {
        return Err(Failure::Input("invalid fixed TC span"));
    }
    // SAFETY: Exact extent/alignment checked; caller supplies live initialized storage.
    Ok(unsafe { &*span.data.cast::<[f32; N]>() })
}
/// # Safety
/// The aligned complete input and consumed spans remain live and immutable,
/// disjoint from all exclusive output/error storage for the synchronous call.
unsafe fn input<'a>(
    pointer: *const FjFilmTcLutInput,
) -> Result<reconstruction::Input<'a>, Failure> {
    if !aligned(pointer) {
        return Err(Failure::Input("invalid TC input record"));
    }
    // SAFETY: Caller retains the complete input; all consumed fixed borrows are
    // formed after extent checks. Inactive surface/hull storage is never borrowed.
    unsafe {
        let input = &*pointer;
        if !matches!(input.method, 0 | 2)
            || input.apply_surface > 1
            || input.compression_active > 1
            || input.hull_available > 1
        {
            return Err(Failure::Input("invalid TC method or flags"));
        }
        let spectra = fixed(input.spectra)?;
        let sensitivity = fixed::<243>(input.sensitivity_rgb)?;
        let sensitivity_rgb = &*sensitivity.as_ptr().cast::<[[f32; 3]; 81]>();
        let reference_illuminant = fixed(input.reference_illuminant)?;
        let method = if input.method == 0 {
            reconstruction::Method::Hanatos {
                spectral_blur: input.spectral_blur,
                surface_rgb: if input.apply_surface != 0 {
                    let surface = fixed::<45>(input.surface_rgb)?;
                    Some(&*surface.as_ptr().cast::<[[f32; 15]; 3]>())
                } else {
                    None
                },
            }
        } else {
            reconstruction::Method::Arctic
        };
        let input_hull = if input.compression_active != 0 && input.hull_available != 0 {
            let xy = fixed::<2050>(input.hull_xy)?;
            Some(gamut::InputHull::new(
                input.hull_center_xy,
                &*xy.as_ptr().cast::<[[f32; 2]; 1025]>(),
            ))
        } else {
            None
        };
        Ok(reconstruction::Input {
            spectra,
            sensitivity_rgb,
            reference_illuminant,
            projection_white_xyz: input.projection_white_xyz,
            method,
            input_compression_active: input.compression_active != 0,
            input_hull,
        })
    }
}
#[cfg(feature = "test-support")]
thread_local! {
    static TC_FAULT:std::cell::Cell<Option<(u32,u32,u32)>>=const{std::cell::Cell::new(None)};
    static TC_CONSUMED:std::cell::Cell<Option<(u32,u32)>>=const{std::cell::Cell::new(None)};
}
#[cfg(feature = "test-support")]
pub(crate) fn arm_fault(operation: u32, index: u32, fault: u32) -> FjStatus {
    clear_fault();
    let valid = (1..=2).contains(&operation) && index != 0 && (1..=4).contains(&fault);
    if valid {
        TC_FAULT.set(Some((operation, index, fault)));
    }
    FjStatus {
        category: if valid {
            FJ_STATUS_SUCCESS
        } else {
            FJ_STATUS_UNSUPPORTED_INPUT
        },
        api: FJ_API_NONE,
        native_code: 0,
    }
}
#[cfg(feature = "test-support")]
pub(crate) fn clear_fault() {
    TC_FAULT.set(None);
    TC_CONSUMED.set(None);
}
#[cfg(feature = "test-support")]
pub(crate) fn take_consumed() -> Option<(u32, u32)> {
    TC_CONSUMED.take()
}
pub(crate) fn fault(operation: u32) -> Result<(), Failure> {
    #[cfg(feature = "test-support")]
    if let Some((armed, index, fault)) = TC_FAULT.get() {
        if armed != operation {
            return Ok(());
        }
        if index > 1 {
            TC_FAULT.set(Some((armed, index - 1, fault)));
            return Ok(());
        }
        TC_FAULT.set(None);
        TC_CONSUMED.set(Some((operation, fault)));
        return injected(fault);
    }
    #[cfg(not(feature = "test-support"))]
    let _ = operation;
    Ok(())
}
#[cfg(feature = "test-support")]
pub(crate) fn injected(fault: u32) -> Result<(), Failure> {
    match fault {
        1 => Err(Failure::Input("TC production boundary fault")),
        2 => panic!("TC production boundary fault"),
        3 => Err(Failure::FilmTc(reconstruction::TcError::AllocationFailure)),
        4 => Err(Failure::FilmTc(
            reconstruction::TcError::NonfiniteIntegratedSample,
        )),
        _ => unreachable!("closed TC fault"),
    }
}
/// # Safety
/// Input/consumed spans are live immutable storage. Output is an aligned,
/// initialized empty/exclusive ownership record, disjoint from input/error.
/// A successful record transfers its one exact Vec token to the caller.
pub(crate) unsafe fn build_call(
    input_record: *const FjFilmTcLutInput,
    out: *mut FjOwnedFilmTcLut,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Record accesses follow the documented caller obligations; no live
    // token is overwritten. RAII retains the Vec until the final infallible transfer.
    unsafe {
        asset_illuminant::run(error, || {
            if !aligned(out) || !(*out).is_empty() {
                return Err(Failure::Input(
                    "TC output must be an empty ownership record",
                ));
            }
            out.write(FjOwnedFilmTcLut::empty());
            let input = input(input_record)?;
            fault()?;
            let completed = reconstruction::build_tc_lut(input).map_err(Failure::FilmTc)?;
            let (data, count, capacity) = completed.into_samples().into_raw_parts();
            out.write(FjOwnedFilmTcLut {
                samples: FjFloatSpan { data, count },
                capacity,
            });
            #[cfg(feature = "test-support")]
            TC_TRANSFERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        })
    }
}
/// # Safety
/// The LUT and XYZ authorize live initialized immutable fixed extents; RGB and
/// error backing are exclusive/disjoint complete storage until return.
pub(crate) unsafe fn sample_call(
    lut: FjFloatSpan,
    xyz: *const f32,
    out: *mut f32,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Checks precede array borrows; output clearing/filling uses the
    // caller-authorized exclusive three-float extent. No borrow escapes.
    unsafe {
        let valid_out = aligned(out) && (out as usize).checked_add(3 * size_of::<f32>()).is_some();
        if valid_out {
            out.cast::<[f32; 3]>().write([0.0; 3]);
        }
        asset_illuminant::run(error, || {
            if !valid_out {
                return Err(Failure::Input("invalid TC sample output"));
            }
            let samples = fixed(lut)?;
            let xyz = fixed::<3>(FjFloatSpan {
                data: xyz,
                count: 3,
            })?;
            fault()?;
            let result = reconstruction::sample_tc_lut(samples, *xyz).map_err(Failure::FilmTc)?;
            out.cast::<[f32; 3]>().write(result);
            Ok(())
        })
    }
}
/// # Safety
/// The exclusive aligned record is the unmodified one live allocation token
/// returned by this module, or all-zero empty. All views/use have ended; no copied
/// live token, corrupt layout, foreign allocator or overlapping release is valid.
pub(crate) unsafe fn release_call(owned: *mut FjOwnedFilmTcLut) -> FjStatus {
    if !aligned(owned) {
        return FjStatus {
            category: FJ_STATUS_UNSUPPORTED_INPUT,
            api: FJ_API_NONE,
            native_code: 0,
        };
    }
    // SAFETY: Caller transfers the sole exact Vec allocation/layout. Clearing the
    // consumed record precedes its infallible f32 deallocation; no allocator,
    // diagnostics, retry, GPU or TLS ownership policy is introduced.
    unsafe {
        let record = owned.read();
        owned.write(FjOwnedFilmTcLut::empty());
        if !record.is_empty() {
            drop(Vec::from_raw_parts(
                record.samples.data.cast_mut(),
                record.samples.count,
                record.capacity,
            ));
            #[cfg(feature = "test-support")]
            TC_RELEASES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    FjStatus {
        category: FJ_STATUS_SUCCESS,
        api: FJ_API_NONE,
        native_code: 0,
    }
}

const _: () = {
    assert!(size_of::<FjFloatSpan>() == 16 && align_of::<FjFloatSpan>() == 8);
    assert!(size_of::<FjFilmTcLutInput>() == 120 && align_of::<FjFilmTcLutInput>() == 8);
    assert!(size_of::<FjOwnedFilmTcLut>() == 24 && align_of::<FjOwnedFilmTcLut>() == 8);
    assert!(std::mem::offset_of!(FjFloatSpan, data) == 0);
    assert!(std::mem::offset_of!(FjFloatSpan, count) == 8);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, spectra) == 0);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, sensitivity_rgb) == 16);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, reference_illuminant) == 32);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, projection_white_xyz) == 48);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, spectral_blur) == 60);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, method) == 64);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, apply_surface) == 68);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, surface_rgb) == 72);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, compression_active) == 88);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, hull_available) == 92);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, hull_center_xy) == 96);
    assert!(std::mem::offset_of!(FjFilmTcLutInput, hull_xy) == 104);
    assert!(std::mem::offset_of!(FjOwnedFilmTcLut, samples) == 0);
    assert!(std::mem::offset_of!(FjOwnedFilmTcLut, capacity) == 16);
};
