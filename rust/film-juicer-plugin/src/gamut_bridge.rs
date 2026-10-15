//! Synchronous fixed input-hull projection. The native Library owns its cache.

use film_juicer_core::color;
use film_juicer_core::gamut;

use crate::asset_bridge::Failure;
use crate::asset_illuminant;
use crate::cuda::sys::{FjErrorBuffer, FjFloatSpan, FjStatus};

#[repr(C)]
pub(crate) struct FjInputHullSource {
    pub x: FjFloatSpan,
    pub y: FjFloatSpan,
    pub z: FjFloatSpan,
    pub d65: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjInputHull {
    pub center_xy: [f32; 2],
    pub xy: [[f32; 2]; 1025],
    pub hash: u64,
}
impl Default for FjInputHull {
    fn default() -> Self {
        Self {
            center_xy: [0.0; 2],
            xy: [[0.0; 2]; 1025],
            hash: 0,
        }
    }
}
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<FjInputHullSource>() == 64 && align_of::<FjInputHullSource>() == 8);
    assert!(offset_of!(FjInputHullSource, x) == 0);
    assert!(offset_of!(FjInputHullSource, y) == 16);
    assert!(offset_of!(FjInputHullSource, z) == 32);
    assert!(offset_of!(FjInputHullSource, d65) == 48);
    assert!(size_of::<FjInputHull>() == 8216 && align_of::<FjInputHull>() == 8);
    assert!(offset_of!(FjInputHull, center_xy) == 0);
    assert!(offset_of!(FjInputHull, xy) == 8);
    assert!(offset_of!(FjInputHull, hash) == 8208);
};
fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}
/// # Safety
/// The caller retains initialized immutable storage, disjoint from outputs,
/// through synchronous return. A checked address alone cannot prove its lifetime.
unsafe fn spectrum<'a>(span: FjFloatSpan) -> Result<&'a [f32; 81], Failure> {
    let bytes = span
        .count
        .checked_mul(size_of::<f32>())
        .ok_or(Failure::Input("input hull extent overflow"))?;
    if span.count != 81
        || bytes > isize::MAX as usize
        || (span.data as usize).checked_add(bytes).is_none()
        || !aligned(span.data)
    {
        return Err(Failure::Input("invalid input hull spectrum"));
    }
    // SAFETY: Exact fixed layout/addressable extent and caller-proven lifetime.
    Ok(unsafe { &*span.data.cast::<[f32; 81]>() })
}

/// # Safety
/// Required initialized records and spans are live and aligned. Writable output
/// and diagnostic storage are mutually disjoint and disjoint from every input.
pub(crate) unsafe fn hull_call(
    source: *const FjInputHullSource,
    out: *mut FjInputHull,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Only caller-authorized output clears. Diagnostic validation and
    // panic containment precede checked immutable borrows; no pointer escapes.
    unsafe {
        let valid_out = aligned(out);
        if valid_out {
            out.write(FjInputHull::default());
        }
        asset_illuminant::run(error, || {
            if !valid_out || !aligned(source) {
                return Err(Failure::Input("invalid input hull records"));
            }
            let source = &*source;
            let input = gamut::HullInput {
                x: spectrum(source.x)?,
                y: spectrum(source.y)?,
                z: spectrum(source.z)?,
                d65: spectrum(source.d65)?,
            };
            fault()?;
            let hull = gamut::build_input_hull(input).map_err(Failure::Hull)?;
            out.write(FjInputHull {
                center_xy: *hull.center(),
                xy: *hull.xy(),
                hash: hull.hash(),
            });
            Ok(())
        })
    }
}

fn production_fault() -> Result<(), Failure> {
    #[cfg(feature = "test-support")]
    return fixtures::inject(false);
    #[cfg(not(feature = "test-support"))]
    Ok(())
}
// FJ_TEMP_BRIDGE: completed input hull into the sole native cache; remove S4.E.
/// # Safety
/// The required fixed source/output/error storage obeys the legacy header.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_gamut_input_hull(
    source: *const FjInputHullSource,
    out: *mut FjInputHull,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: The checked synchronous operation owns no retained foreign data.
    unsafe { hull_call(source, out, error, production_fault) }
}

#[cfg(feature = "test-support")]
mod fixtures {
    use super::*;
    use crate::cuda::sys::{FJ_API_NONE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT};
    thread_local! {
        static FAULT: std::cell::Cell<Option<(bool,u32)>> = const { std::cell::Cell::new(None) };
        static CONSUMED: std::cell::Cell<Option<(bool,u32)>> = const { std::cell::Cell::new(None) };
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_gamut_arm_fault(facade: u32, fault: u32) -> FjStatus {
        FAULT.set(None);
        CONSUMED.set(None);
        let valid = facade <= 1 && (1..=6).contains(&fault);
        if valid {
            FAULT.set(Some((facade != 0, fault)));
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
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_gamut_clear_fault() {
        FAULT.set(None);
        CONSUMED.set(None);
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_gamut_fault_consumed(facade: u32, fault: u32) -> u32 {
        u32::from(CONSUMED.take() == Some((facade != 0, fault)))
    }
    pub(super) fn inject(facade: bool) -> Result<(), Failure> {
        let Some((selected, fault)) = FAULT.get() else {
            return Ok(());
        };
        if selected != facade {
            return Ok(());
        }
        FAULT.set(None);
        CONSUMED.set(Some((selected, fault)));
        match fault {
            1 => Err(Failure::Input("input hull boundary fault")),
            2 => panic!("input hull boundary fault"),
            3 => Err(Failure::Hull(gamut::HullError::LocusAndCenter)),
            4 => Err(Failure::Hull(gamut::HullError::RayIntersections)),
            5 => Err(Failure::Hull(gamut::HullError::InscribedScale)),
            6 => Err(Failure::Hull(gamut::HullError::Identity)),
            _ => unreachable!("closed input hull fault"),
        }
    }
    /// # Safety
    /// The same synchronous fixed borrow and exclusive output contract applies.
    #[unsafe(no_mangle)]
    unsafe extern "C" fn fj_test_gamut_input_hull(
        source: *const FjInputHullSource,
        out: *mut FjInputHull,
        error: *mut FjErrorBuffer,
    ) -> FjStatus {
        // SAFETY: This invokes the shared checked core operation, not the legacy export.
        unsafe { hull_call(source, out, error, || inject(true)) }
    }
}

#[repr(C)]
pub(crate) struct FjOutputGamutSource {
    pub output_space: u32,
    pub rgb_to_xyz: [f32; 9],
    pub xyz_to_rgb: [f32; 9],
    pub white_xyz: [f32; 3],
    pub d65_white_xyz: [f32; 3],
}
#[derive(Default)]
#[repr(C)]
pub(crate) struct FjOutputGamutTransform {
    pub native_rgb_to_d65_xyz: [f32; 9],
    pub d65_xyz_to_native_rgb: [f32; 9],
    pub output_space: u32,
    pub hash: u64,
    pub contract_hash: u64,
}
#[repr(C)]
pub(crate) struct FjOutputBoundarySource {
    pub output_space: u32,
    pub d65_xyz_to_native_rgb: [f32; 9],
    pub transform_hash: u64,
}
#[derive(Default)]
#[repr(C)]
pub(crate) struct FjOutputBoundaryIdentity {
    pub transform_hash: u64,
    pub contract_hash: u64,
    pub hash: u64,
}
#[derive(Default)]
#[repr(C)]
pub(crate) struct FjOutputBoundaryFailure {
    pub requirement: u32,
    pub lightness_index: u32,
    pub hue_index: u32,
}
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<FjOutputGamutSource>() == 100 && align_of::<FjOutputGamutSource>() == 4);
    assert!(
        offset_of!(FjOutputGamutSource, output_space) == 0
            && offset_of!(FjOutputGamutSource, rgb_to_xyz) == 4
    );
    assert!(
        offset_of!(FjOutputGamutSource, xyz_to_rgb) == 40
            && offset_of!(FjOutputGamutSource, white_xyz) == 76
            && offset_of!(FjOutputGamutSource, d65_white_xyz) == 88
    );
    assert!(size_of::<FjOutputGamutTransform>() == 96 && align_of::<FjOutputGamutTransform>() == 8);
    assert!(
        offset_of!(FjOutputGamutTransform, native_rgb_to_d65_xyz) == 0
            && offset_of!(FjOutputGamutTransform, d65_xyz_to_native_rgb) == 36
    );
    assert!(
        offset_of!(FjOutputGamutTransform, output_space) == 72
            && offset_of!(FjOutputGamutTransform, hash) == 80
            && offset_of!(FjOutputGamutTransform, contract_hash) == 88
    );
    assert!(size_of::<FjOutputBoundarySource>() == 48 && align_of::<FjOutputBoundarySource>() == 8);
    assert!(
        offset_of!(FjOutputBoundarySource, output_space) == 0
            && offset_of!(FjOutputBoundarySource, d65_xyz_to_native_rgb) == 4
            && offset_of!(FjOutputBoundarySource, transform_hash) == 40
    );
    assert!(
        size_of::<FjOutputBoundaryIdentity>() == 24 && align_of::<FjOutputBoundaryIdentity>() == 8
    );
    assert!(
        offset_of!(FjOutputBoundaryIdentity, transform_hash) == 0
            && offset_of!(FjOutputBoundaryIdentity, contract_hash) == 8
            && offset_of!(FjOutputBoundaryIdentity, hash) == 16
    );
    assert!(
        size_of::<FjOutputBoundaryFailure>() == 12 && align_of::<FjOutputBoundaryFailure>() == 4
    );
    assert!(
        offset_of!(FjOutputBoundaryFailure, requirement) == 0
            && offset_of!(FjOutputBoundaryFailure, lightness_index) == 4
            && offset_of!(FjOutputBoundaryFailure, hue_index) == 8
    );
};
fn output_status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: crate::cuda::sys::FJ_API_NONE,
        native_code: 0,
    }
}
/// # Safety
/// Present diagnostics authorize live exclusive record/storage with initialized
/// capacity and writable length; data need be initialized only at positive capacity.
unsafe fn output_run(
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    use crate::cuda::sys::{FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT};
    if !error.is_null() {
        if !aligned(error) {
            return output_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: Record extent/alignment checked; caller supplies initialized capacity.
        let capacity = unsafe { std::ptr::addr_of!((*error).capacity).read() };
        // SAFETY: Independently valid record authorizes its length field only.
        unsafe {
            std::ptr::addr_of_mut!((*error).length).write(0);
        }
        if capacity > 0 {
            // SAFETY: A positive capacity requires an initialized data field.
            let data = unsafe { std::ptr::addr_of!((*error).data).read() };
            if data.is_null()
                || capacity > isize::MAX as usize
                || (data as usize).checked_add(capacity).is_none()
            {
                return output_status(FJ_STATUS_UNSUPPORTED_INPUT);
            }
        }
    }
    // SAFETY: Optional record and backing preconditions established before the
    // shared writer. Zero capacity never reads data or constructs a backing slice.
    match unsafe { crate::asset_bridge::run(error, operation) } {
        Ok(()) => output_status(FJ_STATUS_SUCCESS),
        Err(status) => status,
    }
}
/// # Safety
/// Source/output/error obey the synchronous, disjoint storage contract in the header.
unsafe fn output_transform_call(
    source: *const FjOutputGamutSource,
    out: *mut FjOutputGamutTransform,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    let valid_out = aligned(out);
    if valid_out {
        // SAFETY: The caller authorizes this independently checked fixed output.
        unsafe {
            out.write(FjOutputGamutTransform::default());
        }
    }
    // SAFETY: Checked optional containment owns every source borrow and construction.
    unsafe {
        output_run(error, || {
            if !valid_out || !aligned(source) {
                return Err(Failure::Input("invalid output gamut records"));
            }
            let source = &*source;
            let space = color::OutputSpace::try_from(source.output_space)
                .map_err(|()| Failure::Input("invalid output gamut space"))?;
            fault()?;
            let transform = gamut::output_transform(gamut::TransformInput {
                space,
                rgb_to_xyz: &source.rgb_to_xyz,
                xyz_to_rgb: &source.xyz_to_rgb,
                white_xyz: &source.white_xyz,
                d65_white_xyz: &source.d65_white_xyz,
            })
            .map_err(Failure::OutputTransform)?;
            out.write(FjOutputGamutTransform {
                native_rgb_to_d65_xyz: *transform.native_rgb_to_d65_xyz(),
                d65_xyz_to_native_rgb: *transform.d65_xyz_to_native_rgb(),
                output_space: transform.space() as u32,
                hash: transform.hash(),
                contract_hash: transform.contract_hash(),
            });
            Ok(())
        })
    }
}
fn boundary_failure(error: gamut::BoundaryError) -> FjOutputBoundaryFailure {
    use gamut::BoundaryError;
    let requirement = match error {
        BoundaryError::Transform => 1,
        BoundaryError::C2Outside { .. } => 2,
        BoundaryError::FiniteFirstExit => 3,
        BoundaryError::Identity => 4,
        BoundaryError::AllocationFailure => 0,
    };
    match error {
        BoundaryError::C2Outside {
            lightness_index,
            hue_index,
            ..
        } => FjOutputBoundaryFailure {
            requirement,
            lightness_index,
            hue_index,
        },
        _ => FjOutputBoundaryFailure {
            requirement,
            ..FjOutputBoundaryFailure::default()
        },
    }
}
fn cmax_extent(cmax: *mut f32, count: usize) -> bool {
    count == gamut::OUTPUT_BOUNDARY_VALUE_COUNT
        && aligned(cmax)
        && count.checked_mul(size_of::<f32>()).is_some_and(|bytes| {
            bytes <= isize::MAX as usize && (cmax as usize).checked_add(bytes).is_some()
        })
}
/// # Safety
/// Exact Cmax writable storage, fixed outputs, source and optional diagnostics
/// obey the mutually disjoint synchronous extents documented by the legacy header.
unsafe fn output_boundary_call(
    source: *const FjOutputBoundarySource,
    cmax: *mut f32,
    count: usize,
    identity: *mut FjOutputBoundaryIdentity,
    failure: *mut FjOutputBoundaryFailure,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    let valid_identity = aligned(identity);
    let valid_failure = aligned(failure);
    let valid_cmax = cmax_extent(cmax, count);
    // SAFETY: Clear only independently valid caller-authorized writable extents,
    // before diagnostic rejection. Invalid counts authorize no arbitrary write.
    unsafe {
        if valid_identity {
            identity.write(FjOutputBoundaryIdentity::default());
        }
        if valid_failure {
            failure.write(FjOutputBoundaryFailure::default());
        }
        if valid_cmax {
            std::ptr::write_bytes(cmax, 0, count);
        }
        output_run(error, || {
            if !valid_identity || !valid_failure || !valid_cmax || !aligned(source) {
                return Err(Failure::Input(
                    "invalid output boundary records or Cmax extent",
                ));
            }
            let source = &*source;
            let space = color::OutputSpace::try_from(source.output_space)
                .map_err(|()| Failure::Input("invalid output boundary space"))?;
            let input = gamut::BoundaryInput {
                space,
                d65_xyz_to_native_rgb: &source.d65_xyz_to_native_rgb,
                transform_hash: source.transform_hash,
            };
            #[cfg(feature = "test-support")]
            let construction_started = std::time::Instant::now();
            let result = fault().and_then(|()| {
                gamut::build_output_boundary(input).map_err(Failure::OutputBoundary)
            });
            let boundary = match result {
                Ok(boundary) => boundary,
                Err(Failure::OutputBoundary(error)) => {
                    failure.write(boundary_failure(error));
                    return Err(Failure::OutputBoundary(error));
                }
                Err(error) => return Err(error),
            };
            #[cfg(feature = "test-support")]
            let construction_ns = construction_started.elapsed().as_nanos() as u64;
            #[cfg(feature = "test-support")]
            let copy_started = std::time::Instant::now();
            std::ptr::copy_nonoverlapping(boundary.cmax().as_ptr(), cmax, count);
            #[cfg(feature = "test-support")]
            output_fixtures::copied(
                &boundary,
                construction_ns,
                copy_started.elapsed().as_nanos() as u64,
            );
            identity.write(FjOutputBoundaryIdentity {
                transform_hash: boundary.transform_hash(),
                contract_hash: boundary.contract_hash(),
                hash: boundary.hash(),
            });
            Ok(())
        })
    }
}
fn output_fault(operation: u32) -> Result<(), Failure> {
    #[cfg(feature = "test-support")]
    return output_fixtures::inject(operation, false);
    #[cfg(not(feature = "test-support"))]
    {
        let _ = operation;
        Ok(())
    }
}
// FJ_TEMP_BRIDGE: complete output transform into native recipes; remove S4.E.
/// # Safety
/// The synchronous fixed record/optional diagnostic contract in the header applies.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_gamut_output_transform(
    source: *const FjOutputGamutSource,
    out: *mut FjOutputGamutTransform,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Shared checked operation scopes all foreign accesses and containment.
    unsafe { output_transform_call(source, out, error, || output_fault(1)) }
}
// FJ_TEMP_BRIDGE: complete boundary copied into native cache carrier; remove S4.E.
/// # Safety
/// The synchronous fixed records/exact Cmax/optional diagnostic contract applies.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_gamut_output_boundary(
    source: *const FjOutputBoundarySource,
    cmax: *mut f32,
    count: usize,
    identity: *mut FjOutputBoundaryIdentity,
    failure: *mut FjOutputBoundaryFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: No foreign pointer/result owner escapes the checked synchronous call.
    unsafe {
        output_boundary_call(source, cmax, count, identity, failure, error, || {
            output_fault(2)
        })
    }
}

#[cfg(feature = "test-support")]
mod output_fixtures {
    use super::*;
    thread_local! {
        static FAULT:std::cell::Cell<Option<(u32,bool,u32)>>=const{std::cell::Cell::new(None)};
        static CONSUMED:std::cell::Cell<Option<(u32,bool,u32)>>=const{std::cell::Cell::new(None)};
        static VISITS:std::cell::Cell<[u64;2]>=const{std::cell::Cell::new([0;2])};
    }
    thread_local! { static COPY_METRICS:std::cell::Cell<[u64;5]>=const{std::cell::Cell::new([0;5])}; }
    pub(super) fn copied(boundary: &gamut::OutputBoundary, construction_ns: u64, copy_ns: u64) {
        let mut totals = COPY_METRICS.get();
        totals[0] += 1;
        totals[1] += gamut::fixtures::output_allocation_bytes(boundary) as u64;
        totals[2] += std::mem::size_of_val(boundary.cmax()) as u64;
        totals[3] += construction_ns;
        totals[4] += copy_ns;
        COPY_METRICS.set(totals);
    }
    /// # Safety
    /// Caller authorizes five aligned exclusive writable u64 words.
    #[unsafe(no_mangle)]
    unsafe extern "C" fn fj_test_output_gamut_copy_metrics(out: *mut u64, reset: u32) -> FjStatus {
        if !aligned(out.cast::<[u64; 5]>()) || reset > 1 {
            return output_status(crate::cuda::sys::FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: Exact fixed extent checked; caller supplies writable exclusive storage.
        unsafe {
            out.cast::<[u64; 5]>().write(COPY_METRICS.get());
        }
        if reset != 0 {
            COPY_METRICS.set([0; 5]);
        }
        output_status(crate::cuda::sys::FJ_STATUS_SUCCESS)
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_output_gamut_arm_fault(
        operation: u32,
        facade: u32,
        fault: u32,
    ) -> FjStatus {
        FAULT.set(None);
        CONSUMED.set(None);
        let valid = (1..=2).contains(&operation)
            && facade <= 1
            && (1..=if operation == 1 { 5 } else { 7 }).contains(&fault);
        if valid {
            FAULT.set(Some((operation, facade != 0, fault)));
        }
        output_status(if valid {
            crate::cuda::sys::FJ_STATUS_SUCCESS
        } else {
            crate::cuda::sys::FJ_STATUS_UNSUPPORTED_INPUT
        })
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_output_gamut_clear_fault() {
        FAULT.set(None);
        CONSUMED.set(None);
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_output_gamut_fault_consumed(
        operation: u32,
        facade: u32,
        fault: u32,
    ) -> u32 {
        u32::from(CONSUMED.take() == Some((operation, facade != 0, fault)))
    }
    #[unsafe(no_mangle)]
    extern "C" fn fj_test_output_gamut_visits(operation: u32) -> u64 {
        if (1..=2).contains(&operation) {
            VISITS.get()[(operation - 1) as usize]
        } else {
            0
        }
    }
    pub(super) fn inject(operation: u32, facade: bool) -> Result<(), Failure> {
        let mut visits = VISITS.get();
        visits[(operation - 1) as usize] += 1;
        VISITS.set(visits);
        let Some((selected, path, fault)) = FAULT.get() else {
            return Ok(());
        };
        if selected != operation || path != facade {
            return Ok(());
        }
        FAULT.set(None);
        CONSUMED.set(Some((selected, path, fault)));
        match fault {
            1 => Err(Failure::Input("output gamut boundary fault")),
            2 => panic!("output gamut boundary fault"),
            3 => Err(Failure::InjectedAllocation),
            4 | 5 if operation == 1 => Err(Failure::OutputTransform(if fault == 4 {
                gamut::TransformError::FiniteNativeWhiteCat16
            } else {
                gamut::TransformError::Identity
            })),
            4 => Err(Failure::OutputBoundary(gamut::BoundaryError::Transform)),
            5 => Err(Failure::OutputBoundary(gamut::BoundaryError::C2Outside {
                space: color::OutputSpace::Srgb,
                lightness_index: 3,
                hue_index: 7,
            })),
            6 => Err(Failure::OutputBoundary(
                gamut::BoundaryError::FiniteFirstExit,
            )),
            7 => Err(Failure::OutputBoundary(gamut::BoundaryError::Identity)),
            _ => unreachable!("closed operation-specific output gamut fault"),
        }
    }
    /// # Safety
    /// The same fixed record/optional diagnostic contract applies.
    #[unsafe(no_mangle)]
    pub(super) unsafe extern "C" fn fj_test_gamut_output_transform(
        source: *const FjOutputGamutSource,
        out: *mut FjOutputGamutTransform,
        error: *mut FjErrorBuffer,
    ) -> FjStatus {
        // SAFETY: Direct facade shares checked construction, never the legacy export.
        unsafe { output_transform_call(source, out, error, || inject(1, true)) }
    }
    /// # Safety
    /// The same fixed/exact Cmax/optional diagnostic contract applies.
    #[unsafe(no_mangle)]
    pub(super) unsafe extern "C" fn fj_test_gamut_output_boundary(
        source: *const FjOutputBoundarySource,
        cmax: *mut f32,
        count: usize,
        identity: *mut FjOutputBoundaryIdentity,
        failure: *mut FjOutputBoundaryFailure,
        error: *mut FjErrorBuffer,
    ) -> FjStatus {
        // SAFETY: Direct facade shares checked construction, never the legacy export.
        unsafe {
            output_boundary_call(source, cmax, count, identity, failure, error, || {
                inject(2, true)
            })
        }
    }
}

#[cfg(test)]
mod output_tests {
    use super::*;
    #[test]
    fn zero_capacity_never_reads_uninitialized_data_field() {
        use std::mem::MaybeUninit;
        let source = FjOutputGamutSource {
            output_space: 0,
            rgb_to_xyz: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            xyz_to_rgb: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            white_xyz: [0.95, 1.0, 1.09],
            d65_white_xyz: [0.95, 1.0, 1.09],
        };
        let mut transform = FjOutputGamutTransform::default();
        let mut record = MaybeUninit::<FjErrorBuffer>::uninit();
        let error = record.as_mut_ptr();
        // SAFETY: Only required capacity/length fields are initialized; the local
        // edge and shared containment must never read data at zero capacity.
        unsafe {
            std::ptr::addr_of_mut!((*error).capacity).write(0);
            std::ptr::addr_of_mut!((*error).length).write(73);
            assert_eq!(
                fj_legacy_gamut_output_transform(&source, &mut transform, error).category,
                crate::cuda::sys::FJ_STATUS_SUCCESS
            );
            assert_eq!(std::ptr::addr_of!((*error).length).read(), 0);
            let boundary = FjOutputBoundarySource {
                output_space: 0,
                d65_xyz_to_native_rgb: transform.d65_xyz_to_native_rgb,
                transform_hash: transform.hash,
            };
            let mut identity = FjOutputBoundaryIdentity::default();
            let mut failure = FjOutputBoundaryFailure::default();
            let mut table = vec![7.0; gamut::OUTPUT_BOUNDARY_VALUE_COUNT];
            assert_eq!(
                fj_legacy_gamut_output_boundary(
                    &boundary,
                    table.as_mut_ptr(),
                    table.len(),
                    &mut identity,
                    &mut failure,
                    error
                )
                .category,
                crate::cuda::sys::FJ_STATUS_SUCCESS
            );
            assert_eq!(std::ptr::addr_of!((*error).length).read(), 0);
            #[cfg(feature = "test-support")]
            {
                assert_eq!(
                    output_fixtures::fj_test_gamut_output_transform(&source, &mut transform, error)
                        .category,
                    crate::cuda::sys::FJ_STATUS_SUCCESS
                );
                assert_eq!(
                    output_fixtures::fj_test_gamut_output_boundary(
                        &boundary,
                        table.as_mut_ptr(),
                        table.len(),
                        &mut identity,
                        &mut failure,
                        error
                    )
                    .category,
                    crate::cuda::sys::FJ_STATUS_SUCCESS
                );
            }
        }
    }
}
