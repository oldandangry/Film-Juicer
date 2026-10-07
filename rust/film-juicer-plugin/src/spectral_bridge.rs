//! Synchronous fixed spectral value edge; native payload ownership remains native.

use film_juicer_core::spectral::{self, Baseline, Observer, WhiteError};

use crate::asset_bridge::Failure;
use crate::asset_illuminant;
#[cfg(feature = "test-support")]
use crate::cuda::sys::{FJ_API_NONE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT};
use crate::cuda::sys::{FjErrorBuffer, FjFloatSpan, FjStatus};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FjSpectralObserver {
    pub x: FjFloatSpan,
    pub y: FjFloatSpan,
    pub z: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjSpectralInput {
    pub dyes_cmy: FjFloatSpan,
    pub observer: FjSpectralObserver,
    pub illuminant: FjFloatSpan,
    pub baseline_min: FjFloatSpan,
    pub baseline_mid: FjFloatSpan,
    pub illuminant_hash: u64,
}
#[repr(C)]
pub(crate) struct FjSpectralTables {
    pub lambda_nm: [f32; 81],
    pub illuminant: [f32; 81],
    pub observer_xyz: [[f32; 81]; 3],
    pub weighted_xyz: [[f32; 81]; 3],
    pub dyes_cmy: [[f32; 81]; 3],
    pub baseline_min: [f32; 81],
    pub baseline_mid: [f32; 81],
    pub delta_lambda: f32,
    pub inv_yn: f32,
    pub white_xyz: [f32; 3],
    pub reference_white_xyz: [f32; 3],
    pub has_baseline: u32,
    pub illuminant_hash: u64,
    pub tables_hash: u64,
}
impl Default for FjSpectralTables {
    fn default() -> Self {
        Self {
            lambda_nm: [0.0; 81],
            illuminant: [0.0; 81],
            observer_xyz: [[0.0; 81]; 3],
            weighted_xyz: [[0.0; 81]; 3],
            dyes_cmy: [[0.0; 81]; 3],
            baseline_min: [0.0; 81],
            baseline_mid: [0.0; 81],
            delta_lambda: 0.0,
            inv_yn: 0.0,
            white_xyz: [0.0; 3],
            reference_white_xyz: [0.0; 3],
            has_baseline: 0,
            illuminant_hash: 0,
            tables_hash: 0,
        }
    }
}
#[repr(C)]
pub(crate) struct FjSpectralWhiteInput {
    pub observer: FjSpectralObserver,
    pub illuminant: FjFloatSpan,
}
#[repr(C)]
#[derive(Default)]
pub(crate) struct FjSpectralWhite {
    pub normalization: f32,
    pub white_xyz: [f32; 3],
    pub white_xy: [f32; 2],
    pub hash: u64,
}
#[repr(C)]
#[derive(Default)]
pub(crate) struct FjSpectralWhiteFailure {
    pub reason: u32,
    pub luminance_sum: f64,
}
#[repr(C)]
pub(crate) struct FjSpectralSInput {
    pub weighted_x: FjFloatSpan,
    pub weighted_y: FjFloatSpan,
    pub weighted_z: FjFloatSpan,
}
#[repr(C)]
#[derive(Default)]
pub(crate) struct FjSpectralInverse {
    pub matrix: [f32; 9],
}

fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}
/// # Safety
/// Caller authorizes the complete writable aligned record, disjoint from inputs.
unsafe fn clear<T: Default>(out: *mut T) -> bool {
    if !aligned(out) {
        return false;
    }
    // SAFETY: The caller supplies an exclusive writable complete record.
    unsafe { out.write(T::default()) };
    true
}
/// # Safety
/// Caller retains this initialized immutable extent through the synchronous call.
unsafe fn fixed<'a, const N: usize>(span: FjFloatSpan) -> Result<&'a [f32; N], Failure> {
    let bytes = span
        .count
        .checked_mul(size_of::<f32>())
        .ok_or(Failure::Input("spectral extent overflow"))?;
    if span.count != N
        || bytes > isize::MAX as usize
        || (span.data as usize).checked_add(bytes).is_none()
        || !aligned(span.data)
    {
        return Err(Failure::Input("invalid fixed spectral span"));
    }
    // SAFETY: Exact checked count/layout and caller-authorized immutable lifetime.
    Ok(unsafe { &*span.data.cast::<[f32; N]>() })
}
/// # Safety
/// All spans obey fixed's initialized immutable extent and lifetime contract.
unsafe fn observer<'a>(input: FjSpectralObserver) -> Result<Observer<'a>, Failure> {
    // SAFETY: Each span is checked before constructing its scoped source borrow.
    unsafe {
        Ok(Observer {
            x: fixed(input.x)?,
            y: fixed(input.y)?,
            z: fixed(input.z)?,
        })
    }
}
/// # Safety
/// The input and all its initialized spans remain immutable until return.
unsafe fn table_input<'a>(input: *const FjSpectralInput) -> Result<spectral::Input<'a>, Failure> {
    if !aligned(input) {
        return Err(Failure::Input("invalid spectral table input"));
    }
    // SAFETY: The caller authorizes the input record and source extents; fixed
    // verifies their count/address/layout before any array reference is formed.
    unsafe {
        let input = &*input;
        let dyes = fixed::<243>(input.dyes_cmy)?;
        // [f32;243] and [[f32;3];81] have identical size/alignment and no padding.
        let dyes_cmy = &*dyes.as_ptr().cast::<[[f32; 3]; 81]>();
        let absent = |span: FjFloatSpan| span.count == 0 && span.data.is_null();
        let baseline = if absent(input.baseline_min) {
            if !absent(input.baseline_mid) {
                return Err(Failure::Input("spectral midpoint requires minimum"));
            }
            None
        } else {
            Some(Baseline {
                minimum: fixed(input.baseline_min)?,
                midpoint: if absent(input.baseline_mid) {
                    None
                } else {
                    Some(fixed(input.baseline_mid)?)
                },
            })
        };
        Ok(spectral::Input {
            dyes_cmy,
            observer: observer(input.observer)?,
            illuminant: fixed(input.illuminant)?,
            baseline,
            illuminant_hash: input.illuminant_hash,
        })
    }
}
fn project_tables(t: &spectral::Tables) -> FjSpectralTables {
    FjSpectralTables {
        lambda_nm: *t.lambda_nm(),
        illuminant: *t.illuminant(),
        observer_xyz: *t.observer_xyz(),
        weighted_xyz: *t.weighted_xyz(),
        dyes_cmy: *t.dyes_cmy(),
        baseline_min: *t.baseline_min(),
        baseline_mid: *t.baseline_mid(),
        delta_lambda: 5.0,
        inv_yn: t.inv_yn(),
        white_xyz: *t.white_xyz(),
        reference_white_xyz: *t.white_xyz(),
        has_baseline: u32::from(t.has_baseline()),
        illuminant_hash: t.illuminant_hash(),
        tables_hash: t.hash(),
    }
}
fn white_failure(e: &WhiteError) -> FjSpectralWhiteFailure {
    let (reason, luminance_sum) = match *e {
        WhiteError::NonfiniteSample => (1, 0.0),
        WhiteError::InvalidLuminance(sum) => (2, sum),
        WhiteError::InvalidSum => (3, 0.0),
        WhiteError::NonfiniteHashOperand => (4, 0.0),
        WhiteError::ZeroIdentity => (5, 0.0),
    };
    FjSpectralWhiteFailure {
        reason,
        luminance_sum,
    }
}

#[cfg(feature = "test-support")]
thread_local! {static SPECTRAL_FAULT:std::cell::Cell<Option<(u32,u32,u32)>>=const {std::cell::Cell::new(None)};}
#[cfg(feature = "test-support")]
pub(crate) fn arm_fault(operation: u32, index: u32, fault: u32) -> FjStatus {
    clear_fault();
    if !(1..=3).contains(&operation) || index == 0 || !(1..=4).contains(&fault) {
        return FjStatus {
            category: FJ_STATUS_UNSUPPORTED_INPUT,
            api: FJ_API_NONE,
            native_code: 0,
        };
    }
    SPECTRAL_FAULT.set(Some((operation, index, fault)));
    FjStatus {
        category: FJ_STATUS_SUCCESS,
        api: FJ_API_NONE,
        native_code: 0,
    }
}
#[cfg(feature = "test-support")]
pub(crate) fn clear_fault() {
    SPECTRAL_FAULT.set(None);
}
fn inject(operation: u32) -> Result<(), Failure> {
    #[cfg(feature = "test-support")]
    if let Some((selected, remaining, fault)) = SPECTRAL_FAULT.get()
        && selected == operation
    {
        if remaining > 1 {
            SPECTRAL_FAULT.set(Some((selected, remaining - 1, fault)));
            return Ok(());
        }
        SPECTRAL_FAULT.set(None);
        return match fault {
            1 => Err(Failure::Input("spectral injected unsupported input")),
            2 => panic!("spectral injected panic"),
            3 => Err(Failure::InjectedAllocation),
            4 => Err(Failure::Spectral(WhiteError::InvalidLuminance(0.0))),
            _ => unreachable!("closed spectral fault"),
        };
    }
    #[cfg(not(feature = "test-support"))]
    let _ = operation;
    Ok(())
}
/// # Safety
/// Initialized input spans, exclusive output/error follow the C header contract.
pub(crate) unsafe fn tables_call(
    input: *const FjSpectralInput,
    out: *mut FjSpectralTables,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Only authorized valid records clear; containment checks diagnostics.
    unsafe {
        let valid = clear(out);
        asset_illuminant::run(error, || {
            if !valid {
                return Err(Failure::Input("invalid spectral tables output"));
            }
            let input = table_input(input)?;
            fault()?;
            out.write(project_tables(&spectral::build_tables(input)));
            Ok(())
        })
    }
}
/// # Safety
/// Initialized inputs and both exclusive output/error records follow the header.
pub(crate) unsafe fn white_call(
    input: *const FjSpectralWhiteInput,
    out: *mut FjSpectralWhite,
    failure: *mut FjSpectralWhiteFailure,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Cleared outputs are exclusive; scoped checked borrows expire at return.
    unsafe {
        let valid_out = clear(out);
        let valid_failure = clear(failure);
        asset_illuminant::run(error, || {
            if !valid_out || !valid_failure || !aligned(input) {
                return Err(Failure::Input("invalid spectral white records"));
            }
            let input = &*input;
            let observer = observer(input.observer)?;
            let illuminant = fixed(input.illuminant)?;
            fault()?;
            match spectral::integrate_white(observer, illuminant) {
                Ok(w) => {
                    out.write(FjSpectralWhite {
                        normalization: w.normalization(),
                        white_xyz: *w.xyz(),
                        white_xy: *w.xy(),
                        hash: w.hash(),
                    });
                    Ok(())
                }
                Err(e) => {
                    failure.write(white_failure(&e));
                    Err(Failure::Spectral(e))
                }
            }
        })
    }
}
/// # Safety
/// Checked initialized weighted spans and exclusive output/error obey the header.
pub(crate) unsafe fn inverse_call(
    input: *const FjSpectralSInput,
    out: *mut FjSpectralInverse,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Only valid writable output clears; references borrow live caller arrays.
    unsafe {
        let valid = clear(out);
        asset_illuminant::run(error, || {
            if !valid || !aligned(input) {
                return Err(Failure::Input("invalid spectral inverse records"));
            }
            let input = &*input;
            let weights = [
                fixed(input.weighted_x)?,
                fixed(input.weighted_y)?,
                fixed(input.weighted_z)?,
            ];
            fault()?;
            out.write(FjSpectralInverse {
                matrix: spectral::s_inverse(weights),
            });
            Ok(())
        })
    }
}
// FJ_TEMP_BRIDGE: spectral value preparation; remove S4.E.
/// # Safety
/// Required storage obeys the legacy header's synchronous disjoint-storage contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_spectral_tables(
    input: *const FjSpectralInput,
    out: *mut FjSpectralTables,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Caller supplies the complete initialized input and exclusive outputs.
    unsafe { tables_call(input, out, error, || inject(1)) }
}
/// # Safety
/// Required inputs/outputs/error obey the synchronous C spectral contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_spectral_white(
    input: *const FjSpectralWhiteInput,
    out: *mut FjSpectralWhite,
    failure: *mut FjSpectralWhiteFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Caller retains all live source arrays and exclusive output storage.
    unsafe { white_call(input, out, failure, error, || inject(2)) }
}
/// # Safety
/// Weighted arrays and output/error obey the synchronous C spectral contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_spectral_s_inverse(
    input: *const FjSpectralSInput,
    out: *mut FjSpectralInverse,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Caller retains immutable weights and exclusive output/diagnostic storage.
    unsafe { inverse_call(input, out, error, || inject(3)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layouts() {
        assert_eq!(
            (
                size_of::<FjSpectralObserver>(),
                align_of::<FjSpectralObserver>()
            ),
            (48, 8)
        );
        assert_eq!(
            (size_of::<FjSpectralInput>(), align_of::<FjSpectralInput>()),
            (120, 8)
        );
        assert_eq!(
            (
                size_of::<FjSpectralTables>(),
                align_of::<FjSpectralTables>()
            ),
            (4264, 8)
        );
        assert_eq!(
            (
                size_of::<FjSpectralWhiteInput>(),
                size_of::<FjSpectralSInput>()
            ),
            (64, 48)
        );
        assert_eq!(
            (
                size_of::<FjSpectralWhite>(),
                size_of::<FjSpectralWhiteFailure>()
            ),
            (32, 16)
        );
        assert_eq!(
            (
                size_of::<FjSpectralInverse>(),
                align_of::<FjSpectralInverse>()
            ),
            (36, 4)
        );
        assert_eq!(align_of::<FjSpectralWhiteInput>(), 8);
        assert_eq!(align_of::<FjSpectralSInput>(), 8);
        assert_eq!(align_of::<FjSpectralWhite>(), 8);
        assert_eq!(align_of::<FjSpectralWhiteFailure>(), 8);
        assert_eq!(std::mem::offset_of!(FjSpectralObserver, x), 0);
        assert_eq!(std::mem::offset_of!(FjSpectralObserver, y), 16);
        assert_eq!(std::mem::offset_of!(FjSpectralObserver, z), 32);
        assert_eq!(std::mem::offset_of!(FjSpectralInput, dyes_cmy), 0);
        assert_eq!(std::mem::offset_of!(FjSpectralInput, observer), 16);
        assert_eq!(std::mem::offset_of!(FjSpectralInput, illuminant), 64);
        assert_eq!(std::mem::offset_of!(FjSpectralInput, baseline_min), 80);
        assert_eq!(std::mem::offset_of!(FjSpectralInput, baseline_mid), 96);
        assert_eq!(std::mem::offset_of!(FjSpectralInput, illuminant_hash), 112);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, lambda_nm), 0);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, illuminant), 324);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, observer_xyz), 648);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, weighted_xyz), 1620);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, dyes_cmy), 2592);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, baseline_min), 3564);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, baseline_mid), 3888);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, delta_lambda), 4212);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, inv_yn), 4216);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, white_xyz), 4220);
        assert_eq!(
            std::mem::offset_of!(FjSpectralTables, reference_white_xyz),
            4232
        );
        assert_eq!(std::mem::offset_of!(FjSpectralTables, has_baseline), 4244);
        assert_eq!(
            std::mem::offset_of!(FjSpectralTables, illuminant_hash),
            4248
        );
        assert_eq!(std::mem::offset_of!(FjSpectralWhiteInput, observer), 0);
        assert_eq!(std::mem::offset_of!(FjSpectralWhiteInput, illuminant), 48);
        assert_eq!(std::mem::offset_of!(FjSpectralWhite, normalization), 0);
        assert_eq!(std::mem::offset_of!(FjSpectralWhite, white_xyz), 4);
        assert_eq!(std::mem::offset_of!(FjSpectralWhite, white_xy), 16);
        assert_eq!(std::mem::offset_of!(FjSpectralWhiteFailure, reason), 0);
        assert_eq!(
            std::mem::offset_of!(FjSpectralWhiteFailure, luminance_sum),
            8
        );
        assert_eq!(std::mem::offset_of!(FjSpectralSInput, weighted_x), 0);
        assert_eq!(std::mem::offset_of!(FjSpectralSInput, weighted_y), 16);
        assert_eq!(std::mem::offset_of!(FjSpectralSInput, weighted_z), 32);
        assert_eq!(std::mem::offset_of!(FjSpectralInverse, matrix), 0);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, weighted_xyz), 1620);
        assert_eq!(std::mem::offset_of!(FjSpectralTables, tables_hash), 4256);
        assert_eq!(std::mem::offset_of!(FjSpectralWhite, hash), 24);
    }
}
