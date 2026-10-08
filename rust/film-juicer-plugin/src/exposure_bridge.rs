//! Synchronous borrowed reconstruction and complete sensitivity value edge.

use film_juicer_core::{exposure, reconstruction};

use crate::asset_bridge::Failure;
use crate::asset_illuminant;
use crate::cuda::sys::{FjErrorBuffer, FjFloatSpan, FjStatus};
#[cfg(feature = "test-support")]
use crate::cuda::sys::{FJ_API_NONE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT};

#[repr(C)]
pub(crate) struct FjReferenceWhiteInput {
    pub spectra: FjFloatSpan,
    pub white_xyz: [f32; 3],
    pub spectral_blur: f32,
}
#[repr(C)]
pub(crate) struct FjReferenceWhite {
    pub samples: [f32; 81],
}
impl Default for FjReferenceWhite {
    fn default() -> Self {
        Self { samples: [0.0; 81] }
    }
}
#[repr(C)]
pub(crate) struct FjSensitivityInput {
    pub linear_sensitivity_rgb: FjFloatSpan,
    pub reference_illuminant: FjFloatSpan,
    pub method: u32,
    pub band_pass_active: u32,
    pub apply_window: u32,
    pub uv: [f32; 3],
    pub ir: [f32; 3],
    pub window_params: FjFloatSpan,
    pub reconstructed_reference_white: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjSensitivity {
    pub values_rgb: [[f32; 3]; 81],
    pub mallett_green_scale: f32,
    pub hash: u64,
}
impl Default for FjSensitivity {
    fn default() -> Self {
        Self {
            values_rgb: [[0.0; 3]; 81],
            mallett_green_scale: 0.0,
            hash: 0,
        }
    }
}
#[repr(C)]
#[derive(Default)]
pub(crate) struct FjSensitivityFailure {
    pub hash_failure: u32,
    pub hash_sample_index: u32,
}

fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null()
        && (pointer as usize).is_multiple_of(align_of::<T>())
        && (pointer as usize).checked_add(size_of::<T>()).is_some()
}
/// # Safety
/// The caller supplies the complete writable, aligned, exclusive record.
unsafe fn clear<T: Default>(out: *mut T) -> bool {
    if !aligned(out) {
        return false;
    }
    // SAFETY: The complete output is caller-authorized and disjoint from inputs.
    unsafe { out.write(T::default()) };
    true
}
/// # Safety
/// The caller retains the initialized immutable span, disjoint from all outputs,
/// for the whole synchronous call. No dangling pointer can be validated here.
unsafe fn fixed<'a, const N: usize>(span: FjFloatSpan) -> Result<&'a [f32; N], Failure> {
    let bytes = span
        .count
        .checked_mul(size_of::<f32>())
        .ok_or(Failure::Input("exposure extent overflow"))?;
    if span.count != N
        || bytes > isize::MAX as usize
        || (span.data as usize).checked_add(bytes).is_none()
        || !aligned(span.data)
    {
        return Err(Failure::Input("invalid fixed exposure span"));
    }
    // SAFETY: Exact layout/addressable extent and caller-proven immutable lifetime.
    Ok(unsafe { &*span.data.cast::<[f32; N]>() })
}
/// # Safety
/// Input and consumed initialized spans remain immutable through this call.
unsafe fn sensitivity_input<'a>(
    input: *const FjSensitivityInput,
) -> Result<exposure::Input<'a>, Failure> {
    if !aligned(input) {
        return Err(Failure::Input("invalid sensitivity input"));
    }
    // SAFETY: Caller retains the aligned complete record and consumed spans;
    // each array borrow is formed only after its fixed extent/layout checks.
    unsafe {
        let input = &*input;
        if input.method > 2 || input.band_pass_active > 1 || input.apply_window > 1 {
            return Err(Failure::Input("invalid sensitivity method or flags"));
        }
        let linear = fixed::<243>(input.linear_sensitivity_rgb)?;
        // f32 arrays have no padding; 243 scalars and 81 RGB triplets are identical.
        let linear_sensitivity_rgb = &*linear.as_ptr().cast::<[[f32; 3]; 81]>();
        let reference_illuminant = fixed(input.reference_illuminant)?;
        let method = match input.method {
            0 => exposure::Method::Hanatos {
                window: if input.apply_window != 0 {
                    let params = fixed(input.window_params)?;
                    let white = input.reconstructed_reference_white;
                    Some(exposure::Window {
                        params,
                        reference_white: if white.count == 0 && white.data.is_null() {
                            None
                        } else {
                            Some(fixed(white)?)
                        },
                    })
                } else {
                    None
                },
            },
            1 => exposure::Method::Mallett,
            2 => exposure::Method::Arctic,
            _ => unreachable!("closed checked method"),
        };
        Ok(exposure::Input {
            linear_sensitivity_rgb,
            reference_illuminant,
            band_pass: (input.band_pass_active != 0).then_some(exposure::BandPass {
                uv: input.uv,
                ir: input.ir,
            }),
            method,
        })
    }
}

#[cfg(feature = "test-support")]
thread_local! {
    static EXPOSURE_FAULT: std::cell::Cell<Option<(u32,u32,u32)>> = const { std::cell::Cell::new(None) };
    static EXPOSURE_CONSUMED: std::cell::Cell<Option<(u32,u32)>> = const { std::cell::Cell::new(None) };
}
#[cfg(feature = "test-support")]
pub(crate) fn arm_fault(operation: u32, index: u32, fault: u32) -> FjStatus {
    clear_fault();
    let valid = (1..=2).contains(&operation) && index != 0 && (1..=4).contains(&fault);
    if valid {
        EXPOSURE_FAULT.set(Some((operation, index, fault)));
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
    EXPOSURE_FAULT.set(None);
    EXPOSURE_CONSUMED.set(None);
}
#[cfg(feature = "test-support")]
pub(crate) fn fault_consumed(operation: u32, fault: u32) -> bool {
    EXPOSURE_CONSUMED.take() == Some((operation, fault))
}
fn inject(operation: u32) -> Result<(), Failure> {
    #[cfg(feature = "test-support")]
    if let Some((selected, index, fault)) = EXPOSURE_FAULT.get() {
        if selected != operation {
            return Ok(());
        }
        if index > 1 {
            EXPOSURE_FAULT.set(Some((selected, index - 1, fault)));
            return Ok(());
        }
        EXPOSURE_FAULT.set(None);
        EXPOSURE_CONSUMED.set(Some((operation, fault)));
        return match fault {
            1 => Err(Failure::Input("exposure production boundary fault")),
            2 => panic!("exposure production boundary fault"),
            3 => Err(Failure::Reconstruction(
                reconstruction::Error::AllocationFailure,
            )),
            4 => Err(Failure::Reconstruction(
                reconstruction::Error::InvalidKernel,
            )),
            _ => unreachable!("closed exposure fault"),
        };
    }
    #[cfg(not(feature = "test-support"))]
    let _ = operation;
    Ok(())
}

/// # Safety
/// Complete initialized input/spans and exclusive disjoint output/error follow
/// the legacy header's call-local borrow contract.
pub(crate) unsafe fn reference_call(
    input: *const FjReferenceWhiteInput,
    out: *mut FjReferenceWhite,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Only valid writable records clear; the shared run contains panic
    // and checks the diagnostic before any scoped immutable input borrow is used.
    unsafe {
        let valid_out = clear(out);
        asset_illuminant::run(error, || {
            if !valid_out || !aligned(input) {
                return Err(Failure::Input("invalid reference white records"));
            }
            let input = &*input;
            let spectra = fixed(input.spectra)?;
            fault()?;
            let result =
                reconstruction::reference_white(spectra, input.spectral_blur, input.white_xyz)
                    .map_err(Failure::Reconstruction)?;
            out.write(FjReferenceWhite {
                samples: *result.samples(),
            });
            Ok(())
        })
    }
}
/// # Safety
/// Complete initialized inputs and exclusive disjoint result/failure/error
/// records obey the header. No input borrow outlives this synchronous call.
pub(crate) unsafe fn sensitivity_call(
    input: *const FjSensitivityInput,
    out: *mut FjSensitivity,
    failure: *mut FjSensitivityFailure,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: Caller-authorized outputs clear before fallible work; checked
    // borrows live through core execution and only complete results publish.
    unsafe {
        let valid_out = clear(out);
        let valid_failure = clear(failure);
        asset_illuminant::run(error, || {
            if !valid_out || !valid_failure {
                return Err(Failure::Input("invalid sensitivity output records"));
            }
            let input = sensitivity_input(input)?;
            fault()?;
            match exposure::prepare_sensitivity(input) {
                Ok(result) => {
                    out.write(FjSensitivity {
                        values_rgb: *result.values_rgb(),
                        mallett_green_scale: result.mallett_green_scale(),
                        hash: result.hash(),
                    });
                    Ok(())
                }
                Err(e) => {
                    let metadata = match e.hash_failure() {
                        Some(exposure::HashFailure::NonfiniteOperand(index)) => {
                            FjSensitivityFailure {
                                hash_failure: 1,
                                hash_sample_index: index as u32,
                            }
                        }
                        Some(exposure::HashFailure::ZeroIdentity) => FjSensitivityFailure {
                            hash_failure: 2,
                            hash_sample_index: 0,
                        },
                        None => FjSensitivityFailure::default(),
                    };
                    failure.write(metadata);
                    Err(Failure::Exposure(e))
                }
            }
        })
    }
}

// FJ_TEMP_BRIDGE: synchronous native reference preparation; remove S4.E.
/// # Safety
/// Required input/tensor/output/error storage obeys the legacy header contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_reconstruction_reference_white(
    input: *const FjReferenceWhiteInput,
    out: *mut FjReferenceWhite,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Caller retains immutable input and exclusive disjoint writable outputs.
    unsafe { reference_call(input, out, error, || inject(1)) }
}
// FJ_TEMP_BRIDGE: synchronous native sensitivity binding; remove S4.E.
/// # Safety
/// Required initialized spans/result/failure/error obey the synchronous contract.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_exposure_sensitivity(
    input: *const FjSensitivityInput,
    out: *mut FjSensitivity,
    failure: *mut FjSensitivityFailure,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Scoped checked projection borrows caller storage only until return.
    unsafe { sensitivity_call(input, out, failure, error, || inject(2)) }
}

#[cfg(feature = "test-support")]
/// # Safety
/// Initialized fixed params and exclusive scalar/error are caller-authorized.
pub(crate) unsafe fn window_call(
    wavelength: f32,
    params: FjFloatSpan,
    out: *mut f32,
    error: *mut FjErrorBuffer,
    fault: impl FnOnce() -> Result<(), Failure>,
) -> FjStatus {
    // SAFETY: The facade clears the exclusive scalar then borrows checked params.
    unsafe {
        let valid_out = clear(out);
        asset_illuminant::run(error, || {
            if !valid_out {
                return Err(Failure::Input("invalid window sample output"));
            }
            let params = fixed(params)?;
            fault()?;
            out.write(exposure::window_sample_for_test(wavelength, params));
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layouts_and_offsets() {
        assert_eq!(
            (
                size_of::<FjReferenceWhiteInput>(),
                align_of::<FjReferenceWhiteInput>()
            ),
            (32, 8)
        );
        assert_eq!(
            (
                size_of::<FjReferenceWhite>(),
                align_of::<FjReferenceWhite>()
            ),
            (324, 4)
        );
        assert_eq!(
            (
                size_of::<FjSensitivityInput>(),
                align_of::<FjSensitivityInput>()
            ),
            (104, 8)
        );
        assert_eq!(
            (size_of::<FjSensitivity>(), align_of::<FjSensitivity>()),
            (984, 8)
        );
        assert_eq!(
            (
                size_of::<FjSensitivityFailure>(),
                align_of::<FjSensitivityFailure>()
            ),
            (8, 4)
        );
        assert_eq!(std::mem::offset_of!(FjReferenceWhiteInput, spectra), 0);
        assert_eq!(std::mem::offset_of!(FjReferenceWhiteInput, white_xyz), 16);
        assert_eq!(
            std::mem::offset_of!(FjReferenceWhiteInput, spectral_blur),
            28
        );
        assert_eq!(std::mem::offset_of!(FjReferenceWhite, samples), 0);
        assert_eq!(
            std::mem::offset_of!(FjSensitivityInput, linear_sensitivity_rgb),
            0
        );
        assert_eq!(
            std::mem::offset_of!(FjSensitivityInput, reference_illuminant),
            16
        );
        assert_eq!(std::mem::offset_of!(FjSensitivityInput, method), 32);
        assert_eq!(
            std::mem::offset_of!(FjSensitivityInput, band_pass_active),
            36
        );
        assert_eq!(std::mem::offset_of!(FjSensitivityInput, apply_window), 40);
        assert_eq!(std::mem::offset_of!(FjSensitivityInput, uv), 44);
        assert_eq!(std::mem::offset_of!(FjSensitivityInput, ir), 56);
        assert_eq!(std::mem::offset_of!(FjSensitivityInput, window_params), 72);
        assert_eq!(
            std::mem::offset_of!(FjSensitivityInput, reconstructed_reference_white),
            88
        );
        assert_eq!(std::mem::offset_of!(FjSensitivity, values_rgb), 0);
        assert_eq!(
            std::mem::offset_of!(FjSensitivity, mallett_green_scale),
            972
        );
        assert_eq!(std::mem::offset_of!(FjSensitivity, hash), 976);
        assert_eq!(std::mem::offset_of!(FjSensitivityFailure, hash_failure), 0);
        assert_eq!(
            std::mem::offset_of!(FjSensitivityFailure, hash_sample_index),
            4
        );
    }
}
