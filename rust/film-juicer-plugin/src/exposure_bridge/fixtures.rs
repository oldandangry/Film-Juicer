//! Feature-only fixed spectrum projections for retained numerical fixtures.
use super::{aligned, clear, fixed, rgb_tensor};
use crate::asset_bridge::Failure;
use crate::asset_illuminant;
use crate::cuda::sys::{
    FjErrorBuffer, FjFloatSpan, FjStatus, FJ_API_NONE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT,
};
use film_juicer_core::exposure::fixtures;

#[repr(C)]
pub(crate) struct FjHanatosSpectrumInput {
    pub rgb_dwg: [f32; 3],
    pub reference_white: [f32; 3],
    pub spectra: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjTablesSpectrumInput {
    pub rgb_dwg: [f32; 3],
    pub reference_white: [f32; 3],
    pub s_inverse: [f32; 9],
    pub ax: FjFloatSpan,
    pub ay: FjFloatSpan,
    pub az: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjMallettRawInput {
    pub linear_srgb: [f32; 3],
    pub basis_rgb: FjFloatSpan,
    pub illuminant: FjFloatSpan,
    pub sensitivity_rgb: FjFloatSpan,
}
#[repr(C)]
pub(crate) struct FjSpectrumFixture {
    pub pre_xyz: [f32; 3],
    pub consumer_white: [f32; 3],
    pub post_adapt_xyz: [f32; 3],
    pub spectrum: [f32; 81],
}
impl Default for FjSpectrumFixture {
    fn default() -> Self {
        Self {
            pre_xyz: [0.0; 3],
            consumer_white: [0.0; 3],
            post_adapt_xyz: [0.0; 3],
            spectrum: [0.0; 81],
        }
    }
}
impl From<fixtures::Spectrum> for FjSpectrumFixture {
    fn from(value: fixtures::Spectrum) -> Self {
        Self {
            pre_xyz: *value.pre_xyz(),
            consumer_white: *value.consumer_white(),
            post_adapt_xyz: *value.post_adapt_xyz(),
            spectrum: *value.samples(),
        }
    }
}
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<FjHanatosSpectrumInput>() == 40 && align_of::<FjHanatosSpectrumInput>() == 8);
    assert!(
        offset_of!(FjHanatosSpectrumInput, rgb_dwg) == 0
            && offset_of!(FjHanatosSpectrumInput, reference_white) == 12
            && offset_of!(FjHanatosSpectrumInput, spectra) == 24
    );
    assert!(size_of::<FjTablesSpectrumInput>() == 112 && align_of::<FjTablesSpectrumInput>() == 8);
    assert!(
        offset_of!(FjTablesSpectrumInput, rgb_dwg) == 0
            && offset_of!(FjTablesSpectrumInput, reference_white) == 12
            && offset_of!(FjTablesSpectrumInput, s_inverse) == 24
    );
    assert!(
        offset_of!(FjTablesSpectrumInput, ax) == 64
            && offset_of!(FjTablesSpectrumInput, ay) == 80
            && offset_of!(FjTablesSpectrumInput, az) == 96
    );
    assert!(size_of::<FjMallettRawInput>() == 64 && align_of::<FjMallettRawInput>() == 8);
    assert!(
        offset_of!(FjMallettRawInput, linear_srgb) == 0
            && offset_of!(FjMallettRawInput, basis_rgb) == 16
            && offset_of!(FjMallettRawInput, illuminant) == 32
            && offset_of!(FjMallettRawInput, sensitivity_rgb) == 48
    );
    assert!(size_of::<FjSpectrumFixture>() == 360 && align_of::<FjSpectrumFixture>() == 4);
    assert!(
        offset_of!(FjSpectrumFixture, pre_xyz) == 0
            && offset_of!(FjSpectrumFixture, consumer_white) == 12
            && offset_of!(FjSpectrumFixture, post_adapt_xyz) == 24
            && offset_of!(FjSpectrumFixture, spectrum) == 36
    );
};
thread_local! {
    static FAULT:std::cell::Cell<Option<(u32,u32,u32)>>=const {std::cell::Cell::new(None)};
    static CONSUMED:std::cell::Cell<Option<(u32,u32)>>=const {std::cell::Cell::new(None)};
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_exposure_fixture_arm_fault(
    operation: u32,
    index: u32,
    fault: u32,
) -> FjStatus {
    FAULT.set(None);
    CONSUMED.set(None);
    let valid = (1..=3).contains(&operation) && index != 0 && (1..=2).contains(&fault);
    if valid {
        FAULT.set(Some((operation, index, fault)));
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
extern "C" fn fj_test_exposure_fixture_clear_fault() {
    FAULT.set(None);
    CONSUMED.set(None);
}
#[unsafe(no_mangle)]
extern "C" fn fj_test_exposure_fixture_fault_consumed(operation: u32, fault: u32) -> u32 {
    u32::from(CONSUMED.take() == Some((operation, fault)))
}
fn inject(operation: u32) -> Result<(), Failure> {
    if let Some((selected, index, fault)) = FAULT.get() {
        if selected != operation {
            return Ok(());
        }
        if index > 1 {
            FAULT.set(Some((selected, index - 1, fault)));
            return Ok(());
        }
        FAULT.set(None);
        CONSUMED.set(Some((operation, fault)));
        if fault == 1 {
            return Err(Failure::Input("fixed spectrum fixture fault"));
        }
        panic!("fixed spectrum fixture fault");
    }
    Ok(())
}
/// # Safety
/// Input/tensor are initialized and immutable until return. Exclusive aligned
/// output/error storage is disjoint from inputs and each other. No pointer escapes.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_test_exposure_hanatos_spectrum(
    input: *const FjHanatosSpectrumInput,
    out: *mut FjSpectrumFixture,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Checked output clears first; fixed source borrow is synchronous.
    unsafe {
        let valid = clear(out);
        asset_illuminant::run(error, || {
            if !valid || !aligned(input) {
                return Err(Failure::Input("invalid Hanatos spectrum fixture records"));
            }
            let input = &*input;
            let spectra = fixed(input.spectra)?;
            inject(1)?;
            out.write(
                fixtures::hanatos_spectrum(input.rgb_dwg, spectra, input.reference_white).into(),
            );
            Ok(())
        })
    }
}
/// # Safety
/// Complete immutable records/spans and exclusive disjoint output/error storage
/// are initialized and live for this synchronous call. No pointer escapes.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_test_exposure_tables_spectrum(
    input: *const FjTablesSpectrumInput,
    out: *mut FjSpectrumFixture,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Checked fixed borrows are local; only complete fixed results publish.
    unsafe {
        let valid = clear(out);
        asset_illuminant::run(error, || {
            if !valid || !aligned(input) {
                return Err(Failure::Input("invalid tables spectrum fixture records"));
            }
            let input = &*input;
            let request = fixtures::TablesInput {
                rgb_dwg: input.rgb_dwg,
                reference_white: input.reference_white,
                s_inverse: input.s_inverse,
                ax: fixed(input.ax)?,
                ay: fixed(input.ay)?,
                az: fixed(input.az)?,
            };
            inject(2)?;
            out.write(fixtures::tables_spectrum(request).into());
            Ok(())
        })
    }
}
/// # Safety
/// The initialized immutable fixed inputs and exclusive disjoint output/error
/// storage are aligned and live until return. The output is three BGR floats.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_test_exposure_mallett_raw(
    input: *const FjMallettRawInput,
    out: *mut f32,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Layout-checked borrows expire on return, and output clears first.
    unsafe {
        let target = out.cast::<[f32; 3]>();
        let valid = clear(target);
        asset_illuminant::run(error, || {
            if !valid || !aligned(input) {
                return Err(Failure::Input("invalid Mallett leaf fixture records"));
            }
            let input = &*input;
            let basis = rgb_tensor(input.basis_rgb)?;
            let illum = fixed(input.illuminant)?;
            let sensitivity = rgb_tensor(input.sensitivity_rgb)?;
            inject(3)?;
            target.write(fixtures::mallett_raw(
                input.linear_srgb,
                basis,
                illum,
                sensitivity,
            ));
            Ok(())
        })
    }
}
