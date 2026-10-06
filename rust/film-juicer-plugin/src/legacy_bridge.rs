//! Temporary CPU boundaries for native route and color preparation.

use film_juicer_core::color::{self, Whites};
use crate::cuda::sys::{
    FjStatus, FJ_API_NONE, FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT,
    FJ_STATUS_INTERNAL_FAILURE,
};

use film_juicer_core::route::{self, CapturePolarity, RouteSelection, ScanRoute};

const SUCCESS: u32 = 0;
const INVALID_POLARITY: u32 = 1;
const INVALID_SELECTION: u32 = 2;
const NULL_OUTPUT: u32 = 3;
const INTERNAL_PANIC: u32 = 4;

fn resolve_foreign(capture_polarity: u8, selected_route: u8) -> Result<u8, u32> {
    let polarity = match capture_polarity {
        0 | 2 => CapturePolarity::Negative,
        1 => CapturePolarity::Positive,
        _ => return Err(INVALID_POLARITY),
    };
    let selection = match selected_route {
        0 | 2 => RouteSelection::Direct,
        1 | 3 => RouteSelection::Print,
        _ => return Err(INVALID_SELECTION),
    };
    Ok(match route::resolve(polarity, selection) {
        ScanRoute::NegativeDirect => 0,
        ScanRoute::NegativePrint => 1,
        ScanRoute::PositiveDirect => 2,
        ScanRoute::PositivePrint => 3,
    })
}

fn contain_panic(
    resolve: impl FnOnce() -> Result<u8, u32> + std::panic::UnwindSafe,
) -> Result<u8, u32> {
    std::panic::catch_unwind(resolve).unwrap_or(Err(INTERNAL_PANIC))
}

// FJ_TEMP_BRIDGE: route resolution; remove S6.D.
/// Resolve a route for the C++ callers.
///
/// # Safety
/// A nonnull `out_route` must point to one exclusively writable byte until return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_legacy_resolve_route(
    capture_polarity: u8,
    selected_route: u8,
    out_route: *mut u8,
) -> u32 {
    if out_route.is_null() {
        return NULL_OUTPUT;
    }
    match contain_panic(|| resolve_foreign(capture_polarity, selected_route)) {
        Ok(route) => {
            // SAFETY: The foreign caller guarantees one writable byte for this call.
            unsafe { out_route.write(route) };
            SUCCESS
        }
        Err(status) => status,
    }
}

const _: () = assert!(
    std::mem::size_of::<f32>() == 4
        && std::mem::size_of::<[f32; 3]>() == 12
        && std::mem::size_of::<[f32; 9]>() == 36
);

fn color_status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}

#[cfg(feature = "test-support")]
#[derive(Clone, Copy)]
struct ColorFault {
    operation: u32,
    remaining: u32,
    fault: u32,
}

#[cfg(feature = "test-support")]
thread_local! {
    static COLOR_FAULT: std::cell::Cell<Option<ColorFault>> = const { std::cell::Cell::new(None) };
}

#[cfg(feature = "test-support")]
pub(crate) fn arm_color_fault(operation: u32, call_index: u32, fault: u32) -> FjStatus {
    clear_color_fault();
    if !(1..=10).contains(&operation) || call_index == 0 || !(1..=2).contains(&fault) {
        return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    COLOR_FAULT.set(Some(ColorFault {
        operation,
        remaining: call_index,
        fault,
    }));
    color_status(FJ_STATUS_SUCCESS)
}

#[cfg(feature = "test-support")]
pub(crate) fn clear_color_fault() -> FjStatus {
    COLOR_FAULT.set(None);
    color_status(FJ_STATUS_SUCCESS)
}

#[cfg(feature = "test-support")]
fn color_fault(operation: u32) -> Result<(), u32> {
    let Some(mut armed) = COLOR_FAULT.get() else {
        return Ok(());
    };
    if armed.operation != operation {
        return Ok(());
    }
    armed.remaining -= 1;
    if armed.remaining != 0 {
        COLOR_FAULT.set(Some(armed));
        return Ok(());
    }
    COLOR_FAULT.set(None);
    if armed.fault == 1 {
        return Err(FJ_STATUS_UNSUPPORTED_INPUT);
    }
    panic!("color production boundary fault");
}

// FJ_TEMP_BRIDGE: CAT16 host preparation; remove S4.E.
/// Prepare a row-major CAT16 matrix by sampling the three basis vectors.
///
/// # Safety
/// Nonnull inputs authorize three initialized aligned floats each until return;
/// they may alias each other. Nonnull output authorizes nine exclusive aligned
/// floats, disjoint from every input. No storage is retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_legacy_cat16_matrix(
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_row_major: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_row_major.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller provides the exclusive nine-float output extent.
        unsafe { out_row_major.write_bytes(0, 9) };
        if source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(1) {
            return color_status(category);
        }
        // SAFETY: Both nonnull read-only inputs authorize exactly three floats.
        let whites = unsafe {
            Whites {
                source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
            }
        };
        let matrix = color::cat16_matrix(whites);
        // SAFETY: The complete local result is disjoint from the authorized output.
        unsafe { out_row_major.copy_from_nonoverlapping(matrix.as_ptr(), 9) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: CAT16 host preparation; remove S4.E.
/// Adapt an unsanitized XYZ value between the supplied whites.
///
/// # Safety
/// Nonnull inputs authorize three initialized aligned floats each until return;
/// they may alias each other. Nonnull output authorizes three exclusive aligned
/// floats disjoint from every input. No storage is retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_legacy_adapt_cat16(
    xyz: *const f32,
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller provides the exclusive three-float output extent.
        unsafe { out_xyz.write_bytes(0, 3) };
        if xyz.is_null() || source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(2) {
            return color_status(category);
        }
        // SAFETY: Each nonnull read-only input authorizes exactly three floats.
        let (value, whites) = unsafe {
            (
                xyz.cast::<[f32; 3]>().read(),
                Whites {
                    source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                    destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
                },
            )
        };
        let adapted = color::adapt_cat16(value, whites);
        // SAFETY: The complete local result is disjoint from the authorized output.
        unsafe { out_xyz.copy_from_nonoverlapping(adapted.as_ptr(), 3) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: CAT02 host preparation; remove S4.E.
/// Prepare a row-major CAT02 matrix by sampling the three basis vectors.
///
/// # Safety
/// Nonnull inputs authorize three initialized aligned floats each until return;
/// they may alias each other. Nonnull output authorizes nine exclusive aligned
/// floats, disjoint from every input. No storage is retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_legacy_cat02_matrix(
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_row_major: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_row_major.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller provides the exclusive nine-float output extent.
        unsafe { out_row_major.write_bytes(0, 9) };
        if source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(3) {
            return color_status(category);
        }
        // SAFETY: Both nonnull read-only inputs authorize exactly three floats.
        let whites = unsafe {
            Whites {
                source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
            }
        };
        let matrix = color::cat02_matrix(whites);
        // SAFETY: The complete local result is disjoint from the authorized output.
        unsafe { out_row_major.copy_from_nonoverlapping(matrix.as_ptr(), 9) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: CAT02 host preparation; remove S4.E.
/// Adapt an unsanitized XYZ value between the supplied whites.
///
/// # Safety
/// Nonnull inputs authorize three initialized aligned floats each until return;
/// they may alias each other. Nonnull output authorizes three exclusive aligned
/// floats disjoint from every input. No storage is retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_legacy_adapt_cat02(
    xyz: *const f32,
    source_white_xyz: *const f32,
    destination_white_xyz: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller provides the exclusive three-float output extent.
        unsafe { out_xyz.write_bytes(0, 3) };
        if xyz.is_null() || source_white_xyz.is_null() || destination_white_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(4) {
            return color_status(category);
        }
        // SAFETY: Each nonnull read-only input authorizes exactly three floats.
        let (value, whites) = unsafe {
            (
                xyz.cast::<[f32; 3]>().read(),
                Whites {
                    source_xyz: source_white_xyz.cast::<[f32; 3]>().read(),
                    destination_xyz: destination_white_xyz.cast::<[f32; 3]>().read(),
                },
            )
        };
        let adapted = color::adapt_cat02(value, whites);
        // SAFETY: The complete local result is disjoint from the authorized output.
        unsafe { out_xyz.copy_from_nonoverlapping(adapted.as_ptr(), 3) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

#[repr(C)]
pub(crate) struct FjInputColorMatrices {
    pub(crate) rgb_to_xyz: [f32; 9],
    pub(crate) nominal_white_xyz: [f32; 3],
    pub(crate) xyz_to_linear_srgb: [f32; 9],
    pub(crate) d65_white_xyz: [f32; 3],
}

impl From<color::InputMatrices> for FjInputColorMatrices {
    fn from(input: color::InputMatrices) -> Self {
        Self {
            rgb_to_xyz: input.rgb_to_xyz,
            nominal_white_xyz: input.nominal_white_xyz,
            xyz_to_linear_srgb: input.xyz_to_linear_srgb,
            d65_white_xyz: input.d65_white_xyz,
        }
    }
}

#[repr(C)]
pub(crate) struct FjInputColorConversion {
    pub(crate) input_space: u32,
    pub(crate) decode_cctf: u32,
    pub(crate) adapt_xyz: u32,
    pub(crate) rgb_to_xyz: [f32; 9],
    pub(crate) xyz_adapt: [f32; 9],
}

pub(crate) fn input_space(tag: u32) -> Option<color::InputSpace> {
    match tag {
        0 => Some(color::InputSpace::DavinciWideGamut),
        1 => Some(color::InputSpace::Bt2020),
        2 => Some(color::InputSpace::Aces2065_1),
        3 => Some(color::InputSpace::SrgbRec709),
        _ => None,
    }
}

impl FjInputColorConversion {
    pub(crate) fn into_core(self) -> Option<color::InputConversion> {
        let space = input_space(self.input_space)?;
        if self.decode_cctf > 1 || self.adapt_xyz > 1 {
            return None;
        }
        Some(color::InputConversion {
            space,
            decode_cctf: self.decode_cctf != 0,
            rgb_to_xyz: self.rgb_to_xyz,
            xyz_adaptation: (self.adapt_xyz != 0).then_some(self.xyz_adapt),
        })
    }
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<FjInputColorMatrices>() == 96 && align_of::<FjInputColorMatrices>() == 4);
    assert!(offset_of!(FjInputColorMatrices, rgb_to_xyz) == 0);
    assert!(offset_of!(FjInputColorMatrices, nominal_white_xyz) == 36);
    assert!(offset_of!(FjInputColorMatrices, xyz_to_linear_srgb) == 48);
    assert!(offset_of!(FjInputColorMatrices, d65_white_xyz) == 84);
    assert!(size_of::<FjInputColorConversion>() == 84 && align_of::<FjInputColorConversion>() == 4);
    assert!(offset_of!(FjInputColorConversion, input_space) == 0);
    assert!(offset_of!(FjInputColorConversion, decode_cctf) == 4);
    assert!(offset_of!(FjInputColorConversion, adapt_xyz) == 8);
    assert!(offset_of!(FjInputColorConversion, rgb_to_xyz) == 12);
    assert!(offset_of!(FjInputColorConversion, xyz_adapt) == 48);
};

// FJ_TEMP_BRIDGE: input-color host preparation; remove S4.E.
/// Prepare input matrices as caller-owned values.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_input_matrices(
    input_space_tag: u32,
    out: *mut FjInputColorMatrices,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The caller authorizes the complete exclusive output record.
        unsafe { out.write_bytes(0, 1) };
        let Some(space) = input_space(input_space_tag) else {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        };
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(5) {
            return color_status(category);
        }
        let output = FjInputColorMatrices::from(color::input_matrices(space));
        // SAFETY: The initialized local record is committed to the exclusive output.
        unsafe { out.write(output) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: input-color host preparation; remove S4.E.
/// Prepare input to dwg as caller-owned values.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_input_to_dwg(
    input: *const FjInputColorConversion,
    rgb: *const f32,
    clamp_nonnegative: u32,
    out_rgb: *mut f32,
    out_xyz: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        // SAFETY: Each nonnull output authorizes three exclusive aligned floats.
        unsafe {
            if !out_rgb.is_null() {
                out_rgb.write_bytes(0, 3);
            }
            if !out_xyz.is_null() {
                out_xyz.write_bytes(0, 3);
            }
        }
        if input.is_null()
            || rgb.is_null()
            || out_rgb.is_null()
            || out_xyz.is_null()
            || clamp_nonnegative > 1
        {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The input record is fully initialized, including disabled adaptation.
        let Some(input) = (unsafe { input.read() }).into_core() else {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        };
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(6) {
            return color_status(category);
        }
        // SAFETY: The read-only RGB extent authorizes three initialized aligned floats.
        let rgb = unsafe { rgb.cast::<[f32; 3]>().read() };
        let output = color::input_to_dwg(input, rgb, clamp_nonnegative != 0);
        // SAFETY: Outputs are mutually disjoint and disjoint from inputs/local arrays.
        unsafe {
            out_rgb.copy_from_nonoverlapping(output.rgb.as_ptr(), 3);
            out_xyz.copy_from_nonoverlapping(output.xyz.as_ptr(), 3);
        }
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: input-color host preparation; remove S4.E.
/// Prepare input to linear srgb as caller-owned values.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_input_to_linear_srgb(
    input: *const FjInputColorConversion,
    rgb: *const f32,
    xyz_to_linear_srgb: *const f32,
    out_rgb: *mut f32,
    out_xyz: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        // SAFETY: Each nonnull output authorizes three exclusive aligned floats.
        unsafe {
            if !out_rgb.is_null() {
                out_rgb.write_bytes(0, 3);
            }
            if !out_xyz.is_null() {
                out_xyz.write_bytes(0, 3);
            }
        }
        if input.is_null()
            || rgb.is_null()
            || xyz_to_linear_srgb.is_null()
            || out_rgb.is_null()
            || out_xyz.is_null()
        {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The input record is fully initialized, including disabled adaptation.
        let Some(input) = (unsafe { input.read() }).into_core() else {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        };
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(7) {
            return color_status(category);
        }
        // SAFETY: Read-only inputs authorize three RGB and nine matrix floats respectively.
        let (rgb, inverse) = unsafe {
            (
                rgb.cast::<[f32; 3]>().read(),
                xyz_to_linear_srgb.cast::<[f32; 9]>().read(),
            )
        };
        let output = color::input_to_linear_srgb(input, rgb, inverse);
        // SAFETY: Outputs are mutually disjoint and disjoint from inputs/local arrays.
        unsafe {
            out_rgb.copy_from_nonoverlapping(output.rgb.as_ptr(), 3);
            out_xyz.copy_from_nonoverlapping(output.xyz.as_ptr(), 3);
        }
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: input-color host preparation; remove S4.E.
/// Prepare linear srgb to xyz as caller-owned values.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_linear_srgb_to_xyz(
    rgb: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The output authorizes three exclusive aligned floats.
        unsafe { out_xyz.write_bytes(0, 3) };
        if rgb.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(8) {
            return color_status(category);
        }
        // SAFETY: The read-only RGB extent authorizes three initialized aligned floats.
        let rgb = unsafe { rgb.cast::<[f32; 3]>().read() };
        let output = color::linear_srgb_to_xyz(rgb);
        // SAFETY: The initialized local triplet is disjoint from the output extent.
        unsafe { out_xyz.copy_from_nonoverlapping(output.as_ptr(), 3) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: input-color host preparation; remove S4.E.
/// Prepare dwg to xyz as caller-owned values.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_dwg_to_xyz(
    rgb: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The output authorizes three exclusive aligned floats.
        unsafe { out_xyz.write_bytes(0, 3) };
        if rgb.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(9) {
            return color_status(category);
        }
        // SAFETY: The read-only RGB extent authorizes three initialized aligned floats.
        let rgb = unsafe { rgb.cast::<[f32; 3]>().read() };
        let output = color::dwg_to_xyz(rgb);
        // SAFETY: The initialized local triplet is disjoint from the output extent.
        unsafe { out_xyz.copy_from_nonoverlapping(output.as_ptr(), 3) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

// FJ_TEMP_BRIDGE: input-color host preparation; remove S4.E.
/// Prepare project linear rgb to xyz as caller-owned values.
/// # Safety
/// Nonnull read-only inputs authorize complete initialized aligned records/arrays
/// until return and may share storage. RGB triplets have three floats, matrices
/// nine. Even disabled adaptation is initialized. Nonnull outputs authorize their
/// complete aligned extents, are exclusive, mutually disjoint and disjoint from
/// every input. No input mutation/release during the call; no pointer is retained.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn fj_legacy_project_linear_rgb_to_xyz(
    rgb: *const f32,
    rgb_to_xyz: *const f32,
    xyz_adapt: *const f32,
    out_xyz: *mut f32,
) -> FjStatus {
    std::panic::catch_unwind(|| {
        if out_xyz.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        // SAFETY: The output authorizes three exclusive aligned floats.
        unsafe { out_xyz.write_bytes(0, 3) };
        if rgb.is_null() || rgb_to_xyz.is_null() || xyz_adapt.is_null() {
            return color_status(FJ_STATUS_UNSUPPORTED_INPUT);
        }
        #[cfg(feature = "test-support")]
        if let Err(category) = color_fault(10) {
            return color_status(category);
        }
        // SAFETY: Inputs authorize an initialized RGB triplet and two nine-float matrices.
        let (rgb, matrix, adaptation) = unsafe {
            (
                rgb.cast::<[f32; 3]>().read(),
                rgb_to_xyz.cast::<[f32; 9]>().read(),
                xyz_adapt.cast::<[f32; 9]>().read(),
            )
        };
        let output = color::project_linear_rgb_to_xyz(rgb, matrix, adaptation);
        // SAFETY: The initialized local triplet is disjoint from the output extent.
        unsafe { out_xyz.copy_from_nonoverlapping(output.as_ptr(), 3) };
        color_status(FJ_STATUS_SUCCESS)
    })
    .unwrap_or(color_status(FJ_STATUS_INTERNAL_FAILURE))
}

const _: unsafe extern "C" fn(u32, *mut FjInputColorMatrices) -> FjStatus =
    fj_legacy_input_matrices;
const _: unsafe extern "C" fn(
    *const FjInputColorConversion,
    *const f32,
    u32,
    *mut f32,
    *mut f32,
) -> FjStatus = fj_legacy_input_to_dwg;
const _: unsafe extern "C" fn(
    *const FjInputColorConversion,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
) -> FjStatus = fj_legacy_input_to_linear_srgb;
const _: unsafe extern "C" fn(*const f32, *mut f32) -> FjStatus = fj_legacy_linear_srgb_to_xyz;
const _: unsafe extern "C" fn(*const f32, *mut f32) -> FjStatus = fj_legacy_dwg_to_xyz;
const _: unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32) -> FjStatus =
    fj_legacy_project_linear_rgb_to_xyz;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_tags_preserve_the_reference_table() {
        let expected = [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1]];
        for (polarity, routes) in expected.into_iter().enumerate() {
            for (selection, route) in routes.into_iter().enumerate() {
                assert_eq!(resolve_foreign(polarity as u8, selection as u8), Ok(route));
            }
        }
    }

    #[test]
    fn foreign_failures_leave_output_unchanged() {
        let mut route = 99;
        // SAFETY: `route` is a live, exclusively borrowed output byte.
        unsafe {
            assert_eq!(
                fj_legacy_resolve_route(3, 0, &raw mut route),
                INVALID_POLARITY
            );
            assert_eq!(
                fj_legacy_resolve_route(0, 4, &raw mut route),
                INVALID_SELECTION
            );
            assert_eq!(
                fj_legacy_resolve_route(0, 0, std::ptr::null_mut()),
                NULL_OUTPUT
            );
            assert_eq!(route, 99);
            assert_eq!(fj_legacy_resolve_route(1, 0, &raw mut route), SUCCESS);
            assert_eq!(route, 2);
        }
    }

    #[test]
    fn panic_is_contained_as_status() {
        let result = contain_panic(|| panic!("test route panic"));
        assert_eq!(result, Err(INTERNAL_PANIC));
    }
}
