//! Private raw edge for the native process asset/catalog conversion.

use std::collections::TryReserveError;
use std::ffi::c_void;
use std::fmt::{self, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};

use film_juicer_core::assets::{AssetError, Assets};
use film_juicer_core::profile::{Polarity, Role};

use crate::asset_catalog::{CatalogEntryView, CatalogOwner, PathError, PathView};
use crate::cuda::sys::{
    FJ_API_NONE, FJ_POLARITY_NEGATIVE, FJ_POLARITY_POSITIVE, FJ_STATUS_ALLOCATION_FAILURE,
    FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_SUCCESS,
    FJ_STATUS_UNSUPPORTED_INPUT, FjErrorBuffer, FjPathView, FjStatus, FjStringView,
};

#[cfg(target_os = "linux")]
use crate::cuda::sys::FJ_PATH_UNIX_BYTES;
#[cfg(target_os = "windows")]
use crate::cuda::sys::FJ_PATH_WINDOWS_WIDE;

const FJ_PROFILE_ROLE_FILM: u32 = 0;
const FJ_PROFILE_ROLE_PRINT: u32 = 1;

#[repr(C)]
struct FjAssets {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FjCatalog {
    _opaque: [u8; 0],
}
#[repr(C)]
#[derive(Default)]
struct FjCatalogCounts {
    film_count: usize,
    print_count: usize,
}
#[repr(C)]
struct FjCatalogEntryView {
    key: FjStringView,
    label: FjStringView,
    source_path: FjPathView,
    polarity: u32,
}

impl FjCatalogEntryView {
    fn empty() -> Self {
        Self {
            key: text(""),
            label: text(""),
            source_path: FjPathView {
                data: std::ptr::null(),
                count: 0,
                encoding: 0,
            },
            polarity: 0,
        }
    }

    fn from_view(view: CatalogEntryView<'_>) -> Result<Self, Failure> {
        let source_path = match view.source_path {
            #[cfg(target_os = "linux")]
            PathView::Bytes(units) => FjPathView {
                data: units.as_ptr().cast::<c_void>(),
                count: units.len(),
                encoding: FJ_PATH_UNIX_BYTES,
            },
            #[cfg(target_os = "windows")]
            PathView::Wide(units) => FjPathView {
                data: units.as_ptr().cast::<c_void>(),
                count: units.len(),
                encoding: FJ_PATH_WINDOWS_WIDE,
            },
        };
        if source_path.count == 0 {
            return Err(Failure::Internal("empty catalog source path"));
        }
        Ok(Self {
            key: text(view.key),
            label: text(view.label),
            source_path,
            polarity: match view.polarity {
                Polarity::Negative => FJ_POLARITY_NEGATIVE,
                Polarity::Positive => FJ_POLARITY_POSITIVE,
            },
        })
    }
}

fn text(value: &str) -> FjStringView {
    FjStringView {
        data: if value.is_empty() {
            std::ptr::null()
        } else {
            value.as_ptr().cast()
        },
        count: value.len(),
    }
}

enum Failure {
    Input(&'static str),
    Asset(AssetError),
    Capacity(TryReserveError),
    Internal(&'static str),
}
impl Failure {
    fn category(&self) -> u32 {
        match self {
            Self::Input(_) => FJ_STATUS_UNSUPPORTED_INPUT,
            Self::Asset(AssetError::Catalog(_)) => FJ_STATUS_PREPARATION_FAILURE,
            Self::Capacity(_) => FJ_STATUS_ALLOCATION_FAILURE,
            Self::Asset(_) | Self::Internal(_) => FJ_STATUS_INTERNAL_FAILURE,
        }
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Internal(message) => formatter.write_str(message),
            Self::Asset(error) => write!(formatter, "{error}"),
            Self::Capacity(error) => write!(formatter, "catalog path capacity: {error}"),
        }
    }
}
fn status(category: u32) -> FjStatus {
    FjStatus {
        category,
        api: FJ_API_NONE,
        native_code: 0,
    }
}

struct Diagnostic<'a> {
    data: *mut u8,
    capacity: usize,
    length: &'a mut usize,
}
impl Write for Diagnostic<'_> {
    fn write_str(&mut self, message: &str) -> fmt::Result {
        let count = message.len().min(self.capacity - 1 - *self.length);
        // SAFETY: run constructs this writer from disjoint, exclusive foreign
        // bytes. The checked capacity and initialized length bound every write.
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), self.data.add(*self.length), count);
            self.data.add(*self.length + count).write(0);
        }
        *self.length += count;
        Ok(())
    }
}

/// # Safety
/// A nonnull diagnostic is initialized, aligned and exclusively writable; its
/// nonempty backing extent is disjoint from all other call storage.
unsafe fn run<T>(
    error: *mut FjErrorBuffer,
    operation: impl FnOnce() -> Result<T, Failure>,
) -> Result<T, FjStatus> {
    // Captured consumed owners are dropped inside containment even when the
    // diagnostic is malformed and the ordinary operation must be skipped.
    catch_unwind(AssertUnwindSafe(|| {
        let mut diagnostic = if error.is_null() {
            None
        } else {
            // SAFETY: The caller supplies the record; zero capacity deliberately
            // does not read the data field, which need not be initialized.
            let (capacity, length) = unsafe {
                let capacity = std::ptr::addr_of!((*error).capacity).read();
                let length = std::ptr::addr_of_mut!((*error).length);
                length.write(0);
                (capacity, &mut *length)
            };
            if capacity == 0 {
                None
            } else {
                // SAFETY: Nonzero capacity requires an initialized data field.
                let data = unsafe { std::ptr::addr_of!((*error).data).read() };
                if data.is_null() || capacity > isize::MAX as usize {
                    return Err(status(FJ_STATUS_UNSUPPORTED_INPUT));
                }
                // SAFETY: The caller authorizes capacity writable bytes.
                unsafe { data.write(0) };
                Some(Diagnostic {
                    data: data.cast(),
                    capacity,
                    length,
                })
            }
        };
        let result = catch_unwind(AssertUnwindSafe(|| match operation() {
            Ok(value) => Ok(value),
            Err(failure) => {
                if let Some(diagnostic) = &mut diagnostic {
                    let _ = write!(diagnostic, "{failure}");
                }
                Err(status(failure.category()))
            }
        }));
        result.unwrap_or_else(|_| {
            if let Some(diagnostic) = &mut diagnostic {
                *diagnostic.length = 0;
                let _ = diagnostic.write_str("asset catalog boundary panic");
            }
            Err(status(FJ_STATUS_INTERNAL_FAILURE))
        })
    }))
    .unwrap_or(Err(status(FJ_STATUS_INTERNAL_FAILURE)))
}

/// # Safety
/// The caller authorizes the initialized, platform-native input extent.
unsafe fn path(view: FjPathView) -> Result<std::path::PathBuf, Failure> {
    if view.data.is_null() || view.count == 0 {
        return Err(Failure::Input("empty resource root"));
    }
    #[cfg(target_os = "linux")]
    let units = {
        if view.encoding != FJ_PATH_UNIX_BYTES || view.count > isize::MAX as usize {
            return Err(Failure::Input("invalid Unix path extent or encoding"));
        }
        // SAFETY: Bounds checked before forming the caller-authorized byte slice.
        PathView::Bytes(unsafe { std::slice::from_raw_parts(view.data.cast::<u8>(), view.count) })
    };
    #[cfg(target_os = "windows")]
    let units = {
        if view.encoding != FJ_PATH_WINDOWS_WIDE
            || view.count > isize::MAX as usize / 2
            || !(view.data as usize).is_multiple_of(std::mem::align_of::<u16>())
        {
            return Err(Failure::Input("invalid Windows path extent or encoding"));
        }
        // SAFETY: Checked aligned byte extent precedes the authorized u16 slice.
        PathView::Wide(unsafe { std::slice::from_raw_parts(view.data.cast::<u16>(), view.count) })
    };
    units.to_path_buf().map_err(|error| match error {
        PathError::Empty => Failure::Input("empty resource root"),
        PathError::ContainsNul => Failure::Input("resource root contains NUL"),
    })
}

// FJ_TEMP_BRIDGE: asset conversion; remove S4.E.
/// # Safety
/// Inputs have the documented initialized extent. Nonnull outputs/diagnostics
/// are aligned, exclusive and mutually disjoint. Root borrows expire at return.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_assets_create(
    resource_root: FjPathView,
    out_assets: *mut *mut FjAssets,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_assets.is_null() {
        // SAFETY: Caller provides one exclusive output slot.
        unsafe { out_assets.write(std::ptr::null_mut()) };
    }
    // SAFETY: Diagnostic/input/output storage follows the export contract.
    unsafe {
        run(error, || {
            if out_assets.is_null() {
                return Err(Failure::Input("NULL Assets output"));
            }
            let assets = Box::new(Assets::new(path(resource_root)?));
            #[cfg(feature = "test-support")]
            LIVE_ASSETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_assets.write(Box::into_raw(assets).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// assets is NULL or a live matching owner taken once; caller excludes all reads.
/// Diagnostics follow the exclusive disjoint storage contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_assets_destroy(
    assets: *mut FjAssets,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Matching live allocation is consumed, including rejected diagnostics.
    #[cfg(feature = "test-support")]
    if !assets.is_null() {
        LIVE_ASSETS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
    let owner = if assets.is_null() {
        None
    } else {
        // SAFETY: The matching live Box allocation is taken exactly once.
        Some(unsafe { Box::from_raw(assets.cast::<Assets>()) })
    };
    // SAFETY: Captured owner cleanup and diagnostics are contained by run.
    unsafe {
        run(error, move || {
            drop(owner);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// assets is live through return; outputs and diagnostics are exclusive and
/// disjoint. Catalog release is excluded from every returned-view use.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_catalog_acquire(
    assets: *const FjAssets,
    out_catalog: *mut *mut FjCatalog,
    out_counts: *mut FjCatalogCounts,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Each nonnull output is a caller-authorized exclusive slot.
    unsafe {
        if !out_catalog.is_null() {
            out_catalog.write(std::ptr::null_mut());
        }
        if !out_counts.is_null() {
            out_counts.write(FjCatalogCounts::default());
        }
    }
    // SAFETY: The live owner and disjoint output/diagnostic storage are authorized.
    unsafe {
        run(error, || {
            if assets.is_null() || out_catalog.is_null() || out_counts.is_null() {
                return Err(Failure::Input("NULL catalog acquisition input/output"));
            }
            let catalog = (&*assets.cast::<Assets>())
                .catalog()
                .map_err(Failure::Asset)?;
            #[cfg(any(test, feature = "test-support"))]
            inject()?;
            let owner = Box::new(CatalogOwner::new(catalog).map_err(Failure::Capacity)?);
            let counts = FjCatalogCounts {
                film_count: owner.count(Role::Film),
                print_count: owner.count(Role::Print),
            };
            #[cfg(feature = "test-support")]
            LIVE_CATALOGS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out_counts.write(counts);
            out_catalog.write(Box::into_raw(owner).cast());
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// catalog is live through all returned-view use. Outputs and diagnostics are
/// aligned, exclusive and disjoint from owner storage and other call storage.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_catalog_entry(
    catalog: *const FjCatalog,
    role: u32,
    index: usize,
    out_entry: *mut FjCatalogEntryView,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    if !out_entry.is_null() {
        // SAFETY: Caller authorizes this exclusive record.
        unsafe { out_entry.write(FjCatalogEntryView::empty()) };
    }
    // SAFETY: The owner and disjoint output/diagnostics obey the export contract.
    unsafe {
        run(error, || {
            if catalog.is_null() || out_entry.is_null() {
                return Err(Failure::Input("NULL catalog entry input/output"));
            }
            let role = match role {
                FJ_PROFILE_ROLE_FILM => Role::Film,
                FJ_PROFILE_ROLE_PRINT => Role::Print,
                _ => return Err(Failure::Input("unknown profile role")),
            };
            let view = (&*catalog.cast::<CatalogOwner>())
                .entry(role, index)
                .ok_or(Failure::Input("catalog index out of range"))?;
            out_entry.write(FjCatalogEntryView::from_view(view)?);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

/// # Safety
/// catalog is NULL or a matching live owner taken once; release excludes reads
/// and outstanding views. Diagnostics follow the exclusive storage contract.
#[unsafe(no_mangle)]
unsafe extern "C" fn fj_legacy_catalog_release(
    catalog: *mut FjCatalog,
    error: *mut FjErrorBuffer,
) -> FjStatus {
    // SAFETY: Matching live allocation is taken before diagnostic validation.
    #[cfg(feature = "test-support")]
    if !catalog.is_null() {
        LIVE_CATALOGS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
    let owner = if catalog.is_null() {
        None
    } else {
        // SAFETY: The matching live Box allocation is taken exactly once.
        Some(unsafe { Box::from_raw(catalog.cast::<CatalogOwner>()) })
    };
    // SAFETY: Cleanup of the captured owner is contained even on malformed error.
    unsafe {
        run(error, move || {
            drop(owner);
            Ok(())
        })
    }
    .map_or_else(|failure| failure, |()| status(FJ_STATUS_SUCCESS))
}

#[cfg(feature = "test-support")]
static LIVE_ASSETS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "test-support")]
static LIVE_CATALOGS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(any(test, feature = "test-support"))]
thread_local! { static FAULT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
#[cfg(any(test, feature = "test-support"))]
fn inject() -> Result<(), Failure> {
    match FAULT.replace(0) {
        1 => panic!("catalog projection fixture panic"),
        2 => {
            let mut units = Vec::<u16>::new();
            units.try_reserve(usize::MAX).map_err(Failure::Capacity)
        }
        #[cfg(target_os = "windows")]
        3 => {
            crate::asset_catalog::fail_next_path_encoding();
            Ok(())
        }
        _ => Ok(()),
    }
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_catalog_fault(fault: u32) {
    FAULT.set(fault);
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_assets_live_owners() -> usize {
    LIVE_ASSETS.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(feature = "test-support")]
#[unsafe(no_mangle)]
extern "C" fn fj_test_catalog_live_owners() -> usize {
    LIVE_CATALOGS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn layouts_and_explicit_tags() {
        assert_eq!(
            (size_of::<FjCatalogCounts>(), align_of::<FjCatalogCounts>()),
            (16, 8)
        );
        assert_eq!(
            (
                offset_of!(FjCatalogCounts, film_count),
                offset_of!(FjCatalogCounts, print_count)
            ),
            (0, 8)
        );
        assert_eq!(
            (
                size_of::<FjCatalogEntryView>(),
                align_of::<FjCatalogEntryView>()
            ),
            (64, 8)
        );
        assert_eq!(
            (
                offset_of!(FjCatalogEntryView, key),
                offset_of!(FjCatalogEntryView, label),
                offset_of!(FjCatalogEntryView, source_path),
                offset_of!(FjCatalogEntryView, polarity)
            ),
            (0, 16, 32, 56)
        );
        assert_eq!(
            (
                FJ_PROFILE_ROLE_FILM,
                FJ_PROFILE_ROLE_PRINT,
                FJ_POLARITY_NEGATIVE,
                FJ_POLARITY_POSITIVE
            ),
            (0, 1, 0, 1)
        );
    }

    #[test]
    fn rejected_diagnostics_drop_taken_capture_inside_containment() {
        struct Observed<'a>(&'a std::cell::Cell<bool>);
        impl Drop for Observed<'_> {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }
        let dropped = std::cell::Cell::new(false);
        let owner = Observed(&dropped);
        let mut malformed = FjErrorBuffer {
            data: std::ptr::null_mut(),
            capacity: 1,
            length: 99,
        };
        // SAFETY: Exclusive aligned error record; no backing accessed for malformed shape.
        let result = unsafe {
            run(&raw mut malformed, move || {
                drop(owner);
                Ok(())
            })
        };
        assert!(matches!(
            result,
            Err(FjStatus {
                category: FJ_STATUS_UNSUPPORTED_INPUT,
                ..
            })
        ));
        assert!(dropped.get());
        assert_eq!(malformed.length, 0);
    }

    #[test]
    fn panic_and_capacity_leave_acquisition_empty_and_retry_source() {
        let assets =
            Assets::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let borrowed = (&raw const assets).cast::<FjAssets>();
        for (fault, expected) in [
            (1, FJ_STATUS_INTERNAL_FAILURE),
            (2, FJ_STATUS_ALLOCATION_FAILURE),
        ] {
            FAULT.set(fault);
            let mut owner = std::ptr::null_mut();
            let mut counts = FjCatalogCounts {
                film_count: 999,
                print_count: 999,
            };
            // SAFETY: Live stack owner and exclusive disjoint outputs, null diagnostics.
            let result = unsafe {
                fj_legacy_catalog_acquire(
                    borrowed,
                    &raw mut owner,
                    &raw mut counts,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(result.category, expected);
            assert!(owner.is_null());
            assert_eq!((counts.film_count, counts.print_count), (0, 0));
        }
        let snapshot = assets.catalog().unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &snapshot,
            &assets.catalog().unwrap()
        ));
    }
}
