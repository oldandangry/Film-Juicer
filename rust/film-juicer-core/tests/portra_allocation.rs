//! Real allocation failures at the Portra cache and Film v1 preparation boundaries.

#![deny(unsafe_code)]

use std::alloc::Layout;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use film_juicer_core::assets::{
    AssetError, Assets, PortraGrainBank, PortraGrainError, PortraGrainTable,
};
use film_juicer_core::capture;
use film_juicer_core::film::grain::film_v1::{Controls, Raster, Selection};
use film_juicer_core::recipe::{self, FilmGrainFrame, PrepareError};
use film_juicer_core::route::ScanRoute;
use triomphe::Arc;

#[allow(
    unsafe_code,
    reason = "test-only injection at the real allocator boundary"
)]
mod allocation_fault {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FaultAllocator;
    static FAIL_SIZE: AtomicUsize = AtomicUsize::new(0);

    // SAFETY: Successful allocations and all deallocations use System with the
    // original layout. A fault reports failure by returning null.
    unsafe impl GlobalAlloc for FaultAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if FAIL_SIZE
                .compare_exchange(layout.size(), 0, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return ptr::null_mut();
            }
            // SAFETY: The caller supplies a valid nonzero allocation layout.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: The caller supplies the original System pointer and layout.
            unsafe { System.dealloc(pointer, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: FaultAllocator = FaultAllocator;

    // This binary runs one test so no other test can consume the armed fault.
    pub fn fail_once<T>(size: usize, operation: impl FnOnce() -> T) -> T {
        FAIL_SIZE.store(size, Ordering::Relaxed);
        let result = operation();
        assert_eq!(
            FAIL_SIZE.swap(0, Ordering::Relaxed),
            0,
            "{size}-byte allocation was not reached"
        );
        result
    }
}

fn shared_owner_size<T>() -> usize {
    // triomphe 0.1.17 uses a repr(C) atomic count followed by the value.
    Layout::new::<AtomicUsize>()
        .extend(Layout::new::<T>())
        .unwrap()
        .0
        .pad_to_align()
        .size()
}

#[test]
fn portra_allocation_failure_preserves_cache_and_retry() {
    use allocation_fault::fail_once;

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let resource = root.join("Resources");
    let assets = Assets::new(resource.clone());
    let mut params = capture::defaults();
    params.film_profile_key = "kodak_portra_400".into();
    params.scan_route = ScanRoute::NegativeDirect;
    params.spectral_upsampling_mode = 1;
    params.camera_film_format_long_edge_mm = 8.0;
    params.film_grain = Selection::FilmV1(Controls {
        amount_ev: 0.0,
        texel_pitch_um: 4.0,
        chroma_fraction: 1.0,
    });
    let frame = FilmGrainFrame::Raster(Raster {
        rendered_long_edge_px: 4_000,
        candidate_full_grid: None,
    });
    let prepare = || recipe::prepare_for_frame(&assets, &params, frame);
    let bank = assets.portra_grain_bank().unwrap();
    assert!(Arc::ptr_eq(&bank, &assets.portra_grain_bank().unwrap()));
    assets.film(&params.film_profile_key).unwrap();
    let old_table = assets.portra_grain_table(&bank, 0.25).unwrap();

    let scratch = 17 * 128 * 128 * size_of::<f64>();
    let scratch_failure = fail_once(scratch, prepare);
    assert!(matches!(
        scratch_failure,
        Err(PrepareError::Asset(AssetError::PortraGrain(
            PortraGrainError::Allocation(_)
        )))
    ));
    assert!(Arc::ptr_eq(
        &old_table,
        &assets.portra_grain_table(&bank, 0.25).unwrap()
    ));

    let owner_failure = fail_once(shared_owner_size::<PortraGrainTable>(), prepare);
    assert!(matches!(
        owner_failure,
        Err(PrepareError::Asset(AssetError::PortraGrain(
            PortraGrainError::OwnerAllocation
        )))
    ));
    assert!(Arc::ptr_eq(
        &old_table,
        &assets.portra_grain_table(&bank, 0.25).unwrap()
    ));

    let state = prepare().unwrap();
    let table = state.film_grain().unwrap().table();
    assert_eq!(table.footprint.to_bits(), 0.5_f32.to_bits());
    assert!(std::ptr::eq(
        table,
        &*assets.portra_grain_table(&bank, 0.5).unwrap()
    ));

    let assets = Assets::new(resource.clone());
    let failed_bank = fail_once(shared_owner_size::<PortraGrainBank>(), || {
        assets.portra_grain_bank()
    });
    assert!(matches!(
        failed_bank,
        Err(AssetError::PortraGrain(PortraGrainError::OwnerAllocation))
    ));
    assert!(matches!(
        assets.portra_grain_bank(),
        Err(AssetError::PortraGrain(PortraGrainError::OwnerAllocation))
    ));

    let assets = Assets::new(resource);
    let sums_bytes = 3 * 17 * 257 * 257 * size_of::<f32>();
    let error = fail_once(sums_bytes, || assets.portra_grain_bank()).unwrap_err();
    let cause = error
        .source()
        .unwrap()
        .downcast_ref::<PortraGrainError>()
        .unwrap();
    assert!(matches!(cause, PortraGrainError::Allocation(_)));
    assert!(cause.source().is_some());
    assert!(matches!(
        assets.portra_grain_bank(),
        Err(AssetError::PortraGrain(PortraGrainError::Allocation(_)))
    ));

    let scratch = std::env::var_os("JUICER_TEST_ARTIFACT_DIR")
        .map_or_else(
            || root.join("out/validation/rust-core"),
            std::path::PathBuf::from,
        )
        .join(format!("portra-allocation-{}", std::process::id()));
    let bank_dir = scratch.join("grain/portra-400");
    fs::create_dir_all(&bank_dir).unwrap();
    let assets = Assets::new(scratch.clone());
    let failed_io_error = fail_once(shared_owner_size::<std::io::Error>(), || {
        assets.portra_grain_bank()
    });
    assert!(matches!(
        failed_io_error,
        Err(AssetError::PortraGrain(PortraGrainError::OwnerAllocation))
    ));
    assert!(matches!(
        assets.portra_grain_bank(),
        Err(AssetError::PortraGrain(PortraGrainError::OwnerAllocation))
    ));

    fs::write(bank_dir.join("manifest.json"), b"{").unwrap();
    let assets = Assets::new(scratch.clone());
    let failed_manifest_error = fail_once(shared_owner_size::<serde_json::Error>(), || {
        assets.portra_grain_bank()
    });
    assert!(matches!(
        failed_manifest_error,
        Err(AssetError::PortraGrain(PortraGrainError::OwnerAllocation))
    ));
    assert!(matches!(
        assets.portra_grain_bank(),
        Err(AssetError::PortraGrain(PortraGrainError::OwnerAllocation))
    ));
    fs::remove_dir_all(scratch).unwrap();
}
