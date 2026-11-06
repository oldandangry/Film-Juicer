use std::path::PathBuf;

use crate::cuda::Owner;
use crate::runtime::Runtime;
use film_juicer_core::assets::Assets;

fn runtime_construction(resource_dir: PathBuf) -> Result<Runtime, crate::cuda::Failure> {
    Runtime::new(resource_dir)
}

fn runtime_shared_borrows(runtime: &Runtime) {
    let _: &Assets = runtime.assets();
    let _: &Owner = runtime.native();
    let _ = runtime.check_close();
    let _ = runtime.shutdown();
}

fn runtime_consumption(runtime: Runtime) {
    let _ = runtime.close();
}

fn assert_send_sync<T: Send + Sync>() {}

fn runtime_owner_is_send_sync() {
    assert_send_sync::<Runtime>();
    assert_send_sync::<Owner>();
}
