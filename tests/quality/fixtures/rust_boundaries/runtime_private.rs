use crate::cuda::Owner;
use crate::runtime::Runtime;

fn runtime_owners_are_private(runtime: &Runtime, owner: &Owner) {
    let _ = &runtime.assets;
    let _ = &runtime.native;
    let _ = &owner.handle;
}
