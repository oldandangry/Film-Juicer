use crate::cuda::Owner;

fn reuse_consumed_native_owner(owner: Owner) {
    let _ = owner.close();
    let _ = owner.raw();
}
