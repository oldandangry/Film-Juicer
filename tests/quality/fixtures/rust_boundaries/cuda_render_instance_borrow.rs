use film_juicer_core::{assets::Assets, recipe::RenderState};

fn expire_while_borrowed(
    owner: &mut crate::cuda::OwnerBorrow<'_>,
    assets: &Assets,
    state: &RenderState,
    frame: &crate::cuda::Frame<'_>,
    policy: &crate::cuda::FramePolicy,
    instance: crate::instance::Instance,
    abort: &crate::cuda::Abort<'_>,
) {
    let instance_ref = &instance;
    let _ = crate::cuda::render(owner, assets, state, frame, policy, instance_ref, abort);
    drop(instance);
    std::hint::black_box(instance_ref);
}
