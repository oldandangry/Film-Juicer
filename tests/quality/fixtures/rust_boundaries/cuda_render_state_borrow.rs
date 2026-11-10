use film_juicer_core::{assets::Assets, recipe::RenderState};

fn expire_while_borrowed(
    owner: &mut crate::cuda::OwnerBorrow<'_>,
    assets: &Assets,
    state: RenderState,
    frame: &crate::cuda::Frame<'_>,
    policy: &crate::cuda::FramePolicy,
    instance: &crate::instance::Instance,
    abort: &crate::cuda::Abort<'_>,
) {
    let retained = state.s_inverse();
    let _ = crate::cuda::render(owner, assets, &state, frame, policy, instance, abort);
    drop(state);
    let _ = retained[0];
}
