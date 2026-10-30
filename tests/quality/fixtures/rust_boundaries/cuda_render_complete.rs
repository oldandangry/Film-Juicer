use film_juicer_core::{assets::Assets, recipe::RenderState};

fn consume(
    owner: &mut crate::cuda::OwnerBorrow<'_>,
    assets: &Assets,
    state: &RenderState,
    frame: &crate::cuda::Frame<'_>,
    policy: &crate::cuda::FramePolicy,
    latch: &mut crate::cuda::SubmissionLatch<'_>,
    abort: &crate::cuda::Abort<'_>,
) -> Result<crate::cuda::RenderOutcome, crate::cuda::RenderError> {
    crate::cuda::render(owner, assets, state, frame, policy, latch, abort)
}
