use film_juicer_core::{assets::Assets, recipe::RenderState};

fn pass_raw_records(
    owner: &mut crate::cuda::OwnerBorrow<'_>,
    assets: &Assets,
    state: &RenderState,
    raw_frame: &crate::cuda::sys::FjFrame,
    frame: &crate::cuda::Frame<'_>,
    policy: &crate::cuda::FramePolicy,
    latch: &mut crate::cuda::SubmissionLatch<'_>,
    raw_abort: &crate::cuda::sys::FjAbortCallback,
    abort: &crate::cuda::Abort<'_>,
) {
    let _ = crate::cuda::render(owner, assets, state, raw_frame, policy, latch, abort);
    let _ = crate::cuda::render(owner, assets, state, frame, policy, latch, raw_abort);
}
