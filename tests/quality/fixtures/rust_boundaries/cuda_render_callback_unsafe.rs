use crate::{
    cuda::{self, sys},
    instance::Instance,
    runtime::Runtime,
};
use film_juicer_core::recipe::RenderState;

fn raw_callback(
    runtime: &Runtime,
    state: &RenderState,
    frame: &sys::FjFrame,
    policy: &cuda::FramePolicy,
    instance: &Instance,
    abort: sys::FjAbortCallback,
) {
    let _ =
        cuda::edge::render_callback(runtime, state, frame, policy, instance, 0, abort, |_| false);
}
