#![allow(unsafe_code)]

use crate::{
    cuda::{self, sys},
    instance::Instance,
    runtime::Runtime,
};
use film_juicer_core::recipe::RenderState;

fn close(
    runtime: Runtime,
    state: &RenderState,
    frame: &sys::FjFrame,
    policy: &cuda::FramePolicy,
    instance: &Instance,
    abort: sys::FjAbortCallback,
) {
    let retained = &runtime;
    let _ = unsafe {
        cuda::edge::render_callback(retained, state, frame, policy, instance, 0, abort, |_| {
            false
        })
    };
    let _ = runtime.close();
    std::hint::black_box(retained);
}
