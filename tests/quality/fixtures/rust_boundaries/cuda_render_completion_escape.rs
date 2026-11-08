#![allow(unsafe_code)]

use crate::{
    cuda::{self, sys},
    instance::Instance,
    runtime::Runtime,
};
use film_juicer_core::recipe::RenderState;

fn escape(
    runtime: &Runtime,
    state: &RenderState,
    frame: &sys::FjFrame,
    policy: &cuda::FramePolicy,
    instance: &Instance,
    abort: sys::FjAbortCallback,
) {
    let mut escaped = None;
    let _ = unsafe {
        cuda::edge::render_callback(
            runtime,
            state,
            frame,
            policy,
            instance,
            0,
            abort,
            |completion| {
                escaped = Some(completion);
                false
            },
        )
    };
    std::hint::black_box(escaped);
}
