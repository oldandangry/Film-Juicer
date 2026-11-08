#![allow(unsafe_code)]

use crate::{
    cuda::{self, sys},
    instance::Instance,
    runtime::Runtime,
};
use film_juicer_core::recipe::RenderState;

fn send(
    runtime: &Runtime,
    state: &RenderState,
    frame: &sys::FjFrame,
    policy: &cuda::FramePolicy,
    instance: &Instance,
    abort: sys::FjAbortCallback,
) {
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
                std::thread::scope(|scope| {
                    let _ = scope.spawn(move || completion);
                });
                false
            },
        )
    };
}
