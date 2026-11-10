#![allow(unsafe_code)]

use crate::{
    cuda::{self, sys},
    instance::Instance,
    runtime::Runtime,
};
use film_juicer_core::{frame::Domain, recipe::RenderState};

fn callback(
    runtime: &Runtime,
    state: &RenderState,
    frame: &sys::FjFrame,
    instance: &Instance,
    abort: sys::FjAbortCallback,
    domain: Domain,
) {
    let policy = cuda::FramePolicy::prepare(state, domain).unwrap();
    // The real edge requires caller-provided live image and abort backing.
    let _ = unsafe {
        cuda::edge::render_callback(
            runtime,
            state,
            frame,
            &policy,
            instance,
            0,
            abort,
            |completion| completion.fatal(),
        )
    };
}
