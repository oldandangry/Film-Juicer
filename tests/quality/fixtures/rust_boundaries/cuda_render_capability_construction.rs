fn fabricate_policy(policy: crate::cuda::FramePolicy) {
    let _ = crate::cuda::FramePolicy { ..policy };
}
