use film_juicer_core::gamut::{OutputTransform, OutputBoundary};
fn fake(transform: OutputTransform, boundary: OutputBoundary) {
    let _ = OutputTransform { ..transform };
    let _ = OutputBoundary { ..boundary };
}
