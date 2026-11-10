use crate::ofx::host::{Callback, InstanceHandles};

fn use_image(callback: &Callback<'_>, handles: &InstanceHandles) {
    if let Ok(Some(image)) = callback.fetch_image(handles, 0, 1.25) {
        std::hint::black_box(image.bounds);
        std::hint::black_box(image.scale);
    }
}
