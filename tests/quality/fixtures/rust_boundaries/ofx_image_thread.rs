use crate::ofx::host::{Callback, InstanceHandles};

fn move_image(callback: &Callback<'_>, handles: &InstanceHandles) {
    if let Ok(Some(image)) = callback.fetch_image(handles, 0, 1.25) {
        std::thread::scope(|scope| {
            let _ = scope.spawn(move || image);
        });
    }
}
