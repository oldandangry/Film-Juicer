use crate::ofx::host::Callback;

fn move_callback_to_worker(callback: Callback<'_>) {
    std::thread::scope(|scope| {
        let _ = scope.spawn(move || callback);
    });
}
