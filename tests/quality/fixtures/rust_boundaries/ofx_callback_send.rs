use crate::ofx::host::{Callback, Parameters};

fn move_callback_to_worker(callback: Callback<'_>) {
    std::thread::scope(|scope| {
        let _ = scope.spawn(move || callback);
    });
}

fn move_parameters_to_worker(parameters: Parameters<'_>) {
    std::thread::scope(|scope| {
        let _ = scope.spawn(move || parameters);
    });
}
