use crate::ofx::host::{Callback, Parameters, ParameterAccess};

fn move_callback_to_worker(callback: Callback<'_>) {
    std::thread::scope(|scope| {
        let _ = scope.spawn(move || callback);
    });
}

fn move_instance_parameters_to_worker(instance_parameters: ParameterAccess<'_>) {
    std::thread::scope(|scope| {
        let _ = scope.spawn(move || instance_parameters);
    });
}

fn move_parameters_to_worker(parameters: Parameters<'_>) {
    std::thread::scope(|scope| {
        let _ = scope.spawn(move || parameters);
    });
}

fn move_instance_owner_to_worker(callback: &Callback<'_>) {
    let owner = callback.instance().unwrap();
    std::thread::scope(|scope| {
        let _ = scope.spawn(move || owner);
    });
}
