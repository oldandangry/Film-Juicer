use crate::ofx::host::{Admission, Callback, Parameters};

fn construct_callback(admission: &Admission) {
    let _ = Callback::new(admission, std::ptr::null_mut(), std::ptr::null_mut());
}

fn destroy_instance(callback: &Callback<'_>) {
    let _ = callback.destroy();
}

fn construct_parameters(callback: &Callback<'_>) {
    let _ = Parameters::new(callback, std::ptr::null_mut());
}
