use crate::ofx::host::{Admission, Callback, Parameters};

fn construct_callback(admission: &Admission) {
    let _ = Callback::new(admission, std::ptr::null_mut(), std::ptr::null_mut());
}

fn construct_parameters(callback: &Callback<'_>) {
    let _ = Parameters::new(callback, std::ptr::null_mut());
}
