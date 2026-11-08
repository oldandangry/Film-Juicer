use crate::ofx::host::{Admission, Callback};

fn construct_callback(admission: &Admission) {
    let _ = Callback::new(admission, std::ptr::null_mut());
}
