use std::ffi::CStr;

use crate::ofx::host::Callback;

fn use_descriptor(callback: &Callback<'_>, name: &CStr, value: &CStr) {
    if let Ok(properties) = callback.descriptor_properties() {
        let _ = properties.set_string(name, 0, value);
        let _ = properties.set_int(name, 0, 1);
        let _ = properties.dimension(name);
    }
}
