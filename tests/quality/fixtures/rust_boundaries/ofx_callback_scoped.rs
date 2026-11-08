use std::ffi::CStr;

use crate::ofx::host::Callback;

fn use_descriptor(callback: &Callback<'_>, name: &CStr, value: &CStr) {
    if let Ok(properties) = callback.descriptor_properties() {
        let _ = properties.set_string(name, 0, value);
        let _ = properties.set_int(name, 0, 1);
        let _ = properties.dimension(name);
    }
    if let Ok(Some(parameters)) = callback.context_parameters()
        && let Ok(properties) = parameters.define(c"OfxParamTypeDouble", name)
    {
        let _ = properties.set_double(name, 0, 1.0);
        let _ = properties.set_catalog_string(name, 0, "catalog entry");
    }
}
