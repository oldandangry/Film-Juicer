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

fn use_instance(callback: &Callback<'_>) {
    let _owner = callback.instance();
    if let Ok(handles) = callback.instance_handles() {
        let parameters = callback.parameter_access(&handles);
        let _ = parameters.double(c"Exposure");
        let _ = parameters.integer(c"CameraAutoExposure");
        let _ = parameters.double2(c"ScannerUnsharpMask");
        let _ = parameters.double3(c"GrainParticleScale");
        let _ = parameters.string_bytes(c"FilmProfileKey");
    }
}
