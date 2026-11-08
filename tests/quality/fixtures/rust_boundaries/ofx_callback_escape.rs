use crate::ofx::host::{Callback, Properties, Status};

fn escape_properties<'a>(callback: Callback<'a>) -> Result<Properties<'a>, Status> {
    callback.descriptor_properties()
}
