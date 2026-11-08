use crate::ofx::host::{Callback, Parameters, Properties, Status};

fn escape_properties<'a>(callback: Callback<'a>) -> Result<Properties<'a>, Status> {
    callback.descriptor_properties()
}

fn escape_parameters<'a>(callback: Callback<'a>) -> Result<Option<Parameters<'a>>, Status> {
    callback.context_parameters()
}
