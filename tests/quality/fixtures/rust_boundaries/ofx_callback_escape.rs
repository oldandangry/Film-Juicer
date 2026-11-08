use crate::ofx::host::{Callback, Parameters, Properties, Status, InstanceHandles, ParameterAccess};
use std::any::Any;

fn escape_properties<'a>(callback: Callback<'a>) -> Result<Properties<'a>, Status> {
    callback.descriptor_properties()
}

fn escape_instance_parameters<'a>(
    callback: &'a Callback<'a>,
    handles: InstanceHandles,
) -> ParameterAccess<'a> {
    callback.parameter_access(&handles)
}

fn escape_parameters<'a>(callback: Callback<'a>) -> Result<Option<Parameters<'a>>, Status> {
    callback.context_parameters()
}

fn escape_instance_owner(callback: Callback<'_>) -> Result<&'static dyn Any, Status> {
    Ok(callback.instance()?)
}
