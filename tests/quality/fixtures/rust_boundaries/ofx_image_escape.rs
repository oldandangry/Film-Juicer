use crate::ofx::host::{Callback, InstanceHandles, Status};

fn escape_image<'a>(
    callback: Callback<'a>,
    handles: &'a InstanceHandles,
) -> Result<impl Sized + 'a, Status> {
    callback.fetch_image(handles, 0, 1.25)
}
