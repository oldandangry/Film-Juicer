//! Raw property reads for callback-local action arguments.

use std::ffi::{CStr, c_void};

use super::{Callback, Status};

impl Callback<'_> {
    pub(crate) fn action_field(&self) -> Result<Vec<u8>, Status> {
        let mut value = std::ptr::null_mut();
        // SAFETY: The input and suite are callback-live; the local output is
        // unique and no lock or exclusive borrow spans this reentrant call.
        Status::property(unsafe {
            (self.loaded.suites.get_string)(
                self.input,
                c"OfxImageEffectPropFieldToRender".as_ptr(),
                0,
                &mut value,
            )
        })?;
        if value.is_null() {
            return Ok(Vec::new());
        }
        // SAFETY: The host returns a terminated string valid until the next host
        // call. Copy it now without retaining or transferring its pointer.
        super::copy_bytes(unsafe { CStr::from_ptr(value) }.to_bytes())
    }

    pub(crate) fn action_int(&self, name: &CStr, index: i32) -> Result<i32, Status> {
        let mut value = 0;
        // SAFETY: The host supplies a live input property set for this synchronous
        // action, with suite functions valid through the retained admission. The
        // local output is uniquely borrowed for the call; no pointer is retained.
        // Calls may reenter and no lock or exclusive owner borrow spans the call.
        Status::property(unsafe {
            (self.loaded.suites.get_int)(self.input, name.as_ptr(), index, &mut value)
        })?;
        Ok(value)
    }

    pub(crate) fn optional_int(&self, name: &CStr) -> i32 {
        let mut value = 0;
        // SAFETY: The admitted callback's input and suite remain live for this
        // synchronous, potentially reentrant host call. The local output is
        // unique, no pointer escapes and no lock or exclusive borrow is held.
        let _ = unsafe { (self.loaded.suites.get_int)(self.input, name.as_ptr(), 0, &mut value) };
        value
    }

    pub(crate) fn optional_pointer(&self, name: &CStr) -> *mut c_void {
        let mut value = std::ptr::null_mut();
        // SAFETY: The admitted callback's input and suite remain live for this
        // synchronous, potentially reentrant host call. The local output is
        // unique, no pointer escapes the callback and no lock or exclusive borrow
        // is held. Any returned host pointer is opaque and not dereferenced here.
        let _ =
            unsafe { (self.loaded.suites.get_pointer)(self.input, name.as_ptr(), 0, &mut value) };
        value
    }
}
