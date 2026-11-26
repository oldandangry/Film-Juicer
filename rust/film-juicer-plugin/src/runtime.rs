//! Process host ownership and terminal ordering, independent of native resources.

#![forbid(unsafe_code)]
#![expect(
    clippy::result_large_err,
    reason = "terminal orchestration preserves a bounded diagnostic without allocating after failure"
)]

use std::path::PathBuf;

use film_juicer_core::assets::Assets;

use crate::cuda::{self, Failure};

pub(crate) struct Runtime {
    assets: Option<Assets>,
    native: Option<cuda::Owner>,
}

impl Runtime {
    pub(crate) fn new(resource_dir: PathBuf) -> Result<Self, Failure> {
        let assets = Assets::new(resource_dir);
        // Cover host construction before native registration, including failure.
        let mut runtime = Self {
            assets: Some(assets),
            native: None,
        };
        runtime.native = Some(cuda::Owner::new()?);
        Ok(runtime)
    }

    pub(crate) fn assets(&self) -> &Assets {
        self.assets.as_ref().expect("live runtime Assets")
    }

    pub(crate) fn native(&self) -> &cuda::Owner {
        self.native.as_ref().expect("live runtime native owner")
    }

    pub(crate) fn check_close(&self) -> Result<(), Failure> {
        self.native().check_close()
    }

    pub(crate) fn close(mut self) -> Result<(), Failure> {
        self.finish()
    }

    fn finish(&mut self) -> Result<(), Failure> {
        // Move both owners out explicitly: Assets outlives the native terminal
        // attempt on every result, independently of struct field-drop ordering.
        let assets = self.assets.take();
        let native = self.native.take();
        let native_result = native.map_or(Ok(()), cuda::Owner::close);
        drop(assets);
        native_result
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
