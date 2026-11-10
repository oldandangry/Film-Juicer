//! Process host ownership and terminal ordering, independent of native resources.

#![forbid(unsafe_code)]
#![expect(
    clippy::result_large_err,
    reason = "terminal orchestration preserves a bounded diagnostic without allocating after failure"
)]

use std::path::PathBuf;
use std::sync::OnceLock;

use film_juicer_core::assets::Assets;

use crate::cuda::{self, Failure};

pub(crate) struct Runtime {
    assets: Option<Assets>,
    native: Option<cuda::Owner>,
    cache_cleanup: OnceLock<Result<(), Failure>>,
}

impl Runtime {
    pub(crate) fn new(resource_dir: PathBuf) -> Result<Self, Failure> {
        let assets = crate::asset_bridge::runtime_assets(resource_dir);
        // Cover host construction before native registration, including failure.
        let mut runtime = Self {
            assets: Some(assets),
            native: None,
            cache_cleanup: OnceLock::new(),
        };
        #[cfg(any(test, feature = "test-support"))]
        crate::asset_bridge::runtime_checkpoint(1)?;
        runtime.native = Some(cuda::Owner::new()?);
        #[cfg(any(test, feature = "test-support"))]
        crate::asset_bridge::runtime_checkpoint(2)?;
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

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn shutdown(&self) -> Result<(), Failure> {
        self.native().shutdown()?;
        *self
            .cache_cleanup
            .get_or_init(|| crate::asset_bridge::release_runtime_caches(self.assets()))
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
        let cached_result = self.cache_cleanup.get().copied().unwrap_or(Ok(()));
        let host_result = assets.map_or(Ok(()), crate::asset_bridge::close_runtime_assets);
        native_result.and(cached_result).and(host_result)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
