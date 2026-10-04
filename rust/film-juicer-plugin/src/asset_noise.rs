//! Complete noise source retention and owner-tied immutable byte views.

#![forbid(unsafe_code)]

use std::sync::Arc;
use film_juicer_core::assets::NoiseBundle;

pub(crate) struct NoiseOwner {
    bundle: Arc<NoiseBundle>,
}

pub(crate) struct NoiseView<'a> {
    pub stbn: &'a [u8],
    pub stbn_dimensions: [usize; 3],
    pub wang_tiles: &'a [u8],
    pub wang_lut: &'a [u8],
    pub wang_dimensions: [usize; 3],
    pub wang_colors: usize,
}

impl NoiseOwner {
    pub(crate) fn new(bundle: Arc<NoiseBundle>) -> Self {
        Self { bundle }
    }

    pub(crate) fn view(&self) -> NoiseView<'_> {
        NoiseView {
            stbn: self.bundle.stbn().bytes(),
            stbn_dimensions: self.bundle.stbn().dimensions(),
            wang_tiles: self.bundle.wang().tiles(),
            wang_lut: self.bundle.wang().lut(),
            wang_dimensions: self.bundle.wang().dimensions(),
            wang_colors: self.bundle.wang().colors(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use film_juicer_core::assets::Assets;
    use std::path::Path;

    #[test]
    fn views_share_complete_source_and_outlive_assets_until_last_hold() {
        let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let bundle = assets.noise().unwrap();
        let weak = Arc::downgrade(&bundle);
        let owner = NoiseOwner::new(Arc::clone(&bundle));
        let second = NoiseOwner::new(assets.noise().unwrap());
        let view = owner.view();
        assert_eq!(view.stbn.as_ptr(), bundle.stbn().bytes().as_ptr());
        assert_eq!(view.wang_tiles.as_ptr(), bundle.wang().tiles().as_ptr());
        assert_eq!(view.wang_lut.as_ptr(), bundle.wang().lut().as_ptr());
        assert_eq!(second.view().stbn.as_ptr(), view.stbn.as_ptr());
        assert_eq!(view.stbn.len(), 67_108_864);
        assert_eq!(view.wang_tiles.len(), 1_048_576);
        assert_eq!(view.wang_lut.len(), 16);
        assets.release_cached_payloads().unwrap();
        drop(assets);
        drop(bundle);
        assert_eq!(weak.strong_count(), 2);
        std::thread::scope(|scope| {
            scope.spawn(|| assert_eq!(owner.view().wang_lut, second.view().wang_lut));
            scope.spawn(|| assert_eq!(owner.view().stbn, second.view().stbn));
        });
        println!(
            "NoiseOwner={} bytes; NoiseView={} bytes; source=68157456 bytes; NoiseBundle={}; Arc header=2*usize; no source copy",
            size_of::<NoiseOwner>(),
            size_of::<NoiseView<'_>>(),
            size_of::<NoiseBundle>()
        );
        drop(second);
        assert_eq!(weak.strong_count(), 1);
        drop(owner);
        assert!(weak.upgrade().is_none());
    }
}
