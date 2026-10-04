//! Source owners with borrow-tied spectral projections for the private C edge.

#![forbid(unsafe_code)]

use std::sync::Arc;
use film_juicer_core::assets::SpectraLut;
use film_juicer_core::data_io::{CsvPairs, CmfRows};

pub(crate) struct SpectraOwner {
    lut: Arc<SpectraLut>,
}

pub(crate) struct SpectraView<'a> {
    pub samples: &'a [f32],
    pub asset_hash: u64,
}

impl SpectraOwner {
    pub(crate) fn new(lut: Arc<SpectraLut>) -> Self {
        Self { lut }
    }
    pub(crate) fn view(&self) -> SpectraView<'_> {
        SpectraView {
            samples: self.lut.samples(),
            asset_hash: self.lut.asset_hash(),
        }
    }
}

pub(crate) struct MallettOwner {
    basis: Arc<[[f32; 3]; 81]>,
}
impl MallettOwner {
    pub(crate) fn new(basis: Arc<[[f32; 3]; 81]>) -> Self {
        Self { basis }
    }
    pub(crate) fn samples(&self) -> &[[f32; 3]; 81] {
        &self.basis
    }
}

pub(crate) struct CmfOwner {
    rows: Arc<CmfRows>,
}
impl CmfOwner {
    pub(crate) fn new(rows: Arc<CmfRows>) -> Self {
        Self { rows }
    }
    pub(crate) fn rows(&self) -> &[[f32; 4]] {
        self.rows.rows()
    }
}

pub(crate) struct CsvPairsOwner {
    rows: Arc<CsvPairs>,
}
impl CsvPairsOwner {
    pub(crate) fn new(rows: Arc<CsvPairs>) -> Self {
        Self { rows }
    }
    pub(crate) fn rows(&self) -> &[[f32; 2]] {
        self.rows.rows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use film_juicer_core::assets::Assets;
    use std::path::Path;

    #[test]
    fn csv_rows_borrow_source_and_survive_cache_and_assets_release() {
        let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let rows = assets
            .csv_source(film_juicer_core::assets::CsvSource::Kg3)
            .unwrap();
        let weak = Arc::downgrade(&rows);
        let owner = CsvPairsOwner::new(Arc::clone(&rows));
        assert_eq!(owner.rows().as_ptr(), rows.rows().as_ptr());
        assets.release_cached_payloads().unwrap();
        drop(assets);
        drop(rows);
        assert_eq!(weak.strong_count(), 1);
        assert_eq!(owner.rows().len(), 146);
        println!(
            "CSV Box payload={}, Arc count header=2*usize, no row copies in view",
            size_of::<CsvPairsOwner>()
        );
        drop(owner);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn projections_share_sources_and_expire_with_their_owners() {
        let assets = Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let hanatos = assets.hanatos().unwrap();
        let arctic = assets.arctic().unwrap();
        let basis = assets.mallett().unwrap();
        let rows = assets.cmf().unwrap();
        let weak = (
            Arc::downgrade(&hanatos),
            Arc::downgrade(&arctic),
            Arc::downgrade(&basis),
            Arc::downgrade(&rows),
        );
        let owners = (
            SpectraOwner::new(Arc::clone(&hanatos)),
            SpectraOwner::new(Arc::clone(&arctic)),
            MallettOwner::new(Arc::clone(&basis)),
            CmfOwner::new(Arc::clone(&rows)),
        );
        assert_eq!(owners.0.view().samples.as_ptr(), hanatos.samples().as_ptr());
        assert_eq!(owners.0.view().asset_hash, hanatos.asset_hash());
        assert_eq!(owners.1.view().samples.as_ptr(), arctic.samples().as_ptr());
        assert_eq!(owners.1.view().asset_hash, arctic.asset_hash());
        assert_eq!(owners.2.samples().as_ptr(), basis.as_ptr());
        assert_eq!(owners.3.rows().as_ptr(), rows.rows().as_ptr());
        assets.release_cached_payloads().unwrap();
        assert!(Arc::ptr_eq(&hanatos, &assets.hanatos().unwrap()));
        assert!(Arc::ptr_eq(&arctic, &assets.arctic().unwrap()));
        assert!(Arc::ptr_eq(&basis, &assets.mallett().unwrap()));
        assert!(Arc::ptr_eq(&rows, &assets.cmf().unwrap()));
        drop(assets);
        drop((hanatos, arctic, basis, rows));
        assert_eq!(
            (
                weak.0.strong_count(),
                weak.1.strong_count(),
                weak.2.strong_count(),
                weak.3.strong_count()
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(owners.0.view().samples.len(), 192 * 192 * 81);
        assert_eq!(owners.1.view().samples.len(), 192 * 192 * 81);
        assert_eq!(owners.2.samples().len(), 81);
        assert_eq!(owners.3.rows().len(), 81);
        println!(
            "source owner Box payload bytes: LUT={} Mallett={} CMF={}; Arc count header=2*usize; no sample copy",
            size_of::<SpectraOwner>(),
            size_of::<MallettOwner>(),
            size_of::<CmfOwner>()
        );
        drop(owners);
        assert!(weak.0.upgrade().is_none());
        assert!(weak.1.upgrade().is_none());
        assert!(weak.2.upgrade().is_none());
        assert!(weak.3.upgrade().is_none());
    }
}
