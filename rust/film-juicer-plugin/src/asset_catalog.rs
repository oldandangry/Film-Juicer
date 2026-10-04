//! Retained catalog and native path views for the temporary production bridge.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStringExt;

use film_juicer_core::profile::{Catalog, Polarity, Role};

pub(crate) enum PathView<'a> {
    #[cfg(target_os = "linux")]
    Bytes(&'a [u8]),
    #[cfg(target_os = "windows")]
    Wide(&'a [u16]),
}

pub(crate) enum PathError {
    Empty,
    ContainsNul,
}

impl PathView<'_> {
    pub(crate) fn to_path_buf(&self) -> Result<PathBuf, PathError> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Bytes(units) => {
                if units.is_empty() {
                    return Err(PathError::Empty);
                }
                if units.contains(&0) {
                    return Err(PathError::ContainsNul);
                }
                Ok(std::ffi::OsStr::from_bytes(units).to_owned().into())
            }
            #[cfg(target_os = "windows")]
            Self::Wide(units) => {
                if units.is_empty() {
                    return Err(PathError::Empty);
                }
                if units.contains(&0) {
                    return Err(PathError::ContainsNul);
                }
                Ok(std::ffi::OsString::from_wide(units).into())
            }
        }
    }
}

pub(crate) struct CatalogOwner {
    catalog: Arc<Catalog>,
}

pub(crate) struct CatalogEntryView<'a> {
    pub key: &'a str,
    pub label: &'a str,
    pub polarity: Polarity,
}

impl CatalogOwner {
    pub(crate) fn new(catalog: Arc<Catalog>) -> Self {
        Self { catalog }
    }

    pub(crate) fn count(&self, role: Role) -> usize {
        match role {
            Role::Film => self.catalog.films().len(),
            Role::Print => self.catalog.prints().len(),
        }
    }

    pub(crate) fn entry(&self, role: Role, index: usize) -> Option<CatalogEntryView<'_>> {
        let entries = match role {
            Role::Film => self.catalog.films(),
            Role::Print => self.catalog.prints(),
        };
        let entry = entries.get(index)?;
        Some(CatalogEntryView {
            key: entry.key(),
            label: entry.label(),
            polarity: entry.polarity(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use film_juicer_core::assets::Assets;

    #[test]
    fn owner_views_retain_snapshot_and_native_units() {
        let assets =
            Assets::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let catalog = assets.catalog().unwrap();
        let source_bytes: usize = catalog
            .films()
            .iter()
            .chain(catalog.prints())
            .map(|entry| {
                let path_bytes = entry.source_path().as_os_str().as_encoded_bytes().len();
                std::mem::size_of_val(entry) + entry.key().len() + entry.label().len() + path_bytes
            })
            .sum();
        println!(
            "Catalog source entry/field logical payload={source_bytes} bytes; source container inline={} bytes; Arc retention shares source; no deep source copy",
            std::mem::size_of::<Catalog>()
        );
        let weak = Arc::downgrade(&catalog);
        let first = CatalogOwner::new(Arc::clone(&catalog));
        let second = CatalogOwner::new(catalog);
        println!(
            "Catalog retained handle inline={} bytes; two handles share one source allocation",
            std::mem::size_of::<CatalogOwner>()
        );
        drop(assets);
        let view = first.entry(Role::Film, 0).unwrap();
        let repeated = first.entry(Role::Film, 0).unwrap();
        assert_eq!(view.key.as_ptr(), repeated.key.as_ptr());
        assert_eq!(first.count(Role::Film), second.count(Role::Film));
        drop(second);
        assert_eq!(weak.strong_count(), 1);
        assert!(first.entry(Role::Print, first.count(Role::Print)).is_none());
        drop(first);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn native_units_are_lossless_without_unicode_admission() {
        #[cfg(target_os = "linux")]
        {
            let units = b"/root/\xff/film.json";
            let path = PathView::Bytes(units)
                .to_path_buf()
                .unwrap_or_else(|_| panic!("valid bytes"));
            assert_eq!(path.as_os_str().as_bytes(), units);
            assert!(matches!(
                PathView::Bytes(&[]).to_path_buf(),
                Err(PathError::Empty)
            ));
            assert!(matches!(
                PathView::Bytes(b"a\0b").to_path_buf(),
                Err(PathError::ContainsNul)
            ));
        }
        #[cfg(target_os = "windows")]
        {
            let units = [67, 58, 92, 0xd83d, 0xde00, 92, 0xd800];
            let path = PathView::Wide(&units)
                .to_path_buf()
                .unwrap_or_else(|_| panic!("valid units"));
            use std::os::windows::ffi::OsStrExt;
            assert_eq!(path.as_os_str().encode_wide().collect::<Vec<_>>(), units);
            assert!(matches!(
                PathView::Wide(&[]).to_path_buf(),
                Err(PathError::Empty)
            ));
            assert!(matches!(
                PathView::Wide(&[65, 0]).to_path_buf(),
                Err(PathError::ContainsNul)
            ));
        }
    }
}
