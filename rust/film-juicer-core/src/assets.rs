//! Process asset ownership. Published handles outlive cache release and this owner.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::data_io::{ReadError, load_mallett_basis, load_spectra_lut};
use crate::hash;
use crate::profile::{
    Catalog, CatalogError, FilmProfile, PrintProfile, ProfileCompletionError, ProfileError, Role,
    load_catalog, load_film_source, load_print_source,
};

#[derive(Debug)]
pub enum AssetError {
    Catalog(Arc<CatalogError>),
    MissingProfile { role: Role, key: String },
    ProfileDecode(ProfileError),
    ProfileCompletion(ProfileCompletionError),
    ProfileCachePoisoned { role: Role },
    Reconstruction(Arc<ReadError>),
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Catalog(source) => write!(f, "{source}"),
            Self::MissingProfile { role, key } => write!(f, "missing {role:?} profile key {key}"),
            Self::ProfileDecode(source) => write!(f, "{source}"),
            Self::ProfileCompletion(source) => write!(f, "{source}"),
            Self::ProfileCachePoisoned { role } => write!(f, "{role:?} profile cache poisoned"),
            Self::Reconstruction(source) => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for AssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Catalog(source) => Some(source.as_ref()),
            Self::ProfileDecode(source) => Some(source),
            Self::ProfileCompletion(source) => Some(source),
            Self::Reconstruction(source) => Some(source.as_ref()),
            Self::MissingProfile { .. } | Self::ProfileCachePoisoned { .. } => None,
        }
    }
}

/// One decoded 192x192x81 C-order reconstruction LUT and its resource identity.
/// Sample borrows remain valid while their retaining handle is borrowed.
#[derive(Debug)]
pub struct SpectraLut {
    samples: Vec<f32>,
    asset_hash: u64,
}

impl SpectraLut {
    fn new(samples: Vec<f32>) -> Self {
        let asset_hash = nonzero_asset_hash(hash::resource_f32s(&samples));
        Self {
            samples,
            asset_hash,
        }
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn asset_hash(&self) -> u64 {
        self.asset_hash
    }
}

struct CatalogSnapshot {
    catalog: Arc<Catalog>,
    films: HashMap<String, Mutex<Option<Arc<FilmProfile>>>>,
    prints: HashMap<String, Mutex<Option<Arc<PrintProfile>>>>,
}

// The accepted reader's wavelength-major RGB representation, without a wrapper.
type MallettBasis = [[f32; 3]; 81];

pub struct Assets {
    resource_dir: PathBuf,
    catalog: OnceLock<Result<CatalogSnapshot, Arc<CatalogError>>>,
    hanatos: OnceLock<Result<Arc<SpectraLut>, Arc<ReadError>>>,
    arctic: OnceLock<Result<Arc<SpectraLut>, Arc<ReadError>>>,
    mallett: OnceLock<Result<Arc<MallettBasis>, Arc<ReadError>>>,
}

impl Assets {
    /// Fix the resource root without discovering or loading resources.
    pub fn new(resource_dir: PathBuf) -> Self {
        Self {
            resource_dir,
            catalog: OnceLock::new(),
            hanatos: OnceLock::new(),
            arctic: OnceLock::new(),
            mallett: OnceLock::new(),
        }
    }

    fn snapshot(&self) -> Result<&CatalogSnapshot, AssetError> {
        // Catalog discovery is once per owner, including ordinary errors. The
        // finite key storage is built here before any profile publication lock.
        self.catalog
            .get_or_init(|| {
                let catalog = load_catalog(&self.resource_dir).map_err(Arc::new)?;
                let films = catalog
                    .films()
                    .iter()
                    .map(|entry| (entry.key().to_owned(), Mutex::new(None)))
                    .collect();
                let prints = catalog
                    .prints()
                    .iter()
                    .map(|entry| (entry.key().to_owned(), Mutex::new(None)))
                    .collect();
                Ok(CatalogSnapshot {
                    catalog: Arc::new(catalog),
                    films,
                    prints,
                })
            })
            .as_ref()
            .map_err(|source| AssetError::Catalog(Arc::clone(source)))
    }

    pub fn catalog(&self) -> Result<Arc<Catalog>, AssetError> {
        self.snapshot()
            .map(|snapshot| Arc::clone(&snapshot.catalog))
    }

    pub fn film(&self, key: &str) -> Result<Arc<FilmProfile>, AssetError> {
        self.film_with(key, load_film)
    }

    pub fn print(&self, key: &str) -> Result<Arc<PrintProfile>, AssetError> {
        self.print_with(key, load_print)
    }

    /// Retain the first complete success or failure for this Assets lifetime.
    pub fn hanatos(&self) -> Result<Arc<SpectraLut>, AssetError> {
        self.hanatos_with(load_spectra)
    }

    /// Requested independently of Hanatos, Mallett and catalog discovery.
    pub fn arctic(&self) -> Result<Arc<SpectraLut>, AssetError> {
        self.arctic_with(load_spectra)
    }

    /// Wavelength-major RGB basis; no standalone resource identity is needed.
    pub fn mallett(&self) -> Result<Arc<MallettBasis>, AssetError> {
        self.mallett_with(|path| load_mallett_basis(path).map(Arc::new))
    }

    fn hanatos_with(
        &self,
        load: impl FnOnce(&Path) -> Result<Arc<SpectraLut>, ReadError>,
    ) -> Result<Arc<SpectraLut>, AssetError> {
        if self.hanatos.get().is_none() {
            let path = self
                .resource_dir
                .join("luts/spectral_upsampling/irradiance_xy_tc.npy");
            let prepared = load(&path).map_err(Arc::new);
            // Prepare before set; destroy a losing complete outcome afterward.
            // No get_or_init: its initializer would serialize loading/hashing.
            drop(self.hanatos.set(prepared));
        }
        self.hanatos
            .get()
            .expect("a complete reconstruction outcome has been published")
            .as_ref()
            .map(Arc::clone)
            .map_err(|source| AssetError::Reconstruction(Arc::clone(source)))
    }

    fn arctic_with(
        &self,
        load: impl FnOnce(&Path) -> Result<Arc<SpectraLut>, ReadError>,
    ) -> Result<Arc<SpectraLut>, AssetError> {
        if self.arctic.get().is_none() {
            let path = self
                .resource_dir
                .join("luts/spectral_upsampling/arctic2026beta04_reflectance_xy_tc.npy");
            let prepared = load(&path).map_err(Arc::new);
            drop(self.arctic.set(prepared));
        }
        self.arctic
            .get()
            .expect("a complete reconstruction outcome has been published")
            .as_ref()
            .map(Arc::clone)
            .map_err(|source| AssetError::Reconstruction(Arc::clone(source)))
    }

    fn mallett_with(
        &self,
        load: impl FnOnce(&Path) -> Result<Arc<MallettBasis>, ReadError>,
    ) -> Result<Arc<MallettBasis>, AssetError> {
        if self.mallett.get().is_none() {
            let path = self
                .resource_dir
                .join("luts/spectral_upsampling/mallett2019_basis.npy");
            let prepared = load(&path).map_err(Arc::new);
            drop(self.mallett.set(prepared));
        }
        self.mallett
            .get()
            .expect("a complete reconstruction outcome has been published")
            .as_ref()
            .map(Arc::clone)
            .map_err(|source| AssetError::Reconstruction(Arc::clone(source)))
    }

    // Private synchronous preparation seams also let unit tests order cold
    // attempts without installing callbacks or fault controls in the owner.
    fn film_with(
        &self,
        key: &str,
        load: impl FnOnce(&Path) -> Result<Arc<FilmProfile>, AssetError>,
    ) -> Result<Arc<FilmProfile>, AssetError> {
        let snapshot = self.snapshot()?;
        let entry = snapshot
            .catalog
            .film(key)
            .ok_or_else(|| AssetError::MissingProfile {
                role: Role::Film,
                key: key.to_owned(),
            })?;
        let slot = &snapshot.films[key];
        let cached = slot
            .lock()
            .map_err(|_| AssetError::ProfileCachePoisoned { role: Role::Film })?
            .as_ref()
            .map(Arc::clone);
        if let Some(profile) = cached {
            return Ok(profile);
        }
        let prepared = load(entry.source_path());
        let published = {
            let mut cached = slot
                .lock()
                .map_err(|_| AssetError::ProfileCachePoisoned { role: Role::Film })?;
            if cached.is_none()
                && let Ok(profile) = &prepared
            {
                *cached = Some(Arc::clone(profile));
            }
            cached.as_ref().map(Arc::clone)
        };
        // Recheck even after local failure. Losing payloads/errors are dropped
        // after the guard, and only a successful same-key winner may replace them.
        published.map(Ok).unwrap_or(prepared)
    }

    fn print_with(
        &self,
        key: &str,
        load: impl FnOnce(&Path) -> Result<Arc<PrintProfile>, AssetError>,
    ) -> Result<Arc<PrintProfile>, AssetError> {
        let snapshot = self.snapshot()?;
        let entry = snapshot
            .catalog
            .print(key)
            .ok_or_else(|| AssetError::MissingProfile {
                role: Role::Print,
                key: key.to_owned(),
            })?;
        let slot = &snapshot.prints[key];
        let cached = slot
            .lock()
            .map_err(|_| AssetError::ProfileCachePoisoned { role: Role::Print })?
            .as_ref()
            .map(Arc::clone);
        if let Some(profile) = cached {
            return Ok(profile);
        }
        let prepared = load(entry.source_path());
        let published = {
            let mut cached = slot
                .lock()
                .map_err(|_| AssetError::ProfileCachePoisoned { role: Role::Print })?;
            if cached.is_none()
                && let Ok(profile) = &prepared
            {
                *cached = Some(Arc::clone(profile));
            }
            cached.as_ref().map(Arc::clone)
        };
        published.map(Ok).unwrap_or(prepared)
    }

    /// Detach profile cache holds; catalog and reconstruction outcomes remain fixed. Retained
    /// handles stay valid. An overlapping cold load may publish after release;
    /// callers requiring a drained cache must first exclude new loads.
    pub fn release_cached_payloads(&self) -> Result<(), AssetError> {
        // Release on a new owner must not trigger catalog I/O.
        let Some(snapshot) = self.catalog.get() else {
            return Ok(());
        };
        let Ok(snapshot) = snapshot else {
            return Ok(());
        };
        for slot in snapshot.films.values() {
            let removed = slot
                .lock()
                .map_err(|_| AssetError::ProfileCachePoisoned { role: Role::Film })?
                .take();
            drop(removed);
        }
        for slot in snapshot.prints.values() {
            let removed = slot
                .lock()
                .map_err(|_| AssetError::ProfileCachePoisoned { role: Role::Print })?
                .take();
            drop(removed);
        }
        Ok(())
    }
}

fn load_spectra(path: &Path) -> Result<Arc<SpectraLut>, ReadError> {
    load_spectra_lut(path).map(SpectraLut::new).map(Arc::new)
}

// Only this final asset boundary reserves zero for absence.
fn nonzero_asset_hash(hash: u64) -> u64 {
    if hash == 0 { 1 } else { hash }
}

fn load_film(path: &Path) -> Result<Arc<FilmProfile>, AssetError> {
    let source = load_film_source(path).map_err(AssetError::ProfileDecode)?;
    FilmProfile::new(source)
        .map(Arc::new)
        .map_err(AssetError::ProfileCompletion)
}

fn load_print(path: &Path) -> Result<Arc<PrintProfile>, AssetError> {
    let source = load_print_source(path).map_err(AssetError::ProfileDecode)?;
    PrintProfile::new(source)
        .map(Arc::new)
        .map_err(AssetError::ProfileCompletion)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::Duration;

    use super::{
        AssetError, Assets, SpectraLut, load_film, load_print, load_spectra, nonzero_asset_hash,
    };
    use crate::data_io::{ReadError, ReadErrorKind, load_mallett_basis, load_spectra_lut};
    use crate::profile::{ProfileCompletionError, ProfileCompletionErrorKind, Role};

    const WAIT: Duration = Duration::from_secs(15);
    const FILM: &str = "kodak_portra_400";
    const PRINT: &str = "kodak_portra_endura";

    fn assets() -> Assets {
        Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"))
    }

    #[test]
    fn final_asset_zero_normalization_is_local() {
        assert_eq!(nonzero_asset_hash(0), 1);
        for hash in [1, 0x81ebefd4e4cc9926, 0x9262ffb765e3289e, u64::MAX] {
            assert_eq!(nonzero_asset_hash(hash), hash);
        }
    }

    #[test]
    fn spectra_owner_moves_the_decoded_allocation() {
        let path = assets()
            .resource_dir
            .join("luts/spectral_upsampling/irradiance_xy_tc.npy");
        let samples = load_spectra_lut(path).unwrap();
        let pointer = samples.as_ptr();
        let capacity = samples.capacity();
        let lut = Arc::new(SpectraLut::new(samples));
        assert_eq!(lut.samples().as_ptr(), pointer);
        assert_eq!(lut.samples.capacity(), capacity);
        let handle = Arc::clone(&lut);
        assert_eq!(handle.samples().as_ptr(), pointer);
        assert_eq!(handle.asset_hash(), 0x81ebefd4e4cc9926);
    }

    #[test]
    fn hanatos_cold_attempt_returns_winner_and_reclaims_loser() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.hanatos_with(|path| {
                    let candidate = load_spectra(path)?;
                    assert!(assets.hanatos.get().is_none());
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let loser = ready_rx.recv_timeout(WAIT).unwrap();
            assert!(assets.arctic.get().is_none());
            assert!(assets.mallett.get().is_none());
            assert!(assets.catalog.get().is_none());
            let winner = assets.hanatos().unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
            assert!(loser.upgrade().is_none());
        });
    }

    #[test]
    fn arctic_cold_attempt_allows_unrelated_acquisition_and_release() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.arctic_with(|path| {
                    let candidate = load_spectra(path)?;
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let loser = ready_rx.recv_timeout(WAIT).unwrap();
            assert!(assets.hanatos.get().is_none());
            assert!(assets.catalog.get().is_none());
            assert!(assets.mallett().is_ok());
            assert!(assets.film(FILM).is_ok());
            assets.release_cached_payloads().unwrap();
            assert!(assets.arctic.get().is_none());
            let winner = assets.arctic().unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
            assert!(Arc::ptr_eq(&winner, &assets.arctic().unwrap()));
            assert!(loser.upgrade().is_none());
        });
    }

    #[test]
    fn mallett_cold_attempt_returns_shared_winner() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.mallett_with(|path| {
                    let candidate = Arc::new(load_mallett_basis(path)?);
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let loser = ready_rx.recv_timeout(WAIT).unwrap();
            assert!(assets.hanatos.get().is_none());
            assert!(assets.arctic.get().is_none());
            assert!(assets.catalog.get().is_none());
            let winner = assets.mallett().unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
            assert!(loser.upgrade().is_none());
        });
    }

    #[test]
    fn reconstruction_capacity_failure_can_win_and_remains_sticky() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.hanatos_with(|path| {
                    let candidate = load_spectra(path)?;
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let loser = ready_rx.recv_timeout(WAIT).unwrap();
            let failed = assets.hanatos_with(|path| {
                Err(ReadError {
                    path: path.to_owned(),
                    kind: ReadErrorKind::Capacity,
                })
            });
            let Err(AssetError::Reconstruction(winner)) = failed else {
                panic!("expected retained capacity error");
            };
            assets.release_cached_payloads().unwrap();
            assert!(assets.arctic().is_ok());
            resume_tx.send(()).unwrap();
            let Err(AssetError::Reconstruction(returned)) = attempt.join().unwrap() else {
                panic!("expected first published error");
            };
            assert!(Arc::ptr_eq(&winner, &returned));
            assert!(loser.upgrade().is_none());
            let Err(AssetError::Reconstruction(cached)) = assets.hanatos_with(|_| {
                panic!("a retained outcome must not invoke the loader");
            }) else {
                panic!("expected cached error");
            };
            assert!(Arc::ptr_eq(&winner, &cached));
            assert_eq!(cached.kind, ReadErrorKind::Capacity);
        });
    }

    #[test]
    fn local_reconstruction_failure_returns_a_concurrent_success() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.arctic_with(|path| {
                    let failure = load_spectra(&path.join("missing"));
                    assert!(failure.is_err());
                    ready_tx.send(()).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    failure
                })
            });
            ready_rx.recv_timeout(WAIT).unwrap();
            let winner = assets.arctic().unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
            assert!(Arc::ptr_eq(&winner, &assets.arctic().unwrap()));
        });
    }

    #[test]
    fn film_cold_attempt_returns_published_winner_and_drops_loser() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.film_with(FILM, |path| {
                    let candidate = load_film(path)?;
                    let slot = &assets.snapshot().unwrap().films[FILM];
                    assert!(
                        slot.try_lock().is_ok(),
                        "preparation owns no publication guard"
                    );
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let loser = ready_rx.recv_timeout(WAIT).unwrap();
            let winner = assets.film(FILM).unwrap();
            resume_tx.send(()).unwrap();
            let returned = attempt.join().unwrap().unwrap();
            assert!(Arc::ptr_eq(&winner, &returned));
            assert!(Arc::ptr_eq(&winner, &assets.film(FILM).unwrap()));
            assert!(loser.upgrade().is_none());
        });
    }

    #[test]
    fn print_cold_attempt_returns_published_winner_and_drops_loser() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.print_with(PRINT, |path| {
                    let candidate = load_print(path)?;
                    let slot = &assets.snapshot().unwrap().prints[PRINT];
                    assert!(slot.try_lock().is_ok());
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let loser = ready_rx.recv_timeout(WAIT).unwrap();
            let winner = assets.print(PRINT).unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
            assert!(loser.upgrade().is_none());
        });
    }

    #[test]
    fn failed_film_decode_rechecks_successful_same_key_winner() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.film_with(FILM, |path| {
                    let failed = load_film(&path.join("missing.json"));
                    assert!(matches!(&failed, Err(AssetError::ProfileDecode(_))));
                    ready_tx.send(()).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    failed
                })
            });
            ready_rx.recv_timeout(WAIT).unwrap();
            let winner = assets.film(FILM).unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
        });
    }

    #[test]
    fn failed_print_completion_rechecks_successful_same_key_winner() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.print_with(PRINT, |_| {
                    // Bounded local capacity failure; the accepted completion module
                    // separately exercises its real fallible reservation path.
                    let failed = Err(AssetError::ProfileCompletion(ProfileCompletionError {
                        role: Role::Print,
                        stock: PRINT.to_owned(),
                        kind: ProfileCompletionErrorKind::Capacity,
                    }));
                    ready_tx.send(()).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    failed
                })
            });
            ready_rx.recv_timeout(WAIT).unwrap();
            let winner = assets.print(PRINT).unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
        });
    }

    #[test]
    fn release_during_cold_preparation_allows_later_publication() {
        let assets = assets();
        let retained = assets.print(PRINT).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.film_with(FILM, |path| {
                    let candidate = load_film(path)?;
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let prepared = ready_rx.recv_timeout(WAIT).unwrap();
            assets.release_cached_payloads().unwrap();
            assert_eq!(Arc::strong_count(&retained), 1);
            assert!(
                assets.snapshot().unwrap().films[FILM]
                    .lock()
                    .unwrap()
                    .is_none()
            );
            resume_tx.send(()).unwrap();
            let published = attempt.join().unwrap().unwrap();
            assert!(Arc::ptr_eq(&prepared.upgrade().unwrap(), &published));
            assert!(Arc::ptr_eq(&published, &assets.film(FILM).unwrap()));
        });
        let reacquired = assets.print(PRINT).unwrap();
        assert!(!Arc::ptr_eq(&retained, &reacquired));
        assert_eq!(retained.asset_token(), reacquired.asset_token());
    }

    #[test]
    fn preparation_can_release_caches_without_nested_publication_locks() {
        let assets = assets();
        let film = assets
            .film_with(FILM, |path| {
                assets.release_cached_payloads()?;
                load_film(path)
            })
            .unwrap();
        let print = assets
            .print_with(PRINT, |path| {
                assets.release_cached_payloads()?;
                load_print(path)
            })
            .unwrap();
        assert_eq!(film.info().stock(), FILM);
        assert_eq!(print.info().stock(), PRINT);
    }

    #[test]
    fn poisoned_profile_slots_fail_without_repairing_or_using_cached_payloads() {
        let assets = assets();
        let film = assets.film(FILM).unwrap();
        thread::scope(|scope| {
            assert!(
                scope
                    .spawn(|| {
                        let _guard = assets.snapshot().unwrap().films[FILM].lock().unwrap();
                        panic!("bounded film cache poison");
                    })
                    .join()
                    .is_err()
            );
        });
        assert!(matches!(
            assets.film(FILM),
            Err(AssetError::ProfileCachePoisoned { role: Role::Film })
        ));
        assert!(matches!(
            assets.release_cached_payloads(),
            Err(AssetError::ProfileCachePoisoned { role: Role::Film })
        ));
        assert!(assets.print(PRINT).is_ok());
        drop(assets);
        assert_eq!(film.info().stock(), FILM);

        let assets = self::assets();
        assets.print(PRINT).unwrap();
        thread::scope(|scope| {
            assert!(
                scope
                    .spawn(|| {
                        let _guard = assets.snapshot().unwrap().prints[PRINT].lock().unwrap();
                        panic!("bounded print cache poison");
                    })
                    .join()
                    .is_err()
            );
        });
        assert!(matches!(
            assets.print(PRINT),
            Err(AssetError::ProfileCachePoisoned { role: Role::Print })
        ));
        assert!(matches!(
            assets.release_cached_payloads(),
            Err(AssetError::ProfileCachePoisoned { role: Role::Print })
        ));
        assert!(assets.film(FILM).is_ok());
    }
}
