//! Process asset ownership. Published handles outlive cache release and this owner.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::data_io::{
    CsvPairs, CmfRows, ReadError, load_csv_pairs, load_cmf_csv, load_mallett_basis,
    load_spectra_lut,
};
use crate::data_io::calibration::{self, NeutralCalibration, load_neutral_calibration};
use crate::data_io::noise::{self, Stbn, Wang, load_stbn, load_wang};
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
    Cmf(Arc<ReadError>),
    CsvSource { source: CsvSource, error: ReadError },
    CsvCachePoisoned { source: CsvSource },
    Calibration(Arc<calibration::Error>),
    CalibrationCachePoisoned,
    Noise(Arc<noise::Error>),
    NoiseCachePoisoned,
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
            Self::Cmf(source) => write!(f, "{source}"),
            Self::CsvSource { source, error } => write!(f, "{source:?}: {error}"),
            Self::CsvCachePoisoned { source } => write!(f, "{source:?} CSV source cache poisoned"),
            Self::Calibration(source) => write!(f, "{source}"),
            Self::CalibrationCachePoisoned => write!(f, "neutral calibration cache poisoned"),
            Self::Noise(source) => write!(f, "{source}"),
            Self::NoiseCachePoisoned => write!(f, "noise cache poisoned"),
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
            Self::Cmf(source) => Some(source.as_ref()),
            Self::CsvSource { error, .. } => Some(error),
            Self::Calibration(source) => Some(source.as_ref()),
            Self::Noise(source) => Some(source.as_ref()),
            Self::MissingProfile { .. }
            | Self::ProfileCachePoisoned { .. }
            | Self::CsvCachePoisoned { .. }
            | Self::CalibrationCachePoisoned
            | Self::NoiseCachePoisoned => None,
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

    /// Decoded-sample resource fingerprint: canonical zero signs/NaNs, zero maps
    /// to one. No path, metadata, profile evaluator version or NaN-mask stream.
    pub fn asset_hash(&self) -> u64 {
        self.asset_hash
    }
}

/// Fixed decoded wavelength/value sources, before normalization or resampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsvSource {
    D65,
    D55,
    D50,
    T,
    K75p,
    Kg3,
    Canon24F28Is,
}

impl CsvSource {
    fn relative_path(self) -> &'static str {
        match self {
            Self::D65 => "illuminants/D65.csv",
            Self::D55 => "illuminants/D55.csv",
            Self::D50 => "illuminants/D50.csv",
            Self::T => "illuminants/T.csv",
            Self::K75p => "illuminants/K75P.csv",
            Self::Kg3 => "filters/heat_absorbing/schott/KG3.csv",
            Self::Canon24F28Is => "filters/lens_transmission/canon/canon_24_f28_is.csv",
        }
    }
}

/// Both noise families are complete before this immutable owner is published.
#[derive(Debug)]
pub struct NoiseBundle {
    stbn: Stbn,
    wang: Wang,
}

impl NoiseBundle {
    pub fn stbn(&self) -> &Stbn {
        &self.stbn
    }

    pub fn wang(&self) -> &Wang {
        &self.wang
    }
}

struct CatalogSnapshot {
    catalog: Arc<Catalog>,
    films: HashMap<String, Mutex<Option<Arc<FilmProfile>>>>,
    prints: HashMap<String, Mutex<Option<Arc<PrintProfile>>>>,
}

// The accepted reader's wavelength-major RGB representation, without a wrapper.
type MallettBasis = [[f32; 3]; 81];
type CalibrationSnapshot = Result<Arc<NeutralCalibration>, Arc<calibration::Error>>;
type NoiseSnapshot = Result<Arc<NoiseBundle>, Arc<noise::Error>>;

pub struct Assets {
    resource_dir: PathBuf,
    catalog: OnceLock<Result<CatalogSnapshot, Arc<CatalogError>>>,
    hanatos: OnceLock<Result<Arc<SpectraLut>, Arc<ReadError>>>,
    arctic: OnceLock<Result<Arc<SpectraLut>, Arc<ReadError>>>,
    mallett: OnceLock<Result<Arc<MallettBasis>, Arc<ReadError>>>,
    cmf: OnceLock<Result<Arc<CmfRows>, Arc<ReadError>>>,
    csv_sources: [Mutex<Option<Arc<CsvPairs>>>; 7],
    calibration: Mutex<Option<CalibrationSnapshot>>,
    noise: Mutex<Option<NoiseSnapshot>>,
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
            cmf: OnceLock::new(),
            csv_sources: std::array::from_fn(|_| Mutex::new(None)),
            calibration: Mutex::new(None),
            noise: Mutex::new(None),
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

    /// Decoded CMF source rows; the first complete outcome lasts for this owner.
    pub fn cmf(&self) -> Result<Arc<CmfRows>, AssetError> {
        self.cmf_with(|path| load_cmf_csv(path).map(Arc::new))
    }

    /// Cache successful source rows only. A later explicit request may retry a failure.
    pub fn csv_source(&self, source: CsvSource) -> Result<Arc<CsvPairs>, AssetError> {
        self.csv_source_with(source, |path| load_csv_pairs(path).map(Arc::new))
    }

    /// Retain the load snapshot until release. Selected branches are checked by lookup.
    pub fn neutral_calibration(&self) -> Result<Arc<NeutralCalibration>, AssetError> {
        self.calibration_with(|path| load_neutral_calibration(path).map(Arc::new))
    }

    /// Acquire only for an enabled consumer. Capacity failures are not cached.
    pub fn noise(&self) -> Result<Arc<NoiseBundle>, AssetError> {
        self.noise_with(|root| {
            load_noise_with(
                root,
                |path| load_stbn(path),
                |tiles, metadata| load_wang(tiles, metadata),
            )
        })
    }

    fn cmf_with(
        &self,
        load: impl FnOnce(&Path) -> Result<Arc<CmfRows>, ReadError>,
    ) -> Result<Arc<CmfRows>, AssetError> {
        if self.cmf.get().is_none() {
            let prepared = load(&self.resource_dir.join("cie1931_2deg.csv")).map_err(Arc::new);
            drop(self.cmf.set(prepared));
        }
        self.cmf
            .get()
            .expect("a complete CMF source outcome has been published")
            .as_ref()
            .map(Arc::clone)
            .map_err(|source| AssetError::Cmf(Arc::clone(source)))
    }

    fn csv_source_with(
        &self,
        source: CsvSource,
        load: impl FnOnce(&Path) -> Result<Arc<CsvPairs>, ReadError>,
    ) -> Result<Arc<CsvPairs>, AssetError> {
        let slot = &self.csv_sources[source as usize];
        let cached = slot
            .lock()
            .map_err(|_| AssetError::CsvCachePoisoned { source })?
            .as_ref()
            .map(Arc::clone);
        if let Some(rows) = cached {
            return Ok(rows);
        }
        let prepared = load(&self.resource_dir.join(source.relative_path()));
        let published = {
            let mut cached = slot
                .lock()
                .map_err(|_| AssetError::CsvCachePoisoned { source })?;
            if cached.is_none()
                && let Ok(rows) = &prepared
            {
                *cached = Some(Arc::clone(rows));
            }
            cached.as_ref().map(Arc::clone)
        };
        published
            .map(Ok)
            .unwrap_or_else(|| prepared.map_err(|error| AssetError::CsvSource { source, error }))
    }

    fn calibration_with(
        &self,
        load: impl FnOnce(&Path) -> Result<Arc<NeutralCalibration>, calibration::Error>,
    ) -> Result<Arc<NeutralCalibration>, AssetError> {
        let cached = self
            .calibration
            .lock()
            .map_err(|_| AssetError::CalibrationCachePoisoned)?
            .clone();
        if let Some(snapshot) = cached {
            return snapshot.map_err(AssetError::Calibration);
        }
        let prepared =
            load(&self.resource_dir.join("filters/neutral_print_filters.json")).map_err(Arc::new);
        let published = {
            let mut cached = self
                .calibration
                .lock()
                .map_err(|_| AssetError::CalibrationCachePoisoned)?;
            if cached.is_none() {
                *cached = Some(prepared.clone());
            }
            cached
                .as_ref()
                .expect("a complete calibration snapshot has been published")
                .clone()
        };
        // Only Arc holds are cloned. Losing values are destroyed after unlocking.
        published.map_err(AssetError::Calibration)
    }

    fn noise_with(
        &self,
        load: impl FnOnce(&Path) -> Result<Arc<NoiseBundle>, noise::Error>,
    ) -> Result<Arc<NoiseBundle>, AssetError> {
        let cached = self
            .noise
            .lock()
            .map_err(|_| AssetError::NoiseCachePoisoned)?
            .clone();
        if let Some(snapshot) = cached {
            return snapshot.map_err(AssetError::Noise);
        }
        // The complete local attempt has already reclaimed any partial buffers
        // before this publication/recheck. Capacity is the sole nonsticky error.
        let prepared = load(&self.resource_dir).map_err(Arc::new);
        let capacity = prepared
            .as_ref()
            .is_err_and(|error| error.kind() == noise::ErrorKind::Capacity);
        let published = {
            let mut cached = self
                .noise
                .lock()
                .map_err(|_| AssetError::NoiseCachePoisoned)?;
            if cached.is_none() && !capacity {
                *cached = Some(prepared.clone());
            }
            cached.clone()
        };
        published.unwrap_or(prepared).map_err(AssetError::Noise)
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

    /// Detach profile, CSV, calibration and noise holds. Catalog, reconstruction and CMF
    /// outcomes remain fixed. Retained
    /// handles stay valid. An overlapping cold load may publish after release;
    /// callers requiring a drained cache must first exclude new loads.
    pub fn release_cached_payloads(&self) -> Result<(), AssetError> {
        // Release on a new owner must not trigger catalog I/O.
        if let Some(Ok(snapshot)) = self.catalog.get() {
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
        }
        for source in [
            CsvSource::D65,
            CsvSource::D55,
            CsvSource::D50,
            CsvSource::T,
            CsvSource::K75p,
            CsvSource::Kg3,
            CsvSource::Canon24F28Is,
        ] {
            let removed = self.csv_sources[source as usize]
                .lock()
                .map_err(|_| AssetError::CsvCachePoisoned { source })?
                .take();
            drop(removed);
        }
        let calibration = self
            .calibration
            .lock()
            .map_err(|_| AssetError::CalibrationCachePoisoned)?
            .take();
        drop(calibration);
        let noise = self
            .noise
            .lock()
            .map_err(|_| AssetError::NoiseCachePoisoned)?
            .take();
        drop(noise);
        Ok(())
    }
}

fn load_noise_with(
    root: &Path,
    stbn: impl FnOnce(&Path) -> Result<Stbn, noise::Error>,
    wang: impl FnOnce(&Path, &Path) -> Result<Wang, noise::Error>,
) -> Result<Arc<NoiseBundle>, noise::Error> {
    let stbn = stbn(&root.join("Noise/stbn_scalar_512x512x256_u8.bin"))?;
    let wang = wang(
        &root.join("Noise/Wang/wang_tiles_256x256x16_u8.bin"),
        &root.join("Noise/Wang/tiles.json"),
    )?;
    Ok(Arc::new(NoiseBundle { stbn, wang }))
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
        AssetError, Assets, CsvSource, SpectraLut, load_film, load_noise_with, load_print,
        load_spectra, nonzero_asset_hash,
    };
    use crate::data_io::{
        ReadError, ReadErrorKind, load_csv_pairs, load_cmf_csv, load_mallett_basis,
        load_spectra_lut,
    };
    use crate::data_io::calibration::load_neutral_calibration;
    use crate::data_io::noise::{self, load_stbn, load_wang};
    use crate::data_io::noise::test_support::{capacity_error, track_stbn_drop};
    use crate::profile::{ProfileCompletionError, ProfileCompletionErrorKind, Role};

    const WAIT: Duration = Duration::from_secs(15);
    const FILM: &str = "kodak_portra_400";
    const PRINT: &str = "kodak_portra_endura";

    fn assets() -> Assets {
        Assets::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"))
    }

    #[test]
    fn disabled_grain_and_profile_access_leave_source_families_unloaded() {
        let assets = assets();
        let grain_active = false;
        let noise = grain_active.then(|| assets.noise());
        assert!(noise.is_none());
        assert!(assets.catalog().is_ok());
        assert!(assets.film(FILM).is_ok());
        assert!(assets.print(PRINT).is_ok());
        assert!(assets.cmf.get().is_none());
        assert!(assets.hanatos.get().is_none());
        assert!(assets.arctic.get().is_none());
        assert!(assets.mallett.get().is_none());
        assert!(assets.noise.lock().unwrap().is_none());
        assert!(assets.calibration.lock().unwrap().is_none());
        assert!(
            assets
                .csv_sources
                .iter()
                .all(|slot| slot.lock().unwrap().is_none())
        );
    }

    #[test]
    fn cmf_cold_success_returns_winner_and_reclaims_loser() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.cmf_with(|path| {
                    let candidate = Arc::new(load_cmf_csv(path)?);
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let loser = ready_rx.recv_timeout(WAIT).unwrap();
            let winner = assets.cmf().unwrap();
            assets.release_cached_payloads().unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
            assert!(loser.upgrade().is_none());
        });
    }

    #[test]
    fn cmf_first_outcome_wins_including_capacity() {
        for error_first in [false, true] {
            let assets = assets();
            let (ready_tx, ready_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            thread::scope(|scope| {
                let assets = &assets;
                let attempt = scope.spawn(move || {
                    assets.cmf_with(|path| {
                        ready_tx.send(()).unwrap();
                        resume_rx.recv_timeout(WAIT).unwrap();
                        if error_first {
                            load_cmf_csv(path).map(Arc::new)
                        } else {
                            Err(ReadError {
                                path: path.to_owned(),
                                kind: ReadErrorKind::Capacity,
                            })
                        }
                    })
                });
                ready_rx.recv_timeout(WAIT).unwrap();
                if error_first {
                    let Err(AssetError::Cmf(winner)) = assets.cmf_with(|path| {
                        Err(ReadError {
                            path: path.to_owned(),
                            kind: ReadErrorKind::Capacity,
                        })
                    }) else {
                        panic!("capacity outcome expected")
                    };
                    assets.release_cached_payloads().unwrap();
                    resume_tx.send(()).unwrap();
                    let Err(AssetError::Cmf(returned)) = attempt.join().unwrap() else {
                        panic!("first CMF error must remain")
                    };
                    assert!(Arc::ptr_eq(&winner, &returned));
                    assert!(matches!(assets.cmf(), Err(AssetError::Cmf(_))));
                } else {
                    let winner = assets.cmf().unwrap();
                    resume_tx.send(()).unwrap();
                    assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
                }
            });
        }
    }

    #[test]
    fn csv_cold_success_and_failure_return_same_source_winner() {
        for fail in [false, true] {
            let assets = assets();
            let (ready_tx, ready_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            thread::scope(|scope| {
                let assets = &assets;
                let attempt = scope.spawn(move || {
                    assets.csv_source_with(CsvSource::Kg3, |path| {
                        let candidate = if fail {
                            Err(ReadError {
                                path: path.to_owned(),
                                kind: ReadErrorKind::Capacity,
                            })
                        } else {
                            load_csv_pairs(path).map(Arc::new)
                        };
                        ready_tx
                            .send(candidate.as_ref().ok().map(Arc::downgrade))
                            .unwrap();
                        resume_rx.recv_timeout(WAIT).unwrap();
                        candidate
                    })
                });
                let loser = ready_rx.recv_timeout(WAIT).unwrap();
                assert!(assets.csv_source(CsvSource::D65).is_ok());
                let winner = assets.csv_source(CsvSource::Kg3).unwrap();
                resume_tx.send(()).unwrap();
                assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
                if let Some(loser) = loser {
                    assert!(loser.upgrade().is_none());
                }
            });
        }
    }

    #[test]
    fn csv_cold_load_can_publish_after_release() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.csv_source_with(CsvSource::D50, |path| {
                    let candidate = Arc::new(load_csv_pairs(path)?);
                    ready_tx.send(()).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            ready_rx.recv_timeout(WAIT).unwrap();
            assets.release_cached_payloads().unwrap();
            assert!(
                assets.csv_sources[CsvSource::D50 as usize]
                    .lock()
                    .unwrap()
                    .is_none()
            );
            resume_tx.send(()).unwrap();
            let published = attempt.join().unwrap().unwrap();
            assert!(Arc::ptr_eq(
                &published,
                &assets.csv_source(CsvSource::D50).unwrap()
            ));
        });
    }

    #[test]
    fn calibration_success_and_error_races_use_first_snapshot() {
        for error_first in [false, true] {
            let assets = assets();
            let (ready_tx, ready_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            thread::scope(|scope| {
                let assets = &assets;
                let attempt = scope.spawn(move || {
                    assets.calibration_with(|path| {
                        let candidate = if error_first {
                            load_neutral_calibration(path).map(Arc::new)
                        } else {
                            load_neutral_calibration(path.with_file_name("__missing_calibration"))
                                .map(Arc::new)
                        };
                        ready_tx
                            .send(candidate.as_ref().ok().map(Arc::downgrade))
                            .unwrap();
                        resume_rx.recv_timeout(WAIT).unwrap();
                        candidate
                    })
                });
                let loser = ready_rx.recv_timeout(WAIT).unwrap();
                if error_first {
                    let Err(AssetError::Calibration(winner)) = assets.calibration_with(|path| {
                        load_neutral_calibration(path.with_file_name("__missing_calibration"))
                            .map(Arc::new)
                    }) else {
                        panic!("missing snapshot expected")
                    };
                    resume_tx.send(()).unwrap();
                    let Err(AssetError::Calibration(returned)) = attempt.join().unwrap() else {
                        panic!("first error must win")
                    };
                    assert!(Arc::ptr_eq(&winner, &returned));
                    assert!(loser.unwrap().upgrade().is_none());
                } else {
                    let winner = assets.neutral_calibration().unwrap();
                    resume_tx.send(()).unwrap();
                    assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
                }
            });
        }
    }

    #[test]
    fn calibration_cold_load_can_publish_after_release() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.calibration_with(|path| {
                    let candidate = Arc::new(load_neutral_calibration(path)?);
                    ready_tx.send(()).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            ready_rx.recv_timeout(WAIT).unwrap();
            assert!(assets.csv_source(CsvSource::T).is_ok());
            assets.release_cached_payloads().unwrap();
            assert!(assets.calibration.lock().unwrap().is_none());
            resume_tx.send(()).unwrap();
            let published = attempt.join().unwrap().unwrap();
            assert!(Arc::ptr_eq(
                &published,
                &assets.neutral_calibration().unwrap()
            ));
        });
    }

    #[test]
    fn noise_bundle_moves_both_decoded_buffers() {
        let assets = assets();
        let mut stbn_pointer = 0;
        let mut wang_pointer = 0;
        let bundle = load_noise_with(
            &assets.resource_dir,
            |path| {
                let stbn = load_stbn(path)?;
                stbn_pointer = stbn.bytes().as_ptr() as usize;
                Ok(stbn)
            },
            |tiles, metadata| {
                let wang = load_wang(tiles, metadata)?;
                wang_pointer = wang.tiles().as_ptr() as usize;
                Ok(wang)
            },
        )
        .unwrap();
        assert_eq!(bundle.stbn().bytes().as_ptr() as usize, stbn_pointer);
        assert_eq!(bundle.wang().tiles().as_ptr() as usize, wang_pointer);
        let retained = Arc::clone(&bundle);
        assert_eq!(retained.stbn().bytes().as_ptr() as usize, stbn_pointer);
        assert_eq!(retained.wang().tiles().as_ptr() as usize, wang_pointer);
    }

    #[test]
    fn noise_capacity_failure_retries_without_release() {
        let assets = assets();
        let error = assets
            .noise_with(|root| {
                load_noise_with(
                    root,
                    |path| load_stbn(path),
                    |tiles, _| Err(capacity_error(tiles)),
                )
            })
            .unwrap_err();
        let AssetError::Noise(error) = error else {
            panic!("typed noise error expected")
        };
        assert_eq!(error.kind(), noise::ErrorKind::Capacity);
        assert!(
            error
                .path()
                .ends_with("Noise/Wang/wang_tiles_256x256x16_u8.bin")
        );
        assert!(assets.noise.lock().unwrap().is_none());
        let bundle = assets.noise().unwrap();
        assert!(Arc::ptr_eq(&bundle, &assets.noise().unwrap()));
    }

    #[test]
    fn wang_failures_reclaim_completed_stbn_before_publication() {
        for capacity in [false, true] {
            let assets = assets();
            let result = assets.noise_with(|root| {
                let mut probe = None;
                let result = load_noise_with(
                    root,
                    |path| {
                        let (stbn, weak) = track_stbn_drop(load_stbn(path)?);
                        assert!(weak.upgrade().is_some());
                        probe = Some(weak);
                        Ok(stbn)
                    },
                    |tiles, metadata| {
                        if capacity {
                            Err(capacity_error(tiles))
                        } else {
                            load_wang(tiles, metadata.with_file_name("__missing_wang"))
                        }
                    },
                );
                assert!(probe.unwrap().upgrade().is_none());
                assert!(assets.noise.try_lock().unwrap().is_none());
                result
            });
            assert!(matches!(result, Err(AssetError::Noise(_))));
            assert_eq!(assets.noise.lock().unwrap().is_none(), capacity);
        }
    }

    #[test]
    fn noise_capacity_attempt_returns_concurrent_success_or_error() {
        for ordinary_error in [false, true] {
            let assets = assets();
            let (ready_tx, ready_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            thread::scope(|scope| {
                let assets = &assets;
                let attempt = scope.spawn(move || {
                    assets.noise_with(|root| {
                        let result = load_noise_with(
                            root,
                            |path| load_stbn(path),
                            |tiles, _| Err(capacity_error(tiles)),
                        );
                        ready_tx.send(()).unwrap();
                        resume_rx.recv_timeout(WAIT).unwrap();
                        result
                    })
                });
                ready_rx.recv_timeout(WAIT).unwrap();
                if ordinary_error {
                    let Err(AssetError::Noise(winner)) = assets.noise_with(|root| {
                        Err(load_stbn(root.join("__missing_stbn")).unwrap_err())
                    }) else {
                        panic!("ordinary snapshot expected")
                    };
                    resume_tx.send(()).unwrap();
                    let Err(AssetError::Noise(returned)) = attempt.join().unwrap() else {
                        panic!("first ordinary error must win")
                    };
                    assert!(Arc::ptr_eq(&winner, &returned));
                    assert_eq!(returned.kind(), noise::ErrorKind::Missing);
                } else {
                    let winner = assets.noise().unwrap();
                    resume_tx.send(()).unwrap();
                    assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
                }
            });
        }
    }

    #[test]
    fn noise_cold_success_reclaims_loser_and_can_publish_after_release() {
        let assets = assets();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let assets = &assets;
            let attempt = scope.spawn(move || {
                assets.noise_with(|root| {
                    let candidate = load_noise_with(
                        root,
                        |path| load_stbn(path),
                        |tiles, metadata| load_wang(tiles, metadata),
                    )?;
                    ready_tx.send(Arc::downgrade(&candidate)).unwrap();
                    resume_rx.recv_timeout(WAIT).unwrap();
                    Ok(candidate)
                })
            });
            let loser = ready_rx.recv_timeout(WAIT).unwrap();
            assert!(assets.cmf().is_ok());
            assets.release_cached_payloads().unwrap();
            assert!(assets.noise.lock().unwrap().is_none());
            let winner = assets.noise().unwrap();
            resume_tx.send(()).unwrap();
            assert!(Arc::ptr_eq(&winner, &attempt.join().unwrap().unwrap()));
            assert!(loser.upgrade().is_none());
            assert!(Arc::ptr_eq(&winner, &assets.noise().unwrap()));
        });
        // With no competing publication, a cold load may install after release.
        assets.release_cached_payloads().unwrap();
        let published = assets
            .noise_with(|root| {
                assets.release_cached_payloads().unwrap();
                load_noise_with(
                    root,
                    |path| load_stbn(path),
                    |tiles, metadata| load_wang(tiles, metadata),
                )
            })
            .unwrap();
        assert!(Arc::ptr_eq(&published, &assets.noise().unwrap()));
    }

    #[test]
    fn new_family_poison_is_typed_and_not_repaired() {
        let assets = assets();
        let source = CsvSource::D65;
        assert!(
            std::panic::catch_unwind(|| {
                let _guard = assets.csv_sources[source as usize].lock().unwrap();
                panic!("poison CSV publication");
            })
            .is_err()
        );
        assert!(matches!(
            assets.csv_source(source),
            Err(AssetError::CsvCachePoisoned {
                source: CsvSource::D65
            })
        ));
        assert!(matches!(
            assets.release_cached_payloads(),
            Err(AssetError::CsvCachePoisoned { .. })
        ));
        assert!(assets.csv_source(CsvSource::D55).is_ok());
        assert!(
            std::panic::catch_unwind(|| {
                let _guard = assets.calibration.lock().unwrap();
                panic!("poison calibration publication");
            })
            .is_err()
        );
        assert!(matches!(
            assets.neutral_calibration(),
            Err(AssetError::CalibrationCachePoisoned)
        ));
        assert!(
            std::panic::catch_unwind(|| {
                let _guard = assets.noise.lock().unwrap();
                panic!("poison noise publication");
            })
            .is_err()
        );
        assert!(matches!(
            assets.noise(),
            Err(AssetError::NoiseCachePoisoned)
        ));
        assert!(assets.cmf().is_ok());
    }

    #[test]
    fn final_asset_zero_normalization_is_local() {
        assert_eq!(nonzero_asset_hash(0), 1);
        for hash in [1, 0x81ebefd4e4cc9926, 0x9262ffb765e3289e, u64::MAX] {
            assert_eq!(nonzero_asset_hash(hash), hash);
        }
    }

    #[test]
    fn spectral_source_capacity_receipt() {
        let assets = assets();
        for (name, lut) in [
            ("hanatos", assets.hanatos().unwrap()),
            ("arctic", assets.arctic().unwrap()),
        ] {
            assert_eq!(lut.samples.len(), 192 * 192 * 81);
            assert_eq!(lut.samples.capacity(), lut.samples.len());
            println!(
                "source {name}: length={} capacity={} requested_sample_bytes={} inline_payload_bytes={} Arc_handle_bytes={} live_strong_holds={}",
                lut.samples.len(),
                lut.samples.capacity(),
                lut.samples.capacity() * size_of::<f32>(),
                size_of::<SpectraLut>(),
                size_of::<Arc<SpectraLut>>(),
                Arc::strong_count(&lut)
            );
        }
        let mallett = assets.mallett().unwrap();
        println!(
            "source mallett: fixed_rows={} fixed_sample_bytes={} Arc_handle_bytes={} live_strong_holds={}",
            mallett.len(),
            size_of::<[[f32; 3]; 81]>(),
            size_of::<Arc<[[f32; 3]; 81]>>(),
            Arc::strong_count(&mallett)
        );
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
