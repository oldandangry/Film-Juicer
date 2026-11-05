//! Complete-state publication and admission, independent of host and CUDA ownership.

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::recipe::{Params, PrepareError, RenderState};

pub enum Pending {
    Uninitialized,
    /// The caller pairs captured input with its established semantic identity.
    /// Zero is valid; completed recipe hashes do not identify pending input.
    Valid {
        params: Arc<Params>,
        identity: u64,
    },
    InvalidSnapshotControls {
        diagnostic: Arc<str>,
    },
}

#[derive(Debug)]
pub enum Error {
    Uninitialized,
    InvalidSnapshotControls { diagnostic: Arc<str> },
    Preparation(PrepareError),
    Poisoned,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uninitialized => f.write_str("render snapshot is uninitialized"),
            Self::InvalidSnapshotControls { diagnostic } => f.write_str(diagnostic),
            Self::Preparation(error) => error.fmt(f),
            Self::Poisoned => f.write_str("render publication synchronization poisoned"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Preparation(error) => Some(error),
            _ => None,
        }
    }
}

struct Published {
    identity: u64,
    state: Arc<RenderState>,
}

struct State {
    pending: Pending,
    published: Option<Published>,
}

/// Lock order is rebuild gate, then state mutex. Pending updates take only the
/// state mutex; preparation and destruction of retired inputs/states run outside it.
pub struct Publication {
    state: Mutex<State>,
    rebuild: Mutex<()>,
}

impl Default for Publication {
    fn default() -> Self {
        Self::new()
    }
}

impl Publication {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                pending: Pending::Uninitialized,
                published: None,
            }),
            rebuild: Mutex::new(()),
        }
    }

    pub fn set_pending(&self, pending: Pending) -> Result<(), Error> {
        let retired = {
            let mut state = self.state.lock().map_err(|_| Error::Poisoned)?;
            std::mem::replace(&mut state.pending, pending)
        };
        drop(retired);
        Ok(())
    }

    /// Synchronously prepare one complete state outside the state mutex. The
    /// matching recheck/publish (or matching reuse) is the admission point; later
    /// pending updates do not revoke an already admitted immutable hold.
    pub fn admit(
        &self,
        mut prepare: impl FnMut(&Params) -> Result<RenderState, PrepareError>,
    ) -> Result<Arc<RenderState>, Error> {
        #[cfg(test)]
        tests::observe_rebuild_contention(&self.rebuild);
        let _rebuild = self.rebuild.lock().map_err(|_| Error::Poisoned)?;
        loop {
            let (params, identity) = {
                let state = self.state.lock().map_err(|_| Error::Poisoned)?;
                match &state.pending {
                    Pending::Uninitialized => return Err(Error::Uninitialized),
                    Pending::InvalidSnapshotControls { diagnostic } => {
                        return Err(Error::InvalidSnapshotControls {
                            diagnostic: Arc::clone(diagnostic),
                        });
                    }
                    Pending::Valid { params, identity } => {
                        if let Some(published) = &state.published
                            && published.identity == *identity
                        {
                            return Ok(Arc::clone(&published.state));
                        }
                        (Arc::clone(params), *identity)
                    }
                }
            };

            let outcome = prepare(&params).map(Arc::new);
            let mut state = self.state.lock().map_err(|_| Error::Poisoned)?;
            if !matches!(&state.pending, Pending::Valid { identity: current, .. } if *current == identity)
            {
                drop(state);
                drop(outcome);
                continue;
            }
            match outcome {
                Ok(candidate) => {
                    let retired = state.published.replace(Published {
                        identity,
                        state: Arc::clone(&candidate),
                    });
                    drop(state);
                    drop(retired);
                    return Ok(candidate);
                }
                Err(error) => {
                    drop(state);
                    return Err(Error::Preparation(error));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::film;
    use crate::recipe::{self, PreparedRoute};
    use crate::recipe::test_support::{assets, output_boundary_weak, params};
    use crate::route::ScanRoute;
    use crate::test_cases::RowCases;

    const DIAGNOSTIC: &str = "InvalidSnapshotControls field=camera_film_format_long_edge_mm";
    const RENDEZVOUS: Duration = Duration::from_secs(10);

    thread_local! {
        static REBUILD_WAITER: Cell<Option<mpsc::SyncSender<()>>> = const { Cell::new(None) };
    }

    // One-shot observation inside admission: the second caller must encounter
    // the real gate while the first builder is still paused, before its release.
    pub(super) fn observe_rebuild_contention(rebuild: &Mutex<()>) {
        if let Some(waiting) = REBUILD_WAITER.take() {
            assert!(matches!(
                rebuild.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            waiting.send(()).expect("contention observer disconnected");
        }
    }

    fn valid(identity: u64, params: Params) -> Pending {
        Pending::Valid {
            params: Arc::new(params),
            identity,
        }
    }

    fn direct() -> Params {
        params(ScanRoute::NegativeDirect, 1)
    }

    fn invalid() -> Pending {
        Pending::InvalidSnapshotControls {
            diagnostic: DIAGNOSTIC.into(),
        }
    }

    fn assert_invalid(result: Result<Arc<RenderState>, Error>) {
        assert!(
            matches!(result, Err(Error::InvalidSnapshotControls { diagnostic }) if &*diagnostic == DIAGNOSTIC)
        );
    }

    // Small exact reads observe real retained arrays without copying prepared tables.
    fn backing_words(state: &RenderState) -> [u32; 4] {
        [
            state.development().normalized_curves_cmy()[0][0].to_bits(),
            state.exposure_tables().weighted_xyz()[1][40].to_bits(),
            state.scanner().tables().weighted_xyz()[2][50].to_bits(),
            state.scanner().gamut().boundary().unwrap().cmax()[100].to_bits(),
        ]
    }

    #[test]
    fn uninitialized_does_not_build() {
        let publication = Publication::new();
        assert!(matches!(
            publication.admit(|_| panic!("uninitialized input must not build")),
            Err(Error::Uninitialized)
        ));
        assert!(publication.state.lock().unwrap().published.is_none());
    }

    #[test]
    fn matching_identity_reuses_state() {
        let assets = assets();
        let publication = Publication::new();
        publication.set_pending(valid(11, direct())).unwrap();
        let mut builds = 0;
        let first = publication
            .admit(|input| {
                builds += 1;
                assert_eq!(builds, 1, "equivalent pending input must not rebuild");
                // Equivalent input after an intervening update still admits this build.
                publication
                    .set_pending(valid(12, params(ScanRoute::NegativePrint, 1)))
                    .unwrap();
                publication.set_pending(valid(11, direct())).unwrap();
                recipe::prepare(&assets, input)
            })
            .unwrap();
        assert_eq!(builds, 1);
        let curves = first.development().normalized_curves_cmy().as_ptr();
        let tables = first.scanner().tables().weighted_xyz().as_ptr();
        let boundary = first.scanner().gamut().boundary().unwrap().cmax().as_ptr();
        let words = backing_words(&first);
        publication.set_pending(valid(11, direct())).unwrap();
        let same = publication.admit(|_| panic!("same input rebuilt")).unwrap();
        publication
            .set_pending(valid(12, params(ScanRoute::PositivePrint, 1)))
            .unwrap();
        publication.set_pending(valid(11, direct())).unwrap();
        let back = publication
            .admit(|_| panic!("equivalent input rebuilt"))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        assert!(Arc::ptr_eq(&first, &back));
        assert_eq!(back.development().normalized_curves_cmy().as_ptr(), curves);
        assert_eq!(back.scanner().tables().weighted_xyz().as_ptr(), tables);
        assert_eq!(
            back.scanner().gamut().boundary().unwrap().cmax().as_ptr(),
            boundary
        );
        assert_eq!(backing_words(&back), words);

        // Print-only captured controls can differ while direct completed recipes agree.
        let mut changed = direct();
        changed.print_exposure = 2.0;
        publication.set_pending(valid(13, changed)).unwrap();
        let changed = publication
            .admit(|input| recipe::prepare(&assets, input))
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &changed));
        assert_eq!(first.hash(), changed.hash());
    }

    #[test]
    fn zero_identity_is_valid() {
        let assets = assets();
        let publication = Publication::new();
        publication.set_pending(valid(0, direct())).unwrap();
        let mut builds = 0;
        let first = publication
            .admit(|input| {
                builds += 1;
                recipe::prepare(&assets, input)
            })
            .unwrap();
        let second = publication
            .admit(|_| panic!("zero identity rebuilt"))
            .unwrap();
        assert_eq!(builds, 1);
        assert!(Arc::ptr_eq(&first, &second));
        publication.set_pending(invalid()).unwrap();
        assert_invalid(publication.admit(|_| panic!("invalid input built")));
        publication.set_pending(Pending::Uninitialized).unwrap();
        assert!(matches!(
            publication.admit(|_| panic!("uninitialized input built")),
            Err(Error::Uninitialized)
        ));
    }

    #[test]
    fn route_changes_publish_complete_states() {
        let assets = assets();
        let publication = Publication::new();
        let routes = [
            ("negative-direct", ScanRoute::NegativeDirect),
            ("negative-print", ScanRoute::NegativePrint),
            ("positive-direct", ScanRoute::PositiveDirect),
            ("positive-print", ScanRoute::PositivePrint),
        ];
        let mut cases = RowCases::new("publication-routes", routes.map(|(id, _)| id));
        let mut holds = Vec::new();
        for (index, (id, route)) in routes.into_iter().enumerate() {
            let case = cases.case(id);
            publication
                .set_pending(valid(index as u64, params(route, 1)))
                .unwrap();
            let state = publication
                .admit(|input| recipe::prepare(&assets, input))
                .unwrap();
            assert_eq!(state.development().route(), route);
            assert_ne!(state.hash(), 0);
            assert_ne!(state.reference().hash(), 0);
            assert_ne!(state.scanner().lut().hash(), 0);
            assert!(!state.development().normalized_curves_cmy().is_empty());
            match (route, state.route()) {
                (
                    ScanRoute::NegativeDirect | ScanRoute::PositiveDirect,
                    PreparedRoute::Direct(_),
                ) => {}
                (
                    ScanRoute::NegativePrint | ScanRoute::PositivePrint,
                    PreparedRoute::Print(print),
                ) => {
                    assert_ne!(print.balance().hash(), 0);
                    assert_eq!(
                        print.curves().totals().len(),
                        print.profile().tables().interpolation_log_exposure().len()
                    );
                }
                _ => panic!("wrong completed route"),
            }
            let words = backing_words(&state);
            holds.push((state, route, words));
            for (held, route, words) in &holds {
                assert_eq!(held.development().route(), *route);
                assert_eq!(backing_words(held), *words);
            }
            case.passed();
        }
        cases.finish();
    }

    #[test]
    fn invalid_controls_block_admission() {
        let assets = assets();
        let publication = Publication::new();
        publication.set_pending(valid(1, direct())).unwrap();
        let held = publication
            .admit(|input| recipe::prepare(&assets, input))
            .unwrap();
        let words = backing_words(&held);
        publication.set_pending(invalid()).unwrap();
        assert_invalid(publication.admit(|_| panic!("invalid input built")));
        assert_eq!(backing_words(&held), words);
        publication.set_pending(valid(1, direct())).unwrap();
        assert!(Arc::ptr_eq(
            &held,
            &publication
                .admit(|_| panic!("recovery rebuilt equivalent input"))
                .unwrap()
        ));
        publication
            .set_pending(valid(2, params(ScanRoute::NegativePrint, 1)))
            .unwrap();
        let recovered = publication
            .admit(|input| recipe::prepare(&assets, input))
            .unwrap();
        assert_eq!(recovered.development().route(), ScanRoute::NegativePrint);
        assert!(!Arc::ptr_eq(&held, &recovered));
    }

    #[test]
    fn failed_preparation_blocks_old_state() {
        let assets = assets();
        let publication = Publication::new();
        publication.set_pending(valid(1, direct())).unwrap();
        let held = publication
            .admit(|input| recipe::prepare(&assets, input))
            .unwrap();
        let mut bad = direct();
        bad.film_gamma_factor = 0.0;
        publication.set_pending(valid(2, bad)).unwrap();
        assert!(matches!(
            publication.admit(|input| recipe::prepare(&assets, input)),
            Err(Error::Preparation(PrepareError::Development(
                film::Error::InvalidGamma
            )))
        ));
        assert!(Arc::ptr_eq(
            &held,
            &publication
                .state
                .lock()
                .unwrap()
                .published
                .as_ref()
                .unwrap()
                .state
        ));
        publication.set_pending(valid(3, direct())).unwrap();
        let recovered = publication
            .admit(|input| recipe::prepare(&assets, input))
            .unwrap();
        assert!(!Arc::ptr_eq(&held, &recovered));
        assert_eq!(backing_words(&held), backing_words(&recovered));
    }

    #[test]
    fn obsolete_success_is_not_published() {
        let assets = assets();
        let publication = Publication::new();
        let input = Arc::new(direct());
        let input_weak = Arc::downgrade(&input);
        publication
            .set_pending(Pending::Valid {
                params: input,
                identity: 1,
            })
            .unwrap();
        let mut discarded = None;
        let mut builds = 0;
        let admitted = publication
            .admit(|input| {
                builds += 1;
                assert!(
                    builds <= 2,
                    "obsolete success must retry only the current input"
                );
                let state = recipe::prepare(&assets, input)?;
                if builds == 1 {
                    discarded = Some(output_boundary_weak(&state));
                    assets.release_cached_payloads().unwrap();
                    let mut next = params(ScanRoute::PositiveDirect, 1);
                    next.output_gamut_compression_enabled = 0;
                    publication.set_pending(valid(2, next)).unwrap();
                    assert_eq!(input.scan_route, ScanRoute::NegativeDirect);
                    assert!(input_weak.upgrade().is_some());
                } else {
                    assert!(input_weak.upgrade().is_none());
                    assert!(discarded.as_ref().unwrap().upgrade().is_none());
                }
                Ok(state)
            })
            .unwrap();
        assert_eq!(builds, 2);
        assert_eq!(admitted.development().route(), ScanRoute::PositiveDirect);
        assert!(discarded.unwrap().upgrade().is_none());
        assert_eq!(Arc::strong_count(&admitted), 2);
        assert!(Arc::ptr_eq(
            &admitted,
            &publication
                .admit(|_| panic!("current state rebuilt"))
                .unwrap()
        ));
    }

    #[test]
    fn obsolete_failure_rechecks_pending() {
        let mut cases = RowCases::new(
            "publication-obsolete-failure",
            ["valid", "invalid", "uninitialized"],
        );
        for next in ["valid", "invalid", "uninitialized"] {
            let case = cases.case(next);
            let assets = assets();
            let publication = Publication::new();
            publication.set_pending(valid(1, direct())).unwrap();
            let held = publication
                .admit(|input| recipe::prepare(&assets, input))
                .unwrap();
            let mut bad = direct();
            bad.film_gamma_factor = 0.0;
            publication.set_pending(valid(2, bad)).unwrap();
            let mut builds = 0;
            let result = publication.admit(|input| {
                builds += 1;
                assert!(
                    builds <= 2,
                    "obsolete failure must retry only the current input"
                );
                let result = recipe::prepare(&assets, input);
                if builds == 1 {
                    assert!(matches!(
                        &result,
                        Err(PrepareError::Development(film::Error::InvalidGamma))
                    ));
                    publication
                        .set_pending(match next {
                            "valid" => valid(3, direct()),
                            "invalid" => invalid(),
                            "uninitialized" => Pending::Uninitialized,
                            _ => unreachable!(),
                        })
                        .unwrap();
                }
                result
            });
            match next {
                "valid" => {
                    assert_eq!(builds, 2);
                    let current = result.unwrap();
                    assert!(!Arc::ptr_eq(&held, &current));
                    assert_eq!(backing_words(&held), backing_words(&current));
                }
                "invalid" => {
                    assert_eq!(builds, 1);
                    assert_invalid(result);
                }
                "uninitialized" => {
                    assert_eq!(builds, 1);
                    assert!(matches!(result, Err(Error::Uninitialized)));
                }
                _ => unreachable!(),
            }
            case.passed();
        }
        cases.finish();
    }

    #[test]
    fn invalidated_build_is_not_published() {
        let assets = assets();
        let publication = Publication::new();
        publication.set_pending(valid(1, direct())).unwrap();
        let held = publication
            .admit(|input| recipe::prepare(&assets, input))
            .unwrap();
        publication
            .set_pending(valid(2, params(ScanRoute::NegativePrint, 1)))
            .unwrap();
        let mut builds = 0;
        assert_invalid(publication.admit(|input| {
            builds += 1;
            assert_eq!(builds, 1, "invalid input must stop preparation retries");
            let state = recipe::prepare(&assets, input)?;
            publication.set_pending(invalid()).unwrap();
            Ok(state)
        }));
        assert_eq!(builds, 1);
        let state = publication.state.lock().unwrap();
        let saved = state.published.as_ref().unwrap();
        assert_eq!(saved.identity, 1);
        assert!(Arc::ptr_eq(&saved.state, &held));
    }

    #[test]
    fn admitted_holds_survive_replacement_and_cache_release() {
        let assets = assets();
        let publication = Publication::new();
        publication.set_pending(valid(1, direct())).unwrap();
        let held = publication
            .admit(|input| recipe::prepare(&assets, input))
            .unwrap();
        let weak = Arc::downgrade(&held);
        let boundary = output_boundary_weak(&held);
        let words = backing_words(&held);
        let address = held.scanner().gamut().boundary().unwrap().cmax().as_ptr();
        publication.set_pending(invalid()).unwrap();
        assert_invalid(publication.admit(|_| panic!("invalid input built")));
        publication
            .set_pending(valid(2, params(ScanRoute::PositivePrint, 1)))
            .unwrap();
        let replacement = publication
            .admit(|input| recipe::prepare(&assets, input))
            .unwrap();
        let replacement_weak = Arc::downgrade(&replacement);
        assets.release_cached_payloads().unwrap();
        drop(assets);
        drop(publication);
        drop(replacement);
        assert!(replacement_weak.upgrade().is_none());
        assert!(weak.upgrade().is_some());
        assert!(boundary.upgrade().is_some());
        assert_eq!(backing_words(&held), words);
        assert_eq!(
            held.scanner().gamut().boundary().unwrap().cmax().as_ptr(),
            address
        );
        assert_eq!(Arc::strong_count(&held), 1);
        drop(held);
        assert!(weak.upgrade().is_none());
        assert!(boundary.upgrade().is_none());
    }

    #[test]
    fn concurrent_admission_builds_once() {
        let assets = assets();
        let publication = Publication::new();
        publication.set_pending(valid(1, direct())).unwrap();
        let builds = AtomicUsize::new(0);
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let (waiting_tx, waiting_rx) = mpsc::sync_channel(1);
        let (first, second) = thread::scope(|scope| {
            let publication = &publication;
            let assets = &assets;
            let builds = &builds;
            let first = scope.spawn(move || {
                publication.admit(|input| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    started_tx.send(()).unwrap();
                    release_rx
                        .recv_timeout(RENDEZVOUS)
                        .expect("first builder was not released");
                    recipe::prepare(assets, input)
                })
            });
            started_rx
                .recv_timeout(RENDEZVOUS)
                .expect("first builder did not start");
            let second = scope.spawn(move || {
                REBUILD_WAITER.set(Some(waiting_tx));
                publication.admit(|input| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    recipe::prepare(assets, input)
                })
            });
            waiting_rx
                .recv_timeout(RENDEZVOUS)
                .expect("second admission did not encounter the held rebuild gate");
            publication
                .set_pending(valid(2, params(ScanRoute::NegativePrint, 1)))
                .unwrap();
            publication.set_pending(valid(1, direct())).unwrap();
            release_tx.send(()).unwrap();
            let first = first.join();
            let second = second.join();
            (first.unwrap().unwrap(), second.unwrap().unwrap())
        });
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(backing_words(&first), backing_words(&second));
    }

    #[test]
    fn poisoned_state_blocks_admission() {
        let mut cases = RowCases::new("publication-poison", ["state", "rebuild"]);
        for target in ["state", "rebuild"] {
            let case = cases.case(target);
            let assets = assets();
            let publication = Publication::new();
            publication.set_pending(valid(1, direct())).unwrap();
            let held = publication
                .admit(|input| recipe::prepare(&assets, input))
                .unwrap();
            let words = backing_words(&held);
            let panic = catch_unwind(AssertUnwindSafe(|| match target {
                "state" => {
                    let _state = publication.state.lock().unwrap();
                    panic!("intentional state poison");
                }
                "rebuild" => {
                    publication.set_pending(valid(2, direct())).unwrap();
                    let _ = publication.admit(|_| {
                        publication.set_pending(valid(1, direct())).unwrap();
                        panic!("intentional rebuild poison");
                    });
                }
                _ => unreachable!(),
            }));
            assert!(panic.is_err());
            assert!(matches!(
                publication.admit(|_| panic!("poisoned admission built")),
                Err(Error::Poisoned)
            ));
            if target == "state" {
                assert!(matches!(
                    publication.set_pending(valid(3, direct())),
                    Err(Error::Poisoned)
                ));
            } else {
                assert!(publication.rebuild.is_poisoned());
                let state = publication.state.lock().unwrap();
                let saved = state.published.as_ref().unwrap();
                assert_eq!(saved.identity, 1);
                assert!(Arc::ptr_eq(&saved.state, &held));
            }
            assert_eq!(backing_words(&held), words);
            case.passed();
        }
        cases.finish();
    }
}
