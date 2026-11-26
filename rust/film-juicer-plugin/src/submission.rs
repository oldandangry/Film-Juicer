//! Per-instance identity and the current submission latch.
#![forbid(unsafe_code)]

use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Identity {
    pub(crate) session_seed: u64,
    pub(crate) instance_token: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    pub(crate) frame_token: u64,
    pub(crate) device_id: i32,
    pub(crate) context: usize,
    pub(crate) epoch: u64,
    pub(crate) upload_core: u64,
    pub(crate) dir: u64,
    pub(crate) scanner: u64,
    pub(crate) auto_exposure: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Reservation {
    pub(crate) id: u64,
    pub(crate) key: Key,
    revision: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Selection {
    pub(crate) reservation: Arc<Reservation>,
    pub(crate) reused: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Error {
    MissingIdentity,
    DuplicateIdentity,
    Poisoned,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MissingIdentity => "submission identity is uninitialized",
            Self::DuplicateIdentity => "submission identity is already initialized",
            Self::Poisoned => "submission latch is poisoned",
        })
    }
}

struct Latch {
    key: Key,
    id: u64,
}
struct State {
    next_id: u64,
    latch: Option<Latch>,
    revision: u64,
    pending: Vec<Weak<Reservation>>,
}
pub(crate) struct Submission {
    identity: OnceLock<Identity>,
    state: Mutex<State>,
}
impl Submission {
    pub(crate) fn new() -> Self {
        Self {
            identity: OnceLock::new(),
            state: Mutex::new(State {
                next_id: 1,
                latch: None,
                revision: 0,
                pending: Vec::new(),
            }),
        }
    }
    pub(crate) fn bind_identity(
        &self,
        session_seed: u64,
        instance_token: u64,
    ) -> Result<(), Error> {
        self.identity
            .set(Identity {
                session_seed,
                instance_token,
            })
            .map_err(|_| Error::DuplicateIdentity)
    }
    pub(crate) fn identity(&self) -> Option<Identity> {
        self.identity.get().copied()
    }
    pub(crate) fn select(&self, mut key: Key) -> Result<Selection, Error> {
        self.identity().ok_or(Error::MissingIdentity)?;
        key.upload_core = key.upload_core.max(1);
        key.dir = key.dir.max(1);
        key.scanner = key.scanner.max(1);
        key.auto_exposure = key.auto_exposure.max(1);
        let mut state = self.state.lock().map_err(|_| Error::Poisoned)?;
        let revision = state.revision;
        if let Some(latch) = &state.latch
            && latch.key == key
        {
            return Ok(Selection {
                reservation: Arc::new(Reservation {
                    key,
                    id: latch.id,
                    revision,
                }),
                reused: true,
            });
        }
        state.pending.retain(|pending| pending.strong_count() != 0);
        if let Some(reservation) = state
            .pending
            .iter()
            .filter_map(Weak::upgrade)
            .find(|pending| pending.key == key && pending.revision == revision)
        {
            return Ok(Selection {
                reservation,
                reused: true,
            });
        }
        let id = state.next_id;
        state.next_id = id.wrapping_add(1).max(1);
        let reservation = Arc::new(Reservation { key, id, revision });
        state.pending.push(Arc::downgrade(&reservation));
        Ok(Selection {
            reservation,
            reused: false,
        })
    }

    pub(crate) fn publish(&self, selected: &Reservation) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| Error::Poisoned)?;
        if state.revision == selected.revision
            && !state
                .latch
                .as_ref()
                .is_some_and(|latch| latch.id == selected.id && latch.key == selected.key)
        {
            state.latch = Some(Latch {
                key: selected.key,
                id: selected.id,
            });
            state.revision = state.revision.wrapping_add(1);
        }
        Ok(())
    }

    pub(crate) fn invalidate(
        &self,
        instance: u64,
        device_id: i32,
        context: usize,
        epoch: u64,
        id: u64,
    ) -> Result<bool, Error> {
        let mut state = self.state.lock().map_err(|_| Error::Poisoned)?;
        let matches = self
            .identity()
            .is_some_and(|identity| identity.instance_token == instance)
            && state.latch.as_ref().is_some_and(|latch| {
                latch.id == id
                    && latch.key.device_id == device_id
                    && latch.key.context == context
                    && latch.key.epoch == epoch
            });
        if matches {
            state.latch = None;
            state.revision = state.revision.wrapping_add(1);
        }
        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    fn key() -> Key {
        Key {
            frame_token: 0,
            device_id: 2,
            context: 3,
            epoch: 4,
            upload_core: 1,
            dir: 1,
            scanner: 1,
            auto_exposure: 1,
        }
    }

    #[test]
    fn only_admitted_current_reservation_replaces_latch() {
        let owner = Submission::new();
        assert_eq!(owner.select(key()), Err(Error::MissingIdentity));
        owner.bind_identity(0, 5).unwrap();
        assert_eq!(owner.bind_identity(2, 3), Err(Error::DuplicateIdentity));
        let a = owner.select(key()).unwrap();
        let b_key = Key { epoch: 5, ..key() };
        let b = owner.select(b_key).unwrap();
        owner.publish(&b.reservation).unwrap();
        owner.publish(&a.reservation).unwrap();
        assert!(!owner.invalidate(5, 2, 3, 4, a.reservation.id).unwrap());
        let late_a = owner.select(key()).unwrap();
        assert_ne!(late_a.reservation.id, b.reservation.id);
        assert_eq!(
            owner.select(b_key).unwrap().reservation.id,
            b.reservation.id
        );

        let c_key = Key {
            frame_token: 1,
            ..b_key
        };
        let c = owner.select(c_key).unwrap();
        owner.publish(&c.reservation).unwrap();
        assert!(!owner.invalidate(5, 2, 3, 5, b.reservation.id).unwrap());
        assert!(!owner.invalidate(4, 2, 3, 5, c.reservation.id).unwrap());
        assert_eq!(
            owner.select(c_key).unwrap().reservation.id,
            c.reservation.id
        );
        assert!(owner.invalidate(5, 2, 3, 5, c.reservation.id).unwrap());
        assert!(!owner.select(c_key).unwrap().reused);
    }

    #[test]
    fn each_digest_normalizes_before_comparison() {
        for lane in 0..4 {
            let owner = Submission::new();
            owner.bind_identity(4, 5).unwrap();
            let mut k = key();
            let set = |k: &mut Key, value| match lane {
                0 => k.upload_core = value,
                1 => k.dir = value,
                2 => k.scanner = value,
                _ => k.auto_exposure = value,
            };
            set(&mut k, 0);
            let first = owner.select(k).unwrap();
            assert_eq!(first.reservation.key, key());
            owner.publish(&first.reservation).unwrap();
            set(&mut k, 1);
            assert_eq!(
                owner.select(k).unwrap().reservation.id,
                first.reservation.id
            );
            set(&mut k, 2);
            let changed = owner.select(k).unwrap();
            assert_eq!(changed.reservation.id, 2);
            assert_eq!(changed.reservation.key, k);
            assert!(!changed.reused);
        }
    }

    #[test]
    fn overlapping_same_key_shares_tentative_id() {
        let owner = Arc::new(Submission::new());
        owner.bind_identity(1, 2).unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let owner = Arc::clone(&owner);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let selection = owner.select(key()).unwrap();
                    barrier.wait();
                    selection.reservation.id
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), 1);
        }
    }
}
