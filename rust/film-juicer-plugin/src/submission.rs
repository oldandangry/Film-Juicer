//! Per-instance identity and the current submission latch.
#![forbid(unsafe_code)]

use std::fmt;
use std::sync::{Mutex, OnceLock};

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
    pub(crate) upload_core: u64,
    pub(crate) dir: u64,
    pub(crate) scanner: u64,
    pub(crate) auto_exposure: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Selection {
    pub(crate) id: u64,
    pub(crate) key: Key,
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
        if let Some(latch) = &state.latch
            && latch.key == key
        {
            return Ok(Selection {
                id: latch.id,
                key,
                reused: true,
            });
        }
        let id = state.next_id;
        state.next_id = id.wrapping_add(1).max(1);
        state.latch = Some(Latch { key, id });
        Ok(Selection {
            id,
            key,
            reused: false,
        })
    }
    #[cfg(feature = "test-support")]
    pub(crate) fn observe(&self) -> Result<Option<Selection>, Error> {
        let state = self.state.lock().map_err(|_| Error::Poisoned)?;
        Ok(state.latch.as_ref().map(|latch| Selection {
            id: latch.id,
            key: latch.key,
            reused: false,
        }))
    }
    #[cfg(test)]
    pub(crate) fn poison_for_test(&self) {
        let _ = std::panic::catch_unwind(|| {
            let _guard = self.state.lock().unwrap();
            panic!("poison submission latch");
        });
    }
    pub(crate) fn invalidate(&self, device_id: i32, context: usize) -> Result<bool, Error> {
        let mut state = self.state.lock().map_err(|_| Error::Poisoned)?;
        let matches = state
            .latch
            .as_ref()
            .is_some_and(|latch| latch.key.device_id == device_id && latch.key.context == context);
        if matches {
            state.latch = None;
        }
        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    fn key() -> Key {
        Key {
            frame_token: 0,
            device_id: 2,
            context: 3,
            upload_core: 1,
            dir: 1,
            scanner: 1,
            auto_exposure: 1,
        }
    }
    #[test]
    fn identity_and_latch() {
        let owner = Submission::new();
        assert_eq!(owner.identity(), None);
        assert_eq!(owner.select(key()), Err(Error::MissingIdentity));
        owner.bind_identity(0, 0).unwrap();
        assert_eq!(
            owner.identity(),
            Some(Identity {
                session_seed: 0,
                instance_token: 0
            })
        );
        assert_eq!(owner.select(key()).unwrap().id, 1);
        assert_eq!(owner.bind_identity(2, 3), Err(Error::DuplicateIdentity));
        assert!(owner.select(key()).unwrap().reused);
        let mut changed = key();
        changed.frame_token = 1;
        assert_eq!(owner.select(changed).unwrap().id, 2);
        assert_eq!(owner.select(key()).unwrap().id, 3);
        assert!(!owner.invalidate(2, 4).unwrap());
        assert!(owner.select(key()).unwrap().reused);
        assert!(owner.invalidate(2, 3).unwrap());
        assert_eq!(owner.select(key()).unwrap().id, 4);
        let other = Submission::new();
        other.bind_identity(7, 8).unwrap();
        assert_eq!(other.select(key()).unwrap().id, 1);
        for change in 0..2 {
            let mut k = key();
            if change == 0 {
                k.device_id += 1;
            } else {
                k.context += 1;
            }
            assert!(!owner.select(k).unwrap().reused);
        }
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
            assert_eq!(first.key, key());
            set(&mut k, 1);
            assert_eq!(
                owner.select(k).unwrap(),
                Selection {
                    reused: true,
                    ..first
                }
            );
            set(&mut k, 2);
            let changed = owner.select(k).unwrap();
            assert_eq!(changed.id, 2);
            assert_eq!(changed.key, k);
            assert!(!changed.reused);
        }
    }
    #[test]
    fn concurrent_reuse_wrap_and_poison() {
        let owner = Arc::new(Submission::new());
        owner.bind_identity(1, 2).unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let owner = Arc::clone(&owner);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    owner.select(key()).unwrap().id
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), 1);
        }
        {
            let mut state = owner.state.lock().unwrap();
            state.next_id = u64::MAX;
            state.latch = None;
        }
        assert_eq!(owner.select(key()).unwrap().id, u64::MAX);
        let mut changed = key();
        changed.frame_token = 2;
        assert_eq!(owner.select(changed).unwrap().id, 1);
        let poisoned = Arc::clone(&owner);
        let _ = std::thread::spawn(move || {
            let _guard = poisoned.state.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert_eq!(owner.select(key()), Err(Error::Poisoned));
        assert_eq!(owner.invalidate(2, 3), Err(Error::Poisoned));
    }
}
