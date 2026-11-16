//! Resolved highlight reconstruction curve for the camera exposure carrier.

use crate::hash;

pub const MAX_EV: f64 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Controls {
    pub boost_ev: f64,
    pub protection_ev: f64,
    pub range: f64,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            boost_ev: 0.0,
            protection_ev: 4.0,
            range: 0.3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    BoostEv,
    ProtectionEv,
    Range,
}

impl Error {
    pub fn diagnostic(self) -> &'static str {
        match self {
            Self::BoostEv => "InvalidAuthoredControl component=highlight_boost field=boost_ev",
            Self::ProtectionEv => {
                "InvalidAuthoredControl component=highlight_boost field=protection_ev"
            }
            Self::Range => "InvalidAuthoredControl component=highlight_boost field=range",
        }
    }
}

impl Controls {
    pub fn validate_ev(boost_ev: f64) -> Result<f64, Error> {
        bounded(boost_ev, MAX_EV, Error::BoostEv)
    }

    pub fn validate(self) -> Result<Self, Error> {
        let boost_ev = Self::validate_ev(self.boost_ev)?;
        if boost_ev == 0.0 {
            return Ok(Self::default());
        }
        Ok(Self {
            boost_ev,
            protection_ev: bounded(self.protection_ev, MAX_EV, Error::ProtectionEv)?,
            range: bounded(self.range, 1.0, Error::Range)?,
        })
    }
}

fn bounded(value: f64, maximum: f64, error: Error) -> Result<f64, Error> {
    if (0.0..=maximum).contains(&value) {
        Ok(if value == 0.0 { 0.0 } else { value })
    } else {
        Err(error)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    protection: f64,
    sharpness: f64,
    peak_gain: f64,
    hash: u64,
}

impl Recipe {
    pub(crate) fn new(controls: Controls) -> Result<Option<Self>, Error> {
        let controls = controls.validate()?;
        if controls.boost_ev == 0.0 {
            return Ok(None);
        }
        let protection = 0.184 * controls.protection_ev.exp2();
        let sharpness = 28.0_f64.powf(1.0 - controls.range);
        let peak_gain = controls.boost_ev.exp2();
        let identity = hash::u64s(&[
            hash::bytes(b"highlight-boost"),
            protection.to_bits(),
            sharpness.to_bits(),
            peak_gain.to_bits(),
        ]);
        Ok(Some(Self {
            protection,
            sharpness,
            peak_gain,
            hash: if identity == 0 { 1 } else { identity },
        }))
    }

    pub fn protection(self) -> f64 {
        self.protection
    }

    pub fn sharpness(self) -> f64 {
        self.sharpness
    }

    pub fn peak_gain(self) -> f64 {
        self.peak_gain
    }

    pub fn hash(self) -> u64 {
        self.hash
    }
}
