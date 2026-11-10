//! Capture-film scanner/enlarger bounds and the spektrafilm grain minimum.

use std::fmt;

use crate::hash;
use crate::profile::Polarity;
use crate::route::ScanRoute;
use super::Development;

#[derive(Clone, Copy, Debug)]
pub struct Minimum {
    cmy: [f32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Source {
    DirectFilm,
    EnlargerFilm,
}

#[derive(Debug)]
pub enum Error {
    Minimum,
    Span { channel: usize },
    ZeroIdentity,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimum => f.write_str("film grain minimum requires finite nonnegative CMY"),
            Self::Span { channel } => write!(
                f,
                "film density span requires finite positive result channel={channel}"
            ),
            Self::ZeroIdentity => f.write_str("film density bounds identity is zero"),
        }
    }
}
impl std::error::Error for Error {}

impl Minimum {
    pub fn new(cmy: [f32; 3]) -> Result<Self, Error> {
        if cmy.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::Minimum);
        }
        Ok(Self {
            cmy: cmy.map(|v| v.clamp(0.0, 1.0)),
        })
    }

    pub fn cmy(&self) -> &[f32; 3] {
        &self.cmy
    }
}

pub struct Bounds {
    route: ScanRoute,
    polarity: Polarity,
    source: Source,
    minimum_cmy: [f32; 3],
    maximum_cmy: [f32; 3],
    inv_span_cmy: [f32; 3],
    authored_minimum_cmy: [f32; 3],
    authored_maximum_cmy: [f32; 3],
    hash: u64,
}

impl Bounds {
    pub fn new(development: &Development, minimum: &Minimum) -> Result<Self, Error> {
        let route = development.route();
        let source = match route {
            ScanRoute::NegativeDirect | ScanRoute::PositiveDirect => Source::DirectFilm,
            ScanRoute::NegativePrint | ScanRoute::PositivePrint => Source::EnlargerFilm,
        };
        let minimum_cmy = minimum.cmy.map(|v| -v);
        let maximum_cmy = *development.maximum_cmy();
        let mut inv_span_cmy = [0.0; 3];
        for channel in 0..3 {
            let span = maximum_cmy[channel] - minimum_cmy[channel];
            if !span.is_finite() || span <= 0.0 {
                return Err(Error::Span { channel });
            }
            inv_span_cmy[channel] = 1.0 / span;
        }
        let mut result = Self {
            route,
            polarity: development.polarity(),
            source,
            minimum_cmy,
            maximum_cmy,
            inv_span_cmy,
            authored_minimum_cmy: *development.minimum_cmy(),
            authored_maximum_cmy: *development.maximum_cmy(),
            hash: hash::FNV_OFFSET,
        };
        let route_tag = match route {
            ScanRoute::NegativeDirect => 0,
            ScanRoute::NegativePrint => 1,
            ScanRoute::PositiveDirect => 2,
            ScanRoute::PositivePrint => 3,
        };
        let polarity_tag = match result.polarity {
            Polarity::Negative => 0,
            Polarity::Positive => 1,
        };
        let source_tag = match source {
            Source::DirectFilm => 0,
            Source::EnlargerFilm => 1,
        };
        hash::update_bytes(&mut result.hash, &[route_tag, 0, polarity_tag, source_tag]);
        for values in [
            &result.minimum_cmy,
            &result.maximum_cmy,
            &result.inv_span_cmy,
            &result.authored_minimum_cmy,
            &result.authored_maximum_cmy,
        ] {
            for value in values {
                hash::update_bytes(&mut result.hash, &value.to_bits().to_le_bytes());
            }
        }
        #[cfg(feature = "test-support")]
        if test_support::take() {
            result.hash = 0;
        }
        if result.hash == 0 {
            return Err(Error::ZeroIdentity);
        }
        Ok(result)
    }

    pub fn route(&self) -> ScanRoute {
        self.route
    }
    pub fn polarity(&self) -> Polarity {
        self.polarity
    }
    pub fn source(&self) -> Source {
        self.source
    }
    pub fn minimum_cmy(&self) -> &[f32; 3] {
        &self.minimum_cmy
    }
    pub fn maximum_cmy(&self) -> &[f32; 3] {
        &self.maximum_cmy
    }
    pub fn inv_span_cmy(&self) -> &[f32; 3] {
        &self.inv_span_cmy
    }
    pub fn authored_minimum_cmy(&self) -> &[f32; 3] {
        &self.authored_minimum_cmy
    }
    pub fn authored_maximum_cmy(&self) -> &[f32; 3] {
        &self.authored_maximum_cmy
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(feature = "test-support")]
pub mod test_support {
    use std::cell::Cell;
    thread_local! { static ZERO: Cell<bool> = const { Cell::new(false) }; }
    pub fn zero_next_identity() {
        ZERO.set(true);
    }
    pub(super) fn take() -> bool {
        ZERO.replace(false)
    }
}
