//! Capture-film scanner/enlarger bounds and the spektrafilm grain minimum.

use std::fmt;

use crate::hash;
use crate::profile::Polarity;
use crate::route::ScanRoute;
use super::Development;
pub(crate) mod portra;

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
    FilmNormalization,
    FilmProjectionSpan,
    FilmSpectralRange,
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
            Self::FilmNormalization => {
                f.write_str("Film v1 normalization falls outside the Portra density certificate")
            }
            Self::FilmProjectionSpan => {
                f.write_str("Portra Film v1 projection requires a finite CMY span within 64 D")
            }
            Self::FilmSpectralRange => {
                f.write_str("Portra Film v1 exceeds the scanner and enlarger transmission range")
            }
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
        Self::from_range(
            development,
            minimum.cmy.map(|v| -v),
            *development.maximum_cmy(),
        )
    }
    pub(crate) fn film_v1(development: &Development, allowances: [f32; 2]) -> Result<Self, Error> {
        let [lower, upper] = allowances;
        let maximum_cmy = Self::film_v1_clean_maximum(development).map(|d| (d + upper).next_up());
        Self::checked_film_v1(development, [lower; 3], maximum_cmy)
    }
    pub(crate) fn portra_film_v1(
        development: &Development,
        bank: &crate::assets::PortraGrainBank,
        table: &crate::assets::PortraGrainTable,
        gain: f32,
        chroma_fraction: f32,
    ) -> Result<Self, Error> {
        let allowances = portra::allowances(bank, table, gain).ok_or(Error::FilmNormalization)?;
        // Preserve the admitted Film v1 box. Where that
        // enclosure exceeds a safeguard, test the more precise certificate.
        let original = Self::film_v1(development, allowances);
        if matches!(
            original,
            Err(Error::FilmProjectionSpan | Error::FilmSpectralRange)
        ) {
            if let Some([lo, hi]) = portra::density_range(
                bank,
                table,
                gain,
                chroma_fraction,
                Self::film_v1_clean_maximum(development),
            ) {
                return Self::checked_film_v1(development, lo, hi);
            }
            if let Some(allowances) = portra::periodic_allowances(table, gain) {
                return Self::film_v1(development, allowances);
            }
        }
        original
    }
    pub(crate) fn film_v1_clean_maximum(development: &Development) -> [f32; 3] {
        // Development samples its normalized knots with capped linear
        // interpolation. Thinning by a factor in [0, 1] stays inside this
        // interval because it contains zero.
        std::array::from_fn(|channel| {
            let maximum = development.maximum_cmy()[channel];
            let minimum = development.minimum_cmy()[channel];
            maximum.max(maximum - minimum)
        })
    }
    fn checked_film_v1(
        development: &Development,
        minimum_cmy: [f32; 3],
        maximum_cmy: [f32; 3],
    ) -> Result<Self, Error> {
        let lower = minimum_cmy.into_iter().fold(0.0, f32::min);
        let upper = maximum_cmy.into_iter().fold(0.0, f32::max);
        if !(0.0..=64.0).contains(&(f64::from(upper) - f64::from(lower))) {
            return Err(Error::FilmProjectionSpan);
        }
        let bounds = Self::from_range(development, minimum_cmy, maximum_cmy)?;
        let lo = bounds.minimum_cmy.map(f64::from);
        let hi = bounds.maximum_cmy.map(f64::from);
        let tables = development.profile().tables();
        check_spectral_floor(lo, hi, tables.channel_density(), tables.base_density())?;
        Ok(bounds)
    }

    fn from_range(
        development: &Development,
        minimum_cmy: [f32; 3],
        maximum_cmy: [f32; 3],
    ) -> Result<Self, Error> {
        let route = development.route();
        let source = match route {
            ScanRoute::NegativeDirect | ScanRoute::PositiveDirect => Source::DirectFilm,
            ScanRoute::NegativePrint | ScanRoute::PositivePrint => Source::EnlargerFilm,
        };
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
fn check_spectral_floor(
    lo: [f64; 3],
    hi: [f64; 3],
    dyes: &[[f32; 3]],
    baseline: &[f32],
) -> Result<(), Error> {
    for (&dyes, &base) in dyes.iter().zip(baseline) {
        // Signed coefficients select the brightest corner of the reachable box.
        let [c, m, y] = std::array::from_fn(|k| if dyes[k] >= 0.0 { lo[k] } else { hi[k] });
        let density = c * f64::from(dyes[0])
            + m * f64::from(dyes[1])
            + y * f64::from(dyes[2])
            + f64::from(base);
        if density < -30.0 {
            return Err(Error::FilmSpectralRange);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assets::Assets, film};

    #[test]
    fn film_spectral_floor_covers_the_evaluated_upper_node() {
        let assets =
            Assets::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Resources"));
        let film = assets.film("kodak_portra_400").unwrap();
        let dyes = [[-1.0, 0.0, 0.0]];
        let baseline = [-5.0];
        for route in [ScanRoute::NegativeDirect, ScanRoute::NegativePrint] {
            let development = Development::new(
                film.clone(),
                film::Params {
                    gamma_factor: 1.0,
                    layers_required: false,
                    requested_route: route,
                },
            )
            .unwrap();
            let bounds = Bounds::from_range(&development, [0.0; 3], [25.0, 4.0, 3.0]).unwrap();
            assert!(check_spectral_floor([0.0; 3], [25.0, 4.0, 3.0], &dyes, &baseline).is_ok());
            let lo = bounds.minimum_cmy.map(f64::from);
            let hi = bounds.maximum_cmy.map(f64::from);
            let result = check_spectral_floor(lo, hi, &dyes, &baseline);
            assert_eq!(hi[0], 25.0);
            assert!(result.is_ok());
            assert!(matches!(
                check_spectral_floor(lo, hi, &dyes, &[baseline[0].next_down()]),
                Err(Error::FilmSpectralRange)
            ));
        }
    }
}
