//! Explicit Film v1 bank authoring input from the product capture catalog.

use std::{env, error::Error, io, path::PathBuf};

use film_juicer_core::{
    assets::Assets,
    profile::{CatalogEntry, Polarity, Stage},
};
use serde::Serialize;

#[derive(Serialize)]
struct Capture<'a> {
    stock: &'a str,
    polarity: &'static str,
    source: &'a std::path::Path,
}

impl<'a> From<&'a CatalogEntry> for Capture<'a> {
    fn from(entry: &'a CatalogEntry) -> Self {
        Self {
            stock: entry.key(),
            polarity: match entry.polarity() {
                Polarity::Negative => "negative",
                Polarity::Positive => "positive",
            },
            source: entry.source_path(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let (Some(root), Some(selection), None) = (args.next(), args.next(), args.next()) else {
        return Err("usage: export_grain_response RESOURCES STOCK_KEY|--list".into());
    };
    let assets = Assets::new(PathBuf::from(root));
    let catalog = assets.catalog()?;
    let output = io::stdout().lock();
    if selection == "--list" {
        let captures: Vec<_> = catalog.films().iter().map(Capture::from).collect();
        serde_json::to_writer(output, &captures)?;
        return Ok(());
    }

    let key = selection.to_str().ok_or("stock key must be UTF-8")?;
    let entry = catalog
        .film(key)
        .filter(|entry| entry.stage() == Stage::Filming)
        .ok_or_else(|| format!("{key} is not a capture stock key"))?;
    let profile = assets.film(key)?;
    let tables = profile.tables();
    let exposure = tables.interpolation_log_exposure();
    let density = tables.density_curves();
    if !density.iter().flatten().all(|value| value.is_finite()) {
        return Err(format!("{key} has a nonfinite evaluated CDF3 response").into());
    }
    // The completed profile evaluates authored CDF3 coefficients at gamma 1.
    // These are exposure knots before development normalization and DIR.
    let response = serde_json::json!({
        "capture": Capture::from(entry),
        "data": {
            "log_exposure": exposure.iter().copied().map(f64::from).collect::<Vec<_>>(),
            "density_curves": density.iter().map(|row| row.map(f64::from)).collect::<Vec<_>>(),
        },
    });
    serde_json::to_writer(output, &response)?;
    Ok(())
}
