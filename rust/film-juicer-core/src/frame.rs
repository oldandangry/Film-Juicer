//! Callback-local signed domains and physical pixel size. These values do not
//! admit image allocation coverage or downstream positive physical dimensions.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Domain {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainError {
    Range,
}

impl Domain {
    pub fn from_bounds(bounds: Bounds) -> Result<Self, DomainError> {
        let width = i64::from(bounds.x2) - i64::from(bounds.x1);
        let height = i64::from(bounds.y2) - i64::from(bounds.y1);
        Ok(Self {
            x: bounds.x1,
            y: bounds.y1,
            width: i32::try_from(width).map_err(|_| DomainError::Range)?,
            height: i32::try_from(height).map_err(|_| DomainError::Range)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    domain: Domain,
    pixel_size_um: f32,
}

impl Geometry {
    pub fn new(domain: Domain, film_format_long_edge_mm: f32) -> Self {
        let long_edge = f64::from(domain.width.max(domain.height));
        let film_mm = f64::from(film_format_long_edge_mm);
        let pixel_size_um = if film_mm > 0.0 && long_edge > 0.0 {
            ((film_mm * 1000.0) / long_edge) as f32
        } else {
            0.0
        };
        Self {
            domain,
            pixel_size_um,
        }
    }

    pub fn domain(&self) -> Domain {
        self.domain
    }

    pub fn pixel_size_um(&self) -> f32 {
        self.pixel_size_um
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_native_capture() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/frame/geometry-fixtures");
        let read = |name: &str| -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap()
        };
        let manifest = read("manifest.json");
        let preset = match (cfg!(windows), cfg!(debug_assertions)) {
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
        };
        let logical = manifest["presets"][preset]["native"].as_str().unwrap();
        assert_eq!(
            read(manifest["contexts"][logical].as_str().unwrap())["preset"],
            preset
        );
        let captured = read(manifest["payloads"][logical].as_str().unwrap());
        let array =
            |row: &serde_json::Value| -> [i32; 4] { serde_json::from_value(row.clone()).unwrap() };
        for row in captured["domains"].as_array().unwrap() {
            let [x1, y1, x2, y2] = array(&row["bounds"]);
            let result = Domain::from_bounds(Bounds { x1, y1, x2, y2 });
            if row["range"].as_bool().unwrap() {
                assert_eq!(result, Err(DomainError::Range), "{}", row["name"]);
            } else {
                let d = result.unwrap();
                assert_eq!(
                    [d.x, d.y, d.width, d.height],
                    array(&row["domain"]),
                    "{}",
                    row["name"]
                );
            }
        }
        for row in captured["geometry"].as_array().unwrap() {
            let [x, y, width, height] = array(&row["domain"]);
            let domain = Domain {
                x,
                y,
                width,
                height,
            };
            let result = Geometry::new(
                domain,
                f32::from_bits(row["film_bits"].as_u64().unwrap() as u32),
            );
            assert_eq!(result.domain(), domain);
            assert_eq!(
                result.pixel_size_um().to_bits(),
                row["pixel_bits"].as_u64().unwrap() as u32,
                "{}",
                row["name"]
            );
            assert_eq!(
                f64::from(result.pixel_size_um()).to_bits(),
                row["diffusion_bits"].as_u64().unwrap(),
                "{}",
                row["name"]
            );
        }
    }
}
