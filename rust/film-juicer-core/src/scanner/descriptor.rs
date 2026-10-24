//! Complete scanner spectral-LUT metadata. Spectra and labels expire on return.

use crate::{hash, route::ScanRoute};
use std::fmt;

const REPRESENTATION_SCHEMA: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct LutSpectra<'a> {
    pub channel_density_cmy: &'a [[f32; 3]; 81],
    pub base_density: &'a [f32; 81],
}

#[derive(Clone, Copy, Debug)]
pub struct LutDescriptorParams<'a> {
    pub route: ScanRoute,
    pub capture_polarity: u8,
    pub bounds_medium: u8,
    pub density_bounds_hash: u64,
    pub viewing_illuminant: &'a [u8],
    pub observer_identity: &'a [u8],
    pub resolution: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum LutDescriptorInput<'a> {
    Direct {
        params: LutDescriptorParams<'a>,
        film: Option<LutSpectra<'a>>,
    },
    Print {
        params: LutDescriptorParams<'a>,
        print: Option<LutSpectra<'a>>,
        profile_key: &'a [u8],
        profile_token: u64,
        bounds_source: u8,
        output_medium: u8,
        profile_viewing_illuminant: &'a [u8],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LutDescriptorError {
    DirectRecipe,
    DirectBounds,
    DirectObserver,
    DirectIlluminant,
    DirectHash(u32),
    PrintRecipe,
    PrintHash(u32),
}

impl fmt::Display for LutDescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DirectRecipe => {
                "ResourceDescriptorMismatch phase=3A scanner direct recipe unavailable"
            }
            Self::DirectBounds => {
                "ResourceDescriptorMismatch phase=3A scanner density bounds unavailable"
            }
            Self::DirectObserver => {
                "ResourceDescriptorMismatch phase=3A scanner observer identity unavailable"
            }
            Self::DirectIlluminant => {
                "ResourceDescriptorMismatch phase=3A scanner illuminant identity unavailable"
            }
            Self::DirectHash(_) => {
                "ResourceDescriptorMismatch phase=3A scanner descriptor hash invalid"
            }
            Self::PrintRecipe => "ResourceDescriptorMismatch phase=4C scanner print recipe",
            Self::PrintHash(_) => {
                "ResourceDescriptorMismatch phase=4C scanner print descriptor hash"
            }
        })
    }
}
impl std::error::Error for LutDescriptorError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LutDescriptor {
    route: ScanRoute,
    medium: u8,
    capture_polarity: u8,
    density_bounds_hash: u64,
    channel_density_hash: u64,
    base_density_hash: u64,
    viewing_illuminant_hash: u64,
    observer_hash: u64,
    resolution: u32,
    hash: u64,
}

impl LutDescriptor {
    pub fn route(&self) -> ScanRoute {
        self.route
    }
    pub fn medium(&self) -> u8 {
        self.medium
    }
    pub fn capture_polarity(&self) -> u8 {
        self.capture_polarity
    }
    pub fn density_bounds_hash(&self) -> u64 {
        self.density_bounds_hash
    }
    pub fn channel_density_hash(&self) -> u64 {
        self.channel_density_hash
    }
    pub fn base_density_hash(&self) -> u64 {
        self.base_density_hash
    }
    pub fn viewing_illuminant_hash(&self) -> u64 {
        self.viewing_illuminant_hash
    }
    pub fn observer_hash(&self) -> u64 {
        self.observer_hash
    }
    pub fn resolution(&self) -> u32 {
        self.resolution
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

pub fn describe_lut(input: LutDescriptorInput<'_>) -> Result<LutDescriptor, LutDescriptorError> {
    let (params, spectra, medium) = match input {
        LutDescriptorInput::Direct { params, film } => {
            if !matches!(
                params.route,
                ScanRoute::NegativeDirect | ScanRoute::PositiveDirect
            ) {
                return Err(LutDescriptorError::DirectRecipe);
            }
            let film = film.ok_or(LutDescriptorError::DirectRecipe)?;
            if params.density_bounds_hash == 0 || params.bounds_medium != 0 {
                return Err(LutDescriptorError::DirectBounds);
            }
            if params.observer_identity.is_empty() {
                return Err(LutDescriptorError::DirectObserver);
            }
            if params.viewing_illuminant.is_empty() {
                return Err(LutDescriptorError::DirectIlluminant);
            }
            (params, film, 0)
        }
        LutDescriptorInput::Print {
            params,
            print,
            profile_key,
            profile_token,
            bounds_source,
            output_medium,
            profile_viewing_illuminant,
        } => {
            if !matches!(
                params.route,
                ScanRoute::NegativePrint | ScanRoute::PositivePrint
            ) || print.is_none()
                || profile_key.is_empty()
                || profile_token == 0
                || params.density_bounds_hash == 0
                || params.bounds_medium != 1
                || bounds_source != 2
                || output_medium != 1
                || params.viewing_illuminant != profile_viewing_illuminant
                || params.observer_identity.is_empty()
            {
                return Err(LutDescriptorError::PrintRecipe);
            }
            (params, print.ok_or(LutDescriptorError::PrintRecipe)?, 1)
        }
    };
    #[cfg(feature = "test-support")]
    let fault = super::test_support::take_descriptor_fault();
    #[cfg(feature = "test-support")]
    if fault == 6 {
        panic!("scanner LUT descriptor fixture panic");
    }

    let fingerprint = |samples| {
        let streams = hash::f32s_with_nan_mask(samples);
        hash::u64s(&[streams.values, streams.nan_mask])
    };
    let lanes = [
        fingerprint(spectra.channel_density_cmy.as_flattened()),
        fingerprint(spectra.base_density),
        hash::bytes(params.viewing_illuminant),
        hash::bytes(params.observer_identity),
    ];
    #[cfg(feature = "test-support")]
    let lanes = {
        let mut lanes = lanes;
        if (1..=4).contains(&fault) {
            lanes[fault as usize - 1] = 0;
        }
        lanes
    };

    let mut identity = hash::FNV_OFFSET;
    hash::update_bytes(
        &mut identity,
        &[params.route as u8, medium, params.capture_polarity],
    );
    for lane in [
        params.density_bounds_hash,
        lanes[0],
        lanes[1],
        lanes[2],
        lanes[3],
    ] {
        hash::update_bytes(&mut identity, &lane.to_le_bytes());
    }
    hash::update_bytes(&mut identity, &params.resolution.to_le_bytes());
    hash::update_bytes(&mut identity, &REPRESENTATION_SCHEMA.to_le_bytes());
    #[cfg(feature = "test-support")]
    if fault == 5 {
        identity = 0;
    }
    if let Some(index) = lanes
        .iter()
        .chain(std::iter::once(&identity))
        .position(|hash| *hash == 0)
    {
        let lane = index as u32 + 1;
        return Err(if medium == 0 {
            LutDescriptorError::DirectHash(lane)
        } else {
            LutDescriptorError::PrintHash(lane)
        });
    }
    Ok(LutDescriptor {
        route: params.route,
        medium,
        capture_polarity: params.capture_polarity,
        density_bounds_hash: params.density_bounds_hash,
        channel_density_hash: lanes[0],
        base_density_hash: lanes[1],
        viewing_illuminant_hash: lanes[2],
        observer_hash: lanes[3],
        resolution: params.resolution,
        hash: identity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn construct(input: &Value) -> Result<LutDescriptor, LutDescriptorError> {
        let channel: Vec<f32> = input["channel_density_cmy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
            .collect();
        let base: Vec<f32> = input["base_density"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
            .collect();
        let bytes = |name: &str| -> Vec<u8> {
            input[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_u64().unwrap() as u8)
                .collect()
        };
        let viewing = bytes("viewing_illuminant");
        let observer = bytes("observer_identity");
        let key = bytes("print_profile_key");
        let profile_viewing = bytes("profile_viewing_illuminant");
        let route = match input["route"].as_u64().unwrap() {
            0 => ScanRoute::NegativeDirect,
            1 => ScanRoute::NegativePrint,
            2 => ScanRoute::PositiveDirect,
            3 => ScanRoute::PositivePrint,
            _ => unreachable!(),
        };
        let params = LutDescriptorParams {
            route,
            capture_polarity: input["capture_polarity"].as_u64().unwrap() as u8,
            bounds_medium: input["bounds_medium"].as_u64().unwrap() as u8,
            density_bounds_hash: input["density_bounds_hash"].as_u64().unwrap(),
            resolution: input["resolution"].as_u64().unwrap() as u32,
            viewing_illuminant: &viewing,
            observer_identity: &observer,
        };
        let spectra = input["profile_present"]
            .as_bool()
            .unwrap()
            .then_some(LutSpectra {
                channel_density_cmy: channel.as_chunks::<3>().0.try_into().unwrap(),
                base_density: base.as_slice().try_into().unwrap(),
            });
        if input["mode"] == 0 {
            describe_lut(LutDescriptorInput::Direct {
                params,
                film: spectra,
            })
        } else {
            describe_lut(LutDescriptorInput::Print {
                params,
                print: spectra,
                profile_key: &key,
                profile_token: input["print_profile_token"].as_u64().unwrap(),
                bounds_source: input["bounds_source"].as_u64().unwrap() as u8,
                output_medium: input["output_medium"].as_u64().unwrap() as u8,
                profile_viewing_illuminant: &profile_viewing,
            })
        }
    }

    #[test]
    fn independent_native_descriptors_and_expired_inputs() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/scanner/descriptor-fixtures");
        let read = |name| -> Value {
            serde_json::from_slice(&std::fs::read(directory.join(name)).unwrap()).unwrap()
        };
        let manifest = read("manifest.json");
        let preset = match (cfg!(windows), cfg!(debug_assertions)) {
            (false, true) => "linux-debug",
            (false, false) => "linux-release",
            (true, true) => "windows-clang-debug",
            (true, false) => "windows-clang-release",
        };
        let logical = manifest["presets"][preset]["native"].as_str().unwrap();
        let fixture = read(manifest["payloads"][logical].as_str().unwrap());
        let context = read(manifest["contexts"][logical].as_str().unwrap());
        assert_eq!(context["preset"], preset);
        assert_eq!(fixture["cases"].as_array().unwrap().len(), 520);
        assert_eq!(fixture["products"].as_array().unwrap().len(), 4);
        for row in fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .chain(fixture["products"].as_array().unwrap())
        {
            let input = &row["input"];
            if matches!(input["missing"].as_u64().unwrap(), 1..=3)
                || input["route"].as_u64().unwrap() > 3
            {
                continue;
            }
            // All decoding storage in construct has expired before any field is read.
            match construct(input) {
                Ok(value) => {
                    assert_eq!(row["native"]["success"], true);
                    let observed = json!({"route":value.route() as u8,"medium":value.medium(),
                        "capture_polarity":value.capture_polarity(),"resolution":value.resolution(),
                        "density_bounds_hash":value.density_bounds_hash(),"channel_density_hash":value.channel_density_hash(),
                        "base_density_hash":value.base_density_hash(),"viewing_illuminant_hash":value.viewing_illuminant_hash(),
                        "observer_hash":value.observer_hash(),"hash":value.hash()});
                    assert_eq!(
                        observed, row["native"]["descriptor"],
                        "{preset} {}",
                        row["id"]
                    );
                }
                Err(error) => {
                    assert_eq!(row["native"]["success"], false);
                    assert_eq!(
                        error.to_string(),
                        row["native"]["diagnostic"].as_str().unwrap()
                    );
                }
            }
        }
    }
}
