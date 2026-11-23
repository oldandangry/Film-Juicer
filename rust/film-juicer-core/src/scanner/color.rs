//! Complete scanner output preparation; no pixel execution or retained input.

use crate::{color, hash, spectral};
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct ColorInput {
    pub medium: u8,
    pub output_space: u8,
    pub apply_cctf: bool,
    pub illuminant_xyz: [f32; 3],
    pub illuminant_hash: u64,
    pub active_gamut_recipe_hash: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorError {
    Illuminant,
    Output,
    #[cfg(feature = "test-support")]
    OutputBoundary,
    #[cfg(feature = "test-support")]
    Cat02Boundary,
}
impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Illuminant => "scanner illuminant hash invalid for color runtime",
            Self::Output => "generated color space hash invalid",
            #[cfg(feature = "test-support")]
            Self::OutputBoundary => "Output color preparation failed",
            #[cfg(feature = "test-support")]
            Self::Cat02Boundary => "CAT02 matrix preparation failed",
        })
    }
}
impl std::error::Error for ColorError {}

#[derive(Clone, Copy)]
pub struct Color {
    cat02: [f32; 9],
    xyz_to_rgb: [f32; 9],
    illuminant_xyz: [f32; 3],
    output_space: u8,
    apply_cctf: bool,
    cctf: color::Cctf,
    dwg_to_output: [f32; 9],
    active_gamut_recipe_hash: u64,
    hash: u64,
}
impl Color {
    pub fn new(input: ColorInput) -> Result<Self, ColorError> {
        if input.illuminant_hash == 0 {
            return Err(ColorError::Illuminant);
        }
        #[cfg(feature = "test-support")]
        crate::film::input::testing::color_fault(11).map_err(|_| ColorError::OutputBoundary)?;
        let space = color::OutputSpace::try_from(u32::from(input.output_space))
            .unwrap_or(color::OutputSpace::Srgb);
        let output = color::output_color(space);
        if output.hash() == 0 {
            return Err(ColorError::Output);
        }
        let cat02 = match color::checked_cat02_matrix(color::Whites {
            source_xyz: input.illuminant_xyz,
            destination_xyz: *output.white_xyz(),
        }) {
            Ok(matrix) => matrix,
            #[cfg(feature = "test-support")]
            Err(color::CheckedError::UnsupportedInput) => {
                return Err(ColorError::Cat02Boundary);
            }
            #[cfg(not(feature = "test-support"))]
            Err(error) => match error {},
        };
        let encoding = hash::u64s(&[
            u64::from(input.output_space),
            u64::from(input.apply_cctf),
            1,
        ]);
        let hash = hash::u64s(&[
            output.hash(),
            input.illuminant_hash,
            u64::from(input.medium),
            encoding,
            input.active_gamut_recipe_hash,
        ]);
        Ok(Self {
            cat02,
            xyz_to_rgb: *output.xyz_to_rgb(),
            illuminant_xyz: input.illuminant_xyz,
            output_space: input.output_space,
            apply_cctf: input.apply_cctf,
            cctf: output.cctf(),
            dwg_to_output: *output.dwg_to_output(),
            active_gamut_recipe_hash: input.active_gamut_recipe_hash,
            hash,
        })
    }
    pub fn cat02(&self) -> &[f32; 9] {
        &self.cat02
    }
    pub fn xyz_to_rgb(&self) -> &[f32; 9] {
        &self.xyz_to_rgb
    }
    pub fn illuminant_xyz(&self) -> &[f32; 3] {
        &self.illuminant_xyz
    }
    pub fn output_space(&self) -> u8 {
        self.output_space
    }
    pub fn apply_cctf(&self) -> bool {
        self.apply_cctf
    }
    pub fn input_is_output_space(&self) -> bool {
        true
    }
    pub fn cctf(&self) -> color::Cctf {
        self.cctf
    }
    pub fn dwg_to_output(&self) -> &[f32; 9] {
        &self.dwg_to_output
    }
    pub fn active_gamut_recipe_hash(&self) -> u64 {
        self.active_gamut_recipe_hash
    }
    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub(crate) fn spectral_binding_digest(
        &self,
        route: u8,
        profile_key: &str,
        asset_token: u64,
        tables: &spectral::Tables,
        gamut_enabled: bool,
    ) -> u64 {
        let mut digest = hash::FNV_OFFSET;
        hash::update_bytes(&mut digest, &1_u32.to_le_bytes());
        hash::update_bytes(&mut digest, &u32::from(route).to_le_bytes());
        hash::update_bytes(&mut digest, &(profile_key.len() as u64).to_le_bytes());
        hash::update_bytes(&mut digest, profile_key.as_bytes());
        hash::update_bytes(&mut digest, &asset_token.to_le_bytes());
        for lane in tables.dyes_cmy().iter().chain(tables.weighted_xyz()) {
            hash::update_resource_f32s(&mut digest, lane);
        }
        hash::update_bytes(&mut digest, &[u8::from(tables.has_baseline())]);
        if tables.has_baseline() {
            hash::update_resource_f32s(&mut digest, tables.baseline_min());
        }
        hash::update_resource_f32s(&mut digest, &[tables.inv_yn()]);
        hash::update_resource_f32s(&mut digest, &self.illuminant_xyz);
        hash::update_resource_f32s(&mut digest, &self.cat02);
        hash::update_resource_f32s(&mut digest, &self.xyz_to_rgb);
        let flags = u32::from(self.apply_cctf)
            | (u32::from(self.input_is_output_space()) << 1)
            | (u32::from(gamut_enabled) << 2);
        hash::update_bytes(&mut digest, &u32::from(self.output_space).to_le_bytes());
        hash::update_bytes(&mut digest, &flags.to_le_bytes());
        hash::update_bytes(&mut digest, &1_u32.to_le_bytes());
        let cctf = self.cctf;
        hash::update_bytes(&mut digest, &(cctf.kind as u32).to_le_bytes());
        hash::update_resource_f32s(
            &mut digest,
            &[
                cctf.gamma,
                cctf.a,
                cctf.b,
                cctf.c,
                cctf.d,
                cctf.linear_cutoff,
            ],
        );
        hash::update_resource_f32s(&mut digest, &self.dwg_to_output);
        digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_cases::RowCases;
    use serde_json::{Value, json};

    fn words(values: &[f32]) -> Value {
        json!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
    }
    #[test]
    fn independent_complete_values_and_owned_results() {
        let fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/scanner/color-fixtures/native.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let rows = fixture["cases"].as_array().unwrap();
        let mut cases = RowCases::new(
            "scanner-color-inputs",
            rows.iter().map(|row| row["id"].as_str().unwrap()),
        );
        let successful = rows
            .iter()
            .filter(|row| row["input"]["illuminant_hash"].as_u64().unwrap() != 0)
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let mut result_cases = RowCases::new("scanner-color-results", successful);
        let mut held = Vec::new();
        for row in rows {
            let i = &row["input"];
            let id = row["id"].as_str().unwrap();
            let case = cases.case(id);
            let input = ColorInput {
                medium: i["medium"].as_u64().unwrap() as u8,
                output_space: i["output_space"].as_u64().unwrap() as u8,
                apply_cctf: i["apply_cctf"].as_bool().unwrap(),
                illuminant_xyz: std::array::from_fn(|c| {
                    f32::from_bits(i["illuminant_xyz"][c].as_u64().unwrap() as u32)
                }),
                illuminant_hash: i["illuminant_hash"].as_u64().unwrap(),
                active_gamut_recipe_hash: i["active_gamut_recipe_hash"].as_u64().unwrap(),
            };
            if input.illuminant_hash == 0 {
                assert!(matches!(Color::new(input), Err(ColorError::Illuminant)));
                case.passed();
                continue;
            }
            let value = Color::new(input).unwrap();
            held.push((id.to_owned(), value, row["expected"].clone()));
            case.passed();
        }
        cases.finish();
        for (id, value, expected) in held {
            let case = result_cases.case(id);
            assert_eq!(words(value.cat02()), expected["cat02"]);
            assert_eq!(words(value.xyz_to_rgb()), expected["xyz_to_rgb"]);
            assert_eq!(words(value.illuminant_xyz()), expected["illuminant_xyz"]);
            assert_eq!(
                json!([
                    value.output_space(),
                    value.apply_cctf(),
                    value.input_is_output_space()
                ]),
                expected["params"]
            );
            assert_eq!(json!(value.hash()), expected["hash"]);
            assert_eq!(
                json!(value.active_gamut_recipe_hash()),
                expected["active_gamut_recipe_hash"]
            );
            let c = value.cctf();
            assert_eq!(json!(c.kind as u8), expected["encoding"]["kind"]);
            assert_eq!(
                words(&[c.gamma, c.a, c.b, c.c, c.d, c.linear_cutoff]),
                expected["encoding"]["coefficients"]
            );
            assert_eq!(
                words(value.dwg_to_output()),
                expected["encoding"]["dwg_to_output"]
            );
            case.passed();
        }
        result_cases.finish();
    }
    #[cfg(feature = "test-support")]
    #[test]
    fn staged_faults_are_thread_local_counted_and_one_shot() {
        use crate::film::input::testing;
        let input = ColorInput {
            medium: 255,
            output_space: 255,
            apply_cctf: false,
            illuminant_xyz: [0.0; 3],
            illuminant_hash: 1,
            active_gamut_recipe_hash: 0,
        };
        let rows = [
            (11, ColorError::OutputBoundary),
            (3, ColorError::Cat02Boundary),
        ];
        let ids = ["output-boundary", "cat02-boundary"];
        let mut cases = RowCases::new("scanner-color-faults", ids);
        for ((operation, error), id) in rows.into_iter().zip(ids) {
            let case = cases.case(id);
            testing::arm_color(operation, 2, 1);
            assert!(matches!(
                Color::new(ColorInput {
                    illuminant_hash: 0,
                    ..input
                }),
                Err(ColorError::Illuminant)
            ));
            std::thread::spawn(move || assert!(Color::new(input).is_ok()))
                .join()
                .unwrap();
            assert!(Color::new(input).is_ok());
            assert!(matches!(Color::new(input), Err(found) if found == error));
            assert!(Color::new(input).is_ok());
            testing::clear_color();
            case.passed();
        }
        cases.finish();
    }
}
