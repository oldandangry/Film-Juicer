use film_juicer_core::color::{self, Whites};
use serde_json::Value;

fn triplet(value: &Value) -> [f32; 3] {
    std::array::from_fn(|i| f32::from_bits(value[i].as_u64().unwrap() as u32))
}

#[test]
fn native_parent_scalar_and_matrix_bits() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/ffi/fixtures/color/linux-debug.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for row in fixture["leaf"].as_array().unwrap() {
        let whites = Whites {
            source_xyz: triplet(&row["source"]),
            destination_xyz: triplet(&row["destination"]),
        };
        let matrix = color::cat16_matrix(whites);
        let adapted = color::adapt_cat16(triplet(&row["xyz"]), whites);
        for (actual, field) in [
            (matrix.as_slice(), "matrix"),
            (adapted.as_slice(), "adapted"),
        ] {
            for (i, value) in actual.iter().enumerate() {
                let expected = row["expected"][field][i].as_u64().unwrap() as u32;
                if f32::from_bits(expected).is_nan() {
                    assert!(value.is_nan(), "{} {field}[{i}]", row["id"]);
                } else {
                    assert_eq!(value.to_bits(), expected, "{} {field}[{i}]", row["id"]);
                }
            }
        }
    }
}

#[test]
fn cat02_native_parent_scalar_and_matrix_bits() {
    let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
        (false, true) => "linux-debug",
        (false, false) => "linux-release",
        (true, true) => "windows-clang-debug",
        (true, false) => "windows-clang-release",
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../tests/ffi/fixtures/color/cat02-{preset}.json"
    ));
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for row in fixture["leaf"].as_array().unwrap() {
        let whites = Whites {
            source_xyz: triplet(&row["source"]),
            destination_xyz: triplet(&row["destination"]),
        };
        let matrix = color::cat02_matrix(whites);
        let adapted = color::adapt_cat02(triplet(&row["xyz"]), whites);
        for (actual, field) in [
            (matrix.as_slice(), "matrix"),
            (adapted.as_slice(), "adapted"),
        ] {
            for (i, value) in actual.iter().enumerate() {
                let expected = row["expected"][field][i].as_u64().unwrap() as u32;
                if f32::from_bits(expected).is_nan() {
                    assert!(value.is_nan(), "{} {field}[{i}]", row["id"]);
                } else {
                    assert_eq!(value.to_bits(), expected, "{} {field}[{i}]", row["id"]);
                }
            }
        }
    }
}

fn input_space(tag: u64) -> color::InputSpace {
    match tag {
        0 => color::InputSpace::DavinciWideGamut,
        1 => color::InputSpace::Bt2020,
        2 => color::InputSpace::Aces2065_1,
        3 => color::InputSpace::SrgbRec709,
        _ => panic!("unknown fixture input space"),
    }
}

fn matrix(value: &Value) -> [f32; 9] {
    std::array::from_fn(|i| f32::from_bits(value[i].as_u64().unwrap() as u32))
}

fn expect_bits(actual: &[f32], expected: &Value, id: &Value, field: &str) {
    assert_eq!(actual.len(), expected.as_array().unwrap().len());
    for (i, value) in actual.iter().enumerate() {
        let bits = expected[i].as_u64().unwrap() as u32;
        if f32::from_bits(bits).is_nan() {
            assert!(value.is_nan(), "{id} {field}[{i}]");
        } else {
            assert_eq!(value.to_bits(), bits, "{id} {field}[{i}]");
        }
    }
}

#[test]
fn input_color_native_parent_bits() {
    let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
        (false, true) => "linux-debug",
        (false, false) => "linux-release",
        (true, true) => "windows-clang-debug",
        (true, false) => "windows-clang-release",
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../tests/ffi/fixtures/color/input-color-{preset}.json"
    ));
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    // Owner-adopted 2026-10-06 mapping: the producer uses completed native A;
    // standalone/default observations are characterization. Explicit operands,
    // including Windows Release B conversion matrices, remain unchanged.
    let inverse = &fixture["recipes"][0]["expected"]["xyz_to_linear_srgb"];
    for row in fixture["recipes"].as_array().unwrap() {
        assert_eq!(&row["expected"]["xyz_to_linear_srgb"], inverse);
        assert_eq!(&row["expected"]["config_xyz_to_linear_srgb"], inverse);
    }
    for row in fixture["matrices"].as_array().unwrap() {
        let matrices = color::input_matrices(input_space(row["space"].as_u64().unwrap()));
        expect_bits(
            &matrices.rgb_to_xyz,
            &row["rgb_to_xyz"],
            &row["space"],
            "input matrix",
        );
        expect_bits(
            &matrices.nominal_white_xyz,
            &row["nominal_white"],
            &row["space"],
            "nominal white",
        );
        expect_bits(
            &matrices.d65_white_xyz,
            &row["d65_white"],
            &row["space"],
            "input D65",
        );
        expect_bits(
            &matrices.xyz_to_linear_srgb,
            inverse,
            &row["space"],
            "completed inverse",
        );
    }
    for row in fixture["decode"].as_array().unwrap() {
        let actual = color::decode_input(
            input_space(row["space"].as_u64().unwrap()),
            row["decode"].as_u64().unwrap() != 0,
            triplet(&row["rgb"]),
        );
        expect_bits(&actual, &row["expected"]["linear"], &row["id"], "decode");
    }
    for row in fixture["conversion"].as_array().unwrap() {
        let input = color::InputConversion {
            space: input_space(row["space"].as_u64().unwrap()),
            decode_cctf: row["decode"].as_u64().unwrap() != 0,
            rgb_to_xyz: matrix(&row["rgb_to_xyz"]),
            xyz_adaptation: row["adapt"]
                .as_bool()
                .unwrap()
                .then(|| matrix(&row["xyz_adapt"])),
        };
        let rgb = triplet(&row["rgb"]);
        for (clamp, prefix) in [(true, "dwg_clamped"), (false, "dwg_signed")] {
            let actual = color::input_to_dwg(input, rgb, clamp);
            expect_bits(
                &actual.rgb,
                &row["expected"][format!("{prefix}_rgb")],
                &row["id"],
                "DWG",
            );
            expect_bits(
                &actual.xyz,
                &row["expected"][format!("{prefix}_xyz")],
                &row["id"],
                "observed XYZ",
            );
        }
        let actual = color::input_to_linear_srgb(input, rgb, matrix(&row["xyz_to_linear_srgb"]));
        expect_bits(
            &actual.rgb,
            &row["expected"]["srgb_rgb"],
            &row["id"],
            "linear sRGB",
        );
        expect_bits(
            &actual.xyz,
            &row["expected"]["srgb_xyz"],
            &row["id"],
            "observed XYZ",
        );
    }
    for row in fixture["linear"].as_array().unwrap() {
        let rgb = triplet(&row["rgb"]);
        expect_bits(
            &color::linear_srgb_to_xyz(rgb),
            &row["expected"]["srgb_xyz"],
            &row["id"],
            "sRGB leaf",
        );
        expect_bits(
            &color::dwg_to_xyz(rgb),
            &row["expected"]["dwg_xyz"],
            &row["id"],
            "DWG leaf",
        );
    }
    for row in fixture["projection"].as_array().unwrap() {
        let actual = color::project_linear_rgb_to_xyz(
            triplet(&row["rgb"]),
            matrix(&row["rgb_to_xyz"]),
            matrix(&row["xyz_adapt"]),
        );
        expect_bits(
            &actual,
            &row["expected"]["projected_xyz"],
            &row["id"],
            "linear projection",
        );
    }
}
