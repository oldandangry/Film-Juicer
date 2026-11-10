use film_juicer_core::color::{self, Whites};
#[path = "support/case_results.rs"]
mod test_cases;
use test_cases::RowCases;
use serde_json::Value;

fn fixture(logical_name: &str, manifest_name: &str) -> Value {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/ffi/fixtures/color");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join(manifest_name)).unwrap()).unwrap();
    let payload = manifest["payloads"][logical_name].as_str().unwrap();
    serde_json::from_str(&std::fs::read_to_string(root.join(payload)).unwrap()).unwrap()
}

fn triplet(value: &Value) -> [f32; 3] {
    std::array::from_fn(|i| f32::from_bits(value[i].as_u64().unwrap() as u32))
}

fn case_ids(rows: &[Value], field: &str) -> Vec<String> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            row[field]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("row/{index}"))
        })
        .collect()
}

#[test]
fn native_parent_scalar_and_matrix_bits() {
    let fixture = fixture("linux-debug.json", "manifest.json");
    let fixture_rows = fixture["leaf"].as_array().unwrap();
    let ids = case_ids(fixture_rows, "id");
    let mut cases = RowCases::new("native-parent-color-leaves", &ids);
    for (row, id) in fixture_rows.iter().zip(&ids) {
        let case = cases.case(id);
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
        case.passed();
    }
    cases.finish();
}

#[test]
fn cat02_native_parent_scalar_and_matrix_bits() {
    let preset = match (cfg!(target_os = "windows"), cfg!(debug_assertions)) {
        (false, true) => "linux-debug",
        (false, false) => "linux-release",
        (true, true) => "windows-clang-debug",
        (true, false) => "windows-clang-release",
    };
    let fixture = fixture(&format!("cat02-{preset}.json"), "cat02-manifest.json");
    let fixture_rows = fixture["leaf"].as_array().unwrap();
    let ids = case_ids(fixture_rows, "id");
    let mut cases = RowCases::new("cat02-native-parent-color-leaves", &ids);
    for (row, id) in fixture_rows.iter().zip(&ids) {
        let case = cases.case(id);
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
        case.passed();
    }
    cases.finish();
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
    let fixture = fixture(
        &format!("input-color-{preset}.json"),
        "input-color-manifest.json",
    );
    // Owner-adopted 2026-10-06 mapping: the producer uses completed native A;
    // standalone/default observations are characterization. Explicit operands,
    // including Windows Release B conversion matrices, remain unchanged.
    let inverse = &fixture["recipes"][0]["expected"]["xyz_to_linear_srgb"];
    let recipe_rows = fixture["recipes"].as_array().unwrap();
    let recipe_ids = case_ids(recipe_rows, "id");
    let mut recipes = RowCases::new("input-color-recipes", &recipe_ids);
    for (row, id) in recipe_rows.iter().zip(&recipe_ids) {
        let case = recipes.case(id);
        assert_eq!(&row["expected"]["xyz_to_linear_srgb"], inverse);
        assert_eq!(&row["expected"]["config_xyz_to_linear_srgb"], inverse);
        case.passed();
    }
    recipes.finish();
    let matrix_rows = fixture["matrices"].as_array().unwrap();
    let matrix_ids: Vec<String> = matrix_rows
        .iter()
        .map(|row| format!("space/{}", row["space"].as_u64().unwrap()))
        .collect();
    let mut matrices_rows = RowCases::new("input-color-matrices", &matrix_ids);
    for (row, id) in matrix_rows.iter().zip(&matrix_ids) {
        let case = matrices_rows.case(id);
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
        case.passed();
    }
    matrices_rows.finish();
    let decode_rows = fixture["decode"].as_array().unwrap();
    let decode_ids = case_ids(decode_rows, "id");
    let mut decodes = RowCases::new("input-color-decode", &decode_ids);
    for (row, id) in decode_rows.iter().zip(&decode_ids) {
        let case = decodes.case(id);
        let actual = color::decode_input(
            input_space(row["space"].as_u64().unwrap()),
            row["decode"].as_u64().unwrap() != 0,
            triplet(&row["rgb"]),
        );
        expect_bits(&actual, &row["expected"]["linear"], &row["id"], "decode");
        case.passed();
    }
    decodes.finish();
    let conversion_rows = fixture["conversion"].as_array().unwrap();
    let conversion_ids = case_ids(conversion_rows, "id");
    let mut conversions = RowCases::new("input-color-conversion", &conversion_ids);
    for (row, id) in conversion_rows.iter().zip(&conversion_ids) {
        let case = conversions.case(id);
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
        case.passed();
    }
    conversions.finish();
    let linear_rows = fixture["linear"].as_array().unwrap();
    let linear_ids = case_ids(linear_rows, "id");
    let mut linear_cases = RowCases::new("input-color-linear-leaves", &linear_ids);
    for (row, id) in linear_rows.iter().zip(&linear_ids) {
        let case = linear_cases.case(id);
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
        case.passed();
    }
    linear_cases.finish();
    let projection_rows = fixture["projection"].as_array().unwrap();
    let projection_ids = case_ids(projection_rows, "id");
    let mut projections = RowCases::new("input-color-projection", &projection_ids);
    for (row, id) in projection_rows.iter().zip(&projection_ids) {
        let case = projections.case(id);
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
        case.passed();
    }
    projections.finish();
}
