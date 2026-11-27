use film_juicer_core::color::{self, Whites};
use serde_json::Value;

#[path = "support/capture.rs"]
mod capture;

fn fixture(logical_name: &str, manifest_name: &str) -> Value {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/color");
    let manifest = capture::read(&root, manifest_name);
    capture::read_parts(&root, &manifest["payloads"][logical_name])
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
    let cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "native-parent-color-leaves"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "native-parent-color-leaves"
    );
    for (row, id) in fixture_rows.iter().zip(&ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
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
    let cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "cat02-native-parent-color-leaves"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "cat02-native-parent-color-leaves"
    );
    for (row, id) in fixture_rows.iter().zip(&ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
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
    let recipes_ids: Vec<_> = IntoIterator::into_iter(&recipe_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !recipes_ids.is_empty(),
        "empty case set: {}",
        "input-color-recipes"
    );
    let recipes_count = recipes_ids.len();
    let mut recipes_remaining = recipes_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        recipes_remaining.len(),
        recipes_count,
        "duplicate case IDs: {}",
        "input-color-recipes"
    );
    for (row, id) in recipe_rows.iter().zip(&recipe_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        assert_eq!(&row["expected"]["xyz_to_linear_srgb"], inverse);
        assert_eq!(&row["expected"]["config_xyz_to_linear_srgb"], inverse);
        assert!(
            recipes_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        recipes_remaining.is_empty(),
        "missing comparisons: {recipes_remaining:?}"
    );
    let matrix_rows = fixture["matrices"].as_array().unwrap();
    let matrix_ids: Vec<String> = matrix_rows
        .iter()
        .map(|row| format!("space/{}", row["space"].as_u64().unwrap()))
        .collect();
    let matrices_rows_ids: Vec<_> = IntoIterator::into_iter(&matrix_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !matrices_rows_ids.is_empty(),
        "empty case set: {}",
        "input-color-matrices"
    );
    let matrices_rows_count = matrices_rows_ids.len();
    let mut matrices_rows_remaining = matrices_rows_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        matrices_rows_remaining.len(),
        matrices_rows_count,
        "duplicate case IDs: {}",
        "input-color-matrices"
    );
    for (row, id) in matrix_rows.iter().zip(&matrix_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            matrices_rows_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        matrices_rows_remaining.is_empty(),
        "missing comparisons: {matrices_rows_remaining:?}"
    );
    let decode_rows = fixture["decode"].as_array().unwrap();
    let decode_ids = case_ids(decode_rows, "id");
    let decodes_ids: Vec<_> = IntoIterator::into_iter(&decode_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !decodes_ids.is_empty(),
        "empty case set: {}",
        "input-color-decode"
    );
    let decodes_count = decodes_ids.len();
    let mut decodes_remaining = decodes_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        decodes_remaining.len(),
        decodes_count,
        "duplicate case IDs: {}",
        "input-color-decode"
    );
    for (row, id) in decode_rows.iter().zip(&decode_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        let actual = color::decode_input(
            input_space(row["space"].as_u64().unwrap()),
            row["decode"].as_u64().unwrap() != 0,
            triplet(&row["rgb"]),
        );
        expect_bits(&actual, &row["expected"]["linear"], &row["id"], "decode");
        assert!(
            decodes_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        decodes_remaining.is_empty(),
        "missing comparisons: {decodes_remaining:?}"
    );
    let conversion_rows = fixture["conversion"].as_array().unwrap();
    let conversion_ids = case_ids(conversion_rows, "id");
    let conversions_ids: Vec<_> = IntoIterator::into_iter(&conversion_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !conversions_ids.is_empty(),
        "empty case set: {}",
        "input-color-conversion"
    );
    let conversions_count = conversions_ids.len();
    let mut conversions_remaining = conversions_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        conversions_remaining.len(),
        conversions_count,
        "duplicate case IDs: {}",
        "input-color-conversion"
    );
    for (row, id) in conversion_rows.iter().zip(&conversion_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            conversions_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        conversions_remaining.is_empty(),
        "missing comparisons: {conversions_remaining:?}"
    );
    let linear_rows = fixture["linear"].as_array().unwrap();
    let linear_ids = case_ids(linear_rows, "id");
    let linear_cases_ids: Vec<_> = IntoIterator::into_iter(&linear_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !linear_cases_ids.is_empty(),
        "empty case set: {}",
        "input-color-linear-leaves"
    );
    let linear_cases_count = linear_cases_ids.len();
    let mut linear_cases_remaining = linear_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        linear_cases_remaining.len(),
        linear_cases_count,
        "duplicate case IDs: {}",
        "input-color-linear-leaves"
    );
    for (row, id) in linear_rows.iter().zip(&linear_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            linear_cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        linear_cases_remaining.is_empty(),
        "missing comparisons: {linear_cases_remaining:?}"
    );
    let projection_rows = fixture["projection"].as_array().unwrap();
    let projection_ids = case_ids(projection_rows, "id");
    let projections_ids: Vec<_> = IntoIterator::into_iter(&projection_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !projections_ids.is_empty(),
        "empty case set: {}",
        "input-color-projection"
    );
    let projections_count = projections_ids.len();
    let mut projections_remaining = projections_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        projections_remaining.len(),
        projections_count,
        "duplicate case IDs: {}",
        "input-color-projection"
    );
    for (row, id) in projection_rows.iter().zip(&projection_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            projections_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        projections_remaining.is_empty(),
        "missing comparisons: {projections_remaining:?}"
    );
}
