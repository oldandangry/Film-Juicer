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
