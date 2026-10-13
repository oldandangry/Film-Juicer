#![cfg(feature = "test-support")]
use film_juicer_core::gamut::{self, fixtures};
use serde_json::Value;
use std::fs;

fn floats<const N: usize>(value: &Value) -> [f32; N] {
    std::array::from_fn(|i| f32::from_bits(value[i].as_u64().unwrap() as u32))
}
fn doubles<const N: usize>(value: &Value) -> [f64; N] {
    std::array::from_fn(|i| f64::from_bits(value[i].as_u64().unwrap()))
}
fn bits<const N: usize>(value: [f64; N]) -> Vec<u64> {
    value.map(f64::to_bits).to_vec()
}
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/ffi/fixtures/output_gamut")
}
fn load(name: &str) -> Value {
    serde_json::from_slice(&fs::read(root().join(name)).unwrap()).unwrap()
}
fn member() -> Value {
    let platform = if cfg!(target_os = "windows") {
        "windows-clang"
    } else {
        "linux"
    };
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    load("manifest.json")["preset_membership"][format!("{platform}-{profile}")].clone()
}
fn primitive(row: &Value) {
    let c = doubles(&row["coefficients"]);
    let (partitions, count, exit) = fixtures::output_primitive(&c);
    assert_eq!(
        bits(partitions),
        row["partitions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(count, row["count"].as_u64().unwrap() as usize);
    assert_eq!(
        exit.to_bits(),
        row["exit"].as_u64().unwrap(),
        "first exit {c:?}"
    );
    if let Some(evaluations) = row["evaluations"].as_array() {
        for e in evaluations {
            let x = f64::from_bits(e["chroma"].as_u64().unwrap());
            let v = fixtures::output_evaluation(&c, x);
            assert_eq!(v.to_bits(), e["value"].as_u64().unwrap());
            assert_eq!(fixtures::output_cube(v), e["inside"].as_bool().unwrap());
        }
        let a = 3.0 * c[3];
        let b = 2.0 * c[2];
        let cc = c[1];
        let discriminant = b * b - 4.0 * a * cc;
        let square_root = discriminant.sqrt();
        let signed_root = square_root.copysign(b);
        let q = -0.5 * (b + signed_root);
        assert_eq!(
            serde_json::json!(bits([a, b, cc, discriminant, square_root, signed_root, q])),
            row["derivative"]
        );
    }
}
#[test]
fn native_primitive_intermediate_full_family_bits() {
    let member = member();
    let records = load(member["records"].as_str().unwrap());
    let extra = load(member["extra"].as_str().unwrap());
    let tables = fs::read(root().join(member["tables"].as_str().unwrap())).unwrap();
    assert_eq!(
        serde_json::json!(fixtures::output_constants().map(|a| a.map(f32::to_bits))),
        records["oklab"]
    );
    for (i, direction) in records["directions"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            serde_json::json!(bits(fixtures::output_direction(i))),
            *direction,
            "hue libm {i}"
        );
    }
    for row in records["primitives"]
        .as_array()
        .unwrap()
        .iter()
        .chain(extra["primitives"].as_array().unwrap())
    {
        primitive(row);
    }
    for row in extra["cube"].as_array().unwrap() {
        assert_eq!(
            fixtures::output_cube(f64::from_bits(row["value"].as_u64().unwrap())),
            row["inside"].as_bool().unwrap()
        );
    }
    for row in records["rows"]
        .as_array()
        .unwrap()
        .iter()
        .chain(extra["rows"].as_array().unwrap())
    {
        let source = &row["source"];
        let rgb = floats(&source["rgb_to_xyz"]);
        let xyz = floats(&source["xyz_to_rgb"]);
        let white = floats(&source["white"]);
        let d65 = floats(&source["d65"]);
        let space = gamut::OutputSpace::try_from(row["tag"].as_u64().unwrap_or(0) as u32).unwrap();
        let built = gamut::output_transform(gamut::TransformInput {
            space,
            rgb_to_xyz: &rgb,
            xyz_to_rgb: &xyz,
            white_xyz: &white,
            d65_white_xyz: &d65,
        });
        assert_eq!(built.is_ok(), row["transform_ok"].as_bool().unwrap());
        let t = match built {
            Ok(t) => t,
            Err(e) => {
                assert_eq!(e.to_string(), row["diagnostic"].as_str().unwrap());
                continue;
            }
        };
        let expected = &row["transform"];
        assert_eq!(
            serde_json::json!(t.native_rgb_to_d65_xyz().map(f32::to_bits)),
            expected["entry"]
        );
        assert_eq!(
            serde_json::json!(t.d65_xyz_to_native_rgb().map(f32::to_bits)),
            expected["exit"]
        );
        assert_eq!(t.hash(), expected["hash"].as_u64().unwrap());
        assert_eq!(
            t.contract_hash(),
            expected["contract_hash"].as_u64().unwrap()
        );
        assert_eq!(t.space(), space);
        if !row["lms_to_native"].is_null() {
            assert_eq!(
                serde_json::json!(bits(fixtures::output_lms_matrix(t.d65_xyz_to_native_rgb()))),
                row["lms_to_native"]
            );
            for cell in row["cells"].as_array().unwrap() {
                let h = cell["h"].as_u64().unwrap() as usize;
                let l = cell["l"].as_u64().unwrap() as usize;
                let lightness = f64::from(0.02_f32)
                    + (f64::from(1.0_f32) - f64::from(0.02_f32)) * l as f64 / 63.0;
                assert_eq!(lightness.to_bits(), cell["lightness"].as_u64().unwrap());
                let d = fixtures::output_direction(h);
                let c = fixtures::output_cubics(t.d65_xyz_to_native_rgb(), lightness, [d[1], d[2]]);
                assert_eq!(
                    serde_json::json!(c.map(|a| a.map(f64::to_bits))),
                    cell["cubics"]
                );
            }
        }
        let boundary = gamut::build_output_boundary(gamut::BoundaryInput {
            space,
            d65_xyz_to_native_rgb: t.d65_xyz_to_native_rgb(),
            transform_hash: t.hash(),
        });
        assert_eq!(boundary.is_ok(), row["boundary_ok"].as_bool().unwrap());
        match boundary {
            Ok(table) => {
                assert_eq!(table.hash(), row["boundary_hash"].as_u64().unwrap());
                assert_eq!(table.transform_hash(), t.hash());
                assert_eq!(table.contract_hash(), t.contract_hash());
                if let Some(offset) = row["table_offset"].as_u64() {
                    for (i, &v) in table.cmax().iter().enumerate() {
                        let start = offset as usize + i * 4;
                        let word = u32::from_le_bytes(tables[start..start + 4].try_into().unwrap());
                        assert_eq!(v.to_bits(), word, "Cmax {space:?} word {i}");
                    }
                }
            }
            Err(e) => assert_eq!(e.to_string(), row["boundary_diagnostic"].as_str().unwrap()),
        }
    }
    assert_eq!(
        fixtures::output_reserve_failure(),
        gamut::BoundaryError::AllocationFailure
    );
    println!(
        "A9 primitive/intermediate/libm/all-nine/full-table/identity/native failures exact; actual capacity-overflow reserve gate passed"
    );
}
