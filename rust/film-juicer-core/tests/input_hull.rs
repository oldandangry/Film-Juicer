#![cfg(feature = "test-support")]
use film_juicer_core::gamut::{self, fixtures};
use serde_json::{json, Value};
use std::fs;

fn pair(v: &Value) -> [f64; 2] {
    std::array::from_fn(|i| f64::from_bits(v[i].as_u64().unwrap()))
}
fn check(
    path: &str,
    expected: &Value,
    actual: &Value,
    failures: &mut Vec<Value>,
    count: &mut usize,
) {
    match (expected, actual) {
        (Value::Array(e), Value::Array(a)) if e.len() == a.len() => {
            for (i, (e, a)) in e.iter().zip(a).enumerate() {
                check(&format!("{path}/{i}"), e, a, failures, count);
            }
        }
        (Value::Object(e), Value::Object(a)) if e.len() == a.len() => {
            for (key, value) in e {
                check(&format!("{path}/{key}"), value, &a[key], failures, count);
            }
        }
        _ => {
            *count += 1;
            if expected != actual {
                failures.push(json!({"path":path,"expected":expected,"actual":actual}));
            }
        }
    }
}
fn double_array<const N: usize>(v: &[f64; N]) -> Value {
    json!(v.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
}
fn double_pairs<const N: usize>(v: &[[f64; 2]; N]) -> Value {
    json!(v.iter().map(|v| v.map(f64::to_bits)).collect::<Vec<_>>())
}
fn fixture_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/ffi/fixtures/input_hull")
}
fn membership() -> Value {
    let manifest: Value =
        serde_json::from_slice(&fs::read(fixture_root().join("manifest.json")).unwrap()).unwrap();
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
    manifest["preset_membership"][format!("{platform}-{profile}")].clone()
}
#[test]
fn native_ordered_math_and_shared_ray_bits() {
    let member = membership();
    let native: Value = serde_json::from_slice(
        &fs::read(fixture_root().join(member["records"].as_str().unwrap())).unwrap(),
    )
    .unwrap();
    let supplement: Value = serde_json::from_slice(
        &fs::read(fixture_root().join(member["rays"].as_str().unwrap())).unwrap(),
    )
    .unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    let mut cases = 0;
    for row in native["math"].as_array().unwrap() {
        let id = row["id"].as_u64().unwrap();
        // Native vector/axis structure admission precedes fixed core input.
        if [4, 5, 6].contains(&id) {
            continue;
        }
        let sources: [[f32; 81]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|i| {
                f32::from_bits(row["sources"][c]["values"][i].as_u64().unwrap() as u32)
            })
        });
        let input = gamut::HullInput {
            x: &sources[0],
            y: &sources[1],
            z: &sources[2],
            d65: &sources[3],
        };
        let (built, trace) = fixtures::capture(input);
        let mut actual = json!({});
        if let Some((locus, center)) = trace.locus_center {
            actual["locus"] = double_pairs(&locus);
            actual["center"] = double_array(&center);
        }
        if let Some(reach) = trace.reach {
            actual["reach"] = double_array(&reach);
            actual["forward"] = double_pairs(&trace.forward);
            actual["coefficients"] = double_pairs(&trace.coefficients);
            actual["inverse"] = double_pairs(&trace.inverse);
            actual["smoothed"] = double_array(&trace.smoothed.unwrap());
            actual["floored"] = double_array(&trace.floored.unwrap());
            actual["scale"] = json!(trace.scale.unwrap().to_bits());
        }
        check(
            &format!("math/{id}/trace"),
            &row["trace"],
            &actual,
            &mut failures,
            &mut count,
        );
        check(
            &format!("math/{id}/built"),
            &row["built"],
            &json!(built.is_ok()),
            &mut failures,
            &mut count,
        );
        let expected = &row["expected"];
        match built {
            Ok(hull) => {
                let result = json!({"center":hull.center().map(f32::to_bits),"xy":hull.xy().iter().map(|p|p.map(f32::to_bits)).collect::<Vec<_>>(),"hash":hull.hash(),"valid":true,"diagnostic":""});
                check(
                    &format!("math/{id}/completed"),
                    expected,
                    &result,
                    &mut failures,
                    &mut count,
                );
                let direct = gamut::build_input_hull(input).unwrap();
                assert_eq!(direct.hash(), hull.hash());
                assert_eq!(
                    direct.center().map(f32::to_bits),
                    hull.center().map(f32::to_bits)
                );
                assert_eq!(
                    direct.xy().map(|p| p.map(f32::to_bits)),
                    hull.xy().map(|p| p.map(f32::to_bits))
                );
            }
            Err(error) => check(
                &format!("math/{id}/diagnostic"),
                &expected["diagnostic"],
                &json!(error.to_string()),
                &mut failures,
                &mut count,
            ),
        }
        cases += 1;
    }
    let locus: [[f64; 2]; 66] =
        std::array::from_fn(|i| pair(&native["math"][0]["trace"]["locus"][i]));
    let hull: [[f32; 2]; 1025] = std::array::from_fn(|i| {
        std::array::from_fn(|c| {
            f32::from_bits(native["math"][0]["expected"]["xy"][i][c].as_u64().unwrap() as u32)
        })
    });
    let mut square = [[0.; 2]; 66];
    square[..5].copy_from_slice(&[[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]]);
    for (i, row) in native["rays"].as_array().unwrap().iter().enumerate() {
        let center = pair(&row["origin"]);
        let direction = pair(&row["direction"]);
        let result = match row["domain"].as_u64().unwrap() {
            0 => fixtures::locus_ray(center, direction, &locus),
            1 => fixtures::hull_ray(center, direction, &hull),
            2 => fixtures::locus_ray(center, direction, &square),
            _ => unreachable!(),
        };
        check(
            &format!("rays/{i}"),
            &row["result"],
            &json!(result.to_bits()),
            &mut failures,
            &mut count,
        );
    }
    square[..5].copy_from_slice(&[[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
    let mut narrow = [[0_f32; 2]; 1025];
    for i in 0..5 {
        narrow[i] = square[i].map(|v| v as f32);
    }
    for (i, row) in supplement["rays"].as_array().unwrap().iter().enumerate() {
        let center = pair(&row["origin"]);
        let direction = pair(&row["direction"]);
        let result = if row["domain"] == 0 {
            fixtures::locus_ray(center, direction, &square)
        } else {
            fixtures::hull_ray(center, direction, &narrow)
        };
        check(
            &format!("threshold-rays/{i}"),
            &row["result"],
            &json!(result.to_bits()),
            &mut failures,
            &mut count,
        );
    }
    assert_eq!(cases, 11);
    assert_eq!(count, 45850);
    assert!(
        failures.is_empty(),
        "{} mismatches: {:?}",
        failures.len(),
        &failures[..failures.len().min(20)]
    );
}

#[test]
fn native_primitive_context_bits() {
    let member = membership();
    let native: Value = serde_json::from_slice(
        &fs::read(fixture_root().join(member["records"].as_str().unwrap())).unwrap(),
    )
    .unwrap();
    let data = fs::read(fixture_root().join(member["primitives"].as_str().unwrap())).unwrap();
    assert_eq!(data.len(), 519536 * 32);
    for (index, record) in data.as_chunks::<32>().0.iter().enumerate() {
        let word = |i| u64::from_le_bytes(record[i..i + 8].try_into().unwrap());
        let value = f64::from_bits(word(0));
        assert_eq!(value.cos().to_bits(), word(8), "cos {index}");
        assert_eq!(value.sin().to_bits(), word(16), "sin {index}");
    }
    for row in native["weights"].as_array().unwrap() {
        let argument = f64::from_bits(row["argument"].as_u64().unwrap());
        assert_eq!(argument.exp().to_bits(), row["result"].as_u64().unwrap());
    }
}

#[test]
fn completed_hull_survives_source_expiry_and_thread_transfer() {
    let hull = {
        let member = membership();
        let native: Value = serde_json::from_slice(
            &fs::read(fixture_root().join(member["records"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let sources: [[f32; 81]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|i| {
                f32::from_bits(
                    native["math"][0]["sources"][c]["values"][i]
                        .as_u64()
                        .unwrap() as u32,
                )
            })
        });
        gamut::build_input_hull(gamut::HullInput {
            x: &sources[0],
            y: &sources[1],
            z: &sources[2],
            d65: &sources[3],
        })
        .unwrap()
    };
    std::thread::scope(|scope| {
        scope.spawn(|| assert_eq!(hull.hash(), 0x5cda57333bb9b653));
    });
    std::thread::spawn(move || {
        assert_eq!(
            hull.xy()[0].map(f32::to_bits),
            hull.xy()[1024].map(f32::to_bits)
        )
    })
    .join()
    .unwrap();
}
