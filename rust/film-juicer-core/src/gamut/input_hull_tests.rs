use super::{self as gamut, fixtures};
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
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/input-hull")
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

fn ray_case_id(row: &Value, include_case: bool) -> String {
    let domain = row["domain"].as_u64().unwrap();
    let origin_x = row["origin"][0].as_u64().unwrap();
    let origin_y = row["origin"][1].as_u64().unwrap();
    let direction_x = row["direction"][0].as_u64().unwrap();
    let direction_y = row["direction"][1].as_u64().unwrap();
    let coordinates = format!(
        "domain-{domain}/origin-{origin_x:016x}-{origin_y:016x}/direction-{direction_x:016x}-{direction_y:016x}"
    );
    if include_case {
        format!("{}/{coordinates}", row["case"].as_str().unwrap())
    } else {
        coordinates
    }
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
    let math_rows: Vec<_> = native["math"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| ![4, 5, 6].contains(&row["id"].as_u64().unwrap()))
        .collect();
    let ids: Vec<String> = math_rows
        .iter()
        .map(|row| format!("math/{}", row["id"].as_u64().unwrap()))
        .collect();
    let row_cases_ids: Vec<_> = IntoIterator::into_iter(&ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !row_cases_ids.is_empty(),
        "empty case set: {}",
        "input-hull-native-math-cases"
    );
    let row_cases_count = row_cases_ids.len();
    let mut row_cases_remaining = row_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        row_cases_remaining.len(),
        row_cases_count,
        "duplicate case IDs: {}",
        "input-hull-native-math-cases"
    );
    for (row, id) in math_rows.iter().zip(&ids) {
        let case_row = AsRef::<str>::as_ref(&(id)).to_owned();
        let failures_before = failures.len();
        let id = row["id"].as_u64().unwrap();
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
        assert_eq!(failures.len(), failures_before, "input-hull math row {id}");
        assert!(
            row_cases_remaining.remove(&case_row),
            "unexpected or repeated comparison: {case_row}"
        );
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
    let native_ray_rows = native["rays"].as_array().unwrap();
    let native_ray_ids: Vec<String> = native_ray_rows
        .iter()
        .map(|row| ray_case_id(row, false))
        .collect();
    let native_ray_cases_ids: Vec<_> = IntoIterator::into_iter(&native_ray_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !native_ray_cases_ids.is_empty(),
        "empty case set: {}",
        "native-ray-intersections"
    );
    let native_ray_cases_count = native_ray_cases_ids.len();
    let mut native_ray_cases_remaining = native_ray_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        native_ray_cases_remaining.len(),
        native_ray_cases_count,
        "duplicate case IDs: {}",
        "native-ray-intersections"
    );
    for (i, (row, id)) in native_ray_rows.iter().zip(&native_ray_ids).enumerate() {
        let row_case = AsRef::<str>::as_ref(&(id)).to_owned();
        let failures_before = failures.len();
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
        if failures.len() == failures_before {
            assert!(
                native_ray_cases_remaining.remove(&row_case),
                "unexpected or repeated comparison: {row_case}"
            );
        }
    }
    square[..5].copy_from_slice(&[[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
    let mut narrow = [[0_f32; 2]; 1025];
    for i in 0..5 {
        narrow[i] = square[i].map(|v| v as f32);
    }
    let supplement_ray_rows = supplement["rays"].as_array().unwrap();
    let supplement_ray_ids: Vec<String> = supplement_ray_rows
        .iter()
        .map(|row| ray_case_id(row, true))
        .collect();
    let threshold_ray_cases_ids: Vec<_> = IntoIterator::into_iter(&supplement_ray_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !threshold_ray_cases_ids.is_empty(),
        "empty case set: {}",
        "threshold-ray-intersections"
    );
    let threshold_ray_cases_count = threshold_ray_cases_ids.len();
    let mut threshold_ray_cases_remaining = threshold_ray_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        threshold_ray_cases_remaining.len(),
        threshold_ray_cases_count,
        "duplicate case IDs: {}",
        "threshold-ray-intersections"
    );
    for (i, (row, id)) in supplement_ray_rows
        .iter()
        .zip(&supplement_ray_ids)
        .enumerate()
    {
        let row_case = AsRef::<str>::as_ref(&(id)).to_owned();
        let failures_before = failures.len();
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
        if failures.len() == failures_before {
            assert!(
                threshold_ray_cases_remaining.remove(&row_case),
                "unexpected or repeated comparison: {row_case}"
            );
        }
    }
    assert_eq!(cases, 11);
    assert_eq!(count, 45850);
    assert!(
        failures.is_empty(),
        "{} mismatches: {:?}",
        failures.len(),
        &failures[..failures.len().min(20)]
    );
    assert!(
        native_ray_cases_remaining.is_empty(),
        "missing comparisons: {native_ray_cases_remaining:?}"
    );
    assert!(
        threshold_ray_cases_remaining.is_empty(),
        "missing comparisons: {threshold_ray_cases_remaining:?}"
    );
    assert!(
        row_cases_remaining.is_empty(),
        "missing comparisons: {row_cases_remaining:?}"
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
    let cases_ids: Vec<_> = IntoIterator::into_iter(["trigonometry", "exponentials"])
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "input-hull-native-arrays"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "input-hull-native-arrays"
    );
    let row = AsRef::<str>::as_ref(&("trigonometry")).to_owned();
    let mut visited = 0;
    assert_eq!(data.len(), 519536 * 32);
    for (index, record) in data.as_chunks::<32>().0.iter().enumerate() {
        visited += 1;
        let word = |i| u64::from_le_bytes(record[i..i + 8].try_into().unwrap());
        let value = f64::from_bits(word(0));
        assert_eq!(value.cos().to_bits(), word(8), "cos {index}");
        assert_eq!(value.sin().to_bits(), word(16), "sin {index}");
    }
    assert_eq!(visited, 519536);
    assert!(
        cases_remaining.remove(&row),
        "unexpected or repeated comparison: {row}"
    );
    let row = AsRef::<str>::as_ref(&("exponentials")).to_owned();
    let mut visited = 0;
    for row in native["weights"].as_array().unwrap() {
        visited += 1;
        let argument = f64::from_bits(row["argument"].as_u64().unwrap());
        assert_eq!(argument.exp().to_bits(), row["result"].as_u64().unwrap());
    }
    assert_eq!(visited, 1024);
    assert!(
        cases_remaining.remove(&row),
        "unexpected or repeated comparison: {row}"
    );
    assert!(
        cases_remaining.is_empty(),
        "missing comparisons: {cases_remaining:?}"
    );
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

#[test]
fn ray_order_and_nonfinite_boundaries() {
    // Horizontal connectors stay above or below the ray; the remaining points
    // repeat the final vertex so the fixed-size tails add only degenerate edges.
    let vertices = [
        [4.0, -1.0],
        [4.0, 1.0],
        [2.0, 1.0],
        [2.0, -1.0],
        [6.0, -1.0],
        [6.0, 1.0],
        [2.0, 1.0],
        [2.0, -1.0],
    ];
    let mut locus = [[2.0_f64, -1.0]; 66];
    locus[..vertices.len()].copy_from_slice(&vertices);
    let mut hull = [[2.0_f32, -1.0]; 1025];
    for (dst, src) in hull.iter_mut().zip(vertices) {
        *dst = src.map(|value| value as f32);
    }
    let degenerate_locus = [[2.0_f64, -1.0]; 66];
    let degenerate_hull = [[2.0_f32, -1.0]; 1025];
    let cases = [
        ("ordered-nearest-tie", [0.0, 0.0], [1.0, 0.0], 2.0),
        ("signed-zero-forward", [0.0, 0.0], [1.0, -0.0], 2.0),
        ("backward", [0.0, 0.0], [-1.0, 0.0], f64::INFINITY),
        ("parallel-outside", [0.0, 2.0], [1.0, 0.0], f64::INFINITY),
        (
            "signed-zero-vertical",
            [0.0, 0.0],
            [-0.0, 1.0],
            f64::INFINITY,
        ),
        ("origin-x-nan", [f64::NAN, 0.0], [1.0, 0.0], f64::INFINITY),
        ("origin-y-nan", [0.0, f64::NAN], [1.0, 0.0], f64::INFINITY),
        (
            "origin-positive-infinity",
            [f64::INFINITY, 0.0],
            [1.0, 0.0],
            f64::INFINITY,
        ),
        (
            "origin-negative-infinity",
            [f64::NEG_INFINITY, 0.0],
            [1.0, 0.0],
            f64::INFINITY,
        ),
        (
            "origin-y-positive-infinity",
            [0.0, f64::INFINITY],
            [1.0, 0.0],
            f64::INFINITY,
        ),
        (
            "origin-y-negative-infinity",
            [0.0, f64::NEG_INFINITY],
            [1.0, 0.0],
            f64::INFINITY,
        ),
        (
            "direction-x-nan",
            [0.0, 0.0],
            [f64::NAN, 0.0],
            f64::INFINITY,
        ),
        (
            "direction-y-nan",
            [0.0, 0.0],
            [1.0, f64::NAN],
            f64::INFINITY,
        ),
        (
            "direction-positive-infinity",
            [0.0, 0.0],
            [f64::INFINITY, 0.0],
            f64::INFINITY,
        ),
        (
            "direction-negative-infinity",
            [0.0, 0.0],
            [f64::NEG_INFINITY, 0.0],
            f64::INFINITY,
        ),
        (
            "direction-y-positive-infinity",
            [0.0, 0.0],
            [1.0, f64::INFINITY],
            f64::INFINITY,
        ),
        (
            "direction-y-negative-infinity",
            [0.0, 0.0],
            [1.0, f64::NEG_INFINITY],
            f64::INFINITY,
        ),
    ];
    let ids: Vec<_> = cases
        .iter()
        .map(|case| case.0)
        .chain(["degenerate", "zero-polygon"])
        .collect();
    let rows_ids: Vec<_> = IntoIterator::into_iter(ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !rows_ids.is_empty(),
        "empty case set: {}",
        "input-hull-ray-order-boundaries"
    );
    let rows_count = rows_ids.len();
    let mut rows_remaining = rows_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        rows_remaining.len(),
        rows_count,
        "duplicate case IDs: {}",
        "input-hull-ray-order-boundaries"
    );
    for (id, origin, direction, expected) in cases {
        let row = AsRef::<str>::as_ref(&(id)).to_owned();
        assert_eq!(
            fixtures::locus_ray(origin, direction, &locus).to_bits(),
            expected.to_bits(),
            "locus {id}"
        );
        assert_eq!(
            fixtures::hull_ray(origin, direction, &hull).to_bits(),
            expected.to_bits(),
            "hull {id}"
        );
        assert!(
            rows_remaining.remove(&row),
            "unexpected or repeated comparison: {row}"
        );
    }
    let row = AsRef::<str>::as_ref(&("degenerate")).to_owned();
    assert_eq!(
        fixtures::locus_ray([0.0, 0.0], [1.0, 0.0], &degenerate_locus).to_bits(),
        f64::INFINITY.to_bits()
    );
    assert_eq!(
        fixtures::hull_ray([0.0, 0.0], [1.0, 0.0], &degenerate_hull).to_bits(),
        f64::INFINITY.to_bits()
    );
    assert!(
        rows_remaining.remove(&row),
        "unexpected or repeated comparison: {row}"
    );
    let row = AsRef::<str>::as_ref(&("zero-polygon")).to_owned();
    assert_eq!(
        fixtures::locus_ray([0.0, 0.0], [1.0, 0.0], &[[0.0; 2]; 66]).to_bits(),
        f64::INFINITY.to_bits()
    );
    assert_eq!(
        fixtures::hull_ray([0.0, 0.0], [1.0, 0.0], &[[0.0; 2]; 1025]).to_bits(),
        f64::INFINITY.to_bits()
    );
    assert!(
        rows_remaining.remove(&row),
        "unexpected or repeated comparison: {row}"
    );
    assert!(
        rows_remaining.is_empty(),
        "missing comparisons: {rows_remaining:?}"
    );
}
