use film_juicer_core::illuminant::{self, ErrorKind};
use serde_json::Value;

fn words(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(x.as_u64().unwrap() as u32))
        .collect()
}
fn rows(value: &Value) -> Vec<[f32; 2]> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            [
                f32::from_bits(row[0].as_u64().unwrap() as u32),
                f32::from_bits(row[1].as_u64().unwrap() as u32),
            ]
        })
        .collect()
}
fn expect(actual: &[f32], expected: &Value, context: &str) {
    let expected = words(expected);
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        if b.is_nan() {
            assert!(a.is_nan(), "{context}[{i}]");
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "{context}[{i}]");
        }
    }
}
fn fixture() -> Value {
    let platform = if cfg!(windows) {
        "windows-clang"
    } else {
        "linux"
    };
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../tests/ffi/fixtures/illuminants/construction-{platform}-{profile}.json"
    ));
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
#[test]
fn accepted_native_complete_operations() {
    let capture = fixture();
    let leaf = &capture["leaf"];
    for row in leaf["blackbody"].as_array().unwrap() {
        expect(
            &illuminant::blackbody(f32::from_bits(row["temperature"].as_u64().unwrap() as u32)),
            &row["normalized"],
            "blackbody",
        );
    }
    expect(
        &illuminant::equal_energy(),
        &leaf["equal"]["samples"],
        "equal",
    );
    for row in leaf["direct"].as_array().unwrap() {
        let actual = illuminant::from_samples(&rows(&row["rows"]));
        if row["curve"]["samples"].as_array().unwrap().is_empty() {
            assert_eq!(actual.unwrap_err().kind, ErrorKind::Preparation);
        } else {
            expect(
                &actual.unwrap(),
                &row["curve"]["samples"],
                row["id"].as_str().unwrap(),
            );
        }
    }
    for row in leaf["resample"].as_array().unwrap() {
        let source = rows(&row["rows"]);
        let kg3 = illuminant::tungsten_kg3(&source);
        if row["kg3"]["samples"].as_array().unwrap().is_empty() {
            assert_eq!(kg3.unwrap_err().kind, ErrorKind::Preparation);
        } else {
            expect(&kg3.unwrap().0, &row["kg3"]["samples"], "KG3");
        }
        let lens = illuminant::prepare_lens(&source);
        assert_eq!(lens.is_ok(), row["lens_prepared"].as_bool().unwrap());
        if let Ok((lens, _)) = lens {
            expect(
                &illuminant::finish_lens(lens, &source).unwrap().0,
                &row["lens"]["samples"],
                "lens",
            );
        }
    }
}
#[cfg(feature = "test-support")]
#[test]
fn accepted_native_private_leaves() {
    let capture = fixture();
    let leaf = &capture["leaf"];
    for row in leaf["blackbody"].as_array().unwrap() {
        let temperature = f32::from_bits(row["temperature"].as_u64().unwrap() as u32);
        let raw: [f32; 81] = std::array::from_fn(|i| {
            illuminant::test_support::planck_sample(380.0 + 5.0 * i as f32, temperature)
        });
        expect(&raw, &row["raw"], "Planck");
    }
    for row in leaf["resample"].as_array().unwrap() {
        let result = illuminant::test_support::resample(&rows(&row["rows"]));
        if row["resampled"].as_array().unwrap().is_empty() {
            assert_eq!(result.unwrap_err().kind, ErrorKind::Preparation);
        } else {
            let expected: Value = rows(&row["resampled"])
                .iter()
                .map(|row| Value::from(row[1].to_bits()))
                .collect();
            expect(&result.unwrap(), &expected, "Akima");
        }
    }
    for row in leaf["normalization"].as_array().unwrap() {
        let mut samples = words(&row["input"]);
        illuminant::test_support::normalize(&mut samples);
        expect(&samples, &row["expected"], "normalization");
    }
    assert_eq!(
        illuminant::test_support::capacity_failure()
            .unwrap_err()
            .kind,
        ErrorKind::Capacity
    );
}
