use film_juicer_core::illuminant::{self, ErrorKind};
#[path = "support/case_results.rs"]
mod test_cases;
use test_cases::RowCases;
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
fn ids(rows: &[Value], label: &str) -> Vec<String> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            row["id"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{label}/{i}"))
        })
        .collect()
}
#[test]
fn accepted_native_complete_operations() {
    let capture = fixture();
    let leaf = &capture["leaf"];
    let blackbody_rows = leaf["blackbody"].as_array().unwrap();
    let blackbody_ids = ids(blackbody_rows, "blackbody");
    let mut blackbody = RowCases::new("illuminant-blackbody", &blackbody_ids);
    for (row, id) in blackbody_rows.iter().zip(&blackbody_ids) {
        let case = blackbody.case(id);
        expect(
            &illuminant::blackbody(f32::from_bits(row["temperature"].as_u64().unwrap() as u32)),
            &row["normalized"],
            "blackbody",
        );
        case.passed();
    }
    blackbody.finish();
    let equal_case = RowCases::new("illuminant-equal-energy", ["equal-energy"]);
    let mut equal_cases = equal_case;
    let equal = equal_cases.case("equal-energy");
    expect(
        &illuminant::equal_energy(),
        &leaf["equal"]["samples"],
        "equal",
    );
    equal.passed();
    equal_cases.finish();
    let direct_rows = leaf["direct"].as_array().unwrap();
    let direct_ids = ids(direct_rows, "direct");
    let mut direct = RowCases::new("illuminant-direct", &direct_ids);
    for (row, id) in direct_rows.iter().zip(&direct_ids) {
        let case = direct.case(id);
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
        case.passed();
    }
    direct.finish();
    let resample_rows = leaf["resample"].as_array().unwrap();
    let resample_ids = ids(resample_rows, "resample");
    let mut resample_cases = RowCases::new("illuminant-resample", &resample_ids);
    for (row, id) in resample_rows.iter().zip(&resample_ids) {
        let case = resample_cases.case(id);
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
        case.passed();
    }
    resample_cases.finish();
}
#[cfg(feature = "test-support")]
#[test]
fn accepted_native_private_leaves() {
    let capture = fixture();
    let leaf = &capture["leaf"];
    let blackbody_rows = leaf["blackbody"].as_array().unwrap();
    let blackbody_ids = ids(blackbody_rows, "blackbody");
    let mut blackbody = RowCases::new("private-illuminant-blackbody", &blackbody_ids);
    for (row, id) in blackbody_rows.iter().zip(&blackbody_ids) {
        let case = blackbody.case(id);
        let temperature = f32::from_bits(row["temperature"].as_u64().unwrap() as u32);
        let raw: [f32; 81] = std::array::from_fn(|i| {
            illuminant::test_support::planck_sample(380.0 + 5.0 * i as f32, temperature)
        });
        expect(&raw, &row["raw"], "Planck");
        case.passed();
    }
    blackbody.finish();
    let resample_rows = leaf["resample"].as_array().unwrap();
    let resample_ids = ids(resample_rows, "resample");
    let mut resample_cases = RowCases::new("private-illuminant-resample", &resample_ids);
    for (row, id) in resample_rows.iter().zip(&resample_ids) {
        let case = resample_cases.case(id);
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
        case.passed();
    }
    resample_cases.finish();
    let normalization_rows = leaf["normalization"].as_array().unwrap();
    let normalization_ids = ids(normalization_rows, "normalization");
    let mut normalization = RowCases::new("private-illuminant-normalization", &normalization_ids);
    for (row, id) in normalization_rows.iter().zip(&normalization_ids) {
        let case = normalization.case(id);
        let mut samples = words(&row["input"]);
        illuminant::test_support::normalize(&mut samples);
        expect(&samples, &row["expected"], "normalization");
        case.passed();
    }
    normalization.finish();
    let mut capacity = RowCases::new("private-illuminant-capacity", ["capacity-fault"]);
    let row = capacity.case("capacity-fault");
    assert_eq!(
        illuminant::test_support::capacity_failure()
            .unwrap_err()
            .kind,
        ErrorKind::Capacity
    );
    row.passed();
    capacity.finish();
}
