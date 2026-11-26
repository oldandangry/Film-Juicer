use super::{self as illuminant, ErrorKind};
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
    let blackbody_expected_ids: Vec<_> = IntoIterator::into_iter(&blackbody_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !blackbody_expected_ids.is_empty(),
        "empty case set: {}",
        "illuminant-blackbody"
    );
    let blackbody_count = blackbody_expected_ids.len();
    let mut blackbody_remaining = blackbody_expected_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        blackbody_remaining.len(),
        blackbody_count,
        "duplicate case IDs: {}",
        "illuminant-blackbody"
    );
    for (row, id) in blackbody_rows.iter().zip(&blackbody_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        expect(
            &illuminant::blackbody(f32::from_bits(row["temperature"].as_u64().unwrap() as u32)),
            &row["normalized"],
            "blackbody",
        );
        assert!(
            blackbody_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        blackbody_remaining.is_empty(),
        "missing comparisons: {blackbody_remaining:?}"
    );
    let equal_cases_ids: Vec<_> = IntoIterator::into_iter(["equal-energy"])
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !equal_cases_ids.is_empty(),
        "empty case set: {}",
        "illuminant-equal-energy"
    );
    let equal_cases_count = equal_cases_ids.len();
    let mut equal_cases_remaining = equal_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        equal_cases_remaining.len(),
        equal_cases_count,
        "duplicate case IDs: {}",
        "illuminant-equal-energy"
    );
    let equal = AsRef::<str>::as_ref(&("equal-energy")).to_owned();
    expect(
        &illuminant::equal_energy(),
        &leaf["equal"]["samples"],
        "equal",
    );
    assert!(
        equal_cases_remaining.remove(&equal),
        "unexpected or repeated comparison: {equal}"
    );
    assert!(
        equal_cases_remaining.is_empty(),
        "missing comparisons: {equal_cases_remaining:?}"
    );
    let direct_rows = leaf["direct"].as_array().unwrap();
    let direct_ids = ids(direct_rows, "direct");
    let direct_expected_ids: Vec<_> = IntoIterator::into_iter(&direct_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !direct_expected_ids.is_empty(),
        "empty case set: {}",
        "illuminant-direct"
    );
    let direct_count = direct_expected_ids.len();
    let mut direct_remaining = direct_expected_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        direct_remaining.len(),
        direct_count,
        "duplicate case IDs: {}",
        "illuminant-direct"
    );
    for (row, id) in direct_rows.iter().zip(&direct_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            direct_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        direct_remaining.is_empty(),
        "missing comparisons: {direct_remaining:?}"
    );
    let resample_rows = leaf["resample"].as_array().unwrap();
    let resample_ids = ids(resample_rows, "resample");
    let resample_cases_ids: Vec<_> = IntoIterator::into_iter(&resample_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !resample_cases_ids.is_empty(),
        "empty case set: {}",
        "illuminant-resample"
    );
    let resample_cases_count = resample_cases_ids.len();
    let mut resample_cases_remaining = resample_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        resample_cases_remaining.len(),
        resample_cases_count,
        "duplicate case IDs: {}",
        "illuminant-resample"
    );
    for (row, id) in resample_rows.iter().zip(&resample_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
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
        assert!(
            resample_cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        resample_cases_remaining.is_empty(),
        "missing comparisons: {resample_cases_remaining:?}"
    );
}
#[test]
fn accepted_native_private_leaves() {
    let capture = fixture();
    let leaf = &capture["leaf"];
    let blackbody_rows = leaf["blackbody"].as_array().unwrap();
    let blackbody_ids = ids(blackbody_rows, "blackbody");
    let blackbody_expected_ids: Vec<_> = IntoIterator::into_iter(&blackbody_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !blackbody_expected_ids.is_empty(),
        "empty case set: {}",
        "private-illuminant-blackbody"
    );
    let blackbody_count = blackbody_expected_ids.len();
    let mut blackbody_remaining = blackbody_expected_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        blackbody_remaining.len(),
        blackbody_count,
        "duplicate case IDs: {}",
        "private-illuminant-blackbody"
    );
    for (row, id) in blackbody_rows.iter().zip(&blackbody_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        let temperature = f32::from_bits(row["temperature"].as_u64().unwrap() as u32);
        let raw: [f32; 81] =
            std::array::from_fn(|i| illuminant::planck(380.0 + 5.0 * i as f32, temperature));
        expect(&raw, &row["raw"], "Planck");
        assert!(
            blackbody_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        blackbody_remaining.is_empty(),
        "missing comparisons: {blackbody_remaining:?}"
    );
    let resample_rows = leaf["resample"].as_array().unwrap();
    let resample_ids = ids(resample_rows, "resample");
    let resample_cases_ids: Vec<_> = IntoIterator::into_iter(&resample_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !resample_cases_ids.is_empty(),
        "empty case set: {}",
        "private-illuminant-resample"
    );
    let resample_cases_count = resample_cases_ids.len();
    let mut resample_cases_remaining = resample_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        resample_cases_remaining.len(),
        resample_cases_count,
        "duplicate case IDs: {}",
        "private-illuminant-resample"
    );
    for (row, id) in resample_rows.iter().zip(&resample_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        let result = illuminant::resample(&rows(&row["rows"]));
        if row["resampled"].as_array().unwrap().is_empty() {
            assert_eq!(result.unwrap_err().kind, ErrorKind::Preparation);
        } else {
            let expected: Value = rows(&row["resampled"])
                .iter()
                .map(|row| Value::from(row[1].to_bits()))
                .collect();
            expect(&result.unwrap(), &expected, "Akima");
        }
        assert!(
            resample_cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        resample_cases_remaining.is_empty(),
        "missing comparisons: {resample_cases_remaining:?}"
    );
    let normalization_rows = leaf["normalization"].as_array().unwrap();
    let normalization_ids = ids(normalization_rows, "normalization");
    let normalization_expected_ids: Vec<_> = IntoIterator::into_iter(&normalization_ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !normalization_expected_ids.is_empty(),
        "empty case set: {}",
        "private-illuminant-normalization"
    );
    let normalization_count = normalization_expected_ids.len();
    let mut normalization_remaining = normalization_expected_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        normalization_remaining.len(),
        normalization_count,
        "duplicate case IDs: {}",
        "private-illuminant-normalization"
    );
    for (row, id) in normalization_rows.iter().zip(&normalization_ids) {
        let case = AsRef::<str>::as_ref(&(id)).to_owned();
        let mut samples = words(&row["input"]);
        illuminant::normalize(&mut samples);
        expect(&samples, &row["expected"], "normalization");
        assert!(
            normalization_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert!(
        normalization_remaining.is_empty(),
        "missing comparisons: {normalization_remaining:?}"
    );
}

#[test]
fn scratch_rejects_unaddressable_capacity() {
    assert_eq!(
        illuminant::scratch(usize::MAX).unwrap_err().kind,
        ErrorKind::Capacity
    );
}
