use super::*;
use serde_json::Value;

fn fixture() -> Value {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/ffi/fixtures/output-color/native.json"
    ))
    .expect("immutable native output-color fixture");
    serde_json::from_str(&source).unwrap()
}
fn words(values: &[f32]) -> Value {
    serde_json::json!(values.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
}
fn compare(literal: &Literal, row: &Value) {
    assert_eq!(words(&literal.base), row["base"]);
    assert_eq!(words(&literal.rgb_to_xyz), row["rgb_to_xyz"]);
    assert_eq!(words(&literal.xyz_to_rgb), row["xyz_to_rgb"]);
    assert_eq!(words(&literal.coefficients), row["cctf"]);
    assert_eq!(literal.kind as u64, row["kind"].as_u64().unwrap());
    assert_eq!(serde_json::json!(subhashes(literal)), row["subhashes"]);
    assert_eq!(entry_hash(literal), row["hash"].as_u64().unwrap());
}
fn compare_completed(color: &OutputColor, row: &Value) {
    assert_eq!(words(color.dwg_to_output()), row["dwg_to_output"]);
    assert_eq!(color.hash(), row["hash"].as_u64().unwrap());
    assert_eq!(words(color.rgb_to_xyz()), row["rgb_to_xyz"]);
    assert_eq!(words(color.xyz_to_rgb()), row["xyz_to_rgb"]);
    assert_eq!(
        words(color.white_xyz()),
        Value::Array(row["base"].as_array().unwrap()[8..].to_vec())
    );
    let cctf = color.cctf();
    assert_eq!(cctf.kind as u64, row["kind"].as_u64().unwrap());
    assert_eq!(
        words(&[
            cctf.gamma,
            cctf.a,
            cctf.b,
            cctf.c,
            cctf.d,
            cctf.linear_cutoff,
        ]),
        row["cctf"]
    );
}
#[test]
fn native_literals_matrices_and_identities_are_exact() {
    let fixture = fixture();
    let source_cases_ids: Vec<_> = IntoIterator::into_iter(["source"])
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !source_cases_ids.is_empty(),
        "empty case set: {}",
        "output-color-source"
    );
    let source_cases_count = source_cases_ids.len();
    let mut source_cases_remaining = source_cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        source_cases_remaining.len(),
        source_cases_count,
        "duplicate case IDs: {}",
        "output-color-source"
    );
    let source_case = AsRef::<str>::as_ref(&("source")).to_owned();
    compare(&SOURCE, &fixture["source"]);
    assert!(
        source_cases_remaining.remove(&source_case),
        "unexpected or repeated comparison: {source_case}"
    );
    assert!(
        source_cases_remaining.is_empty(),
        "missing comparisons: {source_cases_remaining:?}"
    );
    let entries = fixture["entries"].as_array().unwrap();
    let invalid_tags = [9, 255, u32::MAX];
    let ids = entries
        .iter()
        .map(|row| row["name"].as_str().unwrap().to_owned())
        .chain(invalid_tags.map(|tag| format!("invalid-tag/{tag}")))
        .collect::<Vec<_>>();
    let cases_ids: Vec<_> = IntoIterator::into_iter(ids)
        .map(|id| AsRef::<str>::as_ref(&id).to_owned())
        .collect();
    assert!(
        !cases_ids.is_empty(),
        "empty case set: {}",
        "output-color-entries-and-invalid-tags"
    );
    let cases_count = cases_ids.len();
    let mut cases_remaining = cases_ids
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        cases_remaining.len(),
        cases_count,
        "duplicate case IDs: {}",
        "output-color-entries-and-invalid-tags"
    );
    for (tag, literal) in LITERALS.iter().enumerate() {
        let row = &fixture["entries"][tag];
        let case = AsRef::<str>::as_ref(&(row["name"].as_str().unwrap())).to_owned();
        compare(literal, row);
        let color = output_color(OutputSpace::try_from(tag as u32).unwrap());
        compare_completed(color, row);
        assert!(
            cases_remaining.remove(&case),
            "unexpected or repeated comparison: {case}"
        );
    }
    assert_ne!(
        entry_hash(&SOURCE),
        output_color(OutputSpace::DaVinciWideGamut).hash()
    );
    for tag in invalid_tags {
        let case = AsRef::<str>::as_ref(&(format!("invalid-tag/{tag}"))).to_owned();
        assert!(OutputSpace::try_from(tag).is_err());
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
fn compare_hash_variants(rows: &[Value]) {
    // Negative zero, positive infinity and the frozen quiet NaN in every lane.
    let mut remaining = (0..4)
        .flat_map(|component| [0x8000_0000, 0x7f80_0000, 0x7fc0_0000].map(|word| (component, word)))
        .collect::<std::collections::BTreeSet<_>>();
    for row in rows {
        let component = row["component"].as_u64().unwrap();
        let word = u32::try_from(row["word"].as_u64().unwrap()).unwrap();
        assert!(
            remaining.contains(&(component, word)),
            "unexpected or duplicate hash variant for component {component}, word {word:08x}"
        );
        let original = &LITERALS[0];
        let mut literal = Literal {
            base: original.base,
            rgb_to_xyz: original.rgb_to_xyz,
            xyz_to_rgb: original.xyz_to_rgb,
            kind: original.kind,
            coefficients: original.coefficients,
        };
        let lane = match component {
            0 => &mut literal.base[0],
            1 => &mut literal.rgb_to_xyz[0],
            2 => &mut literal.xyz_to_rgb[0],
            _ => &mut literal.coefficients[0],
        };
        *lane = f32::from_bits(word);
        compare(&literal, &row["entry"]);
        remaining.remove(&(component, word));
    }
    assert!(
        remaining.is_empty(),
        "missing hash variant comparisons: {remaining:?}"
    );
}

#[test]
fn canonical_zero_and_nonfinite_subhash_composition_are_native() {
    compare_hash_variants(fixture()["hash_variants"].as_array().unwrap());
}

#[test]
#[should_panic(expected = "missing hash variant comparisons")]
fn empty_hash_variant_fixture_fails() {
    compare_hash_variants(&[]);
}

#[test]
#[should_panic(expected = "missing hash variant comparisons")]
fn incomplete_hash_variant_fixture_fails() {
    let fixture = fixture();
    let rows = fixture["hash_variants"].as_array().unwrap();
    compare_hash_variants(&rows[..rows.len() - 1]);
}

#[test]
#[should_panic(expected = "unexpected or duplicate hash variant")]
fn duplicate_hash_variant_fixture_fails() {
    let mut fixture = fixture();
    let rows = fixture["hash_variants"].as_array_mut().unwrap();
    rows[1] = rows[0].clone();
    compare_hash_variants(rows);
}
#[test]
fn concurrent_catalogue_initialization_publishes_complete_immutable_entries() {
    const CHILD: &str = "FILM_JUICER_OUTPUT_CATALOGUE_CHILD";
    const COMPLETED: &str = "first-use catalogue assertions passed";
    const TEST: &str = concat!(
        module_path!(),
        "::concurrent_catalogue_initialization_publishes_complete_immutable_entries"
    );
    if std::env::var_os(CHILD).is_none() {
        // libtest names omit the crate component supplied by module_path!().
        let test = TEST.split_once("::").unwrap().1;
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test, "--exact", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .expect("fresh output catalogue unit-test process");
        let stdout = String::from_utf8_lossy(&child.stdout);
        let stderr = String::from_utf8_lossy(&child.stderr);
        assert!(
            child.status.success()
                && stdout.contains("test result: ok. 1 passed; 0 failed;")
                && stdout.contains(COMPLETED),
            "fresh catalogue child {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            child.status
        );
        print!("{stdout}");
        eprint!("{stderr}");
        return;
    }
    assert!(
        CATALOGUE.get().is_none(),
        "fresh child must initialize catalogue"
    );
    let fixture = fixture();
    let barrier = std::sync::Barrier::new(16);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..16)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    std::array::from_fn::<_, 9, _>(|tag| {
                        let space = OutputSpace::try_from(tag as u32).unwrap();
                        let color = output_color(space);
                        compare_completed(color, &fixture["entries"][tag]);
                        assert!(std::ptr::eq(color, output_color(space)));
                        color
                    })
                })
            })
            .collect();
        let mut published: Option<[&OutputColor; 9]> = None;
        for thread in threads {
            let entries = thread.join().unwrap();
            if let Some(first) = published {
                for (entry, expected) in entries.iter().zip(first) {
                    assert!(std::ptr::eq(*entry, expected));
                }
            } else {
                published = Some(entries);
            }
        }
    });
    println!("{COMPLETED}");
}
