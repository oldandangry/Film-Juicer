use super::*;
use crate::test_cases::RowCases;
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
    let mut source_cases = RowCases::new("output-color-source", ["source"]);
    let source_case = source_cases.case("source");
    compare(&SOURCE, &fixture["source"]);
    source_case.passed();
    source_cases.finish();
    let entries = fixture["entries"].as_array().unwrap();
    let invalid_tags = [9, 255, u32::MAX];
    let ids = entries
        .iter()
        .map(|row| row["name"].as_str().unwrap().to_owned())
        .chain(invalid_tags.map(|tag| format!("invalid-tag/{tag}")))
        .collect::<Vec<_>>();
    let mut cases = RowCases::new("output-color-entries-and-invalid-tags", ids);
    for (tag, literal) in LITERALS.iter().enumerate() {
        let row = &fixture["entries"][tag];
        let case = cases.case(row["name"].as_str().unwrap());
        compare(literal, row);
        let color = output_color(OutputSpace::try_from(tag as u32).unwrap());
        compare_completed(color, row);
        case.passed();
    }
    assert_ne!(
        entry_hash(&SOURCE),
        output_color(OutputSpace::DaVinciWideGamut).hash()
    );
    for tag in invalid_tags {
        let case = cases.case(format!("invalid-tag/{tag}"));
        assert!(OutputSpace::try_from(tag).is_err());
        case.passed();
    }
    cases.finish();
}
#[test]
fn canonical_zero_and_nonfinite_subhash_composition_are_native() {
    let fixture = fixture();
    let rows = fixture["hash_variants"].as_array().unwrap();
    let mut cases = RowCases::new(
        "output-color-hash-variants",
        rows.iter().map(|row| {
            format!(
                "component/{}/word/{:08x}",
                row["component"].as_u64().unwrap(),
                row["word"].as_u64().unwrap()
            )
        }),
    );
    for row in rows {
        let case = cases.case(format!(
            "component/{}/word/{:08x}",
            row["component"].as_u64().unwrap(),
            row["word"].as_u64().unwrap()
        ));
        let original = &LITERALS[0];
        let mut literal = Literal {
            base: original.base,
            rgb_to_xyz: original.rgb_to_xyz,
            xyz_to_rgb: original.xyz_to_rgb,
            kind: original.kind,
            coefficients: original.coefficients,
        };
        let lane = match row["component"].as_u64().unwrap() {
            0 => &mut literal.base[0],
            1 => &mut literal.rgb_to_xyz[0],
            2 => &mut literal.xyz_to_rgb[0],
            _ => &mut literal.coefficients[0],
        };
        *lane = f32::from_bits(row["word"].as_u64().unwrap() as u32);
        compare(&literal, &row["entry"]);
        case.passed();
    }
    cases.finish();
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
