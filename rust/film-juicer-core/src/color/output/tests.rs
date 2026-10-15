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
#[test]
fn native_literals_matrices_and_identities_are_exact() {
    let fixture = fixture();
    compare(&SOURCE, &fixture["source"]);
    for (tag, literal) in LITERALS.iter().enumerate() {
        let row = &fixture["entries"][tag];
        compare(literal, row);
        let color = output_color(OutputSpace::try_from(tag as u32).unwrap());
        assert_eq!(words(color.dwg_to_output()), row["dwg_to_output"]);
        assert_eq!(color.hash(), row["hash"].as_u64().unwrap());
        assert_eq!(words(color.rgb_to_xyz()), row["rgb_to_xyz"]);
        assert_eq!(words(color.xyz_to_rgb()), row["xyz_to_rgb"]);
        assert_eq!(words(color.white_xyz()), words(&literal.base[8..]));
        let c = color.cctf();
        assert_eq!(
            words(&[c.gamma, c.a, c.b, c.c, c.d, c.linear_cutoff]),
            row["cctf"]
        );
    }
    assert_ne!(
        entry_hash(&SOURCE),
        output_color(OutputSpace::DaVinciWideGamut).hash()
    );
    for tag in [9, 255, u32::MAX] {
        assert!(OutputSpace::try_from(tag).is_err());
    }
}
#[test]
fn canonical_zero_and_nonfinite_subhash_composition_are_native() {
    let fixture = fixture();
    for row in fixture["hash_variants"].as_array().unwrap() {
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
    }
}
#[test]
fn concurrent_catalogue_initialization_publishes_complete_immutable_entries() {
    let barrier = std::sync::Barrier::new(16);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..16)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    for tag in 0..9 {
                        let space = OutputSpace::try_from(tag).unwrap();
                        let first = output_color(space);
                        assert!(std::ptr::eq(first, output_color(space)));
                        assert_eq!(first.hash(), entry_hash(&LITERALS[tag as usize]));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    });
}
