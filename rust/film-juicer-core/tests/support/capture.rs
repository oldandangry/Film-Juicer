use serde_json::Value;
use std::path::Path;

pub(crate) fn read(root: &Path, name: &str) -> Value {
    let path = root.join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

pub(crate) fn read_parts(root: &Path, files: &Value) -> Value {
    let files = files.as_array().expect("capture part list");
    assert!(!files.is_empty(), "empty capture part list");
    let mut capture = serde_json::Map::new();
    for file in files {
        let name = file.as_str().expect("capture part filename");
        let Value::Object(part) = read(root, name) else {
            panic!("{name}: expected a capture object");
        };
        for (field, value) in part {
            if field == "replace_rows" {
                for (block, replacements) in value.as_object().expect("capture row blocks") {
                    let rows = capture
                        .get_mut(block)
                        .and_then(Value::as_array_mut)
                        .unwrap_or_else(|| panic!("{name}: missing capture row block {block}"));
                    for (index, row) in replacements.as_object().expect("capture row replacements")
                    {
                        let index: usize = index.parse().expect("capture row index");
                        rows[index] = row.clone();
                    }
                }
            } else {
                assert!(
                    capture.insert(field.clone(), value).is_none(),
                    "{name}: repeated capture field {field}"
                );
            }
        }
    }
    Value::Object(capture)
}
