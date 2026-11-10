//! Test-only row accounting for independently expected table-driven cases.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

pub(super) fn json_string(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len() + 2);
    encoded.push('"');
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            character if character < ' ' => {
                write!(encoded, "\\u{:04x}", u32::from(character)).expect("write to String");
            }
            _ => encoded.push(character),
        }
    }
    encoded.push('"');
    encoded
}

static ROW_WRITER: OnceLock<Mutex<Option<BufWriter<File>>>> = OnceLock::new();

fn append(record: String, best_effort: bool) {
    let path = std::env::var_os("FJ_RUST_ROWS");
    let Some(path) = path else {
        return;
    };
    let writer = ROW_WRITER.get_or_init(|| {
        let opened = OpenOptions::new()
            .create(true)
            .append(true)
            .open(PathBuf::from(path));
        match opened {
            Ok(file) => Mutex::new(Some(BufWriter::new(file))),
            Err(error) if best_effort => {
                let _ = error;
                Mutex::new(None)
            }
            Err(error) => panic!("open FJ_RUST_ROWS: {error}"),
        }
    });
    let mut writer = writer
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(writer) = writer.as_mut() else {
        assert!(
            best_effort,
            "FJ_RUST_ROWS writer unavailable after initialization failure"
        );
        return;
    };
    let result = writer
        .write_all(record.as_bytes())
        .and_then(|()| writer.write_all(b"\n"));
    if best_effort {
        let _ = writer.flush();
    }
    if let Err(error) = result
        && !best_effort
    {
        panic!("write FJ_RUST_ROWS: {error}");
    }
}

/// A declared logical cohort and its independently expected row membership.
pub struct RowCases {
    cohort: String,
    expected: BTreeSet<String>,
    started: BTreeSet<String>,
    terminal: BTreeSet<String>,
    function: String,
    binary: String,
    any_failed: bool,
    finished: bool,
}

impl RowCases {
    pub fn new<I, S>(cohort: impl Into<String>, expected_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let expected_values: Vec<String> = expected_ids
            .into_iter()
            .map(|id| id.as_ref().to_owned())
            .collect();
        let expected: BTreeSet<String> = expected_values.iter().cloned().collect();
        assert_eq!(expected.len(), expected_values.len(), "duplicate row ID");
        assert!(
            !expected.is_empty(),
            "row cohort must declare at least one ID"
        );
        assert!(
            expected.iter().all(|id| !id.is_empty()),
            "row IDs must not be empty"
        );
        let cohort = cohort.into();
        assert!(!cohort.is_empty(), "row cohort name must not be empty");
        let function =
            std::env::var("NEXTEST_TEST_NAME").unwrap_or_else(|_| "unknown-test".to_owned());
        let binary =
            std::env::var("NEXTEST_BINARY_ID").unwrap_or_else(|_| "unknown-binary".to_owned());
        let cases = Self {
            cohort,
            expected,
            started: BTreeSet::new(),
            terminal: BTreeSet::new(),
            function,
            binary,
            any_failed: false,
            finished: false,
        };
        let mut record = cases.prefix("membership");
        record.push_str(",\"ids\":[");
        for (index, id) in cases.expected.iter().enumerate() {
            if index != 0 {
                record.push(',');
            }
            record.push_str(&json_string(id));
        }
        record.push_str("]}");
        append(record, false);
        cases
    }

    pub fn case(&mut self, id: impl AsRef<str>) -> RowGuard<'_> {
        let id = id.as_ref().to_owned();
        assert!(self.expected.contains(&id), "undeclared row ID: {id}");
        assert!(self.started.insert(id.clone()), "row started twice: {id}");
        self.event(&id, "start", None, false);
        RowGuard {
            cases: self,
            id,
            active: true,
        }
    }

    fn prefix(&self, event: &str) -> String {
        format!(
            r#"{{"caller":"rust","event":{},"binary":{},"function":{},"cohort":{}"#,
            json_string(event),
            json_string(&self.binary),
            json_string(&self.function),
            json_string(&self.cohort)
        )
    }

    fn event(&self, id: &str, event: &str, outcome: Option<&str>, best_effort: bool) {
        let mut record = self.prefix(event);
        record.push_str(",\"id\":");
        record.push_str(&json_string(id));
        if let Some(outcome) = outcome {
            record.push_str(",\"outcome\":");
            record.push_str(&json_string(outcome));
        }
        record.push('}');
        append(record, best_effort);
    }

    fn complete(&self, outcome: &str, best_effort: bool) {
        let mut record = self.prefix("cohort-terminal");
        record.push_str(",\"outcome\":");
        record.push_str(&json_string(outcome));
        record.push('}');
        append(record, best_effort);
    }

    fn terminal(&mut self, id: &str, outcome: &str, best_effort: bool) {
        if self.terminal.insert(id.to_owned()) {
            self.any_failed |= outcome != "passed";
            self.event(id, "terminal", Some(outcome), best_effort);
        }
    }

    pub fn finish(mut self) {
        assert_eq!(
            self.started, self.expected,
            "row cohort {} did not start its full declared membership",
            self.cohort
        );
        assert!(
            !self.any_failed,
            "row cohort {} contains a failed row",
            self.cohort
        );
        assert_eq!(
            self.terminal, self.expected,
            "row cohort {} did not complete its full declared membership",
            self.cohort
        );
        self.complete("passed", false);
        self.finished = true;
        if let Some(writer) = ROW_WRITER.get()
            && let Ok(mut writer) = writer.lock()
            && let Some(writer) = writer.as_mut()
        {
            writer.flush().expect("flush FJ_RUST_ROWS");
        }
    }
}

impl Drop for RowCases {
    fn drop(&mut self) {
        if !self.finished {
            for id in self
                .started
                .difference(&self.terminal)
                .cloned()
                .collect::<Vec<_>>()
            {
                self.terminal(&id, "failed", true);
            }
            self.complete("failed", true);
        }
    }
}

pub struct RowGuard<'a> {
    cases: &'a mut RowCases,
    id: String,
    active: bool,
}

impl RowGuard<'_> {
    pub fn passed(mut self) {
        self.cases.terminal(&self.id, "passed", false);
        self.active = false;
    }
}

impl Drop for RowGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.cases.terminal(&self.id, "failed", true);
        }
    }
}
