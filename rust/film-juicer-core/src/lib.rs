//! Safe model and application logic for Film-Juicer.

#![forbid(unsafe_code)]

pub mod route;
pub mod film;
pub mod profile;
pub mod data_io;
pub mod hash;
pub mod assets;
pub mod color;
pub mod illuminant;

pub mod spectral;
pub mod reconstruction;
pub mod exposure;

pub mod gamut;

pub mod effects;

pub mod optics;

pub mod print;
pub mod scanner;

pub mod frame;

pub mod recipe;
pub mod state;

#[cfg(test)]
#[path = "../tests/support/case_results.rs"]
mod test_cases;

#[cfg(test)]
mod case_results_tests {
    #[test]
    fn protocol_strings_preserve_unicode_and_controls() {
        for value in [
            r#"quotes " and backslash \"#,
            "\0\u{001f}\n\r\t\u{0008}\u{000c}",
            "é 汉字 🧪",
        ] {
            let encoded = crate::test_cases::json_string(value);
            assert_eq!(serde_json::from_str::<String>(&encoded).unwrap(), value);
        }
    }
}
