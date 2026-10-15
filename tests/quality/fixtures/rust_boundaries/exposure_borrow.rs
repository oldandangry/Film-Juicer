use film_juicer_core::{exposure, reconstruction};
fn reference_result_owns_only_completed_white() -> reconstruction::ReferenceWhite {
    let tensor = vec![1.0; reconstruction::SPECTRA_SAMPLE_COUNT];
    let white = reconstruction::reference_white(tensor.as_slice().try_into().unwrap(), 0.0, [1.0, 1.0, 1.0]).unwrap();
    drop(tensor);
    white
}
fn sensitivity_result_owns_only_completed_values() -> exposure::Sensitivity {
    let source = [[1.0; 3]; 81];
    let illuminant = [1.0; 81];
    exposure::prepare_sensitivity(exposure::Input {
        linear_sensitivity_rgb: &source,
        reference_illuminant: &illuminant,
        band_pass: None,
        method: exposure::Method::Mallett,
    }).unwrap()
}
