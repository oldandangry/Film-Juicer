use film_juicer_core::film::{self, reference};
fn prepare(development: &film::Development) -> reference::Reference {
    let value;
    {
        let mut basis = [[1.0; 3]; 81]; let mut light = [1.0; 81]; let mut sensitivity = [[1.0; 3]; 81];
        value = reference::Reference::new(development, reference::Params { manual_exposure_ev: 0.0, auto_exposure_enabled: false }, Some(reference::Reconstruction::Mallett { basis_rgb: &basis, illuminant: &light, sensitivity_rgb: &sensitivity, green_scale: 1.0 })).unwrap();
        basis.fill([0.0; 3]); light.fill(0.0); sensitivity.fill([0.0; 3]);
    }
    value
}
