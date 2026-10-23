use film_juicer_core::film::reference;
fn escape() -> reference::Reconstruction<'static> {
    let basis = [[1.0; 3]; 81]; let light = [1.0; 81]; let sensitivity = [[1.0; 3]; 81];
    reference::Reconstruction::Mallett { basis_rgb: &basis, illuminant: &light, sensitivity_rgb: &sensitivity, green_scale: 1.0 }
}
