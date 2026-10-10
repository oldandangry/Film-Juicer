use film_juicer_core::{gamut, reconstruction};
fn complete() -> Result<reconstruction::FilmTcLut, reconstruction::TcError> {
    let spectra=vec![1.0;reconstruction::SPECTRA_SAMPLE_COUNT];
    let sensitivity=[[1.0;3];81];let illuminant=[1.0;81];let xy=[[0.0;2];1025];
    reconstruction::build_tc_lut(reconstruction::Input{spectra:spectra.as_slice().try_into().unwrap(),sensitivity_rgb:&sensitivity,reference_illuminant:&illuminant,projection_white_xyz:[1.0;3],method:reconstruction::Method::Arctic,input_compression_active:false,input_hull:Some(gamut::InputHull::new([0.0;2],&xy))})
}
fn transferred(lut: reconstruction::FilmTcLut) { let _ = lut.into_samples(); }
