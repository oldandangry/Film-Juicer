use film_juicer_core::scanner;
fn prepare(input: scanner::LutInput<'_>) -> scanner::Lut {
    let lut = scanner::Lut::new(input).unwrap();
    let _: u32 = lut.resolution();
    let _: [&[f32];6] = [lut.log2_xyz(),lut.slope_c(),lut.slope_m(),lut.slope_y(),lut.cell_min(),lut.cell_max()];
    lut
}
