use film_juicer_core::color::{OutputSpace, output_color};
fn consume() {
    let color = output_color(OutputSpace::DisplayP3);
    let _: &'static [f32; 9] = color.dwg_to_output();
    let _ = (color.rgb_to_xyz(), color.xyz_to_rgb(), color.white_xyz(), color.cctf(), color.hash());
}
