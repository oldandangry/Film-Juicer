use film_juicer_core::gamut::{self, Hull, HullError, HullInput};
fn complete() -> Result<Hull, HullError> {
    let samples = [1.0;81];
    gamut::build_input_hull(HullInput{x:&samples,y:&samples,z:&samples,d65:&samples})
}
fn transfer(hull: Hull) { std::thread::spawn(move || hull.hash()); }
