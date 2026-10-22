use film_juicer_core::print;
fn reject(illuminant: print::Illuminant) {
    let _ = print::Illuminant { ..illuminant };
}
