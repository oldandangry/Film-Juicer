use film_juicer_core::print;
fn prepare() -> print::PreflashRaw {
    let raw;
    {
        let mut base = [0.0;81]; let mut light = [1.0;81]; let mut sensitivity = [[1.0;3];81];
        raw = print::PreflashRaw::new(&base, &light, &sensitivity).unwrap();
        base.fill(1.0); light.fill(0.0); sensitivity.fill([0.0;3]);
    }
    raw
}
