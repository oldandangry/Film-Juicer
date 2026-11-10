use film_juicer_core::print;
fn mutate(raw: print::PreflashRaw) { raw.cmy()[0] = 0.0; }
