use film_juicer_core::gamut::Hull;
fn rejected(hull: Hull) { hull.xy()[0][0] = 0.0; hull.center()[0] = 0.0; }
