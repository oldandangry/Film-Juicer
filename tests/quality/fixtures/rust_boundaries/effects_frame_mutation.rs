use film_juicer_core::effects::frame::Frame;
fn mutation(frame: Frame) { frame.weave()[0] = 0.0; frame.origins()[0].cell_x = 0; }

