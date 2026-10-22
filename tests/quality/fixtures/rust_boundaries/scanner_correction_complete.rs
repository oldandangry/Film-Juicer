use film_juicer_core::scanner;
fn prepare(params: scanner::Params, direct: scanner::DirectInput<'_>, print: scanner::PrintInput<'_>) {
    for correction in [scanner::Correction::direct(params, Some(direct)), scanner::Correction::print(params, Some(print))].into_iter().flatten() {
        let _: u8 = correction.route();
        let _: [bool;3] = [correction.active(), correction.black_correction(), correction.white_correction()];
        let _: [f32;7] = [correction.target_black_linear(), correction.target_white_linear(), correction.reference_black_y(), correction.reference_white_y(), correction.xyz_slope(), correction.xyz_offset(), correction.exposure_scale()];
        let _: u32 = correction.schema_version(); let _: u64 = correction.hash();
        if let Some(trace) = correction.trace() { let _: &[f32;8] = trace.values(); let _: scanner::TraceOutcome = trace.outcome(); let _: u64 = trace.hash(); }
    }
}
