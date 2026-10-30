namespace {
    thread_local unsigned nextEffectsProjection = 0;
    FjPhysicalEffectsParams effects_params(const Json& controls) {
        const auto& amounts = controls.at("amounts");
        return {std::bit_cast<float>(amounts[0].get<std::uint32_t>()),
                std::bit_cast<float>(amounts[1].get<std::uint32_t>()),
                std::bit_cast<float>(amounts[2].get<std::uint32_t>()),
                std::bit_cast<float>(amounts[3].get<std::uint32_t>()),
                std::bit_cast<double>(controls.at("weave").get<std::uint64_t>()),
                controls.at("grain_active").get<bool>() ? 1u : 0u,
                controls.at("grain_debug_view").get<std::int32_t>()};
    }
    PhysicalEffectsControls effects_controls(const FjPhysicalEffectsParams& v) {
        return {v.film_dust_amount, v.film_scratch_amount, v.gate_dust_amount, v.gate_scratch_amount, v.gate_weave_amount};
    }
    Json effects_projection(const FjDust& v) {
        return Json::array({std::bit_cast<std::uint32_t>(v.cell_width_mm),
                            std::bit_cast<std::uint32_t>(v.cell_height_mm),
                            std::bit_cast<std::uint32_t>(v.slot_probability),
                            std::bit_cast<std::uint32_t>(v.softness_min_mm),
                            std::bit_cast<std::uint32_t>(v.softness_max_mm),
                            std::bit_cast<std::uint32_t>(v.softness_size_cap_fraction),
                            std::bit_cast<std::uint32_t>(v.support_x_mm),
                            std::bit_cast<std::uint32_t>(v.support_y_mm),
                            std::bit_cast<std::uint32_t>(v.fiber_fraction),
                            std::bit_cast<std::uint32_t>(v.fiber_drift_fraction),
                            std::bit_cast<std::uint32_t>(v.fiber_first_knot_min),
                            std::bit_cast<std::uint32_t>(v.fiber_first_knot_max),
                            std::bit_cast<std::uint32_t>(v.fiber_second_knot_min),
                            std::bit_cast<std::uint32_t>(v.fiber_second_knot_max),
                            std::bit_cast<std::uint32_t>(v.fiber_interior_width_min_fraction),
                            std::bit_cast<std::uint32_t>(v.fiber_interior_width_max_fraction),
                            std::bit_cast<std::uint32_t>(v.diameter_min_mm),
                            std::bit_cast<std::uint32_t>(v.diameter_bulk_max_mm),
                            std::bit_cast<std::uint32_t>(v.diameter_max_mm),
                            std::bit_cast<std::uint32_t>(v.diameter_tail_fraction),
                            std::bit_cast<std::uint32_t>(v.fiber_length_min_mm),
                            std::bit_cast<std::uint32_t>(v.fiber_length_max_mm),
                            std::bit_cast<std::uint32_t>(v.fiber_width_min_mm),
                            std::bit_cast<std::uint32_t>(v.fiber_width_max_mm),
                            std::bit_cast<std::uint32_t>(v.opacity_faint_cumulative),
                            std::bit_cast<std::uint32_t>(v.opacity_intermediate_cumulative),
                            std::bit_cast<std::uint32_t>(v.compact_opacity_min),
                            std::bit_cast<std::uint32_t>(v.compact_opacity_faint_end),
                            std::bit_cast<std::uint32_t>(v.compact_opacity_intermediate_end),
                            std::bit_cast<std::uint32_t>(v.compact_opacity_max),
                            std::bit_cast<std::uint32_t>(v.fiber_opacity_min),
                            std::bit_cast<std::uint32_t>(v.fiber_opacity_faint_end),
                            std::bit_cast<std::uint32_t>(v.fiber_opacity_intermediate_end),
                            std::bit_cast<std::uint32_t>(v.fiber_opacity_max),
                            std::bit_cast<std::uint32_t>(v.compact_dominant_aspect_min),
                            std::bit_cast<std::uint32_t>(v.compact_dominant_aspect_max),
                            std::bit_cast<std::uint32_t>(v.compact_subsidiary_scale_min),
                            std::bit_cast<std::uint32_t>(v.compact_subsidiary_scale_max),
                            std::bit_cast<std::uint32_t>(v.compact_subsidiary_aspect_min),
                            std::bit_cast<std::uint32_t>(v.compact_subsidiary_aspect_max),
                            std::bit_cast<std::uint32_t>(v.compact_subsidiary_offset_max),
                            std::bit_cast<std::uint32_t>(v.compact_subsidiary_angle_max_radians)});
    }
    Json effects_projection(const DefectDustRecipe& v) {
        return Json::array({std::bit_cast<std::uint32_t>(v.cellWidthMm),
                            std::bit_cast<std::uint32_t>(v.cellHeightMm),
                            std::bit_cast<std::uint32_t>(v.slotProbability),
                            std::bit_cast<std::uint32_t>(v.softnessMinMm),
                            std::bit_cast<std::uint32_t>(v.softnessMaxMm),
                            std::bit_cast<std::uint32_t>(v.softnessSizeCapFraction),
                            std::bit_cast<std::uint32_t>(v.supportXMm),
                            std::bit_cast<std::uint32_t>(v.supportYMm),
                            std::bit_cast<std::uint32_t>(v.fiberFraction),
                            std::bit_cast<std::uint32_t>(v.fiberDriftFraction),
                            std::bit_cast<std::uint32_t>(v.fiberFirstKnotMin),
                            std::bit_cast<std::uint32_t>(v.fiberFirstKnotMax),
                            std::bit_cast<std::uint32_t>(v.fiberSecondKnotMin),
                            std::bit_cast<std::uint32_t>(v.fiberSecondKnotMax),
                            std::bit_cast<std::uint32_t>(v.fiberInteriorWidthMinFraction),
                            std::bit_cast<std::uint32_t>(v.fiberInteriorWidthMaxFraction),
                            std::bit_cast<std::uint32_t>(v.diameterMinMm),
                            std::bit_cast<std::uint32_t>(v.diameterBulkMaxMm),
                            std::bit_cast<std::uint32_t>(v.diameterMaxMm),
                            std::bit_cast<std::uint32_t>(v.diameterTailFraction),
                            std::bit_cast<std::uint32_t>(v.fiberLengthMinMm),
                            std::bit_cast<std::uint32_t>(v.fiberLengthMaxMm),
                            std::bit_cast<std::uint32_t>(v.fiberWidthMinMm),
                            std::bit_cast<std::uint32_t>(v.fiberWidthMaxMm),
                            std::bit_cast<std::uint32_t>(v.opacityFaintCumulative),
                            std::bit_cast<std::uint32_t>(v.opacityIntermediateCumulative),
                            std::bit_cast<std::uint32_t>(v.compactOpacityMin),
                            std::bit_cast<std::uint32_t>(v.compactOpacityFaintEnd),
                            std::bit_cast<std::uint32_t>(v.compactOpacityIntermediateEnd),
                            std::bit_cast<std::uint32_t>(v.compactOpacityMax),
                            std::bit_cast<std::uint32_t>(v.fiberOpacityMin),
                            std::bit_cast<std::uint32_t>(v.fiberOpacityFaintEnd),
                            std::bit_cast<std::uint32_t>(v.fiberOpacityIntermediateEnd),
                            std::bit_cast<std::uint32_t>(v.fiberOpacityMax),
                            std::bit_cast<std::uint32_t>(v.compactDominantAspectMin),
                            std::bit_cast<std::uint32_t>(v.compactDominantAspectMax),
                            std::bit_cast<std::uint32_t>(v.compactSubsidiaryScaleMin),
                            std::bit_cast<std::uint32_t>(v.compactSubsidiaryScaleMax),
                            std::bit_cast<std::uint32_t>(v.compactSubsidiaryAspectMin),
                            std::bit_cast<std::uint32_t>(v.compactSubsidiaryAspectMax),
                            std::bit_cast<std::uint32_t>(v.compactSubsidiaryOffsetMax),
                            std::bit_cast<std::uint32_t>(v.compactSubsidiaryAngleMaxRadians)});
    }
    Json effects_projection(const FjScratch& v) {
        return Json::array({std::bit_cast<std::uint32_t>(v.cell_width_mm),
                            std::bit_cast<std::uint32_t>(v.cell_height_mm),
                            std::bit_cast<std::uint32_t>(v.slot_probability),
                            std::bit_cast<std::uint32_t>(v.softness_min_mm),
                            std::bit_cast<std::uint32_t>(v.softness_max_mm),
                            std::bit_cast<std::uint32_t>(v.softness_size_cap_fraction),
                            std::bit_cast<std::uint32_t>(v.support_x_mm),
                            std::bit_cast<std::uint32_t>(v.support_y_mm),
                            std::bit_cast<std::uint32_t>(v.length_min_mm),
                            std::bit_cast<std::uint32_t>(v.length_bulk_max_mm),
                            std::bit_cast<std::uint32_t>(v.length_max_mm),
                            std::bit_cast<std::uint32_t>(v.length_tail_fraction),
                            std::bit_cast<std::uint32_t>(v.width_min_mm),
                            std::bit_cast<std::uint32_t>(v.width_bulk_max_mm),
                            std::bit_cast<std::uint32_t>(v.width_max_mm),
                            std::bit_cast<std::uint32_t>(v.width_tail_fraction),
                            std::bit_cast<std::uint32_t>(v.drift_fraction),
                            std::bit_cast<std::uint32_t>(v.first_knot_min),
                            std::bit_cast<std::uint32_t>(v.first_knot_max),
                            std::bit_cast<std::uint32_t>(v.second_knot_min),
                            std::bit_cast<std::uint32_t>(v.second_knot_max),
                            std::bit_cast<std::uint32_t>(v.interior_width_min_fraction),
                            std::bit_cast<std::uint32_t>(v.interior_width_max_fraction),
                            std::bit_cast<std::uint32_t>(v.interior_depth_min_fraction),
                            std::bit_cast<std::uint32_t>(v.interior_depth_max_fraction),
                            std::bit_cast<std::uint32_t>(v.endpoint_abrupt_probability),
                            std::bit_cast<std::uint32_t>(v.interruption_probability),
                            std::bit_cast<std::uint32_t>(v.gap_center_min),
                            std::bit_cast<std::uint32_t>(v.gap_center_max),
                            std::bit_cast<std::uint32_t>(v.gap_span_min),
                            std::bit_cast<std::uint32_t>(v.gap_span_max),
                            std::bit_cast<std::uint32_t>(v.scuff_probability),
                            std::bit_cast<std::uint32_t>(v.scuff_length_max_mm),
                            std::bit_cast<std::uint32_t>(v.scuff_angle_max_radians),
                            std::bit_cast<std::uint32_t>(v.strength_min),
                            std::bit_cast<std::uint32_t>(v.strength_max)});
    }
    Json effects_projection(const DefectScratchRecipe& v) {
        return Json::array({std::bit_cast<std::uint32_t>(v.cellWidthMm),
                            std::bit_cast<std::uint32_t>(v.cellHeightMm),
                            std::bit_cast<std::uint32_t>(v.slotProbability),
                            std::bit_cast<std::uint32_t>(v.softnessMinMm),
                            std::bit_cast<std::uint32_t>(v.softnessMaxMm),
                            std::bit_cast<std::uint32_t>(v.softnessSizeCapFraction),
                            std::bit_cast<std::uint32_t>(v.supportXMm),
                            std::bit_cast<std::uint32_t>(v.supportYMm),
                            std::bit_cast<std::uint32_t>(v.lengthMinMm),
                            std::bit_cast<std::uint32_t>(v.lengthBulkMaxMm),
                            std::bit_cast<std::uint32_t>(v.lengthMaxMm),
                            std::bit_cast<std::uint32_t>(v.lengthTailFraction),
                            std::bit_cast<std::uint32_t>(v.widthMinMm),
                            std::bit_cast<std::uint32_t>(v.widthBulkMaxMm),
                            std::bit_cast<std::uint32_t>(v.widthMaxMm),
                            std::bit_cast<std::uint32_t>(v.widthTailFraction),
                            std::bit_cast<std::uint32_t>(v.driftFraction),
                            std::bit_cast<std::uint32_t>(v.firstKnotMin),
                            std::bit_cast<std::uint32_t>(v.firstKnotMax),
                            std::bit_cast<std::uint32_t>(v.secondKnotMin),
                            std::bit_cast<std::uint32_t>(v.secondKnotMax),
                            std::bit_cast<std::uint32_t>(v.interiorWidthMinFraction),
                            std::bit_cast<std::uint32_t>(v.interiorWidthMaxFraction),
                            std::bit_cast<std::uint32_t>(v.interiorDepthMinFraction),
                            std::bit_cast<std::uint32_t>(v.interiorDepthMaxFraction),
                            std::bit_cast<std::uint32_t>(v.endpointAbruptProbability),
                            std::bit_cast<std::uint32_t>(v.interruptionProbability),
                            std::bit_cast<std::uint32_t>(v.gapCenterMin),
                            std::bit_cast<std::uint32_t>(v.gapCenterMax),
                            std::bit_cast<std::uint32_t>(v.gapSpanMin),
                            std::bit_cast<std::uint32_t>(v.gapSpanMax),
                            std::bit_cast<std::uint32_t>(v.scuffProbability),
                            std::bit_cast<std::uint32_t>(v.scuffLengthMaxMm),
                            std::bit_cast<std::uint32_t>(v.scuffAngleMaxRadians),
                            std::bit_cast<std::uint32_t>(v.strengthMin),
                            std::bit_cast<std::uint32_t>(v.strengthMax)});
    }
    Json effects_projection(const FjPhysicalEffectsRecipe& v) {
        return {{"film_dust", effects_projection(v.film_dust)}, {"film_scratch", effects_projection(v.film_scratch)}, {"gate_dust", effects_projection(v.gate_dust)}, {"gate_scratch", effects_projection(v.gate_scratch)}, {"weave", std::bit_cast<std::uint64_t>(v.gate_weave_amount)}, {"active", v.active != 0}, {"hash", v.hash}};
    }
    Json effects_projection(const FilmJuicerEffectsRecipe& v) {
        return {{"film_dust", effects_projection(v.filmDust)}, {"film_scratch", effects_projection(v.filmScratch)}, {"gate_dust", effects_projection(v.gateDust)}, {"gate_scratch", effects_projection(v.gateScratch)}, {"weave", std::bit_cast<std::uint64_t>(v.gateWeaveAmount)}, {"active", v.active}, {"hash", v.hash}};
    }
    FjPhysicalEffectsRecipe effects_sentinel() {
        FjPhysicalEffectsRecipe v{};
        v.film_dust.cell_width_mm = std::bit_cast<float>(UINT32_C(1048576001));
        v.film_dust.cell_height_mm = std::bit_cast<float>(UINT32_C(1048576002));
        v.film_dust.slot_probability = std::bit_cast<float>(UINT32_C(1048576003));
        v.film_dust.softness_min_mm = std::bit_cast<float>(UINT32_C(1048576004));
        v.film_dust.softness_max_mm = std::bit_cast<float>(UINT32_C(1048576005));
        v.film_dust.softness_size_cap_fraction = std::bit_cast<float>(UINT32_C(1048576006));
        v.film_dust.support_x_mm = std::bit_cast<float>(UINT32_C(1048576007));
        v.film_dust.support_y_mm = std::bit_cast<float>(UINT32_C(1048576008));
        v.film_dust.fiber_fraction = std::bit_cast<float>(UINT32_C(1048576009));
        v.film_dust.fiber_drift_fraction = std::bit_cast<float>(UINT32_C(1048576010));
        v.film_dust.fiber_first_knot_min = std::bit_cast<float>(UINT32_C(1048576011));
        v.film_dust.fiber_first_knot_max = std::bit_cast<float>(UINT32_C(1048576012));
        v.film_dust.fiber_second_knot_min = std::bit_cast<float>(UINT32_C(1048576013));
        v.film_dust.fiber_second_knot_max = std::bit_cast<float>(UINT32_C(1048576014));
        v.film_dust.fiber_interior_width_min_fraction = std::bit_cast<float>(UINT32_C(1048576015));
        v.film_dust.fiber_interior_width_max_fraction = std::bit_cast<float>(UINT32_C(1048576016));
        v.film_dust.diameter_min_mm = std::bit_cast<float>(UINT32_C(1048576017));
        v.film_dust.diameter_bulk_max_mm = std::bit_cast<float>(UINT32_C(1048576018));
        v.film_dust.diameter_max_mm = std::bit_cast<float>(UINT32_C(1048576019));
        v.film_dust.diameter_tail_fraction = std::bit_cast<float>(UINT32_C(1048576020));
        v.film_dust.fiber_length_min_mm = std::bit_cast<float>(UINT32_C(1048576021));
        v.film_dust.fiber_length_max_mm = std::bit_cast<float>(UINT32_C(1048576022));
        v.film_dust.fiber_width_min_mm = std::bit_cast<float>(UINT32_C(1048576023));
        v.film_dust.fiber_width_max_mm = std::bit_cast<float>(UINT32_C(1048576024));
        v.film_dust.opacity_faint_cumulative = std::bit_cast<float>(UINT32_C(1048576025));
        v.film_dust.opacity_intermediate_cumulative = std::bit_cast<float>(UINT32_C(1048576026));
        v.film_dust.compact_opacity_min = std::bit_cast<float>(UINT32_C(1048576027));
        v.film_dust.compact_opacity_faint_end = std::bit_cast<float>(UINT32_C(1048576028));
        v.film_dust.compact_opacity_intermediate_end = std::bit_cast<float>(UINT32_C(1048576029));
        v.film_dust.compact_opacity_max = std::bit_cast<float>(UINT32_C(1048576030));
        v.film_dust.fiber_opacity_min = std::bit_cast<float>(UINT32_C(1048576031));
        v.film_dust.fiber_opacity_faint_end = std::bit_cast<float>(UINT32_C(1048576032));
        v.film_dust.fiber_opacity_intermediate_end = std::bit_cast<float>(UINT32_C(1048576033));
        v.film_dust.fiber_opacity_max = std::bit_cast<float>(UINT32_C(1048576034));
        v.film_dust.compact_dominant_aspect_min = std::bit_cast<float>(UINT32_C(1048576035));
        v.film_dust.compact_dominant_aspect_max = std::bit_cast<float>(UINT32_C(1048576036));
        v.film_dust.compact_subsidiary_scale_min = std::bit_cast<float>(UINT32_C(1048576037));
        v.film_dust.compact_subsidiary_scale_max = std::bit_cast<float>(UINT32_C(1048576038));
        v.film_dust.compact_subsidiary_aspect_min = std::bit_cast<float>(UINT32_C(1048576039));
        v.film_dust.compact_subsidiary_aspect_max = std::bit_cast<float>(UINT32_C(1048576040));
        v.film_dust.compact_subsidiary_offset_max = std::bit_cast<float>(UINT32_C(1048576041));
        v.film_dust.compact_subsidiary_angle_max_radians = std::bit_cast<float>(UINT32_C(1048576042));
        v.film_scratch.cell_width_mm = std::bit_cast<float>(UINT32_C(1048576043));
        v.film_scratch.cell_height_mm = std::bit_cast<float>(UINT32_C(1048576044));
        v.film_scratch.slot_probability = std::bit_cast<float>(UINT32_C(1048576045));
        v.film_scratch.softness_min_mm = std::bit_cast<float>(UINT32_C(1048576046));
        v.film_scratch.softness_max_mm = std::bit_cast<float>(UINT32_C(1048576047));
        v.film_scratch.softness_size_cap_fraction = std::bit_cast<float>(UINT32_C(1048576048));
        v.film_scratch.support_x_mm = std::bit_cast<float>(UINT32_C(1048576049));
        v.film_scratch.support_y_mm = std::bit_cast<float>(UINT32_C(1048576050));
        v.film_scratch.length_min_mm = std::bit_cast<float>(UINT32_C(1048576051));
        v.film_scratch.length_bulk_max_mm = std::bit_cast<float>(UINT32_C(1048576052));
        v.film_scratch.length_max_mm = std::bit_cast<float>(UINT32_C(1048576053));
        v.film_scratch.length_tail_fraction = std::bit_cast<float>(UINT32_C(1048576054));
        v.film_scratch.width_min_mm = std::bit_cast<float>(UINT32_C(1048576055));
        v.film_scratch.width_bulk_max_mm = std::bit_cast<float>(UINT32_C(1048576056));
        v.film_scratch.width_max_mm = std::bit_cast<float>(UINT32_C(1048576057));
        v.film_scratch.width_tail_fraction = std::bit_cast<float>(UINT32_C(1048576058));
        v.film_scratch.drift_fraction = std::bit_cast<float>(UINT32_C(1048576059));
        v.film_scratch.first_knot_min = std::bit_cast<float>(UINT32_C(1048576060));
        v.film_scratch.first_knot_max = std::bit_cast<float>(UINT32_C(1048576061));
        v.film_scratch.second_knot_min = std::bit_cast<float>(UINT32_C(1048576062));
        v.film_scratch.second_knot_max = std::bit_cast<float>(UINT32_C(1048576063));
        v.film_scratch.interior_width_min_fraction = std::bit_cast<float>(UINT32_C(1048576064));
        v.film_scratch.interior_width_max_fraction = std::bit_cast<float>(UINT32_C(1048576065));
        v.film_scratch.interior_depth_min_fraction = std::bit_cast<float>(UINT32_C(1048576066));
        v.film_scratch.interior_depth_max_fraction = std::bit_cast<float>(UINT32_C(1048576067));
        v.film_scratch.endpoint_abrupt_probability = std::bit_cast<float>(UINT32_C(1048576068));
        v.film_scratch.interruption_probability = std::bit_cast<float>(UINT32_C(1048576069));
        v.film_scratch.gap_center_min = std::bit_cast<float>(UINT32_C(1048576070));
        v.film_scratch.gap_center_max = std::bit_cast<float>(UINT32_C(1048576071));
        v.film_scratch.gap_span_min = std::bit_cast<float>(UINT32_C(1048576072));
        v.film_scratch.gap_span_max = std::bit_cast<float>(UINT32_C(1048576073));
        v.film_scratch.scuff_probability = std::bit_cast<float>(UINT32_C(1048576074));
        v.film_scratch.scuff_length_max_mm = std::bit_cast<float>(UINT32_C(1048576075));
        v.film_scratch.scuff_angle_max_radians = std::bit_cast<float>(UINT32_C(1048576076));
        v.film_scratch.strength_min = std::bit_cast<float>(UINT32_C(1048576077));
        v.film_scratch.strength_max = std::bit_cast<float>(UINT32_C(1048576078));
        v.gate_dust.cell_width_mm = std::bit_cast<float>(UINT32_C(1048576079));
        v.gate_dust.cell_height_mm = std::bit_cast<float>(UINT32_C(1048576080));
        v.gate_dust.slot_probability = std::bit_cast<float>(UINT32_C(1048576081));
        v.gate_dust.softness_min_mm = std::bit_cast<float>(UINT32_C(1048576082));
        v.gate_dust.softness_max_mm = std::bit_cast<float>(UINT32_C(1048576083));
        v.gate_dust.softness_size_cap_fraction = std::bit_cast<float>(UINT32_C(1048576084));
        v.gate_dust.support_x_mm = std::bit_cast<float>(UINT32_C(1048576085));
        v.gate_dust.support_y_mm = std::bit_cast<float>(UINT32_C(1048576086));
        v.gate_dust.fiber_fraction = std::bit_cast<float>(UINT32_C(1048576087));
        v.gate_dust.fiber_drift_fraction = std::bit_cast<float>(UINT32_C(1048576088));
        v.gate_dust.fiber_first_knot_min = std::bit_cast<float>(UINT32_C(1048576089));
        v.gate_dust.fiber_first_knot_max = std::bit_cast<float>(UINT32_C(1048576090));
        v.gate_dust.fiber_second_knot_min = std::bit_cast<float>(UINT32_C(1048576091));
        v.gate_dust.fiber_second_knot_max = std::bit_cast<float>(UINT32_C(1048576092));
        v.gate_dust.fiber_interior_width_min_fraction = std::bit_cast<float>(UINT32_C(1048576093));
        v.gate_dust.fiber_interior_width_max_fraction = std::bit_cast<float>(UINT32_C(1048576094));
        v.gate_dust.diameter_min_mm = std::bit_cast<float>(UINT32_C(1048576095));
        v.gate_dust.diameter_bulk_max_mm = std::bit_cast<float>(UINT32_C(1048576096));
        v.gate_dust.diameter_max_mm = std::bit_cast<float>(UINT32_C(1048576097));
        v.gate_dust.diameter_tail_fraction = std::bit_cast<float>(UINT32_C(1048576098));
        v.gate_dust.fiber_length_min_mm = std::bit_cast<float>(UINT32_C(1048576099));
        v.gate_dust.fiber_length_max_mm = std::bit_cast<float>(UINT32_C(1048576100));
        v.gate_dust.fiber_width_min_mm = std::bit_cast<float>(UINT32_C(1048576101));
        v.gate_dust.fiber_width_max_mm = std::bit_cast<float>(UINT32_C(1048576102));
        v.gate_dust.opacity_faint_cumulative = std::bit_cast<float>(UINT32_C(1048576103));
        v.gate_dust.opacity_intermediate_cumulative = std::bit_cast<float>(UINT32_C(1048576104));
        v.gate_dust.compact_opacity_min = std::bit_cast<float>(UINT32_C(1048576105));
        v.gate_dust.compact_opacity_faint_end = std::bit_cast<float>(UINT32_C(1048576106));
        v.gate_dust.compact_opacity_intermediate_end = std::bit_cast<float>(UINT32_C(1048576107));
        v.gate_dust.compact_opacity_max = std::bit_cast<float>(UINT32_C(1048576108));
        v.gate_dust.fiber_opacity_min = std::bit_cast<float>(UINT32_C(1048576109));
        v.gate_dust.fiber_opacity_faint_end = std::bit_cast<float>(UINT32_C(1048576110));
        v.gate_dust.fiber_opacity_intermediate_end = std::bit_cast<float>(UINT32_C(1048576111));
        v.gate_dust.fiber_opacity_max = std::bit_cast<float>(UINT32_C(1048576112));
        v.gate_dust.compact_dominant_aspect_min = std::bit_cast<float>(UINT32_C(1048576113));
        v.gate_dust.compact_dominant_aspect_max = std::bit_cast<float>(UINT32_C(1048576114));
        v.gate_dust.compact_subsidiary_scale_min = std::bit_cast<float>(UINT32_C(1048576115));
        v.gate_dust.compact_subsidiary_scale_max = std::bit_cast<float>(UINT32_C(1048576116));
        v.gate_dust.compact_subsidiary_aspect_min = std::bit_cast<float>(UINT32_C(1048576117));
        v.gate_dust.compact_subsidiary_aspect_max = std::bit_cast<float>(UINT32_C(1048576118));
        v.gate_dust.compact_subsidiary_offset_max = std::bit_cast<float>(UINT32_C(1048576119));
        v.gate_dust.compact_subsidiary_angle_max_radians = std::bit_cast<float>(UINT32_C(1048576120));
        v.gate_scratch.cell_width_mm = std::bit_cast<float>(UINT32_C(1048576121));
        v.gate_scratch.cell_height_mm = std::bit_cast<float>(UINT32_C(1048576122));
        v.gate_scratch.slot_probability = std::bit_cast<float>(UINT32_C(1048576123));
        v.gate_scratch.softness_min_mm = std::bit_cast<float>(UINT32_C(1048576124));
        v.gate_scratch.softness_max_mm = std::bit_cast<float>(UINT32_C(1048576125));
        v.gate_scratch.softness_size_cap_fraction = std::bit_cast<float>(UINT32_C(1048576126));
        v.gate_scratch.support_x_mm = std::bit_cast<float>(UINT32_C(1048576127));
        v.gate_scratch.support_y_mm = std::bit_cast<float>(UINT32_C(1048576128));
        v.gate_scratch.length_min_mm = std::bit_cast<float>(UINT32_C(1048576129));
        v.gate_scratch.length_bulk_max_mm = std::bit_cast<float>(UINT32_C(1048576130));
        v.gate_scratch.length_max_mm = std::bit_cast<float>(UINT32_C(1048576131));
        v.gate_scratch.length_tail_fraction = std::bit_cast<float>(UINT32_C(1048576132));
        v.gate_scratch.width_min_mm = std::bit_cast<float>(UINT32_C(1048576133));
        v.gate_scratch.width_bulk_max_mm = std::bit_cast<float>(UINT32_C(1048576134));
        v.gate_scratch.width_max_mm = std::bit_cast<float>(UINT32_C(1048576135));
        v.gate_scratch.width_tail_fraction = std::bit_cast<float>(UINT32_C(1048576136));
        v.gate_scratch.drift_fraction = std::bit_cast<float>(UINT32_C(1048576137));
        v.gate_scratch.first_knot_min = std::bit_cast<float>(UINT32_C(1048576138));
        v.gate_scratch.first_knot_max = std::bit_cast<float>(UINT32_C(1048576139));
        v.gate_scratch.second_knot_min = std::bit_cast<float>(UINT32_C(1048576140));
        v.gate_scratch.second_knot_max = std::bit_cast<float>(UINT32_C(1048576141));
        v.gate_scratch.interior_width_min_fraction = std::bit_cast<float>(UINT32_C(1048576142));
        v.gate_scratch.interior_width_max_fraction = std::bit_cast<float>(UINT32_C(1048576143));
        v.gate_scratch.interior_depth_min_fraction = std::bit_cast<float>(UINT32_C(1048576144));
        v.gate_scratch.interior_depth_max_fraction = std::bit_cast<float>(UINT32_C(1048576145));
        v.gate_scratch.endpoint_abrupt_probability = std::bit_cast<float>(UINT32_C(1048576146));
        v.gate_scratch.interruption_probability = std::bit_cast<float>(UINT32_C(1048576147));
        v.gate_scratch.gap_center_min = std::bit_cast<float>(UINT32_C(1048576148));
        v.gate_scratch.gap_center_max = std::bit_cast<float>(UINT32_C(1048576149));
        v.gate_scratch.gap_span_min = std::bit_cast<float>(UINT32_C(1048576150));
        v.gate_scratch.gap_span_max = std::bit_cast<float>(UINT32_C(1048576151));
        v.gate_scratch.scuff_probability = std::bit_cast<float>(UINT32_C(1048576152));
        v.gate_scratch.scuff_length_max_mm = std::bit_cast<float>(UINT32_C(1048576153));
        v.gate_scratch.scuff_angle_max_radians = std::bit_cast<float>(UINT32_C(1048576154));
        v.gate_scratch.strength_min = std::bit_cast<float>(UINT32_C(1048576155));
        v.gate_scratch.strength_max = std::bit_cast<float>(UINT32_C(1048576156));
        v.gate_weave_amount = std::bit_cast<double>(UINT64_C(0x3ff0000000000001));
        v.active = 1;
        v.hash = 0; // Projection-only sentinel: zero identity is not a readiness flag.
        return v;
    }
    void effects_values(const Json& captured) {
        require(captured.at("cases").size() == 220, "exact effects literal membership");
        for (bool facade : {false, true}) {
            const auto call = facade ? fj_test_physical_effects_prepare : fj_legacy_physical_effects_prepare;
            for (const auto& row : captured.at("cases")) {
                const auto params = effects_params(row.at("controls"));
                auto out = effects_sentinel();
                FjFilmRecipeFailure failure{99, 99, 99};
                status(call(&params, &out, &failure, nullptr));
                metadata(failure);
                require(effects_projection(out) == row.at("recipe"), "220 exact native effects recipes through each checked entry");
            }
        }
    }
    void effects_failures(const Json& zero) {
        const FjPhysicalEffectsParams base{1, 2, 3, 4, 0.1, 0, 0};
        const FjStatus unsupported{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        const FjStatus preparationFailure{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const FjStatus internalFailure{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        for (bool facade : {false, true}) {
            const auto call = facade ? fj_test_physical_effects_prepare : fj_legacy_physical_effects_prepare;
            auto out = effects_sentinel();
            FjFilmRecipeFailure failure{};
            const auto rejected = [&](const FjPhysicalEffectsParams* params, FjStatus expected, std::uint32_t reason, FjErrorBuffer* error = nullptr) {
                out = effects_sentinel();
                failure = {99, 99, 99};
                const auto value = call(params, &out, &failure, error);
                require(value.category == expected.category && value.api == expected.api && value.native_code == expected.native_code, "complete effects failure status");
                metadata(failure, reason);
                require(effects_projection(out) == zero, "every admitted effects failure clears all semantic fields");
            };
            rejected(nullptr, unsupported, FJ_FILM_RECIPE_RAW);
            auto bad = base;
            bad.grain_active = 2;
            bad.grain_debug_view = 3;
            rejected(&bad, unsupported, FJ_FILM_RECIPE_RAW);
            alignas(FjPhysicalEffectsParams) std::array<std::byte, sizeof(FjPhysicalEffectsParams) + 8> input{};
            rejected(reinterpret_cast<const FjPhysicalEffectsParams*>(input.data() + 1), unsupported, FJ_FILM_RECIPE_RAW);
            rejected(reinterpret_cast<const FjPhysicalEffectsParams*>(UINTPTR_MAX - 7), unsupported, FJ_FILM_RECIPE_RAW);
            status(call(&base, nullptr, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_FILM_RECIPE_RAW);
            out = effects_sentinel();
            status(call(&base, &out, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(effects_projection(out) == zero, "independent recipe clearing without failure slot");
            alignas(FjFilmRecipeFailure) std::array<std::byte, sizeof(FjFilmRecipeFailure) + 8> failureStorage{};
            out = effects_sentinel();
            status(call(&base, &out, reinterpret_cast<FjFilmRecipeFailure*>(failureStorage.data() + 1), nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(effects_projection(out) == zero, "independent recipe clearing with misaligned failure slot");
            alignas(FjPhysicalEffectsRecipe) std::array<std::byte, sizeof(FjPhysicalEffectsRecipe) + 8> output{};
            status(call(&base, reinterpret_cast<FjPhysicalEffectsRecipe*>(output.data() + 1), &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_FILM_RECIPE_RAW);
            status(call(&base, reinterpret_cast<FjPhysicalEffectsRecipe*>(UINTPTR_MAX - 7), &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_FILM_RECIPE_RAW);
            const std::string message = "film recipe reason=1 channel=3 sample=18446744073709551615";
            for (std::size_t capacity : {0u, 1u, 12u, 128u}) {
                std::array<char, 132> text{};
                text.fill('!');
                FjErrorBuffer error{capacity ? text.data() : nullptr, capacity, 999};
                rejected(nullptr, unsupported, FJ_FILM_RECIPE_RAW, &error);
                const auto length = capacity ? std::min(capacity - 1, message.size()) : 0;
                require(error.length == length && (!capacity || (std::string_view(text.data(), length) == std::string_view(message).substr(0, length) && text[length] == '\0')), "effects diagnostic exact length/truncation/NUL");
                require(text[capacity] == '!', "diagnostic leaves bytes outside admitted capacity untouched");
            }
            FjErrorBuffer invalid{nullptr, 4, 99};
            rejected(&base, unsupported, FJ_FILM_RECIPE_RAW, &invalid);
            require(invalid.length == 0, "invalid backing independently clears valid diagnostic length");
            invalid = {reinterpret_cast<char*>(UINTPTR_MAX), 8, 99};
            rejected(&base, unsupported, FJ_FILM_RECIPE_RAW, &invalid);
            alignas(FjErrorBuffer) std::array<std::byte, sizeof(FjErrorBuffer) + 8> diagnosticStorage{};
            rejected(&base, unsupported, FJ_FILM_RECIPE_RAW, reinterpret_cast<FjErrorBuffer*>(diagnosticStorage.data() + 1));
            for (unsigned fault : {1u, 2u, 3u, 4u}) {
                status(fj_test_film_recipe_fault(8, fault));
                rejected(&base, fault == 1 ? internalFailure : preparationFailure, fault == 1 ? FJ_FILM_RECIPE_PANIC : FJ_FILM_RECIPE_EFFECTS_PROBABILITY);
                status(call(&base, &out, &failure, nullptr));
                metadata(failure);
                require(out.active == 1, "computed guard fault consumed once");
                status(fj_test_film_recipe_fault(8, fault));
                status(call(&base, nullptr, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
                metadata(failure, FJ_FILM_RECIPE_RAW);
                status(call(&base, &out, &failure, nullptr));
                metadata(failure);
            }
            auto suppressed = base;
            suppressed.grain_active = 1;
            suppressed.grain_debug_view = 3;
            status(fj_test_film_recipe_fault(8, 4));
            status(call(&suppressed, &out, &failure, nullptr));
            require(effects_projection(out) == zero, "suppression consumes private constructor fault before early return");
            status(call(&base, &out, &failure, nullptr));
            metadata(failure);
            status(fj_test_film_recipe_fault(8, 4));
            status(fj_test_film_recipe_fault(8, 0));
            status(call(&base, &out, &failure, nullptr));
        }
        std::atomic<bool> complete{true};
        std::array<std::thread, 4> threads;
        for (auto& thread : threads)
            thread = std::thread([&] {
                for (unsigned n = 0; n < 8; ++n) {
                    FjPhysicalEffectsRecipe out{};
                    FjFilmRecipeFailure failure{};
                    const auto armed = fj_test_film_recipe_fault(8, 4);
                    const auto failed = fj_test_physical_effects_prepare(&base, &out, &failure, nullptr);
                    if (armed.category != FJ_STATUS_SUCCESS || failed.category != FJ_STATUS_PREPARATION_FAILURE || failure.reason != FJ_FILM_RECIPE_EFFECTS_PROBABILITY || effects_projection(out) != zero)
                        complete = false;
                    const auto recovered = fj_test_physical_effects_prepare(&base, &out, &failure, nullptr);
                    if (recovered.category != FJ_STATUS_SUCCESS || recovered.api != FJ_API_NONE || recovered.native_code != 0 || failure.reason != 0 || failure.channel != 3 || failure.sample != UINT64_MAX)
                        complete = false;
                }
            });
        for (auto& thread : threads)
            thread.join();
        require(complete, "effects fault containment and recovery are thread local");
    }
    void effects_native(const Json& captured) {
        const PhysicalEffectsControls controls{1, 2, 3, 4, 0.1};
        VisualGrainRecipe grain;
        FilmJuicerEffectsRecipe out;
        std::string diagnostic;
        for (const auto& row : captured.at("cases")) {
            const auto params = effects_params(row.at("controls"));
            grain.active = params.grain_active != 0;
            grain.debugView = params.grain_debug_view;
            require(JuicerAssets::prepare_physical_effects(effects_controls(params), grain, out, diagnostic) && diagnostic.empty() && effects_projection(out) == row.at("recipe"), "all 156 named native mapping lanes and retained f64/activity/hash");
        }
        grain.active = false;
        grain.debugView = 0;
        for (unsigned fault : {5u, 6u}) {
            nextEffectsProjection = fault;
            require(JuicerAssets::prepare_physical_effects(controls, grain, out, diagnostic), "finite projection sentinel admitted");
            auto expected = effects_sentinel();
            if (fault == 6) {
                expected.active = 0;
                expected.gate_weave_amount = -0.0;
            }
            require(effects_projection(out) == effects_projection(expected), "field-distinct copy sentinel and active hash zero / populated inactive negative-zero admission");
        }
        for (unsigned fault : {1u, 2u, 3u, 4u}) {
            nextEffectsProjection = fault;
            bool rejected = false;
            try {
                (void)JuicerAssets::prepare_physical_effects(controls, grain, out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected && effects_projection(out) == effects_projection(FilmJuicerEffectsRecipe{}), "corrupt effects projection is fatal and cannot partially publish");
        }
        for (unsigned fault = 0; fault < 9; ++fault) {
            wireMutation = WireMutation{};
            wireMutation.active = true;
            wireMutation.operation = 5;
            wireMutation.failure = {FJ_FILM_RECIPE_EFFECTS_PROBABILITY, 3, UINT64_MAX};
            if (fault == 0)
                wireMutation.status.category = FJ_STATUS_SUCCESS;
            if (fault == 1)
                wireMutation.status.api = FJ_API_CUDA_RUNTIME;
            if (fault == 2)
                wireMutation.status.native_code = 1;
            if (fault == 3)
                wireMutation.failure.channel = 0;
            if (fault == 4)
                wireMutation.failure.sample = 0;
            if (fault == 5)
                wireMutation.failure.reason = FJ_FILM_RECIPE_GRAIN_CONTROLS;
            if (fault == 6)
                wireMutation.failure.reason = FJ_FILM_RECIPE_ALLOCATION;
            if (fault == 7)
                wireMutation.failure.reason = FJ_FILM_RECIPE_IDENTITY;
            if (fault == 8)
                wireMutation.failure.reason = 99;
            bool rejected = false;
            try {
                (void)JuicerAssets::prepare_physical_effects(controls, grain, out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected && !wireMutation.active, "stateless effects admits only its closed failure envelope");
        }
        status(fj_test_film_recipe_fault(8, 4));
        require(!JuicerAssets::prepare_physical_effects(controls, grain, out, diagnostic) && diagnostic == "ResourceDescriptorMismatch phase=effects_recipe field=physical_policy", "real computed probability guard preserves ordinary native diagnostic");
    }
    void effects_products(const std::filesystem::path& resources, const Json& captured) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        require(captured.at("products").size() == 92, "exact physical effects product membership");
        for (const auto& row : captured.at("products")) {
            const auto controls = effects_params(row.at("controls"));
            ParamSnapshot params;
            params.filmProfileKey = row.at("key");
            params.printProfileKey = "kodak_portra_endura";
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(row.at("route").get<unsigned>());
            params.spectralUpsamplingMode = 0;
            params.cameraAutoExposureEnabled = 0;
            params.inputCompressionEnabled = 0;
            params.outputGamutCompressionEnabled = 0;
            params.filmDustAmount = controls.film_dust_amount;
            params.filmScratchAmount = controls.film_scratch_amount;
            params.gateDustAmount = controls.gate_dust_amount;
            params.gateScratchAmount = controls.gate_scratch_amount;
            params.gateWeaveAmount = controls.gate_weave_amount;
            params.grainControls.active = controls.grain_active != 0;
            params.grainControls.debugView = controls.grain_debug_view;
            auto product = JuicerTestSupport::prepare_product(params);
            FjPreparedStateView view{};
            FjErrorBuffer error{};
            status(product.view(view, error));
            require(effects_projection(view.effects) == row.at("recipe") && view.recipe_hash == row.at("final_hash") && hash_params(params) == row.at("snapshot_hash") && view.grain.hash == row.at("grain_hash") && view.film_development.hash == row.at("development_hash"), "all four routes preserve complete effects, final identity and preceding products");
        }
        for (unsigned route = 0; route < 4; ++route) {
            ParamSnapshot params;
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.printProfileKey = "kodak_portra_endura";
            params.spectralUpsamplingMode = 0;
            params.cameraAutoExposureEnabled = 0;
            params.inputCompressionEnabled = 0;
            params.outputGamutCompressionEnabled = 0;
            params.grainControls.active = false;
            params.filmDustAmount = 1;
            params.filmScratchAmount = 2;
            params.gateDustAmount = 3;
            params.gateScratchAmount = 4;
            params.gateWeaveAmount = 0.1;
            InstanceState state;
            const auto pending = [&](const ParamSnapshot& value) {
                std::lock_guard<std::mutex> lock(state.pending.m);
                state.pending.value = PendingParamsState::Valid{value, hash_params(value)};
            };
            pending(params);
            const auto admitted = admit_pending_render_state(state);
            require(route % 2 ? bool(admitted.printState) : bool(admitted.directState), "effects publication baseline");
            const auto& held = route % 2 ? admitted.printState->recipe : admitted.directState->recipe;
            const auto hash = held.hash;
            const auto counter = state.buildCounterNext.load();
            auto changed = params;
            changed.gateWeaveAmount = 0.2;
            pending(changed);
            status(fj_test_film_recipe_fault(8, 4));
            const auto rejected = admit_pending_render_state(state);
            require(rejected.status == PendingRenderAdmissionStatus::RebuildFailed && rejected.diagnostic == "ResourceDescriptorMismatch phase=effects_recipe field=physical_policy" && !rejected.directState && !rejected.printState && state.buildCounterNext.load() == counter && held.hash == hash, "effects guard failure publishes no partial route and preserves admitted holds");
            pending(params);
            const auto recovered = admit_pending_render_state(state);
            require(route % 2 ? bool(recovered.printState) : bool(recovered.directState), "effects publication recovers after consumed fault");
            const auto repeated = admit_pending_render_state(state);
            require(repeated.directState == recovered.directState && repeated.printState == recovered.printState, "unchanged recovered effects reuses publication");
            auto invalidGrain = params;
            invalidGrain.grainControls.active = true;
            invalidGrain.grainControls.debugView = 3;
            invalidGrain.grainControls.particleAreaUm2 = 0;
            pending(invalidGrain);
            status(fj_test_film_recipe_fault(8, 4));
            const auto preceding = admit_pending_render_state(state);
            require(preceding.status == PendingRenderAdmissionStatus::RebuildFailed && preceding.diagnostic == "MalformedRequiredProfileData phase=9A field=visual_grain", "required grain failure precedes effects suppression or injected guard");
            auto badDir = params;
            badDir.dirCouplers.active = true;
            badDir.dirCouplers.amount = -1;
            pending(badDir);
            const auto blockedDir = admit_pending_render_state(state);
            require(blockedDir.status == PendingRenderAdmissionStatus::RebuildFailed && blockedDir.diagnostic == "ResourceDescriptorMismatch phase=effects_recipe field=physical_policy" && !blockedDir.directState && !blockedDir.printState, "unconsumed effects guard precedes competing DIR failure");
            status(fj_test_film_recipe_fault(8, 0));
            require(fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "effects failures drain temporary development/DIR holds");
        }
        std::puts("PASS 92 genuine physical effects products across four routes");
    }
} // namespace
namespace JuicerAssets::RecipeTest {
    void effects(FjPhysicalEffectsRecipe& recipe) {
        const auto fault = nextEffectsProjection;
        nextEffectsProjection = 0;
        if (fault == 1)
            recipe.active = 2;
        if (fault == 2)
            recipe.film_dust.cell_width_mm = std::numeric_limits<float>::quiet_NaN();
        if (fault == 3)
            recipe.gate_weave_amount = std::numeric_limits<double>::infinity();
        if (fault == 4)
            recipe.gate_scratch.strength_max = std::numeric_limits<float>::infinity();
        if (fault == 5 || fault == 6)
            recipe = effects_sentinel();
        if (fault == 6) {
            recipe.active = 0;
            recipe.gate_weave_amount = -0.0;
        }
    }
} // namespace JuicerAssets::RecipeTest
