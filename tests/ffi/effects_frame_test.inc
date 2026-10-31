namespace EffectsFrameTest {
    float scalar(const Json& v) {
        return std::bit_cast<float>(v.get<std::uint32_t>());
    }
    double wide(const Json& v) {
        return std::bit_cast<double>(v.get<std::uint64_t>());
    }
    Spektrafilm::FilmJuicerEffectsFrameExtent extent(const Json& v) {
        return {v.at(0), v.at(1), v.at(2), v.at(3)};
    }
    struct Input {
        FilmJuicerEffectsRecipe recipe;
        Spektrafilm::FilmJuicerEffectsFrameDescriptorInput frame;
        explicit Input(const Json& v) {
            frame = JuicerTestSupport::effects_frame_input(v, recipe);
        }
    };
    FjEffectsFrameInput product_input(const Json& v, const FjPhysicalEffectsRecipe& recipe) {
        const auto e = [](const Json& value) {
            return FjExtent{value.at(0), value.at(1), value.at(2), value.at(3)};
        };
        FjEffectsFrameInput out{};
        out.active = recipe.active;
        out.render_extent = e(v.at("render_extent"));
        out.pixel_definition = e(v.at("pixel_definition"));
        out.pixel_size_um = scalar(v.at("pixel_size_um"));
        out.film_format_long_edge_mm = scalar(v.at("film_format_long_edge_mm"));
        out.canonical_x = wide(v.at("canonical_x"));
        out.canonical_y = wide(v.at("canonical_y"));
        out.canonical_width = wide(v.at("canonical_width"));
        out.canonical_height = wide(v.at("canonical_height"));
        out.scale_x = wide(v.at("scale_x"));
        out.scale_y = wide(v.at("scale_y"));
        out.pixel_aspect_ratio = wide(v.at("pixel_aspect_ratio"));
        out.frame_time = wide(v.at("frame_time"));
        out.frame_rate = wide(v.at("frame_rate"));
        out.gate_weave_amount = recipe.gate_weave_amount;
        out.session_seed = v.at("session_seed");
        out.clip_token = v.at("clip_token");
        out.recipe_hash = recipe.hash;
        out.film_dust = recipe.film_dust;
        out.film_scratch = recipe.film_scratch;
        out.gate_dust = recipe.gate_dust;
        out.gate_scratch = recipe.gate_scratch;
        return out;
    }
    Json policy(const DefectDustRecipe& v) {
        return {{"cell_width_mm", std::bit_cast<std::uint32_t>(v.cellWidthMm)},
                {"cell_height_mm", std::bit_cast<std::uint32_t>(v.cellHeightMm)},
                {"slot_probability", std::bit_cast<std::uint32_t>(v.slotProbability)},
                {"softness_min_mm", std::bit_cast<std::uint32_t>(v.softnessMinMm)},
                {"softness_max_mm", std::bit_cast<std::uint32_t>(v.softnessMaxMm)},
                {"softness_size_cap_fraction", std::bit_cast<std::uint32_t>(v.softnessSizeCapFraction)},
                {"support_x_mm", std::bit_cast<std::uint32_t>(v.supportXMm)},
                {"support_y_mm", std::bit_cast<std::uint32_t>(v.supportYMm)},
                {"fiber_fraction", std::bit_cast<std::uint32_t>(v.fiberFraction)},
                {"fiber_drift_fraction", std::bit_cast<std::uint32_t>(v.fiberDriftFraction)},
                {"fiber_first_knot_min", std::bit_cast<std::uint32_t>(v.fiberFirstKnotMin)},
                {"fiber_first_knot_max", std::bit_cast<std::uint32_t>(v.fiberFirstKnotMax)},
                {"fiber_second_knot_min", std::bit_cast<std::uint32_t>(v.fiberSecondKnotMin)},
                {"fiber_second_knot_max", std::bit_cast<std::uint32_t>(v.fiberSecondKnotMax)},
                {"fiber_interior_width_min_fraction", std::bit_cast<std::uint32_t>(v.fiberInteriorWidthMinFraction)},
                {"fiber_interior_width_max_fraction", std::bit_cast<std::uint32_t>(v.fiberInteriorWidthMaxFraction)},
                {"diameter_min_mm", std::bit_cast<std::uint32_t>(v.diameterMinMm)},
                {"diameter_bulk_max_mm", std::bit_cast<std::uint32_t>(v.diameterBulkMaxMm)},
                {"diameter_max_mm", std::bit_cast<std::uint32_t>(v.diameterMaxMm)},
                {"diameter_tail_fraction", std::bit_cast<std::uint32_t>(v.diameterTailFraction)},
                {"fiber_length_min_mm", std::bit_cast<std::uint32_t>(v.fiberLengthMinMm)},
                {"fiber_length_max_mm", std::bit_cast<std::uint32_t>(v.fiberLengthMaxMm)},
                {"fiber_width_min_mm", std::bit_cast<std::uint32_t>(v.fiberWidthMinMm)},
                {"fiber_width_max_mm", std::bit_cast<std::uint32_t>(v.fiberWidthMaxMm)},
                {"opacity_faint_cumulative", std::bit_cast<std::uint32_t>(v.opacityFaintCumulative)},
                {"opacity_intermediate_cumulative", std::bit_cast<std::uint32_t>(v.opacityIntermediateCumulative)},
                {"compact_opacity_min", std::bit_cast<std::uint32_t>(v.compactOpacityMin)},
                {"compact_opacity_faint_end", std::bit_cast<std::uint32_t>(v.compactOpacityFaintEnd)},
                {"compact_opacity_intermediate_end", std::bit_cast<std::uint32_t>(v.compactOpacityIntermediateEnd)},
                {"compact_opacity_max", std::bit_cast<std::uint32_t>(v.compactOpacityMax)},
                {"fiber_opacity_min", std::bit_cast<std::uint32_t>(v.fiberOpacityMin)},
                {"fiber_opacity_faint_end", std::bit_cast<std::uint32_t>(v.fiberOpacityFaintEnd)},
                {"fiber_opacity_intermediate_end", std::bit_cast<std::uint32_t>(v.fiberOpacityIntermediateEnd)},
                {"fiber_opacity_max", std::bit_cast<std::uint32_t>(v.fiberOpacityMax)},
                {"compact_dominant_aspect_min", std::bit_cast<std::uint32_t>(v.compactDominantAspectMin)},
                {"compact_dominant_aspect_max", std::bit_cast<std::uint32_t>(v.compactDominantAspectMax)},
                {"compact_subsidiary_scale_min", std::bit_cast<std::uint32_t>(v.compactSubsidiaryScaleMin)},
                {"compact_subsidiary_scale_max", std::bit_cast<std::uint32_t>(v.compactSubsidiaryScaleMax)},
                {"compact_subsidiary_aspect_min", std::bit_cast<std::uint32_t>(v.compactSubsidiaryAspectMin)},
                {"compact_subsidiary_aspect_max", std::bit_cast<std::uint32_t>(v.compactSubsidiaryAspectMax)},
                {"compact_subsidiary_offset_max", std::bit_cast<std::uint32_t>(v.compactSubsidiaryOffsetMax)},
                {"compact_subsidiary_angle_max_radians", std::bit_cast<std::uint32_t>(v.compactSubsidiaryAngleMaxRadians)}};
    }
    Json policy(const DefectScratchRecipe& v) {
        return {{"cell_width_mm", std::bit_cast<std::uint32_t>(v.cellWidthMm)},
                {"cell_height_mm", std::bit_cast<std::uint32_t>(v.cellHeightMm)},
                {"slot_probability", std::bit_cast<std::uint32_t>(v.slotProbability)},
                {"softness_min_mm", std::bit_cast<std::uint32_t>(v.softnessMinMm)},
                {"softness_max_mm", std::bit_cast<std::uint32_t>(v.softnessMaxMm)},
                {"softness_size_cap_fraction", std::bit_cast<std::uint32_t>(v.softnessSizeCapFraction)},
                {"support_x_mm", std::bit_cast<std::uint32_t>(v.supportXMm)},
                {"support_y_mm", std::bit_cast<std::uint32_t>(v.supportYMm)},
                {"length_min_mm", std::bit_cast<std::uint32_t>(v.lengthMinMm)},
                {"length_bulk_max_mm", std::bit_cast<std::uint32_t>(v.lengthBulkMaxMm)},
                {"length_max_mm", std::bit_cast<std::uint32_t>(v.lengthMaxMm)},
                {"length_tail_fraction", std::bit_cast<std::uint32_t>(v.lengthTailFraction)},
                {"width_min_mm", std::bit_cast<std::uint32_t>(v.widthMinMm)},
                {"width_bulk_max_mm", std::bit_cast<std::uint32_t>(v.widthBulkMaxMm)},
                {"width_max_mm", std::bit_cast<std::uint32_t>(v.widthMaxMm)},
                {"width_tail_fraction", std::bit_cast<std::uint32_t>(v.widthTailFraction)},
                {"drift_fraction", std::bit_cast<std::uint32_t>(v.driftFraction)},
                {"first_knot_min", std::bit_cast<std::uint32_t>(v.firstKnotMin)},
                {"first_knot_max", std::bit_cast<std::uint32_t>(v.firstKnotMax)},
                {"second_knot_min", std::bit_cast<std::uint32_t>(v.secondKnotMin)},
                {"second_knot_max", std::bit_cast<std::uint32_t>(v.secondKnotMax)},
                {"interior_width_min_fraction", std::bit_cast<std::uint32_t>(v.interiorWidthMinFraction)},
                {"interior_width_max_fraction", std::bit_cast<std::uint32_t>(v.interiorWidthMaxFraction)},
                {"interior_depth_min_fraction", std::bit_cast<std::uint32_t>(v.interiorDepthMinFraction)},
                {"interior_depth_max_fraction", std::bit_cast<std::uint32_t>(v.interiorDepthMaxFraction)},
                {"endpoint_abrupt_probability", std::bit_cast<std::uint32_t>(v.endpointAbruptProbability)},
                {"interruption_probability", std::bit_cast<std::uint32_t>(v.interruptionProbability)},
                {"gap_center_min", std::bit_cast<std::uint32_t>(v.gapCenterMin)},
                {"gap_center_max", std::bit_cast<std::uint32_t>(v.gapCenterMax)},
                {"gap_span_min", std::bit_cast<std::uint32_t>(v.gapSpanMin)},
                {"gap_span_max", std::bit_cast<std::uint32_t>(v.gapSpanMax)},
                {"scuff_probability", std::bit_cast<std::uint32_t>(v.scuffProbability)},
                {"scuff_length_max_mm", std::bit_cast<std::uint32_t>(v.scuffLengthMaxMm)},
                {"scuff_angle_max_radians", std::bit_cast<std::uint32_t>(v.scuffAngleMaxRadians)},
                {"strength_min", std::bit_cast<std::uint32_t>(v.strengthMin)},
                {"strength_max", std::bit_cast<std::uint32_t>(v.strengthMax)}};
    }
    Json policy(const FjDust& v) {
        return {{"cell_width_mm", std::bit_cast<std::uint32_t>(v.cell_width_mm)},
                {"cell_height_mm", std::bit_cast<std::uint32_t>(v.cell_height_mm)},
                {"slot_probability", std::bit_cast<std::uint32_t>(v.slot_probability)},
                {"softness_min_mm", std::bit_cast<std::uint32_t>(v.softness_min_mm)},
                {"softness_max_mm", std::bit_cast<std::uint32_t>(v.softness_max_mm)},
                {"softness_size_cap_fraction", std::bit_cast<std::uint32_t>(v.softness_size_cap_fraction)},
                {"support_x_mm", std::bit_cast<std::uint32_t>(v.support_x_mm)},
                {"support_y_mm", std::bit_cast<std::uint32_t>(v.support_y_mm)},
                {"fiber_fraction", std::bit_cast<std::uint32_t>(v.fiber_fraction)},
                {"fiber_drift_fraction", std::bit_cast<std::uint32_t>(v.fiber_drift_fraction)},
                {"fiber_first_knot_min", std::bit_cast<std::uint32_t>(v.fiber_first_knot_min)},
                {"fiber_first_knot_max", std::bit_cast<std::uint32_t>(v.fiber_first_knot_max)},
                {"fiber_second_knot_min", std::bit_cast<std::uint32_t>(v.fiber_second_knot_min)},
                {"fiber_second_knot_max", std::bit_cast<std::uint32_t>(v.fiber_second_knot_max)},
                {"fiber_interior_width_min_fraction", std::bit_cast<std::uint32_t>(v.fiber_interior_width_min_fraction)},
                {"fiber_interior_width_max_fraction", std::bit_cast<std::uint32_t>(v.fiber_interior_width_max_fraction)},
                {"diameter_min_mm", std::bit_cast<std::uint32_t>(v.diameter_min_mm)},
                {"diameter_bulk_max_mm", std::bit_cast<std::uint32_t>(v.diameter_bulk_max_mm)},
                {"diameter_max_mm", std::bit_cast<std::uint32_t>(v.diameter_max_mm)},
                {"diameter_tail_fraction", std::bit_cast<std::uint32_t>(v.diameter_tail_fraction)},
                {"fiber_length_min_mm", std::bit_cast<std::uint32_t>(v.fiber_length_min_mm)},
                {"fiber_length_max_mm", std::bit_cast<std::uint32_t>(v.fiber_length_max_mm)},
                {"fiber_width_min_mm", std::bit_cast<std::uint32_t>(v.fiber_width_min_mm)},
                {"fiber_width_max_mm", std::bit_cast<std::uint32_t>(v.fiber_width_max_mm)},
                {"opacity_faint_cumulative", std::bit_cast<std::uint32_t>(v.opacity_faint_cumulative)},
                {"opacity_intermediate_cumulative", std::bit_cast<std::uint32_t>(v.opacity_intermediate_cumulative)},
                {"compact_opacity_min", std::bit_cast<std::uint32_t>(v.compact_opacity_min)},
                {"compact_opacity_faint_end", std::bit_cast<std::uint32_t>(v.compact_opacity_faint_end)},
                {"compact_opacity_intermediate_end", std::bit_cast<std::uint32_t>(v.compact_opacity_intermediate_end)},
                {"compact_opacity_max", std::bit_cast<std::uint32_t>(v.compact_opacity_max)},
                {"fiber_opacity_min", std::bit_cast<std::uint32_t>(v.fiber_opacity_min)},
                {"fiber_opacity_faint_end", std::bit_cast<std::uint32_t>(v.fiber_opacity_faint_end)},
                {"fiber_opacity_intermediate_end", std::bit_cast<std::uint32_t>(v.fiber_opacity_intermediate_end)},
                {"fiber_opacity_max", std::bit_cast<std::uint32_t>(v.fiber_opacity_max)},
                {"compact_dominant_aspect_min", std::bit_cast<std::uint32_t>(v.compact_dominant_aspect_min)},
                {"compact_dominant_aspect_max", std::bit_cast<std::uint32_t>(v.compact_dominant_aspect_max)},
                {"compact_subsidiary_scale_min", std::bit_cast<std::uint32_t>(v.compact_subsidiary_scale_min)},
                {"compact_subsidiary_scale_max", std::bit_cast<std::uint32_t>(v.compact_subsidiary_scale_max)},
                {"compact_subsidiary_aspect_min", std::bit_cast<std::uint32_t>(v.compact_subsidiary_aspect_min)},
                {"compact_subsidiary_aspect_max", std::bit_cast<std::uint32_t>(v.compact_subsidiary_aspect_max)},
                {"compact_subsidiary_offset_max", std::bit_cast<std::uint32_t>(v.compact_subsidiary_offset_max)},
                {"compact_subsidiary_angle_max_radians", std::bit_cast<std::uint32_t>(v.compact_subsidiary_angle_max_radians)}};
    }
    Json policy(const FjScratch& v) {
        return {{"cell_width_mm", std::bit_cast<std::uint32_t>(v.cell_width_mm)},
                {"cell_height_mm", std::bit_cast<std::uint32_t>(v.cell_height_mm)},
                {"slot_probability", std::bit_cast<std::uint32_t>(v.slot_probability)},
                {"softness_min_mm", std::bit_cast<std::uint32_t>(v.softness_min_mm)},
                {"softness_max_mm", std::bit_cast<std::uint32_t>(v.softness_max_mm)},
                {"softness_size_cap_fraction", std::bit_cast<std::uint32_t>(v.softness_size_cap_fraction)},
                {"support_x_mm", std::bit_cast<std::uint32_t>(v.support_x_mm)},
                {"support_y_mm", std::bit_cast<std::uint32_t>(v.support_y_mm)},
                {"length_min_mm", std::bit_cast<std::uint32_t>(v.length_min_mm)},
                {"length_bulk_max_mm", std::bit_cast<std::uint32_t>(v.length_bulk_max_mm)},
                {"length_max_mm", std::bit_cast<std::uint32_t>(v.length_max_mm)},
                {"length_tail_fraction", std::bit_cast<std::uint32_t>(v.length_tail_fraction)},
                {"width_min_mm", std::bit_cast<std::uint32_t>(v.width_min_mm)},
                {"width_bulk_max_mm", std::bit_cast<std::uint32_t>(v.width_bulk_max_mm)},
                {"width_max_mm", std::bit_cast<std::uint32_t>(v.width_max_mm)},
                {"width_tail_fraction", std::bit_cast<std::uint32_t>(v.width_tail_fraction)},
                {"drift_fraction", std::bit_cast<std::uint32_t>(v.drift_fraction)},
                {"first_knot_min", std::bit_cast<std::uint32_t>(v.first_knot_min)},
                {"first_knot_max", std::bit_cast<std::uint32_t>(v.first_knot_max)},
                {"second_knot_min", std::bit_cast<std::uint32_t>(v.second_knot_min)},
                {"second_knot_max", std::bit_cast<std::uint32_t>(v.second_knot_max)},
                {"interior_width_min_fraction", std::bit_cast<std::uint32_t>(v.interior_width_min_fraction)},
                {"interior_width_max_fraction", std::bit_cast<std::uint32_t>(v.interior_width_max_fraction)},
                {"interior_depth_min_fraction", std::bit_cast<std::uint32_t>(v.interior_depth_min_fraction)},
                {"interior_depth_max_fraction", std::bit_cast<std::uint32_t>(v.interior_depth_max_fraction)},
                {"endpoint_abrupt_probability", std::bit_cast<std::uint32_t>(v.endpoint_abrupt_probability)},
                {"interruption_probability", std::bit_cast<std::uint32_t>(v.interruption_probability)},
                {"gap_center_min", std::bit_cast<std::uint32_t>(v.gap_center_min)},
                {"gap_center_max", std::bit_cast<std::uint32_t>(v.gap_center_max)},
                {"gap_span_min", std::bit_cast<std::uint32_t>(v.gap_span_min)},
                {"gap_span_max", std::bit_cast<std::uint32_t>(v.gap_span_max)},
                {"scuff_probability", std::bit_cast<std::uint32_t>(v.scuff_probability)},
                {"scuff_length_max_mm", std::bit_cast<std::uint32_t>(v.scuff_length_max_mm)},
                {"scuff_angle_max_radians", std::bit_cast<std::uint32_t>(v.scuff_angle_max_radians)},
                {"strength_min", std::bit_cast<std::uint32_t>(v.strength_min)},
                {"strength_max", std::bit_cast<std::uint32_t>(v.strength_max)}};
    }
    Json product_rows(const std::filesystem::path& resources, const Json& inputs) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        Json rows = Json::array();
        for (unsigned route = 0; route < 4; ++route) {
            for (const auto& spec : inputs.at("products")) {
                ParamSnapshot params;
                params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                params.cameraFilmFormatLongEdgeMm = 36.0f;
                params.dirCouplers.active = false;
                params.grainControls.active = spec.at("grain_active");
                params.grainControls.debugView = spec.at("debug");
                params.filmDustAmount = scalar(spec.at("amounts")[0]);
                params.filmScratchAmount = scalar(spec.at("amounts")[1]);
                params.gateDustAmount = scalar(spec.at("amounts")[2]);
                params.gateScratchAmount = scalar(spec.at("amounts")[3]);
                params.gateWeaveAmount = wide(spec.at("weave"));
                auto product = JuicerTestSupport::prepare_product(params);
                FjPreparedStateView view{};
                FjErrorBuffer viewError{};
                status(product.view(view, viewError));
                const auto& effects = view.effects;
                Json literal = inputs.at("cases")[0];
                literal.update(spec.at("overrides"));
                literal["id"] = std::to_string(route) + "/" + spec.at("id").get<std::string>();
                literal["active"] = effects.active != 0;
                literal["recipe_hash"] = effects.hash;
                literal["gate_weave_amount"] = std::bit_cast<std::uint64_t>(effects.gate_weave_amount);
                literal["film_dust"] = policy(effects.film_dust);
                literal["film_scratch"] = policy(effects.film_scratch);
                literal["gate_dust"] = policy(effects.gate_dust);
                literal["gate_scratch"] = policy(effects.gate_scratch);
                const auto raw = product_input(literal, effects);
                FjEffectsFrameMath math{};
                FjEffectsFrameFailure failure{};
                std::array<char, 256> text{};
                FjErrorBuffer error{text.data(), text.size(), 0};
                const auto result = fj_test_effects_frame_prepare(&raw, &math, &failure, &error);
                Spektrafilm::FilmJuicerEffectsFrameDescriptor descriptor{};
                if (result.category == FJ_STATUS_SUCCESS) {
                    std::string diagnostic;
                    require(JuicerEffects::compose_frame_descriptor(math, effects.film_dust, effects.film_scratch, effects.gate_dust, effects.gate_scratch, descriptor, diagnostic), diagnostic.c_str());
                }
                rows.push_back({{"route", route}, {"spec", spec}, {"input", literal}, {"success", result.category == FJ_STATUS_SUCCESS}, {"diagnostic", std::string(text.data(), error.length)}, {"native", JuicerTestSupport::effects_frame_fields(descriptor)}});
            }
        }
        return rows;
    }
    FjEffectsFrameMath expected_math(const Json& row) {
        FjEffectsFrameMath out{};
        if (!row.at("success").get<bool>() || !row.at("input").at("active").get<bool>())
            return out;
        const auto& v = row.at("native");
        out.present = 1;
        out.flags = (v.at("weave_active").get<bool>() ? FJ_EFFECTS_WEAVE : 0u) | (v.at("film_active").get<bool>() ? FJ_EFFECTS_FILM : 0u) | (v.at("gate_transmittance_active").get<bool>() ? FJ_EFFECTS_GATE_TRANSMITTANCE : 0u) | (v.at("gate_output_active").get<bool>() ? FJ_EFFECTS_GATE_OUTPUT : 0u) | (v.at("requires_full_frame").get<bool>() ? FJ_EFFECTS_FULL_FRAME : 0u);
        const auto render = extent(v.at("render_extent")), full = extent(v.at("full_frame_extent"));
        out.render_extent = {render.x, render.y, render.width, render.height};
        out.full_frame_extent = {full.x, full.y, full.width, full.height};
        for (unsigned index = 0; index < 4; ++index) {
            const auto& o = v.at("origins")[index];
            out.origins[index] = {o.at(0), o.at(1), scalar(o.at(2)), scalar(o.at(3))};
        }
        out.sample_step_x_mm = scalar(v.at("sample_step_x_mm"));
        out.sample_step_y_mm = scalar(v.at("sample_step_y_mm"));
        out.weave_dx_px = scalar(v.at("weave_dx_px"));
        out.weave_dy_px = scalar(v.at("weave_dy_px"));
        out.weave_cos_rot = scalar(v.at("weave_cos_rot"));
        out.weave_sin_rot = scalar(v.at("weave_sin_rot"));
        out.roi_offset_x = v.at("roi_offset_x");
        out.roi_offset_y = v.at("roi_offset_y");
        out.gate_width = v.at("gate_width");
        out.gate_height = v.at("gate_height");
        out.session_seed = v.at("session_seed");
        out.clip_token = v.at("clip_token");
        out.recipe_hash = v.at("recipe_hash");
        out.hash = v.at("hash");
        return out;
    }
    void value_rows(const Json& rows) {
        const auto defaults = JuicerTestSupport::effects_frame_fields({});
        for (const auto& row : rows) {
            Input input(row.at("input"));
            const bool success = row.at("success");
            const auto expected = expected_math(row);
            const FjStatus envelope{success ? 0u : 3u, 0, 0};
            const FjEffectsFrameFailure failure{success ? 0u : input.recipe.active ? 3u
                                                                                   : 2u};
            FjEffectsFrameInput raw{};
            std::string diagnostic;
            if (JuicerEffects::project_input(input.frame, raw, diagnostic)) {
                for (unsigned facade = 0; facade < 2; ++facade)
                    require(fj_test_effects_frame_c_check(facade, &raw, &expected, envelope, failure, row.at("diagnostic").get<std::string>().c_str()) != 0, ("strict C11 full effects frame " + row.at("input").at("id").get<std::string>()).c_str());
            } else {
                require(!success && diagnostic == row.at("diagnostic").get<std::string>(), "required effects native pointer guard precedes inactive");
            }
            for (bool facade : {false, true}) {
                Spektrafilm::FilmJuicerEffectsFrameDescriptor out;
                const bool actual = facade ? JuicerTestSupport::prepare_effects_frame(input.frame, out, diagnostic) : JuicerEffects::prepare_frame(input.frame, out, diagnostic);
                require(actual == success && diagnostic == row.at("diagnostic").get<std::string>(), ("effects outcome " + row.at("input").at("id").get<std::string>()).c_str());
                require(JuicerTestSupport::effects_frame_fields(out) == (success ? row.at("native") : defaults), ("effects complete native fields/policies/hash " + row.at("input").at("id").get<std::string>()).c_str());
            }
        }
    }
    void values(const Json& captured, const Json& checked) {
        require(captured.at("cases").size() == 347 && captured.at("products").size() == 72, "effects complete context membership");
        Json ids = Json::array(), products = Json::array();
        for (const auto& row : captured.at("cases"))
            ids.push_back(row.at("input").at("id"));
        for (const auto& row : captured.at("products"))
            products.push_back(row.at("input").at("id"));
        require(ids == checked.at("case_ids") && products == checked.at("product_ids"), "independent ordered effects case membership");
        value_rows(captured.at("cases"));
        value_rows(captured.at("products"));
    }
    void raw_and_faults(const Json& captured) {
        Input input(captured.at("cases")[0].at("input"));
        FjEffectsFrameInput raw{};
        std::string diagnostic;
        require(JuicerEffects::project_input(input.frame, raw, diagnostic), "effects raw baseline input");
        const auto expected = expected_math(captured.at("cases")[0]);
        for (unsigned facade = 0; facade < 2; ++facade) {
            for (unsigned mode = 0; mode < 20; ++mode)
                require(fj_test_effects_frame_c_raw(facade, &raw, mode, &expected) != 0, ("C11 raw effects mode " + std::to_string(mode)).c_str());
        }
        for (unsigned fault = 1; fault <= 6; ++fault) {
            Input attempt(captured.at("cases")[fault == 6 ? 7 : 0].at("input"));
            if (fault == 6) {
                attempt.recipe.active = false;
                attempt.recipe.hash = 0;
            }
            status(fj_test_effects_frame_fault(fault));
            Spektrafilm::FilmJuicerEffectsFrameDescriptor out;
            require(!JuicerTestSupport::prepare_effects_frame(attempt.frame, out, diagnostic), "malformed completed effects result is unusable");
            require(diagnostic == (fault == 1 ? "physical effects frame preparation panicked" : "ResourceDescriptorMismatch phase=effects_descriptor field=rust_result") && JuicerTestSupport::effects_frame_fields(out) == JuicerTestSupport::effects_frame_fields({}), "effects panic/result fault clears native default");
            require(JuicerTestSupport::prepare_effects_frame(attempt.frame, out, diagnostic), "effects fault consumes once");
        }
        status(fj_test_effects_frame_fault(1));
        status(fj_test_effects_frame_fault(7), 2);
        FjEffectsFrameMath math{};
        FjEffectsFrameFailure failure{};
        status(fj_test_effects_frame_prepare(&raw, &math, &failure, nullptr), 8);
        require(failure.reason == 4, "unsupported fault preserves pending effects panic");
        status(fj_test_effects_frame_fault(1));
        status(fj_test_effects_frame_prepare(nullptr, &math, &failure, nullptr), 2);
        status(fj_test_effects_frame_prepare(&raw, &math, &failure, nullptr));
        require(fj_test_grain_frame_fault(1).category == 0, "independent grain fault arm");
        status(fj_test_effects_frame_prepare(&raw, &math, &failure, nullptr));
        FjGrainFrameInput grain{};
        FjGrainFrameMath grainMath{};
        FjGrainFrameFailure grainFailure{};
        status(fj_test_grain_frame_prepare(&grain, &grainMath, &grainFailure, nullptr), 8);
        require(grainFailure.reason == 3, "effects cannot consume grain faults");
        status(fj_test_effects_frame_fault(0));
    }

} // namespace EffectsFrameTest
