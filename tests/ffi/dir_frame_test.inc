namespace DirFrameTest {
    float scalar(const Json& input) {
        return std::bit_cast<float>(input.get<std::uint32_t>());
    }

    Spektrafilm::DirFrameExtent extent(const Json& input) {
        return {input.at(0), input.at(1), input.at(2), input.at(3)};
    }

    DirCouplersRecipe recipe(const Json& input) {
        DirCouplersRecipe out;
        out.active = input.at("active").get<bool>();
        out.diffusionSizeUm = scalar(input.at("size"));
        out.diffusionTailUm = scalar(input.at("tail"));
        out.diffusionTailWeight = scalar(input.at("weight"));
        out.hash = input.at("hash");
        return out;
    }

    Json replacement_case(const Json& input) {
        SpatialDirDescriptor out;
        const auto label = input.at("label").get<std::string>();
        const bool success = JuicerTestSupport::prepare_dir_frame(
            recipe(input), scalar(input.at("pixel")), extent(input.at("render")), extent(input.at("full")), label.c_str(), out);
        return {{"input", input}, {"success", success}, {"native", JuicerTestSupport::dir_frame_fields(out)}};
    }

    Json products(const std::filesystem::path& resources, const Json& inputs) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        Json rows = Json::array();
        for (unsigned route = 0; route < 4; ++route) {
            ParamSnapshot params;
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.grainControls.active = false;
            params.dirCouplers.diffusionSizeUm = 20.0f;
            params.dirCouplers.diffusionTailUm = 200.0f;
            params.dirCouplers.diffusionTailWeight = 0.03f;
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            require(route % 2 ? build_print_render_state_product(params, product, diagnostic)
                              : build_direct_render_state_product(params, product, diagnostic),
                    diagnostic.c_str());
            for (const auto& spec : inputs) {
                const auto& dir = product.recipe.dirCouplers;
                Json input{{"name", std::to_string(route) + "/" + spec.at("name").get<std::string>()},
                           {"active", dir.active},
                           {"size", std::bit_cast<std::uint32_t>(dir.diffusionSizeUm)},
                           {"tail", std::bit_cast<std::uint32_t>(dir.diffusionTailUm)},
                           {"weight", std::bit_cast<std::uint32_t>(dir.diffusionTailWeight)},
                           {"hash", dir.hash},
                           {"pixel", spec.at("pixel")},
                           {"render", spec.at("render")},
                           {"full", spec.at("full")},
                           {"label", route % 2 ? "print" : "direct"}};
                auto row = replacement_case(input);
                row["route"] = route;
                row["recipe_hash"] = product.recipe.hash;
                rows.push_back(row);
            }
        }
        return rows;
    }

    FjDirFrameInput raw_input(const Json& input) {
        const auto render = extent(input.at("render")), full = extent(input.at("full"));
        return {input.at("active").get<bool>() ? 1u : 0u,
                scalar(input.at("size")),
                scalar(input.at("tail")),
                scalar(input.at("weight")),
                scalar(input.at("pixel")),
                {render.x, render.y, render.width, render.height},
                {full.x, full.y, full.width, full.height},
                input.at("hash")};
    }

    Json math_words(const FjDirFrameMath& v) {
        Json out = Json::array({v.present, v.component_count, v.recipe_hash});
        for (const auto& e : {v.render_extent, v.full_frame_extent, v.filter_domain_extent})
            for (int value : {e.x, e.y, e.width, e.height})
                out.push_back(std::bit_cast<std::uint32_t>(value));
        for (const auto& c : v.components) {
            out.push_back(std::bit_cast<std::uint32_t>(c.sigma_px));
            out.push_back(std::bit_cast<std::uint32_t>(c.weight));
            out.push_back(std::bit_cast<std::uint32_t>(c.radius));
            out.push_back(c.reference_operator);
            out.push_back(std::bit_cast<std::uint64_t>(c.feedforward));
            for (double value : c.feedback)
                out.push_back(std::bit_cast<std::uint64_t>(value));
            out.push_back(std::bit_cast<std::uint64_t>(c.normalization_denominator));
            out.push_back(std::bit_cast<std::uint64_t>(c.feedforward_numerator));
            for (double value : c.feedback_numerators)
                out.push_back(std::bit_cast<std::uint64_t>(value));
            out.push_back(std::bit_cast<std::uint64_t>(c.boundary_truncation_accuracy));
            out.push_back(std::bit_cast<std::uint64_t>(c.boundary_certification_tolerance));
            out.push_back(c.boundary_derivation_version);
        }
        require(out.size() == 79, "all 79 DIR mathematical semantic slots");
        return out;
    }

    FjDirFrameMath expected_math(const Json& native) {
        FjDirFrameMath out{};
        if (native.at("support") == 1)
            return out;
        out.present = 1;
        out.component_count = native.at("count");
        out.recipe_hash = native.at("recipe_hash");
        const auto copy_extent = [](const Json& v) -> FjExtent {
            return {v.at(0), v.at(1), v.at(2), v.at(3)};
        };
        out.render_extent = copy_extent(native.at("render"));
        out.full_frame_extent = copy_extent(native.at("full"));
        out.filter_domain_extent = copy_extent(native.at("filter"));
        for (std::size_t i = 0; i < 4; ++i) {
            const auto& c = native.at("components")[i];
            auto& v = out.components[i];
            v.sigma_px = scalar(c.at("sigma"));
            v.weight = scalar(c.at("weight"));
            v.radius = c.at("radius");
            v.reference_operator = c.at("operator");
            const auto f64 = [](const Json& word) {
                return std::bit_cast<double>(word.get<std::uint64_t>());
            };
            v.feedforward = f64(c.at("feedforward"));
            for (std::size_t j = 0; j < 3; ++j) {
                v.feedback[j] = f64(c.at("feedback")[j]);
                v.feedback_numerators[j] = f64(c.at("numerators")[j]);
            }
            v.normalization_denominator = f64(c.at("denominator"));
            v.feedforward_numerator = f64(c.at("numerator"));
            v.boundary_truncation_accuracy = f64(c.at("accuracy"));
            v.boundary_certification_tolerance = f64(c.at("tolerance"));
            v.boundary_derivation_version = c.at("version");
        }
        return out;
    }

    void values(const Json& captured, const Json& checked) {
        require(captured.at("cases").size() == 74 && captured.at("products").size() == 24, "complete independent DIR cohort");
        const std::array fieldNames{"", "pixel_size_um", "render_extent", "full_frame_extent", "partial_render_window", "core", "tail", "component_values", "component_capacity", "radius", "iir_denominator", "iir_output", "empty_plan"};
        for (const auto* rows : {&captured.at("cases"), &captured.at("products")}) {
            for (const auto& row : *rows) {
                const auto input = raw_input(row.at("input"));
                const bool success = row.at("success");
                FjDirFrameMath expected{};
                FjDirFrameFailure rejected{0, 0, 4, {}};
                std::string diagnostic;
                if (success) {
                    expected = expected_math(row.at("native"));
                } else {
                    const auto& details = checked.at("ordinary").at(row.at("input").at("name").get<std::string>());
                    rejected.reason = 2;
                    rejected.field = details[0];
                    rejected.component = details[1];
                    rejected.trace = expected_math(row.at("native"));
                    diagnostic = std::string("SpatialDirFrameFailure field=") + fieldNames[rejected.field];
                }
                for (auto call : {fj_legacy_dir_frame_prepare, fj_test_dir_frame_prepare, fj_test_dir_frame_c_prepare}) {
                    FjDirFrameMath math;
                    FjDirFrameFailure failure;
                    std::memset(&math, 0xa5, sizeof math);
                    std::memset(&failure, 0xa5, sizeof failure);
                    std::array<char, 256> text{};
                    FjErrorBuffer error{text.data(), text.size(), 99};
                    const auto result = call(&input, &math, &failure, &error);
                    status(result, success ? 0 : 3);
                    require(math_words(math) == math_words(expected), "every raw mathematical slot matches independent native capture");
                    require(failure.reason == rejected.reason && failure.field == rejected.field && failure.component == rejected.component &&
                                math_words(failure.trace) == math_words(rejected.trace),
                            "typed failure and complete diagnostic progress");
                    require(error.length == diagnostic.size() && text[error.length] == 0 && std::string_view(text.data(), error.length) == diagnostic, "closed DIR diagnostic");
                    const auto outcome = JuicerDir::MathOutcome::admit(input, result, math, failure, error, text);
                    SpatialDirDescriptor composed;
                    const auto label = row.at("input").at("label").get<std::string>();
                    require(Spektrafilm::compose_spatial_dir_descriptor(outcome, label.c_str(), composed) == success && JuicerTestSupport::dir_frame_fields(composed) == row.at("native"), "complete native composer, defaults, transient tiers and final hash");
                }
                require(fj_test_dir_frame_c_check(&input, &expected, &rejected, success ? 0u : 3u, diagnostic.c_str()), "C11 reader checks all fields and failure trace");
                require(replacement_case(row.at("input")) == Json{{"input", row.at("input")}, {"success", row.at("success")}, {"native", row.at("native")}}, "facade and common native composer");
                const auto outcome = JuicerDir::prepare_math(recipe(row.at("input")), scalar(row.at("input").at("pixel")), extent(row.at("input").at("render")), extent(row.at("input").at("full")));
                SpatialDirDescriptor composed;
                const auto label = row.at("input").at("label").get<std::string>();
                require(Spektrafilm::compose_spatial_dir_descriptor(outcome, label.c_str(), composed) == success && JuicerTestSupport::dir_frame_fields(composed) == row.at("native"), "production bridge and sole native hash");
            }
        }
    }

    void boundary(const Json& rows) {
        const auto input = raw_input(rows[0].at("input"));
        const auto zero = math_words(FjDirFrameMath{});
        for (auto call : {fj_legacy_dir_frame_prepare, fj_test_dir_frame_prepare, fj_test_dir_frame_c_prepare}) {
            for (unsigned mode = 0; mode < 10; ++mode) {
                FjDirFrameMath math;
                FjDirFrameFailure failure;
                std::memset(&math, 0xa5, sizeof math);
                std::memset(&failure, 0xa5, sizeof failure);
                std::array<char, 256> text{};
                FjErrorBuffer error{text.data(), text.size(), 99};
                auto bad = input;
                if (mode == 0)
                    bad.active = 2;
                if (mode == 8)
                    error = {nullptr, 1, 99};
                if (mode == 9)
                    error = {text.data(), SIZE_MAX, 99};
                const auto result = call(mode == 1 ? nullptr : mode == 2 ? reinterpret_cast<const FjDirFrameInput*>(std::uintptr_t{1})
                                                                         : &bad,
                                         mode == 3 ? nullptr : mode == 4 ? reinterpret_cast<FjDirFrameMath*>(std::uintptr_t{1})
                                                                         : &math,
                                         mode == 5 ? nullptr : mode == 6 ? reinterpret_cast<FjDirFrameFailure*>(std::uintptr_t{1})
                                                                         : &failure,
                                         mode == 7 ? reinterpret_cast<FjErrorBuffer*>(std::uintptr_t{1}) : &error);
                status(result, 2);
                if (mode != 3 && mode != 4)
                    require(math_words(math) == zero, "independent raw Math clearing");
                if (mode != 5 && mode != 6)
                    require(failure.reason == 1 && failure.field == 0 && failure.component == 4 && math_words(failure.trace) == zero, "independent raw failure clearing");
                if (mode < 7)
                    require(std::string(text.data()) == "invalid spatial DIR frame arguments", "raw diagnostic text");
            }
            auto invalid = input;
            invalid.pixel_size_um = 0;
            for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{8}, std::size_t{256}}) {
                FjDirFrameMath math{};
                FjDirFrameFailure failure{};
                std::array<char, 256> text{};
                FjErrorBuffer error{capacity ? text.data() : reinterpret_cast<char*>(std::uintptr_t{1}), capacity, 99};
                status(call(&invalid, &math, &failure, &error), 3);
                const std::string_view diagnostic = "SpatialDirFrameFailure field=pixel_size_um";
                const auto count = std::min(diagnostic.size(), capacity ? capacity - 1 : 0);
                require(error.length == count && (!capacity || (text[count] == 0 && std::string_view(text.data(), count) == diagnostic.substr(0, count))) &&
                            failure.reason == 2 && failure.field == 1 && failure.component == 4 && math_words(math) == zero && math_words(failure.trace) == zero,
                        "bounded diagnostic preserves ordinary envelope and clearing");
            }
        }
        for (unsigned fault = 1; fault <= 6; ++fault) {
            const auto& source = fault == 6 ? rows.at(1) : rows.at(0);
            auto literal = recipe(source.at("input"));
            float pixel = scalar(source.at("input").at("pixel"));
            if (fault == 6)
                pixel = 0;
            status(fj_test_dir_frame_fault(fault));
            const auto result = JuicerDir::prepare_math(literal, pixel, extent(source.at("input").at("render")), extent(source.at("input").at("full")));
            SpatialDirDescriptor out;
            require(!Spektrafilm::compose_spatial_dir_descriptor(result, "direct", out) && JuicerTestSupport::dir_frame_fields(out) == JuicerTestSupport::dir_frame_fields(SpatialDirDescriptor{}), "boundary corruption/panic never becomes executable progress");
            require(std::holds_alternative<JuicerDir::MathOutcome::BoundaryFailure>(result.result()), "closed boundary failure kind");
            const auto& failure = std::get<JuicerDir::MathOutcome::BoundaryFailure>(result.result()).failure;
            require(failure.status.category == (fault == 1 ? 8u : 3u), "panic versus corrupt projection classification");
            const auto recovered = JuicerDir::prepare_math(recipe(rows[0].at("input")), 1.0f, {0, 0, 64, 48}, {0, 0, 64, 48});
            require(Spektrafilm::compose_spatial_dir_descriptor(recovered, "direct", out), "fault consumed once");
        }
        status(fj_test_dir_frame_fault(1));
        status(fj_test_dir_frame_fault(7), 2);
        FjDirFrameMath math{};
        FjDirFrameFailure failure{};
        auto invalid = input;
        invalid.active = 2;
        status(fj_legacy_dir_frame_prepare(&invalid, &math, &failure, nullptr), 2);
        std::thread other([&] {
            status(fj_legacy_dir_frame_prepare(&input, &math, &failure, nullptr));
        });
        other.join();
        status(fj_legacy_dir_frame_prepare(&input, &math, &failure, nullptr), 8);
        status(fj_legacy_dir_frame_prepare(&input, &math, &failure, nullptr));
        for (unsigned fault = 2; fault <= 6; ++fault) {
            auto inactive = input;
            inactive.active = 0;
            status(fj_test_dir_frame_fault(fault));
            status(fj_legacy_dir_frame_prepare(&inactive, &math, &failure, nullptr));
            require(math_words(math) == zero && failure.reason == 0, "inapplicable fault consumes without manufacturing work/failure");
            status(fj_legacy_dir_frame_prepare(&input, &math, &failure, nullptr));
        }
        status(fj_test_dir_frame_fault(1));
        status(fj_test_dir_frame_fault(0));
        status(fj_legacy_dir_frame_prepare(&input, &math, &failure, nullptr));
    }
} // namespace DirFrameTest
