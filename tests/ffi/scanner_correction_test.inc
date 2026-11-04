// Frozen native fixtures exercise the checked boundary and actual Scanner callers.
namespace ScannerCorrectionTest {
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    constexpr float kInfinity = std::numeric_limits<float>::infinity();

    Json primitives() {
        Json rows = Json::array();
        const std::array values{-kInfinity, -309.0f, -38.0f, -1.0f, -0.0f, 0.0f, std::numeric_limits<float>::denorm_min(), 1.0e-10f, .04045f, std::nextafter(.04045f, 1.0f), .1234567f, .184f, .99999994f, 1.0f, 2.4f, 38.0f, 100.0f, 308.0f, 324.0f, kInfinity, kNan};
        for (float value : values)
            for (float accumulator : {0.0f, .1234567f, -1.0f}) {
                const float multiplier = .7777777f;
                const float clamped = std::clamp(value, 0.0f, 1.0f);
                const std::array native{std::pow(10.0f, -value), std::log10(value), std::fmax(value, 0.0f), clamped, accumulator + value * multiplier, clamped <= .04045f ? clamped / 12.92f : std::pow((clamped + .055f) / 1.055f, 2.4f)};
                const std::array nativeDouble{std::pow(10.0, -static_cast<double>(value)),
                                              std::log10(static_cast<double>(value)),
                                              std::fmax(static_cast<double>(value), 0.0)};
                std::array<float, 6> rust{};
                std::array<double, 3> rustDouble{};
                fj_test_scanner_correction_primitive(value, accumulator, multiplier, rust.data(), rustDouble.data());
                rows.push_back({{"value", std::bit_cast<std::uint32_t>(value)},
                                {"accumulator", std::bit_cast<std::uint32_t>(accumulator)},
                                {"multiplier", std::bit_cast<std::uint32_t>(multiplier)},
                                {"native", words(native.data(), native.size())},
                                {"rust", words(rust.data(), rust.size())},
                                {"native_double", PrintPreflashTest::double_words(nativeDouble.data(), nativeDouble.size())},
                                {"rust_double", PrintPreflashTest::double_words(rustDouble.data(), rustDouble.size())}});
            }
        return rows;
    }

    Json project(const FjScannerColorCorrection& value);

    Json products(const std::filesystem::path& resources, const Json& expectedProducts) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        auto* assets = JuicerProcess::root().assets().borrow_rust_assets();
        require(assets != nullptr, "scanner correction products have process Rust assets");
        Json out = Json::array();
        std::vector<std::string> recipe_ids;
        for (unsigned route = 0; route < 4; ++route)
            for (int mode = 0; mode < 3; ++mode)
                for (unsigned flags = 0; flags < 4; ++flags)
                    recipe_ids.emplace_back(std::to_string(route) + "/" + std::to_string(mode) + "/" + std::to_string(flags));
        JuicerTests::Cohort recipe_rows("scanner_correction.recipe_tuple", recipe_ids);
        for (unsigned route = 0; route < 4; ++route)
            for (int mode = 0; mode < 3; ++mode)
                for (unsigned flags = 0; flags < 4; ++flags) {
                    auto case_event = recipe_rows.case_event(std::to_string(route) + "/" + std::to_string(mode) + "/" + std::to_string(flags));
                    auto params = PrintPreflashTest::controls();
                    params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                    params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                    params.spectralUpsamplingMode = mode;
                    params.scannerBlackCorrection = (flags & 1u) != 0;
                    params.scannerWhiteCorrection = (flags & 2u) != 0;
                    auto stateOwner = JuicerTestSupport::prepare_product(params);
                    FjPreparedStateView state{};
                    std::array<char, 2048> errorText{};
                    FjErrorBuffer error{errorText.data(), errorText.size(), 0};
                    status(stateOwner.view(state, error));
                    const bool printRoute = route % 2 != 0;

                    // Source-profile bytes remain owned by the live asset views;
                    // the complete prepared state supplies resolved execution data.
                    FjFilmProfile* filmOwner = nullptr;
                    FjPrintProfile* printOwner = nullptr;
                    status(fj_legacy_film_profile_acquire(assets, state.film_profile_key, &filmOwner, &error));
                    FjFilmProfileView film{};
                    FjPrintProfileView paper{};
                    status(fj_legacy_film_profile_view(filmOwner, &film, &error));
                    require(film.asset_token == state.film_profile_asset_version,
                            "correction product film view matches held-state asset version");
                    if (printRoute) {
                        status(fj_legacy_print_profile_acquire(assets, state.print_profile_key, &printOwner, &error));
                        status(fj_legacy_print_profile_view(printOwner, &paper, &error));
                        require(paper.asset_token == state.print_profile_asset_version,
                                "correction product print view matches held-state asset version");
                    }

                    const auto& tables = state.scanner_tables;
                    require(tables.lambda_nm.count == 81 && tables.illuminant.count == 81 &&
                                tables.weighted_xyz[1].count == 81 && tables.dyes_cmy[0].count == 81 &&
                                tables.dyes_cmy[1].count == 81 && tables.dyes_cmy[2].count == 81,
                            "held scanner correction table extents");
                    Json tableInput{{"c", words(tables.dyes_cmy[0].data, tables.dyes_cmy[0].count)},
                                    {"m", words(tables.dyes_cmy[1].data, tables.dyes_cmy[1].count)},
                                    {"y", words(tables.dyes_cmy[2].data, tables.dyes_cmy[2].count)},
                                    {"ay", words(tables.weighted_xyz[1].data, tables.weighted_xyz[1].count)},
                                    {"baseline", words(tables.baseline_min.data, tables.baseline_min.count)},
                                    {"k", tables.lambda_nm.count},
                                    {"has_baseline", tables.has_baseline != 0},
                                    {"inv_yn", std::bit_cast<std::uint32_t>(tables.inv_yn)}};
                    const std::array<float, 81> emptyLight{};
                    const auto lightWords = printRoute
                                                ? words(state.print.main_illuminant.data, state.print.main_illuminant.count)
                                                : words(emptyLight.data(), emptyLight.size());
                    Json input{{"route", state.route},
                               {"flags", unsigned(state.scanner_policy.black_correction) + 2u * unsigned(state.scanner_policy.white_correction)},
                               {"levels", words(std::array{state.scanner_policy.black_level, state.scanner_policy.white_level}.data(), 2)},
                               {"medium", state.scanner_bounds.medium},
                               {"minimum", words(state.scanner_bounds.minimum_cmy, 3)},
                               {"maximum", words(state.scanner_bounds.maximum_cmy, 3)},
                               {"inverse", words(state.scanner_bounds.inv_span_cmy, 3)},
                               {"tables", std::move(tableInput)},
                               {"film_base", words(film.tables.base_density.data, film.tables.base_density.count)},
                               {"film_axis", words(film.tables.log_exposure.data, film.tables.log_exposure.count)},
                               {"film_dyes", words(film.tables.channel_density_cmy.data, film.tables.channel_density_cmy.count)},
                               {"authored_curves", words(film.tables.density_curves_cmy.data, film.tables.density_curves_cmy.count)},
                               {"baseline_density", words(state.film_reference.baseline_density_cmy, 3)},
                               {"reference_hash", state.film_reference.hash},
                               {"light", lightWords},
                               {"preflash", words(state.print.preflash_raw_cmy, 3)},
                               {"normalizer", std::bit_cast<std::uint32_t>(printRoute ? state.print.normalizer : 1.0f)},
                               {"preflash_exposure", std::bit_cast<std::uint32_t>(printRoute ? state.print.preflash_exposure : 0.0f)},
                               {"film_minimum", words(state.enlarger_film_bounds.minimum_cmy, 3)},
                               {"film_maximum", words(state.enlarger_film_bounds.maximum_cmy, 3)}};
                    if (printRoute) {
                        input["print_base"] = words(paper.tables.base_density.data, paper.tables.base_density.count);
                        input["print_axis"] = words(paper.tables.log_exposure.data, paper.tables.log_exposure.count);
                        input["print_curves"] = words(paper.tables.density_curves_cmy.data, paper.tables.density_curves_cmy.count);
                        input["sensitivity"] = words(paper.tables.linear_sensitivity_rgb.data, paper.tables.linear_sensitivity_rgb.count);
                    }
                    const auto capturedCorrection = [&] {
                        const auto descriptor = project(state.scanner_correction);
                        return Json{{"native_outcome", "returned"}, {"native_bool", true}, {"exception_kind", nullptr}, {"descriptor", descriptor}, {"diagnostic", ""}};
                    }();
                    out.push_back({{"id", std::to_string(route) + ":" + std::to_string(mode) + ":" + std::to_string(flags)},
                                   {"mode", mode},
                                   {"print", route % 2 != 0},
                                   {"input", std::move(input)},
                                   {"native", capturedCorrection},
                                   {"snapshot_hash", hash_params(params)},
                                   {"recipe_hash", state.recipe_hash},
                                   {"scanner_hash", state.scanner_recipe_hash},
                                   {"upload_hash", state.upload_core_hash}});
                    if (printOwner)
                        status(fj_legacy_print_profile_release(printOwner, &error));
                    status(fj_legacy_film_profile_release(filmOwner, &error));
                    require(out.back() == expectedProducts.at(out.size() - 1), "per-case complete scanner correction product");
                    case_event.passed();
                }
        return out;
    }

} // namespace ScannerCorrectionTest

namespace ScannerCorrectionTest {
    std::vector<float> floats(const Json& value) {
        std::vector<float> out;
        for (const auto& word : value)
            out.push_back(std::bit_cast<float>(word.get<std::uint32_t>()));
        return out;
    }
    struct Inputs {
        FjScannerCorrectionParams params{};
        std::vector<float> c, m, y, ay, baseline, filmAxis, authored, printAxis, printCurves;
        std::array<float, 81> filmBase{}, printBase{}, light{};
        std::array<float, 243> dyes{}, sensitivity{};
        std::array<float, 3> minimum{}, maximum{}, inverse{}, reference{}, filmMinimum{}, filmMaximum{}, preflash{};
        int count = 0;
        float invYn = 0, normalizer = 0, preflashExposure = 0;
        std::uint32_t medium = 0, hasBaseline = 0;
        std::uint64_t referenceHash = 0;
        explicit Inputs(const Json& input) {
            const auto levels = PrintPreflashTest::array<2>(input.at("levels"));
            params = {input.at("route"), input.at("flags"), levels[0], levels[1]};
            const auto& table = input.at("tables");
            c = floats(table.at("c"));
            m = floats(table.at("m"));
            y = floats(table.at("y"));
            ay = floats(table.at("ay"));
            baseline = floats(table.at("baseline"));
            count = table.at("k");
            hasBaseline = table.at("has_baseline").get<bool>() ? 1u : 0u;
            invYn = std::bit_cast<float>(table.at("inv_yn").get<std::uint32_t>());
            if (input.contains("film_base")) {
                filmBase = PrintPreflashTest::array<81>(input.at("film_base"));
                filmAxis = floats(input.at("film_axis"));
                dyes = PrintPreflashTest::array<243>(input.at("film_dyes"));
            }
            authored = floats(input.at("authored_curves"));
            if (input.contains("print_base")) {
                printBase = PrintPreflashTest::array<81>(input.at("print_base"));
                printAxis = floats(input.at("print_axis"));
                printCurves = floats(input.at("print_curves"));
                sensitivity = PrintPreflashTest::array<243>(input.at("sensitivity"));
            }
            minimum = PrintPreflashTest::array<3>(input.at("minimum"));
            maximum = PrintPreflashTest::array<3>(input.at("maximum"));
            inverse = PrintPreflashTest::array<3>(input.at("inverse"));
            reference = PrintPreflashTest::array<3>(input.at("baseline_density"));
            filmMinimum = PrintPreflashTest::array<3>(input.at("film_minimum"));
            filmMaximum = PrintPreflashTest::array<3>(input.at("film_maximum"));
            preflash = PrintPreflashTest::array<3>(input.at("preflash"));
            light = PrintPreflashTest::array<81>(input.at("light"));
            normalizer = std::bit_cast<float>(input.at("normalizer").get<std::uint32_t>());
            preflashExposure = std::bit_cast<float>(input.at("preflash_exposure").get<std::uint32_t>());
            referenceHash = input.at("reference_hash");
            medium = input.at("medium");
        }
        FjScannerCorrectionTables tables() const {
            return {{c.data(), c.size()}, {m.data(), m.size()}, {y.data(), y.size()}, {ay.data(), ay.size()}, {baseline.data(), baseline.size()}, count, invYn, hasBaseline};
        }
        FjScannerDirectCorrectionInput direct(const FjScannerCorrectionTables& tables) const {
            FjScannerDirectCorrectionInput out{&tables, {filmBase.data(), 81}, {filmAxis.data(), filmAxis.size()}, {authored.data(), authored.size()}, {}, {}, {}, {}, referenceHash, medium};
            std::copy(minimum.begin(), minimum.end(), out.minimum_density_cmy);
            std::copy(maximum.begin(), maximum.end(), out.maximum_density_cmy);
            std::copy(inverse.begin(), inverse.end(), out.inverse_span_cmy);
            std::copy(reference.begin(), reference.end(), out.baseline_density_cmy);
            return out;
        }
        FjScannerPrintCorrectionInput print(const FjScannerCorrectionTables& tables) const {
            FjScannerPrintCorrectionInput out{&tables, {filmBase.data(), 81}, {dyes.data(), 243}, {sensitivity.data(), 243}, {light.data(), 81}, {printBase.data(), 81}, {printAxis.data(), printAxis.size()}, {printCurves.data(), printCurves.size()}, {}, {}, {}, {}, {}, normalizer, preflashExposure, medium};
            std::copy(minimum.begin(), minimum.end(), out.minimum_density_cmy);
            std::copy(inverse.begin(), inverse.end(), out.inverse_span_cmy);
            std::copy(filmMinimum.begin(), filmMinimum.end(), out.film_minimum_density_cmy);
            std::copy(filmMaximum.begin(), filmMaximum.end(), out.film_maximum_density_cmy);
            std::copy(preflash.begin(), preflash.end(), out.preflash_raw_cmy);
            return out;
        }
    };
    Json project(const FjScannerColorCorrection& value) {
        const std::array values{value.target_black_linear, value.target_white_linear, value.reference_black_y, value.reference_white_y, value.xyz_slope, value.xyz_offset, value.exposure_scale};
        return {{"route", value.route}, {"active", value.active != 0}, {"black", value.black_correction != 0}, {"white", value.white_correction != 0}, {"words", words(values.data(), values.size())}, {"schema", value.schema_version}, {"hash", value.hash}};
    }
    FjScannerColorCorrection expected(const Json& value) {
        const auto v = PrintPreflashTest::array<7>(value.at("words"));
        return {value.at("route"), value.at("active").get<bool>() ? 1u : 0u, value.at("black").get<bool>() ? 1u : 0u, value.at("white").get<bool>() ? 1u : 0u, v[0], v[1], v[2], v[3], v[4], v[5], v[6], value.at("schema"), value.at("hash")};
    }
    Json trace_words(const FjScannerCorrectionTrace& value) {
        const std::array values{value.reference_black_y, value.reference_white_y, value.target_black_linear, value.target_white_linear, value.xyz_slope, value.xyz_offset, value.corrected_midgray, value.exposure_scale};
        return words(values.data(), values.size());
    }
    struct Outcome {
        FjStatus status{};
        FjScannerColorCorrection value{};
        FjScannerCorrectionFailure failure{};
        FjScannerCorrectionTrace trace{};
        std::array<char, 512> text{};
        std::size_t length = 0;
        Outcome() {
            std::memset(&value, 0x55, sizeof(value));
            std::memset(&trace, 0x55, sizeof(trace));
            failure.reason = 99;
            text.fill('x');
        }
    };
    using Direct = decltype(&fj_legacy_scanner_direct_correction_prepare);
    using Print = decltype(&fj_legacy_scanner_print_correction_prepare);
    Direct direct_entry(bool facade) {
        return facade ? fj_test_scanner_direct_correction_prepare : fj_legacy_scanner_direct_correction_prepare;
    }
    Print print_entry(bool facade) {
        return facade ? fj_test_scanner_print_correction_prepare : fj_legacy_scanner_print_correction_prepare;
    }
    Outcome invoke(const Inputs& source, bool print, bool facade, bool missing = false) {
        Outcome out;
        const auto tables = source.tables();
        const auto direct = source.direct(tables);
        const auto paper = source.print(tables);
        FjErrorBuffer error{out.text.data(), out.text.size(), 99};
        out.status = print ? print_entry(facade)(&source.params, missing ? nullptr : &paper, &out.value, &out.failure, &out.trace, &error)
                           : direct_entry(facade)(&source.params, missing ? nullptr : &direct, &out.value, &out.failure, &out.trace, &error);
        out.length = error.length;
        require(error.data == out.text.data() && error.capacity == out.text.size() && out.length < out.text.size() && out.text[out.length] == '\0', "scanner bounded diagnostics");
        return out;
    }
    std::uint32_t expected_trace(std::uint32_t reason) {
        if (reason == 5 || reason == 10)
            return FJ_SCANNER_CORRECTION_TRACE_MIDGRAY;
        if (reason == 12)
            return FJ_SCANNER_CORRECTION_TRACE_MALFORMED;
        if (reason == 13)
            return FJ_SCANNER_CORRECTION_TRACE_ZERO_HASH;
        return FJ_SCANNER_CORRECTION_TRACE_NONE;
    }
    void failure(const Outcome& out, unsigned category, FjScannerCorrectionFailure expected) {
        status(out.status, category);
        require(out.failure.reason == expected.reason, "exact scanner failure reason");
        const FjScannerColorCorrection zero{};
        require(project(out.value) == project(zero), "every scanner result field is zero on failure");
        require(out.trace.outcome == expected_trace(expected.reason) && out.trace.hash == 0, "typed scanner rejection trace outcome");
        if (out.trace.outcome == 0)
            require(trace_words(out.trace) == Json::array({0, 0, 0, 0, 0, 0, 0, 0}), "all scanner early-failure trace fields are +0");
    }
    const Json& literal(const Json& captured, const char* id) {
        for (const auto& row : captured.at("cases"))
            if (row.at("id") == id)
                return row;
        throw std::runtime_error("missing scanner correction literal");
    }
    void values(const Json& captured, const Json& checked) {
        require(captured.at("cases").size() == 150 && captured.at("products").size() == 48, "exact scanner characterization membership");
        for (bool facade : {false, true}) {
            for (const auto& row : captured.at("cases")) {
                const auto& norm = checked.at("cases").at(row.at("id").get<std::string>());
                if (norm.at("native_only").get<bool>())
                    continue;
                const auto id = row.at("id").get<std::string>();
                const std::string suffix = facade ? "/test-facade" : "/legacy-facade";
                JuicerTests::CaseEvent facade_event("cpp", id + suffix);
                Inputs source(row.at("input"));
                const bool print = row.at("spec").at("print");
                const bool missing = row.at("spec").value("mutation", "none") == "missing";
                const auto out = invoke(source, print, facade, missing);
                const unsigned reason = norm.at("reason");
                if (reason == 0) {
                    status(out.status);
                    require(out.failure.reason == 0 && out.length == 0, "successful scanner metadata");
                    require(project(out.value) == row.at("native").at("descriptor"), "all seven scanner words, flags, route, schema and hash equal native");
                    require(out.trace.outcome == (out.value.active ? 1u : 0u), "active/inactive scanner trace gate");
                    if (out.value.active) {
                        require(out.trace.hash == out.value.hash && out.trace.corrected_midgray > 0 && std::isfinite(out.trace.corrected_midgray), "accepted scanner trace retains derived midgray and identity");
                        require(words(&out.trace.reference_black_y, 1) == words(&out.value.reference_black_y, 1), "scanner trace/reference continuity");
                    }
                } else {
                    failure(out, norm.at("category"), {reason});
                    require(std::string(out.text.data()) == (norm.at("intentional_exception_replacement").get<bool>()
                                                                 ? checked.at("print_byte_diagnostic").get<std::string>()
                                                                 : row.at("native").at("diagnostic").get<std::string>()),
                            "exact scanner diagnostic independent of typed policy");
                    if (norm.at("intentional_exception_replacement").get<bool>())
                        require(row.at("native").at("native_outcome") == "exception" && row.at("native").at("native_bool").is_null(), "checked print byte rejection does not relabel native exception");
                }
                const auto tables = source.tables();
                const auto direct = source.direct(tables);
                const auto paper = source.print(tables);
                const auto result = reason == 0 ? expected(row.at("native").at("descriptor")) : FjScannerColorCorrection{};
                JuicerTests::CaseEvent c11_event("c11", id + suffix + "/c-call");
                require(fj_test_scanner_correction_c_call(print ? 1u : 0u, &source.params, missing ? nullptr : &direct, missing ? nullptr : &paper, &result, norm.at("category"), reason) == 1, "actual C11 signature/value/failure call");
                c11_event.passed();
                facade_event.passed();
            }
            for (const auto& row : captured.at("products")) {
                JuicerTests::CaseEvent case_event("cpp", "scanner-correction/product/" + row.at("id").get<std::string>() + (facade ? "/test-facade" : "/legacy-facade"));
                Inputs source(row.at("input"));
                const auto out = invoke(source, row.at("print"), facade);
                status(out.status);
                require(out.failure.reason == 0 && out.length == 0, "actual scanner inputs accepted");
                require(project(out.value) == row.at("native").at("descriptor"), "all genuine input corrections exact");
                case_event.passed();
            }
        }
        Outcome retained;
        {
            Inputs ephemeral(literal(captured, "direct-none").at("input"));
            retained = invoke(ephemeral, false, false);
        }
        require(project(retained.value) == literal(captured, "direct-none").at("native").at("descriptor"), "scanner result owns all words after inputs expire");
    }

    void boundary(const Json& captured) {
        JuicerTests::CaseEvent boundary_event("cpp", "scanner-correction/boundary");
        Inputs source(literal(captured, "direct-none").at("input"));
        Inputs paperSource(literal(captured, "print-none").at("input"));
        std::vector<std::string> mode_ids, capacity_ids, scenario_ids, fault_ids;
        for (unsigned facade = 0; facade < 2; ++facade) {
            for (unsigned route = 0; route < 256; ++route)
                for (unsigned flags = 0; flags < 4; ++flags)
                    for (bool print : {false, true})
                        mode_ids.emplace_back("caller-" + std::to_string(facade) + "/route-" + std::to_string(route) + "/flags-" + std::to_string(flags) + "/print-" + std::to_string(print));
            for (const auto capacity : {0u, 1u, 9u, 512u})
                capacity_ids.emplace_back("caller-" + std::to_string(facade) + "/capacity-" + std::to_string(capacity));
            for (unsigned scenario = 0; scenario < 13; ++scenario)
                scenario_ids.emplace_back("caller-" + std::to_string(facade) + "/scenario-" + std::to_string(scenario));
            for (unsigned operation = 0; operation < 2; ++operation)
                fault_ids.emplace_back("caller-" + std::to_string(facade) + "/operation-" + std::to_string(operation));
        }
        JuicerTests::Cohort demand_modes("scanner_correction.demand_mode", mode_ids);
        JuicerTests::Cohort diagnostic_capacities("scanner_correction.diagnostic_capacity", capacity_ids);
        JuicerTests::Cohort malformed_scenarios("scanner_correction.malformed_scenario", scenario_ids);
        JuicerTests::Cohort preparation_faults("scanner_correction.preparation_fault", fault_ids);
        unsigned facade_index = 0;
        for (bool facade : {false, true}) {
            for (unsigned route = 0; route < 256; ++route)
                for (unsigned flags = 0; flags < 4; ++flags)
                    for (bool print : {false, true}) {
                        auto row_event = demand_modes.case_event("caller-" + std::to_string(facade_index) + "/route-" + std::to_string(route) + "/flags-" + std::to_string(flags) + "/print-" + std::to_string(print));
                        const FjScannerCorrectionParams params{route, flags, kNan, kNan};
                        std::uint32_t required = 99;
                        FjScannerCorrectionFailure detail{99};
                        const bool inactive = flags == 0 || (!print && route == 0);
                        const bool active = flags != 0 && (print ? route == 1 || route == 3 : route == 2);
                        const auto query = fj_legacy_scanner_correction_input_demand(&params, print ? 1u : 0u, &required, &detail, nullptr);
                        status(query, inactive || active ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                        require(required == unsigned(active) && detail.reason == (inactive || active ? 0u : print ? 7u
                                                                                                                  : 2u),
                                "single shared demand for every native byte/flag pair");
                        Outcome out;
                        out.status = print ? print_entry(facade)(&params, reinterpret_cast<const FjScannerPrintCorrectionInput*>(std::uintptr_t{1}), &out.value, &out.failure, &out.trace, nullptr)
                                           : direct_entry(facade)(&params, reinterpret_cast<const FjScannerDirectCorrectionInput*>(std::uintptr_t{1}), &out.value, &out.failure, &out.trace, nullptr);
                        if (inactive) {
                            status(out.status);
                            require(out.value.route == route && out.value.active == 0 && out.value.schema_version == 1 && out.value.hash == 0, "inactive raw byte/defaults ignore malformed active pointer");
                        } else
                            failure(out, active ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_PREPARATION_FAILURE, {active ? 1u : print ? 7u
                                                                                                                                    : 2u});
                        row_event.passed();
                    }
            const auto tables = source.tables();
            auto input = source.direct(tables);
            for (unsigned scenario = 0; scenario < 13; ++scenario) {
                auto row_event = malformed_scenarios.case_event("caller-" + std::to_string(facade_index) + "/scenario-" + std::to_string(scenario));
                auto params = source.params;
                auto malformed = input;
                auto malformedTables = tables;
                malformed.tables = &malformedTables;
                Outcome out;
                FjErrorBuffer diagnostic{out.text.data(), out.text.size(), 99};
                const FjScannerCorrectionParams* requested = &params;
                FjScannerColorCorrection* result = &out.value;
                FjScannerCorrectionFailure* detail = &out.failure;
                FjScannerCorrectionTrace* trace = &out.trace;
                if (scenario == 0)
                    requested = nullptr;
                if (scenario == 1)
                    params.route = 256;
                if (scenario == 2)
                    params.flags = 4;
                if (scenario == 3)
                    malformed.film_base_density.count = 80;
                if (scenario == 4)
                    malformed.film_authored_density_curves_cmy.count = 8;
                if (scenario == 5)
                    malformedTables.eps_c.count = 0;
                if (scenario == 6)
                    malformed.medium = 256;
                if (scenario == 7)
                    malformedTables.has_baseline = 2;
                if (scenario == 8)
                    diagnostic = {nullptr, 1, 99};
                if (scenario == 9)
                    result = nullptr;
                if (scenario == 10)
                    detail = nullptr;
                if (scenario == 11)
                    trace = reinterpret_cast<FjScannerCorrectionTrace*>(std::uintptr_t{1});
                if (scenario == 12)
                    malformed.film_log_exposure = {reinterpret_cast<const float*>(std::uintptr_t{4}), SIZE_MAX};
                out.status = direct_entry(facade)(requested, &malformed, result, detail, trace, &diagnostic);
                status(out.status, FJ_STATUS_UNSUPPORTED_INPUT);
                if (detail)
                    require(out.failure.reason == 1, "independent valid failure record receives RAW");
                if (result)
                    require(project(out.value) == project(FjScannerColorCorrection{}), "independent valid result clears when peer is invalid");
                if (scenario != 11)
                    require(out.trace.outcome == 0 && trace_words(out.trace) == Json::array({0, 0, 0, 0, 0, 0, 0, 0}), "independent valid trace clears");
                if (scenario != 8)
                    require(diagnostic.length < diagnostic.capacity && out.text[diagnostic.length] == '\0', "diagnostic remains bounded on structural failure");
                row_event.passed();
            }
            auto ignoredTables = tables;
            ignoredTables.baseline_min = {reinterpret_cast<const float*>(std::uintptr_t{1}), SIZE_MAX};
            auto ignored = source.direct(ignoredTables);
            Outcome ignoredOut;
            ignoredOut.status = direct_entry(facade)(&source.params, &ignored, &ignoredOut.value, &ignoredOut.failure, &ignoredOut.trace, nullptr);
            status(ignoredOut.status);
            require(project(ignoredOut.value) == literal(captured, "direct-none").at("native").at("descriptor"), "disabled baseline backing ignored");
            const std::string message = "ResourceDescriptorMismatch phase=8B print correction inputs";
            for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{9}, std::size_t{512}}) {
                auto row_event = diagnostic_capacities.case_event("caller-" + std::to_string(facade_index) + "/capacity-" + std::to_string(capacity));
                std::array<char, 512> text{};
                text.fill('x');
                FjErrorBuffer error{capacity ? text.data() : reinterpret_cast<char*>(std::uintptr_t{1}), capacity, 99};
                FjScannerCorrectionParams params{255, 3, 0, 1};
                Outcome out;
                out.status = print_entry(facade)(&params, reinterpret_cast<const FjScannerPrintCorrectionInput*>(std::uintptr_t{1}), &out.value, &out.failure, &out.trace, &error);
                failure(out, FJ_STATUS_PREPARATION_FAILURE, {7});
                const auto length = capacity == 0 ? 0 : std::min(capacity - 1, message.size());
                require(error.length == length && (capacity == 0 || (text[length] == '\0' && std::string(text.data()) == message.substr(0, length))), "capacity zero/one/truncated/full preserves checked envelope");
                row_event.passed();
            }
            for (unsigned operation : {0u, 1u}) {
                auto row_event = preparation_faults.case_event("caller-" + std::to_string(facade_index) + "/operation-" + std::to_string(operation));
                auto& active = operation == 0 ? source : paperSource;
                status(fj_test_scanner_correction_fault(operation, 1));
                std::uint32_t demand = 99;
                FjScannerCorrectionFailure detail{99};
                status(fj_legacy_scanner_correction_input_demand(&active.params, operation, &demand, &detail, nullptr));
                require(demand == 1, "query succeeds without consuming prepare panic");
                failure(invoke(active, operation != 0, facade), FJ_STATUS_INTERNAL_FAILURE, {14});
                status(invoke(active, operation != 0, facade).status);
                status(fj_test_scanner_correction_fault(operation, 1));
                Outcome raw;
                raw.status = operation == 0 ? direct_entry(facade)(nullptr, nullptr, &raw.value, &raw.failure, &raw.trace, nullptr)
                                            : print_entry(facade)(nullptr, nullptr, &raw.value, &raw.failure, &raw.trace, nullptr);
                failure(raw, FJ_STATUS_UNSUPPORTED_INPUT, {1});
                status(invoke(active, operation != 0, facade).status);
                status(fj_test_scanner_correction_fault(operation, 1));
                std::thread independent([&] {
                    status(invoke(active, operation != 0, facade).status);
                });
                independent.join();
                failure(invoke(active, operation != 0, facade), FJ_STATUS_INTERNAL_FAILURE, {14});
                status(invoke(active, operation != 0, facade).status);
                row_event.passed();
            }
            ++facade_index;
        }
        status(fj_test_scanner_correction_fault(2, 1), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_test_scanner_correction_fault(0, 2), FJ_STATUS_UNSUPPORTED_INPUT);
        JuicerTests::Cohort demand_query_scenarios("scanner_correction.demand_query_scenario", {"0", "1", "2", "3", "4"});
        for (unsigned scenario = 0; scenario < 5; ++scenario) {
            auto row_event = demand_query_scenarios.case_event(std::to_string(scenario));
            auto params = source.params;
            std::uint32_t required = 99;
            FjScannerCorrectionFailure detail{99};
            FjErrorBuffer malformed{nullptr, 1, 99};
            if (scenario == 3) {
                params.route = 256;
                params.flags = 0;
            }
            const auto result = fj_legacy_scanner_correction_input_demand(scenario == 0 ? nullptr : &params, 0, scenario == 1 ? nullptr : &required, scenario == 2 ? nullptr : &detail, scenario == 4 ? &malformed : nullptr);
            status(result, FJ_STATUS_UNSUPPORTED_INPUT);
            if (scenario != 1)
                require(required == 0, "query independently clears required on invalid peer/Params/all-off width");
            if (scenario != 2)
                require(detail.reason == FJ_SCANNER_CORRECTION_REASON_RAW, "query independently reports RAW on invalid peer");
            row_event.passed();
        }
        {
            auto tables = source.tables();
            tables.eps_m = tables.eps_c;
            tables.eps_y = tables.eps_c;
            const auto input = source.direct(tables);
            Outcome out;
            out.status = fj_legacy_scanner_direct_correction_prepare(&source.params, &input, &out.value, &out.failure, nullptr, nullptr);
            status(out.status);
            require(project(out.value) == literal(captured, "direct-none").at("native").at("descriptor"), "immutable scanner input spans may alias and optional trace may be omitted");
        }
        // The accepted film-recipe selector's two stage-13 witnesses stay untouched.
        std::uint32_t required = 99;
        FjScannerCorrectionFailure detail{99};
        status(fj_legacy_scanner_correction_input_demand(&source.params, 2, &required, &detail, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(required == 0 && detail.reason == 1, "unknown query operation clears required");
        boundary_event.passed();
    }

    void check_products(const std::filesystem::path& resources, const Json& captured) {
        require_equal(products(resources, captured.at("products")), captured.at("products"), "all 48 actual correction products retain controls, resource inputs, descriptor and enclosing identities");
    }
} // namespace ScannerCorrectionTest
