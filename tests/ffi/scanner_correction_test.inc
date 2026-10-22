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

    Json project(const Scanner::ScannerColorCorrectionDescriptor& value) {
        const std::array floats{value.targetBlackLinear, value.targetWhiteLinear, value.referenceBlackY, value.referenceWhiteY, value.xyzSlope, value.xyzOffset, value.exposureScale};
        return {{"route", static_cast<unsigned>(value.route)}, {"active", value.active}, {"black", value.blackCorrection}, {"white", value.whiteCorrection}, {"words", words(floats.data(), floats.size())}, {"schema", value.schemaVersion}, {"hash", value.hash}};
    }

    template <typename Rows>
    Json row_words(const Rows& rows) {
        Json out = Json::array();
        for (const auto& row : rows)
            for (float value : row)
                out.push_back(std::bit_cast<std::uint32_t>(value));
        return out;
    }

    Json inputs(const RenderRecipe& recipe, const Spectral::SpectralTables& tables, const std::array<float, 81>& light, const std::array<float, 3>& preflash, float normalizer, bool print) {
        const auto& output = recipe.scannerOutput;
        const auto& bounds = recipe.densityBounds;
        Json out{{"route", static_cast<unsigned>(recipe.profileRoute.scanRoute)},
                 {"flags", unsigned(output.blackCorrection) + 2u * unsigned(output.whiteCorrection)},
                 {"levels", words(std::array{output.blackLevel, output.whiteLevel}.data(), 2)},
                 {"medium", static_cast<unsigned>(bounds.medium)},
                 {"minimum", words(bounds.dataMinCmy.data(), 3)},
                 {"maximum", words(bounds.dataMaxCmy.data(), 3)},
                 {"inverse", words(bounds.invSpanCmy.data(), 3)},
                 {"tables", {{"c", words(tables.epsC.data(), tables.epsC.size())}, {"m", words(tables.epsM.data(), tables.epsM.size())}, {"y", words(tables.epsY.data(), tables.epsY.size())}, {"ay", words(tables.Ay.data(), tables.Ay.size())}, {"baseline", words(tables.baseDensityMin.data(), tables.baseDensityMin.size())}, {"k", tables.K}, {"has_baseline", tables.hasBaseline}, {"inv_yn", std::bit_cast<std::uint32_t>(tables.invYn)}}}};
        if (recipe.profileRoute.filmProfile) {
            const auto& film = recipe.profileRoute.filmProfile->data;
            out["film_base"] = words(film.baseDensity.data(), 81);
            out["film_axis"] = words(film.logExposure.data(), film.logExposure.size());
            out["film_dyes"] = row_words(film.channelDensity);
        }
        out["authored_curves"] = row_words(recipe.filmDevelop.authoredDensityCurves);
        out["baseline_density"] = words(output.syntheticFilmReference.baselineDensityCmy.data(), 3);
        out["reference_hash"] = output.syntheticFilmReference.hash;
        if (print && recipe.profileRoute.printProfile) {
            const auto& paper = recipe.profileRoute.printProfile->data;
            out["print_base"] = words(paper.baseDensity.data(), 81);
            out["print_axis"] = words(paper.logExposure.data(), paper.logExposure.size());
            out["print_curves"] = row_words(paper.densityCurves);
            out["sensitivity"] = row_words(paper.linearSensitivity);
        }
        out["light"] = words(light.data(), 81);
        out["preflash"] = words(preflash.data(), 3);
        out["normalizer"] = std::bit_cast<std::uint32_t>(normalizer);
        out["preflash_exposure"] = std::bit_cast<std::uint32_t>(recipe.print.exposure.preflashExposure);
        out["film_minimum"] = words(recipe.enlargerFilmBounds.dataMinCmy.data(), 3);
        out["film_maximum"] = words(recipe.enlargerFilmBounds.dataMaxCmy.data(), 3);
        return out;
    }

    struct Literal {
        RenderRecipe recipe;
        std::shared_ptr<Profiles::FilmProfile> film = std::make_shared<Profiles::FilmProfile>();
        std::shared_ptr<Profiles::PrintProfile> paper = std::make_shared<Profiles::PrintProfile>();
        Spectral::SpectralTables tables;
        std::array<float, 81> light{};
        std::array<float, 3> preflash{};
        float normalizer = 1.0f;
        bool missing = false;
        bool nullRecipe = false;

        explicit Literal(const Json& spec) {
            recipe.profileRoute.filmProfile = film;
            recipe.profileRoute.printProfile = paper;
            recipe.profileRoute.scanRoute = static_cast<Spektrafilm::ScanRoute>(spec.at("route").get<unsigned>());
            auto& output = recipe.scannerOutput;
            const unsigned flags = spec.at("flags");
            output.blackCorrection = (flags & 1u) != 0;
            output.whiteCorrection = (flags & 2u) != 0;
            output.blackLevel = 0.0f;
            output.whiteLevel = 1.0f;
            if (spec.contains("levels")) {
                output.blackLevel = std::bit_cast<float>(spec.at("levels")[0].get<std::uint32_t>());
                output.whiteLevel = std::bit_cast<float>(spec.at("levels")[1].get<std::uint32_t>());
            }
            output.syntheticFilmReference.hash = 1;
            output.syntheticFilmReference.baselineDensityCmy = {1, 1, 1};
            auto& bounds = recipe.densityBounds;
            bounds.medium = spec.at("print").get<bool>() ? Spektrafilm::DensityMedium::Print : Spektrafilm::DensityMedium::Film;
            bounds.dataMaxCmy = {3, 3, 3};
            bounds.invSpanCmy = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
            recipe.enlargerFilmBounds.dataMaxCmy = {3, 3, 3};
            film->data.logExposure = {-2, 0, 2};
            recipe.filmDevelop.authoredDensityCurves = {{3, 3, 3}, {1.5f, 1.5f, 1.5f}, {0, 0, 0}};
            paper->data.logExposure = {-2, 0, 2};
            paper->data.densityCurves = {{0, 0, 0}, {1.5f, 1.5f, 1.5f}, {3, 3, 3}};
            for (std::size_t sample = 0; sample < 81; ++sample) {
                film->data.channelDensity[sample] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
                paper->data.linearSensitivity[sample] = {1.0f / 81.0f, 1.0f / 81.0f, 1.0f / 81.0f};
            }
            light.fill(1);
            tables.K = 1;
            tables.epsC = tables.epsM = tables.epsY = {1.0f / 3.0f};
            tables.Ax = tables.Ay = tables.Az = {1};
            const std::string mutation = spec.value("mutation", "none");
            if (mutation == "missing") {
                missing = true;
                recipe.profileRoute.filmProfile.reset();
                recipe.profileRoute.printProfile.reset();
            } else if (mutation == "null-recipe") {
                nullRecipe = true;
            } else if (mutation == "reference") {
                output.syntheticFilmReference.hash = 0;
            } else if (mutation == "scan") {
                tables.invYn = kInfinity;
            } else if (mutation == "midgray") {
                output.blackLevel = output.whiteLevel = 0;
            } else if (mutation == "direct-lookup") {
                recipe.filmDevelop.authoredDensityCurves.clear();
            } else if (mutation == "print-lookup") {
                paper->data.logExposure.clear();
            } else if (mutation == "range") {
                tables.epsC = tables.epsM = tables.epsY = {0};
            } else if (mutation == "normalizer-nan") {
                normalizer = kNan;
            } else if (mutation == "normalizer-zero") {
                normalizer = 0;
            } else if (mutation == "normalizer-negative") {
                normalizer = -1;
            } else if (mutation == "preflash") {
                preflash.fill(.25f);
                recipe.print.exposure.preflashExposure = .1f;
            } else if (mutation == "zero-inverse") {
                bounds.invSpanCmy[0] = 0;
            } else if (mutation == "infinite-inverse") {
                bounds.invSpanCmy[0] = kInfinity;
            } else if (mutation == "narrow-overflow") {
                tables.Ay[0] = std::numeric_limits<float>::max();
                tables.invYn = 2;
            } else if (mutation == "baseline-nan") {
                tables.hasBaseline = true;
                tables.baseDensityMin = {kNan};
            } else if (mutation == "negative-ay") {
                tables.Ay[0] = -1;
            } else if (mutation == "nan-ay") {
                tables.Ay[0] = kNan;
            } else if (mutation == "finite-negative-sensitivity") {
                paper->data.linearSensitivity[0] = {-1, -1, -1};
            } else if (mutation == "nan-light") {
                light[0] = kNan;
            } else if (mutation == "provenance") {
                recipe.filmDevelop.logExposure = {-20, 0, 20};
                film->data.densityCurves = {{13, 13, 13}, {12, 12, 12}, {11, 11, 11}};
            } else if (mutation == "baseline") {
                tables.hasBaseline = true;
                tables.baseDensityMin = {.1234567f};
            } else if (mutation == "baseline-infinite") {
                tables.hasBaseline = true;
                tables.baseDensityMin = {kInfinity};
            } else if (mutation == "nonfinite-eps") {
                tables.epsM[0] = kNan;
            } else if (mutation == "k-zero") {
                tables.K = 0;
            } else if (mutation == "k-81") {
                tables.K = 81;
                tables.epsC.assign(81, .1234567f);
                tables.epsM.assign(81, .33333334f);
                tables.epsY.assign(81, .7777777f);
                tables.Ax = tables.Ay = tables.Az = std::vector<float>(81, 1.0f / 81.0f);
            } else if (mutation == "medium-literal") {
                bounds.medium = static_cast<Spektrafilm::DensityMedium>(255);
                bounds.dataMinCmy = {.1234567f, -.1f, .33333334f};
            } else if (mutation == "medium-opposite") {
                bounds.medium = spec.at("print").get<bool>() ? Spektrafilm::DensityMedium::Film : Spektrafilm::DensityMedium::Print;
                bounds.dataMinCmy = {.1234567f, -.1f, .33333334f};
            } else if (mutation == "base-nonfinite") {
                film->data.baseDensity[0] = paper->data.baseDensity[0] = kNan;
                film->data.baseDensity[1] = paper->data.baseDensity[1] = kInfinity;
            } else if (mutation == "mean-nonfinite") {
                recipe.filmDevelop.authoredDensityCurves[1] = {kNan, 1.5f, kInfinity};
                paper->data.densityCurves[1] = {kNan, 1.5f, kInfinity};
            } else if (mutation == "mean-zero") {
                recipe.filmDevelop.authoredDensityCurves[1] = {kNan, -kInfinity, kInfinity};
                paper->data.densityCurves[1] = {kNan, -kInfinity, kInfinity};
            } else if (mutation == "singleton") {
                film->data.logExposure = {0};
                recipe.filmDevelop.authoredDensityCurves = {{1, 1, 1}};
                paper->data.logExposure = {0};
                paper->data.densityCurves = {{1, 1, 1}};
            } else if (mutation == "duplicate") {
                film->data.logExposure = {-2, 0, 1, 2};
                recipe.filmDevelop.authoredDensityCurves = {{3, 3, 3}, {1.5f, 1.5f, 1.5f}, {1.5f, 1.5f, 1.5f}, {0, 0, 0}};
                paper->data.logExposure = {-2, 0, 0, 2};
                paper->data.densityCurves = {{0, 0, 0}, {1.5f, 1.5f, 1.5f}, {2, 2, 2}, {3, 3, 3}};
            } else if (mutation == "descending") {
                std::reverse(film->data.logExposure.begin(), film->data.logExposure.end());
                std::reverse(recipe.filmDevelop.authoredDensityCurves.begin(), recipe.filmDevelop.authoredDensityCurves.end());
                std::reverse(paper->data.logExposure.begin(), paper->data.logExposure.end());
                std::reverse(paper->data.densityCurves.begin(), paper->data.densityCurves.end());
            } else if (mutation == "mismatch") {
                film->data.logExposure.pop_back();
                paper->data.logExposure.pop_back();
            } else if (mutation == "axis-nan") {
                film->data.logExposure[1] = paper->data.logExposure[1] = kNan;
            } else if (mutation == "axis-infinite") {
                film->data.logExposure = {-kInfinity, 0, kInfinity};
                paper->data.logExposure = {-kInfinity, 0, kInfinity};
            } else if (mutation != "none") {
                throw std::runtime_error("unknown scanner correction capture mutation");
            }
        }
    };

    Json observe(const RenderRecipe& recipe, const Spectral::SpectralTables& tables, const std::array<float, 81>& light, const std::array<float, 3>& preflash, float normalizer, bool print, bool missing = false, bool nullRecipe = false) {
        Scanner::ScannerColorCorrectionDescriptor descriptor;
        descriptor.hash = 99;
        std::string diagnostic = "seeded";
        Json out;
        try {
            const bool returned = print
                                      ? Scanner::build_print_scanner_color_correction_descriptor(
                                            {nullRecipe ? nullptr : &recipe, missing ? nullptr : &tables, missing ? nullptr : light.data(), 81, missing ? nullptr : preflash.data(), normalizer},
                                            descriptor,
                                            diagnostic)
                                      : Scanner::build_direct_scanner_color_correction_descriptor(recipe, tables, descriptor, diagnostic);
            out = {{"native_outcome", "returned"}, {"native_bool", returned}, {"exception_kind", nullptr}};
        } catch (const std::out_of_range&) {
            out = {{"native_outcome", "exception"}, {"native_bool", nullptr}, {"exception_kind", "std::out_of_range"}};
        }
        out["descriptor"] = project(descriptor);
        out["diagnostic"] = diagnostic;
        return out;
    }

    std::streamoff trace_position() {
        if (!JTRACE_ENABLED(2))
            return 0;
        auto& sink = JuicerLogging::sink();
        std::lock_guard lock(sink.m);
        JuicerLogging::ensure_sink_locked(sink);
        sink.stream.flush();
        return static_cast<std::streamoff>(sink.sizeBytes);
    }
    Json correction_trace_since(std::streamoff position) {
        Json rows = Json::array();
        if (!JTRACE_ENABLED(2))
            return rows;
        auto& sink = JuicerLogging::sink();
        std::lock_guard lock(sink.m);
        sink.stream.flush();
        std::ifstream input(sink.path, std::ios::binary);
        require(input.is_open(), "scanner trace sink is readable");
        input.seekg(position);
        for (std::string line; std::getline(input, line);)
            if (line.starts_with("SCAN | event=scanner_correction_descriptor "))
                rows.push_back(line);
        return rows;
    }

    void check_trace(const Json& actual, std::string_view context, const Json& native) {
        require(actual.size() == native.size(), "scanner correction trace event count");
        constexpr std::array fields{"blackLevelEncoded", "whiteLevelEncoded", "targetBlackLinear", "targetWhiteLinear", "referenceBlackY", "referenceWhiteY", "referenceDeltaY", "xyzSlope", "xyzOffset", "correctedMidgrayInput", "exposureScale"};
        const auto is_nan = [](std::string_view token) {
            if (token.starts_with('-') || token.starts_with('+'))
                token.remove_prefix(1);
            return token == "nan" || (token.starts_with("nan(") && token.ends_with(')'));
        };
        for (std::size_t i = 0; i < actual.size(); ++i) {
            auto observed = actual.at(i).get<std::string>();
            auto expected = native.at(i).get<std::string>();
            std::vector<std::string> differences;
            if (observed.find(" outcome=rejected_") != std::string::npos && expected.find(" outcome=rejected_") != std::string::npos) {
                for (const auto* field : fields) {
                    const auto prefix = std::string(" ") + field + "=";
                    const auto observedPosition = observed.find(prefix);
                    const auto expectedPosition = expected.find(prefix);
                    if (observedPosition == std::string::npos || expectedPosition == std::string::npos)
                        continue;
                    const auto observedBegin = observedPosition + prefix.size();
                    const auto expectedBegin = expectedPosition + prefix.size();
                    const auto observedToken = observed.substr(observedBegin, observed.find(' ', observedBegin) - observedBegin);
                    const auto expectedToken = expected.substr(expectedBegin, expected.find(' ', expectedBegin) - expectedBegin);
                    if (observedToken != expectedToken && is_nan(observedToken) && is_nan(expectedToken)) {
                        differences.emplace_back(field).append(" actual=").append(observedToken).append(" native=").append(expectedToken);
                        observed.replace(observedBegin, observedToken.size(), "<diagnostic-nan>");
                        expected.replace(expectedBegin, expectedToken.size(), "<diagnostic-nan>");
                    }
                }
            }
            if (observed != expected) {
                std::fprintf(stderr, "scanner trace disagreement context=%.*s actual=%s native=%s\n", static_cast<int>(context.size()), context.data(), actual.at(i).get<std::string>().c_str(), native.at(i).get<std::string>().c_str());
                require(false, "public scanner trace continuity");
            }
            for (const auto& difference : differences)
                std::fprintf(stderr, "SCANNER_DIAGNOSTIC_NAN context=%.*s %s\n", static_cast<int>(context.size()), context.data(), difference.c_str());
        }
    }

    Json products(const std::filesystem::path& resources, const Json& expectedTraces) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        Json out = Json::array();
        for (unsigned route = 0; route < 4; ++route)
            for (int mode = 0; mode < 3; ++mode)
                for (unsigned flags = 0; flags < 4; ++flags) {
                    auto params = PrintPreflashTest::controls();
                    params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                    params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                    params.spectralUpsamplingMode = mode;
                    params.scannerBlackCorrection = (flags & 1u) != 0;
                    params.scannerWhiteCorrection = (flags & 2u) != 0;
                    FocusedRenderStateBuildProduct product;
                    std::string diagnostic;
                    const bool success = route % 2 ? build_print_render_state_product(params, product, diagnostic)
                                                   : build_direct_render_state_product(params, product, diagnostic);
                    require(success, diagnostic.c_str());
                    const auto& recipe = product.recipe;
                    JTRACE_LEVEL(2, "SCANNER_CAPTURE", std::string("product=") + std::to_string(route) + ":" + std::to_string(mode) + ":" + std::to_string(flags));
                    std::array<float, 81> light{};
                    std::array<float, 3> preflash{};
                    float normalizer = 1;
                    if (route % 2) {
                        JuicerCuda::PrintResourceInput printInput;
                        std::array<float, 81> preflashLight{};
                        require(JuicerCuda::build_print_resource_input(
                                    {&recipe, &JuicerProcess::root().assets(), product.payload.printMainIlluminant ? &*product.payload.printMainIlluminant : nullptr},
                                    printInput,
                                    preflashLight,
                                    diagnostic),
                                diagnostic.c_str());
                        require(printInput.mainIlluminant.size() == light.size(), "prepared correction light extent");
                        std::copy(printInput.mainIlluminant.begin(), printInput.mainIlluminant.end(), light.begin());
                        preflash = printInput.preflashRawCmy;
                        normalizer = printInput.normalizer;
                    }
                    const auto position = trace_position();
                    const auto observation = observe(recipe, product.payload.scannerTables, light, preflash, normalizer, route % 2 != 0);
                    const auto context = "product:" + std::to_string(route) + ":" + std::to_string(mode) + ":" + std::to_string(flags);
                    check_trace(correction_trace_since(position), context, expectedTraces.at(context));
                    out.push_back({{"id", std::to_string(route) + ":" + std::to_string(mode) + ":" + std::to_string(flags)},
                                   {"mode", mode},
                                   {"print", route % 2 != 0},
                                   {"input", inputs(recipe, product.payload.scannerTables, light, preflash, normalizer, route % 2 != 0)},
                                   {"native", observation},
                                   {"snapshot_hash", hash_params(params)},
                                   {"recipe_hash", recipe.hash},
                                   {"scanner_hash", recipe.scannerOutput.hash},
                                   {"upload_hash", product.payload.uploadCoreHash}});
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
                require(fj_test_scanner_correction_c_call(print ? 1u : 0u, &source.params, missing ? nullptr : &direct, missing ? nullptr : &paper, &result, norm.at("category"), reason) == 1, "actual C11 signature/value/failure call");
            }
            for (const auto& row : captured.at("products")) {
                Inputs source(row.at("input"));
                const auto out = invoke(source, row.at("print"), facade);
                status(out.status);
                require(out.failure.reason == 0 && out.length == 0, "actual scanner inputs accepted");
                require(project(out.value) == row.at("native").at("descriptor"), "all genuine input corrections exact");
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
        Inputs source(literal(captured, "direct-none").at("input"));
        Inputs paperSource(literal(captured, "print-none").at("input"));
        for (bool facade : {false, true}) {
            for (unsigned route = 0; route < 256; ++route)
                for (unsigned flags = 0; flags < 4; ++flags)
                    for (bool print : {false, true}) {
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
                    }
            const auto tables = source.tables();
            auto input = source.direct(tables);
            for (unsigned scenario = 0; scenario < 13; ++scenario) {
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
                std::array<char, 512> text{};
                text.fill('x');
                FjErrorBuffer error{capacity ? text.data() : reinterpret_cast<char*>(std::uintptr_t{1}), capacity, 99};
                FjScannerCorrectionParams params{255, 3, 0, 1};
                Outcome out;
                out.status = print_entry(facade)(&params, reinterpret_cast<const FjScannerPrintCorrectionInput*>(std::uintptr_t{1}), &out.value, &out.failure, &out.trace, &error);
                failure(out, FJ_STATUS_PREPARATION_FAILURE, {7});
                const auto length = capacity == 0 ? 0 : std::min(capacity - 1, message.size());
                require(error.length == length && (capacity == 0 || (text[length] == '\0' && std::string(text.data()) == message.substr(0, length))), "capacity zero/one/truncated/full preserves checked envelope");
            }
            for (unsigned operation : {0u, 1u}) {
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
            }
        }
        status(fj_test_scanner_correction_fault(2, 1), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_test_scanner_correction_fault(0, 2), FJ_STATUS_UNSUPPORTED_INPUT);
        for (unsigned scenario = 0; scenario < 5; ++scenario) {
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
    }

    thread_local unsigned corruption = 0;
    void corrupt(FjStatus& status, FjScannerColorCorrection& value, FjScannerCorrectionFailure& failure, FjScannerCorrectionTrace& trace) {
        if (corruption == 0)
            status.api = FJ_API_CUDA_RUNTIME;
        if (corruption == 1)
            failure.reason = 99;
        if (corruption == 2)
            value.schema_version = 2;
        if (corruption == 3)
            value.active = 2;
        if (corruption == 4)
            value.xyz_slope = kNan;
        if (corruption == 5)
            value.hash = 0;
        if (corruption == 6)
            value.route = 255;
        if (corruption == 7)
            trace.outcome = 99;
        if (corruption == 8) {
            value.reference_white_y = value.reference_black_y;
            trace.reference_white_y = value.reference_white_y;
        }
        if (corruption == 9)
            trace.corrected_midgray = kNan;
    }
    void native(const Json& captured, const Json& checked) {
        for (const auto& row : captured.at("cases")) {
            Literal source(row.at("spec"));
            const bool print = row.at("spec").at("print");
            require(inputs(source.recipe, source.tables, source.light, source.preflash, source.normalizer, print) == row.at("input"), "native transport consumes frozen literal inputs");
            Scanner::ScannerColorCorrectionDescriptor out;
            FjScannerCorrectionTrace trace{};
            std::string diagnostic = "seed";
            const Scanner::PrintCorrectionDerivationInput paper{source.nullRecipe ? nullptr : &source.recipe, source.missing ? nullptr : &source.tables, source.missing ? nullptr : source.light.data(), 81, source.missing ? nullptr : source.preflash.data(), source.normalizer};
            const bool success = print ? JuicerScanner::prepare_print(paper, out, trace, diagnostic)
                                       : JuicerScanner::prepare_direct(source.recipe, source.tables, out, trace, diagnostic);
            const auto& norm = checked.at("cases").at(row.at("id").get<std::string>());
            require(success == (norm.at("native_only").get<bool>() ? row.at("native").at("native_bool").get<bool>() : norm.at("reason") == 0), "native adapter returned checked outcome");
            require(project(out) == row.at("native").at("descriptor"), "native success or reset descriptor preserves all captured words/requested byte");
            require(diagnostic == (norm.at("intentional_exception_replacement").get<bool>() ? checked.at("print_byte_diagnostic").get<std::string>()
                                                                                            : row.at("native").at("diagnostic").get<std::string>()),
                    "native exact ordinary diagnostic");
        }
        Literal source(literal(captured, "direct-none").at("spec"));
        for (corruption = 0; corruption < 10; ++corruption) {
            JuicerScanner::Test::set_projection(corrupt);
            bool rejected = false;
            Scanner::ScannerColorCorrectionDescriptor out;
            FjScannerCorrectionTrace trace{};
            std::string diagnostic;
            try {
                (void)JuicerScanner::prepare_direct(source.recipe, source.tables, out, trace, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = true;
                status(error.failure.status, FJ_STATUS_INTERNAL_FAILURE);
            }
            JuicerScanner::Test::set_projection(nullptr);
            require(rejected && !out.active && out.hash == 0 && out.route == source.recipe.profileRoute.scanRoute, "native integrity rejection exposes no partial descriptor");
        }
        SpectralAllocationProbe::arm();
        bool rejected = false;
        Scanner::ScannerColorCorrectionDescriptor out;
        FjScannerCorrectionTrace trace{};
        std::string diagnostic;
        try {
            (void)JuicerScanner::prepare_direct(source.recipe, source.tables, out, trace, diagnostic);
        } catch (const JuicerCuda::ExecutionFailure& error) {
            rejected = true;
            status(error.failure.status, FJ_STATUS_ALLOCATION_FAILURE);
        }
        SpectralAllocationProbe::clear();
        require(rejected && SpectralAllocationProbe::bytes() == 9 * sizeof(float) && !out.active && out.hash == 0, "actual flat-curve temporary allocation is separately typed");
        require(JuicerScanner::prepare_direct(source.recipe, source.tables, out, trace, diagnostic), "native allocation recovery");
        for (unsigned route : {4u, 255u})
            for (unsigned flags = 0; flags < 4; ++flags) {
                source.recipe.profileRoute.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                source.recipe.scannerOutput.blackCorrection = (flags & 1u) != 0;
                source.recipe.scannerOutput.whiteCorrection = (flags & 2u) != 0;
                const Scanner::PrintCorrectionDerivationInput malformed{&source.recipe, reinterpret_cast<const Spectral::SpectralTables*>(std::uintptr_t{1}), reinterpret_cast<const float*>(std::uintptr_t{1}), 81, reinterpret_cast<const float*>(std::uintptr_t{1}), kNan};
                const bool success = JuicerScanner::prepare_print(malformed, out, trace, diagnostic);
                require(success == (flags == 0) && trace.outcome == 0 && !out.active && out.route == source.recipe.profileRoute.scanRoute,
                        "native shared demand precedes malformed active storage for print byte domain");
            }
    }
    void callers(const Json& captured, const Json& checked, const std::filesystem::path& tracePath) {
        const auto expectedTraces = read(tracePath);
        for (const auto& row : captured.at("cases")) {
            Literal source(row.at("spec"));
            const auto position = trace_position();
            const auto observed = observe(source.recipe, source.tables, source.light, source.preflash, source.normalizer, row.at("spec").at("print"), source.missing, source.nullRecipe);
            Json expected = row.at("native");
            if (checked.at("cases").at(row.at("id").get<std::string>()).at("intentional_exception_replacement").get<bool>()) {
                expected["native_outcome"] = "returned";
                expected["native_bool"] = false;
                expected["exception_kind"] = nullptr;
                expected["diagnostic"] = checked.at("print_byte_diagnostic");
            }
            require(observed == expected, "public scanner caller preserves exact native value/reset/diagnostic with separate checked exception replacement");
            const auto context = "case:" + row.at("id").get<std::string>();
            check_trace(correction_trace_since(position), context, expectedTraces.at(context));
        }
    }
    void check_products(const std::filesystem::path& resources, const Json& captured, const Json& expectedTraces) {
        require(products(resources, expectedTraces) == captured.at("products"), "all 48 actual correction callers retain controls, resource inputs, descriptor and enclosing identities");
    }
} // namespace ScannerCorrectionTest
