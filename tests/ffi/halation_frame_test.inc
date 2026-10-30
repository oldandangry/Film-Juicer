namespace HalationFrameTest {
    FjHalationFrameInput raw(const Json& input);
    Json project(const FjScatterHalation& v);
    ScatterHalationOpticsRecipe recipe(const Json& input) {
        ScatterHalationOpticsRecipe out;
        out.scatterActive = input.at("scatter_active");
        out.backReflectionActive = input.at("back_reflection_active");
        const auto scalars = PrintPreflashTest::array<3>(input.at("scalars"));
        out.scatterAmount = scalars[0];
        out.scatterSpatialScale = scalars[1];
        out.halationSpatialScale = scalars[2];
        out.halationFirstSigmaUm = PrintPreflashTest::array<3>(input.at("first_sigma_um"));
        out.totalStrength = PrintPreflashTest::array<3>(input.at("total_strength"));
        out.hash = input.at("recipe_hash");
        return out;
    }

    Json project(const ScatterHalationGaussianDescriptor& v) {
        const std::array<float, 4> coefficients{v.B, v.B1, v.B2, v.B3};
        return {{"kind", static_cast<unsigned>(v.kind)}, {"radius", v.radius}, {"weights", words(v.firWeights.data(), v.firWeights.size())}, {"yvv", words(coefficients.data(), coefficients.size())}};
    }

    Json project(const ScatterHalationFrameDescriptor& v) {
        Json channels = Json::array();
        for (const auto& c : v.channels) {
            Json tail = Json::array(), bounce = Json::array();
            for (const auto& g : c.tail)
                tail.push_back(project(g));
            for (const auto& g : c.bounce)
                bounce.push_back(project(g));
            channels.push_back({{"core", project(c.core)}, {"tail", tail}, {"bounce", bounce}, {"strength", std::bit_cast<std::uint32_t>(c.totalStrength)}});
        }
        return {{"recipe_hash", v.recipeHash}, {"scatter_amount", std::bit_cast<std::uint32_t>(v.scatterAmount)}, {"channels", channels}};
    }

    Json products(const std::filesystem::path& resources, const Json& pixelBits) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        Json rows = Json::array();
        for (unsigned route = 0; route < 4; ++route) {
            for (bool active : {false, true}) {
                ParamSnapshot params;
                params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                params.printProfileKey = "kodak_portra_endura";
                params.scatterHalationControls = {active, 1, 1, 1, 1};
                auto product = JuicerTestSupport::prepare_product(params);
                FjPreparedStateView view{};
                FjErrorBuffer viewError{};
                status(product.view(view, viewError));
                const auto& h = view.optics.scatter_halation;
                for (std::size_t pixel = 0; pixel < (active ? pixelBits.size() : 1); ++pixel) {
                    const auto bits = pixelBits[pixel].get<std::uint32_t>();
                    const std::array<float, 3> scalars{h.scatter_amount, h.scatter_spatial_scale, h.halation_spatial_scale};
                    Json input{{"name", "route/" + std::to_string(route) + "/" + (active ? "active/" + std::to_string(pixel) : "inactive")},
                               {"authority", "completed-route-product"},
                               {"route", route},
                               {"scatter_active", h.scatter_active != 0},
                               {"back_reflection_active", h.back_reflection_active != 0},
                               {"scalars", words(scalars.data(), 3)},
                               {"first_sigma_um", words(h.first_sigma_um, 3)},
                               {"total_strength", words(h.total_strength, 3)},
                               {"recipe_hash", h.hash},
                               {"pixel_size_um", bits}};
                    const auto wire = raw(input);
                    FjScatterHalation descriptor{};
                    FjHalationFrameFailure failure{};
                    std::array<char, 256> diagnostic{};
                    FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                    const auto result = fj_test_scatter_halation_frame_prepare(&wire, &descriptor, &failure, &error);
                    const bool success = result.category == FJ_STATUS_SUCCESS;
                    const bool present = success && wire.recipe.hash != 0;
                    rows.push_back({{"input", input}, {"success", success}, {"present", present}, {"diagnostic", std::string(diagnostic.data(), error.length)}, {"descriptor", present ? project(descriptor) : Json(nullptr)}});
                }
            }
        }
        return rows;
    }
} // namespace HalationFrameTest

namespace HalationFrameTest {
    FjHalationFrameInput raw(const Json& input) {
        FjHalationFrameInput out{};
        out.recipe.scatter_active = input.at("scatter_active").get<bool>() ? 1u : 0u;
        out.recipe.back_reflection_active = input.at("back_reflection_active").get<bool>() ? 1u : 0u;
        const auto scalars = PrintPreflashTest::array<3>(input.at("scalars"));
        const auto firstSigma = PrintPreflashTest::array<3>(input.at("first_sigma_um"));
        const auto strength = PrintPreflashTest::array<3>(input.at("total_strength"));
        out.recipe.scatter_amount = scalars[0];
        out.recipe.scatter_spatial_scale = scalars[1];
        out.recipe.halation_spatial_scale = scalars[2];
        std::copy(firstSigma.begin(), firstSigma.end(), out.recipe.first_sigma_um);
        std::copy(strength.begin(), strength.end(), out.recipe.total_strength);
        out.recipe.hash = input.at("recipe_hash");
        out.pixel_size_um = std::bit_cast<float>(input.at("pixel_size_um").get<std::uint32_t>());
        return out;
    }
    Json project(const FjHalationGaussian& v) {
        const std::array<float, 4> yvv{v.feedforward, v.feedback[0], v.feedback[1], v.feedback[2]};
        return {{"kind", v.kind}, {"radius", v.radius}, {"weights", words(v.fir_weights, 19)}, {"yvv", words(yvv.data(), 4)}};
    }
    Json project(const FjScatterHalation& v) {
        Json channels = Json::array();
        for (const auto& c : v.channels) {
            Json tail = Json::array(), bounce = Json::array();
            for (const auto& g : c.tail)
                tail.push_back(project(g));
            for (const auto& g : c.bounce)
                bounce.push_back(project(g));
            channels.push_back({{"core", project(c.core)}, {"tail", tail}, {"bounce", bounce}, {"strength", std::bit_cast<std::uint32_t>(c.total_strength)}});
        }
        return {{"recipe_hash", v.recipe_hash}, {"scatter_amount", std::bit_cast<std::uint32_t>(v.scatter_amount)}, {"channels", channels}};
    }
    FjHalationGaussian expected_gaussian(const Json& v) {
        FjHalationGaussian out{};
        out.kind = v.at("kind");
        out.radius = v.at("radius");
        copy_words(v.at("weights"), out.fir_weights);
        const auto yvv = PrintPreflashTest::array<4>(v.at("yvv"));
        out.feedforward = yvv[0];
        std::copy(yvv.begin() + 1, yvv.end(), out.feedback);
        return out;
    }
    FjScatterHalation expected(const Json& v) {
        FjScatterHalation out{};
        if (v.is_null())
            return out;
        out.recipe_hash = v.at("recipe_hash");
        out.scatter_amount = std::bit_cast<float>(v.at("scatter_amount").get<std::uint32_t>());
        require(v.at("channels").size() == 3, "halation channels extent");
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const auto& c = v.at("channels")[channel];
            auto& output = out.channels[channel];
            output.core = expected_gaussian(c.at("core"));
            output.total_strength = std::bit_cast<float>(c.at("strength").get<std::uint32_t>());
            require(c.at("tail").size() == 3 && c.at("bounce").size() == 3, "halation schedule extents");
            for (std::size_t index = 0; index < 3; ++index) {
                output.tail[index] = expected_gaussian(c.at("tail")[index]);
                output.bounce[index] = expected_gaussian(c.at("bounce")[index]);
            }
        }
        return out;
    }
    FjHalationFrameFailure expected_failure(const Json& v) {
        return {v.at("reason"), v.at("field"), v.at("channel"), v.at("index"), v.at("sigma_nan").get<bool>() ? std::numeric_limits<double>::quiet_NaN() : std::bit_cast<double>(v.at("sigma_bits").get<std::uint64_t>())};
    }
    void metadata(const FjHalationFrameFailure& a, const FjHalationFrameFailure& b) {
        require(a.reason == b.reason && a.field == b.field && a.channel == b.channel && a.index == b.index &&
                    (std::isnan(b.sigma_px) ? std::isnan(a.sigma_px) : std::bit_cast<std::uint64_t>(a.sigma_px) == std::bit_cast<std::uint64_t>(b.sigma_px)),
                "complete halation frame failure metadata and sigma bits/class");
    }
    void values(const Json& captured, const Json& checked) {
        auto rows = captured.at("cases");
        if (captured.contains("products"))
            for (const auto& row : captured.at("products"))
                rows.push_back(row);
        require(rows.size() == checked.at("cases").size(), "exact halation fixture and checked membership");
        for (const auto& row : rows) {
            const auto input = raw(row.at("input"));
            const auto& contract = checked.at("cases").at(row.at("input").at("name").get<std::string>());
            const auto failureExpected = expected_failure(contract);
            const auto outputExpected = expected(row.at("descriptor"));
            for (auto prepare : {fj_legacy_scatter_halation_frame_prepare, fj_test_scatter_halation_frame_prepare, fj_test_scatter_halation_frame_c_prepare}) {
                FjScatterHalation out{};
                std::memset(&out, 0x55, sizeof(out));
                FjHalationFrameFailure failure{99, 98, 97, 96, -1.0};
                char text[256] = "seed";
                FjErrorBuffer error{text, sizeof(text), 99};
                status(prepare(&input, &out, &failure, &error), contract.at("category"));
                metadata(failure, failureExpected);
                require(project(out) == project(outputExpected), "all21 halation records, nineteen weights/four YVV fields, scalars/hash and unused zeros");
                require(std::string_view(text, error.length) == contract.at("diagnostic").get<std::string>() && text[error.length] == '\0', "exact checked C halation diagnostic base");
            }
            const auto diagnostic = contract.at("diagnostic").get<std::string>();
            require(fj_test_scatter_halation_frame_c_check(&input, &outputExpected, &failureExpected, contract.at("category"), diagnostic.c_str()) != 0, "compiled C11 complete halation fields and envelope reader");
            std::optional<ScatterHalationFrameDescriptor> native{ScatterHalationFrameDescriptor{}};
            JuicerCuda::Failure failure;
            const bool success = JuicerHalation::prepare_frame(recipe(row.at("input")), input.pixel_size_um, native, failure);
            require(success == row.at("success") && native.has_value() == row.at("present"), "native halation optional and success");
            require((native ? project(*native) : Json(nullptr)) == row.at("descriptor"), "native adapter every frozen halation field");
            require(failure.diagnostic == row.at("diagnostic").get<std::string>(), "native platform sigma suffix preserved from typed metadata");
            if (!success)
                status(failure.status, 3);
            std::string publicDiagnostic = "seed";
            require(Spektrafilm::build_scatter_halation_frame_descriptor(recipe(row.at("input")), input.pixel_size_um, native, publicDiagnostic) == success &&
                        native.has_value() == row.at("present") && (native ? project(*native) : Json(nullptr)) == row.at("descriptor") && publicDiagnostic == row.at("diagnostic").get<std::string>(),
                    "actual public producer complete frozen output and native diagnostic");
            if (row.at("input").at("name") == "both") {
                auto changed = outputExpected;
                changed.recipe_hash ^= 1;
                require(!fj_test_scatter_halation_frame_c_check(&input, &changed, &failureExpected, 0, ""), "C11 reader rejects carried hash corruption");
                changed = outputExpected;
                changed.channels[2].bounce[2].fir_weights[18] = 1.0f;
                require(!fj_test_scatter_halation_frame_c_check(&input, &changed, &failureExpected, 0, ""), "C11 reader reaches final unused weight");
                changed = outputExpected;
                changed.channels[2].bounce[2].feedback[2] = std::bit_cast<float>(std::bit_cast<std::uint32_t>(changed.channels[2].bounce[2].feedback[2]) ^ 1u);
                require(!fj_test_scatter_halation_frame_c_check(&input, &changed, &failureExpected, 0, ""), "C11 reader reaches final YVV coefficient");
            }
        }
    }
    void boundary() {
        FjHalationFrameInput input{};
        input.recipe.hash = 17;
        input.recipe.scatter_active = 1;
        input.recipe.scatter_amount = input.recipe.scatter_spatial_scale = 1.0f;
        input.pixel_size_um = 1.0f;
        const FjHalationFrameFailure rawFailure{1, 0, 3, 3, 0};
        for (auto prepare : {fj_legacy_scatter_halation_frame_prepare, fj_test_scatter_halation_frame_prepare, fj_test_scatter_halation_frame_c_prepare}) {
            for (unsigned malformed = 0; malformed < 8; ++malformed) {
                auto probe = input;
                FjScatterHalation out{};
                std::memset(&out, 0x55, sizeof(out));
                FjHalationFrameFailure failure{99, 98, 97, 96, -1};
                alignas(FjScatterHalation) std::array<std::byte, sizeof(FjScatterHalation) + 1> bytes{};
                alignas(FjHalationFrameFailure) std::array<std::byte, sizeof(FjHalationFrameFailure) + 1> failureBytes{};
                char text[128] = "seed";
                FjErrorBuffer error{text, sizeof(text), 99};
                if (malformed == 2) {
                    probe.recipe.hash = 0;
                    probe.recipe.scatter_active = 2;
                }
                if (malformed == 3) {
                    probe.recipe.hash = 0;
                    probe.recipe.back_reflection_active = UINT32_MAX;
                }
                if (malformed == 4)
                    error = {nullptr, 1, 99};
                const auto* source = malformed == 0 ? nullptr : &probe;
                auto* output = malformed == 1 ? nullptr : malformed == 6 ? reinterpret_cast<FjScatterHalation*>(bytes.data() + 1)
                                                                         : &out;
                auto* failed = malformed == 5 ? nullptr : malformed == 7 ? reinterpret_cast<FjHalationFrameFailure*>(failureBytes.data() + 1)
                                                                         : &failure;
                status(prepare(source, output, failed, &error), 2);
                if (output == &out)
                    require(project(out) == project(FjScatterHalation{}), "independent halation output clearing before peer/diagnostic admission");
                if (failed == &failure)
                    metadata(failure, rawFailure);
                if (malformed != 4)
                    require(std::string_view(text) == "invalid scatter/halation frame arguments", "exact raw halation diagnostic");
                else
                    require(error.length == 0, "malformed backing clears diagnostic length");
            }
            for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{7}}) {
                FjScatterHalation out{};
                FjHalationFrameFailure failure{};
                char text[8] = "seed";
                FjErrorBuffer error{capacity ? text : nullptr, capacity, 99};
                status(prepare(nullptr, &out, &failure, &error), 2);
                metadata(failure, rawFailure);
                require(error.length == (capacity ? capacity - 1 : 0) && (!capacity || text[error.length] == '\0'), "bounded/zero-capacity halation diagnostics");
            }
            FjScatterHalation out{};
            FjHalationFrameFailure failure{};
            status(prepare(&input, &out, &failure, nullptr));
            status(prepare(nullptr, &out, &failure, nullptr), 2);
        }
        status(fj_test_scatter_halation_frame_fault(1));
        status(fj_test_scatter_halation_frame_fault(5), 2);
        std::thread other([&] {
            FjScatterHalation out{};
            FjHalationFrameFailure failure{};
            status(fj_legacy_scatter_halation_frame_prepare(&input, &out, &failure, nullptr));
        });
        other.join();
        FjScatterHalation out{};
        FjHalationFrameFailure failure{};
        char text[128]{};
        FjErrorBuffer error{text, sizeof(text), 0};
        status(fj_test_scatter_halation_frame_prepare(&input, &out, &failure, &error), 8);
        metadata(failure, {4, 0, 3, 3, 0});
        require(project(out) == project(FjScatterHalation{}) && std::string_view(text) == "scatter/halation frame preparation panicked", "whole-operation panic clears output and exact envelope");
        status(fj_legacy_scatter_halation_frame_prepare(&input, &out, &failure, nullptr));
        ScatterHalationOpticsRecipe native;
        native.hash = input.recipe.hash;
        native.scatterActive = true;
        native.scatterAmount = native.scatterSpatialScale = 1;
        for (unsigned fault : {1u, 2u, 3u, 4u}) {
            status(fj_test_scatter_halation_frame_fault(fault));
            std::optional<ScatterHalationFrameDescriptor> descriptor{ScatterHalationFrameDescriptor{}};
            JuicerCuda::Failure rejected;
            require(!JuicerHalation::prepare_frame(native, 1, descriptor, rejected) && !descriptor, "native halation rejects panic/corrupt representation absent");
            status(rejected.status, fault == 1 ? 8u : 3u);
            require(rejected.diagnostic == (fault == 1 ? "scatter/halation frame preparation panicked" : "scatter/halation frame projection invalid"), "typed halation projection failure");
            require(JuicerHalation::prepare_frame(native, 1, descriptor, rejected) && descriptor, "one-shot halation fault recovery");
        }
    }
} // namespace HalationFrameTest
