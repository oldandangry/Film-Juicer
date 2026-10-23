namespace FilmReferenceTest {
    thread_local unsigned nextProjection = 0;
    thread_local bool observeHold = false;
    thread_local InstanceState* supersedingState = nullptr;
    thread_local const ParamSnapshot* supersedingParams = nullptr;
    Json project(const SyntheticFilmReferenceRecipe& reference) {
        return {{"baseline_raw_rgb", words(reference.baselineRawRgb.data(), 3)},
                {"compensated_raw_rgb", words(reference.compensatedRawRgb.data(), 3)},
                {"baseline_density_cmy", words(reference.baselineDensityCmy.data(), 3)},
                {"compensated_density_cmy", words(reference.compensatedDensityCmy.data(), 3)},
                {"meter_scale", std::bit_cast<std::uint32_t>(reference.baselineMeterScale)},
                {"hash", reference.hash}};
    }

    ParamSnapshot controls(unsigned method, Spektrafilm::ScanRoute route, unsigned variant) {
        ParamSnapshot params;
        params.scanRoute = route;
        params.filmProfileKey = route < Spektrafilm::ScanRoute::PositiveDirectScan ? "kodak_portra_400" : "fujifilm_provia_100f";
        params.printProfileKey = "kodak_portra_endura";
        params.spectralUpsamplingMode = static_cast<int>(method);
        params.cameraAutoExposureEnabled = variant == 1 ? 0 : 1;
        params.cameraExposureCompensationEv = variant == 0 ? 0.0 : variant == 1 ? .75
                                                                                : -.5;
        params.inputCompressionEnabled = 0;
        params.grainControls.active = false;
        return params;
    }

    Json products(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerProcess::root().ensure_bootstrap();
        Json rows = Json::array();
        for (unsigned method = 0; method < 3; ++method)
            for (unsigned route = 0; route < 4; ++route)
                for (unsigned variant = 0; variant < 3; ++variant) {
                    const auto params = controls(method, static_cast<Spektrafilm::ScanRoute>(route), variant);
                    FocusedRenderStateBuildProduct product;
                    std::string diagnostic;
                    const bool built = route % 2 ? build_print_render_state_product(params, product, diagnostic) : build_direct_render_state_product(params, product, diagnostic);
                    require(built, diagnostic.c_str());
                    const auto& reference = product.recipe.scannerOutput.syntheticFilmReference;
                    Json queries = Json::array();
                    for (unsigned exposure = 0; exposure < 2; ++exposure) {
                        const auto& raw = exposure == 0 ? reference.baselineRawRgb : reference.compensatedRawRgb;
                        for (unsigned channel = 0; channel < 3; ++channel) {
                            const float productValue = raw[channel] * reference.baselineMeterScale;
                            const float argument = productValue + 1.0e-10f;
                            const float nativeLog = std::log10(argument);
                            std::array<float, 6> rust{};
                            std::array<double, 3> rustDouble{};
                            fj_test_scanner_correction_primitive(argument, 0, 1, rust.data(), rustDouble.data());
                            queries.push_back({{"product", std::bit_cast<std::uint32_t>(productValue)}, {"argument", std::bit_cast<std::uint32_t>(argument)}, {"native_log10", std::bit_cast<std::uint32_t>(nativeLog)}, {"rust_log10", std::bit_cast<std::uint32_t>(rust[1])}, {"query", std::bit_cast<std::uint32_t>(nativeLog * product.recipe.filmDevelop.densityCurveGamma[channel])}});
                        }
                    }
                    Json row{{"method", method}, {"route", route}, {"variant", variant}, {"reference", project(reference)}, {"queries", queries}, {"snapshot_hash", hash_params(params)}, {"development_hash", product.recipe.filmDevelop.hash}, {"scanner_hash", product.recipe.scannerOutput.hash}, {"balance_hash", product.recipe.print.balance.hash}, {"print_hash", product.recipe.print.hash}, {"final_hash", product.recipe.hash}, {"upload_hash", product.payload.uploadCoreHash}, {"payload_scanner_hash", product.payload.scannerHash}};
                    if (route == 0 && ((method == 0 && variant == 0) || (method == 1 && variant == 1))) {
                        const auto& raw = product.recipe.filmRaw;
                        const auto& basis = Spectral::context().mallettBasis;
                        Json sensitivity = Json::array();
                        for (const auto& sample : raw.finalSensitivity)
                            for (float value : sample)
                                sensitivity.push_back(std::bit_cast<std::uint32_t>(value));
                        row["input"] = {{"key", params.filmProfileKey}, {"gamma", words(product.recipe.filmDevelop.densityCurveGamma.data(), 3)}, {"manual_ev", std::bit_cast<std::uint32_t>(raw.manualExposureCompensationEv)}, {"auto", raw.autoExposureEnabled}, {"green_scale", std::bit_cast<std::uint32_t>(raw.mallettGreenMidgrayScale)}, {"white", words(raw.projectionWhiteXYZ.data(), 3)}, {"basis", method == 1 ? words(basis.data.data(), basis.data.size()) : Json::array()}, {"illuminant", words(product.payload.exposureTables.illum.data(), 81)}, {"sensitivity", sensitivity}, {"tc", product.payload.filmTcLut ? words(product.payload.filmTcLut->samples().data(), product.payload.filmTcLut->samples().size()) : Json::array()}};
                    }
                    rows.push_back(std::move(row));
                }
        return rows;
    }

    void check_primitives(const Json& captured) {
        require(captured.at("products").size() == 36, "complete reference cohort membership");
        for (const auto& row : captured.at("products"))
            for (const auto& query : row.at("queries"))
                require(query.at("native_log10") == query.at("rust_log10"), "reference log10 primitive differs; owner decision required");
    }

    Json numerical_failures(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerProcess::root().ensure_bootstrap();
        auto params = controls(1, Spektrafilm::ScanRoute::NegativeDirectScan, 0);
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        require(build_direct_render_state_product(params, product, diagnostic), diagnostic.c_str());
        JuicerAssets::AssetBridge assets(resources);
        require(assets.load_catalog().valid, "reference completed-profile catalog");
        const auto source = assets.film_source(params.filmProfileKey);
        auto development = source->prepare_development(params.scanRoute, params.filmGammaFactor, false);
        struct BasisScope {
            Spectral::MallettBasis saved = Spectral::gMallettBasis;
            bool available = Spectral::mallett_available();
            ~BasisScope() {
                Spectral::gMallettBasis = std::move(saved);
                Spectral::gMallettAvailable.store(available);
            }
        } basis;
        Spectral::gMallettBasis.rows = 81;
        Spectral::gMallettBasis.cols = 3;
        Spectral::gMallettBasis.data.assign(243, 1.0f);
        std::array<float, 81> light{};
        light.fill(1);
        Json rows = Json::array();
        for (unsigned mutation = 0; mutation < 9; ++mutation) {
            auto recipe = product.recipe;
            auto& raw = recipe.filmRaw;
            raw.mallettGreenMidgrayScale = 1;
            raw.manualExposureCompensationEv = 0;
            raw.autoExposureEnabled = true;
            for (auto& sample : raw.finalSensitivity)
                sample = {1, 1, 1};
            if (mutation == 0)
                raw.manualExposureCompensationEv = std::numeric_limits<float>::infinity();
            if (mutation == 1)
                raw.manualExposureCompensationEv = std::numeric_limits<float>::quiet_NaN();
            if (mutation == 2)
                raw.mallettGreenMidgrayScale = -1;
            if (mutation == 3)
                raw.mallettGreenMidgrayScale = std::numeric_limits<float>::max();
            if (mutation == 4)
                for (auto& sample : raw.finalSensitivity)
                    sample = {0, 0, 0};
            if (mutation == 5)
                raw.mallettGreenMidgrayScale = 3.0e36f;
            if (mutation == 6)
                for (auto& sample : raw.finalSensitivity)
                    sample = {0, 1.0e-40f, 0};
            if (mutation == 7)
                Spectral::gMallettBasis.rows = 80;
            if (mutation == 8) {
                Spectral::gMallettBasis.rows = 81;
                Spectral::gMallettAvailable = false;
            }
            const auto before = project(recipe.scannerOutput.syntheticFilmReference);
            const auto scanner = recipe.scannerOutput.hash, final = recipe.hash;
            const bool success = Spektrafilm::finish_synthetic_reference_recipes(recipe, development.hold, nullptr, light, nullptr, diagnostic);
            rows.push_back({{"mutation", mutation}, {"success", success}, {"diagnostic", diagnostic}, {"reference_unchanged", project(recipe.scannerOutput.syntheticFilmReference) == before}, {"scanner_unchanged", recipe.scannerOutput.hash == scanner}, {"final_unchanged", recipe.hash == final}, {"result", project(recipe.scannerOutput.syntheticFilmReference)}});
        }
        return rows;
    }

    Json project(const FjFilmReference& reference) {
        return {{"baseline_raw_rgb", words(reference.baseline_raw_rgb, 3)}, {"compensated_raw_rgb", words(reference.compensated_raw_rgb, 3)}, {"baseline_density_cmy", words(reference.baseline_density_cmy, 3)}, {"compensated_density_cmy", words(reference.compensated_density_cmy, 3)}, {"meter_scale", std::bit_cast<std::uint32_t>(reference.meter_scale)}, {"hash", reference.hash}};
    }
    struct Inputs {
        std::vector<float> basis, light, sensitivity, tc;
        FjFilmReferenceInput wire{};
        explicit Inputs(const Json& row) {
            const auto& input = row.at("input");
            const auto load = [](const Json& source, std::vector<float>& target) {
                for (const auto& word : source)
                    target.push_back(std::bit_cast<float>(word.get<std::uint32_t>()));
                return FjFloatSpan{target.empty() ? nullptr : target.data(), target.size()};
            };
            wire.method = row.at("method");
            wire.auto_exposure_enabled = input.at("auto").get<bool>() ? 1u : 0u;
            wire.manual_exposure_ev = std::bit_cast<float>(input.at("manual_ev").get<std::uint32_t>());
            wire.mallett_green_scale = std::bit_cast<float>(input.at("green_scale").get<std::uint32_t>());
            copy_words(input.at("white"), wire.projection_white_xyz);
            wire.basis_rgb = load(input.at("basis"), basis);
            wire.tc_lut = load(input.at("tc"), tc);
            if (wire.method == 1) {
                wire.film_illuminant = load(input.at("illuminant"), light);
                wire.sensitivity_rgb = load(input.at("sensitivity"), sensitivity);
            }
        }
    };
    using Prepare = FjStatus (*)(const FjFilmDevelopmentOwner*, const FjFilmReferenceInput*, FjFilmReference*, FjFilmReferenceFailure*, FjErrorBuffer*);

    void boundary(const std::filesystem::path& resources, const Json& captured) {
        check_primitives(captured);
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerProcess::root().ensure_bootstrap();
        Assets assets(resources);
        unsigned count = 0;
        for (const auto& row : captured.at("products")) {
            if (!row.contains("input"))
                continue;
            ++count;
            auto developmentSpec = row.at("input");
            developmentSpec["route"] = row.at("route");
            JuicerAssets::DevelopmentHold development{assets.development(developmentSpec)};
            Inputs inputs(row);
            for (Prepare prepare : {fj_legacy_film_reference_prepare, fj_test_film_reference_prepare, fj_test_film_reference_c_prepare}) {
                FjFilmReference out{};
                FjFilmReferenceFailure failure{99, 98, 97, 96};
                std::array<char, 512> text{};
                FjErrorBuffer error{text.data(), text.size(), 99};
                status(prepare(development.get(), &inputs.wire, &out, &failure, &error));
                require(project(out) == row.at("reference"), "all frozen reference words and exact 52-byte identity");
                require(failure.reason == 0 && failure.exposure == 0 && failure.operation == 0 && failure.channel == 0 && error.length == 0, "complete success clears failure/diagnostic");
                auto unusedScalars = inputs.wire;
                if (unusedScalars.method == 1)
                    std::fill_n(unusedScalars.projection_white_xyz, 3, std::numeric_limits<float>::quiet_NaN());
                else
                    unusedScalars.mallett_green_scale = std::numeric_limits<float>::quiet_NaN();
                status(prepare(development.get(), &unusedScalars, &out, &failure, &error));
                require(project(out) == row.at("reference"), "method-local scalar dependencies preserve frozen output when unused values change");
                for (unsigned malformed = 0; malformed < 28; ++malformed) {
                    auto input = inputs.wire;
                    const FjFilmReferenceInput* argument = &input;
                    const FjFilmDevelopmentOwner* owner = development.get();
                    auto* result = &out;
                    auto* details = &failure;
                    auto* diagnostic = &error;
                    if (malformed == 0)
                        input.method = 3;
                    if (malformed == 1)
                        input.auto_exposure_enabled = 2;
                    auto& active = input.method == 1 ? input.basis_rgb : input.tc_lut;
                    if (malformed == 2)
                        --active.count;
                    if (malformed == 3)
                        active.data = nullptr;
                    if (malformed == 4)
                        active.count = 0;
                    if (malformed == 5)
                        active.data = reinterpret_cast<const float*>(std::uintptr_t{1});
                    if (malformed == 6)
                        active.count = SIZE_MAX;
                    if (malformed == 7)
                        result = nullptr;
                    if (malformed == 8)
                        details = nullptr;
                    if (malformed == 9)
                        error = {nullptr, 1, 99};
                    if (malformed == 10)
                        diagnostic = reinterpret_cast<FjErrorBuffer*>(std::uintptr_t{1});
                    if (malformed == 11)
                        owner = nullptr;
                    if (malformed == 12)
                        argument = nullptr;
                    if (malformed == 13)
                        owner = reinterpret_cast<const FjFilmDevelopmentOwner*>(std::uintptr_t{1});
                    if (malformed == 14)
                        argument = reinterpret_cast<const FjFilmReferenceInput*>(std::uintptr_t{1});
                    if (malformed == 15)
                        result = reinterpret_cast<FjFilmReference*>(std::uintptr_t{1});
                    if (malformed == 16)
                        details = reinterpret_cast<FjFilmReferenceFailure*>(std::uintptr_t{1});
                    if (malformed == 17)
                        result = reinterpret_cast<FjFilmReference*>(UINTPTR_MAX - 7);
                    if (malformed == 18)
                        details = reinterpret_cast<FjFilmReferenceFailure*>(UINTPTR_MAX - 3);
                    if (malformed == 19)
                        error = {reinterpret_cast<char*>(UINTPTR_MAX - 1), 8, 99};
                    if (malformed == 20)
                        error.capacity = SIZE_MAX;
                    if (malformed == 21)
                        active.data = reinterpret_cast<const float*>(UINTPTR_MAX - 3);
                    auto& unused = input.method == 1 ? input.tc_lut : input.basis_rgb;
                    if (malformed == 22)
                        unused = {inputs.wire.method == 1 ? inputs.basis.data() : inputs.tc.data(), 0};
                    if (malformed == 23)
                        unused = {nullptr, 1};
                    if (malformed == 24)
                        unused = {inputs.wire.method == 1 ? inputs.basis.data() : inputs.tc.data(), 1};
                    if (malformed == 25)
                        input.film_illuminant = {nullptr, 81};
                    if (malformed == 26)
                        input.sensitivity_rgb = {nullptr, 243};
                    if (malformed == 27)
                        diagnostic = reinterpret_cast<FjErrorBuffer*>(UINTPTR_MAX - 7);
                    out.hash = 99;
                    failure = {99, 98, 97, 96};
                    status(prepare(owner, argument, result, details, diagnostic), FJ_STATUS_UNSUPPORTED_INPUT);
                    if (result == &out)
                        require(project(out) == project(FjFilmReference{}), "independent result clearing");
                    if (details == &failure)
                        require(failure.reason == 1 && failure.exposure == 0 && failure.operation == 0 && failure.channel == 0, "independent RAW failure clearing");
                    error = {text.data(), text.size(), 0};
                }
                const std::vector<unsigned> operations = inputs.wire.method == 1 ? std::vector<unsigned>{1, 6} : std::vector<unsigned>{1, 2, 3, 4, 5};
                for (unsigned operation : operations)
                    for (unsigned exposure = 1; exposure <= 2; ++exposure) {
                        for (unsigned fault = 1; fault <= 4; ++fault) {
                            if (operation >= 2 && operation <= 4 && fault > 2)
                                continue;
                            // Fixed A7 raw arming intentionally rejects allocation;
                            // the core hook covers that category without changing it.
                            if (fault == 3 && operation != 5)
                                continue;
                            if (operation == 1 || operation == 6)
                                status(fj_test_exposure_arm_fault(operation == 1 ? 5u : 6u, exposure, fault));
                            else if (operation == 5)
                                status(fj_test_tc_lut_arm_fault(2, exposure, fault));
                            else
                                status(fj_test_color_arm_fault(operation == 2 ? 5u : operation == 3 ? 8u
                                                                                                    : 2u,
                                                               exposure,
                                                               fault));
                            const unsigned category = fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                             : fault == 3   ? FJ_STATUS_ALLOCATION_FAILURE
                                                                                                            : FJ_STATUS_PREPARATION_FAILURE;
                            status(prepare(development.get(), &inputs.wire, &out, &failure, &error), category);
                            require(project(out) == project(FjFilmReference{}), "fault result entirely zero");
                            require(failure.reason == (fault == 2 ? 6u : fault == 4 ? 2u
                                                                                    : 5u) &&
                                        failure.channel == 0 && failure.exposure == (fault == 2 ? 0u : exposure) && failure.operation == (fault == 2 ? 0u : operation),
                                    "exact composite fault details");
                            const std::array messages{"", "Reference source preparation: exposure production boundary fault", "Input matrix preparation failed", "Linear sRGB to XYZ conversion failed", "CAT16 scalar adaptation failed", "Film TC sampling: TC production boundary fault", "Mallett reference preparation: exposure production boundary fault"};
                            require(std::string_view(text.data(), error.length) == (fault == 2 ? "film reference preparation panicked" : fault == 3 ? "film reference dependency allocation failed"
                                                                                                                                     : fault == 4   ? "MalformedRequiredResource phase=3B field=synthetic_reference_raw"
                                                                                                                                                    : messages[operation]),
                                    "exact composite fault diagnostic");
                            status(prepare(development.get(), &inputs.wire, &out, &failure, nullptr));
                            require(project(out) == row.at("reference"), "one-shot composite recovery");
                        }
                    }
                auto missing = inputs.wire;
                if (missing.method == 1)
                    missing.basis_rgb = {};
                else
                    missing.tc_lut = {};
                status(prepare(development.get(), &missing, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
                require(failure.reason == 2 && failure.exposure == 1 && failure.operation == 0 && failure.channel == 0, "missing reconstruction follows source");
                const unsigned leaf = inputs.wire.method == 1 ? 6u : 5u;
                if (leaf == 6)
                    status(fj_test_exposure_arm_fault(6, 1, 1));
                else
                    status(fj_test_tc_lut_arm_fault(2, 1, 1));
                status(prepare(development.get(), &missing, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
                require(failure.operation == 0, "missing resource skips the later armed operation");
                status(prepare(development.get(), &inputs.wire, &out, &failure, &error), FJ_STATUS_UNSUPPORTED_INPUT);
                require(failure.reason == 5 && failure.operation == leaf && failure.exposure == 1, "next matching call consumes the skipped fault");
                status(prepare(development.get(), &inputs.wire, &out, &failure, nullptr));
                FjErrorBuffer zeroCapacity{reinterpret_cast<char*>(std::uintptr_t{1}), 0, 99};
                status(prepare(development.get(), &inputs.wire, &out, &failure, &zeroCapacity));
                require(zeroCapacity.length == 0 && project(out) == row.at("reference"), "zero-capacity diagnostic and owning result");
                if (inputs.wire.method == 1) {
                    std::array<float, 243> basis{}, sensitivity{};
                    std::array<float, 81> light{};
                    basis.fill(1);
                    light.fill(1);
                    for (const auto& expected : captured.at("numerical_failures")) {
                        const unsigned mutation = expected.at("mutation");
                        auto input = inputs.wire;
                        input.basis_rgb = {basis.data(), basis.size()};
                        input.film_illuminant = {light.data(), light.size()};
                        input.sensitivity_rgb = {sensitivity.data(), sensitivity.size()};
                        input.mallett_green_scale = 1;
                        input.manual_exposure_ev = 0;
                        input.auto_exposure_enabled = 1;
                        sensitivity.fill(1);
                        if (mutation == 0)
                            input.manual_exposure_ev = std::numeric_limits<float>::infinity();
                        if (mutation == 1)
                            input.manual_exposure_ev = std::numeric_limits<float>::quiet_NaN();
                        if (mutation == 2)
                            input.mallett_green_scale = -1;
                        if (mutation == 3)
                            input.mallett_green_scale = std::numeric_limits<float>::max();
                        if (mutation == 4)
                            sensitivity.fill(0);
                        if (mutation == 5)
                            input.mallett_green_scale = 3.0e36f;
                        if (mutation == 6)
                            for (std::size_t i = 0; i < 81; ++i) {
                                sensitivity[3 * i] = 0;
                                sensitivity[3 * i + 1] = 1.0e-40f;
                                sensitivity[3 * i + 2] = 0;
                            }
                        if (mutation == 7)
                            input.basis_rgb = {};
                        status(prepare(development.get(), &input, &out, &failure, &error), expected.at("success").get<bool>() ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                        require(std::string(text.data(), error.length) == expected.at("diagnostic").get<std::string>(), "independently frozen native numerical diagnostic");
                        if (expected.at("success").get<bool>()) {
                            require(project(out) == expected.at("result"), "independent native literal reference and identity");
                        } else {
                            require(project(out) == project(FjFilmReference{}), "numerical failure clears complete foreign result");
                            const unsigned reason = mutation < 4 || mutation == 7 ? 2u : mutation < 6 ? 3u
                                                                                                      : 4u;
                            const unsigned exposure = mutation < 2 ? 2u : reason == 3 ? 0u
                                                                                      : 1u;
                            const unsigned operation = mutation < 2 ? 1u : mutation < 4 ? 6u
                                                                       : mutation < 6   ? 7u
                                                                       : mutation == 6  ? 8u
                                                                                        : 0u;
                            require(failure.reason == reason && failure.exposure == exposure && failure.operation == operation && failure.channel == (mutation == 6 ? 1u : 0u), "first numerical failure details retain stage/exposure/channel");
                        }
                    }
                }
            }
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            require(build_direct_render_state_product(controls(row.at("method"), Spektrafilm::ScanRoute::NegativeDirectScan, row.at("variant")), product, diagnostic), diagnostic.c_str());
            std::array<float, 81> light{};
            std::copy_n(product.payload.exposureTables.illum.begin(), 81, light.begin());
            SyntheticFilmReferenceRecipe reference;
            std::array<std::size_t, 2> before{}, after{};
            status(fj_test_tc_lut_allocation_counts(before.data()));
            require(JuicerAssets::prepare_film_reference(development, product.recipe.filmRaw, product.payload.filmTcLut ? &*product.payload.filmTcLut : nullptr, light, reference, diagnostic), diagnostic.c_str());
            status(fj_test_tc_lut_allocation_counts(after.data()));
            require(before == after, "synchronous reference neither transfers nor releases the existing TC allocation");
            require(project(reference) == row.at("reference"), "native adapter projects the complete safe reference");
            for (unsigned fault = 1; fault <= 10; ++fault) {
                nextProjection = fault;
                bool rejected = false;
                try {
                    JuicerAssets::prepare_film_reference(development, product.recipe.filmRaw, product.payload.filmTcLut ? &*product.payload.filmTcLut : nullptr, light, reference, diagnostic);
                } catch (const JuicerCuda::ExecutionFailure& error) {
                    rejected = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE && error.failure.diagnostic == "film reference projection invalid";
                }
                require(rejected && nextProjection == 0 && project(reference) == row.at("reference"), "native integrity rejects corrupt envelopes without mutating the prior reference");
            }
            status(fj_legacy_assets_release_cached_payloads(assets.value, nullptr));
            FjFilmReference detached{};
            FjFilmReferenceFailure failure{};
            status(fj_legacy_film_reference_prepare(development.get(), &inputs.wire, &detached, &failure, nullptr));
            require(project(detached) == row.at("reference"), "live moved Development hold remains usable after source-cache detachment");
        }
        require(count == 2 && fj_test_film_development_live_owners() == 0, "both method witnesses release temporary holds");
    }

    void callers(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerProcess::root().ensure_bootstrap();
        for (unsigned method = 0; method < 3; ++method) {
            for (unsigned route = 0; route < 4; ++route) {
                auto params = controls(method, static_cast<Spektrafilm::ScanRoute>(route), 0);
                InstanceState state;
                const auto pending = [&](const ParamSnapshot& value) {
                    std::lock_guard<std::mutex> lock(state.pending.m);
                    state.pending.value = PendingParamsState::Valid{value, hash_params(value)};
                };
                const auto recipe = [&](const PendingRenderAdmissionResult& value) -> const RenderRecipe& {
                    require(route % 2 ? bool(value.printState) : bool(value.directState), "complete reference publication");
                    return route % 2 ? value.printState->recipe : value.directState->recipe;
                };
                pending(params);
                auto current = admit_pending_render_state(state);
                const auto original = current;
                const auto originalReference = project(recipe(original).scannerOutput.syntheticFilmReference);
                const auto operations = method == 1 ? std::vector<unsigned>{5, 6} : std::vector<unsigned>{2};
                for (unsigned operation : operations) {
                    for (unsigned index : {1u, 2u}) {
                        for (unsigned fault : {1u, 2u, 3u, 4u}) {
                            if (method == 1 && fault == 3)
                                continue; // Existing fixed A7 arm contract rejects allocation injection.
                            const bool tc = method != 1;
                            status(tc ? fj_test_tc_lut_arm_fault(operation, index + 1, fault) : fj_test_exposure_arm_fault(operation, index, fault));
                            const auto counter = state.buildCounterNext.load(), lastHash = state.lastHash.load();
                            const auto reused = admit_pending_render_state(state);
                            require(reused.directState == current.directState && reused.printState == current.printState && state.buildCounterNext.load() == counter, "unchanged publication skips reference and preserves the armed fault");
                            params.cameraExposureCompensationEv += .125;
                            pending(params);
                            PendingRenderAdmissionResult failed;
                            bool allocationFailed = false;
                            observeHold = true;
                            try {
                                failed = admit_pending_render_state(state);
                            } catch (const std::bad_alloc&) {
                                allocationFailed = true;
                            }
                            observeHold = false;
                            require((tc ? fj_test_tc_lut_fault_consumed(operation, fault) : fj_test_exposure_fault_consumed(operation, fault)) == 1, "actual caller consumes the selected reference visit after focused midgray");
                            require(allocationFailed == (fault == 3), "allocation remains typed through admission");
                            if (!allocationFailed)
                                require(failed.status == PendingRenderAdmissionStatus::RebuildFailed && !failed.directState && !failed.printState, "reference failure blocks new admission");
                            require(state.buildCounterNext.load() == counter && state.lastHash.load() == lastHash && fj_test_film_development_live_owners() == 0, "failed reference neither advances identity nor leaks its moved Development hold");
                            const bool selectedPresent = route % 2 ? bool(JuicerAtomic::load_shared_ptr(&state.activePrintState)) : bool(JuicerAtomic::load_shared_ptr(&state.activeDirectState));
                            require(selectedPresent == (fault != 4), "ordinary false clears the selected slot; exceptions preserve the prior publication");
                            current = admit_pending_render_state(state);
                            recipe(current);
                            require(state.buildCounterNext.load() == counter + 1 && fj_test_film_development_live_owners() == 0, "one-shot reference recovery publishes once and releases the hold");
                            require(project(recipe(original).scannerOutput.syntheticFilmReference) == originalReference, "admitted immutable reference remains readable through failures and recovery");
                        }
                    }
                }
                params.cameraExposureCompensationEv += .125;
                pending(params);
                auto latest = params;
                latest.cameraExposureCompensationEv += .125;
                supersedingState = &state;
                supersedingParams = &latest;
                const auto counter = state.buildCounterNext.load();
                status(fj_test_exposure_arm_fault(5, 2, 4));
                current = admit_pending_render_state(state);
                require(!supersedingState && !supersedingParams && hash_params(current.snapshot) == hash_params(latest) && state.lastHash.load() == hash_params(latest) && state.buildCounterNext.load() == counter + 1, "superseded reference failure recovers to the latest complete snapshot");
                require(fj_test_exposure_fault_consumed(5, 4) == 1 && fj_test_film_development_live_owners() == 0, "supersession releases its moved hold and consumes only the failed visit");
            }
        }
    }
} // namespace FilmReferenceTest

namespace JuicerAssets::RecipeTest {
    void film_reference(FjStatus& status, FjFilmReferenceFailure& failure, FjFilmReference&, FjErrorBuffer& error) {
        if (FilmReferenceTest::observeHold)
            require(fj_test_film_development_live_owners() == 1, "one original Development hold survives to synchronous reference completion");
        if (FilmReferenceTest::supersedingState) {
            auto* state = std::exchange(FilmReferenceTest::supersedingState, nullptr);
            const auto* params = std::exchange(FilmReferenceTest::supersedingParams, nullptr);
            std::lock_guard<std::mutex> lock(state->pending.m);
            state->pending.value = PendingParamsState::Valid{*params, hash_params(*params)};
        }
        switch (std::exchange(FilmReferenceTest::nextProjection, 0u)) {
            case 1:
                status.api = 1;
                break;
            case 2:
                status.native_code = 1;
                break;
            case 3:
                status.category = 99;
                break;
            case 4:
                failure.reason = 99;
                break;
            case 5:
                failure.exposure = 99;
                break;
            case 6:
                failure.operation = 99;
                break;
            case 7:
                failure.channel = 99;
                break;
            case 8:
                error.length = error.capacity;
                break;
            case 9:
                status.category = FJ_STATUS_UNSUPPORTED_INPUT;
                failure.reason = FJ_FILM_REFERENCE_RAW;
                break;
            case 10:
                error.length = 1;
                break;
            default:
                break;
        }
    }
} // namespace JuicerAssets::RecipeTest
