#include <exception>

namespace FilmReferenceTest {
    constexpr std::uint32_t kBlockedPendingKind = 6;
    thread_local unsigned nextProjection = 0;

    struct PendingCaptureSupersession {
        RenderInstance* instance = nullptr;
        const ParamSnapshot* latest = nullptr;
        bool fired = false;
        std::exception_ptr failure;
    };

    void supersede_after_prepare(void* opaque) noexcept {
        auto& supersession = *static_cast<PendingCaptureSupersession*>(opaque);
        if (!supersession.latest || supersession.fired)
            return;
        supersession.fired = true;
        try {
            JuicerTestSupport::capture(*supersession.instance, *supersession.latest);
        } catch (...) {
            supersession.failure = std::current_exception();
        }
    }

    Json project(const FjFilmReference& reference);
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

    Json products(const std::filesystem::path& resources, const Json& expectedProducts) {
        JuicerProcess::ProcessRuntime owner;
        owner.create(resources);
        Json rows = Json::array();
        for (unsigned method = 0; method < 3; ++method)
            for (unsigned route = 0; route < 4; ++route)
                for (unsigned variant = 0; variant < 3; ++variant) {
                    JuicerTests::CaseEvent case_event("cpp", "film-reference/" + std::to_string(route) + "/" + std::to_string(method) + "/" + std::to_string(variant));
                    const auto params = controls(method, static_cast<Spektrafilm::ScanRoute>(route), variant);
                    auto state = JuicerTestSupport::prepare_product(owner.borrow().assets, params);
                    FjPreparedStateView product{};
                    FjErrorBuffer error{};
                    status(state.view(product, error));
                    const auto& reference = product.film_reference;
                    Json queries = Json::array();
                    for (unsigned exposure = 0; exposure < 2; ++exposure) {
                        const auto* raw = exposure == 0 ? reference.baseline_raw_rgb : reference.compensated_raw_rgb;
                        for (unsigned channel = 0; channel < 3; ++channel) {
                            const float productValue = raw[channel] * reference.meter_scale;
                            const float argument = productValue + 1.0e-10f;
                            const float nativeLog = std::log10(argument);
                            std::array<float, 6> rust{};
                            std::array<double, 3> rustDouble{};
                            fj_test_scanner_correction_primitive(argument, 0, 1, rust.data(), rustDouble.data());
                            queries.push_back({{"product", std::bit_cast<std::uint32_t>(productValue)}, {"argument", std::bit_cast<std::uint32_t>(argument)}, {"native_log10", std::bit_cast<std::uint32_t>(nativeLog)}, {"rust_log10", std::bit_cast<std::uint32_t>(rust[1])}, {"query", std::bit_cast<std::uint32_t>(nativeLog * product.film_development.gamma_rgb[channel])}});
                        }
                    }
                    Json row{{"method", method}, {"route", route}, {"variant", variant}, {"reference", project(reference)}, {"queries", queries}, {"snapshot_hash", hash_params(params)}, {"development_hash", product.film_development.hash}, {"scanner_hash", product.scanner_recipe_hash}, {"balance_hash", product.print_balance.hash}, {"print_hash", product.print_recipe_hash}, {"final_hash", product.recipe_hash}, {"upload_hash", product.upload_core_hash}, {"payload_scanner_hash", product.scanner_hash}};
                    if (route == 0 && ((method == 0 && variant == 0) || (method == 1 && variant == 1))) {
                        const auto& raw = product.film_exposure;
                        const std::size_t basisCount = method == 1 ? raw.mallett_basis.count : 0;
                        row["input"] = {{"key", params.filmProfileKey}, {"gamma", words(product.film_development.gamma_rgb, 3)}, {"manual_ev", std::bit_cast<std::uint32_t>(raw.manual_exposure_ev)}, {"auto", (raw.flags & FJ_FILM_AUTO_EXPOSURE) != 0}, {"green_scale", std::bit_cast<std::uint32_t>(raw.mallett_green_midgray_scale)}, {"white", words(product.input_policy.projection_white_xyz, 3)}, {"basis", words(raw.mallett_basis.data, basisCount)}, {"illuminant", words(product.exposure_tables.illuminant.data, product.exposure_tables.illuminant.count)}, {"sensitivity", words(raw.sensitivity_rgb.data, raw.sensitivity_rgb.count)}, {"tc", words(raw.tc_lut_rgba.data, raw.tc_lut_rgba.count)}};
                    }
                    rows.push_back(std::move(row));
                    require(rows.back() == expectedProducts.at(rows.size() - 1), "per-case complete film reference product");
                    case_event.passed();
                }
        return rows;
    }

    void check_primitives(const Json& captured) {
        JuicerTests::CaseEvent primitive_event("cpp", "film-reference/primitives");
        require(captured.at("products").size() == 36, "complete reference cohort membership");
        for (const auto& row : captured.at("products"))
            for (const auto& query : row.at("queries"))
                require(query.at("native_log10") == query.at("rust_log10"), "reference log10 primitive differs; owner decision required");
        primitive_event.passed();
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
        JuicerProcess::ProcessRuntime owner;
        owner.create(resources);
        Assets assets(resources);
        const auto case_base = [](const Json& row) {
            return std::to_string(row.at("route").get<unsigned>()) + "/" + std::to_string(row.at("method").get<unsigned>()) + "/" + std::to_string(row.at("variant").get<unsigned>());
        };
        std::vector<std::string> raw_ids, fault_ids, numerical_ids;
        for (const auto& row : captured.at("products")) {
            if (!row.contains("input"))
                continue;
            const auto base = case_base(row);
            const auto method = row.at("method").get<unsigned>();
            for (const auto* api : {"legacy", "facade", "c11"}) {
                for (unsigned malformed = 0; malformed < 28; ++malformed)
                    raw_ids.emplace_back(base + "/" + api + "/" + std::to_string(malformed));
                const std::vector<unsigned> operations = method == 1 ? std::vector<unsigned>{1, 6} : std::vector<unsigned>{1, 2, 3, 4, 5};
                for (unsigned operation : operations)
                    for (unsigned exposure = 1; exposure <= 2; ++exposure)
                        for (unsigned fault = 1; fault <= 4; ++fault) {
                            if ((operation >= 2 && operation <= 4 && fault > 2) || (fault == 3 && operation != 5))
                                continue;
                            fault_ids.emplace_back(base + "/" + api + "/" + std::to_string(operation) + "/" + std::to_string(exposure) + "/" + std::to_string(fault));
                        }
            }
            if (method == 1)
                for (const auto* api : {"legacy", "facade", "c11"})
                    for (const auto& expected : captured.at("numerical_failures"))
                        numerical_ids.emplace_back(base + "/" + api + "/mutation-" + std::to_string(expected.at("mutation").get<unsigned>()));
        }
        JuicerTests::Cohort raw_inputs("film_reference.raw_input", raw_ids);
        JuicerTests::Cohort operation_faults("film_reference.operation_fault", fault_ids);
        JuicerTests::Cohort numerical_failures("film_reference.numerical_failure", numerical_ids);
        unsigned count = 0;
        for (const auto& row : captured.at("products")) {
            if (!row.contains("input"))
                continue;
            JuicerTests::CaseEvent case_event("cpp", "film-reference/" + std::to_string(row.at("route").get<unsigned>()) + "/" + std::to_string(row.at("method").get<unsigned>()) + "/" + std::to_string(row.at("variant").get<unsigned>()));
            ++count;
            auto developmentSpec = row.at("input");
            developmentSpec["route"] = row.at("route");
            const auto releaseDevelopment = [](FjFilmDevelopmentOwner* held) {
                (void)fj_legacy_film_development_release(held, nullptr);
            };
            std::unique_ptr<FjFilmDevelopmentOwner, decltype(releaseDevelopment)> development{assets.development(developmentSpec), releaseDevelopment};
            Inputs inputs(row);
            const std::array<std::pair<Prepare, const char*>, 3> preparers{{{fj_legacy_film_reference_prepare, "legacy"}, {fj_test_film_reference_prepare, "facade"}, {fj_test_film_reference_c_prepare, "c11"}}};
            for (const auto& [prepare, api] : preparers) {
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
                    auto row_event = raw_inputs.case_event(case_base(row) + "/" + api + "/" + std::to_string(malformed));
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
                    row_event.passed();
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
                            auto fault_event = operation_faults.case_event(case_base(row) + "/" + api + "/" + std::to_string(operation) + "/" + std::to_string(exposure) + "/" + std::to_string(fault));
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
                            fault_event.passed();
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
                    // This independent complete state stays held while raw
                    // numerical mutations exercise their source/development
                    // facade. No borrowed projection becomes a second owner.
                    auto heldState = JuicerTestSupport::prepare_product(owner.borrow().assets,
                                                                        controls(row.at("method"), Spektrafilm::ScanRoute::NegativeDirectScan, row.at("variant")));
                    FjPreparedStateView heldBefore{};
                    std::array<char, 2048> heldText{};
                    FjErrorBuffer heldError{heldText.data(), heldText.size(), 0};
                    status(heldState.view(heldBefore, heldError));
                    const auto heldReference = project(heldBefore.film_reference);
                    const auto heldRecipeHash = heldBefore.recipe_hash;
                    const auto heldScannerHash = heldBefore.scanner_recipe_hash;
                    const auto heldViewUnchanged = [&] {
                        FjPreparedStateView current{};
                        heldError.length = 0;
                        heldText[0] = 0;
                        status(heldState.view(current, heldError));
                        require(project(current.film_reference) == heldReference &&
                                    current.recipe_hash == heldRecipeHash &&
                                    current.scanner_recipe_hash == heldScannerHash,
                                "raw literal reference mutation leaves the actual held complete-state identities/words unchanged");
                    };
                    std::array<float, 243> basis{}, sensitivity{};
                    std::array<float, 81> light{};
                    basis.fill(1);
                    light.fill(1);
                    for (const auto& expected : captured.at("numerical_failures")) {
                        auto numerical_event = numerical_failures.case_event(case_base(row) + "/" + api + "/mutation-" + std::to_string(expected.at("mutation").get<unsigned>()));
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
                        heldViewUnchanged();
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
                        numerical_event.passed();
                    }
                }
            }
            case_event.passed();
            status(fj_legacy_assets_release_cached_payloads(assets.value, nullptr));
            FjFilmReference detached{};
            FjFilmReferenceFailure failure{};
            status(fj_legacy_film_reference_prepare(development.get(), &inputs.wire, &detached, &failure, nullptr));
            require(project(detached) == row.at("reference"), "live moved Development hold remains usable after source-cache detachment");
        }
        require(count == 2 && fj_test_film_development_live_owners() == 0, "both method witnesses release temporary holds");
    }

    void callers(const std::filesystem::path& resources) {
        JuicerProcess::ProcessRuntime owner;
        owner.create(resources);
        std::vector<std::string> caller_fault_ids, supersession_ids;
        for (unsigned method = 0; method < 3; ++method)
            for (unsigned route = 0; route < 4; ++route) {
                const std::vector<unsigned> operations = method == 1 ? std::vector<unsigned>{5, 6} : std::vector<unsigned>{2};
                for (unsigned operation : operations)
                    for (unsigned index : {1u, 2u})
                        for (unsigned fault : {1u, 2u, 3u, 4u}) {
                            if (method == 1 && fault == 3)
                                continue;
                            caller_fault_ids.emplace_back("method-" + std::to_string(method) + "/route-" + std::to_string(route) + "/operation-" + std::to_string(operation) + "/index-" + std::to_string(index) + "/fault-" + std::to_string(fault));
                        }
                supersession_ids.emplace_back("method-" + std::to_string(method) + "/route-" + std::to_string(route));
            }
        JuicerTests::Cohort caller_faults("film_reference.actual_caller_fault", caller_fault_ids);
        JuicerTests::Cohort caller_supersessions("film_reference.actual_caller_supersession", supersession_ids);
        for (unsigned method = 0; method < 3; ++method) {
            for (unsigned route = 0; route < 4; ++route) {
                JuicerTests::CaseEvent case_event("cpp", "film-reference/caller/" + std::to_string(method) + "/" + std::to_string(route));
                auto params = controls(method, static_cast<Spektrafilm::ScanRoute>(route), 0);
                auto state = std::make_unique<RenderInstance>();
                JuicerTestSupport::capture(*state, params);
                auto current = JuicerTestSupport::admit(owner.borrow().assets, *state);
                require(current.kind == FJ_ADMISSION_READY && current.state, "complete reference publication");
                const auto publication_view = [&](const JuicerTestSupport::PublicationAdmission& value) {
                    require(value.kind == FJ_ADMISSION_READY && value.state, "complete reference publication");
                    const auto view = JuicerTestSupport::published_view(value);
                    require(view.route == route, "reference publication preserves the selected route");
                    return view;
                };
                const auto original = current;
                const auto originalReference = project(publication_view(original).film_reference);
                const auto operations = method == 1 ? std::vector<unsigned>{5, 6} : std::vector<unsigned>{2};
                for (unsigned operation : operations) {
                    for (unsigned index : {1u, 2u}) {
                        for (unsigned fault : {1u, 2u, 3u, 4u}) {
                            if (method == 1 && fault == 3)
                                continue; // Existing fixed A7 arm contract rejects allocation injection.
                            auto row_event = caller_faults.case_event("method-" + std::to_string(method) + "/route-" + std::to_string(route) + "/operation-" + std::to_string(operation) + "/index-" + std::to_string(index) + "/fault-" + std::to_string(fault));
                            const bool tc = method != 1;
                            status(tc ? fj_test_tc_lut_arm_fault(operation, index + 1, fault) : fj_test_exposure_arm_fault(operation, index, fault));
                            const auto before = JuicerTestSupport::PublicationObserver::metrics(*state);
                            const auto counter = before.build;
                            const auto reused = JuicerTestSupport::admit(owner.borrow().assets, *state);
                            require(reused.kind == FJ_ADMISSION_READY && reused.state &&
                                        JuicerTestSupport::PublicationObserver::backing(*reused.state) == JuicerTestSupport::PublicationObserver::backing(*current.state) &&
                                        JuicerTestSupport::PublicationObserver::metrics(*state).build == counter,
                                    "unchanged publication skips reference and preserves the armed fault");
                            publication_view(reused);
                            params.cameraExposureCompensationEv += .125;
                            JuicerTestSupport::capture(*state, params);
                            JuicerTestSupport::PublicationAdmission failed;
                            bool allocationFailed = false;
                            try {
                                failed = JuicerTestSupport::admit(owner.borrow().assets, *state);
                            } catch (const std::bad_alloc&) {
                                allocationFailed = true;
                            }
                            require((tc ? fj_test_tc_lut_fault_consumed(operation, fault) : fj_test_exposure_fault_consumed(operation, fault)) == 1, "actual caller consumes the selected reference visit after focused midgray");
                            require(allocationFailed == (fault == 3), "allocation remains typed through admission");
                            require(JuicerTestSupport::PublicationObserver::metrics(*state).build == counter && fj_test_film_development_live_owners() == 0,
                                    "failed reference neither advances committed build nor leaks its moved Development hold");
                            if (fault == 2) {
                                require(failed.kind == FJ_ADMISSION_POISONED && !failed.state && failed.delivery.category == FJ_STATUS_INTERNAL_FAILURE,
                                        "reference panic terminally poisons the Rust publication");
                                const auto blocked = JuicerTestSupport::PublicationObserver::metrics(*state);
                                require(blocked.pendingKind == kBlockedPendingKind && blocked.build == counter, "reference panic blocks the owner without changing the committed build");
                                require(project(JuicerTestSupport::published_view(original).film_reference) == originalReference,
                                        "admitted immutable reference remains readable after terminal poison");
                                const auto future = JuicerTestSupport::admit(owner.borrow().assets, *state);
                                require(future.kind == FJ_ADMISSION_POISONED && !future.state && future.delivery.category == FJ_STATUS_INTERNAL_FAILURE,
                                        "future reference admission on the poisoned owner is rejected");
                                state = std::make_unique<RenderInstance>();
                                JuicerTestSupport::capture(*state, params);
                                current = JuicerTestSupport::admit(owner.borrow().assets, *state);
                                publication_view(current);
                            } else {
                                if (!allocationFailed)
                                    require(failed.kind == FJ_ADMISSION_PREPARATION_FAILED && !failed.state, "reference typed failure blocks new admission");
                                const auto retained = JuicerTestSupport::PublicationObserver::metrics(*state);
                                require(retained.publishedIdentity == before.publishedIdentity && retained.build == counter,
                                        "ordinary reference failure retains the private publication");
                                current = JuicerTestSupport::admit(owner.borrow().assets, *state);
                                publication_view(current);
                                require(current.state->build() == counter + 1 && fj_test_film_development_live_owners() == 0,
                                        "one-shot reference recovery commits once and releases the hold");
                            }
                            require(project(JuicerTestSupport::published_view(original).film_reference) == originalReference,
                                    "admitted immutable reference remains readable through failures and recovery");
                            row_event.passed();
                        }
                    }
                }
                auto supersession_event = caller_supersessions.case_event("method-" + std::to_string(method) + "/route-" + std::to_string(route));
                params.cameraExposureCompensationEv += .125;
                JuicerTestSupport::capture(*state, params);
                auto latest = params;
                latest.cameraExposureCompensationEv += .125;
                const auto counter = JuicerTestSupport::PublicationObserver::metrics(*state).build;
                const auto retained = current;
                const auto retainedReference = project(publication_view(retained).film_reference);
                PendingCaptureSupersession supersession{state.get(), &latest, false, {}};
                fj_test_instance_prepare_hooks(nullptr, supersede_after_prepare, &supersession);
                status(fj_test_exposure_arm_fault(5, 2, 4));
                current = JuicerTestSupport::admit(owner.borrow().assets, *state);
                fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                if (supersession.failure)
                    std::rethrow_exception(supersession.failure);
                require(supersession.fired && current.kind == FJ_ADMISSION_READY && current.state && current.state->input_identity() == hash_params(latest) &&
                            JuicerTestSupport::PublicationObserver::metrics(*state).inputIdentity == hash_params(latest) && current.state->build() == counter + 1,
                        "superseded reference failure recovers to the latest complete snapshot");
                (void)publication_view(current);
                require(project(publication_view(retained).film_reference) == retainedReference,
                        "superseded admitted reference remains readable through failed construction and latest recovery");
                require(fj_test_exposure_fault_consumed(5, 4) == 1 && fj_test_film_development_live_owners() == 0, "supersession releases its moved hold and consumes only the failed visit");
                supersession_event.passed();
                case_event.passed();
            }
        }
    }
} // namespace FilmReferenceTest

namespace JuicerAssets::RecipeTest {
    void film_reference(FjStatus& status, FjFilmReferenceFailure& failure, FjFilmReference&, FjErrorBuffer& error) {
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
