namespace PrintBalanceTest {
    void metadata(const FjPrintBalanceFailure& failure, std::uint32_t reason, std::uint32_t channel = UINT32_MAX);
    void zero(const FjPrintBalance& out);
    FjPrintBalanceInput balance_input(const FjPreparedStateView& view) {
        return {view.print.film_base_density,
                view.print.film_density_cmy,
                view.print.sensitivity_cmy,
                view.print.main_illuminant,
                {view.film_reference.baseline_density_cmy[0], view.film_reference.baseline_density_cmy[1], view.film_reference.baseline_density_cmy[2]},
                {view.film_reference.compensated_density_cmy[0], view.film_reference.compensated_density_cmy[1], view.film_reference.compensated_density_cmy[2]},
                view.film_reference.hash,
                view.print_normalization_mode};
    }
    struct CacheCall {
        JuicerCuda::Resources* resources = nullptr;
        JuicerCuda::Failure* failure = nullptr;
        bool consumeFault = false;
        std::uint64_t contentHash = 0;
    };
    bool prepare_cache_input(void* opaque,
                             const JuicerCudaTest::PreparedFrameFixture::Inputs& prepared,
                             std::string& diagnostic) {
        auto& call = *static_cast<CacheCall*>(opaque);
        const auto* input = std::get_if<JuicerCuda::PrintResourceInput>(&prepared.preparation.print);
        if (!input) {
            diagnostic = "prepared print state has no complete balance resource input";
            return false;
        }
        if (!JuicerCuda::prepare_print_resources(*call.resources, *input, nullptr, *call.failure)) {
            diagnostic = call.failure->diagnostic;
            return false;
        }
        PrintPreflashTest::cuda_ok(cudaDeviceSynchronize());
        call.contentHash = prepared.state.print_balance.hash;
        require(call.resources->printBalanceFactorMidgray == input->factorMidgray &&
                    call.resources->printBalanceFactorMidgrayComp == input->factorMidgrayComp &&
                    call.resources->printBalanceNormalizer == input->normalizer,
                "cache binds all completed balance scalars exactly");
        if (call.consumeFault) {
            const auto& view = prepared.state;
            const auto raw = balance_input(view);
            FjPrintBalance output{};
            FjPrintBalanceFailure failure{};
            status(fj_test_print_balance_prepare(&raw, &output, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
            metadata(failure, FJ_PRINT_BALANCE_REASON_PANIC);
            zero(output);
        }
        return true;
    }
    void prepare_cache(PreparedState& state,
                       const ParamSnapshot& controls,
                       JuicerCuda::Resources& resources,
                       JuicerCuda::Failure& failure,
                       std::uint64_t& contentHash,
                       bool consumeFault = true) {
        CacheCall call{&resources, &failure, consumeFault};
        std::string diagnostic;
        require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                    state, PrintPreflashTest::frame_facts(controls), &call, &prepare_cache_input, diagnostic),
                diagnostic.c_str());
        contentHash = call.contentHash;
    }

    Json project(const PrintBalanceRecipe& balance) {
        return {{"baseline_raw_rgb", words(balance.baselinePrintRawRgb.data(), 3)},
                {"compensated_raw_rgb", words(balance.compensatedPrintRawRgb.data(), 3)},
                {"scalars", words(std::array{balance.factorMidgray, balance.factorMidgrayComp, balance.normalizer}.data(), 3)},
                {"filtered_main_illuminant_hash", balance.filteredMainIlluminantHash},
                {"hash", balance.hash}};
    }
    Json project(const FjPrintBalance& balance) {
        return {{"baseline_raw_rgb", words(balance.baseline_raw_rgb, 3)},
                {"compensated_raw_rgb", words(balance.compensated_raw_rgb, 3)},
                {"scalars", words(std::array{balance.factor_midgray, balance.factor_midgray_comp, balance.normalizer}.data(), 3)},
                {"filtered_main_illuminant_hash", balance.filtered_main_illuminant_hash},
                {"hash", balance.hash}};
    }

    Json primitives() {
        Json rows = Json::array();
        const float maximum = std::numeric_limits<float>::max();
        const float minimum = std::numeric_limits<float>::denorm_min();
        const std::array densities{
            std::array{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
            std::array{0.031f, .1234567f, .99999994f, 2.4f, .31f, .777f, .111f},
            std::array{-38.0f, 1.0f, 1.0f, 1.0f, 0x1p-23f, 0x1p-24f, 0x1p-25f},
            std::array{100.0f, 1.0f, 1.0f, 1.0f, minimum, -minimum, minimum},
            std::array{-309.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
            std::array{324.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}};
        const std::array triplets{
            std::array{81.0f, 162.0f, 243.0f},
            std::array{8.1f, 16.2f, 24.3f},
            std::array{0.0f, -0.0f, 0.0f},
            std::array{minimum, 1.0e-10f, std::nextafter(1.0e-10f, 0.0f)},
            std::array{maximum, maximum, maximum},
            std::array{0x1.fffffep-1f, 0x1.000002p0f, 0.33333334f}};
        for (const auto& input : densities)
            for (const auto& raw : triplets)
                for (float denominator : {1.0f, maximum, minimum, 1.0e10f}) {
                    std::array<double, 12> native{};
                    double density = input[0];
                    for (std::size_t c = 0; c < 3; ++c) {
                        density += static_cast<double>(input[c + 1]) * static_cast<double>(input[c + 4]);
                        native[c] = density;
                    }
                    native[3] = std::pow(10.0, -density);
                    double meanLog = 0.0;
                    for (std::size_t c = 0; c < 3; ++c) {
                        native[4 + c] = std::log(std::max(1.0e-10, static_cast<double>(raw[c])));
                        meanLog += native[4 + c];
                        native[7 + c] = meanLog;
                    }
                    native[10] = -meanLog / 3.0;
                    native[11] = std::exp(native[10]);
                    const float factor = static_cast<float>(native[11]);
                    const std::array narrowed{factor, factor / denominator};
                    std::array<double, 12> rust{};
                    std::array<float, 2> rustNarrowed{};
                    fj_test_print_balance_primitive(input.data(), raw.data(), denominator, rust.data(), rustNarrowed.data());
                    rows.push_back({{"density", words(input.data(), input.size())}, {"raw", words(raw.data(), raw.size())}, {"denominator", std::bit_cast<std::uint32_t>(denominator)}, {"native", PrintPreflashTest::double_words(native.data(), native.size())}, {"rust", PrintPreflashTest::double_words(rust.data(), rust.size())}, {"native_narrowed", words(narrowed.data(), narrowed.size())}, {"rust_narrowed", words(rustNarrowed.data(), rustNarrowed.size())}});
                }
        return rows;
    }

    Json product_observations(const std::filesystem::path& resources) {
        Json products = Json::array();
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        for (unsigned route = 0; route < 4; ++route)
            for (int mode = 0; mode < 3; ++mode)
                for (unsigned variant = 0; variant < 10; ++variant) {
                    auto p = PrintPreflashTest::controls();
                    p.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                    p.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                    p.spectralUpsamplingMode = mode;
                    if (variant == 1)
                        p.printUiYmcCc = {2.1234567890123, 3.2345678901234, 4.3456789012345};
                    if (variant == 2)
                        p.outputCctfEncoding = 0;
                    if (variant == 3)
                        p.printPreflashExposure = .25;
                    if (variant == 4)
                        p.cameraExposureCompensationEv = 1.25;
                    if (variant == 5)
                        p.cameraExposureCompensationEv = -0.5;
                    if (variant == 6)
                        p.cameraAutoExposureEnabled = 1;
                    if (variant == 7)
                        p.filmGammaFactor = 1.13f;
                    if (variant == 8)
                        p.printExposureCompensation = 0;
                    if (variant == 9)
                        p.enlIll = 1;
                    std::array<char, 2048> diagnostic{};
                    FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                    std::optional<PreparedState> product;
                    const auto resultStatus = prepare_state(JuicerProcess::root().assets().borrow_rust_assets(), p, product, error);
                    require(resultStatus.category == FJ_STATUS_SUCCESS && product, diagnostic.data());
                    FjPreparedStateView view{};
                    status(product->view(view, error));
                    const auto& print = view.print;
                    auto balance = project(view.print_balance);
                    const auto factors = JuicerTestSupport::captured_print_factors(view);
                    balance["scalars"] = words(factors.data(), factors.size());
                    Json result{{"route", route}, {"mode", mode}, {"variant", variant}, {"snapshot_hash", hash_params(p)}, {"balance", std::move(balance)}, {"reference_hash", view.film_reference.hash}, {"print_hash", view.print_recipe_hash}, {"scanner_hash", view.scanner_recipe_hash}, {"final_hash", view.recipe_hash}, {"upload_hash", view.upload_core_hash}};
                    if (route % 2) {
                        result["film_base_density"] = words(print.film_base_density.data, print.film_base_density.count);
                        result["film_channel_density_cmy"] = words(print.film_density_cmy.data, print.film_density_cmy.count);
                        result["print_sensitivity_rgb"] = words(print.sensitivity_cmy.data, print.sensitivity_cmy.count);
                        result["filtered_main_illuminant"] = words(print.main_illuminant.data, print.main_illuminant.count);
                        result["baseline_density_cmy"] = words(view.film_reference.baseline_density_cmy, 3);
                        result["compensated_density_cmy"] = words(view.film_reference.compensated_density_cmy, 3);
                        result["normalization_mode"] = view.print_normalization_mode;
                    }
                    products.push_back(result);
                }
        return products;
    }

} // namespace PrintBalanceTest

namespace PrintBalanceTest {
    using Prepare = FjStatus (*)(const FjPrintBalanceInput*, FjPrintBalance*, FjPrintBalanceFailure*, FjErrorBuffer*);
    thread_local unsigned nextProjection = 0;
    struct Inputs {
        std::array<float, 81> base, light;
        std::array<float, 243> dye, sensitivity;
        std::array<float, 3> baseline, compensated;
        std::uint64_t referenceHash;
        std::uint32_t mode;
        explicit Inputs(const Json& row)
            : base(PrintPreflashTest::array<81>(row.at("film_base_density"))),
              light(PrintPreflashTest::array<81>(row.at("filtered_main_illuminant"))),
              dye(PrintPreflashTest::array<243>(row.at("film_channel_density_cmy"))),
              sensitivity(PrintPreflashTest::array<243>(row.at("print_sensitivity_rgb"))),
              baseline(PrintPreflashTest::array<3>(row.at("baseline_density_cmy"))),
              compensated(PrintPreflashTest::array<3>(row.at("compensated_density_cmy"))),
              referenceHash(row.at("reference_hash")), mode(row.at("normalization_mode")) {}
        FjPrintBalanceInput wire() const {
            FjPrintBalanceInput out{{base.data(), 81}, {dye.data(), 243}, {sensitivity.data(), 243}, {light.data(), 81}, {}, {}, referenceHash, mode};
            std::copy(baseline.begin(), baseline.end(), out.baseline_density_cmy);
            std::copy(compensated.begin(), compensated.end(), out.compensated_density_cmy);
            return out;
        }
        Profiles::FilmProfileSamples film() const {
            Profiles::FilmProfileSamples out{};
            out.baseDensity = base;
            out.channelDensity = channels(dye);
            return out;
        }
        Profiles::PrintProfileSamples print() const {
            Profiles::PrintProfileSamples out{};
            out.linearSensitivity = channels(sensitivity);
            return out;
        }
        static std::array<std::array<float, 3>, 81> channels(const std::array<float, 243>& flat) {
            std::array<std::array<float, 3>, 81> out{};
            for (std::size_t s = 0; s < 81; ++s)
                for (std::size_t c = 0; c < 3; ++c)
                    out[s][c] = flat[3 * s + c];
            return out;
        }
        SyntheticFilmReferenceRecipe reference() const {
            SyntheticFilmReferenceRecipe out{};
            out.baselineDensityCmy = baseline;
            out.compensatedDensityCmy = compensated;
            out.hash = referenceHash;
            return out;
        }
    };
    void seed(FjPrintBalance& out) {
        std::fill_n(out.baseline_raw_rgb, 3, -13.0f);
        std::fill_n(out.compensated_raw_rgb, 3, -13.0f);
        out.factor_midgray = out.factor_midgray_comp = out.normalizer = -13;
        out.filtered_main_illuminant_hash = out.hash = 99;
    }
    void zero(const FjPrintBalance& out) {
        const FjPrintBalance empty{};
        require(project(out) == project(empty), "all nine floats and both hashes clear independently to +0/0");
    }
    void metadata(const FjPrintBalanceFailure& failure, std::uint32_t reason, std::uint32_t channel) {
        require(failure.reason == reason && failure.channel == channel, "closed balance reason and first-channel metadata");
    }
    void values(const Json& captured, const std::filesystem::path& manifestPath) {
        const auto manifest = read(manifestPath);
        require(captured.at("cases").size() == 42, "exact adopted balance literal membership");
        for (const auto& row : captured.at("cases")) {
            Inputs source(row);
            const auto input = source.wire();
            const auto expected = manifest.at("outcomes").at(row.at("id").get<std::string>());
            for (Prepare prepare : {fj_legacy_print_balance_prepare, fj_test_print_balance_prepare}) {
                FjPrintBalance out{};
                seed(out);
                FjPrintBalanceFailure failure{99, 98};
                status(prepare(&input, &out, &failure, nullptr), expected.at("category"));
                metadata(failure, expected.at("reason"), expected.at("channel"));
                if (row.at("success")) {
                    if (project(out) != row.at("balance"))
                        throw std::runtime_error("exact native balance disagreement " + row.at("id").get<std::string>() + ": " + Json::diff(row.at("balance"), project(out)).dump());
                } else
                    zero(out);
            }
            PrintBalanceRecipe native{};
            native.hash = 99;
            std::string diagnostic;
            const bool success = JuicerAssets::prepare_print_balance(source.film(), source.print(), source.light, source.reference(), static_cast<Spektrafilm::PrintNormalizationMode>(source.mode), native, diagnostic);
            require(success == row.at("success").get<bool>() && diagnostic == row.at("diagnostic").get<std::string>(), "native balance bool and exact old diagnostic");
            require(project(native) == (success ? row.at("balance") : project(PrintBalanceRecipe{})), "native assignment only on complete success; failure carrier defaults");
        }
    }
    void transport(const Json& captured) {
        Inputs source(captured.at("cases")[0]);
        for (Prepare prepare : {fj_legacy_print_balance_prepare, fj_test_print_balance_prepare}) {
            auto input = source.wire();
            FjPrintBalance out{};
            FjPrintBalanceFailure failure{};
            auto call = [&](const FjPrintBalanceInput* in, FjPrintBalance* result, FjPrintBalanceFailure* rejected, FjErrorBuffer* error = nullptr) {
                seed(out);
                failure = {99, 98};
                return prepare(in, result, rejected, error);
            };
            status(call(nullptr, &out, &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_PRINT_BALANCE_REASON_RAW);
            zero(out);
            for (unsigned selected = 0; selected < 4; ++selected)
                for (unsigned fault = 0; fault < 4; ++fault) {
                    input = source.wire();
                    FjFloatSpan* spans[]{&input.film_base_density, &input.film_channel_density_cmy, &input.print_sensitivity_rgb, &input.filtered_main_illuminant};
                    auto& span = *spans[selected];
                    if (fault == 0)
                        --span.count;
                    if (fault == 1)
                        span.data = nullptr;
                    if (fault == 2)
                        span.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(span.data) + 1);
                    if (fault == 3)
                        span.data = reinterpret_cast<const float*>(UINTPTR_MAX & ~std::uintptr_t{3});
                    status(call(&input, &out, &failure), FJ_STATUS_UNSUPPORTED_INPUT);
                    metadata(failure, FJ_PRINT_BALANCE_REASON_RAW);
                    zero(out);
                }
            input = source.wire();
            input.normalization_mode = 256;
            status(call(&input, &out, &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_PRINT_BALANCE_REASON_RAW);
            zero(out);
            input = source.wire();
            for (unsigned mode = 4; mode <= 255; ++mode) {
                input.normalization_mode = mode;
                status(call(&input, &out, &failure), FJ_STATUS_PREPARATION_FAILURE);
                metadata(failure, FJ_PRINT_BALANCE_REASON_BALANCE);
                zero(out);
            }
            input = source.wire();
            alignas(FjPrintBalance) std::array<std::byte, sizeof(FjPrintBalance) + 8> resultStorage{};
            alignas(FjPrintBalanceFailure) std::array<std::byte, sizeof(FjPrintBalanceFailure) + 8> failureStorage{};
            alignas(FjPrintBalanceInput) std::array<std::byte, sizeof(FjPrintBalanceInput) + 8> inputStorage{};
            status(call(&input, reinterpret_cast<FjPrintBalance*>(resultStorage.data() + 1), &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_PRINT_BALANCE_REASON_RAW);
            require(resultStorage == decltype(resultStorage){}, "invalid output backing untouched");
            status(call(&input, &out, reinterpret_cast<FjPrintBalanceFailure*>(failureStorage.data() + 1)), FJ_STATUS_UNSUPPORTED_INPUT);
            zero(out);
            require(failureStorage == decltype(failureStorage){}, "invalid failure backing untouched");
            status(call(reinterpret_cast<FjPrintBalanceInput*>(inputStorage.data() + 1), &out, &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            zero(out);
            status(call(&input, nullptr, &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_PRINT_BALANCE_REASON_RAW);
            status(call(&input, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            zero(out);
            for (unsigned bad = 0; bad < 3; ++bad) {
                std::array<char, 8> backing{};
                backing.fill('z');
                FjErrorBuffer error{backing.data(), backing.size(), 99};
                if (bad == 0)
                    error.data = nullptr;
                if (bad == 1)
                    error.capacity = SIZE_MAX;
                if (bad == 2)
                    error.data = reinterpret_cast<char*>(UINTPTR_MAX - 3);
                status(call(&input, &out, &failure, &error), FJ_STATUS_UNSUPPORTED_INPUT);
                metadata(failure, FJ_PRINT_BALANCE_REASON_RAW);
                zero(out);
                require(error.length == 0 && std::all_of(backing.begin(), backing.end(), [](char c) {
                            return c == 'z';
                        }),
                        "malformed diagnostics never touch invalid backing");
            }
            alignas(FjErrorBuffer) std::array<std::byte, sizeof(FjErrorBuffer) + 8> errorStorage{};
            status(call(&input, &out, &failure, reinterpret_cast<FjErrorBuffer*>(errorStorage.data() + 1)), FJ_STATUS_UNSUPPORTED_INPUT);
            zero(out);
            require(errorStorage == decltype(errorStorage){}, "unaligned diagnostic record untouched");
            for (std::size_t capacity : {0u, 1u, 5u, 128u}) {
                std::array<char, 128> text{};
                text.fill('z');
                FjErrorBuffer error{text.data(), capacity, 99};
                status(call(nullptr, &out, &failure, &error), FJ_STATUS_UNSUPPORTED_INPUT);
                metadata(failure, FJ_PRINT_BALANCE_REASON_RAW);
                const std::string message = "print balance reason=1 channel=4294967295";
                const auto length = capacity ? std::min(capacity - 1, message.size()) : 0;
                require(error.length == length && (capacity == 0 ? text[0] == 'z' : text[length] == '\0' && std::string(text.data()) == message.substr(0, length)), "bounded diagnostics do not affect typed failure");
                zero(out);
            }
            status(fj_test_film_recipe_fault(12, 1));
            status(call(nullptr, &out, &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            status(call(&input, &out, &failure));
            status(fj_test_film_recipe_fault(12, 1));
            status(call(&input, &out, &failure), FJ_STATUS_INTERNAL_FAILURE);
            metadata(failure, FJ_PRINT_BALANCE_REASON_PANIC);
            zero(out);
            status(call(&input, &out, &failure));
            status(fj_test_film_recipe_fault(12, 1));
            status(fj_test_film_recipe_fault(12, 0));
            status(call(&input, &out, &failure));
            status(fj_test_film_recipe_fault(12, 2), FJ_STATUS_UNSUPPORTED_INPUT);
            status(fj_test_film_recipe_fault(13, 1), FJ_STATUS_UNSUPPORTED_INPUT);
            status(fj_test_film_recipe_fault(12, 1));
            FjStatus other{};
            FjPrintBalance otherOut{};
            FjPrintBalanceFailure otherFailure{};
            std::thread thread([&] {
                other = prepare(&input, &otherOut, &otherFailure, nullptr);
            });
            thread.join();
            status(other);
            metadata(otherFailure, FJ_PRINT_BALANCE_REASON_NONE);
            status(call(&input, &out, &failure), FJ_STATUS_INTERNAL_FAILURE);
            metadata(failure, FJ_PRINT_BALANCE_REASON_PANIC);
            zero(out);
            status(call(&input, &out, &failure));
            std::array<float, 243> aliased{};
            aliased.fill(1.0f);
            auto aliasInput = input;
            aliasInput.film_base_density = aliasInput.filtered_main_illuminant = {aliased.data(), 81};
            aliasInput.film_channel_density_cmy = aliasInput.print_sensitivity_rgb = {aliased.data(), 243};
            status(call(&aliasInput, &out, &failure));
            FjPrintBalance retained{};
            {
                Inputs ephemeral(captured.at("cases")[0]);
                const auto wire = ephemeral.wire();
                status(prepare(&wire, &retained, &failure, nullptr));
                ephemeral.base.fill(100);
                ephemeral.light.fill(0);
            }
            require(project(retained) == captured.at("cases")[0].at("balance"), "fixed raw result owns words after all inputs expire");
        }
        Inputs input(captured.at("cases")[0]);
        for (unsigned fault = 1; fault <= 24; ++fault) {
            nextProjection = fault;
            PrintBalanceRecipe out{};
            out.hash = 99;
            std::string diagnostic;
            bool threw = false;
            try {
                JuicerAssets::prepare_print_balance(input.film(), input.print(), input.light, input.reference(), Spektrafilm::PrintNormalizationMode::None, out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                status(error.failure.status, fault == 15 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                threw = true;
            }
            require(threw && nextProjection == 0 && project(out) == project(PrintBalanceRecipe{}), "native integrity rejects malformed envelope/result without publication");
        }
    }
    void caller(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        auto p = PrintPreflashTest::controls();
        for (int normalize : {0, 1})
            for (int compensate : {0, 1}) {
                p.normalizePrintExposure = normalize;
                p.printExposureCompensation = compensate;
                const auto product = PrintPreflashTest::build(p);
                std::array<char, 2048> text{};
                FjErrorBuffer error{text.data(), text.size(), 0};
                FjPreparedStateView view{};
                status(product.view(view, error));
                require(view.print_normalization_mode == static_cast<unsigned>(2 * normalize + compensate), "all four literal snapshot modes retain native boolean policy");
            }
        p = PrintPreflashTest::controls();
        InstanceState state;
        const auto pending = [&](const ParamSnapshot& params) {
            std::lock_guard<std::mutex> lock(state.pending.m);
            state.pending.value = PendingParamsState::Valid{params, hash_params(params)};
        };
        pending(p);
        const auto original = admit_pending_render_state(state);
        require(bool(original.printState), "balance actual publication baseline");
        const auto originalView = JuicerTestSupport::published_view(original);
        const auto originalBalance = project(originalView.print_balance);
        const auto counter = state.buildCounterNext.load();
        status(fj_test_film_recipe_fault(12, 1));
        const auto reused = admit_pending_render_state(state);
        require(reused.printState == original.printState && state.buildCounterNext.load() == counter, "identical current snapshot reuses complete publication without balance construction");
        p.cameraExposureCompensationEv = .25;
        pending(p);
        const auto published = admit_pending_render_state(state);
        require(published.status == PendingRenderAdmissionStatus::AdmittedPrint && published.printState &&
                    published.printState != original.printState &&
                    JuicerAtomic::load_shared_ptr(&state.activePrintState) == published.printState &&
                    state.lastHash.load() == hash_params(p) && state.buildCounterNext.load() == counter + 1,
                "complete Rust balance construction publishes without visiting the raw facade fault");
        require(project(JuicerTestSupport::published_view(original).print_balance) == originalBalance,
                "already admitted immutable balance remains readable through complete replacement");
        FjPreparedStateView publishedView = JuicerTestSupport::published_view(published);
        FjPrintBalance publishedOutput{};
        FjPrintBalanceFailure publishedFailure{};
        const FjPrintBalanceInput publishedRaw = balance_input(publishedView);
        status(fj_test_print_balance_prepare(&publishedRaw, &publishedOutput, &publishedFailure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
        metadata(publishedFailure, FJ_PRINT_BALANCE_REASON_PANIC);
        zero(publishedOutput);
        status(fj_test_print_balance_prepare(&publishedRaw, &publishedOutput, &publishedFailure, nullptr));
        require(project(publishedOutput) == project(publishedView.print_balance),
                "one-shot raw balance facade fault and retry match the actual published held balance");
        auto activeControls = PrintPreflashTest::controls();
        auto active = PrintPreflashTest::build(activeControls);
        status(fj_test_film_recipe_fault(12, 1));
        p.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        auto direct = PrintPreflashTest::build(p);
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjPreparedStateView directView{};
        status(direct.view(directView, error));
        require(directView.print.hash == 0 && directView.print_balance.hash == 0, "direct route omits balance and its input");
        struct BoundFacts {
            float normalizer = 0;
            std::uint64_t hash = 0;
        } bound;
        const auto inspect_input = [](void* opaque, const JuicerCudaTest::PreparedFrameFixture::Inputs& prepared, std::string& diagnostic) {
            auto& facts = *static_cast<BoundFacts*>(opaque);
            const auto* input = std::get_if<JuicerCuda::PrintResourceInput>(&prepared.preparation.print);
            if (!input) {
                diagnostic = "prepared print state has no complete balance resource input";
                return false;
            }
            facts.normalizer = input->normalizer;
            facts.hash = input->descriptors.balance.hash;
            return true;
        };
        std::string diagnostic;
        status(fj_test_film_recipe_fault(12, 1));
        require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                    active, PrintPreflashTest::frame_facts(activeControls), &bound, inspect_input, diagnostic),
                diagnostic.c_str());
        FjPreparedStateView activeView{};
        status(active.view(activeView, error));
        require(bound.normalizer == activeView.print_balance.normalizer && bound.hash == activeView.print.balance_hash,
                "resource assembly binds completed balance and its owning descriptor");
        const auto raw = balance_input(activeView);
        FjPrintBalance output{};
        FjPrintBalanceFailure failure{};
        status(fj_test_print_balance_prepare(&raw, &output, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
        metadata(failure, FJ_PRINT_BALANCE_REASON_PANIC);
        zero(output);
        status(fj_test_print_balance_prepare(&raw, &output, &failure, nullptr));
        require(project(output) == project(activeView.print_balance),
                "unconsumed raw balance fault is one-shot and retry matches the held complete owner");

        // SF_TEMP_BRIDGE_NATIVE_BALANCE_FAULT_CONSUMER: retain the original
        // native constructor's typed exception/retry comparison through I7.
        // Actual complete-state construction and resource assembly above do
        // not visit this legacy facade.
        status(fj_test_film_recipe_fault(12, 1));
        bool threw = false;
        FocusedRenderStateBuildProduct comparison;
        try {
            require(build_print_render_state_product(activeControls, comparison, diagnostic), diagnostic.c_str());
        } catch (const JuicerCuda::ExecutionFailure&) {
            threw = true;
        }
        require(threw, "SF_TEMP_BRIDGE native constructor consumes its legacy balance fault");
        require(build_print_render_state_product(activeControls, comparison, diagnostic), diagnostic.c_str());
    }
    void products(const std::filesystem::path& resources, const Json& captured) {
        const auto actual = product_observations(resources);
        const auto& expected = captured.at("products");
        if (actual != expected)
            throw std::runtime_error("print balance product first difference " + Json::diff(expected, actual).front().dump());
        caller(resources);
    }
} // namespace PrintBalanceTest

namespace JuicerAssets::RecipeTest {
    void print_balance(FjStatus& status, FjPrintBalanceFailure& failure, FjPrintBalance& balance, FjErrorBuffer& error) {
        const auto fault = std::exchange(PrintBalanceTest::nextProjection, 0u);
        switch (fault) {
            case 1:
                status.category = 99;
                break;
            case 2:
                status.api = 99;
                break;
            case 3:
                status.native_code = 1;
                break;
            case 4:
                failure.reason = 99;
                break;
            case 5:
                failure.channel = 0;
                break;
            case 6:
                failure.reason = FJ_PRINT_BALANCE_REASON_BASELINE_RAW;
                failure.channel = 3;
                break;
            case 7:
                status.category = FJ_STATUS_PREPARATION_FAILURE;
                failure.reason = FJ_PRINT_BALANCE_REASON_COMPENSATED_RAW;
                failure.channel = UINT32_MAX;
                break;
            case 8:
                status.category = FJ_STATUS_PREPARATION_FAILURE;
                break;
            case 9:
                failure.reason = FJ_PRINT_BALANCE_REASON_PANIC;
                break;
            case 10:
                failure.reason = FJ_PRINT_BALANCE_REASON_RAW;
                break;
            case 11:
                error.length = error.capacity;
                break;
            case 12:
                error.data = nullptr;
                break;
            case 13:
                error.capacity = 0;
                break;
            case 14:
                error.data[error.length] = 'x';
                break;
            case 15:
                status.category = FJ_STATUS_UNSUPPORTED_INPUT;
                failure.reason = FJ_PRINT_BALANCE_REASON_RAW;
                break;
            case 16:
                status.category = FJ_STATUS_INTERNAL_FAILURE;
                failure.reason = FJ_PRINT_BALANCE_REASON_PANIC;
                break;
            case 17:
                balance.baseline_raw_rgb[1] = -1;
                break;
            case 18:
                balance.compensated_raw_rgb[2] = std::numeric_limits<float>::quiet_NaN();
                break;
            case 19:
                balance.factor_midgray = 0;
                break;
            case 20:
                balance.factor_midgray_comp = std::numeric_limits<float>::infinity();
                break;
            case 21:
                balance.normalizer = 0;
                break;
            case 22:
                balance.hash = 0;
                break;
            case 23:
                balance.filtered_main_illuminant_hash = 0;
                break;
            case 24:
                balance.factor_midgray = -1;
                break;
            default:
                break;
        }
    }
} // namespace JuicerAssets::RecipeTest

namespace PrintBalanceTest {
    void cache(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        PrintPreflashTest::cuda_ok(cudaSetDevice(0));
        PrintPreflashTest::cuda_ok(cudaFree(nullptr));
        void* context = nullptr;
        std::string diagnostic;
        require(JuicerCuda::query_current_cuda_context(context, diagnostic) && context, "balance cache exact native context");
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, context};
        constexpr std::uint64_t kBudgetBytes = std::uint64_t{1024} * 1024 * 1024;
        auto ledger = JuicerCuda::DeviceAllocationLedger::create({0, kBudgetBytes}, diagnostic);
        require(bool(ledger) && ledger->bind_or_validate_cap(kBudgetBytes, diagnostic), diagnostic.c_str());
        auto controls = PrintPreflashTest::controls();
        auto original = PrintPreflashTest::build(controls);
        JuicerCuda::Failure failure;
        {
            JuicerCuda::Resources cached(key, 1, ledger);
            const auto prepare = [&](PreparedState& product, const ParamSnapshot& snapshot, bool consume = true) {
                std::uint64_t contentHash = 0;
                status(fj_test_film_recipe_fault(12, 1));
                prepare_cache(product, snapshot, cached, failure, contentHash, consume);
                return contentHash;
            };
            const auto originalContentHash = prepare(original, controls);
            const auto balanceHash = cached.printBalanceDescriptorHash;
            const auto mainHash = cached.printMainIlluminantDescriptorHash;
            auto* const main = cached.printIllumFiltered;
            const auto charged = ledger->snapshot().chargedBytes;
            prepare(original, controls);
            require(cached.printBalanceDescriptorHash == balanceHash && cached.printIllumFiltered == main && ledger->snapshot().chargedBytes == charged, "identical state preserves balance key, residency and allocation accounting");
            for (unsigned variant = 0; variant < 7; ++variant) {
                auto p = controls;
                if (variant == 0)
                    p.printPreflashExposure = .25;
                if (variant == 1)
                    p.outputCctfEncoding = 0;
                if (variant == 2)
                    p.cameraExposureCompensationEv = 1.25;
                if (variant == 3)
                    p.filmGammaFactor = 1.13f;
                if (variant == 4)
                    p.printUiYmcCc = {2.0, 3.0, 4.0};
                if (variant == 5)
                    p.filmProfileKey = "kodak_ektar_100";
                if (variant == 6)
                    p.printProfileKey = "fujifilm_crystal_archive_typeii";
                auto changed = PrintPreflashTest::build(p);
                prepare(changed, p);
                require((cached.printBalanceDescriptorHash == balanceHash) == (variant < 2), "balance descriptor preserves unrelated/preflash reuse and invalidates reference/gamma/main inputs");
                if (variant < 5)
                    require((cached.printMainIlluminantDescriptorHash == mainHash) == (variant < 4), "main descriptor identity remains distinct from balance content identity");
                if (variant < 4)
                    require(cached.printIllumFiltered == main, "balance-only changes preserve main light residency");
                prepare(original, controls);
                require(cached.printBalanceDescriptorHash == balanceHash, "restoring a recipe restores its exact balance descriptor");
            }
            std::array<std::uint64_t, 4> contentHashes{}, descriptorHashes{};
            for (unsigned mode = 0; mode < 4; ++mode) {
                auto p = controls;
                p.normalizePrintExposure = (mode & 2u) != 0;
                p.printExposureCompensation = (mode & 1u) != 0;
                auto changed = PrintPreflashTest::build(p);
                contentHashes[mode] = prepare(changed, p);
                descriptorHashes[mode] = cached.printBalanceDescriptorHash;
            }
            require(contentHashes[0] == contentHashes[1] && contentHashes[2] == contentHashes[3], "equal reference factors preserve coincident balance content identities without a mode lane");
            require(descriptorHashes[0] != descriptorHashes[1] && descriptorHashes[2] != descriptorHashes[3], "descriptor mode dependencies remain distinct even when content identities coincide");
            prepare(original, controls);
            for (bool film : {false, true}) {
                auto changedControls = controls;
                changedControls.filmProfileKey = film ? "kodak_ektar_100" : controls.filmProfileKey;
                changedControls.printProfileKey = film ? controls.printProfileKey : "fujifilm_crystal_archive_typeii";
                auto changed = PrintPreflashTest::build(changedControls);
                prepare(changed, changedControls);
                require(cached.printBalanceDescriptorHash != balanceHash, "both source tokens contribute to balance descriptor identity");
                prepare(original, controls);
            }
            require(originalContentHash != 0, "complete held state supplies balance content identity");
            require(JuicerCuda::drain_for_context_retire(cached, failure), failure.diagnostic.c_str());
        }
        require(ledger->snapshot().chargedBytes == 0, "balance cache fixture releases device ownership/accounting");
        JuicerCuda::purge_pinned_upload_staging_for_context(0, context, JuicerCuda::PinnedUploadPurgeDisposition::NormalRetire);
    }
} // namespace PrintBalanceTest
