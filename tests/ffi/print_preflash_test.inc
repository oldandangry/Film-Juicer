namespace PrintPreflashTest {
    template <std::size_t N>
    std::array<float, N> array(const Json& value) {
        require(value.size() == N, "preflash input extent");
        std::array<float, N> out{};
        for (std::size_t i = 0; i < N; ++i)
            out[i] = std::bit_cast<float>(value[i].get<std::uint32_t>());
        return out;
    }
    Json double_words(const double* values, std::size_t count) {
        Json out = Json::array();
        for (std::size_t i = 0; i < count; ++i)
            out.push_back(std::bit_cast<std::uint64_t>(values[i]));
        return out;
    }
    void save(const std::filesystem::path& path, const Json& value) {
        require(!std::filesystem::exists(path), "preserve prior preflash capture");
        std::ofstream stream(path);
        stream << value.dump(2) << '\n';
        require(bool(stream), "preflash capture write");
    }
    Json primitives(const std::filesystem::path& resources) {
        Json rows = Json::array();
        std::vector<float> densities{0.0f, -0.0f, 0.001f, -0.001f, 0.1f, 0.31f, 0.99999994f, 1.0f, 2.4f, 38.0f, 100.0f, 308.0f, 309.0f, 323.0f, 324.0f, -38.0f, -308.0f, -309.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        auto& assets = JuicerProcess::root().assets();
        JuicerProcess::root().ensure_bootstrap();
        for (const char* key : {"kodak_portra_400", "fujifilm_provia_100f"}) {
            const auto film = assets.selected_film_profile_for_key(key);
            require(bool(film), "primitive actual film source");
            for (std::size_t sample : {std::size_t{0}, std::size_t{40}, std::size_t{80}})
                densities.push_back(film->data.baseDensity[sample]);
        }
        for (float density : densities)
            for (float illuminant : {0.0f, -0.0f, 1.0f, 0.1234567f, -1.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
                for (float sensitivity : {1.0f, -1.0f, 0.33333334f, std::numeric_limits<float>::max()}) {
                    const double d = density, i = illuminant, s = sensitivity;
                    const double power = std::pow(10.0, -d);
                    const double transmitted = power * i;
                    const double light = std::isnan(transmitted) ? 0.0 : transmitted;
                    const double product = light * s;
                    for (double accumulator : {0.0, -product, 0x1.000001p0, 0x1.ffffffp127}) {
                        const double sum = accumulator + product;
                        const double native[]{power, transmitted, light, product, sum};
                        double rust[5]{};
                        float narrowed{};
                        fj_test_print_preflash_primitive(d, i, s, accumulator, rust, &narrowed);
                        const bool nativeSuccess = std::isfinite(sum) && sum >= 0.0;
                        const bool rustSuccess = std::isfinite(rust[4]) && rust[4] >= 0.0;
                        rows.push_back({{"input", double_words(std::array{d, i, s, accumulator}.data(), 4)},
                                        {"native", double_words(native, 5)},
                                        {"rust", double_words(rust, 5)},
                                        {"native_success", nativeSuccess},
                                        {"rust_success", rustSuccess},
                                        {"native_narrowed", std::bit_cast<std::uint32_t>(static_cast<float>(sum))},
                                        {"rust_narrowed", std::bit_cast<std::uint32_t>(narrowed)}});
                    }
                }
        return rows;
    }
    Json product_observations(const std::filesystem::path& resources) {
        Json products = Json::array();
        {
            JuicerCuda::Owner owner;
            owner.create(resources.string().c_str());
            auto& assets = JuicerProcess::root().assets();
            JuicerProcess::root().ensure_bootstrap();
            for (unsigned route = 0; route < 4; ++route)
                for (int mode = 0; mode < 3; ++mode)
                    for (unsigned variant = 0; variant < 6; ++variant) {
                        ParamSnapshot p;
                        p.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                        p.printProfileKey = "kodak_portra_endura";
                        p.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                        p.spectralUpsamplingMode = mode;
                        p.cameraAutoExposureEnabled = 0;
                        p.inputCompressionEnabled = 0;
                        p.outputGamutCompressionEnabled = 0;
                        p.printPreflashExposure = variant ? (variant == 5 ? 0.25 : 0.125) : 0.0;
                        if (variant == 2) {
                            p.preflashMFilterCc = 5;
                            p.preflashYFilterCc = -3;
                        }
                        if (variant == 3)
                            p.printUiYmcCc = {2.1234567890123, 3.2345678901234, 4.3456789012345};
                        if (variant == 4)
                            p.outputCctfEncoding = 0;
                        FocusedRenderStateBuildProduct product;
                        std::string diagnostic;
                        const bool success = route % 2 ? build_print_render_state_product(p, product, diagnostic) : build_direct_render_state_product(p, product, diagnostic);
                        require(success, diagnostic.c_str());
                        const auto& recipe = product.recipe;
                        Json row{{"route", route}, {"mode", mode}, {"variant", variant}, {"snapshot_hash", hash_params(p)}, {"final_hash", recipe.hash}, {"print_hash", recipe.print.hash}, {"balance_hash", recipe.print.balance.hash}};
                        if (route % 2) {
                            JuicerCuda::PrintResourcePreparation request{&recipe, &assets, &*product.payload.printMainIlluminant};
                            JuicerCuda::PrintResourceInput input;
                            std::array<float, 81> light{};
                            require(JuicerCuda::build_print_resource_input(request, input, light, diagnostic), diagnostic.c_str());
                            row["raw"] = words(input.preflashRawCmy.data(), 3);
                            row["light"] = words(light.data(), 81);
                            row["film_base_density"] = words(recipe.profileRoute.filmProfile->data.baseDensity.data(), 81);
                            Json sensitivity = Json::array();
                            for (const auto& sample : recipe.profileRoute.printProfile->data.linearSensitivity)
                                for (float channel : sample)
                                    sensitivity.push_back(std::bit_cast<std::uint32_t>(channel));
                            row["print_sensitivity_cmy"] = sensitivity;
                            row["descriptors"] = {input.descriptors.hash, input.descriptors.mainIlluminant.hash, input.descriptors.preflashIlluminant.hash, input.descriptors.preflashRaw.hash, input.descriptors.balance.hash};
                        }
                        products.push_back(row);
                    }
        }
        return products;
    }
} // namespace PrintPreflashTest

namespace PrintPreflashTest {
    using Prepare = FjStatus (*)(const FjPrintPreflashInput*, FjPrintPreflashRaw*, FjPrintPreflashFailure*, FjErrorBuffer*);
    void caller(const std::filesystem::path& resources);
    thread_local unsigned nextProjection = 0;
    struct Inputs {
        std::array<float, 81> base{};
        std::array<float, 81> light{};
        std::array<float, 243> sensitivity{};
        explicit Inputs(const Json& row)
            : base(array<81>(row.at("film_base_density"))),
              light(array<81>(row.at("filtered_illuminant"))),
              sensitivity(array<243>(row.at("print_sensitivity_cmy"))) {}
        FjPrintPreflashInput wire() const {
            return {{base.data(), base.size()}, {light.data(), light.size()}, {sensitivity.data(), sensitivity.size()}};
        }
        std::array<std::array<float, 3>, 81> channels() const {
            std::array<std::array<float, 3>, 81> out{};
            for (std::size_t i = 0; i < 81; ++i)
                for (std::size_t c = 0; c < 3; ++c)
                    out[i][c] = sensitivity[i * 3 + c];
            return out;
        }
    };
    void metadata(const FjPrintPreflashFailure& failure, std::uint32_t reason, std::uint32_t channel = UINT32_MAX) {
        require(failure.reason == reason && failure.channel == channel, "closed preflash reason/first-channel metadata");
    }
    void zero(const FjPrintPreflashRaw& out) {
        require(words(out.cmy, 3) == Json::array({0, 0, 0}), "every independently writable preflash result clears to +0");
    }
    void values(const Json& captured, const std::filesystem::path& manifestPath) {
        const auto manifest = read(manifestPath);
        require(captured.at("cases").size() == 19, "adopted preflash literal membership");
        for (const auto& row : captured.at("cases")) {
            Inputs source(row);
            const auto input = source.wire();
            for (Prepare prepare : {fj_legacy_print_preflash_prepare, fj_test_print_preflash_prepare}) {
                FjPrintPreflashRaw out{{-13, -13, -13}};
                FjPrintPreflashFailure failure{99, 98};
                std::array<char, 128> text{};
                FjErrorBuffer error{text.data(), text.size(), 99};
                const bool success = row.at("success");
                status(prepare(&input, &out, &failure, &error), success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (success) {
                    metadata(failure, FJ_PRINT_PREFLASH_REASON_NONE);
                    require(words(out.cmy, 3) == row.at("raw") && error.length == 0 && text[0] == '\0', "exact native preflash words, including narrowing infinity, and empty success diagnostic");
                } else {
                    const auto channel = manifest.at("failure_channels").at(row.at("id").get<std::string>()).get<std::uint32_t>();
                    metadata(failure, FJ_PRINT_PREFLASH_REASON_CHANNEL, channel);
                    zero(out);
                    require(std::string(text.data()) == "print preflash reason=2 channel=" + std::to_string(channel), "exact bounded channel diagnostic");
                }
            }
        }
    }
    void transport(const Json& captured) {
        for (Prepare prepare : {fj_legacy_print_preflash_prepare, fj_test_print_preflash_prepare}) {
            Inputs source(captured.at("cases")[0]);
            auto input = source.wire();
            FjPrintPreflashRaw out{};
            FjPrintPreflashFailure failure{};
            const auto call = [&](const FjPrintPreflashInput* p, FjPrintPreflashRaw* result, FjPrintPreflashFailure* detail, FjErrorBuffer* error = nullptr) {
                std::fill_n(out.cmy, 3, -13.0f);
                failure = {99, 98};
                return prepare(p, result, detail, error);
            };
            const auto raw = [&](const FjPrintPreflashInput* p, FjErrorBuffer* error = nullptr) {
                status(call(p, &out, &failure, error), FJ_STATUS_UNSUPPORTED_INPUT);
                metadata(failure, FJ_PRINT_PREFLASH_REASON_RAW);
                zero(out);
            };
            const auto end = UINTPTR_MAX & ~(static_cast<std::uintptr_t>(alignof(FjPrintPreflashInput)) - 1);
            alignas(FjPrintPreflashInput) std::array<std::byte, sizeof(FjPrintPreflashInput) + 1> storage{};
            raw(nullptr);
            raw(reinterpret_cast<const FjPrintPreflashInput*>(storage.data() + 1));
            raw(reinterpret_cast<const FjPrintPreflashInput*>(end));
            for (auto member : {&FjPrintPreflashInput::film_base_density, &FjPrintPreflashInput::filtered_illuminant, &FjPrintPreflashInput::print_sensitivity_cmy}) {
                for (std::size_t count : {std::size_t{0}, std::size_t{80}, std::size_t{82}, std::size_t{242}, std::size_t{244}, SIZE_MAX}) {
                    auto invalid = input;
                    (invalid.*member).count = count;
                    raw(&invalid);
                }
                for (const float* pointer : {static_cast<const float*>(nullptr), reinterpret_cast<const float*>(storage.data() + 1), reinterpret_cast<const float*>(end)}) {
                    auto invalid = input;
                    (invalid.*member).data = pointer;
                    raw(&invalid);
                }
            }
            status(call(&input, nullptr, &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_PRINT_PREFLASH_REASON_RAW);
            status(call(&input, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            zero(out);
            for (std::uintptr_t address : {end, reinterpret_cast<std::uintptr_t>(storage.data() + 1)}) {
                status(call(&input, reinterpret_cast<FjPrintPreflashRaw*>(address), &failure), FJ_STATUS_UNSUPPORTED_INPUT);
                metadata(failure, FJ_PRINT_PREFLASH_REASON_RAW);
                status(call(&input, &out, reinterpret_cast<FjPrintPreflashFailure*>(address)), FJ_STATUS_UNSUPPORTED_INPUT);
                zero(out);
            }
            FjErrorBuffer invalid{nullptr, 1, 99};
            raw(&input, &invalid);
            require(invalid.length == 0, "invalid backing clears writable error length");
            invalid = {reinterpret_cast<char*>(UINTPTR_MAX), 2, 99};
            raw(&input, &invalid);
            raw(&input, reinterpret_cast<FjErrorBuffer*>(storage.data() + 1));
            raw(&input, reinterpret_cast<FjErrorBuffer*>(end));
            source = Inputs(captured.at("cases")[7]);
            input = source.wire();
            std::array<char, 128> full{};
            FjErrorBuffer error{full.data(), full.size(), 99};
            status(call(&input, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
            metadata(failure, FJ_PRINT_PREFLASH_REASON_CHANNEL, 1);
            zero(out);
            const std::string message(full.data());
            require(error.length == message.size(), "preflash length excludes NUL");
            for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{8}}) {
                std::array<char, 8> text{};
                text.fill('x');
                error = {capacity ? text.data() : reinterpret_cast<char*>(UINTPTR_MAX), capacity, 99};
                status(call(&input, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
                metadata(failure, FJ_PRINT_PREFLASH_REASON_CHANNEL, 1);
                zero(out);
                require(error.length == (capacity ? capacity - 1 : 0), "truncation preserves metadata and bounded length");
                if (capacity)
                    require(text[capacity - 1] == '\0' && std::string(text.data()) == message.substr(0, capacity - 1), "diagnostic prefix/NUL");
                else
                    require(text[0] == 'x', "zero-capacity backing untouched");
            }
            source = Inputs(captured.at("cases")[0]);
            input = source.wire();
            for (bool rejectFirst : {false, true}) {
                status(fj_test_film_recipe_fault(11, 1));
                if (rejectFirst)
                    raw(nullptr);
                else {
                    status(call(&input, &out, &failure), FJ_STATUS_INTERNAL_FAILURE);
                    metadata(failure, FJ_PRINT_PREFLASH_REASON_PANIC);
                    zero(out);
                }
                status(call(&input, &out, &failure));
                metadata(failure, FJ_PRINT_PREFLASH_REASON_NONE);
            }
            status(fj_test_film_recipe_fault(11, 1));
            status(fj_test_film_recipe_fault(11, 0));
            status(call(&input, &out, &failure));
            status(fj_test_film_recipe_fault(11, 2), FJ_STATUS_UNSUPPORTED_INPUT);
            status(fj_test_film_recipe_fault(12, 1), FJ_STATUS_UNSUPPORTED_INPUT);
            status(fj_test_film_recipe_fault(11, 1));
            FjStatus other{};
            FjPrintPreflashRaw otherOut{};
            FjPrintPreflashFailure otherFailure{};
            std::thread thread([&] {
                other = prepare(&input, &otherOut, &otherFailure, nullptr);
            });
            thread.join();
            status(other);
            metadata(otherFailure, FJ_PRINT_PREFLASH_REASON_NONE);
            status(call(&input, &out, &failure), FJ_STATUS_INTERNAL_FAILURE);
            metadata(failure, FJ_PRINT_PREFLASH_REASON_PANIC);
            zero(out);
            status(call(&input, &out, &failure));
            // Immutable sources may alias each other; only writable storage must be disjoint.
            source.sensitivity.fill(1);
            input = {{source.sensitivity.data(), 81}, {source.sensitivity.data(), 81}, {source.sensitivity.data(), 243}};
            status(call(&input, &out, &failure));
            metadata(failure, FJ_PRINT_PREFLASH_REASON_NONE);
            FjPrintPreflashRaw retained{};
            {
                Inputs ephemeral(captured.at("cases")[0]);
                const auto wire = ephemeral.wire();
                status(prepare(&wire, &retained, &failure, nullptr));
                ephemeral.light.fill(0);
            }
            require(words(retained.cmy, 3) == captured.at("cases")[0].at("raw"), "raw facade result owns all words after input destruction");
        }
    }
    void native(const Json& captured) {
        for (const auto& row : captured.at("cases")) {
            Inputs input(row);
            std::array<float, 3> out{-13, -13, -13};
            std::string diagnostic;
            const bool success = JuicerAssets::prepare_print_preflash(input.base, input.light, input.channels(), out, diagnostic);
            require(success == row.at("success").get<bool>(), "native preflash success/failure parity");
            if (success)
                require(words(out.data(), 3) == row.at("raw") && diagnostic.empty(), "native complete exact triplet including infinity");
            else
                require(words(out.data(), 3) == Json::array({0, 0, 0}) && diagnostic == row.at("diagnostic").get<std::string>(), "native exact old malformed-resource diagnostic and cleared result");
        }
        Inputs input(captured.at("cases")[0]);
        for (unsigned fault = 1; fault <= 16; ++fault) {
            nextProjection = fault;
            std::array<float, 3> out{-13, -13, -13};
            std::string diagnostic;
            bool threw = false;
            try {
                JuicerAssets::prepare_print_preflash(input.base, input.light, input.channels(), out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                status(failure.failure.status, fault == 15 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                threw = true;
            }
            require(threw && nextProjection == 0 && words(out.data(), 3) == Json::array({0, 0, 0}), "native preflash rejects contradictory envelope/diagnostic without publication");
        }
    }
    void products(const std::filesystem::path& resources, const Json& captured) {
        require(product_observations(resources) == captured.at("products"), "all 72 frozen caller values, input bindings, descriptor and final/seed identities");
        caller(resources);
    }
} // namespace PrintPreflashTest
namespace JuicerAssets::RecipeTest {
    void print_preflash(FjStatus& status, FjPrintPreflashFailure& failure, FjPrintPreflashRaw&, FjErrorBuffer& error) {
        const auto fault = std::exchange(PrintPreflashTest::nextProjection, 0u);
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
                failure.reason = FJ_PRINT_PREFLASH_REASON_CHANNEL;
                failure.channel = 3;
                break;
            case 7:
                status.category = FJ_STATUS_PREPARATION_FAILURE;
                failure.reason = FJ_PRINT_PREFLASH_REASON_CHANNEL;
                failure.channel = UINT32_MAX;
                break;
            case 8:
                status.category = FJ_STATUS_PREPARATION_FAILURE;
                failure.reason = FJ_PRINT_PREFLASH_REASON_NONE;
                break;
            case 9:
                failure.reason = FJ_PRINT_PREFLASH_REASON_PANIC;
                break;
            case 10:
                failure.reason = FJ_PRINT_PREFLASH_REASON_RAW;
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
                failure.reason = FJ_PRINT_PREFLASH_REASON_RAW;
                break;
            case 16:
                status.category = FJ_STATUS_INTERNAL_FAILURE;
                failure.reason = FJ_PRINT_PREFLASH_REASON_PANIC;
                break;
            default:
                break;
        }
    }
} // namespace JuicerAssets::RecipeTest

namespace PrintPreflashTest {
    ParamSnapshot controls() {
        ParamSnapshot p;
        p.filmProfileKey = "kodak_portra_400";
        p.printProfileKey = "kodak_portra_endura";
        p.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
        p.cameraAutoExposureEnabled = 0;
        p.inputCompressionEnabled = 0;
        p.outputGamutCompressionEnabled = 0;
        p.printPreflashExposure = 0.125;
        return p;
    }
    FocusedRenderStateBuildProduct build(const ParamSnapshot& p) {
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        require(build_print_render_state_product(p, product, diagnostic), diagnostic.c_str());
        return product;
    }
    void expect_panic(const auto& operation) {
        bool threw = false;
        try {
            operation();
        } catch (const JuicerCuda::ExecutionFailure& failure) {
            status(failure.failure.status, FJ_STATUS_INTERNAL_FAILURE);
            threw = true;
        }
        require(threw, "actual preflash constructor consumes the one-shot stage-11 fault");
    }
    void payload_guard() {
        std::array<float, 81> samples{};
        samples.fill(1.0f);
        JuicerCuda::PrintPreparedView view{};
        view.active = true;
        view.preflashActive = true;
        view.spectralSampleCount = 81;
        view.preparationHash = view.filmDensityTablesHash = view.profileTablesHash = view.mainIlluminantHash = view.preflashIlluminantHash = view.preflashRawHash = view.balanceHash = 1;
        view.filmDensityTables.K = 81;
        view.filmDensityTables.hasBaseline = 1;
        view.filmDensityTables.epsC = view.filmDensityTables.epsM = view.filmDensityTables.epsY = view.filmDensityTables.baseDensityMin = samples.data();
        view.printSensC = view.printSensM = view.printSensY = {nullptr, samples.data(), 81, 0, 80};
        view.printDcC = view.printDcM = view.printDcY = {samples.data(), samples.data(), 81, 0, 80};
        view.mainIlluminant = view.preflashIlluminant = samples.data();
        PrintExposureRecipe exposure;
        exposure.printExposure = 1;
        exposure.preflashExposure = 0.125f;
        JuicerCuda::PrintCudaPayloadPack packed;
        std::string diagnostic;
        require(JuicerCuda::pack_print_cuda_payloads(exposure, view, 1, packed, diagnostic), diagnostic.c_str());
        view.preflashRawCmy[0] = std::numeric_limits<float>::infinity();
        require(!JuicerCuda::pack_print_cuda_payloads(exposure, view, 1, packed, diagnostic) && diagnostic == "ResourceDescriptorMismatch phase=4B field=preflash_raw", "later payload guard still rejects successful narrowing infinity");
        view.preflashActive = false;
        require(JuicerCuda::pack_print_cuda_payloads(exposure, view, 1, packed, diagnostic), "disabled preflash omits executable raw admission");
    }
    void caller(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        auto& assets = JuicerProcess::root().assets();
        JuicerProcess::root().ensure_bootstrap();
        auto p = controls();
        auto active = build(p);
        p.printPreflashExposure = 0;
        auto disabled = build(p);
        JuicerCuda::PrintResourcePreparation request{&disabled.recipe, &assets, &*disabled.payload.printMainIlluminant};
        JuicerCuda::PrintResourceInput input;
        std::array<float, 81> light{};
        std::string diagnostic;
        status(fj_test_film_recipe_fault(11, 1));
        p.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        FocusedRenderStateBuildProduct direct;
        require(build_direct_render_state_product(p, direct, diagnostic) && !direct.payload.printMainIlluminant, "direct construction omits preflash");
        require(JuicerCuda::build_print_resource_input(request, input, light, diagnostic), "disabled false/false builder omits raw construction");
        require(words(input.preflashRawCmy.data(), 3) == Json::array({0, 0, 0}), "disabled raw identity/value omitted");
        request = {&active.recipe, &assets, &*active.payload.printMainIlluminant};
        expect_panic([&] {
            JuicerCuda::build_print_resource_input(request, input, light, diagnostic);
        });
        require(JuicerCuda::build_print_resource_input(request, input, light, diagnostic), "unconditional active builder recovers");
        const auto retained = input.preflashRawCmy;
        for (int repeat = 0; repeat < 2; ++repeat) {
            status(fj_test_film_recipe_fault(11, 1));
            expect_panic([&] {
                JuicerCuda::build_print_resource_input(request, input, light, diagnostic);
            });
            require(JuicerCuda::build_print_resource_input(request, input, light, diagnostic) && input.preflashRawCmy == retained, "repeated false/false path always constructs and recovers exact value");
        }
        payload_guard();
    }
    void cuda_ok(cudaError_t code) {
        require(code == cudaSuccess, cudaGetErrorString(code));
    }
    void cache(const std::filesystem::path& resources, const Json& captured) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        auto& assets = JuicerProcess::root().assets();
        JuicerProcess::root().ensure_bootstrap();
        cuda_ok(cudaSetDevice(0));
        cuda_ok(cudaFree(nullptr));
        void* context = nullptr;
        std::string diagnostic;
        require(JuicerCuda::query_current_cuda_context(context, diagnostic) && context, "cache fixture owns current native context");
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, context};
        constexpr std::uint64_t kBudgetBytes = std::uint64_t{1024} * 1024 * 1024;
        auto ledger = JuicerCuda::DeviceAllocationLedger::create({0, kBudgetBytes}, diagnostic);
        require(bool(ledger), diagnostic.c_str());
        require(ledger->bind_or_validate_cap(kBudgetBytes, diagnostic), diagnostic.c_str());
        auto p = controls();
        auto product = build(p);
        const auto request_for = [&](const FocusedRenderStateBuildProduct& value) {
            return JuicerCuda::PrintResourcePreparation{&value.recipe, &assets, &*value.payload.printMainIlluminant};
        };
        JuicerCuda::Failure failure;
        {
            JuicerCuda::Resources cached(key, 1, ledger);
            auto request = request_for(product);
            const auto prepare = [&](const JuicerCuda::PrintResourcePreparation& value) {
                require(JuicerCuda::prepare_print_resources(cached, value, nullptr, failure), failure.diagnostic.c_str());
                cuda_ok(cudaDeviceSynchronize());
            };
            const auto consume_fault = [&] {
                std::array<float, 3> raw{};
                expect_panic([&] {
                    JuicerAssets::prepare_print_preflash(product.recipe.profileRoute.filmProfile->data.baseDensity,
                                                         cached.printPreflashIllumFilteredHost,
                                                         product.recipe.profileRoute.printProfile->data.linearSensitivity,
                                                         raw,
                                                         diagnostic);
                });
            };
            prepare(request);
            const auto originalHash = cached.printPreflashRawDescriptorHash;
            const auto originalLightHash = cached.printPreflashIlluminantDescriptorHash;
            auto* const originalLight = cached.printPreflashIllumFiltered;
            const auto originalRaw = words(cached.printPreflashRaw, 3);
            status(fj_test_film_recipe_fault(11, 1));
            prepare(request);
            consume_fault();
            for (unsigned variant : {5u, 3u, 4u}) {
                auto changedControls = p;
                if (variant == 5)
                    changedControls.printPreflashExposure = .25;
                if (variant == 3)
                    changedControls.printUiYmcCc = {2.1234567890123, 3.2345678901234, 4.3456789012345};
                if (variant == 4)
                    changedControls.outputCctfEncoding = 0;
                auto changed = build(changedControls);
                status(fj_test_film_recipe_fault(11, 1));
                prepare(request_for(changed));
                consume_fault();
                require(cached.printPreflashRawDescriptorHash == originalHash && cached.printPreflashIlluminantDescriptorHash == originalLightHash && cached.printPreflashIllumFiltered == originalLight && words(cached.printPreflashRaw, 3) == originalRaw, "amount/main-only/output edits preserve raw reuse and illumination residency");
            }
            prepare(request);
            cached.printPreflashRawDescriptorHash = 0;
            const auto hostLight = cached.printPreflashIllumFilteredHost;
            status(fj_test_film_recipe_fault(11, 1));
            expect_panic([&] {
                JuicerCuda::prepare_print_resources(cached, request, nullptr, failure);
            });
            require(cached.printPreflashRawDescriptorHash == 0 && words(cached.printPreflashRaw, 3) == originalRaw, "failed raw miss commits no new raw identity or partial triplet");
            prepare(request);
            require(cached.printPreflashRawDescriptorHash == originalHash && cached.printPreflashIllumFiltered == originalLight && cached.printPreflashIllumFilteredHost == hostLight && words(cached.printPreflashRaw, 3) == originalRaw, "illuminant-hit/raw-miss uses retained host light and recovers");
            for (bool film : {true, false}) {
                auto changed = product.recipe;
                if (film)
                    ++changed.profileRoute.filmProfileAssetVersionToken;
                else
                    ++changed.profileRoute.printProfileAssetVersionToken;
                const JuicerCuda::PrintResourcePreparation changedRequest{&changed, &assets, &*product.payload.printMainIlluminant};
                status(fj_test_film_recipe_fault(11, 1));
                expect_panic([&] {
                    JuicerCuda::prepare_print_resources(cached, changedRequest, nullptr, failure);
                });
                require(cached.printPreflashRawDescriptorHash == originalHash, "source identity failure preserves old committed raw key");
                prepare(changedRequest);
                require(cached.printPreflashRawDescriptorHash != originalHash, "film/print source identity contributes to raw key");
                prepare(request);
            }
            auto changedControls = p;
            changedControls.preflashMFilterCc = 5;
            changedControls.preflashYFilterCc = -3;
            auto changed = build(changedControls);
            status(fj_test_film_recipe_fault(11, 1));
            expect_panic([&] {
                JuicerCuda::prepare_print_resources(cached, request_for(changed), nullptr, failure);
            });
            require(cached.printPreflashRawDescriptorHash == originalHash, "failed changed-light integral commits no new raw key");
            prepare(request_for(changed));
            require(cached.printPreflashRawDescriptorHash != originalHash && cached.printPreflashIlluminantDescriptorHash != originalLightHash, "preflash offsets invalidate both light and raw families");
            for (const auto& row : captured.at("products"))
                if (row.at("route") == 1 && row.at("mode") == p.spectralUpsamplingMode && row.at("variant") == 2)
                    require(words(cached.printPreflashRaw, 3) == row.at("raw"), "changed-offset cache result matches frozen actual caller");
            auto disabledControls = p;
            disabledControls.printPreflashExposure = 0;
            auto disabled = build(disabledControls);
            status(fj_test_film_recipe_fault(11, 1));
            prepare(request_for(disabled));
            require(!cached.printPreflashValid && cached.printPreflashRawDescriptorHash == 0 && cached.printPreflashIlluminantDescriptorHash == 0 && words(cached.printPreflashRaw, 3) == Json::array({0, 0, 0}), "disabled route clears preflash validity/value/keys");
            consume_fault();
            JuicerCuda::PrintResourceInput bound;
            std::array<float, 81> light{};
            require(JuicerCuda::build_print_resource_input(request, bound, light, diagnostic), diagnostic.c_str());
            status(fj_test_film_recipe_fault(11, 1));
            require(JuicerCuda::prepare_print_resources(cached, bound, nullptr, failure), failure.diagnostic.c_str());
            cuda_ok(cudaDeviceSynchronize());
            consume_fault();
            require(words(cached.printPreflashRaw, 3) == originalRaw, "supplied-input path binds already-derived raw without a second constructor");
            cached.contextEpoch = 0;
            status(fj_test_film_recipe_fault(11, 1));
            require(!JuicerCuda::prepare_print_resources(cached, request, nullptr, failure), "invalid owner epoch rejects before derivation");
            cached.contextEpoch = 1;
            consume_fault();
            require(JuicerCuda::drain_for_context_retire(cached, failure), failure.diagnostic.c_str());
        }
        require(ledger->snapshot().chargedBytes == 0, "cache fixture releases every device allocation");
        JuicerCuda::purge_pinned_upload_staging_for_context(0, context, JuicerCuda::PinnedUploadPurgeDisposition::NormalRetire);
    }
} // namespace PrintPreflashTest
