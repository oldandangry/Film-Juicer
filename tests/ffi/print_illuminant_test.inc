namespace PrintIlluminantTest {
    using Prepare = FjStatus (*)(const FjPrintIlluminantParams*, FjFloatSpan, FjIlluminant*, FjPrintIlluminantFailure*, FjErrorBuffer*);
    thread_local unsigned nextProjection = 0;

    FjPrintIlluminantParams params(const Json& row) {
        FjPrintIlluminantParams out{};
        copy_words(row.at("edges_nm"), out.edges_nm);
        copy_words(row.at("transitions_nm"), out.transitions_nm);
        copy_words(row.at("cmy_cc"), out.cmy_cc);
        return out;
    }
    void metadata(const FjPrintIlluminantFailure& failure, std::uint32_t reason, std::uint32_t sample = UINT32_MAX) {
        require(failure.reason == reason && failure.sample == sample, "exact print illuminant failure metadata");
    }
    void zero(const FjIlluminant& out) {
        for (float sample : out.samples)
            require(std::bit_cast<std::uint32_t>(sample) == 0, "independently writable print illuminant clears every word to +0");
    }
    void seed(FjIlluminant& out) {
        std::fill_n(out.samples, 81, -13.0f);
    }
    void values(const Json& captured) {
        require(captured.at("cases").size() == 127, "exact adopted print illuminant literal membership");
        for (Prepare prepare : {fj_legacy_print_illuminant_prepare, fj_test_print_illuminant_prepare})
            for (const auto& row : captured.at("cases")) {
                auto input = params(row);
                float source[81];
                copy_words(row.at("source"), source);
                FjIlluminant out{};
                seed(out);
                FjPrintIlluminantFailure failure{99, 98};
                const bool success = row.at("success");
                status(prepare(&input, {source, 81}, &out, &failure, nullptr), success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (success) {
                    metadata(failure, FJ_PRINT_ILLUMINANT_REASON_NONE);
                    require(words(out.samples, 81) == row.at("samples"), "all 81 exact owner-approved print illuminant words");
                } else {
                    std::uint32_t first = UINT32_MAX;
                    for (std::uint32_t i = 0; i < 81; ++i)
                        if (row.at("samples")[i] == 0xc1500000u) {
                            first = i;
                            break;
                        }
                    metadata(failure, first == UINT32_MAX ? FJ_PRINT_ILLUMINANT_REASON_ENERGY : FJ_PRINT_ILLUMINANT_REASON_SAMPLE, first);
                    zero(out);
                }
            }
    }
    void transport(const Json& captured) {
        auto input = params(captured.at("cases")[0]);
        float source[81];
        std::fill_n(source, 81, 1.0f);
        for (Prepare prepare : {fj_legacy_print_illuminant_prepare, fj_test_print_illuminant_prepare}) {
            FjIlluminant out{};
            FjPrintIlluminantFailure failure{};
            const auto call = [&](const FjPrintIlluminantParams* p, FjFloatSpan span, FjIlluminant* result, FjPrintIlluminantFailure* detail, FjErrorBuffer* error = nullptr) {
                seed(out);
                failure = {99, 98};
                return prepare(p, span, result, detail, error);
            };
            const auto raw = [&](const FjPrintIlluminantParams* p, FjFloatSpan span, FjErrorBuffer* error = nullptr) {
                status(call(p, span, &out, &failure, error), FJ_STATUS_UNSUPPORTED_INPUT);
                metadata(failure, FJ_PRINT_ILLUMINANT_REASON_RAW);
                zero(out);
            };
            alignas(FjPrintIlluminantParams) std::array<std::byte, sizeof(FjPrintIlluminantParams) + 1> unaligned{};
            raw(nullptr, {source, 81});
            raw(reinterpret_cast<const FjPrintIlluminantParams*>(unaligned.data() + 1), {source, 81});
            const auto end = UINTPTR_MAX & ~(static_cast<std::uintptr_t>(alignof(float)) - 1);
            raw(reinterpret_cast<const FjPrintIlluminantParams*>(end), {source, 81});
            raw(&input, {nullptr, 81});
            raw(&input, {reinterpret_cast<const float*>(reinterpret_cast<const char*>(source) + 1), 81});
            raw(&input, {reinterpret_cast<const float*>(end), 81});
            for (std::size_t count : {std::size_t{0}, std::size_t{80}, std::size_t{82}, SIZE_MAX})
                raw(&input, {source, count});
            status(call(&input, {source, 81}, nullptr, &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_RAW);
            status(call(&input, {source, 81}, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            zero(out);
            status(call(&input, {source, 81}, reinterpret_cast<FjIlluminant*>(end), &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_RAW);
            status(call(&input, {source, 81}, &out, reinterpret_cast<FjPrintIlluminantFailure*>(end)), FJ_STATUS_UNSUPPORTED_INPUT);
            zero(out);
            status(call(&input, {source, 81}, reinterpret_cast<FjIlluminant*>(unaligned.data() + 1), &failure), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_RAW);
            status(call(&input, {source, 81}, &out, reinterpret_cast<FjPrintIlluminantFailure*>(unaligned.data() + 1)), FJ_STATUS_UNSUPPORTED_INPUT);
            zero(out);

            source[37] = std::numeric_limits<float>::quiet_NaN();
            std::array<char, 256> full{};
            FjErrorBuffer error{full.data(), full.size(), 99};
            status(call(&input, {source, 81}, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_SAMPLE, 37);
            zero(out);
            const std::string diagnostic(full.data());
            require(error.length == diagnostic.size() && error.length > 8, "full diagnostic length excludes NUL");
            for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{8}}) {
                std::array<char, 8> small{};
                small.fill('x');
                error = {capacity ? small.data() : reinterpret_cast<char*>(UINTPTR_MAX), capacity, 99};
                status(call(&input, {source, 81}, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
                metadata(failure, FJ_PRINT_ILLUMINANT_REASON_SAMPLE, 37);
                zero(out);
                require(error.length == (capacity ? capacity - 1 : 0), "bounded diagnostic length");
                if (capacity)
                    require(small[capacity - 1] == '\0' && std::string(small.data()) == diagnostic.substr(0, capacity - 1), "bounded diagnostic prefix and NUL");
                else
                    require(small[0] == 'x', "zero capacity ignores backing");
            }
            error = {nullptr, 1, 99};
            raw(&input, {source, 81}, &error);
            error = {reinterpret_cast<char*>(UINTPTR_MAX), 2, 99};
            raw(&input, {source, 81}, &error);
            alignas(FjErrorBuffer) std::array<std::byte, sizeof(FjErrorBuffer) + 1> misalignedError{};
            raw(&input, {source, 81}, reinterpret_cast<FjErrorBuffer*>(misalignedError.data() + 1));
            raw(&input, {source, 81}, reinterpret_cast<FjErrorBuffer*>(UINTPTR_MAX & ~(static_cast<std::uintptr_t>(alignof(FjErrorBuffer)) - 1)));
            source[37] = 1;
            // Two invalid samples prove first-computed-failure order, followed by recovery.
            source[5] = -1;
            source[37] = std::numeric_limits<float>::infinity();
            status(call(&input, {source, 81}, &out, &failure), FJ_STATUS_PREPARATION_FAILURE);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_SAMPLE, 5);
            zero(out);
            std::fill_n(source, 81, 0.0f);
            status(call(&input, {source, 81}, &out, &failure), FJ_STATUS_PREPARATION_FAILURE);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_ENERGY);
            zero(out);
            std::fill_n(source, 81, 1.0f);
            for (bool rejectFirst : {false, true}) {
                status(fj_test_film_recipe_fault(10, 1));
                if (rejectFirst)
                    raw(nullptr, {source, 81});
                else {
                    status(call(&input, {source, 81}, &out, &failure), FJ_STATUS_INTERNAL_FAILURE);
                    metadata(failure, FJ_PRINT_ILLUMINANT_REASON_PANIC);
                    zero(out);
                }
                status(call(&input, {source, 81}, &out, &failure));
                metadata(failure, FJ_PRINT_ILLUMINANT_REASON_NONE);
                for (float sample : out.samples)
                    require(sample == 1.0f, "ordinary/fault recovery complete result");
            }
            status(fj_test_film_recipe_fault(10, 1));
            status(fj_test_film_recipe_fault(10, 0));
            status(fj_test_film_recipe_fault(10, 2), FJ_STATUS_UNSUPPORTED_INPUT);
            status(call(&input, {source, 81}, &out, &failure));
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_NONE);
            const auto held = words(out.samples, 81);
            std::fill_n(source, 81, 0.0f);
            require(words(out.samples, 81) == held, "result owns samples independently of source mutation");
            std::fill_n(source, 81, 1.0f);
        }
        status(fj_test_film_recipe_fault(10, 1));
        std::atomic<bool> isolated{false};
        std::thread other([&] {
            FjIlluminant out{};
            FjPrintIlluminantFailure failure{};
            const auto result = fj_test_print_illuminant_prepare(&input, {source, 81}, &out, &failure, nullptr);
            isolated = result.category == FJ_STATUS_SUCCESS && failure.reason == FJ_PRINT_ILLUMINANT_REASON_NONE;
        });
        other.join();
        require(isolated, "stage-10 fault is thread-local");
        FjIlluminant out{};
        FjPrintIlluminantFailure failure{};
        status(fj_test_print_illuminant_prepare(&input, {source, 81}, &out, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
        metadata(failure, FJ_PRINT_ILLUMINANT_REASON_PANIC);
        zero(out);
        status(fj_test_print_illuminant_prepare(&input, {source, 81}, &out, &failure, nullptr));
    }
    void products(const std::filesystem::path& resources, const Json& captured) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        auto& assets = JuicerProcess::root().assets();
        JuicerProcess::root().ensure_bootstrap();
        require(captured.at("products").size() == 72, "exact genuine print illuminant product membership");
        for (const auto& expected : captured.at("products")) {
            const unsigned route = expected.at("route");
            const int mode = expected.at("mode");
            const unsigned variant = expected.at("variant");
            ParamSnapshot p;
            p.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            p.printProfileKey = "kodak_portra_endura";
            p.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            p.spectralUpsamplingMode = mode;
            p.cameraAutoExposureEnabled = 0;
            p.inputCompressionEnabled = 0;
            p.outputGamutCompressionEnabled = 0;
            p.printPreflashExposure = variant ? 0.125 : 0.;
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
            Json actual = {{"route", route}, {"mode", mode}, {"variant", variant}, {"success", success}, {"diagnostic", diagnostic}, {"snapshot_hash", hash_params(p)}, {"final_hash", recipe.hash}, {"print_hash", recipe.print.hash}, {"balance_hash", recipe.print.balance.hash}, {"main_present", bool(product.payload.printMainIlluminant)}};
            if (route % 2) {
                const auto& print = recipe.print;
                const float mainCmy[]{print.filters.mainCmyCc.c, print.filters.mainCmyCc.m, print.filters.mainCmyCc.y};
                const float preflashCmy[]{print.filters.preflashCmyCc.c, print.filters.preflashCmyCc.m, print.filters.preflashCmyCc.y};
                actual["main"] = words(product.payload.printMainIlluminant->data(), 81);
                actual["main_cmy"] = words(mainCmy, 3);
                actual["preflash_cmy"] = words(preflashCmy, 3);
                actual["source_key"] = print.illuminant.key;
                JuicerCuda::PrintResourcePreparation request{&recipe, &assets, &*product.payload.printMainIlluminant};
                JuicerCuda::PrintResourceInput input;
                std::array<float, 81> preflash{};
                require(JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic), diagnostic.c_str());
                actual["preflash"] = words(preflash.data(), preflash.size());
                actual["raw"] = words(input.preflashRawCmy.data(), input.preflashRawCmy.size());
                actual["descriptors"] = {input.descriptors.hash, input.descriptors.mainIlluminant.hash, input.descriptors.preflashIlluminant.hash, input.descriptors.preflashRaw.hash, input.descriptors.balance.hash};
            }
            require(actual == expected, "frozen actual main/preflash/direct products, consumed descriptors, balance and final hashes");
        }

        ParamSnapshot controls;
        controls.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
        controls.filmProfileKey = "kodak_portra_400";
        controls.printProfileKey = "kodak_portra_endura";
        controls.cameraAutoExposureEnabled = 0;
        controls.inputCompressionEnabled = 0;
        controls.outputGamutCompressionEnabled = 0;
        InstanceState state;
        const auto pending = [&](const ParamSnapshot& snapshot) {
            std::lock_guard<std::mutex> lock(state.pending.m);
            state.pending.value = PendingParamsState::Valid{snapshot, hash_params(snapshot)};
        };
        pending(controls);
        const auto original = admit_pending_render_state(state);
        require(bool(original.printState), "print illuminant publication baseline");
        const auto counter = state.buildCounterNext.load();
        controls.printUiYmcCc[0] = 2.5;
        pending(controls);
        status(fj_test_film_recipe_fault(10, 1));
        const auto rejected = admit_pending_render_state(state);
        require(rejected.status == PendingRenderAdmissionStatus::RebuildFailed && !rejected.printState && !rejected.directState, "main illuminant panic blocks new admission");
        require(state.buildCounterNext.load() == counter && JuicerAtomic::load_shared_ptr(&state.activePrintState) == original.printState, "failed illuminant build preserves admitted publication");
        const auto recovered = admit_pending_render_state(state);
        require(bool(recovered.printState) && recovered.printState != original.printState, "main illuminant publication recovers");
        require(admit_pending_render_state(state).printState == recovered.printState, "unchanged main illuminant publication reuses");

        controls.printPreflashExposure = 0.125;
        FocusedRenderStateBuildProduct active;
        std::string diagnostic;
        require(build_print_render_state_product(controls, active, diagnostic), "active preflash fixture completes through production");
        JuicerCuda::PrintResourcePreparation request{&recovered.printState->recipe, &assets, &*recovered.printState->payload.printMainIlluminant};
        JuicerCuda::PrintResourceInput input;
        std::array<float, 81> preflash{};
        status(fj_test_film_recipe_fault(10, 1));
        require(JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic), "disabled preflash omits constructor");
        // The disabled call must leave the fault armed for the active caller.
        request.recipe = &active.recipe;
        request.mainIlluminant = &*active.payload.printMainIlluminant;
        bool threw = false;
        try {
            JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic);
        } catch (const JuicerCuda::ExecutionFailure& failure) {
            status(failure.failure.status, FJ_STATUS_INTERNAL_FAILURE);
            threw = true;
        }
        require(threw, "active preflash calls the same stage-10 Rust constructor");
        require(JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic), "active preflash recovers after panic");
    }
    void native(const Json& captured) {
        const auto input = params(captured.at("cases")[0]);
        DichroicFilterRecipe filters;
        std::copy_n(input.edges_nm, 4, filters.customEdgesNm.begin());
        std::copy_n(input.transitions_nm, 4, filters.customTransitionsNm.begin());
        CmyCcTriplet cmy{};
        std::array<float, 81> source{};
        source.fill(1);
        std::array<float, 81> out{};
        std::string diagnostic;
        require(JuicerAssets::prepare_print_illuminant(filters, source, cmy, out, diagnostic) && diagnostic.empty() && words(out.data(), 81) == captured.at("cases")[0].at("samples"), "native named projection and complete copy");
        for (bool energy : {false, true}) {
            source.fill(energy ? 0.0f : 1.0f);
            if (!energy)
                source[37] = -1;
            out.fill(-13);
            require(!JuicerAssets::prepare_print_illuminant(filters, source, cmy, out, diagnostic) && diagnostic == "MalformedRequiredResource phase=4B field=filtered_print_illuminant", "exact native Sample/Energy diagnostic");
            for (float sample : out)
                require(std::bit_cast<std::uint32_t>(sample) == 0, "native failure clears output");
        }
        source.fill(1);
        for (unsigned fault = 1; fault <= 16; ++fault) {
            nextProjection = fault;
            out.fill(-13);
            bool threw = false;
            try {
                JuicerAssets::prepare_print_illuminant(filters, source, cmy, out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                status(failure.failure.status, fault == 15 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                threw = true;
            }
            require(threw && nextProjection == 0, "native rejects contradictory closed status/metadata/diagnostic");
            for (float sample : out)
                require(std::bit_cast<std::uint32_t>(sample) == 0, "native integrity failure publishes no result");
        }
    }
} // namespace PrintIlluminantTest
namespace JuicerAssets::RecipeTest {
    void print_illuminant(FjStatus& status, FjPrintIlluminantFailure& failure, FjIlluminant&, FjErrorBuffer& error) {
        const unsigned fault = PrintIlluminantTest::nextProjection;
        PrintIlluminantTest::nextProjection = 0;
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
                failure.sample = 0;
                break;
            case 6:
                failure.reason = FJ_PRINT_ILLUMINANT_REASON_SAMPLE;
                failure.sample = 81;
                break;
            case 7:
                status.category = FJ_STATUS_PREPARATION_FAILURE;
                failure.reason = FJ_PRINT_ILLUMINANT_REASON_SAMPLE;
                failure.sample = UINT32_MAX;
                break;
            case 8:
                status.category = FJ_STATUS_PREPARATION_FAILURE;
                failure.reason = FJ_PRINT_ILLUMINANT_REASON_ENERGY;
                failure.sample = 0;
                break;
            case 9:
                failure.reason = FJ_PRINT_ILLUMINANT_REASON_PANIC;
                break;
            case 10:
                failure.reason = FJ_PRINT_ILLUMINANT_REASON_RAW;
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
                failure.reason = FJ_PRINT_ILLUMINANT_REASON_RAW;
                break;
            case 16:
                status.category = FJ_STATUS_INTERNAL_FAILURE;
                failure.reason = FJ_PRINT_ILLUMINANT_REASON_PANIC;
                break;
            default:
                break;
        }
    }
} // namespace JuicerAssets::RecipeTest
