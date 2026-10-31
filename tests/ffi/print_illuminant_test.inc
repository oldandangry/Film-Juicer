namespace PrintPreflashTest {
    JuicerCudaTest::PreparedFrameFixture::FrameFacts frame_facts(const ParamSnapshot& p);
}

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
    void metadata(const FjPrintIlluminantFailure& failure, std::uint32_t reason, std::uint32_t sample = UINT32_MAX);
    void zero(const FjIlluminant& out);
    struct PreparedPrintFacts {
        std::uint64_t printHash = 0;
        std::size_t preflashCount = 0;
    };
    bool inspect_prepared_print(void* opaque,
                                const JuicerCudaTest::PreparedFrameFixture::Inputs& prepared,
                                std::string& diagnostic) {
        auto& facts = *static_cast<PreparedPrintFacts*>(opaque);
        const auto* input = std::get_if<JuicerCuda::PrintResourceInput>(&prepared.preparation.print);
        if (!input) {
            diagnostic = "prepared print state has no complete resource input";
            return false;
        }
        facts.printHash = input->descriptors.hash;
        facts.preflashCount = input->preflashIlluminant.size();
        return true;
    }
    void consume_stage10(const Json& literalCases) {
        const auto& row = literalCases.at("cases")[0];
        auto input = params(row);
        float source[81]{};
        copy_words(row.at("source"), source);
        FjIlluminant output{};
        FjPrintIlluminantFailure failure{};
        status(fj_test_print_illuminant_prepare(&input, {source, std::size(source)}, &output, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
        metadata(failure, FJ_PRINT_ILLUMINANT_REASON_PANIC);
        zero(output);
    }
    void metadata(const FjPrintIlluminantFailure& failure, std::uint32_t reason, std::uint32_t sample) {
        require(failure.reason == reason && failure.sample == sample, "exact print illuminant failure metadata");
    }
    void zero(const FjIlluminant& out) {
        for (float sample : out.samples)
            require(std::bit_cast<std::uint32_t>(sample) == 0, "independently writable print illuminant clears every word to +0");
    }
    void seed(FjIlluminant& out) {
        std::fill_n(out.samples, 81, -13.0f);
    }
    void run_SF_TEMP_BRIDGE_native_main_illuminant_panic_retry(const Json& literalCases) {
        const auto& row = literalCases.at("cases")[0];
        const auto input = params(row);
        DichroicFilterRecipe filters;
        std::copy_n(input.edges_nm, 4, filters.customEdgesNm.begin());
        std::copy_n(input.transitions_nm, 4, filters.customTransitionsNm.begin());
        const CmyCcTriplet cmy{input.cmy_cc[0], input.cmy_cc[1], input.cmy_cc[2]};
        std::array<float, 81> source{};
        for (std::size_t index = 0; index < source.size(); ++index)
            source[index] = std::bit_cast<float>(row.at("source")[index].get<std::uint32_t>());
        std::array<float, 81> out{};
        out.fill(-13.0f);
        std::string diagnostic;
        status(fj_test_film_recipe_fault(10, 1));
        bool panicked = false;
        try {
            (void)JuicerAssets::prepare_print_illuminant(filters, source, cmy, out, diagnostic);
        } catch (const JuicerCuda::ExecutionFailure& error) {
            panicked = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
        }
        require(panicked && std::all_of(out.begin(), out.end(), [](float value) {
                    return std::bit_cast<std::uint32_t>(value) == 0;
                }),
                "SF_TEMP_BRIDGE native main illuminant panic clears output");
        require(JuicerAssets::prepare_print_illuminant(filters, source, cmy, out, diagnostic) &&
                    diagnostic.empty() && words(out.data(), out.size()) == row.at("samples"),
                "SF_TEMP_BRIDGE native main illuminant retry matches retained literal result");
    }

    void raw_main_illuminant_followup(const FjPreparedStateView& held) {
        const auto source = held.print_source_illuminant;
        const auto& expected = held.print.main_illuminant;
        require(source.count == 81 && expected.count == 81, "held print state supplies complete main illuminant inputs and output");
        DichroicFilterRecipe filters;
        FjPrintIlluminantParams input{};
        std::copy(filters.customEdgesNm.begin(), filters.customEdgesNm.end(), input.edges_nm);
        std::copy(filters.customTransitionsNm.begin(), filters.customTransitionsNm.end(), input.transitions_nm);
        std::copy_n(held.print.main_cc_cmy, 3, input.cmy_cc);
        FjIlluminant out{};
        seed(out);
        FjPrintIlluminantFailure failure{99, 98};
        status(fj_test_print_illuminant_prepare(&input, source, &out, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
        metadata(failure, FJ_PRINT_ILLUMINANT_REASON_PANIC);
        zero(out);
        status(fj_test_print_illuminant_prepare(&input, source, &out, &failure, nullptr));
        metadata(failure, FJ_PRINT_ILLUMINANT_REASON_NONE);
        require(words(out.samples, 81) == words(expected.data, expected.count), "raw main illuminant retry matches the actual held Rust publication");
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
    void products(const Json& captured, const std::filesystem::path& resources, const Json& literalCases) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        auto& assets = JuicerProcess::root().assets();
        JuicerProcess::root().ensure_bootstrap();
        require(captured.at("products").size() == 72, "exact genuine print illuminant product membership");
        run_SF_TEMP_BRIDGE_native_main_illuminant_panic_retry(literalCases);
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
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<PreparedState> product;
            const auto result = prepare_state(assets.borrow_rust_assets(), p, product, error);
            const bool success = result.category == FJ_STATUS_SUCCESS && product.has_value();
            require(success, diagnostic.data());
            FjPreparedStateView view{};
            status(product->view(view, error));
            const auto& print = view.print;
            Json actual = {{"route", route}, {"mode", mode}, {"variant", variant}, {"success", success}, {"diagnostic", std::string(diagnostic.data(), error.length)}, {"snapshot_hash", hash_params(p)}, {"final_hash", view.recipe_hash}, {"print_hash", view.print_recipe_hash}, {"balance_hash", view.print_balance.hash}, {"main_present", print.main_illuminant.count == 81}};
            if (route % 2) {
                actual["main"] = words(print.main_illuminant.data, print.main_illuminant.count);
                const auto& mainCmy = print.main_cc_cmy;
                actual["main_cmy"] = words(mainCmy, 3);
                if ((print.flags & FJ_PRINT_PREFLASH) != 0) {
                    actual["preflash_cmy"] = words(print.preflash_cc_cmy, 3);
                } else {
                    require(p.printPreflashExposure == 0 && p.preflashMFilterCc == 0 && p.preflashYFilterCc == 0 &&
                                std::all_of(p.printUiYmcCc.begin(), p.printUiYmcCc.end(), [](double value) {
                                    return value == 0;
                                }),
                            "inactive preflash metadata retains the original zero-offset controls");
                    const auto zero = [](float value) {
                        return std::bit_cast<std::uint32_t>(value) == 0;
                    };
                    require(print.preflash_illuminant.data == nullptr && print.preflash_illuminant.count == 0 &&
                                print.preflash_illuminant_hash == 0 && print.preflash_raw_hash == 0 &&
                                zero(print.preflash_exposure) &&
                                std::all_of(std::begin(print.preflash_cc_cmy), std::end(print.preflash_cc_cmy), zero) &&
                                std::all_of(std::begin(print.preflash_raw_cmy), std::end(print.preflash_raw_cmy), zero),
                            "inactive preflash projection is absent/zero before capture metadata mapping");
                    actual["preflash_cmy"] = words(mainCmy, 3);
                }
                const auto source = view.print_source_illuminant;
                const auto curves = assets.illuminant_filter_curves();
                require(p.enlIll == 3 && source.count == curves->tungstenKg3.linear.size() &&
                            std::equal(source.data, source.data + source.count, curves->tungstenKg3.linear.begin()),
                        "completed source illuminant matches the selected TH-KG3 asset identity");
                actual["source_key"] = "TH-KG3";
                std::array<float, 81> preflashScratch{};
                if (print.preflash_illuminant.count != 0) {
                    require(print.preflash_illuminant.count == preflashScratch.size(), "active preflash has 81 samples");
                    std::copy_n(print.preflash_illuminant.data, preflashScratch.size(), preflashScratch.data());
                }
                // Preserve the old fixed 81-word scratch output; disabled input
                // is an empty span in FjPrint and the original scratch stayed +0.
                actual["preflash"] = words(preflashScratch.data(), preflashScratch.size());
                actual["raw"] = words(print.preflash_raw_cmy, 3);
                actual["descriptors"] = {print.hash, print.main_illuminant_hash, print.preflash_illuminant_hash, print.preflash_raw_hash, print.balance_hash};
            }
            if (actual != expected)
                throw std::runtime_error("print illuminant product route=" + std::to_string(route) +
                                         " mode=" + std::to_string(mode) + " variant=" + std::to_string(variant) +
                                         " first difference " + Json::diff(expected, actual).front().dump());
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
        require(original.status == PendingRenderAdmissionStatus::AdmittedPrint && bool(original.printState), "print illuminant publication baseline");
        const auto originalView = JuicerTestSupport::published_view(original);
        require(originalView.print.main_illuminant.count == 81, "baseline publishes all main illuminant samples");
        const auto originalHash = originalView.recipe_hash;
        const auto originalMain = words(originalView.print.main_illuminant.data, originalView.print.main_illuminant.count);
        const auto counter = state.buildCounterNext.load();
        controls.printUiYmcCc[0] = 2.5;
        pending(controls);
        status(fj_test_film_recipe_fault(10, 1));
        const auto admitted = admit_pending_render_state(state);
        require(admitted.status == PendingRenderAdmissionStatus::AdmittedPrint && bool(admitted.printState) &&
                    admitted.printState != original.printState,
                "actual main illuminant publication skips the legacy facade-only panic");
        require(state.buildCounterNext.load() == counter + 1 &&
                    JuicerAtomic::load_shared_ptr(&state.activePrintState) == admitted.printState &&
                    originalView.recipe_hash == originalHash &&
                    words(originalView.print.main_illuminant.data, originalView.print.main_illuminant.count) == originalMain,
                "complete main illuminant publication advances once and preserves prior held values");
        const auto held = JuicerTestSupport::published_view(admitted);
        raw_main_illuminant_followup(held);
        const auto repeated = admit_pending_render_state(state);
        require(repeated.printState == admitted.printState && state.buildCounterNext.load() == counter + 1,
                "unchanged main illuminant publication reuses after raw facade panic/retry");

        std::string diagnostic;
        controls.printPreflashExposure = 0;
        auto disabled = JuicerTestSupport::prepare_product(controls);
        PreparedPrintFacts disabledFacts;
        status(fj_test_film_recipe_fault(10, 1));
        require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                    disabled, PrintPreflashTest::frame_facts(controls), &disabledFacts, &inspect_prepared_print, diagnostic),
                diagnostic.c_str());
        require(disabledFacts.printHash != 0 && disabledFacts.preflashCount == 0, "disabled preflash omits constructor input");
        // The complete disabled input must leave the raw-constructor fault armed.
        consume_stage10(literalCases);
        controls.printPreflashExposure = 0.125;
        auto active = JuicerTestSupport::prepare_product(controls);
        PreparedPrintFacts activeFacts;
        status(fj_test_film_recipe_fault(10, 1));
        require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                    active, PrintPreflashTest::frame_facts(controls), &activeFacts, &inspect_prepared_print, diagnostic),
                diagnostic.c_str());
        require(activeFacts.printHash != 0 && activeFacts.preflashCount == 81,
                "active preflash input is supplied from the completed held state");
        consume_stage10(literalCases);
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
