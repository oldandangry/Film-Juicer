namespace PrintPreflashTest {
    JuicerCudaTest::PreparedFrameFixture::FrameFacts frame_facts(const ParamSnapshot& p);
}

namespace PrintIlluminantTest {
    using Prepare = FjStatus (*)(const FjPrintIlluminantParams*, FjFloatSpan, FjIlluminant*, FjPrintIlluminantFailure*, FjErrorBuffer*);

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
        const auto* input = prepared.preparation.print ? &*prepared.preparation.print : nullptr;
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

    void raw_main_illuminant_followup(const FjPreparedStateView& held) {
        const auto source = held.print_source_illuminant;
        const auto& expected = held.print.main_illuminant;
        require(source.count == 81 && expected.count == 81, "held print state supplies complete main illuminant inputs and output");
        FjPrintIlluminantParams input{{516.0f, 500.0f, 610.0f, 607.0f},
                                      {12.0f, 8.0f, 8.0f, 8.0f},
                                      {}};
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
        const std::array<std::pair<std::string_view, Prepare>, 2> callers{{{"c11", fj_test_print_illuminant_c_prepare}, {"cpp", fj_test_print_illuminant_prepare}}};
        for (const auto& [caller, prepare] : callers)
            for (const auto& row : captured.at("cases")) {
                JuicerTests::CaseEvent case_event(caller, row.at("id").get_ref<const std::string&>());
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
                case_event.passed();
            }
    }
    void transport(const Json& captured) {
        JuicerTests::CaseEvent transport_event("cpp", "print-illuminant/transport");
        auto input = params(captured.at("cases")[0]);
        float source[81];
        std::fill_n(source, 81, 1.0f);
        std::vector<std::string> count_ids, capacity_ids, fault_ids, sample_ids;
        for (unsigned caller = 0; caller < 2; ++caller) {
            for (unsigned count = 0; count < 4; ++count)
                count_ids.emplace_back("caller-" + std::to_string(caller) + "/count-" + std::to_string(count));
            for (const auto capacity : {0u, 1u, 8u})
                capacity_ids.emplace_back("caller-" + std::to_string(caller) + "/capacity-" + std::to_string(capacity));
            for (bool reject_first : {false, true})
                fault_ids.emplace_back("caller-" + std::to_string(caller) + "/reject-first-" + std::to_string(reject_first));
            for (const char* sample : {"invalid-sample", "first-invalid-order", "zero-energy"})
                sample_ids.emplace_back("caller-" + std::to_string(caller) + "/" + sample);
        }
        JuicerTests::Cohort malformed_counts("print_illuminant.span_count", count_ids);
        JuicerTests::Cohort diagnostic_capacities("print_illuminant.diagnostic_capacity", capacity_ids);
        JuicerTests::Cohort preparation_faults("print_illuminant.fault_order", fault_ids);
        JuicerTests::Cohort sample_failures("print_illuminant.sample_failure", sample_ids);
        unsigned caller_index = 0;
        for (Prepare prepare : {fj_test_print_illuminant_c_prepare, fj_test_print_illuminant_prepare}) {
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
            unsigned count_index = 0;
            for (std::size_t count : {std::size_t{0}, std::size_t{80}, std::size_t{82}, SIZE_MAX}) {
                auto row_event = malformed_counts.case_event("caller-" + std::to_string(caller_index) + "/count-" + std::to_string(count_index++));
                raw(&input, {source, count});
                row_event.passed();
            }
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
            auto sample_event = sample_failures.case_event("caller-" + std::to_string(caller_index) + "/invalid-sample");
            std::array<char, 256> full{};
            FjErrorBuffer error{full.data(), full.size(), 99};
            status(call(&input, {source, 81}, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_SAMPLE, 37);
            zero(out);
            const std::string diagnostic(full.data());
            require(error.length == diagnostic.size() && error.length > 8, "full diagnostic length excludes NUL");
            sample_event.passed();
            for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{8}}) {
                auto row_event = diagnostic_capacities.case_event("caller-" + std::to_string(caller_index) + "/capacity-" + std::to_string(capacity));
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
                row_event.passed();
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
            auto first_order_event = sample_failures.case_event("caller-" + std::to_string(caller_index) + "/first-invalid-order");
            status(call(&input, {source, 81}, &out, &failure), FJ_STATUS_PREPARATION_FAILURE);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_SAMPLE, 5);
            zero(out);
            first_order_event.passed();
            std::fill_n(source, 81, 0.0f);
            auto energy_event = sample_failures.case_event("caller-" + std::to_string(caller_index) + "/zero-energy");
            status(call(&input, {source, 81}, &out, &failure), FJ_STATUS_PREPARATION_FAILURE);
            metadata(failure, FJ_PRINT_ILLUMINANT_REASON_ENERGY);
            zero(out);
            energy_event.passed();
            std::fill_n(source, 81, 1.0f);
            for (bool rejectFirst : {false, true}) {
                auto row_event = preparation_faults.case_event("caller-" + std::to_string(caller_index) + "/reject-first-" + std::to_string(rejectFirst));
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
                row_event.passed();
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
            ++caller_index;
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
        transport_event.passed();
    }
    void products(const Json& captured, const std::filesystem::path& resources, const Json& literalCases) {
        JuicerProcess::ProcessRuntime owner;
        owner.create(resources.string().c_str());
        const FjAssets* assets = owner.borrow().assets;
        FjIlluminantsOwner* rawIlluminants = nullptr;
        status(fj_test_assets_illuminants_acquire(assets, &rawIlluminants, nullptr));
        require(rawIlluminants != nullptr, "raw selected illuminant owner acquired");
        struct ReleaseIlluminants {
            FjIlluminantsOwner* owner;
            ~ReleaseIlluminants() {
                if (owner && fj_test_illuminants_release(owner, nullptr).category != FJ_STATUS_SUCCESS)
                    std::abort();
            }
        } release{rawIlluminants};
        FjIlluminantsView rawIlluminantsView{};
        status(fj_test_illuminants_view(rawIlluminants, &rawIlluminantsView, nullptr));
        require(captured.at("products").size() == 72, "exact genuine print illuminant product membership");
        std::vector<std::string> product_ids;
        for (unsigned route = 0; route < 4; ++route)
            for (int mode = 0; mode < 3; ++mode)
                for (unsigned variant = 0; variant < 6; ++variant)
                    product_ids.emplace_back("print-illuminant/route/" + std::to_string(route) + "/mode/" + std::to_string(mode) + "/variant/" + std::to_string(variant));
        JuicerTests::Cohort product_rows("print_illuminant.recipe_tuple", product_ids);
        for (const auto& expected : captured.at("products")) {
            const unsigned route = expected.at("route");
            const int mode = expected.at("mode");
            const unsigned variant = expected.at("variant");
            auto case_event = product_rows.case_event("print-illuminant/route/" + std::to_string(route) + "/mode/" + std::to_string(mode) + "/variant/" + std::to_string(variant));
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
            const auto result = prepare_state(assets, p, product, error);
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
                require(p.enlIll == 3 && source.count == rawIlluminantsView.tungsten_kg3.count &&
                            std::equal(source.data, source.data + source.count, rawIlluminantsView.tungsten_kg3.data),
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
            case_event.passed();
        }

        JuicerTests::CaseEvent publication_event("cpp", "print-illuminant/publication");
        ParamSnapshot controls;
        controls.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
        controls.filmProfileKey = "kodak_portra_400";
        controls.printProfileKey = "kodak_portra_endura";
        controls.cameraAutoExposureEnabled = 0;
        controls.inputCompressionEnabled = 0;
        controls.outputGamutCompressionEnabled = 0;
        RenderInstance state;
        JuicerTestSupport::capture(state, controls);
        const auto original = JuicerTestSupport::admit(owner.borrow().assets, state);
        require(original.kind == FJ_ADMISSION_READY && original.state, "print illuminant publication baseline");
        const auto originalView = JuicerTestSupport::published_view(original);
        require(originalView.print.main_illuminant.count == 81, "baseline publishes all main illuminant samples");
        const auto originalHash = originalView.recipe_hash;
        const auto originalMain = words(originalView.print.main_illuminant.data, originalView.print.main_illuminant.count);
        const auto originalMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
        const auto counter = originalMetrics.build;
        controls.printUiYmcCc[0] = 2.5;
        JuicerTestSupport::capture(state, controls);
        status(fj_test_film_recipe_fault(10, 1));
        const auto admitted = JuicerTestSupport::admit(owner.borrow().assets, state);
        const auto admittedMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
        require(admitted.kind == FJ_ADMISSION_READY && admitted.state &&
                    JuicerTestSupport::PublicationObserver::backing(*admitted.state) != originalMetrics.backingIdentity,
                "actual main illuminant publication skips the legacy facade-only panic");
        require(admittedMetrics.build == counter + 1 && admittedMetrics.publishedIdentity == hash_params(controls) &&
                    admitted.state->build() == counter + 1 &&
                    originalView.recipe_hash == originalHash &&
                    words(originalView.print.main_illuminant.data, originalView.print.main_illuminant.count) == originalMain,
                "complete main illuminant publication advances once and preserves prior held values");
        const auto held = JuicerTestSupport::published_view(admitted);
        raw_main_illuminant_followup(held);
        const auto repeated = JuicerTestSupport::admit(owner.borrow().assets, state);
        require(repeated.kind == FJ_ADMISSION_READY && repeated.state &&
                    JuicerTestSupport::PublicationObserver::backing(*repeated.state) == admittedMetrics.backingIdentity &&
                    repeated.state->build() == counter + 1 &&
                    JuicerTestSupport::PublicationObserver::metrics(state).build == counter + 1,
                "unchanged main illuminant publication reuses after raw facade panic/retry");

        std::string diagnostic;
        controls.printPreflashExposure = 0;
        auto disabled = JuicerTestSupport::prepare_product(owner.borrow().assets, controls);
        PreparedPrintFacts disabledFacts;
        status(fj_test_film_recipe_fault(10, 1));
        require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                    owner.borrow().assets, disabled, PrintPreflashTest::frame_facts(controls), &disabledFacts, &inspect_prepared_print, diagnostic),
                diagnostic.c_str());
        require(disabledFacts.printHash != 0 && disabledFacts.preflashCount == 0, "disabled preflash omits constructor input");
        // The complete disabled input must leave the raw-constructor fault armed.
        consume_stage10(literalCases);
        publication_event.passed();
        controls.printPreflashExposure = 0.125;
        auto active = JuicerTestSupport::prepare_product(owner.borrow().assets, controls);
        PreparedPrintFacts activeFacts;
        status(fj_test_film_recipe_fault(10, 1));
        require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                    owner.borrow().assets, active, PrintPreflashTest::frame_facts(controls), &activeFacts, &inspect_prepared_print, diagnostic),
                diagnostic.c_str());
        require(activeFacts.printHash != 0 && activeFacts.preflashCount == 81,
                "active preflash input is supplied from the completed held state");
        consume_stage10(literalCases);
    }
} // namespace PrintIlluminantTest
