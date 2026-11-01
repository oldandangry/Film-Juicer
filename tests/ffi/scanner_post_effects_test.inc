namespace ScannerPostTest {
    FjScannerPostEffectsInput wire(const Json& in) {
        float values[6];
        copy_words(in.at("values"), values);
        return {in.at("route").get<std::uint32_t>(), in.at("glare_active").get<bool>() ? 1u : 0u, values[0], values[1], values[2], values[3], values[4], values[5]};
    }
    Json project(const FjScannerPostEffectsDescriptor& d) {
        const float values[]{d.glare_percent, d.glare_roughness, d.glare_blur_sigma_px, d.lens_blur_sigma_px, d.unsharp_sigma_px, d.unsharp_amount};
        return {{"route", d.route}, {"glare_active", d.glare_active != 0}, {"values", words(values, 6)}, {"radii", {d.glare_blur_radius, d.lens_blur_radius, d.unsharp_radius}}, {"schema", d.schema}, {"hash", d.hash}};
    }
    template <typename T>
    bool zero(const T& value) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
        return std::all_of(bytes, bytes + sizeof(value), [](unsigned char byte) {
            return byte == 0;
        });
    }
    void values(const Json& rows) {
        for (const auto& row : rows) {
            auto input = wire(row.at("input"));
            for (auto describe : {fj_legacy_scanner_post_describe, fj_test_scanner_post_describe}) {
                FjScannerPostEffectsDescriptor d;
                FjScannerPostEffectsFailure failure;
                std::memset(&d, 0xa5, sizeof(d));
                std::memset(&failure, 0xa5, sizeof(failure));
                char text[256] = "seed";
                FjErrorBuffer error{text, sizeof(text), 99};
                const auto s = describe(&input, &d, &failure, &error);
                const auto& expected = row.at("native");
                if (input.route > 3) {
                    status(s, FJ_STATUS_UNSUPPORTED_INPUT);
                    require(failure.reason == 1 && failure.lane == 0 && zero(d) && zero(failure.rejected), "post raw route failure/clearing");
                    require(std::string_view(text) == "invalid scanner post-effects arguments", "post raw diagnostic");
                } else if (expected.at("success").get<bool>()) {
                    status(s);
                    require(zero(failure) && error.length == 0 && text[0] == '\0', "post success envelope");
                    require(project(d) == expected.at("descriptor"), "complete independent post descriptor fields/hash");
                } else {
                    status(s, FJ_STATUS_PREPARATION_FAILURE);
                    require(zero(d) && failure.reason == 2, "post value result entirely zero");
                    require(project(failure.rejected) == expected.at("descriptor"), "complete independent rejected attempt");
                    const auto& bits = expected.at("descriptor").at("values");
                    unsigned lane = 0;
                    while (lane < 6) {
                        const auto v = std::bit_cast<float>(bits[lane].get<std::uint32_t>());
                        if (!std::isfinite(v) || v < 0.0f)
                            break;
                        ++lane;
                    }
                    require(failure.lane == lane + 1 && std::string(text) == expected.at("diagnostic").get<std::string>(), "post first bad value and exact diagnostic");
                }
            }
        }
    }
    void boundary(const Json& rows) {
        values(rows);
        auto input = wire(rows.at(100).at("input")); // print active; 100 cases/route
        require(input.route == 1, "post active print fixture selection");
        FjScannerPostEffectsDescriptor d;
        FjScannerPostEffectsFailure failure;
        const auto poison = [&]() {
            std::memset(&d, 0xa5, sizeof(d));
            std::memset(&failure, 0xa5, sizeof(failure));
        };
        for (unsigned mode = 0; mode < 7; ++mode) {
            poison();
            auto in = input;
            auto* argument = &in;
            auto* output = &d;
            auto* details = &failure;
            char text[256]{};
            FjErrorBuffer error{text, sizeof(text), 99};
            if (mode == 0)
                argument = nullptr;
            if (mode == 1)
                output = nullptr;
            if (mode == 2)
                details = nullptr;
            if (mode == 3)
                in.route = 4;
            if (mode == 4)
                in.glare_active = 2;
            if (mode == 5)
                error = {nullptr, 1, 99};
            if (mode == 6)
                argument = reinterpret_cast<FjScannerPostEffectsInput*>(reinterpret_cast<char*>(&in) + 1);
            status(fj_test_scanner_post_describe(argument, output, details, &error), FJ_STATUS_UNSUPPORTED_INPUT);
            if (output)
                require(zero(d), "post independent result clearing");
            if (details)
                require(failure.reason == 1 && failure.lane == 0 && zero(failure.rejected), "post independent failure clearing");
        }
        FjErrorBuffer empty{reinterpret_cast<char*>(1), 0, 99};
        status(fj_test_scanner_post_describe(&input, &d, &failure, &empty));
        require(empty.length == 0 && zero(failure), "post zero-capacity diagnostic");
        status(fj_test_scanner_post_describe(&input, &d, &failure, nullptr));
        for (unsigned lane = 1; lane <= 3; ++lane) {
            auto in = input;
            const float threshold = 2147483648.0f / 3.0f;
            if (lane == 1)
                in.glare_blur_sigma_px = threshold;
            if (lane == 2)
                in.lens_blur_sigma_px = threshold;
            if (lane == 3) {
                in.unsharp_sigma_px = threshold;
                in.unsharp_amount = 0.0f;
            }
            in.glare_roughness = NAN; // conversion precedes later value validation
            poison();
            char text[256]{};
            FjErrorBuffer error{text, sizeof(text), 0};
            status(fj_test_scanner_post_describe(&in, &d, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
            require(failure.reason == 3 && failure.lane == lane && zero(d) && zero(failure.rejected), "post bounded radius range/order envelope");
            require(std::string_view(text) == "scanner post-effects Gaussian radius outside int32 range", "post exact radius diagnostic");
        }
        for (unsigned fault : {1u, 2u}) {
            status(fj_test_scanner_post_fault(fault));
            auto invalid = input;
            invalid.route = 4;
            status(fj_test_scanner_post_describe(&invalid, &d, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            invalid = input;
            invalid.glare_roughness = NAN;
            status(fj_test_scanner_post_describe(&invalid, &d, &failure, nullptr), FJ_STATUS_PREPARATION_FAILURE);
            require(failure.reason == 2, "post value failure before fault consumption");
            invalid = input;
            invalid.unsharp_sigma_px = 2147483648.0f / 3.0f;
            status(fj_test_scanner_post_describe(&invalid, &d, &failure, nullptr), FJ_STATUS_PREPARATION_FAILURE);
            require(failure.reason == 3, "post radius failure before fault consumption");
            char text[256]{};
            FjErrorBuffer error{text, sizeof(text), 0};
            status(fj_test_scanner_post_describe(&input, &d, &failure, &error), fault == 1 ? FJ_STATUS_PREPARATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE);
            require(zero(d) && failure.reason == (fault == 1 ? 4u : 5u) && failure.lane == 0, "post one-shot hash/panic envelope");
            if (fault == 1) {
                auto expected = rows.at(100).at("native").at("descriptor");
                expected["hash"] = 0;
                require(project(failure.rejected) == expected && error.length == 0, "post active zero hash attempted fields/empty diagnostic");
            } else
                require(zero(failure.rejected) && std::string_view(text) == "scanner post-effects preparation panicked", "post contained panic");
            status(fj_test_scanner_post_describe(&input, &d, &failure, nullptr));
        }
        status(fj_test_scanner_post_fault(1));
        auto inactive = wire(rows.at(1).at("input"));
        status(fj_test_scanner_post_describe(&inactive, &d, &failure, nullptr));
        require(d.hash == 0 && zero(failure), "post inactive legitimate zero consumes hash fault");
        status(fj_test_scanner_post_describe(&input, &d, &failure, nullptr));
        status(fj_test_scanner_post_fault(3), FJ_STATUS_UNSUPPORTED_INPUT);
    }
    void lifetime(const Json& rows) {
        FjScannerPostEffectsDescriptor held{};
        FjScannerPostEffectsFailure failure{};
        {
            const auto input = wire(rows.at(100).at("input"));
            status(fj_test_scanner_post_describe(&input, &held, &failure, nullptr));
        }
        status(fj_test_scanner_post_fault(1));
        std::thread other([&] {
            const auto input = wire(rows.at(100).at("input"));
            FjScannerPostEffectsDescriptor result{};
            FjScannerPostEffectsFailure rejected{};
            status(fj_test_scanner_post_describe(&input, &result, &rejected, nullptr));
            require(project(result) == project(held), "post owned result survives input expiry and another thread");
        });
        other.join();
        const auto input = wire(rows.at(100).at("input"));
        FjScannerPostEffectsDescriptor result{};
        status(fj_test_scanner_post_describe(&input, &result, &failure, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        require(failure.reason == 4 && project(held) == rows.at(100).at("native").at("descriptor"), "post fault stays thread-local and cannot mutate held result");
    }
    void callers(const std::filesystem::path& resources, const Json& rows) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        for (unsigned route = 0; route < 4; ++route) {
            const auto rowIndex = static_cast<Json::size_type>(route) * 100;
            const auto selected = static_cast<Spektrafilm::ScanRoute>(route);
            auto controls = FilmReferenceTest::controls(1, selected, 0);
            controls.dirCouplers.active = false;
            controls.grainControls.active = false;
            const auto source = wire(rows.at(rowIndex).at("input"));
            controls.glarePercent = source.glare_percent;
            controls.glareRoughness = source.glare_roughness;
            controls.glareBlurSigmaPx = source.glare_blur_sigma_px;
            controls.scannerLensBlurSigmaPx = source.lens_blur_sigma_px;
            controls.scannerUnsharpMask = {source.unsharp_sigma_px, source.unsharp_amount};
            controls.glareActive = source.glare_active != 0;
            auto product = JuicerTestSupport::prepare_product(controls);
            FjPreparedStateView view{};
            FjErrorBuffer error{};
            status(product.view(view, error));
            const auto initial = view.scanner_post_effects;
            require(project(initial) == rows.at(rowIndex).at("native").at("descriptor"), "actual route schedule complete post product");
            controls.scannerLensBlurSigmaPx = 2.0;
            auto changedProduct = JuicerTestSupport::prepare_product(controls);
            FjPreparedStateView changed{};
            status(changedProduct.view(changed, error));
            require(changed.scanner_post_effects.hash != initial.hash && changed.scanner_lut.hash == view.scanner_lut.hash, "post-only change preserves actual LUT descriptor dependency");
            controls.scannerLensBlurSigmaPx = source.lens_blur_sigma_px;
            auto warmProduct = JuicerTestSupport::prepare_product(controls);
            FjPreparedStateView warm{};
            status(warmProduct.view(warm, error));
            require(project(warm.scanner_post_effects) == project(initial), "warm route constructor preserves complete post identity");
            status(fj_test_scanner_post_fault(2));
            status(fj_test_scanner_lut_descriptor_fault(1));
            const auto prepare = [&](std::optional<PreparedState>& out, std::array<char, 2048>& text) {
                FjErrorBuffer prepareError{text.data(), text.size(), 0};
                return prepare_state(
                    JuicerProcess::root().assets().borrow_rust_assets(),
                    controls,
                    out,
                    prepareError);
            };
            bool early = false;
            std::optional<PreparedState> failedProduct;
            std::array<char, 2048> failureText{};
            const FjStatus lutFailure = prepare(failedProduct, failureText);
            early = lutFailure.category != FJ_STATUS_SUCCESS && !failedProduct;
            require(early && std::string_view(failureText.data()).find("scanner") != std::string_view::npos, "early LUT failure reaches route");
            require(early, "earlier route LUT failure preempts post constructor");
            bool post = false;
            failureText.fill('\0');
            const FjStatus postFailure = prepare(failedProduct, failureText);
            post = postFailure.category == FJ_STATUS_INTERNAL_FAILURE && !failedProduct;
            require(post && std::string_view(failureText.data()) == "asset boundary panic", "actual post failure retains typed timing and leaves later frame stages unbuilt");
            require(post, "every warm route invocation visits post constructor");
            auto recoveredProduct = JuicerTestSupport::prepare_product(controls);
            FjPreparedStateView recovered{};
            status(recoveredProduct.view(recovered, error));
            require(project(recovered.scanner_post_effects) == project(initial), "actual route recovers after one-shot failure");
        }
    }
} // namespace ScannerPostTest
