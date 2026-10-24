namespace ScannerPostTest {
    Json project(const Scanner::ScannerPostEffectsDescriptor& d) {
        const float values[]{d.glarePercent, d.glareRoughness, d.glareBlurSigmaPx, d.lensBlurSigmaPx, d.unsharpSigmaPx, d.unsharpAmount};
        return {{"route", static_cast<unsigned>(d.route)}, {"glare_active", d.glareActive}, {"values", words(values, 6)}, {"radii", {d.glareBlurRadius, d.lensBlurRadius, d.unsharpRadius}}, {"schema", d.schemaVersion}, {"hash", d.hash}};
    }
    ScannerOutputRecipe recipe(const Json& in) {
        ScannerOutputRecipe r;
        r.route = static_cast<Spektrafilm::ScanRoute>(in.at("route").get<unsigned>());
        r.glareActive = in.at("glare_active");
        float values[6];
        copy_words(in.at("values"), values);
        r.glarePercent = values[0];
        r.glareRoughness = values[1];
        r.glareBlurSigmaPx = values[2];
        r.lensBlurSigmaPx = values[3];
        r.unsharpSigmaPx = values[4];
        r.unsharpAmount = values[5];
        return r;
    }
    Json native(const Json& in) {
        auto r = recipe(in);
        Scanner::ScannerPostEffectsDescriptor d;
        d.hash = 99;
        std::string diagnostic = "seed";
        bool success = false;
        bool exception = false;
        try {
            success = Scanner::build_scanner_post_effects_descriptor(r, d, diagnostic);
        } catch (const std::out_of_range&) {
            exception = true;
        }
        return {{"success", success}, {"exception", exception}, {"diagnostic", diagnostic}, {"descriptor", project(d)}};
    }
    FjScannerPostEffectsInput wire(const Json& in) {
        const auto r = recipe(in);
        return {static_cast<std::uint32_t>(r.route), std::uint32_t(r.glareActive), r.glarePercent, r.glareRoughness, r.glareBlurSigmaPx, r.lensBlurSigmaPx, r.unsharpSigmaPx, r.unsharpAmount};
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
    void native_values(const Json& rows) {
        for (const auto& row : rows)
            require(native(row.at("input")) == row.at("native"), "post complete native fields/failure/exception parity");
    }
    thread_local unsigned projectionFault = 0;
    void corrupt(FjStatus& status, FjScannerPostEffectsDescriptor& d, FjScannerPostEffectsFailure& failure, FjErrorBuffer& error) {
        const auto fault = std::exchange(projectionFault, 0u);
        switch (fault) {
            case 1:
                status.api = FJ_API_CUDA_RUNTIME;
                break;
            case 2:
                status.native_code = 7;
                break;
            case 3:
                status.category = FJ_STATUS_ALLOCATION_FAILURE;
                break;
            case 4:
                failure.reason = 6;
                break;
            case 5:
                failure.lane = 1;
                break;
            case 6:
                failure.rejected.schema = 2;
                break;
            case 7:
                d.route ^= 1;
                break;
            case 8:
                d.schema = 3;
                break;
            case 9:
                d.glare_active = 2;
                break;
            case 10:
                d.lens_blur_radius = 76;
                break;
            case 11:
                d.unsharp_radius = -1;
                break;
            case 12:
                d.hash = 0;
                break;
            case 13:
                d.glare_percent = NAN;
                break;
            case 14:
                d.unsharp_sigma_px = 0.0f;
                break;
            case 15:
                error.length = error.capacity;
                break;
            case 16:
                error.data = nullptr;
                break;
            case 17:
                failure.rejected.hash = 1;
                break;
            default:
                break;
        }
    }
    void adapters(const Json& rows) {
        struct Reset {
            ~Reset() {
                JuicerScanner::Test::set_post_projection(nullptr);
            }
        } reset;
        JuicerScanner::Test::set_post_projection(corrupt);
        for (unsigned fault = 1; fault <= 17; ++fault) {
            auto input = rows.at(100).at("input");
            if (fault == 17)
                input["values"][1] = 0x7fc12345u;
            projectionFault = fault;
            bool caught = false;
            try {
                (void)native(input);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                caught = true;
                require(error.failure.status.category == FJ_STATUS_PREPARATION_FAILURE && error.failure.status.api == 0 && error.failure.status.native_code == 0 && error.failure.diagnostic == "scanner post-effects projection invalid", "post corrupt envelope typed rejection");
            }
            require(caught && projectionFault == 0, "post corrupt envelope consumed");
            require(native(rows.at(100).at("input")) == rows.at(100).at("native"), "post native projection recovery");
        }
        for (unsigned lane = 1; lane <= 3; ++lane) {
            auto input = rows.at(100).at("input");
            input["values"][lane + 1] = std::bit_cast<std::uint32_t>(2147483648.0f / 3.0f);
            if (lane == 3)
                input["values"][5] = 0;
            const auto actual = native(input);
            Scanner::ScannerPostEffectsDescriptor empty;
            empty.route = Spektrafilm::ScanRoute::NegativePrintScan;
            require(actual.at("success") == false && actual.at("descriptor") == project(empty) && actual.at("diagnostic") == "scanner post-effects Gaussian radius outside int32 range", "post native radius failure keeps route/default carrier");
        }
        status(fj_test_scanner_post_fault(1));
        auto actual = native(rows.at(100).at("input"));
        auto expected = rows.at(100).at("native");
        expected["success"] = false;
        expected["descriptor"]["hash"] = 0;
        require(actual == expected, "post native zero-hash failed attempt and empty diagnostic");
        status(fj_test_scanner_post_fault(2));
        bool panic = false;
        try {
            (void)native(rows.at(100).at("input"));
        } catch (const JuicerCuda::ExecutionFailure& error) {
            panic = true;
            require(error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE && error.failure.status.api == 0 && error.failure.status.native_code == 0 && error.failure.diagnostic == "scanner post-effects preparation panicked", "post native panic typed status");
        }
        require(panic, "post native panic containment");
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
        JuicerProcess::root().ensure_bootstrap();
        for (unsigned route = 0; route < 4; ++route) {
            const auto rowIndex = static_cast<Json::size_type>(route) * 100;
            const auto selected = static_cast<Spektrafilm::ScanRoute>(route);
            const bool print = Spektrafilm::scan_route_is_print(selected);
            auto controls = FilmReferenceTest::controls(1, selected, 0);
            controls.dirCouplers.active = false;
            controls.grainControls.active = false;
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            require(print ? build_print_render_state_product(controls, product, diagnostic)
                          : build_direct_render_state_product(controls, product, diagnostic),
                    diagnostic.c_str());
            const auto source = recipe(rows.at(rowIndex).at("input"));
            auto& output = product.recipe.scannerOutput;
            output.route = source.route;
            output.glareActive = source.glareActive;
            output.glarePercent = source.glarePercent;
            output.glareRoughness = source.glareRoughness;
            output.glareBlurSigmaPx = source.glareBlurSigmaPx;
            output.lensBlurSigmaPx = source.lensBlurSigmaPx;
            output.unsharpSigmaPx = source.unsharpSigmaPx;
            output.unsharpAmount = source.unsharpAmount;
            const JuicerCuda::FrameRect bounds{0, 0, 16, 8};
            const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(bounds, bounds, product.recipe.filmRaw.autoExposureMethod);
            const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
            const std::optional<ScatterHalationFrameDescriptor> halation;
            const Spektrafilm::FilmJuicerEffectsGeometry geometry{};
            const JuicerCuda::ExecutionFrame frame{bounds, bounds, bounds, nullptr, nullptr, nullptr, 0, 0, 4, nullptr, diffusion, halation, geometry, 100.0f, 0.0, 24.0, 1, 1, meter, false, false};
            const auto describe = [&]() {
                return JuicerCuda::describe_execution(product.recipe, product.payload, frame);
            };
            const auto initial = describe();
            require(project(initial.post) == rows.at(rowIndex).at("native").at("descriptor"), "actual route schedule complete post product");
            output.lensBlurSigmaPx = 2.0f;
            const auto changed = describe();
            require(changed.post.hash != initial.post.hash && changed.scanner.hash == initial.scanner.hash, "post-only change preserves actual LUT descriptor dependency");
            output.lensBlurSigmaPx = source.lensBlurSigmaPx;
            require(project(describe().post) == project(initial.post), "warm route constructor preserves complete post identity");
            status(fj_test_scanner_post_fault(2));
            status(fj_test_scanner_lut_descriptor_fault(1));
            bool early = false;
            try {
                (void)describe();
            } catch (const JuicerCuda::ExecutionFailure& error) {
                early = true;
                require(error.failure.diagnostic.find("scanner") != std::string::npos, "early LUT failure reaches route");
            }
            require(early, "earlier route LUT failure preempts post constructor");
            bool post = false;
            try {
                (void)describe();
            } catch (const JuicerCuda::ExecutionFailure& error) {
                post = true;
                require(error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE && error.failure.diagnostic == "scanner post-effects preparation panicked", "actual post failure retains typed timing and leaves later frame stages unbuilt");
            }
            require(post, "every warm route invocation visits post constructor");
            require(project(describe().post) == project(initial.post), "actual route recovers after one-shot failure");
        }
    }
} // namespace ScannerPostTest
