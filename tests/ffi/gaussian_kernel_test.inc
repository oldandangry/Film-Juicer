namespace GaussianTest {
    Json kernel_case(const std::string& name, int radius, float sigma) {
        Json weights = Json::array();
        if (radius > 0 && std::isfinite(sigma) && sigma > 0.0f) {
            JuicerCuda::Failure failure;
            auto kernel = JuicerGaussian::Kernel::acquire(radius, sigma, failure);
            require(kernel.has_value(), "actual Gaussian product construction");
            weights = words(kernel->weights().data(), kernel->weights().size());
        }
        return {{"name", name}, {"radius", radius}, {"sigma_bits", std::bit_cast<std::uint32_t>(sigma)}, {"count", weights.size()}, {"weights", weights}};
    }

    Json products(const std::filesystem::path& resources, const Json& cohort) {
        Json rows = Json::array();
        for (const auto& input : cohort.at("cases"))
            rows.push_back(kernel_case(input.at("name"), input.at("radius"), std::bit_cast<float>(input.at("sigma_bits").get<std::uint32_t>())));

        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        for (unsigned route = 0; route < 4; ++route) {
            ParamSnapshot p;
            p.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            p.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            p.grainControls.active = true;
            p.grainControls.sublayersActive = true;
            p.scannerLensBlurSigmaPx = 0.75;
            p.scannerUnsharpMask = {1.25, 0.5};
            p.dirCouplers.diffusionSizeUm = 20.0f;
            p.dirCouplers.diffusionTailUm = 200.0f;
            p.dirCouplers.diffusionTailWeight = 0.03f;
            auto product = JuicerTestSupport::prepare_product(p);
            FjPreparedStateView view{};
            FjErrorBuffer viewError{};
            status(product.view(view, viewError));
            const auto prefix = std::to_string(route) + "/";
            const auto& post = view.scanner_post_effects;
            rows.push_back(kernel_case(prefix + "scanner-lens", post.lens_blur_radius, post.lens_blur_sigma_px));
            rows.push_back(kernel_case(prefix + "scanner-unsharp", post.unsharp_radius, post.unsharp_sigma_px));
            rows.push_back(kernel_case(prefix + "scanner-glare", post.glare_blur_radius, post.glare_blur_sigma_px));
            for (float pixelSize : cohort.at("products").at("pixel_size_um").get<std::vector<float>>()) {
                const auto scale = prefix + std::to_string(pixelSize) + "/";
                Json grainFrame{{"render_extent", {0, 0, 64, 48}}, {"full_frame_extent", {0, 0, 64, 48}}, {"pixel_size_um", std::bit_cast<std::uint32_t>(pixelSize)}, {"frame_time", std::bit_cast<std::uint64_t>(12.0)}, {"frame_rate", std::bit_cast<std::uint64_t>(24.0)}, {"session_seed", 17}, {"clip_token", 23}};
                const auto rawGrain = GrainFrameTest::product_input(view, grainFrame);
                FjGrainFrameMath grain{};
                FjGrainFrameFailure grainFailure{};
                std::array<char, 256> grainText{};
                FjErrorBuffer grainError{grainText.data(), grainText.size(), 0};
                status(fj_test_grain_frame_prepare(&rawGrain, &grain, &grainFailure, &grainError));
                for (std::size_t c = 0; c < 3; ++c) {
                    const auto& correlation = grain.correlation[c];
                    rows.push_back(kernel_case(scale + "grain-correlation/" + std::to_string(c), correlation.radius, correlation.sigma_px));
                    for (std::size_t layer = 0; layer < 3; ++layer) {
                        const auto& cloud = grain.dye_cloud[layer][c];
                        rows.push_back(kernel_case(scale + "grain-cloud/" + std::to_string(layer) + "/" + std::to_string(c), cloud.radius, cloud.sigma_px));
                    }
                }
                Json dirInput{{"active", view.dir_recipe.mode != 0}, {"size", std::bit_cast<std::uint32_t>(view.dir_recipe.diffusion_size_um)}, {"tail", std::bit_cast<std::uint32_t>(view.dir_recipe.diffusion_tail_um)}, {"weight", std::bit_cast<std::uint32_t>(view.dir_recipe.diffusion_tail_weight)}, {"pixel", std::bit_cast<std::uint32_t>(pixelSize)}, {"render", {0, 0, 64, 48}}, {"full", {0, 0, 64, 48}}, {"hash", view.dir_recipe.hash}};
                const auto rawDir = DirFrameTest::raw_input(dirInput);
                FjDirFrameMath dir{};
                FjDirFrameFailure dirFailure{};
                std::array<char, 256> dirText{};
                FjErrorBuffer dirError{dirText.data(), dirText.size(), 0};
                status(fj_test_dir_frame_prepare(&rawDir, &dir, &dirFailure, &dirError));
                for (std::size_t c = 0; c < dir.component_count; ++c) {
                    const auto& component = dir.components[c];
                    if (component.reference_operator == FJ_DIR_OPERATOR_FIR_REFLECT)
                        rows.push_back(kernel_case(scale + "fir-dir/" + std::to_string(c), component.radius, component.sigma_px));
                }
            }
        }
        return rows;
    }
    void values(const Json& rows) {
        static_assert(!std::is_copy_constructible_v<JuicerGaussian::Kernel>);
        static_assert(std::is_nothrow_move_constructible_v<JuicerGaussian::Kernel>);
        require(rows.size() == 188, "complete Gaussian fixture membership");
        for (const auto& row : rows) {
            const int radius = row.at("radius");
            const auto bits = row.at("sigma_bits").get<std::uint32_t>();
            const float sigma = std::bit_cast<float>(bits);
            if (radius <= 0 || !std::isfinite(sigma) || sigma <= 0.0f)
                continue;
            const FjGaussianKernelInput input{radius, sigma};
            for (auto acquire : {fj_legacy_gaussian_kernel_acquire, fj_test_gaussian_kernel_acquire}) {
                FjGaussianKernelOwner* owner = nullptr;
                FjGaussianKernelFailure failure{99};
                status(acquire(&input, &owner, &failure, nullptr));
                require(owner && failure.reason == 0, "Gaussian complete unique owner");
                for (auto view : {fj_legacy_gaussian_kernel_view, fj_test_gaussian_kernel_view}) {
                    FjGaussianKernelView v{};
                    status(view(owner, &v, &failure, nullptr));
                    require(v.radius == radius && std::bit_cast<std::uint32_t>(v.sigma_px) == bits &&
                                v.weights.count == row.at("count").get<std::size_t>() && failure.reason == 0,
                            "exact Gaussian metadata");
                    require(words(v.weights.data, v.weights.count) == row.at("weights"), "all independent Gaussian words through both checked readers");
                }
                status(fj_legacy_gaussian_kernel_release(owner, nullptr));
            }
            JuicerCuda::Failure failure;
            auto kernel = JuicerGaussian::Kernel::acquire(radius, sigma, failure);
            require(kernel.has_value(), "native Gaussian hold acquired");
            auto moved = std::move(*kernel);
            require(kernel->weights().empty(), "moved Gaussian view cannot expose released ownership");
            require(words(moved.weights().data(), moved.weights().size()) == row.at("weights"), "all frozen native bridge words");
            const auto expected = row.at("weights").get<std::vector<std::uint32_t>>();
            require(fj_test_gaussian_c_call(&input, expected.data(), expected.size()) != 0, "compiled C11 full Gaussian arrays and metadata");
            if (radius == 2048 || (radius == 3 && sigma == 1.0f)) {
                auto changed = expected;
                for (std::size_t index : {std::size_t{0}, expected.size() / 2, expected.size() - 1}) {
                    changed[index] ^= 1;
                    require(fj_test_gaussian_c_call(&input, changed.data(), changed.size()) == 0, "C11 reader rejects first, center and final word corruption");
                    changed[index] ^= 1;
                }
            }
        }
        require(fj_test_gaussian_kernel_live_owners() == 0, "Gaussian values release every owner");
    }
    void callback_failure_diagnostics() {
        struct Reply {
            std::string_view diagnostic;
            bool success = false;
            std::array<float, 7> weights{};
        } reply;
        const FjHostPreparation preparation{
            &reply,
            [](void*, const FjScannerLutInput*, FjScannerLutView*, FjScannerLutFailure*, FjErrorBuffer*) -> FjStatus {
                return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
            },
            [](void* user, const FjGaussianKernelInput* input, FjGaussianKernelView* view, FjGaussianKernelFailure* failure, FjErrorBuffer* error) -> FjStatus {
                const auto& reply = *static_cast<Reply*>(user);
                *view = reply.success ? FjGaussianKernelView{input->radius, input->sigma_px, {reply.weights.data(), reply.weights.size()}} : FjGaussianKernelView{};
                failure->reason = reply.success ? FJ_GAUSSIAN_KERNEL_NONE : FJ_GAUSSIAN_KERNEL_ALLOCATION;
                std::copy(reply.diagnostic.begin(), reply.diagnostic.end(), error->data);
                error->length = reply.diagnostic.size();
                error->data[error->length] = '\0';
                return {reply.success ? FJ_STATUS_SUCCESS : FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0};
            }};
        const FjGaussianKernelInput input{3, 1.0f};
        for (const char* diagnostic : {"Gaussian kernel allocation failed", "Gaussian kernel candidate storage allocation failed", "Gaussian coefficient storage is unavailable", ""}) {
            reply.diagnostic = diagnostic;
            FjGaussianKernelView view{};
            JuicerCuda::Failure failure;
            require(!JuicerCuda::prepare_gaussian_kernel(preparation, input, view, failure), "typed Gaussian allocation callback rejects preparation");
            status(failure.status, FJ_STATUS_ALLOCATION_FAILURE);
            require(failure.diagnostic == diagnostic, "Gaussian allocation category is independent of forwarded diagnostic wording");
            require(view.radius == 0 && std::bit_cast<std::uint32_t>(view.sigma_px) == 0 && !view.weights.data && !view.weights.count, "Gaussian callback failure leaves no published view");
        }
        reply.success = true;
        reply.diagnostic = "unexpected success diagnostic";
        FjGaussianKernelView view{};
        JuicerCuda::Failure failure;
        require(!JuicerCuda::prepare_gaussian_kernel(preparation, input, view, failure), "successful Gaussian transport still requires an empty diagnostic");
        status(failure.status, FJ_STATUS_PREPARATION_FAILURE);
        require(failure.diagnostic == "Gaussian kernel projection invalid", "nonempty Gaussian success retains transport rejection");
        reply.diagnostic = "";
        require(JuicerCuda::prepare_gaussian_kernel(preparation, input, view, failure), "empty Gaussian success diagnostic is admitted");
        require(failure.diagnostic.empty(), "successful Gaussian transport clears the previous rejection diagnostic");
        require(view.weights.data == reply.weights.data() && view.weights.count == reply.weights.size(), "Gaussian transport preserves the literal borrowed span");
    }
    void boundary() {
        callback_failure_diagnostics();
        const FjGaussianKernelInput input{3, 1.0f};
        FjGaussianKernelFailure failure{99};
        FjGaussianKernelOwner* owner = nullptr;
        for (auto acquire : {fj_legacy_gaussian_kernel_acquire, fj_test_gaussian_kernel_acquire}) {
            for (auto probe : {FjGaussianKernelInput{0, 0}, FjGaussianKernelInput{2049, 1}, FjGaussianKernelInput{-1, NAN}, FjGaussianKernelInput{1, 0}, FjGaussianKernelInput{1, -0.0f}, FjGaussianKernelInput{1, -1}, FjGaussianKernelInput{1, INFINITY}, FjGaussianKernelInput{1, NAN}}) {
                const auto reason = probe.radius < 1 || probe.radius > 2048 ? 2u : 3u;
                owner = reinterpret_cast<FjGaussianKernelOwner*>(std::uintptr_t{1});
                char text[128] = "seed";
                FjErrorBuffer error{text, sizeof(text), 99};
                status(acquire(&probe, &owner, &failure, &error), 3);
                require(!owner && failure.reason == reason, "ordered Gaussian domain rejection");
                require(std::string_view(text) == (reason == 2 ? "Gaussian kernel radius outside supported range" : "invalid Gaussian kernel sigma"), "exact Gaussian domain diagnostic");
            }
            for (unsigned missing = 0; missing < 5; ++missing) {
                owner = reinterpret_cast<FjGaussianKernelOwner*>(std::uintptr_t{1});
                failure.reason = 99;
                FjErrorBuffer malformed{nullptr, 1, 99};
                const auto* in = missing == 0 ? nullptr : &input;
                auto* out = missing == 1 ? nullptr : &owner;
                auto* fail = missing == 2 ? nullptr : &failure;
                auto* error = missing == 3 ? &malformed : nullptr;
                if (missing == 4)
                    in = reinterpret_cast<const FjGaussianKernelInput*>(std::uintptr_t{1});
                status(acquire(in, out, fail, error), 2);
                require(!out || !owner, "Gaussian valid owner peer clears independently");
                require(!fail || failure.reason == 1, "Gaussian valid failure peer clears independently");
            }
            FjErrorBuffer zero{reinterpret_cast<char*>(std::uintptr_t{1}), 0, 99};
            status(acquire(&input, &owner, &failure, &zero));
            require(zero.length == 0, "zero capacity never accesses diagnostic backing");
            FjGaussianKernelView v{99, 99, {reinterpret_cast<const float*>(std::uintptr_t{1}), 99}};
            status(fj_test_gaussian_kernel_view(nullptr, &v, &failure, nullptr), 2);
            require(v.radius == 0 && std::bit_cast<std::uint32_t>(v.sigma_px) == 0 && !v.weights.data && !v.weights.count && failure.reason == 1, "entire Gaussian view independently zeroed");
            FjErrorBuffer malformed{nullptr, 1, 99};
            status(fj_test_gaussian_kernel_release(owner, &malformed), 2);
            require(fj_test_gaussian_kernel_live_owners() == 0, "malformed release diagnostics consume actual owner");
        }
        for (unsigned operation = 0; operation < 3; ++operation) {
            if (operation != 0)
                status(fj_legacy_gaussian_kernel_acquire(&input, &owner, &failure, nullptr));
            status(fj_test_gaussian_kernel_fault(operation, 1));
            char text[128]{};
            FjErrorBuffer error{text, sizeof(text), 0};
            if (operation == 0)
                status(fj_legacy_gaussian_kernel_acquire(&input, &owner, &failure, &error), 8);
            else if (operation == 1) {
                FjGaussianKernelView v{99, 99, {nullptr, 99}};
                status(fj_legacy_gaussian_kernel_view(owner, &v, &failure, &error), 8);
                require(!v.weights.data && !v.weights.count && v.radius == 0 && v.sigma_px == 0 && failure.reason == 5, "view panic clears every field");
                status(fj_legacy_gaussian_kernel_release(owner, nullptr));
            } else
                status(fj_legacy_gaussian_kernel_release(owner, &error), 8);
            require(std::string_view(text) == (operation == 2 ? "Gaussian kernel release panicked" : "Gaussian kernel preparation panicked"), "Gaussian panic diagnostic");
            require(fj_test_gaussian_kernel_live_owners() == 0, "panic consumes provisional/owned holds");
        }
        std::array<char, 128> diagnosticText{};
        FjErrorBuffer diagnosticError{diagnosticText.data(), diagnosticText.size(), 0};
        status(fj_legacy_gaussian_kernel_acquire(nullptr, &owner, &failure, &diagnosticError), 2);
        require(!owner && failure.reason == 1 && std::string_view(diagnosticText.data()) == "invalid Gaussian kernel arguments", "exact raw Gaussian failure diagnostic");
        status(fj_test_gaussian_kernel_fault(0, 2));
        status(fj_test_gaussian_kernel_acquire(&input, &owner, &failure, &diagnosticError), 4);
        require(std::string_view(diagnosticText.data()) == "Gaussian kernel allocation failed", "exact Gaussian reservation failure diagnostic");
        require(!owner && failure.reason == 4, "actual coefficient reservation failure");
        for (unsigned fault = 1; fault <= 4; ++fault) {
            status(fj_test_gaussian_kernel_fault(1, fault));
            JuicerCuda::Failure rejected;
            require(!JuicerGaussian::Kernel::acquire(3, 1, rejected), "invalid Gaussian projection rejected before reading weights");
            require(rejected.status.category == (fault == 1 ? 8u : 3u), "Gaussian projection failure origin");
            require(rejected.diagnostic == (fault == 1 ? "Gaussian kernel preparation panicked" : "Gaussian kernel projection invalid"), "Gaussian projection exact diagnostic");
            require(fj_test_gaussian_kernel_live_owners() == 0, "native rejection releases original owner");
        }
        for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{5}, std::size_t{128}}) {
            std::array<char, 128> text{};
            text.fill('Q');
            FjErrorBuffer error{capacity ? text.data() : reinterpret_cast<char*>(std::uintptr_t{1}), capacity, 99};
            const FjGaussianKernelInput invalid{0, NAN};
            status(fj_legacy_gaussian_kernel_acquire(&invalid, &owner, &failure, &error), 3);
            const auto expected = capacity ? std::min(capacity - 1, std::string_view("Gaussian kernel radius outside supported range").size()) : 0;
            require(!owner && failure.reason == 2 && error.length == expected, "bounded Gaussian diagnostic length and ordered failure");
            if (capacity)
                require(text[expected] == '\0', "bounded Gaussian diagnostic terminator");
            for (std::size_t i = capacity; i < text.size(); ++i)
                require(text[i] == 'Q', "diagnostic cannot overwrite capacity");
        }
        status(fj_legacy_gaussian_kernel_acquire(&input, &owner, &failure, nullptr));
        std::array<std::exception_ptr, 2> errors{};
        const auto reader = [&](unsigned index) {
            try {
                if (index == 0)
                    status(fj_test_gaussian_kernel_fault(1, 1));
                FjGaussianKernelView view{};
                FjGaussianKernelFailure failed{};
                status(fj_test_gaussian_kernel_view(owner, &view, &failed, nullptr), index == 0 ? 8u : 0u);
                require(index == 0 ? failed.reason == 5 && !view.weights.data : failed.reason == 0 && view.radius == 3 && view.weights.count == 7 && view.sigma_px == 1.0f, "concurrent immutable views and thread-local one-shot panic");
            } catch (...) {
                errors[index] = std::current_exception();
            }
        };
        {
            std::jthread first(reader, 0), second(reader, 1);
            first.join();
            second.join();
        }
        status(fj_legacy_gaussian_kernel_release(owner, nullptr));
        for (const auto& error : errors)
            if (error)
                std::rethrow_exception(error);
        {
            JuicerCuda::Failure rejected;
            auto first = JuicerGaussian::Kernel::acquire(3, 1, rejected);
            auto second = JuicerGaussian::Kernel::acquire(7, 1.234567f, rejected);
            require(first && second && fj_test_gaussian_kernel_live_owners() == 2, "independent native Gaussian holds");
            *second = std::move(*first);
            require(first->weights().empty() && second->weights().size() == 7 && fj_test_gaussian_kernel_live_owners() == 1, "move assignment releases old matching owner once");
        }
        require(fj_test_gaussian_kernel_live_owners() == 0, "move/unwind scope releases Gaussian holds");
        status(fj_test_gaussian_kernel_fault(3, 0), 2);
        status(fj_test_gaussian_kernel_fault(0, 3), 2);
        status(fj_test_gaussian_kernel_fault(1, 5), 2);
        status(fj_test_gaussian_kernel_fault(2, 2), 2);
        status(fj_legacy_gaussian_kernel_release(nullptr, nullptr));
        status(fj_test_gaussian_kernel_release(nullptr, nullptr));
        JuicerCuda::Failure rejected;
        require(JuicerGaussian::Kernel::acquire(2048, 512, rejected).has_value(), "Gaussian recovery reaches upper bound");
        require(fj_test_gaussian_kernel_live_owners() == 0, "Gaussian boundary owner balance");
    }
} // namespace GaussianTest
