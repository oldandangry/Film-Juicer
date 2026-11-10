#include "host_preparation_fixture.h"

namespace GaussianTest {
    inline const Json* product_expected = nullptr;
    std::string product_context_id(unsigned route, float pixel_size_um) {
        return "route-" + std::to_string(route) + "/pixel-bits-" + std::to_string(std::bit_cast<std::uint32_t>(pixel_size_um));
    }
    Json kernel_case(const std::string& name, int radius, float sigma) {
        JuicerTests::CaseEvent case_event("cpp", name + (product_expected ? "/product" : ""));
        Json weights = Json::array();
        if (radius > 0 && std::isfinite(sigma) && sigma > 0.0f) {
            weights = JuicerTestSupport::with_host_preparation([&](const FjHostPreparation* preparation) {
                JuicerCuda::Failure failure;
                FjGaussianKernelView view{};
                const bool prepared = JuicerCuda::prepare_gaussian_kernel(*preparation, {radius, sigma}, view, failure);
                require(prepared, failure.diagnostic.c_str());
                // Copy into owned JSON while the callback still owns the coefficient backing.
                return words(view.weights.data, view.weights.count);
            });
        }
        Json row{{"name", name}, {"radius", radius}, {"sigma_bits", std::bit_cast<std::uint32_t>(sigma)}, {"count", weights.size()}, {"weights", weights}};
        if (product_expected) {
            const auto expected = std::find_if(product_expected->begin(), product_expected->end(), [&](const Json& value) {
                return value.at("name") == name;
            });
            require(expected != product_expected->end() && row == *expected, "per-case complete Gaussian product");
        }
        case_event.passed();
        return row;
    }

    Json products(const std::filesystem::path& resources, const Json& cohort, const Json::array_t& expected) {
        Json productRows = expected;
        for (auto& row : productRows)
            row.erase("primitives");
        product_expected = &productRows;
        std::vector<std::string> context_ids;
        const auto pixel_sizes = cohort.at("products").at("pixel_size_um").get<std::vector<float>>();
        for (unsigned route = 0; route < 4; ++route)
            for (const auto pixel_size_um : pixel_sizes)
                context_ids.push_back(product_context_id(route, pixel_size_um));
        JuicerTests::Cohort product_contexts("gaussian.product_context", context_ids);
        Json rows = Json::array();
        for (const auto& input : cohort.at("cases"))
            rows.push_back(kernel_case(input.at("name"), input.at("radius"), std::bit_cast<float>(input.at("sigma_bits").get<std::uint32_t>())));
        JuicerTestSupport::RuntimeFixture owner;
        owner.create(resources.string().c_str());
        for (unsigned route = 0; route < 4; ++route) {
            JuicerTestSupport::PreparationInput p;
            p.values.scan_route = static_cast<std::uint32_t>(route);
            p.film_key = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            p.values.grain.active = true;
            p.values.grain.sublayers_active = true;
            p.values.scanner_lens_blur_sigma_px = 0.75;
            p.values.scanner_unsharp_mask[0] = 1.25;
            p.values.scanner_unsharp_mask[1] = 0.5;
            p.values.dir_couplers.diffusion_size_um = 20.0f;
            p.values.dir_couplers.diffusion_tail_um = 200.0f;
            p.values.dir_couplers.diffusion_tail_weight = 0.03f;
            auto product = JuicerTestSupport::prepare_product(owner.borrow().assets, p);
            FjPreparedStateView view{};
            FjErrorBuffer viewError{};
            status(product.view(view, viewError));
            const auto prefix = std::to_string(route) + "/";
            const auto& post = view.scanner_post_effects;
            rows.push_back(kernel_case(prefix + "scanner-lens", post.lens_blur_radius, post.lens_blur_sigma_px));
            rows.push_back(kernel_case(prefix + "scanner-unsharp", post.unsharp_radius, post.unsharp_sigma_px));
            rows.push_back(kernel_case(prefix + "scanner-glare", post.glare_blur_radius, post.glare_blur_sigma_px));
            for (const float pixelSize : pixel_sizes) {
                auto context_event = product_contexts.case_event(product_context_id(route, pixelSize));
                const auto scale = prefix + std::to_string(pixelSize) + "/";
                FjGrainFrameInput grainFrame{};
                grainFrame.render_extent = {0, 0, 64, 48};
                grainFrame.full_frame_extent = grainFrame.render_extent;
                grainFrame.pixel_size_um = pixelSize;
                grainFrame.frame_time = 12.0;
                grainFrame.frame_rate = 24.0;
                grainFrame.session_seed = 17;
                grainFrame.clip_token = 23;
                const auto rawGrain = JuicerTestSupport::grain_frame_input(view, grainFrame);
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
                context_event.passed();
            }
        }
        product_expected = nullptr;
        return rows;
    }
    void values(const Json& rows) {
        require(rows.size() == 188, "complete Gaussian fixture membership");
        for (const auto& row : rows) {
            const auto& name = row.at("name").get_ref<const std::string&>();
            JuicerTests::CaseEvent case_event("cpp", name + std::string("/values"));
            const int radius = row.at("radius");
            const auto bits = row.at("sigma_bits").get<std::uint32_t>();
            const float sigma = std::bit_cast<float>(bits);
            if (radius <= 0 || !std::isfinite(sigma) || sigma <= 0.0f) {
                case_event.passed();
                continue;
            }
            const FjGaussianKernelInput input{radius, sigma};
            FjGaussianKernelOwner* owner = nullptr;
            FjGaussianKernelFailure failure{99};
            status(fj_test_gaussian_kernel_acquire(&input, &owner, &failure, nullptr));
            require(owner && failure.reason == 0, "Gaussian complete unique owner");
            FjGaussianKernelView view{};
            status(fj_test_gaussian_kernel_view(owner, &view, &failure, nullptr));
            require(view.radius == radius && std::bit_cast<std::uint32_t>(view.sigma_px) == bits &&
                        view.weights.count == row.at("count").get<std::size_t>() && failure.reason == 0,
                    "exact Gaussian metadata");
            require(words(view.weights.data, view.weights.count) == row.at("weights"), "all independent Gaussian words through the checked reader");
            status(fj_test_gaussian_kernel_release(owner, nullptr));
            JuicerTestSupport::with_host_preparation([&](const FjHostPreparation* preparation) {
                JuicerCuda::Failure rejected;
                FjGaussianKernelView callbackView{};
                const bool prepared = JuicerCuda::prepare_gaussian_kernel(*preparation, input, callbackView, rejected);
                require(prepared, rejected.diagnostic.c_str());
                require(callbackView.radius == radius && std::bit_cast<std::uint32_t>(callbackView.sigma_px) == bits &&
                            callbackView.weights.count == row.at("count").get<std::size_t>(),
                        "exact callback Gaussian metadata");
                require(words(callbackView.weights.data, callbackView.weights.count) == row.at("weights"), "all frozen callback Gaussian words before backing expires");
                return true;
            });
            const auto expected = row.at("weights").get<std::vector<std::uint32_t>>();
            JuicerTests::CaseEvent c11_event("c11", name);
            require(fj_test_gaussian_c_call(&input, expected.data(), expected.size()) != 0, "compiled C11 full Gaussian arrays and metadata");
            c11_event.passed();
            if (radius == 2048 || (radius == 3 && sigma == 1.0f)) {
                auto changed = expected;
                for (std::size_t index : {std::size_t{0}, expected.size() / 2, expected.size() - 1}) {
                    changed[index] ^= 1;
                    require(fj_test_gaussian_c_call(&input, changed.data(), changed.size()) == 0, "C11 reader rejects first, center and final word corruption");
                    changed[index] ^= 1;
                }
            }
            case_event.passed();
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
        FjGaussianKernelView view{};
        JuicerCuda::Failure failure;
        JuicerTests::Cohort callback_diagnostics("gaussian.callback_diagnostic", {"allocation-failed", "candidate-allocation-failed", "storage-unavailable", "empty-failure", "nonempty-success", "empty-success"});
        unsigned diagnostic_index = 0;
        for (const char* diagnostic : {"Gaussian kernel allocation failed", "Gaussian kernel candidate storage allocation failed", "Gaussian coefficient storage is unavailable", ""}) {
            static const char* const ids[] = {"allocation-failed", "candidate-allocation-failed", "storage-unavailable", "empty-failure"};
            auto row_event = callback_diagnostics.case_event(ids[diagnostic_index++]);
            reply.diagnostic = diagnostic;
            view = {};
            failure = {};
            require(!JuicerCuda::prepare_gaussian_kernel(preparation, input, view, failure), "typed Gaussian allocation callback rejects preparation");
            status(failure.status, FJ_STATUS_ALLOCATION_FAILURE);
            require(failure.diagnostic == diagnostic, "Gaussian allocation category is independent of forwarded diagnostic wording");
            require(view.radius == 0 && std::bit_cast<std::uint32_t>(view.sigma_px) == 0 && !view.weights.data && !view.weights.count, "Gaussian callback failure leaves no published view");
            row_event.passed();
        }
        {
            auto row_event = callback_diagnostics.case_event("nonempty-success");
            reply.success = true;
            reply.diagnostic = "unexpected success diagnostic";
            view = {};
            failure = {};
            require(!JuicerCuda::prepare_gaussian_kernel(preparation, input, view, failure), "successful Gaussian transport still requires an empty diagnostic");
            status(failure.status, FJ_STATUS_PREPARATION_FAILURE);
            require(failure.diagnostic == "Gaussian kernel projection invalid", "nonempty Gaussian success retains transport rejection");
            row_event.passed();
        }
        {
            auto row_event = callback_diagnostics.case_event("empty-success");
            reply.diagnostic = "";
            require(JuicerCuda::prepare_gaussian_kernel(preparation, input, view, failure), "empty Gaussian success diagnostic is admitted");
            require(failure.diagnostic.empty(), "successful Gaussian transport clears the previous rejection diagnostic");
            require(view.weights.data == reply.weights.data() && view.weights.count == reply.weights.size(), "Gaussian transport preserves the literal borrowed span");
            row_event.passed();
        }
    }
    void boundary() {
        JuicerTests::CaseEvent boundary_event("cpp", "gaussian/boundary");
        callback_failure_diagnostics();
        const FjGaussianKernelInput input{3, 1.0f};
        FjGaussianKernelFailure failure{99};
        FjGaussianKernelOwner* owner = nullptr;
        JuicerTests::Cohort domain_probes("gaussian.domain_probe", {"0", "1", "2", "3", "4", "5", "6", "7"});
        unsigned probe_index = 0;
        for (auto probe : {FjGaussianKernelInput{0, 0}, FjGaussianKernelInput{2049, 1}, FjGaussianKernelInput{-1, NAN}, FjGaussianKernelInput{1, 0}, FjGaussianKernelInput{1, -0.0f}, FjGaussianKernelInput{1, -1}, FjGaussianKernelInput{1, INFINITY}, FjGaussianKernelInput{1, NAN}}) {
            auto row_event = domain_probes.case_event(std::to_string(probe_index++));
            const auto reason = probe.radius < 1 || probe.radius > 2048 ? 2u : 3u;
            owner = reinterpret_cast<FjGaussianKernelOwner*>(std::uintptr_t{1});
            char text[128] = "seed";
            FjErrorBuffer error{text, sizeof(text), 99};
            status(fj_test_gaussian_kernel_acquire(&probe, &owner, &failure, &error), 3);
            require(!owner && failure.reason == reason, "ordered Gaussian domain rejection");
            require(std::string_view(text) == (reason == 2 ? "Gaussian kernel radius outside supported range" : "invalid Gaussian kernel sigma"), "exact Gaussian domain diagnostic");
            row_event.passed();
        }
        static const char* const missing_ids[] = {"null-input", "null-owner", "null-failure", "malformed-error", "invalid-input"};
        JuicerTests::Cohort missing_peers("gaussian.missing_peer", {"null-input", "null-owner", "null-failure", "malformed-error", "invalid-input"});
        for (unsigned missing = 0; missing < 5; ++missing) {
            auto row_event = missing_peers.case_event(missing_ids[missing]);
            owner = reinterpret_cast<FjGaussianKernelOwner*>(std::uintptr_t{1});
            failure.reason = 99;
            FjErrorBuffer malformed{nullptr, 1, 99};
            const auto* in = missing == 0 ? nullptr : &input;
            auto* out = missing == 1 ? nullptr : &owner;
            auto* fail = missing == 2 ? nullptr : &failure;
            auto* error = missing == 3 ? &malformed : nullptr;
            if (missing == 4)
                in = reinterpret_cast<const FjGaussianKernelInput*>(std::uintptr_t{1});
            status(fj_test_gaussian_kernel_acquire(in, out, fail, error), 2);
            require(!out || !owner, "Gaussian valid owner peer clears independently");
            require(!fail || failure.reason == 1, "Gaussian valid failure peer clears independently");
            row_event.passed();
        }
        FjErrorBuffer zero{reinterpret_cast<char*>(std::uintptr_t{1}), 0, 99};
        status(fj_test_gaussian_kernel_acquire(&input, &owner, &failure, &zero));
        require(zero.length == 0, "zero capacity never accesses diagnostic backing");
        FjGaussianKernelView v{99, 99, {reinterpret_cast<const float*>(std::uintptr_t{1}), 99}};
        status(fj_test_gaussian_kernel_view(nullptr, &v, &failure, nullptr), 2);
        require(v.radius == 0 && std::bit_cast<std::uint32_t>(v.sigma_px) == 0 && !v.weights.data && !v.weights.count && failure.reason == 1, "entire Gaussian view independently zeroed");
        FjErrorBuffer malformed{nullptr, 1, 99};
        status(fj_test_gaussian_kernel_release(owner, &malformed), 2);
        require(fj_test_gaussian_kernel_live_owners() == 0, "malformed release diagnostics consume actual owner");
        JuicerTests::Cohort boundary_operations("gaussian.boundary_operation", 3);
        for (unsigned operation = 0; operation < 3; ++operation) {
            auto row_event = boundary_operations.case_event(std::to_string(operation));
            if (operation != 0)
                status(fj_test_gaussian_kernel_acquire(&input, &owner, &failure, nullptr));
            status(fj_test_gaussian_kernel_fault(operation, 1));
            char text[128]{};
            FjErrorBuffer error{text, sizeof(text), 0};
            if (operation == 0)
                status(fj_test_gaussian_kernel_acquire(&input, &owner, &failure, &error), 8);
            else if (operation == 1) {
                FjGaussianKernelView v{99, 99, {nullptr, 99}};
                status(fj_test_gaussian_kernel_view(owner, &v, &failure, &error), 8);
                require(!v.weights.data && !v.weights.count && v.radius == 0 && v.sigma_px == 0 && failure.reason == 5, "view panic clears every field");
                status(fj_test_gaussian_kernel_release(owner, nullptr));
            } else
                status(fj_test_gaussian_kernel_release(owner, &error), 8);
            require(std::string_view(text) == (operation == 2 ? "Gaussian kernel release panicked" : "Gaussian kernel preparation panicked"), "Gaussian panic diagnostic");
            require(fj_test_gaussian_kernel_live_owners() == 0, "panic consumes provisional/owned holds");
            row_event.passed();
        }
        std::array<char, 128> diagnosticText{};
        FjErrorBuffer diagnosticError{diagnosticText.data(), diagnosticText.size(), 0};
        status(fj_test_gaussian_kernel_acquire(nullptr, &owner, &failure, &diagnosticError), 2);
        require(!owner && failure.reason == 1 && std::string_view(diagnosticText.data()) == "invalid Gaussian kernel arguments", "exact raw Gaussian failure diagnostic");
        status(fj_test_gaussian_kernel_fault(0, 2));
        status(fj_test_gaussian_kernel_acquire(&input, &owner, &failure, &diagnosticError), 4);
        require(std::string_view(diagnosticText.data()) == "Gaussian kernel allocation failed", "exact Gaussian reservation failure diagnostic");
        require(!owner && failure.reason == 4, "actual coefficient reservation failure");
        JuicerTests::Cohort projection_faults("gaussian.projection_fault", 4);
        for (unsigned fault = 1; fault <= 4; ++fault) {
            auto row_event = projection_faults.case_event(std::to_string(fault - 1));
            status(fj_test_gaussian_kernel_fault(1, fault));
            JuicerTestSupport::with_host_preparation([&](const FjHostPreparation* preparation) {
                JuicerCuda::Failure rejected;
                FjGaussianKernelView view{};
                require(!JuicerCuda::prepare_gaussian_kernel(*preparation, input, view, rejected), "invalid Gaussian callback projection rejected before reading weights");
                status(rejected.status, fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_PREPARATION_FAILURE);
                require(rejected.diagnostic == (fault == 1 ? "Gaussian kernel preparation panicked" : "Gaussian kernel projection invalid"), "Gaussian projection exact diagnostic");
                return true;
            });
            require(fj_test_gaussian_kernel_live_owners() == 0, "callback rejection releases its backing after the synchronous body");
            row_event.passed();
        }
        JuicerTests::Cohort diagnostic_capacities("gaussian.diagnostic_capacity", {"0", "1", "5", "128"});
        for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{5}, std::size_t{128}}) {
            auto row_event = diagnostic_capacities.case_event(std::to_string(capacity));
            std::array<char, 128> text{};
            text.fill('Q');
            FjErrorBuffer error{capacity ? text.data() : reinterpret_cast<char*>(std::uintptr_t{1}), capacity, 99};
            const FjGaussianKernelInput invalid{0, NAN};
            status(fj_test_gaussian_kernel_acquire(&invalid, &owner, &failure, &error), 3);
            const auto expected = capacity ? std::min(capacity - 1, std::string_view("Gaussian kernel radius outside supported range").size()) : 0;
            require(!owner && failure.reason == 2 && error.length == expected, "bounded Gaussian diagnostic length and ordered failure");
            if (capacity)
                require(text[expected] == '\0', "bounded Gaussian diagnostic terminator");
            for (std::size_t i = capacity; i < text.size(); ++i)
                require(text[i] == 'Q', "diagnostic cannot overwrite capacity");
            row_event.passed();
        }
        status(fj_test_gaussian_kernel_acquire(&input, &owner, &failure, nullptr));
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
        status(fj_test_gaussian_kernel_release(owner, nullptr));
        for (const auto& error : errors)
            if (error)
                std::rethrow_exception(error);
        status(fj_test_gaussian_kernel_fault(3, 0), 2);
        status(fj_test_gaussian_kernel_fault(0, 3), 2);
        status(fj_test_gaussian_kernel_fault(1, 5), 2);
        status(fj_test_gaussian_kernel_fault(2, 2), 2);
        status(fj_test_gaussian_kernel_release(nullptr, nullptr));
        JuicerTestSupport::with_host_preparation([&](const FjHostPreparation* preparation) {
            JuicerCuda::Failure rejected;
            FjGaussianKernelView view{};
            require(JuicerCuda::prepare_gaussian_kernel(*preparation, {2048, 512.0f}, view, rejected), "Gaussian callback recovery reaches upper bound");
            require(view.radius == 2048 && view.sigma_px == 512.0f && view.weights.count == 4097, "recovered Gaussian callback metadata remains complete");
            return true;
        });
        require(fj_test_gaussian_kernel_live_owners() == 0, "Gaussian boundary owner balance");
        boundary_event.passed();
    }
} // namespace GaussianTest
