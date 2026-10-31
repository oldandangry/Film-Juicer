namespace OpticsTest {
    thread_local unsigned nextProjection = 0;
    using Prepare = FjStatus (*)(const FjSpatialOpticsParams*, FjSpatialOpticsRecipe*, FjSpatialOpticsFailure*, FjErrorBuffer*);
    Json double_words(const double* values, std::size_t count) {
        Json out = Json::array();
        for (std::size_t i = 0; i < count; ++i)
            out.push_back(std::bit_cast<std::uint64_t>(values[i]));
        return out;
    }
    FjDiffusionParams diffusion_params(const Json& input) {
        FjDiffusionParams out{};
        out.active = input.at("active").get<bool>() ? 1u : 0u;
        out.family = input.at("family").get<std::uint32_t>();
        const auto& v = input.at("f64");
        const std::array<double*, 9> fields{&out.strength, &out.spatial_scale, &out.halo_warmth, &out.core_intensity, &out.core_size, &out.halo_intensity, &out.halo_size, &out.bloom_intensity, &out.bloom_size};
        for (std::size_t i = 0; i < fields.size(); ++i)
            *fields[i] = std::bit_cast<double>(v[i].get<std::uint64_t>());
        return out;
    }
    FjSpatialOpticsParams params(const Json& input) {
        FjSpatialOpticsParams out{};
        out.camera_diffusion = diffusion_params(input.at("camera"));
        out.enlarger_diffusion = diffusion_params(input.at("enlarger"));
        out.camera_lens_blur_um = std::bit_cast<float>(input.at("lens_f32").get<std::uint32_t>());
        out.print_domain = input.at("print").get<bool>() ? 1u : 0u;
        auto& h = out.scatter_halation;
        h.active = input.at("halation_active").get<bool>() ? 1u : 0u;
        const auto& v = input.at("halation_f32");
        h.scatter_amount = std::bit_cast<float>(v[0].get<std::uint32_t>());
        h.scatter_spatial_scale = std::bit_cast<float>(v[1].get<std::uint32_t>());
        h.halation_amount = std::bit_cast<float>(v[2].get<std::uint32_t>());
        h.halation_spatial_scale = std::bit_cast<float>(v[3].get<std::uint32_t>());
        copy_words(input.at("first_sigma_f32"), h.first_sigma_um);
        copy_words(input.at("primary_f32"), h.primary_amount);
        return out;
    }
    Json project(const FjDiffusionRecipe& v) {
        const double values[]{v.scatter_fraction, v.group_weights[0], v.group_weights[1], v.group_weights[2], v.group_lambda_um[0], v.group_lambda_um[1], v.group_lambda_um[2], v.warmth, v.spatial_scale};
        return {{"domain", v.domain}, {"active", v.active != 0}, {"family", v.family}, {"f64", double_words(values, 9)}, {"resolved_hash", v.resolved_hash}, {"hash", v.hash}};
    }
    Json project(const FjSpatialOpticsRecipe& v) {
        const auto& h = v.scatter_halation;
        const float values[]{h.scatter_amount, h.scatter_spatial_scale, h.halation_spatial_scale, h.first_sigma_um[0], h.first_sigma_um[1], h.first_sigma_um[2], h.total_strength[0], h.total_strength[1], h.total_strength[2]};
        return {{"camera", project(v.camera_diffusion)}, {"enlarger", project(v.enlarger_diffusion)}, {"lens_f32", std::bit_cast<std::uint32_t>(v.camera_lens_blur.sigma_um)}, {"lens_hash", v.camera_lens_blur.hash}, {"halation", {{"scatter_active", h.scatter_active != 0}, {"back_reflection_active", h.back_reflection_active != 0}, {"f32", words(values, 9)}, {"hash", h.hash}}}, {"hash", v.hash}};
    }
    Json project(const DiffusionFilterOpticsRecipe& v) {
        const auto& r = v.resolved;
        const double values[]{r.scatterFraction, r.groupWeightsCoreHaloBloom[0], r.groupWeightsCoreHaloBloom[1], r.groupWeightsCoreHaloBloom[2], r.groupCenterLambdaUm[0], r.groupCenterLambdaUm[1], r.groupCenterLambdaUm[2], r.effectiveWarmth, r.spatialScale};
        return {{"domain", static_cast<unsigned>(v.domain)}, {"active", r.active}, {"family", static_cast<unsigned>(r.family)}, {"f64", double_words(values, 9)}, {"resolved_hash", r.hash}, {"hash", v.hash}};
    }
    Json project(const SpatialOptics& v) {
        const auto& h = v.scatterHalation;
        const float values[]{h.scatterAmount, h.scatterSpatialScale, h.halationSpatialScale, h.halationFirstSigmaUm[0], h.halationFirstSigmaUm[1], h.halationFirstSigmaUm[2], h.totalStrength[0], h.totalStrength[1], h.totalStrength[2]};
        return {{"camera", project(v.cameraDiffusion)}, {"enlarger", project(v.enlargerDiffusion)}, {"lens_f32", std::bit_cast<std::uint32_t>(v.cameraLensBlur.sigmaUm)}, {"lens_hash", v.cameraLensBlur.hash}, {"halation", {{"scatter_active", h.scatterActive}, {"back_reflection_active", h.backReflectionActive}, {"f32", words(values, 9)}, {"hash", h.hash}}}, {"hash", v.hash}};
    }
    void metadata(const FjSpatialOpticsFailure& actual, std::uint32_t reason = FJ_OPTICS_REASON_NONE, std::uint32_t component = FJ_OPTICS_COMPONENT_NONE, std::uint32_t field = FJ_OPTICS_FIELD_NONE, std::uint32_t channel = 3) {
        require(actual.reason == reason && actual.component == component && actual.field == field && actual.channel == channel, "complete optics failure envelope");
    }
    FjSpatialOpticsFailure expected_failure(const std::string& diagnostic) {
        FjSpatialOpticsFailure out{FJ_OPTICS_REASON_DIFFUSION, FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION, 0, 3};
        if (diagnostic.find("component=enlarger_diffusion") != std::string::npos)
            out.component = FJ_OPTICS_COMPONENT_ENLARGER_DIFFUSION;
        if (diagnostic.find("component=camera_lens_blur") != std::string::npos) {
            out.reason = FJ_OPTICS_REASON_LENS_BLUR;
            out.component = FJ_OPTICS_COMPONENT_CAMERA_LENS_BLUR;
        }
        if (diagnostic.find("component=scatter_halation") != std::string::npos) {
            out.reason = FJ_OPTICS_REASON_HALATION;
            out.component = FJ_OPTICS_COMPONENT_SCATTER_HALATION;
        }
        const std::array<const char*, 16> fields{"family", "strength", "spatial_scale", "halo_warmth", "core_intensity", "core_size", "halo_intensity", "halo_size", "bloom_intensity", "bloom_size", "scatter_fraction", "group_weights", "group_centers_um", "resolved_hash", "sigma_um", "total_strength"};
        for (std::size_t i = 0; i < fields.size(); ++i)
            if (diagnostic.find(std::string("field=") + fields[i]) != std::string::npos)
                out.field = static_cast<std::uint32_t>(i + 1);
        for (unsigned i = 0; i < 3; ++i)
            if (diagnostic.find("channel=" + std::to_string(i)) != std::string::npos)
                out.channel = i;
        require(out.field != 0, "native capture has a known ordinary failure field");
        return out;
    }
    void zero(const FjSpatialOpticsRecipe& out) {
        require(project(out) == project(FjSpatialOpticsRecipe{}), "every seeded optics result field clears to integer/+float zero");
    }
    void seed(FjSpatialOpticsRecipe& out) {
        std::memset(&out, 0x55, sizeof(out));
    }
    void representations(const FjSpatialOpticsRecipe& out) {
        require(out.camera_diffusion.active <= 1 && out.enlarger_diffusion.active <= 1 && out.scatter_halation.scatter_active <= 1 && out.scatter_halation.back_reflection_active <= 1, "exact boolean output representations");
    }
    void values(const Json& captured) {
        require(captured.at("cases").size() == 553, "exact frozen optics literal membership");
        for (Prepare prepare : {fj_legacy_spatial_optics_prepare, fj_test_spatial_optics_prepare})
            for (const auto& row : captured.at("cases")) {
                auto input = params(row.at("input"));
                FjSpatialOpticsRecipe out{};
                seed(out);
                FjSpatialOpticsFailure failure{99, 98, 97, 96};
                const bool success = row.at("success");
                status(prepare(&input, &out, &failure, nullptr), success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (success) {
                    metadata(failure);
                    representations(out);
                    require(project(out) == row.at("recipe"), "all frozen f32/f64 fields, flags/tags and seven identities");
                } else {
                    const auto expected = expected_failure(row.at("diagnostic").get<std::string>());
                    metadata(failure, expected.reason, expected.component, expected.field, expected.channel);
                    zero(out);
                }
            }
    }
    void raw_and_faults(const Json& captured) {
        const auto& rows = captured.at("cases");
        const auto find = [&](const char* id) -> const Json& {
            for (const auto& row : rows)
                if (row.at("id") == id)
                    return row;
            throw std::runtime_error("missing optics case");
        };
        const auto inactive = params(find("optics-inactive").at("input"));
        const auto active = params(find("optics-active-print").at("input"));
        for (Prepare prepare : {fj_legacy_spatial_optics_prepare, fj_test_spatial_optics_prepare}) {
            FjSpatialOpticsRecipe out{};
            FjSpatialOpticsFailure failure{};
            const auto raw = [&](const FjSpatialOpticsParams* input, FjSpatialOpticsRecipe* result, FjSpatialOpticsFailure* rejected, FjErrorBuffer* error) {
                seed(out);
                failure = {99, 98, 97, 96};
                status(prepare(input, result, rejected, error), FJ_STATUS_UNSUPPORTED_INPUT);
                if (result == &out)
                    zero(out);
                if (rejected == &failure)
                    metadata(failure, FJ_OPTICS_REASON_RAW);
            };
            raw(nullptr, &out, &failure, nullptr);
            raw(&inactive, nullptr, &failure, nullptr);
            raw(&inactive, &out, nullptr, nullptr);
            alignas(FjSpatialOpticsParams) std::array<std::byte, sizeof(FjSpatialOpticsParams) + 8> storage{};
            raw(reinterpret_cast<const FjSpatialOpticsParams*>(storage.data() + 1), &out, &failure, nullptr);
            raw(reinterpret_cast<const FjSpatialOpticsParams*>(UINTPTR_MAX - 7), &out, &failure, nullptr);
            raw(&inactive, reinterpret_cast<FjSpatialOpticsRecipe*>(storage.data() + 1), &failure, nullptr);
            raw(&inactive, &out, reinterpret_cast<FjSpatialOpticsFailure*>(storage.data() + 1), nullptr);
            raw(&inactive, &out, &failure, reinterpret_cast<FjErrorBuffer*>(storage.data() + 1));
            FjErrorBuffer bad{nullptr, 1, 99};
            raw(&inactive, &out, &failure, &bad);
            require(bad.length == 0, "admitted diagnostic length clears before invalid backing");
            bad = {reinterpret_cast<char*>(1), SIZE_MAX, 99};
            raw(&inactive, &out, &failure, &bad);
            require(bad.length == 0, "overflowing diagnostic extent rejected before backing");
            for (unsigned flag = 0; flag < 4; ++flag) {
                auto input = inactive;
                std::array<std::uint32_t*, 4> flags{&input.camera_diffusion.active, &input.scatter_halation.active, &input.enlarger_diffusion.active, &input.print_domain};
                *flags[flag] = 2;
                raw(&input, &out, &failure, nullptr);
            }
            auto competing = inactive;
            competing.camera_diffusion.strength = std::numeric_limits<double>::quiet_NaN();
            competing.enlarger_diffusion.active = 2;
            raw(&competing, &out, &failure, nullptr);
            FjErrorBuffer noBacking{reinterpret_cast<char*>(1), 0, 99};
            status(prepare(&inactive, &out, &failure, &noBacking));
            metadata(failure);
            require(noBacking.length == 0 && project(out) == find("optics-inactive").at("recipe"), "capacity zero never reads backing and differs from raw clearing");
            auto invalid = inactive;
            invalid.camera_diffusion.family = 255;
            for (std::size_t capacity : {std::size_t{1}, std::size_t{3}, std::size_t{256}}) {
                std::array<char, 256> text{};
                text.fill('x');
                FjErrorBuffer error{text.data(), capacity, 99};
                seed(out);
                status(prepare(&invalid, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
                metadata(failure, FJ_OPTICS_REASON_DIFFUSION, FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION, FJ_OPTICS_FIELD_FAMILY);
                zero(out);
                require(error.length < capacity && text[error.length] == '\0' && (capacity == 1 ? error.length == 0 : error.length > 0), "bounded terminated/truncated diagnostics preserve the complete outcome");
            }
            for (unsigned fault : {1u, 2u}) {
                status(fj_test_film_recipe_fault(9, fault));
                seed(out);
                status(prepare(&active, &out, &failure, nullptr), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_PREPARATION_FAILURE);
                if (fault == 1)
                    metadata(failure, FJ_OPTICS_REASON_PANIC);
                else
                    metadata(failure, FJ_OPTICS_REASON_DIFFUSION, FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION, FJ_OPTICS_FIELD_RESOLVED_HASH);
                zero(out);
                status(prepare(&active, &out, &failure, nullptr));
                metadata(failure);
                require(project(out) == find("optics-active-print").at("recipe"), "fault is one shot");
                status(fj_test_film_recipe_fault(9, fault));
                raw(nullptr, &out, &failure, nullptr);
                status(prepare(&active, &out, &failure, nullptr));
                metadata(failure);
                status(fj_test_film_recipe_fault(9, fault));
                raw(&active, nullptr, &failure, nullptr);
                status(prepare(&active, &out, &failure, nullptr));
                metadata(failure);
            }
            status(fj_test_film_recipe_fault(9, 2));
            status(prepare(&inactive, &out, &failure, nullptr));
            status(prepare(&active, &out, &failure, nullptr));
            metadata(failure);
            status(fj_test_film_recipe_fault(9, 1));
            status(fj_test_film_recipe_fault(9, 0));
            status(prepare(&active, &out, &failure, nullptr));
            metadata(failure);
            std::atomic<bool> okay{true};
            std::thread isolated([&] {
                FjSpatialOpticsRecipe v{};
                FjSpatialOpticsFailure f{};
                status(fj_test_film_recipe_fault(9, 2));
                const auto s = prepare(&active, &v, &f, nullptr);
                if (s.category != FJ_STATUS_PREPARATION_FAILURE || f.field != FJ_OPTICS_FIELD_RESOLVED_HASH)
                    okay = false;
            });
            status(prepare(&active, &out, &failure, nullptr));
            metadata(failure);
            isolated.join();
            require(okay.load(), "raw fault and core guard are thread local");
        }
    }
    Spektrafilm::DiffusionFilterAuthoredControls native_diffusion(const FjDiffusionParams& v) {
        return {v.active != 0, static_cast<Spektrafilm::DiffusionFilterFamily>(v.family), v.strength, v.spatial_scale, v.halo_warmth, v.core_intensity, v.core_size, v.halo_intensity, v.halo_size, v.bloom_intensity, v.bloom_size};
    }
    Spektrafilm::SpatialOpticsControls native_controls(const FjSpatialOpticsParams& v) {
        return {native_diffusion(v.camera_diffusion), v.camera_lens_blur_um, {v.scatter_halation.active != 0, v.scatter_halation.scatter_amount, v.scatter_halation.scatter_spatial_scale, v.scatter_halation.halation_amount, v.scatter_halation.halation_spatial_scale}, native_diffusion(v.enlarger_diffusion)};
    }
    Profiles::ProfileDigest native_digest(const FjSpatialOpticsParams& v) {
        Profiles::ProfileDigest out;
        std::copy_n(v.scatter_halation.first_sigma_um, 3, out.halationFirstSigmaUm.begin());
        std::copy_n(v.scatter_halation.primary_amount, 3, out.halationPrimaryAmount.begin());
        return out;
    }
    void native(const Json& captured) {
        for (const auto& row : captured.at("cases")) {
            const auto input = params(row.at("input"));
            SpatialOptics out;
            std::string diagnostic;
            const bool success = JuicerAssets::prepare_spatial_optics(native_controls(input), native_digest(input), input.print_domain != 0, out, diagnostic);
            require(success == row.at("success").get<bool>(), "typed native optics outcome");
            if (success) {
                require(project(out) == row.at("recipe"), "complete native named-field projection including unusual permitted values");
                if (out.cameraLensBlur.hash != 0) {
                    require(!Spektrafilm::preflight_camera_lens_blur(out.cameraLensBlur, diagnostic) && diagnostic == "ExactOpticsNotImplementedForPhase6 component=camera_lens_blur", "positive lens remains rejected by render preflight");
                }
            } else
                require(diagnostic == row.at("diagnostic").get<std::string>(), "typed native diagnostics retain exact words and first failure");
        }
        const auto input = params(captured.at("cases")[3].at("input"));
        {
            SpatialOptics out;
            std::string diagnostic;
            status(fj_test_film_recipe_fault(9, 2));
            require(!JuicerAssets::prepare_spatial_optics(native_controls(input), native_digest(input), true, out, diagnostic) && diagnostic == "ResourceDescriptorMismatch phase=6A component=camera_diffusion field=resolved_hash", "actual resolved-hash guard translates through typed native fields");
            status(fj_test_film_recipe_fault(9, 1));
            bool contained = false;
            try {
                (void)JuicerAssets::prepare_spatial_optics(native_controls(input), native_digest(input), true, out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                contained = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(contained, "caught optics panic stays a typed native boundary failure");
        }
        for (unsigned fault = 1; fault <= 13; ++fault) {
            SpatialOptics out;
            out.hash = 99;
            std::string diagnostic;
            nextProjection = fault;
            bool rejected = false;
            try {
                (void)JuicerAssets::prepare_spatial_optics(native_controls(input), native_digest(input), input.print_domain != 0, out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected && project(out) == project(SpatialOptics{}), "malformed status/metadata/representation/diagnostics cannot partially project");
        }
    }
    struct ProductFixtures {
        Json products;
        Json literals;
        Json ofx;
    };
    void products(const std::filesystem::path& resources, const ProductFixtures& fixtures) {
        const auto& captured = fixtures.products;
        const auto& literals = fixtures.literals;
        const auto& ofx = fixtures.ofx;
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        const auto find = [&](const char* id) -> const Json& {
            for (const auto& row : literals.at("cases"))
                if (row.at("id") == id)
                    return row;
            throw std::runtime_error("missing frozen product input");
        };
        const auto active = params(find("optics-active-print").at("input"));
        const auto inactive = params(find("optics-inactive").at("input"));
        const double cameraStrength = params(find("strength-0-21").at("input")).camera_diffusion.strength;
        const double enlargerStrength = std::bit_cast<double>(ofx.at("ofx")[0].at("input").at("enlarger").at("f64")[0].get<std::uint64_t>());
        const double invalidStrength = params(find("optics-off-invalid-camera").at("input")).camera_diffusion.strength;
        require(captured.at("products").size() == 32, "exact genuine optics product membership");
        for (const auto& row : captured.at("products")) {
            const auto route = row.at("route").get<unsigned>();
            const auto variant = row.at("variant").get<unsigned>();
            auto input = variant == 0 ? inactive : active;
            input.camera_diffusion.family = variant % 4;
            input.camera_diffusion.strength = cameraStrength;
            if (variant == 5)
                input.camera_diffusion.active = 0;
            if (variant == 6)
                input.enlarger_diffusion.strength = enlargerStrength;
            if (variant == 7) {
                input.camera_diffusion.active = 0;
                input.camera_diffusion.strength = invalidStrength;
            }
            const auto controls = native_controls(input);
            ParamSnapshot params;
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.printProfileKey = "kodak_portra_endura";
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.cameraAutoExposureEnabled = 0;
            params.inputCompressionEnabled = 0;
            params.outputGamutCompressionEnabled = 0;
            params.cameraDiffusion = controls.cameraDiffusion;
            params.enlargerDiffusion = controls.enlargerDiffusion;
            params.scatterHalationControls = controls.scatterHalation;
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<PreparedState> product;
            auto* assets = JuicerProcess::root().assets().borrow_rust_assets();
            const FjStatus result = prepare_state(assets, params, product, error);
            const bool built = result.category == FJ_STATUS_SUCCESS && product.has_value();
            const std::string observedDiagnostic(diagnostic.data(), error.length);
            FjPreparedStateView view{};
            if (built)
                status(product->view(view, error));
            require(built == row.at("success").get<bool>() && observedDiagnostic == row.at("diagnostic").get<std::string>(), "genuine held-state route product outcome and first failure");
            Json recipe;
            if (built) {
                recipe = project(view.optics);
            } else {
                require(result.category == FJ_STATUS_PREPARATION_FAILURE && result.api == FJ_API_NONE &&
                            result.native_code == 0 && !product,
                        "failed optics preparation returns no complete owner");
                // SF_TEMP_BRIDGE_NATIVE_OPTICS_FAILURE_OUTPUT: the frozen
                // default C++ output is not a failed Rust state's projection.
                // Retain this original assertion through its I7 owner gate.
                FocusedRenderStateBuildProduct comparison;
                std::string comparisonDiagnostic;
                const bool comparisonBuilt = route % 2 ? build_print_render_state_product(params, comparison, comparisonDiagnostic)
                                                       : build_direct_render_state_product(params, comparison, comparisonDiagnostic);
                require(!comparisonBuilt && comparisonDiagnostic == observedDiagnostic,
                        "original native optics failure retains its exact diagnostic");
                recipe = project(comparison.recipe.spatialOptics);
            }
            const Json actual{{"snapshot_hash", hash_params(params)}, {"final_hash", built ? view.recipe_hash : 0}, {"recipe", std::move(recipe)}};
            const Json expected{{"snapshot_hash", row.at("snapshot_hash")}, {"final_hash", row.at("final_hash")}, {"recipe", row.at("recipe")}};
            require_equal(actual, expected, "native authored snapshot, complete optics and final route identities");
        }
        for (unsigned route = 0; route < 4; ++route) {
            ParamSnapshot baseline;
            baseline.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            baseline.printProfileKey = "kodak_portra_endura";
            baseline.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            baseline.cameraAutoExposureEnabled = 0;
            baseline.inputCompressionEnabled = 0;
            baseline.outputGamutCompressionEnabled = 0;
            baseline.grainControls.active = false;
            baseline.cameraDiffusion.active = true;
            InstanceState state;
            const auto pending = [&](const ParamSnapshot& value) {
                std::lock_guard<std::mutex> lock(state.pending.m);
                state.pending.value = PendingParamsState::Valid{value, hash_params(value)};
            };
            pending(baseline);
            const auto admitted = admit_pending_render_state(state);
            require(route % 2 ? bool(admitted.printState) : bool(admitted.directState), "complete optics publication on both polarities/routes");
            const auto held = JuicerTestSupport::published_view(admitted);
            const auto hash = held.recipe_hash;
            const auto counter = state.buildCounterNext.load();
            auto invalid = baseline;
            invalid.cameraDiffusion.active = false;
            invalid.cameraDiffusion.strength = invalidStrength;
            pending(invalid);
            const auto rejected = admit_pending_render_state(state);
            require(rejected.status == PendingRenderAdmissionStatus::RebuildFailed && !rejected.directState && !rejected.printState && rejected.diagnostic == "ResourceDescriptorMismatch phase=6A component=camera_diffusion field=strength" && held.recipe_hash == hash && state.buildCounterNext.load() == counter, "disabled invalid camera blocks admission without partial publication or revoking held state");
            pending(baseline);
            const auto recovered = admit_pending_render_state(state);
            require(route % 2 ? bool(recovered.printState) : bool(recovered.directState), "optics failure recovery");
            const auto repeated = admit_pending_render_state(state);
            require(repeated.directState == recovered.directState && repeated.printState == recovered.printState, "unchanged recovered publication reuses state");
            auto unused = baseline;
            unused.enlargerDiffusion.family = static_cast<Spektrafilm::DiffusionFilterFamily>(255);
            unused.enlargerDiffusion.strength = invalidStrength;
            std::string diagnostic;
            const auto build = [&](const ParamSnapshot& input) {
                pending(input);
                const auto result = admit_pending_render_state(state);
                diagnostic = result.diagnostic;
                return route % 2 ? bool(result.printState) : bool(result.directState);
            };
            const bool success = build(unused);
            if (route % 2)
                require(!success && diagnostic == "ResourceDescriptorMismatch phase=6A component=enlarger_diffusion field=family", "print domain validates enlarger family before finite controls");
            else {
                const auto direct = admit_pending_render_state(state);
                const auto directView = JuicerTestSupport::published_view(direct);
                require(success && directView.optics.hash == held.optics.hash && hash_params(unused) != hash_params(baseline), "direct ignores enlarger policy while native authored snapshot still changes");
            }
            auto badInput = baseline;
            badInput.inputColorSpace = -1;
            status(fj_test_prepared_optics_resolved_hash_fault());
            const std::string bareInputDiagnostic = "UnsupportedMode phase=3B field=input_color_space";
            require(!build(badInput) &&
                        diagnostic == "RenderStateConstructionFailure category=UnsupportedInput route=" +
                                          std::to_string(route) + " film=" + baseline.filmProfileKey +
                                          " print=" + baseline.printProfileKey + " operation=" + bareInputDiagnostic,
                    "actual admission preserves the unsupported-input category envelope and leaves optics fault pending");
            std::array<char, 2048> rawDiagnosticText{};
            FjErrorBuffer rawError{rawDiagnosticText.data(), rawDiagnosticText.size(), 0};
            std::optional<PreparedState> rawProduct;
            const auto rawStatus = prepare_state(JuicerProcess::root().assets().borrow_rust_assets(), badInput, rawProduct, rawError);
            const std::string observedRawDiagnostic(rawDiagnosticText.data(), rawError.length);
            require(rawStatus.category == FJ_STATUS_UNSUPPORTED_INPUT && rawStatus.api == FJ_API_NONE &&
                        rawStatus.native_code == 0 && !rawProduct && observedRawDiagnostic == bareInputDiagnostic,
                    "direct constructor retains the bare unsupported-input status and diagnostic without an owner");
            auto badGrain = baseline;
            badGrain.grainControls.active = true;
            badGrain.grainControls.particleAreaUm2 = 0;
            require(!build(badGrain) && diagnostic == "ResourceDescriptorMismatch phase=6A component=camera_diffusion field=resolved_hash", "actual optics guard precedes later grain construction");
            require(!build(badGrain) && diagnostic == "MalformedRequiredProfileData phase=9A field=visual_grain", "consumed optics fault permits the later grain failure");
            require(held.recipe_hash == hash && fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "failures preserve publications and drain temporary construction holds");
        }
        std::puts("PASS 32 genuine optics products and four-route publication, ordering, recovery and identity contracts");
    }
} // namespace OpticsTest
namespace JuicerAssets::RecipeTest {
    void optics(FjStatus& status, FjSpatialOpticsFailure& failure, FjSpatialOpticsRecipe& recipe, FjErrorBuffer& error) {
        const unsigned fault = OpticsTest::nextProjection;
        OpticsTest::nextProjection = 0;
        switch (fault) {
            case 1:
                status.category = 999;
                break;
            case 2:
                status.api = 999;
                break;
            case 3:
                status.native_code = 99;
                break;
            case 4:
                failure.reason = 99;
                break;
            case 5:
                failure.component = 99;
                break;
            case 6:
                failure.field = 99;
                break;
            case 7:
                failure.channel = 99;
                break;
            case 8:
                recipe.camera_diffusion.active = 2;
                break;
            case 9:
                recipe.enlarger_diffusion.family = 99;
                break;
            case 10:
                recipe.camera_diffusion.domain = 99;
                break;
            case 11:
                recipe.scatter_halation.back_reflection_active = 2;
                break;
            case 12:
                error.length = error.capacity;
                break;
            case 13:
                failure.reason = FJ_OPTICS_REASON_PANIC;
                break;
            default:
                break;
        }
    }
} // namespace JuicerAssets::RecipeTest
