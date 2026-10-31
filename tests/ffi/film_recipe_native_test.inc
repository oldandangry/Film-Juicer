namespace {
    struct WireMutation {
        bool active = false;
        unsigned operation = 1;
        FjStatus status{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        FjFilmRecipeFailure failure{FJ_FILM_RECIPE_LOG_EXPOSURE, 3, 1};
    };
    thread_local WireMutation wireMutation;
    thread_local unsigned nextViewFault = 0;

    struct RecipeSupersession {
        ParamSnapshot replacement;
        bool fired = false;
    };
    void supersede_recipe_snapshot(InstanceState& state, void* context) {
        auto& selection = *static_cast<RecipeSupersession*>(context);
        if (selection.fired)
            return;
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.value = PendingParamsState::Valid{selection.replacement, hash_params(selection.replacement)};
        selection.fired = true;
    }

    DirCouplersControls native_controls(const FjFilmDirParams& v) {
        DirCouplersControls out;
        out.active = v.active != 0;
        out.amount = v.amount;
        out.inhibitionSameLayer = v.inhibition_same_layer;
        out.inhibitionInterlayer = v.inhibition_interlayer;
        out.diffusionSizeUm = v.diffusion_size_um;
        out.diffusionTailUm = v.diffusion_tail_um;
        out.diffusionTailWeight = v.diffusion_tail_weight;
        out.gammaUseStock = v.gamma_use_stock != 0;
        std::copy_n(v.gamma_same_layer_rgb, 3, out.gammaSameLayerRgb.begin());
        std::copy_n(v.gamma_interlayer_r_to_gb, 2, out.gammaInterlayerRToGb.begin());
        std::copy_n(v.gamma_interlayer_g_to_rb, 2, out.gammaInterlayerGToRb.begin());
        std::copy_n(v.gamma_interlayer_b_to_rg, 2, out.gammaInterlayerBToRg.begin());
        std::copy_n(v.langmuir_donor_k_rgb, 3, out.langmuirDonorKRgb.begin());
        std::copy_n(v.langmuir_receiver_k_rgb, 3, out.langmuirReceiverKRgb.begin());
        return out;
    }
    void native_cases(const std::filesystem::path& fixture, const Json& captured) {
        const auto f1 = read(fixture.parent_path() / (fixture.stem().string() + "-f1.json"));
        for (const auto& row : f1) {
            const auto name = row.at("case").get<std::string>();
            const auto root = (name == "duplicates" || name == "singleton") ? fixture.parent_path().parent_path() / "fixtures/synthetic" / name : fixture.parent_path() / "synthetic" / name;
            JuicerAssets::AssetBridge assets(root);
            require(assets.load_catalog().valid, "F1 native catalog");
            const auto source = assets.film_source("kodak_portra_400");
            auto development = source->prepare_development(Spektrafilm::ScanRoute::NegativeDirectScan, 1.0f, false);
            const auto* address = development.hold.get();
            auto moved = std::move(development.hold);
            require(!development.hold.get() && moved.get() == address, "native move transfers and nulls consuming hold");
            development.hold = std::move(moved);
            // NOLINTNEXTLINE(bugprone-use-after-move): this owner explicitly guarantees a null moved-from handle; the fixture verifies that contract.
            require(!moved.get() && development.hold.get() == address, "native move assignment preserves same development");
            auto raw = params(captured.at("cases")[2].at("controls"));
            const unsigned kind = row.at("kind");
            if (kind == 1)
                raw.active = 0;
            if (kind == 2)
                raw.amount = 0;
            if (kind == 3) {
                raw.inhibition_same_layer = 0;
                raw.inhibition_interlayer = 0;
            }
            if (kind == 4) {
                raw.gamma_use_stock = 0;
                raw.gamma_same_layer_rgb[0] = std::numeric_limits<float>::infinity();
            }
            if (kind == 5)
                raw.langmuir_donor_k_rgb[0] = 0;
            if (kind == 6)
                raw.amount = -1;
            FjFilmDirOwner* owner = nullptr;
            FjFilmRecipeFailure failure{};
            const auto result = fj_legacy_film_dir_acquire(development.hold.get(), &raw, &owner, &failure, nullptr);
            if (row.at("success").get<bool>()) {
                status(result);
                metadata(failure);
                status(fj_legacy_film_dir_release(owner, nullptr));
            } else {
                status(result, FJ_STATUS_PREPARATION_FAILURE);
                require(!owner, "F1 raw failed acquire clears owner");
                if (row.at("diagnostic").get<std::string>().find("log_exposure") != std::string::npos)
                    metadata(failure, FJ_FILM_RECIPE_LOG_EXPOSURE, 3, 1);
            }
            // SF_TEMP_BRIDGE_NATIVE_DIR_SOURCE_DIAGNOSTIC keeps the legacy
            // malformed-source diagnostic comparison until its source owner
            // moves; publication admission below injects at the checked Rust DIR producer.
            DirCouplersRecipe out;
            std::string diagnostic;
            const auto controls = native_controls(raw);
            const bool success = JuicerAssets::prepare_film_dir(development.hold, controls, *source->profile(), development.route, development.recipe.logExposure.size(), out, diagnostic);
            require(success == row.at("success").get<bool>() && diagnostic == row.at("diagnostic").get<std::string>(), "exact native F1 returned diagnostic and bypass order");
            require(fj_test_film_dir_live_owners() == 0, "temporary DIR owners end after copies/failure");
            if (name == "duplicates" && kind == 0) {
                for (unsigned fault = 0; fault < 11; ++fault) {
                    wireMutation = WireMutation{};
                    wireMutation.active = true;
                    switch (fault) {
                        case 0:
                            wireMutation.status.category = FJ_STATUS_SUCCESS;
                            break;
                        case 1:
                            wireMutation.status.category = FJ_STATUS_UNSUPPORTED_INPUT;
                            break;
                        case 2:
                            wireMutation.status.category = FJ_STATUS_ALLOCATION_FAILURE;
                            break;
                        case 3:
                            wireMutation.status.category = FJ_STATUS_INTERNAL_FAILURE;
                            break;
                        case 4:
                            wireMutation.status.api = FJ_API_CUDA_RUNTIME;
                            break;
                        case 5:
                            wireMutation.status.native_code = 1;
                            break;
                        case 6:
                            wireMutation.failure.channel = 0;
                            break;
                        case 7:
                            wireMutation.failure.sample = 0;
                            break;
                        case 8:
                            wireMutation.failure.sample = development.recipe.logExposure.size();
                            break;
                        case 9:
                            wireMutation.failure.sample = UINT64_MAX;
                            break;
                        case 10:
                            wireMutation.failure.reason = 99;
                            break;
                    }
                    bool threw = false;
                    try {
                        (void)JuicerAssets::prepare_film_dir(development.hold, controls, *source->profile(), development.route, development.recipe.logExposure.size(), out, diagnostic);
                    } catch (const JuicerCuda::ExecutionFailure& failure) {
                        threw = failure.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
                    }
                    require(threw, "impossible reason-16 envelope becomes integrity");
                }
            }
            if (name == "singleton" && kind == 0) {
                wireMutation = WireMutation{};
                wireMutation.active = true;
                bool threw = false;
                try {
                    (void)JuicerAssets::prepare_film_dir(development.hold, controls, *source->profile(), development.route, 1, out, diagnostic);
                } catch (const JuicerCuda::ExecutionFailure& failure) {
                    threw = failure.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
                }
                require(threw, "reason-16 impossible for N=1");
            }
            if (name == "duplicates" && kind == 0)
                status(fj_test_film_development_fault(2, 1));
        }
        require(fj_test_film_development_live_owners() == 0, "all F1 development holds released");
        JuicerAssets::AssetBridge assets(fixture.parent_path().parent_path() / "fixtures/synthetic/n21");
        require(assets.load_catalog().valid, "view contract catalog");
        const auto source = assets.film_source("kodak_portra_400");
        auto development = source->prepare_development(Spektrafilm::ScanRoute::NegativeDirectScan, 1.0f, false);
        const auto controls = native_controls(params(captured.at("cases")[2].at("controls")));
        for (bool cleanupFault : {false, true}) {
            DirCouplersRecipe out;
            std::string diagnostic;
            if (cleanupFault)
                status(fj_test_film_recipe_fault(2, 1));
            SpectralAllocationProbe::arm();
            bool failed = false;
            try {
                (void)JuicerAssets::prepare_film_dir(development.hold, controls, *source->profile(), development.route, development.recipe.logExposure.size(), out, diagnostic);
            } catch (const std::bad_alloc&) {
                failed = true;
            }
            SpectralAllocationProbe::clear();
            require(failed && SpectralAllocationProbe::bytes() == development.recipe.logExposure.size() * sizeof(float) && fj_test_film_dir_live_owners() == 0, "native axis copy failure consumes view owner and preserves primary failure through cleanup");
        }
        for (unsigned fault : {1u, 2u, 3u, 4u, 5u}) {
            nextViewFault = fault;
            DirCouplersRecipe out;
            std::string diagnostic;
            bool rejected = false;
            try {
                (void)JuicerAssets::prepare_film_dir(development.hold, controls, *source->profile(), development.route, development.recipe.logExposure.size(), out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected && fj_test_film_dir_live_owners() == 0, "malformed view rejected before copying; temporary owner released");
        }
        for (unsigned operation : {0u, 2u, 3u}) {
            wireMutation = WireMutation{};
            wireMutation.active = true;
            wireMutation.operation = operation;
            bool rejected = false;
            try {
                std::string diagnostic;
                if (operation == 0) {
                    GrainContract out;
                    (void)JuicerAssets::prepare_film_density_minimum(GrainContract{}, out, diagnostic);
                }
                if (operation == 2) {
                    DirCouplersRecipe out;
                    (void)JuicerAssets::prepare_film_dir(development.hold, controls, *source->profile(), development.route, development.recipe.logExposure.size(), out, diagnostic);
                }
                if (operation == 3) {
                    DensityBoundsRecipe out;
                    (void)JuicerAssets::prepare_film_density_bounds(development.hold, GrainContract{}, *source->profile(), development.route, out, diagnostic);
                }
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected && !wireMutation.active && fj_test_film_dir_live_owners() == 0, "reason 16 rejected for minimum/view/bounds operations");
        }
    }

    void products(const std::filesystem::path& resources, const Json& captured) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerProcess::root().ensure_bootstrap();
        for (const auto& row : captured.at("cases")) {
            ParamSnapshot controls;
            controls.filmProfileKey = row.at("key").get<std::string>();
            controls.printProfileKey = "kodak_portra_endura";
            controls.scanRoute = static_cast<Spektrafilm::ScanRoute>(row.at("route").get<unsigned>());
            controls.filmGammaFactor = std::bit_cast<float>(row.at("gamma")[0].get<std::uint32_t>());
            controls.dirCouplers = native_controls(params(row.at("controls")));
            controls.spectralUpsamplingMode = 0;
            controls.cameraAutoExposureEnabled = 0;
            controls.inputCompressionEnabled = 0;
            controls.outputGamutCompressionEnabled = 0;
            controls.grainControls.active = false;
            const bool print = Spektrafilm::scan_route_is_print(controls.scanRoute);
            auto product = JuicerTestSupport::prepare_product(controls);
            FjPreparedStateView view{};
            FjErrorBuffer error{};
            status(product.view(view, error));
            const auto& dir = view.dir_recipe;
            require(hash_params(controls) == row.at("snapshot_hash") && view.recipe_hash == row.at("final_hash"), "64 frozen snapshot/final product identities");
            require(view.film_development.hash == row.at("development_hash") && view.film_exposure.hash == row.at("input_hash"), "accepted development/input identities");
            require(dir.hash == row.at("dir").at("hash") && dir.compensated_axes_hash == row.at("dir").at("axes_hash"), "actual DIR family identities");
            for (std::size_t i = 0; i < 3; ++i)
                require(words(dir.compensated_axes_rgb[i].data, dir.compensated_axes_rgb[i].count) == row.at("dir").at("axes")[i], "all actual adopted axes");
            const auto& bounds = print ? view.enlarger_film_bounds : view.scanner_bounds;
            require(bounds.hash == row.at("bounds").at("hash") && words(bounds.minimum_cmy, 3) == row.at("bounds").at("minimum") && words(bounds.inv_span_cmy, 3) == row.at("bounds").at("inverse"), "actual scanner/enlarger bounds");
            require(fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "no foreign temporary hold in published native products");
        }
        for (unsigned route = 0; route < 4; ++route) {
            ParamSnapshot controls;
            controls.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            controls.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            controls.printProfileKey = "kodak_portra_endura";
            controls.spectralUpsamplingMode = 1;
            controls.cameraAutoExposureEnabled = 0;
            controls.inputCompressionEnabled = 0;
            controls.outputGamutCompressionEnabled = 0;
            controls.grainControls.active = false;
            InstanceState state;
            const auto pending = [&](const ParamSnapshot& snapshot) {
                std::lock_guard<std::mutex> lock(state.pending.m);
                state.pending.value = PendingParamsState::Valid{snapshot, hash_params(snapshot)};
            };
            pending(controls);
            const auto admitted = admit_pending_render_state(state);
            require(route % 2 ? bool(admitted.printState) : bool(admitted.directState), "complete DIR admission baseline");
            const auto original = JuicerTestSupport::published_view(admitted);
            const auto counter = state.buildCounterNext.load();
            const auto oldHash = original.recipe_hash;
            auto changed = controls;
            changed.dirCouplers.active = true;
            changed.dirCouplers.amount = 0.5f;
            pending(changed);
            status(fj_test_film_recipe_fault(5, 3));
            const auto rejected = admit_pending_render_state(state);
            require(rejected.status == PendingRenderAdmissionStatus::RebuildFailed && !rejected.directState && !rejected.printState &&
                        rejected.diagnostic == "MalformedRequiredProfileData phase=3D-3 field=dirCouplers",
                    "checked Rust DIR producer failure blocks complete-state admission");
            require(state.buildCounterNext.load() == counter && original.recipe_hash == oldHash && !JuicerAtomic::load_shared_ptr(&state.activeDirectState) && !JuicerAtomic::load_shared_ptr(&state.activePrintState), "no intermediate publication; old admitted view remains readable after selected-slot clear");
            require(fj_test_film_dir_live_owners() == 0, "failed Rust DIR construction retains no foreign DIR owner");
            const auto recovered = admit_pending_render_state(state);
            require(route % 2 ? bool(recovered.printState) : bool(recovered.directState), "DIR recovery");
            const auto warm = admit_pending_render_state(state);
            require(warm.directState == recovered.directState && warm.printState == recovered.printState, "unchanged snapshot reuse");
            const auto recoveredView = JuicerTestSupport::published_view(recovered);
            auto unused = changed;
            if (route < 2)
                unused.dirCouplers.langmuirReceiverKRgb[0] = 2;
            else
                unused.dirCouplers.langmuirDonorKRgb[0] = 2;
            pending(unused);
            const auto unusedState = admit_pending_render_state(state);
            const auto unusedView = JuicerTestSupport::published_view(unusedState);
            require(unusedView.dir_recipe.hash == recoveredView.dir_recipe.hash, "unused captured K preserves family identity");
            require(fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "admission/recovery owns no temporary foreign graph");
            auto first = unused;
            first.dirCouplers.amount = 0.625f;
            auto latest = first;
            latest.dirCouplers.amount = 0.875f;
            pending(first);
            status(fj_test_film_recipe_fault(5, 3));
            RecipeSupersession selection{latest};
            set_pending_capture_test_hook(supersede_recipe_snapshot, &selection);
            const auto beforeSupersession = state.buildCounterNext.load();
            const auto superseded = admit_pending_render_state(state);
            set_pending_capture_test_hook(nullptr, nullptr);
            require(selection.fired && (route % 2 ? bool(superseded.printState) : bool(superseded.directState)) &&
                        hash_params(superseded.snapshot) == hash_params(latest) && state.lastHash.load() == hash_params(latest) &&
                        state.buildCounterNext.load() == beforeSupersession + 1,
                    "a superseded checked Rust DIR failure retries only the latest complete snapshot");
            require(fj_test_film_dir_live_owners() == 0, "supersession drains foreign DIR owners");
        }
        std::puts("PASS 64 independent complete products, four-route failure/admission/recovery and family reuse");
    }
} // namespace
namespace JuicerAssets::RecipeTest {
    void after_call(unsigned operation, FjStatus& status, FjFilmRecipeFailure& failure) {
        if (wireMutation.active && operation == wireMutation.operation) {
            status = wireMutation.status;
            failure = wireMutation.failure;
            wireMutation.active = false;
        }
    }
    void view(FjFilmDirView& view) {
        if (nextViewFault == 1)
            view.mode = 99;
        if (nextViewFault == 2)
            view.compensated_axes_rgb[0].count = SIZE_MAX;
        if (nextViewFault == 3)
            view.compensated_axes_rgb[0].data = reinterpret_cast<const float*>(std::uintptr_t{1});
        if (nextViewFault == 4) {
            view.mode = 0;
            view.hash = 0;
            view.compensated_axes_hash = 0;
            for (auto& axis : view.compensated_axes_rgb)
                axis = {nullptr, 0};
        }
        if (nextViewFault == 5)
            view.receiver_c_ref_rgb[0] = 1;
        nextViewFault = 0;
    }
} // namespace JuicerAssets::RecipeTest
namespace JuicerAssets::RecipeStateTest {
    void selected_source(std::shared_ptr<const FilmSource>& source) {
        // SF_TEMP_BRIDGE_NATIVE_DIR_SOURCE_DIAGNOSTIC has no publisher override;
        // the raw native comparison is covered directly in native_cases().
        (void)source;
    }
} // namespace JuicerAssets::RecipeStateTest
