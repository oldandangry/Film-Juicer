#include <exception>

namespace {
    struct RecipeSupersession {
        RenderInstance* instance = nullptr;
        ParamSnapshot replacement;
        bool fired = false;
        std::exception_ptr failure;
    };
    void supersede_recipe_snapshot(void* context) noexcept {
        auto& selection = *static_cast<RecipeSupersession*>(context);
        if (selection.fired)
            return;
        try {
            JuicerTestSupport::capture(*selection.instance, selection.replacement);
            selection.fired = true;
        } catch (...) {
            selection.failure = std::current_exception();
        }
    }
    struct RecipePrepareHook final {
        explicit RecipePrepareHook(RecipeSupersession& selection) {
            fj_test_instance_prepare_hooks(nullptr, supersede_recipe_snapshot, &selection);
        }
        RecipePrepareHook(const RecipePrepareHook&) = delete;
        RecipePrepareHook& operator=(const RecipePrepareHook&) = delete;
        ~RecipePrepareHook() {
            fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
        }
    };

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
            JuicerTests::CaseEvent case_event("cpp", "film/default/raw/" + row.at("case").get<std::string>() + "/" + std::to_string(row.at("kind").get<unsigned>()));
            const auto name = row.at("case").get<std::string>();
            const auto root = (name == "duplicates" || name == "singleton") ? fixture.parent_path().parent_path() / "fixtures/synthetic" / name : fixture.parent_path() / "synthetic" / name;
            Assets assets(root);
            auto* development = assets.development(captured.at("cases")[2]);
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
            const auto result = fj_legacy_film_dir_acquire(development, &raw, &owner, &failure, nullptr);
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
            status(fj_legacy_film_development_release(development, nullptr));
            require(fj_test_film_dir_live_owners() == 0, "all F1 raw DIR owners drain after each fixture row");
            case_event.passed();
        }
        require(fj_test_film_development_live_owners() == 0, "all F1 development holds released");
    }

    void products(const std::filesystem::path& resources, const Json& captured) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        for (const auto& row : captured.at("cases")) {
            JuicerTests::CaseEvent case_event("cpp", film_case_id("film/products", row));
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
            case_event.passed();
        }
        JuicerTests::Cohort route_admissions("film_dir.admission_route", 4);
        for (unsigned route = 0; route < 4; ++route) {
            auto route_event = route_admissions.case_event(std::to_string(route));
            ParamSnapshot controls;
            controls.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            controls.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            controls.printProfileKey = "kodak_portra_endura";
            controls.spectralUpsamplingMode = 1;
            controls.cameraAutoExposureEnabled = 0;
            controls.inputCompressionEnabled = 0;
            controls.outputGamutCompressionEnabled = 0;
            controls.grainControls.active = false;
            RenderInstance state;
            JuicerTestSupport::capture(state, controls);
            const auto admitted = JuicerTestSupport::admit(state);
            require(admitted.kind == FJ_ADMISSION_READY && admitted.state,
                    "complete DIR admission baseline");
            const auto original = JuicerTestSupport::published_view(admitted);
            require(original.route == route, "DIR publication retains the selected route");
            const auto originalMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
            const auto counter = originalMetrics.build;
            const auto oldHash = original.recipe_hash;
            auto changed = controls;
            changed.dirCouplers.active = true;
            changed.dirCouplers.amount = 0.5f;
            JuicerTestSupport::capture(state, changed);
            status(fj_test_film_recipe_fault(5, 3));
            const auto rejected = JuicerTestSupport::admit(state);
            require(rejected.kind == FJ_ADMISSION_PREPARATION_FAILED && !rejected.state &&
                        rejected.diagnostic == "MalformedRequiredProfileData phase=3D-3 field=dirCouplers",
                    "checked Rust DIR producer failure blocks complete-state admission");
            const auto failedMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
            require(failedMetrics.inputIdentity == hash_params(changed) && failedMetrics.build == counter &&
                        failedMetrics.publishedIdentity == originalMetrics.publishedIdentity &&
                        failedMetrics.backingIdentity == originalMetrics.backingIdentity && original.recipe_hash == oldHash,
                    "failed current DIR preparation blocks admission while retaining the prior complete publication");
            require(fj_test_film_dir_live_owners() == 0, "failed Rust DIR construction retains no foreign DIR owner");
            JuicerTestSupport::capture(state, controls);
            const auto reusedA = JuicerTestSupport::admit(state);
            require(reusedA.kind == FJ_ADMISSION_READY && reusedA.state && reusedA.state->build() == counter &&
                        JuicerTestSupport::PublicationObserver::backing(*reusedA.state) == originalMetrics.backingIdentity,
                    "restoring A reuses its retained publication after B fails");
            JuicerTestSupport::capture(state, changed);
            const auto recovered = JuicerTestSupport::admit(state);
            require(recovered.kind == FJ_ADMISSION_READY && recovered.state && recovered.state->build() == counter + 1,
                    "DIR recovery commits B once");
            const auto warm = JuicerTestSupport::admit(state);
            require(warm.kind == FJ_ADMISSION_READY && warm.state &&
                        JuicerTestSupport::PublicationObserver::backing(*warm.state) == JuicerTestSupport::PublicationObserver::backing(*recovered.state) &&
                        warm.state->build() == recovered.state->build(),
                    "unchanged snapshot reuse");
            const auto recoveredView = JuicerTestSupport::published_view(recovered);
            auto unused = changed;
            if (route < 2)
                unused.dirCouplers.langmuirReceiverKRgb[0] = 2;
            else
                unused.dirCouplers.langmuirDonorKRgb[0] = 2;
            JuicerTestSupport::capture(state, unused);
            const auto unusedState = JuicerTestSupport::admit(state);
            require(unusedState.kind == FJ_ADMISSION_READY && unusedState.state, "unused captured K admission");
            const auto unusedView = JuicerTestSupport::published_view(unusedState);
            require(unusedView.dir_recipe.hash == recoveredView.dir_recipe.hash, "unused captured K preserves family identity");
            require(fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "admission/recovery owns no temporary foreign graph");
            auto first = unused;
            first.dirCouplers.amount = 0.625f;
            auto latest = first;
            latest.dirCouplers.amount = 0.875f;
            JuicerTestSupport::capture(state, first);
            status(fj_test_film_recipe_fault(5, 3));
            RecipeSupersession selection{&state, latest, false, {}};
            const auto beforeSupersession = JuicerTestSupport::PublicationObserver::metrics(state).build;
            JuicerTestSupport::PublicationAdmission superseded;
            {
                RecipePrepareHook hook(selection);
                superseded = JuicerTestSupport::admit(state);
            }
            if (selection.failure)
                std::rethrow_exception(selection.failure);
            require(selection.fired && superseded.kind == FJ_ADMISSION_READY && superseded.state &&
                        superseded.state->input_identity() == hash_params(latest) &&
                        superseded.state->build() == beforeSupersession + 1,
                    "a superseded checked Rust DIR failure retries only the latest complete snapshot");
            require(fj_test_film_dir_live_owners() == 0, "supersession drains foreign DIR owners");
            route_event.passed();
        }
        std::puts("PASS 64 independent complete products, four-route failure/admission/recovery and family reuse");
    }
} // namespace
