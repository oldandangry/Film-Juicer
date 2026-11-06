namespace {
    FjVisualGrainParams grain_params(const Json& v) {
        FjVisualGrainParams out{};
        out.active = v.at("active").get<bool>() ? 1u : 0u;
        out.sublayers_active = v.at("sublayers_active").get<bool>() ? 1u : 0u;
        out.particle_area_um2 = std::bit_cast<float>(v.at("particle_area_um2")[0].get<std::uint32_t>());
        copy_words(v.at("particle_scale_cmy"), out.particle_scale_cmy);
        copy_words(v.at("particle_scale_layers"), out.particle_scale_layers);
        copy_words(v.at("particle_density_min_cmy"), out.particle_density_min_cmy);
        copy_words(v.at("uniformity_cmy"), out.uniformity_cmy);
        out.correlation_sigma_px = std::bit_cast<float>(v.at("correlation_sigma_px")[0].get<std::uint32_t>());
        out.dye_cloud_blur_um = std::bit_cast<float>(v.at("dye_cloud_blur_um")[0].get<std::uint32_t>());
        copy_words(v.at("micro_structure"), out.micro_structure);
        out.sublayer_count = v.at("sublayer_count").get<std::int32_t>();
        out.amplitude = std::bit_cast<float>(v.at("amplitude")[0].get<std::uint32_t>());
        out.chroma_mix = std::bit_cast<float>(v.at("chroma_mix")[0].get<std::uint32_t>());
        out.chroma_shared_weight = std::bit_cast<float>(v.at("chroma_shared_weight")[0].get<std::uint32_t>());
        out.chroma_independent_weight = std::bit_cast<float>(v.at("chroma_independent_weight")[0].get<std::uint32_t>());
        out.coarse_weight = std::bit_cast<float>(v.at("coarse_weight")[0].get<std::uint32_t>());
        out.mid_weight = std::bit_cast<float>(v.at("mid_weight")[0].get<std::uint32_t>());
        out.size_mix_scale = std::bit_cast<float>(v.at("size_mix_scale")[0].get<std::uint32_t>());
        out.clump_temporal_mix = std::bit_cast<float>(v.at("clump_temporal_mix")[0].get<std::uint32_t>());
        out.clump_morph_period_sec = std::bit_cast<float>(v.at("clump_morph_period_sec")[0].get<std::uint32_t>());
        out.debug_view = v.at("debug_view").get<std::int32_t>();
        return out;
    }
    VisualGrainControls native_grain_controls(const FjVisualGrainParams& v) {
        VisualGrainControls out;
        out.active = v.active != 0;
        out.sublayersActive = v.sublayers_active != 0;
        out.particleAreaUm2 = v.particle_area_um2;
        std::copy_n(v.particle_scale_cmy, 3, out.particleScaleCmy.begin());
        std::copy_n(v.particle_scale_layers, 3, out.particleScaleLayers.begin());
        std::copy_n(v.particle_density_min_cmy, 3, out.visualParticleDensityMinCmy.begin());
        std::copy_n(v.uniformity_cmy, 3, out.uniformityCmy.begin());
        out.correlationSigmaPx = v.correlation_sigma_px;
        out.dyeCloudBlurUm = v.dye_cloud_blur_um;
        std::copy_n(v.micro_structure, 2, out.microStructure.begin());
        out.nSubLayers = v.sublayer_count;
        out.amplitude = v.amplitude;
        out.chromaMix = v.chroma_mix;
        out.chromaSharedWeight = v.chroma_shared_weight;
        out.chromaIndependentWeight = v.chroma_independent_weight;
        out.coarseWeight = v.coarse_weight;
        out.midWeight = v.mid_weight;
        out.sizeMixScale = v.size_mix_scale;
        out.clumpTemporalMix = v.clump_temporal_mix;
        out.clumpMorphPeriodSec = v.clump_morph_period_sec;
        out.debugView = v.debug_view;
        return out;
    }
    Json project(const FjVisualGrainRecipe& v) {
        require(v.active <= 1 && v.sublayers_active <= 1, "closed grain result booleans");
        Json out{{"active", v.active != 0}, {"sublayers_active", v.sublayers_active != 0}};
        Json finite = Json::array(), prefix = Json::array();
        for (std::size_t channel = 0; channel < 3; ++channel) {
            require(v.axis_finite_cmy[channel] <= 1, "closed finite-channel word");
            finite.push_back(v.axis_finite_cmy[channel] != 0);
            prefix.push_back(words(v.axis_block_prefix_max_cmy[channel], 16));
        }
        out["axis_finite_cmy"] = finite;
        out["axis_block_prefix_max_cmy"] = prefix;
        out["particle_area_um2"] = words(&v.particle_area_um2, 1);
        out["particle_scale_cmy"] = words(v.particle_scale_cmy, 3);
        out["particle_scale_layers"] = words(v.particle_scale_layers, 3);
        out["particle_density_min_cmy"] = words(v.particle_density_min_cmy, 3);
        out["uniformity_cmy"] = words(v.uniformity_cmy, 3);
        out["correlation_sigma_px"] = words(&v.correlation_sigma_px, 1);
        out["dye_cloud_blur_um"] = words(&v.dye_cloud_blur_um, 1);
        out["micro_structure"] = words(v.micro_structure, 2);
        out["sublayer_count"] = v.sublayer_count;
        out["amplitude"] = words(&v.amplitude, 1);
        out["chroma_mix"] = words(&v.chroma_mix, 1);
        out["chroma_shared_weight"] = words(&v.chroma_shared_weight, 1);
        out["chroma_independent_weight"] = words(&v.chroma_independent_weight, 1);
        out["coarse_weight"] = words(&v.coarse_weight, 1);
        out["mid_weight"] = words(&v.mid_weight, 1);
        out["size_mix_scale"] = words(&v.size_mix_scale, 1);
        out["clump_temporal_mix"] = words(&v.clump_temporal_mix, 1);
        out["clump_morph_period_sec"] = words(&v.clump_morph_period_sec, 1);
        out["debug_view"] = v.debug_view;
        out["fine_weight"] = words(&v.fine_weight, 1);
        out["layers_hash"] = v.layers_hash;
        out["hash"] = v.hash;
        return out;
    }
    void seed_grain(FjVisualGrainRecipe& out) {
        out.active = 9;
        out.sublayers_active = 9;
        std::fill(std::begin(out.axis_finite_cmy), std::end(out.axis_finite_cmy), 9);
        for (auto& lane : out.axis_block_prefix_max_cmy)
            std::fill(std::begin(lane), std::end(lane), 1.0f);
        out.particle_area_um2 = 1.0f;
        std::fill(std::begin(out.particle_scale_cmy), std::end(out.particle_scale_cmy), 1.0f);
        std::fill(std::begin(out.particle_scale_layers), std::end(out.particle_scale_layers), 1.0f);
        std::fill(std::begin(out.particle_density_min_cmy), std::end(out.particle_density_min_cmy), 1.0f);
        std::fill(std::begin(out.uniformity_cmy), std::end(out.uniformity_cmy), 1.0f);
        out.correlation_sigma_px = 1.0f;
        out.dye_cloud_blur_um = 1.0f;
        std::fill(std::begin(out.micro_structure), std::end(out.micro_structure), 1.0f);
        out.sublayer_count = 9;
        out.amplitude = 1.0f;
        out.chroma_mix = 1.0f;
        out.chroma_shared_weight = 1.0f;
        out.chroma_independent_weight = 1.0f;
        out.fine_weight = 1.0f;
        out.mid_weight = 1.0f;
        out.coarse_weight = 1.0f;
        out.size_mix_scale = 1.0f;
        out.clump_temporal_mix = 1.0f;
        out.clump_morph_period_sec = 1.0f;
        out.debug_view = 9;
        out.layers_hash = 9;
        out.hash = 9;
    }
    bool cleared_grain(const FjVisualGrainRecipe& out, bool inactive = false) {
        const auto zero = [](float v) {
            return std::bit_cast<std::uint32_t>(v) == 0;
        };
        if (out.active != 0)
            return false;
        if (out.sublayers_active != 0)
            return false;
        if (!std::all_of(std::begin(out.axis_finite_cmy), std::end(out.axis_finite_cmy), [](std::uint32_t v) {
                return v == 0;
            }))
            return false;
        for (const auto& lane : out.axis_block_prefix_max_cmy)
            if (!std::all_of(std::begin(lane), std::end(lane), zero))
                return false;
        if (!zero(out.particle_area_um2))
            return false;
        if (!std::all_of(std::begin(out.particle_scale_cmy), std::end(out.particle_scale_cmy), zero))
            return false;
        if (!std::all_of(std::begin(out.particle_scale_layers), std::end(out.particle_scale_layers), zero))
            return false;
        if (!std::all_of(std::begin(out.particle_density_min_cmy), std::end(out.particle_density_min_cmy), zero))
            return false;
        if (!std::all_of(std::begin(out.uniformity_cmy), std::end(out.uniformity_cmy), zero))
            return false;
        if (!zero(out.correlation_sigma_px))
            return false;
        if (!zero(out.dye_cloud_blur_um))
            return false;
        if (!std::all_of(std::begin(out.micro_structure), std::end(out.micro_structure), zero))
            return false;
        if (out.sublayer_count != 0)
            return false;
        if (!zero(out.amplitude))
            return false;
        if (!zero(out.chroma_mix))
            return false;
        if (!zero(out.chroma_shared_weight))
            return false;
        if (!zero(out.chroma_independent_weight))
            return false;
        if (!zero(out.fine_weight))
            return false;
        if (!zero(out.mid_weight))
            return false;
        if (!zero(out.coarse_weight))
            return false;
        if (std::bit_cast<std::uint32_t>(out.size_mix_scale) != (inactive ? 0x3f800000u : 0u))
            return false;
        if (!zero(out.clump_temporal_mix))
            return false;
        if (!zero(out.clump_morph_period_sec))
            return false;
        if (out.debug_view != 0)
            return false;
        if (out.layers_hash != 0)
            return false;
        if (out.hash != 0)
            return false;
        return true;
    }
    struct GrainDevelopment {
        FjFilmDevelopmentOwner* owner = nullptr;
        bool facade;
        GrainDevelopment(const Assets& assets, const Json& row, bool layers, bool useFacade) : facade(useFacade) {
            const auto key = row.at("key").get<std::string>();
            FjFilmProfile* profile = nullptr;
            status(fj_legacy_film_profile_acquire(assets.value, {key.data(), key.size()}, &profile, nullptr));
            const auto acquired = (facade ? fj_test_film_development_acquire : fj_legacy_film_development_acquire)(profile, row.at("route").get<std::uint32_t>(), std::bit_cast<float>(row.at("gamma")[0].get<std::uint32_t>()), layers ? 1u : 0u, &owner, nullptr);
            const auto released = fj_legacy_film_profile_release(profile, nullptr);
            status(acquired);
            status(released);
        }
        ~GrainDevelopment() {
            if (owner && (facade ? fj_test_film_development_release : fj_legacy_film_development_release)(owner, nullptr).category != FJ_STATUS_SUCCESS)
                std::terminate();
        }
        GrainDevelopment(const GrainDevelopment&) = delete;
        GrainDevelopment& operator=(const GrainDevelopment&) = delete;
    };
    void grain_values(const std::filesystem::path& resources, const Json& captured, const Json& synthetic, const std::filesystem::path& fixtureDirectory) {
        require(captured.at("products").size() == 36 && captured.at("controls").size() == 758 && synthetic.at("products").size() == 28, "exact independent grain cohort membership");
        Assets assets(resources);
        for (bool facade : {false, true}) {
            const auto call = facade ? fj_test_visual_grain_prepare : fj_legacy_visual_grain_prepare;
            std::size_t product_index = 0;
            for (const auto& row : captured.at("products")) {
                JuicerTests::CaseEvent event(facade ? "c11" : "cpp", "grain/product-input/" + std::to_string(product_index++) + "/" + row.at("key").get<std::string>() + "/" + std::to_string(row.at("route").get<unsigned>()));
                auto params = grain_params(row.at("controls"));
                FjVisualGrainRecipe out{};
                {
                    GrainDevelopment development(assets, row, params.active && params.sublayers_active, facade);
                    const auto holds = fj_test_film_development_live_owners();
                    status(fj_legacy_assets_release_cached_payloads(assets.value, nullptr));
                    FjFilmRecipeFailure failure{};
                    status(call(development.owner, &params, &out, &failure, nullptr));
                    metadata(failure);
                    require(fj_test_film_development_live_owners() == holds, "grain does not retain/release development");
                }
                require(project(out) == row.at("grain"), "all fixed grain fields survive source/cache/development release");
                event.passed();
            }
            GrainDevelopment withLayers(assets, captured.at("products")[2], true, facade);
            GrainDevelopment withoutLayers(assets, captured.at("products")[2], false, facade);
            std::size_t control_index = 0;
            for (const auto& row : captured.at("controls")) {
                JuicerTests::CaseEvent event(facade ? "c11" : "cpp", "grain/control/" + std::to_string(control_index++) + "/" + row.at("id").get<std::string>());
                const auto params = grain_params(row.at("controls"));
                auto* development = row.at("source_layers_hash").get<std::uint64_t>() == 0 ? withoutLayers.owner : withLayers.owner;
                FjVisualGrainRecipe out{};
                seed_grain(out);
                FjFilmRecipeFailure failure{99, 99, 99};
                const auto result = call(development, &params, &out, &failure, nullptr);
                if (row.at("success").get<bool>()) {
                    status(result);
                    metadata(failure);
                    require(project(out) == row.at("grain"), "758 exact native grain controls, weights, signed zero and raw hash stream");
                } else {
                    status(result, FJ_STATUS_PREPARATION_FAILURE);
                    metadata(failure, row.at("id") == "missing-layers" ? FJ_FILM_RECIPE_GRAIN_LAYERS : FJ_FILM_RECIPE_GRAIN_CONTROLS);
                    require(cleared_grain(out), "every declared field cleared after control/layer failure");
                }
                event.passed();
            }
            std::size_t synthetic_index = 0;
            for (const auto& row : synthetic.at("products")) {
                JuicerTests::CaseEvent event(facade ? "c11" : "cpp", "grain/synthetic/" + std::to_string(synthetic_index++) + "/" + row.at("source").get<std::string>());
                Assets source(fixtureDirectory / "synthetic" / row.at("source").get<std::string>());
                GrainDevelopment development(source, row, true, facade);
                FjFilmDevelopmentView view{};
                status((facade ? fj_test_film_development_view : fj_legacy_film_development_view)(development.owner, &view, nullptr));
                require(std::all_of(view.authored_curves_cmy.data, view.authored_curves_cmy.data + view.authored_curves_cmy.count, [](float v) {
                            return std::isfinite(v);
                        }),
                        "real completed profile has finite sampled values");
                require(std::any_of(view.normalized_curves_cmy.data, view.normalized_curves_cmy.data + view.normalized_curves_cmy.count, [](float v) {
                            return !std::isfinite(v);
                        }) == row.at("nonfinite_normalized").get<bool>(),
                        "genuine nonfinite normalization remains admissible");
                const auto params = grain_params(row.at("controls"));
                FjVisualGrainRecipe out{};
                FjFilmRecipeFailure failure{};
                status(call(development.owner, &params, &out, &failure, nullptr));
                metadata(failure);
                require(project(out) == row.at("grain"), "28 genuine variable-row/polarity/overflow projections");
                event.passed();
            }
        }
        JuicerTests::CaseEvent drain_event("cpp", "grain/owner-drain");
        require(fj_test_film_development_live_owners() == 0, "all grain fixture holds drain");
        drain_event.passed();
    }
    void grain_failures(const std::filesystem::path& resources, const Json& captured) {
        Assets assets(resources);
        std::vector<std::string> capacity_ids, fault_ids;
        for (unsigned caller = 0; caller < 2; ++caller) {
            for (const auto capacity : {0u, 1u, 5u, 256u})
                capacity_ids.emplace_back("caller-" + std::to_string(caller) + "/capacity-" + std::to_string(capacity));
            for (unsigned fault : {1u, 2u})
                fault_ids.emplace_back("caller-" + std::to_string(caller) + "/fault-" + std::to_string(fault));
        }
        JuicerTests::Cohort diagnostic_capacities("visual_grain.diagnostic_capacity", capacity_ids);
        JuicerTests::Cohort recipe_faults("visual_grain.fault_tuple", fault_ids);
        unsigned caller_index = 0;
        for (bool facade : {false, true}) {
            JuicerTests::CaseEvent event(facade ? "c11" : "cpp", "grain/failures");
            GrainDevelopment development(assets, captured.at("products")[2], true, facade);
            GrainDevelopment noLayers(assets, captured.at("products")[2], false, facade);
            const auto base = grain_params(captured.at("products")[2].at("controls"));
            const auto call = facade ? fj_test_visual_grain_prepare : fj_legacy_visual_grain_prepare;
            FjVisualGrainRecipe out{};
            FjFilmRecipeFailure failure{};
            const FjStatus unsupported{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
            const FjStatus preparationFailure{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
            const FjStatus internalFailure{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
            const auto rejected = [&](const FjFilmDevelopmentOwner* owner, const FjVisualGrainParams* params, FjStatus expected, std::uint32_t reason, FjErrorBuffer* error = nullptr) {
                seed_grain(out);
                failure = {99, 99, 99};
                const auto value = call(owner, params, &out, &failure, error);
                require(value.category == expected.category && value.api == expected.api && value.native_code == expected.native_code, "complete rejected grain status envelope");
                metadata(failure, reason);
                require(cleared_grain(out), "seeded grain failure clears every field, including all 48 metadata words and scale");
            };
            auto params = base;
            for (auto word : {2u, UINT32_MAX}) {
                params = base;
                params.active = word;
                rejected(development.owner, &params, unsupported, FJ_FILM_RECIPE_RAW);
                params = base;
                params.active = 0;
                params.sublayers_active = word;
                rejected(development.owner, &params, unsupported, FJ_FILM_RECIPE_RAW);
            }
            rejected(nullptr, &base, unsupported, FJ_FILM_RECIPE_RAW);
            rejected(development.owner, nullptr, unsupported, FJ_FILM_RECIPE_RAW);
            params = base;
            params.particle_area_um2 = 0;
            rejected(noLayers.owner, &params, preparationFailure, FJ_FILM_RECIPE_GRAIN_CONTROLS);
            rejected(noLayers.owner, &base, preparationFailure, FJ_FILM_RECIPE_GRAIN_LAYERS);
            for (std::size_t capacity : {0u, 1u, 5u, 256u}) {
                auto row_event = diagnostic_capacities.case_event("caller-" + std::to_string(caller_index) + "/capacity-" + std::to_string(capacity));
                std::array<char, 256> text{};
                text.fill('Z');
                FjErrorBuffer error{capacity ? text.data() : nullptr, capacity, 99};
                rejected(development.owner, &params, preparationFailure, FJ_FILM_RECIPE_GRAIN_CONTROLS, &error);
                const auto length = std::string("film recipe reason=17 channel=3 sample=18446744073709551615").size();
                require(error.length == (capacity ? std::min(capacity - 1, length) : 0) && (!capacity || text[error.length] == '\0'), "bounded grain diagnostic exact length/NUL");
                require(std::all_of(text.begin() + static_cast<std::ptrdiff_t>(capacity), text.end(), [](char c) {
                            return c == 'Z';
                        }),
                        "diagnostic stays in its admitted backing extent");
                row_event.passed();
            }
            FjErrorBuffer malformed{nullptr, 1, 99};
            rejected(development.owner, &params, unsupported, FJ_FILM_RECIPE_RAW, &malformed);
            require(malformed.length == 0, "malformed diagnostic admission clears admitted length");
            seed_grain(out);
            status(call(development.owner, &base, &out, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(cleared_grain(out), "independent result clearing with inadmissible failure output");
            status(call(development.owner, &base, nullptr, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_FILM_RECIPE_RAW);
            alignas(FjVisualGrainParams) std::array<std::byte, sizeof(FjVisualGrainParams) + 8> inputStorage{};
            rejected(development.owner, reinterpret_cast<const FjVisualGrainParams*>(inputStorage.data() + 1), unsupported, FJ_FILM_RECIPE_RAW);
            alignas(FjVisualGrainRecipe) std::array<std::byte, sizeof(FjVisualGrainRecipe) + 8> outputStorage{};
            outputStorage.fill(std::byte{0x7f});
            status(call(development.owner, &base, reinterpret_cast<FjVisualGrainRecipe*>(outputStorage.data() + 1), &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            metadata(failure, FJ_FILM_RECIPE_RAW);
            require(std::all_of(outputStorage.begin(), outputStorage.end(), [](std::byte v) {
                        return v == std::byte{0x7f};
                    }),
                    "inadmissible output is never dereferenced");
            alignas(FjFilmRecipeFailure) std::array<std::byte, sizeof(FjFilmRecipeFailure) + 8> failureStorage{};
            seed_grain(out);
            status(call(development.owner, &base, &out, reinterpret_cast<FjFilmRecipeFailure*>(failureStorage.data() + 1), nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(cleared_grain(out), "misaligned failure slot does not prevent independent output clearing");
            for (unsigned fault : {1u, 2u}) {
                auto row_event = recipe_faults.case_event("caller-" + std::to_string(caller_index) + "/fault-" + std::to_string(fault));
                status(fj_test_film_recipe_fault(7, fault));
                rejected(development.owner, &base, fault == 1 ? internalFailure : preparationFailure, fault == 1 ? FJ_FILM_RECIPE_PANIC : FJ_FILM_RECIPE_IDENTITY);
                row_event.passed();
            }
            status(fj_test_film_recipe_fault(7, 1));
            status(call(development.owner, &base, nullptr, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            status(call(development.owner, &base, &out, &failure, nullptr));
            metadata(failure);
            require(project(out) == captured.at("products")[2].at("grain"), "entry fault is consumed even if output admission rejects the call");
            params = base;
            params.active = 0;
            params.particle_area_um2 = std::numeric_limits<float>::quiet_NaN();
            status(call(noLayers.owner, &params, &out, &failure, nullptr));
            metadata(failure);
            require(cleared_grain(out, true), "inactive ignores numeric/layer work and explicitly restores scale 1");
            const auto holds = fj_test_film_development_live_owners();
            std::atomic<bool> concurrentOk{true};
            std::array<std::thread, 4> callers;
            for (auto& thread : callers)
                thread = std::thread([&]() {
                    try {
                        for (int repeat = 0; repeat < 32; ++repeat) {
                            FjVisualGrainRecipe value{};
                            FjFilmRecipeFailure metadata{};
                            const auto result = call(development.owner, &base, &value, &metadata, nullptr);
                            if (result.category != FJ_STATUS_SUCCESS || result.api != FJ_API_NONE || result.native_code != 0 || metadata.reason != FJ_FILM_RECIPE_REASON_NONE || metadata.channel != 3 || metadata.sample != UINT64_MAX || project(value) != captured.at("products")[2].at("grain"))
                                concurrentOk.store(false);
                        }
                    } catch (...) {
                        concurrentOk.store(false);
                    }
                });
            for (auto& thread : callers)
                thread.join();
            require(concurrentOk.load() && fj_test_film_development_live_owners() == holds, "concurrent fixed results borrow one shared immutable live development");
            event.passed();
            ++caller_index;
        }
        require(fj_test_film_development_live_owners() == 0, "grain failure/concurrency holds drain");
    }
    void grain_products(const std::filesystem::path& resources, const Json& captured) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        std::size_t product_index = 0;
        for (const auto& row : captured.at("products")) {
            JuicerTests::CaseEvent event("cpp", "grain/product/" + std::to_string(product_index++) + "/" + row.at("key").get<std::string>() + "/" + std::to_string(row.at("route").get<unsigned>()));
            ParamSnapshot params;
            params.filmProfileKey = row.at("key").get<std::string>();
            params.printProfileKey = "kodak_portra_endura";
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(row.at("route").get<unsigned>());
            params.filmGammaFactor = std::bit_cast<float>(row.at("gamma")[0].get<std::uint32_t>());
            params.spectralUpsamplingMode = 0;
            params.cameraAutoExposureEnabled = 0;
            params.inputCompressionEnabled = 0;
            params.outputGamutCompressionEnabled = 0;
            params.grainControls = native_grain_controls(grain_params(row.at("controls")));
            auto product = JuicerTestSupport::prepare_product(params);
            FjPreparedStateView view{};
            FjErrorBuffer error{};
            status(product.view(view, error));
            require(project(view.grain) == row.at("grain"), "all complete direct/print grain values and metadata");
            require(hash_params(params) == row.at("snapshot_hash") && view.recipe_hash == row.at("final_hash") && view.film_development.hash == row.at("development_hash"), "36 exact native snapshot/development/final identities");
            if (row.at("route") != 0 || row.at("gamma")[0] != 1065353216u || !params.grainControls.active || !params.grainControls.sublayersActive) {
                event.passed();
                continue;
            }
            const auto originalGrainHash = view.grain.hash;
            const auto originalBoundsHash = view.scanner_bounds.hash;
            auto* const originalMinimum = view.grain_minimum.cmy;
            auto changed = params;
            changed.grainControls.visualParticleDensityMinCmy[0] = 0.5f;
            auto changedProduct = JuicerTestSupport::prepare_product(changed);
            FjPreparedStateView changedView{};
            status(changedProduct.view(changedView, error));
            require(changedView.grain.hash != originalGrainHash && changedView.scanner_bounds.hash == originalBoundsHash && std::equal(changedView.grain_minimum.cmy, changedView.grain_minimum.cmy + 3, originalMinimum), "visual particle minimum remains distinct from scanner GrainContract");
            changed = params;
            changed.grainControls.chromaMix = 0.5f;
            require(hash_params(changed) == hash_params(params), "synthetic chroma-only change retains intentional snapshot omission");
            auto chromaProduct = JuicerTestSupport::prepare_product(changed);
            FjPreparedStateView chromaView{};
            status(chromaProduct.view(chromaView, error));
            require(chromaView.grain.hash != originalGrainHash, "resolved recipe independently includes chroma mix");
            event.passed();
        }
        for (unsigned route = 0; route < 4; ++route) {
            JuicerTests::CaseEvent event("cpp", "grain/publication/" + std::to_string(route));
            ParamSnapshot params;
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.printProfileKey = "kodak_portra_endura";
            params.grainControls.active = true;
            params.spectralUpsamplingMode = 0;
            params.cameraAutoExposureEnabled = 0;
            params.inputCompressionEnabled = 0;
            params.outputGamutCompressionEnabled = 0;
            RenderInstance state;
            JuicerTestSupport::capture(state, params);
            const auto admitted = JuicerTestSupport::admit(state);
            require(admitted.kind == FJ_ADMISSION_READY && admitted.state, "grain publication baseline");
            const auto held = JuicerTestSupport::published_view(admitted);
            const auto hash = held.recipe_hash;
            const auto baselineMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
            auto changed = params;
            changed.grainControls.particleAreaUm2 = 0;
            JuicerTestSupport::capture(state, changed);
            const auto rejected = JuicerTestSupport::admit(state);
            const auto failedMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
            require(rejected.delivery.category == FJ_STATUS_PREPARATION_FAILURE &&
                        rejected.kind == FJ_ADMISSION_PREPARATION_FAILED && !rejected.state &&
                        rejected.diagnostic == "MalformedRequiredProfileData phase=9A field=visual_grain" &&
                        failedMetrics.build == baselineMetrics.build &&
                        failedMetrics.publishedIdentity == baselineMetrics.publishedIdentity &&
                        failedMetrics.backingIdentity == baselineMetrics.backingIdentity && held.recipe_hash == hash,
                    "ordinary grain failure never publishes partial state or revokes admitted holds");
            JuicerTestSupport::capture(state, params);
            const auto recovered = JuicerTestSupport::admit(state);
            require(recovered.kind == FJ_ADMISSION_READY && recovered.state &&
                        JuicerTestSupport::PublicationObserver::metrics(state).build == baselineMetrics.build,
                    "grain recovery reuses last complete state");
            const auto repeat = JuicerTestSupport::admit(state);
            require(repeat.kind == FJ_ADMISSION_READY && repeat.state &&
                        JuicerTestSupport::PublicationObserver::backing(*repeat.state) == baselineMetrics.backingIdentity &&
                        JuicerTestSupport::PublicationObserver::metrics(state).build == baselineMetrics.build,
                    "unchanged grain publication reuses state");
            event.passed();
        }
        JuicerTests::CaseEvent drain_event("cpp", "grain/product-owner-drain");
        require(fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "all whole-state temporary owners drain");
        drain_event.passed();
        std::puts("PASS 36 independent grain products, four-route publication/recovery, minimum separation and stage order");
    }
} // namespace
