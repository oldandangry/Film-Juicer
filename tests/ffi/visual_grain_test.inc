namespace {
    thread_local unsigned nextGrainProjectionFault = 0;
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
    Json native_grain_projection(const VisualGrainRecipe& v) {
        Json out{{"active", v.active}, {"sublayers_active", v.sublayersActive}, {"axis_finite_cmy", v.grainLayerAxisFinite}};
        Json prefix = Json::array();
        for (const auto& lane : v.grainLayerAxisBlockPrefixMax)
            prefix.push_back(words(lane.data(), 16));
        out["axis_block_prefix_max_cmy"] = prefix;
        out["particle_area_um2"] = words(&v.particleAreaUm2, 1);
        out["particle_scale_cmy"] = words(v.particleScaleCmy.data(), 3);
        out["particle_scale_layers"] = words(v.particleScaleLayers.data(), 3);
        out["particle_density_min_cmy"] = words(v.visualParticleDensityMinCmy.data(), 3);
        out["uniformity_cmy"] = words(v.uniformityCmy.data(), 3);
        out["correlation_sigma_px"] = words(&v.correlationSigmaPx, 1);
        out["dye_cloud_blur_um"] = words(&v.dyeCloudBlurUm, 1);
        out["micro_structure"] = words(v.microStructure.data(), 2);
        out["sublayer_count"] = v.nSubLayers;
        out["amplitude"] = words(&v.amplitude, 1);
        out["chroma_mix"] = words(&v.chromaMix, 1);
        out["chroma_shared_weight"] = words(&v.chromaSharedWeight, 1);
        out["chroma_independent_weight"] = words(&v.chromaIndependentWeight, 1);
        out["coarse_weight"] = words(&v.coarseWeight, 1);
        out["mid_weight"] = words(&v.midWeight, 1);
        out["size_mix_scale"] = words(&v.sizeMixScale, 1);
        out["clump_temporal_mix"] = words(&v.clumpTemporalMix, 1);
        out["clump_morph_period_sec"] = words(&v.clumpMorphPeriodSec, 1);
        out["debug_view"] = v.debugView;
        out["fine_weight"] = words(&v.fineWeight, 1);
        out["layers_hash"] = v.densityCurvesLayersHash;
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
            for (const auto& row : captured.at("products")) {
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
            }
            GrainDevelopment withLayers(assets, captured.at("products")[2], true, facade);
            GrainDevelopment withoutLayers(assets, captured.at("products")[2], false, facade);
            for (const auto& row : captured.at("controls")) {
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
            }
            for (const auto& row : synthetic.at("products")) {
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
            }
        }
        require(fj_test_film_development_live_owners() == 0, "all grain fixture holds drain");
    }
    void grain_failures(const std::filesystem::path& resources, const Json& captured) {
        Assets assets(resources);
        for (bool facade : {false, true}) {
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
                status(fj_test_film_recipe_fault(7, fault));
                rejected(development.owner, &base, fault == 1 ? internalFailure : preparationFailure, fault == 1 ? FJ_FILM_RECIPE_PANIC : FJ_FILM_RECIPE_IDENTITY);
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
        }
        require(fj_test_film_development_live_owners() == 0, "grain failure/concurrency holds drain");
    }
    void grain_native_bridge(const std::filesystem::path& resources, const Json& captured, const Json& synthetic, const std::filesystem::path& fixtureDirectory) {
        JuicerAssets::AssetBridge assets(resources);
        require(assets.load_catalog().valid, "grain native catalog");
        const auto source = assets.film_source("kodak_portra_400");
        auto development = source->prepare_development(Spektrafilm::ScanRoute::NegativeDirectScan, 1.0f, true);
        const auto controls = native_grain_controls(grain_params(captured.at("products")[2].at("controls")));
        VisualGrainRecipe out;
        std::string diagnostic;
        const auto holds = fj_test_film_development_live_owners();
        status(assets.release_cached_payloads());
        require(JuicerAssets::prepare_visual_grain(development.hold, controls, out, diagnostic) && diagnostic.empty(), "native grain bridge completes");
        require(native_grain_projection(out) == captured.at("products")[2].at("grain") && fj_test_film_development_live_owners() == holds, "all native mapping anchors and unchanged holds");
        SpectralAllocationProbe::arm();
        const bool prepared = JuicerAssets::prepare_visual_grain(development.hold, controls, out, diagnostic);
        SpectralAllocationProbe::clear();
        require(prepared && SpectralAllocationProbe::bytes() == 0, "native fixed projection introduces no C++ table copy/allocation");
        DirCouplersRecipe dir;
        DensityBoundsRecipe bounds;
        require(JuicerAssets::prepare_film_dir(development.hold, DirCouplersControls{}, *source->profile(), development.route, development.recipe.logExposure.size(), dir, diagnostic) &&
                    JuicerAssets::prepare_film_density_bounds(development.hold, GrainContract{}, *source->profile(), development.route, bounds, diagnostic) && fj_test_film_development_live_owners() == holds,
                "same retained Development survives cache detach and grain then DIR/bounds without a new hold");
        for (const auto& row : synthetic.at("products")) {
            JuicerAssets::AssetBridge selected(fixtureDirectory / "synthetic" / row.at("source").get<std::string>());
            require(selected.load_catalog().valid, "native variable-row catalog");
            const auto film = selected.film_source("kodak_portra_400");
            auto developed = film->prepare_development(static_cast<Spektrafilm::ScanRoute>(row.at("route").get<unsigned>()), 1.0f, true);
            VisualGrainRecipe grain;
            require(JuicerAssets::prepare_visual_grain(developed.hold, controls, grain, diagnostic) && native_grain_projection(grain) == row.at("grain"), "native mapping retains non-256 zeros and legitimate nonfinite partial prefixes");
        }
        auto bad = controls;
        bad.particleAreaUm2 = 0;
        require(!JuicerAssets::prepare_visual_grain(development.hold, bad, out, diagnostic) && diagnostic == "MalformedRequiredProfileData phase=9A field=visual_grain" && !out.active && out.sizeMixScale == 1, "ordinary grain failure preserves exact diagnostic and native defaults");
        for (unsigned fault = 1; fault <= 6; ++fault) {
            nextGrainProjectionFault = fault;
            bool rejected = false;
            try {
                (void)JuicerAssets::prepare_visual_grain(development.hold, controls, out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& e) {
                rejected = e.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected && !out.active && out.hash == 0 && out.sizeMixScale == 1, "malformed success projection is internal and cannot publish partial native coefficients");
        }
        for (unsigned fault = 0; fault < 8; ++fault) {
            wireMutation = WireMutation{};
            wireMutation.active = true;
            wireMutation.operation = 4;
            wireMutation.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
            wireMutation.failure = {FJ_FILM_RECIPE_GRAIN_CONTROLS, 3, UINT64_MAX};
            if (fault == 0)
                wireMutation.status.category = FJ_STATUS_SUCCESS;
            if (fault == 1)
                wireMutation.status.api = FJ_API_CUDA_RUNTIME;
            if (fault == 2)
                wireMutation.status.native_code = 1;
            if (fault == 3)
                wireMutation.failure.channel = 0;
            if (fault == 4)
                wireMutation.failure.sample = 0;
            if (fault == 5)
                wireMutation.failure.reason = FJ_FILM_RECIPE_LOG_EXPOSURE;
            if (fault == 6)
                wireMutation.failure.reason = FJ_FILM_RECIPE_ALLOCATION;
            if (fault == 7)
                wireMutation.failure.reason = 99;
            bool rejected = false;
            try {
                (void)JuicerAssets::prepare_visual_grain(development.hold, controls, out, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& e) {
                rejected = e.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected && !wireMutation.active, "grain admits only its operation-specific complete status/failure envelope");
        }
        for (unsigned reason : {FJ_FILM_RECIPE_GRAIN_CONTROLS, FJ_FILM_RECIPE_GRAIN_LAYERS})
            for (unsigned operation : {0u, 1u, 2u, 3u}) {
                wireMutation = WireMutation{};
                wireMutation.active = true;
                wireMutation.operation = operation;
                wireMutation.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
                wireMutation.failure = {reason, 3, UINT64_MAX};
                bool rejected = false;
                try {
                    if (operation == 0) {
                        GrainContract value;
                        (void)JuicerAssets::prepare_film_density_minimum(GrainContract{}, value, diagnostic);
                    }
                    if (operation == 1 || operation == 2) {
                        DirCouplersRecipe value;
                        (void)JuicerAssets::prepare_film_dir(development.hold, DirCouplersControls{}, *source->profile(), development.route, development.recipe.logExposure.size(), value, diagnostic);
                    }
                    if (operation == 3) {
                        DensityBoundsRecipe value;
                        (void)JuicerAssets::prepare_film_density_bounds(development.hold, GrainContract{}, *source->profile(), development.route, value, diagnostic);
                    }
                } catch (const JuicerCuda::ExecutionFailure& e) {
                    rejected = e.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
                }
                require(rejected && !wireMutation.active && fj_test_film_dir_live_owners() == 0, "grain reasons 17/18 never become valid for minimum/DIR/bounds");
            }
    }
    void grain_products(const std::filesystem::path& resources, const Json& captured) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerProcess::root().ensure_bootstrap();
        for (const auto& row : captured.at("products")) {
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
            if (row.at("route") != 0 || row.at("gamma")[0] != 1065353216u || !params.grainControls.active || !params.grainControls.sublayersActive)
                continue;
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
        }
        JuicerAssets::AssetBridge sourceAssets(resources);
        require(sourceAssets.load_catalog().valid, "foundation ordering source catalog");
        const auto source = sourceAssets.film_source("kodak_portra_400");
        const auto illuminants = JuicerProcess::root().assets().illuminant_filter_curves();
        for (int competing = 0; competing < 4; ++competing) {
            Spektrafilm::DirectRecipeBuildInput input;
            auto developed = source->prepare_development(Spektrafilm::ScanRoute::NegativeDirectScan, 1.0f, true);
            input.film.filmProfileKey = "kodak_portra_400";
            input.film.filmSource = source.get();
            input.film.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
            input.film.filmDevelop = std::move(developed.recipe);
            input.film.developmentHold = std::move(developed.hold);
            input.film.spectralUpsamplingMode = 1;
            input.film.inputCompressionRequested = false;
            input.film.cameraAutoExposureEnabled = false;
            std::copy_n(illuminants->d55.linear.begin(), 81, input.film.referenceIlluminant.begin());
            input.film.referenceIlluminantValid = true;
            input.film.visualGrain.active = true;
            input.film.visualGrain.particleAreaUm2 = 0;
            input.film.dirCouplers.amount = -1;
            const char* expected = "MalformedRequiredProfileData phase=9A field=visual_grain";
            if (competing == 1) {
                input.film.inputColorSpace = -1;
                expected = "UnsupportedMode phase=3B field=input_color_space";
            }
            if (competing == 2) {
                input.film.spatialOptics.cameraLensBlurUm = std::numeric_limits<float>::quiet_NaN();
                expected = "ResourceDescriptorMismatch phase=6A component=camera_lens_blur field=sigma_um";
            }
            if (competing == 3) {
                input.film.grainContract.densityMinCmy[0] = -1;
                expected = "MalformedRequiredProfileData phase=9A field=grain_contract";
            }
            const auto result = Spektrafilm::build_direct_render_recipe(std::move(input));
            require(!result.valid && result.diagnostic == expected, "input/optics/minimum/grain/DIR retain first-failure order");
        }
        for (unsigned route = 0; route < 4; ++route) {
            ParamSnapshot params;
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.printProfileKey = "kodak_portra_endura";
            params.grainControls.active = true;
            params.spectralUpsamplingMode = 0;
            params.cameraAutoExposureEnabled = 0;
            params.inputCompressionEnabled = 0;
            params.outputGamutCompressionEnabled = 0;
            InstanceState state;
            const auto pending = [&](const ParamSnapshot& v) {
                std::lock_guard<std::mutex> lock(state.pending.m);
                state.pending.value = PendingParamsState::Valid{v, hash_params(v)};
            };
            pending(params);
            const auto admitted = admit_pending_render_state(state);
            require(route % 2 ? bool(admitted.printState) : bool(admitted.directState), "grain publication baseline");
            const auto& held = route % 2 ? admitted.printState->recipe : admitted.directState->recipe;
            const auto hash = held.hash;
            const auto counter = state.buildCounterNext.load();
            auto changed = params;
            changed.grainControls.particleAreaUm2 = 0;
            pending(changed);
            const auto rejected = admit_pending_render_state(state);
            require(rejected.status == PendingRenderAdmissionStatus::RebuildFailed && rejected.diagnostic == "MalformedRequiredProfileData phase=9A field=visual_grain" && !rejected.directState && !rejected.printState && state.buildCounterNext.load() == counter && held.hash == hash, "ordinary grain failure never publishes partial state or revokes admitted holds");
            pending(params);
            const auto recovered = admit_pending_render_state(state);
            require(route % 2 ? bool(recovered.printState) : bool(recovered.directState), "grain recovery");
            const auto repeat = admit_pending_render_state(state);
            require(repeat.directState == recovered.directState && repeat.printState == recovered.printState, "unchanged grain publication reuses state");
        }
        require(fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "all whole-state temporary owners drain");
        std::puts("PASS 36 independent grain products, four-route publication/recovery, minimum separation and stage order");
    }
} // namespace
namespace JuicerAssets::RecipeTest {
    void grain(FjVisualGrainRecipe& value) {
        if (nextGrainProjectionFault == 1)
            value.active = 2;
        if (nextGrainProjectionFault == 2)
            value.axis_finite_cmy[0] = 2;
        if (nextGrainProjectionFault == 3)
            value.hash = 0;
        if (nextGrainProjectionFault == 4)
            value.layers_hash = 0;
        if (nextGrainProjectionFault == 5)
            value.particle_area_um2 = std::numeric_limits<float>::quiet_NaN();
        if (nextGrainProjectionFault == 6)
            value.axis_block_prefix_max_cmy[0][0] = std::numeric_limits<float>::infinity();
        nextGrainProjectionFault = 0;
    }
} // namespace JuicerAssets::RecipeTest
