namespace DiffusionFrameTest {
    FjDiffusionFrameInput raw_frame(const Json& input);
    Json project(const FjDiffusionFrameSet& v);
    double scalar(const Json& input) {
        return std::bit_cast<double>(input.get<std::uint64_t>());
    }

    template <std::size_t N>
    std::array<double, N> doubles(const Json& input) {
        require(input.size() == N, "diffusion input extent");
        std::array<double, N> out{};
        for (std::size_t i = 0; i < N; ++i)
            out[i] = scalar(input[i]);
        return out;
    }

    template <std::size_t N>
    Json bits(const std::array<double, N>& input) {
        Json out = Json::array();
        for (const double value : input)
            out.push_back(std::bit_cast<std::uint64_t>(value));
        return out;
    }

    Spektrafilm::DiffusionFilterResolvedParameters resolved(const Json& input) {
        Spektrafilm::DiffusionFilterResolvedParameters out;
        out.family = static_cast<Spektrafilm::DiffusionFilterFamily>(input.at("family").get<unsigned>());
        out.groupWeightsCoreHaloBloom = doubles<3>(input.at("weights"));
        out.groupCenterLambdaUm = doubles<3>(input.at("centers"));
        out.effectiveWarmth = scalar(input.at("warmth"));
        out.spatialScale = scalar(input.at("scale"));
        out.scatterFraction = scalar(input.at("scatter"));
        out.active = input.at("active");
        out.hash = input.at("resolved_hash");
        return out;
    }

    DiffusionFilterOpticsRecipe component(const Json& input) {
        DiffusionFilterOpticsRecipe out;
        out.resolved = resolved(input);
        out.domain = static_cast<Spektrafilm::SpatialOpticsDomain>(input.at("domain").get<unsigned>());
        out.hash = input.at("hash");
        return out;
    }

    Json project(const Spektrafilm::DiffusionPsfSampleDescriptor& input) {
        return {{"family", static_cast<unsigned>(input.family)},
                {"weights", bits(input.groupWeightsCoreHaloBloom)},
                {"centers", bits(input.groupCenterLambdaUm)},
                {"warmth", std::bit_cast<std::uint64_t>(input.effectiveWarmth)},
                {"scale", std::bit_cast<std::uint64_t>(input.spatialScale)},
                {"pixel", std::bit_cast<std::uint64_t>(input.pixelSizeUm)},
                {"radius", input.radiusPixels},
                {"hash", input.hash}};
    }

    Json project(const Spektrafilm::DiffusionPsfComponents& input) {
        return {{"lambda", bits(input.lambdaPixels)}, {"red", bits(input.redWeights)}, {"green", bits(input.greenWeights)}, {"blue", bits(input.blueWeights)}};
    }

    Json project(const std::optional<Spektrafilm::DiffusionStageFrameDescriptor>& input) {
        if (!input)
            return nullptr;
        return {{"stage", static_cast<unsigned>(input->stage)},
                {"scatter", std::bit_cast<std::uint64_t>(input->scatterFraction)},
                {"sample", project(input->sample)},
                {"hash", input->hash}};
    }

    Json project(const std::optional<Spektrafilm::DiffusionFrameSetDescriptor>& input) {
        if (!input)
            return nullptr;
        const auto& d = input->fullFrame;
        return {{"route", static_cast<unsigned>(input->route)},
                {"extent", {d.originX, d.originY, d.width, d.height}},
                {"camera", project(input->camera)},
                {"enlarger", project(input->enlarger)},
                {"hash", input->hash}};
    }

    Json sample_product(const Json& input) {
        Spektrafilm::DiffusionPsfSampleDescriptor sample;
        std::string diagnostic;
        const bool success = JuicerTestSupport::prepare_diffusion_sample(
            resolved(input), scalar(input.at("pixel")), input.at("width"), input.at("height"), sample, diagnostic);
        Json out{{"input", input}, {"success", success}, {"diagnostic", diagnostic}, {"sample", project(sample)}};
        if (success) {
            Spektrafilm::DiffusionPsfComponents components;
            const bool expanded = JuicerTestSupport::prepare_diffusion_components(sample, components, diagnostic);
            out["expanded"] = expanded;
            out["component_diagnostic"] = diagnostic;
            out["components"] = project(components);
        }
        return out;
    }

    Json frame_product(const Json& input) {
        SpatialOptics optics;
        optics.cameraDiffusion = component(input.at("camera"));
        optics.enlargerDiffusion = component(input.at("enlarger"));
        const auto extent = input.at("extent").get<std::array<int, 4>>();
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> frame;
        std::string diagnostic;
        const bool success = JuicerTestSupport::prepare_diffusion_frame(
            optics, static_cast<Spektrafilm::ScanRoute>(input.at("route").get<unsigned>()), scalar(input.at("pixel")), {extent[0], extent[1], extent[2], extent[3]}, frame, diagnostic);
        return {{"input", input}, {"success", success}, {"diagnostic", diagnostic}, {"frame", project(frame)}};
    }

    Json resolved_input(const DiffusionFilterOpticsRecipe& component) {
        const auto& r = component.resolved;
        return {{"family", static_cast<unsigned>(r.family)}, {"weights", bits(r.groupWeightsCoreHaloBloom)}, {"centers", bits(r.groupCenterLambdaUm)}, {"warmth", std::bit_cast<std::uint64_t>(r.effectiveWarmth)}, {"scale", std::bit_cast<std::uint64_t>(r.spatialScale)}, {"scatter", std::bit_cast<std::uint64_t>(r.scatterFraction)}, {"active", r.active}, {"resolved_hash", r.hash}, {"domain", static_cast<unsigned>(component.domain)}, {"hash", component.hash}};
    }

    Json resolved_input(const FjDiffusionRecipe& r) {
        return {{"family", r.family}, {"weights", bits(std::array<double, 3>{r.group_weights[0], r.group_weights[1], r.group_weights[2]})}, {"centers", bits(std::array<double, 3>{r.group_lambda_um[0], r.group_lambda_um[1], r.group_lambda_um[2]})}, {"warmth", std::bit_cast<std::uint64_t>(r.warmth)}, {"scale", std::bit_cast<std::uint64_t>(r.spatial_scale)}, {"scatter", std::bit_cast<std::uint64_t>(r.scatter_fraction)}, {"active", r.active != 0}, {"resolved_hash", r.resolved_hash}, {"domain", r.domain}, {"hash", r.hash}};
    }

    Json products(const std::filesystem::path& resources, const Json& specs) {
        const auto started = std::chrono::steady_clock::now();
        auto progress = [&](const char* stage, const std::string& name) {
            std::fprintf(stderr, "DiffusionFrame stage=%s case=%s seconds=%.3f\n", stage, name.c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
            std::fflush(stderr);
        };
        progress("bootstrap", "");
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        Json rows = Json::array();
        for (const auto& spec : specs) {
            const auto name = spec.at("name").get<std::string>();
            progress("recipe", name);
            const auto route = spec.at("route").get<unsigned>();
            ParamSnapshot params;
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.printProfileKey = "kodak_portra_endura";
            params.cameraDiffusion.active = spec.at("camera");
            params.enlargerDiffusion.active = spec.at("enlarger");
            params.cameraDiffusion.family = params.enlargerDiffusion.family = static_cast<Spektrafilm::DiffusionFilterFamily>(spec.at("family").get<unsigned>());
            auto product = JuicerTestSupport::prepare_product(params);
            FjPreparedStateView view{};
            FjErrorBuffer viewError{};
            status(product.view(view, viewError));
            Json input{{"name", spec.at("name")}, {"route", route}, {"extent", spec.at("extent")}, {"pixel", std::bit_cast<std::uint64_t>(static_cast<double>(std::bit_cast<float>(spec.at("pixel_bits").get<std::uint32_t>())))}, {"camera", resolved_input(view.optics.camera_diffusion)}, {"enlarger", resolved_input(view.optics.enlarger_diffusion)}};
            progress("frame", name);
            const auto wire = raw_frame(input);
            FjDiffusionFrameSet native{};
            FjDiffusionFailure failure{};
            std::array<char, 256> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            const auto result = fj_test_diffusion_frame_prepare(&wire, &native, &failure, &error);
            Json frame = project(native);
            const auto cleared = project(FjDiffusionFrameSet{});
            if (result.category != FJ_STATUS_SUCCESS || native.flags == 0) {
                require_equal(frame, cleared, "absent diffusion frame clears the complete raw record");
                frame = nullptr;
            } else {
                frame.erase("flags");
                for (const auto& [name, flag] : {std::pair{"camera", 1u}, std::pair{"enlarger", 2u}}) {
                    if ((native.flags & flag) == 0) {
                        require_equal(frame.at(name), cleared.at(name), "absent diffusion stage clears its raw record");
                        frame[name] = nullptr;
                    }
                }
            }
            auto row = Json{{"input", input}, {"success", result.category == FJ_STATUS_SUCCESS}, {"diagnostic", std::string(diagnostic.data(), error.length)}, {"frame", std::move(frame)}};
            row["spec"] = spec;
            row["samples"] = Json::array();
            for (const auto* stage : {"camera", "enlarger"}) {
                if (input.at(stage).at("hash") == 0)
                    continue;
                auto sample = input.at(stage);
                sample["name"] = spec.at("name").get<std::string>() + "/" + stage;
                sample["pixel"] = input.at("pixel");
                sample["width"] = spec.at("extent")[2];
                sample["height"] = spec.at("extent")[3];
                progress(stage, name);
                auto sampleRow = sample_product(sample);
                if (!row.at("frame").is_null()) {
                    const auto rawStage = row.at("frame").at(stage);
                    require(sampleRow.at("success").get<bool>() &&
                                sampleRow.at("sample").at("hash") == rawStage.at("sample_hash"),
                            "actual diffusion sample binds the raw frame stage identity");
                    row["frame"][stage] = {{"stage", input.at(stage).at("domain")},
                                           {"scatter", rawStage.at("scatter")},
                                           {"sample", sampleRow.at("sample")},
                                           {"hash", rawStage.at("hash")}};
                }
                row["samples"].push_back(std::move(sampleRow));
            }
            rows.push_back(row);
            progress("case-complete", name);
        }
        progress("owner-release", "");
        return rows;
    }


} // namespace DiffusionFrameTest

namespace DiffusionFrameTest {
    FjDiffusionSampleInput raw_sample(const Json& input) {
        const auto r = resolved(input);
        FjDiffusionSampleInput out{};
        out.family = input.at("family");
        std::copy(r.groupWeightsCoreHaloBloom.begin(), r.groupWeightsCoreHaloBloom.end(), out.group_weights);
        std::copy(r.groupCenterLambdaUm.begin(), r.groupCenterLambdaUm.end(), out.group_lambda_um);
        out.warmth = r.effectiveWarmth;
        out.spatial_scale = r.spatialScale;
        out.pixel_size_um = scalar(input.at("pixel"));
        out.width = input.at("width");
        out.height = input.at("height");
        return out;
    }
    FjDiffusionSample expected_sample(const Json& input) {
        FjDiffusionSample out{};
        out.family = input.at("family");
        const auto weights = doubles<3>(input.at("weights")), centers = doubles<3>(input.at("centers"));
        std::copy(weights.begin(), weights.end(), out.group_weights);
        std::copy(centers.begin(), centers.end(), out.group_lambda_um);
        out.warmth = scalar(input.at("warmth"));
        out.spatial_scale = scalar(input.at("scale"));
        out.pixel_size_um = scalar(input.at("pixel"));
        out.radius_px = input.at("radius");
        out.hash = input.at("hash");
        return out;
    }
    Json project(const FjDiffusionSample& input) {
        const auto array = [](const double* values) {
            return std::array{values[0], values[1], values[2]};
        };
        return {{"family", input.family}, {"weights", bits(array(input.group_weights))}, {"centers", bits(array(input.group_lambda_um))}, {"warmth", std::bit_cast<std::uint64_t>(input.warmth)}, {"scale", std::bit_cast<std::uint64_t>(input.spatial_scale)}, {"pixel", std::bit_cast<std::uint64_t>(input.pixel_size_um)}, {"radius", input.radius_px}, {"hash", input.hash}};
    }
    template <std::size_t N>
    Json array_bits(const double (&input)[N]) {
        std::array<double, N> copied{};
        std::copy(std::begin(input), std::end(input), copied.begin());
        return bits(copied);
    }
    Json project(const FjDiffusionComponents& input) {
        return {{"lambda", array_bits(input.lambda_px)}, {"red", array_bits(input.red_weights)}, {"green", array_bits(input.green_weights)}, {"blue", array_bits(input.blue_weights)}};
    }
    FjDiffusionRecipe raw_component(const Json& input) {
        const auto r = resolved(input);
        FjDiffusionRecipe out{};
        out.domain = input.at("domain");
        out.active = input.at("active").get<bool>() ? 1u : 0u;
        out.family = input.at("family");
        out.scatter_fraction = r.scatterFraction;
        std::copy(r.groupWeightsCoreHaloBloom.begin(), r.groupWeightsCoreHaloBloom.end(), out.group_weights);
        std::copy(r.groupCenterLambdaUm.begin(), r.groupCenterLambdaUm.end(), out.group_lambda_um);
        out.warmth = r.effectiveWarmth;
        out.spatial_scale = r.spatialScale;
        out.resolved_hash = r.hash;
        out.hash = input.at("hash");
        return out;
    }
    FjDiffusionFrameInput raw_frame(const Json& input) {
        const auto extent = input.at("extent").get<std::array<int, 4>>();
        return {input.at("route"), {extent[0], extent[1], extent[2], extent[3]}, scalar(input.at("pixel")), raw_component(input.at("camera")), raw_component(input.at("enlarger"))};
    }
    FjDiffusion expected_stage(const Json& input) {
        FjDiffusion out{};
        if (input.is_null())
            return out;
        const auto s = expected_sample(input.at("sample"));
        out.family = s.family;
        out.scatter_fraction = scalar(input.at("scatter"));
        std::copy(std::begin(s.group_weights), std::end(s.group_weights), out.group_weights);
        std::copy(std::begin(s.group_lambda_um), std::end(s.group_lambda_um), out.group_lambda_um);
        out.warmth = s.warmth;
        out.spatial_scale = s.spatial_scale;
        out.pixel_size_um = s.pixel_size_um;
        out.radius_px = s.radius_px;
        out.sample_hash = s.hash;
        out.hash = input.at("hash");
        return out;
    }
    FjDiffusionFrameSet expected_frame(const Json& input) {
        FjDiffusionFrameSet out{};
        if (input.is_null())
            return out;
        const auto extent = input.at("extent").get<std::array<int, 4>>();
        out.route = input.at("route");
        out.flags = (!input.at("camera").is_null() ? 1u : 0u) | (!input.at("enlarger").is_null() ? 2u : 0u);
        out.full_frame = {extent[0], extent[1], extent[2], extent[3]};
        out.camera = expected_stage(input.at("camera"));
        out.enlarger = expected_stage(input.at("enlarger"));
        out.hash = input.at("hash");
        return out;
    }
    Json project(const FjDiffusion& v) {
        return {{"family", v.family}, {"scatter", std::bit_cast<std::uint64_t>(v.scatter_fraction)}, {"weights", array_bits(v.group_weights)}, {"centers", array_bits(v.group_lambda_um)}, {"warmth", std::bit_cast<std::uint64_t>(v.warmth)}, {"scale", std::bit_cast<std::uint64_t>(v.spatial_scale)}, {"pixel", std::bit_cast<std::uint64_t>(v.pixel_size_um)}, {"radius", v.radius_px}, {"sample_hash", v.sample_hash}, {"hash", v.hash}};
    }
    Json project(const FjDiffusionFrameSet& v) {
        return {{"route", v.route}, {"flags", v.flags}, {"extent", {v.full_frame.x, v.full_frame.y, v.full_frame.width, v.full_frame.height}}, {"camera", project(v.camera)}, {"enlarger", project(v.enlarger)}, {"hash", v.hash}};
    }
    void metadata(FjStatus s, const FjDiffusionFailure& f, const FjErrorBuffer& e, const char* text, const Json& expected) {
        status(s, expected.at("category"));
        require(f.reason == expected.at("reason") && f.stage == expected.at("stage") && f.field == expected.at("field"), "closed diffusion failure envelope");
        require(e.length == expected.at("diagnostic").get<std::string>().size() && std::string_view(text, e.length) == expected.at("diagnostic").get<std::string>() && text[e.length] == '\0', "exact diffusion diagnostic");
    }
    Json success() {
        return {{"category", 0}, {"reason", 0}, {"stage", 0}, {"field", 0}, {"diagnostic", ""}};
    }
    void sample_values(const Json& row, const Json& checked) {
        const auto input = raw_sample(row.at("input"));
        const bool validSample = row.at("success");
        const auto expected = validSample ? expected_sample(row.at("sample")) : FjDiffusionSample{};
        for (auto prepare : {fj_legacy_diffusion_sample_prepare, fj_test_diffusion_sample_prepare, fj_test_diffusion_sample_c_prepare}) {
            FjDiffusionSample out{};
            std::memset(&out, 0x55, sizeof(out));
            FjDiffusionFailure failure{99, 99, 99};
            char text[256] = "seed";
            FjErrorBuffer error{text, sizeof(text), 99};
            metadata(prepare(&input, &out, &failure, &error), failure, error, text, checked.at("sample"));
            require(project(out) == project(expected), "all diffusion sample words/hash and raw clearing");
        }
        if (!validSample) {
            Spektrafilm::DiffusionPsfSampleDescriptor native;
            JuicerCuda::Failure failure;
            require(!JuicerDiffusion::prepare_sample(resolved(row.at("input")), input.pixel_size_um, input.width, input.height, native, failure) && project(native) == row.at("sample"), "ordinary native sample failure retains default family while raw failure clears");
            return;
        }
        const auto sample = expected_sample(row.at("sample"));
        for (auto prepare : {fj_legacy_diffusion_components_prepare, fj_test_diffusion_components_prepare, fj_test_diffusion_components_c_prepare}) {
            FjDiffusionComponents out{};
            std::memset(&out, 0x55, sizeof(out));
            FjDiffusionFailure failure{99, 99, 99};
            char text[256] = "seed";
            FjErrorBuffer error{text, sizeof(text), 99};
            metadata(prepare(&sample, &out, &failure, &error), failure, error, text, checked.at("components"));
            require(project(out) == row.at("components"), "all 36 diffusion component words including signed zeros/failure clearing");
        }
        Spektrafilm::DiffusionPsfSampleDescriptor native;
        JuicerCuda::Failure failure;
        const bool prepared = JuicerDiffusion::prepare_sample(resolved(row.at("input")), input.pixel_size_um, input.width, input.height, native, failure);
        require(prepared && project(native) == row.at("sample"), "native sample adapter complete projection");
        Spektrafilm::DiffusionPsfComponents expanded;
        const bool expandedOk = JuicerDiffusion::prepare_components(native, expanded, failure);
        require(expandedOk == row.at("expanded").get<bool>() && project(expanded) == row.at("components") && failure.diagnostic == row.at("component_diagnostic").get<std::string>(), "native component adapter full values/failure");
    }
    // Keep frozen whole-producer values separate from independently authored boundary metadata.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void frame_values(const Json& row, const Json& checked) {
        const auto input = raw_frame(row.at("input"));
        const auto expected = expected_frame(row.at("frame"));
        for (auto prepare : {fj_legacy_diffusion_frame_prepare, fj_test_diffusion_frame_prepare, fj_test_diffusion_frame_c_prepare}) {
            FjDiffusionFrameSet out{};
            std::memset(&out, 0x55, sizeof(out));
            FjDiffusionFailure failure{99, 99, 99};
            char text[256] = "seed";
            FjErrorBuffer error{text, sizeof(text), 99};
            metadata(prepare(&input, &out, &failure, &error), failure, error, text, checked);
            require(project(out) == project(expected), "complete diffusion frame route/domain/flags/stages/identities and clearing");
        }
        const auto& authored = row.at("input");
        SpatialOptics optics;
        optics.cameraDiffusion = component(authored.at("camera"));
        optics.enlargerDiffusion = component(authored.at("enlarger"));
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> native;
        JuicerCuda::Failure failure;
        const bool prepared = JuicerDiffusion::prepare_frame(optics, static_cast<Spektrafilm::ScanRoute>(input.route), input.pixel_size_um, {input.full_frame.x, input.full_frame.y, input.full_frame.width, input.full_frame.height}, native, failure);
        require(prepared == row.at("success").get<bool>() && project(native) == row.at("frame") && failure.diagnostic == row.at("diagnostic").get<std::string>(), "native frame adapter exact projection/failure/absence");
    }
    void values(const Json& captured, const Json& checked) {
        require(captured.at("samples").size() == checked.at("samples").size() && captured.at("frames").size() == checked.at("frames").size(), "exact diffusion characterization membership");
        for (const auto& row : captured.at("samples"))
            sample_values(row, checked.at("samples").at(row.at("input").at("name").get<std::string>()));
        for (const auto& row : captured.at("frames"))
            frame_values(row, checked.at("frames").at(row.at("input").at("name").get<std::string>()));
        require(captured.at("products").size() == 92, "exact completed-route product membership");
        for (const auto& row : captured.at("products")) {
            frame_values(row, success());
            for (const auto& sample : row.at("samples"))
                sample_values(sample, {{"sample", success()}, {"components", success()}});
        }
    }
    // Raw cases pair original producer inputs with the separate closed-envelope expectations.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void boundary(const Json& captured, const Json& checked) {
        auto input = raw_sample(captured.at("samples")[0].at("input"));
        auto sample = expected_sample(captured.at("samples")[0].at("sample"));
        auto frame = raw_frame(captured.at("frames")[3].at("input"));
        frame.route = 1;
        for (const auto& row : checked.at("raw")) {
            const unsigned op = row.at("operation");
            FjDiffusionFailure failure{99, 99, 99};
            char text[256];
            FjErrorBuffer error{text, sizeof(text), 0};
            if (op == 1) {
                auto invalid = input;
                invalid.family = 99;
                FjDiffusionSample out{};
                metadata(fj_test_diffusion_sample_prepare(&invalid, &out, &failure, &error), failure, error, text, row);
                require(project(out) == project(FjDiffusionSample{}), "safe invalid-family raw sample clearing");
            } else if (op == 2) {
                auto invalid = sample;
                const auto mutation = row.at("mutation").get<std::string>();
                if (mutation == "zero-hash")
                    invalid.hash = 0;
                if (mutation == "tampered-hash")
                    invalid.hash ^= 1;
                if (mutation == "negative-radius")
                    invalid.radius_px = -1;
                FjDiffusionComponents out{};
                metadata(fj_test_diffusion_components_prepare(&invalid, &out, &failure, &error), failure, error, text, row);
                require(project(out) == project(FjDiffusionComponents{}), "raw component failure clearing");
            } else {
                auto invalid = frame;
                invalid.camera.family = 99;
                FjDiffusionFrameSet out{};
                metadata(fj_test_diffusion_frame_prepare(&invalid, &out, &failure, &error), failure, error, text, row);
                require(project(out) == project(FjDiffusionFrameSet{}), "safe stage sample failure collapse/clearing");
            }
        }
        auto raw = [&]<class I, class O>(auto prepare, const I& valid, O empty, unsigned operation) {
            O out{};
            FjDiffusionFailure failure{};
            for (unsigned mode = 0; mode < 7; ++mode) {
                std::memset(&out, 0x55, sizeof(out));
                failure = {99, 99, 99};
                char text[256] = "seed";
                FjErrorBuffer error{text, sizeof(text), 99};
                alignas(I) std::array<unsigned char, sizeof(I) + 1> unaligned{};
                const I* in = mode == 0 ? nullptr : mode == 5 ? reinterpret_cast<const I*>(unaligned.data() + 1)
                                                              : &valid;
                if (mode == 3)
                    error.data = nullptr;
                if (mode == 4)
                    error.capacity = 0;
                auto* destination = mode == 1 ? nullptr : &out;
                auto* details = mode == 2 ? nullptr : &failure;
                const auto s = prepare(in, destination, details, mode == 6 ? nullptr : &error);
                if (mode == 4 || mode == 6) {
                    status(s);
                    continue;
                }
                status(s, 2);
                if (destination)
                    require(project(out) == project(empty), "independent raw output clearing");
                if (details)
                    require(failure.reason == 1 && failure.stage == 0 && failure.field == 0, "independent raw failure delivery");
            }
            status(fj_test_diffusion_frame_fault(operation, 1));
            status(fj_test_diffusion_frame_fault(2, 6), 2);
            if (operation == 1) {
                FjDiffusionComponents other{};
                FjDiffusionFailure f{};
                status(fj_test_diffusion_components_prepare(&sample, &other, &f, nullptr));
            } else {
                FjDiffusionSample other{};
                FjDiffusionFailure f{};
                status(fj_test_diffusion_sample_prepare(&input, &other, &f, nullptr));
            }
            FjStatus threaded{};
            std::thread other([&] {
                O local{};
                FjDiffusionFailure f{};
                threaded = prepare(&valid, &local, &f, nullptr);
            });
            other.join();
            status(threaded);
            status(prepare(&valid, &out, &failure, nullptr), 8);
            require(failure.reason == 3 && failure.stage == 0 && failure.field == 0 && project(out) == project(empty), "contained panic and cross-thread one-shot isolation");
            status(prepare(&valid, &out, &failure, nullptr));
            status(fj_test_diffusion_frame_fault(operation, 0));
            const std::array messages{"invalid diffusion sample arguments", "invalid diffusion components arguments", "invalid diffusion frame arguments"};
            for (const std::size_t capacity : {0u, 1u, 7u, 256u}) {
                std::array<char, 257> text{};
                text.fill('Q');
                FjErrorBuffer error{capacity ? text.data() : reinterpret_cast<char*>(UINTPTR_MAX), capacity, 99};
                std::memset(&out, 0x55, sizeof(out));
                failure = {99, 99, 99};
                status(prepare(nullptr, &out, &failure, &error), 2);
                const auto count = capacity ? std::min(std::strlen(messages[operation - 1]), capacity - 1) : 0;
                require(error.length == count && failure.reason == 1 && failure.stage == 0 && failure.field == 0 && project(out) == project(empty), "bounded diagnostics preserve status, typed failure and clearing");
                if (capacity)
                    require(text[count] == '\0' && std::string_view(text.data(), count) == std::string_view(messages[operation - 1], count) && text[capacity] == 'Q', "diagnostic truncation, terminator and backing bound");
            }
            alignas(O) std::array<unsigned char, sizeof(O) + 1> badOutput{};
            alignas(FjDiffusionFailure) std::array<unsigned char, sizeof(FjDiffusionFailure) + 1> badFailure{};
            alignas(FjErrorBuffer) std::array<unsigned char, sizeof(FjErrorBuffer) + 1> badError{};
            failure = {99, 99, 99};
            status(prepare(&valid, reinterpret_cast<O*>(badOutput.data() + 1), &failure, nullptr), 2);
            require(failure.reason == 1 && failure.stage == 0 && failure.field == 0 && std::all_of(badOutput.begin(), badOutput.end(), [](unsigned char v) {
                        return v == 0;
                    }),
                    "misaligned output leaves bytes untouched and admits failure independently");
            std::memset(&out, 0x55, sizeof(out));
            status(prepare(&valid, &out, reinterpret_cast<FjDiffusionFailure*>(badFailure.data() + 1), nullptr), 2);
            require(project(out) == project(empty), "misaligned failure still clears admitted output");
            std::memset(&out, 0x55, sizeof(out));
            status(prepare(&valid, &out, &failure, reinterpret_cast<FjErrorBuffer*>(badError.data() + 1)), 2);
            require(project(out) == project(empty) && failure.reason == 1, "misaligned diagnostic still clears and delivers admitted peers");
        };
        raw(fj_test_diffusion_sample_prepare, input, FjDiffusionSample{}, 1);
        raw(fj_test_diffusion_components_prepare, sample, FjDiffusionComponents{}, 2);
        raw(fj_test_diffusion_frame_prepare, frame, FjDiffusionFrameSet{}, 3);
        status(fj_test_diffusion_frame_fault(1, 1));
        status(fj_test_diffusion_frame_fault(2, 0));
        FjDiffusionSample clearedFault{};
        FjDiffusionFailure clearedFailure{};
        status(fj_test_diffusion_sample_prepare(&input, &clearedFault, &clearedFailure, nullptr));
        auto ignored = frame;
        ignored.camera.active = ignored.enlarger.active = UINT32_MAX;
        ignored.camera.resolved_hash = ignored.enlarger.resolved_hash = UINT64_MAX;
        FjDiffusionFrameSet ignoredOut{};
        status(fj_test_diffusion_frame_prepare(&ignored, &ignoredOut, &clearedFailure, nullptr));
        require(project(ignoredOut) == project(expected_frame(captured.at("frames")[7].at("frame"))), "unused active and resolved-hash fields do not admit or select frame activity");
        for (unsigned op : {1u, 2u, 3u}) {
            for (unsigned fault = 2; fault <= (op == 3 ? 6u : op == 1 ? 5u
                                                                      : 2u);
                 ++fault) {
                status(fj_test_diffusion_frame_fault(op, fault));
                JuicerCuda::Failure failure;
                bool success = false;
                if (op == 1) {
                    Spektrafilm::DiffusionPsfSampleDescriptor out;
                    success = JuicerDiffusion::prepare_sample(resolved(captured.at("samples")[0].at("input")), input.pixel_size_um, input.width, input.height, out, failure);
                    require(project(out) == project(Spektrafilm::DiffusionPsfSampleDescriptor{}), "native default sample on corrupt projection");
                } else if (op == 2) {
                    const auto& s = captured.at("samples")[0].at("sample");
                    Spektrafilm::DiffusionPsfSampleDescriptor in;
                    in.family = static_cast<Spektrafilm::DiffusionFilterFamily>(sample.family);
                    in.groupWeightsCoreHaloBloom = doubles<3>(s.at("weights"));
                    in.groupCenterLambdaUm = doubles<3>(s.at("centers"));
                    in.effectiveWarmth = sample.warmth;
                    in.spatialScale = sample.spatial_scale;
                    in.pixelSizeUm = sample.pixel_size_um;
                    in.radiusPixels = sample.radius_px;
                    in.hash = sample.hash;
                    Spektrafilm::DiffusionPsfComponents out;
                    success = JuicerDiffusion::prepare_components(in, out, failure);
                    require(project(out) == project(Spektrafilm::DiffusionPsfComponents{}), "native components clear corrupt envelope");
                } else {
                    const auto& f = captured.at("frames")[3].at("input");
                    SpatialOptics optics;
                    optics.cameraDiffusion = component(f.at("camera"));
                    optics.enlargerDiffusion = component(f.at("enlarger"));
                    std::optional<Spektrafilm::DiffusionFrameSetDescriptor> out;
                    success = JuicerDiffusion::prepare_frame(optics, Spektrafilm::ScanRoute::NegativePrintScan, frame.pixel_size_um, {-17, 9, 4096, 2160}, out, failure);
                    require(!out, "native frame clears corrupt projection");
                }
                require(!success && failure.status.category == 3 && failure.status.api == 0 && failure.status.native_code == 0 && failure.diagnostic == std::string("diffusion ") + (op == 1 ? "sample" : op == 2 ? "components"
                                                                                                                                                                                                                  : "frame") +
                                                                                                                                                            " projection invalid",
                        "native closed envelope/projection rejection");
            }
        }
    }
} // namespace DiffusionFrameTest
