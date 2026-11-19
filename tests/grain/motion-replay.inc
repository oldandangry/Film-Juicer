// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0
// The offline witness uses the real OFX callbacks and the replay host above.

Json read_motion_json(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "motion manifest or sampler receipt unavailable");
    return Json::parse(input);
}

void read_motion_pixels(const std::filesystem::path& path, std::span<float> pixels) {
    require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == pixels.size_bytes(),
            "motion Float32 input size mismatch: " + path.string());
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size_bytes()));
    require(static_cast<bool>(input) && std::all_of(pixels.begin(), pixels.end(), [](float value) {
                return std::isfinite(value);
            }),
            "unreadable or nonfinite motion input: " + path.string());
}

void reverse_motion_rows(std::span<float> pixels, std::size_t rowSize, int height) {
    for (int y = 0; y < height / 2; ++y) {
        auto first = pixels.subspan(static_cast<std::size_t>(y) * rowSize, rowSize);
        auto last = pixels.subspan(static_cast<std::size_t>(height - 1 - y) * rowSize, rowSize);
        std::swap_ranges(first.begin(), first.end(), last.begin());
    }
}

std::uint32_t motion_seed(const Json& value) {
    require(value.is_number_unsigned() && value.get<std::uint64_t>() <= std::numeric_limits<std::uint32_t>::max(),
            "motion seed must be an unsigned 32-bit integer");
    return value.get<std::uint32_t>();
}

void run_fotufilm(const Json& sampler, const Case& test, double longEdge, std::uint32_t seed, const std::filesystem::path& cleanDensity, const std::filesystem::path& grainedDensity, const std::filesystem::path& receipt) {
    std::array arguments{
        sampler.at("executable").get<std::string>(), sampler.at("stock").get<std::string>(), cleanDensity.string(), std::to_string(test.width), std::to_string(test.height), std::to_string(longEdge), std::to_string(seed), grainedDensity.string(), receipt.string()};
    std::array<char*, std::tuple_size_v<decltype(arguments)> + 1> argv{};
    std::transform(arguments.begin(), arguments.end(), argv.begin(), [](auto& argument) {
        return argument.data();
    });
    pid_t child = 0;
    const auto spawnError = posix_spawn(&child, argv.front(), nullptr, nullptr, argv.data(), environ);
    require(spawnError == 0, "could not start Fotufilm sampler: " + std::to_string(spawnError));
    int status = 0;
    pid_t completed;
    do {
        completed = waitpid(child, &status, 0);
    } while (completed < 0 && errno == EINTR);
    require(completed == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "Fotufilm sampler failed");
    const auto metadata = read_motion_json(receipt);
    require(metadata.at("seed") == seed && metadata.at("grainOffEqualsInput") == true &&
                metadata.at("changedChannelSamples").get<std::size_t>() > 0,
            "Fotufilm sampler receipt failed");
}

Json set_motion_controls(InstanceHost& host, OfxImageEffectHandle effect, const Json& settings, double longEdgeMm) {
    require(settings.is_object(), "motion parameters must be an object");
    for (const auto& definition : ofx_test::kInstanceParameters) {
        const std::string name(definition.name);
        if (!settings.contains(name) || definition.kind == ofx_test::InstanceParamKind::PushButton)
            continue;
        const auto& value = settings.at(name);
        require(value.at("type") == replay_type(definition.kind) && value.at("status") == kOfxStatOK,
                "invalid captured motion parameter: " + name);
        host.set_value(effect, name, replay_values(value, definition.kind));
    }
    host.set_value(effect, "Exposure", {1.0});
    host.set_value(effect, "CameraAutoExposure", {0});
    host.set_value(effect, "CameraFilmFormatPreset", {longEdgeMm == 36.0 ? 2 : 0});
    host.set_value(effect, "CameraFilmFormatMm", {longEdgeMm});
    host.set_value(effect, "FilmProfileKey", {std::string("kodak_portra_400")});
    host.set_value(effect, "PrintProfileKey", {std::string("kodak_portra_endura")});
    host.set_value(effect, "ScanRoute", {std::string("NegativePrintScan")});
    host.set_value(effect, "PrintExposureCompensation", {0});
    host.set_value(effect, "GrainActive", {1});
    host.set_value(effect, "GrainModel", {std::string("film_v1")});

    // These preview-only parameters are outside the frozen standard instance census.
    const std::array filmDefaults{
        std::pair{"FilmGrainAmountEv", 0.0},
        std::pair{"FilmGrainTexelPitchUm", 1.0},
        std::pair{"FilmGrainChromaFraction", 0.2406350076198578},
    };
    constexpr auto kind = ofx_test::InstanceParamKind::Double;
    auto filmControls = Json::object();
    for (const auto& [name, fallback] : filmDefaults) {
        if (settings.contains(name)) {
            const auto& value = settings.at(name);
            require(value.at("type") == replay_type(kind) && value.at("status") == kOfxStatOK,
                    std::string("invalid Film grain parameter: ") + name);
            host.set_value(effect, name, replay_values(value, kind));
        } else {
            host.set_value(effect, name, {fallback});
        }
        filmControls[name] = std::get<double>(host.value(effect, name).front());
    }
    return filmControls;
}

int grain_motion_replay(const std::filesystem::path& manifestPath, const std::filesystem::path& outputPath) {
    const auto input = read_motion_json(manifestPath);
    const bool direct = input.value("negative_direct", false);
    const Case test{
        .id = "portra-motion",
        .routeName = direct ? "negative-direct" : "negative-print",
        .route = direct ? "NegativeDirectScan" : "NegativePrintScan",
        .components = input.at("components").get<int>(),
        .width = input.at("width").get<int>(),
        .height = input.at("height").get<int>(),
    };
    const auto frameRate = input.at("frame_rate").get<double>();
    const auto longEdge = input.at("long_edge_mm").get<double>();
    const auto rowOrder = input.at("row_order").get<std::string>();
    const auto fullWidth = input.value("full_width", test.width);
    const auto fullHeight = input.value("full_height", test.height);
    const auto& frames = input.at("frames");
    const auto* sampler = input.contains("fotufilm") ? &input.at("fotufilm") : nullptr;
    const auto captureFilm = input.value("capture_film_density", false);
    const auto grainOffReference = input.value("grain_off_reference", false);
    std::optional<double> projectionFraction;
    if (input.contains("fotufilm_projection_fraction")) {
        projectionFraction = input.at("fotufilm_projection_fraction").get<double>();
        require(sampler && *projectionFraction >= 0.0 && *projectionFraction <= 1.0, "invalid Fotufilm projection fraction");
    }
    require(test.width > 0 && test.height > 0 && fullWidth >= test.width && fullHeight >= test.height &&
                ((fullWidth == test.width) == (fullHeight == test.height)) &&
                (test.components == 3 || test.components == 4) &&
                test.width <= std::numeric_limits<int>::max() / (test.components * static_cast<int>(sizeof(float))) &&
                test.height <= std::numeric_limits<int>::max() / test.width,
            "invalid motion image geometry");
    require(longEdge >= 8.0 && longEdge <= 120.0 && std::isfinite(frameRate) && frameRate > 0.0 &&
                (rowOrder == "top-down" || rowOrder == "bottom-up") &&
                frames.is_array() && !frames.empty(),
            "invalid motion film dimensions, timing or row order");
    if (sampler)
        require(std::filesystem::path(sampler->at("executable").get<std::string>()).is_absolute() &&
                    std::filesystem::path(sampler->at("stock").get<std::string>()).is_absolute(),
                "Fotufilm executable and stock paths must be absolute");
    require(std::filesystem::create_directory(outputPath), "motion output directory must be new");
    for (const auto* arm : {"film", "clean", "density"})
        require(std::filesystem::create_directory(outputPath / arm), "could not create motion output arm");
    if (sampler)
        require(std::filesystem::create_directory(outputPath / "fotufilm"), "could not create Fotufilm arm");
    if (projectionFraction)
        require(std::filesystem::create_directory(outputPath / "fotufilm-projected"), "could not create projected Fotufilm arm");
    if (grainOffReference)
        require(std::filesystem::create_directory(outputPath / "grain-off"), "could not create grain-off output");
    std::filesystem::copy_file(manifestPath, outputPath / "input.json");
    std::ofstream reports(outputPath / "frames.jsonl");
    require(static_cast<bool>(reports), "could not open motion receipt");

    require_driver(cuInit(0), "initialize motion CUDA driver");
    CUcontext original = nullptr;
    require_driver(cuCtxGetCurrent(&original), "read initial motion CUDA context");
    ContextRestore restore(original);
    require_cuda(cudaSetDevice(0), "select motion CUDA device");
    auto* plugin = fj_ofx_plugin(0);
    require(plugin && plugin->pluginIdentifier &&
                std::string_view(plugin->pluginIdentifier) == "com.juicer.Juicer.GrainPreview",
            "motion probe requires the Film v1 preview");
    InstanceHost host;
    Lifecycle lifecycle(plugin, host);
    require(lifecycle.load() == kOfxStatOK, "motion OFX load failed");
    require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
            "motion OFX context description failed");
    auto* effect = host.create_effect();
    const auto filmControls = set_motion_controls(host, effect, input.value("parameters", Json::object()), longEdge);
    host.set_value(effect, "ScanRoute", {std::string(test.route)});
    const auto expectedChroma = filmControls.at("FilmGrainChromaFraction").get<float>();

    const auto rowSize = static_cast<std::size_t>(test.width) * static_cast<std::size_t>(test.components);
    const auto area = static_cast<std::size_t>(test.width) * static_cast<std::size_t>(test.height);
    std::vector<float> source(rowSize * static_cast<std::size_t>(test.height));
    std::vector<float> output(source.size());
    std::vector<float> clean(source.size());
    std::vector<float> density(area * 3);
    std::vector<float> ungrainedDensity(projectionFraction ? density.size() : 0);
    const auto imageBytes = source.size() * sizeof(float);
    DeviceBuffers device;
    require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create motion stream");
    require_cuda(cudaMalloc(&device.source, imageBytes), "allocate motion source");
    require_cuda(cudaMalloc(&device.destination, imageBytes), "allocate motion output");
    configure(host, effect, test, device);
    host.set_effect_property(effect, kOfxImageEffectPropFrameRate, {frameRate});
    for (const auto* clip : {"Source", "Output"}) {
        host.set_image_property(effect, clip, kOfxImagePropRowBytes, {static_cast<int>(rowSize * sizeof(float))});
    }
    auto* args = host.action_args();
    if (input.contains("geometry")) {
        const auto& mapping = input.at("geometry");
        const auto rod = mapping.at("canonical_rod").get<std::array<double, 4>>();
        const auto bounds = mapping.at("bounds").get<std::array<int, 4>>();
        require(static_cast<std::int64_t>(bounds[2]) - bounds[0] == test.width &&
                    static_cast<std::int64_t>(bounds[3]) - bounds[1] == test.height,
                "motion geometry bounds differ from the image shape");
        set_grain_geometry(host, effect, {{rod[0], rod[1], rod[2], rod[3]}, {bounds[0], bounds[1], bounds[2], bounds[3]}, mapping.at("action_scale").get<std::array<double, 2>>()});
    } else if (fullWidth != test.width) {
        host.set_property(args, kOfxImageEffectPropRenderScale, {static_cast<double>(test.width) / fullWidth, static_cast<double>(test.height) / fullHeight});
        for (const auto* clip : {"Source", "Output"})
            host.set_clip_region(effect, clip, OfxRectD{0.0, 0.0, static_cast<double>(fullWidth), static_cast<double>(fullHeight)});
    }
    host.set_property(args, kOfxPropTime, {frames.front().at("time").get<double>()});
    require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "motion OFX instance creation failed");
    lifecycle.created(effect);
    const auto readLongEdge = std::get<double>(host.value(effect, "CameraFilmFormatMm").front());
    const auto readPreset = std::get<int>(host.value(effect, "CameraFilmFormatPreset").front());
    require(readLongEdge == longEdge && readPreset == (longEdge == 36.0 ? 2 : 0), "motion film format readback");
    const auto expectedWidth = static_cast<float>(
        static_cast<double>(static_cast<float>(readLongEdge)) * 1000.0 /
        std::max(test.width, test.height) /
        static_cast<float>(filmControls.at("FilmGrainTexelPitchUm").get<double>()));
    std::string diagnostic;
    host.set_message_callback([&diagnostic](std::string_view, std::string_view, std::string_view message) {
        diagnostic = message;
        return kOfxStatOK;
    });

    const auto render = [&](MotionPass pass, const std::optional<MotionSeedPair>& seeds) {
        MotionDensity tap{.pass = pass, .density = density, .seeds = seeds, .captureFilm = pass == MotionPass::Film && captureFilm};
        const auto sourceBefore = host.image_release_count(effect, "Source");
        const auto outputBefore = host.image_release_count(effect, "Output");
        host.clear_calls();
        diagnostic.clear();
        OfxStatus status;
        {
            const MotionDensityScope scope(tap);
            status = lifecycle.call(kOfxImageEffectActionRender, effect, args);
        }
        require(status == kOfxStatOK && !host.callback_error() &&
                    tap.visited == (pass != MotionPass::GrainOff) && tap.filmCaptured == tap.captureFilm,
                "motion OFX render did not complete with the expected density tap: " + diagnostic);
        require(host.image_release_count(effect, "Source") == sourceBefore + 1 &&
                    host.image_release_count(effect, "Output") == outputBefore + 1,
                "motion OFX images were not both released exactly once");
        require_cuda(cudaStreamSynchronize(device.stream), "complete motion OFX render");
        require_cuda(cudaMemcpy(output.data(), device.destination, imageBytes, cudaMemcpyDeviceToHost), "read motion output");
        require(std::all_of(output.begin(), output.end(), [](float value) {
                    return std::isfinite(value);
                }),
                "nonfinite motion OFX output");
        if (rowOrder == "top-down")
            reverse_motion_rows(output, rowSize, test.height);
        return std::pair{tap.admitted, tap.samplePitch};
    };

    const auto upload_source = [&](const Json& frame) {
        read_motion_pixels(frame.at("source").get<std::string>(), source);
        if (rowOrder == "top-down")
            reverse_motion_rows(source, rowSize, test.height);
        require_cuda(cudaMemcpy(device.source, source.data(), imageBytes, cudaMemcpyHostToDevice), "upload motion frame");
        host.set_property(args, kOfxPropTime, {frame.at("time").get<double>()});
    };

    auto priorTime = -std::numeric_limits<double>::infinity();
    std::optional<std::uint32_t> priorFotufilmSeed;
    for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto& frame = frames[index];
        const auto time = frame.at("time").get<double>();
        require(std::isfinite(time) && time > priorTime, "motion OFX times must advance");
        priorTime = time;
        std::optional<std::uint32_t> fotufilmSeed;
        if (sampler) {
            fotufilmSeed = motion_seed(frame.at("fotufilm_seed"));
            require(priorFotufilmSeed != fotufilmSeed, "Fotufilm frame seed did not advance");
            priorFotufilmSeed = fotufilmSeed;
        }
        std::optional<MotionSeedPair> seeds;
        if (frame.contains("film_seed") || frame.contains("film_next_seed"))
            seeds = MotionSeedPair{motion_seed(frame.at("film_seed")), motion_seed(frame.at("film_next_seed"))};
        require(sampler || seeds, "Film-only replay requires both Film seeds");
        std::ostringstream number;
        number << std::setfill('0') << std::setw(6) << index;
        const auto stem = number.str();
        const auto filename = stem + ".f32";
        const auto cleanDensity = outputPath / "density" / (stem + "-clean.f32");
        const auto grainedDensity = outputPath / "density" / (stem + "-fotufilm.f32");
        const bool witness = index == 0 || index == frames.size() / 2 || index + 1 == frames.size();

        upload_source(frame);
        (void)render(MotionPass::Capture, seeds);
        clean = output;
        if (witness) {
            (void)render(MotionPass::Replay, seeds);
            require(std::memcmp(clean.data(), output.data(), imageBytes) == 0, "zero-grain motion replay changed output");
        }
        write_pixels(outputPath / "clean" / filename, clean);
        write_pixels(cleanDensity, density);
        if (projectionFraction)
            std::copy(density.begin(), density.end(), ungrainedDensity.begin());
        const auto [filmFrame, samplePitch] = render(MotionPass::Film, seeds);
        require(filmFrame.chromaFraction == expectedChroma, "admitted Film Colour differs from the motion host control");
        require(filmFrame.pitch == expectedWidth, "admitted Film footprint differs from the motion host controls");
        require(output != clean, "Film v1 did not change the motion frame");
        write_pixels(outputPath / "film" / filename, output);
        if (captureFilm)
            write_pixels(outputPath / "density" / (stem + "-film.f32"), density);
        Json projectionError;
        if (frame.contains("projection_witness_density")) {
            require(captureFilm && projectionFraction.has_value(), "projection witness requires Film density and a projection fraction");
            std::vector<float> unprojected(density.size());
            read_motion_pixels(frame.at("projection_witness_density").get<std::string>(), unprojected);
            project_motion_density(ungrainedDensity, unprojected, *projectionFraction);
            std::array<double, 3> maxError{};
            for (std::size_t sample = 0; sample < density.size(); ++sample) {
                require(std::isfinite(density[sample]), "nonfinite Film density in projection witness");
                const double error = std::abs(static_cast<double>(density[sample]) - unprojected[sample]);
                auto& maximum = maxError[sample / area];
                maximum = std::max(maximum, error);
            }
            projectionError = maxError;
        }

        if (sampler) {
            run_fotufilm(*sampler, test, longEdge, *fotufilmSeed, cleanDensity, grainedDensity, outputPath / "density" / (stem + "-fotufilm.json"));
            read_motion_pixels(grainedDensity, density);
            (void)render(MotionPass::Replay, seeds);
            require(output != clean, "Fotufilm did not change the motion frame");
            write_pixels(outputPath / "fotufilm" / filename, output);
        }
        if (projectionFraction) {
            project_motion_density(ungrainedDensity, density, *projectionFraction);
            write_pixels(outputPath / "density" / (stem + "-fotufilm-projected.f32"), density);
            (void)render(MotionPass::Replay, seeds);
            require(output != clean, "projected Fotufilm did not change the motion frame");
            write_pixels(outputPath / "fotufilm-projected" / filename, output);
        }
        if (!witness) {
            require(std::filesystem::remove(cleanDensity) && (!sampler || std::filesystem::remove(grainedDensity)),
                    "could not retire completed motion density intermediates");
        }

        reports << Json{
                       {"file", filename},
                       {"time", time},
                       {"frame_rate", frameRate},
                       {"width", filmFrame.width},
                       {"height", filmFrame.height},
                       {"stride", filmFrame.stride},
                       {"origin", {filmFrame.originX, filmFrame.originY}},
                       {"footprint", filmFrame.pitch},
                       {"sample_pitch", samplePitch},
                       {"film_long_edge_mm", readLongEdge},
                       {"film_format_preset", readPreset},
                       {"glare_active", std::get<int>(host.value(effect, "GlareActive").front())},
                       {"film_seed", filmFrame.seed},
                       {"film_next_seed", filmFrame.nextSeed},
                       {"film_seed_override", seeds.has_value()},
                       {"film_controls", filmControls},
                       {"film_chroma_fraction", filmFrame.chromaFraction},
                       {"fotufilm_projection_fraction", projectionFraction ? Json(*projectionFraction) : Json(nullptr)},
                       {"film_projection_max_abs_density_error", projectionError},
                       {"temporal_fraction", filmFrame.temporalFraction},
                       {"fotufilm_seed", fotufilmSeed ? Json(*fotufilmSeed) : Json(nullptr)},
                       {"zero_round_trip", witness ? Json(true) : Json(nullptr)},
                   }
                       .dump()
                << '\n'
                << std::flush;
        require(static_cast<bool>(reports), "could not record completed motion frame");
    }
    if (grainOffReference) {
        upload_source(frames.front());
        host.set_value(effect, "GrainActive", {0});
        auto* change = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "GrainActive"});
        host.set_property(change, kOfxPropTime, {frames.front().at("time").get<double>()});
        require(lifecycle.call(kOfxActionInstanceChanged, effect, change) == kOfxStatReplyDefault,
                "motion grain-off instance change failed");
        (void)render(MotionPass::GrainOff, std::nullopt);
        write_pixels(outputPath / "grain-off" / "000000.f32", output);
    }
    require(lifecycle.destroy(effect) == kOfxStatOK, "motion OFX destroy failed");
    device.release_images();
    device.close();
    require(lifecycle.unload() == kOfxStatOK && !host.callback_error(), "motion OFX unload failed");
    restore.restore();
    reports.close();
    require(static_cast<bool>(reports), "could not close motion frame receipt");
    std::ofstream completion(outputPath / "completed.json");
    completion << Json{{"frames", frames.size()}, {"context_restored", restore.restored}}.dump() << '\n';
    completion.close();
    require(static_cast<bool>(completion), "could not record completed motion session");
    return 0;
}
