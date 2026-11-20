// Copyright 2026 MUAStudio Inc. SPDX-License-Identifier: Apache-2.0

struct ApertureCase {
    const char* id;
    bool print;
    GrainGeometry proxy;
    GrainGeometry reference;
    double millimeters;
    double size;
    double colour;
    double time = 0.0;
};

constexpr GrainGeometry full_geometry(int width, int height) {
    return {{0.0, 0.0, static_cast<double>(width), static_cast<double>(height)},
            {0, 0, width, height},
            {1.0, 1.0}};
}

constexpr std::array kOfxApertures{
    ApertureCase{"clipped-quarter-direct", false, kClippedQuarter, kOffsetQuarter, 36.0, 1.0, 1.0},
    ApertureCase{"clipped-quarter-print", true, kClippedQuarter, kOffsetQuarter, 36.0, 1.0, 1.0},
    ApertureCase{"whole-period", true, {{0, 0, 2000, 1000}, {0, 0, 4, 2}, {1.0 / 500, 1.0 / 500}}, full_geometry(2000, 1000), 32.0, 0.5, 0.24, 37.375},
    ApertureCase{"unequal-4-by-8", true, {{0, 0, 4832, 96}, {0, 0, 1208, 12}, {0.25, 0.125}}, full_geometry(4832, 96), 36.0, 1.0, 1.0},
    ApertureCase{"unequal-8-by-4", true, {{0, 0, 96, 4832}, {0, 0, 12, 1208}, {0.125, 0.25}}, full_geometry(96, 4832), 36.0, 1.0, 1.0},
    ApertureCase{"8mm-proxy", true, kEightMmProxy, full_geometry(8192, 4096), 8.0, 4.0, 0.24},
};
constexpr std::string_view kApertureCaller = "film-ofx-apertures";
constexpr MotionSeedPair kApertureSeeds{1179208781U, 2741821501U};

struct DensityCapture {
    std::vector<float> clean;
    std::vector<float> film;
    float samplePitch = 0.F;
};

DensityCapture capture_aperture(const CudaDriver& driver, const ApertureCase& witness, const GrainGeometry& geometry) {
    const auto& bounds = geometry.bounds;
    const Case test{
        .id = witness.id,
        .routeName = witness.print ? "negative-print" : "negative-direct",
        .route = witness.print ? "NegativePrintScan" : "NegativeDirectScan",
        .components = 4,
        .width = bounds.x2 - bounds.x1,
        .height = bounds.y2 - bounds.y1,
        .grain = 1,
        .time = witness.time,
    };
    InstanceHost host;
    Lifecycle lifecycle(fj_ofx_plugin(0), host);
    require(lifecycle.load() == kOfxStatOK, "aperture OFX load");
    require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
            "aperture OFX context description");
    auto* effect = host.create_effect();
    author_parameters(host, effect, test);
    host.set_value(effect, "CameraFilmFormatPreset", {0});
    host.set_value(effect, "CameraFilmFormatMm", {witness.millimeters});
    host.set_value(effect, "CouplersActive", {0});
    host.set_value(effect, "GlareActive", {0});
    host.set_value(effect, "GrainModel", {std::string("film_v1")});
    host.set_value(effect, "FilmGrainAmountEv", {0.0});
    host.set_value(effect, "FilmGrainTexelPitchUm", {witness.size});
    host.set_value(effect, "FilmGrainChromaFraction", {witness.colour});
    const auto source = flat_grain_source(test);
    const auto bytes = source.size() * sizeof(float);
    DeviceBuffers device;
    require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create aperture stream");
    require_cuda(cudaMalloc(&device.source, bytes), "allocate aperture source");
    require_cuda(cudaMalloc(&device.destination, bytes), "allocate aperture destination");
    configure(host, effect, test, device);
    set_grain_geometry(host, effect, geometry);
    require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "aperture OFX instance creation");
    lifecycle.created(effect);

    const auto words = 3 * static_cast<std::size_t>(test.width) * test.height;
    DensityCapture capture{std::vector<float>(words), std::vector<float>(words)};
    const MotionSeedPair seeds = witness.time == 0.0 ? MotionSeedPair{kApertureSeeds[0], kApertureSeeds[0]} : kApertureSeeds;
    auto render = [&](MotionPass pass, std::span<float> plane) {
        MotionDensity tap{.pass = pass, .density = plane, .seeds = seeds, .captureFilm = pass == MotionPass::Film};
        const MotionDensityScope scope(tap);
        std::vector<std::uint32_t> output;
        (void)run_pass(driver, lifecycle, host, effect, test, "cold", {source, nullptr, {}}, device, output, true);
        require(tap.visited && tap.filmCaptured == tap.captureFilm, "aperture density tap did not complete");
        return tap.samplePitch;
    };
    capture.samplePitch = render(MotionPass::Capture, capture.clean);
    require(render(MotionPass::Film, capture.film) == capture.samplePitch, "sample pitch changed on a paused render");
    require(lifecycle.destroy(effect) == kOfxStatOK, "aperture OFX instance destroy");
    device.close();
    require(lifecycle.unload() == kOfxStatOK, "aperture OFX unload");
    return capture;
}

template <typename Reference>
void compare_aperture_grain(const ApertureCase& witness, const DensityCapture& proxy, Reference reference) {
    const int width = witness.proxy.bounds.x2 - witness.proxy.bounds.x1;
    const int height = witness.proxy.bounds.y2 - witness.proxy.bounds.y1;
    const int stepX = static_cast<int>(1.0 / witness.proxy.scale[0] + 0.5);
    const int stepY = static_cast<int>(1.0 / witness.proxy.scale[1] + 0.5);
    const auto pixels = static_cast<std::size_t>(width) * height;
    constexpr auto infinity = std::numeric_limits<double>::infinity();
    std::array<double, 3> maxError{}, squaredError{};
    std::array<double, 3> minimum{infinity, infinity, infinity};
    std::array<double, 3> maximum{-infinity, -infinity, -infinity};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::array<double, 3> cleanLight{}, grainedLight{};
            for (int fy = y * stepY; fy < (y + 1) * stepY; ++fy) {
                for (int fx = x * stepX; fx < (x + 1) * stepX; ++fx) {
                    const auto [clean, grained] = reference(fx, fy);
                    for (std::size_t channel = 0; channel < clean.size(); ++channel) {
                        cleanLight[channel] += std::pow(10.0, -static_cast<double>(clean[channel]));
                        grainedLight[channel] += std::pow(10.0, -static_cast<double>(grained[channel]));
                    }
                }
            }
            for (std::size_t channel = 0; channel < cleanLight.size(); ++channel) {
                const auto i = channel * pixels + static_cast<std::size_t>(y) * width + x;
                const double measured = static_cast<double>(proxy.film[i]) - proxy.clean[i];
                const double error = measured - std::log10(cleanLight[channel] / grainedLight[channel]);
                maxError[channel] = std::max(maxError[channel], std::abs(error));
                squaredError[channel] += error * error;
                minimum[channel] = std::min(minimum[channel], measured);
                maximum[channel] = std::max(maximum[channel], measured);
            }
        }
    }
    bool passed = true;
    for (std::size_t channel = 0; channel < maxError.size(); ++channel) {
        std::cout << witness.id << " channel=" << channel << " sample_pitch=" << proxy.samplePitch
                  << " max_abs_signed_grain_error_D=" << maxError[channel]
                  << " error_rms_D=" << std::sqrt(squaredError[channel] / static_cast<double>(pixels)) << '\n';
        passed &= minimum[channel] < maximum[channel] && maxError[channel] <= 2e-6 && std::isfinite(squaredError[channel]);
    }
    require(passed, "OFX proxy lost grain variation or disagrees with Full transmission reduction");
}

void check_aperture(const CudaDriver& driver, const ApertureCase& witness) {
    const auto proxy = capture_aperture(driver, witness, witness.proxy);
    const auto id = std::string_view(witness.id);
    if (id.starts_with("clipped-quarter")) {
        const auto square = capture_aperture(driver, witness, witness.reference);
        std::cout << id << " clipped_pitch=" << proxy.samplePitch << " square_pitch=" << square.samplePitch << '\n';
        const auto same_word = [](float left, float right) {
            return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
        };
        require(std::ranges::equal(proxy.clean, square.clean, same_word), "clipped and square Quarter clean C/M/Y words differ");
        require(std::ranges::equal(proxy.film, square.film, same_word) && proxy.film != proxy.clean,
                "clipped and square Quarter grained C/M/Y words differ");
        require(std::bit_cast<std::uint32_t>(proxy.samplePitch) == std::bit_cast<std::uint32_t>(square.samplePitch),
                "clipped Quarter selected a different sample pitch");
        return;
    }
    if (id == "8mm-proxy") {
        constexpr float pitch = 0.244140625F;
        require(proxy.samplePitch == pitch, "8 mm proxy did not admit the exact Full cell pitch");
        auto bank = GrainPrototype::read_film_bank(JUICER_GRAIN_TEST_BANK, 37.5F, JUICER_GRAIN_TEST_TABLES);
        GrainPrototype::calibrate_subtexel(bank, pitch);
        const auto pixels = proxy.clean.size() / 3;
        std::array<float, 3> clean{};
        for (std::size_t channel = 0; channel < clean.size(); ++channel) {
            const auto plane = std::span(proxy.clean).subspan(channel * pixels, pixels);
            clean[channel] = plane.front();
            require(std::ranges::all_of(plane, [density = plane.front()](float value) {
                        return value == density;
                    }),
                    "8 mm bank reference requires a flat developed plane");
        }
        compare_aperture_grain(witness, proxy, [&](int x, int y) {
            std::array<float, 3> grained{};
            for (int channel = 0; channel < 3; ++channel) {
                const float density = clean[channel];
                const auto sample = GrainPrototype::evaluate_sample(bank, {pitch, x, y, kApertureSeeds[0], channel, density, true});
                grained[channel] = static_cast<float>(density + 0.25 * sample.delta);
            }
            project_motion_density(clean, grained, static_cast<float>(witness.colour));
            return std::pair{clean, grained};
        });
        return;
    }
    const auto full = capture_aperture(driver, witness, witness.reference);
    std::cout << id << " proxy_sample_pitch=" << proxy.samplePitch << " full_sample_pitch=" << full.samplePitch << '\n';
    const int width = witness.reference.bounds.x2;
    const auto pixels = full.clean.size() / 3;
    compare_aperture_grain(witness, proxy, [&](int x, int y) {
        const auto pixel = static_cast<std::size_t>(y) * width + x;
        std::array<float, 3> clean{}, grained{};
        for (std::size_t channel = 0; channel < clean.size(); ++channel) {
            const auto i = channel * pixels + pixel;
            clean[channel] = full.clean[i];
            grained[channel] = full.film[i];
        }
        return std::pair{clean, grained};
    });
    require(proxy.samplePitch == full.samplePitch, "OFX proxy table differs from the Full sample table");
}

int grain_preview_apertures() {
    const CudaDriver driver;
    driver.check(driver.init(0), "initialize aperture CUDA driver");
    CUcontext original = nullptr;
    driver.check(driver.get_current(&original), "read aperture CUDA context");
    ContextRestore restore(driver, original);
    require_cuda(cudaSetDevice(0), "select aperture CUDA device");
    JuicerTests::CaseLog cases;
    std::vector<std::string> ids;
    ids.reserve(kOfxApertures.size());
    for (const auto& witness : kOfxApertures)
        ids.emplace_back(witness.id);
    cases.membership(kApertureCaller, ids);
    bool passed = true;
    for (const auto& witness : kOfxApertures) {
        JuicerTests::CaseEvent event(kApertureCaller, witness.id);
        try {
            check_aperture(driver, witness);
            event.passed();
        } catch (const std::exception& error) {
            std::cerr << witness.id << ": " << error.what() << '\n';
            passed = false;
        }
    }
    cases.finish();
    restore.restore();
    return passed ? 0 : 1;
}
