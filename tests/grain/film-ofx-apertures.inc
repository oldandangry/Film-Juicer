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
    float gray = 0.18F;
};

constexpr GrainGeometry full_geometry(int width, int height) {
    return {{0.0, 0.0, static_cast<double>(width), static_cast<double>(height)},
            {0, 0, width, height},
            {1.0, 1.0}};
}

constexpr double kSerialBorderScale = 27.0 / 139.0;
constexpr double kBatchedBorderScale = 3.0 / 273.0;
constexpr GrainGeometry standard_border(int fullWidth, int width, int height, double scale) {
    return {full_geometry(fullWidth, 96).canonical, {0, 0, width, height}, {scale, scale}, {scale, scale}};
}

constexpr std::array kOfxApertures{
    ApertureCase{"clipped-quarter-direct", false, kClippedQuarter, kOffsetQuarter, 36.0, 1.0, 1.0},
    ApertureCase{"clipped-quarter-print", true, kClippedQuarter, kOffsetQuarter, 36.0, 1.0, 1.0},
    ApertureCase{"whole-period", true, {{0, 0, 2000, 1000}, {0, 0, 4, 2}, {1.0 / 500, 1.0 / 500}}, full_geometry(2000, 1000), 32.0, 0.5, 0.24, 37.375},
    ApertureCase{"unequal-4-by-8", true, {{0, 0, 4832, 96}, {0, 0, 1208, 12}, {0.25, 0.125}}, full_geometry(4832, 96), 36.0, 1.0, 1.0},
    ApertureCase{"unequal-8-by-4", true, {{0, 0, 96, 4832}, {0, 0, 12, 1208}, {0.125, 0.25}}, full_geometry(96, 4832), 36.0, 1.0, 1.0},
    ApertureCase{"8mm-proxy", true, kEightMmProxy, full_geometry(8192, 4096), 8.0, 4.0, 0.24},
    ApertureCase{"calibrated-2-by-4", true, {{0, 0, 4832, 96}, {0, 0, 2416, 24}, {0.5, 0.25}}, full_geometry(4832, 96), 36.0, 1.0, 1.0},
    ApertureCase{"calibrated-4-by-2", true, {{0, 0, 4832, 96}, {0, 0, 1208, 48}, {0.25, 0.5}}, full_geometry(4832, 96), 36.0, 1.0, 1.0},
    ApertureCase{"shifted-37-5-equivalent", true, kShiftedQuarter, {kShiftedRegion, {1, 4, 3840, 8}, {1, 1}}, 36.0, 1.0, 1.0},
    ApertureCase{"standard-27-over-139-border", false, standard_border(139, 27, 19, kSerialBorderScale), full_geometry(139, 96), 36.0, 0.5, 1.0},
    ApertureCase{"offset-wide-clipped", false, {{0, 4, 4832, 21}, {0, 0, 1208, 3}, {0.25, 0.125}}, {{0, 4, 4832, 21}, {0, 4, 4832, 21}, {1, 1}}, 36.0, 1.0, 0.24, 37.375, 0.31F},
};
constexpr std::string_view kApertureCaller = "film-ofx-apertures";
constexpr MotionSeedPair kApertureSeeds{1179208781U, 2741821501U};

struct DensityCapture {
    std::vector<float> clean;
    std::vector<float> film;
    float samplePitch = 0.F;
    JuicerCuda::FilmGrain::ProxyGrid grid{};
    std::array<int, 2> placement{};
    std::array<std::uint8_t, 4> calibratedBorder{};
};

struct ApertureSwitch {
    const GrainGeometry& alternate;
    DensityCapture switched{};
    DensityCapture returned{};
};

bool same_density_word(float left, float right) {
    return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
}

bool same_proxy_grid(const JuicerCuda::FilmGrain::ProxyGrid& left, const JuicerCuda::FilmGrain::ProxyGrid& right) {
    return std::ranges::equal(left.extent, right.extent) &&
           std::ranges::equal(left.step, right.step) &&
           std::ranges::equal(left.origin, right.origin);
}

JuicerCuda::FilmGrain::ProxyGrid realized_grid(const GrainGeometry& geometry) {
    const auto& [x1, y1, x2, y2] = geometry.canonical;
    return {{std::ceil(x2) - std::floor(x1), std::ceil(y2) - std::floor(y1)},
            {1.0 / geometry.scale[0], 1.0 / geometry.scale[1]},
            {geometry.bounds.x1 / geometry.scale[0] - std::floor(x1),
             geometry.bounds.y1 / geometry.scale[1] - std::floor(y1)}};
}

bool same_apertures(const JuicerCuda::FilmGrain::ProxyGrid& left, const JuicerCuda::FilmGrain::ProxyGrid& right, std::array<int, 2> size) {
    const auto edge = [](const JuicerCuda::FilmGrain::ProxyGrid& grid, std::size_t axis, int i) {
        return std::clamp(std::fma(i, grid.step[axis], grid.origin[axis]), 0.0, grid.extent[axis]);
    };
    for (std::size_t axis = 0; axis < size.size(); ++axis) {
        for (int i = 0; i <= size[axis]; ++i) {
            if (edge(left, axis, i) != edge(right, axis, i))
                return false;
        }
    }
    return std::ranges::equal(left.extent, right.extent);
}

DensityCapture capture_aperture(const CudaDriver& driver, const ApertureCase& witness, const GrainGeometry& geometry, const GrainGeometry* alternate = nullptr, ApertureSwitch* sequence = nullptr) {
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
    const auto source = flat_grain_source(test, witness.gray);
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
    const auto render_output = [&] {
        std::vector<std::uint32_t> output;
        (void)run_pass(driver, lifecycle, host, effect, test, "cold", {source, nullptr, {}}, device, output, true);
        return output;
    };
    const auto rejects_geometry = [&](std::string_view name) {
        host.clear_calls();
        const auto sourceBefore = host.image_release_count(effect, "Source");
        const auto outputBefore = host.image_release_count(effect, "Output");
        const auto status = lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args());
        const auto messages = std::ranges::count_if(host.calls(), [](const auto& call) {
            return call.operation == "message" && call.property == kOfxMessageError &&
                   call.values.size() == 2 && call.values.front() == InstanceValue{std::string("FilmJuicerFilmGrainGeometry")};
        });
        const auto released = host.image_release_count(effect, "Source") == sourceBefore + 1 &&
                              host.image_release_count(effect, "Output") == outputBefore + 1;
        std::cout << witness.id << " geometry=" << name << " status=" << status
                  << " diagnostic_count=" << messages << " released=" << released << '\n';
        set_grain_geometry(host, effect, geometry);
        return status == kOfxStatFailed && !host.callback_error() && messages == 1 && released;
    };
    constexpr double fractionalScale = 962.0 / 3839.5;
    bool rejectedTight = true;
    if (geometry.scale[0] == fractionalScale && test.width == 963) {
        auto tight = geometry;
        const bool leading = bounds.x1 < 0;
        if (leading)
            ++tight.bounds.x1;
        else
            --tight.bounds.x2;
        set_grain_geometry(host, effect, tight);
        rejectedTight = rejects_geometry(leading ? "positive-leading-column-omitted" : "positive-trailing-column-omitted");
    }
    const auto render = [&](DensityCapture& target, MotionPass pass, std::span<float> plane) {
        MotionDensity tap{
            .pass = pass,
            .density = plane,
            .seeds = seeds,
            .captureFilm = pass == MotionPass::Film,
            .poisonProxyWorkspace = pass == MotionPass::Film && geometry.scale[0] == kBatchedBorderScale && test.width == 4,
        };
        const MotionDensityScope scope(tap);
        (void)render_output();
        require(tap.visited && tap.filmLaunched == (pass == MotionPass::Film) && tap.filmCaptured == tap.captureFilm,
                "aperture density tap did not complete");
        target.grid = tap.admitted.proxy;
        target.placement = {tap.admitted.originX, tap.admitted.originY};
        std::ranges::copy(tap.admitted.calibratedBorder, target.calibratedBorder.begin());
        return tap.samplePitch;
    };
    const auto capture_density = [&](DensityCapture& target) {
        target.samplePitch = render(target, MotionPass::Capture, target.clean);
        require(render(target, MotionPass::Film, target.film) == target.samplePitch, "sample pitch changed on a paused render");
    };
    const auto switch_geometry = [&](const GrainGeometry& next) {
        set_grain_geometry(host, effect, next);
        DensityCapture switched{std::vector<float>(words), std::vector<float>(words)};
        capture_density(switched);
        set_grain_geometry(host, effect, geometry);
        DensityCapture returned{std::vector<float>(words), std::vector<float>(words)};
        capture_density(returned);
        return std::pair{std::move(switched), std::move(returned)};
    };
    capture_density(capture);
    if (alternate) {
        const auto realized = realized_grid(geometry);
        const auto alternative = realized_grid(*alternate);
        require(alternate->bounds.x2 - alternate->bounds.x1 == test.width &&
                    alternate->bounds.y2 - alternate->bounds.y1 == test.height &&
                    same_apertures(realized, alternative, {test.width, test.height}),
                "paused witness encodings must realize identical clamped edges");
        const auto [switched, returned] = switch_geometry(*alternate);
        std::size_t changed = 0;
        for (std::size_t i = 0; i < words; ++i)
            changed += !same_density_word(capture.film[i], switched.film[i]);
        std::cout << "paused-encoding callbacks=complete extent=" << realized.extent[0] << ',' << realized.extent[1]
                  << " step=" << realized.step[0] << ',' << realized.step[1]
                  << " origin=" << realized.origin[0] << ',' << realized.origin[1]
                  << " original_pitch=" << capture.samplePitch << " switched_pitch=" << switched.samplePitch
                  << " returned_pitch=" << returned.samplePitch << " changed_supported_words=" << changed << '\n';
        require(same_density_word(returned.samplePitch, capture.samplePitch) &&
                    std::ranges::equal(returned.film, capture.film, same_density_word),
                "paused return did not restore the original table and words");
        for (const auto* other : {&switched, &returned}) {
            const bool activeGrid = other->grid.step[0] != 0.0 || other->grid.step[1] != 0.0;
            require(!activeGrid || same_proxy_grid(other->grid, other == &switched ? alternative : realized),
                    "active CUDA grid differs from the realized edges");
            require(same_density_word(other->samplePitch, capture.samplePitch) &&
                        std::ranges::equal(other->clean, capture.clean, same_density_word) &&
                        std::ranges::equal(other->film, capture.film, same_density_word) &&
                        same_proxy_grid(other->grid, capture.grid) && other->placement == capture.placement &&
                        other->calibratedBorder == capture.calibratedBorder,
                    "paused encoding switch changed Film table, grid, placement or density words");
        }
    }
    if (sequence) {
        const auto& nextBounds = sequence->alternate.bounds;
        require(!alternate && nextBounds.x2 - nextBounds.x1 == test.width &&
                    nextBounds.y2 - nextBounds.y1 == test.height,
                "paused aperture witness must keep the observed dimensions");
        auto [switched, returned] = switch_geometry(sequence->alternate);
        sequence->switched = std::move(switched);
        sequence->returned = std::move(returned);
    }
    require(rejectedTight, "standard-scale image omitted a positive half-cell without a geometry failure");
    const auto id = std::string_view(witness.id);
    if (id == "standard-27-over-139-border" && test.width == 28) {
        const auto expected = render_output();
        const auto reject = [&](std::string_view name) {
            require(rejects_geometry(name), "rejected border did not report geometry and release both images");
            require(render_output() == expected, "border rejection changed the next successful render");
            std::cout << id << " rejected_and_recovered=" << name << '\n';
        };
        set_grain_geometry(host, effect, standard_border(139, 26, 19, kSerialBorderScale));
        reject("lost-column");
        set_grain_geometry(host, effect, standard_border(139, 29, 19, kSerialBorderScale));
        reject("second-empty-column");
        host.set_image_property(effect, "Output", kOfxImageEffectPropRenderScale, {1.0, 1.0});
        reject("image-scale");
        host.set_image_property(effect, "Output", kOfxImagePropBounds, {0, 0, 27, 19});
        reject("image-bounds");
        host.set_property(host.action_args(), kOfxImageEffectPropRenderWindow, {0, 0, 27, 19});
        reject("partial-window");
        auto nonfinite = geometry;
        nonfinite.canonical.x2 = std::numeric_limits<double>::infinity();
        set_grain_geometry(host, effect, nonfinite);
        reject("nonfinite-canonical");
    }
    if (id == "clipped-quarter-direct" && geometry.scale == kClippedQuarter.scale) {
        std::string diagnostic;
        host.set_message_callback([&diagnostic](std::string_view, std::string_view, std::string_view text) {
            diagnostic = text;
            return kOfxStatOK;
        });
        MotionDensity invalid{.density = {}, .seeds = std::nullopt};
        {
            const MotionDensityScope scope(invalid);
            require(lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args()) == kOfxStatErrFatal,
                    "malformed aperture tap should fail through the OFX callback");
        }
        require(!invalid.visited && diagnostic.starts_with("grain motion density transfer failed: "),
                "malformed aperture tap was not rejected at density transfer");
        host.set_message_callback({});
        const auto untapped = render_output();
        MotionDensity passThrough{.pass = MotionPass::Film, .density = capture.film, .seeds = std::nullopt};
        {
            const MotionDensityScope scope(passThrough);
            require(render_output() == untapped, "untapped Film must match one normal launch with the Rust seeds");
        }
        require(passThrough.visited && passThrough.filmLaunched && !passThrough.filmCaptured,
                "pass-through Film launch did not complete without readback");

        MotionDensity failedLaunch{.pass = MotionPass::Film, .density = capture.film, .seeds = std::nullopt, .captureFilm = true};
        {
            const MotionDensityScope scope(failedLaunch);
            const auto planeSize = words / 3;
            // Valid device planes would permit readback; the real sampler rejects the missing bank.
            const auto result = JuicerCuda::FilmGrain::launch_developed(
                nullptr, {}, passThrough.admitted, {device.destination, device.destination + planeSize, device.destination + 2 * planeSize}, device.stream);
            require(result.status == cudaErrorInvalidValue && !result.failure &&
                        failedLaunch.visited && !failedLaunch.filmLaunched && !failedLaunch.filmCaptured,
                    "a failed Film launch must not be replaced by readback");
        }
        require(render_output() == untapped, "failed launch left the replay tap active");
    }
    require(lifecycle.destroy(effect) == kOfxStatOK, "aperture OFX instance destroy");
    device.close();
    require(lifecycle.unload() == kOfxStatOK, "aperture OFX unload");
    return capture;
}

template <typename Reference>
void compare_aperture_grain(const ApertureCase& witness, const DensityCapture& proxy, Reference reference, std::size_t expectedEmpty = 0) {
    const int width = witness.proxy.bounds.x2 - witness.proxy.bounds.x1;
    const int height = witness.proxy.bounds.y2 - witness.proxy.bounds.y1;
    const auto& [x1, y1, x2, y2] = witness.proxy.canonical;
    const auto interval = [](int pixel, std::array<double, 2> canonical, double scale) {
        const double minimum = std::floor(canonical[0]);
        const double maximum = std::ceil(canonical[1]);
        return std::array{std::clamp(pixel / scale, minimum, maximum) - minimum,
                          std::clamp((pixel + 1.0) / scale, minimum, maximum) - minimum};
    };
    const auto pixels = static_cast<std::size_t>(width) * height;
    constexpr auto infinity = std::numeric_limits<double>::infinity();
    std::array<double, 3> maxError{}, squaredError{};
    std::array<double, 3> measuredSum{}, referenceSum{}, measuredSquared{}, referenceSquared{}, cross{};
    std::array<double, 3> minimum{infinity, infinity, infinity};
    std::array<double, 3> maximum{-infinity, -infinity, -infinity};
    std::size_t positive = 0, empty = 0;
    for (int y = 0; y < height; ++y) {
        const auto [top, bottom] = interval(witness.proxy.bounds.y1 + y, {y1, y2}, witness.proxy.scale[1]);
        for (int x = 0; x < width; ++x) {
            const auto [left, right] = interval(witness.proxy.bounds.x1 + x, {x1, x2}, witness.proxy.scale[0]);
            const auto pixel = static_cast<std::size_t>(y) * width + x;
            if (right == left || bottom == top) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const auto i = channel * pixels + pixel;
                    require(same_density_word(proxy.film[i], proxy.clean[i]), "empty aperture changed developed C/M/Y");
                }
                ++empty;
                continue;
            }
            ++positive;
            std::array<double, 3> cleanLight{}, grainedLight{};
            for (int fy = static_cast<int>(top); fy < static_cast<int>(std::ceil(bottom)); ++fy) {
                const double h = std::min(bottom, fy + 1.0) - std::max(top, static_cast<double>(fy));
                for (int fx = static_cast<int>(left); fx < static_cast<int>(std::ceil(right)); ++fx) {
                    const double weight = h * (std::min(right, fx + 1.0) - std::max(left, static_cast<double>(fx)));
                    const auto [clean, grained] = reference(fx, fy);
                    for (std::size_t channel = 0; channel < clean.size(); ++channel) {
                        cleanLight[channel] += weight * std::pow(10.0, -static_cast<double>(clean[channel]));
                        grainedLight[channel] += weight * std::pow(10.0, -static_cast<double>(grained[channel]));
                    }
                }
            }
            for (std::size_t channel = 0; channel < cleanLight.size(); ++channel) {
                const auto i = channel * pixels + pixel;
                const double measured = static_cast<double>(proxy.film[i]) - proxy.clean[i];
                const double expected = std::log10(cleanLight[channel] / grainedLight[channel]);
                const double error = measured - expected;
                require(std::isfinite(error), "nonfinite signed aperture density error");
                if (x == 3 && y == 0 && std::string_view(witness.id) == "shifted-37-5-equivalent")
                    std::cout << witness.id << " channel=" << channel << " signed_seam_error_D=" << error << '\n';
                maxError[channel] = std::max(maxError[channel], std::abs(error));
                squaredError[channel] += error * error;
                measuredSum[channel] += measured;
                referenceSum[channel] += expected;
                measuredSquared[channel] += measured * measured;
                referenceSquared[channel] += expected * expected;
                cross[channel] += measured * expected;
                minimum[channel] = std::min(minimum[channel], measured);
                maximum[channel] = std::max(maximum[channel], measured);
            }
        }
    }
    require(positive > 0, "no positive aperture coverage");
    require(empty == expectedEmpty, "wrong number of empty aperture pixels");
    const double count = static_cast<double>(positive);
    bool passed = true;
    for (std::size_t channel = 0; channel < maxError.size(); ++channel) {
        const double measuredVariance = measuredSquared[channel] - measuredSum[channel] * measuredSum[channel] / count;
        const double referenceVariance = referenceSquared[channel] - referenceSum[channel] * referenceSum[channel] / count;
        const double correlation = (cross[channel] - measuredSum[channel] * referenceSum[channel] / count) /
                                   std::sqrt(measuredVariance * referenceVariance);
        std::cout << witness.id << " channel=" << channel << " sample_pitch=" << proxy.samplePitch
                  << " max_abs_signed_grain_error_D=" << maxError[channel]
                  << " error_rms_D=" << std::sqrt(squaredError[channel] / count)
                  << " grain_rms_D=" << std::sqrt(measuredSquared[channel] / count)
                  << " reference_rms_D=" << std::sqrt(referenceSquared[channel] / count)
                  << " correlation=" << correlation << " empty_pixels=" << empty << '\n';
        passed &= minimum[channel] < maximum[channel] && maxError[channel] <= 2e-6 && std::isfinite(correlation);
    }
    require(passed, "OFX proxy lost grain variation or disagrees with Full transmission reduction");
}

void compare_with_full(const ApertureCase& witness, const DensityCapture& proxy, const DensityCapture& full, std::size_t expectedEmpty = 0) {
    const int width = witness.reference.bounds.x2 - witness.reference.bounds.x1;
    const auto pixels = full.clean.size() / 3;
    const auto proxyPixels = proxy.clean.size() / 3;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const auto plane = std::span(full.clean).subspan(channel * pixels, pixels);
        const auto reduced = std::span(proxy.clean).subspan(channel * proxyPixels, proxyPixels);
        const auto fixedDensity = [word = plane.front()](float density) {
            return same_density_word(word, density);
        };
        require(std::ranges::all_of(plane, fixedDensity) && std::ranges::all_of(reduced, fixedDensity),
                "weighted comparison requires the same fixed developed C/M/Y");
    }
    compare_aperture_grain(
        witness, proxy, [&](int x, int y) {
            const auto pixel = static_cast<std::size_t>(y) * width + x;
            std::array<float, 3> clean{}, grained{};
            for (std::size_t channel = 0; channel < clean.size(); ++channel) {
                const auto i = channel * pixels + pixel;
                clean[channel] = full.clean[i];
                grained[channel] = full.film[i];
            }
            return std::pair{clean, grained};
        },
        expectedEmpty);
    require(proxy.samplePitch == full.samplePitch, "OFX proxy table differs from the Full sample table");
}

void check_border(const CudaDriver& driver, ApertureCase witness, std::uint64_t expectedSamples) {
    using JuicerCuda::FilmGrain::kProxySerialSamples;
    using JuicerCuda::FilmGrain::proxy_sample_bound;
    const int width = witness.proxy.bounds.x2;
    const int height = witness.proxy.bounds.y2;
    const auto full = capture_aperture(driver, witness, witness.reference);
    const auto reduced = capture_aperture(driver, witness, witness.proxy);
    compare_with_full(witness, reduced, full);
    const int fullWidth = witness.reference.bounds.x2;
    witness.proxy = standard_border(fullWidth, width + 1, height, witness.proxy.scale[0]);
    const auto outward = capture_aperture(driver, witness, witness.proxy);
    compare_with_full(witness, outward, full, static_cast<std::size_t>(height));
    const auto bound = proxy_sample_bound(outward.grid.extent, outward.grid.step);
    require(bound == expectedSamples && proxy_sample_bound(reduced.grid.extent, reduced.grid.step) == bound,
            "border did not exercise the required proxy dispatch");
    std::cout << witness.id << " tight_width=" << width << " outward_width=" << width + 1
              << " sample_bound=" << bound << " kernel=" << (bound <= kProxySerialSamples ? "serial" : "batched") << '\n';
    for (int row = 0; row < 3 * height; ++row) {
        const auto tightStart = static_cast<std::size_t>(row) * width;
        const auto outwardStart = static_cast<std::size_t>(row) * (width + 1);
        require(std::ranges::equal(std::span(reduced.film).subspan(tightStart, width),
                                   std::span(outward.film).subspan(outwardStart, width),
                                   same_density_word),
                "outward border changed the supported pixels");
    }
}

struct EmptyBorderCase {
    std::string_view name;
    GrainGeometry outward{};
    OfxRectI tight{};
};

constexpr GrainGeometry transpose_geometry(const GrainGeometry& geometry) {
    const auto& [x1, y1, x2, y2] = geometry.canonical;
    const auto& [left, top, right, bottom] = geometry.bounds;
    return {{y1, x1, y2, x2}, {top, left, bottom, right}, {geometry.scale[1], geometry.scale[0]}, {geometry.imageScale[1], geometry.imageScale[0]}};
}

constexpr double kOffsetTrailingScale = 27.0 / 161.0;
constexpr double kOffsetLeadingScale = 53.0 / 137.0;
constexpr GrainGeometry kOffsetTrailing{{144, 0, 161, 96}, {24, 0, 28, 17}, {kOffsetTrailingScale, kOffsetTrailingScale}, {kOffsetTrailingScale, kOffsetTrailingScale}};
constexpr GrainGeometry kOffsetLeading{{-137, 0, 2, 96}, {-54, 0, 1, 38}, {kOffsetLeadingScale, kOffsetLeadingScale}, {kOffsetLeadingScale, kOffsetLeadingScale}};
constexpr std::array kEmptyBorderCases{
    EmptyBorderCase{"offset-trailing-x", kOffsetTrailing, {24, 0, 27, 17}},
    EmptyBorderCase{"offset-leading-x", kOffsetLeading, {-53, 0, 1, 38}},
    EmptyBorderCase{"offset-trailing-y", transpose_geometry(kOffsetTrailing), {0, 24, 17, 27}},
    EmptyBorderCase{"offset-leading-y", transpose_geometry(kOffsetLeading), {0, -53, 38, 1}},
    EmptyBorderCase{"below-cutoff-trailing", {{-3501, 0, 100, 96}, {-981, 0, 29, 27}, {0.28, 0.28}, {0.28, 0.28}}, {-981, 0, 28, 27}},
    EmptyBorderCase{"mixed-leading-x-trailing-y", {{-137, -356, 2, 137}, {-54, -138, 1, 54}, {kOffsetLeadingScale, kOffsetLeadingScale}, {kOffsetLeadingScale, kOffsetLeadingScale}}, {-53, -138, 1, 53}},
};

template <typename Check>
void aperture_subcase(std::string_view name, bool& passed, Check check) {
    try {
        check();
        std::cout << "aperture_subcase=" << name << " passed\n";
    } catch (const std::exception& error) {
        std::cerr << "aperture_subcase=" << name << " failed: " << error.what() << '\n';
        passed = false;
    }
}

void compare_calibrated_border(const DensityCapture& reference, const DensityCapture& actual, int width, int height, std::array<std::uint8_t, 4> border) {
    using JuicerCuda::FilmGrain::ProxyGrid;
    require(actual.samplePitch == reference.samplePitch && same_proxy_grid(actual.grid, ProxyGrid{}) &&
                actual.placement == reference.placement && actual.calibratedBorder == border,
            "calibrated aperture selected a different table, placement or border");
    const auto [left, top, right, bottom] = border;
    const auto supportedWidth = width - left - right;
    const auto supportedHeight = height - top - bottom;
    std::size_t empty = 0;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const auto i = (channel * height + y) * width + x;
                if (x < left || y < top || x >= width - right || y >= height - bottom) {
                    require(same_density_word(actual.film[i], actual.clean[i]), "calibrated empty pixel changed its density word");
                    ++empty;
                } else {
                    const auto j = (channel * supportedHeight + y - top) * supportedWidth + x - left;
                    require(same_density_word(actual.clean[i], reference.clean[j]) &&
                                same_density_word(actual.film[i], reference.film[j]),
                            "calibrated aperture changed a supported density word");
                }
            }
        }
    }
    require(reference.film != reference.clean, "calibrated reference must have active Film");
    std::cout << "calibrated-border sample_pitch=" << actual.samplePitch
              << " empty_pixels=" << empty / 3 << " supported_words=exact\n";
}

void check_offset_border(const CudaDriver& driver, ApertureCase witness, const EmptyBorderCase& border, bool& passed) {
    witness.size = 1.0;
    witness.proxy = border.outward;
    const auto& [x1, y1, x2, y2] = border.outward.canonical;
    const OfxRectI fullBounds{static_cast<int>(std::floor(x1)), static_cast<int>(std::floor(y1)), static_cast<int>(std::ceil(x2)), static_cast<int>(std::ceil(y2))};
    witness.reference = {{static_cast<double>(fullBounds.x1), static_cast<double>(fullBounds.y1), static_cast<double>(fullBounds.x2), static_cast<double>(fullBounds.y2)}, fullBounds};
    const auto& wide = border.outward.bounds;
    const auto& tight = border.tight;
    const auto wideWidth = static_cast<std::size_t>(wide.x2 - wide.x1);
    const auto wideHeight = static_cast<std::size_t>(wide.y2 - wide.y1);
    const auto tightWidth = static_cast<std::size_t>(tight.x2 - tight.x1);
    const auto tightHeight = static_cast<std::size_t>(tight.y2 - tight.y1);
    std::optional<DensityCapture> outward, reduced;
    const std::string name{border.name};
    const bool calibrated = border.name == "below-cutoff-trailing";
    const auto reference = capture_aperture(driver, witness, calibrated ? full_geometry(static_cast<int>(tightWidth), static_cast<int>(tightHeight)) : witness.reference);
    aperture_subcase(name + "/outward", passed, [&] {
        auto value = capture_aperture(driver, witness, witness.proxy);
        const std::array removedLeading{tight.x1 > wide.x1, tight.y1 > wide.y1};
        const std::array removedTrailing{tight.x2 < wide.x2, tight.y2 < wide.y2};
        const std::array count{wide.x2 - wide.x1, wide.y2 - wide.y1};
        const auto grid = realized_grid(witness.proxy);
        for (std::size_t axis = 0; axis < count.size(); ++axis) {
            const auto raw = [&](double edge) {
                return std::fma(edge, grid.step[axis], grid.origin[axis]);
            };
            if (removedLeading[axis]) {
                require(raw(1.0) <= 0.0 && raw(2.0) > 0.0, "leading empty cell lacks an adjacent positive cell");
                std::cout << name << " axis=" << axis << " leading_raw=" << raw(1.0) << '\n';
            }
            if (removedTrailing[axis]) {
                const double beyond = raw(count[axis] - 1.0) - grid.extent[axis];
                require(beyond >= 0.0 && raw(count[axis] - 2.0) < grid.extent[axis],
                        "trailing empty cell lacks an adjacent positive cell");
                std::cout << name << " axis=" << axis << " trailing_raw_minus_extent=" << beyond << '\n';
            }
        }
        if (calibrated)
            compare_calibrated_border(reference, value, count[0], count[1], {0, 0, 1, 0});
        else
            compare_with_full(witness, value, reference, wideWidth * wideHeight - tightWidth * tightHeight);
        outward = std::move(value);
    });
    aperture_subcase(name + "/tight", passed, [&] {
        witness.proxy.bounds = tight;
        auto value = capture_aperture(driver, witness, witness.proxy);
        if (calibrated)
            compare_calibrated_border(reference, value, tight.x2 - tight.x1, tight.y2 - tight.y1, {});
        else
            compare_with_full(witness, value, reference);
        reduced = std::move(value);
    });
    if (outward && reduced) {
        aperture_subcase(name + "/supported-words", passed, [&] {
            const auto left = static_cast<std::size_t>(tight.x1 - wide.x1);
            const auto top = static_cast<std::size_t>(tight.y1 - wide.y1);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                for (std::size_t y = 0; y < tightHeight; ++y) {
                    const auto narrowRow = (channel * tightHeight + y) * tightWidth;
                    const auto wideRow = (channel * wideHeight + top + y) * wideWidth + left;
                    require(std::ranges::equal(std::span(reduced->clean).subspan(narrowRow, tightWidth),
                                               std::span(outward->clean).subspan(wideRow, tightWidth),
                                               same_density_word) &&
                                std::ranges::equal(std::span(reduced->film).subspan(narrowRow, tightWidth),
                                                   std::span(outward->film).subspan(wideRow, tightWidth),
                                                   same_density_word),
                            "outward border changed supported density words");
                }
            }
        });
    }
}

void check_adjacent_scales(const CudaDriver& driver, ApertureCase witness, bool& passed) {
    constexpr double scale = 0x1.b4e81b4e81b4fp-7;
    constexpr double lower = 0x1.b4e81b4e81b4ep-7;
    const GrainGeometry tight{{0, 1050.25, 6000, 1125}, {0, 14, 80, 15}, {scale, scale}, {scale, scale}};
    const GrainGeometry adjacent{tight.canonical, tight.bounds, {lower, lower}, {lower, lower}};
    const GrainGeometry resized{tight.canonical, tight.bounds, {scale, scale}};
    const GrainGeometry resizedAdjacent{tight.canonical, tight.bounds, {scale, lower}};
    const std::string route = witness.print ? "print" : "direct";
    witness.proxy = tight;
    witness.reference = {{0, 1050, 6000, 1125}, {0, 1050, 6000, 1125}};
    aperture_subcase("adjacent-standard-" + route, passed, [&] {
        (void)capture_aperture(driver, witness, tight, &adjacent);
    });
    aperture_subcase("adjacent-resized-" + route, passed, [&] {
        (void)capture_aperture(driver, witness, resized, &resizedAdjacent);
    });
    aperture_subcase("adjacent-different-reciprocal-" + route, passed, [&] {
        const GrainGeometry threeRows{{0, -4, 320, 7}, {0, -1, 80, 2}, {0.25, 0.25}};
        auto alternate = threeRows;
        alternate.scale[1] = std::nextafter(0.25, 0.0);
        require(!same_proxy_grid(realized_grid(threeRows), realized_grid(alternate)),
                "reciprocals must differ even though the y edges are [0, 4, 8, 11]");
        const auto actual = capture_aperture(driver, witness, threeRows, &alternate);
        const auto reference = capture_aperture(driver, witness, full_geometry(80, 3));
        compare_calibrated_border(reference, actual, 80, 3, {});
    });
    aperture_subcase("adjacent-empty-row-calibrated-" + route, passed, [&] {
        auto outward = tight;
        ++outward.bounds.y2;
        const auto reference = capture_aperture(driver, witness, full_geometry(80, 1));
        const auto original = capture_aperture(driver, witness, tight, &adjacent);
        compare_calibrated_border(reference, original, 80, 1, {});
        compare_calibrated_border(reference, capture_aperture(driver, witness, outward), 80, 2, {0, 0, 0, 1});
    });
    aperture_subcase("adjacent-empty-row-full-" + route, passed, [&] {
        auto outward = witness;
        outward.size = 2.0; // Same square, now 50 texels wide: above the cutoff.
        ++outward.proxy.bounds.y2;
        const auto full = capture_aperture(driver, outward, outward.reference);
        const auto wide = capture_aperture(driver, outward, outward.proxy);
        const auto bound = JuicerCuda::FilmGrain::proxy_sample_bound(wide.grid.extent, wide.grid.step);
        require(bound == 5775 && bound > JuicerCuda::FilmGrain::kProxySerialSamples,
                "adjacent-scale outward row did not use the real batched grid");
        compare_with_full(outward, wide, full, 80);
        std::cout << "adjacent-outward-row sample_bound=" << bound << " empty_pixels=80\n";
    });
}

void check_calibrated_border_equivalence(const CudaDriver& driver, ApertureCase witness, bool& passed) {
    const GrainGeometry ordinary{{0, 0, 150, 95}, {0, 0, 60, 38}, {0.4, 0.4}};
    const GrainGeometry tight{{0, -130, 150, -35}, {0, -52, 60, -14}, {0.4, 0.4}, {0.4, 0.4}};
    const auto lower = std::nextafter(0.4, 0.0);
    const GrainGeometry outward{tight.canonical, {0, -52, 60, -13}, {lower, lower}, {lower, lower}};
    const GrainGeometry leading{{5, 5, 155, 100}, {1, 1, 62, 40}, {lower, lower}, {lower, lower}};
    const auto normal = capture_aperture(driver, witness, ordinary);
    aperture_subcase("offset-calibrated-tight", passed, [&] {
        const auto value = capture_aperture(driver, witness, ordinary, &tight);
        const auto noGrid = capture_aperture(driver, witness, full_geometry(60, 38));
        require(same_density_word(value.samplePitch, noGrid.samplePitch) &&
                    std::ranges::equal(value.film, noGrid.film, same_density_word) && value.film != value.clean,
                "offset equivalence did not retain ordinary calibrated words");
    });
    aperture_subcase("offset-calibrated-empty-row", passed, [&] {
        require(same_proxy_grid(realized_grid(ordinary), realized_grid(outward)),
                "the supported outward rows must have the same clamped edges as the ordinary control");
        compare_calibrated_border(normal, capture_aperture(driver, witness, outward), 60, 39, {0, 0, 0, 1});
    });
    aperture_subcase("offset-calibrated-leading-row-and-column", passed, [&] {
        const auto grid = realized_grid(leading);
        require(grid.step[0] == 2.5 && grid.step[1] == 2.5 && grid.origin[0] == -2.5 &&
                    grid.origin[1] == -2.5 && grid.extent[0] == 150.0 && grid.extent[1] == 95.0,
                "leading border must leave the same clamped positive apertures");
        compare_calibrated_border(normal, capture_aperture(driver, witness, leading), 61, 39, {1, 1, 0, 0});
    });
}

void check_supported_extent_alias(const CudaDriver& driver, const ApertureCase& witness) {
    constexpr int width = 1208;
    constexpr int height = 2;
    const GrainGeometry at{{0, 0, 4832, 7}, {0, 0, width, height}, {0.25, 2.0 / 7.0}};
    auto alias = at;
    alias.canonical.y2 = std::nextafter(at.canonical.y2, 8.0);
    const auto original = realized_grid(at);
    const auto alternate = realized_grid(alias);
    require(original.extent[0] == 4832.0 && alternate.extent[0] == original.extent[0] &&
                original.extent[1] == 7.0 && alternate.extent[1] == 8.0 &&
                at.canonical.y2 * at.scale[1] == height && alias.canonical.y2 * alias.scale[1] == height,
            "paused extent alias must keep the admitted raster and change only the unused short extent");

    constexpr std::array rows{0.0, 3.5, 7.0};
    constexpr std::array count{width, height};
    for (const auto& grid : {original, alternate}) {
        for (std::size_t axis = 0; axis < count.size(); ++axis) {
            for (int i = 0; i <= count[axis]; ++i) {
                const double expected = axis == 0 ? i * 4.0 : rows[i];
                const double edge = std::clamp(std::fma(i, grid.step[axis], grid.origin[axis]), 0.0, grid.extent[axis]);
                require(std::bit_cast<std::uint64_t>(edge) == std::bit_cast<std::uint64_t>(expected),
                        "paused extent alias changed a positive Float64 edge word");
            }
        }
    }

    const auto calibrated = capture_aperture(driver, witness, full_geometry(width, height));
    ApertureSwitch sequence{alias};
    const auto initial = capture_aperture(driver, witness, at, nullptr, &sequence);
    require(initial.placement == sequence.switched.placement && initial.placement == sequence.returned.placement,
            "paused extent alias changed input placement");
    std::cout << witness.id << " callbacks=complete extents=7,8,7 positive_edges=exact widest_physical_pitch="
              << std::setprecision(17) << 4.0 * (witness.millimeters * 1000.0 / original.extent[0] / witness.size)
              << std::setprecision(6) << " paused_pitches=" << initial.samplePitch << ',' << sequence.switched.samplePitch
              << ',' << sequence.returned.samplePitch << '\n';
    compare_calibrated_border(calibrated, initial, width, height, {});
    compare_calibrated_border(calibrated, sequence.returned, width, height, {});
    compare_calibrated_border(calibrated, sequence.switched, width, height, {});
}

struct CutoffEdges {
    std::span<const double> at;
    std::span<const double> above;
};

void check_cutoff_switch(const CudaDriver& driver, ApertureCase witness, const GrainGeometry& at, CutoffEdges expected) {
    auto above = at;
    above.scale[1] = std::nextafter(at.scale[1], 0.0);
    const int width = at.bounds.x2 - at.bounds.x1;
    const int height = at.bounds.y2 - at.bounds.y1;
    const auto measured_pitch = [&](const GrainGeometry& geometry, std::span<const double> rows) {
        const auto grid = realized_grid(geometry);
        double widest = 0.0;
        const std::array count{width, height};
        require(rows.size() == static_cast<std::size_t>(height) + 1, "wrong number of expected cutoff edges");
        for (std::size_t axis = 0; axis < count.size(); ++axis) {
            double previous = 0.0;
            for (int i = 0; i <= count[axis]; ++i) {
                const double edge = std::clamp(std::fma(i, grid.step[axis], grid.origin[axis]), 0.0, grid.extent[axis]);
                if (axis == 1)
                    require(std::bit_cast<std::uint64_t>(edge) == std::bit_cast<std::uint64_t>(rows[i]),
                            "cutoff witness realized a different Float64 edge word");
                if (i != 0)
                    widest = std::max(widest, edge - previous);
                previous = edge;
            }
        }
        return widest * (witness.millimeters * 1000.0 / grid.extent[0] / witness.size);
    };
    const double atPitch = measured_pitch(at, expected.at);
    const double abovePitch = measured_pitch(above, expected.above);
    require(atPitch == 37.5 && abovePitch > atPitch && static_cast<float>(abovePitch) == static_cast<float>(atPitch),
            "cutoff evidence must distinguish at and above before Float32 narrowing");
    const auto calibrated = capture_aperture(driver, witness, full_geometry(width, height));
    require(!same_apertures(realized_grid(at), realized_grid(above), {width, height}),
            "paused cutoff witness must change positive edges");
    ApertureSwitch sequence{above};
    const auto initial = capture_aperture(driver, witness, at, nullptr, &sequence);
    compare_calibrated_border(calibrated, initial, width, height, {});
    compare_calibrated_border(calibrated, sequence.returned, width, height, {});
    const auto grid = realized_grid(above);
    witness.proxy = above;
    witness.reference = full_geometry(static_cast<int>(grid.extent[0]), static_cast<int>(grid.extent[1]));
    const auto full = capture_aperture(driver, witness, witness.reference);
    std::cout << witness.id << " cutoff_pitch_at=" << std::hexfloat << atPitch
              << " cutoff_pitch_above=" << abovePitch << std::defaultfloat
              << " paused_pitches=" << initial.samplePitch << ',' << sequence.switched.samplePitch << ','
              << sequence.returned.samplePitch << " expected_full_pitch=" << full.samplePitch << '\n';
    require(same_proxy_grid(sequence.switched.grid, grid) && sequence.switched.samplePitch == full.samplePitch &&
                sequence.switched.placement == full.placement && sequence.switched.calibratedBorder == std::array<std::uint8_t, 4>{},
            "above-cutoff render did not bind the expected Full table, placement and grid");
    require(std::ranges::equal(initial.clean, sequence.switched.clean, same_density_word),
            "cutoff switch changed the fixed developed input density");
    compare_with_full(witness, sequence.switched, full);
}

void check_aperture(const CudaDriver& driver, const ApertureCase& witness) {
    const auto id = std::string_view(witness.id);
    if (id == "standard-27-over-139-border") {
        check_border(driver, witness, 64);
        auto batched = witness;
        batched.proxy = standard_border(273, 3, 2, kBatchedBorderScale);
        batched.reference = full_geometry(273, 96);
        check_border(driver, batched, 8649);
        auto narrow = witness;
        narrow.proxy = standard_border(153, 61, 39, 61.0 / 153.0);
        narrow.reference = full_geometry(153, 96);
        narrow.millimeters = 8.0;
        narrow.size = 4.0;
        bool passed = true;
        for (const auto& border : kEmptyBorderCases)
            check_offset_border(driver, witness, border, passed);
        aperture_subcase("paused-61-over-153", passed, [&] {
            auto resized = narrow.proxy;
            resized.imageScale = {1.0, 1.0};
            const auto reference = capture_aperture(driver, narrow, full_geometry(61, 39));
            const auto switched = capture_aperture(driver, narrow, narrow.proxy, &resized);
            compare_calibrated_border(reference, switched, 61, 39, {});
            auto outward = narrow.proxy;
            ++outward.bounds.x2;
            compare_calibrated_border(reference, capture_aperture(driver, narrow, outward), 62, 39, {0, 0, 1, 0});
        });
        aperture_subcase("paused-61-over-153-rod", passed, [&] {
            auto fractional = narrow.proxy;
            fractional.canonical.x2 = 152.99;
            (void)capture_aperture(driver, narrow, narrow.proxy, &fractional);
        });
        aperture_subcase("resized-without-empty-border", passed, [&] {
            const auto resized = capture_aperture(driver, narrow, {{0, 0, 150, 95}, {0, 0, 60, 38}, {0.4, 0.4}});
            const auto calibrated = capture_aperture(driver, narrow, full_geometry(60, 38));
            require(same_density_word(resized.samplePitch, calibrated.samplePitch) &&
                        std::ranges::equal(resized.film, calibrated.film, same_density_word) && resized.film != resized.clean,
                    "ordinary pre-resized square stopped calibrating");
        });
        check_adjacent_scales(driver, narrow, passed);
        auto print = narrow;
        print.print = true;
        check_adjacent_scales(driver, print, passed);
        check_calibrated_border_equivalence(driver, narrow, passed);
        auto cutoff = witness;
        cutoff.size = 1.0;
        for (const bool printed : {false, true}) {
            cutoff.print = printed;
            cutoff.id = printed ? "supported-extent-alias-print" : "supported-extent-alias-direct";
            aperture_subcase(cutoff.id, passed, [&] {
                check_supported_extent_alias(driver, cutoff);
            });
            cutoff.id = printed ? "realized-cutoff-rounded-print" : "realized-cutoff-rounded-direct";
            aperture_subcase(cutoff.id, passed, [&] {
                const GrainGeometry at{{0, -13, 12480, 65}, {0, -1, 961, 5}, {961.0 / 12480.0, 6.0 / 78.0}};
                const std::array atRows{0.0, 13.0, 26.0, 39.0, 52.0, 65.0, 78.0};
                auto aboveRows = atRows;
                aboveRows[4] = std::nextafter(52.0, 53.0);
                check_cutoff_switch(driver, cutoff, at, {.at = atRows, .above = aboveRows});
            });
            cutoff.id = printed ? "realized-cutoff-bins-print" : "realized-cutoff-bins-direct";
            aperture_subcase(cutoff.id, passed, [&] {
                const GrainGeometry at{{0, -8, 3840, 7}, {0, -2, 960, 2}, {0.25, 0.25}};
                const std::array atRows{0.0, 4.0, 8.0, 12.0, 15.0};
                auto aboveRows = atRows;
                aboveRows[1] = 4.0 - 0x1p-50;
                check_cutoff_switch(driver, cutoff, at, {.at = atRows, .above = aboveRows});
            });
        }
        constexpr double scale = 962.0 / 3839.5;
        for (const bool leading : {false, true}) {
            auto fractional = witness;
            fractional.reference = full_geometry(3840, 96);
            fractional.proxy = {{0, 0, 3839.5, 96}, {0, 0, 963, 25}, {scale, scale}, {scale, scale}};
            if (leading) {
                fractional.proxy.canonical.x1 = -3839.5;
                fractional.proxy.canonical.x2 = 0;
                fractional.proxy.bounds.x1 = -963;
                fractional.proxy.bounds.x2 = 0;
                fractional.reference = {{-3840, 0, 0, 96}, {-3840, 0, 0, 96}, {1, 1}};
            }
            const auto outward = capture_aperture(driver, fractional, fractional.proxy);
            const auto full = capture_aperture(driver, fractional, fractional.reference);
            compare_with_full(fractional, outward, full);
            std::cout << "aperture_subcase=" << (leading ? "positive-leading-control" : "positive-trailing-control") << " passed\n";
        }
        require(passed, "one or more clamped-border subcases failed");
        return;
    }
    const auto proxy = capture_aperture(driver, witness, witness.proxy);
    if (id.starts_with("clipped-quarter")) {
        const auto square = capture_aperture(driver, witness, witness.reference);
        std::cout << id << " clipped_pitch=" << proxy.samplePitch << " square_pitch=" << square.samplePitch << '\n';
        require(std::ranges::equal(proxy.clean, square.clean, same_density_word), "clipped and square Quarter clean C/M/Y words differ");
        require(std::ranges::equal(proxy.film, square.film, same_density_word) && proxy.film != proxy.clean,
                "clipped and square Quarter grained C/M/Y words differ");
        require(std::bit_cast<std::uint32_t>(proxy.samplePitch) == std::bit_cast<std::uint32_t>(square.samplePitch),
                "clipped Quarter selected a different sample pitch");
        return;
    }
    if (id == "calibrated-2-by-4" || id == "calibrated-4-by-2") {
        const bool half = id == "calibrated-2-by-4";
        const auto square = capture_aperture(driver, witness, {witness.proxy.canonical, {0, 0, half ? 2416 : 1208, half ? 48 : 24}, {half ? 0.5 : 0.25, half ? 0.5 : 0.25}});
        const auto width = static_cast<std::size_t>(witness.proxy.bounds.x2);
        const auto height = static_cast<std::size_t>(witness.proxy.bounds.y2);
        const auto squareHeight = static_cast<std::size_t>(half ? 48 : 24);
        bool identical = true;
        for (std::size_t channel = 0; channel < 3; ++channel)
            for (std::size_t y = 0; y < std::min(height, squareHeight); ++y)
                for (std::size_t x = 0; x < width; ++x)
                    identical &= same_density_word(proxy.film[(channel * height + y) * width + x],
                                                   square.film[(channel * squareHeight + y) * width + x]);
        require(!identical && square.grid.step[0] == 0.0 && square.grid.step[1] == 0.0,
                "material rectangle must not match calibrated square words");
        if (!half) {
            const auto nominal = capture_aperture(driver, witness, {{0, 0, 4832, 2016}, {0, 0, 1208, 504}, {0.25, 0.25}});
            const auto rounded = capture_aperture(driver, witness, {{0, 0, 4831, 2015}, {0, 0, 1208, 504}, {1208.0 / 4831.0, 504.0 / 2015.0}});
            require(same_density_word(nominal.samplePitch, rounded.samplePitch) &&
                        std::ranges::equal(nominal.clean, rounded.clean, same_density_word) &&
                        std::ranges::equal(nominal.film, rounded.film, same_density_word),
                    "dimension-rounded Quarter changed calibrated words");
            auto shallow = witness;
            shallow.reference = full_geometry(4832, 2);
            shallow.proxy = {{0, 0, 4832, 2}, {0, 0, 1208, 1}, {0.25, 0.25}};
            const auto clippedSquare = capture_aperture(driver, shallow, shallow.proxy);
            shallow.proxy.scale[1] = 0.5;
            const auto clippedRectangle = capture_aperture(driver, shallow, shallow.proxy);
            const auto shallowFull = capture_aperture(driver, shallow, shallow.reference);
            std::cout << "shallow-4-by-2 square_scale_pitch=" << clippedSquare.samplePitch
                      << " unequal_scale_pitch=" << clippedRectangle.samplePitch << '\n';
            compare_with_full(shallow, clippedRectangle, shallowFull);
            shallow.proxy.scale[1] = 0.25;
            compare_with_full(shallow, clippedSquare, shallowFull);
            require(std::ranges::equal(clippedSquare.film, clippedRectangle.film, same_density_word),
                    "equivalent shallow apertures changed Film words");
        }
    }
    if (id == "shifted-37-5-equivalent") {
        const auto clipped = capture_aperture(driver, witness, kShiftedClipped);
        double shiftedResidual = 0.0;
        for (std::size_t i = 0; i < proxy.film.size(); ++i) {
            const double residual = (static_cast<double>(proxy.film[i]) - proxy.clean[i]) -
                                    (static_cast<double>(clipped.film[i]) - clipped.clean[i]);
            shiftedResidual = std::max(shiftedResidual, std::abs(residual));
        }
        std::cout << id << " square_pitch=" << proxy.samplePitch << " clipped_pitch=" << clipped.samplePitch
                  << " max_pair_residual_D=" << shiftedResidual << '\n';
        const auto full = capture_aperture(driver, witness, witness.reference);
        compare_with_full(witness, proxy, full);
        auto alternative = witness;
        alternative.proxy = kShiftedClipped;
        compare_with_full(alternative, clipped, full);
        require(shiftedResidual <= 2e-6 && std::ranges::equal(proxy.film, clipped.film, same_density_word),
                "equivalent shifted apertures disagree");
        const auto square = capture_aperture(driver, witness, {{4, 14, 4836, 110}, {1, 4, 1451, 33}, {0.3, 0.3}, {0.3, 0.3}});
        const auto calibrated = capture_aperture(driver, witness, full_geometry(1450, 29));
        std::cout << "shifted-0.3 square_pitch=" << square.samplePitch << " calibrated_pitch=" << calibrated.samplePitch << '\n';
        require(square.grid.step[0] == 0.0 && square.grid.step[1] == 0.0 &&
                    same_density_word(square.samplePitch, calibrated.samplePitch) &&
                    std::ranges::equal(square.clean, calibrated.clean, same_density_word) &&
                    std::ranges::equal(square.film, calibrated.film, same_density_word) && square.film != square.clean,
                "shifted 0.3 square changed calibrated density words");
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
    if (id == "offset-wide-clipped")
        require(proxy.grid.origin[1] == -4.0 && proxy.grid.step[1] == 8.0 && proxy.grid.extent[1] == 17.0,
                "offset aperture did not publish the expected Full-local grid");
    compare_with_full(witness, proxy, full);
    if (id == "calibrated-2-by-4" || id == "calibrated-4-by-2") {
        const bool transposed = id == "calibrated-2-by-4";
        const auto orient = [transposed](GrainGeometry geometry) {
            return transposed ? transpose_geometry(geometry) : geometry;
        };
        bool passed = true;
        aperture_subcase(transposed ? "two-cell-rounded-y-long" : "two-cell-rounded-x-long", passed, [&] {
            const auto rounded = capture_aperture(driver, witness, orient({{0, 0, 4832, 7}, {0, 0, 1208, 2}, {0.25, 2.0 / 7.0}}));
            const auto calibrated = capture_aperture(driver, witness, orient(full_geometry(1208, 2)));
            std::cout << "two-cell-rounding sample_pitch=" << rounded.samplePitch << " calibrated_pitch=" << calibrated.samplePitch << '\n';
            require(rounded.grid.step[0] == 0.0 && rounded.grid.step[1] == 0.0 &&
                        same_density_word(rounded.samplePitch, calibrated.samplePitch) &&
                        std::ranges::equal(rounded.clean, calibrated.clean, same_density_word) &&
                        std::ranges::equal(rounded.film, calibrated.film, same_density_word) && rounded.film != rounded.clean,
                    "fully supported dimension-rounded two-cell proxy lost calibrated words");
        });
        aperture_subcase(transposed ? "two-cell-clipped-y-long" : "two-cell-clipped-x-long", passed, [&] {
            auto clipped = witness;
            clipped.proxy = orient({{0, 0, 4832, 7}, {0, 0, 1208, 2}, {0.25, 0.2}});
            clipped.reference = orient(full_geometry(4832, 7));
            const auto reduced = capture_aperture(driver, clipped, clipped.proxy);
            const auto clippedFull = capture_aperture(driver, clipped, clipped.reference);
            compare_with_full(clipped, reduced, clippedFull);
        });
        require(passed, "one or more two-cell calibration subcases failed");
    }
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
