#include "juicer_test_api.h"
namespace GrainFrameTest {

    float scalar(const Json& v) {
        return std::bit_cast<float>(v.get<std::uint32_t>());
    }
    double wide(const Json& v) {
        return std::bit_cast<double>(v.get<std::uint64_t>());
    }
    Spektrafilm::VisualGrainFrameExtent extent(const Json& v) {
        return {v.at(0), v.at(1), v.at(2), v.at(3)};
    }
    Json product_rows(const std::filesystem::path& resources, const Json& inputs, const Json::array_t& expectedRows) {
        JuicerTestSupport::RuntimeFixture owner;
        owner.create(resources.string().c_str());
        Json products = Json::array();
        std::size_t expectedIndex = 0;
        for (unsigned route = 0; route < 4; ++route) {
            std::size_t specIndex = 0;
            for (const auto& spec : inputs.at("products")) {
                JuicerTests::CaseEvent caseEvent("cpp", "grain-frame/product/" + std::to_string(route) + "/" + std::to_string(specIndex++));
                JuicerTestSupport::PreparationInput params;
                params.values.scan_route = static_cast<std::uint32_t>(route);
                params.film_key = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                params.values.grain.active = spec.at("active");
                params.values.grain.sublayers_active = spec.at("layers");
                params.values.grain.debug_view = spec.at("debug");
                params.values.grain.chroma_mix = scalar(spec.at("chroma"));
                params.values.grain.amplitude = scalar(spec.value("amplitude", Json(1065353216u)));
                params.values.grain.particle_area_um2 = scalar(spec.value("area", Json(1051428127u)));
                params.values.camera_exposure_compensation_ev = spec.value("ev", 0.0);
                auto product = JuicerTestSupport::prepare_product(owner.borrow().assets, params);
                FjPreparedStateView view{};
                FjErrorBuffer viewError{};
                status(product.view(view, viewError));
                auto facts = JuicerTestSupport::grain_frame_input(inputs.at("cases")[0]);
                const auto raw = JuicerTestSupport::grain_frame_input(view, facts.bind());
                FjGrainFrameMath math{};
                FjGrainFrameFailure failure{};
                std::array<char, 256> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                const auto result = fj_test_grain_frame_prepare(&raw, &math, &failure, &error);
                Spektrafilm::VisualGrainFrameDescriptor out{};
                std::string compositionDiagnostic;
                const bool composed = result.category == FJ_STATUS_SUCCESS && Spektrafilm::compose_visual_grain_frame_descriptor(JuicerTestSupport::grain_coefficients(view.grain), static_cast<Spektrafilm::ProfilePolarity>(view.polarity), math, out, compositionDiagnostic);
                require(composed, "complete prepared-state grain frame composition");
                auto row = Json{{"route", route}, {"spec", spec}, {"success", result.category == FJ_STATUS_SUCCESS}, {"diagnostic", std::string(diagnostic.data(), error.length)}, {"native", JuicerTestSupport::grain_frame_fields(out)}};
                require_equal(row, expectedRows.at(expectedIndex++), "completed grain product row matches captured fixture");
                products.push_back(std::move(row));
                caseEvent.passed();
            }
        }
        require(expectedIndex == expectedRows.size(), "complete grain product fixture membership");
        return products;
    }
    FjGrainFrameMath expected_math(const Json& native) {
        FjGrainFrameMath out{};
        if (!native.at("active").get<bool>())
            return out;
        out.present = 1;
        out.capture_polarity = native.at("capture_polarity").get<std::uint32_t>();
        {
            const auto e = extent(native.at("render_extent"));
            out.render_extent = {e.x, e.y, e.width, e.height};
        }
        {
            const auto e = extent(native.at("full_frame_extent"));
            out.full_frame_extent = {e.x, e.y, e.width, e.height};
        }
        out.pixel_size_um = scalar(native.at("pixel_size_um"));
        out.frame0 = native.at("frame0").get<std::int64_t>();
        out.frame_alpha = scalar(native.at("frame_alpha"));
        out.seed_base = native.at("seed_base").get<std::uint64_t>();
        out.seed_base_next = native.at("seed_base_next").get<std::uint64_t>();
        out.session_seed = native.at("session_seed").get<std::uint64_t>();
        out.clip_token = native.at("clip_token").get<std::uint64_t>();
        out.pitch_px = native.at("pitch_px").get<std::int32_t>();
        out.breathing_period_frames = native.at("breathing_period_frames").get<std::int32_t>();
        out.clump_morph_period_frames = native.at("clump_morph_period_frames").get<std::int32_t>();
        out.wang_cell_mm = scalar(native.at("wang_cell_mm"));
        out.breathing_amplitude = scalar(native.at("breathing_amplitude"));
        out.breathing_cell_um_small = scalar(native.at("breathing_cell_um_small"));
        out.breathing_cell_um_large = scalar(native.at("breathing_cell_um_large"));
        out.breathing_mix = scalar(native.at("breathing_mix"));
        out.breathing_drift_um_per_frame = scalar(native.at("breathing_drift_um_per_frame"));
        out.debug_scale = scalar(native.at("debug_scale"));
        for (unsigned c = 0; c < 3; ++c)
            out.density_max_cmy[c] = scalar(native.at("density_max_cmy")[c]);
        for (unsigned c = 0; c < 3; ++c)
            out.n_particles_cmy[c] = scalar(native.at("n_particles_cmy")[c]);
        for (unsigned c = 0; c < 3; ++c)
            out.od_particle_cmy[c] = scalar(native.at("od_particle_cmy")[c]);
        for (unsigned l = 0; l < 3; ++l)
            for (unsigned c = 0; c < 3; ++c)
                out.density_min_layers[l][c] = scalar(native.at("density_min_layers")[l][c]);
        for (unsigned l = 0; l < 3; ++l)
            for (unsigned c = 0; c < 3; ++c)
                out.density_max_layers[l][c] = scalar(native.at("density_max_layers")[l][c]);
        for (unsigned l = 0; l < 3; ++l)
            for (unsigned c = 0; c < 3; ++c)
                out.n_particles_layers[l][c] = scalar(native.at("n_particles_layers")[l][c]);
        for (unsigned l = 0; l < 3; ++l)
            for (unsigned c = 0; c < 3; ++c)
                out.od_particle_layers[l][c] = scalar(native.at("od_particle_layers")[l][c]);
        for (unsigned c = 0; c < 3; ++c)
            out.size_mix_weights[c] = scalar(native.at("size_mix_weights")[c]);
        out.size_mix_gain = scalar(native.at("size_mix_gain"));
        out.requires_full_frame = native.at("requires_full_frame").get<bool>() ? 1u : 0u;
        out.recipe_hash = native.at("recipe_hash").get<std::uint64_t>();
        out.density_layers_hash = native.at("density_layers_hash").get<std::uint64_t>();
        const auto gaussian = [](const Json& g) -> FjGrainGaussian {
            return {scalar(g.at(0)), g.at(1), g.at(2)};
        };
        for (unsigned l = 0; l < 3; ++l) {
            out.correlation[l] = gaussian(native.at("correlation")[l]);
            for (unsigned c = 0; c < 3; ++c)
                out.dye_cloud[l][c] = gaussian(native.at("dye_cloud")[l][c]);
        }
        return out;
    }

    void values(const Json& captured) {
        require(captured.at("cases").size() == 153, "complete grain frame literal cohort");
        const auto defaults = JuicerTestSupport::grain_frame_fields({});
        std::size_t representableRows = 0;
        std::size_t retiredPointerRows = 0;
        for (const auto& row : captured.at("cases")) {
            const auto& v = row.at("input");
            const bool success = row.at("success");
            std::string diagnostic;
            const auto name = v.at("name").get<std::string>();
            if (name == "missing-recipe" || name == "missing-film" || name == "missing-inactive") {
                ++retiredPointerRows;
                continue;
            }

            ++representableRows;
            const auto input = JuicerTestSupport::grain_frame_input(v);
            const FjGrainFrameInput raw = input.bind();
            const auto expected = success ? expected_math(row.at("native")) : FjGrainFrameMath{};
            const FjGrainFrameFailure failure{success ? 0u : 2u, v.at("expected_field").get<std::uint32_t>(), v.at("expected_layer").get<std::uint32_t>(), v.at("expected_channel").get<std::uint32_t>()};
            JuicerTests::CaseEvent c11Event("c11", name);
            for (unsigned facade = 0; facade < 2; ++facade)
                require(fj_test_grain_frame_c_check(facade, &raw, &expected, {success ? 0u : 3u, 0, 0}, failure, row.at("diagnostic").get<std::string>().c_str()) != 0,
                        ("strict C11 grain complete values/failures: " + v.at("name").get<std::string>()).c_str());
            c11Event.passed();

            JuicerTests::CaseEvent cppEvent("cpp", name);
            Spektrafilm::VisualGrainFrameDescriptor out;
            const bool actual = JuicerTestSupport::prepare_grain_frame(input, out, diagnostic);
            require(actual == success && diagnostic == row.at("diagnostic").get<std::string>(),
                    ("raw grain descriptor outcome: " + v.at("name").get<std::string>()).c_str());
            require(JuicerTestSupport::grain_frame_fields(out) == (success ? row.at("native") : defaults),
                    ("raw grain math/composition/hash/default: " + v.at("name").get<std::string>()).c_str());
            cppEvent.passed();
        }
        require(representableRows == 150 && retiredPointerRows == 3,
                "150 representable raw cases and three retired native pointer-adapter observations");
    }

    void raw_and_faults(const Json& captured) {
        JuicerTests::CaseEvent caseEvent("cpp", "grain-frame/raw-and-faults");
        const auto input = JuicerTestSupport::grain_frame_input(captured.at("cases")[0].at("input"));
        FjGrainFrameInput raw = input.bind();
        std::string diagnostic;
        FjGrainFrameInput layered = raw;
        layered.sublayers_active = 1;
        layered.layer_curves_required = 1;
        const auto cleared = expected_math(Json{{"active", false}});
        const auto check = [&](const FjGrainFrameInput* source, unsigned category, FjGrainFrameFailure expected, const char* text) {
            for (unsigned facade = 0; facade < 2; ++facade)
                require(fj_test_grain_frame_c_check(facade, source, &cleared, {category, 0, 0}, expected, text) != 0, "C11 raw/fault cleared result and exact metadata");
        };
        constexpr const char* kRaw = "visual grain frame arguments are invalid";
        check(nullptr, 2, {1, 0, 3, 3}, kRaw);
        auto bad = raw;
        bad.active = 2;
        check(&bad, 2, {1, 0, 3, 3}, kRaw);
        bad = raw;
        bad.capture_polarity = 3;
        check(&bad, 2, {1, 0, 3, 3}, kRaw);
        bad = raw;
        bad.film_polarity = 3;
        check(&bad, 2, {1, 0, 3, 3}, kRaw);
        bad = raw;
        bad.sublayers_active = 2;
        check(&bad, 2, {1, 0, 3, 3}, kRaw);
        bad = raw;
        bad.sublayers_active = 1;
        bad.layer_curves_required = 2;
        check(&bad, 2, {1, 0, 3, 3}, kRaw);
        bad = raw;
        bad.frame_rate = std::numeric_limits<double>::max();
        check(&bad, 3, {2, 4, 3, 3}, "UnsupportedMode phase=grain_descriptor field=period_frames");
        bad = raw;
        bad.clump_morph_period_sec = std::numeric_limits<float>::quiet_NaN();
        check(&bad, 3, {2, 4, 3, 3}, "UnsupportedMode phase=grain_descriptor field=period_frames");
        float canary = 1.0f;
        bad = raw;
        bad.sublayers_active = 1;
        bad.layer_curves[0][0] = {&canary, SIZE_MAX};
        check(&bad, 2, {1, 9, 0, 0}, kRaw);
        bad.layer_curves[0][0] = {nullptr, 0};
        bad.layer_curves[2][2] = {&canary, SIZE_MAX};
        check(&bad, 3, {2, 9, 0, 0}, "MalformedRequiredProfileData phase=grain_descriptor field=density_layers");
        bad = raw;
        bad.frame_rate = 0;
        bad.sublayers_active = 1;
        bad.layer_curves[0][0] = {&canary, SIZE_MAX};
        check(&bad, 3, {2, 1, 3, 3}, "ResourceDescriptorMismatch phase=grain_descriptor field=frame_input");
        const auto invoke = [&](const FjGrainFrameInput* source, FjGrainFrameMath* out, FjGrainFrameFailure* failure, FjErrorBuffer* error) {
            return fj_test_grain_frame_prepare(source, out, failure, error);
        };
        FjGrainFrameMath out{};
        FjGrainFrameFailure failure{};
        const std::vector<std::string> capacity_ids{"0", "1", "7", std::to_string(std::strlen(kRaw)), std::to_string(std::strlen(kRaw) + 1)};
        JuicerTests::Cohort diagnostic_capacities("grain_frame.diagnostic_capacity", capacity_ids);
        for (const std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::strlen(kRaw), std::strlen(kRaw) + 1}) {
            auto row_event = diagnostic_capacities.case_event(std::to_string(capacity));
            std::array<char, 128> text{};
            text.fill('Z');
            FjErrorBuffer error{capacity ? text.data() : nullptr, capacity, 99};
            status(invoke(nullptr, &out, &failure, &error), 2);
            const auto count = capacity ? std::min(std::strlen(kRaw), capacity - 1) : 0;
            require(error.length == count && (capacity ? text[count] == '\0' && std::memcmp(text.data(), kRaw, count) == 0 && text[capacity] == 'Z' : text[0] == 'Z'), "written-byte diagnostic capacities");
            row_event.passed();
        }
        status(invoke(nullptr, &out, &failure, nullptr), 2);
        require(failure.reason == 1 && failure.field == 0 && failure.layer == 3 && failure.channel == 3, "optional diagnostic metadata");
        FjErrorBuffer malformed{nullptr, 1, 99};
        status(invoke(&raw, &out, &failure, &malformed), 2);
        require(malformed.length == 0, "malformed diagnostic clears written count");
        status(invoke(&raw, nullptr, &failure, nullptr), 2);
        status(invoke(&raw, &out, nullptr, nullptr), 2);
        alignas(8) std::array<unsigned char, sizeof(FjGrainFrameInput) + 8> misaligned{};
        status(invoke(reinterpret_cast<const FjGrainFrameInput*>(misaligned.data() + 1), &out, &failure, nullptr), 2);
        JuicerTests::Cohort raw_faults("grain_frame.raw_fault", {"entry-panic", "layer-panic"});
        for (unsigned fault : {1u, 2u}) {
            auto row_event = raw_faults.case_event(fault == 1 ? "entry-panic" : "layer-panic");
            status(fj_test_grain_frame_fault(fault));
            require(fj_test_grain_frame_c_check(1, fault == 2 ? &layered : &raw, &cleared, {8, 0, 0}, {3, 0, 3, 3}, "visual grain frame preparation panicked") != 0, "contained entry/layer panic");
            row_event.passed();
        }
        status(fj_test_grain_frame_fault(1));
        status(fj_test_grain_frame_fault(99), 2);
        require(fj_test_grain_frame_c_check(0, &raw, &cleared, {8, 0, 0}, {3, 0, 3, 3}, "visual grain frame preparation panicked") != 0, "unsupported fault preserves pending one-shot");
        status(fj_test_grain_frame_fault(1));
        status(invoke(nullptr, &out, &failure, nullptr), 2);
        status(invoke(&raw, &out, &failure, nullptr));
        status(fj_test_grain_frame_fault(2));
        status(invoke(&raw, &out, &failure, nullptr));
        status(invoke(&layered, &out, &failure, nullptr));
        JuicerTests::Cohort projection_faults("grain_frame.native_projection_fault", 4);
        for (unsigned fault = 3; fault <= 6; ++fault) {
            auto row_event = projection_faults.case_event(std::to_string(fault - 3));
            status(fj_test_grain_frame_fault(fault));
            Spektrafilm::VisualGrainFrameDescriptor descriptor;
            require(!JuicerTestSupport::prepare_grain_frame(input, descriptor, diagnostic) &&
                        diagnostic == "ResourceDescriptorMismatch phase=grain_descriptor field=rust_result" &&
                        JuicerTestSupport::grain_frame_fields(descriptor) == JuicerTestSupport::grain_frame_fields({}),
                    "corrupt raw result fails composition and clears descriptor");
            require(JuicerTestSupport::prepare_grain_frame(input, descriptor, diagnostic), "fault consumed once and recovery");
            row_event.passed();
        }
        // A non-sublayer result ignores even numerically invalid layer spans.
        bad = raw;
        bad.layer_curves[0][0] = {&canary, SIZE_MAX};
        status(invoke(&bad, &out, &failure, nullptr));
        caseEvent.passed();
    }
} // namespace GrainFrameTest
