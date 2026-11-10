#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ColorTransforms.h"
#include "RenderValues.h"
#include "owner_fixture.h"
#include "prepared_product_fixture.h"
#include "juicer_test_api.h"
#include "case_results.h"

namespace {
    using Json = nlohmann::json;
    constexpr std::uint32_t kBlockedPendingKind = 6;
    void require(bool condition, const std::string& why) {
        if (!condition)
            throw std::runtime_error(why);
    }
    template <typename Range>
    Json bits(const Range& values) {
        Json out = Json::array();
        for (float v : values)
            out.push_back(std::bit_cast<std::uint32_t>(v));
        return out;
    }
    Json span_bits(FjFloatSpan span, std::size_t count) {
        require(span.count == count && (count == 0 || span.data != nullptr), "prepared span extent");
        Json out = Json::array();
        for (std::size_t i = 0; i < span.count; ++i)
            out.push_back(std::bit_cast<std::uint32_t>(span.data[i]));
        return out;
    }
    Json scanner_color(const FjScannerColor& color) {
        return {{"cat02", bits(std::span(color.cat02))}, {"xyz_to_rgb", bits(std::span(color.xyz_to_rgb))}, {"illuminant_xyz", bits(std::span(color.illuminant_xyz))}, {"encoding", {static_cast<int>(color.output_space), static_cast<bool>(color.flags & 1u), static_cast<bool>(color.flags & 2u)}}, {"gamut_recipe_hash", color.active_gamut_recipe_hash}, {"hash", color.hash}};
    }
    Json scanner_tables(const FjPreparedSpectralView& tables) {
        return {{"lambda", span_bits(tables.lambda_nm, 81)}, {"K", 81}, {"delta_lambda", bits(std::array{5.0f})}, {"inv_yn", bits(std::array{tables.inv_yn})}, {"white_xyz", bits(std::span(tables.white_xyz))}, {"ref_illum_white_xyz", bits(std::span(tables.white_xyz))}, {"ax", span_bits(tables.weighted_xyz[0], 81)}, {"ay", span_bits(tables.weighted_xyz[1], 81)}, {"az", span_bits(tables.weighted_xyz[2], 81)}, {"xbar", span_bits(tables.observer_xyz[0], 81)}, {"ybar", span_bits(tables.observer_xyz[1], 81)}, {"zbar", span_bits(tables.observer_xyz[2], 81)}, {"illum", span_bits(tables.illuminant, 81)}, {"eps_c", span_bits(tables.dyes_cmy[0], 81)}, {"eps_m", span_bits(tables.dyes_cmy[1], 81)}, {"eps_y", span_bits(tables.dyes_cmy[2], 81)}, {"base_density_min", span_bits(tables.baseline_min, 81)}, {"base_density_mid", span_bits(tables.baseline_mid, 81)}, {"has_baseline", tables.has_baseline != 0}, {"baseline_mix", bits(std::array{0.0f})}, {"illuminant_hash", tables.illuminant_hash}, {"tables_hash", tables.hash}};
    }
    Json scanner_descriptor(const FjScannerLutDescriptor& descriptor) {
        return {{"route", descriptor.route}, {"medium", descriptor.medium}, {"polarity", descriptor.capture_polarity}, {"density_bounds_hash", descriptor.density_bounds_hash}, {"channel_density_hash", descriptor.channel_density_hash}, {"base_density_hash", descriptor.base_density_hash}, {"illuminant_hash", descriptor.viewing_illuminant_hash}, {"observer_hash", descriptor.observer_hash}, {"resolution", descriptor.resolution}, {"hash", descriptor.hash}};
    }
    Json complete_result(const FjPreparedStateView& state, std::uint32_t method) {
        const auto& film = state.film_exposure;
        const auto& policy = state.input_policy;
        const auto& reference = state.film_reference;
        const auto& balance = state.print_balance;
        const auto& color = state.output_color;
        require(film.sensitivity_rgb.count == 243 && film.sensitivity_rgb.data, "prepared sensitivity extent");
        return {{"input_rgb_to_xyz", bits(std::span(film.input_rgb_to_xyz))}, {"input_adapt", bits(std::span(film.input_xyz_adapt))}, {"xyz_to_linear_srgb", bits(std::span(film.xyz_to_linear_srgb))}, {"source_white", bits(std::span(policy.nominal_white_xyz))}, {"projection_white", bits(std::span(policy.projection_white_xyz))}, {"config_rgb_to_xyz", bits(std::span(film.input_rgb_to_xyz))}, {"config_adapt", bits(std::span(film.input_xyz_adapt))}, {"raw_midgray", bits(std::span(state.raw_midgray))}, {"midgray_dwg", bits(std::span(state.midgray_dwg_rgb))}, {"midgray_scale_green", bits(std::array{state.midgray_scale, state.raw_green})}, {"config_whites", {bits(std::span(policy.nominal_white_xyz)), bits(std::span(policy.projection_white_xyz)), bits(std::span(policy.projection_white_xyz))}}, {"config_flags", {static_cast<int>(film.input_color_space), static_cast<bool>(film.flags & FJ_FILM_DECODE_CCTF), method == 1 ? 1 : 0, static_cast<bool>(film.flags & FJ_FILM_ADAPT_INPUT), true, true}}, {"spd_s_inv", bits(std::span(state.s_inverse))}, {"baseline_raw", bits(std::span(reference.baseline_raw_rgb))}, {"compensated_raw", bits(std::span(reference.compensated_raw_rgb))}, {"baseline_density", bits(std::span(reference.baseline_density_cmy))}, {"compensated_density", bits(std::span(reference.compensated_density_cmy))}, {"print_baseline_raw", bits(std::span(balance.baseline_raw_rgb))}, {"print_compensated_raw", bits(std::span(balance.compensated_raw_rgb))}, {"print_factors", bits(JuicerTestSupport::captured_print_factors(state))}, {"hashes", {film.hash, film.tc_lut_hash, reference.hash, balance.hash, state.print_recipe_hash, state.scanner_recipe_hash, color.gamut_recipe_hash, state.recipe_hash, state.upload_core_hash, state.scanner_hash}}, {"scanner_color", scanner_color(state.scanner_color)}, {"scanner_tables", scanner_tables(state.scanner_tables)}, {"scanner_descriptor", scanner_descriptor(state.scanner_lut)}, {"gamut_entry", bits(std::span(color.native_rgb_to_d65_xyz))}, {"gamut_exit", bits(std::span(color.d65_xyz_to_native_rgb))}};
    }
    JuicerTestSupport::PreparationInput snapshot(Spektrafilm::ScanRoute route, Spektrafilm::RgbToRawMethod method) {
        JuicerTestSupport::PreparationInput p;
        p.film_key = Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Negative ? "kodak_portra_400" : "fujifilm_provia_100f";
        p.print_key = "kodak_portra_endura";
        p.values.scan_route = static_cast<std::uint32_t>(route);
        p.values.spectral_upsampling_mode = static_cast<int>(method);
        p.values.camera_exposure_compensation_ev = .75;
        p.values.camera_auto_exposure_enabled = 0;
        p.values.input_compression_enabled = 0;
        return p;
    }
    JuicerTestSupport::PreparationInput controls(const Json& row) {
        auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(row.at("polarity").get<int>() * 2 + row.at("route").get<int>()), static_cast<Spektrafilm::RgbToRawMethod>(row.at("method").get<int>()));
        const auto& c = row.at("controls");
        p.values.input_color_space = c.at("inputColorSpace").get<int>();
        p.values.input_cctf_decoding = c.at("inputCctfDecoding").get<int>();
        p.values.output_cctf_encoding = c.at("outputCctfEncoding").get<int>();
        p.values.camera_exposure_compensation_ev = std::bit_cast<double>(c.at("cameraExposureCompensationEv").at("f64_bits").get<std::uint64_t>());
        return p;
    }
    void products(JuicerTestSupport::RuntimeFixture& runtime, const Json& fixture) {
        require(fixture.at("products").size() == 56, "all 56 Mallett product rows retained");
        std::vector<std::string> product_ids;
        for (const auto& row : fixture.at("products"))
            product_ids.emplace_back(std::to_string(row.at("polarity").get<int>()) + "/" + std::to_string(row.at("route").get<int>()) + "/" + std::to_string(row.at("method").get<int>()) + "/" + std::to_string(row.at("variant").get<int>()));
        JuicerTests::Cohort product_rows("mallett.product", product_ids);
        for (std::size_t i = 0; i < fixture.at("products").size(); ++i) {
            const auto& row = fixture.at("products")[i];
            const std::string case_id = std::to_string(row.at("polarity").get<int>()) + "/" + std::to_string(row.at("route").get<int>()) + "/" + std::to_string(row.at("method").get<int>()) + "/" + std::to_string(row.at("variant").get<int>());
            auto event = product_rows.case_event(case_id);
            auto p = controls(row);
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<JuicerTestSupport::PreparedStateHold> owner;
            const auto status = JuicerTestSupport::prepare_state(runtime.borrow().assets, p, owner, error);
            require(status.category == FJ_STATUS_SUCCESS && owner.has_value(), std::string(diagnostic.data(), error.length));
            FjPreparedStateView view{};
            require(owner->view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(view.route == static_cast<std::uint32_t>(p.values.scan_route) && view.film_exposure.method == static_cast<std::uint32_t>(p.values.spectral_upsampling_mode), "prepared route/method identity");
            require(view.raw_midgray_bgr == (p.values.spectral_upsampling_mode == 1 ? 1u : 0u), "prepared raw-midgray channel order");
            const auto observed = complete_result(view, static_cast<std::uint32_t>(p.values.spectral_upsampling_mode));
            require(observed == row.at("expected"), "actual complete product " + std::to_string(i) + ": " + Json::diff(row.at("expected"), observed).dump());
            const auto& meter = fixture.at("metrics")[i];
            require(std::bit_cast<std::uint32_t>(view.film_reference.meter_scale) == meter.at("baseline_meter_scale").get<std::uint32_t>(), "meter exact");
            require(view.print_balance.filtered_main_illuminant_hash == meter.at("filtered_main_illuminant_hash").get<std::uint64_t>(), "filtered print identity exact");
            Json sensitivity = Json::array();
            for (std::size_t channel = 0; channel < 81; ++channel)
                sensitivity.push_back(bits(std::span(view.film_exposure.sensitivity_rgb.data + channel * 3, 3)));
            require(sensitivity == row.at("final_sensitivity") && span_bits(view.exposure_tables.illuminant, 81) == row.at("reference_spd"), "source/sensitivity remain exact");
            event.passed();
        }
    }
    struct FaultScope {
        ~FaultScope() {
            fj_test_mallett_source_clear();
            fj_test_exposure_clear_fault();
            fj_test_spectral_clear_fault();
            fj_test_color_clear_fault();
            fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
        }
    };

    std::uint64_t fnv_bytes(const std::vector<std::uint8_t>& bytes) {
        std::uint64_t hash = 14695981039346656037ull;
        for (std::uint8_t byte : bytes) {
            hash ^= byte;
            hash *= 1099511628211ull;
        }
        return hash;
    }
    std::uint64_t write_mallett_npy(const std::filesystem::path& path, std::array<std::size_t, 2> shape, std::size_t samples) {
        std::ostringstream dimensions;
        dimensions << "(" << shape[0] << ", " << shape[1] << "),";
        std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + dimensions.str() + " }";
        const std::size_t padding = (16 - ((10 + header.size() + 1) % 16)) % 16;
        header.append(padding, ' ');
        header.push_back('\n');
        require(header.size() <= 65535, "Mallett NPY header extent");
        std::vector<std::uint8_t> bytes{0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, static_cast<std::uint8_t>(header.size() & 0xff), static_cast<std::uint8_t>((header.size() >> 8) & 0xff)};
        bytes.insert(bytes.end(), header.begin(), header.end());
        bytes.resize(bytes.size() + samples * sizeof(float), 0);
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        require(file.good(), "open generated Mallett source " + path.string());
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(file.good(), "write generated Mallett source " + path.string());
        return fnv_bytes(bytes);
    }
    struct SourceFailure {
        const char* name;
        std::filesystem::path path;
        std::string diagnostic;
        std::uint64_t hash;
        bool exists;
    };
    FjPathView native_path_view(const std::filesystem::path& path) {
#if defined(_WIN32)
        const auto& native = path.native();
        return {native.data(), native.size(), FJ_PATH_WINDOWS_WIDE};
#else
        const auto& native = path.native();
        return {native.data(), native.size(), FJ_PATH_UNIX_BYTES};
#endif
    }
    void arm_source(const std::filesystem::path& path) {
        std::array<char, 1024> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        const auto status = fj_test_mallett_source_override(native_path_view(path), &error);
        require(status.category == FJ_STATUS_SUCCESS, "arm Mallett source override: " + std::string(diagnostic.data(), error.length));
        require(fj_test_mallett_source_consumed() == 0, "source override starts unconsumed");
    }
    std::array<SourceFailure, 4> source_failures(const std::filesystem::path& root) {
        std::filesystem::create_directories(root);
        const auto missing = root / "missing.npy";
        std::filesystem::remove(missing);
        const auto rows = root / "rows-80x3.npy";
        const auto columns = root / "columns-81x2.npy";
        const auto truncated = root / "payload-240-of-243.npy";
        const auto rowsHash = write_mallett_npy(rows, {80, 3}, 240);
        const auto columnsHash = write_mallett_npy(columns, {81, 2}, 162);
        const auto truncatedHash = write_mallett_npy(truncated, {81, 3}, 240);
        const std::string expectedShape = "InvalidNpy { expected: \"NPY 1.0, little-endian f16/f32/f64, C-order (81, 3) RGB\", reason: \"incompatible shape\" }";
        Json manifest = {
            {"missing", {{"path", missing.string()}, {"expected", "Open(NotFound)"}}},
            {"rows_80x3", {{"path", rows.string()}, {"expected", expectedShape}, {"fnv1a64", rowsHash}}},
            {"columns_81x2", {{"path", columns.string()}, {"expected", expectedShape}, {"fnv1a64", columnsHash}}},
            {"short_payload", {{"path", truncated.string()}, {"expected", "ShortRead(Payload)"}, {"fnv1a64", truncatedHash}}}};
        std::ofstream metadata(root / "source-cases.json", std::ios::trunc);
        metadata << manifest.dump(2) << '\n';
        require(metadata.good(), "write Mallett source case metadata");
        return {{{"missing", missing, missing.string() + ": Open(NotFound)", 0, false},
                 {"rows_80x3", rows, rows.string() + ": " + expectedShape, rowsHash, true},
                 {"columns_81x2", columns, columns.string() + ": " + expectedShape, columnsHash, true},
                 {"short_payload", truncated, truncated.string() + ": ShortRead(Payload)", truncatedHash, true}}};
    }
    void constructor_failures(JuicerTestSupport::RuntimeFixture& runtime, const std::filesystem::path& artifactRoot) {
        const auto sourceCases = source_failures(artifactRoot);
        std::vector<std::string> source_ids, exposure_fault_ids, precedence_ids;
        for (std::uint32_t route = 0; route < 4; ++route) {
            for (const auto& source : sourceCases)
                source_ids.emplace_back(std::to_string(route) + "/" + source.name);
            for (std::uint32_t operation : {3u, 5u, 6u})
                for (std::uint32_t fault : {1u, 2u, 4u})
                    exposure_fault_ids.emplace_back(std::to_string(route) + "/" + std::to_string(operation) + "/" + std::to_string(fault));
            for (int precedence = 0; precedence < 3; ++precedence)
                precedence_ids.emplace_back(std::to_string(route) + "/" + std::to_string(precedence));
        }
        JuicerTests::Cohort source_rows("mallett.constructor.source", source_ids);
        JuicerTests::Cohort exposure_fault_rows("mallett.constructor.exposure_fault", exposure_fault_ids);
        JuicerTests::Cohort precedence_rows("mallett.constructor.precedence", precedence_ids);
        for (std::uint32_t route = 0; route < 4; ++route) {
            auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019);
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<JuicerTestSupport::PreparedStateHold> held;
            require(JuicerTestSupport::prepare_state(runtime.borrow().assets, p, held, error).category == FJ_STATUS_SUCCESS && held, "constructor held-state seed");
            FjPreparedStateView seed{};
            require(held->view(seed, error).category == FJ_STATUS_SUCCESS, "constructor seed view");
            const auto retained = complete_result(seed, 1);
            for (const auto& source : sourceCases) {
                auto event = source_rows.case_event(std::to_string(route) + "/" + source.name);
                FaultScope cleanup;
                arm_source(source.path);
                require(fj_test_exposure_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "constructor midgray fault arm");
                std::optional<JuicerTestSupport::PreparedStateHold> failed;
                const auto status = JuicerTestSupport::prepare_state(runtime.borrow().assets, p, failed, error);
                require(status.category == FJ_STATUS_PREPARATION_FAILURE && status.api == FJ_API_NONE && status.native_code == 0 && !failed, "constructor source failure has no owner");
                require(std::string(diagnostic.data(), error.length) == source.diagnostic, "constructor exact source diagnostic: " + std::string(diagnostic.data(), error.length));
                require(fj_test_mallett_source_consumed() == 1 && fj_test_exposure_fault_consumed(3, 1) == 0, "source decode consumes once before midgray");
                FjPreparedStateView unchanged{};
                require(held->view(unchanged, error).category == FJ_STATUS_SUCCESS && complete_result(unchanged, 1) == retained, "source failure preserves complete held values");
                const auto later = JuicerTestSupport::prepare_state(runtime.borrow().assets, p, failed, error);
                require(later.category == FJ_STATUS_UNSUPPORTED_INPUT && !failed && fj_test_exposure_fault_consumed(3, 1) == 1, "constructor retry reaches skipped midgray fault");
                require(JuicerTestSupport::prepare_state(runtime.borrow().assets, p, failed, error).category == FJ_STATUS_SUCCESS && failed, "constructor source/fault one-shot recovery");
                event.passed();
            }
            for (std::uint32_t operation : {3u, 5u, 6u}) {
                for (std::uint32_t fault : {1u, 2u, 4u}) {
                    auto event = exposure_fault_rows.case_event(std::to_string(route) + "/" + std::to_string(operation) + "/" + std::to_string(fault));
                    FaultScope cleanup;
                    require(fj_test_exposure_arm_fault(operation, 1, fault).category == FJ_STATUS_SUCCESS, "constructor checked exposure arm");
                    std::optional<JuicerTestSupport::PreparedStateHold> failed;
                    const auto status = JuicerTestSupport::prepare_state(runtime.borrow().assets, p, failed, error);
                    const auto expected = fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                                : FJ_STATUS_PREPARATION_FAILURE;
                    require(status.category == expected && status.api == FJ_API_NONE && status.native_code == 0 && !failed, "constructor checked exposure category/empty owner");
                    const std::string expectedDiagnostic = fault == 2 ? "asset boundary panic" : operation == 3 ? std::string("Mallett mid-gray preparation: ") + (fault == 1 ? "exposure production boundary fault" : "fixed exposure preparation category fault")
                                                                                             : fault == 4       ? "MalformedRequiredResource phase=3B field=synthetic_reference_raw"
                                                                                             : operation == 5   ? "Reference source preparation: exposure production boundary fault"
                                                                                                                : "Mallett reference preparation: exposure production boundary fault";
                    require(std::string(diagnostic.data(), error.length) == expectedDiagnostic, "constructor checked exposure diagnostic: " + std::string(diagnostic.data(), error.length));
                    require(fj_test_exposure_fault_consumed(operation, fault) == 1, "constructor checked exposure consumed once");
                    FjPreparedStateView unchanged{};
                    require(held->view(unchanged, error).category == FJ_STATUS_SUCCESS && complete_result(unchanged, 1) == retained, "checked exposure failure preserves complete held values");
                    require(JuicerTestSupport::prepare_state(runtime.borrow().assets, p, failed, error).category == FJ_STATUS_SUCCESS && failed, "constructor checked exposure recovery");
                    event.passed();
                }
            }
            for (int precedence = 0; precedence < 3; ++precedence) {
                auto event = precedence_rows.case_event(std::to_string(route) + "/" + std::to_string(precedence));
                FaultScope cleanup;
                auto controls = p;
                if (precedence == 0)
                    controls.values.output_color_space = 255;
                if (precedence == 1)
                    require(fj_test_spectral_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "constructor inverse precedence arm");
                if (precedence == 2)
                    require(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 1, 1).category == FJ_STATUS_SUCCESS, "constructor color precedence arm");
                arm_source(sourceCases[0].path);
                std::optional<JuicerTestSupport::PreparedStateHold> failed;
                const auto status = JuicerTestSupport::prepare_state(runtime.borrow().assets, controls, failed, error);
                require(status.category == FJ_STATUS_PREPARATION_FAILURE && !failed && std::string(diagnostic.data(), error.length) == sourceCases[0].diagnostic, "source acquisition precedes output policy, inverse and scanner color");
                require(fj_test_mallett_source_consumed() == 1, "constructor precedence real decode consumed");
                if (precedence == 1) {
                    std::array<float, 81> weights{};
                    weights.fill(1.0f);
                    FjSpectralSInput input{{weights.data(), weights.size()}, {weights.data(), weights.size()}, {weights.data(), weights.size()}};
                    FjSpectralInverse inverse{};
                    require(fj_test_boundary_spectral_s_inverse(&input, &inverse, &error).category == FJ_STATUS_UNSUPPORTED_INPUT, "constructor source failure leaves inverse pending");
                }
                if (precedence == 2) {
                    const float white[3]{.95047f, 1, 1.08883f};
                    float matrix[9]{};
                    require(fj_test_boundary_cat02_matrix(white, white, matrix).category == FJ_STATUS_UNSUPPORTED_INPUT, "constructor source failure leaves first scanner color call pending");
                }
                event.passed();
            }
        }
        JuicerTests::Cohort normalization_rows("mallett.constructor.tc_normalization", {"0", "2"});
        for (int method : {0, 2}) {
            auto event = normalization_rows.case_event(std::to_string(method));
            FaultScope cleanup;
            const auto p = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, static_cast<Spektrafilm::RgbToRawMethod>(method));
            require(fj_test_exposure_arm_fault(4, 1, 4).category == FJ_STATUS_SUCCESS, "constructor TC normalization arm");
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<JuicerTestSupport::PreparedStateHold> failed;
            const auto status = JuicerTestSupport::prepare_state(runtime.borrow().assets, p, failed, error);
            require(status.category == FJ_STATUS_PREPARATION_FAILURE && !failed && fj_test_exposure_fault_consumed(4, 4) == 1, "constructor actual TC normalization category/one-shot");
            require(std::string(diagnostic.data(), error.length) == "TC mid-gray normalization: fixed exposure preparation category fault", "constructor TC normalization diagnostic");
            require(JuicerTestSupport::prepare_state(runtime.borrow().assets, p, failed, error).category == FJ_STATUS_SUCCESS && failed, "constructor TC normalization recovery");
            event.passed();
        }
    }
    struct Supersession {
        JuicerTestSupport::InstanceFixture* instance = nullptr;
        JuicerTestSupport::PreparationInput next;
        int mode = 0;
        bool fired = false;
        bool failed = false;
    };
    void supersede_after_prepare(void* opaque) noexcept {
        auto& value = *static_cast<Supersession*>(opaque);
        if (std::exchange(value.fired, true))
            return;
        try {
            if (value.mode == 0) {
                JuicerTestSupport::capture(*value.instance, value.next);
            } else if (value.mode == 1) {
                JuicerTestSupport::invalidate(*value.instance, "new invalid controls");
            } else {
                FjCaptureStart capture{};
                FjErrorBuffer error{nullptr, 0, 0};
                value.instance->begin_capture(FJ_CAPTURE_CHANGE, capture, error);
                value.failed = capture.kind != FJ_CAPTURE_STARTED;
            }
        } catch (...) {
            value.failed = true;
        }
    }
    void admission(JuicerTestSupport::RuntimeFixture& runtime, const Json& contract, const std::filesystem::path& artifactRoot) {
        require(contract.contains("diagnostic"), "frozen historical failure contract remains present");
        const auto sourceCases = source_failures(artifactRoot);
        std::vector<std::string> source_ids, exposure_fault_ids, supersession_ids, precedence_ids;
        for (int polarity = 0; polarity < 2; ++polarity)
            for (int route = 0; route < 2; ++route) {
                for (const auto& source : sourceCases)
                    source_ids.emplace_back(std::to_string(polarity) + "/" + std::to_string(route) + "/" + source.name);
                for (std::uint32_t operation : {3u, 5u, 6u})
                    for (std::uint32_t fault : {1u, 2u, 4u})
                        exposure_fault_ids.emplace_back(std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(operation) + "/" + std::to_string(fault));
                for (bool source_failure : {false, true})
                    for (int mode = 0; mode < 3; ++mode)
                        supersession_ids.emplace_back(std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(source_failure) + "/" + std::to_string(mode));
                for (int earlier = 0; earlier < 3; ++earlier)
                    precedence_ids.emplace_back(std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(earlier));
            }
        JuicerTests::Cohort source_rows("mallett.admission.source", source_ids);
        JuicerTests::Cohort exposure_fault_rows("mallett.admission.exposure_fault", exposure_fault_ids);
        JuicerTests::Cohort supersession_rows("mallett.admission.supersession", supersession_ids);
        JuicerTests::Cohort precedence_rows("mallett.admission.precedence", precedence_ids);
        for (int polarity = 0; polarity < 2; ++polarity)
            for (int route = 0; route < 2; ++route) {
                auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), Spektrafilm::RgbToRawMethod::Mallett2019);
                auto changed = p;
                changed.values.camera_exposure_compensation_ev = 1.5;
                for (const auto& source : sourceCases) {
                    auto event = source_rows.case_event(std::to_string(polarity) + "/" + std::to_string(route) + "/" + source.name);
                    FaultScope cleanup;
                    JuicerTestSupport::InstanceFixture state;
                    JuicerTestSupport::capture(state, p);
                    auto old = JuicerTestSupport::admit(runtime.borrow().assets, state);
                    require(old.kind == FJ_ADMISSION_READY && old.state, "initial admission");
                    const auto oldRecipe = JuicerTestSupport::published_view(old).recipe_hash;
                    const auto oldIdentity = old.state->input_identity();
                    const auto oldBuild = old.state->build();
                    const auto retained = complete_result(JuicerTestSupport::published_view(old), 1);
                    const auto beforeFailure = JuicerTestSupport::PublicationObserver::metrics(state);
                    arm_source(source.path);
                    JuicerTestSupport::capture(state, changed);
                    require(fj_test_exposure_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "mid-gray fault armed");
                    const auto failed = JuicerTestSupport::admit(runtime.borrow().assets, state);
                    require(failed.kind == FJ_ADMISSION_PREPARATION_FAILED && !failed.state && failed.diagnostic == source.diagnostic, "amendment 7 source diagnostic route=" + std::to_string(route) + " source=" + source.name + " observed=" + failed.diagnostic);
                    require(fj_test_mallett_source_consumed() == 1, "real decoder override consumed once");
                    require(fj_test_exposure_fault_consumed(3, 1) == 0, "missing source skips Rust binding");
                    const auto afterFailure = JuicerTestSupport::PublicationObserver::metrics(state);
                    require(afterFailure.publishedIdentity == beforeFailure.publishedIdentity && afterFailure.build == beforeFailure.build && afterFailure.inputIdentity == JuicerTestSupport::hash_input_identity(changed), "ordinary failure retains private A and current B identity");
                    require(old.state->input_identity() == oldIdentity && old.state->build() == oldBuild && JuicerTestSupport::published_view(old).recipe_hash == oldRecipe && complete_result(JuicerTestSupport::published_view(old), 1) == retained, "already admitted complete hold survives");
                    fj_test_mallett_source_clear();
                    const auto exceptional = JuicerTestSupport::admit(runtime.borrow().assets, state);
                    require(exceptional.kind == FJ_ADMISSION_PREPARATION_FAILED && fj_test_exposure_fault_consumed(3, 1) == 1, "valid recovery attempt consumes skipped fault before cleanup");
                    fj_test_exposure_clear_fault();
                    JuicerTestSupport::capture(state, p);
                    const auto reusedA = JuicerTestSupport::admit(runtime.borrow().assets, state);
                    require(reusedA.kind == FJ_ADMISSION_READY && reusedA.state && reusedA.state->input_identity() == oldIdentity && reusedA.state->build() == oldBuild, "return to A reuses its retained complete publication");
                    JuicerTestSupport::capture(state, changed);
                    const auto recovered = JuicerTestSupport::admit(runtime.borrow().assets, state);
                    require(recovered.kind == FJ_ADMISSION_READY && recovered.state && recovered.state->input_identity() == JuicerTestSupport::hash_input_identity(changed) && recovered.state->build() == oldBuild + 1, "recovery complete publication");
                    const auto reused = JuicerTestSupport::admit(runtime.borrow().assets, state);
                    require(reused.kind == FJ_ADMISSION_READY && reused.state && reused.state->input_identity() == recovered.state->input_identity() && reused.state->build() == recovered.state->build(), "same-input owner reuse");
                    require(fj_test_assets_release_cached_payloads(runtime.borrow().assets, nullptr).category == FJ_STATUS_SUCCESS, "release source/profile cache");
                    require(JuicerTestSupport::published_view(old).recipe_hash == oldRecipe && complete_result(JuicerTestSupport::published_view(old), 1) == retained && JuicerTestSupport::published_view(recovered).recipe_hash == JuicerTestSupport::published_view(reused).recipe_hash, "admitted complete holds survive cache release");
                    event.passed();
                }
                for (std::uint32_t operation : {3u, 5u, 6u})
                    for (std::uint32_t fault : {1u, 2u, 4u}) {
                        auto event = exposure_fault_rows.case_event(std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(operation) + "/" + std::to_string(fault));
                        FaultScope cleanup;
                        JuicerTestSupport::InstanceFixture state;
                        JuicerTestSupport::capture(state, p);
                        const auto old = JuicerTestSupport::admit(runtime.borrow().assets, state);
                        require(old.kind == FJ_ADMISSION_READY && old.state, "fault initial admission");
                        const auto metrics = JuicerTestSupport::PublicationObserver::metrics(state);
                        const auto old_complete = complete_result(JuicerTestSupport::published_view(old), 1);
                        JuicerTestSupport::capture(state, changed);
                        require(fj_test_exposure_arm_fault(operation, 1, fault).category == FJ_STATUS_SUCCESS, "actual operation fault arm");
                        const auto result = JuicerTestSupport::admit(runtime.borrow().assets, state);
                        if (fault == 2) {
                            require(result.kind == FJ_ADMISSION_POISONED && !result.state && result.delivery.category == FJ_STATUS_INTERNAL_FAILURE,
                                    "actual preparation panic poisons admission");
                            require(!result.diagnostic.empty(), "first preparation panic diagnostic is retained");
                            require(fj_test_exposure_fault_consumed(operation, fault) == 1, "actual caller consumed panic before cleanup");
                            const auto blocked = JuicerTestSupport::PublicationObserver::metrics(state);
                            require(blocked.pendingKind == kBlockedPendingKind && blocked.build == metrics.build && old.state->build() == metrics.build,
                                    "panic blocks the instance without advancing its committed publication");
                            require(complete_result(JuicerTestSupport::published_view(old), 1) == old_complete,
                                    "admitted old hold remains readable after poison");
                            const auto future = JuicerTestSupport::admit(runtime.borrow().assets, state);
                            require(future.kind == FJ_ADMISSION_POISONED && !future.state && future.delivery.category == FJ_STATUS_INTERNAL_FAILURE,
                                    "future admission on poisoned instance is rejected");
                            fj_test_exposure_clear_fault();
                            JuicerTestSupport::InstanceFixture recovery;
                            JuicerTestSupport::capture(recovery, p);
                            const auto recovery_a = JuicerTestSupport::admit(runtime.borrow().assets, recovery);
                            require(recovery_a.kind == FJ_ADMISSION_READY && recovery_a.state, "independent recovery instance admits A");
                            JuicerTestSupport::capture(recovery, changed);
                            const auto recovery_b = JuicerTestSupport::admit(runtime.borrow().assets, recovery);
                            require(recovery_b.kind == FJ_ADMISSION_READY && recovery_b.state && recovery_b.state->build() == recovery_a.state->build() + 1,
                                    "independent recovery instance admits B");
                            event.passed();
                            continue;
                        }
                        require(result.kind == FJ_ADMISSION_PREPARATION_FAILED && !result.state, "actual caller rejects fault");
                        require(fj_test_exposure_fault_consumed(operation, fault) == 1, "actual caller consumed before cleanup");
                        const auto failedMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
                        require(failedMetrics.publishedIdentity == metrics.publishedIdentity && failedMetrics.build == metrics.build && old.state->build() == metrics.build, "fault retains private publication and committed build");
                        fj_test_exposure_clear_fault();
                        JuicerTestSupport::capture(state, p);
                        const auto reused_a = JuicerTestSupport::admit(runtime.borrow().assets, state);
                        require(reused_a.kind == FJ_ADMISSION_READY && reused_a.state && reused_a.state->build() == metrics.build, "typed-fault return to A reuses the retained publication");
                        JuicerTestSupport::capture(state, changed);
                        const auto retry = JuicerTestSupport::admit(runtime.borrow().assets, state);
                        require(retry.kind == FJ_ADMISSION_READY && retry.state && retry.state->build() == metrics.build + 1, "typed-fault recovery commits B once");
                        event.passed();
                    }
                for (bool sourceFailure : {false, true})
                    for (int mode = 0; mode < 3; ++mode) {
                        auto event = supersession_rows.case_event(std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(sourceFailure) + "/" + std::to_string(mode));
                        FaultScope cleanup;
                        JuicerTestSupport::InstanceFixture state;
                        JuicerTestSupport::capture(state, p);
                        const auto held = JuicerTestSupport::admit(runtime.borrow().assets, state);
                        require(held.kind == FJ_ADMISSION_READY && held.state, "supersession initial");
                        const auto retained = complete_result(JuicerTestSupport::published_view(held), 1);
                        Supersession selection{&state, p, mode};
                        JuicerTestSupport::capture(state, changed);
                        fj_test_instance_prepare_hooks(nullptr, supersede_after_prepare, &selection);
                        if (sourceFailure)
                            arm_source(sourceCases[0].path);
                        else
                            require(fj_test_exposure_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "supersession fault arm");
                        const auto next = JuicerTestSupport::admit(runtime.borrow().assets, state);
                        require(selection.fired && !selection.failed && (sourceFailure ? fj_test_mallett_source_consumed() == 1 : fj_test_exposure_fault_consumed(3, 1) == 1), "obsolete attempt fault consumed before cleanup");
                        require(complete_result(JuicerTestSupport::published_view(held), 1) == retained, "supersession preserves admitted complete values");
                        require(mode == 0   ? next.kind == FJ_ADMISSION_READY && next.state && next.state->input_identity() == JuicerTestSupport::hash_input_identity(p) && next.state->build() == held.state->build()
                                : mode == 1 ? next.kind == FJ_ADMISSION_CAPTURE_FAILED && !next.state
                                            : next.kind == FJ_ADMISSION_CAPTURING && !next.state,
                                "supersession resolves current Rust capture state");
                        fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                        event.passed();
                    }
                for (int earlier = 0; earlier < 3; ++earlier) {
                    auto event = precedence_rows.case_event(std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(earlier));
                    FaultScope cleanup;
                    const auto& source = sourceCases[0];
                    arm_source(source.path);
                    auto bad = p;
                    JuicerTestSupport::InstanceFixture state;
                    JuicerTestSupport::capture(state, bad);
                    if (earlier == 0) {
                        bad.values.output_color_space = 255;
                        JuicerTestSupport::capture(state, bad);
                    }
                    if (earlier == 1)
                        require(fj_test_spectral_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "earlier inverse fault");
                    if (earlier == 2)
                        require(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 1, 1).category == FJ_STATUS_SUCCESS, "later scanner fault at first call");
                    const auto failed = JuicerTestSupport::admit(runtime.borrow().assets, state);
                    require(failed.kind == FJ_ADMISSION_PREPARATION_FAILED && JuicerTestSupport::PublicationObserver::metrics(state).build == 0, "precedence no publication");
                    require(failed.diagnostic == source.diagnostic, "amendment 7 source acquisition precedes later preparation");
                    require(fj_test_mallett_source_consumed() == 1, "precedence source override consumed");
                    fj_test_mallett_source_clear();
                    if (earlier == 1) {
                        std::array<float, 81> x{}, y{}, z{};
                        x.fill(1.0f);
                        y.fill(1.0f);
                        z.fill(1.0f);
                        FjSpectralSInput inverse{{x.data(), x.size()}, {y.data(), y.size()}, {z.data(), z.size()}};
                        FjSpectralInverse result{};
                        require(fj_test_boundary_spectral_s_inverse(&inverse, &result, nullptr).category == FJ_STATUS_UNSUPPORTED_INPUT, "S-inverse fault remained armed behind source failure");
                        fj_test_spectral_clear_fault();
                    }
                    if (earlier == 2) {
                        const float white[3]{.95047f, 1, 1.08883f};
                        float matrix[9];
                        require(fj_test_boundary_cat02_matrix(white, white, matrix).category == FJ_STATUS_UNSUPPORTED_INPUT, "late scanner fault remained armed, observed before cleanup");
                    }
                    event.passed();
                }
            }
        JuicerTests::Cohort tc_normalization_rows("mallett.admission.tc_normalization", {"0", "2"});
        for (int method : {0, 2}) {
            auto event = tc_normalization_rows.case_event(std::to_string(method));
            FaultScope cleanup;
            JuicerTestSupport::InstanceFixture state;
            auto p = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, static_cast<Spektrafilm::RgbToRawMethod>(method));
            JuicerTestSupport::capture(state, p);
            require(fj_test_exposure_arm_fault(4, 1, 4).category == FJ_STATUS_SUCCESS, "TC normalization fault arm");
            require(JuicerTestSupport::admit(runtime.borrow().assets, state).kind == FJ_ADMISSION_PREPARATION_FAILED && fj_test_exposure_fault_consumed(4, 4) == 1, "TC actual consumer");
            fj_test_exposure_clear_fault();
            require(JuicerTestSupport::admit(runtime.borrow().assets, state).kind == FJ_ADMISSION_READY, "TC normalization recovery");
            event.passed();
        }
        {
            JuicerTests::Cohort skipped_normalization("mallett.admission.skipped_tc_normalization", {"mallett"});
            auto event = skipped_normalization.case_event("mallett");
            FaultScope cleanup;
            JuicerTestSupport::InstanceFixture state;
            auto p = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Mallett2019);
            JuicerTestSupport::capture(state, p);
            require(fj_test_exposure_arm_fault(4, 1, 4).category == FJ_STATUS_SUCCESS, "skipped TC fault armed");
            require(JuicerTestSupport::admit(runtime.borrow().assets, state).kind == FJ_ADMISSION_READY, "Mallett skips TC normalization");
            require(fj_test_exposure_fault_consumed(4, 4) == 0, "Mallett leaves TC fault pending");
            FjMidgrayNormalization normalized{};
            std::array<char, 128> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            const auto rawStatus = fj_test_boundary_exposure_tc_midgray(1.0f, &normalized, &error);
            require(rawStatus.category == FJ_STATUS_PREPARATION_FAILURE && rawStatus.api == FJ_API_NONE && rawStatus.native_code == 0, "raw TC fixed-exposure status");
            require(std::string(diagnostic.data(), error.length) == "fixed exposure preparation category fault", "raw TC diagnostic remains unprefixed");
            require(fj_test_exposure_fault_consumed(4, 4) == 1, "raw TC consumes skipped fault");
            event.passed();
        }
    }
} // namespace
void fj_test_mallett_callers(const std::filesystem::path& resources, const Json& fixture, const std::filesystem::path& contractPath, bool admissionOnly, const std::filesystem::path& artifactRoot) {
    JuicerTestSupport::RuntimeFixture runtime;
    runtime.create(resources);
    if (admissionOnly) {
        std::ifstream stream(contractPath);
        const auto contract = Json::parse(stream);
        admission(runtime, contract, artifactRoot);
    } else {
        products(runtime, fixture);
        constructor_failures(runtime, artifactRoot);
    }
    JuicerTests::CaseEvent close_event("mallett.caller", admissionOnly ? "admission_owner_close" : "products_owner_close");
    require(runtime.close().category == FJ_STATUS_SUCCESS, "caller Runtime close");
    close_event.passed();
}
