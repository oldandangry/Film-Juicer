#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ColorTransforms.h"
#include "JuicerState.h"
#include "PreparedState.h"
#include "prepared_product_fixture.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"

namespace {
    using Json = nlohmann::json;
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
    ParamSnapshot snapshot(Spektrafilm::ScanRoute route, Spektrafilm::RgbToRawMethod method) {
        ParamSnapshot p;
        p.filmProfileKey = Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Negative ? "kodak_portra_400" : "fujifilm_provia_100f";
        p.printProfileKey = "kodak_portra_endura";
        p.scanRoute = route;
        p.spectralUpsamplingMode = static_cast<int>(method);
        p.cameraExposureCompensationEv = .75;
        p.cameraAutoExposureEnabled = 0;
        p.inputCompressionEnabled = 0;
        return p;
    }
    ParamSnapshot controls(const Json& row) {
        auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(row.at("polarity").get<int>() * 2 + row.at("route").get<int>()), static_cast<Spektrafilm::RgbToRawMethod>(row.at("method").get<int>()));
        const auto& c = row.at("controls");
        p.inputColorSpace = c.at("inputColorSpace").get<int>();
        p.inputCctfDecoding = c.at("inputCctfDecoding").get<int>();
        p.outputCctfEncoding = c.at("outputCctfEncoding").get<int>();
        p.cameraExposureCompensationEv = std::bit_cast<double>(c.at("cameraExposureCompensationEv").at("f64_bits").get<std::uint64_t>());
        return p;
    }
    void products(FjAssets* assets, const Json& fixture) {
        require(fixture.at("products").size() == 56, "all 56 Mallett product rows retained");
        for (std::size_t i = 0; i < fixture.at("products").size(); ++i) {
            const auto& row = fixture.at("products")[i];
            auto p = controls(row);
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<PreparedState> owner;
            const auto status = prepare_state(assets, p, owner, error);
            require(status.category == FJ_STATUS_SUCCESS && owner.has_value(), std::string(diagnostic.data(), error.length));
            FjPreparedStateView view{};
            require(owner->view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(view.route == static_cast<std::uint32_t>(p.scanRoute) && view.film_exposure.method == static_cast<std::uint32_t>(p.spectralUpsamplingMode), "prepared route/method identity");
            require(view.raw_midgray_bgr == (p.spectralUpsamplingMode == 1 ? 1u : 0u), "prepared raw-midgray channel order");
            const auto observed = complete_result(view, static_cast<std::uint32_t>(p.spectralUpsamplingMode));
            require(observed == row.at("expected"), "actual complete product " + std::to_string(i) + ": " + Json::diff(row.at("expected"), observed).dump());
            const auto& meter = fixture.at("metrics")[i];
            require(std::bit_cast<std::uint32_t>(view.film_reference.meter_scale) == meter.at("baseline_meter_scale").get<std::uint32_t>(), "meter exact");
            require(view.print_balance.filtered_main_illuminant_hash == meter.at("filtered_main_illuminant_hash").get<std::uint64_t>(), "filtered print identity exact");
            Json sensitivity = Json::array();
            for (std::size_t channel = 0; channel < 81; ++channel)
                sensitivity.push_back(bits(std::span(view.film_exposure.sensitivity_rgb.data + channel * 3, 3)));
            require(sensitivity == row.at("final_sensitivity") && span_bits(view.exposure_tables.illuminant, 81) == row.at("reference_spd"), "source/sensitivity remain exact");
        }
    }
    void pending(InstanceState& state, const ParamSnapshot& p) {
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.value = PendingParamsState::Valid{p, hash_params(p)};
    }
    bool admitted(const PendingRenderAdmissionResult& out, bool print) {
        return out.status == (print ? PendingRenderAdmissionStatus::AdmittedPrint : PendingRenderAdmissionStatus::AdmittedDirect) && (print ? bool(out.printState) : bool(out.directState));
    }
    std::uint64_t recipe_hash(const PendingRenderAdmissionResult& out, bool print) {
        require(admitted(out, print), "published recipe has matching route hold");
        return JuicerTestSupport::published_view(out).recipe_hash;
    }
    struct FaultScope {
        ~FaultScope() {
            fj_test_mallett_source_clear();
            fj_test_exposure_clear_fault();
            fj_test_spectral_clear_fault();
            fj_test_color_clear_fault();
            set_pending_capture_test_hook(nullptr, nullptr);
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
    void constructor_failures(FjAssets* assets, const std::filesystem::path& artifactRoot) {
        const auto sourceCases = source_failures(artifactRoot);
        for (std::uint32_t route = 0; route < 4; ++route) {
            auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019);
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<PreparedState> held;
            require(prepare_state(assets, p, held, error).category == FJ_STATUS_SUCCESS && held, "constructor held-state seed");
            FjPreparedStateView seed{};
            require(held->view(seed, error).category == FJ_STATUS_SUCCESS, "constructor seed view");
            const auto retained = complete_result(seed, 1);
            for (const auto& source : sourceCases) {
                FaultScope cleanup;
                arm_source(source.path);
                require(fj_test_exposure_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "constructor midgray fault arm");
                std::optional<PreparedState> failed;
                const auto status = prepare_state(assets, p, failed, error);
                require(status.category == FJ_STATUS_PREPARATION_FAILURE && status.api == FJ_API_NONE && status.native_code == 0 && !failed, "constructor source failure has no owner");
                require(std::string(diagnostic.data(), error.length) == source.diagnostic, "constructor exact source diagnostic: " + std::string(diagnostic.data(), error.length));
                require(fj_test_mallett_source_consumed() == 1 && fj_test_exposure_fault_consumed(3, 1) == 0, "source decode consumes once before midgray");
                FjPreparedStateView unchanged{};
                require(held->view(unchanged, error).category == FJ_STATUS_SUCCESS && complete_result(unchanged, 1) == retained, "source failure preserves complete held values");
                const auto later = prepare_state(assets, p, failed, error);
                require(later.category == FJ_STATUS_UNSUPPORTED_INPUT && !failed && fj_test_exposure_fault_consumed(3, 1) == 1, "constructor retry reaches skipped midgray fault");
                require(prepare_state(assets, p, failed, error).category == FJ_STATUS_SUCCESS && failed, "constructor source/fault one-shot recovery");
            }
            for (std::uint32_t operation : {3u, 5u, 6u}) {
                for (std::uint32_t fault : {1u, 2u, 4u}) {
                    FaultScope cleanup;
                    require(fj_test_exposure_arm_fault(operation, 1, fault).category == FJ_STATUS_SUCCESS, "constructor checked exposure arm");
                    std::optional<PreparedState> failed;
                    const auto status = prepare_state(assets, p, failed, error);
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
                    require(prepare_state(assets, p, failed, error).category == FJ_STATUS_SUCCESS && failed, "constructor checked exposure recovery");
                }
            }
            for (int precedence = 0; precedence < 3; ++precedence) {
                FaultScope cleanup;
                auto controls = p;
                if (precedence == 0)
                    controls.outputColorSpace = 255;
                if (precedence == 1)
                    require(fj_test_spectral_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "constructor inverse precedence arm");
                if (precedence == 2)
                    require(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 1, 1).category == FJ_STATUS_SUCCESS, "constructor color precedence arm");
                arm_source(sourceCases[0].path);
                std::optional<PreparedState> failed;
                const auto status = prepare_state(assets, controls, failed, error);
                require(status.category == FJ_STATUS_PREPARATION_FAILURE && !failed && std::string(diagnostic.data(), error.length) == sourceCases[0].diagnostic, "source acquisition precedes output policy, inverse and scanner color");
                require(fj_test_mallett_source_consumed() == 1, "constructor precedence real decode consumed");
                if (precedence == 1) {
                    std::array<float, 81> weights{};
                    weights.fill(1.0f);
                    FjSpectralSInput input{{weights.data(), weights.size()}, {weights.data(), weights.size()}, {weights.data(), weights.size()}};
                    FjSpectralInverse inverse{};
                    require(fj_legacy_spectral_s_inverse(&input, &inverse, &error).category == FJ_STATUS_UNSUPPORTED_INPUT, "constructor source failure leaves inverse pending");
                }
                if (precedence == 2) {
                    const float white[3]{.95047f, 1, 1.08883f};
                    float matrix[9]{};
                    require(fj_legacy_cat02_matrix(white, white, matrix).category == FJ_STATUS_UNSUPPORTED_INPUT, "constructor source failure leaves first scanner color call pending");
                }
            }
        }
        for (int method : {0, 2}) {
            FaultScope cleanup;
            const auto p = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, static_cast<Spektrafilm::RgbToRawMethod>(method));
            require(fj_test_exposure_arm_fault(4, 1, 4).category == FJ_STATUS_SUCCESS, "constructor TC normalization arm");
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<PreparedState> failed;
            const auto status = prepare_state(assets, p, failed, error);
            require(status.category == FJ_STATUS_PREPARATION_FAILURE && !failed && fj_test_exposure_fault_consumed(4, 4) == 1, "constructor actual TC normalization category/one-shot");
            require(std::string(diagnostic.data(), error.length) == "TC mid-gray normalization: fixed exposure preparation category fault", "constructor TC normalization diagnostic");
            require(prepare_state(assets, p, failed, error).category == FJ_STATUS_SUCCESS && failed, "constructor TC normalization recovery");
        }
    }
    struct Supersession {
        ParamSnapshot next;
        int mode = 0;
        bool fired = false;
    };
    void supersede(InstanceState& state, void* opaque) {
        auto& value = *static_cast<Supersession*>(opaque);
        if (std::exchange(value.fired, true))
            return;
        std::lock_guard<std::mutex> lock(state.pending.m);
        if (value.mode == 1)
            state.pending.value = PendingParamsState::InvalidSnapshotControls{"new invalid controls"};
        else if (value.mode == 2)
            state.pending.value = PendingParamsState::Uninitialized{};
        else
            state.pending.value = PendingParamsState::Valid{value.next, hash_params(value.next)};
    }
    void admission(const Json& contract, const std::filesystem::path& artifactRoot) {
        require(contract.contains("diagnostic"), "frozen historical failure contract remains present");
        const auto sourceCases = source_failures(artifactRoot);
        for (int polarity = 0; polarity < 2; ++polarity)
            for (int route = 0; route < 2; ++route) {
                const bool print = route == 1;
                auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), Spektrafilm::RgbToRawMethod::Mallett2019);
                auto changed = p;
                changed.cameraExposureCompensationEv = 1.5;
                for (const auto& source : sourceCases) {
                    FaultScope cleanup;
                    InstanceState state;
                    pending(state, p);
                    auto old = admit_pending_render_state(state);
                    require(admitted(old, print), "initial admission");
                    const auto oldRecipe = recipe_hash(old, print), hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                    const auto retained = complete_result(JuicerTestSupport::published_view(old), 1);
                    arm_source(source.path);
                    pending(state, changed);
                    require(fj_test_exposure_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "mid-gray fault armed");
                    const auto failed = admit_pending_render_state(state);
                    require(failed.status == PendingRenderAdmissionStatus::RebuildFailed && !failed.directState && !failed.printState && failed.diagnostic == source.diagnostic, "amendment 7 source diagnostic route=" + std::to_string(route) + " source=" + source.name + " observed=" + failed.diagnostic);
                    require(fj_test_mallett_source_consumed() == 1, "real decoder override consumed once");
                    require(fj_test_exposure_fault_consumed(3, 1) == 0, "missing source skips Rust binding");
                    require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "ordinary failure keeps hash/counter");
                    require(print ? !JuicerAtomic::load_shared_ptr(&state.activePrintState) : !JuicerAtomic::load_shared_ptr(&state.activeDirectState), "ordinary failure clears selected slot");
                    require(recipe_hash(old, print) == oldRecipe && complete_result(JuicerTestSupport::published_view(old), 1) == retained, "already admitted complete hold survives");
                    {
                        std::lock_guard<std::mutex> lock(state.pending.m);
                        const auto* current = std::get_if<PendingParamsState::Valid>(&state.pending.value);
                        require(current && current->fullHash == hash_params(changed), "failed changed-current snapshot remains valid pending");
                    }
                    fj_test_mallett_source_clear();
                    const auto exceptional = admit_pending_render_state(state);
                    require(exceptional.status == PendingRenderAdmissionStatus::RebuildFailed && fj_test_exposure_fault_consumed(3, 1) == 1, "valid recovery attempt consumes skipped fault before cleanup");
                    fj_test_exposure_clear_fault();
                    const auto recovered = admit_pending_render_state(state);
                    require(admitted(recovered, print) && state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "recovery complete publication");
                    const auto reused = admit_pending_render_state(state);
                    require(reused.directState == recovered.directState && reused.printState == recovered.printState && state.buildCounterNext.load() == counter + 1, "same-input owner reuse");
                    require(JuicerProcess::root().assets().release_cached_payloads().category == FJ_STATUS_SUCCESS, "release source/profile cache");
                    require(recipe_hash(old, print) == oldRecipe && complete_result(JuicerTestSupport::published_view(old), 1) == retained && recipe_hash(recovered, print) == recipe_hash(reused, print), "admitted complete holds survive cache release");
                }
                for (std::uint32_t operation : {3u, 5u, 6u})
                    for (std::uint32_t fault : {1u, 2u, 4u}) {
                        FaultScope cleanup;
                        InstanceState state;
                        pending(state, p);
                        const auto old = admit_pending_render_state(state);
                        require(admitted(old, print), "fault initial admission");
                        const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                        pending(state, changed);
                        require(fj_test_exposure_arm_fault(operation, 1, fault).category == FJ_STATUS_SUCCESS, "actual operation fault arm");
                        const auto result = admit_pending_render_state(state);
                        require(result.status == PendingRenderAdmissionStatus::RebuildFailed && !result.directState && !result.printState, "actual caller rejects fault");
                        require(fj_test_exposure_fault_consumed(operation, fault) == 1, "actual caller consumed before cleanup");
                        require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "fault keeps counters");
                        if (fault == 4)
                            require(print ? !JuicerAtomic::load_shared_ptr(&state.activePrintState) : !JuicerAtomic::load_shared_ptr(&state.activeDirectState), "ordinary failure slot clear");
                        else
                            require(JuicerAtomic::load_shared_ptr(&state.activeDirectState) == old.directState && JuicerAtomic::load_shared_ptr(&state.activePrintState) == old.printState, "exception retains publication");
                        fj_test_exposure_clear_fault();
                        require(admitted(admit_pending_render_state(state), print), "typed-fault recovery");
                    }
                for (bool sourceFailure : {false, true})
                    for (int mode = 0; mode < 3; ++mode) {
                        FaultScope cleanup;
                        InstanceState state;
                        pending(state, p);
                        const auto held = admit_pending_render_state(state);
                        require(admitted(held, print), "supersession initial");
                        const auto retained = complete_result(JuicerTestSupport::published_view(held), 1);
                        Supersession selection{p, mode, false};
                        pending(state, changed);
                        set_pending_capture_test_hook(supersede, &selection);
                        if (sourceFailure)
                            arm_source(sourceCases[0].path);
                        else
                            require(fj_test_exposure_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "supersession fault arm");
                        const auto next = admit_pending_render_state(state);
                        require(selection.fired && (sourceFailure ? fj_test_mallett_source_consumed() == 1 : fj_test_exposure_fault_consumed(3, 1) == 1), "obsolete attempt fault consumed before cleanup");
                        require(complete_result(JuicerTestSupport::published_view(held), 1) == retained, "supersession preserves admitted complete values");
                        require(mode == 0 ? admitted(next, print) : next.status == (mode == 1 ? PendingRenderAdmissionStatus::InvalidSnapshotControls : PendingRenderAdmissionStatus::NeedsSnapshotAcquisition), "supersession resolves current snapshot");
                        set_pending_capture_test_hook(nullptr, nullptr);
                    }
                for (int earlier = 0; earlier < 3; ++earlier) {
                    FaultScope cleanup;
                    const auto& source = sourceCases[0];
                    arm_source(source.path);
                    auto bad = p;
                    InstanceState state;
                    pending(state, bad);
                    if (earlier == 0) {
                        bad.outputColorSpace = 255;
                        pending(state, bad);
                    }
                    if (earlier == 1)
                        require(fj_test_spectral_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "earlier inverse fault");
                    if (earlier == 2)
                        require(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 1, 1).category == FJ_STATUS_SUCCESS, "later scanner fault at first call");
                    const auto failed = admit_pending_render_state(state);
                    require(failed.status == PendingRenderAdmissionStatus::RebuildFailed && state.lastHash.load() == 0 && state.buildCounterNext.load() == 0, "precedence no publication");
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
                        require(fj_legacy_spectral_s_inverse(&inverse, &result, nullptr).category == FJ_STATUS_UNSUPPORTED_INPUT, "S-inverse fault remained armed behind source failure");
                        fj_test_spectral_clear_fault();
                    }
                    if (earlier == 2) {
                        const float white[3]{.95047f, 1, 1.08883f};
                        float matrix[9];
                        require(fj_legacy_cat02_matrix(white, white, matrix).category == FJ_STATUS_UNSUPPORTED_INPUT, "late scanner fault remained armed, observed before cleanup");
                    }
                }
            }
        for (int method : {0, 2}) {
            FaultScope cleanup;
            InstanceState state;
            auto p = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, static_cast<Spektrafilm::RgbToRawMethod>(method));
            pending(state, p);
            require(fj_test_exposure_arm_fault(4, 1, 4).category == FJ_STATUS_SUCCESS, "TC normalization fault arm");
            require(admit_pending_render_state(state).status == PendingRenderAdmissionStatus::RebuildFailed && fj_test_exposure_fault_consumed(4, 4) == 1, "TC actual consumer");
            fj_test_exposure_clear_fault();
            require(admitted(admit_pending_render_state(state), false), "TC normalization recovery");
        }
        {
            FaultScope cleanup;
            InstanceState state;
            auto p = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Mallett2019);
            pending(state, p);
            require(fj_test_exposure_arm_fault(4, 1, 4).category == FJ_STATUS_SUCCESS, "skipped TC fault armed");
            require(admitted(admit_pending_render_state(state), false), "Mallett skips TC normalization");
            require(fj_test_exposure_fault_consumed(4, 4) == 0, "Mallett leaves TC fault pending");
            FjMidgrayNormalization normalized{};
            std::array<char, 128> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            const auto rawStatus = fj_legacy_exposure_tc_midgray(1.0f, &normalized, &error);
            require(rawStatus.category == FJ_STATUS_PREPARATION_FAILURE && rawStatus.api == FJ_API_NONE && rawStatus.native_code == 0, "raw TC fixed-exposure status");
            require(std::string(diagnostic.data(), error.length) == "fixed exposure preparation category fault", "raw TC diagnostic remains unprefixed");
            require(fj_test_exposure_fault_consumed(4, 4) == 1, "raw TC consumes skipped fault");
        }
    }
} // namespace
void fj_test_mallett_callers(const std::filesystem::path& resources, const Json& fixture, const std::filesystem::path& contractPath, bool admissionOnly, const std::filesystem::path& artifactRoot) {
    JuicerCuda::Owner owner;
    owner.create(resources);
    JuicerProcess::root().ensure_bootstrap();
    FjAssets* assets = JuicerProcess::root().assets().borrow_rust_assets();
    require(assets != nullptr, "prepared-state assets owner");
    if (admissionOnly) {
        std::ifstream stream(contractPath);
        const auto contract = Json::parse(stream);
        admission(contract, artifactRoot);
    } else {
        products(assets, fixture);
        constructor_failures(assets, artifactRoot);
    }
    require(owner.close().category == FJ_STATUS_SUCCESS, "caller owner close");
}
