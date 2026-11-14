#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>

#include "SpectralData.h"
#include "GamutCompression.h"
#include "gamut_fixture.h"
#include "owner_fixture.h"
#include "ProcessRoot.h"
#include "color_fixture.h"
#include "scanner_fixture.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "juicer_test_api.h"
#include "case_results.h"
#include "exposure_fixture.h"
#include "print_illuminant_fixture.h"
#include "juicer_cuda_owner.h"
#include "prepared_product_fixture.h"
#include "prepared_frame_fixture.h"
#include "spectral_source_fixture.h"

static_assert(sizeof(float) == 4 && sizeof(std::array<float, 3>) == 12 && sizeof(std::array<float, 9>) == 36);
extern "C" int fj_test_color_abi_c();
extern "C" int fj_test_cat02_abi_c();
extern "C" int fj_test_input_color_abi_c();

namespace {
    using Json = nlohmann::json;
    constexpr std::uint32_t kBlockedPendingKind = 6;
    void require(bool condition, const std::string& diagnostic) {
        if (!condition) {
            throw std::runtime_error(diagnostic);
        }
    }
    FjPreparedStateView prepared_view(const JuicerTestSupport::PreparedStateHold& product) {
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjPreparedStateView view{};
        const FjStatus status = product.view(view, error);
        require(status.category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
        return view;
    }
    void shared_fixture_loading_tests(const std::filesystem::path& artifacts) {
        const auto root = artifacts / std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        const auto color = root / "color";
        const auto updates = root / "exposure/updates/color";
        std::filesystem::create_directories(color);
        std::filesystem::create_directories(updates);
        const auto write = [](const std::filesystem::path& path, const Json& value) {
            std::ofstream file(path);
            file << value;
        };
        write(color / "manifest.json", {{"payloads", {{"linux-debug.json", "shared.json"}, {"linux-release.json", "shared.json"}}}});
        write(color / "shared.json", {{"expected_blocks", "blocks.json"}, {"recipes", {{{"expected", {{"sensitivity_record", 0}}}}, {{"expected", {{"sensitivity_record", 0}}}}}}});
        write(color / "blocks.json", {{"schema_version", 1}, {"sensitivity", {{0, 0x80000000u, 0x7fc00123u}}}});
        JuicerTests::Cohort logical_rows("color.shared_fixture.logical", {"linux-debug.json", "linux-release.json"});
        for (const auto* logical : {"linux-debug.json", "linux-release.json"}) {
            auto logical_event = logical_rows.case_event(logical);
            const auto word = std::string(logical) == "linux-debug.json" ? 11 : 17;
            write(updates / logical, {{"original_fixture", logical}, {"patch", {{{"op", "test"}, {"path", "/recipes/0/expected/sensitivity/0"}, {"value", 0}}, {{"op", "replace"}, {"path", "/recipes/0/expected/sensitivity/0"}, {"value", word}}, {{"op", "test"}, {"path", "/recipes/1/expected/sensitivity/0"}, {"value", 0}}}}});
            const auto fixture = load_exposure_qualified_fixture(color / logical);
            require(fixture.at("recipes").at(0).at("expected").at("sensitivity").at(0) == word, "logical alias selects its own ordered supplement");
            require(fixture.at("recipes").at(1).at("expected").at("sensitivity") == Json::array({0, 0x80000000u, 0x7fc00123u}), "expanded expected cases retain independent bits and patch state");
            logical_event.passed();
        }
        bool unknownRejected = false;
        try {
            (void)load_shared_fixture(color / "unknown.json", "manifest.json");
        } catch (const Json::out_of_range&) {
            unknownRejected = true;
        }
        require(unknownRejected, "unregistered logical preset is rejected");
        JuicerTests::Cohort index_rows("color.shared_fixture.invalid_index", 3u);
        std::size_t index_ordinal = 0;
        for (const auto& index : Json::array({-1, 0.5, std::numeric_limits<std::uint64_t>::max()})) {
            auto index_event = index_rows.case_event(std::to_string(index_ordinal++));
            write(color / "shared.json", {{"expected_blocks", "blocks.json"}, {"recipes", {{{"expected", {{"sensitivity_record", index}}}}}}});
            bool referenceRejected = false;
            try {
                (void)load_shared_fixture(color / "linux-debug.json", "manifest.json");
            } catch (const std::runtime_error& error) {
                referenceRejected = std::string(error.what()) == "invalid expected block reference";
            }
            require(referenceRejected, "expected block references require a bounded nonnegative integer");
            index_event.passed();
        }
        write(color / "shared.json", {{"recipes", Json::array()}});
        const auto contextManifest = Json{{"payloads", {{"linux-debug.json", "shared.json"}, {"linux-release.json", "shared.json"}}}, {"contexts", {{"linux-debug.json", "context.json"}}}};
        write(color / "manifest.json", contextManifest);
        write(color / "context.json", {{"preset", "linux-debug"}, {"primitives", {0, 0x80000000u, 0x7fc00123u}}});
        const auto contextual = load_shared_fixture(color / "linux-debug.json", "manifest.json");
        require(contextual.at("preset") == "linux-debug" && contextual.at("primitives") == Json::array({0, 0x80000000u, 0x7fc00123u}) && contextual.at("recipes").empty(), "shared payload retains exact preset context");
        bool missingContextRejected = false;
        try {
            (void)load_shared_fixture(color / "linux-release.json", "manifest.json");
        } catch (const Json::out_of_range&) {
            missingContextRejected = true;
        }
        require(missingContextRejected, "registered logical preset requires its context");
        write(color / "context.json", {{"recipes", Json::array()}});
        bool overlapRejected = false;
        try {
            (void)load_shared_fixture(color / "linux-debug.json", "manifest.json");
        } catch (const std::runtime_error& error) {
            overlapRejected = std::string(error.what()) == "shared fixture context overlaps payload";
        }
        require(overlapRejected, "context cannot overwrite shared expectations");
        write(color / "manifest.json", {{"payloads", {{"linux-debug.json", "shared.json"}}}});
        write(updates / "linux-debug.json", {{"original_fixture", "linux-release.json"}, {"patch", Json::array()}});
        bool identityRejected = false;
        try {
            (void)load_exposure_qualified_fixture(color / "linux-debug.json");
        } catch (const std::runtime_error& error) {
            identityRejected = std::string(error.what()) == "wrong A5 fixture supplement";
        }
        require(identityRejected, "aliased payload never bypasses supplement identity admission");
    }
    Json load_preset_witness(const std::filesystem::path& logicalPath, const char* manifestName) {
        std::ifstream manifestFile(logicalPath.parent_path() / manifestName);
        const auto manifest = Json::parse(manifestFile);
        require(manifest.at("payloads").contains(logicalPath.filename().string()), "required logical preset membership");
        std::ifstream source(logicalPath.parent_path() / manifest.at("preset_witnesses").get<std::string>());
        const auto witness = Json::parse(source).at(logicalPath.filename().string());
        require(witness.at("foreign_capture") != logicalPath.filename().string() && manifest.at("payloads").contains(witness.at("foreign_capture").get<std::string>()), "different native logical preset witness membership");
        return witness;
    }
    void check_preset_witness(const Json& actual, const Json& expected, const Json& witness) {
        const auto pointer = Json::json_pointer(witness.at("pointer").get<std::string>());
        require(expected.at(pointer) == witness.at("expected"), "selected native value witness is still the qualified expectation");
        require(actual.at(pointer) == witness.at("expected") && actual.at(pointer) != witness.at("rejected"), "actual native value rejects a different preset baseline");
    }
    template <typename Range>
    Json bits(const Range& values) {
        Json out = Json::array();
        for (float value : values) {
            out.push_back(std::bit_cast<std::uint32_t>(value));
        }
        return out;
    }
    Json bits(const float* values, std::size_t count) {
        Json out = Json::array();
        for (std::size_t i = 0; i < count; ++i)
            out.push_back(std::bit_cast<std::uint32_t>(values[i]));
        return out;
    }
    template <typename Range>
    Json double_bits(const Range& values) {
        Json out = Json::array();
        for (double value : values) {
            out.push_back(std::bit_cast<std::uint64_t>(value));
        }
        return out;
    }
    std::array<float, 3> triplet(const Json& values) {
        std::array<float, 3> out{};
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = std::bit_cast<float>(values.at(i).get<std::uint32_t>());
        }
        return out;
    }
    std::array<float, 3> multiply_matrix(const float matrix[9], const std::array<float, 3>& rgb) {
        return {matrix[0] * rgb[0] + matrix[1] * rgb[1] + matrix[2] * rgb[2],
                matrix[3] * rgb[0] + matrix[4] * rgb[1] + matrix[5] * rgb[2],
                matrix[6] * rgb[0] + matrix[7] * rgb[1] + matrix[8] * rgb[2]};
    }
    Json leaf_result(const Json& input) {
        const auto source = triplet(input.at("source"));
        const auto destination = triplet(input.at("destination"));
        const auto xyz = triplet(input.at("xyz"));
        const Spectral::ChromaticAdaptationWhites whites{source.data(), destination.data()};
        std::array<float, 9> matrix{};
        require(fj_test_cat16_matrix(whites.source, whites.destination, matrix.data()).category == FJ_STATUS_SUCCESS, "matrix facade");
        std::array<float, 3> adapted{};
        require(fj_test_adapt_cat16(xyz.data(), whites.source, whites.destination, adapted.data()).category == FJ_STATUS_SUCCESS, "scalar facade");
        return {{"matrix", bits(matrix)}, {"adapted", bits(adapted)}};
    }
    JuicerTestSupport::PreparationInput snapshot(Spektrafilm::ScanRoute route, Spektrafilm::RgbToRawMethod method) {
        JuicerTestSupport::PreparationInput out;
        out.film_key = Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Negative ? "kodak_portra_400" : "fujifilm_provia_100f";
        out.values.scan_route = static_cast<std::uint32_t>(route);
        out.values.spectral_upsampling_mode = static_cast<int>(method);
        out.values.camera_exposure_compensation_ev = 0.75;
        out.values.camera_auto_exposure_enabled = 0;
        out.values.input_compression_enabled = 0;
        return out;
    }
    std::vector<std::pair<std::string, JuicerTestSupport::PreparationInput>> scenarios() {
        std::vector<std::pair<std::string, JuicerTestSupport::PreparationInput>> out;
        for (int polarity = 0; polarity < 2; ++polarity) {
            for (int route = 0; route < 2; ++route) {
                for (int method = 0; method < 3; ++method) {
                    out.emplace_back("recipe-" + std::to_string(polarity) + "-" + std::to_string(route) + "-" + std::to_string(method), snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), static_cast<Spektrafilm::RgbToRawMethod>(method)));
                }
            }
        }
        auto cctf = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Hanatos2025);
        cctf.values.output_cctf_encoding = 0;
        out.emplace_back("recipe-cctf-off", cctf);
        auto input = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Hanatos2025);
        input.values.input_color_space = Spectral::input_color_space_to_index(Spectral::InputColorSpace::SRGB_Rec709);
        out.emplace_back("recipe-input-srgb", input);
        auto output = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Hanatos2025);
        output.values.output_color_space = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
        out.emplace_back("recipe-output-dci", output);
        return out;
    }
    JuicerTestSupport::PreparedStateHold prepare(const FjAssets* assets, const JuicerTestSupport::PreparationInput& input) {
        return JuicerTestSupport::prepare_product(assets, input);
    }
    Json complete_result(const JuicerTestSupport::PreparedStateHold& product) {
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjPreparedStateView view{};
        const auto inspected = product.view(view, error);
        require(inspected.category == FJ_STATUS_SUCCESS, std::string(text.data(), error.length));
        const auto& raw = view.film_exposure;
        const auto& reference = view.film_reference;
        const auto& balance = view.print_balance;
        return {{"input_adapt", bits(raw.input_xyz_adapt, 9)}, {"source_white", bits(view.input_policy.nominal_white_xyz, 3)}, {"projection_white", bits(view.input_policy.projection_white_xyz, 3)}, {"baseline_raw", bits(reference.baseline_raw_rgb, 3)}, {"compensated_raw", bits(reference.compensated_raw_rgb, 3)}, {"baseline_density", bits(reference.baseline_density_cmy, 3)}, {"compensated_density", bits(reference.compensated_density_cmy, 3)}, {"print_baseline_raw", bits(balance.baseline_raw_rgb, 3)}, {"print_compensated_raw", bits(balance.compensated_raw_rgb, 3)}, {"print_factors", bits(JuicerTestSupport::captured_print_factors(view))}, {"hashes", {raw.hash, raw.tc_lut_hash, reference.hash, balance.hash, view.print_recipe_hash, view.scanner_recipe_hash, view.output_color.gamut_recipe_hash, view.recipe_hash, view.upload_core_hash, view.scanner_hash}}, {"gamut_entry", bits(view.output_color.native_rgb_to_d65_xyz, 9)}, {"gamut_exit", bits(view.output_color.d65_xyz_to_native_rgb, 9)}};
    }
    Json complete_result(JuicerTestSupport::PreparedStateHold&& product) {
        return complete_result(product);
    }
    Json transform_result(int space) {
        Gamut::OutputGamutTransform out;
        std::string diagnostic;
        require(JuicerTestFixture::Gamut::output_transform(static_cast<OutputEncoding::ColorSpace>(space), out, diagnostic), diagnostic);
        return {{"entry", bits(out.nativeRgbToD65Xyz)}, {"exit", bits(out.d65XyzToNativeRgb)}, {"hash", out.hash}, {"contract_hash", out.contractHash}};
    }
    Json table_result(int space) {
        Gamut::OutputGamutTransform transform;
        Gamut::OutputBoundaryTable table;
        std::string diagnostic;
        require(JuicerTestFixture::Gamut::output_transform(static_cast<OutputEncoding::ColorSpace>(space), transform, diagnostic), diagnostic);
        require(JuicerTestFixture::Gamut::output_boundary(transform, table), table.diagnostic);
        return {{"cmax", bits(table.cmax)}, {"hash", table.hash}, {"transform_hash", table.transformHash}, {"contract_hash", table.contractHash}};
    }
    void expect_status(FjStatus status, std::uint32_t category) {
        require(status.category == category && status.api == FJ_API_NONE && status.native_code == 0, "color status category/API/code: expected " + std::to_string(category) + "/0/0, actual " + std::to_string(status.category) + "/" + std::to_string(status.api) + "/" + std::to_string(status.native_code));
    }
    bool same_values(const Json& actual, const Json& expected) {
        if (actual.size() != expected.size()) {
            return false;
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto a = actual.at(i).get<std::uint32_t>();
            const auto e = expected.at(i).get<std::uint32_t>();
            if (std::isnan(std::bit_cast<float>(e))) {
                if (!std::isnan(std::bit_cast<float>(a))) {
                    return false;
                }
            } else if (a != e) {
                return false;
            }
        }
        return true;
    }
    bool color_facade_operation(std::uint32_t operation) {
        return operation == FJ_TEST_CAT02_ADAPT || operation == FJ_TEST_INPUT_TO_DWG || operation == FJ_TEST_INPUT_TO_LINEAR_SRGB || operation == FJ_TEST_DWG_TO_XYZ;
    }
    FjStatus arm_color_fault(std::uint32_t operation, std::uint32_t index, std::uint32_t fault) {
        fj_test_color_clear_fault();
        fj_test_color_clear_facade_fault();
        return color_facade_operation(operation) ? fj_test_color_arm_facade_fault(operation, index, fault) : fj_test_color_arm_fault(operation, index, fault);
    }
    FjStatus raw_call(std::uint32_t operation) {
        constexpr std::array<float, 3> white{0.95045593f, 1.0f, 1.08905775f};
        constexpr std::array<float, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
        const FjInputColorConversion input{FJ_INPUT_SRGB_REC709, 1, 1, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}};
        FjInputColorMatrices matrices{};
        std::array<float, 9> matrix{};
        std::array<float, 3> adapted{}, rgb{};
        switch (operation) {
            case FJ_TEST_CAT16_MATRIX:
                return fj_test_boundary_cat16_matrix(white.data(), white.data(), matrix.data());
            case FJ_TEST_CAT16_ADAPT:
                return fj_test_boundary_adapt_cat16(white.data(), white.data(), white.data(), adapted.data());
            case FJ_TEST_CAT02_MATRIX:
                return fj_test_boundary_cat02_matrix(white.data(), white.data(), matrix.data());
            case FJ_TEST_CAT02_ADAPT:
                return fj_test_adapt_cat02(white.data(), white.data(), white.data(), adapted.data());
            case FJ_TEST_INPUT_MATRICES:
                return fj_test_boundary_input_matrices(FJ_INPUT_DWG, &matrices);
            case FJ_TEST_INPUT_TO_DWG:
                return fj_test_input_to_dwg(&input, white.data(), 1, rgb.data(), adapted.data());
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                return fj_test_input_to_linear_srgb(&input, white.data(), identity.data(), rgb.data(), adapted.data());
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                return fj_test_boundary_linear_srgb_to_xyz(white.data(), adapted.data());
            case FJ_TEST_DWG_TO_XYZ:
                return fj_test_dwg_to_xyz(white.data(), adapted.data());
            case FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ:
                return fj_test_boundary_project_linear_rgb_to_xyz(white.data(), identity.data(), identity.data(), adapted.data());
            default:
                throw std::runtime_error("unknown color operation");
        }
    }
    struct FaultScope {
        FaultScope() = default;
        FaultScope(const FaultScope&) = delete;
        FaultScope& operator=(const FaultScope&) = delete;
        ~FaultScope() {
            const auto status = fj_test_color_clear_fault();
            fj_test_color_clear_facade_fault();
            if (status.category != FJ_STATUS_SUCCESS) {
                std::abort();
            }
            fj_test_profile_fault(0);
            fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
        }
    };
    void native_color_fault_tests(std::uint32_t operation, const char* diagnostic) {
        const auto invoke = [operation] {
            constexpr std::array<float, 3> source{0.95045593f, 1.0f, 1.08905775f};
            constexpr std::array<float, 3> destination{0.96429568f, 1.0f, 0.82510460f};
            constexpr std::array<float, 3> xyz{0.184f, -0.0f, -0.25f};
            const Spectral::ChromaticAdaptationWhites whites{source.data(), destination.data()};
            switch (operation) {
                case FJ_TEST_CAT16_MATRIX:
                    return bits(JuicerTestFixture::Color::cat16_matrix(whites));
                case FJ_TEST_CAT16_ADAPT:
                    return bits(JuicerTestFixture::Color::adapt_cat16(xyz, whites));
                case FJ_TEST_CAT02_MATRIX:
                    return bits(JuicerTestFixture::Color::cat02_matrix(whites));
                default:
                    throw std::runtime_error("unknown native color fault operation");
            }
        };
        const Json expected = invoke();
        JuicerTests::Cohort native_color_faults_rows("color.native_fault/operation-" + std::to_string(operation), 2u);
        std::size_t native_color_faults_index = 0;
        for (const auto fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
            auto native_color_faults_event = native_color_faults_rows.case_event(std::to_string(native_color_faults_index++));
            FaultScope cleanup;
            expect_status(arm_color_fault(operation, 2, fault), FJ_STATUS_SUCCESS);
            require(invoke() == expected, "native bridge consumes exactly one successful call before scheduled fault");
            bool thrown = false;
            try {
                invoke();
            } catch (const JuicerCuda::ExecutionFailure& error) {
                expect_status(error.failure.status, fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                require(error.failure.diagnostic == diagnostic && !error.deferredDirError, "native color bridge exact diagnostic and non-deferred failure");
                thrown = true;
            }
            require(thrown, "native color bridge must propagate the scheduled typed failure");
            require(invoke() == expected, "native color bridge recovers on the next same-thread call with exact bits");
            require(invoke() == expected, "native color bridge fault is consumed once");

            native_color_faults_event.passed();
        }
    }
    void leaf_tests(const Json& fixture) {
        bool scalarDistinct = false;
        for (const auto& row : fixture.at("leaf")) {
            JuicerTests::CaseEvent event("color.leaf", row.at("id").get<std::string>());
            const auto& expected = row.at("expected");
            const auto actual = leaf_result(row);
            const std::string id = row.at("id").get<std::string>();
            require(same_values(actual.at("matrix"), expected.at("matrix")), id + " facade matrix bits/classification");
            require(same_values(actual.at("adapted"), expected.at("adapted")), id + " facade scalar bits/classification");
            const auto source = triplet(row.at("source"));
            const auto destination = triplet(row.at("destination"));
            const auto xyz = triplet(row.at("xyz"));
            const Spectral::ChromaticAdaptationWhites whites{source.data(), destination.data()};
            const auto matrix = JuicerTestFixture::Color::cat16_matrix(whites);
            const auto adapted = JuicerTestFixture::Color::adapt_cat16(xyz, whites);
            require(same_values(bits(matrix), expected.at("matrix")), id + " production matrix bits/classification");
            require(same_values(bits(adapted), expected.at("adapted")), id + " production scalar bits/classification");
            std::array<float, 3> multiplied{};
            multiplied = multiply_matrix(matrix.data(), xyz);
            scalarDistinct = scalarDistinct || !same_values(bits(multiplied), expected.at("adapted"));
            event.passed();
        }
        JuicerTests::CaseEvent controls_event("color.leaf", "abi_fault_and_comparator_controls");
        require(scalarDistinct, "captures distinguish scalar from matrix product shortcut");
        auto altered = fixture.at("leaf").at(7).at("expected").at("matrix");
        const auto expected = altered;
        std::swap(altered[1], altered[3]);
        require(!same_values(altered, expected), "transpose negative control");
        altered = expected;
        altered[0] = altered[0].get<std::uint32_t>() + 1;
        require(!same_values(altered, expected), "one-bit negative control");
        require(fj_test_color_abi_c() == 0, "real C transport/null/panic/canaries");
        FaultScope cleanup;
        JuicerTests::Cohort cat16_invalid_hooks_rows("color.cat16-invalid-hooks", 6u);
        std::size_t cat16_invalid_hooks_index = 0;
        JuicerTests::Cohort cat16_hook_operations_rows("color.cat16-hook-operations", 2u);
        std::size_t cat16_hook_operations_index = 0;
        for (std::uint32_t operation : {FJ_TEST_CAT16_MATRIX, FJ_TEST_CAT16_ADAPT}) {
            auto cat16_hook_operations_event = cat16_hook_operations_rows.case_event(std::to_string(cat16_hook_operations_index++));
            expect_status(arm_color_fault(operation, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation == 1 ? 2 : 1), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            for (const auto& invalid : std::array<std::array<std::uint32_t, 3>, 3>{{{0, 1, 1}, {operation, 0, 1}, {operation, 1, 99}}}) {
                auto cat16_invalid_hooks_event = cat16_invalid_hooks_rows.case_event(std::to_string(cat16_invalid_hooks_index++));
                expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(arm_color_fault(invalid[0], invalid[1], invalid[2]), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);

                cat16_invalid_hooks_event.passed();
            }
            expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            expect_status(arm_color_fault(operation, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);

            cat16_hook_operations_event.passed();
        }
        expect_status(arm_color_fault(FJ_TEST_CAT16_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
        FjStatus otherThread{};
        std::thread worker([&] {
            otherThread = raw_call(FJ_TEST_CAT16_MATRIX);
        });
        worker.join();
        expect_status(otherThread, FJ_STATUS_SUCCESS);
        expect_status(raw_call(FJ_TEST_CAT16_MATRIX), FJ_STATUS_UNSUPPORTED_INPUT);
        native_color_fault_tests(FJ_TEST_CAT16_MATRIX, "CAT16 matrix preparation failed");
        native_color_fault_tests(FJ_TEST_CAT16_ADAPT, "CAT16 scalar adaptation failed");
        controls_event.passed();
        std::puts("CAT16: 933 independent scalar/matrix cases, C ABI, fault and comparator controls passed");
    }
    bool admitted(const JuicerTestSupport::PublicationAdmission& result) {
        return result.kind == FJ_ADMISSION_READY && bool(result.state);
    }
    void require_poisoned_admission(
        const FjAssets* assets,
        const JuicerTestSupport::PublicationAdmission& result,
        JuicerTestSupport::InstanceFixture& instance,
        const std::shared_ptr<const JuicerTestSupport::PreparedStateHold>& held,
        std::uint64_t build) {
        require(result.kind == FJ_ADMISSION_POISONED && !result.state && result.delivery.category == FJ_STATUS_INTERNAL_FAILURE,
                "preparation panic poisons admission with no state hold");
        require(!result.diagnostic.empty(), "first panic diagnostic is retained");
        const auto observation = JuicerTestSupport::PublicationObserver::metrics(instance);
        require(observation.pendingKind == kBlockedPendingKind && observation.build == build, "poison blocks pending admission without advancing committed build");
        require(held && held->build() == build, "pre-poison admitted hold remains alive");
        const auto blocked = JuicerTestSupport::admit(assets, instance);
        require(blocked.kind == FJ_ADMISSION_POISONED && !blocked.state && blocked.delivery.category == FJ_STATUS_INTERNAL_FAILURE,
                "future admission on poisoned instance is rejected");
    }
    void require_preparation_failure(const JuicerTestSupport::PublicationAdmission& result, std::uint32_t fault) {
        require(result.kind == FJ_ADMISSION_PREPARATION_FAILED && !result.state, "current Rust preparation attempt fails without an admitted hold");
        require(result.delivery.category == (fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE), "preparation failure category preserved");
        require(!result.diagnostic.empty(), "preparation failure diagnostic retained");
    }
    struct Supersession {
        JuicerTestSupport::InstanceFixture* instance = nullptr;
        JuicerTestSupport::PreparationInput newer;
        int mode = 0;
        bool fired = false;
        bool failed = false;
    };
    void supersede_capture(void* context) noexcept {
        auto& selection = *static_cast<Supersession*>(context);
        if (std::exchange(selection.fired, true))
            return;
        try {
            if (selection.mode == 0) {
                JuicerTestSupport::capture(*selection.instance, selection.newer);
            } else if (selection.mode == 1) {
                JuicerTestSupport::invalidate(*selection.instance, "new invalid controls");
            } else {
                FjCaptureStart capture{};
                FjErrorBuffer error{nullptr, 0, 0};
                selection.instance->begin_capture(FJ_CAPTURE_CHANGE, capture, error);
                selection.failed = capture.kind != FJ_CAPTURE_STARTED;
            }
        } catch (...) {
            selection.failed = true;
        }
    }
    void admission_tests(const FjAssets* assets) {
        std::size_t failures = 0;
        std::size_t supersessions = 0;
        std::vector<std::string> admission_ids;
        for (int route : {0, 1})
            for (int method : {0, 2})
                for (std::uint32_t operation : {FJ_TEST_CAT16_MATRIX, FJ_TEST_CAT16_ADAPT})
                    for (std::uint32_t index = 1; index <= (operation == FJ_TEST_CAT16_MATRIX ? 1u : 2u); ++index)
                        for (const auto fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC})
                            admission_ids.emplace_back("route-" + std::to_string(route) + "/method-" + std::to_string(method) + "/operation-" + std::to_string(operation) + "/index-" + std::to_string(index) + "/fault-" + std::to_string(fault));
        JuicerTests::Cohort admission_rows("color.admission.fault_tuple", admission_ids);
        JuicerTests::Cohort cat16_supersession_rows("color.cat16-supersession", 72u);
        std::size_t supersession_index = 0;
        for (int route : {0, 1}) {
            for (int method : {0, 2}) {
                const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                auto changed = controls;
                changed.values.camera_exposure_compensation_ev = 1.0;
                for (std::uint32_t operation : {FJ_TEST_CAT16_MATRIX, FJ_TEST_CAT16_ADAPT}) {
                    const std::uint32_t calls = operation == FJ_TEST_CAT16_MATRIX ? 1u : 2u;
                    for (std::uint32_t index = 1; index <= calls; ++index) {
                        for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                            const auto id = "route-" + std::to_string(route) + "/method-" + std::to_string(method) + "/operation-" + std::to_string(operation) + "/index-" + std::to_string(index) + "/fault-" + std::to_string(fault);
                            auto row_event = admission_rows.case_event(id);
                            FaultScope cleanup;
                            JuicerTestSupport::InstanceFixture state;
                            JuicerTestSupport::capture(state, controls);
                            const auto old = JuicerTestSupport::admit(assets, state);
                            require(admitted(old), "initial complete state admitted");
                            const auto old_view = JuicerTestSupport::published_view(old);
                            const auto old_recipe = old_view.recipe_hash;
                            const auto old_build = old.state->build();
                            const auto old_identity = old.state->input_identity();
                            const auto before = JuicerTestSupport::PublicationObserver::metrics(state);
                            JuicerTestSupport::capture(state, changed);
                            expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                            const auto failed = JuicerTestSupport::admit(assets, state);
                            if (fault == FJ_TEST_COLOR_PANIC) {
                                require_poisoned_admission(assets, failed, state, old.state, old_build);
                                for (std::uint32_t call = 0; call < index; ++call)
                                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                require(JuicerTestSupport::published_view(old).recipe_hash == old_recipe, "poison leaves admitted A hold readable");
                                ++failures;

                                JuicerTestSupport::InstanceFixture recovery;
                                JuicerTestSupport::capture(recovery, controls);
                                const auto recovery_a = JuicerTestSupport::admit(assets, recovery);
                                require(admitted(recovery_a), "independent recovery instance admits A");
                                JuicerTestSupport::capture(recovery, changed);
                                const auto recovery_b = JuicerTestSupport::admit(assets, recovery);
                                require(admitted(recovery_b) && recovery_b.state->build() == recovery_a.state->build() + 1,
                                        "independent recovery instance prepares B");

                                for (int mode = 0; mode < 3; ++mode) {
                                    auto event = cat16_supersession_rows.case_event(std::to_string(supersession_index++));
                                    JuicerTestSupport::InstanceFixture superseded;
                                    JuicerTestSupport::capture(superseded, changed);
                                    const auto held_b = JuicerTestSupport::admit(assets, superseded);
                                    require(admitted(held_b), "independent supersession instance publishes B before A attempt");
                                    const auto held_build = held_b.state->build();
                                    const auto held_recipe = JuicerTestSupport::published_view(held_b).recipe_hash;
                                    JuicerTestSupport::capture(superseded, controls);
                                    Supersession selection{&superseded, changed, mode};
                                    if (operation == FJ_TEST_CAT16_ADAPT)
                                        fj_test_instance_prepare_hooks(supersede_capture, nullptr, &selection);
                                    else
                                        fj_test_instance_prepare_hooks(nullptr, supersede_capture, &selection);
                                    expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                    const auto obsolete = JuicerTestSupport::admit(assets, superseded);
                                    require(selection.fired && !selection.failed, "panic hook completed newer capture before poison classification");
                                    require_poisoned_admission(assets, obsolete, superseded, held_b.state, held_build);
                                    require(JuicerTestSupport::published_view(held_b).recipe_hash == held_recipe, "obsolete panic preserves previous B hold");
                                    for (std::uint32_t call = 0; call < index; ++call)
                                        expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                    fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                                    expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                                    ++supersessions;
                                    event.passed();
                                }
                                row_event.passed();
                                continue;
                            }
                            require_preparation_failure(failed, fault);
                            for (std::uint32_t call = 0; call < index; ++call)
                                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                            const auto after = JuicerTestSupport::PublicationObserver::metrics(state);
                            require(after.publishedIdentity == before.publishedIdentity && after.build == before.build && after.inputIdentity == JuicerTestSupport::hash_input_identity(changed), "failed B retains private A and current B identity without committing a build");
                            require(old.state->input_identity() == old_identity && old.state->build() == old_build && JuicerTestSupport::published_view(old).recipe_hash == old_recipe, "already admitted A hold survives failed B");
                            ++failures;

                            JuicerTestSupport::capture(state, controls);
                            const auto reused_a = JuicerTestSupport::admit(assets, state);
                            require(admitted(reused_a) && reused_a.state->input_identity() == old_identity && reused_a.state->build() == old_build, "return to A reuses the retained complete state");
                            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                            JuicerTestSupport::capture(state, changed);
                            const auto recovered = JuicerTestSupport::admit(assets, state);
                            require(admitted(recovered) && recovered.state->input_identity() == JuicerTestSupport::hash_input_identity(changed) && recovered.state->build() == old_build + 1, "successful B commits one publication after recovery");
                            require(JuicerTestSupport::published_view(old).recipe_hash == old_recipe, "held A remains readable after B recovery");

                            for (int mode = 0; mode < 3; ++mode) {
                                auto event = cat16_supersession_rows.case_event(std::to_string(supersession_index++));
                                Supersession selection{&state, changed, mode};
                                JuicerTestSupport::capture(state, controls);
                                fj_test_instance_prepare_hooks(nullptr, supersede_capture, &selection);
                                expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                const auto current = JuicerTestSupport::admit(assets, state);
                                require(selection.fired && !selection.failed, "real preparation hook completed a newer Rust capture");
                                if (mode == 0) {
                                    require(admitted(current) && current.state->input_identity() == JuicerTestSupport::hash_input_identity(changed) && current.state->build() == recovered.state->build(), "obsolete candidate is discarded and current B is reused");
                                } else {
                                    require((mode == 1 ? current.kind == FJ_ADMISSION_CAPTURE_FAILED : current.kind == FJ_ADMISSION_CAPTURING) && !current.state, "current invalid or in-progress capture blocks admission");
                                }
                                const auto metrics = JuicerTestSupport::PublicationObserver::metrics(state);
                                require(metrics.build == recovered.state->build() && JuicerTestSupport::published_view(recovered).recipe_hash == JuicerTestSupport::published_view(current.kind == FJ_ADMISSION_READY ? current : recovered).recipe_hash, "superseded attempt does not advance or mutate committed publication");
                                for (std::uint32_t call = 0; call < index; ++call)
                                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                                expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                                ++supersessions;
                                event.passed();
                            }
                            row_event.passed();
                        }
                    }
                }
            }
        }
        require(failures == 24 && supersessions == 72, "CAT16 actual Rust admission rows remain complete");
        std::printf("CAT16 admission: %zu failures/recoveries and %zu real-owner supersession witnesses passed\n", failures, supersessions);
    }
    void construction_controls(const FjAssets* assets) {
        JuicerTests::Cohort cat16_construction_routes_rows("color.cat16-construction-routes", 2u);
        std::size_t cat16_construction_routes_index = 0;
        for (int route = 0; route < 2; ++route) {
            auto cat16_construction_routes_event = cat16_construction_routes_rows.case_event(std::to_string(cat16_construction_routes_index++));
            JuicerTestSupport::InstanceFixture state;
            const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Hanatos2025);
            JuicerTestSupport::capture(state, controls);
            const auto old = JuicerTestSupport::admit(assets, state);
            require(admitted(old), "complete-state baseline");
            auto changed = controls;
            changed.values.camera_exposure_compensation_ev = 1.0;
            const auto build = old.state->build();
            changed.film_key = "missing-cat16-control-profile";
            JuicerTestSupport::capture(state, changed);
            const auto ordinary = JuicerTestSupport::admit(assets, state);
            require(ordinary.kind == FJ_ADMISSION_PREPARATION_FAILED && !ordinary.state, "invalid profile preparation fails current input");
            require(JuicerTestSupport::PublicationObserver::metrics(state).build == build && old.state->build() == build, "invalid profile retains prior private publication and admitted hold");
            JuicerTestSupport::capture(state, controls);
            const auto reused = JuicerTestSupport::admit(assets, state);
            require(admitted(reused) && reused.state->build() == build, "complete A capture reuses retained publication");

            cat16_construction_routes_event.passed();
        }
        std::puts("Construction: actual missing-profile failure retains the private publication and recovers by recapturing A");
    }
    void preparation_tests(const FjAssets* assets, const Json& fixture) {
        auto identity = fixture.at("transforms").at(0);
        identity["hash"] = identity["hash"].get<std::uint64_t>() ^ 1;
        require(identity != fixture.at("transforms").at(0), "identity negative control");
        for (int space = 0; space < 9; ++space) {
            JuicerTests::CaseEvent event("color.preparation.transform", std::to_string(space));
            require(transform_result(space) == fixture.at("transforms").at(space), "native-parent output transform/identity space=" + std::to_string(space));
            event.passed();
        }
        for (const auto& row : fixture.at("tables")) {
            JuicerTests::CaseEvent event("color.preparation.table", std::to_string(row.at("space").get<int>()));
            require(table_result(row.at("space").get<int>()) == row.at("expected"), "complete Cmax/table/hash");
            event.passed();
        }
        const auto cases = scenarios();
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [id, controls] = cases[i];
            JuicerTests::CaseEvent event("color.preparation.recipe", id);
            auto product = JuicerTestSupport::prepare_product(assets, controls);
            require(id == fixture.at("recipes").at(i).at("id").get<std::string>(), "fixed recipe case membership");
            require(complete_result(product) == fixture.at("recipes").at(i).at("expected"), id + " independent complete arrays/identities");
            auto repeated = JuicerTestSupport::prepare_product(assets, controls);
            require(complete_result(repeated) == complete_result(product), id + " repeat complete construction");
            event.passed();
        }
        for (int route = 0; route < 2; ++route) {
            JuicerTests::CaseEvent event("color.preparation.encoding_reuse", std::to_string(route));
            const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Hanatos2025);
            auto product = JuicerTestSupport::prepare_product(assets, controls);
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            FjPreparedStateView productView{};
            require(product.view(productView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            auto changed = controls;
            changed.values.output_cctf_encoding = 0;
            auto encodedOff = JuicerTestSupport::prepare_product(assets, changed);
            FjPreparedStateView encodedOffView{};
            require(encodedOff.view(encodedOffView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(bits(productView.film_exposure.input_xyz_adapt, 9) == bits(encodedOffView.film_exposure.input_xyz_adapt, 9) && productView.film_exposure.tc_lut_hash == encodedOffView.film_exposure.tc_lut_hash, "encoding change preserves film/TC family");
            require(productView.output_boundary.entry_identity != 0 && productView.output_boundary.entry_identity == encodedOffView.output_boundary.entry_identity && productView.output_boundary.cmax.data == encodedOffView.output_boundary.cmax.data, "encoding change reuses actual boundary owner");
            require(productView.scanner_lut.hash == encodedOffView.scanner_lut.hash, "encoding change preserves scanner descriptor identity");
            require(productView.scanner_recipe_hash != encodedOffView.scanner_recipe_hash && productView.recipe_hash != encodedOffView.recipe_hash && productView.scanner_color.hash != encodedOffView.scanner_color.hash, "encoding identities change at actual owners");
            const auto retainedBits = bits(productView.output_boundary.cmax.data, productView.output_boundary.cmax.count);
            const auto retainedIdentity = productView.output_boundary.entry_identity;
            const auto* retainedBacking = productView.output_boundary.cmax.data;
            expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
            require(bits(productView.output_boundary.cmax.data, productView.output_boundary.cmax.count) == retainedBits && complete_result(product) == complete_result(JuicerTestSupport::prepare_product(assets, controls)), "retained state/table stays readable across cache release");
            auto rebuilt = JuicerTestSupport::prepare_product(assets, controls);
            FjPreparedStateView rebuiltView{};
            require(rebuilt.view(rebuiltView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(rebuiltView.output_boundary.entry_identity != retainedIdentity && rebuiltView.output_boundary.cmax.data != retainedBacking && bits(rebuiltView.output_boundary.cmax.data, rebuiltView.output_boundary.cmax.count) == retainedBits, "cache release creates new equal owner without mutating retained table");
            JuicerTestSupport::InstanceFixture state;
            JuicerTestSupport::capture(state, controls);
            const auto old = JuicerTestSupport::admit(assets, state);
            const auto counter = old.state->build();
            FaultScope cleanup;
            expect_status(arm_color_fault(FJ_TEST_CAT16_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            const auto repeated = JuicerTestSupport::admit(assets, state);
            require(admitted(repeated) && repeated.state->input_identity() == old.state->input_identity() && repeated.state->build() == counter, "unchanged admission reuses actual state");
            expect_status(raw_call(FJ_TEST_CAT16_MATRIX), FJ_STATUS_UNSUPPORTED_INPUT);
            event.passed();
        }
        JuicerTests::CaseEvent controls_event("color.preparation", "admission_and_construction_controls");
        admission_tests(assets);
        construction_controls(assets);
        controls_event.passed();
        std::puts("Preparation: native-parent complete identities, tables, contributing controls and retained-owner reuse passed");
    }

    namespace Cat02Fixtures {
        using Json = nlohmann::json;
        void require(bool condition, const std::string& diagnostic) {
            if (!condition)
                throw std::runtime_error(diagnostic);
        }
        template <typename Range>
        Json bits(const Range& values) {
            Json out = Json::array();
            for (float value : values)
                out.push_back(std::bit_cast<std::uint32_t>(value));
            return out;
        }
        Json bits(const float* values, std::size_t count) {
            Json out = Json::array();
            for (std::size_t i = 0; i < count; ++i)
                out.push_back(std::bit_cast<std::uint32_t>(values[i]));
            return out;
        }
        std::array<float, 3> triplet(const Json& values) {
            std::array<float, 3> out{};
            for (std::size_t i = 0; i < out.size(); ++i)
                out[i] = std::bit_cast<float>(values.at(i).get<std::uint32_t>());
            return out;
        }
        Json leaf_result(const Json& input) {
            const auto source = triplet(input.at("source"));
            const auto destination = triplet(input.at("destination"));
            const auto xyz = triplet(input.at("xyz"));
            const Spectral::ChromaticAdaptationWhites whites{source.data(), destination.data()};
            const auto matrix = JuicerTestFixture::Color::cat02_matrix(whites);
            std::array<float, 3> adapted{};
            expect_status(fj_test_adapt_cat02(xyz.data(), whites.source, whites.destination, adapted.data()), FJ_STATUS_SUCCESS);
            return {{"matrix", bits(matrix)}, {"adapted", bits(adapted)}};
        }
        JuicerTestSupport::PreparationInput snapshot(Spektrafilm::ScanRoute route, Spektrafilm::RgbToRawMethod method) {
            JuicerTestSupport::PreparationInput out;
            out.film_key = Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Negative ? "kodak_portra_400" : "fujifilm_provia_100f";
            out.print_key = "kodak_portra_endura";
            out.values.scan_route = static_cast<std::uint32_t>(route);
            out.values.spectral_upsampling_mode = static_cast<int>(method);
            out.values.camera_exposure_compensation_ev = 0.75;
            out.values.camera_auto_exposure_enabled = 0;
            out.values.input_compression_enabled = 0;
            return out;
        }
        std::vector<std::pair<std::string, JuicerTestSupport::PreparationInput>> scenarios() {
            std::vector<std::pair<std::string, JuicerTestSupport::PreparationInput>> out;
            for (int polarity = 0; polarity < 2; ++polarity) {
                for (int route = 0; route < 2; ++route) {
                    const auto scanRoute = static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route);
                    for (int input = 0; input < 4; ++input) {
                        for (int decoding = 0; decoding < 2; ++decoding) {
                            auto controls = snapshot(scanRoute, Spektrafilm::RgbToRawMethod::Mallett2019);
                            controls.values.input_color_space = input;
                            controls.values.input_cctf_decoding = decoding;
                            out.emplace_back("mallett-" + std::to_string(polarity) + "-" + std::to_string(route) + "-input-" + std::to_string(input) + "-decode-" + std::to_string(decoding), controls);
                        }
                    }
                    for (int method : {0, 2})
                        out.emplace_back("tc-" + std::to_string(polarity) + "-" + std::to_string(route) + "-method-" + std::to_string(method), snapshot(scanRoute, static_cast<Spektrafilm::RgbToRawMethod>(method)));
                    for (int method = 0; method < 3; ++method) {
                        auto controls = snapshot(scanRoute, static_cast<Spektrafilm::RgbToRawMethod>(method));
                        controls.values.output_cctf_encoding = 0;
                        out.emplace_back("encoding-off-" + std::to_string(polarity) + "-" + std::to_string(route) + "-method-" + std::to_string(method), controls);
                    }
                    auto controls = snapshot(scanRoute, Spektrafilm::RgbToRawMethod::Mallett2019);
                    controls.values.output_color_space = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
                    out.emplace_back("output-dci-" + std::to_string(polarity) + "-" + std::to_string(route), controls);
                }
            }
            return out;
        }
        JuicerTestSupport::PreparedStateHold prepare(const FjAssets* assets, const JuicerTestSupport::PreparationInput& input) {
            return JuicerTestSupport::prepare_product(assets, input);
        }
        Json color_result(const Scanner::ColorRuntime& color) {
            return {{"cat02", bits(color.cat02)}, {"xyz_to_rgb", bits(color.xyzToRgb)}, {"illuminant_xyz", bits(color.illuminantXYZ)}, {"encoding", {static_cast<int>(color.encoding.colorSpace), color.encoding.applyCctfEncoding, color.encoding.inputIsOutputSpace}}, {"gamut_recipe_hash", color.outputGamutRecipeHash}, {"hash", color.hash}};
        }
        Json color_result(const FjScannerColor& color) {
            return {{"cat02", bits(color.cat02, 9)}, {"xyz_to_rgb", bits(color.xyz_to_rgb, 9)}, {"illuminant_xyz", bits(color.illuminant_xyz, 3)}, {"encoding", {color.output_space, (color.flags & 1u) != 0, (color.flags & 2u) != 0}}, {"gamut_recipe_hash", color.active_gamut_recipe_hash}, {"hash", color.hash}};
        }
        Json tables_result(const FjPreparedSpectralView& tables) {
            const std::array<float, 1> zeros{};
            return {{"lambda", bits(tables.lambda_nm.data, tables.lambda_nm.count)}, {"K", tables.lambda_nm.count}, {"delta_lambda", bits(std::array{5.0f})}, {"inv_yn", bits(std::array{tables.inv_yn})}, {"white_xyz", bits(tables.white_xyz, 3)}, {"ref_illum_white_xyz", bits(tables.white_xyz, 3)}, {"ax", bits(tables.weighted_xyz[0].data, tables.weighted_xyz[0].count)}, {"ay", bits(tables.weighted_xyz[1].data, tables.weighted_xyz[1].count)}, {"az", bits(tables.weighted_xyz[2].data, tables.weighted_xyz[2].count)}, {"xbar", bits(tables.observer_xyz[0].data, tables.observer_xyz[0].count)}, {"ybar", bits(tables.observer_xyz[1].data, tables.observer_xyz[1].count)}, {"zbar", bits(tables.observer_xyz[2].data, tables.observer_xyz[2].count)}, {"illum", bits(tables.illuminant.data, tables.illuminant.count)}, {"eps_c", bits(tables.dyes_cmy[0].data, tables.dyes_cmy[0].count)}, {"eps_m", bits(tables.dyes_cmy[1].data, tables.dyes_cmy[1].count)}, {"eps_y", bits(tables.dyes_cmy[2].data, tables.dyes_cmy[2].count)}, {"base_density_min", bits(tables.baseline_min.data, tables.baseline_min.count)}, {"base_density_mid", bits(tables.baseline_mid.data, tables.baseline_mid.count)}, {"has_baseline", tables.has_baseline != 0}, {"baseline_mix", bits(zeros)}, {"illuminant_hash", tables.illuminant_hash}, {"tables_hash", tables.hash}};
        }
        Json scanner_descriptor(const FjScannerLutDescriptor& descriptor) {
            return {{"route", static_cast<int>(descriptor.route)}, {"medium", static_cast<int>(descriptor.medium)}, {"polarity", static_cast<int>(descriptor.capture_polarity)}, {"density_bounds_hash", descriptor.density_bounds_hash}, {"channel_density_hash", descriptor.channel_density_hash}, {"base_density_hash", descriptor.base_density_hash}, {"illuminant_hash", descriptor.viewing_illuminant_hash}, {"observer_hash", descriptor.observer_hash}, {"resolution", descriptor.resolution}, {"hash", descriptor.hash}};
        }
        Json complete_result(const JuicerTestSupport::PreparedStateHold& product) {
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            FjPreparedStateView view{};
            const auto inspected = product.view(view, error);
            require(inspected.category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            const auto& raw = view.film_exposure;
            const auto& reference = view.film_reference;
            const auto& balance = view.print_balance;
            const auto& input = view.input_policy;
            const int sensitivityMethod = raw.method == FJ_RAW_MALLETT_2019 ? 1 : 0;
            return {{"input_rgb_to_xyz", bits(raw.input_rgb_to_xyz, 9)}, {"input_adapt", bits(raw.input_xyz_adapt, 9)}, {"xyz_to_linear_srgb", bits(raw.xyz_to_linear_srgb, 9)}, {"source_white", bits(input.nominal_white_xyz, 3)}, {"projection_white", bits(input.projection_white_xyz, 3)}, {"config_rgb_to_xyz", bits(raw.input_rgb_to_xyz, 9)}, {"config_adapt", bits(raw.input_xyz_adapt, 9)}, {"raw_midgray", bits(view.raw_midgray, 3)}, {"midgray_dwg", bits(view.midgray_dwg_rgb, 3)}, {"midgray_scale_green", bits(std::array{view.midgray_scale, view.raw_green})}, {"config_whites", {bits(input.nominal_white_xyz, 3), bits(input.projection_white_xyz, 3), bits(input.projection_white_xyz, 3)}}, {"config_flags", {raw.input_color_space, (raw.flags & FJ_FILM_DECODE_CCTF) != 0, sensitivityMethod, (raw.flags & FJ_FILM_ADAPT_INPUT) != 0, true, true}}, {"spd_s_inv", bits(view.s_inverse, 9)}, {"baseline_raw", bits(reference.baseline_raw_rgb, 3)}, {"compensated_raw", bits(reference.compensated_raw_rgb, 3)}, {"baseline_density", bits(reference.baseline_density_cmy, 3)}, {"compensated_density", bits(reference.compensated_density_cmy, 3)}, {"print_baseline_raw", bits(balance.baseline_raw_rgb, 3)}, {"print_compensated_raw", bits(balance.compensated_raw_rgb, 3)}, {"print_factors", bits(JuicerTestSupport::captured_print_factors(view))}, {"hashes", {raw.hash, raw.tc_lut_hash, reference.hash, balance.hash, view.print_recipe_hash, view.scanner_recipe_hash, view.output_color.gamut_recipe_hash, view.recipe_hash, view.upload_core_hash, view.scanner_hash}}, {"scanner_color", color_result(view.scanner_color)}, {"scanner_tables", tables_result(view.scanner_tables)}, {"scanner_descriptor", scanner_descriptor(view.scanner_lut)}, {"gamut_entry", bits(view.output_color.native_rgb_to_d65_xyz, 9)}, {"gamut_exit", bits(view.output_color.d65_xyz_to_native_rgb, 9)}};
        }
        Scanner::ScannerIlluminant scanner_illuminant(JuicerTestSupport::PreparedStateHold& product) {
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            FjPreparedStateView view{};
            require(product.view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            const auto& tables = view.scanner_tables;
            Scanner::ScannerIlluminant out;
            out.curve.linear.assign(tables.illuminant.data, tables.illuminant.data + tables.illuminant.count);
            out.curve.lambda_nm.assign(tables.lambda_nm.data, tables.lambda_nm.data + tables.lambda_nm.count);
            std::copy_n(view.scanner_color.illuminant_xyz, 3, out.whiteXYZ);
            const auto span = [](const std::vector<float>& samples) -> FjFloatSpan {
                return {samples.data(), samples.size()};
            };
            const FjSpectralWhiteInput input{{tables.observer_xyz[0], tables.observer_xyz[1], tables.observer_xyz[2]}, span(out.curve.linear)};
            FjSpectralWhite white{};
            FjSpectralWhiteFailure failure{};
            std::array<char, 256> bytes{};
            FjErrorBuffer errorWhite{bytes.data(), bytes.size(), 0};
            require(fj_test_spectral_white(&input, &white, &failure, &errorWhite).category == FJ_STATUS_SUCCESS, "direct core scanner illuminant fixture");
            out.normalization = white.normalization;
            std::copy_n(white.white_xy, 2, out.whiteXY);
            out.hash = tables.illuminant_hash;
            require(out.hash != 0, "actual completed scanner illuminant hash");
            return out;
        }
        std::string viewing_illuminant(const FjAssets* assets, const FjPreparedStateView& prepared) {
            require(assets != nullptr, "scanner illuminant source has process Rust assets");
            std::string result;
            if ((prepared.route & 1u) != 0) {
                FjPrintProfile* owner = nullptr;
                expect_status(fj_test_print_profile_acquire(assets, prepared.print_profile_key, &owner, nullptr), FJ_STATUS_SUCCESS);
                FjPrintProfileView view{};
                expect_status(fj_test_print_profile_view(owner, &view, nullptr), FJ_STATUS_SUCCESS);
                require(view.asset_token == prepared.print_profile_asset_version, "scanner illuminant print source token matches held state");
                result.assign(view.viewing_illuminant.label.data, view.viewing_illuminant.label.count);
                expect_status(fj_test_print_profile_release(owner, nullptr), FJ_STATUS_SUCCESS);
            } else {
                FjFilmProfile* owner = nullptr;
                expect_status(fj_test_assets_film_acquire(assets, prepared.film_profile_key, &owner, nullptr), FJ_STATUS_SUCCESS);
                FjFilmProfileView view{};
                expect_status(fj_test_assets_film_view(owner, &view, nullptr), FJ_STATUS_SUCCESS);
                require(view.asset_token == prepared.film_profile_asset_version, "scanner illuminant film source token matches held state");
                result.assign(view.viewing_illuminant.label.data, view.viewing_illuminant.label.count);
                expect_status(fj_test_assets_film_release(owner, nullptr), FJ_STATUS_SUCCESS);
            }
            return result;
        }
        std::vector<Json> helper_inputs() {
            std::vector<Json> out;
            const std::array<std::array<float, 3>, 8> rgb{{{0, 0, 0}, {.184f, .184f, .184f}, {1, 1, 1}, {2, -.25f, .5f}, {-.2f, -.1f, -.3f}, {-0.0f, .05f, -.05f}, {.1f, 1e-30f, .2f}, {.05f, -.5f, 1.5f}}};
            const std::array<std::array<float, 3>, 4> refs{{{.95047f, 1.0f, 1.08883f}, {.96429567f, 1.0f, .8251046f}, {0, 0, 0}, {.8f, 0, 1.2f}}};
            for (std::size_t i = 0; i < rgb.size(); ++i)
                for (std::size_t j = 0; j < refs.size(); ++j)
                    out.push_back({{"id", "helper-rgb-" + std::to_string(i) + "-white-" + std::to_string(j)}, {"rgb", bits(rgb[i])}, {"reference_white", bits(refs[j])}});
            return out;
        }
        Json helper_result(const Json& input, const FjPreparedSpectralView& tables, const std::array<float, 9>& sInv, bool hanatos, FjFloatSpan hanatosSamples) {
            const auto rgb = triplet(input.at("rgb")), white = triplet(input.at("reference_white"));
            FjSpectrumFixture out{};
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            if (hanatos) {
                const FjHanatosSpectrumInput request{{rgb[0], rgb[1], rgb[2]}, {white[0], white[1], white[2]}, hanatosSamples};
                expect_status(fj_test_exposure_hanatos_spectrum(&request, &out, &error), FJ_STATUS_SUCCESS);
            } else {
                FjTablesSpectrumInput request{{rgb[0], rgb[1], rgb[2]}, {white[0], white[1], white[2]}, {}, tables.weighted_xyz[0], tables.weighted_xyz[1], tables.weighted_xyz[2]};
                std::copy(sInv.begin(), sInv.end(), request.s_inverse);
                expect_status(fj_test_exposure_tables_spectrum(&request, &out, &error), FJ_STATUS_SUCCESS);
            }
            return {{"pre_xyz", bits(out.pre_xyz)}, {"consumer_white", bits(out.consumer_white)}, {"post_adapt_xyz", bits(out.post_adapt_xyz)}, {"spectrum", bits(out.spectrum)}};
        }

    } // namespace Cat02Fixtures
    FjStatus facade_call(std::uint32_t operation) {
        constexpr std::array<float, 3> white{0.95045593f, 1.0f, 1.08905775f};
        constexpr std::array<float, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
        const FjInputColorConversion input{FJ_INPUT_SRGB_REC709, 1, 1, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}};
        FjInputColorMatrices matrices{};
        std::array<float, 9> matrix{};
        std::array<float, 3> adapted{}, rgb{};
        switch (operation) {
            case FJ_TEST_CAT16_MATRIX:
                return fj_test_cat16_matrix(white.data(), white.data(), matrix.data());
            case FJ_TEST_CAT16_ADAPT:
                return fj_test_adapt_cat16(white.data(), white.data(), white.data(), adapted.data());
            case FJ_TEST_CAT02_MATRIX:
                return fj_test_cat02_matrix(white.data(), white.data(), matrix.data());
            case FJ_TEST_CAT02_ADAPT:
                return fj_test_adapt_cat02(white.data(), white.data(), white.data(), adapted.data());
            case FJ_TEST_INPUT_MATRICES:
                return fj_test_input_matrices(FJ_INPUT_DWG, &matrices);
            case FJ_TEST_INPUT_TO_DWG:
                return fj_test_input_to_dwg(&input, white.data(), 1, rgb.data(), adapted.data());
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                return fj_test_input_to_linear_srgb(&input, white.data(), identity.data(), rgb.data(), adapted.data());
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                return fj_test_linear_srgb_to_xyz(white.data(), adapted.data());
            case FJ_TEST_DWG_TO_XYZ:
                return fj_test_dwg_to_xyz(white.data(), adapted.data());
            case FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ:
                return fj_test_project_linear_rgb_to_xyz(white.data(), identity.data(), identity.data(), adapted.data());
            default:
                throw std::runtime_error("unknown color operation");
        }
    }
    struct NullColorCase {
        std::uint32_t operation;
        int slot;
    };
    int color_pointer_count(std::uint32_t operation) {
        switch (operation) {
            case FJ_TEST_CAT16_MATRIX:
            case FJ_TEST_CAT02_MATRIX:
                return 3;
            case FJ_TEST_CAT16_ADAPT:
            case FJ_TEST_CAT02_ADAPT:
            case FJ_TEST_INPUT_TO_DWG:
            case FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ:
                return 4;
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                return 5;
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
            case FJ_TEST_DWG_TO_XYZ:
                return 2;
            case FJ_TEST_INPUT_MATRICES:
                return 1;
            default:
                throw std::runtime_error("unknown pointer operation");
        }
    }
    FjStatus null_color_call(NullColorCase selected) {
        constexpr std::array<float, 3> white{0.95045593f, 1.0f, 1.08905775f};
        constexpr std::array<float, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
        const FjInputColorConversion input{FJ_INPUT_SRGB_REC709, 1, 1, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}};
        FjInputColorMatrices matrices{};
        std::array<float, 9> out{};
        std::array<float, 3> rgb{};
        const auto pointer = [&](int slot) {
            return selected.slot == slot ? nullptr : white.data();
        };
        const auto output = [&](int slot) {
            return selected.slot == slot ? nullptr : out.data();
        };
        const auto* const conversion = selected.slot == 0 ? nullptr : &input;
        switch (selected.operation) {
            case FJ_TEST_CAT16_MATRIX:
                return fj_test_boundary_cat16_matrix(pointer(0), pointer(1), output(2));
            case FJ_TEST_CAT02_MATRIX:
                return fj_test_boundary_cat02_matrix(pointer(0), pointer(1), output(2));
            case FJ_TEST_CAT16_ADAPT:
                return fj_test_boundary_adapt_cat16(pointer(0), pointer(1), pointer(2), output(3));
            case FJ_TEST_CAT02_ADAPT:
                return fj_test_adapt_cat02(pointer(0), pointer(1), pointer(2), output(3));
            case FJ_TEST_INPUT_MATRICES:
                return fj_test_boundary_input_matrices(FJ_INPUT_DWG, selected.slot == 0 ? nullptr : &matrices);
            case FJ_TEST_INPUT_TO_DWG:
                return fj_test_input_to_dwg(conversion, pointer(1), 1, selected.slot == 2 ? nullptr : rgb.data(), output(3));
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                return fj_test_input_to_linear_srgb(conversion, pointer(1), selected.slot == 2 ? nullptr : identity.data(), selected.slot == 3 ? nullptr : rgb.data(), output(4));
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                return fj_test_boundary_linear_srgb_to_xyz(pointer(0), output(1));
            case FJ_TEST_DWG_TO_XYZ:
                return fj_test_dwg_to_xyz(pointer(0), output(1));
            case FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ:
                return fj_test_boundary_project_linear_rgb_to_xyz(pointer(0), selected.slot == 1 ? nullptr : identity.data(), selected.slot == 2 ? nullptr : identity.data(), output(3));
            default:
                throw std::runtime_error("unknown null operation");
        }
    }
    void color_fault_tests() {
        FaultScope cleanup;
        constexpr std::array<std::uint32_t, 10> operations{FJ_TEST_CAT16_MATRIX, FJ_TEST_CAT16_ADAPT, FJ_TEST_CAT02_MATRIX, FJ_TEST_CAT02_ADAPT, FJ_TEST_INPUT_MATRICES, FJ_TEST_INPUT_TO_DWG, FJ_TEST_INPUT_TO_LINEAR_SRGB, FJ_TEST_LINEAR_SRGB_TO_XYZ, FJ_TEST_DWG_TO_XYZ, FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ};
        std::vector<std::string> fault_ids;
        for (const auto operation : operations)
            for (const auto fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC})
                fault_ids.emplace_back("operation-" + std::to_string(operation) + "/fault-" + std::to_string(fault));
        JuicerTests::Cohort operation_faults("color.fault.operation_fault", fault_ids);
        std::vector<std::string> unrelated_ids, null_ids, invalid_ids, replacement_ids;
        for (const auto operation : operations) {
            for (const auto fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                const auto prefix = std::to_string(operation) + "/" + std::to_string(fault) + "/";
                for (const auto unrelated : operations)
                    if (unrelated != operation)
                        unrelated_ids.push_back(prefix + std::to_string(unrelated));
                for (int slot = 0; slot < color_pointer_count(operation); ++slot)
                    null_ids.push_back(prefix + std::to_string(slot));
            }
            for (int index = 0; index < 5; ++index)
                invalid_ids.push_back(std::to_string(operation) + "/" + std::to_string(index));
            for (const auto replacement : operations)
                replacement_ids.push_back(std::to_string(operation) + "/" + std::to_string(replacement));
        }
        JuicerTests::Cohort unrelated_rows("color.fault.unrelated", unrelated_ids);
        JuicerTests::Cohort null_rows("color.fault.null", null_ids);
        JuicerTests::Cohort invalid_rows("color.fault.invalid", invalid_ids);
        JuicerTests::Cohort replacement_rows("color.fault.replacement", replacement_ids);
        for (const auto operation : operations) {
            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                auto row_event = operation_faults.case_event("operation-" + std::to_string(operation) + "/fault-" + std::to_string(fault));
                expect_status(arm_color_fault(operation, 2, fault), FJ_STATUS_SUCCESS);
                for (const auto unrelated : operations) {
                    if (unrelated != operation) {
                        auto unrelated_event = unrelated_rows.case_event(std::to_string(operation) + "/" + std::to_string(fault) + "/" + std::to_string(unrelated));
                        expect_status(raw_call(unrelated), FJ_STATUS_SUCCESS);
                        unrelated_event.passed();
                    }
                }
                if (!color_facade_operation(operation)) {
                    expect_status(facade_call(operation), FJ_STATUS_SUCCESS);
                }
                const int pointers = color_pointer_count(operation);
                for (int slot = 0; slot < pointers; ++slot) {
                    auto null_event = null_rows.case_event(std::to_string(operation) + "/" + std::to_string(fault) + "/" + std::to_string(slot));
                    expect_status(null_color_call({.operation = operation, .slot = slot}), FJ_STATUS_UNSUPPORTED_INPUT);
                    null_event.passed();
                }
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                expect_status(raw_call(operation), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                expect_status(arm_color_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                FjStatus otherThread{};
                std::thread worker([&] {
                    otherThread = raw_call(operation);
                });
                worker.join();
                expect_status(otherThread, FJ_STATUS_SUCCESS);
                expect_status(raw_call(operation), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                row_event.passed();
            }
            std::size_t invalid_index = 0;
            for (const auto& invalid : std::array<std::array<std::uint32_t, 3>, 5>{{{0, 1, 1}, {FJ_TEST_OUTPUT_COLOR_FACADE + 1U, 1, 1}, {99, 1, 1}, {operation, 0, 1}, {operation, 1, 99}}}) {
                auto invalid_event = invalid_rows.case_event(std::to_string(operation) + "/" + std::to_string(invalid_index++));
                expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(arm_color_fault(invalid[0], invalid[1], invalid[2]), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                invalid_event.passed();
            }
            for (const auto replacement : operations) {
                auto replacement_event = replacement_rows.case_event(std::to_string(operation) + "/" + std::to_string(replacement));
                expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(arm_color_fault(replacement, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
                if (replacement != operation) {
                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                }
                expect_status(raw_call(replacement), FJ_STATUS_SUCCESS);
                expect_status(raw_call(replacement), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(replacement), FJ_STATUS_SUCCESS);
                replacement_event.passed();
            }
            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
        }
        std::puts("Color fault slot: ten operations, cross-family counting/replacement, facade/null independence and thread locality passed");
    }
    void cat02_leaf_tests(const Json& fixture) {
        bool scalarDistinct = false;
        for (const auto* group : {"leaf", "threshold"}) {
            for (const auto& row : fixture.at(group)) {
                JuicerTests::CaseEvent event("color.cat02_leaf", row.at("id").get<std::string>());
                const auto source = triplet(row.at("source"));
                const auto destination = triplet(row.at("destination"));
                const auto xyz = triplet(row.at("xyz"));
                const Spectral::ChromaticAdaptationWhites whites{source.data(), destination.data()};
                const auto& expected = row.at("expected");
                const std::string id = row.at("id").get<std::string>();
                const auto matrix = JuicerTestFixture::Color::cat02_matrix(whites);
                std::array<float, 3> adapted{};
                expect_status(fj_test_adapt_cat02(xyz.data(), whites.source, whites.destination, adapted.data()), FJ_STATUS_SUCCESS);
                require(same_values(bits(matrix), expected.at("matrix")), id + " production matrix bits/classification");
                require(same_values(bits(adapted), expected.at("adapted")), id + " production scalar bits/classification");
                std::array<float, 9> facadeMatrix{};
                std::array<float, 3> facadeAdapted{};
                expect_status(fj_test_cat02_matrix(source.data(), destination.data(), facadeMatrix.data()), FJ_STATUS_SUCCESS);
                expect_status(fj_test_adapt_cat02(xyz.data(), source.data(), destination.data(), facadeAdapted.data()), FJ_STATUS_SUCCESS);
                require(same_values(bits(facadeMatrix), expected.at("matrix")), id + " facade matrix bits/classification");
                require(same_values(bits(facadeAdapted), expected.at("adapted")), id + " facade scalar bits/classification");
                std::array<float, 3> multiplied{};
                multiplied = multiply_matrix(matrix.data(), xyz);
                scalarDistinct = scalarDistinct || !same_values(bits(multiplied), expected.at("adapted"));
                event.passed();
            }
        }
        require(scalarDistinct, "independent CAT02 captures distinguish scalar from matrix shortcut");
        JuicerTests::CaseEvent controls_event("color.cat02_leaf", "direction_abi_and_fault_controls");
        const auto selected = std::find_if(fixture.at("leaf").begin(), fixture.at("leaf").end(), [](const auto& row) {
            return row.at("id") == "d65-to-output-d50-xyz-0";
        });
        require(selected != fixture.at("leaf").end(), "CAT02 direction comparator case");
        const auto& expected = selected->at("expected").at("matrix");
        auto altered = expected;
        std::swap(altered[1], altered[3]);
        require(!same_values(altered, expected), "CAT02 transpose comparator rejection");
        altered = expected;
        altered[0] = altered[0].get<std::uint32_t>() ^ 1u;
        require(!same_values(altered, expected), "CAT02 one-bit comparator rejection");
        auto swapped = *selected;
        std::swap(swapped["source"], swapped["destination"]);
        require(!same_values(Cat02Fixtures::leaf_result(swapped).at("matrix"), expected), "CAT02 white-direction comparator rejection");
        require(fj_test_cat02_abi_c() == 0, "real C11 CAT02 layout/status/null/clear/canary/alias/nonfinite/panic transport");
        color_fault_tests();
        native_color_fault_tests(FJ_TEST_CAT02_MATRIX, "CAT02 matrix preparation failed");
        controls_event.passed();
        std::printf("CAT02: %zu leaf and %zu strict-threshold cases passed independently through production and facade\n", fixture.at("leaf").size(), fixture.at("threshold").size());
    }
    template <typename Range>
    std::vector<std::uint32_t> retained_bits(const Range& values) {
        std::vector<std::uint32_t> out;
        out.reserve(std::size(values));
        for (float value : values) {
            out.push_back(std::bit_cast<std::uint32_t>(value));
        }
        return out;
    }
    template <typename Range>
    bool same_retained_bits(const Range& values, const std::vector<std::uint32_t>& expected) {
        return std::size(values) == expected.size() && std::equal(std::begin(values), std::end(values), expected.begin(), [](float value, std::uint32_t bits) {
                   return std::bit_cast<std::uint32_t>(value) == bits;
               });
    }
    void retained_comparison_tests() {
        const std::array<std::uint32_t, 8> authored{0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x7fc00001u, 0x7fc00002u};
        const std::vector<std::uint32_t> expected(authored.begin(), authored.end());
        std::array<float, 8> values{};
        std::transform(authored.begin(), authored.end(), values.begin(), [](std::uint32_t bits) {
            return std::bit_cast<float>(bits);
        });
        require(same_retained_bits(values, expected), "retained finite/zero/infinity/NaN payload bits exact");
        for (std::size_t i = 0; i < values.size(); ++i) {
            auto changed = values;
            changed[i] = std::bit_cast<float>(authored[i] ^ 1u);
            require(!same_retained_bits(changed, expected), "each retained sample bit change rejected");
        }
        auto shortExpected = expected;
        shortExpected.pop_back();
        require(!same_retained_bits(values, shortExpected), "retained size mismatch rejected");
        const std::array<float, 0> empty{};
        require(same_retained_bits(empty, {}) && !same_retained_bits(empty, expected), "absent retained samples compare only with empty snapshots");
    }
    struct RetainedValues {
        Json complete;
        Json exposureTables;
        std::vector<std::uint32_t> tcRgba;
        std::vector<std::uint32_t> printIlluminant;
        std::vector<std::uint32_t> boundary;
        std::array<std::uint64_t, 3> boundaryHashes{};
        std::uintptr_t boundaryEntryIdentity = 0;
        const float* boundaryBacking = nullptr;
    };
    RetainedValues retain_values(const JuicerTestSupport::PreparedStateHold& state) {
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjPreparedStateView view{};
        require(state.view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
        RetainedValues out;
        out.complete = Cat02Fixtures::complete_result(state);
        out.exposureTables = Cat02Fixtures::tables_result(view.exposure_tables);
        if (view.film_exposure.tc_lut_rgba.count != 0) {
            out.tcRgba.reserve(view.film_exposure.tc_lut_rgba.count);
            for (std::size_t i = 0; i < view.film_exposure.tc_lut_rgba.count; ++i) {
                out.tcRgba.push_back(std::bit_cast<std::uint32_t>(view.film_exposure.tc_lut_rgba.data[i]));
            }
        }
        if (view.print_source_illuminant.count != 0) {
            out.printIlluminant.reserve(view.print_source_illuminant.count);
            for (std::size_t i = 0; i < view.print_source_illuminant.count; ++i) {
                out.printIlluminant.push_back(std::bit_cast<std::uint32_t>(view.print_source_illuminant.data[i]));
            }
        }
        const auto& boundary = view.output_boundary;
        require(boundary.entry_identity != 0 && boundary.cmax.count == Gamut::kOutputBoundaryValueCount, "completed state boundary owner");
        out.boundary.reserve(boundary.cmax.count);
        for (std::size_t i = 0; i < boundary.cmax.count; ++i) {
            out.boundary.push_back(std::bit_cast<std::uint32_t>(boundary.cmax.data[i]));
        }
        out.boundaryHashes = {boundary.transform_hash, boundary.contract_hash, boundary.hash};
        out.boundaryEntryIdentity = boundary.entry_identity;
        out.boundaryBacking = boundary.cmax.data;
        return out;
    }
    void require_retained_values(const JuicerTestSupport::PreparedStateHold& state, const RetainedValues& expected) {
        const auto actual = retain_values(state);
        require(actual.complete == expected.complete && actual.exposureTables == expected.exposureTables && actual.tcRgba == expected.tcRgba && actual.printIlluminant == expected.printIlluminant && actual.boundary == expected.boundary && actual.boundaryHashes == expected.boundaryHashes && actual.boundaryEntryIdentity == expected.boundaryEntryIdentity && actual.boundaryBacking == expected.boundaryBacking, "retained complete route/color/tables/TC/boundary values remain exact");
    }
    RetainedValues retain_values(const JuicerTestSupport::PublicationAdmission& result) {
        require(result.state != nullptr, "retained complete prepared-state hold");
        return retain_values(*result.state);
    }
    void require_retained_values(const JuicerTestSupport::PublicationAdmission& result, const RetainedValues& expected) {
        require(result.state != nullptr, "retained complete prepared-state hold");
        require_retained_values(*result.state, expected);
    }
    void cat02_admission_tests(const FjAssets* assets) {
        std::size_t failures = 0;
        std::size_t supersessions = 0;
        JuicerTests::Cohort supersession_rows("color.cat02-supersession", 96u);
        JuicerTests::Cohort fault_rows("color.cat02-admission-faults", 32u);
        std::size_t supersession_index = 0;
        std::size_t fault_index = 0;
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                const auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                auto changed = controls;
                changed.values.camera_exposure_compensation_ev = 1.0;
                const std::uint32_t calls = method == 1 ? 2u : 1u;
                for (std::uint32_t index = 1; index <= calls; ++index) {
                    for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                        auto event = fault_rows.case_event(std::to_string(fault_index++));
                        FaultScope cleanup;
                        JuicerTestSupport::InstanceFixture state;
                        JuicerTestSupport::capture(state, controls);
                        const auto old = JuicerTestSupport::admit(assets, state);
                        require(admitted(old), "initial CAT02 state admitted");
                        const auto old_values = retain_values(*old.state);
                        const auto old_build = old.state->build();
                        const auto old_identity = old.state->input_identity();
                        const auto before = JuicerTestSupport::PublicationObserver::metrics(state);
                        JuicerTestSupport::capture(state, changed);
                        expect_status(arm_color_fault(FJ_TEST_CAT02_MATRIX, index, fault), FJ_STATUS_SUCCESS);
                        const auto failed = JuicerTestSupport::admit(assets, state);
                        if (fault == FJ_TEST_COLOR_PANIC) {
                            require_poisoned_admission(assets, failed, state, old.state, old_build);
                            for (std::uint32_t call = 0; call < index; ++call)
                                expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_SUCCESS);
                            require_retained_values(*old.state, old_values);
                            ++failures;

                            JuicerTestSupport::InstanceFixture recovery;
                            JuicerTestSupport::capture(recovery, controls);
                            const auto recovery_a = JuicerTestSupport::admit(assets, recovery);
                            require(admitted(recovery_a), "independent CAT02 recovery instance admits A");
                            JuicerTestSupport::capture(recovery, changed);
                            const auto recovery_b = JuicerTestSupport::admit(assets, recovery);
                            require(admitted(recovery_b) && recovery_b.state->build() == recovery_a.state->build() + 1,
                                    "independent CAT02 recovery instance prepares B");

                            for (int mode = 0; mode < 3; ++mode) {
                                auto supersession_event = supersession_rows.case_event(std::to_string(supersession_index++));
                                JuicerTestSupport::InstanceFixture superseded;
                                JuicerTestSupport::capture(superseded, changed);
                                const auto held_b = JuicerTestSupport::admit(assets, superseded);
                                require(admitted(held_b), "independent CAT02 supersession instance publishes B");
                                const auto held_build = held_b.state->build();
                                const auto held_b_values = retain_values(*held_b.state);
                                JuicerTestSupport::capture(superseded, controls);
                                Supersession selection{&superseded, changed, mode};
                                fj_test_instance_prepare_hooks(supersede_capture, nullptr, &selection);
                                expect_status(arm_color_fault(FJ_TEST_CAT02_MATRIX, index, fault), FJ_STATUS_SUCCESS);
                                const auto obsolete = JuicerTestSupport::admit(assets, superseded);
                                require(selection.fired && !selection.failed, "obsolete CAT02 panic completed newer capture");
                                require_poisoned_admission(assets, obsolete, superseded, held_b.state, held_build);
                                require_retained_values(*held_b.state, held_b_values);
                                for (std::uint32_t call = 0; call < index; ++call)
                                    expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_SUCCESS);
                                fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                                expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                                ++supersessions;
                                supersession_event.passed();
                            }
                            event.passed();
                            continue;
                        }
                        require_preparation_failure(failed, fault);
                        for (std::uint32_t call = 0; call < index; ++call)
                            expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_SUCCESS);
                        const auto after = JuicerTestSupport::PublicationObserver::metrics(state);
                        require(after.publishedIdentity == before.publishedIdentity && after.build == before.build && after.inputIdentity == JuicerTestSupport::hash_input_identity(changed), "failed CAT02 candidate keeps private A and B pending identity");
                        require_retained_values(*old.state, old_values);
                        ++failures;

                        JuicerTestSupport::capture(state, controls);
                        const auto reused = JuicerTestSupport::admit(assets, state);
                        require(admitted(reused) && reused.state->input_identity() == old_identity && reused.state->build() == old_build, "failed CAT02 rebuild retains reusable A");
                        expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                        JuicerTestSupport::capture(state, changed);
                        const auto recovered = JuicerTestSupport::admit(assets, state);
                        require(admitted(recovered) && recovered.state->build() == old_build + 1 && recovered.state->input_identity() == JuicerTestSupport::hash_input_identity(changed), "CAT02 recovery commits one new publication");
                        require_retained_values(*old.state, old_values);
                        const auto recovered_values = retain_values(*recovered.state);

                        for (int mode = 0; mode < 3; ++mode) {
                            auto supersession_event = supersession_rows.case_event(std::to_string(supersession_index++));
                            Supersession selection{&state, changed, mode};
                            JuicerTestSupport::capture(state, controls);
                            fj_test_instance_prepare_hooks(nullptr, supersede_capture, &selection);
                            expect_status(arm_color_fault(FJ_TEST_CAT02_MATRIX, index, fault), FJ_STATUS_SUCCESS);
                            const auto current = JuicerTestSupport::admit(assets, state);
                            require(selection.fired && !selection.failed, "CAT02 preparation hook installed current Rust disposition");
                            if (mode == 0) {
                                require(admitted(current) && current.state->input_identity() == JuicerTestSupport::hash_input_identity(changed) && current.state->build() == recovered.state->build(), "obsolete CAT02 failure retries and reuses current B");
                            } else {
                                require((mode == 1 ? current.kind == FJ_ADMISSION_CAPTURE_FAILED : current.kind == FJ_ADMISSION_CAPTURING) && !current.state, "CAT02 invalid or in-progress capture blocks admission");
                            }
                            require(JuicerTestSupport::PublicationObserver::metrics(state).build == recovered.state->build(), "obsolete CAT02 result does not advance committed build");
                            require_retained_values(*old.state, old_values);
                            require_retained_values(*recovered.state, recovered_values);
                            for (std::uint32_t call = 0; call < index; ++call)
                                expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_SUCCESS);
                            fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                            ++supersessions;
                            supersession_event.passed();
                        }
                        event.passed();
                    }
                }
            }
        }
        require(failures == 32 && supersessions == 96, "CAT02 fault/recovery and supersession row memberships");
        std::printf("CAT02 admission: %zu current failures/recoveries and %zu actual-owner supersession witnesses passed on four routes\n", failures, supersessions);
    }
    template <typename Function>
    void require_color_facade_failure(std::uint32_t operation, std::uint32_t fault, Function&& function) {
        FaultScope cleanup;
        expect_status(arm_color_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
        expect_status(function(), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
        require(fj_test_color_facade_fault_consumed(operation, fault) == 1, "direct current color-facade consumption before cleanup");
        expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
    }
    void require_spectrum_fixture_failure(const FjPreparedSpectralView& tables, const std::array<float, 9>& inverse, std::uint32_t operation, std::uint32_t fault, FjFloatSpan hanatosSamples) {
        const float rgb[3]{2.0f, -.25f, .5f};
        const FjHanatosSpectrumInput hanatos{{rgb[0], rgb[1], rgb[2]}, {tables.white_xyz[0], tables.white_xyz[1], tables.white_xyz[2]}, hanatosSamples};
        FjTablesSpectrumInput request{{rgb[0], rgb[1], rgb[2]}, {tables.white_xyz[0], tables.white_xyz[1], tables.white_xyz[2]}, {}, tables.weighted_xyz[0], tables.weighted_xyz[1], tables.weighted_xyz[2]};
        std::copy(inverse.begin(), inverse.end(), request.s_inverse);
        FjSpectrumFixture out{};
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        expect_status(fj_test_exposure_fixture_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
        expect_status(operation == 1 ? fj_test_exposure_hanatos_spectrum(&hanatos, &out, &error) : fj_test_exposure_tables_spectrum(&request, &out, &error), fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
        require(fj_test_exposure_fixture_fault_consumed(operation, fault) == 1, "closed spectrum operation consumed before cleanup");
        fj_test_exposure_fixture_clear_fault();
    }
    void cat02_helper_tests(const FjAssets* assets, const Json& fixture) {
        auto product = Cat02Fixtures::prepare(assets, Cat02Fixtures::snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Mallett2019));
        require(assets != nullptr, "Hanatos helper source has process Rust assets");
        JuicerTestSupport::HanatosSource hanatosSource(assets);
        const FjFloatSpan hanatosSamples = hanatosSource.view().samples;
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjPreparedStateView view{};
        require(product.view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
        const auto& tables = view.exposure_tables;
        const std::array<float, 9> sInv{view.s_inverse[0], view.s_inverse[1], view.s_inverse[2], view.s_inverse[3], view.s_inverse[4], view.s_inverse[5], view.s_inverse[6], view.s_inverse[7], view.s_inverse[8]};
        require(Cat02Fixtures::tables_result(tables) == fixture.at("helper_tables") && bits(sInv) == fixture.at("helper_s_inv"), "independent canonical helper tables/S inverse");
        require(Json{{"K", Spectral::kReferenceShape.K}, {"dwg_white", bits(Spectral::gDWG_WhitePoint_XYZ)}} == fixture.at("helper_hanatos_axes"), "independent retained Hanatos helper axes/white");
        const auto cases = Cat02Fixtures::helper_inputs();
        require(cases.size() == fixture.at("helpers").size(), "retained helper case membership");
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& input = cases[i];
            const auto& row = fixture.at("helpers").at(i);
            JuicerTests::CaseEvent event("color.cat02_helper", row.at("id").get<std::string>());
            require(input.at("id") == row.at("id") && input.at("rgb") == row.at("rgb") && input.at("reference_white") == row.at("reference_white"), "frozen retained helper inputs");
            require(Cat02Fixtures::helper_result(input, tables, sInv, true, hanatosSamples) == row.at("hanatos_expected"), input.at("id").get<std::string>() + " signed Hanatos sanitation/spectrum");
            require(Cat02Fixtures::helper_result(input, tables, sInv, false, hanatosSamples) == row.at("tables_expected"), input.at("id").get<std::string>() + " tables sanitation/spectrum");
            event.passed();
        }
        const std::array<float, 3> rgb{2.0f, -0.25f, 0.5f};
        const auto held = retain_values(product);
        for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
            JuicerTests::CaseEvent event("color.cat02_helper_fault", std::to_string(fault));
            std::array<float, 3> adapted{};
            require_color_facade_failure(FJ_TEST_CAT02_ADAPT, fault, [&] {
                return fj_test_adapt_cat02(rgb.data(), Spectral::gDWG_WhitePoint_XYZ, tables.white_xyz, adapted.data());
            });
            require_spectrum_fixture_failure(tables, sInv, 1, fault, hanatosSamples);
            require_spectrum_fixture_failure(tables, sInv, 2, fault, hanatosSamples);
            require_retained_values(product, held);
            event.passed();
        }
        std::printf("CAT02 retained helpers: %zu independent pre/post sanitation/spectrum cases per helper and both typed failure categories passed\n", cases.size());
    }
    void cat02_control_tests(const FjAssets* assets) {
        JuicerTests::Cohort cat02_control_routes_rows("color.cat02-control-routes", 12u);
        std::size_t cat02_control_routes_index = 0;
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                auto cat02_control_routes_event = cat02_control_routes_rows.case_event(std::to_string(cat02_control_routes_index++));
                const auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                auto product = Cat02Fixtures::prepare(assets, controls);
                std::array<char, 2048> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                FjPreparedStateView productView{};
                require(product.view(productView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                const auto productHeld = retain_values(product);
                auto decoding = controls;
                decoding.values.input_cctf_decoding = 1;
                auto decoded = Cat02Fixtures::prepare(assets, decoding);
                FjPreparedStateView decodedView{};
                require(decoded.view(decodedView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                require(bits(productView.film_exposure.input_xyz_adapt, 9) == bits(decodedView.film_exposure.input_xyz_adapt, 9) && productView.film_exposure.hash != decodedView.film_exposure.hash && productView.recipe_hash != decodedView.recipe_hash, "input CCTF decoding keeps adaptation and changes film identity");
                auto input = controls;
                input.values.input_color_space = Spectral::input_color_space_to_index(Spectral::InputColorSpace::ACES2065_1);
                auto inputChanged = Cat02Fixtures::prepare(assets, input);
                FjPreparedStateView inputChangedView{};
                require(inputChanged.view(inputChangedView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                require(bits(productView.input_policy.nominal_white_xyz, 3) != bits(inputChangedView.input_policy.nominal_white_xyz, 3) && bits(productView.film_exposure.input_xyz_adapt, 9) != bits(inputChangedView.film_exposure.input_xyz_adapt, 9) && productView.film_exposure.hash != inputChangedView.film_exposure.hash && productView.recipe_hash != inputChangedView.recipe_hash, "input space/white changes contributing matrix and film identity");
                require(Cat02Fixtures::color_result(productView.scanner_color) == Cat02Fixtures::color_result(inputChangedView.scanner_color), "input space leaves selected scanner color values/identity unchanged");
                auto output = controls;
                output.values.output_color_space = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
                auto outputChanged = Cat02Fixtures::prepare(assets, output);
                FjPreparedStateView outputChangedView{};
                require(outputChanged.view(outputChangedView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                require(bits(productView.scanner_color.cat02, 9) != bits(outputChangedView.scanner_color.cat02, 9) && productView.scanner_color.hash != outputChangedView.scanner_color.hash && productView.recipe_hash != outputChangedView.recipe_hash, "output space/white changes scanner matrix/color/final identity");
                require(productView.film_exposure.hash == outputChangedView.film_exposure.hash && bits(productView.film_exposure.input_xyz_adapt, 9) == bits(outputChangedView.film_exposure.input_xyz_adapt, 9), "output space preserves film identity/matrix");
                auto encoding = controls;
                encoding.values.output_cctf_encoding = 0;
                auto encodedOff = Cat02Fixtures::prepare(assets, encoding);
                FjPreparedStateView encodedOffView{};
                require(encodedOff.view(encodedOffView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                require(bits(productView.film_exposure.input_xyz_adapt, 9) == bits(encodedOffView.film_exposure.input_xyz_adapt, 9) && productView.film_exposure.hash == encodedOffView.film_exposure.hash && productView.film_exposure.tc_lut_hash == encodedOffView.film_exposure.tc_lut_hash, "output CCTF-only changes preserve film/TC family");
                require(productView.output_boundary.entry_identity != 0 && productView.output_boundary.entry_identity == encodedOffView.output_boundary.entry_identity && productView.output_boundary.cmax.data == encodedOffView.output_boundary.cmax.data, "output CCTF-only reuses actual boundary table owner");
                require(Cat02Fixtures::tables_result(productView.scanner_tables) == Cat02Fixtures::tables_result(encodedOffView.scanner_tables) && Cat02Fixtures::scanner_descriptor(productView.scanner_lut) == Cat02Fixtures::scanner_descriptor(encodedOffView.scanner_lut), "output CCTF-only preserves complete scanner tables/descriptor");
                require(bits(productView.scanner_color.cat02, 9) == bits(encodedOffView.scanner_color.cat02, 9) && productView.scanner_color.hash != encodedOffView.scanner_color.hash && productView.scanner_recipe_hash != encodedOffView.scanner_recipe_hash && productView.recipe_hash != encodedOffView.recipe_hash, "output CCTF-only changes identities at color/output owners");
                JuicerTestSupport::InstanceFixture state;
                JuicerTestSupport::capture(state, controls);
                const auto old = JuicerTestSupport::admit(assets, state);
                require(admitted(old), "unchanged CAT02 admission baseline");
                const auto counter = old.state->build();
                const auto identity = old.state->input_identity();
                const auto held = retain_values(old);
                FaultScope cleanup;
                expect_status(arm_color_fault(FJ_TEST_CAT02_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
                const auto repeated = JuicerTestSupport::admit(assets, state);
                require(admitted(repeated) && repeated.state->input_identity() == identity && repeated.state->build() == counter, "unchanged CAT02 admission reuses exact publication");
                expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
                require_retained_values(old, held);
                require_retained_values(product, productHeld);
                auto rebuilt = Cat02Fixtures::prepare(assets, controls);
                FjPreparedStateView rebuiltView{};
                require(rebuilt.view(rebuiltView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                const auto oldView = prepared_view(*old.state);
                require(rebuiltView.output_boundary.entry_identity != oldView.output_boundary.entry_identity && rebuiltView.output_boundary.cmax.data != oldView.output_boundary.cmax.data && same_retained_bits(std::span<const float>(rebuiltView.output_boundary.cmax.data, rebuiltView.output_boundary.cmax.count), held.boundary), "cache release makes a new equal table without mutating admitted hold");
                JuicerTestSupport::capture(state, output);
                const auto replacement = JuicerTestSupport::admit(assets, state);
                require(admitted(replacement) && replacement.state->input_identity() == JuicerTestSupport::hash_input_identity(output) && replacement.state->build() == counter + 1, "output-space state replacement publishes complete owner/hash/counter");
                require_retained_values(old, held);

                cat02_control_routes_event.passed();
            }
        }
    }
    void output_color_admission_tests(const FjAssets* assets) {
        JuicerTests::Cohort output_admission_routes_rows("color.output-admission-routes", 4u);
        std::size_t output_admission_routes_index = 0;
        for (int route = 0; route < 4; ++route) {
            auto output_admission_routes_event = output_admission_routes_rows.case_event(std::to_string(output_admission_routes_index++));
            const auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019);
            auto changed = controls;
            changed.values.output_cctf_encoding = 0;
            std::uint32_t visits = 0;
            std::uint32_t completed_probes = 0;
            std::unique_ptr<JuicerTestSupport::InstanceFixture> completed_instance;
            JuicerTestSupport::PublicationAdmission completed;
            for (std::uint32_t index = 1; index <= 16; ++index) {
                FaultScope cleanup;
                auto instance = std::make_unique<JuicerTestSupport::InstanceFixture>();
                JuicerTestSupport::capture(*instance, controls);
                const auto old = JuicerTestSupport::admit(assets, *instance);
                require(admitted(old), "output fault probe baseline admitted");
                const auto old_build = old.state->build();
                JuicerTestSupport::capture(*instance, changed);
                expect_status(arm_color_fault(FJ_TEST_OUTPUT_COLOR, index, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                auto result = JuicerTestSupport::admit(assets, *instance);
                if (result.kind == FJ_ADMISSION_POISONED) {
                    require_poisoned_admission(assets, result, *instance, old.state, old_build);
                    require(JuicerTestSupport::published_view(old).recipe_hash != 0, "output panic preserves admitted old hold");
                    for (std::uint32_t call = 0; call < index; ++call) {
                        FjOutputColorPreparation raw{};
                        expect_status(fj_test_boundary_output_color(0, &raw), FJ_STATUS_SUCCESS);
                    }
                    visits = index - 1;
                    ++completed_probes;
                    continue;
                }
                require(admitted(result) && result.state->build() == old_build + 1, "first unfired output index admits the completed B state");
                visits = index - 1;
                ++completed_probes;
                completed = std::move(result);
                completed_instance = std::move(instance);
                break;
            }
            require(completed_instance && completed.state && completed_probes == visits + 1, "sequential output probes complete through the first unfired index");
            require(visits > 0 && visits < 16, "bounded actual output preparation visit count");
            std::printf("Output color route=%d rebuild visits=%u\n", route, visits);
            FaultScope cleanup;
            expect_status(arm_color_fault(FJ_TEST_OUTPUT_COLOR, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            const auto repeated = JuicerTestSupport::admit(assets, *completed_instance);
            require(admitted(repeated) && repeated.state->input_identity() == completed.state->input_identity() && repeated.state->build() == completed.state->build(), "warm output publication reuses owner without preparation");
            FjOutputColorPreparation raw{};
            expect_status(fj_test_boundary_output_color(0, &raw), FJ_STATUS_INTERNAL_FAILURE);

            output_admission_routes_event.passed();
        }
    }

    void require_scanner_encoding(const Scanner::ColorRuntime& color, const Json& catalogue) {
        const auto& entry = catalogue.at("entries").at(OutputEncoding::toIndex(color.encoding.colorSpace));
        const auto& e = color.preparedEncoding;
        const std::array coefficients{e.gamma, e.a, e.b, e.c, e.d, e.linear_cutoff};
        require(e.schema == 1 && e.cctf_kind == entry.at("kind") && bits(coefficients) == entry.at("cctf") && bits(e.dwg_to_output) == entry.at("dwg_to_output"), "every scanner prepared encoding field matches independent catalogue, also CCTF off");
    }
    void require_scanner_encoding(const FjScannerColor& color, const Json& catalogue) {
        const auto& entry = catalogue.at("entries").at(color.output_space);
        const auto& e = color.encoding;
        const std::array coefficients{e.gamma, e.a, e.b, e.c, e.d, e.linear_cutoff};
        require(e.schema == 1 && e.cctf_kind == entry.at("kind") && Cat02Fixtures::bits(coefficients) == entry.at("cctf") && Cat02Fixtures::bits(e.dwg_to_output) == entry.at("dwg_to_output"), "every held scanner encoding field matches independent catalogue, also CCTF off");
    }
    void cat02_preparation_tests(const FjAssets* assets, const Json& fixture, const std::filesystem::path& fixturePath) {
        std::ifstream catalogueInput{fixturePath.parent_path().parent_path() / "output-color/native.json"};
        Json catalogue;
        catalogueInput >> catalogue;
        cat02_admission_tests(assets);
        const auto cases = Cat02Fixtures::scenarios();
        require(cases.size() == fixture.at("recipes").size(), "frozen CAT02 recipe case membership");
        const auto witness = load_preset_witness(fixturePath, "cat02-manifest.json");
        bool wrongPresetRejected = false;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [id, controls] = cases[i];
            const auto& row = fixture.at("recipes").at(i);
            JuicerTests::CaseEvent event("color.cat02_recipe", row.at("id").get<std::string>());
            require(id == row.at("id").get<std::string>(), "frozen CAT02 recipe ordered case identity");
            const auto& settings = row.at("controls");
            require(settings.at("filmProfileKey") == controls.film_key && settings.at("printProfileKey") == controls.print_key && settings.at("scanRoute") == static_cast<int>(controls.values.scan_route) && settings.at("spectralUpsamplingMode") == controls.values.spectral_upsampling_mode && settings.at("inputColorSpace") == controls.values.input_color_space && settings.at("inputCctfDecoding") == controls.values.input_cctf_decoding && settings.at("outputColorSpace") == controls.values.output_color_space && settings.at("outputCctfEncoding") == controls.values.output_cctf_encoding && settings.at("cameraAutoExposureEnabled") == controls.values.camera_auto_exposure_enabled && settings.at("inputCompressionEnabled") == controls.values.input_compression_enabled && settings.at("cameraExposureCompensationEv").at("f64_bits") == std::bit_cast<std::uint64_t>(controls.values.camera_exposure_compensation_ev), "frozen CAT02 selected controls");
            auto product = Cat02Fixtures::prepare(assets, controls);
            FjPreparedStateView view{};
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            require(product.view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require_scanner_encoding(view.scanner_color, catalogue);
            const auto actual = Cat02Fixtures::complete_result(product);
            require(actual == row.at("expected"), id + " independent complete film/raw-midgray/scanner/tables/descriptors/identities: " + Json::diff(row.at("expected"), actual).dump());
            if (id == witness.at("recipe_id").get<std::string>()) {
                check_preset_witness(actual, row.at("expected"), witness);
                wrongPresetRejected = true;
            }
            if (i == 0) {
                auto changedIdentity = actual;
                changedIdentity["hashes"][0] = changedIdentity["hashes"][0].get<std::uint64_t>() ^ 1u;
                require(changedIdentity != row.at("expected"), id + " complete identity comparator rejection");
            }
            event.passed();
        }
        require(wrongPresetRejected, "different native-preset value/identity baselines reject incorrect expectation");
        std::array<Scanner::ScannerIlluminant, 4> illuminants;
        for (int route = 0; route < 4; ++route) {
            JuicerTests::CaseEvent event("color.cat02_scanner_illuminant", std::to_string(route));
            auto product = Cat02Fixtures::prepare(assets, Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019));
            illuminants[route] = Cat02Fixtures::scanner_illuminant(product);
            const auto& row = fixture.at("scanner_illuminants").at(route);
            const auto& illuminant = illuminants[route];
            FjPreparedStateView view{};
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            require(product.view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(row.at("route") == route && row.at("viewing_illuminant") == Cat02Fixtures::viewing_illuminant(assets, view) && row.at("white_xyz") == bits(illuminant.whiteXYZ) && row.at("white_xy") == bits(illuminant.whiteXY) && row.at("normalization") == bits(std::array{illuminant.normalization}) && row.at("hash") == illuminant.hash && row.at("samples") == bits(illuminant.curve.linear), "actual selected-route scanner illuminant exact fields/identity");
            event.passed();
        }
        for (const auto& row : fixture.at("scanner_cases")) {
            JuicerTests::CaseEvent event("color.cat02_scanner", row.at("id").get<std::string>());
            const OutputEncoding::Params encoding{static_cast<OutputEncoding::ColorSpace>(row.at("space").get<int>()), row.at("encoding").get<int>() != 0, true};
            const auto color = JuicerTestFixture::Scanner::build_color_runtime(static_cast<Scanner::ScannerMedium>(row.at("medium").get<int>()), illuminants[row.at("route").get<int>()], encoding, row.at("gamut_recipe_hash").get<std::uint64_t>());
            require(Cat02Fixtures::color_result(color) == row.at("expected"), row.at("id").get<std::string>() + " real scanner color exact arrays/encoding/hash");
            require_scanner_encoding(color, catalogue);
            event.passed();
        }
        for (const auto& row : fixture.at("scanner_invalid_cases")) {
            JuicerTests::CaseEvent event("color.cat02_scanner_invalid", std::to_string(row.at("medium").get<int>()));
            auto invalid = illuminants[0];
            invalid.hash = 0;
            FaultScope cleanup;
            expect_status(arm_color_fault(FJ_TEST_CAT02_MATRIX, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            const auto color = JuicerTestFixture::Scanner::build_color_runtime(static_cast<Scanner::ScannerMedium>(row.at("medium").get<int>()), invalid, {}, 0);
            require(Cat02Fixtures::color_result(color) == row.at("expected"), "invalid scanner illuminant keeps early zero-hash color result");
            expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_INTERNAL_FAILURE);
            event.passed();
        }
        JuicerTests::CaseEvent controls_event("color.cat02_preparation", "admission_helpers_controls");
        output_color_admission_tests(assets);
        cat02_helper_tests(assets, fixture);
        cat02_control_tests(assets);
        controls_event.passed();
        std::printf("CAT02 preparation: %zu independent recipes, %zu scanner cases, distinct controls and full retained arrays passed\n", cases.size(), fixture.at("scanner_cases").size());
    }

    namespace InputFixtures {
        Json controls_json(const FjDiffusionParams& p) {
            Json out;
            out["active"] = bool(p.active);
            out["family"] = static_cast<int>(p.family);
            out["strength"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.strength)}};
            out["spatialScale"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.spatial_scale)}};
            out["haloWarmth"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.halo_warmth)}};
            out["coreIntensity"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.core_intensity)}};
            out["coreSize"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.core_size)}};
            out["haloIntensity"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.halo_intensity)}};
            out["haloSize"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.halo_size)}};
            out["bloomIntensity"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.bloom_intensity)}};
            out["bloomSize"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.bloom_size)}};
            return out;
        }
        Json controls_json(const FjFilmDirParams& p) {
            Json out;
            out["active"] = bool(p.active);
            out["amount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.amount)}};
            out["inhibitionSameLayer"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.inhibition_same_layer)}};
            out["inhibitionInterlayer"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.inhibition_interlayer)}};
            out["diffusionSizeUm"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.diffusion_size_um)}};
            out["diffusionTailUm"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.diffusion_tail_um)}};
            out["diffusionTailWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.diffusion_tail_weight)}};
            out["gammaUseStock"] = bool(p.gamma_use_stock);
            out["gammaSameLayerRgb"] = Json{{"f32_bits", bits(p.gamma_same_layer_rgb)}};
            out["gammaInterlayerRToGb"] = Json{{"f32_bits", bits(p.gamma_interlayer_r_to_gb)}};
            out["gammaInterlayerGToRb"] = Json{{"f32_bits", bits(p.gamma_interlayer_g_to_rb)}};
            out["gammaInterlayerBToRg"] = Json{{"f32_bits", bits(p.gamma_interlayer_b_to_rg)}};
            out["langmuirDonorKRgb"] = Json{{"f32_bits", bits(p.langmuir_donor_k_rgb)}};
            out["langmuirReceiverKRgb"] = Json{{"f32_bits", bits(p.langmuir_receiver_k_rgb)}};
            return out;
        }
        Json controls_json(const FjVisualGrainParams& p) {
            Json out;
            out["active"] = bool(p.active);
            out["sublayersActive"] = bool(p.sublayers_active);
            out["particleAreaUm2"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.particle_area_um2)}};
            out["particleScaleCmy"] = Json{{"f32_bits", bits(p.particle_scale_cmy)}};
            out["particleScaleLayers"] = Json{{"f32_bits", bits(p.particle_scale_layers)}};
            out["visualParticleDensityMinCmy"] = Json{{"f32_bits", bits(p.particle_density_min_cmy)}};
            out["uniformityCmy"] = Json{{"f32_bits", bits(p.uniformity_cmy)}};
            out["correlationSigmaPx"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.correlation_sigma_px)}};
            out["dyeCloudBlurUm"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.dye_cloud_blur_um)}};
            out["microStructure"] = Json{{"f32_bits", bits(p.micro_structure)}};
            out["nSubLayers"] = p.sublayer_count;
            out["amplitude"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.amplitude)}};
            out["chromaMix"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.chroma_mix)}};
            out["chromaSharedWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.chroma_shared_weight)}};
            out["chromaIndependentWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.chroma_independent_weight)}};
            out["coarseWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.coarse_weight)}};
            out["midWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.mid_weight)}};
            out["sizeMixScale"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.size_mix_scale)}};
            out["clumpTemporalMix"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.clump_temporal_mix)}};
            out["clumpMorphPeriodSec"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.clump_morph_period_sec)}};
            out["debugView"] = p.debug_view;
            return out;
        }
        Json controls_json(const FjPreparationHalation& p) {
            Json out;
            out["active"] = bool(p.active);
            out["scatterAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.scatter_amount)}};
            out["scatterSpatialScale"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.scatter_spatial_scale)}};
            out["halationAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.halation_amount)}};
            out["halationSpatialScale"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.halation_spatial_scale)}};
            return out;
        }
        Json controls_json(const JuicerTestSupport::PreparationInput& p) {
            Json out;
            out["printExposure"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.print_exposure)}};
            out["printPreflashExposure"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.print_preflash_exposure)}};
            out["printGammaFactor"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.print_gamma_factor)}};
            out["preflashMFilterCc"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.preflash_m_filter_cc)}};
            out["preflashYFilterCc"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.preflash_y_filter_cc)}};
            out["printShadowCompensationFactor"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.print_shadow_compensation_factor)}};
            out["printShadowCompensationDensity"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.print_shadow_compensation_density)}};
            out["printShadowCompensationTransition"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.print_shadow_compensation_transition)}};
            out["glarePercent"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.glare_percent)}};
            out["glareRoughness"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.glare_roughness)}};
            out["glareBlurSigmaPx"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.glare_blur_sigma_px)}};
            out["cameraExposureCompensationEv"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.camera_exposure_compensation_ev)}};
            out["cameraFilmFormatLongEdgeMm"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.values.camera_film_format_long_edge_mm)}};
            out["filmGammaFactor"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.values.film_gamma_factor)}};
            out["scannerLensBlurSigmaPx"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.scanner_lens_blur_sigma_px)}};
            out["scannerBlackLevel"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.scanner_black_level)}};
            out["scannerWhiteLevel"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.scanner_white_level)}};
            out["gateWeaveAmount"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.values.gate_weave_amount)}};
            out["scannerUnsharpMask"] = Json{{"f64_bits", double_bits(p.values.scanner_unsharp_mask)}};
            out["printUiYmcCc"] = Json{{"f64_bits", double_bits(p.values.print_ui_ymc_cc)}};
            out["cameraFilterUV"] = Json{{"f64_bits", double_bits(p.values.camera_filter_uv)}};
            out["cameraFilterIR"] = Json{{"f64_bits", double_bits(p.values.camera_filter_ir)}};
            out["cameraDiffusion"] = controls_json(p.values.camera_diffusion);
            out["enlargerDiffusion"] = controls_json(p.values.enlarger_diffusion);
            out["dirCouplers"] = controls_json(p.values.dir_couplers);
            out["filmProfileKey"] = p.film_key;
            out["printProfileKey"] = p.print_key;
            out["spectralUpsamplingMode"] = p.values.spectral_upsampling_mode;
            out["refIll"] = p.values.ref_ill;
            out["enlIll"] = p.values.enl_ill;
            out["normalizePrintExposure"] = p.values.normalize_print_exposure;
            out["printExposureCompensation"] = p.values.print_exposure_compensation;
            out["inputColorSpace"] = p.values.input_color_space;
            out["inputCctfDecoding"] = p.values.input_cctf_decoding;
            out["inputCompressionEnabled"] = p.values.input_compression_enabled;
            out["hanatos2025AdaptationWindow"] = p.values.hanatos2025_adaptation_window;
            out["hanatos2025AdaptationSurface"] = p.values.hanatos2025_adaptation_surface;
            out["cameraAutoExposureEnabled"] = p.values.camera_auto_exposure_enabled;
            out["cameraMeteringMethod"] = p.values.camera_metering_method;
            out["scatterHalationControls"] = controls_json(p.values.scatter_halation);
            out["scannerBlackCorrection"] = p.values.scanner_black_correction;
            out["scannerWhiteCorrection"] = p.values.scanner_white_correction;
            out["scannerLutResolution"] = p.values.scanner_lut_resolution;
            out["outputColorSpace"] = p.values.output_color_space;
            out["outputCctfEncoding"] = p.values.output_cctf_encoding;
            out["outputGamutCompressionEnabled"] = p.values.output_gamut_compression_enabled;
            out["grainControls"] = controls_json(p.values.grain);
            out["filmDustAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.values.film_dust_amount)}};
            out["filmScratchAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.values.film_scratch_amount)}};
            out["gateDustAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.values.gate_dust_amount)}};
            out["gateScratchAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.values.gate_scratch_amount)}};
            out["scanRoute"] = static_cast<int>(p.values.scan_route);
            out["glareActive"] = bool(p.values.glare_active);
            out["cameraFilterOverride"] = bool(p.values.camera_filter_override);
            return out;
        }

        template <std::size_t N>
        std::array<float, N> floats(const Json& input) {
            std::array<float, N> out{};
            for (std::size_t i = 0; i < N; ++i) {
                out[i] = std::bit_cast<float>(input.at(i).get<std::uint32_t>());
            }
            return out;
        }
        Spectral::Mat3 matrix(const Json& input) {
            Spectral::Mat3 out{};
            const auto values = floats<9>(input);
            std::copy(values.begin(), values.end(), out.m);
            return out;
        }
        FjFilmExposure config(const Json& row) {
            FjFilmExposure out{};
            out.input_color_space = static_cast<std::uint32_t>(row.at("space").get<int>());
            out.flags = (row.at("decode").get<int>() != 0 ? FJ_FILM_DECODE_CCTF : 0u) |
                        (row.at("adapt").get<bool>() ? FJ_FILM_ADAPT_INPUT : 0u);
            const auto rgbToXyz = floats<9>(row.at("rgb_to_xyz"));
            const auto adapt = floats<9>(row.at("xyz_adapt"));
            const auto inverse = floats<9>(row.at("xyz_to_linear_srgb"));
            std::copy(rgbToXyz.begin(), rgbToXyz.end(), out.input_rgb_to_xyz);
            std::copy(adapt.begin(), adapt.end(), out.input_xyz_adapt);
            std::copy(inverse.begin(), inverse.end(), out.xyz_to_linear_srgb);
            return out;
        }
        FjInputColorConversion foreign(const FjFilmExposure& config) {
            FjInputColorConversion out{};
            out.input_space = config.input_color_space;
            out.decode_cctf = (config.flags & FJ_FILM_DECODE_CCTF) != 0;
            out.adapt_xyz = (config.flags & FJ_FILM_ADAPT_INPUT) != 0;
            std::copy_n(config.input_rgb_to_xyz, 9, out.rgb_to_xyz);
            std::copy_n(config.input_xyz_adapt, 9, out.xyz_adapt);
            return out;
        }
        std::vector<std::pair<std::string, JuicerTestSupport::PreparationInput>> scenarios() {
            std::vector<std::pair<std::string, JuicerTestSupport::PreparationInput>> out;
            for (int polarity = 0; polarity < 2; ++polarity) {
                for (int route = 0; route < 2; ++route) {
                    for (int space = 0; space < 4; ++space) {
                        for (int decode = 0; decode < 2; ++decode) {
                            for (int method = 0; method < 3; ++method) {
                                auto p = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                                p.values.input_color_space = space;
                                p.values.input_cctf_decoding = decode;
                                out.emplace_back("base-" + std::to_string(polarity) + "-" + std::to_string(route) + "-" + std::to_string(space) + "-" + std::to_string(decode) + "-" + std::to_string(method), p);
                            }
                        }
                    }
                }
            }
            for (int polarity = 0; polarity < 2; ++polarity) {
                for (int route = 0; route < 2; ++route) {
                    for (int method = 0; method < 3; ++method) {
                        auto p = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                        p.values.input_color_space = 3;
                        p.values.input_cctf_decoding = 1;
                        const auto id = std::to_string(polarity) + "-" + std::to_string(route) + "-" + std::to_string(method);
                        p.values.camera_auto_exposure_enabled = 1;
                        out.emplace_back("auto-" + id, p);
                        p.values.camera_auto_exposure_enabled = 0;
                        if (method != 1) {
                            p.values.input_compression_enabled = 1;
                            out.emplace_back("compression-" + id, p);
                            p.values.input_compression_enabled = 0;
                        }
                        p.values.output_cctf_encoding = 0;
                        out.emplace_back("output-off-" + id, p);
                        p.values.output_cctf_encoding = 1;
                        p.values.output_color_space = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
                        out.emplace_back("output-dci-" + id, p);
                    }
                }
            }
            return out;
        }
        Json complete_result(const JuicerTestSupport::PreparedStateHold& product) {
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            FjPreparedStateView view{};
            require(product.view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            const auto& raw = view.film_exposure;
            auto result = Cat02Fixtures::complete_result(product);
            result["config_xyz_to_linear_srgb"] = bits(raw.xyz_to_linear_srgb, 9);
            result["recipe_mallett_scale"] = bits(std::array{raw.mallett_green_midgray_scale});
            result["meter_scale"] = bits(std::array{view.film_reference.meter_scale});
            result["raw_flags"] = {raw.input_color_space, (raw.flags & FJ_FILM_DECODE_CCTF) != 0, raw.method, (raw.flags & FJ_FILM_AUTO_EXPOSURE) != 0, view.input_policy.input_compression_active != 0};
            result["sensitivity_hash"] = raw.sensitivity_hash;
            for (std::size_t i = 0; i < raw.sensitivity_rgb.count; i += 3) {
                result["sensitivity"].push_back(bits(raw.sensitivity_rgb.data + i, 3));
            }
            result["tc_source_hash"] = view.input_policy.tc_source_hash;
            result["tc_lut_size"] = raw.tc_lut_rgba.count;
            if (raw.tc_lut_rgba.count != 0) {
                for (std::size_t index : std::array<std::size_t, 5>{0, 768, 73728, 100000, 147452}) {
                    result["tc_samples"].push_back({{"index", index}, {"rgba", bits(raw.tc_lut_rgba.data + index, 4)}});
                }
            }
            return result;
        }
        Json publication_complete_result(const JuicerTestSupport::PreparedStateHold& product) {
            const auto view = prepared_view(product);
            auto result = complete_result(product);
            result["config_xyz_to_linear_srgb"] = bits(view.film_exposure.xyz_to_linear_srgb, 9);
            result["recipe_mallett_scale"] = bits(std::array{view.film_exposure.mallett_green_midgray_scale});
            result["meter_scale"] = bits(std::array{view.film_reference.meter_scale});
            result["raw_flags"] = {view.film_exposure.input_color_space,
                                   (view.film_exposure.flags & FJ_FILM_DECODE_CCTF) != 0,
                                   view.film_exposure.method,
                                   (view.film_exposure.flags & FJ_FILM_AUTO_EXPOSURE) != 0,
                                   view.input_policy.input_compression_active != 0};
            result["sensitivity_hash"] = view.film_exposure.sensitivity_hash;
            result["sensitivity"] = Json::array();
            for (std::size_t i = 0; i < view.film_exposure.sensitivity_rgb.count; i += 3)
                result["sensitivity"].push_back(bits(view.film_exposure.sensitivity_rgb.data + i, 3));
            result["tc_source_hash"] = view.input_policy.tc_source_hash;
            result["tc_lut_size"] = view.film_exposure.tc_lut_rgba.count;
            result["tc_samples"] = Json::array();
            if (view.film_exposure.tc_lut_rgba.count != 0) {
                for (std::size_t index : std::array<std::size_t, 5>{0, 768, 73728, 100000, 147452})
                    result["tc_samples"].push_back({{"index", index}, {"rgba", bits(view.film_exposure.tc_lut_rgba.data + index, 4)}});
            }
            return result;
        }
    } // namespace InputFixtures

    void input_leaf_tests(const Json& fixture) {
        std::vector<std::string> conversion_ids;
        for (const auto& row : fixture.at("conversion"))
            for (bool clamp : {false, true})
                conversion_ids.push_back(row.at("id").get<std::string>() + "/clamp-" + std::to_string(clamp));
        JuicerTests::Cohort conversion_rows("color.input_conversion.clamp", conversion_ids);

        const auto& inverse = fixture.at("recipes").at(0).at("expected").at("xyz_to_linear_srgb");
        for (const auto& row : fixture.at("recipes")) {
            JuicerTests::CaseEvent event("color.input_leaf.recipe", row.at("id").get<std::string>());
            require(row.at("expected").at("xyz_to_linear_srgb") == inverse && row.at("expected").at("config_xyz_to_linear_srgb") == inverse, "amended producer reference uses agreeing completed native A");
            event.passed();
        }
        // The standalone inverse and unready default remain native characterization.
        // Explicit conversion operands, including B, are consumed exactly as frozen.
        for (const auto& row : fixture.at("matrices")) {
            JuicerTests::CaseEvent event("color.input_leaf.matrix", std::to_string(row.at("space").get<int>()));
            const auto space = row.at("space").get<int>();
            const auto actual = JuicerTestFixture::Color::input_matrices(static_cast<Spectral::InputColorSpace>(space));
            FjInputColorMatrices facade{};
            expect_status(fj_test_input_matrices(static_cast<std::uint32_t>(space), &facade), FJ_STATUS_SUCCESS);
            require(bits(actual.rgbToXyz) == row.at("rgb_to_xyz") && bits(actual.nominalWhiteXYZ) == row.at("nominal_white") && bits(actual.d65WhiteXYZ) == row.at("d65_white") && bits(actual.xyzToLinearSrgb) == inverse, "production input matrices/nominal whites/inverse A exact");
            require(bits(facade.rgb_to_xyz) == row.at("rgb_to_xyz") && bits(facade.nominal_white_xyz) == row.at("nominal_white") && bits(facade.d65_white_xyz) == row.at("d65_white") && bits(facade.xyz_to_linear_srgb) == inverse, "direct-core input matrices exact");
            event.passed();
        }
        for (const auto& row : fixture.at("decode")) {
            JuicerTests::CaseEvent event("color.input_leaf.decode", row.at("id").get<std::string>());
            const auto rgb = triplet(row.at("rgb"));
            std::array<float, 3> actual{};
            expect_status(fj_test_decode_input(row.at("space").get<std::uint32_t>(), row.at("decode").get<std::uint32_t>(), rgb.data(), actual.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(actual), row.at("expected").at("linear")), row.at("id").get<std::string>() + " decoder branch/zero/nonfinite/overflow");
            event.passed();
        }
        for (const auto& row : fixture.at("conversion")) {
            JuicerTests::CaseEvent event("color.input_leaf.conversion", row.at("id").get<std::string>());
            const auto config = InputFixtures::config(row);
            const auto input = InputFixtures::foreign(config);
            const auto rgb = triplet(row.at("rgb"));
            const auto& expected = row.at("expected");
            for (bool clamp : {false, true}) {
                auto conversion_event = conversion_rows.case_event(row.at("id").get<std::string>() + "/clamp-" + std::to_string(clamp));
                std::array<float, 3> facadeRgb{}, facadeXyz{};
                expect_status(fj_test_input_to_dwg(&input, rgb.data(), clamp ? 1u : 0u, facadeRgb.data(), facadeXyz.data()), FJ_STATUS_SUCCESS);
                const std::string prefix = clamp ? "dwg_clamped" : "dwg_signed";
                require(same_values(bits(facadeRgb), expected.at(prefix + "_rgb")) && same_values(bits(facadeXyz), expected.at(prefix + "_xyz")), row.at("id").get<std::string>() + " facade DWG conversion/XYZ observation");
                conversion_event.passed();
            }
            std::array<float, 3> facadeRgb{}, facadeXyz{};
            expect_status(fj_test_input_to_linear_srgb(&input, rgb.data(), config.xyz_to_linear_srgb, facadeRgb.data(), facadeXyz.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(facadeRgb), expected.at("srgb_rgb")) && same_values(bits(facadeXyz), expected.at("srgb_xyz")), row.at("id").get<std::string>() + " facade explicit inverse linear sRGB/XYZ");
            event.passed();
        }
        for (const auto& row : fixture.at("linear")) {
            JuicerTests::CaseEvent event("color.input_leaf.linear", row.at("id").get<std::string>());
            const auto rgb = triplet(row.at("rgb"));
            std::array<float, 3> srgb{}, dwg{};
            expect_status(fj_test_linear_srgb_to_xyz(rgb.data(), srgb.data()), FJ_STATUS_SUCCESS);
            expect_status(fj_test_dwg_to_xyz(rgb.data(), dwg.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(srgb), row.at("expected").at("srgb_xyz")) && same_values(bits(JuicerTestFixture::Color::linear_srgb_to_xyz(rgb)), row.at("expected").at("srgb_xyz")), row.at("id").get<std::string>() + " unsanitized linear sRGB leaf");
            require(same_values(bits(dwg), row.at("expected").at("dwg_xyz")), row.at("id").get<std::string>() + " unsanitized DWG leaf");
            event.passed();
        }
        for (const auto& row : fixture.at("projection")) {
            JuicerTests::CaseEvent event("color.input_leaf.projection", row.at("id").get<std::string>());
            const auto rgb = triplet(row.at("rgb"));
            const auto matrix = InputFixtures::matrix(row.at("rgb_to_xyz"));
            const auto adaptation = InputFixtures::matrix(row.at("xyz_adapt"));
            const auto actual = JuicerTestFixture::Color::project_linear_rgb_to_xyz(rgb, matrix, adaptation);
            std::array<float, 3> facade{};
            expect_status(fj_test_project_linear_rgb_to_xyz(rgb.data(), matrix.m, adaptation.m, facade.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(actual), row.at("expected").at("projected_xyz")) && same_values(bits(facade), row.at("expected").at("projected_xyz")), row.at("id").get<std::string>() + " ordered already-linear projection");
            event.passed();
        }
        JuicerTests::CaseEvent comparator_event("color.input_leaf", "projection_order_comparator");
        const auto projectionCase = std::find_if(fixture.at("projection").begin(), fixture.at("projection").end(), [](const auto& row) {
            return row.at("id") == "projection-3-rgb-5";
        });
        require(projectionCase != fixture.at("projection").end(), "projection comparator case");
        const auto rgb = triplet(projectionCase->at("rgb"));
        const auto forward = InputFixtures::matrix(projectionCase->at("rgb_to_xyz"));
        const auto adapt = InputFixtures::matrix(projectionCase->at("xyz_adapt"));
        const auto expectedProjection = projectionCase->at("expected").at("projected_xyz");
        require(!same_values(bits(JuicerTestFixture::Color::project_linear_rgb_to_xyz(rgb, adapt, forward)), expectedProjection), "reversed matrix order rejected");
        auto transposed = forward;
        std::swap(transposed.m[1], transposed.m[3]);
        std::swap(transposed.m[2], transposed.m[6]);
        std::swap(transposed.m[5], transposed.m[7]);
        require(!same_values(bits(JuicerTestFixture::Color::project_linear_rgb_to_xyz(rgb, transposed, adapt)), expectedProjection), "transposed matrix rejected");
        std::array<float, 3> wronglyDecoded{};
        expect_status(fj_test_decode_input(3, 1, rgb.data(), wronglyDecoded.data()), FJ_STATUS_SUCCESS);
        require(!same_values(bits(JuicerTestFixture::Color::project_linear_rgb_to_xyz(wronglyDecoded, forward, adapt)), expectedProjection), "decoding already-linear TC projection rejected");
        auto coefficient = fixture.at("matrices").at(3).at("rgb_to_xyz");
        coefficient[0] = coefficient[0].get<std::uint32_t>() ^ 1u;
        require(coefficient != bits(JuicerTestFixture::Color::input_matrices(Spectral::InputColorSpace::SRGB_Rec709).rgbToXyz), "one coefficient bit rejected");
        require(bits(JuicerTestFixture::Color::input_matrices(Spectral::InputColorSpace::ACES2065_1).rgbToXyz) != fixture.at("matrices").at(3).at("rgb_to_xyz"), "input tag swap rejected");
        bool signedOutput = false;
        for (const auto& row : fixture.at("conversion")) {
            const auto& expected = row.at("expected").at("srgb_rgb");
            auto clamped = triplet(expected);
            for (float& value : clamped) {
                value = std::max(0.0f, value);
            }
            if (!same_values(bits(clamped), expected)) {
                signedOutput = true;
                break;
            }
        }
        require(signedOutput, "signed linear sRGB clamping rejected");
        comparator_event.passed();
        JuicerTests::CaseEvent controls_event("color.input_leaf", "abi_fault_and_default_controls");
        require(fj_test_input_color_abi_c() == 0, "input-color C11 layouts/signatures/nulls/flags/partial clearing/panic/canaries/readonly aliases");
        color_fault_tests();
        JuicerTests::Cohort input_countdown_operations_rows("color.input-countdown-operations", 6u);
        std::size_t input_countdown_operations_index = 0;
        for (std::uint32_t operation = 5; operation <= 10; ++operation) {
            auto input_countdown_operations_event = input_countdown_operations_rows.case_event(std::to_string(input_countdown_operations_index++));
            FaultScope cleanup;
            expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            std::array<float, 3> decoded{};
            const std::array<float, 3> rgb{.184f, -0.0f, -1.0f};
            expect_status(fj_test_decode_input(3, 1, rgb.data(), decoded.data()), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_INTERNAL_FAILURE);

            input_countdown_operations_event.passed();
        }
        JuicerTests::Cohort input_invalid_conversion_rows("color.input-invalid-conversion", 12u);
        std::size_t input_invalid_conversion_index = 0;
        JuicerTests::Cohort input_malformed_faults_rows("color.input-malformed-faults", 6u);
        std::size_t input_malformed_faults_index = 0;
        for (std::uint32_t operation : {FJ_TEST_INPUT_MATRICES, FJ_TEST_INPUT_TO_DWG, FJ_TEST_INPUT_TO_LINEAR_SRGB}) {
            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                auto input_malformed_faults_event = input_malformed_faults_rows.case_event(std::to_string(input_malformed_faults_index++));
                FaultScope cleanup;
                expect_status(arm_color_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                FjInputColorMatrices matrices{};
                const std::array<float, 3> rgb{.184f, -.25f, .5f};
                const std::array<float, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
                FjInputColorConversion input{FJ_INPUT_DWG, 0, 0, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}};
                std::array<float, 3> outRgb{}, outXyz{};
                if (operation == FJ_TEST_INPUT_MATRICES) {
                    expect_status(fj_test_boundary_input_matrices(4, &matrices), FJ_STATUS_UNSUPPORTED_INPUT);
                } else {
                    for (int invalid = 0; invalid < 3; ++invalid) {
                        auto input_invalid_conversion_event = input_invalid_conversion_rows.case_event(std::to_string(input_invalid_conversion_index++));
                        input.input_space = invalid == 0 ? 4u : 0u;
                        input.decode_cctf = invalid == 1 ? 2u : 0u;
                        input.adapt_xyz = invalid == 2 ? 2u : 0u;
                        const auto status = operation == FJ_TEST_INPUT_TO_DWG
                                                ? fj_test_input_to_dwg(&input, rgb.data(), 0, outRgb.data(), outXyz.data())
                                                : fj_test_input_to_linear_srgb(&input, rgb.data(), identity.data(), outRgb.data(), outXyz.data());
                        expect_status(status, FJ_STATUS_UNSUPPORTED_INPUT);

                        input_invalid_conversion_event.passed();
                    }
                    if (operation == FJ_TEST_INPUT_TO_DWG) {
                        input.input_space = 0;
                        input.decode_cctf = 0;
                        input.adapt_xyz = 0;
                        expect_status(fj_test_input_to_dwg(&input, rgb.data(), 2, outRgb.data(), outXyz.data()), FJ_STATUS_UNSUPPORTED_INPUT);
                    }
                }
                expect_status(raw_call(operation), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);

                input_malformed_faults_event.passed();
            }
        }
        require(fixture.at("input_d65") != fixture.at("process_dwg_white"), "input D65 and process DWG white remain distinct");
        require(Spectral::input_color_space_from_index(-1) == Spectral::InputColorSpace::DaVinciWideGamut && Spectral::input_color_space_from_index(4) == Spectral::InputColorSpace::DaVinciWideGamut, "native integer transport fallback retained");
        controls_event.passed();
        std::puts("Input color: exact matrices/A, decode, explicit A/B conversions, linear leaves/projection and C11/ten-operation fault/default/facade controls passed");
    }

    struct InputValues {
        std::vector<std::uint32_t> bits;
        std::array<std::uint64_t, 13> hashes{};
        bool operator==(const InputValues&) const = default;
    };
    InputValues input_values(const JuicerTestSupport::PreparedStateHold& state) {
        const auto view = prepared_view(state);
        const auto& raw = view.film_exposure;
        InputValues out;
        const auto append = [&](const float* samples, std::size_t count) {
            for (std::size_t i = 0; i < count; ++i)
                out.bits.push_back(std::bit_cast<std::uint32_t>(samples[i]));
        };
        append(raw.input_rgb_to_xyz, 9);
        append(raw.input_xyz_adapt, 9);
        append(raw.xyz_to_linear_srgb, 9);
        append(view.input_policy.nominal_white_xyz, 3);
        append(view.input_policy.projection_white_xyz, 3);
        append(raw.input_rgb_to_xyz, 9);
        append(raw.input_xyz_adapt, 9);
        append(raw.xyz_to_linear_srgb, 9);
        append(view.raw_midgray, 3);
        append(view.midgray_dwg_rgb, 3);
        append(view.input_policy.nominal_white_xyz, 3);
        append(view.input_policy.projection_white_xyz, 3);
        append(view.input_policy.projection_white_xyz, 3);
        const std::array scales{view.midgray_scale, view.raw_green, raw.mallett_green_midgray_scale, view.film_reference.meter_scale};
        for (float value : scales)
            out.bits.push_back(std::bit_cast<std::uint32_t>(value));
        for (auto value : {static_cast<std::uint32_t>(raw.input_color_space),
                           static_cast<std::uint32_t>((raw.flags & FJ_FILM_DECODE_CCTF) != 0),
                           static_cast<std::uint32_t>(raw.method == FJ_RAW_MALLETT_2019 ? 1 : 0),
                           static_cast<std::uint32_t>((raw.flags & FJ_FILM_ADAPT_INPUT) != 0),
                           1u,
                           1u,
                           static_cast<std::uint32_t>(raw.input_color_space),
                           static_cast<std::uint32_t>((raw.flags & FJ_FILM_DECODE_CCTF) != 0),
                           raw.method,
                           static_cast<std::uint32_t>((raw.flags & FJ_FILM_AUTO_EXPOSURE) != 0),
                           view.input_policy.input_compression_active})
            out.bits.push_back(value);
        out.hashes = {raw.hash, raw.tc_lut_hash, view.film_reference.hash, view.print_balance.hash, view.print_recipe_hash, view.scanner_recipe_hash, view.output_color.gamut_recipe_hash, view.recipe_hash, view.upload_core_hash, view.scanner_hash, raw.sensitivity_hash, view.input_policy.tc_source_hash, raw.tc_lut_rgba.count};
        return out;
    }
    InputValues input_values(const JuicerTestSupport::PublicationAdmission& result) {
        require(result.state != nullptr, "retained input-color prepared-state hold");
        return input_values(*result.state);
    }
    void input_admission_tests(const FjAssets* assets) {
        std::size_t failures = 0;
        std::size_t heldSupersessions = 0;
        std::size_t counts = 0;
        std::size_t skippedOperation10Supersessions = 0;
        std::vector<std::string> fault_ids;
        for (int route = 0; route < 4; ++route)
            for (int method = 0; method < 3; ++method)
                for (int space : {1, 3}) {
                    const std::array<std::uint32_t, 6> calls{method == 1 ? 1u : 3u, 0u, 0u, method == 1 ? 0u : 2u, 0u, method == 1 ? 0u : 1u};
                    for (std::uint32_t operation = 5; operation <= 10; ++operation)
                        for (std::uint32_t index = 1; index <= calls[operation - 5]; ++index)
                            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC})
                                fault_ids.emplace_back("route-" + std::to_string(route) + "/method-" + std::to_string(method) + "/space-" + std::to_string(space) + "/operation-" + std::to_string(operation) + "/index-" + std::to_string(index) + "/fault-" + std::to_string(fault));
                }
        JuicerTests::Cohort fault_rows("color.input_admission.fault_tuple", fault_ids);
        std::vector<std::string> supersession_ids;
        for (const auto& id : fault_ids)
            for (int mode = 0; mode < 3; ++mode)
                supersession_ids.push_back(id + "/mode-" + std::to_string(mode));
        JuicerTests::Cohort supersession_rows("color.input_admission.supersession", supersession_ids);
        JuicerTests::Cohort call_count_rows("color.input_admission.call_counts", 144u);
        std::size_t call_count_index = 0;
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                for (int space : {1, 3}) {
                    auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                    controls.values.input_color_space = space;
                    controls.values.input_cctf_decoding = 1;
                    auto changed = controls;
                    changed.values.camera_exposure_compensation_ev = 1.0;
                    auto next = changed;
                    next.values.input_color_space = space == 1 ? 3 : 1;
                    const std::array<std::uint32_t, 6> callCounts{method == 1 ? 1u : 3u, 0u, 0u, method == 1 ? 0u : 2u, 0u, method == 1 ? 0u : 1u};
                    for (std::uint32_t operation = 5; operation <= 10; ++operation) {
                        const auto calls = callCounts[operation - 5];
                        auto call_count_event = call_count_rows.case_event(std::to_string(call_count_index++));
                        {
                            FaultScope cleanup;
                            const auto heldCalls = operation == FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ ? 0u : calls;
                            expect_status(arm_color_fault(operation, heldCalls + 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                            (void)Cat02Fixtures::prepare(assets, controls);
                            expect_status(raw_call(operation), FJ_STATUS_INTERNAL_FAILURE);
                            ++counts;
                        }
                        call_count_event.passed();

                        if (operation == FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ && calls == 1) {
                            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                                const auto id = "route-" + std::to_string(route) + "/method-" + std::to_string(method) + "/space-" + std::to_string(space) + "/operation-" + std::to_string(operation) + "/index-1/fault-" + std::to_string(fault);
                                auto fault_event = fault_rows.case_event(id);
                                for (int mode = 0; mode < 3; ++mode) {
                                    auto event = supersession_rows.case_event(id + "/mode-" + std::to_string(mode));
                                    FaultScope cleanup;
                                    JuicerTestSupport::InstanceFixture state;
                                    JuicerTestSupport::capture(state, controls);
                                    const auto old = JuicerTestSupport::admit(assets, state);
                                    require(admitted(old), "projection supersession baseline admitted");
                                    const auto oldValues = input_values(old);
                                    const auto oldBuild = old.state->build();
                                    JuicerTestSupport::capture(state, changed);
                                    Supersession selection{&state, next, mode};
                                    fj_test_instance_prepare_hooks(nullptr, supersede_capture, &selection);
                                    expect_status(arm_color_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                                    const auto result = JuicerTestSupport::admit(assets, state);
                                    require(selection.fired && !selection.failed, "actual Rust recipe hook captured superseding projection input");
                                    if (mode == 0) {
                                        require(admitted(result) && result.state->input_identity() == JuicerTestSupport::hash_input_identity(next) && result.state->build() == oldBuild + 1, "valid C supersession commits only current C");
                                        require(input_values(result) != oldValues, "current C input products differ from retained A");
                                    } else {
                                        require((mode == 1 ? result.kind == FJ_ADMISSION_CAPTURE_FAILED : result.kind == FJ_ADMISSION_CAPTURING) && !result.state, "invalid or capturing current input blocks admission");
                                        require(JuicerTestSupport::PublicationObserver::metrics(state).build == oldBuild, "obsolete B is not counted as a publication");
                                    }
                                    require(input_values(old) == oldValues, "superseded projection retains complete A hold");
                                    fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                                    expect_status(raw_call(operation), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                    ++skippedOperation10Supersessions;
                                    event.passed();
                                }
                                fault_event.passed();
                            }
                            continue;
                        }
                        if (calls == 0)
                            continue;
                        for (std::uint32_t index = 1; index <= calls; ++index) {
                            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                                const auto id = "route-" + std::to_string(route) + "/method-" + std::to_string(method) + "/space-" + std::to_string(space) + "/operation-" + std::to_string(operation) + "/index-" + std::to_string(index) + "/fault-" + std::to_string(fault);
                                auto row_event = fault_rows.case_event(id);
                                FaultScope cleanup;
                                JuicerTestSupport::InstanceFixture state;
                                JuicerTestSupport::capture(state, controls);
                                const auto old = JuicerTestSupport::admit(assets, state);
                                require(admitted(old), "input-color baseline admitted");
                                const auto oldValues = input_values(old);
                                const auto oldBuild = old.state->build();
                                const auto oldIdentity = old.state->input_identity();
                                const auto before = JuicerTestSupport::PublicationObserver::metrics(state);
                                JuicerTestSupport::capture(state, changed);
                                expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                const auto result = JuicerTestSupport::admit(assets, state);
                                if (fault == FJ_TEST_COLOR_PANIC) {
                                    require_poisoned_admission(assets, result, state, old.state, oldBuild);
                                    for (std::uint32_t call = 0; call < index; ++call)
                                        expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                    require(input_values(old) == oldValues, "current input panic preserves admitted A hold");
                                    ++failures;

                                    JuicerTestSupport::InstanceFixture recovery;
                                    JuicerTestSupport::capture(recovery, controls);
                                    const auto recovery_a = JuicerTestSupport::admit(assets, recovery);
                                    require(admitted(recovery_a), "independent input recovery instance admits A");
                                    JuicerTestSupport::capture(recovery, changed);
                                    const auto recovery_b = JuicerTestSupport::admit(assets, recovery);
                                    require(admitted(recovery_b) && recovery_b.state->build() == recovery_a.state->build() + 1,
                                            "independent input recovery instance prepares B");

                                    for (int mode = 0; mode < 3; ++mode) {
                                        auto event = supersession_rows.case_event(id + "/mode-" + std::to_string(mode));
                                        JuicerTestSupport::InstanceFixture superseded;
                                        JuicerTestSupport::capture(superseded, changed);
                                        const auto held_b = JuicerTestSupport::admit(assets, superseded);
                                        require(admitted(held_b), "independent input supersession instance publishes B");
                                        const auto held_build = held_b.state->build();
                                        const auto held_values = input_values(held_b);
                                        JuicerTestSupport::capture(superseded, controls);
                                        Supersession selection{&superseded, changed, mode};
                                        if (operation == FJ_TEST_LINEAR_SRGB_TO_XYZ || index > 1)
                                            fj_test_instance_prepare_hooks(supersede_capture, nullptr, &selection);
                                        else
                                            fj_test_instance_prepare_hooks(nullptr, supersede_capture, &selection);
                                        expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                        const auto obsolete = JuicerTestSupport::admit(assets, superseded);
                                        require(selection.fired && !selection.failed, "obsolete input panic completed newer capture");
                                        require_poisoned_admission(assets, obsolete, superseded, held_b.state, held_build);
                                        require(input_values(held_b) == held_values, "obsolete panic leaves earlier B hold exact");
                                        for (std::uint32_t call = 0; call < index; ++call)
                                            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                        fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                                        expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                                        ++heldSupersessions;
                                        event.passed();
                                    }
                                    row_event.passed();
                                    continue;
                                }
                                require_preparation_failure(result, fault);
                                for (std::uint32_t call = 0; call < index; ++call)
                                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                const auto after = JuicerTestSupport::PublicationObserver::metrics(state);
                                require(after.publishedIdentity == before.publishedIdentity && after.build == oldBuild && after.inputIdentity == JuicerTestSupport::hash_input_identity(changed), "input preparation fault keeps private A and current B identity");
                                require(input_values(old) == oldValues, "failed input preparation preserves complete A values");
                                ++failures;

                                JuicerTestSupport::capture(state, controls);
                                const auto reused = JuicerTestSupport::admit(assets, state);
                                require(admitted(reused) && reused.state->input_identity() == oldIdentity && reused.state->build() == oldBuild, "failed input rebuild returns to retained A without rebuilding");
                                expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                                JuicerTestSupport::capture(state, changed);
                                const auto recovered = JuicerTestSupport::admit(assets, state);
                                require(admitted(recovered) && recovered.state->input_identity() == JuicerTestSupport::hash_input_identity(changed) && recovered.state->build() == oldBuild + 1, "input-color recovery commits one new build");
                                const auto recoveredValues = input_values(recovered);
                                require(input_values(old) == oldValues, "input-color recovery leaves A hold exact");

                                for (int mode = 0; mode < 3; ++mode) {
                                    auto event = supersession_rows.case_event(id + "/mode-" + std::to_string(mode));
                                    Supersession selection{&state, changed, mode};
                                    JuicerTestSupport::capture(state, controls);
                                    fj_test_instance_prepare_hooks(nullptr, supersede_capture, &selection);
                                    expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                    const auto current = JuicerTestSupport::admit(assets, state);
                                    require(selection.fired && !selection.failed, "input-color production recipe hook installed current capture");
                                    if (mode == 0) {
                                        require(admitted(current) && current.state->input_identity() == JuicerTestSupport::hash_input_identity(changed) && current.state->build() == recovered.state->build(), "obsolete input failure retries and reuses current B");
                                    } else {
                                        require((mode == 1 ? current.kind == FJ_ADMISSION_CAPTURE_FAILED : current.kind == FJ_ADMISSION_CAPTURING) && !current.state, "invalid or capturing input blocks admission");
                                    }
                                    require(JuicerTestSupport::PublicationObserver::metrics(state).build == recovered.state->build(), "superseded input attempt does not advance committed build");
                                    require(input_values(old) == oldValues && input_values(recovered) == recoveredValues, "supersession retains complete A and B holds");
                                    for (std::uint32_t call = 0; call < index; ++call)
                                        expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                    fj_test_instance_prepare_hooks(nullptr, nullptr, nullptr);
                                    expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                                    ++heldSupersessions;
                                    event.passed();
                                }
                                row_event.passed();
                            }
                        }
                    }
                }
            }
        }
        require(counts == 144 && failures == 176 && heldSupersessions == 528 && skippedOperation10Supersessions == 96 && heldSupersessions + skippedOperation10Supersessions == 624, "actual input preparation and supersession memberships");
        std::printf("Input-color admission: %zu call-count rows, %zu typed failure/recovery rows, %zu failing-leaf and %zu skipped-projection supersessions passed\n", counts, failures, heldSupersessions, skippedOperation10Supersessions);
    }

    template <typename Function>
    void require_input_failure(std::uint32_t operation, std::uint32_t fault, Function&& function) {
        FaultScope cleanup;
        expect_status(arm_color_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
        bool thrown = false;
        try {
            function();
        } catch (const JuicerCuda::ExecutionFailure& error) {
            require(error.failure.status.category == (fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE) && error.failure.status.api == FJ_API_NONE && error.failure.status.native_code == 0 && !error.deferredDirError && !error.failure.diagnostic.empty(), "input-color typed failure propagates");
            thrown = true;
        }
        require(thrown, "real input-color primitive/helper failure is terminal");
        expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
    }

    void input_helper_tests(const FjAssets* assets, const Json& fixture) {
        auto product = Cat02Fixtures::prepare(assets, Cat02Fixtures::snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Mallett2019));
        require(assets != nullptr, "Hanatos input helper source has process Rust assets");
        JuicerTestSupport::HanatosSource hanatosSource(assets);
        const FjFloatSpan hanatosSamples = hanatosSource.view().samples;
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjPreparedStateView view{};
        require(product.view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
        const auto& tables = view.exposure_tables;
        const std::array<float, 9> inverse{view.s_inverse[0], view.s_inverse[1], view.s_inverse[2], view.s_inverse[3], view.s_inverse[4], view.s_inverse[5], view.s_inverse[6], view.s_inverse[7], view.s_inverse[8]};
        require(Cat02Fixtures::tables_result(tables) == fixture.at("helper_tables") && bits(inverse) == fixture.at("helper_s_inv"), "native-parent helper preparation exact");
        for (const auto& row : fixture.at("helpers")) {
            JuicerTests::CaseEvent event("color.input_helper", row.at("id").get<std::string>());
            require(Cat02Fixtures::helper_result(row, tables, inverse, true, hanatosSamples) == row.at("hanatos_expected"), row.at("id").get<std::string>() + " retained Hanatos helper exact");
            require(Cat02Fixtures::helper_result(row, tables, inverse, false, hanatosSamples) == row.at("tables_expected"), row.at("id").get<std::string>() + " retained table helper exact");
            event.passed();
        }
        const std::array<float, 3> rgb{2.0f, -.25f, .5f};
        for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
            JuicerTests::CaseEvent event("color.input_helper_fault", std::to_string(fault));
            require_spectrum_fixture_failure(tables, inverse, 1, fault, hanatosSamples);
            require_spectrum_fixture_failure(tables, inverse, 2, fault, hanatosSamples);
            std::array<float, 3> output{}, xyz{};
            require_color_facade_failure(FJ_TEST_DWG_TO_XYZ, fault, [&] {
                return fj_test_dwg_to_xyz(rgb.data(), output.data());
            });
            const auto conversion = InputFixtures::foreign(view.film_exposure);
            require_color_facade_failure(FJ_TEST_INPUT_TO_DWG, fault, [&] {
                return fj_test_input_to_dwg(&conversion, rgb.data(), 0, output.data(), xyz.data());
            });
            require_color_facade_failure(FJ_TEST_INPUT_TO_LINEAR_SRGB, fault, [&] {
                return fj_test_input_to_linear_srgb(&conversion, rgb.data(), view.film_exposure.xyz_to_linear_srgb, output.data(), xyz.data());
            });
            require_input_failure(FJ_TEST_LINEAR_SRGB_TO_XYZ, fault, [&] {
                (void)JuicerTestFixture::Color::linear_srgb_to_xyz(rgb);
            });
            require_input_failure(FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ, fault, [&] {
                Spectral::Mat3 inputRgbToXyz{}, inputXyzAdapt{};
                std::copy_n(view.film_exposure.input_rgb_to_xyz, 9, inputRgbToXyz.m);
                std::copy_n(view.film_exposure.input_xyz_adapt, 9, inputXyzAdapt.m);
                (void)JuicerTestFixture::Color::project_linear_rgb_to_xyz(rgb, inputRgbToXyz, inputXyzAdapt);
            });
            require_input_failure(FJ_TEST_INPUT_MATRICES, fault, [&] {
                (void)JuicerTestFixture::Color::input_matrices(Spectral::InputColorSpace::ITU_R_BT2020);
            });
            event.passed();
        }
        std::puts("Input-color helpers: independent spectra and both-category direct primitive/DWG helper failures passed");
    }

    void input_preparation_tests(const FjAssets* assets, const Json& fixture, const std::filesystem::path& fixturePath) {
        JuicerTests::CaseEvent retained_event("color.input_preparation", "retained_comparison_and_defaults");
        retained_comparison_tests();
        const auto cases = InputFixtures::scenarios();
        require(cases.size() == 140 && cases.size() == fixture.at("recipes").size(), "96 base plus 44 bounded input-color products");
        require(InputFixtures::controls_json(JuicerTestSupport::PreparationInput{}) == fixture.at("defaults"), "all frozen parent defaults exact");
        retained_event.passed();
        const auto witness = load_preset_witness(fixturePath, "input-color-manifest.json");
        bool wrongPreset = false;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [id, controls] = cases[i];
            const auto& row = fixture.at("recipes").at(i);
            JuicerTests::CaseEvent event("color.input_recipe", row.at("id").get<std::string>());
            require(id == row.at("id").get<std::string>() && InputFixtures::controls_json(controls) == row.at("controls"), "ordered frozen product inputs/settings exact");
            auto product = Cat02Fixtures::prepare(assets, controls);
            const auto actual = InputFixtures::complete_result(product);
            if (actual != row.at("expected")) {
                throw std::runtime_error(id + " completed product mismatch: " + Json::diff(row.at("expected"), actual).dump());
            }
            if (id == witness.at("recipe_id").get<std::string>()) {
                check_preset_witness(actual, row.at("expected"), witness);
                wrongPreset = true;
            }
            if (i == 0) {
                auto altered = actual;
                altered["hashes"][0] = altered["hashes"][0].get<std::uint64_t>() ^ 1u;
                require(altered != row.at("expected"), "required completed identity negative control");
            }
            event.passed();
        }
        require(wrongPreset, "native preset differences are observable and reject wrong references");
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                JuicerTests::CaseEvent event("color.input_admission", std::to_string(route) + "/" + std::to_string(method));
                auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                controls.values.input_color_space = 1;
                controls.values.input_cctf_decoding = 1;
                JuicerTestSupport::InstanceFixture state;
                JuicerTestSupport::capture(state, controls);
                const auto old = JuicerTestSupport::admit(assets, state);
                require(admitted(old), "retained input-color route baseline");
                const auto held = retain_values(old);
                const auto fullHeld = InputFixtures::publication_complete_result(*old.state);
                const auto counter = old.state->build();
                const auto identity = old.state->input_identity();
                const auto repeated = JuicerTestSupport::admit(assets, state);
                require(admitted(repeated) && repeated.state->input_identity() == identity && repeated.state->build() == counter, "unchanged input uses the exact Rust publication");
                auto decoding = controls;
                decoding.values.input_cctf_decoding = 0;
                auto undecoded = Cat02Fixtures::prepare(assets, decoding);
                FjPreparedStateView undecodedView{};
                std::array<char, 2048> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                require(undecoded.view(undecodedView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                const auto oldView = prepared_view(*old.state);
                require(oldView.film_exposure.hash != undecodedView.film_exposure.hash, "input decoding remains a contributing film identity control");
                if (method != 1) {
                    require(bits(oldView.raw_midgray, 3) == bits(undecodedView.raw_midgray, 3), "TC mid-gray remains already linear without CCTF decoding");
                }
                expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
                auto changed = controls;
                changed.values.input_color_space = 3;
                JuicerTestSupport::capture(state, changed);
                const auto replacement = JuicerTestSupport::admit(assets, state);
                require(admitted(replacement) && replacement.state->input_identity() == JuicerTestSupport::hash_input_identity(changed) && replacement.state->build() == counter + 1, "input choice replaces complete publication with captured identity and committed build");
                require_retained_values(old, held);
                require(InputFixtures::publication_complete_result(*old.state) == fullHeld, "full input-color config/scales/matrices retained after cache release/replacement");
                auto output = changed;
                output.values.output_cctf_encoding = 0;
                auto encodedOff = Cat02Fixtures::prepare(assets, output);
                FjPreparedStateView encodedOffView{};
                require(encodedOff.view(encodedOffView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                const auto liveView = prepared_view(*replacement.state);
                require(liveView.film_exposure.hash == encodedOffView.film_exposure.hash && liveView.film_exposure.tc_lut_hash == encodedOffView.film_exposure.tc_lut_hash && liveView.scanner_hash == encodedOffView.scanner_hash && Cat02Fixtures::tables_result(liveView.scanner_tables) == Cat02Fixtures::tables_result(encodedOffView.scanner_tables), "output encoding preserves film/TC and scanner spectral families");
                event.passed();
            }
        }
        input_helper_tests(assets, fixture);
        JuicerTests::CaseEvent admission_event("color.input_preparation", "input_admission_failure_contract");
        input_admission_tests(assets);
        admission_event.passed();
        std::puts("Input-color preparation: 140 exact products/identities, controls, retained owners/cache replacement and terminal admission gates passed");
    }

    namespace FrameFixture = JuicerCudaTest::PreparedFrameFixture;

    struct GpuReuseCall {
        GpuReuseCall(JuicerTestSupport::RuntimeFixture& processRuntime,
                     JuicerCuda::ResourceManager::DeviceContextKey contextKey)
            : runtime(processRuntime), key(contextKey) {}

        JuicerTestSupport::RuntimeFixture& runtime;
        JuicerCuda::ResourceManager::DeviceContextKey key;
        cudaStream_t stream = nullptr;
        std::uint64_t sequence = 1;
        std::uint64_t baselineScannerLutHash = 0;
        std::uint64_t tcLutHash = 0;
        std::array<const float*, 6> previous{};
        std::vector<std::uint32_t> previousContent;
        std::uint64_t previousHash = 0;
        const float* previousTc = nullptr;
        std::vector<std::uint32_t> previousTcContent;
        std::array<long long, 3> preparationUs{};
        std::size_t transition = 0;
    };

    bool prepare_and_check_gpu_reuse(void* opaque,
                                     const FrameFixture::Inputs& prepared,
                                     std::string& diagnostic) {
        auto& call = *static_cast<GpuReuseCall*>(opaque);
        const FjRuntimeView runtimeView = call.runtime.borrow();
        const auto& state = prepared.state;
        call.tcLutHash = state.film_exposure.tc_lut_hash;
        if (call.transition == 0)
            call.baselineScannerLutHash = state.scanner_lut.hash;
        else
            require(state.scanner_lut.hash == call.baselineScannerLutHash,
                    "CCTF-only scanner descriptor identity");

        JuicerCuda::ResourceManager::SubmissionSnapshot submission;
        submission.instanceToken.value = 0x4341543136ull;
        submission.frameToken.value = call.sequence;
        submission.snapshotId = call.sequence++;
        submission.deviceContextKey = call.key;
        submission.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            state.upload_core_hash, state.dir_recipe.hash, state.scanner_hash, 0);

        const auto begin = std::chrono::steady_clock::now();
        const auto operation = static_cast<std::uint32_t>(5 + (call.sequence % 6));
        expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
        JuicerCuda::Failure error;
        auto frame = JuicerCuda::borrow_root(runtimeView.cuda).prepare_cuda_frame(call.key, submission, prepared.preparation, {}, call.stream, error, prepared.hostPreparation);
        call.preparationUs[call.transition++] = std::chrono::duration_cast<std::chrono::microseconds>(
                                                    std::chrono::steady_clock::now() - begin)
                                                    .count();
        if (!frame.active()) {
            diagnostic = error.diagnostic;
            return false;
        }

        const auto focused = frame.focused_resources();
        const auto& tc = state.film_exposure.tc_lut_rgba;
        if (tc.count != 0) {
            require(focused.film.filmTcLut && focused.film.filmTcLutExtent == 192,
                    "actual TC device allocation bound");
            std::vector<float> uploaded(tc.count);
            require(cudaMemcpy(uploaded.data(), focused.film.filmTcLut, uploaded.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
                    "TC full device content read");
            std::vector<std::uint32_t> tcContent;
            tcContent.reserve(uploaded.size());
            for (std::size_t i = 0; i < uploaded.size(); ++i) {
                tcContent.push_back(std::bit_cast<std::uint32_t>(uploaded[i]));
                require(tcContent.back() == std::bit_cast<std::uint32_t>(tc.data[i]),
                        "TC upload preserves complete Rust-owned table bits");
            }
            if (call.previousTc) {
                require(call.previousTc == focused.film.filmTcLut &&
                            tcContent == call.previousTcContent,
                        "CCTF-only and warm reuse actual TC allocation/content");
            } else {
                call.previousTc = focused.film.filmTcLut;
                call.previousTcContent = std::move(tcContent);
            }
        } else {
            require(!focused.film.filmTcLut && focused.film.filmTcLutExtent == 0,
                    "Mallett has no TC device allocation");
        }
        require(bits(focused.film.inputRGBToXYZ) == bits(state.film_exposure.input_rgb_to_xyz, 9) &&
                    bits(focused.film.inputXYZAdapt) == bits(state.film_exposure.input_xyz_adapt, 9) &&
                    bits(focused.film.xyzToLinearSrgb) == bits(state.film_exposure.xyz_to_linear_srgb, 9),
                "prepared native film projection binds exact completed matrices");

        const auto* lut = focused.scanLut;
        require(lut && lut->canonical_ready(), "actual owning scanner LUT prepared");
        const std::array<const float*, 6> addresses{
            lut->log2PchipXYZ, lut->slopeC, lut->slopeM, lut->slopeY, lut->cellMin, lut->cellMax};
        const std::size_t voxels = static_cast<std::size_t>(lut->res) * lut->res * lut->res * 3;
        const std::size_t cells = static_cast<std::size_t>(lut->res - 1) * (lut->res - 1) *
                                  (lut->res - 1) * 3;
        std::vector<std::uint32_t> content;
        for (std::size_t plane = 0; plane < addresses.size(); ++plane) {
            std::vector<float> values(plane < 4 ? voxels : cells);
            require(cudaMemcpy(values.data(), addresses[plane], values.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
                    "owning scanner LUT content read");
            for (float value : values)
                content.push_back(std::bit_cast<std::uint32_t>(value));
        }
        if (call.previous[0]) {
            require(addresses == call.previous && content == call.previousContent &&
                        lut->hash == call.previousHash,
                    "CCTF-only and warm reuse actual six device allocations/content/identity");
        } else {
            call.previous = addresses;
            call.previousContent = std::move(content);
            call.previousHash = lut->hash;
        }
        int* flag = nullptr;
        require(frame.prepare_scan_error_stage(flag, nullptr, error) ==
                    JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready,
                error.diagnostic);
        require(frame.finalize_scan_error_stage(flag, nullptr, error), error.diagnostic);
        require(frame.finish(call.stream, error), error.diagnostic);
        expect_status(raw_call(operation), FJ_STATUS_INTERNAL_FAILURE);
        return true;
    }

    void gpu_reuse_tests(JuicerTestSupport::RuntimeFixture& runtime) {
        const FjRuntimeView runtimeView = runtime.borrow();
        const FjAssets* assets = runtimeView.assets;
        std::vector<std::string> binding_ids;
        std::vector<std::string> context_ids;
        for (int scenario = 0; scenario < 7; ++scenario)
            for (int route = 0; route < 2; ++route) {
                const auto id = std::to_string(scenario) + "/" + std::to_string(route);
                context_ids.push_back(id);
                for (unsigned binding = 0; binding < 3; ++binding)
                    binding_ids.push_back(id + "/" + std::to_string(binding));
            }
        const JuicerTests::Cohort bindings("color.gpu_reuse.binding", binding_ids);
        const JuicerTests::Cohort contexts("color.gpu_reuse.context_retire", context_ids);
        require(cudaSetDevice(0) == cudaSuccess && cudaFree(nullptr) == cudaSuccess, "GPU initialization");
        void* context = nullptr;
        std::string contextError;
        require(JuicerCuda::query_current_cuda_context(context, contextError), contextError);
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, context};
        std::uint64_t sequence = 1;
        // Keep the accepted input-color cases and exercise both TC methods/polarities.
        for (int scenario = 0; scenario < 7; ++scenario) {
            const int method = scenario == 0 || scenario == 5 ? 0 : (scenario == 4 || scenario == 6 ? 2 : 1);
            const int polarity = scenario >= 5 ? 1 : 0;
            for (int route = 0; route < 2; ++route) {
                const int selectedRoute = polarity * 2 + route;
                auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(selectedRoute), static_cast<Spektrafilm::RgbToRawMethod>(method));
                if (method == 1) {
                    controls.print_key = "kodak_portra_endura";
                }
                if (scenario == 2 || scenario == 3) {
                    controls.values.input_color_space = scenario == 2 ? 1 : 3;
                    controls.values.input_cctf_decoding = 1;
                }
                controls.values.grain.active = false;
                controls.values.dir_couplers.active = false;
                auto product = prepare(assets, controls);
                controls.values.output_cctf_encoding = 0;
                auto encodedOff = prepare(assets, controls);
                GpuReuseCall call{runtime, key};
                call.sequence = sequence;
                FrameFixture::FrameFacts facts{};
                facts.renderExtent = {0, 0, 16, 16};
                facts.fullFrameExtent = facts.renderExtent;
                facts.pixelDefinition = facts.renderExtent;
                facts.pixelSizeUm = 6.0f;
                facts.filmFormatLongEdgeMm = controls.values.camera_film_format_long_edge_mm;
                facts.frameRate = 24.0;
                facts.sessionSeed = 0x243f6a8885a308d3ull;
                facts.clipToken = 0x13198a2e03707344ull;
                unsigned binding = 0;
                for (const JuicerTestSupport::PreparedStateHold* prepared : {&product, &encodedOff, &product}) {
                    auto binding_event = bindings.case_event(std::to_string(scenario) + "/" + std::to_string(route) + "/" + std::to_string(binding++));
                    std::string diagnostic;
                    require(FrameFixture::with_prepared_frame_inputs(
                                assets, *prepared, facts, &call, &prepare_and_check_gpu_reuse, diagnostic),
                            diagnostic);
                    binding_event.passed();
                }
                sequence = call.sequence;
                std::printf("GPU scanner route=%d method=%d input=%d decode=%d preparation_us cold=%lld cctf=%lld warm=%lld identity=%llu exact_values=%zu allocations=[%p,%p,%p,%p,%p,%p]\n",
                            selectedRoute,
                            method,
                            controls.values.input_color_space,
                            controls.values.input_cctf_decoding,
                            call.preparationUs[0],
                            call.preparationUs[1],
                            call.preparationUs[2],
                            static_cast<unsigned long long>(call.previousHash),
                            call.previousContent.size(),
                            static_cast<const void*>(call.previous[0]),
                            static_cast<const void*>(call.previous[1]),
                            static_cast<const void*>(call.previous[2]),
                            static_cast<const void*>(call.previous[3]),
                            static_cast<const void*>(call.previous[4]),
                            static_cast<const void*>(call.previous[5]));
                std::printf("GPU TC route=%d method=%d identity=%llu exact_values=%zu allocation=%p\n", selectedRoute, method, static_cast<unsigned long long>(call.tcLutHash), call.previousTcContent.size(), static_cast<const void*>(call.previousTc));
                std::string diagnostic;
                auto context_event = contexts.case_event(std::to_string(scenario) + "/" + std::to_string(route));
                require(JuicerCuda::borrow_root(runtimeView.cuda).retire_idle_context(0, context, diagnostic), diagnostic);
                context_event.passed();
            }
        }
        std::puts("GPU reuse: Hanatos/Arctic both polarities and retained Mallett input-color cases; direct/print scanner and TC allocations/content/identity reuse; prepared binding leaves math faults unconsumed");
    }

} // namespace

int main(int argc, char** argv) try {
    JuicerTests::CaseLog case_log;
    if (argc == 5 && std::string_view(argv[argc - 1]) == "--list-cases") {
        const std::string group = argv[3];
        if (group == "fixture-format") {
            JuicerTests::list_case("color_preparation", group);
            return 0;
        }
        if (group == "reuse-gpu") {
            JuicerTests::list_case("color_preparation", "owner_close");
            // Inner tuple membership is independently declared before execution.
            for (int scenario = 0; scenario < 7; ++scenario)
                for (int route = 0; route < 2; ++route) {
                    const auto id = std::to_string(scenario) + "/" + std::to_string(route);
                    JuicerTests::list_case("color.gpu_reuse.context_retire", id);
                    for (unsigned binding = 0; binding < 3; ++binding)
                        JuicerTests::list_case("color.gpu_reuse.binding", id + "/" + std::to_string(binding));
                }
            return 0;
        }
        const Json listed = PrintIlluminantFixtures::apply(argv[2], load_exposure_qualified_fixture(argv[2]));
        const auto list_rows = [&](std::string_view caller, const Json& rows) {
            for (const auto& row : rows)
                JuicerTests::list_case(caller, row.at("id").get<std::string>());
        };
        if (group == "leaf") {
            list_rows("color.leaf", listed.at("leaf"));
            JuicerTests::list_case("color.leaf", "abi_fault_and_comparator_controls");
        } else if (group == "preparation") {
            for (int space = 0; space < 9; ++space)
                JuicerTests::list_case("color.preparation.transform", std::to_string(space));
            for (const auto& row : listed.at("tables"))
                JuicerTests::list_case("color.preparation.table", std::to_string(row.at("space").get<int>()));
            list_rows("color.preparation.recipe", listed.at("recipes"));
            for (int route = 0; route < 2; ++route)
                JuicerTests::list_case("color.preparation.encoding_reuse", std::to_string(route));
            JuicerTests::list_case("color.preparation", "admission_and_construction_controls");
        } else if (group == "cat02-leaf") {
            list_rows("color.cat02_leaf", listed.at("leaf"));
            list_rows("color.cat02_leaf", listed.at("threshold"));
            JuicerTests::list_case("color.cat02_leaf", "direction_abi_and_fault_controls");
        } else if (group == "cat02-preparation") {
            list_rows("color.cat02_recipe", listed.at("recipes"));
            for (int route = 0; route < 4; ++route)
                JuicerTests::list_case("color.cat02_scanner_illuminant", std::to_string(route));
            list_rows("color.cat02_scanner", listed.at("scanner_cases"));
            for (const auto& row : listed.at("scanner_invalid_cases"))
                JuicerTests::list_case("color.cat02_scanner_invalid", std::to_string(row.at("medium").get<int>()));
            list_rows("color.cat02_helper", listed.at("helpers"));
            for (auto fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC})
                JuicerTests::list_case("color.cat02_helper_fault", std::to_string(fault));
            JuicerTests::list_case("color.cat02_preparation", "admission_helpers_controls");
        } else if (group == "input-leaf") {
            list_rows("color.input_leaf.recipe", listed.at("recipes"));
            list_rows("color.input_leaf.decode", listed.at("decode"));
            list_rows("color.input_leaf.conversion", listed.at("conversion"));
            list_rows("color.input_leaf.linear", listed.at("linear"));
            list_rows("color.input_leaf.projection", listed.at("projection"));
            for (const auto& row : listed.at("matrices"))
                JuicerTests::list_case("color.input_leaf.matrix", std::to_string(row.at("space").get<int>()));
            JuicerTests::list_case("color.input_leaf", "projection_order_comparator");
            JuicerTests::list_case("color.input_leaf", "abi_fault_and_default_controls");
        } else if (group == "input-preparation") {
            list_rows("color.input_recipe", listed.at("recipes"));
            JuicerTests::list_case("color.input_preparation", "retained_comparison_and_defaults");
            for (int route = 0; route < 4; ++route)
                for (int method = 0; method < 3; ++method)
                    JuicerTests::list_case("color.input_admission", std::to_string(route) + "/" + std::to_string(method));
            list_rows("color.input_helper", listed.at("helpers"));
            for (auto fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC})
                JuicerTests::list_case("color.input_helper_fault", std::to_string(fault));
            JuicerTests::list_case("color.input_preparation", "input_admission_failure_contract");
        }
        JuicerTests::list_case("color_preparation", "owner_close");
        return 0;
    }
    JuicerTestSupport::RuntimeFixture owner;
    try {
        require(argc == 4, "resources, immutable fixture and group required");
        const std::string group = argv[3];
        if (group == "fixture-format") {
            JuicerTests::CaseEvent event("color_preparation", group);
            shared_fixture_loading_tests(argv[1]);
            event.passed();
            case_log.finish();
            return 0;
        }
        owner.create(argv[1]);
        const auto fixture = PrintIlluminantFixtures::apply(argv[2], load_exposure_qualified_fixture(argv[2]));
        const FjRuntimeView runtimeView = owner.borrow();
        const FjAssets* assets = runtimeView.assets;
        if (group == "leaf") {
            leaf_tests(fixture);
        } else if (group == "preparation") {
            preparation_tests(assets, fixture);
        } else if (group == "cat02-leaf") {
            cat02_leaf_tests(fixture);
        } else if (group == "cat02-preparation") {
            cat02_preparation_tests(assets, fixture, argv[2]);
        } else if (group == "input-leaf") {
            input_leaf_tests(fixture);
        } else if (group == "input-preparation") {
            input_preparation_tests(assets, fixture, argv[2]);
        } else if (group == "reuse-gpu") {
            gpu_reuse_tests(owner);
        } else {
            throw std::runtime_error("unknown color test group");
        }
        {
            JuicerTests::CaseEvent close_event("color_preparation", "owner_close");
            expect_status(owner.close(), FJ_STATUS_SUCCESS);
            close_event.passed();
        }
        case_log.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Color tests: %s\n", error.what());
        return 1;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        std::fprintf(stderr, "Color construction status=%u: %s\n", error.failure.status.category, error.failure.diagnostic.c_str());
        return 1;
    }
} catch (const std::exception& error) {
    std::fprintf(stderr, "Color test setup/output: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("Color test setup/output failed\n", stderr);
    return 1;
}
