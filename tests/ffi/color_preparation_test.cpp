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
#include <mutex>
#include <span>
#include <thread>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>

#include "ColorTransforms.h"
#include "GamutCompression.h"
#include "RustGamutBridge.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RustAssetBridge.h"
#include "RustColorBridge.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "juicer_legacy_api.h"
#include "juicer_test_api.h"
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
    void require(bool condition, const std::string& diagnostic) {
        if (!condition) {
            throw std::runtime_error(diagnostic);
        }
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
        for (const auto* logical : {"linux-debug.json", "linux-release.json"}) {
            const auto word = std::string(logical) == "linux-debug.json" ? 11 : 17;
            write(updates / logical, {{"original_fixture", logical}, {"patch", {{{"op", "test"}, {"path", "/recipes/0/expected/sensitivity/0"}, {"value", 0}}, {{"op", "replace"}, {"path", "/recipes/0/expected/sensitivity/0"}, {"value", word}}, {{"op", "test"}, {"path", "/recipes/1/expected/sensitivity/0"}, {"value", 0}}}}});
            const auto fixture = load_exposure_qualified_fixture(color / logical);
            require(fixture.at("recipes").at(0).at("expected").at("sensitivity").at(0) == word, "logical alias selects its own ordered supplement");
            require(fixture.at("recipes").at(1).at("expected").at("sensitivity") == Json::array({0, 0x80000000u, 0x7fc00123u}), "expanded expected cases retain independent bits and patch state");
        }
        bool unknownRejected = false;
        try {
            (void)load_shared_fixture(color / "unknown.json", "manifest.json");
        } catch (const Json::out_of_range&) {
            unknownRejected = true;
        }
        require(unknownRejected, "unregistered logical preset is rejected");
        for (const auto& index : Json::array({-1, 0.5, std::numeric_limits<std::uint64_t>::max()})) {
            write(color / "shared.json", {{"expected_blocks", "blocks.json"}, {"recipes", {{{"expected", {{"sensitivity_record", index}}}}}}});
            bool referenceRejected = false;
            try {
                (void)load_shared_fixture(color / "linux-debug.json", "manifest.json");
            } catch (const std::runtime_error& error) {
                referenceRejected = std::string(error.what()) == "invalid expected block reference";
            }
            require(referenceRejected, "expected block references require a bounded nonnegative integer");
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
    ParamSnapshot snapshot(Spektrafilm::ScanRoute route, Spektrafilm::RgbToRawMethod method) {
        ParamSnapshot out;
        out.filmProfileKey = Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Negative ? "kodak_portra_400" : "fujifilm_provia_100f";
        out.scanRoute = route;
        out.spectralUpsamplingMode = static_cast<int>(method);
        out.cameraExposureCompensationEv = 0.75;
        out.cameraAutoExposureEnabled = 0;
        out.inputCompressionEnabled = 0;
        return out;
    }
    std::vector<std::pair<std::string, ParamSnapshot>> scenarios() {
        std::vector<std::pair<std::string, ParamSnapshot>> out;
        for (int polarity = 0; polarity < 2; ++polarity) {
            for (int route = 0; route < 2; ++route) {
                for (int method = 0; method < 3; ++method) {
                    out.emplace_back("recipe-" + std::to_string(polarity) + "-" + std::to_string(route) + "-" + std::to_string(method), snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), static_cast<Spektrafilm::RgbToRawMethod>(method)));
                }
            }
        }
        auto cctf = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Hanatos2025);
        cctf.outputCctfEncoding = 0;
        out.emplace_back("recipe-cctf-off", cctf);
        auto input = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Hanatos2025);
        input.inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::SRGB_Rec709);
        out.emplace_back("recipe-input-srgb", input);
        auto output = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Hanatos2025);
        output.outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
        out.emplace_back("recipe-output-dci", output);
        return out;
    }
    PreparedState prepare(const ParamSnapshot& input) {
        return JuicerTestSupport::prepare_product(input);
    }
    Json complete_result(PreparedState& product) {
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
    Json complete_result(PreparedState&& product) {
        return complete_result(product);
    }
    Json transform_result(int space) {
        Gamut::OutputGamutTransform out;
        std::string diagnostic;
        require(JuicerGamut::output_transform(static_cast<OutputEncoding::ColorSpace>(space), out, diagnostic), diagnostic);
        return {{"entry", bits(out.nativeRgbToD65Xyz)}, {"exit", bits(out.d65XyzToNativeRgb)}, {"hash", out.hash}, {"contract_hash", out.contractHash}};
    }
    Json table_result(int space) {
        Gamut::OutputGamutTransform transform;
        Gamut::OutputBoundaryTable table;
        std::string diagnostic;
        require(JuicerGamut::output_transform(static_cast<OutputEncoding::ColorSpace>(space), transform, diagnostic), diagnostic);
        require(JuicerGamut::output_boundary(transform, table), table.diagnostic);
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
                return fj_legacy_cat16_matrix(white.data(), white.data(), matrix.data());
            case FJ_TEST_CAT16_ADAPT:
                return fj_legacy_adapt_cat16(white.data(), white.data(), white.data(), adapted.data());
            case FJ_TEST_CAT02_MATRIX:
                return fj_legacy_cat02_matrix(white.data(), white.data(), matrix.data());
            case FJ_TEST_CAT02_ADAPT:
                return fj_test_adapt_cat02(white.data(), white.data(), white.data(), adapted.data());
            case FJ_TEST_INPUT_MATRICES:
                return fj_legacy_input_matrices(FJ_INPUT_DWG, &matrices);
            case FJ_TEST_INPUT_TO_DWG:
                return fj_test_input_to_dwg(&input, white.data(), 1, rgb.data(), adapted.data());
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                return fj_test_input_to_linear_srgb(&input, white.data(), identity.data(), rgb.data(), adapted.data());
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                return fj_legacy_linear_srgb_to_xyz(white.data(), adapted.data());
            case FJ_TEST_DWG_TO_XYZ:
                return fj_test_dwg_to_xyz(white.data(), adapted.data());
            case FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ:
                return fj_legacy_project_linear_rgb_to_xyz(white.data(), identity.data(), identity.data(), adapted.data());
            default:
                throw std::runtime_error("unknown color operation");
        }
    }
    thread_local unsigned constructionFault = 0;
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
            constructionFault = 0;
            set_pending_capture_test_hook(nullptr, nullptr);
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
                    return bits(JuicerColor::cat16_matrix(whites));
                case FJ_TEST_CAT16_ADAPT:
                    return bits(JuicerColor::adapt_cat16(xyz, whites));
                case FJ_TEST_CAT02_MATRIX:
                    return bits(JuicerColor::cat02_matrix(whites));
                default:
                    throw std::runtime_error("unknown native color fault operation");
            }
        };
        const Json expected = invoke();
        for (const auto fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
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
        }
    }
    void leaf_tests(const Json& fixture) {
        bool scalarDistinct = false;
        for (const auto& row : fixture.at("leaf")) {
            const auto& expected = row.at("expected");
            const auto actual = leaf_result(row);
            const std::string id = row.at("id").get<std::string>();
            require(same_values(actual.at("matrix"), expected.at("matrix")), id + " facade matrix bits/classification");
            require(same_values(actual.at("adapted"), expected.at("adapted")), id + " facade scalar bits/classification");
            const auto source = triplet(row.at("source"));
            const auto destination = triplet(row.at("destination"));
            const auto xyz = triplet(row.at("xyz"));
            const Spectral::ChromaticAdaptationWhites whites{source.data(), destination.data()};
            const auto matrix = JuicerColor::cat16_matrix(whites);
            const auto adapted = JuicerColor::adapt_cat16(xyz, whites);
            require(same_values(bits(matrix), expected.at("matrix")), id + " production matrix bits/classification");
            require(same_values(bits(adapted), expected.at("adapted")), id + " production scalar bits/classification");
            std::array<float, 3> multiplied{};
            multiplied = multiply_matrix(matrix.data(), xyz);
            scalarDistinct = scalarDistinct || !same_values(bits(multiplied), expected.at("adapted"));
        }
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
        for (std::uint32_t operation : {FJ_TEST_CAT16_MATRIX, FJ_TEST_CAT16_ADAPT}) {
            expect_status(arm_color_fault(operation, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation == 1 ? 2 : 1), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            for (const auto& invalid : std::array<std::array<std::uint32_t, 3>, 3>{{{0, 1, 1}, {operation, 0, 1}, {operation, 1, 99}}}) {
                expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(arm_color_fault(invalid[0], invalid[1], invalid[2]), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            }
            expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            expect_status(arm_color_fault(operation, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
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
        std::puts("CAT16: 933 independent scalar/matrix cases, C ABI, fault and comparator controls passed");
    }
    void pending(InstanceState& state, const ParamSnapshot& controls) {
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.value = PendingParamsState::Valid{controls, hash_params(controls)};
    }
    bool admitted(const PendingRenderAdmissionResult& result, bool print) {
        return result.status == (print ? PendingRenderAdmissionStatus::AdmittedPrint : PendingRenderAdmissionStatus::AdmittedDirect) && (print ? bool(result.printState) : bool(result.directState));
    }
    void unchanged(const InstanceState& state, const PendingRenderAdmissionResult& old, std::uint64_t hash, std::uint64_t counter) {
        require(JuicerAtomic::load_shared_ptr(&state.activeDirectState) == old.directState && JuicerAtomic::load_shared_ptr(&state.activePrintState) == old.printState, "exception preserves publications");
        require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "exception preserves hash/counter");
    }
    struct FaultCase {
        std::uint32_t operation;
        std::uint32_t callIndex;
        std::uint32_t fault;
    };
    enum class FailureStage : std::uint8_t {
        Leaf,
        FilmReference,
        ScannerColor
    };
    void failed(const PendingRenderAdmissionResult& result, const FaultCase& selected, const std::string& film = "kodak_portra_400", FailureStage stage = FailureStage::Leaf) {
        require(result.status == PendingRenderAdmissionStatus::RebuildFailed && !result.directState && !result.printState, "current exceptional attempt fails with empty result owners");
        require(result.diagnostic.find(selected.fault == 1 ? "UnsupportedInput" : "InternalFailure") != std::string::npos, "original category diagnostic");
        const char* operation = nullptr;
        switch (selected.operation) {
            case FJ_TEST_CAT16_MATRIX:
                operation = "CAT16 matrix";
                break;
            case FJ_TEST_CAT16_ADAPT:
                operation = "CAT16 scalar";
                break;
            case FJ_TEST_CAT02_MATRIX:
                operation = "CAT02 matrix";
                break;
            case FJ_TEST_CAT02_ADAPT:
                operation = "CAT02 scalar";
                break;
            case FJ_TEST_INPUT_MATRICES:
                operation = "Input matrix";
                break;
            case FJ_TEST_INPUT_TO_DWG:
                operation = "Input to DWG";
                break;
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                operation = "Input to linear sRGB";
                break;
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                operation = "Linear sRGB to XYZ";
                break;
            case FJ_TEST_DWG_TO_XYZ:
                operation = "DWG to XYZ";
                break;
            case FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ:
                operation = "Linear RGB to XYZ projection";
                break;
            default:
                throw std::runtime_error("unknown failure operation");
        }
        if (selected.fault == FJ_TEST_COLOR_PANIC) {
            if (stage == FailureStage::ScannerColor)
                operation = "scanner color preparation panicked";
            else if (stage == FailureStage::FilmReference)
                operation = "film reference preparation panicked";
        }
        require(result.diagnostic.find(operation) != std::string::npos, "stage-specific operation diagnostic");
        require(result.diagnostic.find("film=" + film) != std::string::npos && result.diagnostic.find("route=") != std::string::npos, "attempted selection diagnostic");
    }
    struct Supersession {
        ParamSnapshot newer;
        int mode = 0;
        bool fired = false;
    };
    void supersede(InstanceState& state, void* context) {
        auto& selection = *static_cast<Supersession*>(context);
        if (std::exchange(selection.fired, true)) {
            return;
        }
        std::lock_guard<std::mutex> lock(state.pending.m);
        if (selection.mode == 1) {
            state.pending.value = PendingParamsState::InvalidSnapshotControls{"new invalid controls"};
        } else if (selection.mode == 2) {
            state.pending.value = PendingParamsState::Uninitialized{};
        } else {
            state.pending.value = PendingParamsState::Valid{selection.newer, hash_params(selection.newer)};
        }
    }
    void admission_tests() {
        std::size_t failures = 0;
        std::size_t supersessions = 0;
        for (int route = 0; route < 2; ++route) {
            for (int method : {0, 2}) {
                const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                auto changed = controls;
                changed.cameraExposureCompensationEv = 1.0;
                for (std::uint32_t operation : {FJ_TEST_CAT16_MATRIX, FJ_TEST_CAT16_ADAPT}) {
                    const std::uint32_t calls = operation == 1 ? 1 : 2;
                    for (std::uint32_t index = 1; index <= calls; ++index) {
                        for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                            const FaultCase selected{operation, index, fault};
                            FaultScope cleanup;
                            InstanceState state;
                            pending(state, controls);
                            const auto old = admit_pending_render_state(state);
                            require(admitted(old, route == 1), "initial state admitted");
                            const auto hash = state.lastHash.load();
                            const auto counter = state.buildCounterNext.load();
                            const auto oldRecipe = route == 1 ? old.printState->recipe.hash : old.directState->recipe.hash;
                            pending(state, changed);
                            expect_status(arm_color_fault(selected.operation, selected.callIndex, selected.fault), FJ_STATUS_SUCCESS);
                            const auto result = admit_pending_render_state(state);
                            failed(result, selected, controls.filmProfileKey, operation == FJ_TEST_CAT16_ADAPT ? FailureStage::FilmReference : FailureStage::Leaf);
                            unchanged(state, old, hash, counter);
                            pending(state, controls);
                            const auto reused = admit_pending_render_state(state);
                            require(reused.directState == old.directState && reused.printState == old.printState, "revert to matching retained publication reuses owner");
                            unchanged(state, old, hash, counter);
                            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                            pending(state, changed);
                            const auto recovered = admit_pending_render_state(state);
                            require(admitted(recovered, route == 1), "recovery admitted");
                            require(state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "recovery publication/counter");
                            require((route == 1 ? old.printState->recipe.hash : old.directState->recipe.hash) == oldRecipe, "admitted old hold stays readable");
                            ++failures;
                            for (int mode = 0; mode < 3; ++mode) {
                                // The newer valid snapshot reuses the current successful input,
                                // isolating the obsolete fault from any successful publication.
                                Supersession selection{changed, mode, false};
                                const auto liveHash = state.lastHash.load();
                                const auto liveCounter = state.buildCounterNext.load();
                                pending(state, controls);
                                set_pending_capture_test_hook(supersede, &selection);
                                expect_status(arm_color_fault(selected.operation, selected.callIndex, selected.fault), FJ_STATUS_SUCCESS);
                                const auto newer = admit_pending_render_state(state);
                                require(selection.fired, "captured snapshot superseded");
                                if (mode == 0) {
                                    require(admitted(newer, route == 1) && newer.directState == recovered.directState && newer.printState == recovered.printState, "superseded valid attempt retries current publication");
                                } else {
                                    require(newer.status == (mode == 1 ? PendingRenderAdmissionStatus::InvalidSnapshotControls : PendingRenderAdmissionStatus::NeedsSnapshotAcquisition) && !newer.directState && !newer.printState, "superseded invalid/uninitialized normal acquisition");
                                }
                                unchanged(state, recovered, liveHash, liveCounter);
                                // N2: same admission thread, valid storage, original armed count.
                                // An unconsumed countdown would fire on one of these calls.
                                for (std::uint32_t call = 0; call < index; ++call) {
                                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                }
                                set_pending_capture_test_hook(nullptr, nullptr);
                                expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                                ++supersessions;
                            }
                        }
                    }
                }
            }
        }
        std::printf("Admission: %zu current failures/recoveries and %zu N2 supersession witnesses passed\n", failures, supersessions);
    }
    void construction_controls() {
        for (int route = 0; route < 2; ++route) {
            InstanceState state;
            const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Hanatos2025);
            pending(state, controls);
            const auto old = admit_pending_render_state(state);
            require(admitted(old, route == 1), "complete-state baseline");
            auto changed = controls;
            changed.cameraExposureCompensationEv = 1.0;
            const auto hash = state.lastHash.load();
            const auto counter = state.buildCounterNext.load();
            changed.filmProfileKey = "missing-cat16-control-profile";
            pending(state, changed);
            const auto ordinary = admit_pending_render_state(state);
            require(ordinary.status == PendingRenderAdmissionStatus::RebuildFailed && !ordinary.directState && !ordinary.printState, "ordinary false control");
            require(route == 1 ? !JuicerAtomic::load_shared_ptr(&state.activePrintState) : !JuicerAtomic::load_shared_ptr(&state.activeDirectState), "ordinary false keeps publication clearing");
            require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "ordinary false counter/hash");
            pending(state, controls);
            require(admitted(admit_pending_render_state(state), route == 1), "ordinary false recovery");
        }
        std::puts("Construction: actual missing-profile failure clears only the selected route and recovers");
    }
    void preparation_tests(const Json& fixture) {
        auto identity = fixture.at("transforms").at(0);
        identity["hash"] = identity["hash"].get<std::uint64_t>() ^ 1;
        require(identity != fixture.at("transforms").at(0), "identity negative control");
        for (int space = 0; space < 9; ++space) {
            require(transform_result(space) == fixture.at("transforms").at(space), "native-parent output transform/identity space=" + std::to_string(space));
        }
        for (const auto& row : fixture.at("tables")) {
            require(table_result(row.at("space").get<int>()) == row.at("expected"), "complete Cmax/table/hash");
        }
        const auto cases = scenarios();
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [id, controls] = cases[i];
            auto product = JuicerTestSupport::prepare_product(controls);
            require(id == fixture.at("recipes").at(i).at("id").get<std::string>(), "fixed recipe case membership");
            require(complete_result(product) == fixture.at("recipes").at(i).at("expected"), id + " independent complete arrays/identities");
            auto repeated = JuicerTestSupport::prepare_product(controls);
            require(complete_result(repeated) == complete_result(product), id + " repeat complete construction");
        }
        for (int route = 0; route < 2; ++route) {
            const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Hanatos2025);
            auto product = JuicerTestSupport::prepare_product(controls);
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            FjPreparedStateView productView{};
            require(product.view(productView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            auto changed = controls;
            changed.outputCctfEncoding = 0;
            auto encodedOff = JuicerTestSupport::prepare_product(changed);
            FjPreparedStateView encodedOffView{};
            require(encodedOff.view(encodedOffView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(bits(productView.film_exposure.input_xyz_adapt, 9) == bits(encodedOffView.film_exposure.input_xyz_adapt, 9) && productView.film_exposure.tc_lut_hash == encodedOffView.film_exposure.tc_lut_hash, "encoding change preserves film/TC family");
            require(productView.output_boundary.entry_identity != 0 && productView.output_boundary.entry_identity == encodedOffView.output_boundary.entry_identity && productView.output_boundary.cmax.data == encodedOffView.output_boundary.cmax.data, "encoding change reuses actual boundary owner");
            require(productView.scanner_lut.hash == encodedOffView.scanner_lut.hash, "encoding change preserves scanner descriptor identity");
            require(productView.scanner_recipe_hash != encodedOffView.scanner_recipe_hash && productView.recipe_hash != encodedOffView.recipe_hash && productView.scanner_color.hash != encodedOffView.scanner_color.hash, "encoding identities change at actual owners");
            const auto retainedBits = bits(productView.output_boundary.cmax.data, productView.output_boundary.cmax.count);
            const auto retainedIdentity = productView.output_boundary.entry_identity;
            const auto* retainedBacking = productView.output_boundary.cmax.data;
            expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            require(bits(productView.output_boundary.cmax.data, productView.output_boundary.cmax.count) == retainedBits && complete_result(product) == complete_result(JuicerTestSupport::prepare_product(controls)), "retained state/table stays readable across cache release");
            auto rebuilt = JuicerTestSupport::prepare_product(controls);
            FjPreparedStateView rebuiltView{};
            require(rebuilt.view(rebuiltView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(rebuiltView.output_boundary.entry_identity != retainedIdentity && rebuiltView.output_boundary.cmax.data != retainedBacking && bits(rebuiltView.output_boundary.cmax.data, rebuiltView.output_boundary.cmax.count) == retainedBits, "cache release creates new equal owner without mutating retained table");
            InstanceState state;
            pending(state, controls);
            const auto old = admit_pending_render_state(state);
            const auto counter = state.buildCounterNext.load();
            FaultScope cleanup;
            expect_status(arm_color_fault(FJ_TEST_CAT16_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            const auto repeated = admit_pending_render_state(state);
            require(repeated.directState == old.directState && repeated.printState == old.printState && state.buildCounterNext.load() == counter, "unchanged admission reuses actual state");
            expect_status(raw_call(FJ_TEST_CAT16_MATRIX), FJ_STATUS_UNSUPPORTED_INPUT);
        }
        admission_tests();
        construction_controls();
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
            const auto matrix = JuicerColor::cat02_matrix(whites);
            std::array<float, 3> adapted{};
            expect_status(fj_test_adapt_cat02(xyz.data(), whites.source, whites.destination, adapted.data()), FJ_STATUS_SUCCESS);
            return {{"matrix", bits(matrix)}, {"adapted", bits(adapted)}};
        }
        ParamSnapshot snapshot(Spektrafilm::ScanRoute route, Spektrafilm::RgbToRawMethod method) {
            ParamSnapshot out;
            out.filmProfileKey = Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Negative ? "kodak_portra_400" : "fujifilm_provia_100f";
            out.printProfileKey = "kodak_portra_endura";
            out.scanRoute = route;
            out.spectralUpsamplingMode = static_cast<int>(method);
            out.cameraExposureCompensationEv = 0.75;
            out.cameraAutoExposureEnabled = 0;
            out.inputCompressionEnabled = 0;
            return out;
        }
        std::vector<std::pair<std::string, ParamSnapshot>> scenarios() {
            std::vector<std::pair<std::string, ParamSnapshot>> out;
            for (int polarity = 0; polarity < 2; ++polarity) {
                for (int route = 0; route < 2; ++route) {
                    const auto scanRoute = static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route);
                    for (int input = 0; input < 4; ++input) {
                        for (int decoding = 0; decoding < 2; ++decoding) {
                            auto controls = snapshot(scanRoute, Spektrafilm::RgbToRawMethod::Mallett2019);
                            controls.inputColorSpace = input;
                            controls.inputCctfDecoding = decoding;
                            out.emplace_back("mallett-" + std::to_string(polarity) + "-" + std::to_string(route) + "-input-" + std::to_string(input) + "-decode-" + std::to_string(decoding), controls);
                        }
                    }
                    for (int method : {0, 2})
                        out.emplace_back("tc-" + std::to_string(polarity) + "-" + std::to_string(route) + "-method-" + std::to_string(method), snapshot(scanRoute, static_cast<Spektrafilm::RgbToRawMethod>(method)));
                    for (int method = 0; method < 3; ++method) {
                        auto controls = snapshot(scanRoute, static_cast<Spektrafilm::RgbToRawMethod>(method));
                        controls.outputCctfEncoding = 0;
                        out.emplace_back("encoding-off-" + std::to_string(polarity) + "-" + std::to_string(route) + "-method-" + std::to_string(method), controls);
                    }
                    auto controls = snapshot(scanRoute, Spektrafilm::RgbToRawMethod::Mallett2019);
                    controls.outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
                    out.emplace_back("output-dci-" + std::to_string(polarity) + "-" + std::to_string(route), controls);
                }
            }
            return out;
        }
        PreparedState prepare(const ParamSnapshot& input) {
            return JuicerTestSupport::prepare_product(input);
        }
        Json color_result(const Scanner::ColorRuntime& color) {
            return {{"cat02", bits(color.cat02)}, {"xyz_to_rgb", bits(color.xyzToRgb)}, {"illuminant_xyz", bits(color.illuminantXYZ)}, {"encoding", {static_cast<int>(color.encoding.colorSpace), color.encoding.applyCctfEncoding, color.encoding.inputIsOutputSpace}}, {"gamut_recipe_hash", color.outputGamutRecipeHash}, {"hash", color.hash}};
        }
        Json color_result(const FjScannerColor& color) {
            return {{"cat02", bits(color.cat02, 9)}, {"xyz_to_rgb", bits(color.xyz_to_rgb, 9)}, {"illuminant_xyz", bits(color.illuminant_xyz, 3)}, {"encoding", {color.output_space, (color.flags & 1u) != 0, (color.flags & 2u) != 0}}, {"gamut_recipe_hash", color.active_gamut_recipe_hash}, {"hash", color.hash}};
        }
        Json tables_result(const Spectral::SpectralTables& tables) {
            return {{"lambda", bits(tables.lambda)}, {"K", tables.K}, {"delta_lambda", bits(std::array{tables.deltaLambda})}, {"inv_yn", bits(std::array{tables.invYn})}, {"white_xyz", bits(tables.whiteXYZ)}, {"ref_illum_white_xyz", bits(tables.refIllumWhiteXYZ)}, {"ax", bits(tables.Ax)}, {"ay", bits(tables.Ay)}, {"az", bits(tables.Az)}, {"xbar", bits(tables.Xbar)}, {"ybar", bits(tables.Ybar)}, {"zbar", bits(tables.Zbar)}, {"illum", bits(tables.illum)}, {"eps_c", bits(tables.epsC)}, {"eps_m", bits(tables.epsM)}, {"eps_y", bits(tables.epsY)}, {"base_density_min", bits(tables.baseDensityMin)}, {"base_density_mid", bits(tables.baseDensityMid)}, {"has_baseline", tables.hasBaseline}, {"baseline_mix", bits(std::array{tables.densityBaselineMixReference})}, {"illuminant_hash", tables.illuminantHash}, {"tables_hash", tables.tablesHash}};
        }
        Json tables_result(const FjPreparedSpectralView& tables) {
            const std::array<float, 1> zeros{};
            return {{"lambda", bits(tables.lambda_nm.data, tables.lambda_nm.count)}, {"K", tables.lambda_nm.count}, {"delta_lambda", bits(std::array{5.0f})}, {"inv_yn", bits(std::array{tables.inv_yn})}, {"white_xyz", bits(tables.white_xyz, 3)}, {"ref_illum_white_xyz", bits(tables.white_xyz, 3)}, {"ax", bits(tables.weighted_xyz[0].data, tables.weighted_xyz[0].count)}, {"ay", bits(tables.weighted_xyz[1].data, tables.weighted_xyz[1].count)}, {"az", bits(tables.weighted_xyz[2].data, tables.weighted_xyz[2].count)}, {"xbar", bits(tables.observer_xyz[0].data, tables.observer_xyz[0].count)}, {"ybar", bits(tables.observer_xyz[1].data, tables.observer_xyz[1].count)}, {"zbar", bits(tables.observer_xyz[2].data, tables.observer_xyz[2].count)}, {"illum", bits(tables.illuminant.data, tables.illuminant.count)}, {"eps_c", bits(tables.dyes_cmy[0].data, tables.dyes_cmy[0].count)}, {"eps_m", bits(tables.dyes_cmy[1].data, tables.dyes_cmy[1].count)}, {"eps_y", bits(tables.dyes_cmy[2].data, tables.dyes_cmy[2].count)}, {"base_density_min", bits(tables.baseline_min.data, tables.baseline_min.count)}, {"base_density_mid", bits(tables.baseline_mid.data, tables.baseline_mid.count)}, {"has_baseline", tables.has_baseline != 0}, {"baseline_mix", bits(zeros)}, {"illuminant_hash", tables.illuminant_hash}, {"tables_hash", tables.hash}};
        }
        // Temporary I4 publication observer; switch this to holder-backed projections in I5.
        template <typename State>
        Json scanner_descriptor(const State& product) {
            Scanner::ScannerSpectralLutDescriptor out;
            std::string diagnostic;
            const auto& recipe = product.recipe;
            const bool ok = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic) : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic);
            require(ok, diagnostic);
            return {{"route", static_cast<int>(out.route)}, {"medium", static_cast<int>(out.medium)}, {"polarity", static_cast<int>(out.polarity)}, {"density_bounds_hash", out.densityBoundsHash}, {"channel_density_hash", out.channelDensityHash}, {"base_density_hash", out.baseDensityHash}, {"illuminant_hash", out.scanIlluminantHash}, {"observer_hash", out.observerHash}, {"resolution", out.lutResolution}, {"hash", out.hash}};
        }
        // Temporary I4 publication observer; switch this to holder-backed projections in I5.
        template <typename State>
        Json complete_result(const State& product) {
            const auto& recipe = product.recipe;
            const auto& raw = recipe.filmRaw;
            const auto& reference = recipe.scannerOutput.syntheticFilmReference;
            const auto& balance = recipe.print.balance;
            const auto& config = product.payload.filmRawConfig;
            return {{"input_rgb_to_xyz", bits(raw.inputRgbToXyz)}, {"input_adapt", bits(raw.inputXyzAdapt)}, {"xyz_to_linear_srgb", bits(raw.xyzToLinearSrgb)}, {"source_white", bits(raw.inputNominalWhiteXYZ)}, {"projection_white", bits(raw.projectionWhiteXYZ)}, {"config_rgb_to_xyz", bits(config.inputRGBToXYZ.m)}, {"config_adapt", bits(config.inputXYZAdapt.m)}, {"raw_midgray", bits(config.rawMidgray)}, {"midgray_dwg", bits(config.midgrayDWG)}, {"midgray_scale_green", bits(std::array{config.midgrayScale, config.rawMidgrayGreen})}, {"config_whites", {bits(config.inputWhiteXYZ), bits(config.workingWhiteXYZ), bits(config.refIllumWhiteXYZ)}}, {"config_flags", {static_cast<int>(config.inputColorSpace), config.applyCctfDecoding, static_cast<int>(config.spectralUpsamplingMode), config.applyInputChromaticAdapt, config.hasRefIllumWhite, config.valid}}, {"spd_s_inv", bits(product.payload.spdSInv)}, {"baseline_raw", bits(reference.baselineRawRgb)}, {"compensated_raw", bits(reference.compensatedRawRgb)}, {"baseline_density", bits(reference.baselineDensityCmy)}, {"compensated_density", bits(reference.compensatedDensityCmy)}, {"print_baseline_raw", bits(balance.baselinePrintRawRgb)}, {"print_compensated_raw", bits(balance.compensatedPrintRawRgb)}, {"print_factors", bits(std::array{balance.factorMidgray, balance.factorMidgrayComp, balance.normalizer})}, {"hashes", {raw.hash, raw.tcLutHash, reference.hash, balance.hash, recipe.print.hash, recipe.scannerOutput.hash, recipe.scannerOutput.outputGamut.hash, recipe.hash, product.payload.uploadCoreHash, product.payload.scannerHash}}, {"scanner_color", color_result(product.payload.scannerColor)}, {"scanner_tables", tables_result(product.payload.scannerTables)}, {"scanner_descriptor", scanner_descriptor(product)}, {"gamut_entry", bits(recipe.scannerOutput.outputGamut.transform.nativeRgbToD65Xyz)}, {"gamut_exit", bits(recipe.scannerOutput.outputGamut.transform.d65XyzToNativeRgb)}};
        }
        Json scanner_descriptor(const FjScannerLutDescriptor& descriptor) {
            return {{"route", static_cast<int>(descriptor.route)}, {"medium", static_cast<int>(descriptor.medium)}, {"polarity", static_cast<int>(descriptor.capture_polarity)}, {"density_bounds_hash", descriptor.density_bounds_hash}, {"channel_density_hash", descriptor.channel_density_hash}, {"base_density_hash", descriptor.base_density_hash}, {"illuminant_hash", descriptor.viewing_illuminant_hash}, {"observer_hash", descriptor.observer_hash}, {"resolution", descriptor.resolution}, {"hash", descriptor.hash}};
        }
        Json complete_result(PreparedState& product) {
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
        Scanner::ScannerIlluminant scanner_illuminant(PreparedState& product) {
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
        std::string viewing_illuminant(const FjPreparedStateView& prepared) {
            auto* assets = JuicerProcess::root().assets().borrow_rust_assets();
            require(assets != nullptr, "scanner illuminant source has process Rust assets");
            std::string result;
            if ((prepared.route & 1u) != 0) {
                FjPrintProfile* owner = nullptr;
                expect_status(fj_legacy_print_profile_acquire(assets, prepared.print_profile_key, &owner, nullptr), FJ_STATUS_SUCCESS);
                FjPrintProfileView view{};
                expect_status(fj_legacy_print_profile_view(owner, &view, nullptr), FJ_STATUS_SUCCESS);
                require(view.asset_token == prepared.print_profile_asset_version, "scanner illuminant print source token matches held state");
                result.assign(view.viewing_illuminant.label.data, view.viewing_illuminant.label.count);
                expect_status(fj_legacy_print_profile_release(owner, nullptr), FJ_STATUS_SUCCESS);
            } else {
                FjFilmProfile* owner = nullptr;
                expect_status(fj_legacy_film_profile_acquire(assets, prepared.film_profile_key, &owner, nullptr), FJ_STATUS_SUCCESS);
                FjFilmProfileView view{};
                expect_status(fj_legacy_film_profile_view(owner, &view, nullptr), FJ_STATUS_SUCCESS);
                require(view.asset_token == prepared.film_profile_asset_version, "scanner illuminant film source token matches held state");
                result.assign(view.viewing_illuminant.label.data, view.viewing_illuminant.label.count);
                expect_status(fj_legacy_film_profile_release(owner, nullptr), FJ_STATUS_SUCCESS);
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
                return fj_legacy_cat16_matrix(pointer(0), pointer(1), output(2));
            case FJ_TEST_CAT02_MATRIX:
                return fj_legacy_cat02_matrix(pointer(0), pointer(1), output(2));
            case FJ_TEST_CAT16_ADAPT:
                return fj_legacy_adapt_cat16(pointer(0), pointer(1), pointer(2), output(3));
            case FJ_TEST_CAT02_ADAPT:
                return fj_test_adapt_cat02(pointer(0), pointer(1), pointer(2), output(3));
            case FJ_TEST_INPUT_MATRICES:
                return fj_legacy_input_matrices(FJ_INPUT_DWG, selected.slot == 0 ? nullptr : &matrices);
            case FJ_TEST_INPUT_TO_DWG:
                return fj_test_input_to_dwg(conversion, pointer(1), 1, selected.slot == 2 ? nullptr : rgb.data(), output(3));
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                return fj_test_input_to_linear_srgb(conversion, pointer(1), selected.slot == 2 ? nullptr : identity.data(), selected.slot == 3 ? nullptr : rgb.data(), output(4));
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                return fj_legacy_linear_srgb_to_xyz(pointer(0), output(1));
            case FJ_TEST_DWG_TO_XYZ:
                return fj_test_dwg_to_xyz(pointer(0), output(1));
            case FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ:
                return fj_legacy_project_linear_rgb_to_xyz(pointer(0), selected.slot == 1 ? nullptr : identity.data(), selected.slot == 2 ? nullptr : identity.data(), output(3));
            default:
                throw std::runtime_error("unknown null operation");
        }
    }
    void color_fault_tests() {
        FaultScope cleanup;
        constexpr std::array<std::uint32_t, 10> operations{FJ_TEST_CAT16_MATRIX, FJ_TEST_CAT16_ADAPT, FJ_TEST_CAT02_MATRIX, FJ_TEST_CAT02_ADAPT, FJ_TEST_INPUT_MATRICES, FJ_TEST_INPUT_TO_DWG, FJ_TEST_INPUT_TO_LINEAR_SRGB, FJ_TEST_LINEAR_SRGB_TO_XYZ, FJ_TEST_DWG_TO_XYZ, FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ};
        for (const auto operation : operations) {
            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                expect_status(arm_color_fault(operation, 2, fault), FJ_STATUS_SUCCESS);
                for (const auto unrelated : operations) {
                    if (unrelated != operation) {
                        expect_status(raw_call(unrelated), FJ_STATUS_SUCCESS);
                    }
                }
                if (!color_facade_operation(operation)) {
                    expect_status(facade_call(operation), FJ_STATUS_SUCCESS);
                }
                const int pointers = color_pointer_count(operation);
                for (int slot = 0; slot < pointers; ++slot) {
                    expect_status(null_color_call({.operation = operation, .slot = slot}), FJ_STATUS_UNSUPPORTED_INPUT);
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
            }
            for (const auto& invalid : std::array<std::array<std::uint32_t, 3>, 5>{{{0, 1, 1}, {FJ_TEST_OUTPUT_COLOR_FACADE + 1U, 1, 1}, {99, 1, 1}, {operation, 0, 1}, {operation, 1, 99}}}) {
                expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(arm_color_fault(invalid[0], invalid[1], invalid[2]), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            }
            for (const auto replacement : operations) {
                expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(arm_color_fault(replacement, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
                if (replacement != operation) {
                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                }
                expect_status(raw_call(replacement), FJ_STATUS_SUCCESS);
                expect_status(raw_call(replacement), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(replacement), FJ_STATUS_SUCCESS);
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
                const auto source = triplet(row.at("source"));
                const auto destination = triplet(row.at("destination"));
                const auto xyz = triplet(row.at("xyz"));
                const Spectral::ChromaticAdaptationWhites whites{source.data(), destination.data()};
                const auto& expected = row.at("expected");
                const std::string id = row.at("id").get<std::string>();
                const auto matrix = JuicerColor::cat02_matrix(whites);
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
            }
        }
        require(scalarDistinct, "independent CAT02 captures distinguish scalar from matrix shortcut");
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
    template <typename State>
    RetainedValues retain_values(const State& state) {
        RetainedValues out;
        out.complete = Cat02Fixtures::complete_result(state);
        out.exposureTables = Cat02Fixtures::tables_result(state.payload.exposureTables);
        if (state.payload.filmTcLut) {
            out.tcRgba = retained_bits(state.payload.filmTcLut->samples());
        }
        if (state.payload.printMainIlluminant) {
            out.printIlluminant = retained_bits(*state.payload.printMainIlluminant);
        }
        require(bool(state.payload.outputBoundaryTable), "completed state boundary owner");
        const auto& boundary = *state.payload.outputBoundaryTable;
        require(boundary.entryIdentity != 0 && boundary.cmax.size() == Gamut::kOutputBoundaryValueCount, "retained completed boundary owner and extent");
        out.boundary = retained_bits(boundary.cmax);
        out.boundaryHashes = {boundary.transformHash, boundary.contractHash, boundary.hash};
        out.boundaryEntryIdentity = boundary.entryIdentity;
        out.boundaryBacking = boundary.cmax.data();
        return out;
    }
    RetainedValues retain_values(PreparedState& state) {
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
    template <typename State>
    void require_retained_values(const State& state, const RetainedValues& expected) {
        const auto complete = Cat02Fixtures::complete_result(state);
        const auto exposureTables = Cat02Fixtures::tables_result(state.payload.exposureTables);
        const auto& payload = state.payload;
        const bool tcExact = payload.filmTcLut ? same_retained_bits(payload.filmTcLut->samples(), expected.tcRgba) : expected.tcRgba.empty();
        const bool printExact = payload.printMainIlluminant ? same_retained_bits(*payload.printMainIlluminant, expected.printIlluminant) : expected.printIlluminant.empty();
        require(bool(payload.outputBoundaryTable), "completed state boundary owner");
        const auto& boundary = *payload.outputBoundaryTable;
        require(boundary.entryIdentity != 0 && boundary.cmax.size() == Gamut::kOutputBoundaryValueCount, "retained completed boundary owner and extent");
        const std::array<std::uint64_t, 3> boundaryHashes{boundary.transformHash, boundary.contractHash, boundary.hash};
        require(complete == expected.complete && exposureTables == expected.exposureTables && tcExact && printExact && same_retained_bits(boundary.cmax, expected.boundary) && boundaryHashes == expected.boundaryHashes && boundary.entryIdentity == expected.boundaryEntryIdentity && boundary.cmax.data() == expected.boundaryBacking, "retained complete route/color/tables/TC/boundary values remain exact");
    }
    void require_retained_values(PreparedState& state, const RetainedValues& expected) {
        const auto actual = retain_values(state);
        require(actual.complete == expected.complete && actual.exposureTables == expected.exposureTables && actual.tcRgba == expected.tcRgba && actual.printIlluminant == expected.printIlluminant && actual.boundary == expected.boundary && actual.boundaryHashes == expected.boundaryHashes && actual.boundaryEntryIdentity == expected.boundaryEntryIdentity && actual.boundaryBacking == expected.boundaryBacking, "retained complete route/color/tables/TC/boundary values remain exact");
    }
    RetainedValues retain_values(const PendingRenderAdmissionResult& result) {
        if (result.printState) {
            return retain_values(*result.printState);
        }
        require(bool(result.directState), "admitted retained state owner");
        return retain_values(*result.directState);
    }
    void require_retained_values(const PendingRenderAdmissionResult& result, const RetainedValues& expected) {
        if (result.printState) {
            require_retained_values(*result.printState, expected);
        } else {
            require(bool(result.directState), "admitted retained state owner");
            require_retained_values(*result.directState, expected);
        }
    }
    void cat02_admission_tests() {
        std::size_t failures = 0;
        std::size_t supersessions = 0;
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                const auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                auto changed = controls;
                changed.cameraExposureCompensationEv = 1.0;
                const std::uint32_t calls = method == 1 ? 2 : 1;
                for (std::uint32_t index = 1; index <= calls; ++index) {
                    for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                        const FaultCase selected{FJ_TEST_CAT02_MATRIX, index, fault};
                        FaultScope cleanup;
                        InstanceState state;
                        pending(state, controls);
                        const auto old = admit_pending_render_state(state);
                        require(admitted(old, route % 2 == 1), "initial CAT02 route admitted");
                        const auto oldValues = retain_values(old);
                        const auto hash = state.lastHash.load();
                        const auto counter = state.buildCounterNext.load();
                        pending(state, changed);
                        expect_status(arm_color_fault(selected.operation, index, fault), FJ_STATUS_SUCCESS);
                        const auto result = admit_pending_render_state(state);
                        failed(result, selected, controls.filmProfileKey, index == calls ? FailureStage::ScannerColor : FailureStage::Leaf);
                        unchanged(state, old, hash, counter);
                        require_retained_values(old, oldValues);
                        for (std::uint32_t call = 0; call < index; ++call) {
                            expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_SUCCESS);
                        }
                        pending(state, controls);
                        const auto reused = admit_pending_render_state(state);
                        require(reused.directState == old.directState && reused.printState == old.printState, "failed CAT02 rebuild retains reusable accepted publication");
                        unchanged(state, old, hash, counter);
                        pending(state, changed);
                        const auto recovered = admit_pending_render_state(state);
                        require(admitted(recovered, route % 2 == 1), "CAT02 recovery admitted");
                        require(recovered.directState != old.directState || recovered.printState != old.printState, "CAT02 recovery replaces route publication");
                        require(state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "CAT02 recovery publishes matching hash/counter");
                        require_retained_values(old, oldValues);
                        const auto recoveredValues = retain_values(recovered);
                        ++failures;
                        for (int mode = 0; mode < 3; ++mode) {
                            Supersession selection{changed, mode, false};
                            const auto liveHash = state.lastHash.load();
                            const auto liveCounter = state.buildCounterNext.load();
                            pending(state, controls);
                            set_pending_capture_test_hook(supersede, &selection);
                            expect_status(arm_color_fault(selected.operation, index, fault), FJ_STATUS_SUCCESS);
                            const auto newer = admit_pending_render_state(state);
                            require(selection.fired, "CAT02 captured snapshot superseded");
                            if (mode == 0) {
                                require(admitted(newer, route % 2 == 1) && newer.directState == recovered.directState && newer.printState == recovered.printState, "CAT02 superseded valid attempt reuses current successful publication");
                            } else {
                                require(newer.status == (mode == 1 ? PendingRenderAdmissionStatus::InvalidSnapshotControls : PendingRenderAdmissionStatus::NeedsSnapshotAcquisition) && !newer.directState && !newer.printState, "CAT02 superseded invalid/uninitialized normal acquisition");
                            }
                            unchanged(state, recovered, liveHash, liveCounter);
                            require_retained_values(old, oldValues);
                            require_retained_values(recovered, recoveredValues);
                            // Same admission thread and original matching count, before cleanup.
                            for (std::uint32_t call = 0; call < index; ++call) {
                                expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_SUCCESS);
                            }
                            set_pending_capture_test_hook(nullptr, nullptr);
                            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                            ++supersessions;
                        }
                    }
                }
            }
        }
        std::printf("CAT02 admission: %zu current failure/recovery and %zu N2 supersession witnesses on four routes passed\n", failures, supersessions);
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
    void cat02_helper_tests(const Json& fixture) {
        auto product = Cat02Fixtures::prepare(Cat02Fixtures::snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Mallett2019));
        auto* assets = JuicerProcess::root().assets().borrow_rust_assets();
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
        require(Json{{"K", Spectral::gShape.K}, {"dwg_white", bits(Spectral::gDWG_WhitePoint_XYZ)}} == fixture.at("helper_hanatos_axes"), "independent retained Hanatos helper axes/white");
        const auto cases = Cat02Fixtures::helper_inputs();
        require(cases.size() == fixture.at("helpers").size(), "retained helper case membership");
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& input = cases[i];
            const auto& row = fixture.at("helpers").at(i);
            require(input.at("id") == row.at("id") && input.at("rgb") == row.at("rgb") && input.at("reference_white") == row.at("reference_white"), "frozen retained helper inputs");
            require(Cat02Fixtures::helper_result(input, tables, sInv, true, hanatosSamples) == row.at("hanatos_expected"), input.at("id").get<std::string>() + " signed Hanatos sanitation/spectrum");
            require(Cat02Fixtures::helper_result(input, tables, sInv, false, hanatosSamples) == row.at("tables_expected"), input.at("id").get<std::string>() + " tables sanitation/spectrum");
        }
        const std::array<float, 3> rgb{2.0f, -0.25f, 0.5f};
        const auto held = retain_values(product);
        for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
            std::array<float, 3> adapted{};
            require_color_facade_failure(FJ_TEST_CAT02_ADAPT, fault, [&] {
                return fj_test_adapt_cat02(rgb.data(), Spectral::gDWG_WhitePoint_XYZ, tables.white_xyz, adapted.data());
            });
            require_spectrum_fixture_failure(tables, sInv, 1, fault, hanatosSamples);
            require_spectrum_fixture_failure(tables, sInv, 2, fault, hanatosSamples);
            require_retained_values(product, held);
        }
        std::printf("CAT02 retained helpers: %zu independent pre/post sanitation/spectrum cases per helper and both typed failure categories passed\n", cases.size());
    }
    void cat02_control_tests() {
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                const auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                auto product = Cat02Fixtures::prepare(controls);
                std::array<char, 2048> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                FjPreparedStateView productView{};
                require(product.view(productView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                const auto productHeld = retain_values(product);
                auto decoding = controls;
                decoding.inputCctfDecoding = 1;
                auto decoded = Cat02Fixtures::prepare(decoding);
                FjPreparedStateView decodedView{};
                require(decoded.view(decodedView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                require(bits(productView.film_exposure.input_xyz_adapt, 9) == bits(decodedView.film_exposure.input_xyz_adapt, 9) && productView.film_exposure.hash != decodedView.film_exposure.hash && productView.recipe_hash != decodedView.recipe_hash, "input CCTF decoding keeps adaptation and changes film identity");
                auto input = controls;
                input.inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::ACES2065_1);
                auto inputChanged = Cat02Fixtures::prepare(input);
                FjPreparedStateView inputChangedView{};
                require(inputChanged.view(inputChangedView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                require(bits(productView.input_policy.nominal_white_xyz, 3) != bits(inputChangedView.input_policy.nominal_white_xyz, 3) && bits(productView.film_exposure.input_xyz_adapt, 9) != bits(inputChangedView.film_exposure.input_xyz_adapt, 9) && productView.film_exposure.hash != inputChangedView.film_exposure.hash && productView.recipe_hash != inputChangedView.recipe_hash, "input space/white changes contributing matrix and film identity");
                require(Cat02Fixtures::color_result(productView.scanner_color) == Cat02Fixtures::color_result(inputChangedView.scanner_color), "input space leaves selected scanner color values/identity unchanged");
                auto output = controls;
                output.outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
                auto outputChanged = Cat02Fixtures::prepare(output);
                FjPreparedStateView outputChangedView{};
                require(outputChanged.view(outputChangedView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                require(bits(productView.scanner_color.cat02, 9) != bits(outputChangedView.scanner_color.cat02, 9) && productView.scanner_color.hash != outputChangedView.scanner_color.hash && productView.recipe_hash != outputChangedView.recipe_hash, "output space/white changes scanner matrix/color/final identity");
                require(productView.film_exposure.hash == outputChangedView.film_exposure.hash && bits(productView.film_exposure.input_xyz_adapt, 9) == bits(outputChangedView.film_exposure.input_xyz_adapt, 9), "output space preserves film identity/matrix");
                auto encoding = controls;
                encoding.outputCctfEncoding = 0;
                auto encodedOff = Cat02Fixtures::prepare(encoding);
                FjPreparedStateView encodedOffView{};
                require(encodedOff.view(encodedOffView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                require(bits(productView.film_exposure.input_xyz_adapt, 9) == bits(encodedOffView.film_exposure.input_xyz_adapt, 9) && productView.film_exposure.hash == encodedOffView.film_exposure.hash && productView.film_exposure.tc_lut_hash == encodedOffView.film_exposure.tc_lut_hash, "output CCTF-only changes preserve film/TC family");
                require(productView.output_boundary.entry_identity != 0 && productView.output_boundary.entry_identity == encodedOffView.output_boundary.entry_identity && productView.output_boundary.cmax.data == encodedOffView.output_boundary.cmax.data, "output CCTF-only reuses actual boundary table owner");
                require(Cat02Fixtures::tables_result(productView.scanner_tables) == Cat02Fixtures::tables_result(encodedOffView.scanner_tables) && Cat02Fixtures::scanner_descriptor(productView.scanner_lut) == Cat02Fixtures::scanner_descriptor(encodedOffView.scanner_lut), "output CCTF-only preserves complete scanner tables/descriptor");
                require(bits(productView.scanner_color.cat02, 9) == bits(encodedOffView.scanner_color.cat02, 9) && productView.scanner_color.hash != encodedOffView.scanner_color.hash && productView.scanner_recipe_hash != encodedOffView.scanner_recipe_hash && productView.recipe_hash != encodedOffView.recipe_hash, "output CCTF-only changes identities at color/output owners");
                InstanceState state;
                pending(state, controls);
                const auto old = admit_pending_render_state(state);
                require(admitted(old, route % 2 == 1), "unchanged CAT02 admission baseline");
                const auto hash = state.lastHash.load();
                const auto counter = state.buildCounterNext.load();
                const auto held = retain_values(old);
                FaultScope cleanup;
                expect_status(arm_color_fault(FJ_TEST_CAT02_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
                const auto repeated = admit_pending_render_state(state);
                require(repeated.directState == old.directState && repeated.printState == old.printState, "unchanged CAT02 admission reuses exact publication");
                unchanged(state, old, hash, counter);
                expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
                require_retained_values(old, held);
                require_retained_values(product, productHeld);
                auto rebuilt = Cat02Fixtures::prepare(controls);
                FjPreparedStateView rebuiltView{};
                require(rebuilt.view(rebuiltView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                const auto& oldPayload = old.printState ? old.printState->payload : old.directState->payload;
                require(rebuiltView.output_boundary.entry_identity != oldPayload.outputBoundaryTable->entryIdentity && rebuiltView.output_boundary.cmax.data != oldPayload.outputBoundaryTable->cmax.data() && same_retained_bits(std::span<const float>(rebuiltView.output_boundary.cmax.data, rebuiltView.output_boundary.cmax.count), held.boundary), "cache release makes a new equal table without mutating admitted hold");
                pending(state, output);
                const auto replacement = admit_pending_render_state(state);
                require(admitted(replacement, route % 2 == 1) && (replacement.directState != old.directState || replacement.printState != old.printState) && state.lastHash.load() == hash_params(output) && state.buildCounterNext.load() == counter + 1, "output-space state replacement publishes complete owner/hash/counter");
                require_retained_values(old, held);
            }
        }
    }
    void output_color_admission_tests() {
        for (int route = 0; route < 4; ++route) {
            auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019);
            InstanceState state;
            pending(state, controls);
            const auto original = admit_pending_render_state(state);
            auto changed = controls;
            changed.outputCctfEncoding = 0;
            std::uint32_t visits = 0;
            for (std::uint32_t index = 1; index <= 16; ++index) {
                FaultScope cleanup;
                pending(state, changed);
                const auto counter = state.buildCounterNext.load();
                expect_status(arm_color_fault(FJ_TEST_OUTPUT_COLOR, index, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                const auto result = admit_pending_render_state(state);
                const bool failedPreparation = result.status == PendingRenderAdmissionStatus::RebuildFailed;
                if (failedPreparation) {
                    require(!result.directState && !result.printState && result.diagnostic.find("InternalFailure") != std::string::npos, "output failure retains category and empty admission owners");
                } else {
                    require(admitted(result, route % 2 == 1), "completed output-color admission");
                }
                if (!failedPreparation) {
                    visits = index - 1;
                    FjOutputColorPreparation result{};
                    expect_status(fj_legacy_output_color(0, &result), FJ_STATUS_INTERNAL_FAILURE);
                    break;
                }
                require(state.buildCounterNext.load() == counter, "failed output completion never publishes");
                require(JuicerAtomic::load_shared_ptr(&state.activeDirectState) == original.directState && JuicerAtomic::load_shared_ptr(&state.activePrintState) == original.printState, "failed output completion retains older immutable publication");
            }
            require(visits > 0 && visits < 16, "bounded actual output preparation visit count");
            std::printf("Output color route=%d rebuild visits=%u\n", route, visits);
            FaultScope cleanup;
            const auto admitted = admit_pending_render_state(state);
            expect_status(arm_color_fault(FJ_TEST_OUTPUT_COLOR, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            const auto repeated = admit_pending_render_state(state);
            require(repeated.directState == admitted.directState && repeated.printState == admitted.printState, "warm output publication reuses owner without preparation");
            FjOutputColorPreparation result{};
            expect_status(fj_legacy_output_color(0, &result), FJ_STATUS_INTERNAL_FAILURE);
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
    void cat02_preparation_tests(const Json& fixture, const std::filesystem::path& fixturePath) {
        std::ifstream catalogueInput{fixturePath.parent_path().parent_path() / "output-color/native.json"};
        Json catalogue;
        catalogueInput >> catalogue;
        cat02_admission_tests();
        const auto cases = Cat02Fixtures::scenarios();
        require(cases.size() == fixture.at("recipes").size(), "frozen CAT02 recipe case membership");
        const auto witness = load_preset_witness(fixturePath, "cat02-manifest.json");
        bool wrongPresetRejected = false;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [id, controls] = cases[i];
            const auto& row = fixture.at("recipes").at(i);
            require(id == row.at("id").get<std::string>(), "frozen CAT02 recipe ordered case identity");
            const auto& settings = row.at("controls");
            require(settings.at("filmProfileKey") == controls.filmProfileKey && settings.at("printProfileKey") == controls.printProfileKey && settings.at("scanRoute") == static_cast<int>(controls.scanRoute) && settings.at("spectralUpsamplingMode") == controls.spectralUpsamplingMode && settings.at("inputColorSpace") == controls.inputColorSpace && settings.at("inputCctfDecoding") == controls.inputCctfDecoding && settings.at("outputColorSpace") == controls.outputColorSpace && settings.at("outputCctfEncoding") == controls.outputCctfEncoding && settings.at("cameraAutoExposureEnabled") == controls.cameraAutoExposureEnabled && settings.at("inputCompressionEnabled") == controls.inputCompressionEnabled && settings.at("cameraExposureCompensationEv").at("f64_bits") == std::bit_cast<std::uint64_t>(controls.cameraExposureCompensationEv), "frozen CAT02 selected controls");
            auto product = Cat02Fixtures::prepare(controls);
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
        }
        require(wrongPresetRejected, "different native-preset value/identity baselines reject incorrect expectation");
        std::array<Scanner::ScannerIlluminant, 4> illuminants;
        for (int route = 0; route < 4; ++route) {
            auto product = Cat02Fixtures::prepare(Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019));
            illuminants[route] = Cat02Fixtures::scanner_illuminant(product);
            const auto& row = fixture.at("scanner_illuminants").at(route);
            const auto& illuminant = illuminants[route];
            FjPreparedStateView view{};
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            require(product.view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(row.at("route") == route && row.at("viewing_illuminant") == Cat02Fixtures::viewing_illuminant(view) && row.at("white_xyz") == bits(illuminant.whiteXYZ) && row.at("white_xy") == bits(illuminant.whiteXY) && row.at("normalization") == bits(std::array{illuminant.normalization}) && row.at("hash") == illuminant.hash && row.at("samples") == bits(illuminant.curve.linear), "actual selected-route scanner illuminant exact fields/identity");
        }
        for (const auto& row : fixture.at("scanner_cases")) {
            const OutputEncoding::Params encoding{static_cast<OutputEncoding::ColorSpace>(row.at("space").get<int>()), row.at("encoding").get<int>() != 0, true};
            const auto color = Scanner::build_color_runtime(static_cast<Scanner::ScannerMedium>(row.at("medium").get<int>()), illuminants[row.at("route").get<int>()], encoding, row.at("gamut_recipe_hash").get<std::uint64_t>());
            require(Cat02Fixtures::color_result(color) == row.at("expected"), row.at("id").get<std::string>() + " real scanner color exact arrays/encoding/hash");
            require_scanner_encoding(color, catalogue);
        }
        for (const auto& row : fixture.at("scanner_invalid_cases")) {
            auto invalid = illuminants[0];
            invalid.hash = 0;
            FaultScope cleanup;
            expect_status(arm_color_fault(FJ_TEST_CAT02_MATRIX, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            const auto color = Scanner::build_color_runtime(static_cast<Scanner::ScannerMedium>(row.at("medium").get<int>()), invalid, {}, 0);
            require(Cat02Fixtures::color_result(color) == row.at("expected"), "invalid scanner illuminant keeps early zero-hash color result");
            expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_INTERNAL_FAILURE);
        }
        output_color_admission_tests();
        cat02_helper_tests(fixture);
        cat02_control_tests();
        std::printf("CAT02 preparation: %zu independent recipes, %zu scanner cases, distinct controls and full retained arrays passed\n", cases.size(), fixture.at("scanner_cases").size());
    }

    namespace InputFixtures {
        Json controls_json(const Spektrafilm::DiffusionFilterAuthoredControls& p) {
            Json out;
            out["active"] = p.active;
            out["family"] = static_cast<int>(p.family);
            out["strength"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.strength)}};
            out["spatialScale"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.spatialScale)}};
            out["haloWarmth"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.haloWarmth)}};
            out["coreIntensity"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.coreIntensity)}};
            out["coreSize"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.coreSize)}};
            out["haloIntensity"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.haloIntensity)}};
            out["haloSize"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.haloSize)}};
            out["bloomIntensity"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.bloomIntensity)}};
            out["bloomSize"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.bloomSize)}};
            return out;
        }
        Json controls_json(const Spektrafilm::DirCouplersControls& p) {
            Json out;
            out["active"] = p.active;
            out["amount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.amount)}};
            out["inhibitionSameLayer"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.inhibitionSameLayer)}};
            out["inhibitionInterlayer"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.inhibitionInterlayer)}};
            out["diffusionSizeUm"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.diffusionSizeUm)}};
            out["diffusionTailUm"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.diffusionTailUm)}};
            out["diffusionTailWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.diffusionTailWeight)}};
            out["gammaUseStock"] = p.gammaUseStock;
            out["gammaSameLayerRgb"] = Json{{"f32_bits", bits(p.gammaSameLayerRgb)}};
            out["gammaInterlayerRToGb"] = Json{{"f32_bits", bits(p.gammaInterlayerRToGb)}};
            out["gammaInterlayerGToRb"] = Json{{"f32_bits", bits(p.gammaInterlayerGToRb)}};
            out["gammaInterlayerBToRg"] = Json{{"f32_bits", bits(p.gammaInterlayerBToRg)}};
            out["langmuirDonorKRgb"] = Json{{"f32_bits", bits(p.langmuirDonorKRgb)}};
            out["langmuirReceiverKRgb"] = Json{{"f32_bits", bits(p.langmuirReceiverKRgb)}};
            return out;
        }
        Json controls_json(const Spektrafilm::VisualGrainControls& p) {
            Json out;
            out["active"] = p.active;
            out["sublayersActive"] = p.sublayersActive;
            out["particleAreaUm2"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.particleAreaUm2)}};
            out["particleScaleCmy"] = Json{{"f32_bits", bits(p.particleScaleCmy)}};
            out["particleScaleLayers"] = Json{{"f32_bits", bits(p.particleScaleLayers)}};
            out["visualParticleDensityMinCmy"] = Json{{"f32_bits", bits(p.visualParticleDensityMinCmy)}};
            out["uniformityCmy"] = Json{{"f32_bits", bits(p.uniformityCmy)}};
            out["correlationSigmaPx"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.correlationSigmaPx)}};
            out["dyeCloudBlurUm"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.dyeCloudBlurUm)}};
            out["microStructure"] = Json{{"f32_bits", bits(p.microStructure)}};
            out["nSubLayers"] = p.nSubLayers;
            out["amplitude"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.amplitude)}};
            out["chromaMix"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.chromaMix)}};
            out["chromaSharedWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.chromaSharedWeight)}};
            out["chromaIndependentWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.chromaIndependentWeight)}};
            out["coarseWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.coarseWeight)}};
            out["midWeight"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.midWeight)}};
            out["sizeMixScale"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.sizeMixScale)}};
            out["clumpTemporalMix"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.clumpTemporalMix)}};
            out["clumpMorphPeriodSec"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.clumpMorphPeriodSec)}};
            out["debugView"] = p.debugView;
            return out;
        }
        Json controls_json(const ScatterHalationControls& p) {
            Json out;
            out["active"] = p.active;
            out["scatterAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.scatterAmount)}};
            out["scatterSpatialScale"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.scatterSpatialScale)}};
            out["halationAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.halationAmount)}};
            out["halationSpatialScale"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.halationSpatialScale)}};
            return out;
        }
        Json controls_json(const ParamSnapshot& p) {
            Json out;
            out["printExposure"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.printExposure)}};
            out["printPreflashExposure"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.printPreflashExposure)}};
            out["printGammaFactor"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.printGammaFactor)}};
            out["preflashMFilterCc"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.preflashMFilterCc)}};
            out["preflashYFilterCc"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.preflashYFilterCc)}};
            out["printShadowCompensationFactor"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.printShadowCompensationFactor)}};
            out["printShadowCompensationDensity"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.printShadowCompensationDensity)}};
            out["printShadowCompensationTransition"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.printShadowCompensationTransition)}};
            out["glarePercent"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.glarePercent)}};
            out["glareRoughness"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.glareRoughness)}};
            out["glareBlurSigmaPx"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.glareBlurSigmaPx)}};
            out["cameraExposureCompensationEv"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.cameraExposureCompensationEv)}};
            out["cameraFilmFormatLongEdgeMm"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.cameraFilmFormatLongEdgeMm)}};
            out["filmGammaFactor"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.filmGammaFactor)}};
            out["scannerLensBlurSigmaPx"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.scannerLensBlurSigmaPx)}};
            out["scannerBlackLevel"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.scannerBlackLevel)}};
            out["scannerWhiteLevel"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.scannerWhiteLevel)}};
            out["gateWeaveAmount"] = Json{{"f64_bits", std::bit_cast<std::uint64_t>(p.gateWeaveAmount)}};
            out["scannerUnsharpMask"] = Json{{"f64_bits", double_bits(p.scannerUnsharpMask)}};
            out["printUiYmcCc"] = Json{{"f64_bits", double_bits(p.printUiYmcCc)}};
            out["cameraFilterUV"] = Json{{"f64_bits", double_bits(p.cameraFilterUV)}};
            out["cameraFilterIR"] = Json{{"f64_bits", double_bits(p.cameraFilterIR)}};
            out["cameraDiffusion"] = controls_json(p.cameraDiffusion);
            out["enlargerDiffusion"] = controls_json(p.enlargerDiffusion);
            out["dirCouplers"] = controls_json(p.dirCouplers);
            out["filmProfileKey"] = p.filmProfileKey;
            out["printProfileKey"] = p.printProfileKey;
            out["spectralUpsamplingMode"] = p.spectralUpsamplingMode;
            out["refIll"] = p.refIll;
            out["enlIll"] = p.enlIll;
            out["normalizePrintExposure"] = p.normalizePrintExposure;
            out["printExposureCompensation"] = p.printExposureCompensation;
            out["inputColorSpace"] = p.inputColorSpace;
            out["inputCctfDecoding"] = p.inputCctfDecoding;
            out["inputCompressionEnabled"] = p.inputCompressionEnabled;
            out["hanatos2025AdaptationWindow"] = p.hanatos2025AdaptationWindow;
            out["hanatos2025AdaptationSurface"] = p.hanatos2025AdaptationSurface;
            out["cameraAutoExposureEnabled"] = p.cameraAutoExposureEnabled;
            out["cameraMeteringMethod"] = p.cameraMeteringMethod;
            out["scatterHalationControls"] = controls_json(p.scatterHalationControls);
            out["scannerBlackCorrection"] = p.scannerBlackCorrection;
            out["scannerWhiteCorrection"] = p.scannerWhiteCorrection;
            out["scannerLutResolution"] = p.scannerLutResolution;
            out["outputColorSpace"] = p.outputColorSpace;
            out["outputCctfEncoding"] = p.outputCctfEncoding;
            out["outputGamutCompressionEnabled"] = p.outputGamutCompressionEnabled;
            out["grainControls"] = controls_json(p.grainControls);
            out["filmDustAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.filmDustAmount)}};
            out["filmScratchAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.filmScratchAmount)}};
            out["gateDustAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.gateDustAmount)}};
            out["gateScratchAmount"] = Json{{"f32_bits", std::bit_cast<std::uint32_t>(p.gateScratchAmount)}};
            out["scanRoute"] = static_cast<int>(p.scanRoute);
            out["glareActive"] = p.glareActive;
            out["cameraFilterOverride"] = p.cameraFilterOverride;
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
        std::vector<std::pair<std::string, ParamSnapshot>> scenarios() {
            std::vector<std::pair<std::string, ParamSnapshot>> out;
            for (int polarity = 0; polarity < 2; ++polarity) {
                for (int route = 0; route < 2; ++route) {
                    for (int space = 0; space < 4; ++space) {
                        for (int decode = 0; decode < 2; ++decode) {
                            for (int method = 0; method < 3; ++method) {
                                auto p = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                                p.inputColorSpace = space;
                                p.inputCctfDecoding = decode;
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
                        p.inputColorSpace = 3;
                        p.inputCctfDecoding = 1;
                        const auto id = std::to_string(polarity) + "-" + std::to_string(route) + "-" + std::to_string(method);
                        p.cameraAutoExposureEnabled = 1;
                        out.emplace_back("auto-" + id, p);
                        p.cameraAutoExposureEnabled = 0;
                        if (method != 1) {
                            p.inputCompressionEnabled = 1;
                            out.emplace_back("compression-" + id, p);
                            p.inputCompressionEnabled = 0;
                        }
                        p.outputCctfEncoding = 0;
                        out.emplace_back("output-off-" + id, p);
                        p.outputCctfEncoding = 1;
                        p.outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
                        out.emplace_back("output-dci-" + id, p);
                    }
                }
            }
            return out;
        }
        Json complete_result(PreparedState& product) {
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
        template <typename State>
        Json publication_complete_result(const State& product) {
            auto result = Cat02Fixtures::complete_result(product);
            const auto& raw = product.recipe.filmRaw;
            result["config_xyz_to_linear_srgb"] = bits(product.payload.filmRawConfig.xyzToLinearSrgb.m);
            result["recipe_mallett_scale"] = bits(std::array{raw.mallettGreenMidgrayScale});
            result["meter_scale"] = bits(std::array{product.recipe.scannerOutput.syntheticFilmReference.baselineMeterScale});
            result["raw_flags"] = {raw.inputColorSpace, raw.inputCctfDecoding, static_cast<int>(raw.rgbToRawMethod), raw.autoExposureEnabled, raw.inputCompressionActive};
            result["sensitivity_hash"] = raw.finalSensitivityHash;
            for (const auto& sample : raw.finalSensitivity) {
                result["sensitivity"].push_back(bits(sample));
            }
            result["tc_source_hash"] = raw.tcSourceAssetHash;
            result["tc_lut_size"] = product.payload.filmTcLut ? product.payload.filmTcLut->samples().size() : 0;
            if (product.payload.filmTcLut) {
                const auto& data = product.payload.filmTcLut->samples();
                for (std::size_t index : std::array<std::size_t, 5>{0, 768, 73728, 100000, 147452}) {
                    result["tc_samples"].push_back({{"index", index}, {"rgba", bits(std::array{data[index], data[index + 1], data[index + 2], data[index + 3]})}});
                }
            }
            return result;
        }
    } // namespace InputFixtures

    void input_leaf_tests(const Json& fixture) {
        const auto& inverse = fixture.at("recipes").at(0).at("expected").at("xyz_to_linear_srgb");
        for (const auto& row : fixture.at("recipes")) {
            require(row.at("expected").at("xyz_to_linear_srgb") == inverse && row.at("expected").at("config_xyz_to_linear_srgb") == inverse, "amended producer reference uses agreeing completed native A");
        }
        // The standalone inverse and unready default remain native characterization.
        // Explicit conversion operands, including B, are consumed exactly as frozen.
        for (const auto& row : fixture.at("matrices")) {
            const auto space = row.at("space").get<int>();
            const auto actual = JuicerColor::input_matrices(static_cast<Spectral::InputColorSpace>(space));
            FjInputColorMatrices facade{};
            expect_status(fj_test_input_matrices(static_cast<std::uint32_t>(space), &facade), FJ_STATUS_SUCCESS);
            require(bits(actual.rgbToXyz) == row.at("rgb_to_xyz") && bits(actual.nominalWhiteXYZ) == row.at("nominal_white") && bits(actual.d65WhiteXYZ) == row.at("d65_white") && bits(actual.xyzToLinearSrgb) == inverse, "production input matrices/nominal whites/inverse A exact");
            require(bits(facade.rgb_to_xyz) == row.at("rgb_to_xyz") && bits(facade.nominal_white_xyz) == row.at("nominal_white") && bits(facade.d65_white_xyz) == row.at("d65_white") && bits(facade.xyz_to_linear_srgb) == inverse, "direct-core input matrices exact");
        }
        for (const auto& row : fixture.at("decode")) {
            const auto rgb = triplet(row.at("rgb"));
            std::array<float, 3> actual{};
            expect_status(fj_test_decode_input(row.at("space").get<std::uint32_t>(), row.at("decode").get<std::uint32_t>(), rgb.data(), actual.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(actual), row.at("expected").at("linear")), row.at("id").get<std::string>() + " decoder branch/zero/nonfinite/overflow");
        }
        for (const auto& row : fixture.at("conversion")) {
            const auto config = InputFixtures::config(row);
            const auto input = InputFixtures::foreign(config);
            const auto rgb = triplet(row.at("rgb"));
            const auto& expected = row.at("expected");
            for (bool clamp : {false, true}) {
                std::array<float, 3> facadeRgb{}, facadeXyz{};
                expect_status(fj_test_input_to_dwg(&input, rgb.data(), clamp ? 1u : 0u, facadeRgb.data(), facadeXyz.data()), FJ_STATUS_SUCCESS);
                const std::string prefix = clamp ? "dwg_clamped" : "dwg_signed";
                require(same_values(bits(facadeRgb), expected.at(prefix + "_rgb")) && same_values(bits(facadeXyz), expected.at(prefix + "_xyz")), row.at("id").get<std::string>() + " facade DWG conversion/XYZ observation");
            }
            std::array<float, 3> facadeRgb{}, facadeXyz{};
            expect_status(fj_test_input_to_linear_srgb(&input, rgb.data(), config.xyz_to_linear_srgb, facadeRgb.data(), facadeXyz.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(facadeRgb), expected.at("srgb_rgb")) && same_values(bits(facadeXyz), expected.at("srgb_xyz")), row.at("id").get<std::string>() + " facade explicit inverse linear sRGB/XYZ");
        }
        for (const auto& row : fixture.at("linear")) {
            const auto rgb = triplet(row.at("rgb"));
            std::array<float, 3> srgb{}, dwg{};
            expect_status(fj_test_linear_srgb_to_xyz(rgb.data(), srgb.data()), FJ_STATUS_SUCCESS);
            expect_status(fj_test_dwg_to_xyz(rgb.data(), dwg.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(srgb), row.at("expected").at("srgb_xyz")) && same_values(bits(JuicerColor::linear_srgb_to_xyz(rgb)), row.at("expected").at("srgb_xyz")), row.at("id").get<std::string>() + " unsanitized linear sRGB leaf");
            require(same_values(bits(dwg), row.at("expected").at("dwg_xyz")), row.at("id").get<std::string>() + " unsanitized DWG leaf");
        }
        for (const auto& row : fixture.at("projection")) {
            const auto rgb = triplet(row.at("rgb"));
            const auto matrix = InputFixtures::matrix(row.at("rgb_to_xyz"));
            const auto adaptation = InputFixtures::matrix(row.at("xyz_adapt"));
            const auto actual = JuicerColor::project_linear_rgb_to_xyz(rgb, matrix, adaptation);
            std::array<float, 3> facade{};
            expect_status(fj_test_project_linear_rgb_to_xyz(rgb.data(), matrix.m, adaptation.m, facade.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(actual), row.at("expected").at("projected_xyz")) && same_values(bits(facade), row.at("expected").at("projected_xyz")), row.at("id").get<std::string>() + " ordered already-linear projection");
        }
        const auto projectionCase = std::find_if(fixture.at("projection").begin(), fixture.at("projection").end(), [](const auto& row) {
            return row.at("id") == "projection-3-rgb-5";
        });
        require(projectionCase != fixture.at("projection").end(), "projection comparator case");
        const auto rgb = triplet(projectionCase->at("rgb"));
        const auto forward = InputFixtures::matrix(projectionCase->at("rgb_to_xyz"));
        const auto adapt = InputFixtures::matrix(projectionCase->at("xyz_adapt"));
        const auto expectedProjection = projectionCase->at("expected").at("projected_xyz");
        require(!same_values(bits(JuicerColor::project_linear_rgb_to_xyz(rgb, adapt, forward)), expectedProjection), "reversed matrix order rejected");
        auto transposed = forward;
        std::swap(transposed.m[1], transposed.m[3]);
        std::swap(transposed.m[2], transposed.m[6]);
        std::swap(transposed.m[5], transposed.m[7]);
        require(!same_values(bits(JuicerColor::project_linear_rgb_to_xyz(rgb, transposed, adapt)), expectedProjection), "transposed matrix rejected");
        std::array<float, 3> wronglyDecoded{};
        expect_status(fj_test_decode_input(3, 1, rgb.data(), wronglyDecoded.data()), FJ_STATUS_SUCCESS);
        require(!same_values(bits(JuicerColor::project_linear_rgb_to_xyz(wronglyDecoded, forward, adapt)), expectedProjection), "decoding already-linear TC projection rejected");
        auto coefficient = fixture.at("matrices").at(3).at("rgb_to_xyz");
        coefficient[0] = coefficient[0].get<std::uint32_t>() ^ 1u;
        require(coefficient != bits(JuicerColor::input_matrices(Spectral::InputColorSpace::SRGB_Rec709).rgbToXyz), "one coefficient bit rejected");
        require(bits(JuicerColor::input_matrices(Spectral::InputColorSpace::ACES2065_1).rgbToXyz) != fixture.at("matrices").at(3).at("rgb_to_xyz"), "input tag swap rejected");
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
        require(fj_test_input_color_abi_c() == 0, "input-color C11 layouts/signatures/nulls/flags/partial clearing/panic/canaries/readonly aliases");
        color_fault_tests();
        for (std::uint32_t operation = 5; operation <= 10; ++operation) {
            FaultScope cleanup;
            expect_status(arm_color_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            Spectral::FilmRawConfig unready;
            require(!unready.valid, "default carrier remains unready");
            const auto identity = Spectral::make_identity_mat3();
            require(bits(unready.inputRGBToXYZ.m) == bits(identity.m) && bits(unready.inputXYZAdapt.m) == bits(identity.m) && bits(unready.xyzToLinearSrgb.m) == bits(identity.m), "unready carrier has only identity matrices and performs no derivation");
            std::array<float, 3> decoded{};
            const std::array<float, 3> rgb{.184f, -0.0f, -1.0f};
            expect_status(fj_test_decode_input(3, 1, rgb.data(), decoded.data()), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_INTERNAL_FAILURE);
        }
        for (std::uint32_t operation : {FJ_TEST_INPUT_MATRICES, FJ_TEST_INPUT_TO_DWG, FJ_TEST_INPUT_TO_LINEAR_SRGB}) {
            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                FaultScope cleanup;
                expect_status(arm_color_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                FjInputColorMatrices matrices{};
                const std::array<float, 3> rgb{.184f, -.25f, .5f};
                const std::array<float, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
                FjInputColorConversion input{FJ_INPUT_DWG, 0, 0, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}};
                std::array<float, 3> outRgb{}, outXyz{};
                if (operation == FJ_TEST_INPUT_MATRICES) {
                    expect_status(fj_legacy_input_matrices(4, &matrices), FJ_STATUS_UNSUPPORTED_INPUT);
                } else {
                    for (int invalid = 0; invalid < 3; ++invalid) {
                        input.input_space = invalid == 0 ? 4u : 0u;
                        input.decode_cctf = invalid == 1 ? 2u : 0u;
                        input.adapt_xyz = invalid == 2 ? 2u : 0u;
                        const auto status = operation == FJ_TEST_INPUT_TO_DWG
                                                ? fj_test_input_to_dwg(&input, rgb.data(), 0, outRgb.data(), outXyz.data())
                                                : fj_test_input_to_linear_srgb(&input, rgb.data(), identity.data(), outRgb.data(), outXyz.data());
                        expect_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
                    }
                    if (operation == FJ_TEST_INPUT_TO_DWG) {
                        input.input_space = 0;
                        input.decode_cctf = 0;
                        input.adapt_xyz = 0;
                        expect_status(fj_test_input_to_dwg(&input, rgb.data(), 2, outRgb.data(), outXyz.data()), FJ_STATUS_UNSUPPORTED_INPUT);
                    }
                }
                expect_status(raw_call(operation), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
            }
        }
        require(fixture.at("input_d65") != fixture.at("process_dwg_white"), "input D65 and process DWG white remain distinct");
        require(Spectral::inputColorSpaceFromIndex(-1) == Spectral::InputColorSpace::DaVinciWideGamut && Spectral::inputColorSpaceFromIndex(4) == Spectral::InputColorSpace::DaVinciWideGamut, "native integer transport fallback retained");
        std::puts("Input color: exact matrices/A, decode, explicit A/B conversions, linear leaves/projection and C11/ten-operation fault/default/facade controls passed");
    }

    struct InputValues {
        std::vector<std::uint32_t> bits;
        std::array<std::uint64_t, 13> hashes{};
        bool operator==(const InputValues&) const = default;
    };
    // Temporary I4 publication observer; switch to a holder-backed view in I5.
    template <typename State>
    InputValues input_values(const State& state) {
        InputValues out;
        const auto& raw = state.recipe.filmRaw;
        const auto& config = state.payload.filmRawConfig;
        const auto append = [&](const auto& samples) {
            for (float value : samples)
                out.bits.push_back(std::bit_cast<std::uint32_t>(value));
        };
        append(raw.inputRgbToXyz);
        append(raw.inputXyzAdapt);
        append(raw.xyzToLinearSrgb);
        append(raw.inputNominalWhiteXYZ);
        append(raw.projectionWhiteXYZ);
        append(config.inputRGBToXYZ.m);
        append(config.inputXYZAdapt.m);
        append(config.xyzToLinearSrgb.m);
        append(config.rawMidgray);
        append(config.midgrayDWG);
        append(config.inputWhiteXYZ);
        append(config.workingWhiteXYZ);
        append(config.refIllumWhiteXYZ);
        append(std::array{config.midgrayScale, config.rawMidgrayGreen, raw.mallettGreenMidgrayScale, state.recipe.scannerOutput.syntheticFilmReference.baselineMeterScale});
        for (auto value : {static_cast<std::uint32_t>(config.inputColorSpace), std::uint32_t(config.applyCctfDecoding), static_cast<std::uint32_t>(config.spectralUpsamplingMode), std::uint32_t(config.applyInputChromaticAdapt), std::uint32_t(config.hasRefIllumWhite), std::uint32_t(config.valid), std::uint32_t(raw.inputColorSpace), std::uint32_t(raw.inputCctfDecoding), static_cast<std::uint32_t>(raw.rgbToRawMethod), std::uint32_t(raw.autoExposureEnabled), std::uint32_t(raw.inputCompressionActive)})
            out.bits.push_back(value);
        out.hashes = {raw.hash, raw.tcLutHash, state.recipe.scannerOutput.syntheticFilmReference.hash, state.recipe.print.balance.hash, state.recipe.print.hash, state.recipe.scannerOutput.hash, state.recipe.scannerOutput.outputGamut.hash, state.recipe.hash, state.payload.uploadCoreHash, state.payload.scannerHash, raw.finalSensitivityHash, raw.tcSourceAssetHash, state.payload.filmTcLut ? state.payload.filmTcLut->samples().size() : 0};
        return out;
    }
    InputValues input_values(const PendingRenderAdmissionResult& result) {
        require(bool(result.directState) || bool(result.printState), "retained input-color owner");
        return result.printState ? input_values(*result.printState) : input_values(*result.directState);
    }
    void input_admission_tests() {
        std::size_t failures = 0, supersessions = 0, counts = 0;
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                for (int space : {1, 3}) {
                    auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                    controls.inputColorSpace = space;
                    controls.inputCctfDecoding = 1;
                    auto changed = controls;
                    changed.cameraExposureCompensationEv = 1.0;
                    InstanceState baseline;
                    pending(baseline, controls);
                    const auto publication = admit_pending_render_state(baseline);
                    require(admitted(publication, route % 2 == 1), "input-color immutable baseline admitted");
                    const auto oldValues = input_values(publication);
                    const std::array<std::uint32_t, 6> callCounts{method == 1 ? 1u : 3u, 0u, 0u, method == 1 ? 0u : 2u, 0u, method == 1 ? 0u : 1u};
                    for (std::uint32_t operation = 5; operation <= 10; ++operation) {
                        const auto calls = callCounts[operation - 5];
                        {
                            FaultScope cleanup;
                            expect_status(arm_color_fault(operation, calls + 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                            // SF_TEMP_BRIDGE_NATIVE_INPUT_OPERATION_COUNTS: retain
                            // the original native visit counts until I7.
                            FocusedRenderStateBuildProduct SF_TEMP_BRIDGE_nativeProduct;
                            std::string SF_TEMP_BRIDGE_nativeDiagnostic;
                            require(route % 2 == 1
                                        ? build_print_render_state_product(controls, SF_TEMP_BRIDGE_nativeProduct, SF_TEMP_BRIDGE_nativeDiagnostic)
                                        : build_direct_render_state_product(controls, SF_TEMP_BRIDGE_nativeProduct, SF_TEMP_BRIDGE_nativeDiagnostic),
                                    SF_TEMP_BRIDGE_nativeDiagnostic);
                            expect_status(raw_call(operation), FJ_STATUS_INTERNAL_FAILURE);
                            ++counts;
                        }
                        {
                            FaultScope cleanup;
                            // Complete preparation has no native projection-wrapper
                            // visit. Its other checked input operations are shared.
                            const auto heldCalls = operation == FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ ? 0u : calls;
                            expect_status(arm_color_fault(operation, heldCalls + 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                            (void)Cat02Fixtures::prepare(controls);
                            expect_status(raw_call(operation), FJ_STATUS_INTERNAL_FAILURE);
                        }
                        for (std::uint32_t index = 1; index <= calls; ++index) {
                            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                                const FaultCase selected{operation, index, fault};
                                const auto stage = operation == FJ_TEST_LINEAR_SRGB_TO_XYZ || (operation == FJ_TEST_INPUT_MATRICES && index > 1) ? FailureStage::FilmReference : FailureStage::Leaf;
                                FaultScope cleanup;
                                InstanceState cold;
                                pending(cold, controls);
                                expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                const auto coldFailure = admit_pending_render_state(cold);
                                failed(coldFailure, selected, controls.filmProfileKey, stage);
                                require(!JuicerAtomic::load_shared_ptr(&cold.activeDirectState) && !JuicerAtomic::load_shared_ptr(&cold.activePrintState) && cold.lastHash.load() == 0 && cold.buildCounterNext.load() == 0, "cold failed construction publishes no owner/hash/counter");
                                for (std::uint32_t call = 0; call < index; ++call) {
                                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                }
                                InstanceState state;
                                // Fault counters, pending state and recovery stay independent of the baseline.
                                JuicerAtomic::store_shared_ptr(&state.activeDirectState, publication.directState);
                                JuicerAtomic::store_shared_ptr(&state.activePrintState, publication.printState);
                                state.lastHash.store(baseline.lastHash.load());
                                state.buildCounterNext.store(baseline.buildCounterNext.load());
                                pending(state, controls);
                                const auto old = admit_pending_render_state(state);
                                require(admitted(old, route % 2 == 1) && old.directState == publication.directState && old.printState == publication.printState, "input-color current reuses immutable baseline");
                                const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                                pending(state, changed);
                                expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                const auto result = admit_pending_render_state(state);
                                failed(result, selected, controls.filmProfileKey, stage);
                                unchanged(state, old, hash, counter);
                                require(input_values(old) == oldValues, "retained input-color fields/identities remain exact");
                                // Same matching raw export, original count, admission thread, before cleanup.
                                for (std::uint32_t call = 0; call < index; ++call) {
                                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                }
                                pending(state, controls);
                                const auto reused = admit_pending_render_state(state);
                                require(reused.directState == old.directState && reused.printState == old.printState, "failed input-color rebuild retains reusable publication");
                                unchanged(state, old, hash, counter);
                                pending(state, changed);
                                const auto recovered = admit_pending_render_state(state);
                                require(admitted(recovered, route % 2 == 1), "input-color recovery admitted");
                                require(recovered.directState != old.directState || recovered.printState != old.printState, "recovery replaces route publication");
                                require(state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "recovery publishes complete hash/counter");
                                require(input_values(old) == oldValues, "retained input-color fields/identities remain exact");
                                const auto recoveredValues = input_values(recovered);
                                ++failures;
                                for (int mode = 0; mode < 3; ++mode) {
                                    Supersession selection{changed, mode, false};
                                    const auto liveHash = state.lastHash.load(), liveCounter = state.buildCounterNext.load();
                                    pending(state, controls);
                                    set_pending_capture_test_hook(supersede, &selection);
                                    expect_status(arm_color_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                    const auto newer = admit_pending_render_state(state);
                                    require(selection.fired, "captured input-color snapshot superseded");
                                    if (mode == 0) {
                                        require(admitted(newer, route % 2 == 1) && newer.directState == recovered.directState && newer.printState == recovered.printState, "superseded valid input reuses successful publication");
                                    } else {
                                        require(newer.status == (mode == 1 ? PendingRenderAdmissionStatus::InvalidSnapshotControls : PendingRenderAdmissionStatus::NeedsSnapshotAcquisition) && !newer.directState && !newer.printState, "superseded invalid/uninitialized acquisition");
                                    }
                                    unchanged(state, recovered, liveHash, liveCounter);
                                    require(input_values(old) == oldValues, "retained input-color fields/identities remain exact");
                                    require(input_values(recovered) == recoveredValues, "recovered input-color fields/identities remain exact");
                                    for (std::uint32_t call = 0; call < index; ++call) {
                                        expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                    }
                                    set_pending_capture_test_hook(nullptr, nullptr);
                                    expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
                                    ++supersessions;
                                }
                            }
                        }
                    }
                }
            }
        }
        require(counts == 144 && failures == 208 && supersessions == 624, "all input-color operation/route/algorithm/fault witnesses retained");
        std::printf("Input-color admission: %zu exact operation counts; %zu early/late failure/recovery and %zu N2 supersession witnesses on four routes/two decoded spaces passed\n", counts, failures, supersessions);
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

    void input_helper_tests(const Json& fixture) {
        auto product = Cat02Fixtures::prepare(Cat02Fixtures::snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Mallett2019));
        auto* assets = JuicerProcess::root().assets().borrow_rust_assets();
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
            require(Cat02Fixtures::helper_result(row, tables, inverse, true, hanatosSamples) == row.at("hanatos_expected"), row.at("id").get<std::string>() + " retained Hanatos helper exact");
            require(Cat02Fixtures::helper_result(row, tables, inverse, false, hanatosSamples) == row.at("tables_expected"), row.at("id").get<std::string>() + " retained table helper exact");
        }
        const std::array<float, 3> rgb{2.0f, -.25f, .5f};
        for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
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
                (void)JuicerColor::linear_srgb_to_xyz(rgb);
            });
            require_input_failure(FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ, fault, [&] {
                Spectral::Mat3 inputRgbToXyz{}, inputXyzAdapt{};
                std::copy_n(view.film_exposure.input_rgb_to_xyz, 9, inputRgbToXyz.m);
                std::copy_n(view.film_exposure.input_xyz_adapt, 9, inputXyzAdapt.m);
                (void)JuicerColor::project_linear_rgb_to_xyz(rgb, inputRgbToXyz, inputXyzAdapt);
            });
            require_input_failure(FJ_TEST_INPUT_MATRICES, fault, [&] {
                (void)JuicerColor::input_matrices(Spectral::InputColorSpace::ITU_R_BT2020);
            });
        }
        std::puts("Input-color helpers: independent spectra and both-category direct primitive/DWG helper failures passed");
    }

    void input_preparation_tests(const Json& fixture, const std::filesystem::path& fixturePath) {
        retained_comparison_tests();
        const auto cases = InputFixtures::scenarios();
        require(cases.size() == 140 && cases.size() == fixture.at("recipes").size(), "96 base plus 44 bounded input-color products");
        require(InputFixtures::controls_json(ParamSnapshot{}) == fixture.at("defaults"), "all frozen parent defaults exact");
        const auto witness = load_preset_witness(fixturePath, "input-color-manifest.json");
        bool wrongPreset = false;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [id, controls] = cases[i];
            const auto& row = fixture.at("recipes").at(i);
            require(id == row.at("id").get<std::string>() && InputFixtures::controls_json(controls) == row.at("controls"), "ordered frozen product inputs/settings exact");
            auto product = Cat02Fixtures::prepare(controls);
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
        }
        require(wrongPreset, "native preset differences are observable and reject wrong references");
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                controls.inputColorSpace = 1;
                controls.inputCctfDecoding = 1;
                InstanceState state;
                pending(state, controls);
                const auto old = admit_pending_render_state(state);
                require(admitted(old, route % 2 == 1), "retained input-color route baseline");
                const auto held = retain_values(old);
                const auto fullHeld = old.printState ? InputFixtures::publication_complete_result(*old.printState) : InputFixtures::publication_complete_result(*old.directState);
                const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                const auto repeated = admit_pending_render_state(state);
                require(repeated.directState == old.directState && repeated.printState == old.printState, "unchanged input uses exact owners");
                unchanged(state, old, hash, counter);
                auto decoding = controls;
                decoding.inputCctfDecoding = 0;
                auto undecoded = Cat02Fixtures::prepare(decoding);
                FjPreparedStateView undecodedView{};
                std::array<char, 2048> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                require(undecoded.view(undecodedView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                const auto& oldRaw = old.printState ? old.printState->recipe.filmRaw : old.directState->recipe.filmRaw;
                require(oldRaw.hash != undecodedView.film_exposure.hash, "input decoding remains a contributing film identity control");
                if (method != 1) {
                    require((old.printState ? bits(old.printState->payload.filmRawConfig.rawMidgray) : bits(old.directState->payload.filmRawConfig.rawMidgray)) == bits(undecodedView.raw_midgray, 3), "TC mid-gray remains already linear without CCTF decoding");
                }
                expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
                auto changed = controls;
                changed.inputColorSpace = 3;
                pending(state, changed);
                const auto replacement = admit_pending_render_state(state);
                require(admitted(replacement, route % 2 == 1) && (replacement.directState != old.directState || replacement.printState != old.printState), "input choice replaces complete publication");
                require(state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "replacement publishes complete input identity/counter");
                require_retained_values(old, held);
                require((old.printState ? InputFixtures::publication_complete_result(*old.printState) : InputFixtures::publication_complete_result(*old.directState)) == fullHeld, "full input-color config/scales/matrices retained after cache release/replacement");
                auto output = changed;
                output.outputCctfEncoding = 0;
                auto encodedOff = Cat02Fixtures::prepare(output);
                FjPreparedStateView encodedOffView{};
                require(encodedOff.view(encodedOffView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                const auto& liveRecipe = replacement.printState ? replacement.printState->recipe : replacement.directState->recipe;
                const auto& livePayload = replacement.printState ? replacement.printState->payload : replacement.directState->payload;
                require(liveRecipe.filmRaw.hash == encodedOffView.film_exposure.hash && liveRecipe.filmRaw.tcLutHash == encodedOffView.film_exposure.tc_lut_hash && livePayload.scannerHash == encodedOffView.scanner_hash && Cat02Fixtures::tables_result(livePayload.scannerTables) == Cat02Fixtures::tables_result(encodedOffView.scanner_tables), "output encoding preserves film/TC and scanner spectral families");
            }
        }
        input_helper_tests(fixture);
        input_admission_tests();
        std::puts("Input-color preparation: 140 exact products/identities, controls, retained owners/cache replacement and terminal admission gates passed");
    }

    namespace FrameFixture = JuicerCudaTest::PreparedFrameFixture;

    struct GpuReuseCall {
        explicit GpuReuseCall(JuicerCuda::ResourceManager::DeviceContextKey contextKey)
            : key(contextKey) {}

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
        auto frame = JuicerProcess::root().prepare_cuda_frame(
            call.key, submission, prepared.preparation, {}, call.stream, error, prepared.hostPreparation);
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

    void gpu_reuse_tests() {
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
                    controls.printProfileKey = "kodak_portra_endura";
                }
                if (scenario == 2 || scenario == 3) {
                    controls.inputColorSpace = scenario == 2 ? 1 : 3;
                    controls.inputCctfDecoding = 1;
                }
                controls.grainControls.active = false;
                controls.dirCouplers.active = false;
                auto product = prepare(controls);
                controls.outputCctfEncoding = 0;
                auto encodedOff = prepare(controls);
                GpuReuseCall call{key};
                call.sequence = sequence;
                FrameFixture::FrameFacts facts{};
                facts.renderExtent = {0, 0, 16, 16};
                facts.fullFrameExtent = facts.renderExtent;
                facts.pixelDefinition = facts.renderExtent;
                facts.pixelSizeUm = 6.0f;
                facts.filmFormatLongEdgeMm = controls.cameraFilmFormatLongEdgeMm;
                facts.frameRate = 24.0;
                facts.sessionSeed = 0x243f6a8885a308d3ull;
                facts.clipToken = 0x13198a2e03707344ull;
                for (const PreparedState* prepared : {&product, &encodedOff, &product}) {
                    std::string diagnostic;
                    require(FrameFixture::with_prepared_frame_inputs(
                                *prepared, facts, &call, &prepare_and_check_gpu_reuse, diagnostic),
                            diagnostic);
                }
                sequence = call.sequence;
                std::printf("GPU scanner route=%d method=%d input=%d decode=%d preparation_us cold=%lld cctf=%lld warm=%lld identity=%llu exact_values=%zu allocations=[%p,%p,%p,%p,%p,%p]\n",
                            selectedRoute,
                            method,
                            controls.inputColorSpace,
                            controls.inputCctfDecoding,
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
                require(JuicerProcess::root().retire_idle_context(0, context, diagnostic), diagnostic);
            }
        }
        std::puts("GPU reuse: Hanatos/Arctic both polarities and retained Mallett input-color cases; direct/print scanner and TC allocations/content/identity reuse; prepared binding leaves math faults unconsumed");
    }

} // namespace

namespace JuicerAssets::ProfileTest {
    void film_view(FjFilmProfileView& view) {
        if (constructionFault == 5) {
            constructionFault = 0;
            --view.tables.density_curves_cmy.count;
        }
    }
    void print_view(FjPrintProfileView& view) {
        (void)view;
    }
    void after_tables() {
        const unsigned fault = std::exchange(constructionFault, 0);
        if (fault == 3) {
            throw std::runtime_error("standard construction control");
        }
        if (fault != 0) {
            const auto category = fault == 1 ? FJ_STATUS_CANCELLED : fault == 4 ? FJ_STATUS_UNSUPPORTED_INPUT
                                                                                : FJ_STATUS_INTERNAL_FAILURE;
            throw JuicerCuda::ExecutionFailure{{{category, FJ_API_NONE, 0}, "construction rethrow control"}, fault == 2 || fault == 4};
        }
    }
    void before_publication() {}
    void before_catalog_publication() {}
} // namespace JuicerAssets::ProfileTest

int main(int argc, char** argv) {
    JuicerCuda::Owner owner;
    try {
        require(argc == 4, "resources, immutable fixture and group required");
        const std::string group = argv[3];
        if (group == "fixture-format") {
            shared_fixture_loading_tests(argv[1]);
            return 0;
        }
        owner.create(argv[1]);
        JuicerProcess::root().ensure_bootstrap();
        const auto fixture = PrintIlluminantFixtures::apply(argv[2], load_exposure_qualified_fixture(argv[2]));
        if (group == "leaf") {
            leaf_tests(fixture);
        } else if (group == "preparation") {
            preparation_tests(fixture);
        } else if (group == "cat02-leaf") {
            cat02_leaf_tests(fixture);
        } else if (group == "cat02-preparation") {
            cat02_preparation_tests(fixture, argv[2]);
        } else if (group == "input-leaf") {
            input_leaf_tests(fixture);
        } else if (group == "input-preparation") {
            input_preparation_tests(fixture, argv[2]);
        } else if (group == "reuse-gpu") {
            gpu_reuse_tests();
        } else {
            throw std::runtime_error("unknown color test group");
        }
        expect_status(owner.close(), FJ_STATUS_SUCCESS);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Color tests: %s\n", error.what());
        return 1;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        std::fprintf(stderr, "Color construction status=%u: %s\n", error.failure.status.category, error.failure.diagnostic.c_str());
        return 1;
    }
}
