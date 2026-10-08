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
#include <mutex>
#include <thread>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>

#include "SpectralProcessing.h"
#include "ColorTransforms.h"
#include "GamutCompression.h"
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
#include "juicer_cuda_owner.h"

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
    template <typename Range>
    Json bits(const Range& values) {
        Json out = Json::array();
        for (float value : values) {
            out.push_back(std::bit_cast<std::uint32_t>(value));
        }
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
    FocusedRenderStateBuildProduct build(const ParamSnapshot& input) {
        FocusedRenderStateBuildProduct out;
        std::string diagnostic;
        const bool built = Spektrafilm::scan_route_is_print(input.scanRoute) ? build_print_render_state_product(input, out, diagnostic) : build_direct_render_state_product(input, out, diagnostic);
        require(built, "complete recipe: " + diagnostic);
        return out;
    }
    Json complete_result(const FocusedRenderStateBuildProduct& product) {
        const auto& recipe = product.recipe;
        const auto& raw = recipe.filmRaw;
        const auto& reference = recipe.scannerOutput.syntheticFilmReference;
        const auto& balance = recipe.print.balance;
        return {{"input_adapt", bits(raw.inputXyzAdapt)}, {"source_white", bits(raw.inputNominalWhiteXYZ)}, {"projection_white", bits(raw.projectionWhiteXYZ)}, {"baseline_raw", bits(reference.baselineRawRgb)}, {"compensated_raw", bits(reference.compensatedRawRgb)}, {"baseline_density", bits(reference.baselineDensityCmy)}, {"compensated_density", bits(reference.compensatedDensityCmy)}, {"print_baseline_raw", bits(balance.baselinePrintRawRgb)}, {"print_compensated_raw", bits(balance.compensatedPrintRawRgb)}, {"print_factors", bits(std::array{balance.factorMidgray, balance.factorMidgrayComp, balance.normalizer})}, {"hashes", {raw.hash, raw.tcLutHash, reference.hash, balance.hash, recipe.print.hash, recipe.scannerOutput.hash, recipe.scannerOutput.outputGamut.hash, recipe.hash, product.payload.uploadCoreHash, product.payload.scannerHash}}, {"gamut_entry", bits(recipe.scannerOutput.outputGamut.transform.nativeRgbToD65Xyz)}, {"gamut_exit", bits(recipe.scannerOutput.outputGamut.transform.d65XyzToNativeRgb)}};
    }
    Json transform_result(int space) {
        Gamut::OutputGamutTransform out;
        std::string diagnostic;
        require(Gamut::build_output_gamut_transform(static_cast<OutputEncoding::ColorSpace>(space), out, diagnostic), diagnostic);
        return {{"entry", bits(out.nativeRgbToD65Xyz)}, {"exit", bits(out.d65XyzToNativeRgb)}, {"hash", out.hash}, {"contract_hash", Gamut::output_boundary_contract_hash(out)}};
    }
    Json table_result(int space) {
        Gamut::OutputGamutTransform transform;
        Gamut::OutputBoundaryTable table;
        std::string diagnostic;
        require(Gamut::build_output_gamut_transform(static_cast<OutputEncoding::ColorSpace>(space), transform, diagnostic), diagnostic);
        require(Gamut::build_output_boundary_table(transform, table), table.diagnostic);
        return {{"cmax", bits(table.cmax)}, {"hash", table.hash}, {"transform_hash", table.transformHash}, {"contract_hash", table.contractHash}};
    }
    void expect_status(FjStatus status, std::uint32_t category) {
        require(status.category == category && status.api == FJ_API_NONE && status.native_code == 0, "color status category/API/code");
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
                return fj_legacy_adapt_cat02(white.data(), white.data(), white.data(), adapted.data());
            case FJ_TEST_INPUT_MATRICES:
                return fj_legacy_input_matrices(FJ_INPUT_DWG, &matrices);
            case FJ_TEST_INPUT_TO_DWG:
                return fj_legacy_input_to_dwg(&input, white.data(), 1, rgb.data(), adapted.data());
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                return fj_legacy_input_to_linear_srgb(&input, white.data(), identity.data(), rgb.data(), adapted.data());
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                return fj_legacy_linear_srgb_to_xyz(white.data(), adapted.data());
            case FJ_TEST_DWG_TO_XYZ:
                return fj_legacy_dwg_to_xyz(white.data(), adapted.data());
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
            if (status.category != FJ_STATUS_SUCCESS) {
                std::abort();
            }
            fj_test_profile_fault(0);
            constructionFault = 0;
            set_pending_capture_test_hook(nullptr, nullptr);
        }
    };
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
            expect_status(fj_test_color_arm_fault(operation, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation == 1 ? 2 : 1), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            for (const auto& invalid : std::array<std::array<std::uint32_t, 3>, 3>{{{0, 1, 1}, {operation, 0, 1}, {operation, 1, 99}}}) {
                expect_status(fj_test_color_arm_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(fj_test_color_arm_fault(invalid[0], invalid[1], invalid[2]), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            }
            expect_status(fj_test_color_arm_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            expect_status(fj_test_color_arm_fault(operation, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
            expect_status(fj_test_color_clear_fault(), FJ_STATUS_SUCCESS);
        }
        expect_status(fj_test_color_arm_fault(FJ_TEST_CAT16_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
        FjStatus otherThread{};
        std::thread worker([&] {
            otherThread = raw_call(FJ_TEST_CAT16_MATRIX);
        });
        worker.join();
        expect_status(otherThread, FJ_STATUS_SUCCESS);
        expect_status(raw_call(FJ_TEST_CAT16_MATRIX), FJ_STATUS_UNSUPPORTED_INPUT);
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
    void failed(const PendingRenderAdmissionResult& result, const FaultCase& selected, const std::string& film = "kodak_portra_400") {
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
        require(result.diagnostic.find(operation) != std::string::npos, "original operation diagnostic");
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
                    const std::uint32_t calls = operation == 1 ? 3 : 2;
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
                            expect_status(fj_test_color_arm_fault(selected.operation, selected.callIndex, selected.fault), FJ_STATUS_SUCCESS);
                            const auto result = admit_pending_render_state(state);
                            failed(result, selected);
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
                                expect_status(fj_test_color_arm_fault(selected.operation, selected.callIndex, selected.fault), FJ_STATUS_SUCCESS);
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
            FaultScope cleanup;
            InstanceState state;
            const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Hanatos2025);
            pending(state, controls);
            const auto old = admit_pending_render_state(state);
            require(admitted(old, route == 1), "construction control baseline");
            auto changed = controls;
            changed.cameraExposureCompensationEv = 1.0;
            const auto hash = state.lastHash.load();
            const auto counter = state.buildCounterNext.load();
            for (unsigned fault : {1u, 2u}) {
                expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
                pending(state, changed);
                fj_test_profile_fault(fault);
                if (fault == 1) {
                    const auto result = admit_pending_render_state(state);
                    require(result.status == PendingRenderAdmissionStatus::RebuildFailed && !result.directState && !result.printState && result.diagnostic.find("InternalFailure") != std::string::npos && result.diagnostic.find("asset boundary panic") != std::string::npos, "shared asset InternalFailure admission mapping: " + result.diagnostic);
                } else {
                    bool propagated = false;
                    try {
                        (void)admit_pending_render_state(state);
                    } catch (const std::bad_alloc&) {
                        propagated = true;
                    }
                    require(propagated, "real asset capacity remains bad_alloc");
                }
                unchanged(state, old, hash, counter);
            }
            expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            constructionFault = 5;
            pending(state, changed);
            const auto malformed = admit_pending_render_state(state);
            require(malformed.status == PendingRenderAdmissionStatus::RebuildFailed && !malformed.directState && !malformed.printState && malformed.diagnostic.find("InternalFailure") != std::string::npos && constructionFault == 0, "shared malformed asset-view InternalFailure mapping: " + malformed.diagnostic);
            unchanged(state, old, hash, counter);
            for (unsigned fault : {1u, 2u, 3u, 4u}) {
                expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
                constructionFault = fault;
                pending(state, changed);
                bool propagated = false;
                try {
                    (void)admit_pending_render_state(state);
                } catch (const JuicerCuda::ExecutionFailure& error) {
                    require(error.failure.status.category == (fault == 1 ? FJ_STATUS_CANCELLED : fault == 4 ? FJ_STATUS_UNSUPPORTED_INPUT
                                                                                                            : FJ_STATUS_INTERNAL_FAILURE) &&
                                error.failure.status.api == FJ_API_NONE && error.failure.status.native_code == 0 && error.deferredDirError == (fault == 2 || fault == 4) && error.failure.diagnostic == "construction rethrow control",
                            "typed category/deferred/diagnostic preserved");
                    propagated = true;
                } catch (const std::runtime_error& error) {
                    require(fault == 3 && std::string(error.what()) == "standard construction control", "standard exception preserved");
                    propagated = true;
                }
                require(propagated && constructionFault == 0, "rethrow control executed inside construction");
                unchanged(state, old, hash, counter);
            }
            changed.filmProfileKey = "missing-cat16-control-profile";
            pending(state, changed);
            const auto ordinary = admit_pending_render_state(state);
            require(ordinary.status == PendingRenderAdmissionStatus::RebuildFailed && !ordinary.directState && !ordinary.printState, "ordinary false control");
            require(route == 1 ? !JuicerAtomic::load_shared_ptr(&state.activePrintState) : !JuicerAtomic::load_shared_ptr(&state.activeDirectState), "ordinary false keeps publication clearing");
            require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "ordinary false counter/hash");
            pending(state, controls);
            require(admitted(admit_pending_render_state(state), route == 1), "ordinary false recovery");
        }
        std::puts("Construction: shared asset panic, capacity, other/deferred/standard rethrow and ordinary false controls passed");
    }
    Scanner::ScannerSpectralLutDescriptor scanner_descriptor(const FocusedRenderStateBuildProduct& product) {
        Scanner::ScannerSpectralLutDescriptor out;
        std::string diagnostic;
        const auto& recipe = product.recipe;
        const bool ok = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic) : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic);
        require(ok, diagnostic);
        return out;
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
            const auto product = build(controls);
            require(id == fixture.at("recipes").at(i).at("id").get<std::string>(), "fixed recipe case membership");
            require(complete_result(product) == fixture.at("recipes").at(i).at("expected"), id + " independent complete arrays/identities");
            require(complete_result(build(controls)) == complete_result(product), id + " repeat complete construction");
        }
        for (int route = 0; route < 2; ++route) {
            const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Hanatos2025);
            const auto product = build(controls);
            auto changed = controls;
            changed.outputCctfEncoding = 0;
            const auto encodedOff = build(changed);
            require(product.recipe.filmRaw.inputXyzAdapt == encodedOff.recipe.filmRaw.inputXyzAdapt && product.recipe.filmRaw.tcLutHash == encodedOff.recipe.filmRaw.tcLutHash, "encoding change preserves film/TC family");
            require(product.payload.outputBoundaryTable == encodedOff.payload.outputBoundaryTable, "encoding change reuses actual boundary owner");
            require(scanner_descriptor(product).hash == scanner_descriptor(encodedOff).hash, "encoding change preserves scanner descriptor identity");
            require(product.recipe.scannerOutput.hash != encodedOff.recipe.scannerOutput.hash && product.recipe.hash != encodedOff.recipe.hash && product.payload.scannerColor.hash != encodedOff.payload.scannerColor.hash, "encoding identities change at actual owners");
            const auto retained = product.payload.outputBoundaryTable;
            const auto retainedBits = bits(retained->cmax);
            expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            require(bits(retained->cmax) == retainedBits && complete_result(product) == complete_result(build(controls)), "retained state/table stays readable across cache release");
            const auto rebuilt = build(controls);
            require(rebuilt.payload.outputBoundaryTable != retained && bits(rebuilt.payload.outputBoundaryTable->cmax) == retainedBits, "cache release creates new equal owner without mutating retained table");
            InstanceState state;
            pending(state, controls);
            const auto old = admit_pending_render_state(state);
            const auto counter = state.buildCounterNext.load();
            FaultScope cleanup;
            expect_status(fj_test_color_arm_fault(FJ_TEST_CAT16_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
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
            adapted = JuicerColor::adapt_cat02(xyz, whites);
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
        FocusedRenderStateBuildProduct build(const ParamSnapshot& input) {
            FocusedRenderStateBuildProduct out;
            std::string diagnostic;
            const bool built = Spektrafilm::scan_route_is_print(input.scanRoute) ? build_print_render_state_product(input, out, diagnostic) : build_direct_render_state_product(input, out, diagnostic);
            require(built, "complete recipe: " + diagnostic);
            return out;
        }
        Json color_result(const Scanner::ColorRuntime& color) {
            return {{"cat02", bits(color.cat02)}, {"xyz_to_rgb", bits(color.xyzToRgb)}, {"illuminant_xyz", bits(color.illuminantXYZ)}, {"encoding", {static_cast<int>(color.encoding.colorSpace), color.encoding.applyCctfEncoding, color.encoding.inputIsOutputSpace}}, {"gamut_recipe_hash", color.outputGamutRecipeHash}, {"hash", color.hash}};
        }
        Json tables_result(const Spectral::SpectralTables& tables) {
            return {{"lambda", bits(tables.lambda)}, {"K", tables.K}, {"delta_lambda", bits(std::array{tables.deltaLambda})}, {"inv_yn", bits(std::array{tables.invYn})}, {"white_xyz", bits(tables.whiteXYZ)}, {"ref_illum_white_xyz", bits(tables.refIllumWhiteXYZ)}, {"ax", bits(tables.Ax)}, {"ay", bits(tables.Ay)}, {"az", bits(tables.Az)}, {"xbar", bits(tables.Xbar)}, {"ybar", bits(tables.Ybar)}, {"zbar", bits(tables.Zbar)}, {"illum", bits(tables.illum)}, {"eps_c", bits(tables.epsC)}, {"eps_m", bits(tables.epsM)}, {"eps_y", bits(tables.epsY)}, {"base_density_min", bits(tables.baseDensityMin)}, {"base_density_mid", bits(tables.baseDensityMid)}, {"has_baseline", tables.hasBaseline}, {"baseline_mix", bits(std::array{tables.densityBaselineMixReference})}, {"illuminant_hash", tables.illuminantHash}, {"tables_hash", tables.tablesHash}};
        }
        template <typename State>
        Json scanner_descriptor(const State& product) {
            Scanner::ScannerSpectralLutDescriptor out;
            std::string diagnostic;
            const auto& recipe = product.recipe;
            const bool ok = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic) : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic);
            require(ok, diagnostic);
            return {{"route", static_cast<int>(out.route)}, {"medium", static_cast<int>(out.medium)}, {"polarity", static_cast<int>(out.polarity)}, {"density_bounds_hash", out.densityBoundsHash}, {"channel_density_hash", out.channelDensityHash}, {"base_density_hash", out.baseDensityHash}, {"illuminant_hash", out.scanIlluminantHash}, {"observer_hash", out.observerHash}, {"resolution", out.lutResolution}, {"hash", out.hash}};
        }
        template <typename State>
        Json complete_result(const State& product) {
            const auto& recipe = product.recipe;
            const auto& raw = recipe.filmRaw;
            const auto& reference = recipe.scannerOutput.syntheticFilmReference;
            const auto& balance = recipe.print.balance;
            const auto& config = product.payload.filmRawConfig;
            return {{"input_rgb_to_xyz", bits(raw.inputRgbToXyz)}, {"input_adapt", bits(raw.inputXyzAdapt)}, {"xyz_to_linear_srgb", bits(raw.xyzToLinearSrgb)}, {"source_white", bits(raw.inputNominalWhiteXYZ)}, {"projection_white", bits(raw.projectionWhiteXYZ)}, {"config_rgb_to_xyz", bits(config.inputRGBToXYZ.m)}, {"config_adapt", bits(config.inputXYZAdapt.m)}, {"raw_midgray", bits(config.rawMidgray)}, {"midgray_dwg", bits(config.midgrayDWG)}, {"midgray_scale_green", bits(std::array{config.midgrayScale, config.rawMidgrayGreen})}, {"config_whites", {bits(config.inputWhiteXYZ), bits(config.workingWhiteXYZ), bits(config.refIllumWhiteXYZ)}}, {"config_flags", {static_cast<int>(config.inputColorSpace), config.applyCctfDecoding, static_cast<int>(config.spectralUpsamplingMode), config.applyInputChromaticAdapt, config.hasRefIllumWhite, config.valid}}, {"spd_s_inv", bits(product.payload.spdSInv)}, {"baseline_raw", bits(reference.baselineRawRgb)}, {"compensated_raw", bits(reference.compensatedRawRgb)}, {"baseline_density", bits(reference.baselineDensityCmy)}, {"compensated_density", bits(reference.compensatedDensityCmy)}, {"print_baseline_raw", bits(balance.baselinePrintRawRgb)}, {"print_compensated_raw", bits(balance.compensatedPrintRawRgb)}, {"print_factors", bits(std::array{balance.factorMidgray, balance.factorMidgrayComp, balance.normalizer})}, {"hashes", {raw.hash, raw.tcLutHash, reference.hash, balance.hash, recipe.print.hash, recipe.scannerOutput.hash, recipe.scannerOutput.outputGamut.hash, recipe.hash, product.payload.uploadCoreHash, product.payload.scannerHash}}, {"scanner_color", color_result(product.payload.scannerColor)}, {"scanner_tables", tables_result(product.payload.scannerTables)}, {"scanner_descriptor", scanner_descriptor(product)}, {"gamut_entry", bits(recipe.scannerOutput.outputGamut.transform.nativeRgbToD65Xyz)}, {"gamut_exit", bits(recipe.scannerOutput.outputGamut.transform.d65XyzToNativeRgb)}};
        }
        Scanner::ScannerIlluminant scanner_illuminant(const FocusedRenderStateBuildProduct& product) {
            const auto& tables = product.payload.scannerTables;
            Scanner::ScannerIlluminant out;
            out.curve.linear = tables.illum;
            out.curve.lambda_nm = tables.lambda;
            std::copy_n(product.payload.scannerColor.illuminantXYZ, 3, out.whiteXYZ);
            const auto span = [](const std::vector<float>& samples) -> FjFloatSpan {
                return {samples.data(), samples.size()};
            };
            const FjSpectralWhiteInput input{{span(Spectral::gXBar.linear), span(Spectral::gYBar.linear), span(Spectral::gZBar.linear)}, span(out.curve.linear)};
            FjSpectralWhite white{};
            FjSpectralWhiteFailure failure{};
            std::array<char, 256> bytes{};
            FjErrorBuffer diagnostic{bytes.data(), bytes.size(), 0};
            require(fj_test_spectral_white(&input, &white, &failure, &diagnostic).category == FJ_STATUS_SUCCESS, "direct core scanner illuminant fixture");
            out.normalization = white.normalization;
            std::copy_n(white.white_xy, 2, out.whiteXY);
            out.hash = tables.illuminantHash;
            require(out.hash != 0, "actual completed scanner illuminant hash");
            return out;
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
        Json helper_result(const Json& input, const Spectral::SpectralTables& canonicalTables, const std::array<float, 9>& sInv, bool hanatos) {
            const auto rgb = triplet(input.at("rgb"));
            const auto reference = triplet(input.at("reference_white"));
            std::array<float, 3> xyz{}, white{}, adapted{};
            expect_status(fj_test_dwg_to_xyz(rgb.data(), xyz.data()), FJ_STATUS_SUCCESS);
            if (hanatos) {
                Spectral::sanitize_nonfinite_triplet(xyz.data());
                Spectral::sanitize_ref_white_or_dwg(reference.data(), white.data());
            } else {
                auto original = xyz;
                Spectral::sanitize_nonnegative_triplet_sp(xyz.data(), original.data());
                Spectral::sanitize_nonnegative_triplet_sp(white.data(), reference.data());
                if (white[1] <= 0.0f)
                    std::copy_n(Spectral::gDWG_WhitePoint_XYZ, 3, white.data());
            }
            const Spectral::ChromaticAdaptationWhites whites{Spectral::gDWG_WhitePoint_XYZ, white.data()};
            adapted = JuicerColor::adapt_cat02(xyz, whites);
            if (hanatos)
                Spectral::sanitize_nonfinite_triplet(adapted.data());
            else
                Spectral::clamp_triplet_nonnegative(adapted.data());
            std::vector<float> spectrum;
            if (hanatos)
                Spectral::reconstruct_Ee_from_DWG_RGB_hanatos(rgb.data(), spectrum, reference.data());
            else {
                auto tables = canonicalTables;
                std::copy(reference.begin(), reference.end(), tables.refIllumWhiteXYZ);
                Spectral::reconstruct_Ee_from_DWG_RGB_with_tables(rgb.data(), tables, sInv.data(), spectrum);
            }
            require(spectrum.size() == 81, "canonical helper spectrum size");
            return {{"pre_xyz", bits(xyz)}, {"consumer_white", bits(white)}, {"post_adapt_xyz", bits(adapted)}, {"spectrum", bits(spectrum)}};
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
                return fj_legacy_adapt_cat02(pointer(0), pointer(1), pointer(2), output(3));
            case FJ_TEST_INPUT_MATRICES:
                return fj_legacy_input_matrices(FJ_INPUT_DWG, selected.slot == 0 ? nullptr : &matrices);
            case FJ_TEST_INPUT_TO_DWG:
                return fj_legacy_input_to_dwg(conversion, pointer(1), 1, selected.slot == 2 ? nullptr : rgb.data(), output(3));
            case FJ_TEST_INPUT_TO_LINEAR_SRGB:
                return fj_legacy_input_to_linear_srgb(conversion, pointer(1), selected.slot == 2 ? nullptr : identity.data(), selected.slot == 3 ? nullptr : rgb.data(), output(4));
            case FJ_TEST_LINEAR_SRGB_TO_XYZ:
                return fj_legacy_linear_srgb_to_xyz(pointer(0), output(1));
            case FJ_TEST_DWG_TO_XYZ:
                return fj_legacy_dwg_to_xyz(pointer(0), output(1));
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
                expect_status(fj_test_color_arm_fault(operation, 2, fault), FJ_STATUS_SUCCESS);
                for (const auto unrelated : operations) {
                    if (unrelated != operation) {
                        expect_status(raw_call(unrelated), FJ_STATUS_SUCCESS);
                    }
                }
                expect_status(facade_call(operation), FJ_STATUS_SUCCESS);
                const int pointers = color_pointer_count(operation);
                for (int slot = 0; slot < pointers; ++slot) {
                    expect_status(null_color_call({.operation = operation, .slot = slot}), FJ_STATUS_UNSUPPORTED_INPUT);
                }
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                expect_status(raw_call(operation), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                expect_status(fj_test_color_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                FjStatus otherThread{};
                std::thread worker([&] {
                    otherThread = raw_call(operation);
                });
                worker.join();
                expect_status(otherThread, FJ_STATUS_SUCCESS);
                expect_status(raw_call(operation), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            }
            for (const auto& invalid : std::array<std::array<std::uint32_t, 3>, 5>{{{0, 1, 1}, {11, 1, 1}, {99, 1, 1}, {operation, 0, 1}, {operation, 1, 99}}}) {
                expect_status(fj_test_color_arm_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(fj_test_color_arm_fault(invalid[0], invalid[1], invalid[2]), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            }
            for (const auto replacement : operations) {
                expect_status(fj_test_color_arm_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                expect_status(fj_test_color_arm_fault(replacement, 2, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
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
                const auto adapted = JuicerColor::adapt_cat02(xyz, whites);
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
    struct RetainedValues {
        Json complete;
        Json exposureTables;
        std::vector<std::uint32_t> tcRgba;
        std::vector<std::uint32_t> printIlluminant;
        std::vector<std::uint32_t> boundary;
        std::array<std::uint64_t, 3> boundaryHashes{};
        std::string boundaryDiagnostic;
    };
    template <typename State>
    RetainedValues retain_values(const State& state) {
        RetainedValues out;
        out.complete = Cat02Fixtures::complete_result(state);
        out.exposureTables = Cat02Fixtures::tables_result(state.payload.exposureTables);
        if (state.payload.filmTcLut) {
            out.tcRgba = retained_bits(state.payload.filmTcLut->rgba);
        }
        if (state.payload.printMainIlluminant) {
            out.printIlluminant = retained_bits(*state.payload.printMainIlluminant);
        }
        require(bool(state.payload.outputBoundaryTable), "completed state boundary owner");
        const auto& boundary = *state.payload.outputBoundaryTable;
        require(boundary.valid, "retained completed boundary table validity");
        out.boundary = retained_bits(boundary.cmax);
        out.boundaryHashes = {boundary.transformHash, boundary.contractHash, boundary.hash};
        out.boundaryDiagnostic = boundary.diagnostic;
        return out;
    }
    template <typename State>
    void require_retained_values(const State& state, const RetainedValues& expected) {
        const auto actual = retain_values(state);
        require(actual.complete == expected.complete && actual.exposureTables == expected.exposureTables && actual.tcRgba == expected.tcRgba && actual.printIlluminant == expected.printIlluminant && actual.boundary == expected.boundary && actual.boundaryHashes == expected.boundaryHashes && actual.boundaryDiagnostic == expected.boundaryDiagnostic, "retained complete route/color/tables/TC/boundary values remain exact");
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
                        expect_status(fj_test_color_arm_fault(selected.operation, index, fault), FJ_STATUS_SUCCESS);
                        const auto result = admit_pending_render_state(state);
                        failed(result, selected, controls.filmProfileKey);
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
                            expect_status(fj_test_color_arm_fault(selected.operation, index, fault), FJ_STATUS_SUCCESS);
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
    void require_cat02_scalar_failure(Function&& function, std::uint32_t fault) {
        FaultScope cleanup;
        expect_status(fj_test_color_arm_fault(FJ_TEST_CAT02_ADAPT, 1, fault), FJ_STATUS_SUCCESS);
        bool thrown = false;
        try {
            function();
        } catch (const JuicerCuda::ExecutionFailure& error) {
            require(error.failure.status.category == (fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE) && error.failure.status.api == FJ_API_NONE && error.failure.status.native_code == 0 && !error.deferredDirError && error.failure.diagnostic.find("CAT02 scalar") != std::string::npos, "CAT02 scalar typed failure and diagnostic preserved");
            thrown = true;
        }
        require(thrown, "actual scalar consumer propagates CAT02 failure");
        expect_status(raw_call(FJ_TEST_CAT02_ADAPT), FJ_STATUS_SUCCESS);
    }
    void cat02_helper_tests(const Json& fixture) {
        const auto product = Cat02Fixtures::build(Cat02Fixtures::snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Mallett2019));
        const auto& tables = product.payload.exposureTables;
        const auto& sInv = product.payload.spdSInv;
        require(Cat02Fixtures::tables_result(tables) == fixture.at("helper_tables") && bits(sInv) == fixture.at("helper_s_inv"), "independent canonical helper tables/S inverse");
        require(Json{{"K", Spectral::gShape.K}, {"dwg_white", bits(Spectral::gDWG_WhitePoint_XYZ)}} == fixture.at("helper_hanatos_axes"), "independent retained Hanatos helper axes/white");
        const auto cases = Cat02Fixtures::helper_inputs();
        require(cases.size() == fixture.at("helpers").size(), "retained helper case membership");
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& input = cases[i];
            const auto& row = fixture.at("helpers").at(i);
            require(input.at("id") == row.at("id") && input.at("rgb") == row.at("rgb") && input.at("reference_white") == row.at("reference_white"), "frozen retained helper inputs");
            require(Cat02Fixtures::helper_result(input, tables, sInv, true) == row.at("hanatos_expected"), input.at("id").get<std::string>() + " signed Hanatos sanitation/spectrum");
            require(Cat02Fixtures::helper_result(input, tables, sInv, false) == row.at("tables_expected"), input.at("id").get<std::string>() + " tables sanitation/spectrum");
        }
        const std::array<float, 3> rgb{2.0f, -0.25f, 0.5f};
        const Spectral::ChromaticAdaptationWhites whites{Spectral::gDWG_WhitePoint_XYZ, tables.refIllumWhiteXYZ};
        const auto held = retain_values(product);
        for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
            require_cat02_scalar_failure([&] {
                (void)JuicerColor::adapt_cat02(rgb, whites);
            },
                                         fault);
            require_cat02_scalar_failure([&] {
                std::vector<float> spectrum;
                Spectral::reconstruct_Ee_from_DWG_RGB_hanatos(rgb.data(), spectrum, tables.refIllumWhiteXYZ);
            },
                                         fault);
            require_cat02_scalar_failure([&] {
                std::vector<float> spectrum;
                Spectral::reconstruct_Ee_from_DWG_RGB_with_tables(rgb.data(), tables, sInv.data(), spectrum);
            },
                                         fault);
            require_retained_values(product, held);
        }
        std::printf("CAT02 retained helpers: %zu independent pre/post sanitation/spectrum cases per helper and both typed failure categories passed\n", cases.size());
    }
    void cat02_control_tests() {
        for (int route = 0; route < 4; ++route) {
            for (int method = 0; method < 3; ++method) {
                const auto controls = Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                const auto product = Cat02Fixtures::build(controls);
                const auto productHeld = retain_values(product);
                auto decoding = controls;
                decoding.inputCctfDecoding = 1;
                const auto decoded = Cat02Fixtures::build(decoding);
                require(product.recipe.filmRaw.inputXyzAdapt == decoded.recipe.filmRaw.inputXyzAdapt && product.recipe.filmRaw.hash != decoded.recipe.filmRaw.hash && product.recipe.hash != decoded.recipe.hash, "input CCTF decoding keeps adaptation and changes film identity");
                auto input = controls;
                input.inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::ACES2065_1);
                const auto inputChanged = Cat02Fixtures::build(input);
                require(product.recipe.filmRaw.inputNominalWhiteXYZ != inputChanged.recipe.filmRaw.inputNominalWhiteXYZ && product.recipe.filmRaw.inputXyzAdapt != inputChanged.recipe.filmRaw.inputXyzAdapt && product.recipe.filmRaw.hash != inputChanged.recipe.filmRaw.hash && product.recipe.hash != inputChanged.recipe.hash, "input space/white changes contributing matrix and film identity");
                require(Cat02Fixtures::color_result(product.payload.scannerColor) == Cat02Fixtures::color_result(inputChanged.payload.scannerColor), "input space leaves selected scanner color values/identity unchanged");
                auto output = controls;
                output.outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::DCI_P3);
                const auto outputChanged = Cat02Fixtures::build(output);
                require(bits(product.payload.scannerColor.cat02) != bits(outputChanged.payload.scannerColor.cat02) && product.payload.scannerColor.hash != outputChanged.payload.scannerColor.hash && product.recipe.hash != outputChanged.recipe.hash, "output space/white changes scanner matrix/color/final identity");
                require(product.recipe.filmRaw.hash == outputChanged.recipe.filmRaw.hash && product.recipe.filmRaw.inputXyzAdapt == outputChanged.recipe.filmRaw.inputXyzAdapt, "output space preserves film identity/matrix");
                auto encoding = controls;
                encoding.outputCctfEncoding = 0;
                const auto encodedOff = Cat02Fixtures::build(encoding);
                require(product.recipe.filmRaw.inputXyzAdapt == encodedOff.recipe.filmRaw.inputXyzAdapt && product.recipe.filmRaw.hash == encodedOff.recipe.filmRaw.hash && product.recipe.filmRaw.tcLutHash == encodedOff.recipe.filmRaw.tcLutHash, "output CCTF-only changes preserve film/TC family");
                require(product.payload.outputBoundaryTable == encodedOff.payload.outputBoundaryTable, "output CCTF-only reuses actual boundary table owner");
                require(Cat02Fixtures::tables_result(product.payload.scannerTables) == Cat02Fixtures::tables_result(encodedOff.payload.scannerTables) && Cat02Fixtures::scanner_descriptor(product) == Cat02Fixtures::scanner_descriptor(encodedOff), "output CCTF-only preserves complete scanner tables/descriptor");
                require(bits(product.payload.scannerColor.cat02) == bits(encodedOff.payload.scannerColor.cat02) && product.payload.scannerColor.hash != encodedOff.payload.scannerColor.hash && product.recipe.scannerOutput.hash != encodedOff.recipe.scannerOutput.hash && product.recipe.hash != encodedOff.recipe.hash, "output CCTF-only changes identities at color/output owners");
                InstanceState state;
                pending(state, controls);
                const auto old = admit_pending_render_state(state);
                require(admitted(old, route % 2 == 1), "unchanged CAT02 admission baseline");
                const auto hash = state.lastHash.load();
                const auto counter = state.buildCounterNext.load();
                const auto held = retain_values(old);
                FaultScope cleanup;
                expect_status(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
                const auto repeated = admit_pending_render_state(state);
                require(repeated.directState == old.directState && repeated.printState == old.printState, "unchanged CAT02 admission reuses exact publication");
                unchanged(state, old, hash, counter);
                expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
                require_retained_values(old, held);
                require_retained_values(product, productHeld);
                const auto rebuilt = Cat02Fixtures::build(controls);
                const auto& oldPayload = old.printState ? old.printState->payload : old.directState->payload;
                require(rebuilt.payload.outputBoundaryTable != oldPayload.outputBoundaryTable && retained_bits(rebuilt.payload.outputBoundaryTable->cmax) == held.boundary, "cache release makes a new equal table without mutating admitted hold");
                pending(state, output);
                const auto replacement = admit_pending_render_state(state);
                require(admitted(replacement, route % 2 == 1) && (replacement.directState != old.directState || replacement.printState != old.printState) && state.lastHash.load() == hash_params(output) && state.buildCounterNext.load() == counter + 1, "output-space state replacement publishes complete owner/hash/counter");
                require_retained_values(old, held);
            }
        }
    }
    void cat02_preparation_tests(const Json& fixture, const std::filesystem::path& fixturePath) {
        cat02_admission_tests();
        const auto cases = Cat02Fixtures::scenarios();
        require(cases.size() == fixture.at("recipes").size(), "frozen CAT02 recipe case membership");
        std::vector<Json> otherPresets;
        for (const auto* preset : {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"}) {
            const auto path = fixturePath.parent_path() / (std::string("cat02-") + preset + ".json");
            if (path != fixturePath) {
                otherPresets.push_back(load_exposure_qualified_fixture(path));
            }
        }
        bool wrongPresetRejected = false;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [id, controls] = cases[i];
            const auto& row = fixture.at("recipes").at(i);
            require(id == row.at("id").get<std::string>(), "frozen CAT02 recipe ordered case identity");
            const auto& settings = row.at("controls");
            require(settings.at("filmProfileKey") == controls.filmProfileKey && settings.at("printProfileKey") == controls.printProfileKey && settings.at("scanRoute") == static_cast<int>(controls.scanRoute) && settings.at("spectralUpsamplingMode") == controls.spectralUpsamplingMode && settings.at("inputColorSpace") == controls.inputColorSpace && settings.at("inputCctfDecoding") == controls.inputCctfDecoding && settings.at("outputColorSpace") == controls.outputColorSpace && settings.at("outputCctfEncoding") == controls.outputCctfEncoding && settings.at("cameraAutoExposureEnabled") == controls.cameraAutoExposureEnabled && settings.at("inputCompressionEnabled") == controls.inputCompressionEnabled && settings.at("cameraExposureCompensationEv").at("f64_bits") == std::bit_cast<std::uint64_t>(controls.cameraExposureCompensationEv), "frozen CAT02 selected controls");
            const auto product = Cat02Fixtures::build(controls);
            const auto actual = Cat02Fixtures::complete_result(product);
            require(actual == row.at("expected"), id + " independent complete film/raw-midgray/scanner/tables/descriptors/identities");
            for (const auto& other : otherPresets) {
                const auto& wrong = other.at("recipes").at(i);
                require(wrong.at("id") == row.at("id") && wrong.at("controls") == row.at("controls"), "wrong-preset comparator uses identical inputs");
                if (wrong.at("expected") != row.at("expected")) {
                    require(actual != wrong.at("expected"), id + " wrong-preset value/identity comparator rejection");
                    wrongPresetRejected = true;
                }
            }
            auto changedIdentity = actual;
            changedIdentity["hashes"][0] = changedIdentity["hashes"][0].get<std::uint64_t>() ^ 1u;
            require(changedIdentity != row.at("expected"), id + " complete identity comparator rejection");
        }
        require(wrongPresetRejected, "different native-preset value/identity baselines reject incorrect expectation");
        std::array<Scanner::ScannerIlluminant, 4> illuminants;
        for (int route = 0; route < 4; ++route) {
            const auto product = Cat02Fixtures::build(Cat02Fixtures::snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019));
            illuminants[route] = Cat02Fixtures::scanner_illuminant(product);
            const auto& row = fixture.at("scanner_illuminants").at(route);
            const auto& illuminant = illuminants[route];
            require(row.at("route") == route && row.at("viewing_illuminant") == product.recipe.scannerOutput.viewingIlluminant && row.at("white_xyz") == bits(illuminant.whiteXYZ) && row.at("white_xy") == bits(illuminant.whiteXY) && row.at("normalization") == bits(std::array{illuminant.normalization}) && row.at("hash") == illuminant.hash && row.at("samples") == bits(illuminant.curve.linear), "actual selected-route scanner illuminant exact fields/identity");
        }
        for (const auto& row : fixture.at("scanner_cases")) {
            const OutputEncoding::Params encoding{static_cast<OutputEncoding::ColorSpace>(row.at("space").get<int>()), row.at("encoding").get<int>() != 0, true};
            const auto color = Scanner::build_color_runtime(static_cast<Scanner::ScannerMedium>(row.at("medium").get<int>()), illuminants[row.at("route").get<int>()], encoding, row.at("gamut_recipe_hash").get<std::uint64_t>());
            require(Cat02Fixtures::color_result(color) == row.at("expected"), row.at("id").get<std::string>() + " real scanner color exact arrays/encoding/hash");
        }
        for (const auto& row : fixture.at("scanner_invalid_cases")) {
            auto invalid = illuminants[0];
            invalid.hash = 0;
            FaultScope cleanup;
            expect_status(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
            const auto color = Scanner::build_color_runtime(static_cast<Scanner::ScannerMedium>(row.at("medium").get<int>()), invalid, {}, 0);
            require(Cat02Fixtures::color_result(color) == row.at("expected"), "invalid scanner illuminant keeps early zero-hash color result");
            expect_status(raw_call(FJ_TEST_CAT02_MATRIX), FJ_STATUS_INTERNAL_FAILURE);
        }
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
        Spectral::FilmRawConfig config(const Json& row) {
            Spectral::FilmRawConfig out;
            out.inputColorSpace = static_cast<Spectral::InputColorSpace>(row.at("space").get<int>());
            out.applyCctfDecoding = row.at("decode").get<int>() != 0;
            out.applyInputChromaticAdapt = row.at("adapt").get<bool>();
            out.inputRGBToXYZ = matrix(row.at("rgb_to_xyz"));
            out.inputXYZAdapt = matrix(row.at("xyz_adapt"));
            out.xyzToLinearSrgb = matrix(row.at("xyz_to_linear_srgb"));
            return out;
        }
        FjInputColorConversion foreign(const Spectral::FilmRawConfig& config) {
            FjInputColorConversion out{};
            out.input_space = static_cast<std::uint32_t>(config.inputColorSpace);
            out.decode_cctf = config.applyCctfDecoding ? 1u : 0u;
            out.adapt_xyz = config.applyInputChromaticAdapt ? 1u : 0u;
            std::copy_n(config.inputRGBToXYZ.m, 9, out.rgb_to_xyz);
            std::copy_n(config.inputXYZAdapt.m, 9, out.xyz_adapt);
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
        template <typename State>
        Json complete_result(const State& product) {
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
            result["tc_lut_size"] = product.payload.filmTcLut ? product.payload.filmTcLut->rgba.size() : 0;
            if (product.payload.filmTcLut) {
                const auto& data = product.payload.filmTcLut->rgba;
                for (std::size_t index : std::array<std::size_t, 5>{0, 768, 73728, 100000, 147452}) {
                    result["tc_samples"].push_back({{"index", index}, {"rgba", bits(std::array{data.at(index), data.at(index + 1), data.at(index + 2), data.at(index + 3)})}});
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
                const auto actual = JuicerColor::input_to_dwg(config, rgb, clamp);
                std::array<float, 3> facadeRgb{}, facadeXyz{};
                expect_status(fj_test_input_to_dwg(&input, rgb.data(), clamp ? 1u : 0u, facadeRgb.data(), facadeXyz.data()), FJ_STATUS_SUCCESS);
                const std::string prefix = clamp ? "dwg_clamped" : "dwg_signed";
                require(same_values(bits(actual.rgb), expected.at(prefix + "_rgb")) && same_values(bits(actual.xyz), expected.at(prefix + "_xyz")), row.at("id").get<std::string>() + " production DWG conversion/XYZ observation");
                require(same_values(bits(facadeRgb), expected.at(prefix + "_rgb")) && same_values(bits(facadeXyz), expected.at(prefix + "_xyz")), row.at("id").get<std::string>() + " facade DWG conversion/XYZ observation");
            }
            const auto actual = JuicerColor::input_to_linear_srgb(config, rgb);
            std::array<float, 3> facadeRgb{}, facadeXyz{};
            expect_status(fj_test_input_to_linear_srgb(&input, rgb.data(), config.xyzToLinearSrgb.m, facadeRgb.data(), facadeXyz.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(actual.rgb), expected.at("srgb_rgb")) && same_values(bits(actual.xyz), expected.at("srgb_xyz")), row.at("id").get<std::string>() + " production linear sRGB/XYZ observation");
            require(same_values(bits(facadeRgb), expected.at("srgb_rgb")) && same_values(bits(facadeXyz), expected.at("srgb_xyz")), row.at("id").get<std::string>() + " facade explicit inverse linear sRGB/XYZ");
        }
        for (const auto& row : fixture.at("linear")) {
            const auto rgb = triplet(row.at("rgb"));
            std::array<float, 3> srgb{}, dwg{};
            expect_status(fj_test_linear_srgb_to_xyz(rgb.data(), srgb.data()), FJ_STATUS_SUCCESS);
            expect_status(fj_test_dwg_to_xyz(rgb.data(), dwg.data()), FJ_STATUS_SUCCESS);
            require(same_values(bits(srgb), row.at("expected").at("srgb_xyz")) && same_values(bits(JuicerColor::linear_srgb_to_xyz(rgb)), row.at("expected").at("srgb_xyz")), row.at("id").get<std::string>() + " unsanitized linear sRGB leaf");
            require(same_values(bits(dwg), row.at("expected").at("dwg_xyz")) && same_values(bits(JuicerColor::dwg_to_xyz(rgb)), row.at("expected").at("dwg_xyz")), row.at("id").get<std::string>() + " unsanitized DWG leaf");
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
            expect_status(fj_test_color_arm_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
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
                expect_status(fj_test_color_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
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
                                                ? fj_legacy_input_to_dwg(&input, rgb.data(), 0, outRgb.data(), outXyz.data())
                                                : fj_legacy_input_to_linear_srgb(&input, rgb.data(), identity.data(), outRgb.data(), outXyz.data());
                        expect_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
                    }
                    if (operation == FJ_TEST_INPUT_TO_DWG) {
                        input.input_space = 0;
                        input.decode_cctf = 0;
                        input.adapt_xyz = 0;
                        expect_status(fj_legacy_input_to_dwg(&input, rgb.data(), 2, outRgb.data(), outXyz.data()), FJ_STATUS_UNSUPPORTED_INPUT);
                    }
                }
                expect_status(raw_call(operation), fault == FJ_TEST_COLOR_UNSUPPORTED_INPUT ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
            }
        }
        require(fixture.at("input_d65") != fixture.at("process_dwg_white"), "input D65 and process DWG white remain distinct");
        require(Spectral::inputColorSpaceFromIndex(-1) == Spectral::InputColorSpace::DaVinciWideGamut && Spectral::inputColorSpaceFromIndex(4) == Spectral::InputColorSpace::DaVinciWideGamut, "native integer transport fallback retained");
        std::puts("Input color: exact matrices/A, decode, explicit A/B conversions, linear leaves/projection and C11/ten-operation fault/default/facade controls passed");
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
                    const std::array<std::uint32_t, 6> callCounts{method == 1 ? 1u : 3u, method == 1 ? 1u : 0u, method == 1 ? 1u : 0u, method == 1 ? 0u : 2u, 0u, method == 1 ? 0u : 1u};
                    for (std::uint32_t operation = 5; operation <= 10; ++operation) {
                        const auto calls = callCounts[operation - 5];
                        {
                            FaultScope cleanup;
                            expect_status(fj_test_color_arm_fault(operation, calls + 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                            (void)Cat02Fixtures::build(controls);
                            expect_status(raw_call(operation), FJ_STATUS_INTERNAL_FAILURE);
                            ++counts;
                        }
                        for (std::uint32_t index = 1; index <= calls; ++index) {
                            for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
                                const FaultCase selected{operation, index, fault};
                                FaultScope cleanup;
                                InstanceState cold;
                                pending(cold, controls);
                                expect_status(fj_test_color_arm_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                const auto coldFailure = admit_pending_render_state(cold);
                                failed(coldFailure, selected, controls.filmProfileKey);
                                require(!JuicerAtomic::load_shared_ptr(&cold.activeDirectState) && !JuicerAtomic::load_shared_ptr(&cold.activePrintState) && cold.lastHash.load() == 0 && cold.buildCounterNext.load() == 0, "cold failed construction publishes no owner/hash/counter");
                                for (std::uint32_t call = 0; call < index; ++call) {
                                    expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
                                }
                                InstanceState state;
                                pending(state, controls);
                                const auto old = admit_pending_render_state(state);
                                require(admitted(old, route % 2 == 1), "initial input-color route admitted");
                                const auto oldValues = retain_values(old);
                                const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                                pending(state, changed);
                                expect_status(fj_test_color_arm_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                const auto result = admit_pending_render_state(state);
                                failed(result, selected, controls.filmProfileKey);
                                unchanged(state, old, hash, counter);
                                require_retained_values(old, oldValues);
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
                                require_retained_values(old, oldValues);
                                const auto recoveredValues = retain_values(recovered);
                                ++failures;
                                for (int mode = 0; mode < 3; ++mode) {
                                    Supersession selection{changed, mode, false};
                                    const auto liveHash = state.lastHash.load(), liveCounter = state.buildCounterNext.load();
                                    pending(state, controls);
                                    set_pending_capture_test_hook(supersede, &selection);
                                    expect_status(fj_test_color_arm_fault(operation, index, fault), FJ_STATUS_SUCCESS);
                                    const auto newer = admit_pending_render_state(state);
                                    require(selection.fired, "captured input-color snapshot superseded");
                                    if (mode == 0) {
                                        require(admitted(newer, route % 2 == 1) && newer.directState == recovered.directState && newer.printState == recovered.printState, "superseded valid input reuses successful publication");
                                    } else {
                                        require(newer.status == (mode == 1 ? PendingRenderAdmissionStatus::InvalidSnapshotControls : PendingRenderAdmissionStatus::NeedsSnapshotAcquisition) && !newer.directState && !newer.printState, "superseded invalid/uninitialized acquisition");
                                    }
                                    unchanged(state, recovered, liveHash, liveCounter);
                                    require_retained_values(old, oldValues);
                                    require_retained_values(recovered, recoveredValues);
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
        std::printf("Input-color admission: %zu exact operation counts; %zu early/late failure/recovery and %zu N2 supersession witnesses on four routes/two decoded spaces passed\n", counts, failures, supersessions);
    }

    template <typename Function>
    void require_input_failure(std::uint32_t operation, std::uint32_t fault, Function&& function) {
        FaultScope cleanup;
        expect_status(fj_test_color_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
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
        const auto product = Cat02Fixtures::build(Cat02Fixtures::snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Mallett2019));
        const auto& tables = product.payload.exposureTables;
        const auto& inverse = product.payload.spdSInv;
        require(Cat02Fixtures::tables_result(tables) == fixture.at("helper_tables") && bits(inverse) == fixture.at("helper_s_inv"), "native-parent helper preparation exact");
        for (const auto& row : fixture.at("helpers")) {
            require(Cat02Fixtures::helper_result(row, tables, inverse, true) == row.at("hanatos_expected"), row.at("id").get<std::string>() + " retained Hanatos helper exact");
            require(Cat02Fixtures::helper_result(row, tables, inverse, false) == row.at("tables_expected"), row.at("id").get<std::string>() + " retained table helper exact");
        }
        const std::array<float, 3> rgb{2.0f, -.25f, .5f};
        for (std::uint32_t fault : {FJ_TEST_COLOR_UNSUPPORTED_INPUT, FJ_TEST_COLOR_PANIC}) {
            require_input_failure(FJ_TEST_DWG_TO_XYZ, fault, [&] {
                std::vector<float> spectrum;
                Spectral::reconstruct_Ee_from_DWG_RGB_hanatos(rgb.data(), spectrum, tables.refIllumWhiteXYZ);
            });
            require_input_failure(FJ_TEST_DWG_TO_XYZ, fault, [&] {
                std::vector<float> spectrum;
                Spectral::reconstruct_Ee_from_DWG_RGB_with_tables(rgb.data(), tables, inverse.data(), spectrum);
            });
            require_input_failure(FJ_TEST_DWG_TO_XYZ, fault, [&] {
                (void)JuicerColor::dwg_to_xyz(rgb);
            });
            require_input_failure(FJ_TEST_LINEAR_SRGB_TO_XYZ, fault, [&] {
                (void)JuicerColor::linear_srgb_to_xyz(rgb);
            });
            require_input_failure(FJ_TEST_PROJECT_LINEAR_RGB_TO_XYZ, fault, [&] {
                (void)JuicerColor::project_linear_rgb_to_xyz(rgb, product.payload.filmRawConfig.inputRGBToXYZ, product.payload.filmRawConfig.inputXYZAdapt);
            });
            require_input_failure(FJ_TEST_INPUT_MATRICES, fault, [&] {
                (void)JuicerColor::input_matrices(Spectral::InputColorSpace::ITU_R_BT2020);
            });
            require_input_failure(FJ_TEST_INPUT_TO_DWG, fault, [&] {
                (void)JuicerColor::input_to_dwg(product.payload.filmRawConfig, rgb, false);
            });
            require_input_failure(FJ_TEST_INPUT_TO_LINEAR_SRGB, fault, [&] {
                (void)JuicerColor::input_to_linear_srgb(product.payload.filmRawConfig, rgb);
            });
        }
        std::puts("Input-color helpers: independent spectra and both-category direct primitive/DWG helper failures passed");
    }

    void input_preparation_tests(const Json& fixture, const std::filesystem::path& fixturePath) {
        const auto cases = InputFixtures::scenarios();
        require(cases.size() == 140 && cases.size() == fixture.at("recipes").size(), "96 base plus 44 bounded input-color products");
        require(InputFixtures::controls_json(ParamSnapshot{}) == fixture.at("defaults"), "all frozen parent defaults exact");
        std::vector<Json> otherPresets;
        for (const auto* preset : {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"}) {
            const auto path = fixturePath.parent_path() / (std::string("input-color-") + preset + ".json");
            if (path != fixturePath) {
                otherPresets.push_back(load_exposure_qualified_fixture(path));
            }
        }
        bool wrongPreset = false;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [id, controls] = cases[i];
            const auto& row = fixture.at("recipes").at(i);
            require(id == row.at("id").get<std::string>() && InputFixtures::controls_json(controls) == row.at("controls"), "ordered frozen product inputs/settings exact");
            const auto product = Cat02Fixtures::build(controls);
            const auto actual = InputFixtures::complete_result(product);
            if (actual != row.at("expected")) {
                throw std::runtime_error(id + " completed product mismatch: " + Json::diff(row.at("expected"), actual).dump());
            }
            for (const auto& other : otherPresets) {
                const auto& expected = other.at("recipes").at(i);
                require(expected.at("id").get<std::string>() == id && expected.at("controls") == row.at("controls"), "wrong-preset comparison keeps exact authored inputs");
                if (expected.at("expected") != row.at("expected")) {
                    require(actual != expected.at("expected"), "wrong-preset expected values/identities rejected");
                    wrongPreset = true;
                }
            }
            auto altered = actual;
            altered["hashes"][0] = altered["hashes"][0].get<std::uint64_t>() ^ 1u;
            require(altered != row.at("expected"), "required completed identity negative control");
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
                const auto fullHeld = old.printState ? InputFixtures::complete_result(*old.printState) : InputFixtures::complete_result(*old.directState);
                const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                const auto repeated = admit_pending_render_state(state);
                require(repeated.directState == old.directState && repeated.printState == old.printState, "unchanged input uses exact owners");
                unchanged(state, old, hash, counter);
                auto decoding = controls;
                decoding.inputCctfDecoding = 0;
                const auto undecoded = Cat02Fixtures::build(decoding);
                const auto& oldRaw = old.printState ? old.printState->recipe.filmRaw : old.directState->recipe.filmRaw;
                require(oldRaw.hash != undecoded.recipe.filmRaw.hash, "input decoding remains a contributing film identity control");
                if (method != 1) {
                    require((old.printState ? bits(old.printState->payload.filmRawConfig.rawMidgray) : bits(old.directState->payload.filmRawConfig.rawMidgray)) == bits(undecoded.payload.filmRawConfig.rawMidgray), "TC mid-gray remains already linear without CCTF decoding");
                }
                expect_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
                auto changed = controls;
                changed.inputColorSpace = 3;
                pending(state, changed);
                const auto replacement = admit_pending_render_state(state);
                require(admitted(replacement, route % 2 == 1) && (replacement.directState != old.directState || replacement.printState != old.printState), "input choice replaces complete publication");
                require(state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "replacement publishes complete input identity/counter");
                require_retained_values(old, held);
                require((old.printState ? InputFixtures::complete_result(*old.printState) : InputFixtures::complete_result(*old.directState)) == fullHeld, "full input-color config/scales/matrices retained after cache release/replacement");
                auto output = changed;
                output.outputCctfEncoding = 0;
                const auto encodedOff = Cat02Fixtures::build(output);
                const auto& liveRecipe = replacement.printState ? replacement.printState->recipe : replacement.directState->recipe;
                const auto& livePayload = replacement.printState ? replacement.printState->payload : replacement.directState->payload;
                require(liveRecipe.filmRaw.hash == encodedOff.recipe.filmRaw.hash && liveRecipe.filmRaw.tcLutHash == encodedOff.recipe.filmRaw.tcLutHash && livePayload.scannerHash == encodedOff.payload.scannerHash && Cat02Fixtures::tables_result(livePayload.scannerTables) == Cat02Fixtures::tables_result(encodedOff.payload.scannerTables), "output encoding preserves film/TC and scanner spectral families");
            }
        }
        input_helper_tests(fixture);
        input_admission_tests();
        std::puts("Input-color preparation: 140 exact products/identities, controls, retained owners/cache replacement and terminal admission gates passed");
    }

    void gpu_reuse_tests() {
        require(cudaSetDevice(0) == cudaSuccess && cudaFree(nullptr) == cudaSuccess, "GPU initialization");
        void* context = nullptr;
        std::string contextError;
        require(JuicerCuda::query_current_cuda_context(context, contextError), contextError);
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, context};
        std::uint64_t sequence = 1;
        for (int scenario = 0; scenario < 4; ++scenario) {
            const int method = scenario == 0 ? 0 : 1;
            for (int route = 0; route < 2; ++route) {
                auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                if (method == 1) {
                    controls.printProfileKey = "kodak_portra_endura";
                }
                if (scenario >= 2) {
                    controls.inputColorSpace = scenario == 2 ? 1 : 3;
                    controls.inputCctfDecoding = 1;
                }
                controls.grainControls.active = false;
                controls.dirCouplers.active = false;
                const auto product = build(controls);
                controls.outputCctfEncoding = 0;
                const auto encodedOff = build(controls);
                const auto descriptor = scanner_descriptor(product);
                const auto offDescriptor = scanner_descriptor(encodedOff);
                require(descriptor.hash == offDescriptor.hash, "CCTF-only scanner identity");
                std::array<const float*, 6> previous{};
                std::vector<std::uint32_t> previousContent;
                std::uint64_t previousHash = 0;
                std::array<long long, 3> preparationUs{};
                std::size_t transition = 0;
                for (const auto* prepared : {&product, &encodedOff, &product}) {
                    JuicerProcess::Root::CudaFramePreparationRequest request;
                    request.recipe = &prepared->recipe;
                    request.exposureTables = &prepared->payload.exposureTables;
                    request.filmRawConfig = &prepared->payload.filmRawConfig;
                    request.filmTcLut = prepared->payload.filmTcLut ? &*prepared->payload.filmTcLut : nullptr;
                    request.printMainIlluminant = prepared->payload.printMainIlluminant ? &*prepared->payload.printMainIlluminant : nullptr;
                    request.scannerTables = &prepared->payload.scannerTables;
                    request.scannerColor = &prepared->payload.scannerColor;
                    request.scannerLutDescriptor = &descriptor;
                    request.outputBoundaryTable = prepared->payload.outputBoundaryTable.get();
                    request.requestedWidth = 16;
                    request.requestedHeight = 16;
                    JuicerCuda::ResourceManager::SubmissionSnapshot submission;
                    submission.instanceToken.value = 0x4341543136ull;
                    submission.frameToken.value = sequence;
                    submission.snapshotId = sequence++;
                    submission.deviceContextKey = key;
                    submission.keyDigests = JuicerCuda::ResourceManager::make_key_digests(prepared->payload.uploadCoreHash, prepared->recipe.dirCouplers.hash, prepared->payload.scannerHash, 0);
                    JuicerCuda::Failure error;
                    const auto begin = std::chrono::steady_clock::now();
                    const auto operation = static_cast<std::uint32_t>(5 + (sequence % 6));
                    expect_status(fj_test_color_arm_fault(operation, 1, FJ_TEST_COLOR_PANIC), FJ_STATUS_SUCCESS);
                    auto frame = JuicerProcess::root().prepare_cuda_frame(key, submission, request, {}, nullptr, error);
                    preparationUs[transition++] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
                    require(frame.active(), error.diagnostic);
                    const auto focused = frame.focused_resources();
                    require(bits(focused.film.inputRGBToXYZ) == bits(prepared->payload.filmRawConfig.inputRGBToXYZ.m) &&
                                bits(focused.film.inputXYZAdapt) == bits(prepared->payload.filmRawConfig.inputXYZAdapt.m) &&
                                bits(focused.film.xyzToLinearSrgb) == bits(prepared->payload.filmRawConfig.xyzToLinearSrgb.m),
                            "prepared native film projection binds exact completed matrices");
                    const auto* lut = focused.scanLut;
                    require(lut && lut->canonical_ready(), "actual owning scanner LUT prepared");
                    const std::array<const float*, 6> addresses{lut->log2PchipXYZ, lut->slopeC, lut->slopeM, lut->slopeY, lut->cellMin, lut->cellMax};
                    const std::size_t voxels = static_cast<std::size_t>(lut->res) * lut->res * lut->res * 3;
                    const std::size_t cells = static_cast<std::size_t>(lut->res - 1) * (lut->res - 1) * (lut->res - 1) * 3;
                    std::vector<std::uint32_t> content;
                    for (std::size_t plane = 0; plane < addresses.size(); ++plane) {
                        std::vector<float> values(plane < 4 ? voxels : cells);
                        require(cudaMemcpy(values.data(), addresses[plane], values.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "owning scanner LUT content read");
                        for (float value : values) {
                            content.push_back(std::bit_cast<std::uint32_t>(value));
                        }
                    }
                    if (previous[0]) {
                        require(addresses == previous && content == previousContent && lut->hash == previousHash, "CCTF-only and warm reuse actual six device allocations/content/identity");
                    } else {
                        previous = addresses;
                        previousContent = std::move(content);
                        previousHash = lut->hash;
                    }
                    int* flag = nullptr;
                    require(frame.prepare_scan_error_stage(flag, nullptr, error) == JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready, error.diagnostic);
                    require(frame.finalize_scan_error_stage(flag, nullptr, error), error.diagnostic);
                    require(frame.finish(nullptr, error), error.diagnostic);
                    expect_status(raw_call(operation), FJ_STATUS_INTERNAL_FAILURE);
                }
                std::printf("GPU scanner route=%d method=%d input=%d decode=%d preparation_us cold=%lld cctf=%lld warm=%lld identity=%llu exact_values=%zu allocations=[%p,%p,%p,%p,%p,%p]\n",
                            route,
                            method,
                            controls.inputColorSpace,
                            controls.inputCctfDecoding,
                            preparationUs[0],
                            preparationUs[1],
                            preparationUs[2],
                            static_cast<unsigned long long>(previousHash),
                            previousContent.size(),
                            static_cast<const void*>(previous[0]),
                            static_cast<const void*>(previous[1]),
                            static_cast<const void*>(previous[2]),
                            static_cast<const void*>(previous[3]),
                            static_cast<const void*>(previous[4]),
                            static_cast<const void*>(previous[5]));
                std::string diagnostic;
                require(JuicerProcess::root().retire_idle_context(0, context, diagnostic), diagnostic);
            }
        }
        std::puts("GPU reuse: Hanatos/Mallett and decoded BT.2020/sRGB direct/print six-allocation/content/identity reuse; prepared binding leaves math faults unconsumed");
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
        owner.create(argv[1]);
        JuicerProcess::root().ensure_bootstrap();
        const auto fixture = load_exposure_qualified_fixture(argv[2]);
        const std::string group = argv[3];
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
