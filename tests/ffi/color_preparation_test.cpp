#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
#include "juicer_cuda_owner.h"

static_assert(sizeof(float) == 4 && sizeof(std::array<float, 3>) == 12 && sizeof(std::array<float, 9>) == 36);
extern "C" int fj_test_color_abi_c();

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
    std::array<float, 3> triplet(const Json& values) {
        std::array<float, 3> out{};
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = std::bit_cast<float>(values.at(i).get<std::uint32_t>());
        }
        return out;
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
        require(status.category == category && status.api == FJ_API_NONE && status.native_code == 0, "CAT16 status category/API/code");
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
        if (operation == FJ_TEST_CAT16_MATRIX) {
            std::array<float, 9> out{};
            return fj_legacy_cat16_matrix(white.data(), white.data(), out.data());
        }
        std::array<float, 3> out{};
        return fj_legacy_adapt_cat16(white.data(), white.data(), white.data(), out.data());
    }
    thread_local unsigned constructionFault = 0;
    struct FaultScope {
        FaultScope() = default;
        FaultScope(const FaultScope&) = delete;
        FaultScope& operator=(const FaultScope&) = delete;
        ~FaultScope() {
            const auto status = fj_test_cat16_clear_fault();
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
            Spectral::mul_3x3_vec3(matrix.data(), xyz.data(), multiplied.data());
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
            expect_status(fj_test_cat16_arm_fault(operation, 2, FJ_TEST_CAT16_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation == 1 ? 2 : 1), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            for (const auto& invalid : std::array<std::array<std::uint32_t, 3>, 3>{{{0, 1, 1}, {operation, 0, 1}, {operation, 1, 99}}}) {
                expect_status(fj_test_cat16_arm_fault(operation, 1, FJ_TEST_CAT16_PANIC), FJ_STATUS_SUCCESS);
                expect_status(fj_test_cat16_arm_fault(invalid[0], invalid[1], invalid[2]), FJ_STATUS_UNSUPPORTED_INPUT);
                expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            }
            expect_status(fj_test_cat16_arm_fault(operation, 1, FJ_TEST_CAT16_PANIC), FJ_STATUS_SUCCESS);
            expect_status(fj_test_cat16_arm_fault(operation, 2, FJ_TEST_CAT16_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_SUCCESS);
            expect_status(raw_call(operation), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_status(fj_test_cat16_clear_fault(), FJ_STATUS_SUCCESS);
            expect_status(fj_test_cat16_clear_fault(), FJ_STATUS_SUCCESS);
        }
        expect_status(fj_test_cat16_arm_fault(FJ_TEST_CAT16_MATRIX, 1, FJ_TEST_CAT16_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
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
    void failed(const PendingRenderAdmissionResult& result, const FaultCase& selected) {
        require(result.status == PendingRenderAdmissionStatus::RebuildFailed && !result.directState && !result.printState, "current exceptional attempt fails with empty result owners");
        require(result.diagnostic.find(selected.fault == 1 ? "UnsupportedInput" : "InternalFailure") != std::string::npos, "original category diagnostic");
        require(result.diagnostic.find(selected.operation == 1 ? "CAT16 matrix" : "CAT16 scalar") != std::string::npos, "original operation diagnostic");
        require(result.diagnostic.find("film=kodak_portra_400") != std::string::npos && result.diagnostic.find("route=") != std::string::npos, "attempted selection diagnostic");
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
                        for (std::uint32_t fault : {FJ_TEST_CAT16_UNSUPPORTED_INPUT, FJ_TEST_CAT16_PANIC}) {
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
                            expect_status(fj_test_cat16_arm_fault(selected.operation, selected.callIndex, selected.fault), FJ_STATUS_SUCCESS);
                            const auto result = admit_pending_render_state(state);
                            failed(result, selected);
                            unchanged(state, old, hash, counter);
                            pending(state, controls);
                            const auto reused = admit_pending_render_state(state);
                            require(reused.directState == old.directState && reused.printState == old.printState, "revert to matching retained publication reuses owner");
                            unchanged(state, old, hash, counter);
                            expect_status(fj_test_cat16_clear_fault(), FJ_STATUS_SUCCESS);
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
                                expect_status(fj_test_cat16_arm_fault(selected.operation, selected.callIndex, selected.fault), FJ_STATUS_SUCCESS);
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
                                expect_status(fj_test_cat16_clear_fault(), FJ_STATUS_SUCCESS);
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
            expect_status(fj_test_cat16_arm_fault(FJ_TEST_CAT16_MATRIX, 1, FJ_TEST_CAT16_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
            const auto repeated = admit_pending_render_state(state);
            require(repeated.directState == old.directState && repeated.printState == old.printState && state.buildCounterNext.load() == counter, "unchanged admission reuses actual state");
            expect_status(raw_call(FJ_TEST_CAT16_MATRIX), FJ_STATUS_UNSUPPORTED_INPUT);
        }
        admission_tests();
        construction_controls();
        std::puts("Preparation: native-parent complete identities, tables, contributing controls and retained-owner reuse passed");
    }

    void gpu_reuse_tests() {
        require(cudaSetDevice(0) == cudaSuccess && cudaFree(nullptr) == cudaSuccess, "GPU initialization");
        void* context = nullptr;
        std::string contextError;
        require(JuicerCuda::query_current_cuda_context(context, contextError), contextError);
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, context};
        std::uint64_t sequence = 1;
        for (int route = 0; route < 2; ++route) {
            auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Hanatos2025);
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
                auto frame = JuicerProcess::root().prepare_cuda_frame(key, submission, request, {}, nullptr, error);
                require(frame.active(), error.diagnostic);
                const auto* lut = frame.focused_resources().scanLut;
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
            }
        }
        std::string diagnostic;
        require(JuicerProcess::root().retire_idle_context(0, context, diagnostic), diagnostic);
        std::puts("GPU reuse: direct/print CCTF-only and warm scanner allocation/content/identity reuse passed");
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
        std::ifstream file(argv[2]);
        const auto fixture = Json::parse(file);
        const std::string group = argv[3];
        if (group == "leaf") {
            leaf_tests(fixture);
        } else if (group == "preparation") {
            preparation_tests(fixture);
        } else if (group == "reuse-gpu") {
            gpu_reuse_tests();
        } else {
            throw std::runtime_error("unknown color test group");
        }
        expect_status(owner.close(), FJ_STATUS_SUCCESS);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CAT16 tests: %s\n", error.what());
        return 1;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        std::fprintf(stderr, "CAT16 construction status=%u: %s\n", error.failure.status.category, error.failure.diagnostic.c_str());
        return 1;
    }
}
