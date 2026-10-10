#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "ColorTransforms.h"
#include "JuicerState.h"
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
    Json color_result(const Scanner::ColorRuntime& color) {
        return {{"cat02", bits(color.cat02)}, {"xyz_to_rgb", bits(color.xyzToRgb)}, {"illuminant_xyz", bits(color.illuminantXYZ)}, {"encoding", {static_cast<int>(color.encoding.colorSpace), color.encoding.applyCctfEncoding, color.encoding.inputIsOutputSpace}}, {"gamut_recipe_hash", color.outputGamutRecipeHash}, {"hash", color.hash}};
    }
    Json tables_result(const Spectral::SpectralTables& tables) {
        return {{"lambda", bits(tables.lambda)}, {"K", tables.K}, {"delta_lambda", bits(std::array{tables.deltaLambda})}, {"inv_yn", bits(std::array{tables.invYn})}, {"white_xyz", bits(tables.whiteXYZ)}, {"ref_illum_white_xyz", bits(tables.refIllumWhiteXYZ)}, {"ax", bits(tables.Ax)}, {"ay", bits(tables.Ay)}, {"az", bits(tables.Az)}, {"xbar", bits(tables.Xbar)}, {"ybar", bits(tables.Ybar)}, {"zbar", bits(tables.Zbar)}, {"illum", bits(tables.illum)}, {"eps_c", bits(tables.epsC)}, {"eps_m", bits(tables.epsM)}, {"eps_y", bits(tables.epsY)}, {"base_density_min", bits(tables.baseDensityMin)}, {"base_density_mid", bits(tables.baseDensityMid)}, {"has_baseline", tables.hasBaseline}, {"baseline_mix", bits(std::array{tables.densityBaselineMixReference})}, {"illuminant_hash", tables.illuminantHash}, {"tables_hash", tables.tablesHash}};
    }
    Json scanner_descriptor(const FocusedRenderStateBuildProduct& product) {
        Scanner::ScannerSpectralLutDescriptor out;
        std::string diagnostic;
        const auto& recipe = product.recipe;
        const bool ok = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic) : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic);
        require(ok, diagnostic);
        return {{"route", static_cast<int>(out.route)}, {"medium", static_cast<int>(out.medium)}, {"polarity", static_cast<int>(out.polarity)}, {"density_bounds_hash", out.densityBoundsHash}, {"channel_density_hash", out.channelDensityHash}, {"base_density_hash", out.baseDensityHash}, {"illuminant_hash", out.scanIlluminantHash}, {"observer_hash", out.observerHash}, {"resolution", out.lutResolution}, {"hash", out.hash}};
    }
    Json complete_result(const FocusedRenderStateBuildProduct& product) {
        const auto& recipe = product.recipe;
        const auto& raw = recipe.filmRaw;
        const auto& reference = recipe.scannerOutput.syntheticFilmReference;
        const auto& balance = recipe.print.balance;
        const auto& config = product.payload.filmRawConfig;
        return {{"input_rgb_to_xyz", bits(raw.inputRgbToXyz)}, {"input_adapt", bits(raw.inputXyzAdapt)}, {"xyz_to_linear_srgb", bits(raw.xyzToLinearSrgb)}, {"source_white", bits(raw.inputNominalWhiteXYZ)}, {"projection_white", bits(raw.projectionWhiteXYZ)}, {"config_rgb_to_xyz", bits(config.inputRGBToXYZ.m)}, {"config_adapt", bits(config.inputXYZAdapt.m)}, {"raw_midgray", bits(config.rawMidgray)}, {"midgray_dwg", bits(config.midgrayDWG)}, {"midgray_scale_green", bits(std::array{config.midgrayScale, config.rawMidgrayGreen})}, {"config_whites", {bits(config.inputWhiteXYZ), bits(config.workingWhiteXYZ), bits(config.refIllumWhiteXYZ)}}, {"config_flags", {static_cast<int>(config.inputColorSpace), config.applyCctfDecoding, static_cast<int>(config.spectralUpsamplingMode), config.applyInputChromaticAdapt, config.hasRefIllumWhite, config.valid}}, {"spd_s_inv", bits(product.payload.spdSInv)}, {"baseline_raw", bits(reference.baselineRawRgb)}, {"compensated_raw", bits(reference.compensatedRawRgb)}, {"baseline_density", bits(reference.baselineDensityCmy)}, {"compensated_density", bits(reference.compensatedDensityCmy)}, {"print_baseline_raw", bits(balance.baselinePrintRawRgb)}, {"print_compensated_raw", bits(balance.compensatedPrintRawRgb)}, {"print_factors", bits(std::array{balance.factorMidgray, balance.factorMidgrayComp, balance.normalizer})}, {"hashes", {raw.hash, raw.tcLutHash, reference.hash, balance.hash, recipe.print.hash, recipe.scannerOutput.hash, recipe.scannerOutput.outputGamut.hash, recipe.hash, product.payload.uploadCoreHash, product.payload.scannerHash}}, {"scanner_color", color_result(product.payload.scannerColor)}, {"scanner_tables", tables_result(product.payload.scannerTables)}, {"scanner_descriptor", scanner_descriptor(product)}, {"gamut_entry", bits(recipe.scannerOutput.outputGamut.transform.nativeRgbToD65Xyz)}, {"gamut_exit", bits(recipe.scannerOutput.outputGamut.transform.d65XyzToNativeRgb)}};
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
    bool build(const ParamSnapshot& p, FocusedRenderStateBuildProduct& out, std::string& diagnostic) {
        return Spektrafilm::scan_route_is_print(p.scanRoute) ? build_print_render_state_product(p, out, diagnostic) : build_direct_render_state_product(p, out, diagnostic);
    }
    void products(const Json& fixture) {
        for (std::size_t i = 0; i < fixture.at("products").size(); ++i) {
            const auto& row = fixture.at("products")[i];
            auto p = controls(row);
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            require(build(p, product, diagnostic), diagnostic);
            const auto observed = complete_result(product);
            require(observed == row.at("expected"), "actual complete product " + std::to_string(i) + ": " + Json::diff(row.at("expected"), observed).dump());
            const auto& meter = fixture.at("metrics")[i];
            require(std::bit_cast<std::uint32_t>(product.recipe.scannerOutput.syntheticFilmReference.baselineMeterScale) == meter.at("baseline_meter_scale").get<std::uint32_t>(), "meter exact");
            require(product.recipe.print.balance.filteredMainIlluminantHash == meter.at("filtered_main_illuminant_hash").get<std::uint64_t>(), "filtered print identity exact");
            Json sensitivity = Json::array();
            for (auto rgb : product.recipe.filmRaw.finalSensitivity)
                sensitivity.push_back(bits(rgb));
            require(sensitivity == row.at("final_sensitivity") && bits(product.payload.exposureTables.illum) == row.at("reference_spd"), "source/sensitivity remain exact");
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
        return print ? out.printState->recipe.hash : out.directState->recipe.hash;
    }
    struct BasisScope {
        Spectral::MallettBasis saved = Spectral::gMallettBasis;
        bool available = Spectral::mallett_available();
        ~BasisScope() {
            Spectral::gMallettBasis = std::move(saved);
            Spectral::gMallettAvailable.store(available);
            fj_test_exposure_clear_fault();
            fj_test_spectral_clear_fault();
            fj_test_color_clear_fault();
            set_pending_capture_test_hook(nullptr, nullptr);
        }
        void restore() {
            Spectral::gMallettBasis = saved;
            Spectral::gMallettAvailable.store(available);
        }
        void missing(int defect) {
            restore();
            if (defect == 0) {
                Spectral::gMallettBasis = {};
                Spectral::gMallettAvailable = false;
            }
            if (defect == 1)
                Spectral::gMallettBasis.rows = 80;
            if (defect == 2)
                Spectral::gMallettBasis.cols = 2;
            if (defect == 3)
                Spectral::gMallettBasis.data.resize(240);
            if (defect == 4)
                Spectral::gMallettAvailable = false;
        }
    };
    struct Supersession {
        ParamSnapshot next;
        int mode = 0;
        bool fired = false;
        BasisScope* basis = nullptr;
    };
    void supersede(InstanceState& state, void* opaque) {
        auto& value = *static_cast<Supersession*>(opaque);
        if (std::exchange(value.fired, true))
            return;
        value.basis->restore();
        std::lock_guard<std::mutex> lock(state.pending.m);
        if (value.mode == 1)
            state.pending.value = PendingParamsState::InvalidSnapshotControls{"new invalid controls"};
        else if (value.mode == 2)
            state.pending.value = PendingParamsState::Uninitialized{};
        else
            state.pending.value = PendingParamsState::Valid{value.next, hash_params(value.next)};
    }
    void admission(const Json& contract) {
        const auto missingDiagnostic = contract.at("diagnostic").get<std::string>();
        for (int polarity = 0; polarity < 2; ++polarity)
            for (int route = 0; route < 2; ++route) {
                const bool print = route == 1;
                auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), Spektrafilm::RgbToRawMethod::Mallett2019);
                auto changed = p;
                changed.cameraExposureCompensationEv = 1.5;
                for (int defect = 0; defect < 5; ++defect) {
                    BasisScope basis;
                    InstanceState state;
                    pending(state, p);
                    auto old = admit_pending_render_state(state);
                    require(admitted(old, print), "initial admission");
                    const auto oldRecipe = recipe_hash(old, print), hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                    basis.missing(defect);
                    pending(state, changed);
                    require(fj_test_exposure_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "mid-gray fault armed");
                    const auto failed = admit_pending_render_state(state);
                    require(failed.status == PendingRenderAdmissionStatus::RebuildFailed && !failed.directState && !failed.printState && failed.diagnostic == missingDiagnostic, "approved early ordinary-false diagnostic route=" + std::to_string(route) + " defect=" + std::to_string(defect) + " observed=" + failed.diagnostic);
                    require(fj_test_exposure_fault_consumed(3, 1) == 0, "missing source skips Rust binding");
                    require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "ordinary failure keeps hash/counter");
                    require(print ? !JuicerAtomic::load_shared_ptr(&state.activePrintState) : !JuicerAtomic::load_shared_ptr(&state.activeDirectState), "ordinary failure clears selected slot");
                    require(recipe_hash(old, print) == oldRecipe, "already admitted hold survives");
                    {
                        std::lock_guard<std::mutex> lock(state.pending.m);
                        const auto* current = std::get_if<PendingParamsState::Valid>(&state.pending.value);
                        require(current && current->fullHash == hash_params(changed), "failed changed-current snapshot remains valid pending");
                    }
                    basis.restore();
                    const auto exceptional = admit_pending_render_state(state);
                    require(exceptional.status == PendingRenderAdmissionStatus::RebuildFailed && fj_test_exposure_fault_consumed(3, 1) == 1, "valid recovery attempt consumes skipped fault before cleanup");
                    fj_test_exposure_clear_fault();
                    const auto recovered = admit_pending_render_state(state);
                    require(admitted(recovered, print) && state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "recovery complete publication");
                    const auto reused = admit_pending_render_state(state);
                    require(reused.directState == recovered.directState && reused.printState == recovered.printState && state.buildCounterNext.load() == counter + 1, "same-input owner reuse");
                    require(JuicerProcess::root().assets().release_cached_payloads().category == FJ_STATUS_SUCCESS, "release source/profile cache");
                    require(recipe_hash(old, print) == oldRecipe && recipe_hash(recovered, print) == recipe_hash(reused, print), "admitted holds survive cache release");
                }
                for (std::uint32_t operation : {3u, 5u, 6u})
                    for (std::uint32_t fault : {1u, 2u, 4u}) {
                        BasisScope basis;
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
                for (int mode = 0; mode < 3; ++mode) {
                    BasisScope basis;
                    InstanceState state;
                    pending(state, p);
                    require(admitted(admit_pending_render_state(state), print), "supersession initial");
                    Supersession selection{p, mode, false, &basis};
                    pending(state, changed);
                    set_pending_capture_test_hook(supersede, &selection);
                    require(fj_test_exposure_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "supersession fault arm");
                    const auto next = admit_pending_render_state(state);
                    require(selection.fired && fj_test_exposure_fault_consumed(3, 1) == 1, "obsolete attempt fault consumed before cleanup");
                    require(mode == 0 ? admitted(next, print) : next.status == (mode == 1 ? PendingRenderAdmissionStatus::InvalidSnapshotControls : PendingRenderAdmissionStatus::NeedsSnapshotAcquisition), "supersession resolves current snapshot");
                    set_pending_capture_test_hook(nullptr, nullptr);
                }
                for (int earlier = 0; earlier < 3; ++earlier) {
                    BasisScope basis;
                    basis.missing(0);
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
                        require(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 2, 1).category == FJ_STATUS_SUCCESS, "later scanner fault");
                    const auto failed = admit_pending_render_state(state);
                    require(failed.status == PendingRenderAdmissionStatus::RebuildFailed && state.lastHash.load() == 0 && state.buildCounterNext.load() == 0, "precedence no publication");
                    if (earlier == 0)
                        require(failed.diagnostic == "ResourceDescriptorMismatch component=output_gamut_recipe field=output_color_space", "earlier recipe failure precedence");
                    if (earlier == 1)
                        require(failed.diagnostic.find("Film S inverse preparation") != std::string::npos, "earlier S-inverse precedence");
                    if (earlier == 2) {
                        require(failed.diagnostic == missingDiagnostic, "missing preempts later scanner");
                        const float white[3]{.95047f, 1, 1.08883f};
                        float matrix[9];
                        require(fj_legacy_cat02_matrix(white, white, matrix).category == FJ_STATUS_UNSUPPORTED_INPUT, "late scanner fault remained armed, observed before cleanup");
                    }
                }
            }
        for (int method : {0, 2}) {
            BasisScope basis;
            InstanceState state;
            auto p = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, static_cast<Spektrafilm::RgbToRawMethod>(method));
            pending(state, p);
            require(fj_test_exposure_arm_fault(4, 1, 4).category == FJ_STATUS_SUCCESS, "TC normalization fault arm");
            require(admit_pending_render_state(state).status == PendingRenderAdmissionStatus::RebuildFailed && fj_test_exposure_fault_consumed(4, 4) == 1, "TC actual consumer");
            fj_test_exposure_clear_fault();
            require(admitted(admit_pending_render_state(state), false), "TC normalization recovery");
        }
    }
} // namespace
void fj_test_mallett_callers(const std::filesystem::path& resources, const Json& fixture, const std::filesystem::path& contractPath, bool admissionOnly) {
    JuicerCuda::Owner owner;
    owner.create(resources);
    JuicerProcess::root().ensure_bootstrap();
    if (admissionOnly) {
        std::ifstream stream(contractPath);
        const auto contract = Json::parse(stream);
        admission(contract);
    } else
        products(fixture);
    require(owner.close().category == FJ_STATUS_SUCCESS, "caller owner close");
}
