#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "GamutCompression.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RustSpectralBridge.h"
#include "SpectralProcessing.h"
#include "juicer_test_api.h"
#include "juicer_cuda_owner.h"

void fj_test_spectral_native_allocation(const nlohmann::json&);

namespace {
    namespace SpectralFixtures {
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
        template <typename Range>
        Json double_bits(const Range& values) {
            Json out = Json::array();
            for (double value : values)
                out.push_back(std::bit_cast<std::uint64_t>(value));
            return out;
        }
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
    } // namespace SpectralFixtures

    using namespace SpectralFixtures;
    namespace fs = std::filesystem;
    void pending(InstanceState& state, const ParamSnapshot& controls) {
        std::lock_guard lock(state.pending.m);
        state.pending.value = PendingParamsState::Valid{controls, hash_params(controls)};
    }
    bool admitted(const PendingRenderAdmissionResult& result) {
        return (result.status == PendingRenderAdmissionStatus::AdmittedDirect && bool(result.directState)) || (result.status == PendingRenderAdmissionStatus::AdmittedPrint && bool(result.printState));
    }
    std::optional<PendingRenderAdmissionResult> failed_attempt(InstanceState& state, std::uint32_t category, bool ordinary) {
        try {
            auto result = admit_pending_render_state(state);
            require(result.status == PendingRenderAdmissionStatus::RebuildFailed && !result.directState && !result.printState, "failed current attempt returns no admitted state");
            require(ordinary || category == FJ_STATUS_UNSUPPORTED_INPUT || category == FJ_STATUS_INTERNAL_FAILURE, "other fatal categories must propagate");
            return result;
        } catch (const JuicerCuda::ExecutionFailure& error) {
            require(!ordinary && (category == FJ_STATUS_ALLOCATION_FAILURE || category == FJ_STATUS_PREPARATION_FAILURE) && error.failure.status.category == category && error.failure.status.api == FJ_API_NONE && error.failure.status.native_code == 0, "other fatal categories propagate typed through admission");
            return std::nullopt;
        }
    }
    Json held(const PendingRenderAdmissionResult& result) {
        require(bool(result.directState) || bool(result.printState), "retained admitted state");
        Json out = result.printState ? complete_result(*result.printState) : complete_result(*result.directState);
        const auto& p = result.printState ? result.printState->payload : result.directState->payload;
        out["exposure_tables"] = tables_result(p.exposureTables);
        if (p.filmTcLut)
            out["tc_rgba"] = bits(p.filmTcLut->rgba);
        if (p.printMainIlluminant)
            out["print_light"] = bits(*p.printMainIlluminant);
        if (p.outputBoundaryTable)
            out["boundary"] = bits(p.outputBoundaryTable->cmax);
        return out;
    }
    void products(const Json& fixture, const fs::path& root) {
        auto& assets = JuicerProcess::root().assets();
        std::size_t count = 0;
        for (int polarity = 0; polarity < 2; ++polarity) {
            auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2), Spektrafilm::RgbToRawMethod::Mallett2019);
            const auto filmPath = root / "profiles" / (controls.filmProfileKey + ".json"), printPath = root / "profiles/kodak_portra_endura.json";
            const auto read = [](const fs::path& p) {
                std::ifstream f(p);
                return Json::parse(f);
            };
            const auto original = read(filmPath), originalPrint = read(printPath);
            for (const auto& row : fixture.at("products")) {
                if (row.at("polarity") != polarity)
                    continue;
                auto film = original, print = originalPrint;
                const auto& label = row.at("label");
                film["info"]["reference_illuminant"] = label;
                film["info"]["viewing_illuminant"] = label;
                print["info"]["viewing_illuminant"] = label;
                {
                    std::ofstream f(filmPath);
                    f << film;
                }
                {
                    std::ofstream f(printPath);
                    f << print;
                }
                require(assets.release_cached_payloads().category == FJ_STATUS_SUCCESS, "product source release");
                const int route = row.at("route"), method = row.at("method");
                auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                p.enlIll = 0;
                p.printPreflashExposure = route ? .1 : 0;
                p.preflashMFilterCc = 12.5;
                p.preflashYFilterCc = 7.25;
                require(controls_json(p) == row.at("controls"), "all serialized native controls preserved");
                FocusedRenderStateBuildProduct product;
                std::string diagnostic;
                const bool ok = route ? build_print_render_state_product(p, product, diagnostic) : build_direct_render_state_product(p, product, diagnostic);
                require(ok == row.at("built"), "native product completion/failure " + label.get<std::string>());
                if (ok) {
                    auto result = complete_result(product);
                    result["exposure_tables"] = tables_result(product.payload.exposureTables);
                    result["final_sensitivity_hash"] = product.recipe.filmRaw.finalSensitivityHash;
                    for (const auto& rgb : product.recipe.filmRaw.finalSensitivity)
                        result["final_sensitivity"].push_back(bits(rgb));
                    if (route) {
                        result["main_light"] = bits(*product.payload.printMainIlluminant);
                        JuicerCuda::PrintResourcePreparation request{&product.recipe, &assets, &*product.payload.printMainIlluminant};
                        JuicerCuda::PrintResourceInput input;
                        std::array<float, 81> preflash{};
                        require(JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic), "print resource input");
                        result["preflash_light"] = bits(preflash);
                        result["preflash_raw"] = bits(input.preflashRawCmy);
                        result["print_descriptors"] = {input.descriptors.hash, input.descriptors.mainIlluminant.hash, input.descriptors.preflashIlluminant.hash, input.descriptors.preflashRaw.hash, input.descriptors.balance.hash};
                    }
                    if (result != row.at("expected")) {
                        const auto diff = Json::diff(row.at("expected"), result);
                        throw std::runtime_error("completed product first difference " + std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(method) + "/" + label.get<std::string>() + ": " + diff[0].dump());
                    }
                } else {
                    const std::string expected = row.at("diagnostic");
                    require(!diagnostic.empty(), "native failed product diagnostic retained");
                    const bool sourceFailure = expected.starts_with("profile ");
                    std::string comparison = sourceFailure ? "source failure stage retained with scratch-root path: " : "computed failure diagnostic exact: ";
                    comparison += diagnostic;
                    comparison += " expected ";
                    comparison += expected;
                    require(sourceFailure ? diagnostic.ends_with("info.reference_illuminant requires Illuminant") : diagnostic == expected, comparison);
                }
                ++count;
            }
            {
                std::ofstream f(filmPath);
                f << original;
            }
            {
                std::ofstream f(printPath);
                f << originalPrint;
            }
            require(assets.release_cached_payloads().category == FJ_STATUS_SUCCESS, "final source release");
        }
        require(count == 72, "complete native capture product membership");
    }
    void admission() {
        auto& assets = JuicerProcess::root().assets();
        std::size_t witnesses = 0;
        for (int route = 0; route < 4; ++route) {
            const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019);
            auto changed = controls;
            changed.cameraExposureCompensationEv = 1.0;
            const auto raw = [&](std::uint32_t operation) {
                std::array<float, 81> ill{};
                ill.fill(1);
                const FjSpectralObserver observer{{Spectral::gXBar.linear.data(), 81}, {Spectral::gYBar.linear.data(), 81}, {Spectral::gZBar.linear.data(), 81}};
                std::array<std::array<float, 3>, 81> dyes{};
                FjSpectralInput table{{dyes.front().data(), 243}, observer, {ill.data(), 81}, {nullptr, 0}, {nullptr, 0}, 1};
                FjSpectralWhiteInput w{observer, {ill.data(), 81}};
                FjSpectralSInput s{observer.x, observer.y, observer.z};
                FjSpectralTables t{};
                FjSpectralWhite white{};
                FjSpectralWhiteFailure failure{};
                FjSpectralInverse inverse{};
                std::array<char, 256> bytes{};
                FjErrorBuffer error{bytes.data(), bytes.size(), 0};
                if (operation == 1)
                    return fj_legacy_spectral_tables(&table, &t, &error);
                if (operation == 2)
                    return fj_legacy_spectral_white(&w, &white, &failure, &error);
                return fj_legacy_spectral_s_inverse(&s, &inverse, &error);
            };
            for (std::uint32_t op = 1; op <= 3; ++op)
                for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                    std::printf("Spectral admission route=%d op=%u fault=%u\n", route, op, fault);
                    std::fflush(stdout);
                    const std::uint32_t category[] = {0, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE};
                    const std::uint32_t index = (op == 2 && fault == 4) ? 2 : 1;
                    require(fj_test_spectral_arm_fault(op, index, fault).category == FJ_STATUS_SUCCESS, "arm direct builder fault");
                    FocusedRenderStateBuildProduct directProduct;
                    std::string directDiagnostic;
                    bool thrown = false;
                    bool built = false;
                    try {
                        built = route % 2 ? build_print_render_state_product(controls, directProduct, directDiagnostic) : build_direct_render_state_product(controls, directProduct, directDiagnostic);
                    } catch (const JuicerCuda::ExecutionFailure& error) {
                        thrown = true;
                        require(error.failure.status.category == category[fault] && error.failure.status.api == FJ_API_NONE && error.failure.status.native_code == 0, "actual builder typed spectral failure");
                    }
                    require(!built && thrown != (op == 2 && fault == 4), "ordinary false versus exception builder distinction");
                    require(raw(op).category == FJ_STATUS_SUCCESS, "builder raw fault consumed");
                    InstanceState cold;
                    pending(cold, controls);
                    require(fj_test_spectral_arm_fault(op, index, fault).category == FJ_STATUS_SUCCESS, "arm cold spectral fault");
                    const auto coldFailure = failed_attempt(cold, category[fault], op == 2 && fault == 4);
                    require((!coldFailure || !admitted(*coldFailure)) && !JuicerAtomic::load_shared_ptr(&cold.activeDirectState) && !JuicerAtomic::load_shared_ptr(&cold.activePrintState) && cold.lastHash.load() == 0, "cold failure admits/publishes nothing");
                    require(raw(op).category == FJ_STATUS_SUCCESS, "cold failure consumes actual same-thread operation");
                    InstanceState state;
                    pending(state, controls);
                    const auto old = admit_pending_render_state(state);
                    require(admitted(old), "initial spectral publication");
                    const auto oldValues = held(old);
                    const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                    pending(state, changed);
                    require(fj_test_spectral_arm_fault(op, index, fault).category == FJ_STATUS_SUCCESS, "arm current spectral fault");
                    const auto failed = failed_attempt(state, category[fault], op == 2 && fault == 4);
                    require(!failed || !admitted(*failed), "changed-current failure cannot admit older state");
                    const bool ordinary = op == 2 && fault == 4;
                    if (!ordinary) {
                        const char* names[] = {"", "UnsupportedInput", "InternalFailure", "AllocationFailure", "PreparationFailure"};
                        require(!failed || failed->diagnostic.find(names[fault]) != std::string::npos, "fatal category retained in admission diagnostic");
                        require(JuicerAtomic::load_shared_ptr(&state.activeDirectState) == old.directState && JuicerAtomic::load_shared_ptr(&state.activePrintState) == old.printState, "exceptional preparation retains active publication");
                    } else {
                        require(!JuicerAtomic::load_shared_ptr(&state.activeDirectState) && !JuicerAtomic::load_shared_ptr(&state.activePrintState), "ordinary white false clears active publication");
                    }
                    require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "failed attempt publishes no new identity/counter");
                    require(raw(op).category == FJ_STATUS_SUCCESS, "same-thread raw export proves consumption before recovery");
                    require(held(old) == oldValues, "already admitted complete owners survive failure");
                    require(assets.release_cached_payloads().category == FJ_STATUS_SUCCESS, "cache release after fault");
                    require(held(old) == oldValues, "retained values survive cache release");
                    pending(state, changed);
                    const auto recovered = admit_pending_render_state(state);
                    require(admitted(recovered) && state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "successful recovery publishes changed-current complete state");
                    require(recovered.directState != old.directState || recovered.printState != old.printState, "recovery replaces publication");
                    require(held(old) == oldValues, "old admitted owner survives replacement");
                    ++witnesses;
                }
            for (int method = 0; method < 3; ++method)
                for (std::uint32_t op = 1; op <= 3; ++op) {
                    const std::uint32_t count = op == 1 ? 2 : op == 2 ? (method == 2 ? 4 : 3)
                                                                      : 1;
                    auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                    FocusedRenderStateBuildProduct product;
                    std::string diagnostic;
                    require(fj_test_spectral_arm_fault(op, count + 1, 1).category == FJ_STATUS_SUCCESS, "arm exact production operation count");
                    require(route % 2 ? build_print_render_state_product(p, product, diagnostic) : build_direct_render_state_product(p, product, diagnostic), "exact count build remains complete");
                    require(raw(op).category == FJ_STATUS_UNSUPPORTED_INPUT, "next same-thread raw call proves complete production call count");
                    require(raw(op).category == FJ_STATUS_SUCCESS, "count witness consumed");
                }
            {
                InstanceState current;
                pending(current, controls);
                const auto admittedBeforeGate = admit_pending_render_state(current);
                require(admitted(admittedBeforeGate), "before film gate admitted owner");
                const auto heldBeforeGate = held(admittedBeforeGate);
                struct ObserverRestore {
                    std::array<std::vector<float>, 3> saved{Spectral::gXBar.linear, Spectral::gYBar.linear, Spectral::gZBar.linear};
                    ~ObserverRestore() {
                        Spectral::gXBar.linear = std::move(saved[0]);
                        Spectral::gYBar.linear = std::move(saved[1]);
                        Spectral::gZBar.linear = std::move(saved[2]);
                    }
                } restore;
                for (auto* cmf : {&Spectral::gXBar.linear, &Spectral::gYBar.linear, &Spectral::gZBar.linear})
                    for (auto& value : *cmf)
                        value = std::ldexp(value, -136);
                require(fj_test_spectral_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "arm inverse suppression witness");
                FocusedRenderStateBuildProduct product;
                std::string diagnostic;
                const bool ok = route % 2 ? build_print_render_state_product(changed, product, diagnostic) : build_direct_render_state_product(changed, product, diagnostic);
                require(!ok && diagnostic == "MalformedRequiredResource component=focused_film_payload field=exposure_tables", "real f32 reciprocal overflow reaches film table gate");
                require(raw(3).category == FJ_STATUS_UNSUPPORTED_INPUT, "failed film white/hash gate never consumes inverse operation");
                require(raw(3).category == FJ_STATUS_SUCCESS, "suppressed inverse fault consumption then recovery");
                InstanceState cold;
                pending(cold, changed);
                const auto failedCold = admit_pending_render_state(cold);
                require(!admitted(failedCold) && !failedCold.directState && !failedCold.printState, "real film gate cold failure admits nothing");
                pending(current, changed);
                const auto failedCurrent = admit_pending_render_state(current);
                require(!admitted(failedCurrent) && !failedCurrent.directState && !failedCurrent.printState && !JuicerAtomic::load_shared_ptr(&current.activeDirectState) && !JuicerAtomic::load_shared_ptr(&current.activePrintState), "real film gate current false clears slot and cannot admit prior publication");
                require(held(admittedBeforeGate) == heldBeforeGate, "film gate retains previously admitted arrays/owners");
            }
            InstanceState state;
            pending(state, controls);
            auto old = admit_pending_render_state(state);
            require(admitted(old), "output-only initial");
            auto output = controls;
            output.outputCctfEncoding = !output.outputCctfEncoding;
            pending(state, output);
            auto newState = admit_pending_render_state(state);
            require(admitted(newState), "output-only new publication");
            const auto& a = old.printState ? old.printState->payload : old.directState->payload;
            const auto& b = newState.printState ? newState.printState->payload : newState.directState->payload;
            require(a.exposureTables.tablesHash == b.exposureTables.tablesHash && a.scannerTables.tablesHash == b.scannerTables.tablesHash, "unrelated encoding preserves table family identity");
        }
        std::printf("PASS %zu cold/current spectral fault-consumption, hold/cache/recovery witnesses\n", witnesses);
        fj_test_spectral_clear_fault();
    }
} // namespace
void fj_test_spectral_products(const nlohmann::json& fixture, const std::filesystem::path& resource, const std::filesystem::path& scratch, bool faults) {
    std::filesystem::create_directories(scratch);
    std::filesystem::copy(resource, scratch, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
    JuicerCuda::Owner owner;
    owner.create(scratch);
    JuicerProcess::root().ensure_bootstrap();
    if (faults) {
        fj_test_spectral_native_allocation(fixture);
        admission();
    } else
        products(fixture, scratch);
    require(owner.close().category == FJ_STATUS_SUCCESS, "spectral fixture owner close");
}
