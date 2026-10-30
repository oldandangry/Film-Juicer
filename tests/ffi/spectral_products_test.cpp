#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
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
#include "ColorTransforms.h"
#include "juicer_test_api.h"
#include "juicer_cuda_owner.h"
#include "prepared_product_fixture.h"
#include "spectral_source_fixture.h"

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
        Json bits(const float* values, std::size_t count) {
            Json out = Json::array();
            for (std::size_t i = 0; i < count; ++i)
                out.push_back(std::bit_cast<std::uint32_t>(values[i]));
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
        Json color_result(const FjScannerColor& color) {
            return {{"cat02", bits(color.cat02, 9)},
                    {"xyz_to_rgb", bits(color.xyz_to_rgb, 9)},
                    {"illuminant_xyz", bits(color.illuminant_xyz, 3)},
                    {"encoding", {color.output_space, (color.flags & 1u) != 0, (color.flags & 2u) != 0}},
                    {"gamut_recipe_hash", color.active_gamut_recipe_hash},
                    {"hash", color.hash}};
        }
        Json tables_result(const Spectral::SpectralTables& tables) {
            return {{"lambda", bits(tables.lambda)}, {"K", tables.K}, {"delta_lambda", bits(std::array{tables.deltaLambda})}, {"inv_yn", bits(std::array{tables.invYn})}, {"white_xyz", bits(tables.whiteXYZ)}, {"ref_illum_white_xyz", bits(tables.refIllumWhiteXYZ)}, {"ax", bits(tables.Ax)}, {"ay", bits(tables.Ay)}, {"az", bits(tables.Az)}, {"xbar", bits(tables.Xbar)}, {"ybar", bits(tables.Ybar)}, {"zbar", bits(tables.Zbar)}, {"illum", bits(tables.illum)}, {"eps_c", bits(tables.epsC)}, {"eps_m", bits(tables.epsM)}, {"eps_y", bits(tables.epsY)}, {"base_density_min", bits(tables.baseDensityMin)}, {"base_density_mid", bits(tables.baseDensityMid)}, {"has_baseline", tables.hasBaseline}, {"baseline_mix", bits(std::array{tables.densityBaselineMixReference})}, {"illuminant_hash", tables.illuminantHash}, {"tables_hash", tables.tablesHash}};
        }
        Json tables_result(const FjPreparedSpectralView& tables) {
            const auto lambdaCount = tables.lambda_nm.count;
            return {{"lambda", bits(tables.lambda_nm.data, lambdaCount)},
                    {"K", lambdaCount},
                    {"delta_lambda", bits(std::array{5.0f})},
                    {"inv_yn", bits(std::array{tables.inv_yn})},
                    {"white_xyz", bits(tables.white_xyz, 3)},
                    {"ref_illum_white_xyz", bits(tables.white_xyz, 3)},
                    {"ax", bits(tables.weighted_xyz[0].data, tables.weighted_xyz[0].count)},
                    {"ay", bits(tables.weighted_xyz[1].data, tables.weighted_xyz[1].count)},
                    {"az", bits(tables.weighted_xyz[2].data, tables.weighted_xyz[2].count)},
                    {"xbar", bits(tables.observer_xyz[0].data, tables.observer_xyz[0].count)},
                    {"ybar", bits(tables.observer_xyz[1].data, tables.observer_xyz[1].count)},
                    {"zbar", bits(tables.observer_xyz[2].data, tables.observer_xyz[2].count)},
                    {"illum", bits(tables.illuminant.data, tables.illuminant.count)},
                    {"eps_c", bits(tables.dyes_cmy[0].data, tables.dyes_cmy[0].count)},
                    {"eps_m", bits(tables.dyes_cmy[1].data, tables.dyes_cmy[1].count)},
                    {"eps_y", bits(tables.dyes_cmy[2].data, tables.dyes_cmy[2].count)},
                    {"base_density_min", bits(tables.baseline_min.data, tables.baseline_min.count)},
                    {"base_density_mid", bits(tables.baseline_mid.data, tables.baseline_mid.count)},
                    {"has_baseline", tables.has_baseline != 0},
                    {"baseline_mix", bits(std::array{0.0f})},
                    {"illuminant_hash", tables.illuminant_hash},
                    {"tables_hash", tables.hash}};
        }
        // Temporary I4 publication observer; switch this to a holder-backed view in I5.
        template <typename State>
        Json publication_scanner_descriptor(const State& product) {
            Scanner::ScannerSpectralLutDescriptor out;
            std::string diagnostic;
            const auto& recipe = product.recipe;
            const bool ok = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic) : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, out, diagnostic);
            require(ok, diagnostic);
            return {{"route", static_cast<int>(out.route)}, {"medium", static_cast<int>(out.medium)}, {"polarity", static_cast<int>(out.polarity)}, {"density_bounds_hash", out.densityBoundsHash}, {"channel_density_hash", out.channelDensityHash}, {"base_density_hash", out.baseDensityHash}, {"illuminant_hash", out.scanIlluminantHash}, {"observer_hash", out.observerHash}, {"resolution", out.lutResolution}, {"hash", out.hash}};
        }
        // Temporary I4 publication observer; switch this to a holder-backed view in I5.
        template <typename State>
        Json publication_complete_result(const State& product) {
            const auto& recipe = product.recipe;
            const auto& raw = recipe.filmRaw;
            const auto& reference = recipe.scannerOutput.syntheticFilmReference;
            const auto& balance = recipe.print.balance;
            const auto& config = product.payload.filmRawConfig;
            return {{"input_rgb_to_xyz", bits(raw.inputRgbToXyz)}, {"input_adapt", bits(raw.inputXyzAdapt)}, {"xyz_to_linear_srgb", bits(raw.xyzToLinearSrgb)}, {"source_white", bits(raw.inputNominalWhiteXYZ)}, {"projection_white", bits(raw.projectionWhiteXYZ)}, {"config_rgb_to_xyz", bits(config.inputRGBToXYZ.m)}, {"config_adapt", bits(config.inputXYZAdapt.m)}, {"raw_midgray", bits(config.rawMidgray)}, {"midgray_dwg", bits(config.midgrayDWG)}, {"midgray_scale_green", bits(std::array{config.midgrayScale, config.rawMidgrayGreen})}, {"config_whites", {bits(config.inputWhiteXYZ), bits(config.workingWhiteXYZ), bits(config.refIllumWhiteXYZ)}}, {"config_flags", {static_cast<int>(config.inputColorSpace), config.applyCctfDecoding, static_cast<int>(config.spectralUpsamplingMode), config.applyInputChromaticAdapt, config.hasRefIllumWhite, config.valid}}, {"spd_s_inv", bits(product.payload.spdSInv)}, {"baseline_raw", bits(reference.baselineRawRgb)}, {"compensated_raw", bits(reference.compensatedRawRgb)}, {"baseline_density", bits(reference.baselineDensityCmy)}, {"compensated_density", bits(reference.compensatedDensityCmy)}, {"print_baseline_raw", bits(balance.baselinePrintRawRgb)}, {"print_compensated_raw", bits(balance.compensatedPrintRawRgb)}, {"print_factors", bits(std::array{balance.factorMidgray, balance.factorMidgrayComp, balance.normalizer})}, {"hashes", {raw.hash, raw.tcLutHash, reference.hash, balance.hash, recipe.print.hash, recipe.scannerOutput.hash, recipe.scannerOutput.outputGamut.hash, recipe.hash, product.payload.uploadCoreHash, product.payload.scannerHash}}, {"scanner_color", color_result(product.payload.scannerColor)}, {"scanner_tables", tables_result(product.payload.scannerTables)}, {"scanner_descriptor", publication_scanner_descriptor(product)}, {"gamut_entry", bits(recipe.scannerOutput.outputGamut.transform.nativeRgbToD65Xyz)}, {"gamut_exit", bits(recipe.scannerOutput.outputGamut.transform.d65XyzToNativeRgb)}};
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
            const auto sensitivityMethod = raw.method == FJ_RAW_MALLETT_2019 ? 1 : 0;
            return {{"input_rgb_to_xyz", bits(raw.input_rgb_to_xyz, 9)},
                    {"input_adapt", bits(raw.input_xyz_adapt, 9)},
                    {"xyz_to_linear_srgb", bits(raw.xyz_to_linear_srgb, 9)},
                    {"source_white", bits(input.nominal_white_xyz, 3)},
                    {"projection_white", bits(input.projection_white_xyz, 3)},
                    {"config_rgb_to_xyz", bits(raw.input_rgb_to_xyz, 9)},
                    {"config_adapt", bits(raw.input_xyz_adapt, 9)},
                    {"raw_midgray", bits(view.raw_midgray, 3)},
                    {"midgray_dwg", bits(view.midgray_dwg_rgb, 3)},
                    {"midgray_scale_green", bits(std::array{view.midgray_scale, view.raw_green})},
                    {"config_whites", {bits(input.nominal_white_xyz, 3), bits(input.projection_white_xyz, 3), bits(input.projection_white_xyz, 3)}},
                    {"config_flags", {raw.input_color_space, (raw.flags & FJ_FILM_DECODE_CCTF) != 0, sensitivityMethod, (raw.flags & FJ_FILM_ADAPT_INPUT) != 0, true, true}},
                    {"spd_s_inv", bits(view.s_inverse, 9)},
                    {"baseline_raw", bits(reference.baseline_raw_rgb, 3)},
                    {"compensated_raw", bits(reference.compensated_raw_rgb, 3)},
                    {"baseline_density", bits(reference.baseline_density_cmy, 3)},
                    {"compensated_density", bits(reference.compensated_density_cmy, 3)},
                    {"print_baseline_raw", bits(balance.baseline_raw_rgb, 3)},
                    {"print_compensated_raw", bits(balance.compensated_raw_rgb, 3)},
                    {"print_factors", bits(JuicerTestSupport::captured_print_factors(view))},
                    {"hashes", {raw.hash, raw.tc_lut_hash, reference.hash, balance.hash, view.print_recipe_hash, view.scanner_recipe_hash, view.output_color.gamut_recipe_hash, view.recipe_hash, view.upload_core_hash, view.scanner_hash}},
                    {"scanner_color", color_result(view.scanner_color)},
                    {"scanner_tables", tables_result(view.scanner_tables)},
                    {"scanner_descriptor", scanner_descriptor(view.scanner_lut)},
                    {"gamut_entry", bits(view.output_color.native_rgb_to_d65_xyz, 9)},
                    {"gamut_exit", bits(view.output_color.d65_xyz_to_native_rgb, 9)}};
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
    // SF_TEMP_BRIDGE_NATIVE_SPECTRAL_PUBLICATION: preserve the old native
    // terminal mapping until I5 switches publication and I7 removes its owner.
    std::optional<PendingRenderAdmissionResult> SF_TEMP_BRIDGE_native_failed_attempt(InstanceState& state, std::uint32_t category, bool ordinary) {
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
    std::uint64_t spectral_source_hash(FjFloatSpan source) {
        std::uint64_t hash = UINT64_C(0xcbf29ce484222325);
        for (std::size_t i = 0; i < source.count; ++i) {
            const auto word = std::bit_cast<std::uint32_t>(source.data[i]);
            for (unsigned shift = 0; shift < 32; shift += 8) {
                hash = (hash ^ ((word >> shift) & 255u)) * UINT64_C(0x100000001b3);
            }
        }
        return hash;
    }
    fs::path scaled_cmf_root(const fs::path& resources, const fs::path& scratch, int route) {
        const auto root = scratch / ("spectral-inverse-suppression-route-" + std::to_string(route));
        fs::remove_all(root);
        fs::create_directories(root.parent_path());
        fs::copy(resources, root, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        const auto sourcePath = resources / "cie1931_2deg.csv";
        const auto targetPath = root / "cie1931_2deg.csv";
        std::ifstream input(sourcePath);
        std::ofstream output(targetPath, std::ios::trunc);
        require(input.good() && output.good(), "open isolated CMF source files");
        std::string line;
        std::size_t rows = 0;
        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        while (std::getline(input, line)) {
            std::istringstream fields(line);
            std::array<std::string, 4> authored{};
            for (auto& field : authored) {
                require(static_cast<bool>(std::getline(fields, field, ',')), "CMF source row has four columns");
            }
            output << authored[0];
            for (std::size_t column = 1; column < authored.size(); ++column) {
                const float sample = std::strtof(authored[column].c_str(), nullptr);
                output << ',' << std::ldexp(sample, -136);
            }
            output << '\n';
            ++rows;
        }
        require(rows == 81 && input.eof() && output.good(), "write all 81 scaled CMF rows");
        return root;
    }
    struct HeldValues {
        Json complete;
        Json exposureTables;
        std::vector<std::uint32_t> tcRgba, printLight, boundary;
        bool operator==(const HeldValues&) const = default;
    };
    template <typename Range>
    std::vector<std::uint32_t> retained_bits(const Range& samples) {
        std::vector<std::uint32_t> out;
        out.reserve(samples.size());
        for (float value : samples)
            out.push_back(std::bit_cast<std::uint32_t>(value));
        return out;
    }
    HeldValues held(const PendingRenderAdmissionResult& result) {
        require(bool(result.directState) || bool(result.printState), "retained admitted state");
        HeldValues out;
        out.complete = result.printState ? publication_complete_result(*result.printState) : publication_complete_result(*result.directState);
        const auto& p = result.printState ? result.printState->payload : result.directState->payload;
        out.exposureTables = tables_result(p.exposureTables);
        if (p.filmTcLut)
            out.tcRgba = retained_bits(p.filmTcLut->samples());
        if (p.printMainIlluminant)
            out.printLight = retained_bits(*p.printMainIlluminant);
        if (p.outputBoundaryTable)
            out.boundary = retained_bits(p.outputBoundaryTable->cmax);
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
                if (row.contains("variant")) {
                    const int variant = row.at("variant");
                    p.hanatos2025AdaptationWindow = variant == 1 || variant >= 3;
                    p.hanatos2025AdaptationSurface = variant == 2 || variant == 4;
                    p.cameraFilterOverride = variant >= 3;
                    p.cameraFilterUV = {.75, 410, 8};
                    p.cameraFilterIR = {.5, 675, 15};
                }
                require(controls_json(p) == row.at("controls"), "all serialized native controls preserved");
                std::array<char, 2048> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                std::optional<PreparedState> product;
                const auto prepared = prepare_state(assets.borrow_rust_assets(), p, product, error);
                const bool ok = prepared.category == FJ_STATUS_SUCCESS && product.has_value();
                require(ok == row.at("built"), "native product completion/failure " + label.get<std::string>());
                if (ok) {
                    FjPreparedStateView view{};
                    const auto viewStatus = product->view(view, error);
                    require(viewStatus.category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
                    auto result = complete_result(*product);
                    result["exposure_tables"] = tables_result(view.exposure_tables);
                    result["final_sensitivity_hash"] = view.film_exposure.sensitivity_hash;
                    if (row.contains("variant")) {
                        result["mallett_green_scale"] = bits(std::array{view.film_exposure.mallett_green_midgray_scale});
                        FjFilmProfile* acquiredProfile = nullptr;
                        const auto acquired = fj_legacy_film_profile_acquire(assets.borrow_rust_assets(), view.film_profile_key, &acquiredProfile, nullptr);
                        require(acquired.category == FJ_STATUS_SUCCESS && acquired.api == FJ_API_NONE && acquired.native_code == 0 && acquiredProfile,
                                "selected film profile source acquire");
                        const auto releaseProfile = [](FjFilmProfile* owner) noexcept {
                            const auto released = fj_legacy_film_profile_release(owner, nullptr);
                            if (released.category != FJ_STATUS_SUCCESS || released.api != FJ_API_NONE || released.native_code != 0)
                                std::abort();
                        };
                        std::unique_ptr<FjFilmProfile, decltype(releaseProfile)> profileOwner(acquiredProfile, releaseProfile);
                        FjFilmProfileView profile{};
                        const auto viewed = fj_legacy_film_profile_view(profileOwner.get(), &profile, nullptr);
                        require(viewed.category == FJ_STATUS_SUCCESS && viewed.api == FJ_API_NONE && viewed.native_code == 0,
                                "selected film profile source view");
                        require(profile.asset_token == view.film_profile_asset_version, "profile source token agrees with held complete state");
                        for (std::size_t i = 0; i < profile.tables.linear_sensitivity_rgb.count; i += 3)
                            result["profile_linear"].push_back(bits(profile.tables.linear_sensitivity_rgb.data + i, 3));
                        result["reference_spd"] = bits(view.exposure_tables.illuminant.data, view.exposure_tables.illuminant.count);
                        result["reference_white"] = bits(view.exposure_tables.white_xyz, 3);
                        result["recipe_window"] = bits(view.input_policy.hanatos_window.data, view.input_policy.hanatos_window.count);
                        result["recipe_blur"] = bits(std::array{view.input_policy.hanatos_spectral_blur});
                        result["recipe_band_pass_uv"] = bits(view.input_policy.uv, 3);
                        result["recipe_band_pass_ir"] = bits(view.input_policy.ir, 3);
                    }
                    for (std::size_t i = 0; i < view.film_exposure.sensitivity_rgb.count; i += 3)
                        result["final_sensitivity"].push_back(bits(view.film_exposure.sensitivity_rgb.data + i, 3));
                    if (route) {
                        result["main_light"] = bits(view.print.main_illuminant.data, view.print.main_illuminant.count);
                        result["preflash_light"] = bits(view.print.preflash_illuminant.data, view.print.preflash_illuminant.count);
                        result["preflash_raw"] = bits(view.print.preflash_raw_cmy, 3);
                        result["print_descriptors"] = {view.print.hash, view.print.main_illuminant_hash, view.print.preflash_illuminant_hash, view.print.preflash_raw_hash, view.print.balance_hash};
                    }
                    const auto& expected = fixture.contains("expected_records")
                                               ? fixture.at("expected_records").at(row.at("expected_record").get<std::size_t>())
                                               : row.at("expected");
                    if (result != expected) {
                        const auto diff = Json::diff(expected, result);
                        throw std::runtime_error("completed product first difference " + std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(method) + "/" + label.get<std::string>() + ": " + diff[0].dump());
                    }
                } else {
                    const std::string expected = row.at("diagnostic");
                    const std::string actualDiagnostic(diagnostic.data(), error.length);
                    require(prepared.category == FJ_STATUS_PREPARATION_FAILURE && prepared.api == FJ_API_NONE &&
                                prepared.native_code == 0 && !product,
                            "computed product failure returns its exact category and no owner");
                    require(!actualDiagnostic.empty(), "native failed product diagnostic retained");
                    const bool sourceFailure = expected.starts_with("profile ");
                    const std::string expectedRust = expected == "MalformedRequiredProfileData phase=3B field=final_sensitivity"
                                                         ? expected + " requirement=BandPassResponse"
                                                         : expected;
                    std::string comparison = sourceFailure ? "source failure stage retained with scratch-root path: " : "computed failure diagnostic exact: ";
                    comparison += actualDiagnostic;
                    comparison += " expected ";
                    comparison += expectedRust;
                    require(sourceFailure ? actualDiagnostic.ends_with("info.reference_illuminant requires Illuminant") : actualDiagnostic == expectedRust, comparison);

                    // SF_TEMP_BRIDGE_NATIVE_FAILURE_DIAGNOSTIC: retain each
                    // frozen native diagnostic until its I7 constructor gate.
                    FocusedRenderStateBuildProduct nativeComparison;
                    std::string nativeDiagnostic;
                    const bool nativeBuilt = route ? build_print_render_state_product(p, nativeComparison, nativeDiagnostic)
                                                   : build_direct_render_state_product(p, nativeComparison, nativeDiagnostic);
                    require(!nativeBuilt, "original native product rejects the unchanged failure input");
                    require(sourceFailure ? nativeDiagnostic.ends_with("info.reference_illuminant requires Illuminant") : nativeDiagnostic == expected,
                            "frozen native failure diagnostic retained: " + nativeDiagnostic);
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
        require(count == (fixture.contains("product_count") ? fixture.at("product_count").get<std::size_t>() : 72u), "complete native capture product membership");
    }
    void admission(const fs::path& resources, const fs::path& scratch) {
        auto& assets = JuicerProcess::root().assets();
        const JuicerTestSupport::CmfSource defaultCmf(assets.borrow_rust_assets());
        std::size_t witnesses = 0;
        for (int route = 0; route < 4; ++route) {
            const auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), Spektrafilm::RgbToRawMethod::Mallett2019);
            auto changed = controls;
            changed.cameraExposureCompensationEv = 1.0;
            const auto raw = [&](std::uint32_t operation, const JuicerTestSupport::CmfSource& cmf) {
                std::array<float, 81> ill{};
                ill.fill(1);
                const FjSpectralObserver observer{cmf.x(), cmf.y(), cmf.z()};
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
                    require(fj_test_spectral_arm_fault(op, index, fault).category == FJ_STATUS_SUCCESS, "arm direct prepared-state fault");
                    std::array<char, 2048> directDiagnostic{};
                    FjErrorBuffer directError{directDiagnostic.data(), directDiagnostic.size(), 0};
                    std::optional<PreparedState> directProduct;
                    const auto directStatus = prepare_state(assets.borrow_rust_assets(), controls, directProduct, directError);
                    require(directStatus.category == category[fault] && !directProduct && directStatus.api == FJ_API_NONE && directStatus.native_code == 0,
                            "actual prepared-state spectral status/category");
                    require(raw(op, defaultCmf).category == FJ_STATUS_SUCCESS, "prepared-state raw fault consumed");
                    InstanceState cold;
                    pending(cold, controls);
                    require(fj_test_spectral_arm_fault(op, index, fault).category == FJ_STATUS_SUCCESS, "arm cold spectral fault");
                    const auto coldFailure = SF_TEMP_BRIDGE_native_failed_attempt(cold, category[fault], op == 2 && fault == 4);
                    require((!coldFailure || !admitted(*coldFailure)) && !JuicerAtomic::load_shared_ptr(&cold.activeDirectState) && !JuicerAtomic::load_shared_ptr(&cold.activePrintState) && cold.lastHash.load() == 0, "cold failure admits/publishes nothing");
                    require(raw(op, defaultCmf).category == FJ_STATUS_SUCCESS, "cold failure consumes actual same-thread operation");
                    InstanceState state;
                    pending(state, controls);
                    const auto old = admit_pending_render_state(state);
                    require(admitted(old), "initial spectral publication");
                    const auto oldValues = held(old);
                    const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                    pending(state, changed);
                    require(fj_test_spectral_arm_fault(op, index, fault).category == FJ_STATUS_SUCCESS, "arm current spectral fault");
                    const auto failed = SF_TEMP_BRIDGE_native_failed_attempt(state, category[fault], op == 2 && fault == 4);
                    require(!failed || !admitted(*failed), "changed-current failure cannot admit older state");
                    const bool ordinary = op == 2 && fault == 4;
                    if (!ordinary) {
                        const char* names[] = {"", "UnsupportedInput", "InternalFailure", "AllocationFailure", "PreparationFailure"};
                        require(!failed || failed->diagnostic.find(names[fault]) != std::string::npos, "fatal category retained in admission diagnostic");
                        require(JuicerAtomic::load_shared_ptr(&state.activeDirectState) == old.directState && JuicerAtomic::load_shared_ptr(&state.activePrintState) == old.printState, "exceptional preparation retains active publication");
                    } else {
                        require(!JuicerAtomic::load_shared_ptr(&state.activeDirectState) && !JuicerAtomic::load_shared_ptr(&state.activePrintState), "ordinary preparation false clears active publication");
                    }
                    require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "failed attempt publishes no new identity/counter");
                    require(raw(op, defaultCmf).category == FJ_STATUS_SUCCESS, "same-thread raw export proves consumption before recovery");
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
                    std::array<char, 2048> diagnostic{};
                    FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                    std::optional<PreparedState> product;
                    require(fj_test_spectral_arm_fault(op, count + 1, 1).category == FJ_STATUS_SUCCESS, "arm exact production operation count");
                    const auto prepared = prepare_state(assets.borrow_rust_assets(), p, product, error);
                    require(prepared.category == FJ_STATUS_SUCCESS && product, "exact count complete prepared state");
                    require(raw(op, defaultCmf).category == FJ_STATUS_UNSUPPORTED_INPUT, "next same-thread raw call proves complete production call count");
                    require(raw(op, defaultCmf).category == FJ_STATUS_SUCCESS, "count witness consumed");
                }
            const auto isolatedRoot = scaled_cmf_root(resources, scratch, route);
            const JuicerTestSupport::SpectralSourceAssets isolatedAssets(isolatedRoot);
            const JuicerTestSupport::CmfSource scaledCmf(isolatedAssets.get());
            {
                const auto originalHash = spectral_source_hash(defaultCmf.rows());
                const auto scaledHash = spectral_source_hash(scaledCmf.rows());
                require(defaultCmf.rows().count == 324 && scaledCmf.rows().count == 324 && scaledHash != originalHash,
                        "inverse suppression uses distinct complete Rust-owned CMF source");
                for (std::size_t i = 0; i < 81; ++i) {
                    require(std::bit_cast<std::uint32_t>(scaledCmf.rows().data[4 * i]) == std::bit_cast<std::uint32_t>(defaultCmf.rows().data[4 * i]),
                            "scaled CMF preserves authored wavelength words");
                    for (std::size_t lane = 1; lane < 4; ++lane) {
                        require(std::bit_cast<std::uint32_t>(scaledCmf.rows().data[4 * i + lane]) ==
                                    std::bit_cast<std::uint32_t>(std::ldexp(defaultCmf.rows().data[4 * i + lane], -136)),
                                "scaled Rust CMF input is exact ldexp f32 source");
                    }
                }
                std::printf("inverse suppression route=%d cmf_hash=%016llx scaled_hash=%016llx first_xyz=%08x,%08x,%08x\n",
                            route,
                            static_cast<unsigned long long>(originalHash),
                            static_cast<unsigned long long>(scaledHash),
                            std::bit_cast<std::uint32_t>(scaledCmf.x().data[0]),
                            std::bit_cast<std::uint32_t>(scaledCmf.y().data[0]),
                            std::bit_cast<std::uint32_t>(scaledCmf.z().data[0]));
                require(fj_test_spectral_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "arm inverse suppression witness");
                std::array<char, 2048> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                std::optional<PreparedState> product;
                const auto prepared = prepare_state(isolatedAssets.get(), changed, product, error);
                require(prepared.category == FJ_STATUS_PREPARATION_FAILURE && prepared.api == FJ_API_NONE && prepared.native_code == 0 && !product &&
                            std::string(diagnostic.data(), error.length) == "MalformedRequiredResource component=focused_film_payload field=exposure_tables",
                        "real Rust-owned f32 reciprocal overflow reaches film table gate");
                require(raw(3, scaledCmf).category == FJ_STATUS_UNSUPPORTED_INPUT, "failed film white/hash gate never consumes inverse operation");
                require(raw(3, scaledCmf).category == FJ_STATUS_SUCCESS, "suppressed inverse fault consumption then recovery");
            }
            {
                // SF_TEMP_BRIDGE_NATIVE_SPECTRAL_PUBLICATION: remove with the I7 native recipe/publication bridge.
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
                require(Spectral::gXBar.linear.size() == 81 && Spectral::gYBar.linear.size() == 81 && Spectral::gZBar.linear.size() == 81,
                        "native comparison CMF lanes retain source extents");
                for (auto* lane : {&Spectral::gXBar.linear, &Spectral::gYBar.linear, &Spectral::gZBar.linear})
                    for (auto& value : *lane)
                        value = std::ldexp(value, -136);
                for (std::size_t i = 0; i < 81; ++i) {
                    require(std::bit_cast<std::uint32_t>(Spectral::gXBar.linear[i]) == std::bit_cast<std::uint32_t>(scaledCmf.x().data[i]) &&
                                std::bit_cast<std::uint32_t>(Spectral::gYBar.linear[i]) == std::bit_cast<std::uint32_t>(scaledCmf.y().data[i]) &&
                                std::bit_cast<std::uint32_t>(Spectral::gZBar.linear[i]) == std::bit_cast<std::uint32_t>(scaledCmf.z().data[i]),
                            "Rust scaled CMF source reproduces prior native observer input words");
                }
                require(fj_test_spectral_arm_fault(3, 1, 1).category == FJ_STATUS_SUCCESS, "arm inverse suppression comparison witness");
                require(raw(3, scaledCmf).category == FJ_STATUS_UNSUPPORTED_INPUT, "scaled source raw call consumes separately armed bridge fault");
                require(raw(3, scaledCmf).category == FJ_STATUS_SUCCESS, "scaled source raw follow-up recovers");
                InstanceState cold;
                pending(cold, changed);
                const auto failedCold = admit_pending_render_state(cold);
                require(!admitted(failedCold) && !failedCold.directState && !failedCold.printState, "real film gate cold failure admits nothing");
                pending(current, changed);
                const auto failedCurrent = admit_pending_render_state(current);
                require(!admitted(failedCurrent) && !failedCurrent.directState && !failedCurrent.printState && !JuicerAtomic::load_shared_ptr(&current.activeDirectState) && !JuicerAtomic::load_shared_ptr(&current.activePrintState), "real film gate current false clears slot and cannot admit prior publication");
                require(held(admittedBeforeGate) == heldBeforeGate, "film gate retains previously admitted arrays/owners");
            }
            require(raw(3, defaultCmf).category == FJ_STATUS_SUCCESS, "default Rust CMF observer restored after bridge comparison");
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
    void exposure_admission() {
        auto& assets = JuicerProcess::root().assets();
        std::size_t witnesses = 0;
        const auto attempt = [](InstanceState& state, std::uint32_t fault) {
            try {
                const auto result = admit_pending_render_state(state);
                require(fault != 3 && result.status == PendingRenderAdmissionStatus::RebuildFailed && !admitted(result), "A5 failed admission disposition");
            } catch (const std::bad_alloc&) {
                require(fault == 3, "A5 allocation keeps memory terminal");
            }
        };
        for (int route = 0; route < 4; ++route)
            for (int method = 0; method < 3; ++method) {
                auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
                controls.hanatos2025AdaptationWindow = 1;
                auto changed = controls;
                changed.cameraExposureCompensationEv += 1.0;
                for (std::uint32_t op = 1; op <= 2; ++op) {
                    if (op == 1 && method != 0)
                        continue;
                    for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                        InstanceState cold;
                        pending(cold, controls);
                        require(fj_test_exposure_arm_fault(op, 1, fault).category == FJ_STATUS_SUCCESS, "arm cold A5 fault");
                        attempt(cold, fault);
                        require(fj_test_exposure_fault_consumed(op, fault) == 1, "cold actual A5 consumption before recovery");
                        require(!JuicerAtomic::load_shared_ptr(&cold.activeDirectState) && !JuicerAtomic::load_shared_ptr(&cold.activePrintState) && cold.lastHash.load() == 0, "A5 cold failure publishes nothing");
                        InstanceState current;
                        pending(current, controls);
                        const auto old = admit_pending_render_state(current);
                        require(admitted(old), "A5 initial publication");
                        const auto values = held(old);
                        const auto hash = current.lastHash.load(), counter = current.buildCounterNext.load();
                        pending(current, changed);
                        require(fj_test_exposure_arm_fault(op, 1, fault).category == FJ_STATUS_SUCCESS, "arm changed-current A5 fault");
                        attempt(current, fault);
                        require(fj_test_exposure_fault_consumed(op, fault) == 1, "changed-current actual A5 consumption before cleanup/recovery");
                        if (fault == 4)
                            require(!JuicerAtomic::load_shared_ptr(&current.activeDirectState) && !JuicerAtomic::load_shared_ptr(&current.activePrintState), "A5 ordinary false clears publication");
                        else
                            require(JuicerAtomic::load_shared_ptr(&current.activeDirectState) == old.directState && JuicerAtomic::load_shared_ptr(&current.activePrintState) == old.printState, "A5 exceptional failure preserves publication");
                        require(current.lastHash.load() == hash && current.buildCounterNext.load() == counter, "A5 failed current adds no identity/counter");
                        require(held(old) == values && assets.release_cached_payloads().category == FJ_STATUS_SUCCESS && held(old) == values, "A5 held complete owner survives failure/cache release");
                        pending(current, changed);
                        const auto recovered = admit_pending_render_state(current);
                        require(admitted(recovered) && current.lastHash.load() == hash_params(changed) && current.buildCounterNext.load() == counter + 1, "A5 recovery publishes changed-current once");
                        require(held(old) == values, "A5 old admitted arrays survive replacement");
                        ++witnesses;
                    }
                }
                // An armed reference operation must stay untouched for every skipped branch.
                if (method != 0) {
                    require(fj_test_exposure_arm_fault(1, 1, 1).category == FJ_STATUS_SUCCESS, "arm skipped A5 reference");
                    std::array<char, 2048> diagnostic{};
                    FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                    std::optional<PreparedState> product;
                    const auto prepared = prepare_state(assets.borrow_rust_assets(), controls, product, error);
                    require(prepared.category == FJ_STATUS_SUCCESS && product, "non-Hanatos prepared state skips reference");
                    require(fj_test_exposure_fault_consumed(1, 1) == 0, "skipped reference does not consume fault");
                    std::vector<float> spectra(std::size_t{192} * 192u * 81u, 1);
                    FjReferenceWhiteInput input{{spectra.data(), spectra.size()}, {1, 1, 1}, 0};
                    FjReferenceWhite output{};
                    std::array<char, 256> bytes{};
                    FjErrorBuffer rawError{bytes.data(), bytes.size(), 0};
                    require(fj_legacy_reconstruction_reference_white(&input, &output, &rawError).category == FJ_STATUS_UNSUPPORTED_INPUT && fj_test_exposure_fault_consumed(1, 1) == 1, "next raw call consumes skipped reference witness");
                }
            }
        fj_test_exposure_clear_fault();
        std::printf("PASS %zu A5 cold/current, actual fault-consumption, memory/ordinary terminals, hold/cache/recovery witnesses\n", witnesses);
    }
} // namespace
void fj_test_spectral_products(const nlohmann::json& fixture, const std::filesystem::path& resource, const std::filesystem::path& scratch, bool faults, void (*nativeAllocationCheck)(const nlohmann::json&)) {
    if (!faults) {
        std::filesystem::create_directories(scratch);
        std::filesystem::copy(resource, scratch, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
    }
    JuicerCuda::Owner owner;
    owner.create(faults ? resource : scratch);
    JuicerProcess::root().ensure_bootstrap();
    if (faults) {
        require(nativeAllocationCheck != nullptr, "spectral admission requires its owning allocation check");
        nativeAllocationCheck(fixture);
        admission(resource, scratch);
    } else
        products(fixture, scratch);
    require(owner.close().category == FJ_STATUS_SUCCESS, "spectral fixture owner close");
}

void fj_test_exposure_admission(const std::filesystem::path& resource) {
    JuicerCuda::Owner owner;
    owner.create(resource);
    JuicerProcess::root().ensure_bootstrap();
    exposure_admission();
    require(owner.close().category == FJ_STATUS_SUCCESS, "A5 admission owner close");
}

void fj_test_tc_lut_products(const nlohmann::json& fixture) {
    for (const auto& row : fixture["products"]) {
        const int polarity = row["polarity"], route = row["route"], method = row["method"], variant = row["variant"];
        auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), static_cast<Spektrafilm::RgbToRawMethod>(method));
        p.hanatos2025AdaptationSurface = variant == 1;
        p.hanatos2025AdaptationWindow = variant == 1;
        p.inputCompressionEnabled = variant == 2;
        p.printPreflashExposure = route ? .1 : 0;
        p.preflashMFilterCc = 12.5;
        p.preflashYFilterCc = 7.25;
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        std::optional<PreparedState> product;
        const auto prepared = prepare_state(JuicerProcess::root().assets().borrow_rust_assets(), p, product, error);
        const bool built = prepared.category == FJ_STATUS_SUCCESS && product.has_value();
        require(built == row["built"].get<bool>() && std::string(diagnostic.data(), error.length) == row["diagnostic"].get<std::string>(), "TC actual product status");
        if (!built)
            continue;
        FjPreparedStateView view{};
        require(product->view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
        const auto actual = complete_result(*product);
        for (const auto& [name, value] : actual.items())
            require(value == row["expected"][name], "TC complete native product field: " + name);
        require((view.film_exposure.tc_lut_rgba.count != 0) == (method != 1), "Mallett no-TC owner");
        if (view.film_exposure.tc_lut_rgba.count) {
            std::uint64_t hash = Hash::kFnvOffset;
            const auto samples = view.film_exposure.tc_lut_rgba;
            Hash::hash_bytes_update(hash, samples.data, samples.count * sizeof(float));
            require(hash == row["expected"]["tc_table"]["native_byte_digest_fnv1a64"], "actual TC product complete table");
        }
        if (variant == 0) {
            const auto key = view.film_exposure.tc_lut_hash;
            auto changed = p;
            changed.outputCctfEncoding = changed.outputCctfEncoding ? 0 : 1;
            auto outputOnly = JuicerTestSupport::prepare_product(changed);
            FjPreparedStateView outputView{};
            require(outputOnly.view(outputView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(outputView.film_exposure.tc_lut_hash == key && outputView.upload_core_hash != view.upload_core_hash, "CCTF-only keeps TC identity and changes the enclosing upload identity");
            changed = p;
            changed.cameraExposureCompensationEv += .25;
            auto exposureOnly = JuicerTestSupport::prepare_product(changed);
            FjPreparedStateView exposureView{};
            require(exposureOnly.view(exposureView, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
            require(exposureView.film_exposure.tc_lut_hash == key && exposureView.recipe_hash != view.recipe_hash, "manual exposure excludes TC but invalidates enclosing recipe");
        }
    }
}
void fj_test_tc_lut_admission() {
    std::size_t count = 0;
    for (int route = 0; route < 4; ++route)
        for (int method : {0, 2}) {
            auto p = snapshot(static_cast<Spektrafilm::ScanRoute>(route), static_cast<Spektrafilm::RgbToRawMethod>(method));
            InstanceState baseline;
            pending(baseline, p);
            const auto publication = admit_pending_render_state(baseline);
            require(admitted(publication), "TC immutable baseline admitted");
            for (std::uint32_t operation = 1; operation <= 2; ++operation)
                for (std::uint32_t index = 1; index <= (operation == 1 ? 1u : 3u); ++index)
                    for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                        for (bool current : {false, true}) {
                            InstanceState state;
                            std::optional<PendingRenderAdmissionResult> old;
                            if (current) {
                                // Only the prior immutable publication is shared; fault/recovery construction is fresh.
                                JuicerAtomic::store_shared_ptr(&state.activeDirectState, publication.directState);
                                JuicerAtomic::store_shared_ptr(&state.activePrintState, publication.printState);
                                state.lastHash.store(baseline.lastHash.load());
                                state.buildCounterNext.store(baseline.buildCounterNext.load());
                                pending(state, p);
                                old = admit_pending_render_state(state);
                                require(admitted(*old) && old->directState == publication.directState && old->printState == publication.printState, "TC current reuses immutable baseline");
                            }
                            const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                            auto changed = p;
                            changed.cameraExposureCompensationEv += .25;
                            pending(state, changed);
                            require(fj_test_tc_lut_arm_fault(operation, index, fault).category == FJ_STATUS_SUCCESS, "arm real TC caller");
                            bool memory = false;
                            try {
                                const auto result = admit_pending_render_state(state);
                                require(result.status == PendingRenderAdmissionStatus::RebuildFailed && !result.directState && !result.printState, "TC failed current never admits older state");
                            } catch (const std::bad_alloc&) {
                                memory = true;
                            }
                            require(fj_test_tc_lut_fault_consumed(operation, fault) == 1, "actual raw consumption before cleanup/recovery");
                            require(memory == (fault == 3), "allocation reaches native bad_alloc terminal");
                            require(state.lastHash.load() == hash && state.buildCounterNext.load() == counter, "TC failure preserves identities/counters");
                            if (fault == 4)
                                require(!JuicerAtomic::load_shared_ptr(&state.activeDirectState) && !JuicerAtomic::load_shared_ptr(&state.activePrintState), "ordinary TC false clears selected publication");
                            else if (current)
                                require(JuicerAtomic::load_shared_ptr(&state.activeDirectState) == old->directState && JuicerAtomic::load_shared_ptr(&state.activePrintState) == old->printState, "exceptional TC failure preserves publication");
                            if (current) {
                                const auto& payload = old->printState ? old->printState->payload : old->directState->payload;
                                std::array<float, 3> rgb{};
                                std::string diagnostic;
                                require(Spectral::sample_film_tc_lut(*payload.filmTcLut, {1, 0, 0}, rgb, diagnostic), "admitted immutable hold survives failure");
                            }
                            fj_test_tc_lut_clear_fault();
                            const auto recovered = admit_pending_render_state(state);
                            require(admitted(recovered) && state.lastHash.load() == hash_params(changed) && state.buildCounterNext.load() == counter + 1, "TC recovery publishes once");
                            ++count;
                        }
                    }
            auto mallett = p;
            mallett.spectralUpsamplingMode = 1;
            for (std::uint32_t operation = 1; operation <= 2; ++operation) {
                require(fj_test_tc_lut_arm_fault(operation, 1, 1).category == FJ_STATUS_SUCCESS, "arm skipped TC");
                std::array<char, 2048> diagnostic{};
                FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
                std::optional<PreparedState> product;
                const auto prepared = prepare_state(JuicerProcess::root().assets().borrow_rust_assets(), mallett, product, error);
                require(prepared.category == FJ_STATUS_SUCCESS && product, "Mallett complete state");
                require(fj_test_tc_lut_fault_consumed(operation, 1) == 0, "Mallett skipped call cannot consume TC fault");
                fj_test_tc_lut_clear_fault();
            }
        }
    require(count == 256, "all TC route/algorithm/call-position/fault/cold/current witnesses retained");
    std::printf("PASS %zu TC actual-caller cold/current/failure/retention/recovery witnesses\n", count);
}

void fj_test_input_hull_products(const nlohmann::json& fixture, const std::filesystem::path& tablesPath) {
    require(fj_test_gamut_arm_fault(0, 1).category == FJ_STATUS_SUCCESS, "arm cold no-work hull witness");
    for (int method : {0, 1, 2}) {
        auto controls = snapshot(Spektrafilm::ScanRoute::NegativeDirectScan, static_cast<Spektrafilm::RgbToRawMethod>(method));
        controls.inputCompressionEnabled = method == 1;
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        std::optional<PreparedState> product;
        const auto prepared = prepare_state(JuicerProcess::root().assets().borrow_rust_assets(), controls, product, error);
        require(prepared.category == FJ_STATUS_SUCCESS && product, "disabled/Mallett complete state");
        FjPreparedStateView view{};
        require(product->view(view, error).category == FJ_STATUS_SUCCESS && view.input_hull.xy.count == 0, "disabled/Mallett acquires no hull");
        require(fj_test_gamut_fault_consumed(0, 1) == 0, "no-work does not consume construction fault");
    }
    const auto d65 = JuicerProcess::root().assets().illuminant_filter_curves();
    const JuicerTestSupport::CmfSource cmf(JuicerProcess::root().assets().borrow_rust_assets());
    FjInputHullSource source{cmf.x(), cmf.y(), cmf.z(), {d65->d65.linear.data(), 81}};
    FjInputHull hull{};
    FjErrorBuffer error{nullptr, 0, 0};
    require(fj_legacy_gamut_input_hull(&source, &hull, &error).category == FJ_STATUS_UNSUPPORTED_INPUT && fj_test_gamut_fault_consumed(0, 1) == 1, "next real call consumes skipped construction fault before cleanup");
    fj_test_gamut_clear_fault();
    std::ifstream tables(tablesPath, std::ios::binary);
    for (const auto& row : fixture["products"]) {
        const int route = row.value("route", 0), polarity = row.value("polarity", 0), method = row["method"];
        auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route), static_cast<Spektrafilm::RgbToRawMethod>(method));
        controls.inputCompressionEnabled = row["compression"];
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        std::optional<PreparedState> product;
        const auto prepared = prepare_state(JuicerProcess::root().assets().borrow_rust_assets(), controls, product, error);
        require(prepared.category == FJ_STATUS_SUCCESS && product, "actual A8 prepared state completes");
        FjPreparedStateView view{};
        require(product->view(view, error).category == FJ_STATUS_SUCCESS, std::string(diagnostic.data(), error.length));
        require(complete_result(*product) == row["expected"], "all native A8 product fields exact");
        require((view.input_hull.xy.count != 0) == row.contains("hull"), "route/method hull ownership");
        if (view.input_hull.xy.count) {
            require(view.input_hull.hash == row["hull"]["hash"] && bits(view.input_hull.center_xy, 2) == row["hull"]["center"], "actual held hull identity");
            for (std::size_t i = 0; i < view.input_hull.xy.count / 2; ++i)
                require(bits(view.input_hull.xy.data + i * 2, 2) == row["hull"]["xy"][i], "actual held hull polygon");
        }
        require((view.film_exposure.tc_lut_rgba.count != 0) == row.contains("table_count"), "actual TC ownership");
        if (view.film_exposure.tc_lut_rgba.count) {
            const auto samples = view.film_exposure.tc_lut_rgba;
            require(samples.count == row["table_count"], "complete TC extent");
            std::vector<char> expected(samples.count * sizeof(float));
            tables.seekg(row["table_offset"].get<std::streamoff>());
            tables.read(expected.data(), static_cast<std::streamsize>(expected.size()));
            require(tables.good() && std::memcmp(expected.data(), samples.data, expected.size()) == 0, "every native TC byte including padding");
        }
    }
}

void fj_test_input_hull_admission(std::uint32_t fault) {
    require(fault >= 1 && fault <= 6, "closed hull admission witness");
    auto controls = snapshot(Spektrafilm::ScanRoute::PositivePrintScan, Spektrafilm::RgbToRawMethod::Arctic2026beta04);
    controls.inputCompressionEnabled = 1;
    auto& library = JuicerProcess::root().assets();
    FjAssets* const cacheOwner = library.borrow_rust_assets();
    struct ClearOwnerFault {
        FjAssets* owner;
        ~ClearOwnerFault() {
            (void)fj_test_assets_input_hull_fault(owner, 0, nullptr);
        }
    } clearOwnerFault{cacheOwner};
    const auto armOwnerFault = [&](std::uint32_t selected) {
        require(fj_test_assets_input_hull_fault(cacheOwner, selected, nullptr).category == FJ_STATUS_SUCCESS, "arm actual input-hull owner");
    };
    const auto consumed = [&] {
        std::uint64_t count = 0;
        require(fj_test_cache_fault_consumed(cacheOwner, FJ_TEST_CACHE_INPUT_HULL, &count, nullptr).category == FJ_STATUS_SUCCESS, "read actual hull fault consumption");
        return count;
    };
    const auto pendingFault = [&] {
        std::uint32_t active = 0;
        require(fj_test_assets_input_hull_fault_pending(cacheOwner, &active, nullptr).category == FJ_STATUS_SUCCESS, "read actual pending hull fault");
        return active;
    };
    InstanceState cold;
    pending(cold, controls);
    armOwnerFault(fault);
    const auto coldFailure = admit_pending_render_state(cold);
    require(coldFailure.status == PendingRenderAdmissionStatus::RebuildFailed && !admitted(coldFailure) && cold.lastHash.load() == 0 && !JuicerAtomic::load_shared_ptr(&cold.activePrintState), "cold hull failure admits and publishes nothing");
    require(consumed() == 1, "cold fault consumed before cleanup/retry");
    InstanceState current;
    auto disabled = controls;
    disabled.inputCompressionEnabled = 0;
    pending(current, disabled);
    const auto old = admit_pending_render_state(current);
    require(admitted(old), "disabled hull initial publication");
    const auto retained = held(old);
    const auto previousHash = current.lastHash.load(), counter = current.buildCounterNext.load();
    pending(current, controls);
    armOwnerFault(fault);
    const auto failed = admit_pending_render_state(current);
    require(failed.status == PendingRenderAdmissionStatus::RebuildFailed && !admitted(failed), "failed current cannot admit old publication");
    require(consumed() == 1, "current fault consumed before cleanup/recovery");
    require(current.lastHash.load() == previousHash && current.buildCounterNext.load() == counter && held(old) == retained, "failed build preserves counters and admitted immutable hold");
    if (fault >= 3)
        require(!JuicerAtomic::load_shared_ptr(&current.activePrintState), "ordinary missing hull clears publication");
    else
        require(JuicerAtomic::load_shared_ptr(&current.activePrintState) == old.printState, "exceptional hull fault preserves publication");
    armOwnerFault(0);
    const auto recovered = admit_pending_render_state(current);
    require(admitted(recovered) && current.lastHash.load() == hash_params(controls) && current.buildCounterNext.load() == counter + 1, "current recovery publishes complete state once");
    require(admitted(admit_pending_render_state(cold)), "cold retry recovers");
    armOwnerFault(1);
    auto changed = controls;
    changed.cameraExposureCompensationEv += .25;
    pending(current, changed);
    require(admitted(admit_pending_render_state(current)) && consumed() == 0 && pendingFault() == 1, "warm Library bypasses hull construction");
    const auto& hull = recovered.printState->recipe.filmRaw.inputCompressionHull;
    const auto warm = library.input_compression_hull();
    require(hull && warm && hull->entryIdentity == warm->entryIdentity && hull->xy.data() == warm->xy.data(), "recipe retains sole cached Rust backing");
    JuicerAssets::Library unrelated(JuicerProcess::data_directory());
    require(unrelated.input_compression_hull() && pendingFault() == 1, "another owner cannot consume the armed warm-owner fault");
    armOwnerFault(0);
    require(pendingFault() == 0, "explicit owner fault cleanup");
    // The standalone raw ABI retains its separate ownerless fault witness.
    require(fj_test_gamut_arm_fault(0, 1).category == FJ_STATUS_SUCCESS, "arm standalone mathematical boundary");
    const auto curves = JuicerProcess::root().assets().illuminant_filter_curves();
    const JuicerTestSupport::CmfSource cmf(JuicerProcess::root().assets().borrow_rust_assets());
    FjInputHullSource source{cmf.x(), cmf.y(), cmf.z(), {curves->d65.linear.data(), 81}};
    FjInputHull out{};
    FjErrorBuffer error{nullptr, 0, 0};
    require(fj_legacy_gamut_input_hull(&source, &out, &error).category == FJ_STATUS_UNSUPPORTED_INPUT && fj_test_gamut_fault_consumed(0, 1) == 1, "standalone mathematical boundary consumes its separate fault");
    fj_test_gamut_clear_fault();
}
