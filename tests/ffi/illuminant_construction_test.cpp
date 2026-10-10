#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "Illuminants.h"
#include "ColorTransforms.h"
#include "RustColorBridge.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RustAssetBridge.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "CudaRenderProjection.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
template <typename T>
concept RvalueCsvView = requires(T&& value) { std::move(value).view(); };
static_assert(!std::is_default_constructible_v<JuicerAssets::CsvRows> && !std::is_copy_constructible_v<JuicerAssets::CsvRows> && std::is_move_constructible_v<JuicerAssets::CsvRows> && !RvalueCsvView<JuicerAssets::CsvRows>);
static_assert(!std::is_default_constructible_v<JuicerIlluminant::Lens> && !std::is_copy_constructible_v<JuicerIlluminant::Lens>);

namespace IlluminantConstruction {
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
    ParamSnapshot snapshot(Spektrafilm::ScanRoute route) {
        ParamSnapshot out;
        out.filmProfileKey = Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Negative ? "kodak_portra_400" : "fujifilm_provia_100f";
        out.printProfileKey = "kodak_portra_endura";
        out.scanRoute = route;
        out.spectralUpsamplingMode = 1;
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
} // namespace IlluminantConstruction

namespace IlluminantConstruction {
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
    namespace fs = std::filesystem;
    fs::path resources, fixtures, scratch;
    void initialize(const fs::path& root, const fs::path& fixtureRoot, const fs::path& artifactRoot) {
        resources = root;
        fixtures = fixtureRoot;
        scratch = artifactRoot;
    }
    Json fixture() {
#if defined(_WIN32)
        const std::string platform = "windows-clang";
#else
        const std::string platform = "linux";
#endif
#if defined(NDEBUG)
        const std::string profile = "release";
#else
        const std::string profile = "debug";
#endif
        std::ifstream file(fixtures / ("construction-" + platform + "-" + profile + ".json"));
        return Json::parse(file);
    }
    struct Diagnostic {
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
    };
    void status(FjStatus actual, std::uint32_t category) {
        EXPECT_EQ(actual.category, category);
        EXPECT_EQ(actual.api, FJ_API_NONE);
        EXPECT_EQ(actual.native_code, 0);
    }
    std::vector<float> scalars(const Json& pairs) {
        std::vector<float> out;
        for (const auto& row : pairs)
            for (const auto& word : row)
                out.push_back(std::bit_cast<float>(word.get<std::uint32_t>()));
        return out;
    }
    FjFloatSpan span(const std::vector<float>& values) {
        return {values.empty() ? nullptr : values.data(), values.size()};
    }
    void samples(const FjIlluminant& actual, const Json& expected) {
        ASSERT_EQ(expected.size(), 81u);
        for (std::size_t i = 0; i < 81; ++i) {
            const auto word = expected[i].get<std::uint32_t>();
            if (std::isnan(std::bit_cast<float>(word)))
                EXPECT_TRUE(std::isnan(actual.samples[i]));
            else
                EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.samples[i]), word) << i;
        }
    }
    bool cleared(const FjIlluminant& curve) {
        return std::all_of(std::begin(curve.samples), std::end(curve.samples), [](float value) {
            return std::bit_cast<std::uint32_t>(value) == 0;
        });
    }
    FjIlluminantLens* lens(const std::vector<float>& rows) {
        Diagnostic d;
        FjIlluminantLens* owner = nullptr;
        FjIlluminantCoverage c{};
        status(fj_legacy_illuminant_lens_prepare(span(rows), &owner, &c, &d.error), FJ_STATUS_SUCCESS);
        return owner;
    }

    TEST(IlluminantConstruction, RawAbiCoreFacadeAndWrappersMatchFrozenLeaves) {
        const auto f = fixture();
        Diagnostic d;
        FjIlluminant curve{}, core{};
        FjIlluminantCoverage c{};
        for (const auto& row : f["leaf"]["blackbody"]) {
            SCOPED_TRACE(row["temperature"].dump());
            const auto temperature = std::bit_cast<float>(row["temperature"].get<std::uint32_t>());
            status(fj_legacy_illuminant_blackbody(temperature, &curve, &d.error), FJ_STATUS_SUCCESS);
            samples(curve, row["normalized"]);
            status(fj_test_illuminant_curve(2, {}, {}, temperature, &core, &c, &d.error), FJ_STATUS_SUCCESS);
            samples(core, row["normalized"]);
            EXPECT_EQ(bits(JuicerIlluminant::blackbody(temperature).linear), row["normalized"]);
        }
        status(fj_legacy_illuminant_equal_energy(&curve, &d.error), FJ_STATUS_SUCCESS);
        samples(curve, f["leaf"]["equal"]["samples"]);
        EXPECT_EQ(bits(JuicerIlluminant::equal_energy().linear), f["leaf"]["equal"]["samples"]);
        for (const auto& row : f["leaf"]["direct"]) {
            SCOPED_TRACE(row["id"].dump());
            const auto input = scalars(row["rows"]);
            const auto expected = row["curve"]["samples"].empty() ? FJ_STATUS_PREPARATION_FAILURE : FJ_STATUS_SUCCESS;
            status(fj_legacy_illuminant_from_samples(span(input), &curve, &d.error), expected);
            status(fj_test_illuminant_curve(1, span(input), {}, 0, &core, &c, &d.error), expected);
            if (expected == FJ_STATUS_SUCCESS) {
                samples(curve, row["curve"]["samples"]);
                samples(core, row["curve"]["samples"]);
            } else {
                EXPECT_TRUE(cleared(curve));
                EXPECT_TRUE(cleared(core));
            }
        }
        for (const auto& row : f["leaf"]["resample"]) {
            SCOPED_TRACE(row["id"].dump());
            const auto input = scalars(row["rows"]);
            const auto expected = row["kg3"]["samples"].empty() ? FJ_STATUS_PREPARATION_FAILURE : FJ_STATUS_SUCCESS;
            status(fj_legacy_illuminant_tungsten_kg3(span(input), &curve, &c, &d.error), expected);
            if (expected == FJ_STATUS_SUCCESS)
                samples(curve, row["kg3"]["samples"]);
            else
                EXPECT_TRUE(cleared(curve));
            FjIlluminantLens* owner = nullptr;
            status(fj_legacy_illuminant_lens_prepare(span(input), &owner, &c, &d.error), row["lens_prepared"].get<bool>() ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
            if (owner) {
                status(fj_legacy_illuminant_lens_finish(&owner, span(input), &curve, &c, &d.error), FJ_STATUS_SUCCESS);
                EXPECT_EQ(owner, nullptr);
                samples(curve, row["lens"]["samples"]);
                status(fj_test_illuminant_curve(5, span(input), span(input), 0, &core, &c, &d.error), FJ_STATUS_SUCCESS);
                samples(core, row["lens"]["samples"]);
            }
            EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);
        }
    }
    TEST(IlluminantConstruction, BorrowedRowsAndMeasuredScratchHaveScopedLifetimes) {
        auto bridge = std::make_unique<JuicerAssets::AssetBridge>(resources);
        {
            auto borrowed = bridge->csv_rows(JuicerAssets::CsvSource::D65);
            status(bridge->release_cached_payloads(), FJ_STATUS_SUCCESS);
            bridge.reset();
            EXPECT_EQ(fj_test_csv_live_owners(), 1u);
            const auto curve = JuicerIlluminant::from_samples(borrowed, "D65 retained borrow");
            EXPECT_EQ(curve.linear.size(), 81u);
        }
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        Diagnostic d;
        FjIlluminant curve{};
        FjIlluminantCoverage c{};
        const auto f = fixture();
        for (const auto& row : f["leaf"]["resample"]) {
            const auto input = scalars(row["rows"]);
            (void)fj_test_illuminant_curve(8, span(input), {}, 0, &curve, &c, &d.error);
            std::array<std::size_t, 10> bytes{};
            const auto count = fj_test_illuminant_scratch_capacities(bytes.data());
            EXPECT_LE(count, 10u);
            std::size_t total = 0;
            for (std::size_t index = 0; index < count; ++index)
                total += bytes[index];
            std::printf("Rust Akima %s: actual Vec capacities=%zu buffers/%zu cumulative bytes; no retained scratch, excludes metadata/RSS\n", row["id"].get<std::string>().c_str(), count, total);
        }
    }

    TEST(IlluminantConstruction, MalformedDescriptorsClearAndConsume) {
        Diagnostic d;
        FjIlluminant curve{};
        FjIlluminantCoverage c{};
        const std::vector<float> input{370, 1, 790, 1};
        const std::array<FjFloatSpan, 5> invalid{{{nullptr, 2}, {input.data(), 1}, {input.data(), std::numeric_limits<std::size_t>::max() - 1}, {reinterpret_cast<const float*>(reinterpret_cast<const char*>(input.data()) + 1), 2}, {reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3), 2}}};
        for (const auto& view : invalid) {
            std::fill(std::begin(curve.samples), std::end(curve.samples), 9);
            status(fj_legacy_illuminant_from_samples(view, &curve, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_TRUE(cleared(curve));
            EXPECT_GT(d.error.length, 0u);
            auto* owner = lens(input);
            status(fj_legacy_illuminant_lens_finish(&owner, view, &curve, &c, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(owner, nullptr);
            EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);
        }
        for (unsigned fault = 0; fault < 4; ++fault) {
            auto* owner = lens(input);
            FjErrorBuffer bad{nullptr, 1, 99};
            FjErrorBuffer* error = fault == 0 ? nullptr : fault == 1 ? &bad
                                                                     : &d.error;
            FjIlluminant* out = fault == 2 ? nullptr : &curve;
            FjIlluminantCoverage* coverage = fault == 3 ? nullptr : &c;
            status(fj_legacy_illuminant_lens_finish(&owner, span(input), out, coverage, error), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(owner, nullptr);
            EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);
        }
        auto* owner = lens(input);
        status(fj_legacy_illuminant_lens_release(&owner, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(owner, nullptr);
        status(fj_legacy_illuminant_lens_release(&owner, &d.error), FJ_STATUS_SUCCESS);
        status(fj_legacy_illuminant_equal_energy(&curve, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_TRUE(cleared(curve));
        status(fj_legacy_illuminant_lens_release(nullptr, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
    }
    TEST(IlluminantConstruction, WarningsFailuresAndRawFaultConsumption) {
        Diagnostic d;
        FjIlluminant curve{};
        FjIlluminantCoverage c{};
        const std::vector<float> input{370, 1, 790, 1}, narrow{380, 1, 780, 1}, single{500, 1};
        FjIlluminantLens* owner = nullptr;
        status(fj_legacy_illuminant_lens_prepare(span(single), &owner, &c, &d.error), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(c.warnings, 3u);
        EXPECT_EQ(c.min_nm, 500);
        EXPECT_EQ(owner, nullptr);
        status(fj_legacy_illuminant_lens_prepare(span(narrow), &owner, &c, &d.error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(c.warnings, 3u);
        status(fj_legacy_illuminant_lens_finish(&owner, span(narrow), &curve, &c, &d.error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(c.warnings, 3u);
        for (std::uint32_t operation = 1; operation <= 7; ++operation)
            for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                SCOPED_TRACE(std::to_string(operation) + "/" + std::to_string(fault));
                owner = operation >= 6 ? lens(input) : nullptr;
                status(fj_test_illuminant_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                FjStatus actual{};
                switch (operation) {
                    case 1:
                        actual = fj_legacy_illuminant_from_samples(span(input), &curve, &d.error);
                        break;
                    case 2:
                        actual = fj_legacy_illuminant_blackbody(3200, &curve, &d.error);
                        break;
                    case 3:
                        actual = fj_legacy_illuminant_equal_energy(&curve, &d.error);
                        break;
                    case 4:
                        actual = fj_legacy_illuminant_tungsten_kg3(span(input), &curve, &c, &d.error);
                        break;
                    case 5:
                        actual = fj_legacy_illuminant_lens_prepare(span(input), &owner, &c, &d.error);
                        break;
                    case 6:
                        actual = fj_legacy_illuminant_lens_finish(&owner, span(input), &curve, &c, &d.error);
                        break;
                    case 7:
                        actual = fj_legacy_illuminant_lens_release(&owner, &d.error);
                        break;
                    default:
                        FAIL();
                }
                const auto category = fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                             : fault == 3   ? FJ_STATUS_ALLOCATION_FAILURE
                                                                                            : FJ_STATUS_PREPARATION_FAILURE;
                status(actual, category);
                if (operation >= 5) {
                    EXPECT_EQ(owner, nullptr);
                }
                // Repeat the intended raw operation before cleanup: no armed fault remains.
                switch (operation) {
                    case 1:
                        status(fj_legacy_illuminant_from_samples(span(input), &curve, &d.error), FJ_STATUS_PREPARATION_FAILURE);
                        break;
                    case 2:
                        status(fj_legacy_illuminant_blackbody(3200, &curve, &d.error), FJ_STATUS_SUCCESS);
                        break;
                    case 3:
                        status(fj_legacy_illuminant_equal_energy(&curve, &d.error), FJ_STATUS_SUCCESS);
                        break;
                    case 4:
                        status(fj_legacy_illuminant_tungsten_kg3(span(input), &curve, &c, &d.error), FJ_STATUS_SUCCESS);
                        break;
                    case 5:
                        status(fj_legacy_illuminant_lens_prepare(span(input), &owner, &c, &d.error), FJ_STATUS_SUCCESS);
                        status(fj_legacy_illuminant_lens_release(&owner, &d.error), FJ_STATUS_SUCCESS);
                        break;
                    case 6:
                        owner = lens(input);
                        status(fj_legacy_illuminant_lens_finish(&owner, span(input), &curve, &c, &d.error), FJ_STATUS_SUCCESS);
                        break;
                    case 7:
                        status(fj_legacy_illuminant_lens_release(&owner, &d.error), FJ_STATUS_SUCCESS);
                        break;
                    default:
                        FAIL();
                }
                EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);
            }
        status(fj_test_illuminant_arm_fault(3, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_test_illuminant_curve(3, {}, {}, 0, &curve, &c, &d.error), FJ_STATUS_SUCCESS);
        status(fj_legacy_illuminant_equal_energy(&curve, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_illuminant_equal_energy(&curve, &d.error), FJ_STATUS_SUCCESS);
        fj_test_illuminant_arm_facade_fault();
        status(fj_legacy_illuminant_equal_energy(&curve, &d.error), FJ_STATUS_SUCCESS);
        status(fj_test_illuminant_curve(3, {}, {}, 0, &curve, &c, &d.error), FJ_STATUS_INTERNAL_FAILURE);
        status(fj_test_illuminant_curve(3, {}, {}, 0, &curve, &c, &d.error), FJ_STATUS_SUCCESS);
        const std::array<float, 3> white{.95047f, 1.0f, 1.08883f};
        std::array<float, 9> matrix{};
        status(fj_test_color_arm_fault(FJ_TEST_CAT16_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
        status(fj_legacy_illuminant_equal_energy(&curve, &d.error), FJ_STATUS_SUCCESS);
        status(fj_legacy_cat16_matrix(white.data(), white.data(), matrix.data()), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_cat16_matrix(white.data(), white.data(), matrix.data()), FJ_STATUS_SUCCESS);
        auto csvAssets = std::make_unique<JuicerAssets::AssetBridge>(resources);
        fj_test_csv_fault(1);
        status(fj_legacy_illuminant_equal_energy(&curve, &d.error), FJ_STATUS_SUCCESS);
        EXPECT_THROW(csvAssets->csv_rows(JuicerAssets::CsvSource::D65), JuicerCuda::ExecutionFailure);
        csvAssets.reset();
        fj_test_illuminant_clear_fault();
    }
    TEST(IlluminantConstruction, PublishedCallerProductsMatchFrozenNativeParent) {
        const auto f = fixture();
        const auto root = scratch / "construction-products";
        fs::remove_all(root);
        fs::copy(resources, root, fs::copy_options::recursive);
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto& assets = JuicerProcess::root().assets();
        const auto read = [](const fs::path& path) {
            std::ifstream file(path);
            return Json::parse(file);
        };
        const auto negative = read(root / "profiles/kodak_portra_400.json"), positive = read(root / "profiles/fujifilm_provia_100f.json"), paper = read(root / "profiles/kodak_portra_endura.json");
        for (const auto& row : f["products"]) {
            SCOPED_TRACE(row["label"].dump() + "/" + row["controls"]["enlIll"].dump());
            const int polarity = row["polarity"], route = row["route"];
            auto controls = snapshot(static_cast<Spektrafilm::ScanRoute>(polarity * 2 + route));
            controls.enlIll = row["controls"]["enlIll"];
            controls.printPreflashExposure = route ? .1 : 0;
            controls.preflashMFilterCc = 12.5;
            controls.preflashYFilterCc = 7.25;
            ASSERT_EQ(controls_json(controls), row["controls"]);
            auto film = polarity ? positive : negative, print = paper;
            film["info"]["reference_illuminant"] = row["profile_reference"];
            film["info"]["viewing_illuminant"] = row["profile_viewing"];
            print["info"]["viewing_illuminant"] = row["print_viewing"];
            {
                std::ofstream file(root / "profiles" / (controls.filmProfileKey + ".json"));
                file << film;
            }
            {
                std::ofstream file(root / "profiles/kodak_portra_endura.json");
                file << print;
            }
            status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            const bool built = route ? build_print_render_state_product(controls, product, diagnostic) : build_direct_render_state_product(controls, product, diagnostic);
            ASSERT_EQ(built, row["built"].get<bool>()) << diagnostic;
            if (!built) {
                EXPECT_FALSE(diagnostic.empty());
                continue;
            }
            auto result = complete_result(product);
            result["exposure_tables"] = tables_result(product.payload.exposureTables);
            result["final_sensitivity_hash"] = product.recipe.filmRaw.finalSensitivityHash;
            for (const auto& rgb : product.recipe.filmRaw.finalSensitivity)
                result["final_sensitivity"].push_back(bits(rgb));
            if (route) {
                ASSERT_TRUE(product.payload.printMainIlluminant);
                result["main_light"] = bits(*product.payload.printMainIlluminant);
                JuicerCuda::PrintResourcePreparation request{&product.recipe, &assets, &*product.payload.printMainIlluminant};
                JuicerCuda::PrintResourceInput input;
                std::array<float, 81> preflash{};
                ASSERT_TRUE(JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic)) << diagnostic;
                result["preflash_light"] = bits(preflash);
                result["preflash_raw"] = bits(input.preflashRawCmy);
                result["print_descriptors"] = {input.descriptors.hash, input.descriptors.mainIlluminant.hash, input.descriptors.preflashIlluminant.hash, input.descriptors.preflashRaw.hash, input.descriptors.balance.hash};
            }
            EXPECT_EQ(result, row["expected"]);
        }
    }

    unsigned nativeSubmissions = 0;
    TEST(IlluminantConstruction, ConstructionAdmissionAndEqualPreflashProjection) {
        const auto root = scratch / "construction-failures";
        fs::remove_all(root);
        fs::copy(resources, root, fs::copy_options::recursive);
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto& assets = JuicerProcess::root().assets();
        {
            const auto path = root / "profiles/kodak_portra_400.json";
            std::ifstream file(path);
            auto film = Json::parse(file);
            film["info"]["reference_illuminant"] = "BB3200.5";
            std::ofstream changed(path);
            changed << film;
        }
        const auto pending = [](InstanceState& state, const ParamSnapshot& p) {
            std::lock_guard<std::mutex> lock(state.pending.m);
            state.pending.value = PendingParamsState::Valid{p, hash_params(p)};
        };
        for (std::uint32_t operation : {1u, 2u, 3u, 4u, 5u, 6u})
            for (std::uint32_t fault : {1u, 2u, 3u}) {
                auto controls = snapshot(operation == 3 ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan);
                controls.enlIll = 7;
                InstanceState cold;
                pending(cold, controls);
                status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
                status(fj_test_illuminant_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                if (fault == 3) {
                    try {
                        (void)admit_pending_render_state(cold);
                        FAIL() << "cold allocation admitted";
                    } catch (const JuicerCuda::ExecutionFailure& error) {
                        status(error.failure.status, FJ_STATUS_ALLOCATION_FAILURE);
                    }
                } else {
                    EXPECT_EQ(admit_pending_render_state(cold).status, PendingRenderAdmissionStatus::RebuildFailed);
                }
                EXPECT_FALSE(JuicerAtomic::load_shared_ptr(&cold.activeDirectState));
                EXPECT_FALSE(JuicerAtomic::load_shared_ptr(&cold.activePrintState));
                EXPECT_EQ(cold.lastHash.load(), 0u);
                EXPECT_EQ(cold.buildCounterNext.load(), 0u);
                const auto coldRecovery = admit_pending_render_state(cold);
                ASSERT_TRUE(coldRecovery.directState || coldRecovery.printState);
                InstanceState state;
                pending(state, controls);
                const auto old = admit_pending_render_state(state);
                ASSERT_TRUE(old.directState || old.printState);
                const auto hash = state.lastHash.load(), counter = state.buildCounterNext.load();
                auto changed = controls;
                changed.cameraExposureCompensationEv = 1.25;
                pending(state, changed);
                status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
                status(fj_test_illuminant_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                if (fault == 3) {
                    try {
                        (void)admit_pending_render_state(state);
                        FAIL() << "allocation failure admitted";
                    } catch (const JuicerCuda::ExecutionFailure& error) {
                        status(error.failure.status, FJ_STATUS_ALLOCATION_FAILURE);
                    }
                } else {
                    const auto failure = admit_pending_render_state(state);
                    EXPECT_EQ(failure.status, PendingRenderAdmissionStatus::RebuildFailed);
                    EXPECT_EQ(failure.directState, nullptr);
                }
                EXPECT_EQ(JuicerAtomic::load_shared_ptr(&state.activeDirectState), old.directState);
                EXPECT_EQ(JuicerAtomic::load_shared_ptr(&state.activePrintState), old.printState);
                EXPECT_EQ(state.lastHash.load(), hash);
                EXPECT_EQ(state.buildCounterNext.load(), counter);
                EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);
                EXPECT_EQ(fj_test_csv_live_owners(), 0u);
                const auto recovered = admit_pending_render_state(state);
                ASSERT_TRUE(recovered.directState || recovered.printState);
                EXPECT_TRUE(recovered.directState != old.directState || recovered.printState != old.printState);
            }
        auto controls = snapshot(Spektrafilm::ScanRoute::NegativePrintScan);
        controls.enlIll = 7;
        controls.printPreflashExposure = .1;
        controls.grainControls.active = false;
        controls.dirCouplers.active = false;
        InstanceState published;
        pending(published, controls);
        const auto admitted = admit_pending_render_state(published);
        ASSERT_TRUE(admitted.printState);
        const auto& product = *admitted.printState;
        const auto retainedHash = product.recipe.hash;
        const auto retainedCounter = published.buildCounterNext.load();
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
        std::optional<ScatterHalationFrameDescriptor> halation;
        Spektrafilm::FilmJuicerEffectsGeometry geometry{};
        JuicerCuda::AutoExposurePreviewDescriptor meter{};
        JuicerCuda::ExecutionFrame frame{{0, 0, 16, 16}, {0, 0, 16, 16}, {0, 0, 16, 16}, nullptr, nullptr, nullptr, 0, 0, 4, nullptr, diffusion, halation, geometry, 10.0f, 0, 24, 1, 0, meter, false, false};
        FjFrame raw{};
        JuicerCuda::ResourceManager::SubmissionSnapshot submission{};
        for (std::uint32_t fault : {1u, 2u, 3u, 4u}) {
            Diagnostic d;
            FjIlluminant value{};
            JuicerCuda::PendingContextLossRecovery recovery;
            std::string diagnostic;
            nativeSubmissions = 0;
            status(fj_test_illuminant_arm_fault(3, 1, fault), FJ_STATUS_SUCCESS);
            {
                JuicerCuda::NativeCall call(JuicerCuda::borrowed_owner());
                const auto result = JuicerCuda::project_and_render(call, product.recipe, product.payload, frame, raw, submission, recovery, {}, nullptr, {}, diagnostic);
                status(result.status, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                             : fault == 3   ? FJ_STATUS_ALLOCATION_FAILURE
                                                                                            : FJ_STATUS_PREPARATION_FAILURE);
            }
            EXPECT_EQ(nativeSubmissions, 0u);
            EXPECT_EQ(product.recipe.hash, retainedHash);
            EXPECT_EQ(JuicerAtomic::load_shared_ptr(&published.activePrintState), admitted.printState);
            EXPECT_EQ(published.buildCounterNext.load(), retainedCounter);
            status(fj_legacy_illuminant_equal_energy(&value, &d.error), FJ_STATUS_SUCCESS);
        }
    }

} // namespace IlluminantConstruction

namespace JuicerAssets::IlluminantTest {
    void before_native_render() noexcept {
        ++IlluminantConstruction::nativeSubmissions;
    }
} // namespace JuicerAssets::IlluminantTest
