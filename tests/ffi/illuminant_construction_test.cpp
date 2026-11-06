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
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "ProcessRuntime.h"
#include "gtest_cases.h"
#include "ColorTransforms.h"
#include "RustColorBridge.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "test_asset_workspace.h"
#include "test_asset_bridge.h"
#include "Cuda/JuicerCudaResources.h"
#include "prepared_boundary.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "exposure_fixture.h"
#include "print_illuminant_fixture.h"
#include "prepared_product_fixture.h"
namespace IlluminantConstruction {
    using Json = nlohmann::json;
    void require(bool condition, const std::string& diagnostic) {
        if (!condition)
            throw std::runtime_error(diagnostic);
    }
    void require_poisoned(const FjAssets* assets, RenderInstance& state, const std::string& context) {
        FjCaptureStart capture{};
        FjErrorBuffer error{nullptr, 0, 0};
        const auto captureStatus = state.begin_capture(FJ_CAPTURE_CHANGE, capture, error);
        require(captureStatus.category == FJ_STATUS_INTERNAL_FAILURE && capture.kind == FJ_CAPTURE_START_NONE && capture.capture == 0,
                context + " rejects future capture on terminal instance");
        const auto admission = JuicerTestSupport::admit(assets, state);
        require(admission.delivery.category == FJ_STATUS_INTERNAL_FAILURE && admission.kind == FJ_ADMISSION_POISONED && !admission.state,
                context + " rejects future admission on terminal instance");
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
    Json color_result(const FjScannerColor& color) {
        return {{"cat02", bits(color.cat02)}, {"xyz_to_rgb", bits(color.xyz_to_rgb)}, {"illuminant_xyz", bits(color.illuminant_xyz)}, {"encoding", {static_cast<int>(color.output_space), bool(color.flags & FJ_COLOR_ENCODE_CCTF), bool(color.flags & FJ_COLOR_INPUT_IS_OUTPUT_SPACE)}}, {"gamut_recipe_hash", color.active_gamut_recipe_hash}, {"hash", color.hash}};
    }
    Json span_bits(FjFloatSpan span) {
        Json out = Json::array();
        for (std::size_t i = 0; i < span.count; ++i)
            out.push_back(std::bit_cast<std::uint32_t>(span.data[i]));
        return out;
    }
    Json tables_result(const FjPreparedSpectralView& tables) {
        require(tables.lambda_nm.count == 81 && tables.illuminant.count == 81, "prepared spectral axis and illuminant extents");
        for (unsigned i = 0; i < 3; ++i)
            require(tables.observer_xyz[i].count == 81 && tables.weighted_xyz[i].count == 81 && tables.dyes_cmy[i].count == 81, "prepared spectral lane extents");
        return {{"lambda", span_bits(tables.lambda_nm)}, {"K", 81}, {"delta_lambda", bits(std::array{5.0f})}, {"inv_yn", bits(std::array{tables.inv_yn})}, {"white_xyz", bits(tables.white_xyz)}, {"ref_illum_white_xyz", bits(tables.white_xyz)}, {"ax", span_bits(tables.weighted_xyz[0])}, {"ay", span_bits(tables.weighted_xyz[1])}, {"az", span_bits(tables.weighted_xyz[2])}, {"xbar", span_bits(tables.observer_xyz[0])}, {"ybar", span_bits(tables.observer_xyz[1])}, {"zbar", span_bits(tables.observer_xyz[2])}, {"illum", span_bits(tables.illuminant)}, {"eps_c", span_bits(tables.dyes_cmy[0])}, {"eps_m", span_bits(tables.dyes_cmy[1])}, {"eps_y", span_bits(tables.dyes_cmy[2])}, {"base_density_min", span_bits(tables.baseline_min)}, {"base_density_mid", span_bits(tables.baseline_mid)}, {"has_baseline", bool(tables.has_baseline)}, {"baseline_mix", bits(std::array{0.0f})}, {"illuminant_hash", tables.illuminant_hash}, {"tables_hash", tables.hash}};
    }
    Json scanner_descriptor(const FjScannerLutDescriptor& out) {
        return {{"route", static_cast<int>(out.route)}, {"medium", static_cast<int>(out.medium)}, {"polarity", static_cast<int>(out.capture_polarity)}, {"density_bounds_hash", out.density_bounds_hash}, {"channel_density_hash", out.channel_density_hash}, {"base_density_hash", out.base_density_hash}, {"illuminant_hash", out.viewing_illuminant_hash}, {"observer_hash", out.observer_hash}, {"resolution", out.resolution}, {"hash", out.hash}};
    }
    Json complete_result(const FjPreparedStateView& state, const ParamSnapshot& controls) {
        const auto& raw = state.film_exposure;
        const auto& policy = state.input_policy;
        const auto& reference = state.film_reference;
        const auto& balance = state.print_balance;
        const auto& output = state.output_color;
        const bool decode = (raw.flags & FJ_FILM_DECODE_CCTF) != 0;
        const bool adapt = (raw.flags & FJ_FILM_ADAPT_INPUT) != 0;
        Json sensitivity = Json::array();
        require(raw.sensitivity_rgb.count == std::size_t{81} * 3, "prepared sensitivity shape");
        for (std::size_t row = 0; row < 81; ++row)
            sensitivity.push_back(span_bits({raw.sensitivity_rgb.data + row * 3, 3}));
        return {{"input_rgb_to_xyz", bits(raw.input_rgb_to_xyz)}, {"input_adapt", bits(raw.input_xyz_adapt)}, {"xyz_to_linear_srgb", bits(raw.xyz_to_linear_srgb)}, {"source_white", bits(policy.nominal_white_xyz)}, {"projection_white", bits(policy.projection_white_xyz)}, {"config_rgb_to_xyz", bits(raw.input_rgb_to_xyz)}, {"config_adapt", bits(raw.input_xyz_adapt)}, {"raw_midgray", bits(state.raw_midgray)}, {"midgray_dwg", bits(state.midgray_dwg_rgb)}, {"midgray_scale_green", bits(std::array{state.midgray_scale, state.raw_green})}, {"config_whites", {bits(policy.nominal_white_xyz), bits(policy.projection_white_xyz), bits(policy.projection_white_xyz)}}, {"config_flags", {static_cast<int>(raw.input_color_space), decode, static_cast<int>(controls.spectralUpsamplingMode == 1), adapt, true, true}}, {"spd_s_inv", bits(state.s_inverse)}, {"baseline_raw", bits(reference.baseline_raw_rgb)}, {"compensated_raw", bits(reference.compensated_raw_rgb)}, {"baseline_density", bits(reference.baseline_density_cmy)}, {"compensated_density", bits(reference.compensated_density_cmy)}, {"print_baseline_raw", bits(balance.baseline_raw_rgb)}, {"print_compensated_raw", bits(balance.compensated_raw_rgb)}, {"print_factors", bits(JuicerTestSupport::captured_print_factors(state))}, {"hashes", {raw.hash, raw.tc_lut_hash, reference.hash, balance.hash, state.print_recipe_hash, state.scanner_recipe_hash, output.gamut_recipe_hash, state.recipe_hash, state.upload_core_hash, state.scanner_hash}}, {"scanner_color", color_result(state.scanner_color)}, {"scanner_tables", tables_result(state.scanner_tables)}, {"scanner_descriptor", scanner_descriptor(state.scanner_lut)}, {"gamut_entry", bits(output.native_rgb_to_d65_xyz)}, {"gamut_exit", bits(output.d65_xyz_to_native_rgb)}, {"final_sensitivity_hash", raw.sensitivity_hash}, {"final_sensitivity", sensitivity}};
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
    const Json& fixture() {
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
        static const auto path = fixtures / ("construction-" + platform + "-" + profile + ".json");
        static const Json decoded = PrintIlluminantFixtures::apply(path, load_shared_fixture(path, "construction-manifest.json"));
        return decoded;
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

    TEST(IlluminantConstruction, RawAbiAndCoreFacadeMatchFrozenLeaves) {
        const auto& f = fixture();
        Diagnostic d;
        FjIlluminant curve{}, core{};
        FjIlluminantCoverage c{};
        std::vector<std::string> blackbody_ids;
        for (const auto& expected : f["leaf"]["blackbody"])
            blackbody_ids.push_back(std::to_string(expected.at("temperature").get<std::uint32_t>()));
        JuicerTests::GtestCohort blackbody_cases("leaves-blackbody", blackbody_ids);
        for (const auto& row : f["leaf"]["blackbody"]) {
            auto blackbody_case = blackbody_cases.case_event(std::to_string(row.at("temperature").get<std::uint32_t>()));
            SCOPED_TRACE(row["temperature"].dump());
            const auto temperature = std::bit_cast<float>(row["temperature"].get<std::uint32_t>());
            status(fj_legacy_illuminant_blackbody(temperature, &curve, &d.error), FJ_STATUS_SUCCESS);
            samples(curve, row["normalized"]);
            status(fj_test_illuminant_curve(2, {}, {}, temperature, &core, &c, &d.error), FJ_STATUS_SUCCESS);
            samples(core, row["normalized"]);

            blackbody_case.passed();
        }
        status(fj_legacy_illuminant_equal_energy(&curve, &d.error), FJ_STATUS_SUCCESS);
        samples(curve, f["leaf"]["equal"]["samples"]);
        std::vector<std::string> direct_ids;
        for (const auto& expected : f["leaf"]["direct"])
            direct_ids.push_back(expected.at("id").get<std::string>());
        JuicerTests::GtestCohort direct_curve_cases("leaves-direct", direct_ids);
        for (const auto& row : f["leaf"]["direct"]) {
            auto direct_curve_case = direct_curve_cases.case_event(row.at("id").get<std::string>());
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

            direct_curve_case.passed();
        }
        std::vector<std::string> resample_ids;
        for (const auto& expected : f["leaf"]["resample"])
            resample_ids.push_back(expected.at("id").get<std::string>());
        JuicerTests::GtestCohort resample_curve_cases("leaves-resample", resample_ids);
        for (const auto& row : f["leaf"]["resample"]) {
            auto resample_curve_case = resample_curve_cases.case_event(row.at("id").get<std::string>());
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

            resample_curve_case.passed();
        }
    }
    TEST(IlluminantConstruction, BorrowedRowsAndMeasuredScratchHaveScopedLifetimes) {
        auto bridge = std::make_unique<JuicerAssets::AssetBridge>(resources);
        Diagnostic d;
        {
            FjCsvPairs* acquired = nullptr;
            status(fj_legacy_csv_acquire(bridge->borrow_assets(), FJ_CSV_D65, &acquired, &d.error), FJ_STATUS_SUCCESS);
            const auto releaseCsv = [](FjCsvPairs* held) {
                (void)fj_legacy_csv_release(held, nullptr);
            };
            std::unique_ptr<FjCsvPairs, decltype(releaseCsv)> borrowed(acquired, releaseCsv);
            FjFloatSpan rows{};
            status(fj_legacy_csv_view(borrowed.get(), &rows, &d.error), FJ_STATUS_SUCCESS);
            status(bridge->release_cached_payloads(), FJ_STATUS_SUCCESS);
            bridge.reset();
            EXPECT_EQ(fj_test_csv_live_owners(), 1u);
            FjIlluminant retainedCurve{};
            status(fj_legacy_illuminant_from_samples(rows, &retainedCurve, &d.error), FJ_STATUS_SUCCESS);
            EXPECT_EQ(rows.count, 162u);
        }
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        FjIlluminant curve{};
        FjIlluminantCoverage c{};
        const auto& f = fixture();
        std::vector<std::string> borrowed_resample_ids;
        for (const auto& row : f["leaf"]["resample"])
            borrowed_resample_ids.push_back(row.at("id").get<std::string>());
        JuicerTests::GtestCohort borrowed_resample_cases("borrowed-resample", borrowed_resample_ids);
        for (const auto& row : f["leaf"]["resample"]) {
            auto borrowed_resample_case = borrowed_resample_cases.case_event(row.at("id").get<std::string>());
            const auto input = scalars(row["rows"]);
            (void)fj_test_illuminant_curve(8, span(input), {}, 0, &curve, &c, &d.error);
            std::array<std::size_t, 10> bytes{};
            const auto count = fj_test_illuminant_scratch_capacities(bytes.data());
            EXPECT_LE(count, 10u);
            std::size_t total = 0;
            for (std::size_t index = 0; index < count; ++index)
                total += bytes[index];
            std::printf("Rust Akima %s: actual Vec capacities=%zu buffers/%zu cumulative bytes; no retained scratch, excludes metadata/RSS\n", row["id"].get<std::string>().c_str(), count, total);

            borrowed_resample_case.passed();
        }
    }

    TEST(IlluminantConstruction, MalformedDescriptorsClearAndConsume) {
        Diagnostic d;
        FjIlluminant curve{};
        FjIlluminantCoverage c{};
        const std::vector<float> input{370, 1, 790, 1};
        const std::array<FjFloatSpan, 5> invalid{{{nullptr, 2}, {input.data(), 1}, {input.data(), std::numeric_limits<std::size_t>::max() - 1}, {reinterpret_cast<const float*>(reinterpret_cast<const char*>(input.data()) + 1), 2}, {reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3), 2}}};
        JuicerTests::GtestCohort malformed_span_cases("malformed-spans", std::size(invalid));
        std::size_t malformed_span_index = 0;
        for (const auto& view : invalid) {
            auto malformed_span_case = malformed_span_cases.case_event(std::to_string(malformed_span_index++));
            std::fill(std::begin(curve.samples), std::end(curve.samples), 9);
            status(fj_legacy_illuminant_from_samples(view, &curve, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_TRUE(cleared(curve));
            EXPECT_GT(d.error.length, 0u);
            auto* owner = lens(input);
            status(fj_legacy_illuminant_lens_finish(&owner, view, &curve, &c, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(owner, nullptr);
            EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);

            malformed_span_case.passed();
        }
        JuicerTests::GtestCohort construction_fault_cases("construction-faults", 4u);
        std::size_t construction_fault_index = 0;
        for (unsigned fault = 0; fault < 4; ++fault) {
            auto construction_fault_case = construction_fault_cases.case_event(std::to_string(construction_fault_index++));
            auto* owner = lens(input);
            FjErrorBuffer bad{nullptr, 1, 99};
            FjErrorBuffer* error = fault == 0 ? nullptr : fault == 1 ? &bad
                                                                     : &d.error;
            FjIlluminant* out = fault == 2 ? nullptr : &curve;
            FjIlluminantCoverage* coverage = fault == 3 ? nullptr : &c;
            status(fj_legacy_illuminant_lens_finish(&owner, span(input), out, coverage, error), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(owner, nullptr);
            EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);

            construction_fault_case.passed();
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
        JuicerTests::GtestCohort operation_fault_cases("operation-faults", 28u);
        std::size_t operation_fault_index = 0;
        for (std::uint32_t operation = 1; operation <= 7; ++operation)
            for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                auto operation_fault_case = operation_fault_cases.case_event(std::to_string(operation_fault_index++));
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

                operation_fault_case.passed();
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
        FjCsvPairs* rejected = reinterpret_cast<FjCsvPairs*>(std::uintptr_t{1});
        status(fj_legacy_csv_acquire(csvAssets->borrow_assets(), FJ_CSV_D65, &rejected, &d.error), FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_EQ(rejected, nullptr);
        csvAssets.reset();
        fj_test_illuminant_clear_fault();
    }
    TEST(IlluminantConstruction, PublishedCallerProductsMatchFrozenNativeParent) {
        const auto& f = fixture();
        const auto root = JuicerTestSupport::test_asset_workspaces.create(scratch / "construction-products");
        JuicerTestSupport::copy_positive_print_preparation_assets(resources, root);
        JuicerProcess::ProcessRuntime owner;
        owner.create(root);
        const FjAssets* assets = owner.borrow().assets;
        const auto read = [](const fs::path& path) {
            std::ifstream file(path);
            return Json::parse(file);
        };
        const auto negative = read(root / "profiles/kodak_portra_400.json"), positive = read(root / "profiles/fujifilm_provia_100f.json"), paper = read(root / "profiles/kodak_portra_endura.json");
        std::vector<std::string> product_ids;
        product_ids.reserve(f["products"].size());
        for (std::size_t i = 0; i < f["products"].size(); ++i)
            product_ids.push_back(std::to_string(i) + "/" + f["products"][i].at("label").get<std::string>());
        JuicerTests::GtestCohort product_cases("products", product_ids);
        std::size_t product_index = 0;
        for (const auto& row : f["products"]) {
            auto product_case = product_cases.case_event(product_ids[product_index++]);
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
            status(fj_legacy_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
            std::optional<PreparedState> product;
            std::array<char, 2048> productText{};
            FjErrorBuffer productError{productText.data(), productText.size(), 0};
            const auto productStatus = prepare_state(assets, controls, product, productError);
            const bool built = productStatus.category == FJ_STATUS_SUCCESS;
            ASSERT_EQ(built, row["built"].get<bool>()) << std::string(productText.data(), productError.length);
            FjStatus preparedStatus{};
            std::string preparedDiagnostic;
            auto preparedState = JuicerCudaTest::prepare_state(assets, controls, preparedStatus, preparedDiagnostic);
            ASSERT_EQ(bool(preparedState), built) << preparedDiagnostic;
            if (!built) {
                EXPECT_FALSE(std::string(productText.data(), productError.length).empty());
                EXPECT_NE(preparedStatus.category, FJ_STATUS_SUCCESS);
                EXPECT_FALSE(preparedDiagnostic.empty());
                product_case.passed();
                continue;
            }
            status(preparedStatus, FJ_STATUS_SUCCESS);
            FjPreparedStateView productView{};
            status(product->view(productView, productError), FJ_STATUS_SUCCESS);
            FjPreparedStateView prepared{};
            productError.length = 0;
            productText[0] = 0;
            status(preparedState->view(prepared, productError), FJ_STATUS_SUCCESS);
            EXPECT_EQ(prepared.recipe_hash, productView.recipe_hash);
            EXPECT_EQ(prepared.film_reference.hash, productView.film_reference.hash);
            ASSERT_EQ(prepared.exposure_tables.illuminant.count, productView.exposure_tables.illuminant.count);
            EXPECT_EQ(std::memcmp(prepared.exposure_tables.illuminant.data, productView.exposure_tables.illuminant.data, prepared.exposure_tables.illuminant.count * sizeof(float)), 0);
            ASSERT_EQ(prepared.scanner_tables.illuminant.count, productView.scanner_tables.illuminant.count);
            EXPECT_EQ(std::memcmp(prepared.scanner_tables.illuminant.data, productView.scanner_tables.illuminant.data, prepared.scanner_tables.illuminant.count * sizeof(float)), 0);
            ASSERT_EQ(prepared.film_exposure.tc_lut_rgba.count, productView.film_exposure.tc_lut_rgba.count);
            EXPECT_EQ(std::memcmp(prepared.film_exposure.tc_lut_rgba.data, productView.film_exposure.tc_lut_rgba.data, prepared.film_exposure.tc_lut_rgba.count * sizeof(float)), 0);
            EXPECT_EQ(prepared.input_hull.hash, productView.input_hull.hash);
            auto result = complete_result(productView, controls);
            result["exposure_tables"] = tables_result(productView.exposure_tables);
            if (route) {
                ASSERT_EQ(productView.print.main_illuminant.count, 81u);
                ASSERT_EQ(productView.print.preflash_illuminant.count, 81u);
                result["main_light"] = span_bits(productView.print.main_illuminant);
                FjPrintPreflashInput input{productView.print.film_base_density, productView.print.preflash_illuminant, productView.print.sensitivity_cmy};
                FjPrintPreflashRaw preflash{};
                FjPrintPreflashFailure failure{};
                ASSERT_EQ(fj_test_print_preflash_prepare(&input, &preflash, &failure, nullptr).category, FJ_STATUS_SUCCESS);
                result["preflash_light"] = span_bits(productView.print.preflash_illuminant);
                result["preflash_raw"] = bits(preflash.cmy);
                result["print_descriptors"] = {productView.print.hash, productView.print.main_illuminant_hash, productView.print.preflash_illuminant_hash, productView.print.preflash_raw_hash, productView.print.balance_hash};
                EXPECT_EQ(prepared.print.hash, productView.print.hash);
                EXPECT_EQ(prepared.print_balance.hash, productView.print_balance.hash);
                ASSERT_EQ(prepared.print.main_illuminant.count, productView.print.main_illuminant.count);
                EXPECT_EQ(std::memcmp(prepared.print.main_illuminant.data, productView.print.main_illuminant.data, prepared.print.main_illuminant.count * sizeof(float)), 0);
                ASSERT_EQ(prepared.print.preflash_illuminant.count, productView.print.preflash_illuminant.count);
                EXPECT_EQ(std::memcmp(prepared.print.preflash_illuminant.data, productView.print.preflash_illuminant.data, prepared.print.preflash_illuminant.count * sizeof(float)), 0);
            }
            const bool matches = result == row["expected"];
            EXPECT_TRUE(matches) << "first complete-product difference: " << (matches ? std::string{} : Json::diff(row["expected"], result).front().dump());

            product_case.passed();
        }
    }

    TEST(IlluminantConstruction, ConstructionAdmissionAndEqualPreflashPublication) {
        const auto root = JuicerTestSupport::test_asset_workspaces.create(scratch / "construction-failures");
        JuicerTestSupport::copy_positive_print_preparation_assets(resources, root);
        JuicerProcess::ProcessRuntime owner;
        owner.create(root);
        const FjAssets* assets = owner.borrow().assets;
        {
            const auto path = root / "profiles/kodak_portra_400.json";
            std::ifstream file(path);
            auto film = Json::parse(file);
            film["info"]["reference_illuminant"] = "BB3200.5";
            std::ofstream changed(path);
            changed << film;
        }
        JuicerTests::GtestCohort admission_fault_cases("construction-operation-faults", 18u);
        std::size_t admission_fault_index = 0;
        for (std::uint32_t operation : {1u, 2u, 3u, 4u, 5u, 6u})
            for (std::uint32_t fault : {1u, 2u, 3u}) {
                auto admission_fault_case = admission_fault_cases.case_event(std::to_string(admission_fault_index++));
                auto controls = snapshot(operation == 3 ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan);
                controls.enlIll = 7;
                RenderInstance cold;
                JuicerTestSupport::capture(cold, controls);
                status(fj_legacy_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
                if (operation == 2 || operation == 3)
                    status(fj_test_illuminant_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                else
                    status(fj_test_assets_illuminant_operation_fault(assets, operation, fault, nullptr), FJ_STATUS_SUCCESS);
                if (fault == 3) {
                    bool allocationFailed = false;
                    try {
                        (void)JuicerTestSupport::admit(assets, cold);
                    } catch (const std::bad_alloc&) {
                        allocationFailed = true;
                    }
                    ASSERT_TRUE(allocationFailed) << "cold allocation admitted";
                } else {
                    const auto coldFailure = JuicerTestSupport::admit(assets, cold);
                    if (fault == 2) {
                        EXPECT_EQ(coldFailure.delivery.category, FJ_STATUS_INTERNAL_FAILURE);
                        EXPECT_EQ(coldFailure.kind, FJ_ADMISSION_POISONED);
                        require_poisoned(assets, cold, "cold illuminant panic");
                    } else {
                        EXPECT_EQ(coldFailure.kind, FJ_ADMISSION_PREPARATION_FAILED);
                    }
                }
                if (operation != 2 && operation != 3) {
                    std::uint32_t consumed = 0;
                    status(fj_test_assets_illuminant_operation_consumed(assets, &consumed, nullptr), FJ_STATUS_SUCCESS);
                    EXPECT_EQ(consumed, 1u);
                }
                EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(cold).build, 0u);
                RenderInstance coldRetry;
                if (fault == 2)
                    JuicerTestSupport::capture(coldRetry, controls);
                const auto coldRecovery = fault == 2 ? JuicerTestSupport::admit(assets, coldRetry) : JuicerTestSupport::admit(assets, cold);
                ASSERT_EQ(coldRecovery.kind, FJ_ADMISSION_READY);
                RenderInstance state;
                JuicerTestSupport::capture(state, controls);
                const auto old = JuicerTestSupport::admit(assets, state);
                ASSERT_EQ(old.kind, FJ_ADMISSION_READY);
                FjPreparedStateView oldView{};
                std::array<char, 1024> oldViewText{};
                FjErrorBuffer oldViewError{oldViewText.data(), oldViewText.size(), 0};
                ASSERT_EQ(old.state->view(oldView, oldViewError).category, FJ_STATUS_SUCCESS);
                const auto oldRecipeHash = oldView.recipe_hash;
                const auto previous = JuicerTestSupport::PublicationObserver::metrics(state);
                auto changed = controls;
                changed.cameraExposureCompensationEv = 1.25;
                JuicerTestSupport::capture(state, changed);
                status(fj_legacy_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
                if (operation == 2 || operation == 3)
                    status(fj_test_illuminant_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                else
                    status(fj_test_assets_illuminant_operation_fault(assets, operation, fault, nullptr), FJ_STATUS_SUCCESS);
                if (fault == 3) {
                    bool allocationFailed = false;
                    try {
                        (void)JuicerTestSupport::admit(assets, state);
                    } catch (const std::bad_alloc&) {
                        allocationFailed = true;
                    }
                    ASSERT_TRUE(allocationFailed) << "allocation failure admitted";
                } else {
                    const auto failure = JuicerTestSupport::admit(assets, state);
                    if (fault == 2) {
                        EXPECT_EQ(failure.delivery.category, FJ_STATUS_INTERNAL_FAILURE);
                        EXPECT_EQ(failure.kind, FJ_ADMISSION_POISONED);
                    } else {
                        EXPECT_EQ(failure.kind, FJ_ADMISSION_PREPARATION_FAILED);
                    }
                    EXPECT_FALSE(failure.state);
                }
                const auto afterFailure = JuicerTestSupport::PublicationObserver::metrics(state);
                EXPECT_EQ(afterFailure.build, previous.build);
                EXPECT_EQ(afterFailure.publishedIdentity, previous.publishedIdentity);
                EXPECT_EQ(afterFailure.backingIdentity, previous.backingIdentity);
                FjPreparedStateView retainedView{};
                ASSERT_EQ(old.state->view(retainedView, oldViewError).category, FJ_STATUS_SUCCESS);
                EXPECT_EQ(retainedView.recipe_hash, oldRecipeHash);
                EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);
                EXPECT_EQ(fj_test_csv_live_owners(), 0u);
                if (operation != 2 && operation != 3) {
                    std::uint32_t consumed = 0;
                    status(fj_test_assets_illuminant_operation_consumed(assets, &consumed, nullptr), FJ_STATUS_SUCCESS);
                    EXPECT_EQ(consumed, 1u);
                }
                if (fault == 2) {
                    require_poisoned(assets, state, "current illuminant panic");
                    RenderInstance retry;
                    JuicerTestSupport::capture(retry, changed);
                    const auto recovered = JuicerTestSupport::admit(assets, retry);
                    ASSERT_EQ(recovered.kind, FJ_ADMISSION_READY);
                    EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(retry).build, 1u);
                } else {
                    JuicerTestSupport::capture(state, controls);
                    const auto reused = JuicerTestSupport::admit(assets, state);
                    ASSERT_EQ(reused.kind, FJ_ADMISSION_READY);
                    ASSERT_TRUE(reused.state);
                    EXPECT_EQ(JuicerTestSupport::PublicationObserver::backing(*reused.state), JuicerTestSupport::PublicationObserver::backing(*old.state));
                    EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(state).build, previous.build);
                    JuicerTestSupport::capture(state, changed);
                    const auto recovered = JuicerTestSupport::admit(assets, state);
                    ASSERT_EQ(recovered.kind, FJ_ADMISSION_READY);
                    EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(state).build, previous.build + 1);
                }

                admission_fault_case.passed();
            }
        auto controls = snapshot(Spektrafilm::ScanRoute::NegativePrintScan);
        controls.enlIll = 7;
        controls.printPreflashExposure = .1;
        controls.grainControls.active = false;
        controls.dirCouplers.active = false;
        auto heldRustState = JuicerTestSupport::prepare_product(assets, controls);
        FjPreparedStateView heldRustView{};
        std::array<char, 2048> heldRustText{};
        FjErrorBuffer heldRustError{heldRustText.data(), heldRustText.size(), 0};
        status(heldRustState.view(heldRustView, heldRustError), FJ_STATUS_SUCCESS);
        const auto heldRustHash = heldRustView.recipe_hash;
        const std::array<std::uint32_t, 4> faultCategories{
            FJ_STATUS_UNSUPPORTED_INPUT,
            FJ_STATUS_INTERNAL_FAILURE,
            FJ_STATUS_ALLOCATION_FAILURE,
            FJ_STATUS_PREPARATION_FAILURE};
        const std::array<std::string, 4> faultDiagnostics{
            "illuminant injected unsupported input",
            "asset boundary panic",
            "illuminant scratch capacity",
            "illuminant CSV axis mismatch"};
        JuicerTests::GtestCohort publication_fault_cases("publication-faults", 4u);
        std::size_t publication_fault_index = 0;
        for (std::uint32_t fault : {1u, 2u, 3u, 4u}) {
            auto publication_fault_case = publication_fault_cases.case_event(std::to_string(publication_fault_index++));
            status(fj_test_illuminant_arm_fault(3, 1, fault), FJ_STATUS_SUCCESS);
            std::optional<PreparedState> failed;
            std::array<char, 2048> failureText{};
            FjErrorBuffer failureError{failureText.data(), failureText.size(), 0};
            const auto failure = prepare_state(assets, controls, failed, failureError);
            const auto faultIndex = static_cast<std::size_t>(fault - 1);
            status(failure, faultCategories[faultIndex]);
            EXPECT_FALSE(failed);
            EXPECT_EQ(std::string(failureText.data(), failureError.length), faultDiagnostics[faultIndex]);
            // This independent factory call preserves the raw category and
            // diagnostic contract, separate from the active publisher witness below.
            FjPreparedStateView stillHeld{};
            status(heldRustState.view(stillHeld, heldRustError), FJ_STATUS_SUCCESS);
            EXPECT_EQ(stillHeld.recipe_hash, heldRustHash);

            Diagnostic retryLeaf;
            FjIlluminant equalEnergy{};
            status(fj_legacy_illuminant_equal_energy(&equalEnergy, &retryLeaf.error), FJ_STATUS_SUCCESS);
            std::optional<PreparedState> retried;
            FjErrorBuffer retryError{nullptr, 0, 0};
            status(prepare_state(assets, controls, retried, retryError), FJ_STATUS_SUCCESS);
            ASSERT_TRUE(retried);
            FjPreparedStateView retriedView{};
            status(retried->view(retriedView, retryError), FJ_STATUS_SUCCESS);
            EXPECT_EQ(retriedView.recipe_hash, heldRustHash);

            publication_fault_case.passed();
        }

        // Actual complete-state publication for EQUAL faults is independent
        // from the raw factory diagnostics above.
        auto equalPublisher = std::make_unique<RenderInstance>();
        auto retainedControls = controls;
        JuicerTestSupport::capture(*equalPublisher, retainedControls);
        auto retainedPublication = JuicerTestSupport::admit(assets, *equalPublisher);
        ASSERT_EQ(retainedPublication.kind, FJ_ADMISSION_READY);
        ASSERT_TRUE(retainedPublication.state);
        const auto retainedCounterStart = JuicerTestSupport::PublicationObserver::metrics(*equalPublisher).build;
        JuicerTests::GtestCohort preflash_fault_cases("preflash-faults", 4u);
        std::size_t preflash_fault_index = 0;
        for (std::uint32_t fault : {1u, 2u, 3u, 4u}) {
            auto preflash_fault_case = preflash_fault_cases.case_event(std::to_string(preflash_fault_index++));
            auto changed = retainedControls;
            changed.cameraExposureCompensationEv += static_cast<double>(fault) * 0.01;
            JuicerTestSupport::capture(*equalPublisher, changed);
            const auto previousMetrics = JuicerTestSupport::PublicationObserver::metrics(*equalPublisher);
            const auto previousCounter = previousMetrics.build;
            const auto priorPublicationHold = retainedPublication;
            const auto previousValues = complete_result(JuicerTestSupport::published_view(retainedPublication), retainedControls);
            status(fj_test_illuminant_arm_fault(3, 1, fault), FJ_STATUS_SUCCESS);

            bool failed = false;
            JuicerTestSupport::PublicationAdmission failure;
            if (fault == 3) {
                try {
                    (void)JuicerTestSupport::admit(assets, *equalPublisher);
                    FAIL() << "actual publisher allocation fault admitted";
                } catch (const std::bad_alloc&) {
                    failed = true;
                }
            } else {
                failure = JuicerTestSupport::admit(assets, *equalPublisher);
                if (fault == 2)
                    failed = failure.delivery.category == FJ_STATUS_INTERNAL_FAILURE && failure.kind == FJ_ADMISSION_POISONED;
                else
                    failed = failure.kind == FJ_ADMISSION_PREPARATION_FAILED;
            }
            ASSERT_TRUE(failed) << "actual equal-energy publisher fault=" << fault;
            const auto failedMetrics = JuicerTestSupport::PublicationObserver::metrics(*equalPublisher);
            EXPECT_EQ(failedMetrics.build, previousCounter);
            EXPECT_EQ(failedMetrics.publishedIdentity, previousMetrics.publishedIdentity);
            EXPECT_EQ(failedMetrics.backingIdentity, previousMetrics.backingIdentity);
            EXPECT_FALSE(failure.state);
            EXPECT_EQ(complete_result(JuicerTestSupport::published_view(retainedPublication), retainedControls), previousValues);
            EXPECT_EQ(complete_result(JuicerTestSupport::published_view(priorPublicationHold), retainedControls), previousValues);

            std::uint64_t recoveredBuild = 0;
            if (fault == 2) {
                require_poisoned(assets, *equalPublisher, "equal-energy panic");
                equalPublisher = std::make_unique<RenderInstance>();
                JuicerTestSupport::capture(*equalPublisher, retainedControls);
                retainedPublication = JuicerTestSupport::admit(assets, *equalPublisher);
                ASSERT_EQ(retainedPublication.kind, FJ_ADMISSION_READY) << "fresh publisher rebuilds retained A after terminal panic";
                EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(*equalPublisher).build, 1u);
                JuicerTestSupport::capture(*equalPublisher, changed);
                retainedPublication = JuicerTestSupport::admit(assets, *equalPublisher);
                ASSERT_EQ(retainedPublication.kind, FJ_ADMISSION_READY) << "fresh publisher retries equal-energy fault";
                recoveredBuild = 2;
            } else {
                JuicerTestSupport::capture(*equalPublisher, retainedControls);
                const auto reused = JuicerTestSupport::admit(assets, *equalPublisher);
                ASSERT_EQ(reused.kind, FJ_ADMISSION_READY) << "failed attempt retained previous state fault=" << fault;
                ASSERT_TRUE(reused.state);
                EXPECT_EQ(JuicerTestSupport::PublicationObserver::backing(*reused.state), JuicerTestSupport::PublicationObserver::backing(*retainedPublication.state));
                EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(*equalPublisher).build, previousCounter);
                JuicerTestSupport::capture(*equalPublisher, changed);
                retainedPublication = JuicerTestSupport::admit(assets, *equalPublisher);
                ASSERT_EQ(retainedPublication.kind, FJ_ADMISSION_READY) << "equal-energy retry fault=" << fault;
                recoveredBuild = previousCounter + 1;
            }
            EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(*equalPublisher).inputIdentity, hash_params(changed));
            EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(*equalPublisher).build, recoveredBuild);
            const auto retryValues = complete_result(JuicerTestSupport::published_view(retainedPublication), changed);
            EXPECT_NE(retryValues, previousValues);
            const auto warm = JuicerTestSupport::admit(assets, *equalPublisher);
            ASSERT_TRUE(warm.state);
            EXPECT_EQ(JuicerTestSupport::PublicationObserver::backing(*warm.state), JuicerTestSupport::PublicationObserver::backing(*retainedPublication.state));
            EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(*equalPublisher).build, recoveredBuild);
            retainedControls = changed;

            preflash_fault_case.passed();
        }
        EXPECT_EQ(JuicerTestSupport::PublicationObserver::metrics(*equalPublisher).build, retainedCounterStart + 3);
    }

} // namespace IlluminantConstruction
