#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <source_location>
#include <string>
#include <string_view>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "JuicerState.h"
#include "Illuminants.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RustAssetBridge.h"
#include "juicer_cuda_owner.h"
#include "prepared_boundary.h"
#include "prepared_descriptors.h"
#include "prepared_state_native.h"
#include "prepared_state_oracle.h"
#include "juicer_test_api.h"
#include "juicer_cuda_call_scope.h"

extern "C" const std::size_t* fj_test_prepared_state_c_facts(std::size_t*);
extern "C" const std::size_t* fj_test_prepared_call_c_facts(std::size_t*);
extern "C" int fj_test_host_preparation_c_call();
extern "C" const std::size_t* fj_test_prepared_call_abi_facts(std::size_t*);
extern "C" FjStatus fj_test_prepared_state_create_c(FjAssets*, const FjPreparationParams*, FjPreparedState**, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_state_view_c(const FjPreparedState*, FjPreparedStateView*, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_state_release_c(FjPreparedState*, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_illuminants_acquire_c(const FjAssets*, FjIlluminantsOwner**, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_illuminants_view_c(const FjIlluminantsOwner*, FjIlluminantsView*, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_illuminants_release_c(FjIlluminantsOwner*, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_hull_acquire_c(const FjAssets*, FjInputHullOwner**, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_hull_view_c(const FjInputHullOwner*, FjInputHullView*, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_hull_release_c(FjInputHullOwner*, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_boundary_acquire_c(const FjAssets*, const FjOutputBoundarySource*, std::uint64_t, FjOutputBoundaryOwner**, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_boundary_view_c(const FjOutputBoundaryOwner*, FjOutputBoundaryView*, FjErrorBuffer*);
extern "C" FjStatus fj_test_prepared_boundary_release_c(FjOutputBoundaryOwner*, FjErrorBuffer*);

static_assert(std::is_same_v<decltype(&fj_legacy_prepared_state_create), decltype(&fj_test_prepared_state_create_c)>);
static_assert(std::is_same_v<decltype(&fj_legacy_prepared_state_view), decltype(&fj_test_prepared_state_view_c)>);
static_assert(std::is_same_v<decltype(&fj_legacy_prepared_state_release), decltype(&fj_test_prepared_state_release_c)>);

using FjCallPrecheckFn = FjStatus (*)(void*, FjErrorBuffer*);
using FjCallAdmitFn = FjStatus (*)(void*, FjErrorBuffer*);
using FjCallReleaseFn = void (*)(void*);
using FjCallInspectFn = FjStatus (*)(void*, const FjFrame*, FjCudaContext*, FjErrorBuffer*);
using FjCallPreviewFn = FjStatus (*)(void*, const FjRect*, const FjRect*, std::uint32_t, FjAutoExposure*, FjErrorBuffer*);
using FjCallFrameDescriptorsFn = FjStatus (*)(void*, const FjFrameMathInput*, FjNativeFrameDescriptors*, FjErrorBuffer*);
using FjCallRenderFn = FjRenderOutcome (*)(void*, const FjCudaContext*, const FjFrame*, const FjSubmission*, const FjPreparedHostData*, const FjHostPreparation*, FjAbortCallback, FjErrorBuffer*);
using FjCallCompleteFn = FjCallCompletionResult (*)(void*, const FjCallCompletionInput*);
using FjHostScannerLutFn = FjStatus (*)(void*, const FjScannerLutInput*, FjScannerLutView*, FjScannerLutFailure*, FjErrorBuffer*);
using FjHostGaussianFn = FjStatus (*)(void*, const FjGaussianKernelInput*, FjGaussianKernelView*, FjGaussianKernelFailure*, FjErrorBuffer*);
using FjSubmissionSelectFn = FjStatus (*)(void*, const FjCudaContext*, const FjSubmission*, FjSubmission*, FjErrorBuffer*);
using FjPolicyEntryFn = FjStatus (*)(const FjPreparedState*, const FjExtent*, FjPreparedFramePolicy*, FjErrorBuffer*);
using FjRenderEntryFn = FjCallCompletionResult (*)(const FjAssets*, const FjPreparedState*, FjCudaCallAccess, const FjFrame*, const FjPreparedFramePolicy*, FjSubmissionAccess, FjAbortCallback, FjErrorBuffer*);
using FjUnavailableEntryFn = FjCallCompletionResult (*)(const FjAssets*, const FjPreparedState*, std::uint32_t, FjCudaCallAccess, const FjFrame*, FjAbortCallback, FjErrorBuffer*);

static_assert(std::is_same_v<decltype(FjCudaCallAccess::precheck), FjCallPrecheckFn>);
static_assert(std::is_same_v<decltype(FjCudaCallAccess::admit), FjCallAdmitFn>);
static_assert(std::is_same_v<decltype(FjCudaCallAccess::release), FjCallReleaseFn>);
static_assert(std::is_same_v<decltype(FjCudaCallAccess::inspect), FjCallInspectFn>);
static_assert(std::is_same_v<decltype(FjCudaCallAccess::preview), FjCallPreviewFn>);
static_assert(std::is_same_v<decltype(FjCudaCallAccess::frame_descriptors), FjCallFrameDescriptorsFn>);
static_assert(std::is_same_v<decltype(FjCudaCallAccess::render), FjCallRenderFn>);
static_assert(std::is_same_v<decltype(FjCudaCallAccess::complete), FjCallCompleteFn>);
static_assert(std::is_same_v<decltype(FjHostPreparation::scanner_lut), FjHostScannerLutFn>);
static_assert(std::is_same_v<decltype(FjHostPreparation::gaussian), FjHostGaussianFn>);
static_assert(std::is_same_v<decltype(FjSubmissionAccess::select), FjSubmissionSelectFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_prepared_state_frame_policy), FjPolicyEntryFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_prepared_state_render), FjRenderEntryFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_prepared_unavailable), FjUnavailableEntryFn>);

namespace {
    std::filesystem::path artifactRoot;
    using Json = nlohmann::json;
    constexpr std::array<std::string_view, 31> kProjectionCases{
        "NegativeDirectHanatos", "NegativeDirectMallett", "NegativeDirectArctic", "NegativePrintHanatos", "NegativePrintMallett", "NegativePrintArctic", "PositiveDirectHanatos", "PositiveDirectMallett", "PositiveDirectArctic", "PositivePrintHanatos", "PositivePrintMallett", "PositivePrintArctic", "DirectIgnoresPrintControls", "PreflashAndNormalizationModes[norm=0,compensate=0]", "PreflashAndNormalizationModes[norm=0,compensate=1]", "PreflashAndNormalizationModes[norm=1,compensate=0]", "PreflashAndNormalizationModes[norm=1,compensate=1]", "GammaAndGrainLayers", "PhysicalDefectsAndWeave", "CameraAndEnlargerOptics", "DirAndCompressionOff", "SignedZeroGlare[+0]", "SignedZeroGlare[-0]", "CorrectionsAndScannerControls[print]", "CorrectionsAndScannerControls[direct]", "EncodingOnlyReusesMediumIdentities[encoded]", "EncodingOnlyReusesMediumIdentities[linear]", "SnapshotIntegerBooleansPreserveNonzeroCasts[direct]", "SnapshotIntegerBooleansPreserveNonzeroCasts[print]", "DisabledFamiliesExcludeUnusedControls", "GrainDebugSuppressesPhysicalEffects"};
    std::optional<Json> projectionOracle;
    Json projectionRows = Json::array();
    std::vector<std::string> observedProjectionCases;
    bool captureProjection = false;

    void exact_keys(const Json& value, std::initializer_list<std::string_view> names) {
        if (!value.is_object() || value.size() != names.size())
            throw std::runtime_error("prepared-state oracle object key count");
        for (const auto name : names)
            if (!value.contains(std::string(name)))
                throw std::runtime_error("prepared-state oracle missing key " + std::string(name));
    }

    void require_projection_membership(const Json& cases) {
        if (!cases.is_array() || cases.size() != kProjectionCases.size())
            throw std::runtime_error("prepared-state oracle requires exactly 31 rows");
        for (std::size_t i = 0; i < kProjectionCases.size(); ++i)
            if (cases.at(i).at("case_id") != std::string(kProjectionCases[i]))
                throw std::runtime_error("prepared-state oracle row membership/order");
    }

    void require_projection_inputs(const Json& inputs, const std::string& preset) {
        exact_keys(inputs, {"resource_directory", "parameter_type"});
        if (inputs.at("resource_directory") != "out/build/" + preset + "/tests/Resources" || inputs.at("parameter_type") != "FjPreparationParams")
            throw std::runtime_error("wrong prepared-state oracle inputs");
    }

    Json read_projection(const std::filesystem::path& path, const std::string& preset) {
        std::ifstream source(path, std::ios::binary);
        if (!source)
            throw std::runtime_error("missing prepared-state oracle " + path.string());
        std::vector<std::set<std::string>> objects;
        const auto parsed = Json::parse(source, [&](int, Json::parse_event_t event, Json& value) {
            if (event == Json::parse_event_t::object_start)
                objects.emplace_back();
            else if (event == Json::parse_event_t::key && !objects.back().insert(value.get<std::string>()).second)
                throw std::runtime_error("duplicate prepared-state oracle object key");
            else if (event == Json::parse_event_t::object_end)
                objects.pop_back();
            return true;
        });
        exact_keys(parsed, {"schema_version", "preset", "producer", "toolchain", "configuration", "inputs", "cases"});
        if (parsed.at("schema_version") != 1 || parsed.at("preset") != preset ||
            (preset != "linux-debug" && preset != "linux-release" && preset != "windows-clang-debug" && preset != "windows-clang-release"))
            throw std::runtime_error("unsupported prepared-state oracle schema/preset");
        const auto& producer = parsed.at("producer");
        exact_keys(producer, {"authority", "target", "manifest_entry"});
        if (producer.at("authority") != "Film-Juicer native composition" || producer.at("target") != "JuicerPreparedStateTests" || producer.at("manifest_entry") != preset)
            throw std::runtime_error("wrong prepared-state oracle producer");
        const auto& toolchain = parsed.at("toolchain");
        exact_keys(toolchain, {"cxx", "rust_toolchain", "pointer_bits"});
        if (toolchain.at("cxx") != __VERSION__ || toolchain.at("rust_toolchain") != "1.98.1" || toolchain.at("pointer_bits") != sizeof(void*) * 8)
            throw std::runtime_error("wrong prepared-state oracle toolchain");
        const auto& configuration = parsed.at("configuration");
        exact_keys(configuration, {"build_type", "testing"});
        if (configuration.at("build_type") != (preset.ends_with("debug") ? "Debug" : "Release") || configuration.at("testing") != true)
            throw std::runtime_error("wrong prepared-state oracle configuration");
        require_projection_inputs(parsed.at("inputs"), preset);
        const auto& cases = parsed.at("cases");
        require_projection_membership(cases);
        for (std::size_t i = 0; i < kProjectionCases.size(); ++i) {
            const auto& row = cases.at(i);
            exact_keys(row, {"case_id", "input", "status", "api", "native_code", "diagnostic", "expected", "captured_actual", "backing_relations"});
            if (row.at("case_id") != std::string(kProjectionCases[i]) || row.at("status") != FJ_STATUS_SUCCESS || row.at("api") != FJ_API_NONE || row.at("native_code") != 0 || row.at("diagnostic") != "")
                throw std::runtime_error("prepared-state oracle row membership/status");
            // Provenance is structurally checked; only expected supplies acceptance values.
            PreparedStateOracle::compare(row.at("expected"), row.at("captured_actual"));
            if (!row.at("backing_relations").is_array())
                throw std::runtime_error("prepared-state oracle backing relation shape");
            for (const auto& edge : row.at("backing_relations")) {
                exact_keys(edge, {"left", "right", "equal"});
                if (!edge.at("left").is_string() || !edge.at("right").is_string() || edge.at("equal") != true)
                    throw std::runtime_error("prepared-state oracle backing relation");
            }
        }
        return parsed;
    }

    std::string projection_case(const ParamSnapshot& params) {
        std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        if (name == "PreflashAndNormalizationModes")
            name += "[norm=" + std::to_string(params.normalizePrintExposure) + ",compensate=" + std::to_string(params.printExposureCompensation) + "]";
        else if (name == "SignedZeroGlare")
            name += std::signbit(params.glarePercent) ? "[-0]" : "[+0]";
        else if (name == "CorrectionsAndScannerControls" || name == "SnapshotIntegerBooleansPreserveNonzeroCasts")
            name += Spektrafilm::scan_route_is_print(params.scanRoute) ? "[print]" : "[direct]";
        else if (name == "EncodingOnlyReusesMediumIdentities")
            name += params.outputCctfEncoding != 0 ? "[encoded]" : "[linear]";
        return name;
    }

    Json backing_edge(std::string left, std::string right, bool equal) {
        return {{"left", std::move(left)}, {"right", std::move(right)}, {"equal", equal}};
    }

    const Json& oracle_row(std::string_view name) {
        for (const auto& row : projectionOracle->at("cases"))
            if (row.at("case_id") == std::string(name))
                return row;
        throw std::runtime_error("unaccounted prepared-state oracle row");
    }

    bool encoding_edge(const Json& edge) {
        return edge.at("left").get<std::string>().starts_with("EncodingOnlyReusesMediumIdentities[");
    }

    void compare_backing(const Json& expected, const Json& actual) {
        if (expected != actual)
            throw std::runtime_error("prepared-state oracle live backing relationship mismatch");
    }

    void record_projection(const ParamSnapshot& params, const FjPreparedStateView& expected, const FjPreparedStateView& actual) {
        const auto name = projection_case(params);
        if (std::find(kProjectionCases.begin(), kProjectionCases.end(), name) == kProjectionCases.end() ||
            std::find(observedProjectionCases.begin(), observedProjectionCases.end(), name) != observedProjectionCases.end())
            throw std::runtime_error("unknown or duplicate prepared-state projection case " + name);
        observedProjectionCases.push_back(name);
        const auto expectedRecord = PreparedStateOracle::view(expected);
        const auto actualRecord = PreparedStateOracle::view(actual);
        PreparedStateOracle::compare(expectedRecord, actualRecord);
        Json relations = Json::array();
        if (expected.input_hull.xy.count != 0) {
            relations.push_back(backing_edge("native.input_hull.entry_identity", "actual.input_hull.entry_identity", expected.input_hull.entry_identity == actual.input_hull.entry_identity));
            relations.push_back(backing_edge("native.input_hull.xy", "actual.input_hull.xy", expected.input_hull.xy.data == actual.input_hull.xy.data));
        }
        if (expected.output_boundary.cmax.count != 0) {
            relations.push_back(backing_edge("native.output_boundary.entry_identity", "actual.output_boundary.entry_identity", expected.output_boundary.entry_identity == actual.output_boundary.entry_identity));
            relations.push_back(backing_edge("native.output_boundary.cmax", "actual.output_boundary.cmax", expected.output_boundary.cmax.data == actual.output_boundary.cmax.data));
        }
        for (const auto& edge : relations)
            EXPECT_EQ(edge.at("equal"), true);
        const auto input = PreparedStateOracle::params(preparation_params(params));
        if (projectionOracle) {
            const auto& row = oracle_row(name);
            PreparedStateOracle::compare(row.at("input"), input);
            PreparedStateOracle::compare(row.at("expected"), actualRecord);
            Json recordedRelations = Json::array();
            for (const auto& edge : row.at("backing_relations"))
                if (!encoding_edge(edge))
                    recordedRelations.push_back(edge);
            compare_backing(recordedRelations, relations);
        }
        if (captureProjection)
            projectionRows.push_back({{"case_id", name}, {"input", input}, {"status", FJ_STATUS_SUCCESS}, {"api", FJ_API_NONE}, {"native_code", 0}, {"diagnostic", ""}, {"expected", expectedRecord}, {"captured_actual", actualRecord}, {"backing_relations", std::move(relations)}});
    }

    void observe_encoding_relation(bool equal) {
        const auto edge = backing_edge("EncodingOnlyReusesMediumIdentities[encoded].output_boundary.entry_identity", "EncodingOnlyReusesMediumIdentities[linear].output_boundary.entry_identity", equal);
        for (const auto* const name : {"EncodingOnlyReusesMediumIdentities[encoded]", "EncodingOnlyReusesMediumIdentities[linear]"}) {
            if (projectionOracle) {
                Json edges = Json::array();
                for (const auto& recorded : oracle_row(name).at("backing_relations"))
                    if (encoding_edge(recorded))
                        edges.push_back(recorded);
                compare_backing(edges, Json::array({edge}));
            }
            if (captureProjection)
                for (auto& row : projectionRows)
                    if (row.at("case_id") == name)
                        row.at("backing_relations").push_back(edge);
        }
    }

    std::string successful_projection_filter() {
        std::set<std::string> names;
        for (const auto name : kProjectionCases)
            names.insert("PreparedStateProducts." + std::string(name.substr(0, name.find('['))));
        std::string filter;
        for (const auto& name : names) {
            if (!filter.empty())
                filter += ':';
            filter += name;
        }
        return filter;
    }

    Json projection_payload(const std::string& preset, const std::string& resources) {
        return {{"schema_version", 1}, {"preset", preset}, {"producer", {{"authority", "Film-Juicer native composition"}, {"target", "JuicerPreparedStateTests"}, {"manifest_entry", preset}}}, {"toolchain", {{"cxx", __VERSION__}, {"rust_toolchain", "1.98.1"}, {"pointer_bits", sizeof(void*) * 8}}}, {"configuration", {{"build_type", preset.ends_with("debug") ? "Debug" : "Release"}, {"testing", true}}}, {"inputs", {{"resource_directory", resources}, {"parameter_type", "FjPreparationParams"}}}, {"cases", projectionRows}};
    }
    FjAssets* assets() {
        JuicerProcess::root().ensure_bootstrap();
        return JuicerProcess::root().assets().borrow_rust_assets();
    }

    FjStatus state_view(const std::optional<PreparedState>& owner, FjPreparedStateView& out) {
        if (!owner)
            return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        FjErrorBuffer error{nullptr, 0, 0};
        return owner->view(out, error);
    }

    template <class T>
    void equal_bits(T actual, T expected) {
        static_assert(std::is_floating_point_v<T>);
        using Bytes = std::array<std::byte, sizeof(T)>;
        EXPECT_EQ(std::bit_cast<Bytes>(actual), std::bit_cast<Bytes>(expected));
    }

    void equal_span(FjFloatSpan actual, std::span<const float> expected, std::source_location call = std::source_location::current()) {
        SCOPED_TRACE(::testing::Message() << call.file_name() << ':' << call.line());
        ASSERT_EQ(actual.count, expected.size());
        if (!expected.empty()) {
            ASSERT_NE(actual.data, nullptr);
            EXPECT_EQ(std::memcmp(actual.data, expected.data(), expected.size_bytes()), 0);
        }
    }

    void equal_triplets(FjFloatSpan actual, const std::vector<std::array<float, 3>>& expected) {
        static_assert(sizeof(std::array<float, 3>) == 3 * sizeof(float));
        ASSERT_EQ(actual.count, expected.size() * 3);
        if (!expected.empty()) {
            ASSERT_NE(actual.data, nullptr);
            EXPECT_EQ(std::memcmp(actual.data, expected.data(), expected.size() * sizeof(expected.front())), 0);
        }
    }

    void equal_tables(const FjPreparedSpectralView& actual, const Spectral::SpectralTables& expected) {
        equal_span(actual.lambda_nm, expected.lambda);
        equal_span(actual.illuminant, expected.illum);
        equal_span(actual.observer_xyz[0], expected.Xbar);
        equal_span(actual.observer_xyz[1], expected.Ybar);
        equal_span(actual.observer_xyz[2], expected.Zbar);
        equal_span(actual.weighted_xyz[0], expected.Ax);
        equal_span(actual.weighted_xyz[1], expected.Ay);
        equal_span(actual.weighted_xyz[2], expected.Az);
        equal_span(actual.dyes_cmy[0], expected.epsC);
        equal_span(actual.dyes_cmy[1], expected.epsM);
        equal_span(actual.dyes_cmy[2], expected.epsY);
        equal_span(actual.baseline_min, expected.baseDensityMin);
        equal_span(actual.baseline_mid, expected.baseDensityMid);
        equal_bits(actual.inv_yn, expected.invYn);
        for (std::size_t c = 0; c < 3; ++c)
            equal_bits(actual.white_xyz[c], expected.whiteXYZ[c]);
        EXPECT_EQ(actual.has_baseline, expected.hasBaseline);
        EXPECT_EQ(actual.illuminant_hash, expected.illuminantHash);
        EXPECT_EQ(actual.hash, expected.tablesHash);
    }

    void equal_bounds(const FjFilmDensityBounds& actual, const DensityBoundsRecipe& expected) {
        EXPECT_EQ(actual.route, static_cast<unsigned>(expected.route));
        EXPECT_EQ(actual.medium, static_cast<unsigned>(expected.medium));
        EXPECT_EQ(actual.polarity, static_cast<unsigned>(expected.polarity));
        EXPECT_EQ(actual.source, static_cast<unsigned>(expected.source));
        equal_span({actual.minimum_cmy, 3}, expected.dataMinCmy);
        equal_span({actual.maximum_cmy, 3}, expected.dataMaxCmy);
        equal_span({actual.inv_span_cmy, 3}, expected.invSpanCmy);
        equal_span({actual.authored_minimum_cmy, 3}, expected.authoredMinCmy);
        equal_span({actual.authored_maximum_cmy, 3}, expected.authoredMaxCmy);
        EXPECT_EQ(actual.hash, expected.hash);
    }

    void equal_encoding(const FjOutputEncoding& actual, const FjOutputEncoding& expected) {
        EXPECT_EQ(actual.schema, expected.schema);
        EXPECT_EQ(actual.cctf_kind, expected.cctf_kind);
        equal_bits(actual.gamma, expected.gamma);
        equal_bits(actual.a, expected.a);
        equal_bits(actual.b, expected.b);
        equal_bits(actual.c, expected.c);
        equal_bits(actual.d, expected.d);
        equal_bits(actual.linear_cutoff, expected.linear_cutoff);
        equal_span({actual.dwg_to_output, 9}, expected.dwg_to_output);
    }

    void equal_diffusion(const FjDiffusionRecipe& actual, const DiffusionFilterOpticsRecipe& expected) {
        const auto& resolved = expected.resolved;
        EXPECT_EQ(actual.domain, static_cast<unsigned>(expected.domain));
        EXPECT_EQ(actual.active, resolved.active);
        EXPECT_EQ(actual.family, static_cast<unsigned>(resolved.family));
        equal_bits(actual.scatter_fraction, resolved.scatterFraction);
        equal_bits(actual.warmth, resolved.effectiveWarmth);
        equal_bits(actual.spatial_scale, resolved.spatialScale);
        for (std::size_t i = 0; i < 3; ++i) {
            equal_bits(actual.group_weights[i], resolved.groupWeightsCoreHaloBloom[i]);
            equal_bits(actual.group_lambda_um[i], resolved.groupCenterLambdaUm[i]);
        }
        EXPECT_EQ(actual.resolved_hash, resolved.hash);
        EXPECT_EQ(actual.hash, expected.hash);
    }

    void equal_fixed_recipes(const FjPreparedStateView& actual, const RenderRecipe& expected, const FocusedRenderPayload& payload) {
        const auto& grain = expected.visualGrain;
        EXPECT_EQ(actual.grain.active, grain.active);
        EXPECT_EQ(actual.grain.sublayers_active, grain.sublayersActive);
        EXPECT_EQ(actual.grain.sublayer_count, grain.nSubLayers);
        EXPECT_EQ(actual.grain.debug_view, grain.debugView);
        EXPECT_EQ(actual.grain.layers_hash, grain.densityCurvesLayersHash);
        equal_span({actual.grain.particle_scale_cmy, 3}, grain.particleScaleCmy);
        equal_span({actual.grain.particle_scale_layers, 3}, grain.particleScaleLayers);
        equal_span({actual.grain.particle_density_min_cmy, 3}, grain.visualParticleDensityMinCmy);
        equal_span({actual.grain.uniformity_cmy, 3}, grain.uniformityCmy);
        equal_span({actual.grain.micro_structure, 2}, grain.microStructure);
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_EQ(actual.grain.axis_finite_cmy[c], grain.grainLayerAxisFinite[c]);
            equal_span({actual.grain.axis_block_prefix_max_cmy[c], 16}, grain.grainLayerAxisBlockPrefixMax[c]);
        }
        equal_bits(actual.grain.particle_area_um2, grain.particleAreaUm2);
        equal_bits(actual.grain.correlation_sigma_px, grain.correlationSigmaPx);
        equal_bits(actual.grain.dye_cloud_blur_um, grain.dyeCloudBlurUm);
        equal_bits(actual.grain.amplitude, grain.amplitude);
        equal_bits(actual.grain.chroma_mix, grain.chromaMix);
        equal_bits(actual.grain.chroma_shared_weight, grain.chromaSharedWeight);
        equal_bits(actual.grain.chroma_independent_weight, grain.chromaIndependentWeight);
        equal_bits(actual.grain.fine_weight, grain.fineWeight);
        equal_bits(actual.grain.mid_weight, grain.midWeight);
        equal_bits(actual.grain.coarse_weight, grain.coarseWeight);
        equal_bits(actual.grain.size_mix_scale, grain.sizeMixScale);
        equal_bits(actual.grain.clump_temporal_mix, grain.clumpTemporalMix);
        equal_bits(actual.grain.clump_morph_period_sec, grain.clumpMorphPeriodSec);
        const auto& optics = expected.spatialOptics;
        equal_diffusion(actual.optics.camera_diffusion, optics.cameraDiffusion);
        equal_diffusion(actual.optics.enlarger_diffusion, optics.enlargerDiffusion);
        equal_bits(actual.optics.camera_lens_blur.sigma_um, optics.cameraLensBlur.sigmaUm);
        EXPECT_EQ(actual.optics.camera_lens_blur.hash, optics.cameraLensBlur.hash);
        const auto& halation = optics.scatterHalation;
        EXPECT_EQ(actual.optics.scatter_halation.scatter_active, halation.scatterActive);
        EXPECT_EQ(actual.optics.scatter_halation.back_reflection_active, halation.backReflectionActive);
        equal_bits(actual.optics.scatter_halation.scatter_amount, halation.scatterAmount);
        equal_bits(actual.optics.scatter_halation.scatter_spatial_scale, halation.scatterSpatialScale);
        equal_bits(actual.optics.scatter_halation.halation_spatial_scale, halation.halationSpatialScale);
        equal_span({actual.optics.scatter_halation.first_sigma_um, 3}, halation.halationFirstSigmaUm);
        equal_span({actual.optics.scatter_halation.total_strength, 3}, halation.totalStrength);
        EXPECT_EQ(actual.optics.scatter_halation.hash, halation.hash);
        JuicerCuda::PreparedDescriptors fixed;
        fixed.color = payload.scannerColor;
        fixed.outputGamut = expected.scannerOutput.outputGamut;
        fixed.effects.emplace();
        fixed.effects->filmDust = expected.filmJuicerEffects.filmDust;
        fixed.effects->filmScratch = expected.filmJuicerEffects.filmScratch;
        fixed.effects->gateDust = expected.filmJuicerEffects.gateDust;
        fixed.effects->gateScratch = expected.filmJuicerEffects.gateScratch;
        FjPreparedHostData wire{};
        JuicerCudaTest::encode_prepared_descriptors(fixed, wire);
        // These records contain exactly the listed float words and no padding.
        static_assert(sizeof(FjDust) == 42 * sizeof(float) && alignof(FjDust) == alignof(float));
        static_assert(sizeof(FjScratch) == 36 * sizeof(float) && alignof(FjScratch) == alignof(float));
        EXPECT_EQ(std::memcmp(&actual.effects.film_dust, &wire.effects.film_dust, sizeof(FjDust)), 0);
        EXPECT_EQ(std::memcmp(&actual.effects.gate_dust, &wire.effects.gate_dust, sizeof(FjDust)), 0);
        EXPECT_EQ(std::memcmp(&actual.effects.film_scratch, &wire.effects.film_scratch, sizeof(FjScratch)), 0);
        EXPECT_EQ(std::memcmp(&actual.effects.gate_scratch, &wire.effects.gate_scratch, sizeof(FjScratch)), 0);
        EXPECT_EQ(actual.effects.active, expected.filmJuicerEffects.active);
        equal_bits(actual.effects.gate_weave_amount, expected.filmJuicerEffects.gateWeaveAmount);
        const auto& color = wire.output_color;
        EXPECT_EQ(actual.output_color.color_space, color.color_space);
        EXPECT_EQ(actual.output_color.flags, color.flags);
        equal_span({actual.output_color.cat02, 9}, color.cat02);
        equal_span({actual.output_color.xyz_to_rgb, 9}, color.xyz_to_rgb);
        equal_span({actual.output_color.illuminant_xyz, 3}, color.illuminant_xyz);
        equal_span({actual.output_color.native_rgb_to_d65_xyz, 9}, color.native_rgb_to_d65_xyz);
        equal_span({actual.output_color.d65_xyz_to_native_rgb, 9}, color.d65_xyz_to_native_rgb);
        equal_span({actual.output_color.lightness_knee, 3}, color.lightness_knee);
        equal_span({actual.output_color.chroma_knee, 3}, color.chroma_knee);
        EXPECT_EQ(actual.output_color.gamut_recipe_hash, color.gamut_recipe_hash);
        equal_encoding(actual.output_color.encoding, color.encoding);
        equal_encoding(actual.scanner_color.encoding, color.encoding);
        EXPECT_EQ(actual.scanner_color.output_space, color.color_space);
        EXPECT_EQ(actual.scanner_color.flags, color.flags & ~FJ_COLOR_GAMUT_COMPRESSION);
        EXPECT_EQ(actual.scanner_color.active_gamut_recipe_hash, payload.scannerColor.outputGamutRecipeHash);
    }

    void compare_product(const ParamSnapshot& params) {
        // Unchanged accepted native assembly is the independent composition oracle.
        FocusedRenderStateBuildProduct expected;
        std::string diagnostic;
        const bool print = params.scanRoute == Spektrafilm::ScanRoute::NegativePrintScan ||
                           params.scanRoute == Spektrafilm::ScanRoute::PositivePrintScan;
        ASSERT_TRUE(print ? build_print_render_state_product(params, expected, diagnostic) : build_direct_render_state_product(params, expected, diagnostic)) << diagnostic;
        FjStatus status{};
        auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_TRUE(owner.has_value());
        FjPreparedStateView actual{};
        ASSERT_EQ(state_view(owner, actual).category, FJ_STATUS_SUCCESS);
        const auto& recipe = expected.recipe;
        const auto& payload = expected.payload;
        equal_fixed_recipes(actual, recipe, payload);
        EXPECT_EQ(actual.recipe_hash, recipe.hash);
        EXPECT_EQ(actual.profile_route_hash, recipe.profileRoute.hash);
        EXPECT_EQ(actual.upload_core_hash, payload.uploadCoreHash);
        EXPECT_EQ(actual.scanner_hash, payload.scannerHash);
        EXPECT_EQ(actual.route, static_cast<unsigned>(recipe.profileRoute.scanRoute));
        EXPECT_EQ(actual.polarity, static_cast<unsigned>(recipe.profileRoute.capturePolarity));
        EXPECT_EQ(std::string(actual.film_profile_key.data, actual.film_profile_key.count), recipe.profileRoute.filmProfileKey);
        EXPECT_EQ(actual.film_profile_asset_version, recipe.profileRoute.filmProfileAssetVersionToken);
        EXPECT_EQ(actual.print_profile_asset_version, recipe.profileRoute.printProfileAssetVersionToken);
        EXPECT_EQ(actual.film_exposure.hash, recipe.filmRaw.hash);
        const auto& input = recipe.filmRaw;
        EXPECT_EQ(actual.film_exposure.input_color_space, input.inputColorSpace);
        EXPECT_EQ(actual.film_exposure.method, static_cast<unsigned>(input.rgbToRawMethod));
        EXPECT_EQ(actual.film_exposure.flags, (input.inputCctfDecoding ? FJ_FILM_DECODE_CCTF : 0u) | (payload.filmRawConfig.applyInputChromaticAdapt ? FJ_FILM_ADAPT_INPUT : 0u) | (input.autoExposureEnabled ? FJ_FILM_AUTO_EXPOSURE : 0u));
        equal_bits(actual.film_exposure.manual_exposure_ev, input.manualExposureCompensationEv);
        equal_bits(actual.film_exposure.mallett_green_midgray_scale, input.mallettGreenMidgrayScale);
        equal_span({actual.film_exposure.input_rgb_to_xyz, 9}, input.inputRgbToXyz);
        equal_span({actual.film_exposure.input_xyz_adapt, 9}, input.inputXyzAdapt);
        equal_span({actual.film_exposure.xyz_to_linear_srgb, 9}, input.xyzToLinearSrgb);
        EXPECT_EQ(actual.film_exposure.sensitivity_hash, input.finalSensitivityHash);
        EXPECT_EQ(actual.film_exposure.tc_lut_hash, input.tcLutHash);
        const auto& policy = actual.input_policy;
        EXPECT_EQ(policy.meter, static_cast<unsigned>(input.autoExposureMethod));
        EXPECT_EQ(policy.band_pass_active, input.cameraBandPass.active);
        EXPECT_EQ(policy.apply_hanatos_window, input.hanatos.applyWindow);
        EXPECT_EQ(policy.apply_hanatos_surface, input.hanatos.applySurface);
        EXPECT_EQ(policy.input_compression_active, input.inputCompressionActive);
        equal_bits(policy.film_format_long_edge_mm, input.filmFormatLongEdgeMm);
        equal_bits(policy.hanatos_spectral_blur, input.hanatos.spectralGaussianBlur);
        equal_span({policy.uv, 3}, input.cameraBandPass.uv);
        equal_span({policy.ir, 3}, input.cameraBandPass.ir);
        equal_span({policy.nominal_white_xyz, 3}, input.inputNominalWhiteXYZ);
        equal_span({policy.projection_white_xyz, 3}, input.projectionWhiteXYZ);
        equal_span(policy.hanatos_window, input.hanatos.windowParams);
        ASSERT_EQ(policy.hanatos_surface_rgb.count, 45u);
        EXPECT_EQ(std::memcmp(policy.hanatos_surface_rgb.data, input.hanatos.surfaceParams.data(), sizeof(input.hanatos.surfaceParams)), 0);
        EXPECT_EQ(policy.input_hull_hash, input.inputCompressionHullHash);
        EXPECT_EQ(policy.tc_source_hash, input.tcSourceAssetHash);
        EXPECT_EQ(policy.projection_illuminant, input.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025 ? FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE : FJ_FILM_INPUT_ILLUMINANT_D65);
        if (input.inputCompressionHull) {
            const auto& hull = *input.inputCompressionHull;
            EXPECT_EQ(actual.input_hull.hash, hull.hash);
            EXPECT_EQ(actual.input_hull.entry_identity, hull.entryIdentity);
            EXPECT_EQ(actual.input_hull.xy.data, hull.xy.data());
            equal_span({actual.input_hull.center_xy, 2}, hull.center);
            ASSERT_EQ(actual.input_hull.xy.count, hull.xy.size() * 2);
            EXPECT_EQ(std::memcmp(actual.input_hull.xy.data, hull.xy.data(), actual.input_hull.xy.count * sizeof(float)), 0);
        } else {
            EXPECT_EQ(actual.input_hull.xy.count, 0u);
            EXPECT_EQ(actual.input_hull.hash, 0u);
        }
        ASSERT_EQ(actual.film_exposure.sensitivity_rgb.count, 81u * 3);
        ASSERT_NE(actual.film_exposure.sensitivity_rgb.data, nullptr);
        EXPECT_EQ(std::memcmp(actual.film_exposure.sensitivity_rgb.data, recipe.filmRaw.finalSensitivity.data(), sizeof(recipe.filmRaw.finalSensitivity)), 0);
        equal_span(actual.film_development.log_exposure, recipe.filmDevelop.logExposure);
        equal_triplets(actual.film_development.density_rgb, recipe.filmDevelop.normalizedDensityCurves);
        equal_span({actual.film_development.gamma_rgb, 3}, recipe.filmDevelop.densityCurveGamma);
        EXPECT_EQ(actual.film_development.hash, recipe.filmDevelop.hash);
        EXPECT_EQ(actual.film_development.density_curves_hash, recipe.filmDevelop.normalizedDensityCurvesHash);
        EXPECT_EQ(actual.film_development.density_layers_hash, recipe.filmDevelop.densityCurvesLayersHash);
        for (std::size_t layer = 0; layer < 3; ++layer)
            for (std::size_t c = 0; c < 3; ++c)
                equal_span(actual.film_development.density_layers[layer][c], recipe.filmDevelop.densityCurvesLayers[layer][c]);
        const auto& dir = recipe.dirCouplers;
        EXPECT_EQ(actual.dir_recipe.polarity, static_cast<unsigned>(dir.polarity));
        EXPECT_EQ(actual.dir_recipe.hash, dir.hash);
        EXPECT_EQ(actual.dir_recipe.compensated_axes_hash, dir.compensatedDensityCurveAxesHash);
        EXPECT_EQ(actual.dir_recipe.mode, static_cast<unsigned>(dir.nonlinearMode));
        EXPECT_EQ(std::memcmp(actual.dir_recipe.matrix_rgb, dir.matrixRgb.data(), sizeof(actual.dir_recipe.matrix_rgb)), 0);
        equal_span({actual.dir_recipe.density_max_rgb, 3}, dir.densityMaxRgb);
        equal_span({actual.dir_recipe.density_ref_rgb, 3}, dir.densityRefRgb);
        equal_span({actual.dir_recipe.donor_k_rgb, 3}, dir.donorKRgb);
        equal_span({actual.dir_recipe.receiver_c_ref_rgb, 3}, dir.receiverCRefRgb);
        equal_span({actual.dir_recipe.receiver_kr_rgb, 3}, dir.receiverKrRgb);
        for (std::size_t c = 0; c < 3; ++c)
            equal_span(actual.dir_recipe.compensated_axes_rgb[c], dir.compensatedDensityCurveAxesRgb[c]);
        equal_bits(actual.dir_recipe.diffusion_size_um, dir.diffusionSizeUm);
        equal_bits(actual.dir_recipe.diffusion_tail_um, dir.diffusionTailUm);
        equal_bits(actual.dir_recipe.diffusion_tail_weight, dir.diffusionTailWeight);
        equal_bounds(actual.scanner_bounds, recipe.densityBounds);
        if (print) {
            equal_bounds(actual.enlarger_film_bounds, recipe.enlargerFilmBounds);
        } else {
            // Direct states own no print branch; absent wire records are zero.
            EXPECT_EQ(actual.enlarger_film_bounds.hash, 0u);
            equal_span({actual.enlarger_film_bounds.minimum_cmy, 3}, std::array<float, 3>{});
            equal_span({actual.enlarger_film_bounds.maximum_cmy, 3}, std::array<float, 3>{});
            equal_span({actual.enlarger_film_bounds.inv_span_cmy, 3}, std::array<float, 3>{});
        }
        equal_span({actual.grain_minimum.cmy, 3}, recipe.grainContract.densityMinCmy);
        equal_tables(actual.exposure_tables, payload.exposureTables);
        equal_tables(actual.scanner_tables, payload.scannerTables);
        equal_span({actual.s_inverse, 9}, payload.spdSInv);
        equal_span({actual.midgray_dwg_rgb, 3}, payload.filmRawConfig.midgrayDWG);
        equal_span({actual.raw_midgray, 3}, payload.filmRawConfig.rawMidgray);
        equal_bits(actual.raw_green, payload.filmRawConfig.rawMidgrayGreen);
        equal_bits(actual.midgray_scale, payload.filmRawConfig.midgrayScale);
        EXPECT_EQ(actual.raw_midgray_bgr, params.spectralUpsamplingMode == 1);
        if (payload.filmTcLut)
            equal_span(actual.film_exposure.tc_lut_rgba, payload.filmTcLut->samples());
        else
            EXPECT_EQ(actual.film_exposure.tc_lut_rgba.count, 0u);
        if (input.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019) {
            equal_span(actual.film_exposure.mallett_illuminant, payload.exposureTables.illum);
            equal_span(actual.film_exposure.mallett_basis, Spectral::gMallettBasis.data);
        } else {
            EXPECT_EQ(actual.film_exposure.mallett_illuminant.count, 0u);
            EXPECT_EQ(actual.film_exposure.mallett_basis.count, 0u);
        }
        EXPECT_EQ(actual.scanner_recipe_hash, recipe.scannerOutput.hash);
        const auto& scan = recipe.scannerOutput;
        const auto& scanPolicy = actual.scanner_policy;
        EXPECT_EQ(scanPolicy.black_correction, scan.blackCorrection);
        EXPECT_EQ(scanPolicy.white_correction, scan.whiteCorrection);
        EXPECT_EQ(scanPolicy.glare_active, scan.glareActive);
        equal_bits(scanPolicy.black_level, scan.blackLevel);
        equal_bits(scanPolicy.white_level, scan.whiteLevel);
        equal_bits(scanPolicy.glare_percent, scan.glarePercent);
        equal_bits(scanPolicy.glare_roughness, scan.glareRoughness);
        equal_bits(scanPolicy.glare_blur_sigma_px, scan.glareBlurSigmaPx);
        equal_bits(scanPolicy.lens_blur_sigma_px, scan.lensBlurSigmaPx);
        equal_bits(scanPolicy.unsharp_sigma_px, scan.unsharpSigmaPx);
        equal_bits(scanPolicy.unsharp_amount, scan.unsharpAmount);
        Scanner::ScannerSpectralLutDescriptor lut;
        ASSERT_TRUE(print ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &scan}, lut, diagnostic) : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &scan}, lut, diagnostic)) << diagnostic;
        EXPECT_EQ(actual.scanner_lut.hash, lut.hash);
        EXPECT_EQ(actual.scanner_lut.route, static_cast<unsigned>(lut.route));
        EXPECT_EQ(actual.scanner_lut.medium, static_cast<unsigned>(lut.medium));
        EXPECT_EQ(actual.scanner_lut.capture_polarity, static_cast<unsigned>(lut.polarity));
        EXPECT_EQ(actual.scanner_lut.resolution, lut.lutResolution);
        EXPECT_EQ(actual.scanner_lut.density_bounds_hash, lut.densityBoundsHash);
        EXPECT_EQ(actual.scanner_lut.channel_density_hash, lut.channelDensityHash);
        EXPECT_EQ(actual.scanner_lut.base_density_hash, lut.baseDensityHash);
        EXPECT_EQ(actual.scanner_lut.viewing_illuminant_hash, lut.scanIlluminantHash);
        EXPECT_EQ(actual.scanner_lut.observer_hash, lut.observerHash);
        Scanner::ScannerPostEffectsDescriptor post;
        ASSERT_TRUE(Scanner::build_scanner_post_effects_descriptor(scan, post, diagnostic)) << diagnostic;
        EXPECT_EQ(actual.scanner_post_effects.hash, post.hash);
        EXPECT_EQ(actual.scanner_post_effects.route, static_cast<unsigned>(post.route));
        EXPECT_EQ(actual.scanner_post_effects.glare_active, post.glareActive);
        equal_bits(actual.scanner_post_effects.glare_percent, post.glarePercent);
        equal_bits(actual.scanner_post_effects.glare_roughness, post.glareRoughness);
        equal_bits(actual.scanner_post_effects.glare_blur_sigma_px, post.glareBlurSigmaPx);
        equal_bits(actual.scanner_post_effects.lens_blur_sigma_px, post.lensBlurSigmaPx);
        equal_bits(actual.scanner_post_effects.unsharp_sigma_px, post.unsharpSigmaPx);
        equal_bits(actual.scanner_post_effects.unsharp_amount, post.unsharpAmount);
        EXPECT_EQ(actual.scanner_post_effects.glare_blur_radius, post.glareBlurRadius);
        EXPECT_EQ(actual.scanner_post_effects.lens_blur_radius, post.lensBlurRadius);
        EXPECT_EQ(actual.scanner_post_effects.unsharp_radius, post.unsharpRadius);
        EXPECT_EQ(actual.scanner_post_effects.schema, post.schemaVersion);
        EXPECT_EQ(actual.scanner_color.hash, payload.scannerColor.hash);
        equal_span({actual.scanner_color.cat02, 9}, payload.scannerColor.cat02);
        equal_span({actual.scanner_color.xyz_to_rgb, 9}, payload.scannerColor.xyzToRgb);
        equal_span({actual.scanner_color.illuminant_xyz, 3}, payload.scannerColor.illuminantXYZ);
        const auto& ref = recipe.scannerOutput.syntheticFilmReference;
        EXPECT_EQ(actual.film_reference.hash, ref.hash);
        equal_span({actual.film_reference.baseline_raw_rgb, 3}, ref.baselineRawRgb);
        equal_span({actual.film_reference.compensated_raw_rgb, 3}, ref.compensatedRawRgb);
        equal_span({actual.film_reference.baseline_density_cmy, 3}, ref.baselineDensityCmy);
        equal_span({actual.film_reference.compensated_density_cmy, 3}, ref.compensatedDensityCmy);
        equal_bits(actual.film_reference.meter_scale, ref.baselineMeterScale);
        if (payload.outputBoundaryTable) {
            equal_span(actual.output_boundary.cmax, payload.outputBoundaryTable->cmax);
            equal_span(actual.output_color.gamut_cmax, payload.outputBoundaryTable->cmax);
            EXPECT_EQ(actual.output_color.gamut_table_hash, payload.outputBoundaryTable->hash);
            EXPECT_EQ(actual.output_boundary.hash, payload.outputBoundaryTable->hash);
            EXPECT_EQ(actual.output_boundary.entry_identity, payload.outputBoundaryTable->entryIdentity);
            EXPECT_EQ(actual.output_boundary.cmax.data, payload.outputBoundaryTable->cmax.data());
            EXPECT_EQ(actual.output_boundary.transform_hash, payload.outputBoundaryTable->transformHash);
            EXPECT_EQ(actual.output_boundary.contract_hash, payload.outputBoundaryTable->contractHash);
        } else {
            EXPECT_EQ(actual.output_color.gamut_cmax.count, 0u);
            EXPECT_EQ(actual.output_color.gamut_table_hash, 0u);
        }
        EXPECT_EQ(actual.grain.hash, recipe.visualGrain.hash);
        EXPECT_EQ(actual.effects.hash, recipe.filmJuicerEffects.hash);
        EXPECT_EQ(actual.optics.hash, recipe.spatialOptics.hash);
        EXPECT_EQ(actual.print_recipe_hash, recipe.print.hash);
        Scanner::ScannerColorCorrectionDescriptor correction;
        if (print) {
            EXPECT_EQ(std::string(actual.print_profile_key.data, actual.print_profile_key.count), recipe.profileRoute.printProfileKey);
            equal_triplets(actual.print.density_cmy, recipe.print.develop.densityCurves);
            const auto& filmData = recipe.profileRoute.filmProfile->data;
            const auto& printData = recipe.profileRoute.printProfile->data;
            const auto calibration = JuicerProcess::root().assets().neutral_print_calibration(recipe.profileRoute.printProfile->info.stock, recipe.print.illuminant.key, recipe.profileRoute.filmProfile->info.stock);
            ASSERT_NE(calibration.status, JuicerAssets::NeutralPrintCalibrationStatus::Malformed);
            EXPECT_EQ(actual.calibration, static_cast<unsigned>(calibration.status));
            constexpr std::array<float, 3> kCurrentNeutral{0.0f, 65.0f, 55.0f};
            equal_span({actual.neutral_cmy_cc, 3}, calibration.status == JuicerAssets::NeutralPrintCalibrationStatus::Found ? calibration.cmyCc : kCurrentNeutral);
            const auto lights = JuicerProcess::root().assets().illuminant_filter_curves();
            std::span<const float> sourceLight;
            Spectral::Curve equalLight;
            const auto& key = recipe.print.illuminant.key;
            if (key == "D65")
                sourceLight = lights->d65.linear;
            else if (key == "D55")
                sourceLight = lights->d55.linear;
            else if (key == "D50")
                sourceLight = lights->d50.linear;
            else if (key == "T")
                sourceLight = lights->tungsten.linear;
            else if (key == "K75P")
                sourceLight = lights->kinoton75P.linear;
            else if (key == "TH-KG3")
                sourceLight = lights->tungstenKg3.linear;
            else if (key == "TH-KG3-L")
                sourceLight = lights->tungstenKg3Lens.linear;
            else if (key == "EQUAL") {
                equalLight = JuicerIlluminant::equal_energy();
                sourceLight = equalLight.linear;
            }
            ASSERT_EQ(sourceLight.size(), 81u) << key;
            equal_span(actual.print_source_illuminant, sourceLight);
            ASSERT_EQ(actual.print.film_density_cmy.count, 243u);
            EXPECT_EQ(std::memcmp(actual.print.film_density_cmy.data, filmData.channelDensity.data(), sizeof(filmData.channelDensity)), 0);
            equal_span(actual.print.film_base_density, filmData.baseDensity);
            ASSERT_EQ(actual.print.sensitivity_cmy.count, 243u);
            EXPECT_EQ(std::memcmp(actual.print.sensitivity_cmy.data, printData.linearSensitivity.data(), sizeof(printData.linearSensitivity)), 0);
            equal_span(actual.print.log_exposure, printData.logExposure);
            equal_bits(actual.print_gamma, recipe.print.develop.gammaFactor);
            EXPECT_EQ(actual.print_curves_hash, recipe.print.develop.densityCurvesHash);
            EXPECT_EQ(actual.print_filters_hash, recipe.print.filters.hash);
            EXPECT_EQ(actual.print_exposure_hash, recipe.print.exposure.hash);
            EXPECT_EQ(actual.print_illuminant_hash, recipe.print.illuminant.hash);
            equal_span(actual.print.main_illuminant, *payload.printMainIlluminant);
            const auto& balance = recipe.print.balance;
            equal_span({actual.print_balance.baseline_raw_rgb, 3}, balance.baselinePrintRawRgb);
            equal_span({actual.print_balance.compensated_raw_rgb, 3}, balance.compensatedPrintRawRgb);
            equal_bits(actual.print_balance.factor_midgray, balance.factorMidgray);
            equal_bits(actual.print_balance.factor_midgray_comp, balance.factorMidgrayComp);
            equal_bits(actual.print_balance.normalizer, balance.normalizer);
            EXPECT_EQ(actual.print_balance.filtered_main_illuminant_hash, balance.filteredMainIlluminantHash);
            EXPECT_EQ(actual.print_balance.hash, balance.hash);
            JuicerCuda::PrintResourceInput resources;
            std::array<float, 81> preflash{};
            ASSERT_TRUE(JuicerCuda::build_print_resource_input({&recipe, &JuicerProcess::root().assets(), &*payload.printMainIlluminant}, resources, preflash, diagnostic)) << diagnostic;
            const auto& descriptors = resources.descriptors;
            EXPECT_EQ(actual.print.hash, descriptors.hash);
            EXPECT_EQ(actual.print.profile_tables_hash, descriptors.profileTables.hash);
            EXPECT_EQ(actual.print.film_density_hash, descriptors.filmDensityTables.hash);
            EXPECT_EQ(actual.print.main_illuminant_hash, descriptors.mainIlluminant.hash);
            EXPECT_EQ(actual.print.preflash_illuminant_hash, descriptors.preflashIlluminant.hash);
            EXPECT_EQ(actual.print.preflash_raw_hash, descriptors.preflashRaw.hash);
            EXPECT_EQ(actual.print.balance_hash, descriptors.balance.hash);
            EXPECT_EQ(actual.print.flags, descriptors.preflashActive ? FJ_PRINT_PREFLASH : 0u);
            if (descriptors.preflashActive) {
                equal_span(actual.print.preflash_illuminant, resources.preflashIlluminant);
            } else {
                // The native scratch span is unconsumed when preflash is off.
                EXPECT_EQ(actual.print.preflash_illuminant.count, 0u);
                EXPECT_EQ(actual.print.preflash_illuminant.data, nullptr);
            }
            equal_span({actual.print.preflash_raw_cmy, 3}, resources.preflashRawCmy);
            equal_bits(actual.print.normalizer, resources.normalizer);
            const auto& filters = recipe.print.filters;
            const std::array<float, 3> mainCc{filters.mainCmyCc.c, filters.mainCmyCc.m, filters.mainCmyCc.y};
            const std::array<float, 3> preflashCc{filters.preflashCmyCc.c, filters.preflashCmyCc.m, filters.preflashCmyCc.y};
            equal_span({actual.print_filter_main_cmy_cc, 3}, mainCc);
            equal_span({actual.print_filter_preflash_cmy_cc, 3}, preflashCc);
            equal_span({actual.print.main_cc_cmy, 3}, mainCc);
            equal_span({actual.print.preflash_cc_cmy, 3}, descriptors.preflashActive ? preflashCc : std::array<float, 3>{});
            equal_bits(actual.print.exposure, recipe.print.exposure.printExposure);
            equal_bits(actual.print.preflash_exposure, recipe.print.exposure.preflashExposure);
            EXPECT_EQ(actual.print_normalization_mode, static_cast<unsigned>(recipe.print.exposure.normalizationMode));
            EXPECT_EQ(actual.normalize_print_exposure, recipe.print.exposure.normalizePrintExposure);
            EXPECT_EQ(actual.print_exposure_compensation, recipe.print.exposure.printExposureCompensation);
            ASSERT_TRUE(Scanner::build_print_scanner_color_correction_descriptor({&recipe, &payload.scannerTables, payload.printMainIlluminant->data(), 81, resources.preflashRawCmy.data(), resources.normalizer}, correction, diagnostic)) << diagnostic;
        } else {
            ASSERT_TRUE(Scanner::build_direct_scanner_color_correction_descriptor(recipe, payload.scannerTables, correction, diagnostic)) << diagnostic;
        }
        const auto& wireCorrection = actual.scanner_correction;
        EXPECT_EQ(wireCorrection.route, static_cast<unsigned>(correction.route));
        EXPECT_EQ(wireCorrection.active, correction.active);
        EXPECT_EQ(wireCorrection.black_correction, correction.blackCorrection);
        EXPECT_EQ(wireCorrection.white_correction, correction.whiteCorrection);
        equal_bits(wireCorrection.target_black_linear, correction.targetBlackLinear);
        equal_bits(wireCorrection.target_white_linear, correction.targetWhiteLinear);
        equal_bits(wireCorrection.reference_black_y, correction.referenceBlackY);
        equal_bits(wireCorrection.reference_white_y, correction.referenceWhiteY);
        equal_bits(wireCorrection.xyz_slope, correction.xyzSlope);
        equal_bits(wireCorrection.xyz_offset, correction.xyzOffset);
        equal_bits(wireCorrection.exposure_scale, correction.exposureScale);
        EXPECT_EQ(wireCorrection.schema_version, correction.schemaVersion);
        EXPECT_EQ(wireCorrection.hash, correction.hash);
        equal_bits(actual.film_exposure.route_correction_scale, print ? 1.0f : correction.exposureScale);
        PreparedStateOracle::NativeBacking backing;
        FjPreparedStateView nativeView{};
        ASSERT_TRUE(PreparedStateOracle::SF_TEMP_BRIDGE_native_prepared_view(expected, params, nativeView, backing, diagnostic)) << diagnostic;
        record_projection(params, nativeView, actual);
    }

    void baseline(bool positive, bool print, int method) {
        ParamSnapshot params;
        params.filmProfileKey = positive ? "fujifilm_provia_100f" : "kodak_portra_400";
        params.printProfileKey = "kodak_portra_endura";
        params.scanRoute = positive ? (print ? Spektrafilm::ScanRoute::PositivePrintScan : Spektrafilm::ScanRoute::PositiveDirectScan) : (print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan);
        params.spectralUpsamplingMode = method;
        params.grainControls.active = false;
        compare_product(params);
    }

#define FJ_STATE_BASELINE(name, positive, print, method) \
    TEST(PreparedStateProducts, name) {                  \
        baseline(positive, print, method);               \
    }
    FJ_STATE_BASELINE(NegativeDirectHanatos, false, false, 0)
    FJ_STATE_BASELINE(NegativeDirectMallett, false, false, 1)
    FJ_STATE_BASELINE(NegativeDirectArctic, false, false, 2)
    FJ_STATE_BASELINE(NegativePrintHanatos, false, true, 0)
    FJ_STATE_BASELINE(NegativePrintMallett, false, true, 1)
    FJ_STATE_BASELINE(NegativePrintArctic, false, true, 2)
    FJ_STATE_BASELINE(PositiveDirectHanatos, true, false, 0)
    FJ_STATE_BASELINE(PositiveDirectMallett, true, false, 1)
    FJ_STATE_BASELINE(PositiveDirectArctic, true, false, 2)
    FJ_STATE_BASELINE(PositivePrintHanatos, true, true, 0)
    FJ_STATE_BASELINE(PositivePrintMallett, true, true, 1)
    FJ_STATE_BASELINE(PositivePrintArctic, true, true, 2)
#undef FJ_STATE_BASELINE

    ParamSnapshot print_params() {
        ParamSnapshot params;
        params.filmProfileKey = "kodak_portra_400";
        params.printProfileKey = "kodak_portra_endura";
        params.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
        params.spectralUpsamplingMode = 1;
        params.grainControls.active = false;
        return params;
    }

    TEST(PreparedStateProducts, OracleReaderRejectsFieldShapeWidthTagAndBackingMutations) {
        const auto params = print_params();
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        ASSERT_TRUE(build_print_render_state_product(params, product, diagnostic)) << diagnostic;
        PreparedStateOracle::NativeBacking backing;
        FjPreparedStateView native{};
        ASSERT_TRUE(PreparedStateOracle::SF_TEMP_BRIDGE_native_prepared_view(product, params, native, backing, diagnostic)) << diagnostic;
        const auto expected = PreparedStateOracle::view(native);
        FjStatus status{};
        auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        FjPreparedStateView actual{};
        ASSERT_EQ(state_view(owner, actual).category, FJ_STATUS_SUCCESS);
        const auto current = PreparedStateOracle::view(actual);
        ASSERT_NO_THROW(PreparedStateOracle::compare(expected, current));
        auto damaged = expected;
        damaged.at("grain").erase("amplitude");
        EXPECT_THROW(PreparedStateOracle::compare(damaged, current), std::runtime_error);
        damaged = expected;
        damaged.at("output_color")["unaccounted"] = 0;
        EXPECT_THROW(PreparedStateOracle::compare(damaged, current), std::runtime_error);
        damaged = expected;
        damaged.at("film_exposure").at("sensitivity_rgb")["shape"] = Json::array({81});
        EXPECT_THROW(PreparedStateOracle::compare(damaged, current), std::runtime_error);
        damaged = expected;
        damaged.at("print_gamma")["bits"] = "00000000";
        EXPECT_THROW(PreparedStateOracle::compare(damaged, current), std::runtime_error);
        damaged = expected;
        damaged.at("route")["value"] = 4;
        EXPECT_THROW(PreparedStateOracle::compare(damaged, current), std::runtime_error);
        damaged = expected;
        ASSERT_NE(native.output_boundary.cmax.count, 0u);
        damaged.at("output_boundary").at("entry_identity")["relation"] = "unrelated_backing";
        EXPECT_THROW(PreparedStateOracle::compare(damaged, current), std::runtime_error);
        const Json positiveZero{{"type", "f32"}, {"bits", "00000000"}};
        const Json negativeZero{{"type", "f32"}, {"bits", "80000000"}};
        EXPECT_THROW(PreparedStateOracle::compare(positiveZero, negativeZero), std::runtime_error);
        const Json edge = Json::array({backing_edge("native.input_hull.xy", "actual.input_hull.xy", true)});
        auto changedEdge = edge;
        changedEdge.at(0)["equal"] = false;
        EXPECT_THROW(compare_backing(changedEdge, edge), std::runtime_error);
        // Membership literals test the reader, never provide scientific products.
        Json membership = Json::array();
        for (const auto name : kProjectionCases)
            membership.push_back({{"case_id", std::string(name)}});
        ASSERT_NO_THROW(require_projection_membership(membership));
        auto wrongMembership = membership;
        wrongMembership.erase(wrongMembership.begin());
        EXPECT_THROW(require_projection_membership(wrongMembership), std::runtime_error);
        wrongMembership = membership;
        wrongMembership.at(1) = wrongMembership.at(0);
        EXPECT_THROW(require_projection_membership(wrongMembership), std::runtime_error);
        EXPECT_THROW(exact_keys(Json{{"schema_version", 1}, {"unknown", 0}}, {"schema_version"}), std::runtime_error);
        const Json inputs{{"resource_directory", std::string("out/build/") + FJ_PREPARED_STATE_PRESET + "/tests/Resources"}, {"parameter_type", "FjPreparationParams"}};
        ASSERT_NO_THROW(require_projection_inputs(inputs, FJ_PREPARED_STATE_PRESET));
        auto wrongInputs = inputs;
        wrongInputs["resource_directory"] = "out/build/foreign/tests/Resources";
        EXPECT_THROW(require_projection_inputs(wrongInputs, FJ_PREPARED_STATE_PRESET), std::runtime_error);
        std::filesystem::create_directories(artifactRoot);
        const auto duplicatePath = artifactRoot / "reader-duplicate.json";
        {
            std::ofstream duplicate(duplicatePath, std::ios::binary);
            duplicate << "{\"schema_version\":1,\"schema_version\":1}";
        }
        EXPECT_THROW((void)read_projection(duplicatePath, FJ_PREPARED_STATE_PRESET), std::runtime_error);
    }

    TEST(PreparedStateProducts, DirectIgnoresPrintControls) {
        auto params = print_params();
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        params.printProfileKey = "__absent_print";
        params.printGammaFactor = std::numeric_limits<double>::quiet_NaN();
        params.printExposure = std::numeric_limits<double>::infinity();
        params.glarePercent = std::numeric_limits<double>::quiet_NaN();
        compare_product(params);
    }

    TEST(PreparedStateProducts, PreflashAndNormalizationModes) {
        for (int normalize = 0; normalize < 2; ++normalize)
            for (int compensate = 0; compensate < 2; ++compensate) {
                SCOPED_TRACE(::testing::Message() << normalize << '/' << compensate);
                auto params = print_params();
                params.normalizePrintExposure = normalize;
                params.printExposureCompensation = compensate;
                params.printPreflashExposure = 0.1;
                params.printExposure = 1.2;
                params.cameraExposureCompensationEv = 0.75;
                params.preflashMFilterCc = 2.5;
                params.preflashYFilterCc = -4.0;
                params.printUiYmcCc = {3.0, -2.0, 1.0};
                params.enlIll = 7;
                compare_product(params);
            }
    }

    TEST(PreparedStateProducts, GammaAndGrainLayers) {
        auto params = print_params();
        params.grainControls.active = true;
        params.grainControls.sublayersActive = true;
        params.filmGammaFactor = 1.125f;
        params.printGammaFactor = 0.875;
        compare_product(params);
    }

    TEST(PreparedStateProducts, PhysicalDefectsAndWeave) {
        auto params = print_params();
        params.filmDustAmount = 0.3f;
        params.filmScratchAmount = 0.2f;
        params.gateDustAmount = 0.1f;
        params.gateScratchAmount = 0.4f;
        params.gateWeaveAmount = 0.5;
        compare_product(params);
    }

    TEST(PreparedStateProducts, CameraAndEnlargerOptics) {
        auto params = print_params();
        params.cameraDiffusion.active = true;
        params.enlargerDiffusion.active = true;
        params.scatterHalationControls.active = true;
        compare_product(params);
    }

    TEST(PreparedStateProducts, DirAndCompressionOff) {
        auto params = print_params();
        params.dirCouplers.active = false;
        params.inputCompressionEnabled = 0;
        params.outputGamutCompressionEnabled = 0;
        compare_product(params);
    }

    TEST(PreparedStateProducts, SignedZeroGlare) {
        auto params = print_params();
        params.glarePercent = 0.0;
        compare_product(params);
        params.glarePercent = -0.0;
        compare_product(params);
    }

    TEST(PreparedStateProducts, CorrectionsAndScannerControls) {
        auto params = print_params();
        params.scannerBlackCorrection = 1;
        params.scannerWhiteCorrection = 1;
        params.scannerLensBlurSigmaPx = 0.4;
        params.scannerUnsharpMask = {0.8, 0.6};
        params.outputColorSpace = 5;
        params.outputCctfEncoding = 0;
        compare_product(params);
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        compare_product(params);
    }

    void equal_status(FjStatus actual, std::uint32_t category) {
        EXPECT_EQ(actual.category, category);
        EXPECT_EQ(actual.api, FJ_API_NONE);
        EXPECT_EQ(actual.native_code, 0);
    }

    void equal_preparation_trace(std::initializer_list<std::uint32_t> expected) {
        const auto trace = fj_test_prepared_trace();
        if (trace.count <= std::size(trace.stages)) {
            EXPECT_EQ(std::vector<std::uint32_t>(trace.stages, trace.stages + trace.count), std::vector<std::uint32_t>(expected));
            for (std::size_t i = trace.count; i < std::size(trace.stages); ++i)
                EXPECT_EQ(trace.stages[i], 0u);
        } else {
            ADD_FAILURE() << "invalid preparation trace count " << trace.count;
        }
    }

    void competing_precheck(const ParamSnapshot& params, const char* expectedDiagnostic, std::initializer_list<std::uint32_t> expectedTrace) {
        struct ClearExposureFault {
            ~ClearExposureFault() {
                fj_test_exposure_clear_fault();
            }
        } clearExposureFault;
        const bool print = Spektrafilm::scan_route_is_print(params.scanRoute);
        auto repaired = print_params();
        repaired.scanRoute = params.scanRoute;
        const auto nativeBuild = [&](const ParamSnapshot& input, FocusedRenderStateBuildProduct& result, std::string& diagnostic) {
            return print ? build_print_render_state_product(input, result, diagnostic) : build_direct_render_state_product(input, result, diagnostic);
        };
        for (bool cold : {true, false}) {
            SCOPED_TRACE(cold ? "cold" : "warm");
            if (cold)
                equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            equal_status(fj_test_exposure_arm_fault(5, 1, 1), FJ_STATUS_SUCCESS);
            FocusedRenderStateBuildProduct native;
            std::string nativeDiagnostic;
            ASSERT_FALSE(nativeBuild(params, native, nativeDiagnostic));
            EXPECT_EQ(nativeDiagnostic, expectedDiagnostic);
            const auto earlierDiagnostic = nativeDiagnostic;
            EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 0u);
            try {
                (void)nativeBuild(repaired, native, nativeDiagnostic);
                FAIL() << "native reference source fault remained unconsumed";
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                equal_status(failure.failure.status, FJ_STATUS_UNSUPPORTED_INPUT);
                EXPECT_EQ(failure.failure.diagnostic, "Reference source preparation: exposure production boundary fault");
            }
            EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 1u);
            if (cold)
                equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            equal_status(fj_test_exposure_arm_fault(5, 1, 1), FJ_STATUS_SUCCESS);
            FjStatus status{};
            std::string diagnostic;
            auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
            equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_FALSE(owner.has_value());
            EXPECT_EQ(diagnostic, earlierDiagnostic);
            equal_preparation_trace(expectedTrace);
            EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 0u);
            owner = JuicerCudaTest::prepare_state(assets(), repaired, status, diagnostic);
            equal_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_FALSE(owner.has_value());
            EXPECT_EQ(diagnostic, "Reference source preparation: exposure production boundary fault");
            EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 1u);
        }
    }

    TEST(PreparedStateProducts, DirectScannerBeforeInputPolicy) {
        auto params = print_params();
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        params.inputColorSpace = -1;
        params.scannerBlackLevel = std::numeric_limits<double>::quiet_NaN();
        competing_precheck(params, "ResourceDescriptorMismatch phase=8 field=scanner_output", {FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_ROUTE_PRECHECKS});
    }

    TEST(PreparedStateProducts, PrintGammaBeforeInputPolicy) {
        auto params = print_params();
        params.inputColorSpace = -1;
        params.printGammaFactor = 0.1;
        competing_precheck(params, "InvalidAuthoredControl component=print_development field=print_gamma_factor", {FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_CALIBRATION, FJ_PREPARE_ROUTE_PRECHECKS});
    }

    TEST(PreparedStateProducts, PrintInputBeforeInputPolicy) {
        auto params = print_params();
        params.inputColorSpace = -1;
        params.printExposure = std::numeric_limits<double>::infinity();
        competing_precheck(params, "ResourceDescriptorMismatch phase=4A field=print_recipe_input", {FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_CALIBRATION, FJ_PREPARE_ROUTE_PRECHECKS});
    }

    TEST(PreparedStateProducts, DevelopmentBeforeCalibration) {
        struct ClearCalibrationFault {
            ~ClearCalibrationFault() {
                fj_test_calibration_read_fault(0);
            }
        } clearCalibrationFault;
        auto params = print_params();
        params.filmGammaFactor = std::numeric_limits<float>::quiet_NaN();
        auto repaired = params;
        repaired.filmGammaFactor = 1.0f;
        constexpr const char* kGammaDiagnostic = "InvalidAuthoredControl component=film_development field=film_gamma_factor";
        for (bool cold : {true, false}) {
            SCOPED_TRACE(cold ? "cold" : "warm");
            equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            FocusedRenderStateBuildProduct native;
            std::string nativeDiagnostic;
            if (!cold) {
                ASSERT_FALSE(build_print_render_state_product(params, native, nativeDiagnostic));
            }
            fj_test_calibration_read_fault(1);
            ASSERT_FALSE(build_print_render_state_product(params, native, nativeDiagnostic));
            EXPECT_EQ(nativeDiagnostic, kGammaDiagnostic);
            EXPECT_EQ(fj_test_calibration_reads(), 0u);
            EXPECT_THROW((void)build_print_render_state_product(repaired, native, nativeDiagnostic), std::bad_alloc);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            constexpr char kKey[] = "pending";
            const FjStringView selection{kKey, sizeof(kKey) - 1};
            std::array<char, 1024> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            FjNeutralCalibrationResult raw{};
            equal_status(fj_legacy_neutral_calibration_lookup(assets(), selection, selection, selection, &raw, &error), FJ_STATUS_ALLOCATION_FAILURE);
            const std::string dependencyDiagnostic(bytes.data(), error.length);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            fj_test_calibration_read_fault(0);
            equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
            FjStatus status{};
            std::string diagnostic;
            if (!cold) {
                auto warm = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
                EXPECT_FALSE(warm.has_value());
                equal_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
            }
            fj_test_calibration_read_fault(1);
            auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
            equal_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_FALSE(owner.has_value());
            EXPECT_EQ(diagnostic, kGammaDiagnostic);
            equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT});
            EXPECT_EQ(fj_test_calibration_reads(), 0u);
            owner = JuicerCudaTest::prepare_state(assets(), repaired, status, diagnostic);
            equal_status(status, FJ_STATUS_ALLOCATION_FAILURE);
            EXPECT_FALSE(owner.has_value());
            EXPECT_EQ(diagnostic, dependencyDiagnostic);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            fj_test_calibration_read_fault(0);
        }
    }

    TEST(PreparedStateProducts, InputAllocationBeforeScannerControl) {
        struct ClearCurveFault {
            ~ClearCurveFault() {
                fj_test_illuminant_clear_fault();
                (void)fj_test_assets_curve_fault(assets(), FJ_CSV_D65, 0, nullptr);
            }
        } clearCurveFault;
        auto params = print_params();
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        params.scannerBlackLevel = std::numeric_limits<double>::quiet_NaN();
        equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
        equal_status(fj_test_assets_curve_fault(assets(), FJ_CSV_D65, 1, nullptr), FJ_STATUS_SUCCESS);
        FocusedRenderStateBuildProduct native;
        std::string nativeDiagnostic;
        FjStatus nativeStatus{};
        try {
            (void)build_direct_render_state_product(params, native, nativeDiagnostic);
            FAIL() << "native cold illuminant reservation fault was not observed";
        } catch (const JuicerCuda::ExecutionFailure& failure) {
            nativeStatus = failure.failure.status;
            nativeDiagnostic = failure.failure.diagnostic;
        }
        equal_status(nativeStatus, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_EQ(nativeDiagnostic, "Illuminant preparation (D65): illuminant scratch capacity");
        const auto dependencyDiagnostic = nativeDiagnostic;
        ASSERT_FALSE(build_direct_render_state_product(params, native, nativeDiagnostic));
        EXPECT_EQ(nativeDiagnostic, "ResourceDescriptorMismatch phase=8 field=scanner_output");
        equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
        equal_status(fj_test_assets_curve_fault(assets(), FJ_CSV_D65, 1, nullptr), FJ_STATUS_SUCCESS);
        FjStatus status{};
        std::string diagnostic;
        auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, nativeStatus.category);
        EXPECT_EQ(diagnostic, dependencyDiagnostic);
        EXPECT_FALSE(owner.has_value());
        equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_ILLUMINANTS});
        std::uint32_t pending = 99;
        equal_status(fj_test_assets_curve_fault_pending(assets(), &pending, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(pending, 0u);
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(diagnostic, nativeDiagnostic);
        EXPECT_FALSE(owner.has_value());
        equal_status(fj_test_assets_curve_fault(assets(), FJ_CSV_D65, 1, nullptr), FJ_STATUS_SUCCESS);
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(diagnostic, nativeDiagnostic);
        equal_status(fj_test_assets_curve_fault_pending(assets(), &pending, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(pending, 1u) << "warm entry must skip the derived acquisition";
        equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_EQ(diagnostic, dependencyDiagnostic);
        equal_status(fj_test_assets_curve_fault_pending(assets(), &pending, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(pending, 0u);
    }

    TEST(PreparedStateProducts, CalibrationBeforePrintGamma) {
        struct ClearExposureFault {
            ~ClearExposureFault() {
                fj_test_exposure_clear_fault();
            }
        } clearExposureFault;
        const auto sourceRoot = artifactRoot / "malformed-calibration-input";
        std::filesystem::create_directories(sourceRoot / "filters");
        {
            std::ofstream file(sourceRoot / "filters/neutral_print_filters.json", std::ios::binary);
            file << R"({"kodak_portra_endura":{"TH-KG3":{"kodak_portra_400":[1,2]}}})";
            ASSERT_TRUE(file.good());
        }
        const JuicerAssets::NativePathArgument path(sourceRoot);
        FjAssets* rawSource = nullptr;
        equal_status(fj_legacy_assets_create(path.view(), &rawSource, nullptr), FJ_STATUS_SUCCESS);
        ASSERT_NE(rawSource, nullptr);
        const auto destroy = [](FjAssets* owner) {
            (void)fj_legacy_assets_destroy(owner, nullptr);
        };
        std::unique_ptr<FjAssets, decltype(destroy)> source(rawSource, destroy);
        auto params = print_params();
        params.printGammaFactor = 0.1;
        params.inputColorSpace = -1;
        constexpr const char* kExpectedDiagnostic = "MalformedNeutralPrintCalibration phase=4A field=cmy_cc";
        for (bool cold : {true, false}) {
            for (bool nativeActor : {true, false}) {
                SCOPED_TRACE(::testing::Message() << cold << '/' << nativeActor);
                equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
                FocusedRenderStateBuildProduct native;
                std::string diagnostic;
                FjStatus status{};
                if (!cold) {
                    auto warm = print_params();
                    warm.filmGammaFactor = std::numeric_limits<float>::quiet_NaN();
                    if (nativeActor) {
                        ASSERT_FALSE(build_print_render_state_product(warm, native, diagnostic));
                    } else {
                        auto ignored = JuicerCudaTest::prepare_state(assets(), warm, status, diagnostic);
                        EXPECT_FALSE(ignored.has_value());
                        equal_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
                    }
                }
                equal_status(fj_test_assets_calibration_source(assets(), source.get(), nullptr), FJ_STATUS_SUCCESS);
                equal_status(fj_test_exposure_arm_fault(5, 1, 1), FJ_STATUS_SUCCESS);
                if (nativeActor) {
                    ASSERT_FALSE(build_print_render_state_product(params, native, diagnostic));
                } else {
                    auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
                    EXPECT_FALSE(owner.has_value());
                    equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
                    equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_CALIBRATION});
                }
                EXPECT_EQ(diagnostic, kExpectedDiagnostic);
                EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 0u);
                equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
                if (nativeActor) {
                    try {
                        (void)build_print_render_state_product(print_params(), native, diagnostic);
                        FAIL() << "native later reference fault was consumed early";
                    } catch (const JuicerCuda::ExecutionFailure& failure) {
                        equal_status(failure.failure.status, FJ_STATUS_UNSUPPORTED_INPUT);
                        EXPECT_EQ(failure.failure.diagnostic, "Reference source preparation: exposure production boundary fault");
                    }
                } else {
                    auto owner = JuicerCudaTest::prepare_state(assets(), print_params(), status, diagnostic);
                    EXPECT_FALSE(owner.has_value());
                    equal_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
                    EXPECT_EQ(diagnostic, "Reference source preparation: exposure production boundary fault");
                }
                EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 1u);
            }
        }
    }

    TEST(PreparedStateProducts, EncodingOnlyReusesMediumIdentities) {
        auto params = print_params();
        compare_product(params);
        FjStatus status{};
        std::string diagnostic;
        auto encoded = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        FjPreparedStateView first{}, second{};
        equal_status(state_view(encoded, first), FJ_STATUS_SUCCESS);
        params.outputCctfEncoding = 0;
        compare_product(params);
        auto linear = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        equal_status(state_view(linear, second), FJ_STATUS_SUCCESS);
        EXPECT_NE(first.recipe_hash, second.recipe_hash);
        EXPECT_NE(first.scanner_color.hash, second.scanner_color.hash);
        EXPECT_EQ(first.scanner_lut.hash, second.scanner_lut.hash);
        EXPECT_EQ(first.scanner_tables.hash, second.scanner_tables.hash);
        EXPECT_EQ(first.scanner_hash, second.scanner_hash);
        EXPECT_EQ(first.print.hash, second.print.hash);
        EXPECT_EQ(first.output_boundary.entry_identity, second.output_boundary.entry_identity);
        observe_encoding_relation(first.output_boundary.entry_identity == second.output_boundary.entry_identity);
    }

    TEST(PreparedStateProducts, SnapshotIntegerBooleansPreserveNonzeroCasts) {
        for (bool print : {false, true}) {
            auto params = print ? print_params() : ParamSnapshot{};
            if (!print)
                params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
            params.normalizePrintExposure = -3;
            params.printExposureCompensation = 2;
            params.inputCctfDecoding = -3;
            params.inputCompressionEnabled = 2;
            params.hanatos2025AdaptationWindow = -3;
            params.hanatos2025AdaptationSurface = 2;
            params.cameraAutoExposureEnabled = -3;
            params.scannerBlackCorrection = 2;
            params.scannerWhiteCorrection = -3;
            params.outputCctfEncoding = 2;
            params.outputGamutCompressionEnabled = -3;
            compare_product(params);
        }
    }

    TEST(PreparedStateProducts, DisabledFamiliesExcludeUnusedControls) {
        auto params = print_params();
        params.dirCouplers.active = false;
        FjStatus status{};
        std::string diagnostic;
        auto baseline = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        FjPreparedStateView first{}, second{};
        equal_status(state_view(baseline, first), FJ_STATUS_SUCCESS);
        params.cameraDiffusion.strength = 0.25;
        params.enlargerDiffusion.spatialScale = 2.0;
        params.scatterHalationControls.scatterAmount = std::numeric_limits<float>::quiet_NaN();
        params.dirCouplers.amount = std::numeric_limits<float>::quiet_NaN();
        params.grainControls.amplitude = std::numeric_limits<float>::quiet_NaN();
        compare_product(params);
        auto changed = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        equal_status(state_view(changed, second), FJ_STATUS_SUCCESS);
        EXPECT_EQ(first.recipe_hash, second.recipe_hash);
        EXPECT_EQ(first.optics.hash, 0u);
        EXPECT_EQ(second.optics.hash, 0u);
        EXPECT_EQ(second.grain.hash, 0u);
        EXPECT_EQ(second.dir_recipe.hash, 0u);
    }

    TEST(PreparedStateProducts, GrainDebugSuppressesPhysicalEffects) {
        auto params = print_params();
        params.grainControls.active = true;
        params.grainControls.debugView = 1;
        params.filmDustAmount = 0.5f;
        params.gateScratchAmount = 0.5f;
        compare_product(params);
        FjStatus status{};
        std::string diagnostic;
        auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        FjPreparedStateView view{};
        equal_status(state_view(owner, view), FJ_STATUS_SUCCESS);
        EXPECT_NE(view.grain.hash, 0u);
        EXPECT_EQ(view.effects.hash, 0u);
        EXPECT_EQ(view.effects.active, 0u);
    }

    TEST(PreparedStateProducts, EarlyAndLatePanicsReturnNoOwner) {
        struct ClearFaults {
            FjAssets* owner;
            ~ClearFaults() {
                fj_test_exposure_clear_fault();
                (void)fj_test_assets_curve_fault(owner, FJ_CSV_D55, 0, nullptr);
            }
        } clearFaults{assets()};
        const auto baseline = fj_test_prepared_live_states();
        const auto params = print_params();
        equal_status(JuicerProcess::root().assets().release_cached_payloads(), FJ_STATUS_SUCCESS);
        equal_status(fj_test_assets_curve_fault(assets(), FJ_CSV_D55, 2, nullptr), FJ_STATUS_SUCCESS);
        FjStatus status{};
        std::string diagnostic;
        auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_EQ(diagnostic, "asset boundary panic");
        EXPECT_FALSE(owner.has_value());
        EXPECT_EQ(fj_test_prepared_live_states(), baseline);
        equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS});
        equal_status(fj_test_exposure_arm_fault(5, 1, 2), FJ_STATUS_SUCCESS);
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_EQ(diagnostic, "asset boundary panic");
        EXPECT_FALSE(owner.has_value());
        EXPECT_EQ(fj_test_exposure_fault_consumed(5, 2), 1u);
        EXPECT_EQ(fj_test_prepared_live_states(), baseline);
        equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_CALIBRATION, FJ_PREPARE_ROUTE_PRECHECKS, FJ_PREPARE_FOUNDATION, FJ_PREPARE_FILM_PAYLOAD, FJ_PREPARE_PRINT_MAIN_ILLUMINANT, FJ_PREPARE_FILM_REFERENCE});
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_SUCCESS);
        EXPECT_TRUE(owner.has_value());
        EXPECT_TRUE(diagnostic.empty());
    }

    TEST(PreparedStateProducts, ReservedZeroIdentitiesRejectCompletion) {
        struct ClearIdentityFault {
            ~ClearIdentityFault() {
                (void)fj_test_prepared_identity_fault(0);
            }
        } clearIdentityFault;
        const auto baseline = fj_test_prepared_live_states();
        for (bool print : {false, true}) {
            for (std::uint32_t target : {1u, 2u}) {
                SCOPED_TRACE(::testing::Message() << print << '/' << target);
                auto params = print_params();
                if (!print)
                    params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
                equal_status(fj_test_prepared_identity_fault(target), FJ_STATUS_SUCCESS);
                FjStatus status{};
                std::string diagnostic;
                auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
                equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
                EXPECT_FALSE(owner.has_value());
                EXPECT_EQ(diagnostic, print ? "ResourceDescriptorMismatch phase=4C focused print publication payload build failed" : "ResourceDescriptorMismatch focused direct publication payload build failed");
                EXPECT_EQ(fj_test_prepared_live_states(), baseline);
                const auto trace = fj_test_prepared_trace();
                if (trace.count > 0 && trace.count <= std::size(trace.stages)) {
                    EXPECT_EQ(trace.stages[trace.count - 1], FJ_PREPARE_FINAL_IDENTITY);
                } else {
                    ADD_FAILURE() << "invalid preparation trace count " << trace.count;
                }
                owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
                equal_status(status, FJ_STATUS_SUCCESS);
                EXPECT_TRUE(owner.has_value());
                EXPECT_TRUE(diagnostic.empty());
            }
        }
    }

    // Exhaustive bindings make added ABI fields require an explicit clearing reader.
    auto cleared_fields(const FjFilmDensityBounds& value) {
        const auto& [route, medium, polarity, source, minimum_cmy, maximum_cmy, inv_span_cmy, authored_minimum_cmy, authored_maximum_cmy, hash] = value;
        return std::tie(route, medium, polarity, source, minimum_cmy, maximum_cmy, inv_span_cmy, authored_minimum_cmy, authored_maximum_cmy, hash);
    }

    auto cleared_fields(const FjFilmDensityMinimum& value) {
        const auto& [cmy] = value;
        return std::tie(cmy);
    }

    auto cleared_fields(const FjFloatSpan& value) {
        const auto& [data, count] = value;
        return std::tie(data, count);
    }

    auto cleared_fields(const FjFilmDevelopment& value) {
        const auto& [log_exposure, density_rgb, density_layers, gamma_rgb, density_curves_hash, density_layers_hash, hash] = value;
        return std::tie(log_exposure, density_rgb, density_layers, gamma_rgb, density_curves_hash, density_layers_hash, hash);
    }

    auto cleared_fields(const FjFilmDirView& value) {
        const auto& [polarity, mode, matrix_rgb, diffusion_size_um, diffusion_tail_um, diffusion_tail_weight, density_max_rgb, density_ref_rgb, donor_k_rgb, receiver_c_ref_rgb, receiver_kr_rgb, compensated_axes_rgb, compensated_axes_hash, hash] = value;
        return std::tie(polarity, mode, matrix_rgb, diffusion_size_um, diffusion_tail_um, diffusion_tail_weight, density_max_rgb, density_ref_rgb, donor_k_rgb, receiver_c_ref_rgb, receiver_kr_rgb, compensated_axes_rgb, compensated_axes_hash, hash);
    }

    auto cleared_fields(const FjFilmExposure& value) {
        const auto& [input_color_space, method, flags, manual_exposure_ev, route_correction_scale, mallett_green_midgray_scale, input_rgb_to_xyz, input_xyz_adapt, xyz_to_linear_srgb, sensitivity_rgb, tc_lut_rgba, mallett_illuminant, mallett_basis, sensitivity_hash, tc_lut_hash, hash] = value;
        return std::tie(input_color_space, method, flags, manual_exposure_ev, route_correction_scale, mallett_green_midgray_scale, input_rgb_to_xyz, input_xyz_adapt, xyz_to_linear_srgb, sensitivity_rgb, tc_lut_rgba, mallett_illuminant, mallett_basis, sensitivity_hash, tc_lut_hash, hash);
    }

    auto cleared_fields(const FjFilmReference& value) {
        const auto& [baseline_raw_rgb, compensated_raw_rgb, baseline_density_cmy, compensated_density_cmy, meter_scale, hash] = value;
        return std::tie(baseline_raw_rgb, compensated_raw_rgb, baseline_density_cmy, compensated_density_cmy, meter_scale, hash);
    }

    auto cleared_fields(const FjInputHullView& value) {
        const auto& [center_xy, xy, hash, entry_identity] = value;
        return std::tie(center_xy, xy, hash, entry_identity);
    }

    auto cleared_fields(const FjOutputBoundaryView& value) {
        const auto& [cmax, transform_hash, contract_hash, hash, entry_identity] = value;
        return std::tie(cmax, transform_hash, contract_hash, hash, entry_identity);
    }

    auto cleared_fields(const FjOutputEncoding& value) {
        const auto& [schema, cctf_kind, gamma, a, b, c, d, linear_cutoff, dwg_to_output] = value;
        return std::tie(schema, cctf_kind, gamma, a, b, c, d, linear_cutoff, dwg_to_output);
    }

    auto cleared_fields(const FjOutputColor& value) {
        const auto& [cat02, xyz_to_rgb, illuminant_xyz, color_space, flags, native_rgb_to_d65_xyz, d65_xyz_to_native_rgb, lightness_knee, chroma_knee, gamut_cmax, gamut_table_hash, gamut_recipe_hash, encoding] = value;
        return std::tie(cat02, xyz_to_rgb, illuminant_xyz, color_space, flags, native_rgb_to_d65_xyz, d65_xyz_to_native_rgb, lightness_knee, chroma_knee, gamut_cmax, gamut_table_hash, gamut_recipe_hash, encoding);
    }

    auto cleared_fields(const FjDust& value) {
        const auto& [cell_width_mm, cell_height_mm, slot_probability, softness_min_mm, softness_max_mm, softness_size_cap_fraction, support_x_mm, support_y_mm, fiber_fraction, fiber_drift_fraction, fiber_first_knot_min, fiber_first_knot_max, fiber_second_knot_min, fiber_second_knot_max, fiber_interior_width_min_fraction, fiber_interior_width_max_fraction, diameter_min_mm, diameter_bulk_max_mm, diameter_max_mm, diameter_tail_fraction, fiber_length_min_mm, fiber_length_max_mm, fiber_width_min_mm, fiber_width_max_mm, opacity_faint_cumulative, opacity_intermediate_cumulative, compact_opacity_min, compact_opacity_faint_end, compact_opacity_intermediate_end, compact_opacity_max, fiber_opacity_min, fiber_opacity_faint_end, fiber_opacity_intermediate_end, fiber_opacity_max, compact_dominant_aspect_min, compact_dominant_aspect_max, compact_subsidiary_scale_min, compact_subsidiary_scale_max, compact_subsidiary_aspect_min, compact_subsidiary_aspect_max, compact_subsidiary_offset_max, compact_subsidiary_angle_max_radians] = value;
        return std::tie(cell_width_mm, cell_height_mm, slot_probability, softness_min_mm, softness_max_mm, softness_size_cap_fraction, support_x_mm, support_y_mm, fiber_fraction, fiber_drift_fraction, fiber_first_knot_min, fiber_first_knot_max, fiber_second_knot_min, fiber_second_knot_max, fiber_interior_width_min_fraction, fiber_interior_width_max_fraction, diameter_min_mm, diameter_bulk_max_mm, diameter_max_mm, diameter_tail_fraction, fiber_length_min_mm, fiber_length_max_mm, fiber_width_min_mm, fiber_width_max_mm, opacity_faint_cumulative, opacity_intermediate_cumulative, compact_opacity_min, compact_opacity_faint_end, compact_opacity_intermediate_end, compact_opacity_max, fiber_opacity_min, fiber_opacity_faint_end, fiber_opacity_intermediate_end, fiber_opacity_max, compact_dominant_aspect_min, compact_dominant_aspect_max, compact_subsidiary_scale_min, compact_subsidiary_scale_max, compact_subsidiary_aspect_min, compact_subsidiary_aspect_max, compact_subsidiary_offset_max, compact_subsidiary_angle_max_radians);
    }

    auto cleared_fields(const FjScratch& value) {
        const auto& [cell_width_mm, cell_height_mm, slot_probability, softness_min_mm, softness_max_mm, softness_size_cap_fraction, support_x_mm, support_y_mm, length_min_mm, length_bulk_max_mm, length_max_mm, length_tail_fraction, width_min_mm, width_bulk_max_mm, width_max_mm, width_tail_fraction, drift_fraction, first_knot_min, first_knot_max, second_knot_min, second_knot_max, interior_width_min_fraction, interior_width_max_fraction, interior_depth_min_fraction, interior_depth_max_fraction, endpoint_abrupt_probability, interruption_probability, gap_center_min, gap_center_max, gap_span_min, gap_span_max, scuff_probability, scuff_length_max_mm, scuff_angle_max_radians, strength_min, strength_max] = value;
        return std::tie(cell_width_mm, cell_height_mm, slot_probability, softness_min_mm, softness_max_mm, softness_size_cap_fraction, support_x_mm, support_y_mm, length_min_mm, length_bulk_max_mm, length_max_mm, length_tail_fraction, width_min_mm, width_bulk_max_mm, width_max_mm, width_tail_fraction, drift_fraction, first_knot_min, first_knot_max, second_knot_min, second_knot_max, interior_width_min_fraction, interior_width_max_fraction, interior_depth_min_fraction, interior_depth_max_fraction, endpoint_abrupt_probability, interruption_probability, gap_center_min, gap_center_max, gap_span_min, gap_span_max, scuff_probability, scuff_length_max_mm, scuff_angle_max_radians, strength_min, strength_max);
    }

    auto cleared_fields(const FjPhysicalEffectsRecipe& value) {
        const auto& [film_dust, film_scratch, gate_dust, gate_scratch, gate_weave_amount, active, hash] = value;
        return std::tie(film_dust, film_scratch, gate_dust, gate_scratch, gate_weave_amount, active, hash);
    }

    auto cleared_fields(const FjPreparedInputPolicy& value) {
        const auto& [meter, band_pass_active, apply_hanatos_window, apply_hanatos_surface, projection_illuminant, input_compression_active, film_format_long_edge_mm, uv, ir, hanatos_spectral_blur, nominal_white_xyz, projection_white_xyz, hanatos_window, hanatos_surface_rgb, input_hull_hash, tc_source_hash] = value;
        return std::tie(meter, band_pass_active, apply_hanatos_window, apply_hanatos_surface, projection_illuminant, input_compression_active, film_format_long_edge_mm, uv, ir, hanatos_spectral_blur, nominal_white_xyz, projection_white_xyz, hanatos_window, hanatos_surface_rgb, input_hull_hash, tc_source_hash);
    }

    auto cleared_fields(const FjPreparedScannerPolicy& value) {
        const auto& [black_correction, white_correction, glare_active, black_level, white_level, glare_percent, glare_roughness, glare_blur_sigma_px, lens_blur_sigma_px, unsharp_sigma_px, unsharp_amount] = value;
        return std::tie(black_correction, white_correction, glare_active, black_level, white_level, glare_percent, glare_roughness, glare_blur_sigma_px, lens_blur_sigma_px, unsharp_sigma_px, unsharp_amount);
    }

    auto cleared_fields(const FjPreparedSpectralView& value) {
        const auto& [lambda_nm, illuminant, observer_xyz, weighted_xyz, dyes_cmy, baseline_min, baseline_mid, inv_yn, white_xyz, has_baseline, illuminant_hash, hash] = value;
        return std::tie(lambda_nm, illuminant, observer_xyz, weighted_xyz, dyes_cmy, baseline_min, baseline_mid, inv_yn, white_xyz, has_baseline, illuminant_hash, hash);
    }

    auto cleared_fields(const FjPrint& value) {
        const auto& [film_density_cmy, film_base_density, sensitivity_cmy, log_exposure, density_cmy, main_illuminant, preflash_illuminant, flags, main_cc_cmy, preflash_cc_cmy, exposure, preflash_exposure, preflash_raw_cmy, normalizer, film_density_hash, profile_tables_hash, main_illuminant_hash, preflash_illuminant_hash, preflash_raw_hash, balance_hash, hash] = value;
        return std::tie(film_density_cmy, film_base_density, sensitivity_cmy, log_exposure, density_cmy, main_illuminant, preflash_illuminant, flags, main_cc_cmy, preflash_cc_cmy, exposure, preflash_exposure, preflash_raw_cmy, normalizer, film_density_hash, profile_tables_hash, main_illuminant_hash, preflash_illuminant_hash, preflash_raw_hash, balance_hash, hash);
    }

    auto cleared_fields(const FjPrintBalance& value) {
        const auto& [baseline_raw_rgb, compensated_raw_rgb, factor_midgray, factor_midgray_comp, normalizer, filtered_main_illuminant_hash, hash] = value;
        return std::tie(baseline_raw_rgb, compensated_raw_rgb, factor_midgray, factor_midgray_comp, normalizer, filtered_main_illuminant_hash, hash);
    }

    auto cleared_fields(const FjScannerColor& value) {
        const auto& [cat02, xyz_to_rgb, illuminant_xyz, output_space, flags, encoding, active_gamut_recipe_hash, hash] = value;
        return std::tie(cat02, xyz_to_rgb, illuminant_xyz, output_space, flags, encoding, active_gamut_recipe_hash, hash);
    }

    auto cleared_fields(const FjScannerColorCorrection& value) {
        const auto& [route, active, black_correction, white_correction, target_black_linear, target_white_linear, reference_black_y, reference_white_y, xyz_slope, xyz_offset, exposure_scale, schema_version, hash] = value;
        return std::tie(route, active, black_correction, white_correction, target_black_linear, target_white_linear, reference_black_y, reference_white_y, xyz_slope, xyz_offset, exposure_scale, schema_version, hash);
    }

    auto cleared_fields(const FjScannerLutDescriptor& value) {
        const auto& [route, medium, capture_polarity, resolution, density_bounds_hash, channel_density_hash, base_density_hash, viewing_illuminant_hash, observer_hash, hash] = value;
        return std::tie(route, medium, capture_polarity, resolution, density_bounds_hash, channel_density_hash, base_density_hash, viewing_illuminant_hash, observer_hash, hash);
    }

    auto cleared_fields(const FjScannerPostEffectsDescriptor& value) {
        const auto& [route, glare_active, glare_percent, glare_roughness, glare_blur_sigma_px, glare_blur_radius, lens_blur_sigma_px, lens_blur_radius, unsharp_sigma_px, unsharp_radius, unsharp_amount, schema, hash] = value;
        return std::tie(route, glare_active, glare_percent, glare_roughness, glare_blur_sigma_px, glare_blur_radius, lens_blur_sigma_px, lens_blur_radius, unsharp_sigma_px, unsharp_radius, unsharp_amount, schema, hash);
    }

    auto cleared_fields(const FjDiffusionRecipe& value) {
        const auto& [domain, active, family, scatter_fraction, group_weights, group_lambda_um, warmth, spatial_scale, resolved_hash, hash] = value;
        return std::tie(domain, active, family, scatter_fraction, group_weights, group_lambda_um, warmth, spatial_scale, resolved_hash, hash);
    }

    auto cleared_fields(const FjHalationRecipe& value) {
        const auto& [scatter_active, back_reflection_active, scatter_amount, scatter_spatial_scale, halation_spatial_scale, first_sigma_um, total_strength, hash] = value;
        return std::tie(scatter_active, back_reflection_active, scatter_amount, scatter_spatial_scale, halation_spatial_scale, first_sigma_um, total_strength, hash);
    }

    auto cleared_fields(const FjLensBlurRecipe& value) {
        const auto& [sigma_um, hash] = value;
        return std::tie(sigma_um, hash);
    }

    auto cleared_fields(const FjSpatialOpticsRecipe& value) {
        const auto& [camera_diffusion, camera_lens_blur, scatter_halation, enlarger_diffusion, hash] = value;
        return std::tie(camera_diffusion, camera_lens_blur, scatter_halation, enlarger_diffusion, hash);
    }

    auto cleared_fields(const FjStringView& value) {
        const auto& [data, count] = value;
        return std::tie(data, count);
    }

    auto cleared_fields(const FjVisualGrainRecipe& value) {
        const auto& [active, sublayers_active, axis_finite_cmy, axis_block_prefix_max_cmy, particle_area_um2, particle_scale_cmy, particle_scale_layers, particle_density_min_cmy, uniformity_cmy, correlation_sigma_px, dye_cloud_blur_um, micro_structure, sublayer_count, amplitude, chroma_mix, chroma_shared_weight, chroma_independent_weight, fine_weight, mid_weight, coarse_weight, size_mix_scale, clump_temporal_mix, clump_morph_period_sec, debug_view, layers_hash, hash] = value;
        return std::tie(active, sublayers_active, axis_finite_cmy, axis_block_prefix_max_cmy, particle_area_um2, particle_scale_cmy, particle_scale_layers, particle_density_min_cmy, uniformity_cmy, correlation_sigma_px, dye_cloud_blur_um, micro_structure, sublayer_count, amplitude, chroma_mix, chroma_shared_weight, chroma_independent_weight, fine_weight, mid_weight, coarse_weight, size_mix_scale, clump_temporal_mix, clump_morph_period_sec, debug_view, layers_hash, hash);
    }

    auto cleared_fields(const FjPreparedStateView& value) {
        const auto& [film_profile_key, print_profile_key, film_profile_asset_version, print_profile_asset_version, profile_route_hash, recipe_hash, upload_core_hash, scanner_hash, route, polarity, input_policy, scanner_policy, film_exposure, film_development, dir_recipe, grain_minimum, scanner_bounds, enlarger_film_bounds, film_reference, optics, grain, effects, exposure_tables, scanner_tables, s_inverse, midgray_dwg_rgb, raw_midgray, raw_green, midgray_scale, raw_midgray_bgr, input_hull, output_boundary, scanner_lut, scanner_color, scanner_correction, scanner_post_effects, output_color, print, print_gamma, print_filter_main_cmy_cc, print_filter_preflash_cmy_cc, print_normalization_mode, normalize_print_exposure, print_exposure_compensation, calibration, neutral_cmy_cc, print_source_illuminant, print_balance, scanner_recipe_hash, print_recipe_hash, print_filters_hash, print_exposure_hash, print_illuminant_hash, print_curves_hash] = value;
        return std::tie(film_profile_key, print_profile_key, film_profile_asset_version, print_profile_asset_version, profile_route_hash, recipe_hash, upload_core_hash, scanner_hash, route, polarity, input_policy, scanner_policy, film_exposure, film_development, dir_recipe, grain_minimum, scanner_bounds, enlarger_film_bounds, film_reference, optics, grain, effects, exposure_tables, scanner_tables, s_inverse, midgray_dwg_rgb, raw_midgray, raw_green, midgray_scale, raw_midgray_bgr, input_hull, output_boundary, scanner_lut, scanner_color, scanner_correction, scanner_post_effects, output_color, print, print_gamma, print_filter_main_cmy_cc, print_filter_preflash_cmy_cc, print_normalization_mode, normalize_print_exposure, print_exposure_compensation, calibration, neutral_cmy_cc, print_source_illuminant, print_balance, scanner_recipe_hash, print_recipe_hash, print_filters_hash, print_exposure_hash, print_illuminant_hash, print_curves_hash);
    }

    template <class T>
    void expect_cleared(const T& value) {
        if constexpr (std::is_array_v<T>) {
            for (const auto& element : value)
                expect_cleared(element);
        } else if constexpr (requires { cleared_fields(value); }) {
            std::apply([](const auto&... fields) {
                (expect_cleared(fields), ...);
            },
                       cleared_fields(value));
        } else if constexpr (std::is_floating_point_v<T>) {
            equal_bits(value, T{});
        } else {
            EXPECT_EQ(value, T{});
        }
    }

    TEST(PreparedStateAbi, FailureClearsEveryDefinedFieldAndReleaseConsumes) {
        const auto baseline = fj_test_prepared_live_states();
        auto params = print_params();
        params.printPreflashExposure = 0.1;
        params.grainControls.active = true;
        params.scatterHalationControls.active = true;
        params.filmDustAmount = 0.2f;
        params.gateScratchAmount = 0.3f;
        FjStatus status{};
        std::string diagnostic;
        auto owner = JuicerCudaTest::prepare_raw_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_EQ(fj_test_prepared_live_states(), baseline + 1);
        FjPreparedStateView populated{};
        ASSERT_EQ(fj_test_prepared_state_view(owner.get(), &populated, nullptr).category, FJ_STATUS_SUCCESS);
        const auto rejected = [](FjStatus result) {
            EXPECT_EQ(result.category, FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(result.api, FJ_API_NONE);
            EXPECT_EQ(result.native_code, 0);
        };
        auto cleared = populated;
        rejected(fj_test_prepared_state_view(nullptr, &cleared, nullptr));
        expect_cleared(cleared);
        FjErrorBuffer invalid{nullptr, 1, 99};
        cleared = populated;
        rejected(fj_test_prepared_state_view(owner.get(), &cleared, &invalid));
        EXPECT_EQ(invalid.length, 0u);
        expect_cleared(cleared);
        FjErrorBuffer zeroCapacity{reinterpret_cast<char*>(std::uintptr_t{1}), 0, 99};
        cleared = populated;
        rejected(fj_test_prepared_state_view(nullptr, &cleared, &zeroCapacity));
        EXPECT_EQ(zeroCapacity.length, 0u);
        expect_cleared(cleared);
        std::array<char, 4> bytes{};
        FjErrorBuffer overflow{bytes.data(), std::numeric_limits<std::size_t>::max(), 99};
        cleared = populated;
        rejected(fj_test_prepared_state_view(owner.get(), &cleared, &overflow));
        EXPECT_EQ(overflow.length, 0u);
        expect_cleared(cleared);
        rejected(fj_test_prepared_state_view(owner.get(), nullptr, nullptr));
        EXPECT_EQ(fj_test_prepared_live_states(), baseline + 1);
        const auto wire = preparation_params(params);
        FjPreparedState* candidate = owner.get(); // A borrowed sentinel, never another hold.
        rejected(fj_test_prepared_state_create(nullptr, &wire, &candidate, nullptr));
        EXPECT_EQ(candidate, nullptr);
        candidate = owner.get();
        rejected(fj_test_prepared_state_create(assets(), nullptr, &candidate, nullptr));
        EXPECT_EQ(candidate, nullptr);
        EXPECT_EQ(fj_test_prepared_live_states(), baseline + 1);
        invalid.length = 99;
        rejected(fj_test_prepared_state_release(owner.release(), &invalid));
        EXPECT_EQ(invalid.length, 0u);
        EXPECT_EQ(fj_test_prepared_live_states(), baseline);
        EXPECT_EQ(fj_test_prepared_state_release(nullptr, nullptr).category, FJ_STATUS_SUCCESS);
    }

    TEST(PreparedStateAbi, DiagnosticTruncationPreservesStatusAndOwner) {
        auto params = print_params();
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        FjStatus status{};
        std::string diagnostic;
        auto owner = JuicerCudaTest::prepare_raw_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        const auto before = fj_test_prepared_live_states();
        params.scannerBlackLevel = std::numeric_limits<double>::quiet_NaN();
        const auto wire = preparation_params(params);
        const std::string expected = "ResourceDescriptorMismatch phase=8 field=scanner_output";
        for (std::size_t capacity : {0u, 1u, 4u, 128u}) {
            SCOPED_TRACE(capacity);
            std::array<char, 128> bytes{};
            bytes.fill('#');
            FjErrorBuffer error{capacity == 0 ? reinterpret_cast<char*>(std::uintptr_t{1}) : bytes.data(), capacity, 99};
            FjPreparedState* result = owner.get(); // Borrowed sentinel, never another hold.
            equal_status(fj_test_prepared_state_create(assets(), &wire, &result, &error), FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_EQ(result, nullptr);
            EXPECT_EQ(fj_test_prepared_live_states(), before);
            const auto count = capacity == 0 ? 0 : std::min(expected.size(), capacity - 1);
            EXPECT_EQ(error.length, count);
            if (capacity != 0) {
                EXPECT_EQ(std::string(bytes.data(), count), expected.substr(0, count));
                EXPECT_EQ(bytes[count], '\0');
            }
        }
        FjPreparedStateView view{};
        equal_status(fj_test_prepared_state_view(owner.get(), &view, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_NE(view.recipe_hash, 0u);
    }

    TEST(PreparedStateAbi, LayoutAndRealC11Calls) {
        static const std::size_t cppFacts[] = {
#define FJ_PREPARED_TYPE(type) sizeof(type), alignof(type),
#define FJ_PREPARED_FIELD(type, field) offsetof(type, field),
#include "prepared_state_abi_facts.inc"
#undef FJ_PREPARED_TYPE
#undef FJ_PREPARED_FIELD
        };
        std::size_t cCount = 0, rustCount = 0;
        const auto* cFacts = fj_test_prepared_state_c_facts(&cCount);
        const auto* rustFacts = fj_test_prepared_state_abi_facts(&rustCount);
        ASSERT_EQ(cCount, std::size(cppFacts));
        ASSERT_EQ(rustCount, std::size(cppFacts));
        for (std::size_t i = 0; i < cCount; ++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(cFacts[i], cppFacts[i]);
            EXPECT_EQ(rustFacts[i], cppFacts[i]);
        }
        ParamSnapshot params;
        params.grainControls.active = false;
        const auto wire = preparation_params(params);
        FjPreparedState* owner = nullptr;
        ASSERT_EQ(fj_test_prepared_state_create_c(assets(), &wire, &owner, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_NE(owner, nullptr);
        JuicerCudaTest::RawPreparedStateHold hold(owner);
        FjPreparedStateView first{}, second{};
        EXPECT_EQ(fj_test_prepared_state_view_c(owner, &first, nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(fj_test_prepared_state_view(owner, &second, nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(first.recipe_hash, second.recipe_hash);
        EXPECT_EQ(first.film_development.density_rgb.data, second.film_development.density_rgb.data);
        EXPECT_EQ(first.scanner_tables.dyes_cmy[2].data, second.scanner_tables.dyes_cmy[2].data);
        EXPECT_EQ(fj_test_prepared_state_release_c(hold.release(), nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(fj_test_prepared_state_release_c(nullptr, nullptr).category, FJ_STATUS_SUCCESS);
    }

    TEST(PreparedStateAbi, CallScopeLayoutAndRustC11Witnesses) {
        static_assert(std::is_same_v<decltype(FjGrainCoefficients::flags), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjGrainCoefficients::debug_view), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjGrainCoefficients::sublayer_count), std::size_t>);
        static_assert(std::is_same_v<decltype(FjGrainCoefficients::micro_structure), float[2]>);
        static_assert(std::is_same_v<decltype(FjGrainCoefficients::particle_density_min_cmy), float[3]>);
        static_assert(std::is_same_v<decltype(FjGrainCoefficients::uniformity_cmy), float[3]>);
        static_assert(std::is_same_v<decltype(FjGrainCoefficients::layer_axis_block_prefix_max), float[3][16]>);
        static_assert(std::is_same_v<decltype(FjCallCompletionInput::render_outcome_valid), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjCallCompletionInput::abort_failed), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjCallCompletionInput::recovery_scope_entered), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjCallCompletionInput::has_inspected_context), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjCallCompletionInput::force_fatal), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjCallCompletionResult::fatal), std::uint32_t>);
        static const std::size_t cppFacts[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#define FJ_ABI_VALUE(tag, value) tag,
#include "prepared_call_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
        };
        std::size_t cCount = 0, rustCount = 0;
        const auto* cFacts = fj_test_prepared_call_c_facts(&cCount);
        const auto* rustFacts = fj_test_prepared_call_abi_facts(&rustCount);
        ASSERT_EQ(cCount, std::size(cppFacts));
        ASSERT_EQ(rustCount, std::size(cppFacts));
        for (std::size_t i = 0; i < cCount; ++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(cFacts[i], cppFacts[i]);
            EXPECT_EQ(rustFacts[i], cppFacts[i]);
        }
    }

    TEST(PreparedStateAbi, HostPreparationFacadeIsScopedAndC11Callable) {
        const auto before = fj_test_gaussian_kernel_live_owners();
        EXPECT_EQ(fj_test_host_preparation_c_call(), 1);
        EXPECT_EQ(fj_test_gaussian_kernel_live_owners(), before);
        EXPECT_EQ(fj_test_scanner_lut_live_owners(), 0u);
    }

    TEST(PreparedStateAbi, InputHullFaultsBelongToActualOwnerAndRetainedWarmEntry) {
        constexpr std::array<const char*, 4> kRequirements{
            "cie1931_380_700_and_d65", "finite_locus_ray_intersections", "finite_inscribed_scale", "stable_identity"};
        const auto release = [](FjInputHullOwner* owner) {
            (void)fj_test_input_hull_release(owner, nullptr);
        };
        using HullHold = std::unique_ptr<FjInputHullOwner, decltype(release)>;
        for (std::uint32_t fault = 1; fault <= 6; ++fault) {
            SCOPED_TRACE(fault);
            JuicerAssets::Library library(JuicerProcess::data_directory());
            FjAssets* const owner = library.borrow_rust_assets();
            equal_status(fj_test_assets_input_hull_fault(owner, fault, nullptr), FJ_STATUS_SUCCESS);
            std::array<char, 512> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            FjInputHullOwner* acquired = nullptr;
            const auto failed = fj_test_assets_input_hull_acquire(owner, &acquired, &error);
            equal_status(failed, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                       : FJ_STATUS_PREPARATION_FAILURE);
            ASSERT_EQ(acquired, nullptr);
            if (fault >= 3) {
                const auto index = static_cast<std::size_t>(fault - 3);
                ASSERT_LT(index, kRequirements.size());
                EXPECT_EQ(std::string(message.data(), error.length), std::string("MalformedRequiredResource component=input_gamut_hull requirement=") + kRequirements[index]);
            }
            std::uint64_t consumed = 0;
            equal_status(fj_test_cache_fault_consumed(owner, FJ_TEST_CACHE_INPUT_HULL, &consumed, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(consumed, 1u);
            equal_status(fj_test_assets_input_hull_acquire(owner, &acquired, nullptr), FJ_STATUS_SUCCESS);
            HullHold retained(acquired, release);
            FjInputHullView first{};
            equal_status(fj_test_input_hull_view(retained.get(), &first, nullptr), FJ_STATUS_SUCCESS);
            equal_status(fj_test_assets_input_hull_fault(owner, 1, nullptr), FJ_STATUS_SUCCESS);
            equal_status(fj_test_assets_input_hull_acquire(owner, &acquired, nullptr), FJ_STATUS_SUCCESS);
            HullHold warm(acquired, release);
            FjInputHullView next{};
            equal_status(fj_test_input_hull_view(warm.get(), &next, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(first.entry_identity, next.entry_identity);
            EXPECT_EQ(first.xy.data, next.xy.data);
            equal_status(fj_test_cache_fault_consumed(owner, FJ_TEST_CACHE_INPUT_HULL, &consumed, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(consumed, 0u);
            std::uint32_t pending = 0;
            equal_status(fj_test_assets_input_hull_fault_pending(owner, &pending, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(pending, 1u);
            JuicerAssets::Library other(JuicerProcess::data_directory());
            equal_status(fj_test_assets_input_hull_acquire(other.borrow_rust_assets(), &acquired, nullptr), FJ_STATUS_SUCCESS);
            HullHold unrelated(acquired, release);
            equal_status(fj_test_assets_input_hull_fault_pending(owner, &pending, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(pending, 1u);
            equal_status(fj_test_assets_input_hull_fault(owner, 0, nullptr), FJ_STATUS_SUCCESS);
            equal_status(fj_test_assets_input_hull_fault_pending(owner, &pending, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(pending, 0u);
        }
    }

    TEST(PreparedStateAbi, CacheFamiliesShareBackingAndOutliveAssets) {
        const auto path = JuicerProcess::data_directory();
        const auto& native = path.native();
#if defined(_WIN32)
        constexpr auto kEncoding = FJ_PATH_WINDOWS_WIDE;
#else
        constexpr auto kEncoding = FJ_PATH_UNIX_BYTES;
#endif
        FjAssets* resourceOwner = nullptr;
        ASSERT_EQ(fj_legacy_assets_create({native.data(), native.size(), kEncoding}, &resourceOwner, nullptr).category, FJ_STATUS_SUCCESS);
        const auto destroyAssets = [](FjAssets* owner) {
            (void)fj_legacy_assets_destroy(owner, nullptr);
        };
        std::unique_ptr<FjAssets, decltype(destroyAssets)> retainedAssets(resourceOwner, destroyAssets);
        auto params = print_params();
        params.spectralUpsamplingMode = 0;
        FjStatus status{};
        std::string diagnostic;
        auto state = JuicerCudaTest::prepare_state(resourceOwner, params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        FjPreparedStateView before{};
        ASSERT_EQ(state_view(state, before).category, FJ_STATUS_SUCCESS);
        for (int repeat = 0; repeat < 100; ++repeat) {
            FjPreparedStateView warm{};
            equal_status(state_view(state, warm), FJ_STATUS_SUCCESS);
            EXPECT_EQ(warm.film_development.density_rgb.data, before.film_development.density_rgb.data);
            EXPECT_EQ(warm.exposure_tables.illuminant.data, before.exposure_tables.illuminant.data);
            EXPECT_EQ(warm.scanner_tables.dyes_cmy[2].data, before.scanner_tables.dyes_cmy[2].data);
            EXPECT_EQ(warm.film_exposure.tc_lut_rgba.data, before.film_exposure.tc_lut_rgba.data);
            EXPECT_EQ(warm.input_hull.xy.data, before.input_hull.xy.data);
            EXPECT_EQ(warm.output_boundary.cmax.data, before.output_boundary.cmax.data);
            EXPECT_EQ(warm.recipe_hash, before.recipe_hash);
        }
        FjIlluminantsOwner* illuminants = nullptr;
        FjInputHullOwner* hull = nullptr;
        FjOutputBoundaryOwner* boundary = nullptr;
        const auto releaseLight = [](FjIlluminantsOwner* owner) {
            (void)fj_test_prepared_illuminants_release_c(owner, nullptr);
        };
        const auto releaseHull = [](FjInputHullOwner* owner) {
            (void)fj_test_prepared_hull_release_c(owner, nullptr);
        };
        const auto releaseBoundary = [](FjOutputBoundaryOwner* owner) {
            (void)fj_test_prepared_boundary_release_c(owner, nullptr);
        };
        ASSERT_EQ(fj_test_prepared_illuminants_acquire_c(resourceOwner, &illuminants, nullptr).category, FJ_STATUS_SUCCESS);
        std::unique_ptr<FjIlluminantsOwner, decltype(releaseLight)> lightHold(illuminants, releaseLight);
        ASSERT_EQ(fj_test_prepared_hull_acquire_c(resourceOwner, &hull, nullptr).category, FJ_STATUS_SUCCESS);
        std::unique_ptr<FjInputHullOwner, decltype(releaseHull)> hullHold(hull, releaseHull);
        FjOutputBoundarySource source{};
        source.output_space = before.output_color.color_space;
        std::copy_n(before.output_color.d65_xyz_to_native_rgb, 9, source.d65_xyz_to_native_rgb);
        source.transform_hash = before.output_boundary.transform_hash;
        ASSERT_EQ(fj_test_prepared_boundary_acquire_c(resourceOwner, &source, before.output_boundary.contract_hash, &boundary, nullptr).category, FJ_STATUS_SUCCESS);
        std::unique_ptr<FjOutputBoundaryOwner, decltype(releaseBoundary)> boundaryHold(boundary, releaseBoundary);
        FjIlluminantsView light{};
        FjInputHullView input{};
        FjOutputBoundaryView output{};
        ASSERT_EQ(fj_test_prepared_illuminants_view_c(illuminants, &light, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(fj_test_prepared_hull_view_c(hull, &input, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(fj_test_prepared_boundary_view_c(boundary, &output, nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(input.entry_identity, before.input_hull.entry_identity);
        EXPECT_EQ(input.xy.data, before.input_hull.xy.data);
        EXPECT_EQ(output.entry_identity, before.output_boundary.entry_identity);
        EXPECT_EQ(output.cmax.data, before.output_boundary.cmax.data);
        ASSERT_EQ(light.d65.count, 81u);
        const std::vector<float> d65(light.d65.data, light.d65.data + light.d65.count);
        const std::vector<float> xy(input.xy.data, input.xy.data + input.xy.count);
        const std::vector<float> cmax(output.cmax.data, output.cmax.data + output.cmax.count);
        ASSERT_EQ(fj_legacy_assets_release_cached_payloads(resourceOwner, nullptr).category, FJ_STATUS_SUCCESS);
        FjIlluminantsOwner* nextLights = nullptr;
        ASSERT_EQ(fj_test_assets_illuminants_acquire(resourceOwner, &nextLights, nullptr).category, FJ_STATUS_SUCCESS);
        std::unique_ptr<FjIlluminantsOwner, decltype(releaseLight)> nextLightHold(nextLights, releaseLight);
        FjIlluminantsView nextLight{};
        ASSERT_EQ(fj_test_illuminants_view(nextLights, &nextLight, nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_NE(light.entry_identity, nextLight.entry_identity);
        equal_span(nextLight.d65, d65);
        EXPECT_EQ(fj_test_illuminants_release(nextLightHold.release(), nullptr).category, FJ_STATUS_SUCCESS);
        retainedAssets.reset();
        // All nine production C11 entry calls use the same retained allocations.
        ASSERT_EQ(fj_test_prepared_illuminants_view_c(illuminants, &light, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(fj_test_prepared_hull_view_c(hull, &input, nullptr).category, FJ_STATUS_SUCCESS);
        ASSERT_EQ(fj_test_prepared_boundary_view_c(boundary, &output, nullptr).category, FJ_STATUS_SUCCESS);
        equal_span(light.d65, d65);
        equal_span(input.xy, xy);
        equal_span(output.cmax, cmax);
        FjPreparedStateView after{};
        ASSERT_EQ(state_view(state, after).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(after.recipe_hash, before.recipe_hash);
        EXPECT_EQ(after.film_development.density_rgb.data, before.film_development.density_rgb.data);
        EXPECT_EQ(after.input_hull.xy.data, input.xy.data);
        EXPECT_EQ(after.output_boundary.cmax.data, output.cmax.data);
        EXPECT_EQ(fj_test_prepared_illuminants_release_c(lightHold.release(), nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(fj_test_prepared_hull_release_c(hullHold.release(), nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(fj_test_prepared_boundary_release_c(boundaryHold.release(), nullptr).category, FJ_STATUS_SUCCESS);
    }
} // namespace

// MSVC filesystem conversion is reported as escaping even through catch (...).
// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) {
    try {
        ::testing::InitGoogleTest(&argc, argv);
        if (argc != 3 && argc != 6)
            return 2;
        artifactRoot = argv[2];
        std::string preset;
        bool checkProjection = false;
        std::filesystem::path capturePath;
        if (argc == 6) {
            preset = argv[5];
            const std::string mode = argv[3];
            if (mode == "--capture-state-projection") {
                captureProjection = true;
                capturePath = argv[4];
                if (std::filesystem::exists(capturePath))
                    throw std::runtime_error("preserve the existing prepared-state capture");
                const auto parent = std::filesystem::absolute(capturePath.parent_path()).lexically_normal();
                if (parent != std::filesystem::absolute(artifactRoot).lexically_normal())
                    throw std::runtime_error("prepared-state capture must use its explicit artifact directory");
            } else if (mode == "--check-state-projection") {
                projectionOracle = read_projection(argv[4], preset);
                checkProjection = true;
            } else {
                return 2;
            }
            ::testing::GTEST_FLAG(filter) = successful_projection_filter();
        } else {
            const char* configuredPreset = std::getenv("JUICER_TEST_PRESET_NAME");
            if (configuredPreset == nullptr)
                throw std::runtime_error("prepared-state tests require the configured native preset");
            preset = configuredPreset;
            const auto fixture = std::filesystem::path(__FILE__).parent_path() / "fixtures/prepared-state-projection" / (preset + ".json");
            projectionOracle = read_projection(fixture, preset);
        }
        if (preset != FJ_PREPARED_STATE_PRESET)
            throw std::runtime_error("prepared-state preset does not match its compiled native profile");
        const auto resourceDirectory = std::string("out/build/") + preset + "/tests/Resources";
        const auto sourceRoot = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
        if (!std::filesystem::equivalent(sourceRoot / resourceDirectory, std::filesystem::path(argv[1])))
            throw std::runtime_error("prepared-state resources do not match the declared native preset");
        JuicerCuda::Owner owner;
        owner.create(std::filesystem::path(argv[1]));
        JuicerProcess::root().ensure_bootstrap();
        const int result = RUN_ALL_TESTS();
        if (result != 0)
            return result;
        if (captureProjection || checkProjection || (::testing::GTEST_FLAG(filter) == "PreparedStateProducts.*" && projectionOracle)) {
            if (observedProjectionCases.size() != kProjectionCases.size())
                throw std::runtime_error("prepared-state projection did not execute all 31 rows");
            for (std::size_t i = 0; i < kProjectionCases.size(); ++i)
                if (observedProjectionCases[i] != kProjectionCases[i])
                    throw std::runtime_error("prepared-state projection row order mismatch");
        }
        if (captureProjection) {
            std::filesystem::create_directories(artifactRoot);
            std::ofstream output(capturePath, std::ios::binary);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output << projection_payload(preset, resourceDirectory).dump() << '\n';
            output.close();
        }
        return 0;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        std::cerr << error.failure.diagnostic << '\n';
        return 1;

    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "unknown prepared-state test failure\n";
        return 1;
    }
}
