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

#include "ProcessRuntime.h"
#include "JuicerState.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "ProcessRoot.h"
#include "test_asset_library.h"
#include "NativePath.h"
#include "juicer_cuda_owner.h"
#include "prepared_boundary.h"
#include "prepared_descriptors.h"
#include "prepared_state_oracle.h"
#include "juicer_test_api.h"
#include "juicer_cuda_call_scope.h"
#include "gtest_cases.h"

extern "C" const std::size_t* fj_test_prepared_state_c_facts(std::size_t*);
extern "C" const std::size_t* fj_test_instance_c_facts(std::size_t*);
extern "C" std::uint32_t fj_test_instance_c_witness(const FjAssets*, const FjPreparationParams*, std::uint32_t);
extern "C" const std::size_t* fj_test_prepared_call_c_facts(std::size_t*);
extern "C" int fj_test_host_preparation_c_call();
extern "C" const std::size_t* fj_test_prepared_call_abi_facts(std::size_t*);
extern "C" FjStatus fj_test_prepared_state_create_c(const FjAssets*, const FjPreparationParams*, FjPreparedState**, FjErrorBuffer*);
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

using FjInstanceCreateFn = FjStatus (*)(FjInstance**, FjErrorBuffer*);
using FjBeginCaptureFn = FjStatus (*)(FjInstance*, std::uint32_t, FjCaptureStart*, FjErrorBuffer*);
using FjCaptureCurrentFn = FjStatus (*)(const FjInstance*, std::uint64_t, std::uint32_t*, FjErrorBuffer*);
using FjBeginHostWriteFn = FjStatus (*)(FjInstance*, std::uint64_t, FjHostWrite**, std::uint32_t*, FjErrorBuffer*);
using FjHostWriteFinishFn = FjStatus (*)(FjHostWrite*, std::uint32_t, FjStringView, std::uint32_t*, FjErrorBuffer*);
using FjBeginSnapshotFn = FjStatus (*)(FjInstance*, std::uint64_t, FjSnapshotStart*, FjErrorBuffer*);
using FjInstallFn = FjStatus (*)(FjInstance*, std::uint64_t, std::uint64_t, const FjPreparationParams*, std::uint64_t, std::uint32_t*, FjErrorBuffer*);
using FjInvalidateFn = FjStatus (*)(FjInstance*, std::uint64_t, std::uint32_t, FjStringView, std::uint32_t*, FjErrorBuffer*);
using FjInstanceAdmitFn = FjStatus (*)(FjInstance*, const FjAssets*, FjAdmission*, FjErrorBuffer*);
using FjInstanceDestroyFn = FjStatus (*)(FjInstance*, FjErrorBuffer*);
using FjInstanceBindIdentityFn = FjStatus (*)(const FjInstance*, std::uint64_t, std::uint64_t, FjErrorBuffer*);
using FjInstanceIdentityFn = FjStatus (*)(const FjInstance*, std::uint32_t*, std::uint64_t*, std::uint64_t*, FjErrorBuffer*);
using FjInstanceInvalidateSubmissionFn = FjStatus (*)(const FjInstance*, std::int32_t, std::uintptr_t, std::uint32_t*, FjErrorBuffer*);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_create), FjInstanceCreateFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_begin_capture), FjBeginCaptureFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_capture_is_current), FjCaptureCurrentFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_begin_host_write), FjBeginHostWriteFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_host_write_finish), FjHostWriteFinishFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_begin_snapshot), FjBeginSnapshotFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_install), FjInstallFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_invalidate), FjInvalidateFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_admit), FjInstanceAdmitFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_destroy), FjInstanceDestroyFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_bind_identity), FjInstanceBindIdentityFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_identity), FjInstanceIdentityFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_instance_invalidate_submission), FjInstanceInvalidateSubmissionFn>);

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
using FjPolicyEntryFn = FjStatus (*)(const FjPreparedState*, const FjExtent*, FjPreparedFramePolicy*, FjErrorBuffer*);
using FjTraceSubmissionFn = void (*)(void*, const FjSubmission*, std::uint32_t);
using FjRenderEntryFn = FjCallCompletionResult (*)(const FjAssets*, const FjPreparedState*, FjCudaCallAccess, const FjFrame*, const FjPreparedFramePolicy*, const FjInstance*, std::uint64_t, FjAbortCallback, FjErrorBuffer*);
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
static_assert(std::is_same_v<decltype(FjCudaCallAccess::trace_submission), FjTraceSubmissionFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_prepared_state_frame_policy), FjPolicyEntryFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_prepared_state_render), FjRenderEntryFn>);
static_assert(std::is_same_v<decltype(&fj_legacy_prepared_unavailable), FjUnavailableEntryFn>);

namespace {
    std::filesystem::path artifactRoot;
    using Json = nlohmann::json;
    constexpr std::array<std::string_view, 31> kProjectionCases{
        "NegativeDirectHanatos", "NegativeDirectMallett", "NegativeDirectArctic", "NegativePrintHanatos", "NegativePrintMallett", "NegativePrintArctic", "PositiveDirectHanatos", "PositiveDirectMallett", "PositiveDirectArctic", "PositivePrintHanatos", "PositivePrintMallett", "PositivePrintArctic", "DirectIgnoresPrintControls", "PreflashAndNormalizationModes[norm=0,compensate=0]", "PreflashAndNormalizationModes[norm=0,compensate=1]", "PreflashAndNormalizationModes[norm=1,compensate=0]", "PreflashAndNormalizationModes[norm=1,compensate=1]", "GammaAndGrainLayers", "PhysicalDefectsAndWeave", "CameraAndEnlargerOptics", "DirAndCompressionOff", "SignedZeroGlare[+0]", "SignedZeroGlare[-0]", "CorrectionsAndScannerControls[print]", "CorrectionsAndScannerControls[direct]", "EncodingOnlyReusesMediumIdentities[encoded]", "EncodingOnlyReusesMediumIdentities[linear]", "SnapshotIntegerBooleansPreserveNonzeroCasts[direct]", "SnapshotIntegerBooleansPreserveNonzeroCasts[print]", "DisabledFamiliesExcludeUnusedControls", "GrainDebugSuppressesPhysicalEffects"};
    std::optional<Json> projectionOracle;
    std::vector<std::string> observedProjectionCases;

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

    void record_projection(const ParamSnapshot& params, const FjAssets* resourceOwner, const FjPreparedStateView& actual) {
        const auto name = projection_case(params);
        if (std::find(kProjectionCases.begin(), kProjectionCases.end(), name) == kProjectionCases.end() ||
            std::find(observedProjectionCases.begin(), observedProjectionCases.end(), name) != observedProjectionCases.end())
            throw std::runtime_error("unknown or duplicate prepared-state projection case " + name);
        observedProjectionCases.push_back(name);
        const auto& row = oracle_row(name);
        PreparedStateOracle::compare(row.at("input"), PreparedStateOracle::params(preparation_params(params)));
        PreparedStateOracle::compare(row.at("expected"), PreparedStateOracle::view(actual));
        Json relations = Json::array();
        // Frozen native.* labels now identify raw consumers of the same Rust cache entry.
        if (actual.input_hull.xy.count != 0) {
            FjInputHullOwner* hull = nullptr;
            ASSERT_EQ(fj_test_assets_input_hull_acquire(resourceOwner, &hull, nullptr).category, FJ_STATUS_SUCCESS);
            const auto release = [](FjInputHullOwner* owner) {
                (void)fj_test_input_hull_release(owner, nullptr);
            };
            std::unique_ptr<FjInputHullOwner, decltype(release)> hold(hull, release);
            FjInputHullView input{};
            ASSERT_EQ(fj_test_input_hull_view(hold.get(), &input, nullptr).category, FJ_STATUS_SUCCESS);
            relations.push_back(backing_edge("native.input_hull.entry_identity", "actual.input_hull.entry_identity", input.entry_identity == actual.input_hull.entry_identity));
            relations.push_back(backing_edge("native.input_hull.xy", "actual.input_hull.xy", input.xy.data == actual.input_hull.xy.data));
        }
        if (actual.output_boundary.cmax.count != 0) {
            FjOutputBoundarySource source{};
            source.output_space = actual.output_color.color_space;
            std::copy_n(actual.output_color.d65_xyz_to_native_rgb, 9, source.d65_xyz_to_native_rgb);
            source.transform_hash = actual.output_boundary.transform_hash;
            FjOutputBoundaryOwner* boundary = nullptr;
            ASSERT_EQ(fj_test_assets_output_boundary_acquire(resourceOwner, &source, actual.output_boundary.contract_hash, &boundary, nullptr).category, FJ_STATUS_SUCCESS);
            const auto release = [](FjOutputBoundaryOwner* owner) {
                (void)fj_test_output_boundary_release(owner, nullptr);
            };
            std::unique_ptr<FjOutputBoundaryOwner, decltype(release)> hold(boundary, release);
            FjOutputBoundaryView output{};
            ASSERT_EQ(fj_test_output_boundary_view(hold.get(), &output, nullptr).category, FJ_STATUS_SUCCESS);
            relations.push_back(backing_edge("native.output_boundary.entry_identity", "actual.output_boundary.entry_identity", output.entry_identity == actual.output_boundary.entry_identity));
            relations.push_back(backing_edge("native.output_boundary.cmax", "actual.output_boundary.cmax", output.cmax.data == actual.output_boundary.cmax.data));
        }
        Json recordedRelations = Json::array();
        for (const auto& edge : row.at("backing_relations"))
            if (!encoding_edge(edge))
                recordedRelations.push_back(edge);
        compare_backing(recordedRelations, relations);
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

    std::optional<bool> consume_projection_oracle_expectation(int& argc, char** argv) {
        constexpr std::string_view prefix = "--expect-projection-oracle=";
        std::optional<bool> expected;
        int retained = 1;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (!argument.starts_with(prefix)) {
                argv[retained++] = argv[index];
                continue;
            }
            if (expected.has_value())
                throw std::runtime_error("duplicate prepared-state oracle expectation");
            const auto value = argument.substr(prefix.size());
            if (value == "loaded")
                expected = true;
            else if (value == "unused")
                expected = false;
            else
                throw std::runtime_error("invalid prepared-state oracle expectation");
        }
        argc = retained;
        argv[argc] = nullptr;
        return expected;
    }

    bool projection_case_test(std::string_view testName) {
        return std::any_of(kProjectionCases.begin(), kProjectionCases.end(), [testName](std::string_view name) {
            return name.substr(0, name.find('[')) == testName;
        });
    }

    bool consumes_projection_oracle(const testing::TestInfo& test) {
        if (test.test_suite_name() != std::string_view("PreparedStateProducts"))
            return false;
        return projection_case_test(test.name()) || test.name() == std::string_view("OracleReaderRejectsFieldShapeWidthTagAndBackingMutations");
    }

    bool selected_tests_consume_projection_oracle() {
        const auto* const unit = testing::UnitTest::GetInstance();
        for (int suiteIndex = 0; suiteIndex < unit->total_test_suite_count(); ++suiteIndex) {
            const auto* const suite = unit->GetTestSuite(suiteIndex);
            for (int testIndex = 0; testIndex < suite->total_test_count(); ++testIndex) {
                const auto* const test = suite->GetTestInfo(testIndex);
                if (test->should_run() && consumes_projection_oracle(*test))
                    return true;
            }
        }
        return false;
    }

    bool all_projection_case_tests_selected() {
        std::set<std::string_view> selectedNames;
        const auto* const unit = testing::UnitTest::GetInstance();
        for (int suiteIndex = 0; suiteIndex < unit->total_test_suite_count(); ++suiteIndex) {
            const auto* const suite = unit->GetTestSuite(suiteIndex);
            if (suite->name() != std::string_view("PreparedStateProducts"))
                continue;
            for (int testIndex = 0; testIndex < suite->total_test_count(); ++testIndex) {
                const auto* const test = suite->GetTestInfo(testIndex);
                if (test->should_run())
                    selectedNames.insert(test->name());
            }
        }
        return std::all_of(kProjectionCases.begin(), kProjectionCases.end(), [&selectedNames](std::string_view name) {
            return selectedNames.contains(name.substr(0, name.find('[')));
        });
    }

    class ProjectionOracleEnvironment final : public testing::Environment {
    public:
        ProjectionOracleEnvironment(std::filesystem::path path, std::string preset, bool checkProjection)
            : path_(std::move(path)), preset_(std::move(preset)), checkProjection_(checkProjection) {}

        void SetUp() override {
            if (checkProjection_ || selected_tests_consume_projection_oracle())
                projectionOracle = read_projection(path_, preset_);
        }

    private:
        std::filesystem::path path_;
        std::string preset_;
        bool checkProjection_;
    };

    class PreparedStateRuntimeFixture : public testing::Test {
    protected:
        void SetUp() override {
            runtime.create(JuicerProcess::data_directory());
        }

        const FjAssets* assets() const {
            return runtime.borrow().assets;
        }

        JuicerProcess::ProcessRuntime runtime;
    };

    class PreparedStateProducts : public PreparedStateRuntimeFixture {};
    class PreparedStateAbi : public PreparedStateRuntimeFixture {};

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

    void compare_product(const FjAssets* assets, const ParamSnapshot& params) {
        JuicerTests::CaseEvent case_event("prepared_projection", projection_case(params));
        FjStatus status{};
        std::string diagnostic;
        auto owner = JuicerCudaTest::prepare_state(assets, params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        ASSERT_TRUE(owner.has_value());
        FjPreparedStateView actual{};
        ASSERT_EQ(state_view(owner, actual).category, FJ_STATUS_SUCCESS);
        record_projection(params, assets, actual);
        if (!::testing::Test::HasFailure())
            case_event.passed();
    }

    void baseline(const FjAssets* assets, bool positive, bool print, int method) {
        JuicerTests::GtestCohort cohort("baseline-product", std::vector<std::string>{"selected-profile-route-method"});
        auto caseEvent = cohort.case_event("selected-profile-route-method");
        ParamSnapshot params;
        params.filmProfileKey = positive ? "fujifilm_provia_100f" : "kodak_portra_400";
        params.printProfileKey = "kodak_portra_endura";
        params.scanRoute = positive ? (print ? Spektrafilm::ScanRoute::PositivePrintScan : Spektrafilm::ScanRoute::PositiveDirectScan) : (print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan);
        params.spectralUpsamplingMode = method;
        params.grainControls.active = false;
        compare_product(assets, params);
        caseEvent.passed();
    }

#define FJ_STATE_BASELINE(name, positive, print, method) \
    TEST_F(PreparedStateProducts, name) {                \
        baseline(assets(), positive, print, method);     \
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

    TEST_F(PreparedStateProducts, OracleReaderRejectsFieldShapeWidthTagAndBackingMutations) {
        const auto params = print_params();
        const auto expected = oracle_row("NegativePrintMallett").at("expected");
        std::string diagnostic;
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
        ASSERT_NE(actual.output_boundary.cmax.count, 0u);
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

    TEST_F(PreparedStateProducts, DirectIgnoresPrintControls) {
        JuicerTests::GtestCohort cohort("direct-ignores-print-controls", std::vector<std::string>{"direct-route"});
        auto caseEvent = cohort.case_event("direct-route");
        auto params = print_params();
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        params.printProfileKey = "__absent_print";
        params.printGammaFactor = std::numeric_limits<double>::quiet_NaN();
        params.printExposure = std::numeric_limits<double>::infinity();
        params.glarePercent = std::numeric_limits<double>::quiet_NaN();
        compare_product(assets(), params);
        caseEvent.passed();
    }

    TEST_F(PreparedStateProducts, PreflashAndNormalizationModes) {
        const std::vector<std::string> ids{"normalize=0/compensate=0", "normalize=0/compensate=1", "normalize=1/compensate=0", "normalize=1/compensate=1"};
        JuicerTests::GtestCohort cohort("preflash-normalization-modes", ids);
        for (int normalize = 0; normalize < 2; ++normalize)
            for (int compensate = 0; compensate < 2; ++compensate) {
                const auto id = "normalize=" + std::to_string(normalize) + "/compensate=" + std::to_string(compensate);
                auto caseEvent = cohort.case_event(id);
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
                compare_product(assets(), params);
                caseEvent.passed();
            }
    }

    TEST_F(PreparedStateProducts, GammaAndGrainLayers) {
        JuicerTests::GtestCohort cohort("gamma-and-grain-layers", std::vector<std::string>{"active-sublayers"});
        auto caseEvent = cohort.case_event("active-sublayers");
        auto params = print_params();
        params.grainControls.active = true;
        params.grainControls.sublayersActive = true;
        params.filmGammaFactor = 1.125f;
        params.printGammaFactor = 0.875;
        compare_product(assets(), params);
        caseEvent.passed();
    }

    TEST_F(PreparedStateProducts, PhysicalDefectsAndWeave) {
        JuicerTests::GtestCohort cohort("physical-defects-and-weave", std::vector<std::string>{"configured-amounts"});
        auto caseEvent = cohort.case_event("configured-amounts");
        auto params = print_params();
        params.filmDustAmount = 0.3f;
        params.filmScratchAmount = 0.2f;
        params.gateDustAmount = 0.1f;
        params.gateScratchAmount = 0.4f;
        params.gateWeaveAmount = 0.5;
        compare_product(assets(), params);
        caseEvent.passed();
    }

    TEST_F(PreparedStateProducts, CameraAndEnlargerOptics) {
        JuicerTests::GtestCohort cohort("camera-enlarger-optics", std::vector<std::string>{"enabled-optics"});
        auto caseEvent = cohort.case_event("enabled-optics");
        auto params = print_params();
        params.cameraDiffusion.active = true;
        params.enlargerDiffusion.active = true;
        params.scatterHalationControls.active = true;
        compare_product(assets(), params);
        caseEvent.passed();
    }

    TEST_F(PreparedStateProducts, DirAndCompressionOff) {
        JuicerTests::GtestCohort cohort("dir-compression-off", std::vector<std::string>{"disabled-compression"});
        auto caseEvent = cohort.case_event("disabled-compression");
        auto params = print_params();
        params.dirCouplers.active = false;
        params.inputCompressionEnabled = 0;
        params.outputGamutCompressionEnabled = 0;
        compare_product(assets(), params);
        caseEvent.passed();
    }

    TEST_F(PreparedStateProducts, SignedZeroGlare) {
        JuicerTests::GtestCohort cohort("signed-zero-glare", std::vector<std::string>{"positive-zero", "negative-zero"});
        auto params = print_params();
        params.glarePercent = 0.0;
        auto positiveCase = cohort.case_event("positive-zero");
        compare_product(assets(), params);
        positiveCase.passed();
        params.glarePercent = -0.0;
        auto negativeCase = cohort.case_event("negative-zero");
        compare_product(assets(), params);
        negativeCase.passed();
    }

    TEST_F(PreparedStateProducts, CorrectionsAndScannerControls) {
        JuicerTests::GtestCohort cohort("correction-scanner-route", std::vector<std::string>{"print", "direct"});
        auto params = print_params();
        params.scannerBlackCorrection = 1;
        params.scannerWhiteCorrection = 1;
        params.scannerLensBlurSigmaPx = 0.4;
        params.scannerUnsharpMask = {0.8, 0.6};
        params.outputColorSpace = 5;
        params.outputCctfEncoding = 0;
        auto printCase = cohort.case_event("print");
        compare_product(assets(), params);
        printCase.passed();
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        auto directCase = cohort.case_event("direct");
        compare_product(assets(), params);
        directCase.passed();
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

    void competing_precheck(const FjAssets* assets, const ParamSnapshot& params, const char* expectedDiagnostic, std::initializer_list<std::uint32_t> expectedTrace) {
        struct ClearExposureFault {
            ~ClearExposureFault() {
                fj_test_exposure_clear_fault();
            }
        } clearExposureFault;
        auto repaired = print_params();
        repaired.scanRoute = params.scanRoute;
        JuicerTests::GtestCohort cohort("competing-precheck-temperature", std::vector<std::string>{"cold", "warm"});
        for (bool cold : {true, false}) {
            auto caseEvent = cohort.case_event(cold ? "cold" : "warm");
            SCOPED_TRACE(cold ? "cold" : "warm");
            if (cold)
                equal_status(fj_legacy_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
            equal_status(fj_test_exposure_arm_fault(5, 1, 1), FJ_STATUS_SUCCESS);
            FjStatus status{};
            std::string diagnostic;
            auto owner = JuicerCudaTest::prepare_state(assets, params, status, diagnostic);
            equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_FALSE(owner.has_value());
            EXPECT_EQ(diagnostic, expectedDiagnostic);
            equal_preparation_trace(expectedTrace);
            EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 0u);
            owner = JuicerCudaTest::prepare_state(assets, repaired, status, diagnostic);
            equal_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_FALSE(owner.has_value());
            EXPECT_EQ(diagnostic, "Reference source preparation: exposure production boundary fault");
            EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 1u);
            caseEvent.passed();
        }
    }

    TEST_F(PreparedStateProducts, DirectScannerBeforeInputPolicy) {
        auto params = print_params();
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        params.inputColorSpace = -1;
        params.scannerBlackLevel = std::numeric_limits<double>::quiet_NaN();
        competing_precheck(assets(), params, "ResourceDescriptorMismatch phase=8 field=scanner_output", {FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_ROUTE_PRECHECKS});
    }

    TEST_F(PreparedStateProducts, PrintGammaBeforeInputPolicy) {
        auto params = print_params();
        params.inputColorSpace = -1;
        params.printGammaFactor = 0.1;
        competing_precheck(assets(), params, "InvalidAuthoredControl component=print_development field=print_gamma_factor", {FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_CALIBRATION, FJ_PREPARE_ROUTE_PRECHECKS});
    }

    TEST_F(PreparedStateProducts, PrintInputBeforeInputPolicy) {
        auto params = print_params();
        params.inputColorSpace = -1;
        params.printExposure = std::numeric_limits<double>::infinity();
        competing_precheck(assets(), params, "ResourceDescriptorMismatch phase=4A field=print_recipe_input", {FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_CALIBRATION, FJ_PREPARE_ROUTE_PRECHECKS});
    }

    TEST_F(PreparedStateProducts, DevelopmentBeforeCalibration) {
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
        JuicerTests::GtestCohort cohort("development-before-calibration-temperature", std::vector<std::string>{"cold", "warm"});
        for (bool cold : {true, false}) {
            auto caseEvent = cohort.case_event(cold ? "cold" : "warm");
            SCOPED_TRACE(cold ? "cold" : "warm");
            equal_status(fj_legacy_assets_release_cached_payloads(assets(), nullptr), FJ_STATUS_SUCCESS);
            fj_test_calibration_read_fault(1);
            constexpr char kKey[] = "pending";
            const FjStringView selection{kKey, sizeof(kKey) - 1};
            std::array<char, 1024> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            FjNeutralCalibrationResult raw{};
            equal_status(fj_legacy_neutral_calibration_lookup(assets(), selection, selection, selection, &raw, &error), FJ_STATUS_ALLOCATION_FAILURE);
            const std::string dependencyDiagnostic(bytes.data(), error.length);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            fj_test_calibration_read_fault(0);
            equal_status(fj_legacy_assets_release_cached_payloads(assets(), nullptr), FJ_STATUS_SUCCESS);
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
            caseEvent.passed();
        }
    }

    TEST_F(PreparedStateProducts, InputAllocationBeforeScannerControl) {
        struct ClearCurveFault {
            const FjAssets* owner;
            ~ClearCurveFault() {
                fj_test_illuminant_clear_fault();
                (void)fj_test_assets_curve_fault(owner, FJ_CSV_D65, 0, nullptr);
            }
        } clearCurveFault{assets()};
        auto params = print_params();
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        params.scannerBlackLevel = std::numeric_limits<double>::quiet_NaN();
        JuicerTests::GtestCohort cohort("input-allocation-scanner-control-faults", std::vector<std::string>{"cold-allocation-failure", "cold-scanner-rejection", "warm-scanner-rejection", "released-cold-allocation-failure"});
        equal_status(fj_legacy_assets_release_cached_payloads(assets(), nullptr), FJ_STATUS_SUCCESS);
        equal_status(fj_test_assets_curve_fault(assets(), FJ_CSV_D65, 1, nullptr), FJ_STATUS_SUCCESS);
        constexpr const char* kDependencyDiagnostic = "Illuminant preparation (D65): illuminant scratch capacity";
        constexpr const char* kScannerDiagnostic = "ResourceDescriptorMismatch phase=8 field=scanner_output";
        FjStatus status{};
        std::string diagnostic;
        auto coldAllocationCase = cohort.case_event("cold-allocation-failure");
        auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_EQ(diagnostic, kDependencyDiagnostic);
        EXPECT_FALSE(owner.has_value());
        equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_ILLUMINANTS});
        std::uint32_t pending = 99;
        equal_status(fj_test_assets_curve_fault_pending(assets(), &pending, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(pending, 0u);
        coldAllocationCase.passed();
        auto coldScannerCase = cohort.case_event("cold-scanner-rejection");
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(diagnostic, kScannerDiagnostic);
        EXPECT_FALSE(owner.has_value());
        coldScannerCase.passed();
        auto warmScannerCase = cohort.case_event("warm-scanner-rejection");
        equal_status(fj_test_assets_curve_fault(assets(), FJ_CSV_D65, 1, nullptr), FJ_STATUS_SUCCESS);
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(diagnostic, kScannerDiagnostic);
        equal_status(fj_test_assets_curve_fault_pending(assets(), &pending, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(pending, 1u) << "warm entry must skip the derived acquisition";
        warmScannerCase.passed();
        auto releasedColdCase = cohort.case_event("released-cold-allocation-failure");
        equal_status(fj_legacy_assets_release_cached_payloads(assets(), nullptr), FJ_STATUS_SUCCESS);
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_EQ(diagnostic, kDependencyDiagnostic);
        equal_status(fj_test_assets_curve_fault_pending(assets(), &pending, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(pending, 0u);
        releasedColdCase.passed();
    }

    TEST_F(PreparedStateProducts, CalibrationBeforePrintGamma) {
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
        JuicerTests::GtestCohort cohort("calibration-before-gamma-temperature", std::vector<std::string>{"cold", "warm"});
        for (bool cold : {true, false}) {
            auto caseEvent = cohort.case_event(cold ? "cold" : "warm");
            SCOPED_TRACE(cold ? "cold" : "warm");
            equal_status(fj_legacy_assets_release_cached_payloads(assets(), nullptr), FJ_STATUS_SUCCESS);
            std::string diagnostic;
            FjStatus status{};
            if (!cold) {
                auto warm = print_params();
                warm.filmGammaFactor = std::numeric_limits<float>::quiet_NaN();
                auto ignored = JuicerCudaTest::prepare_state(assets(), warm, status, diagnostic);
                EXPECT_FALSE(ignored.has_value());
                equal_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
            }
            equal_status(fj_test_assets_calibration_source(assets(), source.get(), nullptr), FJ_STATUS_SUCCESS);
            equal_status(fj_test_exposure_arm_fault(5, 1, 1), FJ_STATUS_SUCCESS);
            auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
            EXPECT_FALSE(owner.has_value());
            equal_status(status, FJ_STATUS_PREPARATION_FAILURE);
            equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_CALIBRATION});
            EXPECT_EQ(diagnostic, kExpectedDiagnostic);
            EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 0u);
            equal_status(fj_legacy_assets_release_cached_payloads(assets(), nullptr), FJ_STATUS_SUCCESS);
            owner = JuicerCudaTest::prepare_state(assets(), print_params(), status, diagnostic);
            EXPECT_FALSE(owner.has_value());
            equal_status(status, FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(diagnostic, "Reference source preparation: exposure production boundary fault");
            EXPECT_EQ(fj_test_exposure_fault_consumed(5, 1), 1u);
            caseEvent.passed();
        }
    }

    TEST_F(PreparedStateProducts, EncodingOnlyReusesMediumIdentities) {
        JuicerTests::GtestCohort cohort("encoding-only-medium-identities", std::vector<std::string>{"encoded", "linear"});
        auto params = print_params();
        auto encodedCase = cohort.case_event("encoded");
        compare_product(assets(), params);
        FjStatus status{};
        std::string diagnostic;
        auto encoded = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        FjPreparedStateView first{}, second{};
        equal_status(state_view(encoded, first), FJ_STATUS_SUCCESS);
        params.outputCctfEncoding = 0;
        auto linearCase = cohort.case_event("linear");
        compare_product(assets(), params);
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
        encodedCase.passed();
        linearCase.passed();
    }

    TEST_F(PreparedStateProducts, SnapshotIntegerBooleansPreserveNonzeroCasts) {
        JuicerTests::GtestCohort cohort("snapshot-integer-bool-casts", std::vector<std::string>{"direct", "print"});
        for (bool print : {false, true}) {
            auto caseEvent = cohort.case_event(print ? "print" : "direct");
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
            compare_product(assets(), params);
            caseEvent.passed();
        }
    }

    TEST_F(PreparedStateProducts, DisabledFamiliesExcludeUnusedControls) {
        JuicerTests::GtestCohort cohort("disabled-families-unused-controls", std::vector<std::string>{"disabled-controls"});
        auto caseEvent = cohort.case_event("disabled-controls");
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
        compare_product(assets(), params);
        auto changed = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        equal_status(state_view(changed, second), FJ_STATUS_SUCCESS);
        EXPECT_EQ(first.recipe_hash, second.recipe_hash);
        EXPECT_EQ(first.optics.hash, 0u);
        EXPECT_EQ(second.optics.hash, 0u);
        EXPECT_EQ(second.grain.hash, 0u);
        EXPECT_EQ(second.dir_recipe.hash, 0u);
        caseEvent.passed();
    }

    TEST_F(PreparedStateProducts, GrainDebugSuppressesPhysicalEffects) {
        JuicerTests::GtestCohort cohort("grain-debug-physical-effects", std::vector<std::string>{"grain-debug-view"});
        auto caseEvent = cohort.case_event("grain-debug-view");
        auto params = print_params();
        params.grainControls.active = true;
        params.grainControls.debugView = 1;
        params.filmDustAmount = 0.5f;
        params.gateScratchAmount = 0.5f;
        compare_product(assets(), params);
        FjStatus status{};
        std::string diagnostic;
        auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        ASSERT_EQ(status.category, FJ_STATUS_SUCCESS) << diagnostic;
        FjPreparedStateView view{};
        equal_status(state_view(owner, view), FJ_STATUS_SUCCESS);
        EXPECT_NE(view.grain.hash, 0u);
        EXPECT_EQ(view.effects.hash, 0u);
        EXPECT_EQ(view.effects.active, 0u);
        caseEvent.passed();
    }

    TEST_F(PreparedStateProducts, EarlyAndLatePanicsReturnNoOwner) {
        struct ClearFaults {
            const FjAssets* owner;
            ~ClearFaults() {
                fj_test_exposure_clear_fault();
                (void)fj_test_assets_curve_fault(owner, FJ_CSV_D55, 0, nullptr);
            }
        } clearFaults{assets()};
        JuicerTests::GtestCohort cohort("early-late-panic-boundaries", std::vector<std::string>{"early-panic", "late-panic", "recovery"});
        const auto baseline = fj_test_prepared_live_states();
        const auto params = print_params();
        auto earlyCase = cohort.case_event("early-panic");
        equal_status(fj_legacy_assets_release_cached_payloads(assets(), nullptr), FJ_STATUS_SUCCESS);
        equal_status(fj_test_assets_curve_fault(assets(), FJ_CSV_D55, 2, nullptr), FJ_STATUS_SUCCESS);
        FjStatus status{};
        std::string diagnostic;
        auto owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_EQ(diagnostic, "asset boundary panic");
        EXPECT_FALSE(owner.has_value());
        EXPECT_EQ(fj_test_prepared_live_states(), baseline);
        equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS});
        earlyCase.passed();
        auto lateCase = cohort.case_event("late-panic");
        equal_status(fj_test_exposure_arm_fault(5, 1, 2), FJ_STATUS_SUCCESS);
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_EQ(diagnostic, "asset boundary panic");
        EXPECT_FALSE(owner.has_value());
        EXPECT_EQ(fj_test_exposure_fault_consumed(5, 2), 1u);
        EXPECT_EQ(fj_test_prepared_live_states(), baseline);
        equal_preparation_trace({FJ_PREPARE_SELECTED_FILM, FJ_PREPARE_SELECTED_PRINT, FJ_PREPARE_ILLUMINANTS, FJ_PREPARE_DEVELOPMENT, FJ_PREPARE_OBSERVER, FJ_PREPARE_RECONSTRUCTION_SOURCE, FJ_PREPARE_CALIBRATION, FJ_PREPARE_ROUTE_PRECHECKS, FJ_PREPARE_FOUNDATION, FJ_PREPARE_FILM_PAYLOAD, FJ_PREPARE_PRINT_MAIN_ILLUMINANT, FJ_PREPARE_FILM_REFERENCE});
        lateCase.passed();
        auto recoveryCase = cohort.case_event("recovery");
        owner = JuicerCudaTest::prepare_state(assets(), params, status, diagnostic);
        equal_status(status, FJ_STATUS_SUCCESS);
        EXPECT_TRUE(owner.has_value());
        EXPECT_TRUE(diagnostic.empty());
        recoveryCase.passed();
    }

    TEST_F(PreparedStateProducts, ReservedZeroIdentitiesRejectCompletion) {
        struct ClearIdentityFault {
            ~ClearIdentityFault() {
                (void)fj_test_prepared_identity_fault(0);
            }
        } clearIdentityFault;
        const std::vector<std::string> ids{"direct/identity-1", "direct/identity-2", "print/identity-1", "print/identity-2"};
        JuicerTests::GtestCohort cohort("reserved-zero-identities", ids);
        const auto baseline = fj_test_prepared_live_states();
        for (bool print : {false, true}) {
            for (std::uint32_t target : {1u, 2u}) {
                const auto id = std::string(print ? "print" : "direct") + "/identity-" + std::to_string(target);
                auto caseEvent = cohort.case_event(id);
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
                caseEvent.passed();
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

    TEST_F(PreparedStateAbi, ProjectionOracleExpectationRejectsInvalidAndDuplicateFlags) {
        char program[] = "prepared-state-tests";
        char invalid[] = "--expect-projection-oracle=present";
        char* invalidArguments[]{program, invalid, nullptr};
        int invalidCount = 2;
        EXPECT_THROW(consume_projection_oracle_expectation(invalidCount, invalidArguments), std::runtime_error);

        char loaded[] = "--expect-projection-oracle=loaded";
        char unused[] = "--expect-projection-oracle=unused";
        char* duplicateArguments[]{program, loaded, unused, nullptr};
        int duplicateCount = 3;
        EXPECT_THROW(consume_projection_oracle_expectation(duplicateCount, duplicateArguments), std::runtime_error);
    }

    TEST_F(PreparedStateAbi, FailureClearsEveryDefinedFieldAndReleaseConsumes) {
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

    TEST_F(PreparedStateAbi, DiagnosticTruncationPreservesStatusAndOwner) {
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
        const std::vector<std::string> ids{"capacity/0", "capacity/1", "capacity/4", "capacity/128"};
        JuicerTests::GtestCohort cohort("diagnostic-capacity", ids);
        for (std::size_t capacity : {0u, 1u, 4u, 128u}) {
            auto caseEvent = cohort.case_event("capacity/" + std::to_string(capacity));
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
                EXPECT_EQ(std::string(bytes.data(), count), std::string(expected.substr(0, count)));
                EXPECT_EQ(bytes[count], '\0');
            }
            caseEvent.passed();
        }
        FjPreparedStateView view{};
        equal_status(fj_test_prepared_state_view(owner.get(), &view, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_NE(view.recipe_hash, 0u);
    }

    TEST_F(PreparedStateAbi, LayoutAndRealC11Calls) {
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

    TEST_F(PreparedStateAbi, InstanceBoundaryLayoutAndC11Signatures) {
        static_assert(std::is_same_v<decltype(FjCaptureStart::capture), std::uint64_t>);
        static_assert(std::is_same_v<decltype(FjCaptureStart::kind), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjSnapshotStart::write_revision), std::uint64_t>);
        static_assert(std::is_same_v<decltype(FjSnapshotStart::kind), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjAdmission::state), FjPreparedState*>);
        static_assert(std::is_same_v<decltype(FjAdmission::input_identity), std::uint64_t>);
        static_assert(std::is_same_v<decltype(FjAdmission::build), std::uint64_t>);
        static_assert(std::is_same_v<decltype(FjAdmission::kind), std::uint32_t>);
        static_assert(std::is_same_v<decltype(FjAdmission::route), std::uint32_t>);
        static_assert(FJ_CAPTURE_INITIAL == 0 && FJ_CAPTURE_CHANGE == 1 && FJ_CAPTURE_REFRESH == 2);
        static_assert(FJ_CAPTURE_START_NONE == 0 && FJ_CAPTURE_STARTED == 1 && FJ_CAPTURE_NOT_STARTED == 2);
        static_assert(FJ_HOST_WRITE_NONE == 0 && FJ_HOST_WRITE_STARTED == 1 && FJ_HOST_WRITE_IGNORED == 2 && FJ_HOST_WRITE_SUCCEEDED == 0);
        static_assert(FJ_SNAPSHOT_NONE == 0 && FJ_SNAPSHOT_READY == 1 && FJ_SNAPSHOT_DEFERRED == 2 && FJ_SNAPSHOT_IGNORED == 3);
        static_assert(FJ_UPDATE_NONE == 0 && FJ_UPDATE_APPLIED == 1 && FJ_UPDATE_IGNORED == 2 && FJ_UPDATE_DEFERRED == 3);
        static_assert(FJ_CAPTURE_INVALID_CONTROLS == 1 && FJ_CAPTURE_HOST_FAILURE == 2 && FJ_CAPTURE_ALLOCATION_FAILURE == 3);
        static_assert(FJ_ADMISSION_NONE == 0 && FJ_ADMISSION_READY == 1 && FJ_ADMISSION_NEEDS_CAPTURE == 2 && FJ_ADMISSION_CAPTURING == 3 && FJ_ADMISSION_CAPTURE_FAILED == 4 && FJ_ADMISSION_PREPARATION_FAILED == 5 && FJ_ADMISSION_POISONED == 6 && FJ_ADMISSION_NEEDS_REFRESH == 7);
        static_assert(sizeof(FjCaptureStart) == 16 && alignof(FjCaptureStart) == 8);
        static_assert(offsetof(FjCaptureStart, capture) == 0 && offsetof(FjCaptureStart, kind) == 8);
        static_assert(sizeof(FjSnapshotStart) == 16 && alignof(FjSnapshotStart) == 8);
        static_assert(offsetof(FjSnapshotStart, write_revision) == 0 && offsetof(FjSnapshotStart, kind) == 8);
        static_assert(sizeof(FjAdmission) == 32 && alignof(FjAdmission) == 8);
        static_assert(offsetof(FjAdmission, state) == 0 && offsetof(FjAdmission, input_identity) == 8 &&
                      offsetof(FjAdmission, build) == 16 && offsetof(FjAdmission, kind) == 24 &&
                      offsetof(FjAdmission, route) == 28);
        const std::size_t cppFacts[] = {
            sizeof(FjCaptureStart), alignof(FjCaptureStart), offsetof(FjCaptureStart, capture), offsetof(FjCaptureStart, kind), sizeof(FjSnapshotStart), alignof(FjSnapshotStart), offsetof(FjSnapshotStart, write_revision), offsetof(FjSnapshotStart, kind), sizeof(FjAdmission), alignof(FjAdmission), offsetof(FjAdmission, state), offsetof(FjAdmission, input_identity), offsetof(FjAdmission, build), offsetof(FjAdmission, kind), offsetof(FjAdmission, route), FJ_CAPTURE_INITIAL, FJ_CAPTURE_CHANGE, FJ_CAPTURE_REFRESH, FJ_CAPTURE_START_NONE, FJ_CAPTURE_STARTED, FJ_CAPTURE_NOT_STARTED, FJ_HOST_WRITE_NONE, FJ_HOST_WRITE_STARTED, FJ_HOST_WRITE_IGNORED, FJ_HOST_WRITE_SUCCEEDED, FJ_SNAPSHOT_NONE, FJ_SNAPSHOT_READY, FJ_SNAPSHOT_DEFERRED, FJ_SNAPSHOT_IGNORED, FJ_UPDATE_NONE, FJ_UPDATE_APPLIED, FJ_UPDATE_IGNORED, FJ_UPDATE_DEFERRED, FJ_CAPTURE_INVALID_CONTROLS, FJ_CAPTURE_HOST_FAILURE, FJ_CAPTURE_ALLOCATION_FAILURE, FJ_ADMISSION_NONE, FJ_ADMISSION_READY, FJ_ADMISSION_NEEDS_CAPTURE, FJ_ADMISSION_CAPTURING, FJ_ADMISSION_CAPTURE_FAILED, FJ_ADMISSION_PREPARATION_FAILED, FJ_ADMISSION_POISONED, FJ_ADMISSION_NEEDS_REFRESH};
        std::size_t cCount = 0;
        const auto* cFacts = fj_test_instance_c_facts(&cCount);
        ASSERT_EQ(cCount, std::size(cppFacts));
        for (std::size_t i = 0; i < cCount; ++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(cFacts[i], cppFacts[i]);
        }
        std::size_t rustCount = 0;
        const auto* rustFacts = fj_test_instance_abi_facts(&rustCount);
        ASSERT_EQ(rustCount, 15u);
        ASSERT_GE(cCount, rustCount);
        for (std::size_t i = 0; i < rustCount; ++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(rustFacts[i], cppFacts[i]);
        }
    }

    TEST_F(PreparedStateAbi, InstanceCaptureWriteAdmissionC11Protocol) {
        constexpr std::array<std::string_view, 10> ids{
            "capture-write-refresh-ready-retain", "current-failure-applied", "obsolete-failure-ignored", "malformed-finish-consumed", "c-output-clearing-invalid-ticket-revision", "decode-failure-current-and-superseded", "capture-failure-status-pairing", "preparation-failure-retry", "panic-poisons-instance", "malformed-diagnostic-finish-consumed"};
        std::vector<std::string> cohortIds;
        cohortIds.reserve(ids.size());
        for (auto id : ids)
            cohortIds.emplace_back(id);
        JuicerTests::GtestCohort cohort("instance-c11-protocol", cohortIds);
        ParamSnapshot params;
        params.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        params.grainControls.active = false;
        const FjPreparationParams wire = preparation_params(params);
        constexpr std::array<std::uint32_t, 10> scenarios{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (std::size_t i = 0; i < scenarios.size(); ++i) {
            auto caseEvent = cohort.case_event(ids[i]);
            SCOPED_TRACE(ids[i]);
            ASSERT_EQ(fj_test_instance_c_witness(assets(), &wire, scenarios[i]), 0u);
            caseEvent.passed();
        }
    }

    TEST_F(PreparedStateAbi, InstanceDiagnosticsAreBoundedAndDestroyConsumesMalformedRecord) {
        constexpr std::string_view expected = "instance boundary panic";
        constexpr std::array<std::size_t, 5> capacities{0, 1, 8, expected.size(), expected.size() + 1};
        std::vector<std::string> ids;
        ids.reserve(capacities.size() + 1);
        for (const auto capacity : capacities)
            ids.push_back("panic/capacity/" + std::to_string(capacity));
        ids.push_back("destroy/malformed-diagnostic");
        JuicerTests::GtestCohort cohort("instance-diagnostic-capacity", ids);

        ParamSnapshot snapshot;
        snapshot.grainControls.active = false;
        const FjPreparationParams wire = preparation_params(snapshot);
        for (const auto capacity : capacities) {
            auto caseEvent = cohort.case_event("panic/capacity/" + std::to_string(capacity));
            FjInstance* instance = nullptr;
            ASSERT_EQ(fj_legacy_instance_create(&instance, nullptr).category, FJ_STATUS_SUCCESS);
            ASSERT_NE(instance, nullptr);
            FjCaptureStart capture{};
            ASSERT_EQ(fj_legacy_instance_begin_capture(instance, FJ_CAPTURE_INITIAL, &capture, nullptr).category, FJ_STATUS_SUCCESS);
            ASSERT_EQ(capture.kind, FJ_CAPTURE_STARTED);
            FjSnapshotStart read{};
            ASSERT_EQ(fj_legacy_instance_begin_snapshot(instance, capture.capture, &read, nullptr).category, FJ_STATUS_SUCCESS);
            ASSERT_EQ(read.kind, FJ_SNAPSHOT_READY);
            std::uint32_t update = FJ_UPDATE_NONE;
            ASSERT_EQ(fj_legacy_instance_install(instance, capture.capture, read.write_revision, &wire, 11, &update, nullptr).category, FJ_STATUS_SUCCESS);
            ASSERT_EQ(update, FJ_UPDATE_APPLIED);

            std::array<char, 32> bytes{};
            bytes.fill('#');
            FjErrorBuffer error{capacity == 0 ? reinterpret_cast<char*>(std::uintptr_t{1}) : bytes.data(), capacity, 99};
            FjAdmission admission{};
            ASSERT_EQ(fj_test_spectral_arm_fault(3, 1, 2).category, FJ_STATUS_SUCCESS);
            const auto status = fj_legacy_instance_admit(instance, assets(), &admission, &error);
            EXPECT_EQ(status.category, FJ_STATUS_INTERNAL_FAILURE);
            EXPECT_EQ(admission.kind, FJ_ADMISSION_POISONED);
            EXPECT_EQ(admission.state, nullptr);
            const auto count = capacity == 0 ? 0 : std::min(expected.size(), capacity - 1);
            EXPECT_EQ(error.length, count);
            if (capacity != 0) {
                EXPECT_EQ(std::string(bytes.data(), count), expected.substr(0, count));
                EXPECT_EQ(bytes[count], '\0');
            }

            // Destroy consumes the instance even if its diagnostic record is malformed.
            FjErrorBuffer malformed{nullptr, 1, 99};
            EXPECT_EQ(fj_legacy_instance_destroy(instance, &malformed).category, FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(malformed.length, 0u);
            caseEvent.passed();
        }

        auto destroyEvent = cohort.case_event("destroy/malformed-diagnostic");
        FjInstance* instance = nullptr;
        ASSERT_EQ(fj_legacy_instance_create(&instance, nullptr).category, FJ_STATUS_SUCCESS);
        FjErrorBuffer malformed{nullptr, 1, 99};
        EXPECT_EQ(fj_legacy_instance_destroy(instance, &malformed).category, FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(malformed.length, 0u);
        destroyEvent.passed();
    }

    TEST_F(PreparedStateAbi, CallScopeLayoutAndRustC11Witnesses) {
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

    TEST_F(PreparedStateAbi, HostPreparationFacadeIsScopedAndC11Callable) {
        const auto before = fj_test_gaussian_kernel_live_owners();
        EXPECT_EQ(fj_test_host_preparation_c_call(), 1);
        EXPECT_EQ(fj_test_gaussian_kernel_live_owners(), before);
        EXPECT_EQ(fj_test_scanner_lut_live_owners(), 0u);
    }

    TEST_F(PreparedStateAbi, InputHullFaultsBelongToActualOwnerAndRetainedWarmEntry) {
        constexpr std::array<const char*, 4> kRequirements{
            "cie1931_380_700_and_d65", "finite_locus_ray_intersections", "finite_inscribed_scale", "stable_identity"};
        const auto release = [](FjInputHullOwner* owner) {
            (void)fj_test_input_hull_release(owner, nullptr);
        };
        using HullHold = std::unique_ptr<FjInputHullOwner, decltype(release)>;
        std::vector<std::string> ids;
        for (std::uint32_t fault = 1; fault <= 6; ++fault)
            ids.push_back("fault/" + std::to_string(fault));
        JuicerTests::GtestCohort cohort("input-hull-faults", ids);
        for (std::uint32_t fault = 1; fault <= 6; ++fault) {
            auto caseEvent = cohort.case_event("fault/" + std::to_string(fault));
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
            caseEvent.passed();
        }
    }

    TEST_F(PreparedStateAbi, CacheFamiliesShareBackingAndOutliveAssets) {
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
        const auto expectedOracleLoad = consume_projection_oracle_expectation(argc, argv);
        if (::testing::GTEST_FLAG(list_tests))
            return RUN_ALL_TESTS();
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            for (const auto name : kProjectionCases)
                JuicerTests::list_case("prepared_projection", name);
            return 0;
        }
        JuicerTests::CaseLog case_log;
        if (argc != 3 && argc != 6)
            return 2;
        artifactRoot = argv[2];
        std::string preset;
        bool checkProjection = false;
        std::filesystem::path oraclePath;
        if (argc == 6) {
            preset = argv[5];
            const std::string mode = argv[3];
            if (mode == "--check-state-projection") {
                oraclePath = argv[4];
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
            oraclePath = std::filesystem::path(__FILE__).parent_path() / "fixtures/prepared-state-projection" / (preset + ".json");
        }
        if (preset != FJ_PREPARED_STATE_PRESET)
            throw std::runtime_error("prepared-state preset does not match its compiled native profile");
        const auto sourceRoot = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
        if (!std::filesystem::equivalent(sourceRoot / "Resources", std::filesystem::path(argv[1])))
            throw std::runtime_error("prepared-state tests require the configured source Resources");
        ::testing::AddGlobalTestEnvironment(new ProjectionOracleEnvironment(std::move(oraclePath), preset, checkProjection));
        const int result = RUN_ALL_TESTS();
        if (expectedOracleLoad.has_value() && projectionOracle.has_value() != *expectedOracleLoad)
            throw std::runtime_error("prepared-state projection oracle load expectation mismatch");
        if (result != 0)
            return result;
        const bool requestedFullProductFilter = ::testing::GTEST_FLAG(filter) == "PreparedStateProducts.*";
        if (checkProjection || requestedFullProductFilter || all_projection_case_tests_selected()) {
            if (observedProjectionCases.size() != kProjectionCases.size())
                throw std::runtime_error("prepared-state projection did not execute all 31 rows");
            for (std::size_t i = 0; i < kProjectionCases.size(); ++i)
                if (observedProjectionCases[i] != kProjectionCases[i])
                    throw std::runtime_error("prepared-state projection row order mismatch");
        }
        case_log.finish();
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
