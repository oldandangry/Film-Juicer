#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "RustAssetBridge.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "juicer_test_api.h"
#include "case_results.h"
#include "print_illuminant_fixture.h"
#include "exposure_fixture.h"
#include "juicer_cuda_owner.h"
#include "spectral_source_fixture.h"
#include "test_asset_workspace.h"

extern "C" int fj_test_spectral_preparation_abi_c(void);
namespace {
    using Json = nlohmann::json;
    void require(bool condition, const std::string& why) {
        if (!condition)
            throw std::runtime_error(why);
    }
    void status(FjStatus actual, std::uint32_t expected) {
        require(actual.category == expected && actual.api == FJ_API_NONE && actual.native_code == 0, "spectral typed status " + std::to_string(actual.category) + " expected " + std::to_string(expected));
    }
    struct Diagnostic {
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
    };
    template <typename Range>
    Json bits(const Range& values) {
        Json out = Json::array();
        for (float v : values)
            out.push_back(std::bit_cast<std::uint32_t>(v));
        return out;
    }
    template <std::size_t N>
    std::array<float, N> decode(const Json& values) {
        require(values.size() == N, "fixed input size");
        std::array<float, N> out{};
        for (std::size_t i = 0; i < N; ++i)
            out[i] = std::bit_cast<float>(values[i].get<std::uint32_t>());
        return out;
    }
    template <std::size_t N>
    FjFloatSpan span(const std::array<float, N>& values) {
        return {values.data(), N};
    }
    void exact(const Json& actual, const Json& expected, const std::string& why) {
        require(actual.size() == expected.size(), why + " count");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto a = actual[i].get<std::uint32_t>(), b = expected[i].get<std::uint32_t>();
            if (a != b)
                require(std::isnan(std::bit_cast<float>(a)) && std::isnan(std::bit_cast<float>(b)), why + " first difference at " + std::to_string(i) + ": " + std::to_string(a) + " expected " + std::to_string(b));
        }
    }
    struct Input {
        std::array<float, 243> dyes{};
        std::array<std::array<float, 81>, 3> observer{};
        std::array<float, 81> illuminant{}, minimum{}, midpoint{};
        FjSpectralInput view{};
        explicit Input(const Json& j) {
            for (std::size_t i = 0; i < 81; ++i) {
                auto row = decode<3>(j.at("dyes")[i]);
                std::copy(row.begin(), row.end(), dyes.data() + i * 3);
            }
            for (std::size_t c = 0; c < 3; ++c)
                observer[c] = decode<81>(j.at("observer")[c]);
            illuminant = decode<81>(j.at("illuminant"));
            view = {span(dyes), {span(observer[0]), span(observer[1]), span(observer[2])}, span(illuminant), {nullptr, 0}, {nullptr, 0}, j.at("illuminant_hash").get<std::uint64_t>()};
            if (!j.at("baseline_min").empty()) {
                minimum = decode<81>(j.at("baseline_min"));
                view.baseline_min = span(minimum);
            }
            if (!j.at("baseline_mid").empty()) {
                midpoint = decode<81>(j.at("baseline_mid"));
                view.baseline_mid = span(midpoint);
            }
        }
    };
    void compare_tables(const FjSpectralTables& t, const Json& j) {
        exact(bits(t.lambda_nm), j.at("lambda"), "lambda");
        exact(bits(t.illuminant), j.at("illum"), "illuminant");
        const char* obs[] = {"xbar", "ybar", "zbar"};
        const char* weighted[] = {"ax", "ay", "az"};
        const char* dyes[] = {"eps_c", "eps_m", "eps_y"};
        for (std::size_t c = 0; c < 3; ++c) {
            exact(bits(t.observer_xyz[c]), j.at(obs[c]), obs[c]);
            exact(bits(t.weighted_xyz[c]), j.at(weighted[c]), weighted[c]);
            exact(bits(t.dyes_cmy[c]), j.at(dyes[c]), dyes[c]);
        }
        exact(bits(t.baseline_min), j.at("base_density_min"), "baseline min");
        exact(bits(t.baseline_mid), j.at("base_density_mid"), "baseline mid");
        exact(bits(std::array{t.delta_lambda}), j.at("delta_lambda"), "delta");
        exact(bits(std::array{t.inv_yn}), j.at("inv_yn"), "invYn");
        exact(bits(t.white_xyz), j.at("white_xyz"), "table white");
        exact(bits(t.reference_white_xyz), j.at("ref_illum_white_xyz"), "reference white");
        require(t.has_baseline == (j.at("has_baseline").get<bool>() ? 1u : 0u) && t.illuminant_hash == j.at("illuminant_hash") && t.tables_hash == j.at("tables_hash"), "exact table flags/identities");
    }
    void tables(const Json& fixture) {
        for (const auto& row : fixture.at("leaf").at("tables")) {
            const auto row_id = std::to_string(row.at("input").at("id").get<int>());
            Input in(row.at("input"));
            for (bool facade : {false, true}) {
                JuicerTests::CaseEvent event("spectral.tables", std::string(facade ? "facade/" : "legacy/") + row_id);
                FjSpectralTables out{};
                Diagnostic d;
                status(facade ? fj_test_spectral_tables(&in.view, &out, &d.error) : fj_legacy_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_SUCCESS);
                compare_tables(out, row.at("expected"));
                require(d.error.length == 0, "success clears diagnostic");
                event.passed();
            }
        }
        Input reference(fixture.at("leaf").at("tables")[0].at("input"));
        FjSpectralTables baseline{};
        Diagnostic d;
        status(fj_legacy_spectral_tables(&reference.view, &baseline, &d.error), FJ_STATUS_SUCCESS);
        for (const auto& row : fixture.at("dependencies")) {
            JuicerTests::CaseEvent event("spectral.dependencies", std::to_string(row.at("input").at("id").get<int>()));
            Input in(row.at("input"));
            FjSpectralTables out{};
            status(fj_legacy_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_SUCCESS);
            compare_tables(out, row.at("expected"));
            require(out.illuminant_hash == baseline.illuminant_hash && out.tables_hash != baseline.tables_hash, "isolated source dependency changes table family identity with supplied identity preserved");
            event.passed();
        }
        require(fixture.at("leaf").at("bool_encoding") == Json::array({1, 0, 1}), "frozen bool byte encoding");
    }
    void whites(const Json& fixture) {
        for (const auto& row : fixture.at("leaf").at("tables")) {
            const auto row_id = std::to_string(row.at("input").at("id").get<int>());
            Input in(row.at("input"));
            const FjSpectralWhiteInput input{in.view.observer, in.view.illuminant};
            for (bool facade : {false, true}) {
                JuicerTests::CaseEvent event("spectral.white", std::string(facade ? "facade/" : "legacy/") + row_id);
                FjSpectralWhite out{};
                FjSpectralWhiteFailure failure{};
                Diagnostic d;
                const bool built = row.at("white").at("built");
                status(facade ? fj_test_spectral_white(&input, &out, &failure, &d.error) : fj_legacy_spectral_white(&input, &out, &failure, &d.error), built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (built) {
                    const auto& w = row.at("white");
                    exact(bits(std::array{out.normalization}), w.at("normalization"), "white normalization");
                    exact(bits(out.white_xyz), w.at("xyz"), "white XYZ");
                    exact(bits(out.white_xy), w.at("xy"), "white xy");
                    require(out.hash == w.at("hash") && failure.reason == 0 && failure.luminance_sum == 0, "white identity/completion");
                } else {
                    require(out.hash == 0 && out.normalization == 0 && out.white_xyz[0] == 0 && out.white_xyz[1] == 0 && out.white_xyz[2] == 0 && out.white_xy[0] == 0 && out.white_xy[1] == 0, "failed white cleared");
                    require(failure.reason == row.at("white_failure").at("reason") && std::bit_cast<std::uint64_t>(failure.luminance_sum) == row.at("white_failure").at("luminance_sum"), "closed native white failure/sum");
                    require(d.error.length > 0, "computed white failure diagnostic");
                }
                event.passed();
            }
        }
    }
    void inverses(const Json& fixture) {
        for (const auto& row : fixture.at("leaf").at("inverse")) {
            const auto row_id = std::to_string(row.at("id").get<int>());
            const auto x = decode<81>(row.at("input")[0]), y = decode<81>(row.at("input")[1]), z = decode<81>(row.at("input")[2]);
            const FjSpectralSInput input{span(x), span(y), span(z)};
            for (bool facade : {false, true}) {
                JuicerTests::CaseEvent event("spectral.inverse", std::string(facade ? "facade/" : "legacy/") + row_id);
                FjSpectralInverse out{};
                Diagnostic d;
                status(facade ? fj_test_spectral_s_inverse(&input, &out, &d.error) : fj_legacy_spectral_s_inverse(&input, &out, &d.error), FJ_STATUS_SUCCESS);
                exact(bits(out.matrix), row.at("expected"), "S inverse " + std::to_string(row.at("id").get<int>()));
                event.passed();
            }
        }
    }
    void edge(const Json& fixture, const std::filesystem::path& resources) {
        require(fj_test_spectral_preparation_abi_c() == 1, "compiled C11 prototypes/layouts");
        Input in(fixture.at("leaf").at("tables")[0].at("input"));
        FjSpectralTables out{};
        Diagnostic d;
        auto malformed = in.view;
        JuicerTests::Cohort raw_input_rows("spectral.edge.raw_input", {"illuminant-count-0", "illuminant-count-80", "illuminant-count-82", "missing-baseline-min", "misaligned-dyes", "illuminant-count-overflow", "illuminant-address-overflow", "null-input", "null-output", "null-diagnostic"});
        for (auto count : {0u, 80u, 82u}) {
            auto event = raw_input_rows.case_event("illuminant-count-" + std::to_string(count));
            malformed.illuminant.count = count;
            out.tables_hash = 123;
            status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(out.tables_hash == 0 && out.lambda_nm[0] == 0 && d.error.length > 0, "malformed tables cleared");
            event.passed();
        }
        malformed = in.view;
        malformed.baseline_min = {nullptr, 0};
        malformed.baseline_mid = span(in.midpoint);
        auto baseline_event = raw_input_rows.case_event("missing-baseline-min");
        status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        baseline_event.passed();
        malformed = in.view;
        malformed.dyes_cmy.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(in.dyes.data()) + 1);
        auto dyes_event = raw_input_rows.case_event("misaligned-dyes");
        status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        dyes_event.passed();
        malformed = in.view;
        malformed.illuminant.count = std::numeric_limits<std::size_t>::max();
        auto overflow_count_event = raw_input_rows.case_event("illuminant-count-overflow");
        status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        overflow_count_event.passed();
        malformed = in.view;
        malformed.illuminant.data = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3);
        auto overflow_address_event = raw_input_rows.case_event("illuminant-address-overflow");
        status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        overflow_address_event.passed();
        auto null_input_event = raw_input_rows.case_event("null-input");
        status(fj_legacy_spectral_tables(nullptr, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        null_input_event.passed();
        auto null_output_event = raw_input_rows.case_event("null-output");
        status(fj_legacy_spectral_tables(&in.view, nullptr, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        null_output_event.passed();
        auto null_diagnostic_event = raw_input_rows.case_event("null-diagnostic");
        status(fj_legacy_spectral_tables(&in.view, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        null_diagnostic_event.passed();
        const FjSpectralWhiteInput w{in.view.observer, in.view.illuminant};
        FjSpectralWhite white{};
        FjSpectralWhiteFailure failure{};
        const FjSpectralSInput s{in.view.observer.x, in.view.observer.y, in.view.observer.z};
        FjSpectralInverse inverse{};
        const auto call = [&](std::uint32_t op) {
            switch (op) {
                case 1:
                    return fj_legacy_spectral_tables(&in.view, &out, &d.error);
                case 2:
                    return fj_legacy_spectral_white(&w, &white, &failure, &d.error);
                default:
                    return fj_legacy_spectral_s_inverse(&s, &inverse, &d.error);
            }
        };
        std::vector<std::string> operation_fault_ids;
        for (std::uint32_t op = 1; op <= 3; ++op)
            for (std::uint32_t fault = 1; fault <= 4; ++fault)
                operation_fault_ids.emplace_back(std::to_string(op) + "/" + std::to_string(fault));
        JuicerTests::Cohort operation_fault_rows("spectral.edge.operation_fault", operation_fault_ids);
        for (std::uint32_t op = 1; op <= 3; ++op)
            for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                auto event = operation_fault_rows.case_event(std::to_string(op) + "/" + std::to_string(fault));
                status(fj_test_spectral_arm_fault(op, 2, fault), FJ_STATUS_SUCCESS);
                status(call(op), FJ_STATUS_SUCCESS);
                const std::uint32_t category[] = {0, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE};
                status(call(op), category[fault]);
                require(out.tables_hash == 0 || op != 1, "fault tables clear");
                require((white.hash == 0 && failure.reason == 0 && failure.luminance_sum == 0) || op != 2, "fault white/meta clear");
                require(inverse.matrix[0] == 0 || op != 3, "fault inverse clear");
                status(call(op), FJ_STATUS_SUCCESS);
                event.passed();
            }
        status(fj_test_spectral_arm_fault(1, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_SUCCESS);
        status(call(1), FJ_STATUS_UNSUPPORTED_INPUT);
        status(call(1), FJ_STATUS_SUCCESS);
        fj_test_spectral_arm_facade_fault();
        status(call(1), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_INTERNAL_FAILURE);
        status(fj_test_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_arm_fault(1, 1, 1), FJ_STATUS_SUCCESS);
        std::thread other([&] {
            FjSpectralTables result{};
            Diagnostic diag;
            status(fj_legacy_spectral_tables(&in.view, &result, &diag.error), FJ_STATUS_SUCCESS);
        });
        other.join();
        status(call(1), FJ_STATUS_UNSUPPORTED_INPUT);
        status(call(1), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_arm_fault(1, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_legacy_spectral_tables(nullptr, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(call(1), FJ_STATUS_UNSUPPORTED_INPUT);
        status(call(1), FJ_STATUS_SUCCESS);
        struct CsvOwners {
            FjAssets* assets = nullptr;
            FjCsvPairs* pairs = nullptr;
            ~CsvOwners() {
                if (pairs && fj_legacy_csv_release(pairs, nullptr).category != FJ_STATUS_SUCCESS)
                    std::abort();
                if (assets && fj_legacy_assets_destroy(assets, nullptr).category != FJ_STATUS_SUCCESS)
                    std::abort();
            }
        } owners;
        const JuicerAssets::NativePathArgument path(resources);
        status(fj_legacy_assets_create(path.view(), &owners.assets, &d.error), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_arm_fault(1, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
        status(fj_test_illuminant_arm_fault(FJ_TEST_ILLUMINANT_EQUAL_ENERGY, 1, 1), FJ_STATUS_SUCCESS);
        fj_test_csv_fault(1);
        status(call(1), FJ_STATUS_UNSUPPORTED_INPUT);
        const std::array<float, 3> nominal{1.0f, 1.0f, 1.0f};
        std::array<float, 9> matrix{};
        status(fj_legacy_cat02_matrix(nominal.data(), nominal.data(), matrix.data()), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_cat02_matrix(nominal.data(), nominal.data(), matrix.data()), FJ_STATUS_SUCCESS);
        FjIlluminant light{};
        status(fj_legacy_illuminant_equal_energy(&light, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_illuminant_equal_energy(&light, &d.error), FJ_STATUS_SUCCESS);
        status(fj_legacy_csv_acquire(owners.assets, FJ_CSV_D65, &owners.pairs, &d.error), FJ_STATUS_INTERNAL_FAILURE);
        require(owners.pairs == nullptr, "CSV fault publishes no owner independently of spectral math");
        status(fj_legacy_csv_acquire(owners.assets, FJ_CSV_D65, &owners.pairs, &d.error), FJ_STATUS_SUCCESS);
        status(call(1), FJ_STATUS_SUCCESS);
        fj_test_spectral_clear_fault();
    }
} // namespace
int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            const Json listed = load_exposure_qualified_fixture(argv[2]);
            const std::string selected = argv[1];
            if (selected == "tables") {
                for (const auto& row : listed.at("leaf").at("tables"))
                    for (const char* caller : {"legacy", "facade"})
                        JuicerTests::list_case("spectral.tables", std::string(caller) + "/" + std::to_string(row.at("input").at("id").get<int>()));
                for (const auto& row : listed.at("dependencies"))
                    JuicerTests::list_case("spectral.dependencies", std::to_string(row.at("input").at("id").get<int>()));
            } else if (selected == "white") {
                for (const auto& row : listed.at("leaf").at("tables"))
                    for (const char* caller : {"legacy", "facade"})
                        JuicerTests::list_case("spectral.white", std::string(caller) + "/" + std::to_string(row.at("input").at("id").get<int>()));
            } else if (selected == "inverse") {
                for (const auto& row : listed.at("leaf").at("inverse"))
                    for (const char* caller : {"legacy", "facade"})
                        JuicerTests::list_case("spectral.inverse", std::string(caller) + "/" + std::to_string(row.at("id").get<int>()));
            } else if (selected == "products") {
                JuicerTests::list_case("spectral_preparation", selected);
                for (const auto& row : listed.at("products")) {
                    const auto id = "polarity/" + std::to_string(row.at("polarity").get<int>()) + "/route/" +
                                    std::to_string(row.at("route").get<int>()) + "/method/" +
                                    std::to_string(row.at("method").get<int>()) + "/label/" +
                                    row.at("label").get<std::string>() + "/variant/" + std::to_string(row.value("variant", 0));
                    JuicerTests::list_case("spectral.product", id);
                }
            } else if (selected == "admission") {
                JuicerTests::list_case("spectral_preparation", selected);
                for (int route = 0; route < 4; ++route)
                    for (std::uint32_t operation = 1; operation <= 3; ++operation)
                        for (std::uint32_t fault = 1; fault <= 4; ++fault)
                            JuicerTests::list_case("spectral.admission", "route/" + std::to_string(route) + "/operation/" + std::to_string(operation) + "/fault/" + std::to_string(fault));
                for (int route = 0; route < 4; ++route)
                    for (int method = 0; method < 3; ++method)
                        for (std::uint32_t operation = 1; operation <= 3; ++operation)
                            JuicerTests::list_case("spectral.admission.count", "route/" + std::to_string(route) + "/method/" + std::to_string(method) + "/operation/" + std::to_string(operation));
            } else if (selected == "edge") {
                for (const auto* id : {"illuminant-count-0", "illuminant-count-80", "illuminant-count-82", "missing-baseline-min", "misaligned-dyes", "illuminant-count-overflow", "illuminant-address-overflow", "null-input", "null-output", "null-diagnostic"})
                    JuicerTests::list_case("spectral.edge.raw_input", id);
                for (std::uint32_t operation = 1; operation <= 3; ++operation)
                    for (std::uint32_t fault = 1; fault <= 4; ++fault)
                        JuicerTests::list_case("spectral.edge.operation_fault", std::to_string(operation) + "/" + std::to_string(fault));
                JuicerTests::list_case("spectral_preparation", selected);
            } else {
                JuicerTests::list_case("spectral_preparation", selected);
            }
            return 0;
        }
        require(argc == 3 || argc == 5, "group, preset fixture and optional product resource/scratch required");
        const Json fixture = PrintIlluminantFixtures::apply(argv[2], load_exposure_qualified_fixture(argv[2]));
        const std::string group = argv[1];
        if (group == "tables")
            tables(fixture);
        else if (group == "white")
            whites(fixture);
        else if (group == "inverse")
            inverses(fixture);
        else if (group == "edge" && argc == 5) {
            JuicerTests::CaseEvent event("spectral_preparation", "edge");
            edge(fixture, argv[3]);
            event.passed();
        } else if ((group == "products" || group == "admission") && argc == 5) {
            JuicerTests::CaseEvent event("spectral_preparation", group);
            fj_test_spectral_products(fixture, {.resources = argv[3], .scratch = argv[4]}, group == "admission");
            event.passed();
        } else
            throw std::runtime_error("unknown spectral group");
        std::puts("PASS spectral frozen native values/contract");
        int exitCode = 0;
        if (group == "products" || group == "admission") {
            const bool ownersDrained = fj_test_spectral_live_owners() == 0 && fj_test_assets_live_owners() == 0;
            exitCode = JuicerTestSupport::test_asset_workspaces.finish(0, ownersDrained);
        }
        case_log.finish();
        return exitCode;
    } catch (const JuicerCuda::ExecutionFailure& e) {
        std::fprintf(stderr, "spectral execution failure category=%u: %s\n", e.failure.status.category, e.failure.diagnostic.c_str());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
