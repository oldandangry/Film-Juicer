#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "juicer_test_api.h"
#include "case_results.h"

#define FJ_EXPOSURE_SIZE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_EXPOSURE_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "exposure_abi_facts.inc"
#undef FJ_EXPOSURE_SIZE
#undef FJ_EXPOSURE_FIELD

#include "exposure_fixture.h"
#include "print_illuminant_fixture.h"
#include "spectral_source_fixture.h"
#include "test_asset_workspace.h"

extern "C" int fj_test_exposure_abi_c(const FjReferenceWhiteInput*, FjReferenceWhite*, const FjSensitivityInput*, FjSensitivity*, FjSensitivityFailure*, FjErrorBuffer*);

void fj_test_exposure_admission(const std::filesystem::path&);

namespace {
    using Json = nlohmann::json;
    void require(bool condition, const std::string& why) {
        if (!condition)
            throw std::runtime_error(why);
    }
    void status(FjStatus actual, std::uint32_t expected) {
        require(actual.category == expected && actual.api == FJ_API_NONE && actual.native_code == 0,
                "exposure status " + std::to_string(actual.category) + " expected " + std::to_string(expected));
    }
    struct Diagnostic {
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
    };
    template <std::size_t N>
    std::array<float, N> decode(const Json& values) {
        require(values.size() == N, "fixed exposure input size");
        std::array<float, N> result{};
        for (std::size_t i = 0; i < N; ++i)
            result[i] = std::bit_cast<float>(values[i].get<std::uint32_t>());
        return result;
    }
    template <std::size_t N>
    FjFloatSpan span(const std::array<float, N>& values) {
        return {values.data(), N};
    }
    void exact(float actual, std::uint32_t expected) {
        const auto bits = std::bit_cast<std::uint32_t>(actual);
        require(bits == expected || (std::isnan(actual) && std::isnan(std::bit_cast<float>(expected))),
                "exposure bit mismatch " + std::to_string(bits) + " expected " + std::to_string(expected));
    }
    std::string fixture_row_id(const Json& row, std::size_t ordinal) {
        if (!row.contains("id"))
            return std::to_string(ordinal);
        const auto& id = row.at("id");
        if (id.is_string())
            return id.get<std::string>();
        if (id.is_number_unsigned())
            return std::to_string(id.get<std::uint64_t>());
        if (id.is_number_integer())
            return std::to_string(id.get<std::int64_t>());
        throw std::runtime_error("exposure fixture row id must be a string or integer");
    }
    void cleared(const FjSensitivity& result, const FjSensitivityFailure& failure) {
        for (const auto& rgb : result.values_rgb)
            for (float v : rgb)
                require(v == 0.0f, "failed sensitivity published values");
        require(result.hash == 0 && result.mallett_green_scale == 0.0f, "failed sensitivity published identity/scale");
        require(failure.hash_failure <= 2, "failure metadata closed tag");
    }
    struct SensitivityInput {
        std::array<std::array<float, 3>, 81> linear{};
        std::array<float, 81> illuminant{}, reference{};
        std::array<float, 4> window{};
        FjSensitivityInput view{};
        explicit SensitivityInput(const Json& row) {
            for (std::size_t i = 0; i < 81; ++i)
                linear[i] = decode<3>(row.at("linear")[i]);
            illuminant = decode<81>(row.at("illuminant"));
            reference = decode<81>(row.at("reference"));
            window = decode<4>(row.at("window"));
            const auto uv = decode<3>(row.at("uv")), ir = decode<3>(row.at("ir"));
            const bool activeWindow = row.at("method") == 0 && row.at("apply_window").get<bool>();
            const bool white = activeWindow && row.at("reference_valid").get<bool>();
            view = {{linear.front().data(), 243}, span(illuminant), row.at("method").get<std::uint32_t>(), row.at("active").get<bool>() ? 1u : 0u, row.at("apply_window").get<bool>() ? 1u : 0u, {uv[0], uv[1], uv[2]}, {ir[0], ir[1], ir[2]}, {activeWindow ? window.data() : nullptr, activeWindow ? 4u : 0u}, {white ? reference.data() : nullptr, white ? 81u : 0u}};
        }
    };
    void sensitivity(const Json& fixture) {
        std::size_t ordinal = 0;
        for (const auto& row : fixture.at("sensitivity")) {
            const auto row_id = fixture_row_id(row, ordinal++);
            SensitivityInput input(row.at("input"));
            const auto& expected = row.at("expected");
            for (bool facade : {false, true}) {
                JuicerTests::CaseEvent event("exposure.sensitivity", std::string(facade ? "facade/" : "legacy/") + row_id);
                FjSensitivity result{};
                FjSensitivityFailure failure{};
                Diagnostic error;
                const auto call = facade ? fj_test_exposure_sensitivity : fj_legacy_exposure_sensitivity;
                const bool built = expected.at("built").get<bool>();
                status(call(&input.view, &result, &failure, &error.error), built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (built) {
                    for (std::size_t i = 0; i < 81; ++i)
                        for (std::size_t c = 0; c < 3; ++c)
                            exact(result.values_rgb[i][c], expected.at("expected")[i][c].get<std::uint32_t>());
                    exact(result.mallett_green_scale, expected.at("scale").get<std::uint32_t>());
                    require(result.hash == expected.at("hash").get<std::uint64_t>() && !failure.hash_failure && !failure.hash_sample_index, "complete sensitivity identity/metadata");
                } else {
                    cleared(result, failure);
                    const bool hashFailure = !expected.at("hash_failure_index").is_null();
                    require(failure.hash_failure == (hashFailure ? 1u : 0u), "earlier hash metadata");
                    if (hashFailure)
                        require(failure.hash_sample_index == expected.at("hash_failure_index").get<std::uint32_t>(), "first nonfinite hash index");
                }
                event.passed();
            }
        }
    }
    void reference(const Json& fixture, const std::filesystem::path& resources) {
        const JuicerTestSupport::SpectralSourceAssets assets(resources);
        const JuicerTestSupport::HanatosSource real(assets.get());
        std::vector<float> spectra(std::size_t{192} * 192u * 81u);
        for (std::size_t i = 0; i < spectra.size(); ++i)
            spectra[i] = std::bit_cast<float>(0x3e000123u + static_cast<std::uint32_t>(i % 4093u) * 173u);
        std::size_t ordinal = 0;
        for (const auto& row : fixture.at("reference")) {
            const auto row_id = fixture_row_id(row, ordinal++);
            const bool acceptedHanatos = row.at("source") == "accepted_hanatos";
            const bool altered = row.contains("source_index");
            const auto index = altered ? row.at("source_index").get<std::size_t>() : 0u;
            std::vector<float> alteredAccepted;
            FjFloatSpan source{};
            if (acceptedHanatos) {
                source = real.view().samples;
                if (altered) {
                    alteredAccepted.assign(source.data, source.data + source.count);
                    source = {alteredAccepted.data(), alteredAccepted.size()};
                }
            } else {
                source = {spectra.data(), spectra.size()};
            }
            const float saved = altered ? source.data[index] : 0.0f;
            if (altered) {
                if (acceptedHanatos) {
                    alteredAccepted[index] = std::bit_cast<float>(row.at("source_value").get<std::uint32_t>());
                } else {
                    spectra[index] = std::bit_cast<float>(row.at("source_value").get<std::uint32_t>());
                    source = {spectra.data(), spectra.size()};
                }
            }
            const auto white = decode<3>(row.at("white"));
            const float blur = std::bit_cast<float>(row.at("blur").get<std::uint32_t>());
            const FjReferenceWhiteInput input{source, {white[0], white[1], white[2]}, blur};
            for (bool facade : {false, true}) {
                JuicerTests::CaseEvent event("exposure.reference", std::string(facade ? "facade/" : "legacy/") + row_id);
                FjReferenceWhite result{};
                Diagnostic error;
                status((facade ? fj_test_reconstruction_reference_white : fj_legacy_reconstruction_reference_white)(&input, &result, &error.error), row.at("built").get<bool>() ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (row.at("built").get<bool>()) {
                    for (std::size_t i = 0; i < 81; ++i)
                        exact(result.samples[i], row.at("expected")[i].get<std::uint32_t>());
                } else {
                    for (float value : result.samples)
                        require(value == 0.0f, "failed white publishes no samples");
                    require(std::string(error.bytes.data(), error.error.length) == row.at("diagnostic").get<std::string>(), "reference failure requirement");
                }
                event.passed();
            }
            // The raw facade above remains the sole expected-value producer.
            if (altered && !acceptedHanatos)
                spectra[index] = saved;
        }
    }
    void boundary(const Json& fixture) {
        SensitivityInput input(fixture.at("sensitivity")[0].at("input"));
        Diagnostic error;
        FjSensitivity result{};
        FjSensitivityFailure failure{};
        auto invalid = input.view;
        JuicerTests::Cohort sensitivity_faults("exposure.boundary.sensitivity_fault", 4);
        for (std::uint32_t fault = 1; fault <= 4; ++fault) {
            auto row_event = sensitivity_faults.case_event(std::to_string(fault - 1));
            status(fj_test_exposure_arm_fault(2, 1, fault), FJ_STATUS_SUCCESS);
            const auto expected = std::array{FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE};
            status(fj_legacy_exposure_sensitivity(&input.view, &result, &failure, &error.error), expected[fault - 1]);
            require(fj_test_exposure_fault_consumed(2, fault) == 1, "same-thread raw fault consumption witness");
            cleared(result, failure);
            status(fj_legacy_exposure_sensitivity(&input.view, &result, &failure, &error.error), FJ_STATUS_SUCCESS);
            row_event.passed();
        }
        JuicerTests::Cohort malformed_sensitivity("exposure.boundary.sensitivity_malformed", 11);
        for (std::size_t bad = 0; bad < 11; ++bad) {
            auto row_event = malformed_sensitivity.case_event(std::to_string(bad));
            invalid = input.view;
            switch (bad) {
                case 0:
                    invalid.method = 3;
                    break;
                case 1:
                    invalid.band_pass_active = 2;
                    break;
                case 2:
                    invalid.apply_window = 2;
                    break;
                case 3:
                    invalid.linear_sensitivity_rgb.count = 242;
                    break;
                case 4:
                    invalid.reference_illuminant.count = 80;
                    break;
                case 5:
                    invalid.linear_sensitivity_rgb.data = nullptr;
                    break;
                case 6:
                    invalid.linear_sensitivity_rgb.count = std::numeric_limits<std::size_t>::max();
                    break;
                case 7:
                    invalid.reference_illuminant.data = reinterpret_cast<const float*>(1);
                    break;
                case 8:
                    invalid.apply_window = 1;
                    invalid.window_params = {nullptr, 4};
                    break;
                case 9:
                    invalid.apply_window = 1;
                    invalid.window_params = span(input.window);
                    invalid.reconstructed_reference_white = {nullptr, 81};
                    break;
                case 10:
                    invalid.reference_illuminant.data = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3u);
                    break;
            }
            result.hash = 1;
            failure.hash_failure = 1;
            status(fj_legacy_exposure_sensitivity(&invalid, &result, &failure, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
            cleared(result, failure);
            require(!failure.hash_failure && !failure.hash_sample_index, "malformed metadata clears");
            row_event.passed();
        }
        invalid = input.view;
        invalid.window_params = {reinterpret_cast<const float*>(1), std::numeric_limits<std::size_t>::max()};
        invalid.reconstructed_reference_white = invalid.window_params;
        status(fj_legacy_exposure_sensitivity(&invalid, &result, &failure, &error.error), FJ_STATUS_SUCCESS);
        invalid.method = 1;
        invalid.apply_window = 1;
        status(fj_legacy_exposure_sensitivity(&invalid, &result, &failure, &error.error), FJ_STATUS_SUCCESS);
        status(fj_legacy_exposure_sensitivity(nullptr, &result, &failure, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_sensitivity(&input.view, nullptr, &failure, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_sensitivity(&input.view, &result, nullptr, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_sensitivity(&input.view, &result, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        fj_test_exposure_arm_facade_fault();
        status(fj_test_exposure_sensitivity(&input.view, &result, &failure, &error.error), FJ_STATUS_INTERNAL_FAILURE);
        require(fj_test_exposure_facade_fault_consumed() == 1, "independent facade fault witness");
        status(fj_legacy_exposure_sensitivity(&input.view, &result, &failure, &error.error), FJ_STATUS_SUCCESS);
        std::vector<float> spectra(std::size_t{192} * 192u * 81u, 1.0f);
        FjReferenceWhiteInput referenceInput{{spectra.data(), spectra.size()}, {1.0f, 1.0f, 1.0f}, 0.0f};
        FjReferenceWhite white{};
        require(fj_test_exposure_abi_c(&referenceInput, &white, &input.view, &result, &failure, &error.error) == 1, "real C11 export values/prototypes");
        for (float value : white.samples)
            require(value == 1.0f, "C11 reconstructed values");
        JuicerTests::Cohort reference_faults("exposure.boundary.reference_fault", 4);
        for (std::uint32_t fault = 1; fault <= 4; ++fault) {
            auto row_event = reference_faults.case_event(std::to_string(fault - 1));
            status(fj_test_exposure_arm_fault(1, 1, fault), FJ_STATUS_SUCCESS);
            const auto categories = std::array{FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE};
            status(fj_legacy_reconstruction_reference_white(&referenceInput, &white, &error.error), categories[fault - 1]);
            require(fj_test_exposure_fault_consumed(1, fault) == 1, "raw reference fault consumed");
            for (float value : white.samples)
                require(value == 0.0f, "failed reference clears output");
            row_event.passed();
        }
        JuicerTests::Cohort malformed_reference("exposure.boundary.reference_malformed", 5);
        for (std::size_t bad = 0; bad < 5; ++bad) {
            auto row_event = malformed_reference.case_event(std::to_string(bad));
            auto malformed = referenceInput;
            if (bad == 0)
                malformed.spectra.count -= 1;
            if (bad == 1)
                malformed.spectra.data = nullptr;
            if (bad == 2)
                malformed.spectra.data = reinterpret_cast<const float*>(1);
            if (bad == 3)
                malformed.spectra.count = std::numeric_limits<std::size_t>::max();
            if (bad == 4)
                malformed.spectra.data = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3u);
            status(fj_legacy_reconstruction_reference_white(&malformed, &white, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
            row_event.passed();
        }
        fj_test_exposure_arm_facade_fault();
        status(fj_test_reconstruction_reference_white(&referenceInput, &white, &error.error), FJ_STATUS_INTERNAL_FAILURE);
        require(fj_test_exposure_facade_fault_consumed() == 1, "independent reference facade witness");
        status(fj_legacy_reconstruction_reference_white(&referenceInput, &white, &error.error), FJ_STATUS_SUCCESS);
        std::array<std::exception_ptr, 2> threadErrors{};
        std::array<std::thread, 2> threads;
        for (std::size_t t = 0; t < threads.size(); ++t)
            threads[t] = std::thread([&, t] {
                try {
                    Diagnostic localError;
                    FjSensitivity local{};
                    FjSensitivityFailure localFailure{};
                    for (int repeat = 0; repeat < 8; ++repeat) {
                        status(fj_legacy_exposure_sensitivity(&input.view, &local, &localFailure, &localError.error), FJ_STATUS_SUCCESS);
                        require(local.hash == result.hash, "concurrent independent output exact");
                    }
                } catch (...) {
                    threadErrors[t] = std::current_exception();
                }
            });
        for (auto& thread : threads)
            thread.join();
        for (const auto& failure : threadErrors)
            if (failure)
                std::rethrow_exception(failure);
        FjSensitivity retained{};
        {
            SensitivityInput temporary(fixture.at("sensitivity").at(0).at("input"));
            status(fj_legacy_exposure_sensitivity(&temporary.view, &retained, &failure, &error.error), FJ_STATUS_SUCCESS);
        }
        require(retained.hash == result.hash, "completed output survives source lifetime");
        referenceInput.spectral_blur = std::bit_cast<float>(0x5c000000u);
        status(fj_legacy_reconstruction_reference_white(&referenceInput, &white, &error.error), FJ_STATUS_ALLOCATION_FAILURE);
        referenceInput.spectral_blur = std::bit_cast<float>(0x5e000000u);
        status(fj_legacy_reconstruction_reference_white(&referenceInput, &white, &error.error), FJ_STATUS_PREPARATION_FAILURE);
        for (float value : white.samples)
            require(value == 0.0f, "size failure result clears");
        fj_test_exposure_clear_fault();
    }
} // namespace

int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            const std::string selected = argv[1];
            if (selected == "sensitivity" || selected == "reference") {
                std::ifstream listing(argv[2]);
                Json original;
                listing >> original;
                const auto listed = PrintIlluminantFixtures::apply(argv[2], original);
                if (selected == "sensitivity") {
                    for (std::size_t ordinal = 0; ordinal < listed.at("sensitivity").size(); ++ordinal)
                        for (const char* caller : {"legacy", "facade"})
                            JuicerTests::list_case("exposure.sensitivity", std::string(caller) + "/" + fixture_row_id(listed.at("sensitivity")[ordinal], ordinal));
                } else {
                    for (std::size_t ordinal = 0; ordinal < listed.at("reference").size(); ++ordinal)
                        for (const char* caller : {"legacy", "facade"})
                            JuicerTests::list_case("exposure.reference", std::string(caller) + "/" + fixture_row_id(listed.at("reference")[ordinal], ordinal));
                }
            } else if (selected == "products") {
                const auto listed = PrintIlluminantFixtures::apply(argv[2], load_exposure_product_fixture(argv[2]));
                JuicerTests::list_case("exposure_preparation", selected);
                for (const auto& row : listed.at("products")) {
                    const auto id = "polarity/" + std::to_string(row.at("polarity").get<int>()) + "/route/" +
                                    std::to_string(row.at("route").get<int>()) + "/method/" +
                                    std::to_string(row.at("method").get<int>()) + "/label/" +
                                    row.at("label").get<std::string>() + "/variant/" + std::to_string(row.value("variant", 0));
                    JuicerTests::list_case("spectral.product", id);
                }
            } else if (selected == "admission") {
                JuicerTests::list_case("exposure_preparation", selected);
                for (int route = 0; route < 4; ++route)
                    for (int method = 0; method < 3; ++method)
                        for (std::uint32_t operation = 1; operation <= 2; ++operation) {
                            if (operation == 1 && method != 0)
                                continue;
                            for (std::uint32_t fault = 1; fault <= 4; ++fault)
                                JuicerTests::list_case("exposure.admission", "route/" + std::to_string(route) + "/method/" + std::to_string(method) + "/operation/" + std::to_string(operation) + "/fault/" + std::to_string(fault));
                        }
                for (int route = 0; route < 4; ++route)
                    for (int method : {1, 2})
                        JuicerTests::list_case("exposure.admission", "route/" + std::to_string(route) + "/method/" + std::to_string(method) + "/skipped-reference");
            } else {
                JuicerTests::list_case("exposure_preparation", selected);
            }
            return 0;
        }
        require(argc >= 2, "exposure group required");
        const std::string group = argv[1];
        if (group == "admission") {
            JuicerTests::CaseEvent event("exposure_preparation", group);
            require(argc == 3, "exposure admission requires resources");
            fj_test_exposure_admission(argv[2]);
            std::puts("PASS exposure admission contract");
            event.passed();
            case_log.finish();
            return 0;
        }
        require(argc == 3 || argc == 5, "exposure fixture, optional resource/scratch required");
        std::ifstream file(argv[2]);
        const auto fixture = PrintIlluminantFixtures::apply(argv[2], group == "products" ? load_exposure_product_fixture(argv[2]) : Json::parse(file));
        if (group == "reference") {
            require(argc == 5, "reference resources and scratch required");
            reference(fixture, argv[3]);
        } else if (group == "sensitivity")
            sensitivity(fixture);
        else if (group == "boundary") {
            JuicerTests::CaseEvent boundary_event("exposure_preparation", group);
            boundary(fixture);
            boundary_event.passed();
        } else if (group == "products") {
            JuicerTests::CaseEvent products_event("exposure_preparation", group);
            require(argc == 5, "exposure products require resource and scratch");
            fj_test_spectral_products(fixture, {.resources = argv[3], .scratch = argv[4]}, false);
            products_event.passed();
        } else
            throw std::runtime_error("unknown exposure group");
        std::puts("PASS exposure construction/ABI/failure contract");
        int exitCode = 0;
        if (group == "products") {
            const bool ownersDrained = fj_test_spectral_live_owners() == 0 && fj_test_assets_live_owners() == 0;
            exitCode = JuicerTestSupport::test_asset_workspaces.finish(0, ownersDrained);
        }
        case_log.finish();
        return exitCode;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
