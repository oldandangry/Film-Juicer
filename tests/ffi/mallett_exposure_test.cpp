#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "SpectralData.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "juicer_test_api.h"
#include "case_results.h"
#include "exposure_fixture.h"
#include "print_illuminant_fixture.h"
#include "spectral_source_fixture.h"

#define FJ_MALLETT_SIZE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_MALLETT_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "mallett_exposure_abi_facts.inc"
#undef FJ_MALLETT_SIZE
#undef FJ_MALLETT_FIELD
extern "C" int fj_test_mallett_exposure_abi_c(const FjMallettMidgrayInput*, FjMallettMidgray*, float, FjMidgrayNormalization*, float, float*, const FjMallettReferenceInput*, FjReferenceRaw*, FjErrorBuffer*);

extern "C" int fj_test_spectrum_abi_c(const FjHanatosSpectrumInput*, FjSpectrumFixture*, const FjTablesSpectrumInput*, FjSpectrumFixture*, const FjMallettRawInput*, float*, FjErrorBuffer*);

void fj_test_mallett_callers(const std::filesystem::path&, const nlohmann::json&, const std::filesystem::path&, bool, const std::filesystem::path&);

namespace {
    using Json = nlohmann::json;
    void require(bool condition, const std::string& message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void status(FjStatus value, std::uint32_t category) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "fixed exposure status " + std::to_string(value.category) + " expected " + std::to_string(category));
    }
    template <std::size_t N>
    std::array<float, N> decode(const Json& row) {
        require(row.size() == N, "fixed fixture extent");
        std::array<float, N> out{};
        for (std::size_t i = 0; i < N; ++i)
            out[i] = std::bit_cast<float>(row[i].get<std::uint32_t>());
        return out;
    }
    template <std::size_t N>
    void exact(const float* actual, const Json& expected) {
        require(expected.size() == N, "fixed expected extent");
        for (std::size_t i = 0; i < N; ++i) {
            const auto word = expected[i].get<std::uint32_t>();
            require(std::bit_cast<std::uint32_t>(actual[i]) == word || (std::isnan(actual[i]) && std::isnan(std::bit_cast<float>(word))), "fixed exposure bit mismatch at " + std::to_string(i));
        }
    }
    struct Diagnostic {
        std::array<char, 514> bytes{};
        FjErrorBuffer error{bytes.data() + 1, 512, 0};
        Diagnostic() {
            bytes.front() = 'L';
            bytes.back() = 'R';
        }
        void guards() const {
            require(bytes.front() == 'L' && bytes.back() == 'R', "diagnostic canaries");
        }
    };
    struct Operands {
        std::array<std::array<float, 3>, 81> basis{}, sensitivity{};
        std::array<float, 81> illuminant{};
        explicit Operands(const Json& row) {
            for (std::size_t i = 0; i < 81; ++i) {
                basis[i] = decode<3>(row.at("basis")[i]);
                sensitivity[i] = decode<3>(row.at("sensitivity")[i]);
            }
            illuminant = decode<81>(row.at("illuminant"));
        }
        FjFloatSpan basis_span() const {
            return {basis.front().data(), 243};
        }
        FjFloatSpan sensitivity_span() const {
            return {sensitivity.front().data(), 243};
        }
        FjFloatSpan illuminant_span() const {
            return {illuminant.data(), 81};
        }
    };
    struct FocusedInput : Operands {
        FjMallettMidgrayInput input{};
        explicit FocusedInput(const Json& row) : Operands(row.at("input")) {
            const auto& value = row.at("input");
            input.color.input_space = value.at("space").get<std::uint32_t>();
            input.color.decode_cctf = value.at("decode").get<bool>() ? 1u : 0u;
            input.color.adapt_xyz = value.at("adapt").get<bool>() ? 1u : 0u;
            const auto rgb = decode<9>(value.at("rgb_to_xyz")), adapt = decode<9>(value.at("xyz_adapt")), final = decode<9>(value.at("xyz_to_srgb"));
            std::copy(rgb.begin(), rgb.end(), input.color.rgb_to_xyz);
            std::copy(adapt.begin(), adapt.end(), input.color.xyz_adapt);
            std::copy(final.begin(), final.end(), input.xyz_to_linear_srgb);
            input.basis_rgb = basis_span();
            input.illuminant = illuminant_span();
            input.sensitivity_rgb = sensitivity_span();
        }
    };
    struct ReferenceInput : Operands {
        FjMallettReferenceInput input{};
        explicit ReferenceInput(const Json& row) : Operands(row) {
            const auto source = decode<2>(row.at("source_scale"));
            input = {basis_span(), illuminant_span(), sensitivity_span(), source[0], source[1]};
        }
    };
    void check(const FjMallettMidgray& actual, const Json& expected) {
        exact<3>(actual.midgray_dwg_rgb, expected.at("dwg"));
        exact<3>(actual.raw_midgray_bgr, expected.at("raw_bgr"));
        const float normalization[]{actual.raw_green, actual.scale};
        exact<2>(normalization, expected.at("normalization"));
    }
    std::string case_id(const Json& row, std::string_view cohort, std::size_t index) {
        if (!row.contains("id"))
            return "row-" + std::to_string(index);
        const std::string id = row.at("id").is_string() ? row.at("id").get<std::string>() : std::to_string(row.at("id").get<std::int64_t>());
        if (cohort == "focused") {
            const auto& input = row.at("input");
            return id + "/space-" + std::to_string(input.at("space").get<int>()) + "/adapt-" + std::to_string(input.at("adapt").get<bool>()) + "/decode-" + std::to_string(input.at("decode").get<bool>());
        }
        if (cohort == "references")
            return id + "/source-scale-" + std::to_string(row.at("source_scale")[0].get<std::uint32_t>()) + "-" + std::to_string(row.at("source_scale")[1].get<std::uint32_t>());
        return id;
    }
    std::vector<std::string> fixture_case_ids(const Json& fixture, std::string_view group) {
        std::vector<std::string> ids;
        const auto& rows = fixture.at(group);
        ids.reserve(rows.size() * 2);
        for (std::size_t index = 0; index < rows.size(); ++index)
            for (const char* caller : {"legacy", "facade"})
                ids.emplace_back(std::string(caller) + "/" + case_id(rows[index], group, index));
        return ids;
    }
    void numerical(const Json& fixture) {
        JuicerTests::Cohort focused_rows("mallett.focused", fixture_case_ids(fixture, "focused"));
        JuicerTests::Cohort normalization_rows("mallett.normalizations", fixture_case_ids(fixture, "normalizations"));
        JuicerTests::Cohort source_rows("mallett.sources", fixture_case_ids(fixture, "sources"));
        JuicerTests::Cohort reference_rows("mallett.references", fixture_case_ids(fixture, "references"));
        for (std::size_t index = 0; index < fixture.at("focused").size(); ++index) {
            const auto& row = fixture.at("focused")[index];
            const auto id = case_id(row, "focused", index);
            FocusedInput input(row);
            Diagnostic error;
            for (bool facade : {false, true}) {
                auto event = focused_rows.case_event(std::string(facade ? "facade/" : "legacy/") + id);
                FjMallettMidgray result{};
                status((facade ? fj_test_exposure_mallett_midgray : fj_legacy_exposure_mallett_midgray)(&input.input, &result, &error.error), FJ_STATUS_SUCCESS);
                check(result, row.at("expected"));
                error.guards();
                event.passed();
            }
        }
        for (std::size_t index = 0; index < fixture.at("normalizations").size(); ++index) {
            const auto& row = fixture.at("normalizations")[index];
            const auto id = case_id(row, "normalizations", index);
            const float green = std::bit_cast<float>(row.at("green").get<std::uint32_t>());
            const Json expected = Json::array({row.at("expected")[0], row.at("expected")[2]});
            for (bool facade : {false, true}) {
                auto event = normalization_rows.case_event(std::string(facade ? "facade/" : "legacy/") + id);
                FjMidgrayNormalization result{};
                Diagnostic error;
                status((facade ? fj_test_exposure_tc_midgray : fj_legacy_exposure_tc_midgray)(green, &result, &error.error), FJ_STATUS_SUCCESS);
                const float values[]{result.raw_green, result.scale};
                exact<2>(values, expected);
                error.guards();
                event.passed();
            }
        }
        for (std::size_t index = 0; index < fixture.at("sources").size(); ++index) {
            const auto& row = fixture.at("sources")[index];
            const auto id = case_id(row, "sources", index);
            const float ev = std::bit_cast<float>(row.at("ev").get<std::uint32_t>());
            const bool built = row.at("built").get<bool>();
            for (bool facade : {false, true}) {
                auto event = source_rows.case_event(std::string(facade ? "facade/" : "legacy/") + id);
                float result = -1;
                Diagnostic error;
                status((facade ? fj_test_exposure_reference_source : fj_legacy_exposure_reference_source)(ev, &result, &error.error), built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (built)
                    exact<1>(&result, Json::array({row.at("exp2_source")[1]}));
                else
                    require(std::bit_cast<std::uint32_t>(result) == 0, "failed source clear");
                error.guards();
                event.passed();
            }
        }
        for (std::size_t index = 0; index < fixture.at("references").size(); ++index) {
            const auto& row = fixture.at("references")[index];
            const auto id = case_id(row, "references", index);
            ReferenceInput input(row);
            const bool built = row.at("built").get<bool>();
            for (bool facade : {false, true}) {
                auto event = reference_rows.case_event(std::string(facade ? "facade/" : "legacy/") + id);
                FjReferenceRaw result{};
                Diagnostic error;
                status((facade ? fj_test_exposure_mallett_reference_raw : fj_legacy_exposure_mallett_reference_raw)(&input.input, &result, &error.error), built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (built)
                    exact<3>(result.rgb, row.at("expected"));
                else
                    require(std::all_of(std::begin(result.rgb), std::end(result.rgb), [](float v) {
                                return std::bit_cast<std::uint32_t>(v) == 0;
                            }),
                            "failed reference clear");
                error.guards();
                event.passed();
            }
        }
        FocusedInput focused(fixture.at("focused")[0]);
        ReferenceInput reference(fixture.at("references")[0]);
        FjMallettMidgray result{};
        FjMidgrayNormalization normalization{};
        FjReferenceRaw raw{};
        float source = 0;
        Diagnostic error;
        JuicerTests::CaseEvent c_abi_event("mallett_exposure", "c11_abi");
        require(fj_test_mallett_exposure_abi_c(&focused.input, &result, 1.0f, &normalization, 0.0f, &source, &reference.input, &raw, &error.error) == 0, "C11 prototypes/status transport");
        check(result, fixture.at("focused")[0].at("expected"));
        exact<3>(raw.rgb, fixture.at("references")[0].at("expected"));
        require(normalization.raw_green == 1.0f && normalization.scale == 1.0f, "C11 TC values");
        exact<1>(&source, Json::array({fixture.at("sources")[0].at("exp2_source")[1]}));
        c_abi_event.passed();
    }
    void spectrum_exact(const FjSpectrumFixture& actual, const Json& expected) {
        exact<3>(actual.pre_xyz, expected.at("pre_xyz"));
        exact<3>(actual.consumer_white, expected.at("consumer_white"));
        exact<3>(actual.post_adapt_xyz, expected.at("post_adapt_xyz"));
        exact<81>(actual.spectrum, expected.at("spectrum"));
    }
    void spectra(const Json& fixture, const std::filesystem::path& resources, std::string_view fixture_id) {
        const JuicerTestSupport::SpectralSourceAssets assets(resources);
        const JuicerTestSupport::HanatosSource source(assets.get());
        Diagnostic error;
        const auto ax = decode<81>(fixture.at("helper_tables").at("ax")), ay = decode<81>(fixture.at("helper_tables").at("ay")), az = decode<81>(fixture.at("helper_tables").at("az"));
        const auto sInverse = decode<9>(fixture.at("helper_s_inv"));
        std::array<std::array<float, 3>, 81> basis{}, sensitivity{};
        std::array<float, 81> illuminant{};
        for (auto& row : basis)
            row.fill(1);
        for (auto& row : sensitivity)
            row.fill(1);
        illuminant.fill(1);
        const std::array<std::uint32_t, 8> authored{0u, 0x80000000u, 0x3f800000u, 0xc0000000u, 0x7f800000u, 0xff800000u, 0x7fc46000u, 0xffc6a000u};
        for (std::size_t i = 0; i < authored.size(); ++i)
            basis[i / 3][i % 3] = std::bit_cast<float>(authored[i]);
        const FjMallettRawInput mallett{{1, 1, 1}, {basis.front().data(), 243}, {illuminant.data(), 81}, {sensitivity.front().data(), 243}};
        bool checkFaults = true;
        for (const auto& row : fixture.at("helpers")) {
            const std::string row_id = row.at("id").is_string() ? row.at("id").get<std::string>() : std::to_string(row.at("id").get<std::int64_t>());
            JuicerTests::CaseEvent event("mallett.spectra.helper", std::string(fixture_id) + "/" + row_id);
            const auto rgb = decode<3>(row.at("rgb")), white = decode<3>(row.at("reference_white"));
            const FjHanatosSpectrumInput hanatos{{rgb[0], rgb[1], rgb[2]}, {white[0], white[1], white[2]}, source.view().samples};
            FjTablesSpectrumInput tables{{rgb[0], rgb[1], rgb[2]}, {white[0], white[1], white[2]}, {}, {ax.data(), 81}, {ay.data(), 81}, {az.data(), 81}};
            std::copy(sInverse.begin(), sInverse.end(), tables.s_inverse);
            FjSpectrumFixture hanatosOut{}, tablesOut{};
            float bgr[3]{};
            require(fj_test_spectrum_abi_c(&hanatos, &hanatosOut, &tables, &tablesOut, &mallett, bgr, &error.error) == 0, "C11 spectrum prototypes/values");
            spectrum_exact(hanatosOut, row.at("hanatos_expected"));
            spectrum_exact(tablesOut, row.at("tables_expected"));
            require(bgr[0] == 235 && bgr[1] == 235 && bgr[2] == 235, "independent retained BGR leaf");
            event.passed();
            if (!checkFaults)
                continue;
            checkFaults = false;
            for (std::uint32_t operation : {1u, 2u, 3u})
                for (std::uint32_t fault : {1u, 2u}) {
                    JuicerTests::CaseEvent fault_event("mallett.spectra.fault", std::string(fixture_id) + "/" + std::to_string(operation) + "/" + std::to_string(fault));
                    status(fj_test_exposure_fixture_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                    std::memset(&hanatosOut, 0x7f, sizeof(hanatosOut));
                    std::memset(&tablesOut, 0x7f, sizeof(tablesOut));
                    std::fill_n(bgr, 3, 99.0f);
                    const auto actual = operation == 1 ? fj_test_exposure_hanatos_spectrum(&hanatos, &hanatosOut, &error.error) : operation == 2 ? fj_test_exposure_tables_spectrum(&tables, &tablesOut, &error.error)
                                                                                                                                                 : fj_test_exposure_mallett_raw(&mallett, bgr, &error.error);
                    status(actual, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                    require(fj_test_exposure_fixture_fault_consumed(operation, fault) == 1 && fj_test_exposure_fixture_fault_consumed(operation, fault) == 0, "closed fixture consumption before cleanup");
                    const auto* out = operation == 1 ? &hanatosOut : &tablesOut;
                    if (operation == 3)
                        require(bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 0, "leaf failure clear");
                    else {
                        require(std::all_of(std::begin(out->spectrum), std::end(out->spectrum), [](float v) {
                                    return v == 0;
                                }),
                                "spectrum failure clear");
                        require(out->pre_xyz[0] == 0 && out->consumer_white[0] == 0 && out->post_adapt_xyz[0] == 0, "spectrum observations clear");
                    }
                    fj_test_exposure_fixture_clear_fault();
                    fault_event.passed();
                }
        }
    }
    bool cleared(const FjMallettMidgray& result) {
        return std::all_of(std::begin(result.midgray_dwg_rgb), std::end(result.midgray_dwg_rgb), [](float v) {
                   return std::bit_cast<std::uint32_t>(v) == 0;
               }) &&
               std::all_of(std::begin(result.raw_midgray_bgr), std::end(result.raw_midgray_bgr), [](float v) {
                   return std::bit_cast<std::uint32_t>(v) == 0;
               }) &&
               result.raw_green == 0 && result.scale == 0;
    }
    void boundary(const Json& fixture) {
        FocusedInput input(fixture.at("focused")[0]);
        ReferenceInput reference(fixture.at("references")[0]);
        Diagnostic error;
        constexpr std::array<const char*, 12> focused_fields{"input_space", "decode_cctf", "adapt_xyz", "basis_extent", "illuminant_extent", "sensitivity_extent", "basis_null", "illuminant_null", "sensitivity_null", "basis_oversized", "basis_unaligned", "basis_overflow_address"};
        JuicerTests::Cohort focused_defects("mallett.boundary.focused", std::vector<std::string>(focused_fields.begin(), focused_fields.end()));
        for (int bad = 0; bad < 12; ++bad) {
            auto event = focused_defects.case_event(focused_fields[bad]);
            auto view = input.input;
            FjMallettMidgray result{};
            std::memset(&result, 0x7f, sizeof(result));
            if (bad == 0)
                view.color.input_space = 4;
            if (bad == 1)
                view.color.decode_cctf = 2;
            if (bad == 2)
                view.color.adapt_xyz = 2;
            if (bad == 3)
                view.basis_rgb.count = 242;
            if (bad == 4)
                view.illuminant.count = 80;
            if (bad == 5)
                view.sensitivity_rgb.count = 244;
            if (bad == 6)
                view.basis_rgb.data = nullptr;
            if (bad == 7)
                view.illuminant.data = nullptr;
            if (bad == 8)
                view.sensitivity_rgb.data = nullptr;
            if (bad == 9)
                view.basis_rgb.count = std::numeric_limits<std::size_t>::max();
            if (bad == 10)
                view.basis_rgb.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(input.basis.data()) + 1);
            if (bad == 11)
                view.basis_rgb.data = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3u);
            status(fj_legacy_exposure_mallett_midgray(&view, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(cleared(result), "structural failure clears completed fields");
            error.guards();
            event.passed();
        }
        JuicerTests::CaseEvent arguments_event("mallett.boundary.arguments", "null_inputs_and_outputs");
        FjMallettMidgray result{};
        status(fj_legacy_exposure_mallett_midgray(nullptr, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_mallett_midgray(&input.input, nullptr, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        FjErrorBuffer invalid{nullptr, 8, 77};
        std::memset(&result, 0x7f, sizeof(result));
        status(fj_legacy_exposure_mallett_midgray(&input.input, &result, &invalid), FJ_STATUS_UNSUPPORTED_INPUT);
        require(cleared(result), "invalid diagnostic clears output");
        arguments_event.passed();
        JuicerTests::Cohort diagnostic_capacities("mallett.boundary.diagnostic_capacity", {"0", "1", "7", "512"});
        for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{512}}) {
            auto event = diagnostic_capacities.case_event(std::to_string(capacity));
            error.error.capacity = capacity;
            error.error.length = 99;
            status(fj_legacy_exposure_mallett_midgray(nullptr, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(capacity == 0 ? error.error.length == 0 : error.error.length < capacity && error.error.data[error.error.length] == '\0', "bounded diagnostic/terminator");
            error.guards();
            event.passed();
        }
        constexpr std::array<const char*, 4> reference_fields{"basis_extent", "illuminant_null", "sensitivity_oversized", "null_input"};
        JuicerTests::Cohort reference_defects("mallett.boundary.reference", std::vector<std::string>(reference_fields.begin(), reference_fields.end()));
        for (int bad = 0; bad < 4; ++bad) {
            auto event = reference_defects.case_event(reference_fields[bad]);
            auto view = reference.input;
            FjReferenceRaw raw;
            std::memset(&raw, 0x7f, sizeof(raw));
            if (bad == 0)
                view.basis_rgb.count = 242;
            if (bad == 1)
                view.illuminant.data = nullptr;
            if (bad == 2)
                view.sensitivity_rgb.count = std::numeric_limits<std::size_t>::max();
            status(fj_legacy_exposure_mallett_reference_raw(bad == 3 ? nullptr : &view, &raw, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(raw.rgb[0] == 0 && raw.rgb[1] == 0 && raw.rgb[2] == 0, "reference structural clearing");
            event.passed();
        }
        JuicerTests::CaseEvent rejected_arguments("mallett.boundary.rejected_arguments", "raw_tc_and_reference_outputs");
        status(fj_legacy_exposure_mallett_reference_raw(&reference.input, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_tc_midgray(1, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_reference_source(0, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        rejected_arguments.passed();
        error.error.capacity = 512;
        std::vector<std::string> fault_ids;
        for (std::uint32_t operation = 3; operation <= 6; ++operation)
            for (const std::uint32_t fault : {1u, 2u, 4u})
                fault_ids.emplace_back(std::to_string(operation) + "/" + std::to_string(fault));
        JuicerTests::Cohort operation_faults("mallett.boundary.operation_fault", fault_ids);
        for (std::uint32_t operation = 3; operation <= 6; ++operation) {
            status(fj_test_exposure_arm_fault(operation, 1, 3), FJ_STATUS_UNSUPPORTED_INPUT);
            for (std::uint32_t fault : {1u, 2u, 4u}) {
                auto event = operation_faults.case_event(std::to_string(operation) + "/" + std::to_string(fault));
                status(fj_test_exposure_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                FjMidgrayNormalization normalized{};
                float source = -1;
                FjReferenceRaw raw{};
                std::memset(&result, 0x7f, sizeof(result));
                std::memset(&normalized, 0x7f, sizeof(normalized));
                std::memset(&raw, 0x7f, sizeof(raw));
                const auto observed = operation == 3 ? fj_legacy_exposure_mallett_midgray(&input.input, &result, &error.error) : operation == 4 ? fj_legacy_exposure_tc_midgray(1, &normalized, &error.error)
                                                                                                                             : operation == 5   ? fj_legacy_exposure_reference_source(0, &source, &error.error)
                                                                                                                                                : fj_legacy_exposure_mallett_reference_raw(&reference.input, &raw, &error.error);
                status(observed, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                       : FJ_STATUS_PREPARATION_FAILURE);
                require(fj_test_exposure_fault_consumed(operation, fault) == 1 && fj_test_exposure_fault_consumed(operation, fault) == 0, "same-thread consume-once before cleanup");
                require(operation != 3 || cleared(result), "panic/fault focused clear");
                require(operation != 4 || (normalized.raw_green == 0 && normalized.scale == 0), "TC fault clear");
                require(operation != 5 || source == 0, "source fault clear");
                require(operation != 6 || (raw.rgb[0] == 0 && raw.rgb[1] == 0 && raw.rgb[2] == 0), "reference fault clear");
                fj_test_exposure_clear_fault();
                event.passed();
            }
        }
        JuicerTests::CaseEvent facade_event("mallett.boundary.facade_fault", "midgray");
        fj_test_exposure_arm_facade_fault();
        status(fj_test_exposure_mallett_midgray(&input.input, &result, &error.error), FJ_STATUS_INTERNAL_FAILURE);
        require(fj_test_exposure_facade_fault_consumed() == 1 && cleared(result), "direct-core facade witness");
        facade_event.passed();
        JuicerTests::CaseEvent skip_event("mallett.boundary.malformed_skips_math", "midgray");
        status(fj_test_exposure_arm_fault(3, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_legacy_exposure_mallett_midgray(nullptr, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_exposure_fault_consumed(3, 1) == 0, "malformed input skips mathematics");
        status(fj_legacy_exposure_mallett_midgray(&input.input, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_exposure_fault_consumed(3, 1) == 1, "later valid call consumes armed fault");
        skip_event.passed();
        fj_test_exposure_clear_fault();
        status(fj_test_exposure_arm_fault(3, 1, 1), FJ_STATUS_SUCCESS);
        FjStatus other{};
        std::thread worker([&] {
            FjMallettMidgray out{};
            FjErrorBuffer silent{nullptr, 0, 0};
            other = fj_legacy_exposure_mallett_midgray(&input.input, &out, &silent);
        });
        worker.join();
        status(other, FJ_STATUS_SUCCESS);
        status(fj_legacy_exposure_mallett_midgray(&input.input, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_exposure_fault_consumed(3, 1) == 1, "thread-local fault witness");
        fj_test_exposure_clear_fault();
        FjMallettMidgray retained{};
        {
            FocusedInput temporary(fixture.at("focused")[0]);
            status(fj_legacy_exposure_mallett_midgray(&temporary.input, &retained, &error.error), FJ_STATUS_SUCCESS);
        }
        check(retained, fixture.at("focused")[0].at("expected"));
    }
} // namespace
int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 3 && std::string_view(argv[argc - 1]) == "--list-cases") {
            const std::string selected = argv[3];
            std::ifstream manifestFile(argv[1]);
            const auto manifest = Json::parse(manifestFile);
            if (selected == "numerical") {
                const auto path = std::filesystem::path(argv[1]).parent_path() / manifest.at("presets").at(argv[2]).at("numerical").get<std::string>();
                std::ifstream values(path);
                const auto fixture = Json::parse(values);
                for (const auto& [group, caller] : {std::pair{"focused", "mallett.focused"}, std::pair{"normalizations", "mallett.normalizations"}, std::pair{"sources", "mallett.sources"}, std::pair{"references", "mallett.references"}})
                    for (const auto& id : fixture_case_ids(fixture, group))
                        JuicerTests::list_case(caller, id);
                JuicerTests::list_case("mallett_exposure", "c11_abi");
            } else if (selected == "products" || selected == "admission") {
                const auto path = std::filesystem::path(argv[1]).parent_path() / manifest.at("presets").at(argv[2]).at("products").get<std::string>();
                std::ifstream values(path);
                const auto products = Json::parse(values);
                if (selected == "products") {
                    for (const auto& row : products.at("products"))
                        JuicerTests::list_case("mallett.product", std::to_string(row.at("polarity").get<int>()) + "/" + std::to_string(row.at("route").get<int>()) + "/" + std::to_string(row.at("method").get<int>()) + "/" + std::to_string(row.at("variant").get<int>()));
                    for (int route = 0; route < 4; ++route) {
                        for (const char* source : {"missing", "rows_80x3", "columns_81x2", "short_payload"})
                            JuicerTests::list_case("mallett.constructor.source", std::to_string(route) + "/" + source);
                        for (int operation : {3, 5, 6})
                            for (int fault : {1, 2, 4})
                                JuicerTests::list_case("mallett.constructor.exposure_fault", std::to_string(route) + "/" + std::to_string(operation) + "/" + std::to_string(fault));
                        for (int precedence = 0; precedence < 3; ++precedence)
                            JuicerTests::list_case("mallett.constructor.precedence", std::to_string(route) + "/" + std::to_string(precedence));
                    }
                    for (int method : {0, 2})
                        JuicerTests::list_case("mallett.constructor.tc_normalization", std::to_string(method));
                    JuicerTests::list_case("mallett.caller", "products_owner_close");
                } else {
                    for (int polarity = 0; polarity < 2; ++polarity)
                        for (int route = 0; route < 2; ++route) {
                            for (const char* source : {"missing", "rows_80x3", "columns_81x2", "short_payload"})
                                JuicerTests::list_case("mallett.admission.source", std::to_string(polarity) + "/" + std::to_string(route) + "/" + source);
                            for (int operation : {3, 5, 6})
                                for (int fault : {1, 2, 4})
                                    JuicerTests::list_case("mallett.admission.exposure_fault", std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(operation) + "/" + std::to_string(fault));
                            for (int source_failure = 0; source_failure < 2; ++source_failure)
                                for (int mode = 0; mode < 3; ++mode)
                                    JuicerTests::list_case("mallett.admission.supersession", std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(source_failure) + "/" + std::to_string(mode));
                            for (int earlier = 0; earlier < 3; ++earlier)
                                JuicerTests::list_case("mallett.admission.precedence", std::to_string(polarity) + "/" + std::to_string(route) + "/" + std::to_string(earlier));
                        }
                    for (int method : {0, 2})
                        JuicerTests::list_case("mallett.admission.tc_normalization", std::to_string(method));
                    JuicerTests::list_case("mallett.admission.skipped_tc_normalization", "mallett");
                    JuicerTests::list_case("mallett.caller", "admission_owner_close");
                }
            } else if (selected == "spectra") {
                for (int index : {5, 6}) {
                    const std::string fixture_id = index == 5 ? "cat02" : "input-color";
                    const auto captured = load_shared_fixture(argv[index], index == 5 ? "cat02-manifest.json" : "input-color-manifest.json");
                    for (const auto& row : captured.at("helpers")) {
                        const std::string row_id = row.at("id").is_string() ? row.at("id").get<std::string>() : std::to_string(row.at("id").get<std::int64_t>());
                        std::string case_id = fixture_id;
                        case_id += "/";
                        case_id += row_id;
                        JuicerTests::list_case("mallett.spectra.helper", case_id);
                    }
                    for (int operation : {1, 2, 3})
                        for (int fault : {1, 2})
                            JuicerTests::list_case("mallett.spectra.fault", fixture_id + "/" + std::to_string(operation) + "/" + std::to_string(fault));
                }
            } else if (selected == "boundary") {
                for (const char* id : {"input_space", "decode_cctf", "adapt_xyz", "basis_extent", "illuminant_extent", "sensitivity_extent", "basis_null", "illuminant_null", "sensitivity_null", "basis_oversized", "basis_unaligned", "basis_overflow_address"})
                    JuicerTests::list_case("mallett.boundary.focused", id);
                JuicerTests::list_case("mallett.boundary.arguments", "null_inputs_and_outputs");
                for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{512}})
                    JuicerTests::list_case("mallett.boundary.diagnostic_capacity", std::to_string(capacity));
                for (const char* id : {"basis_extent", "illuminant_null", "sensitivity_oversized", "null_input"})
                    JuicerTests::list_case("mallett.boundary.reference", id);
                JuicerTests::list_case("mallett.boundary.rejected_arguments", "raw_tc_and_reference_outputs");
                for (std::uint32_t operation = 3; operation <= 6; ++operation)
                    for (std::uint32_t fault : {1u, 2u, 4u})
                        JuicerTests::list_case("mallett.boundary.operation_fault", std::to_string(operation) + "/" + std::to_string(fault));
                JuicerTests::list_case("mallett.boundary.facade_fault", "midgray");
                JuicerTests::list_case("mallett.boundary.malformed_skips_math", "midgray");
            } else {
                JuicerTests::list_case("mallett_exposure", selected);
            }
            return 0;
        }
        require(argc == 4 || argc == 5 || argc == 6 || argc == 7, "manifest preset numerical|boundary|products|admission|spectra [resources artifacts fixtures]");
        const std::filesystem::path manifestPath(argv[1]);
        std::ifstream stream(manifestPath);
        const auto manifest = Json::parse(stream);
        std::ifstream values(manifestPath.parent_path() / manifest.at("presets").at(argv[2]).at("numerical").get<std::string>());
        const auto fixture = Json::parse(values);
        if ((std::string(argv[3]) == "products" || std::string(argv[3]) == "admission") && argc == 6) {
            std::ifstream productFile(manifestPath.parent_path() / manifest.at("presets").at(argv[2]).at("products").get<std::string>());
            const auto products = PrintIlluminantFixtures::apply(manifestPath.parent_path() / manifest.at("presets").at(argv[2]).at("products").get<std::string>(), Json::parse(productFile));
            fj_test_mallett_callers(argv[4], products, manifestPath.parent_path() / "mallett-failure-contract.json", std::string(argv[3]) == "admission", argv[5]);
        } else if (std::string(argv[3]) == "numerical")
            numerical(fixture);
        else if (std::string(argv[3]) == "spectra" && argc == 7) {
            for (int i : {5, 6}) {
                const auto captured = load_shared_fixture(argv[i], i == 5 ? "cat02-manifest.json" : "input-color-manifest.json");
                spectra(captured, argv[4], i == 5 ? "cat02" : "input-color");
            }
        } else if (std::string(argv[3]) == "boundary") {
            boundary(fixture);
        } else
            throw std::runtime_error("unknown fixed exposure group");
        std::puts("Mallett exposure qualification PASS");
        case_log.finish();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    } catch (const JuicerCuda::ExecutionFailure& e) {
        std::fprintf(stderr, "%s\n", e.failure.diagnostic.c_str());
        return 1;
    }
}
