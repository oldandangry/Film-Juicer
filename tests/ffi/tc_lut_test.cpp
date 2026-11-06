#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "ProcessRuntime.h"
#include "juicer_test_api.h"
#include "case_results.h"
#include "print_illuminant_fixture.h"
#include "spectral_source_fixture.h"

#define FJ_TC_SIZE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_TC_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "tc_lut_abi_facts.inc"
#undef FJ_TC_SIZE
#undef FJ_TC_FIELD

extern "C" int fj_test_tc_lut_abi_c(void);
void fj_test_tc_lut_products(const FjAssets*, const nlohmann::json&);
void fj_test_tc_lut_admission(const FjAssets*);

namespace {
    using Json = nlohmann::json;
    void require(bool value, const std::string& why) {
        if (!value)
            throw std::runtime_error(why);
    }
    float word(const Json& value) {
        return std::bit_cast<float>(value.get<std::uint32_t>());
    }
    template <std::size_t N>
    std::array<float, N> values(const Json& data) {
        std::array<float, N> out{};
        require(data.size() == N, "fixture extent");
        for (std::size_t i = 0; i < N; ++i)
            out[i] = word(data.at(i));
        return out;
    }
    struct Owned {
        FjOwnedFilmTcLut record{};
        Owned() = default;
        Owned(const Owned&) = delete;
        Owned& operator=(const Owned&) = delete;
        ~Owned() noexcept {
            (void)fj_legacy_reconstruction_release_tc_lut(&record);
        }
    };
    struct Diagnostic {
        std::array<char, 513> bytes{};
        FjErrorBuffer buffer{bytes.data(), bytes.size() - 1, 0};
        Diagnostic() {
            bytes.back() = 67;
        }
        std::string text() const {
            return {bytes.data(), buffer.length};
        }
        void check() const {
            require(buffer.length < buffer.capacity && bytes[buffer.length] == 0 && bytes.back() == 67, "bounded diagnostic/canary");
        }
    };
    std::uint64_t digest(FjFloatSpan span) {
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        for (std::size_t i = 0; i < span.count; ++i) {
            auto bits = std::bit_cast<std::uint32_t>(span.data[i]);
            for (int byte = 0; byte < 4; ++byte) {
                hash ^= (bits >> (byte * 8)) & 255u;
                hash *= 0x100000001b3ULL;
            }
        }
        return hash;
    }
    void numerical(JuicerProcess::ProcessRuntime& runtime, const Json& fixture, const std::filesystem::path& tablesPath) {
        const JuicerTestSupport::HanatosSource hanatos(runtime.borrow().assets);
        const JuicerTestSupport::ArcticSource arctic(runtime.borrow().assets);
        std::vector<float> procedural(std::size_t{192} * 192u * 81u);
        for (std::size_t i = 0; i < procedural.size(); ++i)
            procedural[i] = std::bit_cast<float>(0x3e000123u + static_cast<std::uint32_t>(i % 4093u) * 173u);
        std::array<float, 2050> hull{};
        for (std::size_t i = 0; i < 1025; ++i) {
            hull[i * 2] = word(fixture["leaves"]["hull"]["xy"][i][0]);
            hull[i * 2 + 1] = word(fixture["leaves"]["hull"]["xy"][i][1]);
        }
        const auto center = values<2>(fixture["leaves"]["hull"]["center"]);
        std::ifstream tables(tablesPath, std::ios::binary);
        require(bool(tables), "frozen complete tables");
        for (const auto& row : fixture["leaves"]["cases"]) {
            const auto& input = row["input"];
            const int id = input["id"];
            if (id == 16 || id == 17)
                continue; // Native method/key guard, not the raw schema.
            if (id == 21)
                procedural[0] = std::numeric_limits<float>::infinity();
            if (id == 22)
                procedural[0] = std::numeric_limits<float>::quiet_NaN();
            const FjFloatSpan source = id == 7 ? hanatos.view().samples : id == 8 ? arctic.view().samples
                                                                                  : FjFloatSpan{procedural.data(), procedural.size()};
            std::array<float, 243> sensitivity{};
            std::array<float, 45> surface{};
            for (std::size_t i = 0; i < 81; ++i)
                for (std::size_t c = 0; c < 3; ++c)
                    sensitivity[i * 3 + c] = word(input["sensitivity"][i][c]);
            for (std::size_t c = 0; c < 3; ++c)
                for (std::size_t i = 0; i < 15; ++i)
                    surface[c * 15 + i] = word(input["surface_params"][c][i]);
            const auto spd = values<81>(input["spd"]);
            const auto white = values<3>(input["white"]);
            const FjFilmTcLutInput request{source, {sensitivity.data(), sensitivity.size()}, {spd.data(), spd.size()}, {white[0], white[1], white[2]}, word(input["blur"]), input["method"].get<std::uint32_t>(), input["surface"].get<bool>() ? 1u : 0u, {surface.data(), surface.size()}, input["compression"].get<bool>() ? 1u : 0u, input["hull_available"].get<bool>() ? 1u : 0u, {center[0], center[1]}, {hull.data(), hull.size()}};
            for (int facade = 0; facade < 2; ++facade) {
                JuicerTests::CaseEvent event("tc_lut.raw", std::string(facade ? "facade/" : "legacy/") + std::to_string(id));
                Owned owned;
                Diagnostic error;
                const auto status = facade ? fj_test_reconstruction_tc_lut(&request, &owned.record, &error.buffer) : fj_legacy_reconstruction_tc_lut(&request, &owned.record, &error.buffer);
                error.check();
                require(status.api == FJ_API_NONE && status.native_code == 0, "TC status fields");
                require((status.category == FJ_STATUS_SUCCESS) == row["built"].get<bool>(), "frozen build status " + std::to_string(id));
                if (status.category != FJ_STATUS_SUCCESS) {
                    require(status.category == FJ_STATUS_PREPARATION_FAILURE && error.text() == row["diagnostic"].get<std::string>() && !owned.record.samples.data && !owned.record.samples.count && !owned.record.capacity, "no partial owner/diagnostic");
                    event.passed();
                    continue;
                }
                require(owned.record.samples.count == 147456 && owned.record.capacity >= owned.record.samples.count, "actual length/capacity");
                require(digest(owned.record.samples) == row["table"]["native_byte_digest_fnv1a64"], "complete native table digest " + std::to_string(id));
                if (id < 9) {
                    tables.seekg(row["table"]["byte_offset"].get<std::streamoff>());
                    std::vector<float> expected(147456);
                    tables.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(expected.size() * sizeof(float)));
                    require(bool(tables), "full table bytes");
                    for (std::size_t i = 0; i < expected.size(); ++i)
                        require(std::bit_cast<std::uint32_t>(expected[i]) == std::bit_cast<std::uint32_t>(owned.record.samples.data[i]), "exact full RGB/padding word");
                }
                for (const auto& sample : row["samples"]) {
                    const auto xyz = values<3>(sample["xyz"]);
                    std::array<float, 4> rgb{9, 9, 9, 77};
                    const auto sampled = facade ? fj_test_reconstruction_sample_tc_lut(owned.record.samples, xyz.data(), rgb.data(), &error.buffer) : fj_legacy_reconstruction_sample_tc_lut(owned.record.samples, xyz.data(), rgb.data(), &error.buffer);
                    require(rgb[3] == 77, "sample output canary");
                    error.check();
                    if (sample["classification"] == "native_defined") {
                        require(sampled.category == FJ_STATUS_SUCCESS, "defined sample success");
                        for (std::size_t c = 0; c < 3; ++c)
                            require(std::bit_cast<std::uint32_t>(rgb[c]) == sample["expected"][c].get<std::uint32_t>(), "exact Mitchell result");
                    } else {
                        require(sampled.category == FJ_STATUS_PREPARATION_FAILURE && rgb[0] == 0 && rgb[1] == 0 && rgb[2] == 0, "checked unusable coordinate failure");
                    }
                }
                event.passed();
            }
            procedural[0] = std::bit_cast<float>(0x3e000123u);
        }
    }
    void boundary() {
        require(fj_test_tc_lut_abi_c() == 1, "C11 real values, source expiry and consume-once release");
        std::vector<float> source(std::size_t{192} * 192u * 81u, 1.0f);
        std::array<float, 243> sensitivity{};
        sensitivity.fill(1);
        std::array<float, 81> spd{};
        spd.fill(1);
        FjFilmTcLutInput input{{source.data(), source.size()}, {sensitivity.data(), sensitivity.size()}, {spd.data(), spd.size()}, {1, 1, 1}, 0, 2, 0, {nullptr, 0}, 0, 0, {0, 0}, {nullptr, 0}};
        const std::array<float, 3> xyz{1, 0, 0};
        Diagnostic error;
        JuicerTests::Cohort malformed_inputs("tc_lut.boundary.malformed_input", 6);
        for (int invalid = 0; invalid < 6; ++invalid) {
            auto row_event = malformed_inputs.case_event(std::to_string(invalid));
            auto malformed = input;
            if (invalid == 0)
                --malformed.spectra.count;
            if (invalid == 1)
                malformed.spectra.data = nullptr;
            if (invalid == 2)
                malformed.method = 1;
            if (invalid == 3)
                malformed.apply_surface = 2;
            if (invalid == 4)
                malformed.sensitivity_rgb.count = std::numeric_limits<std::size_t>::max();
            if (invalid == 5)
                malformed.reference_illuminant.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(spd.data()) + 1);
            Owned owned;
            require(fj_legacy_reconstruction_tc_lut(&malformed, &owned.record, &error.buffer).category == FJ_STATUS_UNSUPPORTED_INPUT && !owned.record.samples.data, "malformed consumed span/tag");
            row_event.passed();
        }
        // Inactive spans are ignored, even if they cannot be borrowed.
        input.surface_rgb = {nullptr, std::numeric_limits<std::size_t>::max()};
        input.hull_xy = {nullptr, std::numeric_limits<std::size_t>::max()};
        Owned owned;
        require(fj_legacy_reconstruction_tc_lut(&input, &owned.record, &error.buffer).category == FJ_STATUS_SUCCESS, "inactive spans ignored");
        std::array<float, 3> rgb{};
        require(fj_legacy_reconstruction_sample_tc_lut({owned.record.samples.data, 147455}, xyz.data(), rgb.data(), &error.buffer).category == FJ_STATUS_UNSUPPORTED_INPUT, "wrong sample extent");
        const std::array<std::uint32_t, 4> categories{FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE};
        std::vector<std::string> fault_ids;
        for (std::uint32_t operation = 1; operation <= 2; ++operation)
            for (std::uint32_t fault = 1; fault <= 4; ++fault)
                fault_ids.emplace_back("operation-" + std::to_string(operation) + "/fault-" + std::to_string(fault));
        JuicerTests::Cohort boundary_faults("tc_lut.boundary.operation_fault", fault_ids);
        for (std::uint32_t operation = 1; operation <= 2; ++operation)
            for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                auto row_event = boundary_faults.case_event("operation-" + std::to_string(operation) + "/fault-" + std::to_string(fault));
                require(fj_test_tc_lut_arm_fault(operation, 1, fault).category == FJ_STATUS_SUCCESS, "arm closed raw fault");
                Owned failed;
                auto status = operation == 1 ? fj_legacy_reconstruction_tc_lut(&input, &failed.record, &error.buffer) : fj_legacy_reconstruction_sample_tc_lut(owned.record.samples, xyz.data(), rgb.data(), &error.buffer);
                require(fj_test_tc_lut_fault_consumed(operation, fault) == 1, "same-thread consumption before cleanup");
                require(status.category == categories[fault - 1] && !failed.record.samples.data, "fault category and empty result");
                error.check();
                fj_test_tc_lut_clear_fault();
                row_event.passed();
            }
        fj_test_tc_lut_arm_facade_fault();
        Owned failed;
        require(fj_test_reconstruction_tc_lut(&input, &failed.record, &error.buffer).category == FJ_STATUS_INTERNAL_FAILURE && fj_test_tc_lut_facade_fault_consumed() == 1, "separate direct-core facade witness");
        require(fj_test_tc_lut_fault_consumed(1, 2) == 0, "facade never calls raw legacy export");
        FjErrorBuffer noText{nullptr, 0, 99};
        auto malformed = input;
        malformed.method = 99;
        require(fj_legacy_reconstruction_tc_lut(&malformed, &failed.record, &noText).category == FJ_STATUS_UNSUPPORTED_INPUT && noText.length == 0, "zero-capacity status");
        std::array<char, 3> tiny{67, 67, 67};
        FjErrorBuffer small{tiny.data(), 2, 0};
        require(fj_legacy_reconstruction_tc_lut(&malformed, &failed.record, &small).category == FJ_STATUS_UNSUPPORTED_INPUT && small.length == 1 && tiny[1] == 0 && tiny[2] == 67, "diagnostic truncation");
        // Transfer the record, never copy it into a second live owner.
        FjOwnedFilmTcLut transferred = owned.record;
        owned.record = {};
        source.clear();
        source.shrink_to_fit();
        std::thread finalRelease([record = transferred]() mutable {
            require(fj_legacy_reconstruction_release_tc_lut(&record).category == FJ_STATUS_SUCCESS && !record.samples.data && record.capacity == 0, "cross-thread final release");
        });
        transferred = {};
        finalRelease.join();
    }
} // namespace

int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            const std::string selected = argv[1];
            if (selected == "numerical") {
                std::ifstream manifestFile(argv[3]);
                const auto manifest = Json::parse(manifestFile);
                const auto& member = manifest["preset_membership"][argv[4]];
                std::ifstream rows(std::filesystem::path(argv[3]).parent_path() / member["records"].get<std::string>());
                const auto fixture = PrintIlluminantFixtures::apply(std::filesystem::path(argv[3]).parent_path() / member["records"].get<std::string>(), Json::parse(rows));
                for (const auto& row : fixture["leaves"]["cases"]) {
                    const int id = row["input"]["id"];
                    if (id == 16 || id == 17)
                        continue;
                    for (const char* caller : {"legacy", "facade"})
                        JuicerTests::list_case("tc_lut.raw", std::string(caller) + "/" + std::to_string(id));
                }
                JuicerTests::list_case("tc_lut", "owner_close");
            } else if (selected == "products") {
                const std::filesystem::path manifestPath(argv[3]);
                std::ifstream manifestFile(manifestPath);
                const auto manifest = Json::parse(manifestFile);
                const auto& member = manifest["preset_membership"][argv[4]];
                const auto recordsPath = manifestPath.parent_path() / member["records"].get<std::string>();
                std::ifstream records(recordsPath);
                const auto fixture = PrintIlluminantFixtures::apply(recordsPath, Json::parse(records));
                for (const auto& row : fixture["products"]) {
                    const auto id = "polarity/" + std::to_string(row["polarity"].get<int>()) + "/route/" +
                                    std::to_string(row["route"].get<int>()) + "/method/" +
                                    std::to_string(row["method"].get<int>()) + "/variant/" +
                                    std::to_string(row["variant"].get<int>());
                    JuicerTests::list_case("tc_lut.product", id);
                }
                JuicerTests::list_case("tc_lut", "products");
                JuicerTests::list_case("tc_lut", "owner_close");
            } else if (selected == "admission") {
                JuicerTests::list_case("tc_lut", "admission");
                for (int route = 0; route < 4; ++route)
                    for (int method : {0, 2}) {
                        for (std::uint32_t operation = 1; operation <= 2; ++operation)
                            for (std::uint32_t index = 1; index <= (operation == 1 ? 1u : 3u); ++index)
                                for (std::uint32_t fault = 1; fault <= 4; ++fault)
                                    for (bool current : {false, true})
                                        JuicerTests::list_case("tc_lut.admission", "route/" + std::to_string(route) + "/method/" + std::to_string(method) + "/operation/" + std::to_string(operation) + "/index/" + std::to_string(index) + "/fault/" + std::to_string(fault) + (current ? "/current" : "/cold"));
                        for (std::uint32_t operation = 1; operation <= 2; ++operation)
                            JuicerTests::list_case("tc_lut.admission", "route/" + std::to_string(route) + "/method/" + std::to_string(method) + "/skipped-mallett/operation/" + std::to_string(operation));
                    }
                JuicerTests::list_case("tc_lut", "owner_close");
            } else {
                JuicerTests::list_case("tc_lut", selected);
            }
            return 0;
        }
        require(argc == 5, "mode, resources, manifest and preset required");
        if (std::string(argv[1]) == "boundary") {
            JuicerTests::CaseEvent event("tc_lut", "boundary");
            boundary();
            event.passed();
        } else {
            JuicerProcess::ProcessRuntime runtime;
            runtime.create(argv[2]);
            const std::filesystem::path manifestPath(argv[3]);
            std::ifstream manifestFile(manifestPath);
            const auto manifest = Json::parse(manifestFile);
            const auto& membership = manifest["preset_membership"][argv[4]];
            std::ifstream input(manifestPath.parent_path() / membership["records"].get<std::string>());
            const auto fixture = PrintIlluminantFixtures::apply(manifestPath.parent_path() / membership["records"].get<std::string>(), Json::parse(input));
            if (std::string(argv[1]) == "products") {
                JuicerTests::CaseEvent event("tc_lut", "products");
                fj_test_tc_lut_products(runtime.borrow().assets, fixture);
                event.passed();
            } else if (std::string(argv[1]) == "admission") {
                JuicerTests::CaseEvent event("tc_lut", "admission");
                fj_test_tc_lut_admission(runtime.borrow().assets);
                event.passed();
            } else
                numerical(runtime, fixture, manifestPath.parent_path() / membership["tables"].get<std::string>());
            JuicerTests::CaseEvent close_event("tc_lut", "owner_close");
            require(runtime.close().category == FJ_STATUS_SUCCESS, "close Runtime");
            close_event.passed();
        }
        case_log.finish();
        std::puts("TC numerical/ownership boundary PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
