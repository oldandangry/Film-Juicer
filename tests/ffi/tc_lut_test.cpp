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

#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "SpectralData.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"

#define FJ_TC_SIZE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_TC_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "tc_lut_abi_facts.inc"
#undef FJ_TC_SIZE
#undef FJ_TC_FIELD

extern "C" int fj_test_tc_lut_abi_c(void);
void fj_test_tc_lut_products(const nlohmann::json&);
void fj_test_tc_lut_admission();

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
    void numerical(const Json& fixture, const std::filesystem::path& tablesPath) {
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
            const auto& source = id == 7 ? Spectral::gHanSpectra.data : id == 8 ? Spectral::context().arcticSpectra.data
                                                                                : procedural;
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
            const FjFilmTcLutInput request{{source.data(), source.size()}, {sensitivity.data(), sensitivity.size()}, {spd.data(), spd.size()}, {white[0], white[1], white[2]}, word(input["blur"]), input["method"].get<std::uint32_t>(), input["surface"].get<bool>() ? 1u : 0u, {surface.data(), surface.size()}, input["compression"].get<bool>() ? 1u : 0u, input["hull_available"].get<bool>() ? 1u : 0u, {center[0], center[1]}, {hull.data(), hull.size()}};
            for (int facade = 0; facade < 2; ++facade) {
                Owned owned;
                Diagnostic error;
                const auto status = facade ? fj_test_reconstruction_tc_lut(&request, &owned.record, &error.buffer) : fj_legacy_reconstruction_tc_lut(&request, &owned.record, &error.buffer);
                error.check();
                require(status.api == FJ_API_NONE && status.native_code == 0, "TC status fields");
                require((status.category == FJ_STATUS_SUCCESS) == row["built"].get<bool>(), "frozen build status " + std::to_string(id));
                if (status.category != FJ_STATUS_SUCCESS) {
                    require(status.category == FJ_STATUS_PREPARATION_FAILURE && error.text() == row["diagnostic"].get<std::string>() && !owned.record.samples.data && !owned.record.samples.count && !owned.record.capacity, "no partial owner/diagnostic");
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
        for (int invalid = 0; invalid < 6; ++invalid) {
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
        }
        // Inactive spans are ignored, even if they cannot be borrowed.
        input.surface_rgb = {nullptr, std::numeric_limits<std::size_t>::max()};
        input.hull_xy = {nullptr, std::numeric_limits<std::size_t>::max()};
        Owned owned;
        require(fj_legacy_reconstruction_tc_lut(&input, &owned.record, &error.buffer).category == FJ_STATUS_SUCCESS, "inactive spans ignored");
        std::array<float, 3> rgb{};
        require(fj_legacy_reconstruction_sample_tc_lut({owned.record.samples.data, 147455}, xyz.data(), rgb.data(), &error.buffer).category == FJ_STATUS_UNSUPPORTED_INPUT, "wrong sample extent");
        const std::array<std::uint32_t, 4> categories{FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE};
        for (std::uint32_t operation = 1; operation <= 2; ++operation)
            for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                require(fj_test_tc_lut_arm_fault(operation, 1, fault).category == FJ_STATUS_SUCCESS, "arm closed raw fault");
                Owned failed;
                auto status = operation == 1 ? fj_legacy_reconstruction_tc_lut(&input, &failed.record, &error.buffer) : fj_legacy_reconstruction_sample_tc_lut(owned.record.samples, xyz.data(), rgb.data(), &error.buffer);
                require(fj_test_tc_lut_fault_consumed(operation, fault) == 1, "same-thread consumption before cleanup");
                require(status.category == categories[fault - 1] && !failed.record.samples.data, "fault category and empty result");
                error.check();
                fj_test_tc_lut_clear_fault();
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
        require(argc == 5, "mode, resources, manifest and preset required");
        if (std::string(argv[1]) == "boundary") {
            boundary();
        } else {
            JuicerCuda::Owner owner;
            owner.create(argv[2]);
            JuicerProcess::root().ensure_bootstrap();
            const std::filesystem::path manifestPath(argv[3]);
            std::ifstream manifestFile(manifestPath);
            const auto manifest = Json::parse(manifestFile);
            const auto& membership = manifest["preset_membership"][argv[4]];
            std::ifstream input(manifestPath.parent_path() / membership["records"].get<std::string>());
            const auto fixture = Json::parse(input);
            if (std::string(argv[1]) == "products")
                fj_test_tc_lut_products(fixture);
            else if (std::string(argv[1]) == "admission")
                fj_test_tc_lut_admission();
            else
                numerical(fixture, manifestPath.parent_path() / membership["tables"].get<std::string>());
            require(owner.close().category == FJ_STATUS_SUCCESS, "close host owner");
        }
        std::puts("TC numerical/ownership boundary PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
