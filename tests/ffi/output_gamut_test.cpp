#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "GamutCompression.h"
#include "RustGamutBridge.h"
#include "juicer_test_api.h"

#define FJ_GAMUT_SIZE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_GAMUT_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "output_gamut_abi_facts.inc"
#undef FJ_GAMUT_SIZE
#undef FJ_GAMUT_FIELD

extern "C" FjStatus fj_test_output_transform_abi_c(const FjOutputGamutSource*, FjOutputGamutTransform*, FjErrorBuffer*);
extern "C" FjStatus fj_test_output_boundary_abi_c(const FjOutputBoundarySource*, float*, size_t, FjOutputBoundaryIdentity*, FjOutputBoundaryFailure*, FjErrorBuffer*);
extern "C" FjStatus fj_test_output_transform_status_c(const FjOutputGamutSource*, FjOutputGamutTransform*, unsigned, size_t*);
extern "C" FjStatus fj_test_output_boundary_status_c(const FjOutputBoundarySource*, float*, size_t, FjOutputBoundaryIdentity*, FjOutputBoundaryFailure*, unsigned, size_t*);

namespace {
    using Json = nlohmann::json;
    void require(bool condition, const std::string& message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void status(FjStatus value, std::uint32_t category) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "complete status category/API/code");
    }
    template <typename Range>
    Json bits(const Range& values) {
        Json out = Json::array();
        for (float value : values)
            out.push_back(std::bit_cast<std::uint32_t>(value));
        return out;
    }
    template <std::size_t N>
    void decode(float (&out)[N], const Json& source) {
        require(source.size() == N, "fixed fixture shape");
        for (std::size_t i = 0; i < N; ++i)
            out[i] = std::bit_cast<float>(source[i].get<std::uint32_t>());
    }
    FjOutputGamutSource source(const Json& row) {
        FjOutputGamutSource out{};
        out.output_space = row.value("tag", 0u);
        decode(out.rgb_to_xyz, row["source"]["rgb_to_xyz"]);
        decode(out.xyz_to_rgb, row["source"]["xyz_to_rgb"]);
        decode(out.white_xyz, row["source"]["white"]);
        decode(out.d65_white_xyz, row["source"]["d65"]);
        return out;
    }
    FjOutputBoundarySource boundary_source(const Json& row) {
        FjOutputBoundarySource out{};
        out.output_space = row.value("tag", 0u);
        decode(out.d65_xyz_to_native_rgb, row["transform"]["exit"]);
        out.transform_hash = row["transform"]["hash"].get<std::uint64_t>();
        return out;
    }
    template <typename T>
    struct Guarded {
        std::uint64_t before = 123;
        T value{};
        std::uint64_t after = 456;
        void check() const {
            require(before == 123 && after == 456, "fixed output canaries");
        }
    };
    struct FailureRequirement {
        unsigned value = 0;
    };
    struct Invocation {
        unsigned operation;
        unsigned facade;
    };
    struct Result {
        Guarded<FjOutputGamutTransform> transform;
        Guarded<FjOutputBoundaryIdentity> identity;
        Guarded<FjOutputBoundaryFailure> failure;
        std::vector<float> table = std::vector<float>(Gamut::kOutputBoundaryValueCount + 2, 7.0f);
        Result() {
            std::fill_n(transform.value.native_rgb_to_d65_xyz, 9, 7.0f);
            std::fill_n(transform.value.d65_xyz_to_native_rgb, 9, 7.0f);
            transform.value.output_space = 99;
            transform.value.hash = transform.value.contract_hash = 99;
            identity.value = {99, 99, 99};
            failure.value = {99, 99, 99};
            table.front() = 123.0f;
            table.back() = 456.0f;
        }
        float* cmax() {
            return table.data() + 1;
        }
        Json fields(unsigned operation) const {
            if (operation == 1)
                return {{"entry", bits(transform.value.native_rgb_to_d65_xyz)}, {"exit", bits(transform.value.d65_xyz_to_native_rgb)}, {"tag", transform.value.output_space}, {"hash", transform.value.hash}, {"contract", transform.value.contract_hash}};
            return {{"transform", identity.value.transform_hash}, {"contract", identity.value.contract_hash}, {"hash", identity.value.hash}, {"failure", {failure.value.requirement, failure.value.lightness_index, failure.value.hue_index}}, {"cmax", bits(std::span(table.data() + 1, Gamut::kOutputBoundaryValueCount))}};
        }
        void check() const {
            transform.check();
            identity.check();
            failure.check();
            require(table.front() == 123.0f && table.back() == 456.0f, "Cmax canaries");
        }
        void cleared(unsigned operation, FailureRequirement expected = {}) const {
            check();
            if (operation == 1) {
                require(transform.value.hash == 0 && transform.value.contract_hash == 0 && transform.value.output_space == 0, "cleared transform identities/tag");
                for (float value : transform.value.native_rgb_to_d65_xyz)
                    require(std::bit_cast<std::uint32_t>(value) == 0, "cleared entry");
                for (float value : transform.value.d65_xyz_to_native_rgb)
                    require(std::bit_cast<std::uint32_t>(value) == 0, "cleared exit");
            } else {
                require(identity.value.transform_hash == 0 && identity.value.contract_hash == 0 && identity.value.hash == 0, "cleared table identities");
                require(failure.value.requirement == expected.value, "typed failure requirement");
                if (expected.value != 2)
                    require(failure.value.lightness_index == 0 && failure.value.hue_index == 0, "non-grid metadata cleared");
                for (std::size_t i = 1; i + 1 < table.size(); ++i)
                    require(std::bit_cast<std::uint32_t>(table[i]) == 0, "cleared exact Cmax");
            }
        }
    };
    struct Diagnostic {
        std::array<char, 514> bytes{};
        FjErrorBuffer record{};
        explicit Diagnostic(std::size_t capacity) {
            bytes.fill(73);
            record = {bytes.data() + 1, capacity, 99};
            if (capacity == 0)
                record.data = reinterpret_cast<char*>(std::uintptr_t{1});
        }
        void check() const {
            require(bytes.front() == 73 && bytes[record.capacity + 1] == 73, "diagnostic canaries");
            if (record.capacity == 0)
                require(record.length == 0 && bytes[1] == 73, "zero capacity status only");
            else
                require(record.length < record.capacity && bytes[record.length + 1] == 0, "bounded text length/NUL");
        }
    };
    struct FaultScope {
        ~FaultScope() {
            fj_test_output_gamut_clear_fault();
        }
    };
    FjStatus call(Invocation invocation, const FjOutputGamutSource* input, const FjOutputBoundarySource* boundary, Result& out, FjErrorBuffer* error) {
        if (invocation.operation == 1)
            return (invocation.facade ? fj_test_gamut_output_transform : fj_test_output_transform_abi_c)(input, &out.transform.value, error);
        return (invocation.facade ? fj_test_gamut_output_boundary : fj_test_output_boundary_abi_c)(boundary, out.cmax(), Gamut::kOutputBoundaryValueCount, &out.identity.value, &out.failure.value, error);
    }
    unsigned category(unsigned fault) {
        return fault == 0 ? FJ_STATUS_SUCCESS : fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT
                                            : fault == 2   ? FJ_STATUS_INTERNAL_FAILURE
                                            : fault == 3   ? FJ_STATUS_ALLOCATION_FAILURE
                                                           : FJ_STATUS_PREPARATION_FAILURE;
    }
    unsigned requirement(unsigned operation, unsigned fault) {
        return operation == 2 && fault >= 4 ? fault - 3 : 0;
    }
    void diagnostics(const Json& fixture) {
        const auto input = source(fixture["rows"][0]);
        const auto boundary = boundary_source(fixture["rows"][0]);
        for (unsigned operation : {1u, 2u})
            for (unsigned facade : {0u, 1u})
                for (unsigned fault = 0; fault <= (operation == 1 ? 5u : 7u); ++fault) {
                    FaultScope scope;
                    const auto invoke = [&](Result& out, FjErrorBuffer* error) {
                        if (fault)
                            status(fj_test_output_gamut_arm_fault(operation, facade, fault), FJ_STATUS_SUCCESS);
                        const auto result = call({operation, facade}, &input, &boundary, out, error);
                        status(result, category(fault));
                        if (fault) {
                            require(fj_test_output_gamut_fault_consumed(operation, facade, fault) == 1 && fj_test_output_gamut_fault_consumed(operation, facade, fault) == 0, "consume once before cleanup");
                            out.cleared(operation, {requirement(operation, fault)});
                        }
                        out.check();
                    };
                    Result expected;
                    Diagnostic full(512);
                    invoke(expected, &full.record);
                    full.check();
                    const auto fields = expected.fields(operation);
                    Result null;
                    invoke(null, nullptr);
                    require(null.fields(operation) == fields, "D1 null diagnostic exact status/defined fields");
                    for (std::size_t capacity : {0u, 1u, 7u, 512u}) {
                        Result actual;
                        Diagnostic text(capacity);
                        invoke(actual, &text.record);
                        text.check();
                        require(actual.fields(operation) == fields, "D2/D3 text-independent full result/metadata");
                        if (capacity > 0 && fault) {
                            const std::string complete(full.bytes.data() + 1, full.record.length);
                            require(std::string(text.bytes.data() + 1, text.record.length) == complete.substr(0, capacity - 1), "D3 exact bounded prefix");
                        }
                    }
                    if (facade == 0)
                        for (unsigned zero : {0u, 1u}) {
                            Result actual;
                            std::size_t length = 99;
                            if (fault)
                                status(fj_test_output_gamut_arm_fault(operation, 0, fault), FJ_STATUS_SUCCESS);
                            const auto result = operation == 1 ? fj_test_output_transform_status_c(&input, &actual.transform.value, zero, &length) : fj_test_output_boundary_status_c(&boundary, actual.cmax(), Gamut::kOutputBoundaryValueCount, &actual.identity.value, &actual.failure.value, zero, &length);
                            status(result, category(fault));
                            require(length == 0 && actual.fields(operation) == fields, "actual C11 distinct null/zero-capacity forms");
                            if (fault)
                                require(fj_test_output_gamut_fault_consumed(operation, 0, fault) == 1, "C11 fault consumed");
                        }
                }
        for (unsigned operation : {1u, 2u})
            for (unsigned facade : {0u, 1u})
                for (unsigned malformed = 0; malformed < 5; ++malformed) {
                    FaultScope scope;
                    Result out;
                    Diagnostic valid(32);
                    FjErrorBuffer* error = &valid.record;
                    if (malformed == 0)
                        error = reinterpret_cast<FjErrorBuffer*>(std::uintptr_t{1});
                    if (malformed == 1)
                        error = reinterpret_cast<FjErrorBuffer*>(std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{7});
                    if (malformed == 2)
                        valid.record.data = nullptr;
                    if (malformed == 3)
                        valid.record.capacity = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) + 1;
                    if (malformed == 4)
                        valid.record.data = reinterpret_cast<char*>(std::numeric_limits<std::uintptr_t>::max() - 15);
                    const auto before = fj_test_output_gamut_visits(operation);
                    status(fj_test_output_gamut_arm_fault(operation, facade, 2), FJ_STATUS_SUCCESS);
                    status(call({operation, facade}, nullptr, nullptr, out, error), FJ_STATUS_UNSUPPORTED_INPUT);
                    out.cleared(operation);
                    require(fj_test_output_gamut_visits(operation) == before, "D4 no source borrow/construction/fault visit");
                    require(std::all_of(valid.bytes.begin(), valid.bytes.end(), [](char v) {
                                return v == 73;
                            }),
                            "D4 backing untouched");
                    if (malformed >= 2)
                        require(valid.record.length == 0, "D4 independently valid diagnostic length cleared");
                    Diagnostic later(32);
                    status(call({operation, facade}, &input, &boundary, out, &later.record), FJ_STATUS_INTERNAL_FAILURE);
                    require(fj_test_output_gamut_fault_consumed(operation, facade, 2) == 1, "D4 armed fault survives malformed entry");
                    out.cleared(operation);
                }
        for (unsigned operation : {1u, 2u})
            for (unsigned facade : {0u, 1u})
                for (int diagnostic = 0; diagnostic < 3; ++diagnostic)
                    for (int invalid = 0; invalid < (operation == 1 ? 5 : 10); ++invalid) {
                        FaultScope scope;
                        Result out;
                        Diagnostic text(diagnostic == 1 ? 0 : 32);
                        auto src = input;
                        auto bound = boundary;
                        const FjOutputGamutSource* sourcePointer = &src;
                        const FjOutputBoundarySource* boundaryPointer = &bound;
                        auto* transformPointer = &out.transform.value;
                        auto* identityPointer = &out.identity.value;
                        auto* failurePointer = &out.failure.value;
                        auto* cmax = out.cmax();
                        auto count = Gamut::kOutputBoundaryValueCount;
                        if (invalid == 0) {
                            sourcePointer = nullptr;
                            boundaryPointer = nullptr;
                        }
                        if (invalid == 1) {
                            src.output_space = bound.output_space = 9;
                        }
                        if (invalid == 2) {
                            sourcePointer = reinterpret_cast<const FjOutputGamutSource*>(std::uintptr_t{1});
                            boundaryPointer = reinterpret_cast<const FjOutputBoundarySource*>(std::uintptr_t{1});
                        }
                        if (invalid == 3) {
                            transformPointer = nullptr;
                            identityPointer = nullptr;
                        }
                        if (invalid == 4) {
                            transformPointer = reinterpret_cast<FjOutputGamutTransform*>(std::uintptr_t{1});
                            failurePointer = nullptr;
                        }
                        if (invalid == 5)
                            count = 0;
                        if (invalid == 6)
                            --count;
                        if (invalid == 7)
                            count = std::numeric_limits<std::size_t>::max();
                        if (invalid == 8)
                            cmax = nullptr;
                        if (invalid == 9)
                            cmax = reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{3});
                        const auto before = fj_test_output_gamut_visits(operation);
                        status(fj_test_output_gamut_arm_fault(operation, facade, 2), FJ_STATUS_SUCCESS);
                        FjErrorBuffer* error = diagnostic == 0 ? nullptr : &text.record;
                        const auto result = operation == 1 ? (facade ? fj_test_gamut_output_transform : fj_test_output_transform_abi_c)(sourcePointer, transformPointer, error) : (facade ? fj_test_gamut_output_boundary : fj_test_output_boundary_abi_c)(boundaryPointer, cmax, count, identityPointer, failurePointer, error);
                        status(result, FJ_STATUS_UNSUPPORTED_INPUT);
                        out.check();
                        require(fj_test_output_gamut_visits(operation) == before, "D5 malformed inputs before faults");
                        if (operation == 1 && invalid < 3)
                            out.cleared(1);
                        if (operation == 2) {
                            if (invalid != 3)
                                require(out.identity.value.hash == 0 && out.identity.value.transform_hash == 0 && out.identity.value.contract_hash == 0, "D5 independently valid identities clear");
                            if (invalid != 4)
                                require(out.failure.value.requirement == 0 && out.failure.value.lightness_index == 0 && out.failure.value.hue_index == 0, "D5 independently valid metadata clear");
                            for (std::size_t i = 1; i + 1 < out.table.size(); ++i)
                                require(out.table[i] == (invalid >= 5 ? 7.0f : 0.0f), "D5 exact/invalid extent clearing and no arbitrary write");
                        }
                        Result later;
                        status(call({operation, facade}, &input, &boundary, later, error), FJ_STATUS_INTERNAL_FAILURE);
                        require(fj_test_output_gamut_fault_consumed(operation, facade, 2) == 1, "D5 fault remains armed for a valid call");
                    }
        std::puts("A9 D1-D5 both exports/facades and actual C11 null/zero-capacity forms PASS");
    }
    void boundary(const Json& fixture, const std::filesystem::path& tablePath) {
        std::ifstream data(tablePath, std::ios::binary);
        for (const auto& row : fixture["rows"]) {
            const auto input = source(row);
            for (unsigned facade : {0u, 1u}) {
                Result out;
                Diagnostic text(512);
                const bool expected = row["transform_ok"];
                status(call({1, facade}, &input, nullptr, out, &text.record), expected ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                text.check();
                out.check();
                if (!expected) {
                    out.cleared(1);
                    continue;
                }
                require(bits(out.transform.value.native_rgb_to_d65_xyz) == row["transform"]["entry"] && bits(out.transform.value.d65_xyz_to_native_rgb) == row["transform"]["exit"], "C11/facade exact transform words");
                require(out.transform.value.hash == row["transform"]["hash"].get<std::uint64_t>() && out.transform.value.contract_hash == row["transform"]["contract_hash"].get<std::uint64_t>() && out.transform.value.output_space == input.output_space, "C11/facade exact transform identities/tag");
                const auto bound = boundary_source(row);
                const bool built = row["boundary_ok"];
                status(call({2, facade}, nullptr, &bound, out, &text.record), built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                text.check();
                if (!built) {
                    out.cleared(2, {2});
                    continue;
                }
                require(out.identity.value.hash == row["boundary_hash"].get<std::uint64_t>() && out.identity.value.transform_hash == out.transform.value.hash && out.identity.value.contract_hash == out.transform.value.contract_hash, "C11/facade exact table identities");
                std::vector<std::uint32_t> words(Gamut::kOutputBoundaryValueCount);
                data.clear();
                data.seekg(row["table_offset"].get<std::streamoff>());
                data.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
                require(bool(data), "complete frozen table read");
                for (std::size_t i = 0; i < words.size(); ++i)
                    require(std::bit_cast<std::uint32_t>(out.cmax()[i]) == words[i], "C11/facade every Cmax word");
            }
        }
        diagnostics(fixture);
    }
    void binding(const Json& fixture) {
        for (const auto& row : fixture["rows"])
            if (row["variant"] == 0) {
                Gamut::OutputGamutTransform transform;
                std::string diagnostic;
                require(JuicerGamut::output_transform(static_cast<OutputEncoding::ColorSpace>(row["tag"].get<int>()), transform, diagnostic), diagnostic);
                require(transform.hash == row["transform"]["hash"].get<std::uint64_t>() && transform.contractHash == row["transform"]["contract_hash"].get<std::uint64_t>() && bits(transform.nativeRgbToD65Xyz) == row["transform"]["entry"] && bits(transform.d65XyzToNativeRgb) == row["transform"]["exit"], "native binding exact completed transform");
                auto table = std::make_unique<Gamut::OutputBoundaryTable>();
                require(JuicerGamut::output_boundary(transform, *table), table->diagnostic);
                require(table->hash == row["boundary_hash"].get<std::uint64_t>(), "native binding complete table identity");
            }
        for (unsigned operation : {1u, 2u})
            for (unsigned fault : {2u, 3u, 4u, 5u}) {
                FaultScope scope;
                Gamut::OutputGamutTransform transform;
                std::string diagnostic;
                require(JuicerGamut::output_transform(OutputEncoding::ColorSpace::sRGB, transform, diagnostic), diagnostic);
                auto table = std::make_unique<Gamut::OutputBoundaryTable>();
                status(fj_test_output_gamut_arm_fault(operation, 0, fault), FJ_STATUS_SUCCESS);
                bool ordinary = false, caught = false;
                try {
                    ordinary = !(operation == 1 ? JuicerGamut::output_transform(OutputEncoding::ColorSpace::sRGB, transform, diagnostic) : JuicerGamut::output_boundary(transform, *table));
                } catch (const std::bad_alloc&) {
                    require(fault == 3, "native allocation mapping");
                    caught = true;
                } catch (const JuicerCuda::ExecutionFailure& e) {
                    status(e.failure.status, FJ_STATUS_INTERNAL_FAILURE);
                    require(fault == 2, "native exceptional mapping");
                    caught = true;
                }
                require((fault >= 4 && ordinary) || (fault < 4 && caught), "typed native ordinary/exceptional separation");
                require(fj_test_output_gamut_fault_consumed(operation, 0, fault) == 1, "native fault consumed once");
            }
        std::puts("A9 native fixed carrier/math/failure binding PASS");
    }
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 4, "case, fixture manifest, preset");
        std::ifstream input(argv[2]);
        Json manifest;
        input >> manifest;
        const auto member = manifest["preset_membership"].at(argv[3]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        std::ifstream records(directory / member["records"].get<std::string>());
        Json fixture;
        records >> fixture;
        const std::string selected = argv[1];
        if (selected == "boundary")
            boundary(fixture, directory / member["tables"].get<std::string>());
        else if (selected == "binding")
            binding(fixture);
        else
            throw std::runtime_error("unknown output gamut case");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
