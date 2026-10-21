#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "GamutCompression.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RenderRecipe.h"
#include "RustGamutBridge.h"
#include "SpectralData.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "print_illuminant_fixture.h"
#include "spectral_allocation_probe.h"

#define FJ_HULL_SIZE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_HULL_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "input_hull_abi_facts.inc"
#undef FJ_HULL_SIZE
#undef FJ_HULL_FIELD

extern "C" FjStatus fj_test_input_hull_abi_c(const FjInputHullSource*, FjInputHull*, FjErrorBuffer*);
void fj_test_input_hull_products(const nlohmann::json&, const std::filesystem::path&);
void fj_test_input_hull_admission(std::uint32_t fault);

namespace JuicerGamut::Test {
    std::barrier<>* start = nullptr;
    std::atomic<std::size_t> entered{0};
    std::array<Gamut::InputCompressionHull*, 4> candidates{};
    void before_build(Gamut::InputCompressionHull& out) {
        if (start) {
            candidates.at(entered.fetch_add(1)) = &out;
            start->arrive_and_wait();
        }
    }
} // namespace JuicerGamut::Test

namespace {
    using Json = nlohmann::json;
    void require(bool ok, const std::string& message) {
        if (!ok)
            throw std::runtime_error(message);
    }
    void status(FjStatus value, std::uint32_t category) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "complete status category/API/code");
    }
    std::vector<float> values(const Json& data) {
        std::vector<float> out;
        for (const auto& value : data)
            out.push_back(std::bit_cast<float>(value.get<std::uint32_t>()));
        return out;
    }
    template <typename Range>
    Json bits(const Range& data) {
        Json out = Json::array();
        for (float v : data)
            out.push_back(std::bit_cast<std::uint32_t>(v));
        return out;
    }
    void completed(const FjInputHull& hull, const Json& expected) {
        require(bits(hull.center_xy) == expected["center"] && hull.hash == expected["hash"], "center/hash exact");
        for (std::size_t i = 0; i < 1025; ++i)
            require(bits(std::array{hull.xy[i * 2], hull.xy[i * 2 + 1]}) == expected["xy"][i], "complete polygon words");
    }
    struct Diagnostic {
        std::array<char, 513> data{};
        FjErrorBuffer record{data.data(), 512, 0};
        Diagnostic() {
            data.back() = 73;
        }
        void check() const {
            require(record.length < record.capacity && data[record.length] == 0 && data.back() == 73, "diagnostic terminator/canary");
        }
        std::string text() const {
            return {data.data(), record.length};
        }
    };
    struct Output {
        std::uint64_t before = 123;
        FjInputHull value{};
        std::uint64_t after = 456;
        void poison() {
            std::fill_n(value.center_xy, 2, -1.f);
            std::fill_n(value.xy, 2050, -1.f);
            value.hash = 17;
        }
        void check(bool cleared) const {
            require(before == 123 && after == 456, "output canaries");
            if (cleared) {
                require(value.hash == 0, "failure hash cleared");
                for (float v : value.center_xy)
                    require(std::bit_cast<std::uint32_t>(v) == 0, "center cleared");
                for (float v : value.xy)
                    require(std::bit_cast<std::uint32_t>(v) == 0, "polygon cleared");
            }
        }
    };
    struct Source {
        std::array<std::vector<float>, 4> data;
        FjInputHullSource record() const {
            return {{data[0].data(), data[0].size()}, {data[1].data(), data[1].size()}, {data[2].data(), data[2].size()}, {data[3].data(), data[3].size()}};
        }
        explicit Source(const Json& row) {
            for (std::size_t i = 0; i < 4; ++i)
                data[i] = values(row["sources"][i]["values"]);
        }
    };
    void boundary(const Json& fixture) {
        for (const auto& row : fixture["math"]) {
            Source source(row);
            const auto request = source.record();
            for (auto call : {fj_test_input_hull_abi_c, fj_test_gamut_input_hull}) {
                Output out;
                out.poison();
                Diagnostic diagnostic;
                const auto result = call(&request, &out.value, &diagnostic.record);
                const int id = row["id"];
                const bool built = row["built"].get<bool>() || id == 5;
                status(result, built ? FJ_STATUS_SUCCESS : (id == 4 || id == 6 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_PREPARATION_FAILURE));
                out.check(!built);
                diagnostic.check();
                if (built)
                    completed(out.value, id == 5 ? fixture["math"][0]["expected"] : row["expected"]);
                else if (result.category == FJ_STATUS_PREPARATION_FAILURE)
                    require(diagnostic.text() == row["expected"]["diagnostic"].get<std::string>(), "typed producer diagnostic");
            }
        }
        Source source(fixture["math"][0]);
        const auto request = source.record();
        for (std::uint32_t facade = 0; facade <= 1; ++facade) {
            const auto call = facade ? fj_test_gamut_input_hull : fj_test_input_hull_abi_c;
            for (std::uint32_t fault = 1; fault <= 6; ++fault) {
                status(fj_test_gamut_arm_fault(facade, fault), FJ_STATUS_SUCCESS);
                Output out;
                out.poison();
                Diagnostic diagnostic;
                status(call(&request, &out.value, &diagnostic.record), fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                                                             : FJ_STATUS_PREPARATION_FAILURE);
                require(fj_test_gamut_fault_consumed(facade, fault) == 1 && fj_test_gamut_fault_consumed(facade, fault) == 0, "fault consumed exactly once before cleanup");
                out.check(true);
                diagnostic.check();
                status(call(&request, &out.value, &diagnostic.record), FJ_STATUS_SUCCESS);
                completed(out.value, fixture["math"][0]["expected"]);
            }
        }
        for (int condition = 0; condition < 13; ++condition) {
            Output out;
            out.poison();
            Diagnostic diagnostic;
            auto input = request;
            const FjInputHullSource* record = &input;
            FjInputHull* output = &out.value;
            FjErrorBuffer* error = &diagnostic.record;
            switch (condition) {
                case 0:
                    record = nullptr;
                    break;
                case 1:
                    record = reinterpret_cast<const FjInputHullSource*>(std::uintptr_t{1});
                    break;
                case 2:
                    input.x.count = 80;
                    break;
                case 3:
                    input.y.count = 82;
                    break;
                case 4:
                    input.z.count = std::numeric_limits<std::size_t>::max();
                    break;
                case 5:
                    input.d65.data = nullptr;
                    break;
                case 6:
                    input.x.data = reinterpret_cast<const float*>(std::uintptr_t{1});
                    break;
                case 7:
                    input.x.data = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3);
                    break;
                case 8:
                    output = nullptr;
                    break;
                case 9:
                    output = reinterpret_cast<FjInputHull*>(std::uintptr_t{1});
                    break;
                case 10:
                    error = nullptr;
                    break;
                case 11:
                    diagnostic.record.data = nullptr;
                    break;
                case 12:
                    diagnostic.record.data = reinterpret_cast<char*>(std::numeric_limits<std::uintptr_t>::max() - 3);
                    break;
            }
            status(fj_test_input_hull_abi_c(record, output, error), FJ_STATUS_UNSUPPORTED_INPUT);
            out.check(condition != 8 && condition != 9);
        }
        Output out;
        FjErrorBuffer statusOnly{nullptr, 0, 99};
        status(fj_test_input_hull_abi_c(&request, &out.value, &statusOnly), FJ_STATUS_SUCCESS);
        require(statusOnly.length == 0, "status-only length");
        completed(out.value, fixture["math"][0]["expected"]);
        std::array<char, 3> tiny{'a', 'b', 'c'};
        FjErrorBuffer truncated{tiny.data(), 2, 99};
        status(fj_test_input_hull_abi_c(nullptr, &out.value, &truncated), FJ_STATUS_UNSUPPORTED_INPUT);
        require(truncated.length == 1 && tiny[1] == 0 && tiny[2] == 'c', "truncated diagnostic canary");
        auto alias = request;
        alias.y = alias.x;
        alias.z = alias.x;
        alias.d65 = alias.x;
        status(fj_test_input_hull_abi_c(&alias, &out.value, &statusOnly), FJ_STATUS_PREPARATION_FAILURE);
        fj_test_gamut_clear_fault();
    }
    void binding(const Json& fixture) {
        Gamut::InputCompressionHull retained;
        for (const auto& row : fixture["math"]) {
            std::array<Spectral::Curve, 4> source;
            for (std::size_t i = 0; i < 4; ++i) {
                source[i].linear = values(row["sources"][i]["values"]);
                source[i].lambda_nm = values(row["sources"][i]["axis"]);
            }
            const bool built = JuicerGamut::input_hull(source[0], source[1], source[2], source[3], retained);
            require(built == row["built"].get<bool>() && retained.valid == built && retained.hash == row["expected"]["hash"] && retained.diagnostic == row["expected"]["diagnostic"].get<std::string>(), "native binding disposition");
            require(bits(retained.center) == row["expected"]["center"], "native binding center");
            for (std::size_t i = 0; i < 1025; ++i)
                require(bits(retained.xy[i]) == row["expected"]["xy"][i], "native binding xy");
        }
        Source source(fixture["math"][0]);
        const auto request = source.record();
        for (std::uint32_t fault : {1u, 2u, 3u, 4u, 5u, 6u}) {
            std::array<Spectral::Curve, 4> curves;
            for (std::size_t i = 0; i < 4; ++i) {
                curves[i].linear = source.data[i];
                curves[i].lambda_nm.resize(81);
            }
            status(fj_test_gamut_arm_fault(0, fault), FJ_STATUS_SUCCESS);
            bool exceptional = false;
            bool built = false;
            try {
                built = JuicerGamut::input_hull(curves[0], curves[1], curves[2], curves[3], retained);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                exceptional = true;
                status(error.failure.status, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
            }
            require(!built && exceptional == (fault <= 2) && !retained.valid && retained.hash == 0, "native ordinary/exceptional mapping");
            require(fj_test_gamut_fault_consumed(0, fault) == 1, "native fault before cleanup");
        }
        FjInputHull owned{};
        Diagnostic diagnostic;
        status(fj_test_input_hull_abi_c(&request, &owned, &diagnostic.record), FJ_STATUS_SUCCESS);
        for (auto& values : source.data) {
            values.clear();
            values.shrink_to_fit();
        }
        std::thread([owned, expected = fixture["math"][0]["expected"]] {
            completed(owned, expected);
        }).join();
    }
    void cache(const std::filesystem::path& resources, const Json& fixture) {
        {
            JuicerAssets::Library library(resources);
            const auto curves = library.illuminant_filter_curves();
            require(bool(curves), "prewarm source before native candidate allocation witness");
            status(fj_test_gamut_arm_fault(0, 1), FJ_STATUS_SUCCESS);
            SpectralAllocationProbe::arm();
            bool memory = false;
            try {
                (void)library.input_compression_hull();
            } catch (const std::bad_alloc&) {
                memory = true;
            }
            const auto bytes = SpectralAllocationProbe::bytes();
            SpectralAllocationProbe::clear();
            require(memory && bytes >= sizeof(Gamut::InputCompressionHull), "real native shared candidate allocation retains bad_alloc");
            require(fj_test_gamut_fault_consumed(0, 1) == 0, "allocation precedes Rust computation");
            bool exceptional = false;
            try {
                (void)library.input_compression_hull();
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                exceptional = true;
                status(failure.failure.status, FJ_STATUS_UNSUPPORTED_INPUT);
            }
            require(exceptional && fj_test_gamut_fault_consumed(0, 1) == 1, "next cold candidate consumes pending fault before cleanup");
            fj_test_gamut_clear_fault();
            require(bool(library.input_compression_hull()), "allocation/exceptional failure permits retry");
        }
        {
            JuicerAssets::Library retry(resources);
            status(fj_test_gamut_arm_fault(0, 4), FJ_STATUS_SUCCESS);
            require(!retry.input_compression_hull(), "ordinary cold failure publishes no hull");
            require(fj_test_gamut_fault_consumed(0, 4) == 1, "cold Library consumes fault before retry");
            const auto recovered = retry.input_compression_hull();
            require(recovered && recovered->hash == fixture["math"][0]["expected"]["hash"], "cold retry constructs complete hull");
        }
        std::shared_ptr<const Gamut::InputCompressionHull> retained;
        FilmRawRecipe recipe;
        {
            JuicerAssets::Library library(resources);
            std::array<std::shared_ptr<const Gamut::InputCompressionHull>, 4> results;
            std::barrier start(4);
            JuicerGamut::Test::start = &start;
            JuicerGamut::Test::entered = 0;
            std::array<std::thread, 4> workers;
            for (std::size_t i = 0; i < workers.size(); ++i)
                workers[i] = std::thread([&, i] {
                    results[i] = library.input_compression_hull();
                });
            for (auto& worker : workers)
                worker.join();
            JuicerGamut::Test::start = nullptr;
            require(JuicerGamut::Test::entered == 4, "four cold candidates rendezvous before computation outside cache mutex");
            for (std::size_t i = 0; i < 4; ++i) {
                require(results[i] && results[i] == results[0], "one published concurrent winner");
                for (std::size_t j = 0; j < i; ++j)
                    require(JuicerGamut::Test::candidates[i] != JuicerGamut::Test::candidates[j], "distinct concurrent native candidates");
            }
            retained = results[0];
            recipe.inputCompressionHull = retained;
            for (auto& result : results)
                result.reset();
            require(retained.use_count() == 3, "only Library, external and recipe holds survive publication");
            std::weak_ptr source = library.illuminant_filter_curves();
            status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
            require(source.expired(), "source curves expire independently of hull");
            status(fj_test_gamut_arm_fault(0, 1), FJ_STATUS_SUCCESS);
            const float old = Spectral::gXBar.linear[80];
            Spectral::gXBar.linear[80] += .125f;
            require(library.input_compression_hull() == retained && fj_test_gamut_fault_consumed(0, 1) == 0, "warm acquisition ignores changed source and bypasses computation");
            const auto curves = library.illuminant_filter_curves();
            FjInputHullSource request{{Spectral::gXBar.linear.data(), 81}, {Spectral::gYBar.linear.data(), 81}, {Spectral::gZBar.linear.data(), 81}, {curves->d65.linear.data(), 81}};
            FjInputHull output{};
            FjErrorBuffer error{nullptr, 0, 0};
            status(fj_test_input_hull_abi_c(&request, &output, &error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(fj_test_gamut_fault_consumed(0, 1) == 1, "warm fault consumed by next actual call before cleanup");
            JuicerAssets::Library changed(resources);
            const auto rebuilt = changed.input_compression_hull();
            Spectral::gXBar.linear[80] = old;
            require(rebuilt && rebuilt->hash == fixture["math"][1]["expected"]["hash"] && rebuilt->hash != retained->hash, "new cold Library observes changed tail source");
            require(library.input_compression_hull() == retained, "warm pointer/hash retained after source/cache release");
        }
        require(retained.use_count() == 2 && recipe.inputCompressionHull == retained && retained->valid, "Library expiry releases only its own hold");
        recipe.inputCompressionHull.reset();
        require(retained.use_count() == 1, "external final hold");
        std::weak_ptr weak = retained;
        std::thread([hold = std::move(retained)] {
            require(hold->hash == 0x5cda57333bb9b653ULL, "retained hull thread transfer");
        }).join();
        require(weak.expired(), "last external hold releases completed value");
        fj_test_gamut_clear_fault();
    }
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 5, "case resources manifest preset");
        std::ifstream manifestFile(argv[3]);
        Json manifest;
        manifestFile >> manifest;
        const auto directory = std::filesystem::path(argv[3]).parent_path();
        const auto member = manifest["preset_membership"][argv[4]];
        std::ifstream file(directory / member["records"].get<std::string>());
        Json fixture;
        file >> fixture;
        fixture = PrintIlluminantFixtures::apply(directory / member["records"].get<std::string>(), fixture);
        const std::string mode = argv[1];
        if (mode == "boundary")
            boundary(fixture);
        else if (mode == "binding")
            binding(fixture);
        else {
            JuicerCuda::Owner owner;
            owner.create(argv[2]);
            JuicerProcess::root().ensure_bootstrap();
            if (mode == "products")
                fj_test_input_hull_products(fixture, directory / member["tables"].get<std::string>());
            else if (mode == "cache")
                cache(argv[2], fixture);
            else if (mode.starts_with("admission-"))
                fj_test_input_hull_admission(static_cast<std::uint32_t>(std::stoul(mode.substr(10))));
            else
                throw std::runtime_error("unknown input hull case");
            status(owner.close(), FJ_STATUS_SUCCESS);
        }
        std::puts("PASS input hull contract");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
