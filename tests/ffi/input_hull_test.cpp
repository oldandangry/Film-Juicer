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
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "GamutCompression.h"
#include "ProcessRoot.h"
#include "ProcessRuntime.h"
#include "test_asset_library.h"
#include "RenderValues.h"
#include "JuicerState.h"
#include "RustGamutBridge.h"
#include "SpectralData.h"
#include "juicer_test_api.h"
#include "case_results.h"
#include "PreparedState.h"
#include "print_illuminant_fixture.h"
#include "prepared_boundary.h"
#include "spectral_allocation_probe.h"
#include "spectral_source_fixture.h"

#define FJ_HULL_SIZE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_HULL_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "input_hull_abi_facts.inc"
#undef FJ_HULL_SIZE
#undef FJ_HULL_FIELD

extern "C" FjStatus fj_test_input_hull_abi_c(const FjInputHullSource*, FjInputHull*, FjErrorBuffer*);
void fj_test_input_hull_products(const FjAssets*, const nlohmann::json&, const std::filesystem::path&);
void fj_test_input_hull_admission(const FjAssets*, std::uint32_t fault);


namespace {
    using Json = nlohmann::json;
    void require(bool ok, const std::string& message) {
        if (!ok)
            throw std::runtime_error(message);
    }
    void status(FjStatus value, std::uint32_t category) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "complete status category/API/code");
    }
    struct HullHold {
        std::shared_ptr<FjInputHullOwner> owner;
        FjInputHullView view{};
        explicit operator bool() const {
            return bool(owner);
        }
        void reset() {
            owner.reset();
            view = {};
        }
        friend bool operator==(const HullHold& a, const HullHold& b) {
            return a.view.entry_identity == b.view.entry_identity;
        }
    };
    struct ObservationRelease {
        void operator()(FjCacheObservationOwner* owner) const noexcept {
            (void)fj_test_cache_observation_release(owner, nullptr);
        }
    };
    using ObservationHold = std::unique_ptr<FjCacheObservationOwner, ObservationRelease>;
    std::barrier<>* rustHullStart = nullptr;
    std::atomic<unsigned> rustHullEntered{0};
    std::array<std::uintptr_t, 4> rustHullCandidates{};
    std::uint32_t before_rust_hull(std::uintptr_t entry, std::size_t bytes) noexcept {
        try {
            require(entry != 0 && bytes >= sizeof(FjInputHull), "actual substantial Rust candidate allocation before math");
            if (rustHullStart) {
                rustHullCandidates.at(rustHullEntered.fetch_add(1)) = entry;
                rustHullStart->arrive_and_wait();
            }
            return FJ_STATUS_SUCCESS;
        } catch (const std::bad_alloc&) {
            return FJ_STATUS_ALLOCATION_FAILURE;
        } catch (...) {
            return FJ_STATUS_INTERNAL_FAILURE;
        }
    }
    class InputHullCacheProbe {
        JuicerAssets::Library native;

    public:
        explicit InputHullCacheProbe(const std::filesystem::path& root)
            : native(root) {}
        FjAssets* owner() const {
            return native.borrow_rust_assets();
        }
        HullHold acquire() {
            FjInputHullOwner* acquired = nullptr;
            const FjInputHullCacheHooks hooks{before_rust_hull};
            const auto result = fj_test_assets_input_hull_acquire_observed(owner(), &hooks, &acquired, nullptr);
            if (result.category != FJ_STATUS_SUCCESS) {
                require(acquired == nullptr, "failed hull returns no hold");
                if (result.category == FJ_STATUS_ALLOCATION_FAILURE)
                    throw std::bad_alloc{};
                if (result.category == FJ_STATUS_PREPARATION_FAILURE)
                    return {};
                throw JuicerCuda::ExecutionFailure{{result, "Rust input hull acquisition failed"}};
            }
            HullHold held{std::shared_ptr<FjInputHullOwner>(acquired, [](FjInputHullOwner* value) {
                              (void)fj_test_input_hull_release(value, nullptr);
                          }),
                          {}};
            status(fj_test_input_hull_view(acquired, &held.view, nullptr), FJ_STATUS_SUCCESS);
            return held;
        }
        ObservationHold observation() const {
            FjCacheObservationOwner* acquired = nullptr;
            status(fj_test_cache_observation_acquire(owner(), &acquired, nullptr), FJ_STATUS_SUCCESS);
            return ObservationHold(acquired);
        }
        void clear() const {
            status(fj_legacy_assets_release_cached_payloads(owner(), nullptr), FJ_STATUS_SUCCESS);
        }
        void fault(unsigned value) const {
            status(fj_test_cache_fault(owner(), FJ_TEST_CACHE_INPUT_HULL, value, nullptr), FJ_STATUS_SUCCESS);
        }
        std::uint64_t consumed() const {
            std::uint64_t value = 0;
            status(fj_test_cache_fault_consumed(owner(), FJ_TEST_CACHE_INPUT_HULL, &value, nullptr), FJ_STATUS_SUCCESS);
            return value;
        }
    };
    FjCacheMetrics hull_metrics(const ObservationHold& owner) {
        FjCacheMetrics out{};
        status(fj_test_cache_observation_metrics(owner.get(), FJ_TEST_CACHE_INPUT_HULL, &out, nullptr), FJ_STATUS_SUCCESS);
        return out;
    }
    bool entry_live(const ObservationHold& owner, unsigned family, std::uintptr_t entry) {
        std::uint32_t value = 0;
        status(fj_test_cache_observation_entry_live(owner.get(), family, entry, &value, nullptr), FJ_STATUS_SUCCESS);
        return value != 0;
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
        constexpr std::size_t kPolygonPoints = 1025;
        require(expected["xy"].size() == kPolygonPoints, "frozen native hull primitive extent");
        require(bits(hull.center_xy) == expected["center"] && hull.hash == expected["hash"], "center/hash exact");
        std::size_t visited = 0;
        for (std::size_t i = 0; i < kPolygonPoints; ++i) {
            require(bits(std::array{hull.xy[i * 2], hull.xy[i * 2 + 1]}) == expected["xy"][i], "complete polygon words");
            ++visited;
        }
        require(visited == kPolygonPoints, "actual native hull polygon traversal count");
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
            const auto row_id = std::to_string(row.at("id").get<int>());
            Source source(row);
            const auto request = source.record();
            for (auto call : {fj_test_input_hull_abi_c, fj_test_gamut_input_hull}) {
                const std::string caller = call == fj_test_input_hull_abi_c ? "c11" : "facade";
                std::string case_id = caller;
                case_id += "/";
                case_id += row_id;
                JuicerTests::CaseEvent event("input_hull.boundary", case_id);
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
                event.passed();
            }
        }
        Source source(fixture["math"][0]);
        const auto request = source.record();
        for (std::uint32_t facade = 0; facade <= 1; ++facade) {
            const auto call = facade ? fj_test_gamut_input_hull : fj_test_input_hull_abi_c;
            for (std::uint32_t fault = 1; fault <= 6; ++fault) {
                JuicerTests::CaseEvent event("input_hull.boundary_fault", std::to_string(facade) + "/" + std::to_string(fault));
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
                event.passed();
            }
        }
        for (int condition = 0; condition < 13; ++condition) {
            JuicerTests::CaseEvent event("input_hull.boundary_condition", std::to_string(condition));
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
            event.passed();
        }
        JuicerTests::CaseEvent tail_event("input_hull.boundary", "c11_status_and_alias_contract");
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
        tail_event.passed();
    }
    void binding(const Json& fixture) {
        Gamut::InputCompressionHull retained;
        for (const auto& row : fixture["math"]) {
            JuicerTests::CaseEvent event("input_hull.binding", std::to_string(row.at("id").get<int>()));
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
            event.passed();
        }
        Source source(fixture["math"][0]);
        const auto request = source.record();
        for (std::uint32_t fault : {1u, 2u, 3u, 4u, 5u, 6u}) {
            JuicerTests::CaseEvent event("input_hull.binding_fault", std::to_string(fault));
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
            event.passed();
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
        JuicerTests::Cohort allocation_failures("input_hull.cache_allocation", {"1", "2"});
        for (unsigned allocation = 1; allocation <= 2; ++allocation) {
            auto row_event = allocation_failures.case_event(std::to_string(allocation));
            JuicerAssets::Library native(resources);
            (void)native.input_compression_hull();
            FjCacheObservationOwner* raw = nullptr;
            status(fj_test_cache_observation_acquire(native.borrow_rust_assets(), &raw, nullptr), FJ_STATUS_SUCCESS);
            ObservationHold observation(raw);
            const auto before = hull_metrics(observation);
            SpectralAllocationProbe::arm(allocation);
            bool failed = false;
            try {
                (void)native.input_compression_hull();
            } catch (const std::bad_alloc&) {
                failed = true;
            }
            const auto bytes = SpectralAllocationProbe::bytes();
            SpectralAllocationProbe::clear();
            require(failed && bytes != 0 && bytes < sizeof(FjInputHull), "actual native hull handle/control-block allocation failure");
            require(hull_metrics(observation).allocations == before.allocations, "native wrapper failure does not allocate another Rust hull");
            status(native.close(), FJ_STATUS_SUCCESS);
            require(hull_metrics(observation).live_bytes == 0, "failed native hull wrapper releases its acquired foreign hold");
            row_event.passed();
        }
        {
            InputHullCacheProbe library(resources);
            auto observation = library.observation();
            library.fault(1);
            status(fj_test_input_hull_entry_fault(library.owner(), 1, nullptr), FJ_STATUS_SUCCESS);
            bool memory = false;
            try {
                (void)library.acquire();
            } catch (const std::bad_alloc&) {
                memory = true;
            }
            std::uint32_t allocationConsumed = 0;
            status(fj_test_input_hull_entry_fault_consumed(library.owner(), &allocationConsumed, nullptr), FJ_STATUS_SUCCESS);
            require(memory && allocationConsumed == 1 && hull_metrics(observation).allocations == 0, "real failed entry reservation precedes math and publishes no backing");
            require(library.consumed() == 0, "entry reservation failure leaves mathematical fault pending");
            bool exceptional = false;
            try {
                (void)library.acquire();
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                exceptional = true;
                status(failure.failure.status, FJ_STATUS_UNSUPPORTED_INPUT);
            }
            require(exceptional && library.consumed() == 1, "next actual cold operation consumes pending math fault once");
            require(hull_metrics(observation).allocations == 1 && hull_metrics(observation).destructions == 1 && hull_metrics(observation).live_bytes == 0, "failed cold candidate's actual storage reclaimed");
            const auto recovered = library.acquire();
            require(bool(recovered), "entry and mathematical failures permit independent retry");
            const auto before = hull_metrics(observation);
            SpectralAllocationProbe::arm();
            bool wrapperFailure = false;
            try {
                (void)library.acquire();
            } catch (const std::bad_alloc&) {
                wrapperFailure = true;
            }
            const auto bytes = SpectralAllocationProbe::bytes();
            SpectralAllocationProbe::clear();
            require(wrapperFailure && bytes != 0 && bytes < sizeof(FjInputHull), "native wrapper allocation failure is separate from substantial Rust entry allocation");
            require(hull_metrics(observation).allocations == before.allocations && library.acquire() == recovered, "wrapper allocation failure preserves the Rust winner");
        }
        {
            InputHullCacheProbe retry(resources);
            retry.fault(4);
            require(!retry.acquire(), "ordinary cold failure publishes no hull");
            require(retry.consumed() == 1, "cold owner consumes fault before retry");
            const auto recovered = retry.acquire();
            require(recovered && recovered.view.hash == fixture["math"][0]["expected"]["hash"], "cold retry constructs complete hull");
        }
        HullHold retained;
        std::optional<PreparedState> recipeHold;
        ObservationHold observation;
        std::uintptr_t identity = 0;
        {
            InputHullCacheProbe library(resources);
            observation = library.observation();
            std::array<HullHold, 4> results;
            std::barrier start(4);
            rustHullStart = &start;
            rustHullEntered = 0;
            std::array<std::thread, 4> workers;
            for (std::size_t i = 0; i < workers.size(); ++i)
                workers[i] = std::thread([&, i] {
                    results[i] = library.acquire();
                });
            for (auto& worker : workers)
                worker.join();
            rustHullStart = nullptr;
            require(rustHullEntered == 4, "four actual cold candidates rendezvous before math outside cache mutex");
            for (std::size_t i = 0; i < 4; ++i) {
                require(results[i] && results[i] == results[0], "one published Rust winner");
                for (std::size_t j = 0; j < i; ++j)
                    require(rustHullCandidates[i] != rustHullCandidates[j], "distinct fallible Rust candidate allocations");
            }
            const auto metrics = hull_metrics(observation);
            require(metrics.allocations == 4 && metrics.derivations == 4 && metrics.destructions == 3 && metrics.live_bytes != 0, "all losing Rust candidates reclaimed, one live backing");
            retained = results[0];
            identity = retained.view.entry_identity;
            const ParamSnapshot captured;
            std::array<char, 2048> errorText{};
            FjErrorBuffer error{errorText.data(), errorText.size(), 0};
            status(prepare_state(library.owner(), captured, recipeHold, error), FJ_STATUS_SUCCESS);
            FjPreparedStateView state{};
            status(recipeHold->view(state, error), FJ_STATUS_SUCCESS);
            require(state.input_hull.entry_identity == identity && state.input_hull.xy.data == retained.view.xy.data, "complete recipe and foreign cache holds share actual backing");
            for (auto& result : results)
                result.reset();
            FjIlluminantsOwner* source = nullptr;
            status(fj_test_assets_illuminants_acquire(library.owner(), &source, nullptr), FJ_STATUS_SUCCESS);
            FjIlluminantsView sourceView{};
            status(fj_test_illuminants_view(source, &sourceView, nullptr), FJ_STATUS_SUCCESS);
            const auto sourceIdentity = sourceView.entry_identity;
            status(fj_test_illuminants_release(source, nullptr), FJ_STATUS_SUCCESS);
            library.clear();
            require(!entry_live(observation, FJ_TEST_CACHE_ILLUMINANTS, sourceIdentity), "source curves expire independently of hull and complete state");
            library.fault(1);
            status(fj_test_input_hull_source_tail(library.owner(), 1, .125f, nullptr), FJ_STATUS_SUCCESS);
            require(library.acquire() == retained && library.consumed() == 0, "warm owner bypasses changed cold source and mathematical fault");
            status(fj_test_gamut_arm_fault(0, 1), FJ_STATUS_SUCCESS);
            status(fj_test_assets_illuminants_acquire(library.owner(), &source, nullptr), FJ_STATUS_SUCCESS);
            status(fj_test_illuminants_view(source, &sourceView, nullptr), FJ_STATUS_SUCCESS);
            JuicerTestSupport::CmfSource cmfSource(library.owner());
            const auto xSource = cmfSource.x();
            const auto ySource = cmfSource.y();
            const auto zSource = cmfSource.z();
            require(xSource.count == 81 && ySource.count == 81 && zSource.count == 81,
                    "held source CMF curves each contain 81 samples");
            std::vector<float> x(xSource.data, xSource.data + xSource.count);
            x[80] += .125f;
            const FjInputHullSource request{{x.data(), x.size()}, ySource, zSource, sourceView.d65};
            FjInputHull output{};
            FjErrorBuffer diagnostic{nullptr, 0, 0};
            status(fj_test_input_hull_abi_c(&request, &output, &diagnostic), FJ_STATUS_UNSUPPORTED_INPUT);
            require(fj_test_gamut_fault_consumed(0, 1) == 1, "standalone mathematical fault consumed by next actual C call");
            status(fj_test_illuminants_release(source, nullptr), FJ_STATUS_SUCCESS);
            InputHullCacheProbe changed(resources);
            status(fj_test_input_hull_source_tail(changed.owner(), 1, .125f, nullptr), FJ_STATUS_SUCCESS);
            const auto rebuilt = changed.acquire();
            require(rebuilt && rebuilt.view.hash == fixture["math"][1]["expected"]["hash"] && rebuilt.view.hash != retained.view.hash, "separate cold owner observes controlled tail variation without published-source mutation");
            for (std::size_t i = 0; i < 1025; ++i)
                require(bits(std::array{rebuilt.view.xy.data[i * 2], rebuilt.view.xy.data[i * 2 + 1]}) == fixture["math"][1]["expected"]["xy"][i], "all varied-source hull words");
            require(library.acquire() == retained && library.consumed() == 0, "independent owner never consumes this warm owner's fault");
            library.fault(0);
            require(hull_metrics(observation).allocations == 4 && hull_metrics(observation).derivations == 4, "warm state and cache acquisition allocate/derive no hull");
        }
        require(entry_live(observation, FJ_TEST_CACHE_INPUT_HULL, identity), "Assets destruction releases only its cache hold");
        FjPreparedStateView heldState{};
        std::array<char, 2048> errorText{};
        FjErrorBuffer error{errorText.data(), errorText.size(), 0};
        status(recipeHold->view(heldState, error), FJ_STATUS_SUCCESS);
        require(heldState.input_hull.entry_identity == identity && heldState.input_hull.hash == retained.view.hash, "complete state remains readable after Assets expires");
        heldState = {};
        recipeHold.reset();
        require(entry_live(observation, FJ_TEST_CACHE_INPUT_HULL, identity), "external final hold survives complete state release");
        std::thread([hold = std::move(retained)] {
            require(hold.view.hash == 0x5cda57333bb9b653ULL, "retained Rust hull thread transfer");
        }).join();
        require(hull_metrics(observation).live_bytes == 0 && hull_metrics(observation).destructions == 4, "last external hold reclaims actual backing after owner destruction");
        fj_test_gamut_clear_fault();
    }
} // namespace

int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc == 6 && std::string_view(argv[argc - 1]) == "--list-cases") {
            std::ifstream manifestFile(argv[3]);
            Json manifest;
            manifestFile >> manifest;
            const auto member = manifest["preset_membership"][argv[4]];
            const auto directory = std::filesystem::path(argv[3]).parent_path();
            std::ifstream records(directory / member["records"].get<std::string>());
            const Json listed = PrintIlluminantFixtures::apply(directory / member["records"].get<std::string>(), Json::parse(records));
            const std::string selected = argv[1];
            if (selected == "boundary") {
                for (const auto& row : listed["math"])
                    for (const char* caller : {"c11", "facade"})
                        JuicerTests::list_case("input_hull.boundary", std::string(caller) + "/" + std::to_string(row.at("id").get<int>()));
                for (std::uint32_t facade = 0; facade <= 1; ++facade)
                    for (std::uint32_t fault = 1; fault <= 6; ++fault)
                        JuicerTests::list_case("input_hull.boundary_fault", std::to_string(facade) + "/" + std::to_string(fault));
                for (int condition = 0; condition < 13; ++condition)
                    JuicerTests::list_case("input_hull.boundary_condition", std::to_string(condition));
                JuicerTests::list_case("input_hull.boundary", "c11_status_and_alias_contract");
            } else if (selected == "binding") {
                for (const auto& row : listed["math"])
                    JuicerTests::list_case("input_hull.binding", std::to_string(row.at("id").get<int>()));
                for (std::uint32_t fault = 1; fault <= 6; ++fault)
                    JuicerTests::list_case("input_hull.binding_fault", std::to_string(fault));
            } else if (selected == "products") {
                JuicerTests::list_case("input_hull", selected);
                for (int method : {0, 1, 2})
                    JuicerTests::list_case("input_hull.product", "no-work/method/" + std::to_string(method));
                JuicerTests::list_case("input_hull.product", "no-work/consume-pending-fault");
                for (const auto& row : listed["products"]) {
                    const int route = row.value("route", 0), polarity = row.value("polarity", 0), method = row.at("method").get<int>();
                    const auto id = method == 1
                                        ? "method/1/compression/" + std::to_string(row.at("compression").get<int>())
                                        : "route/" + std::to_string(route) + "/polarity/" + std::to_string(polarity) + "/method/" + std::to_string(method) + "/compression/" + std::to_string(row.at("compression").get<int>());
                    JuicerTests::list_case("input_hull.product", id);
                }
                JuicerTests::list_case("input_hull", "owner_close");
            } else if (selected == "cache") {
                JuicerTests::list_case("input_hull", selected);
                for (unsigned allocation = 1; allocation <= 2; ++allocation)
                    JuicerTests::list_case("input_hull.cache_allocation", std::to_string(allocation));
                JuicerTests::list_case("input_hull", "owner_close");
            } else if (selected.starts_with("admission-")) {
                const auto fault = std::stoul(selected.substr(10));
                const auto id = "route/3/method/2/fault/" + std::to_string(fault);
                JuicerTests::list_case("input_hull", selected);
                for (const char* stage : {"cold", "current", "warm-cache", "standalone-raw"})
                    JuicerTests::list_case("input_hull.admission", id + "/" + stage);
                JuicerTests::list_case("input_hull", "owner_close");
            } else {
                JuicerTests::list_case("input_hull", selected);
                JuicerTests::list_case("input_hull", "owner_close");
            }
            return 0;
        }
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
            JuicerProcess::ProcessRuntime runtime;
            runtime.create(argv[2]);
            if (mode == "products") {
                JuicerTests::CaseEvent event("input_hull", mode);
                fj_test_input_hull_products(runtime.borrow().assets, fixture, directory / member["tables"].get<std::string>());
                event.passed();
            } else if (mode == "cache") {
                JuicerTests::CaseEvent event("input_hull", mode);
                cache(argv[2], fixture);
                event.passed();
            } else if (mode.starts_with("admission-")) {
                JuicerTests::CaseEvent event("input_hull", mode);
                fj_test_input_hull_admission(runtime.borrow().assets, static_cast<std::uint32_t>(std::stoul(mode.substr(10))));
                event.passed();
            } else
                throw std::runtime_error("unknown input hull case");
            JuicerTests::CaseEvent close_event("input_hull", "owner_close");
            status(runtime.close(), FJ_STATUS_SUCCESS);
            close_event.passed();
        }
        std::puts("PASS input hull contract");
        case_log.finish();
        return 0;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        std::fprintf(stderr, "%s\n", error.failure.diagnostic.c_str());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
