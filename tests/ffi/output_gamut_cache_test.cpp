#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "GamutCompression.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RustGamutBridge.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "spectral_allocation_probe.h"

namespace JuicerAssets::OutputGamutTest {
    thread_local bool rejectAllocation = false;
    thread_local bool rejectWrapperAllocation = false;
    thread_local std::optional<std::uint64_t> forcedGeneration;
    thread_local std::uint64_t observedGeneration = 0;
    thread_local bool rejectIlluminantClear = false;
    std::atomic<unsigned> violations{0};
    struct Gate {
        std::mutex mutex;
        std::condition_variable condition;
        bool entered = false;
        bool released = false;
        void pause() {
            std::unique_lock<std::mutex> lock(mutex);
            entered = true;
            condition.notify_all();
            if (!condition.wait_for(lock, std::chrono::seconds(10), [&] {
                    return released;
                }))
                throw std::runtime_error("finite cache gate release timeout");
        }
        void acknowledge() {
            std::unique_lock<std::mutex> lock(mutex);
            if (!condition.wait_for(lock, std::chrono::seconds(10), [&] {
                    return entered;
                }))
                throw std::runtime_error("finite cache gate entry timeout");
        }
        void resume() {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
            condition.notify_all();
        }
    };
    thread_local Gate* missGate = nullptr;
    thread_local Gate* selectedGate = nullptr;
    thread_local bool selectedWarm = false;
    void after_miss(std::uint64_t generation) {
        observedGeneration = generation;
        if (missGate)
            missGate->pause();
    }
    void after_selection(bool warm, std::uint64_t generation, std::uintptr_t) {
        observedGeneration = generation;
        if (selectedGate && warm == selectedWarm)
            selectedGate->pause();
    }
} // namespace JuicerAssets::OutputGamutTest

namespace {
    using Json = nlohmann::json;
    namespace Hooks = JuicerAssets::OutputGamutTest;
    void require(bool condition, const std::string& message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void status(FjStatus value, std::uint32_t category) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "complete cache/cleanup status");
    }
    struct BoundaryHold {
        FjOutputBoundaryOwner* owner = nullptr;
        FjOutputBoundaryView view{};
        std::span<const float> cmax;
        std::uint64_t transformHash = 0, contractHash = 0, hash = 0;
        bool valid = false;
        ~BoundaryHold() {
            (void)fj_test_output_boundary_release(owner, nullptr);
        }
    };
    class Hold {
        std::shared_ptr<BoundaryHold> value;

    public:
        Hold() = default;
        explicit Hold(std::shared_ptr<BoundaryHold> input) : value(std::move(input)) {}
        explicit operator bool() const {
            return bool(value);
        }
        const BoundaryHold* operator->() const {
            return value.get();
        }
        void reset() {
            value.reset();
        }
        friend bool operator==(const Hold& a, const Hold& b) {
            return (!a && !b) || (a && b && a->view.entry_identity == b->view.entry_identity);
        }
    };
    struct HullHold {
        std::shared_ptr<FjInputHullOwner> owner;
        FjInputHullView view{};
        explicit operator bool() const {
            return bool(owner);
        }
        friend bool operator==(const HullHold& a, const HullHold& b) {
            return a.view.entry_identity == b.view.entry_identity;
        }
    };
    std::atomic<std::uint64_t> rustAllocations{0}, rustDestructions{0};
    class OutputCacheProbe {
        // Observed calls borrow the native Library's existing Assets owner and
        // select/retain its actual Rust entries.
        JuicerAssets::Library native;
        bool lateFailure = false;
        static std::uint32_t after_miss(std::uint64_t generation) noexcept {
            try {
                Hooks::after_miss(generation);
                return FJ_STATUS_SUCCESS;
            } catch (...) {
                return FJ_STATUS_INTERNAL_FAILURE;
            }
        }
        static std::uint32_t after_selection(std::uint32_t warm, std::uint64_t generation) noexcept {
            try {
                Hooks::after_selection(warm != 0, generation, 0);
                return FJ_STATUS_SUCCESS;
            } catch (...) {
                return FJ_STATUS_INTERNAL_FAILURE;
            }
        }

    public:
        explicit OutputCacheProbe(const std::filesystem::path& resources) : native(resources) {}
        ~OutputCacheProbe() {
            FjCacheMetrics m{};
            const auto result = fj_test_cache_metrics(owner(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, &m, nullptr);
            if (result.category != FJ_STATUS_SUCCESS || result.api != FJ_API_NONE || result.native_code != 0)
                ++Hooks::violations;
            rustAllocations += m.allocations;
            rustDestructions += m.destructions;
        }
        FjAssets* owner() const {
            return native.borrow_rust_assets();
        }
        FjCacheMetrics metrics() const {
            FjCacheMetrics out{};
            status(fj_test_cache_metrics(owner(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, &out, nullptr), FJ_STATUS_SUCCESS);
            return out;
        }
        Hold output_boundary_table(const Gamut::OutputGamutTransform& transform, std::string& diagnostic) {
            if (Hooks::forcedGeneration) {
                status(fj_test_output_cache_set_generation(owner(), *Hooks::forcedGeneration, nullptr), FJ_STATUS_SUCCESS);
                Hooks::forcedGeneration.reset();
            }
            if (std::exchange(Hooks::rejectAllocation, false))
                status(fj_test_cache_fault(owner(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, 3, nullptr), FJ_STATUS_SUCCESS);
            FjOutputBoundarySource source{};
            source.output_space = static_cast<std::uint32_t>(transform.outputColorSpace);
            std::copy(transform.d65XyzToNativeRgb.begin(), transform.d65XyzToNativeRgb.end(), source.d65_xyz_to_native_rgb);
            source.transform_hash = transform.hash;
            const FjOutputCacheHooks hooks{after_miss, after_selection};
            FjOutputBoundaryOwner* acquired = nullptr;
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            diagnostic.clear();
            const auto result = fj_test_assets_output_boundary_acquire_observed(owner(), &source, transform.contractHash, &hooks, &acquired, &error);
            if (result.category != FJ_STATUS_SUCCESS) {
                require(acquired == nullptr, "failed Rust acquisition publishes no hold");
                diagnostic.assign(bytes.data(), error.length);
                if (result.category == FJ_STATUS_ALLOCATION_FAILURE)
                    throw std::bad_alloc{};
                if (result.category != FJ_STATUS_PREPARATION_FAILURE)
                    throw JuicerCuda::ExecutionFailure{{result, diagnostic}};
                return {};
            }
            std::unique_ptr<FjOutputBoundaryOwner, decltype([](FjOutputBoundaryOwner* value) {
                                (void)fj_test_output_boundary_release(value, nullptr);
                            })>
                guard(acquired);
            if (std::exchange(Hooks::rejectWrapperAllocation, false))
                SpectralAllocationProbe::arm();
            auto held = std::make_shared<BoundaryHold>();
            held->owner = guard.release();
            status(fj_test_output_boundary_view(held->owner, &held->view, nullptr), FJ_STATUS_SUCCESS);
            held->cmax = {held->view.cmax.data, held->view.cmax.count};
            held->transformHash = held->view.transform_hash;
            held->contractHash = held->view.contract_hash;
            held->hash = held->view.hash;
            held->valid = true;
            return Hold(std::move(held));
        }
        HullHold input_compression_hull() {
            FjInputHullOwner* acquired = nullptr;
            status(fj_test_assets_input_hull_acquire(owner(), &acquired, nullptr), FJ_STATUS_SUCCESS);
            HullHold out{std::shared_ptr<FjInputHullOwner>(acquired, [](FjInputHullOwner* value) {
                             (void)fj_test_input_hull_release(value, nullptr);
                         }),
                         {}};
            status(fj_test_input_hull_view(acquired, &out.view, nullptr), FJ_STATUS_SUCCESS);
            return out;
        }
        void fail_clear_late() {
            lateFailure = true;
        }
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) {
            status(fj_test_cache_clear_failures(owner(), std::exchange(Hooks::rejectIlluminantClear, false), std::exchange(lateFailure, false), nullptr), FJ_STATUS_SUCCESS);
            const auto result = fj_legacy_assets_release_cached_payloads(owner(), error);
            status(fj_test_output_cache_generation(owner(), &Hooks::observedGeneration, nullptr), FJ_STATUS_SUCCESS);
            return result;
        }
    };
    struct Expected {
        Gamut::OutputGamutTransform transform;
        std::vector<std::uint32_t> words;
        std::uint64_t hash = 0;
    };
    enum class FailureTiming : std::uint8_t {
        SameGeneration,
        AfterClear,
        AfterRepeatedClear
    };
    struct ControlInputs {
        const Expected& matching;
        const Expected& conflicting;
    };
    Expected expected(const Json& row, std::ifstream& tables) {
        FjOutputGamutSource source{};
        source.output_space = row["tag"].get<std::uint32_t>();
        const auto decode = [](float* destination, const Json& data) {
            for (std::size_t i = 0; i < data.size(); ++i)
                destination[i] = std::bit_cast<float>(data[i].get<std::uint32_t>());
        };
        decode(source.rgb_to_xyz, row["source"]["rgb_to_xyz"]);
        decode(source.xyz_to_rgb, row["source"]["xyz_to_rgb"]);
        decode(source.white_xyz, row["source"]["white"]);
        decode(source.d65_white_xyz, row["source"]["d65"]);
        FjOutputGamutTransform result{};
        status(fj_legacy_gamut_output_transform(&source, &result, nullptr), FJ_STATUS_SUCCESS);
        Expected out;
        std::copy_n(result.native_rgb_to_d65_xyz, 9, out.transform.nativeRgbToD65Xyz.begin());
        std::copy_n(result.d65_xyz_to_native_rgb, 9, out.transform.d65XyzToNativeRgb.begin());
        out.transform.outputColorSpace = static_cast<OutputEncoding::ColorSpace>(source.output_space);
        out.transform.hash = result.hash;
        out.transform.contractHash = result.contract_hash;
        out.transform.valid = true;
        out.hash = row["boundary_hash"].get<std::uint64_t>();
        out.words.resize(Gamut::kOutputBoundaryValueCount);
        tables.clear();
        tables.seekg(row["table_offset"].get<std::streamoff>());
        tables.read(reinterpret_cast<char*>(out.words.data()), static_cast<std::streamsize>(out.words.size() * sizeof(std::uint32_t)));
        require(bool(tables), "frozen complete cache operands/table");
        return out;
    }
    void completed(const Hold& hold, const Expected& expected) {
        require(hold && hold->valid && hold->transformHash == expected.transform.hash && hold->contractHash == expected.transform.contractHash && hold->hash == expected.hash, "retained complete identities");
        require(hold->cmax.size() == expected.words.size() && std::memcmp(hold->cmax.data(), expected.words.data(), expected.words.size() * sizeof(std::uint32_t)) == 0, "retained every table word");
    }
    Hold acquire(OutputCacheProbe& library, const Expected& expected) {
        std::string diagnostic;
        auto hold = library.output_boundary_table(expected.transform, diagnostic);
        require(bool(hold), diagnostic);
        completed(hold, expected);
        return hold;
    }
    void clear(OutputCacheProbe& library, std::uint64_t generation) {
        status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        require(Hooks::observedGeneration == generation, "clear generation/completion acknowledgement");
    }
    struct Outcome {
        Hold hold;
        std::string diagnostic;
        std::uint32_t category = FJ_STATUS_SUCCESS;
        bool consumed = false;
    };
    struct Worker {
        Hooks::Gate miss;
        Hooks::Gate selected;
        std::future<Outcome> result;
        std::thread thread;
        Worker(OutputCacheProbe& library, const Expected& input, bool pauseMiss, bool pauseSelected = false, bool warm = false, unsigned fault = 0, bool entryAllocation = false) {
            std::promise<Outcome> promise;
            result = promise.get_future();
            thread = std::thread([&, pauseMiss, pauseSelected, warm, fault, entryAllocation, promise = std::move(promise)]() mutable {
                Hooks::missGate = pauseMiss ? &miss : nullptr;
                Hooks::selectedGate = pauseSelected ? &selected : nullptr;
                Hooks::selectedWarm = warm;
                Outcome out;
                try {
                    if (fault)
                        status(fj_test_cache_fault(library.owner(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, fault, nullptr), FJ_STATUS_SUCCESS);
                    Hooks::rejectAllocation = entryAllocation;
                    out.hold = library.output_boundary_table(input.transform, out.diagnostic);
                    if (!out.hold)
                        out.category = FJ_STATUS_PREPARATION_FAILURE;
                } catch (const std::bad_alloc&) {
                    out.category = FJ_STATUS_ALLOCATION_FAILURE;
                } catch (const JuicerCuda::ExecutionFailure& error) {
                    out.category = error.failure.status.category;
                    out.diagnostic = error.failure.diagnostic;
                } catch (const std::exception& error) {
                    out.category = FJ_STATUS_INTERNAL_FAILURE;
                    out.diagnostic = error.what();
                }
                if (fault) {
                    std::uint64_t first = 0, second = 0;
                    status(fj_test_cache_fault_consumed(library.owner(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, &first, nullptr), FJ_STATUS_SUCCESS);
                    status(fj_test_cache_fault_consumed(library.owner(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, &second, nullptr), FJ_STATUS_SUCCESS);
                    out.consumed = first == 1 && second == 0;
                }
                status(fj_test_cache_fault(library.owner(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, 0, nullptr), FJ_STATUS_SUCCESS);
                Hooks::rejectAllocation = false;
                Hooks::missGate = nullptr;
                Hooks::selectedGate = nullptr;
                promise.set_value(std::move(out));
            });
        }
        ~Worker() {
            miss.resume();
            selected.resume();
            if (thread.joinable())
                thread.join();
        }
        Outcome finish() {
            require(result.wait_for(std::chrono::seconds(10)) == std::future_status::ready, "finite return completion acknowledgement");
            auto out = result.get();
            thread.join();
            return out;
        }
    };
    void stale(OutputCacheProbe& library, const Expected& a, const Expected& b, bool repeated) {
        Worker actor(library, a, true);
        actor.miss.acknowledge();
        const auto old = acquire(library, b);
        clear(library, 1);
        Hold n1, n2;
        if (repeated) {
            n1 = acquire(library, b);
            clear(library, 2);
            n2 = acquire(library, b);
        }
        actor.miss.resume();
        const auto result = actor.finish();
        require(result.category == FJ_STATUS_SUCCESS && result.diagnostic.empty(), "stale success never consults former/new winners");
        completed(result.hold, a);
        require(result.hold != old, "stale private owner distinct from returned B");
        completed(old, b);
        if (repeated) {
            require(result.hold != n1 && result.hold != n2 && n1 != n2, "distinct retained generations/private owner");
            require(acquire(library, b) == n2 && Hooks::observedGeneration == 2, "generation-2 winner unchanged");
            completed(n1, b);
            completed(n2, b);
        } else {
            const auto fresh = acquire(library, b);
            require(fresh != result.hold && fresh != old && Hooks::observedGeneration == 1, "stale return leaves slot empty for N1");
            require(acquire(library, b) == fresh, "N1 warm winner");
            completed(fresh, b);
        }
        completed(result.hold, a);
        completed(old, b);
    }
    void selection(OutputCacheProbe& library, const Expected& a, const Expected& b, int variant, bool matching) {
        Hold old;
        if (variant == 0)
            old = acquire(library, b);
        Worker actor(library, a, variant != 0, true, variant == 0);
        if (variant != 0) {
            actor.miss.acknowledge();
            if (variant == 1)
                old = acquire(library, b);
            actor.miss.resume();
        }
        actor.selected.acknowledge();
        clear(library, 1);
        const auto current = acquire(library, b);
        actor.selected.resume();
        const auto result = actor.finish();
        if (matching) {
            completed(result.hold, a);
            require(result.hold != current && (variant == 2 || result.hold == old), "selected pre-clear hold remains authoritative");
        } else {
            require(!result.hold && result.diagnostic.find("published_identity") != std::string::npos, "selected conflicting hold fails after clear");
        }
        require(acquire(library, b) == current, "later winner remains unchanged");
        completed(current, b);
        if (old)
            completed(old, b);
    }
    void own_failure(OutputCacheProbe& library, const Expected& a, const Expected& b, FailureTiming timing, unsigned fault, bool matching, bool entryAllocation) {
        Worker actor(library, a, true, false, false, fault, entryAllocation);
        actor.miss.acknowledge();
        const auto old = acquire(library, b);
        Hold current = old;
        if (timing != FailureTiming::SameGeneration) {
            clear(library, 1);
            current.reset();
        }
        Hold n1;
        if (timing == FailureTiming::AfterRepeatedClear) {
            n1 = acquire(library, b);
            clear(library, 2);
            current = acquire(library, b);
        }
        const auto generation = static_cast<std::uint64_t>(timing);
        actor.miss.resume();
        const auto result = actor.finish();
        require(!result.hold && result.category == (entryAllocation || fault == 3 ? FJ_STATUS_ALLOCATION_FAILURE : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                                                              : FJ_STATUS_PREPARATION_FAILURE),
                "own failure never substitutes winner/private success");
        require(entryAllocation || result.consumed, "one operation-specific fault consumed with no hidden retry");
        completed(old, b);
        if (n1)
            completed(n1, b);
        if (current) {
            require(acquire(library, b) == current && Hooks::observedGeneration == generation, "own failure preserves slot/generation");
            if (matching)
                require(acquire(library, a) == current, "matching retry reuses current winner");
            else {
                std::string diagnostic;
                require(!library.output_boundary_table(a.transform, diagnostic) && diagnostic.find("published_identity") != std::string::npos, "separate conflict retry rejects current winner");
                clear(library, generation + 1);
                const auto retried = acquire(library, a);
                require(acquire(library, a) == retried, "explicit clear allows independent conflict retry");
            }
        } else {
            const auto retried = acquire(library, a);
            require(Hooks::observedGeneration == generation && acquire(library, a) == retried, "empty-cache separate retry publishes");
        }
    }
    void controls(const std::filesystem::path& resources, const ControlInputs& inputs) {
        const auto& a = inputs.matching;
        const auto& conflict = inputs.conflicting;
        {
            OutputCacheProbe library(resources);
            Worker actor(library, a, true);
            actor.miss.acknowledge();
            const auto winner = acquire(library, a);
            actor.miss.resume();
            const auto result = actor.finish();
            require(result.hold == winner, "same-generation matching contenders select winner");
            completed(winner, a);
        }
        {
            OutputCacheProbe library(resources);
            Worker actor(library, conflict, true);
            actor.miss.acknowledge();
            const auto winner = acquire(library, a);
            actor.miss.resume();
            const auto result = actor.finish();
            require(!result.hold && result.diagnostic.find("published_identity") != std::string::npos && acquire(library, a) == winner, "same-generation conflicts reject without replacement");
        }
        {
            OutputCacheProbe library(resources);
            Worker actor(library, a, true);
            actor.miss.acknowledge();
            clear(library, 1);
            actor.miss.resume();
            const auto result = actor.finish();
            completed(result.hold, a);
            require(acquire(library, a) != result.hold, "clear before any publication leaves stale candidate uncached");
        }
        {
            OutputCacheProbe library(resources);
            const auto winner = acquire(library, a);
            auto bad = a.transform;
            bad.hash ^= 1;
            bad.contractHash ^= 1;
            std::string diagnostic;
            auto before = library.metrics().derivations;
            require(!library.output_boundary_table(bad, diagnostic) && diagnostic.find("published_identity") != std::string::npos, "transform mismatch precedes contract");
            bad.hash = a.transform.hash;
            require(!library.output_boundary_table(bad, diagnostic) && diagnostic.find("contract_identity") != std::string::npos, "ordered contract mismatch");
            require(library.metrics().derivations == before && acquire(library, a) == winner, "warm mismatches do not rebuild/replace");
            before = library.metrics().derivations;
            const auto allocations = library.metrics().allocations;
            for (int i = 0; i < 100; ++i)
                require(acquire(library, a) == winner, "warm owner");
            require(library.metrics().derivations == before && library.metrics().allocations == allocations, "warm no foreign build or table allocation");
            Hooks::rejectWrapperAllocation = true;
            bool wrapperFailure = false;
            try {
                (void)library.output_boundary_table(a.transform, diagnostic);
            } catch (const std::bad_alloc&) {
                wrapperFailure = true;
            }
            require(wrapperFailure && library.metrics().allocations == allocations && acquire(library, a) == winner, "native wrapper allocation failure leaves actual Rust winner intact");
        }
        {
            OutputCacheProbe library(resources);
            auto old = acquire(library, a);
            auto peer = old;
            clear(library, 1);
            const auto current = acquire(library, a);
            old.reset();
            completed(peer, a);
            require(acquire(library, a) == current, "older payload release preserves current winner and peer");
            const auto before = library.metrics();
            peer.reset();
            const auto after = library.metrics();
            require(after.destructions == before.destructions + 1 && after.live_bytes != 0 && after.live_bytes < before.live_bytes, "last hold reclaims actual old Rust backing");
            completed(current, a);
            require(acquire(library, a) == current, "last older payload release preserves current winner");
        }
        for (bool earlier : {false, true}) {
            OutputCacheProbe library(resources);
            const auto winner = acquire(library, a);
            const auto hull = library.input_compression_hull();
            require(bool(hull), "A8 retained hull available");
            Hooks::forcedGeneration = std::numeric_limits<std::uint64_t>::max();
            require(acquire(library, a) == winner && Hooks::observedGeneration == std::numeric_limits<std::uint64_t>::max(), "forced exhaustion established before clear");
            Hooks::rejectIlluminantClear = earlier;
            library.fail_clear_late();
            std::array<char, 256> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            status(library.release_cached_payloads(&error), earlier ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE);
            require(Hooks::observedGeneration == std::numeric_limits<std::uint64_t>::max() && acquire(library, a) == winner, "exhaustion preserves slots/generation");
            require(std::string(bytes.data(), error.length) == (earlier ? "host cache cleanup allocation failed" : "output boundary cache generation exhausted"), "cleanup first-failure text/precedence");
            {
                OutputCacheProbe independent(resources);
                status(independent.release_cached_payloads(), FJ_STATUS_SUCCESS);
            }
            status(library.release_cached_payloads(), FJ_STATUS_INTERNAL_FAILURE);
            require(library.input_compression_hull() == hull, "A8 retained-hull rule unchanged");
            completed(winner, a);
        }
    }
} // namespace
int main(int argc, char** argv) {
    JuicerCuda::Owner owner;
    try {
        require(argc == 4, "resources, fixture manifest, preset");
        owner.create(argv[1]);
        JuicerProcess::root().ensure_bootstrap();
        std::ifstream input(argv[2]);
        Json manifest;
        input >> manifest;
        const auto member = manifest["preset_membership"].at(argv[3]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        std::ifstream records(directory / member["records"].get<std::string>());
        Json fixture;
        records >> fixture;
        std::ifstream tables(directory / member["tables"].get<std::string>(), std::ios::binary);
        const auto b = expected(fixture["rows"][0], tables);
        const auto conflict = expected(fixture["rows"][1], tables);
        for (unsigned allocation = 1; allocation <= 2; ++allocation) {
            JuicerAssets::Library native(argv[1]);
            std::string diagnostic;
            auto warm = native.output_boundary_table(b.transform, diagnostic);
            require(warm && warm->hash == b.hash && std::memcmp(warm->cmax.data(), b.words.data(), b.words.size() * sizeof(std::uint32_t)) == 0, "native forwarded boundary matches every frozen word");
            SpectralAllocationProbe::arm(allocation);
            bool failed = false;
            try {
                (void)native.output_boundary_table(b.transform, diagnostic);
            } catch (const std::bad_alloc&) {
                failed = true;
            }
            require(failed && SpectralAllocationProbe::bytes() != 0, "actual native boundary handle/control-block allocation failure");
            SpectralAllocationProbe::clear();
            status(native.release_cached_payloads(), FJ_STATUS_SUCCESS);
            FjCacheMetrics metrics{};
            status(fj_test_cache_metrics(native.borrow_rust_assets(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, &metrics, nullptr), FJ_STATUS_SUCCESS);
            require(metrics.allocations == 1 && metrics.destructions == 0, "native warm hold survives cache clear without a replacement allocation");
            // Releasing the only successful wrapper must reclaim the entry; no
            // failed wrapper may have leaked its earlier foreign acquisition.
            warm.reset();
            status(fj_test_cache_metrics(native.borrow_rust_assets(), FJ_TEST_CACHE_OUTPUT_BOUNDARY, &metrics, nullptr), FJ_STATUS_SUCCESS);
            require(metrics.destructions == 1 && metrics.live_bytes == 0, "failed native boundary wrappers release their foreign holds");
        }
        for (bool matching : {true, false}) {
            const auto& a = matching ? b : conflict;
            for (bool repeated : {false, true}) {
                OutputCacheProbe library(argv[1]);
                stale(library, a, b, repeated);
                std::printf("W%u completion/identity/all-word/retention PASS\n", repeated ? matching ? 3u : 4u : matching ? 1u
                                                                                                                          : 2u);
            }
            for (int variant = 0; variant < 2; ++variant) {
                OutputCacheProbe library(argv[1]);
                selection(library, a, b, variant, matching);
                std::printf("W%u selected hold %s PASS\n", variant == 0 ? 6u : 7u, matching ? "matching" : "conflicting");
            }
            for (auto timing : {FailureTiming::SameGeneration, FailureTiming::AfterClear, FailureTiming::AfterRepeatedClear})
                for (unsigned fault : {2u, 3u, 5u}) {
                    OutputCacheProbe library(argv[1]);
                    own_failure(library, a, b, timing, fault, matching, false);
                }
        }
        {
            OutputCacheProbe library(argv[1]);
            selection(library, b, b, 2, true);
            std::puts("W7 self-publication selection before clear PASS");
        }
        for (auto timing : {FailureTiming::SameGeneration, FailureTiming::AfterClear, FailureTiming::AfterRepeatedClear}) {
            OutputCacheProbe library(argv[1]);
            own_failure(library, b, b, timing, 0, true, true);
        }
        controls(argv[1], {b, conflict});
        require(Hooks::violations.load() == 0 && rustAllocations.load() > 0 && rustDestructions.load() > 0, "allocation/free/foreign/format/wait scope controls");
        status(owner.close(), FJ_STATUS_SUCCESS);
        std::puts("W1-W7, own failures/retries, same-generation, mismatch order, exhaustion/precedence/Rust cleanup, destruction scopes and A8 lifetime PASS");
        return 0;
    } catch (const JuicerCuda::ExecutionFailure& error) {
        std::fprintf(stderr, "%s\n", error.failure.diagnostic.c_str());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
