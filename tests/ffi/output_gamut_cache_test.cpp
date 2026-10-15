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

namespace JuicerAssets::OutputGamutTest {
    thread_local bool locked = false;
    thread_local bool rejectAllocation = false;
    thread_local std::optional<std::uint64_t> forcedGeneration;
    thread_local std::uint64_t observedGeneration = 0;
    thread_local bool rejectIlluminantClear = false;
    std::atomic<unsigned> violations{0};
    std::atomic<unsigned> allocations{0};
    std::atomic<unsigned> destructions{0};
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
    void enter_lock() noexcept {
        locked = true;
    }
    void leave_lock() noexcept {
        locked = false;
    }
    void observe_generation(std::uint64_t& generation) noexcept {
        if (forcedGeneration) {
            generation = *forcedGeneration;
            forcedGeneration.reset();
        }
    }
    void after_miss(std::uint64_t generation) {
        if (locked)
            ++violations;
        observedGeneration = generation;
        if (missGate)
            missGate->pause();
    }
    void after_selection(bool warm, std::uint64_t generation, const std::shared_ptr<const Gamut::OutputBoundaryTable>&) {
        if (locked)
            ++violations;
        observedGeneration = generation;
        if (selectedGate && warm == selectedWarm)
            selectedGate->pause();
    }
    void before_illuminant_clear() {
        if (locked)
            ++violations;
        if (std::exchange(rejectIlluminantClear, false))
            throw std::bad_alloc{};
    }
    void after_clear(bool, std::uint64_t generation) noexcept {
        if (locked)
            ++violations;
        observedGeneration = generation;
    }
} // namespace JuicerAssets::OutputGamutTest
namespace JuicerGamut::OutputTest {
    void before_build() {
        if (JuicerAssets::OutputGamutTest::locked)
            ++JuicerAssets::OutputGamutTest::violations;
    }
} // namespace JuicerGamut::OutputTest
// Fixture-only allocator observations/rejection; never linked into the product.
// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
void* operator new(std::size_t bytes) {
    if (JuicerAssets::OutputGamutTest::locked)
        ++JuicerAssets::OutputGamutTest::violations;
    if (bytes >= sizeof(Gamut::OutputBoundaryTable))
        ++JuicerAssets::OutputGamutTest::allocations;
    if (std::exchange(JuicerAssets::OutputGamutTest::rejectAllocation, false))
        throw std::bad_alloc{};
    if (void* storage = std::malloc(bytes == 0 ? 1 : bytes))
        return storage;
    throw std::bad_alloc{};
}
void operator delete(void* storage) noexcept {
    if (JuicerAssets::OutputGamutTest::locked)
        ++JuicerAssets::OutputGamutTest::violations;
    ++JuicerAssets::OutputGamutTest::destructions;
    std::free(storage);
}
void operator delete(void* storage, std::size_t) noexcept {
    if (JuicerAssets::OutputGamutTest::locked)
        ++JuicerAssets::OutputGamutTest::violations;
    ++JuicerAssets::OutputGamutTest::destructions;
    std::free(storage);
}
// NOLINTEND(readability-inconsistent-declaration-parameter-name)

namespace {
    using Hold = std::shared_ptr<const Gamut::OutputBoundaryTable>;
    using Json = nlohmann::json;
    namespace Hooks = JuicerAssets::OutputGamutTest;
    void require(bool condition, const std::string& message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void status(FjStatus value, std::uint32_t category) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "complete cache/cleanup status");
    }
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
        require(std::memcmp(hold->cmax.data(), expected.words.data(), sizeof(hold->cmax)) == 0, "retained every table word");
    }
    Hold acquire(JuicerAssets::Library& library, const Expected& expected) {
        std::string diagnostic;
        auto hold = library.output_boundary_table(expected.transform, diagnostic);
        require(bool(hold), diagnostic);
        completed(hold, expected);
        return hold;
    }
    void clear(JuicerAssets::Library& library, std::uint64_t generation) {
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
        Worker(JuicerAssets::Library& library, const Expected& input, bool pauseMiss, bool pauseSelected = false, bool warm = false, unsigned fault = 0, bool nativeAllocation = false) {
            std::promise<Outcome> promise;
            result = promise.get_future();
            thread = std::thread([&, pauseMiss, pauseSelected, warm, fault, nativeAllocation, promise = std::move(promise)]() mutable {
                Hooks::missGate = pauseMiss ? &miss : nullptr;
                Hooks::selectedGate = pauseSelected ? &selected : nullptr;
                Hooks::selectedWarm = warm;
                Outcome out;
                try {
                    if (fault)
                        status(fj_test_output_gamut_arm_fault(2, 0, fault), FJ_STATUS_SUCCESS);
                    Hooks::rejectAllocation = nativeAllocation;
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
                if (fault)
                    out.consumed = fj_test_output_gamut_fault_consumed(2, 0, fault) == 1 && fj_test_output_gamut_fault_consumed(2, 0, fault) == 0;
                fj_test_output_gamut_clear_fault();
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
    void stale(JuicerAssets::Library& library, const Expected& a, const Expected& b, bool repeated) {
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
    void selection(JuicerAssets::Library& library, const Expected& a, const Expected& b, int variant, bool matching) {
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
    void own_failure(JuicerAssets::Library& library, const Expected& a, const Expected& b, FailureTiming timing, unsigned fault, bool matching, bool nativeAllocation) {
        Worker actor(library, a, true, false, false, fault, nativeAllocation);
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
        require(!result.hold && result.category == (nativeAllocation || fault == 3 ? FJ_STATUS_ALLOCATION_FAILURE : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                                                               : FJ_STATUS_PREPARATION_FAILURE),
                "own failure never substitutes winner/private success");
        require(nativeAllocation || result.consumed, "one operation-specific fault consumed with no hidden retry");
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
            JuicerAssets::Library library(resources);
            Worker actor(library, a, true);
            actor.miss.acknowledge();
            const auto winner = acquire(library, a);
            actor.miss.resume();
            const auto result = actor.finish();
            require(result.hold == winner, "same-generation matching contenders select winner");
            completed(winner, a);
        }
        {
            JuicerAssets::Library library(resources);
            Worker actor(library, conflict, true);
            actor.miss.acknowledge();
            const auto winner = acquire(library, a);
            actor.miss.resume();
            const auto result = actor.finish();
            require(!result.hold && result.diagnostic.find("published_identity") != std::string::npos && acquire(library, a) == winner, "same-generation conflicts reject without replacement");
        }
        {
            JuicerAssets::Library library(resources);
            Worker actor(library, a, true);
            actor.miss.acknowledge();
            clear(library, 1);
            actor.miss.resume();
            const auto result = actor.finish();
            completed(result.hold, a);
            require(acquire(library, a) != result.hold, "clear before any publication leaves stale candidate uncached");
        }
        {
            JuicerAssets::Library library(resources);
            const auto winner = acquire(library, a);
            auto bad = a.transform;
            bad.hash ^= 1;
            bad.contractHash ^= 1;
            std::string diagnostic;
            auto before = fj_test_output_gamut_visits(2);
            require(!library.output_boundary_table(bad, diagnostic) && diagnostic.find("published_identity") != std::string::npos, "transform mismatch precedes contract");
            bad.hash = a.transform.hash;
            require(!library.output_boundary_table(bad, diagnostic) && diagnostic.find("contract_identity") != std::string::npos, "ordered contract mismatch");
            require(fj_test_output_gamut_visits(2) == before && acquire(library, a) == winner, "warm mismatches do not rebuild/replace");
            before = fj_test_output_gamut_visits(2);
            const auto allocations = Hooks::allocations.load();
            for (int i = 0; i < 100; ++i)
                require(acquire(library, a) == winner, "warm owner");
            require(fj_test_output_gamut_visits(2) == before && Hooks::allocations.load() == allocations, "warm no foreign build or table allocation");
        }
        {
            JuicerAssets::Library library(resources);
            auto old = acquire(library, a);
            auto peer = old;
            clear(library, 1);
            const auto current = acquire(library, a);
            old.reset();
            completed(peer, a);
            require(acquire(library, a) == current, "older payload release preserves current winner and peer");
            peer.reset();
            completed(current, a);
            require(acquire(library, a) == current, "last older payload release preserves current winner");
        }
        for (bool earlier : {false, true}) {
            JuicerAssets::Library library(resources);
            const auto winner = acquire(library, a);
            const auto hull = library.input_compression_hull();
            require(bool(hull), "A8 retained hull available");
            Hooks::forcedGeneration = std::numeric_limits<std::uint64_t>::max();
            require(acquire(library, a) == winner && Hooks::observedGeneration == std::numeric_limits<std::uint64_t>::max(), "forced exhaustion established before clear");
            Hooks::rejectIlluminantClear = earlier;
            fj_test_cache_release_failure();
            std::array<char, 256> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            status(library.release_cached_payloads(&error), earlier ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE);
            require(Hooks::observedGeneration == std::numeric_limits<std::uint64_t>::max() && acquire(library, a) == winner, "exhaustion preserves slots/generation");
            require(std::string(bytes.data(), error.length) == (earlier ? "host cache cleanup allocation failed" : "output boundary cache generation exhausted"), "cleanup first-failure text/precedence");
            {
                JuicerAssets::Library independent(resources);
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
        for (bool matching : {true, false}) {
            const auto& a = matching ? b : conflict;
            for (bool repeated : {false, true}) {
                JuicerAssets::Library library(argv[1]);
                stale(library, a, b, repeated);
                std::printf("W%u completion/identity/all-word/retention PASS\n", repeated ? matching ? 3u : 4u : matching ? 1u
                                                                                                                          : 2u);
            }
            for (int variant = 0; variant < 2; ++variant) {
                JuicerAssets::Library library(argv[1]);
                selection(library, a, b, variant, matching);
                std::printf("W%u selected hold %s PASS\n", variant == 0 ? 6u : 7u, matching ? "matching" : "conflicting");
            }
            for (auto timing : {FailureTiming::SameGeneration, FailureTiming::AfterClear, FailureTiming::AfterRepeatedClear})
                for (unsigned fault : {2u, 3u, 5u}) {
                    JuicerAssets::Library library(argv[1]);
                    own_failure(library, a, b, timing, fault, matching, false);
                }
        }
        {
            JuicerAssets::Library library(argv[1]);
            selection(library, b, b, 2, true);
            std::puts("W7 self-publication selection before clear PASS");
        }
        for (auto timing : {FailureTiming::SameGeneration, FailureTiming::AfterClear, FailureTiming::AfterRepeatedClear}) {
            JuicerAssets::Library library(argv[1]);
            own_failure(library, b, b, timing, 0, true, true);
        }
        controls(argv[1], {b, conflict});
        require(Hooks::violations.load() == 0 && Hooks::allocations.load() > 0 && Hooks::destructions.load() > 0, "allocation/free/foreign/format/wait scope controls");
        status(owner.close(), FJ_STATUS_SUCCESS);
        std::puts("W1-W7, own failures/retries, same-generation, mismatch order, exhaustion/precedence/Rust cleanup, destruction scopes and A8 lifetime PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
