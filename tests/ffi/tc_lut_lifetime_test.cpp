#include <atomic>
#include <bit>
#include <new>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RustReconstructionBridge.h"
#include "ColorTransforms.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "spectral_allocation_probe.h"

namespace {
    void require(bool condition, const char* why) {
        if (!condition)
            throw std::runtime_error(why);
    }
    struct Witness {
        InstanceState* state = nullptr;
        const float* wanted = nullptr;
        std::atomic<bool> requested{false}, done{false}, outside{false};
        std::atomic<unsigned> count{0};
    };
    Witness* witness = nullptr; // Non-owning test observation, never an old-state hold.
    void release_observer(const float* samples) noexcept {
        if (witness && samples == witness->wanted) {
            witness->count.fetch_add(1);
            witness->requested.store(true, std::memory_order_release);
            witness->requested.notify_one();
            while (!witness->done.load(std::memory_order_acquire))
                witness->done.wait(false);
        }
    }
    const float* current_pointer(InstanceState& state, bool print) {
        // This temporary lookup ends before replacement. Only the slot retains it.
        if (print) {
            const auto value = JuicerAtomic::load_shared_ptr(&state.activePrintState);
            return value->payload.filmTcLut->samples().data();
        }
        const auto value = JuicerAtomic::load_shared_ptr(&state.activeDirectState);
        return value->payload.filmTcLut->samples().data();
    }
    ParamSnapshot snapshot(bool positive, bool print) {
        ParamSnapshot p;
        p.filmProfileKey = positive ? "fujifilm_provia_100f" : "kodak_portra_400";
        p.printProfileKey = "kodak_portra_endura";
        p.scanRoute = static_cast<Spektrafilm::ScanRoute>((positive ? 2 : 0) + (print ? 1 : 0));
        p.spectralUpsamplingMode = 0;
        p.cameraAutoExposureEnabled = 0;
        p.inputCompressionEnabled = 0;
        return p;
    }
    bool rebuild(InstanceState& s, const ParamSnapshot& p) {
        return Spektrafilm::scan_route_is_print(p.scanRoute) ? rebuild_print_render_state(s, p) : rebuild_direct_render_state(s, p);
    }
    void publication(bool positive, bool fromPrint, int action) {
        InstanceState state;
        auto p = snapshot(positive, fromPrint);
        require(rebuild(state, p), "initial actual publication");
        Witness observation;
        observation.state = &state;
        observation.wanted = current_pointer(state, fromPrint);
        const auto oldCounter = state.buildCounterNext.load();
        std::array<std::size_t, 2> countsBefore{};
        require(fj_test_tc_lut_allocation_counts(countsBefore.data()).category == FJ_STATUS_SUCCESS, "Rust allocation counters before publication");
        witness = &observation;
        Spectral::TcLutTest::set_release_observer(release_observer);
        std::thread inspect([&] {
            while (!observation.requested.load(std::memory_order_acquire))
                observation.requested.wait(false);
            bool publicationFree = false, rebuildFree = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            do {
                if (!publicationFree && state.m.try_lock()) {
                    state.m.unlock();
                    publicationFree = true;
                }
                if (!rebuildFree && state.rebuildMutex.try_lock()) {
                    state.rebuildMutex.unlock();
                    rebuildFree = true;
                }
                if (publicationFree && rebuildFree)
                    break;
                std::this_thread::yield();
            } while (std::chrono::steady_clock::now() < deadline);
            observation.outside.store(publicationFree && rebuildFree);
            observation.done.store(true, std::memory_order_release);
            observation.done.notify_one();
        });
        if (action == 1)
            p = snapshot(positive, !fromPrint);
        else
            p.cameraExposureCompensationEv += .25;
        if (action == 2)
            require(fj_test_tc_lut_arm_fault(1, 1, 4).category == FJ_STATUS_SUCCESS, "ordinary failure injection");
        const bool result = rebuild(state, p);
        // If there was no matching release, wake the observer and fail boundedly.
        if (!observation.requested.load()) {
            observation.requested.store(true);
            observation.requested.notify_one();
        }
        inspect.join();
        Spectral::TcLutTest::set_release_observer(nullptr);
        witness = nullptr;
        std::array<std::size_t, 2> countsAfter{};
        require(fj_test_tc_lut_allocation_counts(countsAfter.data()).category == FJ_STATUS_SUCCESS && countsAfter[1] == countsBefore[1] + 1, "Rust deallocation reached exactly once");
        require(observation.count == 1 && observation.outside, "exactly-once no-reader release outside both lock scopes");
        require(result == (action != 2), "publication result preserved");
        if (action == 2) {
            require(fj_test_tc_lut_fault_consumed(1, 4) == 1, "ordinary failure consumed before cleanup");
            require(state.buildCounterNext == oldCounter, "failed rebuild counter preserved");
            require(fromPrint ? !JuicerAtomic::load_shared_ptr(&state.activePrintState) : !JuicerAtomic::load_shared_ptr(&state.activeDirectState), "selected failure slot clears");
            fj_test_tc_lut_clear_fault();
        } else
            require(state.buildCounterNext == oldCounter + 1, "successful rebuild counter");
        std::printf("F1 positive=%d from_print=%d action=%d release_count=1 publication_lock_free=1 rebuild_lock_free=1\n", positive, fromPrint, action);
    }
    void retained_reader() {
        InstanceState state;
        auto p = snapshot(false, false);
        require(rebuild(state, p), "retained initial");
        auto held = JuicerAtomic::load_shared_ptr(&state.activeDirectState);
        const float* address = held->payload.filmTcLut->samples().data();
        p.cameraExposureCompensationEv += .5;
        require(rebuild(state, p), "retained replacement");
        require(held->payload.filmTcLut->samples().data() == address, "retained original allocation");
        require(JuicerProcess::root().assets().release_cached_payloads().category == FJ_STATUS_SUCCESS, "cache release");
        std::array<float, 3> rgb{};
        std::string diagnostic;
        require(Spectral::sample_film_tc_lut(*held->payload.filmTcLut, {1, 0, 0}, rgb, diagnostic), "retained sampling after replacement/cache release");
        std::thread finalRelease([hold = std::move(held)]() mutable {
            hold.reset();
        });
        finalRelease.join();
    }
    const float* moveWanted = nullptr;
    unsigned moveReleases = 0;
    void move_observer(const float* samples) noexcept {
        if (samples == moveWanted)
            ++moveReleases;
    }
    std::optional<Spectral::FilmTcLut> make_owner() {
        FilmRawRecipe recipe;
        recipe.tcLutHash = 1;
        recipe.inputCompressionActive = false;
        recipe.rgbToRawMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
        for (auto& row : recipe.finalSensitivity)
            row = {1, 2, 4};
        std::array<float, 81> spd{};
        spd.fill(1);
        std::optional<Spectral::FilmTcLut> owner;
        std::string diagnostic;
        require(Spectral::build_film_tc_lut(recipe, Spectral::gHanSpectra, spd, owner, diagnostic), "native completed owner");
        return owner;
    }
    void native_owner_moves_and_allocation_terminal() {
        auto first = make_owner(), second = make_owner();
        moveWanted = first->samples().data();
        const auto* secondAddress = second->samples().data();
        moveReleases = 0;
        Spectral::TcLutTest::set_release_observer(move_observer);
        std::optional<Spectral::FilmTcLut> moved{std::move(*first)};
        require(first->samples().empty() && moved->samples().data() == moveWanted, "move preserves allocation and empties source");
        auto* same = &*moved;
        *moved = std::move(*same);
        require(moveReleases == 0 && moved->samples().data() == moveWanted, "safe self move");
        *moved = std::move(*second);
        require(moveReleases == 1 && second->samples().empty() && moved->samples().data() == secondAddress, "move assignment releases previous allocation once");
        moveWanted = secondAddress;
        moveReleases = 0;
        std::thread release([owner = std::optional<Spectral::FilmTcLut>{std::move(*moved)}]() mutable {
            owner.reset();
        });
        release.join();
        require(moveReleases == 1 && moved->samples().empty(), "native cross-thread last release");
        moveReleases = 0;
        bool unwound = false;
        try {
            auto owner = make_owner();
            moveWanted = owner->samples().data();
            throw std::runtime_error("owner unwind witness");
        } catch (const std::runtime_error&) {
            unwound = true;
        }
        require(unwound && moveReleases == 1, "exception unwind releases native owner once");
        Spectral::TcLutTest::set_release_observer(nullptr);
        FilmRawRecipe recipe;
        recipe.tcLutHash = 1;
        recipe.inputCompressionActive = false;
        recipe.rgbToRawMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
        recipe.hanatos.spectralGaussianBlur = std::bit_cast<float>(0x5c000000u);
        for (auto& row : recipe.finalSensitivity)
            row = {1, 1, 1};
        std::array<float, 81> spd{};
        spd.fill(1);
        std::optional<Spectral::FilmTcLut> owner;
        std::string diagnostic;
        bool memory = false;
        SpectralAllocationProbe::arm();
        try {
            (void)Spectral::build_film_tc_lut(recipe, Spectral::gHanSpectra, spd, owner, diagnostic);
        } catch (const std::bad_alloc&) {
            memory = true;
        }
        SpectralAllocationProbe::clear();
        require(memory && !owner && SpectralAllocationProbe::bytes() == 0, "actual failed Rust reservation becomes bad_alloc before diagnostic allocation");
        std::puts("Native move/assignment/self-move/unwind/cross-thread and real failed-reservation terminal PASS");
    }
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "resource root required");
        JuicerCuda::Owner owner;
        owner.create(argv[1]);
        JuicerProcess::root().ensure_bootstrap();
        for (bool positive : {false, true})
            for (bool print : {false, true})
                for (int action = 0; action < 3; ++action)
                    publication(positive, print, action);
        retained_reader();
        native_owner_moves_and_allocation_terminal();
        require(owner.close().category == FJ_STATUS_SUCCESS, "close host owner");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
