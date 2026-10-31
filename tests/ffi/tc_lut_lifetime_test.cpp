#include <atomic>
#include <array>
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
#include "prepared_product_fixture.h"
#include "spectral_allocation_probe.h"

namespace {
    void require(bool condition, const char* why) {
        if (!condition)
            throw std::runtime_error(why);
    }
    struct Witness {
        InstanceState* state = nullptr;
        const float* wanted = nullptr;
        std::atomic<bool> requested{false}, done{false}, publicationOutside{false}, rebuildOutside{false};
        std::atomic<unsigned> count{0};
    };
    Witness* witness = nullptr; // Non-owning test observation, never an old-state hold.
    extern "C" void release_observer(const float* samples) noexcept {
        if (witness && samples == witness->wanted) {
            witness->count.fetch_add(1);
            witness->requested.store(true, std::memory_order_release);
            witness->requested.notify_one();
            while (!witness->done.load(std::memory_order_acquire))
                witness->done.wait(false);
        }
    }
    const float* current_pointer(InstanceState& state, bool print) {
        FjPreparedStateView view{};
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        if (print) {
            const auto publication = JuicerAtomic::load_shared_ptr(&state.activePrintState);
            require(bool(publication), "print publication slot has state");
            require(publication->prepared.view(view, error).category == FJ_STATUS_SUCCESS, "published prepared-state view");
        } else {
            const auto publication = JuicerAtomic::load_shared_ptr(&state.activeDirectState);
            require(bool(publication), "direct publication slot has state");
            require(publication->prepared.view(view, error).category == FJ_STATUS_SUCCESS, "published prepared-state view");
        }
        require(view.film_exposure.tc_lut_rgba.count == std::size_t{192} * 192u * 4u, "published TC LUT span extent");
        return view.film_exposure.tc_lut_rgba.data;
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
        const std::size_t preparedLiveBefore = fj_test_prepared_live_states();
        std::array<std::size_t, 2> facadeCountsBefore{};
        const FjStatus facadeCountsBeforeStatus = fj_test_tc_lut_allocation_counts(facadeCountsBefore.data());
        witness = &observation;
        fj_test_prepared_state_release_observer(release_observer);
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
            observation.publicationOutside.store(publicationFree);
            observation.rebuildOutside.store(rebuildFree);
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
        fj_test_prepared_state_release_observer(nullptr);
        witness = nullptr;
        std::array<std::size_t, 2> facadeCountsAfter{};
        const FjStatus facadeCountsAfterStatus = fj_test_tc_lut_allocation_counts(facadeCountsAfter.data());
        const std::size_t preparedLiveAfter = fj_test_prepared_live_states();
        std::fprintf(stderr,
                     "TC publication positive=%d from_print=%d action=%d result=%d old_tc=%p release_pointer_matches=%u publication_lock_free=%d rebuild_lock_free=%d prepared_live=%zu->%zu facade_status=%u/%u facade_transfers=%zu->%zu facade_releases=%zu->%zu\n",
                     positive,
                     fromPrint,
                     action,
                     result,
                     static_cast<const void*>(observation.wanted),
                     observation.count.load(),
                     observation.publicationOutside.load(),
                     observation.rebuildOutside.load(),
                     preparedLiveBefore,
                     preparedLiveAfter,
                     facadeCountsBeforeStatus.category,
                     facadeCountsAfterStatus.category,
                     facadeCountsBefore[0],
                     facadeCountsAfter[0],
                     facadeCountsBefore[1],
                     facadeCountsAfter[1]);
        require(facadeCountsBeforeStatus.category == FJ_STATUS_SUCCESS &&
                    facadeCountsAfterStatus.category == FJ_STATUS_SUCCESS,
                "SF_TEMP_BRIDGE facade counters remain queryable during publication observation");
        require(preparedLiveBefore != 0 &&
                    preparedLiveAfter == preparedLiveBefore - static_cast<std::size_t>(action == 2),
                "published PreparedState owner count follows replacement or selected-slot failure");
        require(observation.count.load() == 1 && observation.publicationOutside.load() &&
                    observation.rebuildOutside.load(),
                "exactly-once old TC pointer release outside both publication and rebuild locks");
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
        FjPreparedStateView view{};
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        require(held->prepared.view(view, error).category == FJ_STATUS_SUCCESS, "retained prepared-state view");
        const FjFloatSpan samples = view.film_exposure.tc_lut_rgba;
        const float* address = samples.data;
        require(samples.count == std::size_t{192} * 192u * 4u, "retained complete TC LUT span");
        p.cameraExposureCompensationEv += .5;
        require(rebuild(state, p), "retained replacement");
        FjPreparedStateView retained{};
        require(held->prepared.view(retained, error).category == FJ_STATUS_SUCCESS && retained.film_exposure.tc_lut_rgba.data == address, "retained original allocation");
        require(JuicerProcess::root().assets().release_cached_payloads().category == FJ_STATUS_SUCCESS, "cache release");
        std::array<float, 3> rgb{};
        const std::array<float, 3> xyz{1, 0, 0};
        require(fj_legacy_reconstruction_sample_tc_lut(retained.film_exposure.tc_lut_rgba, xyz.data(), rgb.data(), &error).category == FJ_STATUS_SUCCESS, "retained sampling after replacement/cache release");
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
        std::array<std::size_t, 2> facadeCountsBeforeReplacement{};
        require(fj_test_tc_lut_allocation_counts(facadeCountsBeforeReplacement.data()).category == FJ_STATUS_SUCCESS,
                "SF_TEMP_BRIDGE facade release counter before move replacement");
        *moved = std::move(*second);
        std::array<std::size_t, 2> facadeCountsAfterReplacement{};
        require(fj_test_tc_lut_allocation_counts(facadeCountsAfterReplacement.data()).category == FJ_STATUS_SUCCESS &&
                    facadeCountsAfterReplacement[1] == facadeCountsBeforeReplacement[1] + 1,
                "SF_TEMP_BRIDGE direct reconstruction facade release counter advances once for move-replaced owner");
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
