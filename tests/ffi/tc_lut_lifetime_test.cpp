#include <atomic>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "case_results.h"
#include "prepared_product_fixture.h"

namespace {
    void require(bool condition, const char* why) {
        if (!condition)
            throw std::runtime_error(why);
    }
    struct Witness {
        RenderInstance* publication = nullptr;
        const float* wanted = nullptr;
        std::atomic<bool> requested{false}, done{false}, stateLockFree{false};
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
    const float* current_pointer(const PreparedState& state) {
        FjPreparedStateView view{};
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        require(state.view(view, error).category == FJ_STATUS_SUCCESS, "published prepared-state view");
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
    void publication(bool positive, bool fromPrint, int action) {
        RenderInstance instance;
        auto p = snapshot(positive, fromPrint);
        JuicerTestSupport::capture(instance, p);
        auto publication = JuicerTestSupport::admit(instance);
        require(publication.kind == FJ_ADMISSION_READY && publication.state, "initial actual publication");
        const auto initialMetrics = JuicerTestSupport::PublicationObserver::metrics(instance);
        Witness observation;
        observation.publication = &instance;
        observation.wanted = current_pointer(*publication.state);
        const std::size_t preparedLiveBefore = fj_test_prepared_live_states();
        if (action == 1)
            p = snapshot(positive, !fromPrint);
        else
            p.cameraExposureCompensationEv += .25;
        if (action == 2) {
            require(fj_test_tc_lut_arm_fault(1, 1, 4).category == FJ_STATUS_SUCCESS, "ordinary failure injection");
            JuicerTestSupport::capture(instance, p);
            const auto failed = JuicerTestSupport::admit(instance);
            require(failed.delivery.category == FJ_STATUS_PREPARATION_FAILURE &&
                        failed.kind == FJ_ADMISSION_PREPARATION_FAILED && !failed.state,
                    "TC production failure is returned without admitting the retained previous state");
            require(fj_test_tc_lut_fault_consumed(1, 4) == 1, "ordinary failure consumed before cleanup");
            const auto metrics = JuicerTestSupport::PublicationObserver::metrics(instance);
            require(metrics.inputIdentity == hash_params(p) && metrics.publishedIdentity == initialMetrics.publishedIdentity &&
                        metrics.build == initialMetrics.build && metrics.backingIdentity == initialMetrics.backingIdentity,
                    "TC preparation failure keeps the prior complete publication privately reusable");
            require(current_pointer(*publication.state) == observation.wanted,
                    "admitted TC backing remains readable after current preparation failure");
            JuicerTestSupport::capture(instance, snapshot(positive, fromPrint));
            auto reused = JuicerTestSupport::admit(instance);
            require(reused.kind == FJ_ADMISSION_READY && reused.state &&
                        JuicerTestSupport::PublicationObserver::metrics(instance).build == initialMetrics.build &&
                        JuicerTestSupport::PublicationObserver::metrics(instance).backingIdentity == initialMetrics.backingIdentity,
                    "returning to the admitted identity reuses its retained TC state");
            fj_test_tc_lut_clear_fault();
            return;
        }

        JuicerTestSupport::capture(instance, p);
        auto replacement = JuicerTestSupport::admit(instance);
        require(replacement.kind == FJ_ADMISSION_READY && replacement.state, "successful replacement admission");
        const auto replacementMetrics = JuicerTestSupport::PublicationObserver::metrics(instance);
        require(replacementMetrics.build == initialMetrics.build + 1 &&
                    replacementMetrics.publishedIdentity == hash_params(p) &&
                    replacementMetrics.backingIdentity != initialMetrics.backingIdentity,
                "successful replacement commits matching identity, backing, and build metadata");
        require(fj_test_prepared_live_states() == preparedLiveBefore + 1,
                "replacement admission owns a second complete prepared-state handle");

        witness = &observation;
        fj_test_prepared_state_release_observer(release_observer);
        std::thread inspect([&] {
            while (!observation.requested.load(std::memory_order_acquire))
                observation.requested.wait(false);
            bool stateFree = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            do {
                try {
                    const auto metrics = JuicerTestSupport::PublicationObserver::metrics(*observation.publication);
                    stateFree = metrics.publishedIdentity == hash_params(p) && metrics.build == replacementMetrics.build;
                } catch (...) {
                    stateFree = false;
                }
                if (stateFree)
                    break;
                std::this_thread::yield();
            } while (std::chrono::steady_clock::now() < deadline);
            observation.stateLockFree.store(stateFree);
            observation.done.store(true, std::memory_order_release);
            observation.done.notify_one();
        });
        publication.state.reset();
        if (!observation.requested.load()) {
            observation.requested.store(true);
            observation.requested.notify_one();
        }
        inspect.join();
        fj_test_prepared_state_release_observer(nullptr);
        witness = nullptr;
        require(observation.count.load() == 1 && observation.stateLockFree.load(),
                "retired admitted TC handle releases once while publication observation remains available");
        require(fj_test_prepared_live_states() == preparedLiveBefore,
                "retired TC handle is released after replacement while the new handle remains live");
        std::printf("F1 positive=%d from_print=%d action=%d release_count=1 state_lock_free=1\n", positive, fromPrint, action);
    }
    void retained_reader() {
        RenderInstance instance;
        auto p = snapshot(false, false);
        JuicerTestSupport::capture(instance, p);
        auto held = JuicerTestSupport::admit(instance);
        require(held.kind == FJ_ADMISSION_READY && held.state, "retained initial");
        FjPreparedStateView view{};
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        require(held.state->view(view, error).category == FJ_STATUS_SUCCESS, "retained prepared-state view");
        const FjFloatSpan samples = view.film_exposure.tc_lut_rgba;
        const float* address = samples.data;
        require(samples.count == std::size_t{192} * 192u * 4u, "retained complete TC LUT span");
        p.cameraExposureCompensationEv += .5;
        JuicerTestSupport::capture(instance, p);
        auto replacement = JuicerTestSupport::admit(instance);
        require(replacement.kind == FJ_ADMISSION_READY && replacement.state, "retained replacement");
        FjPreparedStateView retained{};
        require(held.state->view(retained, error).category == FJ_STATUS_SUCCESS && retained.film_exposure.tc_lut_rgba.data == address, "retained original allocation");
        require(JuicerProcess::root().assets().release_cached_payloads().category == FJ_STATUS_SUCCESS, "cache release");
        std::array<float, 3> rgb{};
        const std::array<float, 3> xyz{1, 0, 0};
        require(fj_legacy_reconstruction_sample_tc_lut(retained.film_exposure.tc_lut_rgba, xyz.data(), rgb.data(), &error).category == FJ_STATUS_SUCCESS, "retained sampling after replacement/cache release");
        std::thread finalRelease([hold = std::move(held.state)]() mutable {
            hold.reset();
        });
        finalRelease.join();
    }

} // namespace
int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            for (const char* route : {"negative_direct", "negative_print", "positive_direct", "positive_print"})
                for (int action = 0; action < 3; ++action)
                    JuicerTests::list_case("tc_lut_lifetime", std::string(route) + ".publication." + std::to_string(action));
            JuicerTests::list_case("tc_lut_lifetime", "retained_reader");
            return 0;
        }
        require(argc == 2, "resource root required");
        JuicerCuda::Owner owner;
        owner.create(argv[1]);
        for (bool positive : {false, true})
            for (bool print : {false, true})
                for (int action = 0; action < 3; ++action) {
                    const std::string route = std::string(positive ? "positive_" : "negative_") + (print ? "print" : "direct");
                    JuicerTests::CaseEvent event("tc_lut_lifetime", route + ".publication." + std::to_string(action));
                    publication(positive, print, action);
                    event.passed();
                }
        JuicerTests::CaseEvent retained_event("tc_lut_lifetime", "retained_reader");
        retained_reader();
        retained_event.passed();
        require(owner.close().category == FJ_STATUS_SUCCESS, "close host owner");
        case_log.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
