#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "instance_host.h"
#include "juicer_ofx_api.h"
#include "../ffi/case_results.h"
#include "../ffi/juicer_test_api.h"

extern "C" {
void fj_test_ofx_instance_control(uint32_t kind);
void fj_test_ofx_instance_counts(uint64_t* created, uint64_t* reclaimed, uint64_t* retired);
FjStatus fj_test_ofx_instance_observe(OfxImageEffectHandle effect, FjInstanceObservation** out, FjErrorBuffer* error);
FjStatus fj_test_ofx_instance_admit(OfxImageEffectHandle effect, FjPreparedState** out, FjErrorBuffer* error);
}

namespace {
    constexpr std::string_view kCaller = "ofx.instance_controls";
    constexpr std::array kCases{"allocation", "panic", "rollback", "terminal", "capture_races"};
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    OfxStatus call(OfxPlugin* plugin, const char* action, const void* effect = nullptr, OfxPropertySetHandle input = nullptr) {
        return plugin->mainEntry(action, effect, input, nullptr);
    }
    struct Counts {
        uint64_t created, reclaimed, retired;
    };
    Counts counts() {
        Counts value{};
        fj_test_ofx_instance_counts(&value.created, &value.reclaimed, &value.retired);
        return value;
    }
    struct Session {
        OfxPlugin* plugin;
        ofx_test::InstanceHost host;
        OfxImageEffectHandle effect;
        explicit Session(OfxPlugin* p) : plugin(p), effect(nullptr) {
            plugin->setHost(host.binding());
            require(call(plugin, kOfxActionLoad) == kOfxStatOK, "load");
            require(call(plugin, kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK, "context");
            effect = host.create_effect();
        }
        void create() {
            require(call(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "create");
        }
        OfxStatus change(const char* name, const char* reason = kOfxChangeUserEdited) {
            auto* args = host.change_args({reason, kOfxTypeParameter, name});
            return call(plugin, kOfxActionInstanceChanged, effect, args);
        }
        void destroy() {
            require(call(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "destroy");
        }
        void unload() {
            require(call(plugin, kOfxActionUnload) == kOfxStatOK, "unload");
        }
        FjInstanceObservationView observe() {
            FjInstanceObservation* owner = nullptr;
            require(fj_test_ofx_instance_observe(effect, &owner, nullptr).category == FJ_STATUS_SUCCESS && owner, "observe");
            FjInstanceObservationView view{};
            require(fj_test_instance_observation_view(owner, &view, nullptr).category == FJ_STATUS_SUCCESS, "observation view");
            require(fj_test_instance_observation_release(owner, nullptr).category == FJ_STATUS_SUCCESS, "observation release");
            return view;
        }
        void admit() {
            FjPreparedState* state = nullptr;
            require(fj_test_ofx_instance_admit(effect, &state, nullptr).category == FJ_STATUS_SUCCESS && state, "admit");
            require(fj_test_prepared_state_release(state, nullptr).category == FJ_STATUS_SUCCESS, "release admitted state");
        }
    };
    void allocation(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "allocation");
        const JuicerTests::Cohort rows("ofx.instance_controls.allocation", {"owner", "key_copy", "ui_text"});
        {
            auto row = rows.case_event("owner");
            Session s(plugin);
            const auto before = counts();
            fj_test_ofx_instance_control(1);
            require(call(plugin, kOfxActionCreateInstance, s.effect) == kOfxStatErrMemory, "owner allocation status");
            require(!s.host.instance_data(s.effect), "allocation published owner");
            const auto after = counts();
            require(after.created == before.created && after.reclaimed == before.reclaimed, "allocation changed owner counts");
            s.unload();
            row.passed();
        }
        {
            auto row = rows.case_event("key_copy");
            Session s(plugin);
            s.create();
            s.host.reenter_next("param_get/HalationActive", [&] {
                fj_test_ofx_instance_control(2);
            });
            require(s.change("Exposure") == kOfxStatErrMemory, "capture copy allocation status");
            require(s.observe().pending_kind == 5, "copy failure did not invalidate capture");
            s.destroy();
            s.unload();
            row.passed();
        }
        {
            auto row = rows.case_event("ui_text");
            Session s(plugin);
            s.create();
            const auto before = counts();
            s.host.clear_calls();
            fj_test_ofx_instance_control(8);
            require(s.change("Exposure") == kOfxStatErrMemory, "hint text allocation status");
            const auto state = s.observe();
            require(state.pending_kind == 5 && state.failure_kind == FJ_CAPTURE_ALLOCATION_FAILURE, "hint allocation did not fail pending capture");
            require(std::none_of(s.host.calls().begin(), s.host.calls().end(), [](const auto& call) {
                        return call.operation == "set_string" && call.property == kOfxParamPropHint;
                    }),
                    "hint setter reached host after allocation failure");
            s.destroy();
            s.unload();
            const auto after = counts();
            require(after.reclaimed == before.reclaimed + 1 && after.retired == before.retired + 1, "hint allocation cleanup was not once-only");
            row.passed();
        }
        test.passed();
    }
    void panic(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "panic");
        const JuicerTests::Cohort rows("ofx.instance_controls.panic", {"published", "capture", "matching_module"});
        {
            auto row = rows.case_event("published");
            Session s(plugin);
            const auto before = counts();
            fj_test_ofx_instance_control(3);
            require(call(plugin, kOfxActionCreateInstance, s.effect) == kOfxStatErrFatal, "published panic status");
            require(!s.host.instance_data(s.effect), "published panic failed to detach");
            const auto after = counts();
            require(after.created == before.created + 1 && after.reclaimed == before.reclaimed + 1, "published panic ownership");
            s.unload();
            row.passed();
        }
        {
            auto row = rows.case_event("capture");
            Session s(plugin);
            s.create();
            fj_test_ofx_instance_control(4);
            require(s.change("Exposure") == kOfxStatErrFatal, "capture panic status");
            require(s.observe().pending_kind == 6, "capture panic did not block publication");
            s.destroy();
            s.unload();
            row.passed();
        }
        {
            auto row = rows.case_event("matching_module");
            Session s(plugin);
            s.create();
            fj_test_ofx_instance_control(4);
            require(s.change("Exposure") == kOfxStatErrFatal, "module panic status");
            require(call(plugin, kOfxActionDescribe, s.host.descriptor_handle()) != kOfxStatOK, "panic retained ordinary admission");
            require(s.host.instance_data(s.effect), "panic lost cleanup owner");
            s.destroy();
            s.unload();
            Session rebound(plugin);
            rebound.create();
            ofx_test::InstanceHost replacement;
            const auto before = counts();
            rebound.host.clear_calls();
            replacement.clear_calls();
            plugin->setHost(replacement.binding());
            require(call(plugin, kOfxActionDescribe, rebound.host.descriptor_handle()) != kOfxStatOK, "rebind admitted ordinary callback");
            require(call(plugin, kOfxActionUnload) == kOfxStatErrFatal, "rebind unloaded a retained owner");
            require(rebound.host.instance_data(rebound.effect), "rebind lost original owner");
            require(call(plugin, kOfxActionDestroyInstance, rebound.effect) == kOfxStatOK, "rebind terminal cleanup failed");
            const auto after = counts();
            require(!rebound.host.instance_data(rebound.effect) && after.reclaimed == before.reclaimed + 1, "rebind failed to reclaim original owner");
            require(!rebound.host.calls().empty() && replacement.calls().empty(), "terminal cleanup used replacement host binding");
            rebound.unload();
            row.passed();
        }
        test.passed();
    }
    void rollback(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "rollback");
        const JuicerTests::Cohort rows("ofx.instance_controls.rollback", {"before_publication", "stored_then_failed", "detach_failed", "retained_destroy"});
        {
            auto row = rows.case_event("before_publication");
            Session s(plugin);
            const auto before = counts();
            s.host.fail_next("param_handle/Exposure", kOfxStatErrBadHandle);
            require(call(plugin, kOfxActionCreateInstance, s.effect) == kOfxStatErrBadHandle, "acquisition failure status");
            require(!s.host.instance_data(s.effect), "failed acquisition published owner");
            const auto after = counts();
            require(after.created == before.created && after.reclaimed == before.reclaimed, "prepublication owner allocation");
            s.unload();
            Session panic_session(plugin);
            const auto panic_before = counts();
            fj_test_ofx_instance_control(7);
            require(call(plugin, kOfxActionCreateInstance, panic_session.effect) == kOfxStatErrFatal, "unpublished panic status");
            const auto panic_after = counts();
            require(!panic_session.host.instance_data(panic_session.effect) && panic_after.created == panic_before.created + 1 && panic_after.reclaimed == panic_before.reclaimed + 1, "unpublished panic ownership");
            panic_session.unload();
            row.passed();
        }
        {
            auto row = rows.case_event("stored_then_failed");
            Session s(plugin);
            const auto before = counts();
            s.host.store_then_fail_next("set_pointer/OfxPropInstanceData", kOfxStatFailed);
            require(call(plugin, kOfxActionCreateInstance, s.effect) == kOfxStatFailed, "publication failure status");
            require(!s.host.instance_data(s.effect), "failed publication not detached");
            const auto after = counts();
            require(after.created == before.created + 1 && after.reclaimed == before.reclaimed + 1, "publication failure reclamation");
            s.unload();
            row.passed();
        }
        for (const auto* id : {"detach_failed", "retained_destroy"}) {
            auto row = rows.case_event(id);
            Session s(plugin);
            const auto before = counts();
            s.host.reenter_next("set_pointer/OfxPropInstanceData", [&] {
                s.host.fail_next("set_pointer/OfxPropInstanceData", kOfxStatFailed);
            });
            fj_test_ofx_instance_control(3);
            require(call(plugin, kOfxActionCreateInstance, s.effect) == kOfxStatErrFatal, "failed detach status");
            require(s.host.instance_data(s.effect), "failed detach lost owner");
            auto retained = counts();
            require(retained.created == before.created + 1 && retained.reclaimed == before.reclaimed, "failed detach reclaimed live pointer");
            require(s.change("Exposure") != kOfxStatReplyDefault, "failed owner admitted ordinary action");
            require(call(plugin, kOfxActionDestroyInstance, s.effect) == kOfxStatOK, "retained owner destroy");
            require(!s.host.instance_data(s.effect), "retained owner pointer not cleared");
            const auto after = counts();
            require(after.reclaimed == before.reclaimed + 1, "retained owner not reclaimed once");
            s.unload();
            row.passed();
        }
        test.passed();
    }
    void terminal(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "terminal");
        const JuicerTests::Cohort rows("ofx.instance_controls.terminal", {"clear_failure", "retire_failure", "first_failure", "once_only"});
        for (const auto* id : {"clear_failure", "retire_failure", "first_failure", "once_only"}) {
            auto row = rows.case_event(id);
            Session s(plugin);
            s.create();
            require(s.change("Exposure") == kOfxStatReplyDefault, "terminal initial capture");
            s.admit();
            const auto before = counts();
            const auto which = std::string_view(id);
            uint64_t retired_at_clear = UINT64_MAX;
            s.host.reenter_next("set_pointer/OfxPropInstanceData", [&] {
                retired_at_clear = counts().retired;
            });
            if (which == "clear_failure" || which == "first_failure" || which == "once_only")
                s.host.fail_next("set_pointer/OfxPropInstanceData", kOfxStatFailed);
            if (which == "retire_failure" || which == "first_failure")
                fj_test_ofx_instance_control(5);
            require(call(plugin, kOfxActionDestroyInstance, s.effect) == kOfxStatErrFatal, "terminal failure must be fatal");
            const auto after = counts();
            require(retired_at_clear == before.retired, "native retirement ran before pointer clear");
            require(after.reclaimed == before.reclaimed + 1 && after.retired == before.retired + 1, "terminal cleanup not once-only");
            if (which == "retire_failure")
                require(!s.host.instance_data(s.effect), "retire failure did not clear pointer");
            // A failed clear leaves an invalid host pointer by contract; never read it through the plugin.
            s.unload();
            const auto closed = counts();
            require(closed.reclaimed == after.reclaimed && closed.retired == after.retired, "unload repeated terminal cleanup");
            if (which == "retire_failure") {
                Session panic_session(plugin);
                panic_session.create();
                require(panic_session.change("Exposure") == kOfxStatReplyDefault, "retirement panic initial capture");
                panic_session.admit();
                const auto panic_before = counts();
                fj_test_ofx_instance_control(6);
                require(call(plugin, kOfxActionDestroyInstance, panic_session.effect) == kOfxStatErrFatal, "retirement panic status");
                const auto panic_after = counts();
                require(!panic_session.host.instance_data(panic_session.effect) && panic_after.retired == panic_before.retired + 1 && panic_after.reclaimed == panic_before.reclaimed + 1, "retirement panic cleanup");
                panic_session.unload();
            }
            row.passed();
        }
        test.passed();
    }
    void capture_races(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "capture_races");
        const JuicerTests::Cohort rows("ofx.instance_controls.capture_races", {"stale_getter", "deferred_write", "refresh_after_write", "partial_write"});
        {
            auto row = rows.case_event("stale_getter");
            Session s(plugin);
            s.create();
            require(s.change("Exposure") == kOfxStatReplyDefault, "initial capture");
            s.host.set_value(s.effect, "FilmProfileKey", {std::string("\xff", 1)});
            OfxStatus nested = kOfxStatFailed;
            s.host.reenter_next("param_get/Exposure", [&] {
                s.host.set_value(s.effect, "FilmProfileKey", {std::string("kodak_portra_400")});
                s.host.set_value(s.effect, "Exposure", {2.0});
                nested = s.change("Exposure");
            });
            require(s.change("Exposure") == kOfxStatReplyDefault && nested == kOfxStatReplyDefault, "nested capture status");
            const auto state = s.observe();
            require(state.pending_kind == 4 && state.captured.camera_exposure_compensation_ev == 2.0, "stale getter replaced newer capture");
            s.destroy();
            s.unload();
            Session failed(plugin);
            failed.create();
            require(failed.change("Exposure") == kOfxStatReplyDefault, "stale host failure initial capture");
            OfxStatus newer = kOfxStatFailed;
            failed.host.fail_next("param_get/Exposure", kOfxStatErrBadHandle);
            failed.host.reenter_next("param_get/Exposure", [&] {
                failed.host.set_value(failed.effect, "Exposure", {3.0});
                newer = failed.change("Exposure");
            });
            require(failed.change("Exposure") == kOfxStatErrBadHandle && newer == kOfxStatReplyDefault, "stale getter failure status was suppressed");
            const auto newer_state = failed.observe();
            require(newer_state.pending_kind == 4 && newer_state.captured.camera_exposure_compensation_ev == 3.0, "stale getter failure disturbed newer state");
            failed.destroy();
            failed.unload();
            row.passed();
        }
        for (const auto* id : {"deferred_write", "refresh_after_write"}) {
            auto row = rows.case_event(id);
            Session s(plugin);
            s.create();
            require(s.change("Exposure") == kOfxStatReplyDefault, "initial capture");
            s.host.set_value(s.effect, "CameraFilmFormatPreset", {2});
            OfxStatus nested = kOfxStatFailed;
            uint32_t pending_during_write = 0;
            uint64_t outstanding_during_write = 0;
            s.host.reenter_next("param_set/CameraFilmFormatMm", [&] {
                nested = s.change("Exposure", kOfxChangeTime);
                const auto deferred = s.observe();
                pending_during_write = deferred.pending_kind;
                outstanding_during_write = deferred.outstanding_writes;
            });
            require(s.change("CameraFilmFormatPreset") == kOfxStatReplyDefault && nested == kOfxStatReplyDefault, "deferred write status");
            require(pending_during_write == 2 && outstanding_during_write == 1, "nested capture did not wait for the registered write");
            const auto state = s.observe();
            require(state.pending_kind == 4 && state.outstanding_writes == 0 && state.captured.camera_film_format_long_edge_mm == 36.0f, "refresh did not capture actual host value");
            if (std::string_view(id) == "refresh_after_write") {
                s.host.set_value(s.effect, "Exposure", {1.5});
                OfxStatus worker_status = kOfxStatFailed;
                std::thread worker([&] {
                    worker_status = s.change("Exposure", kOfxChangeTime);
                });
                worker.join();
                const auto worker_state = s.observe();
                require(worker_status == kOfxStatReplyDefault && worker_state.pending_kind == 4 && worker_state.captured.camera_exposure_compensation_ev == 1.5, "sequential worker callback capture");
            }
            s.destroy();
            s.unload();
            row.passed();
        }
        {
            auto row = rows.case_event("partial_write");
            Session s(plugin);
            s.create();
            require(s.change("Exposure") == kOfxStatReplyDefault, "initial capture");
            s.host.set_value(s.effect, "CameraFilmFormatPreset", {2});
            s.host.store_then_fail_next("param_set/CameraFilmFormatMm", kOfxStatFailed);
            OfxStatus nested = kOfxStatFailed;
            s.host.reenter_next("param_set/CameraFilmFormatMm", [&] {
                nested = s.change("Exposure", kOfxChangeTime);
            });
            require(s.change("CameraFilmFormatPreset") == kOfxStatFailed && nested == kOfxStatReplyDefault, "late physical setter failure status");
            require(std::get<double>(s.host.value(s.effect, "CameraFilmFormatMm").at(0)) == 36.0, "setter did not physically write");
            require(s.observe().pending_kind == 5, "partial write did not invalidate pending");
            s.destroy();
            s.unload();
            row.passed();
        }
        test.passed();
    }
} // namespace

int main(int argc, char** argv) try {
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto* id : kCases)
            JuicerTests::list_case(kCaller, id);
        return 0;
    }
    JuicerTests::CaseLog log;
    auto* plugin = fj_ofx_plugin(0);
    require(plugin && plugin->mainEntry && plugin->setHost, "private callbacks missing");
    allocation(plugin);
    panic(plugin);
    rollback(plugin);
    terminal(plugin);
    capture_races(plugin);
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "instance control fixture: %s\n", error.what());
    return 1;
}
