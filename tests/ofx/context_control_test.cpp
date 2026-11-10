#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "context_host.h"
#include "module_test.h"
#include "../ffi/case_results.h"
#include "../ffi/juicer_test_api.h"

namespace {
    constexpr std::string_view kCaller = "ofx.context.controls";

    void require(bool condition, const char* detail) {
        if (!condition)
            throw std::runtime_error(detail);
    }

    OfxStatus action(OfxPlugin* plugin, const char* name) {
        return plugin->mainEntry(name, nullptr, nullptr, nullptr);
    }

    OfxStatus context(OfxPlugin* plugin, ofx_test::ContextHost& host) {
        return plugin->mainEntry(kOfxImageEffectActionDescribeInContext, host.effect_handle(), host.context_handle(), nullptr);
    }

    void load(OfxPlugin* plugin, ofx_test::ContextHost& host) {
        plugin->setHost(host.binding());
        require(action(plugin, kOfxActionLoad) == kOfxStatOK, "context control load");
        host.reset_trace();
    }

    void cleanup(OfxPlugin* plugin) {
        require(action(plugin, kOfxActionUnload) == kOfxStatOK, "context control unload");
        require(!fj_test_ofx_native() && fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0,
                "context control retained owner");
    }

    void catalog_failure(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "catalog_failure");
        const JuicerTests::Cohort rows("ofx.context.controls.catalog_failure", {"readiness_prefix", "host_precedence", "terminal_cleanup"});
        ofx_test::ContextHost host;
        load(plugin, host);
        ofx_test::ContextHost baseline;
        require(context(plugin, baseline) == kOfxStatOK && !baseline.callback_error, "catalog positive control");
        {
            auto row = rows.case_event("readiness_prefix");
            fj_test_ofx_context_fault(1);
            require(context(plugin, host) == kOfxStatErrFatal, "catalog failure status");
            constexpr std::array names{"Source", "Output", "InputGroup", "Exposure", "CameraMeteringMethod", "CameraAutoExposure", "CameraFilmFormatPreset", "CameraFilmFormatMm"};
            require(host.definitions.size() == names.size(), "catalog failure definition count");
            for (std::size_t i = 0; i < names.size(); ++i)
                require(host.definitions[i].name == names[i], "catalog failure definition prefix");
            require(host.calls.size() < baseline.calls.size() &&
                        std::equal(host.calls.begin(), host.calls.end(), baseline.calls.begin()) &&
                        baseline.calls[host.calls.size()].operation == "param_define" &&
                        baseline.calls[host.calls.size()].property == "FilmProfileKey" && !host.callback_error,
                    "catalog readiness call prefix");
            row.passed();
        }
        {
            auto row = rows.case_event("host_precedence");
            ofx_test::ContextHost failed;
            failed.statuses["param_define/InputGroup"] = kOfxStatErrMemory;
            fj_test_ofx_context_fault(1);
            require(context(plugin, failed) == kOfxStatErrMemory, "earlier host failure precedence");
            require(failed.definitions.size() == 3 && !failed.calls.empty() &&
                        failed.calls.back().operation == "param_define" && failed.calls.back().property == "InputGroup" &&
                        failed.calls.size() <= baseline.calls.size() &&
                        std::equal(failed.calls.begin(), failed.calls.end(), baseline.calls.begin()) && !failed.callback_error,
                    "host failure did not stop at the expected prefix");
            row.passed();
        }
        {
            auto row = rows.case_event("terminal_cleanup");
            cleanup(plugin);
            row.passed();
        }
        test.passed();
    }

    void allocation(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "allocation");
        const JuicerTests::Cohort rows("ofx.context.controls.allocation", {"before_writes", "owned_string", "one_shot_recovery"});
        ofx_test::ContextHost host;
        load(plugin, host);
        ofx_test::ContextHost baseline;
        require(context(plugin, baseline) == kOfxStatOK && !baseline.callback_error, "allocation positive control");
        {
            auto row = rows.case_event("before_writes");
            fj_test_ofx_context_fault(2);
            require(context(plugin, host) == kOfxStatErrMemory, "early allocation status");
            require(host.definitions.empty() && !host.callback_error && host.calls.size() < baseline.calls.size() &&
                        std::equal(host.calls.begin(), host.calls.end(), baseline.calls.begin()) &&
                        baseline.calls[host.calls.size()].operation == "clip_define" &&
                        std::none_of(host.calls.begin(), host.calls.end(), [](const auto& call) {
                            return call.operation == "clip_define" || call.operation == "param_define" ||
                                   call.operation == "string" || call.operation == "int" || call.operation == "double";
                        }),
                    "early allocation made writes");
            row.passed();
        }
        {
            auto row = rows.case_event("owned_string");
            ofx_test::ContextHost failed;
            fj_test_ofx_context_fault(3);
            require(context(plugin, failed) == kOfxStatErrMemory, "catalog string allocation status");
            constexpr std::array names{"Source", "Output", "InputGroup", "Exposure", "CameraMeteringMethod", "CameraAutoExposure", "CameraFilmFormatPreset", "CameraFilmFormatMm", "FilmProfileKey"};
            require(failed.definitions.size() == names.size(), "catalog string allocation prefix count");
            for (std::size_t i = 0; i < names.size(); ++i)
                require(failed.definitions[i].name == names[i], "catalog string allocation prefix");
            require(!failed.calls.empty() && failed.calls.size() < baseline.calls.size() &&
                        std::equal(failed.calls.begin(), failed.calls.end(), baseline.calls.begin()) &&
                        failed.calls.back().operation == "dimension" &&
                        failed.calls.back().property == "OfxParamPropChoiceEnum" &&
                        baseline.calls[failed.calls.size()].operation == "string" &&
                        baseline.calls[failed.calls.size()].property == "OfxParamPropChoiceEnum" && !failed.callback_error,
                    "catalog string allocation call prefix");
            row.passed();
        }
        {
            auto row = rows.case_event("one_shot_recovery");
            ofx_test::ContextHost recovered;
            require(context(plugin, recovered) == kOfxStatOK && recovered.definitions.size() == 130 && !recovered.callback_error,
                    "allocation fault did not clear");
            row.passed();
        }
        cleanup(plugin);
        test.passed();
    }

    void panic(OfxPlugin* plugin) {
        JuicerTests::CaseEvent test(kCaller, "panic");
        const JuicerTests::Cohort rows("ofx.context.controls.panic", {"admitted_invalidation", "terminal_cleanup", "reload"});
        ofx_test::ContextHost host;
        load(plugin, host);
        ofx_test::ContextHost baseline;
        require(context(plugin, baseline) == kOfxStatOK && !baseline.callback_error, "panic positive control");
        {
            auto row = rows.case_event("admitted_invalidation");
            auto* native = fj_test_ofx_native();
            const auto* assets = fj_test_ofx_assets();
            OfxStatus result = kOfxStatOK;
            std::thread worker([&] {
                fj_test_ofx_arm_pause(2 | 16);
                fj_test_ofx_context_fault(4);
                result = context(plugin, host);
            });
            const bool paused = fj_test_ofx_wait_pause(2) == 1;
            const bool excluded = paused && action(plugin, kOfxActionUnload) == kOfxStatErrFatal &&
                                  fj_test_ofx_native() == native && fj_test_ofx_assets() == assets;
            const bool first_released = fj_test_ofx_release_pause() == 1;
            const bool invalidated = fj_test_ofx_wait_pause(16) == 1;
            std::vector<ofx_test::ContextCall> panic_calls;
            bool rejected = false;
            if (invalidated) {
                panic_calls = host.calls;
                host.reset_trace();
                rejected = context(plugin, host) == kOfxStatFailed && host.calls.empty() &&
                           action(plugin, kOfxActionUnload) == kOfxStatErrFatal &&
                           fj_test_ofx_native() == native && fj_test_ofx_assets() == assets;
            }
            const bool second_released = fj_test_ofx_release_pause() == 1;
            worker.join();
            require(paused && excluded && first_released && invalidated && rejected && second_released &&
                        result == kOfxStatFailed,
                    "context panic admission containment");
            require(host.definitions.size() == 2 && host.definitions[0].name == "Source" &&
                        host.definitions[1].name == "Output" && panic_calls.size() < baseline.calls.size() &&
                        std::equal(panic_calls.begin(), panic_calls.end(), baseline.calls.begin()) &&
                        baseline.calls[panic_calls.size()].operation == "param_define" &&
                        baseline.calls[panic_calls.size()].property == "InputGroup" && !host.callback_error,
                    "context panic declaration prefix");
            host.reset_trace();
            require(context(plugin, host) == kOfxStatFailed && host.calls.empty(), "context panic did not invalidate owner");
            row.passed();
        }
        {
            auto row = rows.case_event("terminal_cleanup");
            cleanup(plugin);
            row.passed();
        }
        {
            auto row = rows.case_event("reload");
            ofx_test::ContextHost fresh;
            load(plugin, fresh);
            require(context(plugin, fresh) == kOfxStatOK && !fresh.callback_error, "context panic reload");
            cleanup(plugin);
            row.passed();
        }
        test.passed();
    }
} // namespace

int main(int argc, char** argv) try {
    constexpr std::array cases{"catalog_failure", "allocation", "panic"};
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto* id : cases)
            JuicerTests::list_case(kCaller, id);
        return 0;
    }
    JuicerTests::CaseLog log;
    auto* plugin = fj_ofx_plugin(0);
    require(plugin && plugin->mainEntry && plugin->setHost, "real module callbacks missing");
    catalog_failure(plugin);
    allocation(plugin);
    panic(plugin);
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "context control fixture: %s\n", error.what());
    return 1;
}
