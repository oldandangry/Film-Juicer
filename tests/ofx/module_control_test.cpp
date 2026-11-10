#include <array>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "ProcessRoot.h"
#include "../ffi/native_path.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_owner.h"
#include "module_host.h"
#include "module_test.h"
#include "../ffi/case_results.h"
#include "../ffi/juicer_test_api.h"

namespace {
    void require(bool condition, const char* detail) {
        if (!condition) {
            throw std::runtime_error(detail);
        }
    }

    OfxStatus action(OfxPlugin* plugin, const char* name, const void* handle = nullptr) {
        return plugin->mainEntry(name, handle, nullptr, nullptr);
    }

    class PausedAction final {
    public:
        PausedAction(std::function<OfxStatus()> operation, uint32_t pauses)
            : worker_([this, operation = std::move(operation), pauses] {
                  fj_test_ofx_arm_pause(pauses);
                  result_ = operation();
              }) {
        }
        PausedAction(const PausedAction&) = delete;
        PausedAction& operator=(const PausedAction&) = delete;
        ~PausedAction() {
            fj_test_ofx_release_pause();
            if (worker_.joinable()) {
                worker_.join();
            }
        }
        void wait(uint32_t stage) const {
            require(fj_test_ofx_wait_pause(stage) == 1, "module action did not reach bounded pause");
        }
        OfxStatus finish() {
            const auto released = fj_test_ofx_release_pause();
            worker_.join();
            require(released == 1, "bounded pause expired before interleaving completed");
            return result_;
        }

    private:
        OfxStatus result_ = kOfxStatFailed;
        std::thread worker_;
    };

    void no_owner() {
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0 && !fj_test_ofx_native(),
                "module retained unpublished host ownership");
        FjCuda* probe = nullptr;
        require(fj_cuda_create(&probe, nullptr).category == FJ_STATUS_SUCCESS && probe,
                "construction rollback retained native registration");
        require(fj_cuda_destroy(probe, nullptr).category == FJ_STATUS_SUCCESS, "rollback probe close");
    }

    std::filesystem::path path_observation(uint32_t mode, const std::filesystem::path& input = {}) {
        const JuicerAssets::NativePathArgument argument(input);
        const auto length = fj_test_ofx_path(mode, argument.view(), nullptr, 0);
        require(length != std::numeric_limits<std::size_t>::max(), "path observation rejected native input");
        std::filesystem::path::string_type storage(length, 0);
        require(fj_test_ofx_path(mode, argument.view(), storage.data(), storage.size()) == length,
                "path observation changed between size and copy");
        return std::filesystem::path(storage);
    }

    void platform() {
        const JuicerTests::Cohort rows("ofx.module.platform", {"abi", "current_module", "unicode", "lexical", "relative", "no_parent", "one_parent", "lookup_failure", "native_units", "root_module", "root_directory", "immediate_dot", "repeated_separators", "trailing_separator", "parent_component"
#ifdef _WIN32
                                                               ,
                                                               "growth",
                                                               "drive_relative",
                                                               "mixed_separators",
                                                               "unc_path",
                                                               "unc_parent",
                                                               "unc_root_parent",
                                                               "device_parent",
                                                               "device_drive_parent",
                                                               "drive_relative_parent",
                                                               "separator_root",
                                                               "drive_root"
#endif
                                                              });
        {
            auto row = rows.case_event("abi");
            require(fj_test_ofx_platform_c() == 1, "native C11/platform ABI mismatch");
            row.passed();
        }
        {
            auto row = rows.case_event("current_module");
            const auto path = path_observation(0);
            require(!path.empty() && path == JuicerProcess::data_directory(), "actual containing-module path mismatch");
            row.passed();
        }
        const std::vector<std::pair<std::string_view, std::filesystem::path>> paths{
            {"unicode", std::filesystem::path(u"/tmp/å中/Contents/Linux-x86-64/Juicer.ofx")},
            {"lexical", std::filesystem::path("/tmp/a/../b/./Contents/Linux-x86-64/Juicer.ofx")},
            {"relative", std::filesystem::path("bundle/Contents/Linux-x86-64/Juicer.ofx")},
            {"no_parent", std::filesystem::path("Juicer.ofx")},
            {"one_parent", std::filesystem::path("Linux-x86-64/Juicer.ofx")},
            {"lookup_failure", std::filesystem::path{}},
            {"root_module", std::filesystem::path("/Juicer.ofx")},
            {"root_directory", std::filesystem::path("/")},
            {"immediate_dot", std::filesystem::path("/bundle/Contents/Linux-x86-64/./Juicer.ofx")},
            {"repeated_separators", std::filesystem::path("/bundle//Contents///Linux-x86-64/Juicer.ofx")},
            {"trailing_separator", std::filesystem::path("/bundle/Contents/Linux-x86-64/")},
            {"parent_component", std::filesystem::path("/bundle/Contents/Linux-x86-64/../Juicer.ofx")},
#ifdef _WIN32
            {"native_units", std::filesystem::path(std::wstring(L"C:\\tmp\\") + wchar_t(0xd800) + L"\\Contents\\Win64\\Juicer.ofx")},
            {"drive_relative", std::filesystem::path("C:Juicer.ofx")},
            {"mixed_separators", std::filesystem::path(R"(C:\bundle/Contents\Win64/./Juicer.ofx)")},
            {"unc_path", std::filesystem::path(R"(\\server\share\Contents\Win64\Juicer.ofx)")},
            {"unc_parent", std::filesystem::path(R"(\\server\share\..\Contents\Win64\Juicer.ofx)")},
            {"unc_root_parent", std::filesystem::path(R"(\\server\share\..\..\Contents\Win64\Juicer.ofx)")},
            {"device_parent", std::filesystem::path(R"(\\?\UNC\server\share\..\Contents\Win64\Juicer.ofx)")},
            {"device_drive_parent", std::filesystem::path(R"(\\?\C:\folder\..\Contents\Win64\Juicer.ofx)")},
            {"drive_relative_parent", std::filesystem::path(R"(C:folder\..\Contents\Win64\Juicer.ofx)")},
            {"separator_root", std::filesystem::path(R"(\\)")},
            {"drive_root", std::filesystem::path(R"(C:\Juicer.ofx)")}
#else
            {"native_units", std::filesystem::path(std::string("/tmp/\xff/Contents/Linux-x86-64/Juicer.ofx"))}
#endif
        };
        for (const auto& [id, input] : paths) {
            auto row = rows.case_event(id);
            std::filesystem::path expected;
            if (id != "lookup_failure" && !input.parent_path().empty() && !input.parent_path().parent_path().empty()) {
                expected = (input.parent_path().parent_path() / "Resources").lexically_normal().make_preferred();
            }
            require(path_observation(id == "lookup_failure" ? 2 : 1, input) == expected,
                    "native path units or lexical parent normalization changed");
            row.passed();
        }
#ifdef _WIN32
        {
            auto row = rows.case_event("growth");
            require(path_observation(3) == path_observation(0), "native module-name growth changed path");
            row.passed();
        }
#endif
    }

    void construction(OfxPlugin* plugin, ofx_test::Host& host, std::string_view caller) {
        {
            JuicerTests::CaseEvent test(caller, "platform");
            platform();
            no_owner();
            test.passed();
        }
        for (const auto& [id, stages] : std::array<std::pair<const char*, std::vector<uint32_t>>, 3>{{{"before_native", {1}}, {"after_native", {2}}, {"panic", {4, 5}}}}) {
            JuicerTests::CaseEvent test(caller, id);
            std::vector<std::string> ids;
            for (const auto stage : stages) {
                ids.push_back(std::to_string(stage));
            }
            if (std::string_view(id) == "panic") {
                ids.emplace_back("load_panic_replacement");
                ids.emplace_back("invalidated_rollback");
            }
            const JuicerTests::Cohort rows(std::string("ofx.module.construction.") + id, ids);
            for (const auto stage : stages) {
                auto row = rows.case_event(std::to_string(stage));
                plugin->setHost(host.binding());
                fj_test_runtime_fault(stage);
                require(action(plugin, kOfxActionLoad) == (stage < 4 ? kOfxStatErrMemory : kOfxStatFailed),
                        "construction failure status");
                no_owner();
                require(action(plugin, kOfxActionUnload) == kOfxStatOK, "unpublished unload");
                plugin->setHost(host.binding());
                require(action(plugin, kOfxActionLoad) == kOfxStatOK && fj_test_assets_live_owners() == 1,
                        "construction recovery failed");
                require(action(plugin, kOfxActionUnload) == kOfxStatOK, "construction recovery close");
                no_owner();
                row.passed();
            }
            if (std::string_view(id) == "panic") {
                {
                    auto row = rows.case_event("load_panic_replacement");
                    plugin->setHost(host.binding());
                    PausedAction failed([&] {
                        fj_test_runtime_fault(5);
                        return action(plugin, kOfxActionLoad);
                    },
                                        1);
                    failed.wait(1);
                    // The old failing callback is still live at its outer C
                    // boundary. Its own rollback is complete; a replacement may
                    // now load, and late panic cleanup must not invalidate it.
                    no_owner();
                    plugin->setHost(host.binding());
                    require(action(plugin, kOfxActionLoad) == kOfxStatOK, "replacement after construction panic failed");
                    auto* replacement = fj_test_ofx_native();
                    const auto* assets = fj_test_ofx_assets();
                    host.reset_trace();
                    require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatOK, "replacement Describe before old return");
                    require(failed.finish() == kOfxStatFailed, "construction panic boundary status");
                    host.reset_trace();
                    require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatOK &&
                                fj_test_ofx_native() == replacement && fj_test_ofx_assets() == assets,
                            "old construction panic invalidated replacement");
                    require(action(plugin, kOfxActionUnload) == kOfxStatOK, "panic replacement close");
                    no_owner();
                    row.passed();
                }
                {
                    auto row = rows.case_event("invalidated_rollback");
                    plugin->setHost(host.binding());
                    PausedAction invalidated([&] {
                        return action(plugin, kOfxActionLoad);
                    },
                                             4 | 8);
                    invalidated.wait(4);
                    plugin->setHost(host.binding());
                    fj_test_ofx_release_pause();
                    invalidated.wait(8);
                    host.reset_trace();
                    require(fj_test_assets_live_owners() == 1, "unpublished Runtime not live during rollback pause");
                    FjCuda* rejected = nullptr;
                    require(fj_cuda_create(&rejected, nullptr).category == FJ_STATUS_PREPARATION_FAILURE && !rejected,
                            "rollback pause did not retain native registration");
                    require(action(plugin, kOfxActionLoad) == kOfxStatFailed && host.calls.empty(),
                            "rollback admitted suite fetch or second construction");
                    require(action(plugin, kOfxActionUnload) == kOfxStatErrFatal,
                            "rollback reported terminal completion while native owner remained");
                    require(invalidated.finish() == kOfxStatFailed, "invalidated Load returned success");
                    no_owner();
                    plugin->setHost(host.binding());
                    require(action(plugin, kOfxActionLoad) == kOfxStatOK, "load after complete rollback failed");
                    require(action(plugin, kOfxActionUnload) == kOfxStatOK, "rollback replacement close");
                    no_owner();
                    row.passed();
                }
            }
            test.passed();
        }
    }

    void panic_controls(OfxPlugin* plugin, ofx_test::Host& host, std::string_view caller) {
        for (const auto& [id, stage] : std::array<std::pair<const char*, uint32_t>, 2>{{{"action", 1}, {"void", 2}}}) {
            JuicerTests::CaseEvent test(caller, id);
            const std::vector<std::string> ids = stage == 1
                                                     ? std::vector<std::string>{"serial", "admitted_panic", "describe_replacement"}
                                                     : std::vector<std::string>{"serial"};
            const JuicerTests::Cohort rows(std::string("ofx.module.panic.") + id, ids);
            {
                auto row = rows.case_event("serial");
                plugin->setHost(host.binding());
                require(action(plugin, kOfxActionLoad) == kOfxStatOK, "panic control load");
                const auto* original = fj_test_ofx_native();
                fj_test_ofx_panic(stage);
                if (stage == 1) {
                    require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatFailed, "action panic escaped");
                } else {
                    plugin->setHost(host.binding());
                }
                require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatFailed,
                        "panic admitted ordinary callbacks");
                require(fj_test_ofx_native() == original && fj_test_assets_live_owners() == 1,
                        "panic lost terminal cleanup owner");
                require(action(plugin, kOfxActionUnload) == kOfxStatOK, "panic cleanup failed");
                no_owner();
                plugin->setHost(host.binding());
                require(action(plugin, kOfxActionLoad) == kOfxStatOK, "panic recovery load");
                require(action(plugin, kOfxActionUnload) == kOfxStatOK, "panic recovery close");
                row.passed();
            }
            if (stage == 1) {
                {
                    auto row = rows.case_event("admitted_panic");
                    plugin->setHost(host.binding());
                    require(action(plugin, kOfxActionLoad) == kOfxStatOK, "admitted panic load");
                    auto* original = fj_test_ofx_native();
                    const auto* assets = fj_test_ofx_assets();
                    PausedAction failed([&] {
                        fj_test_ofx_panic(1);
                        return action(plugin, kOfxActionDescribe, host.effect_handle());
                    },
                                        2);
                    failed.wait(2);
                    require(action(plugin, kOfxActionUnload) == kOfxStatErrFatal, "panic released callback exclusion before cleanup");
                    require(action(plugin, kOfxActionLoad) == kOfxStatFailed &&
                                fj_test_ofx_native() == original && fj_test_ofx_assets() == assets,
                            "admitted panic lost matching owner");
                    require(failed.finish() == kOfxStatFailed, "admitted panic boundary status");
                    host.reset_trace();
                    require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatFailed && host.calls.empty(),
                            "panic did not invalidate its own module");
                    require(action(plugin, kOfxActionUnload) == kOfxStatOK, "admitted panic terminal cleanup");
                    no_owner();
                    row.passed();
                }
                {
                    auto row = rows.case_event("describe_replacement");
                    plugin->setHost(host.binding());
                    require(action(plugin, kOfxActionLoad) == kOfxStatOK, "old Describe load");
                    PausedAction failed([&] {
                        fj_test_ofx_panic(1);
                        return action(plugin, kOfxActionDescribe, host.effect_handle());
                    },
                                        1);
                    failed.wait(1);
                    require(action(plugin, kOfxActionUnload) == kOfxStatOK, "settled Describe panic close");
                    no_owner();
                    plugin->setHost(host.binding());
                    require(action(plugin, kOfxActionLoad) == kOfxStatOK, "replacement during old Describe return");
                    auto* replacement = fj_test_ofx_native();
                    const auto* assets = fj_test_ofx_assets();
                    require(failed.finish() == kOfxStatFailed, "old Describe panic boundary status");
                    host.reset_trace();
                    require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatOK &&
                                fj_test_ofx_native() == replacement && fj_test_ofx_assets() == assets,
                            "old Describe panic invalidated later lifecycle");
                    require(action(plugin, kOfxActionUnload) == kOfxStatOK, "Describe replacement close");
                    no_owner();
                    row.passed();
                }
            }
            test.passed();
        }
    }

    void terminal(OfxPlugin* plugin, ofx_test::Host& host, std::string_view caller) {
        plugin->setHost(host.binding());
        require(action(plugin, kOfxActionLoad) == kOfxStatOK, "terminal load");
        auto* native = fj_test_ofx_native();
        const auto* assets = fj_test_ofx_assets();
        require(native && assets, "terminal owner backing missing");
        {
            JuicerTests::CaseEvent test(caller, "active_borrow");
            const JuicerTests::Cohort rows("ofx.module.terminal.precheck", {"callback", "native_call"});
            {
                auto row = rows.case_event("callback");
                host.reset_trace();
                host.reentry = [&] {
                    require(action(plugin, kOfxActionUnload) == kOfxStatErrFatal, "active callback close accepted");
                    require(fj_test_ofx_native() == native && fj_test_ofx_assets() == assets && fj_test_assets_live_owners() == 1, "active callback backing expired");
                };
                require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatOK && !host.callback_error,
                        "reentrant host callback lost admission");
                row.passed();
            }
            {
                auto row = rows.case_event("native_call");
                {
                    JuicerCuda::NativeCall call(native);
                    require(action(plugin, kOfxActionUnload) == kOfxStatErrFatal, "native precheck rejection not fatal");
                    require(fj_test_ofx_native() == native && fj_test_ofx_assets() == assets && fj_test_assets_live_owners() == 1, "precheck did not restore same owner");
                }
                host.reset_trace();
                require(action(plugin, kOfxActionDescribe, host.effect_handle()) == kOfxStatOK, "rejected precheck poisoned live owner");
                row.passed();
            }
            test.passed();
        }
        static int context_token;
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, &context_token};
        JuicerCuda::ResourceManager::RegistryContextSnapshot snapshot{};
        {
            JuicerTests::CaseEvent test(caller, "close_failure");
            require(JuicerCuda::ResourceManager::registry_begin_submission(key, snapshot), "metadata submission setup");
            require(action(plugin, kOfxActionUnload) == kOfxStatErrFatal, "consuming native failure not fatal");
            require(!fj_test_ofx_native() && fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0,
                    "consuming failure retained Rust host owner");
            require(fj_cuda_check_terminal(native, nullptr).category == FJ_STATUS_SUCCESS,
                    "consuming failure lost authoritative native retention");
            require(!JuicerCuda::borrow_root(native).begin_frame_preparation().active(), "retention reopened admission");
            require(JuicerCuda::ResourceManager::registry_note_submission_end(key), "metadata submission end");
            test.passed();
        }
        {
            JuicerTests::CaseEvent test(caller, "consumed");
            require(action(plugin, kOfxActionUnload) == kOfxStatOK, "consumed unload retried owner");
            require(JuicerCuda::ResourceManager::registry_begin_owner_retire(key, snapshot), "consumed unload retried native retirement");
            test.passed();
        }
        {
            JuicerTests::CaseEvent test(caller, "replacement_blocked");
            FjCuda* replacement = nullptr;
            require(fj_cuda_create(&replacement, nullptr).category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
                    "retained native graph permitted replacement");
            plugin->setHost(host.binding());
            require(action(plugin, kOfxActionLoad) == kOfxStatFailed && !fj_test_ofx_native() && fj_test_assets_live_owners() == 0,
                    "failed native replacement published candidate Runtime");
            require(action(plugin, kOfxActionUnload) == kOfxStatOK, "failed replacement cleanup");
            test.passed();
        }
    }
} // namespace

int main(int argc, char** argv) try {
    require(argc >= 2, "expected construction/panic/terminal mode");
    const std::string_view mode(argv[1]);
    const std::string caller = std::string("ofx.module.") + std::string(mode);
    const std::vector<std::string_view> cases = mode == "construction" ? std::vector<std::string_view>{"platform", "before_native", "after_native", "panic"}
                                                : mode == "panic"      ? std::vector<std::string_view>{"action", "void"}
                                                : mode == "terminal"   ? std::vector<std::string_view>{"active_borrow", "close_failure", "consumed", "replacement_blocked"}
                                                                       : throw std::runtime_error("unknown module control mode");
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto id : cases) {
            JuicerTests::list_case(caller, id);
        }
        return 0;
    }
    JuicerTests::CaseLog log;
    auto* plugin = fj_ofx_plugin(0);
    require(plugin && plugin->mainEntry && plugin->setHost, "real module callbacks missing");
    ofx_test::Host host;
    if (mode == "construction") {
        construction(plugin, host, caller);
    } else if (mode == "panic") {
        panic_controls(plugin, host, caller);
    } else {
        terminal(plugin, host, caller);
    }
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "module control fixture: %s\n", error.what());
    return 1;
}
