#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "instance_host.h"
#include "juicer_ofx_api.h"
#include "../ffi/juicer_test_api.h"
#include "module_test.h"
#include "ofxsImageEffect.h"
#include "../ffi/case_results.h"

extern "C" FjStatus fj_test_ofx_instance_admit(OfxImageEffectHandle effect, FjPreparedState** out, FjErrorBuffer* error);

namespace {
    using ofx_test::InstanceHost;
    constexpr std::string_view kCaller = "ofx.render_controls";
    constexpr std::array kCases{"non_cuda", "faults", "admission"};
    struct Fixture {
        InstanceHost host;
        OfxPlugin* plugin = nullptr;
        OfxImageEffectHandle effect = nullptr;
    };
    void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }
    OfxStatus action(OfxPlugin* plugin, const char* name, const void* effect = nullptr, OfxPropertySetHandle input = nullptr) {
        return plugin->mainEntry(name, effect, input, nullptr);
    }
    void start(Fixture& fixture, OfxPlugin* plugin, double time, bool cuda, bool source_available = true, std::string_view film = "kodak_portra_400") {
        fixture.plugin = plugin;
        plugin->setHost(fixture.host.binding());
        require(action(plugin, kOfxActionLoad) == kOfxStatOK, "test-support plugin load");
        require(action(plugin, kOfxImageEffectActionDescribeInContext, fixture.host.descriptor_handle(), fixture.host.context_args()) == kOfxStatOK, "describe filter context");
        fixture.effect = fixture.host.create_effect();
        require(action(plugin, kOfxActionCreateInstance, fixture.effect) == kOfxStatOK, "create instance");
        fixture.host.set_value(fixture.effect, "FilmProfileKey", {std::string(film)});
        fixture.host.set_value(fixture.effect, "PrintProfileKey", {std::string("kodak_portra_endura")});
        fixture.host.set_effect_property(fixture.effect, kOfxImageEffectPropFrameRate, {24.0});
        fixture.host.set_clip_property(fixture.effect, "Source", kOfxImageEffectPropFrameRate, {24.0});
        fixture.host.set_clip_property(fixture.effect, "Output", kOfxImageEffectPropFrameRate, {24.0});
        for (const auto* clip : {"Source", "Output"}) {
            fixture.host.set_clip_property(fixture.effect, clip, kOfxImageEffectPropPixelDepth, {std::string(kOfxBitDepthFloat)});
            fixture.host.set_clip_property(fixture.effect, clip, kOfxImageEffectPropSupportedComponents, {std::string(kOfxImageComponentRGBA)});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImagePropData, {static_cast<void*>(nullptr)});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImagePropBounds, {0, 0, 4, 3});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImagePropRowBytes, {64});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImageEffectPropPixelDepth, {std::string(kOfxBitDepthFloat)});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImageEffectPropComponents, {std::string(kOfxImageComponentRGBA)});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImageEffectPropPreMultiplication, {std::string("OfxImageAlphaUnPremultiplied")});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImagePropField, {std::string(kOfxImageFieldNone)});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImagePropUniqueIdentifier, {std::string(clip) + "-image"});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImagePropRegionOfDefinition, {0, 0, 4, 3});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImageEffectPropRenderScale, {1.0, 1.0});
            fixture.host.set_image_property(fixture.effect, clip, kOfxImagePropPixelAspectRatio, {1.0});
            fixture.host.set_clip_region(fixture.effect, clip, {0.0, 0.0, 4.0, 3.0});
        }
        fixture.host.set_image_available(fixture.effect, "Source", source_available);
        fixture.host.set_property(fixture.host.action_args(), kOfxPropTime, {time});
        fixture.host.set_property(fixture.host.action_args(), kOfxImageEffectPropRenderScale, {1.0, 1.0});
        fixture.host.set_property(fixture.host.action_args(), kOfxImageEffectPropRenderWindow, {0, 0, 4, 3});
        fixture.host.set_property(fixture.host.action_args(), kOfxImageEffectPropOpenCLEnabled, {0});
        fixture.host.set_property(fixture.host.action_args(), kOfxImageEffectPropCudaEnabled, {cuda ? 1 : 0});
        fixture.host.set_property(fixture.host.action_args(), kOfxImageEffectPropMetalEnabled, {0});
        fixture.host.set_property(fixture.host.action_args(), kOfxImageEffectPropFieldToRender, {std::string(kOfxImageFieldNone)});
        fixture.host.set_property(fixture.host.action_args(), kOfxImageEffectPropCudaStream, {});
        fixture.host.clear_calls();
    }
    FjOfxRenderObservation observe(Fixture& fixture) {
        FjOfxRenderObservation observation{};
        require(fj_test_ofx_render_observe(fixture.effect, &observation) != 0, "render observation available");
        return observation;
    }
    void finish(Fixture& fixture) {
        require(action(fixture.plugin, kOfxActionDestroyInstance, fixture.effect) == kOfxStatOK, "destroy instance after render control");
        require(action(fixture.plugin, kOfxActionUnload) == kOfxStatOK, "unload after render control");
        require(!fixture.host.callback_error(), "host callback containment");
    }
    double time_from_bits(std::uint64_t bits) {
        return std::bit_cast<double>(bits);
    }
    void non_cuda(OfxPlugin* plugin) {
        const JuicerTests::Cohort rows("ofx.render_controls.non_cuda", {"film_copy_allocation", "print_copy_allocation", "diagnostic_allocation"});
        const struct Case {
            const char* id;
            std::uint32_t control;
            std::string_view failing_profile;
        } cases[]{
            {"film_copy_allocation", 1, "kodak_portra_400"},
            {"print_copy_allocation", 2, "kodak_portra_400"},
            {"diagnostic_allocation", 3, "kodak_portra_400"}};
        for (const auto& test : cases) {
            auto row = rows.case_event(test.id);
            Fixture fixture;
            start(fixture, plugin, 2.0, false, true, test.failing_profile);
            fj_test_ofx_render_control(test.control);
            const auto status = action(plugin, kOfxImageEffectActionRender, fixture.effect, fixture.host.action_args());
            require(status == kOfxStatErrMemory, std::string(test.id) + ": recoverable allocation maps to Memory");
            require(fixture.host.image_release_count(fixture.effect, "Source") == 0 &&
                        fixture.host.image_release_count(fixture.effect, "Output") == 0,
                    std::string(test.id) + ": non-CUDA allocation preceded image acquisition");
            require(std::none_of(fixture.host.calls().begin(), fixture.host.calls().end(), [](const auto& call) {
                        return call.operation == "clip_get_image";
                    }),
                    std::string(test.id) + ": non-CUDA allocation reached images");
            row.passed();
            finish(fixture);
        }
    }
    void faults(OfxPlugin* plugin) {
        const JuicerTests::Cohort rows("ofx.render_controls.faults", {"action_panic", "release_panic"});
        {
            auto row = rows.case_event("action_panic");
            Fixture fixture;
            start(fixture, plugin, 1.0, true);
            fj_test_ofx_render_control(4);
            const auto status = action(plugin, kOfxImageEffectActionRender, fixture.effect, fixture.host.action_args());
            require(status == kOfxStatErrFatal, "render action panic is contained as Fatal");
            const auto observation = observe(fixture);
            require(observation.native_entry_count == 0, "action panic preceded native entry");
            require(fixture.host.image_release_count(fixture.effect, "Source") == 0 &&
                        fixture.host.image_release_count(fixture.effect, "Output") == 0,
                    "action panic preceded image acquisition");
            row.passed();
            finish(fixture);
        }
        {
            auto row = rows.case_event("release_panic");
            Fixture fixture;
            start(fixture, plugin, 1.0, true, false);
            fj_test_ofx_render_control(6);
            const auto status = action(plugin, kOfxImageEffectActionRender, fixture.effect, fixture.host.action_args());
            require(status == kOfxStatOK, "cleanup panic preserves missing-image success");
            require(fixture.host.image_release_count(fixture.effect, "Source") == 0 &&
                        fixture.host.image_release_count(fixture.effect, "Output") == 1,
                    "cleanup panic follows exactly one actual Output release");
            const auto observation = observe(fixture);
            require(observation.native_entry_count == 0, "missing image stopped before native entry");
            require(action(plugin, kOfxActionBeginInstanceEdit, fixture.effect) == kOfxStatFailed,
                    "cleanup panic invalidated future ordinary admission");
            row.passed();
            finish(fixture);
        }
    }
    void admission(OfxPlugin* plugin) {
        const JuicerTests::Cohort rows("ofx.render_controls.admission", {"invalid_current", "first_capture_reentry", "release_once", "no_native_entry", "no_submission_mutation"});
        {
            auto row = rows.case_event("invalid_current");
            Fixture fixture;
            start(fixture, plugin, time_from_bits(UINT64_C(0x43e0000000000000)), true);
            require(action(plugin, kOfxImageEffectActionRender, fixture.effect, fixture.host.action_args()) == kOfxStatErrFatal,
                    "first safe host render publishes complete state before out-of-domain rejection");
            require(fixture.host.image_release_count(fixture.effect, "Source") == 1 &&
                        fixture.host.image_release_count(fixture.effect, "Output") == 1,
                    "publication-seeding render released each image once");
            FjPreparedState* published = nullptr;
            require(fj_test_ofx_instance_admit(fixture.effect, &published, nullptr).category == FJ_STATUS_SUCCESS && published,
                    "admit the older complete publication before invalidating current input");
            require(fj_test_prepared_state_release(published, nullptr).category == FJ_STATUS_SUCCESS,
                    "release fixture's admitted owner while publication remains");
            fixture.host.set_value(fixture.effect, "FilmProfileKey", {std::string("missing-profile")});
            auto* change = fixture.host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "FilmProfileKey"});
            require(action(plugin, kOfxActionInstanceChanged, fixture.effect, change) == kOfxStatReplyDefault,
                    "invalid profile event marks current controls invalid");
            const auto status = action(plugin, kOfxImageEffectActionRender, fixture.effect, fixture.host.action_args());
            require(status == kOfxStatErrFatal, "invalid current profile blocks an older complete publication");
            require(fixture.host.image_release_count(fixture.effect, "Source") == 2 &&
                        fixture.host.image_release_count(fixture.effect, "Output") == 2,
                    "failed current admission releases both images");
            const auto observation = observe(fixture);
            require(observation.native_entry_count == 0, "failed current admission preceded native entry");
            row.passed();
            finish(fixture);
        }
        {
            auto row = rows.case_event("first_capture_reentry");
            Fixture fixture;
            start(fixture, plugin, time_from_bits(UINT64_C(0x43e0000000000000)), true);
            OfxStatus nested = kOfxStatOK;
            fixture.host.reenter_next("param_get/Exposure", [&] {
                nested = action(plugin, kOfxImageEffectActionRender, fixture.effect, fixture.host.action_args());
            });
            const auto status = action(plugin, kOfxImageEffectActionRender, fixture.effect, fixture.host.action_args());
            require(status == kOfxStatErrFatal, "out-of-domain first render rejected after capture");
            require(nested == kOfxStatFailed, "overlapping first capture rejected without reentry deadlock");
            require(fixture.host.image_release_count(fixture.effect, "Source") == 2 &&
                        fixture.host.image_release_count(fixture.effect, "Output") == 2,
                    "outer and reentrant renders release their images once");
            const auto observation = observe(fixture);
            require(observation.native_entry_count == 0, "time rejection left native entry count unchanged");
            row.passed();
            finish(fixture);
        }
        const struct Case {
            const char* id;
        } cases[]{{"release_once"}, {"no_native_entry"}, {"no_submission_mutation"}};
        for (const auto& test : cases) {
            auto row = rows.case_event(test.id);
            Fixture fixture;
            start(fixture, plugin, time_from_bits(UINT64_C(0x43e0000000000000)), true);
            const auto status = action(plugin, kOfxImageEffectActionRender, fixture.effect, fixture.host.action_args());
            require(status == kOfxStatErrFatal, std::string(test.id) + ": out-of-domain frame position rejected");
            const auto observation = observe(fixture);
            require(std::any_of(fixture.host.calls().begin(), fixture.host.calls().end(), [](const auto& call) {
                        return call.operation == "get_double" && call.property == kOfxImageEffectPropFrameRate;
                    }),
                    std::string(test.id) + ": guard follows effect frame-rate access");
            if (std::string_view(test.id) == "release_once")
                require(fixture.host.image_release_count(fixture.effect, "Source") == 1 &&
                            fixture.host.image_release_count(fixture.effect, "Output") == 1,
                        "time rejection releases both images exactly once");
            if (std::string_view(test.id) == "no_native_entry")
                require(observation.native_entry_count == 0, "time rejection precedes native entry");
            if (std::string_view(test.id) == "no_submission_mutation")
                require(observation.submission_id == 0 && observation.frame_token == 0,
                        "time rejection does not select or publish a submission token");
            row.passed();
            finish(fixture);
        }
    }
} // namespace

int main(int argc, char** argv) try {
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto* id : kCases)
            JuicerTests::list_case(kCaller, id);
        return 0;
    }
    require(argc == 1, "render control fixture takes no arguments");
    JuicerTests::CaseLog log;
    auto* plugin = fj_ofx_plugin(0);
    require(plugin != nullptr, "Rust OFX callback available");
    {
        JuicerTests::CaseEvent event(kCaller, "non_cuda");
        non_cuda(plugin);
        event.passed();
    }
    {
        JuicerTests::CaseEvent event(kCaller, "faults");
        faults(plugin);
        event.passed();
    }
    {
        JuicerTests::CaseEvent event(kCaller, "admission");
        admission(plugin);
        event.passed();
    }
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "render control fixture: %s\n", error.what());
    return 1;
}
