#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "instance_host.h"
#include "host_reference.h"

#include "juicer_ofx_api.h"
#include "../ffi/case_results.h"

namespace {
    using ofx_test::InstanceCall;
    using ofx_test::InstanceHost;
    using ofx_test::InstanceValue;
    constexpr std::string_view kCaller = "ofx.render";
    constexpr std::array kCases{"non_cuda", "images", "formats", "time"};
    constexpr auto kFallbackFilm = "kodak_portra_400";
    constexpr auto kFallbackPrint = "kodak_portra_endura";
    struct Result {
        OfxStatus status = kOfxStatOK;
        std::vector<InstanceCall> calls;
        std::size_t source_releases = 0;
        std::size_t output_releases = 0;
    };
    using ofx_test::reference::Json;
    Json reference_rows = Json::array();
    std::size_t reference_cursor = 0;
    std::string reference_group;
    Json encode(const Result& result) {
        return Json::array({result.status, ofx_test::reference::calls(result.calls), result.source_releases, result.output_releases});
    }
    Result decode(const Json& item) {
        if (!item.is_array() || item.size() != 4)
            throw std::runtime_error("render reference shape");
        Result result{item[0].get<int>(), ofx_test::reference::instance_calls(item[1]), item[2].get<std::size_t>(), item[3].get<std::size_t>()};
        if (encode(result) != item)
            throw std::runtime_error("render reference encoding");
        return result;
    }
    struct Fault {
        std::string_view operation;
        OfxStatus status = kOfxStatOK;
        int occurrence = 1;
    };
    struct NullGetter {
        std::string_view property;
        int occurrence = 1;
    };
    struct ImageSpec {
        std::string_view components = kOfxImageComponentRGBA;
        std::string_view depth = kOfxBitDepthFloat;
        OfxRectI bounds{0, 0, 4, 3};
    };
    struct RenderInput {
        std::string_view film = "film-key";
        std::string_view print = "print-key";
        double time = 0.0;
        bool cuda = false;
        OfxRectI window{0, 0, 4, 3};
        bool stream_present = false;
        ImageSpec source{};
        bool source_available = true;
        bool output_available = true;
        OfxStatus source_status = kOfxStatOK;
        OfxStatus output_status = kOfxStatOK;
        OfxStatus output_release_status = kOfxStatOK;
        std::string_view missing_action_property = "";
        Fault fault{};
        NullGetter null_getter{};
    };
    struct ExpectedProfileReads {
        std::string_view film;
        std::string_view print;
        std::size_t count;
        OfxStatus status;
    };
    void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }
    OfxStatus action(OfxPlugin* plugin, const char* name, const void* effect = nullptr, OfxPropertySetHandle input = nullptr, OfxPropertySetHandle output = nullptr) {
        return plugin->mainEntry(name, effect, input, output);
    }
    void set_render_args(InstanceHost& host, double time = 0.0, bool cuda = false, OfxRectI window = {0, 0, 4, 3}, bool stream_present = false) {
        auto* args = host.action_args();
        host.set_property(args, kOfxPropTime, {time});
        host.set_property(args, kOfxImageEffectPropRenderScale, {1.0, 1.0});
        host.set_property(args, kOfxImageEffectPropRenderWindow, {window.x1, window.y1, window.x2, window.y2});
        host.set_property(args, kOfxImageEffectPropOpenCLEnabled, {0});
        host.set_property(args, kOfxImageEffectPropCudaEnabled, {cuda ? 1 : 0});
        host.set_property(args, kOfxImageEffectPropMetalEnabled, {0});
        host.set_property(args, kOfxImageEffectPropOpenCLCommandQueue, {static_cast<void*>(nullptr)});
        host.set_property(args, kOfxImageEffectPropMetalCommandQueue, {static_cast<void*>(nullptr)});
        host.set_property(args, kOfxImageEffectPropSequentialRenderStatus, {0});
        host.set_property(args, kOfxImageEffectPropInteractiveRenderStatus, {0});
        host.set_property(args, kOfxImageEffectPropRenderQualityDraft, {0});
        host.set_property(args, kOfxImageEffectPropFieldToRender, {std::string(kOfxImageFieldNone)});
        if (stream_present)
            host.set_property(args, kOfxImageEffectPropCudaStream, {static_cast<void*>(nullptr)});
        else
            host.set_property(args, kOfxImageEffectPropCudaStream, {});
    }
    void set_image(InstanceHost& host, OfxImageEffectHandle effect, std::string_view clip, const ImageSpec& image) {
        host.set_clip_property(effect, clip, kOfxImageEffectPropPixelDepth, {std::string(image.depth)});
        host.set_clip_property(effect, clip, kOfxImageEffectPropSupportedComponents, {std::string(image.components)});
        host.set_image_property(effect, clip, kOfxImagePropData, {static_cast<void*>(nullptr)});
        host.set_image_property(effect, clip, kOfxImagePropBounds, {image.bounds.x1, image.bounds.y1, image.bounds.x2, image.bounds.y2});
        host.set_image_property(effect, clip, kOfxImagePropRowBytes, {(image.bounds.x2 - image.bounds.x1) * 4 * static_cast<int>(sizeof(float))});
        host.set_image_property(effect, clip, kOfxImageEffectPropPixelDepth, {std::string(image.depth)});
        host.set_image_property(effect, clip, kOfxImageEffectPropComponents, {std::string(image.components)});
        host.set_image_property(effect, clip, kOfxImageEffectPropPreMultiplication, {std::string("OfxImageAlphaUnPremultiplied")});
        host.set_image_property(effect, clip, kOfxImagePropField, {std::string(kOfxImageFieldNone)});
        host.set_image_property(effect, clip, kOfxImagePropUniqueIdentifier, {std::string(clip) + "-image"});
        host.set_image_property(effect, clip, kOfxImagePropRegionOfDefinition, {image.bounds.x1, image.bounds.y1, image.bounds.x2, image.bounds.y2});
        host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, {1.0, 1.0});
        host.set_image_property(effect, clip, kOfxImagePropPixelAspectRatio, {1.0});
        host.set_clip_region(effect, clip, {static_cast<double>(image.bounds.x1), static_cast<double>(image.bounds.y1), static_cast<double>(image.bounds.x2), static_cast<double>(image.bounds.y2)});
    }
    Result run(OfxPlugin* plugin, InstanceHost& host, const RenderInput& input, std::string_view reference_id = {}) {
        if (!reference_id.empty()) {
            if (reference_cursor >= reference_rows.size())
                throw std::runtime_error("missing render reference row");
            const auto& row = reference_rows[reference_cursor++];
            if (!row.is_array() || row.size() != 2 || row[0] != reference_group + "/" + std::string(reference_id))
                throw std::runtime_error("render reference row order");
            const auto expected = decode(row[1]);
            return expected;
        }
        require(plugin != nullptr, "reference descriptor reacquired before Load");
        plugin->setHost(host.binding());
        require(action(plugin, kOfxActionLoad) == kOfxStatOK, "plugin load");
        require(action(plugin, kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK, "describe filter context");
        auto* const effect = host.create_effect();
        require(action(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "create render instance");
        host.set_value(effect, "FilmProfileKey", {std::string(input.film)});
        host.set_value(effect, "PrintProfileKey", {std::string(input.print)});
        host.set_effect_property(effect, kOfxImageEffectPropFrameRate, {24.0});
        host.set_clip_property(effect, "Source", kOfxImageEffectPropFrameRate, {24.0});
        host.set_clip_property(effect, "Output", kOfxImageEffectPropFrameRate, {24.0});
        set_image(host, effect, "Source", input.source);
        set_image(host, effect, "Output", ImageSpec{});
        host.set_image_available(effect, "Source", input.source_available);
        host.set_image_available(effect, "Output", input.output_available);
        host.set_image_get_status(effect, "Source", input.source_status);
        host.set_image_get_status(effect, "Output", input.output_status);
        host.set_image_release_status(effect, "Output", input.output_release_status);
        set_render_args(host, input.time, input.cuda, input.window, input.stream_present);
        if (!input.missing_action_property.empty())
            host.set_property(host.action_args(), std::string(input.missing_action_property), {});
        if (!input.fault.operation.empty())
            host.fail_on_nth(input.fault.occurrence, std::string(input.fault.operation), input.fault.status);
        if (!input.null_getter.property.empty())
            host.null_on_nth("get_string/" + std::string(input.null_getter.property), input.null_getter.occurrence);
        host.clear_calls();
        Result result;
        result.status = action(plugin, kOfxImageEffectActionRender, effect, host.action_args(), host.output_args());
        result.calls = host.calls();
        result.source_releases = host.image_release_count(effect, "Source");
        result.output_releases = host.image_release_count(effect, "Output");
        require(action(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "destroy render instance");
        require(action(plugin, kOfxActionUnload) == kOfxStatOK, "plugin unload");
        require(!host.callback_error(), "host callback did not throw");
        return result;
    }
    void same(const Result& reference, const Result& candidate, std::string_view id) {
        require(reference.status == candidate.status, std::string(id) + ": status differs from C++ mainEntry");
        require(ofx_test::reference::calls(reference.calls) == ofx_test::reference::calls(candidate.calls), std::string(id) + ": callback trace differs from C++ mainEntry");
        require(reference.source_releases == candidate.source_releases, std::string(id) + ": Source release count differs");
        require(reference.output_releases == candidate.output_releases, std::string(id) + ": Output release count differs");
        if (id == "keys") {
            auto altered = reference;
            altered.status = -100;
            require(altered.status != candidate.status, "candidate accepted altered render status");
            altered = reference;
            require(altered.calls.size() > 1, "render reference order boundary");
            std::swap(altered.calls[0], altered.calls[1]);
            require(ofx_test::reference::calls(altered.calls) != ofx_test::reference::calls(candidate.calls),
                    "candidate accepted reordered render calls");
        }
    }
    void require_getters(const Result& result, ExpectedProfileReads expected, std::string_view id) {
        std::vector<const InstanceCall*> reads;
        for (const auto& call : result.calls)
            if (call.operation == "param_get" && (call.target == "FilmProfileKey" || call.target == "PrintProfileKey"))
                reads.push_back(&call);
        require(reads.size() == expected.count, std::string(id) + ": ordered profile getter count");
        require(reads.front()->target == "FilmProfileKey", std::string(id) + ": Film getter first");
        if (expected.count == 2)
            require(reads[1]->target == "PrintProfileKey", std::string(id) + ": Print getter second");
        require(reads.front()->values == std::vector<InstanceValue>{std::string(expected.film)}, std::string(id) + ": Film raw getter value");
        if (expected.count == 2)
            require(reads[1]->values == std::vector<InstanceValue>{std::string(expected.print)}, std::string(id) + ": Print raw getter value");
        require(result.status == expected.status, std::string(id) + ": independent status");
        require(result.source_releases == 0 && result.output_releases == 0, std::string(id) + ": non-CUDA fetched an image");
    }
    void non_cuda(OfxPlugin* reference_plugin, OfxPlugin* candidate_plugin) {
        std::vector<std::string> ids{"keys", "film_empty", "print_empty", "both_empty"};
        const struct ArgCase {
            std::string_view id;
            std::string_view fault;
            std::string_view missing;
            OfxStatus status;
            OfxStatus expected;
        } arg_cases[]{
            {"args/time_status", "get_double/OfxPropTime", "", kOfxStatErrMemory, kOfxStatErrMemory},
            {"args/scale_status", "get_double/OfxImageEffectPropRenderScale", "", kOfxStatErrUnsupported, kOfxStatErrMissingHostFeature},
            {"args/window_status", "get_int/OfxImageEffectPropRenderWindow", "", kOfxStatErrBadIndex, kOfxStatErrBadIndex},
            {"args/field_status", "get_string/OfxImageEffectPropFieldToRender", "", kOfxStatErrValue, kOfxStatFailed},
            {"args/missing_time", "", kOfxPropTime, kOfxStatOK, kOfxStatErrMissingHostFeature},
            {"args/missing_field", "", kOfxImageEffectPropFieldToRender, kOfxStatOK, kOfxStatErrMissingHostFeature},
            {"args/stream_dimension_status", "dimension/OfxImageEffectPropCudaStream", "", kOfxStatErrMemory, kOfxStatErrFatal}};
        for (const auto& test : arg_cases)
            ids.emplace_back(test.id);
        for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status) {
            ids.push_back("film_status/status_" + std::to_string(status));
            ids.push_back("print_status/status_" + std::to_string(status));
        }
        const JuicerTests::Cohort rows("ofx.render.non_cuda", ids);
        struct Case {
            std::string id;
            std::string film;
            std::string print;
            std::string_view fault;
            OfxStatus injected;
            OfxStatus expected;
            std::size_t expected_reads;
        } cases[]{
            {"keys", "film-key", "print-key", "", kOfxStatOK, kOfxStatErrFatal, 2},
            {"film_empty", "", "print-key", "", kOfxStatOK, kOfxStatErrFatal, 2},
            {"print_empty", "film-key", "", "", kOfxStatOK, kOfxStatErrFatal, 2},
            {"both_empty", "", "", "", kOfxStatOK, kOfxStatErrFatal, 2}};
        const auto accepted = [](OfxStatus status) {
            return status == kOfxStatOK || status == kOfxStatReplyYes ||
                   status == kOfxStatReplyNo || status == kOfxStatReplyDefault;
        };
        const auto expected_status = [&](OfxStatus status) {
            return accepted(status) ? kOfxStatErrFatal : status;
        };
        std::vector<Case> all_cases(std::begin(cases), std::end(cases));
        for (int status = kOfxStatOK; status <= kOfxStatReplyDefault; ++status) {
            all_cases.push_back({"film_status/status_" + std::to_string(status), "film-key", "print-key", "param_get/FilmProfileKey", status, expected_status(status), accepted(status) ? 2u : 1u});
            all_cases.push_back({"print_status/status_" + std::to_string(status), "film-key", "print-key", "param_get/PrintProfileKey", status, expected_status(status), 2});
        }
        for (const auto& test : all_cases) {
            auto row = rows.case_event(test.id);
            InstanceHost reference_host, candidate_host;
            const auto expected = run(reference_plugin, reference_host, RenderInput{.film = test.film, .print = test.print, .time = 2.25, .fault = {.operation = test.fault, .status = test.injected}}, test.id);
            const auto actual = run(candidate_plugin, candidate_host, RenderInput{.film = test.film, .print = test.print, .time = 2.25, .fault = {.operation = test.fault, .status = test.injected}});
            same(expected, actual, test.id);
            require_getters(actual, ExpectedProfileReads{.film = test.film, .print = test.print, .count = test.expected_reads, .status = test.expected}, test.id);
            const auto reads = std::count_if(actual.calls.begin(), actual.calls.end(), [](const auto& call) {
                return call.operation == "param_get" && (call.target == "FilmProfileKey" || call.target == "PrintProfileKey");
            });
            require(reads == static_cast<std::ptrdiff_t>(test.expected_reads), test.id + ": first-failure getter prefix");
            for (const auto& call : actual.calls)
                require(call.operation != "clip_get_image" && call.operation != "clip_release_image", std::string(test.id) + ": cutoff preceded image I/O");
            row.passed();
        }
        for (const auto& test : arg_cases) {
            auto row = rows.case_event(test.id);
            InstanceHost reference_host, candidate_host;
            const auto expected = run(reference_plugin, reference_host, RenderInput{.film = "film-key", .print = "print-key", .time = 2.25, .missing_action_property = test.missing, .fault = {.operation = test.fault, .status = test.status}}, test.id);
            const auto actual = run(candidate_plugin, candidate_host, RenderInput{.film = "film-key", .print = "print-key", .time = 2.25, .missing_action_property = test.missing, .fault = {.operation = test.fault, .status = test.status}});
            same(expected, actual, test.id);
            require(actual.status == test.expected, std::string(test.id) + ": required/optional property status");
            require(std::none_of(actual.calls.begin(), actual.calls.end(), [](const auto& call) {
                        return call.operation == "clip_get_image" || call.operation == "clip_release_image";
                    }),
                    std::string(test.id) + ": action argument failure stopped before cutoff getter/image work");
            const auto profile_reads = std::count_if(actual.calls.begin(), actual.calls.end(), [](const auto& call) {
                return call.operation == "param_get" && (call.target == "FilmProfileKey" || call.target == "PrintProfileKey");
            });
            if (std::string_view(test.id) == "args/stream_dimension_status") {
                require(profile_reads == 2 && actual.status == kOfxStatErrFatal,
                        "optional stream dimension failure continues through required cutoff getters");
            } else {
                require(profile_reads == 0,
                        std::string(test.id) + ": required argument failure precedes profile getters");
            }
            if (std::string_view(test.id) != "args/stream_dimension_status")
                require(std::none_of(actual.calls.begin(), actual.calls.end(), [](const auto& call) {
                            return call.operation == "param_get";
                        }),
                        std::string(test.id) + ": required argument failure precedes profile getters");
            row.passed();
        }
    }
    void images(OfxPlugin* reference_plugin, OfxPlugin* candidate_plugin) {
        constexpr std::array metadata_faults{
            std::pair{"row_bytes", "get_int/OfxImagePropRowBytes"},
            std::pair{"pixel_aspect", "get_double/OfxImagePropPixelAspectRatio"},
            std::pair{"components", "get_string/OfxImageEffectPropComponents"},
            std::pair{"pixel_depth", "get_string/OfxImageEffectPropPixelDepth"},
            std::pair{"premultiplication", "get_string/OfxImageEffectPropPreMultiplication"},
            std::pair{"rod", "get_int/OfxImagePropRegionOfDefinition"},
            std::pair{"bounds", "get_int/OfxImagePropBounds"},
            std::pair{"field", "get_string/OfxImagePropField"},
            std::pair{"unique_id", "get_string/OfxImagePropUniqueIdentifier"},
            std::pair{"render_scale", "get_double/Source.image.0"},
            std::pair{"data", "get_pointer/OfxImagePropData"}};
        constexpr std::array null_string_properties{
            std::pair{"components", kOfxImageEffectPropComponents},
            std::pair{"depth", kOfxImageEffectPropPixelDepth},
            std::pair{"premultiplication", kOfxImageEffectPropPreMultiplication},
            std::pair{"field", kOfxImagePropField},
            std::pair{"unique_id", kOfxImagePropUniqueIdentifier}};
        std::vector<std::string> ids{"source_missing", "output_missing", "both_missing", "remaining_empty", "source_metadata", "source_status", "output_status", "release_status", "source_reply_yes", "source_reply_no", "source_reply_default", "output_reply_yes", "output_reply_no", "output_reply_default", "null/source_components", "null/output_components", "null/source_depth", "null/output_depth", "null/source_premultiplication", "null/output_premultiplication", "null/source_field", "null/output_field", "null/source_unique_id", "null/output_unique_id"};
        for (const auto& [id, fault] : metadata_faults) {
            (void)fault;
            ids.push_back(std::string("metadata/") + id);
        }
        const JuicerTests::Cohort rows("ofx.render.images", ids);
        struct Case {
            std::string id;
            bool source;
            bool output;
            OfxRectI window;
            std::string_view metadata_fault;
            OfxStatus source_status;
            OfxStatus output_status;
            OfxStatus release_status;
            std::string_view null_property = "";
            int null_occurrence = 1;
        };
        std::vector<Case> cases{
            {"source_missing", false, true, {0, 0, 4, 3}, "", kOfxStatOK, kOfxStatOK, kOfxStatOK},
            {"output_missing", true, false, {0, 0, 4, 3}, "", kOfxStatOK, kOfxStatOK, kOfxStatOK},
            {"both_missing", false, false, {0, 0, 4, 3}, "", kOfxStatOK, kOfxStatOK, kOfxStatOK},
            {"remaining_empty", true, true, {0, 0, 0, 0}, "", kOfxStatOK, kOfxStatOK, kOfxStatOK},
            {"source_metadata", true, true, {0, 0, 4, 3}, "get_int/OfxImagePropBounds", kOfxStatOK, kOfxStatOK, kOfxStatOK},
            {"source_status", true, true, {0, 0, 4, 3}, "", kOfxStatErrMemory, kOfxStatOK, kOfxStatOK},
            {"output_status", true, true, {0, 0, 4, 3}, "", kOfxStatOK, kOfxStatErrUnsupported, kOfxStatOK},
            {"release_status", true, true, {0, 0, 0, 0}, "", kOfxStatOK, kOfxStatOK, kOfxStatErrBadHandle},
            {"source_reply_yes", true, true, {0, 0, 4, 3}, "", kOfxStatReplyYes, kOfxStatOK, kOfxStatOK},
            {"source_reply_no", true, true, {0, 0, 4, 3}, "", kOfxStatReplyNo, kOfxStatOK, kOfxStatOK},
            {"source_reply_default", true, true, {0, 0, 4, 3}, "", kOfxStatReplyDefault, kOfxStatOK, kOfxStatOK},
            {"output_reply_yes", true, true, {0, 0, 4, 3}, "", kOfxStatOK, kOfxStatReplyYes, kOfxStatOK},
            {"output_reply_no", true, true, {0, 0, 4, 3}, "", kOfxStatOK, kOfxStatReplyNo, kOfxStatOK},
            {"output_reply_default", true, true, {0, 0, 4, 3}, "", kOfxStatOK, kOfxStatReplyDefault, kOfxStatOK}};
        for (const auto& [id, fault] : metadata_faults) {
            cases.push_back({std::string("metadata/") + id, true, true, {0, 0, 4, 3}, fault, kOfxStatOK, kOfxStatOK, kOfxStatOK});
        }
        for (const auto& [id, property] : null_string_properties) {
            const bool empty_output = id == std::string_view("components") || id == std::string_view("depth");
            cases.push_back({std::string("null/source_") + id, true, true, {0, 0, 4, 3}, "", kOfxStatOK, kOfxStatOK, kOfxStatOK, property, 1});
            cases.push_back({std::string("null/output_") + id, true, true, empty_output ? OfxRectI{0, 0, 0, 3} : OfxRectI{0, 0, 4, 3}, "", kOfxStatOK, kOfxStatOK, kOfxStatOK, property, 2});
        }
        for (const auto& test : cases) {
            auto row = rows.case_event(test.id);
            InstanceHost reference_host, candidate_host;
            const auto source_bounds = std::string_view(test.id) == "remaining_empty" || test.release_status != kOfxStatOK
                                           ? OfxRectI{0, 0, 0, 0}
                                           : OfxRectI{0, 0, 4, 3};
            const bool metadata_failure = !test.metadata_fault.empty();
            const auto metadata_status = metadata_failure ? kOfxStatErrMemory : kOfxStatOK;
            const auto metadata_occurrence = std::string_view(test.id) == "metadata/render_scale" ? 2 : 1;
            const RenderInput input{
                .film = "film-key",
                .print = "print-key",
                .time = 1.0,
                .cuda = true,
                .window = test.window,
                .source = {.components = kOfxImageComponentRGBA, .depth = kOfxBitDepthFloat, .bounds = source_bounds},
                .source_available = test.source,
                .output_available = test.output,
                .source_status = test.source_status,
                .output_status = test.output_status,
                .output_release_status = test.release_status,
                .fault = {.operation = test.metadata_fault, .status = metadata_status, .occurrence = metadata_occurrence},
                .null_getter = {.property = test.null_property, .occurrence = test.null_occurrence}};
            const auto expected = run(reference_plugin, reference_host, input, test.id);
            const auto actual = run(candidate_plugin, candidate_host, input);
            const bool premultiplication_null = test.null_property == kOfxImageEffectPropPreMultiplication;
            if (metadata_failure || (premultiplication_null && test.null_occurrence == 1)) {
                require(expected.status == actual.status && expected.calls.size() + 1 == actual.calls.size(), "candidate metadata cleanup keeps native error prefix");
                require(std::equal(expected.calls.begin(), expected.calls.end(), actual.calls.begin()), "candidate metadata failure preserves the C++ getter prefix");
                require(actual.calls.back().operation == "clip_release_image" && actual.calls.back().target == "Source.image.0", "candidate releases the image the C++ constructor leaked");
                require(expected.source_releases == 0 && expected.output_releases == 0, "C++ metadata-failure leak remains the reference observation");
                if (premultiplication_null)
                    require(actual.status == kOfxStatFailed, "null Source premultiplication fails at the property mapping");
            } else if (premultiplication_null && test.null_occurrence == 2) {
                require(expected.status == actual.status && expected.calls.size() + 1 == actual.calls.size(), "candidate Output metadata cleanup keeps the C++ error prefix");
                require(actual.status == kOfxStatFailed, "null Output premultiplication fails at the property mapping");
                require(expected.calls.size() > 0 && expected.calls.back().operation == "clip_release_image" && expected.calls.back().target == "Source.image.0",
                        "C++ releases its completed Source image while unwinding Output construction");
                require(std::equal(expected.calls.begin(), expected.calls.end() - 1, actual.calls.begin()), "candidate Output failure preserves the C++ getter prefix");
                require(actual.calls[expected.calls.size() - 1].operation == "clip_release_image" && actual.calls[expected.calls.size() - 1].target == "Output.image.0",
                        "candidate releases the image the C++ constructor leaked");
                require(actual.calls.back().operation == "clip_release_image" && actual.calls.back().target == "Source.image.0",
                        "candidate then releases the completed Source image");
                require(actual.source_releases == 1 && actual.output_releases == 1, "candidate releases both acquired images after Output metadata failure");
                require(expected.source_releases == 1 && expected.output_releases == 0, "C++ releases Source but leaks the Output constructor image");
            } else {
                same(expected, actual, test.id);
            }
            if (test.null_property == kOfxImageEffectPropComponents || test.null_property == kOfxImageEffectPropPixelDepth) {
                if (test.null_occurrence == 1)
                    require(actual.status == kOfxStatErrFatal, "null Source format string reaches format rejection");
                else
                    require(actual.status == kOfxStatOK, "null Output format string is accepted for an empty render window");
            }
            const auto source_fetch = std::find_if(actual.calls.begin(), actual.calls.end(), [](const auto& call) {
                return call.operation == "clip_get_image" && call.target == "Source";
            });
            const auto output_fetch = std::find_if(actual.calls.begin(), actual.calls.end(), [](const auto& call) {
                return call.operation == "clip_get_image" && call.target == "Output";
            });
            require(source_fetch != actual.calls.end(), std::string(test.id) + ": Source fetch attempted first");
            if (metadata_failure || (premultiplication_null && test.null_occurrence == 1) || test.source_status == kOfxStatErrMemory || test.source_status == kOfxStatErrUnsupported) {
                require(output_fetch == actual.calls.end(), std::string(test.id) + ": failed Source acquisition/metadata stopped before Output fetch");
            } else {
                require(output_fetch != actual.calls.end() && source_fetch < output_fetch, std::string(test.id) + ": Source then Output fetch");
            }
            if (!test.source && test.output)
                require(actual.output_releases == 1, "missing Source still fetched and released Output");
            if (!test.output && test.source)
                require(actual.source_releases == 1, "missing Output released Source");
            if (metadata_failure)
                require(actual.source_releases == 1 && actual.output_releases == 0, "Source metadata failure released its image before Output fetch");
            if (std::string_view(test.id) == "source_status")
                require(actual.status == kOfxStatErrMemory && actual.source_releases == 0 && actual.output_releases == 0, "Source acquisition error stops before Output fetch");
            if (std::string_view(test.id) == "output_status")
                require(actual.status == kOfxStatErrUnsupported && actual.source_releases == 1, "Output acquisition error releases acquired Source");
            if (std::string_view(test.id) == "release_status")
                require(actual.status == kOfxStatOK && actual.source_releases == 1 && actual.output_releases == 1, "release status does not replace empty-render success");
            row.passed();
        }
    }
    void formats(OfxPlugin* reference_plugin, OfxPlugin* candidate_plugin) {
        const JuicerTests::Cohort rows("ofx.render.formats", {"unknown_components", "unknown_depth"});
        const struct Case {
            const char* id;
            const char* components;
            const char* depth;
        } cases[]{
            {"unknown_components", "OfxImageComponentUnsupported", kOfxBitDepthFloat},
            {"unknown_depth", kOfxImageComponentRGBA, "OfxBitDepthUnsupported"}};
        for (const auto& test : cases) {
            auto row = rows.case_event(test.id);
            InstanceHost reference_host, candidate_host;
            const RenderInput input{
                .film = "film-key",
                .print = "print-key",
                .time = 1.0,
                .cuda = true,
                .window = std::string_view(test.id) == "unknown_components" ? OfxRectI{0, 0, 0, 3} : OfxRectI{0, 0, 4, 3},
                .source = {.components = test.components, .depth = test.depth}};
            const auto expected = run(reference_plugin, reference_host, input, test.id);
            const auto actual = run(candidate_plugin, candidate_host, input);
            same(expected, actual, test.id);
            require(actual.source_releases == 1 && actual.output_releases == 1, std::string(test.id) + ": rejected image formats released both leases");
            row.passed();
        }
    }
    void time(OfxPlugin* reference_plugin, OfxPlugin* candidate_plugin) {
        const JuicerTests::Cohort rows("ofx.render.time", {"non_cuda", "missing_source", "missing_output", "empty_window", "source_format", "capture_failure", "frame_rate_failure", "lower_outside", "upper_boundary", "upper_outside", "finite_max", "finite_min"});
        const double lower_outside = std::bit_cast<double>(UINT64_C(0xc3e0000000000001));
        const double upper_boundary = std::bit_cast<double>(UINT64_C(0x43e0000000000000));
        const double upper_outside = std::bit_cast<double>(UINT64_C(0x43e0000000000001));
        const double finite_max = std::bit_cast<double>(UINT64_C(0x7fefffffffffffff));
        const double finite_min = std::bit_cast<double>(UINT64_C(0xffefffffffffffff));
        struct Case {
            std::string_view id;
            bool safe_reference;
            RenderInput input;
        } cases[]{
            {.id = "non_cuda", .safe_reference = true, .input = {.time = lower_outside}},
            {.id = "missing_source", .safe_reference = true, .input = {.time = lower_outside, .cuda = true, .source_available = false}},
            {.id = "missing_output", .safe_reference = true, .input = {.time = upper_boundary, .cuda = true, .output_available = false}},
            {.id = "empty_window", .safe_reference = true, .input = {.time = finite_max, .cuda = true, .window = {0, 0, 0, 0}, .source = {.bounds = {0, 0, 0, 0}}}},
            {.id = "source_format", .safe_reference = true, .input = {.time = upper_boundary, .cuda = true, .source = {.components = "OfxImageComponentUnsupported"}}},
            {.id = "capture_failure", .safe_reference = true, .input = {.film = "missing_profile_key", .time = finite_max, .cuda = true}},
            {.id = "frame_rate_failure", .safe_reference = true, .input = {.film = kFallbackFilm, .print = kFallbackPrint, .time = lower_outside, .cuda = true, .fault = {.operation = "get_double/OfxImageEffectPropFrameRate", .status = kOfxStatErrMemory}}},
            {.id = "lower_outside", .safe_reference = false, .input = {.film = kFallbackFilm, .print = kFallbackPrint, .time = lower_outside, .cuda = true}},
            {.id = "upper_boundary", .safe_reference = false, .input = {.film = kFallbackFilm, .print = kFallbackPrint, .time = upper_boundary, .cuda = true}},
            {.id = "upper_outside", .safe_reference = false, .input = {.film = kFallbackFilm, .print = kFallbackPrint, .time = upper_outside, .cuda = true}},
            {.id = "finite_max", .safe_reference = false, .input = {.film = kFallbackFilm, .print = kFallbackPrint, .time = finite_max, .cuda = true}},
            {.id = "finite_min", .safe_reference = false, .input = {.film = kFallbackFilm, .print = kFallbackPrint, .time = finite_min, .cuda = true}}};
        for (const auto& test : cases) {
            auto row = rows.case_event(test.id);
            InstanceHost candidate_host;
            if (test.safe_reference) {
                InstanceHost reference_host;
                const auto expected = run(reference_plugin, reference_host, test.input, test.id);
                const auto actual = run(candidate_plugin, candidate_host, test.input);
                same(expected, actual, test.id);
                if (!test.input.fault.operation.empty()) {
                    require(actual.status == kOfxStatErrMemory, "frame-rate getter memory error preserved");
                    require(std::any_of(actual.calls.begin(), actual.calls.end(), [](const auto& call) {
                                return call.operation == "get_double" && call.property == kOfxImageEffectPropFrameRate;
                            }),
                            "frame-rate failure row reaches its intended getter");
                }
            } else {
                const auto actual = run(candidate_plugin, candidate_host, test.input);
                require(actual.status == kOfxStatErrFatal, std::string(test.id) + ": finite out-of-domain time must return Fatal");
                require(actual.source_releases == 1 && actual.output_releases == 1, std::string(test.id) + ": rejected time releases each acquired image once");
                require(std::any_of(actual.calls.begin(), actual.calls.end(), [](const auto& call) {
                            return call.operation == "get_double" && call.property == kOfxImageEffectPropFrameRate;
                        }),
                        std::string(test.id) + ": time guard follows effect frame-rate read");
            }
            row.passed();
        }
    }
} // namespace

int main(int argc, char** argv) try {
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto* id : kCases)
            JuicerTests::list_case(kCaller, id);
        return 0;
    }
    require(argc == 3 && std::string_view(argv[1]) == "--reference", "usage: render fixture --reference file");
    JuicerTests::CaseLog log;
    auto* candidate_plugin = OfxGetPlugin(0);
    OfxPlugin* reference_plugin = nullptr;
    require(candidate_plugin, "candidate descriptor");
    {
        reference_rows = ofx_test::reference::read(argv[2], "render");
        require(reference_rows.is_array() && reference_rows.size() == 85, "render reference membership");
        auto altered = reference_rows[0][1];
        altered[0] = -100;
        require(encode(decode(altered)) != reference_rows[0][1], "render recorded value negative control");
        require(reference_rows[0][0] != reference_rows[1][0], "render recorded order negative control");
    }
    {
        JuicerTests::CaseEvent event(kCaller, "non_cuda");
        reference_group = "non_cuda";
        non_cuda(reference_plugin, candidate_plugin);
        event.passed();
    }
    {
        JuicerTests::CaseEvent event(kCaller, "images");
        reference_group = "images";
        images(reference_plugin, candidate_plugin);
        event.passed();
    }
    {
        JuicerTests::CaseEvent event(kCaller, "formats");
        reference_group = "formats";
        formats(reference_plugin, candidate_plugin);
        event.passed();
    }
    {
        JuicerTests::CaseEvent event(kCaller, "time");
        reference_group = "time";
        time(reference_plugin, candidate_plugin);
        event.passed();
    }
    require(reference_rows.size() == 85 && reference_cursor == reference_rows.size(), "render reference complete consumption");
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "render fixture: %s\n", error.what());
    return 1;
}
